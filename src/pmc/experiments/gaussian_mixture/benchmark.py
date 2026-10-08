##############################
# Import necessary libraries #
##############################

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import warnings
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from pmc.algorithms import MLPPMC, LinearPMC, consensus_discrepancy
from pmc.pmc_gp import GaussianProcessPMC, PMCConfig

from .exact_oracle import build_exact_product

########################
# Scientific constants #
########################

_GBF_IMPLEMENTATION = "fitted_univariate_mixture_dcfusion_adapter"
_GBF_REASON = None
_GBF_VARIANCE_FLOOR = 1e-8
_GBF_WEIGHT_FLOOR = 1e-12
_SCHEMA_VERSION = 4
_METHODS = ("Oracle", "PMC-OLS", "PMC-MLP", "PMC-GP", "GBF")
_SEED_STREAM_OFFSETS = {
    "raw_components": 0,
    "pmc_mlp": 100_000_000,
    "gbf": 200_000_000,
    "pmc_gp": 300_000_000,
    "oracle": 400_000_000,
}
_COMMON_ROW_FIELDS = {
    "algorithm_seed",
    "batch_index",
    "execution_key",
    "expanded_component_count",
    "global_repetition",
    "model_seed",
    "n",
    "node_count",
    "provenance_signature",
    "raw_seed",
    "reference_log_normalizer",
    "reference_normalizer",
    "repetition",
    "scenario",
}

############################
# Validation and sampling  #
############################

# Validate one positive integer.
def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

# Validate one nonnegative integer.
def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")

# Validate one finite positive number.
def _positive_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive.")

# Draw seeded samples independently from every Gaussian-mixture node.
def sample_raw_components(nodes, sample_count, seed):
    build_exact_product(nodes)
    _positive_integer(sample_count, "sample_count")
    _nonnegative_integer(seed, "seed")
    generator = np.random.default_rng(seed)
    samples = np.empty((sample_count, len(nodes), 1), dtype=float)
    for node_index, node in enumerate(nodes):
        weights = np.asarray(node["weights"], dtype=float)
        means = np.asarray(node["means"], dtype=float)
        standard_deviations = np.asarray(
            node["standard_deviations"],
            dtype=float,
        )
        indices = generator.choice(len(weights), size=sample_count, p=weights)
        samples[:, node_index, 0] = generator.normal(
            means[indices],
            standard_deviations[indices],
        )
    return samples

# Prepare PMC consensus and discrepancy from raw samples only.
def prepare_pmc_inputs(raw_samples):
    samples = np.asarray(raw_samples, dtype=float)
    if (
        samples.ndim != 3
        or samples.shape[0] < 1
        or samples.shape[1] < 2
        or samples.shape[2] != 1
        or np.any(~np.isfinite(samples))
    ):
        raise ValueError(
            "Raw PMC samples must be a finite nonempty n by K by 1 tensor."
        )
    node_count = samples.shape[1]
    weights = [np.eye(1) / node_count for _ in range(node_count)]
    return consensus_discrepancy(
        samples,
        w=weights,
        sampling_method="iid",
        discrepancy_method="consecutive",
    )

# Resolve fixed or adaptive PMC training steps.
def resolve_pmc_training_steps(training_steps, sample_count):
    _positive_integer(sample_count, "sample_count")
    if training_steps == "adaptive":
        return int(
            np.clip(
                round(200 * np.sqrt(sample_count / 100)),
                200,
                8_000,
            )
        )
    _positive_integer(training_steps, "n_training_steps")
    return training_steps

# Configure deterministic CPU execution.
def configure_torch(seed):
    _nonnegative_integer(seed, "seed")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

# Return software and hardware provenance.
def runtime_provenance():
    return {
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "runtime_device": "cpu",
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }

# Return target transforms and exact truths.
def build_target_specs(oracle):
    return (
        ("identity", None, oracle.mean()),
        ("square", np.square, oracle.second_moment()),
    )

# Resolve one target-specific learning rate.
def _learning_rate(pmc, target):
    rates = pmc["base_learning_rate"]
    return rates[target] if isinstance(rates, dict) else rates


def resolve_pmc_device(device):
    """Resolve the estimator device while honoring CUDA availability."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("pmc.device is cuda but CUDA is unavailable.")
    return device

# Report the configured device without assuming an available CUDA runtime.
def _pmc_runtime_device(pmc, method):
    if method == "PMC-GP":
        return pmc["gp"]["device"]
    if method == "PMC-OLS":
        return "cpu"
    if pmc["device"] == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pmc["device"]

# Build the public PMC-GP configuration from validated controls.
def _pmc_gp_config(pmc):
    controls = dict(pmc["gp"])
    controls.pop("confidence_level")
    return PMCConfig(**controls)

# Fit and predict one PMC method target using raw-derived inputs.
def fit_pmc_method(pmc, method, target, transform, inputs, model_seed):
    consensus, discrepancy = inputs
    if method == "PMC-OLS":
        estimator = LinearPMC(h=transform, degree=1)
    elif method == "PMC-MLP":
        configure_torch(model_seed)
        estimator = MLPPMC(
            h=transform,
            hidden_features=tuple(pmc["hidden_features"]),
            n_training_steps=resolve_pmc_training_steps(
                pmc["n_training_steps"],
                len(consensus),
            ),
            optimizer="Adam",
            learning_rate=_learning_rate(pmc, target),
            batch_size=min(pmc["batch_size"], len(consensus)),
            device=resolve_pmc_device(pmc["device"]),
            use_cosine_schedule=pmc["learning_rate_schedule"] == "cosine",
            loss_weighting="uniform",
            input_degree=1,
            activation="relu",
            discrepancy_centering="mean",
            ensemble_size=pmc["ensemble_size"],
            regression_loss="mse",
        )
    elif method == "PMC-GP":
        consensus_values = np.asarray(consensus, dtype=float)
        response = (
            consensus_values
            if transform is None
            else transform(consensus_values)
        )
        response = np.asarray(response, dtype=float).reshape(-1)
        discrepancy_values = np.asarray(discrepancy, dtype=float)
        if discrepancy_values.shape[1] == 1:
            discrepancy_values = discrepancy_values[:, 0]
            query = 0.0
        else:
            query = np.zeros(discrepancy_values.shape[1])
        estimator = GaussianProcessPMC(_pmc_gp_config(pmc))
        start = perf_counter()
        estimator.fit(discrepancy_values, response, seed=model_seed)
        result = estimator.predict(
            query,
            confidence_level=pmc["gp"]["confidence_level"],
        )
        estimate = float(result.estimate)
        elapsed = perf_counter() - start
        if not math.isfinite(estimate):
            raise ValueError("PMC prediction must be finite.")
        return estimate, elapsed
    else:
        raise ValueError(f"Unsupported PMC method {method}.")
    start = perf_counter()
    estimator.fit(consensus, discrepancy)
    estimate = float(
        np.asarray(estimator.predict(np.zeros(discrepancy.shape[1]))).squeeze()
    )
    elapsed = perf_counter() - start
    if not math.isfinite(estimate):
        raise ValueError("PMC prediction must be finite.")
    return estimate, elapsed

########################
# Fitted GBF execution #
########################

# Return component counts without exposing generating parameters.
def _node_component_counts(nodes):
    return [len(node["weights"]) for node in nodes]

# Validate raw observations for mixture fitting.
def _validate_gbf_raw_samples(raw_samples):
    samples = np.asarray(raw_samples, dtype=float)
    if (
        samples.ndim != 3
        or samples.shape[0] < 2
        or samples.shape[1] < 2
        or samples.shape[2] != 1
        or np.any(~np.isfinite(samples))
    ):
        raise ValueError(
            "Raw GBF samples must be a finite n by K by 1 tensor with "
            "at least two draws and nodes."
        )
    return samples

# Fit one node's univariate Gaussian mixture by regularized EM.
def _fit_em_mixture(values, component_count, seed, n_init, max_iter=100, tol=1e-3):
    model = GaussianMixture(
        n_components=component_count,
        covariance_type="full",
        random_state=seed,
        reg_covar=_GBF_VARIANCE_FLOOR,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )
    model.fit(values.reshape(-1, 1))
    return model

# Convert one fitted EM model to sorted, floored node parameters.
def _fitted_node_parameters(model):
    weights = np.asarray(model.weights_, dtype=float)
    means = np.asarray(model.means_, dtype=float).reshape(-1)
    variances = np.asarray(model.covariances_, dtype=float).reshape(-1)
    order = np.argsort(means)
    weights = weights[order]
    means = means[order]
    standard_deviations = np.sqrt(
        np.maximum(variances[order], _GBF_VARIANCE_FLOOR)
    )
    weights = np.maximum(weights, _GBF_WEIGHT_FLOOR)
    weights = weights / np.sum(weights)
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(means))
        or np.any(~np.isfinite(standard_deviations))
        or np.any(standard_deviations <= 0.0)
    ):
        raise ValueError(
            "Fitted mixture parameters must be finite and positive."
        )
    return {
        "weights": weights.tolist(),
        "means": means.tolist(),
        "standard_deviations": standard_deviations.tolist(),
    }

# Fit each node's mixture parameters from its raw observations.
def fit_gbf_node_mixtures(raw_samples, component_counts, seed):
    samples = _validate_gbf_raw_samples(raw_samples)
    _nonnegative_integer(seed, "GBF seed")
    if len(component_counts) != samples.shape[1]:
        raise ValueError("GBF component counts must match the node count.")
    nodes = []
    for node_index, component_count in enumerate(component_counts):
        _positive_integer(component_count, "GBF component count")
        if component_count > samples.shape[0]:
            raise ValueError("GBF component count cannot exceed raw sample count.")
        model = _fit_em_mixture(
            samples[:, node_index, 0],
            component_count,
            seed + node_index,
            n_init=3,
        )
        if not model.converged_:
            raise ValueError("Fitted GBF mixture did not converge.")
        nodes.append(_fitted_node_parameters(model))
    return nodes

# Persist the same raw node observations used by PMC.
def _write_gbf_raw_samples(raw_samples, sample_path):
    samples = _validate_gbf_raw_samples(raw_samples)
    header = ",".join(
        f"node_{node_index}" for node_index in range(1, samples.shape[1] + 1)
    )
    np.savetxt(
        sample_path,
        samples[:, :, 0],
        delimiter=",",
        header=header,
        comments="",
        fmt="%.17g",
    )

# Load only equal-weight terminal samples from the R adapter.
def _load_gbf_equal_weight_samples(artifact_dir, diagnostics):
    if diagnostics.get("scoring_sample_role") != "equal_weight_terminal_resample":
        raise ValueError("GBF scoring requires equal-weight terminal samples.")
    samples = np.loadtxt(
        artifact_dir / "gbf_samples.csv",
        delimiter=",",
        skiprows=1,
    )
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    if samples.ndim != 1 or np.any(~np.isfinite(samples)):
        raise ValueError("GBF equal-weight samples must be a finite vector.")
    n_inference = diagnostics.get("n_inference")
    if len(samples) != n_inference:
        raise ValueError("GBF equal-weight sample count must match diagnostics.")
    return samples

# Execute the modified node-specific GBF adapter on fitted parameters.
def run_fitted_gbf(raw_samples, component_counts, artifact_dir, seed, gbf):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fitting_start = perf_counter()
    fitted_nodes = fit_gbf_node_mixtures(raw_samples, component_counts, seed)
    fitting_seconds = perf_counter() - fitting_start

    preparation_start = perf_counter()
    node_config = {"num_nodes": len(fitted_nodes), "nodes": fitted_nodes}
    gbf_config = {
        **gbf,
        "n_inference": int(np.asarray(raw_samples).shape[0]),
        "seed": seed,
    }
    node_config_path = artifact_dir / "gbf_fitted_node_config.json"
    gbf_config_path = artifact_dir / "gbf_config.json"
    sample_path = artifact_dir / "gbf_raw_samples.csv"
    node_config_path.write_text(
        json.dumps(node_config, indent=2) + "\n",
        encoding="utf-8",
    )
    gbf_config_path.write_text(
        json.dumps(gbf_config, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_gbf_raw_samples(raw_samples, sample_path)
    preparation_seconds = perf_counter() - preparation_start

    command = [
        "Rscript",
        str(Path(__file__).resolve().with_name("run_gbf.R")),
        str(node_config_path),
        str(gbf_config_path),
        str(sample_path),
        str(artifact_dir),
    ]
    fusion_start = perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        error.gbf_fitting_seconds = fitting_seconds
        error.gbf_preparation_seconds = preparation_seconds
        error.gbf_fusion_seconds = perf_counter() - fusion_start
        raise
    subprocess_seconds = perf_counter() - fusion_start
    diagnostics = json.loads(
        (artifact_dir / "gbf_diagnostics.json").read_text(encoding="utf-8")
    )
    if (
        diagnostics.get("gbf_implementation") != _GBF_IMPLEMENTATION
        or diagnostics.get("parameter_source") != "fitted_from_raw_node_samples"
        or diagnostics.get("n_inference") != len(raw_samples)
        or diagnostics.get("seed") != seed
        or diagnostics.get("scoring_sample_role")
        != "equal_weight_terminal_resample"
    ):
        raise ValueError("GBF adapter diagnostics violate the fitted-run contract.")
    fusion_seconds = diagnostics.get("runtime_seconds", subprocess_seconds)
    _validate_runtime(fusion_seconds, "gbf_fusion_runtime_seconds")
    equal_weight_samples = _load_gbf_equal_weight_samples(
        artifact_dir, diagnostics
    )
    return {
        "diagnostics": diagnostics,
        "fitted_nodes": fitted_nodes,
        "fitting_seconds": fitting_seconds,
        "fusion_seconds": float(fusion_seconds),
        "preparation_seconds": preparation_seconds,
        "samples": equal_weight_samples,
    }

############################
# Configuration validation #
############################

# Validate PMC-GP controls through the public configuration.
def _validate_pmc_gp(gp):
    required = set(PMCConfig.__dataclass_fields__) | {"confidence_level"}
    if not isinstance(gp, dict) or set(gp) != required:
        raise ValueError(f"pmc.gp must contain exactly {sorted(required)}.")
    confidence_level = gp["confidence_level"]
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(float(confidence_level))
        or not 0 < confidence_level < 1
    ):
        raise ValueError(
            "pmc.gp.confidence_level must lie strictly between zero and one."
        )
    try:
        _pmc_gp_config({"gp": gp})
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid pmc.gp controls: {error}") from error

# Validate PMC controls.
def _validate_pmc(pmc):
    required = {
        "base_learning_rate",
        "batch_size",
        "device",
        "ensemble_size",
        "hidden_features",
        "learning_rate_schedule",
        "gp",
        "n_training_steps",
    }
    if not isinstance(pmc, dict) or set(pmc) != required:
        raise ValueError(f"pmc must contain exactly {sorted(required)}.")
    for field in ("batch_size", "ensemble_size"):
        _positive_integer(pmc[field], f"pmc.{field}")
    if pmc["n_training_steps"] != "adaptive":
        _positive_integer(pmc["n_training_steps"], "pmc.n_training_steps")
    if pmc["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("pmc.device must be auto, cpu, or cuda.")
    hidden = pmc["hidden_features"]
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("pmc.hidden_features must be a nonempty list.")
    for width in hidden:
        _positive_integer(width, "pmc.hidden_features entry")
    rates = pmc["base_learning_rate"]
    if not isinstance(rates, dict) or set(rates) != {"identity", "square"}:
        raise ValueError("Target-specific learning rates are incomplete.")
    for rate in rates.values():
        _positive_number(rate, "pmc.base_learning_rate")
    if pmc["learning_rate_schedule"] not in {"constant", "cosine"}:
        raise ValueError("PMC learning-rate schedule is unsupported.")
    _validate_pmc_gp(pmc["gp"])

# Validate controls for the fitted node-specific GBF adapter.
def _validate_gbf(gbf):
    required = {
        "cess_0_threshold",
        "ess_threshold",
        "n_cores",
        "time_mesh",
    }
    if not isinstance(gbf, dict) or set(gbf) != required:
        raise ValueError(f"gbf must contain exactly {sorted(required)}.")
    _positive_integer(gbf["n_cores"], "gbf.n_cores")
    if gbf["time_mesh"] != "optimal":
        try:
            time_mesh = np.asarray(gbf["time_mesh"], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "gbf.time_mesh must be optimal or finite, increasing, "
                "and start at zero."
            ) from error
        if (
            time_mesh.ndim != 1
            or len(time_mesh) < 2
            or not np.all(np.isfinite(time_mesh))
            or time_mesh[0] != 0
            or np.any(np.diff(time_mesh) <= 0)
        ):
            raise ValueError(
                "gbf.time_mesh must be optimal or finite, increasing, "
                "and start at zero."
            )
    for field in ("cess_0_threshold", "ess_threshold"):
        value = gbf[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"gbf.{field} must be between zero and one.")

# Validate one scenario and its exact product.
def _validate_scenario(scenario):
    if not isinstance(scenario, dict) or set(scenario) != {"name", "nodes"}:
        raise ValueError("Each scenario must contain name and nodes.")
    if (
        not isinstance(scenario["name"], str)
        or re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", scenario["name"])
        is None
    ):
        raise ValueError("Each scenario name must use lowercase snake case.")
    if not isinstance(scenario["nodes"], list) or len(scenario["nodes"]) < 2:
        raise ValueError("Each benchmark scenario must contain at least two nodes.")
    build_exact_product(scenario["nodes"])

# Validate a comparison configuration.
def validate_config(config):
    required = {
        "base_seed",
        "component_sample_sizes",
        "pmc",
        "gbf",
        "methods",
        "n_inference",
        "repetition_start",
        "repetitions",
        "repetitions_per_batch",
        "scenarios",
        "targets",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Configuration must contain exactly {sorted(required)}.")
    _positive_integer(config["base_seed"], "base_seed")
    _nonnegative_integer(config["repetition_start"], "repetition_start")
    _positive_integer(config["repetitions"], "repetitions")
    _positive_integer(
        config["repetitions_per_batch"],
        "repetitions_per_batch",
    )
    if config["repetitions"] % config["repetitions_per_batch"] != 0:
        raise ValueError("repetitions_per_batch must divide repetitions.")
    if config["methods"] != list(_METHODS):
        raise ValueError(f"methods must equal {list(_METHODS)}.")
    if config["n_inference"] is not None:
        raise ValueError(
            "n_inference must be null for direct point-prediction methods."
        )
    repetition_end = config["repetition_start"] + config["repetitions"] - 1
    if repetition_end >= 1_000:
        raise ValueError("The global repetition end must be below 1000.")
    sizes = config["component_sample_sizes"]
    if not isinstance(sizes, list) or not sizes or len(sizes) > 100:
        raise ValueError(
            "component_sample_sizes must be a nonempty list of at most 100 "
            "values."
        )
    for sample_size in sizes:
        _positive_integer(sample_size, "component_sample_sizes entry")
    if sizes != sorted(set(sizes)):
        raise ValueError("component_sample_sizes must increase uniquely.")
    if config["targets"] != ["identity", "square"]:
        raise ValueError("targets must contain identity followed by square.")
    scenarios = config["scenarios"]
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or len(scenarios) > 100
    ):
        raise ValueError("scenarios must be a nonempty list of at most 100 values.")
    for scenario in scenarios:
        _validate_scenario(scenario)
    names = [scenario["name"] for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Scenario names must be unique.")
    _validate_pmc(config["pmc"])
    _validate_gbf(config["gbf"])
    maximum_seed = (
        config["base_seed"]
        + max(_SEED_STREAM_OFFSETS.values())
        + (len(scenarios) - 1) * 1_000_000
        + (len(sizes) - 1) * 10_000
        + repetition_end * 10
        + 1
    )
    if maximum_seed > np.iinfo(np.uint32).max:
        raise ValueError("Derived comparison seeds exceed the uint32 range.")
    return config

# Load and validate a comparison configuration.
def load_config(config_path):
    with Path(config_path).open(encoding="utf-8") as config_file:
        return validate_config(json.load(config_file))

###############################
# Deterministic seed schedule #
###############################

# Derive one nonoverlapping comparison seed.
def comparison_seed(
    base_seed,
    stream,
    scenario_index,
    sample_size_index,
    repetition,
    target_index=0,
):
    return int(
        base_seed
        + _SEED_STREAM_OFFSETS[stream]
        + scenario_index * 1_000_000
        + sample_size_index * 10_000
        + repetition * 10
        + target_index
    )

# Resolve repetitions for one selected batch.
def _selected_repetitions(config, batch_index):
    start = config["repetition_start"]
    repetitions = config["repetitions"]
    if batch_index is None:
        return range(start, start + repetitions)
    _nonnegative_integer(batch_index, "batch_index")
    per_batch = config["repetitions_per_batch"]
    batch_count = repetitions // per_batch
    if batch_index >= batch_count:
        raise ValueError(f"batch_index must be below {batch_count}.")
    batch_start = start + batch_index * per_batch
    return range(batch_start, batch_start + per_batch)

# Select configured methods in canonical order.
def _selected_methods(config, methods):
    if methods is None:
        return tuple(config["methods"])
    if isinstance(methods, str):
        methods = methods.split(",")
    if not isinstance(methods, (list, tuple)) or not methods:
        raise ValueError("methods must select at least one configured method.")
    if len(methods) != len(set(methods)):
        raise ValueError("methods must not contain duplicates.")
    unsupported = set(methods) - set(config["methods"])
    if unsupported:
        raise ValueError(f"Unsupported selected methods: {sorted(unsupported)}.")
    return tuple(method for method in config["methods"] if method in methods)

# Select configured work without renumbering seed-design indices.
def _selected_work(
    config, batch_index, sample_size, repetition_start, repetition_count
):
    base_repetitions = _selected_repetitions(config, batch_index)
    if sample_size is None:
        sizes = config["component_sample_sizes"]
    else:
        _positive_integer(sample_size, "sample_size")
        if sample_size not in config["component_sample_sizes"]:
            raise ValueError("sample_size must occur in the full configuration.")
        sizes = [sample_size]
    if repetition_start is None and repetition_count is None:
        repetitions = base_repetitions
    else:
        if repetition_start is None or repetition_count is None:
            raise ValueError(
                "repetition_start and repetition_count must be supplied together."
            )
        _nonnegative_integer(repetition_start, "repetition_start")
        _positive_integer(repetition_count, "repetition_count")
        if (
            repetition_start < base_repetitions.start
            or repetition_start + repetition_count > base_repetitions.stop
        ):
            raise ValueError(
                "Selected repetitions must lie within the configured batch."
            )
        repetitions = range(
            repetition_start, repetition_start + repetition_count
        )
    selection = {
        "sample_size": sample_size,
        "repetition_start": repetition_start,
        "repetition_count": repetition_count,
    }
    return sizes, repetitions, selection

########################
# Comparison execution #
########################

# Return common rectangular fields for one row.
def _base_row(common, method, target, status, fit_group):
    return {
        **common,
        "absolute_error": None,
        "base_learning_rate": None,
        "batch_size": None,
        "ensemble_size": None,
        "estimate": None,
        "failure_exception_type": None,
        "failure_message": None,
        "failure_stage": None,
        "fit_prediction_runtime_group": fit_group,
        "fit_prediction_runtime_seconds": None,
        "gbf_artifact_dir": None,
        "gbf_fitting_runtime_group": None,
        "gbf_fitting_runtime_seconds": None,
        "gbf_fusion_runtime_group": None,
        "gbf_fusion_runtime_seconds": None,
        "gbf_implementation": _GBF_IMPLEMENTATION,
        "gbf_preparation_runtime_group": None,
        "gbf_preparation_runtime_seconds": None,
        "gbf_runtime_group": None,
        "gbf_not_applicable_reason": None,
        "inference_mode": "not_applicable",
        "learning_rate_schedule": None,
        "method": method,
        "n_inference": None,
        "n_training_steps": None,
        "oracle_em_converged": None,
        "preparation_runtime_group": None,
        "preparation_runtime_seconds": None,
        "runtime_clock": "perf_counter_wall",
        "runtime_device": "cpu",
        "runtime_scope": None,
        "runtime_shared_across_rows": False,
        "signed_error": None,
        "squared_error": None,
        "standalone_runtime_seconds": 0.0,
        "status": status,
        "target": target,
        "truth": None,
    }

# Return the exact persisted row schema.
def _row_fields():
    return set(_base_row({}, "Oracle", "identity", "success", None)) | (
        _COMMON_ROW_FIELDS
    )

# Return resolved controls for one PMC row.
def _pmc_controls(pmc, method, target, sample_count, model_seed):
    if method == "PMC-OLS":
        return {
            "base_learning_rate": None,
            "batch_size": None,
            "ensemble_size": None,
            "learning_rate_schedule": None,
            "model_seed": None,
            "n_training_steps": None,
        }
    if method == "PMC-GP":
        gp = pmc["gp"]
        return {
            "base_learning_rate": gp["learning_rate"],
            "batch_size": sample_count if gp["full_batch"] else min(
                gp["batch_size"], sample_count
            ),
            "ensemble_size": None,
            "learning_rate_schedule": None,
            "model_seed": model_seed,
            "n_training_steps": gp["maximum_updates"],
        }
    return {
        "base_learning_rate": _learning_rate(pmc, target),
        "batch_size": min(pmc["batch_size"], sample_count),
        "ensemble_size": pmc["ensemble_size"],
        "learning_rate_schedule": pmc["learning_rate_schedule"],
        "model_seed": model_seed,
        "n_training_steps": resolve_pmc_training_steps(
            pmc["n_training_steps"],
            sample_count,
        ),
    }

# Fit every node under its true family and combine the fits exactly.
def fit_oracle_product(raw_samples, component_counts, seed):
    samples = np.asarray(raw_samples, dtype=float)
    fitted_nodes = []
    converged = True
    for node_index, component_count in enumerate(component_counts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = _fit_em_mixture(
                samples[:, node_index, 0],
                component_count,
                seed + node_index,
                n_init=10,
                max_iter=1000,
                tol=1e-8,
            )
        converged = converged and bool(model.converged_)
        fitted_nodes.append(_fitted_node_parameters(model))
    return build_exact_product(fitted_nodes), converged

# Build fitted Oracle rows sharing one EM fit across both targets.
def _oracle_rows(raw_samples, component_counts, seed, targets, common):
    start = perf_counter()
    product, converged = fit_oracle_product(raw_samples, component_counts, seed)
    estimates = {
        "identity": product.mean(),
        "square": product.second_moment(),
    }
    elapsed = perf_counter() - start
    rows = []
    for target, _, truth in targets:
        row = _base_row(
            common,
            "Oracle",
            target,
            "success",
            f"{common['execution_key']}_oracle",
        )
        error = estimates[target] - truth
        row.update(
            {
                "absolute_error": abs(error),
                "algorithm_seed": seed,
                "estimate": estimates[target],
                "fit_prediction_runtime_seconds": elapsed,
                "oracle_em_converged": converged,
                "runtime_scope": "fitted_em_mixture_exact_product",
                "runtime_shared_across_rows": True,
                "signed_error": error,
                "squared_error": error**2,
                "standalone_runtime_seconds": elapsed,
                "truth": truth,
            }
        )
        rows.append(row)
    return rows

# Build GBF success rows from the equal-weight terminal sample.
def _gbf_success_rows(targets, common, result, runtime_group, artifact_dir, seed):
    estimates = {
        "identity": float(np.mean(result["samples"])),
        "square": float(np.mean(np.square(result["samples"]))),
    }
    rows = []
    for target, _, truth in targets:
        row = _base_row(common, "GBF", target, "success", None)
        error = estimates[target] - truth
        row.update(
            {
                "absolute_error": abs(error),
                "algorithm_seed": seed,
                "estimate": estimates[target],
                "gbf_artifact_dir": str(artifact_dir),
                "gbf_fitting_runtime_group": f"{runtime_group}_fitting",
                "gbf_fitting_runtime_seconds": result["fitting_seconds"],
                "gbf_fusion_runtime_group": f"{runtime_group}_fusion",
                "gbf_fusion_runtime_seconds": result["fusion_seconds"],
                "gbf_preparation_runtime_group": f"{runtime_group}_preparation",
                "gbf_preparation_runtime_seconds": result["preparation_seconds"],
                "gbf_runtime_group": runtime_group,
                "inference_mode": "terminal_equal_weight_resample",
                "n_inference": result["diagnostics"]["n_inference"],
                "runtime_scope": "mixture_fitting_preparation_plus_gbf_fusion_wall",
                "runtime_shared_across_rows": True,
                "signed_error": error,
                "squared_error": error**2,
                "standalone_runtime_seconds": (
                    result["preparation_seconds"]
                    + result["fitting_seconds"]
                    + result["fusion_seconds"]
                ),
                "truth": truth,
            }
        )
        rows.append(row)
    return rows

# Build GBF failure rows without scientific fallback estimates.
def _gbf_failure_rows(
    targets,
    common,
    runtime_group,
    artifact_dir,
    seed,
    stage,
    error,
    preparation_seconds=0.0,
    fitting_seconds=0.0,
    fusion_seconds=0.0,
):
    rows = []
    for target, _, truth in targets:
        row = _base_row(common, "GBF", target, "failure", None)
        row.update(
            {
                "algorithm_seed": seed,
                "failure_exception_type": type(error).__name__,
                "failure_message": str(error),
                "failure_stage": stage,
                "gbf_artifact_dir": str(artifact_dir),
                "gbf_fitting_runtime_group": f"{runtime_group}_fitting",
                "gbf_fitting_runtime_seconds": fitting_seconds,
                "gbf_fusion_runtime_group": f"{runtime_group}_fusion",
                "gbf_fusion_runtime_seconds": fusion_seconds,
                "gbf_preparation_runtime_group": f"{runtime_group}_preparation",
                "gbf_preparation_runtime_seconds": preparation_seconds,
                "gbf_runtime_group": runtime_group,
                "inference_mode": "terminal_equal_weight_resample",
                "n_inference": common["n"],
                "runtime_scope": "gbf_time_until_failure",
                "runtime_shared_across_rows": True,
                "standalone_runtime_seconds": (
                    preparation_seconds + fitting_seconds + fusion_seconds
                ),
                "truth": truth,
            }
        )
        rows.append(row)
    return rows

# Attempt fitted GBF once and share its result across both targets.
def _gbf_rows(targets, common, scenario, raw_samples, artifact_dir, seed, gbf):
    runtime_group = f"{common['execution_key']}_gbf"
    component_counts = _node_component_counts(scenario["nodes"])
    try:
        result = run_fitted_gbf(
            raw_samples, component_counts, artifact_dir, seed, gbf
        )
    except subprocess.CalledProcessError as error:
        return _gbf_failure_rows(
            targets,
            common,
            runtime_group,
            artifact_dir,
            seed,
            "gbf_fusion",
            error,
            preparation_seconds=getattr(error, "gbf_preparation_seconds", 0.0),
            fitting_seconds=getattr(error, "gbf_fitting_seconds", 0.0),
            fusion_seconds=getattr(error, "gbf_fusion_seconds", 0.0),
        )
    except Exception as error:
        return _gbf_failure_rows(
            targets,
            common,
            runtime_group,
            artifact_dir,
            seed,
            "gbf_mixture_fitting",
            error,
        )
    return _gbf_success_rows(
        targets, common, result, runtime_group, artifact_dir, seed
    )

# Build one PMC success row.
def _pmc_success_row(
    pmc,
    common,
    method,
    target,
    truth,
    estimate,
    preparation_seconds,
    fit_seconds,
    fit_group,
    model_seed,
):
    row = _base_row(common, method, target, "success", fit_group)
    error = estimate - truth
    row.update(
        {
            "absolute_error": abs(error),
            "estimate": estimate,
            "fit_prediction_runtime_seconds": fit_seconds,
            "inference_mode": "direct_point_prediction",
            "preparation_runtime_group": (
                f"{common['execution_key']}_pmc_preparation"
            ),
            "preparation_runtime_seconds": preparation_seconds,
            "runtime_device": _pmc_runtime_device(pmc, method),
            "runtime_scope": (
                "consensus_discrepancy_preparation_plus_method_target_fit_"
                "prediction_wall"
            ),
            "runtime_shared_across_rows": True,
            "signed_error": error,
            "squared_error": error**2,
            "standalone_runtime_seconds": preparation_seconds + fit_seconds,
            "truth": truth,
            **_pmc_controls(
                pmc,
                method,
                target,
                common["n"],
                model_seed,
            ),
        }
    )
    return row

# Build one attempted PMC failure row.
def _pmc_failure_row(
    pmc,
    common,
    method,
    target,
    truth,
    preparation_seconds,
    fit_seconds,
    fit_group,
    model_seed,
    stage,
    error,
):
    row = _base_row(common, method, target, "failure", fit_group)
    row.update(
        {
            "failure_exception_type": type(error).__name__,
            "failure_message": str(error),
            "failure_stage": stage,
            "fit_prediction_runtime_seconds": fit_seconds,
            "inference_mode": "direct_point_prediction",
            "preparation_runtime_group": (
                f"{common['execution_key']}_pmc_preparation"
            ),
            "preparation_runtime_seconds": preparation_seconds,
            "runtime_device": _pmc_runtime_device(pmc, method),
            "runtime_scope": (
                "consensus_discrepancy_preparation_plus_time_until_failure"
            ),
            "runtime_shared_across_rows": True,
            "standalone_runtime_seconds": (
                preparation_seconds + (fit_seconds or 0.0)
            ),
            "truth": truth,
            **_pmc_controls(
                pmc,
                method,
                target,
                common["n"],
                model_seed,
            ),
        }
    )
    return row

# Resolve a method-specific model seed without sharing stochastic streams.
def _pmc_model_seed(method, model_seeds, target_index):
    if method not in model_seeds:
        return None
    return model_seeds[method] + target_index

# Build selected pre-method failures after raw sampling fails.
def _raw_failure_rows(pmc, targets, common, methods, model_seeds, error):
    rows = []
    for target_index, (target, _, truth) in enumerate(targets):
        for method in methods:
            current_seed = _pmc_model_seed(
                method, model_seeds, target_index
            )
            row = _base_row(common, method, target, "failure", None)
            row.update(
                {
                    "failure_exception_type": type(error).__name__,
                    "failure_message": str(error),
                    "failure_stage": "raw_component_sampling",
                    "inference_mode": "direct_point_prediction",
                    "runtime_clock": "not_run",
                    "runtime_device": "not_run",
                    "runtime_scope": (
                        "raw_component_sampling_failed_before_estimator_timing"
                    ),
                    "truth": truth,
                    **_pmc_controls(
                        pmc,
                        method,
                        target,
                        common["n"],
                        current_seed,
                    ),
                }
            )
            rows.append(row)
    return rows

# Attempt selected PMC rows after one shared preparation.
def _pmc_rows(
    pmc, targets, raw_samples, common, stem, methods, model_seeds
):
    start = perf_counter()
    try:
        inputs = prepare_pmc_inputs(raw_samples)
    except Exception as error:
        preparation_seconds = perf_counter() - start
        return [
            _pmc_failure_row(
                pmc,
                common,
                method,
                target,
                truth,
                preparation_seconds,
                None,
                f"{stem}_{method.lower()}_{target}",
                _pmc_model_seed(method, model_seeds, target_index),
                "pmc_preparation",
                error,
            )
            for target_index, (target, _, truth) in enumerate(targets)
            for method in methods
        ]
    preparation_seconds = perf_counter() - start
    rows = []
    for target_index, (target, transform, truth) in enumerate(targets):
        for method in methods:
            current_seed = _pmc_model_seed(
                method, model_seeds, target_index
            )
            fit_group = f"{stem}_{method.lower()}_{target}"
            fit_start = perf_counter()
            try:
                estimate, fit_seconds = fit_pmc_method(
                    pmc,
                    method,
                    target,
                    transform,
                    inputs,
                    current_seed,
                )
                rows.append(
                    _pmc_success_row(
                        pmc,
                        common,
                        method,
                        target,
                        truth,
                        estimate,
                        preparation_seconds,
                        fit_seconds,
                        fit_group,
                        current_seed,
                    )
                )
            except Exception as error:
                rows.append(
                    _pmc_failure_row(
                        pmc,
                        common,
                        method,
                        target,
                        truth,
                        preparation_seconds,
                        perf_counter() - fit_start,
                        fit_group,
                        current_seed,
                        f"{method.lower().replace('-', '_')}_fit_prediction",
                        error,
                    )
                )
    return rows

##########################
# Persistence and resume #
##########################

# Return one canonical JSON signature.
def _signature(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

# Return the exact configuration and batch signature.
def _config_signature(config, batch_index, selection):
    return _signature(
        {"batch_index": batch_index, "config": config, "selection": selection}
    )

# Return the SHA-256 digest of exact file bytes.
def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Serialize rectangular rows to canonical CSV bytes.
def _serialize_rows(rows):
    fields = sorted({field for row in rows for field in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")

# Validate one finite nonnegative runtime.
def _validate_runtime(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(f"Persisted {field} must be finite and nonnegative.")

# Validate persisted row status, accuracy, and runtime semantics.
def _validate_rows(rows):
    fit_groups = []
    preparation_times = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _row_fields():
            raise ValueError("Persisted row schema is incompatible.")
        valid = {
            "Oracle": {"success"},
            "PMC-OLS": {"success", "failure"},
            "PMC-MLP": {"success", "failure"},
            "PMC-GP": {"success", "failure"},
            "GBF": {"success", "failure"},
        }
        if row["method"] not in valid or row["status"] not in valid[row["method"]]:
            raise ValueError("Persisted method-status combination is invalid.")
        if row["target"] not in {"identity", "square"}:
            raise ValueError("Persisted target is invalid.")
        if row["runtime_clock"] not in {"not_run", "perf_counter_wall"}:
            raise ValueError("Persisted runtime clock is invalid.")
        if row["runtime_device"] not in {"cpu", "cuda", "not_run"}:
            raise ValueError("Persisted runtime device is invalid.")
        if row["gbf_implementation"] != _GBF_IMPLEMENTATION:
            raise ValueError("Persisted GBF implementation marker is invalid.")
        if row["gbf_not_applicable_reason"] is not None:
            raise ValueError("Persisted GBF applicability metadata is invalid.")
        if row["method"] == "GBF":
            _positive_integer(row["n_inference"], "persisted n_inference")
        elif row["n_inference"] is not None:
            raise ValueError("Persisted non-GBF n_inference must be null.")
        if row["method"] == "Oracle":
            if not isinstance(row["oracle_em_converged"], bool):
                raise ValueError("Persisted Oracle EM convergence is invalid.")
        elif row["oracle_em_converged"] is not None:
            raise ValueError("Persisted non-Oracle EM convergence must be null.")
        expected_mode = (
            "terminal_equal_weight_resample"
            if row["method"] == "GBF"
            else (
                "direct_point_prediction"
                if row["method"].startswith("PMC")
                else "not_applicable"
            )
        )
        if row["inference_mode"] != expected_mode:
            raise ValueError("Persisted inference mode is invalid.")
        _validate_runtime(
            row["standalone_runtime_seconds"],
            "standalone_runtime_seconds",
        )
        raw_failure = row["failure_stage"] == "raw_component_sampling"
        failure_fields = (
            "failure_exception_type",
            "failure_message",
            "failure_stage",
        )
        if row["status"] != "failure" and any(
            row[field] is not None for field in failure_fields
        ):
            raise ValueError("Persisted non-failure row has failure metadata.")
        if row["method"] == "GBF":
            for field in (
                "gbf_fitting_runtime_seconds",
                "gbf_fusion_runtime_seconds",
                "gbf_preparation_runtime_seconds",
            ):
                _validate_runtime(row[field], field)
            for field in (
                "gbf_fitting_runtime_group",
                "gbf_fusion_runtime_group",
                "gbf_preparation_runtime_group",
                "gbf_runtime_group",
                "gbf_artifact_dir",
            ):
                if not isinstance(row[field], str) or not row[field]:
                    raise ValueError(f"Persisted {field} is invalid.")
            if any(
                row[field] is not None
                for field in (
                    "fit_prediction_runtime_group",
                    "fit_prediction_runtime_seconds",
                    "preparation_runtime_group",
                    "preparation_runtime_seconds",
                )
            ):
                raise ValueError("Persisted GBF PMC runtime fields are invalid.")
            expected_total = (
                row["gbf_preparation_runtime_seconds"]
                + row["gbf_fitting_runtime_seconds"]
                + row["gbf_fusion_runtime_seconds"]
            )
            if not math.isclose(
                row["standalone_runtime_seconds"],
                expected_total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Persisted GBF runtime total is inconsistent.")
            if (
                row["runtime_shared_across_rows"] is not True
                or row["runtime_clock"] != "perf_counter_wall"
                or row["runtime_device"] != "cpu"
            ):
                raise ValueError("Persisted GBF runtime metadata is invalid.")
            if row["status"] == "success":
                for field in (
                    "absolute_error",
                    "estimate",
                    "signed_error",
                    "squared_error",
                    "truth",
                ):
                    value = row[field]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(
                            f"Persisted successful {field} is invalid."
                        )
                error = row["estimate"] - row["truth"]
                if (
                    not math.isclose(
                        row["signed_error"], error, abs_tol=1e-12
                    )
                    or not math.isclose(
                        row["absolute_error"], abs(error), abs_tol=1e-12
                    )
                    or not math.isclose(
                        row["squared_error"], error**2, abs_tol=1e-12
                    )
                    or row["runtime_scope"]
                    != "mixture_fitting_preparation_plus_gbf_fusion_wall"
                ):
                    raise ValueError("Persisted GBF success is inconsistent.")
            else:
                if any(
                    row[field] is not None
                    for field in (
                        "absolute_error",
                        "estimate",
                        "signed_error",
                        "squared_error",
                    )
                ):
                    raise ValueError("Persisted GBF failure has estimates.")
                if any(
                    not isinstance(row[field], str) or not row[field]
                    for field in failure_fields
                ):
                    raise ValueError("Persisted GBF failure metadata is invalid.")
                if row["failure_stage"] not in {
                    "raw_component_sampling",
                    "gbf_mixture_fitting",
                    "gbf_fusion",
                }:
                    raise ValueError("Persisted GBF failure stage is invalid.")
                if row["runtime_scope"] != "gbf_time_until_failure":
                    raise ValueError("Persisted GBF failure scope is invalid.")
                if (
                    isinstance(row["truth"], bool)
                    or not isinstance(row["truth"], Real)
                    or not math.isfinite(float(row["truth"]))
                ):
                    raise ValueError("Persisted GBF failure truth is invalid.")
            continue
        if any(
            row[field] is not None
            for field in (
                "gbf_artifact_dir",
                "gbf_fitting_runtime_group",
                "gbf_fitting_runtime_seconds",
                "gbf_fusion_runtime_group",
                "gbf_fusion_runtime_seconds",
                "gbf_preparation_runtime_group",
                "gbf_preparation_runtime_seconds",
                "gbf_runtime_group",
            )
        ):
            raise ValueError("Persisted non-GBF runtime fields are invalid.")
        if row["method"].startswith("PMC") and not raw_failure:
            preparation = row["preparation_runtime_seconds"]
            fit_seconds = row["fit_prediction_runtime_seconds"]
            _validate_runtime(preparation, "preparation_runtime_seconds")
            if fit_seconds is not None:
                _validate_runtime(fit_seconds, "fit_prediction_runtime_seconds")
            expected = preparation + (fit_seconds or 0.0)
            if not math.isclose(
                row["standalone_runtime_seconds"],
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Persisted PMC runtime total is inconsistent.")
            if row["runtime_shared_across_rows"] is not True:
                raise ValueError("Persisted PMC preparation must be shared.")
            fit_group = row["fit_prediction_runtime_group"]
            if not isinstance(fit_group, str) or not fit_group:
                raise ValueError("Persisted fit runtime group is invalid.")
            fit_groups.append(fit_group)
            group = row["preparation_runtime_group"]
            expected_group = f"{row['execution_key']}_pmc_preparation"
            if group != expected_group:
                raise ValueError("Persisted preparation runtime group is invalid.")
            preparation_times.setdefault(group, set()).add(preparation)
        elif row["method"] == "Oracle":
            fit_seconds = row["fit_prediction_runtime_seconds"]
            _validate_runtime(fit_seconds, "fit_prediction_runtime_seconds")
            if (
                row["fit_prediction_runtime_group"]
                != f"{row['execution_key']}_oracle"
                or row["preparation_runtime_group"] is not None
                or row["preparation_runtime_seconds"] is not None
                or row["runtime_shared_across_rows"] is not True
                or row["standalone_runtime_seconds"] != fit_seconds
            ):
                raise ValueError("Persisted Oracle runtime fields are invalid.")
        elif any(
            row[field] is not None
            for field in (
                "fit_prediction_runtime_group",
                "fit_prediction_runtime_seconds",
                "preparation_runtime_group",
                "preparation_runtime_seconds",
            )
        ) or (
            row["runtime_shared_across_rows"] is not False
            or row["standalone_runtime_seconds"] != 0.0
        ):
            raise ValueError("Persisted not-run runtime fields are invalid.")
        if raw_failure and (
            row["status"] != "failure"
            or row["runtime_clock"] != "not_run"
            or row["runtime_device"] != "not_run"
            or row["runtime_scope"]
            != "raw_component_sampling_failed_before_estimator_timing"
        ):
            raise ValueError("Persisted raw failure runtime is invalid.")
        if (
            not raw_failure
            and (
                row["runtime_clock"] != "perf_counter_wall"
                or (
                    row["method"] not in {"PMC-MLP", "PMC-GP"}
                    and row["runtime_device"] != "cpu"
                )
                or (
                    row["method"] in {"PMC-MLP", "PMC-GP"}
                    and row["runtime_device"] not in {"cpu", "cuda"}
                )
            )
        ):
            raise ValueError("Persisted attempted runtime metadata is invalid.")
        if row["status"] == "success":
            if (
                row["method"].startswith("PMC")
                and row["fit_prediction_runtime_seconds"] is None
            ):
                raise ValueError("Persisted successful PMC fit runtime is missing.")
            for field in (
                "absolute_error",
                "estimate",
                "signed_error",
                "squared_error",
                "truth",
            ):
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"Persisted successful {field} is invalid.")
            error = row["estimate"] - row["truth"]
            if not math.isclose(row["signed_error"], error, abs_tol=1e-12):
                raise ValueError("Persisted signed error is inconsistent.")
            if not math.isclose(row["absolute_error"], abs(error), abs_tol=1e-12):
                raise ValueError("Persisted absolute error is inconsistent.")
            if not math.isclose(row["squared_error"], error**2, abs_tol=1e-12):
                raise ValueError("Persisted squared error is inconsistent.")
            expected_scope = (
                "fitted_em_mixture_exact_product"
                if row["method"] == "Oracle"
                else (
                    "consensus_discrepancy_preparation_plus_method_target_fit_"
                    "prediction_wall"
                )
            )
            if row["runtime_scope"] != expected_scope:
                raise ValueError("Persisted successful runtime scope is invalid.")
        elif row["status"] == "failure":
            if any(
                row[field] is not None
                for field in (
                    "absolute_error",
                    "estimate",
                    "signed_error",
                    "squared_error",
                )
            ):
                raise ValueError("Persisted failure has scientific outputs.")
            for field in failure_fields:
                if not isinstance(row[field], str) or not row[field]:
                    raise ValueError(f"Persisted failure {field} is invalid.")
            if (
                isinstance(row["truth"], bool)
                or not isinstance(row["truth"], Real)
                or not math.isfinite(float(row["truth"]))
            ):
                raise ValueError("Persisted failure truth is invalid.")
            allowed_stages = {
                "raw_component_sampling",
                "pmc_preparation",
                f"{row['method'].lower().replace('-', '_')}_fit_prediction",
            }
            if row["failure_stage"] not in allowed_stages:
                raise ValueError("Persisted PMC failure stage is invalid.")
            if not raw_failure and row["runtime_scope"] != (
                "consensus_discrepancy_preparation_plus_time_until_failure"
            ):
                raise ValueError("Persisted failure runtime scope is invalid.")
    if len(fit_groups) != len(set(fit_groups)):
        raise ValueError("Persisted fit runtime groups must be unique.")
    if any(len(values) != 1 for values in preparation_times.values()):
        raise ValueError("Shared PMC preparation runtimes are inconsistent.")

# Validate persisted provenance without consulting the current host.
def _validate_provenance(provenance):
    required = {
        "machine",
        "numpy_version",
        "platform",
        "processor",
        "python_version",
        "runtime_device",
        "torch_deterministic_algorithms",
        "torch_num_threads",
        "torch_version",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("Persisted provenance schema is invalid.")
    if (
        provenance["runtime_device"] != "cpu"
        or provenance["torch_deterministic_algorithms"] is not True
        or isinstance(provenance["torch_num_threads"], bool)
        or provenance["torch_num_threads"] != 1
    ):
        raise ValueError("Persisted provenance invariants are invalid.")
    for field in required - {
        "torch_deterministic_algorithms",
        "torch_num_threads",
    }:
        if not isinstance(provenance[field], str):
            raise ValueError(f"Persisted provenance {field} is invalid.")

# Validate complete persisted execution rectangles and seeds.
def _validate_executions(
    rows,
    config,
    batch_index,
    planned_keys,
    methods,
    sample_size_seed_index,
):
    groups = {}
    for row in rows:
        _positive_integer(row["n"], "persisted n")
        _nonnegative_integer(row["repetition"], "persisted repetition")
        _nonnegative_integer(
            row["global_repetition"],
            "persisted global_repetition",
        )
        key = row["scenario"], row["n"], row["repetition"]
        if key not in planned_keys:
            raise ValueError(f"Persisted execution {key} is not configured.")
        groups.setdefault(key, []).append(row)
    expected = {
        (method, target)
        for method in methods
        for target in ("identity", "square")
    }
    oracles = {
        scenario["name"]: build_exact_product(scenario["nodes"])
        for scenario in config["scenarios"]
    }
    for key, group in groups.items():
        scenario_name, sample_size, repetition = key
        if len(group) != 2 * len(methods) or {
            (row["method"], row["target"]) for row in group
        } != expected:
            raise ValueError(f"Persisted execution {key} is partial or invalid.")
        shared_fields = {
            "GBF": (
                "algorithm_seed",
                "failure_exception_type",
                "failure_message",
                "failure_stage",
                "gbf_artifact_dir",
                "gbf_fitting_runtime_group",
                "gbf_fitting_runtime_seconds",
                "gbf_fusion_runtime_group",
                "gbf_fusion_runtime_seconds",
                "gbf_preparation_runtime_group",
                "gbf_preparation_runtime_seconds",
                "gbf_runtime_group",
                "n_inference",
                "runtime_scope",
                "standalone_runtime_seconds",
                "status",
            ),
            "Oracle": (
                "algorithm_seed",
                "fit_prediction_runtime_group",
                "fit_prediction_runtime_seconds",
                "oracle_em_converged",
                "runtime_scope",
                "standalone_runtime_seconds",
                "status",
            ),
        }
        for method, fields in shared_fields.items():
            pair = [row for row in group if row["method"] == method]
            if pair and any(
                row[field] != pair[0][field]
                for row in pair[1:]
                for field in fields
            ):
                raise ValueError(
                    f"Persisted execution {key} {method} rows disagree."
                )
        scenario_index = next(
            index
            for index, scenario in enumerate(config["scenarios"])
            if scenario["name"] == scenario_name
        )
        size_index = (
            config["component_sample_sizes"].index(sample_size)
            if sample_size_seed_index is None
            else sample_size_seed_index
        )
        raw_seed = comparison_seed(
            config["base_seed"],
            "raw_components",
            scenario_index,
            size_index,
            repetition,
        )
        model_seeds = {
            "PMC-MLP": comparison_seed(
                config["base_seed"],
                "pmc_mlp",
                scenario_index,
                size_index,
                repetition,
            ),
            "PMC-GP": comparison_seed(
                config["base_seed"],
                "pmc_gp",
                scenario_index,
                size_index,
                repetition,
            ),
        }
        algorithm_seeds = {
            method: comparison_seed(
                config["base_seed"],
                stream,
                scenario_index,
                size_index,
                repetition,
            )
            for method, stream in (("GBF", "gbf"), ("Oracle", "oracle"))
        }
        oracle = oracles[scenario_name]
        truths = {target: truth for target, _, truth in build_target_specs(oracle)}
        for row in group:
            reference_log_normalizer = row["reference_log_normalizer"]
            if (
                isinstance(reference_log_normalizer, bool)
                or not isinstance(reference_log_normalizer, Real)
                or not math.isfinite(float(reference_log_normalizer))
            ):
                raise ValueError(
                    f"Persisted execution {key} log-normalizer is invalid."
                )
            target_index = 0 if row["target"] == "identity" else 1
            expected_model = _pmc_model_seed(
                row["method"], model_seeds, target_index
            )
            expected_controls = (
                _pmc_controls(
                    config["pmc"],
                    row["method"],
                    row["target"],
                    sample_size,
                    expected_model,
                )
                if row["method"] in {"PMC-OLS", "PMC-MLP", "PMC-GP"}
                else {
                    "base_learning_rate": None,
                    "batch_size": None,
                    "ensemble_size": None,
                    "learning_rate_schedule": None,
                    "model_seed": None,
                    "n_training_steps": None,
                }
            )
            if (
                row["batch_index"] != batch_index
                or row["algorithm_seed"]
                != algorithm_seeds.get(row["method"])
                or row["global_repetition"] != repetition
                or (
                    row["method"] == "GBF"
                    and row["n_inference"] != sample_size
                )
                or row["execution_key"] != "|".join(map(str, key))
                or row["raw_seed"] != raw_seed
                or row["model_seed"] != expected_model
                or (
                    row["method"] in {"PMC-MLP", "PMC-GP"}
                    and row["failure_stage"] != "raw_component_sampling"
                    and row["runtime_device"]
                    != _pmc_runtime_device(config["pmc"], row["method"])
                )
                or row["node_count"]
                != len(config["scenarios"][scenario_index]["nodes"])
                or row["expanded_component_count"] != len(oracle.weights)
                or reference_log_normalizer != oracle.log_normalizer
                or row["reference_normalizer"] != oracle.normalizer
                or row["truth"] != truths[row["target"]]
            ):
                raise ValueError(f"Persisted execution {key} metadata is invalid.")
            if any(
                row[field] != value
                for field, value in expected_controls.items()
            ):
                raise ValueError(f"Persisted execution {key} controls are invalid.")
    return set(groups)

# Write exact bytes atomically and durably by file.
def _atomic_write(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary = Path(output_file.name)
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

# Return exact static design metadata.
def _design(config, batch_index, selection, provenance):
    return {
        "atomicity": (
            "Each file is atomic; the CSV/JSON pair is not transactional, "
            "and resume rejects a torn pair."
        ),
        "batch_index": batch_index,
        "pmc_input_rule": "Raw component samples only.",
        "pmc_discrepancy_centering": "mean",
        "pmc_gp_controls": config["pmc"]["gp"],
        "selected_methods": selection["methods"],
        "gbf_implementation": _GBF_IMPLEMENTATION,
        "gbf_input_rule": (
            "Fit node-specific weights, means, and standard deviations from "
            "raw samples using configured component counts; run the modified "
            "node-specific R adapter without generating parameters."
        ),
        "gbf_not_applicable_reason": _GBF_REASON,
        "gbf_controls": config["gbf"],
        "inference_mode": "direct_point_prediction_or_terminal_resample",
        "n_inference": None,
        "runtime_accounting": (
            "PMC preparation is shared across selected PMC rows; GBF fitting, "
            "preparation, and fusion are shared across both target rows."
        ),
        "runtime_provenance": provenance,
        "seed_derivation": (
            "base_seed + stream_offset + 1000000*scenario_index + "
            "10000*sample_size_index + 10*global_repetition + target_index"
        ),
        "selection": selection,
    }

# Persist canonical CSV and JSON artifacts atomically by file.
def _persist(artifact_dir, config, batch_index, selection, rows, provenance):
    _validate_rows(rows)
    csv_path = artifact_dir / "comparison_rows.csv"
    _atomic_write(csv_path, _serialize_rows(rows))
    output = {
        "config_signature": _config_signature(config, batch_index, selection),
        "csv_sha256": _file_sha256(csv_path),
        "design": _design(config, batch_index, selection, provenance),
        "rows": rows,
        "schema_version": _SCHEMA_VERSION,
    }
    payload = (json.dumps(output, allow_nan=False, indent=2) + "\n").encode()
    _atomic_write(artifact_dir / "comparison_results.json", payload)
    return output

# Load and strictly validate persisted artifacts.
def _load_resume(artifact_dir, config, batch_index, selection, planned_keys):
    csv_path = artifact_dir / "comparison_rows.csv"
    json_path = artifact_dir / "comparison_results.json"
    if not csv_path.exists() or not json_path.exists():
        raise ValueError("Resume requires both CSV and JSON artifacts.")
    output = json.loads(json_path.read_text(encoding="utf-8"))
    if set(output) != {
        "config_signature",
        "csv_sha256",
        "design",
        "rows",
        "schema_version",
    }:
        raise ValueError("Persisted JSON schema is incompatible.")
    if output["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Persisted schema version is incompatible.")
    if output["config_signature"] != _config_signature(config, batch_index, selection):
        raise ValueError("Persisted config signature is incompatible.")
    if output["csv_sha256"] != _file_sha256(csv_path):
        raise ValueError("Persisted CSV checksum is incompatible.")
    rows = output["rows"]
    if not isinstance(rows, list) or _serialize_rows(rows) != csv_path.read_bytes():
        raise ValueError("Persisted JSON and CSV rows are incompatible.")
    _validate_rows(rows)
    if not isinstance(output["design"], dict):
        raise ValueError("Persisted design metadata is incompatible.")
    provenance = output["design"].get("runtime_provenance")
    _validate_provenance(provenance)
    if output["design"] != _design(config, batch_index, selection, provenance):
        raise ValueError("Persisted design metadata is incompatible.")
    signature = _signature(provenance)
    if any(row["provenance_signature"] != signature for row in rows):
        raise ValueError("Persisted row provenance is inconsistent.")
    complete = _validate_executions(
        rows,
        config,
        batch_index,
        planned_keys,
        selection["methods"],
        selection.get("sample_size_seed_index"),
    )
    canonical = (json.dumps(output, allow_nan=False, indent=2) + "\n").encode()
    if json_path.read_bytes() != canonical:
        raise ValueError("Persisted JSON is not canonically serialized.")
    return output, rows, complete

#####################
# Comparison runner #
#####################

# Run one selected development comparison batch.
def run_comparison(
    config_path,
    artifact_dir,
    resume=False,
    batch_index=None,
    sample_size=None,
    repetition_start=None,
    repetition_count=None,
    methods=None,
    sample_size_seed_index=None,
):
    config = load_config(config_path)
    artifact_dir = Path(artifact_dir)
    csv_path = artifact_dir / "comparison_rows.csv"
    json_path = artifact_dir / "comparison_results.json"
    if resume and not (csv_path.exists() and json_path.exists()):
        raise ValueError("Resume requires both CSV and JSON artifacts.")
    if not resume and (csv_path.exists() or json_path.exists()):
        raise FileExistsError("A new run refuses to overwrite final artifacts.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sizes, repetitions, selection = _selected_work(
        config, batch_index, sample_size, repetition_start, repetition_count
    )
    if sample_size_seed_index is not None:
        _nonnegative_integer(
            sample_size_seed_index,
            "sample_size_seed_index",
        )
        if sample_size is None:
            raise ValueError(
                "sample_size_seed_index requires exactly one selected "
                "sample_size."
            )
    if sample_size_seed_index is not None:
        selection["sample_size_seed_index"] = sample_size_seed_index
    selected_methods = _selected_methods(config, methods)
    selection["methods"] = list(selected_methods)
    planned_keys = {
        (scenario["name"], current_size, repetition)
        for scenario in config["scenarios"]
        for current_size in sizes
        for repetition in repetitions
    }
    if resume:
        output, rows, complete = _load_resume(
            artifact_dir,
            config,
            batch_index,
            selection,
            planned_keys,
        )
        if complete == planned_keys:
            return output
    else:
        rows, complete = [], set()
    configure_torch(config["base_seed"])
    provenance = runtime_provenance()
    if resume and provenance != output["design"]["runtime_provenance"]:
        raise ValueError("Incomplete resume requires original provenance.")
    provenance_signature = _signature(provenance)
    oracles = [
        build_exact_product(scenario["nodes"])
        for scenario in config["scenarios"]
    ]
    latest = None
    for scenario_index, scenario in enumerate(config["scenarios"]):
        oracle = oracles[scenario_index]
        targets = build_target_specs(oracle)
        for sample_size in sizes:
            size_index = (
                config["component_sample_sizes"].index(sample_size)
                if sample_size_seed_index is None
                else sample_size_seed_index
            )
            for repetition in repetitions:
                key = scenario["name"], sample_size, repetition
                if key in complete:
                    continue
                raw_seed = comparison_seed(
                    config["base_seed"],
                    "raw_components",
                    scenario_index,
                    size_index,
                    repetition,
                )
                model_seeds = {
                    "PMC-MLP": comparison_seed(
                        config["base_seed"],
                        "pmc_mlp",
                        scenario_index,
                        size_index,
                        repetition,
                    ),
                    "PMC-GP": comparison_seed(
                        config["base_seed"],
                        "pmc_gp",
                        scenario_index,
                        size_index,
                        repetition,
                    ),
                }
                gbf_seed = comparison_seed(
                    config["base_seed"],
                    "gbf",
                    scenario_index,
                    size_index,
                    repetition,
                )
                oracle_seed = comparison_seed(
                    config["base_seed"],
                    "oracle",
                    scenario_index,
                    size_index,
                    repetition,
                )
                common = {
                    "algorithm_seed": None,
                    "batch_index": batch_index,
                    "execution_key": "|".join(map(str, key)),
                    "expanded_component_count": len(oracle.weights),
                    "global_repetition": repetition,
                    "model_seed": None,
                    "n": sample_size,
                    "node_count": len(scenario["nodes"]),
                    "provenance_signature": provenance_signature,
                    "raw_seed": raw_seed,
                    "reference_log_normalizer": oracle.log_normalizer,
                    "reference_normalizer": oracle.normalizer,
                    "repetition": repetition,
                    "scenario": scenario["name"],
                }
                stem = f"{scenario['name']}_n_{sample_size}_rep_{repetition}"
                execution_rows = []
                pmc_methods = [
                    method
                    for method in selected_methods
                    if method.startswith("PMC")
                ]
                # Let sampling errors propagate when the failure-free Oracle is selected.
                sampling_errors = (
                    () if "Oracle" in selected_methods else (Exception,)
                )
                try:
                    raw_samples = sample_raw_components(
                        scenario["nodes"],
                        sample_size,
                        raw_seed,
                    )
                except sampling_errors as error:
                    execution_rows.extend(
                        _raw_failure_rows(
                            config["pmc"],
                            targets,
                            common,
                            pmc_methods,
                            model_seeds,
                            error,
                        )
                    )
                    if "GBF" in selected_methods:
                        gbf_dir = artifact_dir / "gbf_artifacts" / stem
                        execution_rows.extend(
                            _gbf_failure_rows(
                                targets,
                                common,
                                f"{common['execution_key']}_gbf",
                                gbf_dir,
                                gbf_seed,
                                "raw_component_sampling",
                                error,
                            )
                        )
                else:
                    if "Oracle" in selected_methods:
                        execution_rows.extend(
                            _oracle_rows(
                                raw_samples,
                                _node_component_counts(scenario["nodes"]),
                                oracle_seed,
                                targets,
                                common,
                            )
                        )
                    if pmc_methods:
                        execution_rows.extend(
                            _pmc_rows(
                                config["pmc"],
                                targets,
                                raw_samples,
                                common,
                                stem,
                                pmc_methods,
                                model_seeds,
                            )
                        )
                    if "GBF" in selected_methods:
                        gbf_dir = artifact_dir / "gbf_artifacts" / stem
                        execution_rows.extend(
                            _gbf_rows(
                                targets,
                                common,
                                scenario,
                                raw_samples,
                                gbf_dir,
                                gbf_seed,
                                config["gbf"],
                            )
                        )
                rows.extend(execution_rows)
                latest = _persist(
                    artifact_dir,
                    config,
                    batch_index,
                    selection,
                    rows,
                    provenance,
                )
    return latest

# Run the command-line comparison.
def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: benchmark.py CONFIG ARTIFACT_DIR [--resume] "
            "[--batch-index INDEX] [--sample-size N] "
            "[--repetition-start START --repetition-count COUNT] "
            "[--methods METHOD[,METHOD...]] "
            "[--sample-size-seed-index INDEX]"
        )
    optional = sys.argv[3:]
    resume = "--resume" in optional
    values = {
        "--batch-index": None,
        "--sample-size": None,
        "--sample-size-seed-index": None,
        "--repetition-start": None,
        "--repetition-count": None,
    }
    for flag in values:
        if flag in optional:
            position = optional.index(flag)
            try:
                values[flag] = int(optional[position + 1])
            except (IndexError, ValueError) as error:
                raise SystemExit(f"{flag} requires an integer.") from error
            del optional[position : position + 2]
    methods = None
    if "--methods" in optional:
        position = optional.index("--methods")
        try:
            methods = optional[position + 1].split(",")
        except IndexError as error:
            raise SystemExit("--methods requires a comma-separated value.") from error
        del optional[position : position + 2]
    if resume:
        optional.remove("--resume")
    if optional:
        raise SystemExit(f"Unsupported arguments: {' '.join(optional)}")
    run_comparison(
        sys.argv[1],
        sys.argv[2],
        resume=resume,
        batch_index=values["--batch-index"],
        sample_size=values["--sample-size"],
        repetition_start=values["--repetition-start"],
        repetition_count=values["--repetition-count"],
        methods=methods,
        sample_size_seed_index=values["--sample-size-seed-index"],
    )

if __name__ == "__main__":
    main()
