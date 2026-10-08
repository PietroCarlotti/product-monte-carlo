"""Fit PMC estimators using only the saved respiratory batch posteriors."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.pmc.algorithms import MLPPMC, LinearPMC, consensus_discrepancy
from src.pmc.pmc_gp import GaussianProcessPMC, recommended_pmc_gp_config

DEFAULT_METHODS = ("PMC-OLS", "PMC-MLP")
SUPPORTED_METHODS = ("PMC-OLS", "PMC-MLP", "PMC-GP")


def load_batch_samples(path):
    """Load and flatten chain-grouped HMC coefficient samples."""
    samples = np.load(path)["beta"]
    if samples.ndim != 3 or not np.isfinite(samples).all():
        raise ValueError(
            f"{path} must contain finite beta samples with shape (chains, draws, coefficients)."
        )
    return samples.reshape(-1, samples.shape[-1])


def fit_estimate(estimator, consensus, discrepancy, seed):
    """Fit one PMC estimator and return its estimate and runtime."""
    torch.manual_seed(seed)
    started = perf_counter()
    estimator.fit(consensus, discrepancy)
    estimate = estimator.predict(np.zeros(discrepancy.shape[1]))
    return np.asarray(estimate, dtype=float).reshape(-1), perf_counter() - started


def fit_gp_estimate(target, transform, consensus, discrepancy, seed, overrides):
    """Fit one PMC-GP per coefficient and return the estimates and total runtime."""
    started = perf_counter()
    response = consensus if transform is None else transform(consensus)
    query = np.zeros(discrepancy.shape[1])
    estimate = []
    for index in range(consensus.shape[1]):
        estimator = GaussianProcessPMC(recommended_pmc_gp_config(target, **overrides))
        estimator.fit(discrepancy, response[:, index], seed=seed)
        estimate.append(estimator.predict(query).estimate)
    return np.asarray(estimate, dtype=float), perf_counter() - started


def build_mlp(config, transform, learning_rate, n_consensus):
    """Construct the configured MLP PMC estimator."""
    return MLPPMC(
        h=transform,
        hidden_features=tuple(config["hidden_features"]),
        n_training_steps=config["n_training_steps"],
        learning_rate=learning_rate,
        batch_size=min(config["batch_size"], n_consensus),
        device=config["device"],
        use_cosine_schedule=config["use_cosine_schedule"],
        ensemble_size=config["ensemble_size"],
        input_degree=1,
        activation="relu",
        discrepancy_centering=config["discrepancy_centering"],
        regression_loss="mse",
    )


def run(config_path):
    """Run batch-only PMC and persist estimates and runtimes."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    methods = config.get("methods", list(DEFAULT_METHODS))
    unsupported = [method for method in methods if method not in SUPPORTED_METHODS]
    if unsupported or not methods:
        raise ValueError(f"methods must be a nonempty subset of {SUPPORTED_METHODS}.")
    gp_overrides = config.get("gp", {})
    output_dir = config_path.parent
    batch_paths = [output_dir / path for path in config["batch_inputs"]]
    batch_samples = [load_batch_samples(path) for path in batch_paths]
    n_samples = min(len(samples) for samples in batch_samples)
    node_samples = np.stack([samples[:n_samples] for samples in batch_samples], axis=1)
    consensus, discrepancy = consensus_discrepancy(
        node_samples, sampling_method="iid", discrepancy_method="consecutive"
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    runtimes = {}
    for target, transform in (("identity", None), ("square", np.square)):
        seed = config["seeds"][target]
        for method in SUPPORTED_METHODS:
            if method not in methods:
                continue
            if method == "PMC-OLS":
                estimator = LinearPMC(h=transform, degree=1)
                estimate, runtime = fit_estimate(estimator, consensus, discrepancy, seed)
            elif method == "PMC-MLP":
                learning_rate = config["learning_rates"][target]
                estimator = build_mlp(config, transform, learning_rate, len(consensus))
                estimate, runtime = fit_estimate(estimator, consensus, discrepancy, seed)
            else:
                estimate, runtime = fit_gp_estimate(
                    target, transform, consensus, discrepancy, seed, gp_overrides
                )
            runtimes[f"{method}:{target}"] = runtime
            rows.extend(
                {
                    "method": method,
                    "target": target,
                    "coefficient": f"beta_{index}",
                    "estimate": float(value),
                }
                for index, value in enumerate(estimate)
            )
    with (output_dir / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("method", "target", "coefficient", "estimate"))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "runtime.json").write_text(json.dumps(runtimes, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config)
