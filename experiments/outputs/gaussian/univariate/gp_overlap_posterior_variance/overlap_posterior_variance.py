#!/usr/bin/env python3

"""Measure PMC-GP latent posterior variance as Gaussian overlap decreases."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

# Third-party libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot

# PMC library
from pmc import GaussianProcessPMC, PMCConfig

#########################
# Configuration helpers #
#########################

# Load and validate the overlap experiment configuration
def load_configuration(path):
    with path.open() as handle:
        configuration = json.load(handle)
    required = {
        "repetitions",
        "sample_size",
        "mu_values",
        "seed",
        "device",
        "band_quantiles",
        "gp",
        "results_filename",
        "figure_filename",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    for name in ["repetitions", "sample_size", "seed"]:
        value = configuration[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if configuration["repetitions"] <= 0 or configuration["sample_size"] <= 1:
        raise ValueError("repetitions must be positive and sample_size must exceed one")
    mu_values = configuration["mu_values"]
    if (
        not isinstance(mu_values, list)
        or not mu_values
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in mu_values
        )
    ):
        raise ValueError("mu_values must contain finite numeric values")
    if len(set(mu_values)) != len(mu_values) or 0 not in mu_values:
        raise ValueError("mu_values must be unique and include zero")
    band_quantiles = configuration["band_quantiles"]
    if (
        not isinstance(band_quantiles, list)
        or len(band_quantiles) != 2
        or not 0 < band_quantiles[0] < 0.5 < band_quantiles[1] < 1
    ):
        raise ValueError("band_quantiles must straddle 0.5 inside (0, 1)")
    if configuration["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    for name in ["results_filename", "figure_filename"]:
        value = configuration[name]
        if not isinstance(value, str) or Path(value).name != value:
            raise ValueError(f"{name} must be a plain filename")
    model_configuration(configuration)
    return configuration


# Build the production PMC-GP configuration
def model_configuration(configuration):
    gp = configuration["gp"]
    arguments = {
        "inducing_count": int(gp["inducing_count"]),
        "maximum_updates": int(gp["maximum_updates"]),
        "batch_size": int(gp["batch_size"]),
        "learning_rate": float(gp["learning_rate"]),
        "natural_gradient_learning_rate": float(
            gp["natural_gradient_learning_rate"]
        ),
        "minimum_epochs": int(gp["minimum_epochs"]),
        "patience": int(gp["patience"]),
        "relative_tolerance": float(gp["relative_tolerance"]),
        "learn_inducing_locations": bool(gp["learn_inducing_locations"]),
        "full_batch": bool(gp["full_batch"]),
        "dtype": gp["dtype"],
        "torch_num_threads": int(gp["torch_num_threads"]),
        "kernel": gp["kernel"],
        "device": configuration["device"],
    }
    return PMCConfig(**arguments)


# Hash the complete experiment configuration for checkpoint provenance
def configuration_fingerprint(configuration):
    payload = json.dumps(
        configuration,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

################
# Data helpers #
################

# Draw common standard-normal node values for one repetition
def build_base_draws(configuration, repetition):
    dataset_seed = int(configuration["seed"]) + repetition
    generator = np.random.default_rng(dataset_seed)
    sample_size = int(configuration["sample_size"])
    return (
        generator.standard_normal(sample_size),
        generator.standard_normal(sample_size),
        dataset_seed,
    )


# Construct consensus and discrepancy from shared base draws at one mean shift
def build_dataset(first_base, second_base, mu):
    first_node = first_base
    second_node = second_base + mu
    consensus = (first_node + second_node) / 2
    discrepancy = (first_node - second_node)[:, np.newaxis]
    return discrepancy, consensus

#################
# Model fitting #
#################

# Fit the complete-overlap model and return its frozen fit metadata
def fit_reference_model(configuration, discrepancy, response, seed):
    estimator = GaussianProcessPMC(model_configuration(configuration)).fit(
        discrepancy,
        response,
        seed=seed,
    )
    result = estimator.predict(discrepancy=0.0)
    diagnostics = result.diagnostics
    metadata = {
        "lengthscale": float(diagnostics["final_lengthscale"]),
        "outputscale": float(diagnostics["final_outputscale"]),
        "noise": float(diagnostics["final_noise"]),
        "response_scale": float(response.std()),
        "training_time_seconds": float(result.training_time_seconds),
    }
    return estimator, metadata


# Run one common-random-number repetition across every mean shift
def run_repetition(configuration, repetition):
    first_base, second_base, dataset_seed = build_base_draws(
        configuration, repetition
    )
    reference_discrepancy, reference_response = build_dataset(
        first_base,
        second_base,
        0.0,
    )
    hyperparameter_seed = int(configuration["seed"]) + 100_000 + repetition
    estimator, hyperparameters = fit_reference_model(
        configuration,
        reference_discrepancy,
        reference_response,
        hyperparameter_seed,
    )
    rows = []
    fingerprint = configuration_fingerprint(configuration)
    mu_values = [float(value) for value in configuration["mu_values"]]
    for mu_index, mu in enumerate(mu_values):
        result = estimator.predict(discrepancy=mu)
        posterior_variance = result.posterior_standard_deviation**2
        prior_variance = (
            hyperparameters["outputscale"]
            * hyperparameters["response_scale"] ** 2
        )
        rows.append(
            {
                "repetition": repetition,
                "configuration_fingerprint": fingerprint,
                "mu": mu,
                "n": int(configuration["sample_size"]),
                "dataset_seed": dataset_seed,
                "hyperparameter_seed": hyperparameter_seed,
                "prediction_index": mu_index,
                "device": configuration["device"],
                "kernel": configuration["gp"]["kernel"],
                "inducing_count": int(configuration["gp"]["inducing_count"]),
                "maximum_updates": int(configuration["gp"]["maximum_updates"]),
                "estimate": result.estimate + mu / 2,
                "truth": mu / 2,
                "physical_query_discrepancy": 0.0,
                "reference_query_discrepancy": mu,
                "translation_equivalence": (
                    "D_mu=D_0-mu; C_mu=C_0+mu/2"
                ),
                "posterior_standard_deviation": (
                    result.posterior_standard_deviation
                ),
                "posterior_variance": posterior_variance,
                "prior_variance": prior_variance,
                "learned_lengthscale": hyperparameters["lengthscale"],
                "learned_outputscale": hyperparameters["outputscale"],
                "learned_noise": hyperparameters["noise"],
                "response_scale": hyperparameters["response_scale"],
                "hyperparameter_training_time_seconds": hyperparameters[
                    "training_time_seconds"
                ],
                "training_time_seconds": hyperparameters[
                    "training_time_seconds"
                ],
                "interval_method": result.interval_method,
                "query_outside_training_range": result.diagnostics[
                    "query_outside_training_range"
                ],
                "local_effective_sample_size": result.diagnostics[
                    "local_effective_sample_size"
                ],
            }
        )
    return rows

##########################
# Results and checkpoint #
##########################

# Load a prior checkpoint or return an empty result frame
def load_checkpoint(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# Reject checkpoint rows produced by any other experiment configuration
def validate_checkpoint(frame, configuration):
    if frame.empty:
        return
    if "configuration_fingerprint" not in frame.columns:
        raise ValueError("Checkpoint is missing configuration provenance")
    observed = set(frame["configuration_fingerprint"])
    expected = {configuration_fingerprint(configuration)}
    if observed != expected:
        raise ValueError("Checkpoint was produced by a different configuration")


# Report whether one repetition contains every requested mean shift
def repetition_is_complete(frame, repetition, mu_values):
    if frame.empty or "repetition" not in frame.columns:
        return False
    rows = frame[frame["repetition"] == repetition]
    return len(rows) == len(mu_values) and set(rows["mu"]) == set(mu_values)


# Replace the checkpoint atomically with completed repetition rows
def checkpoint(frame, rows, path):
    updated = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    temporary_path = path.with_suffix(".tmp.csv")
    updated.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
    return updated


# Validate the complete tidy result design and positive latent variances
def validate_results(frame, configuration):
    expected_repetitions = set(range(int(configuration["repetitions"])))
    expected_mu = {float(value) for value in configuration["mu_values"]}
    observed_repetitions = set(frame["repetition"])
    if observed_repetitions != expected_repetitions:
        raise ValueError("Results do not contain every configured repetition")
    for repetition, rows in frame.groupby("repetition"):
        if len(rows) != len(expected_mu) or set(rows["mu"]) != expected_mu:
            raise ValueError(
                f"Repetition {repetition} does not contain every mean shift"
            )
    for column in ["posterior_variance", "prior_variance"]:
        values = frame[column]
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must contain positive finite values")


# Run and checkpoint every configured repetition
def run_experiment(configuration, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / configuration["results_filename"]
    frame = load_checkpoint(path)
    validate_checkpoint(frame, configuration)
    mu_values = [float(value) for value in configuration["mu_values"]]
    for repetition in range(int(configuration["repetitions"])):
        if repetition_is_complete(frame, repetition, mu_values):
            continue
        if not frame.empty and "repetition" in frame.columns:
            frame = frame[frame["repetition"] != repetition].copy()
        rows = run_repetition(configuration, repetition)
        frame = checkpoint(frame, rows, path)
        print(
            f"completed repetition {repetition + 1}/"
            f"{configuration['repetitions']}",
            flush=True,
        )
    validate_results(frame, configuration)
    return frame.sort_values(["repetition", "mu"]).reset_index(drop=True)

#####################
# Figure generation #
#####################

# Summarize latent posterior variance across repetitions
def summarize_results(frame, band_quantiles):
    lower_quantile, upper_quantile = band_quantiles
    return (
        frame.groupby("mu", as_index=False)["posterior_variance"]
        .agg(
            median="median",
            lower=lambda values: values.quantile(lower_quantile),
            upper=lambda values: values.quantile(upper_quantile),
        )
        .sort_values("mu")
    )


# Save the median latent variance, uncertainty band, and prior reference
def save_figure(
    frame,
    configuration,
    output_directory,
    maximum_plot_mu=None,
    show_band=True,
    show_prior_reference=True,
):
    plot_frame = frame
    if maximum_plot_mu is not None:
        plot_frame = frame[frame["mu"] <= maximum_plot_mu]
        if plot_frame.empty:
            raise ValueError("maximum_plot_mu excludes every mean shift")
    summary = summarize_results(
        plot_frame,
        configuration["band_quantiles"],
    )
    prior_reference = float(
        frame.drop_duplicates("repetition")["prior_variance"].median()
    )
    lower_quantile, upper_quantile = configuration["band_quantiles"]
    figure, axis = pyplot.subplots(figsize=(5.2, 3.4))
    if show_band:
        axis.fill_between(
            summary["mu"],
            summary["lower"],
            summary["upper"],
            color="#56B4E9",
            alpha=0.3,
            label=(
                f"{100 * lower_quantile:.0f}th--"
                f"{100 * upper_quantile:.0f}th percentile"
            ),
        )
    axis.plot(
        summary["mu"],
        summary["median"],
        color="#0072B2",
        marker="o",
    )
    if show_prior_reference:
        axis.axhline(
            prior_reference,
            color="black",
            linestyle="--",
            label="Median latent prior variance",
        )
    axis.set_yscale("log")
    axis.set_xlabel(r"$\mu$")
    axis.set_ylabel(
        r"$\operatorname{Var}\!\left(f_h(0)\mid C^{(1:S)},D^{(1:S)}\right)$"
    )
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False)
    figure.tight_layout()
    path = output_directory / configuration["figure_filename"]
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return path

######################
# Command-line entry #
######################

# Parse the config path and optional output-directory override
def parse_arguments(experiment_directory):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=experiment_directory / "config.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--maximum-plot-mu", type=float)
    parser.add_argument("--hide-band", action="store_true")
    parser.add_argument("--hide-prior-reference", action="store_true")
    return parser.parse_args()


# Run the configured experiment and generate its summary figure
def main():
    experiment_directory = Path(__file__).resolve().parent
    arguments = parse_arguments(experiment_directory)
    config_path = arguments.config.resolve()
    configuration = load_configuration(config_path)
    output_directory = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else config_path.parent
    )
    frame = run_experiment(configuration, output_directory)
    figure_path = save_figure(
        frame,
        configuration,
        output_directory,
        maximum_plot_mu=arguments.maximum_plot_mu,
        show_band=not arguments.hide_band,
        show_prior_reference=not arguments.hide_prior_reference,
    )
    print(
        f"results={output_directory / configuration['results_filename']}",
        flush=True,
    )
    print(f"figure={figure_path}", flush=True)


if __name__ == "__main__":
    main()
