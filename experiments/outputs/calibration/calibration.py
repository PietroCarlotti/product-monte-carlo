#!/usr/bin/env python3

"""Run the combined raw and bootstrap PMC-GP calibration experiment."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import json
import time
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# PMC library
from pmc import GaussianProcessPMC, PMCConfig
from pmc.algorithms import consensus_discrepancy
from pmc.dgp import UnivariateGaussianDGP


########################
# Configuration helpers #
########################

# Load and validate the flat JSON experiment specification.
def load_configuration(path):
    with path.open() as handle:
        configuration = json.load(handle)
    required = {
        "n_iterations", "iteration_start", "seed", "sample_size",
        "bootstrap_repetitions", "mu", "sigma", "coverage_levels", "gp",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    return configuration


# Return the output CSV path declared by the run specification.
def results_path(configuration, root):
    return root / configuration.get(
        "results_filename", "results.csv"
    )


# Return the requested outer repetition identifiers.
def repetition_ids(configuration):
    start = int(configuration["iteration_start"])
    stop = start + int(configuration["n_iterations"])
    return range(start, stop)


#########################
# Model and data helpers #
#########################

# Construct one independent PMC dataset for an outer repetition.
def build_dataset(configuration, repetition):
    dataset_seed = int(configuration["seed"]) + repetition
    dgp = UnivariateGaussianDGP(
        mu=configuration["mu"], sigma=configuration["sigma"]
    )
    random_state = np.random.get_state()
    try:
        np.random.seed(dataset_seed)
        nodes = dgp.sample(int(configuration["sample_size"]))
    finally:
        np.random.set_state(random_state)
    consensus, discrepancy = consensus_discrepancy(
        nodes, sampling_method="iid", discrepancy_method="consecutive"
    )
    truth = float(dgp.mean_product(h_name="identity")[0])
    return discrepancy, consensus[:, 0], truth, dataset_seed


# Return the PMC-GP configuration from the run specification.
def model_configuration(configuration):
    gp = configuration["gp"]
    return PMCConfig(
        inducing_count=int(gp["inducing_count"]),
        maximum_updates=int(gp["maximum_updates"]),
        batch_size=int(configuration["sample_size"]),
        learning_rate=float(gp["learning_rate"]),
        natural_gradient_learning_rate=float(gp["natural_gradient_learning_rate"]),
        minimum_epochs=int(gp["minimum_epochs"]),
        patience=int(gp["patience"]),
        relative_tolerance=float(gp["relative_tolerance"]),
        learn_inducing_locations=bool(gp["learn_inducing_locations"]),
        full_batch=bool(gp["full_batch"]),
        fixed_hyperparameters=bool(gp["fixed_hyperparameters"]),
        dtype=gp["dtype"],
        torch_num_threads=int(gp["torch_num_threads"]),
        kernel=gp["kernel"],
    )


# Fit PMC-GP and return its posterior estimate at the query point.
def fit_estimate(configuration, discrepancy, response, seed):
    model = GaussianProcessPMC(model_configuration(configuration))
    return model.fit(discrepancy, response, seed=seed).predict(confidence_level=0.95)


########################
# Checkpointing helpers #
########################

# Load the checkpoint CSV or return an empty frame.
def load_checkpoint(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# Report whether an outer repetition is already fully present in the checkpoint.
def is_completed(frame, repetition, coverage_count):
    if frame.empty or "outer_repetition" not in frame:
        return False
    rows = frame[frame["outer_repetition"] == repetition]
    return len(rows) == 2 * coverage_count


# Append one completed repetition atomically to the flat output CSV.
def checkpoint(frame, rows, path):
    updated = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    temporary_path = path.with_suffix(".tmp.csv")
    updated.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
    return updated


########################
# Calibration execution
########################

# Fit bootstrap refits and return raw plus bootstrap interval rows.
def run_repetition(configuration, repetition):
    discrepancy, response, truth, dataset_seed = build_dataset(configuration, repetition)
    original = fit_estimate(configuration, discrepancy, response, int(configuration["seed"]) + 10_000 + repetition)
    bootstrap_rng = np.random.default_rng(int(configuration["seed"]) + 20_000 + repetition)
    estimates = []
    started = time.perf_counter()
    for bootstrap_repetition in range(int(configuration["bootstrap_repetitions"])):
        indices = bootstrap_rng.integers(0, len(response), size=len(response))
        result = fit_estimate(
            configuration,
            discrepancy[indices],
            response[indices],
            int(configuration["seed"]) + 30_000 + 1_000 * repetition + bootstrap_repetition,
        )
        estimates.append(result.estimate)
    bootstrap_seconds = time.perf_counter() - started
    rows = []
    for level in configuration["coverage_levels"]:
        level = float(level)
        interval_z = norm.ppf((1.0 + level) / 2.0)
        raw_lower = original.estimate - interval_z * original.posterior_standard_deviation
        raw_upper = original.estimate + interval_z * original.posterior_standard_deviation
        tail_probability = (1.0 - level) / 2.0
        bootstrap_lower, bootstrap_upper = np.quantile(
            estimates, [tail_probability, 1.0 - tail_probability]
        )
        common = {
            "n": int(configuration["sample_size"]),
            "outer_repetition": repetition,
            "dataset_seed": dataset_seed,
            "nominal_coverage": level,
            "estimate": original.estimate,
            "truth": truth,
            "original_posterior_standard_deviation": original.posterior_standard_deviation,
            "bootstrap_repetitions": int(configuration["bootstrap_repetitions"]),
            "bootstrap_estimate_standard_deviation": float(np.std(estimates, ddof=1)),
            "bootstrap_wall_time_seconds": bootstrap_seconds,
            "kernel": configuration["gp"]["kernel"],
            "inducing_count": int(configuration["gp"]["inducing_count"]),
            "hyperparameter_mode": "empirical_bayes",
        }
        rows.extend(
            [
                {
                    **common,
                    "interval_method": "raw_gaussian",
                    "interval_lower": float(raw_lower),
                    "interval_upper": float(raw_upper),
                    "interval_width": float(raw_upper - raw_lower),
                    "covered": bool(raw_lower <= truth <= raw_upper),
                },
                {
                    **common,
                    "interval_method": "pairs_bootstrap_percentile",
                    "interval_lower": float(bootstrap_lower),
                    "interval_upper": float(bootstrap_upper),
                    "interval_width": float(bootstrap_upper - bootstrap_lower),
                    "covered": bool(bootstrap_lower <= truth <= bootstrap_upper),
                },
            ]
        )
    return rows


# Run requested repetitions and checkpoint every completed repetition.
def run_experiment(configuration, root):
    path = results_path(configuration, root)
    frame = load_checkpoint(path)
    coverage_count = len(configuration["coverage_levels"])
    repetition_range = repetition_ids(configuration)
    repetition_list = list(repetition_range)
    for repetition in repetition_list:
        if is_completed(frame, repetition, coverage_count):
            continue
        frame = checkpoint(frame, run_repetition(configuration, repetition), path)
        print(
            f"completed outer repetition {repetition + 1}/{max(repetition_list) + 1}",
            flush=True,
        )
    return frame


######################
# Script entry point #
######################

# Parse the optional run-spec path.
def parse_arguments(root):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", type=Path,
        default=root / "calibration.json",
    )
    return parser.parse_args()


# Run the configured calibration batch and report its checkpoint.
def main():
    root = Path(__file__).resolve().parent
    args = parse_arguments(root)
    configuration = load_configuration(args.config.resolve())
    frame = run_experiment(configuration, root)
    print(f"rows={len(frame)}", flush=True)
    print(f"results={results_path(configuration, root)}", flush=True)


if __name__ == "__main__":
    main()
