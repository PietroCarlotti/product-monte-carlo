#!/usr/bin/env python3

"""Compare PMC-GP and GBF interval calibration across per-node sample sizes."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import functools
import importlib.util
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# PMC library
from pmc.algorithms import consensus_discrepancy
from pmc.dgp import UnivariateGaussianDGP

####################
# Shared constants #
####################

EXPERIMENT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = EXPERIMENT_DIR.parent / "calibration" / "calibration.py"
GBF_RUNNER_PATH = EXPERIMENT_DIR / "gbf_runner.R"
SAMPLE_SIZE_SEED_STRIDE = 100_000
MAXIMUM_SAMPLE_SIZE_COUNT = 10
PMC_SEED_OFFSET = 1_000_000
GBF_SEED_OFFSET = 2_000_000
# Place per-worker PSOCK ports outside R's random range to avoid collisions
GBF_PARALLEL_PORT_BASE = 21_000

#########################
# Configuration helpers #
#########################

# Load and validate the JSON experiment specification.
def load_configuration(path):
    with path.open() as handle:
        configuration = json.load(handle)
    required = {
        "seed", "mu", "sigma", "sample_sizes", "n_repetitions",
        "coverage_levels", "gp", "gbf",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    if len(configuration["sample_sizes"]) > MAXIMUM_SAMPLE_SIZE_COUNT:
        raise ValueError("Too many sample sizes for the seed layout")
    if int(configuration["n_repetitions"]) > SAMPLE_SIZE_SEED_STRIDE:
        raise ValueError("Too many repetitions for the seed layout")
    n_inference = configuration["gbf"].get("n_inference")
    if n_inference != "sample_size" and (
        isinstance(n_inference, bool)
        or not isinstance(n_inference, int)
        or n_inference <= 0
    ):
        raise ValueError(
            'gbf n_inference must be a positive integer or "sample_size"'
        )
    return configuration

# Return the dataset seed for one sample-size index and repetition.
def dataset_seed(configuration, sample_size_index, repetition):
    stride = SAMPLE_SIZE_SEED_STRIDE * sample_size_index
    return int(configuration["seed"]) + stride + repetition

# Return the number of GBF output particles for one sample size.
def resolve_n_inference(configuration, sample_size):
    n_inference = configuration["gbf"]["n_inference"]
    if n_inference == "sample_size":
        return int(sample_size)
    return int(n_inference)

##########################
# Model and data helpers #
##########################

# Draw one dataset and its consensus/discrepancy representation.
def build_dataset(configuration, sample_size, seed):
    dgp = UnivariateGaussianDGP(mu=configuration["mu"], sigma=configuration["sigma"])
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        nodes = dgp.sample(int(sample_size))
    finally:
        np.random.set_state(random_state)
    consensus, discrepancy = consensus_discrepancy(
        nodes, sampling_method="iid", discrepancy_method="consecutive"
    )
    return {
        "nodes": nodes[:, :, 0],
        "discrepancy": discrepancy,
        "response": consensus[:, 0],
    }

# Return the exact product mean implied by the node parameters.
def product_truth(configuration):
    dgp = UnivariateGaussianDGP(mu=configuration["mu"], sigma=configuration["sigma"])
    return float(dgp.mean_product(h_name="identity")[0])

# Import the sibling calibration runner to reuse its PMC-GP fitting logic.
@functools.cache
def load_calibration_module():
    spec = importlib.util.spec_from_file_location("calibration", CALIBRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Fit PMC-GP once and return its posterior summary and runtime.
def fit_pmc_gp(gp, discrepancy, response, seed):
    specification = {"gp": gp, "sample_size": len(response)}
    calibration = load_calibration_module()
    started = time.perf_counter()
    result = calibration.fit_estimate(specification, discrepancy, response, seed)
    return {
        "estimate": float(result.estimate),
        "standard_error": float(result.posterior_standard_deviation),
        "runtime_seconds": time.perf_counter() - started,
    }

#######################
# GBF batch execution #
#######################

# Write node samples for a chunk of repetitions in the R runner's layout.
def write_gbf_samples(path, chunk):
    frames = []
    for repetition, seed, nodes in chunk:
        frame = pd.DataFrame(
            nodes, columns=[f"node_{index + 1}" for index in range(nodes.shape[1])]
        )
        frame.insert(0, "gbf_seed", seed)
        frame.insert(0, "repetition", repetition)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format="%.17g")

# Run GBF for all repetitions across parallel Rscript processes.
def run_gbf_chunks(configuration, sample_size, tasks, workers, work_dir):
    settings = {
        **configuration["gbf"],
        "n_inference": resolve_n_inference(configuration, sample_size),
    }
    settings_path = work_dir / "gbf_settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    chunks = [chunk for chunk in np.array_split(np.arange(len(tasks)), workers) if len(chunk)]
    processes = []
    for chunk_index, indices in enumerate(chunks):
        sample_path = work_dir / f"samples_{chunk_index}.csv"
        output_path = work_dir / f"gbf_{chunk_index}.csv"
        log_path = work_dir / f"gbf_{chunk_index}.log"
        write_gbf_samples(sample_path, [tasks[index] for index in indices])
        command = [
            "Rscript", str(GBF_RUNNER_PATH),
            str(sample_path), str(settings_path), str(output_path),
        ]
        environment = {
            **os.environ,
            "R_PARALLEL_PORT": str(GBF_PARALLEL_PORT_BASE + chunk_index),
        }
        log_handle = log_path.open("w")
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, env=environment
        )
        processes.append((process, log_handle, log_path, output_path))
    frames = []
    for process, log_handle, log_path, output_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            raise RuntimeError(
                f"GBF runner failed ({return_code}):\n{log_path.read_text()}"
            )
        frames.append(pd.read_csv(output_path))
    return pd.concat(frames, ignore_index=True).set_index("repetition")

#########################
# Interval construction #
#########################

# Expand one fitted estimate into interval rows for every nominal level.
def interval_rows(configuration, common, estimate, standard_error, truth):
    rows = []
    for level in configuration["coverage_levels"]:
        level = float(level)
        interval_z = norm.ppf((1.0 + level) / 2.0)
        lower = estimate - interval_z * standard_error
        upper = estimate + interval_z * standard_error
        covered = bool(lower <= truth <= upper)
        rows.append(
            {
                **common,
                "nominal_coverage": level,
                "estimate": estimate,
                "truth": truth,
                "standard_error": standard_error,
                "interval_lower": lower,
                "interval_upper": upper,
                "interval_width": upper - lower,
                "covered": covered,
            }
        )
    return rows

# Convert one GBF repetition summary into interval rows.
def gbf_rows(configuration, common, summary, truth):
    n_inference = resolve_n_inference(configuration, common["sample_size"])
    final_ess = summary["final_ess"]
    if np.isfinite(final_ess):
        effective_size = final_ess
        effective_size_source = "final_ess"
    else:
        effective_size = n_inference
        effective_size_source = "n_inference"
    standard_error = summary["draws_standard_deviation"] / np.sqrt(effective_size)
    gbf_common = {
        **common,
        "method": "gbf",
        "runtime_seconds": summary["runtime_seconds"],
        "final_ess": final_ess,
        "final_cess": summary["final_cess"],
        "effective_size_source": effective_size_source,
        "n_inference": n_inference,
        "n_time_steps": summary["n_time_steps"],
    }
    return interval_rows(
        configuration, gbf_common, summary["estimate"], standard_error, truth
    )

# Convert one PMC-GP fit into interval rows.
def pmc_rows(configuration, common, summary, truth):
    pmc_common = {
        **common,
        "method": "pmc_gp",
        "runtime_seconds": summary["runtime_seconds"],
    }
    return interval_rows(
        configuration, pmc_common, summary["estimate"], summary["standard_error"], truth
    )

#########################
# Checkpointing helpers #
#########################

# Load the checkpoint CSV or return an empty frame.
def load_checkpoint(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

# Return the repetitions already complete for one sample size.
def completed_repetitions(frame, sample_size, row_count):
    if frame.empty:
        return set()
    rows = frame[frame["sample_size"] == sample_size]
    counts = rows.groupby("repetition").size()
    return set(counts[counts == row_count].index)

# Append completed rows atomically to the flat output CSV.
def checkpoint(frame, rows, path):
    updated = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    temporary_path = path.with_suffix(".tmp.csv")
    updated.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
    return updated

########################
# Experiment execution #
########################

# Run both methods on one batch of repetitions and return their rows.
def run_batch(configuration, sample_size_index, repetitions, workers, pool, root):
    sample_size = int(configuration["sample_sizes"][sample_size_index])
    truth = product_truth(configuration)
    seeds = {
        repetition: dataset_seed(configuration, sample_size_index, repetition)
        for repetition in repetitions
    }
    datasets = {
        repetition: build_dataset(configuration, sample_size, seeds[repetition])
        for repetition in repetitions
    }
    pmc_futures = {
        repetition: pool.submit(
            fit_pmc_gp,
            configuration["gp"],
            datasets[repetition]["discrepancy"],
            datasets[repetition]["response"],
            seeds[repetition] + PMC_SEED_OFFSET,
        )
        for repetition in repetitions
    }
    gbf_tasks = [
        (repetition, seeds[repetition] + GBF_SEED_OFFSET, datasets[repetition]["nodes"])
        for repetition in repetitions
    ]
    with tempfile.TemporaryDirectory(dir=root, prefix="gbf_batch_") as work_dir:
        gbf_summaries = run_gbf_chunks(
            configuration, sample_size, gbf_tasks, workers, Path(work_dir)
        )
    rows = []
    for repetition in repetitions:
        common = {
            "sample_size": sample_size,
            "repetition": repetition,
            "dataset_seed": seeds[repetition],
        }
        rows.extend(pmc_rows(configuration, common, pmc_futures[repetition].result(), truth))
        rows.extend(gbf_rows(configuration, common, gbf_summaries.loc[repetition], truth))
    return rows

# Return original sample-size indices to run, optionally restricted to one size.
def sample_size_indices(configuration, sample_size):
    sample_sizes = [int(size) for size in configuration["sample_sizes"]]
    if sample_size is None:
        return list(range(len(sample_sizes)))
    if sample_size not in sample_sizes:
        raise ValueError(f"Sample size {sample_size} is not in {sample_sizes}")
    return [sample_sizes.index(sample_size)]

# Run every pending repetition for the selected sample sizes with batch checkpoints.
def run_experiment(configuration, root, workers, indices):
    path = root / configuration.get("results_filename", "results.csv")
    frame = load_checkpoint(path)
    row_count = 2 * len(configuration["coverage_levels"])
    batch_size = int(configuration.get("checkpoint_batch_size", 4 * workers))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for sample_size_index in indices:
            sample_size = configuration["sample_sizes"][sample_size_index]
            done = completed_repetitions(frame, int(sample_size), row_count)
            pending = [
                repetition for repetition in range(int(configuration["n_repetitions"]))
                if repetition not in done
            ]
            for start in range(0, len(pending), batch_size):
                repetitions = pending[start:start + batch_size]
                started = time.perf_counter()
                rows = run_batch(
                    configuration, sample_size_index, repetitions, workers, pool, root
                )
                frame = checkpoint(frame, rows, path)
                print(
                    f"S={sample_size} repetitions {repetitions[0]}-{repetitions[-1]} "
                    f"done in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return frame, path

######################
# Script entry point #
######################

# Parse the run-spec path, output directory, worker count, and sample size.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", type=Path,
        default=EXPERIMENT_DIR / "calibration_sample_size.json",
    )
    parser.add_argument("output_dir", nargs="?", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()

# Run the configured calibration sweep and report its checkpoint.
def main():
    args = parse_arguments()
    configuration = load_configuration(args.config.resolve())
    workers = args.workers or int(configuration.get("n_workers", 1))
    indices = sample_size_indices(configuration, args.sample_size)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frame, path = run_experiment(configuration, root, workers, indices)
    print(f"rows={len(frame)}", flush=True)
    print(f"results={path}", flush=True)

if __name__ == "__main__":
    main()
