"""Fit the shared Bayesian logistic model independently on data batches."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from src.applications.hmc_checkpointing import (
    atomic_savez,
    atomic_write_json,
    canonical_result_is_complete,
    chain_checkpoint_is_valid,
    chain_checkpoint_path,
    chain_runtime_path,
    combine_chain_checkpoints,
)
from src.models.bayesian_logistic_regression import (
    BayesianLogisticConfig,
    BayesianLogisticResult,
    add_diagnostics,
    diagnostics_payload,
    fit,
)


# Validate data columns and construct numbered partitions.
def _prepare_partitions(
    input_path: Path,
    n_batches: int | None,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
    partition_column: str | None,
) -> tuple[pd.DataFrame, tuple[str, ...], int, list[tuple[int, np.ndarray]]]:
    data = pd.read_csv(input_path)
    if partition_column is not None:
        if partition_column not in data.columns:
            raise ValueError(f"Unknown partition column: {partition_column}")
        n_batches = data[partition_column].nunique()
    if n_batches is None or n_batches < 1:
        raise ValueError("Could not infer a positive number of batches.")
    excluded = set(metadata_columns + (response_column,))
    required = excluded.difference(data.columns)
    if required:
        raise ValueError(f"Input is missing required columns: {sorted(required)}")
    features = feature_columns or tuple(c for c in data.columns if c not in excluded)
    if not set(features).issubset(data.columns):
        raise ValueError("Input is missing one or more feature columns.")
    if partition_column is None:
        partitions = [
            (batch_number, indices)
            for batch_number, indices in enumerate(
                np.array_split(np.arange(len(data)), n_batches), start=1
            )
        ]
    else:
        labels = sorted(int(label) for label in data[partition_column].unique())
        expected_labels = list(range(1, n_batches + 1))
        if labels != expected_labels:
            raise ValueError(
                f"{partition_column} must contain the contiguous labels {expected_labels}."
            )
        partitions = [
            (label, data.index[data[partition_column] == label].to_numpy()) for label in labels
        ]
    return data, tuple(features), n_batches, partitions


# Calculate the local prior scale for one partition.
def _local_prior_scale(
    config: BayesianLogisticConfig,
    partition_column: str | None,
    partition_size: int,
    total_size: int,
) -> float:
    if partition_column is None:
        return config.prior_scale
    return config.prior_scale / np.sqrt(partition_size / total_size)


# Build the canonical per-batch runtime payload.
def _batch_runtime_payload(
    batch_number: int,
    n_observations: int,
    local_prior_scale: float,
    sampler_runtime: float,
    end_to_end_runtime: float,
    config: BayesianLogisticConfig,
    per_chain_runtimes: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "batch": batch_number,
        "n_observations": n_observations,
        "local_prior_scale": local_prior_scale,
        "local_prior_variance": local_prior_scale**2,
        "sampler_runtime_seconds": sampler_runtime,
        "end_to_end_runtime_seconds": end_to_end_runtime,
        "chains": config.chains,
        "parallel_chains": config.parallel_chains if per_chain_runtimes is None else False,
        "chain_execution": "parallel"
        if config.parallel_chains and per_chain_runtimes is None
        else "sequential" if per_chain_runtimes is None else "sequential_processes",
        "per_chain_sampler_runtime_seconds": per_chain_runtimes,
        "serial_sampler_runtime_seconds": sampler_runtime
        if not config.parallel_chains or per_chain_runtimes is not None
        else None,
    }


# Build settings that determine checkpoint compatibility.
def _sampler_settings(config: BayesianLogisticConfig) -> dict[str, Any]:
    return {
        "chains": config.chains,
        "warmup": config.warmup,
        "post_warmup": config.post_warmup,
        "prior_scale": config.prior_scale,
        "seed": config.seed,
        "dense_mass": config.dense_mass,
        "initialization": "map" if config.map_initialization else "default",
        "map_max_iterations": config.map_max_iterations,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


# Build the configuration metadata for batched sampling.
def _config_payload(
    input_path: Path,
    n_observations: int,
    n_batches: int,
    features: tuple[str, ...],
    response_column: str,
    metadata_columns: tuple[str, ...],
    partition_column: str | None,
    config: BayesianLogisticConfig,
    local_prior_scales: dict[int, float],
) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "n_observations": n_observations,
        "n_batches": n_batches,
        "feature_names": list(features),
        "response_column": response_column,
        "metadata_columns": list(metadata_columns),
        "partition_column": partition_column,
        "prior_scale": config.prior_scale,
        "local_prior_scales": local_prior_scales,
        "chains": config.chains,
        "warmup": config.warmup,
        "post_warmup": config.post_warmup,
        "seed": config.seed,
        "dense_mass": config.dense_mass,
        "initialization": "map" if config.map_initialization else "default",
        "map_max_iterations": config.map_max_iterations,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "sampler_settings": _sampler_settings(config),
        "sampler": "numpyro_nuts",
        "run_status": "completed",
    }


# Check whether saved metadata matches the requested sampling settings.
def _checkpoint_settings_match(
    metadata: dict[str, Any],
    config: BayesianLogisticConfig,
) -> bool:
    saved_settings = metadata.get("sampler_settings")
    if saved_settings is None:
        return (
            not config.dense_mass
            and not config.map_initialization
            and not bool(jax.config.jax_enable_x64)
        )
    return saved_settings == _sampler_settings(config)


# Check whether a completed result matches the requested sampling settings.
def _completed_settings_match(
    batch_dir: Path,
    config: BayesianLogisticConfig,
) -> bool:
    config_path = batch_dir / "config.json"
    if not config_path.is_file():
        return (
            not config.dense_mass
            and not config.map_initialization
            and not bool(jax.config.jax_enable_x64)
        )
    payload = json.loads(config_path.read_text())
    saved_settings = payload.get("sampler_settings")
    if saved_settings is None:
        return (
            not config.dense_mass
            and not config.map_initialization
            and not bool(jax.config.jax_enable_x64)
        )
    return saved_settings == _sampler_settings(config)


# Run the original batched workflow while saving draws before diagnostics.
def run(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    n_batches: int | None,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None = None,
    partition_column: str | None = None,
    batch_index: int | None = None,
) -> None:
    started_total = time.perf_counter()
    data, features, n_batches, partitions = _prepare_partitions(
        input_path,
        n_batches,
        response_column,
        metadata_columns,
        feature_columns,
        partition_column,
    )
    is_batch_worker = batch_index is not None
    if is_batch_worker:
        if batch_index not in {label for label, _ in partitions}:
            raise ValueError("batch_index is outside the processed batch range.")
        partitions = [(label, indices) for label, indices in partitions if label == batch_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    local_prior_scales = {
        label: _local_prior_scale(config, partition_column, len(indices), len(data))
        for label, indices in partitions
    }
    for batch_number, indices in partitions:
        batch_dir = output_dir / f"batch_{batch_number}"
        batch_config = replace(config, prior_scale=local_prior_scales[batch_number])
        batch_started = time.perf_counter()
        design = data.iloc[indices][list(features)].to_numpy()
        response = data.iloc[indices][response_column].to_numpy()
        result = fit(design, response, batch_config, features)
        atomic_savez(batch_dir / "hmc_samples.npz", result.draws)
        add_diagnostics(result)
        atomic_write_json(batch_dir / "diagnostics.json", diagnostics_payload(result))
        summary = _batch_runtime_payload(
            batch_number,
            len(indices),
            batch_config.prior_scale,
            result.runtime_seconds,
            time.perf_counter() - batch_started,
            config,
        )
        summaries.append(summary)
        atomic_write_json(batch_dir / "runtime.json", summary)
    config_payload = _config_payload(
        input_path,
        len(data),
        n_batches,
        features,
        response_column,
        metadata_columns,
        partition_column,
        config,
        local_prior_scales,
    )
    if not is_batch_worker:
        atomic_write_json(output_dir / "config.json", config_payload)
        aggregate_runtime = {
            "batch_execution": "serial",
            "n_batches": n_batches,
            "serial_sampler_runtime_seconds": sum(
                summary["sampler_runtime_seconds"] for summary in summaries
            ),
            "serial_end_to_end_runtime_seconds": sum(
                summary["end_to_end_runtime_seconds"] for summary in summaries
            ),
            "wall_clock_seconds": time.perf_counter() - started_total,
        }
        atomic_write_json(output_dir / "runtime.json", aggregate_runtime)
    else:
        batch_number = partitions[0][0]
        config_payload["batch"] = batch_number
        config_payload["n_observations"] = len(partitions[0][1])
        config_payload["local_prior_scales"] = {
            str(batch_number): local_prior_scales[batch_number]
        }
        atomic_write_json(output_dir / f"batch_{batch_number}" / "config.json", config_payload)


# Run and atomically checkpoint one chain for one batch.
def run_batch_chain(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    n_batches: int | None,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
    partition_column: str | None,
    batch_index: int,
    chain_index: int,
) -> None:
    if not 1 <= chain_index <= config.chains:
        raise ValueError("chain_index is outside the configured chain range.")
    batch_dir = output_dir / f"batch_{batch_index}"
    expected_parameters = len(feature_columns) if feature_columns is not None else None
    if expected_parameters is not None and canonical_result_is_complete(
        batch_dir,
        config.chains,
        config.post_warmup,
        expected_parameters,
    ):
        if _completed_settings_match(batch_dir, config):
            return
        raise ValueError(f"Completed batch {batch_index} uses incompatible settings.")
    checkpoint_path = chain_checkpoint_path(batch_dir, chain_index)
    runtime_path = chain_runtime_path(batch_dir, chain_index)
    if expected_parameters is not None and chain_checkpoint_is_valid(
        batch_dir,
        chain_index,
        config.post_warmup,
        expected_parameters,
    ):
        metadata = json.loads(runtime_path.read_text())
        if _checkpoint_settings_match(metadata, config):
            return
        raise ValueError(
            f"Checkpoint for batch {batch_index}, chain {chain_index} "
            "uses incompatible settings."
        )
    if checkpoint_path.exists() or runtime_path.exists():
        raise ValueError(
            f"Invalid existing checkpoint for batch {batch_index}, chain {chain_index}."
        )
    atomic_write_json(
        batch_dir / "status.json",
        {
            "run_status": "in_progress",
            "phase": "sampling",
            "batch": batch_index,
            "chain": chain_index,
        },
    )
    started_total = time.perf_counter()
    data, features, n_batches, partitions = _prepare_partitions(
        input_path,
        n_batches,
        response_column,
        metadata_columns,
        feature_columns,
        partition_column,
    )
    partition_map = dict(partitions)
    if batch_index not in partition_map:
        raise ValueError("batch_index is outside the processed batch range.")
    indices = partition_map[batch_index]
    local_scale = _local_prior_scale(config, partition_column, len(indices), len(data))
    chain_config = replace(
        config,
        prior_scale=local_scale,
        chains=1,
        seed=config.seed + chain_index - 1,
        parallel_chains=False,
    )
    design = data.iloc[indices][list(features)].to_numpy()
    response = data.iloc[indices][response_column].to_numpy()
    result = fit(design, response, chain_config, features)
    atomic_savez(checkpoint_path, result.draws)
    atomic_write_json(
        runtime_path,
        {
            "batch": batch_index,
            "chain": chain_index,
            "seed": chain_config.seed,
            "sampler_runtime_seconds": result.runtime_seconds,
            "end_to_end_runtime_seconds": time.perf_counter() - started_total,
            "n_observations": len(indices),
            "total_n_observations": len(data),
            "n_batches": n_batches,
            "local_prior_scale": local_scale,
            "feature_names": list(features),
            "backend": result.diagnostics["backend"],
            "sampler_settings": _sampler_settings(config),
        },
    )
    atomic_write_json(
        batch_dir / "status.json",
        {
            "run_status": "checkpointed",
            "phase": "sampling",
            "batch": batch_index,
            "chain": chain_index,
        },
    )


# Load and validate metadata from every chain in one batch.
def _load_chain_metadata(batch_dir: Path, chains: int) -> list[dict[str, Any]]:
    metadata = [
        json.loads(chain_runtime_path(batch_dir, chain_index).read_text())
        for chain_index in range(1, chains + 1)
    ]
    keys = (
        "batch",
        "n_observations",
        "total_n_observations",
        "n_batches",
        "local_prior_scale",
        "feature_names",
    )
    if any(any(item[key] != metadata[0][key] for key in keys) for item in metadata[1:]):
        raise ValueError("Chain checkpoint metadata is inconsistent.")
    saved_settings = [item.get("sampler_settings") for item in metadata]
    if all(settings is not None for settings in saved_settings) and any(
        settings != saved_settings[0] for settings in saved_settings[1:]
    ):
        raise ValueError("Chain checkpoint sampler settings are inconsistent.")
    return metadata


# Finalize one batch from its saved chains in an isolated process.
def finalize_batch(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
    partition_column: str | None,
    batch_index: int,
) -> None:
    if feature_columns is None:
        raise ValueError("feature_columns is required when finalizing checkpoints.")
    batch_dir = output_dir / f"batch_{batch_index}"
    if canonical_result_is_complete(
        batch_dir,
        config.chains,
        config.post_warmup,
        len(feature_columns),
    ):
        return
    atomic_write_json(
        batch_dir / "status.json",
        {"run_status": "in_progress", "phase": "finalizing", "batch": batch_index},
    )
    chain_metadata = _load_chain_metadata(batch_dir, config.chains)
    if not all(_checkpoint_settings_match(item, config) for item in chain_metadata):
        raise ValueError("Chain checkpoints use incompatible sampler settings.")
    draws = combine_chain_checkpoints(
        batch_dir,
        config.chains,
        config.post_warmup,
        len(feature_columns),
    )
    sampler_runtimes = [float(item["sampler_runtime_seconds"]) for item in chain_metadata]
    end_to_end_runtimes = [float(item["end_to_end_runtime_seconds"]) for item in chain_metadata]
    result = BayesianLogisticResult(
        draws=draws,
        runtime_seconds=sum(sampler_runtimes),
        diagnostics={
            "num_chains": config.chains,
            "num_warmup": config.warmup,
            "num_samples": config.post_warmup,
            "backend": chain_metadata[0]["backend"],
            "parallel_chains": False,
            "dense_mass": config.dense_mass,
            "initialization": "map" if config.map_initialization else "default",
            "map_max_iterations": config.map_max_iterations,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        feature_names=feature_columns,
    )
    atomic_savez(batch_dir / "hmc_samples.npz", draws)
    add_diagnostics(result)
    atomic_write_json(batch_dir / "diagnostics.json", diagnostics_payload(result))
    local_scale = float(chain_metadata[0]["local_prior_scale"])
    runtime_payload = _batch_runtime_payload(
        batch_index,
        int(chain_metadata[0]["n_observations"]),
        local_scale,
        sum(sampler_runtimes),
        sum(end_to_end_runtimes),
        config,
        sampler_runtimes,
    )
    atomic_write_json(batch_dir / "runtime.json", runtime_payload)
    config_payload = _config_payload(
        input_path,
        int(chain_metadata[0]["n_observations"]),
        int(chain_metadata[0]["n_batches"]),
        feature_columns,
        response_column,
        metadata_columns,
        partition_column,
        config,
        {batch_index: local_scale},
    )
    config_payload["batch"] = batch_index
    atomic_write_json(batch_dir / "config.json", config_payload)
    atomic_write_json(
        batch_dir / "status.json",
        {"run_status": "completed", "phase": "finalized", "batch": batch_index},
    )


# Aggregate completed batch metadata without loading the source dataset.
def aggregate_batches(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    n_batches: int | None,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
    partition_column: str | None,
) -> None:
    if n_batches is None or n_batches < 1:
        raise ValueError("--batches is required for aggregate-only mode.")
    if feature_columns is None:
        raise ValueError("feature_columns is required for aggregate-only mode.")
    summaries = []
    for batch_index in range(1, n_batches + 1):
        batch_dir = output_dir / f"batch_{batch_index}"
        if not canonical_result_is_complete(
            batch_dir,
            config.chains,
            config.post_warmup,
            len(feature_columns),
        ):
            raise ValueError(f"Batch {batch_index} is not complete.")
        summaries.append(json.loads((batch_dir / "runtime.json").read_text()))
    local_scales = {
        int(summary["batch"]): float(summary["local_prior_scale"]) for summary in summaries
    }
    atomic_write_json(
        output_dir / "config.json",
        _config_payload(
            input_path,
            sum(int(summary["n_observations"]) for summary in summaries),
            n_batches,
            feature_columns,
            response_column,
            metadata_columns,
            partition_column,
            config,
            local_scales,
        ),
    )
    aggregate_runtime = {
        "batch_execution": "serial_processes",
        "n_batches": n_batches,
        "serial_sampler_runtime_seconds": sum(
            float(summary["sampler_runtime_seconds"]) for summary in summaries
        ),
        "serial_end_to_end_runtime_seconds": sum(
            float(summary["end_to_end_runtime_seconds"]) for summary in summaries
        ),
        "wall_clock_seconds": sum(
            float(summary["end_to_end_runtime_seconds"]) for summary in summaries
        ),
    }
    atomic_write_json(output_dir / "runtime.json", aggregate_runtime)
    atomic_write_json(
        output_dir / "status.json",
        {"run_status": "completed", "phase": "aggregated"},
    )


# Parse command-line arguments and dispatch the requested run phase.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--response-column", default="outcome")
    parser.add_argument("--metadata-columns", default="center,id")
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--partition-column", default=None)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--sequential-chains", action="store_true")
    parser.add_argument("--dense-mass", action="store_true")
    parser.add_argument("--map-initialization", action="store_true")
    parser.add_argument("--map-max-iterations", type=int, default=500)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--chain-index", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.chain_index is not None and args.batch_index is None:
        parser.error("--chain-index requires --batch-index.")
    if args.finalize_only and args.batch_index is None:
        parser.error("--finalize-only requires --batch-index.")
    if sum((args.finalize_only, args.aggregate_only, args.chain_index is not None)) > 1:
        parser.error("Worker modes cannot be combined.")
    features = (
        tuple(c for c in args.feature_columns.split(",") if c) if args.feature_columns else None
    )
    metadata_columns = tuple(c for c in args.metadata_columns.split(",") if c)
    config = BayesianLogisticConfig(
        prior_scale=args.prior_scale,
        chains=args.chains,
        warmup=args.warmup,
        post_warmup=args.samples,
        seed=args.seed,
        parallel_chains=not args.sequential_chains,
        dense_mass=args.dense_mass,
        map_initialization=args.map_initialization,
        map_max_iterations=args.map_max_iterations,
    )
    try:
        if args.aggregate_only:
            aggregate_batches(
                args.input,
                args.output_dir,
                config,
                args.batches,
                args.response_column,
                metadata_columns,
                features,
                args.partition_column,
            )
        elif args.finalize_only:
            finalize_batch(
                args.input,
                args.output_dir,
                config,
                args.response_column,
                metadata_columns,
                features,
                args.partition_column,
                args.batch_index,
            )
        elif args.chain_index is not None:
            run_batch_chain(
                args.input,
                args.output_dir,
                config,
                args.batches,
                args.response_column,
                metadata_columns,
                features,
                args.partition_column,
                args.batch_index,
                args.chain_index,
            )
        else:
            run(
                args.input,
                args.output_dir,
                config,
                args.batches,
                args.response_column,
                metadata_columns,
                features,
                args.partition_column,
                args.batch_index,
            )
    except Exception as error:
        if args.finalize_only or args.chain_index is not None:
            batch_dir = args.output_dir / f"batch_{args.batch_index}"
            atomic_write_json(
                batch_dir / "status.json",
                {
                    "run_status": "failed",
                    "phase": "finalizing" if args.finalize_only else "sampling",
                    "batch": args.batch_index,
                    "chain": args.chain_index,
                    "error_type": type(error).__name__,
                },
            )
        raise


if __name__ == "__main__":
    main()
