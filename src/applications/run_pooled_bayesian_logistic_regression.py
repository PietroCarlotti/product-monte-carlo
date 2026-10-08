"""Run the shared Bayesian logistic model on the respiratory trial data."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
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


# Load and validate the dataset columns used by the model.
def _load_model_data(
    input_path: Path,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, object, object, tuple[str, ...]]:
    data = pd.read_csv(input_path)
    required = set(metadata_columns + (response_column,))
    if not required.issubset(data.columns):
        raise ValueError(
            f"Input is missing required columns: {sorted(required.difference(data.columns))}"
        )
    feature_names = feature_columns or tuple(c for c in data.columns if c not in required)
    if not set(feature_names).issubset(data.columns):
        raise ValueError("Input is missing one or more feature columns.")
    design = data[list(feature_names)].to_numpy()
    response = data[response_column].to_numpy()
    return data, design, response, tuple(feature_names)


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


# Build the metadata shared by every final posterior result.
def _config_payload(
    input_path: Path,
    n_observations: int,
    feature_names: tuple[str, ...],
    response_column: str,
    metadata_columns: tuple[str, ...],
    config: BayesianLogisticConfig,
) -> dict[str, object]:
    return {
        "input": str(input_path),
        "n_observations": n_observations,
        "feature_names": list(feature_names),
        "response_column": response_column,
        "metadata_columns": list(metadata_columns),
        "config": config.__dict__,
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
    output_dir: Path,
    config: BayesianLogisticConfig,
) -> bool:
    config_path = output_dir / "config.json"
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


# Run the original all-chains workflow while saving draws before diagnostics.
def run(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    response_column: str = "outcome",
    metadata_columns: tuple[str, ...] = ("center", "id"),
    feature_columns: tuple[str, ...] | None = None,
) -> None:
    started_total = time.perf_counter()
    data, design, response, feature_names = _load_model_data(
        input_path,
        response_column,
        metadata_columns,
        feature_columns,
    )
    result = fit(design, response, config, feature_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(output_dir / "hmc_samples.npz", result.draws)
    add_diagnostics(result)
    atomic_write_json(output_dir / "diagnostics.json", diagnostics_payload(result))
    atomic_write_json(
        output_dir / "config.json",
        _config_payload(
            input_path,
            len(data),
            result.feature_names,
            response_column,
            metadata_columns,
            config,
        ),
    )
    runtime_payload = {
        "sampler_runtime_seconds": result.runtime_seconds,
        "end_to_end_runtime_seconds": time.perf_counter() - started_total,
        "chains": config.chains,
        "parallel_chains": config.parallel_chains,
        "chain_execution": "parallel" if config.parallel_chains else "sequential",
        "per_chain_sampler_runtime_seconds": None,
        "serial_sampler_runtime_seconds": result.runtime_seconds
        if not config.parallel_chains
        else None,
        "wall_clock_seconds": time.perf_counter() - started_total,
    }
    atomic_write_json(output_dir / "runtime.json", runtime_payload)


# Run and atomically checkpoint one chain for restartable sampling.
def run_chain(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
    chain_index: int,
) -> None:
    if not 1 <= chain_index <= config.chains:
        raise ValueError("chain_index is outside the configured chain range.")
    expected_parameters = len(feature_columns) if feature_columns is not None else None
    if expected_parameters is not None and canonical_result_is_complete(
        output_dir,
        config.chains,
        config.post_warmup,
        expected_parameters,
    ):
        if _completed_settings_match(output_dir, config):
            return
        raise ValueError("Completed pooled result uses incompatible settings.")
    checkpoint_path = chain_checkpoint_path(output_dir, chain_index)
    runtime_path = chain_runtime_path(output_dir, chain_index)
    if expected_parameters is not None and chain_checkpoint_is_valid(
        output_dir,
        chain_index,
        config.post_warmup,
        expected_parameters,
    ):
        metadata = json.loads(runtime_path.read_text())
        if _checkpoint_settings_match(metadata, config):
            return
        raise ValueError(
            f"Checkpoint for pooled chain {chain_index} uses incompatible settings."
        )
    if checkpoint_path.exists() or runtime_path.exists():
        raise ValueError(f"Invalid existing checkpoint for chain {chain_index}.")
    atomic_write_json(
        output_dir / "status.json",
        {"run_status": "in_progress", "phase": "sampling", "chain": chain_index},
    )
    started_total = time.perf_counter()
    data, design, response, feature_names = _load_model_data(
        input_path,
        response_column,
        metadata_columns,
        feature_columns,
    )
    chain_config = replace(
        config,
        chains=1,
        seed=config.seed + chain_index - 1,
        parallel_chains=False,
    )
    result = fit(design, response, chain_config, feature_names)
    atomic_savez(checkpoint_path, result.draws)
    atomic_write_json(
        runtime_path,
        {
            "chain": chain_index,
            "seed": chain_config.seed,
            "sampler_runtime_seconds": result.runtime_seconds,
            "end_to_end_runtime_seconds": time.perf_counter() - started_total,
            "n_observations": len(data),
            "feature_names": list(feature_names),
            "backend": result.diagnostics["backend"],
            "sampler_settings": _sampler_settings(config),
        },
    )
    atomic_write_json(
        output_dir / "status.json",
        {"run_status": "checkpointed", "phase": "sampling", "chain": chain_index},
    )


# Load consistent metadata from every completed chain.
def _load_chain_metadata(output_dir: Path, chains: int) -> list[dict[str, object]]:
    metadata = [
        json.loads(chain_runtime_path(output_dir, chain_index).read_text())
        for chain_index in range(1, chains + 1)
    ]
    reference_names = metadata[0]["feature_names"]
    reference_observations = metadata[0]["n_observations"]
    if any(
        item["feature_names"] != reference_names
        or item["n_observations"] != reference_observations
        for item in metadata[1:]
    ):
        raise ValueError("Chain checkpoint metadata is inconsistent.")
    saved_settings = [item.get("sampler_settings") for item in metadata]
    if all(settings is not None for settings in saved_settings) and any(
        settings != saved_settings[0] for settings in saved_settings[1:]
    ):
        raise ValueError("Chain checkpoint sampler settings are inconsistent.")
    return metadata


# Finalize saved chains and compute diagnostics in an isolated process.
def finalize_chains(
    input_path: Path,
    output_dir: Path,
    config: BayesianLogisticConfig,
    response_column: str,
    metadata_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
) -> None:
    if feature_columns is None:
        raise ValueError("feature_columns is required when finalizing checkpoints.")
    if canonical_result_is_complete(
        output_dir,
        config.chains,
        config.post_warmup,
        len(feature_columns),
    ):
        if _completed_settings_match(output_dir, config):
            return
        raise ValueError("Completed pooled result uses incompatible settings.")
    atomic_write_json(
        output_dir / "status.json",
        {"run_status": "in_progress", "phase": "finalizing"},
    )
    chain_metadata = _load_chain_metadata(output_dir, config.chains)
    if not all(_checkpoint_settings_match(item, config) for item in chain_metadata):
        raise ValueError("Chain checkpoints use incompatible sampler settings.")
    draws = combine_chain_checkpoints(
        output_dir,
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
    atomic_savez(output_dir / "hmc_samples.npz", draws)
    add_diagnostics(result)
    atomic_write_json(output_dir / "diagnostics.json", diagnostics_payload(result))
    atomic_write_json(
        output_dir / "config.json",
        _config_payload(
            input_path,
            int(chain_metadata[0]["n_observations"]),
            feature_columns,
            response_column,
            metadata_columns,
            config,
        ),
    )
    runtime_payload = {
        "sampler_runtime_seconds": sum(sampler_runtimes),
        "end_to_end_runtime_seconds": sum(end_to_end_runtimes),
        "chains": config.chains,
        "parallel_chains": False,
        "chain_execution": "sequential_processes",
        "per_chain_sampler_runtime_seconds": sampler_runtimes,
        "serial_sampler_runtime_seconds": sum(sampler_runtimes),
        "wall_clock_seconds": sum(end_to_end_runtimes),
    }
    atomic_write_json(output_dir / "runtime.json", runtime_payload)
    atomic_write_json(
        output_dir / "status.json",
        {"run_status": "completed", "phase": "finalized"},
    )


# Parse command-line arguments and dispatch the requested run phase.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/respiratory/respiratory_processed.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/respiratory/oracle"))
    parser.add_argument("--response-column", default="outcome")
    parser.add_argument("--metadata-columns", default="center,id")
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--sequential-chains", action="store_true")
    parser.add_argument("--dense-mass", action="store_true")
    parser.add_argument("--map-initialization", action="store_true")
    parser.add_argument("--map-max-iterations", type=int, default=500)
    parser.add_argument("--chain-index", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only and args.chain_index is not None:
        parser.error("--finalize-only and --chain-index cannot be combined.")
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
    metadata_columns = tuple(c for c in args.metadata_columns.split(",") if c)
    feature_columns = (
        tuple(c for c in args.feature_columns.split(",") if c) if args.feature_columns else None
    )
    try:
        if args.finalize_only:
            finalize_chains(
                args.input,
                args.output_dir,
                config,
                args.response_column,
                metadata_columns,
                feature_columns,
            )
        elif args.chain_index is not None:
            run_chain(
                args.input,
                args.output_dir,
                config,
                args.response_column,
                metadata_columns,
                feature_columns,
                args.chain_index,
            )
        else:
            run(
                args.input,
                args.output_dir,
                config,
                args.response_column,
                metadata_columns,
                feature_columns,
            )
    except Exception as error:
        if args.finalize_only or args.chain_index is not None:
            atomic_write_json(
                args.output_dir / "status.json",
                {
                    "run_status": "failed",
                    "phase": "finalizing" if args.finalize_only else "sampling",
                    "chain": args.chain_index,
                    "error_type": type(error).__name__,
                },
            )
        raise


if __name__ == "__main__":
    main()
