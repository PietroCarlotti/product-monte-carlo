"""Persist and validate restartable Bayesian logistic HMC artifacts."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# Atomically persist posterior arrays.
def atomic_savez(path: Path, draws: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **draws)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


# Atomically persist JSON metadata.
def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


# Load and validate a posterior draw archive.
def load_valid_draws(
    path: Path,
    expected_chains: int,
    expected_samples: int,
    expected_parameters: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"beta"}:
            raise ValueError(f"Unexpected posterior variables in {path}.")
        beta = np.asarray(archive["beta"])
    expected_shape = (expected_chains, expected_samples, expected_parameters)
    if beta.shape != expected_shape:
        raise ValueError(f"Expected beta shape {expected_shape} in {path}, found {beta.shape}.")
    if not np.isfinite(beta).all():
        raise ValueError(f"Posterior draws in {path} must be finite.")
    return {"beta": beta}


# Check whether a canonical result is complete and valid.
def canonical_result_is_complete(
    output_dir: Path,
    expected_chains: int,
    expected_samples: int,
    expected_parameters: int,
) -> bool:
    required_paths = (
        output_dir / "hmc_samples.npz",
        output_dir / "diagnostics.json",
        output_dir / "runtime.json",
    )
    if not all(path.is_file() for path in required_paths):
        return False
    try:
        load_valid_draws(
            required_paths[0],
            expected_chains,
            expected_samples,
            expected_parameters,
        )
        for path in required_paths[1:]:
            json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


# Return the stable path for a chain checkpoint.
def chain_checkpoint_path(output_dir: Path, chain_index: int) -> Path:
    return output_dir / "checkpoints" / f"chain_{chain_index}.npz"


# Return the stable path for chain runtime metadata.
def chain_runtime_path(output_dir: Path, chain_index: int) -> Path:
    return output_dir / "checkpoints" / f"chain_{chain_index}_runtime.json"


# Check whether a chain checkpoint is valid without modifying it.
def chain_checkpoint_is_valid(
    output_dir: Path,
    chain_index: int,
    expected_samples: int,
    expected_parameters: int,
) -> bool:
    checkpoint_path = chain_checkpoint_path(output_dir, chain_index)
    runtime_path = chain_runtime_path(output_dir, chain_index)
    if not checkpoint_path.is_file() or not runtime_path.is_file():
        return False
    try:
        load_valid_draws(checkpoint_path, 1, expected_samples, expected_parameters)
        json.loads(runtime_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


# Combine validated single-chain checkpoints in chain order.
def combine_chain_checkpoints(
    output_dir: Path,
    chains: int,
    expected_samples: int,
    expected_parameters: int,
) -> dict[str, np.ndarray]:
    checkpoints = [
        load_valid_draws(
            chain_checkpoint_path(output_dir, chain_index),
            1,
            expected_samples,
            expected_parameters,
        )["beta"]
        for chain_index in range(1, chains + 1)
    ]
    return {"beta": np.concatenate(checkpoints, axis=0)}
