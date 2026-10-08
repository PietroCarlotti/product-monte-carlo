"""Fit PMC OLS, MLP, and GP moment estimators to saved HSP90 posterior draws."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import json
from pathlib import Path
from time import perf_counter

# Third-party libraries
import numpy as np
import torch

# PMC library
from pmc.algorithms import MLPPMC, LinearPMC, consensus_discrepancy
from pmc.pmc_gp import GaussianProcessPMC, recommended_pmc_gp_config

####################
# Experiment setup #
####################

HIDDEN_FEATURES = (32, 32)
SAMPLING_METHOD = "iid"
DISCREPANCY_METHOD = "consecutive"
TARGET_TRANSFORMS = (("identity", None), ("square", np.square))

# Parse a positive integer from the command line.
def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

# Parse a positive floating-point value from the command line.
def _positive_float(value):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed

# Parse the HSP90 PMC command-line arguments.
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draws", type=Path, help="NPZ output from run_hsp90_sbi.py.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=_positive_int, default=1000)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--ensemble-size", type=_positive_int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--identity-seed", type=int, default=94011)
    parser.add_argument("--square-seed", type=int, default=94012)
    return parser.parse_args()

# Load and validate node-by-sample posterior draws.
def _load_draws(path):
    with np.load(path, allow_pickle=False) as saved:
        if "draws_z" not in saved:
            raise ValueError(f"{path} must contain draws_z")
        draws_z = np.asarray(saved["draws_z"])

    if draws_z.ndim != 2:
        raise ValueError("draws_z must be a two-dimensional KxS array")
    if not (
        np.issubdtype(draws_z.dtype, np.integer)
        or np.issubdtype(draws_z.dtype, np.floating)
    ):
        raise ValueError("draws_z must contain real numeric values")
    if not np.isfinite(draws_z).all():
        raise ValueError("draws_z must contain only finite values")

    num_nodes, num_samples = draws_z.shape
    if num_nodes < 2:
        raise ValueError("draws_z must contain at least two nodes")
    if num_samples < 1:
        raise ValueError("draws_z must contain at least one sample")
    return draws_z, num_nodes, num_samples

# Wait for pending CUDA work so that wall-clock timings are accurate.
def _synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()

# Fit one PMC target and predict it at zero discrepancy with its runtime.
def _fit_target(estimator, consensus, discrepancy, seed, device):
    torch.manual_seed(seed)
    _synchronize(device)
    started = perf_counter()
    estimator.fit(consensus, discrepancy)
    prediction = estimator.predict(np.zeros(discrepancy.shape[1]))
    _synchronize(device)
    runtime = perf_counter() - started
    return float(np.asarray(prediction).squeeze()), runtime

# Fit one PMC-GP target at zero discrepancy with its runtime.
def _fit_gp_target(target, transform, consensus, discrepancy, seed, device):
    _synchronize(device)
    started = perf_counter()
    response = consensus if transform is None else transform(consensus)
    estimator = GaussianProcessPMC(
        recommended_pmc_gp_config(target, device=device)
    )
    estimator.fit(discrepancy, response[:, 0], seed=seed)
    prediction = estimator.predict(np.zeros(discrepancy.shape[1])).estimate
    _synchronize(device)
    runtime = perf_counter() - started
    return float(np.asarray(prediction).squeeze()), runtime

# Fit identity and square targets for one PMC method.
def _fit_moments(fit_one, seeds):
    # Run one untimed identical fit so startup and allocator costs are excluded
    fit_one("identity", None, seeds["identity"])
    moments = {}
    runtimes = {}
    for target, transform in TARGET_TRANSFORMS:
        moments[target], runtimes[target] = fit_one(
            target, transform, seeds[target]
        )
    return moments["identity"], moments["square"], runtimes

# Build a serializable moment summary without altering the variance estimate.
def _build_summary(
    mean_z,
    second_moment_z,
    runtimes,
    num_nodes,
    num_samples,
    settings,
    seeds,
    device,
):
    variance_z = second_moment_z - mean_z**2
    sd_z = None if variance_z < 0 else float(np.sqrt(variance_z))
    return {
        "mean_z": mean_z,
        "second_moment_z": second_moment_z,
        "variance_z": variance_z,
        "sd_z": sd_z,
        "K": num_nodes,
        "S": num_samples,
        "settings": settings,
        "seeds": seeds,
        "runtime_seconds": runtimes,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "warmup": True,
    }

# Save one moment summary as formatted JSON.
def _save_summary(path, summary):
    path.write_text(json.dumps(summary, indent=2) + "\n")

##################
# PMC experiment #
##################

# Run the HSP90 PMC moment workflow and save its outputs.
def main():
    args = _parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    print("Loading HSP90 posterior draws...")
    draws_z, num_nodes, num_samples = _load_draws(args.draws)
    node_samples = draws_z.T[:, :, None]
    consensus, discrepancy = consensus_discrepancy(
        node_samples,
        sampling_method=SAMPLING_METHOD,
        discrepancy_method=DISCREPANCY_METHOD,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = {
        "identity": args.identity_seed,
        "square": args.square_seed,
    }
    shared_settings = {
        "sampling_method": SAMPLING_METHOD,
        "discrepancy_method": DISCREPANCY_METHOD,
    }

    print("Fitting PMC OLS moments...")
    ols_moments = _fit_moments(
        lambda target, transform, seed: _fit_target(
            LinearPMC(h=transform, degree=1), consensus, discrepancy, seed, "cpu"
        ),
        seeds,
    )
    ols_settings = {
        **shared_settings,
        "method": "PMC-OLS",
        "device": "cpu",
        "degree": 1,
    }
    ols_summary = _build_summary(
        *ols_moments,
        num_nodes,
        num_samples,
        ols_settings,
        seeds,
        "cpu",
    )
    _save_summary(args.output_dir / "pmc_ols_moments.json", ols_summary)

    print("Fitting PMC MLP moments...")
    batch_size = min(args.batch_size, num_samples)
    mlp_settings = {
        **shared_settings,
        "method": "PMC-MLP",
        "hidden_features": list(HIDDEN_FEATURES),
        "n_training_steps": args.steps,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": batch_size,
        "device": device,
        "use_cosine_schedule": True,
        "ensemble_size": args.ensemble_size,
        "input_degree": 1,
        "activation": "relu",
        "discrepancy_centering": "target",
        "regression_loss": "mse",
    }

    # Construct a fresh estimator for each transformed regression target.
    def make_mlp(transform):
        return MLPPMC(
            h=transform,
            hidden_features=HIDDEN_FEATURES,
            n_training_steps=args.steps,
            optimizer="Adam",
            learning_rate=args.learning_rate,
            batch_size=batch_size,
            device=device,
            use_cosine_schedule=True,
            ensemble_size=args.ensemble_size,
            input_degree=1,
            activation="relu",
            discrepancy_centering="target",
            regression_loss="mse",
        )

    mlp_moments = _fit_moments(
        lambda target, transform, seed: _fit_target(
            make_mlp(transform), consensus, discrepancy, seed, device
        ),
        seeds,
    )
    mlp_summary = _build_summary(
        *mlp_moments,
        num_nodes,
        num_samples,
        mlp_settings,
        seeds,
        device,
    )
    _save_summary(args.output_dir / "pmc_mlp_moments.json", mlp_summary)

    print("Fitting PMC GP moments...")
    gp_moments = _fit_moments(
        lambda target, transform, seed: _fit_gp_target(
            target, transform, consensus, discrepancy, seed, device
        ),
        seeds,
    )
    gp_settings = {
        **shared_settings,
        "method": "PMC-GP",
        "device": device,
        "kernels": {"identity": "matern_32", "square": "rbf"},
    }
    gp_summary = _build_summary(
        *gp_moments,
        num_nodes,
        num_samples,
        gp_settings,
        seeds,
        device,
    )
    _save_summary(args.output_dir / "pmc_gp_moments.json", gp_summary)
    print("HSP90 PMC stage complete.")

if __name__ == "__main__":
    main()
