"""Reproduce HSP90 tail and KDE-product diagnostics from saved outputs."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import json
from pathlib import Path

# Third-party libraries
import numpy as np
from scipy.special import expit, logsumexp

####################
# Diagnostic setup #
####################

THETA_UPPER_BOUND = 19.0
QUANTILE_PROBABILITIES = (0, 0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999, 1)
BANDWIDTH_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)
KDE_CHUNK_SIZE = 1024

# Parse a finite floating-point value from the command line.
def _finite_float(value):
    parsed = float(value)
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed

# Parse a positive integer from the command line.
def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

# Parse the HSP90 diagnostics command-line arguments.
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sbi_draws", type=Path, help="NPZ output from run_hsp90_sbi.py.")
    parser.add_argument(
        "pmc_mlp_moments",
        type=Path,
        help="PMC MLP moments JSON output from run_hsp90_pmc.py.",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grid-min", type=_finite_float, default=-1.0)
    parser.add_argument("--grid-max", type=_finite_float, default=2.0)
    parser.add_argument("--grid-points", type=_positive_int, default=12001)
    return parser.parse_args()

# Validate a finite real KxS draw array.
def _validate_draw_array(values, name, expected_shape=None):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional KxS array")
    if expected_shape is not None and values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if 0 in values.shape:
        raise ValueError(f"{name} must have positive K and S dimensions")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values.astype(float, copy=False)

# Load and validate the saved SBI posterior draws.
def _load_draws(path):
    with np.load(path, allow_pickle=False) as saved:
        if "draws_z" not in saved:
            raise ValueError(f"{path} must contain draws_z")
        draws_z = _validate_draw_array(saved["draws_z"], "draws_z")
        if "draws_theta" in saved:
            draws_theta = _validate_draw_array(
                saved["draws_theta"],
                "draws_theta",
                expected_shape=draws_z.shape,
            )
        else:
            draws_theta = THETA_UPPER_BOUND * expit(draws_z)
    return draws_z, draws_theta

# Validate one finite numeric JSON moment.
def _finite_json_moment(summary, name, allow_null=False):
    if name not in summary:
        raise ValueError(f"PMC MLP moments JSON must contain {name}")
    value = summary[name]
    if allow_null and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)

# Load and validate the saved PMC MLP moments.
def _load_mlp_moments(path):
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict):
        raise ValueError("PMC MLP moments JSON must contain an object")

    moments = {
        "mean_z": _finite_json_moment(summary, "mean_z"),
        "variance_z": _finite_json_moment(summary, "variance_z"),
        "sd_z": _finite_json_moment(summary, "sd_z", allow_null=True),
    }
    if "second_moment_z" in summary:
        moments["second_moment_z"] = _finite_json_moment(
            summary,
            "second_moment_z",
        )
    if moments["variance_z"] >= 0 and moments["sd_z"] is None:
        raise ValueError("sd_z must be finite when variance_z is nonnegative")
    return moments

# Summarize pooled quantiles with stable probability labels.
def _pooled_quantiles(values):
    quantiles = np.quantile(values, QUANTILE_PROBABILITIES)
    return {
        format(probability, "g"): float(quantile)
        for probability, quantile in zip(QUANTILE_PROBABILITIES, quantiles, strict=True)
    }

# Summarize one threshold count and fraction.
def _threshold_summary(mask):
    count = int(np.count_nonzero(mask))
    return {"count": count, "fraction": count / mask.size}

# Compute per-image central-moment diagnostics in z-space.
def _per_image_diagnostics(draws_z):
    diagnostics = []
    for image_index, samples in enumerate(draws_z, start=1):
        mean_z = float(np.mean(samples))
        centered = samples - mean_z
        sd_z = float(np.sqrt(np.mean(centered**2)))
        if sd_z == 0:
            skewness_z = None
            excess_kurtosis_z = None
        else:
            standardized = centered / sd_z
            skewness_z = float(np.mean(standardized**3))
            excess_kurtosis_z = float(np.mean(standardized**4) - 3)
        diagnostics.append(
            {
                "image": image_index,
                "mean_z": mean_z,
                "median_z": float(np.median(samples)),
                "sd_z": sd_z,
                "skewness_z": skewness_z,
                "excess_kurtosis_z": excess_kurtosis_z,
            }
        )
    return diagnostics

# Build diagnostics that describe only the final saved draws.
def _build_tail_diagnostics(draws_z, draws_theta):
    num_images, num_samples = draws_z.shape
    return {
        "draws_scope": "final saved SBI draws only",
        "K": num_images,
        "S": num_samples,
        "pooled_z_quantiles": _pooled_quantiles(draws_z),
        "pooled_theta_quantiles": _pooled_quantiles(draws_theta),
        "threshold_diagnostics": {
            "z_le_-5": _threshold_summary(draws_z <= -5),
            "z_ge_5": _threshold_summary(draws_z >= 5),
            "theta_le_0.1": _threshold_summary(draws_theta <= 0.1),
            "theta_ge_18.9": _threshold_summary(draws_theta >= 18.9),
        },
        "per_image_z": _per_image_diagnostics(draws_z),
        "per_image_sd_convention": "Population standard deviation",
        "per_image_shape_convention": "Population standardized central moments",
    }

# Evaluate one Gaussian KDE in log space using a Scott bandwidth multiplier.
def _log_kde(grid, samples, multiplier):
    num_samples = samples.size
    if num_samples < 2:
        raise ValueError("each image must have at least two draws for KDE estimation")
    sample_sd = float(np.std(samples, ddof=1))
    if not np.isfinite(sample_sd) or sample_sd <= 0:
        raise ValueError("each image must have positive finite draw variance for KDE estimation")

    bandwidth = multiplier * num_samples ** (-1 / 5) * sample_sd
    log_density = np.empty(grid.size)
    log_scale = np.log(num_samples) + np.log(bandwidth) + 0.5 * np.log(2 * np.pi)
    for start in range(0, grid.size, KDE_CHUNK_SIZE):
        stop = min(start + KDE_CHUNK_SIZE, grid.size)
        standardized = (grid[start:stop, None] - samples[None, :]) / bandwidth
        log_density[start:stop] = logsumexp(-0.5 * standardized**2, axis=1) - log_scale
    return log_density

# Normalize one product density and calculate its z-space moments.
def _kde_product_moments(draws_z, grid, multiplier):
    log_product = np.zeros(grid.size)
    for samples in draws_z:
        log_product += _log_kde(grid, samples, multiplier)

    step = grid[1] - grid[0]
    trapezoid_weights = np.ones(grid.size)
    trapezoid_weights[[0, -1]] = 0.5
    log_normalizer = logsumexp(log_product + np.log(trapezoid_weights)) + np.log(step)
    density = np.exp(log_product - log_normalizer)
    mean_z = float(step * np.sum(trapezoid_weights * grid * density))
    second_moment_z = float(step * np.sum(trapezoid_weights * grid**2 * density))
    variance_z = second_moment_z - mean_z**2
    if variance_z < 0 and np.isclose(variance_z, 0):
        variance_z = 0.0
    if variance_z < 0:
        raise ValueError("KDE product produced a negative variance")
    return {
        "bandwidth_multiplier": multiplier,
        "mean_z": mean_z,
        "second_moment_z": second_moment_z,
        "variance_z": variance_z,
        "sd_z": float(np.sqrt(variance_z)),
    }

# Build the direct KDE-product comparison summary.
def _build_kde_product_check(draws_z, mlp_moments, grid_min, grid_max, grid_points):
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    if grid_min >= grid_max:
        raise ValueError("grid_min must be less than grid_max")

    grid = np.linspace(grid_min, grid_max, grid_points)
    kde_results = [
        _kde_product_moments(draws_z, grid, multiplier)
        for multiplier in BANDWIDTH_MULTIPLIERS
    ]
    return {
        "grid": {"min_z": grid_min, "max_z": grid_max, "points": grid_points},
        "kde_product_moments": kde_results,
        "pmc_mlp_moments": mlp_moments,
    }

# Save one JSON diagnostic with stable formatting.
def _save_json(path, contents):
    path.write_text(json.dumps(contents, indent=2) + "\n")

#######################
# Diagnostic workflow #
#######################

# Reproduce HSP90 diagnostics from existing SBI and PMC outputs.
def main():
    args = _parse_args()

    print("Loading saved HSP90 outputs...")
    draws_z, draws_theta = _load_draws(args.sbi_draws)
    mlp_moments = _load_mlp_moments(args.pmc_mlp_moments)

    print("Computing final-draw tail diagnostics...")
    tail_diagnostics = _build_tail_diagnostics(draws_z, draws_theta)

    print("Computing direct Gaussian KDE product checks...")
    kde_product_check = _build_kde_product_check(
        draws_z,
        mlp_moments,
        args.grid_min,
        args.grid_max,
        args.grid_points,
    )

    print("Saving diagnostics...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(args.output_dir / "tail_diagnostics.json", tail_diagnostics)
    _save_json(args.output_dir / "kde_product_check.json", kde_product_check)
    print("HSP90 diagnostics complete.")

if __name__ == "__main__":
    main()
