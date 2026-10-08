##############################
# Import necessary libraries #
##############################

# Standard libraries
import json
import os
from pathlib import Path

# Force headless rendering before Matplotlib can initialize a GUI backend.
os.environ["MPLBACKEND"] = "Agg"

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma

########################
# Experiment constants #
########################

SAMPLE_SIZE = 500
POSTERIOR_PREDICTIVE_DRAWS = 1000
VALIDATION_REPETITIONS = 100
VALIDATION_POSTERIOR_PREDICTIVE_DRAWS = 500
LACK_OF_FIT_BINS = 10
DATA_SEED = 2026100301
POSTERIOR_SEED = 2026100302
VALIDATION_SEED_BASE = 2026100400
X_1_MEAN = 0.0
X_1_STANDARD_DEVIATION = 1.0
X_2_MEAN = 2.0
X_2_STANDARD_DEVIATION = 2.0
TRUE_PRODUCT_EXPECTATION = 0.96
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifact" / "figures" / "diagnostics"
FIGURE_PATH = OUTPUT_DIR / "curvature_ppc.pdf"
FIT_FIGURE_PATH = OUTPUT_DIR / "conditional_mean_fits.pdf"
COMBINED_FIGURE_PATH = OUTPUT_DIR / "pmc_ppc_misspecification.pdf"

######################
# Experiment helpers #
######################

# Evaluate the square target function without introducing a response variable name.
def target_function(consensus):
    return consensus**2

# Evaluate the known conditional mean of the square target.
def true_conditional_mean(discrepancy):
    return 0.96 + 0.24 * discrepancy + 0.09 * discrepancy**2

# Simulate consensus, discrepancy, and target values from the Gaussian components.
def simulate_pmc_data(sample_size, seed):
    generator = np.random.default_rng(seed)
    x_1 = generator.normal(
        X_1_MEAN, X_1_STANDARD_DEVIATION, sample_size
    )
    x_2 = generator.normal(
        X_2_MEAN, X_2_STANDARD_DEVIATION, sample_size
    )
    consensus = (x_1 + x_2) / 2
    discrepancy = x_2 - x_1
    return consensus, discrepancy, target_function(consensus)

# Build a polynomial regression design through the requested degree.
def polynomial_design(discrepancy, degree):
    return np.polynomial.polynomial.polyvander(discrepancy, degree)

# Draw replicated target values from a conjugate Bayesian regression posterior.
def posterior_predictive_draws(
    discrepancy,
    target_values,
    degree,
    n_draws,
    seed,
):
    design = polynomial_design(discrepancy, degree)
    coefficients = np.linalg.lstsq(design, target_values, rcond=None)[0]
    residuals = target_values - design @ coefficients
    degrees_of_freedom = len(target_values) - design.shape[1]
    residual_sum_of_squares = float(residuals @ residuals)
    coefficient_covariance = np.linalg.inv(design.T @ design)
    covariance_factor = np.linalg.cholesky(coefficient_covariance)
    generator = np.random.default_rng(seed)
    noise_variances = invgamma.rvs(
        degrees_of_freedom / 2,
        scale=residual_sum_of_squares / 2,
        size=n_draws,
        random_state=generator,
    )
    standard_normal_coefficients = generator.normal(
        size=(n_draws, design.shape[1])
    )
    posterior_coefficients = coefficients + (
        np.sqrt(noise_variances)[:, None]
        * standard_normal_coefficients
        @ covariance_factor.T
    )
    posterior_means = posterior_coefficients @ design.T
    posterior_noise = generator.normal(
        scale=np.sqrt(noise_variances)[:, None],
        size=(n_draws, len(target_values)),
    )
    return coefficients, posterior_means + posterior_noise

# Compute a normalized binned-residual lack-of-fit statistic.
def lack_of_fit_statistic(discrepancy, target_values, degree):
    design = polynomial_design(discrepancy, degree)
    target_array = np.asarray(target_values)
    scalar_output = target_array.ndim == 1
    target_matrix = (
        target_array[:, None] if scalar_output else target_array.T
    )
    coefficients = np.linalg.lstsq(
        design, target_matrix, rcond=None
    )[0]
    residuals = target_matrix - design @ coefficients
    sorted_residuals = residuals[np.argsort(discrepancy)]
    bin_indices = np.array_split(
        np.arange(len(discrepancy)), LACK_OF_FIT_BINS
    )
    bin_means = np.stack(
        [sorted_residuals[indices].mean(axis=0) for indices in bin_indices]
    )
    bin_weights = np.asarray(
        [len(indices) for indices in bin_indices], dtype=float
    ) / len(discrepancy)
    binned_residual_rms = np.sqrt(
        np.sum(bin_weights[:, None] * bin_means**2, axis=0)
    )
    overall_residual_rms = np.sqrt(np.mean(residuals**2, axis=0))
    statistics = np.divide(
        binned_residual_rms,
        overall_residual_rms,
        out=np.zeros_like(binned_residual_rms),
        where=overall_residual_rms > 0,
    )
    return float(statistics[0]) if scalar_output else statistics

# Summarize one model using the shared posterior predictive lack-of-fit check.
def lack_of_fit_check(
    discrepancy,
    target_values,
    degree,
    n_draws,
    seed,
):
    coefficients, replicated_target_values = posterior_predictive_draws(
        discrepancy,
        target_values,
        degree=degree,
        n_draws=n_draws,
        seed=seed,
    )
    observed_lack_of_fit = lack_of_fit_statistic(
        discrepancy, target_values, degree
    )
    replicated_lack_of_fit = lack_of_fit_statistic(
        discrepancy, replicated_target_values, degree
    )
    posterior_predictive_p_value = (
        np.count_nonzero(
            replicated_lack_of_fit >= observed_lack_of_fit
        )
        + 1
    ) / (n_draws + 1)
    summary = {
        "degree": degree,
        "coefficients": coefficients.tolist(),
        "observed_lack_of_fit_statistic": observed_lack_of_fit,
        "replicated_lack_of_fit_interval": np.quantile(
            replicated_lack_of_fit, [0.025, 0.975]
        ).tolist(),
        "posterior_predictive_p_value": float(
            posterior_predictive_p_value
        ),
    }
    return summary, replicated_lack_of_fit

# Validate the lack-of-fit check across independent simulated datasets.
def repeated_validation():
    linear_p_values = []
    quadratic_p_values = []
    linear_interval_contains_observed = []
    quadratic_interval_contains_observed = []
    for repetition in range(VALIDATION_REPETITIONS):
        data_seed = VALIDATION_SEED_BASE + 3 * repetition
        posterior_seed = data_seed + 1
        consensus, discrepancy, target_values = simulate_pmc_data(
            SAMPLE_SIZE, data_seed
        )
        linear_check, _ = lack_of_fit_check(
            discrepancy,
            target_values,
            degree=1,
            n_draws=VALIDATION_POSTERIOR_PREDICTIVE_DRAWS,
            seed=posterior_seed,
        )
        quadratic_check, _ = lack_of_fit_check(
            discrepancy,
            target_values,
            degree=2,
            n_draws=VALIDATION_POSTERIOR_PREDICTIVE_DRAWS,
            seed=posterior_seed + 1,
        )
        linear_p_values.append(linear_check["posterior_predictive_p_value"])
        quadratic_p_values.append(
            quadratic_check["posterior_predictive_p_value"]
        )
        linear_interval_contains_observed.append(
            linear_check["replicated_lack_of_fit_interval"][0]
            <= linear_check["observed_lack_of_fit_statistic"]
            <= linear_check["replicated_lack_of_fit_interval"][1]
        )
        quadratic_interval_contains_observed.append(
            quadratic_check["replicated_lack_of_fit_interval"][0]
            <= quadratic_check["observed_lack_of_fit_statistic"]
            <= quadratic_check["replicated_lack_of_fit_interval"][1]
        )
    return {
        "repetitions": VALIDATION_REPETITIONS,
        "posterior_predictive_draws_per_model": (
            VALIDATION_POSTERIOR_PREDICTIVE_DRAWS
        ),
        "linear_model_flag_rate_at_0.05": float(
            np.mean(np.asarray(linear_p_values) <= 0.05)
        ),
        "quadratic_model_flag_rate_at_0.05": float(
            np.mean(np.asarray(quadratic_p_values) <= 0.05)
        ),
        "linear_model_median_ppc_value": float(np.median(linear_p_values)),
        "quadratic_model_median_ppc_value": float(
            np.median(quadratic_p_values)
        ),
        "linear_model_lack_of_fit_interval_inclusion_rate": float(
            np.mean(linear_interval_contains_observed)
        ),
        "quadratic_model_lack_of_fit_interval_inclusion_rate": float(
            np.mean(quadratic_interval_contains_observed)
        ),
    }

# Draw the replicated lack-of-fit distribution on one axis.
def draw_lack_of_fit_check(
    axis,
    title,
    summary,
    replicated_lack_of_fit,
):
    axis.hist(
        replicated_lack_of_fit,
        bins=30,
        color="#56B4E9",
        edgecolor="white",
    )
    axis.axvline(
        summary["observed_lack_of_fit_statistic"],
        color="#D55E00",
        linewidth=2,
    )
    axis.set_title(title)
    axis.set_xlabel(r"$T$")

# Draw the true, linear, and quadratic conditional means on one axis.
def draw_conditional_mean_fits(
    axis,
    discrepancy,
    target_values,
    linear_coefficients,
    quadratic_coefficients,
):
    grid = np.linspace(discrepancy.min(), discrepancy.max(), 400)
    linear_fit = polynomial_design(grid, 1) @ linear_coefficients
    quadratic_fit = polynomial_design(grid, 2) @ quadratic_coefficients
    axis.scatter(
        discrepancy,
        target_values,
        color="#999999",
        alpha=0.28,
        s=12,
    )
    axis.plot(
        grid,
        true_conditional_mean(grid),
        color="#CC79A7",
        linewidth=2,
        label="True conditional mean",
    )
    axis.plot(
        grid,
        quadratic_fit,
        color="#0072B2",
        linewidth=2,
        label="Well-specified model",
    )
    axis.plot(
        grid,
        linear_fit,
        color="#D55E00",
        linewidth=2,
        label="Misspecified model",
    )
    axis.set_xlabel(r"$D$")
    axis.set_ylabel(r"$h(C)$")
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    legend_order = (0, 2, 1)
    axis.legend(
        [legend_handles[index] for index in legend_order],
        [legend_labels[index] for index in legend_order],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
    )

# Plot the replicated lack-of-fit distributions against the observed statistics.
def plot_lack_of_fit_checks(checks):
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for axis, (title, summary, replicated_lack_of_fit) in zip(
        axes, checks, strict=True
    ):
        draw_lack_of_fit_check(
            axis, title, summary, replicated_lack_of_fit
        )
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)

# Plot the true, linear, and quadratic conditional mean functions.
def plot_conditional_mean_fits(
    discrepancy,
    target_values,
    linear_coefficients,
    quadratic_coefficients,
):
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    draw_conditional_mean_fits(
        axis,
        discrepancy,
        target_values,
        linear_coefficients,
        quadratic_coefficients,
    )
    figure.tight_layout()
    figure.savefig(FIT_FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)

# Combine the conditional mean and lack-of-fit checks into one figure.
def plot_combined_figure(
    discrepancy,
    target_values,
    linear_check,
    quadratic_check,
    linear_lack_of_fit,
    quadratic_lack_of_fit,
):
    combined_style = {
        "font.size": 17,
        "axes.titlesize": 21,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
    with plt.rc_context(combined_style):
        figure = plt.figure(figsize=(10.5, 5.6), layout="constrained")
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(1.35, 1),
            hspace=0.08,
            wspace=0.25,
        )
        conditional_mean_axis = figure.add_subplot(grid[:, 0])
        linear_axis = figure.add_subplot(grid[0, 1])
        quadratic_axis = figure.add_subplot(grid[1, 1])
        draw_conditional_mean_fits(
            conditional_mean_axis,
            discrepancy,
            target_values,
            np.asarray(linear_check["coefficients"]),
            np.asarray(quadratic_check["coefficients"]),
        )
        draw_lack_of_fit_check(
            linear_axis,
            "Misspecified",
            linear_check,
            linear_lack_of_fit,
        )
        draw_lack_of_fit_check(
            quadratic_axis,
            "Well-specified",
            quadratic_check,
            quadratic_lack_of_fit,
        )
        figure.savefig(COMBINED_FIGURE_PATH, bbox_inches="tight")
        plt.close(figure)

########################
# Experiment execution #
########################

# Run the posterior predictive lack-of-fit checks.
def run_experiment():
    consensus, discrepancy, target_values = simulate_pmc_data(
        SAMPLE_SIZE, DATA_SEED
    )
    linear_check, linear_lack_of_fit = lack_of_fit_check(
        discrepancy,
        target_values,
        degree=1,
        n_draws=POSTERIOR_PREDICTIVE_DRAWS,
        seed=POSTERIOR_SEED,
    )
    quadratic_check, quadratic_lack_of_fit = lack_of_fit_check(
        discrepancy,
        target_values,
        degree=2,
        n_draws=POSTERIOR_PREDICTIVE_DRAWS,
        seed=POSTERIOR_SEED + 1,
    )
    summary = {
        "sample_size": SAMPLE_SIZE,
        "posterior_predictive_draws": POSTERIOR_PREDICTIVE_DRAWS,
        "lack_of_fit_bins": LACK_OF_FIT_BINS,
        "data_seed": DATA_SEED,
        "posterior_seed": POSTERIOR_SEED,
        "component_parameters": {
            "x_1_mean": X_1_MEAN,
            "x_1_standard_deviation": X_1_STANDARD_DEVIATION,
            "x_2_mean": X_2_MEAN,
            "x_2_standard_deviation": X_2_STANDARD_DEVIATION,
        },
        "target": "h(u) = u^2",
        "true_product_expectation": TRUE_PRODUCT_EXPECTATION,
        "linear_model": linear_check,
        "quadratic_model": quadratic_check,
        "repeated_validation": repeated_validation(),
    }
    plot_lack_of_fit_checks(
        [
            ("Linear model", linear_check, linear_lack_of_fit),
            ("Quadratic model", quadratic_check, quadratic_lack_of_fit),
        ]
    )
    plot_conditional_mean_fits(
        discrepancy,
        target_values,
        np.asarray(linear_check["coefficients"]),
        np.asarray(quadratic_check["coefficients"]),
    )
    plot_combined_figure(
        discrepancy,
        target_values,
        linear_check,
        quadratic_check,
        linear_lack_of_fit,
        quadratic_lack_of_fit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved figure to {FIGURE_PATH}")
    print(f"Saved figure to {FIT_FIGURE_PATH}")
    print(f"Saved figure to {COMBINED_FIGURE_PATH}")

if __name__ == "__main__":
    run_experiment()
