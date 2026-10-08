#!/usr/bin/env python3

"""Plot PMC-GP and GBF calibration at S = 1000 for several GBF particle counts."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
import scipy.stats

# Sibling paper scripts
from convergence_plots import DATA_ROOT, FIGURES_ROOT, METHOD_COLORS, load_pyplot

######################
# Inputs and outputs #
######################

SAMPLE_SIZE = 1000
RESULTS_ROOT = DATA_ROOT / "calibration_sample_size"
RESULTS_PATHS = [
    RESULTS_ROOT / "results.csv",
    RESULTS_ROOT / "particles_10000" / "results.csv",
    RESULTS_ROOT / "particles_100000" / "results.csv",
]
DEFAULT_OUTPUT = (
    FIGURES_ROOT / "calibration" / "univariate_gaussian_k2_calibration.pdf"
)

################
# Method style #
################

METHOD_LABELS = {"pmc_gp": "PMC-GP", "gbf": "GBF"}

# Map each GBF particle count to a darkening shade of the GBF pink
GBF_COLORS = {1000: "#E2A9C9", 10000: "#CC79A7", 100000: "#8A4570"}

####################
# Coverage summary #
####################

# Return Wilson confidence limits for a binomial coverage estimate.
def wilson_limits(successes, total, confidence_level=0.95):
    estimate = successes / total
    z_value = scipy.stats.norm.ppf((1.0 + confidence_level) / 2.0)
    denominator = 1.0 + z_value**2 / total
    center = (estimate + z_value**2 / (2.0 * total)) / denominator
    half_width = z_value * np.sqrt(
        estimate * (1.0 - estimate) / total
        + z_value**2 / (4.0 * total**2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)

# Summarize coverage and Wilson limits by method and nominal level.
def summarize(frame):
    rows = []
    for (method, level), group in frame.groupby(
        ["method", "nominal_coverage"], sort=True
    ):
        successes = int(group["covered"].sum())
        total = len(group)
        lower, upper = wilson_limits(successes, total)
        rows.append(
            {
                "method": method,
                "nominal_coverage": level,
                "empirical_coverage": successes / total,
                "wilson_lower_95": lower,
                "wilson_upper_95": upper,
            }
        )
    return pd.DataFrame(rows)

# Load rows at the fixed sample size, keeping PMC-GP from the first file only.
def load_results(paths):
    frames = []
    for index, path in enumerate(paths):
        frame = pd.read_csv(path)
        frame = frame[frame["sample_size"] == SAMPLE_SIZE]
        if index > 0:
            frame = frame[frame["method"] == "gbf"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

# Summarize coverage and Wilson limits by method, particle count, and level.
def summarize_by_particles(frame):
    summaries = []
    gbf = frame[frame["method"] == "gbf"]
    for particles, group in gbf.groupby("n_inference", sort=True):
        summary = summarize(group)
        summary.insert(1, "particles", int(particles))
        summaries.append(summary)
    pmc_gp = summarize(frame[frame["method"] == "pmc_gp"])
    pmc_gp.insert(1, "particles", np.nan)
    summaries.append(pmc_gp)
    return pd.concat(summaries, ignore_index=True)

##########
# Figure #
##########

# Draw one calibration curve with circle markers.
def plot_curve(axis, subset, color, label):
    axis.plot(
        subset["nominal_coverage"], subset["empirical_coverage"],
        color=color, marker="o", linewidth=1.8, markersize=5,
        label=label, zorder=3,
    )

# Save the calibration panel overlaying every GBF particle count on PMC-GP.
def save_plot(summary, output_path):
    plt = load_pyplot()
    figure, axis = plt.subplots(figsize=(7.6, 6.2))
    axis.plot(
        [0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.3,
        label="Perfect calibration", zorder=1,
    )
    plot_curve(
        axis, summary[summary["method"] == "pmc_gp"],
        METHOD_COLORS[METHOD_LABELS["pmc_gp"]], METHOD_LABELS["pmc_gp"],
    )
    for particles, color in GBF_COLORS.items():
        particle_multiple = particles // SAMPLE_SIZE
        particle_label = (
            r"$S$" if particle_multiple == 1
            else rf"${particle_multiple}\,S$"
        )
        plot_curve(
            axis, summary[summary["particles"] == particles], color,
            f"{METHOD_LABELS['gbf']} with {particle_label} particles",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    ticks = np.arange(0.0, 1.01, 0.2)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.grid(alpha=0.25)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.14, right=0.70, bottom=0.13, top=0.96)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

##########################
# Command-line interface #
##########################

# Parse the optional figure output path.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()

# Load the particle-count calibration results and generate their figure.
def main():
    arguments = parse_arguments()
    summary = summarize_by_particles(load_results(RESULTS_PATHS))
    print(summary.to_string(index=False))
    save_plot(summary, arguments.output_path)
    print(f"plot: {arguments.output_path}")

if __name__ == "__main__":
    main()
