#!/usr/bin/env python3

"""Plot the HSP90 image-level, Oracle combined and PMC posterior summaries."""

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import json
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd

# PMC
from convergence_plots import (
    DISPLAY_SCALE,
    FIGURES_ROOT,
    FONT_SIZES,
    PANEL_SIZE_ONE_ROW,
    PROJECT_ROOT,
    load_pyplot,
    save_figure,
)

#####################
# Experiment layout #
#####################

HSP90_ROOT = PROJECT_ROOT / "data" / "results" / "hsp90"
DEFAULT_DRAWS = HSP90_ROOT / "sbi" / "posterior_draws.npz"
DEFAULT_ORACLE_DENSITY = HSP90_ROOT / "oracle" / "oracle_density.csv"
DEFAULT_OUTPUT = FIGURES_ROOT / "sbi" / "hsp90_posteriors.pdf"
PMC_METHODS = (
    ("PMC-OLS", "pmc_ols_moments.json"),
    ("PMC-MLP", "pmc_mlp_moments.json"),
    ("PMC-GP", "pmc_gp_moments.json"),
)
WINDOW_MIN = 0.40
WINDOW_MAX = 0.80
DISPLAY_X_MIN = 0.35
WINDOW_POINTS = 801
HSP90_COLORS = {
    "Image-level": "#7f7f7f",
    "Combined": "#0072B2",
    "PMC-OLS": "#1baf7a",
    "PMC-MLP": "#008300",
    "PMC-GP": "#00441b",
}
IMAGE_LINE_ALPHA = 0.5
IMAGE_LINE_WIDTH = 0.8 * DISPLAY_SCALE * 2
PMC_LINE_WIDTH = 1.6 * 0.8 * DISPLAY_SCALE * 2
ORACLE_LINE_WIDTH = 1.25 * PMC_LINE_WIDTH
PMC_LINE_STYLE = "--"
LEFT_MARGIN = 0.42
BOTTOM_MARGIN = 0.40
TOP_MARGIN = 0.10
RIGHT_MARGIN = 0.10

################
# Data loading #
################

# Load the draws of every image-level posterior on the z scale.
def load_image_draws(path):
    with np.load(path) as saved:
        return saved["draws_z"]

# Load the Oracle combined posterior density on its fixed window grid.
def load_oracle_density(path):
    return pd.read_csv(path)

# Load the PMC point estimates of the mean keyed by method label.
def load_pmc_means(result_dir):
    means = {}
    for label, filename in PMC_METHODS:
        with (result_dir / "pmc" / filename).open(encoding="utf-8") as source:
            means[label] = float(json.load(source)["mean_z"])
    return means

######################
# Density estimation #
######################

# Estimate a Gaussian KDE with a bandwidth based on the interquartile range only.
def robust_kde(draws, grid):
    # The IQR rule is used because the heavy tails inflate the standard deviation
    lower, upper = np.percentile(draws, [25, 75])
    bandwidth = 0.9 * (upper - lower) / 1.34 * draws.size ** (-1 / 5)
    scaled = (grid[:, None] - draws[None, :]) / bandwidth
    kernel = np.exp(-0.5 * scaled**2) / np.sqrt(2 * np.pi)
    return kernel.mean(axis=1) / bandwidth

#################
# Panel drawing #
#################

# Draw the image-level, Oracle combined and PMC mean summaries on one axes.
def draw_posterior_panel(axis, draws, oracle_density, pmc_means):
    grid = np.linspace(WINDOW_MIN, WINDOW_MAX, WINDOW_POINTS)
    for image_draws in draws:
        axis.plot(
            grid,
            robust_kde(image_draws, grid),
            color=HSP90_COLORS["Image-level"],
            alpha=IMAGE_LINE_ALPHA,
            linewidth=IMAGE_LINE_WIDTH,
            zorder=2,
        )
    axis.plot(
        oracle_density["z"],
        oracle_density["density"],
        color=HSP90_COLORS["Combined"],
        linewidth=ORACLE_LINE_WIDTH,
        zorder=3,
    )
    label_positions = {
        "PMC-MLP": (0.67, 0.70, "left"),
        "PMC-GP": (0.67, 0.55, "left"),
        "PMC-OLS": (0.67, 0.40, "left"),
    }
    for label, mean in pmc_means.items():
        axis.axvline(
            mean,
            ymin=0,
            ymax=1,
            color=HSP90_COLORS[label],
            linestyle=PMC_LINE_STYLE,
            linewidth=PMC_LINE_WIDTH,
            zorder=2.5,
        )
        text_x, text_y, alignment = label_positions[label]
        axis.text(
            text_x,
            text_y,
            label,
            color=HSP90_COLORS[label],
            fontsize=FONT_SIZES["legend"],
            ha=alignment,
            va="center",
            transform=axis.transAxes,
            zorder=4,
        )
    axis.set_xlim(DISPLAY_X_MIN, WINDOW_MAX)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("$z$", fontsize=FONT_SIZES["label"], labelpad=12 * DISPLAY_SCALE)
    axis.set_ylabel("Density", fontsize=FONT_SIZES["label"], labelpad=12 * DISPLAY_SCALE)
    axis.grid(True, color="#b0b0b0", alpha=0.3, linewidth=0.8 * DISPLAY_SCALE)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.6)
        spine.set_zorder(10)
    axis.tick_params(
        axis="both",
        labelsize=FONT_SIZES["tick"],
        pad=8 * DISPLAY_SCALE,
        color="black",
        labelcolor="black",
        width=0.8 * DISPLAY_SCALE,
        length=3.5 * DISPLAY_SCALE,
    )
    axis.text(
        0.03,
        0.74,
        "Individual\nposteriors",
        color=HSP90_COLORS["Image-level"],
        fontsize=FONT_SIZES["legend"],
        ha="left",
        va="top",
        transform=axis.transAxes,
        zorder=4,
    )
    axis.text(
        0.03,
        0.95,
        "Combined\nposterior",
        color=HSP90_COLORS["Combined"],
        fontsize=FONT_SIZES["legend"],
        ha="left",
        va="top",
        transform=axis.transAxes,
        zorder=4,
    )

###################
# Figure assembly #
###################

# Build the SBI figure with its fixed-size square posterior panel.
def create_figure(draws, oracle_density, pmc_means):
    pyplot = load_pyplot()
    width = LEFT_MARGIN + PANEL_SIZE_ONE_ROW + RIGHT_MARGIN
    height = BOTTOM_MARGIN + PANEL_SIZE_ONE_ROW + TOP_MARGIN
    figure = pyplot.figure(figsize=(width, height))
    axis = figure.add_axes(
        [
            LEFT_MARGIN / width,
            BOTTOM_MARGIN / height,
            PANEL_SIZE_ONE_ROW / width,
            PANEL_SIZE_ONE_ROW / height,
        ]
    )
    draw_posterior_panel(axis, draws, oracle_density, pmc_means)
    return figure

##########################
# Command-line interface #
##########################

# Generate the HSP90 posterior figure from saved draws and Oracle density.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=DEFAULT_DRAWS)
    parser.add_argument(
        "--oracle-density", type=Path, default=DEFAULT_ORACLE_DENSITY
    )
    parser.add_argument("--result-dir", type=Path, default=HSP90_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    figure = create_figure(
        load_image_draws(arguments.draws),
        load_oracle_density(arguments.oracle_density),
        load_pmc_means(arguments.result_dir),
    )
    output_path = save_figure(figure, arguments.output)
    print(f"plot: {output_path}")

if __name__ == "__main__":
    main()
