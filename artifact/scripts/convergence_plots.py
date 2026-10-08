#!/usr/bin/env python3

##############################
# Import necessary libraries #
##############################

# Standard libraries
import argparse
import math
import os
import tempfile
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd

###########################
# Canonical project paths #
###########################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "experiments" / "outputs"
FIGURES_ROOT = PROJECT_ROOT / "artifact" / "figures"

#######################
# Shared method style #
#######################

METHOD_COLORS = {
    "Oracle": "#D55E00",
    "MCF": "#8C564B",
    "BF": "#0072B2",
    "GBF": "#CC79A7",
    "PMC-OLS": "#1baf7a",
    "PMC-MLP": "#008300",
    "PMC-GP": "#00441b",
}
METHOD_ORDER = tuple(METHOD_COLORS)

#################
# Figure layout #
#################

# Panel geometry in inches, identical for every figure
# Panel side in inches for figures laid out in three or more rows
PANEL_SIZE_MANY_ROWS = 0.55
# Panel side in inches for figures laid out in exactly two rows
PANEL_SIZE_TWO_ROWS = 0.80
# Panel side in inches for figures laid out in a single row
PANEL_SIZE_ONE_ROW = 1.00
PANEL_GAP = 0.06
LEFT_MARGIN = 0.42
BOTTOM_MARGIN = 0.40
TITLE_BAND = 0.22
RIGHT_BAND = 0.38
OUTER_MARGIN = 0.05
BLOCK_GAP = 0.12
BLOCK_SPACING = 0.06
# Fraction of the x range left blank at each end of a panel, keeping edge tick labels apart
X_MARGIN = 0.12
# Blank padding kept around the cropped figure content, in inches
CROP_PADDING = 0.02
# Vertical distance between the legend title and the method names, in inches
LEGEND_TITLE_GAP = 0.03
LEGEND_WIDTH = 0.60
MAX_FIGURE_WIDTH = 6.75
# Factor by which the original style values are scaled to keep the printed look
DISPLAY_SCALE = 0.2
# Font sizes in points
FONT_SIZES = {
    "tick": 24 * DISPLAY_SCALE,
    "label": 32 * DISPLAY_SCALE,
    "title": 32 * DISPLAY_SCALE,
    "annotation": 28 * DISPLAY_SCALE,
    "legend": 5,
}
CONVERGENCE_METRICS = [
    ("mse", "MSE"),
    ("training_time_seconds", "Runtime (s)"),
]
# Largest number of nodes or dimension shown in the scaling figures
MAX_SCALING_VALUE = 20

################
# Data loading #
################

# Load and concatenate deterministically ordered CSV batches beneath the data root.
def load_csv_batches(pattern):
    paths = [
        path
        for path in sorted(DATA_ROOT.glob(pattern))
        if not path.name.endswith("_failures.csv")
    ]
    if not paths:
        raise FileNotFoundError(
            f"No CSV files match {pattern!r} beneath {DATA_ROOT}"
        )

    return pd.concat(
        (pd.read_csv(path, low_memory=False) for path in paths),
        ignore_index=True,
    )

########################
# Method normalization #
########################

# Normalize method labels without mutating the input data.
def normalize_methods(data):
    if "method" not in data.columns:
        raise KeyError("Method normalization requires a 'method' column")

    normalized = data.copy()
    normalized["method_label"] = normalized["method"].astype(object)
    if "algorithm" not in normalized.columns:
        return normalized

    normalized.loc[normalized["algorithm"] == "Oracle", "method_label"] = "Oracle"
    pmc_rows = normalized["algorithm"] == "PMC"
    pmc_labels = normalized.loc[pmc_rows, "method"].map(
        {"linear": "PMC-OLS", "MLP": "PMC-MLP", "GP": "PMC-GP"}
    )
    normalized.loc[pmc_rows, "method_label"] = pmc_labels.fillna(
        normalized.loc[pmc_rows, "method"]
    )
    return normalized

#########################
# Convergence summaries #
#########################

# Summarize positive convergence metrics on the base-10 logarithmic scale.
def summarize_convergence(data, metric, group_columns):
    required_columns = ["n", metric, *group_columns]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    columns = [*group_columns, "n", metric]
    working = data.loc[data[metric] > 0, columns].copy()
    working["log_metric"] = np.log10(working[metric])
    summary = (
        working.groupby([*group_columns, "n"], dropna=False, sort=True)[
            "log_metric"
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary["sem"] = summary["sem"].fillna(0.0)
    summary["log_n"] = np.log10(summary["n"])
    output_columns = [*group_columns, "n", "log_n", "mean", "sem"]
    return summary[output_columns].sort_values(
        [*group_columns, "n"], kind="stable"
    ).reset_index(drop=True)

###########################
# Headless plotting setup #
###########################

# Configure and load Matplotlib for headless plotting with a writable cache.
def load_pyplot():
    cache_directory = Path(tempfile.gettempdir()) / "pmc-matplotlib"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_directory)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot

#######################
# Plotting primitives #
#######################

# Draw one convergence series with markers and error bars.
def draw_convergence_series(
    axis,
    summary,
    color,
    x_column="log_n",
    y_column="mean",
    y_error_column="sem",
):
    if isinstance(y_error_column, tuple):
        y_error = summary[list(y_error_column)].to_numpy().T
    else:
        y_error = summary[y_error_column]
    return axis.errorbar(
        summary[x_column],
        summary[y_column],
        yerr=y_error,
        color=color,
        fmt="o",
        markersize=4 * DISPLAY_SCALE,
        markeredgewidth=1.0 * DISPLAY_SCALE,
        linewidth=1.5 * DISPLAY_SCALE,
        linestyle="-",
        elinewidth=2 * DISPLAY_SCALE,
        capsize=4 * DISPLAY_SCALE,
        capthick=2 * DISPLAY_SCALE,
        zorder=3,
    )

# List even integer powers of ten inside an axis range, thinning them when crowded.
def _even_power_ticks(lower, upper, max_ticks=4):
    step = 2
    ticks = list(
        range(math.ceil(lower / step) * step, math.floor(upper / step) * step + 1, step)
    )
    while len(ticks) > max_ticks:
        step *= 2
        ticks = list(
            range(
                math.ceil(lower / step) * step,
                math.floor(upper / step) * step + 1,
                step,
            )
        )
    return ticks or [2 * round((lower + upper) / 4)]

# Style one panel with power-of-ten ticks on both axes.
def _style_axis(axis, *, show_x_ticks, show_y_ticks):
    from matplotlib.ticker import FuncFormatter

    axis.grid(True, alpha=0.3, linewidth=0.8 * DISPLAY_SCALE)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8 * DISPLAY_SCALE)
    axis.tick_params(
        axis="both",
        labelsize=FONT_SIZES["tick"],
        pad=8 * DISPLAY_SCALE,
        width=0.8 * DISPLAY_SCALE,
        length=3.5 * DISPLAY_SCALE,
    )
    axis.set_xticks(_even_power_ticks(*axis.get_xlim()))
    axis.set_yticks(_even_power_ticks(*axis.get_ylim()))
    formatter = FuncFormatter(lambda value, _: rf"$10^{{{value:g}}}$")
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)
    axis.tick_params(labelbottom=show_x_ticks, labelleft=show_y_ticks)

# Build legend handles in canonical method order.
def _legend_handles(pyplot, methods):
    return [
        pyplot.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=1,
            label=method,
        )
        for method in METHOD_ORDER
        if method in methods
    ]

####################
# Combined figures #
####################

# Choose the square panel side in inches from the number of rows in the figure layout.
def _panel_size(layout_rows):
    if layout_rows == 1:
        return PANEL_SIZE_ONE_ROW
    if layout_rows == 2:
        return PANEL_SIZE_TWO_ROWS
    return PANEL_SIZE_MANY_ROWS

# Build one combined MSE and runtime figure of fixed-size square panels.
def create_combined_figure(
    panels,
    *,
    column_levels,
    row_levels,
    column_label,
    row_label,
    methods,
    y_error_column="sem",
):
    pyplot = load_pyplot()
    num_rows = len(row_levels)
    num_columns = len(column_levels)
    side_by_side = num_rows > 1 or num_columns == 1
    panel_size = _panel_size(num_rows if side_by_side else 2)
    block_width = (
        LEFT_MARGIN + num_columns * panel_size + (num_columns - 1) * PANEL_GAP
    )
    block_stride = block_width + BLOCK_SPACING
    if side_by_side:
        core_width = (
            2 * OUTER_MARGIN
            + block_width
            + block_stride
            + (RIGHT_BAND if row_label else 0)
        )
        core_height = (
            BOTTOM_MARGIN
            + num_rows * panel_size
            + (num_rows - 1) * PANEL_GAP
            + TITLE_BAND
        )
    else:
        core_width = 2 * OUTER_MARGIN + block_width
        core_height = (
            BOTTOM_MARGIN + 2 * panel_size + BLOCK_GAP + TITLE_BAND
        )
    width = core_width + LEGEND_WIDTH
    height = core_height
    assert width <= MAX_FIGURE_WIDTH + 1e-6, f"Figure too wide: {width:.2f} in"

    figure = pyplot.figure(figsize=(width, height))
    axes = {}
    first_axis = None
    for metric_index in range(len(CONVERGENCE_METRICS)):
        block_first_axis = None
        for row_index in range(num_rows):
            for column_index in range(num_columns):
                if side_by_side:
                    x_origin = (
                        OUTER_MARGIN
                        + metric_index * block_stride
                        + LEFT_MARGIN
                        + column_index * (panel_size + PANEL_GAP)
                    )
                    y_origin = BOTTOM_MARGIN + (
                        num_rows - 1 - row_index
                    ) * (panel_size + PANEL_GAP)
                else:
                    x_origin = (
                        OUTER_MARGIN
                        + LEFT_MARGIN
                        + column_index * (panel_size + PANEL_GAP)
                    )
                    y_origin = BOTTOM_MARGIN + (1 - metric_index) * (
                        panel_size + BLOCK_GAP
                    )
                shared = {}
                if first_axis is not None:
                    shared["sharex"] = first_axis
                if block_first_axis is not None:
                    shared["sharey"] = block_first_axis
                axis = figure.add_axes(
                    [
                        x_origin / width,
                        y_origin / height,
                        panel_size / width,
                        panel_size / height,
                    ],
                    **shared,
                )
                axis.margins(x=X_MARGIN)
                axes[(metric_index, row_index, column_index)] = axis
                first_axis = first_axis or axis
                block_first_axis = block_first_axis or axis

    for (metric_index, row_index, column_index), axis in axes.items():
        summary = panels[(metric_index, row_index, column_index)]
        for method in METHOD_ORDER:
            method_summary = summary[summary["method_label"] == method]
            if not method_summary.empty:
                draw_convergence_series(
                    axis,
                    method_summary,
                    METHOD_COLORS[method],
                    y_error_column=y_error_column,
                )
        is_bottom = (
            row_index == num_rows - 1 if side_by_side else metric_index == 1
        )
        is_top = row_index == 0 if side_by_side else metric_index == 0
        _style_axis(
            axis, show_x_ticks=is_bottom, show_y_ticks=column_index == 0
        )
        if is_bottom:
            axis.set_xlabel(
                r"$S$",
                fontsize=FONT_SIZES["label"],
                labelpad=12 * DISPLAY_SCALE,
            )
        if is_top and column_label and column_levels[column_index] is not None:
            axis.set_title(
                f"{column_label} = {column_levels[column_index]}",
                fontsize=FONT_SIZES["title"],
                pad=24 * DISPLAY_SCALE,
            )
        if (
            side_by_side
            and metric_index == 1
            and column_index == num_columns - 1
            and row_label
            and row_levels[row_index] is not None
        ):
            axis.annotate(
                f"{row_label} = {row_levels[row_index]}",
                xy=(1.06, 0.5),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=FONT_SIZES["annotation"],
            )

    for metric_index, (_, y_label) in enumerate(CONVERGENCE_METRICS):
        if side_by_side:
            x_text = (OUTER_MARGIN + metric_index * block_stride + 0.11) / width
            y_text = (
                BOTTOM_MARGIN
                + (num_rows * panel_size + (num_rows - 1) * PANEL_GAP) / 2
            ) / height
        else:
            x_text = (OUTER_MARGIN + 0.11) / width
            y_text = (
                BOTTOM_MARGIN
                + (1 - metric_index) * (panel_size + BLOCK_GAP)
                + panel_size / 2
            ) / height
        figure.text(
            x_text,
            y_text,
            y_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FONT_SIZES["label"],
        )

    handles = _legend_handles(pyplot, methods)
    panels_center = (BOTTOM_MARGIN + core_height - TITLE_BAND) / 2
    legend = figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=((core_width - 0.02) / width, panels_center / height),
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        handlelength=1.5,
    )
    legend_box = legend.get_window_extent(
        figure.canvas.get_renderer()
    ).transformed(figure.transFigure.inverted())
    figure.text(
        legend_box.x0,
        legend_box.y1 + LEGEND_TITLE_GAP / height,
        "Method",
        ha="left",
        va="bottom",
        fontsize=FONT_SIZES["legend"] + 0.5,
    )
    return figure

# Build one single-metric figure of fixed-size square panels.
def create_single_metric_figure(
    panels,
    *,
    column_levels,
    column_label,
    methods,
    x_label,
    y_label,
    y_error_column="sem",
):
    pyplot = load_pyplot()
    num_columns = len(column_levels)
    panel_size = _panel_size(1)
    block_width = (
        LEFT_MARGIN + num_columns * panel_size + (num_columns - 1) * PANEL_GAP
    )
    core_width = 2 * OUTER_MARGIN + block_width
    width = core_width + LEGEND_WIDTH
    height = BOTTOM_MARGIN + panel_size + TITLE_BAND
    assert width <= MAX_FIGURE_WIDTH + 1e-6, f"Figure too wide: {width:.2f} in"

    figure = pyplot.figure(figsize=(width, height))
    axes = []
    first_axis = None
    for column_index in range(num_columns):
        x_origin = (
            OUTER_MARGIN
            + LEFT_MARGIN
            + column_index * (panel_size + PANEL_GAP)
        )
        shared = {}
        if first_axis is not None:
            shared = {"sharex": first_axis, "sharey": first_axis}
        axis = figure.add_axes(
            [
                x_origin / width,
                BOTTOM_MARGIN / height,
                panel_size / width,
                panel_size / height,
            ],
            **shared,
        )
        axis.margins(x=X_MARGIN)
        axes.append(axis)
        first_axis = first_axis or axis

    for column_index, axis in enumerate(axes):
        summary = panels[column_index]
        for method in METHOD_ORDER:
            method_summary = summary[summary["method_label"] == method]
            if not method_summary.empty:
                draw_convergence_series(
                    axis,
                    method_summary,
                    METHOD_COLORS[method],
                    y_error_column=y_error_column,
                )
        _style_axis(
            axis,
            show_x_ticks=True,
            show_y_ticks=column_index == 0,
        )
        axis.set_xlabel(
            x_label,
            fontsize=FONT_SIZES["label"],
            labelpad=12 * DISPLAY_SCALE,
        )
        if column_label and column_levels[column_index] is not None:
            axis.set_title(
                f"{column_label} = {column_levels[column_index]}",
                fontsize=FONT_SIZES["title"],
                pad=24 * DISPLAY_SCALE,
            )

    figure.text(
        (OUTER_MARGIN + 0.11) / width,
        (BOTTOM_MARGIN + panel_size / 2) / height,
        y_label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT_SIZES["label"],
    )

    handles = _legend_handles(pyplot, methods)
    panels_center = (BOTTOM_MARGIN + height - TITLE_BAND) / 2
    legend = figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=((core_width - 0.02) / width, panels_center / height),
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        handlelength=1.5,
    )
    legend_box = legend.get_window_extent(
        figure.canvas.get_renderer()
    ).transformed(figure.transFigure.inverted())
    figure.text(
        legend_box.x0,
        legend_box.y1 + LEGEND_TITLE_GAP / height,
        "Method",
        ha="left",
        va="bottom",
        fontsize=FONT_SIZES["legend"] + 0.5,
    )
    return figure

#################
# Figure output #
#################

# Save and close one figure as a PDF at its exact size.
def save_figure(figure, output_path):
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".pdf":
        raise ValueError(f"Figure output must use a .pdf suffix: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=CROP_PADDING)
    load_pyplot().close(figure)
    return output_path

####################
# Convergence sets #
####################

# List the sorted levels of a facet column, or a single placeholder.
def _facet_levels(summary, facet):
    if facet is None:
        return [None]
    return summary[facet].drop_duplicates().sort_values(na_position="last").tolist()

# Restrict a summary to one level of a facet column.
def _facet_subset(summary, facet, level):
    if facet is None:
        return summary
    match = summary[facet].isna() if pd.isna(level) else summary[facet] == level
    return summary[match]

# Generate all combined convergence PDFs for the supplied targets.
def generate_convergence_set(
    data,
    output_directory,
    facet_row=None,
    facet_column=None,
    facet_row_label=None,
    facet_column_label=None,
):
    facet_columns = [
        column for column in [facet_row, facet_column] if column is not None
    ]
    required_columns = [
        "h",
        "method_label",
        "mse",
        "training_time_seconds",
        *facet_columns,
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    output_directory = Path(output_directory)
    methods = set(data["method_label"].dropna())
    group_columns = [*facet_columns, "method_label"]
    generated_paths = []
    for target in sorted(data["h"].dropna().unique()):
        target_data = data[data["h"] == target]
        summaries = [
            summarize_convergence(target_data, metric, group_columns)
            for metric, _ in CONVERGENCE_METRICS
        ]
        row_levels = _facet_levels(summaries[0], facet_row)
        column_levels = _facet_levels(summaries[0], facet_column)
        panels = {}
        for metric_index, summary in enumerate(summaries):
            for row_index, row_level in enumerate(row_levels):
                for column_index, column_level in enumerate(column_levels):
                    panels[(metric_index, row_index, column_index)] = (
                        _facet_subset(
                            _facet_subset(summary, facet_row, row_level),
                            facet_column,
                            column_level,
                        )
                    )
        figure = create_combined_figure(
            panels,
            column_levels=column_levels,
            row_levels=row_levels,
            column_label=(
                (facet_column_label or facet_column) if facet_column else None
            ),
            row_label=(facet_row_label or facet_row) if facet_row else None,
            methods=methods,
        )
        generated_paths.append(
            save_figure(figure, output_directory / f"h_{target}.pdf")
        )
    return generated_paths

###################
# Scaling figures #
###################

# Create a complete two-metric scaling figure for one target.
def create_scaling_figure(data, target, facet_column, facet_label):
    required_columns = [
        "h",
        "method_label",
        "n",
        "mse",
        "training_time_seconds",
        facet_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    target_data = data[data["h"] == target]
    if target_data.empty:
        raise ValueError(f"Target {target!r} is absent from the scaling data")

    facet_levels = sorted(target_data[facet_column].dropna().unique())
    panels = {}
    for metric_index, (metric, _) in enumerate(CONVERGENCE_METRICS):
        for column_index, facet_level in enumerate(facet_levels):
            panel = target_data[target_data[facet_column] == facet_level]
            panels[(metric_index, 0, column_index)] = summarize_convergence(
                panel, metric, ["method_label"]
            )
    return create_combined_figure(
        panels,
        column_levels=facet_levels,
        row_levels=[None],
        column_label=facet_label,
        row_label=None,
        methods=set(target_data["method_label"].dropna()),
    )

###############################
# Univariate Gaussian figures #
###############################

# Generate the complete and main-paper univariate Gaussian figure sets.
def generate_univariate_gaussian_k2(output_root=FIGURES_ROOT):
    data = normalize_methods(
        load_csv_batches(
            "gaussian/univariate/location_scale_grid/k_2/results/"
            "n_*/batch_*.csv"
        )
    )
    output_directory = (
        Path(output_root)
        / "gaussian"
        / "univariate"
        / "k_2"
        / "location_scale_grid"
    )
    shared_arguments = {
        "facet_row": "sigma_2",
        "facet_column": "mu_2",
        "facet_row_label": r"$\sigma_2$",
        "facet_column_label": r"$\mu_2$",
    }
    full_paths = generate_convergence_set(
        data,
        output_directory,
        **shared_arguments,
    )

    main_data = data[
        (data["h"] == "identity")
        & data["mu_2"].isin([1, 2, 3])
        & data["sigma_2"].isin([1, 2, 3])
    ]
    expected_levels = {1, 2, 3}
    for column in ["mu_2", "sigma_2"]:
        selected_levels = set(main_data[column].dropna().unique())
        if selected_levels != expected_levels:
            raise ValueError(
                f"Expected {column} levels {sorted(expected_levels)}, "
                f"found {sorted(selected_levels)}"
            )
    main_paths = generate_convergence_set(
        main_data,
        output_directory / "main_3x3",
        **shared_arguments,
    )
    return [*full_paths, *main_paths]

#################################
# PMC dimension scaling figures #
#################################

# Validate one set of dedicated PMC dimension-scaling results
def _validate_dimension_scaling_results(
    data,
    *,
    result_name,
    expected_algorithm,
    expected_method,
    expected_dimensions,
    expected_iterations,
    expected_rows,
    expected_h="normalized_sum",
    allow_incomplete=False,
):
    required_columns = [
        "algorithm",
        "method",
        "h",
        "p",
        "n",
        "iteration_id",
        "mse",
        "training_time_seconds",
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing {result_name} dimension-scaling columns: "
            f"{missing_columns}"
        )
    expected_values = {
        "algorithm": {expected_algorithm} if expected_algorithm else set(),
        "method": {expected_method} if expected_method else set(),
        "h": {expected_h},
        "p": set(expected_dimensions),
        "n": {100, 1000, 10000, 100000, 1000000},
    }
    for column, expected in expected_values.items():
        observed = set(data[column].dropna().unique())
        if observed != expected:
            raise ValueError(
                f"Expected {column} values {sorted(expected)}, found "
                f"{sorted(observed)}"
            )
    expected_iteration_ids = set(range(expected_iterations))
    for (dimension, sample_size), group in data.groupby(["p", "n"]):
        observed_iterations = set(group["iteration_id"])
        complete = (
            len(group) == expected_iterations
            and observed_iterations == expected_iteration_ids
        )
        available = (
            0 < len(group) <= expected_iterations
            and len(group) == len(observed_iterations)
            and observed_iterations <= expected_iteration_ids
        )
        if not (available if allow_incomplete else complete):
            raise ValueError(
                "Expected one result for each iteration in cell "
                f"p={dimension}, n={sample_size}"
            )
    if expected_rows is not None and len(data) != expected_rows:
        raise ValueError(
            f"{result_name} dimension-scaling results must contain "
            f"{expected_rows} rows"
        )
    for metric in ["mse", "training_time_seconds"]:
        if not np.isfinite(data[metric]).all():
            raise ValueError(
                f"{result_name} dimension-scaling results must contain "
                f"finite {metric} values"
            )
        if (data[metric] <= 0).any():
            raise ValueError(
                f"{result_name} dimension-scaling results must contain "
                f"strictly positive {metric} values"
            )

# Load and validate the dedicated PMC-OLS dimension-scaling results
def _load_ols_dimension_scaling():
    paths = sorted(
        DATA_ROOT.glob(
            "gaussian/multivariate/ols_dimension_scaling/k_2/p_*/"
            "results/batch_*.csv"
        )
    )
    data = pd.concat(
        (pd.read_csv(path, low_memory=False) for path in paths),
        ignore_index=True,
    )
    _validate_dimension_scaling_results(
        data,
        result_name="PMC-OLS",
        expected_algorithm="PMC",
        expected_method="linear",
        expected_dimensions={1, 10, 100, 1000},
        expected_iterations=20,
        expected_rows=400,
    )
    return data

# Validate first-component truth against the ambient dimension
def _validate_first_component_true_mean(data, result_name):
    if "true_mean" not in data.columns:
        raise KeyError(
            f"Missing {result_name} dimension-scaling column: true_mean"
        )
    true_mean = data["true_mean"]
    expected_true_mean = 1 / np.sqrt(data["p"].astype(float))
    if not np.isfinite(true_mean).all() or not np.allclose(
        true_mean,
        expected_true_mean,
        rtol=0,
        atol=2 * np.finfo(float).eps,
    ):
        raise ValueError(
            f"{result_name} dimension-scaling results must have true_mean "
            "1 / sqrt(p)"
        )

# Load and validate the first-component PMC-OLS dimension-scaling results
def _load_first_component_ols_dimension_scaling():
    data = load_csv_batches(
        "gaussian/multivariate/first_component_ols_dimension_scaling/"
        "k_2/p_*/results/batch_*.csv"
    )
    result_name = "First-component PMC-OLS"
    _validate_dimension_scaling_results(
        data,
        result_name=result_name,
        expected_algorithm="PMC",
        expected_method="linear",
        expected_dimensions={1, 10, 100, 1000},
        expected_iterations=20,
        expected_rows=400,
        expected_h="identity",
    )
    _validate_first_component_true_mean(data, result_name)
    return data

# Load and validate the first-component Oracle dimension-scaling results
def _load_first_component_oracle_dimension_scaling():
    data = load_csv_batches(
        "gaussian/multivariate/first_component_oracle_dimension_scaling/"
        "k_2/p_*/results/batch_*.csv"
    )
    result_name = "First-component Oracle"
    _validate_dimension_scaling_results(
        data,
        result_name=result_name,
        expected_algorithm="Oracle",
        expected_method=None,
        expected_dimensions={1, 10, 100, 1000},
        expected_iterations=20,
        expected_rows=400,
        expected_h="identity",
    )
    _validate_first_component_true_mean(data, result_name)
    return data

# Load and validate available first-component MLP or GP results
def _load_first_component_nonlinear_dimension_scaling(method):
    data = load_csv_batches(
        "gaussian/multivariate/"
        "first_component_gp_dimension_scaling_fixed_updates/"
        "k_2/p_*/results/batch_*.csv"
    )
    data = data[data["method"] == method].copy()
    result_name = f"First-component PMC-{method}"
    dimensions = {1, 10, 100, 1000}
    sample_sizes = {100, 1000, 10000, 100000, 1000000}
    _validate_dimension_scaling_results(
        data,
        result_name=result_name,
        expected_algorithm="PMC",
        expected_method=method,
        expected_dimensions=dimensions,
        expected_iterations=20,
        expected_rows=None,
        expected_h="identity",
        allow_incomplete=True,
    )
    expected_cells = {
        (dimension, sample_size)
        for dimension in dimensions
        for sample_size in sample_sizes
    }
    observed_cells = set(zip(data["p"], data["n"]))
    if observed_cells != expected_cells:
        raise ValueError(
            f"{result_name} dimension-scaling results must contain every "
            "configured dimension and sample-size cell"
        )
    _validate_first_component_true_mean(data, result_name)
    return data

# Load and validate the completed PMC-MLP dimension-scaling results
def _load_mlp_dimension_scaling():
    result_directory = (
        DATA_ROOT
        / "gaussian"
        / "multivariate"
        / "mlp_dimension_scaling"
        / "k_2"
    )
    result_paths = [
        result_directory / f"p_{dimension}" / "results" / "results.csv"
        for dimension in [1, 10, 100, 1000]
    ]
    data = pd.concat(
        (pd.read_csv(path, low_memory=False) for path in result_paths),
        ignore_index=True,
    )
    _validate_dimension_scaling_results(
        data,
        result_name="PMC-MLP",
        expected_algorithm="PMC",
        expected_method="MLP",
        expected_dimensions={1, 10, 100, 1000},
        expected_iterations=20,
        expected_rows=400,
    )
    if "true_mean" in data.columns:
        true_mean = data["true_mean"]
        if not np.isfinite(true_mean).all() or not np.allclose(
            true_mean,
            1,
            rtol=0,
            atol=2 * np.finfo(float).eps,
        ):
            raise ValueError(
                "PMC-MLP dimension-scaling results must have true_mean 1"
            )
    return data

# Load and validate the dedicated PMC-GP dimension-scaling results
def _load_gp_dimension_scaling():
    data = load_csv_batches(
        "gaussian/multivariate/gp_dimension_scaling_fixed_updates/"
        "k_2/p_*/results/batch_*.csv"
    )
    data = data[data["method"] == "GP"].copy()
    _validate_dimension_scaling_results(
        data,
        result_name="PMC-GP",
        expected_algorithm="PMC",
        expected_method="GP",
        expected_dimensions={1, 10, 100, 1000},
        expected_iterations=20,
        expected_rows=400,
    )
    return data

# Load and validate the dedicated Oracle dimension-scaling results
def _load_oracle_dimension_scaling():
    data = load_csv_batches(
        "gaussian/multivariate/oracle_dimension_scaling/k_2/p_*/"
        "results/batch_*.csv"
    )
    recovery = load_csv_batches(
        "gaussian/multivariate/oracle_pseudoinverse_recovery/k_2/p_*/"
        "results/batch_*.csv"
    )
    recovery_cells = pd.MultiIndex.from_frame(
        recovery[["p", "n"]].drop_duplicates()
    )
    data_cells = pd.MultiIndex.from_frame(data[["p", "n"]])
    data = pd.concat(
        [data[~data_cells.isin(recovery_cells)], recovery],
        ignore_index=True,
    )
    _validate_dimension_scaling_results(
        data,
        result_name="Oracle",
        expected_algorithm="Oracle",
        expected_method=None,
        expected_dimensions={1, 10, 100, 1000},
        expected_iterations=20,
        expected_rows=400,
    )
    return data

# Load and validate the dedicated GBF dimension-scaling results
def _load_gbf_dimension_scaling():
    data = load_csv_batches(
        "gaussian/multivariate/gbf_dimension_scaling/k_2/p_*/"
        "n_*/results/batch_*.csv"
    )
    # Retain only complete GBF cells for the 20-replication figure.
    complete_cells = (
        data.groupby(["p", "n"])["iteration_id"].transform("count") == 20
    )
    data = data[complete_cells].copy()
    _validate_dimension_scaling_results(
        data,
        result_name="GBF",
        expected_algorithm=None,
        expected_method="GBF",
        expected_dimensions={1, 10, 100},
        expected_iterations=20,
        expected_rows=240,
    )
    return data

# Summarize an arithmetic metric with exact log-scale mean-plus-or-minus-SEM endpoints
def _summarize_arithmetic_metric(data, metric, group_columns):
    summary = (
        data.groupby(group_columns, sort=True)[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )
    lower = summary["mean"] - summary["sem"]
    upper = summary["mean"] + summary["sem"]
    if (lower <= 0).any():
        raise ValueError(
            f"Arithmetic mean minus SEM must be positive for {metric}"
        )
    summary["log_n"] = np.log10(summary["n"])
    log_mean = np.log10(summary["mean"])
    summary["log_sem_lower"] = log_mean - np.log10(lower)
    summary["log_sem_upper"] = np.log10(upper) - log_mean
    summary["mean"] = log_mean
    return summary

# Create one combined dimension-scaling figure for the requested methods
def _create_dimension_scaling_figure(data, methods):
    dimensions = [1, 10, 100, 1000]
    normalized = normalize_methods(data)
    panels = {}
    for metric_index, (metric, _) in enumerate(CONVERGENCE_METRICS):
        metric_data = normalized
        if metric == "training_time_seconds":
            metric_data = normalized[
                normalized["method_label"] != "Oracle"
            ]
        summary = _summarize_arithmetic_metric(
            metric_data,
            metric,
            ["p", "n", "method_label"],
        )
        for column_index, dimension in enumerate(dimensions):
            panels[(metric_index, 0, column_index)] = summary[
                summary["p"] == dimension
            ]
    return create_combined_figure(
        panels,
        column_levels=dimensions,
        row_levels=[None],
        column_label=r"$p$",
        row_label=None,
        methods=methods,
        y_error_column=("log_sem_lower", "log_sem_upper"),
    )

# Create the combined normalized-sum PMC dimension-scaling figure
def _create_ols_dimension_scaling_figure(data):
    return _create_dimension_scaling_figure(
        data,
        {"Oracle", "GBF", "PMC-OLS", "PMC-MLP", "PMC-GP"},
    )

# Create the combined first-component dimension-scaling figure
def _create_first_component_dimension_scaling_figure(data):
    return _create_dimension_scaling_figure(
        data,
        {"Oracle", "PMC-OLS", "PMC-MLP", "PMC-GP"},
    )

# Generate the dedicated PMC dimension-scaling figures
def generate_ols_dimension_scaling(output_root=FIGURES_ROOT):
    data = pd.concat(
        [
            _load_ols_dimension_scaling(),
            _load_mlp_dimension_scaling(),
            _load_gp_dimension_scaling(),
            _load_gbf_dimension_scaling(),
            _load_oracle_dimension_scaling(),
        ],
        ignore_index=True,
    )
    output_directory = (
        Path(output_root)
        / "gaussian"
        / "multivariate"
        / "k_2"
        / "ols_dimension_scaling"
    )
    figure = _create_ols_dimension_scaling_figure(data)
    output_path = output_directory / "normalized_sum_mse_runtime.pdf"
    return [save_figure(figure, output_path)]

# Generate the available first-component dimension-scaling figure
def generate_first_component_dimension_scaling(output_root=FIGURES_ROOT):
    data = pd.concat(
        [
            _load_first_component_ols_dimension_scaling(),
            _load_first_component_oracle_dimension_scaling(),
            _load_first_component_nonlinear_dimension_scaling("MLP"),
            _load_first_component_nonlinear_dimension_scaling("GP"),
        ],
        ignore_index=True,
    )
    output_directory = (
        Path(output_root)
        / "gaussian"
        / "multivariate"
        / "k_2"
        / "first_component_ols_dimension_scaling"
    )
    figure = _create_first_component_dimension_scaling_figure(data)
    output_path = output_directory / "first_component_mse_runtime.pdf"
    return [save_figure(figure, output_path)]

# Preserve the original first-component CLI generator name
def generate_first_component_ols_dimension_scaling(output_root=FIGURES_ROOT):
    return generate_first_component_dimension_scaling(output_root)

##############################
# Bivariate Gaussian figures #
##############################

# Generate every complete bivariate Gaussian convergence figure set.
def generate_bivariate_gaussian_k2(output_root=FIGURES_ROOT):
    configurations = [
        ("mean_shift", "mu_2_1", r"$\mu$", False),
        ("variance_scaling", "cov_2_1_1", r"$\sigma$", True),
        ("anisotropic_variance", "cov_2_2_2", r"$\sigma_2$", True),
        ("correlation", "cov_2_1_2", r"$\rho$", False),
    ]
    generated_paths = []
    for scenario, facet_column, facet_column_label, is_variance in configurations:
        data = normalize_methods(
            load_csv_batches(
                f"gaussian/bivariate/{scenario}/k_2/results/n_*/batch_*.csv"
            )
        )
        if is_variance:
            # Display the scale as a standard deviation rather than a variance
            data[facet_column] = np.sqrt(data[facet_column]).round(10)
            # Use integers for whole-number scales so labels omit decimals
            if (data[facet_column] % 1 == 0).all():
                data[facet_column] = data[facet_column].astype(int)
        deduplication_columns = [
            column
            for column in [
                "n",
                "n_sample",
                facet_column,
                "method_label",
                "h",
                "iteration_id",
                "seed",
            ]
            if column in data.columns
        ]
        data = data.drop_duplicates(subset=deduplication_columns)
        output_directory = (
            Path(output_root)
            / "gaussian"
            / "bivariate"
            / "k_2"
            / scenario
        )
        generated_paths.extend(
            generate_convergence_set(
                data,
                output_directory,
                facet_column=facet_column,
                facet_column_label=facet_column_label,
            )
        )
    return generated_paths

################
# Beta figures #
################

# Load one Beta scenario with normalized methods and no duplicate rows.
def load_beta_scenario(scenario):
    data = normalize_methods(
        load_csv_batches(f"beta/{scenario}/k_2/results/n_*/batch_*.csv")
    )
    deduplication_columns = [
        column
        for column in [
            "n",
            "n_sample",
            "method_label",
            "h",
            "iteration_id",
            "seed",
        ]
        if column in data.columns
    ]
    return data.drop_duplicates(subset=deduplication_columns)

# Generate the faceted Beta concentration figure set with one column per a.
def generate_beta_concentration_figures(output_root=FIGURES_ROOT):
    scenarios = [
        "uniform",
        "baseline_symmetric",
        "increased_concentration",
    ]
    data = pd.concat(
        [load_beta_scenario(scenario) for scenario in scenarios],
        ignore_index=True,
    )
    data["a_1"] = data["a_1"].astype(int)
    return generate_convergence_set(
        data,
        Path(output_root) / "beta" / "concentration",
        facet_column="a_1",
        facet_column_label=r"$a$",
    )

# Generate every complete Beta convergence figure set in paper order.
def generate_beta_figures(output_root=FIGURES_ROOT):
    scenarios = [
        "heterogeneous_concentration",
        "matching_skew",
        "opposing_skew",
        "endpoint_singular_stress",
    ]
    generated_paths = generate_beta_concentration_figures(output_root)
    for scenario in scenarios:
        generated_paths.extend(
            generate_convergence_set(
                load_beta_scenario(scenario),
                Path(output_root) / "beta" / scenario,
            )
        )
    return generated_paths

############################
# Gaussian mixture figures #
############################

# Generate every complete Gaussian mixture convergence figure set.
def generate_gaussian_mixture_figures(output_root=FIGURES_ROOT):
    configurations = [
        (2, "symmetric_bimodal"),
        (2, "heterogeneous_unequal"),
        (2, "weak_overlap_narrow"),
        (2, "opposing_mode_weights"),
        (2, "heterogeneous_scale_unimodal"),
        (3, "mixed_components"),
    ]
    required_columns = [
        "n",
        "status",
        "method",
        "target",
        "squared_error",
        "standalone_runtime_seconds",
    ]
    generated_paths = []
    for num_nodes, scenario in configurations:
        data = load_csv_batches(
            f"gaussian/mixture/{scenario}/k_{num_nodes}/results/"
            "n_*/batch_*.csv"
        )
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing Gaussian mixture columns for {scenario}: "
                f"{missing_columns}"
            )

        data = data[data["status"] == "success"]
        data = normalize_methods(data).rename(
            columns={
                "target": "h",
                "squared_error": "mse",
                "standalone_runtime_seconds": "training_time_seconds",
            }
        )
        output_directory = (
            Path(output_root)
            / "gaussian"
            / "mixture"
            / f"k_{num_nodes}"
            / scenario
        )
        generated_paths.extend(
            generate_convergence_set(data, output_directory)
        )
    return generated_paths

############################
# Fusion benchmark figures #
############################

# Generate every complete fusion benchmark convergence figure set.
def generate_fusion_benchmark_figures(output_root=FIGURES_ROOT):
    configurations = [
        (
            "univariate",
            "fusion_benchmark/univariate/location_scale_grid/k_2/results/"
            "n_*/batch_*.csv",
            "sigma_2",
            "mu_2",
            r"$\sigma_2$",
            r"$\mu_2$",
        ),
        (
            "bivariate",
            "fusion_benchmark/bivariate/mean_shift/k_2/results/"
            "n_*/batch_*.csv",
            None,
            "mu_2_1",
            None,
            r"$\mu$",
        ),
        (
            "multivariate",
            "fusion_benchmark/multivariate/dimension_scaling/k_2/p_*/"
            "results/n_*/batch_*.csv",
            None,
            "p",
            None,
            "p",
        ),
    ]
    generated_paths = []
    for (
        family,
        pattern,
        facet_row,
        facet_column,
        facet_row_label,
        facet_column_label,
    ) in configurations:
        data = load_csv_batches(pattern)
        facets = [
            column for column in [facet_row, facet_column] if column is not None
        ]
        required_columns = [
            "n",
            "method",
            "h",
            "mse",
            "simulation_time_seconds",
            *facets,
        ]
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing fusion benchmark columns for {family}: "
                f"{missing_columns}"
            )

        data = normalize_methods(data)
        data["training_time_seconds"] = data["simulation_time_seconds"]
        execution_columns = [
            column
            for column in [
                "method",
                "n",
                "n_sample",
                *facets,
                "h",
                "iteration_id",
                "seed",
                "data_seed",
                "inference_seed",
            ]
            if column in data.columns
        ]
        duplicate_runtime_rows = data.duplicated(
            subset=execution_columns, keep="first"
        )
        data.loc[duplicate_runtime_rows, "training_time_seconds"] = np.nan
        output_directory = (
            Path(output_root) / "fusion_methods_benchmark" / family
        )
        generated_paths.extend(
            generate_convergence_set(
                data,
                output_directory,
                facet_row=facet_row,
                facet_column=facet_column,
                facet_row_label=facet_row_label,
                facet_column_label=facet_column_label,
            )
        )
    return generated_paths

########################
# Scaling data figures #
########################

# Validate and collapse repeated coordinate rows for one scaling dataset.
def _collapse_scaling_rows(data, facet_column, context):
    execution_columns = [
        column
        for column in [
            "iteration_id",
            "n",
            facet_column,
            "method_label",
            "h",
        ]
        if column in data.columns
    ]
    metric_columns = ["mse", "training_time_seconds"]
    metric_counts = data.groupby(
        execution_columns, dropna=False
    )[metric_columns].nunique(dropna=False)
    if (metric_counts > 1).any().any():
        raise ValueError(
            f"{context} coordinate rows disagree within an execution"
        )
    return data.drop_duplicates(subset=execution_columns)

# Generate complete dimension- and node-scaling figure sets.
def generate_scaling_figures(output_root=FIGURES_ROOT):
    configurations = [
        (
            "dimension",
            "gaussian/multivariate/dimension_scaling/k_2/p_*/results/"
            "n_*/batch_*.csv",
            "p",
            r"$p$",
            Path(output_root)
            / "gaussian"
            / "multivariate"
            / "k_2"
            / "dimension_scaling",
            "dimension_scaling",
        ),
        (
            "homogeneous",
            "gaussian/univariate/k_scaling/homogeneous/k_*/results/"
            "n_*/batch_*.csv",
            "num_nodes",
            r"$K$",
            Path(output_root)
            / "gaussian"
            / "univariate"
            / "k_scaling"
            / "homogeneous",
            "k_scaling",
        ),
        (
            "heterogeneous",
            "gaussian/univariate/k_scaling/heterogeneous/k_*/results/"
            "n_*/batch_*.csv",
            "num_nodes",
            r"$K$",
            Path(output_root)
            / "gaussian"
            / "univariate"
            / "k_scaling"
            / "heterogeneous",
            "k_scaling",
        ),
    ]
    generated_paths = []
    for (
        context,
        pattern,
        facet_column,
        facet_label,
        output_directory,
        filename_prefix,
    ) in configurations:
        data = load_csv_batches(pattern)
        required_columns = [
            "n",
            "method",
            "h",
            "mse",
            "training_time_seconds",
            facet_column,
        ]
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing scaling columns for {context}: {missing_columns}"
            )

        data = _collapse_scaling_rows(
            normalize_methods(data), facet_column, context
        )
        data = data[data[facet_column] <= MAX_SCALING_VALUE]
        for target in ["identity", "square"]:
            figure = create_scaling_figure(
                data,
                target,
                facet_column,
                facet_label,
            )
            generated_paths.append(
                save_figure(
                    figure,
                    output_directory / f"{filename_prefix}_h_{target}.pdf",
                )
            )
    return generated_paths

##########################
# Command-line interface #
##########################

# Regenerate all convergence figure families or one selected family.
def main():
    generators = {
        "univariate-gaussian": generate_univariate_gaussian_k2,
        "bivariate-gaussian": generate_bivariate_gaussian_k2,
        "beta": generate_beta_figures,
        "gaussian-mixture": generate_gaussian_mixture_figures,
        "fusion-benchmark": generate_fusion_benchmark_figures,
        "scaling": generate_scaling_figures,
        "ols-dimension-scaling": generate_ols_dimension_scaling,
        "first-component-dimension-scaling": (
            generate_first_component_dimension_scaling
        ),
        "first-component-ols-dimension-scaling": (
            generate_first_component_ols_dimension_scaling
        ),
    }
    parser = argparse.ArgumentParser(
        description="Regenerate convergence figures used in the paper."
    )
    parser.add_argument(
        "--family",
        choices=["all", *generators],
        default="all",
        help="Figure family to regenerate (default: all).",
    )
    arguments = parser.parse_args()
    selected_generators = (
        [
            generator
            for name, generator in generators.items()
            if name not in {
                "ols-dimension-scaling",
                "first-component-dimension-scaling",
                "first-component-ols-dimension-scaling",
            }
        ]
        if arguments.family == "all"
        else [generators[arguments.family]]
    )
    for generator in selected_generators:
        generator()

if __name__ == "__main__":
    main()
