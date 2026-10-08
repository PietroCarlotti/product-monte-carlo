##############################
# Import necessary libraries #
##############################

# Standard libraries
import os

# Third-party libraries
import matplotlib
import numpy as np
import torch
from scipy.stats import beta, norm

# Configure matplotlib for headless rendering before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PMC library
from .utils import (
    bivariate_normal_density_grid,
    gaussian_marginal_1d,
    gaussian_slice_2d,
    gaussian_x_range,
)

###################################
# Plot save/show and grid helpers #
###################################

# Saves the figure if requested, then either displays it or closes it
def _finalize_plot(fig, show, save, filename, plots_directory, bbox_inches=None):
    # Save the figure to disk if requested
    if save:
        fig.savefig(os.path.join(plots_directory, filename), bbox_inches=bbox_inches)
    # Display the figure interactively, or close it to free its memory if not shown
    if show:
        plt.show()
    else:
        plt.close(fig)

# Builds a p x p subplot grid, delegating diagonal/lower-triangle panels to callbacks
def _plot_pxp_grid(p, diagonal_fn, lower_triangle_fn, figsize_scale=3):
    # Create the p x p grid of subplots
    fig, axes = plt.subplots(
        p, p, figsize=(p * figsize_scale, p * figsize_scale), squeeze=False
    )

    # Dispatch each cell to the diagonal or lower-triangle panel, leaving the upper triangle blank
    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if i == j:
                diagonal_fn(ax, i)
            elif i > j:
                lower_triangle_fn(ax, i, j)
            else:
                ax.axis('off')

    fig.tight_layout()
    return fig, axes

# Plots the true pdf or cdf curve (per mode) for the given scipy.stats distribution
def _plot_overlay_curve(ax, x, dist, mode, *params):
    values = dist.cdf(x, *params) if mode == 'cdf' else dist.pdf(x, *params)
    ax.plot(x, values, color='blue', linestyle='--', alpha=0.7)

# Plots a 1D density curve over the given grid, with standard product-distribution styling
def _plot_1d_density(x_values, pdf_values):
    # Create the figure and axes for the density plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the density curve over the grid
    ax.plot(x_values, pdf_values, label='Product Distribution', color='blue', alpha=0.7)

    # Apply the standard title, axis labels, legend, and grid styling
    ax.set_title('Product Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    return fig, ax

# Plots a 2D density contour over the given grid, with standard product-distribution styling
def _plot_2d_density(x, y, density):
    # Create the figure and axes for the density plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the density contours over the grid
    ax.contourf(x, y, density, cmap='Blues')

    # Apply the standard title and axis labels
    ax.set_title('Product Distribution')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    return fig, ax

# Builds a 1D grid over dimension i and evaluates the estimator's density, other dims held at 0
def _estimator_density_1d(estimator, i, p, vmin, vmax, discrepancy_condition, n_points=100):
    # Build the 1D evaluation grid over dimension i
    grid = np.linspace(vmin, vmax, n_points)

    # Hold all other dimensions at zero, only dimension i varies
    density_grid = np.zeros((n_points, p))
    density_grid[:, i] = grid

    # Evaluate the estimator's log-density on the grid
    log_p_density_grid = estimator.log_prob(density_grid, discrepancy_condition)
    return grid, np.exp(log_p_density_grid)

######################
# Distribution plots #
######################

# Function to plot the product distribution
def plot_product_distribution(parametric_form, parameters, show=False, save=False, filename='product_distribution.pdf', plots_directory='plots'):
    if parametric_form == 'univariate_Gaussian':
        # Extract parameters for univariate Gaussian
        mu = parameters['mu']
        sigma = parameters['sigma']

        # Create a grid of values around the product distribution mean
        x_values = gaussian_x_range(mu, sigma, n_points=1000)

        # Compute the PDF of the product distribution
        pdf_values = norm.pdf(x_values, loc=mu, scale=sigma)

        # Create the plot
        fig, ax = _plot_1d_density(x_values, pdf_values)

    elif parametric_form == 'bivariate_Gaussian':
        # Extract parameters for bivariate Gaussian
        mu = parameters['mu']
        cov = parameters['cov']

        # Compute the PDF of the product distribution over a 2D grid
        x, y, density = bivariate_normal_density_grid(mu, cov)

        # Create the plot
        fig, ax = _plot_2d_density(x, y, density)

    elif parametric_form == 'multivariate_Gaussian':
        # Extract parameters for multivariate Gaussian
        mu = parameters['mu']
        cov = parameters['cov']
        p = len(mu)

        # Diagonal panel: plot the marginal distribution for dimension i
        def _diagonal(ax, i):
            mu_1d, sigma_1d = gaussian_marginal_1d(mu, cov, i)
            x = gaussian_x_range(mu_1d, sigma_1d)
            ax.plot(x, norm.pdf(x, mu_1d, sigma_1d), color='blue', alpha=0.7)
            ax.set_title(f'Dim {i+1}')

        # Lower-triangle panel: plot the joint contours for dimensions i, j
        def _lower_triangle(ax, i, j):
            mu_2d, sigma_2d = gaussian_slice_2d(mu, cov, i, j)

            # Create local grid for these two dimensions
            x, y, density = bivariate_normal_density_grid(mu_2d, sigma_2d)
            ax.contourf(x, y, density, cmap='Blues')
            if i == p - 1: ax.set_xlabel(f'Dim {j+1}')
            if j == 0: ax.set_ylabel(f'Dim {i+1}')

        fig, axes = _plot_pxp_grid(p, _diagonal, _lower_triangle)

        # Match the previous behaviour of returning the last-visited panel
        ax = axes[p - 1, p - 1]

    elif parametric_form == 'Beta':
        # Extract parameters for Beta distribution
        a = parameters['a']
        b = parameters['b']

        # Create a grid of values between 0 and 1
        x_values = np.linspace(0, 1, 1000)

        # Compute the PDF of the product distribution (which is also a Beta distribution)
        pdf_values = beta.pdf(x_values, a, b)

        # Create the plot
        fig, ax = _plot_1d_density(x_values, pdf_values)

    # Save/show/close the plot as requested
    _finalize_plot(fig, show, save, filename, plots_directory)

    # Return the plot objects for potential further manipulation
    return fig, ax

# Function to plot the distribution of Consensus and Discrepancy
def plot_discrepancy_distribution(discrepancy, show=False, save=False,
                                   filename='D_distribution.pdf', plots_directory='plots'):

    # Determine the dimensionality of discrepancy
    p_discrepancy = discrepancy.shape[1]

    # Diagonal panel: plot the marginal distribution of discrepancy dim i
    def _diagonal(ax, i):
        ax.hist(discrepancy[:, i], bins=30, density=True, color='red', alpha=0.7)
        ax.axvline(0, color='orange', linestyle='--', linewidth=2)
        ax.set_title(f'Dimension {i+1}')

    # Lower-triangle panel: plot the 2D hexbin for discrepancy dims i, j
    def _lower_triangle(ax, i, j):
        ax.hexbin(
            discrepancy[:, j], discrepancy[:, i],
            gridsize=30, cmap='Reds', mincnt=1
        )
        if i == p_discrepancy - 1: ax.set_xlabel(f'Dim {j+1}')
        if j == 0: ax.set_ylabel(f'Dim {i+1}')

    fig, axes = _plot_pxp_grid(p_discrepancy, _diagonal, _lower_triangle)

    # Save/show/close the plot as requested
    _finalize_plot(fig, show, save, filename, plots_directory)

    return fig, axes

# Function to plot the training loss
def plot_training_loss(training_loss, show=False, save=False, filename='training_loss.pdf', plots_directory='plots'):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(training_loss, label='Training Loss', color='purple')
    ax.set_title('Training Loss History')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    # Keep zero visible on the y-axis even when the loss never goes negative
    ax.set_ylim(bottom=min(0, min(training_loss)))

    # Save/show/close the plot as requested
    _finalize_plot(fig, show, save, filename, plots_directory)

    return fig, ax

# Function to plot the inference distribution
def plot_inference_distribution(consensus_inference, mode, estimator=None,
                                 discrepancy_condition=None, dgp=None, show=False, save=False,
                                 filename='training_loss.pdf', plots_directory='plots'):

    # Ensure consensus_inference is a numpy array for matplotlib compatibility
    if torch.is_tensor(consensus_inference):
        consensus_inference = consensus_inference.detach().cpu().numpy()

    # Normalize shape to (n, p) for downstream plotting logic.
    if consensus_inference.ndim == 1:
        consensus_inference = consensus_inference[:, None]
    elif consensus_inference.ndim == 3 and consensus_inference.shape[1] == 1:
        consensus_inference = consensus_inference[:, 0, :]
    elif consensus_inference.ndim != 2:
        raise ValueError(
            f"consensus_inference must have shape (n, p), got {consensus_inference.shape}"
        )

    # Determine the dimensionality of the inferred distribution
    p = consensus_inference.shape[1]

    # Per-dimension ranges used to build the plotting grids below
    consensus_min, consensus_max = consensus_inference.min(axis=0), consensus_inference.max(axis=0)

    # Diagonal panel: marginal plot for dimension i
    def _diagonal(ax, i):
        # Plot histogram of inferred samples
        if mode == 'samples':
            ax.hist(
                consensus_inference[:, i], bins=100, density=True,
                color='green', alpha=0.7
            )

        # Plot the density from the estimator
        elif mode == 'pdf':
            if estimator is not None:
                consensus_grid, p_consensus_density_grid = _estimator_density_1d(
                    estimator, i, p, consensus_min[i], consensus_max[i], discrepancy_condition
                )

                ax.plot(consensus_grid, p_consensus_density_grid, color='green')

        # Plot the cumulative distribution
        elif mode == 'cdf':
            if estimator is not None:
                consensus_grid, p_consensus_density_grid = _estimator_density_1d(
                    estimator, i, p, consensus_min[i], consensus_max[i], discrepancy_condition
                )

                # Numerical integration
                d_consensus = consensus_grid[1] - consensus_grid[0]
                p_consensus_cdf_grid = np.cumsum(p_consensus_density_grid) * d_consensus
                if p_consensus_cdf_grid[-1] > 0:
                    p_consensus_cdf_grid /= p_consensus_cdf_grid[-1]

                ax.plot(consensus_grid, p_consensus_cdf_grid, color='green')
            # Plot the empirical cumulative distribution from the samples
            else:
                consensus_sorted = np.sort(consensus_inference[:, i])
                p_consensus_cdf_grid = np.linspace(0, 1, len(consensus_sorted))
                ax.step(
                    consensus_sorted, p_consensus_cdf_grid, where='post', color='green'
                )

        ax.set_title(f'Dimension {i+1}')

        # Overlay the true marginal distribution if dgp is provided
        if dgp is not None:
            if (
                dgp.parametric_form == 'univariate_Gaussian'
                or dgp.parametric_form == 'bivariate_Gaussian'
                or dgp.parametric_form == 'multivariate_Gaussian'
            ):
                # Univariate Gaussian: the marginal is the product distribution itself
                if dgp.parametric_form == 'univariate_Gaussian':
                    mu_1d = dgp.mu_product
                    sigma_1d = dgp.sigma_product

                # Bivariate Gaussian: marginal for dimension i
                elif dgp.parametric_form == 'bivariate_Gaussian':
                    mu_1d, sigma_1d = gaussian_marginal_1d(dgp.mu_product, dgp.cov_product, i)

                # Multivariate Gaussian: marginal for dimension i
                elif dgp.parametric_form == 'multivariate_Gaussian':
                    mu_1d, sigma_1d = gaussian_marginal_1d(dgp.mu_product, dgp.cov_product, i)

                x = gaussian_x_range(mu_1d, sigma_1d)
                _plot_overlay_curve(ax, x, norm, mode, mu_1d, sigma_1d)

            # Beta product distribution
            elif dgp.parametric_form == 'Beta':
                a, b = dgp.a_product, dgp.b_product
                x = np.linspace(0, 1, 100)
                _plot_overlay_curve(ax, x, beta, mode, a, b)

    # Lower-triangle panel: bivariate plot for dimensions i, j
    def _lower_triangle(ax, i, j):
        # Plot histogram of inferred samples
        if mode == 'samples' or (mode == 'cdf' and estimator is None):
            ax.hexbin(
                consensus_inference[:, j], consensus_inference[:, i],
                gridsize=30, cmap='Greens', mincnt=1
            )

        # Plot the density from the estimator
        elif mode == 'pdf' or mode == 'cdf':
            if estimator is not None:
                # Build a 2D grid over dimensions j (x-axis) and i (y-axis)
                consensus_grid_component_1 = np.linspace(
                    consensus_min[j], consensus_max[j], 50
                )
                consensus_grid_component_2 = np.linspace(
                    consensus_min[i], consensus_max[i], 50
                )
                consensus_mesh_component_1, consensus_mesh_component_2 = np.meshgrid(
                    consensus_grid_component_1, consensus_grid_component_2
                )

                # Hold all other dimensions at zero, only dimensions i and j vary
                consensus_density_grid = np.zeros((2500, p))
                consensus_density_grid[:, j] = consensus_mesh_component_1.ravel()
                consensus_density_grid[:, i] = consensus_mesh_component_2.ravel()

                # Evaluate the estimator's log-density on the grid
                log_p_consensus_density_grid = estimator.log_prob(
                    consensus_density_grid, discrepancy_condition
                )

                # Convert log-density to density, reshaped to the mesh grid shape
                p_consensus_density_grid = np.exp(
                    log_p_consensus_density_grid
                ).reshape(consensus_mesh_component_1.shape)

                ax.contourf(
                    consensus_mesh_component_1, consensus_mesh_component_2,
                    p_consensus_density_grid, cmap='Greens'
                )

        # Overlay the true marginal distribution if dgp is provided
        if dgp is not None and dgp.parametric_form == 'multivariate_Gaussian':
            mu_2d, sigma_2d = gaussian_slice_2d(dgp.mu_product, dgp.cov_product, i, j)

            x, y, density = bivariate_normal_density_grid(mu_2d, sigma_2d)
            ax.contour(x, y, density, colors='blue', linestyles='--', alpha=0.7, linewidths=0.5)
        elif dgp is not None and dgp.parametric_form == 'bivariate_Gaussian':
            mu_2d, sigma_2d = gaussian_slice_2d(dgp.mu_product, dgp.cov_product, i, j)

            x, y, density = bivariate_normal_density_grid(mu_2d, sigma_2d)
            ax.contour(x, y, density, colors='blue', linestyles='--', alpha=0.7, linewidths=0.5)

        if i == p - 1: ax.set_xlabel(f'Dimension {j+1}')
        if j == 0: ax.set_ylabel(f'Dimension {i+1}')

    fig, axes = _plot_pxp_grid(p, _diagonal, _lower_triangle)

    # Save/show/close the plot as requested
    _finalize_plot(fig, show, save, filename, plots_directory)

    return fig, axes
