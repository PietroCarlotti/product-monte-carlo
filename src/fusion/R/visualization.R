############
# Plotting #
############

#' Assemble a p x p panel grid, delegating diagonal/lower-triangle panels
#'
#' Diagonal panels show marginals, lower-triangle panels show pairwise
#' joints, and upper-triangle panels are left blank.
#'
#' @param p number of dimensions
#' @param diagonal_fn function(i) returning the diagonal panel for dimension i
#' @param lower_triangle_fn function(i, j) returning the joint panel for
#'   dimensions i, j
#' @return patchwork object combining all panels
#' @keywords internal
.build_pxp_grid <- function(p, diagonal_fn, lower_triangle_fn) {
  # Pre-allocate the flat list of p x p panels
  panels <- vector("list", p * p)

  # Dispatch cells to diagonal/lower-triangle panels, blanking upper triangle
  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      index <- (i - 1) * p + j
      if (i == j) {
        panels[[index]] <- diagonal_fn(i)
      } else if (i > j) {
        panels[[index]] <- lower_triangle_fn(i, j)
      } else {
        panels[[index]] <- patchwork::plot_spacer()
      }
    }
  }

  # Compose the panels into the final p x p grid
  patchwork::wrap_plots(panels, nrow = p, ncol = p)
}

#' Saves the plot if requested, then prints it if requested
#'
#' @param p ggplot or patchwork object to finalize
#' @param show logical; if TRUE, print the plot
#' @param save logical; if TRUE, save the plot to disk
#' @param filename output filename (used when save = TRUE)
#' @param plots_directory directory to save to (used when save = TRUE)
#' @param width ggsave width
#' @param height ggsave height
#' @return p, invisibly
#' @keywords internal
.finalize_plot <- function(p, show, save, filename, plots_directory,
                           width, height) {
  # Save the figure to disk if requested
  if (save) {
    ggsave(
      file.path(plots_directory, filename), plot = p,
      width = width, height = height
    )
  }
  # Display the figure interactively if requested
  if (show) print(p)
  invisible(p)
}

#' Build the diagonal marginal panel for dimension i of the inference grid
#'
#' @param samples n x p matrix of posterior samples
#' @param dgp list returned by a `bivariate_gaussian_dgp()` or
#'   `multivariate_gaussian_dgp()` constructor
#' @param i dimension index (1-based)
#' @return ggplot object for the marginal panel
#' @keywords internal
.diagonal_inference_panel <- function(samples, dgp, i) {
  # Get the marginal mean/sd for dimension i and build the plotting x-range
  marginal <- .gaussian_marginal_1d(dgp$mu_product, dgp$cov_product, i)
  mu_1d <- marginal$mu_1d
  sigma_1d <- marginal$sigma_1d
  x_seq <- .gaussian_x_range(mu_1d, sigma_1d)
  x_min <- x_seq[1]
  x_max <- x_seq[length(x_seq)]

  # Build data frames for the empirical samples and the true marginal density
  df_samples <- data.frame(x = samples[, i])
  df_target <- data.frame(x = x_seq, pdf = dnorm(x_seq, mu_1d, sigma_1d))

  # Plot a histogram of the samples overlaid with the true density curve
  ggplot(df_samples, aes(x = .data$x)) +
    geom_histogram(
      aes(y = after_stat(density)),
      bins = 100,
      fill = "steelblue",
      colour = "white",
      alpha = 0.7
    ) +
    geom_line(
      data = df_target,
      aes(x = .data$x, y = .data$pdf),
      colour = "red",
      linewidth = 1
    ) +
    xlim(x_min, x_max) +
    labs(x = paste0("Dimension ", i), y = "Density") +
    theme_bw()
}

#' Build the lower-triangle joint panel for a pair of dimensions of the
#' inference grid
#'
#' @param samples n x p matrix of posterior samples
#' @param dgp list returned by a `bivariate_gaussian_dgp()` or
#'   `multivariate_gaussian_dgp()` constructor
#' @param i row dimension index (1-based)
#' @param j column dimension index (1-based)
#' @return ggplot object for the joint panel
#' @keywords internal
.lower_triangle_panel <- function(samples, dgp, i, j) {
  # Build a data frame of the sample pair for dimensions j (x) and i (y)
  df_samples <- data.frame(x = samples[, j], y = samples[, i])

  # Get the 2D Gaussian slice for dims i, j and build the true-density grid
  slice <- .gaussian_slice_2d(dgp$mu_product, dgp$cov_product, i, j)
  df_density <- .bivariate_normal_density_grid(slice$mu_2d, slice$cov_2d)

  # Plot a hexbin of the samples overlaid with the true density contour
  ggplot() +
    geom_hex(data = df_samples, aes(x = .data$x, y = .data$y), bins = 30) +
    scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
    geom_contour(
      data = df_density,
      aes(x = .data$x, y = .data$y, z = .data$density),
      colour = "blue", linetype = "dashed", alpha = 0.7
    ) +
    labs(x = paste0("Dimension ", j), y = paste0("Dimension ", i)) +
    theme_bw() +
    theme(legend.position = "none")
}

#' Assemble the p x p panel grid for a multivariate-Gaussian inference plot
#'
#' Diagonal panels show marginals, lower-triangle panels show pairwise
#' joints, and upper-triangle panels are left blank.
#'
#' @param samples n x p matrix of posterior samples
#' @param dgp list returned by a `bivariate_gaussian_dgp()` or
#'   `multivariate_gaussian_dgp()` constructor
#' @return patchwork object combining all panels
#' @keywords internal
# Bind samples/dgp into per-panel closures and delegate to the grid builder
.build_inference_grid <- function(samples, dgp) {
  diagonal_fn <- function(i) .diagonal_inference_panel(samples, dgp, i)
  lower_triangle_fn <- function(i, j) .lower_triangle_panel(samples, dgp, i, j)
  .build_pxp_grid(dgp$p, diagonal_fn, lower_triangle_fn)
}

#' Plot the fusion distribution against the true product distribution
#'
#' @param samples numeric vector of fusion draws (or an n x p matrix for
#'   bivariate/multivariate Gaussian DGPs)
#' @param dgp list returned by a `*_dgp()` constructor
#' @param show logical; if TRUE, print the plot
#' @param save logical; if TRUE, save the plot to disk
#' @param filename output filename (used when save = TRUE)
#' @param plots_directory directory to save to (used when save = TRUE)
#' @return ggplot object (invisibly)
#' @import ggplot2
#' @export
plot_inference_distribution <- function(
  samples,
  dgp,
  show = TRUE,
  save = FALSE,
  filename = "inference_distribution.pdf",
  plots_directory = "plots"
) {
  if (dgp$parametric_form == "univariate_Gaussian") {
    # Build the plotting x-range from the true univariate marginal
    x_seq <- .gaussian_x_range(dgp$mu_product, dgp$sigma_product)
    x_min <- x_seq[1]
    x_max <- x_seq[length(x_seq)]

    # Build data frames for the empirical samples and the true density
    df_samples <- data.frame(x = samples)
    df_target <- data.frame(
      x = x_seq,
      pdf = dnorm(x_seq, mean = dgp$mu_product, sd = dgp$sigma_product)
    )

    # Plot a histogram of the samples overlaid with the true density curve
    p <- ggplot(df_samples, aes(x = .data$x)) +
      geom_histogram(
        aes(y = after_stat(density)),
        bins = 100,
        fill = "steelblue",
        colour = "white",
        alpha = 0.7
      ) +
      geom_line(
        data = df_target,
        aes(x = .data$x, y = .data$pdf),
        colour = "red",
        linewidth = 1
      ) +
      xlim(x_min, x_max) +
      labs(x = "x", y = "Density") +
      theme_bw()

    plot_width <- 8
    plot_height <- 5

  } else if (dgp$parametric_form %in%
               c("bivariate_Gaussian", "multivariate_Gaussian")) {
    # Delegate to the shared p x p panel grid for bivariate/multivariate dims
    p <- .build_inference_grid(samples, dgp)
    plot_width <- dgp$p * 3
    plot_height <- dgp$p * 3

  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }

  .finalize_plot(p, show, save, filename, plots_directory,
                 plot_width, plot_height)
}

#' Build the diagonal marginal panel for dimension i of the product-
#' distribution grid
#'
#' @param parameters list with mu (vector) and cov (matrix)
#' @param i dimension index (1-based)
#' @return ggplot object for the marginal panel
#' @keywords internal
.diagonal_product_panel <- function(parameters, i) {
  # Get the marginal mean/sd for dimension i and build the plotting x-range
  marginal <- .gaussian_marginal_1d(parameters$mu, parameters$cov, i)
  mu_1d <- marginal$mu_1d
  sigma_1d <- marginal$sigma_1d
  x_seq <- .gaussian_x_range(mu_1d, sigma_1d)
  x_min <- x_seq[1]
  x_max <- x_seq[length(x_seq)]

  # Build a data frame for the true marginal density
  df_target <- data.frame(x = x_seq, pdf = dnorm(x_seq, mu_1d, sigma_1d))

  # Plot the true density curve
  ggplot(df_target, aes(x = .data$x, y = .data$pdf)) +
    geom_line(colour = "red", linewidth = 1) +
    xlim(x_min, x_max) +
    labs(x = paste0("Dimension ", i), y = "Density") +
    theme_bw()
}

#' Build the lower-triangle joint panel for a pair of dimensions of the
#' product-distribution grid
#'
#' @param parameters list with mu (vector) and cov (matrix)
#' @param i row dimension index (1-based)
#' @param j column dimension index (1-based)
#' @return ggplot object for the joint panel
#' @keywords internal
.lower_triangle_product_panel <- function(parameters, i, j) {
  # Get the 2D Gaussian slice for dims i, j and build the true-density grid
  slice <- .gaussian_slice_2d(parameters$mu, parameters$cov, i, j)
  df_density <- .bivariate_normal_density_grid(slice$mu_2d, slice$cov_2d)

  # Plot the true density as a filled contour
  ggplot(df_density, aes(x = .data$x, y = .data$y, z = .data$density)) +
    geom_contour_filled(bins = 6) +
    scale_fill_brewer(palette = "Blues") +
    labs(x = paste0("Dimension ", j), y = paste0("Dimension ", i)) +
    theme_bw() +
    theme(legend.position = "none")
}

#' Assemble the p x p panel grid for a multivariate-Gaussian product-
#' distribution plot
#'
#' Diagonal panels show marginals, lower-triangle panels show pairwise
#' joints, and upper-triangle panels are left blank.
#'
#' @param parameters list with mu (vector) and cov (matrix)
#' @return patchwork object combining all panels
#' @keywords internal
# Bind parameters into per-panel closures and delegate to the grid builder
.build_product_grid <- function(parameters) {
  diagonal_fn <- function(i) .diagonal_product_panel(parameters, i)
  lower_triangle_fn <- function(i, j) {
    .lower_triangle_product_panel(parameters, i, j)
  }
  .build_pxp_grid(length(parameters$mu), diagonal_fn, lower_triangle_fn)
}

#' Plot the analytic true product distribution
#'
#' Unlike `plot_inference_distribution`, no samples are involved: this
#' visualises only the analytic density implied by `parameters`.
#'
#' @param parametric_form one of "univariate_Gaussian", "bivariate_Gaussian",
#'   or "multivariate_Gaussian"
#' @param parameters list with mu/sigma (univariate) or mu/cov
#'   (bivariate/multivariate)
#' @param show logical; if TRUE, print the plot
#' @param save logical; if TRUE, save the plot to disk
#' @param filename output filename (used when save = TRUE)
#' @param plots_directory directory to save to (used when save = TRUE)
#' @return ggplot object (invisibly)
#' @import ggplot2
#' @export
plot_product_distribution <- function(
  parametric_form,
  parameters,
  show = TRUE,
  save = FALSE,
  filename = "product_distribution.pdf",
  plots_directory = "plots"
) {
  if (parametric_form == "univariate_Gaussian") {
    # Extract parameters and build the plotting x-range
    mu <- parameters$mu
    sigma <- parameters$sigma
    x_seq <- .gaussian_x_range(mu, sigma)
    x_min <- x_seq[1]
    x_max <- x_seq[length(x_seq)]

    # Build a data frame for the true density
    df_target <- data.frame(
      x = x_seq, pdf = dnorm(x_seq, mean = mu, sd = sigma)
    )

    # Plot the true density curve
    p <- ggplot(df_target, aes(x = .data$x, y = .data$pdf)) +
      geom_line(colour = "red", linewidth = 1) +
      xlim(x_min, x_max) +
      labs(x = "x", y = "Density") +
      theme_bw()

    plot_width <- 8
    plot_height <- 5

  } else if (parametric_form %in%
               c("bivariate_Gaussian", "multivariate_Gaussian")) {
    # Delegate to the shared p x p panel grid for bivariate/multivariate dims
    p <- .build_product_grid(parameters)
    plot_width <- length(parameters$mu) * 3
    plot_height <- length(parameters$mu) * 3

  } else {
    stop(paste("Unsupported parametric form:", parametric_form))
  }

  .finalize_plot(p, show, save, filename, plots_directory,
                 plot_width, plot_height)
}
