#############################
# Shared validation helpers #
#############################

#' Validate that a value is a positive integer
#'
#' @param value Value to validate
#' @param label Human-readable label used in the error message
#' @keywords internal
.validate_positive_int <- function(value, label) {
  if (!is.numeric(value) || length(value) != 1 ||
        value != as.integer(value) || value <= 0) {
    stop(paste0(
      "The ", label, " must be a positive integer, got ", value
    ))
  }
}

#' Validate that x is numeric and contains only finite values
#'
#' @param x Value to validate
#' @param name Human-readable name used in the error message
#' @return x, confirmed to be numeric and finite
#' @keywords internal
.to_finite_numeric <- function(x, name) {
  if (!is.numeric(x)) {
    stop(paste0(name, " must contain only numeric values."))
  }
  if (!all(is.finite(x))) {
    stop(paste0(name, " cannot contain NaN or Inf."))
  }
  x
}

#' Validate that x has no dim attribute (i.e. is a plain 1D vector)
#'
#' @param x Value to validate
#' @param name Human-readable name used in the error message
#' @keywords internal
.validate_is_vector <- function(x, name) {
  if (!is.null(dim(x))) {
    stop(paste0(name, " must be a 1D vector."))
  }
}

#' Validate that all elements of x are strictly positive
#'
#' @param x Value to validate
#' @param label Human-readable label used in the error message
#' @keywords internal
.validate_positive <- function(x, label) {
  if (any(x <= 0)) {
    stop(paste0(label, " must be strictly positive."))
  }
}

#############################
# Shared CSV output helpers #
#############################

#' Append result rows to a shared CSV file
#'
#' Uses a symlink as an atomic POSIX lock to allow concurrent workers to
#' append to the same CSV file safely.
#'
#' @param row_df data frame containing one or more rows to append
#' @param csv_path path to the CSV file
#' @param lock_path path used for the symlink lock
#' @return Called for its side effect of writing to `csv_path`
#' @export
append_to_csv <- function(row_df, csv_path, lock_path) {
  # Append to the final CSV; use a symlink as an atomic POSIX lock
  repeat {
    if (isTRUE(file.symlink(as.character(Sys.getpid()), lock_path))) break
    Sys.sleep(0.05)
  }
  tryCatch({
    header_needed <- !file.exists(csv_path) || file.info(csv_path)$size == 0
    # Refuse to append when the existing file's columns do not match
    if (!header_needed) {
      existing_columns <- names(
        utils::read.csv(csv_path, nrows = 0, check.names = FALSE)
      )
      new_columns <- names(row_df)
      columns_match <- length(existing_columns) == length(new_columns) &&
        all(existing_columns == new_columns)
      if (!columns_match) {
        missing <- setdiff(existing_columns, new_columns)
        extra <- setdiff(new_columns, existing_columns)
        stop(paste0(
          "Column mismatch appending to ", csv_path, ": ",
          "missing columns [", paste(missing, collapse = ", "), "], ",
          "extra columns [", paste(extra, collapse = ", "), "]"
        ))
      }
    }
    write.table(
      row_df,
      file = csv_path,
      sep = ",",
      row.names = FALSE,
      col.names = header_needed,
      append = !header_needed,
      quote = TRUE,
      na = ""
    )
  }, finally = unlink(lock_path))
}

##############################################
# Shared DGP-parameter serialization helpers #
##############################################

#' Build a 1-indexed, underscore-joined DGP-parameter CSV column name
#'
#' @param key column name prefix (e.g. "mu", "cov", "w")
#' @param node 1-indexed node number
#' @param ... additional 1-indexed dimension indices
#' @return character column name
#' @keywords internal
.dgp_param_column <- function(key, node, ...) {
  paste(c(key, node, ...), collapse = "_")
}

#################################
# Shared distributional metrics #
#################################

#' Compute the symmetric matrix square root via eigendecomposition
#' @noRd
.sqrtm <- function(mat) {
  eig <- eigen(mat, symmetric = TRUE)
  eig$vectors %*% diag(sqrt(pmax(eig$values, 0))) %*% t(eig$vectors)
}

#' Compute the analytical KL divergence between two multivariate Gaussians
#' @noRd
.kl_gaussian <- function(mu_hat, cov_hat, mu_true, cov_true) {
  d <- length(mu_true)
  inv_true <- solve(cov_true)
  delta <- mu_true - mu_hat
  0.5 * (sum(diag(inv_true %*% cov_hat)) +
           as.numeric(t(delta) %*% inv_true %*% delta) -
           d + log(det(cov_true) / det(cov_hat)))
}

#' Compute the analytical W2 (Bures metric) distance between two Gaussians
#' @noRd
.w2_gaussian <- function(mu_hat, cov_hat, mu_true, cov_true) {
  sqrt_hat <- .sqrtm(cov_hat)
  bures_mat <- sqrt_hat %*% cov_true %*% sqrt_hat
  bures_sq <- sum((mu_hat - mu_true)^2) +
    sum(diag(cov_hat)) + sum(diag(cov_true)) - 2 * sum(diag(.sqrtm(bures_mat)))
  sqrt(max(bures_sq, 0))
}

#' Compute the empirical Wasserstein-1 distance between two sample sets via OT
#' @noRd
.w1_ot <- function(samples_a, samples_b) {
  transport::wasserstein(
    transport::pp(samples_a), transport::pp(samples_b), p = 1
  )
}

###########################
# Shared Gaussian helpers #
###########################

#' Return the mean and standard deviation of a 1D Gaussian marginal
#'
#' @param mu mean vector
#' @param cov covariance matrix
#' @param i dimension index (1-based)
#' @return list with mu_1d, sigma_1d
#' @keywords internal
.gaussian_marginal_1d <- function(mu, cov, i) {
  list(mu_1d = mu[i], sigma_1d = sqrt(cov[i, i]))
}

#' Return the mean vector and covariance slice of a 2D Gaussian marginal
#'
#' @param mu mean vector
#' @param cov covariance matrix
#' @param i row dimension index (1-based)
#' @param j column dimension index (1-based)
#' @return list with mu_2d, cov_2d
#' @keywords internal
.gaussian_slice_2d <- function(mu, cov, i, j) {
  list(mu_2d = mu[c(j, i)], cov_2d = cov[c(j, i), c(j, i)])
}

#' Build the symmetric Gaussian x-range used for 1D density/marginal curves
#'
#' @param mu mean
#' @param sigma standard deviation
#' @param n_std number of standard deviations either side of mu
#' @param n_points number of grid points
#' @return numeric vector x_seq
#' @keywords internal
.gaussian_x_range <- function(mu, sigma, n_std = 4, n_points = 500) {
  seq(mu - n_std * sigma, mu + n_std * sigma, length.out = n_points)
}

#' Compute a bivariate Gaussian density on a grid, for contour overlays
#'
#' @param mu_2d length-2 mean vector
#' @param cov_2d 2x2 covariance matrix
#' @param n_grid number of grid points per axis
#' @return data frame with columns x, y, density
#' @keywords internal
.bivariate_normal_density_grid <- function(mu_2d, cov_2d, n_grid = 100) {
  x_std <- sqrt(cov_2d[1, 1])
  y_std <- sqrt(cov_2d[2, 2])
  x_seq <- .gaussian_x_range(mu_2d[1], x_std, n_std = 3, n_points = n_grid)
  y_seq <- .gaussian_x_range(mu_2d[2], y_std, n_std = 3, n_points = n_grid)

  grid <- expand.grid(x = x_seq, y = y_seq)
  inv_cov <- solve(cov_2d)
  norm_const <- 1 / (2 * pi * sqrt(det(cov_2d)))

  dx <- grid$x - mu_2d[1]
  dy <- grid$y - mu_2d[2]
  quad_form <- inv_cov[1, 1] * dx^2 + 2 * inv_cov[1, 2] * dx * dy +
    inv_cov[2, 2] * dy^2
  grid$density <- norm_const * exp(-0.5 * quad_form)

  grid
}
