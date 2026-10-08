##########################
# Stable transformations #
##########################

#' Map finite log-odds values to probabilities stably
#'
#' @param values Numeric vector of finite logit values
#' @return Numeric vector of probabilities in (0, 1)
#' @export
stable_sigmoid <- function(values) {
  if (!is.numeric(values) || any(!is.finite(values))) {
    stop("Logit values must be finite.")
  }
  stats::plogis(values)
}

#' Map interior probabilities to log-odds values stably
#'
#' @param values Numeric vector of probabilities strictly between zero and one
#' @return Numeric vector of logit values
#' @export
stable_logit <- function(values) {
  if (!is.numeric(values) || any(!is.finite(values)) ||
        any(values <= 0.0) || any(values >= 1.0)) {
    stop("Probabilities must be finite and strictly between zero and one.")
  }
  log(values) - log1p(-values)
}

##################################
# Jacobian-corrected GBF factors #
##################################

# Validate one positive finite scalar parameter.
.positive_float <- function(value, name) {
  if (length(value) != 1 || !is.numeric(value) || !is.finite(value) ||
        value <= 0.0) {
    stop(paste0(name, " must be finite and positive."))
  }

  as.numeric(value)
}

#' Build the exact target proportional to the raw Beta component product
#'
#' Recomputes the shape correction for the raw (uncorrected) product of the
#' node Beta densities: the Beta-logit GBF specialization corrects for this
#' Jacobian-shrinkage before running its exact-algorithm transition, so the
#' raw product itself is only used here to validate feasibility.
#'
#' @param nodes List of per-node lists, each with exactly `alpha` and `beta`
#' @return List with `alpha` and `beta`, the raw product's shape parameters
#' @export
build_raw_product_oracle <- function(nodes) {
  if (!is.list(nodes) || length(nodes) == 0) {
    stop("At least one Beta component is required.")
  }
  alphas <- numeric(length(nodes))
  betas <- numeric(length(nodes))
  for (node_index in seq_along(nodes)) {
    node <- nodes[[node_index]]
    if (!is.list(node) || !setequal(names(node), c("alpha", "beta")) ||
          length(names(node)) != 2) {
      stop(paste("Node", node_index, "must contain exactly alpha and beta."))
    }
    alphas[node_index] <- .positive_float(node$alpha, "alpha")
    betas[node_index] <- .positive_float(node$beta, "beta")
  }
  correction <- length(nodes) - 1
  alpha <- sum(alphas) - correction
  beta <- sum(betas) - correction
  if (alpha <= 0.0 || beta <= 0.0) {
    stop("The raw Beta product is improper because a shape is nonpositive.")
  }

  list(alpha = alpha, beta = beta)
}

#' Validate or construct the Jacobian-share vector
#'
#' @param node_count Positive integer number of Beta nodes
#' @param shares Numeric vector of length `node_count` summing to one, or
#'   NULL to fall back to the uniform allocation
#' @return Numeric vector of validated Jacobian shares
#' @export
validate_jacobian_shares <- function(node_count, shares = NULL) {
  .validate_positive_int(node_count, "node_count")
  if (is.null(shares)) {
    return(rep(1.0 / node_count, node_count))
  }
  parsed <- as.numeric(shares)
  if (length(parsed) != node_count || any(!is.finite(parsed))) {
    stop("Jacobian shares must be one finite value per node.")
  }
  if (any(parsed <= 0.0) ||
        abs(sum(parsed) - 1.0) > sqrt(.Machine$double.eps)) {
    stop("Jacobian shares must be positive and sum to one.")
  }

  parsed
}

#' Select permutation-equivariant shares with the largest common safety margin
#'
#' @param nodes List of per-node lists, each with exactly `alpha` and `beta`
#' @param numerical_margin Minimum acceptable common auxiliary-shape margin
#' @return List with `shares`, `lower_bounds`, `slack`, and
#'   `minimum_auxiliary_margin`
#' @export
select_jacobian_shares <- function(nodes, numerical_margin = 1e-12) {
  build_raw_product_oracle(nodes)
  margin <- .positive_float(numerical_margin, "numerical_margin")
  lower_bounds <- vapply(nodes, function(node) {
    max(0.0, 1.0 - as.numeric(node$alpha), 1.0 - as.numeric(node$beta))
  }, numeric(1))
  slack <- 1.0 - sum(lower_bounds)
  common_margin <- slack / length(nodes)
  if (slack <= 0.0) {
    stop(paste(
      "The raw Beta product is proper, but no feasible corrected-logit",
      "factorization exists because the Jacobian-share lower bounds",
      "sum to at least one."
    ))
  }
  if (common_margin <= margin) {
    stop(paste(
      "A corrected-logit factorization exists mathematically, but its",
      "maximum common auxiliary-shape margin is below the configured",
      "numerical margin."
    ))
  }
  shares <- lower_bounds + common_margin
  shares <- shares + (1.0 - sum(shares)) / length(nodes)
  validate_jacobian_shares(length(nodes), shares)

  list(
    shares = shares,
    lower_bounds = lower_bounds,
    slack = slack,
    minimum_auxiliary_margin = common_margin
  )
}

#' Build corrected logit factors whose product targets the raw Beta product
#'
#' @param nodes List of per-node lists, each with exactly `alpha` and `beta`
#' @param jacobian_shares Numeric vector of Jacobian shares, or NULL to
#'   select them automatically via `select_jacobian_shares`
#' @return List of per-node factor lists, each with `alpha`, `beta`,
#'   `jacobian_share`, `auxiliary_alpha`, and `auxiliary_beta`
#' @export
build_logit_factors <- function(nodes, jacobian_shares = NULL) {
  oracle <- build_raw_product_oracle(nodes)
  shares <- if (is.null(jacobian_shares)) {
    select_jacobian_shares(nodes)$shares
  } else {
    validate_jacobian_shares(length(nodes), jacobian_shares)
  }
  factors <- lapply(seq_along(nodes), function(index) {
    node <- nodes[[index]]
    alpha <- .positive_float(node$alpha, "alpha")
    beta <- .positive_float(node$beta, "beta")
    share <- .positive_float(shares[index], "jacobian_share")
    auxiliary_alpha <- alpha + share - 1.0
    auxiliary_beta <- beta + share - 1.0
    if (auxiliary_alpha <= 0.0 || auxiliary_beta <= 0.0) {
      stop("The Jacobian-corrected auxiliary Beta shapes must be positive.")
    }

    list(
      alpha = alpha,
      beta = beta,
      jacobian_share = share,
      auxiliary_alpha = auxiliary_alpha,
      auxiliary_beta = auxiliary_beta
    )
  })
  transformed_alpha <- sum(vapply(factors, function(f) f$auxiliary_alpha, 1))
  transformed_beta <- sum(vapply(factors, function(f) f$auxiliary_beta, 1))
  if (abs(transformed_alpha - oracle$alpha) > sqrt(.Machine$double.eps) *
        max(1, abs(oracle$alpha)) ||
        abs(transformed_beta - oracle$beta) > sqrt(.Machine$double.eps) *
          max(1, abs(oracle$beta))) {
    stop("Corrected logit factors do not recover the raw product target.")
  }

  factors
}

############################
# Fitted GBF preprocessing #
############################

# Validate a raw n x num_nodes component-sample matrix.
.raw_sample_matrix <- function(raw_samples) {
  samples <- as.matrix(raw_samples)
  if (!is.numeric(samples) || nrow(samples) < 2 || ncol(samples) == 0 ||
        any(!is.finite(samples)) || any(samples <= 0.0) ||
        any(samples >= 1.0)) {
    stop(paste(
      "Raw Beta samples must be an interior finite matrix with at least",
      "two rows."
    ))
  }

  samples
}

#' Fit fixed-support Beta parameters from raw component samples
#'
#' Fits shape-only maximum-likelihood Beta parameters with support fixed at
#' [0, 1] for each node column, matching Python's
#' `scipy.stats.beta.fit(samples, floc=0, fscale=1)`, by seeding R's
#' `MASS::fitdistr` optimizer at the method-of-moments shape estimate.
#'
#' @param raw_samples Numeric n x num_nodes matrix of interior Beta samples
#' @return List of per-node lists, each with `alpha` and `beta`
#' @export
fit_beta_components <- function(raw_samples) {
  samples <- .raw_sample_matrix(raw_samples)
  lapply(seq_len(ncol(samples)), function(node_index) {
    x <- samples[, node_index]
    variance <- stats::var(x)
    if (variance <= .Machine$double.eps) {
      stop(
        "Fixed-support Beta MLE requires positive empirical variance."
      )
    }
    mean_x <- mean(x)
    common <- mean_x * (1.0 - mean_x) / variance - 1.0
    fit <- suppressWarnings(MASS::fitdistr(
      x,
      "beta",
      start = list(shape1 = mean_x * common, shape2 = (1.0 - mean_x) * common)
    ))
    alpha <- unname(fit$estimate["shape1"])
    beta <- unname(fit$estimate["shape2"])
    if (!is.finite(alpha) || !is.finite(beta)) {
      stop("Fixed-support Beta MLE returned nonfinite shapes.")
    }

    list(alpha = alpha, beta = beta)
  })
}

#' Transport raw draws into fitted Jacobian-corrected auxiliary logits
#'
#' @param raw_samples Numeric n x num_nodes matrix of interior Beta samples
#' @return List with `auxiliary_logits`, `auxiliary_x`, `cdf_clip_lower`,
#'   `cdf_clip_upper`, `construction`, `estimator`, `fitted_nodes`,
#'   `jacobian_shares`, `preprocessing_runtime_scope`, and
#'   `preprocessing_seconds`
#' @export
preprocess_beta_gbf <- function(raw_samples) {
  start_time <- proc.time()
  samples <- .raw_sample_matrix(raw_samples)
  fitted_nodes <- fit_beta_components(samples)
  shares <- select_jacobian_shares(fitted_nodes)$shares
  factors <- build_logit_factors(fitted_nodes, shares)
  lower <- .Machine$double.eps
  upper <- 1.0 - lower
  auxiliary_x <- matrix(NA_real_, nrow = nrow(samples), ncol = ncol(samples))
  auxiliary_logits <- auxiliary_x
  for (node_index in seq_len(ncol(samples))) {
    node <- fitted_nodes[[node_index]]
    factor <- factors[[node_index]]
    probabilities <- stats::pbeta(samples[, node_index], node$alpha, node$beta)
    probabilities <- pmin(pmax(probabilities, lower), upper)
    values <- stats::qbeta(
      probabilities, factor$auxiliary_alpha, factor$auxiliary_beta
    )
    values <- pmin(pmax(values, lower), upper)
    auxiliary_x[, node_index] <- values
    auxiliary_logits[, node_index] <- stable_logit(values)
  }

  list(
    auxiliary_logits = auxiliary_logits,
    auxiliary_x = auxiliary_x,
    cdf_clip_lower = lower,
    cdf_clip_upper = upper,
    construction = "fitted_beta_cdf_to_corrected_beta_quantile_logit",
    estimator = "fixed_support_beta_maximum_likelihood",
    fitted_nodes = fitted_nodes,
    jacobian_shares = shares,
    preprocessing_runtime_scope =
      "fixed_support_beta_mle_share_selection_and_quantile_transport",
    preprocessing_seconds = unname((proc.time() - start_time)["elapsed"])
  )
}
