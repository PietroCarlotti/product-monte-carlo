###########################
# Univariate Gaussian DGP #
###########################

#' Constructor for a univariate Gaussian DGP
#'
#' Each node k is f_k(x) = N(x | mu_k, sigma_k^2).
#' The product distribution is f(x) \propto prod_k f_k(x) =
#'   N(x | mu_product, sigma_product^2).
#'
#' @param mu Numeric vector of means
#' @param sigma Numeric vector of standard deviations
#' @return A list containing:
#' \itemize{
#'  \item parametric_form: "univariate_Gaussian"
#'  \item num_nodes: Number of nodes (K in the paper)
#'  \item mu: Vector of means
#'  \item sigma: Vector of standard deviations
#'  \item mu_product: Mean of the product distribution
#'  \item sigma_product: Standard deviation of the product distribution
#' }
#' @export
univariate_gaussian_dgp <- function(mu, sigma) {
  # Initial conformity checks on mu and sigma
  mu <- .to_finite_numeric(mu, "mu")
  sigma <- .to_finite_numeric(sigma, "sigma")

  # Number of nodes
  num_nodes <- length(mu)
  .validate_positive_int(num_nodes, "number of nodes")

  # Univariate-Gaussian-specific dimensionality & positivity checks
  .validate_is_vector(mu, "mu")
  .validate_is_vector(sigma, "sigma")

  if (length(mu) != length(sigma)) {
    stop(paste0(
      "Mismatch: mu length ", length(mu),
      " != sigma length ", length(sigma)
    ))
  }
  .validate_positive(sigma, "Standard deviation (sigma)")

  # Vector of precisions
  tau <- 1 / sigma^2

  # Output
  list(
    parametric_form = "univariate_Gaussian",
    num_nodes = num_nodes,
    mu = mu,
    sigma = sigma,
    mu_product = sum(mu * tau) / sum(tau),
    sigma_product = sqrt(1 / sum(tau))
  )
}

##################################
# Shared Gaussian-family helpers #
##################################

#' Validate shape, symmetry and positive-definiteness of multi-dimensional
#' Gaussian DGP inputs
#'
#' Shared by `bivariate_gaussian_dgp` and `multivariate_gaussian_dgp`.
#'
#' @param mu list of num_nodes mean vectors
#' @param cov list of num_nodes covariance matrices (Sigma in the paper)
#' @return list with num_nodes, p, mu (list of vectors), cov (list of
#'   matrices)
#' @keywords internal
.validate_gaussian_dgp_inputs <- function(mu, cov) {
  if (!is.list(mu)) {
    stop("mu must be a list of per-node mean vectors.")
  }
  if (!is.list(cov)) {
    stop("cov must be a list of per-node covariance matrices.")
  }

  # Validate every node's mu and cov, converting them to finite arrays
  mu <- lapply(mu, .to_finite_numeric, name = "mu")
  cov <- lapply(cov, .to_finite_numeric, name = "cov")

  # Consistency checks
  k_mu <- length(mu)
  k_cov <- length(cov)
  if (k_mu != k_cov) {
    stop(paste0(
      "Number of nodes mismatch: mu has ", k_mu, " nodes, cov has ",
      k_cov, " nodes"
    ))
  }

  # Number of nodes is determined by mu's length
  num_nodes <- k_mu
  .validate_positive_int(num_nodes, "number of nodes")

  # Dimensionality is determined by mu's length
  p_mu <- length(mu[[1]])
  for (k in seq_len(num_nodes)) {
    .validate_is_vector(mu[[k]], paste0("mu[[", k, "]]"))
    if (length(mu[[k]]) != p_mu) {
      stop(paste0(
        "Dimension p mismatch: mu is not consistently ", p_mu, "-dim"
      ))
    }
    cov[[k]] <- as.matrix(cov[[k]])
    p_sig1 <- nrow(cov[[k]])
    p_sig2 <- ncol(cov[[k]])
    if (p_mu != p_sig1 || p_mu != p_sig2) {
      stop(paste0(
        "Dimension p mismatch: mu is ", p_mu, "-dim, cov[[", k,
        "]] is ", p_sig1, "x", p_sig2
      ))
    }
  }
  p <- p_mu

  # Dimensionality & positivity checks
  for (k in seq_len(num_nodes)) {
    matrix_k <- cov[[k]]

    # Check for symmetry
    if (!isSymmetric(matrix_k)) {
      stop(paste0("Covariance matrix for node ", k, " is not symmetric."))
    }

    # Check for positive definiteness
    eigenvalues <- eigen(
      matrix_k, symmetric = TRUE, only.values = TRUE
    )$values
    if (!all(eigenvalues > 0)) {
      stop(paste0(
        "Covariance matrix for node ", k, " is not positive definite."
      ))
    }
  }

  list(num_nodes = num_nodes, p = p, mu = mu, cov = cov)
}

#' Combine per-node Gaussians into the resulting product-Gaussian parameters
#'
#' Shared by `bivariate_gaussian_dgp` and `multivariate_gaussian_dgp`.
#'
#' @param mu list of num_nodes mean vectors
#' @param cov list of num_nodes covariance matrices (Sigma in the paper)
#' @return list with mu_product, cov_product
#' @keywords internal
.combine_gaussians <- function(mu, cov) {
  # Compute parameters for the resulting Gaussian distribution obtained
  # by multiplying num_nodes Gaussian distributions
  inv_covs <- lapply(cov, solve)
  inv_cov_product <- Reduce("+", inv_covs)
  cov_product <- solve(inv_cov_product)
  mu_product <- as.vector(
    cov_product %*% Reduce("+", mapply(
      function(inv_cov, m) inv_cov %*% m, inv_covs, mu, SIMPLIFY = FALSE
    ))
  )

  list(mu_product = mu_product, cov_product = cov_product)
}

##########################
# Bivariate Gaussian DGP #
##########################

#' Constructor for a bivariate Gaussian DGP
#'
#' Each node k is f_k(x) = N(x | mu_k, cov_k), x in R^2.
#' The product distribution is N(x | mu_product, cov_product) where
#' cov_product^{-1} = sum_k cov_k^{-1}.
#'
#' @param mu list of num_nodes 2D mean vectors
#' @param cov list of num_nodes 2x2 covariance matrices (Sigma in the paper)
#' @return list with parametric_form, num_nodes, p, mu, cov, mu_product,
#'   cov_product
#' @export
bivariate_gaussian_dgp <- function(mu, cov) {
  validated <- .validate_gaussian_dgp_inputs(mu, cov)

  if (validated$p != 2) {
    stop(paste0(
      "bivariate_gaussian_dgp requires 2D inputs, got ", validated$p,
      "D — use multivariate_gaussian_dgp for other dimensions"
    ))
  }

  combined <- .combine_gaussians(validated$mu, validated$cov)

  list(
    parametric_form = "bivariate_Gaussian",
    num_nodes = validated$num_nodes,
    p = validated$p,
    mu = validated$mu,
    cov = validated$cov,
    mu_product = combined$mu_product,
    cov_product = combined$cov_product
  )
}

#############################
# Multivariate Gaussian DGP #
#############################

#' Constructor for a multivariate Gaussian DGP
#'
#' Each node k is f_k(x) = N(x | mu_k, cov_k).
#' The product distribution is N(x | mu_product, cov_product) where
#' cov_product^{-1} = sum_k cov_k^{-1}.
#'
#' @param mu list of num_nodes mean vectors
#' @param cov list of num_nodes covariance matrices (Sigma in the paper)
#' @return list with parametric_form, num_nodes, p, mu, cov, mu_product,
#'   cov_product
#' @export
multivariate_gaussian_dgp <- function(mu, cov) {
  validated <- .validate_gaussian_dgp_inputs(mu, cov)
  combined <- .combine_gaussians(validated$mu, validated$cov)

  list(
    parametric_form = "multivariate_Gaussian",
    num_nodes = validated$num_nodes,
    p = validated$p,
    mu = validated$mu,
    cov = validated$cov,
    mu_product = combined$mu_product,
    cov_product = combined$cov_product
  )
}

############
# Beta DGP #
############

#' Constructor for a Beta DGP
#'
#' Each node k is f_k(x) = Beta(x | a_k, b_k).
#' The product distribution is Beta(x | a_product, b_product) where
#' a_product = sum_k a_k - (num_nodes - 1) and
#' b_product = sum_k b_k - (num_nodes - 1).
#'
#' @param a Numeric vector of first shape parameters
#' @param b Numeric vector of second shape parameters
#' @return A list containing:
#' \itemize{
#'  \item parametric_form: "Beta"
#'  \item num_nodes: Number of nodes (K in the paper)
#'  \item a: Vector of first shape parameters
#'  \item b: Vector of second shape parameters
#'  \item a_product: First shape parameter of the product distribution
#'  \item b_product: Second shape parameter of the product distribution
#' }
#' @export
beta_dgp <- function(a, b) {
  # Initial conformity checks on a and b
  a <- .to_finite_numeric(a, "a")
  b <- .to_finite_numeric(b, "b")

  # Shape validation
  .validate_is_vector(a, "a")
  .validate_is_vector(b, "b")

  if (length(a) != length(b)) {
    stop(paste0(
      "Mismatch: a length ", length(a), " != b length ", length(b)
    ))
  }

  # Number of nodes is determined by the length of a and b
  num_nodes <- length(a)
  .validate_positive_int(num_nodes, "number of nodes")

  # Beta-specific positivity checks
  .validate_positive(a, "Shape parameter a")
  .validate_positive(b, "Shape parameter b")

  # Compute parameters for the resulting Beta distribution obtained by
  # multiplying num_nodes Beta distributions
  a_product <- sum(a) - (num_nodes - 1)
  b_product <- sum(b) - (num_nodes - 1)

  if (a_product <= 0 || b_product <= 0) {
    stop(paste(
      "Resulting shape parameters for the product Beta distribution",
      "must be strictly positive."
    ))
  }

  list(
    parametric_form = "Beta",
    num_nodes = num_nodes,
    a = a,
    b = b,
    a_product = a_product,
    b_product = b_product
  )
}

#####################
# Per-node sampling #
#####################

#' Sampler from a parametric DGP
#'
#' Draw n observations from each node of a DGP
#'
#' @param dgp Parameters of the DGP
#' @param n Number of observations per node
#' @return list of num_nodes numeric vectors (univariate Gaussian and Beta)
#'   or num_nodes matrices n x p (bivariate and multivariate Gaussian)
#' @export
sample_dgp <- function(dgp, n) {
  .validate_positive_int(n, "sample size")

  if (dgp$parametric_form == "univariate_Gaussian") {
    lapply(seq_len(dgp$num_nodes), function(k) {
      rnorm(n, mean = dgp$mu[k], sd = dgp$sigma[k])
    })
  } else if (dgp$parametric_form %in%
               c("bivariate_Gaussian", "multivariate_Gaussian")) {
    lapply(seq_len(dgp$num_nodes), function(k) {
      MASS::mvrnorm(n, mu = dgp$mu[[k]], Sigma = dgp$cov[[k]])
    })
  } else if (dgp$parametric_form == "Beta") {
    lapply(seq_len(dgp$num_nodes), function(k) rbeta(n, dgp$a[k], dgp$b[k]))
  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }
}

########################
# Product-distribution #
########################

#' Sample from the product distribution of a DGP
#'
#' Draw n observations from the product distribution
#' f(x) \propto prod_k f_k(x) implied by a DGP.
#'
#' @param dgp Parameters of the DGP, as returned by a `*_dgp` constructor
#' @param n Number of observations to draw
#' @return an n x p matrix (p = 1 for univariate Gaussian and Beta)
#' @export
sample_product_dgp <- function(dgp, n) {
  if (dgp$parametric_form == "univariate_Gaussian") {
    matrix(rnorm(n, mean = dgp$mu_product, sd = dgp$sigma_product), ncol = 1)
  } else if (dgp$parametric_form %in%
               c("bivariate_Gaussian", "multivariate_Gaussian")) {
    MASS::mvrnorm(n, mu = dgp$mu_product, Sigma = dgp$cov_product)
  } else if (dgp$parametric_form == "Beta") {
    matrix(rbeta(n, dgp$a_product, dgp$b_product), ncol = 1)
  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }
}

#' Compute the closed-form product-distribution mean for a DGP
#'
#' @param dgp Parameters of the DGP, as returned by a `*_dgp` constructor
#' @return Closed-form E[X] under the product distribution
#' @keywords internal
closed_form_mean_product_dgp <- function(dgp) {
  gaussian_forms <- c(
    "univariate_Gaussian", "bivariate_Gaussian", "multivariate_Gaussian"
  )
  if (dgp$parametric_form %in% gaussian_forms) {
    dgp$mu_product
  } else if (dgp$parametric_form == "Beta") {
    dgp$a_product / (dgp$a_product + dgp$b_product)
  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }
}

# Compute the closed-form componentwise second moment under a product DGP
.square_mean_product_dgp <- function(dgp) {
  gaussian_forms <- c(
    "univariate_Gaussian", "bivariate_Gaussian", "multivariate_Gaussian"
  )
  if (dgp$parametric_form == "univariate_Gaussian") {
    dgp$mu_product^2 + dgp$sigma_product^2
  } else if (dgp$parametric_form %in% gaussian_forms) {
    dgp$mu_product^2 + diag(dgp$cov_product)
  } else if (dgp$parametric_form == "Beta") {
    total <- dgp$a_product + dgp$b_product
    dgp$a_product * (dgp$a_product + 1) / (total * (total + 1))
  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }
}

#' Resolve an h transform name to its function
#'
#' @param h_name Name of the transform: "identity", "normalized_sum", or
#'   "square"
#' @return Function applied to samples, or NULL for the identity transform
#' @keywords internal
resolve_h_function <- function(h_name) {
  switch(h_name,
    "identity" = NULL,
    "normalized_sum" = function(x) {
      matrix(rowSums(x) / sqrt(ncol(x)), ncol = 1)
    },
    "square" = function(x) x^2,
    stop(paste("Unknown h transform:", h_name))
  )
}

#' Compute E[h(X)] under the product distribution of a DGP
#'
#' Returns exact moments for the identity and square transforms when named,
#' otherwise estimates E[h(X)] via Monte Carlo using product draws.
#'
#' @param dgp Parameters of the DGP, as returned by a `*_dgp` constructor
#' @param h Function applied once to the full n x p matrix of samples
#'   drawn from the product distribution (not row-by-row), returning a
#'   result with the same number of rows, or NULL
#' @param n_mc Number of Monte Carlo samples used when `h` is not NULL
#' @param h_name Optional transform name used to dispatch exact moments
#' @return E[h(X)] (or E[X] when `h` is NULL) under the product
#'   distribution
#' @export
mean_product_dgp <- function(dgp, h = NULL, n_mc = 100000, h_name = NULL) {
  if (!is.null(h_name) && h_name == "square") {
    return(.square_mean_product_dgp(dgp))
  }
  if (!is.null(h_name) && h_name == "normalized_sum") {
    mean <- closed_form_mean_product_dgp(dgp)
    return(sum(mean) / sqrt(length(mean)))
  }
  if (is.null(h) || (!is.null(h_name) && h_name == "identity")) {
    return(closed_form_mean_product_dgp(dgp))
  }

  samples <- sample_product_dgp(dgp, n_mc)
  colMeans(h(samples))
}
