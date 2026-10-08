######################
# Evaluation metrics #
######################

#' KL between the fusion distribution and the true product distribution
#'
#' @param dgp list returned by a `*_dgp()` constructor
#' @param samples numeric vector (univariate) or matrix n x d (multivariate)
#' @return scalar KL divergence estimate
#' @export
kl <- function(dgp, samples) {
  if (dgp$parametric_form == "univariate_Gaussian") {
    # Method-of-moments fit: estimate a Gaussian from the fusion samples
    mu_hat <- mean(samples)
    sigma_hat <- sd(samples)
    mu_true <- dgp$mu_product
    sigma_true <- dgp$sigma_product

    # Closed-form KL between the fitted and true univariate Gaussians
    log(sigma_true / sigma_hat) +
      (sigma_hat^2 + (mu_hat - mu_true)^2) / (2 * sigma_true^2) - 0.5
  } else if (dgp$parametric_form %in%
               c("bivariate_Gaussian", "multivariate_Gaussian")) {
    # Method-of-moments fit: estimate a Gaussian from the fusion samples
    mu_hat <- colMeans(samples)
    cov_hat <- cov(samples)

    # Closed-form KL between the fitted and true Gaussians
    .kl_gaussian(mu_hat, cov_hat, dgp$mu_product, dgp$cov_product)
  } else if (dgp$parametric_form == "Beta") {
    # Fixed-support Beta MLE fit: estimate a Beta from the fusion samples
    fitted <- fit_beta_components(matrix(samples, ncol = 1))[[1]]
    a_hat <- fitted$alpha
    b_hat <- fitted$beta
    a_true <- dgp$a_product
    b_true <- dgp$b_product

    # Closed-form KL between the fitted and true Betas
    lbeta(a_true, b_true) - lbeta(a_hat, b_hat) +
      (a_hat - a_true) * digamma(a_hat) + (b_hat - b_true) * digamma(b_hat) -
      (a_hat - a_true + b_hat - b_true) * digamma(a_hat + b_hat)
  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }
}

#' Wasserstein distances between fusion samples and the true product
#' distribution
#'
#' Univariate: 1D empirical quantile transport (exact W1, W2).
#' Multivariate: W2 via analytical Bures metric on sample moments;
#'   W1 via empirical OT on n_subsample points (biased but informative).
#'
#' @param dgp list returned by a `*_dgp()` constructor
#' @param samples numeric vector (univariate) or matrix n x d (multivariate)
#' @param n_subsample number of points for multivariate OT W1 (default 2000)
#' @return list(w1, w2)
#' @export
wasserstein <- function(dgp, samples, n_subsample = 2000) {
  if (dgp$parametric_form %in% c("univariate_Gaussian", "Beta")) {
    # True-side samples: fresh draws from the true product distribution
    n <- length(samples)
    consensus_true <- sample_product_dgp(dgp, n)[, 1]

    # Exact W1/W2 via 1D empirical quantile transport
    w1 <- mean(abs(sort(consensus_true) - sort(samples)))
    w2 <- sqrt(mean((sort(consensus_true) - sort(samples))^2))
  } else if (dgp$parametric_form %in%
               c("bivariate_Gaussian", "multivariate_Gaussian")) {

    # Method-of-moments fit: estimate a Gaussian from the fusion samples
    mu_hat <- colMeans(samples)
    cov_hat <- cov(samples)

    # Closed-form W2 (Bures metric) between the fitted and true Gaussians
    w2 <- .w2_gaussian(mu_hat, cov_hat, dgp$mu_product, dgp$cov_product)

    # Estimated-side samples: subsample of the actual fusion samples
    m <- min(n_subsample, nrow(samples))
    idx <- sample.int(nrow(samples), m)

    # True-side samples: fresh draws from the true product distribution
    consensus_true <- sample_product_dgp(dgp, m)

    # Approximate W1 via empirical optimal transport on the two sample sets
    w1 <- .w1_ot(samples[idx, ], consensus_true)
  } else {
    stop(paste("Unsupported parametric form:", dgp$parametric_form))
  }
  list(w1 = w1, w2 = w2)
}
