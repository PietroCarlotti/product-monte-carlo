##################
# Shared helpers #
##################

# Estimate a full or diagonal sample covariance matrix
#' @noRd
.estimate_sample_covariance <- function(sample, covariance_structure) {
  sample_covariance <- stats::cov(sample)
  if (covariance_structure == "full") {
    return(sample_covariance)
  }
  if (covariance_structure == "diagonal") {
    dimension <- ncol(sample)
    return(diag(diag(sample_covariance), nrow = dimension, ncol = dimension))
  }
  stop("covariance_structure must be 'full' or 'diagonal'.")
}

#' Compute preconditioning matrix for each node
#'
#' @param precondition "identity" (all 1), "variance" (per-node sample
#'   variance), or a numeric vector of length k
#' @param samples list of k numeric vectors of sub-posterior samples
#' @param k number of nodes
#' @return numeric vector of length k
#' @keywords internal
.precondition_values <- function(precondition, samples, k) {
  if (precondition == "identity") {
    rep(1.0, k)
  } else if (precondition == "variance") {
    sapply(samples, var)
  } else if (is.numeric(precondition) && length(precondition) == k) {
    precondition
  } else {
    stop(
      "precondition must be 'identity', 'variance',",
      " or a numeric vector of length k"
    )
  }
}

#' Preconditioning matrices for multivariate Gaussian nodes
#' @param precondition "identity", "covariance", or a list of matrices
#' @param samples list of k matrices (n x d)
#' @param k number of nodes
#' @return list of two lists: precondition_matrices, inv_precondition_matrices
#' @keywords internal
.precondition_matrices <- function(precondition, samples, k) {
  d <- ncol(samples[[1]])
  if (is.character(precondition) && length(precondition) == 1L &&
        precondition == "identity") {
    mats <- replicate(k, diag(d), simplify = FALSE)
    list(mats = mats, inv_mats = mats)
  } else if (is.character(precondition) && length(precondition) == 1L &&
               precondition == "covariance") {
    mats <- lapply(samples, cov)
    inv_mats <- lapply(mats, solve)
    list(mats = mats, inv_mats = inv_mats)
  } else if (
    is.list(precondition) && length(precondition) == k &&
      all(vapply(precondition, function(value) {
        is.matrix(value) && identical(dim(value), c(d, d))
      }, logical(1)))
  ) {
    list(mats = precondition, inv_mats = lapply(precondition, solve))
  } else {
    stop(
      "precondition must be 'identity', 'covariance',",
      " or a list of square matrices"
    )
  }
}

# Extracts per-node standard deviations and correlations from a list of
# 2x2 covariance matrices, as required by the bivariate DCFusion API.
#' @noRd
.bivariate_sd_corr <- function(sigma_matrix) {
  sd_vecs <- lapply(sigma_matrix, function(s) sqrt(diag(s)))
  corrs <- sapply(sigma_matrix, function(s) {
    sds <- sqrt(diag(s))
    s[1, 2] / prod(sds)
  })
  list(sd_vecs = sd_vecs, corrs = corrs)
}

#' Compute optimal coalescence time and time grid for BF, GBF, or MCF
#'
#' Selects the coalescence time T and time grid under the SSH condition. For MCF
#' the convention T = 1 is used.
#'
#' @param method "BF", "GBF", or "MCF"
#' @param samples list of k numeric vectors of sub-posterior samples
#' @param k number of sub-posteriors
#' @param n data size — number of observations per sub-posterior
#' @param cess_0_threshold minimum acceptable CESS at initialisation
#'   (default 0.2)
#' @param covariance_structure "full" or "diagonal"
#' @details d (dimension) is inferred automatically from the samples.
#'
#' @return list with:
#'   coalescence_time — scalar T
#'   time_grid — numeric vector from 0 to T
#'   n_time_steps — number of intervals (length(time_grid) - 1)
#' @export
optimal_time_grid <- function(method, samples, k, n, cess_0_threshold = 0.2,
                              covariance_structure = "full") {

  # Use default time grid for MCF
  if (method == "MCF") {
    return(list(
      coalescence_time = 1,
      time_grid = c(0, 1),
      n_time_steps = 1L
    ))
  }

  vanilla <- method == "BF"
  d <- NCOL(samples[[1]])
  multivariate <- d > 1

  if (multivariate) {
    sub_means <- do.call(rbind, lapply(samples, colMeans))
    sub_covs <- lapply(
      samples, .estimate_sample_covariance,
      covariance_structure = covariance_structure
    )
    sub_prec <- lapply(sub_covs, solve)
    lambda <- if (vanilla) NULL else solve(Reduce("+", sub_prec))
  } else {
    sub_means <- sapply(samples, mean)
    sub_covs <- as.list(sapply(samples, var))
    sub_prec <- as.list(1 / sapply(samples, var))
    lambda <- NULL
  }

  guide <- DCFusion::BF_guidance(
    condition = "SSH",
    CESS_0_threshold = cess_0_threshold,
    sub_posterior_samples = samples,
    C = k,
    d = d,
    data_size = n,
    sub_posterior_means = sub_means,
    precondition_matrices = if (vanilla) NULL else sub_covs,
    inv_precondition_matrices = if (vanilla) NULL else sub_prec,
    Lambda = lambda,
    k3 = 1,
    k4 = 1,
    vanilla = vanilla
  )

  # Output
  list(
    coalescence_time = guide$min_T,
    time_grid = guide$mesh,
    n_time_steps = as.integer(length(guide$mesh) - 1L)
  )
}

##############################################
# MCF — Monte Carlo Fusion (Dai et al. 2019) #
##############################################

# Packages an MCF rejection-sampling result into the standard MCF output
# list, computing the rho, Q, and overall acceptance-probability
# diagnostics.
#' @noRd
.mcf_output <- function(result, n_inference) {
  list(
    samples = result$samples,
    rho_acceptance_probability = result$iters_Q / result$iters_rho,
    Q_acceptance_probability = n_inference / result$iters_Q,
    acceptance_probability = n_inference / result$iters_rho
  )
}

# Generic MCF rejection-sampling loop shared by the bivariate and
# multivariate heterogeneous fusion helpers: draws proposals, applies the
# rho step, then delegates the density-specific Q step to log_q_prob_fn.
#' @noRd
.mcf_rejection_sample <- function(
  n_inference, m, time, samples_to_fuse, out_dim,
  inv_precondition_matrices, log_q_prob_fn
) {
  fusion_samples <- matrix(data = NA, nrow = n_inference, ncol = out_dim)
  i <- 0
  iters_rho <- 0
  iters_q <- 0

  proposal_cov <- DCFusion::calculate_proposal_cov(
    time = time,
    weights = inv_precondition_matrices
  )
  inverse_sum_inv <- DCFusion::inverse_sum_matrices(
    matrices = inv_precondition_matrices
  )

  while (i < n_inference) {
    iters_rho <- iters_rho + 1
    x <- t(sapply(
      samples_to_fuse, function(core) core[sample(nrow(core), 1), ]
    ))
    x_mean <- DCFusion::weighted_mean_multivariate(
      matrix = x,
      weights = inv_precondition_matrices,
      inverse_sum_weights = inverse_sum_inv
    )
    log_rho_prob <- DCFusion::log_rho_multivariate(
      x = x,
      x_mean = x_mean,
      time = time,
      inv_precondition_matrices = inv_precondition_matrices
    )
    if (log(runif(1, 0, 1)) < log_rho_prob) {
      iters_q <- iters_q + 1
      y <- as.vector(
        DCFusion::mvrnormArma(N = 1, mu = x_mean, Sigma = proposal_cov)
      )
      log_q_prob <- log_q_prob_fn(x, y)
      if (log(runif(1, 0, 1)) < log_q_prob) {
        i <- i + 1
        fusion_samples[i, ] <- y
      }
    }
  }

  list(samples = fusion_samples, iters_rho = iters_rho, iters_Q = iters_q)
}

# Computes per-node to_Z/to_X transform matrices (matrix square roots of the
# inverse/forward preconditioning matrices) for the heterogeneous MCF Q-step.
#' @noRd
.transform_matrices <- function(
  m, precondition_matrices, inv_precondition_matrices
) {
  lapply(1:m, function(c) {
    list(
      to_Z = expm::sqrtm(inv_precondition_matrices[[c]]),
      to_X = expm::sqrtm(precondition_matrices[[c]])
    )
  })
}

#' Monte Carlo Fusion for univariate Gaussians
#'
#' Sample from the fusion target distribution using Monte Carlo Fusion
#' (rejection-sampling algorithm of Dai et al. 2019).
#'
#' @param n_inference number of fusion samples
#' @param samples list of k numeric vectors; element k contains samples
#'   from node k
#' @param mu length-k vector: sub-posterior means
#' @param sigma length-k vector: sub-posterior standard deviations
#' @param coalescence_time fusion time T (smaller T increases acceptance
#'   but tightens proposal)
#' @param precondition "identity" (Lambda_c=1), "variance"
#'   (Lambda_c=var(samples_c)), or a numeric vector of length k
#'
#' @return A list containing
#' \itemize{
#'   \item samples — length-N vector of fusion draws
#'   \item rho_acceptance_probability — fraction of proposals accepted
#'     at the rho step
#'   \item Q_acceptance_probability — fraction of rho-accepted proposals
#'     accepted at the Q step
#'   \item acceptance_probability — overall fraction of proposals that
#'     produce a fusion sample
#' }
#' @export
mcf_univariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma,
  coalescence_time,
  precondition = "identity"
) {

  # Preliminary conformity checks
  k <- length(samples)
  stopifnot(length(mu) == k, length(sigma) == k)

  # Run MCF
  result <- DCFusion::fusion_uniGaussian(
    N = n_inference,
    m = k,
    time = coalescence_time,
    samples_to_fuse = samples,
    means = mu,
    sds = sigma,
    betas = rep(1, k),
    precondition_values = .precondition_values(precondition, samples, k)
  )

  # Output
  .mcf_output(result, n_inference)
}

# Replicates fusion_biGaussian with per-sub-posterior mean_vec / sd_vec / corr.
#' @noRd
.mcf_bivariate_gaussian_het <- function(
  n_inference, m, time, samples_to_fuse,
  mean_vecs, sd_vecs, corrs, betas,
  precondition_matrices, inv_precondition_matrices
) {
  transform_matrices <- .transform_matrices(
    m, precondition_matrices, inv_precondition_matrices
  )

  log_q_prob_fn <- function(x, y) {
    sum(sapply(1:m, function(c) {
      DCFusion::ea_biGaussian_DL_PT(
        x0 = x[c, ],
        y = y,
        s = 0,
        t = time,
        mean_vec = mean_vecs[[c]],
        sd_vec = sd_vecs[[c]],
        corr = corrs[c],
        beta = betas[c],
        precondition_mat = precondition_matrices[[c]],
        transform_mats = transform_matrices[[c]],
        diffusion_estimator = "Poisson",
        logarithm = TRUE
      )
    }))
  }

  .mcf_rejection_sample(
    n_inference = n_inference,
    m = m,
    time = time,
    samples_to_fuse = samples_to_fuse,
    out_dim = 2,
    inv_precondition_matrices = inv_precondition_matrices,
    log_q_prob_fn = log_q_prob_fn
  )
}

#' Monte Carlo Fusion for bivariate Gaussians (heterogeneous sub-posteriors)
#'
#' Heterogeneous version of MCF for bivariate Gaussians: each sub-posterior c
#' has its own mean vector, standard deviations, and correlation. Replicates
#' the DCFusion::fusion_biGaussian rejection sampler with per-sub-posterior
#' parameters.
#'
#' @param n_inference number of fusion samples
#' @param samples list of k matrices (n x 2)
#' @param mu list of k mean vectors (each of length 2)
#' @param sigma_matrix list of k 2x2 covariance matrices
#' @param coalescence_time fusion time T
#' @param precondition "identity" or "covariance"
#' @return list(samples, rho_acceptance_probability, Q_acceptance_probability,
#'   acceptance_probability)
#' @export
mcf_bivariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma_matrix,
  coalescence_time,
  precondition = "identity"
) {
  k <- length(samples)
  stopifnot(length(mu) == k, length(sigma_matrix) == k)

  pc <- .precondition_matrices(precondition, samples, k)
  sd_corr <- .bivariate_sd_corr(sigma_matrix)

  result <- .mcf_bivariate_gaussian_het(
    n_inference = n_inference,
    m = k,
    time = coalescence_time,
    samples_to_fuse = samples,
    mean_vecs = mu,
    sd_vecs = sd_corr$sd_vecs,
    corrs = sd_corr$corrs,
    betas = rep(1, k),
    precondition_matrices = pc$mats,
    inv_precondition_matrices = pc$inv_mats
  )

  .mcf_output(result, n_inference)
}

# Replicates fusion_multiGaussian with per-sub-posterior mu / inv_Sigma.
#' @noRd
.mcf_multivariate_gaussian_het <- function(
  n_inference, m, time, samples_to_fuse, dim,
  mus, inv_sigmas, betas,
  precondition_matrices, inv_precondition_matrices
) {
  transform_matrices <- .transform_matrices(
    m, precondition_matrices, inv_precondition_matrices
  )
  inv_sigma_z <- lapply(1:m, function(c) {
    to_x <- transform_matrices[[c]]$to_X
    to_x %*% inv_sigmas[[c]] %*% to_x
  })

  log_q_prob_fn <- function(x, y) {
    sum(sapply(1:m, function(c) {
      DCFusion::ea_multiGaussian_DL_PT(
        x0 = x[c, ],
        y = y,
        s = 0,
        t = time,
        dim = dim,
        mu = mus[[c]],
        inv_Sigma = inv_sigmas[[c]],
        inv_Sigma_Z = inv_sigma_z[[c]],
        beta = betas[c],
        precondition_mat = precondition_matrices[[c]],
        transform_mats = transform_matrices[[c]],
        diffusion_estimator = "Poisson",
        logarithm = TRUE
      )
    }))
  }

  .mcf_rejection_sample(
    n_inference = n_inference,
    m = m,
    time = time,
    samples_to_fuse = samples_to_fuse,
    out_dim = dim,
    inv_precondition_matrices = inv_precondition_matrices,
    log_q_prob_fn = log_q_prob_fn
  )
}

#' Monte Carlo Fusion for multivariate Gaussians (heterogeneous sub-posteriors)
#'
#' Heterogeneous version of MCF for multivariate Gaussians: each sub-posterior
#' c has its own mean vector and covariance matrix. Replicates the
#' DCFusion::fusion_multiGaussian rejection sampler with per-sub-posterior
#' parameters.
#'
#' @param n_inference number of fusion samples
#' @param samples list of k matrices (n x d)
#' @param mu list of k mean vectors
#' @param sigma_matrix list of k covariance matrices
#' @param coalescence_time fusion time T
#' @param precondition "identity" or "covariance"
#' @return list(samples, rho_acceptance_probability, Q_acceptance_probability,
#'   acceptance_probability)
#' @export
mcf_multivariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma_matrix,
  coalescence_time,
  precondition = "identity"
) {
  k <- length(samples)
  d <- ncol(samples[[1]])
  stopifnot(length(mu) == k, length(sigma_matrix) == k)

  pc <- .precondition_matrices(precondition, samples, k)
  inv_sigmas <- lapply(sigma_matrix, solve)

  result <- .mcf_multivariate_gaussian_het(
    n_inference = n_inference,
    m = k,
    time = coalescence_time,
    samples_to_fuse = samples,
    dim = d,
    mus = mu,
    inv_sigmas = inv_sigmas,
    betas = rep(1, k),
    precondition_matrices = pc$mats,
    inv_precondition_matrices = pc$inv_mats
  )

  .mcf_output(result, n_inference)
}

##########################################
# BF — Bayesian Fusion (Dai et al. 2021) #
##########################################

#' Bayesian Fusion for univariate Gaussians
#'
#' Identical to GBF with identity preconditioning (Lambda_c = I).
#'
#' @param n_inference number of output particles
#' @param samples list of k numeric vectors; element k contains samples
#'   from node k
#' @param mu length-k vector: sub-posterior means
#' @param sigma length-k vector: sub-posterior SDs
#' @param time_grid numeric vector {0 = t_0 < ... < t_n = T} (length >= 2)
#' @param ess_threshold resample when ESS/N drops below this value (default 0.5)
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer seed, or NULL
#' @param n_cores number of parallel cores
#'
#' @return list with:
#'   samples — length-n_inference equally-weighted draws
#'   ESS — vector of ESS values at each time step
#'   CESS — vector of conditional ESS values at each time step
#'   elapsed — wall-clock seconds per grid interval
#' @export
bf_univariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma,
  time_grid,
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {

  gbf_univariate_gaussian(
    n_inference = n_inference,
    samples = samples,
    mu = mu,
    sigma = sigma,
    time_grid = time_grid,
    precondition = "identity",
    ess_threshold = ess_threshold,
    diffusion_estimator = diffusion_estimator,
    seed = seed,
    n_cores = n_cores
  )
}

#' Bayesian Fusion for bivariate Gaussians (identity preconditioning)
#'
#' @param n_inference number of output particles
#' @param samples list of k matrices (n x 2)
#' @param mu list of k mean vectors (length 2)
#' @param sigma_matrix list of k 2x2 covariance matrices
#' @param time_grid numeric vector from 0 to T
#' @param ess_threshold resampling threshold (default 0.5)
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer or NULL
#' @param n_cores number of cores
#' @return list(samples, ESS, CESS, elapsed)
#' @export
bf_bivariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma_matrix,
  time_grid,
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {
  gbf_bivariate_gaussian(
    n_inference = n_inference,
    samples = samples,
    mu = mu,
    sigma_matrix = sigma_matrix,
    time_grid = time_grid,
    precondition = "identity",
    ess_threshold = ess_threshold,
    diffusion_estimator = diffusion_estimator,
    seed = seed,
    n_cores = n_cores
  )
}

#' Bayesian Fusion for multivariate Gaussians (identity preconditioning)
#'
#' @param n_inference number of output particles
#' @param samples list of k matrices (n x d)
#' @param mu list of k mean vectors
#' @param sigma_matrix list of k covariance matrices
#' @param time_grid numeric vector from 0 to T
#' @param ess_threshold resampling threshold (default 0.5)
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer or NULL
#' @param n_cores number of cores
#' @return list(samples, ESS, CESS, elapsed)
#' @export
bf_multivariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma_matrix,
  time_grid,
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {
  gbf_multivariate_gaussian(
    n_inference = n_inference,
    samples = samples,
    mu = mu,
    sigma_matrix = sigma_matrix,
    time_grid = time_grid,
    precondition = "identity",
    ess_threshold = ess_threshold,
    diffusion_estimator = diffusion_estimator,
    seed = seed,
    n_cores = n_cores
  )
}

########################################################
# GBF — Generalised Bayesian Fusion (Chan et al. 2023) #
########################################################

# Validates that a time grid is a numeric vector of length >= 2, starting
# at 0 and sorted in increasing order.
#' @noRd
.validate_time_grid <- function(time_grid) {
  if (!is.numeric(time_grid) || length(time_grid) < 2 || time_grid[1] != 0) {
    stop("time_grid must be a numeric vector of length >= 2 starting at 0")
  }
  if (any(time_grid != sort(time_grid))) {
    stop("time_grid must be strictly increasing")
  }
}

# Builds the standard GBF/BF output list from a fusion result: resamples the
# final particle set (or takes it directly if the last time step was
# already resampled), then packages it with ESS/CESS/elapsed diagnostics.
# The particles are a plain vector when multivariate is FALSE, and a
# matrix (one row per particle) when multivariate is TRUE.
#' @noRd
.gbf_output <- function(result, n_inference, time_grid, multivariate = FALSE) {
  # Number of steps until coalescence
  n_steps <- length(time_grid)

  # Resample if the last time step was not already resampled
  if (result$resampled[n_steps]) {
    fusion_samples <- result$particles$y_samples
  } else if (multivariate) {
    idx <- sample.int(
      n_inference, size = n_inference, replace = TRUE,
      prob = result$particles$normalised_weights
    )
    fusion_samples <- result$particles$y_samples[idx, ]
  } else {
    fusion_samples <- sample(
      result$particles$y_samples,
      size = n_inference,
      replace = TRUE,
      prob = result$particles$normalised_weights
    )
  }

  list(
    samples = fusion_samples,
    ESS = result$ESS,
    CESS = result$CESS,
    elapsed = result$elapsed_time
  )
}

# Converts a GBF precondition option into DCFusion's D&C argument shape.
#' @noRd
.dc_gbf_precondition <- function(precondition, samples, k) {
  if (identical(precondition, "variance")) {
    TRUE
  } else if (identical(precondition, "identity")) {
    FALSE
  } else if (is.numeric(precondition) && length(precondition) == k) {
    as.list(precondition)
  } else {
    stop(
      "precondition must be 'identity', 'variance',",
      " or a numeric vector of length k"
    )
  }
}

# Builds standard output from a progressive D&C GBF result.
#' @noRd
.dc_gbf_output <- function(result, n_inference, seed = NULL) {
  terminal_particles <- result$particles[[1]]
  terminal_resampled <- result$resampled[[1]]

  if (terminal_resampled[length(terminal_resampled)]) {
    fusion_samples <- terminal_particles$y_samples
  } else {
    if (!is.null(seed)) {
      set.seed(seed)
    }
    fusion_samples <- sample(
      terminal_particles$y_samples,
      size = n_inference,
      replace = TRUE,
      prob = terminal_particles$normalised_weights
    )
  }

  list(
    samples = fusion_samples,
    ESS = result$ESS[[1]],
    CESS = result$CESS[[1]],
    elapsed = unlist(result$time, use.names = FALSE)
  )
}

# Builds standard output from a multivariate D&C GBF result.
#' @noRd
.dc_gbf_multivariate_output <- function(result, n_inference, seed = NULL) {
  terminal_particles <- result$particles[[1]]
  terminal_resampled <- result$resampled[[1]]
  terminal_samples <- terminal_particles$y_samples

  if (terminal_resampled[length(terminal_resampled)]) {
    fusion_samples <- terminal_samples
  } else {
    if (!is.null(seed)) {
      set.seed(seed)
    }
    indices <- sample.int(
      nrow(terminal_samples),
      size = n_inference,
      replace = TRUE,
      prob = terminal_particles$normalised_weights
    )
    fusion_samples <- terminal_samples[indices, , drop = FALSE]
  }

  list(
    samples = as.numeric(fusion_samples[, 1]),
    ESS = unlist(result$ESS, use.names = FALSE),
    CESS = unlist(result$CESS, use.names = FALSE),
    elapsed = unlist(result$time, use.names = FALSE)
  )
}

# Builds a factor merge schedule for balanced binary D&C GBF.
#' @noRd
.dc_gbf_default_merge_schedule <- function(k) {
  factors <- integer(0)
  remaining <- k
  divisor <- 2L
  while (remaining > 1L) {
    while (remaining %% divisor == 0L) {
      factors <- c(factors, divisor)
      remaining <- remaining / divisor
    }
    divisor <- divisor + 1L
  }
  factors
}

# Validates a balanced D&C GBF merge schedule.
#' @noRd
.validate_dc_gbf_schedule <- function(k, merge_schedule) {
  if (
    !is.numeric(merge_schedule) ||
      any(is.na(merge_schedule)) ||
      any(merge_schedule != floor(merge_schedule)) ||
      any(merge_schedule < 2L) ||
      prod(merge_schedule) != k
  ) {
    stop("DC-GBF-HET requires integer merge factors >= 2 whose product is K.")
  }
  TRUE
}

# Runs an expression away from the repository working directory.
#' @noRd
.with_temp_working_dir <- function(expr) {
  old_dir <- getwd()
  temp_dir <- tempfile("dc_gbf_")
  dir.create(temp_dir)
  on.exit(setwd(old_dir), add = TRUE)
  setwd(temp_dir)
  force(expr)
}

#' Generalised Bayesian Fusion for univariate Gaussians
#'
#' Sample from the fusion target distribution using Generalized Bayesian Fusion
#' (SMC algorithm of Chan et al. (2023))
#'
#' @param n_inference number of particles
#' @param samples list of k numeric vectors; element k contains samples
#'   from node k
#' @param mu length-k vector: sub-posterior means
#' @param sigma length-k vector: sub-posterior SDs
#' @param time_grid numeric vector {0 = t_0 < t_1 < ... < t_n = T} (length >= 2)
#' @param precondition "identity", "variance" (recommended), or numeric
#'   vector of length k
#' @param ess_threshold resample when ESS/N drops below this value (default 0.5)
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer seed, or NULL
#' @param n_cores number of parallel cores
#'
#' @return list with:
#'   samples — length-N equally-weighted draws
#'   ESS — vector of ESS values at each time step
#'   CESS — vector of conditional ESS values at each time step
#'   elapsed — wall-clock seconds per grid interval
#' @export
gbf_univariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma,
  time_grid,
  precondition = "variance",
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {

  # Preliminary conformity checks
  k <- length(samples)
  stopifnot(length(mu) == k, length(sigma) == k)
  .validate_time_grid(time_grid)

  # Sample particles
  particles <- DCFusion::initialise_particle_sets(
    samples_to_fuse = samples,
    multivariate = FALSE,
    number_of_steps = length(time_grid)
  )

  # Run GBF
  result <- DCFusion::parallel_GBF_uniGaussian(
    particles_to_fuse = particles,
    N = n_inference,
    m = k,
    time_mesh = time_grid,
    means = mu,
    sds = sigma,
    betas = rep(1, k),
    precondition_values = .precondition_values(precondition, samples, k),
    resampling_method = "multi",
    ESS_threshold = ess_threshold,
    diffusion_estimator = diffusion_estimator,
    seed = seed,
    n_cores = n_cores
  )

  # Output
  .gbf_output(result, n_inference, time_grid, multivariate = FALSE)
}

#' Divide-and-conquer GBF for homogeneous univariate Gaussians
#'
#' @param n_inference number of particles
#' @param samples list of k numeric vectors; element k contains samples
#'   from node k
#' @param mu length-k vector: fitted sub-posterior means
#' @param sigma length-k vector: fitted sub-posterior SDs
#' @param time_grid numeric vector with one interval per progressive fusion
#' @param precondition "identity", "variance", or numeric vector of length k
#' @param ess_threshold resample when ESS/N drops below this value
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer seed, or NULL
#' @param n_cores number of parallel cores
#' @return list(samples, ESS, CESS, elapsed)
#' @export
dc_gbf_univariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma,
  time_grid,
  precondition = "variance",
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {
  k <- length(samples)
  stopifnot(length(mu) == k, length(sigma) == k)
  .validate_time_grid(time_grid)
  if (length(time_grid) != k) {
    stop("DC-GBF requires length(time_grid) == length(samples).")
  }

  if (any(!is.finite(mu)) || any(!is.finite(sigma)) || any(sigma <= 0)) {
    stop("DC-GBF requires finite fitted Gaussian means and positive SDs.")
  }

  suppressWarnings(suppressPackageStartupMessages(require(DCFusion)))
  result <- .with_temp_working_dir(
    DCFusion::progressive_fusion_SMC_uniGaussian(
      N_schedule = rep(n_inference, k - 1L),
      time_schedule = diff(time_grid),
      base_samples = samples,
      mean = mean(mu),
      sd = mean(sigma),
      start_beta = 1 / k,
      precondition = .dc_gbf_precondition(precondition, samples, k),
      resampling_method = "multi",
      ESS_threshold = ess_threshold,
      diffusion_estimator = diffusion_estimator,
      seed = seed,
      n_cores = n_cores
    )
  )

  .dc_gbf_output(result, n_inference, seed = seed)
}

#' Balanced-binary D&C GBF for heterogeneous univariate Gaussians
#'
#' @param n_inference number of particles
#' @param samples list of k numeric vectors; element k contains samples
#'   from node k
#' @param mu length-k vector: fitted sub-posterior means
#' @param sigma length-k vector: fitted sub-posterior SDs
#' @param time_grid numeric vector used as the per-stage DCFusion time mesh
#' @param precondition "identity" or "variance"
#' @param merge_schedule optional integer factors whose product is k
#' @param n_schedule optional particle counts, one per merge factor
#' @param ess_threshold resample when ESS/N drops below this value
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer seed, or NULL
#' @param n_cores number of parallel cores
#' @return list(samples, ESS, CESS, elapsed)
#' @export
dc_gbf_het_unigaussian <- function(
  n_inference,
  samples,
  mu,
  sigma,
  time_grid,
  precondition = "variance",
  merge_schedule = NULL,
  n_schedule = NULL,
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {
  k <- length(samples)
  stopifnot(length(mu) == k, length(sigma) == k)
  .validate_time_grid(time_grid)
  if (any(!is.finite(mu)) || any(!is.finite(sigma)) || any(sigma <= 0)) {
    stop("DC-GBF-HET requires finite fitted Gaussian means and positive SDs.")
  }
  if (identical(precondition, "variance")) {
    precondition_matrices <- lapply(samples, function(sample) {
      matrix(stats::var(sample), nrow = 1L)
    })
  } else if (identical(precondition, "identity")) {
    precondition_matrices <- replicate(k, diag(1L), simplify = FALSE)
  } else if (is.numeric(precondition) && length(precondition) == k) {
    precondition_matrices <- lapply(precondition, function(value) {
      matrix(value, nrow = 1L)
    })
  } else {
    stop(
      "precondition must be 'identity', 'variance',",
      " or a numeric vector of length k"
    )
  }
  if (is.null(merge_schedule)) {
    merge_schedule <- .dc_gbf_default_merge_schedule(k)
  }
  .validate_dc_gbf_schedule(k, merge_schedule)
  if (is.null(n_schedule)) {
    n_schedule <- rep(n_inference, length(merge_schedule))
  }
  if (
    !is.numeric(n_schedule) || length(n_schedule) != length(merge_schedule) ||
      any(!is.finite(n_schedule)) || any(n_schedule != floor(n_schedule)) ||
      any(n_schedule < 1L)
  ) {
    stop("n_schedule must contain one positive integer per merge factor.")
  }

  suppressWarnings(suppressPackageStartupMessages(require(DCFusion)))
  result <- .with_temp_working_dir(
    DCFusion::bal_binary_GBF_multiGaussian(
      N_schedule = as.integer(n_schedule),
      m_schedule = as.integer(merge_schedule),
      time_mesh = time_grid,
      base_samples = lapply(samples, matrix, ncol = 1L),
      L = length(merge_schedule) + 1L,
      dim = 1L,
      mean_vecs = lapply(mu, function(value) c(value)),
      Sigmas = lapply(sigma, function(value) matrix(value^2, nrow = 1L)),
      C = k,
      precondition = precondition_matrices,
      resampling_method = "multi",
      ESS_threshold = ess_threshold,
      diffusion_estimator = diffusion_estimator,
      seed = seed,
      n_cores = n_cores
    )
  )

  .dc_gbf_multivariate_output(result, n_inference, seed = seed)
}

#' Generalised Bayesian Fusion for bivariate Gaussians
#'
#' @param n_inference number of particles
#' @param samples list of k matrices (n x 2)
#' @param mu list of k mean vectors (length 2)
#' @param sigma_matrix list of k 2x2 covariance matrices
#' @param time_grid numeric vector from 0 to T
#' @param precondition "identity" (BF) or "covariance" (GBF, recommended)
#' @param ess_threshold resampling threshold (default 0.5)
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer or NULL
#' @param n_cores number of cores
#' @return list(samples, ESS, CESS, elapsed)
#' @export
gbf_bivariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma_matrix,
  time_grid,
  precondition = "covariance",
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {
  k <- length(samples)
  stopifnot(length(mu) == k, length(sigma_matrix) == k)
  .validate_time_grid(time_grid)

  pc <- .precondition_matrices(precondition, samples, k)
  sd_corr <- .bivariate_sd_corr(sigma_matrix)

  particles <- DCFusion::initialise_particle_sets(
    samples_to_fuse = samples,
    multivariate = TRUE,
    number_of_steps = length(time_grid)
  )

  result <- DCFusion::parallel_GBF_biGaussian(
    particles_to_fuse = particles,
    N = n_inference,
    m = k,
    time_mesh = time_grid,
    mean_vecs = mu,
    sd_vecs = sd_corr$sd_vecs,
    corrs = sd_corr$corrs,
    betas = rep(1, k),
    precondition_matrices = pc$mats,
    resampling_method = "multi",
    ESS_threshold = ess_threshold,
    diffusion_estimator = diffusion_estimator,
    seed = seed,
    n_cores = n_cores
  )

  .gbf_output(result, n_inference, time_grid, multivariate = TRUE)
}

#' Generalised Bayesian Fusion for multivariate Gaussians
#'
#' @param n_inference number of particles
#' @param samples list of k matrices (n x d)
#' @param mu list of k mean vectors
#' @param sigma_matrix list of k covariance matrices
#' @param time_grid numeric vector from 0 to T
#' @param precondition "identity" (BF) or "covariance" (GBF, recommended)
#' @param ess_threshold resampling threshold (default 0.5)
#' @param diffusion_estimator "Poisson" or "NB"
#' @param seed integer or NULL
#' @param n_cores number of cores
#' @return list(samples, ESS, CESS, elapsed)
#' @export
gbf_multivariate_gaussian <- function(
  n_inference,
  samples,
  mu,
  sigma_matrix,
  time_grid,
  precondition = "covariance",
  ess_threshold = 0.5,
  diffusion_estimator = "Poisson",
  seed = NULL,
  n_cores = 1
) {
  k <- length(samples)
  d <- ncol(samples[[1]])
  stopifnot(length(mu) == k, length(sigma_matrix) == k)
  .validate_time_grid(time_grid)

  pc <- .precondition_matrices(precondition, samples, k)

  particles <- DCFusion::initialise_particle_sets(
    samples_to_fuse = samples,
    multivariate = TRUE,
    number_of_steps = length(time_grid)
  )

  result <- DCFusion::parallel_GBF_multiGaussian(
    particles_to_fuse = particles,
    N = n_inference,
    m = k,
    time_mesh = time_grid,
    dim = d,
    mean_vecs = mu,
    Sigmas = sigma_matrix,
    precondition_matrices = pc$mats,
    resampling_method = "multi",
    ESS_threshold = ess_threshold,
    diffusion_estimator = diffusion_estimator,
    seed = seed,
    n_cores = n_cores
  )

  .gbf_output(result, n_inference, time_grid, multivariate = TRUE)
}

################################################################
# Beta-logit GBF — Jacobian-corrected fusion for Beta nodes    #
################################################################

# Derive one deterministic stage seed without sharing random streams.
.derive_beta_stage_seed <- function(seed, stage_index) {
  .validate_positive_int(stage_index, "stage_index")
  if (length(seed) != 1 || !is.numeric(seed) || !is.finite(seed) ||
        seed != floor(seed) || seed < 0 || seed >= .Machine$integer.max) {
    stop("seed must be an integer between zero and the R integer maximum.")
  }

  as.integer(
    (as.double(seed) + 104729 * as.double(stage_index)) %%
      .Machine$integer.max
  )
}

# Evaluate the preconditioned DCFusion potential in logit coordinates.
.beta_logit_phi <- function(x, factor, precondition) {
  probability <- stats::plogis(x)
  exponent_sum <- factor$auxiliary_alpha + factor$auxiliary_beta
  score <- factor$auxiliary_alpha - exponent_sum * probability
  curvature <- -exponent_sum * probability * (1 - probability)
  0.5 * precondition * (score^2 + curvature)
}

# Return exact preconditioned potential bounds over a logit interval.
.beta_logit_phi_bounds <- function(lower, upper, factor, precondition) {
  if (length(lower) != 1 || length(upper) != 1 ||
        !is.finite(lower) || !is.finite(upper) || lower >= upper) {
    stop("Potential interval endpoints must be finite and increasing.")
  }
  lower_probability <- stats::plogis(lower)
  upper_probability <- stats::plogis(upper)
  exponent_sum <- factor$auxiliary_alpha + factor$auxiliary_beta
  constant <- 0.5 * factor$auxiliary_alpha^2
  linear <- -0.5 * exponent_sum * (2 * factor$auxiliary_alpha + 1)
  quadratic <- 0.5 * exponent_sum * (exponent_sum + 1)
  minimizer <- max(
    lower_probability,
    min(upper_probability, -linear / (2 * quadratic))
  )
  # Evaluate the exact potential quadratic at one probability.
  evaluate <- function(probability) {
    precondition * (
      constant + linear * probability + quadratic * probability^2
    )
  }

  list(
    lower = evaluate(minimizer),
    upper = max(evaluate(lower_probability), evaluate(upper_probability)),
    minimizer_probability = minimizer
  )
}

# Return exact global preconditioned potential bounds.
.beta_logit_phi_global_bounds <- function(factor, precondition) {
  exponent_sum <- factor$auxiliary_alpha + factor$auxiliary_beta
  constant <- 0.5 * factor$auxiliary_alpha^2
  linear <- -0.5 * exponent_sum * (2 * factor$auxiliary_alpha + 1)
  quadratic <- 0.5 * exponent_sum * (exponent_sum + 1)
  minimizer <- max(0, min(1, -linear / (2 * quadratic)))
  lower <- precondition * (
    constant + linear * minimizer + quadratic * minimizer^2
  )
  upper <- 0.5 * precondition * max(
    factor$auxiliary_alpha^2,
    factor$auxiliary_beta^2
  )

  list(lower = lower, upper = upper, minimizer_probability = minimizer)
}

# Estimate one exact-algorithm transition correction by Poisson thinning.
.beta_logit_ea_pt <- function(x0, y, s, t, factor, precondition) {
  normalized_x0 <- x0 / sqrt(precondition)
  normalized_y <- y / sqrt(precondition)
  layer <- layeredBB::bessel_layer_simulation(
    x = normalized_x0, y = normalized_y, s = s, t = t, mult = 0.1
  )
  lower <- sqrt(precondition) * layer$L
  upper <- sqrt(precondition) * layer$U
  bounds <- .beta_logit_phi_bounds(
    lower = lower, upper = upper, factor = factor, precondition = precondition
  )
  global_bounds <- .beta_logit_phi_global_bounds(factor, precondition)
  lower_phi <- bounds$lower
  upper_phi <- bounds$upper
  rate <- max(0, (upper_phi - lower_phi) * (t - s))
  poisson_count <- stats::rpois(1, rate)
  log_acceptance <- 0
  if (poisson_count > 0) {
    bridge <- layeredBB::layered_brownian_bridge(
      x = normalized_x0, y = normalized_y, s = s, t = t,
      bessel_layer = layer, times = stats::runif(poisson_count, s, t)
    )
    path <- sqrt(precondition) * bridge$simulated_path[1, ]
    phi <- .beta_logit_phi(path, factor, precondition)
    differences <- upper_phi - phi
    tolerance <- sqrt(.Machine$double.eps) * max(1, abs(upper_phi), abs(phi))
    if (any(differences < -tolerance)) {
      stop("The exact potential exceeded its analytic interval upper bound.")
    }
    differences <- pmax(0, differences)
    log_acceptance <- sum(log(differences))
  }
  gap <- upper_phi - lower_phi
  if (poisson_count > 0 && gap <= 0) {
    stop("A positive Poisson count requires a positive potential range.")
  }
  log_weight <- -(lower_phi - global_bounds$lower) * (t - s)
  if (poisson_count > 0) {
    log_weight <- log_weight - poisson_count * log(gap) + log_acceptance
  }

  log_weight
}

# Verify terminal coalescence across every node coordinate.
.validate_beta_coalescence <- function(particles, num_nodes) {
  valid <- vapply(
    particles,
    function(values) {
      is.numeric(values) && length(values) == num_nodes &&
        all(is.finite(values))
    },
    logical(1)
  )
  if (length(particles) < 1 || any(!valid)) {
    stop("Every terminal particle must contain one finite value per node.")
  }
  values <- unlist(particles, use.names = FALSE)
  tolerance <- sqrt(.Machine$double.eps) * max(1, abs(values))
  spreads <- vapply(particles, function(value) diff(range(value)), numeric(1))
  max_spread <- max(spreads)
  if (max_spread > tolerance) {
    stop("Terminal node coordinates did not coalesce within tolerance.")
  }

  list(max_spread = max_spread, tolerance = tolerance)
}

# Apply the DCFusion initial importance proposal without a socket cluster.
.rho_is_beta_logit_serial <- function(
  particles_to_fuse, n_inference, num_nodes, end_time, precondition_values,
  number_of_steps, time_mesh, seed
) {
  for (node_index in seq_len(num_nodes)) {
    if (particles_to_fuse[[node_index]]$N != n_inference) {
      particles_to_fuse[[node_index]] <- DCFusion::resample_particle_y_samples(
        N = n_inference,
        particle_set = particles_to_fuse[[node_index]],
        multivariate = FALSE,
        resampling_method = "multi",
        seed = .derive_beta_stage_seed(seed, node_index)
      )
    }
  }
  combined_weights <- rowSums(data.frame(lapply(
    particles_to_fuse, function(particle) particle$log_weights
  )))
  samples_to_fuse <- lapply(
    particles_to_fuse, function(particle) particle$y_samples
  )
  set.seed(.derive_beta_stage_seed(seed, num_nodes + 1))
  proposed <- DCFusion::rho_IS_univariate_(
    samples_to_fuse = samples_to_fuse,
    N = n_inference,
    m = num_nodes,
    time = end_time,
    precondition_values = precondition_values
  )
  normalized <- DCFusion::particle_ESS(
    log_weights = combined_weights + proposed$log_weights
  )
  particle_set <- new.env(parent = emptyenv())
  particle_set$y_samples <- rep(NA_real_, n_inference)
  particle_set$x_samples <- proposed$x_samples
  particle_set$x_means <- proposed$x_means
  particle_set$log_weights <- normalized$log_weights
  particle_set$normalised_weights <- normalized$normalised_weights
  particle_set$ESS <- normalized$ESS
  particle_set$CESS <- c(
    DCFusion::particle_ESS(log_weights = proposed$log_weights)$ESS,
    rep(NA_real_, number_of_steps - 1)
  )
  particle_set$resampled <- rep(FALSE, number_of_steps)
  particle_set$N <- n_inference
  particle_set$number_of_steps <- number_of_steps
  particle_set$time_mesh <- time_mesh
  class(particle_set) <- "particle"

  particle_set
}

# Apply exact Beta-logit corrections across the fixed time mesh.
.rho_j_beta_logit <- function(
  particle_set, num_nodes, time_mesh, factors, precondition_values,
  ess_threshold, seed
) {
  n_inference <- particle_set$N
  interval_count <- length(time_mesh) - 1
  elapsed <- rep(NA_real_, interval_count)
  ess <- c(particle_set$ESS[1], rep(NA_real_, interval_count))
  cess <- c(particle_set$CESS[1], rep(NA_real_, interval_count))
  adaptive_resampled <- rep(FALSE, interval_count)
  proposal_seeds <- integer(interval_count)
  exact_algorithm_seeds <- integer(interval_count)
  adaptive_seeds <- integer(interval_count)
  precondition_matrices <- lapply(precondition_values, as.matrix)
  lambda <- DCFusion::inverse_sum_matrices(
    lapply(1 / precondition_values, as.matrix)
  )
  end_time <- tail(time_mesh, 1)

  for (step in 2:length(time_mesh)) {
    start_time <- proc.time()
    interval_index <- step - 1
    proposal_seeds[interval_index] <- .derive_beta_stage_seed(
      seed, 3 * interval_index + 1
    )
    exact_algorithm_seeds[interval_index] <- .derive_beta_stage_seed(
      seed, 3 * interval_index + 2
    )
    adaptive_seeds[interval_index] <- .derive_beta_stage_seed(
      seed, 3 * interval_index + 3
    )
    if (particle_set$ESS < n_inference * ess_threshold) {
      adaptive_resampled[interval_index] <- TRUE
      particle_set <- DCFusion::resample_particle_x_samples(
        N = n_inference,
        particle_set = particle_set,
        multivariate = FALSE,
        step = interval_index,
        resampling_method = "multi",
        seed = adaptive_seeds[interval_index]
      )
    }
    proposal_covariance <- DCFusion::construct_V(
      s = time_mesh[step - 1],
      t = time_mesh[step],
      end_time = end_time,
      C = num_nodes,
      d = 1,
      precondition_matrices = precondition_matrices,
      Lambda = lambda,
      iteration = step
    )
    set.seed(proposal_seeds[interval_index])
    proposed <- vector("list", n_inference)
    proposed_means <- numeric(n_inference)
    for (particle_index in seq_len(n_inference)) {
      proposal_mean <- DCFusion::construct_M(
        s = time_mesh[step - 1],
        t = time_mesh[step],
        end_time = end_time,
        C = num_nodes,
        d = 1,
        sub_posterior_samples = particle_set$x_samples[[particle_index]],
        sub_posterior_mean = particle_set$x_means[particle_index],
        iteration = step
      )
      proposed[[particle_index]] <- as.vector(
        mvtnorm::rmvnorm(1, mean = proposal_mean, sigma = proposal_covariance)
      )
      # Project singular-normal noise onto the exact terminal diagonal.
      if (step == length(time_mesh)) {
        terminal_value <- DCFusion::weighted_mean_univariate(
          x = proposed[[particle_index]], weights = 1 / precondition_values
        )
        proposed[[particle_index]] <- rep(terminal_value, num_nodes)
      }
      proposed_means[particle_index] <- DCFusion::weighted_mean_univariate(
        x = proposed[[particle_index]], weights = 1 / precondition_values
      )
    }
    set.seed(exact_algorithm_seeds[interval_index])
    log_weights <- numeric(n_inference)
    for (particle_index in seq_len(n_inference)) {
      log_weights[particle_index] <- sum(vapply(
        seq_len(num_nodes),
        function(node_index) {
          .beta_logit_ea_pt(
            x0 = particle_set$x_samples[[particle_index]][node_index],
            y = proposed[[particle_index]][node_index],
            s = time_mesh[step - 1],
            t = time_mesh[step],
            factor = factors[[node_index]],
            precondition = precondition_values[node_index]
          )
        },
        numeric(1)
      ))
    }
    particle_set$x_samples <- proposed
    particle_set$x_means <- proposed_means
    normalized <- DCFusion::particle_ESS(
      log_weights = particle_set$log_weights + log_weights
    )
    particle_set$log_weights <- normalized$log_weights
    particle_set$normalised_weights <- normalized$normalised_weights
    particle_set$ESS <- normalized$ESS
    ess[step] <- particle_set$ESS
    particle_set$CESS[step] <- DCFusion::particle_ESS(
      log_weights = log_weights
    )$ESS
    cess[step] <- particle_set$CESS[step]
    elapsed[interval_index] <- (proc.time() - start_time)["elapsed"]
  }

  coalescence <- .validate_beta_coalescence(
    particle_set$x_samples, num_nodes
  )
  weighted_logits <- vapply(
    particle_set$x_samples, function(values) values[1], numeric(1)
  )
  normalized_weights <- particle_set$normalised_weights
  if (any(!is.finite(normalized_weights)) || any(normalized_weights < 0) ||
        abs(sum(normalized_weights) - 1) > sqrt(.Machine$double.eps)) {
    stop("Terminal particle weights must be finite and sum to one.")
  }
  particle_set$y_samples <- weighted_logits
  terminal_seed <- .derive_beta_stage_seed(seed, 3 * interval_count + 4)
  particle_set <- DCFusion::resample_particle_y_samples(
    N = n_inference,
    particle_set = particle_set,
    multivariate = FALSE,
    resampling_method = "multi",
    seed = terminal_seed
  )

  list(
    weighted_logits = weighted_logits,
    normalized_weights = normalized_weights,
    equal_weight_logits = particle_set$y_samples,
    ess = ess,
    cess = cess,
    adaptive_resampled = adaptive_resampled,
    elapsed = elapsed,
    max_terminal_spread = coalescence$max_spread,
    terminal_coalescence_tolerance = coalescence$tolerance
  )
}

#' Generalised Bayesian Fusion for Beta nodes via a Jacobian-corrected logit
#' transform
#'
#' DCFusion has no native Beta entry point, so each node's raw Beta samples
#' are first fit (fixed-support maximum likelihood) and transported into a
#' Jacobian-corrected auxiliary logit coordinate system (`beta_math.R`'s
#' `fit_beta_components`/`select_jacobian_shares`/`build_logit_factors`),
#' then fused with an exact-algorithm SMC sampler over that coordinate
#' system, and finally mapped back through the inverse logit.
#'
#' @param n_inference number of particles
#' @param samples list of k numeric vectors of raw (0, 1)-valued Beta draws;
#'   element k contains samples from node k
#' @param time_grid numeric vector {0 = t_0 < t_1 < ... < t_n = T}
#'   (length >= 2)
#' @param ess_threshold resample when ESS/N drops below this value
#'   (default 0.5)
#' @param seed integer seed
#'
#' @return list with:
#'   samples — length-n_inference equally-weighted draws in (0, 1)
#'   ESS — vector of ESS values at each time step
#'   CESS — vector of conditional ESS values at each time step
#'   elapsed — wall-clock seconds per grid interval
#'   fitted_nodes — per-node fitted alpha/beta shapes
#'   jacobian_shares — per-node Jacobian shares used in the correction
#' @export
gbf_beta_logit <- function(
  n_inference, samples, time_grid, ess_threshold = 0.5, seed
) {
  .validate_positive_int(n_inference, "n_inference")
  .validate_time_grid(time_grid)
  if (length(ess_threshold) != 1 || !is.numeric(ess_threshold) ||
        !is.finite(ess_threshold) || ess_threshold < 0 ||
        ess_threshold > 1) {
    stop("ess_threshold must be between zero and one.")
  }
  num_nodes <- length(samples)
  preprocessed <- preprocess_beta_gbf(do.call(cbind, samples))
  fitted_nodes <- preprocessed$fitted_nodes
  jacobian_shares <- preprocessed$jacobian_shares
  factors <- build_logit_factors(fitted_nodes, jacobian_shares)
  auxiliary_samples <- lapply(
    seq_len(num_nodes), function(index) preprocessed$auxiliary_logits[, index]
  )
  precondition_values <- vapply(auxiliary_samples, stats::var, numeric(1))
  if (any(!is.finite(precondition_values)) || any(precondition_values <= 0)) {
    stop("Every auxiliary logit sample must have positive finite variance.")
  }

  particles <- DCFusion::initialise_particle_sets(
    samples_to_fuse = auxiliary_samples,
    multivariate = FALSE,
    number_of_steps = length(time_grid)
  )
  initial_seed <- .derive_beta_stage_seed(seed, 1)
  start_time <- proc.time()
  initial <- .rho_is_beta_logit_serial(
    particles_to_fuse = particles,
    n_inference = n_inference,
    num_nodes = num_nodes,
    end_time = tail(time_grid, 1),
    precondition_values = precondition_values,
    number_of_steps = length(time_grid),
    time_mesh = time_grid,
    seed = initial_seed
  )
  corrected <- .rho_j_beta_logit(
    particle_set = initial,
    num_nodes = num_nodes,
    time_mesh = time_grid,
    factors = factors,
    precondition_values = precondition_values,
    ess_threshold = ess_threshold,
    seed = .derive_beta_stage_seed(seed, 2)
  )
  runtime_seconds <- unname((proc.time() - start_time)["elapsed"])

  list(
    samples = stats::plogis(corrected$equal_weight_logits),
    ESS = corrected$ess,
    CESS = corrected$cess,
    elapsed = corrected$elapsed,
    adaptive_resampled = corrected$adaptive_resampled,
    max_terminal_spread = corrected$max_terminal_spread,
    terminal_coalescence_tolerance =
      corrected$terminal_coalescence_tolerance,
    runtime_seconds = runtime_seconds,
    fitted_nodes = fitted_nodes,
    jacobian_shares = jacobian_shares,
    preprocessing_seconds = preprocessed$preprocessing_seconds
  )
}

##############
# Regression #
##############

#' Point-estimate wrapper: runs a Fusion sampler and returns its posterior mean
#'
#' Generic regression-style wrapper around any of the sampling functions in
#' this file (mcf_*, bf_*, gbf_*): calls the sampler and reduces its samples
#' to a posterior mean point estimate, mirroring the Sampling-to-Regression
#' relationship without introducing new statistics or a D covariate.
#'
#' @param sampling_fn one of the *_gaussian sampling functions in this file
#' @param ... arguments forwarded to sampling_fn
#' @return numeric scalar (univariate) or numeric vector
#'   (bivariate/multivariate) giving the posterior mean of the fused samples
#' @export
fusion_regression <- function(sampling_fn, ...) {
  result <- sampling_fn(...)
  if (is.matrix(result$samples)) {
    colMeans(result$samples)
  } else {
    mean(result$samples)
  }
}
