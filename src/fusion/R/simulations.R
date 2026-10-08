###############
# Simulations #
###############

# Vector-per-node DGP-parameter keys (mu for bivariate/multivariate Gaussian)
.vector_dgp_keys <- c("mu")

# Matrix-per-node DGP-parameter keys (cov for bivariate/multivariate Gaussian)
.matrix_dgp_keys <- c("cov")

# Report whether a result schema omits quadratic parameter expansions
#' @noRd
.is_compact_result_schema <- function(schema) {
  if (!is.null(schema$compact)) {
    return(isTRUE(schema$compact))
  }
  schema$max_p >= 50L
}

# Build a diagnostic plot filename encoding the config/DGP parameters
#' @noRd
.plot_filename <- function(prefix, parametric_form, method, n, n_sample,
                           dgp_parameters, dgp, iteration_id) {
  fn_parts <- if (parametric_form == "univariate_Gaussian") {
    # Encode the per-node mu/sigma values directly in the filename
    paste(
      prefix,
      method,
      parametric_form,
      paste0("n_", n),
      paste0("n_sample_", n_sample),
      paste0("mu_", paste(dgp_parameters$mu, collapse = "_")),
      paste0("sigma_", paste(dgp_parameters$sigma, collapse = "_")),
      paste0("iter_", iteration_id),
      sep = "_"
    )
  } else {
    # Multivariate/bivariate: encode dimensionality instead of full parameters
    paste(
      prefix,
      method,
      parametric_form,
      paste0("n_", n),
      paste0("n_sample_", n_sample),
      paste0("p_", dgp$p),
      paste0("iter_", iteration_id),
      sep = "_"
    )
  }
  paste0(fn_parts, ".pdf")
}

# Construct the DGP for the given parametric form
#' @noRd
.build_dgp <- function(parametric_form, dgp_parameters) {
  if (parametric_form == "univariate_Gaussian") {
    # Univariate Gaussian: product of independent per-node Gaussians
    univariate_gaussian_dgp(
      mu = dgp_parameters$mu,
      sigma = dgp_parameters$sigma
    )
  } else if (parametric_form == "multivariate_Gaussian") {
    # Multivariate Gaussian: product of independent per-node Gaussian vectors
    multivariate_gaussian_dgp(
      mu = dgp_parameters$mu,
      cov = dgp_parameters$cov
    )
  } else if (parametric_form == "bivariate_Gaussian") {
    # Bivariate Gaussian: product of independent per-node bivariate Gaussians
    bivariate_gaussian_dgp(
      mu = dgp_parameters$mu,
      cov = dgp_parameters$cov
    )
  } else if (parametric_form == "Beta") {
    # Beta: product of independent per-node Beta distributions
    beta_dgp(
      a = dgp_parameters$a,
      b = dgp_parameters$b
    )
  } else {
    stop(paste("Unsupported parametric form:", parametric_form))
  }
}

# Build the SSH mesh-guidance samples (raw data, or Beta's auxiliary logits)
#' @noRd
.guidance_samples <- function(parametric_form, data, num_nodes) {
  if (identical(parametric_form, "Beta")) {
    preprocessed <- preprocess_beta_gbf(do.call(cbind, data))
    return(lapply(
      seq_len(num_nodes),
      function(index) preprocessed$auxiliary_logits[, index]
    ))
  }

  data
}

# Resolve the coalescence time, number of time steps, and time grid
#' @noRd
.resolve_time_grid <- function(method, data, num_nodes, n_sample,
                               coalescence_time_cfg, n_time_steps_cfg,
                               parametric_form = NULL,
                               covariance_structure = "full") {
  # Run SSH guidance if coalescence_time or n_time_steps is "optimal"
  opt_needed <- as.character(coalescence_time_cfg) == "optimal" ||
    as.character(n_time_steps_cfg) == "optimal"

  if (opt_needed) {
    guidance_method <- if (identical(method, "DC-GBF")) "GBF" else method
    opt <- optimal_time_grid(
      method = guidance_method,
      samples = .guidance_samples(parametric_form, data, num_nodes),
      k = num_nodes,
      n = n_sample,
      covariance_structure = covariance_structure
    )
  }

  # Use the guided coalescence time, or the configured fixed value
  coalescence_time <- if (as.character(coalescence_time_cfg) == "optimal") {
    opt$coalescence_time
  } else {
    as.numeric(coalescence_time_cfg)
  }

  # Use the guided number of time steps, or the configured fixed value
  n_time_steps <- if (as.character(n_time_steps_cfg) == "optimal") {
    opt$n_time_steps
  } else {
    as.integer(n_time_steps_cfg)
  }

  both_optimal <- opt_needed &&
    as.character(coalescence_time_cfg) == "optimal" &&
    as.character(n_time_steps_cfg) == "optimal"

  # Use one progressive-fusion interval per merge for homogeneous D&C GBF
  if (both_optimal && identical(method, "DC-GBF")) {
    time_grid <- seq(0, coalescence_time, length.out = num_nodes)
    n_time_steps <- as.integer(length(time_grid) - 1L)
  } else if (both_optimal) {
    time_grid <- opt$time_grid
  } else {
    time_grid <- seq(0, coalescence_time, length.out = n_time_steps + 1L)
  }

  list(
    coalescence_time = coalescence_time,
    n_time_steps = n_time_steps,
    time_grid = time_grid
  )
}

# Estimate per-node sub-posterior parameters from raw samples
#' @noRd
.estimate_subposterior_params <- function(
  parametric_form, data, covariance_structure = "full"
) {
  if (parametric_form == "univariate_Gaussian") {
    list(mu_hat = sapply(data, mean), sigma_hat = sapply(data, sd))
  } else if (parametric_form %in%
               c("multivariate_Gaussian", "bivariate_Gaussian")) {
    list(
      mu_hat = lapply(data, colMeans),
      sigma_matrix_hat = lapply(
        data, .estimate_sample_covariance,
        covariance_structure = covariance_structure
      )
    )
  }
}

# List the (key, node, dims) specs for the scalar-per-node DGP-parameter keys
#' @noRd
.scalar_dgp_specs <- function(schema) {
  specs <- list()
  for (key in schema$scalar_keys) {
    for (node in seq_len(schema$max_num_nodes)) {
      specs[[length(specs) + 1]] <- list(
        key = key, node = node, dims = integer(0)
      )
    }
  }
  specs
}

# List the (key, node, dims) specs for the vector-per-node DGP-parameter keys
#' @noRd
.vector_dgp_specs <- function(schema) {
  specs <- list()
  for (key in .vector_dgp_keys) {
    for (node in seq_len(schema$max_num_nodes)) {
      for (dim in seq_len(schema$max_p)) {
        specs[[length(specs) + 1]] <- list(key = key, node = node, dims = dim)
      }
    }
  }
  specs
}

# List the (key, node, dims) specs for the matrix-per-node DGP-parameter keys
#' @noRd
.matrix_dgp_specs <- function(schema) {
  key <- .matrix_dgp_keys[1]
  specs <- list()
  for (node in seq_len(schema$max_num_nodes)) {
    for (d1 in seq_len(schema$max_p)) {
      for (d2 in d1:schema$max_p) {
        specs[[length(specs) + 1]] <- list(
          key = key, node = node, dims = c(d1, d2)
        )
      }
    }
  }
  specs
}

# List every (key, node, dims) triple spanning the DGP-parameter columns
#' @noRd
.iter_dgp_param_specs <- function(schema) {
  specs <- .scalar_dgp_specs(schema)
  if (.is_compact_result_schema(schema)) {
    return(specs)
  }
  c(specs, .vector_dgp_specs(schema), .matrix_dgp_specs(schema))
}

# Build the ordered list of DGP-parameter CSV column names for the given schema
#' @noRd
.dgp_param_columns <- function(schema) {
  specs <- .iter_dgp_param_specs(schema)
  vapply(specs, function(spec) {
    do.call(.dgp_param_column, c(list(spec$key, spec$node), as.list(spec$dims)))
  }, character(1))
}

# Look up a scalar-per-node DGP-parameter value for one spec, or NA
#' @noRd
.scalar_dgp_spec_value <- function(spec, scalar_keys, num_nodes,
                                   dgp_parameters) {
  if (!(spec$key %in% scalar_keys) || spec$node > num_nodes) {
    return(NA_real_)
  }
  dgp_parameters[[spec$key]][[spec$node]]
}

# Look up a vector-per-node DGP-parameter value for one spec, or NA
#' @noRd
.vector_dgp_spec_value <- function(spec, is_vector_form, num_nodes, mu_list) {
  if (!is_vector_form || spec$node > num_nodes) {
    return(NA_real_)
  }
  dim <- spec$dims[1]
  node_mu <- mu_list[[spec$node]]
  if (dim > length(node_mu)) {
    return(NA_real_)
  }
  node_mu[dim]
}

# Look up a matrix-per-node DGP-parameter value for one spec, or NA
#' @noRd
.matrix_dgp_spec_value <- function(spec, is_vector_form, num_nodes, cov_list) {
  if (!is_vector_form || spec$node > num_nodes) {
    return(NA_real_)
  }
  d1 <- spec$dims[1]
  d2 <- spec$dims[2]
  node_cov <- cov_list[[spec$node]]
  if (d1 > nrow(node_cov) || d2 > ncol(node_cov)) {
    return(NA_real_)
  }
  node_cov[d1, d2]
}

# Dispatch to the scalar/vector/matrix value lookup for one DGP-parameter spec
#' @noRd
.dgp_param_spec_value <- function(spec, scalar_keys, is_vector_form, num_nodes,
                                  mu_list, cov_list, dgp_parameters) {
  n_dims <- length(spec$dims)
  if (n_dims == 0) {
    return(.scalar_dgp_spec_value(spec, scalar_keys, num_nodes, dgp_parameters))
  }
  if (n_dims == 1) {
    return(.vector_dgp_spec_value(spec, is_vector_form, num_nodes, mu_list))
  }
  .matrix_dgp_spec_value(spec, is_vector_form, num_nodes, cov_list)
}

# Build DGP-parameter values (NA where inapplicable) in .dgp_param_columns order
#' @noRd
.dgp_param_values <- function(schema, parametric_form, dgp_parameters) {
  scalar_keys <- .scalar_keys_by_form[[parametric_form]]
  if (is.null(scalar_keys)) {
    scalar_keys <- character(0)
  }
  is_vector_form <- parametric_form %in%
    c("bivariate_Gaussian", "multivariate_Gaussian")
  mu_list <- if (is_vector_form) dgp_parameters$mu else NULL
  cov_list <- if (is_vector_form) dgp_parameters$cov else NULL
  num_nodes <- if (is_vector_form) {
    length(mu_list)
  } else if (length(scalar_keys) > 0) {
    length(dgp_parameters[[scalar_keys[1]]])
  } else {
    0
  }

  values <- list()
  for (spec in .iter_dgp_param_specs(schema)) {
    column <- do.call(
      .dgp_param_column, c(list(spec$key, spec$node), as.list(spec$dims))
    )
    values[[column]] <- .dgp_param_spec_value(
      spec, scalar_keys, is_vector_form, num_nodes, mu_list, cov_list,
      dgp_parameters
    )
  }
  values
}

# Build the weight-matrix CSV column names for schema (R keeps them for parity)
#' @noRd
.weight_columns <- function(schema) {
  if (.is_compact_result_schema(schema)) {
    return(character(0))
  }
  max_dim <- max(schema$max_p, 1)
  columns <- character(0)
  for (node in seq_len(schema$max_num_nodes)) {
    for (d1 in seq_len(max_dim)) {
      for (d2 in d1:max_dim) {
        columns <- c(columns, .dgp_param_column("w", node, d1, d2))
      }
    }
  }
  columns
}

# Build the sampling-weight-matrix values, always NA on the R side
#' @noRd
.weight_na_values <- function(schema) {
  columns <- .weight_columns(schema)
  stats::setNames(as.list(rep(NA_real_, length(columns))), columns)
}

# Build the PMC-engine-only result values, always NA on the R side except h
#' @noRd
.pmc_engine_na_values <- function(h_name, uses_cnf) {
  values <- list(
    algorithm = NA_character_,
    sampling_method = NA_character_,
    discrepancy_method = NA_character_,
    n_transforms = NA_integer_,
    hidden_features = NA_character_,
    n_training_steps = NA_integer_,
    optimizer = NA_character_,
    weight_decay = NA_real_,
    base_learning_rate = NA_real_,
    learning_rate_schedule = NA_character_,
    batch_size = NA_integer_,
    h = h_name,
    degree = NA_integer_,
    input_degree = NA_integer_,
    activation = NA_character_,
    discrepancy_centering = NA_character_,
    ensemble_size = NA_integer_,
    regression_loss = NA_character_,
    huber_delta = NA_real_,
    loss_weighting = NA_character_,
    target_bandwidth_quantile = NA_real_,
    target_bandwidth_scale = NA_real_,
    alpha = NA_real_,
    gp_kernel = NA_character_,
    gp_inducing_count = NA_integer_,
    gp_maximum_updates = NA_integer_,
    gp_natural_gradient_learning_rate = NA_real_,
    gp_minimum_epochs = NA_integer_,
    gp_patience = NA_integer_,
    gp_relative_tolerance = NA_real_,
    gp_learn_inducing_locations = NA,
    gp_torch_num_threads = NA_integer_,
    gp_confidence_level = NA_real_,
    gp_local_neighborhood_exponent = NA_real_,
    gp_minimum_local_count = NA_integer_,
    gp_maximum_local_count = NA_integer_,
    gp_full_observation_count = NA_integer_,
    gp_requested_local_count = NA_integer_,
    gp_actual_local_count = NA_integer_,
    gp_local_fraction = NA_real_,
    gp_neighborhood_radius = NA_real_,
    gp_posterior_standard_deviation = NA_real_,
    gp_interval_lower = NA_real_,
    gp_interval_upper = NA_real_,
    gp_interval_method = NA_character_,
    gp_diagnostics = NA_character_,
    converged = NA,
    relative_improvement = NA_real_,
    training_log_likelihood = NA_real_
  )
  # Drop the CNF-only columns entirely when no config in this batch uses CNF
  if (!uses_cnf) {
    values[c("n_transforms", "training_log_likelihood")] <- NULL
  }
  values
}

# Read user plus system CPU process time from a proc.time value
#' @noRd
.process_time_seconds <- function(time_value) {
  unname(time_value[["user.self"]] + time_value[["sys.self"]])
}

# Build cheap hardware/context values for runtime comparisons
#' @noRd
.hardware_context_values <- function() {
  sys_info <- Sys.info()
  list(
    hostname = unname(sys_info[["nodename"]]),
    platform_system = unname(sys_info[["sysname"]]),
    platform_release = unname(sys_info[["release"]]),
    platform_machine = unname(sys_info[["machine"]]),
    cpu_count_logical = parallel::detectCores(logical = TRUE),
    OMP_NUM_THREADS = Sys.getenv("OMP_NUM_THREADS", unset = NA_character_),
    MKL_NUM_THREADS = Sys.getenv("MKL_NUM_THREADS", unset = NA_character_),
    OPENBLAS_NUM_THREADS = Sys.getenv(
      "OPENBLAS_NUM_THREADS", unset = NA_character_
    ),
    CUDA_VISIBLE_DEVICES = Sys.getenv(
      "CUDA_VISIBLE_DEVICES", unset = NA_character_
    ),
    torch_cuda_available = NA,
    torch_cuda_device_name = NA_character_,
    simulation_device = "cpu"
  )
}

# Return the ordered result columns expected for a DGP schema
#' @noRd
.result_columns <- function(schema, paired_methods = FALSE) {
  core_columns <- c(
    "parametric_form", "num_nodes", "p", "n", "n_sample", "method",
    "preprocessing", "n_inference", "seed"
  )
  if (paired_methods) {
    core_columns <- c(core_columns, "data_seed", "inference_seed")
  }
  core_columns <- c(
    core_columns,
    "timestamp", "iteration_id", "training_time_seconds",
    "training_process_time_seconds", "training_cuda_time_seconds",
    "simulation_time_seconds", "simulation_process_time_seconds",
    "simulation_cuda_time_seconds", names(.hardware_context_values())
  )
  pmc_columns <- names(.pmc_engine_na_values("identity", schema$uses_cnf))
  c(
    core_columns,
    .dgp_param_columns(schema), .weight_columns(schema), pmc_columns,
    "coalescence_time", "n_time_steps", "rho_acceptance_probability",
    "Q_acceptance_probability", "acceptance_probability", "final_ESS",
    "final_CESS", "beta_fitted_nodes", "beta_jacobian_shares",
    "beta_max_terminal_spread", "beta_preprocessing_seconds",
    "kl", "w1", "w2", "mse", "C_hat", "true_mean"
  )
}

# Extract the acceptance-probability diagnostics from an MCF result
#' @noRd
.mcf_diagnostics <- function(result) {
  list(
    rho_acceptance_probability = result$rho_acceptance_probability,
    Q_acceptance_probability = result$Q_acceptance_probability,
    acceptance_probability = result$acceptance_probability,
    final_ess = NA_real_,
    final_cess = NA_real_
  )
}

# Extract the ESS/CESS diagnostics from a BF or GBF result
#' @noRd
.particle_diagnostics <- function(result) {
  list(
    rho_acceptance_probability = NA_real_,
    Q_acceptance_probability = NA_real_,
    acceptance_probability = NA_real_,
    final_ess = tail(result$ESS, 1),
    final_cess = tail(result$CESS, 1)
  )
}

# Encode per-node fitted Beta shapes without commas or quotes for CSV safety.
.encode_beta_fitted_nodes <- function(fitted_nodes) {
  paste(
    vapply(fitted_nodes, function(node) {
      sprintf("alpha=%.17g;beta=%.17g", node$alpha, node$beta)
    }, character(1)),
    collapse = "|"
  )
}

# Build the Beta-GBF-only result values, NA unless a Beta result is given
#' @noRd
.beta_fusion_values <- function(result = NULL) {
  if (is.null(result)) {
    return(list(
      beta_fitted_nodes = NA_character_,
      beta_jacobian_shares = NA_character_,
      beta_max_terminal_spread = NA_real_,
      beta_preprocessing_seconds = NA_real_
    ))
  }

  list(
    beta_fitted_nodes = .encode_beta_fitted_nodes(result$fitted_nodes),
    beta_jacobian_shares = paste(
      sprintf("%.17g", result$jacobian_shares), collapse = ";"
    ),
    beta_max_terminal_spread = result$max_terminal_spread,
    beta_preprocessing_seconds = result$preprocessing_seconds
  )
}

# Assemble the unified results row (shared column schema with Python runner)
#' @noRd
.build_result_row <- function(config, dgp_schema, parametric_form, num_nodes,
                              n, method, n_inference, dgp_parameters, dgp,
                              coalescence_time, n_time_steps, n_sample,
                              training_time_seconds,
                              training_cpu_seconds,
                              simulation_time_seconds,
                              simulation_cpu_seconds,
                              hardware_context, diagnostics, kl_divergence,
                              wasserstein_distance, h_name, c_hat, mse,
                              true_mean, beta_result = NULL) {
  # p is always populated: 1 for univariate_Gaussian/Beta, else DGP dimension
  p <- if (parametric_form %in% c("univariate_Gaussian", "Beta")) {
    1L
  } else {
    dgp$p
  }

  seed_values <- list(
    seed = if (!is.null(config$seed)) config$seed else NA_integer_
  )
  if (!is.null(config$seed_strategy) &&
        config$seed_strategy == "paired_methods") {
    seed_values$data_seed <- config$data_seed
    seed_values$inference_seed <- config$inference_seed
  }

  core <- c(list(
    parametric_form = parametric_form,
    num_nodes = num_nodes,
    p = p,
    n = n,
    n_sample = n_sample,
    method = method,
    preprocessing = config$preprocessing,
    n_inference = n_inference
  ), seed_values, list(
    timestamp = config$timestamp,
    iteration_id = config$iteration_id,
    training_time_seconds = training_time_seconds,
    training_process_time_seconds = training_cpu_seconds,
    training_cuda_time_seconds = NA_real_,
    simulation_time_seconds = simulation_time_seconds,
    simulation_process_time_seconds = simulation_cpu_seconds,
    simulation_cuda_time_seconds = NA_real_
  ))
  core <- c(core, hardware_context)

  fusion_only <- c(list(
    coalescence_time = coalescence_time,
    n_time_steps = n_time_steps,
    rho_acceptance_probability = diagnostics$rho_acceptance_probability,
    Q_acceptance_probability = diagnostics$Q_acceptance_probability,
    acceptance_probability = diagnostics$acceptance_probability,
    final_ESS = diagnostics$final_ess,
    final_CESS = diagnostics$final_cess
  ), .beta_fusion_values(beta_result))

  metrics <- list(
    kl = kl_divergence,
    w1 = wasserstein_distance$w1,
    w2 = wasserstein_distance$w2,
    mse = mse,
    C_hat = c_hat,
    true_mean = true_mean
  )

  c(
    core,
    .dgp_param_values(dgp_schema, parametric_form, dgp_parameters),
    .weight_na_values(dgp_schema),
    .pmc_engine_na_values(h_name, dgp_schema$uses_cnf),
    fusion_only,
    metrics
  )
}

# Per-parametric-form dispatch info: the covariance-argument name expected by
# the mcf_*/bf_*/gbf_* samplers, the GBF precondition value, and the name of
# each method's sampler function
.fusion_dispatch <- list(
  univariate_Gaussian = list(
    cov_arg = "sigma",
    gbf_precondition = "variance",
    MCF = "mcf_univariate_gaussian",
    BF = "bf_univariate_gaussian",
    GBF = "gbf_univariate_gaussian",
    DC_GBF = "dc_gbf_univariate_gaussian",
    DC_GBF_HET = "dc_gbf_het_unigaussian"
  ),
  multivariate_Gaussian = list(
    cov_arg = "sigma_matrix",
    gbf_precondition = "covariance",
    MCF = "mcf_multivariate_gaussian",
    BF = "bf_multivariate_gaussian",
    GBF = "gbf_multivariate_gaussian"
  ),
  bivariate_Gaussian = list(
    cov_arg = "sigma_matrix",
    gbf_precondition = "covariance",
    MCF = "mcf_bivariate_gaussian",
    BF = "bf_bivariate_gaussian",
    GBF = "gbf_bivariate_gaussian"
  )
)

# Run the Beta-logit GBF specialization and compute its diagnostics
#' @noRd
.run_beta_fusion <- function(method, n_inference, data, time_grid, seed) {
  if (!identical(method, "GBF")) {
    stop(paste("Unsupported method for Beta:", method))
  }
  result <- gbf_beta_logit(
    n_inference = n_inference, samples = data, time_grid = time_grid,
    seed = seed
  )

  list(result = result, diagnostics = .particle_diagnostics(result))
}

# Report whether a univariate Gaussian config has homogeneous node parameters
#' @noRd
.is_homogeneous_unigaussian <- function(parametric_form, dgp_parameters) {
  if (!identical(parametric_form, "univariate_Gaussian")) {
    return(FALSE)
  }
  mu <- as.numeric(dgp_parameters$mu)
  sigma <- as.numeric(dgp_parameters$sigma)
  length(mu) > 0 && length(sigma) == length(mu) &&
    isTRUE(all.equal(mu, rep(mu[[1]], length(mu)))) &&
    isTRUE(all.equal(sigma, rep(sigma[[1]], length(sigma))))
}

# Run the configured fusion sampler and compute its diagnostics
#' @noRd
.run_fusion <- function(parametric_form, method, n_inference, data, mu_hat,
                        cov_hat, coalescence_time, time_grid, seed,
                        dgp_parameters = NULL,
                        covariance_structure = "full") {
  if (identical(parametric_form, "Beta")) {
    return(.run_beta_fusion(method, n_inference, data, time_grid, seed))
  }
  if (
    identical(method, "DC-GBF") &&
      !.is_homogeneous_unigaussian(parametric_form, dgp_parameters)
  ) {
    stop(
      "DC-GBF is supported only for homogeneous univariate Gaussian configs."
    )
  }
  spec <- .fusion_dispatch[[parametric_form]]
  if (is.null(spec)) {
    stop(paste("Unsupported parametric form:", parametric_form))
  }
  method_key <- if (identical(method, "DC-GBF")) {
    "DC_GBF"
  } else if (identical(method, "DC-GBF-HET")) {
    "DC_GBF_HET"
  } else {
    method
  }
  if (is.null(spec[[method_key]])) {
    stop(paste("Unsupported method:", method))
  }
  sampler <- get(spec[[method_key]], mode = "function")

  # Every sampler takes n_inference/samples/mu plus a form-specific
  # covariance argument (sigma for univariate, sigma_matrix otherwise)
  shared_args <- c(
    list(n_inference = n_inference, samples = data, mu = mu_hat),
    setNames(list(cov_hat), spec$cov_arg)
  )

  if (method == "MCF") {
    # MCF: rejection sampling, yields rho/Q acceptance diagnostics
    result <- do.call(sampler, c(
      shared_args, list(coalescence_time = coalescence_time)
    ))
    diagnostics <- .mcf_diagnostics(result)
  } else {
    # BF/GBF: SMC over the time grid, yields ESS/CESS diagnostics
    time_args <- list(time_grid = time_grid, seed = seed)
    if (method %in% c("GBF", "DC-GBF", "DC-GBF-HET")) {
      time_args$precondition <- if (
        covariance_structure == "diagonal" &&
          spec$gbf_precondition == "covariance"
      ) {
        cov_hat
      } else {
        spec$gbf_precondition
      }
    }
    result <- do.call(sampler, c(shared_args, time_args))
    diagnostics <- .particle_diagnostics(result)
  }

  list(result = result, diagnostics = diagnostics)
}

#' Run a single simulation
#'
#' @param config named list of simulation parameters
#' @param plots_directory path to the directory where plots are saved
#' @return list of named lists of results (one per `config$h_values` entry),
#'   each including evaluation metrics and timing
#' @export
simulate_config <- function(config, plots_directory = "plots") {

  # Start simulation timer
  simulation_start_time <- proc.time()
  hardware_context <- .hardware_context_values()

  # Fall back to a per-config schema for calls that bypass build_configs
  dgp_schema <- if (!is.null(config$dgp_schema)) {
    config$dgp_schema
  } else {
    .compute_dgp_schema(list(config))
  }

  # Extract parameters from config
  parametric_form <- config$parametric_form
  dgp_parameters <- config$DGP_parameters
  num_nodes <- config$num_nodes
  n <- config$n
  method <- config$method
  n_inference <- config$n_inference
  coalescence_time_cfg <- config$coalescence_time
  n_time_steps_cfg <- config$n_time_steps
  covariance_structure <- if (is.null(config$covariance_structure)) {
    "full"
  } else {
    config$covariance_structure
  }
  if (
    length(covariance_structure) != 1L ||
      !(covariance_structure %in% c("full", "diagonal"))
  ) {
    stop("covariance_structure must be 'full' or 'diagonal'.")
  }
  seed <- config$seed
  seed_strategy <- if (!is.null(config$seed_strategy)) {
    config$seed_strategy
  } else {
    "independent"
  }
  data_seed <- config$data_seed
  inference_seed <- config$inference_seed
  if (
    seed_strategy == "paired_methods" &&
      (is.null(data_seed) || is.null(inference_seed))
  ) {
    stop("paired_methods requires data_seed and inference_seed.")
  }

  # Set seed for reproducibility
  if (seed_strategy == "paired_methods" && !is.null(data_seed)) {
    set.seed(data_seed)
  } else if (!is.null(seed)) {
    set.seed(seed)
  }

  # Define the DGP
  dgp <- .build_dgp(parametric_form, dgp_parameters)

  # Draw the realised sample size (no sample data generated yet)
  n_sample <- rpois(1, lambda = n)

  # Plot the true product distribution (Beta is unsupported by visualization.R)
  if (!is.null(plots_directory) && parametric_form != "Beta") {
    product_parameters <- if (parametric_form == "univariate_Gaussian") {
      list(mu = dgp$mu_product, sigma = dgp$sigma_product)
    } else {
      list(mu = dgp$mu_product, cov = dgp$cov_product)
    }
    plot_product_distribution(
      parametric_form,
      parameters = product_parameters,
      show = FALSE,
      save = TRUE,
      filename = .plot_filename(
        "product_distribution", parametric_form, method, n, n_sample,
        dgp_parameters, dgp, config$iteration_id
      ),
      plots_directory = plots_directory
    )
  }

  # Generate data from the DGP
  data <- sample_dgp(dgp, n_sample)

  # Resolve coalescence time, number of time steps, and time grid
  time_grid_spec <- .resolve_time_grid(
    method, data, num_nodes, n_sample, coalescence_time_cfg, n_time_steps_cfg,
    parametric_form = parametric_form,
    covariance_structure = covariance_structure
  )
  coalescence_time <- time_grid_spec$coalescence_time
  n_time_steps <- time_grid_spec$n_time_steps
  time_grid <- time_grid_spec$time_grid

  # Start training timer
  training_start_time <- proc.time()

  # Estimate sub-posterior parameters from data
  subposterior_params <- .estimate_subposterior_params(
    parametric_form, data, covariance_structure
  )
  mu_hat <- subposterior_params$mu_hat
  cov_hat <- if (is.null(subposterior_params$sigma_hat)) {
    subposterior_params$sigma_matrix_hat
  } else {
    subposterior_params$sigma_hat
  }

  # Reset the paired-method stream immediately before fusion inference
  if (seed_strategy == "paired_methods" && !is.null(inference_seed)) {
    set.seed(inference_seed)
  }

  # Run the sampling algorithm and compute its diagnostics
  sampler_seed <- if (seed_strategy == "paired_methods") {
    inference_seed
  } else {
    seed
  }
  fusion <- .run_fusion(
    parametric_form, method, n_inference, data, mu_hat, cov_hat,
    coalescence_time, time_grid, sampler_seed,
    dgp_parameters = dgp_parameters,
    covariance_structure = covariance_structure
  )
  result <- fusion$result
  diagnostics <- fusion$diagnostics

  # Stop training timer
  training_end_time <- proc.time()
  training_time_seconds <- training_end_time[["elapsed"]] -
    training_start_time[["elapsed"]]
  training_cpu_seconds <- .process_time_seconds(training_end_time) -
    .process_time_seconds(training_start_time)

  # Plot inference distribution (Beta is unsupported by visualization.R)
  if (!is.null(plots_directory) && parametric_form != "Beta") {
    plot_inference_distribution(
      samples = result$samples,
      dgp = dgp,
      show = FALSE,
      save = TRUE,
      filename = .plot_filename(
        "inference_distribution", parametric_form, method, n, n_sample,
        dgp_parameters, dgp, config$iteration_id
      ),
      plots_directory = plots_directory
    )
  }

  # Evaluate
  wasserstein_distance <- wasserstein(dgp, result$samples)
  kl_divergence <- kl(dgp, result$samples)

  # Record total simulation time
  simulation_end_time <- proc.time()
  simulation_time_seconds <- simulation_end_time[["elapsed"]] -
    simulation_start_time[["elapsed"]]
  simulation_cpu_seconds <- .process_time_seconds(simulation_end_time) -
    .process_time_seconds(simulation_start_time)

  # Fusion samples are an n x p matrix, or a plain vector when p = 1
  fusion_samples <- as.matrix(result$samples)

  # Build one result row per h transform, reusing the single fusion fit above
  rows <- lapply(config$h_values, function(h_name) {
    h_fn <- resolve_h_function(h_name)
    c_hat <- if (is.null(h_fn)) {
      colMeans(fusion_samples)
    } else {
      colMeans(h_fn(fusion_samples))
    }
    true_mean <- mean_product_dgp(dgp, h = h_fn, h_name = h_name)
    mse <- sum((c_hat - true_mean)^2)

    .build_result_row(
      config, dgp_schema, parametric_form, num_nodes, n, method, n_inference,
      dgp_parameters, dgp, coalescence_time, n_time_steps, n_sample,
      training_time_seconds, training_cpu_seconds,
      simulation_time_seconds, simulation_cpu_seconds, hardware_context,
      diagnostics,
      kl_divergence, wasserstein_distance, h_name, c_hat, mse, true_mean,
      beta_result = if (parametric_form == "Beta") result else NULL
    )
  })

  rows
}

################################
# Timeout and retry safety net #
################################

# Wall-clock timeout in seconds for the single fusion sampler attempt
.default_timeout_seconds <- 14400

# Run one fusion sampler attempt by default
.max_timeout_retries <- 1L

# Build the config for one retry attempt, reseeding deterministically
#' @noRd
.reseed_config_for_attempt <- function(config, attempt) {
  if (attempt <= 1) {
    return(config)
  }
  if (!is.null(config$inference_seed)) {
    # Bump only inference_seed (data_seed untouched) so paired_methods
    # pairing with other methods at the same iteration is preserved
    config$inference_seed <- config$inference_seed + attempt - 1
  } else if (!is.null(config$seed)) {
    # No separate inference_seed exists without changing the drawn data, so
    # retry with a different combined data/inference seed instead
    config$seed <- config$seed + attempt - 1
  }
  config
}

# Run one simulate_config attempt in an isolated, hard-killable subprocess
#' @noRd
.run_simulate_config_attempt <- function(config, plots_directory,
                                         src_fusion_dir, timeout_seconds) {
  callr::r(
    func = function(config, plots_directory, src_fusion_dir) {
      suppressPackageStartupMessages({
        library(DCFusion)
        library(layeredBB)
        library(MASS)
        library(transport)
      })
      devtools::load_all(src_fusion_dir, quiet = TRUE)
      simulate_config(config, plots_directory)
    },
    args = list(
      config = config, plots_directory = plots_directory,
      src_fusion_dir = src_fusion_dir
    ),
    timeout = timeout_seconds,
    supervise = TRUE
  )
}

#' Run simulate_config with a wall-clock timeout and reseeded retries
#'
#' Runs each attempt of `simulate_config` in an isolated subprocess via
#' `callr::r`, hard-killing it on timeout rather than relying on R-level
#' interrupts, which are not always reliably delivered when the hang lives
#' inside compiled DCFusion C++ code. On timeout, retries the same config
#' with a deterministic-but-different `inference_seed` so the retry stays
#' auditable, while `data_seed` is left unchanged to preserve
#' `paired_methods` pairing with other methods at the same iteration. If
#' every attempt times out, logs a message distinguishable from a genuine
#' algorithmic error and returns NULL rather than crashing the batch.
#'
#' @param config named list of simulation parameters
#' @param plots_directory path to the directory where plots are saved
#' @param timeout_seconds wall-clock timeout in seconds for each attempt
#' @param max_retries maximum number of attempts before giving up
#' @return list of named lists of results (as `simulate_config` returns),
#'   or NULL if every attempt errored or timed out
#' @export
simulate_config_with_retry <- function(
  config, plots_directory = "plots",
  timeout_seconds = .default_timeout_seconds,
  max_retries = .max_timeout_retries
) {
  src_fusion_dir <- system.file(package = "fusion")
  attempt_config <- config

  for (attempt in seq_len(max_retries)) {
    attempt_config <- .reseed_config_for_attempt(config, attempt)

    # Distinguish "sampler succeeded", "timed out" (retry), and "genuine
    # error" (stop retrying, mirror the caller's existing error logging)
    outcome <- tryCatch(
      list(status = "ok", value = .run_simulate_config_attempt(
        attempt_config, plots_directory, src_fusion_dir, timeout_seconds
      )),
      callr_timeout_error = function(e) list(status = "timeout"),
      error = function(e) {
        message(conditionMessage(e))
        stop(e)
      }
    )

    if (outcome$status == "ok") {
      return(outcome$value)
    }
  }

  final_seed_label <- if (!is.null(attempt_config$inference_seed)) {
    paste0("inference_seed tried=", attempt_config$inference_seed)
  } else {
    paste0("seed tried=", attempt_config$seed)
  }
  message(
    "Timed out after ", max_retries,
    " attempts (config seed=", config$seed, ", final ", final_seed_label, ")."
  )
  NULL
}
