##############################
# Import necessary libraries #
##############################

suppressWarnings(
  suppressPackageStartupMessages(
    library(DCFusion)
  )
)

#################################
# Mixture adapter prerequisites #
#################################

# Validate and normalize every node-specific mixture specification.
.validate_nodes <- function(nodes) {
  if (length(nodes) < 1) {
    stop("At least one node is required.")
  }

  required_names <- c("weights", "means", "standard_deviations")
  validated <- vector("list", length(nodes))
  for (node_index in seq_along(nodes)) {
    node <- nodes[[node_index]]
    if (!setequal(names(node), required_names) ||
          length(names(node)) != length(required_names)) {
      stop(paste(
        "Node", node_index,
        "must contain exactly weights, means, and standard_deviations."
      ))
    }

    weights <- unlist(node$weights, use.names = FALSE)
    means <- unlist(node$means, use.names = FALSE)
    standard_deviations <- unlist(
      node$standard_deviations,
      use.names = FALSE
    )
    if (!is.numeric(weights) || !is.numeric(means) ||
          !is.numeric(standard_deviations) || length(weights) < 1 ||
          length(means) != length(weights) ||
          length(standard_deviations) != length(weights)) {
      stop(paste(
        "Node", node_index,
        "mixture parameters must be numeric vectors of equal positive length."
      ))
    }
    if (any(!is.finite(weights)) || any(weights <= 0)) {
      stop(paste("Node", node_index, "weights must be finite and positive."))
    }
    if (abs(sum(weights) - 1) > sqrt(.Machine$double.eps)) {
      stop(paste("Node", node_index, "weights must sum to one."))
    }
    if (any(!is.finite(means))) {
      stop(paste("Node", node_index, "means must be finite."))
    }
    if (any(!is.finite(standard_deviations)) ||
          any(standard_deviations <= 0)) {
      stop(paste(
        "Node", node_index,
        "standard deviations must be finite and positive."
      ))
    }

    validated[[node_index]] <- list(
      weights = weights,
      means = means,
      standard_deviations = standard_deviations
    )
  }

  validated
}

# Validate one resolved numeric GBF time mesh.
.validate_time_mesh <- function(time_mesh) {
  if (!is.numeric(time_mesh) || !is.null(dim(time_mesh)) ||
        length(time_mesh) < 2 ||
        any(!is.finite(time_mesh)) || time_mesh[1] != 0 ||
        any(diff(time_mesh) <= 0)) {
    stop("time_mesh must be finite, strictly increasing, and start at zero.")
  }

  invisible(TRUE)
}

# Validate a positive integer-valued scalar.
.validate_positive_integer <- function(value, name) {
  if (length(value) != 1 || !is.numeric(value) || !is.finite(value) ||
        value != floor(value) || value <= 0 ||
        value >= .Machine$integer.max) {
    stop(paste(name, "must be a positive integer below the R integer maximum."))
  }

  invisible(TRUE)
}

# Validate a scalar ESS resampling threshold.
.validate_ess_threshold <- function(ess_threshold) {
  if (length(ess_threshold) != 1 || !is.numeric(ess_threshold) ||
        !is.finite(ess_threshold) || ess_threshold < 0 ||
        ess_threshold > 1) {
    stop("ess_threshold must be a finite scalar between zero and one.")
  }

  invisible(TRUE)
}

# Validate a scalar initial CESS guidance threshold.
.validate_cess_0_threshold <- function(cess_0_threshold) {
  if (length(cess_0_threshold) != 1 || !is.numeric(cess_0_threshold) ||
        !is.finite(cess_0_threshold) || cess_0_threshold < 0 ||
        cess_0_threshold > 1) {
    stop("cess_0_threshold must be a finite scalar between zero and one.")
  }

  invisible(TRUE)
}

# Resolve a fixed or author-guided GBF time mesh.
.resolve_time_mesh <- function(time_mesh, samples, cess_0_threshold) {
  .validate_cess_0_threshold(cess_0_threshold)
  sample_lengths <- vapply(samples, length, integer(1))
  if (length(unique(sample_lengths)) != 1) {
    stop("Every node sample must have the same length for GBF guidance.")
  }
  num_nodes <- length(samples)
  sample_count <- sample_lengths[1]
  if (is.character(time_mesh) && length(time_mesh) == 1) {
    if (time_mesh != "optimal") {
      stop("time_mesh must be optimal or a numeric mesh.")
    }
    if (!exists("optimal_time_grid", mode = "function")) {
      stop("optimal_time_grid must be sourced for author-guided GBF meshes.")
    }
    guided <- optimal_time_grid(
      method = "GBF",
      samples = samples,
      k = num_nodes,
      n = sample_count,
      cess_0_threshold = cess_0_threshold
    )
    resolved <- guided$time_grid
    .validate_time_mesh(resolved)
    list(
      time_mesh = resolved,
      requested_time_mesh = "optimal",
      guidance_role = "author_guided",
      guidance_source = "original_repo_dcfusion_jmlr_guidance",
      cess_0_threshold = cess_0_threshold,
      interval_count = guided$n_time_steps,
      coalescence_time = guided$coalescence_time,
      end_time = tail(resolved, 1)
    )
  } else {
    .validate_time_mesh(time_mesh)
    list(
      time_mesh = time_mesh,
      requested_time_mesh = as.numeric(time_mesh),
      guidance_role = "fixed_config",
      guidance_source = "numeric_config_mesh",
      cess_0_threshold = cess_0_threshold,
      interval_count = as.integer(length(time_mesh) - 1L),
      coalescence_time = tail(time_mesh, 1),
      end_time = tail(time_mesh, 1)
    )
  }
}

# Validate and normalize persisted component sample vectors.
.validate_samples <- function(samples) {
  if (!is.list(samples) || length(samples) < 1) {
    stop("samples must be a nonempty list of numeric vectors.")
  }
  sample_lengths <- vapply(samples, length, integer(1))
  if (any(sample_lengths < 2) || length(unique(sample_lengths)) != 1) {
    stop("Every node sample must have the same length of at least two.")
  }
  numeric_samples <- vapply(samples, is.numeric, logical(1))
  finite_samples <- vapply(
    samples,
    function(values) all(is.finite(values)),
    logical(1)
  )
  if (any(!numeric_samples) || any(!finite_samples)) {
    stop("Every node sample must be numeric and finite.")
  }

  lapply(samples, as.vector)
}

# Compute node-specific variance preconditioners from persisted samples.
.variance_preconditioners <- function(samples) {
  values <- vapply(samples, stats::var, numeric(1))
  if (any(!is.finite(values)) || any(values <= 0)) {
    stop("Every node sample must have a positive finite variance.")
  }

  values
}

# Verify that every terminal particle has coalesced across node coordinates.
.validate_terminal_coalescence <- function(particles, num_nodes) {
  if (!is.list(particles) || length(particles) < 1) {
    stop("Terminal particles must be a nonempty list.")
  }
  valid <- vapply(
    particles,
    function(values) {
      is.numeric(values) && length(values) == num_nodes &&
        all(is.finite(values))
    },
    logical(1)
  )
  if (any(!valid)) {
    stop("Each terminal particle must contain one finite coordinate per node.")
  }

  particle_values <- unlist(particles, use.names = FALSE)
  scale <- max(1, abs(particle_values))
  tolerance <- sqrt(.Machine$double.eps) * scale
  spreads <- vapply(
    particles,
    function(values) diff(range(values)),
    numeric(1)
  )
  max_spread <- max(spreads)
  if (max_spread > tolerance) {
    stop(paste(
      "Terminal node coordinates did not coalesce within tolerance:",
      max_spread, ">", tolerance
    ))
  }

  list(max_spread = max_spread, tolerance = tolerance)
}

# Derive a deterministic seed for one uniquely indexed GBF stage.
.derive_stage_seed <- function(seed, stage_index) {
  if (is.null(seed)) {
    return(NULL)
  }
  if (length(seed) != 1 || !is.numeric(seed) || !is.finite(seed) ||
        seed != floor(seed) || seed < 0 ||
        seed >= .Machine$integer.max) {
    stop("seed must be an integer between zero and the R integer maximum.")
  }
  if (length(stage_index) != 1 || !is.numeric(stage_index) ||
        !is.finite(stage_index) || stage_index != floor(stage_index) ||
        stage_index < 1 || stage_index >= .Machine$integer.max) {
    stop("stage_index must be a positive integer below the R integer maximum.")
  }

  as.integer(
    (as.double(seed) + 104729 * as.double(stage_index)) %%
      .Machine$integer.max
  )
}

# Audit mixture EA numerics heuristically over conservative finite envelopes.
.ea_numerical_preflight <- function(
  node_parameters,
  precondition_values,
  grid_size = 200001,
  envelope_sds = 8,
  bounds_multiplier = 1.1
) {
  reports <- vector("list", length(node_parameters))
  for (node_index in seq_along(node_parameters)) {
    node <- node_parameters[[node_index]]
    precondition <- precondition_values[node_index]
    lower <- min(node$means - envelope_sds * node$standard_deviations)
    upper <- max(node$means + envelope_sds * node$standard_deviations)
    grid <- seq(lower, upper, length.out = grid_size)
    phi <- tryCatch(
      DCFusion::ea_phi_mixG_DL(
        x = grid,
        n_comp = length(node$weights),
        weights = node$weights,
        means = node$means,
        sds = node$standard_deviations,
        beta = 1,
        precondition = precondition
      ),
      error = function(condition) {
        stop(paste(
          "Node", node_index,
          "EA numerical preflight phi evaluation failed:",
          conditionMessage(condition)
        ))
      }
    )
    if (any(!is.finite(phi))) {
      stop(paste(
        "Node", node_index,
        "EA numerical preflight found nonfinite phi values;",
        "the mixture lies outside the supported numerical envelope."
      ))
    }

    global_lb <- tryCatch(
      suppressWarnings(DCFusion::ea_phi_mixG_DL_LB(
        n_comp = length(node$weights),
        weights = node$weights,
        means = node$means,
        sds = node$standard_deviations,
        beta = 1,
        precondition = precondition,
        bounds_multiplier = bounds_multiplier
      )),
      error = function(condition) NA_real_
    )
    bounds <- tryCatch(
      suppressWarnings(DCFusion::ea_phi_mixG_DL_bounds(
        n_comp = length(node$weights),
        weights = node$weights,
        means = node$means,
        sds = node$standard_deviations,
        beta = 1,
        precondition = precondition,
        PHI = global_lb,
        lower = lower,
        upper = upper,
        bounds_multiplier = bounds_multiplier,
        seq_length = grid_size
      )),
      error = function(condition) list(LB = NA_real_, UB = NA_real_)
    )
    bound_values <- c(global_lb, bounds$LB, bounds$UB)
    if (length(global_lb) != 1 || length(bounds$LB) != 1 ||
          length(bounds$UB) != 1 || any(!is.finite(bound_values))) {
      stop(paste(
        "Node", node_index,
        "EA numerical preflight found nonfinite DCFusion bounds;",
        "the mixture lies outside the supported numerical envelope."
      ))
    }

    min_phi <- min(phi)
    max_phi <- max(phi)
    margins <- c(
      global_lower = min_phi - global_lb,
      interval_lower = min_phi - bounds$LB,
      interval_upper = bounds$UB - max_phi
    )
    tolerance <- sqrt(.Machine$double.eps) * max(
      1,
      abs(c(min_phi, max_phi, bound_values))
    )
    if (any(margins < -tolerance)) {
      stop(paste(
        "Node", node_index,
        "EA numerical preflight bounds do not contain the dense-grid phi",
        "range within floating-point tolerance."
      ))
    }

    reports[[node_index]] <- list(
      node_index = node_index,
      audit_label = "heuristic_ea_numerical_preflight",
      certificate_scope = "not_a_global_certificate",
      interval_lower = lower,
      interval_upper = upper,
      grid_size = grid_size,
      envelope_component_standard_deviations = envelope_sds,
      bounds_multiplier = bounds_multiplier,
      grid_min_phi = min_phi,
      grid_max_phi = max_phi,
      global_lower_bound = global_lb,
      interval_lower_bound = bounds$LB,
      interval_upper_bound = bounds$UB,
      global_lower_margin = unname(margins["global_lower"]),
      interval_lower_margin = unname(margins["interval_lower"]),
      interval_upper_margin = unname(margins["interval_upper"]),
      containment_tolerance = tolerance,
      passed = TRUE
    )
  }

  reports
}

#############################
# Mixture-specific GBF step #
#############################

# Apply the mixture exact-algorithm correction across the resolved time mesh.
rho_j_mixture <- function(
  particle_set,
  num_nodes,
  time_mesh,
  node_parameters,
  betas,
  precondition_values,
  ess_threshold = 0.5,
  seed = NULL,
  n_cores = 1
) {
  if (!inherits(particle_set, "particle")) {
    stop("particle_set must be a DCFusion particle object.")
  }
  .validate_positive_integer(num_nodes, "num_nodes")
  .validate_positive_integer(n_cores, "n_cores")
  .validate_ess_threshold(ess_threshold)
  .derive_stage_seed(seed, 1)
  .validate_time_mesh(time_mesh)
  if (length(betas) != num_nodes ||
        length(precondition_values) != num_nodes ||
        length(node_parameters) != num_nodes) {
    stop(paste(
      "node_parameters, betas, and precondition_values must match",
      "num_nodes."
    ))
  }
  node_parameters <- .validate_nodes(node_parameters)
  n_inference <- particle_set$N
  .validate_positive_integer(n_inference, "n_inference")
  cluster <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cluster), add = TRUE)

  max_per_core <- ceiling(n_inference / n_cores)
  split_indices <- split(
    seq_len(n_inference),
    ceiling(seq_len(n_inference) / max_per_core)
  )
  elapsed <- rep(NA_real_, length(time_mesh) - 1)
  ess <- c(particle_set$ESS[1], rep(NA_real_, length(time_mesh) - 1))
  archive_cess <- c(
    particle_set$CESS[1],
    rep(NA_real_, length(time_mesh) - 1)
  )
  adaptive_resampled <- rep(FALSE, length(time_mesh) - 1)
  interval_indices <- seq_len(length(time_mesh) - 1)
  if (is.null(seed)) {
    proposal_seeds <- rep(NA_integer_, length(interval_indices))
    adaptive_seeds <- rep(NA_integer_, length(interval_indices))
  } else {
    proposal_seeds <- vapply(
      interval_indices,
      function(index) .derive_stage_seed(seed, 2 * index),
      integer(1)
    )
    adaptive_seeds <- vapply(
      interval_indices,
      function(index) .derive_stage_seed(seed, 2 * index + 1),
      integer(1)
    )
  }
  precondition_matrices <- lapply(precondition_values, as.matrix)
  lambda <- DCFusion::inverse_sum_matrices(
    lapply(1 / precondition_values, as.matrix)
  )
  end_time <- tail(time_mesh, 1)

  for (step in 2:length(time_mesh)) {
    start_time <- proc.time()
    interval_index <- step - 1
    proposal_seed <- if (is.null(seed)) NULL else proposal_seeds[interval_index]
    adaptive_seed <- if (is.null(seed)) NULL else adaptive_seeds[interval_index]
    if (!is.null(proposal_seed)) {
      parallel::clusterSetRNGStream(cluster, iseed = proposal_seed)
    }
    if (particle_set$ESS < n_inference * ess_threshold) {
      adaptive_resampled[step - 1] <- TRUE
      particle_set <- DCFusion::resample_particle_x_samples(
        N = n_inference,
        particle_set = particle_set,
        multivariate = FALSE,
        step = step - 1,
        resampling_method = "multi",
        seed = adaptive_seed
      )
    }

    split_samples <- lapply(
      split_indices,
      function(indices) particle_set$x_samples[indices]
    )
    split_means <- lapply(
      split_indices,
      function(indices) particle_set$x_means[indices]
    )
    proposal_cov <- DCFusion::construct_V(
      s = time_mesh[step - 1],
      t = time_mesh[step],
      end_time = end_time,
      C = num_nodes,
      d = 1,
      precondition_matrices = precondition_matrices,
      Lambda = lambda,
      iteration = step
    )

    weighted <- parallel::parLapply(
      cluster,
      seq_along(split_indices),
      function(core_index) {
        split_count <- length(split_indices[[core_index]])
        proposed <- vector("list", split_count)
        proposed_means <- numeric(split_count)
        log_weights <- numeric(split_count)

        for (particle_index in seq_len(split_count)) {
          proposal_mean <- DCFusion::construct_M(
            s = time_mesh[step - 1],
            t = time_mesh[step],
            end_time = end_time,
            C = num_nodes,
            d = 1,
            sub_posterior_samples =
              split_samples[[core_index]][[particle_index]],
            sub_posterior_mean =
              split_means[[core_index]][particle_index],
            iteration = step
          )
          proposed[[particle_index]] <- as.vector(
            mvtnorm::rmvnorm(1, mean = proposal_mean, sigma = proposal_cov)
          )
          proposed_means[particle_index] <-
            DCFusion::weighted_mean_univariate(
              x = proposed[[particle_index]],
              weights = 1 / precondition_values
            )
          log_weights[particle_index] <- sum(vapply(
            seq_len(num_nodes),
            function(node_index) {
              node <- node_parameters[[node_index]]
              DCFusion::ea_mixG_DL_PT(
                x0 = split_samples[[core_index]][[particle_index]][node_index],
                y = proposed[[particle_index]][node_index],
                s = time_mesh[step - 1],
                t = time_mesh[step],
                n_comp = length(node$weights),
                weights = node$weights,
                means = node$means,
                sds = node$standard_deviations,
                beta = betas[node_index],
                precondition = precondition_values[node_index],
                logarithm = TRUE
              )
            },
            numeric(1)
          ))
        }

        list(
          proposed = proposed,
          proposed_means = proposed_means,
          log_weights = log_weights
        )
      }
    )

    particle_set$x_samples <- unlist(
      lapply(weighted, function(item) item$proposed),
      recursive = FALSE
    )
    particle_set$x_means <- unlist(
      lapply(weighted, function(item) item$proposed_means)
    )
    log_weights <- unlist(lapply(weighted, function(item) item$log_weights))
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
    archive_cess[step] <- particle_set$CESS[step]
    elapsed[step - 1] <- (proc.time() - start_time)["elapsed"]
  }

  coalescence <- .validate_terminal_coalescence(
    particle_set$x_samples,
    num_nodes
  )
  weighted_particles <- vapply(
    particle_set$x_samples,
    function(values) values[1],
    numeric(1)
  )
  normalized_weights <- particle_set$normalised_weights
  if (length(normalized_weights) != n_inference ||
        any(!is.finite(normalized_weights)) ||
        any(normalized_weights < 0) ||
        abs(sum(normalized_weights) - 1) > sqrt(.Machine$double.eps)) {
    stop("Terminal particle weights must be finite and sum to one.")
  }
  particle_set$y_samples <- weighted_particles
  terminal_seed <- .derive_stage_seed(seed, 2 * length(time_mesh))
  particle_set <- DCFusion::resample_particle_y_samples(
    N = n_inference,
    particle_set = particle_set,
    multivariate = FALSE,
    resampling_method = "multi",
    seed = terminal_seed
  )
  equal_weight_samples <- particle_set$y_samples

  list(
    particle_set = particle_set,
    weighted_particles = weighted_particles,
    normalized_weights = normalized_weights,
    equal_weight_samples = equal_weight_samples,
    elapsed = elapsed,
    ess = ess,
    archive_cess = archive_cess,
    adaptive_resampled = adaptive_resampled,
    proposal_seeds = proposal_seeds,
    adaptive_resampling_seeds = adaptive_seeds,
    terminal_resampling_seed = terminal_seed,
    max_terminal_spread = coalescence$max_spread,
    terminal_coalescence_tolerance = coalescence$tolerance
  )
}

# Run GBF using node-specific mixtures and persisted component samples.
gbf_univariate_mixture <- function(
  samples,
  nodes,
  n_inference,
  time_mesh,
  ess_threshold = 0.5,
  cess_0_threshold = 0.2,
  seed = NULL,
  n_cores = 1
) {
  .validate_positive_integer(n_inference, "n_inference")
  .validate_positive_integer(n_cores, "n_cores")
  .validate_ess_threshold(ess_threshold)
  .derive_stage_seed(seed, 1)
  num_nodes <- length(samples)
  if (length(nodes) != num_nodes) {
    stop("The sample and node counts must agree.")
  }

  node_parameters <- .validate_nodes(nodes)
  samples <- .validate_samples(samples)
  mesh_info <- .resolve_time_mesh(
    time_mesh,
    samples,
    cess_0_threshold
  )
  time_mesh <- mesh_info$time_mesh
  precondition_values <- .variance_preconditioners(samples)
  ea_preflight <- .ea_numerical_preflight(
    node_parameters,
    precondition_values
  )
  particles <- DCFusion::initialise_particle_sets(
    samples_to_fuse = samples,
    multivariate = FALSE,
    number_of_steps = length(time_mesh)
  )
  initial_seed <- .derive_stage_seed(seed, 1)
  if (!is.null(initial_seed)) {
    set.seed(initial_seed)
  }

  start_time <- proc.time()
  initial <- DCFusion::rho_IS_univariate(
    particles_to_fuse = particles,
    N = n_inference,
    m = num_nodes,
    time = tail(time_mesh, 1),
    precondition_values = precondition_values,
    number_of_steps = length(time_mesh),
    time_mesh = time_mesh,
    resampling_method = "multi",
    seed = initial_seed,
    n_cores = n_cores
  )
  corrected <- rho_j_mixture(
    particle_set = initial,
    num_nodes = num_nodes,
    time_mesh = time_mesh,
    node_parameters = node_parameters,
    betas = rep(1, num_nodes),
    precondition_values = precondition_values,
    ess_threshold = ess_threshold,
    seed = seed,
    n_cores = n_cores
  )

  list(
    equal_weight_samples = corrected$equal_weight_samples,
    weighted_particles = corrected$weighted_particles,
    normalized_weights = corrected$normalized_weights,
    ess = corrected$ess,
    archive_cess = corrected$archive_cess,
    adaptive_resampled = corrected$adaptive_resampled,
    requested_time_mesh = mesh_info$requested_time_mesh,
    resolved_time_mesh = time_mesh,
    coalescence_time = mesh_info$coalescence_time,
    end_time = mesh_info$end_time,
    guidance_role = mesh_info$guidance_role,
    guidance_source = mesh_info$guidance_source,
    cess_0_threshold = mesh_info$cess_0_threshold,
    interval_count = mesh_info$interval_count,
    ess_threshold = ess_threshold,
    resampling_method = "multi",
    bounds_multiplier = 1.1,
    dcfusion_guidance_reference = paste(
      "DCFusion BF_guidance under the JMLR SSH condition"
    ),
    initial_seed = initial_seed,
    proposal_seeds = corrected$proposal_seeds,
    adaptive_resampling_seeds = corrected$adaptive_resampling_seeds,
    terminal_resampled = TRUE,
    terminal_resampling_seed = corrected$terminal_resampling_seed,
    max_terminal_spread = corrected$max_terminal_spread,
    terminal_coalescence_tolerance =
      corrected$terminal_coalescence_tolerance,
    ea_numerical_preflight = ea_preflight,
    interval_seconds = corrected$elapsed,
    runtime_seconds = unname((proc.time() - start_time)["elapsed"]),
    precondition_values = precondition_values
  )
}
