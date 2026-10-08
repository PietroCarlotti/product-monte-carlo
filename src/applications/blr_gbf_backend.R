##########################
# Shared BLR GBF backend #
##########################

# Convert binary-logistic observations into DCFusion's BLR input format.
.build_data_input <- function(response, design) {
  counts <- HMCBLR::unique_row_count(y = response, X = design)
  list(y = as.vector(response), X = as.matrix(design),
       full_data_count = counts$full_data_count,
       design_count = counts$design_count)
}

# Fuse BLR node posteriors through the shared DCFusion GBF implementation.
.run_blr_gbf <- function(node_draws, data_split, config) {
  gbf <- config$gbf
  started <- proc.time()[["elapsed"]]
  output <- DCFusion::bal_binary_GBF_BLR(
    N_schedule = unlist(gbf$particles_per_level),
    m_schedule = unlist(gbf$merge_factors),
    time_mesh = unlist(gbf$time_mesh), base_samples = node_draws,
    L = length(gbf$particles_per_level) + 1,
    dim = ncol(data_split[[1]]$X), data_split = data_split,
    prior_means = rep(0, ncol(data_split[[1]]$X)),
    prior_variances = rep(config$prior$variance, ncol(data_split[[1]]$X)),
    C = length(data_split), precondition = TRUE,
    resampling_method = gbf$resampling_method,
    ESS_threshold = gbf$ess_threshold,
    diffusion_estimator = gbf$diffusion_estimator,
    seed = config$seeds$gbf, n_cores = gbf$n_cores,
    print_progress_iters = gbf$print_progress_iters)
  particle_set <- output$particles[[1]]
  if (!inherits(particle_set, "particle") || is.null(particle_set$y_samples) ||
      is.null(particle_set$normalised_weights)) {
    stop("GBF did not return a terminal particle set.")
  }
  draws <- as.matrix(particle_set$y_samples)
  weights <- as.numeric(particle_set$normalised_weights)
  weights <- weights / sum(weights)
  resampled <- DCFusion::resample_particle_y_samples(
    N = nrow(draws), particle_set = particle_set, multivariate = TRUE,
    resampling_method = gbf$resampling_method, seed = config$seeds$gbf + 1)
  list(draws = as.matrix(resampled$y_samples),
       runtime_seconds = proc.time()[["elapsed"]] - started,
       ess_min = min(unlist(output$ESS, use.names = FALSE)),
       cess_min = min(unlist(output$CESS, use.names = FALSE)),
       terminal_ess = 1 / sum(weights^2),
       terminal_weight_min = min(weights),
       terminal_weight_max = max(weights))
}
