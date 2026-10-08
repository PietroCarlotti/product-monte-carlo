##############################
# Import necessary libraries #
##############################

suppressWarnings(suppressPackageStartupMessages({
  library(DCFusion)
  library(HMCBLR)
  library(jsonlite)
}))

###############################
# HIGGS validation and fusion #
###############################

# Validate the HIGGS configuration and GBF schedule.
validate_higgs_config <- function(config, n_batches) {
  required <- c(
    "dataset", "raw_data", "output_dir", "prior", "seeds", "gbf",
    "run_status"
  )
  if (!all(required %in% names(config)) ||
        config$dataset != "UCI HIGGS" ||
        !(config$run_status %in% c("production", "pilot")) ||
        length(config$prior$variance) != 1 ||
        !is.finite(config$prior$variance) ||
        config$prior$variance <= 0) {
    stop("Invalid HIGGS configuration.")
  }

  particles <- unlist(config$gbf$particles_per_level, use.names = FALSE)
  merge_factors <- unlist(config$gbf$merge_factors, use.names = FALSE)
  time_mesh <- unlist(config$gbf$time_mesh, use.names = FALSE)
  precondition <- config$gbf$precondition
  if (length(particles) < 1 ||
        length(particles) != length(merge_factors) ||
        any(!is.finite(particles)) ||
        any(particles <= 0) ||
        any(particles %% 1 != 0)) {
    stop("HIGGS GBF particle and merge schedules must have equal lengths.")
  }
  if (!is.null(precondition) &&
        (length(precondition) != 1 ||
           !is.logical(precondition) ||
           is.na(precondition))) {
    stop("HIGGS GBF preconditioning must be true or false.")
  }
  if (length(time_mesh) < 2 ||
        any(!is.finite(time_mesh)) ||
        any(time_mesh <= 0) ||
        any(time_mesh > 1) ||
        any(diff(time_mesh) <= 0) ||
        tail(time_mesh, 1) != 1) {
    stop("HIGGS GBF time mesh must increase to one.")
  }

  remaining_nodes <- n_batches
  for (merge_factor in rev(merge_factors)) {
    if (!is.finite(merge_factor) ||
          merge_factor < 2 ||
          merge_factor %% 1 != 0 ||
          remaining_nodes %% merge_factor != 0) {
      stop("HIGGS GBF merge factors must reduce every level exactly.")
    }
    remaining_nodes <- remaining_nodes / merge_factor
  }
  if (remaining_nodes != 1) {
    stop("HIGGS GBF schedule must fuse all batches into one posterior.")
  }

  invisible(TRUE)
}

# Derive the canonical HIGGS coefficient ordering from the processed input.
.higgs_feature_names <- function(raw) {
  native_features <- grep("^feature_[0-9]+$", names(raw), value = TRUE)
  feature_indices <- as.integer(sub("^feature_", "", native_features))
  expected_indices <- seq_along(native_features)
  if (!("intercept" %in% names(raw)) ||
        length(native_features) < 1 ||
        anyDuplicated(feature_indices) ||
        any(sort(feature_indices) != expected_indices) ||
        !setequal(native_features, paste0("feature_", expected_indices))) {
    stop("Processed HIGGS input has invalid feature columns.")
  }
  c("intercept", paste0("feature_", expected_indices))
}

# Convert one HIGGS partition into DCFusion's BLR input format.
.build_higgs_input <- function(data, feature_names) {
  design <- as.matrix(data[feature_names])
  counts <- HMCBLR::unique_row_count(y = data$outcome, X = design)
  list(
    y = as.vector(data$outcome),
    X = design,
    full_data_count = counts$full_data_count,
    design_count = counts$design_count
  )
}

# Fuse all HIGGS node posteriors through the DCFusion BLR GBF implementation.
.run_higgs_gbf <- function(node_draws, data_split, config) {
  gbf <- config$gbf
  dimension <- ncol(data_split[[1]]$X)
  started <- proc.time()[["elapsed"]]
  output <- DCFusion::bal_binary_GBF_BLR(
    N_schedule = unlist(gbf$particles_per_level),
    m_schedule = unlist(gbf$merge_factors),
    time_mesh = unlist(gbf$time_mesh),
    base_samples = node_draws,
    L = length(gbf$particles_per_level) + 1,
    dim = dimension,
    data_split = data_split,
    prior_means = rep(0, dimension),
    prior_variances = rep(config$prior$variance, dimension),
    C = length(data_split),
    precondition = isTRUE(gbf$precondition),
    resampling_method = gbf$resampling_method,
    ESS_threshold = gbf$ess_threshold,
    diffusion_estimator = gbf$diffusion_estimator,
    seed = config$seeds$gbf,
    n_cores = gbf$n_cores,
    print_progress_iters = gbf$print_progress_iters
  )
  particle_set <- output$particles[[1]]
  if (!inherits(particle_set, "particle") ||
        is.null(particle_set$y_samples) ||
        is.null(particle_set$normalised_weights)) {
    stop("GBF did not return a terminal particle set.")
  }
  draws <- as.matrix(particle_set$y_samples)
  weights <- as.numeric(particle_set$normalised_weights)
  if (ncol(draws) != dimension ||
        any(!is.finite(draws)) ||
        any(!is.finite(weights)) ||
        sum(weights) <= 0) {
    stop("GBF returned invalid terminal particles.")
  }
  weights <- weights / sum(weights)
  resampled <- DCFusion::resample_particle_y_samples(
    N = nrow(draws),
    particle_set = particle_set,
    multivariate = TRUE,
    resampling_method = gbf$resampling_method,
    seed = config$seeds$gbf + 1
  )
  list(
    draws = as.matrix(resampled$y_samples),
    runtime_seconds = proc.time()[["elapsed"]] - started,
    terminal_ess = 1 / sum(weights^2),
    terminal_weight_min = min(weights),
    terminal_weight_max = max(weights),
    ess_min = min(unlist(output$ESS, use.names = FALSE)),
    cess_min = min(unlist(output$CESS, use.names = FALSE))
  )
}

# Load all HIGGS partitions and fuse their saved posterior draws.
run_higgs_gbf <- function(config_path, node_dir) {
  config <- fromJSON(config_path, simplifyVector = FALSE)
  expected_batches <- seq_len(10)
  validate_higgs_config(config, length(expected_batches))
  config_dir <- dirname(normalizePath(config_path))
  raw_path <- normalizePath(
    file.path(config_dir, config$raw_data),
    mustWork = TRUE
  )
  output_dir <- file.path(config_dir, config$output_dir)
  raw <- read.csv(raw_path, check.names = FALSE)
  feature_names <- .higgs_feature_names(raw)
  required_columns <- c(feature_names, "outcome", "batch")
  finite_features <- vapply(
    raw[feature_names],
    function(column) all(is.finite(column)),
    logical(1)
  )
  if (!all(required_columns %in% names(raw)) ||
        !all(finite_features) ||
        any(!raw$outcome %in% c(0, 1))) {
    stop("Processed HIGGS input is missing valid required columns.")
  }
  batches <- sort(unique(raw$batch))
  if (length(batches) != length(expected_batches) ||
        any(batches != expected_batches)) {
    stop("Processed HIGGS input must contain exactly batches one through ten.")
  }

  expected_files <- paste0("batch_", expected_batches, "_draws.csv")
  discovered_files <- list.files(
    node_dir,
    pattern = "^batch_[0-9]+_draws\\.csv$"
  )
  if (length(discovered_files) != length(expected_files) ||
        !setequal(discovered_files, expected_files)) {
    stop("Expected posterior draws for exactly batches one through ten.")
  }
  node_draws <- lapply(expected_files, function(file_name) {
    path <- file.path(node_dir, file_name)
    data <- read.csv(path, check.names = FALSE)
    draws <- as.matrix(data)
    if (nrow(draws) < 1 ||
          ncol(draws) != length(feature_names) ||
          any(names(data) != feature_names) ||
          any(!is.finite(draws))) {
      stop("HIGGS posterior draws have invalid dimensions or ordering.")
    }
    draws
  })
  draw_counts <- vapply(node_draws, nrow, integer(1))
  if (any(draw_counts != draw_counts[[1]])) {
    stop("HIGGS posterior batches must contain equal numbers of draws.")
  }

  data_split <- lapply(expected_batches, function(batch) {
    .build_higgs_input(
      raw[raw$batch == batch, , drop = FALSE],
      feature_names
    )
  })
  gbf <- .run_higgs_gbf(node_draws, data_split, config)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  colnames(gbf$draws) <- feature_names
  write.csv(
    gbf$draws,
    file.path(output_dir, "gbf_samples.csv"),
    row.names = FALSE
  )
  write_json(
    list(runtime_seconds = gbf$runtime_seconds),
    file.path(output_dir, "runtime.json"),
    pretty = TRUE,
    auto_unbox = TRUE
  )
  write_json(
    list(
      terminal_ess = gbf$terminal_ess,
      terminal_weight_min = gbf$terminal_weight_min,
      terminal_weight_max = gbf$terminal_weight_max,
      ess_min = gbf$ess_min,
      cess_min = gbf$cess_min
    ),
    file.path(output_dir, "diagnostics.json"),
    pretty = TRUE,
    auto_unbox = TRUE
  )
  invisible(TRUE)
}
