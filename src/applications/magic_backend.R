##############################
# Import necessary libraries #
##############################

suppressWarnings(suppressPackageStartupMessages({
  library(DCFusion)
  library(HMCBLR)
  library(jsonlite)
}))

# Validate the MAGIC smoke configuration.
validate_magic_config <- function(config) {
  required <- c("raw_data", "output_dir", "dataset", "prior", "seeds", "gbf")
  if (!all(required %in% names(config)) || config$dataset != "UCI MAGIC Gamma Telescope" ||
      config$prior$variance != 4.0) {
    stop("Invalid MAGIC smoke configuration.")
  }
  invisible(TRUE)
}

# Run the MAGIC posterior and record backend artifacts.
run_magic_backend <- function(config_path, sample_paths) {
  config <- fromJSON(config_path, simplifyVector = FALSE)
  validate_magic_config(config)
  config_dir <- dirname(normalizePath(config_path))
  config$raw_data <- normalizePath(file.path(config_dir, config$raw_data), mustWork = TRUE)
  config$output_dir <- file.path(config_dir, config$output_dir)
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
  raw <- read.csv(config$raw_data, check.names = FALSE)
  feature_names <- grep("^feature_[0-9]+$", names(raw), value = TRUE)
  if (length(feature_names) < 1) stop("MAGIC input has no feature columns.")
  design <- as.matrix(raw[feature_names])
  centers <- sort(unique(raw$batch))
  if (length(centers) < 1) stop("The MAGIC input has no batches.")
  merge_factors <- unlist(config$gbf$merge_factors)
  particles_per_level <- unlist(config$gbf$particles_per_level)
  if (length(merge_factors) != length(particles_per_level) || length(merge_factors) < 1) {
    stop("MAGIC GBF schedules must have the same positive number of levels.")
  }
  remaining_nodes <- length(centers)
  for (merge_factor in rev(merge_factors)) {
    if (!is.finite(merge_factor) || merge_factor < 2 || merge_factor %% 1 != 0 ||
        remaining_nodes %% merge_factor != 0) {
      stop("MAGIC GBF merge factors must reduce the batch count exactly at every level.")
    }
    remaining_nodes <- remaining_nodes / merge_factor
  }
  if (remaining_nodes != 1) stop("MAGIC GBF schedule must fuse all batches into one posterior.")
  data_split <- lapply(centers, function(batch) {
    rows <- raw$batch == batch
    .build_data_input(raw$outcome[rows], design[rows, , drop = FALSE])
  })
  if (length(sample_paths) != length(centers)) stop("Expected one posterior-sample file per batch.")
  node_draws <- lapply(sample_paths, function(path) {
    draws <- as.matrix(read.csv(path, check.names = FALSE))
    if (ncol(draws) != ncol(design)) stop("MAGIC posterior draw dimension mismatch.")
    draws
  })
  gbf <- .run_blr_gbf(node_draws, data_split, config)
  write.csv(gbf$draws, file.path(config$output_dir, "gbf_samples.csv"),
            row.names = FALSE)
  write_json(list(runtime_seconds = gbf$runtime_seconds),
             file.path(config$output_dir, "runtime.json"), pretty = TRUE,
             auto_unbox = TRUE)
  invisible(TRUE)
}
