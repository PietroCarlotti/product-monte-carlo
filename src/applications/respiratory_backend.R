##############################
# Import necessary libraries #
##############################

suppressWarnings(suppressPackageStartupMessages({
  library(DCFusion)
  library(HMCBLR)
  library(jsonlite)
}))

# Validate the respiratory smoke configuration.
validate_respiratory_config <- function(config) {
  required <- c("raw_data", "output_dir", "visit", "prior", "seeds", "gbf")
  if (!all(required %in% names(config)) || config$visit != 4 ||
      config$prior$variance != 4.0) {
    stop("Invalid respiratory smoke configuration.")
  }
  invisible(TRUE)
}

# Run the respiratory posterior and record backend artifacts.
run_respiratory_backend <- function(config_path, sample_paths) {
  config <- fromJSON(config_path, simplifyVector = FALSE)
  validate_respiratory_config(config)
  config_dir <- dirname(normalizePath(config_path))
  config$raw_data <- normalizePath(file.path(config_dir, config$raw_data), mustWork = TRUE)
  config$output_dir <- file.path(config_dir, config$output_dir)
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
  raw <- read.csv(config$raw_data, check.names = FALSE)
  feature_names <- c("intercept", "treat", "sex", "age", "baseline")
  required_columns <- c("center", "outcome", feature_names)
  if (!all(required_columns %in% names(raw))) {
    stop("Processed respiratory input is missing required columns.")
  }
  design <- as.matrix(raw[feature_names])
  centers <- sort(unique(raw$center))
  data_split <- lapply(centers, function(center) {
    rows <- raw$center == center
    .build_data_input(raw$outcome[rows], design[rows, , drop = FALSE])
  })
  if (length(sample_paths) != length(centers)) stop("Expected one posterior-sample file per center.")
  node_draws <- lapply(sample_paths, function(path) {
    draws <- as.matrix(read.csv(path, check.names = FALSE))
    if (ncol(draws) != ncol(design)) stop("Posterior draw dimension mismatch.")
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
