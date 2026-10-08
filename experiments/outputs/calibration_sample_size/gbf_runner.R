##############################
# Import necessary libraries #
##############################

suppressPackageStartupMessages(library(jsonlite))

###################
# Package loading #
###################

# Return the directory containing this script when run with Rscript.
.script_directory <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub(
    "^--file=", "", grep("^--file=", command_args, value = TRUE)
  )
  if (length(script_path) != 1) {
    stop("The GBF runner must be executed with Rscript.")
  }

  dirname(normalizePath(script_path))
}

# Load the fusion package from the repository source tree.
.load_fusion_package <- function(script_directory) {
  fusion_directory <- normalizePath(file.path(
    script_directory, "..", "..", "..", "src", "fusion"
  ))
  suppressPackageStartupMessages({
    library(DCFusion)
    library(layeredBB)
    devtools::load_all(fusion_directory, quiet = TRUE)
  })

  invisible(fusion_directory)
}

###################
# GBF repetitions #
###################

# Fit node Gaussians, choose the optimal time grid, and run GBF once.
.run_repetition <- function(repetition, samples, settings, seed) {
  set.seed(seed)
  started <- proc.time()[["elapsed"]]
  mu <- vapply(samples, mean, numeric(1))
  sigma <- vapply(samples, sd, numeric(1))
  grid <- fusion::optimal_time_grid(
    method = "GBF",
    samples = samples,
    k = length(samples),
    n = length(samples[[1]]),
    cess_0_threshold = settings$cess_0_threshold
  )
  result <- fusion::gbf_univariate_gaussian(
    n_inference = settings$n_inference,
    samples = samples,
    mu = mu,
    sigma = sigma,
    time_grid = grid$time_grid,
    precondition = settings$precondition,
    ess_threshold = settings$ess_threshold,
    diffusion_estimator = settings$diffusion_estimator,
    seed = seed
  )
  data.frame(
    repetition = repetition,
    estimate = mean(result$samples),
    draws_standard_deviation = sd(result$samples),
    final_ess = as.numeric(tail(result$ESS, 1)),
    final_cess = as.numeric(tail(result$CESS, 1)),
    n_draws = length(result$samples),
    coalescence_time = grid$coalescence_time,
    n_time_steps = grid$n_time_steps,
    runtime_seconds = proc.time()[["elapsed"]] - started
  )
}

# Append one result row to the output CSV, writing the header once.
.append_row <- function(row, output_path) {
  utils::write.table(
    row,
    output_path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(output_path),
    append = file.exists(output_path)
  )
}

# Run GBF on every repetition in the samples CSV.
run_gbf_batch <- function(sample_path, settings_path, output_path) {
  settings <- jsonlite::fromJSON(settings_path, simplifyVector = TRUE)
  sample_frame <- utils::read.csv(sample_path, check.names = FALSE)
  node_columns <- grep("^node_", names(sample_frame), value = TRUE)
  for (repetition in unique(sample_frame$repetition)) {
    rows <- sample_frame[sample_frame$repetition == repetition, ]
    samples <- lapply(node_columns, function(column) rows[[column]])
    row <- .run_repetition(
      repetition, samples, settings, as.integer(rows$gbf_seed[[1]])
    )
    .append_row(row, output_path)
  }

  invisible(output_path)
}

######################
# Script entry point #
######################

# Parse the command-line interface and run the GBF batch.
.main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 3) {
    stop("Expected samples CSV, settings JSON, and output CSV paths.")
  }
  .load_fusion_package(.script_directory())
  run_gbf_batch(args[1], args[2], args[3])
}

if (sys.nframe() == 0) {
  .main()
}
