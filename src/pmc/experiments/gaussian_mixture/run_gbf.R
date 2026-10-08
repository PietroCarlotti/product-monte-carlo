##############################
# Import necessary libraries #
##############################

suppressPackageStartupMessages(library(jsonlite))

#######################
# Runtime provenance #
#######################

# Collect the installed version and optional source commit of a package.
.package_provenance <- function(package_name, include_remote_sha = FALSE) {
  description <- utils::packageDescription(package_name)
  provenance <- list(version = as.character(description$Version))
  if (include_remote_sha) {
    remote_sha <- description$RemoteSha
    if (is.null(remote_sha) || length(remote_sha) != 1 || is.na(remote_sha)) {
      remote_sha <- NA_character_
    }
    provenance$remote_sha <- remote_sha
  }

  provenance
}

# Collect R and GBF dependency versions for reproducibility.
.runtime_provenance <- function() {
  list(
    r = list(
      version = R.version.string,
      platform = R.version$platform
    ),
    packages = list(
      DCFusion = .package_provenance("DCFusion", TRUE),
      layeredBB = .package_provenance("layeredBB", TRUE),
      mvtnorm = .package_provenance("mvtnorm"),
      jsonlite = .package_provenance("jsonlite")
    )
  )
}

####################
# Fitted GBF runner #
####################

# Require a JSON object with exactly the permitted field names.
.validate_json_fields <- function(value, expected, label) {
  if (!is.list(value) || is.null(names(value)) ||
        length(names(value)) != length(expected) ||
        !setequal(names(value), expected)) {
    stop(paste(
      label, "must contain exactly:", paste(expected, collapse = ", ")
    ))
  }

  invisible(TRUE)
}

# Validate the fitted-node document without admitting generator metadata.
.validate_fitted_nodes <- function(fitted_nodes, adapter) {
  .validate_json_fields(
    fitted_nodes, c("num_nodes", "nodes"), "Fitted-node JSON"
  )
  adapter$.validate_positive_integer(fitted_nodes$num_nodes, "num_nodes")
  if (!is.list(fitted_nodes$nodes) ||
        length(fitted_nodes$nodes) != fitted_nodes$num_nodes) {
    stop("Fitted-node JSON must contain one mixture per node.")
  }

  adapter$.validate_nodes(fitted_nodes$nodes)
}

# Validate the GBF controls before creating any output artifacts.
.validate_gbf_config <- function(gbf_config, adapter) {
  .validate_json_fields(
    gbf_config,
    c("n_inference", "seed", "n_cores", "time_mesh",
      "ess_threshold", "cess_0_threshold"),
    "GBF config JSON"
  )
  adapter$.validate_positive_integer(gbf_config$n_inference, "n_inference")
  adapter$.validate_positive_integer(gbf_config$n_cores, "n_cores")
  adapter$.derive_stage_seed(gbf_config$seed, 1)
  adapter$.validate_ess_threshold(gbf_config$ess_threshold)
  adapter$.validate_cess_0_threshold(gbf_config$cess_0_threshold)
  if (is.character(gbf_config$time_mesh)) {
    if (length(gbf_config$time_mesh) != 1 ||
          gbf_config$time_mesh != "optimal") {
      stop("time_mesh must be optimal or a numeric mesh.")
    }
  } else {
    adapter$.validate_time_mesh(gbf_config$time_mesh)
  }

  invisible(TRUE)
}

# Read raw samples with one explicitly named column per fitted node.
.read_node_samples <- function(sample_path, num_nodes, adapter) {
  sample_frame <- utils::read.csv(sample_path, check.names = FALSE)
  expected_names <- paste0("node_", seq_len(num_nodes))
  if (length(names(sample_frame)) != length(expected_names) ||
        !all(names(sample_frame) == expected_names)) {
    stop("Raw samples must have ordered node_1 through node_k columns.")
  }

  adapter$.validate_samples(unname(sample_frame))
}

# Run the fitted node-specific DCFusion adaptation and persist raw artifacts.
run_gbf <- function(
  fitted_nodes_path,
  gbf_config_path,
  sample_path,
  artifact_dir,
  adapter_path
) {
  adapter <- new.env(parent = globalenv())
  sys.source(adapter_path, envir = adapter)
  repository_dir <- dirname(dirname(dirname(dirname(dirname(
    normalizePath(adapter_path)
  )))))
  algorithms_path <- file.path(
    repository_dir, "src", "fusion", "R", "algorithms.R"
  )
  sys.source(algorithms_path, envir = adapter)

  fitted_nodes <- jsonlite::fromJSON(
    fitted_nodes_path, simplifyVector = FALSE
  )
  gbf_config <- jsonlite::fromJSON(
    gbf_config_path, simplifyVector = TRUE
  )
  nodes <- .validate_fitted_nodes(fitted_nodes, adapter)
  .validate_gbf_config(gbf_config, adapter)
  samples <- .read_node_samples(
    sample_path, fitted_nodes$num_nodes, adapter
  )
  result <- adapter$gbf_univariate_mixture(
    samples = samples,
    nodes = nodes,
    n_inference = gbf_config$n_inference,
    time_mesh = gbf_config$time_mesh,
    ess_threshold = gbf_config$ess_threshold,
    cess_0_threshold = gbf_config$cess_0_threshold,
    seed = gbf_config$seed,
    n_cores = gbf_config$n_cores
  )

  dir.create(artifact_dir, recursive = TRUE, showWarnings = FALSE)
  sample_output <- file.path(artifact_dir, "gbf_samples.csv")
  weighted_output <- file.path(artifact_dir, "gbf_weighted_particles.csv")
  diagnostic_output <- file.path(artifact_dir, "gbf_diagnostics.json")
  utils::write.csv(
    data.frame(sample = result$equal_weight_samples),
    sample_output,
    row.names = FALSE
  )
  utils::write.csv(
    data.frame(
      particle = result$weighted_particles,
      normalized_weight = result$normalized_weights
    ),
    weighted_output,
    row.names = FALSE
  )
  diagnostics <- result[
    !names(result) %in% c(
      "equal_weight_samples",
      "weighted_particles",
      "normalized_weights"
    )
  ]
  diagnostics$gbf_implementation <-
    "fitted_univariate_mixture_dcfusion_adapter"
  diagnostics$parameter_source <- "fitted_from_raw_node_samples"
  diagnostics$n_inference <- gbf_config$n_inference
  diagnostics$seed <- gbf_config$seed
  diagnostics$component_samples <- basename(sample_path)
  diagnostics$fitted_nodes <- basename(fitted_nodes_path)
  diagnostics$scoring_sample_role <- "equal_weight_terminal_resample"
  diagnostics$weighted_particles <- basename(weighted_output)
  diagnostics$archive_cess_definition <-
    "DCFusion/JMLR Equation (27) incremental-weight CESS."
  diagnostics$runtime_provenance <- .runtime_provenance()
  jsonlite::write_json(
    diagnostics,
    diagnostic_output,
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = NA
  )

  invisible(diagnostics)
}

# Parse the strict command-line interface and invoke the fitted GBF runner.
.main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 4) {
    stop(paste(
      "Expected fitted nodes JSON, GBF config JSON, raw samples CSV,",
      "and artifact directory."
    ))
  }
  command_args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub(
    "^--file=", "", grep("^--file=", command_args, value = TRUE)
  )
  if (length(script_path) != 1) {
    stop("The GBF runner must be executed with Rscript.")
  }
  adapter_path <- file.path(
    dirname(normalizePath(script_path)), "gbf_adapter.R"
  )
  run_gbf(args[1], args[2], args[3], args[4], adapter_path)
}

if (sys.nframe() == 0) {
  .main()
}
