##############################
# Import necessary libraries #
##############################

# Set a timeout for downloading packages
options(timeout = 30)

# Base R
library(parallel)

# CRAN
cran_packages <- c(
  "devtools",
  "pbapply",
  "jsonlite",
  "MASS",
  "transport",
  "callr"
)

for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# GitHub
github_repos <- c(
  "rchan26/layeredBB",
  "rchan26/HMCBLR",
  "rchan26/HMCBRR",
  "rchan26/HMCGLMR",
  "rchan26/DCFusion"
)

for (repo in github_repos) {
  pkg <- basename(repo)
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE))
      install.packages("remotes")
    remotes::install_github(repo)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

#############################
# Set the working directory #
#############################

# Get the directory of this script
working_directory <- tryCatch(
  dirname(rstudioapi::getSourceEditorContext()$path),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("--file=", "", file_arg[1])))
    } else {
      getwd()
    }
  }
)

# Set the working directory
setwd(working_directory)

# Optional output directory passed as argument
args <- commandArgs(trailingOnly = TRUE)
config_files <- args[grepl("\\.json$", args)]
out_args <- args[!grepl("\\.json$", args)]

# Override the output directory if one was provided
if (length(out_args) >= 1) {
  setwd(normalizePath(out_args[1], mustWork = FALSE))
}

# Override the results filename if one was provided
results_filename <- if (length(out_args) >= 2) out_args[2] else "results.csv"

# Results CSV path
csv_path <- file.path(getwd(), results_filename)

###########
# Package #
###########

# Package root directory
src_fusion_dir <- normalizePath(file.path(
  working_directory, "..", "..", "src", "fusion"
))

# Load the fusion package
devtools::load_all(src_fusion_dir)

##################
# Configurations #
##################

# Parse an optional positive-integer configuration limit
parse_positive_integer_env <- function(name) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) {
    return(NULL)
  }
  parsed_value <- suppressWarnings(as.numeric(value))
  if (!grepl("^[1-9][0-9]*$", value) || !is.finite(parsed_value)) {
    stop(
      paste0(name, " must be a positive integer, got ", dQuote(value), ".")
    )
  }
  parsed_value
}

# Parse the optional existing-results resume flag
parse_resume_existing <- function() {
  value <- Sys.getenv("FUSION_RESUME_EXISTING", unset = "0")
  if (value == "") {
    value <- "0"
  }
  if (!(value %in% c("0", "1"))) {
    stop(
      paste0(
        "FUSION_RESUME_EXISTING must be 0 or 1, got ",
        dQuote(value), "."
      )
    )
  }
  value == "1"
}

# Parse a strict binary environment flag
parse_binary_env <- function(name) {
  value <- Sys.getenv(name, unset = "0")
  if (!(value %in% c("0", "1"))) {
    stop(paste0(name, " must be 0 or 1, got ", dQuote(value), "."))
  }
  value == "1"
}

# Normalize one recovery identity scalar to the planner representation
normalize_recovery_scalar <- function(value, label) {
  if (is.null(value)) {
    return(NA_character_)
  }
  if (length(value) != 1 || is.list(value)) {
    stop(paste0(label, " must be a scalar or null."))
  }
  if (is.numeric(value)) {
    if (is.na(value) || !is.finite(value)) {
      return(NA_character_)
    }
    if (value == trunc(value)) {
      return(format(value, scientific = FALSE, trim = TRUE))
    }
  }
  normalized <- as.character(value)
  if (is.na(normalized)) {
    return(NA_character_)
  }
  normalized
}

# Encode one normalized recovery scalar without collisions
recovery_scalar_key <- function(value) {
  if (is.na(value)) {
    return("N")
  }
  paste0("S", nchar(value, type = "bytes"), ":", value)
}

# Build a collision-safe exact recovery identity key
recovery_identity_key <- function(record, label, require_fields = FALSE) {
  fields <- c(
    "seed", "iteration_id", "parametric_form", "num_nodes", "n",
    "algorithm", "method"
  )
  if (!is.list(record)) {
    stop(paste0(label, " must be an object."))
  }
  missing_fields <- fields[!(fields %in% names(record))]
  if (require_fields && length(missing_fields) > 0) {
    stop(
      paste0(
        label, " is missing identity fields: ",
        paste(missing_fields, collapse = ", "), "."
      )
    )
  }
  values <- vapply(
    fields,
    function(field) {
      normalize_recovery_scalar(
        record[[field]],
        paste0(label, ".", field)
      )
    },
    character(1)
  )
  encoded <- vapply(values, recovery_scalar_key, character(1))
  paste(encoded, collapse = "|")
}

# Validate target counts and return normalized missing targets
validate_recovery_targets <- function(config, label) {
  rows_per_h <- config$expected_rows_per_h
  if (
    is.null(rows_per_h) || length(rows_per_h) != 1 ||
      !is.numeric(rows_per_h) || is.na(rows_per_h) ||
      !is.finite(rows_per_h) || rows_per_h <= 0 ||
      rows_per_h != trunc(rows_per_h)
  ) {
    stop(
      paste0(
        label,
        ".expected_rows_per_h must be a positive integer."
      )
    )
  }
  h_values <- config$h_values
  if (!is.list(h_values) || length(h_values) == 0) {
    stop(paste0(label, ".h_values must be a nonempty list."))
  }
  normalized_h <- vapply(
    seq_along(h_values),
    function(index) {
      normalize_recovery_scalar(
        h_values[[index]],
        paste0(label, ".h_values[", index, "]")
      )
    },
    character(1)
  )
  h_keys <- vapply(normalized_h, recovery_scalar_key, character(1))
  if (anyDuplicated(h_keys)) {
    stop(paste0(label, ".h_values must be distinct."))
  }
  observed <- config$observed_rows_by_h
  if (!is.list(observed) || length(observed) != length(h_values)) {
    stop(paste0(label, ".observed_rows_by_h must match h_values."))
  }
  observed_h <- vapply(
    seq_along(observed),
    function(index) {
      target <- observed[[index]]
      target_label <- paste0(
        label,
        ".observed_rows_by_h[",
        index,
        "]"
      )
      if (!is.list(target) || !("h" %in% names(target))) {
        stop(paste0(target_label, " must be an object with h."))
      }
      normalize_recovery_scalar(
        target$h,
        paste0(target_label, ".h")
      )
    },
    character(1)
  )
  observed_keys <- vapply(
    observed_h,
    recovery_scalar_key,
    character(1)
  )
  if (anyDuplicated(observed_keys) || !setequal(observed_keys, h_keys)) {
    stop(paste0(label, ".observed_rows_by_h must match h_values."))
  }
  observed_counts <- vapply(
    seq_along(observed),
    function(index) {
      count <- observed[[index]]$observed_row_count
      if (
        is.null(count) || length(count) != 1 ||
          !is.numeric(count) || is.na(count) || !is.finite(count) ||
          count < 0 || count != trunc(count)
      ) {
        stop(
          paste0(
            label,
            ".observed_rows_by_h[",
            index,
            "].observed_row_count must be a nonnegative integer."
          )
        )
      }
      as.integer(count)
    },
    integer(1)
  )
  counts <- observed_counts[match(h_keys, observed_keys)]
  status <- config$status
  if (identical(status, "surplus")) {
    stop("Fusion recovery plan contains a surplus configuration.")
  }
  if (any(!(counts %in% c(0L, as.integer(rows_per_h))))) {
    stop(
      paste0(label, " contains an unsafe within-target partial count.")
    )
  }
  missing <- counts == 0L
  if (identical(status, "complete") && any(missing)) {
    stop(paste0(label, " complete status contradicts target counts."))
  }
  if (identical(status, "missing") && !all(missing)) {
    stop(paste0(label, " missing status contradicts target counts."))
  }
  if (identical(status, "partial") && (!any(missing) || all(missing))) {
    stop(paste0(label, " partial status contradicts target counts."))
  }
  list(
    h_values = normalized_h,
    h_keys = h_keys,
    missing_h_values = normalized_h[missing],
    missing_h_keys = h_keys[missing]
  )
}

# Filter native Fusion configs to exact recoverable targets
filter_recovery_configs <- function(configurations, plan_path) {
  normalized_path <- normalizePath(plan_path, mustWork = TRUE)
  plan <- tryCatch(
    jsonlite::read_json(normalized_path, simplifyVector = FALSE),
    error = function(error) {
      stop(
        paste0(
          "Could not read Fusion recovery plan: ",
          conditionMessage(error)
        ),
        call. = FALSE
      )
    }
  )
  if (
    is.null(plan$schema_version) || length(plan$schema_version) != 1 ||
      !is.numeric(plan$schema_version) || plan$schema_version != 1
  ) {
    stop("Fusion recovery plan must have schema_version 1.")
  }
  if (
    is.null(plan$result) || !is.list(plan$result) ||
      !identical(plan$result$state, "valid")
  ) {
    stop("Fusion recovery requires a plan with valid result state.")
  }
  if (
    is.null(plan$outside_requested_design) ||
      !is.list(plan$outside_requested_design) ||
      length(plan$outside_requested_design) != 0
  ) {
    stop("Fusion recovery plan contains rows outside the requested design.")
  }
  summary <- plan$summary
  if (
    !is.list(summary) ||
      !identical(summary$outside_requested_design_identity_count, 0L) ||
      !identical(summary$outside_requested_design_row_count, 0L)
  ) {
    stop("Fusion recovery summary reports outside-design rows.")
  }
  plan_configs <- plan$configurations
  if (!is.list(plan_configs) || length(plan_configs) == 0) {
    stop("Fusion recovery plan contains no requested configurations.")
  }
  stages <- vapply(
    plan_configs,
    function(config) {
      if (
        is.null(config$stage) || length(config$stage) != 1 ||
          !(config$stage %in% c("pmc", "fusion"))
      ) {
        stop("Fusion recovery plan contains an invalid stage.")
      }
      config$stage
    },
    character(1)
  )
  statuses <- vapply(
    plan_configs,
    function(config) {
      if (
        is.null(config$status) || length(config$status) != 1 ||
          !(config$status %in% c(
            "complete", "missing", "partial", "surplus"
          ))
      ) {
        stop("Fusion recovery plan contains an invalid status.")
      }
      config$status
    },
    character(1)
  )
  targets <- lapply(
    seq_along(plan_configs),
    function(index) {
      validate_recovery_targets(
        plan_configs[[index]],
        paste0("configuration ", index)
      )
    }
  )
  if (any(stages == "pmc" & statuses != "complete")) {
    stop("Fusion recovery requires every PMC configuration to be complete.")
  }
  plan_keys <- vapply(
    seq_along(plan_configs),
    function(index) {
      recovery_identity_key(
        plan_configs[[index]],
        paste0("configuration ", index),
        require_fields = TRUE
      )
    },
    character(1)
  )
  if (anyDuplicated(plan_keys)) {
    stop("Fusion recovery plan contains duplicate configuration identities.")
  }
  requested_indices <- which(
    stages == "fusion" & statuses %in% c("missing", "partial")
  )
  if (length(requested_indices) == 0) {
    stop(
      paste0(
        "Fusion recovery plan contains no missing or recoverable ",
        "partial configurations."
      )
    )
  }
  requested_keys <- plan_keys[requested_indices]
  config_keys <- vapply(
    seq_along(configurations),
    function(index) {
      recovery_identity_key(
        configurations[[index]],
        paste0("expanded Fusion configuration ", index)
      )
    },
    character(1)
  )
  if (anyDuplicated(config_keys)) {
    stop("Expanded Fusion configurations contain duplicate identities.")
  }
  selected_indices <- match(requested_keys, config_keys)
  if (anyNA(selected_indices)) {
    stop("Fusion recovery plan does not match expanded configurations.")
  }
  selected <- lapply(
    seq_along(selected_indices),
    function(index) {
      config <- configurations[[selected_indices[[index]]]]
      native_h_values <- config$h_values
      if (is.null(native_h_values) || length(native_h_values) == 0) {
        stop("Expanded Fusion configuration has invalid h_values.")
      }
      normalized_h <- vapply(
        seq_along(native_h_values),
        function(h_index) {
          normalize_recovery_scalar(
            native_h_values[[h_index]],
            paste0("expanded h_values[", h_index, "]")
          )
        },
        character(1)
      )
      native_keys <- vapply(
        normalized_h,
        recovery_scalar_key,
        character(1)
      )
      plan_targets <- targets[[requested_indices[[index]]]]
      if (
        anyDuplicated(native_keys) ||
          !identical(native_keys, plan_targets$h_keys)
      ) {
        stop(
          paste0(
            "Fusion recovery plan h_values do not match expanded ",
            "configuration."
          )
        )
      }
      keep <- native_keys %in% plan_targets$missing_h_keys
      config$h_values <- native_h_values[keep]
      if (length(config$h_values) != length(plan_targets$missing_h_keys)) {
        stop("Fusion recovery target matching is not one-to-one.")
      }
      config
    }
  )
  selected_keys <- vapply(
    seq_along(selected),
    function(index) {
      recovery_identity_key(
        selected[[index]],
        paste0("selected Fusion configuration ", index)
      )
    },
    character(1)
  )
  if (!identical(selected_keys, requested_keys)) {
    stop("Fusion recovery identity matching is not one-to-one.")
  }
  missing_target_count <- sum(vapply(
    requested_indices,
    function(index) length(targets[[index]]$missing_h_keys),
    integer(1)
  ))
  message(
    "Fusion recovery selected ", length(selected),
    " recoverable configurations with ", missing_target_count,
    " missing h targets from ", normalized_path, "."
  )
  selected
}

# Extract and validate the completion keys for one configuration
configuration_completion_keys <- function(config) {
  seed <- config$seed
  if (
    is.null(seed) || length(seed) != 1 || !is.numeric(seed) ||
      is.na(seed) || !is.finite(seed)
  ) {
    stop("Resume requires every configuration to have a finite seed.")
  }
  method <- config$method
  if (
    is.null(method) || length(method) != 1 || is.na(method) ||
      !nzchar(as.character(method))
  ) {
    stop("Resume requires every configuration to have a method.")
  }
  method <- as.character(method)
  rows_per_h <- 1L
  if (
    !is.null(config$p) && length(config$p) == 1 &&
      is.numeric(config$p) && !is.na(config$p) &&
      is.finite(config$p) && config$p > 0
  ) {
    rows_per_h <- as.integer(config$p)
  } else if (
    !is.null(config$DGP_parameters$mu) &&
      length(config$DGP_parameters$mu) >= 1
  ) {
    first_mu <- config$DGP_parameters$mu[[1]]
    if (is.numeric(first_mu) && length(first_mu) > 0) {
      rows_per_h <- as.integer(length(first_mu))
    }
  }
  h_values <- config$h_values
  if (
    is.null(h_values) || length(h_values) == 0 ||
      any(vapply(h_values, is.null, logical(1)))
  ) {
    stop("Resume requires every configuration to have configured h_values.")
  }
  h_values <- as.character(unlist(h_values, use.names = FALSE))
  if (
    length(h_values) == 0 || any(is.na(h_values)) || any(!nzchar(h_values)) ||
      anyDuplicated(h_values)
  ) {
    stop(
      paste0(
        "Resume requires distinct, nonempty h_values for seed ", seed, "."
      )
    )
  }
  list(
    seed = seed, method = method, h_values = h_values,
    rows_per_h = rows_per_h
  )
}

# Remove configurations already completed in an existing results CSV
filter_completed_configs <- function(
  configurations,
  csv_path,
  validation_configurations = configurations,
  allow_target_partial = FALSE
) {
  completion_keys <- lapply(
    validation_configurations,
    configuration_completion_keys
  )
  seeds <- vapply(completion_keys, function(keys) keys$seed, numeric(1))
  if (anyDuplicated(seeds)) {
    stop("Resume requires a unique seed for every candidate configuration.")
  }
  selected_keys <- lapply(
    configurations,
    configuration_completion_keys
  )

  if (!file.exists(csv_path)) {
    message(
      "Resume skipped 0 completed configurations; ",
      length(configurations), " configurations remain."
    )
    return(configurations)
  }

  existing <- tryCatch(
    utils::read.csv(
      csv_path, stringsAsFactors = FALSE, check.names = FALSE
    ),
    error = function(error) {
      stop(
        paste0(
          "Could not read existing results CSV: ",
          conditionMessage(error)
        ),
        call. = FALSE
      )
    }
  )
  required_columns <- c("seed", "method", "h")
  if (!all(required_columns %in% names(existing))) {
    stop("Existing results CSV must contain seed, method, and h columns.")
  }
  paired_methods <- any(vapply(
    validation_configurations,
    function(config) {
      !is.null(config$seed_strategy) &&
        config$seed_strategy == "paired_methods"
    },
    logical(1)
  ))
  expected_columns <- fusion:::.result_columns(
    validation_configurations[[1]]$dgp_schema,
    paired_methods = paired_methods
  )
  if (!identical(names(existing), expected_columns)) {
    stop("Existing results CSV has an incompatible result schema.")
  }
  existing_seeds <- suppressWarnings(as.numeric(as.character(existing$seed)))
  existing_methods <- as.character(existing$method)
  complete_targets <- character(0)

  for (i in seq_along(validation_configurations)) {
    keys <- completion_keys[[i]]
    rows <- existing[
      !is.na(existing_seeds) & existing_seeds == keys$seed &
        !is.na(existing_methods) & existing_methods == keys$method,
      ,
      drop = FALSE
    ]
    if (nrow(rows) == 0) {
      next
    }

    existing_h <- as.character(rows$h)
    h_counts <- vapply(
      keys$h_values,
      function(h_value) sum(existing_h == h_value, na.rm = TRUE),
      integer(1)
    )
    unexpected_h <- is.na(existing_h) |
      !(existing_h %in% keys$h_values)
    if (any(unexpected_h)) {
      stop(
        paste0(
          "Existing results contain an unexpected h for seed ",
          keys$seed, "."
        )
      )
    }
    if (any(h_counts > keys$rows_per_h)) {
      stop(
        paste0(
          "Existing results are duplicated for seed ",
          keys$seed, "; refusing to append duplicate rows."
        )
      )
    }
    if (any(!(h_counts %in% c(0L, keys$rows_per_h)))) {
      stop(
        paste0(
          "Existing results contain a partial target for seed ",
          keys$seed, "; refusing unsafe recovery."
        )
      )
    }
    full_h <- keys$h_values[h_counts == keys$rows_per_h]
    complete_targets <- c(
      complete_targets,
      paste(keys$seed, keys$method, full_h, sep = "\034")
    )
    if (all(h_counts == keys$rows_per_h)) {
      next
    }
    if (any(h_counts > 0L) && !allow_target_partial) {
      stop(
        paste0(
          "Existing results are incomplete for seed ",
          keys$seed, "; refusing to append duplicate rows."
        )
      )
    }
  }

  remaining <- lapply(
    seq_along(configurations),
    function(index) {
      config <- configurations[[index]]
      keys <- selected_keys[[index]]
      target_keys <- paste(
        keys$seed,
        keys$method,
        keys$h_values,
        sep = "\034"
      )
      keep_h <- !(target_keys %in% complete_targets)
      config$h_values <- config$h_values[keep_h]
      config
    }
  )
  keep <- vapply(
    remaining,
    function(config) length(config$h_values) > 0,
    logical(1)
  )
  message(
    "Resume skipped ", sum(!keep), " completed configurations; ",
    sum(keep), " configurations remain."
  )
  remaining[keep]
}

# Get the list of config files
if (length(config_files) == 0)
  stop("No config files provided. Pass one or more JSON files as arguments.")

message("Loading configurations.")

configurations <- build_configs(config_files)

# Filter only after seed-bearing configurations have been built
max_n <- parse_positive_integer_env("FUSION_MAX_N")
if (!is.null(max_n)) {
  configurations <- Filter(
    function(config) config$n <= max_n,
    configurations
  )
  if (length(configurations) == 0) {
    stop("FUSION_MAX_N removed all configurations.")
  }
}

# Restrict execution to one exact sample size when requested
exact_n <- parse_positive_integer_env("PMC_EXACT_N")
if (!is.null(exact_n)) {
  configurations <- Filter(
    function(config) config$n == exact_n,
    configurations
  )
  if (length(configurations) == 0) {
    stop(paste0("PMC_EXACT_N=", exact_n, " removed all configurations."))
  }
}

resume_configurations <- configurations

# Restrict recovery execution to exact missing Fusion configurations
recovery_plan_path <- Sys.getenv("FUSION_RECOVERY_PLAN", unset = "")
recovery_dry_run <- parse_binary_env("FUSION_RECOVERY_DRY_RUN")
if (recovery_dry_run && !nzchar(recovery_plan_path)) {
  stop("FUSION_RECOVERY_DRY_RUN requires FUSION_RECOVERY_PLAN.")
}
if (nzchar(recovery_plan_path)) {
  configurations <- filter_recovery_configs(
    configurations,
    recovery_plan_path
  )
}

# Skip only configurations whose target rows are unambiguously complete
if (parse_resume_existing()) {
  configurations <- filter_completed_configs(
    configurations,
    csv_path,
    validation_configurations = resume_configurations,
    allow_target_partial = nzchar(recovery_plan_path)
  )
}

# Exit before hardware or simulation setup for recovery selection checks
if (recovery_dry_run) {
  message(
    "Fusion recovery dry run selected ", length(configurations),
    " configurations after resume filtering."
  )
  quit(save = "no", status = 0, runLast = FALSE)
}

message("Loaded ", length(configurations), " configurations.")

##################
# Hardware setup #
##################

# Set the maximum number of CPUs to use
max_cpu <- 13
max_cpu_override <- Sys.getenv("FUSION_MAX_CPU", unset = NA_character_)
if (!is.na(max_cpu_override)) {
  parsed_max_cpu <- suppressWarnings(as.integer(max_cpu_override))
  if (!grepl("^[1-9][0-9]*$", max_cpu_override) || is.na(parsed_max_cpu)) {
    stop(
      paste0(
        "FUSION_MAX_CPU must be a positive integer, got ",
        dQuote(max_cpu_override), "."
      )
    )
  }
  max_cpu <- parsed_max_cpu
}

# Check the number of available CPUs
available_cpu <- max(1L, parallel::detectCores() - 1L, na.rm = TRUE)

# Determine the number of CPUs to use
n_cpu <- min(max_cpu, available_cpu)

# Use sequential execution when only one CPU is requested
cluster <- NULL

# Create workers only for multi-core execution
if (n_cpu > 1L) {
  message("Starting PSOCK cluster with ", n_cpu, " workers.")

  # Worker message()/warning/error output would otherwise be discarded
  workers_log_path <- file.path(
    dirname(csv_path),
    paste0(tools::file_path_sans_ext(results_filename), "_workers.log")
  )
  cluster <- parallel::makeCluster(n_cpu, outfile = workers_log_path)
  on.exit(parallel::stopCluster(cluster), add = TRUE)

  message("Worker output is being logged to ", workers_log_path, ".")

  message("Loading fusion dependencies on PSOCK workers.")

  # Setting the work environment for every worker
  invisible(parallel::clusterCall(cluster, function(wd, src_fusion_dir) {
    setwd(wd)
    library(DCFusion)
    library(layeredBB)
    library(MASS)
    library(transport)
    devtools::load_all(src_fusion_dir)
    invisible(NULL)
  },
  working_directory, src_fusion_dir
  ))

  message("PSOCK cluster is ready.")
} else {
  message("Using sequential fusion execution.")
}

###############
# Simulations #
###############

# Create a unique intermediate directory for incremental results
tmp_dir <- file.path(
  getwd(),
  paste0(
    ".sim_results_tmp_",
    tools::file_path_sans_ext(results_filename)
  )
)
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)

# Per-invocation failure marker shared by the main process and PSOCK workers
invocation_failure_path <- file.path(tmp_dir, ".invocation_failed")
unlink(invocation_failure_path)

# Lock file path (workers need absolute paths)
lock_path <- file.path(tmp_dir, ".csv.lock")

# Parse an optional per-config wall-clock timeout override (seconds); NULL
# Falls back to simulate_config_with_retry's own generous default
timeout_override <- parse_positive_integer_env(
  "FUSION_SIMULATION_TIMEOUT_SECONDS"
)

# Failure-tracking CSV path, sibling to the main results CSV
failures_csv_path <- file.path(
  dirname(csv_path),
  paste0(tools::file_path_sans_ext(results_filename), "_failures.csv")
)

# Failure-tracking lock file, distinct from the results CSV lock
failures_lock_path <- file.path(tmp_dir, ".failures_csv.lock")

# Share output paths with workers when a cluster is active
if (!is.null(cluster)) {
  parallel::clusterExport(
    cluster, c(
      "tmp_dir", "csv_path", "lock_path", "timeout_override",
      "failures_csv_path", "failures_lock_path", "invocation_failure_path"
    )
  )
}

# Run all configurations in parallel
pbapply::pblapply(
  configurations,
  function(config) {
    # Save plots to a per-run directory only when the config opts in
    plots_directory <- if (isTRUE(config$save_plots)) {
      file.path(getwd(), "plots")
    } else {
      NULL
    }
    # ggsave requires the target directory to already exist
    if (!is.null(plots_directory)) {
      dir.create(plots_directory, showWarnings = FALSE, recursive = TRUE)
    }
    # Run the simulation with a wall-clock timeout and reseeded retries;
    # tryCatch here is a defense-in-depth backstop for anything that
    # escapes simulate_config_with_retry's own internal error handling
    simulation <- tryCatch(
      list(
        result = if (is.null(timeout_override)) {
          simulate_config_with_retry(config, plots_directory)
        } else {
          simulate_config_with_retry(
            config, plots_directory, timeout_seconds = timeout_override
          )
        },
        error_message = NULL
      ),
      error = function(e) {
        message(conditionMessage(e))
        list(result = NULL, error_message = conditionMessage(e))
      }
    )
    if (is.null(simulation$result)) {
      error_message <- if (is.null(simulation$error_message)) {
        "Simulation returned no result after exhausting its retries."
      } else {
        simulation$error_message
      }
      # Record a structured failure row so it is never silently lost
      failure_row <- data.frame(
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        seed = if (!is.null(config$seed)) config$seed else NA,
        method = if (!is.null(config$method)) config$method else NA,
        n = if (!is.null(config$n)) config$n else NA,
        iteration_id = if (!is.null(config$iteration_id)) {
          config$iteration_id
        } else {
          NA
        },
        error_message = error_message,
        stringsAsFactors = FALSE
      )
      append_to_csv(failure_row, failures_csv_path, failures_lock_path)
      file.create(invocation_failure_path)
    }
    # Append every row from one completed configuration under one lock
    if (!is.null(simulation$result)) {
      result_df <- do.call(
        rbind,
        lapply(simulation$result, as.data.frame)
      )
      append_to_csv(result_df, csv_path, lock_path)
    }
    invisible(NULL)
  },
  cl = cluster
)

# Check if simulations ran successfully
results_missing <- !file.exists(csv_path) || file.info(csv_path)$size == 0

# Check only failures recorded during this invocation
invocation_failed <- file.exists(invocation_failure_path)

# Clean up intermediate directory
unlink(tmp_dir, recursive = TRUE)

if (results_missing) {
  stop("All simulations failed. No results were produced.")
}

if (invocation_failed) {
  stop(
    paste0(
      "One or more configurations failed; see ", failures_csv_path, "."
    )
  )
}

message("\nAll simulations completed.")
