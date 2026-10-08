##########################
# Configuration building #
##########################

# Scalar-per-node DGP-parameter keys by parametric form (univariate_Gaussian's
# mu/sigma, Beta's a/b)
.scalar_keys_by_form <- list(
  univariate_Gaussian = c("mu", "sigma"),
  Beta = c("a", "b")
)

# Canonical column order for scalar DGP-parameter keys across all forms
.scalar_keys_canonical_order <- unique(unlist(
  .scalar_keys_by_form, use.names = FALSE
))

# Use the compact result schema for high-dimensional Gaussian configurations
.compact_result_dimension <- 50

# Build a stable key for the random data-generating scenario
#' @noRd
.data_scenario_key <- function(config) {
  scenario <- config[c(
    "parametric_form", "DGP_parameters", "num_nodes", "n"
  )]
  jsonlite::toJSON(scenario, auto_unbox = TRUE, digits = NA)
}

#' Compute the DGP schema (max num_nodes/p, scalar keys, CNF usage) for a batch
#'
#' @param configurations list of per-simulation configuration lists
#' @return list with max_num_nodes, max_p, scalar_keys, uses_cnf, compact
#' @keywords internal
.compute_dgp_schema <- function(configurations) {
  max_num_nodes <- max(sapply(configurations, function(cfg) cfg$num_nodes))
  max_p <- 0
  scalar_keys_in_use <- character(0)
  for (cfg in configurations) {
    if (cfg$parametric_form %in%
          c("multivariate_Gaussian", "bivariate_Gaussian")) {
      p <- length(cfg$DGP_parameters$mu[[1]])
      max_p <- max(max_p, p)
    }
    scalar_keys_in_use <- union(
      scalar_keys_in_use, .scalar_keys_by_form[[cfg$parametric_form]]
    )
  }
  scalar_keys <- .scalar_keys_canonical_order[
    .scalar_keys_canonical_order %in% scalar_keys_in_use
  ]
  uses_cnf <- any(vapply(
    configurations, function(cfg) cfg$method == "CNF", logical(1)
  ))
  list(
    max_num_nodes = max_num_nodes, max_p = max_p, scalar_keys = scalar_keys,
    uses_cnf = uses_cnf,
    compact = max_p >= .compact_result_dimension
  )
}

# Validate a reserved selector without adding it to the configuration grid
.validate_selector <- function(value, name) {
  if (length(value) != 1 || !is.integer(value) || is.na(value) ||
        value <= 0) {
    stop(paste0(name, " must be a positive integer."))
  }
  value
}

#' Build the list of simulation configurations from JSON config files
#'
#' Loads and merges one or more JSON config files into a grid of
#' per-simulation configuration lists, expanding parameter combinations
#' and repeating them across `n_iterations`. The `seed` field retains its
#' legacy per-configuration sequence. Setting `seed_strategy` to
#' `paired_methods` additionally creates shared `data_seed` values across
#' methods and distinct `inference_seed` values across full configurations.
#'
#' @param config_files character vector of paths to JSON config files
#' @return list of per-simulation configuration lists
#' @export
build_configs <- function(config_files) {
  # Require at least one JSON config file
  if (length(config_files) == 0) {
    stop("At least one JSON config file must be provided.")
  }

  # Load and merge all JSON config files
  seed <- NULL
  seed_strategy <- "independent"
  n_iterations <- NULL
  iteration_start <- 0
  h_values <- NULL
  configuration_parameters <- list()
  selectors <- list()

  for (cf in config_files) {
    ext <- jsonlite::fromJSON(cf, simplifyVector = FALSE)
    for (key in c("selected_n", "selected_p")) {
      if (key %in% names(ext)) {
        selectors[key] <- list(ext[[key]])
      }
    }
    if ("seed" %in% names(ext))
      seed <- ext$seed
    if ("seed_strategy" %in% names(ext))
      seed_strategy <- ext$seed_strategy
    if (!is.null(ext$n_iterations))
      n_iterations <- as.integer(ext$n_iterations)
    if ("iteration_start" %in% names(ext))
      iteration_start <- ext$iteration_start
    if (!is.null(ext$h))
      h_values <- ext$h
    reserved_keys <- c(
      "seed", "seed_strategy", "n_iterations", "iteration_start", "h",
      "selected_n", "selected_p"
    )
    for (k in setdiff(names(ext), reserved_keys)) {
      configuration_parameters[[k]] <- switch(k,
        "DGP_parameters" = lapply(ext[[k]], function(dgp) {
          if (all(c("mu", "cov") %in% names(dgp))) {
            list(
              mu = lapply(dgp$mu, as.numeric),
              cov = lapply(dgp$cov, function(node_sigma) {
                matrix(
                  as.numeric(unlist(node_sigma)),
                  nrow = length(node_sigma), byrow = TRUE
                )
              })
            )
          } else if (all(c("a", "b") %in% names(dgp))) {
            list(
              a = as.numeric(unlist(dgp$a)),
              b = as.numeric(unlist(dgp$b))
            )
          } else {
            list(
              mu = as.numeric(unlist(dgp$mu)),
              sigma = as.numeric(unlist(dgp$sigma))
            )
          }
        }),
        "n" = as.integer(unlist(ext[[k]])),
        "num_nodes" = as.integer(unlist(ext[[k]])),
        "n_inference" = as.integer(unlist(ext[[k]])),
        "method" = as.character(unlist(ext[[k]])),
        if (is.null(ext[[k]])) NA else ext[[k]]
      )
    }
  }

  # n_iterations must be set explicitly in a config file
  if (is.null(n_iterations)) {
    stop("n_iterations must be set in a config file.")
  }
  if (
    length(iteration_start) != 1 || is.logical(iteration_start) ||
      !is.numeric(iteration_start) || is.na(iteration_start) ||
      !is.finite(iteration_start) || iteration_start < 0 ||
      iteration_start != floor(iteration_start)
  ) {
    stop("iteration_start must be a nonnegative integer.")
  }
  iteration_start <- suppressWarnings(as.integer(iteration_start))
  if (is.na(iteration_start)) {
    stop("iteration_start must be a nonnegative integer.")
  }
  if (
    length(seed_strategy) != 1 || !is.character(seed_strategy) ||
      is.na(seed_strategy) ||
      !(seed_strategy %in% c("independent", "paired_methods"))
  ) {
    stop(
      "seed_strategy must be one of: independent, paired_methods."
    )
  }
  if (seed_strategy == "paired_methods" && is.null(seed)) {
    stop("seed must be set when seed_strategy is paired_methods.")
  }
  selected_n <- if ("selected_n" %in% names(selectors)) {
    .validate_selector(selectors[["selected_n"]], "selected_n")
  } else {
    NULL
  }
  selected_p <- if ("selected_p" %in% names(selectors)) {
    .validate_selector(selectors[["selected_p"]], "selected_p")
  } else {
    NULL
  }

  # h defaults to the identity transform when not set in a config file
  if (is.null(h_values)) {
    h_values <- list("identity")
  }

  # Create a grid of all possible configurations for scalar parameters
  scalar_keys <- setdiff(names(configuration_parameters), "DGP_parameters")
  scalar_grid <- do.call(
    expand.grid,
    c(configuration_parameters[scalar_keys], stringsAsFactors = FALSE)
  )

  # MCF always has 2 time steps, so drop duplicate rows
  n_time_steps_vary <- "n_time_steps" %in% names(scalar_grid) &&
    length(unique(scalar_grid$n_time_steps)) > 1
  if (n_time_steps_vary) {
    dedup_cols <- setdiff(names(scalar_grid), "n_time_steps")
    scalar_grid <- scalar_grid[
      scalar_grid$method != "MCF" |
        !duplicated(scalar_grid[, dedup_cols, drop = FALSE]),
    ]
  }

  # Create a grid of all possible configurations for all parameters
  base_configurations <- unlist(lapply(seq_len(nrow(scalar_grid)), function(i) {
    lapply(configuration_parameters$DGP_parameters, function(dgp) {
      config <- as.list(scalar_grid[i, ])
      config$DGP_parameters <- dgp
      config
    })
  }), recursive = FALSE)

  # Compute the DGP schema shared across this batch for CSV rectangularity
  dgp_schema <- .compute_dgp_schema(base_configurations)

  # Repeat each configuration for the specified number of iterations
  n_base <- length(base_configurations)
  scenario_keys <- vapply(
    base_configurations, .data_scenario_key, character(1)
  )
  scenario_indices <- match(scenario_keys, unique(scenario_keys))
  n_scenarios <- length(unique(scenario_keys))
  iterations <- iteration_start + seq_len(n_iterations) - 1
  configurations <- unlist(lapply(iterations, function(iteration) {
    lapply(seq_along(base_configurations), function(i) {
      config <- base_configurations[[i]]
      # Preserve the legacy full-configuration seed sequence
      config$seed <- if (!is.null(seed)) {
        seed + iteration * n_base + i - 1
      } else {
        NULL
      }
      if (seed_strategy == "paired_methods") {
        config$seed_strategy <- seed_strategy
        data_index <- scenario_indices[[i]]
        config$data_seed <- 2 * (
          seed + iteration * n_scenarios + data_index - 1
        )
        config$inference_seed <- 2 * config$seed + 1
      }
      config$timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      config$iteration_id <- iteration
      config$dgp_schema <- dgp_schema
      config$h_values <- h_values
      config
    })
  }), recursive = FALSE)

  # Select from the complete batch to preserve schema and seed indices
  if (!is.null(selected_n) || !is.null(selected_p)) {
    configurations <- Filter(function(config) {
      n_matches <- is.null(selected_n) || config$n == selected_n
      p <- if (config$parametric_form %in%
                 c("multivariate_Gaussian", "bivariate_Gaussian")) {
        length(config$DGP_parameters$mu[[1]])
      } else {
        1
      }
      n_matches && (is.null(selected_p) || p == selected_p)
    }, configurations)
  }
  if (length(configurations) == 0) {
    stop("Selected n/p excludes all configurations.")
  }

  configurations
}
