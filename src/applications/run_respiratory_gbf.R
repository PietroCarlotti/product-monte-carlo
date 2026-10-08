##############################
# Import necessary libraries #
##############################

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 2) stop("Usage: Rscript run_respiratory_gbf.R CONFIG_PATH BATCH_1_SAMPLES ... BATCH_K_SAMPLES")
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(normalizePath(script_path)), "respiratory_backend.R"))
source(file.path(dirname(normalizePath(script_path)), "blr_gbf_backend.R"))
config_path <- normalizePath(arguments[[1]], mustWork = TRUE)
output_dir <- dirname(config_path)

tryCatch(
  run_respiratory_backend(config_path, normalizePath(arguments[2:length(arguments)], mustWork = TRUE)),
  error = function(error) {
    message(conditionMessage(error))
    writeLines(conditionMessage(error), file.path(output_dir, "error.txt"))
    quit(status = 1)
  }
)
