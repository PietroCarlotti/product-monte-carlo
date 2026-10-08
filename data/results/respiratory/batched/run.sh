#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
input="${repo_root}/data/processed/respiratory/respiratory_processed.csv"
output="${repo_root}/data/results/respiratory/batched"
"${repo_root}/.venv-pmc/bin/python" -m src.applications.run_batched_bayesian_logistic_regression --input "${input}" --output-dir "${output}" --response-column outcome --metadata-columns partition,batch,center,id --feature-columns intercept,treat,sex,age,baseline --chains 4 --warmup 1000 --samples 10000 --prior-scale 2.0 --partition-column batch --seed 91011 --sequential-chains
