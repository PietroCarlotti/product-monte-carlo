#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
"${repo_root}/.venv-pmc/bin/python" -m src.applications.run_pooled_bayesian_logistic_regression --input "${repo_root}/data/processed/magic/magic_processed.csv" --output-dir "${repo_root}/data/results/magic/oracle" --response-column class --metadata-columns partition,batch --feature-columns intercept,fLength,fWidth,fSize,fConc,fConc1,fAsym,fM3Long,fM3Trans,fAlpha,fDist --chains 4 --warmup 1000 --samples 10000 --prior-scale 2.0 --seed 91011 --sequential-chains --dense-mass

# Runtime design: chains are sequential; batch-level parallelism is handled by the orchestration layer.
