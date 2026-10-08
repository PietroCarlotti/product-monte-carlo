#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
python_bin="${repo_root}/.venv-pmc/bin/python"
output_dir="${repo_root}/data/results/higgs/oracle"
gpu_id="${HIGGS_ORACLE_GPU:-0}"

common_args=(
  --input "${repo_root}/data/processed/higgs/higgs_processed.csv"
  --output-dir "${output_dir}"
  --response-column outcome
  --metadata-columns partition,batch
  --feature-columns intercept,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,feature_19,feature_20,feature_21,feature_22,feature_23,feature_24,feature_25,feature_26,feature_27,feature_28
  --chains 4
  --warmup 5000
  --samples 10000
  --prior-scale 2.0
  --seed 91011
  --sequential-chains
  --dense-mass
  --map-initialization
  --map-max-iterations 500
)

for chain_index in 1 2 3 4; do
  printf 'Starting HIGGS Oracle production chain %s on GPU %s.\n' \
    "${chain_index}" "${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" JAX_ENABLE_X64=1 "${python_bin}" \
    -m src.applications.run_pooled_bayesian_logistic_regression \
    "${common_args[@]}" --chain-index "${chain_index}"
  printf 'Finished HIGGS Oracle production chain %s.\n' "${chain_index}"
done

printf 'Finalizing HIGGS Oracle production on CPU.\n'
JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu "${python_bin}" \
  -m src.applications.run_pooled_bayesian_logistic_regression \
  "${common_args[@]}" --finalize-only
printf 'HIGGS Oracle production completed.\n'
