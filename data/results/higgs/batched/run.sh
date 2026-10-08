#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
python_bin="${repo_root}/.venv-pmc/bin/python"
output_dir="${repo_root}/data/results/higgs/batched"
log_dir="${output_dir}/logs"
gpu_1="${HIGGS_BATCHED_GPU_1:-0}"
gpu_2="${HIGGS_BATCHED_GPU_2:-1}"

mkdir -p "${log_dir}"

common_args=(
  --input "${repo_root}/data/processed/higgs/higgs_processed.csv"
  --output-dir "${output_dir}"
  --batches 10
  --response-column outcome
  --metadata-columns partition,batch
  --feature-columns intercept,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,feature_19,feature_20,feature_21,feature_22,feature_23,feature_24,feature_25,feature_26,feature_27,feature_28
  --chains 4
  --warmup 5000
  --samples 10000
  --prior-scale 2.0
  --partition-column batch
  --seed 91011
  --sequential-chains
  --dense-mass
  --map-initialization
  --map-max-iterations 500
)

run_worker() {
  local worker_name="$1"
  local gpu_id="$2"
  local first_batch="$3"
  local last_batch="$4"
  local worker_log="${log_dir}/${worker_name}.log"

  printf '[%s] Starting batches %s-%s on GPU %s\n' \
    "${worker_name}" "${first_batch}" "${last_batch}" "${gpu_id}" | \
    tee -a "${worker_log}"
  for batch_index in $(seq "${first_batch}" "${last_batch}"); do
    for chain_index in 1 2 3 4; do
      printf '[%s] Starting batch %s chain %s\n' \
        "${worker_name}" "${batch_index}" "${chain_index}" | \
        tee -a "${worker_log}"
      if ! CUDA_VISIBLE_DEVICES="${gpu_id}" JAX_ENABLE_X64=1 \
        "${python_bin}" \
        -m src.applications.run_batched_bayesian_logistic_regression \
        "${common_args[@]}" --batch-index "${batch_index}" \
        --chain-index "${chain_index}" >>"${worker_log}" 2>&1; then
        printf '[%s] FAILED at batch %s chain %s\n' \
          "${worker_name}" "${batch_index}" "${chain_index}" | \
          tee -a "${worker_log}"
        return 1
      fi
      printf '[%s] Finished batch %s chain %s\n' \
        "${worker_name}" "${batch_index}" "${chain_index}" | \
        tee -a "${worker_log}"
    done
  done
  printf '[%s] Finished all assigned batches\n' "${worker_name}" | \
    tee -a "${worker_log}"
}

run_worker worker_1 "${gpu_1}" 1 5 &
worker_1_pid=$!
run_worker worker_2 "${gpu_2}" 6 10 &
worker_2_pid=$!

worker_1_status=0
worker_2_status=0
wait "${worker_1_pid}" || worker_1_status=$?
wait "${worker_2_pid}" || worker_2_status=$?

if ((worker_1_status != 0 || worker_2_status != 0)); then
  printf 'HIGGS batched sampling failed: worker_1=%s worker_2=%s\n' \
    "${worker_1_status}" "${worker_2_status}" >&2
  exit 1
fi

printf 'All sampling workers succeeded; finalizing batches on CPU.\n'
for batch_index in {1..10}; do
  printf 'Finalizing batch %s on CPU.\n' "${batch_index}"
  JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu "${python_bin}" \
    -m src.applications.run_batched_bayesian_logistic_regression \
    "${common_args[@]}" --batch-index "${batch_index}" --finalize-only
done

printf 'Aggregating completed batches on CPU.\n'
JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu "${python_bin}" \
  -m src.applications.run_batched_bayesian_logistic_regression \
  "${common_args[@]}" --aggregate-only
printf 'HIGGS batched production completed.\n'
