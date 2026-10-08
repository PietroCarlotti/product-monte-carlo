#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
batch_dir="${repo_root}/data/results/higgs/batched"
config_path="${repo_root}/data/results/higgs/pmc/config.json"
python_path="${repo_root}/.venv-pmc/bin/python"

declare -a batch_files=()
for batch in {1..10}; do
  batch_files+=("${batch_dir}/batch_${batch}/hmc_samples.npz")
done

mapfile -d '' discovered_files < <(
  find "${batch_dir}" -mindepth 2 -maxdepth 2 -type f \
    -path "${batch_dir}/batch_*/hmc_samples.npz" -print0 | sort -zV
)
if [[ "${#discovered_files[@]}" -ne "${#batch_files[@]}" ]]; then
  echo "Expected HIGGS posterior samples for exactly batches 1 through 10." >&2
  exit 1
fi
for index in "${!batch_files[@]}"; do
  if [[ "${discovered_files[index]}" != "${batch_files[index]}" ]]; then
    echo "Expected HIGGS posterior samples for exactly batches 1 through 10." >&2
    exit 1
  fi
done

"${python_path}" - "${config_path}" "${batch_files[@]}" <<'PYTHON'
import json
import sys
from pathlib import Path

import numpy as np

config_path = Path(sys.argv[1])
batch_paths = [Path(path) for path in sys.argv[2:]]
config = json.loads(config_path.read_text())
expected_inputs = [
    f"../batched/batch_{batch}/hmc_samples.npz" for batch in range(1, 11)
]
if config.get("run_status") != "production" or config.get("batch_inputs") != expected_inputs:
    raise ValueError("Invalid HIGGS PMC production configuration.")

expected_features = ["intercept", *[f"feature_{index}" for index in range(1, 29)]]
for batch, sample_path in enumerate(batch_paths, start=1):
    batch_output = sample_path.parent
    metadata = json.loads((batch_output / "config.json").read_text())
    status = json.loads((batch_output / "status.json").read_text())
    settings = metadata.get("sampler_settings", {})
    if (
        status != {"run_status": "completed", "phase": "finalized", "batch": batch}
        or metadata.get("feature_names") != expected_features
        or metadata.get("n_batches") != 10
        or metadata.get("batch") != batch
        or settings.get("chains") != 4
        or settings.get("warmup") != 5000
        or settings.get("post_warmup") != 10000
        or settings.get("dense_mass") is not True
        or settings.get("initialization") != "map"
        or settings.get("jax_enable_x64") is not True
    ):
        raise ValueError(f"Batch {batch} is not a finalized corrected production fit.")
    with np.load(sample_path) as archive:
        samples = archive["beta"]
    if samples.shape != (4, 10000, len(expected_features)):
        raise ValueError(f"Batch {batch} has invalid production dimensions.")
    if not np.isfinite(samples).all():
        raise ValueError(f"Batch {batch} contains non-finite production samples.")
PYTHON

"${python_path}" -m src.applications.run_respiratory_pmc "${config_path}"
