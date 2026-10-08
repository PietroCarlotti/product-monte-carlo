#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
batch_dir="${repo_root}/data/results/higgs/batched"
gbf_dir="${repo_root}/data/results/higgs/gbf"
input_path="${repo_root}/data/processed/higgs/higgs_processed.csv"
python_path="${repo_root}/.venv-pmc/bin/python"
if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [CONFIG_PATH]" >&2
  exit 1
fi
config_path="${1:-${gbf_dir}/config.json}"
config_path="$(realpath "${config_path}")"
output_dir="$("${python_path}" - "${config_path}" <<'PYTHON'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
print((config_path.parent / config["output_dir"]).resolve())
PYTHON
)"
node_dir="$(mktemp -d)"
mkdir -p "${output_dir}"
trap 'rm -rf "${node_dir}"; rm -f "${output_dir}/SMC_BLR_outfile.txt" \
  "${output_dir}/bal_binary_GBF_BLR.txt" \
  "${output_dir}/rho_j_BLR_progress.txt"' EXIT

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

"${python_path}" - "${input_path}" "${node_dir}" "${batch_files[@]}" <<'PYTHON'
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

input_path = Path(sys.argv[1])
node_dir = Path(sys.argv[2])
batch_paths = [Path(path) for path in sys.argv[3:]]
with input_path.open(newline="") as input_file:
    header = next(csv.reader(input_file))
native_indices = sorted(
    int(match.group(1))
    for name in header
    if (match := re.fullmatch(r"feature_([0-9]+)", name))
)
if native_indices != list(range(1, len(native_indices) + 1)):
    raise ValueError("Processed HIGGS input has invalid feature columns.")
feature_names = ["intercept", *[f"feature_{index}" for index in native_indices]]
if "intercept" not in header or len(feature_names) != 29:
    raise ValueError("Processed HIGGS input must define 29 ordered coefficients.")

reference_shape = None
for batch, sample_path in enumerate(batch_paths, start=1):
    batch_output = sample_path.parent
    metadata = json.loads((batch_output / "config.json").read_text())
    status = json.loads((batch_output / "status.json").read_text())
    settings = metadata.get("sampler_settings", {})
    if (
        status != {"run_status": "completed", "phase": "finalized", "batch": batch}
        or metadata.get("feature_names") != feature_names
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
    if samples.ndim != 3 or samples.shape[-1] != len(feature_names):
        raise ValueError(f"Batch {batch} has invalid coefficient dimensions.")
    if samples.shape[:2] != (4, 10000) or not np.isfinite(samples).all():
        raise ValueError(f"Batch {batch} has invalid production samples.")
    if reference_shape is None:
        reference_shape = samples.shape
    elif samples.shape != reference_shape:
        raise ValueError("HIGGS posterior batches must have equal sample shapes.")
    draws = samples.reshape(-1, samples.shape[-1])
    np.savetxt(
        node_dir / f"batch_{batch}_draws.csv",
        draws,
        delimiter=",",
        header=",".join(feature_names),
        comments="",
    )
PYTHON

cd "${output_dir}"
Rscript "${repo_root}/src/applications/run_higgs_gbf.R" \
  "${config_path}" "${node_dir}"

"${python_path}" - "${output_dir}" <<'PYTHON'
import csv
import sys
from pathlib import Path

import numpy as np

output_dir = Path(sys.argv[1])
csv_path = output_dir / "gbf_samples.csv"
samples = np.loadtxt(csv_path, delimiter=",", skiprows=1)
if samples.ndim == 1:
    samples = samples.reshape(1, -1)
if samples.ndim != 2 or samples.shape[1] != 29:
    raise ValueError("GBF returned invalid HIGGS sample dimensions.")
if not np.isfinite(samples).all():
    raise ValueError("GBF returned non-finite HIGGS samples.")
np.savez_compressed(output_dir / "gbf_samples.npz", samples=samples)
estimates = (
    ("identity", samples.mean(axis=0)),
    ("square", np.square(samples).mean(axis=0)),
)
rows = [
    {
        "method": "GBF",
        "target": target,
        "coefficient": f"beta_{index}",
        "estimate": float(value),
    }
    for target, values in estimates
    for index, value in enumerate(values)
]
with (output_dir / "results.csv").open("w", newline="") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=("method", "target", "coefficient", "estimate"),
    )
    writer.writeheader()
    writer.writerows(rows)
csv_path.unlink()
PYTHON
