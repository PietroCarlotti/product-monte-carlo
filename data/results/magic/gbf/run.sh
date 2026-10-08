#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
gbf_dir="${repo_root}/data/results/magic/gbf"
mkdir -p "${gbf_dir}/nodes"
n_batches=$(find "${repo_root}/data/results/magic/batched" -mindepth 1 -maxdepth 1 -type d -name "batch_*" | wc -l)
trap 'rm -rf "${gbf_dir}/nodes"; rm -f "${gbf_dir}/magic_input.csv" "${gbf_dir}/SMC_BLR_outfile.txt" "${gbf_dir}/bal_binary_GBF_BLR.txt" "${gbf_dir}/rho_j_BLR_progress.txt"' EXIT
N_BATCHES="${n_batches}" REPO_ROOT="${repo_root}" "${repo_root}/.venv-pmc/bin/python" -c 'import os; from pathlib import Path; import pandas as pd; root=Path(os.environ["REPO_ROOT"]); source=pd.read_csv(root/"data/processed/magic/magic_processed.csv"); model_features=["intercept","fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist"]; data=source[model_features].copy(); data.columns=[f"feature_{i}" for i in range(11)]; data["outcome"]=source["class"]; data["batch"]=source["batch"]; data.to_csv(root/"data/results/magic/gbf/magic_input.csv", index=False)'
N_BATCHES="${n_batches}" REPO_ROOT="${repo_root}" "${repo_root}/.venv-pmc/bin/python" -c 'import numpy as np, pandas as pd, os; from pathlib import Path; root=Path(os.environ["REPO_ROOT"]); out=root/"data/results/magic/gbf/nodes"; [pd.DataFrame(np.load(root/f"data/results/magic/batched/batch_{i}/hmc_samples.npz")["beta"].reshape(-1,11), columns=[f"feature_{j}" for j in range(11)]).to_csv(out/f"batch_{i}_draws.csv", index=False) for i in range(1,int(os.environ["N_BATCHES"])+1)]'
cd "${gbf_dir}"
node_files=("${gbf_dir}"/nodes/batch_*_draws.csv)
Rscript "${repo_root}/src/applications/run_magic_gbf.R" "${gbf_dir}/config.json" "${node_files[@]}"
