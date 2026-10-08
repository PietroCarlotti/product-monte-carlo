#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
gbf_dir="${repo_root}/data/results/respiratory/gbf"
n_batches=$(find "${repo_root}/data/results/respiratory/batched" -mindepth 1 -maxdepth 1 -type d -name "batch_*" | wc -l)
trap 'rm -f "${gbf_dir}"/batch_*_samples.csv "${gbf_dir}/SMC_BLR_outfile.txt" "${gbf_dir}/bal_binary_GBF_BLR.txt" "${gbf_dir}/rho_j_BLR_progress.txt"' EXIT
N_BATCHES="${n_batches}" REPO_ROOT="${repo_root}" GBF_DIR="${gbf_dir}" "${repo_root}/.venv-pmc/bin/python" -c 'import os; from pathlib import Path; import numpy as np; root=Path(os.environ["REPO_ROOT"]); out=Path(os.environ["GBF_DIR"]); [np.savetxt(out/f"batch_{i}_samples.csv", np.load(root/f"data/results/respiratory/batched/batch_{i}/hmc_samples.npz")["beta"].reshape(-1, 5), delimiter=",", header=",".join(f"beta_{j}" for j in range(5)), comments="") for i in range(1, int(os.environ["N_BATCHES"])+1)]'
cd "${gbf_dir}"
node_files=("${gbf_dir}"/batch_*_samples.csv)
Rscript "${repo_root}/src/applications/run_respiratory_gbf.R" "${gbf_dir}/config.json" "${node_files[@]}"
REPO_ROOT="${repo_root}" "${repo_root}/.venv-pmc/bin/python" -c 'import csv, os; from pathlib import Path; import numpy as np; root=Path(os.environ["REPO_ROOT"])/"data/results/respiratory/gbf"; csv_path=root/"gbf_samples.csv"; samples=np.loadtxt(csv_path, delimiter=",", skiprows=1); np.savez_compressed(root/"gbf_samples.npz", samples=samples); csv_path.unlink(); estimates=(("identity", samples.mean(axis=0)), ("square", np.square(samples).mean(axis=0))); rows=[{"method":"GBF","target":target,"coefficient":f"beta_{i}","estimate":float(value)} for target, values in estimates for i, value in enumerate(values)]; f=(root/"results.csv").open("w", newline=""); writer=csv.DictWriter(f, fieldnames=("method","target","coefficient","estimate")); writer.writeheader(); writer.writerows(rows); f.close()'
