repo_root="$(git rev-parse --show-toplevel)"
batch_dir="${repo_root}/data/results/respiratory/batched"
REPO_ROOT="${repo_root}" BATCH_DIR="${batch_dir}" "${repo_root}/.venv-pmc/bin/python" -c 'import json, os; from pathlib import Path; d=Path(os.environ["BATCH_DIR"]); p=d.parent/"pmc"/"config.json"; c=json.loads(p.read_text()); c["batch_inputs"]=[os.path.relpath(x, p.parent) for x in sorted(d.glob("batch_*/hmc_samples.npz"), key=lambda x: int(x.parent.name.split("_")[1]))]; p.write_text(json.dumps(c, indent=2)+"\n")'
"${repo_root}/.venv-pmc/bin/python" -m src.applications.run_respiratory_pmc "${repo_root}/data/results/respiratory/pmc/config.json"
