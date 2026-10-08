#!/bin/bash
set -euo pipefail

# Run the combined calibration experiment.
WORKING_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$WORKING_DIR/../../.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-$REPO_DIR/.venv-pmc/bin/python3}"
if [ ! -x "$PYTHON_EXE" ]; then
    PYTHON_EXE="$(command -v python3)"
fi
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
CONFIG_PATH="${1:-$WORKING_DIR/calibration.json}"

"$PYTHON_EXE" "$WORKING_DIR/calibration.py" "$CONFIG_PATH"
