#!/bin/bash
set -euo pipefail

# Optionally run shared-cluster experiments at low scheduling priority.
# Priority changes are opt-in because some macOS/cluster environments reject
# them, and `nice` without a utility is not portable across shells.
if [ "${PMC_USE_NICE:-0}" = "1" ]; then
    export PMC_USE_NICE=0
    exec /usr/bin/nice -n 19 "$BASH" "$0" "$@"
fi

WORKING_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$WORKING_DIR/../.." && pwd)"
PYTHON_EXE="$REPO_DIR/.venv-pmc/bin/python3"
if [ ! -x "$PYTHON_EXE" ]; then
    PYTHON_EXE="$(command -v python3 || true)"
fi
if [ -z "$PYTHON_EXE" ] || [ ! -x "$PYTHON_EXE" ]; then
    echo "No usable Python interpreter found (checked $REPO_DIR/.venv-pmc/bin/python3 and PATH)." >&2
    exit 1
fi
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
SLACK_WEBHOOK="${PMC_SLACK_WEBHOOK_URL:-}"

# Keep numerical libraries from oversubscribing the experiment workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export FUSION_RESUME_EXISTING="${FUSION_RESUME_EXISTING:-1}"

FUSION_PROCESS_GROUP=""
FUSION_WATCHDOG_PID=""
FUSION_TIMEOUT_MARKER=""
FUSION_KILL_GRACE_SECONDS=30

notify_slack() {
    local status="$1"
    [ -n "$SLACK_WEBHOOK" ] || return 0
    local message="launcher.sh completed with status $status: $*"
    curl -sS -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"$message\"}" \
        "$SLACK_WEBHOOK" >/dev/null 2>&1 || true
}

# Remove only the temporary method configs created for a unified launch.
cleanup_unified_configs() {
    for name in shared pmc_reference pmc_linear pmc_mlp pmc_oracle \
        pmc_gp fusion_empty fusion; do
        unlink "$UNIFIED_TMP_DIR/$name.json" 2>/dev/null || true
    done
    rmdir "$UNIFIED_TMP_DIR" 2>/dev/null || true
}

if [ "${1:-}" = "--config" ]; then
    GP_ONLY=0
    UNIFIED_DRY_RUN=0
    if [ "$#" -eq 3 ] && [ "$3" = "--gp-only" ]; then
        GP_ONLY=1
    elif [ "$#" -eq 3 ] && [ "$3" = "--dry-run" ]; then
        UNIFIED_DRY_RUN=1
    elif [ "$#" -eq 4 ] && [ "$3" = "--gp-only" ] && \
        [ "$4" = "--dry-run" ]; then
        GP_ONLY=1
        UNIFIED_DRY_RUN=1
    elif [ "$#" -ne 2 ]; then
        echo "Usage: $0 --config ALL_METHODS.json [--gp-only] [--dry-run]" >&2
        exit 2
    fi

    export PYTHONDONTWRITEBYTECODE=1
    UNIFIED_TMP_DIR=""
    if [ "$UNIFIED_DRY_RUN" -eq 0 ]; then
        UNIFIED_TMP_DIR="$(mktemp -d /tmp/pmc-launcher.XXXXXXXX)"
        trap cleanup_unified_configs EXIT
    fi

    if ! unified_metadata="$("$PYTHON_EXE" - "$2" "$UNIFIED_TMP_DIR" \
        "$GP_ONLY" <<'PY_CONFIG'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
temp_dir = sys.argv[2]
gp_only = sys.argv[3] == "1"
try:
    config = json.loads(source.read_text())
    if not isinstance(config, dict):
        raise ValueError("the top level must be an object")
    for key in ("shared", "pmc", "fusion"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"{key} must be an object")
    sample_sizes = config["shared"].get("n")
    if (not isinstance(sample_sizes, list) or not sample_sizes or
            any(type(size) is not int or size <= 0 for size in sample_sizes)):
        raise ValueError("shared.n must be a nonempty list of positive integers")
    for key in ("pmc", "fusion"):
        methods = config[key].get("method")
        if not isinstance(methods, list) or any(
                not isinstance(method, str) for method in methods):
            raise ValueError(f"{key}.method must be a list of strings")
    algorithms = config["pmc"].get("algorithm")
    if not isinstance(algorithms, list) or any(
            not isinstance(algorithm, str) for algorithm in algorithms):
        raise ValueError("pmc.algorithm must be a list of strings")
    pmc_methods = config["pmc"]["method"]
    has_gp = "GP" in pmc_methods
    has_linear = "linear" in pmc_methods
    has_oracle = "Oracle" in algorithms
    if gp_only and not has_gp:
        raise ValueError("--gp-only requires GP in pmc.method")
    if has_gp:
        unsupported_methods = sorted(set(pmc_methods) - {"linear", "MLP", "GP"})
        unsupported_algorithms = sorted(set(algorithms) - {"PMC", "Oracle"})
        if unsupported_methods or unsupported_algorithms:
            raise ValueError(
                "unified paired-GP staging supports only PMC methods "
                "linear, MLP, GP and algorithms PMC, Oracle; got unsupported "
                f"methods={unsupported_methods}, algorithms={unsupported_algorithms}"
            )
    supported_gp_forms = {
        "univariate_Gaussian", "bivariate_Gaussian", "multivariate_Gaussian",
        "Beta",
    }
    if has_gp and (
            "MLP" not in config["pmc"]["method"] or
            "PMC" not in algorithms or
            config["shared"].get("parametric_form") not in supported_gp_forms or
            type(config["shared"].get("num_nodes")) is not int or
            config["shared"].get("num_nodes") <= 0):
        raise ValueError(
            "paired GP requires MLP and a positive-node supported PMC form "
            "(univariate_Gaussian, bivariate_Gaussian, "
            "multivariate_Gaussian, or Beta)"
        )
    filename = config.get("results_filename")
    if (not isinstance(filename, str) or not filename.endswith(".csv") or
            Path(filename).name != filename or
            any(char in filename for char in "\r\n")):
        raise ValueError("results_filename must be one CSV basename")
    output_dir = config.get("output_dir")
    if (not isinstance(output_dir, str) or not output_dir.strip() or
            any(char in output_dir for char in "\r\n")):
        raise ValueError("output_dir must be one directory path")
    if output_dir.count("{n}") > 1:
        raise ValueError("output_dir may contain at most one {n} placeholder")
    exact_n_text = os.environ.get("PMC_EXACT_N", "").strip()
    if "{n}" in output_dir:
        if not exact_n_text.isdigit():
            raise ValueError(
                "output_dir uses {n}; set PMC_EXACT_N to one configured sample size"
            )
        exact_n = int(exact_n_text)
        if exact_n not in sample_sizes:
            raise ValueError("PMC_EXACT_N is not present in shared.n")
        output_dir = output_dir.replace("{n}", str(exact_n))
        selected_sizes = [exact_n]
    else:
        selected_sizes = sample_sizes
    output_path = (source.parent / output_dir).resolve()
    if exact_n_text and "{n}" not in config["output_dir"]:
        exact_n = int(exact_n_text)
        if exact_n not in sample_sizes:
            raise ValueError("PMC_EXACT_N is not present in shared.n")
        selected_sizes = [exact_n]
    if "{n}" in config["output_dir"] and not output_path.is_dir():
        raise ValueError(f"canonical output directory is missing: {output_path}")
    if temp_dir:
        method_configs = {
            "shared": config["shared"],
            "pmc_reference": {
                **config["pmc"],
                "method": [
                    method for method in config["pmc"]["method"]
                    if method != "GP"
                ],
            },
            "pmc_linear": {
                **config["pmc"],
                "algorithm": ["PMC"],
                "method": ["linear"],
            },
            "pmc_mlp": {
                **config["pmc"],
                "algorithm": ["PMC"],
                "method": ["MLP"],
            },
            "pmc_oracle": {
                **config["pmc"],
                "algorithm": ["Oracle"],
                "method": ["linear"],
            },
            "pmc_gp": {
                **config["pmc"],
                "algorithm": ["PMC"],
                "method": ["GP"],
            },
            "fusion_empty": {"method": []},
            "fusion": config["fusion"],
        }
        for key, value in method_configs.items():
            destination = Path(temp_dir) / f"{key}.json"
            destination.write_text(json.dumps(value, indent=2) + "\n")
    print(filename)
    print(output_path)
    print(", ".join(str(size) for size in selected_sizes))
    print("1" if has_gp else "0")
    print("1" if has_linear else "0")
    print("1" if has_oracle else "0")
except (OSError, ValueError, KeyError) as exc:
    raise SystemExit(f"Invalid unified config: {exc}") from exc
PY_CONFIG
)"; then
        exit 2
    fi
    unified_fields=()
    while IFS= read -r field; do
        unified_fields+=("$field")
    done <<< "$unified_metadata"
    RESULTS_FILENAME="${unified_fields[0]}"
    OUTPUT_DIR="${unified_fields[1]}"
    if [ "$UNIFIED_DRY_RUN" -eq 1 ]; then
        echo "Unified config is valid: $2"
        echo "One run would cover n=${unified_fields[2]} in one CSV:"
        echo "  $OUTPUT_DIR/$RESULTS_FILENAME"
        if [ "$GP_ONLY" -eq 1 ]; then
            echo "Stages: paired-seed PMC GP only."
        elif [ "${unified_fields[3]}" = "1" ]; then
            stage_description="Stages:"
            if [ "${unified_fields[4]}" = "1" ]; then
                stage_description="$stage_description PMC linear,"
            fi
            stage_description="$stage_description PMC MLP,"
            if [ "${unified_fields[5]}" = "1" ]; then
                stage_description="$stage_description Oracle once,"
            fi
            echo "$stage_description paired-seed PMC GP, then Fusion."
        else
            echo "Stages: PMC methods, then Fusion."
        fi
        echo "Existing strict resume checks apply to each stage."
        echo "No jobs launched."
        exit 0
    fi

    if [ "$GP_ONLY" -eq 1 ]; then
        (
            export PMC_PAIRED_SEED_METHOD=MLP
            unset PMC_PAIRED_SEED_DRY_RUN PMC_PAIRED_SEED_REQUIRE_COMPLETE
            PMC_USE_NICE=0 "$BASH" "$0" "$RESULTS_FILENAME" \
                "$UNIFIED_TMP_DIR/shared.json" \
                "$UNIFIED_TMP_DIR/pmc_gp.json" \
                "$UNIFIED_TMP_DIR/fusion_empty.json" "$OUTPUT_DIR"
        )
    elif [ "${unified_fields[3]}" = "1" ]; then
        if [ "${unified_fields[4]}" = "1" ]; then
            (
                unset PMC_PAIRED_SEED_METHOD PMC_PAIRED_SEED_DRY_RUN
                unset PMC_PAIRED_SEED_REQUIRE_COMPLETE
                PMC_USE_NICE=0 "$BASH" "$0" "$RESULTS_FILENAME" \
                    "$UNIFIED_TMP_DIR/shared.json" \
                    "$UNIFIED_TMP_DIR/pmc_linear.json" \
                    "$UNIFIED_TMP_DIR/fusion_empty.json" "$OUTPUT_DIR"
            )
        fi
        (
            unset PMC_PAIRED_SEED_METHOD PMC_PAIRED_SEED_DRY_RUN
            unset PMC_PAIRED_SEED_REQUIRE_COMPLETE
            PMC_USE_NICE=0 "$BASH" "$0" "$RESULTS_FILENAME" \
                "$UNIFIED_TMP_DIR/shared.json" \
                "$UNIFIED_TMP_DIR/pmc_mlp.json" \
                "$UNIFIED_TMP_DIR/fusion_empty.json" "$OUTPUT_DIR"
        )
        if [ "${unified_fields[5]}" = "1" ]; then
            (
                unset PMC_PAIRED_SEED_METHOD PMC_PAIRED_SEED_DRY_RUN
                unset PMC_PAIRED_SEED_REQUIRE_COMPLETE
                PMC_USE_NICE=0 "$BASH" "$0" "$RESULTS_FILENAME" \
                    "$UNIFIED_TMP_DIR/shared.json" \
                    "$UNIFIED_TMP_DIR/pmc_oracle.json" \
                    "$UNIFIED_TMP_DIR/fusion_empty.json" "$OUTPUT_DIR"
            )
        fi
        (
            export PMC_PAIRED_SEED_METHOD=MLP
            unset PMC_PAIRED_SEED_DRY_RUN PMC_PAIRED_SEED_REQUIRE_COMPLETE
            PMC_USE_NICE=0 "$BASH" "$0" "$RESULTS_FILENAME" \
                "$UNIFIED_TMP_DIR/shared.json" \
                "$UNIFIED_TMP_DIR/pmc_gp.json" \
                "$UNIFIED_TMP_DIR/fusion.json" "$OUTPUT_DIR"
        )
    else
        PMC_USE_NICE=0 "$BASH" "$0" "$RESULTS_FILENAME" \
            "$UNIFIED_TMP_DIR/shared.json" \
            "$UNIFIED_TMP_DIR/pmc_reference.json" \
            "$UNIFIED_TMP_DIR/fusion.json" "$OUTPUT_DIR"
    fi
    exit $?
fi

if [ "$#" -ge 2 ] && [ "$1" = "--spec" ]; then
    is_dry_run=0
    [ "${3:-}" = "--dry-run" ] && is_dry_run=1
    set +e
    "$PYTHON_EXE" "$WORKING_DIR/launcher.py" "$@"
    status=$?
    set -e
    [ "$is_dry_run" -eq 1 ] || notify_slack "$status" "spec=$2"
    exit "$status"
fi

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 --spec RUN_SPEC.json [--dry-run]" >&2
    echo "   or: $0 --config ALL_METHODS.json [--gp-only] [--dry-run]" >&2
    echo "   or: $0 RESULTS_FILENAME SHARED_CONFIG PMC_CONFIG FUSION_CONFIG OUTPUT_DIR" >&2
    exit 2
fi

RESULTS_FILENAME="$1"
SHARED_CONFIG="$2"
PMC_CONFIG="$3"
FUSION_CONFIG="$4"
OUTPUT_DIR="$5"

# Require explicit CUDA visibility whenever the PMC config includes MLP.
if ! "$PYTHON_EXE" - "$PMC_CONFIG" <<'PY_CHECK'
import json
import os
import sys

config = json.load(open(sys.argv[1]))
if "MLP" in config.get("method", []) and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    raise SystemExit("PMC-MLP requires CUDA_VISIBLE_DEVICES; refusing CPU fallback.")
PY_CHECK
then
    status=$?
    notify_slack "$status" "results=$RESULTS_FILENAME"
    exit "$status"
fi

mkdir -p "$OUTPUT_DIR"
LOCK_DIR="$OUTPUT_DIR/.${RESULTS_FILENAME}.launcher_lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Launcher output is already locked: $OUTPUT_DIR/$RESULTS_FILENAME" >&2
    exit 2
fi

# List descendants and child process groups owned by an isolated fusion stage.
fusion_process_ids() {
    local root_pid="$1"
    ps -eo pid=,ppid=,pgid= | awk -v root="$root_pid" '
        {
            pids[NR] = $1
            parents[$1] = $2
            groups[$1] = $3
        }
        END {
            selected[root] = 1
            owned_groups[root] = 1
            changed = 1
            while (changed) {
                changed = 0
                for (row_index = 1; row_index <= NR; row_index++) {
                    pid = pids[row_index]
                    if (!selected[pid] &&
                        (selected[parents[pid]] || owned_groups[groups[pid]])) {
                        selected[pid] = 1
                        owned_groups[groups[pid]] = 1
                        changed = 1
                    }
                }
            }
            for (row_index = 1; row_index <= NR; row_index++) {
                pid = pids[row_index]
                if (selected[pid] && pid != root) {
                    print pid
                }
            }
            print root
        }
    '
}

# Terminate the isolated fusion tree, including detached child process groups.
terminate_fusion_process_tree() {
    local root_pid="${1:-}"
    local grace_seconds="${2:-${FUSION_KILL_GRACE_SECONDS:-30}}"
    local process_ids=()
    local process_pid
    [ -n "$root_pid" ] || return 0
    while IFS= read -r process_pid; do
        process_ids+=("$process_pid")
    done < <(fusion_process_ids "$root_pid")
    for process_pid in "${process_ids[@]}"; do
        kill -TERM "$process_pid" 2>/dev/null || true
    done
    local deadline=$((SECONDS + grace_seconds))
    local processes_alive
    while [ "$SECONDS" -lt "$deadline" ]; do
        processes_alive=0
        for process_pid in "${process_ids[@]}"; do
            if kill -0 "$process_pid" 2>/dev/null; then
                processes_alive=1
                break
            fi
        done
        [ "$processes_alive" -eq 1 ] || break
        sleep 1
    done
    for process_pid in "${process_ids[@]}"; do
        if kill -0 "$process_pid" 2>/dev/null; then
            kill -KILL "$process_pid" 2>/dev/null || true
        fi
    done
}

# Clean up launcher-owned processes and the per-output lock.
cleanup_launcher() {
    local watchdog_pid="${FUSION_WATCHDOG_PID:-}"
    local process_group="${FUSION_PROCESS_GROUP:-}"
    local timeout_marker="${FUSION_TIMEOUT_MARKER:-}"
    local grace_seconds="${FUSION_KILL_GRACE_SECONDS:-30}"
    if [ -n "$watchdog_pid" ]; then
        kill "$watchdog_pid" 2>/dev/null || true
        wait "$watchdog_pid" 2>/dev/null || true
    fi
    terminate_fusion_process_tree \
        "$process_group" "$grace_seconds"
    [ -z "$timeout_marker" ] || \
        unlink "$timeout_marker" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_launcher EXIT
RUN_LOG="$OUTPUT_DIR/run.log"
echo "Launcher started at $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$RUN_LOG"

# Run fusion in its own process group and stop every descendant at the deadline.
run_fusion_with_timeout() {
    local timeout_seconds="$1"
    local grace_seconds="$2"
    local timeout_marker="$LOCK_DIR/fusion_timeout"
    local process_group
    local watchdog_pid
    local command_status
    shift 2

    FUSION_TIMEOUT_MARKER="$timeout_marker"
    unlink "$timeout_marker" 2>/dev/null || true
    "$PYTHON_EXE" -c '
import os
import sys

os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
' "$@" >> "$RUN_LOG" 2>&1 &
    process_group=$!
    FUSION_PROCESS_GROUP="$process_group"

    (
        sleep "$timeout_seconds"
        : > "$timeout_marker"
        echo "R fusion exceeded ${timeout_seconds}s; terminating its process tree." \
            >> "$RUN_LOG"
        terminate_fusion_process_tree "$process_group" "$grace_seconds"
    ) &
    watchdog_pid=$!
    FUSION_WATCHDOG_PID="$watchdog_pid"

    wait "$process_group"
    command_status=$?
    if [ -e "$timeout_marker" ]; then
        wait "$watchdog_pid" 2>/dev/null || true
    else
        kill "$watchdog_pid" 2>/dev/null || true
        wait "$watchdog_pid" 2>/dev/null || true
    fi
    FUSION_WATCHDOG_PID=""

    terminate_fusion_process_tree "$process_group" "$grace_seconds"
    FUSION_PROCESS_GROUP=""
    if [ -e "$timeout_marker" ]; then
        unlink "$timeout_marker"
        FUSION_TIMEOUT_MARKER=""
        return 124
    fi
    FUSION_TIMEOUT_MARKER=""
    return "$command_status"
}

set +e

# Run PMC configs when present; fusion-only batches skip the Python stage.
if "$PYTHON_EXE" - "$PMC_CONFIG" <<'PY_METHOD_CHECK'
import json
import sys
config = json.load(open(sys.argv[1]))
raise SystemExit(0 if config.get("method") else 1)
PY_METHOD_CHECK
then
    "$PYTHON_EXE" "$WORKING_DIR/launcher.py" \
        "$SHARED_CONFIG" "$PMC_CONFIG" "$OUTPUT_DIR" "$RESULTS_FILENAME" \
        >> "$RUN_LOG" 2>&1
    status=$?
else
    echo "No PMC methods requested; skipping Python stage." >> "$RUN_LOG"
    status=0
fi

# Run R fusion only after Python succeeds and allow its worker cluster to cool down.
if [ "$status" -eq 0 ] && "$PYTHON_EXE" - "$FUSION_CONFIG" <<'PY_FUSION'
import json
import sys
config = json.load(open(sys.argv[1]))
raise SystemExit(0 if config.get("method") else 1)
PY_FUSION
then
    echo "Cooling down for 60 seconds before R fusion." >> "$RUN_LOG"
    sleep 60
    FUSION_TIMEOUT_SECONDS="${PMC_FUSION_TIMEOUT_SECONDS:-43200}"
    FUSION_KILL_GRACE_SECONDS="${PMC_FUSION_KILL_GRACE_SECONDS:-30}"
    case "$FUSION_TIMEOUT_SECONDS" in
        ""|*[!0-9]*|0)
            echo "PMC_FUSION_TIMEOUT_SECONDS must be a positive integer." \
                >> "$RUN_LOG"
            status=2
            ;;
    esac
    case "$FUSION_KILL_GRACE_SECONDS" in
        ""|*[!0-9]*|0)
            echo "PMC_FUSION_KILL_GRACE_SECONDS must be a positive integer." \
                >> "$RUN_LOG"
            status=2
            ;;
    esac
    echo "R fusion timeout: ${FUSION_TIMEOUT_SECONDS}s." >> "$RUN_LOG"
    if [ "$status" -eq 0 ]; then
        run_fusion_with_timeout \
            "$FUSION_TIMEOUT_SECONDS" "$FUSION_KILL_GRACE_SECONDS" \
            Rscript "$WORKING_DIR/launcher.R" "$SHARED_CONFIG" \
            "$FUSION_CONFIG" "$OUTPUT_DIR" "$RESULTS_FILENAME"
        status=$?
    fi
fi
set -e
notify_slack "$status" "results=$RESULTS_FILENAME"
echo "Launcher finished at $(date '+%Y-%m-%d %H:%M:%S %Z') with status $status" >> "$RUN_LOG"
exit "$status"
