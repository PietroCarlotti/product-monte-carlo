##############################
# Import necessary libraries #
##############################

# Standard libraries
import importlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import current_process, get_context
from pathlib import Path
from typing import Any


# Report whether a config can use CUDA-backed simulation
def _is_gpu_config(config):
    if config.get('algorithm') != 'PMC':
        return False
    return config.get('method') == 'MLP' or (
        config.get('method') == 'GP' and config.get('gp_device', 'cpu') == 'cuda'
    )


# Reject explicit PMC-GP CUDA work when no validated GPU is available
def _validate_gpu_scheduling(gpu_configurations, n_gpu):
    requires_cuda = any(
        config.get('algorithm') == 'PMC'
        and config.get('method') == 'GP'
        and config.get('gp_device', 'cpu') == 'cuda'
        for config in gpu_configurations
    )
    if requires_cuda and n_gpu == 0:
        raise RuntimeError(
            "PMC-GP gp_device='cuda' requires at least one requested and "
            "available CUDA GPU."
        )


# Parse the requested CUDA device list from the runner environment
def _requested_cuda_devices():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if devices == "":
        return []
    return [device.strip() for device in devices.split(",") if device.strip()]


# Set conservative thread caps before numerical libraries initialize
def _configure_thread_environment():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# Report whether deterministic CUDA behavior was requested
def _deterministic_cuda_enabled():
    return os.environ.get("PMC_DETERMINISTIC_CUDA") == "1"


# Set deterministic CUDA environment variables before Torch initializes CUDA
def _configure_deterministic_cuda_environment():
    if not _deterministic_cuda_enabled():
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# Enable deterministic Torch CUDA settings when requested
def _configure_torch_determinism(torch_module):
    if not _deterministic_cuda_enabled():
        return
    torch_module.use_deterministic_algorithms(True)
    has_cudnn_backend = hasattr(torch_module, "backends") and hasattr(
        torch_module.backends, "cudnn"
    )
    if has_cudnn_backend:
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False


# Validate that requested CUDA devices are visible to PyTorch
def _validate_cuda_request(torch_module, requested_devices):
    if not requested_devices:
        return 0
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch reports that CUDA is unavailable."
        )
    available_gpu = torch_module.cuda.device_count()
    if available_gpu != len(requested_devices):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES requested "
            f"{len(requested_devices)} device(s), but PyTorch sees "
            f"{available_gpu} device(s)."
        )
    return available_gpu


# Parse the requested number of GPU workers per visible device
def _gpu_workers_per_device():
    workers_value = os.environ.get("PMC_GPU_WORKERS_PER_DEVICE", "1")
    try:
        workers_per_device = int(workers_value)
    except ValueError:
        raise ValueError(
            "PMC_GPU_WORKERS_PER_DEVICE must be a positive integer, "
            f"got {workers_value!r}."
        ) from None
    if workers_per_device <= 0:
        raise ValueError(
            "PMC_GPU_WORKERS_PER_DEVICE must be a positive integer, "
            f"got {workers_value!r}."
        )
    return workers_per_device


# Parse an optional exact sample size from the runner environment
def _exact_sample_size():
    value = os.environ.get("PMC_EXACT_N", "")
    if value == "":
        return None
    try:
        sample_size = int(value)
    except ValueError:
        sample_size = 0
    if sample_size <= 0 or str(sample_size) != value:
        raise ValueError(
            f"PMC_EXACT_N must be a positive integer, got {value!r}."
        )
    return sample_size


# Parse a strict binary environment flag
def _binary_env(name):
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}.")
    return value == "1"

# Normalize one recovery identity scalar to the planner representation
def _recovery_value(value):
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

# Build one exact seven-field recovery identity
def _recovery_identity(record, label, require_fields=False):
    fields = (
        "seed",
        "iteration_id",
        "parametric_form",
        "num_nodes",
        "n",
        "algorithm",
        "method",
    )
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    if require_fields:
        missing_fields = [field for field in fields if field not in record]
        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise ValueError(
                f"{label} is missing identity fields: {missing_text}"
            )
    return tuple(_recovery_value(record.get(field)) for field in fields)

# Validate target counts and return normalized missing targets
def _recovery_targets(config, label):
    rows_per_h = config.get("expected_rows_per_h")
    if (
        isinstance(rows_per_h, bool)
        or not isinstance(rows_per_h, int)
        or rows_per_h <= 0
    ):
        raise ValueError(f"{label}.expected_rows_per_h must be a positive integer")
    h_values = config.get("h_values")
    if not isinstance(h_values, list) or not h_values:
        raise ValueError(f"{label}.h_values must be a nonempty list")
    if any(value is not None and not isinstance(value, str) for value in h_values):
        raise ValueError(f"{label}.h_values must contain strings or null")
    if len(h_values) != len(set(h_values)):
        raise ValueError(f"{label}.h_values must be distinct")
    observed = config.get("observed_rows_by_h")
    if not isinstance(observed, list) or len(observed) != len(h_values):
        raise ValueError(f"{label}.observed_rows_by_h must match h_values")
    counts = {}
    for index, target in enumerate(observed):
        target_label = f"{label}.observed_rows_by_h[{index}]"
        if not isinstance(target, dict) or "h" not in target:
            raise ValueError(f"{target_label} must be an object with h")
        h_value = target["h"]
        if h_value is not None and not isinstance(h_value, str):
            raise ValueError(f"{target_label}.h must be a string or null")
        count = target.get("observed_row_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"{target_label}.observed_row_count must be a nonnegative integer"
            )
        if h_value in counts:
            raise ValueError(f"{label}.observed_rows_by_h has duplicate h values")
        counts[h_value] = count
    if set(counts) != set(h_values):
        raise ValueError(f"{label}.observed_rows_by_h must match h_values")
    ordered_counts = [counts[h_value] for h_value in h_values]
    status = config.get("status")
    if status == "surplus":
        raise ValueError("PMC recovery plan contains a surplus configuration")
    if any(count not in {0, rows_per_h} for count in ordered_counts):
        raise ValueError(
            f"{label} contains an unsafe within-target partial count"
        )
    missing_targets = [
        h_value
        for h_value, count in zip(h_values, ordered_counts, strict=True)
        if count == 0
    ]
    if status == "complete" and missing_targets:
        raise ValueError(f"{label} complete status contradicts target counts")
    if status == "missing" and len(missing_targets) != len(h_values):
        raise ValueError(f"{label} missing status contradicts target counts")
    if status == "partial" and (
        not missing_targets or len(missing_targets) == len(h_values)
    ):
        raise ValueError(f"{label} partial status contradicts target counts")
    return missing_targets

# Filter native PMC configs to exact missing recovery identities
def _filter_recovery_configs(configurations, plan_path):
    resolved_path = Path(plan_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise ValueError(f"PMC recovery plan does not exist: {resolved_path}")
    try:
        plan = json.loads(resolved_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read PMC recovery plan {resolved_path}: {error}"
        ) from error
    if not isinstance(plan, dict):
        raise ValueError("PMC recovery plan root must be an object")
    if (
        isinstance(plan.get("schema_version"), bool)
        or plan.get("schema_version") != 1
    ):
        raise ValueError("PMC recovery plan must have schema_version 1")
    result = plan.get("result")
    if not isinstance(result, dict):
        raise ValueError("PMC recovery plan result must be an object")
    result_state = result.get("state")
    if result_state not in {"valid", "missing", "empty"}:
        raise ValueError(
            "PMC recovery result state must be valid, missing, or empty"
        )
    outside = plan.get("outside_requested_design")
    if not isinstance(outside, list) or outside:
        raise ValueError(
            "PMC recovery plan contains rows outside the requested design"
        )
    summary = plan.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("PMC recovery plan summary must be an object")
    for field in (
        "outside_requested_design_identity_count",
        "outside_requested_design_row_count",
    ):
        if summary.get(field) != 0:
            raise ValueError(
                "PMC recovery plan summary reports rows outside the design"
            )
    plan_configs = plan.get("configurations")
    if not isinstance(plan_configs, list) or not plan_configs:
        raise ValueError("PMC recovery plan contains no configurations")
    requested = []
    requested_keys = set()
    plan_keys = set()
    statuses = []
    for index, config in enumerate(plan_configs):
        label = f"configuration {index}"
        if not isinstance(config, dict):
            raise ValueError(f"{label} must be an object")
        stage = config.get("stage")
        status = config.get("status")
        if stage not in {"pmc", "fusion"}:
            raise ValueError(f"{label} has an invalid stage")
        if status not in {"complete", "missing", "partial", "surplus"}:
            raise ValueError(f"{label} has an invalid status")
        statuses.append(status)
        identity = _recovery_identity(
            config,
            label,
            require_fields=True,
        )
        if identity in plan_keys:
            raise ValueError(
                "PMC recovery plan contains duplicate configuration identities"
            )
        plan_keys.add(identity)
        missing_targets = _recovery_targets(config, label)
        if stage == "pmc" and status in {"missing", "partial"}:
            if identity in requested_keys:
                raise ValueError(
                    "PMC recovery plan contains duplicate requested identities"
                )
            requested_keys.add(identity)
            requested.append((identity, config, missing_targets))
    if result_state in {"missing", "empty"} and any(
        status != "missing" for status in statuses
    ):
        raise ValueError(
            "Missing or empty PMC recovery plans require every "
            "configuration to be missing"
        )
    if not requested:
        raise ValueError(
            "PMC recovery plan contains no missing PMC configurations"
        )
    config_keys = [
        _recovery_identity(config, f"expanded PMC configuration {index}")
        for index, config in enumerate(configurations)
    ]
    if len(config_keys) != len(set(config_keys)):
        raise ValueError(
            "Expanded PMC configurations contain duplicate identities"
        )
    config_by_key = dict(zip(config_keys, configurations, strict=True))
    selected = []
    missing_target_count = 0
    for identity, plan_config, missing_targets in requested:
        if identity not in config_by_key:
            raise ValueError(
                "PMC recovery plan does not exactly match expanded configurations"
            )
        native_config = config_by_key[identity]
        native_h_values = native_config.get("h_values")
        if not isinstance(native_h_values, list) or not native_h_values:
            raise ValueError("Expanded PMC configuration has invalid h_values")
        native_normalized = [_recovery_value(value) for value in native_h_values]
        if (
            len(native_normalized) != len(set(native_normalized))
            or native_normalized != plan_config["h_values"]
        ):
            raise ValueError(
                "PMC recovery plan h_values do not match expanded configuration"
            )
        missing_set = set(missing_targets)
        selected_config = native_config.copy()
        selected_config["h_values"] = [
            value
            for value, normalized in zip(
                native_h_values, native_normalized, strict=True
            )
            if normalized in missing_set
        ]
        if len(selected_config["h_values"]) != len(missing_targets):
            raise ValueError("PMC recovery target matching is not one-to-one")
        selected.append(selected_config)
        missing_target_count += len(missing_targets)
    selected_keys = [
        _recovery_identity(config, f"selected PMC configuration {index}")
        for index, config in enumerate(selected)
    ]
    if selected_keys != [identity for identity, _, _ in requested]:
        raise ValueError("PMC recovery identity matching is not one-to-one")
    print(
        f"PMC recovery selected {len(selected)} recoverable "
        f"configuration(s) with {missing_target_count} missing h target(s) "
        f"from {resolved_path}."
    )
    return selected

# Initialize a CPU worker before importing Torch-backed simulation code
def _initialize_cpu_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _configure_thread_environment()
    _configure_deterministic_cuda_environment()


# Initialize a GPU worker on a selected CUDA device before Torch import
def _initialize_gpu_worker(requested_devices):
    worker_index = (current_process()._identity[0] - 1) % len(requested_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = requested_devices[worker_index]
    _configure_thread_environment()
    _configure_deterministic_cuda_environment()


# Run one simulation config after the worker device environment is fixed
def _run_config(config, plots_directory):
    import torch

    _configure_torch_determinism(torch)

    from pmc.simulations import simulate_config

    return simulate_config(config, plots_directory)


# Submit configs to the requested executor and track their execution class
def _submit_configs(executor, configs, plots_directory, label):
    return {
        executor.submit(
            _run_config,
            config,
            plots_directory if config.get('save_plots') else None,
        ): {'worker_type': label, 'config': config}
        for config in configs
    }


# Return the dimension count expected in each config's per-target output rows
def _config_dimension(config):
    if config['parametric_form'] in ('bivariate_Gaussian', 'multivariate_Gaussian'):
        return len(config['DGP_parameters']['mu'][0])
    return 1


# Return the result-row count for one target under a configuration
def _target_result_rows(config, h_value):
    if config.get('selected_component') is not None or h_value == 'normalized_sum':
        return 1
    return _config_dimension(config)


# Normalize scalar CSV/config values for strict resume comparisons
def _resume_value(value):
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# Base fields identify one scheduled configuration independently of DGP values
_RESUME_IDENTITY_FIELDS = (
    'seed', 'iteration_id', 'parametric_form', 'num_nodes', 'n',
    'algorithm', 'method'
)


# Return the scheduler-level identity used for stage filtering
def _resume_base_identity(value):
    return tuple(
        _resume_value(value.get(key))
        for key in _RESUME_IDENTITY_FIELDS
    )


# Return semantic DGP and sampling-weight values in result-column order
def _resume_semantic_values(value, schema):
    simulations = importlib.import_module('pmc.simulations')
    if simulations._is_ultra_compact_result_schema(schema):
        if 'DGP_parameters' in value:
            fingerprint = simulations._semantic_fingerprint(
                value['parametric_form'],
                value['DGP_parameters'],
                value.get('w', []),
            )
        else:
            fingerprint = value.get('semantic_fingerprint')
        return (_resume_value(fingerprint),)

    columns = [
        *simulations._dgp_param_columns(schema),
        *simulations._weight_columns(schema),
    ]
    if 'DGP_parameters' in value:
        semantic_values = {
            **simulations._dgp_param_values(
                schema,
                value['parametric_form'],
                value['DGP_parameters'],
            ),
            **simulations._weight_values(schema, value.get('w', [])),
        }
    else:
        semantic_values = value
    return tuple(
        _resume_value(semantic_values.get(column))
        for column in columns
    )


# Return the semantic identity used to skip already-complete rows
def _resume_identity_from_config(config, schema=None):
    if schema is None:
        schema = config.get('dgp_schema')
    if schema is None:
        raise ValueError('Resume identity requires a DGP schema.')
    return (
        _resume_base_identity(config)
        + _resume_semantic_values(config, schema)
    )


# Return the number of result rows a completed config should contribute
def _expected_result_rows(config):
    return sum(
        _target_result_rows(config, h_value)
        for h_value in config['h_values']
    )


# Identify a GP configuration and its MLP reference by shared design values
def _paired_seed_identity(value, schema):
    fields = (
        'iteration_id', 'parametric_form', 'num_nodes', 'n',
        'sampling_method', 'discrepancy_method', 'preprocessing', 'n_inference',
    )
    return (
        tuple(_resume_value(value.get(field)) for field in fields)
        + _resume_semantic_values(value, schema)
    )


# Replace GP seeds with complete matching MLP reference seeds
def _pair_gp_seeds_with_mlp(configurations, results_path, pd, result_columns):
    supported_gp_forms = {
        'univariate_Gaussian', 'bivariate_Gaussian', 'multivariate_Gaussian',
        'Beta',
    }
    if not configurations or any(
        config.get('algorithm') != 'PMC'
        or config.get('method') != 'GP'
        or config.get('parametric_form') not in supported_gp_forms
        or type(config.get('num_nodes')) is not int
        or config.get('num_nodes') <= 0
        for config in configurations
    ):
        raise ValueError(
            'PMC_PAIRED_SEED_METHOD=MLP requires positive-node supported '
            'PMC-GP configurations (univariate_Gaussian, bivariate_Gaussian, '
            'multivariate_Gaussian, or Beta) only.'
        )
    if not os.path.isfile(results_path) or os.path.getsize(results_path) == 0:
        raise ValueError(
            f'Cannot pair GP seeds: MLP reference output is missing: {results_path}'
        )

    existing = pd.read_csv(results_path, float_precision='round_trip')
    schema = configurations[0]['dgp_schema']
    if list(existing.columns) != result_columns(schema):
        raise ValueError(
            f'Cannot pair GP seeds in {results_path}: result schema mismatch.'
        )

    expected = {}
    for config in configurations:
        identity = _paired_seed_identity(config, schema)
        if identity in expected:
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: duplicate GP design.'
            )
        targets = [_resume_value(value) for value in config['h_values']]
        if not targets or len(targets) != len(set(targets)):
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: invalid GP targets.'
            )
        expected[identity] = {
            target: _target_result_rows(config, target)
            for target in targets
        }

    observed = {}
    for _, row in existing.iterrows():
        if row.get('algorithm') != 'PMC' or row.get('method') != 'MLP':
            continue
        identity = _paired_seed_identity(row, schema)
        if identity not in expected:
            continue
        target = _resume_value(row.get('h'))
        if target not in expected[identity]:
            continue
        seed_text = _resume_value(row.get('seed'))
        sample_text = _resume_value(row.get('n_sample'))
        if (
            seed_text is None
            or not seed_text.lstrip('-').isdigit()
            or sample_text is None
            or not sample_text.isdigit()
            or int(sample_text) <= 0
        ):
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: invalid MLP seed '
                'or sample count.'
            )
        source = (int(seed_text), int(sample_text))
        reference = observed.setdefault(identity, {})
        target_reference = reference.get(target)
        if target_reference is None:
            reference[target] = {'source': source, 'count': 1}
            continue
        if target_reference['source'] != source:
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: inconsistent MLP '
                'target rows.'
            )
        target_reference['count'] += 1
        if target_reference['count'] > expected[identity][target]:
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: excessive MLP '
                'target rows.'
            )

    paired_seeds = set()
    for config in configurations:
        identity = _paired_seed_identity(config, schema)
        reference = observed.get(identity, {})
        if (
            set(reference) != set(expected[identity])
            or any(
                reference[target]['count'] != required_count
                for target, required_count in expected[identity].items()
            )
        ):
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: incomplete MLP '
                'reference targets.'
            )
        source_values = {
            target_reference['source']
            for target_reference in reference.values()
        }
        if len(source_values) != 1:
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: inconsistent MLP '
                'reference targets.'
            )
        config['seed'] = next(iter(source_values))[0]
        if config['seed'] in paired_seeds:
            raise ValueError(
                f'Cannot pair GP seeds in {results_path}: MLP seed reused '
                'across GP configurations.'
            )
        paired_seeds.add(config['seed'])

# Return config identities that are already complete in an existing output file
def _completed_resume_identities(
    results_path,
    configurations,
    pd,
    result_columns=None,
    allow_target_partial=False,
    completed_targets=None,
):
    if not os.path.exists(results_path):
        return set()
    if os.path.getsize(results_path) == 0:
        return set()

    existing = pd.read_csv(results_path, float_precision='round_trip')
    required_columns = {
        "seed",
        "iteration_id",
        "parametric_form",
        "num_nodes",
        "n",
        "algorithm",
        "method",
        "h",
    }
    missing_columns = sorted(required_columns - set(existing.columns))
    if missing_columns:
        raise ValueError(
            f"Cannot resume {results_path}: missing identity columns {missing_columns}"
        )
    if result_columns is not None:
        expected_columns = result_columns(configurations[0]["dgp_schema"])
        if list(existing.columns) != expected_columns:
            raise ValueError(
                f"Cannot resume {results_path}: result schema mismatch."
            )

    schema = configurations[0]["dgp_schema"]
    expected_targets = {}
    for config in configurations:
        identity = _resume_identity_from_config(config)
        if identity in expected_targets:
            raise ValueError(
                f"Cannot resume {results_path}: requested configs have duplicate "
                "resume identities."
            )
        h_values = [_resume_value(value) for value in config["h_values"]]
        if not h_values or len(h_values) != len(set(h_values)):
            raise ValueError(
                f"Cannot resume {results_path}: requested config has invalid h_values."
            )
        expected_targets[identity] = {
            h_value: _target_result_rows(config, h_value)
            for h_value in h_values
        }
    expected_signatures = {
        _resume_base_identity(config)[-2:]
        for config in configurations
    }
    observed_counts = {}
    for _, row in existing.iterrows():
        identity = _resume_identity_from_config(row, schema)
        if _resume_base_identity(row)[-2:] not in expected_signatures:
            continue
        if identity not in expected_targets:
            raise ValueError(
                f"Cannot resume {results_path}: found a row outside the requested design."
            )
        h_value = _resume_value(row.get("h"))
        if h_value not in expected_targets[identity]:
            raise ValueError(
                f"Cannot resume {results_path}: found a row outside the requested design."
            )
        target = (identity, h_value)
        observed_counts[target] = observed_counts.get(target, 0) + 1

    completed = set()
    if completed_targets is None:
        completed_targets = set()
    for identity, targets in expected_targets.items():
        counts = {
            h_value: observed_counts.get((identity, h_value), 0)
            for h_value in targets
        }
        if any(count > targets[h_value] for h_value, count in counts.items()):
            raise ValueError(
                f"Cannot resume {results_path}: found duplicate target output rows."
            )
        if any(
            count not in {0, targets[h_value]}
            for h_value, count in counts.items()
        ):
            raise ValueError(
                f"Cannot resume {results_path}: found a partial target output."
            )
        completed_targets.update(
            (identity, h_value)
            for h_value, count in counts.items()
            if count == targets[h_value]
        )
        if all(count == targets[h_value] for h_value, count in counts.items()):
            completed.add(identity)
        elif any(count > 0 for count in counts.values()) and not allow_target_partial:
            raise ValueError(
                f"Cannot resume {results_path}: found a partial config output."
            )
    return completed


# Replace a launcher-reserved empty output file with the first result rows
def _replace_reserved_output(df, path):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            csv_text = df.to_csv(index=False, header=True)
            temporary_file.write(csv_text.encode("utf-8"))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


# Append result rows while honoring an empty file reserved by the launcher
def _append_result_rows(df, results_path, csv_state, append_to_csv):
    if df.empty:
        return
    if (
        not csv_state["initialized"]
        and os.path.exists(results_path)
        and os.path.getsize(results_path) == 0
    ):
        _replace_reserved_output(df, results_path)
        csv_state["initialized"] = True
        csv_state["mode"] = "a"
        csv_state["header"] = False
        csv_state["columns"] = list(df.columns)
        return
    append_to_csv(df, results_path, csv_state)


# Build the failure-log path paired with the requested results CSV
def _failure_results_path(results_path):
    root, extension = os.path.splitext(results_path)
    return f"{root}_failures{extension or '.csv'}"


# Serialize config values for compact failure diagnostics
def _failure_value(value):
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return value


# Build a structured failure row for a failed worker config
def _failure_row(metadata, error, traceback_text):
    config = metadata['config']
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'worker_type': metadata['worker_type'],
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback_text,
        'seed': config.get('seed'),
        'iteration_id': config.get('iteration_id'),
        'parametric_form': config.get('parametric_form'),
        'num_nodes': config.get('num_nodes'),
        'n': config.get('n'),
        'algorithm': config.get('algorithm'),
        'method': config.get('method'),
        'sampling_method': config.get('sampling_method'),
        'discrepancy_method': config.get('discrepancy_method'),
        'h_values': _failure_value(config.get('h_values')),
    }


###############
# Entry point #
###############


# Config-driven manifest orchestration lives here, not in the shell wrapper.
"""Config-driven experiment run specifications.

This module is deliberately independent of the existing launchers.  It
provides the common validation and deterministic batch expansion layer that
will later be consumed by ``launcher.sh``.
"""

class RunSpecError(ValueError):
    """Raised when a run specification is invalid."""


@dataclass(frozen=True)
class Batch:
    scenario: str
    sample_size: int
    iteration_start: int
    iteration_stop: int
    output_dir: Path
    results_filename: str


def load(path: str | Path) -> dict[str, Any]:
    """Load and validate one run specification."""
    source = Path(path)
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"cannot read run specification {source}: {exc}") from exc
    validate(value, source)
    return value


def validate(spec: dict[str, Any], source: str | Path = "run specification") -> None:
    """Validate the stable, launcher-independent portion of a run spec."""
    if not isinstance(spec, dict):
        raise RunSpecError(f"{source} must contain a JSON object")
    required = {"schema_version", "run_id", "grid", "configs", "batches", "output"}
    missing = sorted(required.difference(spec))
    if missing:
        raise RunSpecError(f"{source} is missing: {', '.join(missing)}")
    if spec["schema_version"] != 1:
        raise RunSpecError("schema_version must be 1")
    if not isinstance(spec["run_id"], str) or not spec["run_id"].strip():
        raise RunSpecError("run_id must be a non-empty string")
    grid = spec["grid"]
    for key in ("scenarios", "sample_sizes", "targets", "methods"):
        if not isinstance(grid.get(key), list) or not grid[key]:
            raise RunSpecError(f"grid.{key} must be a non-empty list")
    if not isinstance(grid.get("iterations"), int) or grid["iterations"] <= 0:
        raise RunSpecError("grid.iterations must be a positive integer")
    if any(not isinstance(x, str) or not x for x in grid["scenarios"]):
        raise RunSpecError("grid.scenarios must contain non-empty strings")
    if any(Path(x).name != x or x in {".", ".."} for x in grid["scenarios"]):
        raise RunSpecError("grid.scenarios must contain plain directory names")
    if any(not isinstance(x, int) or x <= 0 for x in grid["sample_sizes"]):
        raise RunSpecError("grid.sample_sizes must contain positive integers")
    for key in ("scenarios", "sample_sizes", "targets", "methods"):
        if len(set(grid[key])) != len(grid[key]):
            raise RunSpecError(f"grid.{key} must not contain duplicates")
    batches = spec["batches"]
    if not isinstance(batches.get("chunk_size"), int) or batches["chunk_size"] <= 0:
        raise RunSpecError("batches.chunk_size must be a positive integer")
    configs = spec["configs"]
    for key in ("shared", "pmc", "fusion"):
        if not isinstance(configs.get(key), str) or not configs[key]:
            raise RunSpecError(f"configs.{key} must be a non-empty path")
    output = spec["output"]
    if not isinstance(output.get("root"), str) or not output["root"]:
        raise RunSpecError("output.root must be a non-empty path")
    filename = output.get("results_filename")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".csv")
    ):
        raise RunSpecError("output.results_filename must be a plain .csv filename")
    resume = spec.get("resume", {})
    if not isinstance(resume, dict) or resume.get("policy", "strict") not in {"strict", "fresh"}:
        raise RunSpecError("resume.policy must be strict or fresh")
    if not isinstance(resume.get("allow_partial", True), bool):
        raise RunSpecError("resume.allow_partial must be boolean")
    resources = spec.get("resources", {})
    if not isinstance(resources, dict):
        raise RunSpecError("resources must be an object")
    for key in ("workers", "cooldown_seconds"):
        if key in resources and (not isinstance(resources[key], int) or resources[key] < 0):
            raise RunSpecError(f"resources.{key} must be a non-negative integer")
    if (
        "cuda_devices" in resources
        and resources["cuda_devices"] is not None
        and not isinstance(resources["cuda_devices"], str)
    ):
        raise RunSpecError("resources.cuda_devices must be a string or null")


def expand(spec: dict[str, Any]) -> tuple[Batch, ...]:
    """Expand scenarios, sample sizes, and iteration chunks deterministically."""
    validate(spec)
    grid, batching, output = spec["grid"], spec["batches"], spec["output"]
    start = batching.get("start", 0)
    stop = batching.get("end", grid["iterations"])
    if (
        not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= grid["iterations"]
    ):
        raise RunSpecError("batches.start/end must define a range within grid.iterations")
    size = batching["chunk_size"]
    return tuple(
        Batch(
            scenario,
            sample_size,
            offset,
            min(offset + size, stop),
            Path(output["root"]),
            output["results_filename"],
        )
        for scenario in grid["scenarios"]
        for sample_size in grid["sample_sizes"]
        for offset in range(start, stop, size)
    )


def batch_label(batch: Batch) -> str:
    """Return the stable inclusive iteration label used in output names."""
    return f"{batch.iteration_start:03d}_{batch.iteration_stop - 1:03d}"


def batch_results_path(spec: dict[str, Any], batch: Batch) -> Path:
    """Return the canonical result path for one planned batch."""
    root = Path(spec["output"]["root"])
    return root / batch.scenario / f"n_{batch.sample_size}" / f"batch_{batch_label(batch)}.csv"


def batch_shared_config_path(spec: dict[str, Any], batch: Batch) -> Path:
    """Return the materialized shared-config path for one planned batch."""
    root = Path(spec["output"]["root"])
    return root / batch.scenario / "configs" / f"shared_configs_batch_{batch_label(batch)}.json"


def batch_method_config_path(spec: dict[str, Any], batch: Batch, kind: str) -> Path:
    """Return the materialized method-config path for one planned batch."""
    root = Path(spec["output"]["root"])
    return root / batch.scenario / "configs" / f"{kind}_batch_{batch_label(batch)}.json"


def execute(spec_path: str | Path, *, runner: str | Path) -> None:
    """Execute every planned batch through the existing five-argument runner."""
    source = Path(spec_path).resolve()
    spec = load(source)
    base = source.parent
    spec = dict(spec)
    spec["configs"] = {
        key: str((base / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in spec["configs"].items()
    }
    spec["output"] = dict(spec["output"])
    output_root = Path(spec["output"]["root"])
    spec["output"]["root"] = (
        str((base / output_root).resolve()) if not output_root.is_absolute() else str(output_root)
    )
    runner_path = Path(runner).resolve()
    for batch in expand(spec):
        result_path = batch_results_path(spec, batch)
        shared_path = batch_shared_config_path(spec, batch)
        if result_path.exists() and spec.get("resume", {}).get("policy", "strict") == "fresh":
            raise RunSpecError(f"fresh run collides with existing output: {result_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        materialize_shared_config(spec, batch, shared_path)
        pmc_path = batch_method_config_path(spec, batch, "pmc")
        fusion_path = batch_method_config_path(spec, batch, "fusion")
        materialize_method_config(spec, "pmc", pmc_path)
        materialize_method_config(spec, "fusion", fusion_path)
        command = (["bash"] if runner_path.suffix == ".sh" else []) + [
            str(runner_path),
            result_path.name,
            str(shared_path),
            str(pmc_path),
            str(fusion_path),
            str(result_path.parent),
        ]
        environment = dict(os.environ)
        environment["PMC_RESUME"] = (
            "1" if spec.get("resume", {}).get("policy", "strict") == "strict" else "0"
        )
        completed = subprocess.run(command, check=False, env=environment)
        if completed.returncode:
            raise RunSpecError(
                f"batch {batch_label(batch)} failed with exit code {completed.returncode}"
            )


def materialize_shared_config(spec: dict[str, Any], batch: Batch, destination: str | Path) -> Path:
    """Write one batch-scoped shared config without mutating the source."""
    source = Path(spec["configs"]["shared"])
    if not source.is_file():
        raise RunSpecError(f"shared config does not exist: {source}")
    try:
        config = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"cannot read shared config {source}: {exc}") from exc
    if not isinstance(config, dict):
        raise RunSpecError("shared config must contain a JSON object")
    config["n"] = [batch.sample_size]
    config["n_iterations"] = batch.iteration_stop - batch.iteration_start
    config["iteration_start"] = batch.iteration_start
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            if json.loads(target.read_text()) != config:
                raise RunSpecError(f"existing batch config conflicts: {target}")
        except json.JSONDecodeError as exc:
            raise RunSpecError(f"existing batch config is invalid: {target}") from exc
        return target
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


def materialize_method_config(spec: dict[str, Any], kind: str, destination: str | Path) -> Path:
    """Materialize a method config selected by the manifest."""
    source = Path(spec["configs"][kind])
    try:
        config = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"cannot read {kind} config {source}: {exc}") from exc
    if not isinstance(config, dict):
        raise RunSpecError(f"{kind} config must contain a JSON object")
    if kind == "pmc" and isinstance(config.get("method"), list):
        selected = set(spec["grid"]["methods"])
        original_methods = config["method"]
        config["method"] = [method for method in original_methods if method in selected]
        if original_methods and not config["method"]:
            raise RunSpecError("grid.methods selects no methods in the PMC config")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except json.JSONDecodeError as exc:
            raise RunSpecError(f"existing method config is invalid: {target}") from exc
        if existing != config:
            raise RunSpecError(f"existing method config conflicts: {target}")
        return target
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


IS_SPEC_MODE = sys.argv[1:2] == ["--spec"]

if __name__ == "__main__" and IS_SPEC_MODE:
    spec_path = sys.argv[2]
    if len(sys.argv) == 4 and sys.argv[3] == "--dry-run":
        sys.argv = [sys.argv[0], spec_path, str(Path(__file__).with_name("launcher.sh")), "1"]
    elif len(sys.argv) == 3:
        sys.argv = [sys.argv[0], spec_path, str(Path(__file__).with_name("launcher.sh")), "0"]
    else:
        raise RunSpecError("usage: launcher.py --spec RUN_SPEC.json [--dry-run]")
    source = Path(sys.argv[1]).resolve()
    spec = load(source)
    if sys.argv[3] == "1":
        base = source.parent
        # Resolve one run-spec path relative to its source
        def resolve(value):
            path = Path(value)
            return (base / path).resolve() if not path.is_absolute() else path

        for key in ("shared", "pmc", "fusion"):
            config_path = resolve(spec["configs"][key])
            if not config_path.is_file():
                raise RunSpecError(f"config file does not exist: {config_path}")
        output_root = resolve(spec["output"]["root"])
        if not output_root.is_dir() or not os.access(output_root, os.W_OK):
            raise RunSpecError(f"output directory must exist and be writable: {output_root}")
        print(f"Run specification is valid: {source}")
        batches = expand(spec)
        print(f"Planned batches: {len(batches)}")
        for batch in batches:
            print(
                f"  {batch.scenario} n={batch.sample_size} "
                f"iterations={batch.iteration_start}:{batch.iteration_stop} "
                f"results={batch_results_path(spec, batch)} "
                f"shared={batch_shared_config_path(spec, batch)}"
            )
    else:
        execute(spec_path, runner=Path(__file__).with_name("launcher.sh"))

if __name__ == '__main__' and not IS_SPEC_MODE:

    _configure_thread_environment()
    _configure_deterministic_cuda_environment()

    #############################
    # Set the working directory #
    #############################

    # Default working directory (one level up from this script, matching pre-restructure behavior)
    working_directory = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )

    # Optional output directory passed as argument
    _out_args = [a for a in sys.argv[1:] if not a.endswith('.json')]

    # Override the output directory if one was provided
    if _out_args:
        working_directory = os.path.normpath(os.path.abspath(_out_args[0]))
    os.chdir(working_directory)

    # Override the results filename if one was provided
    results_filename = _out_args[1] if len(_out_args) >= 2 else 'results.csv'

    ##################
    # Configurations #
    ##################

    # Third-party libraries
    pd = importlib.import_module('pandas')

    # PMC library
    build_configs = importlib.import_module('pmc.config').build_configs
    result_columns = importlib.import_module('pmc.simulations').result_columns

    # Get the list of config files
    _config_files = [a for a in sys.argv[1:] if a.endswith('.json')]

    # Check if any config files were provided
    if not _config_files:
        print("Error: no config files provided. Pass one or more JSON files.")
        sys.exit(1)

    configurations = build_configs(_config_files)
    exact_n = _exact_sample_size()
    if exact_n is not None:
        configurations = [
            config for config in configurations if config.get("n") == exact_n
        ]
        if not configurations:
            raise ValueError(f"PMC_EXACT_N={exact_n} removed all configurations.")
    paired_seed_method = os.environ.get('PMC_PAIRED_SEED_METHOD', '')
    paired_seed_dry_run = _binary_env('PMC_PAIRED_SEED_DRY_RUN')
    paired_seed_require_complete = _binary_env('PMC_PAIRED_SEED_REQUIRE_COMPLETE')
    if paired_seed_method not in {'', 'MLP'}:
        raise ValueError(
            'PMC_PAIRED_SEED_METHOD must be MLP when provided.'
        )
    if paired_seed_dry_run and paired_seed_method != 'MLP':
        raise ValueError(
            'PMC_PAIRED_SEED_DRY_RUN requires PMC_PAIRED_SEED_METHOD=MLP.'
        )
    if paired_seed_require_complete and not paired_seed_dry_run:
        raise ValueError(
            'PMC_PAIRED_SEED_REQUIRE_COMPLETE requires '
            'PMC_PAIRED_SEED_DRY_RUN=1.'
        )
    if paired_seed_method == 'MLP':
        _pair_gp_seeds_with_mlp(
            configurations,
            results_filename,
            pd,
            result_columns,
        )
    resume_configurations = configurations

    # Restrict recovery execution to exact missing PMC configurations
    recovery_plan_path = os.environ.get("PMC_RECOVERY_PLAN", "")
    recovery_dry_run = _binary_env("PMC_RECOVERY_DRY_RUN")
    if recovery_dry_run and not recovery_plan_path:
        raise ValueError("PMC_RECOVERY_DRY_RUN requires PMC_RECOVERY_PLAN.")
    if paired_seed_method and recovery_plan_path:
        raise ValueError(
            'PMC_PAIRED_SEED_METHOD cannot be combined with PMC_RECOVERY_PLAN.'
        )
    if recovery_plan_path:
        configurations = _filter_recovery_configs(
            configurations,
            recovery_plan_path,
        )

    completed_targets = set()
    completed_identities = _completed_resume_identities(
        results_filename,
        resume_configurations,
        pd,
        result_columns,
        allow_target_partial=bool(recovery_plan_path),
        completed_targets=completed_targets,
    )
    selected_completed = {
        _resume_identity_from_config(config)
        for config in configurations
        if _resume_identity_from_config(config) in completed_identities
    }
    remaining_configurations = []
    for config in configurations:
        identity = _resume_identity_from_config(config)
        remaining_h = [
            h_value
            for h_value in config["h_values"]
            if (identity, _resume_value(h_value)) not in completed_targets
        ]
        if remaining_h:
            selected_config = config.copy()
            selected_config["h_values"] = remaining_h
            remaining_configurations.append(selected_config)
    configurations = remaining_configurations
    if selected_completed:
        print(f"Skipping {len(selected_completed)} already-complete config(s).")

    # Exit before CUDA validation or worker setup for recovery selection checks
    if recovery_dry_run:
        print(
            f"PMC recovery dry run selected {len(configurations)} "
            "configuration(s) after resume filtering."
        )
        sys.exit(0)

    if paired_seed_dry_run:
        print(
            f'PMC paired-seed dry run selected {len(configurations)} '
            'configuration(s) after strict resume filtering.'
        )
        if paired_seed_require_complete and configurations:
            raise RuntimeError(
                f'{len(configurations)} paired PMC-GP configuration(s) '
                'remain incomplete.'
            )
        sys.exit(0)
    if not configurations:
        print('\nAll requested simulations were already complete in', results_filename)
        sys.exit(0)

    ##################
    # Hardware setup #
    ##################

    # Third-party libraries
    torch = importlib.import_module('torch')
    tqdm = importlib.import_module('tqdm').tqdm
    _configure_torch_determinism(torch)

    # PMC library
    append_to_csv = importlib.import_module('pmc.utils').append_to_csv

    requested_devices = _requested_cuda_devices()
    n_gpu = _validate_cuda_request(torch, requested_devices)
    gpu_workers_per_device = _gpu_workers_per_device()

    # Set the maximum number of CPUs from the environment when provided
    max_cpu_value = os.environ.get('PMC_MAX_CPU', '13')
    try:
        max_cpu = int(max_cpu_value)
    except ValueError:
        raise ValueError(
            f"PMC_MAX_CPU must be a positive integer, got {max_cpu_value!r}."
        ) from None
    if max_cpu <= 0:
        raise ValueError(
            f"PMC_MAX_CPU must be a positive integer, got {max_cpu_value!r}."
        )

    # Check the number of available CPUs
    available_cpu = os.cpu_count()

    # Determine the number of CPUs to use
    n_cpu = min(max_cpu, available_cpu) if available_cpu is not None else max_cpu

    # Set the number of threads for PyTorch to 1 to avoid oversubscription in parallel processing
    torch.set_num_threads(1)

    gpu_configurations = [config for config in configurations if _is_gpu_config(config)]
    cpu_configurations = [config for config in configurations if not _is_gpu_config(config)]
    _validate_gpu_scheduling(gpu_configurations, n_gpu)
    if gpu_configurations and n_gpu == 0:
        print("Running PMC-MLP simulations on CPU because CUDA was not requested.")

    ###############
    # Simulations #
    ###############

    # Results storage
    results = []
    results_path = results_filename
    failure_results_path = _failure_results_path(results_path)
    csv_state = {'initialized': False, 'mode': 'w', 'header': True}
    failure_csv_state = {'initialized': False, 'mode': 'w', 'header': True}
    plots_directory = os.path.join(working_directory, 'plots')
    future_labels = {}
    executors = []
    failures = []

    if cpu_configurations or n_gpu == 0:
        cpu_pool_configs = cpu_configurations + (gpu_configurations if n_gpu == 0 else [])
        if cpu_pool_configs:
            cpu_executor = ProcessPoolExecutor(
                max_workers=min(n_cpu, len(cpu_pool_configs)),
                mp_context=get_context("spawn"),
                initializer=_initialize_cpu_worker,
            )
            executors.append(cpu_executor)
            future_labels.update(
                _submit_configs(cpu_executor, cpu_pool_configs, plots_directory, "CPU")
            )

    if gpu_configurations and n_gpu > 0:
        max_gpu_workers = min(
            n_gpu * gpu_workers_per_device,
            len(gpu_configurations),
        )
        gpu_executor = ProcessPoolExecutor(
            max_workers=max_gpu_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_gpu_worker,
            initargs=(requested_devices,),
        )
        executors.append(gpu_executor)
        future_labels.update(
            _submit_configs(gpu_executor, gpu_configurations, plots_directory, "GPU")
        )

    try:
        for future in tqdm(
            as_completed(future_labels),
            total=len(future_labels),
            desc='Processing Simulations',
        ):
            try:
                # Get the result from the completed future
                result = future.result()
                rows = [result] if isinstance(result, dict) else result
                results.extend(rows)
                _append_result_rows(
                    pd.DataFrame(rows), results_path, csv_state, append_to_csv
                )

            except Exception:
                error = sys.exc_info()[1]
                traceback_text = traceback.format_exc()
                print(traceback_text)
                metadata = future_labels[future]
                append_to_csv(
                    pd.DataFrame([
                        _failure_row(metadata, error, traceback_text)
                    ]),
                    failure_results_path,
                    failure_csv_state,
                )
                failures.append(metadata)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

    if failures:
        raise RuntimeError(
            f"{len(failures)} simulation worker(s) failed; see traceback above."
        )

    if not results:
        raise RuntimeError("All simulations failed; no results were produced.")

    print('\nAll simulations completed. Results saved to', results_path)
