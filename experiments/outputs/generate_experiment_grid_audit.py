"""Generate the experiment-grid coverage audit from the adjacent manifest."""

import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

##########################
# Audit helper functions #
##########################

AUDIT_FIELDS = [
    "experiment_id",
    "role",
    "family",
    "category",
    "scenario",
    "num_nodes",
    "dimension",
    "sample_size_policy",
    "sample_size",
    "logical_sample_folder",
    "status",
    "expected_iterations",
    "expected_method_iterations",
    "observed_iteration_count",
    "observed_iterations",
    "missing_iterations",
    "extra_iterations",
    "expected_methods",
    "observed_methods",
    "missing_methods",
    "additional_methods",
    "expected_targets",
    "observed_targets",
    "missing_targets",
    "expected_design_rows_per_cell",
    "complete_expected_cells",
    "expected_cells",
    "complete_expected_rows",
    "expected_rows",
    "observed_rows",
    "source_csv_count",
    "source_csvs",
    "invalid_source_csv_count",
    "invalid_source_csvs",
    "invalid_result_source_csv_count",
    "invalid_result_source_csvs",
    "artifact_source_csv_count",
    "artifact_source_csvs",
    "conflicting_design_slot_count",
    "conflicting_design_slots",
    "conflicting_source_csv_count",
    "conflicting_source_csvs",
    "surplus_design_slot_count",
    "surplus_design_slots",
    "current_result_roots",
    "artifact_roots",
    "current_sample_containers",
    "current_config_root",
    "current_config_container",
    "config_policy",
    "config_status",
    "current_config_path_exists",
    "canonical_config_root",
    "canonical_config_container",
    "canonical_result_root",
    "canonical_cell_path",
    "active_markers",
    "intentional_exception",
    "notes",
]

# Parse an integer identifier without numeric coercion.
def _parse_integer(value, minimum):
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+", text) is None:
        return None
    parsed = int(text)
    if parsed < minimum:
        return None
    return parsed

# Return a strictly positive sample-size identifier.
def _sample_size(row):
    return _parse_integer(row.get("n"), 1)

# Return the effective method name across current result schemas.
def _method_name(row):
    method = row.get("method", "").strip()
    if method:
        return method
    return row.get("algorithm", "").strip()

# Return the effective target name across current result schemas.
def _target_name(row):
    target = row.get("target", "").strip()
    if target:
        return target
    return row.get("h", "").strip() or None

# Return the replication identifier across current result schemas.
def _iteration_id(row):
    identifiers = []
    for field in ("global_repetition", "iteration_id", "iteration", "repetition"):
        raw_value = row.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        value = _parse_integer(raw_value, 0)
        if value is None:
            return None
        identifiers.append(value)
    if not identifiers or len(set(identifiers)) != 1:
        return None
    return identifiers[0]

# Read valid CSV rows only from newline-terminated bytes.
def _read_complete_rows(path):
    data = path.read_bytes()
    truncated = bool(data) and not data.endswith((b"\n", b"\r"))
    newline_index = max(data.rfind(b"\n"), data.rfind(b"\r"))
    if newline_index < 0:
        return [], True
    complete_data = data[: newline_index + 1]
    try:
        text = complete_data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], True
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    rows = []
    malformed = False
    try:
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                malformed = True
                continue
            rows.append(row)
    except csv.Error:
        malformed = True
    return rows, truncated or malformed or not rows

# Resolve all existing legacy and canonical result roots.
def _result_roots(experiment, outputs_root):
    relative_roots = list(experiment["current_result_roots"])
    canonical_root = experiment["canonical_result_root"]
    if canonical_root not in relative_roots:
        relative_roots.append(canonical_root)
    roots = []
    for relative_root in relative_roots:
        root = outputs_root / relative_root
        if root.exists() and root not in roots:
            roots.append(root)
    return roots

# Resolve declared artifact roots that must remain traceable but cannot count.
def _artifact_roots(experiment, outputs_root):
    roots = []
    for relative_root in experiment.get("artifact_roots", []):
        root = outputs_root / relative_root
        if root.exists() and root not in roots:
            roots.append(root)
    return roots

# Find active simulation markers beneath result roots.
def _active_markers(roots, outputs_root):
    markers = set()
    for root in roots:
        for path in root.rglob(".sim_results_tmp*"):
            markers.add(path.relative_to(outputs_root).as_posix())
    return sorted(markers)

# Infer sample sizes named by an active marker.
def _marker_sample_sizes(marker):
    matches = re.findall(
        r"(?:^|/)n_([0-9]+(?:_[0-9]+)*)(?:/|$)",
        marker,
    )
    if not matches:
        marker_match = re.search(r"\.sim_results_tmp[^/]*n_([0-9_]+)", marker)
        if marker_match is not None:
            matches = [marker_match.group(1)]
    if not matches:
        return None
    return {int(value) for value in matches[-1].strip("_").split("_")}

# Determine whether active markers plausibly cover a logical cell.
def _cell_active(markers, sample_size):
    for marker in markers:
        marker_sizes = _marker_sample_sizes(marker)
        if marker_sizes is None or sample_size is None:
            return True
        if sample_size in marker_sizes:
            return True
    return False

# Infer sample sizes from exact or grouped path components.
def _path_sample_sizes(path):
    matches = re.findall(
        r"(?:^|/)n_([0-9]+(?:_[0-9]+)*)(?:/|$)",
        path,
    )
    if not matches:
        return None
    return {int(value) for value in matches[-1].split("_")}

# Load result rows and preserve file-level validity metadata.
def _load_result_rows(roots, artifact_roots, outputs_root):
    loaded = []
    files = {}
    seen_paths = set()
    for root in roots:
        for path in sorted(root.rglob("*.csv")):
            if any(path.is_relative_to(artifact_root) for artifact_root in artifact_roots):
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            source = path.relative_to(outputs_root).as_posix()
            artifact = path.name.endswith("_failures.csv")
            if artifact:
                files[source] = {
                    "invalid": True,
                    "artifact": True,
                    "sample_sizes": _path_sample_sizes(source),
                    "invalid_sizes": _path_sample_sizes(source),
                    "root": root,
                }
                continue
            rows, invalid = _read_complete_rows(path)
            files[source] = {
                "invalid": invalid,
                "artifact": False,
                "sample_sizes": _path_sample_sizes(source),
                "invalid_sizes": _path_sample_sizes(source) if invalid else set(),
                "root": root,
            }
            for row in rows:
                loaded.append((row, source, root))
    return loaded, files

# Index artifact CSVs as invalid sources without loading result rows.
def _load_artifact_files(roots, outputs_root):
    files = {}
    for root in roots:
        for path in sorted(root.rglob("*.csv")):
            source = path.relative_to(outputs_root).as_posix()
            files[source] = {
                "invalid": True,
                "sample_sizes": _path_sample_sizes(source),
                "root": root,
            }
    return files

# Determine whether a logical cell is an intentional exception.
def _is_exception(experiment, sample_size):
    exceptions = experiment["intentional_exceptions"]
    if exceptions == "all":
        return True
    return sample_size in exceptions

# Map a logical sample size to its grouped computational config declaration.
def _config_group(experiment, sample_size):
    if sample_size is None:
        return None
    for group in experiment["config_sample_groups"]:
        if sample_size in group["sample_sizes"]:
            return group
    return None

# Resolve the config root during hierarchy migration.
def _current_config_root(experiment, outputs_root):
    current_root = experiment["current_config_root"]
    canonical_root = experiment["canonical_config_root"]
    if (outputs_root / current_root).exists():
        return current_root
    if (outputs_root / canonical_root).exists():
        return canonical_root
    return current_root

# Resolve current and canonical computational config paths.
def _config_paths(experiment, sample_size, outputs_root):
    group = _config_group(experiment, sample_size)
    current_root = _current_config_root(experiment, outputs_root)
    canonical_root = experiment["canonical_config_root"]
    if group is None:
        return current_root, canonical_root
    canonical_container = group["container"]
    current_container = group.get("current_container", canonical_container)
    direct_path = f"{current_root}/{current_container}"
    batched_path = f"{current_root}/batches/{current_container}"
    if (outputs_root / batched_path).exists():
        current_path = batched_path
    elif (outputs_root / direct_path).exists():
        current_path = direct_path
    elif current_root == canonical_root:
        current_path = batched_path
    else:
        current_path = direct_path
    return current_path, f"{canonical_root}/batches/{canonical_container}"

# Return effective iteration counts for every expected method.
def _method_iteration_counts(experiment):
    overrides = experiment.get("expected_method_iterations", {})
    return {
        method: overrides.get(method, experiment["expected_iterations"])
        for method in experiment["expected_methods"]
    }

# Build the expected method-target-iteration cells.
def _expected_cells(experiment):
    method_iterations = _method_iteration_counts(experiment)
    targets = experiment["expected_targets"] or [None]
    return {
        (iteration, method, target)
        for method, iterations in method_iterations.items()
        for iteration in range(iterations)
        for target in targets
    }

# Convert a result row to a stable signature for copy deduplication.
def _row_signature(row):
    return tuple(sorted(row.items()))

# Classify one logical sample-size cell.
def _status(experiment, complete, has_rows, active, intentional_exception):
    if experiment["role"] == "deferred":
        return "DEFERRED"
    if complete:
        if experiment["role"] == "auxiliary":
            return "AUXILIARY_COMPLETE"
        return "COMPLETE"
    if active:
        return "ACTIVE"
    if intentional_exception:
        return "EXCEPTION"
    if has_rows:
        return "PARTIAL"
    return "MISSING"

# Join sorted values for deterministic CSV output.
def _join(values):
    return "|".join(str(value) for value in sorted(values))

# Identify the container holding a source CSV.
def _sample_container(source, root, outputs_root):
    source_path = outputs_root / source
    try:
        relative_parent = source_path.parent.relative_to(root)
    except ValueError:
        return source_path.parent.relative_to(outputs_root).as_posix()
    if relative_parent == Path("."):
        return root.relative_to(outputs_root).as_posix()
    return (root / relative_parent).relative_to(outputs_root).as_posix()

# Mark a result source invalid for the affected logical sample sizes.
def _mark_invalid_source(metadata, sample_sizes):
    metadata["invalid"] = True
    current_sizes = metadata["invalid_sizes"]
    if current_sizes is None or sample_sizes is None:
        metadata["invalid_sizes"] = None
    else:
        current_sizes.update(sample_sizes)

# Determine whether invalid source metadata applies to a logical cell.
def _invalid_source_applies(metadata, sample_size):
    invalid_sizes = metadata["invalid_sizes"]
    return sample_size is None or invalid_sizes is None or sample_size in invalid_sizes

# Format a source-local design-slot identifier.
def _format_design_slot(slot):
    iteration, method, target, ordinal = slot
    target_text = "" if target is None else target
    return f"{iteration}:{method}:{target_text}:{ordinal}"

# Report config availability separately from result coverage.
def _config_status(experiment, current_path, outputs_root):
    if experiment["config_policy"] == "intentionally_absent":
        return "INTENTIONALLY_ABSENT"
    path = outputs_root / current_path
    if path.is_file() and path.suffix == ".json":
        return "AVAILABLE"
    if path.is_dir() and any(path.rglob("*.json")):
        return "AVAILABLE"
    return "MISSING"

# Resolve an experiment's declared config policy.
def _config_policy(manifest, experiment):
    policy = manifest["config_policy"]
    if experiment["id"] in policy["intentionally_absent"]:
        return "intentionally_absent"
    return policy["default"]

# Audit one experiment into logical sample-size rows.
def _audit_experiment(experiment, outputs_root):
    roots = _result_roots(experiment, outputs_root)
    artifact_roots = _artifact_roots(experiment, outputs_root)
    active_markers = _active_markers(roots, outputs_root)
    loaded_rows, result_files = _load_result_rows(
        roots,
        artifact_roots,
        outputs_root,
    )
    artifact_files = _load_artifact_files(artifact_roots, outputs_root)
    expected_cells = _expected_cells(experiment)
    method_iterations = _method_iteration_counts(experiment)
    expected_iteration_ids = {
        iteration
        for iterations in method_iterations.values()
        for iteration in range(iterations)
    }
    sample_sizes = experiment["expected_sample_sizes"]
    if experiment["sample_size_policy"] == "no_grid":
        sample_sizes = [None]
    else:
        observed_sample_sizes = {
            sample_size
            for row, _, _ in loaded_rows
            if (sample_size := _sample_size(row)) is not None
        }
        undeclared_sizes = observed_sample_sizes - set(sample_sizes)
        if undeclared_sizes:
            sizes = ", ".join(str(value) for value in sorted(undeclared_sizes))
            raise ValueError(f"{experiment['id']}: undeclared sample sizes: {sizes}")
    valid_rows = []
    for row, source, _ in loaded_rows:
        metadata = result_files[source]
        row_sample_size = None
        if sample_sizes != [None]:
            row_sample_size = _sample_size(row)
            if row_sample_size is None:
                _mark_invalid_source(metadata, metadata["sample_sizes"])
                continue
            path_sizes = metadata["sample_sizes"]
            if path_sizes is not None and row_sample_size not in path_sizes:
                _mark_invalid_source(metadata, path_sizes | {row_sample_size})
                continue
        iteration = _iteration_id(row)
        if iteration is None:
            invalid_sizes = None if sample_sizes == [None] else {row_sample_size}
            _mark_invalid_source(metadata, invalid_sizes)
            continue
        valid_rows.append(
            (
                row,
                source,
                metadata["root"],
                row_sample_size,
                iteration,
                _method_name(row),
                _target_name(row),
            )
        )
    audit_rows = []
    for sample_size in sample_sizes:
        cell_rows = []
        for record in valid_rows:
            row_sample_size = record[3]
            if sample_size is None or row_sample_size == sample_size:
                cell_rows.append(record)
        observed_iterations = set()
        observed_methods = set()
        observed_targets = set()
        observed_cell_counts = Counter()
        source_records = {}
        source_csvs = set()
        invalid_source_csvs = set()
        invalid_result_source_csvs = set()
        artifact_source_csvs = set()
        containers = set()
        for row, source, root, _, iteration, method, target in cell_rows:
            observed_iterations.add(iteration)
            if method:
                observed_methods.add(method)
            if target:
                observed_targets.add(target)
            source_records.setdefault(source, []).append(
                (row, iteration, method, target)
            )
            source_csvs.add(source)
            containers.add(_sample_container(source, root, outputs_root))
        for source, metadata in result_files.items():
            if metadata["invalid"] and _invalid_source_applies(metadata, sample_size):
                invalid_source_csvs.add(source)
                if metadata["artifact"]:
                    artifact_source_csvs.add(source)
                else:
                    invalid_result_source_csvs.add(source)
                containers.add(
                    _sample_container(
                        source,
                        metadata["root"],
                        outputs_root,
                    )
                )
        for source, metadata in artifact_files.items():
            inferred_sizes = metadata["sample_sizes"]
            applies = sample_size is None or inferred_sizes is None
            if inferred_sizes is not None and sample_size in inferred_sizes:
                applies = True
            if applies:
                invalid_source_csvs.add(source)
                artifact_source_csvs.add(source)
                containers.add(
                    _sample_container(
                        source,
                        metadata["root"],
                        outputs_root,
                    )
                )
        slot_records = {}
        slot_sources = {}
        conflicting_slots = set()
        conflicting_sources = set()
        for source, records in source_records.items():
            seen_signatures = set()
            ordinals = Counter()
            for row, iteration, method, target in records:
                signature = _row_signature(row)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                cell = (iteration, method, target)
                ordinal = ordinals[cell]
                ordinals[cell] += 1
                slot = (*cell, ordinal)
                if slot in slot_records and slot_records[slot] != signature:
                    conflicting_slots.add(slot)
                    conflicting_sources.update(slot_sources[slot])
                    conflicting_sources.add(source)
                else:
                    slot_records.setdefault(slot, signature)
                    slot_sources.setdefault(slot, set()).add(source)
        for slot in slot_records:
            if slot in conflicting_slots:
                continue
            iteration, method, target, _ = slot
            if method:
                observed_cell_counts[(iteration, method, target)] += 1
        observed_row_count = len(slot_records)
        design_rows = experiment["expected_design_rows_per_cell"]
        surplus_slots = {
            slot
            for slot in slot_records
            if slot not in conflicting_slots
            and slot[:3] in expected_cells
            and slot[3] >= design_rows
        }
        complete_cells = {
            cell for cell in expected_cells if observed_cell_counts[cell] == design_rows
        }
        missing_iterations = expected_iteration_ids - observed_iterations
        extra_iterations = observed_iterations - expected_iteration_ids
        expected_methods = set(experiment["expected_methods"])
        expected_targets = set(experiment["expected_targets"])
        missing_methods = expected_methods - observed_methods
        additional_methods = observed_methods - expected_methods
        missing_targets = expected_targets - observed_targets
        complete = (
            expected_cells <= complete_cells
            and not invalid_result_source_csvs
            and not conflicting_slots
            and not surplus_slots
        )
        intentional_exception = _is_exception(experiment, sample_size)
        active = _cell_active(active_markers, sample_size)
        status = _status(
            experiment,
            complete,
            bool(cell_rows),
            active,
            intentional_exception,
        )
        canonical_root = experiment["canonical_result_root"]
        current_config_root = _current_config_root(experiment, outputs_root)
        canonical_config_root = experiment["canonical_config_root"]
        current_config_container, canonical_config_container = _config_paths(
            experiment,
            sample_size,
            outputs_root,
        )
        config_status = _config_status(
            experiment,
            current_config_container,
            outputs_root,
        )
        logical_folder = "" if sample_size is None else f"n_{sample_size}"
        canonical_cell_path = canonical_root
        if logical_folder:
            canonical_cell_path = f"{canonical_root}/{logical_folder}"
        audit_rows.append(
            {
                "experiment_id": experiment["id"],
                "role": experiment["role"],
                "family": experiment["family"],
                "category": experiment["category"],
                "scenario": experiment["scenario"],
                "num_nodes": experiment["num_nodes"],
                "dimension": experiment["dimension"],
                "sample_size_policy": experiment["sample_size_policy"],
                "sample_size": "" if sample_size is None else sample_size,
                "logical_sample_folder": logical_folder,
                "status": status,
                "expected_iterations": experiment["expected_iterations"],
                "expected_method_iterations": _join(
                    f"{method}:{iterations}"
                    for method, iterations in method_iterations.items()
                ),
                "observed_iteration_count": len(observed_iterations),
                "observed_iterations": _join(observed_iterations),
                "missing_iterations": _join(missing_iterations),
                "extra_iterations": _join(extra_iterations),
                "expected_methods": _join(expected_methods),
                "observed_methods": _join(observed_methods),
                "missing_methods": _join(missing_methods),
                "additional_methods": _join(additional_methods),
                "expected_targets": _join(expected_targets),
                "observed_targets": _join(observed_targets),
                "missing_targets": _join(missing_targets),
                "expected_design_rows_per_cell": design_rows,
                "complete_expected_cells": len(complete_cells),
                "expected_cells": len(expected_cells),
                "complete_expected_rows": len(complete_cells) * design_rows,
                "expected_rows": len(expected_cells) * design_rows,
                "observed_rows": observed_row_count,
                "source_csv_count": len(source_csvs),
                "source_csvs": _join(source_csvs),
                "invalid_source_csv_count": len(invalid_source_csvs),
                "invalid_source_csvs": _join(invalid_source_csvs),
                "invalid_result_source_csv_count": len(
                    invalid_result_source_csvs
                ),
                "invalid_result_source_csvs": _join(
                    invalid_result_source_csvs
                ),
                "artifact_source_csv_count": len(artifact_source_csvs),
                "artifact_source_csvs": _join(artifact_source_csvs),
                "conflicting_design_slot_count": len(conflicting_slots),
                "conflicting_design_slots": _join(
                    _format_design_slot(slot) for slot in conflicting_slots
                ),
                "conflicting_source_csv_count": len(conflicting_sources),
                "conflicting_source_csvs": _join(conflicting_sources),
                "surplus_design_slot_count": len(surplus_slots),
                "surplus_design_slots": _join(
                    _format_design_slot(slot) for slot in surplus_slots
                ),
                "current_result_roots": _join(
                    root.relative_to(outputs_root).as_posix() for root in roots
                ),
                "artifact_roots": _join(
                    root.relative_to(outputs_root).as_posix()
                    for root in artifact_roots
                ),
                "current_sample_containers": _join(containers),
                "current_config_root": current_config_root,
                "current_config_container": current_config_container,
                "config_policy": experiment["config_policy"],
                "config_status": config_status,
                "current_config_path_exists": str(
                    (outputs_root / current_config_container).exists()
                ).upper(),
                "canonical_config_root": canonical_config_root,
                "canonical_config_container": canonical_config_container,
                "canonical_result_root": canonical_root,
                "canonical_cell_path": canonical_cell_path,
                "active_markers": _join(active_markers),
                "intentional_exception": str(intentional_exception).upper(),
                "notes": experiment["notes"],
            }
        )
    return audit_rows

# Validate the manifest fields required by the audit.
def _validate_manifest(manifest):
    config_policy = manifest.get("config_policy")
    if not isinstance(config_policy, dict):
        raise ValueError("The manifest must define config_policy")
    if config_policy.get("default") != "required":
        raise ValueError("The default config policy must be required")
    absent_ids = config_policy.get("intentionally_absent")
    if not isinstance(absent_ids, list) or len(absent_ids) != len(set(absent_ids)):
        raise ValueError("Intentionally absent config ids must be unique")
    required_fields = {
        "id",
        "role",
        "family",
        "category",
        "scenario",
        "num_nodes",
        "dimension",
        "current_result_roots",
        "current_config_root",
        "canonical_config_root",
        "canonical_result_root",
        "sample_size_policy",
        "expected_sample_sizes",
        "expected_iterations",
        "expected_methods",
        "expected_targets",
        "expected_design_rows_per_cell",
        "config_sample_groups",
        "intentional_exceptions",
        "notes",
    }
    experiment_ids = set()
    for experiment in manifest.get("experiments", []):
        missing = required_fields - set(experiment)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{experiment.get('id', '<unknown>')}: missing {missing_text}")
        if experiment["id"] in experiment_ids:
            raise ValueError(f"Duplicate experiment id: {experiment['id']}")
        experiment_ids.add(experiment["id"])
        if experiment["role"] not in {"canonical", "auxiliary", "deferred"}:
            raise ValueError(f"{experiment['id']}: invalid role")
        if experiment["role"] == "canonical" and experiment["id"] in absent_ids:
            raise ValueError(
                f"{experiment['id']}: canonical configs cannot be intentionally absent"
            )
        if experiment["sample_size_policy"] not in {"grid", "no_grid"}:
            raise ValueError(f"{experiment['id']}: invalid sample-size policy")
        if experiment["expected_iterations"] <= 0:
            raise ValueError(f"{experiment['id']}: expected iterations must be positive")
        method_iterations = experiment.get("expected_method_iterations", {})
        if not isinstance(method_iterations, dict):
            raise ValueError(
                f"{experiment['id']}: expected method iterations must be a mapping"
            )
        unknown_methods = set(method_iterations) - set(experiment["expected_methods"])
        if unknown_methods:
            unknown_text = ", ".join(sorted(unknown_methods))
            raise ValueError(
                f"{experiment['id']}: iteration counts for unknown methods: "
                f"{unknown_text}"
            )
        invalid_counts = {
            method: count
            for method, count in method_iterations.items()
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0
        }
        if invalid_counts:
            invalid_text = ", ".join(
                f"{method}:{count}" for method, count in sorted(invalid_counts.items())
            )
            raise ValueError(
                f"{experiment['id']}: method iteration counts must be positive "
                f"integers: {invalid_text}"
            )
        if experiment["expected_design_rows_per_cell"] <= 0:
            raise ValueError(f"{experiment['id']}: design-row count must be positive")
        if not experiment["expected_methods"]:
            raise ValueError(f"{experiment['id']}: expected methods are required")
        if experiment["sample_size_policy"] == "no_grid":
            if experiment["expected_sample_sizes"]:
                raise ValueError(
                    f"{experiment['id']}: no-grid experiments cannot define sample sizes"
                )
        elif not experiment["expected_sample_sizes"]:
            raise ValueError(f"{experiment['id']}: expected sample sizes are required")
        expected_sizes = set(experiment["expected_sample_sizes"])
        grouped_sizes = []
        for group in experiment["config_sample_groups"]:
            grouped_sizes.extend(group["sample_sizes"])
        if len(grouped_sizes) != len(set(grouped_sizes)):
            raise ValueError(f"{experiment['id']}: config sample groups overlap")
        if not set(grouped_sizes) <= expected_sizes:
            raise ValueError(f"{experiment['id']}: config sample group is outside grid")
        exceptions = experiment["intentional_exceptions"]
        if exceptions != "all" and not set(exceptions) <= expected_sizes:
            raise ValueError(f"{experiment['id']}: exception is outside grid")
    if not experiment_ids:
        raise ValueError("The manifest must define at least one experiment")
    unknown_absent_ids = set(absent_ids) - experiment_ids
    if unknown_absent_ids:
        unknown_text = ", ".join(sorted(unknown_absent_ids))
        raise ValueError(f"Unknown intentionally absent config ids: {unknown_text}")

# Generate deterministic audit rows from a manifest.
def generate_audit(manifest_path, outputs_root):
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_manifest(manifest)
    rows = []
    for declared_experiment in manifest["experiments"]:
        experiment = dict(declared_experiment)
        experiment["config_policy"] = _config_policy(
            manifest,
            experiment,
        )
        rows.extend(_audit_experiment(experiment, outputs_root))
    return rows

# Write audit rows with a stable schema and ordering.
def write_audit(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=AUDIT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

# Parse command-line arguments.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the experiment-grid audit from a manifest."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "experiment_grid_manifest.json",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=script_dir,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "experiment_grid_audit.csv",
    )
    return parser.parse_args()

# Run the experiment-grid audit command.
def main():
    args = _parse_args()
    rows = generate_audit(args.manifest, args.outputs_root)
    write_audit(rows, args.output)
    print(f"Wrote {len(rows)} audit rows to {args.output}")

if __name__ == "__main__":
    main()
