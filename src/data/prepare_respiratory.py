"""Prepare the respiratory trial data under the shared real-data contract."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "center",
    "id",
    "treat",
    "sex",
    "age",
    "baseline",
    "visit",
    "outcome",
)
MODEL_COVARIATE_COLUMNS = ("treat", "sex", "age", "baseline")
PREDICTOR_COLUMNS = ("intercept", *MODEL_COVARIATE_COLUMNS)
RESPONSE_COLUMN = "outcome"
PARTITION_COLUMN = "partition"
BATCH_COLUMN = "batch"


# Load and validate the visit-four respiratory observations.
def load_respiratory_visit(path, visit=4):
    data = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Respiratory data is missing columns: {sorted(missing)}")
    data = data.loc[data["visit"] == visit, list(REQUIRED_COLUMNS)].copy()
    if data.empty:
        raise ValueError(f"Respiratory data has no rows for visit {visit}.")
    if data.duplicated(["center", "id"]).any():
        raise ValueError("The selected respiratory visit must have one row per patient.")
    if data.isna().any().any():
        raise ValueError("The selected respiratory visit must not contain missing values.")
    return data.sort_values(["center", "id"]).reset_index(drop=True)


# Encode and standardize every non-intercept respiratory model covariate.
def build_respiratory_design(data):
    encoded = data.copy()
    for column, mapping in (("treat", {"A": 0, "P": 1}), ("sex", {"F": 0, "M": 1})):
        encoded[column] = encoded[column].map(mapping)
        if encoded[column].isna().any():
            raise ValueError(f"Unexpected value in respiratory column {column}.")
    covariates = encoded[list(MODEL_COVARIATE_COLUMNS)].astype(float)
    feature_means = covariates.mean()
    feature_stds = covariates.std(ddof=0)
    if (feature_stds <= 0).any():
        raise ValueError(
            "Respiratory model covariates must have positive population standard deviations."
        )
    design = (covariates - feature_means) / feature_stds
    design.insert(0, "intercept", 1.0)
    return encoded, design, feature_means, feature_stds


# Build a canonical manifest for respiratory preprocessing.
def build_metadata(raw_path, prepared, feature_means, feature_stds, center_to_batch):
    class_counts = prepared[RESPONSE_COLUMN].value_counts().sort_index()
    batch_counts = (
        prepared.groupby(BATCH_COLUMN)[RESPONSE_COLUMN].value_counts().unstack(fill_value=0)
    )
    return {
        "source_path": str(raw_path),
        "n_observations": len(prepared),
        "feature_names": list(PREDICTOR_COLUMNS),
        "native_feature_names": list(MODEL_COVARIATE_COLUMNS),
        "response_column": RESPONSE_COLUMN,
        "partition_column": PARTITION_COLUMN,
        "batch_column": BATCH_COLUMN,
        "n_batches": len(center_to_batch),
        "partition_rule": "deterministic rank after sorting by center and patient identifier",
        "batch_rule": "observed hospital center mapped to one batch per center",
        "center_to_batch": {str(center): int(batch) for center, batch in center_to_batch.items()},
        "class_counts": {str(label): int(count) for label, count in class_counts.items()},
        "class_proportions": {
            str(label): float(count / len(prepared)) for label, count in class_counts.items()
        },
        "batch_sizes": {
            str(batch): int((prepared[BATCH_COLUMN] == batch).sum())
            for batch in sorted(prepared[BATCH_COLUMN].unique())
        },
        "batch_class_counts": {
            str(batch): {str(label): int(batch_counts.loc[batch].get(label, 0)) for label in (0, 1)}
            for batch in batch_counts.index
        },
        "features_removed": [],
        "feature_transform": (
            "intercept_unchanged; "
            "z_score_using_full_dataset_population_mean_and_standard_deviation_for_all_non_intercept_features"
        ),
        "feature_means": {name: float(feature_means[name]) for name in MODEL_COVARIATE_COLUMNS},
        "feature_standard_deviations": {
            name: float(feature_stds[name]) for name in MODEL_COVARIATE_COLUMNS
        },
    }


# Prepare and persist canonical respiratory model data and metadata.
def prepare_respiratory_data(raw_path, output_path, metadata_path=None):
    selected = load_respiratory_visit(raw_path)
    encoded, design, feature_means, feature_stds = build_respiratory_design(selected)
    centers = sorted(encoded["center"].unique())
    center_to_batch = {center: index + 1 for index, center in enumerate(centers)}
    metadata = encoded[["center", "id", RESPONSE_COLUMN]].reset_index(drop=True)
    metadata[PARTITION_COLUMN] = np.arange(len(metadata), dtype=int)
    metadata[BATCH_COLUMN] = metadata["center"].map(center_to_batch)
    prepared = pd.concat([design.reset_index(drop=True), metadata], axis=1)
    prepared = prepared[
        [*PREDICTOR_COLUMNS, RESPONSE_COLUMN, PARTITION_COLUMN, BATCH_COLUMN, "center", "id"]
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else output_path.with_name(f"{output_path.stem}_metadata.json")
    )
    metadata_path.write_text(
        json.dumps(
            build_metadata(raw_path, prepared, feature_means, feature_stds, center_to_batch),
            indent=2,
        )
        + "\n"
    )
    return prepared
