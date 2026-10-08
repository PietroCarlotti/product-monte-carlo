"""Download and preprocess the UCI MAGIC Gamma Telescope data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
)
RESPONSE_COLUMN = "class"
PARTITION_COLUMN = "partition"
BATCH_COLUMN = "batch"
INTERCEPT_COLUMN = "intercept"
DEFAULT_BATCHES = 10
DEFAULT_LIMIT = 1000
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
ARCHIVE_SHA256 = ""


def download_archive(path: Path) -> None:
    """Download the UCI archive when it is not already present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urlopen(ARCHIVE_URL) as response, path.open("wb") as output:
        output.write(response.read())


def _sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_processed_data(
    archive_path: Path,
    output_path: Path,
    seed: int = 91001,
    feature_count: int = 10,
    n_batches: int = DEFAULT_BATCHES,
    limit: int | None = DEFAULT_LIMIT,
) -> dict:
    """Extract, validate, and persist all native MAGIC observations."""
    if feature_count not in (2, 10):
        raise ValueError("feature_count must be 2 or 10.")
    if n_batches < 1 or len(range(n_batches)) != n_batches:
        raise ValueError("n_batches must be positive.")
    selected_features = FEATURE_COLUMNS[:feature_count]
    if limit is not None and (limit < 1 or limit > 19020):
        raise ValueError("limit must be between 1 and 19020.")
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("magic04.data") as raw:
            data = pd.read_csv(raw, header=None, names=[*FEATURE_COLUMNS, RESPONSE_COLUMN])
    data[RESPONSE_COLUMN] = data[RESPONSE_COLUMN].map({"g": 1, "h": 0})
    if data.isna().any().any():
        raise ValueError("MAGIC preprocessing produced missing values.")
    if len(data) != 19020 or data[list(FEATURE_COLUMNS)].shape[1] != 10:
        raise ValueError("Unexpected MAGIC row or feature count.")
    if not np.isfinite(data[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("MAGIC features must be finite.")
    if not data[RESPONSE_COLUMN].isin((0, 1)).all():
        raise ValueError("MAGIC response must be binary after encoding.")
    feature_means = data[list(FEATURE_COLUMNS)].mean()
    feature_stds = data[list(FEATURE_COLUMNS)].std(ddof=0)
    data.loc[:, list(FEATURE_COLUMNS)] = (
        data[list(FEATURE_COLUMNS)] - feature_means
    ) / feature_stds
    data[INTERCEPT_COLUMN] = 1.0
    rng = np.random.default_rng(seed)
    if limit is not None:
        data = data.iloc[rng.permutation(len(data))[:limit]].reset_index(drop=True)
    permutation = rng.permutation(len(data))
    data[PARTITION_COLUMN] = np.empty(len(data), dtype=int)
    data.loc[permutation, PARTITION_COLUMN] = np.arange(len(data))
    if len(data) % n_batches:
        raise ValueError("MAGIC row count must be divisible by n_batches.")
    data[BATCH_COLUMN] = (data[PARTITION_COLUMN] // (len(data) // n_batches)) + 1
    data = data[
        [INTERCEPT_COLUMN, *selected_features, RESPONSE_COLUMN, PARTITION_COLUMN, BATCH_COLUMN]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    counts = data[RESPONSE_COLUMN].value_counts().sort_index().to_dict()
    metadata = {
        "source_url": ARCHIVE_URL,
        "archive_sha256": _sha256(archive_path),
        "archive_sha256_expected": ARCHIVE_SHA256 or None,
        "raw_member": "magic04.data",
        "n_observations": len(data),
        "feature_names": [INTERCEPT_COLUMN, *selected_features],
        "native_feature_names": list(selected_features),
        "response_column": RESPONSE_COLUMN,
        "response_encoding": {"h": 0, "g": 1},
        "partition_column": PARTITION_COLUMN,
        "batch_column": BATCH_COLUMN,
        "n_batches": n_batches,
        "partition_rule": (
            "fixed seeded permutation rank; batch assignment uses contiguous rank blocks"
        ),
        "partition_seed": seed,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "class_proportions": {str(key): float(value / len(data)) for key, value in counts.items()},
        "features_removed": [],
        "feature_transform": (
            "intercept_unchanged; "
            "z_score_using_full_dataset_population_mean_and_standard_deviation_for_all_non_intercept_features"
        ),
        "feature_count": feature_count,
        "feature_means": {name: float(feature_means[name]) for name in selected_features},
        "feature_standard_deviations": {
            name: float(feature_stds[name]) for name in selected_features
        },
    }
    return metadata


def main() -> None:
    """Parse paths and build the processed MAGIC dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/magic/magic_gamma_telescope.zip")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/magic/magic_processed.csv")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/magic/magic_processed_metadata.json")
    )
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--feature-count", type=int, default=10)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    download_archive(args.archive)
    metadata = build_processed_data(
        args.archive, args.output, args.seed, args.feature_count, args.batches, args.limit
    )
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
