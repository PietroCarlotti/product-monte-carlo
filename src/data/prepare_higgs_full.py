"""Prepare the complete standardized HIGGS dataset with deterministic partitions."""

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

FEATURE_COLUMNS = tuple(f"feature_{index}" for index in range(1, 29))
RESPONSE_COLUMN = "outcome"
PARTITION_COLUMN = "partition"
BATCH_COLUMN = "batch"
INTERCEPT_COLUMN = "intercept"
DEFAULT_BATCHES = 10
DEFAULT_SEED = 91011
EXPECTED_OBSERVATIONS = 11_000_000
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
ARCHIVE_SHA256 = ""


# Stream numeric HIGGS rows from the compressed archive.
def stream_higgs_rows(archive_path: Path):
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("HIGGS.csv.gz") as compressed:
            with gzip.open(compressed, mode="rt", newline="") as stream:
                for row in csv.reader(stream):
                    if len(row) != 29:
                        raise ValueError(f"Expected 29 HIGGS columns, found {len(row)}.")
                    values = np.asarray(row, dtype=np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError("HIGGS rows must contain only finite values.")
                    if values[0] not in (0.0, 1.0):
                        raise ValueError("HIGGS outcome must be binary.")
                    yield values


# Compute the SHA-256 digest of a file.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Compute full-data population moments with a numerically stable streaming update.
def compute_feature_moments(archive_path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    count = 0
    means = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    sum_squares = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    class_counts = np.zeros(2, dtype=np.int64)
    for values in stream_higgs_rows(archive_path):
        features = values[1:]
        count += 1
        delta = features - means
        means += delta / count
        sum_squares += delta * (features - means)
        class_counts[int(values[0])] += 1
    if count != EXPECTED_OBSERVATIONS:
        raise ValueError(f"Expected {EXPECTED_OBSERVATIONS} HIGGS rows, found {count}.")
    standard_deviations = np.sqrt(sum_squares / count)
    if np.any(standard_deviations <= 0):
        raise ValueError("HIGGS features must have positive population standard deviations.")
    return count, means, standard_deviations, class_counts


# Build a seeded, balanced batch assignment from a permutation rank.
def build_partitions(
    n_observations: int, n_batches: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_observations % n_batches:
        raise ValueError("HIGGS row count must be divisible by the number of batches.")
    ranks = np.empty(n_observations, dtype=np.int32)
    permutation = np.random.default_rng(seed).permutation(n_observations)
    ranks[permutation] = np.arange(n_observations, dtype=np.int32)
    batches = (ranks // (n_observations // n_batches)) + 1
    return ranks, batches


# Standardize and write every HIGGS row with reproducible partition metadata.
def write_processed_data(
    archive_path: Path,
    output_path: Path,
    n_batches: int,
    seed: int,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    ranks: np.ndarray,
    batches: np.ndarray,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    batch_class_counts = np.zeros((n_batches, 2), dtype=np.int64)
    fieldnames = [
        INTERCEPT_COLUMN,
        *FEATURE_COLUMNS,
        RESPONSE_COLUMN,
        PARTITION_COLUMN,
        BATCH_COLUMN,
    ]
    with temporary_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(fieldnames)
        for row_index, values in enumerate(stream_higgs_rows(archive_path)):
            outcome = int(values[0])
            batch = int(batches[row_index])
            standardized_features = (values[1:] - means) / standard_deviations
            writer.writerow([1.0, *standardized_features, outcome, int(ranks[row_index]), batch])
            batch_class_counts[batch - 1, outcome] += 1
    temporary_path.replace(output_path)
    return batch_class_counts


# Build, validate, and persist the complete HIGGS preprocessing contract.
def prepare_higgs_full(
    archive_path: Path, output_path: Path, metadata_path: Path, n_batches: int, seed: int
) -> dict:
    if n_batches < 1:
        raise ValueError("n_batches must be positive.")
    n_observations, means, standard_deviations, class_counts = compute_feature_moments(archive_path)
    ranks, batches = build_partitions(n_observations, n_batches, seed)
    batch_class_counts = write_processed_data(
        archive_path, output_path, n_batches, seed, means, standard_deviations, ranks, batches
    )
    metadata = {
        "source_url": ARCHIVE_URL,
        "archive_sha256": sha256(archive_path),
        "archive_sha256_expected": ARCHIVE_SHA256 or None,
        "raw_member": "HIGGS.csv.gz",
        "n_observations": n_observations,
        "feature_names": [INTERCEPT_COLUMN, *FEATURE_COLUMNS],
        "native_feature_names": list(FEATURE_COLUMNS),
        "response_column": RESPONSE_COLUMN,
        "response_encoding": {"0": 0, "1": 1},
        "partition_column": PARTITION_COLUMN,
        "batch_column": BATCH_COLUMN,
        "n_batches": n_batches,
        "partition_rule": (
            "fixed seeded permutation rank; batch assignment uses contiguous rank blocks"
        ),
        "partition_seed": seed,
        "class_counts": {str(label): int(class_counts[label]) for label in range(2)},
        "class_proportions": {
            str(label): float(class_counts[label] / n_observations) for label in range(2)
        },
        "batch_sizes": {
            str(batch): int(np.sum(batches == batch)) for batch in range(1, n_batches + 1)
        },
        "batch_class_counts": {
            str(batch): {
                str(label): int(batch_class_counts[batch - 1, label]) for label in range(2)
            }
            for batch in range(1, n_batches + 1)
        },
        "features_removed": [],
        "feature_transform": (
            "intercept_unchanged; "
            "z_score_using_full_dataset_population_mean_and_standard_deviation_for_all_non_intercept_features"
        ),
        "feature_means": {name: float(means[index]) for index, name in enumerate(FEATURE_COLUMNS)},
        "feature_standard_deviations": {
            name: float(standard_deviations[index]) for index, name in enumerate(FEATURE_COLUMNS)
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


# Parse paths and build the complete HIGGS preprocessing artifacts.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/higgs/higgs.zip"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/higgs/higgs_processed.csv")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/higgs/higgs_processed_metadata.json")
    )
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    metadata = prepare_higgs_full(args.archive, args.output, args.metadata, args.batches, args.seed)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
