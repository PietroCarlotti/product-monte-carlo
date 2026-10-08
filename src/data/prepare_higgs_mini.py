"""Extract a reproducible mini HIGGS dataset from the compressed archive."""

import argparse
import csv
import gzip
import zipfile
from pathlib import Path


# Write the requested prefix of the HIGGS archive with stable column names.
def prepare_higgs_mini(archive_path: Path, output_path: Path, limit: int, n_batches: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["batch", "outcome", *[f"feature_{index}" for index in range(1, 11)]]
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("HIGGS.csv.gz") as compressed:
            with gzip.open(compressed, mode="rt", newline="") as stream:
                reader = csv.reader(stream)
                with output_path.open("w", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(fieldnames)
                    rows_written = 0
                    for row in reader:
                        if rows_written == limit:
                            break
                        if len(row) != 29:
                            raise ValueError(f"Expected 29 columns, found {len(row)}")
                        writer.writerow([(rows_written % n_batches) + 1, *row[:11]])
                        rows_written += 1
    if rows_written != limit:
        raise ValueError(f"Archive contained fewer than {limit} rows.")


# Parse command-line paths and extract the mini HIGGS CSV.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/higgs/higgs.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/higgs/higgs_processed_mini.csv"))
    parser.add_argument("--limit", type=int, default=11_000)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()
    prepare_higgs_mini(args.archive, args.output, args.limit, args.batches)


if __name__ == "__main__":
    main()
