#!/usr/bin/env bash

# Download the original datasets into data/raw.
# The HIGGS source is an archive; only its CSV payload is retained locally.
# Raw data is ignored by Git because the HIGGS file is very large.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${repo_root}/data/raw"
tmp_archive="$(mktemp --suffix=.zip)"
trap 'rm -f "${tmp_archive}"' EXIT

mkdir -p "${data_root}/higgs" "${data_root}/respiratory"

# Download the HIGGS archive temporarily, extract its CSV payload, then remove the archive.
curl --fail --location --retry 3 \
  --output "${tmp_archive}" \
  "https://archive.ics.uci.edu/static/public/280/higgs.zip"
higgs_member="$(unzip -Z1 "${tmp_archive}" | grep -Ei '(^|/)HIGGS\.csv(\.gz)?$' | head -n 1)"
[ -n "${higgs_member}" ]
unzip -p "${tmp_archive}" "${higgs_member}" | gzip -dc > "${data_root}/higgs/HIGGS.csv"

# Download the small respiratory trial CSV directly.
curl --fail --location --retry 3 \
  --output "${data_root}/respiratory/respiratory.csv" \
  "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/respiratory.csv"
