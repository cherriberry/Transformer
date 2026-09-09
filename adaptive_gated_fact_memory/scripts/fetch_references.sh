#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reference_dir="$project_dir/references"
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT
mkdir -p "$reference_dir"

fetch_archive() {
  local owner_repo=$1
  local revision=$2
  local destination=$3
  if [[ -d "$destination" ]]; then
    echo "already present: $destination"
    return
  fi
  local archive="$temporary_dir/archive.tar.gz"
  curl -L --fail --retry 3 \
    -o "$archive" \
    "https://codeload.github.com/${owner_repo}/tar.gz/${revision}"
  mkdir -p "$destination"
  tar -xzf "$archive" --strip-components=1 -C "$destination"
}

google_revision=952b8d85a0389ba59d27de382046a6dd9c01b0e5
google_destination="$reference_dir/google-research"
if [[ ! -d "$google_destination" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-research/google-research.git \
    "$google_destination"
  git -C "$google_destination" sparse-checkout set performer LICENSE
fi

fetch_archive \
  google/gemma_pytorch \
  014acb7ac4563a5f77c76d7ff98f31b568c16508 \
  "$reference_dir/gemma_pytorch"
fetch_archive \
  mit-han-lab/streaming-llm \
  2e5042606d69933d88fbf909bd77907456b9b4dd \
  "$reference_dir/streaming-llm"
fetch_archive \
  WeianMao/triattention \
  a4bc3c8f709db60f016ef42c3feb290fd0c00c1b \
  "$reference_dir/triattention"
fetch_archive \
  qywu/memformers \
  4575d14a76284eb32d2f5e2f52cbf2378ad829f2 \
  "$reference_dir/memformers"

echo "reference sources are ready under $reference_dir"
