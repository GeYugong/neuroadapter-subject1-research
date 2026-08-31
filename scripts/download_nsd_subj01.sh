#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/matengyu/geyugong/neuroadapter-subject1-research}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/raw/nsd}
FINGERPRINT_DIR=${FINGERPRINT_DIR:-$PROJECT_ROOT/data/fingerprints}
AWS_BIN=${AWS_BIN:-$(command -v aws)}

if [[ -z "$AWS_BIN" || ! -x "$AWS_BIN" ]]; then
  echo "aws CLI is required" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/experiments/nsd"
mkdir -p "$DATA_ROOT/stimuli"
mkdir -p "$DATA_ROOT/betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR"
mkdir -p "$FINGERPRINT_DIR"

exec 9>"$PROJECT_ROOT/logs/download_nsd_subj01.lock"
if ! flock -n 9; then
  echo "another NSD download is already running" >&2
  exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export AWS_EC2_METADATA_DISABLED=true
export AWS_DEFAULT_REGION=us-east-1

copy_atomic() {
  local source=$1
  local target=$2
  local temp="${target}.part"

  if [[ -s "$target" ]]; then
    echo "[skip] $target"
    return
  fi

  rm -f "$temp"
  echo "[copy] $source"
  "$AWS_BIN" s3 cp --no-sign-request --only-show-errors "$source" "$temp"
  test -s "$temp"
  mv "$temp" "$target"
}

echo "started_at=$(date --iso-8601=seconds)"
echo "project_root=$PROJECT_ROOT"
echo "data_root=$DATA_ROOT"

"$AWS_BIN" s3 ls --no-sign-request --recursive \
  s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/ \
  > "$FINGERPRINT_DIR/nsd_subj01_beta_s3_inventory.txt"

copy_atomic \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat \
  "$DATA_ROOT/experiments/nsd/nsd_expdesign.mat"

copy_atomic \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
  "$DATA_ROOT/experiments/nsd/nsd_stim_info_merged.csv"

copy_atomic \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl \
  "$DATA_ROOT/experiments/nsd/nsd_stim_info_merged.pkl"

copy_atomic \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  "$DATA_ROOT/stimuli/nsd_stimuli.hdf5"

echo "[sync] Subject 1 fsaverage betas"
"$AWS_BIN" s3 sync --no-sign-request --only-show-errors \
  s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/ \
  "$DATA_ROOT/betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/"

find "$DATA_ROOT" -type f -printf '%P\t%s\n' | LC_ALL=C sort \
  > "$FINGERPRINT_DIR/nsd_subj01_local_inventory.tsv"

echo "finished_at=$(date --iso-8601=seconds)"
du -sh "$DATA_ROOT"

