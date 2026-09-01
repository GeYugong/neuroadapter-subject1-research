#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT to the experiment root}"
REVISION=35b5664bec8822e2f77da5e090e96f91d0095be6
BASE_URL="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/$REVISION/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label"
TARGET_DIR="$PROJECT_ROOT/data/raw/schaefer/fsaverage/label"

mkdir -p "$TARGET_DIR"

for hemi in lh rh; do
  name="${hemi}.Schaefer2018_1000Parcels_7Networks_order.annot"
  target="$TARGET_DIR/$name"
  if [[ ! -s "$target" ]]; then
    curl --fail --location --retry 5 --retry-delay 5 \
      "$BASE_URL/$name" --output "${target}.part"
    mv "${target}.part" "$target"
  fi
done

printf '%s\n' "$REVISION" > "$TARGET_DIR/CBIG_REVISION"
sha256sum "$TARGET_DIR"/*.annot > "$TARGET_DIR/SHA256SUMS"
