#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/matengyu/geyugong/neuroadapter-subject1-research}
CONDA_BIN=${CONDA_BIN:-/data/matengyu/miniconda3/bin/conda}
PREFIX=${PREFIX:-$PROJECT_ROOT/envs/neuroadapter}

export CONDA_PKGS_DIRS="$PROJECT_ROOT/cache/conda"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=180

mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$PROJECT_ROOT/cache/wheels"

if [[ ! -x "$PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$PREFIX" python=3.11 pip=25.2 setuptools wheel
fi

PYTHON="$PREFIX/bin/python"

"$PYTHON" -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

"$PYTHON" -m pip install -r "$PROJECT_ROOT/repo/environment/requirements-candidate.txt"
"$PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT/repo/vendor/CLIP"

"$PYTHON" -m pip check
"$PYTHON" -m pip freeze --all | LC_ALL=C sort \
  > "$PROJECT_ROOT/data/fingerprints/requirements-freeze-candidate.txt"

"$PYTHON" - <<'PY'
import json
import platform
from pathlib import Path

import accelerate
import diffusers
import numpy
import torch
import torchvision
import transformers

root = Path("/data/matengyu/geyugong/neuroadapter-subject1-research")
payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "accelerate": accelerate.__version__,
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
    "cuda_runtime_in_wheel": torch.version.cuda,
}
(root / "data/fingerprints/environment_candidate.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY

