#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?project root required}
RUNTIME=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/runs/experiments/generator-semantic-last-v1"
PY="$ROOT/envs/neuroadapter/bin/python"
export PYTHONPATH="$RUNTIME/scripts:$ROOT/runtime/subject01-4090-1a1fcfa/src"
export CUDA_VISIBLE_DEVICES=0,1
cd "$RUNTIME"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ]]; then
  echo "GPUs are occupied; refusing to preempt" >&2
  exit 20
fi

"$PY" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/experiments/generator-semantic-last-v1'
assert json.load(open(out/'chain-audit/audit.json'))['status']=='passed'
cal=json.load(open(out/'calibration/lambda.json'));assert cal['status']=='passed' and len(cal['ratios'])==16
a=out/'tests/M-20/snapshots/snapshot-update-00000020/model.pt'
b=out/'tests/M-20-resumed/snapshots/snapshot-update-00000020/model.pt'
import hashlib
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(a)==digest(b)
PY

date -Is > "$OUT/pilot-started-at.txt"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_generator_semantic.py --root "$ROOT" --arm K --maximum 5000 \
  2>&1 | tee "$OUT/train-K.log"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_generator_semantic.py --root "$ROOT" --arm M --maximum 20000 --pilot-stop \
  2>&1 | tee "$OUT/train-M-pilot.log"

export CUDA_VISIBLE_DEVICES=0
"$PY" scripts/evaluate_generator_semantic.py --root "$ROOT" --phase score --label B0
for label in K-5000 M-5000; do
  "$PY" scripts/evaluate_generator_semantic.py --root "$ROOT" --phase decode --label "$label" \
    2>&1 | tee "$OUT/decode-$label.log"
  "$PY" scripts/evaluate_generator_semantic.py --root "$ROOT" --phase score --label "$label" \
    2>&1 | tee "$OUT/score-$label.log"
done
"$PY" scripts/evaluate_generator_semantic.py --root "$ROOT" --phase pilot-summary
date -Is > "$OUT/pilot-completed-at.txt"
echo complete > "$OUT/PILOT_COMPLETE"

