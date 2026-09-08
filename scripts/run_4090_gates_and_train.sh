#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the dedicated 4090 project directory}"
: "${RUNTIME:?Set RUNTIME to the frozen clean protocol checkout}"
source "$PROJECT_ROOT/repo/scripts/activate_project.sh"
export PYTHONPATH="$RUNTIME/src" PYTHONUNBUFFERED=1
PYTHON="$PROJECT_ROOT/envs/neuroadapter/bin/python"
CONFIG="$PROJECT_ROOT/configs/calibration/subject01_4090_preferred.yaml"
FALLBACK="$PROJECT_ROOT/configs/calibration/subject01_4090_fallback.yaml"
ARTIFACTS="$PROJECT_ROOT/artifacts/gates-4090"
RUNS="$PROJECT_ROOT/runs/calibration"
HELPER="$PROJECT_ROOT/repo/scripts/gate_preflight_inference.py"
LOG="$PROJECT_ROOT/repo/EXPERIMENT_LOG.md"

mkdir -p "$ARTIFACTS" "$RUNS"
exec 9>"$ARTIFACTS/pipeline.lock"
flock -n 9
[[ $(cat "$PROJECT_ROOT/artifacts/migration-20260908/data-verification.exit") == 0 ]]
[[ $(cat "$PROJECT_ROOT/artifacts/migration-20260908/hardware.exit") == 0 ]]

stage() {
  local name=$1
  shift
  local status="$ARTIFACTS/$name.exit"
  local logfile="$PROJECT_ROOT/logs/4090-$name.log"
  [[ ! -e "$status" && ! -e "$logfile" ]]
  {
    printf '\n### %s：执行 %s\n\n' "$(date --iso-8601=seconds)" "$name"
    printf '运行代码：`%s`。日志：`%s`。\n\n```bash\n' "$RUNTIME" "$logfile"
    printf '%q ' "$@"
    printf '\n```\n'
  } >> "$LOG"
  local started=$SECONDS
  if bash "$RUNTIME/scripts/run_with_status.sh" "$status" "$logfile" "$@"; then
    printf '\n结果：退出码 0，耗时 %s 秒；证据保存在上述输出路径。\n' "$((SECONDS - started))" >> "$LOG"
  else
    local code=$?
    printf '\n结果：失败，退出码 %s，耗时 %s 秒。流程停止，未跳过门禁。\n' "$code" "$((SECONDS - started))" >> "$LOG"
    return "$code"
  fi
}

idle() { bash "$RUNTIME/scripts/check_gpu_idle.sh"; }
TRAIN=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 "$RUNTIME/scripts/train_subject01.py")
VERIFY=("$PYTHON" "$RUNTIME/scripts/verify_repeatability_gate.py" --config "$CONFIG")

stage training-cache-verification "$PYTHON" "$RUNTIME/scripts/verify_training_cache.py" \
  --cache "$PROJECT_ROOT/data/derived/training/subject01_train_pool_top100.h5" \
  --project-root "$PROJECT_ROOT" \
  --manifest "$PROJECT_ROOT/data/fingerprints/training_cache_manifest.json" \
  --data-fingerprint "$PROJECT_ROOT/data/fingerprints/data_fingerprint.json" \
  --metadata "$PROJECT_ROOT/data/derived/neural_data/metadata_sub-01.npy" \
  --selection-train-ids "$PROJECT_ROOT/data/derived/splits/selection_train_ids.txt" \
  --validation-ids "$PROJECT_ROOT/data/derived/splits/validation_ids.txt" \
  --output "$PROJECT_ROOT/artifacts/migration-20260908/training-cache-verification.json"
idle
stage forward "$PYTHON" "$RUNTIME/scripts/gate_forward_alignment.py" \
  --config "$CONFIG" --output "$ARTIFACTS/forward_alignment.json"
idle
stage batch-preferred "${TRAIN[@]}" --config "$CONFIG" --run-mode gate \
  --max-updates-override 532 --output-override "$RUNS/4090-preferred-532"
idle
stage batch-fallback "${TRAIN[@]}" --config "$FALLBACK" --run-mode gate \
  --max-updates-override 532 --output-override "$RUNS/4090-fallback-532"
stage batch-verification "$PYTHON" "$RUNTIME/scripts/verify_batch_gate.py" \
  --config "$CONFIG" --preferred-run "$RUNS/4090-preferred-532" \
  --fallback-run "$RUNS/4090-fallback-532" --selected preferred \
  --output "$ARTIFACTS/batch_gate.json"

idle
stage resume-continuous "${TRAIN[@]}" --config "$CONFIG" --run-mode gate \
  --max-updates-override 100 --output-override "$RUNS/4090-continuous-100"
idle
stage resume-first "${TRAIN[@]}" --config "$CONFIG" --run-mode gate \
  --max-updates-override 50 --output-override "$RUNS/4090-resumed-100"
idle
stage resume-second "${TRAIN[@]}" --config "$CONFIG" --run-mode gate \
  --max-updates-override 100 --output-override "$RUNS/4090-resumed-100" \
  --resume "$RUNS/4090-resumed-100/checkpoints/checkpoint-update-00000050"
stage resume-verification "${VERIFY[@]}" --gate resume_equivalence \
  --left "$RUNS/4090-continuous-100/checkpoints/checkpoint-update-00000100" \
  --right "$RUNS/4090-resumed-100/checkpoints/checkpoint-update-00000100" \
  --left-aux "$RUNS/4090-continuous-100/traces" \
  --right-aux "$RUNS/4090-resumed-100/traces" --output "$ARTIFACTS/resume_equivalence.json"

SNAPSHOT="$RUNS/4090-preferred-532/snapshots/snapshot-update-00000532"
DECODES="$RUNS/4090-decode"
idle
stage decode-same-process "$PYTHON" "$HELPER" --runtime "$RUNTIME" \
  --config "$CONFIG" --action decode --snapshot "$SNAPSHOT" \
  --output "$DECODES/normal" --repeats 2
idle
stage decode-new-process "$PYTHON" "$HELPER" --runtime "$RUNTIME" \
  --config "$CONFIG" --action decode --snapshot "$SNAPSHOT" \
  --output "$DECODES/reversed" --reverse
stage decode-same-verification "${VERIFY[@]}" --gate decode_determinism \
  --left "$DECODES/normal/pass-0/decode_manifest.json" \
  --right "$DECODES/normal/pass-1/decode_manifest.json" \
  --output "$ARTIFACTS/decode_same_process.json"
stage decode-new-verification "${VERIFY[@]}" --gate decode_determinism \
  --left "$DECODES/normal/pass-0/decode_manifest.json" \
  --right "$DECODES/reversed/pass-0/decode_manifest.json" \
  --output "$ARTIFACTS/decode_determinism.json"

for repeat in a b; do
  idle
  stage "evaluate-$repeat" "$PYTHON" "$HELPER" --runtime "$RUNTIME" \
    --config "$CONFIG" --action evaluate \
    --decode-manifest "$DECODES/normal/pass-0/decode_manifest.json" \
    --output "$RUNS/4090-evaluator-$repeat"
done
stage evaluate-verification "${VERIFY[@]}" --gate evaluator_repeatability \
  --left "$RUNS/4090-evaluator-a/evaluation.json" \
  --right "$RUNS/4090-evaluator-b/evaluation.json" \
  --left-aux "$RUNS/4090-evaluator-a/per_pair.csv" \
  --right-aux "$RUNS/4090-evaluator-b/per_pair.csv" \
  --output "$ARTIFACTS/evaluator_repeatability.json"

FORMAL="$PROJECT_ROOT/configs/formal/subject01_selection.yaml"
APPROVAL="$ARTIFACTS/selection_approval.json"
mkdir -p "$(dirname "$FORMAL")"
[[ ! -e "$FORMAL" ]]
cp -- "$CONFIG" "$FORMAL"
cmp -- "$CONFIG" "$FORMAL"
stage formal-approval "$PYTHON" "$RUNTIME/scripts/create_formal_approval.py" \
  --config "$FORMAL" --output "$APPROVAL" --approve
idle
printf '\n正式 selection 启动命令即将执行。门禁权重不参与初始化；从固定 canonical 权重及全新 AdamW 开始。\n' >> "$LOG"
stage formal-selection "${TRAIN[@]}" --config "$FORMAL" --run-mode formal --approval-file "$APPROVAL"
