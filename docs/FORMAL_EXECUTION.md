# 正式执行顺序

本文只给出阶段顺序和命令接口。所有路径均以实验根目录 `$PROJECT_ROOT` 为基准；实际 frozen config 记为 `$CONFIG`。当前 Schaefer 审计为 `mismatch`，以下 GPU 门禁可以准备，但 approval 和 formal training 不得执行。

## 1. 冻结前提

1. 明确作者实际使用的 Schaefer token 顺序和 RH membership；
2. 使 `schaefer_upstream_equivalence.json` 达到 `status: indexed_equivalent`；
3. 若 parcel 输入改变，重建 training cache、data fingerprint 和 canonical initialization；
4. 把正式 YAML 标记为 `frozen`，将 `protocol_commit` 设置为当前干净 Git 提交；
5. selection 与 final 均固定同一 backend、GPU 数、global batch 和 canonical initialization。

## 2. CPU 门禁

```bash
cd "$PROJECT_ROOT/repo"
PYTHONPATH=src "$PROJECT_ROOT/envs/neuroadapter/bin/python" -m pytest -q
```

fresh run 和 resume 均由训练器全量验证 39 GB stimuli SHA、Stable Diffusion 13 个运行时文件以及五个 vendor HEAD。缓存目录 `.cache/huggingface/` 被显式排除；其他未列入 manifest 的模型文件仍会导致失败。

## 3. GPU 门禁

两张 GPU 同时空闲后执行：

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/gate_hardware.py \
  --config "$CONFIG" \
  --output "$PROJECT_ROOT/data/gates/hardware_gate.json"

PYTHONPATH=src "$PROJECT_ROOT/envs/neuroadapter/bin/python" \
  scripts/gate_forward_alignment.py \
  --config "$CONFIG" \
  --output "$PROJECT_ROOT/data/gates/forward_alignment.json"
```

随后分别运行 `8/GPU x accumulation 1` 与 `4/GPU x accumulation 2` 至少 532 updates。两者只比较稳定性和显存，不声明严格权重等价；用 `verify_batch_gate.py` 选择一种，并保证正式 YAML 的 SHA 与所选运行一致。

## 4. 恢复、解码和评价重复性

恢复门禁比较连续 100 updates 与 `50 + resume + 50` 的完整模型、optimizer、trainer、两个 rank RNG state，以及两个 rank 的 trace：

```bash
PYTHONPATH=src "$PROJECT_ROOT/envs/neuroadapter/bin/python" \
  scripts/verify_repeatability_gate.py \
  --gate resume_equivalence --config "$CONFIG" \
  --left CONTINUOUS_CHECKPOINT --right RESUMED_CHECKPOINT \
  --left-aux CONTINUOUS_TRACE_DIR --right-aux RESUMED_TRACE_DIR \
  --output "$PROJECT_ROOT/data/gates/resume_equivalence.json"
```

`decode_determinism` 使用两份独立输出的 `decode_manifest.json`；`evaluator_repeatability` 使用两份 evaluator JSON，并通过 `--left-aux/--right-aux` 指向对应 per-image CSV。验证器只在内容和全部 PNG/CSV SHA 完全一致时生成 `status: passed`。

## 5. Selection

每 25 reference epochs 保存一份 inference snapshot。每份 snapshot 先运行：

```text
validation_loss.py
decode_validation.py --candidate-count 2
evaluate_validation.py
```

将全部一级 evaluator JSON 一次传给：

```bash
scripts/select_checkpoint.py --stage shortlist --input EVAL_JSON...
```

只对返回的 5 个 update 重新生成 8 candidates 并评价，再一次性执行：

```bash
scripts/select_checkpoint.py --stage final --input FIVE_EVAL_JSONS...
```

最终选择器强制同一 config、500 图 ID、评价资产和负样本池，并要求 final 阶段恰好包含 5 个 checkpoint。

## 6. Approval、Final 与测试集

全部六项 GPU/重复性门禁通过、canonical manifest 为 `frozen`、Schaefer 为 indexed-equivalent 后，显式生成 approval：

```bash
scripts/create_formal_approval.py --config "$CONFIG" --approve --output APPROVAL_JSON
```

selection formal run 结束并得到 `U*` 后，final run 从相同 canonical initialization 和新的 AdamW 开始，在全部 9000 张训练图上连续运行恰好 `U*` updates。`export_final_model.py` 验证原子 snapshot，导出 PT 与 safetensors，并生成 `MODEL_LOCK.json`。

标准 test 只有在模型文件 SHA、Git HEAD、release tag 和 16-member brain encoder full-forward gate 同时通过后，才能由 `authorize_test_access.py` 生成访问凭据。
