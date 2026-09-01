# 正式执行顺序

本文只给出阶段顺序和命令接口。所有路径均以实验根目录 `$PROJECT_ROOT` 为基准；selection 正式配置记为 `$SELECTION_CONFIG`。当前尚未执行 GPU 门禁，不能生成 formal approval，也不能启动 formal training。

## 1. 冻结前提

1. `decoder_atlas_audit.json` 必须为 `status: verified`，并固定 CBIG commit、左右 annotation SHA、top-SNR 排序、最终 200-token vertex hash 和 `max_voxels`；
2. `configs/protocol/subject01_selection_plan.json` 与 `configs/protocol/gate_requirements.yaml` 必须保持 frozen，不允许运行时覆盖；
3. 把 selection YAML 标记为 `frozen`，将 `protocol_commit` 设置为当前干净 Git 提交；
4. canonical initialization 标记为 `frozen`；
5. selection 与 final 必须具有相同 `method_fingerprint`，只允许修改协议列出的运行字段。

`brain_encoder_parcel_audit.json` 是最终 brain encoder forward/test 的独立门禁。公开资产目前不能证明与 checkpoint 训练时使用的 parcel 文件同源，因此它不参与 decoder formal approval，但在解决前禁止标准 test 的 encoder-selected 口径。

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
  --config "$SELECTION_CONFIG" \
  --output "$PROJECT_ROOT/data/gates/hardware_gate.json"

PYTHONPATH=src "$PROJECT_ROOT/envs/neuroadapter/bin/python" \
  scripts/gate_forward_alignment.py \
  --config "$SELECTION_CONFIG" \
  --output "$PROJECT_ROOT/data/gates/forward_alignment.json"
```

硬件门禁的 `sm_120`、双 RTX 5090、BF16、NCCL、30 分钟压力时长、Xid 检查和 GPU UUID 均来自固定 `gate_requirements.yaml`。随后分别运行 `8/GPU x accumulation 1` 与 `4/GPU x accumulation 2` 至少 532 updates。两者只比较稳定性和显存，不声明严格权重等价；用 `verify_batch_gate.py` 选择一种，所选方案峰值 reserved memory 必须小于等于 29 GiB。

## 4. 恢复、解码和评价重复性

恢复门禁比较连续 100 updates 与 `50 + resume + 50` 的完整模型、optimizer、trainer、两个 rank RNG state，以及两个 rank 的 trace：

```bash
PYTHONPATH=src "$PROJECT_ROOT/envs/neuroadapter/bin/python" \
  scripts/verify_repeatability_gate.py \
  --gate resume_equivalence --config "$SELECTION_CONFIG" \
  --left CONTINUOUS_CHECKPOINT --right RESUMED_CHECKPOINT \
  --left-aux CONTINUOUS_TRACE_DIR --right-aux RESUMED_TRACE_DIR \
  --output "$PROJECT_ROOT/data/gates/resume_equivalence.json"
```

`decode_determinism` 使用两份独立输出的 `decode_manifest.json`；`evaluator_repeatability` 使用两份 evaluator JSON，并通过 `--left-aux/--right-aux` 指向对应 per-image CSV。验证器只在内容和全部 PNG/CSV SHA 完全一致时生成 `status: passed`。

## 5. Selection

每 25 reference epochs 保存一份 inference snapshot。固定计划预先列出 20 个精确 optimizer update；snapshot 的 update、模型 SHA、manifest SHA、metadata SHA、formal identity 和方法指纹均从原子 snapshot 读取，不接受手工填写。每份 snapshot 先运行：

```text
validation_loss.py --config ... --snapshot ...
decode_validation.py --config ... --snapshot ... --stage screening
evaluate_validation.py --config ... --decode-manifest ... --validation-loss ...
```

将全部一级 evaluator JSON 一次传给：

```bash
scripts/select_checkpoint.py --config "$SELECTION_CONFIG" \
  --stage shortlist --input EVAL_JSON... --output SHORTLIST_JSON
```

只对返回的 5 个 update 重新生成 8 candidates 并评价，再一次性执行：

```bash
scripts/select_checkpoint.py --config "$SELECTION_CONFIG" \
  --stage final --shortlist-manifest SHORTLIST_JSON \
  --input FIVE_EVAL_JSONS... --output FINAL_SELECTION_JSON
```

shortlist 阶段要求输入 update 集合与固定 20 点完全一致；final 阶段要求 5 个输入与 shortlist manifest 完全一致。全部脚本共同绑定 500 图及顺序、推理步数、guidance、candidate 数、评价 batch、指标源码哈希、snapshot provenance 和 `method_fingerprint`。

## 6. Approval、Final 与测试集

全部六项 GPU/重复性门禁通过、canonical manifest 为 `frozen`、decoder atlas 为 `verified` 后，生成 selection approval：

```bash
scripts/create_formal_approval.py --config "$SELECTION_CONFIG" \
  --approve --output SELECTION_APPROVAL_JSON
```

selection formal run 结束并得到 `U*` 后，从 selection config 和 final selection manifest 派生 final config：

```bash
scripts/derive_final_config.py \
  --selection-config "$SELECTION_CONFIG" \
  --selection-manifest FINAL_SELECTION_JSON \
  --train-pool-ids "$PROJECT_ROOT/data/derived/splits/train_pool_ids.txt" \
  --output-dir "$PROJECT_ROOT/runs/final/subject01-final-v1" \
  --output-config "$PROJECT_ROOT/configs/formal/subject01_final.yaml"

scripts/create_formal_approval.py \
  --config "$PROJECT_ROOT/configs/formal/subject01_final.yaml" \
  --approve --output FINAL_APPROVAL_JSON
```

final approval 会证明两份配置的 `method_fingerprint` 相同，只允许 `run_name`、`run_kind`、`split_ids`、`output_dir`、`max_updates` 和 selection 证据路径发生变化，并绑定 `U*`、final selection manifest 与完整 9000 图 train pool。Final run 从相同 canonical initialization 和新的 AdamW 开始，连续运行恰好 `U*` updates。

inference snapshot 已支持幂等恢复：同名完整 snapshot 只有在 metadata 和模型结构哈希完全相同时才复用，否则硬失败；分布式保存错误会同步到所有 rank。`export_final_model.py` 还会验证 snapshot 确实来自完成的 formal final run，再导出 PT、safetensors 和 `MODEL_LOCK.json`。

标准 test 只有在模型文件 SHA、Git HEAD、release tag、brain encoder 资产/parcel 身份和 16-member full-forward gate 同时通过后，才能由 `authorize_test_access.py` 生成访问凭据。当前 brain encoder parcel 来源门禁未通过，因此 decoder 权重可以训练和锁定，但 encoder-selected 标准 test 仍保持阻断。
