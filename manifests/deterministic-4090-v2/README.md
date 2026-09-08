# 正式 Selection 启动证据

本轮固定训练和验收代码为 `1a1fcfa66e06de07a04dfbb48cc6f9ad108ed567`。六项门禁在此提交及 det-v2 配置下全部重新运行，没有修改或复用 det-v1 通过记录的指纹。

`INDEX.json` 记录服务器原文件与相对路径公开副本的 SHA；公开副本不是服务器实际运行配置。环境包和确定性后端不变，环境锁公开副本位于 `../deterministic-4090/requirements-freeze.txt`。

关键证据：

- `cpu_preflight.json`：真实缓存、图像映射和 decoder atlas 审计接口通过。
- `hardware_gate.json`：双卡 30 分钟 BF16/NCCL 压力测试及 Xid 检查。
- `forward_alignment.json`、`batch_gate.json`：上游 forward 对齐，两种 532 步 batch 配置实测与首选配置冻结。
- `resume_equivalence.json`：连续 100 步与 50+50 步的完整状态及两卡 traces 一致。
- `decode_same_process.json`、`decode_determinism.json`：同进程重复、跨进程倒序生成的 64 张候选 PNG 一致。
- `evaluator_repeatability.json`：两次八项指标及逐样本 CSV 一致。此处为工程样本，不能当正式效果指标。
- `selection_approval.json`、`effective_run.json`、`formal_start_verified.json`：正式许可、真实启动身份、从 canonical 第 0 步开始及启动阶段有限值检查。
- `calibration_code_equivalence.json`：额外诊断证明审计接口修复前后，532 步首选校准的数值记录和 snapshot 完全一致；不替代本轮任何门禁。

正式训练于 2026-09-08 15:10:49 启动。这些材料证明启动验收通过，不代表训练完成、图像质量达到论文水平或最终模型已经锁定。后续进展按时间追加到主日志。
