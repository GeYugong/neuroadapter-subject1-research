# 双 RTX 4090 确定性配置证据

本目录保存 2026-09-08 确定性候选配置的可公开审计副本。服务器原文件是运行输入；副本将项目绝对路径转换为相对路径。`INDEX.json` 同时记录原文件和公开副本 SHA，二者不应混用。

此 det-v1 候选六项门禁均通过，但正式许可被数据审计 schema 接口错误阻断，未启动正式训练。后续修复使用独立 det-v2 记录；本目录不得冒充 det-v2 的通过证据。

- 训练代码固定为 `bce13f220c30494104a397c18fd91009b1e12993`。
- 验收工具代码固定为 `4a54d9236fc40307c1ecb78e3c91369c1333a12c`。
- 实际配置为 `preferred_config.json`：双卡、每卡 microbatch 4、梯度累积 2、global batch 16，启用确定性计算及 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- `old_resume_diagnostics.json` 记录旧非确定性配置的失败，不能作为当前配置的通过证据。
- `two_fresh_runs_probe.json` 只是两个独立 2 步运行的诊断，不是恢复验收；正式恢复证据仅为 `resume_equivalence.json` 中的连续 100 步与 50+50 步完整状态比较。
- `decode_determinism.json` 和 `evaluator_repeatability.json` 只验证工程重复性。评价样本包含重复 GT，不属于正式效果指标，不能用于选择权重或与论文比较。
- `selection_approval.json` 仅在六项验收全部通过后生成。正式训练必须重新加载固定 canonical 初始化及全新优化器，不继承验收运行权重。

旧候选的硬件、batch 等记录位于 `../migration-20260908/`，不得与本目录的不同配置指纹混合使用。完整时间线、命令和运行路径见根目录 `EXPERIMENT_LOG.md`。
