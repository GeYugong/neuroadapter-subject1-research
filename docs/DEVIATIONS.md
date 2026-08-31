# 第一阶段已知差异

本文档只记录训练并冻结正式权重阶段相对论文文字或固定上游实现的差异。

## D001：Global batch 采用论文口径

论文 reproducibility statement 报告 batch size 16。固定上游 shell 将 `train_batch_size=16` 与 2 个 Accelerate processes、`split_batches=False` 同时使用，按 Accelerate 语义可能得到 global batch 32。

本阶段优先采用论文文字：

```text
global batch = 16
per-GPU batch = 8
gradient accumulation = 1
```

若正式训练前出现 OOM，只允许切换为保持 global batch 16 的 `4/GPU x accumulation 2`，并在正式 selection run 前永久冻结。

## D002：BF16 是明确冻结的工程选择

固定提交的 `acc_config.yaml` 写 BF16，而参数解析器默认 FP16，shell 未显式传递该参数，因此无法仅凭公开仓库确定论文正式权重的精度。

本阶段在 RTX 5090 上优先测试 BF16。只有通过 forward、梯度、NCCL、恢复一致性和压力测试后才将其写入最终环境锁。正式报告不把 BF16 描述为已经证实的作者原始配置。

## D003：使用可完整恢复的新训练器

固定上游训练器不提供完整的 mid-run load 路径，也没有显式保存 sampler cursor、每 rank RNG 和 accumulation 状态。本阶段保持模型结构和 loss 不变，增加完整状态恢复、原子 checkpoint、配置哈希与确定性门禁。

## D004：内部 validation 不使用 brain encoder 选候选

公开 whole-brain encoder 使用其 validation split 建立 voxel confidence，并明确建议只在 test 或未见数据上评价。新划出的 500 张 validation 来自原 9000 张 decoder 训练池，不能把现成 encoder 当成完全独立的 checkpoint 选择器。

因此 validation 使用固定候选和图像指标选择训练时长。brain encoder 只用于模型锁定后的标准 test 论文口径评价，以及不参与决策的附加诊断。

## D005：Final run 按 optimizer updates 迁移

selection 使用 8500 张图，final 使用 9000 张图。直接复用同一 epoch 数会增加约 5.88% 的样本曝光量。因此 selection 记录最优 checkpoint 对应的 `U* optimizer updates`，final 使用全部 9000 张图训练恰好 `U*` 次更新。

## D006：`max_voxels=626` 作为数据指纹

626 是旧转换流程得到的结果，不是论文规定的超参数。本阶段由固定 annotation 和转换程序计算实际最大 parcel vertex 数；若结果不是 626，则停止并检查数据、medial wall 和 label 索引，而不是强制修改数据满足 626。

## D007：禁用上游明文外部服务凭据

固定上游训练脚本包含明文 W&B 登录凭据。该凭据不复制、不使用、不写入日志。W&B 默认关闭；任何外部服务凭据只能通过运行时环境变量提供。

## D008：保留并审计官方 beta 中未选中顶点的 NaN

官方 Subject 1 `rh.betas_session11.mgh` 在 10 个固定顶点的全部 750 个 trial 中包含 NaN，共 7500 个非有限值。下载文件大小与官方 S3 inventory 一致；这些顶点全部属于右半球 Schaefer parcel 320（不含 medial wall 的零基索引），该 parcel 的 mean-ncsnr 排名为 461，与正式 top-100 parcel 无交集。

本阶段不对源值执行 `nan_to_num`。HDF5 转换原样保留 NaN，并生成 `source_nonfinite_values.json`。完整数据指纹扫描必须证明所有正式选中顶点均为有限值；训练缓存只抽取经过该门禁的 top-100 parcel。若后续任一选中顶点出现非有限值，数据准备立即失败。
