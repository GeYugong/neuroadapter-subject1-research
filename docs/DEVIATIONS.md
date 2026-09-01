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

因此 validation 使用固定候选和图像指标选择训练时长。一级评价对每张图使用 2 个固定 candidate 以降低单 seed 偶然性，二级评价仅对 5 个 shortlist checkpoint 使用完整 8 候选。brain encoder 只用于模型锁定后的标准 test 论文口径评价，以及不参与决策的附加诊断。

## D005：Final run 按 optimizer updates 迁移

selection 使用 8500 张图，final 使用 9000 张图。直接复用同一 epoch 数会增加约 5.88% 的样本曝光量。因此 selection 记录最优 checkpoint 对应的 `U* optimizer updates`，final 使用全部 9000 张图训练恰好 `U*` 次更新。由于 global batch 保持 16，总样本曝光量也固定为 `I*=16 x U*`；epoch 只作为派生量记录。

## D006：`max_voxels=626` 作为数据指纹

626 是旧转换流程得到的结果，不是论文规定的超参数。本阶段由固定 annotation 和转换程序计算实际最大 parcel vertex 数；若结果不是 626，则停止并检查数据、medial wall 和 label 索引，而不是强制修改数据满足 626。

## D007：禁用上游明文外部服务凭据

固定上游训练脚本包含明文 W&B 登录凭据。该凭据不复制、不使用、不写入日志。W&B 默认关闭；任何外部服务凭据只能通过运行时环境变量提供。

## D008：保留并审计官方 beta 中未选中顶点的 NaN

官方 Subject 1 `rh.betas_session11.mgh` 在 10 个固定顶点的全部 750 个 trial 中包含 NaN，共 7500 个非有限值。下载文件大小与官方 S3 inventory 一致；这些顶点全部属于右半球 Schaefer parcel 320（不含 medial wall 的零基索引），该 parcel 的 mean-ncsnr 排名为 461，与正式 top-100 parcel 无交集。

本阶段不对源值执行 `nan_to_num`。HDF5 转换原样保留 NaN，并生成 `source_nonfinite_values.json`。完整数据指纹扫描必须证明所有正式选中顶点均为有限值；训练缓存只抽取经过该门禁的 top-100 parcel。若后续任一选中顶点出现非有限值，数据准备立即失败。

## D009：500 reference epochs 是搜索上限，不是论文训练时长

论文 reproducibility statement 报告 300 epochs；固定上游 shell 使用 500 epochs。本阶段把 500 reference epochs 作为 selection 的预设搜索上限，通过内部 validation 选择实际 `U*`，不声称作者正式权重训练了 500 epochs。

## D010：8500/500 内部 validation 是新增设计

论文和公开训练脚本没有当前这份固定的 8500/500 划分。本阶段从 9000 张 Subject 1 train pool 中建立内部 validation，只用于选择训练时长；final run 仍使用全部 9000 张训练图。

## D011：selection 与 final 复用同一 canonical initialization

为隔离训练集范围造成的差异，两轮训练从完全相同的 adapter 初始化权重开始。该约束是本项目新增的控制措施，不是论文明确报告的流程。

## D012：sampler 允许 global batch 跨越 permutation 边界

无丢样本的确定性 sampler 将无限 epoch permutation 串接为样本流，因此一个 global batch 可以包含前一 permutation 的末尾和下一 permutation 的开头。reference epoch 仅表示累计样本曝光量。

## D013：batch fallback 不声明严格权重等价

`8/GPU x 1` 与 `4/GPU x accumulation 2` 具有相同样本顺序和 global batch，但当前随机张量按 microbatch shape 生成，不能保证每个样本获得完全相同的 VAE epsilon、diffusion noise、timestep 和 token mask。因此不再要求两种配置权重逐步等价；只分别做数值稳定性检查，并在 selection 前冻结一种配置，selection 与 final 不得切换。

## D014：执行后端显式冻结

当前候选配置明确使用 `allow_tf32=true`、`cudnn_benchmark=false`、`deterministic_algorithms=false`、`adamw_fused=false`、`adamw_foreach=false`。这些是待 RTX 5090 门禁验证的工程选择，不作为作者原始设置。selection 与 final 必须保持一致。

## D015：图像 resize 实现不同

本项目数据集使用 torchvision v2 的 bilinear resize 并显式设置 `antialias=true`；固定上游代码使用其当时环境中的 torchvision transform。该实现差异已冻结并纳入环境锁。

## D016：新增 fixed 与 seed-mean 报告口径

最终评价除论文口径的 brain-encoder-selected candidate 外，还报告 candidate 0 和 8-seed mean。后两项用于减少候选选择器对模型质量解释的混淆，属于新增报告口径。

## D017：CBIG 与公开 whole_brain_encoder Schaefer 资产不等价

逐 hemisphere、逐 label 的 fsaverage vertex membership 审计结果为：LH 的 501 个顶点集合完全相同，但上游 label 顺序与 CBIG annotation 顺序不同，按索引仅 7/501 相同；RH 与 CBIG 派生集合的交集为 0/501，并且公开 `whole_brain_encoder` 的 LH/RH 文件包含完全相同的 501 个顶点集合。当前训练缓存来自固定 CBIG annotation，尚未改写。必须先确定作者训练时的真实 token 顺序和 RH 资产来源，再决定是否重建缓存；重新验证数据指纹前不得启动正式训练。

## D018：固定上游训练脚本不能整体导入

固定提交中的 `train_brain_adapter.py` 导入了当前 `brain_adapter.dataset` 不提供的 `nsd_groupwise_topk_parcel_dataset`，因此无法作为 Python 模块直接导入。forward alignment gate 通过 AST 从该固定文件中提取唯一的 `setup_ip_adapter()` 原始函数体，在显式提供其原始依赖后执行；gate 同时记录完整训练脚本和函数体 SHA-256。该处理只隔离无关的坏 import，不重写作者的 adapter 构造函数。
