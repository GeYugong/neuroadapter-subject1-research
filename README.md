# NeuroAdapter Subject 1 Research

## 实验目的

本项目以 NSD Subject 1 为研究对象，从公开原始数据和固定上游代码开始，建立一套可审计、可恢复、可重复的 NeuroAdapter 研究基础。完整研究目的包括：

1. 重新构建并严格验证 Subject 1 的训练与测试数据；
2. 训练、选择并永久冻结一个可信的 NeuroAdapter 正式权重；
3. 以同一权重为基础，研究不同脑区对图像重建的贡献、类别特异性、分布式冗余及模型内部注意机制；
4. 为后续结果提供统一的数据指纹、模型哈希、随机性控制、评价协议和连续实验记录。

当前只执行第 1 个实验阶段：**训练并冻结正式权重**。后续研究方向仅用于说明项目总目标，其具体实验方案尚未制定，也不属于当前阶段。

## 当前阶段边界

**2026-09-10 决定更新：不再进行任何续训或 9000 图全量重训。** 保留原定 500 图验证、20 候选初筛和 5 候选复评规则，从已经完成的 selection 权重中选定一个，汇报后直接用于后续研究。该权重的真实训练样本数仍为 8500，不改写为全量训练模型。公开 HF 备份只使用 `gugabobo` 项目独立凭据。下方完整阶段列表为原始计划，涉及全量重训的步骤现已取消。

当前阶段包含：

- 从官方来源重新下载数据、代码和模型资产；
- 独立转换并验证 Subject 1 数据；
- 建立项目专用运行环境和可完整恢复的训练器；
- 使用内部验证集选择训练时长；
- 使用全部 9000 张训练图重新训练正式权重；
- 在模型锁定后进行一次标准测试集评价；
- 导出唯一的正式权重及其完整审计材料。

自 2026-09-08 起，项目迁移到双 RTX 4090 服务器，旧 RTX 5090 服务器仅保留迁移源，不再承载后续运行。任何服务器上其他人的任务都不得被终止、暂停或抢占。迁移校验及正式训练门禁全部通过后才允许启动 selection。

## 证据优先级

发生实现歧义时按以下顺序处理：

1. ICLR 2026 论文 v2 的科学方法和正式报告口径；
2. NeuroAdapter 固定提交 `ae07c183844f298f5dee4c002f1439f23285c6d9`；
3. whole_brain_encoder 固定提交 `767f25afc2240f568f4db8c0ce09604e6f83aa72`；
4. 经过测试并在 `docs/DEVIATIONS.md` 中明确记录的必要工程修复。

## 记录制度

根目录的 `EXPERIMENT_LOG.md` 是本项目唯一的连续实验日志。所有下载、转换、测试、训练、恢复、解码、评价、异常和科研决策必须按时间追加到该文件，不得覆盖既有历史。

其他文档只承担固定职责：

- `docs/PHASE1_TRAINING_PROTOCOL.md`：当前训练阶段的冻结协议；
- `docs/FORMAL_EXECUTION.md`：正式门禁、selection、final 与模型锁的执行顺序；
- `docs/DEVIATIONS.md`：相对论文或上游实现的已知差异；
- `docs/TEST_ACCESS_POLICY.md`：标准测试集访问规则；
- `manifests/`：来源、文件、环境、数据和模型哈希。

## 数据与权重管理

Git 只跟踪代码、配置、小型清单、统计摘要和文档。原始数据、转换数据、模型缓存、训练 checkpoint、完整重建图片和私密凭据均不得提交到 Git。

## 可公开复核的冻结证据

`manifests/frozen/` 保存服务器丢失后仍需保留的完整小型证据，包括：9000/1000 与 8500/500 实际 ID、完整 9000 图 train pool、parcel token map、原始 NSD 文件 SHA、数据指纹、训练缓存清单、decoder atlas 审计、brain encoder parcel/权重审计、模型资产树、canonical initialization manifest、非有限值审计和 NSD 图像映射审计。所有服务器绝对路径均已转换为项目相对路径；beta、HDF5、模型权重和 checkpoint 不进入 Git。

`manifests/frozen/INDEX.json` 固定每个公开清单的来源与 SHA-256。服务器上的运行清单是正式训练输入，Git 中的副本用于长期审计；每次正式输入变化后必须重新导出并提交。

## 可迁移运行方式

仓库不保存个人服务器路径。shell 脚本要求调用者显式设置实验根目录：

```bash
export PROJECT_ROOT=/absolute/path/to/neuroadapter-subject1-research
bash "$PROJECT_ROOT/repo/scripts/download_nsd_subj01.sh"
```

Python 下载脚本同样要求 `--project-root "$PROJECT_ROOT"`。训练 YAML 只在顶层设置一次绝对 `project_root`，其余路径全部相对于该目录。

## 当前正式训练状态

正式 selection 已于 **2026-09-10 09:28:49 完成**，共 265625 updates / 500 epochs，退出码 0，耗时 42 小时 18 分钟。训练代码固定为 `1a1fcfa66e06de07a04dfbb48cc6f9ad108ed567`，运行名为 `subject01-selection-4090-deterministic-v2`；后续报告提交不改动冻结运行目录。

- 8500 张训练图、500 张内部验证图，200 个 parcel；标准测试集不参与选择。
- 双卡每卡 microbatch 4、梯度累积 2、global batch 16；BF16、确定性计算、AdamW、学习率 `1e-4`。
- 从固定 canonical 初始化和全新优化器开始，selection 上限 265625 updates，固定保留 20 个验证候选 snapshot。
- 运行目录：`runs/selection/subject01-selection-4090-deterministic-v2`；实际正式配置：`configs/formal/subject01_selection_v2.yaml`；训练记录：该运行目录的 `training.jsonl`。
- 可公开复核证据见 `manifests/deterministic-4090-v2/`，连续时间线见 `EXPERIMENT_LOG.md`。此前 det-v1 的许可失败记录保留，不能作为本轮正式训练记录。

当前任务是备份现有 20 个权重，并按原定内部验证规则直接选定研究用权重。`scripts/select_existing_weights.py` 只允许调用四个验证/选优工具，不包含训练入口；结束时生成 `RESEARCH_WEIGHT_LOCK.json` 和报告，不调用面向全量重训的 `export_final_model.py`，不生成冒充 9000 图模型的锁定材料。

decoder 训练所用 atlas 已单独完成审计：CBIG 来源、左右 annotation、每侧 500 parcels、top-SNR 排序、最终 200-token 顺序和 `max_voxels=626` 均可验证，当前 9000 图训练缓存无需重建。公开 `whole_brain_encoder` parcel 文件与作者内部训练资产的关系仍无法由公开材料证明，因此该问题只阻断模型锁定后的 brain encoder forward/test 门禁，不再错误阻断 decoder 的 selection/final 训练。

正式执行还受到以下条件约束：固定 GPU 门禁全部通过、正式 YAML 与 protocol commit 冻结、canonical initialization 标记为 `frozen`、selection/final approval 闭合。仓库按研究协作需要保持公开。

## 当前服务器入口

本机 SSH 别名：`neuroadapter-4090`（中文别名 `双卡4090`）。VS Code Remote-SSH 连接后打开：

```text
/data1/matengyu/geyugong/neuroadapter-subject1-research/repo
```

实验根目录下 `data/`、`models/`、`envs/`、`runs/`、`logs/`、`configs/` 与 `repo/` 并列，全部属于同一次研究项目。完整迁移流程和验收证据见 `docs/SERVER_MIGRATION.md` 与主日志。SSH 密码、私钥和代理凭据不进入 Git。
