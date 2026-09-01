# NeuroAdapter Subject 1 Research

## 实验目的

本项目以 NSD Subject 1 为研究对象，从公开原始数据和固定上游代码开始，建立一套可审计、可恢复、可重复的 NeuroAdapter 研究基础。完整研究目的包括：

1. 重新构建并严格验证 Subject 1 的训练与测试数据；
2. 训练、选择并永久冻结一个可信的 NeuroAdapter 正式权重；
3. 以同一权重为基础，研究不同脑区对图像重建的贡献、类别特异性、分布式冗余及模型内部注意机制；
4. 为后续结果提供统一的数据指纹、模型哈希、随机性控制、评价协议和连续实验记录。

当前只执行第 1 个实验阶段：**训练并冻结正式权重**。后续研究方向仅用于说明项目总目标，其具体实验方案尚未制定，也不属于当前阶段。

## 当前阶段边界

当前阶段包含：

- 从官方来源重新下载数据、代码和模型资产；
- 独立转换并验证 Subject 1 数据；
- 建立项目专用运行环境和可完整恢复的训练器；
- 使用内部验证集选择训练时长；
- 使用全部 9000 张训练图重新训练正式权重；
- 在模型锁定后进行一次标准测试集评价；
- 导出唯一的正式权重及其完整审计材料。

当前准备工作结束后停在正式训练启动之前。两张 RTX 5090 上已有的其他任务不得被终止、暂停或抢占。

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

`manifests/frozen/` 保存服务器丢失后仍需保留的完整小型证据，包括：9000/1000 与 8500/500 实际 ID、parcel token map、原始 NSD 文件 SHA、数据指纹、训练缓存清单、模型资产树、brain encoder 资产审计、canonical initialization manifest、非有限值审计和 NSD 图像映射审计。所有服务器绝对路径均已转换为项目相对路径；beta、HDF5、模型权重和 checkpoint 不进入 Git。

`manifests/frozen/INDEX.json` 固定每个公开清单的来源与 SHA-256。服务器上的运行清单是正式训练输入，Git 中的副本用于长期审计；每次正式输入变化后必须重新导出并提交。

## 可迁移运行方式

仓库不保存个人服务器路径。shell 脚本要求调用者显式设置实验根目录：

```bash
export PROJECT_ROOT=/absolute/path/to/neuroadapter-subject1-research
bash "$PROJECT_ROOT/repo/scripts/download_nsd_subj01.sh"
```

Python 下载脚本同样要求 `--project-root "$PROJECT_ROOT"`。训练 YAML 只在顶层设置一次绝对 `project_root`，其余路径全部相对于该目录。

## 当前正式训练状态

正式配置仍为 `draft`，两张 RTX 5090 门禁尚未执行，正式训练未启动。新增 atlas 审计还发现：LH 顶点集合等价但 token 顺序不同，RH 与 CBIG 派生 atlas 不匹配，且公开上游 RH 资产与其 LH 资产集合相同。在作者实际 token 顺序和 RH 资产来源明确、相关数据指纹重新确认以及全部 approval 门禁通过之前，训练器不得进入 formal mode。
