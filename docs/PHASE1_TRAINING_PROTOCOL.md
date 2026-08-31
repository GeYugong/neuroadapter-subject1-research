# 第一阶段协议：训练并冻结 Subject 1 正式权重

## 1. 阶段目标

本阶段从公开原始数据和固定上游提交开始，得到一个可作为后续全部研究共同基础的 Subject 1 NeuroAdapter 权重。正式权重必须满足：

- 数据来源、转换过程和 parcel 映射可追溯；
- 训练配置从开始到结束保持一致；
- 中断恢复不会重置 optimizer、sampler 或随机状态；
- checkpoint 选择不使用可能存在评价依赖的 whole-brain encoder；
- 标准 test 不参与训练长度或 checkpoint 选择；
- 最终模型由 SHA-256 和 `MODEL_LOCK.json` 永久锁定。

本文件只规定训练权重阶段。后续脑区分析、注意力分析和因果干预不在本协议中展开。

## 2. 固定输入

### 2.1 上游代码

| 组件 | 仓库 | 固定提交 |
| --- | --- | --- |
| NeuroAdapter | `https://github.com/kriegeskorte-lab/NeuroAdapter.git` | `ae07c183844f298f5dee4c002f1439f23285c6d9` |
| whole_brain_encoder | `https://github.com/kriegeskorte-lab/whole_brain_encoder.git` | `767f25afc2240f568f4db8c0ce09604e6f83aa72` |

上游仓库作为只读依赖保留。本项目的修复和扩展放在独立源码目录中，不直接形成无法审计的 vendor 修改。上游训练脚本包含明文 W&B 凭据，该凭据不复制、不调用，W&B 默认关闭。

### 2.2 NSD 数据

Subject 1 使用以下公开资产：

- `nsd_expdesign.mat`；
- `nsd_stim_info_merged.csv`；
- `nsd_stim_info_merged.pkl`；
- `nsd_stimuli.hdf5`；
- `nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/`；
- 左右半球 40 个 session beta 与 `ncsnr.mgh`。

每个 S3 object key、文件大小、下载时间和 SHA-256 均写入资产清单。

### 2.3 Schaefer parcellation

使用 ThomasYeoLab/CBIG 的 FreeSurfer `fsaverage`：

```text
Schaefer2018_1000Parcels_7Networks_order.annot
```

左右半球分别转换，索引 0 只允许表示 medial wall。转换结果必须保留每个 parcel 的 vertex 列表和内容哈希。

### 2.4 基础模型和评价资产

训练使用固定 snapshot 的 Stable Diffusion v1.5。正式评价所需的 brain encoder、CLIP、AlexNet、InceptionV3、EfficientNet-B1、SwAV 和 DINOv2 均固定代码 revision、权重 revision、预处理参数和 SHA-256。正式运行阶段禁止在线动态下载。

## 3. 数据转换协议

### 3.1 Trial 层硬门禁

必须同时满足：

```text
sessions                    40
trials per session          750
total trials                30000
vertices per hemisphere     163842
selected parcel NaN / Inf   0
```

官方源文件中的非有限值不得静默替换。转换结果必须原样保留源值，并生成逐文件、trial 和 vertex 审计；只有全部正式 top-SNR parcel 顶点通过完整有限值扫描后，训练数据门禁才算通过。位于未选中 parcel 的源异常必须记录，但不构成修改原始数据的理由。

### 3.2 图像映射硬门禁

必须验证：

- Subject 1 共 10000 个唯一 NSD image ID；
- 每个 image ID 恰好呈现 3 次；
- 三次呈现合计 30000 trials；
- MATLAB 1-based ID 与 HDF5 0-based index 转换明确；
- 至少随机抽查 100 个 image ID 的 session、trial、master ordering、NSD index 和实际图像内容。

### 3.3 重复平均

同一图片的三次 fMRI response 先平均，再进入 NeuroAdapter。输入、累加和输出 dtype 必须冻结并记录；selection 与 final 之间不得改变。

### 3.4 数据划分

首先重建论文标准划分：

```text
train pool    9000 unique images
test          1000 shared images
intersection  0
```

随后仅在 9000 张 train pool 内进行固定的图像级划分：

```text
selection_train  8500
validation        500
split_seed        20260901
```

划分发生在唯一图片层面，不得在 30000 个 trial 层面切分。生成并冻结三组 image ID 清单及其 SHA-256。

### 3.5 Parcel 选择

- 每侧应有 500 个非 medial-wall Schaefer parcels；
- 使用官方 Subject 1 `lh/rh.ncsnr.mgh`，按 parcel 内 vertex 的 mean ncsnr 排序；
- 每侧选择 top 100，共形成 200 个 model tokens；
- selection、validation、final 和 test 使用同一 parcel 顺序；
- `max_voxels` 由实际 vertex membership 计算，旧结果 626 只作为预期指纹，不得人为截断或补齐来满足该数值。

生成 `parcel_token_map.csv`，至少包含：

```text
model token
hemisphere
Schaefer label ID
SNR rank
mean ncsnr
vertex count
vertex-list SHA-256
```

`ncsnr.mgh` 是固定的扫描质量/可靠性元数据，不从新划出的 8500/500 split 重新估计。该选择复现公开数据管线，只用于固定输入维度，不参与 checkpoint 比较。

## 4. 环境与硬件门禁

初始候选环境为：

```text
Python       3.11
PyTorch      2.11.0+cu128
torchvision  0.26.0
NumPy        1.26.4
precision    BF16
```

Accelerate 和其他依赖只在完整兼容性测试通过后冻结。必须记录 OS、kernel、driver、CUDA runtime、cuDNN、NCCL、GPU UUID、软件版本和 wheel SHA-256。

硬件门禁：

- 恰好识别 2 张 RTX 5090；
- PyTorch wheel 原生支持 `sm_120`；
- BF16 matmul、卷积和 forward/backward 通过；
- NCCL all-reduce 通过；
- 双卡压力测试至少 30 分钟；
- 无 Xid、NCCL timeout 和数值异常；
- 首选配置峰值 reserved memory 不超过 29.0 GiB/GPU。

## 5. 科学方法与正式配置

保持论文和固定上游代码的核心方法：

```text
subject                  1
architecture             linear_projection
parcels                  top 100 / hemisphere
condition dimension      768
base model               Stable Diffusion v1.5
text condition           empty prompt
trainable modules        parcel mapper + IP-style cross-attention
token dropout            upstream distribution
Min-SNR gamma             5.0
optimizer                 AdamW
learning rate             1e-4
betas                     (0.9, 0.999)
epsilon                   1e-8
weight decay              1e-6
scheduler                 none
gradient clipping         1.0
precision                 BF16 after gate
global batch              16
preferred micro batch     8/GPU, accumulation 1
OOM fallback              4/GPU, accumulation 2
selection upper bound     500 reference epochs
```

global batch 16 以论文 reproducibility statement 为优先。官方 shell 在双进程和 `split_batches=False` 下可能形成 global batch 32，该差异写入 `docs/DEVIATIONS.md`。

OOM fallback 只能在正式 selection run 前决定一次。selection 和 final 必须使用相同 GPU 数、global batch、micro batch、accumulation、精度和 optimizer 实现。

## 6. 随机性与初始化

建立并冻结一份 canonical adapter initialization。selection 与 final 都从该完全相同的初始化文件开始，避免初始化依赖进程启动顺序。

随机流分离：

```text
model initialization
distributed sampler
training noise and timestep per rank
token dropout per rank
validation generation namespace
test generation namespace
```

每个评价候选 seed 由协议版本、split、NSD image ID 和 candidate index 的 SHA-256 确定。VAE latent epsilon、初始 diffusion noise 和 scheduler 随机噪声均由该样本自己的随机流控制，不得依赖全局执行顺序。

## 7. Checkpoint 与恢复协议

完整 checkpoint 至少保存：

- 所有可训练权重；
- AdamW 完整状态；
- global optimizer update；
- logical epoch 和 images seen；
- sampler permutation、epoch 和 cursor；
- 每个 rank 的 Python、NumPy、CPU torch 和 CUDA RNG；
- gradient accumulation 状态；
- config、数据指纹、源码 commit 和初始化权重哈希。

checkpoint 只在 optimizer update 边界保存。收到 SIGTERM 时只设置退出标记，在当前 update 完成后执行全 rank barrier、原子保存、校验和写入 `COMPLETE` 标记，再安全退出。

保存采用临时目录、fsync、重新加载验证、哈希计算和原子 rename。任何缺少 `COMPLETE` 的目录均不得恢复。

## 8. 正式训练前验证

正式训练前必须全部通过：

1. 冻结模块与可训练模块审计；
2. 新训练器和固定上游 forward/loss 对齐；
3. `8/GPU x 1` 与 `4/GPU x 2` 的有效 global batch 等价测试；
4. 连续 100 updates 与 `50 + save + 新进程恢复 + 50` 等价测试；
5. 一个完整 reference epoch 的吞吐和显存测试；
6. 8 张固定 validation 图片、每张 8 candidates 的端到端解码；
7. 同进程重跑、进程重启和图片顺序改变后的 PNG SHA 一致性；
8. 固定 20 对 GT/prediction 的 evaluator 重跑一致性。

样本 ID、timestep、noise 和 dropout checksum 必须完全一致。BF16/DDP 权重若不能 bitwise 一致，必须在正式运行前冻结严格数值容差。所有门禁权重在验证后删除，不得进入正式结果。

## 9. Selection run

### 9.1 训练

```text
training data       8500 selection_train images
validation data      500 images
maximum duration     500 reference epochs
initialization       canonical adapter initialization
optimizer state      new and continuous
```

训练中不得修改 GPU 数、batch、precision、LR、dropout 或数据顺序规则，也不得通过重启重置 optimizer。reference epoch 使用固定 global batch 并记录实际 optimizer updates，最终训练长度以 update 数而非 epoch 数表示。

### 9.2 一级验证

- 每 5 reference epochs 计算固定随机输入的 deterministic validation loss；
- 每 25 reference epochs 保存 inference snapshot；
- 对全部 500 张 validation 图片生成 2 个固定 candidate；
- 使用 50 denoising steps、guidance scale 4.0；
- 计算完整八项官方图像指标；
- 不使用 whole-brain encoder 选择候选。

两个 candidate index 分别形成完整的 500 图评价集，再对两套指标求平均。该设计避免单一扩散随机状态主导一级排序，同时将生成成本限制在完整八候选评价的四分之一。

### 9.3 Shortlist

一级验证完成后得到 5 个候选 checkpoint：

```text
SemanticScore 前 3 名
+ LowLevelRank 最低者
+ deterministic validation loss 最低者
```

其中：

```text
LowLevelRank = mean(rank(PixCorr), rank(SSIM), rank(AlexNet-2))
```

所有 rank 均按“越高越好”的方向计算。若低层或 loss 候选已在前三，则按 SemanticScore 顺序补足到 5 个，候选不得少于 5 个。

### 9.4 二级验证

五个 checkpoint 分别生成：

```text
500 validation images x 8 fixed candidates
```

每个 candidate index 单独形成一套完整的 500 图评价集，随后对 8 套指标求平均。统计单位为图片，8 个 seed 是同一图片内重复，使用 image-level paired cluster bootstrap。

whole-brain encoder 可以生成附加诊断，但不得参与 shortlist、主分数或最终 checkpoint 选择。

### 9.5 选择规则

主分数：

```text
SemanticScore = mean(AlexNet-5, Inception, CLIP)
```

选择顺序：

1. 找到 SemanticScore 最高的 checkpoint；
2. 对图像进行 paired cluster bootstrap，并将与最佳值的差小于等于配对差值一个标准误的 checkpoint 纳入 one-standard-error 候选集合；
3. 在集合内使用低层指标和高层指标各占 50% 的 BalancedRank；
4. 若仍并列，选择 deterministic validation loss 更低者；
5. 若仍并列，选择更早的 optimizer update。

最终记录：

```text
selected snapshot
selected reference epoch
selected global optimizer update U*
selected images seen
selection manifest SHA-256
```

如果最优点位于预设 500-epoch 上限，只记录该边界事实，不在看到结果后临时延长训练。

## 10. Final run

Final run 重新加载 canonical adapter initialization，创建新的 AdamW，使用全部 9000 张 train pool 图片，连续训练恰好 `U*` 个 optimizer updates。由于 selection 与 final 使用相同 global batch 16，该规则同时固定总样本曝光量：

```text
I* = 16 x U* images seen
```

selection checkpoint 的 reference epoch 只作为可读的派生量，不直接迁移为 final epoch；final 可能停在一个数据遍历周期内部。

Final run 中：

- 不读取标准 test；
- 不根据中途 loss、预览图或 checkpoint 改变停止点；
- 不执行模型质量选择；
- 只允许数值健康监控和完整状态恢复；
- 到达 `U*` 后自动停止。

## 11. 模型锁定与最终评价

Final run 完成后先导出 inference-only 权重，验证可重新加载，生成 `MODEL_LOCK.json` 并固定 SHA-256。只有模型锁和 Git tag 均存在时，标准 test loader 才允许运行。

最终 test 配置：

```text
test images             1000
candidates per image       8
denoising steps            50
guidance scale             4.0
brain encoder              full 4-layer x 2-run x 2-hemisphere ensemble
```

同时报告：

1. Fixed candidate：candidate 0；
2. Seed mean：8 套 candidate 指标的平均；
3. Encoder selected：完整 brain encoder 选择结果，用于论文口径比较。

八项指标均报告 mean、standard deviation 和 image-level bootstrap 95% CI。标准 test 在旧项目中已有历史访问，因此正式表述限定为“本阶段协议冻结后、模型锁定前未再次访问”，不声称其在整个研究历史中从未被查看。

## 12. 阶段交付物

```text
subject01_final.safetensors
subject01_final.pt
subject01_final.sha256
MODEL_LOCK.json
train_config.yaml
source_manifest.json
data_fingerprint.json
split_manifest.json
parcel_token_map.csv
environment.lock
selection_results.csv
metrics_test_fixed.json
metrics_test_seed_mean.json
metrics_test_encoder_selected.json
candidate_seed_manifest.json
model_card_zh.md
training_report_zh.md
preview_grid.png
failure_cases.csv
```

完整 resume 归档与 canonical 推理权重分开保存。后续任何研究运行前必须校验 checkpoint SHA-256 与 `MODEL_LOCK.json` 一致。

## 13. 启动与停止门禁

以下任一条件成立时不得启动正式训练：

- 两张正式 GPU 未同时空闲；
- 数据指纹未通过；
- 资产 revision 或哈希不完整；
- forward 对齐失败；
- 恢复一致性失败；
- 解码或 evaluator 不可重复；
- 正式配置仍存在未冻结字段；
- 当前 Git 工作树不干净或 protocol commit 未记录。
