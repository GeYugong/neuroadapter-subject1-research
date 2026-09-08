# 实验日志

## 记录规则

本文件是项目唯一的连续实验日志。所有实际发生的仓库操作、下载、转换、环境变更、测试、训练、恢复、解码、评价、异常和科研决策均按时间向末尾追加，不覆盖、不删除既有记录。

每条记录至少包含：

- 时间和阶段；
- 实际工作内容；
- 命令或脚本；
- Git commit；
- 输入来源、路径和哈希；
- 输出路径和哈希；
- GPU/CPU/内存/磁盘条件；
- 运行时长；
- 验证结果；
- 异常与处理；
- 当前结论和明确停止点。

密码、访问 token、私钥和第三方 API key 永不写入本文件。

---

## 2026-09-01 项目初始化与只读硬件审计

### 目的

建立一个从原始 NSD 数据开始、最终冻结 Subject 1 正式 NeuroAdapter 权重，并为后续脑区贡献研究提供统一基础的独立私有项目。当前只制定并准备第一阶段：训练并冻结正式权重。

### 已核实的上游状态

```text
NeuroAdapter commit:
ae07c183844f298f5dee4c002f1439f23285c6d9

whole_brain_encoder commit:
767f25afc2240f568f4db8c0ce09604e6f83aa72
```

代码审计确认：

- 论文报告 2 张 NVIDIA L40、300 epochs、batch size 16、约 25 小时；
- 固定上游 shell 使用 500 epochs、每进程 batch 16；
- Accelerate 配置使用 2 processes 和 `split_batches=False`，global batch 可能为 32；
- `acc_config.yaml` 指定 BF16，但训练参数默认 FP16；
- 官方解码路径的 VAE latent、初始 noise 和 scheduler 随机性依赖全局 RNG；
- 官方 metric 通过可变的 `facebookresearch/swav:main` 加载 SwAV；
- 上游训练脚本包含明文 W&B 凭据，本项目不使用该凭据。

### 新服务器只读审计

```text
hostname:       asus-System-Product-Name
OS:             Ubuntu 24.04.4 LTS
project user:   matengyu
project root:   /data/matengyu/geyugong/neuroadapter-subject1-research
GPU:            2 x NVIDIA GeForce RTX 5090
VRAM:           32607 MiB / GPU
data disk:      15T total, about 6.9T free
system disk:    1.8T total, about 1.7T free
```

审计时两张 GPU 均有长时运行任务，显存分别约占用 18.6 GiB 和 18.1 GiB。未终止、暂停、重启或修改任何现有进程。`/data/matengyu/geyugong/researchx` 和 `/data/matengyu/geyugong/smr` 属于其他工作，不纳入本项目且不做修改。

### 访问控制

建立了仅用于该服务器的独立 SSH 密钥，指纹为：

```text
SHA256:nnoKIE53Zt40ozI5mTs5HT5eVw04CVqAj3KKRlvXjgU
```

密码和私钥未写入项目文件。后续 GitHub 写权限使用项目级 deploy key，不在服务器保存个人 GitHub token。

### 当前结论

硬件、磁盘和外网条件足以开展本项目。正式训练必须等待两张 RTX 5090 同时空闲；在此之前只进行不会抢占 GPU 的仓库、下载、转换、环境和验证准备。

---

## 2026-09-01 私有仓库、服务器目录与固定源码

### GitHub 仓库

建立私有仓库：

```text
https://github.com/GeYugong/neuroadapter-subject1-research
```

首个提交：

```text
24a94c0 docs(protocol): define subject 1 research and training phase
```

该提交建立：

- 完整研究目的与当前阶段边界；
- 第一阶段训练权重详细协议；
- 已知实现差异；
- 标准 test 访问规则；
- 唯一连续实验日志制度；
- 初始上游来源清单。

第二个提交：

```text
3e3be5c chore(setup): pin sources and add data preparation scripts
```

该提交加入：

- NeuroAdapter 固定 submodule；
- whole_brain_encoder 固定 submodule；
- Subject 1 数据配置；
- NSD、Schaefer 下载脚本；
- Schaefer 转换脚本；
- NSD metadata/HDF5 转换脚本；
- 文件树 SHA-256 清单工具；
- 正式训练 GPU 空闲门禁。

### 服务器项目结构

项目只使用：

```text
/data/matengyu/geyugong/neuroadapter-subject1-research
```

建立目录：

```text
repo/
data/raw/
data/derived/
data/fingerprints/
models/stable-diffusion-v1-5/
models/brain-encoder/
models/evaluation/
cache/huggingface/
cache/torch/
cache/wheels/
envs/
runs/calibration/
runs/selection/
runs/final/
archives/resume/
artifacts/subject01-final-v1/
logs/
credentials/
```

GitHub 访问采用只对该私有仓库有效的 read-write deploy key。服务器未保存个人 GitHub token。服务器 checkout 与 `origin/main` 一致。

### Submodule 部署异常

首次执行递归 submodule 初始化时失败：固定 NeuroAdapter 提交内部包含 `whole_brain_encoder` gitlink，但该上游提交没有为该路径提供可用的 `.gitmodules` URL。

处理方式：

```text
顶层 NeuroAdapter submodule：非递归初始化
顶层 whole_brain_encoder submodule：独立固定到 767f25a
```

两份源码最终 HEAD 均通过精确 SHA 校验。失败过程中没有修改上游源码；whole_brain_encoder 首次 checkout 未展开的工作树通过其已固定 index 补出，最终 submodule 工作树干净。

---

## 2026-09-01 原始数据下载启动

### NSD Subject 1

启动时间：

```text
2026-09-01T02:54:58+08:00
```

运行位置：

```text
tmux session: na_nsd_download
script: repo/scripts/download_nsd_subj01.sh
log: logs/download_nsd_subj01.log
target: data/raw/nsd
```

调度优先级：

```text
nice -n 15
ionice -c 3
GPU usage: none
```

下载来源为 NSD 官方公开 S3。脚本使用文件级原子落盘、单实例文件锁、远端 object inventory 和本地 size inventory。启动后确认已有 GPU 训练进程保持运行，未修改其 PID、显存或调度状态。

当前状态：下载仍在进行，完成状态、总大小和 SHA-256 在结束后另行追加。

### Schaefer2018 annotation

固定来源：

```text
repository: ThomasYeoLab/CBIG
commit: 35b5664bec8822e2f77da5e090e96f91d0095be6
surface: FreeSurfer5.3/fsaverage
atlas: Schaefer2018 1000 Parcels, 7 Networks
```

服务器直接访问 `raw.githubusercontent.com` 长时间无数据，因此终止了本项目自己的 curl 进程，改由本机已认证 GitHub API 获取同一 commit 的二进制内容，再通过 SSH 传入服务器。未终止任何其他进程。

最终文件：

```text
lh annotation
size: 1336179 bytes
sha256: ae529bddcb84b3ea8c5d7fdf577326a1e4922c9e0439ddf20e4052cf0497681b

rh annotation
size: 1336303 bytes
sha256: f0c93933c447616aff151d312074ee82189cf5a6de2f889cbbc186c2f3f7b097
```

存放位置：

```text
data/raw/schaefer/fsaverage/label/
```

---

## 2026-09-01 候选环境第一次安装与依赖修正

第一次候选环境安装使用独立 Conda prefix：

```text
/data/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter
```

运行方式：

```text
tmux session: na_environment
script: repo/scripts/setup_environment.sh
log: logs/setup_environment.log
nice: 15
ionice: idle class
GPU usage: none
```

Python 3.11、PyTorch 2.11.0+cu128 和 torchvision 0.26.0 已成功安装，随后候选依赖解析失败。失败原因是：

```text
accelerate 1.14.0       -> huggingface-hub >=0.21.0
diffusers 0.40.0        -> huggingface-hub >=1.23.0,<2.0
transformers 4.57.6     -> huggingface-hub >=0.34.0,<1.0
```

`diffusers 0.40.0` 与 `transformers 4.57.6` 对 Hugging Face Hub 的版本区间无交集。本次失败发生在依赖解析阶段，没有启动 GPU 进程，也没有进入训练。

上游 NeuroAdapter 固定提交的日期为 2026-03-01。为靠近上游代码时期并保留 Python 3.11、PyTorch 2.11 对 RTX 5090 的支持，候选版本修正为：

```text
diffusers==0.36.0
huggingface-hub==0.36.0
transformers==4.57.6
accelerate==1.14.0
```

`diffusers 0.36.0` 与其余三个版本的声明依赖区间存在交集。模型资产将按固定 commit 下载为本地快照，后续训练只接受经过哈希验证的本地模型路径，不从浮动 Hub 分支直接加载远程代码。

---

## 2026-09-01 候选环境、NSD下载与Schaefer转换完成

### 候选环境

修正后的独立环境于 `2026-09-01T03:24:32+08:00` 安装完成，`pip check` 无依赖冲突。当前候选版本为：

```text
Python             3.11.16
PyTorch            2.11.0+cu128
torchvision        0.26.0+cu128
NumPy              1.26.4
Accelerate         1.14.0
Diffusers          0.36.0
Transformers       4.57.6
Hugging Face Hub   0.36.0
```

环境仅安装在项目专用 prefix，没有修改共享 Conda 环境。CPU import 门禁通过；GPU、NCCL、BF16 与显存门禁仍等待两张 RTX 5090 同时空闲。

### NSD Subject 1

官方 S3 下载于 `2026-09-01T03:26:26+08:00` 完成，原始目录约 74 GiB。远端与本地 inventory 核对结果：

```text
session beta files           80
ncsnr files                   6
beta objects                 86
beta bytes          39326038008
all local files              90
all local bytes     78919016728
stimuli bytes       39556877048
```

关键清单哈希：

```text
S3 inventory SHA-256
85a388645bf65e6b1cdab1b22c741b3553abcb0497455f1b35f5b36a672d8ee3

local size inventory SHA-256
5eac93c452152b7cf4ebc4eec84b9c8d643717829c230f80218a244ac61f0548
```

完整原始文件树 SHA-256 清单已生成：

```text
data/fingerprints/nsd_subj01_sha256_inventory.json
```

所有下载和哈希任务均使用低 CPU/IO 优先级且不使用 GPU。

### Schaefer parcel

CBIG 固定提交的左右半球 annotation 已成功转换。每侧均得到 501 个 label（含 medial wall）和 500 个可用 parcel，顶点总数均为 163842。转换文件和每个 parcel 的 vertex-list SHA-256 已写入：

```text
data/derived/parcels/schaefer/schaefer_summary.json
```

---

## 2026-09-01 可恢复训练基础设施

按独立里程碑建立了正式训练所需的确定性基础：

```text
00e4aed feat(training): add deterministic data and sampling layer
97b38c8 feat(checkpoint): add exact resumable state format
03e31fe feat(model): freeze canonical adapter initialization
356096c feat(training): add guarded resumable DDP trainer
```

训练器固定使用显式 DDP 批次规划，保存每个 rank 的训练随机生成器、进程 RNG、sampler 状态、optimizer、下一 update 和输入哈希。正式模式同时要求：

```text
status: frozen
完整 protocol commit
Git worktree clean
与 config SHA-256 完全匹配的 approval file
```

当前配置模板保持 `status: draft`，没有创建 approval file，也没有启动正式训练。服务器项目环境的 CPU 测试结果为：

```text
16 passed, 2 import warnings
```

两项 warning 均来自在 `CUDA_VISIBLE_DEVICES=` 条件下导入 Diffusers 时自动关闭 CUDA autocast，不影响CPU测试结论。

---

## 2026-09-01 模型资产本地下载

服务器无法稳定访问 Hugging Face，因此按同一固定 revision 在本机下载后再传入服务器。固定资产为：

```text
Stable Diffusion v1.5
revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14

Subject 1 brain encoder
revision d8a978abb212eb2965b5d01673f96536b77e2ea0
```

本地下载共 98 个文件、`12772335879` bytes，文件树 SHA-256 清单已生成。首次下载实际完成后，脚本因目标清单父目录不存在而在最后一步退出；已下载文件完整保留。提交 `659d969` 修复父目录创建后复用已下载文件，第二次运行在约3秒内完成校验并成功写出下载记录，没有重新下载12.77GB内容。

评价资产中的 torchvision AlexNet、InceptionV3 和 EfficientNet 权重已在服务器下载约465 MiB。OpenAI CLIP 下载中断后产生的文件未通过官方URL内置SHA-256校验，因此不得作为正式资产；后续使用通过哈希验证的完整文件覆盖。

---

## 2026-09-01 NSD转换异常审计与第二次运行

第一次转换于 `2026-09-01T04:05:54+08:00` 启动，在处理右半球 session 11 时因非有限值门禁退出。没有生成正式 `betas_sub-01.h5`，只留下未完成的 `.tmp`。

逐块扫描确认：

```text
file             rh.betas_session11.mgh
NaN count        7500
affected trials  750 / 750
affected vertices 10
Inf count        0
```

10个顶点为：

```text
6706, 31007, 84758, 84759, 84760,
110759, 110760, 134628, 134629, 134632
```

相邻的右半球 session 10、12和左半球 session 11均无非有限值。10个异常顶点全部属于右半球 Schaefer parcel 320（不含 medial wall 的零基索引），mean-ncsnr 排名461，与top-100正式输入没有交集；正式top-100的 `max_voxels` 自然计算为626。

处理决策为：保留官方源NaN并单独审计，不执行静默置零；完整扫描必须证明所有选中顶点有限，训练缓存只读取选中parcel。对应提交：

```text
659d969 fix(data): audit source nonfinite beta values
```

新增合成测试覆盖源异常记录、未选中异常允许和选中异常硬失败。服务器测试结果：

```text
19 passed, 2 import warnings
```

第二次低优先级转换于 `2026-09-01T04:46:10+08:00` 启动：

```text
tmux session  na_nsd_convert_v2
GPU           disabled
nice          15
ionice        idle class
```

运行开始后再次确认两张GPU仍由既有任务占用，本项目没有终止、暂停或修改这些进程。

第二次转换于 `2026-09-01T04:53:22+08:00` 正常结束，退出码为0。正式产物：

```text
data/derived/neural_data/metadata_sub-01.npy
size 1631462 bytes

data/derived/neural_data/betas_sub-01.h5
size 39322447152 bytes

data/derived/neural_data/source_nonfinite_values.json
size 10602 bytes
```

异常审计文件准确记录1个源文件、7500个NaN、750个trial和10个vertex；HDF5属性同时标记 `source_nonfinite_values_preserved=true`。随后于 `2026-09-01T04:56:12+08:00` 启动完整低优先级数据扫描，用于验证全部正式选中顶点、生成100图映射抽查、parcel token映射和最终数据指纹。

---

## 2026-09-01 数据完整扫描、划分与训练缓存

### 完整数据门禁

完整扫描于 `2026-09-01T05:00:42+08:00` 完成。结果为：

```text
presentations                  30000
unique NSD images              10000
repetitions per image              3
standard train images           9000
standard test images            1000
selected parcels                 200
selected lh vertices           25479
selected rh vertices           27328
max_voxels                       626
selected vertex NaN / Inf          0
```

右半球全表扫描仍准确观察到此前记录的7500个源NaN，位置全部在未选中的parcel 320；正式输入顶点没有非有限值。关键指纹为：

```text
parcel map SHA-256
2764c8c62f1544065267c661b9e33f5aceab818b9ec4ab6dd4be0b299fe7f4a7

100-image mapping audit SHA-256
3f2b216396e6e4bec5f74e4234e9ea7c1493bf669fd3868a2d69bbd386e822fa
```

正式数据指纹位于：

```text
data/fingerprints/data_fingerprint.json
```

### Selection划分

划分于唯一图片ID层面执行，随机算法为NumPy `Generator(PCG64)`，seed为`20260901`。结果于 `2026-09-01T05:02:50+08:00` 冻结：

```text
selection_train     8500
validation           500
standard test       1000
```

清单哈希：

```text
selection_train
4681431e4f9fc3054f1d758bf0a95f485082f29eea1674b78a72f93c769d74c4

validation
1eb801c0460754e803c9d86f6c60aec0512da0a49205971d1efb0fef1cc256ad

standard test
d1afe140491cd887f7b8c612e144d104e56e412a33956f99bfefa3e27aac6605

9000-image train pool
11e107c402acfffea891215e39375312f3e98cce5a2d332811dc2181a2953f4e
```

### 9000图训练缓存

训练缓存以低优先级、无GPU方式构建，完成时间为 `2026-09-01T05:09:34+08:00`，退出码为0：

```text
path
data/derived/training/subject01_train_pool_top100.h5

shape         [9000, 200, 626]
dtype         float32
size          1924581142 bytes
SHA-256       88218e827856562a8efef1353d1cbaea0b4b00a2b139a5d9b09feb9892820400
```

独立验证器不复用构建过程中的统计值，于 `2026-09-01T05:24:04+08:00` 对缓存重新执行完整扫描：

```bash
python scripts/verify_training_cache.py \
  --cache data/derived/training/subject01_train_pool_top100.h5 \
  --manifest data/fingerprints/training_cache_manifest.json \
  --metadata data/derived/neural_data/metadata_sub-01.npy \
  --selection-train-ids data/derived/splits/selection_train_ids.txt \
  --validation-ids data/derived/splits/validation_ids.txt \
  --output data/fingerprints/training_cache_verification.json
```

独立验证结果：

```text
status                       verified
image IDs                    9000 unique train-pool IDs
standard test overlap        0
selected values finite       yes
padding outside valid mask   all zero
cache hash match             yes
```

有效parcel值的完整统计为：

```text
count    475263000
min      -30.841522216796875
max       59.4533805847168
mean       1.1830519251528038
std        1.6606457324689756
```

---

## 2026-09-01 模型资产同步与离线审计

### 模型树同步

由于服务器不能稳定访问Hugging Face，固定revision资产在本机完成下载和哈希后传入项目专用`models/`。本机基准清单包含50个正式文件：

```text
file count    50
total bytes   14165180204
```

服务器重新计算SHA-256后的比较结果：

```text
missing files       0
hash mismatches     0
server-only files   3
```

3个服务器独有文件是此前已通过官方来源下载的torchvision权重：

```text
alexnet-owt-7be5be79.pth
inception_v3_google-0cc3c7bd.pth
efficientnet_b1-c27df63c.pth
```

排除Hugging Face `.cache`目录后，服务器正式模型树包含53个文件、`14550139191` bytes。完整清单位于：

```text
data/fingerprints/model_assets_sha256.json
```

### 图像评价模型离线门禁

在以下环境约束下执行验证：

```text
CUDA_VISIBLE_DEVICES=
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

命令：

```bash
python scripts/download_evaluation_assets.py \
  --project-root /data/matengyu/geyugong/neuroadapter-subject1-research \
  --verify-only
```

于 `2026-09-01T05:50:00+08:00` 确认以下六类模型均可从固定本地代码和权重离线构建：

```text
AlexNet ImageNet1K V1
InceptionV3 default
EfficientNet-B1 default
OpenAI CLIP ViT-L/14
SwAV ResNet-50
DINOv2 ViT-B/14
```

其中CLIP、SwAV和DINOv2关键哈希分别为：

```text
CLIP    b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
SwAV    029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb
DINOv2  0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

### whole-brain encoder ensemble

新增可重复的纯CPU审计脚本，提交为：

```text
be2e860 feat(assets): verify brain encoder ensemble
```

运行命令：

```bash
python scripts/verify_brain_encoder_assets.py \
  --root models/brain-encoder/dinov2_q_transformer/schaefer/subj_01 \
  --output data/fingerprints/brain_encoder_assets_verification.json
```

审计于 `2026-09-01T05:58:03+08:00` 完成：

```text
layers                       1, 3, 5, 7
runs                         1, 2
hemispheres                  lh, rh
ensemble members             16 / 16
model tensors per member     25
model values per member      132734018
nonfinite model values       0
confidence shape             [163842] for every member
nonfinite confidence values  0
```

---

## 2026-09-01 Canonical初始化与协议修订

### Canonical adapter initialization

Stable Diffusion v1.5在离线、纯CPU模式下成功加载。canonical adapter初始化随后生成并在新建模型实例中重新加载，全部张量逐项位级相等：

```bash
python scripts/create_canonical_initialization.py \
  --model-path models/stable-diffusion-v1-5 \
  --model-manifest data/fingerprints/model_assets_sha256.json \
  --data-fingerprint data/fingerprints/data_fingerprint.json \
  --output models/canonical/subject01_adapter_init.pt \
  --manifest models/canonical/subject01_adapter_init.json \
  --seed 20260901
```

结果：

```text
created at                 2026-09-01T05:52:02+08:00
size                       464285745 bytes
SHA-256                    dc363931727f5f5e445d267f9b31e1a366b134b2e62a34dc72ae12693d875fca
trainable tensors          38
trainable parameters       116068608
frozen parameters          859520964
reload bitwise equal       true
```

selection和final必须加载这一完全相同的初始化文件，不允许依赖进程启动顺序重新随机初始化。

### Checkpoint选择规则更新

第一阶段协议已收紧为：

```text
一级评价     每个checkpoint对500张validation图生成2个固定candidate
shortlist    5个checkpoint
二级评价     每个shortlist checkpoint生成500 x 8 candidates
主选择器     SemanticScore
brain encoder不参与validation选择
```

final训练长度固定为selection选中checkpoint的`U* optimizer updates`。在global batch 16保持不变时，总样本曝光量同步固定为`I*=16 x U*`，不直接迁移8500图数据上的epoch数。

### 当前测试与停止边界

服务器项目环境在全部上述代码更新后的CPU测试结果为：

```text
26 passed
2 harmless Diffusers CPU import warnings
```

截至 `2026-09-01T06:02:28+08:00`，既有GPU任务仍在运行：

```text
GPU 0   17242 / 32607 MiB   utilization 81%
GPU 1   18057 / 32607 MiB   utilization 100%
tmux    diffinverse_sd_rgb, diffinverse_sd_thermal
```

本项目未终止、暂停、重启或修改这些任务。GPU、NCCL、BF16、显存、forward对齐、精确恢复和确定性解码门禁尚未执行。正式配置保持`status: draft`，`protocol_commit`仍未冻结，没有创建approval file，也没有启动正式训练。

---

## 2026-09-01 非GPU准备阶段停止审计

停止前完成逐项检查：

```text
GitHub repository     GeYugong/neuroadapter-subject1-research
visibility            private
default branch        main
protocol/log commit   ec34cfbf1709767210701597946030bb4af0755c
unique experiment log EXPERIMENT_LOG.md
future-stage plans    none
project secret hits   none
formal approval       absent
MODEL_LOCK.json       absent
formal train process  absent
```

离线评价模型加载曾在`vendor/swav`生成一个未跟踪的`__pycache__/hubconf.cpython-311.pyc`。确认该文件为本次导入产生的Python字节码后，仅删除该文件和空缓存目录；所有vendor源码和固定submodule提交保持不变。清理后服务器Git工作树重新为clean。

停止审计时，本地`main`、GitHub`origin/main`与服务器`main`均指向同一提交，关键数据、模型和canonical初始化产物全部存在。两张GPU仍由既有任务占用，因此按协议不执行任何GPU门禁。下一次工作应从“等待两张GPU同时空闲并执行训练前GPU门禁”开始，而不是直接启动正式selection训练。

---

## 2026-09-01 正式训练基础设施完整性加固

### 仓库与执行边界

GitHub 仓库当前保持 public：

```text
https://github.com/GeYugong/neuroadapter-subject1-research
```

本轮只执行 CPU、磁盘与代码审计，没有启动训练，也没有占用或修改服务器上既有 GPU 任务。检查时两张 RTX 5090 仍分别占用约 17.2 GiB 和 18.1 GiB，利用率为 79% 和 100%。

### Git 冻结证据

新增 `manifests/frozen/`，导出并公开保存 19 份小型证据文件及 `INDEX.json`，包括：

```text
9000/1000 与 8500/500 实际 image IDs
parcel token map
原始 NSD SHA-256 inventory
源非有限值审计
data fingerprint
training cache manifest 与 verification
Stable Diffusion、评价模型和 brain encoder 资产清单
canonical initialization manifest
NSD 图像映射 JSON/CSV
Schaefer 上游等价性审计
论文固定入口
```

导出脚本会把 `$PROJECT_ROOT` 下的绝对路径改写成项目相对路径，并拒绝实验根路径、个人 Windows 路径或既有服务器前缀残留。Git 中不保存 beta、HDF5、模型权重或 checkpoint。

### NSD 图像映射第二证据链

新增 `audit_nsd_image_mapping.py`，用官方 `nsd_expdesign.mat` 的 `masterordering -> subjectim` 映射，独立核对本地 metadata 与 `nsd_stim_info_merged.csv`。结果：

```text
presentation trials             30000
Subject 1 unique train images    9000
Subject 1 unique test images     1000
train shared1000=False           9000 / 9000
test shared1000=True             1000 / 1000
all subject1=True               10000 / 10000
metadata presentation order      exact match
```

完整 trial 级 CSV 已进入 frozen manifests。

### Schaefer membership 审计与正式阻断

逐顶点集合比较当前 CBIG-derived 资产与固定 `whole_brain_encoder/parcels/schaefer` 后得到：

```text
LH indexed equal                   7 / 501
LH unordered set intersection    501 / 501
LH relationship                  permuted_equivalent
RH indexed equal                   0 / 501
RH unordered set intersection      0 / 501
RH relationship                  different_memberships
upstream LH/RH set intersection  501 / 501
upstream LH/RH sets equal         true
```

因此，LH 使用相同的 501 个顶点集合但 label/token 顺序发生置换；RH 不只是顺序不同，公开上游 RH 文件包含的 501 个集合与其 LH 文件完全相同，并不匹配 CBIG-derived RH。当前 cache 仍保持原样，没有在证据不足时重建。

该审计已成为代码级 formal gate：`schaefer_upstream_equivalence.json` 不是 `indexed_equivalent` 时，`create_formal_approval.py` 和 formal trainer 都会拒绝。正式训练当前明确阻断，必须先确定作者实际 token 顺序和 RH membership 来源。

### 输入完整性与 canonical 绑定

训练配置改为只设置一个绝对 `project_root`，其余路径必须是无 `..` 的项目相对路径。正式 fresh run 和 resume 均执行：

```text
完整 nsd_stimuli.hdf5 文件大小和 SHA-256
Stable Diffusion 运行时文件逐文件 size/SHA-256
NeuroAdapter、whole_brain_encoder、CLIP、SwAV、DINOv2 HEAD
training cache verification 与实际 cache/manifest/fingerprint 绑定
NSD 图像映射四项检查
Schaefer indexed equivalence
```

canonical manifest 新增 environment lock、upstream source manifest、`modeling.py` SHA 和生成 Git commit。若环境或模型构造发生变化，旧初始化不得直接沿用。

真实服务器资产验证首次运行时，39 GB stimuli 已通过，但 Stable Diffusion tree 因 Hugging Face 自动生成的 `.cache/huggingface/*` 元数据被误判为额外模型文件。修正后只白名单排除该非运行时缓存前缀，其他额外文件仍被拒绝。重跑结果：

```text
stimuli SHA-256          a1a801da16f55bc6fbc9875ca7fec0666ac95eb333919e4a078f2a487a4e7031
SD runtime files         13
SD runtime bytes         4266679766
SD tree SHA-256          3191726b025fe6d3a182f7a763277d8993c2eece44299c9d9e36ff0ea603e75a
vendor HEADs             5 / 5 exact
vendor heads digest      0e8a66ace3ae92504c507559854f4ea502c75fc5b45d6f741c030ba621c2583b
```

### 训练、checkpoint 与随机轨迹

协议不再声称 `8/GPU x 1` 与 `4/GPU x 2` 权重严格等价。两种方案只分别验证稳定性和显存，并在 selection 前冻结一种；selection 与 final 不得切换。

训练器新增：

```text
显式 TF32、cuDNN benchmark/deterministic、AdamW fused/foreach 配置
每 rank 独立 trace-rank-XXXXX.jsonl
image IDs、timestep、VAE latent、diffusion noise、token mask SHA
崩溃残留 .incomplete 原子隔离到 corrupt/
完整 resume checkpoint 只保留最近 2 个
每 25 reference epochs 保存 inference-only snapshot
```

### Validation、选择与最终模型锁

正式 selection 所需工具已在训练前实现：

```text
validation_loss.py          500 图、每图 1 个固定 draw、VAE mean、dropout off、Min-SNR
decode_validation.py        每图固定 2/8 candidate seed、50 steps、guidance 4.0
evaluate_validation.py      完整八指标、固定 500 图负样本池、image 内 seed mean
select_checkpoint.py        shortlist 与 paired-bootstrap one-SE 选择
export_final_model.py       PT/safetensors 位级重载与 MODEL_LOCK
authorize_test_access.py    Git tag、模型 SHA 与 brain encoder full-forward 门禁
```

评价结果强制绑定同一 config、snapshot 和 validation ID SHA。shortlist 只接受 2-candidate 结果；final selector 只接受恰好 5 个 8-candidate checkpoint。LowLevelRank、HighLevelRank、Eff/SwAV 方向、average ties 和 one-SE 集合内 BalancedRank 均已在协议与代码中固定。

### GPU 门禁实现状态

已实现但尚未执行：

```text
gate_hardware.py               双 5090、sm_120、NCCL、BF16
gate_forward_alignment.py      固定上游与新 forward/loss 对齐
verify_batch_gate.py           preferred/fallback 稳定性与配置冻结
verify_repeatability_gate.py   100 vs 50+50、解码和 evaluator 内容重复性
verify_brain_encoder_forward.py 16-member 完整 forward 与 confidence ensemble
```

静态 brain encoder 审计已验证 16/16 checkpoint args、状态有限性、163842 顶点置信度与 8-member/hemisphere softmax 权重和；`full_forward_verified` 仍为 false，不能代替 GPU forward gate。

固定上游 `train_brain_adapter.py` 的整体导入检查还发现，它引用了当前仓库中不存在的 `nsd_groupwise_topk_parcel_dataset`。forward gate 因此改为从固定文件 AST 中提取唯一的原始 `setup_ip_adapter()` 函数体，并把完整文件与函数体 SHA 写入 gate，避免把无关坏 import 隐藏为本项目实现差异。

### 测试结果与当前结论

服务器项目环境最终 CPU 测试：

```text
48 passed
14 个固定上游 matplotlib/pyparsing deprecation warnings
10 个新增 CLI 均可离线 import 并显示 --help
python compileall passed
git diff --check passed
```

`ruff` 未安装在冻结候选环境中，因此未临时安装或修改环境。当前没有 formal approval、formal training process 或 `MODEL_LOCK.json`。下一步不是直接训练，而是先解决 Schaefer token/RH 资产来源，再在两张 GPU 同时空闲时执行全部 GPU 门禁。

## 2026-09-01：刷新 canonical 初始化证据绑定

在服务器 Git 工作树逐文件确认与提交 `e331389fc52b2bc71e5b274ed77c508a68e226dc` 一致后，使用 CPU 执行：

```bash
PYTHONPATH=repo envs/neuroadapter/bin/python repo/scripts/create_canonical_initialization.py \
  --model-path models/stable-diffusion-v1-5 \
  --model-manifest data/fingerprints/model_assets_sha256.json \
  --data-fingerprint data/fingerprints/data_fingerprint.json \
  --environment-lock data/fingerprints/requirements-freeze-candidate.txt \
  --source-manifest repo/manifests/upstream_sources.json \
  --repository-root repo \
  --output models/canonical/subject01_adapter_init.pt \
  --manifest models/canonical/subject01_adapter_init.json \
  --seed 20260901 \
  --environment-status candidate \
  --refresh-manifest
```

结果：

```text
initialization SHA-256  dc363931727f5f5e445d267f9b31e1a366b134b2e62a34dc72ae12693d875fca
initialization size     464285745 bytes
reload bitwise equal    true
repository commit       e331389fc52b2bc71e5b274ed77c508a68e226dc
environment status      candidate
```

初始化权重的 SHA-256 与刷新前完全一致；本次只补齐 environment lock、upstream source manifest、`modeling.py` 和 Git commit 的证据绑定，没有生成新权重。随后重新导出 `manifests/frozen/`。由于 Schaefer 等价性门禁尚未通过且 GPU 门禁尚未执行，canonical 状态继续保持 `candidate`，不得用于 formal training。

---

## 2026-09-01：正式协议第二轮审计闭合

### 执行边界

本轮只执行代码、CPU、磁盘和清单审计。检查期间两张 RTX 5090 均由既有任务占用，本项目没有终止、暂停、重启或修改这些进程，也没有启动任何 GPU 门禁或训练。GitHub 仓库继续保持 public；仓库可见性和分支保护不属于本轮技术修复范围。

以下结论取代本日志早先“公开 whole_brain_encoder Schaefer 文件不等价，因此 decoder formal training 必须阻断”的判断，但不改写原始历史记录。

### Decoder atlas 与 brain encoder parcel 分离

原有单一 Schaefer equivalence gate 被拆为两个职责不同的门禁。

Decoder atlas 使用以下命令重新审计：

```bash
cd /data/matengyu/geyugong/neuroadapter-subject1-research
PYTHONPATH=repo envs/neuroadapter/bin/python repo/scripts/audit_decoder_atlas.py \
  --project-root . \
  --annotation-dir data/raw/schaefer/fsaverage/label \
  --parcel-dir data/derived/parcels/schaefer \
  --metadata data/derived/neural_data/metadata_sub-01.npy \
  --parcel-map data/fingerprints/parcel_token_map.csv \
  --cache-manifest data/fingerprints/training_cache_manifest.json \
  --source-manifest repo/manifests/upstream_sources.json \
  --output data/fingerprints/decoder_atlas_audit.json
```

结果：

```text
gate                              decoder_atlas
status                            verified
surface                           fsaverage
CBIG commit                       35b5664bec8822e2f77da5e090e96f91d0095be6
parcels / hemisphere              500
selected parcels / hemisphere     100
model tokens                      200
top-SNR ranking                   verified
ordered 200-token vertex SHA-256  4ffebb4a915787fb159f8a5d943559c579e7bf14e6448da6375f7ef125be2c13
max_voxels                        626
```

CBIG annotation、派生 parcel、metadata、parcel token map 和 training cache manifest 均已纳入审计。结论是当前 decoder 输入链自洽，9000 图训练缓存不需要重建；公开 WBE 文件的 label 顺序不再错误阻断 decoder selection/final。

Brain encoder parcel 使用以下命令独立审计：

```bash
PYTHONPATH=repo envs/neuroadapter/bin/python repo/scripts/audit_brain_encoder_parcels.py \
  --asset-root models/brain-encoder/dinov2_q_transformer/schaefer/subj_01 \
  --parcel-dir repo/vendor/whole_brain_encoder/parcels/schaefer \
  --repository-root repo \
  --source-manifest repo/manifests/upstream_sources.json \
  --output data/fingerprints/brain_encoder_parcel_audit.json
```

结果：

```text
gate                         brain_encoder_parcel
status                       blocked
checkpoint count             16 / 16
query embedding shape        [501, 768] for all checkpoints
checkpoint parcel_dir        ./parcels/schaefer
fixed WBE commit             767f25afc2240f568f4db8c0ce09604e6f83aa72
runtime source relation      unverified
public LH/RH parcel masks    identical canonical membership hash
```

所有 checkpoint 结构、checkpoint SHA、query 数量、公开 parcel Git blob 和运行时 mask 均已记录。阻断原因不是 forward 失败，而是 checkpoint 没有携带作者训练时原始 parcel 文件的哈希，公开文件与内部相对路径之间缺少可验证的身份链。该问题只阻断最终 brain encoder full-forward、encoder-selected 标准 test 和 test access。

### Selection provenance 冻结

新增 frozen `subject01_selection_plan.json`，固定 20 个一级 snapshot update：

```text
13282, 26563, 39844, 53125, 66407,
79688, 92969, 106250, 119532, 132813,
146094, 159375, 172657, 185938, 199219,
212500, 225782, 239063, 252344, 265625
```

同时固定：

```text
validation images        500，ID 文件和实际顺序分别哈希
screening candidates     2
final candidates         8
denoising steps          50
guidance scale           4.0
validation-loss draws    1
validation batch         8
evaluation batch         16
bootstrap draws          10000
bootstrap seed           20260901
metric source SHA        evaluate_validation.py, metrics.py, selection.py
```

`validation_loss.py`、`decode_validation.py`、`evaluate_validation.py` 和 `select_checkpoint.py` 不再接受可改变科学设置的自由 CLI 参数。Optimizer update 从 atomic snapshot metadata 读取；每条结果绑定 snapshot model/manifest/metadata SHA、formal approval、完整 config、方法指纹、图片顺序和指标实现。Shortlist 必须恰好覆盖全部 20 个 update，二级 5 个 checkpoint 必须与 shortlist manifest 完全一致。

Selection plan 现场校验结果：

```text
selection plan SHA-256        1cf772d05ce9a7bc389f759bf2ac6532971cb87017f2e6b7e52ed965c7f3dce2
validation IDs SHA-256        73137068416d3400708e1a3d5bda09600be4ea7b2164f11322a26f4de1b37c55
image order SHA-256           1eb801c0460754e803c9d86f6c60aec0512da0a49205971d1efb0fef1cc256ad
metric implementation SHA-256 aec036bffe87ff6adfc580986ba3fd6a9ffe0dd7d8b9adbbd7f3276219decf02
```

### Selection 到 final 的 approval 闭合

新增 `method_fingerprint`，绑定科学超参数、执行后端、数据、缓存、模型、环境、源码、decoder atlas、selection plan 和 gate requirements；排除 run name、run kind、split、output 和 `max_updates`。因此 selection 与 final 可以复用已经通过的 GPU/重复性门禁，同时不能改变训练方法。

新增 final config template、完整 `train_pool_ids.txt` 和 `derive_final_config.py`。Selection approval 绑定 selection config 与六项门禁；final approval 进一步绑定 final selection manifest、`U*`、selection config 和 9000 图 train pool。Final config 只允许机械修改协议列出的运行字段。

9000 图 train pool 结果：

```text
count                       9000
ordered integer-array SHA   11e107c402acfffea891215e39375312f3e98cce5a2d332811dc2181a2953f4e
train_pool_ids.txt SHA-256  b23342d713b303cbe2931bee52009bf9b1adec2cfa35b91aae0fbea6bc6a5d1d
```

### GPU、checkpoint、export 与 test gate 加固

新增 frozen `gate_requirements.yaml`：双 RTX 5090、compute capability 12.0、原生 `sm_120`、BF16、forward tolerance `1e-6`、batch 至少 532 updates、双卡压力测试至少 1800 秒、reserved memory 上限 29 GiB 和必须可读的 Xid 检查。硬件门禁会执行 BF16 matmul backward、卷积 forward/backward、NCCL、UUID/driver/型号检查和持续压力测试，CLI 不能降低阈值。

Inference snapshot 新增与序列化无关的 tensor structure hash。恢复后再次到达已存在 snapshot 时，只有模型和 metadata 完全一致才幂等复用；冲突硬失败。完整 distributed checkpoint 在 prepare、各 rank 保存、publish 和 verify 阶段同步错误，避免单 rank 失败后其他 rank 卡在 barrier。

Final exporter 现在强制验证 snapshot 来自 `formal/final`、config/method/approval 与 `U*` 一致、run status 已完成、split 为完整 9000 图，并将 selection、snapshot、环境、源码、评价资产和 brain encoder 资产身份写入 `MODEL_LOCK.json`。标准 test access 还要求 annotated release tag，并逐项比较 brain encoder forward gate 与 model lock。

### 冻结证据导出

执行：

```bash
cd /data/matengyu/geyugong/neuroadapter-subject1-research
PYTHONPATH=repo envs/neuroadapter/bin/python repo/scripts/export_frozen_manifests.py \
  --project-root . \
  --repository-root repo
```

`manifests/frozen/` 新增：

```text
decoder_atlas_audit.json
brain_encoder_parcel_audit.json
train_pool_ids.txt
```

并更新 `INDEX.json`、`split_manifest.json` 和 Schaefer 来源审计。两份大 CSV 的服务器工作树差异仅为换行格式，`git diff --ignore-space-at-eol` 无语义差异，因此没有制造无关 CSV 变更。

### CPU 验证

最终同步代码后在项目专用环境执行：

```bash
cd /data/matengyu/geyugong/neuroadapter-subject1-research/repo
PYTHONPATH=src ../envs/neuroadapter/bin/python -m pytest -q
../envs/neuroadapter/bin/python -m compileall -q src scripts tests
```

结果：

```text
58 passed
14 个固定上游 matplotlib/pyparsing deprecation warnings
compileall passed
11 个关键 CLI --help 检查通过
frozen selection plan 输入校验通过
```

### 当前状态

```text
decoder atlas gate          verified
brain encoder parcel gate   blocked，仅阻断 encoder-selected test
GPU gates                   尚未执行
canonical status            candidate
formal selection approval   不存在
formal training             未启动
MODEL_LOCK.json             不存在
```

下一步仍不是直接训练。必须等待两张 RTX 5090 同时空闲，冻结最终 protocol commit、canonical manifest 和正式 selection YAML，再依次执行固定 GPU/forward/batch/resume/decode/evaluator 门禁。

## 2026-09-08：切换双 RTX 4090，启动完整项目迁移

### 决策与范围

根据新的服务器安排，后续工作转到双 RTX 4090，不再等待旧服务器。旧端项目保留作来源与备份，未停止或修改任何其他项目任务。GitHub 仍使用公开仓库 `GeYugong/neuroadapter-subject1-research`。

旧端源提交：`61dcf52715693117771f97ed2746583d515af813`，源 Git 工作区干净，正式训练未启动。原始数据、转换结果和模型直接迁移，不重新随机划分或重复从外网下载。

### 服务器初检

2026-09-08 11:48（北京时间）：Ubuntu 22.04.5，driver 580.173.02，两张 RTX 4090，各 24564 MiB 显存；显存使用 54/15 MiB，GPU 利用率均为 0%，无 compute process。主机内存 125 GiB，可用约 117 GiB。系统盘仅约 91 GiB 可用，`/data1` 约 1.2 TiB 可用。

新实验根目录：`/data1/matengyu/geyugong/neuroadapter-subject1-research`，代码目录为其 `repo/`。本机新增 SSH 别名 `neuroadapter-4090` / `双卡4090`，安装公开 SSH 公钥；密码与私钥不写入仓库。新端 kernel journal 可读。

### 迁移命令与证据

服务器间 rsync 使用只读、仅限旧项目的临时 SSH 授权，禁止端口与 agent 转发。传输 `repo/`、`cache/`、`models/`、`data/`，不传输 `credentials/`。核心命令结构如下（`RSYNC_RSH` 只引用服务器内临时私钥路径，不包含私钥文本）：

```bash
rsync -a --no-owner --no-group --safe-links --partial \
  --partial-dir=.rsync-partial --info=progress2 --bwlimit=80000 \
  matengyu@SOURCE_HOST:data/ "$PROJECT_ROOT/data/"
```

源数据 SHA inventory 用基线仓库 `scripts/hash_tree.py` 对完整 `data/` 生成；目标端用 `scripts/verify_migration.py` 对每个文件验证大小与 SHA-256，不以 rsync 退出成功代替内容校验。

旧环境初次 conda-pack 因 setuptools 的 Conda/pip 元数据不一致失败。记录后使用 conda-pack 0.8.1 的 `--ignore-missing-files --ignore-editable-packages --format tar` 打包实际环境；不修改旧环境。新端必须执行 conda-unpack、重新绑定 editable 包、依赖检查与 CPU/CUDA 测试，才能宣布环境迁移成功。

### 代码与配置修订

固定硬件要求改为双 RTX 4090、compute capability 8.9、现有 wheel 的兼容 `sm_86` cubin；30 分钟压力时长和 Xid 检查不变，显存上限改为 22 GiB/GPU。首选 `4/GPU × accumulation 2`，备用 `2/GPU × accumulation 4`，保持 global batch 16。测试两种配置不等于接续正式训练，门禁权重不作为正式结果。

修复压力测试各 rank 独立计时退出可能造成 collective 顺序不一致的问题，改为 all-reduce 统一退出判断。训练日志的峰值显存改为全 rank 最大值，避免仅记录 rank 0 而漏掉另一张卡超限。增加迁移校验 CLI、硬件配置一致性测试、环境入口和项目级 AGENTS 指令。

### 此时状态

本条记录时迁移仍在进行，尚未完成源/目标 SHA 校验，尚未完成新环境验收，正式 selection 未启动。后续实际结果继续追加，不覆盖本条阶段记录。

### 12:16：新端环境恢复与 CPU 验收完成

环境归档 SHA-256：`6e347a27fa92f9aca88b03faf777d3b0ad544ffd6380a46a9f4b168db9ebe0bd`，在新端提取前核对一致。`conda-unpack`、CLIP 与本项目 editable 路径重绑定完成，`pip check` 无依赖冲突。旧 freeze 中的 116 个固定发行包版本全部匹配，无缺包或版本差异。

```text
59 passed, 14 warnings in 5.73s
compileall passed
environment parity: verified, pinned_distribution_count=116, errors=[]
```

14 个 warning 均来自固定上游 matplotlib/pyparsing 的弃用提示。日志：新实验根目录 `logs/migration-environment-20260908.log`；版本核对：`artifacts/migration-20260908/environment-parity.json`。此时源端完整数据 SHA 清单已生成，新端原始 NSD 仍在传输。

训练代码将以干净提交复制到独立 `runtime/`，正式配置的源码、selection plan、gate requirements 路径绑定该副本；`repo/` 继续记录日志和报告，避免文档提交改变正在运行的 protocol HEAD。复制后的 Git 与 vendor 状态仍须验证，不能只复制 Python 文件。

### 12:36：传输完成，补齐自动启动与工程推理门禁

服务器间传输于 12:29:10 完成；数据树共传输 120,175,910,826 字节。新端完整 SHA 校验于 12:30:35 启动，尚未完成，不能据此宣布校验成功。固定 runtime 为 `runtime/subject01-4090-bce13f2`，训练协议提交为 `bce13f220c30494104a397c18fd91009b1e12993`，复制后 HEAD、工作区与五个 vendor 均验证一致。

canonical 权重原样保留，SHA-256 为 `dc363931727f5f5e445d267f9b31e1a366b134b2e62a34dc72ae12693d875fca`；使用新端环境锁刷新单独的 `artifacts/migration-20260908/canonical_manifest.json`，重新加载逐 tensor 一致。可训练参数 116,068,608，冻结参数 859,520,964。历史模型目录中的 manifest 未改写。

用户明确授权准备成功后直接开始正式训练，无须二次询问。增加顺序运行门禁与正式 selection 的脚本，每项运行追加主日志；只有全部门禁通过才生成绑定配置、输入与门禁 SHA 的正式 approval。首选配置若失败，脚本停止，不自动跳过失败或更改学习率/batch。

增加独立工程推理辅助脚本，解决正式 snapshot 尚不存在而推理重复性门禁必须先完成的启动顺序问题。辅助脚本直接调用固定 runtime 的推理函数，并抽取执行其未经修改的八指标计算段；绑定自己的脚本 SHA。固定验证集前 8 张图、每图 8 个候选用于重复/逆序解码；固定 20 个图像对用于评价程序重复性，不作为正式指标或模型选择依据，也不读取标准测试集。

新端测试：`63 passed, 14 warnings in 129.15s`；工程辅助 CLI `--help` 通过；自动启动脚本 `bash -n` 通过。此次运行同时进行完整数据读取，因此测试墙钟时间包含磁盘争用。新增四项测试覆盖固定评价段抽取、拒绝歧义、固定样本对以及不改写评价源文件。

30 分钟双卡硬件压力测试从 12:20:07 开始，截至本条仍在运行。此时尚未启动任何正式训练，当前 GPU 占用来自压力测试。

### 12:43：完整数据迁移校验通过，自动门禁流程就绪

完整数据校验于 12:40:27 成功退出，127 个文件共 120,175,910,826 字节逐一验证大小与 SHA-256，无缺失或额外文件。数据树 SHA-256：`3ac79739d4aa25ce3a653b7569767d9062c637f1c9ae7bf17c4f5833b9536f0c`。源清单 SHA-256：`73208c296dc9640a0555e41b222ead97dd960d8a565993dd0cfa07b98c94e5ef`。日志耗时约 9 分 52 秒。12:40:57 前后撤销旧端只读临时 authorized-key 条目，并删除新端迁移私钥；普通 SSH 访问仍使用本机已有密钥。

12:41:29 建立 tmux 会话 `neuroadapter-4090-gates-train`。其依赖等待程序只等待当前硬件门禁退出文件，最多等待 30 分钟；门禁非零退出会使流程停止。后续运行代码绑定 `runtime/subject01-4090-bce13f2`，工程启动与推理辅助代码来自提交 `5e39f6fcdd7261b9552fba975c865e40fe6b303a`。

```text
总流程日志：logs/4090-gates-train-pipeline.log
总流程退出码：artifacts/gates-4090/pipeline.exit
各阶段日志：logs/4090-<stage>.log
各阶段退出码：artifacts/gates-4090/<stage>.exit
正式训练目标目录：runs/selection/subject01-selection-4090-v1
```

新端数据校验、完整源数据 inventory、环境版本一致性和更新后的 canonical manifest 已导出到 `manifests/migration-20260908/`；INDEX 同时记录运行源文件 SHA 和公开副本 SHA。缺少的后续门禁文件不会被导出或伪装成通过。此时仍处于硬件压力测试，正式训练尚未开始。

### 2026-09-08T12:50:29+08:00：执行 training-cache-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-training-cache-verification.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/verify_training_cache.py --cache /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/training/subject01_train_pool_top100.h5 --project-root /data1/matengyu/geyugong/neuroadapter-subject1-research --manifest /data1/matengyu/geyugong/neuroadapter-subject1-research/data/fingerprints/training_cache_manifest.json --data-fingerprint /data1/matengyu/geyugong/neuroadapter-subject1-research/data/fingerprints/data_fingerprint.json --metadata /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/neural_data/metadata_sub-01.npy --selection-train-ids /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/splits/selection_train_ids.txt --validation-ids /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/splits/validation_ids.txt --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/migration-20260908/training-cache-verification.json 
```

结果：退出码 0，耗时 7 秒；证据保存在上述输出路径。

### 2026-09-08T12:50:36+08:00：执行 forward

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-forward.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/gate_forward_alignment.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090/forward_alignment.json 
```

结果：失败，退出码 1，耗时 39 秒。流程停止，未跳过门禁。

### 12:53：硬件与训练缓存通过，修复独立 forward 检查的设备前置条件

两张 RTX 4090 分别连续运行 1800.00057/1800.00065 秒，BF16 矩阵计算、卷积反向及 NCCL all-reduce 均通过，journalctl 无 Xid 事件。硬件配置及方法指纹已固化。训练缓存重新扫描通过：9000 图、200 parcels、626 最大顶点数，8500/500 内部划分无交集，标准测试集重叠为 0，无 NaN/Inf，padding 为 0。

12:51:15，第一次 forward 对齐检查在固定上游 `min_snr_loss_weights()` 报错：GPU timesteps 索引 CPU `alphas_cumprod`。正式上游训练会先调用 scheduler `add_noise()`，该调用把缓冲区移到相应设备；独立 forward 检查直接接收固定 noisy tensor，因此没有触发这一前置操作。此问题发生在检查程序，不是新训练器 loss，也没有产生训练权重。

修复仅在独立检查调用上游 loss 前显式把 `alphas_cumprod` 放到目标设备，数值、loss 公式和模型不变。修复后的工程检查从 `repo/scripts/gate_forward_alignment.py` 执行，并在证据中记录该文件 SHA 与工程提交；它的训练模块仍从冻结 runtime 导入，且重新核对上游 vendor HEAD。`runtime/subject01-4090-bce13f2` 未修改，硬件门禁及训练方法指纹不改变。失败日志 `logs/4090-forward.log` 和退出文件完整保留。

启动脚本加入独立 attempt 日志前缀，第二次从头运行其余门禁，避免覆盖第一次失败记录。CPU 回归测试 `63 passed, 14 warnings in 4.20s`，`bash -n` 通过。正式训练仍未启动。

### 2026-09-08T12:54:49+08:00：执行 training-cache-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-training-cache-verification.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/verify_training_cache.py --cache /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/training/subject01_train_pool_top100.h5 --project-root /data1/matengyu/geyugong/neuroadapter-subject1-research --manifest /data1/matengyu/geyugong/neuroadapter-subject1-research/data/fingerprints/training_cache_manifest.json --data-fingerprint /data1/matengyu/geyugong/neuroadapter-subject1-research/data/fingerprints/data_fingerprint.json --metadata /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/neural_data/metadata_sub-01.npy --selection-train-ids /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/splits/selection_train_ids.txt --validation-ids /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/splits/validation_ids.txt --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/migration-20260908/training-cache-verification.json
```

结果：退出码 0，耗时 6 秒；证据保存在上述输出路径。

### 2026-09-08T12:54:55+08:00：执行 forward

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-forward.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/repo/scripts/gate_forward_alignment.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090/forward_alignment.json
```

结果：退出码 0，耗时 7 秒；证据保存在上述输出路径。

### 2026-09-08T12:55:02+08:00：执行 batch-preferred

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-batch-preferred.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --run-mode gate --max-updates-override 532 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-preferred-532
```

结果：退出码 0，耗时 328 秒；证据保存在上述输出路径。

### 2026-09-08T13:00:30+08:00：执行 batch-fallback

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-batch-fallback.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_fallback.yaml --run-mode gate --max-updates-override 532 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-fallback-532
```

结果：退出码 0，耗时 363 秒；证据保存在上述输出路径。

### 2026-09-08T13:06:33+08:00：执行 batch-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-batch-verification.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/verify_batch_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --preferred-run /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-preferred-532 --fallback-run /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-fallback-532 --selected preferred --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090/batch_gate.json
```

结果：退出码 0，耗时 1 秒；证据保存在上述输出路径。

### 2026-09-08T13:06:34+08:00：执行 resume-continuous

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-resume-continuous.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --run-mode gate --max-updates-override 100 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-continuous-100
```

结果：退出码 0，耗时 97 秒；证据保存在上述输出路径。

### 2026-09-08T13:08:11+08:00：执行 resume-first

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-resume-first.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --run-mode gate --max-updates-override 50 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-resumed-100
```

结果：退出码 0，耗时 68 秒；证据保存在上述输出路径。

### 2026-09-08T13:09:19+08:00：执行 resume-second

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-resume-second.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --run-mode gate --max-updates-override 100 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-resumed-100 --resume /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-resumed-100/checkpoints/checkpoint-update-00000050
```

结果：退出码 0，耗时 70 秒；证据保存在上述输出路径。

### 2026-09-08T13:10:29+08:00：执行 resume-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-v2-resume-verification.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/verify_repeatability_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_preferred.yaml --gate resume_equivalence --left /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-continuous-100/checkpoints/checkpoint-update-00000100 --right /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-resumed-100/checkpoints/checkpoint-update-00000100 --left-aux /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-continuous-100/traces --right-aux /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/4090-resumed-100/traces --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090/resume_equivalence.json
```

结果：失败，退出码 1，耗时 4 秒。流程停止，未跳过门禁。

### 13:18：非确定性候选关闭，启用确定性新候选重新执行门禁

原候选首选/备用均完成 532 updates。首选峰值 reserved 14,849,933,312 bytes，备用 12,073,304,064 bytes；两者 loss 与 gradient norm 均有限，首选稳态约 0.51 s/update。该结论只属于原候选，不能自动套用到新计算配置。

13:10:33，100 步连续训练与 50+50 恢复的模型状态不一致。诊断证据为 `artifacts/gates-4090/resume-diagnostics.json`。两卡 100 步、所有 microbatch 的 image IDs、timestep、VAE latent、noise、dropout trace 完全一致；但第一步 loss 相同而梯度范数已不同（0.26291686 / 0.26282498），第二步开始 loss 也分歧。38 个参数张量不同，100 步时最大绝对差约 0.00503。因此不能把差异归因于第 50 步保存/恢复，原配置的计算重复性本身未达标，也不能声称该恢复门禁通过。

增加两次独立的 2 步诊断，仅将 `deterministic_algorithms=true`，并显式设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；数据、初始化、batch、学习率及训练代码不变。两次诊断的模型、优化器、trainer、两卡 RNG 和 trace 完全相同。证据 `artifacts/gates-4090/deterministic-two-run-probe.json` 由结构比较工具生成；其中工具的 `gate=resume_equivalence` 字段只是复用了比较器标识，实际比较的是两次从零开始的短运行，**不是正式 100 步恢复门禁，也不会被正式配置引用**。尚未逐算子定位到唯一的非确定性 kernel。

同时发现结构哈希工具不能对 AdamW 的零维 `step` tensor 直接执行 `view(uint8)`。修复为先 `reshape(-1)` 再读取字节，哈希头仍保存原始 dtype/shape；没有忽略任何状态、没有引入误差容忍。新增 scalar float/BF16、形状区别与空 tensor 测试，当前 CPU 回归为 `65 passed, 14 warnings in 14.37s`。原始模型分歧早于该工具错误，二者分别处理。

新候选名称：`subject01-selection-4090-deterministic-v1`。固定训练代码仍为 `bce13f220c30494104a397c18fd91009b1e12993`，已有后端配置函数支持确定性开关，因此没有修改运行副本。新配置为 `configs/calibration/subject01_4090_deterministic_preferred.yaml` / `..._fallback.yaml`；新门禁目录为 `artifacts/gates-4090-deterministic/`；新校准输出位于 `runs/calibration/deterministic-v1/`。原始候选的配置、校准权重和失败日志全部保留。

新环境锁 `artifacts/deterministic-4090/requirements-freeze.txt` 保留原依赖版本，增加显式的确定性及 cuBLAS runtime 要求。新 canonical manifest 位于同目录，重新验证后仅刷新环境绑定，canonical 权重 SHA 不变。六项门禁必须为这个新方法指纹重新生成，旧硬件/forward/batch 的通过证据不能直接授权新配置。新 30 分钟硬件门禁已启动；正式训练仍未开始。

### 2026-09-08T13:46:59+08:00：执行 training-cache-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-training-cache-verification.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/verify_training_cache.py --cache /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/training/subject01_train_pool_top100.h5 --project-root /data1/matengyu/geyugong/neuroadapter-subject1-research --manifest /data1/matengyu/geyugong/neuroadapter-subject1-research/data/fingerprints/training_cache_manifest.json --data-fingerprint /data1/matengyu/geyugong/neuroadapter-subject1-research/data/fingerprints/data_fingerprint.json --metadata /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/neural_data/metadata_sub-01.npy --selection-train-ids /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/splits/selection_train_ids.txt --validation-ids /data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/splits/validation_ids.txt --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/migration-20260908/training-cache-verification.json
```

结果：退出码 0，耗时 6 秒；证据保存在上述输出路径。

### 2026-09-08T13:47:05+08:00：执行 forward

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-forward.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/gate_forward_alignment.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/forward_alignment.json
```

结果：退出码 0，耗时 6 秒；证据保存在上述输出路径。

### 2026-09-08T13:47:11+08:00：执行 batch-preferred

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-batch-preferred.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --run-mode gate --max-updates-override 532 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-preferred-532
```

结果：退出码 0，耗时 351 秒；证据保存在上述输出路径。

### 2026-09-08T13:53:02+08:00：执行 batch-fallback

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-batch-fallback.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_fallback.yaml --run-mode gate --max-updates-override 532 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-fallback-532
```

结果：退出码 0，耗时 402 秒；证据保存在上述输出路径。

### 2026-09-08T13:59:44+08:00：执行 batch-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-batch-verification.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/verify_batch_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --preferred-run /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-preferred-532 --fallback-run /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-fallback-532 --selected preferred --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/batch_gate.json
```

结果：退出码 0，耗时 1 秒；证据保存在上述输出路径。

### 2026-09-08T13:59:45+08:00：执行 resume-continuous

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-resume-continuous.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --run-mode gate --max-updates-override 100 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-continuous-100
```

结果：退出码 0，耗时 103 秒；证据保存在上述输出路径。

### 2026-09-08T14:01:28+08:00：执行 resume-first

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-resume-first.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --run-mode gate --max-updates-override 50 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-resumed-100
```

结果：退出码 0，耗时 72 秒；证据保存在上述输出路径。

### 2026-09-08T14:02:40+08:00：执行 resume-second

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-resume-second.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/train_subject01.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --run-mode gate --max-updates-override 100 --output-override /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-resumed-100 --resume /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-resumed-100/checkpoints/checkpoint-update-00000050
```

结果：退出码 0，耗时 74 秒；证据保存在上述输出路径。

### 2026-09-08T14:03:54+08:00：执行 resume-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-resume-verification.log`。

```bash
env PYTHONPATH=/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/src /data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/verify_repeatability_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --gate resume_equivalence --left /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-continuous-100/checkpoints/checkpoint-update-00000100 --right /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-resumed-100/checkpoints/checkpoint-update-00000100 --left-aux /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-continuous-100/traces --right-aux /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-resumed-100/traces --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/resume_equivalence.json
```

结果：退出码 0，耗时 10 秒；证据保存在上述输出路径。

### 2026-09-08T14:04:04+08:00：执行 decode-same-process

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-decode-same-process.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/gate_preflight_inference.py --runtime /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2 --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --action decode --snapshot /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-preferred-532/snapshots/snapshot-update-00000532 --output /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/normal --repeats 2
```

结果：退出码 0，耗时 227 秒；证据保存在上述输出路径。

### 2026-09-08T14:07:51+08:00：执行 decode-new-process

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-decode-new-process.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/gate_preflight_inference.py --runtime /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2 --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --action decode --snapshot /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-preferred-532/snapshots/snapshot-update-00000532 --output /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/reversed --reverse
```

结果：退出码 0，耗时 116 秒；证据保存在上述输出路径。

### 2026-09-08T14:09:47+08:00：执行 decode-same-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-decode-same-verification.log`。

```bash
env PYTHONPATH=/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/src /data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/verify_repeatability_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --gate decode_determinism --left /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/normal/pass-0/decode_manifest.json --right /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/normal/pass-1/decode_manifest.json --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/decode_same_process.json
```

结果：退出码 0，耗时 1 秒；证据保存在上述输出路径。

### 2026-09-08T14:09:48+08:00：执行 decode-new-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-decode-new-verification.log`。

```bash
env PYTHONPATH=/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/src /data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/verify_repeatability_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --gate decode_determinism --left /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/normal/pass-0/decode_manifest.json --right /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/reversed/pass-0/decode_manifest.json --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/decode_determinism.json
```

结果：退出码 0，耗时 1 秒；证据保存在上述输出路径。

### 2026-09-08T14:09:49+08:00：执行 evaluate-a

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-evaluate-a.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/gate_preflight_inference.py --runtime /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2 --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --action evaluate --decode-manifest /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/normal/pass-0/decode_manifest.json --output /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-evaluator-a
```

结果：退出码 0，耗时 9 秒；证据保存在上述输出路径。

### 2026-09-08T14:09:58+08:00：执行 evaluate-b

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-evaluate-b.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/gate_preflight_inference.py --runtime /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2 --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --action evaluate --decode-manifest /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-decode/normal/pass-0/decode_manifest.json --output /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-evaluator-b
```

结果：退出码 0，耗时 8 秒；证据保存在上述输出路径。

### 2026-09-08T14:10:06+08:00：执行 evaluate-verification

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-evaluate-verification.log`。

```bash
env PYTHONPATH=/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/src /data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/engineering-4090-4a54d92/scripts/verify_repeatability_gate.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/calibration/subject01_4090_deterministic_preferred.yaml --gate evaluator_repeatability --left /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-evaluator-a/evaluation.json --right /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-evaluator-b/evaluation.json --left-aux /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-evaluator-a/per_pair.csv --right-aux /data1/matengyu/geyugong/neuroadapter-subject1-research/runs/calibration/deterministic-v1/4090-evaluator-b/per_pair.csv --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/evaluator_repeatability.json
```

结果：退出码 0，耗时 1 秒；证据保存在上述输出路径。

### 2026-09-08T14:10:07+08:00：执行 formal-approval

运行代码：`/data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2`。日志：`/data1/matengyu/geyugong/neuroadapter-subject1-research/logs/4090-det-v1-formal-approval.log`。

```bash
/data1/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python /data1/matengyu/geyugong/neuroadapter-subject1-research/runtime/subject01-4090-bce13f2/scripts/create_formal_approval.py --config /data1/matengyu/geyugong/neuroadapter-subject1-research/configs/formal/subject01_selection.yaml --output /data1/matengyu/geyugong/neuroadapter-subject1-research/artifacts/gates-4090-deterministic/selection_approval.json --approve
```

结果：失败，退出码 1，耗时 1 秒。流程停止，未跳过门禁。

### 2026-09-08T14:13:45+08:00：定位正式许可的真实数据 schema 不匹配

- det-v1 的六项硬件/数值门禁全部通过。连续 100 与 50+50 更新的完整状态、两卡 traces 完全一致；同进程与新进程倒序生成的 64 张 PNG 完全一致；两次八项指标 JSON 和逐样本 CSV 完全一致。
- 14:10:08 正式许可检查失败，尚未创建任何 formal training run。错误 `decoder atlas audit has not passed` 并非 atlas 审计未通过，而是 `validate_subject1_audits()` 从 cache verification 读取不存在的 `max_voxels` 和 `parcel_map_sha256`。真实报告使用 `brain_shape`，parcel SHA 位于已经通过哈希绑定的 data fingerprint。
- 修复为从已绑定 data fingerprint 读取 max_voxels 与合法 SHA，并要求 cache `brain_shape == [9000, 200, max_voxels]`，然后与 atlas 比较；没有放宽状态、尺寸或哈希要求。回归测试改用实际报告 schema，并增加错误尺寸、错误 parcel SHA 拒绝检查。服务器全套测试 65 passed，14 条上游弃用警告，耗时 3.55 秒。
- 增加 CPU 真实数据审计入口，在 GPU 验收前检查接口；下一轮使用新冻结提交、新 config、独立 det-v2 目录重新验收。旧许可失败日志、全部门禁、checkpoint 和图片均保留，不修改旧通过记录的指纹。
- 抽查 det-v1 的 `4090-decode/normal/pass-0/images/00095/candidate-00.png`：512x512 图像正常可读，无空白或损坏。仅检查解码输出有效性，不据此判断重建质量，也不用于权重选择。
