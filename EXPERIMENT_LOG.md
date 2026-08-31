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
