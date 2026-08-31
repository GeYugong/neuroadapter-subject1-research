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
