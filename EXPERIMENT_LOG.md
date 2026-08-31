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

