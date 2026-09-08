# 双 RTX 4090 服务器迁移

## 迁移范围

2026-09-08 将执行平台从双 RTX 5090 切换到双 RTX 4090。GitHub 仓库仍为 `GeYugong/neuroadapter-subject1-research`，保持公开，不另建重复仓库。旧端基线提交为 `61dcf52715693117771f97ed2746583d515af813`，没有已启动的正式训练或需要接续的正式 optimizer state。

新实验根目录为 `/data1/matengyu/geyugong/neuroadapter-subject1-research`，代码位于其 `repo/`。初始检查发现系统盘只剩约 91 GiB，而 `/data1` 约有 1.2 TiB 可用，因此大数据及专用环境均放在 `/data1`。

迁移保留已重新下载和独立转换的原始 NSD、转换 beta、训练缓存、划分 ID、CBIG parcel、模型资产、canonical initialization、完整 Git 历史和五个 vendor checkout。不重新随机划分，不利用历史标准 test 结果调整方法，不复制任何服务凭据，不删除旧端备份。

## 传输与环境

使用服务器间 SSH/rsync 直传，临时授权限制为旧项目目录内只读访问；传输结束后撤销。环境使用 [conda-pack](https://conda.github.io/conda-pack/) 打包并在新根目录执行 prefix relocation，再重新安装两个 editable 本地包、执行 `pip check`、CPU 测试与实际 CUDA 检查。

旧环境中 setuptools 曾被 pip 替换，Conda 元数据仍记录旧文件，因此初次 conda-pack 拒绝打包。后续打包显式使用 `--ignore-missing-files --ignore-editable-packages`，不改变旧环境；新端必须核对实际依赖版本、修复 editable 路径并通过测试，不能仅凭打包成功宣布环境可用。

数据迁移与数据有效性分开验证：源端完整数据树 SHA-256 与新端逐文件比对，关键原始文件、训练缓存、模型和初始化还使用已有独立清单验证。原始审计文件保留其内容哈希，不能为适配路径改写历史证据。环境位置改变后的新 freeze 与平台信息另存，canonical manifest 经验证刷新，初始化权重本身不变。

## 硬件修订

- 实际设备要求改为双 RTX 4090，compute capability `[8, 9]`。
- 固定 PyTorch 2.11.0+cu128 wheel 含 `sm_86` 而不含单独 `sm_89`；使用其可兼容 cubin，不声称 wheel 含原生 `sm_89`。NVIDIA 明确支持 [8.6 cubin 在 8.9 GPU 运行](https://docs.nvidia.com/cuda/ada-compatibility-guide/)。实际 BF16、卷积、反向及 NCCL 测试仍必须通过。
- 首选 `4/GPU × accumulation 2`，备用 `2/GPU × accumulation 4`，两者 global batch 都为 16。正式训练前只冻结一种，不声称两种 geometry 逐步权重相同。
- 显存上限从 29 GiB 收紧为 22 GiB/GPU，训练日志取两个 rank 的最大 reserved memory。
- 压力测试仍至少 1800 秒，Xid 必须可读且无异常；通过 all-reduce 统一退出条件，直到每个 rank 都满足时长，避免独立计时造成 collective 不匹配。

## 尚需遵守的门禁

运行代码通过 `scripts/freeze_runtime.py` 从干净提交复制到独立 `runtime/`。训练配置将源码清单和固定计划绑定该副本，并显式以该副本 `src/` 为 PYTHONPATH。`repo/` 可以继续追加日志与提交报告；运行副本不随文档提交改变。该安排不放松 HEAD、工作区或方法指纹校验。

正式启动仍需 hardware、forward alignment、batch、resume、decode determinism、evaluator repeatability 六项证据全部通过。旧脑编码器 parcel 来源问题仍只阻断 brain-encoder-selected 最终 test，不因换机器自动解决。迁移进度、实际命令、日志和验收结果逐次追加主日志，不能把正在传输或尚未运行的门禁记为完成。
