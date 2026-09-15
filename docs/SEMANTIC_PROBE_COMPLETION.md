# 原始目标完成核验

本表对应原始学习率收尾与独立语义探针任务，不将探针结果视为 NeuroAdapter 复现成功。

| 要求 | 已检查的证据 |
| --- | --- |
| 结束LR，保留B0/H/L/R | completion_audit.json 四份权重SHA及快照验证；未修改权重选择 |
| 已有64图轨迹和完整视觉报告 | paired-lr-probe-v1/closure 三份评分CSV、summary、32页图册SHA；已逐页视觉审阅的图册再次校验一致 |
| 固定7500/1000/500划分 | split_ids 四文件，固定namespace哈希函数，互斥与9000唯一ID审计 |
| 相同200parcel、有效顶点与图片ID对齐 | source_sha256、vertex_order、input_alignment的12条逐元素缓存检查 |
| 训练侧预处理，不影响NeuroAdapter | fit代码作用域与scaler/PCA样本数审计；8500均值方差独立重算 |
| 固定CLIP和原终点rawGT路径 | feature_manifest的权重SHA、batch16与两次GT逐元素一致；3份特征文件再次校验SHA |
| 10候选、固定并列规则、8500重新拟合 | tuning_results共60行；真实配对和5打乱各10行；selected_configs独立排序核验 |
| 记录方差、正则化曲线及搜索边界 | fit_summary、完整tuning_results、中文报告；未扩展搜索 |
| 均值与5次打乱对照 | 8500均值预测独立误差0；全部置换序列按预定seed重建一致；不报告精确置换p |
| 同500池cosine、前向二选一、Top1/5 | 九方法独立标量排名重算误差0；R/L候选先评分再图内平均 |
| 图级配对区间与解释范围 | 32项区间独立逐次重算误差0；报告注明固定池与探索性内部验证 |
| 不训练NeuroAdapter、不新生成、不访问标准test | 执行入口无扩散加载或更新；数据只引用train8500和val500，标准test不参与；GPU计算进程为空 |
| 报告回答三个问题及不能排除的解释 | SEMANTIC_PROBE_C_V1.md结论三项；明确不同监督目标、不能定位ParcelMapper或证明位置数量恢复 |
| 指定目录与线上归档 | 服务器runs/diagnostics/semantic-probe-c-v1/REPORT.md及所有指定JSON/CSV/ID文件；public manifests和本地报告 |

原始证据索引：[INDEX.json](../manifests/semantic-probe-c-v1/INDEX.json)。第一次独立审计：[completion_audit.json](../manifests/semantic-probe-c-v1/completion_audit.json)。补充核验：[finalization_audit.json](../manifests/semantic-probe-c-v1/finalization_audit.json)。后者不覆盖前者，记录在原始索引之后产生。

补充核验命令：

```bash
ROOT=/data1/matengyu/geyugong/neuroadapter-subject1-research
OPENBLAS_NUM_THREADS=8 "$ROOT/envs/neuroadapter/bin/python" \
  "$ROOT/repo/scripts/finalize_semantic_probe.py" --root "$ROOT"
PYTHONPATH="$ROOT/repo:$ROOT/runtime/subject01-4090-1a1fcfa/src" \
  "$ROOT/envs/neuroadapter/bin/python" -m pytest "$ROOT/repo/tests" -q \
  --basetemp="$ROOT/runs/diagnostics/semantic-probe-final-tests"
```

实际补充核验退出码0，完整测试89 passed、14条依赖警告。首次测试因PYTHONPATH缺少repo发生收集错误，已在主日志保留并更正，不隐去失败尝试。无新的模型拟合或扩散生成。
