# D：固定八候选语义选图结果

## 结论

本轮已完成。冻结 C 的正确脑输入对选图有帮助，但相对八候选均值基线，主识别率仅增加 **0.5735 个百分点**，97.5%区间 **[-0.5587, 1.6756]**。相对 Mean-selected 增加 **3.3267 个百分点**，区间 **[2.2088, 4.6044]**。未达到预设2百分点投入筛查幅度，不宣布整体修复、不提升正式管线、不替换R。

## 协议

使用原239063的500×8既有PNG，全部4000份SHA和可读性检查通过；C的1024维/alpha0.1、8500图拟合产物与预测均冻结。本轮新增NeuroAdapter更新、C拟合、扩散生成均为0，不访问标准test，不扩候选数或guidance。

选择函数仅输入C预测和候选特征，按cosine选最大，平局取最小索引，不接受GT。五组无固定点错配提前固定；全部保留，只有描述意义，不是精确置换检验。Mean在已用正确fMRI生成的池中挑图，因此不应要求其表现为机会水平。

所有特征识别均为预测行到固定500张GT列的float64 cosine前向二选一，平局半分；与旧官方相关性口径不同。CLIP GT与C逐元素相同。Uniform先评分各候选，再图内平均，不平均embedding；八候选基线独立重算，不混用旧两候选数字。

## 主要数值

| 方法 | CLIP二选一 | Top1 | Top5 | CLIP cosine | Inception二选一 | AlexNet5二选一 | PixCorr | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 85.1724% | 7.175% | 23.15% | 0.629364 | 79.5852% | 86.2671% | 0.089302 | 0.290518 |
| C-selected | 85.7459% | 9.4% | 26.2% | 0.676556 | 82.3535% | 86.6176% | 0.095519 | 0.292574 |
| Mean-selected | 82.4192% | 6.8% | 22.4% | 0.664699 | 80.1651% | 84.6593% | 0.093485 | 0.293232 |
| Oracle-cosine（仅诊断） | 92.7924% | 18.8% | 44.8% | 0.701392 | 85.8497% | 88.4313% | 0.093145 | 0.290958 |
| Oracle-identification（仅诊断） | 95.5194% | 26.8% | 53.2% | 0.679197 | 85.4060% | 89.1214% | 0.094869 | 0.287416 |

五组错配CLIP识别率分别为82.5639%、82.5090%、82.0437%、82.2385%、82.4641%。所有方法、所有指标与逐图结果见[paired_comparisons.json](../manifests/semantic-rerank-d-v1/paired_comparisons.json)和[per_image_scores.csv](../manifests/semantic-rerank-d-v1/per_image_scores.csv)。

两个主比较固定10000次图级bootstrap、97.5%区间。辅助Inception的C−Uniform为+2.7683百分点，区间[1.4523,4.0790]；AlexNet5、PixCorr、SSIM区间跨零。辅助比较未进行额外多重比较校正，不作为正式多指标结论。cosine增加0.0472不能单独解释为修复。区间条件于既有内部验证样本、当前模型和固定候选池，不代表跨被试或训练seed。

两种oracle都读取GT、仅离线诊断、不可部署。cosine oracle只界定本池cosine上限，identification oracle只界定本池前向二选一上限；不能作为其他指标、视觉质量或所有随机种子的共同上限。

## 视觉与停止点

全部32个固定验证候选池已先隐藏选择器标记逐图审阅，记录在提交a11bd10冻结后才揭示索引。审阅者为AI，不是多人盲评；“匹配”仅为粗粒度类别和场景，不代表细节重建。

12/32池有此类匹配候选，C选中10个、Mean选中8个、Uniform期望5.625个。两例有匹配但C错过：13223选飞机而非池内列车，43211选滑雪而非持冲浪板人物。滑板16800、货列29511、甜甜圈56963、门廊51865的候选池均未恢复关键内容。不能由这32例外推500图成功率。

结果是混合的：oracle显示排序空间，但部分池完全缺目标内容；C未可靠实现指标上限。下一研究方向可定位排序适配，本轮不自动开展新实验或重训。任何未来R+C输出改善都不能解释为R参数变好，更不能直接定位ParcelMapper。ROI因果研究不得在干预后让读取完整脑信号的C重新选图。

## 交付与核验

- 含全部32张图的本地报告：`artifacts/semantic-rerank-d-v1/REPORT.md`，图片在同级 `annotated_gallery/`。
- 服务器：`runs/diagnostics/semantic-rerank-d-v1/REPORT.md`，与本地同内容。
- [逐图视觉记录](../manifests/semantic-rerank-d-v1/visual_review.csv)、[完整性及选择审计](../manifests/semantic-rerank-d-v1/completion_audit.json)、[冻结计划](../manifests/semantic-rerank-d-v1/plan.json)。
- 数据数组、特征矩阵和图册不上传Git；公开小型结果与SHA，避免发布刺激数据。

执行命令（ROOT为服务器项目根目录，项目Python与冻结runtime）：

```bash
ROOT=/data1/matengyu/geyugong/neuroadapter-subject1-research
export PYTHONPATH="$ROOT/runtime/subject01-4090-1a1fcfa/src"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
"$ROOT/envs/neuroadapter/bin/python" "$ROOT/repo/scripts/prepare_semantic_rerank.py" --root "$ROOT"
"$ROOT/envs/neuroadapter/bin/python" "$ROOT/repo/scripts/run_semantic_rerank.py" --root "$ROOT" --phase features
"$ROOT/envs/neuroadapter/bin/python" "$ROOT/repo/scripts/run_semantic_rerank.py" --root "$ROOT" --phase score
# Freeze all blinded visual judgments before the following report phase.
"$ROOT/envs/neuroadapter/bin/python" "$ROOT/repo/scripts/report_semantic_rerank.py" --root "$ROOT"
"$ROOT/envs/neuroadapter/bin/python" "$ROOT/repo/scripts/audit_semantic_rerank.py" --root "$ROOT"
```

各阶段退出码0。特征/像素评分内部计时62.99秒（不含前置文件读取校验和后续视觉审阅），GPU0只提取固定特征。全套测试92 passed，14条依赖警告，5.61秒。paired_comparisons中原始`visual_review_complete=false`是评分时状态，保留不覆盖；后续visual_review_summary与completion_audit证明审查已完成。
