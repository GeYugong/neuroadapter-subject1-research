# 标准测试集访问规则

## 历史事实

此前的 NeuroAdapter 复现工作已经使用过 NSD Subject 1 标准 test，生成过重建图片并比较过多个旧 checkpoint。因此该 test 不能被描述为对整个研究历史完全未见。

## 本阶段规则

从本协议冻结开始到 `MODEL_LOCK.json` 生成之前：

- 训练器不得加载 test split；
- checkpoint 选择不得使用 test 图片、fMRI、指标或 brain encoder correlation；
- 解码确定性门禁只使用内部 validation；
- evaluator 开发只使用 validation 或人工构造样本；
- 任何 test loader 调用均必须记录并视为协议违规。

只有满足以下条件后才能运行标准 test：

```text
final run 已达到 U*
canonical 权重已导出并重新加载验证
MODEL_LOCK.json 已生成
checkpoint SHA-256 已冻结
Git tag subject01-final-v1 已建立
Git 工作树干净
```

正式报告采用准确表述：

> 标准 test 在旧复现项目中已有历史访问；在本次重新训练协议冻结后，未用于 checkpoint 选择或训练长度决策，并仅在最终模型哈希锁定后重新评价。

