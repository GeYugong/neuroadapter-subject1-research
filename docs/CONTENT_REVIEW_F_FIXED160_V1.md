# F：固定160图人工内容确认

制包已完成：160个案例、去重后406个输出、16页HTML和空白ratings.csv。本地位置为 `artifacts/content-review-f-fixed160-v1/reviewer_pack/`。所有包文件SHA与服务器一致；16页在1440与390像素宽度下逐页检查图片加载及横向溢出通过。99项测试通过。当前唯一待办为人工独立填写，尚无人工内容结果；不得将制包完成误写成验收完成。

## 状态与目的

本轮冻结R239063、原C（PCA1024、alpha0.1）、E原公式与500×8候选池。不改写E的“数值筛查通过、完整视觉条件未满足”；也不将原32图净少两例升级为总体视觉退步或方法失败。新增有限内容审查，不训练、不生成、不抽取特征、不访问标准test。人工标签尚未获得，不能宣布内容验收完成。

## 冻结抽样

命名空间 `content-review-f-fixed160-v1`。排除D已审阅32图，从468图按SHA256(namespace|sample|十进制image_id)升序取前160图，平局按ID排序。不使用分数、类别或选择改善抽样。固定随机候选为SHA256(namespace|random-candidate|十进制image_id)转整数后模8。它每图只选一张，与八候选评分平均Uniform不同。

匿名输出顺序使用一次生成、单独保存的随机盐固定；先去重候选索引，再分配A/B/C。同一候选仅评一次，再映射至所有选择它的方法。包中不含ID、索引、方法名、指标或映射。密封映射及其SHA保存于实验目录，不随评阅包或Git公开。此为标签盲法，不是安全隔离；评阅者应避免主动查询旧结果和映射。

## 标签与统计

主要对象、动作关系、场景、数量布局四维分开记correct/partial/wrong/uncertain；best允许并列、all_wrong、uncertain。详见评阅说明。预计一至两位人工评阅者；真实人数和独立性在收回时记录，不由程序假定。两人必须先各自完成再一起揭盲，不把两位评分当成320个独立样本。

统计分别报告每人的四维绝对计数及确认正确占160图比例。E−C与E−FixedRandom按图配对，10000次bootstrap、seed20261002、97.5%描述性区间；无法判断项单独报告，并补双方均可判断子集的比较，不把无法判断静默删除。partial不算完全正确，但保留其数量；不合成总分。区间未跨四维作多重比较校正，不能据某一个区间宣布整体通过。偏好报告E胜、对方胜、并列最佳、两者均非最佳、全部错误、无法判断。所有原始标签先校验完整并保存SHA，再揭盲；不允许缺行就评分。

## 有限决策

160图结束即停止，不加样本、不调公式。若人工内容较固定随机更好且相对C无清晰退步，可保留R+E为待独立验证的固定输出方案，不称已达到论文效果。若内容收益未确认则如实记录；若三者频繁缺失主体或E清晰退步，后续另行考虑生成端受控研究。本轮不自动启动该研究，不以图像指标提升恢复ROI/IBBI结论。

## 操作

服务器根目录ROOT为 `/data1/matengyu/geyugong/neuroadapter-subject1-research`。

准备：`$ROOT/envs/neuroadapter/bin/python $ROOT/repo/scripts/content_review_f.py prepare --root $ROOT`。

交付：`runs/diagnostics/content-review-f-fixed160-v1/reviewer_pack/`，打开index.html，在ratings.csv填写。仅发送此文件夹给评阅者；代码与协议公开，映射暂不公开。原始刺激和图片不进Git。

收回全部人工标签后才运行：`$ROOT/envs/neuroadapter/bin/python $ROOT/repo/scripts/content_review_f.py report --out $ROOT/runs/diagnostics/content-review-f-fixed160-v1 --ratings /path/to/rater1.csv /path/to/rater2.csv`。一位评阅者只提供一个文件。首次结果目录生成后禁止覆盖，不自动反复尝试不同标签或汇总方式。
