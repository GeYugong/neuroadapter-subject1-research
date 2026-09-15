# rater-2 AI 内容评阅记录

## 身份与范围

- 评阅者：rater-2，当前任务中的单个 AI 助手。全部评分为 AI 视觉评阅，不是人工评阅，也不声称人工独立完成。
- 用户授权两名 AI 分别评阅；本记录只证明本 AI 的工作，不证明另一评阅者的完成情况或结果。
- 日期：2026-09-15（任务提供的本地日期）。
- 唯一素材来源：`D:/0code/Research/neuroadapter-subject1-research/artifacts/content-review-f-fixed160-v1/reviewer_pack/`。
- 已读取该目录的 `README.md`、`ratings.csv`，并查看 `index.html` 导航及 `images` 文件名。没有读取包外报告、分数、方法映射或其他评阅者文件。
- 实际视觉查看：F001 至 F160，连续全部 160 案例；每例 GT 和 ratings.csv 列出的全部匿名 A/B/C 输出，共 160 张 GT、406 张输出。没有跳过案例或替换样本。

## 查看方式与路径

使用 `exec_command` 调用 Python/PIL，仅读取输入图片并制作排版图册；每例 GT、A、B、C 从左至右排列，缺候选的位置留空。每册四例，1280×1408 像素，图像保持宽高比，单格最大 316×320 像素，标明案例与匿名标签。排版未修改输入图片，不使用特征、相似度、自动规则或脚本推断评分。

全部图册均通过 `view_image` 实际打开，并把返回图像通过 `image(...)` 输出到本 AI 上下文后进行视觉判断。图册目录：

`D:/0code/Research/neuroadapter-subject1-research/artifacts/content-review-f-ai-v1/rater-2/atlases/`

实际查看的 40 个图册文件及其全部案例范围如下：

```text
F001-F004.jpg  F005-F008.jpg  F009-F012.jpg  F013-F016.jpg
F017-F020.jpg  F021-F024.jpg  F025-F028.jpg  F029-F032.jpg
F033-F036.jpg  F037-F040.jpg  F041-F044.jpg  F045-F048.jpg
F049-F052.jpg  F053-F056.jpg  F057-F060.jpg  F061-F064.jpg
F065-F068.jpg  F069-F072.jpg  F073-F076.jpg  F077-F080.jpg
F081-F084.jpg  F085-F088.jpg  F089-F092.jpg  F093-F096.jpg
F097-F100.jpg  F101-F104.jpg  F105-F108.jpg  F109-F112.jpg
F113-F116.jpg  F117-F120.jpg  F121-F124.jpg  F125-F128.jpg
F129-F132.jpg  F133-F136.jpg  F137-F140.jpg  F141-F144.jpg
F145-F148.jpg  F149-F152.jpg  F153-F156.jpg  F157-F160.jpg
```

另通过 `view_image` 打开以下输入原图复核小主体、模糊物体或镜面关系；原图前缀均为唯一素材目录下的 `images/`：

```text
F008-GT.png
F069-GT.png
F093-GT.png
F116-A.png
F127-B.png
F144-GT.png
```

## 评分与保存

- 按 README 分别判断 object、action_relation、scene、count_layout，不合成总分，不以锐利度或鲜艳度代替内容对应。
- 每个匿名输出的 notes 均为本 AI 根据实际可见内容手动撰写的简洁中文差异说明。
- best 允许单选、并列及 all_wrong；明显全部偏离时没有被迫选择一个。仅一个候选时也没有自动选中。
- 使用 `apply_patch` 手动写入真实评分，依次在 F012、F020、F032、F040、F052、F060、F072、F080、F092、F100、F112、F120、F132、F140、F152、F160 后增量落盘。
- 没有用脚本生成评分或 notes；脚本仅用于图册排版与结构校验。
- 最终评分：`D:/0code/Research/neuroadapter-subject1-research/artifacts/content-review-f-ai-v1/rater-2/ratings.csv`。

## 校验结果

使用 PowerShell `Import-Csv` 读取输入空表与本评阅者输出进行结构核对，结果为：

- 数据行数：406。
- 唯一案例数：160。
- 与输入 case/output 逐行比较：完整保留原顺序，没有删行、加行或改标签。
- 四维评分：1624 个字段全部存在且属于 correct、partial、wrong、uncertain。
- notes：406 行全部非空。
- best：每案例所有行一致，所选匿名标签均存在于该案例，或使用规定特殊值。
- 校验错误：0。

## 局限与隔离

本结果是单个 AI 对匿名静态图像的主观内容判断，不是人工标注、事实身份鉴定或方法质量总分。主要借助缩放图册，可能漏掉很小的物体、被遮挡的配件及细微动作；必要时已复核上述原图。生成图像的畸变与低分辨率可能影响数量、物种和动作判断。F116-A、F127-B 的难辨内容已在相关维度使用 uncertain；F115-A 的场景缺少足够证据，也使用 uncertain。其他评分只描述可见内容，不推断隐藏关系。

是否接触方法映射：否。只见匿名 A/B/C，不推测或还原其方法名称，不读取 rater-1 或任何其他评阅者目录，不查找或继承其他对话。

本任务未联网、未 SSH、未派代理、未修改项目代码、未 Git 提交、未追加主日志；所有新写入文件均位于 rater-2 独占输出目录，输入包保持未修改。
