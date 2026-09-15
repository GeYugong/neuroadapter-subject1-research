# 匿名图像内容 AI 评阅记录

## 身份与范围

- 评阅者：rater-1，本任务中的单个 AI 助手。不是人工评阅，也不是人工独立盲评。
- 本记录仅证明 rater-1 的实际工作；未核实或代述另一个 AI 的完成情况。
- 评阅日期：2026-09-15（本任务环境日期，Asia/Shanghai）。
- 唯一输入根目录：`D:/0code/Research/neuroadapter-subject1-research/artifacts/content-review-f-fixed160-v1/reviewer_pack/`。
- 已读取其中的 `README.md` 和空白 `ratings.csv`，并读取 `images/` 中对应图像。
- 实际查看：F001 至 F160 的全部 160 张 GT，以及输入 CSV 所列全部 406 张匿名输出，共 566 张输入图像。没有跳过、换样本或推测去重前候选。
- 没有查看或查找其他对话，没有读取方法映射、仓库报告、已有分数或另一评阅者文件；未读取 rater-2 目录。未接触匿名标签背后的方法名称，也未推测方法归属。
- 没有联网、SSH、派生代理、修改代码、Git 提交或追加主日志；所有写入均限于本独占输出目录。

## 实际视觉查看方式

- 使用 `functions` 调用 `exec_command`，通过 PowerShell 读取说明与 CSV；通过 PIL 排版图像。
- PIL 仅负责读取图片、等比例缩放、白底补边、添加案例及输出标签和保存图册；没有计算图像特征、相似度或自动生成评分。
- 每张图册 1280 x 1360 像素，四行案例；每行从左至右为 GT、A、B、C。缺少的候选保持空白，不补造输出。每幅图片的最大显示区域为 320 x 312 像素，无裁剪。
- 依次使用 `view_image` 打开全部 40 张图册，并用 `image(result.image_url)` 将图像实际输出到本 AI 的上下文进行视觉判断，不是仅检查文件存在。
- 图册根目录：`D:/0code/Research/neuroadapter-subject1-research/artifacts/content-review-f-ai-v1/rater-1/`。
- 以下每一个文件均已实际打开查看：

```text
atlas-001-004.jpg  atlas-005-008.jpg  atlas-009-012.jpg  atlas-013-016.jpg
atlas-017-020.jpg  atlas-021-024.jpg  atlas-025-028.jpg  atlas-029-032.jpg
atlas-033-036.jpg  atlas-037-040.jpg  atlas-041-044.jpg  atlas-045-048.jpg
atlas-049-052.jpg  atlas-053-056.jpg  atlas-057-060.jpg  atlas-061-064.jpg
atlas-065-068.jpg  atlas-069-072.jpg  atlas-073-076.jpg  atlas-077-080.jpg
atlas-081-084.jpg  atlas-085-088.jpg  atlas-089-092.jpg  atlas-093-096.jpg
atlas-097-100.jpg  atlas-101-104.jpg  atlas-105-108.jpg  atlas-109-112.jpg
atlas-113-116.jpg  atlas-117-120.jpg  atlas-121-124.jpg  atlas-125-128.jpg
atlas-129-132.jpg  atlas-133-136.jpg  atlas-137-140.jpg  atlas-141-144.jpg
atlas-145-148.jpg  atlas-149-152.jpg  atlas-153-156.jpg  atlas-157-160.jpg
```

另用 `view_image` 实际打开以下原图复核，原图根目录为输入目录下 `images/`：

- `F008-GT.png`：确认坡上主体为白马，修正候选羊状动物的对象评分和 best。
- `F069-GT.png`：确认小主体为林中大象。
- `F107-A.png`：复核彩色布料及圆扣状物，佩戴关系仍无法可靠判断。
- `F116-A.png`：复核枝叶后的模糊动物，类别仍不确定。
- `F127-B.png`：复核圆形开口与内部结构，具体类别仍不确定。
- `F144-GT.png`：确认横向车内后视镜中为狗脸。

## 评分与保存

- 依据 README 的 object、action_relation、scene、count_layout 四维分别判断，不合成总分。
- 所有评分与简洁中文具体差异均由本 AI 在实际看图后逐行撰写，使用 `apply_patch` 手动增量写入 UTF-8 `ratings.csv`；没有以脚本规则代替视觉判断。
- 每批保存至 F012、F024、F036、F048、F060、F072、F084、F096、F108、F120、F132、F144、F160。
- best 是案例内部整体内容忠实度选择，不表示所选输出四维全对；允许并列。所有候选明显不对应时使用 all_wrong，单候选也不自动认可。
- 对无法可靠辨认的主体、佩戴关系、场景等使用 uncertain，并在 notes 中说明。不凭锐利、鲜艳、照片感判内容正确。

## 完整性校验

用 PowerShell `Import-Csv` 读取输入空白表和输出评分表，进行只读校验，不由程序更改评分：

- 输出数据行：406。
- 案例：160，F001–F160；每个 case/output 与输入逐行一致。
- 四维有效且完整：1624 个单元，均属于 correct、partial、wrong、uncertain。
- notes 非空：406 条。
- 同案例所有行 best 一致；非 all_wrong/uncertain 的选择均来自该案例实际候选。
- 校验错误：0。

最终文件：本目录的 `ratings.csv` 与 `review_record.md`。保留本次实际使用的 40 张排版图册便于核查。

## 局限

- 这是单个 AI 的视觉内容评阅结果，不能作为人工结果或多人一致性证据。
- 大部分图像以约 312 像素方形区域查看，图册包含 JPEG 重采样；小物体、文字、遮挡部位和畸形合成内容可能存在辨认误差。对选定疑点已复核原 PNG，但没有逐张以原始分辨率重新审阅。
- 不使用图像外语义资料，不能确认不可见动作、细分物种、人物身份或细小食品配料；notes 中的“状”“似”“难辨”保留这些限制。
- 四维和 best 均含 AI 视觉判断的主观性；未与其他评阅者交流、校准或比较评分。
