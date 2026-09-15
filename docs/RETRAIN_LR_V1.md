# T1/T2 从canonical重新训练

本次新授权覆盖旧“不再训练”的限制，仅允许这两条训练及固定评价。独立分支feat/retrain-lr-v1；不修改旧runtime、旧权重或RESEARCH_WEIGHT_LOCK，不上传新权重至HF。

T1固定3e-5；T2按 `1e-5+0.5*(1e-4-1e-5)*(1+cos(pi*completed_updates/159374))` 在每次optimizer update前设置一次。各159375更新，53,125/106,250/159,375保存推理snapshot和完整恢复状态。无warmup、重启、EMA或其他调参。

同一canonical SHA：`dc363931727f5f5e445d267f9b31e1a366b134b2e62a34dc72ae12693d875fca`。两次全新AdamW，8500/500原划分；两卡4090，每卡4、累积2、global16；保持原BF16、Min-SNR、token dropout、backend与随机流。仅LR不同，名称与输出目录不进入种子派生。

## 启动检查

复用原数据/初始化/资产校验函数，按既有指纹校验一次后记录文件大小和mtime；变化时拒绝复用。参数审计38 tensor、116068608 trainable参数。T1/T2首update前loss/梯度/输入一致，更新后可训练组件变化、所有冻结参数哈希不变。T2连续20与新进程10+10比较模型、optimizer、四独立随机流、Python/NumPy/CPU/CUDA RNG、sampler及LR；真实周期始终159375。测试权重独立，不用于正式开发训练初始化。

新训练源从新commit建立冻结checkout；原primitives显式从runtime/subject01-4090-1a1fcfa/src导入，记录真实路径和新代码身份，不能冒用旧运行提交。旧正式approval不伪造也不放宽。

## 持久控制器

run_retrain_lr_suite.py顺序执行T1双卡→完整终点校验→T2双卡→统一六份新snapshot评价→四份旧同条件基线→选优及导出。启动和每个阶段检查GPU计算进程，存在其他任务立即停止，不抢占。fcntl锁防止本任务多份运行。控制器与子进程PID、完整命令、开始结束、退出码记录pipeline.json；每阶段终端日志单独保存。技术失败不启动下一阶段。

每100更新记录平均loss、裁剪前梯度范数、LR、样本数、reference epoch、吞吐和两卡峰值显存。每5000保存滚动恢复点，保留最近两个；三个里程碑恢复点放milestones永久保留。SIGTERM只设标记，在完整更新边界保存。仅同arm/同config/同源完整状态可恢复；重放后缀前将旧日志保存resume-history，不能把只恢复模型当连续训练。

启动或恢复均使用同一个冻结入口：`$ROOT/envs/neuroadapter/bin/python $FROZEN/scripts/run_retrain_lr_suite.py --root $ROOT`，环境PYTHONPATH指向旧primitives，CUBLAS_WORKSPACE_CONFIG=:4096:8，CUDA_VISIBLE_DEVICES=0,1。恢复控制器自动查找本arm最新完整checkpoint；已完成训练不重跑。不要手动删除输出后重启。

## 固定评价与导出

原500图、两候选、batch2、50步、guidance4，无选图。复用既有相同协议实际噪声，旧两候选PNG先查manifest/SHA并作小批逐像素回放。缺失或不兼容时仅在对应旧checkpoint存在的情况下补生成；缺失旧资产明确报告。所有可用新旧模型统一用原八项函数重评分，不混入C/D/E cosine前向识别率或八候选均值。

语义综合分为原百分制AlexNet5、Inception、CLIP二选一均值；六者最高者导出best_new_candidate/model.pt，完全平局按PixCorr高、更新早、arm字典序。配对bootstrap固定seed20260901、10000次，95%区间，负例池先固定500唯一图；区间仅开发描述，不独立确认。即使未改善也导出最佳新候选，记录8500图训练与500图选优，不覆盖旧锁。

内容图册预先固定有序验证列表32个等距位置，保留GT、239063及六个新权重的全部双候选。自动生成图册不等于实际视觉检查；训练评价导出完成后还须实际打开32张图册作AI定性说明。不重新发起F类评阅，不延长训练、不启动第三条。

## 当前状态

已实际启动T1双卡训练，启动检查时已完成200更新；T2排队，未声称已经训练。源码commit `4a6766e9a1628bfc8c98261c5d9ff59506e44c3f`，冻结入口 `$ROOT/runtime/retrain-lr-v1-final/`。首次有效update=1，global loss0.1590818763；记录时update200，LR3e-5，最近100步平均loss0.1129493840。

tmux会话 `neuroadapter-retrain-lr-v1`；启动时控制器PID303877，torchrun PID303901，两个worker PID303909/303910。PID仅是启动证据，恢复或系统重启后应以pipeline.json与实际进程核验。

T1/T2配置、初始化SHA、完整恢复命令见 `manifests/retrain-lr-v1/plan.json`；T1实际配置另存T1-effective-config.json。当前控制器日志为 `$ROOT/runs/experiments/retrain-lr-v1/controller.log`，训练终端train-T1.log，逐更新汇总T1/training.jsonl，阶段/退出码pipeline.json。当前阶段退出码null表示仍运行，不是成功完成。

必要短测通过：LR纯函数3 tests；T1/T2初始参数、loss、梯度一致且只LR配置不同；所有冻结参数未变；T2真实周期连续20与10+10的模型、AdamW、rank随机流、进程RNG、sampler和LR逐项完全一致。旧四个基线8/8候选逐像素回放通过，新snapshot小批读取八指标成功（smoke_only，不作性能结果）。启动证据、小型manifest均已归档。

当前自动队列为T1→T2→六新四旧同口径评价→最佳新候选导出。另有Codex每30分钟任务跟进，首次达到1000更新后依据实测估时，并在控制器完成后实际查看固定32图册、补中文内容结论和最终交付。没有准确1000步实测前不承诺完成时刻。新权重不公开上传HF，不扩展第三条实验。
