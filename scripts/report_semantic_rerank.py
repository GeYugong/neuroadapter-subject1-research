"""Join already frozen visual judgments with selector outputs and report D."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_semantic_rerank import sha, write
from run_semantic_rerank import setup, rows_csv


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def run(root):
    out, plan = setup(root)
    blind_path = root/'repo/docs/semantic_rerank_d_blind_review.csv'
    blind = read(blind_path)
    assert len(blind) == 32 and {int(r['image_id']) for r in blind} == set(plan['visual_ids'])
    selection = {(r['method'],int(r['image_id'])):int(r['candidate_index']) for r in read(out/'selector_indices.csv')}
    result = json.loads((out/'paired_comparisons.json').read_text())
    gallery = out/'annotated_gallery'
    gallery.mkdir(exist_ok=False)
    joined=[]
    for r in blind:
        i=int(r['image_id'])
        acceptable={int(k) for k in r['category_scene_acceptable_indices'].split(';') if k}
        chosen={name:selection[name,i] for name in ['C-selected','Mean-selected','Oracle-cosine','Oracle-identification']}
        joined.append({**r,**chosen,'pool_has_category_scene_match':bool(acceptable),
            'C_hits_category_scene_match':chosen['C-selected'] in acceptable,
            'Mean_hits_category_scene_match':chosen['Mean-selected'] in acceptable,
            'Uniform_category_scene_rate':len(acceptable)/8,
            'pool_has_match_but_C_misses':bool(acceptable) and chosen['C-selected'] not in acceptable})
        image=Image.open(out/'blind_gallery'/f'{i}.jpg').convert('RGB')
        page=Image.new('RGB',(image.width,image.height+50),'white')
        page.paste(image,(0,50))
        ImageDraw.Draw(page).text((5,12),f"ID {i} | C={chosen['C-selected']} | Mean={chosen['Mean-selected']} | Oracle cosine={chosen['Oracle-cosine']} | Oracle identification={chosen['Oracle-identification']} (GT-only)",fill='black')
        page.save(gallery/f'{i}.jpg',quality=95)
    rows_csv(out/'visual_review.csv',joined)
    counts={k:sum(bool(r[k]) for r in joined) for k in ['pool_has_category_scene_match','C_hits_category_scene_match','Mean_hits_category_scene_match','pool_has_match_but_C_misses']}
    counts['Uniform_expected_category_scene_matches']=sum(r['Uniform_category_scene_rate'] for r in joined)
    write(out/'visual_review_summary.json',{'counts':counts,'reviewer':'AI visual review; not human raters',
        'blinded_labels_commit':'a11bd10','blind_review_sha256':sha(blind_path),
        'criterion':'approximate main category and scene match; not faithful reconstruction or formal psychophysics',
        'all32_inspected_before_selector_labels':True,
        'annotated_gallery':{p.name:sha(p) for p in gallery.glob('*.jpg')}})
    lines=['# D：固定八候选的语义选图与覆盖能力诊断','',
        '## 结论','',
        '冻结 C 可以利用正确脑输入改善相对通用均值选择器的排序，但相对不选图的八候选期望基线，主识别率增益较小且区间跨零。Inception 出现辅助改善，不能将结果归为完全无效，也不能宣布整体修复。已有候选池的识别 oracle 明显更高，下一步定位应优先考虑排序适配；本轮到此停止，不自动开展新适配、训练或扩大候选池。','',
        '本轮不提升为正式 R+C-selector 研究管线，不替换 R，不恢复 ROI/IBBI 结论性实验。','',
        '## 方法与边界','',
        'R=原239063，固定500张内部验证图，每图8张既有PNG，共4000张。文件SHA、可读性和manifest一致性均通过。C保持PCA1024/alpha0.1和8500图拟合产物不变；其原搜索边界限制仍成立。新增NeuroAdapter更新、C拟合、扩散生成均为0，未访问标准test。','',
        '选择函数只接收C预测和候选特征，用cosine最大值选图，精确平局取最小候选索引。GT只用于评分和离线oracle。Mean使用同一训练均值，5组错配为预先固定的无固定点置换，不选择其中最佳结果，不报告精确置换p。Mean-selected不应被要求等于机会水平，因为八候选本身已由正确fMRI生成。','',
        'CLIP、AlexNet5和Inception均采用float64 cosine、预测行到固定500张GT列的前向二选一，平局半分。CLIP预处理与C一致，GT特征逐元素核验一致。非CLIP特征识别不是旧官方相关性指标。Uniform为八候选先逐个评分再图内平均；并未平均embedding，也没有使用旧两候选分数代替本轮基线。','',
        '## 完整500图结果','',
        '| 方法 | CLIP cosine | CLIP二选一% | Top1% | Top5% | AlexNet5二选一% | Inception二选一% | PixCorr | SSIM |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,m in result['summary'].items():
        vals=[m['CLIP_cosine'],m['CLIP_forward_two_way']*100,m['CLIP_top1']*100,m['CLIP_top5']*100,m['AlexNet5_forward_two_way']*100,m['Inception_forward_two_way']*100,m['PixCorr'],m['SSIM']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','Oracle两行均使用GT，仅诊断、不可部署。Oracle-cosine仅是当前池的cosine上限，Oracle-identification仅是前向二选一上限；它们不是其他指标或视觉质量的共同上限，也不界定所有seed或采样器的能力。','',
        '## 配对比较','',
        '固定seed、10000次图级bootstrap；以下两项是主比较，均为CLIP前向二选一，报告97.5%区间。区间条件于当前固定池、样本、生成器和探针，不代表跨被试或跨训练seed。','',
        '| 比较 | 增益（百分点） | 97.5%区间（百分点） |','| --- | ---: | --- |']
    for name in ['Uniform','Mean-selected']:
        m=result['C_minus'][name]['CLIP_forward_two_way']
        lines.append(f"| C−{name} | {100*m['mean']:.4f} | [{100*m['ci97_5'][0]:.4f}, {100*m['ci97_5'][1]:.4f}] |")
    lines+=['','C−Uniform未达到预先规定的2百分点投入筛查幅度，不能将0.0472的cosine增加等同于主识别改善。C−Mean区间高于零，五组错配的识别率也低于C，支持正确对应关系有用。','',
        '辅助指标：C−Uniform的Inception前向识别增益约2.7683百分点，97.5%区间[1.4523,4.0790]；AlexNet5、PixCorr和SSIM差值区间跨零。辅助比较未作为额外多重比较校正后的正式结论。','',
        '## 固定32图视觉审查','',
        '所有32张均先在隐藏选择器标记的3×3图册中逐张看过，再冻结记录（提交a11bd10），之后才读取选择结果。评阅者为AI视觉审阅，不是多人盲评或预注册心理物理评分。“类别场景匹配”仅指粗粒度主要类别和场景近似，不代表数量、布局和细节恢复；详见逐图文字。','',
        f"32图中{counts['pool_has_category_scene_match']}个候选池至少有一个粗粒度类别场景匹配；C选中{counts['C_hits_category_scene_match']}个，Mean选中{counts['Mean_hits_category_scene_match']}个，Uniform期望为{counts['Uniform_expected_category_scene_matches']:.3f}个；其中{counts['pool_has_match_but_C_misses']}个池有匹配但C没选到。这是固定小样本描述，不外推为500图成功率。",'',
        '滑板16800几乎全部变为飞机；货列29511全部缺失列车；甜甜圈56963全部变为海浪；门廊51865全部变成厨房或浴室。这些失败无法仅靠现有池内选图恢复。另一些图有明确类别匹配候选，例如斑马37899、公交62587、电脑桌71264，但人数、物体数量和位置仍不忠实。','',
        '## 研究含义','',
        '该结果支持继续定位候选排序的适配问题，而不支持立刻重训生成器或继续扫学习率。Oracle显示池内有指标空间，但盲看也证实部分图的全部候选都缺失核心内容；两者同时成立。当前C相似度并未可靠实现该上限。','',
        '即使未来选图有效，改善的也是R+C两条分支组成的系统，不是R参数变好，更不能定位ParcelMapper根因。研究R内部ROI作用时，应固定候选随机输入，不允许干预后由读取完整脑输入的C重新选图来混淆因果解释。','',
        '## 全部32图与选择索引','',
        '索引从0开始；两种oracle明确使用GT。图册只重排既有图，不是新生成。']
    for r in joined:
        i=r['image_id']
        lines += ['',f'### 图片 {i}',r['target_description']+'。'+r['blind_notes']+'。'+r['count_position_layout']+'。',
                  f"C={r['C-selected']}；Mean={r['Mean-selected']}；Oracle-cosine={r['Oracle-cosine']}；Oracle-identification={r['Oracle-identification']}。",f'![固定八候选 {i}](annotated_gallery/{i}.jpg)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
