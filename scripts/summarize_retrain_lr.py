"""Six-snapshot selection, paired comparisons, and loadable best-new export."""
import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

import diagnose_condition_path as diag
from evaluate_retrain_lr import labels, snapshot
from retrain_lr_core import settings, output, NODES, CANONICAL, code_identity
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.data import Subject1TrainingDataset
from run_residual_rerank import read, save_csv


def rank_key(label,metrics):
    arm,update=label.split('-')
    semantic=np.mean([metrics[k] for k in ('AlexNet5','Inception','CLIP_identification')])
    return (-float(semantic),-metrics['PixCorr'],int(update),arm)


@torch.no_grad()
def run(root):
    base,spec,_=settings(root,'T1');out=output(root);ev=out/'evaluation'
    for arm in ('T1','T2'):
        st=json.loads((out/arm/'status.json').read_text());assert st['status']=='completed' and st['completed_updates']==159375
    protocol=json.loads((ev/'protocol.json').read_text())
    available=[l for l in labels() if not l.startswith('OLD') or protocol['references'][l]['available']]
    summaries={l:json.loads((ev/f'{l}-summary.json').read_text()) for l in available}
    assert all(m['image_count']==500 and not m['smoke_only'] for m in summaries.values())
    arrays={};all_rows=[]
    for l in available:
        rows=read(ev/f'{l}-image-scores.csv');assert [int(r['image_id']) for r in rows]==protocol['ids']
        arrays[l]={k:np.array([float(r[k]) for r in rows]) for k in summaries[l]['mean']}
        all_rows += [{'label':l,**r} for r in rows]
    save_csv(ev/'per_image_metrics.csv',all_rows)
    new=[l for l in available if not l.startswith('OLD')];assert len(new)==6
    best=min(new,key=lambda l:rank_key(l,summaries[l]['mean']))
    indices=np.random.default_rng(20260901).integers(500,size=(10000,500))
    comparisons={}
    for l in new:
        for ref in [f"OLD-{l.split('-')[1]}",'OLD-239063']:
            if ref not in arrays:
                comparisons[f'{l}_minus_{ref}']={'missing_baseline':True};continue
            comparisons[f'{l}_minus_{ref}']={}
            for metric in arrays[l]:
                delta=arrays[l][metric]-arrays[ref][metric]
                comparisons[f'{l}_minus_{ref}'][metric]={'mean':float(delta.mean()),
                    'ci95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
                    'higher_is_better':metric not in ('EfficientNet','SwAV')}
    write_json_atomic(ev/'comparisons.json',{'bootstrap_draws':10000,'seed':20260901,'fixed_negative_pool':500,
        'scope':'exploratory reused internal validation; six-way selection; no multiplicity correction','comparisons':comparisons})
    write_json_atomic(ev/'metrics_summary.json',{'models':summaries,'best_new_candidate':best,
        'semantic_score':-rank_key(best,summaries[best]['mean'])[0],
        'improved_over_previous_semantic_point_estimate':None if 'OLD-239063' not in summaries else rank_key(best,summaries[best]['mean'])[0]<rank_key('OLD-239063',summaries['OLD-239063']['mean'])[0],
        'qualified_for_next_experiment':False})
    target=out/'best_new_candidate';target.mkdir(exist_ok=True)
    source=snapshot(root,best);model_hash=sha256_file(source/'model.pt')
    if (target/'model.pt').exists(): assert sha256_file(target/'model.pt')==model_hash
    else: shutil.copyfile(source/'model.pt',target/'model.pt')
    assert sha256_file(target/'model.pt')==model_hash
    (target/'model.sha256').write_text(model_hash+'  model.pt\n')
    arm,update=best.split('-')
    effective=json.loads((out/arm/'effective_config.json').read_text())
    (target/'config.yaml').write_text(yaml.safe_dump(effective,sort_keys=False))
    metadata={'best_new_candidate':best,'source_snapshot':str(source),'model_sha256':model_hash,
        'training_images':8500,'selection_images':500,'canonical_sha256':CANONICAL,'training_source':effective['source'],
        'arm':arm,'update':int(update),'schedule':effective['schedule'],'base_model':'stable-diffusion-v1-5',
        'base_model_manifest_sha256':json.loads((out/'assets.json').read_text())['input_hashes']['model_manifest'],
        'base_model_tree_sha256':json.loads((out/'assets.json').read_text())['stable_diffusion_tree'],
        'qualified_for_next_experiment':False,'export_source':code_identity(root)}
    write_json_atomic(target/'metrics_validation.json',summaries[best])
    configure_torch_backend(base.training)
    ds=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['validation_ids'])
    backbone,bundle=diag.load_models(base,target/'model.pt',torch.device('cuda:0'),torch.bfloat16)
    i=protocol['ids'][0]
    with patch.object(diag,'NAMESPACE',protocol['namespace']):
        result=diag.generate(backbone,bundle,ds[0]['brain'],i,'validation',torch.device('cuda:0'),torch.bfloat16,4.,
            torch.load(root/protocol['noise_paths'][str(i)],weights_only=True))
    for c in range(2):
        expected=np.array(Image.open(ev/best/f'candidate-{c:02d}/{i:05d}.png'))
        actual=(result[c].permute(1,2,0).float().numpy()*255).round().astype(np.uint8)
        assert np.array_equal(actual,expected),'export reload differs from snapshot output'
    metadata['reload_two_candidates_pixels_exact']=True;write_json_atomic(target/'metadata.json',metadata)
    # Fixed32: GT, old239063 and six new models, both candidates retained.
    gallery=ev/'content_gallery';gallery.mkdir(exist_ok=True)
    shown=['OLD-239063',*new]
    for image_id in protocol['visual_ids']:
        j=protocol['ids'].index(image_id);gt=ds[j]['image']
        tiles=[('GT',Image.fromarray(((gt+1)/2*255).round().clamp(0,255).byte().permute(1,2,0).numpy()))]
        for l in shown:
            if l not in available:
                tiles += [(l+' MISSING',Image.new('RGB',(320,320),'gray'))]*2;continue
            folder=Path(protocol['references'][l]['folder']) if l.startswith('OLD') else ev/l
            m=json.loads((folder/'decode_manifest.json').read_text())
            for c in range(2): tiles.append((f'{l} cand{c}',Image.open(folder/m['records'][j]['files'][c]['path'])))
        canvas=Image.new('RGB',(1600,1020),'white');draw=ImageDraw.Draw(canvas)
        for n,(title,im) in enumerate(tiles):
            x=n%5*320;y=n//5*340
            draw.text((x+4,y+4),title,fill='black');canvas.paste(im.convert('RGB').resize((320,320)),(x,y+20))
        canvas.save(gallery/f'{image_id}.jpg',quality=95)
    ds.close()
    metrics=list(summaries[best]['mean'])
    lines=['# T1/T2 从canonical重新训练结果','',
        '两条训练均完成159375次更新。以下是内部500图开发评价，非独立测试；不经过选图，两个候选先分别评分再图内平均。',
        '','八项沿用原官方式评价函数；AlexNet2/5、Inception、CLIP为原行中心化特征GT→重建严格二选一百分数。EfficientNet/SwAV相关距离越低越好，其他六项越高越好。',
        '', '| 权重 | '+' | '.join(metrics)+' | 语义综合分 |','| --- |'+' ---: |'*(len(metrics)+1)]
    for l in available:
        m=summaries[l]['mean'];lines.append('| '+l+' | '+' | '.join(f'{m[k]:.6f}' for k in metrics)+f' | {-rank_key(l,m)[0]:.6f} |')
    lines+=['','## 最佳新候选',f'本轮最佳：{best}。model.pt SHA256：`{model_hash}`。',
        '仅表示六个新snapshot中排名第一。是否超过原权重需结合各项差值与内容表现；不自动认定可开展生物学结论实验，不覆盖旧锁。',
        '','每候选与candidate0结果、完整500图分数、旧新同更新及239063的配对区间均在evaluation目录保存。缺少旧资产时明确标记missing_baseline，不伪造。',
        '','## 加载',f'`{root}/envs/neuroadapter/bin/python {Path(__file__).parent}/load_retrain_lr_candidate.py --root {root} --destination {target}/load-example`',
        '','## 内容展示状态','图册已自动生成，尚待实际打开进行AI定性检查；此时不能声称主体、场景、动作或布局已修复。不是新一轮大规模评阅。']
    for i in protocol['visual_ids']:lines += ['',f'### 图像 {i}',f'![GT和七组权重的两个候选](evaluation/content_gallery/{i}.jpg)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (target/'MODEL_CARD_ZH.md').write_text(f'# 最佳新候选 {best}\n\n训练8500图、内部500图六选一，非已验收正式模型。\n\n权重SHA256：{model_hash}\n\n'+lines[2]+'\n',encoding='utf-8')
    write_json_atomic(out/'delivery.json',{'training_and_fixed_evaluation_complete':True,'best_new_candidate':best,
        'model_sha256':model_hash,'visual_review_complete':False,'next_stage':'fixed32 qualitative viewing only; no more training'})
    print(f'EXPORTED {best} {model_hash}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
