"""Fixed two-candidate validation using the original inference and eight metrics."""
import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import diagnose_condition_path as diag
import evaluate_validation as ev
from retrain_lr_core import settings, output, NODES, CANONICAL
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import verify_inference_snapshot
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.protocol import method_fingerprint
from run_residual_rerank import save_csv


def labels():
    return [f'{arm}-{u}' for arm in ('T1','T2') for u in NODES]+[f'OLD-{u}' for u in (*NODES,239063)]


def snapshot(root,label):
    base,_,_=settings(root,'T1')
    arm,step=label.split('-');step=int(step)
    if arm=='OLD': folder=base.paths['output_dir']
    else:
        assert arm in ('T1','T2') and step in NODES
        folder=output(root)/arm
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='completed' and status['completed_updates']==NODES[-1]
    path=folder/'snapshots'/f'snapshot-update-{step:08d}'
    verify_inference_snapshot(path)
    if arm!='OLD':
        meta=json.loads((path/'metadata.json').read_text())
        assert meta['canonical_sha256']==CANONICAL and meta['arm']==arm and not meta['test_only']
    return path


@torch.no_grad()
def prepare(root):
    base,_,_=settings(root,'T1');out=output(root)/'evaluation';out.mkdir(exist_ok=True)
    target=out/'protocol.json'
    if target.exists() and (out/'replay.json').exists():
        assert json.loads((out/'replay.json').read_text())['passed']
        return
    old=root/'runs/experiments/paired-lr-probe-v1/evaluation'
    oldplan=json.loads((old/'protocol.json').read_text())
    plan=json.loads(base.paths['selection_plan'].read_text())
    ids=np.loadtxt(base.paths['validation_ids'],dtype=int).tolist()
    assert ids==oldplan['ids'] and len(ids)==500 and plan['screening_candidates']==2
    noise={str(i):str((old/'noise'/f'{i}.pt').relative_to(root)) for i in ids}
    for i,p in noise.items(): assert sha256_file(root/p)==oldplan['noise_sha256'][i]
    refs={}
    for u in (*NODES,239063):
        label=f'OLD-{u}'
        try:
            snap=snapshot(root,label)
        except (OSError,ValueError) as exc:
            refs[label]={'available':False,'reason':repr(exc)}
            continue
        folder=base.paths['output_dir']/'evaluation-20260910/screening'/f'update-{u:08d}'/'decode'
        try:
            m=json.loads((folder/'decode_manifest.json').read_text())
            for k,v in {'candidate_count':2,'status':'complete','selection_stage':'screening','denoising_steps':50,
                        'guidance_scale':4,'config_sha256':base.sha256,'protocol_namespace':plan['protocol_namespace'],
                        'repository_commit':base.raw['protocol_commit'],'snapshot_model_sha256':sha256_file(snap/'model.pt'),
                        'method_fingerprint':method_fingerprint(base)}.items(): assert m[k]==v,k
            assert [r['image_id'] for r in m['records']]==ids
            for r in m['records']:
                assert [f['candidate_index'] for f in r['files']]==[0,1]
                for f in r['files']: assert sha256_file(folder/f['path'])==f['sha256']
            refs[label]={'available':True,'folder':str(folder),'reuse_existing':True,'manifest_sha256':sha256_file(folder/'decode_manifest.json')}
        except (OSError,ValueError,AssertionError,KeyError) as exc:
            refs[label]={'available':True,'folder':str(out/label),'reuse_existing':False,'reason':repr(exc)}
    write_json_atomic(target,{'ids':ids,'noise_paths':noise,'noise_sha256':oldplan['noise_sha256'],
        'namespace':plan['protocol_namespace'],'references':refs,'candidate_count':2,'candidate_batch_size':2,
        'denoising_steps':50,'guidance':4,'metrics':'original Pearson-row GT-to-reconstruction strict two-way; percentages; fixed500 negatives',
        'visual_ids':[ids[j] for j in np.linspace(0,499,32,dtype=int)],'old_noise_protocol_sha256':sha256_file(old/'protocol.json')})
    configure_torch_backend(base.training)
    ds=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['validation_ids'])
    checks=[]
    with patch.object(diag,'NAMESPACE',plan['protocol_namespace']):
        for label,r in refs.items():
            if not r.get('reuse_existing'): continue
            backbone,bundle=diag.load_models(base,snapshot(root,label),torch.device('cuda:0'),torch.bfloat16)
            i=ids[0];values=diag.generate(backbone,bundle,ds[0]['brain'],i,'validation',torch.device('cuda:0'),torch.bfloat16,4.,torch.load(root/noise[str(i)],weights_only=True))
            m=json.loads((Path(r['folder'])/'decode_manifest.json').read_text())
            for c in range(2):
                actual=(values[c].permute(1,2,0).float().numpy()*255).round().astype(np.uint8)
                expected=np.array(Image.open(Path(r['folder'])/m['records'][0]['files'][c]['path']))
                assert np.array_equal(actual,expected),'old screening/noise replay differs'
                checks.append({'label':label,'image_id':i,'candidate':c,'pixels_exact':True})
            del backbone,bundle;torch.cuda.empty_cache()
    ds.close();write_json_atomic(out/'replay.json',{'passed':True,'checks':checks})


@torch.no_grad()
def decode(root,label,smoke=False):
    base,_,_=settings(root,'T1');out=output(root)/'evaluation'
    protocol=json.loads((out/'protocol.json').read_text())
    if not smoke and label.startswith('OLD') and protocol['references'][label]['reuse_existing']: return
    source=output(root)/'tests/T2-full/snapshots/snapshot-update-00000020' if smoke else snapshot(root,label)
    target=out/('smoke' if smoke else label);target.mkdir(exist_ok=True)
    if (target/'decode_manifest.json').exists(): return
    configure_torch_backend(base.training)
    backbone,bundle=diag.load_models(base,source,torch.device('cuda:0'),torch.bfloat16)
    ds=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['validation_ids'])
    records=[]
    with patch.object(diag,'NAMESPACE',protocol['namespace']):
        for j,i in enumerate(protocol['ids'][:2] if smoke else protocol['ids']):
            record=target/f'{i}.json'
            if record.exists():
                r=json.loads(record.read_text());assert r['image_id']==i
                assert r['source_sha256']==sha256_file(source/'model.pt')
                for f in r['files']: assert sha256_file(target/f['path'])==f['sha256']
                records.append(r);continue
            noise=root/protocol['noise_paths'][str(i)];assert sha256_file(noise)==protocol['noise_sha256'][str(i)]
            sample=ds[j];assert int(sample['nsd_image_id'])==i
            values=diag.generate(backbone,bundle,sample['brain'],i,'validation',torch.device('cuda:0'),torch.bfloat16,4.,torch.load(noise,weights_only=True))
            files=[]
            for c,value in enumerate(values):
                rel=f'candidate-{c:02d}/{i:05d}.png';diag.png(target/rel,value)
                files.append({'candidate_index':c,'path':rel,'sha256':sha256_file(target/rel)})
            r={'image_id':i,'files':files,'source_sha256':sha256_file(source/'model.pt')}
            write_json_atomic(record,r);records.append(r)
            if (j+1)%25==0: print(f'{label}: {j+1}/500',flush=True)
    write_json_atomic(target/'decode_manifest.json',{'status':'complete','split':'validation','candidate_count':2,
        'records':records,'experiment_type':'retrain_lr_v1','snapshot_sha256':sha256_file(source/'model.pt'),
        'protocol_sha256':sha256_file(out/'protocol.json'),'smoke_only':smoke})
    ds.close()


@torch.no_grad()
def score(root,label,smoke=False):
    base,_,_=settings(root,'T1');out=output(root)/'evaluation'
    result=out/f'{label}-summary.json'
    if result.exists(): return
    protocol=json.loads((out/'protocol.json').read_text())
    folder=Path(protocol['references'][label]['folder']) if label.startswith('OLD') else out/label
    configure_torch_backend(base.training)
    if smoke:
        import h5py
        m=json.loads((folder/'decode_manifest.json').read_text());ids=[r['image_id'] for r in m['records']]
        with h5py.File(base.paths['stimuli'],'r') as h: gt=np.stack([h['imgBrick'][i] for i in ids])
        images=[np.stack([np.array(Image.open(folder/r['files'][c]['path']).convert('RGB')) for r in m['records']]) for c in range(2)]
    else:
        ids,gt,images,_=ev.load_decode_set(folder/'decode_manifest.json',base.paths['stimuli'])
        assert ids==protocol['ids'] and len(images)==2
    scores=score_arrays(root,base,gt,images)
    rows=[{'image_id':i,'candidate':c,**{m:float(v[c][j]) for m,v in scores.items()}} for j,i in enumerate(ids) for c in range(2)]
    save_csv(out/f'{label}-candidate-scores.csv',rows)
    means=[{'image_id':i,**{m:float(np.mean([v[c][j] for c in range(2)])) for m,v in scores.items()}} for j,i in enumerate(ids)]
    save_csv(out/f'{label}-image-scores.csv',means)
    write_json_atomic(result,{'mean':{m:float(np.mean(v)) for m,v in scores.items()},
        'candidate0':{m:float(np.mean(v[0])) for m,v in scores.items()},'image_count':len(ids),'smoke_only':smoke})


def score_arrays(root,base,gt,images):
    # Same frozen evaluator calls and scales as evaluate_lr_probe.score; no new metric definitions.
    assets=ev.verify_evaluation_assets(root,base.paths['evaluation_manifest'])
    os.environ['TORCH_HOME']=str(root/'models/evaluation/torch')
    torch.hub.set_dir(str(root/'models/evaluation/torch/hub'))
    batch=int(json.loads(base.paths['selection_plan'].read_text())['evaluation_batch_size'])
    device=torch.device('cuda:0');scores={'PixCorr':[],'SSIM':[]}
    for im in images:
        pieces=[ev.pixel_metrics(gt[s:s+16],im[s:s+16]) for s in range(0,len(gt),16)]
        scores['PixCorr'].append(np.concatenate([p[0] for p in pieces]));scores['SSIM'].append(np.concatenate([p[1] for p in pieces]))
    def feats(model,size,mean,std,key=None):
        return ev.model_features(gt,images,model,ev.preprocess_transform(size,mean,std),device=device,batch_size=batch,output_key=key)
    alex=ev.create_feature_extractor(ev.alexnet(weights=ev.AlexNet_Weights.IMAGENET1K_V1),return_nodes={'features.4':'layer2','features.11':'layer5'}).to(device).eval().requires_grad_(False)
    for key,name in [('layer2','AlexNet2'),('layer5','AlexNet5')]:
        orig,cand=feats(alex,256,ev.IMAGENET_MEAN,ev.IMAGENET_STD,key);scores[name]=ev.identification_by_seed(orig,cand)
    del alex,orig,cand
    model=ev.create_feature_extractor(ev.inception_v3(weights=ev.Inception_V3_Weights.DEFAULT),return_nodes={'avgpool':'avgpool'}).to(device).eval().requires_grad_(False)
    orig,cand=feats(model,342,ev.IMAGENET_MEAN,ev.IMAGENET_STD,'avgpool');scores['Inception']=ev.identification_by_seed(orig,cand)
    del model,orig,cand
    sys.path.insert(0,str(root/'repo/vendor/CLIP'));import clip
    model,_=clip.load(str(assets['clip_vit_l_14']),device=device,jit=False)
    orig,cand=feats(model.encode_image,224,ev.CLIP_MEAN,ev.CLIP_STD);scores['CLIP_identification']=ev.identification_by_seed(orig,cand)
    del model,orig,cand
    for name,factory,size in [('EfficientNet',lambda:ev.efficientnet_b1(weights=ev.EfficientNet_B1_Weights.DEFAULT),255),
        ('SwAV',lambda:torch.hub.load(str(root/'repo/vendor/swav'),'resnet50',source='local',pretrained=True),224)]:
        model=ev.create_feature_extractor(factory(),return_nodes={'avgpool':'avgpool'}).to(device).eval().requires_grad_(False)
        orig,cand=feats(model,size,ev.IMAGENET_MEAN,ev.IMAGENET_STD,'avgpool');scores[name]=[ev.paired_correlation_distance(orig,f) for f in cand]
        del model,orig,cand
    assert len(scores)==8 and all(np.isfinite(v).all() for v in scores.values())
    return scores


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['prepare','decode','score','smoke'],required=True)
    p.add_argument('--label',choices=labels());a=p.parse_args()
    if a.phase=='prepare':prepare(a.root)
    elif a.phase=='smoke':
        decode(a.root,'smoke',True);score(a.root,'smoke',True)
    elif a.phase=='decode':decode(a.root,a.label)
    else:score(a.root,a.label)
