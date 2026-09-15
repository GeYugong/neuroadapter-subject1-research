"""Score a frozen candidate pool without training or diffusion generation."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from prepare_semantic_rerank import sha, write
from semantic_probe_core import retrieval
from semantic_rerank_core import select_candidates


def setup(root):
    out = root / 'runs/diagnostics/semantic-rerank-d-v1'
    plan = json.loads((out / 'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        assert sha(root / name) == expected, name
    return out, plan


def rows_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(root):
    import torch
    import evaluate_validation as ev
    from PIL import Image, ImageDraw
    from neuroadapter_research.config import load_training_config
    from neuroadapter_research.backend import configure_torch_backend
    out, plan = setup(root)
    if (out/'features_manifest.json').exists():
        raise RuntimeError('completed features exist; do not overwrite')
    config = load_training_config(root/'configs/formal/subject01_selection_v2.yaml', require_frozen=True)
    configure_torch_backend(config.training)
    assets = ev.verify_evaluation_assets(root, config.paths['evaluation_manifest'])
    os.environ['TORCH_HOME'] = str(root/'models/evaluation/torch')
    torch.hub.set_dir(str(root/'models/evaluation/torch/hub'))
    inventory = json.loads((out/'candidate_manifest.json').read_text())
    ids, gt, candidates, _ = ev.load_decode_set(root/inventory['source'], config.paths['stimuli'])
    assert ids == plan['image_ids'] and len(candidates) == 8
    started = time.time()
    scores = {}
    device = torch.device('cuda:0')
    batch = 16

    def extract(name, model, size, mean, std, key=None):
        path = out/f'{name}_features.npz'
        if path.exists():
            raise RuntimeError(f'partial features exist: {path}; inspect before retry')
        prep = ev.preprocess_transform(size, mean, std)
        def one(images):
            return ev.extract_features(images, model, prep, device=device, batch_size=batch, output_key=key)
        target = one(gt)
        generated = []
        for k, images in enumerate(candidates):
            generated.append(one(images))
            print(f'{name} candidate {k+1}/8 complete', flush=True)
        generated = np.stack(generated)
        np.savez(path, image_ids=ids, candidate_indices=np.arange(8), target=target, candidates=generated)
        if name == 'CLIP':
            previous = np.load(root/'runs/diagnostics/semantic-probe-c-v1/gt_features.npz')
            assert previous['image_ids'][-500:].tolist() == ids
            assert np.array_equal(previous['features'][-500:], target), 'CLIP target differs from frozen C'
        per = [retrieval(a,target) for a in generated]
        if name == 'CLIP':
            for metric in per[0]:
                scores[f'CLIP_{metric}'] = np.stack([p[metric] for p in per],axis=1)
        else:
            scores[f'{name}_forward_two_way'] = np.stack([p['forward_two_way'] for p in per],axis=1)

    sys.path.insert(0,str(root/'repo/vendor/CLIP'))
    import clip
    model,_ = clip.load(str(assets['clip_vit_l_14']), device=device, jit=False)
    model.eval().requires_grad_(False)
    extract('CLIP',model.encode_image,224,ev.CLIP_MEAN,ev.CLIP_STD)
    del model
    torch.cuda.empty_cache()
    alex = ev.create_feature_extractor(ev.alexnet(weights=ev.AlexNet_Weights.IMAGENET1K_V1),
        return_nodes={'features.11':'layer5'}).to(device).eval().requires_grad_(False)
    extract('AlexNet5',alex,256,ev.IMAGENET_MEAN,ev.IMAGENET_STD,'layer5')
    del alex
    torch.cuda.empty_cache()
    inception = ev.create_feature_extractor(ev.inception_v3(weights=ev.Inception_V3_Weights.DEFAULT),
        return_nodes={'avgpool':'avgpool'}).to(device).eval().requires_grad_(False)
    extract('Inception',inception,342,ev.IMAGENET_MEAN,ev.IMAGENET_STD,'avgpool')
    del inception
    torch.cuda.empty_cache()
    pixels, ssims = [], []
    for k, images in enumerate(candidates):
        # Bound temporary float64 pixel matrices without changing per-image operations.
        pieces = [ev.pixel_metrics(gt[s:s+16],images[s:s+16]) for s in range(0,500,16)]
        pixels.append(np.concatenate([p[0] for p in pieces]))
        ssims.append(np.concatenate([p[1] for p in pieces]))
        print(f'pixel metrics candidate {k+1}/8 complete',flush=True)
    scores['PixCorr'] = np.stack(pixels,axis=1)
    scores['SSIM'] = np.stack(ssims,axis=1)
    np.savez(out/'candidate_scores.npz',image_ids=ids,**scores)
    rows_csv(out/'per_candidate_scores.csv',[
        {'image_id':i,'candidate_index':k,**{m:float(a[j,k]) for m,a in scores.items()}}
        for j,i in enumerate(ids) for k in range(8)])
    gallery = out/'blind_gallery'
    gallery.mkdir(exist_ok=False)
    for image_id in plan['visual_ids']:
        j = ids.index(image_id)
        sheet = Image.new('RGB',(960,1020),'white')
        draw = ImageDraw.Draw(sheet)
        for k, array in enumerate([gt[j]]+[a[j] for a in candidates]):
            x,y=(k%3)*320,(k//3)*340
            sheet.paste(Image.fromarray(array).resize((320,320)),(x,y+20))
            draw.text((x+5,y+3),f'GT {image_id}' if k==0 else f'candidate {k-1}',fill='black')
        sheet.save(gallery/f'{image_id}.jpg',quality=95)
    write(out/'features_manifest.json',{
        'files':{p.name:sha(p) for p in out.glob('*.npz')},
        'source_script_sha256':sha(Path(__file__)), 'batch_size':batch,
        'assets':{name:sha(path) for name,path in assets.items()},
        'gt_matches_C_exactly':True,'seconds':time.time()-started,
        'blind_gallery':{p.name:sha(p) for p in gallery.glob('*.jpg')},
        'neuroadapter_updates':0,'probe_fits':0,'new_diffusion_images':0})


def score(root):
    out, plan = setup(root)
    if (out/'paired_comparisons.json').exists():
        raise RuntimeError('scores already completed')
    meta=json.loads((out/'features_manifest.json').read_text())
    for name, expected in meta['files'].items():
        assert sha(out/name)==expected
    data=np.load(out/'CLIP_features.npz')
    assert data['image_ids'].tolist()==plan['image_ids']
    candidates=data['candidates'].transpose(1,0,2)
    croot=root/'runs/diagnostics/semantic-probe-c-v1'
    pred=np.load(croot/'C-predictions.npz')['features']
    mean=np.load(croot/'mean-predictions.npz')['features']
    indices,ranking={},{}
    # This selector calls only the pure, target-free interface.
    for name, p in [('C-selected',pred),('Mean-selected',mean)]+[
        (f'Mismatched-C-{j+1}',pred[plan['mismatch_indices'][str(seed)]]) for j,seed in enumerate(plan['mismatch_seeds'])]:
        indices[name],ranking[name]=select_candidates(p,candidates)
    raw=np.load(out/'candidate_scores.npz')
    metrics={k:raw[k] for k in raw.files if k!='image_ids'}
    assert all(a.shape==(500,8) and np.isfinite(a).all() for a in metrics.values())
    indices['Oracle-cosine']=metrics['CLIP_cosine'].argmax(axis=1)
    indices['Oracle-identification']=metrics['CLIP_forward_two_way'].argmax(axis=1)
    rows=[]
    for name, chosen in indices.items():
        for j,i in enumerate(plan['image_ids']):
            oracle=name.startswith('Oracle')
            rows.append({'method':name,'image_id':i,'candidate_index':int(chosen[j]),
                'uses_ground_truth':oracle,'diagnostic_only':oracle,'deployable':not oracle,
                'ranking_scores':json.dumps(ranking[name][j].tolist()) if name in ranking else ''})
    rows_csv(out/'selector_indices.csv',rows)
    results={'Uniform':{k:a.mean(1) for k,a in metrics.items()}}
    for name, chosen in indices.items():
        results[name]={k:a[np.arange(500),chosen] for k,a in metrics.items()}
    rows_csv(out/'per_image_scores.csv',[
        {'method':name,'image_id':i,**{k:float(a[j]) for k,a in values.items()}}
        for name,values in results.items() for j,i in enumerate(plan['image_ids'])])
    summary={name:{k:float(a.mean()) for k,a in values.items()} for name,values in results.items()}
    rng=np.random.default_rng(plan['bootstrap_seed'])
    draws=rng.integers(500,size=(plan['bootstrap_draws'],500))
    comparisons={}
    for name in ['Uniform','Mean-selected']+[f'Mismatched-C-{j+1}' for j in range(5)]:
        comparisons[name]={}
        for metric in metrics:
            delta=results['C-selected'][metric]-results[name][metric]
            comparisons[name][metric]={'mean':float(delta.mean()),
                'ci97_5':np.quantile(delta[draws].mean(1),[.0125,.9875]).tolist(),
                'primary':name in ['Uniform','Mean-selected'] and metric==plan['primary_metric']}
    write(out/'paired_comparisons.json',{'summary':summary,'C_minus':comparisons,'scope':plan['scope'],
        'mismatches_descriptive_only':True,'permutation_p_value':None,'visual_review_complete':False})
    write(out/'oracle_bounds.json',{
        'uses_ground_truth':True,'diagnostic_only':True,'deployable':False,
        'Oracle-cosine':{'upper_bound_for':'CLIP_cosine only','scores':summary['Oracle-cosine']},
        'Oracle-identification':{'upper_bound_for':'CLIP_forward_two_way only','scores':summary['Oracle-identification']},
        'different_selection_count':int(np.sum(indices['Oracle-cosine']!=indices['Oracle-identification']))})
    print('Frozen-pool scoring complete; blinded visual review and report remain',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['features','score'],required=True)
    args=parser.parse_args()
    globals()[args.phase](args.root)
