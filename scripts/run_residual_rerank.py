"""One fixed residual selector; reuse C/D artifacts without fitting or inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_semantic_rerank import sha, write


def residual_select(predictions, mean, candidates):
    p, mu, g = [np.asarray(a, dtype=np.float64) for a in (predictions, mean, candidates)]
    if p.ndim != 2 or mu.shape != (p.shape[1],) or g.ndim != 3 or g.shape[0] != len(p) or g.shape[2] != p.shape[1]:
        raise ValueError('expected raw [N,D], mean [D], candidates [N,K,D]')
    if g.shape[1] == 0 or not all(np.isfinite(a).all() for a in (p,mu,g)):
        raise ValueError('empty or non-finite features')
    norms = np.linalg.norm(g, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('zero candidate feature')
    u = g / norms
    common = np.einsum('d,nkd->nk',mu,u)
    residual = np.einsum('nd,nkd->nk',p-mu,u)
    return np.argmax(residual,axis=1), common, residual


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(root):
    base=root/'runs/diagnostics'
    c,d=base/'semantic-probe-c-v1',base/'semantic-rerank-d-v1'
    out=base/'semantic-rerank-e-residual-v1'
    out.mkdir(exist_ok=False)
    plan=json.loads((d/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        assert sha(root/name)==expected,name
    featuremeta=json.loads((d/'features_manifest.json').read_text())
    for name in ['CLIP_features.npz','candidate_scores.npz']:
        assert sha(d/name)==featuremeta['files'][name]
    ids=plan['image_ids']
    pred=np.load(c/'C-predictions.npz');means=np.load(c/'mean-predictions.npz')
    feat=np.load(d/'CLIP_features.npz');raw=np.load(d/'candidate_scores.npz')
    for data in [pred,means,feat,raw]:assert data['image_ids'].tolist()==ids
    assert feat['candidate_indices'].tolist()==list(range(8))
    p=pred['features'].astype(np.float64)
    assert np.array_equal(means['features'],np.broadcast_to(means['features'][0],means['features'].shape))
    mu=means['features'][0].astype(np.float64)
    gt=np.load(c/'gt_features.npz')
    train=np.loadtxt(c/'split_ids/train8500.txt',dtype=int).tolist()
    assert len(train)==8500 and len(set(train))==8500 and set(train).isdisjoint(ids)
    lookup={int(i):j for j,i in enumerate(gt['image_ids'])}
    y=gt['features'][[lookup[i] for i in train]].astype(np.float64)
    expected=(y/np.linalg.norm(y,axis=1,keepdims=True)).mean(0)
    mean_error=float(np.max(abs(mu-expected)));assert mean_error<1e-12
    sources=[c/'C-predictions.npz',c/'mean-predictions.npz',c/'gt_features.npz',c/'split_ids/train8500.txt',
             d/'plan.json',d/'CLIP_features.npz',d/'candidate_scores.npz',d/'visual_review.csv',d/'selector_indices.csv']
    write(out/'plan.json',{'formula':'(raw_C_prediction - mean_of_unit_train8500_CLIP) dot unit_candidate',
        'namespace':out.name,'bootstrap_seed':20261001,'bootstrap_draws':10000,'interval':.975,
        'primary_comparisons':['E-selected minus Uniform','E-selected minus C-selected'],
        'primary_metric':'CLIP_forward_two_way','screening_gain':.02,'no_search':True,
        'stop_rule':'retain development candidate only if E-Uniform>=.02, both primary lower CI>0 and no clear nonCLIP/visual reversal; otherwise no automatic search or retraining',
        'source_sha256':{str(path.relative_to(root)):sha(path) for path in sources},
        'code_sha256':sha(Path(__file__)),'mismatch_indices':plan['mismatch_indices'],
        'image_ids':ids,'visual_ids':plan['visual_ids'],'mean_crosscheck_max_error':mean_error,
        'new_training':0,'new_feature_extraction':0,'new_diffusion_images':0,'standard_test_access':False})
    g=feat['candidates'].transpose(1,0,2)
    selected, common, specific=residual_select(p,mu,g)
    oldrows=read(d/'selector_indices.csv')
    old={name:np.array([int(r['candidate_index']) for r in oldrows if r['method']==name])
         for name in ['C-selected','Mean-selected','Oracle-cosine','Oracle-identification']}
    for name in old:
        assert [int(r['image_id']) for r in oldrows if r['method']==name]==ids
    # Check decomposition recovers the original ranking, without changing D.
    assert np.array_equal((common+specific).argmax(1),old['C-selected'])
    selections={'E-selected':selected}
    ranking={'E-selected':specific}
    for j,seed in enumerate(plan['mismatch_seeds']):
        perm=np.array(plan['mismatch_indices'][str(seed)])
        assert sorted(perm)==list(range(500)) and np.all(perm!=np.arange(500))
        ix,_,s=residual_select(p[perm],mu,g)
        selections[f'Mismatched-E-{j+1}']=ix;ranking[f'Mismatched-E-{j+1}']=s
    save_csv(out/'selector_indices.csv',[{'method':name,'image_id':i,'candidate_index':int(ix[j]),
        'scores':json.dumps(ranking[name][j].tolist())} for name,ix in selections.items() for j,i in enumerate(ids)])
    save_csv(out/'decomposition.csv',[{'image_id':i,'common_scores':json.dumps(common[j].tolist()),
        'residual_scores':json.dumps(specific[j].tolist()),'common_std':float(common[j].std()),
        'residual_std':float(specific[j].std()),'same_as_C':bool(selected[j]==old['C-selected'][j]),
        'same_as_Mean':bool(selected[j]==old['Mean-selected'][j])} for j,i in enumerate(ids)])
    metrics={k:raw[k] for k in raw.files if k!='image_ids'}
    results={'Uniform':{k:a.mean(1) for k,a in metrics.items()}}
    for name,ix in {**old,**selections}.items():results[name]={k:a[np.arange(500),ix] for k,a in metrics.items()}
    save_csv(out/'per_image_scores.csv',[{'method':name,'image_id':i,**{k:float(a[j]) for k,a in m.items()}}
        for name,m in results.items() for j,i in enumerate(ids)])
    draws=np.random.default_rng(20261001).integers(500,size=(10000,500))
    comparisons={}
    for name in ['Uniform','C-selected']+[f'Mismatched-E-{j+1}' for j in range(5)]:
        comparisons[name]={}
        for k in metrics:
            delta=results['E-selected'][k]-results[name][k]
            comparisons[name][k]={'mean':float(delta.mean()),'ci97_5':np.quantile(delta[draws].mean(1),[.0125,.9875]).tolist(),
                'primary':name in ['Uniform','C-selected'] and k=='CLIP_forward_two_way'}
    summary={name:{k:float(v.mean()) for k,v in m.items()} for name,m in results.items()}
    write(out/'paired_comparisons.json',{'summary':summary,'E_minus':comparisons,
        'scope':'exploratory reused internal validation; fixed pool and models; auxiliary comparisons descriptive',
        'oracle_background_only':True,'permutation_p_value':None})
    reviews=[]
    for r in read(d/'visual_review.csv'):
        i=int(r['image_id']);j=ids.index(i)
        acceptable={int(k) for k in r['category_scene_acceptable_indices'].split(';') if k}
        e_hit=int(selected[j]) in acceptable;c_hit=int(old['C-selected'][j]) in acceptable
        reviews.append({**r,'E-selected':int(selected[j]),'E_hits_category_scene_match':e_hit,
            'E_recovers_C_miss':bool(acceptable) and e_hit and not c_hit,'E_loses_C_hit':c_hit and not e_hit,
            'E_differs_C':bool(selected[j]!=old['C-selected'][j])})
    save_csv(out/'visual_review.csv',reviews)
    counts={k:sum(bool(r[k]) for r in reviews) for k in ['E_hits_category_scene_match','E_recovers_C_miss','E_loses_C_hit','E_differs_C']}
    write(out/'visual_summary.json',{'counts':counts,'frozen_D_labels_unchanged':True,
        'all500_E_differs_C':int(np.sum(selected!=old['C-selected'])),
        'all500_E_same_Mean':int(np.sum(selected==old['Mean-selected'])),'review_pending':True})
    # Show every changed choice among the fixed32, with unchanged D labels.
    inventory=json.loads((d/'candidate_manifest.json').read_text())
    decode=(root/inventory['source']).parent
    paths={(r['image_id'],r['candidate_index']):decode/r['path'] for r in inventory['records']}
    gallery=out/'changed_gallery';gallery.mkdir()
    changed=[r for r in reviews if r['E_differs_C']]
    for start in range(0,len(changed),4):
        sheet=Image.new('RGB',(960,340*len(changed[start:start+4])),'white');draw=ImageDraw.Draw(sheet)
        for row,r in enumerate(changed[start:start+4]):
            i=int(r['image_id'])
            gt_tile=Image.open(d/'blind_gallery'/f'{i}.jpg').crop((0,20,320,340))
            tiles=[gt_tile,Image.open(paths[i,int(r['C-selected'])]),Image.open(paths[i,int(r['E-selected'])])]
            for col,im in enumerate(tiles):
                sheet.paste(im.convert('RGB').resize((320,320)),(col*320,row*340+20))
                draw.text((col*320+3,row*340+3),f"ID {i} "+(['GT',f"C {r['C-selected']}",f"E {r['E-selected']}"][col]),fill='black')
        sheet.save(gallery/f'page-{start//4+1:02d}.jpg',quality=95)
    print(json.dumps({'summary':summary,'visual':counts,'changed500':int(np.sum(selected!=old['C-selected']))},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
