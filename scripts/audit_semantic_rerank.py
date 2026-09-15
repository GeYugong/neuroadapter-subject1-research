"""Audit D-specific alignment, target-free selectors and paired statistics."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from prepare_semantic_rerank import sha, write
from report_semantic_rerank import read
from run_semantic_rerank import setup
from semantic_rerank_core import select_candidates


def run(root):
    out, plan=setup(root)
    assert list(inspect.signature(select_candidates).parameters)==['predictions','candidates']
    featuremeta=json.loads((out/'features_manifest.json').read_text())
    for name,expected in featuremeta['files'].items():
        assert sha(out/name)==expected
    ids=plan['image_ids']
    per=read(out/'per_image_scores.csv')
    selected=read(out/'selector_indices.csv')
    raw=np.load(out/'candidate_scores.npz')
    assert raw['image_ids'].tolist()==ids
    clip=np.load(out/'CLIP_features.npz')
    assert clip['image_ids'].tolist()==ids and clip['candidate_indices'].tolist()==list(range(8))
    def norm(a):
        a=np.asarray(a,dtype=np.float64)
        return a/np.linalg.norm(a,axis=-1,keepdims=True)
    candidates=norm(clip['candidates'].transpose(1,0,2))
    croot=root/'runs/diagnostics/semantic-probe-c-v1'
    cp=np.load(croot/'C-predictions.npz')['features']
    mean=np.load(croot/'mean-predictions.npz')['features']
    method_indices={}
    max_score_error=0.0
    for method in sorted({r['method'] for r in selected}):
        rows=[r for r in selected if r['method']==method]
        assert [int(r['image_id']) for r in rows]==ids and len(rows)==500
        actual=np.array([int(r['candidate_index']) for r in rows])
        assert np.all((actual>=0)&(actual<8))
        oracle=method.startswith('Oracle')
        assert all(r['uses_ground_truth']==str(oracle) and r['deployable']==str(not oracle) and r['diagnostic_only']==str(oracle) for r in rows)
        if oracle:
            metric='CLIP_cosine' if method=='Oracle-cosine' else 'CLIP_forward_two_way'
            score=raw[metric]
        else:
            pred=mean if method=='Mean-selected' else cp
            if method.startswith('Mismatched'):
                j=int(method[-1])-1
                perm=np.array(plan['mismatch_indices'][str(plan['mismatch_seeds'][j])])
                assert sorted(perm)==list(range(500)) and np.all(perm!=np.arange(500))
                pred=cp[perm]
            score=np.sum(norm(pred)[:,None,:]*candidates,axis=-1)
            error=float(np.max(np.abs(score-np.array([json.loads(r['ranking_scores']) for r in rows]))))
            max_score_error=max(error,max_score_error)
            assert error<1e-12
        expected=np.array([max(range(8),key=lambda k:(row[k],-k)) for row in score])
        assert np.array_equal(actual,expected)
        method_indices[method]=actual
    for method in ['Uniform']+list(method_indices):
        rows=[r for r in per if r['method']==method]
        assert [int(r['image_id']) for r in rows]==ids
        for metric in raw.files:
            if metric=='image_ids':continue
            values=raw[metric].mean(1) if method=='Uniform' else raw[metric][np.arange(500),method_indices[method]]
            assert np.allclose(values,[float(r[metric]) for r in rows],atol=1e-12,rtol=0)
    result=json.loads((out/'paired_comparisons.json').read_text())
    rng=np.random.default_rng(plan['bootstrap_seed'])
    draws=rng.integers(500,size=(plan['bootstrap_draws'],500))
    ci_error=0.0
    for method in ['Uniform','Mean-selected']:
        baseline=raw['CLIP_forward_two_way'].mean(1) if method=='Uniform' else raw['CLIP_forward_two_way'][np.arange(500),method_indices[method]]
        delta=raw['CLIP_forward_two_way'][np.arange(500),method_indices['C-selected']]-baseline
        ci=np.quantile([delta[d].mean() for d in draws],[.0125,.9875])
        error=float(np.max(abs(ci-result['C_minus'][method]['CLIP_forward_two_way']['ci97_5'])))
        ci_error=max(ci_error,error)
        assert error<1e-12
    review=read(out/'visual_review.csv')
    assert len(review)==32 and {int(r['image_id']) for r in review}==set(plan['visual_ids'])
    for r in review:
        i=int(r['image_id'])
        for method in ['C-selected','Mean-selected','Oracle-cosine','Oracle-identification']:
            assert int(r[method])==method_indices[method][ids.index(i)]
        assert (out/'annotated_gallery'/f'{i}.jpg').exists()
    assert len(list((out/'annotated_gallery').glob('*.jpg')))==32
    write(out/'completion_audit.json',{'passed':True,'methods_verified':10,'selector_score_max_error':max_score_error,
        'primary_bootstrap_max_error':ci_error,'candidate_count':4000,'visual_review_count':32,
        'target_free_function_signature':list(inspect.signature(select_candidates).parameters),
        'all_frozen_sources_sha_verified':True,'new_probe_fits':0,'new_neuroadapter_updates':0,'new_diffusion_images':0,
        'report_sha256':sha(out/'REPORT.md'),'source_sha256':sha(Path(__file__))})
    print('D alignment, all selector indices, oracle flags, score joins and primary intervals passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
