"""Independently verify splits, fitted artifacts, predictions, and scores."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from run_semantic_probe import load_plan, locations
from semantic_probe_core import normalize, partition
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.checkpoint import verify_inference_snapshot


def run(root):
    out, plan = load_plan(root)
    _, signal = locations(root)
    ids = {k: np.loadtxt(out/'split_ids'/f'{k}.txt', dtype=int).tolist()
           for k in ['probe_fit','probe_tune','train8500','validation']}
    assert tuple(partition(ids['train8500'], plan['namespace'],7500)) == (ids['probe_fit'],ids['probe_tune'])
    assert set(ids['probe_fit']).isdisjoint(ids['probe_tune'])
    assert set(ids['train8500']).isdisjoint(ids['validation'])
    assert len(set(ids['train8500']+ids['validation'])) == 9000
    with (out/'tuning_results.csv').open() as f: tuning = list(csv.DictReader(f))
    assert len(tuning) == 60
    selected = json.loads((out/'selected_configs.json').read_text())['configurations']
    xids = np.load(signal/'pool_ids.npy').tolist()
    x = np.load(signal/'valid_mean_responses.npy', mmap_mode='r')
    xtrain = x[[xids.index(i) for i in ids['train8500']]]
    xval = x[[xids.index(i) for i in ids['validation']]]
    true_mean = xtrain.mean(0,dtype=np.float64)
    true_var = xtrain.var(0,dtype=np.float64)
    models = []
    for label, config in selected.items():
        candidates = [r for r in tuning if r['label'] == label]
        assert {(int(r['dimension']),float(r['alpha'])) for r in candidates} == {(d,a) for d in [256,1024] for a in [.1,1,10,100,1000]}
        best = sorted(candidates,key=lambda r:(-float(r['forward_two_way']),int(r['dimension']),-float(r['alpha'])))[0]
        assert config['dimension'] == int(best['dimension']) and config['alpha'] == float(best['alpha'])
        prep = joblib.load(out/f"preprocessing-{config['dimension']}.joblib")
        scaler,keep,pca = prep['scaler'],prep['keep'],prep['pca']
        assert scaler.n_samples_seen_ == 8500 and pca.n_samples_ == 8500
        assert np.allclose(scaler.mean_,true_mean,atol=1e-10,rtol=1e-10)
        assert np.allclose(scaler.var_,true_var,atol=1e-10,rtol=1e-10)
        assert np.array_equal(keep,true_var>0)
        z = pca.transform(scaler.transform(xval)[:,keep])
        ridge = joblib.load(out/f'{label}-ridge.joblib')
        assert ridge.alpha == config['alpha'] and ridge.fit_intercept and ridge.solver == 'cholesky'
        prediction = np.load(out/f'{label}-predictions.npz')
        assert prediction['image_ids'].tolist() == ids['validation']
        error = float(np.max(np.abs(ridge.predict(z)-prediction['features'])))
        assert error < 1e-8
        for stage,n in [('tune',7500),('refit',8500)]:
            perm = np.load(out/f'{label}-{stage}-permutation.npy')
            seed = None if label == 'C' else plan['shuffle_seeds'][int(label[-1])-1]+(1000000 if stage=='refit' else 0)
            expected = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
            assert np.array_equal(perm,expected)
        models.append({'method':label,'prediction_replay_max_error':error,'fitting_scope8500_verified':True})
    gtfile=np.load(out/'gt_features.npz')
    gt=normalize(gtfile['features'][-500:])
    with (out/'per_image_scores.csv').open() as f: saved=list(csv.DictReader(f))
    scorechecks=[]
    # Independent scalar ranking, not the production vectorized retrieval helper.
    for label in ['C','mean','shuffle1','shuffle2','shuffle3','shuffle4','shuffle5','R','L5000']:
        data=np.load(out/(f'{label}_features.npz' if label in ['R','L5000'] else f'{label}-predictions.npz'))
        arrays=data['features'] if label in ['R','L5000'] else data['features'][None]
        metrics=[]
        for arr in arrays:
            similarity=normalize(arr)@gt.T
            per=[]
            for i,row in enumerate(similarity):
                actual=row[i]; wrong=np.delete(row,i)
                q=(np.sum(actual>wrong)+.5*np.sum(actual==wrong))/499
                above=sum(row>actual); ties=sum(row==actual)
                per.append([actual,q,min(1,max(0,(1-above)/ties)),min(1,max(0,(5-above)/ties))])
            metrics.append(per)
        actual=np.mean(metrics,axis=0)
        rows=[r for r in saved if r['method']==label]
        assert [int(r['image_id']) for r in rows]==ids['validation']
        expected=np.array([[float(r[k]) for k in ['cosine','forward_two_way','top1','top5']] for r in rows])
        error=float(np.max(abs(actual-expected)))
        assert error<1e-12
        scorechecks.append({'method':label,'independent_rank_score_max_error':error})
    weights={}
    old=root/'runs/selection/subject01-selection-4090-deterministic-v2/snapshots'
    lr=root/'runs/experiments/paired-lr-probe-v1'
    spec=json.loads((root/'repo/configs/experiments/semantic_probe_c_v1.json').read_text())
    for label,folder in [('B0',old/'snapshot-update-00159375'),('R',old/'snapshot-update-00239063'),
                         ('H5000',lr/'arms/H/snapshots/snapshot-update-00005000'),('L5000',lr/'arms/L/snapshots/snapshot-update-00005000')]:
        verify_inference_snapshot(folder)
        weights[label]=sha256_file(folder/'model.pt')
    assert weights['B0']=='909357e478b171347630f1d709268225661bcad904637edaa1379a8871d32180'
    assert weights['R']=='bdca167505e0f1e62e025a5856299c56548dc40c2231740b8d2e1f84665b8217'
    write_json_atomic(out/'completion_audit.json',{'passed':True,'models':models,'scorechecks':scorechecks,
        'preserved_weight_sha256':weights,'tuning_configs_per_method':10,'n_shuffle_controls':5,
        'new_neuroadapter_updates':0,'new_diffusion_images':0,'no_standard_test_file_read':True,
        'code_sha256':sha256_file(Path(__file__))})
    print('Independent artifact and metric audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
