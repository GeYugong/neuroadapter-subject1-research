#!/usr/bin/env python3
"""Bounded independent PCA/Ridge probe; never loads a diffusion model."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.linear_model import Ridge

from semantic_probe_core import partition, normalize, retrieval, fit_preprocessing, transform, choose
from audit_signal_generalization_v2 import csv_rows
from neuroadapter_research.atomic import sha256_file, write_json_atomic


def locations(root):
    return root/'runs/diagnostics/semantic-probe-c-v1', root/'runs/diagnostics/generalization-diagnosis-v2/signal'


def load_plan(root):
    out, _ = locations(root)
    plan = json.loads((out/'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert sha256_file(root/path) == expected, path
    return out, plan


def prepare(root):
    import torch
    import evaluate_validation as ev
    from neuroadapter_research.config import load_training_config
    from neuroadapter_research.backend import configure_torch_backend
    out, signal = locations(root)
    out.mkdir(parents=True, exist_ok=False)
    (out/'split_ids').mkdir()
    specfile = root/'repo/configs/experiments/semantic_probe_c_v1.json'
    spec = json.loads(specfile.read_text())
    train = np.loadtxt(root/'data/derived/splits/selection_train_ids.txt', dtype=int).tolist()
    val = np.loadtxt(root/'data/derived/splits/validation_ids.txt', dtype=int).tolist()
    pool = np.load(signal/'pool_ids.npy').tolist()
    assert len(train) == 8500 and len(val) == 500 and len(set(train+val)) == 9000
    assert set(pool) == set(train+val) and len(pool) == 9000
    fit, tune = partition(train, spec['namespace'], 7500)
    for name, values in [('probe_fit', fit), ('probe_tune', tune), ('train8500', train), ('validation', val)]:
        np.savetxt(out/'split_ids'/f'{name}.txt', values, fmt='%d')
    sourcefiles = [specfile, signal/'pool_ids.npy', signal/'valid_mean_responses.npy', signal/'vertex_order.json',
                   root/'data/derived/splits/selection_train_ids.txt', root/'data/derived/splits/validation_ids.txt',
                   root/'repo/scripts/run_semantic_probe.py', root/'repo/scripts/semantic_probe_core.py',
                   root/'repo/scripts/evaluate_validation.py', root/'data/fingerprints/evaluation_downloads.json']
    plan = {**spec, 'created_unix': time.time(), 'source_commit': subprocess.check_output(
        ['git', '-C', str(root/'repo'), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in sourcefiles}}
    write_json_atomic(out/'plan.json', plan)
    means = np.load(signal/'valid_mean_responses.npy', mmap_mode='r')
    order = json.loads((signal/'vertex_order.json').read_text())
    assert [r['token'] for r in order] == list(range(200))
    assert means.shape == (9000, sum(len(r['vertices']) for r in order))
    checks = []
    with h5py.File(root/'data/derived/training/subject01_train_pool_top100.h5', 'r') as f:
        lookup = {int(i): j for j, i in enumerate(f['image_ids'][:])}
        for image_id in fit[:4]+tune[:4]+val[:4]:
            padded = f['brain'][lookup[image_id]]
            valid = np.concatenate([padded[k, :len(r['vertices'])] for k, r in enumerate(order)])
            assert np.array_equal(valid, means[pool.index(image_id)]), image_id
            checks.append({'image_id': image_id, 'cache_valid_vertices_exact': True})
    write_json_atomic(out/'input_alignment.json', {'shape': list(means.shape), 'sample_checks': checks,
        'parcels': 200, 'padding_removed': True, 'standard_test_accessed': False})
    base = load_training_config(root/'configs/formal/subject01_selection_v2.yaml', require_frozen=True)
    configure_torch_backend(base.training)
    selection = json.loads(base.paths['selection_plan'].read_text())
    batch = int(selection['evaluation_batch_size'])
    assets = ev.verify_evaluation_assets(root, base.paths['evaluation_manifest'])
    sys.path.insert(0, str(root/'repo/vendor/CLIP'))
    import clip
    model, _ = clip.load(str(assets['clip_vit_l_14']), device='cuda:0', jit=False)
    model.eval().requires_grad_(False)
    prep = ev.preprocess_transform(224, ev.CLIP_MEAN, ev.CLIP_STD)

    def features(images):
        return ev.extract_features(images, model.encode_image, prep, device=torch.device('cuda:0'), batch_size=batch)

    # Keep validation batches identical to the prior endpoint evaluator.
    rawparts = []
    with h5py.File(base.paths['stimuli'], 'r') as h5:
        for split, ids in [('train', train), ('validation', val)]:
            for start in range(0, len(ids), batch*32):
                selected = ids[start:start+batch*32]
                rawparts.append(features(np.stack([np.asarray(h5['imgBrick'][i]) for i in selected])))
                print(f'CLIP GT {split}: {min(start+batch*32,len(ids))}/{len(ids)}', flush=True)
    rawgt = np.concatenate(rawparts)
    np.savez(out/'gt_features.npz', image_ids=np.array(train+val), features=rawgt)
    old = root/'runs/experiments/paired-lr-probe-v1/evaluation'
    protocol = json.loads((old/'protocol.json').read_text())
    featurechecks = []
    for label in ['R', 'L5000']:
        folder = Path(protocol['references']['R']['decode_root']) if label == 'R' else old/label
        ids, gt, images, _ = ev.load_decode_set(folder/'decode_manifest.json', base.paths['stimuli'])
        assert ids == val
        actualgt = features(gt)
        assert np.array_equal(actualgt, rawgt[-500:]), 'GT feature batch alignment changed'
        candidates = [features(a) for a in images]
        np.savez(out/f'{label}_features.npz', image_ids=np.array(ids), features=np.stack(candidates))
        with (old/f'{label}-image-scores.csv').open() as f:
            previous = list(csv.DictReader(f))
        assert [int(r['image_id']) for r in previous] == val
        cosine = np.mean([np.sum(normalize(a)*normalize(actualgt), axis=1) for a in candidates], axis=0)
        error = float(np.max(np.abs(cosine-np.array([float(r['CLIP_cosine']) for r in previous]))))
        assert error < 1e-6, error
        featurechecks.append({'label': label, 'gt_features_exact': True, 'max_cosine_error': error,
                              'decode_manifest_sha256': sha256_file(folder/'decode_manifest.json')})
    write_json_atomic(out/'feature_manifest.json', {'files': {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
        for p in out.glob('*_features.npz')}, 'clip_weights_sha256': sha256_file(assets['clip_vit_l_14']),
        'preprocessing': 'evaluate_validation.preprocess_transform(224,CLIP_MEAN,CLIP_STD); raw imgBrick uint8',
        'batch_size': batch, 'endpoint_checks': featurechecks, 'new_diffusion_images': 0})


def fit(root):
    out, plan = load_plan(root)
    _, signal = locations(root)
    featuremeta = json.loads((out/'feature_manifest.json').read_text())
    assert sha256_file(out/'gt_features.npz') == featuremeta['files']['gt_features.npz']['sha256']
    f = np.load(out/'gt_features.npz')
    yids = f['image_ids'].tolist()
    xids = np.load(signal/'pool_ids.npy').tolist()
    xall = np.load(signal/'valid_mean_responses.npy', mmap_mode='r')
    yall = normalize(f['features'])
    ids = {name: np.loadtxt(out/'split_ids'/f'{name}.txt', dtype=int).tolist()
           for name in ['probe_fit', 'probe_tune', 'train8500', 'validation']}
    def data(name):
        ii = ids[name]
        return np.asarray(xall[[xids.index(i) for i in ii]], dtype=np.float32), yall[[yids.index(i) for i in ii]]
    xf, yf = data('probe_fit'); xt, yt = data('probe_tune')
    methods = [('C', None)] + [(f'shuffle{j+1}', seed) for j, seed in enumerate(plan['shuffle_seeds'])]
    tuning, preps = [], []
    started = time.time()
    for dim in plan['pca_dimensions']:
        print(f'PCA fit7500 dimension={dim}', flush=True)
        scaler, keep, pca, zf = fit_preprocessing(xf, dim, plan)
        zt = transform(xt, scaler, keep, pca)
        preps.append({'stage': 'tuning', 'dimension': dim, 'fit_count': len(xf), 'zero_variance_removed': int((~keep).sum()),
                      'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum())})
        for label, seed in methods:
            permutation = np.arange(len(yf)) if seed is None else np.random.default_rng(seed).permutation(len(yf))
            np.save(out/f'{label}-tune-permutation.npy', permutation)
            for alpha in plan['alphas']:
                ridge = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(zf, yf[permutation])
                metrics = retrieval(ridge.predict(zt), yt)
                tuning.append({'label': label, 'dimension': dim, 'alpha': alpha,
                               **{k: float(v.mean()) for k, v in metrics.items()}})
                print(tuning[-1], flush=True)
        del scaler, keep, pca, zf, zt
    csv_rows(out/'tuning_results.csv', tuning)
    selected = {label: choose([r for r in tuning if r['label'] == label]) for label, _ in methods}
    for row in selected.values():
        row['alpha_on_boundary'] = row['alpha'] in [min(plan['alphas']), max(plan['alphas'])]
        row['dimension_on_upper_boundary'] = row['dimension'] == max(plan['pca_dimensions'])
    write_json_atomic(out/'selected_configs.json', {'selected_using': 'probe_tune1000 only', 'configurations': selected})
    del xf, xt
    xtrain, ytrain = data('train8500')
    xval, _ = data('validation')
    # Refit all unsupervised stages on8500 only; sharing across permutations is algebraically identical.
    for dim in sorted(set(r['dimension'] for r in selected.values())):
        print(f'PCA refit8500 dimension={dim}', flush=True)
        scaler, keep, pca, ztrain = fit_preprocessing(xtrain, dim, plan)
        zval = transform(xval, scaler, keep, pca)
        joblib.dump({'scaler': scaler, 'keep': keep, 'pca': pca}, out/f'preprocessing-{dim}.joblib')
        preps.append({'stage': 'refit', 'dimension': dim, 'fit_count': len(xtrain), 'zero_variance_removed': int((~keep).sum()),
                      'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum())})
        for label, seed in methods:
            if selected[label]['dimension'] != dim:
                continue
            perm = np.arange(len(ytrain)) if seed is None else np.random.default_rng(seed+1000000).permutation(len(ytrain))
            np.save(out/f'{label}-refit-permutation.npy', perm)
            ridge = Ridge(alpha=selected[label]['alpha'], fit_intercept=True, solver='cholesky').fit(ztrain, ytrain[perm])
            joblib.dump(ridge, out/f'{label}-ridge.joblib')
            np.savez(out/f'{label}-predictions.npz', image_ids=ids['validation'], features=ridge.predict(zval))
        del scaler, keep, pca, ztrain, zval
    np.savez(out/'mean-predictions.npz', image_ids=ids['validation'], features=np.tile(ytrain.mean(0), (500,1)))
    mean_tune = retrieval(np.tile(yf.mean(0), (1000,1)), yt)
    write_json_atomic(out/'fit_summary.json', {'preprocessing': preps, 'seconds': time.time()-started,
        'mean_tune': {k: float(v.mean()) for k,v in mean_tune.items()}, 'neuroadapter_updates': 0,
        'prediction_sha256': {p.name: sha256_file(p) for p in out.glob('*-predictions.npz')},
        'standard_test_accessed': False, 'validation_used_for_selection': False})


def evaluate(root):
    out, plan = load_plan(root)
    manifest = json.loads((out/'feature_manifest.json').read_text())
    fitted = json.loads((out/'fit_summary.json').read_text())
    val = np.loadtxt(out/'split_ids/validation.txt', dtype=int).tolist()
    features = np.load(out/'gt_features.npz')
    assert features['image_ids'][-500:].tolist() == val
    gt = features['features'][-500:]
    scores, rows, candidates = {}, [], []
    for label in ['C', 'mean', 'shuffle1', 'shuffle2', 'shuffle3', 'shuffle4', 'shuffle5', 'R', 'L5000']:
        filename = f'{label}_features.npz' if label in ['R', 'L5000'] else f'{label}-predictions.npz'
        expected = manifest['files'][filename]['sha256'] if label in ['R', 'L5000'] else fitted['prediction_sha256'][filename]
        assert sha256_file(out/filename) == expected
        data = np.load(out/filename)
        assert data['image_ids'].tolist() == val
        arrays = data['features'] if label in ['R', 'L5000'] else data['features'][None]
        per = [retrieval(a, gt) for a in arrays]
        scores[label] = {k: np.mean([m[k] for m in per], axis=0) for k in per[0]}
        for c, metrics in enumerate(per):
            candidates += [{'method': label, 'image_id': i, 'candidate': c, **{k: float(v[j]) for k,v in metrics.items()}}
                           for j,i in enumerate(val)]
        rows += [{'method': label, 'image_id': i, **{k: float(v[j]) for k,v in scores[label].items()}} for j,i in enumerate(val)]
    csv_rows(out/'per_image_scores.csv', rows)
    csv_rows(out/'per_candidate_scores.csv', candidates)
    summary = {label: {k: float(v.mean()) for k,v in metrics.items()} for label,metrics in scores.items()}
    rng = np.random.default_rng(plan['bootstrap_seed'])
    indices = rng.integers(500, size=(plan['bootstrap_draws'],500))
    comparisons = {}
    for label in ['mean', 'shuffle1', 'shuffle2', 'shuffle3', 'shuffle4', 'shuffle5', 'R', 'L5000']:
        comparisons[label] = {}
        for key in scores['C']:
            delta = scores['C'][key]-scores[label][key]
            comparisons[label][key] = {'mean': float(delta.mean()), 'ci97_5': np.quantile(delta[indices].mean(1), [.0125,.9875]).tolist()}
    assert abs(summary['mean']['forward_two_way']-.5) < 1e-12
    assert abs(summary['mean']['top1']-.002) < 1e-12 and abs(summary['mean']['top5']-.01) < 1e-12
    write_json_atomic(out/'control_results.json', {'summary': summary, 'C_minus_comparator': comparisons,
        'five_shuffles_are_descriptive': True, 'permutation_p_value': None,
        'bootstrap_unit': 'image; fixed target candidate pool; average candidate scores first',
        'inference_scope': plan['scope'], 'additional_training_or_search': False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'fit', 'evaluate'], required=True)
    args = parser.parse_args()
    globals()[args.phase](args.root)
