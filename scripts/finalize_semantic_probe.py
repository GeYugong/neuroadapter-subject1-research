"""Verify archived inference evidence and install the experiment report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    out = root / 'runs/diagnostics/semantic-probe-c-v1'
    archive = root / 'repo/manifests/semantic-probe-c-v1'
    plan = json.loads((out / 'plan.json').read_text())
    features = json.loads((out / 'feature_manifest.json').read_text())
    for name, info in features['files'].items():
        assert digest(out / name) == info['sha256'], name
    index = json.loads((archive / 'INDEX.json').read_text(encoding='utf-8-sig'))
    for entry in index['files']:
        assert digest(archive / entry['path']) == entry['sha256']
        source = out.parent / entry['path'] if entry['path'].endswith('.log') else out / entry['path']
        assert digest(source) == entry['sha256'], source
    gt = np.load(out / 'gt_features.npz')
    train_ids = np.loadtxt(out / 'split_ids/train8500.txt', dtype=int)
    assert np.array_equal(gt['image_ids'][:8500], train_ids)
    y = gt['features'][:8500].astype(np.float64)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    mean = np.load(out / 'mean-predictions.npz')['features']
    mean_error = float(np.max(np.abs(mean - y.mean(0))))
    assert mean_error < 1e-12
    with (out / 'per_image_scores.csv').open() as f:
        rows = list(csv.DictReader(f))
    results = json.loads((out / 'control_results.json').read_text())
    scores = {label: {key: np.array([float(r[key]) for r in rows if r['method'] == label])
                     for key in ['cosine', 'forward_two_way', 'top1', 'top5']}
              for label in results['summary']}
    rng = np.random.default_rng(plan['bootstrap_seed'])
    indices = rng.integers(500, size=(plan['bootstrap_draws'], 500))
    max_error = 0.0
    for label, metrics in results['C_minus_comparator'].items():
        for key, reference in metrics.items():
            delta = scores['C'][key] - scores[label][key]
            draws = np.array([np.mean(delta[idx]) for idx in indices])
            ci = np.quantile(draws, [.0125, .9875])
            error = float(np.max(np.abs(ci - reference['ci97_5'])))
            max_error = max(max_error, error)
            assert error < 1e-12
            assert abs(delta.mean() - reference['mean']) < 1e-12
    report = (root / 'repo/docs/SEMANTIC_PROBE_C_V1.md').read_text(encoding='utf-8')
    (out / 'REPORT.md').write_text(report.replace('../manifests/', '../../../repo/manifests/'), encoding='utf-8')
    audit = {'passed': True, 'mean_feature_max_error': mean_error,
             'bootstrap_interval_max_error': max_error, 'bootstrap_comparisons': 32,
             'archive_files_verified': len(index['files']),
             'feature_files_verified': len(features['files']),
             'report_sha256': digest(out / 'REPORT.md'), 'code_sha256': digest(Path(__file__)),
             'additional_model_fitting': False, 'additional_diffusion_generation': False}
    (out / 'finalization_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
