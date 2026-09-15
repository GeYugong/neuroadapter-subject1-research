"""Inventory every existing R candidate and freeze D before scoring."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_rerank_core import derangement


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def run(root):
    out = root / 'runs/diagnostics/semantic-rerank-d-v1'
    out.mkdir(parents=True, exist_ok=False)
    repo = root / 'repo'
    specpath = repo / 'configs/experiments/semantic_rerank_d_v1.json'
    spec = json.loads(specpath.read_text())
    source = root / ('runs/selection/subject01-selection-4090-deterministic-v2/'
                     'evaluation-20260910/final/update-00239063/decode/decode_manifest.json')
    manifest = json.loads(source.read_text())
    croot = root / 'runs/diagnostics/semantic-probe-c-v1'
    validation = np.loadtxt(croot / 'split_ids/validation.txt', dtype=int).tolist()
    assert manifest['status'] == 'complete' and manifest['split'] == 'validation'
    assert manifest['candidate_count'] == 8 and manifest['optimizer_update'] == 239063
    assert manifest['snapshot_model_sha256'] == spec['generator_sha256']
    assert [r['image_id'] for r in manifest['records']] == validation
    records, failures = [], []
    for record in manifest['records']:
        assert [f['candidate_index'] for f in record['files']] == list(range(8))
        for f in record['files']:
            path = (source.parent / f['path']).resolve()
            assert path.is_relative_to(source.parent.resolve())
            try:
                assert sha(path) == f['sha256'], 'SHA mismatch'
                with Image.open(path) as image:
                    image.verify()
                records.append({'image_id': record['image_id'], **f})
            except (OSError, AssertionError) as error:
                failures.append({'image_id': record['image_id'], 'path': f['path'], 'error': str(error)})
    write(out / 'candidate_manifest.json', {'source': str(source.relative_to(root)),
          'source_sha256': sha(source), 'verified_count': len(records), 'failures': failures,
          'records': records, 'new_generation_allowed': False})
    if failures:
        raise RuntimeError('candidate audit failed; report missing/corrupt files without regenerating')
    assert len(records) == 4000
    fitted = json.loads((croot / 'fit_summary.json').read_text())
    for name in ['C-predictions.npz', 'mean-predictions.npz']:
        assert sha(croot / name) == fitted['prediction_sha256'][name]
        assert np.load(croot / name)['image_ids'].tolist() == validation
    diagnostic = root / 'runs/diagnostics/20260914-condition-path/protocol.json'
    pairs = json.loads(diagnostic.read_text())['pairs']['validation']
    visual_ids = [p['image_id'] for p in pairs]
    assert len(visual_ids) == 32 and len(set(visual_ids)) == 32 and set(visual_ids) <= set(validation)
    sources = [source, specpath, diagnostic, croot/'C-predictions.npz', croot/'mean-predictions.npz',
               croot/'C-ridge.joblib', croot/'preprocessing-1024.joblib', croot/'feature_manifest.json',
               croot/'gt_features.npz', repo/'scripts/semantic_rerank_core.py',
               repo/'scripts/prepare_semantic_rerank.py']
    plan = {**spec, 'code_commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'source_sha256': {str(p.relative_to(root)): sha(p) for p in sources},
            'image_ids': validation, 'visual_ids': visual_ids,
            'mismatch_indices': {str(s): derangement(500,s).tolist() for s in spec['mismatch_seeds']}}
    write(out / 'plan.json', plan)
    print('Verified 4000 existing PNGs, frozen C IDs and 32 visual IDs; plan frozen, no generation', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
