#!/usr/bin/env python3
"""Score existing diagnostic PNGs and assemble complete trajectory galleries."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from audit_signal_generalization_v2 import csv_rows
from evaluate_validation import CLIP_MEAN, CLIP_STD, extract_features, preprocess_transform
from summarize_condition_diagnosis import gallery, read_rows
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.metrics import pixel_metrics


@torch.no_grad()
def run(root):
    out = root/'runs/experiments/paired-lr-probe-v1'
    old = root/'runs/diagnostics/20260914-condition-path'
    ab = root/'runs/diagnostics/generalization-diagnosis-v2'
    target = out/'trajectory'
    target.mkdir(exist_ok=False)
    pairs = json.loads((old/'protocol.json').read_text())['pairs']
    asset = json.loads((root/'data/fingerprints/evaluation_downloads.json').read_text())['files']['clip_vit_l_14']
    assert sha256_file(root/asset['path']) == asset['sha256']
    sys.path.insert(0, str(root/'repo/vendor/CLIP'))
    import clip
    model, _ = clip.load(str(root/asset['path']), device='cuda:0', jit=False)
    transform = preprocess_transform(224, CLIP_MEAN, CLIP_STD)

    def features(arrays):
        values = extract_features(arrays, model.encode_image, transform, device=torch.device('cuda:0'), batch_size=8)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    labels = ['B0', 'H1000', 'L1000', 'H2500', 'L2500', 'H5000', 'L5000', 'R']
    raw, averaged, tables, pages = [], [], [], []
    for split, items in pairs.items():
        ids = [p['image_id'] for p in items]
        index = {i: j for j, i in enumerate(ids)}
        gt = np.stack([np.array(Image.open(old/split/'gt'/f'{i}.png')) for i in ids])
        gf = features(gt)
        lookup = {}
        for label in labels:
            folder = ab/split/'159375' if label == 'B0' else old/split/'bf16' if label == 'R' else out/'diagnostic'/label/split
            records = read_rows(folder/'images.jsonl')
            conditions = ['correct', 'wrong'] if label in ['B0', 'R', 'H5000', 'L5000'] else ['correct']
            expected = {(p['image_id'], p['donor_id'], c, j) for p in items for c in conditions for j in range(2)}
            assert len(records) == len(expected)
            assert {(r['image_id'], r['donor_id'], r['condition'], r['candidate']) for r in records} == expected
            for r in records:
                assert sha256_file(folder/r['path']) == r['sha256']
                assert sha256_file(old/split/'bf16/noise'/f"{r['image_id']}.pt") == r['noise_sha256']
                lookup[label, r['condition'], r['image_id'], r['candidate']] = folder/r['path']
            images = np.stack([np.array(Image.open(folder/r['path'])) for r in records])
            ff = features(images)
            pc, ss = pixel_metrics(gt[[index[r['image_id']] for r in records]], images)
            scores = {}
            for j, r in enumerate(records):
                row = {'label': label, 'split': split, **r, 'clip': float(ff[j] @ gf[index[r['image_id']]]),
                       'pixcorr': float(pc[j]), 'ssim': float(ss[j])}
                raw.append(row)
                scores[r['image_id'], r['condition'], r['candidate']] = row
            local = []
            for p in items:
                values = {f'{c}_{m}': float(np.mean([scores[p['image_id'], c, j][m] for j in range(2)]))
                          for c in conditions for m in ('clip', 'pixcorr', 'ssim')}
                if 'wrong' in conditions:
                    values['advantage_clip'] = values['correct_clip'] - values['wrong_clip']
                local.append({'label': label, 'split': split, **p, **values})
            averaged += local
            tables.append({'label': label, 'split': split, **{k: float(np.mean([r[k] for r in local])) for k in values}})
        for candidate in range(2):
            columns = [('GT', lambda p: old/split/'gt'/f"{p['image_id']}.png")]
            columns += [(label, lambda p, a=label: lookup[a, 'correct', p['image_id'], candidate]) for label in labels]
            pages += gallery(target, split, f'trajectory-c{candidate}', items, columns)
            columns = [('GT', lambda p: old/split/'gt'/f"{p['image_id']}.png"),
                       ('Donor GT', lambda p: old/split/'gt'/f"{p['donor_id']}.png")]
            for label in ['B0', 'H5000', 'L5000', 'R']:
                columns += [(f'{label} {cond}', lambda p, a=label, c=cond: lookup[a, c, p['image_id'], candidate])
                            for cond in ['correct', 'wrong']]
            pages += gallery(target, split, f'endpoint-c{candidate}', items, columns)
    # Union schema keeps absent intermediate wrong-input scores empty, not zero.
    def union_csv(path, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        csv_rows(path, [{k: r.get(k, '') for k in keys} for r in rows])
    union_csv(target/'per_candidate_scores.csv', raw)
    union_csv(target/'per_image_scores.csv', averaged)
    union_csv(target/'trajectory.csv', tables)
    write_json_atomic(target/'summary.json', {'table': tables, 'galleries': pages,
        'new_diffusion_images': 0, 'neuroadapter_updates': 0, 'visual_review': 'pending',
        'intermediate_checkpoint_selection': False, 'candidate_unit': 'average scores within image'})
    print(json.dumps(tables, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
