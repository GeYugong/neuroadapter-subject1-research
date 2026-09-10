"""Audit a completed selection and export reports without any GPU execution."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import (
    load_selection_plan, method_fingerprint, validate_selection_snapshot,
    verify_protocol_repository,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    runtime = root / 'runtime/subject01-4090-1a1fcfa'
    config_path = root / 'configs/formal/subject01_selection_v2.yaml'
    config = load_training_config(config_path, require_frozen=True)
    verify_protocol_repository(runtime, config.raw['protocol_commit'])
    plan = load_selection_plan(config.paths['selection_plan'], require_frozen=True)
    fingerprint = method_fingerprint(config)
    run = config.paths['output_dir']
    evaluation = run / 'evaluation-20260910'
    assert (root / 'artifacts/selection-existing-20260910.exit').read_text().strip() == '0'
    assert json.loads((evaluation / 'status.json').read_text())['status'] == 'completed'
    lock = json.loads((evaluation / 'RESEARCH_WEIGHT_LOCK.json').read_text())
    assert lock['no_more_training'] and not lock['retrained_on_9000'] and not lock['standard_test_used']
    assert lock['training_image_count'] == 8500 and lock['validation_image_count'] == 500
    assert sha256_file(evaluation / 'final_selection.json') == lock['selection_manifest_sha256']
    out = root / 'artifacts/final-selection-20260910'
    out.mkdir(exist_ok=False)
    public = out / 'public'
    public.mkdir()
    metrics = []
    audits = []
    inputs = {}
    for stage, count, expected in [('screening', 2, 20), ('final', 8, 5)]:
        paths = sorted((evaluation / stage).glob('*/evaluation.json'))
        assert len(paths) == expected
        inputs[stage] = paths
        for path in paths:
            work = path.parent
            payload = json.loads(path.read_text())
            record = payload['checkpoints'][0]
            update = record['optimizer_update']
            assert payload['status'] == 'complete' and payload['image_count'] == 500
            assert payload['candidate_count'] == count
            snapshot = run / 'snapshots' / f'snapshot-update-{update:08d}'
            provenance = validate_selection_snapshot(snapshot, config=config, plan=plan, fingerprint=fingerprint)
            assert provenance['model_sha256'] == record['snapshot_model_sha256']
            decode = json.loads((work / 'decode/decode_manifest.json').read_text())
            assert len(decode['records']) == 500 and decode['candidate_count'] == count
            assert decode['snapshot_model_sha256'] == provenance['model_sha256']
            assert len(list((work / 'decode').glob('candidate-*/*.png'))) == 500 * count
            for name in ['decode_validation.py', 'evaluate_validation.py']:
                status = json.loads((work / f'{name}.status.json').read_text())
                assert status['status'] == 'completed' and status['exit_code'] == 0
            metrics.append({'stage': stage, 'update': update, 'validation_loss': record['validation_loss'], **record['metrics']})
            audits.append({'stage': stage, 'update': update, 'model_sha256': provenance['model_sha256'], 'evaluation_sha256': sha256_file(path), 'decode_manifest_sha256': sha256_file(work / 'decode/decode_manifest.json'), 'png_count': 500 * count})
            shutil.copy2(path, public / f'{stage}-{update:08d}-evaluation.json')
    for stage, input_stage in [('shortlist', 'screening'), ('final', 'final')]:
        target = out / f'recomputed-{stage}.json'
        argv = [sys.executable, str(runtime / 'scripts/select_checkpoint.py'), '--config', str(config_path), '--stage', stage, '--input', *map(str, inputs[input_stage]), '--output', str(target)]
        if stage == 'final':
            argv.extend(['--shortlist-manifest', str(evaluation / 'shortlist.json')])
        subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
        original = evaluation / ('shortlist.json' if stage == 'shortlist' else 'final_selection.json')
        assert json.loads(target.read_text()) == json.loads(original.read_text())
    chosen = lock['selected_update']
    assert sha256_file(evaluation / 'selected_snapshot/model.pt') == lock['model_sha256']
    backup = json.loads((root / 'artifacts/hf-backup-20260910.json').read_text())
    assert backup['status'] == 'verified' and backup['public']
    inventory = json.loads((root / 'artifacts/hf-public-selection-20260910/BACKUP_MANIFEST.json').read_text())
    hf_path = f'snapshots/snapshot-update-{chosen:08d}/model.pt'
    assert next(x for x in inventory['files'] if x['path'] == hf_path)['sha256'] == lock['model_sha256']
    for name in ['RESEARCH_WEIGHT_LOCK.json', 'final_selection.json', 'shortlist.json']:
        shutil.copy2(evaluation / name, public / name)
    write_json_atomic(public / 'metrics_summary.json', metrics)
    write_json_atomic(public / 'completion_audit.json', {'status': 'verified', 'scope': 'snapshot hashes, evaluation bindings, counts, successful steps, exact CPU statistical replay, selected hardlink hash and verified HF inventory; full PNG payload hashes not replayed', 'no_gpu_used': True, 'no_more_training': True, 'recomputed_selection_identical': True, 'selected_update': chosen, 'selected_model_sha256': lock['model_sha256'], 'snapshot_evaluations': audits, 'hf_backup_revision': backup['revision']})
    records = json.loads((evaluation / f'final/update-{chosen:08d}/decode/decode_manifest.json').read_text())['records']
    indices = np.linspace(0, 499, 12, dtype=int).tolist()
    columns = [(39844, 0), (199219, 0), (212500, 0), (chosen, 0), (265625, 0), (chosen, 1)]
    with h5py.File(config.paths['stimuli'], 'r') as stimuli:
        for page in range(2):
            canvas = Image.new('RGB', (1260, 1224), 'white')
            draw = ImageDraw.Draw(canvas)
            for row, index in enumerate(indices[page * 6:page * 6 + 6]):
                image_id = records[index]['image_id']
                canvas.paste(Image.fromarray(np.asarray(stimuli['imgBrick'][image_id])).resize((180, 180)), (0, row * 204 + 24))
                draw.text((3, row * 204 + 5), f'GT id={image_id}', fill='black')
                for col, (update, seed) in enumerate(columns, 1):
                    path = evaluation / f'final/update-{update:08d}/decode/candidate-{seed:02d}/{image_id:05d}.png'
                    with Image.open(path) as image:
                        canvas.paste(image.resize((180, 180)), (col * 180, row * 204 + 24))
                    label = f'{update} / cand{seed}' + (' SELECTED' if update == chosen else '')
                    draw.text((col * 180 + 2, row * 204 + 5), label, fill='black')
            canvas.save(out / f'comparison-{page + 1}.jpg', quality=92)
    write_json_atomic(public / 'visual_sampling.json', {'rule': '12 evenly spaced positions in frozen validation order; no outcome-based sampling or candidate selection', 'indices': indices, 'image_ids': [records[i]['image_id'] for i in indices], 'columns': columns})
    print(json.dumps({'status': 'audited', 'selected_update': chosen, 'output': str(out), 'final_metrics': [x for x in metrics if x['stage'] == 'final']}, indent=2))


if __name__ == '__main__':
    main()
