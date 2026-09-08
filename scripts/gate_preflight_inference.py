#!/usr/bin/env python3
"""Exercise pinned inference/evaluator code before formal checkpoints can exist."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import load_inference_snapshot_provenance
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.inference import generate_candidates, install_inference_state
from neuroadapter_research.integrity import verify_submodule_heads
from neuroadapter_research.modeling import build_adapter, load_frozen_backbone
from neuroadapter_research.protocol import (
    load_selection_plan, method_fingerprint, read_ordered_ids,
    validate_selection_plan_inputs, verify_protocol_repository,
)
from neuroadapter_research.reproducibility import verify_decode_tree


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_program(path: Path):
    """Execute the unchanged metric block, with a small engineering-only pair pool."""
    tree = ast.parse(path.read_text(), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    starts = [i for i, node in enumerate(main.body) if isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name) and node.target.id == 'by_metric']
    ends = [i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'seed_mean' for target in node.targets)]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError('pinned evaluator metric block is not uniquely identifiable')
    isolated = ast.Module(body=main.body[starts[0]:ends[0] + 1], type_ignores=[])
    return compile(ast.fix_missing_locations(isolated), str(path), 'exec')


def select_pairs(manifest: dict) -> list[tuple[dict, int]]:
    """Fix the engineering sample order independently of reconstruction quality."""
    records = manifest['records']
    if len(records) != 8 or len({record['image_id'] for record in records}) != 8:
        raise ValueError('preflight requires eight unique validation images')
    if any(len(record['files']) != 8 for record in records):
        raise ValueError('each preflight image requires eight candidates')
    return [(record, candidate) for candidate in range(8) for record in records][:20]


def binding(config, runtime: Path) -> dict:
    verify_protocol_repository(runtime, config.raw['protocol_commit'])
    plan = load_selection_plan(config.paths['selection_plan'], require_frozen=True)
    frozen = validate_selection_plan_inputs(plan, validation_ids_path=config.paths['validation_ids'], repository_root=runtime)
    verify_submodule_heads(runtime, config.paths['source_manifest'])
    return {
        'config_sha256': config.sha256,
        'method_fingerprint': method_fingerprint(config),
        'repository_commit': config.raw['protocol_commit'],
        'preflight_script_sha256': sha256_file(Path(__file__)),
        **frozen,
    }


def decode(args, config, common: dict) -> None:
    provenance = load_inference_snapshot_provenance(args.snapshot)
    metadata = provenance['metadata']
    for key in ('config_sha256', 'method_fingerprint'):
        if metadata.get(key) != common[key]:
            raise ValueError(f'gate checkpoint differs in {key}')
    if metadata.get('run_mode') != 'gate':
        raise ValueError('preflight accepts only engineering gate snapshots')
    dataset = Subject1TrainingDataset(config.paths['training_cache'], config.paths['stimuli'], config.paths['validation_ids'])
    if len(dataset) != 500:
        raise ValueError('preflight requires the frozen 500-image validation pool')
    expected_ids = read_ordered_ids(config.paths['validation_ids'])[:8]
    samples = [dataset[index] for index in range(8)]
    if [int(sample['nsd_image_id']) for sample in samples] != expected_ids:
        raise ValueError('preflight sample order differs from the frozen ID order')
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    backbone = load_frozen_backbone(config.paths['stable_diffusion'])
    bundle = build_adapter(backbone.unet, dataset.num_parcels, dataset.max_voxels)
    install_inference_state(bundle, args.snapshot)
    bundle.neuro_adapter.to(device=device, dtype=dtype)
    bundle.guidance_generator.to(device=device, dtype=torch.float32)
    backbone.text_encoder.to(device=device, dtype=dtype).eval()
    backbone.vae.to(device=device, dtype=dtype).eval()
    png = load_script(args.runtime / 'scripts/decode_validation.py', 'pinned_decode').save_png_atomic
    plan = load_selection_plan(config.paths['selection_plan'], require_frozen=True)
    order = list(reversed(samples)) if args.reverse else samples
    for repeat in range(args.repeats):
        destination = args.output / f'pass-{repeat}'
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        records = {}
        for sample in order:
            image_id = int(sample['nsd_image_id'])
            images = generate_candidates(
                bundle=bundle, backbone=backbone, brain=sample['brain'], image_id=image_id,
                candidate_count=8, protocol=plan.raw['protocol_namespace'], split='validation',
                device=device, dtype=dtype, denoising_steps=plan.raw['denoising_steps'],
                guidance_scale=plan.raw['guidance_scale'],
            )
            files = []
            for candidate, image in enumerate(images):
                relative = f'images/{image_id:05d}/candidate-{candidate:02d}.png'
                png(destination / relative, image)
                files.append({'candidate_index': candidate, 'path': relative, 'sha256': sha256_file(destination / relative)})
            records[image_id] = {'image_id': image_id, 'files': files}
            print(f'decode repeat={repeat} image_id={image_id} candidates=8', flush=True)
        write_json_atomic(destination / 'decode_manifest.json', {
            'schema_version': 1, 'status': 'complete', 'split': 'validation', 'run_mode': 'gate',
            'image_count': 8, 'candidate_count': 8, 'denoising_steps': plan.raw['denoising_steps'],
            'guidance_scale': plan.raw['guidance_scale'], **common,
            'snapshot_model_sha256': sha256_file(args.snapshot / 'model.pt'),
            'records': [records[image_id] for image_id in expected_ids],
        })
    dataset.close()


def evaluate(args, config, common: dict) -> None:
    manifest = verify_decode_tree(args.decode_manifest)
    if manifest.get('run_mode') != 'gate' or manifest.get('image_count') != 8 or manifest.get('candidate_count') != 8:
        raise ValueError('preflight evaluation requires the 8-image, 8-candidate gate decode')
    for name, expected in common.items():
        if manifest.get(name) != expected:
            raise ValueError(f'preflight decode has incorrect {name}')
    pairs = select_pairs(manifest)
    with h5py.File(config.paths['stimuli'], 'r') as handle:
        originals = np.stack([np.asarray(handle['imgBrick'][record['image_id']]) for record, _ in pairs])
    reconstructed = [np.stack([np.asarray(Image.open(args.decode_manifest.parent / record['files'][candidate]['path']).convert('RGB')) for record, candidate in pairs])]
    project_root = Path(config.raw['project_root'])
    # The frozen evaluator uses this vendor location; verify it as well as runtime.
    verify_submodule_heads(project_root / 'repo', config.paths['source_manifest'])
    metric_path = args.runtime / 'scripts/evaluate_validation.py'
    evaluator = load_script(metric_path, 'pinned_evaluator')
    assets = evaluator.verify_evaluation_assets(project_root, config.paths['evaluation_manifest'])
    os.environ['TORCH_HOME'] = str(project_root / 'models/evaluation/torch')
    torch.hub.set_dir(str(project_root / 'models/evaluation/torch/hub'))
    namespace = dict(vars(evaluator))
    plan = load_selection_plan(config.paths['selection_plan'], require_frozen=True)
    namespace.update(originals=originals, reconstructed=reconstructed, project_root=project_root,
                     assets=assets, device=torch.device('cuda:0'),
                     batch_size=int(plan.raw['evaluation_batch_size']))
    exec(metric_program(metric_path), namespace)
    means = namespace['seed_mean']
    if len(means) != 8 or any(not np.isfinite(values).all() for values in means.values()):
        raise ValueError('preflight metrics are incomplete or nonfinite')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [{'image_id': int(record['image_id']), 'candidate_index': candidate,
             **{name: float(values[index]) for name, values in means.items()}}
            for index, (record, candidate) in enumerate(pairs)]
    with (args.output / 'per_pair.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'schema_version': 1, 'status': 'complete', 'run_mode': 'gate', 'pair_count': 20,
        'purpose': 'engineering repeatability only; repeated GT IDs, not formal scores',
        **common, 'decode_manifest_sha256': sha256_file(args.decode_manifest),
        'metric_source_sha256': sha256_file(metric_path),
        'per_image_csv_sha256': sha256_file(args.output / 'per_pair.csv'),
        'metrics': {name: float(value.mean()) for name, value in means.items()},
    }
    write_json_atomic(args.output / 'evaluation.json', payload)
    print(json.dumps(payload, indent=2), flush=True)
    namespace.clear()
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--action', choices=('decode', 'evaluate'), required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--decode-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=(1, 2), default=1)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    if args.action == 'decode' and args.snapshot is None:
        parser.error('--snapshot is required for decode')
    if args.action == 'evaluate' and args.decode_manifest is None:
        parser.error('--decode-manifest is required for evaluate')
    config = load_training_config(args.config, require_frozen=True)
    configure_torch_backend(config.training)
    common = binding(config, args.runtime)
    (decode if args.action == 'decode' else evaluate)(args, config, common)


if __name__ == '__main__':
    main()
