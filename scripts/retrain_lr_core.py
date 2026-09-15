"""Bounded LR experiment configuration, identities and restart helpers."""
import copy
import json
import math
import subprocess
from pathlib import Path

import yaml

U = 159375
NODES = (53125, 106250, U)
CANONICAL = 'dc363931727f5f5e445d267f9b31e1a366b134b2e62a34dc72ae12693d875fca'
REPO = Path(__file__).resolve().parents[1]
SPEC = REPO/'configs/experiments/retrain_lr_v1.yaml'


def lr_at(arm, completed):
    if arm not in ('T1', 'T2') or not isinstance(completed, int) or not 0 <= completed < U:
        raise ValueError('invalid arm or completed_updates')
    return 3e-5 if arm == 'T1' else 1e-5 + .5*(1e-4-1e-5)*(1+math.cos(math.pi*completed/(U-1)))


def lr_state(arm, completed):
    if not 0 <= completed <= U:
        raise ValueError('invalid LR cursor')
    return {'kind': 'constant' if arm == 'T1' else 'cosine', 'U': U,
            'start': lr_at(arm, 0), 'end': lr_at(arm, U-1), 'completed_updates': completed,
            'used_lr': lr_at(arm, completed-1) if completed else None,
            'next_lr': lr_at(arm, completed) if completed < U else None}


def output(root):
    return root/'runs/experiments/retrain-lr-v1'


def settings(root, arm):
    from neuroadapter_research.config import load_training_config
    base = load_training_config(root/'configs/formal/subject01_selection_v2.yaml', require_frozen=True)
    spec = yaml.safe_load(SPEC.read_text())
    assert spec == {'experiment_type':'retrain_lr_v1','max_updates':U,'snapshot_updates':list(NODES),
        'canonical_sha256':CANONICAL,'arms':{'T1':{'kind':'constant','start':3e-5,'end':3e-5},
        'T2':{'kind':'cosine','start':1e-4,'end':1e-5}},'bootstrap_seed':20260901,'bootstrap_draws':10000}
    expected = dict(world_size=2, micro_batch_size=4, gradient_accumulation_steps=2, global_batch_size=16,
        base_seed=20260901, sampler_seed=20260901, dataloader_workers=4, precision='bf16',
        allow_tf32=True, cudnn_benchmark=False, deterministic_algorithms=True,
        adamw_fused=False, adamw_foreach=False)
    assert all(base.training[k] == v for k, v in expected.items())
    training = copy.deepcopy(base.training)
    training.update(max_updates=U, learning_rate=lr_at(arm, 0), log_every_updates=100)
    return base, spec, training


def code_identity(root):
    from neuroadapter_research.atomic import sha256_file
    import neuroadapter_research.trainer as trainer
    primitives = root/'runtime/subject01-4090-1a1fcfa/src/neuroadapter_research'
    assert Path(trainer.__file__).resolve().parent == primitives.resolve(), trainer.__file__
    return {'commit': subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            'checkout': str(REPO), 'primitives': str(primitives),
            'files': {str(p.relative_to(REPO)):sha256_file(p) for p in
                      [SPEC,*sorted((REPO/'scripts').glob('*.py'))]}}


def latest_checkpoint(folder):
    paths = sorted(folder.glob('checkpoint-update-*'))
    return next((p for p in reversed(paths) if (p/'COMPLETE').exists()), None)


def verify_cached_assets(root):
    from neuroadapter_research.atomic import sha256_file
    record = json.loads((output(root)/'assets.json').read_text())
    for name, stat in record['stats'].items():
        s = (root/name).stat()
        if [s.st_size,s.st_mtime_ns] != stat:
            raise ValueError(f'asset changed; full revalidation required: {name}')
    assert sha256_file(root/'models/canonical/subject01_adapter_init.pt') == CANONICAL
    return sha256_file(output(root)/'assets.json')
