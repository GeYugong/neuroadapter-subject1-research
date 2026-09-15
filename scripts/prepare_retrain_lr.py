"""One-time asset verification and bounded preflight comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from retrain_lr_core import settings, output, CANONICAL, NODES, code_identity
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.integrity import verify_file_against_manifest, verify_tree_against_manifest, verify_submodule_heads, validate_subject1_audits
from neuroadapter_research.trainer import validate_canonical_initialization
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.checkpoint import load_distributed_checkpoint


def assets(root):
    base,_,_=settings(root,'T1')
    out=output(root);out.mkdir(parents=True,exist_ok=True)
    if (out/'assets.json').exists():
        from retrain_lr_core import verify_cached_assets
        verify_cached_assets(root); return
    historical=json.loads((root/'repo/manifests/deterministic-4090-v2/effective_run.json').read_text())
    preferred=json.loads((root/'repo/manifests/deterministic-4090-v2/preferred_config.json').read_text())
    assert base.sha256==historical['config_sha256'] and base.training==preferred['training']
    canonical=validate_canonical_initialization(base,require_frozen=True)
    assert canonical['initialization_sha256']==CANONICAL
    validate_subject1_audits(base.paths)
    raw=verify_file_against_manifest(base.paths['stimuli'],base.paths['raw_nsd_manifest'],'stimuli/nsd_stimuli.hdf5')
    sd=verify_tree_against_manifest(base.paths['stable_diffusion'],base.paths['model_manifest'],
        manifest_prefix='stable-diffusion-v1-5',ignored_extra_prefixes=('.cache/huggingface',))
    primitives=root/'runtime/subject01-4090-1a1fcfa'
    verify_submodule_heads(primitives,base.paths['source_manifest'])
    assert raw['sha256']==historical['input_hashes']['stimuli']
    assert sd['tree_sha256']==historical['input_hashes']['stable_diffusion_tree']
    keys=[k for k in historical['input_hashes'] if k in base.paths]
    hashes={k:sha256_file(base.paths[k]) for k in keys}
    assert all(hashes[k]==historical['input_hashes'][k] for k in hashes)
    ds=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['split_ids'])
    vs=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['validation_ids'])
    assert len(ds)==8500 and len(vs)==500 and not set(ds.image_ids)&set(vs.image_ids)
    assert ds.max_voxels==626 and ds.num_parcels==200
    ids=[int(i) for i in vs.image_ids]; ds.close();vs.close()
    watched={base.path,*[base.paths[k] for k in keys]}
    watched.update(p for p in base.paths['stable_diffusion'].rglob('*') if p.is_file() and '.cache' not in p.parts)
    watched.update(p for p in (primitives/'src').rglob('*.py'))
    watched.update(p for p in (primitives/'vendor').rglob('*.py'))
    write_json_atomic(out/'assets.json',{'canonical_sha256':CANONICAL,'input_hashes':hashes,
        'stats':{str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in watched},
        'train_count':8500,'validation_ids':ids,'visual_ids':[ids[j] for j in np.linspace(0,499,32,dtype=int)],
        'canonical':canonical,'stable_diffusion_tree':sd['tree_sha256'],'original_effective_sha256':sha256_file(root/'repo/manifests/deterministic-4090-v2/effective_run.json')})
    print('Assets verified against original fingerprints; canonical and 8500/500 IDs unchanged',flush=True)


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def audit(root):
    out=output(root); a=out/'tests/T1-first'; b=out/'tests/T2-full'; r=out/'tests/T2-resume'
    science=[]
    for folder in (a,b):
        config=json.loads((folder/'effective_config.json').read_text())
        for k in ('arm','schedule','test_case','output'):config.pop(k)
        config['training'].pop('learning_rate');science.append(config)
    assert science[0]==science[1],'T1/T2 have non-LR scientific differences'
    assert json.loads((a/'initialization.json').read_text())==json.loads((b/'initialization.json').read_text())
    reports=[]
    for rank in range(2):
        first=[json.loads((p/f'first_update_rank{rank}.json').read_text()) for p in (a,b)]
        assert first[0]==first[1] and first[0]['frozen_unchanged'] and all(first[0]['updated_groups'].values())
        lhs=load_distributed_checkpoint(b/'checkpoints/checkpoint-update-00000020',rank,2)
        rhs=load_distributed_checkpoint(r/'checkpoints/checkpoint-update-00000020',rank,2)
        for key in ('model','optimizer','rank'): assert equal(lhs[key],rhs[key]),f'{rank}/{key}'
        for key in ('sampler','lr_state','next_update'): assert equal(lhs['trainer'][key],rhs['trainer'][key])
        assert (b/f'random_trace_rank{rank}.jsonl').read_bytes()==(r/f'random_trace_rank{rank}.jsonl').read_bytes()
        reports.append({'rank':rank,'first_update_equal':True,'frozen_unchanged':True,
            'resume_model_optimizer_rng_sampler_lr_exact':True,'random_trace_sha256':sha256_file(b/f'random_trace_rank{rank}.jsonl')})
    write_json_atomic(out/'preflight.json',{'passed':True,'source':code_identity(root),'ranks':reports,
        'test_updates':{'T1':1,'T2_continuous':20,'T2_split':20},'real_schedule_U':159375,
        'only_lr_scientific_difference':True})
    print('First gradient/loss equality and 20 vs10+10 exact resume PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['assets','audit'],required=True)
    a=p.parse_args();(assets if a.phase=='assets' else audit)(a.root)
