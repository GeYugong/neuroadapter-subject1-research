"""Fresh canonical T1/T2 training using the original BF16/DDP primitives."""
import argparse
import contextlib
import hashlib
import inspect
import json
import signal
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP

from retrain_lr_core import U, NODES, CANONICAL, settings, output, code_identity, lr_at, lr_state, verify_cached_assets
from neuroadapter_research.atomic import write_json_atomic, write_bytes_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import save_distributed_checkpoint, save_inference_snapshot, load_distributed_checkpoint, prune_full_checkpoints
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.modeling import load_frozen_backbone, build_adapter, load_trainable_state_dict, trainable_state_dict, tensor_state_sha256, audit_trainable_parameters, NeuroAdapterTrainingModule
from neuroadapter_research.rng import TrainingGenerators, capture_process_rng_state, restore_process_rng_state
from neuroadapter_research.sampler import DeterministicDistributedBatchPlan
from neuroadapter_research.trainer import initialize_distributed, set_process_seed, all_ranks_agree_on_hash, build_dataloader, token_keep_mask, min_snr_weights, tensor_sha256, append_json_line, synchronize_distributed_error, run_rank_zero_action, move_optimizer_state_to_device, TerminationFlag


def run(root, arm, test_case=None, test_stop=None, resume=None):
    base, spec, t = settings(root, arm)
    assets = verify_cached_assets(root)
    identity = code_identity(root)
    if test_case:
        assert (test_case,arm) in [('T1-first','T1'),('T2-full','T2'),('T2-resume','T2')]
        limit = 1 if arm == 'T1' else 20
        assert test_stop in (None,10) and (test_stop is None or test_case=='T2-resume')
    else:
        assert test_stop is None
        limit = U
    out = output(root)/'tests'/test_case if test_case else output(root)/arm
    if resume is None and out.exists():
        raise FileExistsError(f'fresh output already exists: {out}')
    if resume is not None:
        assert resume.resolve().parent in [(out/'checkpoints').resolve(),(out/'milestones').resolve()]
    c = initialize_distributed(2)
    set_process_seed(t['base_seed'],c.rank)
    backend = configure_torch_backend(t)
    effective = {'experiment_type':'retrain_lr_v1','arm':arm,'training':t,'schedule':spec['arms'][arm],
        'source':identity,'canonical_sha256':CANONICAL,'base_config_sha256':base.sha256,
        'assets_sha256':assets,'backend':backend,'test_case':test_case,'output':str(out)}
    config_hash=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
    ds=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['split_ids'])
    assert len(ds)==8500 and ds.num_parcels==200 and ds.max_voxels==626
    plan=DeterministicDistributedBatchPlan(8500,16,2,4,2,t['sampler_seed'])
    backbone=load_frozen_backbone(base.paths['stable_diffusion'])
    bundle=build_adapter(backbone.unet,200,626)
    load_trainable_state_dict(bundle,torch.load(base.paths['canonical_initialization'],map_location='cpu',weights_only=True))
    audit=audit_trainable_parameters(bundle)
    assert audit['trainable_tensors']==38 and audit['trainable_parameters']==116068608
    initial=tensor_state_sha256(trainable_state_dict(bundle))
    initial_groups={k:tensor_state_sha256({k:v}) for k,v in trainable_state_dict(bundle).items()}
    all_ranks_agree_on_hash(initial,c)
    payload=None; start=0
    if resume:
        payload=load_distributed_checkpoint(resume,c.rank,2)
        st=payload['trainer']
        assert st['config_hash']==config_hash and st['arm']==arm and st['source']==identity
        start=st['next_update']; assert 0<=start<limit
        assert st['lr_state']==lr_state(arm,start)
        plan.validate_state(st['sampler'])
        load_trainable_state_dict(bundle,payload['model'])
    module=NeuroAdapterTrainingModule(bundle).to(c.device)
    backbone.vae.to(c.device,dtype=torch.bfloat16).eval()
    backbone.text_encoder.to(c.device,dtype=torch.bfloat16).eval()
    kwargs=dict(device_ids=[c.local_rank],output_device=c.local_rank,broadcast_buffers=False,find_unused_parameters=False)
    if 'init_sync' in inspect.signature(DDP).parameters: kwargs['init_sync']=False
    ddp=DDP(module,**kwargs)
    parameters=[p for p in ddp.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr_at(arm,0),betas=(t['adam_beta1'],t['adam_beta2']),
        eps=t['adam_epsilon'],weight_decay=t['weight_decay'],fused=False,foreach=False)
    assert not optimizer.state
    generators=TrainingGenerators.create(t['base_seed'],c.rank,c.device)
    if payload:
        optimizer.load_state_dict(payload['optimizer']); move_optimizer_state_to_device(optimizer,c.device)
        expected_lr=lr_state(arm,start)['used_lr'] if start else lr_at(arm,0)
        assert all(p['lr']==expected_lr for p in optimizer.param_groups)
        # Preserve logs from work after the last durable update, then replay that suffix.
        for name,key in [(f'random_trace_rank{c.rank}.jsonl','update')]+([('training.jsonl','completed_updates')] if c.is_main else []):
            path=out/name
            if path.exists():
                original=path.read_bytes();lines=original.decode().splitlines()
                keep=[]
                for line in lines:
                    try: item=json.loads(line)
                    except json.JSONDecodeError: break
                    if item[key]<=start: keep.append(line)
                if len(keep)!=len(lines):
                    history=out/'resume-history';history.mkdir(exist_ok=True)
                    write_bytes_atomic(history/f'{time.time_ns()}-{name}',original)
                    write_bytes_atomic(path,(''.join(l+'\n' for l in keep)).encode())
    if c.is_main:
        out.mkdir(parents=True,exist_ok=True)
        if not resume:
            write_json_atomic(out/'effective_config.json',effective)
            write_json_atomic(out/'initialization.json',{'canonical_tensor_sha256':initial,'groups':initial_groups,
                'optimizer_state_entries':0,'parameter_audit':audit})
        write_json_atomic(out/'status.json',{'status':'running','completed_updates':start,'config_hash':config_hash})
    dist.barrier()
    text_ids=backbone.tokenizer('',max_length=backbone.tokenizer.model_max_length,padding='max_length',truncation=True,return_tensors='pt').input_ids.to(c.device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): text=backbone.text_encoder(text_ids)[0]
    alphas=backbone.noise_scheduler.alphas_cumprod.to(c.device)
    iterator=iter(build_dataloader(ds,plan,start,limit,c,4,t['base_seed']+c.rank))
    if payload:
        generators.load_state_dict(payload['rank']['training_generators'])
        restore_process_rng_state(payload['rank']['process_rng'],c.device)
    def frozen_hash():
        return tensor_state_sha256({name:{k:p for k,p in model.named_parameters() if not p.requires_grad}
            for name,model in [('module',module),('vae',backbone.vae),('text',backbone.text_encoder)]})
    frozen_before=frozen_hash() if start==0 else None
    termination=TerminationFlag()
    signal.signal(signal.SIGTERM,termination.handler); signal.signal(signal.SIGINT,termination.handler)

    def save(completed, snapshot=False):
        state=trainable_state_dict(bundle) if c.is_main else {}
        meta={'experiment_type':'retrain_lr_v1','arm':arm,'optimizer_update':completed,
            'canonical_sha256':CANONICAL,'config_hash':config_hash,'source':identity,'training_images':8500,
            'lr_state':lr_state(arm,completed),'test_only':test_case is not None}
        if snapshot:
            run_rank_zero_action(c,'snapshot',lambda:save_inference_snapshot(out/'snapshots',completed,state,meta))
        folder=out/('milestones' if completed in NODES else 'checkpoints')
        save_distributed_checkpoint(folder,completed,c.rank,2,state,optimizer.state_dict() if c.is_main else {},
            {**meta,'next_update':completed,'sampler':plan.state_before_update(completed).to_dict()},
            {'rank':c.rank,'training_generators':generators.state_dict(),'process_rng':capture_process_rng_state(c.device)},
            dist.barrier,lambda e:synchronize_distributed_error(c,e))
        # Only this new arm's rolling checkpoints; milestones and old runs untouched.
        run_rank_zero_action(c,'prune own rolling checkpoints',lambda:prune_full_checkpoints(out/'checkpoints',keep_latest=2))

    if not resume: save(0)
    started=time.perf_counter(); window=started; losses_window=[]; norms_window=[]; stop=False
    for update in range(start,limit):
        lr=lr_at(arm,update)
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        losses=[]; traces=[]
        trace=update<20 or (update+1)%1000==0 or update+1==limit
        for micro in range(2):
            batch=next(iterator)
            images=batch['image'].to(c.device,dtype=torch.bfloat16,non_blocking=True)
            brain=batch['brain'].to(c.device,dtype=torch.float32,non_blocking=True)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                latents=backbone.vae.encode(images).latent_dist.sample(generator=generators['vae_latent'])
                latents=latents*backbone.vae.config.scaling_factor
            noise=torch.randn(latents.shape,device=c.device,dtype=latents.dtype,generator=generators['diffusion_noise'])
            ts=torch.randint(0,backbone.noise_scheduler.config.num_train_timesteps,(latents.shape[0],),device=c.device,generator=generators['timestep']).long()
            noisy=backbone.noise_scheduler.add_noise(latents,noise,ts)
            keep=token_keep_mask(latents.shape[0],200,c.device,generators['token_dropout'])
            with (ddp.no_sync() if micro==0 else contextlib.nullcontext()),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=ddp(noisy,ts,text.expand(latents.shape[0],-1,-1),brain,keep)
                per=F.mse_loss(pred.float(),noise.float(),reduction='none').mean((1,2,3))
                loss=(per*min_snr_weights(ts,alphas,t['min_snr_gamma'])).mean()
                (loss/2).backward()
            if not torch.isfinite(loss): raise FloatingPointError(f'loss at {update}')
            losses.append(loss.detach())
            if trace:
                traces.append({'micro':micro,'ids':batch['nsd_image_id'].tolist(),'latents':tensor_sha256(latents),
                    'noise':tensor_sha256(noise),'timesteps':tensor_sha256(ts),'dropout':tensor_sha256(keep)})
        first=None
        if update==0:
            gradients={k:p.grad for k,p in module.named_parameters() if p.requires_grad}
            assert all(v is not None and torch.isfinite(v).all() for v in gradients.values())
            first={'initial':initial,'losses':[float(l) for l in losses],
                'gradients':tensor_state_sha256({'gradient':gradients}),'trace':traces,'optimizer_entries_before':len(optimizer.state)}
        norm=torch.nn.utils.clip_grad_norm_(parameters,t['max_grad_norm'])
        if not torch.isfinite(norm): raise FloatingPointError(f'gradient at {update}')
        optimizer.step()
        if first is not None:
            after={k:tensor_state_sha256({k:v}) for k,v in trainable_state_dict(bundle).items()}
            first['updated_groups']={k:v!=initial_groups[k] for k,v in after.items()}
            first['frozen_unchanged']=frozen_hash()==frozen_before
            assert all(first['updated_groups'].values()) and first['frozen_unchanged']
            write_json_atomic(out/f'first_update_rank{c.rank}.json',first)
        completed=update+1
        if trace: append_json_line(out/f'random_trace_rank{c.rank}.jsonl',{'update':completed,'microbatches':traces})
        mean=torch.stack(losses).mean(); dist.all_reduce(mean); mean/=2
        losses_window.append(float(mean)); norms_window.append(float(norm))
        if completed%100==0 or completed==1 or completed==limit or test_case:
            now=time.perf_counter()
            memory=[None,None]; dist.all_gather_object(memory,torch.cuda.max_memory_reserved(c.device))
            if c.is_main:
                record={'completed_updates':completed,'images_seen':completed*16,'reference_epoch':completed*16/8500,
                    'lr':lr,'mean_loss':sum(losses_window)/len(losses_window),'gradient_norm_before_clip':sum(norms_window)/len(norms_window),
                    'window_updates':len(losses_window),'updates_per_second':len(losses_window)/(now-window),
                    'peak_reserved_bytes_by_rank':memory,'elapsed_seconds':now-started}
                append_json_line(out/'training.jsonl',record)
                write_json_atomic(out/'status.json',{'status':'running',**record,'config_hash':config_hash})
                print(json.dumps({'arm':arm,**record}),flush=True)
            losses_window=[]; norms_window=[]; window=time.perf_counter()
        signal_stop=torch.tensor(int(termination.requested),device=c.device)
        dist.all_reduce(signal_stop,op=dist.ReduceOp.MAX)
        stop=bool(signal_stop.item()) or completed==test_stop
        if completed%5000==0 or completed in NODES or completed==limit or stop:
            save(completed,snapshot=completed in NODES or (test_case and completed==limit))
        if stop: break
    if c.is_main:
        write_json_atomic(out/'status.json',{'status':'completed' if completed==limit else 'interrupted',
            'completed_updates':completed,'config_hash':config_hash,'seconds':time.perf_counter()-started,'test_only':test_case is not None})
    ds.close(); dist.barrier(); dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['T1','T2'],required=True)
    p.add_argument('--test-case',choices=['T1-first','T2-full','T2-resume']);p.add_argument('--test-stop',type=int)
    p.add_argument('--resume',type=Path);a=p.parse_args()
    run(a.root,a.arm,a.test_case,a.test_stop,a.resume)
