#!/usr/bin/env python3
"""Independent paired LR experiment, reusing the immutable training primitives."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import save_distributed_checkpoint, save_inference_snapshot, load_distributed_checkpoint
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.modeling import (load_frozen_backbone, build_adapter, load_trainable_state_dict,
    trainable_state_dict, tensor_state_sha256, audit_trainable_parameters, NeuroAdapterTrainingModule)
from neuroadapter_research.rng import TrainingGenerators, capture_process_rng_state, restore_process_rng_state
from neuroadapter_research.sampler import DeterministicDistributedBatchPlan
from neuroadapter_research.trainer import (initialize_distributed, set_process_seed, all_ranks_agree_on_hash,
    build_dataloader, token_keep_mask, min_snr_weights, tensor_sha256, append_json_line,
    synchronize_distributed_error, run_rank_zero_action, move_optimizer_state_to_device, TerminationFlag)

SPEC_PATH = Path(__file__).resolve().parents[1]/"configs/experiments/paired_lr_probe_v1.yaml"


def settings(root, arm):
    base = load_training_config(root/"configs/formal/subject01_selection_v2.yaml", require_frozen=True)
    spec = yaml.safe_load(SPEC_PATH.read_text())
    if spec["arms"] != {"H":1e-4,"L":1e-5} or arm not in spec["arms"]:
        raise ValueError("only the prespecified paired learning rates are allowed")
    if spec["max_local_updates"] != 5000 or spec["save_local_updates"] != [0,1000,2500,5000]:
        raise ValueError("the local budget and observation nodes are fixed")
    if spec["source_update"] != 159375 or spec["optimizer_reset"] is not True:
        raise ValueError("this is a fresh-optimizer experiment from snapshot 159375")
    for k in ("world_size","micro_batch_size","gradient_accumulation_steps","global_batch_size"):
        if spec[k] != base.training[k]:
            raise ValueError(f"batch geometry differs: {k}")
    training = copy.deepcopy(base.training)
    training.update(learning_rate=spec["arms"][arm],base_seed=spec["pair_seed"],
                    sampler_seed=spec["pair_seed"],max_updates=5000)
    source = base.paths["output_dir"]/"snapshots/snapshot-update-00159375/model.pt"
    if sha256_file(source) != spec["source_sha256"]:
        raise ValueError("source checkpoint hash mismatch")
    return base,spec,training,source


def run(root, arm, preflight=False, resume=None):
    base,spec,t,source=settings(root,arm)
    c=initialize_distributed(2)
    set_process_seed(t["base_seed"],c.rank)
    backend=configure_torch_backend(t)
    out=root/"runs/experiments/paired-lr-probe-v1"/("preflight" if preflight else "arms")/arm
    if resume is None and out.exists():
        raise FileExistsError(out)
    if resume is not None and (preflight or resume.resolve().parent != (out/"checkpoints").resolve()):
        raise ValueError("resume must belong to this exact experimental arm")
    effective={"experiment_type":"paired_lr_probe","arm":arm,"source_update":159375,
        "source_sha256":spec["source_sha256"],"optimizer_reset":True,"training":t,
        "base_config_sha256":base.sha256,"experiment_spec_sha256":sha256_file(SPEC_PATH),
        "script_sha256":sha256_file(Path(__file__)),"backend":backend,
        "split_sha256":sha256_file(base.paths["split_ids"]),"output":str(out),"preflight":preflight}
    config_hash=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
    dataset=Subject1TrainingDataset(base.paths["training_cache"],base.paths["stimuli"],base.paths["split_ids"])
    assert len(dataset)==8500
    plan=DeterministicDistributedBatchPlan(8500,16,2,4,2,spec["pair_seed"])
    backbone=load_frozen_backbone(base.paths["stable_diffusion"])
    bundle=build_adapter(backbone.unet,200,dataset.max_voxels)
    state=torch.load(source,map_location="cpu",weights_only=True)
    load_trainable_state_dict(bundle,state)
    audit=audit_trainable_parameters(bundle)
    initial=tensor_state_sha256(trainable_state_dict(bundle))
    all_ranks_agree_on_hash(initial,c)
    initial_groups={k:tensor_state_sha256({k:v}) for k,v in trainable_state_dict(bundle).items()}
    module=NeuroAdapterTrainingModule(bundle).to(c.device)
    backbone.vae.to(c.device,dtype=torch.bfloat16).eval()
    backbone.text_encoder.to(c.device,dtype=torch.bfloat16).eval()
    kwargs=dict(device_ids=[c.local_rank],output_device=c.local_rank,broadcast_buffers=False,find_unused_parameters=False)
    if "init_sync" in inspect.signature(DDP).parameters:
        kwargs["init_sync"]=False
    ddp=DDP(module,**kwargs)
    params=[p for p in ddp.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=t["learning_rate"],betas=(t["adam_beta1"],t["adam_beta2"]),
        eps=t["adam_epsilon"],weight_decay=t["weight_decay"],fused=t["adamw_fused"],foreach=t["adamw_foreach"])
    assert not optimizer.state and params
    generators=TrainingGenerators.create(t["base_seed"],c.rank,c.device)
    start=0
    payload=None
    if resume:
        payload=load_distributed_checkpoint(resume,c.rank,2)
        st=payload["trainer"]
        assert st["config_hash"]==config_hash and st["source_update"]==159375 and st["arm"]==arm
        plan.validate_state(st["sampler"])
        start=st["next_update"]
        assert 0<=start<5000
        load_trainable_state_dict(bundle,payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        move_optimizer_state_to_device(optimizer,c.device)
    if c.is_main:
        out.mkdir(parents=True,exist_ok=True)
        if not resume:
            write_json_atomic(out/"effective_config.json",effective)
            write_json_atomic(out/"initialization.json",{"model_tensor_sha256":initial,
                "groups":initial_groups,"optimizer_state_entries":0,"parameter_audit":audit})
    dist.barrier()
    ids=backbone.tokenizer("",max_length=backbone.tokenizer.model_max_length,padding="max_length",
        truncation=True,return_tensors="pt").input_ids.to(c.device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        text=backbone.text_encoder(ids)[0]
    alphas=backbone.noise_scheduler.alphas_cumprod.to(c.device)
    loader=build_dataloader(dataset,plan,start,1 if preflight else 5000,c,t["dataloader_workers"],t["base_seed"]+c.rank)
    iterator=iter(loader)
    if payload:
        generators.load_state_dict(payload["rank"]["training_generators"])
        restore_process_rng_state(payload["rank"]["process_rng"],c.device)
    termination=TerminationFlag()
    signal.signal(signal.SIGTERM,termination.handler)
    signal.signal(signal.SIGINT,termination.handler)

    def save(update):
        modelstate=trainable_state_dict(bundle) if c.is_main else {}
        meta={"experiment_type":"paired_lr_probe","arm":arm,"source_update":159375,
              "local_update":update,"optimizer_update":update,"optimizer_reset":True,
              "config_hash":config_hash,"source_sha256":spec["source_sha256"]}
        if update>0:
            run_rank_zero_action(c,"save local snapshot",lambda:save_inference_snapshot(out/"snapshots",update,modelstate,meta))
        save_distributed_checkpoint(out/"checkpoints",update,c.rank,2,modelstate,
            optimizer.state_dict() if c.is_main else {},
            {**meta,"next_update":update,"sampler":plan.state_before_update(update).to_dict()},
            {"rank":c.rank,"training_generators":generators.state_dict(),"process_rng":capture_process_rng_state(c.device)},
            dist.barrier,lambda e:synchronize_distributed_error(c,e))

    if not preflight and not resume:
        save(0)
    begin=time.time()
    stopped=False
    for update in range(start,1 if preflight else 5000):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        traces=[]
        for micro in range(2):
            batch=next(iterator)
            images=batch["image"].to(c.device,dtype=torch.bfloat16,non_blocking=True)
            brain=batch["brain"].to(c.device,dtype=torch.float32,non_blocking=True)
            vae_rng=tensor_sha256(generators["vae_latent"].get_state())
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                latents=backbone.vae.encode(images).latent_dist.sample(generator=generators["vae_latent"])
                latents=latents*backbone.vae.config.scaling_factor
            noise=torch.randn(latents.shape,device=c.device,dtype=latents.dtype,generator=generators["diffusion_noise"])
            ts=torch.randint(0,backbone.noise_scheduler.config.num_train_timesteps,(latents.shape[0],),
                             device=c.device,generator=generators["timestep"]).long()
            noisy=backbone.noise_scheduler.add_noise(latents,noise,ts)
            keep=token_keep_mask(latents.shape[0],200,c.device,generators["token_dropout"])
            with (ddp.no_sync() if micro==0 else contextlib.nullcontext()),torch.autocast("cuda",dtype=torch.bfloat16):
                pred=ddp(noisy,ts,text.expand(latents.shape[0],-1,-1),brain,keep)
                per=F.mse_loss(pred.float(),noise.float(),reduction="none").mean((1,2,3))
                loss=(per*min_snr_weights(ts,alphas,t["min_snr_gamma"])).mean()
                (loss/2).backward()
            assert torch.isfinite(loss)
            losses.append(loss.detach())
            traces.append({"micro":micro,"image_ids":batch["nsd_image_id"].tolist(),"vae_rng_before":vae_rng,
                "latent_sha256":tensor_sha256(latents),"noise_sha256":tensor_sha256(noise),
                "timestep_sha256":tensor_sha256(ts),"dropout_sha256":tensor_sha256(keep)})
        first=None
        if update==0:
            grad={name:p.grad for name,p in module.named_parameters() if p.requires_grad}
            assert all(v is not None and torch.isfinite(v).all() for v in grad.values())
            first={"initial_model_sha256":initial,"losses":[float(x) for x in losses],
                   "gradient_sha256":tensor_state_sha256({"gradient":grad}),"microbatches":traces,
                   "optimizer_entries_before":len(optimizer.state)}
        norm=torch.nn.utils.clip_grad_norm_(params,t["max_grad_norm"])
        assert torch.isfinite(norm)
        optimizer.step()
        if first is not None:
            after={k:tensor_state_sha256({k:v}) for k,v in trainable_state_dict(bundle).items()}
            first["updated_groups"]={k:after[k]!=initial_groups[k] for k in after}
            assert all(first["updated_groups"].values()), "a trainable component did not update"
            write_json_atomic(out/f"first_update_rank{c.rank}.json",first)
        append_json_line(out/f"random_trace_rank{c.rank}.jsonl",{"local_update":update+1,"microbatches":traces})
        ml=torch.stack(losses).mean()
        dist.all_reduce(ml)
        if c.is_main and ((update+1)%10==0 or update==0):
            append_json_line(out/"training.jsonl",{"source_update":159375,"local_update":update+1,
                "loss":float(ml/2),"gradient_norm_before_clip":float(norm),"lr":t["learning_rate"],
                "elapsed_seconds":time.time()-begin,"images_seen_local":(update+1)*16})
            print(f"{arm} local_update={update+1} loss={float(ml/2):.6f} elapsed={time.time()-begin:.1f}s",flush=True)
        stop=torch.tensor(int(termination.requested),device=c.device)
        dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        stopped=bool(stop.item())
        if not preflight and (update+1 in spec["save_local_updates"] or stopped):
            save(update+1)
        if stopped:
            break
    if c.is_main:
        write_json_atomic(out/"status.json",{"status":"interrupted" if stopped else "completed",
            "local_update":update+1,"seconds":time.time()-begin,"preflight_only":preflight,
            "source_update":159375,"config_hash":config_hash,"optimizer_reset":True})
    dist.barrier()
    dataset.close()
    dist.destroy_process_group()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--arm",choices=["H","L"],required=True)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--resume",type=Path)
    a=p.parse_args()
    run(a.root,a.arm,a.preflight,a.resume)
