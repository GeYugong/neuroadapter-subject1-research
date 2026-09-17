"""One-shot, train-only gradient-ratio calibration for lambda_sem."""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from generator_semantic_core import base_config, output, semantic_loss, source_identity, source_snapshot, spec
from train_generator_semantic import CombinedTrainingModule, load_semantic_assets
from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.modeling import build_adapter, load_frozen_backbone, load_trainable_state_dict
from neuroadapter_research.rng import TrainingGenerators, namespace_seed
from neuroadapter_research.sampler import DeterministicDistributedBatchPlan
from neuroadapter_research.trainer import (build_dataloader, initialize_distributed, min_snr_weights,
    set_process_seed, token_keep_mask)


def grad_norm(parameters) -> float:
    values = [parameter.grad.detach().float().norm(2).square() for parameter in parameters
              if parameter.grad is not None]
    result = torch.stack(values).sum().sqrt()
    if not torch.isfinite(result) or result <= 0: raise FloatingPointError(f"bad gradient norm: {result}")
    return float(result)


def main(root: Path) -> None:
    protocol, cfg = spec(), base_config(root)
    context = initialize_distributed(2); set_process_seed(protocol["pair_seed"], context.rank)
    configure_torch_backend({**cfg.training, "base_seed": protocol["pair_seed"],
                             "sampler_seed": protocol["pair_seed"]})
    dataset = Subject1TrainingDataset(cfg.paths["training_cache"], cfg.paths["stimuli"], cfg.paths["split_ids"])
    plan = DeterministicDistributedBatchPlan(8500, 16, 2, 4, 2, protocol["pair_seed"])
    backbone = load_frozen_backbone(cfg.paths["stable_diffusion"])
    bundle = build_adapter(backbone.unet, 200, 626)
    load_trainable_state_dict(bundle, torch.load(source_snapshot(root)/"model.pt", map_location="cpu", weights_only=True))
    module = CombinedTrainingModule(bundle).to(context.device)
    backbone.vae.to(context.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    backbone.text_encoder.to(context.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    cache, rows, clip_model = load_semantic_assets(root, context.device)
    ddp = DDP(module, device_ids=[context.local_rank], output_device=context.local_rank,
              broadcast_buffers=False, find_unused_parameters=False)
    parameters = [p for p in ddp.parameters() if p.requires_grad]
    main_rng = TrainingGenerators.create(protocol["pair_seed"], context.rank, context.device)
    aux = {}
    for name in ("aux_timestep", "aux_noise"):
        value = torch.Generator(device=context.device); value.manual_seed(namespace_seed(protocol["pair_seed"], name, context.rank)); aux[name] = value
    loader = iter(build_dataloader(dataset, plan, 0, protocol["lambda_calibration_updates"],
                                   context, 4, protocol["pair_seed"] + context.rank))
    ids = backbone.tokenizer("", max_length=backbone.tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(context.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): text = backbone.text_encoder(ids)[0]
    alphas = backbone.noise_scheduler.alphas_cumprod.to(context.device, dtype=torch.float32)
    ratios, records = [], []
    for update in range(protocol["lambda_calibration_updates"]):
        ddp.zero_grad(set_to_none=True); last = None
        for micro in range(2):
            batch = next(loader); last = batch
            image = batch["image"].to(context.device, dtype=torch.bfloat16); brain = batch["brain"].to(context.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = backbone.vae.encode(image).latent_dist.sample(generator=main_rng["vae_latent"]) * backbone.vae.config.scaling_factor
            noise = torch.randn(latent.shape, device=context.device, dtype=latent.dtype, generator=main_rng["diffusion_noise"])
            timestep = torch.randint(0, 1000, (4,), device=context.device, generator=main_rng["timestep"]).long()
            noisy = backbone.noise_scheduler.add_noise(latent, noise, timestep)
            keep = token_keep_mask(4, 200, context.device, main_rng["token_dropout"])
            with (ddp.no_sync() if micro == 0 else contextlib.nullcontext()), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, _ = ddp(noisy, timestep, text.expand(4,-1,-1), brain, keep)
                per = F.mse_loss(prediction.float(), noise.float(), reduction="none").mean((1,2,3))
                loss = (per * min_snr_weights(timestep, alphas, 5.0)).mean()
                (loss/2).backward()
        diffusion_norm = grad_norm(parameters)
        ddp.zero_grad(set_to_none=True)
        image = last["image"][-1:].to(context.device, dtype=torch.bfloat16); brain = last["brain"][-1:].to(context.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = (backbone.vae.encode(image).latent_dist.mode() * backbone.vae.config.scaling_factor).float()
        timestep = torch.randint(protocol["semantic_timestep_min"], protocol["semantic_timestep_max"]+1,
            (1,), device=context.device, generator=aux["aux_timestep"]).long()
        epsilon = torch.randn(z0.shape, device=context.device, generator=aux["aux_noise"])
        alpha = alphas[timestep].view(1,1,1,1); noisy = (alpha.sqrt()*z0 + (1-alpha).sqrt()*epsilon).to(torch.bfloat16)
        target_rows = [rows[int(value)] for value in last["nsd_image_id"][-1:].tolist()]
        target = cache["residual_features"][target_rows].to(context.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, prediction = ddp(noisy, timestep, text, brain, torch.ones((1,200,1),device=context.device),
                                noisy, timestep, brain)
            zhat = (noisy.float()-(1-alpha).sqrt()*prediction.float())/alpha.sqrt()
            decoded = backbone.vae.decode(zhat/backbone.vae.config.scaling_factor).sample
            loss_sem, stats = semantic_loss(decoded, target, clip_model, cache["mean_feature"].to(context.device))
            loss_sem.backward()
        semantic_norm = grad_norm(parameters)
        ratio = diffusion_norm / (semantic_norm + 1e-12)
        ratios.append(ratio); records.append({"update":update,"diffusion_grad_norm":diffusion_norm,
            "semantic_grad_norm":semantic_norm,"ratio":ratio, **stats})
        if context.is_main: print(json.dumps(records[-1]), flush=True)
    gathered = [None, None]; dist.all_gather_object(gathered, records)
    if context.is_main:
        # DDP has already averaged gradients, so both ranks must observe the same norms.
        for left, right in zip(gathered[0], gathered[1]):
            assert abs(left["ratio"]-right["ratio"]) <= max(1e-7, abs(left["ratio"])*1e-6)
        value = protocol["lambda_gradient_fraction"] * statistics.median(ratios)
        assert value > 0 and torch.isfinite(torch.tensor(value))
        target = output(root)/"calibration"; target.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target/"lambda.json", {"status":"passed", "lambda_sem":value,
            "gradient_fraction":protocol["lambda_gradient_fraction"], "ratios":ratios,
            "records":records, "source":source_identity(root),
            "aggregation":"two main microbatches with DDP average; one semantic sample per rank"})
    dataset.close(); dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)

