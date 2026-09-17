"""Paired R->K/M training for the final bounded generator experiment."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import signal
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from generator_semantic_core import (base_config, clip_asset, output, semantic_loss,
                                     source_identity, source_snapshot, spec)
from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import (load_distributed_checkpoint,
    prune_full_checkpoints, save_distributed_checkpoint, save_inference_snapshot,
    verify_inference_snapshot)
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.modeling import (NeuroAdapterTrainingModule, audit_trainable_parameters,
    build_adapter, load_frozen_backbone, load_trainable_state_dict, tensor_state_sha256,
    trainable_state_dict)
from neuroadapter_research.rng import (TrainingGenerators, capture_process_rng_state,
    namespace_seed, restore_process_rng_state)
from neuroadapter_research.sampler import DeterministicDistributedBatchPlan
from neuroadapter_research.trainer import (TerminationFlag, all_ranks_agree_on_hash,
    append_json_line, build_dataloader, initialize_distributed, min_snr_weights,
    move_optimizer_state_to_device, run_rank_zero_action, set_process_seed,
    synchronize_distributed_error, token_keep_mask)


class CombinedTrainingModule(NeuroAdapterTrainingModule):
    """Single DDP traversal for the final main microbatch and semantic branch."""
    def forward(self, noisy_latents, timesteps, text_embeddings, brain, token_keep_mask,
                aux_noisy=None, aux_timesteps=None, aux_brain=None):
        main = super().forward(noisy_latents, timesteps, text_embeddings, brain, token_keep_mask)
        if aux_noisy is None:
            return main, None
        ones = torch.ones((aux_brain.shape[0], brain.shape[1], 1),
                          device=brain.device, dtype=brain.dtype)
        aux = super().forward(aux_noisy, aux_timesteps,
                              text_embeddings[:aux_brain.shape[0]], aux_brain, ones)
        return main, aux


def config_hash(arm: str, maximum: int, lambda_sem: float | None, identity: dict) -> str:
    payload = {"arm": arm, "maximum": maximum, "lambda_sem": lambda_sem,
               "identity": identity, "spec": spec()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_semantic_assets(root: Path, device: torch.device):
    cache_path = output(root) / "semantic-features/train8500-clip-vit-l14.pt"
    manifest = output(root) / "semantic-features/manifest.json"
    from neuroadapter_research.atomic import sha256_file
    record = json.loads(manifest.read_text())
    assert record["cache_sha256"] == sha256_file(cache_path)
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    ids = cached["image_ids"].tolist()
    row = {int(image_id): index for index, image_id in enumerate(ids)}
    sys.path.insert(0, str(root / "repo/vendor/CLIP"))
    import clip
    model, _ = clip.load(str(clip_asset(root)), device=device, jit=False)
    model.eval().requires_grad_(False)
    return cached, row, model


def run(root: Path, arm: str, maximum: int, resume: Path | None = None,
        test_stop: int | None = None, pilot_stop: bool = False) -> None:
    cfg, protocol = base_config(root), spec()
    assert arm in ("K", "M")
    assert maximum == (5000 if arm == "K" else protocol["maximum_semantic_updates"])
    if pilot_stop and arm != "M": raise ValueError("pilot stop is only valid for M")
    limit = test_stop or (protocol["pilot_updates"] if pilot_stop else maximum)
    identity = source_identity(root)
    lambda_sem = None
    if arm == "M":
        calibration = json.loads((output(root) / "calibration/lambda.json").read_text())
        assert calibration["status"] == "passed" and len(calibration["ratios"]) == 16
        lambda_sem = float(calibration["lambda_sem"])
        assert lambda_sem > 0
    run_dir = output(root) / (f"tests/{arm}-{limit}" if test_stop else arm)
    if resume is None and run_dir.exists():
        raise FileExistsError(run_dir)
    context = initialize_distributed(2)
    set_process_seed(protocol["pair_seed"], context.rank)
    training = dict(cfg.training)
    training.update(base_seed=protocol["pair_seed"], sampler_seed=protocol["pair_seed"],
                    learning_rate=protocol["learning_rate"], max_updates=maximum)
    backend = configure_torch_backend(training)
    digest = config_hash(arm, maximum, lambda_sem, identity)
    dataset = Subject1TrainingDataset(cfg.paths["training_cache"], cfg.paths["stimuli"], cfg.paths["split_ids"])
    plan = DeterministicDistributedBatchPlan(8500, 16, 2, 4, 2, protocol["pair_seed"])
    backbone = load_frozen_backbone(cfg.paths["stable_diffusion"])
    bundle = build_adapter(backbone.unet, 200, 626)
    verify_inference_snapshot(source_snapshot(root))
    load_trainable_state_dict(bundle, torch.load(source_snapshot(root) / "model.pt",
                                                  map_location="cpu", weights_only=True))
    audit = audit_trainable_parameters(bundle)
    initial_hash = tensor_state_sha256(trainable_state_dict(bundle))
    all_ranks_agree_on_hash(initial_hash, context)
    module = CombinedTrainingModule(bundle).to(context.device)
    backbone.vae.to(context.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    backbone.text_encoder.to(context.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    semantic_cache = semantic_rows = clip_model = None
    if arm == "M":
        semantic_cache, semantic_rows, clip_model = load_semantic_assets(root, context.device)
    kwargs = dict(device_ids=[context.local_rank], output_device=context.local_rank,
                  broadcast_buffers=False, find_unused_parameters=False)
    if "init_sync" in inspect.signature(DDP).parameters: kwargs["init_sync"] = False
    ddp = DDP(module, **kwargs)
    parameters = [p for p in ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=protocol["learning_rate"], betas=(.9, .999),
                                  eps=1e-8, weight_decay=1e-6, fused=False, foreach=False)
    main_rng = TrainingGenerators.create(protocol["pair_seed"], context.rank, context.device)
    aux_generators = {}
    for name in ("aux_timestep", "aux_noise"):
        generator = torch.Generator(device=context.device)
        generator.manual_seed(namespace_seed(protocol["pair_seed"], name, context.rank))
        aux_generators[name] = generator
    aux_cuda_state = torch.Generator(device=context.device)
    aux_cuda_state.manual_seed(namespace_seed(protocol["pair_seed"], "aux_module", context.rank))
    aux_cuda_rng = aux_cuda_state.get_state()
    payload, start = None, 0
    if resume is not None:
        payload = load_distributed_checkpoint(resume, context.rank, 2)
        state = payload["trainer"]
        assert state["config_hash"] == digest and state["arm"] == arm
        start = int(state["next_update"]); assert 0 <= start < limit
        plan.validate_state(state["sampler"])
        load_trainable_state_dict(bundle, payload["model"])
        optimizer.load_state_dict(payload["optimizer"]); move_optimizer_state_to_device(optimizer, context.device)
        main_rng.load_state_dict(payload["rank"]["main_rng"])
        for name, generator in aux_generators.items(): generator.set_state(payload["rank"][name].cpu())
        aux_cuda_rng = payload["rank"]["aux_cuda_rng"].cpu()
        restore_process_rng_state(payload["rank"]["process_rng"], context.device)
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        effective = {"experiment": protocol["experiment_type"], "arm": arm,
            "source": identity, "local_maximum": maximum, "lambda_sem": lambda_sem,
            "training": training, "backend": backend, "config_hash": digest,
            "parameter_audit": audit, "initial_tensor_sha256": initial_hash,
            "test_only": test_stop is not None, "pilot_stop": pilot_stop}
        if not resume: write_json_atomic(run_dir / "effective_config.json", effective)
        write_json_atomic(run_dir / "status.json", {"status": "running", "completed_updates": start})
    dist.barrier()
    text_ids = backbone.tokenizer("", max_length=backbone.tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(context.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        text = backbone.text_encoder(text_ids)[0]
    alphas = backbone.noise_scheduler.alphas_cumprod.to(context.device, dtype=torch.float32)
    iterator = iter(build_dataloader(dataset, plan, start, limit, context, 4,
                                     protocol["pair_seed"] + context.rank))
    termination = TerminationFlag()
    signal.signal(signal.SIGTERM, termination.handler); signal.signal(signal.SIGINT, termination.handler)

    def save(completed: int, snapshot: bool = False):
        state = trainable_state_dict(bundle) if context.is_main else {}
        metadata = {"experiment": protocol["experiment_type"], "arm": arm,
            "source_update": protocol["source_update"], "source_R_sha256": protocol["source_sha256"],
            "local_update": completed, "optimizer_update": completed,
            "lambda_sem": lambda_sem, "config_hash": digest,
            "implementation_commit": identity["implementation_commit"], "test_only": test_stop is not None,
            "pilot_stop": pilot_stop}
        if snapshot:
            run_rank_zero_action(context, "snapshot", lambda: save_inference_snapshot(
                run_dir / "snapshots", completed, state, metadata))
        save_distributed_checkpoint(run_dir / "checkpoints", completed, context.rank, 2, state,
            optimizer.state_dict() if context.is_main else {},
            {**metadata, "next_update": completed, "sampler": plan.state_before_update(completed).to_dict()},
            {"rank": context.rank, "main_rng": main_rng.state_dict(),
             **{name: gen.get_state().cpu() for name, gen in aux_generators.items()},
             "aux_cuda_rng": aux_cuda_rng.cpu(),
             "process_rng": capture_process_rng_state(context.device)},
            dist.barrier, lambda error: synchronize_distributed_error(context, error))
        run_rank_zero_action(context, "prune checkpoints", lambda: prune_full_checkpoints(
            run_dir / "checkpoints", keep_latest=2))

    if resume is None: save(0)
    started = time.perf_counter(); window = started; losses, sem_losses, saturations = [], [], []
    completed = start
    for update in range(start, limit):
        optimizer.zero_grad(set_to_none=True)
        per_update = []
        for micro in range(2):
            batch = next(iterator)
            images = batch["image"].to(context.device, dtype=torch.bfloat16, non_blocking=True)
            brain = batch["brain"].to(context.device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latents = backbone.vae.encode(images).latent_dist.sample(generator=main_rng["vae_latent"])
                latents = latents * backbone.vae.config.scaling_factor
            noise = torch.randn(latents.shape, device=context.device, dtype=latents.dtype,
                                generator=main_rng["diffusion_noise"])
            timesteps = torch.randint(0, backbone.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=context.device, generator=main_rng["timestep"]).long()
            noisy = backbone.noise_scheduler.add_noise(latents, noise, timesteps)
            keep = token_keep_mask(latents.shape[0], 200, context.device, main_rng["token_dropout"])
            aux_noisy = aux_t = aux_brain = target = None
            if arm == "M" and micro == 1:
                z0 = latents[-1:].detach().float()
                aux_t = torch.randint(protocol["semantic_timestep_min"], protocol["semantic_timestep_max"] + 1,
                    (1,), device=context.device, generator=aux_generators["aux_timestep"]).long()
                eps = torch.randn(z0.shape, device=context.device, dtype=torch.float32,
                                  generator=aux_generators["aux_noise"])
                alpha = alphas[aux_t].view(1, 1, 1, 1)
                aux_noisy = (alpha.sqrt() * z0 + (1 - alpha).sqrt() * eps).to(torch.bfloat16)
                aux_brain = brain[-1:]
                rows = [semantic_rows[int(value)] for value in batch["nsd_image_id"][-1:].tolist()]
                target = semantic_cache["residual_features"][rows].to(context.device)
            sync = contextlib.nullcontext() if micro == 1 else ddp.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                if aux_noisy is not None:
                    with torch.random.fork_rng(devices=[context.device.index]):
                        torch.cuda.set_rng_state(aux_cuda_rng, context.device)
                        prediction, aux_prediction = ddp(noisy, timesteps, text.expand(4, -1, -1),
                            brain, keep, aux_noisy, aux_t, aux_brain)
                        aux_cuda_rng = torch.cuda.get_rng_state(context.device).cpu()
                else:
                    prediction, aux_prediction = ddp(noisy, timesteps, text.expand(4, -1, -1), brain, keep)
                sample_loss = F.mse_loss(prediction.float(), noise.float(), reduction="none").mean((1,2,3))
                diffusion = (sample_loss * min_snr_weights(timesteps, alphas, 5.0)).mean()
                total = diffusion / 2.0
                if aux_prediction is not None:
                    alpha = alphas[aux_t].view(1, 1, 1, 1)
                    z0_hat = (aux_noisy.float() - (1-alpha).sqrt()*aux_prediction.float()) / alpha.sqrt()
                    decoded = backbone.vae.decode(z0_hat / backbone.vae.config.scaling_factor).sample
                    sem, stats = semantic_loss(decoded, target, clip_model,
                        semantic_cache["mean_feature"].to(context.device))
                    total = total + lambda_sem * sem
                    sem_losses.append(float(sem.detach())); saturations.append(stats["clamp_saturation_fraction"])
                total.backward()
            if not torch.isfinite(diffusion): raise FloatingPointError(f"nonfinite loss at {update}")
            per_update.append(float(diffusion.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm): raise FloatingPointError(f"nonfinite gradient at {update}")
        optimizer.step(); completed = update + 1; losses.append(sum(per_update) / 2)
        if completed == 1 or completed % 100 == 0 or completed == limit:
            now = time.perf_counter(); memory = [None, None]
            dist.all_gather_object(memory, torch.cuda.max_memory_reserved(context.device))
            if context.is_main:
                record = {"completed_updates": completed, "source_update": protocol["source_update"],
                    "mean_diffusion_loss": sum(losses)/len(losses),
                    "mean_semantic_loss": (sum(sem_losses)/len(sem_losses) if sem_losses else None),
                    "mean_clamp_saturation_fraction": (sum(saturations)/len(saturations) if saturations else None),
                    "gradient_norm_before_clip": float(norm), "updates_per_second": len(losses)/(now-window),
                    "elapsed_seconds": now-started, "peak_reserved_bytes_by_rank": memory}
                append_json_line(run_dir / "training.jsonl", record)
                write_json_atomic(run_dir / "status.json", {"status":"running", **record})
                print(json.dumps({"arm": arm, **record}), flush=True)
            losses, sem_losses, saturations, window = [], [], [], now
        signal_stop = torch.tensor(int(termination.requested), device=context.device)
        dist.all_reduce(signal_stop, op=dist.ReduceOp.MAX)
        stop = bool(signal_stop.item())
        if completed % 5000 == 0 or completed == limit or stop:
            save(completed, snapshot=completed in protocol["snapshot_updates"] or completed == limit)
        if stop: break
    if context.is_main:
        final_status = "pilot_complete" if pilot_stop and completed == limit else ("completed" if completed == limit else "interrupted")
        write_json_atomic(run_dir / "status.json", {"status": final_status,
            "completed_updates": completed, "seconds": time.perf_counter()-started,
            "exit_code": 0, "test_only": test_stop is not None})
    dataset.close(); dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["K", "M"], required=True)
    parser.add_argument("--maximum", type=int, required=True); parser.add_argument("--resume", type=Path)
    parser.add_argument("--test-stop", type=int); parser.add_argument("--pilot-stop", action="store_true")
    args = parser.parse_args(); run(args.root, args.arm, args.maximum, args.resume, args.test_stop, args.pilot_stop)
