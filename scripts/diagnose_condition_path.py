#!/usr/bin/env python3
"""Paired exploratory diagnosis of a frozen checkpoint; never trains a model."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from neuroadapter_research import inference
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.modeling import (
    NeuroAdapterTrainingModule, build_adapter, load_frozen_backbone,
)

NAMESPACE = "condition-diagnosis-20260914-v1"
WEIGHT_SHA = "bdca167505e0f1e62e025a5856299c56548dc40c2231740b8d2e1f84665b8217"
TIMESTEPS = [50, 200, 500, 800, 950]


def choose_pairs(ids, split, count=32):
    ids = [int(i) for i in ids]
    if len(set(ids)) != len(ids) or len(ids) < count or count < 2:
        raise ValueError("unique IDs and at least two selected samples required")
    ranked = sorted(ids, key=lambda i: hashlib.sha256(
        f"{NAMESPACE}:{split}:{i}".encode()).digest())[:count]
    return [{"image_id": i, "donor_id": ranked[(j + 1) % count]}
            for j, i in enumerate(ranked)]


def jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()


def png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite image: {path}")
    array = value.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255).round().astype(np.uint8)).save(path)


class NoiseReplay:
    """Replay real tensors, including every ancestral sampling noise draw."""
    def __init__(self, tensors):
        self.tensors = tensors
        self.index = 0

    def take(self, shape, device, dtype):
        if self.index >= len(self.tensors):
            raise AssertionError("unexpected extra noise draw")
        value = self.tensors[self.index]
        self.index += 1
        if tuple(value.shape) != tuple(shape):
            raise AssertionError(f"noise shape mismatch: {value.shape} != {shape}")
        result = value.to(device=device, dtype=dtype)
        if not torch.equal(result.float().cpu(), value.float().cpu()):
            raise AssertionError("noise values changed during precision conversion")
        return result

    def initial(self, shape, generators, *, device, dtype):
        return self.take((len(generators), *shape), device, dtype)

    def step(self, shape, generator=None, device=None, dtype=None, **kwargs):
        return self.take(shape, device, dtype)

    def complete(self):
        if self.index != len(self.tensors):
            raise AssertionError("not all frozen noise tensors consumed")


def prepare_noise(backbone, image_id, split, device):
    from diffusers.utils.torch_utils import randn_tensor
    generators = [torch.Generator(device=device).manual_seed(
        inference.sample_seed(NAMESPACE, split, image_id, j)) for j in range(2)]
    values = [inference._randn_per_generator(
        (4, 64, 64), generators, device=device, dtype=torch.bfloat16).cpu()
        for _ in range(2)]
    backbone.noise_scheduler.set_timesteps(50, device=device)
    for timestep in backbone.noise_scheduler.timesteps:
        if int(timestep) > 0:
            values.append(randn_tensor((2, 4, 64, 64), generator=generators,
                                      device=device, dtype=torch.bfloat16).cpu())
    return values


def generate(backbone, bundle, brain, image_id, split, device, dtype, guidance, noise):
    replay = NoiseReplay(noise)
    with patch.object(inference, "_randn_per_generator", replay.initial), patch(
        "diffusers.schedulers.scheduling_ddpm.randn_tensor", replay.step
    ):
        result = inference.generate_candidates(
            bundle=bundle, backbone=backbone, brain=brain, image_id=image_id,
            candidate_count=2, protocol=NAMESPACE, split=split, device=device,
            dtype=dtype, denoising_steps=50, guidance_scale=guidance)
    replay.complete()
    return result


def load_models(config, snapshot, device, dtype):
    # Always reload FP32 source weights: BF16 -> FP32 casting is not a reference.
    backbone = load_frozen_backbone(config.paths["stable_diffusion"])
    bundle = build_adapter(backbone.unet, 200, 626)
    inference.install_inference_state(bundle, snapshot)
    bundle.neuro_adapter.to(device=device, dtype=dtype).requires_grad_(False)
    bundle.guidance_generator.to(device=device, dtype=torch.float32).requires_grad_(False)
    backbone.vae.to(device=device, dtype=dtype)
    backbone.text_encoder.to(device=device, dtype=dtype)
    return backbone, bundle


def autocast(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()


def freeze(config, out, snapshot):
    if out.exists():
        raise FileExistsError(out)
    train = np.loadtxt(config.paths["split_ids"], dtype=np.int64, ndmin=1)
    val = np.loadtxt(config.paths["validation_ids"], dtype=np.int64, ndmin=1)
    assert len(train) == 8500 and len(val) == 500 and not set(train) & set(val)
    assert sha256_file(snapshot / "model.pt") == WEIGHT_SHA
    out.mkdir(parents=True)
    manifest = {
        "namespace": NAMESPACE, "status": "frozen_exploratory_diagnostic",
        "checkpoint_role": "diagnostic_object_not_performance_accepted",
        "checkpoint": str(snapshot), "checkpoint_sha256": WEIGHT_SHA,
        "runtime_commit": config.raw["protocol_commit"], "config_sha256": config.sha256,
        "diagnostic_script_sha256": sha256_file(Path(__file__)),
        "diagnostic_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip(),
        "split_hashes": {name: sha256_file(config.paths[key]) for name, key in
                         [("train", "split_ids"), ("validation", "validation_ids")]},
        "selection": "lowest SHA256(namespace:split:image_id); cyclic donor derangement",
        "pairs": {"train": choose_pairs(train, "train"), "validation": choose_pairs(val, "validation")},
        "candidate_count": 2, "candidate_batch_size": 2, "steps": 50,
        "timesteps": TIMESTEPS, "guidance": [4.0, 2.0, 6.0],
        "noise": "pre-generated current BF16 tensors; exact values replayed in FP32",
        "fp32_reference": "fresh source weights; autocast off; TF32 off",
        "statistics": "image-level paired means over two candidates; exploratory bootstrap CI; no selection",
        "limitations": ["previously used internal validation", "one fixed donor per target",
                        "32 images per split", "no independent test", "no checkpoint selection"],
    }
    write_json_atomic(out / "protocol.json", manifest)
    print(json.dumps(manifest, indent=2))


@torch.no_grad()
def worker(config, out, snapshot, split, phase, device):
    started = time.time()
    protocol = json.loads((out / "protocol.json").read_text())
    assert sha256_file(snapshot / "model.pt") == protocol["checkpoint_sha256"]
    assert sha256_file(Path(__file__)) == protocol["diagnostic_script_sha256"]
    target = out / split / phase
    target.mkdir(parents=True, exist_ok=False)
    dataset = Subject1TrainingDataset(config.paths["training_cache"], config.paths["stimuli"],
                                     config.paths["split_ids" if split == "train" else "validation_ids"])
    lookup = {int(value): i for i, value in enumerate(dataset.image_ids)}
    dtype = torch.float32 if phase == "fp32" else torch.bfloat16
    settings = configure_torch_backend(config.training)
    if phase in ("vae", "fp32"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    write_json_atomic(target / "environment.json", {
        "torch": torch.__version__, "device": torch.cuda.get_device_name(device),
        "settings": settings, "actual_tf32": torch.backends.cuda.matmul.allow_tf32,
        "script_sha256": sha256_file(Path(__file__)), "checkpoint_sha256": WEIGHT_SHA,
        "protocol_sha256": sha256_file(out / "protocol.json"), "phase": phase,
    })
    if phase == "vae":
        from diffusers import AutoencoderKL
        for precision, vae_dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            vae = AutoencoderKL.from_pretrained(config.paths["stable_diffusion"], subfolder="vae",
                                              local_files_only=True).to(device=device, dtype=vae_dtype).eval()
            # Restore the actual current backend for the BF16 half of the audit.
            torch.backends.cuda.matmul.allow_tf32 = precision == "bf16" and settings["allow_tf32"]
            torch.backends.cudnn.allow_tf32 = precision == "bf16" and settings["allow_tf32"]
            for pair in protocol["pairs"][split]:
                image_id = pair["image_id"]
                image = dataset[lookup[image_id]]["image"].unsqueeze(0).to(device)
                png(out / split / "gt" / f"{image_id}.png", (image[0] + 1) / 2)
                with autocast(vae_dtype):
                    distribution = vae.encode(image.to(vae_dtype)).latent_dist
                    generator = torch.Generator(device=device).manual_seed(
                        inference.sample_seed(NAMESPACE + "vae", split, image_id, 0))
                    epsilon = torch.randn(distribution.mean.shape, generator=generator, device=device,
                                          dtype=torch.bfloat16).to(vae_dtype)
                    for mode, latent in [("mode", distribution.mode()),
                                         ("sample", distribution.mean + distribution.std * epsilon)]:
                        scaled = latent * vae.config.scaling_factor
                        reconstruction = vae.decode(scaled / vae.config.scaling_factor).sample.float()
                        assert torch.isfinite(reconstruction).all() and torch.isfinite(latent).all()
                        png(target / precision / mode / f"{image_id}.png", (reconstruction[0] + 1) / 2)
                        jsonl(target / "metrics.jsonl", {
                            "image_id": image_id, "precision": precision, "mode": mode,
                            "mse_rgb": float(((image - reconstruction) / 2).square().mean()),
                            "input_min": float(image.min()), "input_max": float(image.max()),
                            "latent_mean": float(latent.float().mean()), "latent_std": float(latent.float().std()),
                            "decoded_min": float(reconstruction.min()), "decoded_max": float(reconstruction.max()),
                            "scaling_factor": float(vae.config.scaling_factor), "finite": True})
            del vae
            torch.cuda.empty_cache()
    else:
        backbone, bundle = load_models(config, snapshot, device, dtype)
        model = NeuroAdapterTrainingModule(bundle).eval()
        for index, pair in enumerate(protocol["pairs"][split]):
            image_id, donor_id = pair["image_id"], pair["donor_id"]
            sample, donor = dataset[lookup[image_id]], dataset[lookup[donor_id]]
            assert sample["brain"].shape == donor["brain"].shape == (200, 626)
            assert not torch.equal(sample["brain"], donor["brain"])
            if phase == "bf16":
                with autocast(dtype):
                    image = sample["image"].unsqueeze(0).to(device=device, dtype=dtype)
                    generator = torch.Generator(device=device).manual_seed(
                        inference.sample_seed(NAMESPACE + "loss", split, image_id, 0))
                    dist = backbone.vae.encode(image).latent_dist
                    latent = dist.sample(generator=generator) * backbone.vae.config.scaling_factor
                    noise = torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=generator)
                    empty_ids = backbone.tokenizer("", max_length=backbone.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                    empty = backbone.text_encoder(empty_ids)[0]
                    for t in TIMESTEPS:
                        timesteps = torch.tensor([t], device=device)
                        noisy = backbone.noise_scheduler.add_noise(latent, noise, timesteps)
                        losses = []
                        predictions = []
                        for item in (sample, donor):
                            prediction = model(noisy, timesteps, empty, item["brain"].unsqueeze(0).to(device),
                                               torch.ones((1, 200, 1), device=device))
                            assert torch.isfinite(prediction).all()
                            losses.append(float((prediction.float() - noise.float()).square().mean()))
                            predictions.append(prediction.float())
                        jsonl(target / "denoising.jsonl", {**pair, "timestep": t,
                            "correct_mse": losses[0], "wrong_mse": losses[1], "delta": losses[1] - losses[0],
                            "prediction_difference_rms": float((predictions[0]-predictions[1]).square().mean().sqrt())})
                noise_bank = prepare_noise(backbone, image_id, split, device)
                noise_path = target / "noise" / f"{image_id}.pt"
                noise_path.parent.mkdir(exist_ok=True)
                torch.save(noise_bank, noise_path)
            else:
                noise_path = out / split / "bf16" / "noise" / f"{image_id}.pt"
                noise_bank = torch.load(noise_path, weights_only=True)
            settings_to_run = [(4., "correct", sample), (4., "wrong", donor)] if phase != "guidance" else [
                (2., "correct", sample), (6., "correct", sample)]
            for guidance, condition, item in settings_to_run:
                candidates = generate(backbone, bundle, item["brain"], image_id, split, device,
                                      dtype, guidance, noise_bank)
                if phase == "bf16" and index == 0 and condition == "correct":
                    original = inference.generate_candidates(bundle=bundle, backbone=backbone,
                        brain=item["brain"], image_id=image_id, candidate_count=2, protocol=NAMESPACE,
                        split=split, device=device, dtype=dtype, denoising_steps=50, guidance_scale=4.)
                    assert torch.equal(original, candidates), "noise replay changed frozen BF16 inference"
                    write_json_atomic(target / "noise_equivalence.json", {
                        "image_id": image_id, "original_vs_replay_exact": True,
                        "max_difference": float((original-candidates).abs().max()), "draws": len(noise_bank)})
                for candidate, value in enumerate(candidates):
                    relative = Path(f"g{int(guidance)}-{condition}") / f"{image_id}-{candidate}.png"
                    png(target / relative, value)
                    jsonl(target / "images.jsonl", {**pair, "candidate": candidate,
                        "guidance": guidance, "condition": condition, "path": str(relative),
                        "sha256": sha256_file(target / relative), "noise_sha256": sha256_file(noise_path)})
            print(f"{split} {phase} {index+1}/32 id={image_id} elapsed={time.time()-started:.1f}s", flush=True)
    dataset.close()
    write_json_atomic(target / "complete.json", {"status": "complete", "seconds": time.time()-started})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["freeze", "vae", "bf16", "fp32", "guidance"], required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.project_root
    config = load_training_config(root / "configs/formal/subject01_selection_v2.yaml", require_frozen=True)
    snapshot = config.paths["output_dir"] / "evaluation-20260910/selected_snapshot"
    out = root / "runs/diagnostics/20260914-condition-path"
    if args.phase == "freeze":
        freeze(config, out, snapshot)
    else:
        worker(config, out, snapshot, args.split, args.phase, torch.device(args.device))


if __name__ == "__main__":
    main()
