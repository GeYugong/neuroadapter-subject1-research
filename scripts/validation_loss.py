#!/usr/bin/env python3
"""Compute the frozen deterministic loss on all 500 validation images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.inference import install_inference_state, sample_seed
from neuroadapter_research.modeling import (
    NeuroAdapterTrainingModule,
    build_adapter,
    load_frozen_backbone,
)
from neuroadapter_research.trainer import min_snr_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--protocol", default="subject01-selection-v1")
    args = parser.parse_args()
    if "test" in args.validation_ids.name.lower():
        raise ValueError("validation loss refuses a test-named ID file")

    config = load_training_config(args.config, require_frozen=False)
    configure_torch_backend(config.training)
    dataset = Subject1TrainingDataset(
        config.paths["training_cache"], config.paths["stimuli"], args.validation_ids
    )
    if len(dataset) != 500:
        raise ValueError("deterministic validation loss requires exactly 500 images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    backbone = load_frozen_backbone(config.paths["stable_diffusion"])
    bundle = build_adapter(backbone.unet, dataset.num_parcels, dataset.max_voxels)
    install_inference_state(bundle, args.snapshot)
    module = NeuroAdapterTrainingModule(bundle).to(device).eval()
    backbone.vae.to(device=device, dtype=dtype).eval()
    backbone.text_encoder.to(device=device, dtype=dtype).eval()
    empty_ids = backbone.tokenizer(
        "",
        max_length=backbone.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        empty_text = backbone.text_encoder(empty_ids)[0]
    alphas_cumprod = backbone.noise_scheduler.alphas_cumprod.to(device)

    rows = []
    with torch.no_grad():
        for batch in loader:
            image_ids = [int(value) for value in batch["nsd_image_id"].tolist()]
            images = batch["image"].to(device=device, dtype=dtype)
            brain = batch["brain"].to(device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=dtype):
                latents = backbone.vae.encode(images).latent_dist.mean
                latents = latents * backbone.vae.config.scaling_factor
            noises = []
            timesteps = []
            for image_id in image_ids:
                generator = torch.Generator(device=device).manual_seed(
                    sample_seed(args.protocol, "validation-loss", image_id, 0)
                )
                noises.append(
                    torch.randn(
                        tuple(latents.shape[1:]),
                        generator=generator,
                        device=device,
                        dtype=latents.dtype,
                    )
                )
                timesteps.append(
                    torch.randint(
                        0,
                        backbone.noise_scheduler.config.num_train_timesteps,
                        (1,),
                        generator=generator,
                        device=device,
                    )
                )
            noise = torch.stack(noises)
            timestep = torch.cat(timesteps).long()
            noisy = backbone.noise_scheduler.add_noise(latents, noise, timestep)
            keep_mask = torch.ones(
                (len(image_ids), dataset.num_parcels, 1), device=device, dtype=torch.bool
            )
            with torch.autocast("cuda", dtype=dtype):
                prediction = module(
                    noisy,
                    timestep,
                    empty_text.expand(len(image_ids), -1, -1),
                    brain,
                    keep_mask,
                )
            mse = F.mse_loss(prediction.float(), noise.float(), reduction="none").mean(
                dim=(1, 2, 3)
            )
            weights = min_snr_weights(timestep, alphas_cumprod, 5.0)
            losses = mse * weights
            rows.extend(
                {
                    "image_id": image_id,
                    "timestep": int(step),
                    "loss": float(loss),
                }
                for image_id, step, loss in zip(
                    image_ids, timestep.cpu().tolist(), losses.cpu().tolist(), strict=True
                )
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "timestep", "loss"))
        writer.writeheader()
        writer.writerows(rows)
    values = torch.tensor([row["loss"] for row in rows], dtype=torch.float64)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "image_count": len(rows),
        "draws_per_image": 1,
        "vae_posterior": "mean",
        "token_dropout": "disabled",
        "min_snr_gamma": 5.0,
        "aggregation": "equal mean over images",
        "config_sha256": config.sha256,
        "mean_loss": float(values.mean()),
        "std_loss": float(values.std(unbiased=True)),
        "snapshot_sha256": sha256_file(
            args.snapshot / "model.pt" if args.snapshot.is_dir() else args.snapshot
        ),
        "validation_ids_sha256": sha256_file(args.validation_ids),
        "per_image_csv_sha256": sha256_file(args.output_csv),
    }
    write_json_atomic(args.output_json, payload)
    dataset.close()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
