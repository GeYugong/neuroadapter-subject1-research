#!/usr/bin/env python3
"""Deterministically decode an internal-validation split from one snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neuroadapter_research.atomic import fsync_directory, sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.inference import generate_candidates, install_inference_state
from neuroadapter_research.modeling import build_adapter, load_frozen_backbone


def save_png_atomic(path: Path, image: torch.Tensor) -> None:
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG", compress_level=9)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, choices=(2, 8), required=True)
    parser.add_argument("--protocol", default="subject01-selection-v1")
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()
    if "test" in args.validation_ids.name.lower():
        raise ValueError("validation decoder refuses a test-named ID file")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("decode output directory must be empty")

    config = load_training_config(args.config, require_frozen=False)
    configure_torch_backend(config.training)
    dataset = Subject1TrainingDataset(
        config.paths["training_cache"], config.paths["stimuli"], args.validation_ids
    )
    if len(dataset) != 500:
        raise ValueError("deterministic selection decoding requires exactly 500 images")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    backbone = load_frozen_backbone(config.paths["stable_diffusion"])
    bundle = build_adapter(backbone.unet, dataset.num_parcels, dataset.max_voxels)
    install_inference_state(bundle, args.snapshot)
    bundle.neuro_adapter.to(device=device, dtype=dtype)
    bundle.guidance_generator.to(device=device, dtype=torch.float32)
    backbone.text_encoder.to(device=device, dtype=dtype).eval()
    backbone.vae.to(device=device, dtype=dtype).eval()

    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image_id = int(sample["nsd_image_id"])
        candidates = generate_candidates(
            bundle=bundle,
            backbone=backbone,
            brain=sample["brain"],
            image_id=image_id,
            candidate_count=args.candidate_count,
            protocol=args.protocol,
            split="validation",
            device=device,
            dtype=dtype,
            denoising_steps=args.denoising_steps,
            guidance_scale=args.guidance_scale,
        )
        files = []
        for candidate_index, image in enumerate(candidates):
            relative = Path(f"candidate-{candidate_index:02d}") / f"{image_id:05d}.png"
            output = args.output_dir / relative
            save_png_atomic(output, image)
            files.append(
                {
                    "candidate_index": candidate_index,
                    "path": relative.as_posix(),
                    "sha256": sha256_file(output),
                }
            )
        records.append({"image_id": image_id, "files": files})

    payload = {
        "schema_version": 1,
        "status": "complete",
        "split": "validation",
        "image_count": len(records),
        "candidate_count": args.candidate_count,
        "denoising_steps": args.denoising_steps,
        "guidance_scale": args.guidance_scale,
        "protocol": args.protocol,
        "config_sha256": config.sha256,
        "snapshot_sha256": sha256_file(
            args.snapshot / "model.pt" if args.snapshot.is_dir() else args.snapshot
        ),
        "validation_ids_sha256": sha256_file(args.validation_ids),
        "records": records,
    }
    write_json_atomic(args.output_dir / "decode_manifest.json", payload)
    dataset.close()
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
