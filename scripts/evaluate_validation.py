#!/usr/bin/env python3
"""Offline eight-metric evaluation for one deterministic decode manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode, v2

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.integrity import load_json_mapping, verify_file_record
from neuroadapter_research.metrics import paired_correlation_distance, pixel_metrics
from neuroadapter_research.protocol import (
    image_order_sha256,
    load_selection_plan,
    method_fingerprint,
    validate_selection_config_and_plan,
    validate_selection_plan_inputs,
    verify_protocol_repository,
)
from neuroadapter_research.selection import per_image_two_way_identification


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def verify_evaluation_assets(project_root: Path, manifest_path: Path) -> dict[str, Path]:
    manifest = load_json_mapping(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise ValueError("evaluation manifest has no named files mapping")
    result = {}
    for name, record in records.items():
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError(f"invalid evaluation asset record: {name}")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"evaluation asset path is not contained: {relative}")
        path = project_root / relative
        verify_file_record(path, record)
        result[name] = path
    return result


def load_decode_set(
    decode_manifest_path: Path, stimuli_path: Path
) -> tuple[list[int], np.ndarray, list[np.ndarray], dict]:
    manifest = load_json_mapping(decode_manifest_path)
    if manifest.get("status") != "complete" or manifest.get("split") != "validation":
        raise ValueError("evaluator accepts only complete internal-validation decodes")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("decode manifest must contain exactly 500 image records")
    image_ids = [int(record["image_id"]) for record in records]
    if len(set(image_ids)) != 500:
        raise ValueError("decode manifest image IDs are not unique")
    candidate_count = int(manifest["candidate_count"])
    if candidate_count not in (2, 8):
        raise ValueError("candidate count must be 2 or 8")
    root = decode_manifest_path.parent

    reconstructed = [[] for _ in range(candidate_count)]
    for record in records:
        files = record.get("files")
        if not isinstance(files, list) or len(files) != candidate_count:
            raise ValueError("decode record has the wrong candidate count")
        for candidate_index, file_record in enumerate(files):
            if int(file_record["candidate_index"]) != candidate_index:
                raise ValueError("candidate files are not in canonical order")
            path = root / file_record["path"]
            if sha256_file(path) != file_record["sha256"]:
                raise ValueError(f"decoded PNG hash mismatch: {path}")
            reconstructed[candidate_index].append(np.asarray(Image.open(path).convert("RGB")))
    reconstructed_arrays = [np.stack(images) for images in reconstructed]
    with h5py.File(stimuli_path, "r") as h5:
        originals = np.stack([np.asarray(h5["imgBrick"][image_id]) for image_id in image_ids])
    return image_ids, originals, reconstructed_arrays, manifest


def preprocess_transform(size: int, mean: list[float], std: list[float]) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def extract_features(
    images: np.ndarray,
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    transform: Callable[[np.ndarray], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    output_key: str | None = None,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([transform(image) for image in images[start : start + batch_size]])
        value = model(batch.to(device))
        if output_key is not None:
            value = value[output_key]
        outputs.append(value.float().flatten(1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def model_features(
    originals: np.ndarray,
    reconstructed: list[np.ndarray],
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    transform: Callable[[np.ndarray], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    output_key: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    original = extract_features(
        originals, model, transform, device=device, batch_size=batch_size, output_key=output_key
    )
    candidates = [
        extract_features(
            images, model, transform, device=device, batch_size=batch_size, output_key=output_key
        )
        for images in reconstructed
    ]
    return original, candidates


def identification_by_seed(original: np.ndarray, candidates: list[np.ndarray]) -> list[np.ndarray]:
    return [100.0 * per_image_two_way_identification(original, value) for value in candidates]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decode-manifest", type=Path, required=True)
    parser.add_argument("--validation-loss", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_training_config(args.config, require_frozen=True)
    configure_torch_backend(config.training)
    repository = Path(__file__).resolve().parents[1]
    verify_protocol_repository(repository, config.raw["protocol_commit"])
    plan = load_selection_plan(config.paths["selection_plan"], require_frozen=True)
    validate_selection_config_and_plan(config, plan)
    plan_binding = validate_selection_plan_inputs(
        plan,
        validation_ids_path=config.paths["validation_ids"],
        repository_root=repository,
    )
    fingerprint = method_fingerprint(config)
    project_root = Path(config.raw["project_root"]).resolve()
    assets = verify_evaluation_assets(project_root, config.paths["evaluation_manifest"])
    os.environ["TORCH_HOME"] = str(project_root / "models/evaluation/torch")
    torch.hub.set_dir(str(project_root / "models/evaluation/torch/hub"))
    device = torch.device(args.device)
    image_ids, originals, reconstructed, decode_manifest = load_decode_set(
        args.decode_manifest, config.paths["stimuli"]
    )
    expected_decode = {
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        **plan_binding,
        "protocol_namespace": plan.raw["protocol_namespace"],
        "denoising_steps": plan.raw["denoising_steps"],
        "guidance_scale": plan.raw["guidance_scale"],
        "repository_commit": config.raw["protocol_commit"],
    }
    for name, expected in expected_decode.items():
        if decode_manifest.get(name) != expected:
            raise ValueError(f"decode manifest has invalid frozen binding: {name}")
    if image_order_sha256(image_ids) != plan.raw["image_order_sha256"]:
        raise ValueError("decode manifest image order differs from the selection plan")
    expected_candidates = {
        "screening": plan.raw["screening_candidates"],
        "final": plan.raw["final_candidates"],
    }
    stage = decode_manifest.get("selection_stage")
    if stage not in expected_candidates or int(decode_manifest["candidate_count"]) != int(
        expected_candidates[stage]
    ):
        raise ValueError("decode candidate count differs from its selection stage")
    batch_size = int(plan.raw["evaluation_batch_size"])
    by_metric: dict[str, list[np.ndarray]] = {"PixCorr": [], "SSIM": []}
    for candidate in reconstructed:
        pixcorr, ssim = pixel_metrics(originals, candidate)
        by_metric["PixCorr"].append(pixcorr)
        by_metric["SSIM"].append(ssim)

    alex = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        return_nodes={"features.4": "layer2", "features.11": "layer5"},
    ).to(device).eval().requires_grad_(False)
    alex_transform = preprocess_transform(256, IMAGENET_MEAN, IMAGENET_STD)
    for key, metric_name in (("layer2", "AlexNet-2"), ("layer5", "AlexNet-5")):
        original, candidates = model_features(
            originals,
            reconstructed,
            alex,
            alex_transform,
            device=device,
            batch_size=batch_size,
            output_key=key,
        )
        by_metric[metric_name] = identification_by_seed(original, candidates)
    del alex
    torch.cuda.empty_cache()

    inception = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT),
        return_nodes={"avgpool": "avgpool"},
    ).to(device).eval().requires_grad_(False)
    original, candidates = model_features(
        originals,
        reconstructed,
        inception,
        preprocess_transform(342, IMAGENET_MEAN, IMAGENET_STD),
        device=device,
        batch_size=batch_size,
        output_key="avgpool",
    )
    by_metric["Inception"] = identification_by_seed(original, candidates)
    del inception
    torch.cuda.empty_cache()

    clip_source = project_root / "repo/vendor/CLIP"
    sys.path.insert(0, str(clip_source))
    import clip  # type: ignore

    clip_model, _ = clip.load(str(assets["clip_vit_l_14"]), device=device, jit=False)
    original, candidates = model_features(
        originals,
        reconstructed,
        clip_model.encode_image,
        preprocess_transform(224, CLIP_MEAN, CLIP_STD),
        device=device,
        batch_size=batch_size,
    )
    by_metric["CLIP"] = identification_by_seed(original, candidates)
    del clip_model
    torch.cuda.empty_cache()

    efficient = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
        return_nodes={"avgpool": "avgpool"},
    ).to(device).eval().requires_grad_(False)
    original, candidates = model_features(
        originals,
        reconstructed,
        efficient,
        preprocess_transform(255, IMAGENET_MEAN, IMAGENET_STD),
        device=device,
        batch_size=batch_size,
        output_key="avgpool",
    )
    by_metric["EffCorrDistance"] = [
        paired_correlation_distance(original, value) for value in candidates
    ]
    del efficient
    torch.cuda.empty_cache()

    swav = torch.hub.load(
        str(project_root / "repo/vendor/swav"),
        "resnet50",
        source="local",
        pretrained=True,
    )
    swav = create_feature_extractor(swav, return_nodes={"avgpool": "avgpool"})
    swav.to(device).eval().requires_grad_(False)
    original, candidates = model_features(
        originals,
        reconstructed,
        swav,
        preprocess_transform(224, IMAGENET_MEAN, IMAGENET_STD),
        device=device,
        batch_size=batch_size,
        output_key="avgpool",
    )
    by_metric["SwAVCorrDistance"] = [
        paired_correlation_distance(original, value) for value in candidates
    ]

    seed_mean = {
        name: np.stack(values, axis=0).mean(axis=0) for name, values in by_metric.items()
    }
    rows = []
    for index, image_id in enumerate(image_ids):
        row = {"image_id": image_id}
        row.update({name: float(values[index]) for name, values in seed_mean.items()})
        row["SemanticScore"] = float(
            np.mean([row["AlexNet-5"], row["Inception"], row["CLIP"]])
        )
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    validation_loss = load_json_mapping(args.validation_loss)
    if int(validation_loss["image_count"]) != 500:
        raise ValueError("validation loss does not cover the fixed 500-image pool")
    shared_fields = (
        "optimizer_update",
        "config_sha256",
        "method_fingerprint",
        "selection_plan_sha256",
        "validation_ids_sha256",
        "image_order_sha256",
        "metric_implementation_sha256",
        "protocol_namespace",
        "repository_commit",
        "snapshot_model_sha256",
        "snapshot_manifest_sha256",
        "snapshot_metadata_sha256",
        "formal_approval_sha256",
    )
    for name in shared_fields:
        if validation_loss.get(name) != decode_manifest.get(name):
            raise ValueError(f"validation loss and decode differ in {name}")
    record = {
        "optimizer_update": int(decode_manifest["optimizer_update"]),
        "validation_loss": float(validation_loss["mean_loss"]),
        "metrics": {
            name: float(values.mean()) for name, values in seed_mean.items()
        },
        "per_image_semantic_score": [row["SemanticScore"] for row in rows],
        "snapshot_model_sha256": decode_manifest["snapshot_model_sha256"],
        "snapshot_manifest_sha256": decode_manifest["snapshot_manifest_sha256"],
        "snapshot_metadata_sha256": decode_manifest["snapshot_metadata_sha256"],
        "run_mode": "formal",
        "run_kind": "selection",
        "training_config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        "formal_approval_sha256": decode_manifest["formal_approval_sha256"],
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "metric_units": {
            "identification": "percent",
            "PixCorr": "correlation",
            "SSIM": "index",
            "EffCorrDistance": "correlation_distance_lower_is_better",
            "SwAVCorrDistance": "correlation_distance_lower_is_better",
        },
        "image_count": 500,
        "candidate_count": int(decode_manifest["candidate_count"]),
        "negative_pool": "the same 500 unique validation image IDs for every candidate seed",
        "candidate_aggregation": "per-image arithmetic mean after fixed-pool scoring",
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        **plan_binding,
        "protocol_namespace": plan.raw["protocol_namespace"],
        "selection_stage": stage,
        "denoising_steps": plan.raw["denoising_steps"],
        "guidance_scale": plan.raw["guidance_scale"],
        "evaluation_batch_size": batch_size,
        "repository_commit": config.raw["protocol_commit"],
        "decode_manifest_sha256": sha256_file(args.decode_manifest),
        "evaluation_manifest_sha256": sha256_file(config.paths["evaluation_manifest"]),
        "per_image_csv_sha256": sha256_file(args.output_csv),
        "checkpoints": [record],
    }
    write_json_atomic(args.output_json, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
