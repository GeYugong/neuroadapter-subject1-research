"""Frozen identities and shared math for the final bounded generator experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "configs/experiments/generator_semantic_last_v1.yaml"
EXPERIMENT = "generator-semantic-last-v1"


def spec() -> dict:
    value = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert value["experiment_type"] == "generator_semantic_last_v1"
    assert value["pilot_updates"] == 5000
    assert value["maximum_semantic_updates"] == 20000
    return value


def output(root: Path) -> Path:
    return Path(root) / "runs/experiments" / EXPERIMENT


def base_config(root: Path):
    from neuroadapter_research.config import load_training_config
    return load_training_config(
        Path(root) / "configs/formal/subject01_selection_v2.yaml", require_frozen=True
    )


def source_snapshot(root: Path) -> Path:
    root = Path(root)
    lock = json.loads((root / "runs/selection/subject01-selection-4090-deterministic-v2/evaluation-20260910/RESEARCH_WEIGHT_LOCK.json").read_text())
    cfg = spec()
    assert lock["selected_update"] == cfg["source_update"]
    assert lock["model_sha256"] == cfg["source_sha256"]
    path = root / lock["snapshot_relative_path"]
    from neuroadapter_research.atomic import sha256_file
    assert sha256_file(path / "model.pt") == cfg["source_sha256"]
    return path


def source_identity(root: Path) -> dict:
    from neuroadapter_research.atomic import sha256_file
    snap = source_snapshot(root)
    return {
        "source_update": spec()["source_update"],
        "source_model": str(snap / "model.pt"),
        "source_sha256": sha256_file(snap / "model.pt"),
        "implementation_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "spec_sha256": sha256_file(SPEC_PATH),
    }


def clip_asset(root: Path) -> Path:
    root = Path(root)
    manifest = json.loads((root / "data/fingerprints/evaluation_downloads.json").read_text())
    item = manifest["files"]["clip_vit_l_14"]
    path = root / item["path"]
    from neuroadapter_research.atomic import sha256_file
    assert sha256_file(path) == item["sha256"]
    return path


def clip_preprocess_tensor(images: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent of the frozen evaluator's CLIP image transform."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected NCHW RGB tensor, got {tuple(images.shape)}")
    images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False,
                           antialias=True)
    mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (images - mean) / std


def residual_features(unit_features: torch.Tensor, mean_feature: torch.Tensor) -> torch.Tensor:
    return F.normalize(unit_features.float() - mean_feature.float(), dim=-1, eps=1e-6)


def semantic_loss(
    predicted_rgb: torch.Tensor,
    target_residual: torch.Tensor,
    clip_model: torch.nn.Module,
    mean_feature: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    unclamped = (predicted_rgb.float() + 1.0) / 2.0
    saturation = ((unclamped < 0.0) | (unclamped > 1.0)).float().mean()
    images = clip_preprocess_tensor(unclamped.clamp(0.0, 1.0))
    unit = F.normalize(clip_model.encode_image(images).float(), dim=-1, eps=1e-6)
    residual = residual_features(unit, mean_feature)
    loss = (1.0 - (residual * target_residual.float()).sum(dim=-1)).mean()
    return loss, {"clamp_saturation_fraction": float(saturation.detach())}


def state_hash(payload: object) -> str:
    buffer = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(buffer).hexdigest()

