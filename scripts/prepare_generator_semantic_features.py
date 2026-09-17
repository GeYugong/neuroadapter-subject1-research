"""Precompute train-only CLIP residual targets for the semantic training arm."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generator_semantic_core import (base_config, clip_asset, clip_preprocess_tensor,
                                     output, residual_features, source_identity)
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.data import Subject1TrainingDataset


@torch.no_grad()
def main(root: Path) -> None:
    started = time.time()
    cfg = base_config(root)
    out = output(root) / "semantic-features"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "train8500-clip-vit-l14.pt"
    if target.exists():
        payload = torch.load(target, map_location="cpu", weights_only=True)
        assert payload["image_ids"].shape == (8500,)
        assert payload["unit_features"].shape == (8500, 768)
        print(json.dumps({"status": "reused", "sha256": sha256_file(target)}))
        return
    sys.path.insert(0, str(root / "repo/vendor/CLIP"))
    import clip
    device = torch.device("cuda:0")
    model, _ = clip.load(str(clip_asset(root)), device=device, jit=False)
    model.eval().requires_grad_(False)
    ds = Subject1TrainingDataset(cfg.paths["training_cache"], cfg.paths["stimuli"], cfg.paths["split_ids"])
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    features, ids = [], []
    for index, batch in enumerate(loader):
        rgb = ((batch["image"].to(device, non_blocking=True) + 1.0) / 2.0).clamp(0, 1)
        unit = F.normalize(model.encode_image(clip_preprocess_tensor(rgb)).float(), dim=-1, eps=1e-6)
        features.append(unit.cpu()); ids.append(batch["nsd_image_id"].cpu())
        if (index + 1) % 100 == 0:
            print(f"semantic features: {min((index+1)*16, 8500)}/8500", flush=True)
    unit_features = torch.cat(features)
    image_ids = torch.cat(ids)
    mean_feature = unit_features.mean(dim=0)
    residual = residual_features(unit_features, mean_feature)
    assert image_ids.tolist() == [int(value) for value in ds.image_ids]
    assert unit_features.shape == residual.shape == (8500, 768)
    assert torch.isfinite(unit_features).all() and torch.isfinite(residual).all()
    torch.save({"image_ids": image_ids, "unit_features": unit_features,
                "mean_feature": mean_feature, "residual_features": residual}, target)
    write_json_atomic(out / "manifest.json", {
        "status": "complete", "image_count": 8500, "feature_dim": 768,
        "clip": "ViT-L/14", "clip_sha256": sha256_file(clip_asset(root)),
        "cache_sha256": sha256_file(target), "source": source_identity(root),
        "split_ids_sha256": sha256_file(cfg.paths["split_ids"]),
        "preprocess": "deterministic CPU tensor bicubic 224 antialias + frozen CLIP normalization",
        "seconds": time.time() - started,
    })
    ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
