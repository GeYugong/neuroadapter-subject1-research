#!/usr/bin/env python3
"""Preload the frozen evaluation backbones without using a GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/matengyu/geyugong/neuroadapter-subject1-research"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    torch_home = root / "models" / "evaluation" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    torch.hub.set_dir(str(torch_home / "hub"))

    models = [
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        inception_v3(weights=Inception_V3_Weights.DEFAULT),
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
    ]
    del models

    clip_source = root / "repo" / "vendor" / "CLIP"
    sys.path.insert(0, str(clip_source))
    import clip  # type: ignore

    clip_root = root / "models" / "evaluation" / "clip"
    clip_root.mkdir(parents=True, exist_ok=True)
    clip_model, _ = clip.load("ViT-L/14", device="cpu", download_root=str(clip_root))
    del clip_model

    swav_source = root / "repo" / "vendor" / "swav"
    swav_model = torch.hub.load(str(swav_source), "resnet50", source="local", pretrained=True)
    del swav_model

    dinov2_source = root / "repo" / "vendor" / "dinov2"
    dino_model = torch.hub.load(str(dinov2_source), "dinov2_vitb14", source="local", pretrained=True)
    del dino_model

    payload = {
        "torchvision": ["AlexNet_IMAGENET1K_V1", "InceptionV3_DEFAULT", "EfficientNetB1_DEFAULT"],
        "clip": "ViT-L/14",
        "swav_source": str(swav_source),
        "dinov2_source": str(dinov2_source),
        "torch_home": str(torch_home),
    }
    output = root / "data" / "fingerprints" / "evaluation_downloads.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

