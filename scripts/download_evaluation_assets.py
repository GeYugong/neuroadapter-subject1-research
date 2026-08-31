#!/usr/bin/env python3
"""Preload the frozen evaluation backbones without using a GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
)

from neuroadapter_research.atomic import sha256_file


CLIP_FILENAME = "ViT-L-14.pt"
SWAV_FILENAME = "swav_800ep_pretrain.pth.tar"
DINO_FILENAME = "dinov2_vitb14_pretrain.pth"


def checkpoint_filename(url: str) -> str:
    return Path(urlparse(url).path).name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/matengyu/geyugong/neuroadapter-subject1-research"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    torch_home = root / "models" / "evaluation" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    torch.hub.set_dir(str(torch_home / "hub"))

    checkpoint_root = torch_home / "hub" / "checkpoints"
    torchvision_weights = {
        "alexnet": AlexNet_Weights.IMAGENET1K_V1,
        "inception_v3": Inception_V3_Weights.DEFAULT,
        "efficientnet_b1": EfficientNet_B1_Weights.DEFAULT,
    }
    required = {
        name: checkpoint_root / checkpoint_filename(weights.url)
        for name, weights in torchvision_weights.items()
    }
    required.update(
        {
            "clip_vit_l_14": root / "models" / "evaluation" / "clip" / CLIP_FILENAME,
            "swav_resnet50": checkpoint_root / SWAV_FILENAME,
            "dinov2_vitb14": checkpoint_root / DINO_FILENAME,
        }
    )
    if args.verify_only:
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"evaluation assets are missing: {missing}")

    models = [
        alexnet(weights=torchvision_weights["alexnet"]),
        inception_v3(weights=torchvision_weights["inception_v3"]),
        efficientnet_b1(weights=torchvision_weights["efficientnet_b1"]),
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
        "schema_version": 1,
        "verify_only": args.verify_only,
        "torchvision": ["AlexNet_IMAGENET1K_V1", "InceptionV3_DEFAULT", "EfficientNetB1_DEFAULT"],
        "clip": "ViT-L/14",
        "swav_source": "repo/vendor/swav",
        "dinov2_source": "repo/vendor/dinov2",
        "torch_home": "models/evaluation/torch",
        "files": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
    }
    output = root / "data" / "fingerprints" / "evaluation_downloads.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
