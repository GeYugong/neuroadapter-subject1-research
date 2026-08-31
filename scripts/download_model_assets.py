#!/usr/bin/env python3
"""Download frozen Stable Diffusion and Subject 1 brain-encoder assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


SD_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
BRAIN_REPOSITORY = "ehwang/brain_encoder_weights"
BRAIN_REVISION = "d8a978abb212eb2965b5d01673f96536b77e2ea0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/matengyu/geyugong/neuroadapter-subject1-research"),
    )
    parser.add_argument("--asset", choices=("all", "sd15", "brain_encoder"), default="all")
    args = parser.parse_args()

    root = args.project_root.resolve()
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    records: dict[str, object] = {}

    if args.asset in ("all", "sd15"):
        target = root / "models" / "stable-diffusion-v1-5"
        path = snapshot_download(
            repo_id=SD_REPOSITORY,
            revision=SD_REVISION,
            local_dir=target,
            allow_patterns=[
                "README.md",
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        )
        records["stable_diffusion_v1_5"] = {
            "repository": SD_REPOSITORY,
            "revision": SD_REVISION,
            "path": str(path),
        }

    if args.asset in ("all", "brain_encoder"):
        target = root / "models" / "brain-encoder"
        path = snapshot_download(
            repo_id=BRAIN_REPOSITORY,
            revision=BRAIN_REVISION,
            local_dir=target,
            allow_patterns=["dinov2_q_transformer/schaefer/subj_01/**"],
        )
        records["brain_encoder"] = {
            "repository": BRAIN_REPOSITORY,
            "revision": BRAIN_REVISION,
            "path": str(path),
        }

    output = root / "data" / "fingerprints" / "model_downloads.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
