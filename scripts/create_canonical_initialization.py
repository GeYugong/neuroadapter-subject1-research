#!/usr/bin/env python3
"""Create and verify the one canonical adapter initialization."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from neuroadapter_research.atomic import fsync_directory, sha256_file, write_json_atomic
from neuroadapter_research.modeling import (
    audit_trainable_parameters,
    build_adapter,
    load_frozen_backbone,
    load_trainable_state_dict,
    trainable_state_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--data-fingerprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("canonical initialization or manifest already exists")

    fingerprint = json.loads(args.data_fingerprint.read_text(encoding="utf-8"))
    num_parcels = int(fingerprint["selected_parcel_count"])
    max_voxels = int(fingerprint["max_voxels"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    backbone = load_frozen_backbone(args.model_path)
    bundle = build_adapter(backbone.unet, num_parcels, max_voxels)
    audit = audit_trainable_parameters(bundle)
    state = trainable_state_dict(bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    fsync_directory(args.output.parent)

    del bundle, backbone
    gc.collect()
    verification_backbone = load_frozen_backbone(args.model_path)
    verification_bundle = build_adapter(
        verification_backbone.unet, num_parcels, max_voxels
    )
    loaded = torch.load(args.output, map_location="cpu", weights_only=True)
    load_trainable_state_dict(verification_bundle, loaded)
    reloaded = trainable_state_dict(verification_bundle)
    for group in state:
        for name in state[group]:
            torch.testing.assert_close(
                reloaded[group][name], state[group][name], rtol=0, atol=0
            )

    payload = {
        "schema_version": 1,
        "subject": 1,
        "architecture": "linear_projection",
        "seed": args.seed,
        "num_parcels": num_parcels,
        "max_voxels": max_voxels,
        "condition_dim": 768,
        "initialization_sha256": sha256_file(args.output),
        "initialization_size": args.output.stat().st_size,
        "data_fingerprint_sha256": sha256_file(args.data_fingerprint),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "parameter_audit": audit,
        "reload_bitwise_equal": True,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
