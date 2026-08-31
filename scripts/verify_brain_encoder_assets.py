#!/usr/bin/env python3
"""Verify the fixed Subject 1 whole-brain encoder ensemble assets."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path, PosixPath

import numpy as np
import torch

from neuroadapter_research.atomic import sha256_file, write_json_atomic


LAYERS = (1, 3, 5, 7)
RUNS = (1, 2)
HEMISPHERES = ("lh", "rh")
FSAVERAGE_VERTEX_COUNT = 163842


def verify_model_pair(
    checkpoint_path: Path,
    correlation_path: Path,
    expected_vertices: int = FSAVERAGE_VERTEX_COUNT,
) -> dict[str, object]:
    if not checkpoint_path.is_file() or not correlation_path.is_file():
        raise FileNotFoundError(
            f"missing brain encoder asset: {checkpoint_path} or {correlation_path}"
        )

    torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError(f"unexpected checkpoint structure: {checkpoint_path}")
    model = checkpoint["model"]
    if not isinstance(model, dict) or not model:
        raise ValueError(f"empty model state: {checkpoint_path}")
    if any(".orig_mod" in name for name in model):
        raise ValueError(f"checkpoint contains unsupported .orig_mod keys: {checkpoint_path}")

    tensor_count = 0
    value_count = 0
    for name, value in model.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model state value is not a tensor: {name}")
        tensor_count += 1
        value_count += value.numel()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"model state contains NaN/Inf: {checkpoint_path}:{name}")

    correlation = np.load(correlation_path, allow_pickle=False)
    if correlation.shape != (expected_vertices,):
        raise ValueError(
            f"unexpected voxel-confidence shape {correlation.shape}: {correlation_path}"
        )
    if not np.issubdtype(correlation.dtype, np.floating):
        raise ValueError(f"voxel-confidence array must be floating point: {correlation_path}")
    if np.isinf(correlation).any():
        raise ValueError(f"voxel-confidence array contains Inf: {correlation_path}")
    finite = correlation[np.isfinite(correlation)]
    if not finite.size:
        raise ValueError(f"voxel-confidence array has no finite values: {correlation_path}")

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_tensor_count": tensor_count,
        "model_value_count": value_count,
        "correlation_path": str(correlation_path),
        "correlation_size": correlation_path.stat().st_size,
        "correlation_sha256": sha256_file(correlation_path),
        "correlation_shape": list(correlation.shape),
        "correlation_dtype": str(correlation.dtype),
        "correlation_nonfinite_count": int(correlation.size - finite.size),
        "correlation_minimum": float(finite.min()),
        "correlation_maximum": float(finite.max()),
        "correlation_mean": float(finite.astype(np.float64).mean()),
    }
    del checkpoint, model, correlation, finite
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for layer in LAYERS:
        for run in RUNS:
            for hemisphere in HEMISPHERES:
                model_dir = (
                    args.root
                    / f"enc_{layer}"
                    / f"run_{run}"
                    / hemisphere
                )
                entry = verify_model_pair(
                    model_dir / "checkpoint_nonavg.pth",
                    model_dir / f"{hemisphere}_val_corr_nonavg.npy",
                )
                entry.update(
                    {"encoder_layer": layer, "run": run, "hemisphere": hemisphere}
                )
                entries.append(entry)

    if len(entries) != 16:
        raise AssertionError(f"expected 16 ensemble members, found {len(entries)}")
    payload = {
        "schema_version": 1,
        "status": "verified",
        "subject": 1,
        "backbone": "dinov2_q_transformer",
        "parcel_strategy": "schaefer",
        "layers": list(LAYERS),
        "runs": list(RUNS),
        "hemispheres": list(HEMISPHERES),
        "ensemble_member_count": len(entries),
        "entries": entries,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
