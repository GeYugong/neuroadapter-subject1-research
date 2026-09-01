#!/usr/bin/env python3
"""Instantiate all 16 brain encoders, run one image, and verify weighted ensembles."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
from pathlib import Path, PosixPath

import numpy as np
import torch
from scipy.special import softmax

from neuroadapter_research.atomic import sha256_file, write_json_atomic


LAYERS = (1, 3, 5, 7)
RUNS = (1, 2)
HEMISPHERES = ("lh", "rh")
VERTICES = 163842


class SyntheticParcelDataset:
    def __init__(self, parcels: list[torch.Tensor]) -> None:
        self.num_parcels = len(parcels)
        self.num_hemi_voxels = VERTICES
        self.valid_voxel_mask = torch.ones(VERTICES, dtype=torch.bool)
        self.parcels = parcels


@contextlib.contextmanager
def redirect_dinov2_hub(vendor_dinov2: Path):
    original = torch.hub.load

    def mapped(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).endswith("facebookresearch_dinov2_main"):
            kwargs["source"] = "local"
            return original(str(vendor_dinov2), model, *args, **kwargs)
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = mapped
    try:
        yield
    finally:
        torch.hub.load = original


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    torch.hub.set_dir(str(args.torch_home.resolve() / "hub"))
    wbe = repository / "vendor/whole_brain_encoder"
    sys.path.insert(0, str(wbe))
    torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])
    from models.brain_encoder import brain_encoder

    device = torch.device(args.device)
    torch.manual_seed(20260901)
    image = torch.linspace(-1, 1, 3 * 425 * 425, dtype=torch.float32).reshape(
        1, 3, 425, 425
    )
    predictions: dict[str, list[np.ndarray]] = {hemi: [] for hemi in HEMISPHERES}
    correlations: dict[str, list[np.ndarray]] = {hemi: [] for hemi in HEMISPHERES}
    entries = []
    with redirect_dinov2_hub(repository / "vendor/dinov2"):
        for hemisphere in HEMISPHERES:
            parcels = torch.load(
                args.parcel_dir / f"{hemisphere}_labels_s01.pt",
                map_location="cpu",
                weights_only=True,
            )
            dataset = SyntheticParcelDataset(parcels)
            for run in RUNS:
                for layer in LAYERS:
                    model_dir = (
                        args.asset_root
                        / f"enc_{layer}"
                        / f"run_{run}"
                        / hemisphere
                    )
                    checkpoint_path = model_dir / "checkpoint_nonavg.pth"
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                    checkpoint_args = checkpoint["args"]
                    checkpoint_args.device = device
                    model = brain_encoder(checkpoint_args, dataset)
                    state = {
                        name.replace("_orig_mod.", ""): value
                        for name, value in checkpoint["model"].items()
                    }
                    incompatibility = model.load_state_dict(state, strict=False)
                    unexpected = list(incompatibility.unexpected_keys)
                    forbidden_missing = [
                        name
                        for name in incompatibility.missing_keys
                        if not name.startswith("backbone_model.")
                    ]
                    if unexpected or forbidden_missing:
                        raise ValueError(
                            f"state load mismatch for {checkpoint_path}: "
                            f"unexpected={unexpected}, missing={forbidden_missing}"
                        )
                    model.to(device).eval()
                    with torch.no_grad():
                        prediction = model(image.to(device))["pred"].float().cpu()
                    if prediction.shape != (1, VERTICES) or not torch.isfinite(prediction).all():
                        raise ValueError(f"invalid forward output: {checkpoint_path}")
                    correlation = np.nan_to_num(
                        np.load(model_dir / f"{hemisphere}_val_corr_nonavg.npy")
                    )
                    predictions[hemisphere].append(prediction.numpy()[0])
                    correlations[hemisphere].append(correlation)
                    entries.append(
                        {
                            "encoder_layer": layer,
                            "run": run,
                            "hemisphere": hemisphere,
                            "checkpoint_sha256": sha256_file(checkpoint_path),
                            "prediction_sha256": tensor_sha256(prediction),
                            "missing_frozen_backbone_key_count": len(
                                incompatibility.missing_keys
                            ),
                        }
                    )
                    del model, checkpoint, prediction
                    gc.collect()
                    torch.cuda.empty_cache()

    ensemble = {}
    for hemisphere in HEMISPHERES:
        prediction = np.stack(predictions[hemisphere])
        weights = softmax(20 * np.stack(correlations[hemisphere]), axis=0)
        weighted = (weights * prediction).sum(axis=0)
        if weighted.shape != (VERTICES,) or not np.isfinite(weighted).all():
            raise ValueError(f"invalid weighted ensemble: {hemisphere}")
        ensemble[hemisphere] = {
            "member_count": 8,
            "weights_sum_max_abs_error": float(np.abs(weights.sum(axis=0) - 1).max()),
            "prediction_sha256": hashlib.sha256(
                weighted.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
        }
    payload = {
        "schema_version": 1,
        "status": "passed",
        "subject": 1,
        "input": "fixed synthetic 425x425 RGB tensor",
        "member_count": len(entries),
        "state_dict_loading_verified": True,
        "full_forward_verified": True,
        "confidence_weighted_ensemble_verified": True,
        "entries": entries,
        "ensemble": ensemble,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
