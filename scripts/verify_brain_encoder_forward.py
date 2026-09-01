#!/usr/bin/env python3
"""Instantiate all 16 brain encoders, run one image, and verify weighted ensembles."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import subprocess
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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--parcel-audit", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    project_root = args.project_root.resolve()
    asset_manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    parcel_audit = json.loads(args.parcel_audit.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")
    )
    if (
        asset_manifest.get("status") != "verified"
        or parcel_audit.get("gate") != "brain_encoder_parcel"
        or parcel_audit.get("status") != "verified"
    ):
        raise ValueError("brain encoder asset and parcel audits must both be verified")
    sources = source_manifest["sources"]
    observed_wbe = subprocess.check_output(
        ["git", "-C", str(repository / "vendor/whole_brain_encoder"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    observed_dino = subprocess.check_output(
        ["git", "-C", str(repository / "vendor/dinov2"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_wbe != sources["whole_brain_encoder"]["commit"]:
        raise ValueError("whole_brain_encoder commit differs from the source manifest")
    if observed_dino != sources["dinov2"]["commit"]:
        raise ValueError("DINOv2 commit differs from the source manifest")
    dino_record = evaluation_manifest["files"]["dinov2_vitb14"]
    dino_weight = project_root / dino_record["path"]
    if (
        dino_weight.stat().st_size != int(dino_record["size"])
        or sha256_file(dino_weight) != dino_record["sha256"]
    ):
        raise ValueError("DINOv2 weight differs from the evaluation manifest")
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
    static_entries = {
        (entry["encoder_layer"], entry["run"], entry["hemisphere"]): entry
        for entry in asset_manifest["entries"]
    }
    with redirect_dinov2_hub(repository / "vendor/dinov2"):
        for hemisphere in HEMISPHERES:
            parcels = torch.load(
                args.parcel_dir / f"{hemisphere}_labels_s01.pt",
                map_location="cpu",
                weights_only=True,
            )
            parcel_file_sha256 = sha256_file(
                args.parcel_dir / f"{hemisphere}_labels_s01.pt"
            )
            parcel_record = parcel_audit["parcels"][hemisphere]
            if parcel_file_sha256 != parcel_record["runtime_file_sha256"]:
                raise ValueError("runtime parcel file differs from the parcel audit")
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
                    if (
                        str(getattr(checkpoint_args, "parcel_dir", ""))
                        != parcel_audit["checkpoint_parcel_dir"]
                    ):
                        raise ValueError("checkpoint parcel_dir differs from the parcel audit")
                    static_entry = static_entries[(layer, run, hemisphere)]
                    if sha256_file(checkpoint_path) != static_entry["checkpoint_sha256"]:
                        raise ValueError("checkpoint differs from the static asset manifest")
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
                    parcel_mask_sha256 = tensor_sha256(model.parcel_mask.float().cpu())
                    if (
                        parcel_mask_sha256
                        != parcel_record["canonical_float32_parcel_mask_sha256"]
                    ):
                        raise ValueError("instantiated parcel_mask differs from the parcel audit")
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
                            "checkpoint_parcel_dir": str(checkpoint_args.parcel_dir),
                            "query_embed_shape": list(model.query_embed.weight.shape),
                            "parcel_file_sha256": parcel_file_sha256,
                            "parcel_mask_sha256": parcel_mask_sha256,
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
        "gate": "brain_encoder_forward",
        "status": "passed",
        "subject": 1,
        "input": "fixed synthetic 425x425 RGB tensor",
        "member_count": len(entries),
        "state_dict_loading_verified": True,
        "full_forward_verified": True,
        "confidence_weighted_ensemble_verified": True,
        "brain_encoder_asset_manifest_sha256": sha256_file(args.asset_manifest),
        "brain_encoder_parcel_audit_sha256": sha256_file(args.parcel_audit),
        "parcel_files_sha256": {
            hemi: parcel_audit["parcels"][hemi]["runtime_file_sha256"]
            for hemi in HEMISPHERES
        },
        "whole_brain_encoder_commit": observed_wbe,
        "dinov2_commit": observed_dino,
        "dinov2_weight_sha256": sha256_file(dino_weight),
        "environment_lock_sha256": sha256_file(args.environment_lock),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "entries": entries,
        "ensemble": ensemble,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
