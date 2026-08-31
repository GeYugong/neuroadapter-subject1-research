#!/usr/bin/env python3
"""Verify converted Subject 1 data and emit parcel/data fingerprints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


HEMISPHERES = ("lh", "rh")
TRIALS = 30000
VERTICES = 163842


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_scan(dataset: h5py.Dataset, chunk_size: int = 16) -> None:
    for start in range(0, dataset.shape[0], chunk_size):
        array = np.asarray(dataset[start : start + chunk_size])
        if not np.isfinite(array).all():
            raise ValueError(f"{dataset.name}: NaN/Inf in rows {start}:{start + len(array)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-nsd", type=Path, required=True)
    parser.add_argument("--neural-data", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-scan", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.neural_data / "metadata_sub-01.npy"
    betas_path = args.neural_data / "betas_sub-01.h5"
    stimuli_path = args.raw_nsd / "stimuli" / "nsd_stimuli.hdf5"
    metadata = np.load(metadata_path, allow_pickle=True).item()

    presentation = np.asarray(metadata["img_presentation_order"], dtype=np.int64)
    unique_images, repeat_counts = np.unique(presentation, return_counts=True)
    train = np.asarray(metadata["train_img_num"], dtype=np.int64)
    test = np.asarray(metadata["test_img_num"], dtype=np.int64)
    if presentation.shape != (TRIALS,):
        raise ValueError(f"expected {TRIALS} presentations")
    if unique_images.size != 10000 or not np.all(repeat_counts == 3):
        raise ValueError("expected 10000 images with exactly three repeats")
    if train.size != 9000 or test.size != 1000 or np.intersect1d(train, test).size:
        raise ValueError("invalid 9000/1000 train/test split")

    beta_summary: dict[str, object] = {}
    with h5py.File(betas_path, "r") as h5:
        for hemi in HEMISPHERES:
            dataset = h5[f"{hemi}_betas"]
            if dataset.shape != (TRIALS, VERTICES) or dataset.dtype != np.dtype("float32"):
                raise ValueError(f"invalid {hemi} beta dataset: {dataset.shape}, {dataset.dtype}")
            sample_rows = np.unique(np.linspace(0, TRIALS - 1, 64, dtype=np.int64))
            sample = np.asarray(dataset[sample_rows])
            if not np.isfinite(sample).all():
                raise ValueError(f"{hemi}: sampled beta data contains NaN/Inf")
            if args.full_scan:
                finite_scan(dataset)
            beta_summary[hemi] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "sample_min": float(sample.min()),
                "sample_max": float(sample.max()),
                "sample_mean": float(sample.mean(dtype=np.float64)),
                "sample_std": float(sample.std(dtype=np.float64)),
                "full_finite_scan": args.full_scan,
            }

    with h5py.File(stimuli_path, "r") as h5:
        images = h5["imgBrick"]
        if images.shape[0] <= int(presentation.max()):
            raise ValueError("stimulus HDF5 does not cover all presented image IDs")
        sample_ids = np.random.Generator(np.random.PCG64(20260901)).choice(
            unique_images, size=100, replace=False
        )
        audit_rows = []
        for image_id in np.sort(sample_ids):
            trial_indices = np.where(presentation == image_id)[0]
            image = np.asarray(images[int(image_id)])
            audit_rows.append(
                {
                    "image_id": int(image_id),
                    "trial_indices": ";".join(str(int(value)) for value in trial_indices),
                    "sessions": ";".join(str(int(value // 750 + 1)) for value in trial_indices),
                    "trial_in_session": ";".join(str(int(value % 750 + 1)) for value in trial_indices),
                    "image_sha256": sha256_bytes(image.tobytes()),
                }
            )

    with (args.output_dir / "image_mapping_audit_100.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)

    parcel_rows = []
    selected: dict[str, list[int]] = {}
    max_voxels = 0
    for hemi in HEMISPHERES:
        parcels = torch.load(args.parcel_dir / f"{hemi}_labels_s01.pt", weights_only=True)
        if len(parcels) != 501:
            raise ValueError(f"{hemi}: expected 501 labels including medial wall")
        ncsnr = np.asarray(metadata[f"{hemi}_ncsnr"], dtype=np.float32)
        scores = np.asarray([float(ncsnr[vertices.numpy()].mean()) for vertices in parcels[1:]])
        top_indices = np.argsort(scores)[::-1][:100]
        selected[hemi] = [int(value) for value in top_indices]
        for rank, parcel_index in enumerate(top_indices, start=1):
            vertices = parcels[int(parcel_index) + 1].numpy().astype("<i8", copy=False)
            max_voxels = max(max_voxels, int(vertices.size))
            parcel_rows.append(
                {
                    "model_token": len(parcel_rows),
                    "hemisphere": hemi,
                    "parcel_index_zero_based_excluding_medial_wall": int(parcel_index),
                    "schaefer_label_id": int(parcel_index) + 1,
                    "snr_rank": rank,
                    "mean_ncsnr": float(scores[int(parcel_index)]),
                    "vertex_count": int(vertices.size),
                    "vertex_sha256": sha256_bytes(vertices.tobytes()),
                }
            )

    with (args.output_dir / "parcel_token_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=parcel_rows[0].keys())
        writer.writeheader()
        writer.writerows(parcel_rows)

    payload = {
        "schema_version": 1,
        "subject": 1,
        "metadata_sha256": sha256_file(metadata_path),
        "betas_size": betas_path.stat().st_size,
        "stimuli_size": stimuli_path.stat().st_size,
        "presentations": int(presentation.size),
        "unique_images": int(unique_images.size),
        "repeat_count_values": np.unique(repeat_counts).tolist(),
        "train_images": int(train.size),
        "test_images": int(test.size),
        "beta_summary": beta_summary,
        "selected_parcels": selected,
        "selected_parcel_count": len(parcel_rows),
        "max_voxels": max_voxels,
        "expected_max_voxels_reference": 626,
        "parcel_map_sha256": sha256_file(args.output_dir / "parcel_token_map.csv"),
        "image_mapping_audit_sha256": sha256_file(args.output_dir / "image_mapping_audit_100.csv"),
    }
    output = args.output_dir / "data_fingerprint.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

