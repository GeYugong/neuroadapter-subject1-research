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


def nonfinite_scan(
    dataset: h5py.Dataset, selected_vertex_mask: np.ndarray, chunk_size: int = 16
) -> dict[str, object]:
    nonfinite_count = 0
    nan_count = 0
    positive_inf_count = 0
    negative_inf_count = 0
    affected_rows: set[int] = set()
    affected_vertices: set[int] = set()
    for start in range(0, dataset.shape[0], chunk_size):
        array = np.asarray(dataset[start : start + chunk_size])
        invalid = ~np.isfinite(array)
        if invalid[:, selected_vertex_mask].any():
            raise ValueError(
                f"{dataset.name}: selected top-SNR vertices contain NaN/Inf in "
                f"rows {start}:{start + len(array)}"
            )
        if invalid.any():
            row_indices, vertex_indices = np.nonzero(invalid)
            nonfinite_count += int(invalid.sum())
            nan_count += int(np.isnan(array).sum())
            positive_inf_count += int(np.isposinf(array).sum())
            negative_inf_count += int(np.isneginf(array).sum())
            affected_rows.update(int(start + value) for value in row_indices)
            affected_vertices.update(int(value) for value in vertex_indices)
    return {
        "nonfinite_count": nonfinite_count,
        "nan_count": nan_count,
        "positive_inf_count": positive_inf_count,
        "negative_inf_count": negative_inf_count,
        "affected_row_count": len(affected_rows),
        "affected_vertex_count": len(affected_vertices),
        "affected_rows_zero_based": sorted(affected_rows),
        "affected_vertices_zero_based": sorted(affected_vertices),
    }


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

    parcel_rows = []
    selected: dict[str, list[int]] = {}
    selected_vertex_masks: dict[str, np.ndarray] = {}
    max_voxels = 0
    for hemi in HEMISPHERES:
        parcels = torch.load(args.parcel_dir / f"{hemi}_labels_s01.pt", weights_only=True)
        if len(parcels) != 501:
            raise ValueError(f"{hemi}: expected 501 labels including medial wall")
        ncsnr = np.asarray(metadata[f"{hemi}_ncsnr"], dtype=np.float32)
        scores = np.asarray([float(ncsnr[vertices.numpy()].mean()) for vertices in parcels[1:]])
        top_indices = np.argsort(scores)[::-1][:100]
        selected[hemi] = [int(value) for value in top_indices]
        selected_vertex_mask = np.zeros(VERTICES, dtype=np.bool_)
        for rank, parcel_index in enumerate(top_indices, start=1):
            vertices = parcels[int(parcel_index) + 1].numpy().astype("<i8", copy=False)
            selected_vertex_mask[vertices] = True
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
        selected_vertex_masks[hemi] = selected_vertex_mask

    beta_summary: dict[str, object] = {}
    with h5py.File(betas_path, "r") as h5:
        for hemi in HEMISPHERES:
            dataset = h5[f"{hemi}_betas"]
            if dataset.shape != (TRIALS, VERTICES) or dataset.dtype != np.dtype("float32"):
                raise ValueError(f"invalid {hemi} beta dataset: {dataset.shape}, {dataset.dtype}")
            sample_rows = np.unique(np.linspace(0, TRIALS - 1, 64, dtype=np.int64))
            sample = np.asarray(dataset[sample_rows])
            selected_sample = sample[:, selected_vertex_masks[hemi]]
            if not np.isfinite(selected_sample).all():
                raise ValueError(f"{hemi}: sampled selected beta data contains NaN/Inf")
            nonfinite_summary = None
            if args.full_scan:
                nonfinite_summary = nonfinite_scan(dataset, selected_vertex_masks[hemi])
            beta_summary[hemi] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "selected_vertex_count": int(selected_vertex_masks[hemi].sum()),
                "selected_sample_min": float(selected_sample.min()),
                "selected_sample_max": float(selected_sample.max()),
                "selected_sample_mean": float(selected_sample.mean(dtype=np.float64)),
                "selected_sample_std": float(selected_sample.std(dtype=np.float64)),
                "full_nonfinite_scan": args.full_scan,
                "full_nonfinite_summary": nonfinite_summary,
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
