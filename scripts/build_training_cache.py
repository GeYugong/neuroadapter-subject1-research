#!/usr/bin/env python3
"""Build a compact 9000-image Subject 1 cache for formal training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from neuroadapter_research.atomic import fsync_directory, sha256_file, write_json_atomic


HEMISPHERES = ("lh", "rh")
VERTICES = 163842


def selected_parcels(
    metadata: dict[str, np.ndarray], parcel_dir: Path
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[int]], int]:
    selected_vertices: dict[str, list[torch.Tensor]] = {}
    selected_indices: dict[str, list[int]] = {}
    maximum = 0
    for hemi in HEMISPHERES:
        parcels = torch.load(parcel_dir / f"{hemi}_labels_s01.pt", weights_only=True)
        if len(parcels) != 501:
            raise ValueError(f"{hemi}: expected 501 labels including medial wall")
        usable = parcels[1:]
        ncsnr = np.asarray(metadata[f"{hemi}_ncsnr"], dtype=np.float32)
        scores = np.asarray([ncsnr[value.numpy()].mean() for value in usable])
        indices = np.argsort(scores)[::-1][:100]
        selected_indices[hemi] = [int(value) for value in indices]
        selected_vertices[hemi] = [usable[int(value)] for value in indices]
        maximum = max(maximum, *(int(value.numel()) for value in selected_vertices[hemi]))
    return selected_vertices, selected_indices, maximum


def pad_parcels(
    average: np.ndarray, parcels: list[torch.Tensor], maximum: int
) -> np.ndarray:
    output = np.zeros((len(parcels), maximum), dtype=np.float32)
    for parcel_index, vertices in enumerate(parcels):
        vertex_indices = vertices.numpy()
        output[parcel_index, : vertex_indices.size] = average[vertex_indices]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--betas", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to rebuild")
    metadata = np.load(args.metadata, allow_pickle=True).item()
    train_ids = np.sort(np.asarray(metadata["train_img_num"], dtype=np.int64))
    test_ids = np.asarray(metadata["test_img_num"], dtype=np.int64)
    presentation = np.asarray(metadata["img_presentation_order"], dtype=np.int64)
    if train_ids.shape != (9000,) or np.unique(train_ids).size != 9000:
        raise ValueError("expected 9000 unique train-pool image IDs")
    if test_ids.shape != (1000,) or np.intersect1d(train_ids, test_ids).size:
        raise ValueError("invalid standard test partition")

    parcels, parcel_indices, maximum = selected_parcels(metadata, args.parcel_dir)
    trial_rows = {int(image_id): np.flatnonzero(presentation == image_id) for image_id in train_ids}
    if any(value.shape != (3,) for value in trial_rows.values()):
        raise ValueError("each train-pool image must have exactly three presentations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(args.betas, "r") as source, h5py.File(temporary, "w", libver="latest") as target:
        target.attrs["schema_version"] = 1
        target.attrs["cache_kind"] = "subject01_train_pool_top100_per_hemisphere"
        target.attrs["subject"] = 1
        target.attrs["repeat_reduction"] = "float32 arithmetic mean of three presentations"
        target.attrs["parcel_order"] = "lh SNR rank 1..100, then rh SNR rank 1..100"
        target.create_dataset("image_ids", data=train_ids, dtype="int64")
        mask = np.zeros((200, maximum), dtype=np.bool_)
        for offset, hemi in enumerate(HEMISPHERES):
            for parcel_index, vertices in enumerate(parcels[hemi]):
                mask[offset * 100 + parcel_index, : vertices.numel()] = True
        target.create_dataset("parcel_valid_mask", data=mask, dtype="bool")
        brain = target.create_dataset(
            "brain",
            shape=(9000, 200, maximum),
            dtype="float32",
            chunks=(1, 200, maximum),
            compression="lzf",
        )
        for output_row, image_id in enumerate(tqdm(train_ids, desc="train-pool images")):
            rows = trial_rows[int(image_id)]
            hemispheres = []
            for hemi in HEMISPHERES:
                repetitions = np.asarray(source[f"{hemi}_betas"][rows], dtype=np.float32)
                if repetitions.shape != (3, VERTICES) or not np.isfinite(repetitions).all():
                    raise ValueError(f"invalid {hemi} repetitions for image {image_id}")
                average = repetitions.mean(axis=0, dtype=np.float32)
                hemispheres.append(pad_parcels(average, parcels[hemi], maximum))
            value = np.concatenate(hemispheres, axis=0)
            if value.shape != (200, maximum) or not np.isfinite(value).all():
                raise ValueError(f"invalid cache row for image {image_id}")
            brain[output_row] = value
        target.flush()
    os.replace(temporary, args.output)
    fsync_directory(args.output.parent)

    payload = {
        "schema_version": 1,
        "subject": 1,
        "cache_kind": "subject01_train_pool_top100_per_hemisphere",
        "image_count": 9000,
        "parcel_count": 200,
        "max_voxels": maximum,
        "selected_parcel_indices_excluding_medial_wall": parcel_indices,
        "metadata_sha256": sha256_file(args.metadata),
        "betas_sha256": sha256_file(args.betas),
        "schaefer_summary_sha256": sha256_file(args.parcel_dir / "schaefer_summary.json"),
        "cache_size": args.output.stat().st_size,
        "cache_sha256": sha256_file(args.output),
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
