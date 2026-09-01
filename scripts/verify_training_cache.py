#!/usr/bin/env python3
"""Independently verify the compact Subject 1 formal-training cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def load_ids(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if values.ndim != 1 or np.unique(values).size != values.size:
        raise ValueError(f"invalid unique image ID file: {path}")
    return values


def scan_brain_dataset(
    brain: h5py.Dataset, valid_mask: np.ndarray, chunk_size: int = 16
) -> dict[str, float | int]:
    if valid_mask.shape != brain.shape[1:]:
        raise ValueError(
            f"valid mask {valid_mask.shape} differs from brain tail {brain.shape[1:]}"
        )
    count = 0
    total = 0.0
    total_square = 0.0
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, brain.shape[0], chunk_size):
        values = np.asarray(brain[start : start + chunk_size], dtype=np.float32)
        selected = values[:, valid_mask]
        padding = values[:, ~valid_mask]
        if not np.isfinite(selected).all():
            raise ValueError(f"training cache contains NaN/Inf in rows {start}:{start + len(values)}")
        if np.any(padding != 0):
            raise ValueError(f"training cache padding is nonzero in rows {start}:{start + len(values)}")
        selected64 = selected.astype(np.float64, copy=False)
        count += int(selected64.size)
        total += float(selected64.sum())
        total_square += float(np.square(selected64).sum())
        minimum = min(minimum, float(selected64.min()))
        maximum = max(maximum, float(selected64.max()))
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    return {
        "valid_value_count": count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "std": variance**0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-fingerprint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--selection-train-ids", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data_fingerprint = json.loads(args.data_fingerprint.read_text(encoding="utf-8"))
    expected_max_voxels = int(data_fingerprint["max_voxels"])
    metadata = np.load(args.metadata, allow_pickle=True).item()
    train_pool = np.sort(np.asarray(metadata["train_img_num"], dtype=np.int64))
    test = np.asarray(metadata["test_img_num"], dtype=np.int64)
    selection_train = np.sort(load_ids(args.selection_train_ids))
    validation = np.sort(load_ids(args.validation_ids))
    if train_pool.shape != (9000,) or np.unique(train_pool).size != 9000:
        raise ValueError("metadata does not contain the expected 9000-image train pool")
    if test.shape != (1000,) or np.intersect1d(train_pool, test).size:
        raise ValueError("standard train/test image IDs are invalid")
    if selection_train.shape != (8500,) or validation.shape != (500,):
        raise ValueError("selection split must contain 8500/500 image IDs")
    if np.intersect1d(selection_train, validation).size:
        raise ValueError("selection_train and validation overlap")
    if not np.array_equal(np.sort(np.concatenate([selection_train, validation])), train_pool):
        raise ValueError("selection split does not exactly cover the train pool")

    with h5py.File(args.cache, "r") as h5:
        if str(h5.attrs.get("cache_kind", "")) != "subject01_train_pool_top100_per_hemisphere":
            raise ValueError("unexpected training cache kind")
        image_ids = np.asarray(h5["image_ids"], dtype=np.int64)
        valid_mask = np.asarray(h5["parcel_valid_mask"], dtype=np.bool_)
        brain = h5["brain"]
        if image_ids.shape != (9000,) or not np.array_equal(image_ids, train_pool):
            raise ValueError("training cache image IDs differ from the sorted train pool")
        if np.intersect1d(image_ids, test).size:
            raise ValueError("standard test image IDs appear in the training cache")
        if brain.shape != (9000, 200, expected_max_voxels) or brain.dtype != np.dtype("float32"):
            raise ValueError(f"unexpected training cache brain dataset: {brain.shape}, {brain.dtype}")
        if valid_mask.shape != (200, expected_max_voxels) or valid_mask.dtype != np.dtype("bool"):
            raise ValueError(f"unexpected parcel valid mask: {valid_mask.shape}, {valid_mask.dtype}")
        statistics = scan_brain_dataset(brain, valid_mask)

    cache_sha256 = sha256_file(args.cache)
    if manifest["cache_sha256"] != cache_sha256:
        raise ValueError("training cache SHA-256 differs from the build manifest")
    if int(manifest["cache_size"]) != args.cache.stat().st_size:
        raise ValueError("training cache size differs from the build manifest")
    if int(manifest["image_count"]) != 9000 or int(manifest["parcel_count"]) != 200:
        raise ValueError("training cache manifest has unexpected dimensions")
    if int(manifest["max_voxels"]) != expected_max_voxels:
        raise ValueError("training cache manifest max_voxels differs from the data fingerprint")
    if manifest["metadata_sha256"] != sha256_file(args.metadata):
        raise ValueError("training cache manifest metadata hash mismatch")

    payload = {
        "schema_version": 1,
        "status": "verified",
        "cache_path": args.cache.resolve().relative_to(project_root).as_posix(),
        "cache_size": args.cache.stat().st_size,
        "cache_sha256": cache_sha256,
        "image_count": 9000,
        "selection_train_count": 8500,
        "validation_count": 500,
        "standard_test_overlap": 0,
        "brain_shape": [9000, 200, expected_max_voxels],
        "brain_dtype": "float32",
        "max_voxels_source": "data_fingerprint.json",
        "historical_reference_max_voxels": 626,
        "matches_historical_reference": expected_max_voxels == 626,
        "padding_policy": "zero outside parcel_valid_mask",
        "statistics_over_valid_values": statistics,
        "build_manifest_sha256": sha256_file(args.manifest),
        "data_fingerprint_sha256": sha256_file(args.data_fingerprint),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
