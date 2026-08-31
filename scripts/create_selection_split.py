#!/usr/bin/env python3
"""Create the frozen 8500/500 image-level split inside the NSD train pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--validation-size", type=int, default=500)
    args = parser.parse_args()

    metadata = np.load(args.metadata, allow_pickle=True).item()
    train_pool = np.asarray(metadata["train_img_num"], dtype=np.int64)
    test = np.asarray(metadata["test_img_num"], dtype=np.int64)
    if train_pool.size != 9000 or np.unique(train_pool).size != 9000:
        raise ValueError("metadata must contain 9000 unique train image IDs")
    if test.size != 1000 or np.intersect1d(train_pool, test).size:
        raise ValueError("invalid standard test split")

    generator = np.random.Generator(np.random.PCG64(args.seed))
    permutation = generator.permutation(train_pool)
    validation = np.sort(permutation[: args.validation_size])
    selection_train = np.sort(permutation[args.validation_size :])

    if validation.size != 500 or selection_train.size != 8500:
        raise ValueError("expected selection_train/validation sizes 8500/500")
    if np.intersect1d(selection_train, validation).size:
        raise ValueError("selection_train and validation overlap")
    if not np.array_equal(np.sort(np.concatenate([selection_train, validation])), np.sort(train_pool)):
        raise ValueError("selection split does not exactly partition the train pool")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "selection_train": selection_train,
        "validation": validation,
        "test": np.sort(test),
    }
    for name, values in arrays.items():
        write_text_atomic(
            args.output_dir / f"{name}_ids.txt",
            "".join(f"{int(value)}\n" for value in values),
        )

    payload = {
        "schema_version": 1,
        "subject": 1,
        "unit": "unique_nsd_image_id_zero_based",
        "algorithm": "numpy.random.Generator(PCG64)",
        "seed": args.seed,
        "counts": {name: int(values.size) for name, values in arrays.items()},
        "sha256": {name: array_sha256(values) for name, values in arrays.items()},
        "train_pool_sha256": array_sha256(np.sort(train_pool)),
    }
    write_text_atomic(
        args.output_dir / "split_manifest.json",
        json.dumps(payload, indent=2) + "\n",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

