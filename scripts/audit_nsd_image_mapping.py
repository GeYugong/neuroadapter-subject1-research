#!/usr/bin/env python3
"""Cross-check Subject 1 image IDs against the official NSD design and stimulus table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expdesign", type=Path, required=True)
    parser.add_argument("--stim-info", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    metadata = np.load(args.metadata, allow_pickle=True).item()
    design = loadmat(args.expdesign)
    masterordering = np.asarray(design["masterordering"], dtype=np.int64).reshape(-1)
    subjectim = np.asarray(design["subjectim"], dtype=np.int64)
    if masterordering.shape != (30000,) or subjectim.shape != (8, 10000):
        raise ValueError("unexpected NSD experiment design dimensions")

    subject_slots = masterordering - 1
    image_ids = subjectim[0, subject_slots] - 1
    metadata_order = np.asarray(metadata["img_presentation_order"], dtype=np.int64)
    if not np.array_equal(image_ids, metadata_order):
        raise ValueError("metadata presentation order differs from official masterordering")

    stimulus = pd.read_csv(args.stim_info)
    if stimulus.shape[0] != 73000 or not np.array_equal(
        stimulus["nsdId"].to_numpy(dtype=np.int64), np.arange(73000)
    ):
        raise ValueError("NSD stimulus table has an unexpected nsdId index")

    train = set(np.asarray(metadata["train_img_num"], dtype=np.int64).tolist())
    test = set(np.asarray(metadata["test_img_num"], dtype=np.int64).tolist())
    if len(train) != 9000 or len(test) != 1000 or train & test:
        raise ValueError("metadata does not contain the expected disjoint 9000/1000 split")

    rows = []
    for trial_index, (slot, image_id) in enumerate(zip(masterordering, image_ids, strict=True)):
        source = stimulus.iloc[int(image_id)]
        split = "train" if int(image_id) in train else "test"
        rows.append(
            {
                "trial_index_0based": trial_index,
                "masterordering_1based": int(slot),
                "subjectim_slot_1based": int(slot),
                "nsd_image_id_0based": int(image_id),
                "nsd_image_id_1based": int(image_id) + 1,
                "split": split,
                "shared1000": bool(source["shared1000"]),
                "subject1": bool(source["subject1"]),
                "coco_id": int(source["cocoId"]),
                "coco_split": str(source["cocoSplit"]),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unique = stimulus.iloc[sorted(train | test)]
    train_table = stimulus.iloc[sorted(train)]
    test_table = stimulus.iloc[sorted(test)]
    checks = {
        "train_shared1000_false": bool((~train_table["shared1000"].astype(bool)).all()),
        "test_shared1000_true": bool(test_table["shared1000"].astype(bool).all()),
        "all_subject1_true": bool(unique["subject1"].astype(bool).all()),
        "presentation_order_equal": True,
    }
    if not all(checks.values()):
        raise ValueError(f"NSD image mapping audit failed: {checks}")
    payload = {
        "schema_version": 1,
        "status": "verified",
        "subject": 1,
        "train_image_count": len(train),
        "test_image_count": len(test),
        "trial_count": len(rows),
        "checks": checks,
        "metadata_sha256": sha256_file(args.metadata),
        "expdesign_sha256": sha256_file(args.expdesign),
        "stim_info_sha256": sha256_file(args.stim_info),
        "mapping_csv_sha256": sha256_file(args.output_csv),
    }
    write_json_atomic(args.output_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
