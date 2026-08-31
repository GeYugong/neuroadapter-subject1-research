#!/usr/bin/env python3
"""Convert frozen Schaefer fsaverage annotations to NeuroAdapter label lists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(raw_dir: Path, output_dir: Path, subject: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"subject": subject, "hemispheres": {}}

    for hemi in ("lh", "rh"):
        annotation = raw_dir / f"{hemi}.Schaefer2018_1000Parcels_7Networks_order.annot"
        labels, _color_table, names = fsio.read_annot(str(annotation), orig_ids=False)
        labels = np.asarray(labels, dtype=np.int64)

        if labels.shape != (163842,):
            raise ValueError(f"{hemi}: expected 163842 vertices, got {labels.shape}")
        unique_labels = np.unique(labels)
        if unique_labels.tolist() != list(range(501)):
            raise ValueError(f"{hemi}: expected labels 0..500, got {unique_labels.tolist()[:10]}...")
        if len(names) != 501:
            raise ValueError(f"{hemi}: expected 501 annotation names, got {len(names)}")

        tensor = torch.from_numpy(labels.copy()).long()
        parcels = [torch.where(tensor == label_id)[0].long() for label_id in range(501)]
        output = output_dir / f"{hemi}_labels_s{subject:02}.pt"
        torch.save(parcels, output)

        vertex_hashes = []
        for label_id, vertices in enumerate(parcels):
            vertex_hashes.append(
                {
                    "label_id": label_id,
                    "vertex_count": int(vertices.numel()),
                    "vertex_sha256": hashlib.sha256(vertices.numpy().tobytes()).hexdigest(),
                }
            )

        summary["hemispheres"][hemi] = {
            "annotation": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "vertices": 163842,
            "labels_including_medial_wall": 501,
            "parcels_excluding_medial_wall": 500,
            "medial_wall_vertices": int(parcels[0].numel()),
            "minimum_parcel_vertices": min(int(p.numel()) for p in parcels[1:]),
            "maximum_parcel_vertices": max(int(p.numel()) for p in parcels[1:]),
            "parcel_vertex_hashes": vertex_hashes,
        }

    summary_path = output_dir / "schaefer_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(convert(args.raw_dir, args.output_dir, args.subject), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

