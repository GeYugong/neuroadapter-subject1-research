#!/usr/bin/env python3
"""Verify the CBIG-derived decoder atlas and final 200-token ordering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neuroadapter_research.atomic import sha256_file, write_json_atomic


HEMISPHERES = ("lh", "rh")
VERTICES = 163842


def vertices_sha256(vertices: torch.Tensor) -> str:
    value = vertices.detach().cpu().to(torch.int64).contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(value.tobytes()).hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def relative_to_project(path: Path, project_root: Path) -> str:
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()


def load_parcels(path: Path) -> list[torch.Tensor]:
    parcels = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(parcels, list) or len(parcels) != 501:
        raise ValueError(f"expected 501 labels including medial wall: {path}")
    for label, vertices in enumerate(parcels):
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 1:
            raise ValueError(f"invalid parcel tensor {label}: {path}")
        values = vertices.to(torch.int64)
        if values.numel() == 0 or values.min() < 0 or values.max() >= VERTICES:
            raise ValueError(f"invalid fsaverage vertices in label {label}: {path}")
        if not torch.equal(values, torch.unique(values, sorted=True)):
            raise ValueError(f"parcel vertices are not sorted and unique: {path}:{label}")
    return parcels


def load_token_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or [int(row["model_token"]) for row in rows] != list(range(200)):
        raise ValueError("parcel token map must contain model tokens 0..199 in order")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parcel-map", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))["sources"]
    source = sources["schaefer2018"]
    revision = str(source["revision"])
    if (args.annotation_dir / "CBIG_REVISION").read_text(encoding="ascii").strip() != revision:
        raise ValueError("downloaded CBIG revision differs from the source manifest")
    metadata = np.load(args.metadata, allow_pickle=True).item()
    cache_manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    token_rows = load_token_rows(args.parcel_map)

    hemispheres: dict[str, dict] = {}
    ordered_tokens = []
    maximum = 0
    for hemi_index, hemi in enumerate(HEMISPHERES):
        annotation_name = f"{hemi}.Schaefer2018_1000Parcels_7Networks_order.annot"
        annotation = args.annotation_dir / annotation_name
        parcel_path = args.parcel_dir / f"{hemi}_labels_s01.pt"
        parcels = load_parcels(parcel_path)
        usable = parcels[1:]
        ncsnr = np.asarray(metadata[f"{hemi}_ncsnr"], dtype=np.float32)
        if ncsnr.shape != (VERTICES,) or not np.isfinite(ncsnr).all():
            raise ValueError(f"invalid {hemi} ncsnr metadata")
        scores = np.asarray([ncsnr[value.numpy()].mean() for value in usable])
        expected_indices = np.argsort(scores)[::-1][:100]
        manifest_indices = np.asarray(
            cache_manifest["selected_parcel_indices_excluding_medial_wall"][hemi],
            dtype=np.int64,
        )
        if not np.array_equal(expected_indices, manifest_indices):
            raise ValueError(f"{hemi} top-SNR ranking differs from the cache manifest")

        rows = token_rows[hemi_index * 100 : (hemi_index + 1) * 100]
        for rank, (row, index) in enumerate(zip(rows, expected_indices, strict=True), start=1):
            label_id = int(index) + 1
            vertices = usable[int(index)]
            digest = vertices_sha256(vertices)
            expected_row = {
                "model_token": (hemi_index * 100) + rank - 1,
                "hemisphere": hemi,
                "parcel_index": int(index),
                "schaefer_label_id": label_id,
                "snr_rank": rank,
                "vertex_count": int(vertices.numel()),
                "vertex_sha256": digest,
            }
            observed = {
                "model_token": int(row["model_token"]),
                "hemisphere": row["hemisphere"],
                "parcel_index": int(row["parcel_index_zero_based_excluding_medial_wall"]),
                "schaefer_label_id": int(row["schaefer_label_id"]),
                "snr_rank": int(row["snr_rank"]),
                "vertex_count": int(row["vertex_count"]),
                "vertex_sha256": row["vertex_sha256"],
            }
            if observed != expected_row:
                raise ValueError(f"model token map mismatch at token {observed['model_token']}")
            maximum = max(maximum, expected_row["vertex_count"])
            ordered_tokens.append(expected_row)

        hemispheres[hemi] = {
            "annotation_relative_path": relative_to_project(annotation, project_root),
            "annotation_size": annotation.stat().st_size,
            "annotation_sha256": sha256_file(annotation),
            "annotation_git_blob_sha1": git_blob_sha1(annotation),
            "parcel_file_relative_path": relative_to_project(parcel_path, project_root),
            "parcel_file_sha256": sha256_file(parcel_path),
            "vertices": VERTICES,
            "labels_including_medial_wall": len(parcels),
            "parcels_excluding_medial_wall": len(usable),
            "selected_parcels": 100,
            "selected_indices_excluding_medial_wall": [int(value) for value in expected_indices],
        }

    if maximum != int(cache_manifest["max_voxels"]):
        raise ValueError("selected atlas max_voxels differs from the cache manifest")
    token_digest = hashlib.sha256()
    for record in ordered_tokens:
        token_digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        token_digest.update(b"\n")

    payload = {
        "schema_version": 1,
        "gate": "decoder_atlas",
        "status": "verified",
        "subject": 1,
        "surface_space": "fsaverage",
        "source": {
            "repository": source["repository"],
            "commit": revision,
            "relative_directory": "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label",
        },
        "runtime_inputs": {
            "metadata_relative_path": relative_to_project(args.metadata, project_root),
            "metadata_sha256": sha256_file(args.metadata),
            "parcel_map_relative_path": relative_to_project(args.parcel_map, project_root),
            "parcel_map_sha256": sha256_file(args.parcel_map),
            "cache_manifest_relative_path": relative_to_project(args.cache_manifest, project_root),
            "cache_manifest_sha256": sha256_file(args.cache_manifest),
        },
        "hemispheres": hemispheres,
        "top_snr_ranking_verified": True,
        "model_token_count": len(ordered_tokens),
        "model_token_order": "lh SNR rank 1..100, then rh SNR rank 1..100",
        "ordered_model_token_vertex_sha256": token_digest.hexdigest(),
        "max_voxels": maximum,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
