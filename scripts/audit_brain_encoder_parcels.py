#!/usr/bin/env python3
"""Audit parcel/query identity required by the fixed brain encoder checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PosixPath

import torch

from neuroadapter_research.atomic import sha256_file, write_json_atomic


LAYERS = (1, 3, 5, 7)
RUNS = (1, 2)
HEMISPHERES = ("lh", "rh")
VERTICES = 163842


def canonical_parcel_mask_sha256(parcels: list[torch.Tensor]) -> str:
    labels = torch.full((VERTICES,), -1, dtype=torch.int64)
    for label, vertices in enumerate(parcels):
        values = vertices.to(torch.int64)
        if torch.any(labels[values] != -1):
            raise ValueError("brain encoder parcels overlap")
        labels[values] = label
    if torch.any(labels < 0):
        raise ValueError("brain encoder parcels do not cover fsaverage")
    digest = hashlib.sha256()
    for start in range(0, VERTICES, 1024):
        chunk_labels = labels[start : start + 1024]
        mask = torch.zeros((len(chunk_labels), len(parcels)), dtype=torch.float32)
        mask[torch.arange(len(chunk_labels)), chunk_labels] = 1.0
        digest.update(mask.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def git_blob(repository: Path, relative: str) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "HEAD", relative], text=True
    ).strip()
    fields = output.split()
    if len(fields) < 3 or fields[1] != "blob":
        raise ValueError(f"parcel file is not a Git blob: {relative}")
    return fields[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository_root.resolve()
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    wbe_source = source_manifest["sources"]["whole_brain_encoder"]
    observed_head = subprocess.check_output(
        ["git", "-C", str(repository / "vendor/whole_brain_encoder"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_head != wbe_source["commit"]:
        raise ValueError("whole_brain_encoder checkout differs from source manifest")

    parcel_records = {}
    parcel_count = None
    for hemi in HEMISPHERES:
        path = args.parcel_dir / f"{hemi}_labels_s01.pt"
        parcels = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(parcels, list) or len(parcels) != 501:
            raise ValueError(f"invalid brain encoder parcel file: {path}")
        parcel_count = len(parcels) if parcel_count is None else parcel_count
        parcel_records[hemi] = {
            "runtime_file_sha256": sha256_file(path),
            "parcel_count": len(parcels),
            "canonical_float32_parcel_mask_sha256": canonical_parcel_mask_sha256(parcels),
            "upstream_repository": wbe_source["url"],
            "upstream_commit": observed_head,
            "upstream_relative_path": f"parcels/schaefer/{hemi}_labels_s01.pt",
            "upstream_git_blob_sha1": git_blob(
                repository / "vendor/whole_brain_encoder",
                f"parcels/schaefer/{hemi}_labels_s01.pt",
            ),
        }

    torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])
    checkpoint_entries = []
    checkpoint_parcel_dirs = set()
    for layer in LAYERS:
        for run in RUNS:
            for hemi in HEMISPHERES:
                path = (
                    args.asset_root
                    / f"enc_{layer}"
                    / f"run_{run}"
                    / hemi
                    / "checkpoint_nonavg.pth"
                )
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                checkpoint_args = checkpoint["args"]
                checkpoint_parcel_dir = str(getattr(checkpoint_args, "parcel_dir", ""))
                checkpoint_parcel_dirs.add(checkpoint_parcel_dir)
                query_keys = [
                    name for name in checkpoint["model"] if name.endswith("query_embed.weight")
                ]
                if len(query_keys) != 1:
                    raise ValueError(f"checkpoint has no unique query embedding: {path}")
                query_shape = list(checkpoint["model"][query_keys[0]].shape)
                if query_shape[0] != parcel_count:
                    raise ValueError(f"query count differs from parcel count: {path}")
                checkpoint_entries.append(
                    {
                        "encoder_layer": layer,
                        "run": run,
                        "hemisphere": hemi,
                        "checkpoint_sha256": sha256_file(path),
                        "checkpoint_parcel_dir": checkpoint_parcel_dir,
                        "query_embed_key": query_keys[0],
                        "query_embed_shape": query_shape,
                    }
                )
                del checkpoint
    if len(checkpoint_entries) != 16 or len(checkpoint_parcel_dirs) != 1:
        raise ValueError("brain encoder checkpoints do not share one parcel identity")

    relation = "unverified"
    identity_sha256 = None
    if args.identity_manifest is not None:
        identity = json.loads(args.identity_manifest.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "checkpoint_parcel_dir": next(iter(checkpoint_parcel_dirs)),
            "files": {
                hemi: {"sha256": parcel_records[hemi]["runtime_file_sha256"]}
                for hemi in HEMISPHERES
            },
        }
        if identity != expected:
            raise ValueError("brain encoder parcel identity manifest does not match runtime assets")
        relation = "verified_by_identity_manifest"
        identity_sha256 = sha256_file(args.identity_manifest)

    payload = {
        "schema_version": 1,
        "gate": "brain_encoder_parcel",
        "status": "verified" if relation != "unverified" else "blocked",
        "subject": 1,
        "checkpoint_count": len(checkpoint_entries),
        "checkpoint_query_counts_verified": True,
        "checkpoint_parcel_dir": next(iter(checkpoint_parcel_dirs)),
        "runtime_parcel_source_relation": relation,
        "identity_manifest_sha256": identity_sha256,
        "whole_brain_encoder_commit": observed_head,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "parcels": parcel_records,
        "checkpoints": checkpoint_entries,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
