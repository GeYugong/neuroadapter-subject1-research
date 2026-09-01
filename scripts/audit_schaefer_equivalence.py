#!/usr/bin/env python3
"""Compare CBIG-derived and pinned whole_brain_encoder parcel memberships."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def vertices_sha256(value: torch.Tensor) -> str:
    vertices = value.detach().cpu().to(torch.int64).contiguous()
    return hashlib.sha256(vertices.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def load_parcels(path: Path) -> list[torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, list) or len(value) != 501:
        raise ValueError(f"expected 501 parcel lists: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    repository = args.repository_root.resolve()
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source = source_manifest["sources"]["whole_brain_encoder"]
    upstream_repository = repository / "vendor/whole_brain_encoder"
    observed_commit = subprocess.check_output(
        ["git", "-C", str(upstream_repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed_commit != source["commit"]:
        raise ValueError("whole_brain_encoder checkout differs from source manifest")

    hemispheres = {}
    loaded: dict[tuple[str, str], list[torch.Tensor]] = {}
    for hemi in ("lh", "rh"):
        derived_path = args.derived_dir / f"{hemi}_labels_s01.pt"
        upstream_path = args.upstream_dir / f"{hemi}_labels_s01.pt"
        derived = load_parcels(derived_path)
        upstream = load_parcels(upstream_path)
        loaded[("derived", hemi)] = derived
        loaded[("upstream", hemi)] = upstream
        derived_hashes = [vertices_sha256(value) for value in derived]
        upstream_hashes = [vertices_sha256(value) for value in upstream]
        upstream_index = {digest: index for index, digest in enumerate(upstream_hashes)}
        labels = []
        for label, (left, right, left_hash, right_hash) in enumerate(
            zip(derived, upstream, derived_hashes, upstream_hashes, strict=True)
        ):
            record = {
                "label": label,
                "derived_vertex_count": int(left.numel()),
                "upstream_vertex_count": int(right.numel()),
                "derived_vertex_sha256": left_hash,
                "upstream_vertex_sha256": right_hash,
                "equal": bool(torch.equal(left, right)),
                "matching_upstream_label": upstream_index.get(left_hash),
            }
            labels.append(record)
        equal_count = sum(record["equal"] for record in labels)
        set_intersection = len(set(derived_hashes) & set(upstream_hashes))
        indexed_equal = equal_count == len(labels)
        unordered_equal = set(derived_hashes) == set(upstream_hashes)
        hemispheres[hemi] = {
            "derived_file_sha256": sha256_file(derived_path),
            "upstream_file_sha256": sha256_file(upstream_path),
            "upstream_git_blob_sha1": subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(upstream_repository),
                    "rev-parse",
                    f"HEAD:parcels/schaefer/{hemi}_labels_s01.pt",
                ],
                text=True,
            ).strip(),
            "equal_label_count": equal_count,
            "label_count": len(labels),
            "all_labels_equal": indexed_equal,
            "unordered_membership_intersection": set_intersection,
            "unordered_membership_equal": unordered_equal,
            "relationship": (
                "indexed_equivalent"
                if indexed_equal
                else "permuted_equivalent"
                if unordered_equal
                else "different_memberships"
            ),
            "labels": labels,
        }

    upstream_lh = {vertices_sha256(value) for value in loaded[("upstream", "lh")]}
    upstream_rh = {vertices_sha256(value) for value in loaded[("upstream", "rh")]}
    cross_hemisphere = {
        "upstream_membership_intersection": len(upstream_lh & upstream_rh),
        "upstream_membership_sets_equal": upstream_lh == upstream_rh,
        "upstream_indexed_memberships_equal": all(
            torch.equal(left, right)
            for left, right in zip(
                loaded[("upstream", "lh")],
                loaded[("upstream", "rh")],
                strict=True,
            )
        ),
    }
    all_indexed_equal = all(
        hemisphere["all_labels_equal"] for hemisphere in hemispheres.values()
    )
    payload = {
        "schema_version": 1,
        "status": "indexed_equivalent" if all_indexed_equal else "mismatch",
        "comparison": "exact sorted fsaverage vertex membership, with indexed and permutation-aware checks",
        "provenance": {
            "derived_directory": args.derived_dir.resolve().relative_to(project_root).as_posix(),
            "upstream_runtime_directory": args.upstream_dir.resolve().relative_to(
                project_root
            ).as_posix(),
            "upstream_repository": source["url"],
            "upstream_commit": observed_commit,
            "upstream_relative_directory": "parcels/schaefer",
            "source_manifest_sha256": sha256_file(args.source_manifest),
        },
        "hemispheres": hemispheres,
        "cross_hemisphere": cross_hemisphere,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "lh_relationship": hemispheres["lh"]["relationship"],
                "rh_relationship": hemispheres["rh"]["relationship"],
                "upstream_lh_rh_sets_equal": cross_hemisphere[
                    "upstream_membership_sets_equal"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
