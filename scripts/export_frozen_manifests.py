#!/usr/bin/env python3
"""Export small immutable evidence into the Git-tracked manifest tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from neuroadapter_research.atomic import sha256_file, write_json_atomic


JSON_SOURCES = {
    "raw_nsd_inventory_summary.json": "data/fingerprints/nsd_subj01_raw_verification.json",
    "raw_nsd_sha256_inventory.json": "data/fingerprints/nsd_subj01_sha256_inventory.json",
    "source_nonfinite_values.json": "data/derived/neural_data/source_nonfinite_values.json",
    "data_fingerprint.json": "data/fingerprints/data_fingerprint.json",
    "split_manifest.json": "data/derived/splits/split_manifest.json",
    "training_cache_manifest.json": "data/fingerprints/training_cache_manifest.json",
    "training_cache_verification.json": "data/fingerprints/training_cache_verification.json",
    "model_assets_sha256.json": "data/fingerprints/model_assets_sha256.json",
    "evaluation_assets.json": "data/fingerprints/evaluation_downloads.json",
    "brain_encoder_assets_verification.json": "data/fingerprints/brain_encoder_assets_verification.json",
    "canonical_initialization.json": "models/canonical/subject01_adapter_init.json",
    "nsd_image_mapping_subject01.json": "data/fingerprints/nsd_image_mapping_subject01.json",
    "schaefer_upstream_equivalence.json": "data/fingerprints/schaefer_upstream_equivalence.json",
}
TEXT_SOURCES = {
    "parcel_token_map.csv": "data/fingerprints/parcel_token_map.csv",
    "selection_train_ids.txt": "data/derived/splits/selection_train_ids.txt",
    "validation_ids.txt": "data/derived/splits/validation_ids.txt",
    "test_ids.txt": "data/derived/splits/test_ids.txt",
    "nsd_image_mapping_subject01.csv": "data/fingerprints/nsd_image_mapping_subject01.csv",
}


def sanitize(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, project_root) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        root = project_root.as_posix().rstrip("/")
        if normalized == root:
            return "."
        if normalized.startswith(root + "/"):
            return normalized[len(root) + 1 :]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.repository_root.resolve() / "manifests" / "frozen"
    output_root.mkdir(parents=True, exist_ok=True)

    exported = []
    for output_name, source_name in JSON_SOURCES.items():
        source = project_root / source_name
        payload = json.loads(source.read_text(encoding="utf-8"))
        output = output_root / output_name
        write_json_atomic(output, sanitize(payload, project_root))
        exported.append(
            {"path": output_name, "source": source_name, "sha256": sha256_file(output)}
        )

    for output_name, source_name in TEXT_SOURCES.items():
        source = project_root / source_name
        output = output_root / output_name
        shutil.copyfile(source, output)
        exported.append(
            {"path": output_name, "source": source_name, "sha256": sha256_file(output)}
        )

    paper = args.repository_root.resolve() / "manifests" / "paper.json"
    paper_output = output_root / "paper.json"
    shutil.copyfile(paper, paper_output)
    exported.append(
        {"path": "paper.json", "source": "manifests/paper.json", "sha256": sha256_file(paper_output)}
    )

    forbidden = (project_root.as_posix(), "/data/matengyu/", "C:/Users/")
    for record in exported:
        text = (output_root / record["path"]).read_text(encoding="utf-8", errors="strict")
        if any(token in text.replace("\\", "/") for token in forbidden):
            raise ValueError(f"export still contains a private absolute path: {record['path']}")

    index = {
        "schema_version": 1,
        "policy": "Git tracks manifests, IDs, hashes, and statistics only; no data or weights.",
        "files": sorted(exported, key=lambda item: item["path"]),
    }
    write_json_atomic(output_root / "INDEX.json", index)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
