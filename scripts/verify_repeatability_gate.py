#!/usr/bin/env python3
"""Create exact resume, decode, or evaluator repeatability gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import load_gate_requirements, method_fingerprint
from neuroadapter_research.reproducibility import (
    compare_resume_outputs,
    structural_sha256,
    verify_decode_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=("resume_equivalence", "decode_determinism", "evaluator_repeatability"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-aux", type=Path)
    parser.add_argument("--right-aux", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"gate output already exists: {args.output}")
    config = load_training_config(args.config, require_frozen=True)
    requirements = load_gate_requirements(config.paths["gate_requirements"])
    fingerprint = method_fingerprint(config)

    if args.gate == "resume_equivalence":
        if args.left_aux is None or args.right_aux is None:
            raise ValueError("resume comparison requires both trace directories")
        evidence = compare_resume_outputs(
            args.left, args.right, args.left_aux, args.right_aux
        )
    elif args.gate == "decode_determinism":
        left = verify_decode_tree(args.left)
        right = verify_decode_tree(args.right)
        if left != right:
            raise ValueError("deterministic decode manifests differ")
        if left.get("method_fingerprint") != fingerprint:
            raise ValueError("decode artifact uses a different method")
        evidence = {
            "decode_manifest_structural_sha256": structural_sha256(left),
            "png_count": int(left["image_count"]) * int(left["candidate_count"]),
        }
    else:
        if args.left_aux is None or args.right_aux is None:
            raise ValueError("evaluator comparison requires both per-image CSV files")
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
        if left != right:
            raise ValueError("evaluator JSON outputs differ")
        if left.get("method_fingerprint") != fingerprint:
            raise ValueError("evaluator artifact uses a different method")
        left_csv = sha256_file(args.left_aux)
        if left_csv != sha256_file(args.right_aux):
            raise ValueError("evaluator per-image CSV outputs differ")
        if left.get("per_image_csv_sha256") != left_csv:
            raise ValueError("evaluator JSON does not bind the per-image CSV")
        evidence = {
            "evaluator_structural_sha256": structural_sha256(left),
            "per_image_csv_sha256": left_csv,
        }

    payload = {
        "schema_version": 1,
        "gate": args.gate,
        "status": "passed",
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        "gate_requirements_sha256": requirements.sha256,
        "left_input_sha256": sha256_file(args.left / "MANIFEST.json")
        if args.left.is_dir()
        else sha256_file(args.left),
        "right_input_sha256": sha256_file(args.right / "MANIFEST.json")
        if args.right.is_dir()
        else sha256_file(args.right),
        "evidence": evidence,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
