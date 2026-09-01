#!/usr/bin/env python3
"""Create the exact approval payload accepted by formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.integrity import (
    load_json_mapping,
    validate_gate_artifact,
    validate_subject1_audits,
)


GATE_PATHS = (
    "hardware_gate",
    "forward_alignment",
    "batch_gate",
    "resume_equivalence",
    "decode_determinism",
    "evaluator_repeatability",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve:
        raise ValueError("approval creation requires the explicit --approve flag")
    if args.output.exists():
        raise FileExistsError(f"approval file already exists: {args.output}")

    config = load_training_config(args.config, require_frozen=True)
    for name in GATE_PATHS:
        validate_gate_artifact(
            config.paths[name], expected_gate=name, config_sha256=config.sha256
        )
    canonical = load_json_mapping(config.paths["canonical_manifest"])
    if canonical.get("status") != "frozen":
        raise ValueError("canonical initialization is not frozen")
    validate_subject1_audits(config.paths)

    payload = {
        "approved": True,
        "config_sha256": config.sha256,
        "protocol_commit": config.raw["protocol_commit"],
        "environment_lock_sha256": sha256_file(config.paths["environment_lock"]),
        "hardware_gate_sha256": sha256_file(config.paths["hardware_gate"]),
        "forward_alignment_sha256": sha256_file(config.paths["forward_alignment"]),
        "batch_gate_sha256": sha256_file(config.paths["batch_gate"]),
        "resume_equivalence_sha256": sha256_file(config.paths["resume_equivalence"]),
        "decode_determinism_sha256": sha256_file(config.paths["decode_determinism"]),
        "evaluator_repeatability_sha256": sha256_file(
            config.paths["evaluator_repeatability"]
        ),
        "data_fingerprint_sha256": sha256_file(config.paths["data_fingerprint"]),
        "model_assets_manifest_sha256": sha256_file(config.paths["model_manifest"]),
        "canonical_initialization_sha256": sha256_file(
            config.paths["canonical_initialization"]
        ),
        "training_cache_verification_sha256": sha256_file(
            config.paths["training_cache_verification"]
        ),
        "nsd_image_mapping_sha256": sha256_file(config.paths["nsd_image_mapping"]),
        "schaefer_equivalence_sha256": sha256_file(
            config.paths["schaefer_equivalence"]
        ),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
