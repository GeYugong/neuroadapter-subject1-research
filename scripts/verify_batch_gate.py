#!/usr/bin/env python3
"""Verify stable preferred/fallback gate runs and freeze one batch geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config


def load_run(path: Path, expected_geometry: dict[str, int], minimum_updates: int) -> dict[str, Any]:
    effective_path = path / "effective_run.json"
    status_path = path / "run_status.json"
    log_path = path / "training.jsonl"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if effective.get("run_mode") != "gate":
        raise ValueError("batch evidence is not a gate run")
    if effective.get("batch_geometry") != expected_geometry:
        raise ValueError("batch gate run has unexpected geometry")
    if status.get("status") != "completed" or int(status["last_completed_update"]) < minimum_updates:
        raise ValueError("batch gate run did not complete the minimum updates")
    if not rows or max(int(row["optimizer_update"]) for row in rows) < minimum_updates:
        raise ValueError("batch gate training log is incomplete")
    for row in rows:
        for name in ("loss", "gradient_norm_before_clip", "max_memory_reserved_bytes"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"batch gate contains non-finite {name}")
    return {
        "config_sha256": effective["config_sha256"],
        "input_hashes": effective["input_hashes"],
        "canonical_initialization_sha256": effective["input_hashes"][
            "canonical_initialization"
        ],
        "backend": effective["backend"],
        "batch_geometry": effective["batch_geometry"],
        "completed_updates": int(status["last_completed_update"]),
        "maximum_memory_reserved_bytes": max(
            int(row["max_memory_reserved_bytes"]) for row in rows
        ),
        "effective_run_sha256": sha256_file(effective_path),
        "run_status_sha256": sha256_file(status_path),
        "training_log_sha256": sha256_file(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preferred-run", type=Path, required=True)
    parser.add_argument("--fallback-run", type=Path, required=True)
    parser.add_argument("--selected", choices=("preferred", "fallback"), required=True)
    parser.add_argument("--minimum-updates", type=int, default=532)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"batch gate output already exists: {args.output}")
    config = load_training_config(args.config, require_frozen=True)
    common = {"world_size": 2, "global_batch_size": 16}
    preferred = load_run(
        args.preferred_run,
        {**common, "micro_batch_size": 8, "gradient_accumulation_steps": 1},
        args.minimum_updates,
    )
    fallback = load_run(
        args.fallback_run,
        {**common, "micro_batch_size": 4, "gradient_accumulation_steps": 2},
        args.minimum_updates,
    )
    for name in ("backend", "canonical_initialization_sha256"):
        if preferred[name] != fallback[name]:
            raise ValueError(f"batch gate runs differ in {name}")
    left_inputs = {key: value for key, value in preferred["input_hashes"].items() if key != "config"}
    right_inputs = {key: value for key, value in fallback["input_hashes"].items() if key != "config"}
    if left_inputs != right_inputs:
        raise ValueError("batch gate runs use different frozen inputs")
    selected = preferred if args.selected == "preferred" else fallback
    if selected["config_sha256"] != config.sha256:
        raise ValueError("selected batch gate run differs from the frozen config")
    payload = {
        "schema_version": 1,
        "gate": "batch_gate",
        "status": "passed",
        "config_sha256": config.sha256,
        "strict_weight_equivalence_claimed": False,
        "selection": args.selected,
        "preferred": preferred,
        "fallback": fallback,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
