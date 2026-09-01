#!/usr/bin/env python3
"""Build a shortlist or select U* from frozen validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import (
    load_selection_plan,
    method_fingerprint,
    validate_selection_config_and_plan,
    validate_selection_plan_inputs,
    verify_protocol_repository,
)
from neuroadapter_research.selection import (
    EVALUATION_BINDING_FIELDS,
    build_shortlist,
    merge_evaluation_payloads,
    select_checkpoint,
)


def validate_shortlist_manifest(
    payload: dict,
    *,
    expected_binding: dict,
    expected_updates: list[int],
) -> list[int]:
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "shortlist"
        or payload.get("status") != "complete"
    ):
        raise ValueError("shortlist manifest is not complete schema version 1")
    for name, expected in expected_binding.items():
        if payload.get(name) != expected:
            raise ValueError(f"shortlist manifest has an invalid frozen binding: {name}")
    if payload.get("expected_snapshot_updates") != expected_updates:
        raise ValueError("shortlist manifest uses a different snapshot schedule")
    updates = [int(value) for value in payload.get("shortlist_updates", [])]
    if len(updates) != 5 or len(set(updates)) != 5:
        raise ValueError("shortlist manifest must contain five unique updates")
    if not set(updates).issubset(expected_updates):
        raise ValueError("shortlist manifest contains an unplanned update")
    return updates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("shortlist", "final"), required=True)
    parser.add_argument("--shortlist-manifest", type=Path)
    args = parser.parse_args()
    config = load_training_config(args.config, require_frozen=True)
    repository = Path(__file__).resolve().parents[1]
    verify_protocol_repository(repository, config.raw["protocol_commit"])
    plan = load_selection_plan(config.paths["selection_plan"], require_frozen=True)
    validate_selection_config_and_plan(config, plan)
    plan_binding = validate_selection_plan_inputs(
        plan,
        validation_ids_path=config.paths["validation_ids"],
        repository_root=repository,
    )
    fingerprint = method_fingerprint(config)
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    expected_candidates = int(
        plan.raw["screening_candidates"]
        if args.stage == "shortlist"
        else plan.raw["final_candidates"]
    )
    records = merge_evaluation_payloads(
        sources, expected_candidate_count=expected_candidates
    )
    binding = {name: sources[0][name] for name in EVALUATION_BINDING_FIELDS}
    expected_binding = {
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        **plan_binding,
        "image_count": 500,
        "candidate_count": expected_candidates,
        "negative_pool": "the same 500 unique validation image IDs for every candidate seed",
        "candidate_aggregation": "per-image arithmetic mean after fixed-pool scoring",
        "evaluation_manifest_sha256": sha256_file(config.paths["evaluation_manifest"]),
        "protocol_namespace": plan.raw["protocol_namespace"],
        "selection_stage": "screening" if args.stage == "shortlist" else "final",
        "denoising_steps": plan.raw["denoising_steps"],
        "guidance_scale": plan.raw["guidance_scale"],
        "evaluation_batch_size": plan.raw["evaluation_batch_size"],
        "repository_commit": config.raw["protocol_commit"],
    }
    for name, expected in expected_binding.items():
        if binding.get(name) != expected:
            raise ValueError(f"evaluation binding differs from selection plan: {name}")
    source_hashes = [
        {"name": path.name, "sha256": sha256_file(path)} for path in args.input
    ]
    if args.stage == "shortlist":
        if args.shortlist_manifest is not None:
            raise ValueError("shortlist stage does not accept --shortlist-manifest")
        payload = {
            "schema_version": 1,
            "stage": "shortlist",
            "status": "complete",
            **binding,
            "sources": source_hashes,
            "expected_snapshot_updates": plan.raw["expected_snapshot_updates"],
            "shortlist_updates": build_shortlist(
                records, expected_updates=plan.raw["expected_snapshot_updates"]
            ),
        }
    else:
        if args.shortlist_manifest is None:
            raise ValueError("final selection requires --shortlist-manifest")
        shortlist = json.loads(args.shortlist_manifest.read_text(encoding="utf-8"))
        shortlist_binding = dict(expected_binding)
        shortlist_binding["candidate_count"] = int(plan.raw["screening_candidates"])
        shortlist_binding["selection_stage"] = "screening"
        shortlist_updates = validate_shortlist_manifest(
            shortlist,
            expected_binding=shortlist_binding,
            expected_updates=plan.raw["expected_snapshot_updates"],
        )
        observed_updates = [int(record["optimizer_update"]) for record in records]
        if set(observed_updates) != set(shortlist_updates):
            raise ValueError("final checkpoint set differs from the shortlist manifest")
        if len(records) != len(shortlist_updates):
            raise ValueError("final selection requires exactly five shortlisted checkpoints")
        result = select_checkpoint(
            records,
            bootstrap_draws=int(plan.raw["bootstrap_draws"]),
            bootstrap_seed=int(plan.raw["bootstrap_seed"]),
        )
        payload = {
            "schema_version": 1,
            "stage": "final_selection",
            "status": "complete",
            **binding,
            "sources": source_hashes,
            "shortlist_manifest_sha256": sha256_file(args.shortlist_manifest),
            "shortlist_updates": shortlist_updates,
            "bootstrap_draws": plan.raw["bootstrap_draws"],
            "bootstrap_seed": plan.raw["bootstrap_seed"],
            "selected_update_u_star": result.selected_update,
            "selected_images_seen": 16 * result.selected_update,
            "best_semantic_update": result.best_semantic_update,
            "one_se_updates": list(result.one_se_updates),
            "diagnostics": {str(key): value for key, value in result.diagnostics.items()},
        }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
