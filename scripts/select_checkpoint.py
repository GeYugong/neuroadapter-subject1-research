#!/usr/bin/env python3
"""Build a shortlist or select U* from frozen validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.selection import (
    build_shortlist,
    merge_evaluation_payloads,
    select_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("shortlist", "final"), required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    expected_candidates = 2 if args.stage == "shortlist" else 8
    records = merge_evaluation_payloads(
        sources, expected_candidate_count=expected_candidates
    )
    source_hashes = [
        {"name": path.name, "sha256": sha256_file(path)} for path in args.input
    ]
    if args.stage == "shortlist":
        payload = {
            "schema_version": 1,
            "stage": "shortlist",
            "sources": source_hashes,
            "shortlist_updates": build_shortlist(records),
        }
    else:
        if len(records) != 5:
            raise ValueError("final selection requires exactly five shortlisted checkpoints")
        result = select_checkpoint(
            records,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        payload = {
            "schema_version": 1,
            "stage": "final_selection",
            "sources": source_hashes,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.bootstrap_seed,
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
