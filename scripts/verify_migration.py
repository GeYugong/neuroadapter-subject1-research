#!/usr/bin/env python3
"""Verify a relocated tree against its unmodified source SHA-256 inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.integrity import verify_tree_against_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root.resolve()
    if args.output.resolve().is_relative_to(root):
        raise ValueError("migration report must be outside the verified tree")
    result = verify_tree_against_manifest(root, args.source_manifest)
    payload = {
        "schema_version": 1,
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_inventory_sha256": sha256_file(args.source_manifest),
        "tree_name": root.name,
        **result,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
