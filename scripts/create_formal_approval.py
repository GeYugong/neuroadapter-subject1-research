#!/usr/bin/env python3
"""Create the exact approval payload accepted by formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.approval import expected_approval_payload
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import verify_protocol_repository


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
    verify_protocol_repository(
        Path(__file__).resolve().parents[1], config.raw["protocol_commit"]
    )
    payload = expected_approval_payload(config)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
