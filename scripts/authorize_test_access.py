#!/usr/bin/env python3
"""Issue a local token after every standard-test access gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.test_access import verify_test_access


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--required-tag", default="subject01-final-v1")
    parser.add_argument("--brain-encoder-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"test access token already exists: {args.output}")
    payload = verify_test_access(
        model_lock_path=args.model_lock,
        repository_root=args.repository_root,
        required_tag=args.required_tag,
        brain_encoder_gate_path=args.brain_encoder_gate,
    )
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
