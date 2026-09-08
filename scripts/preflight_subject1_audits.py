#!/usr/bin/env python3
"""Check real data audit interfaces before spending GPU time on gates."""

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.integrity import validate_subject1_audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_training_config(args.config, require_frozen=True)
    results = validate_subject1_audits(config.paths)
    payload = {
        "status": "verified",
        "purpose": "CPU data audit interfaces only; not a formal approval",
        "config_sha256": config.sha256,
        "audits": {name: result["status"] for name, result in results.items()},
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
