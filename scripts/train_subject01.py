#!/usr/bin/env python3
"""CLI for gate or formally approved Subject 1 training runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroadapter_research.config import load_training_config
from neuroadapter_research.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("gate", "formal"), required=True)
    parser.add_argument("--max-updates-override", type=int)
    parser.add_argument("--output-override", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--trace-updates", type=int, default=100)
    args = parser.parse_args()
    config = load_training_config(
        args.config, require_frozen=args.run_mode == "formal"
    )
    run_training(
        config=config,
        run_mode=args.run_mode,
        max_updates_override=args.max_updates_override,
        output_override=args.output_override,
        resume=args.resume,
        approval_path=args.approval_file,
        trace_updates=args.trace_updates,
    )


if __name__ == "__main__":
    main()
