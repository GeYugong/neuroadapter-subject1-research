#!/usr/bin/env python3
"""Snapshot a clean repository so reporting commits cannot alter a running protocol."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from neuroadapter_research.protocol import verify_protocol_repository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("runtime name must be one directory component")
    root = args.project_root.resolve()
    source = root / "repo"
    if not (source / ".git").is_dir():
        raise ValueError("runtime snapshot requires a standalone repository")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    verify_protocol_repository(source, commit)
    destination = root / "runtime" / args.name
    destination.parent.mkdir(exist_ok=True)
    shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
    verify_protocol_repository(destination, commit)
    print(f"runtime={destination}\nprotocol_commit={commit}")


if __name__ == "__main__":
    main()
