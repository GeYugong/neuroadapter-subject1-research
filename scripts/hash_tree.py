#!/usr/bin/env python3
"""Create a deterministic SHA-256 inventory for a file tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-directory", action="append", default=[])
    args = parser.parse_args()

    root = args.root.resolve()
    files = []
    excluded = set(args.exclude_directory)
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        if excluded.intersection(path.relative_to(root).parts):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )

    payload = {"root": str(root), "files": files, "total_bytes": sum(item["size"] for item in files)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
