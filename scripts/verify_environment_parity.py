#!/usr/bin/env python3
"""Check relocated distribution versions without treating editable paths as packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {}
    for line in args.source_freeze.read_text().splitlines():
        if not line.strip() or line.startswith(("#", "-e ")):
            continue
        requirement = Requirement(line)
        expected[canonicalize_name(requirement.name)] = requirement
    actual: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            actual.setdefault(canonicalize_name(name), set()).add(distribution.version)
    errors = []
    for name, requirement in expected.items():
        versions = actual.get(name, set())
        if len(versions) != 1 or not all(version in requirement.specifier for version in versions):
            errors.append({"name": name, "expected": str(requirement.specifier), "actual": sorted(versions)})
    payload = {
        "schema_version": 1,
        "status": "failed" if errors else "verified",
        "source_freeze_sha256": sha256_file(args.source_freeze),
        "pinned_distribution_count": len(expected),
        "errors": errors,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
