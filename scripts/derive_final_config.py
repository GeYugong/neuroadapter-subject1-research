#!/usr/bin/env python3
"""Derive the only allowed final-run config from frozen selection evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from neuroadapter_research.approval import validate_final_transition
from neuroadapter_research.atomic import write_bytes_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import method_fingerprint


def contained_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-pool-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    if args.output_config.exists():
        raise FileExistsError(f"final config already exists: {args.output_config}")

    selection = load_training_config(args.selection_config, require_frozen=True)
    if selection.raw["run_kind"] != "selection":
        raise ValueError("source config is not a selection config")
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selected_update = int(manifest.get("selected_update_u_star", -1))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "final_selection"
        or manifest.get("status") != "complete"
        or manifest.get("config_sha256") != selection.sha256
        or manifest.get("method_fingerprint") != method_fingerprint(selection)
        or selected_update <= 0
    ):
        raise ValueError("selection manifest is not bound to the source config")

    root = Path(selection.raw["project_root"])
    payload = yaml.safe_load(args.selection_config.read_text(encoding="utf-8"))
    payload["run_name"] = f"subject01-final-v1-u{selected_update:08d}"
    payload["run_kind"] = "final"
    payload["paths"]["split_ids"] = contained_relative(args.train_pool_ids, root)
    payload["paths"]["selection_config"] = contained_relative(
        args.selection_config, root
    )
    payload["paths"]["selection_manifest"] = contained_relative(
        args.selection_manifest, root
    )
    payload["paths"]["output_dir"] = contained_relative(args.output_dir, root)
    payload["training"]["max_updates"] = selected_update
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(
        args.output_config,
        yaml.safe_dump(payload, sort_keys=False).encode("utf-8"),
    )
    final = load_training_config(args.output_config, require_frozen=True)
    validate_final_transition(selection, final)
    if method_fingerprint(selection) != method_fingerprint(final):
        raise ValueError("derived final config changed the method fingerprint")
    print(
        json.dumps(
            {
                "status": "derived",
                "selected_update_u_star": selected_update,
                "selection_config_sha256": selection.sha256,
                "final_config_sha256": final.sha256,
                "method_fingerprint": method_fingerprint(final),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
