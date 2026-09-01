#!/usr/bin/env python3
"""Export and lock the one final Subject 1 inference state."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.inference import load_inference_state


def flatten_state(state: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    expected = {"image_proj", "ip_adapter", "guidance_generator"}
    if set(state) != expected:
        raise ValueError("final state has unexpected groups")
    return {
        f"{group}.{name}": tensor.detach().cpu().contiguous()
        for group in sorted(state)
        for name, tensor in sorted(state[group].items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--final-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("final export directory must be empty")

    config = load_training_config(args.config, require_frozen=True)
    if config.raw["run_kind"] != "final":
        raise ValueError("final exporter requires a final-run config")
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selected_update = int(selection["selected_update_u_star"])
    if not args.final_snapshot.is_dir():
        raise ValueError("final snapshot must be an atomic snapshot directory")
    metadata = json.loads((args.final_snapshot / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["optimizer_update"]) != selected_update:
        raise ValueError("final snapshot update differs from selected U*")

    state = load_inference_state(args.final_snapshot)
    flat = flatten_state(state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = args.output_dir / "subject01_final.pt"
    safetensors_path = args.output_dir / "subject01_final.safetensors"
    shutil.copyfile(args.final_snapshot / "model.pt", pt_path)
    save_file(flat, safetensors_path, metadata={"subject": "1", "architecture": "linear_projection"})
    reloaded = load_file(safetensors_path, device="cpu")
    if set(reloaded) != set(flat):
        raise ValueError("safetensors key set differs after reload")
    for name, tensor in flat.items():
        torch.testing.assert_close(reloaded[name], tensor, rtol=0, atol=0)

    hashes = {
        "subject01_final.pt": sha256_file(pt_path),
        "subject01_final.safetensors": sha256_file(safetensors_path),
    }
    (args.output_dir / "subject01_final.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="ascii",
    )
    repository = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": 1,
        "status": "locked",
        "subject": 1,
        "architecture": "linear_projection",
        "optimizer_update_u_star": selected_update,
        "images_seen": 16 * selected_update,
        "config_sha256": config.sha256,
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "final_snapshot_manifest_sha256": sha256_file(args.final_snapshot / "MANIFEST.json"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {
            name: {"size": (args.output_dir / name).stat().st_size, "sha256": digest}
            for name, digest in hashes.items()
        },
        "safetensors_reload_bitwise_equal": True,
    }
    write_json_atomic(args.output_dir / "MODEL_LOCK.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
