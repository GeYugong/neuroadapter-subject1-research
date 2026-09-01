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

from neuroadapter_research.approval import expected_approval_payload
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.checkpoint import load_inference_snapshot_provenance
from neuroadapter_research.config import load_training_config
from neuroadapter_research.inference import load_inference_state
from neuroadapter_research.protocol import method_fingerprint, read_ordered_ids


def validate_final_run_evidence(
    *,
    config,
    selection: dict,
    snapshot_metadata: dict,
    run_status: dict,
    approval_sha256: str,
) -> int:
    selected_update = int(selection.get("selected_update_u_star", -1))
    fingerprint = method_fingerprint(config)
    expected_metadata = {
        "optimizer_update": selected_update,
        "run_mode": "formal",
        "run_kind": "final",
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        "formal_approval_sha256": approval_sha256,
    }
    for name, expected in expected_metadata.items():
        if snapshot_metadata.get(name) != expected:
            raise ValueError(f"final snapshot has invalid {name}")
    expected_status = {
        "status": "completed",
        "formal_training": True,
        "run_mode": "formal",
        "run_kind": "final",
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        "formal_approval_sha256": approval_sha256,
        "max_updates": selected_update,
        "last_completed_update": selected_update,
    }
    for name, expected in expected_status.items():
        if run_status.get(name) != expected:
            raise ValueError(f"final run status has invalid {name}")
    if int(config.training["max_updates"]) != selected_update:
        raise ValueError("final config max_updates differs from selected U*")
    return selected_update


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
    parser.add_argument("--final-approval", type=Path, required=True)
    parser.add_argument("--final-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("final export directory must be empty")

    config = load_training_config(args.config, require_frozen=True)
    if config.raw["run_kind"] != "final":
        raise ValueError("final exporter requires a final-run config")
    approval = json.loads(args.final_approval.read_text(encoding="utf-8"))
    if approval != expected_approval_payload(config):
        raise ValueError("final approval does not match the final config")
    approval_sha256 = sha256_file(args.final_approval)
    selection = json.loads(config.paths["selection_manifest"].read_text(encoding="utf-8"))
    if not args.final_snapshot.is_dir():
        raise ValueError("final snapshot must be an atomic snapshot directory")
    snapshot = load_inference_snapshot_provenance(args.final_snapshot)
    run_status_path = config.paths["output_dir"] / "run_status.json"
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    selected_update = validate_final_run_evidence(
        config=config,
        selection=selection,
        snapshot_metadata=snapshot["metadata"],
        run_status=run_status,
        approval_sha256=approval_sha256,
    )
    split_ids = read_ordered_ids(config.paths["split_ids"])
    if len(split_ids) != 9000 or snapshot["metadata"].get("split_ids_sha256") != sha256_file(
        config.paths["split_ids"]
    ):
        raise ValueError("final snapshot is not bound to the complete train pool")

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
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config.raw["protocol_commit"]:
        raise ValueError("final export Git HEAD differs from the protocol commit")
    source_manifest = json.loads(config.paths["source_manifest"].read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        config.paths["evaluation_manifest"].read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "status": "locked",
        "subject": 1,
        "architecture": "linear_projection",
        "optimizer_update_u_star": selected_update,
        "images_seen": 16 * selected_update,
        "config_sha256": config.sha256,
        "method_fingerprint": method_fingerprint(config),
        "final_approval_sha256": approval_sha256,
        "selection_manifest_sha256": sha256_file(config.paths["selection_manifest"]),
        "final_snapshot_manifest_sha256": sha256_file(args.final_snapshot / "MANIFEST.json"),
        "final_snapshot_model_sha256": snapshot["model_sha256"],
        "final_snapshot_metadata_sha256": snapshot["metadata_sha256"],
        "final_run_status_sha256": sha256_file(run_status_path),
        "train_pool_ids_sha256": sha256_file(config.paths["split_ids"]),
        "environment_lock_sha256": sha256_file(config.paths["environment_lock"]),
        "source_manifest_sha256": sha256_file(config.paths["source_manifest"]),
        "evaluation_manifest_sha256": sha256_file(config.paths["evaluation_manifest"]),
        "brain_encoder_assets_sha256": sha256_file(config.paths["brain_encoder_assets"]),
        "brain_encoder_parcel_audit_sha256": sha256_file(
            config.paths["brain_encoder_parcel_audit"]
        ),
        "whole_brain_encoder_commit": source_manifest["sources"][
            "whole_brain_encoder"
        ]["commit"],
        "dinov2_commit": source_manifest["sources"]["dinov2"]["commit"],
        "dinov2_weight_sha256": evaluation_manifest["files"]["dinov2_vitb14"][
            "sha256"
        ],
        "repository_commit": head,
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
