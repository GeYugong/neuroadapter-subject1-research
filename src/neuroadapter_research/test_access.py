"""Model-lock and Git gates for standard-test access."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .atomic import sha256_file


def verify_test_access(
    *,
    model_lock_path: Path,
    repository_root: Path,
    required_tag: str,
    brain_encoder_gate_path: Path,
) -> dict[str, Any]:
    model_lock_path = Path(model_lock_path).resolve()
    repository_root = Path(repository_root).resolve()
    lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked":
        raise ValueError("model lock is not finalized")
    brain_gate_path = Path(brain_encoder_gate_path).resolve()
    brain_gate = json.loads(brain_gate_path.read_text(encoding="utf-8"))
    if (
        brain_gate.get("gate") != "brain_encoder_forward"
        or brain_gate.get("status") != "passed"
        or not brain_gate.get("full_forward_verified")
    ):
        raise ValueError("brain encoder full-forward gate has not passed")
    brain_bindings = {
        "brain_encoder_asset_manifest_sha256": "brain_encoder_assets_sha256",
        "brain_encoder_parcel_audit_sha256": "brain_encoder_parcel_audit_sha256",
        "whole_brain_encoder_commit": "whole_brain_encoder_commit",
        "dinov2_commit": "dinov2_commit",
        "dinov2_weight_sha256": "dinov2_weight_sha256",
        "environment_lock_sha256": "environment_lock_sha256",
        "source_manifest_sha256": "source_manifest_sha256",
        "evaluation_manifest_sha256": "evaluation_manifest_sha256",
        "repository_commit": "repository_commit",
    }
    for gate_name, lock_name in brain_bindings.items():
        if brain_gate.get(gate_name) != lock.get(lock_name):
            raise ValueError(f"brain encoder gate differs from model lock: {gate_name}")
    for name, record in lock.get("files", {}).items():
        path = model_lock_path.parent / name
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise ValueError(f"locked model file is missing or has wrong size: {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"locked model file hash mismatch: {name}")
    head = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tag = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-list", "-n", "1", required_tag],
        text=True,
    ).strip()
    tag_type = subprocess.check_output(
        ["git", "-C", str(repository_root), "cat-file", "-t", required_tag],
        text=True,
    ).strip()
    if tag_type != "tag":
        raise ValueError("standard-test access requires an annotated release tag")
    if head != lock["repository_commit"] or tag != head:
        raise ValueError("required release tag, Git HEAD, and model lock commit differ")
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        text=True,
    )
    if status:
        raise ValueError("standard-test access requires a clean Git worktree")
    return {
        "status": "authorized",
        "model_lock_sha256": sha256_file(model_lock_path),
        "repository_commit": head,
        "release_tag": required_tag,
        "brain_encoder_gate_sha256": sha256_file(brain_gate_path),
    }
