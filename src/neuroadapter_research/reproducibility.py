"""Structural hashes and comparisons for deterministic gate artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .atomic import sha256_file
from .checkpoint import load_distributed_checkpoint, verify_checkpoint


def structural_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}\0".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"ndarray:{array.dtype}:{array.shape}\0".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, repr(candidate))):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}\0".encode())
            for child in item:
                update(child)
        elif isinstance(item, bytes):
            digest.update(f"bytes:{len(item)}\0".encode())
            digest.update(item)
        elif isinstance(item, Path):
            update(item.as_posix())
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(
                (type(item).__name__ + ":" + json.dumps(item, sort_keys=True) + "\0").encode()
            )
        else:
            raise TypeError(f"unsupported deterministic state value: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def compare_resume_outputs(
    left: Path, right: Path, left_traces: Path, right_traces: Path
) -> dict[str, Any]:
    left_manifest = verify_checkpoint(left)
    right_manifest = verify_checkpoint(right)
    if left_manifest["optimizer_update"] != right_manifest["optimizer_update"]:
        raise ValueError("resume comparison checkpoints use different updates")
    if left_manifest["world_size"] != right_manifest["world_size"]:
        raise ValueError("resume comparison checkpoints use different world sizes")
    world_size = int(left_manifest["world_size"])
    comparisons: dict[str, str] = {}
    for rank in range(world_size):
        left_state = load_distributed_checkpoint(left, rank, world_size)
        right_state = load_distributed_checkpoint(right, rank, world_size)
        for name in ("model", "optimizer", "trainer", "rank"):
            left_hash = structural_sha256(left_state[name])
            right_hash = structural_sha256(right_state[name])
            if left_hash != right_hash:
                raise ValueError(f"resume state differs for rank {rank}: {name}")
            comparisons[f"rank_{rank:05d}_{name}"] = left_hash
    trace_hashes = {}
    for rank in range(world_size):
        name = f"trace-rank-{rank:05d}.jsonl"
        left_path = Path(left_traces) / name
        right_path = Path(right_traces) / name
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(f"resume trace is missing: {name}")
        left_hash = sha256_file(left_path)
        if left_hash != sha256_file(right_path):
            raise ValueError(f"resume trace differs: {name}")
        trace_hashes[name] = left_hash
    return {
        "optimizer_update": int(left_manifest["optimizer_update"]),
        "world_size": world_size,
        "structural_state_sha256": comparisons,
        "trace_sha256": trace_hashes,
    }


def verify_decode_tree(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("split") != "validation":
        raise ValueError("decode manifest is not a complete validation artifact")
    root = Path(manifest_path).parent
    file_count = 0
    for record in payload.get("records", []):
        for artifact in record.get("files", []):
            relative = Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("decode manifest contains an uncontained path")
            path = root / relative
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"decoded PNG hash mismatch: {relative}")
            file_count += 1
    expected = int(payload["image_count"]) * int(payload["candidate_count"])
    if file_count != expected:
        raise ValueError("decode manifest has the wrong number of PNG artifacts")
    return payload
