"""Atomic, content-addressed checkpoints for exact distributed resumption."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

from .atomic import fsync_directory, sha256_file, write_bytes_atomic, write_json_atomic


SCHEMA_VERSION = 1
Barrier = Callable[[], None]


def _torch_save(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_torch_file(path: Path) -> None:
    torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_name(update: int) -> str:
    if update < 0:
        raise ValueError("update must be non-negative")
    return f"checkpoint-update-{update:08d}"


def save_distributed_checkpoint(
    output_dir: Path,
    update: int,
    rank: int,
    world_size: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    trainer_state: dict[str, Any],
    rank_state: dict[str, Any],
    barrier: Barrier,
) -> Path:
    """Save one checkpoint at an optimizer boundary.

    Every rank writes its own RNG state. Rank zero writes shared state, verifies
    every file, hashes the directory contents, and atomically publishes it.
    """

    if rank < 0 or rank >= world_size:
        raise ValueError("rank is outside world_size")
    output_dir = Path(output_dir)
    final = output_dir / _checkpoint_name(update)
    temporary = output_dir / f".{_checkpoint_name(update)}.incomplete"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise FileExistsError(f"checkpoint path already exists: {final} or {temporary}")
        temporary.mkdir()
        fsync_directory(output_dir)
    barrier()

    rank_path = temporary / f"rank-{rank:05d}.pt"
    _torch_save(rank_path, rank_state)
    _verify_torch_file(rank_path)
    barrier()

    if rank == 0:
        _torch_save(temporary / "model.pt", model_state)
        _torch_save(temporary / "optimizer.pt", optimizer_state)
        write_json_atomic(temporary / "trainer_state.json", trainer_state)
        _verify_torch_file(temporary / "model.pt")
        _verify_torch_file(temporary / "optimizer.pt")

        expected_rank_files = [f"rank-{value:05d}.pt" for value in range(world_size)]
        for name in expected_rank_files:
            if not (temporary / name).is_file():
                raise RuntimeError(f"rank state is missing: {name}")
            _verify_torch_file(temporary / name)

        artifact_names = [
            "model.pt",
            "optimizer.pt",
            "trainer_state.json",
            *expected_rank_files,
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "optimizer_update": update,
            "world_size": world_size,
            "files": {
                name: {
                    "size": (temporary / name).stat().st_size,
                    "sha256": sha256_file(temporary / name),
                }
                for name in artifact_names
            },
        }
        write_json_atomic(temporary / "MANIFEST.json", manifest)
        manifest_sha256 = sha256_file(temporary / "MANIFEST.json")
        write_bytes_atomic(
            temporary / "COMPLETE", f"{manifest_sha256}  MANIFEST.json\n".encode("ascii")
        )
        fsync_directory(temporary)
        os.replace(temporary, final)
        fsync_directory(output_dir)
    barrier()
    if not final.is_dir():
        raise RuntimeError(f"checkpoint was not published: {final}")
    return final


def verify_checkpoint(path: Path, expected_world_size: int | None = None) -> dict[str, Any]:
    path = Path(path)
    complete_path = path / "COMPLETE"
    manifest_path = path / "MANIFEST.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"incomplete checkpoint: {path}")
    fields = complete_path.read_text(encoding="ascii").strip().split()
    if fields != [sha256_file(manifest_path), "MANIFEST.json"]:
        raise ValueError("checkpoint COMPLETE marker does not match MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if expected_world_size is not None and manifest.get("world_size") != expected_world_size:
        raise ValueError("checkpoint world size does not match the current run")
    for name, record in manifest["files"].items():
        artifact = path / name
        if not artifact.is_file():
            raise ValueError(f"checkpoint artifact is missing: {name}")
        if artifact.stat().st_size != record["size"]:
            raise ValueError(f"checkpoint artifact size mismatch: {name}")
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"checkpoint artifact hash mismatch: {name}")
    return manifest


def load_distributed_checkpoint(
    path: Path, rank: int, world_size: int
) -> dict[str, Any]:
    manifest = verify_checkpoint(path, expected_world_size=world_size)
    rank_path = Path(path) / f"rank-{rank:05d}.pt"
    if not rank_path.is_file():
        raise ValueError(f"checkpoint has no state for rank {rank}")
    return {
        "manifest": manifest,
        "model": torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False),
        "optimizer": torch.load(
            Path(path) / "optimizer.pt", map_location="cpu", weights_only=False
        ),
        "trainer": json.loads(
            (Path(path) / "trainer_state.json").read_text(encoding="utf-8")
        ),
        "rank": torch.load(rank_path, map_location="cpu", weights_only=False),
    }
