"""Immutable protocol bindings shared by training and model selection."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from .atomic import sha256_file

if TYPE_CHECKING:
    from .config import LoadedTrainingConfig


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FIXED_GATE_REQUIREMENTS: dict[str, Any] = {
    "schema_version": 1,
    "required_cuda_arch": "sm_120",
    "required_compute_capability": [12, 0],
    "required_world_size": 2,
    "required_gpu_name": "NVIDIA GeForce RTX 5090",
    "required_bf16": True,
    "forward_atol": 1.0e-6,
    "batch_minimum_updates": 532,
    "stress_minimum_seconds": 1800,
    "max_reserved_memory_bytes": 31_138_512_896,
    "require_xid_check": True,
}
SELECTION_FIXED_VALUES: dict[str, Any] = {
    "schema_version": 1,
    "subject": 1,
    "protocol_namespace": "subject01-selection-v1",
    "screening_candidates": 2,
    "final_candidates": 8,
    "denoising_steps": 50,
    "guidance_scale": 4.0,
    "validation_loss_draws": 1,
    "validation_loss_batch_size": 8,
    "evaluation_batch_size": 16,
    "bootstrap_draws": 10_000,
    "bootstrap_seed": 20260901,
}
METHOD_PATHS = (
    "model_manifest",
    "raw_nsd_manifest",
    "training_cache_manifest",
    "training_cache_verification",
    "canonical_initialization",
    "canonical_manifest",
    "data_fingerprint",
    "nsd_image_mapping",
    "decoder_atlas_audit",
    "split_manifest",
    "source_manifest",
    "environment_lock",
    "selection_plan",
    "gate_requirements",
    "evaluation_manifest",
)


@dataclass(frozen=True)
class LoadedSelectionPlan:
    path: Path
    sha256: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LoadedGateRequirements:
    path: Path
    sha256: str
    raw: dict[str, Any]


def _require_exact_keys(payload: dict[str, Any], expected: set[str], context: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _read_mapping(path: Path, *, yaml_document: bool = False) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml_document else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"protocol document must be a mapping: {path}")
    return payload


def load_gate_requirements(path: Path) -> LoadedGateRequirements:
    path = Path(path).resolve()
    payload = _read_mapping(path, yaml_document=True)
    if payload != FIXED_GATE_REQUIREMENTS:
        raise ValueError("gate requirements differ from the protocol implementation")
    return LoadedGateRequirements(path=path, sha256=sha256_file(path), raw=payload)


def load_selection_plan(path: Path, *, require_frozen: bool) -> LoadedSelectionPlan:
    path = Path(path).resolve()
    payload = _read_mapping(path)
    expected = {
        *SELECTION_FIXED_VALUES,
        "status",
        "expected_snapshot_updates",
        "validation_ids_sha256",
        "image_order_sha256",
        "metric_sources",
    }
    _require_exact_keys(payload, expected, "selection plan")
    for name, expected_value in SELECTION_FIXED_VALUES.items():
        if payload[name] != expected_value:
            raise ValueError(f"selection plan {name} differs from the frozen protocol")
    if require_frozen and payload["status"] != "frozen":
        raise ValueError("formal selection requires a frozen selection plan")
    if payload["status"] not in {"draft", "frozen"}:
        raise ValueError("selection plan status must be draft or frozen")

    updates = payload["expected_snapshot_updates"]
    if (
        not isinstance(updates, list)
        or len(updates) != 20
        or any(not isinstance(value, int) or value <= 0 for value in updates)
        or updates != sorted(set(updates))
    ):
        raise ValueError("selection plan must contain 20 sorted unique snapshot updates")
    for name in ("validation_ids_sha256", "image_order_sha256"):
        if not isinstance(payload[name], str) or not SHA256_PATTERN.fullmatch(payload[name]):
            raise ValueError(f"selection plan has an invalid {name}")
    sources = payload["metric_sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("selection plan must bind metric source files")
    for relative, digest in sources.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"metric source path is not contained: {relative}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"metric source has an invalid SHA-256: {relative}")
    return LoadedSelectionPlan(path=path, sha256=sha256_file(path), raw=payload)


def read_ordered_ids(path: Path) -> list[int]:
    values = [int(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"ID file must be non-empty and unique: {path}")
    return values


def image_order_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", value))
    return digest.hexdigest()


def validate_selection_plan_inputs(
    plan: LoadedSelectionPlan,
    *,
    validation_ids_path: Path,
    repository_root: Path,
) -> dict[str, str]:
    validation_ids_path = Path(validation_ids_path)
    file_digest = sha256_file(validation_ids_path)
    if file_digest != plan.raw["validation_ids_sha256"]:
        raise ValueError("validation ID file differs from the selection plan")
    order_digest = image_order_sha256(read_ordered_ids(validation_ids_path))
    if order_digest != plan.raw["image_order_sha256"]:
        raise ValueError("validation image order differs from the selection plan")
    repository_root = Path(repository_root).resolve()
    for relative, expected in plan.raw["metric_sources"].items():
        if sha256_file(repository_root / relative) != expected:
            raise ValueError(f"metric source differs from the selection plan: {relative}")
    return {
        "selection_plan_sha256": plan.sha256,
        "validation_ids_sha256": file_digest,
        "image_order_sha256": order_digest,
        "metric_implementation_sha256": hashlib.sha256(
            json.dumps(
                plan.raw["metric_sources"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }


def method_fingerprint_payload(config: "LoadedTrainingConfig") -> dict[str, Any]:
    training = {
        name: value
        for name, value in config.training.items()
        if name != "max_updates"
    }
    return {
        "schema_version": 1,
        "subject": config.raw["subject"],
        "protocol_commit": config.raw["protocol_commit"],
        "training_method": training,
        "input_sha256": {
            name: sha256_file(config.paths[name]) for name in METHOD_PATHS
        },
    }


def method_fingerprint(config: "LoadedTrainingConfig") -> str:
    payload = method_fingerprint_payload(config)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_protocol_repository(repository_root: Path, expected_commit: str) -> str:
    repository_root = Path(repository_root).resolve()
    observed = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != expected_commit:
        raise ValueError(
            f"repository HEAD {observed} differs from protocol commit {expected_commit}"
        )
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
        raise ValueError("formal protocol execution requires a clean Git worktree")
    return observed


def validate_selection_config_and_plan(
    config: "LoadedTrainingConfig", plan: LoadedSelectionPlan
) -> None:
    if config.raw["run_kind"] != "selection":
        raise ValueError("model selection accepts only a selection-run config")
    if sha256_file(config.paths["selection_plan"]) != plan.sha256:
        raise ValueError("loaded selection plan differs from the training config")
    if int(config.training["max_updates"]) != int(plan.raw["expected_snapshot_updates"][-1]):
        raise ValueError("selection max_updates differs from the final planned snapshot")


def validate_selection_snapshot(
    snapshot_path: Path,
    *,
    config: "LoadedTrainingConfig",
    plan: LoadedSelectionPlan,
    fingerprint: str,
) -> dict[str, Any]:
    from .checkpoint import load_inference_snapshot_provenance

    provenance = load_inference_snapshot_provenance(snapshot_path)
    metadata = provenance["metadata"]
    required = {
        "schema_version": 1,
        "run_mode": "formal",
        "run_kind": "selection",
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        "selection_plan_sha256": plan.sha256,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"selection snapshot has invalid {name}")
    update = int(metadata.get("optimizer_update", -1))
    if update not in plan.raw["expected_snapshot_updates"]:
        raise ValueError("snapshot update is absent from the selection plan")
    approval = metadata.get("formal_approval_sha256")
    if not isinstance(approval, str) or not SHA256_PATTERN.fullmatch(approval):
        raise ValueError("selection snapshot has no formal approval binding")
    return provenance
