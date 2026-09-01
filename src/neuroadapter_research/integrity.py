"""Content verification for immutable research inputs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .atomic import sha256_file
from .protocol import LoadedGateRequirements


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
VENDOR_SOURCES = {
    "neuroadapter": "vendor/NeuroAdapter",
    "whole_brain_encoder": "vendor/whole_brain_encoder",
    "openai_clip": "vendor/CLIP",
    "swav": "vendor/swav",
    "dinov2": "vendor/dinov2",
}


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be a mapping: {path}")
    return payload


def validate_gate_artifact(
    path: Path,
    *,
    expected_gate: str,
    method_fingerprint: str,
    gate_requirements: LoadedGateRequirements,
) -> dict[str, Any]:
    payload = load_json_mapping(path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"formal gate has an unsupported schema: {expected_gate}")
    if payload.get("gate") != expected_gate:
        raise ValueError(f"formal gate identity mismatch: {expected_gate}")
    if payload.get("status") not in {"passed", "verified"}:
        raise ValueError(f"formal gate has not passed: {expected_gate}")
    if payload.get("method_fingerprint") != method_fingerprint:
        raise ValueError(f"formal gate uses a different method: {expected_gate}")
    if payload.get("gate_requirements_sha256") != gate_requirements.sha256:
        raise ValueError(f"formal gate uses different requirements: {expected_gate}")
    config_sha256 = payload.get("config_sha256")
    if not isinstance(config_sha256, str) or not SHA256_PATTERN.fullmatch(config_sha256):
        raise ValueError(f"formal gate has no source config identity: {expected_gate}")

    requirements = gate_requirements.raw
    if expected_gate == "hardware_gate":
        if (
            payload.get("required_cuda_arch") != requirements["required_cuda_arch"]
            or payload.get("native_arch_available") is not True
            or float(payload.get("stress_duration_seconds", -1))
            < requirements["stress_minimum_seconds"]
            or payload.get("xid_check_passed") is not True
        ):
            raise ValueError("hardware gate evidence is weaker than the requirements")
        ranks = payload.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != requirements["required_world_size"]:
            raise ValueError("hardware gate has the wrong world size")
        for record in ranks:
            if (
                record.get("device_name") != requirements["required_gpu_name"]
                or record.get("compute_capability")
                != requirements["required_compute_capability"]
                or record.get("bf16_supported") is not requirements["required_bf16"]
                or record.get("bf16_matmul_finite") is not True
                or record.get("bf16_conv_backward_finite") is not True
                or record.get("nccl_all_reduce_verified") is not True
            ):
                raise ValueError("hardware rank evidence does not meet requirements")
        inventory = payload.get("system_evidence", {}).get("nvidia_smi")
        if not isinstance(inventory, list) or len(inventory) != requirements[
            "required_world_size"
        ]:
            raise ValueError("hardware gate has incomplete nvidia-smi inventory")
        uuids = [record.get("uuid") for record in inventory]
        if (
            any(record.get("name") != requirements["required_gpu_name"] for record in inventory)
            or any(not record.get("driver_version") for record in inventory)
            or any(not value for value in uuids)
            or len(set(uuids)) != len(uuids)
        ):
            raise ValueError("hardware gate has invalid GPU identity evidence")
    elif expected_gate == "forward_alignment":
        tolerance = float(payload.get("absolute_tolerance", -1))
        if tolerance != requirements["forward_atol"]:
            raise ValueError("forward gate tolerance differs from requirements")
        if (
            float(payload.get("prediction_max_abs_error", float("inf"))) > tolerance
            or float(payload.get("loss_abs_error", float("inf"))) > tolerance
        ):
            raise ValueError("forward gate errors exceed the frozen tolerance")
    elif expected_gate == "batch_gate":
        if payload.get("minimum_updates") != requirements["batch_minimum_updates"]:
            raise ValueError("batch gate duration differs from requirements")
        selected_name = payload.get("selection")
        selected = payload.get(selected_name) if isinstance(selected_name, str) else None
        if (
            not isinstance(selected, dict)
            or int(selected.get("maximum_memory_reserved_bytes", 2**63))
            > requirements["max_reserved_memory_bytes"]
        ):
            raise ValueError("selected batch geometry exceeds the memory limit")
    return payload


def validate_subject1_audits(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    cache = load_json_mapping(paths["training_cache_verification"])
    if cache.get("schema_version") != 1 or cache.get("status") != "verified":
        raise ValueError("training cache verification has not passed")
    expected_cache = {
        "cache_sha256": sha256_file(paths["training_cache"]),
        "build_manifest_sha256": sha256_file(paths["training_cache_manifest"]),
        "data_fingerprint_sha256": sha256_file(paths["data_fingerprint"]),
    }
    if any(cache.get(name) != value for name, value in expected_cache.items()):
        raise ValueError("training cache verification is not bound to current inputs")

    mapping = load_json_mapping(paths["nsd_image_mapping"])
    expected_checks = {
        "train_shared1000_false",
        "test_shared1000_true",
        "all_subject1_true",
        "presentation_order_equal",
    }
    checks = mapping.get("checks")
    if (
        mapping.get("schema_version") != 1
        or mapping.get("status") != "verified"
        or not isinstance(checks, dict)
        or set(checks) != expected_checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("NSD image mapping audit has not passed")

    decoder_atlas = load_json_mapping(paths["decoder_atlas_audit"])
    hemispheres = decoder_atlas.get("hemispheres")
    if (
        decoder_atlas.get("schema_version") != 1
        or decoder_atlas.get("gate") != "decoder_atlas"
        or decoder_atlas.get("status") != "verified"
        or decoder_atlas.get("surface_space") != "fsaverage"
        or decoder_atlas.get("model_token_count") != 200
        or decoder_atlas.get("max_voxels") != cache.get("max_voxels")
        or decoder_atlas.get("top_snr_ranking_verified") is not True
        or not isinstance(hemispheres, dict)
        or set(hemispheres) != {"lh", "rh"}
    ):
        raise ValueError("decoder atlas audit has not passed")
    runtime = decoder_atlas.get("runtime_inputs", {})
    if (
        runtime.get("cache_manifest_sha256")
        != sha256_file(paths["training_cache_manifest"])
        or runtime.get("parcel_map_sha256") != cache.get("parcel_map_sha256")
    ):
        raise ValueError("decoder atlas audit is not bound to the training cache")
    return {
        "training_cache": cache,
        "nsd_image_mapping": mapping,
        "decoder_atlas": decoder_atlas,
    }


def _normalized_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest path must be a non-empty string")
    normalized = value.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"manifest path is not relative and contained: {value}")
    return candidate.as_posix()


def validate_tree_manifest(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("tree manifest must contain a non-empty files list")
    indexed: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            raise ValueError("tree manifest records require path, size, and sha256")
        path = _normalized_relative_path(raw["path"])
        size = raw["size"]
        digest = raw["sha256"]
        if path in indexed:
            raise ValueError(f"duplicate manifest path: {path}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid manifest size for {path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"invalid manifest SHA-256 for {path}")
        indexed[path] = {"path": path, "size": size, "sha256": digest}
        total_bytes += size
    if "total_bytes" in payload and int(payload["total_bytes"]) != total_bytes:
        raise ValueError("tree manifest total_bytes does not match its file records")
    return indexed


def verify_file_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    size = path.stat().st_size
    if size != int(record["size"]):
        raise ValueError(f"input size mismatch: {path}")
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise ValueError(f"input SHA-256 mismatch: {path}")
    return {"size": size, "sha256": digest}


def verify_file_against_manifest(
    path: Path, manifest_path: Path, manifest_relative_path: str
) -> dict[str, Any]:
    payload = load_json_mapping(manifest_path)
    indexed = validate_tree_manifest(payload)
    relative = _normalized_relative_path(manifest_relative_path)
    if relative not in indexed:
        raise ValueError(f"file is absent from manifest: {relative}")
    result = verify_file_record(path, indexed[relative])
    return {"manifest_path": relative, **result}


def tree_records_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_tree_against_manifest(
    root: Path,
    manifest_path: Path,
    *,
    manifest_prefix: str = "",
    reject_extra: bool = True,
    ignored_extra_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"required input tree is missing: {root}")
    indexed = validate_tree_manifest(load_json_mapping(manifest_path))
    prefix = _normalized_relative_path(manifest_prefix).rstrip("/") if manifest_prefix else ""
    selected: dict[str, dict[str, Any]] = {}
    prefix_marker = prefix + "/" if prefix else ""
    for path, record in indexed.items():
        if path.startswith(prefix_marker):
            local = path[len(prefix_marker) :]
            if local:
                selected[local] = {**record, "path": local}
    if not selected:
        raise ValueError(f"manifest has no records below prefix {prefix!r}")

    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(selected)
    missing = sorted(expected - observed)
    normalized_ignored = tuple(
        _normalized_relative_path(value).rstrip("/") for value in ignored_extra_prefixes
    )
    extra = sorted(
        path
        for path in observed - expected
        if not any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in normalized_ignored
        )
    )
    if missing or (reject_extra and extra):
        raise ValueError(
            f"input tree differs from manifest: missing={missing[:10]}, extra={extra[:10]}"
        )
    verified = []
    for relative in sorted(expected):
        result = verify_file_record(root / relative, selected[relative])
        verified.append({"path": relative, **result})
    return {
        "file_count": len(verified),
        "total_bytes": sum(record["size"] for record in verified),
        "tree_sha256": tree_records_sha256(verified),
    }


def verify_submodule_heads(
    repository_root: Path, source_manifest_path: Path
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    sources = load_json_mapping(source_manifest_path).get("sources")
    if not isinstance(sources, dict):
        raise ValueError("source manifest has no sources mapping")
    records = []
    for source_name, relative_path in VENDOR_SOURCES.items():
        source = sources.get(source_name)
        if not isinstance(source, dict):
            raise ValueError(f"source manifest is missing {source_name}")
        expected = source.get("commit")
        if not isinstance(expected, str) or not COMMIT_PATTERN.fullmatch(expected):
            raise ValueError(f"source {source_name} has no full commit")
        path = repository_root / relative_path
        observed = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if observed != expected:
            raise ValueError(
                f"submodule {source_name} is at {observed}, expected {expected}"
            )
        records.append({"source": source_name, "path": relative_path, "commit": observed})
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"submodules": records, "submodule_heads_sha256": digest}
