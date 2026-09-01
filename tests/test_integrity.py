from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from neuroadapter_research.integrity import (
    validate_gate_artifact,
    validate_subject1_audits,
    verify_file_against_manifest,
    verify_tree_against_manifest,
)
from neuroadapter_research.protocol import FIXED_GATE_REQUIREMENTS, load_gate_requirements


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"files": records, "total_bytes": sum(int(x["size"]) for x in records)}),
        encoding="utf-8",
    )


def test_file_and_tree_verification(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    payload = b"fixed"
    (root / "asset.bin").write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [{"path": "prefix/asset.bin", "size": len(payload), "sha256": digest(payload)}],
    )

    record = verify_file_against_manifest(root / "asset.bin", manifest, "prefix/asset.bin")
    tree = verify_tree_against_manifest(root, manifest, manifest_prefix="prefix")
    assert record["sha256"] == digest(payload)
    assert tree["file_count"] == 1


def test_tree_verification_rejects_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "asset.bin").write_bytes(b"fixed")
    (root / "extra.bin").write_bytes(b"extra")
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [{"path": "asset.bin", "size": 5, "sha256": digest(b"fixed")}],
    )
    with pytest.raises(ValueError, match="differs from manifest"):
        verify_tree_against_manifest(root, manifest)

    cache = root / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "metadata").write_text("ignored", encoding="utf-8")
    (root / "extra.bin").unlink()
    verified = verify_tree_against_manifest(
        root, manifest, ignored_extra_prefixes=(".cache/huggingface",)
    )
    assert verified["file_count"] == 1


def test_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [{"path": "../asset.bin", "size": 1, "sha256": digest(b"x")}],
    )
    with pytest.raises(ValueError, match="not relative"):
        verify_file_against_manifest(tmp_path / "asset.bin", manifest, "../asset.bin")


def test_gate_artifact_is_bound_to_identity_and_method(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.yaml"
    requirements_path.write_text(
        yaml.safe_dump(FIXED_GATE_REQUIREMENTS, sort_keys=False), encoding="utf-8"
    )
    requirements = load_gate_requirements(requirements_path)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "decode_determinism",
                "status": "passed",
                "config_sha256": "a" * 64,
                "method_fingerprint": "b" * 64,
                "gate_requirements_sha256": requirements.sha256,
            }
        ),
        encoding="utf-8",
    )
    assert validate_gate_artifact(
        gate,
        expected_gate="decode_determinism",
        method_fingerprint="b" * 64,
        gate_requirements=requirements,
    )["status"] == "passed"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_gate_artifact(
            gate,
            expected_gate="evaluator_repeatability",
            method_fingerprint="b" * 64,
            gate_requirements=requirements,
        )


def test_hardware_gate_requires_gpu_identity_inventory(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.yaml"
    requirements_path.write_text(
        yaml.safe_dump(FIXED_GATE_REQUIREMENTS, sort_keys=False), encoding="utf-8"
    )
    requirements = load_gate_requirements(requirements_path)
    rank = {
        "device_name": FIXED_GATE_REQUIREMENTS["required_gpu_name"],
        "compute_capability": [12, 0],
        "bf16_supported": True,
        "bf16_matmul_finite": True,
        "bf16_conv_backward_finite": True,
        "nccl_all_reduce_verified": True,
    }
    payload = {
        "schema_version": 1,
        "gate": "hardware_gate",
        "status": "passed",
        "config_sha256": "a" * 64,
        "method_fingerprint": "b" * 64,
        "gate_requirements_sha256": requirements.sha256,
        "required_cuda_arch": "sm_120",
        "native_arch_available": True,
        "stress_duration_seconds": 1800.0,
        "xid_check_passed": True,
        "ranks": [dict(rank), dict(rank)],
        "system_evidence": {
            "nvidia_smi": [
                {
                    "uuid": "GPU-a",
                    "name": FIXED_GATE_REQUIREMENTS["required_gpu_name"],
                    "driver_version": "999.0",
                },
                {
                    "uuid": "GPU-b",
                    "name": FIXED_GATE_REQUIREMENTS["required_gpu_name"],
                    "driver_version": "999.0",
                },
            ]
        },
    }
    gate = tmp_path / "hardware.json"
    gate.write_text(json.dumps(payload), encoding="utf-8")
    validate_gate_artifact(
        gate,
        expected_gate="hardware_gate",
        method_fingerprint="b" * 64,
        gate_requirements=requirements,
    )
    payload["system_evidence"]["nvidia_smi"][1]["uuid"] = "GPU-a"
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU identity"):
        validate_gate_artifact(
            gate,
            expected_gate="hardware_gate",
            method_fingerprint="b" * 64,
            gate_requirements=requirements,
        )


def test_subject1_audits_require_verified_decoder_atlas(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "training_cache",
            "training_cache_manifest",
            "training_cache_verification",
            "data_fingerprint",
            "nsd_image_mapping",
            "decoder_atlas_audit",
        )
    }
    paths["training_cache"].write_bytes(b"cache")
    paths["training_cache_manifest"].write_bytes(b"manifest")
    paths["data_fingerprint"].write_bytes(b"fingerprint")
    paths["training_cache_verification"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified",
                "cache_sha256": digest(b"cache"),
                "build_manifest_sha256": digest(b"manifest"),
                "data_fingerprint_sha256": digest(b"fingerprint"),
                "max_voxels": 626,
                "parcel_map_sha256": "p" * 64,
            }
        ),
        encoding="utf-8",
    )
    paths["nsd_image_mapping"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified",
                "checks": {
                    "train_shared1000_false": True,
                    "test_shared1000_true": True,
                    "all_subject1_true": True,
                    "presentation_order_equal": True,
                },
            }
        ),
        encoding="utf-8",
    )
    paths["decoder_atlas_audit"].write_text(
        json.dumps({"schema_version": 1, "gate": "decoder_atlas", "status": "blocked"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has not passed"):
        validate_subject1_audits(paths)

    paths["decoder_atlas_audit"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "decoder_atlas",
                "status": "verified",
                "surface_space": "fsaverage",
                "model_token_count": 200,
                "max_voxels": 626,
                "top_snr_ranking_verified": True,
                "hemispheres": {
                    "lh": {},
                    "rh": {},
                },
                "runtime_inputs": {
                    "cache_manifest_sha256": digest(b"manifest"),
                    "parcel_map_sha256": "p" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    assert validate_subject1_audits(paths)["training_cache"]["status"] == "verified"
