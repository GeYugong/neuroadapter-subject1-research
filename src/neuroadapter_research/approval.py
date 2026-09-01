"""Exact selection and final approval payloads for formal training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .atomic import sha256_file
from .config import LoadedTrainingConfig, load_training_config
from .integrity import load_json_mapping, validate_gate_artifact, validate_subject1_audits
from .protocol import (
    image_order_sha256,
    load_gate_requirements,
    load_selection_plan,
    method_fingerprint,
    read_ordered_ids,
    validate_selection_config_and_plan,
    validate_selection_plan_inputs,
)


GATE_PATHS = (
    "hardware_gate",
    "forward_alignment",
    "batch_gate",
    "resume_equivalence",
    "decode_determinism",
    "evaluator_repeatability",
)


def validate_final_transition(
    selection: LoadedTrainingConfig, final: LoadedTrainingConfig
) -> None:
    left = deepcopy(selection.raw)
    right = deepcopy(final.raw)
    for payload in (left, right):
        payload.pop("run_name")
        payload.pop("run_kind")
        for name in ("split_ids", "selection_config", "selection_manifest", "output_dir"):
            payload["paths"].pop(name)
        payload["training"].pop("max_updates")
    if left != right:
        raise ValueError("selection and final configs differ outside approved run fields")


def _validate_gate_set(
    config: LoadedTrainingConfig,
    fingerprint: str,
    *,
    expected_source_config_sha256: str,
) -> dict[str, str]:
    requirements = load_gate_requirements(config.paths["gate_requirements"])
    hashes = {}
    for name in GATE_PATHS:
        payload = validate_gate_artifact(
            config.paths[name],
            expected_gate=name,
            method_fingerprint=fingerprint,
            gate_requirements=requirements,
        )
        if payload["config_sha256"] != expected_source_config_sha256:
            raise ValueError(f"formal gate comes from a different config: {name}")
        hashes[f"{name}_sha256"] = sha256_file(config.paths[name])
    return hashes


def expected_approval_payload(config: LoadedTrainingConfig) -> dict[str, Any]:
    fingerprint = method_fingerprint(config)
    canonical = load_json_mapping(config.paths["canonical_manifest"])
    if canonical.get("status") != "frozen":
        raise ValueError("canonical initialization is not frozen")
    validate_subject1_audits(config.paths)
    plan = load_selection_plan(config.paths["selection_plan"], require_frozen=True)
    plan_binding = validate_selection_plan_inputs(
        plan,
        validation_ids_path=config.paths["validation_ids"],
        repository_root=Path(__file__).resolve().parents[2],
    )
    split_ids = read_ordered_ids(config.paths["split_ids"])
    final_fields: dict[str, Any] = {}

    if config.raw["run_kind"] == "selection":
        validate_selection_config_and_plan(config, plan)
        if len(split_ids) != 8500:
            raise ValueError("selection approval requires exactly 8500 training IDs")
        gate_source_config_sha256 = config.sha256
    else:
        if len(split_ids) != 9000:
            raise ValueError("final approval requires exactly 9000 train-pool IDs")
        split_manifest = load_json_mapping(config.paths["split_manifest"])
        if image_order_sha256(split_ids) != split_manifest.get("train_pool_sha256"):
            raise ValueError("final split does not match the frozen 9000-image train pool")
        selection_config = load_training_config(
            config.paths["selection_config"], require_frozen=True
        )
        validate_final_transition(selection_config, config)
        selection_fingerprint = method_fingerprint(selection_config)
        if selection_fingerprint != fingerprint:
            raise ValueError("selection and final configs use different methods")
        selection_manifest = json.loads(
            config.paths["selection_manifest"].read_text(encoding="utf-8")
        )
        selected_update = int(selection_manifest.get("selected_update_u_star", -1))
        if (
            selection_manifest.get("schema_version") != 1
            or selection_manifest.get("stage") != "final_selection"
            or selection_manifest.get("status") != "complete"
            or selection_manifest.get("config_sha256") != selection_config.sha256
            or selection_manifest.get("method_fingerprint") != fingerprint
            or selection_manifest.get("selection_plan_sha256") != plan.sha256
            or selected_update != int(config.training["max_updates"])
        ):
            raise ValueError("final selection manifest does not authorize this final config")
        gate_source_config_sha256 = selection_config.sha256
        final_fields = {
            "selection_config_sha256": selection_config.sha256,
            "final_selection_manifest_sha256": sha256_file(
                config.paths["selection_manifest"]
            ),
            "selected_update_u_star": selected_update,
        }
    gate_hashes = _validate_gate_set(
        config,
        fingerprint,
        expected_source_config_sha256=gate_source_config_sha256,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "approved": True,
        "approval_kind": config.raw["run_kind"],
        "config_sha256": config.sha256,
        "method_fingerprint": fingerprint,
        "protocol_commit": config.raw["protocol_commit"],
        "gate_source_config_sha256": gate_source_config_sha256,
        "environment_lock_sha256": sha256_file(config.paths["environment_lock"]),
        "gate_requirements_sha256": sha256_file(config.paths["gate_requirements"]),
        **gate_hashes,
        "data_fingerprint_sha256": sha256_file(config.paths["data_fingerprint"]),
        "model_assets_manifest_sha256": sha256_file(config.paths["model_manifest"]),
        "canonical_initialization_sha256": sha256_file(
            config.paths["canonical_initialization"]
        ),
        "training_cache_verification_sha256": sha256_file(
            config.paths["training_cache_verification"]
        ),
        "nsd_image_mapping_sha256": sha256_file(config.paths["nsd_image_mapping"]),
        "decoder_atlas_audit_sha256": sha256_file(
            config.paths["decoder_atlas_audit"]
        ),
        "selection_plan_sha256": plan.sha256,
        "validation_ids_sha256": plan_binding["validation_ids_sha256"],
        "split_ids_sha256": sha256_file(config.paths["split_ids"]),
        "split_image_order_sha256": image_order_sha256(split_ids),
        "split_image_count": len(split_ids),
        **final_fields,
    }
    return payload
