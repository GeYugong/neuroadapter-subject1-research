from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from neuroadapter_research.approval import validate_final_transition
from neuroadapter_research.config import LoadedTrainingConfig
from neuroadapter_research.protocol import (
    FIXED_GATE_REQUIREMENTS,
    SELECTION_FIXED_VALUES,
    image_order_sha256,
    load_gate_requirements,
    load_selection_plan,
    validate_selection_plan_inputs,
)


def test_gate_requirements_cannot_be_weakened(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump(FIXED_GATE_REQUIREMENTS), encoding="utf-8")
    assert load_gate_requirements(path).raw["stress_minimum_seconds"] == 1800
    weakened = dict(FIXED_GATE_REQUIREMENTS)
    weakened["stress_minimum_seconds"] = 1
    path.write_text(yaml.safe_dump(weakened), encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        load_gate_requirements(path)


def test_selection_plan_binds_validation_order_and_metric_sources(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    source = repository / "metric.py"
    source.parent.mkdir()
    source.write_text("fixed\n", encoding="utf-8")
    ids = tmp_path / "validation_ids.txt"
    ids.write_text("3\n7\n", encoding="utf-8")
    payload = {
        **SELECTION_FIXED_VALUES,
        "status": "frozen",
        "expected_snapshot_updates": list(range(1, 21)),
        "validation_ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest(),
        "image_order_sha256": image_order_sha256([3, 7]),
        "metric_sources": {
            "metric.py": hashlib.sha256(source.read_bytes()).hexdigest()
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_selection_plan(plan_path, require_frozen=True)
    binding = validate_selection_plan_inputs(
        plan, validation_ids_path=ids, repository_root=repository
    )
    assert binding["image_order_sha256"] == payload["image_order_sha256"]
    ids.write_text("7\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ID file differs"):
        validate_selection_plan_inputs(
            plan, validation_ids_path=ids, repository_root=repository
        )


def test_final_transition_allows_only_run_specific_fields() -> None:
    selection_raw = {
        "run_name": "selection",
        "run_kind": "selection",
        "paths": {
            "split_ids": "selection_train_ids.txt",
            "selection_config": "selection.yaml",
            "selection_manifest": "selection.json",
            "output_dir": "selection-run",
            "fixed": "same",
        },
        "training": {"max_updates": 20, "learning_rate": 1e-4},
    }
    final_raw = json.loads(json.dumps(selection_raw))
    final_raw["run_name"] = "final"
    final_raw["run_kind"] = "final"
    final_raw["paths"]["split_ids"] = "train_pool_ids.txt"
    final_raw["paths"]["output_dir"] = "final-run"
    final_raw["training"]["max_updates"] = 10
    selection = LoadedTrainingConfig(Path("selection"), "a" * 64, selection_raw)
    final = LoadedTrainingConfig(Path("final"), "b" * 64, final_raw)
    validate_final_transition(selection, final)
    final_raw["training"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="outside approved"):
        validate_final_transition(
            selection, LoadedTrainingConfig(Path("final"), "b" * 64, final_raw)
        )
