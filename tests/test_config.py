from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neuroadapter_research.config import load_training_config


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "training"
    / "subject01_selection.template.yaml"
)


def test_draft_template_is_valid_for_gate_runs() -> None:
    config = load_training_config(TEMPLATE, require_frozen=False)
    assert config.training["global_batch_size"] == 16
    assert config.raw["run_kind"] == "selection"


def test_draft_template_is_rejected_for_formal_runs() -> None:
    with pytest.raises(ValueError, match="status: frozen"):
        load_training_config(TEMPLATE, require_frozen=True)


def test_test_split_is_rejected(tmp_path) -> None:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    payload["paths"]["split_ids"] = "data/derived/splits/test_ids.txt"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="standard test"):
        load_training_config(path, require_frozen=False)


def test_paths_are_resolved_below_project_root() -> None:
    config = load_training_config(TEMPLATE, require_frozen=False)
    assert config.paths["stimuli"] == Path(config.raw["project_root"]) / config.raw["paths"]["stimuli"]


def test_absolute_individual_path_is_rejected(tmp_path) -> None:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    payload["paths"]["stimuli"] = "/tmp/nsd_stimuli.hdf5"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="relative to project_root"):
        load_training_config(path, require_frozen=False)


def test_parent_traversal_is_rejected(tmp_path) -> None:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    payload["paths"]["stimuli"] = "../outside/nsd_stimuli.hdf5"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="relative to project_root"):
        load_training_config(path, require_frozen=False)
