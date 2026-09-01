from __future__ import annotations

import torch

from scripts.export_final_model import flatten_state, validate_final_run_evidence


def test_flatten_state_has_stable_names() -> None:
    state = {
        "image_proj": {"weight": torch.ones(1)},
        "ip_adapter": {"0.weight": torch.ones(2)},
        "guidance_generator": {"bias": torch.zeros(1)},
    }
    assert list(flatten_state(state)) == [
        "guidance_generator.bias",
        "image_proj.weight",
        "ip_adapter.0.weight",
    ]


def test_final_run_evidence_rejects_selection_snapshot(monkeypatch) -> None:
    class Config:
        sha256 = "a" * 64
        training = {"max_updates": 100}

    monkeypatch.setattr("scripts.export_final_model.method_fingerprint", lambda _config: "b" * 64)
    metadata = {
        "optimizer_update": 100,
        "run_mode": "formal",
        "run_kind": "selection",
        "config_sha256": "a" * 64,
        "method_fingerprint": "b" * 64,
        "formal_approval_sha256": "c" * 64,
    }
    status = {
        "status": "completed",
        "formal_training": True,
        "run_mode": "formal",
        "run_kind": "final",
        "config_sha256": "a" * 64,
        "method_fingerprint": "b" * 64,
        "formal_approval_sha256": "c" * 64,
        "max_updates": 100,
        "last_completed_update": 100,
    }
    import pytest

    with pytest.raises(ValueError, match="run_kind"):
        validate_final_run_evidence(
            config=Config(),
            selection={"selected_update_u_star": 100},
            snapshot_metadata=metadata,
            run_status=status,
            approval_sha256="c" * 64,
        )
