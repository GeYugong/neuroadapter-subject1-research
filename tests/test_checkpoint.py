from __future__ import annotations

import json

import pytest
import torch

from neuroadapter_research.checkpoint import (
    load_distributed_checkpoint,
    prune_full_checkpoints,
    save_inference_snapshot,
    save_distributed_checkpoint,
    verify_checkpoint,
    verify_inference_snapshot,
)


def no_barrier() -> None:
    return None


def test_checkpoint_round_trip(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = save_distributed_checkpoint(
        output_dir=tmp_path,
        update=17,
        rank=0,
        world_size=1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        trainer_state={"next_update": 17, "accumulation_step": 0},
        rank_state={"torch_cpu": torch.get_rng_state()},
        barrier=no_barrier,
    )
    assert output.name == "checkpoint-update-00000017"
    manifest = verify_checkpoint(output, expected_world_size=1)
    assert manifest["optimizer_update"] == 17
    loaded = load_distributed_checkpoint(output, rank=0, world_size=1)
    assert loaded["trainer"]["next_update"] == 17
    assert set(loaded["model"]) == set(model.state_dict())


def test_checkpoint_rejects_tampering(tmp_path) -> None:
    output = save_distributed_checkpoint(
        output_dir=tmp_path,
        update=1,
        rank=0,
        world_size=1,
        model_state={"weight": torch.ones(1)},
        optimizer_state={},
        trainer_state={"next_update": 1},
        rank_state={},
        barrier=no_barrier,
    )
    trainer_path = output / "trainer_state.json"
    payload = json.loads(trainer_path.read_text(encoding="utf-8"))
    payload["next_update"] = 2
    trainer_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_checkpoint(output)


def test_checkpoint_refuses_overwrite(tmp_path) -> None:
    kwargs = dict(
        output_dir=tmp_path,
        update=1,
        rank=0,
        world_size=1,
        model_state={},
        optimizer_state={},
        trainer_state={},
        rank_state={},
        barrier=no_barrier,
    )
    save_distributed_checkpoint(**kwargs)
    with pytest.raises(FileExistsError):
        save_distributed_checkpoint(**kwargs)


def test_checkpoint_quarantines_stale_incomplete(tmp_path) -> None:
    stale = tmp_path / ".checkpoint-update-00000003.incomplete"
    stale.mkdir()
    (stale / "partial").write_text("incomplete", encoding="utf-8")
    output = save_distributed_checkpoint(
        output_dir=tmp_path,
        update=3,
        rank=0,
        world_size=1,
        model_state={"value": torch.tensor([1.0])},
        optimizer_state={"state": {}},
        trainer_state={"next_update": 3},
        rank_state={"rank": 0},
        barrier=no_barrier,
    )
    assert output.is_dir()
    assert len(list((tmp_path / "corrupt").iterdir())) == 1


def test_inference_snapshot_and_full_retention(tmp_path) -> None:
    snapshot = save_inference_snapshot(
        tmp_path / "snapshots",
        25,
        {"value": torch.tensor([2.0])},
        {"optimizer_update": 25},
    )
    assert (snapshot / "COMPLETE").is_file()
    assert verify_inference_snapshot(snapshot)["optimizer_update"] == 25

    checkpoints = tmp_path / "checkpoints"
    for update in (1, 2, 3):
        save_distributed_checkpoint(
            output_dir=checkpoints,
            update=update,
            rank=0,
            world_size=1,
            model_state={"value": torch.tensor([float(update)])},
            optimizer_state={"state": {}},
            trainer_state={"next_update": update},
            rank_state={"rank": 0},
            barrier=no_barrier,
        )
    removed = prune_full_checkpoints(checkpoints, keep_latest=2)
    assert [path.name for path in removed] == ["checkpoint-update-00000001"]


def test_inference_snapshot_rejects_tampering(tmp_path) -> None:
    snapshot = save_inference_snapshot(
        tmp_path / "snapshots",
        5,
        {"value": torch.tensor([2.0])},
        {"optimizer_update": 5},
    )
    (snapshot / "metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_inference_snapshot(snapshot)
