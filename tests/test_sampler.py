from __future__ import annotations

import numpy as np
import pytest

from neuroadapter_research.sampler import DeterministicDistributedBatchPlan


def make_plan(micro_batch_size: int = 8, accumulation_steps: int = 1):
    return DeterministicDistributedBatchPlan(
        sample_count=17,
        global_batch_size=16,
        world_size=2,
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        seed=20260901,
    )


def test_global_batches_have_fixed_size_and_cross_epoch_without_drops() -> None:
    plan = make_plan()
    batches = [plan.global_indices(update) for update in range(17)]
    stream = np.concatenate(batches)
    assert all(batch.shape == (16,) for batch in batches)
    assert sorted(stream[:17].tolist()) == list(range(17))
    assert sorted(stream[17:34].tolist()) == list(range(17))


def test_ranks_partition_the_global_batch() -> None:
    plan = make_plan()
    for update in range(5):
        rank_zero = np.concatenate(plan.local_micro_batches(update, 0))
        rank_one = np.concatenate(plan.local_micro_batches(update, 1))
        np.testing.assert_array_equal(
            np.concatenate([rank_zero, rank_one]), plan.global_indices(update)
        )


def test_accumulation_changes_only_micro_batch_partition() -> None:
    direct = make_plan(micro_batch_size=8, accumulation_steps=1)
    accumulated = make_plan(micro_batch_size=4, accumulation_steps=2)
    for update in range(10):
        for rank in range(2):
            np.testing.assert_array_equal(
                np.concatenate(direct.local_micro_batches(update, rank)),
                np.concatenate(accumulated.local_micro_batches(update, rank)),
            )


def test_state_round_trip_detects_tampering() -> None:
    plan = make_plan()
    payload = plan.state_before_update(123).to_dict()
    assert plan.validate_state(payload).next_update == 123
    payload["cursor"] = int(payload["cursor"]) + 1
    with pytest.raises(ValueError, match="frozen plan"):
        plan.validate_state(payload)


def test_invalid_global_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="global batch mismatch"):
        DeterministicDistributedBatchPlan(
            sample_count=17,
            global_batch_size=32,
            world_size=2,
            micro_batch_size=8,
            accumulation_steps=1,
            seed=1,
        )
