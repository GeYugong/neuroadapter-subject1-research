import pytest
import torch

from scripts.diagnose_condition_path import NoiseReplay, choose_pairs, prepare_noise


def test_frozen_selection_and_whole_sample_derangement():
    pairs = choose_pairs(range(100), "train")
    assert pairs == choose_pairs(reversed(range(100)), "train")
    assert len(pairs) == 32
    assert all(p["image_id"] != p["donor_id"] for p in pairs)
    assert {p["image_id"] for p in pairs} == {p["donor_id"] for p in pairs}
    assert pairs != choose_pairs(range(100), "validation")


def test_noise_replay_preserves_values_between_precisions():
    values = [torch.randn(2, 4, 3, 3).bfloat16() for _ in range(3)]
    replay = NoiseReplay(values)
    for value in values:
        actual = replay.take(value.shape, "cpu", torch.float32)
        assert torch.equal(actual, value.float())
    replay.complete()
    with pytest.raises(AssertionError):
        replay.take(values[0].shape, "cpu", torch.float32)


def test_noise_replay_rejects_shape_and_unconsumed_draws():
    replay = NoiseReplay([torch.zeros(2, 4)])
    with pytest.raises(AssertionError):
        replay.complete()
    with pytest.raises(AssertionError):
        replay.take((1, 4), "cpu", torch.float32)


def test_noise_bank_preserves_current_mixed_precision_draws():
    from types import SimpleNamespace
    scheduler = SimpleNamespace(timesteps=[1, 0], set_timesteps=lambda *a, **k: None)
    bank = prepare_noise(SimpleNamespace(noise_scheduler=scheduler), 1, "train", torch.device("cpu"))
    assert [value.dtype for value in bank] == [torch.bfloat16, torch.float32, torch.bfloat16]
