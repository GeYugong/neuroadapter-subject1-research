from __future__ import annotations

import torch

from neuroadapter_research.trainer import (
    min_snr_weights,
    reference_epoch_checkpoints,
    token_keep_mask,
)


def test_reference_epoch_schedule_uses_first_complete_update() -> None:
    checkpoints = reference_epoch_checkpoints(
        sample_count=8500,
        global_batch_size=16,
        interval=25,
        max_updates=265625,
    )
    assert min(checkpoints) == 13282
    assert max(checkpoints) == 265625
    assert len(checkpoints) == 20


def test_min_snr_weighting() -> None:
    alphas = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
    weights = min_snr_weights(torch.tensor([0, 1, 2]), alphas, gamma=5.0)
    expected = torch.tensor([1.0, 5.0 / 9.0, 5.0 / 99.0])
    torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-7)


def test_token_dropout_is_generator_replayable() -> None:
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    left = token_keep_mask(8, 200, torch.device("cpu"), first)
    right = token_keep_mask(8, 200, torch.device("cpu"), second)
    assert left.shape == (8, 200, 1)
    assert left.dtype == torch.bool
    torch.testing.assert_close(left, right, rtol=0, atol=0)
