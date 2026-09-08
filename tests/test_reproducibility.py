from __future__ import annotations

import numpy as np
import torch

from neuroadapter_research.reproducibility import structural_sha256


def test_structural_hash_is_order_independent_for_mappings() -> None:
    left = {"tensor": torch.arange(3), "array": np.arange(4), "value": [1, 2]}
    right = {"value": [1, 2], "array": np.arange(4), "tensor": torch.arange(3)}
    assert structural_sha256(left) == structural_sha256(right)
    right["tensor"][0] = 9
    assert structural_sha256(left) != structural_sha256(right)


def test_structural_hash_supports_scalar_optimizer_steps() -> None:
    left = {"step": torch.tensor(50.0), "moment": torch.arange(3.0)}
    assert structural_sha256(left) == structural_sha256(
        {"moment": torch.arange(3.0), "step": torch.tensor(50.0)}
    )
    assert structural_sha256(left) != structural_sha256(
        {"step": torch.tensor(51.0), "moment": torch.arange(3.0)}
    )
    assert structural_sha256(torch.tensor(50.0)) != structural_sha256(torch.tensor([50.0]))


def test_structural_hash_supports_scalar_bfloat16_and_empty_tensor() -> None:
    assert len(structural_sha256(torch.tensor(1.0, dtype=torch.bfloat16))) == 64
    assert len(structural_sha256(torch.empty(0))) == 64
