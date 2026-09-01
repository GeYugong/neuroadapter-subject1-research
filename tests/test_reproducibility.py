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
