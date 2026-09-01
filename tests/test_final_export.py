from __future__ import annotations

import torch

from scripts.export_final_model import flatten_state


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
