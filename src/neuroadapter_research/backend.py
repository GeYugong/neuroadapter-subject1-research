"""Explicit PyTorch backend configuration shared by training and evaluation."""

from __future__ import annotations

from typing import Any

import torch


def configure_torch_backend(training: dict[str, Any]) -> dict[str, bool]:
    settings = {
        name: bool(training[name])
        for name in (
            "allow_tf32",
            "cudnn_benchmark",
            "deterministic_algorithms",
            "adamw_fused",
            "adamw_foreach",
        )
    }
    torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["allow_tf32"]
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = settings["deterministic_algorithms"]
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
    return settings
