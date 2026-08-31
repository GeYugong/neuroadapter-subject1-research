import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.verify_brain_encoder_assets import verify_model_pair


def write_assets(
    root: Path,
    *,
    model_value: float = 1.0,
    correlation_shape: tuple[int, ...] = (8,),
) -> tuple[Path, Path]:
    checkpoint = root / "checkpoint_nonavg.pth"
    correlation = root / "lh_val_corr_nonavg.npy"
    torch.save(
        {
            "model": {"weight": torch.tensor([model_value])},
            "args": argparse.Namespace(subj=1),
        },
        checkpoint,
    )
    np.save(correlation, np.ones(correlation_shape, dtype=np.float32))
    return checkpoint, correlation


def test_verify_model_pair_accepts_expected_assets(tmp_path: Path) -> None:
    checkpoint, correlation = write_assets(tmp_path)

    summary = verify_model_pair(checkpoint, correlation, expected_vertices=8)

    assert summary["model_tensor_count"] == 1
    assert summary["model_value_count"] == 1
    assert summary["correlation_shape"] == [8]
    assert summary["correlation_nonfinite_count"] == 0


def test_verify_model_pair_rejects_nonfinite_weights(tmp_path: Path) -> None:
    checkpoint, correlation = write_assets(tmp_path, model_value=float("nan"))

    with pytest.raises(ValueError, match="NaN/Inf"):
        verify_model_pair(checkpoint, correlation, expected_vertices=8)


def test_verify_model_pair_rejects_wrong_correlation_shape(tmp_path: Path) -> None:
    checkpoint, correlation = write_assets(tmp_path, correlation_shape=(7,))

    with pytest.raises(ValueError, match="voxel-confidence shape"):
        verify_model_pair(checkpoint, correlation, expected_vertices=8)
