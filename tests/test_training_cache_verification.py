from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.verify_training_cache import scan_brain_dataset


def write_cache(path: Path, values: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("brain", data=values)


def test_cache_scan_accepts_finite_values_and_zero_padding(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    values = np.zeros((3, 2, 4), dtype=np.float32)
    values[:, :, :2] = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    write_cache(path, values)
    mask = np.zeros((2, 4), dtype=np.bool_)
    mask[:, :2] = True

    with h5py.File(path, "r") as h5:
        summary = scan_brain_dataset(h5["brain"], mask, chunk_size=2)

    assert summary["valid_value_count"] == 12
    assert summary["minimum"] == 0.0
    assert summary["maximum"] == 11.0


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_cache_scan_rejects_nonfinite_selected_values(
    tmp_path: Path, value: float
) -> None:
    path = tmp_path / "cache.h5"
    values = np.zeros((1, 1, 2), dtype=np.float32)
    values[0, 0, 0] = value
    write_cache(path, values)
    mask = np.array([[True, False]])

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="NaN/Inf"):
        scan_brain_dataset(h5["brain"], mask)


def test_cache_scan_rejects_nonzero_padding(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    values = np.zeros((1, 1, 2), dtype=np.float32)
    values[0, 0, 1] = 1.0
    write_cache(path, values)
    mask = np.array([[True, False]])

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="padding"):
        scan_brain_dataset(h5["brain"], mask)
