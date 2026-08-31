from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.build_training_cache import pad_parcels
from scripts.prepare_nsd_subj01 import summarize_nonfinite
from scripts.verify_nsd_subj01 import nonfinite_scan


def test_source_nonfinite_values_are_described_without_replacement() -> None:
    array = np.zeros((3, 5), dtype=np.float32)
    array[:, 4] = np.nan

    summary = summarize_nonfinite(array, Path("rh.betas_session11.mgh"))

    assert summary is not None
    assert summary["nonfinite_count"] == 3
    assert summary["nan_count"] == 3
    assert summary["affected_trial_indices_zero_based"] == [0, 1, 2]
    assert summary["affected_vertex_indices_zero_based"] == [4]
    assert np.isnan(array[:, 4]).all()


def test_full_scan_allows_only_unselected_source_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "betas.h5"
    with h5py.File(path, "w") as h5:
        data = np.zeros((4, 8), dtype=np.float32)
        data[2, 6] = np.nan
        h5.create_dataset("rh_betas", data=data)

    selected = np.zeros(8, dtype=np.bool_)
    selected[1] = True
    with h5py.File(path, "r") as h5:
        summary = nonfinite_scan(h5["rh_betas"], selected, chunk_size=2)
    assert summary["nonfinite_count"] == 1
    assert summary["affected_rows_zero_based"] == [2]
    assert summary["affected_vertices_zero_based"] == [6]

    selected[6] = True
    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="selected"):
        nonfinite_scan(h5["rh_betas"], selected, chunk_size=2)


def test_training_cache_rejects_nonfinite_selected_vertex() -> None:
    average = np.zeros(8, dtype=np.float32)
    average[6] = np.nan
    parcels = [torch.tensor([1, 6], dtype=torch.int64)]

    with pytest.raises(ValueError, match="selected parcel"):
        pad_parcels(average, parcels, maximum=2)
