"""Lazy HDF5 dataset used by the frozen Subject 1 training protocol."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


def load_id_file(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"invalid image ID file: {path}")
    if np.unique(values).size != values.size:
        raise ValueError(f"duplicate image IDs in {path}")
    return values


class Subject1TrainingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache_path: Path,
        stimuli_path: Path,
        split_ids_path: Path,
        image_size: int = 512,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.stimuli_path = Path(stimuli_path)
        self.split_ids_path = Path(split_ids_path)
        self.image_size = image_size
        self._cache: h5py.File | None = None
        self._stimuli: h5py.File | None = None

        split_ids = load_id_file(self.split_ids_path)
        with h5py.File(self.cache_path, "r") as cache:
            cache_ids = np.asarray(cache["image_ids"], dtype=np.int64)
            brain_shape = tuple(int(value) for value in cache["brain"].shape)
            cache_kind = str(cache.attrs.get("cache_kind", ""))
        if cache_kind != "subject01_train_pool_top100_per_hemisphere":
            raise ValueError(f"unexpected cache kind: {cache_kind!r}")
        if cache_ids.shape != (9000,) or np.unique(cache_ids).size != 9000:
            raise ValueError("training cache must contain 9000 unique image IDs")
        if brain_shape[0] != 9000 or brain_shape[1] != 200:
            raise ValueError(f"unexpected brain cache shape: {brain_shape}")

        row_by_id = {int(image_id): row for row, image_id in enumerate(cache_ids)}
        missing = [int(image_id) for image_id in split_ids if int(image_id) not in row_by_id]
        if missing:
            raise ValueError(f"split contains IDs absent from the train cache: {missing[:10]}")
        self.image_ids = split_ids
        self.cache_rows = np.asarray([row_by_id[int(value)] for value in split_ids])
        self.num_parcels = brain_shape[1]
        self.max_voxels = brain_shape[2]

        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(
                    image_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return int(self.image_ids.size)

    def _ensure_open(self) -> None:
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, "r")
        if self._stimuli is None:
            self._stimuli = h5py.File(self.stimuli_path, "r")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._ensure_open()
        assert self._cache is not None
        assert self._stimuli is not None
        image_id = int(self.image_ids[index])
        cache_row = int(self.cache_rows[index])
        brain = np.asarray(self._cache["brain"][cache_row], dtype=np.float32)
        image = np.asarray(self._stimuli["imgBrick"][image_id])
        return {
            "nsd_image_id": torch.tensor(image_id, dtype=torch.int64),
            "brain": torch.from_numpy(brain),
            "image": self.image_transform(image),
        }

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        if self._stimuli is not None:
            self._stimuli.close()
            self._stimuli = None

    def __del__(self) -> None:
        self.close()
