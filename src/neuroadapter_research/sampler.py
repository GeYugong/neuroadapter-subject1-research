"""Stateless distributed sample planning with exact restart semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


def _epoch_seed(base_seed: int, epoch: int) -> int:
    material = f"neuroadapter-subject1|sampler|{base_seed}|{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _int64_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


@dataclass(frozen=True)
class SamplerState:
    next_update: int
    images_seen: int
    epoch: int
    cursor: int
    permutation: tuple[int, ...]
    permutation_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "next_update": self.next_update,
            "images_seen": self.images_seen,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "permutation": list(self.permutation),
            "permutation_sha256": self.permutation_sha256,
        }


class DeterministicDistributedBatchPlan:
    """Map optimizer updates to fixed global batches and rank-local micro-batches.

    The stream is an infinite concatenation of independent epoch permutations. A
    global batch may cross an epoch boundary, so no image is dropped and every
    optimizer update has the same number of samples.
    """

    def __init__(
        self,
        sample_count: int,
        global_batch_size: int,
        world_size: int,
        micro_batch_size: int,
        accumulation_steps: int,
        seed: int,
    ) -> None:
        values = {
            "sample_count": sample_count,
            "global_batch_size": global_batch_size,
            "world_size": world_size,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        expected = world_size * micro_batch_size * accumulation_steps
        if global_batch_size != expected:
            raise ValueError(
                "global batch mismatch: "
                f"{global_batch_size} != {world_size} * {micro_batch_size} * "
                f"{accumulation_steps}"
            )
        self.sample_count = sample_count
        self.global_batch_size = global_batch_size
        self.world_size = world_size
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps = accumulation_steps
        self.seed = seed

    @property
    def local_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    def epoch_permutation(self, epoch: int) -> np.ndarray:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        generator = np.random.Generator(np.random.PCG64(_epoch_seed(self.seed, epoch)))
        return generator.permutation(self.sample_count).astype(np.int64, copy=False)

    def global_indices(self, update: int) -> np.ndarray:
        if update < 0:
            raise ValueError("update must be non-negative")
        position = update * self.global_batch_size
        remaining = self.global_batch_size
        pieces: list[np.ndarray] = []
        while remaining:
            epoch, cursor = divmod(position, self.sample_count)
            permutation = self.epoch_permutation(epoch)
            take = min(remaining, self.sample_count - cursor)
            pieces.append(permutation[cursor : cursor + take])
            position += take
            remaining -= take
        return np.concatenate(pieces)

    def local_micro_batches(self, update: int, rank: int) -> tuple[tuple[int, ...], ...]:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}")
        global_batch = self.global_indices(update)
        local_start = rank * self.local_batch_size
        local = global_batch[local_start : local_start + self.local_batch_size]
        return tuple(
            tuple(int(value) for value in local[start : start + self.micro_batch_size])
            for start in range(0, self.local_batch_size, self.micro_batch_size)
        )

    def iter_micro_batches(
        self, start_update: int, stop_update: int, rank: int
    ) -> Iterator[list[int]]:
        if stop_update < start_update:
            raise ValueError("stop_update must not precede start_update")
        for update in range(start_update, stop_update):
            for batch in self.local_micro_batches(update, rank):
                yield list(batch)

    def state_before_update(self, next_update: int) -> SamplerState:
        if next_update < 0:
            raise ValueError("next_update must be non-negative")
        images_seen = next_update * self.global_batch_size
        epoch, cursor = divmod(images_seen, self.sample_count)
        permutation = self.epoch_permutation(epoch)
        return SamplerState(
            next_update=next_update,
            images_seen=images_seen,
            epoch=epoch,
            cursor=cursor,
            permutation=tuple(int(value) for value in permutation),
            permutation_sha256=_int64_sha256(permutation),
        )

    def validate_state(self, payload: dict[str, object]) -> SamplerState:
        next_update = int(payload["next_update"])
        expected = self.state_before_update(next_update)
        observed_permutation = np.asarray(payload["permutation"], dtype=np.int64)
        observed_hash = str(payload["permutation_sha256"])
        if observed_permutation.shape != (self.sample_count,):
            raise ValueError("checkpoint sampler permutation has the wrong length")
        if _int64_sha256(observed_permutation) != observed_hash:
            raise ValueError("checkpoint sampler permutation hash is invalid")
        if payload != expected.to_dict():
            raise ValueError("checkpoint sampler state does not match the frozen plan")
        return expected


class PlannedBatchSampler(Sequence[list[int]]):
    """Finite DataLoader batch sampler backed by a deterministic plan."""

    def __init__(
        self,
        plan: DeterministicDistributedBatchPlan,
        start_update: int,
        stop_update: int,
        rank: int,
    ) -> None:
        self.plan = plan
        self.start_update = start_update
        self.stop_update = stop_update
        self.rank = rank

    def __iter__(self) -> Iterator[list[int]]:
        return self.plan.iter_micro_batches(
            self.start_update, self.stop_update, self.rank
        )

    def __len__(self) -> int:
        return (self.stop_update - self.start_update) * self.plan.accumulation_steps

    def __getitem__(self, index: int) -> list[int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        update = self.start_update + index // self.plan.accumulation_steps
        micro = index % self.plan.accumulation_steps
        return list(self.plan.local_micro_batches(update, self.rank)[micro])
