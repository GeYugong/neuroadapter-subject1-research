"""Independent random streams and restartable process RNG state."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


STREAM_NAMES = ("vae_latent", "diffusion_noise", "timestep", "token_dropout")


def namespace_seed(base_seed: int, namespace: str, rank: int) -> int:
    material = (
        f"neuroadapter-subject1|rng|{base_seed}|{namespace}|rank={rank}".encode(
            "ascii"
        )
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


@dataclass
class TrainingGenerators:
    generators: dict[str, torch.Generator]
    device_type: str

    @classmethod
    def create(
        cls, base_seed: int, rank: int, device: torch.device
    ) -> "TrainingGenerators":
        generators: dict[str, torch.Generator] = {}
        generator_device = device.type if device.type == "cuda" else "cpu"
        for name in STREAM_NAMES:
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(namespace_seed(base_seed, name, rank))
            generators[name] = generator
        return cls(generators=generators, device_type=generator_device)

    def __getitem__(self, name: str) -> torch.Generator:
        return self.generators[name]

    def state_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "states": {
                name: generator.get_state().cpu()
                for name, generator in self.generators.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload["device_type"] != self.device_type:
            raise ValueError(
                f"generator device mismatch: {payload['device_type']} != {self.device_type}"
            )
        states = payload["states"]
        if set(states) != set(self.generators):
            raise ValueError("generator stream names do not match")
        for name, generator in self.generators.items():
            generator.set_state(states[name].cpu())


def capture_process_rng_state(device: torch.device | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if device is not None and device.type == "cuda":
        payload["torch_cuda"] = torch.cuda.get_rng_state(device).cpu()
        payload["cuda_device_index"] = device.index
    return payload


def restore_process_rng_state(
    payload: dict[str, Any], device: torch.device | None = None
) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if "torch_cuda" in payload:
        if device is None or device.type != "cuda":
            raise ValueError("checkpoint contains CUDA RNG state but no CUDA device was given")
        if payload.get("cuda_device_index") != device.index:
            raise ValueError("CUDA RNG checkpoint belongs to a different local device")
        torch.cuda.set_rng_state(payload["torch_cuda"].cpu(), device)
