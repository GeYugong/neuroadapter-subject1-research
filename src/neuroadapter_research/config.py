"""Strict loading and validation of frozen training configurations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .atomic import sha256_file


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class LoadedTrainingConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        return {name: Path(value) for name, value in self.raw["paths"].items()}

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], context: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def load_training_config(path: Path, *, require_frozen: bool) -> LoadedTrainingConfig:
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "run_name",
            "run_kind",
            "subject",
            "protocol_commit",
            "paths",
            "training",
        },
        "root",
    )
    if payload["schema_version"] != 1 or payload["subject"] != 1:
        raise ValueError("only Subject 1 training schema version 1 is supported")
    if payload["run_kind"] not in {"selection", "final"}:
        raise ValueError("run_kind must be selection or final")
    if require_frozen:
        if payload["status"] != "frozen":
            raise ValueError("formal training requires status: frozen")
        if not COMMIT_PATTERN.fullmatch(str(payload["protocol_commit"])):
            raise ValueError("formal training requires a full frozen protocol commit")

    paths = payload["paths"]
    _require_exact_keys(
        paths,
        {
            "stable_diffusion",
            "model_manifest",
            "training_cache",
            "training_cache_manifest",
            "stimuli",
            "split_ids",
            "canonical_initialization",
            "canonical_manifest",
            "data_fingerprint",
            "split_manifest",
            "source_manifest",
            "environment_lock",
            "output_dir",
        },
        "paths",
    )
    for name, value in paths.items():
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(f"paths.{name} must be absolute")
    if "test" in Path(paths["split_ids"]).name.lower():
        raise ValueError("training config may not point to the standard test split")

    training = payload["training"]
    _require_exact_keys(
        training,
        {
            "max_updates",
            "world_size",
            "global_batch_size",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "dataloader_workers",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
            "max_grad_norm",
            "min_snr_gamma",
            "precision",
            "allow_tf32",
            "base_seed",
            "sampler_seed",
            "log_every_updates",
            "checkpoint_every_updates",
            "checkpoint_reference_epochs",
        },
        "training",
    )
    integer_fields = (
        "max_updates",
        "world_size",
        "global_batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "dataloader_workers",
        "base_seed",
        "sampler_seed",
        "log_every_updates",
        "checkpoint_every_updates",
        "checkpoint_reference_epochs",
    )
    for name in integer_fields:
        if not isinstance(training[name], int) or training[name] < 0:
            raise ValueError(f"training.{name} must be a non-negative integer")
    positive_fields = set(integer_fields) - {"dataloader_workers"}
    for name in positive_fields:
        if training[name] <= 0:
            raise ValueError(f"training.{name} must be positive")
    if training["world_size"] != 2:
        raise ValueError("the frozen hardware protocol requires exactly two processes")
    expected_global = (
        training["world_size"]
        * training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
    )
    if training["global_batch_size"] != expected_global:
        raise ValueError("training global batch arithmetic is inconsistent")
    if training["global_batch_size"] != 16:
        raise ValueError("the first-phase protocol fixes global batch at 16")
    if training["precision"] != "bf16":
        raise ValueError("the current first-phase protocol requires bf16")
    expected_scalars = {
        "learning_rate": 1e-4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 1e-6,
        "max_grad_norm": 1.0,
        "min_snr_gamma": 5.0,
    }
    for name, expected in expected_scalars.items():
        if float(training[name]) != expected:
            raise ValueError(f"training.{name} differs from the frozen method")
    if not isinstance(training["allow_tf32"], bool):
        raise ValueError("training.allow_tf32 must be boolean")

    return LoadedTrainingConfig(path=path, sha256=sha256_file(path), raw=payload)


def verify_config_inputs(config: LoadedTrainingConfig) -> None:
    directory_names = {"stable_diffusion", "output_dir"}
    for name, path in config.paths.items():
        if name == "output_dir":
            path.parent.mkdir(parents=True, exist_ok=True)
            continue
        if name in directory_names:
            if not path.is_dir():
                raise FileNotFoundError(f"paths.{name} is missing: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"paths.{name} is missing: {path}")
