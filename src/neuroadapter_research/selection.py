"""Frozen checkpoint-selection statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata


LOW_LEVEL = ("PixCorr", "SSIM", "AlexNet-2")
HIGH_LEVEL_DESC = ("AlexNet-5", "Inception", "CLIP")
HIGH_LEVEL_ASC = ("EffCorrDistance", "SwAVCorrDistance")
EVALUATION_BINDING_FIELDS = (
    "config_sha256",
    "method_fingerprint",
    "image_count",
    "candidate_count",
    "negative_pool",
    "candidate_aggregation",
    "evaluation_manifest_sha256",
    "validation_ids_sha256",
    "image_order_sha256",
    "selection_plan_sha256",
    "metric_implementation_sha256",
    "protocol_namespace",
    "selection_stage",
    "denoising_steps",
    "guidance_scale",
    "evaluation_batch_size",
    "repository_commit",
)
RECORD_PROVENANCE_FIELDS = (
    "snapshot_model_sha256",
    "snapshot_manifest_sha256",
    "snapshot_metadata_sha256",
    "run_mode",
    "run_kind",
    "training_config_sha256",
    "method_fingerprint",
    "formal_approval_sha256",
)


def merge_evaluation_payloads(
    payloads: list[dict[str, Any]], *, expected_candidate_count: int
) -> list[dict[str, Any]]:
    """Merge independently written evaluator outputs under one frozen binding."""

    if not payloads:
        raise ValueError("at least one evaluation payload is required")
    reference: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("schema_version") != 1 or payload.get("status") != "complete":
            raise ValueError("evaluation payload is not complete schema version 1")
        binding = {name: payload.get(name) for name in EVALUATION_BINDING_FIELDS}
        if binding["image_count"] != 500:
            raise ValueError("evaluation payload does not cover 500 validation images")
        if binding["candidate_count"] != expected_candidate_count:
            raise ValueError(
                f"selection stage requires {expected_candidate_count} candidates"
            )
        if any(binding[name] is None for name in EVALUATION_BINDING_FIELDS):
            raise ValueError("evaluation payload is missing a frozen binding field")
        if reference is None:
            reference = binding
        elif binding != reference:
            raise ValueError("evaluation payloads use different frozen inputs")
        payload_records = payload.get("checkpoints")
        if not isinstance(payload_records, list) or not payload_records:
            raise ValueError("evaluation payload has no checkpoint records")
        for record in payload_records:
            if any(record.get(name) is None for name in RECORD_PROVENANCE_FIELDS):
                raise ValueError("checkpoint record is missing snapshot provenance")
            if record["run_mode"] != "formal" or record["run_kind"] != "selection":
                raise ValueError("checkpoint record is not from formal selection")
            if record["training_config_sha256"] != binding["config_sha256"]:
                raise ValueError("checkpoint record uses a different training config")
            if record["method_fingerprint"] != binding["method_fingerprint"]:
                raise ValueError("checkpoint record uses a different method")
        records.extend(payload_records)
    updates = [int(record["optimizer_update"]) for record in records]
    if len(set(updates)) != len(updates):
        raise ValueError("evaluation payloads contain duplicate checkpoint updates")
    return records


def _rows_normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must have shape [images, features]")
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("features contain a constant or non-finite row")
    return centered / norms


def per_image_two_way_identification(
    original_features: np.ndarray, reconstructed_features: np.ndarray
) -> np.ndarray:
    """Compute per-image identification once on a fixed, unique negative pool."""

    original = _rows_normalized(original_features)
    reconstructed = _rows_normalized(reconstructed_features)
    if original.shape != reconstructed.shape or original.shape[0] < 2:
        raise ValueError("original and reconstructed feature pools must match and contain >=2 images")
    similarities = original @ reconstructed.T
    diagonal = np.diag(similarities)
    successes = (similarities < diagonal[:, None]).sum(axis=1)
    return successes.astype(np.float64) / (original.shape[0] - 1)


def seed_mean_two_way_identification(
    original_features: np.ndarray, reconstructed_by_seed: np.ndarray
) -> np.ndarray:
    values = np.asarray(reconstructed_by_seed)
    if values.ndim != 3:
        raise ValueError("reconstructed features must have shape [seeds, images, features]")
    per_seed = np.stack(
        [per_image_two_way_identification(original_features, seed) for seed in values]
    )
    return per_seed.mean(axis=0)


def semantic_score(metrics: dict[str, float]) -> float:
    return float(np.mean([metrics[name] for name in HIGH_LEVEL_DESC]))


def _ranks(values: Iterable[float], descending: bool) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("ranking values must be finite")
    return rankdata(-array if descending else array, method="average")


def balanced_ranks(records: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    if not records:
        raise ValueError("cannot rank an empty checkpoint set")
    updates = [int(record["optimizer_update"]) for record in records]
    if len(set(updates)) != len(updates):
        raise ValueError("checkpoint updates must be unique")
    low_components = [
        _ranks((record["metrics"][name] for record in records), descending=True)
        for name in LOW_LEVEL
    ]
    high_components = [
        _ranks((record["metrics"][name] for record in records), descending=True)
        for name in HIGH_LEVEL_DESC
    ] + [
        _ranks((record["metrics"][name] for record in records), descending=False)
        for name in HIGH_LEVEL_ASC
    ]
    low = np.stack(low_components).mean(axis=0)
    high = np.stack(high_components).mean(axis=0)
    balanced = 0.5 * low + 0.5 * high
    return {
        update: {
            "low_level_rank": float(low[index]),
            "high_level_rank": float(high[index]),
            "balanced_rank": float(balanced[index]),
        }
        for index, update in enumerate(updates)
    }


def build_shortlist(
    records: list[dict[str, Any]], *, expected_updates: list[int], count: int = 5
) -> list[int]:
    observed = {int(record["optimizer_update"]) for record in records}
    if observed != set(expected_updates) or len(records) != len(expected_updates):
        raise ValueError("screening checkpoint set differs from the frozen selection plan")
    if len(records) < count:
        raise ValueError(f"at least {count} checkpoints are required")
    by_semantic = sorted(
        records,
        key=lambda record: (-semantic_score(record["metrics"]), int(record["optimizer_update"])),
    )
    ranks = balanced_ranks(records)
    by_low = min(
        records,
        key=lambda record: (
            ranks[int(record["optimizer_update"])]["low_level_rank"],
            int(record["optimizer_update"]),
        ),
    )
    by_loss = min(
        records,
        key=lambda record: (
            float(record["validation_loss"]),
            int(record["optimizer_update"]),
        ),
    )
    selected: list[int] = []
    for record in [*by_semantic[:3], by_low, by_loss, *by_semantic]:
        update = int(record["optimizer_update"])
        if update not in selected:
            selected.append(update)
        if len(selected) == count:
            break
    return selected


def paired_bootstrap_standard_error(
    differences: np.ndarray, *, draws: int, seed: int
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired differences must be a finite one-dimensional image vector")
    if draws < 2:
        raise ValueError("bootstrap draws must be at least two")
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return float(means.std(ddof=1))


@dataclass(frozen=True)
class SelectionResult:
    selected_update: int
    best_semantic_update: int
    one_se_updates: tuple[int, ...]
    diagnostics: dict[int, dict[str, float]]


def select_checkpoint(
    records: list[dict[str, Any]], *, bootstrap_draws: int = 10000, bootstrap_seed: int
) -> SelectionResult:
    if len(records) < 2:
        raise ValueError("selection requires at least two checkpoint records")
    per_image = {}
    for record in records:
        update = int(record["optimizer_update"])
        values = np.asarray(record["per_image_semantic_score"], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"invalid per-image semantic scores for update {update}")
        per_image[update] = values
    sizes = {values.size for values in per_image.values()}
    if len(sizes) != 1:
        raise ValueError("all checkpoints must use the same image pool")

    means = {update: float(values.mean()) for update, values in per_image.items()}
    best = min(means, key=lambda update: (-means[update], update))
    diagnostics: dict[int, dict[str, float]] = {}
    one_se = []
    for update, values in per_image.items():
        differences = per_image[best] - values
        mean_difference = float(differences.mean())
        standard_error = paired_bootstrap_standard_error(
            differences, draws=bootstrap_draws, seed=bootstrap_seed
        )
        diagnostics[update] = {
            "semantic_mean": means[update],
            "difference_from_best": mean_difference,
            "paired_bootstrap_standard_error": standard_error,
        }
        if mean_difference <= standard_error + 1e-15:
            one_se.append(update)

    eligible = [record for record in records if int(record["optimizer_update"]) in one_se]
    ranks = balanced_ranks(eligible)
    for update, values in ranks.items():
        diagnostics[update].update(values)
    selected = min(
        eligible,
        key=lambda record: (
            ranks[int(record["optimizer_update"])]["balanced_rank"],
            float(record["validation_loss"]),
            int(record["optimizer_update"]),
        ),
    )
    return SelectionResult(
        selected_update=int(selected["optimizer_update"]),
        best_semantic_update=best,
        one_se_updates=tuple(sorted(one_se)),
        diagnostics=diagnostics,
    )
