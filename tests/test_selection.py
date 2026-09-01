from __future__ import annotations

import numpy as np

from neuroadapter_research.selection import (
    balanced_ranks,
    build_shortlist,
    merge_evaluation_payloads,
    per_image_two_way_identification,
    select_checkpoint,
)


def record(update: int, semantic: float, low: float, distance: float, loss: float):
    return {
        "optimizer_update": update,
        "validation_loss": loss,
        "metrics": {
            "PixCorr": low,
            "SSIM": low,
            "AlexNet-2": low,
            "AlexNet-5": semantic,
            "Inception": semantic,
            "CLIP": semantic,
            "EffCorrDistance": distance,
            "SwAVCorrDistance": distance,
        },
        "per_image_semantic_score": [semantic - 0.01, semantic, semantic + 0.01],
    }


def test_two_way_identification_uses_fixed_pool() -> None:
    original = np.eye(3, dtype=np.float64)
    reconstructed = original.copy()
    scores = per_image_two_way_identification(original, reconstructed)
    np.testing.assert_allclose(scores, np.ones(3))


def test_balanced_rank_directions() -> None:
    records = [record(1, 0.8, 0.1, 0.5, 2.0), record(2, 0.9, 0.9, 0.1, 1.0)]
    ranks = balanced_ranks(records)
    assert ranks[2]["low_level_rank"] < ranks[1]["low_level_rank"]
    assert ranks[2]["high_level_rank"] < ranks[1]["high_level_rank"]


def test_shortlist_is_unique_and_has_five_updates() -> None:
    records = [record(i, 1.0 - i / 100, i / 10, i / 20, float(10 - i)) for i in range(1, 8)]
    shortlist = build_shortlist(records)
    assert len(shortlist) == len(set(shortlist)) == 5


def test_one_se_selection_uses_balanced_rank_then_ties() -> None:
    first = record(10, 0.80, 0.50, 0.50, 1.0)
    second = record(20, 0.80, 0.80, 0.20, 0.9)
    result = select_checkpoint([first, second], bootstrap_draws=100, bootstrap_seed=7)
    assert result.one_se_updates == (10, 20)
    assert result.selected_update == 20


def evaluation_payload(update: int, candidate_count: int = 2):
    return {
        "schema_version": 1,
        "status": "complete",
        "config_sha256": "d" * 64,
        "image_count": 500,
        "candidate_count": candidate_count,
        "negative_pool": "fixed-500",
        "candidate_aggregation": "seed-mean",
        "evaluation_manifest_sha256": "a" * 64,
        "validation_ids_sha256": "b" * 64,
        "checkpoints": [record(update, 0.8, 0.7, 0.2, 1.0)],
    }


def test_evaluation_merge_requires_matching_bindings() -> None:
    merged = merge_evaluation_payloads(
        [evaluation_payload(10), evaluation_payload(20)], expected_candidate_count=2
    )
    assert [item["optimizer_update"] for item in merged] == [10, 20]

    changed = evaluation_payload(30)
    changed["validation_ids_sha256"] = "c" * 64
    with np.testing.assert_raises_regex(ValueError, "different frozen inputs"):
        merge_evaluation_payloads(
            [evaluation_payload(10), changed], expected_candidate_count=2
        )
