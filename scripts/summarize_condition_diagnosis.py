#!/usr/bin/env python3
"""Preserve all paired diagnostic scores and unfiltered contact sheets."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.metrics import pixel_metrics
from evaluate_validation import CLIP_MEAN, CLIP_STD, extract_features, preprocess_transform


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def estimate(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260914)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"n_images": len(values), "mean": float(values.mean()),
            "ci95_exploratory": np.quantile(means, [.025, .975]).tolist(),
            "positive_images": int((values > 0).sum())}


def csv_file(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def gallery(out, split, name, pairs, columns):
    paths = []
    for start in range(0, len(pairs), 8):
        canvas = Image.new("RGB", (160 * len(columns), 190 * 8), "white")
        draw = ImageDraw.Draw(canvas)
        for r, pair in enumerate(pairs[start:start + 8]):
            for c, (label, path_fn) in enumerate(columns):
                path = path_fn(pair)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as source:
                    value = source.convert("RGB").resize((160, 160))
                canvas.paste(value, (c * 160, r * 190 + 30))
                draw.text((c * 160 + 2, r * 190 + 2), label, fill="black")
                draw.text((c * 160 + 2, r * 190 + 15), f"id={pair['image_id']}", fill="black")
        relative = Path("galleries") / f"{split}-{name}-{start//8+1}.jpg"
        (out / relative).parent.mkdir(exist_ok=True)
        canvas.save(out / relative, quality=92)
        paths.append(str(relative))
    return paths


@torch.no_grad()
def summarize(root, vae_only):
    out = root / "runs/diagnostics/20260914-condition-path"
    protocol = json.loads((out / "protocol.json").read_text())
    summary = {"exploratory": True, "vae": {}, "denoising": {}, "generation": {},
               "precision": {}, "guidance": {}, "galleries": []}
    for split in ("train", "validation"):
        folder = out / split
        assert (folder / "vae/complete.json").exists()
        rows = read_rows(folder / "vae/metrics.jsonl")
        assert len(rows) == 128
        summary["vae"][split] = {
            f"{precision}_{mode}": estimate([r["mse_rgb"] for r in rows if
                r["precision"] == precision and r["mode"] == mode])
            for precision in ("fp32", "bf16") for mode in ("mode", "sample")}
        columns = [("GT preprocessed", lambda p: folder / "gt" / f"{p['image_id']}.png")]
        for precision, mode in [("fp32", "mode"), ("bf16", "mode"), ("fp32", "sample"), ("bf16", "sample")]:
            columns.append((f"{precision} {mode}", lambda p, a=precision, b=mode:
                            folder / "vae" / a / b / f"{p['image_id']}.png"))
        summary["galleries"] += gallery(out, split, "vae", protocol["pairs"][split], columns)
    if vae_only:
        write_json_atomic(out / "vae_summary.json", summary)
        print(json.dumps(summary["vae"], indent=2))
        return

    sys.path.insert(0, str(root / "repo/vendor/CLIP"))
    import clip
    manifest = json.loads((root / "data/fingerprints/evaluation_downloads.json").read_text())
    record = manifest["files"]["clip_vit_l_14"]
    asset = root / record["path"]
    assert sha256_file(asset) == record["sha256"]
    clip_model, _ = clip.load(str(asset), device="cuda:0", jit=False)
    transform = preprocess_transform(224, CLIP_MEAN, CLIP_STD)
    all_scores = []
    all_pairs = []
    for split in ("train", "validation"):
        folder = out / split
        denoising = read_rows(folder / "bf16/denoising.jsonl")
        assert len(denoising) == 160
        summary["denoising"][split] = {
            str(t): {**estimate([r["delta"] for r in denoising if r["timestep"] == t]),
                     "correct_mse": float(np.mean([r["correct_mse"] for r in denoising if r["timestep"] == t])),
                     "wrong_mse": float(np.mean([r["wrong_mse"] for r in denoising if r["timestep"] == t]))}
            for t in (50, 200, 500, 800, 950)}
        csv_file(folder / "denoising.csv", denoising)
        ids = [p["image_id"] for p in protocol["pairs"][split]]
        originals = np.stack([np.asarray(Image.open(folder / "gt" / f"{i}.png")) for i in ids])
        gt_features = extract_features(originals, clip_model.encode_image, transform,
                                       device=torch.device("cuda:0"), batch_size=8)
        gt_features /= np.linalg.norm(gt_features, axis=1, keepdims=True)
        id_index = {value: i for i, value in enumerate(ids)}
        scores = {}
        records_by_phase = {}
        for phase in ("bf16", "fp32", "guidance"):
            if not (folder / phase / "complete.json").exists():
                continue
            records = read_rows(folder / phase / "images.jsonl")
            assert len(records) == 128
            records_by_phase[phase] = records
            arrays = np.stack([np.asarray(Image.open(folder / phase / r["path"])) for r in records])
            assert all(sha256_file(folder / phase / r["path"]) == r["sha256"] for r in records)
            features = extract_features(arrays, clip_model.encode_image, transform,
                                        device=torch.device("cuda:0"), batch_size=8)
            features /= np.linalg.norm(features, axis=1, keepdims=True)
            target = originals[[id_index[r["image_id"]] for r in records]]
            corr, ssim = pixel_metrics(target, arrays)
            for j, record in enumerate(records):
                row = {"split": split, "phase": phase, **record,
                       "clip_target_cosine": float(features[j] @ gt_features[id_index[record["image_id"]]]),
                       "clip_donor_cosine": float(features[j] @ gt_features[id_index[record["donor_id"]]]),
                       "pixcorr": float(corr[j]), "ssim": float(ssim[j])}
                key = (phase, int(row["guidance"]), row["condition"], row["image_id"], row["candidate"])
                scores[key] = row
                all_scores.append(row)
        for phase in ("bf16", "fp32"):
            if phase not in records_by_phase:
                continue
            differences = defaultdict(list)
            for pair in protocol["pairs"][split]:
                image_id = pair["image_id"]
                mean_values = defaultdict(list)
                for candidate in range(2):
                    correct = scores[phase, 4, "correct", image_id, candidate]
                    wrong = scores[phase, 4, "wrong", image_id, candidate]
                    assert correct["noise_sha256"] == wrong["noise_sha256"]
                    ca = np.asarray(Image.open(folder / phase / correct["path"])).astype(float) / 255
                    wa = np.asarray(Image.open(folder / phase / wrong["path"])).astype(float) / 255
                    values = {"changed_pixel_mae": float(np.abs(ca-wa).mean())}
                    for metric in ("clip_target_cosine", "pixcorr", "ssim"):
                        values[metric + "_advantage"] = correct[metric] - wrong[metric]
                    all_pairs.append({"split": split, "phase": phase, **pair, "candidate": candidate, **values})
                    for name, value in values.items():
                        mean_values[name].append(value)
                for name, value in mean_values.items():
                    differences[name].append(float(np.mean(value)))
            summary["generation"][f"{split}_{phase}"] = {k: estimate(v) for k, v in differences.items()}
            columns = [("GT", lambda p: folder / "gt" / f"{p['image_id']}.png"),
                       ("Donor GT", lambda p: folder / "gt" / f"{p['donor_id']}.png")]
            for condition, candidate in [("correct", 0), ("wrong", 0), ("correct", 1), ("wrong", 1)]:
                columns.append((f"{condition} seed{candidate}", lambda p, a=condition, b=candidate:
                                folder / phase / f"g4-{a}" / f"{p['image_id']}-{b}.png"))
            summary["galleries"] += gallery(out, split, phase, protocol["pairs"][split], columns)
        for phase, g, field in [("fp32", 4, "precision"), ("guidance", 2, "guidance"), ("guidance", 6, "guidance")]:
            if phase not in records_by_phase:
                continue
            values = defaultdict(list)
            for image_id in ids:
                for metric in ("clip_target_cosine", "pixcorr", "ssim"):
                    delta = [scores[phase, g, "correct", image_id, c][metric] -
                             scores["bf16", 4, "correct", image_id, c][metric] for c in range(2)]
                    values[metric].append(float(np.mean(delta)))
            summary[field][f"{split}_{phase}_g{g}_minus_bf16_g4"] = {k: estimate(v) for k, v in values.items()}
        if "guidance" in records_by_phase:
            columns = [("GT", lambda p: folder / "gt" / f"{p['image_id']}.png")]
            for phase, g, c in [("guidance", 2, 0), ("bf16", 4, 0), ("guidance", 6, 0),
                                ("guidance", 2, 1), ("bf16", 4, 1), ("guidance", 6, 1)]:
                columns.append((f"g{g} seed{c}", lambda p, a=phase, b=g, d=c:
                                folder / a / f"g{b}-correct" / f"{p['image_id']}-{d}.png"))
            summary["galleries"] += gallery(out, split, "guidance", protocol["pairs"][split], columns)
    csv_file(out / "per_image_scores.csv", all_scores)
    csv_file(out / "paired_scores.csv", all_pairs)
    write_json_atomic(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "galleries"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--vae-only", action="store_true")
    args = parser.parse_args()
    summarize(args.project_root, args.vae_only)
