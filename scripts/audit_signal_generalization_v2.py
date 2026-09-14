#!/usr/bin/env python3
"""Independent input reconstruction and train-only signal diagnostics."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def csv_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def layout(root):
    data = root / "data/derived"
    meta = np.load(data / "neural_data/metadata_sub-01.npy", allow_pickle=True).item()
    manifest = json.loads((root / "data/fingerprints/training_cache_manifest.json").read_text())
    assert sha256_file(data / "neural_data/metadata_sub-01.npy") == manifest["metadata_sha256"]
    parcels = []
    token_rows = []
    for hi, hemi in enumerate(("lh", "rh")):
        labels = torch.load(data / f"parcels/schaefer/{hemi}_labels_s01.pt", weights_only=True)[1:]
        snr = np.asarray(meta[f"{hemi}_ncsnr"], dtype=np.float32)
        scores = np.array([snr[x.numpy()].mean() for x in labels])
        selected = np.argsort(scores)[::-1][:100]
        assert selected.tolist() == manifest["selected_parcel_indices_excluding_medial_wall"][hemi]
        for rank, parcel in enumerate(selected):
            vertices = labels[int(parcel)].numpy()
            parcels.append(vertices)
            token_rows.append({"token": hi * 100 + rank, "hemisphere": hemi,
                "parcel_index_excluding_wall": int(parcel), "vertices": vertices.tolist()})
    return meta, parcels, token_rows


def input_audit(root):
    out = root / "runs/diagnostics/generalization-diagnosis-v2/signal"
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((out.parent / "config.json").read_text())
    meta, parcels, tokens = layout(root)
    order = meta["img_presentation_order"]
    write_json_atomic(out / "vertex_order.json", tokens)
    records = []
    with h5py.File(root / "data/derived/neural_data/betas_sub-01.h5", "r") as source, h5py.File(
        root / "data/derived/training/subject01_train_pool_top100.h5", "r") as cache:
        ids = cache["image_ids"][:]
        mask = cache["parcel_valid_mask"][:]
        assert mask.shape == (200, 626)
        for j, v in enumerate(parcels):
            assert np.array_equal(mask[j], np.arange(626) < len(v))
        for split, pairs in spec["pairs"].items():
            for p in pairs:
                image_id = p["image_id"]
                trials = np.flatnonzero(order == image_id)
                assert len(trials) == 3 and image_id in ids
                rebuilt = np.zeros((200, 626), np.float32)
                for hi, hemi in enumerate(("lh", "rh")):
                    reps = np.asarray(source[f"{hemi}_betas"][trials], dtype=np.float32)
                    # Independent explicit arithmetic, preserving source float32 reduction order.
                    avg = ((reps[0] + reps[1]) + reps[2]) / np.float32(3)
                    for j in range(100):
                        v = parcels[hi*100+j]
                        rebuilt[hi*100+j, :len(v)] = avg[v]
                cached = cache["brain"][int(np.flatnonzero(ids == image_id)[0])]
                delta = np.abs(cached - rebuilt)
                records.append({"split": split, "image_id": image_id,
                    "trials_zero_based": trials.tolist(), "sessions_one_based": (trials//750+1).tolist(),
                    "within_session_trials_one_based": (trials%750+1).tolist(),
                    "exact": bool(np.array_equal(cached, rebuilt)),
                    "max_valid_difference": float(delta[mask].max()),
                    "padding_nonzero": int(np.count_nonzero(cached[~mask])),
                    "finite": bool(np.isfinite(cached).all() and np.isfinite(rebuilt).all())})
    passed = all(r["exact"] and r["finite"] and r["padding_nonzero"] == 0 for r in records)
    write_json_atomic(out / "input_audit.json", {"passed": passed, "n_images": len(records),
        "vertex_order_sha256": sha256_file(out / "vertex_order.json"), "records": records})
    print(f"B1 exact input reconstruction: {sum(r['exact'] for r in records)}/64; passed={passed}", flush=True)
    assert passed, "unexplained cache mismatch: stop performance attribution"


def spatial_corr(x, y, starts, lengths):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    sx, sy = np.add.reduceat(x, starts), np.add.reduceat(y, starts)
    cov = np.add.reduceat(x*y, starts) - sx*sy/lengths
    vx = np.maximum(np.add.reduceat(x*x, starts)-sx*sx/lengths, 0)
    vy = np.maximum(np.add.reduceat(y*y, starts)-sy*sy/lengths, 0)
    return np.divide(cov, np.sqrt(vx*vy), out=np.full_like(cov, np.nan), where=vx*vy > 0)


def full_audit(root):
    start = time.time()
    out = root / "runs/diagnostics/generalization-diagnosis-v2/signal"
    assert json.loads((out / "input_audit.json").read_text())["passed"]
    spec = json.loads((out.parent / "config.json").read_text())
    meta, parcels, tokens = layout(root)
    train = np.loadtxt(root / "data/derived/splits/selection_train_ids.txt", dtype=int)
    val = np.loadtxt(root / "data/derived/splits/validation_ids.txt", dtype=int)
    ids = np.sort(np.concatenate((train, val)))
    assert len(train) == 8500 and len(val) == 500 and len(set(ids)) == 9000
    is_train = np.isin(ids, train)
    order = meta["img_presentation_order"]
    rows = np.array([np.flatnonzero(order == i) for i in ids])
    assert rows.shape == (9000, 3)
    sessions = rows//750+1
    sizes = np.array([len(v) for v in parcels])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    offsets = [0, sizes[:100].sum(), sizes.sum()]
    # Only the 27000 training-pool presentations are requested from source HDF5.
    reps = np.empty((9000, 3, sizes.sum()), np.float32)
    sorted_flat = np.argsort(rows.ravel())
    sorted_trials = rows.ravel()[sorted_flat]
    flat = reps.reshape(27000, -1)
    with h5py.File(root / "data/derived/neural_data/betas_sub-01.h5", "r") as source:
        for hi, hemi in enumerate(("lh", "rh")):
            v = np.concatenate(parcels[hi*100:(hi+1)*100])
            for k in range(0, len(sorted_trials), 32):
                block = source[f"{hemi}_betas"][sorted_trials[k:k+32]]
                flat[sorted_flat[k:k+32], offsets[hi]:offsets[hi+1]] = block[:, v]
            print(f"B2 read {hemi} training-pool presentations elapsed={time.time()-start:.1f}s", flush=True)
    assert np.isfinite(reps).all()
    means = reps.mean(1, dtype=np.float32)
    train_sum = means[is_train].sum(0, dtype=np.float64)
    baseline = train_sum/8500
    train_std = means[is_train].std(0, dtype=np.float64)
    # Store diagnostics only; the source/cache and model inputs are never replaced.
    np.save(out / "valid_mean_responses.npy", means)
    np.save(out / "pool_ids.npy", ids)
    group_masks = {"train8500": is_train, "validation500": ~is_train,
        "train32": np.isin(ids, [p["image_id"] for p in spec["pairs"]["train"]]),
        "validation32": np.isin(ids, [p["image_id"] for p in spec["pairs"]["validation"]])}
    vertex_stats = {}
    parcel_stats = []
    session_stats = {}
    for group, mask in group_masks.items():
        vm, vs = means[mask].mean(0, dtype=np.float64), means[mask].std(0, dtype=np.float64)
        vertex_stats[group+"_mean"], vertex_stats[group+"_std"] = vm, vs
        session_stats[group] = {str(s): int((sessions[mask] == s).sum()) for s in range(1,41)}
        for j, (a, n) in enumerate(zip(starts, sizes)):
            values = means[mask, a:a+n]
            parcel_stats.append({"group": group, "token": j, "n_vertices": int(n),
                "value_mean": float(values.mean(dtype=np.float64)), "value_std": float(values.std(dtype=np.float64)),
                "parcel_mean_across_images_std": float(values.mean(1).std(dtype=np.float64))})
    np.savez(out / "vertex_distribution.npz", **vertex_stats)
    csv_rows(out / "parcel_distribution.csv", parcel_stats)
    write_json_atomic(out / "session_distribution.json", session_stats)
    candidates = defaultdict(list)
    session_candidates = defaultdict(list)
    pairs = list(itertools.combinations(range(3),2))
    for i in range(len(ids)):
        for a,b in pairs:
            candidates[(bool(is_train[i]), int(sessions[i,a]), int(sessions[i,b]))].append((i,a,b))
        for b in range(3):
            session_candidates[(bool(is_train[i]), int(sessions[i,b]))].append((i,b))
    # Parcel, hemisphere and whole-input spatial correlations exclude padding.
    def correlations(x, y):
        return np.r_[spatial_corr(x,y,starts,sizes),
            spatial_corr(x,y,np.array(offsets[:-1]),np.diff(offsets)),
            spatial_corr(x,y,np.array([0]),np.array([sizes.sum()]))]
    metrics = np.empty((9000, 3, 4, 203), np.float64)
    controls = []
    quality = []
    for i, image_id in enumerate(ids):
        own_mean = means[i].astype(np.float64)
        ownbase = (train_sum-own_mean)/8499 if is_train[i] else baseline
        for pi,(a,b) in enumerate(pairs):
            key = (bool(is_train[i]),int(sessions[i,a]),int(sessions[i,b]))
            eligible = [x for x in candidates[key] if x[0] != i]
            exact = bool(eligible)
            if eligible:
                j, da, db = eligible[int(image_id+pi)%len(eligible)]
            else:
                eligible2 = [x for x in session_candidates[(bool(is_train[i]),int(sessions[i,b]))] if x[0] != i]
                assert eligible2
                j, db = eligible2[int(image_id+pi)%len(eligible2)]
                da = -1
            # Control comparison excludes both images from a training-derived baseline.
            base = (train_sum-own_mean-means[j])/8498 if is_train[i] else baseline
            metrics[i,pi,0] = correlations(reps[i,a], reps[i,b])
            metrics[i,pi,1] = correlations(reps[i,a]-ownbase, reps[i,b]-ownbase)
            metrics[i,pi,2] = correlations(reps[i,a], reps[j,db])
            metrics[i,pi,3] = (correlations(reps[i,a]-base,reps[i,b]-base)
                               - correlations(reps[i,a]-base,reps[j,db]-base))
            controls.append({"image_id": int(image_id), "repeat_a": a, "repeat_b": b,
                "donor_image_id": int(ids[j]), "donor_repeat_a": da, "donor_repeat_b": db,
                "target_session_a": int(sessions[i,a]), "target_session_b": int(sessions[i,b]),
                "matched_session_pair": exact})
        quality.append({"image_id": int(image_id), "split": "train" if is_train[i] else "validation",
            "rms": float(np.sqrt(np.mean(own_mean**2))),
            "extreme_fraction_train_5sd": float(np.mean(np.abs(own_mean-baseline) > 5*np.maximum(train_std,1e-12))),
            "raw_repeat_corr": float(metrics[i,:,0,-1].mean()),
            "centered_repeat_corr": float(metrics[i,:,1,-1].mean()),
            "different_image_raw_corr": float(metrics[i,:,2,-1].mean()),
            "centered_repeat_specificity": float(metrics[i,:,3,-1].mean()),
            "lh_repeat_specificity": float(metrics[i,:,3,200].mean()),
            "rh_repeat_specificity": float(metrics[i,:,3,201].mean()),
            "session_1": int(sessions[i,0]), "session_2": int(sessions[i,1]), "session_3": int(sessions[i,2])})
        if (i+1)%1000 == 0:
            print(f"B3 consistency {i+1}/9000 elapsed={time.time()-start:.1f}s", flush=True)
    np.save(out / "repeat_correlations.npy", metrics)
    csv_rows(out / "session_matched_controls.csv", controls)
    csv_rows(out / "per_image_quality.csv", quality)
    strata = np.quantile([q["centered_repeat_specificity"] for q in quality if q["split"] == "train"], [1/3,2/3])
    group_stats = {}
    for group, mask in group_masks.items():
        group_stats[group] = {key: {"mean":float(np.mean([quality[i][key] for i in np.flatnonzero(mask)])),
            "quantiles_10_50_90":np.quantile([quality[i][key] for i in np.flatnonzero(mask)], [.1,.5,.9]).tolist()}
            for key in ("rms", "extreme_fraction_train_5sd", "raw_repeat_corr", "centered_repeat_corr", "centered_repeat_specificity")}
    write_json_atomic(out / "summary.json", {"groups": group_stats, "quality_tertiles_train_only": strata.tolist(),
        "control_exact_session_pairs":sum(c["matched_session_pair"] for c in controls), "control_total":len(controls),
        "metrics_axes": ["image", "repeat_pair_01_02_12", "raw_same,centered_same,raw_different,centered_same_minus_different", "200parcels,lh,rh,all"],
        "finite_correlations": bool(np.isfinite(metrics).all()), "seconds":time.time()-start,
        "standardization_of_author_input": "unknown; no input normalization change",
        "source_scope": "only 9000 training-pool images; no standard-test beta or stimulus rows requested"})
    print(json.dumps(group_stats, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=["input", "full"], required=True)
    a = p.parse_args()
    (input_audit if a.phase == "input" else full_audit)(a.root)
