#!/usr/bin/env python3
"""Image-level paired trajectory statistics and complete, unselected galleries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from audit_signal_generalization_v2 import csv_rows
from diagnose_generalization_v2 import WEIGHTS, locations
from evaluate_validation import CLIP_MEAN, CLIP_STD, extract_features, preprocess_transform
from summarize_condition_diagnosis import estimate, gallery, read_rows
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.metrics import pixel_metrics


def paired_changes(rows):
    result = {}
    for split in ("train", "validation"):
        for before, after in ((106250,159375),(159375,239063)):
            a = {r["image_id"]:r for r in rows if r["split"] == split and r["checkpoint"] == before}
            b = {r["image_id"]:r for r in rows if r["split"] == split and r["checkpoint"] == after}
            assert len(a) == 32 and a.keys() == b.keys()
            result[f"{split}_{after}_minus_{before}"] = {
                key: estimate([b[i][key]-a[i][key] for i in sorted(a)])
                for key in ("correct_clip", "wrong_clip", "advantage_clip", "correct_pixcorr", "correct_ssim")}
    return result


@torch.no_grad()
def summarize(root):
    old, out = locations(root)
    spec = json.loads((out / "config.json").read_text())
    sys.path.insert(0, str(root / "repo/vendor/CLIP"))
    import clip
    record = json.loads((root / "data/fingerprints/evaluation_downloads.json").read_text())["files"]["clip_vit_l_14"]
    asset = root / record["path"]
    assert sha256_file(asset) == record["sha256"]
    model, _ = clip.load(str(asset), device="cuda:0", jit=False)
    transform = preprocess_transform(224, CLIP_MEAN, CLIP_STD)
    def features(arrays):
        f = extract_features(arrays, model.encode_image, transform, device=torch.device("cuda:0"), batch_size=8)
        return f / np.linalg.norm(f,axis=1,keepdims=True)
    raw_scores, image_scores, denoising, sheets = [], [], {}, []
    all_features = {}
    for split, pairs in spec["pairs"].items():
        ids = [p["image_id"] for p in pairs]
        index = {v:i for i,v in enumerate(ids)}
        gt = np.stack([np.array(Image.open(old/split/"gt"/f"{i}.png")) for i in ids])
        gf = features(gt)
        all_features[split+"_gt"] = gf
        all_features[split+"_ids"] = np.array(ids)
        for step in WEIGHTS:
            folder = old/split/"bf16" if step == 239063 else out/split/str(step)
            assert (out/split/str(step)/"complete.json").is_file()
            records = read_rows(folder/"images.jsonl")
            assert len(records) == 128
            expected = {(p["image_id"],p["donor_id"],c,j) for p in pairs for c in ("correct","wrong") for j in range(2)}
            assert {(r["image_id"],r["donor_id"],r["condition"],r["candidate"]) for r in records} == expected
            for r in records:
                assert sha256_file(folder/r["path"]) == r["sha256"]
                assert r["noise_sha256"] == sha256_file(old/split/"bf16/noise"/f"{r['image_id']}.pt")
            arrays = np.stack([np.array(Image.open(folder/r["path"])) for r in records])
            ff = features(arrays)
            pc, ss = pixel_metrics(gt[[index[r["image_id"]] for r in records]], arrays)
            scored = {}
            for j,r in enumerate(records):
                row = {"checkpoint":step,"split":split,**r,
                    "clip":float(ff[j]@gf[index[r["image_id"]]]), "pixcorr":float(pc[j]), "ssim":float(ss[j])}
                raw_scores.append(row)
                scored[r["image_id"],r["condition"],r["candidate"]] = row
            all_features[f"{split}_{step}_generation"] = ff
            all_features[f"{split}_{step}_record_ids"] = np.array([r["image_id"] for r in records])
            all_features[f"{split}_{step}_conditions"] = np.array([r["condition"] for r in records])
            for pair in pairs:
                i = pair["image_id"]
                values = {f"{c}_{m}":float(np.mean([scored[i,c,j][m] for j in range(2)]))
                          for c in ("correct","wrong") for m in ("clip","pixcorr","ssim")}
                image_scores.append({"checkpoint":step,"split":split,**pair,**values,
                    "advantage_clip":values["correct_clip"]-values["wrong_clip"],
                    "gt_donor_clip":float(gf[index[i]]@gf[index[pair["donor_id"]]])})
            rows = read_rows(out/split/str(step)/"denoising.jsonl")
            assert len(rows) == 640
            denoising[f"{split}_{step}"] = {}
            for t in spec["timesteps"]:
                averaged = []
                for i in ids:
                    rr = [r for r in rows if r["image_id"] == i and r["timestep"] == t]
                    assert sorted(r["draw"] for r in rr) == list(range(4))
                    assert len({r["noise_sha256"] for r in rr}) == 1
                    assert rr[0]["noise_sha256"] == sha256_file(out/split/"denoising_noise"/f"{i}.pt")
                    averaged.append({"image_id":i, "timestep":t, **{key:float(np.mean([r[key] for r in rr]))
                        for key in ("correct_mse","wrong_mse","delta_mse")}})
                denoising[f"{split}_{step}"][str(t)] = {key:estimate([r[key] for r in averaged])
                    for key in ("correct_mse","wrong_mse","delta_mse")}
                csv_rows(out/f"{split}-{step}-t{t}-denoising.csv",averaged)
        for condition in ("correct","wrong"):
            columns = [("GT", lambda p:old/split/"gt"/f"{p['image_id']}.png"),
                       ("Donor GT", lambda p:old/split/"gt"/f"{p['donor_id']}.png")]
            for step in WEIGHTS:
                for j in range(2):
                    columns.append((f"{step} {condition} c{j}", lambda p,s=step,c=j:
                        (old/split/"bf16"/f"g4-{condition}"/f"{p['image_id']}-{c}.png") if s==239063 else
                        (out/split/str(s)/condition/f"{p['image_id']}-{c}.png")))
            sheets += gallery(out,split,condition,pairs,columns)
    csv_rows(out/"per_candidate_scores.csv",raw_scores)
    csv_rows(out/"per_image_scores.csv",image_scores)
    np.savez(out/"clip_features.npz",**all_features)
    table = []
    for step in WEIGHTS:
        train = [r for r in image_scores if r["checkpoint"]==step and r["split"]=="train"]
        val = [r for r in image_scores if r["checkpoint"]==step and r["split"]=="validation"]
        table.append({"checkpoint":step, "train_correct_clip":float(np.mean([r["correct_clip"] for r in train])),
            "validation_correct_clip":float(np.mean([r["correct_clip"] for r in val])),
            "train_advantage_clip":float(np.mean([r["advantage_clip"] for r in train])),
            "validation_advantage_clip":float(np.mean([r["advantage_clip"] for r in val])),
            "validation_pixcorr":float(np.mean([r["correct_pixcorr"] for r in val])),
            "validation_ssim":float(np.mean([r["correct_ssim"] for r in val]))})
    csv_rows(out/"trajectory.csv",table)
    write_json_atomic(out/"trajectory_summary.json",{"table":table,"paired_changes":paired_changes(image_scores),
        "denoising":denoising,"galleries":sheets,"bootstrap_unit":"image; 4 draws/2 candidates averaged first",
        "advantage_ci":"conditional on frozen donor assignment; exploratory, unadjusted",
        "checkpoint_reselection":False,"n_images_generated":512,"n_images_reused":256})
    print(json.dumps(table,indent=2),flush=True)


def join_signal(root):
    import csv
    _, out = locations(root)
    with (out/"per_image_scores.csv").open() as f:
        scores = list(csv.DictReader(f))
    with (out/"signal/per_image_quality.csv").open() as f:
        quality = {int(r["image_id"]):r for r in csv.DictReader(f)}
    sig = json.loads((out/"signal/summary.json").read_text())
    bounds = sig["quality_tertiles_train_only"]
    joined = [{**r,**{k:v for k,v in quality[int(r["image_id"])].items() if k not in r},
               "quality_stratum":int(np.searchsorted(bounds,float(quality[int(r["image_id"])]["centered_repeat_specificity"])))} for r in scores]
    csv_rows(out/"signal_generation_join.csv",joined)
    associations, strata = {}, []
    for step in WEIGHTS:
        for split in ("train","validation"):
            rows = [r for r in joined if int(r["checkpoint"])==step and r["split"]==split]
            stats = {}
            for quality_key in ("centered_repeat_specificity","centered_repeat_corr","rms","session_1","session_2","session_3"):
                for outcome in ("correct_clip","advantage_clip"):
                    rho = spearmanr([float(r[quality_key]) for r in rows],[float(r[outcome]) for r in rows])
                    stats[quality_key+"_vs_"+outcome] = {"rho":float(rho.statistic),"p_exploratory_unadjusted":float(rho.pvalue)}
            associations[f"{split}_{step}"] = stats
            for stratum in range(3):
                group = [r for r in rows if r["quality_stratum"]==stratum]
                if group:
                    strata.append({"checkpoint":step,"split":split,"quality_stratum":stratum,"n_images":len(group),
                        "correct_clip":float(np.mean([float(r["correct_clip"]) for r in group])),
                        "advantage_clip":float(np.mean([float(r["advantage_clip"]) for r in group]))})
    csv_rows(out/"quality_stratified_scores.csv",strata)
    write_json_atomic(out/"quality_generation_associations.json",associations)
    pool_ids = np.load(out/"signal/pool_ids.npy")
    repeats = np.load(out/"signal/repeat_correlations.npy",mmap_mode="r")
    parcel_summary = []
    for split in ("train","validation"):
        for sampled in (False,True):
            mask = np.array([quality[int(i)]["split"]==split for i in pool_ids])
            if sampled:
                chosen = {int(r["image_id"]) for r in scores if r["split"]==split}
                mask &= np.isin(pool_ids,list(chosen))
            avg = repeats[mask].mean((0,1))
            for token in range(203):
                parcel_summary.append({"split":split,"sampled32":sampled,"token":token,
                    "raw_repeat_corr":float(avg[0,token]),"centered_repeat_corr":float(avg[1,token]),
                    "raw_different_corr":float(avg[2,token]),"centered_specificity":float(avg[3,token])})
    csv_rows(out/"signal/parcel_repeat_summary.csv",parcel_summary)
    sessions = json.loads((out/"signal/session_distribution.json").read_text())
    differences = {}
    for sampled in (False,True):
        groups = []
        for split in ("train","validation"):
            selected = {int(r["image_id"]) for r in scores if r["split"]==split}
            groups.append([q for i,q in quality.items() if q["split"]==split and (not sampled or i in selected)])
        group_result = {}
        for key in ("rms","centered_repeat_corr","centered_repeat_specificity"):
            x,y = [np.array([float(q[key]) for q in group]) for group in groups]
            rng = np.random.default_rng(20260915)
            boot = []
            for _ in range(100):
                boot.extend(y[rng.integers(len(y),size=(100,len(y)))].mean(1)
                            - x[rng.integers(len(x),size=(100,len(x)))].mean(1))
            group_result[key] = {"validation_minus_train":float(y.mean()-x.mean()),
                                "ci95_exploratory":np.quantile(boot,[.025,.975]).tolist()}
        names = ("train32","validation32") if sampled else ("train8500","validation500")
        h = [np.array(list(sessions[n].values()),dtype=float) for n in names]
        group_result["session_total_variation"] = float(np.abs(h[0]/h[0].sum()-h[1]/h[1].sum()).sum()/2)
        differences["sample32" if sampled else "full_pool"] = group_result
    write_json_atomic(out/"signal/distribution_differences.json",differences)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--phase",choices=["scores","join"],required=True)
    a=p.parse_args()
    (summarize if a.phase=="scores" else join_signal)(a.root)
