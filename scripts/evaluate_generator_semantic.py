"""Frozen two-candidate validation and pilot gate for K/M generator snapshots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import diagnose_condition_path as diag
import evaluate_retrain_lr as old_eval
from generator_semantic_core import base_config, output, source_snapshot, spec
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import verify_inference_snapshot
from neuroadapter_research.data import Subject1TrainingDataset
from run_residual_rerank import save_csv


def snapshot(root: Path, label: str) -> Path:
    if label == "B0": return source_snapshot(root)
    arm, update = label.split("-"); update = int(update)
    assert arm in ("K", "M") and update in spec()["snapshot_updates"]
    path = output(root)/arm/"snapshots"/f"snapshot-update-{update:08d}"
    verify_inference_snapshot(path)
    metadata = json.loads((path/"metadata.json").read_text())
    assert metadata["arm"] == arm and metadata["local_update"] == update
    assert metadata["source_R_sha256"] == spec()["source_sha256"]
    return path


def protocol(root: Path) -> dict:
    target = output(root)/"evaluation/protocol.json"
    if target.exists(): return json.loads(target.read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    old = root/"runs/experiments/retrain-lr-v1/evaluation/protocol.json"
    value = json.loads(old.read_text())
    assert len(value["ids"]) == 500 and value["candidate_count"] == 2
    value = {"ids":value["ids"], "noise_paths":value["noise_paths"],
             "noise_sha256":value["noise_sha256"], "namespace":value["namespace"],
             "candidate_count":2,"candidate_batch_size":2,"denoising_steps":50,"guidance":4,
             "visual_ids":value["visual_ids"],"source_protocol_sha256":sha256_file(old),
             "selection":"none; score each candidate then average within image"}
    write_json_atomic(target,value); return value


@torch.no_grad()
def decode(root: Path, label: str) -> None:
    p, cfg = protocol(root), base_config(root)
    target = output(root)/"evaluation"/label
    if (target/"decode_manifest.json").exists(): return
    target.mkdir(parents=True, exist_ok=True); configure_torch_backend(cfg.training)
    source = snapshot(root,label); source_hash=sha256_file(source/"model.pt")
    backbone,bundle=diag.load_models(cfg,source,torch.device("cuda:0"),torch.bfloat16)
    ds=Subject1TrainingDataset(cfg.paths["training_cache"],cfg.paths["stimuli"],cfg.paths["validation_ids"])
    records=[]
    with patch.object(diag,"NAMESPACE",p["namespace"]):
        for index,image_id in enumerate(p["ids"]):
            record=target/f"{image_id}.json"
            if record.exists(): records.append(json.loads(record.read_text()));continue
            noise=root/p["noise_paths"][str(image_id)]
            assert sha256_file(noise)==p["noise_sha256"][str(image_id)]
            sample=ds[index]; assert int(sample["nsd_image_id"])==image_id
            values=diag.generate(backbone,bundle,sample["brain"],image_id,"validation",
                torch.device("cuda:0"),torch.bfloat16,4.,torch.load(noise,weights_only=True))
            files=[]
            for candidate,value in enumerate(values):
                relative=f"candidate-{candidate:02d}/{image_id:05d}.png";diag.png(target/relative,value)
                files.append({"candidate_index":candidate,"path":relative,"sha256":sha256_file(target/relative)})
            item={"image_id":image_id,"files":files,"source_sha256":source_hash}
            write_json_atomic(record,item);records.append(item)
            if (index+1)%25==0: print(f"{label}: {index+1}/500",flush=True)
    write_json_atomic(target/"decode_manifest.json",{"status":"complete","records":records,
        "candidate_count":2,"snapshot_sha256":source_hash,"protocol_sha256":sha256_file(output(root)/"evaluation/protocol.json")})
    ds.close()


@torch.no_grad()
def score(root: Path,label: str) -> None:
    out=output(root)/"evaluation"; result=out/f"{label}-summary.json"
    if result.exists(): return
    cfg=base_config(root);p=protocol(root);folder=out/label
    if label=="B0":
        previous=root/"runs/experiments/retrain-lr-v1/evaluation"
        # Reuse only the exact fixed protocol already replay-verified in the preceding experiment.
        assert sha256_file(previous/"protocol.json")==p["source_protocol_sha256"]
        for suffix in ("summary.json","candidate-scores.csv","image-scores.csv"):
            source=previous/f"OLD-239063-{suffix}"; target=out/f"B0-{suffix}"
            target.write_bytes(source.read_bytes())
        return
    configure_torch_backend(cfg.training)
    ids,gt,images,_=old_eval.ev.load_decode_set(folder/"decode_manifest.json",cfg.paths["stimuli"])
    assert ids==p["ids"] and len(images)==2
    scores=old_eval.score_arrays(root,cfg,gt,images)
    rows=[{"image_id":image_id,"candidate":candidate,
           **{name:float(values[candidate][index]) for name,values in scores.items()}}
          for index,image_id in enumerate(ids) for candidate in range(2)]
    means=[{"image_id":image_id,**{name:float(np.mean([values[c][index] for c in range(2)]))
            for name,values in scores.items()}} for index,image_id in enumerate(ids)]
    save_csv(out/f"{label}-candidate-scores.csv",rows);save_csv(out/f"{label}-image-scores.csv",means)
    write_json_atomic(result,{"mean":{name:float(np.mean(values)) for name,values in scores.items()},
                              "image_count":500,"candidate_count":2,"selection":"none"})


def summarize(root: Path, labels: list[str], gate: bool=False) -> dict:
    out=output(root)/"evaluation"; summaries={label:json.loads((out/f"{label}-summary.json").read_text()) for label in labels}
    rows=[]
    for label,value in summaries.items():
        mean=value["mean"]; semantic=np.mean([mean["AlexNet5"],mean["Inception"],mean["CLIP_identification"]])
        nonclip=np.mean([mean["AlexNet5"],mean["Inception"]])
        rows.append({"label":label,"semantic_score":float(semantic),"non_clip_high_level":float(nonclip),**mean})
    result={"rows":rows}
    if gate:
        by={row["label"]:row for row in rows};b,k,m=by["B0"],by["K-5000"],by["M-5000"];g=spec()["gate"]
        checks={"semantic_vs_baseline":m["semantic_score"]-b["semantic_score"]>=g["semantic_score_vs_baseline_pp"],
            "semantic_vs_control":m["semantic_score"]-k["semantic_score"]>=g["semantic_score_vs_control_pp"],
            "non_clip_above_both":m["non_clip_high_level"]>b["non_clip_high_level"] and m["non_clip_high_level"]>k["non_clip_high_level"],
            "pixcorr_floor":m["PixCorr"]-b["PixCorr"]>=g["pixcorr_floor_vs_baseline"]}
        result["gate"]={"passed_numeric":all(checks.values()),"checks":checks,
            "deltas":{"S_vs_B0":m["semantic_score"]-b["semantic_score"],"S_vs_K":m["semantic_score"]-k["semantic_score"],
                      "N_vs_B0":m["non_clip_high_level"]-b["non_clip_high_level"],"N_vs_K":m["non_clip_high_level"]-k["non_clip_high_level"],
                      "PixCorr_vs_B0":m["PixCorr"]-b["PixCorr"]}}
    write_json_atomic(out/("pilot-gate.json" if gate else "final-summary.json"),result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--phase",choices=["decode","score","pilot-summary","final-summary"],required=True)
    parser.add_argument("--label");args=parser.parse_args()
    if args.phase=="decode":decode(args.root,args.label)
    elif args.phase=="score":score(args.root,args.label)
    elif args.phase=="pilot-summary":summarize(args.root,["B0","K-5000","M-5000"],True)
    else:
        labels=["B0","K-5000","M-5000"]+[label for label in ("M-10000","M-20000") if (output(args.root)/"evaluation"/f"{label}-summary.json").exists()]
        summarize(args.root,labels,False)

