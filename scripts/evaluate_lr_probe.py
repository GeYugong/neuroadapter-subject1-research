#!/usr/bin/env python3
"""Frozen endpoint comparison; never selects a checkpoint or starts training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import diagnose_condition_path as diag
from audit_signal_generalization_v2 import csv_rows
from summarize_condition_diagnosis import read_rows, gallery
from train_lr_probe import settings
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.checkpoint import verify_inference_snapshot
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.protocol import image_order_sha256, method_fingerprint
import evaluate_validation as ev


def paths(root):
    return root/"runs/experiments/paired-lr-probe-v1", root/"runs/diagnostics/20260914-condition-path"


def model_path(root,label):
    base,spec,_,_=settings(root,"H")
    out,_=paths(root)
    if label in ("B0","R"):
        step=159375 if label=="B0" else 239063
        path=base.paths["output_dir"]/"snapshots"/f"snapshot-update-{step:08d}"
        expected=spec["source_sha256" if label=="B0" else "reference_sha256"]
        assert sha256_file(path/"model.pt")==expected
    else:
        arm=label[0]; update=int(label[1:])
        assert arm in ("H","L") and update in (1000,2500,5000)
        assert json.loads((out/"arms"/arm/"status.json").read_text())["local_update"]==5000
        path=out/"arms"/arm/"snapshots"/f"snapshot-update-{update:08d}"
        verify_inference_snapshot(path)
        meta=json.loads((path/"metadata.json").read_text())
        assert meta["local_update"]==update and meta["arm"]==arm and meta["optimizer_reset"]
        assert meta["source_sha256"]==spec["source_sha256"] and meta["source_update"]==159375
    return path


@torch.no_grad()
def prepare(root):
    base,spec,_,_=settings(root,"H")
    out,old=paths(root)
    plan=json.loads(base.paths["selection_plan"].read_text())
    ids=np.loadtxt(base.paths["validation_ids"],dtype=int).tolist()
    assert len(ids)==500 and image_order_sha256(ids)==plan["image_order_sha256"]
    assert plan["screening_candidates"]==2 and plan["denoising_steps"]==50 and plan["guidance_scale"]==4
    target=out/"evaluation"
    target.mkdir(exist_ok=False)
    (target/"noise").mkdir()
    configure_torch_backend(base.training)
    ds=Subject1TrainingDataset(base.paths["training_cache"],base.paths["stimuli"],base.paths["validation_ids"])
    references={}
    for label,step in (("B0",159375),("R",239063)):
        path=base.paths["output_dir"]/"evaluation-20260910/screening"/f"update-{step:08d}"/"decode"
        manifest=json.loads((path/"decode_manifest.json").read_text())
        for k,v in {"candidate_count":2,"selection_stage":"screening","status":"complete",
            "denoising_steps":50,"guidance_scale":4,"config_sha256":base.sha256,
            "protocol_namespace":plan["protocol_namespace"],"repository_commit":base.raw["protocol_commit"],
            "snapshot_model_sha256":sha256_file(model_path(root,label)/"model.pt"),
            "method_fingerprint":method_fingerprint(base)}.items():
            assert manifest[k]==v,(k,manifest[k],v)
        assert [r["image_id"] for r in manifest["records"]]==ids
        for r in manifest["records"]:
            assert [f["candidate_index"] for f in r["files"]]==[0,1]
            assert all(sha256_file(path/f["path"])==f["sha256"] for f in r["files"])
        references[label]={"decode_root":str(path),"manifest_sha256":sha256_file(path/"decode_manifest.json")}
        backbone,bundle=diag.load_models(base,model_path(root,label),torch.device("cuda:0"),torch.bfloat16)
        with patch.object(diag,"NAMESPACE",plan["protocol_namespace"]):
            if label=="B0":
                for i in ids:
                    torch.save(diag.prepare_noise(backbone,i,"validation",torch.device("cuda:0")),target/"noise"/f"{i}.pt")
            checks=[]
            for j in range(2):
                sample=ds[j]; i=int(sample["nsd_image_id"])
                bank=torch.load(target/"noise"/f"{i}.pt",weights_only=True)
                generated=diag.generate(backbone,bundle,sample["brain"],i,"validation",torch.device("cuda:0"),torch.bfloat16,4.,bank)
                for c in range(2):
                    arr=(generated[c].permute(1,2,0).float().numpy()*255).round().astype(np.uint8)
                    expected=np.array(Image.open(path/manifest["records"][j]["files"][c]["path"]))
                    assert np.array_equal(arr,expected),"existing screening decode fails exact replay"
                    checks.append({"image_id":i,"candidate":c,"pixels_exact":True})
            references[label]["replay_checks"]=checks
        del backbone,bundle
        torch.cuda.empty_cache()
    write_json_atomic(target/"protocol.json",{"ids":ids,"references":references,
        "namespace":plan["protocol_namespace"],"config_sha256":base.sha256,
        "noise_sha256":{str(i):sha256_file(target/"noise"/f"{i}.pt") for i in ids},
        "spec":spec["evaluation"],"exploratory_internal_validation":True,
        "diagnostic_protocol_sha256":sha256_file(old/"protocol.json")})
    ds.close()
    print("Verified 2000 existing reference PNGs; exact replay 8/8; stored 500 noise banks",flush=True)


@torch.no_grad()
def decode(root,arm):
    base,spec,_,_=settings(root,arm)
    out,old=paths(root)
    configure_torch_backend(base.training)
    protocol=json.loads((out/"evaluation/protocol.json").read_text())
    old_spec=json.loads((old/"protocol.json").read_text())
    for update in (1000,2500,5000):
        label=f"{arm}{update}"
        snapshot=model_path(root,label)
        snapshot_sha=sha256_file(snapshot/"model.pt")
        backbone,bundle=diag.load_models(base,snapshot,torch.device("cuda:0"),torch.bfloat16)
        for split in ("train","validation"):
            ds=Subject1TrainingDataset(base.paths["training_cache"],base.paths["stimuli"],base.paths["split_ids" if split=="train" else "validation_ids"])
            lookup={int(v):j for j,v in enumerate(ds.image_ids)}
            target=out/"diagnostic"/label/split
            target.mkdir(parents=True,exist_ok=False)
            for pair in old_spec["pairs"][split]:
                i=pair["image_id"]; noise=old/split/"bf16/noise"/f"{i}.pt"
                bank=torch.load(noise,weights_only=True)
                for condition in (["correct","wrong"] if update==5000 else ["correct"]):
                    item=ds[lookup[i if condition=="correct" else pair["donor_id"]]]
                    values=diag.generate(backbone,bundle,item["brain"],i,split,torch.device("cuda:0"),torch.bfloat16,4.,bank)
                    for c,value in enumerate(values):
                        rel=f"{condition}/{i}-{c}.png"; diag.png(target/rel,value)
                        diag.jsonl(target/"images.jsonl",{**pair,"condition":condition,"candidate":c,"path":rel,
                            "sha256":sha256_file(target/rel),"noise_sha256":sha256_file(noise),"model_sha256":snapshot_sha})
            ds.close()
        if update==5000:
            target=out/"evaluation"/label
            target.mkdir(exist_ok=False)
            ds=Subject1TrainingDataset(base.paths["training_cache"],base.paths["stimuli"],base.paths["validation_ids"])
            records=[]
            with patch.object(diag,"NAMESPACE",protocol["namespace"]):
                for j,i in enumerate(protocol["ids"]):
                    sample=ds[j]; assert int(sample["nsd_image_id"])==i
                    noise=out/"evaluation/noise"/f"{i}.pt"
                    assert sha256_file(noise)==protocol["noise_sha256"][str(i)]
                    values=diag.generate(backbone,bundle,sample["brain"],i,"validation",torch.device("cuda:0"),
                        torch.bfloat16,4.,torch.load(noise,weights_only=True))
                    files=[]
                    for c,value in enumerate(values):
                        rel=f"candidate-{c:02d}/{i:05d}.png"; diag.png(target/rel,value)
                        files.append({"candidate_index":c,"path":rel,"sha256":sha256_file(target/rel)})
                    records.append({"image_id":i,"files":files})
                    if (j+1)%25==0: print(f"{label} full validation {j+1}/500",flush=True)
            write_json_atomic(target/"decode_manifest.json",{"status":"complete","split":"validation",
                "candidate_count":2,"records":records,"model_sha256":sha256_file(snapshot/"model.pt"),
                "experiment_protocol_sha256":sha256_file(out/"evaluation/protocol.json")})
            ds.close()
        del backbone,bundle
        torch.cuda.empty_cache()
    write_json_atomic(out/f"decode-{arm}-complete.json",{"status":"complete"})


@torch.no_grad()
def score(root,label):
    base,spec,_,_=settings(root,"H")
    out,_=paths(root)
    configure_torch_backend(base.training)
    p=json.loads((out/"evaluation/protocol.json").read_text())
    folder=Path(p["references"][label]["decode_root"]) if label in ("B0","R") else out/"evaluation"/label
    ids,gt,images,_=ev.load_decode_set(folder/"decode_manifest.json",base.paths["stimuli"])
    assert ids==p["ids"] and len(images)==2
    assets=ev.verify_evaluation_assets(root,base.paths["evaluation_manifest"])
    os.environ["TORCH_HOME"]=str(root/"models/evaluation/torch")
    torch.hub.set_dir(str(root/"models/evaluation/torch/hub"))
    plan=json.loads(base.paths["selection_plan"].read_text())
    batch=int(plan["evaluation_batch_size"])
    device=torch.device("cuda:0")
    scores={"PixCorr":[],"SSIM":[]}
    for im in images:
        pc,ss=ev.pixel_metrics(gt,im); scores["PixCorr"].append(pc); scores["SSIM"].append(ss)
    def feats(model,size,mean,std,key=None):
        return ev.model_features(gt,images,model,ev.preprocess_transform(size,mean,std),device=device,batch_size=batch,output_key=key)
    alex=ev.create_feature_extractor(ev.alexnet(weights=ev.AlexNet_Weights.IMAGENET1K_V1),
        return_nodes={"features.4":"layer2","features.11":"layer5"}).to(device).eval().requires_grad_(False)
    for key,name in (("layer2","AlexNet2"),("layer5","AlexNet5")):
        orig,cand=feats(alex,256,ev.IMAGENET_MEAN,ev.IMAGENET_STD,key)
        scores[name]=ev.identification_by_seed(orig,cand)
    del alex,orig,cand
    model=ev.create_feature_extractor(ev.inception_v3(weights=ev.Inception_V3_Weights.DEFAULT),return_nodes={"avgpool":"avgpool"}).to(device).eval().requires_grad_(False)
    orig,cand=feats(model,342,ev.IMAGENET_MEAN,ev.IMAGENET_STD,"avgpool")
    scores["Inception"]=ev.identification_by_seed(orig,cand)
    del model,orig,cand
    sys.path.insert(0,str(root/"repo/vendor/CLIP")); import clip
    model,_=clip.load(str(assets["clip_vit_l_14"]),device=device,jit=False)
    orig,cand=feats(model.encode_image,224,ev.CLIP_MEAN,ev.CLIP_STD)
    scores["CLIP_identification"]=ev.identification_by_seed(orig,cand)
    norm=lambda f:f/np.linalg.norm(f,axis=1,keepdims=True)
    scores["CLIP_cosine"]=[np.sum(norm(orig)*norm(f),axis=1) for f in cand]
    del model,orig,cand
    for name,factory,size in [
        ("EfficientNet",lambda:ev.efficientnet_b1(weights=ev.EfficientNet_B1_Weights.DEFAULT),255),
        ("SwAV",lambda:torch.hub.load(str(root/"repo/vendor/swav"),"resnet50",source="local",pretrained=True),224)]:
        model=ev.create_feature_extractor(factory(),return_nodes={"avgpool":"avgpool"}).to(device).eval().requires_grad_(False)
        orig,cand=feats(model,size,ev.IMAGENET_MEAN,ev.IMAGENET_STD,"avgpool")
        scores[name]=[ev.paired_correlation_distance(orig,f) for f in cand]
        del model,orig,cand
    rows=[{"image_id":i,"candidate":c,**{m:float(v[c][j]) for m,v in scores.items()}} for j,i in enumerate(ids) for c in range(2)]
    csv_rows(out/"evaluation"/f"{label}-candidate-scores.csv",rows)
    means=[{"image_id":i,**{m:float(np.mean([v[c][j] for c in range(2)])) for m,v in scores.items()}} for j,i in enumerate(ids)]
    csv_rows(out/"evaluation"/f"{label}-image-scores.csv",means)
    write_json_atomic(out/"evaluation"/f"{label}-summary.json",{m:float(np.mean(v)) for m,v in scores.items()})


def decision(root):
    import csv
    out,_=paths(root)
    p=json.loads((out/"evaluation/protocol.json").read_text())
    values={}
    for label in ("B0","H5000","L5000","R"):
        with (out/"evaluation"/f"{label}-image-scores.csv").open() as f: rows=list(csv.DictReader(f))
        assert [int(r["image_id"]) for r in rows]==p["ids"]
        values[label]=np.array([float(r["CLIP_cosine"]) for r in rows])
    rng=np.random.default_rng(20260915)
    indices=rng.integers(500,size=(10000,500))
    results={}
    for baseline in ("B0","H5000","R"):
        delta=values["L5000"]-values[baseline]
        results["L5000_minus_"+baseline]={"mean":float(delta.mean()),
            "individual_ci97_5":np.quantile(delta[indices].mean(1),[.0125,.9875]).tolist()}
    useful=(results["L5000_minus_B0"]["mean"]>=.01 and
            all(results[k]["individual_ci97_5"][0]>0 for k in ("L5000_minus_B0","L5000_minus_H5000")))
    write_json_atomic(out/"evaluation/decision.json",{"comparisons":results,"useful_primary_signal":useful,
        "next_recommendation":"second paired training seed" if useful else "semantic probe C",
        "next_action_executed":False,"automatic_weight_replacement":False,
        "qualification":"exploratory internal validation; inspect auxiliary metrics and visual tradeoffs; one paired training seed only"})
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--phase",choices=["prepare","decode","score","decision"],required=True)
    p.add_argument("--arm",choices=["H","L"]); p.add_argument("--label",choices=["B0","R","H5000","L5000"])
    a=p.parse_args()
    if a.phase=="prepare":prepare(a.root)
    elif a.phase=="decode":decode(a.root,a.arm)
    elif a.phase=="score":score(a.root,a.label)
    else:decision(a.root)
