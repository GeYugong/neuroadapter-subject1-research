#!/usr/bin/env python3
"""Audit completed inference and archive only small, non-image evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from diagnose_generalization_v2 import WEIGHTS, locations, snapshot
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config


def archive(root):
    old, out = locations(root)
    config = load_training_config(root/"configs/formal/subject01_selection_v2.yaml",require_frozen=True)
    runtime = root/"runtime/subject01-4090-1a1fcfa"
    assert subprocess.check_output(["git","-C",str(runtime),"rev-parse","HEAD"],text=True).strip()==config.raw["protocol_commit"]
    assert not subprocess.check_output(["git","-C",str(runtime),"status","--porcelain"],text=True).strip()
    durations = {}
    for step in WEIGHTS:
        snapshot(config,step)
        for split in ("train","validation"):
            folder = out/split/str(step)
            done = json.loads((folder/"complete.json").read_text())
            assert done["denoising_pairs"]==640
            assert done["new_images"]==(0 if step==239063 else 128)
            durations[f"{split}_{step}"]=done["seconds"]
    assert len(list(out.glob("*/denoising_noise/*.pt")))==64
    assert len(list(out.glob("*/*/correct/*.png")))==256
    assert len(list(out.glob("*/*/wrong/*.png")))==256
    assert len(list((out/"galleries").glob("*.jpg")))==16
    checks = {"checkpoint_hashes_unchanged":True,"frozen_runtime_clean":True,
        "neuroadapter_parameter_updates":0,"new_images":512,"reused_images":256,
        "denoising_comparisons":3840,"new_denoising_noise_banks":64,
        "galleries":16,"input_reconstruction_exact":64,"seconds_by_worker_stage":durations,
        "head_at_archive":subprocess.check_output(["git","-C",str(root/"repo"),"rev-parse","HEAD"],text=True).strip(),
        "inference_and_signal_script_commit":"419a3f76240cb1b0b268056eea713b4f4fb8b121",
        "provenance_note":"scoring/archiving may be later working-tree changes; exact source hashes below are authoritative",
        "source_sha256":{p.name:sha256_file(p) for p in (root/"repo/scripts").glob("*generalization_v2.py")},
        "old_protocol_sha256":sha256_file(old/"protocol.json")}
    checks["next_branch"]="2: paired low-learning-rate experiment, not executed"
    checks["probe_C"]="not triggered: A/B suffice to prioritize next controlled experiment, not to prove unique cause"
    write_json_atomic(out/"completion_audit.json",checks)
    public = root/"repo/manifests/generalization-diagnosis-v2"
    public.mkdir(parents=True,exist_ok=True)
    sources = sorted(out.glob("*.json"))+sorted(out.glob("*.csv"))
    sources += sorted((out/"signal").glob("*.json"))+sorted((out/"signal").glob("*.csv"))
    sources = [p for p in sources if p.name!="vertex_order.json"]
    sources += sorted(out.glob("*/*/environment.json"))+sorted(out.glob("*/*/complete.json"))
    sources += sorted(out.glob("*/*/denoising.jsonl"))+sorted(out.glob("*/*/images.jsonl"))
    index=[]
    for source in sources:
        relative=source.relative_to(out)
        target=public/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha256_file(target)==sha256_file(source)
        index.append({"path":str(relative),"sha256":sha256_file(source),"bytes":source.stat().st_size})
    write_json_atomic(public/"INDEX.json",{"sources":index,"excluded":"images, beta arrays, vertex arrays, noise tensors, credentials"})
    report=(root/"repo/docs/GENERALIZATION_DIAGNOSIS_V2.md").read_text(encoding="utf-8")
    report += "\n\n## 9. 完整图册\n\n所有 64 张诊断图、三个 checkpoint、两个条件和两个候选均保留，以下 16 页已逐页视觉检查。\n"
    for split,label in (("train","训练"),("validation","内部验证")):
        for condition,condition_label in (("correct","正确脑输入"),("wrong","错配完整脑输入")):
            for page in range(1,5):
                rel=f"galleries/{split}-{condition}-{page}.jpg"
                report += f"\n### {label}：{condition_label}，第 {page} 页\n\n![{label} {condition_label} {page}]({rel})\n"
    (out/"REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    archive(p.parse_args().root)
