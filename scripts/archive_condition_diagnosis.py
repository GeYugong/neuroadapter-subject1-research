#!/usr/bin/env python3
"""CPU-only completion audit and explicit text-only public export."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.protocol import verify_protocol_repository


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(root):
    out = root / "runs/diagnostics/20260914-condition-path"
    protocol = json.loads((out / "protocol.json").read_text())
    runtime = root / "runtime/subject01-4090-1a1fcfa"
    verify_protocol_repository(runtime, protocol["runtime_commit"])
    assert sha256_file(Path(protocol["checkpoint"]) / "model.pt") == protocol["checkpoint_sha256"]
    amendment = json.loads((out / "code_amendment.json").read_text())
    assert amendment["protocol_sha256"] == sha256_file(out / "protocol.json")
    assert amendment["diagnostic_script_sha256"] == sha256_file(root / "repo/scripts/diagnose_condition_path.py")
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["generation"]) == 4 and len(summary["precision"]) == 2
    assert len(summary["guidance"]) == 4 and len(summary["galleries"]) == 32
    precision_audit = json.loads((out / "fp32_reference_audit.json").read_text())
    assert len(precision_audit["cases"]) == 2
    assert all(x["explicitly_disabled_equal"] and x["saved_png_equal"] for x in precision_audit["cases"])
    evidence = []
    images = []
    noises = []
    phases = []
    for split in ("train", "validation"):
        pairs = protocol["pairs"][split]
        donor_map = {p["image_id"]: p["donor_id"] for p in pairs}
        assert len(donor_map) == 32 and all(i != j for i, j in donor_map.items())
        equivalent = json.loads((out / split / "bf16/noise_equivalence.json").read_text())
        assert equivalent["original_vs_replay_exact"] and equivalent["max_difference"] == 0
        evidence.append(out / split / "bf16/noise_equivalence.json")
        for image_id in donor_map:
            path = out / split / "bf16/noise" / f"{image_id}.pt"
            tensors = torch.load(path, map_location="cpu", weights_only=True)
            assert len(tensors) == 52
            assert [x.dtype for x in tensors] == [torch.bfloat16, torch.float32] + [torch.bfloat16] * 50
            assert all(x.shape == (2, 4, 64, 64) and torch.isfinite(x).all() for x in tensors)
            noises.append({"split": split, "image_id": image_id, "path": str(path.relative_to(out)),
                           "sha256": sha256_file(path), "draw_count": 52})
        loss_rows = rows(out / split / "bf16/denoising.jsonl")
        assert {(x["image_id"], x["timestep"]) for x in loss_rows} == {
            (image_id, t) for image_id in donor_map for t in protocol["timesteps"]}
        evidence.extend([out / split / "bf16/denoising.jsonl", out / split / "denoising.csv"])
        for phase in ("vae", "bf16", "fp32", "guidance"):
            folder = out / split / phase
            complete = json.loads((folder / "complete.json").read_text())
            assert complete["status"] == "complete"
            env = json.loads((folder / "environment.json").read_text())
            assert env["checkpoint_sha256"] == protocol["checkpoint_sha256"]
            assert env["protocol_sha256"] == sha256_file(out / "protocol.json")
            assert env["script_sha256"] == (protocol["diagnostic_script_sha256"] if phase == "vae"
                                            else amendment["diagnostic_script_sha256"])
            assert env["actual_tf32"] == (phase not in ("vae", "fp32"))
            phases.append({"split": split, "phase": phase, **complete})
            evidence.extend([folder / "complete.json", folder / "environment.json"])
            if phase == "vae":
                records = rows(folder / "metrics.jsonl")
                assert len(records) == 128 and all(x["finite"] for x in records)
                evidence.append(folder / "metrics.jsonl")
                continue
            records = rows(folder / "images.jsonl")
            assert len(records) == 128
            expected = {(image_id, c, g, condition) for image_id in donor_map for c in (0, 1)
                        for g, condition in ([(2., "correct"), (6., "correct")] if phase == "guidance"
                                             else [(4., "correct"), (4., "wrong")])}
            assert {(x["image_id"], x["candidate"], x["guidance"], x["condition"]) for x in records} == expected
            for record in records:
                assert record["donor_id"] == donor_map[record["image_id"]]
                noise_hash = next(n["sha256"] for n in noises if
                                  n["split"] == split and n["image_id"] == record["image_id"])
                assert record["noise_sha256"] == noise_hash
                path = folder / record["path"]
                assert sha256_file(path) == record["sha256"]
                images.append({"split": split, "phase": phase, **record,
                               "path": str(path.relative_to(out))})
            evidence.append(folder / "images.jsonl")
    assert len(images) == 768 and len(noises) == 64
    score_rows = list(csv.DictReader((out / "per_image_scores.csv").open()))
    assert len(score_rows) == 768
    def score_key(row):
        return (row["split"], row["phase"], int(row["image_id"]),
                int(row["candidate"]), float(row["guidance"]), row["condition"])
    image_map = {score_key(row): row for row in images}
    assert len({score_key(row) for row in score_rows}) == 768
    for row in score_rows:
        image = image_map[score_key(row)]
        assert row["sha256"] == image["sha256"] and row["noise_sha256"] == image["noise_sha256"]
        assert int(row["donor_id"]) == image["donor_id"]
    assert len(list(csv.DictReader((out / "paired_scores.csv").open()))) == 256
    assert len(list(out.glob("*/vae/*/*/*.png"))) == 256
    assert len(list(out.glob("*/gt/*.png"))) == 64
    for name in summary["galleries"]:
        assert (out / name).is_file()
    public = root / "repo/manifests/diagnosis-20260914"
    public.mkdir(parents=True, exist_ok=False)
    evidence.extend(out / name for name in ["protocol.json", "code_amendment.json", "stage4_gate.json", "guidance_gate.json", "fp32_reference_audit.json",
                    "vae_summary.json", "summary-bf16.json", "summary-fp32.json", "summary.json", "per_image_scores.csv", "paired_scores.csv"])
    index = []
    for path in evidence:
        relative = path.relative_to(out)
        destination = public / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert path.suffix in (".json", ".jsonl", ".csv")
        shutil.copy2(path, destination)
        index.append({"path": str(relative), "sha256": sha256_file(path)})
    write_json_atomic(public / "image_index.json", images)
    write_json_atomic(public / "noise_index.json", noises)
    evaluator_assets = json.loads((root / "data/fingerprints/evaluation_downloads.json").read_text())
    write_json_atomic(public / "metric_protocol.json", {
        "clip": "paired image embedding cosine, not identification accuracy",
        "clip_asset": evaluator_assets["files"]["clip_vit_l_14"],
        "ground_truth": "actual preprocessed 512x512 RGB, PNG quantized",
        "pixel_metrics": "frozen helper at resize 425; not a replacement for official evaluation",
        "aggregation": "average the two fixed candidates per image, then average 32 images",
        "ci": "10000 bootstrap draws over 32 image-level paired means, seed 20260914; exploratory and unadjusted",
        "guidance": "BF16 g2 and g6 vs BF16 g4; no per-image choice and no FP32-guidance interaction tested",
    })
    write_json_atomic(public / "INDEX.json", index)
    audit = {"status": "complete_and_verified", "runtime_clean": True, "weight_unchanged": True,
             "checkpoint_sha256": protocol["checkpoint_sha256"], "training_performed": False,
             "accepted_as_formal_model": False, "raw_generation_pngs": 768, "vae_outputs": 256,
             "gt_images": 64, "noise_banks": 64, "paired_denoising_comparisons": 320,
             "all_generated_image_hashes_verified": True, "all_noise_hash_bindings_verified": True,
             "phases": phases, "image_gallery_count": 32,
             "public_export": "JSON/JSONL/CSV only; no image, fMRI, noise tensor or credential",
             "summarizer_sha256": sha256_file(root / "repo/scripts/summarize_condition_diagnosis.py"),
             "audit_script_sha256": sha256_file(Path(__file__))}
    write_json_atomic(out / "completion_audit.json", audit)
    write_json_atomic(public / "completion_audit.json", audit)
    report = (root / "repo/docs/CONDITION_DIAGNOSIS_20260914.md").read_text(encoding="utf-8")
    report += "\n## 完整图像附录\n\n"
    report += "各页严格按冻结样本顺序排列，没有挑选成功样本。GT 是目标原图，correct/wrong 是正确/错配脑输入；相同 seed 编号固定同一候选噪声。Donor GT 是错配脑输入对应图，g2/g4/g6 表示 guidance。mode/sample 是 VAE 后验众数/采样。所有原始 PNG 仍在服务器保留。\n\n"
    for name in summary["galleries"]:
        split, phase, page = Path(name).stem.split("-")
        split_label = "训练组" if split == "train" else "验证组"
        phase_label = {"vae": "VAE 往返", "bf16": "当前精度条件对照", "fp32": "FP32 条件对照", "guidance": "Guidance 对照"}[phase]
        title = f"{split_label}：{phase_label}，第 {page} 页"
        report += f"### {title}\n\n![{title}]({name})\n\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    main(parser.parse_args().project_root)
