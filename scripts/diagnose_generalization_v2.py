#!/usr/bin/env python3
"""Read-only three-snapshot trajectory replay. No optimizer or training entry."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_condition_path import (TIMESTEPS, autocast, generate, jsonl,
                                     load_models, png)
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.inference import sample_seed
from neuroadapter_research.modeling import NeuroAdapterTrainingModule

WEIGHTS = {
    106250: "139dccbf42830845b24731efcc444642cbebe6fd27826ffceb7456908d8bf93c",
    159375: "909357e478b171347630f1d709268225661bcad904637edaa1379a8871d32180",
    239063: "bdca167505e0f1e62e025a5856299c56548dc40c2231740b8d2e1f84665b8217",
}
OLD_PROTOCOL_SHA = "8fe04623a09bf9861db8d6d0ea3863c25c51c0d688934787519b4fd2c685c6b5"
HELPER_SHA = "0ccfc1753d19490d722dd0e615a72df8b66dd989197264c44d61dee2a16b9aa0"


def locations(root):
    return (root / "runs/diagnostics/20260914-condition-path",
            root / "runs/diagnostics/generalization-diagnosis-v2")


def snapshot(config, step):
    if step not in WEIGHTS:
        raise ValueError("checkpoint is not in the three-snapshot whitelist")
    path = config.paths["output_dir"] / "snapshots" / f"snapshot-update-{step:08d}"
    if sha256_file(path / "model.pt") != WEIGHTS[step]:
        raise ValueError("checkpoint SHA mismatch")
    return path


def freeze(root, config):
    old, out = locations(root)
    assert sha256_file(old / "protocol.json") == OLD_PROTOCOL_SHA
    assert sha256_file(Path(__file__).with_name("diagnose_condition_path.py")) == HELPER_SHA
    protocol = json.loads((old / "protocol.json").read_text())
    bindings = []
    for step in WEIGHTS:
        snapshot(config, step)
    for split, pairs in protocol["pairs"].items():
        ids = np.loadtxt(config.paths["split_ids" if split == "train" else "validation_ids"], dtype=int)
        assert len(ids) == (8500 if split == "train" else 500)
        assert sha256_file(config.paths["split_ids" if split == "train" else "validation_ids"]) == protocol["split_hashes"][split]
        folder = old / split / "bf16"
        env = json.loads((folder / "environment.json").read_text())
        assert env["checkpoint_sha256"] == WEIGHTS[239063]
        assert env["protocol_sha256"] == OLD_PROTOCOL_SHA
        assert (folder / "complete.json").is_file()
        records = [json.loads(x) for x in (folder / "images.jsonl").read_text().splitlines()]
        expected = {(p["image_id"], p["donor_id"], c, j) for p in pairs
                    for c in ("correct", "wrong") for j in range(2)}
        assert len(records) == 128
        assert {(r["image_id"], r["donor_id"], r["condition"], r["candidate"]) for r in records} == expected
        for p in pairs:
            assert p["image_id"] in ids and p["donor_id"] in ids
            bank = folder / "noise" / f"{p['image_id']}.pt"
            tensors = torch.load(bank, map_location="cpu", weights_only=True)
            assert len(tensors) == 52 and all(tuple(t.shape) == (2, 4, 64, 64) and torch.isfinite(t).all() for t in tensors)
            for r in [r for r in records if r["image_id"] == p["image_id"]]:
                assert r["guidance"] == 4 and r["noise_sha256"] == sha256_file(bank)
                assert r["sha256"] == sha256_file(folder / r["path"])
                bindings.append({"split": split, **r, "source": str(folder / r["path"])})
    out.mkdir(parents=True, exist_ok=False)
    write_json_atomic(out / "config.json", {
        "name": "generalization-diagnosis-v2", "weights": WEIGHTS,
        "pairs": protocol["pairs"], "old_protocol_sha256": OLD_PROTOCOL_SHA,
        "config_sha256": config.sha256, "helper_sha256": HELPER_SHA,
        "script_sha256": sha256_file(Path(__file__)), "candidate_batch": 2,
        "candidate_count": 2, "steps": 50, "guidance": 4, "precision": "bf16",
        "timesteps": TIMESTEPS, "denoising_draws": 4, "priority_cosine_delta": 0.01,
        "training": False, "test_access": False, "checkpoint_reselection": False,
        "statistics": "candidate/draw mean within image; paired image bootstrap; fixed donor conditional",
        "quality_strata": "training-only repeat-specificity tertiles; no outcome-informed boundaries",
        "extreme_values": "abs(value - train vertex mean) > 5 * train vertex std",
    })
    write_json_atomic(out / "reused_images.json", bindings)
    print(f"verified 3 checkpoints, 64 noise banks and {len(bindings)} reusable PNGs", flush=True)


@torch.no_grad()
def run(root, config, split, device):
    old, out = locations(root)
    spec = json.loads((out / "config.json").read_text())
    assert spec["script_sha256"] == sha256_file(Path(__file__))
    assert spec["helper_sha256"] == sha256_file(Path(__file__).with_name("diagnose_condition_path.py"))
    assert json.loads((out / "signal/input_audit.json").read_text())["passed"]
    configure_torch_backend(config.training)
    dataset = Subject1TrainingDataset(config.paths["training_cache"], config.paths["stimuli"],
                                     config.paths["split_ids" if split == "train" else "validation_ids"])
    lookup = {int(v): i for i, v in enumerate(dataset.image_ids)}
    noise_dir = out / split / "denoising_noise"
    noise_dir.mkdir(parents=True, exist_ok=False)
    for pair in spec["pairs"][split]:
        image_id = pair["image_id"]
        g = torch.Generator(device="cpu").manual_seed(sample_seed("generalization-diagnosis-v2-loss", split, image_id, 0))
        bank = [{"epsilon": torch.randn((1, 4, 64, 64), generator=g, dtype=torch.bfloat16),
                 "noise": torch.randn((1, 4, 64, 64), generator=g, dtype=torch.bfloat16)} for _ in range(4)]
        torch.save(bank, noise_dir / f"{image_id}.pt")
    for step in WEIGHTS:
        start = time.time()
        target = out / split / str(step)
        target.mkdir(parents=True, exist_ok=False)
        backbone, bundle = load_models(config, snapshot(config, step), device, torch.bfloat16)
        model = NeuroAdapterTrainingModule(bundle).eval()
        assert not any(p.requires_grad for p in model.parameters())
        write_json_atomic(target / "environment.json", {
            "checkpoint_sha256": WEIGHTS[step], "config_sha256": sha256_file(out / "config.json"),
            "script_sha256": sha256_file(Path(__file__)), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device), "grad_enabled": torch.is_grad_enabled(),
        })
        empty_ids = backbone.tokenizer("", max_length=backbone.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        empty = backbone.text_encoder(empty_ids)[0]
        for index, pair in enumerate(spec["pairs"][split]):
            image_id = pair["image_id"]
            sample, donor = dataset[lookup[image_id]], dataset[lookup[pair["donor_id"]]]
            path = noise_dir / f"{image_id}.pt"
            bank = torch.load(path, weights_only=True)
            with autocast(torch.bfloat16):
                dist = backbone.vae.encode(sample["image"].unsqueeze(0).to(device, torch.bfloat16)).latent_dist
                for draw, noise in enumerate(bank):
                    latent = (dist.mean + dist.std * noise["epsilon"].to(device)) * backbone.vae.config.scaling_factor
                    epsilon = noise["noise"].to(device)
                    for t in TIMESTEPS:
                        ts = torch.tensor([t], device=device)
                        noisy = backbone.noise_scheduler.add_noise(latent, epsilon, ts)
                        losses = []
                        for item in (sample, donor):
                            pred = model(noisy, ts, empty, item["brain"].unsqueeze(0).to(device),
                                         torch.ones((1, 200, 1), device=device))
                            assert torch.isfinite(pred).all()
                            losses.append(float((pred.float() - epsilon.float()).square().mean()))
                        jsonl(target / "denoising.jsonl", {**pair, "draw": draw, "timestep": t,
                            "correct_mse": losses[0], "wrong_mse": losses[1],
                            "delta_mse": losses[1] - losses[0], "noise_sha256": sha256_file(path)})
            if step != 239063:
                path = old / split / "bf16/noise" / f"{image_id}.pt"
                bank = torch.load(path, weights_only=True)
                for condition, item in (("correct", sample), ("wrong", donor)):
                    values = generate(backbone, bundle, item["brain"], image_id, split, device,
                                      torch.bfloat16, 4., bank)
                    for j, value in enumerate(values):
                        rel = Path(condition) / f"{image_id}-{j}.png"
                        png(target / rel, value)
                        jsonl(target / "images.jsonl", {**pair, "condition": condition, "candidate": j,
                            "path": str(rel), "sha256": sha256_file(target / rel),
                            "noise_sha256": sha256_file(path), "guidance": 4})
            print(f"{split} step={step} image={index+1}/32 elapsed={time.time()-start:.1f}s", flush=True)
        write_json_atomic(target / "complete.json", {"seconds": time.time()-start, "denoising_pairs": 640,
            "new_images": 128 if step != 239063 else 0})
        del model, bundle, backbone
        torch.cuda.empty_cache()
    dataset.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=["freeze", "run"], required=True)
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    config = load_training_config(a.root / "configs/formal/subject01_selection_v2.yaml", require_frozen=True)
    if a.phase == "freeze":
        freeze(a.root, config)
    else:
        run(a.root, config, a.split, torch.device(a.device))


if __name__ == "__main__":
    main()
