"""Audit FP32-target autocast against explicitly disabled autocast on fixed cases."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from diagnose_condition_path import generate, load_models
from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.data import Subject1TrainingDataset


def collect_dtypes(value):
    if isinstance(value, torch.Tensor):
        return {str(value.dtype)}
    if isinstance(value, dict):
        return set().union(*(collect_dtypes(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(collect_dtypes(v) for v in value))
    return set()


@torch.no_grad()
def main(root):
    out = root / "runs/diagnostics/20260914-condition-path"
    protocol = json.loads((out / "protocol.json").read_text())
    config = load_training_config(root / "configs/formal/subject01_selection_v2.yaml", require_frozen=True)
    configure_torch_backend(config.training)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    backbone, bundle = load_models(config, Path(protocol["checkpoint"]), device, torch.float32)
    modules = {"unet": backbone.unet, "text_encoder": backbone.text_encoder,
               "vae_encoder": backbone.vae.encoder, "vae_decoder": backbone.vae.decoder,
               "guidance_generator": bundle.guidance_generator,
               "image_projection": bundle.neuro_adapter.image_proj_model}
    results = []
    for split in ("train", "validation"):
        image_id = protocol["pairs"][split][0]["image_id"]
        dataset = Subject1TrainingDataset(config.paths["training_cache"], config.paths["stimuli"],
                                         config.paths["split_ids" if split == "train" else "validation_ids"])
        brain = dataset[list(dataset.image_ids).index(image_id)]["brain"]
        noise = torch.load(out / split / "bf16/noise" / f"{image_id}.pt", weights_only=True)
        captures = {}
        images = []
        for mode in ("fp32_target_autocast", "explicitly_disabled"):
            dtypes = {name: set() for name in modules}
            def hook(name):
                def record(module, inputs, output):
                    dtypes[name].update(collect_dtypes(output))
                return record
            handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
            if mode == "explicitly_disabled":
                # Only replace torch.autocast, not the underlying amp class.
                with patch.object(torch, "autocast", lambda *a, **kw: torch.amp.autocast("cuda", enabled=False)):
                    value = generate(backbone, bundle, brain, image_id, split, device, torch.float32, 4., noise)
            else:
                value = generate(backbone, bundle, brain, image_id, split, device, torch.float32, 4., noise)
            for handle in handles:
                handle.remove()
            assert all(types == {"torch.float32"} for types in dtypes.values()), dtypes
            captures[mode] = {name: sorted(types) for name, types in dtypes.items()}
            images.append(value)
        assert torch.equal(images[0], images[1])
        for candidate, value in enumerate(images[0]):
            array = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            existing = np.asarray(Image.open(out / split / "fp32/g4-correct" / f"{image_id}-{candidate}.png"))
            assert np.array_equal(array, existing)
        results.append({"split": split, "image_id": image_id, "candidate_count": 2,
                        "explicitly_disabled_equal": True, "max_difference": 0.,
                        "saved_png_equal": True, "activation_dtypes": captures})
        dataset.close()
    audit = {"torch": torch.__version__, "fp32_context_enabled": True,
             "fp32_autocast_target": "torch.float32", "tf32": False,
             "correction": "In PyTorch 2.11 the frozen generator's FP32-target autocast context remains enabled; it is not low-precision autocast. Two fixed first-image cases were tested against explicitly disabled autocast.",
             "scope": "four candidate tensors, six module output dtype families; not a rerun of all 64 images",
             "cases": results}
    write_json_atomic(out / "fp32_reference_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    main(parser.parse_args().project_root)
