#!/usr/bin/env python3
"""Compare the new adapter forward/loss path with the pinned upstream path."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.modeling import (
    NeuroAdapterTrainingModule,
    build_adapter,
    load_frozen_backbone,
    load_trainable_state_dict,
)
from neuroadapter_research.protocol import load_gate_requirements, method_fingerprint
from neuroadapter_research.trainer import min_snr_weights


def fixed_inputs(max_voxels: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260901)
    return {
        "noisy": torch.randn((1, 4, 64, 64), generator=generator),
        "timesteps": torch.tensor([731], dtype=torch.long),
        "text": torch.randn((1, 77, 768), generator=generator),
        "brain": torch.randn((1, 200, max_voxels), generator=generator),
        "mask": (
            torch.rand((1, 200, 1), generator=generator) > 0.35
        ),
        "target": torch.randn((1, 4, 64, 64), generator=generator),
    }


def load_upstream_setup_ip_adapter(vendor: Path):
    """Execute only the pinned function body; the full upstream module has a bad import."""

    vendor = vendor.resolve()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    source_path = vendor / "train_brain_adapter.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup_ip_adapter"
    ]
    if len(matches) != 1:
        raise ValueError("pinned upstream source has no unique setup_ip_adapter function")
    function_node = matches[0]
    function_source = ast.get_source_segment(source, function_node)
    if function_source is None:
        raise ValueError("cannot recover the pinned setup_ip_adapter source")

    from brain_adapter.ip_adapter.ip_adapter import ImageProjModel
    from brain_adapter.ip_adapter.utils import is_torch2_available

    if is_torch2_available():
        from brain_adapter.ip_adapter.attention_processor import (
            AttnProcessor2_0 as AttnProcessor,
        )
        from brain_adapter.ip_adapter.attention_processor import (
            IPAttnProcessor2_0 as IPAttnProcessor,
        )
    else:
        from brain_adapter.ip_adapter.attention_processor import (
            AttnProcessor,
            IPAttnProcessor,
        )

    namespace = {
        "torch": torch,
        "ImageProjModel": ImageProjModel,
        "AttnProcessor": AttnProcessor,
        "IPAttnProcessor": IPAttnProcessor,
    }
    isolated = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return (
        namespace["setup_ip_adapter"],
        sha256_file(source_path),
        hashlib.sha256(function_source.encode("utf-8")).hexdigest(),
    )


@torch.no_grad()
def run_current(
    model_path: Path,
    state: dict,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = load_frozen_backbone(model_path)
    bundle = build_adapter(backbone.unet, 200, inputs["brain"].shape[-1])
    load_trainable_state_dict(bundle, state)
    module = NeuroAdapterTrainingModule(bundle).to(device).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = module(
            inputs["noisy"].to(device),
            inputs["timesteps"].to(device),
            inputs["text"].to(device),
            inputs["brain"].to(device),
            inputs["mask"].to(device),
        )
    weights = min_snr_weights(
        inputs["timesteps"].to(device),
        backbone.noise_scheduler.alphas_cumprod.to(device),
        5.0,
    )
    loss = (
        F.mse_loss(
            prediction.float(), inputs["target"].to(device), reduction="none"
        ).mean(dim=(1, 2, 3))
        * weights
    ).mean()
    result = prediction.float().cpu(), loss.float().cpu()
    del module, bundle, backbone, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def run_upstream(
    repository: Path,
    model_path: Path,
    state: dict,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    vendor = repository / "vendor/NeuroAdapter"
    sys.path.insert(0, str(vendor))
    from brain_adapter.loss import min_snr_loss_weights
    from brain_adapter.model import GuidanceGenerator, NeuroAdapter

    setup_ip_adapter, source_sha256, function_sha256 = load_upstream_setup_ip_adapter(
        vendor
    )

    backbone = load_frozen_backbone(model_path)
    args = SimpleNamespace(topk=100, condition_dim=768)
    image_projection, adapter_modules, token_count = setup_ip_adapter(backbone.unet, args)
    if token_count != 200:
        raise ValueError("upstream adapter did not construct 200 tokens")
    neuro_adapter = NeuroAdapter(
        backbone.unet, image_projection, adapter_modules, ckpt_path=None
    )
    guidance = GuidanceGenerator(
        num_parcels=200,
        max_voxels=inputs["brain"].shape[-1],
        num_decoder_queries=50,
        output_dim=768,
        sub_approach="linear_projection",
    )
    image_projection.load_state_dict(state["image_proj"], strict=True)
    adapter_modules.load_state_dict(state["ip_adapter"], strict=True)
    guidance.load_state_dict(state["guidance_generator"], strict=True)
    neuro_adapter.to(device).eval()
    guidance.to(device).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        condition, _ = guidance(inputs["brain"].to(device))
        condition = condition * inputs["mask"].to(device)
        prediction, _ = neuro_adapter(
            inputs["noisy"].to(device),
            inputs["timesteps"].to(device),
            inputs["text"].to(device),
            condition,
        )
    weights = min_snr_loss_weights(
        inputs["timesteps"].to(device), backbone.noise_scheduler, gamma=5.0
    )
    loss = (
        F.mse_loss(
            prediction.float(), inputs["target"].to(device), reduction="none"
        ).mean(dim=(1, 2, 3))
        * weights
    ).mean()
    return prediction.float().cpu(), loss.float().cpu(), source_sha256, function_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"forward gate output already exists: {args.output}")
    config = load_training_config(args.config, require_frozen=True)
    requirements = load_gate_requirements(config.paths["gate_requirements"])
    tolerance = float(requirements.raw["forward_atol"])
    configure_torch_backend(config.training)
    repository = Path(__file__).resolve().parents[1]
    fingerprint = json.loads(config.paths["data_fingerprint"].read_text(encoding="utf-8"))
    max_voxels = int(fingerprint["max_voxels"])
    state = torch.load(
        config.paths["canonical_initialization"], map_location="cpu", weights_only=True
    )
    inputs = fixed_inputs(max_voxels)
    device = torch.device(args.device)
    current_prediction, current_loss = run_current(
        config.paths["stable_diffusion"], state, inputs, device
    )
    upstream_prediction, upstream_loss, upstream_source, upstream_function = run_upstream(
        repository, config.paths["stable_diffusion"], state, inputs, device
    )
    prediction_error = float((current_prediction - upstream_prediction).abs().max())
    loss_error = float((current_loss - upstream_loss).abs())
    if prediction_error > tolerance or loss_error > tolerance:
        raise ValueError(
            f"forward alignment failed: prediction={prediction_error}, loss={loss_error}"
        )
    payload = {
        "schema_version": 1,
        "gate": "forward_alignment",
        "status": "passed",
        "config_sha256": config.sha256,
        "method_fingerprint": method_fingerprint(config),
        "gate_requirements_sha256": requirements.sha256,
        "canonical_initialization_sha256": sha256_file(
            config.paths["canonical_initialization"]
        ),
        "input_seed": 20260901,
        "dtype": "bfloat16 autocast",
        "prediction_max_abs_error": prediction_error,
        "loss_abs_error": loss_error,
        "absolute_tolerance": tolerance,
        "upstream_train_script_sha256": upstream_source,
        "upstream_setup_ip_adapter_sha256": upstream_function,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
