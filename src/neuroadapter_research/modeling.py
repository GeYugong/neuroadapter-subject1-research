"""Construction and state handling for the fixed NeuroAdapter architecture."""

from __future__ import annotations

import itertools
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def install_vendor_path() -> Path:
    path = repository_root() / "vendor" / "NeuroAdapter"
    if not path.is_dir():
        raise FileNotFoundError(f"NeuroAdapter vendor tree is missing: {path}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path


@dataclass
class FrozenBackbone:
    noise_scheduler: DDPMScheduler
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel


@dataclass
class AdapterBundle:
    neuro_adapter: torch.nn.Module
    guidance_generator: torch.nn.Module

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return itertools.chain(
            self.neuro_adapter.image_proj_model.parameters(),
            self.neuro_adapter.adapter_modules.parameters(),
            self.guidance_generator.parameters(),
        )


class NeuroAdapterTrainingModule(torch.nn.Module):
    """One DDP boundary containing every trainable adapter component."""

    def __init__(self, bundle: AdapterBundle) -> None:
        super().__init__()
        self.neuro_adapter = bundle.neuro_adapter
        self.guidance_generator = bundle.guidance_generator

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        brain: torch.Tensor,
        token_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        condition_tokens, _ = self.guidance_generator(brain)
        condition_tokens = condition_tokens * token_keep_mask
        prediction, _ = self.neuro_adapter(
            noisy_latents, timesteps, text_embeddings, condition_tokens
        )
        return prediction

    def as_bundle(self) -> AdapterBundle:
        return AdapterBundle(
            neuro_adapter=self.neuro_adapter,
            guidance_generator=self.guidance_generator,
        )


def load_frozen_backbone(model_path: Path) -> FrozenBackbone:
    model_path = Path(model_path).resolve()
    required = ("scheduler", "tokenizer", "text_encoder", "vae", "unet")
    missing = [name for name in required if not (model_path / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Stable Diffusion snapshot is incomplete: {missing}")
    common = {"local_files_only": True}
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_path, subfolder="scheduler", **common
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", **common
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", **common
    )
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **common)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **common)
    for model in (text_encoder, vae, unet):
        model.requires_grad_(False)
        model.eval()
    return FrozenBackbone(
        noise_scheduler=noise_scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )


def build_adapter(
    unet: UNet2DConditionModel,
    num_parcels: int,
    max_voxels: int,
    condition_dim: int = 768,
    sub_approach: str = "linear_projection",
) -> AdapterBundle:
    if num_parcels != 200:
        raise ValueError(f"formal Subject 1 adapter requires 200 parcels, got {num_parcels}")
    if sub_approach != "linear_projection":
        raise ValueError("the frozen first-phase protocol uses linear_projection")
    install_vendor_path()
    from brain_adapter.ip_adapter.attention_processor import (
        AttnProcessor2_0,
        IPAttnProcessor2_0,
    )
    from brain_adapter.ip_adapter.ip_adapter import ImageProjModel
    from brain_adapter.model import GuidanceGenerator, NeuroAdapter

    image_projection = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=condition_dim,
    )
    processors: dict[str, torch.nn.Module] = {}
    unet_state = unet.state_dict()
    for name in unet.attn_processors:
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"unrecognized UNet attention processor: {name}")

        if cross_attention_dim is None:
            processors[name] = AttnProcessor2_0()
        else:
            layer_name = name.removesuffix(".processor")
            processor = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_parcels,
            )
            processor.load_state_dict(
                {
                    "to_k_ip.weight": unet_state[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_state[layer_name + ".to_v.weight"],
                },
                strict=True,
            )
            processors[name] = processor
    unet.set_attn_processor(processors)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    guidance = GuidanceGenerator(
        num_parcels=num_parcels,
        max_voxels=max_voxels,
        num_decoder_queries=50,
        output_dim=condition_dim,
        sub_approach=sub_approach,
    )
    neuro_adapter = NeuroAdapter(unet, image_projection, adapter_modules, ckpt_path=None)
    return AdapterBundle(neuro_adapter=neuro_adapter, guidance_generator=guidance)


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def unwrap_training_module(module: torch.nn.Module) -> NeuroAdapterTrainingModule:
    unwrapped = unwrap_module(module)
    if not isinstance(unwrapped, NeuroAdapterTrainingModule):
        raise TypeError(f"unexpected training module type: {type(unwrapped)!r}")
    return unwrapped


def trainable_state_dict(bundle: AdapterBundle) -> dict[str, dict[str, torch.Tensor]]:
    adapter = unwrap_module(bundle.neuro_adapter)
    guidance = unwrap_module(bundle.guidance_generator)
    return {
        "image_proj": {
            name: value.detach().cpu() for name, value in adapter.image_proj_model.state_dict().items()
        },
        "ip_adapter": {
            name: value.detach().cpu() for name, value in adapter.adapter_modules.state_dict().items()
        },
        "guidance_generator": {
            name: value.detach().cpu() for name, value in guidance.state_dict().items()
        },
    }


def load_trainable_state_dict(
    bundle: AdapterBundle, payload: dict[str, Any]
) -> None:
    if set(payload) != {"image_proj", "ip_adapter", "guidance_generator"}:
        raise ValueError("trainable state has unexpected top-level keys")
    adapter = unwrap_module(bundle.neuro_adapter)
    guidance = unwrap_module(bundle.guidance_generator)
    adapter.image_proj_model.load_state_dict(payload["image_proj"], strict=True)
    adapter.adapter_modules.load_state_dict(payload["ip_adapter"], strict=True)
    guidance.load_state_dict(payload["guidance_generator"], strict=True)


def audit_trainable_parameters(bundle: AdapterBundle) -> dict[str, int]:
    adapter = unwrap_module(bundle.neuro_adapter)
    guidance = unwrap_module(bundle.guidance_generator)
    allowed_ids = {
        id(parameter)
        for parameter in itertools.chain(
            adapter.image_proj_model.parameters(),
            adapter.adapter_modules.parameters(),
            guidance.parameters(),
        )
    }
    observed_trainable = {
        id(parameter)
        for parameter in itertools.chain(adapter.parameters(), guidance.parameters())
        if parameter.requires_grad
    }
    if observed_trainable != allowed_ids:
        raise ValueError(
            "trainable parameter boundary mismatch: "
            f"missing={len(allowed_ids - observed_trainable)}, "
            f"unexpected={len(observed_trainable - allowed_ids)}"
        )
    return {
        "trainable_tensors": len(allowed_ids),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in itertools.chain(
                adapter.image_proj_model.parameters(),
                adapter.adapter_modules.parameters(),
                guidance.parameters(),
            )
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in adapter.parameters()
            if not parameter.requires_grad
        ),
    }


def tensor_state_sha256(payload: dict[str, dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for group_name in sorted(payload):
        for tensor_name in sorted(payload[group_name]):
            tensor = payload[group_name][tensor_name].detach().cpu().contiguous()
            header = (
                f"{group_name}\0{tensor_name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0"
            ).encode("utf-8")
            digest.update(header)
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
