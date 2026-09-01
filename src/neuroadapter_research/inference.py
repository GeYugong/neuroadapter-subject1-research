"""Deterministic NeuroAdapter inference helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .checkpoint import verify_inference_snapshot
from .modeling import AdapterBundle, load_trainable_state_dict


def sample_seed(
    protocol: str, split: str, image_id: int, candidate_index: int
) -> int:
    if image_id < 0 or candidate_index < 0:
        raise ValueError("image_id and candidate_index must be non-negative")
    material = f"{protocol}\0{split}\0{image_id}\0{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") & ((1 << 63) - 1)


def load_inference_state(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        verify_inference_snapshot(path)
        model_path = path / "model.pt"
    else:
        model_path = path
    if not model_path.is_file():
        raise FileNotFoundError(f"inference state is missing: {model_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("inference state must be a mapping")
    return state


def install_inference_state(bundle: AdapterBundle, path: Path) -> None:
    load_trainable_state_dict(bundle, load_inference_state(path))
    bundle.neuro_adapter.eval()
    bundle.guidance_generator.eval()


def _randn_per_generator(
    sample_shape: tuple[int, ...],
    generators: list[torch.Generator],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randn(sample_shape, generator=generator, device=device, dtype=dtype)
            for generator in generators
        ]
    )


@torch.no_grad()
def generate_candidates(
    *,
    bundle: AdapterBundle,
    backbone: Any,
    brain: torch.Tensor,
    image_id: int,
    candidate_count: int,
    protocol: str,
    split: str,
    device: torch.device,
    dtype: torch.dtype,
    denoising_steps: int = 50,
    guidance_scale: float = 4.0,
) -> torch.Tensor:
    if candidate_count <= 0 or denoising_steps <= 0:
        raise ValueError("candidate_count and denoising_steps must be positive")
    generators = [
        torch.Generator(device=device).manual_seed(
            sample_seed(protocol, split, image_id, candidate_index)
        )
        for candidate_index in range(candidate_count)
    ]

    brain = brain.to(device=device, dtype=torch.float32).unsqueeze(0)
    zero_image = torch.zeros((1, 3, 512, 512), device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        latent_distribution = backbone.vae.encode(zero_image).latent_dist
    latent_mean = latent_distribution.mean
    latent_std = latent_distribution.std
    epsilon = _randn_per_generator(
        tuple(latent_mean.shape[1:]), generators, device=device, dtype=latent_mean.dtype
    )
    latents = (latent_mean + latent_std * epsilon) * backbone.vae.config.scaling_factor

    backbone.noise_scheduler.set_timesteps(denoising_steps, device=device)
    timesteps = backbone.noise_scheduler.timesteps
    initial_noise = _randn_per_generator(
        tuple(latents.shape[1:]), generators, device=device, dtype=latents.dtype
    )
    latents = backbone.noise_scheduler.add_noise(latents, initial_noise, timesteps[:1])

    empty_ids = backbone.tokenizer(
        "",
        max_length=backbone.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.autocast("cuda", dtype=dtype):
        empty_text = backbone.text_encoder(empty_ids)[0]
        condition_tokens, _ = bundle.guidance_generator(brain)
        condition_tokens = condition_tokens.expand(candidate_count, -1, -1)
        projected_cond = bundle.neuro_adapter.image_proj_model(condition_tokens)
        projected_uncond = bundle.neuro_adapter.image_proj_model(
            torch.zeros_like(condition_tokens)
        )
        text = empty_text.expand(candidate_count, -1, -1)
        hidden_cond = torch.cat([text, projected_cond], dim=1)
        hidden_uncond = torch.cat([text, projected_uncond], dim=1)

        for timestep in timesteps:
            combined_latents = torch.cat([latents, latents], dim=0)
            combined_hidden = torch.cat([hidden_uncond, hidden_cond], dim=0)
            prediction = bundle.neuro_adapter.unet(
                combined_latents, timestep, combined_hidden
            ).sample
            prediction_uncond, prediction_cond = prediction.chunk(2)
            guided = prediction_uncond + guidance_scale * (
                prediction_cond - prediction_uncond
            )
            latents = backbone.noise_scheduler.step(
                guided, timestep, latents, generator=generators
            ).prev_sample

        decoded = backbone.vae.decode(
            latents / backbone.vae.config.scaling_factor
        ).sample
    return (decoded.float() / 2 + 0.5).clamp(0, 1).cpu()
