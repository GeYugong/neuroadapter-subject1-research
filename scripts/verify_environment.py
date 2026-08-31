#!/usr/bin/env python3
"""Run CPU-only import and API compatibility checks for the candidate stack."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import accelerate
import diffusers
import h5py
import nibabel
import numpy
import scipy
import skimage
import torch
import torchvision
import transformers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    neuroadapter = project_root / "vendor" / "NeuroAdapter"
    sys.path.insert(0, str(neuroadapter))

    from brain_adapter.model import GuidanceGenerator, NeuroAdapter  # noqa: F401
    from brain_adapter.dataset import nsd_topk_parcel_dataset  # noqa: F401

    scheduler_signature = inspect.signature(DDPMScheduler.step)
    payload = {
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "nibabel": nibabel.__version__,
            "skimage": skimage.__version__,
            "accelerate": accelerate.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
        "imports": {
            "AutoencoderKL": AutoencoderKL.__name__,
            "UNet2DConditionModel": UNet2DConditionModel.__name__,
            "DDPMScheduler": DDPMScheduler.__name__,
            "NeuroAdapter": NeuroAdapter.__name__,
            "GuidanceGenerator": GuidanceGenerator.__name__,
        },
        "ddpm_step_parameters": list(scheduler_signature.parameters),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

