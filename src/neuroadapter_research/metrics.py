"""Pure metric operations used by the frozen eight-metric evaluator."""

from __future__ import annotations

import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


def pixel_metrics(
    originals: np.ndarray, reconstructions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    original = torch.from_numpy(np.asarray(originals)).permute(0, 3, 1, 2).float() / 255
    reconstructed = (
        torch.from_numpy(np.asarray(reconstructions)).permute(0, 3, 1, 2).float() / 255
    )
    resize = v2.Resize(425, interpolation=InterpolationMode.BILINEAR, antialias=True)
    original = resize(original)
    reconstructed = resize(reconstructed)
    left = original.flatten(1).numpy().astype(np.float64, copy=False)
    right = reconstructed.flatten(1).numpy().astype(np.float64, copy=False)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    pixcorr = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros(left.shape[0], dtype=np.float64),
        where=denominator > 0,
    )

    original_gray = rgb2gray(original.permute(0, 2, 3, 1).numpy())
    reconstructed_gray = rgb2gray(reconstructed.permute(0, 2, 3, 1).numpy())
    ssim = np.asarray(
        [
            structural_similarity(
                reconstructed_gray[index],
                original_gray[index],
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
            for index in range(original_gray.shape[0])
        ],
        dtype=np.float64,
    )
    return pixcorr, ssim


def paired_correlation_distance(
    original_features: np.ndarray, reconstructed_features: np.ndarray
) -> np.ndarray:
    left = np.asarray(original_features, dtype=np.float64)
    right = np.asarray(reconstructed_features, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("paired feature matrices must have the same [images, features] shape")
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    correlations = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros(left.shape[0], dtype=np.float64),
        where=denominator > 0,
    )
    return 1.0 - correlations
