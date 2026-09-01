from __future__ import annotations

import numpy as np

from neuroadapter_research.metrics import paired_correlation_distance, pixel_metrics


def test_correlation_distance_is_zero_for_identical_features() -> None:
    features = np.asarray([[1.0, 2.0, 4.0], [4.0, 1.0, 2.0]])
    np.testing.assert_allclose(paired_correlation_distance(features, features), 0.0, atol=1e-12)


def test_identical_images_have_unit_pixel_metrics() -> None:
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(1, 32, 32, 3)
    pixcorr, ssim = pixel_metrics(image, image)
    np.testing.assert_allclose(pixcorr, 1.0, atol=1e-12)
    np.testing.assert_allclose(ssim, 1.0, atol=1e-12)
