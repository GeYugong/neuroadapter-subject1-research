import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from generator_semantic_core import clip_preprocess_tensor, residual_features, semantic_loss, spec


class TinyClip(torch.nn.Module):
    def encode_image(self, value):
        return value.mean((2, 3)).repeat(1, 256)


def test_frozen_protocol_bounds():
    value = spec()
    assert value["pilot_updates"] == 5000
    assert value["maximum_semantic_updates"] == 20000
    assert value["snapshot_updates"] == [5000, 10000, 20000]
    assert value["source_update"] == 239063


def test_tensor_clip_path_retains_gradient():
    image = torch.zeros((1, 3, 32, 32), requires_grad=True)
    mean = torch.zeros(768)
    target = residual_features(torch.ones((1, 768)), mean)
    loss, stats = semantic_loss(image, target, TinyClip(), mean)
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert stats["clamp_saturation_fraction"] == 0.0


def test_clip_preprocess_shape_and_finiteness():
    result = clip_preprocess_tensor(torch.rand(2, 3, 64, 64))
    assert result.shape == (2, 3, 224, 224)
    assert torch.isfinite(result).all()
