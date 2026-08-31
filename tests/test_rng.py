from __future__ import annotations

import torch

from neuroadapter_research.rng import TrainingGenerators, namespace_seed


def test_namespaces_and_ranks_have_distinct_seeds() -> None:
    seeds = {
        namespace_seed(42, namespace, rank)
        for namespace in ("vae_latent", "diffusion_noise", "timestep", "token_dropout")
        for rank in (0, 1)
    }
    assert len(seeds) == 8


def test_training_generators_resume_exactly() -> None:
    first = TrainingGenerators.create(42, rank=0, device=torch.device("cpu"))
    _ = torch.randn((4,), generator=first["diffusion_noise"])
    state = first.state_dict()
    expected = torch.randn((8,), generator=first["diffusion_noise"])

    restored = TrainingGenerators.create(42, rank=0, device=torch.device("cpu"))
    restored.load_state_dict(state)
    actual = torch.randn((8,), generator=restored["diffusion_noise"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
