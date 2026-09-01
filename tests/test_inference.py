from neuroadapter_research.inference import sample_seed


def test_sample_seed_is_stable_and_namespaced() -> None:
    value = sample_seed("protocol", "validation", 12, 3)
    assert value == sample_seed("protocol", "validation", 12, 3)
    assert value != sample_seed("protocol", "validation", 12, 4)
    assert value != sample_seed("protocol", "test", 12, 3)
