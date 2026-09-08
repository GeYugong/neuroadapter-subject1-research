from pathlib import Path

from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import FIXED_GATE_REQUIREMENTS, load_gate_requirements


def test_4090_protocol_is_consistent_with_both_templates() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = load_gate_requirements(root / "configs/protocol/gate_requirements.yaml")
    assert requirements.raw == FIXED_GATE_REQUIREMENTS
    assert requirements.raw["required_gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert requirements.raw["required_compute_capability"] == [8, 9]
    assert requirements.raw["required_cuda_arch"] == "sm_86"
    assert requirements.raw["max_reserved_memory_bytes"] == 22 * 1024**3
    for kind in ("selection", "final"):
        config = load_training_config(
            root / f"configs/training/subject01_{kind}.template.yaml", require_frozen=False
        )
        assert config.training["micro_batch_size"] == 4
        assert config.training["gradient_accumulation_steps"] == 2
        assert config.training["global_batch_size"] == 16
