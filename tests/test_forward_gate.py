from __future__ import annotations

from pathlib import Path

from scripts.gate_forward_alignment import load_upstream_setup_ip_adapter


def test_upstream_setup_function_can_be_isolated_from_broken_module_import() -> None:
    vendor = Path(__file__).resolve().parents[1] / "vendor" / "NeuroAdapter"
    function, source_sha256, function_sha256 = load_upstream_setup_ip_adapter(vendor)
    assert callable(function)
    assert len(source_sha256) == len(function_sha256) == 64
