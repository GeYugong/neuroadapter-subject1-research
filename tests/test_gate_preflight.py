import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'gate_preflight', ROOT / 'scripts/gate_preflight_inference.py'
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_pinned_metric_block_compiles_without_changing_source() -> None:
    path = ROOT / 'scripts/evaluate_validation.py'
    before = path.read_bytes()
    program = GATE.metric_program(path)
    assert program.co_filename == str(path)
    assert path.read_bytes() == before


def test_metric_program_executes_only_the_named_block(tmp_path: Path) -> None:
    path = tmp_path / 'metric.py'
    path.write_text(
        'def main():\n'
        '    raise RuntimeError("outside start")\n'
        '    by_metric: dict = {"value": [3, 5]}\n'
        '    seed_mean = {k: sum(v) / len(v) for k, v in by_metric.items()}\n'
        '    raise RuntimeError("outside end")\n'
    )
    namespace = {}
    exec(GATE.metric_program(path), namespace)
    assert namespace['seed_mean'] == {'value': 4.0}


def test_metric_program_rejects_ambiguous_anchors(tmp_path: Path) -> None:
    path = tmp_path / 'metric.py'
    path.write_text(
        'def main():\n'
        '    by_metric: dict = {}\n'
        '    by_metric: dict = {}\n'
        '    seed_mean = {}\n'
    )
    with pytest.raises(ValueError, match='not uniquely identifiable'):
        GATE.metric_program(path)


def test_engineering_pairs_are_fixed_and_not_formal_unique_image_pool() -> None:
    records = [{'image_id': i, 'files': list(range(8))} for i in range(8)]
    pairs = GATE.select_pairs({'records': records})
    assert [(r['image_id'], c) for r, c in pairs] == [
        (i, c) for c in range(3) for i in range(8)
    ][:20]
    assert len(pairs) == 20
    assert len({r['image_id'] for r, _ in pairs}) == 8
    records[1]['image_id'] = 0
    with pytest.raises(ValueError, match='unique'):
        GATE.select_pairs({'records': records})
