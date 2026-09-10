import importlib.util
from pathlib import Path

import pytest


def load_controller():
    path = Path(__file__).resolve().parents[1] / 'scripts/select_existing_weights.py'
    spec = importlib.util.spec_from_file_location('existing_selection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['train_subject01.py', 'derive_final_config.py', 'export_final_model.py', '../train_subject01.py'])
def test_controller_cannot_launch_training_or_final_export(name):
    controller = load_controller()
    with pytest.raises(ValueError, match='forbidden'):
        controller.command(Path('/frozen'), name, [])


def test_controller_only_launches_four_fixed_evaluation_tools():
    controller = load_controller()
    assert controller.ALLOWED_SCRIPTS == {
        'validation_loss.py', 'decode_validation.py',
        'evaluate_validation.py', 'select_checkpoint.py',
    }
    result = controller.command(Path('/frozen'), 'decode_validation.py', ['--stage', 'screening'])
    assert result[1:] == ['/frozen/scripts/decode_validation.py', '--stage', 'screening']
