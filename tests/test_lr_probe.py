import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("lr_runner", Path(__file__).parents[1]/"scripts/run_lr_probe.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(out):
    for arm, lr in (("H", 1e-4), ("L", 1e-5)):
        p = out/"preflight"/arm
        p.mkdir(parents=True)
        values = {"effective_config.json": {"arm": arm, "output": str(p),
                  "training": {"learning_rate": lr, "base_seed": 20260915}},
                  "initialization.json": {"optimizer_state_entries": 0, "model_tensor_sha256": "same"}}
        for rank in range(2):
            values[f"first_update_rank{rank}.json"] = {"losses": [0.1, 0.2],
                "gradient_sha256": "same", "updated_groups": {"image_proj": True, "ip_adapter": True, "guidance_generator": True}}
            (p/f"random_trace_rank{rank}.jsonl").write_text('{"local_update":1}\n')
        for name, value in values.items():
            (p/name).write_text(json.dumps(value))


def test_paired_audit_accepts_only_lr_difference(tmp_path):
    fixture(tmp_path)
    assert runner.paired_audit(tmp_path, "preflight")["passed"]


@pytest.mark.parametrize("name,field,value", [
    ("effective_config.json", "extra_science", True),
    ("first_update_rank0.json", "gradient_sha256", "different"),
    ("initialization.json", "optimizer_state_entries", 3),
])
def test_paired_audit_rejects_mismatch(tmp_path, name, field, value):
    fixture(tmp_path)
    path = tmp_path/"preflight/L"/name
    data = json.loads(path.read_text()); data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        runner.paired_audit(tmp_path, "preflight")


def test_random_trace_mismatch_stops_pipeline(tmp_path):
    fixture(tmp_path)
    (tmp_path/"preflight/L/random_trace_rank1.jsonl").write_text("different")
    with pytest.raises(AssertionError):
        runner.paired_audit(tmp_path, "preflight")
