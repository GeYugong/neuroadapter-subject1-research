#!/usr/bin/env python3
"""Sequential paired training and bounded evaluation, with fail-closed audits."""
import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def paired_audit(out, folder):
    parents = [out/folder/arm for arm in ("H", "L")]
    configs = [json.loads((p/"effective_config.json").read_text()) for p in parents]
    science = copy.deepcopy(configs)
    assert [c["training"]["learning_rate"] for c in science] == [1e-4, 1e-5]
    for c in science:
        for key in ("arm", "output"):
            c.pop(key)
        c["training"].pop("learning_rate")
    assert science[0] == science[1], "unpaired scientific configuration"
    initial = [json.loads((p/"initialization.json").read_text()) for p in parents]
    assert initial[0] == initial[1] and initial[0]["optimizer_state_entries"] == 0
    ranks = []
    for rank in range(2):
        records = [json.loads((p/f"first_update_rank{rank}.json").read_text()) for p in parents]
        assert records[0] == records[1], "first loss/gradient/random input mismatch"
        assert all(records[0]["updated_groups"].values())
        hashes = [sha256_file(p/f"random_trace_rank{rank}.jsonl") for p in parents]
        assert hashes[0] == hashes[1], "training random sequence mismatch"
        ranks.append({"rank": rank, "first_update": records[0], "random_trace_sha256": hashes[0]})
    result = {"passed": True, "folder": folder, "configuration_only_lr_differs": True,
              "initialization": initial[0], "ranks": ranks}
    write_json_atomic(out/f"{folder}-paired-audit.json", result)
    return result


def run(root):
    out = root/"runs/experiments/paired-lr-probe-v1"
    repo = root/"repo"
    assert not (out/"pipeline.json").exists(), "pipeline already started; inspect existing state"
    paired_audit(out, "preflight")
    protocol = json.loads((out/"evaluation/protocol.json").read_text())
    assert len(protocol["ids"]) == 500
    assert all(len(v["replay_checks"]) == 4 for v in protocol["references"].values())
    files = [repo/"scripts"/name for name in ("train_lr_probe.py", "evaluate_lr_probe.py", "run_lr_probe.py")]
    files.append(repo/"configs/experiments/paired_lr_probe_v1.yaml")
    hashes = {str(p): sha256_file(p) for p in files}
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4",
               PYTHONPATH=str(root/"runtime/subject01-4090-1a1fcfa/src"))
    python = str(root/"envs/neuroadapter/bin/python")
    state = {"status": "running", "source_commit": commit, "files": hashes, "commands": [],
             "automatic_weight_replacement": False, "automatic_extension": False}

    def note(message):
        stamp = datetime.now(timezone.utc).isoformat()
        with (repo/"EXPERIMENT_LOG.md").open("a", encoding="utf-8") as f:
            f.write(f"\n\n### paired-lr-probe-v1 自动阶段记录 {stamp}\n\n{message}\n")

    def execute(stage, args, gpus="0,1"):
        assert all(sha256_file(Path(p)) == h for p, h in hashes.items()), "frozen experiment code changed"
        state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        command = [python, *args]
        state["commands"].append({"stage": stage, "argv": command, "gpus": gpus})
        write_json_atomic(out/"pipeline.json", state)
        note(f"开始 `{stage}`，GPU `{gpus}`；运行提交 `{commit}`。完整命令及源码 SHA 记录在 `runs/experiments/paired-lr-probe-v1/pipeline.json`，终端日志为 `{stage}.log`。")
        with (out/f"{stage}.log").open("x") as log:
            subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=gpus), stdout=log, stderr=subprocess.STDOUT, check=True)
        note(f"`{stage}` 退出码 0。保存原始输出；尚未据此选优或替换权重。")

    try:
        for arm in ("H", "L"):
            execute(f"train-{arm}", ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                str(repo/"scripts/train_lr_probe.py"), "--root", str(root), "--arm", arm])
            status = json.loads((out/"arms"/arm/"status.json").read_text())
            assert status["status"] == "completed" and status["local_update"] == 5000
        paired_audit(out, "arms")
        for arm in ("H", "L"):
            execute(f"decode-{arm}", [str(repo/"scripts/evaluate_lr_probe.py"), "--root", str(root),
                "--phase", "decode", "--arm", arm], "0" if arm == "H" else "1")
        for label in ("B0", "H5000", "L5000", "R"):
            execute(f"score-{label}", [str(repo/"scripts/evaluate_lr_probe.py"), "--root", str(root),
                "--phase", "score", "--label", label], "0")
        execute("decision", [str(repo/"scripts/evaluate_lr_probe.py"), "--root", str(root), "--phase", "decision"], "")
        state.update(status="endpoint_statistics_complete", stage="awaiting_visual_and_trajectory_review")
        note("两组训练、完整500图终点评分与预定统计计算结束。小样本轨迹汇总、完整图册视觉审查和最终中文报告仍待完成，不宣称全部实验已验收；不启动任何后续训练。")
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        note(f"阶段 `{state.get('stage')}` 失败，已停止后续执行。错误 `{exc!r}`。保留所有权重和失败记录，不自动重训。")
        raise
    finally:
        write_json_atomic(out/"pipeline.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    if a.audit_only:
        paired_audit(a.root/"runs/experiments/paired-lr-probe-v1", "preflight")
    else:
        run(a.root)
