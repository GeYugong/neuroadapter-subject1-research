#!/usr/bin/env python3
"""Copy completed endpoint evidence to Git without model or image payloads."""
import argparse
import json
import shutil
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic


def archive(root):
    source = root / "runs/experiments/paired-lr-probe-v1"
    target = root / "repo/manifests/paired-lr-probe-v1"
    pipeline = json.loads((source / "pipeline.json").read_text())
    assert pipeline["status"] == "endpoint_statistics_complete"
    assert json.loads((source / "arms-paired-audit.json").read_text())["passed"]
    files = list(source.glob("*.json"))
    files += list(source.glob("*.log"))
    files += list(source.glob("evaluation/*.json"))
    files += list(source.glob("evaluation/*.csv"))
    files += list(source.glob("evaluation/*/decode_manifest.json"))
    files += list(source.glob("diagnostic/*/*/images.jsonl"))
    for arm in ("H", "L"):
        folder = source / "arms" / arm
        status = json.loads((folder / "status.json").read_text())
        assert status["status"] == "completed" and status["local_update"] == 5000
        files += list(folder.glob("*.json"))
        files += [folder / "training.jsonl"]
        files += list(folder.glob("snapshots/*/metadata.json"))
        files += list(folder.glob("snapshots/*/MANIFEST.json"))
    index = []
    for path in sorted(set(files)):
        assert path.stat().st_size < 10_000_000, path
        relative = path.relative_to(source)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assert sha256_file(path) == sha256_file(dest)
        index.append({"path": relative.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json_atomic(target / "INDEX.json", {
        "sources": index, "training_commit": pipeline["source_commit"],
        "status": "endpoint_statistics_complete; visual and trajectory review pending",
        "excluded": "weights, optimizer states, images, brain arrays, noise tensors, credentials; random traces represented by paired audit hashes"})
    print(f"Archived {len(index)} files, {sum(x['bytes'] for x in index)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    archive(parser.parse_args().root)
