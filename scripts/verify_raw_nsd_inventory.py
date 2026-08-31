#!/usr/bin/env python3
"""Compare the downloaded Subject 1 NSD tree with the official S3 listing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroadapter_research.atomic import sha256_file, write_json_atomic


REMOTE_PREFIX = (
    "nsddata_betas/ppdata/subj01/fsaverage/"
    "betas_fithrf_GLMdenoise_RR/"
)
LOCAL_PREFIX = "betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/"


def parse_s3_inventory(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: malformed S3 listing")
        size = int(fields[2])
        key = fields[3]
        if not key.startswith(REMOTE_PREFIX):
            raise ValueError(f"unexpected S3 key: {key}")
        relative = key.removeprefix(REMOTE_PREFIX)
        if not relative or relative in result:
            raise ValueError(f"duplicate or empty S3 key: {key}")
        result[relative] = size
    return result


def parse_local_inventory(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            relative, raw_size = line.split("\t")
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: malformed local inventory") from error
        if relative in result:
            raise ValueError(f"duplicate local path: {relative}")
        result[relative] = int(raw_size)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-nsd", type=Path, required=True)
    parser.add_argument("--s3-inventory", type=Path, required=True)
    parser.add_argument("--local-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    remote = parse_s3_inventory(args.s3_inventory)
    local = parse_local_inventory(args.local_inventory)
    local_beta = {
        path.removeprefix(LOCAL_PREFIX): size
        for path, size in local.items()
        if path.startswith(LOCAL_PREFIX)
    }
    if remote != local_beta:
        missing = sorted(set(remote) - set(local_beta))
        extra = sorted(set(local_beta) - set(remote))
        mismatched = sorted(
            path
            for path in set(remote) & set(local_beta)
            if remote[path] != local_beta[path]
        )
        raise ValueError(
            "NSD beta inventory mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}, size_mismatch={mismatched[:10]}"
        )

    expected_sessions = {
        f"{hemi}.betas_session{session:02}.mgh"
        for hemi in ("lh", "rh")
        for session in range(1, 41)
    }
    expected_noise_ceiling = {
        f"{hemi}.{suffix}.mgh"
        for hemi in ("lh", "rh")
        for suffix in ("ncsnr", "ncsnr_split1", "ncsnr_split2")
    }
    if not expected_sessions.issubset(remote):
        raise ValueError("official listing does not contain all 80 session beta files")
    if not expected_noise_ceiling.issubset(remote):
        raise ValueError("official listing does not contain all six ncsnr files")

    core_files = [
        "experiments/nsd/nsd_expdesign.mat",
        "experiments/nsd/nsd_stim_info_merged.csv",
        "experiments/nsd/nsd_stim_info_merged.pkl",
        "stimuli/nsd_stimuli.hdf5",
    ]
    for relative in core_files:
        path = args.raw_nsd / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing NSD core file: {path}")
        if local.get(relative) != path.stat().st_size:
            raise ValueError(f"local inventory is stale for {relative}")

    payload = {
        "schema_version": 1,
        "subject": 1,
        "source": "s3://natural-scenes-dataset",
        "beta_object_count": len(remote),
        "session_beta_count": len(expected_sessions),
        "ncsnr_count": len(expected_noise_ceiling),
        "beta_total_bytes": sum(remote.values()),
        "all_local_file_count": len(local),
        "all_local_total_bytes": sum(local.values()),
        "core_file_sizes": {relative: local[relative] for relative in core_files},
        "s3_inventory_sha256": sha256_file(args.s3_inventory),
        "local_inventory_sha256": sha256_file(args.local_inventory),
        "status": "sizes_match_official_inventory",
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
