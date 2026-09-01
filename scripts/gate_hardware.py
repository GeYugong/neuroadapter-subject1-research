#!/usr/bin/env python3
"""Run the frozen two-GPU RTX 5090/NCCL/BF16 hardware gate under torchrun."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_training_config(args.config, require_frozen=True)
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2 or config.training["world_size"] != 2:
        raise ValueError("hardware gate requires exactly two torchrun processes")
    if rank < 0 or local_rank < 0 or not torch.cuda.is_available():
        raise RuntimeError("hardware gate must run under CUDA torchrun")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend = configure_torch_backend(config.training)
    dist.init_process_group("nccl", init_method="env://")

    properties = torch.cuda.get_device_properties(device)
    capability = list(torch.cuda.get_device_capability(device))
    collective = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(collective)
    left = torch.arange(256 * 256, device=device, dtype=torch.bfloat16).reshape(256, 256)
    product = left @ left.T
    torch.cuda.synchronize(device)
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": properties.name,
        "compute_capability": capability,
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "bf16_matmul_finite": bool(torch.isfinite(product).all()),
        "nccl_all_reduce_value": float(collective.item()),
    }
    gathered: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)
    records = [record for record in gathered if record is not None]
    errors = []
    if len(records) != 2:
        errors.append("hardware gate did not collect two rank records")
    for record in records:
        if "5090" not in str(record["device_name"]):
            errors.append(f"unexpected GPU: {record['device_name']}")
        if record["compute_capability"] != [12, 0]:
            errors.append(f"unexpected compute capability: {record['compute_capability']}")
        if not record["bf16_supported"] or not record["bf16_matmul_finite"]:
            errors.append("BF16 hardware gate failed")
        if record["nccl_all_reduce_value"] != 3.0:
            errors.append("NCCL all-reduce gate failed")
    if errors:
        raise ValueError("; ".join(errors))
    output_error = [
        f"hardware gate output already exists: {args.output}"
        if rank == 0 and args.output.exists()
        else None
    ]
    dist.broadcast_object_list(output_error, src=0, device=device)
    if output_error[0] is not None:
        raise FileExistsError(output_error[0])
    if rank == 0:
        payload = {
            "schema_version": 1,
            "gate": "hardware_gate",
            "status": "passed",
            "config_sha256": config.sha256,
            "environment_lock_sha256": sha256_file(config.paths["environment_lock"]),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "backend": backend,
            "ranks": records,
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
