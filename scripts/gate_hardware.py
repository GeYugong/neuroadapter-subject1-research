#!/usr/bin/env python3
"""Run the frozen two-GPU RTX 5090/NCCL/BF16 hardware gate under torchrun."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import load_gate_requirements, method_fingerprint


def nvidia_smi_inventory() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    records = []
    for line in output.splitlines():
        index, uuid, name, driver = [value.strip() for value in line.split(",", 3)]
        records.append(
            {"index": index, "uuid": uuid, "name": name, "driver_version": driver}
        )
    return records


def xid_events_since(start_epoch: int) -> dict[str, object]:
    commands = (
        ["journalctl", "-k", "--since", f"@{start_epoch}", "--no-pager"],
        ["dmesg", "--color=never"],
    )
    failures = []
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        combined = result.stdout + "\n" + result.stderr
        if result.returncode != 0 or "permission" in combined.lower():
            failures.append({"command": command[0], "returncode": result.returncode})
            continue
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if "xid" in line.lower() and ("nvrm" in line.lower() or "nvidia" in line.lower())
        ]
        return {"available": True, "source": command[0], "events": lines}
    return {"available": False, "failures": failures, "events": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_training_config(args.config, require_frozen=True)
    requirements = load_gate_requirements(config.paths["gate_requirements"])
    required = requirements.raw
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != required["required_world_size"] or config.training[
        "world_size"
    ] != required["required_world_size"]:
        raise ValueError("hardware gate requires exactly two torchrun processes")
    if rank < 0 or local_rank < 0 or not torch.cuda.is_available():
        raise RuntimeError("hardware gate must run under CUDA torchrun")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend = configure_torch_backend(config.training)
    dist.init_process_group("nccl", init_method="env://")

    started_wall = int(time.time())
    started = time.perf_counter()
    properties = torch.cuda.get_device_properties(device)
    capability = list(torch.cuda.get_device_capability(device))
    arch_list = torch.cuda.get_arch_list()
    left = torch.arange(2048 * 2048, device=device, dtype=torch.bfloat16).reshape(
        2048, 2048
    )
    product = left @ left.T
    convolution = nn.Conv2d(32, 32, 3, padding=1, bias=True).to(
        device=device, dtype=torch.bfloat16
    )
    convolution_input = torch.randn(
        (8, 32, 128, 128), device=device, dtype=torch.bfloat16, requires_grad=True
    )
    convolution_loss = convolution(convolution_input).float().square().mean()
    convolution_loss.backward()
    conv_finite = bool(
        torch.isfinite(convolution_loss).item()
        and convolution_input.grad is not None
        and torch.isfinite(convolution_input.grad).all().item()
        and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in convolution.parameters()
        )
    )

    stress_iterations = 0
    nccl_verified = True
    stress_finite = True
    minimum_seconds = int(required["stress_minimum_seconds"])
    while time.perf_counter() - started < minimum_seconds:
        left.grad = None
        stress_input = left.detach().requires_grad_(True)
        stress_loss = (stress_input @ stress_input.T).float().square().mean()
        stress_loss.backward()
        collective = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(collective)
        nccl_verified = nccl_verified and float(collective.item()) == 3.0
        stress_finite = stress_finite and bool(
            torch.isfinite(stress_loss).item()
            and stress_input.grad is not None
            and torch.isfinite(stress_input.grad).all().item()
        )
        stress_iterations += 1
    torch.cuda.synchronize(device)
    stress_duration = time.perf_counter() - started
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": properties.name,
        "compute_capability": capability,
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "bf16_matmul_finite": bool(torch.isfinite(product).all()),
        "bf16_conv_backward_finite": conv_finite,
        "stress_finite": stress_finite,
        "stress_iterations": stress_iterations,
        "stress_duration_seconds": stress_duration,
        "nccl_all_reduce_verified": nccl_verified,
        "maximum_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    gathered: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)
    records = [record for record in gathered if record is not None]
    errors = []
    if len(records) != 2:
        errors.append("hardware gate did not collect two rank records")
    for record in records:
        if record["device_name"] != required["required_gpu_name"]:
            errors.append(f"unexpected GPU: {record['device_name']}")
        if record["compute_capability"] != required["required_compute_capability"]:
            errors.append(f"unexpected compute capability: {record['compute_capability']}")
        if (
            record["bf16_supported"] is not required["required_bf16"]
            or not record["bf16_matmul_finite"]
            or not record["bf16_conv_backward_finite"]
            or not record["stress_finite"]
        ):
            errors.append("BF16 hardware gate failed")
        if not record["nccl_all_reduce_verified"]:
            errors.append("NCCL all-reduce gate failed")
        if float(record["stress_duration_seconds"]) < minimum_seconds:
            errors.append("hardware stress run was shorter than required")
    if required["required_cuda_arch"] not in arch_list:
        errors.append(f"PyTorch has no native {required['required_cuda_arch']} build")

    system_evidence: list[dict | None] = [None]
    if rank == 0:
        try:
            inventory = nvidia_smi_inventory()
        except Exception as exc:
            inventory = []
            errors.append(f"nvidia-smi inventory failed: {type(exc).__name__}: {exc}")
        if len(inventory) != required["required_world_size"]:
            errors.append("nvidia-smi inventory does not contain exactly two GPUs")
        inventory_uuids = [record.get("uuid", "") for record in inventory]
        if (
            any(record.get("name") != required["required_gpu_name"] for record in inventory)
            or any(not record.get("driver_version") for record in inventory)
            or any(not value for value in inventory_uuids)
            or len(set(inventory_uuids)) != len(inventory_uuids)
        ):
            errors.append("nvidia-smi GPU identity evidence is incomplete or inconsistent")
        try:
            xid = xid_events_since(started_wall)
        except Exception as exc:
            xid = {"available": False, "events": [], "error": str(exc)}
            errors.append(f"Xid query failed: {type(exc).__name__}: {exc}")
        system_evidence[0] = {"nvidia_smi": inventory, "xid": xid}
        if required["require_xid_check"] and not xid["available"]:
            errors.append("kernel Xid log is not readable")
        if xid["events"]:
            errors.append("NVIDIA Xid event occurred during the stress window")
    error_box: list[list[str] | None] = [errors if rank == 0 else None]
    dist.broadcast_object_list(system_evidence, src=0, device=device)
    dist.broadcast_object_list(error_box, src=0, device=device)
    if error_box[0]:
        raise ValueError("; ".join(error_box[0]))
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
            "method_fingerprint": method_fingerprint(config),
            "gate_requirements_sha256": requirements.sha256,
            "environment_lock_sha256": sha256_file(config.paths["environment_lock"]),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "backend": backend,
            "required_cuda_arch": required["required_cuda_arch"],
            "torch_cuda_arch_list": arch_list,
            "native_arch_available": required["required_cuda_arch"] in arch_list,
            "stress_duration_seconds": min(
                float(record["stress_duration_seconds"]) for record in records
            ),
            "xid_check_passed": bool(
                system_evidence[0]["xid"]["available"]
                and not system_evidence[0]["xid"]["events"]
            ),
            "system_evidence": system_evidence[0],
            "ranks": records,
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
