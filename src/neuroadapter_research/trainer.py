"""Two-process DDP trainer for the frozen Subject 1 method."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import random
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .atomic import sha256_file, write_json_atomic
from .backend import configure_torch_backend
from .checkpoint import (
    load_distributed_checkpoint,
    prune_full_checkpoints,
    save_distributed_checkpoint,
    save_inference_snapshot,
)
from .config import LoadedTrainingConfig, verify_config_inputs
from .data import Subject1TrainingDataset
from .integrity import (
    load_json_mapping,
    validate_gate_artifact,
    validate_subject1_audits,
    verify_file_against_manifest,
    verify_submodule_heads,
    verify_tree_against_manifest,
)
from .modeling import (
    NeuroAdapterTrainingModule,
    audit_trainable_parameters,
    build_adapter,
    load_frozen_backbone,
    load_trainable_state_dict,
    tensor_state_sha256,
    trainable_state_dict,
    unwrap_training_module,
)
from .rng import (
    TrainingGenerators,
    capture_process_rng_state,
    restore_process_rng_state,
)
from .sampler import DeterministicDistributedBatchPlan, PlannedBatchSampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class TerminationFlag:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handler(self, signal_number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = signal_number


def initialize_distributed(expected_world_size: int) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != expected_world_size or rank < 0 or local_rank < 0:
        raise RuntimeError(
            "trainer must be launched by torchrun with the configured world size"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < expected_world_size:
        raise RuntimeError("configured CUDA devices are not available")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def set_process_seed(base_seed: int, rank: int) -> None:
    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def all_ranks_agree_on_hash(value: str, context: DistributedContext) -> None:
    encoded = torch.tensor(
        list(bytes.fromhex(value)), dtype=torch.uint8, device=context.device
    )
    gathered = [torch.empty_like(encoded) for _ in range(context.world_size)]
    dist.all_gather(gathered, encoded)
    observed = {bytes(item.cpu().tolist()).hex() for item in gathered}
    if observed != {value}:
        raise RuntimeError(f"rank state mismatch: {sorted(observed)}")


def collect_input_hashes(
    config: LoadedTrainingConfig, context: DistributedContext
) -> dict[str, str]:
    names = (
        "training_cache",
        "training_cache_manifest",
        "training_cache_verification",
        "model_manifest",
        "raw_nsd_manifest",
        "split_ids",
        "canonical_initialization",
        "canonical_manifest",
        "data_fingerprint",
        "nsd_image_mapping",
        "schaefer_equivalence",
        "split_manifest",
        "source_manifest",
        "environment_lock",
    )
    payload: list[dict[str, str] | None] = [None]
    if context.is_main:
        stimulus = verify_file_against_manifest(
            config.paths["stimuli"],
            config.paths["raw_nsd_manifest"],
            "stimuli/nsd_stimuli.hdf5",
        )
        stable_diffusion = verify_tree_against_manifest(
            config.paths["stable_diffusion"],
            config.paths["model_manifest"],
            manifest_prefix="stable-diffusion-v1-5",
            ignored_extra_prefixes=(".cache/huggingface",),
        )
        vendor = verify_submodule_heads(
            Path(__file__).resolve().parents[2], config.paths["source_manifest"]
        )
        payload[0] = {
            "config": config.sha256,
            **{name: sha256_file(config.paths[name]) for name in names},
            "stimuli": stimulus["sha256"],
            "stable_diffusion_tree": stable_diffusion["tree_sha256"],
            "vendor_submodule_heads": vendor["submodule_heads_sha256"],
        }
    dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0] is None:
        raise RuntimeError("rank zero did not broadcast input hashes")
    return payload[0]


def validate_canonical_initialization(
    config: LoadedTrainingConfig, *, require_frozen: bool
) -> dict[str, Any]:
    manifest = load_json_mapping(config.paths["canonical_manifest"])
    if require_frozen and manifest.get("status") != "frozen":
        raise ValueError("formal canonical initialization is not bound to a frozen environment")
    if manifest.get("status") not in {"candidate", "frozen"}:
        raise ValueError("canonical initialization has an invalid environment status")
    observed = sha256_file(config.paths["canonical_initialization"])
    if manifest["initialization_sha256"] != observed:
        raise ValueError("canonical initialization hash mismatch")
    if int(manifest["num_parcels"]) != 200:
        raise ValueError("canonical initialization does not contain 200 parcels")
    if manifest["data_fingerprint_sha256"] != sha256_file(
        config.paths["data_fingerprint"]
    ):
        raise ValueError("canonical initialization uses a different data fingerprint")
    if manifest["model_manifest_sha256"] != sha256_file(
        config.paths["model_manifest"]
    ):
        raise ValueError("canonical initialization uses a different model manifest")
    if manifest["environment_lock_sha256"] != sha256_file(
        config.paths["environment_lock"]
    ):
        raise ValueError("canonical initialization uses a different environment lock")
    repository = Path(__file__).resolve().parents[2]
    if manifest["modeling_sha256"] != sha256_file(
        repository / "src/neuroadapter_research/modeling.py"
    ):
        raise ValueError("canonical initialization uses different model construction code")
    if manifest["source_manifest_sha256"] != sha256_file(
        config.paths["source_manifest"]
    ):
        raise ValueError("canonical initialization uses a different source manifest")
    repository_commit = manifest.get("repository_commit")
    if not isinstance(repository_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", repository_commit
    ):
        raise ValueError("canonical initialization has no generating repository commit")
    return manifest


def validate_formal_approval(config: LoadedTrainingConfig, approval_path: Path) -> None:
    approval = load_json_mapping(Path(approval_path))
    gate_fields = {
        "environment_lock_sha256": "environment_lock",
        "hardware_gate_sha256": "hardware_gate",
        "forward_alignment_sha256": "forward_alignment",
        "batch_gate_sha256": "batch_gate",
        "resume_equivalence_sha256": "resume_equivalence",
        "decode_determinism_sha256": "decode_determinism",
        "evaluator_repeatability_sha256": "evaluator_repeatability",
        "data_fingerprint_sha256": "data_fingerprint",
        "model_assets_manifest_sha256": "model_manifest",
        "canonical_initialization_sha256": "canonical_initialization",
        "training_cache_verification_sha256": "training_cache_verification",
        "nsd_image_mapping_sha256": "nsd_image_mapping",
        "schaefer_equivalence_sha256": "schaefer_equivalence",
    }
    expected = {
        "approved": True,
        "config_sha256": config.sha256,
        "protocol_commit": config.raw["protocol_commit"],
        **{
            field: sha256_file(config.paths[path_name])
            for field, path_name in gate_fields.items()
        },
    }
    if approval != expected:
        raise ValueError("formal approval file does not match the frozen config")
    for path_name in (
        "hardware_gate",
        "forward_alignment",
        "batch_gate",
        "resume_equivalence",
        "decode_determinism",
        "evaluator_repeatability",
    ):
        validate_gate_artifact(
            config.paths[path_name],
            expected_gate=path_name,
            config_sha256=config.sha256,
        )
    validate_subject1_audits(config.paths)
    repository = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config.raw["protocol_commit"]:
        raise ValueError(f"Git HEAD {head} differs from protocol commit")
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        text=True,
    )
    if status:
        raise ValueError("formal training requires a clean Git worktree")


def token_keep_mask(
    batch_size: int,
    num_parcels: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    ratios = torch.rand((batch_size, 1), device=device, generator=generator)
    values = torch.rand(
        (batch_size, num_parcels), device=device, generator=generator
    )
    return (values <= ratios).unsqueeze(-1)


def min_snr_weights(
    timesteps: torch.Tensor, alphas_cumprod: torch.Tensor, gamma: float
) -> torch.Tensor:
    alpha = alphas_cumprod[timesteps]
    epsilon = 1e-8
    snr = alpha / (1 - alpha + epsilon)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + epsilon)


def reference_epoch_checkpoints(
    sample_count: int,
    global_batch_size: int,
    interval: int,
    max_updates: int,
) -> set[int]:
    maximum_epochs = max_updates * global_batch_size / sample_count
    result = set()
    epoch = interval
    while epoch <= maximum_epochs + 1e-12:
        result.add(math.ceil(epoch * sample_count / global_batch_size))
        epoch += interval
    return result


def build_dataloader(
    dataset: Subject1TrainingDataset,
    plan: DeterministicDistributedBatchPlan,
    start_update: int,
    stop_update: int,
    context: DistributedContext,
    workers: int,
    worker_seed: int,
) -> DataLoader:
    batch_sampler = PlannedBatchSampler(
        plan=plan,
        start_update=start_update,
        stop_update=stop_update,
        rank=context.rank,
    )
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": workers,
        "pin_memory": True,
        "generator": torch.Generator().manual_seed(worker_seed),
    }
    if workers:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**arguments)


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def run_training(
    config: LoadedTrainingConfig,
    run_mode: str,
    max_updates_override: int | None,
    output_override: Path | None,
    resume: Path | None,
    approval_path: Path | None,
    trace_updates: int,
) -> None:
    if run_mode not in {"gate", "formal"}:
        raise ValueError("run_mode must be gate or formal")
    if run_mode == "formal":
        if max_updates_override is not None or output_override is not None:
            raise ValueError("formal mode forbids update and output overrides")
        if approval_path is None:
            raise ValueError("formal mode requires an approval file")
        validate_formal_approval(config, approval_path)

    verify_config_inputs(config, require_gate_artifacts=run_mode == "formal")
    training = config.training
    context = initialize_distributed(training["world_size"])
    termination = TerminationFlag()
    signal.signal(signal.SIGTERM, termination.handler)
    signal.signal(signal.SIGINT, termination.handler)
    set_process_seed(training["base_seed"], context.rank)
    backend_settings = configure_torch_backend(training)

    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else config.paths["output_dir"]
    )
    max_updates = (
        int(max_updates_override)
        if max_updates_override is not None
        else int(training["max_updates"])
    )
    if max_updates <= 0:
        raise ValueError("effective max updates must be positive")
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fresh run output directory is not empty: {output_dir}")
    if resume is not None:
        expected_output = Path(resume).resolve().parent.parent
        if expected_output != output_dir:
            raise ValueError(
                f"resume checkpoint belongs to {expected_output}, not {output_dir}"
            )

    dataset = Subject1TrainingDataset(
        cache_path=config.paths["training_cache"],
        stimuli_path=config.paths["stimuli"],
        split_ids_path=config.paths["split_ids"],
    )
    expected_samples = 8500 if config.raw["run_kind"] == "selection" else 9000
    if len(dataset) != expected_samples:
        raise ValueError(f"expected {expected_samples} training images, got {len(dataset)}")
    plan = DeterministicDistributedBatchPlan(
        sample_count=len(dataset),
        global_batch_size=training["global_batch_size"],
        world_size=context.world_size,
        micro_batch_size=training["micro_batch_size"],
        accumulation_steps=training["gradient_accumulation_steps"],
        seed=training["sampler_seed"],
    )
    input_hashes = collect_input_hashes(config, context)
    cache_manifest = json.loads(
        config.paths["training_cache_manifest"].read_text(encoding="utf-8")
    )
    if cache_manifest["cache_sha256"] != input_hashes["training_cache"]:
        raise ValueError("training cache hash differs from its frozen manifest")
    canonical_manifest = validate_canonical_initialization(
        config, require_frozen=run_mode == "formal"
    )
    if int(canonical_manifest["max_voxels"]) != dataset.max_voxels:
        raise ValueError("canonical initialization max_voxels differs from the dataset")

    backbone = load_frozen_backbone(config.paths["stable_diffusion"])
    bundle = build_adapter(backbone.unet, dataset.num_parcels, dataset.max_voxels)
    canonical_state = torch.load(
        config.paths["canonical_initialization"], map_location="cpu", weights_only=True
    )
    load_trainable_state_dict(bundle, canonical_state)
    parameter_audit = audit_trainable_parameters(bundle)
    module = NeuroAdapterTrainingModule(bundle)

    start_update = 0
    generators = TrainingGenerators.create(
        training["base_seed"], context.rank, context.device
    )
    resume_payload: dict[str, Any] | None = None
    if resume is not None:
        resume_payload = load_distributed_checkpoint(
            Path(resume), context.rank, context.world_size
        )
        trainer_state = resume_payload["trainer"]
        if trainer_state["input_hashes"] != input_hashes:
            raise ValueError("resume checkpoint input hashes differ from the current run")
        plan.validate_state(trainer_state["sampler"])
        start_update = int(trainer_state["next_update"])
        if start_update >= max_updates:
            raise ValueError("resume checkpoint is already at or beyond max_updates")
        load_trainable_state_dict(bundle, resume_payload["model"])

    initial_state_hash = tensor_state_sha256(trainable_state_dict(bundle))
    all_ranks_agree_on_hash(initial_state_hash, context)
    module.to(context.device)
    backbone.vae.to(context.device, dtype=torch.bfloat16)
    backbone.text_encoder.to(context.device, dtype=torch.bfloat16)
    backbone.vae.eval()
    backbone.text_encoder.eval()

    ddp_arguments: dict[str, Any] = {
        "device_ids": [context.local_rank],
        "output_device": context.local_rank,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    if "init_sync" in inspect.signature(DistributedDataParallel).parameters:
        ddp_arguments["init_sync"] = False
    distributed_module = DistributedDataParallel(module, **ddp_arguments)
    parameters = [value for value in distributed_module.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        betas=(training["adam_beta1"], training["adam_beta2"]),
        eps=training["adam_epsilon"],
        weight_decay=training["weight_decay"],
        fused=training["adamw_fused"],
        foreach=training["adamw_foreach"],
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        move_optimizer_state_to_device(optimizer, context.device)
        generators.load_state_dict(resume_payload["rank"]["training_generators"])
        restore_process_rng_state(resume_payload["rank"]["process_rng"], context.device)

    empty_ids = backbone.tokenizer(
        "",
        max_length=backbone.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(context.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        empty_text = backbone.text_encoder(empty_ids)[0]
    alphas_cumprod = backbone.noise_scheduler.alphas_cumprod.to(context.device)
    dataloader = build_dataloader(
        dataset,
        plan,
        start_update,
        max_updates,
        context,
        training["dataloader_workers"],
        training["base_seed"] + context.rank,
    )
    iterator = iter(dataloader)

    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        effective = {
            "run_mode": run_mode,
            "run_name": config.raw["run_name"],
            "run_kind": config.raw["run_kind"],
            "config_path": str(config.path),
            "config_sha256": config.sha256,
            "start_update": start_update,
            "max_updates": max_updates,
            "input_hashes": input_hashes,
            "canonical_manifest": canonical_manifest,
            "initial_state_sha256": initial_state_hash,
            "parameter_audit": parameter_audit,
            "world_size": context.world_size,
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(context.world_size)
            ],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "backend": {
                **backend_settings,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            },
            "batch_geometry": {
                name: training[name]
                for name in (
                    "world_size",
                    "global_batch_size",
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                )
            },
        }
        write_json_atomic(output_dir / "effective_run.json", effective)
    dist.barrier()

    interval_checkpoints = set(range(training["checkpoint_every_updates"], max_updates + 1, training["checkpoint_every_updates"]))
    epoch_checkpoints = reference_epoch_checkpoints(
        len(dataset),
        training["global_batch_size"],
        training["checkpoint_reference_epochs"],
        max_updates,
    )
    full_checkpoint_updates = interval_checkpoints | {max_updates}
    snapshot_updates = epoch_checkpoints | {max_updates}
    log_path = output_dir / "training.jsonl"
    trace_path = output_dir / "traces" / f"trace-rank-{context.rank:05d}.jsonl"
    started = time.perf_counter()
    last_saved_update: int | None = None
    exit_after_checkpoint = False

    for update in range(start_update, max_updates):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[torch.Tensor] = []
        trace: list[dict[str, Any]] = []
        for micro_step in range(training["gradient_accumulation_steps"]):
            batch = next(iterator)
            image_ids = batch["nsd_image_id"]
            images = batch["image"].to(
                context.device, dtype=torch.bfloat16, non_blocking=True
            )
            brain = batch["brain"].to(
                context.device, dtype=torch.float32, non_blocking=True
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent_distribution = backbone.vae.encode(images).latent_dist
                latents = latent_distribution.sample(generator=generators["vae_latent"])
                latents = latents * backbone.vae.config.scaling_factor
            noise = torch.randn(
                latents.shape,
                device=context.device,
                dtype=latents.dtype,
                generator=generators["diffusion_noise"],
            )
            timesteps = torch.randint(
                0,
                backbone.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=context.device,
                generator=generators["timestep"],
            ).long()
            noisy_latents = backbone.noise_scheduler.add_noise(
                latents, noise, timesteps
            )
            keep_mask = token_keep_mask(
                latents.shape[0],
                dataset.num_parcels,
                context.device,
                generators["token_dropout"],
            )
            text = empty_text.expand(latents.shape[0], -1, -1)
            sync_context = (
                distributed_module.no_sync()
                if micro_step + 1 < training["gradient_accumulation_steps"]
                else contextlib.nullcontext()
            )
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = distributed_module(
                    noisy_latents, timesteps, text, brain, keep_mask
                )
                per_sample = F.mse_loss(
                    prediction.float(), noise.float(), reduction="none"
                ).mean(dim=(1, 2, 3))
                weights = min_snr_weights(
                    timesteps, alphas_cumprod, training["min_snr_gamma"]
                )
                loss = (per_sample * weights).mean()
                (loss / training["gradient_accumulation_steps"]).backward()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {update}")
            micro_losses.append(loss.detach())
            if update < trace_updates:
                trace.append(
                    {
                        "micro_step": micro_step,
                        "image_ids": [int(value) for value in image_ids.tolist()],
                        "timesteps_sha256": tensor_sha256(timesteps),
                        "vae_latent_sha256": tensor_sha256(latents),
                        "noise_sha256": tensor_sha256(noise),
                        "dropout_sha256": tensor_sha256(keep_mask),
                    }
                )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, training["max_grad_norm"]
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        next_update = update + 1
        if update < trace_updates:
            append_json_line(
                trace_path,
                {
                    "optimizer_update": next_update,
                    "rank": context.rank,
                    "microbatches": trace,
                },
            )

        mean_loss = torch.stack(micro_losses).mean()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss = mean_loss / context.world_size
        should_log = (
            next_update % training["log_every_updates"] == 0
            or next_update == 1
            or next_update == max_updates
            or update < trace_updates
        )
        if context.is_main and should_log:
            append_json_line(
                log_path,
                {
                    "optimizer_update": next_update,
                    "images_seen": next_update * training["global_batch_size"],
                    "reference_epoch": next_update
                    * training["global_batch_size"]
                    / len(dataset),
                    "loss": float(mean_loss.cpu()),
                    "gradient_norm_before_clip": float(gradient_norm.cpu()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                        context.device
                    ),
                    "trace_file": trace_path.relative_to(output_dir).as_posix()
                    if update < trace_updates
                    else None,
                },
            )

        local_stop = torch.tensor(
            1 if termination.requested else 0,
            dtype=torch.int32,
            device=context.device,
        )
        dist.all_reduce(local_stop, op=dist.ReduceOp.MAX)
        exit_after_checkpoint = bool(local_stop.item())
        if next_update in snapshot_updates:
            training_module = unwrap_training_module(distributed_module)
            if context.is_main:
                save_inference_snapshot(
                    output_dir / "snapshots",
                    next_update,
                    trainable_state_dict(training_module.as_bundle()),
                    {
                        "schema_version": 1,
                        "optimizer_update": next_update,
                        "images_seen": next_update * training["global_batch_size"],
                        "reference_epoch": next_update
                        * training["global_batch_size"]
                        / len(dataset),
                        "input_hashes": input_hashes,
                    },
                )
            dist.barrier()
        if next_update in full_checkpoint_updates or exit_after_checkpoint:
            training_module = unwrap_training_module(distributed_module)
            state = (
                trainable_state_dict(training_module.as_bundle())
                if context.is_main
                else {}
            )
            trainer_state = {
                "schema_version": 1,
                "run_mode": run_mode,
                "next_update": next_update,
                "images_seen": next_update * training["global_batch_size"],
                "accumulation_step": 0,
                "sampler": plan.state_before_update(next_update).to_dict(),
                "input_hashes": input_hashes,
                "interrupted": exit_after_checkpoint,
                "signal_number": termination.signal_number,
            }
            rank_state = {
                "rank": context.rank,
                "training_generators": generators.state_dict(),
                "process_rng": capture_process_rng_state(context.device),
            }
            save_distributed_checkpoint(
                output_dir=output_dir / "checkpoints",
                update=next_update,
                rank=context.rank,
                world_size=context.world_size,
                model_state=state,
                optimizer_state=optimizer.state_dict() if context.is_main else {},
                trainer_state=trainer_state,
                rank_state=rank_state,
                barrier=dist.barrier,
            )
            if context.is_main:
                prune_full_checkpoints(output_dir / "checkpoints", keep_latest=2)
            dist.barrier()
            last_saved_update = next_update
        if exit_after_checkpoint:
            break

    if context.is_main:
        write_json_atomic(
            output_dir / "run_status.json",
            {
                "status": "interrupted" if exit_after_checkpoint else "completed",
                "last_completed_update": next_update,
                "last_saved_update": last_saved_update,
                "formal_training": run_mode == "formal",
            },
        )
    dist.barrier()
    dataset.close()
    dist.destroy_process_group()
