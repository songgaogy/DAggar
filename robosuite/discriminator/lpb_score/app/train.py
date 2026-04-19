"""Training orchestration for the window-conditioned latent DSM.

Execution order in ``run_train``:
    1) Distributed setup (optional DDP) and seeds.
    2) ``build_flow_encoder``: load multitask flow policy checkpoint as encoder.
    3) ``build_split_refs``: sample train/val demo references from HDF5 trees.
    4) ``build_training_datasets``: materialize or index trajectories, build
       ``LatentTransitionDataset`` (window samples + optional RAM residency).
    5) ``build_dsm_model``: wrap encoder + ``ChunkConditionedDSM`` predictor.
    6) ``Trainer``: normalization warmup (positive latents), then epoch loop with
       AMP, gradient clip, optional EMA, checkpoints via ``_build_payload``.

Checkpoint payloads include DSM weights, policy encoder export metadata, latent
normalization buffers, and Hydra config for reproducibility.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.lpb_score.app.pipeline import (
    build_flow_encoder,
    build_training_datasets,
    now_tag,
)
from robosuite.discriminator.lpb_score.core.dataset import (
    LatentTransitionDataset,
    build_split_refs,
    resolve_window_size,
)
from robosuite.discriminator.lpb_score.core.model import MODEL_ARCHITECTURE, DSMModel, build_dsm_model
from robosuite.discriminator.lpb_score.core.trainer import Trainer, TrainerConfig


def _task_names_from_index(task_to_index: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(task_to_index.items(), key=lambda item: int(item[1]))]


def _cfg_value(cfg: object, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _parameter_counts(module: torch.nn.Module) -> tuple[int, int]:
    total = int(sum(param.numel() for param in module.parameters()))
    trainable = int(sum(param.numel() for param in module.parameters() if param.requires_grad))
    return total, trainable


def _build_payload(
    *,
    model_state: dict,
    history: dict[str, dict[str, dict[str, float]]],
    cfg: DictConfig,
    model: DSMModel,
    task_to_index: dict[str, int],
    split_summary: dict[str, dict[str, dict[str, int]]],
    epoch: int,
    policy_checkpoint_payload: dict,
    model_online_state: dict | None = None,
) -> dict:
    payload: dict = {
        "model": model_state,
        "history": history,
        "cfg": cfg,
        "model_architecture": MODEL_ARCHITECTURE,
        "latent_dim": int(model.latent_dim),
        "window_size": int(model.window_size),
        "chunk_dim": int(model.chunk_dim),
        "num_tasks": int(len(task_to_index)),
        "task_to_index": dict(task_to_index),
        "task_names": _task_names_from_index(task_to_index),
        "split_summary": split_summary,
        "image_size": int(model.policy_encoder.image_size),
        "epoch": int(epoch),
        "policy_checkpoint_payload": policy_checkpoint_payload,
        "normalization_stats": {
            "latent_mean": model.latent_mean.detach().cpu(),
            "latent_var": model.latent_var.detach().cpu(),
            "normalize_enabled": bool(model.normalize_enabled.item()),
        },
    }
    if model_online_state is not None:
        payload["model_online"] = model_online_state
    return payload


def _build_trainer(
    cfg: DictConfig,
    model: DSMModel,
    train_dataset: LatentTransitionDataset,
    val_dataset: LatentTransitionDataset | None,
    *,
    device: str,
    distributed: bool,
    distributed_rank: int,
    distributed_world_size: int,
    distributed_local_rank: int,
) -> Trainer:
    trainer_cfg = TrainerConfig(
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        learning_rate=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        epochs=int(cfg.training.epochs),
        positive_sampling_ratio=float(cfg.training.positive_ratio),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        log_every=int(cfg.training.log_every),
        device=str(device),
        use_ema=bool(getattr(cfg.training, "use_ema", False)),
        ema_decay=float(getattr(cfg.training, "ema_decay", 0.999)),
        val_use_ema=bool(getattr(cfg.training, "val_use_ema", True)),
        save_ema_in_checkpoint=bool(getattr(cfg.training, "save_ema_in_checkpoint", True)),
        shared_lr_multiplier=float(getattr(cfg.training, "shared_lr_multiplier", 1.0)),
        chunk_branch_lr_multiplier=float(getattr(cfg.training, "chunk_branch_lr_multiplier", 1.0)),
        encoder_branch_lr_multiplier=float(getattr(cfg.training, "encoder_branch_lr_multiplier", 1.0)),
        pin_memory=bool(getattr(cfg.training, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 4)),
        profile_enabled=bool(getattr(cfg.training, "profile_enabled", False)),
        profile_log_every=int(getattr(cfg.training, "profile_log_every", 20)),
        profile_warmup_steps=int(getattr(cfg.training, "profile_warmup_steps", 0)),
        profile_cuda_sync=bool(getattr(cfg.training, "profile_cuda_sync", True)),
        profile_first_step_immediate=bool(getattr(cfg.training, "profile_first_step_immediate", True)),
        profile_norm_progress_every=int(getattr(cfg.training, "profile_norm_progress_every", 10)),
        input_pipeline=str(getattr(cfg.training, "input_pipeline", "batched_iterator")),
        val_input_pipeline=str(getattr(cfg.training, "val_input_pipeline", "legacy_dataloader")),
        batch_prefetch_depth=int(getattr(cfg.training, "batch_prefetch_depth", 2)),
        train_prefetch_pinned=bool(getattr(cfg.training, "train_prefetch_pinned", True)),
        train_prefetch_thread=bool(getattr(cfg.training, "train_prefetch_thread", True)),
        train_batch_build_workers=int(getattr(cfg.training, "train_batch_build_workers", 4)),
        cuda_prefetch=bool(getattr(cfg.training, "cuda_prefetch", True)),
        batched_trajectory_cache_gb=float(getattr(cfg.training, "batched_trajectory_cache_gb", 8.0)),
        amp_enabled=bool(getattr(cfg.training, "amp_enabled", True)),
        amp_dtype=str(getattr(cfg.training, "amp_dtype", "bf16")),
        seed=int(cfg.seed),
        distributed=bool(distributed),
        distributed_rank=int(distributed_rank),
        distributed_world_size=int(distributed_world_size),
        distributed_local_rank=int(distributed_local_rank),
        ddp_find_unused_parameters=bool(getattr(cfg.training, "ddp_find_unused_parameters", False)),
        ddp_static_graph=bool(getattr(cfg.training, "ddp_static_graph", True)),
        ddp_gradient_as_bucket_view=bool(getattr(cfg.training, "ddp_gradient_as_bucket_view", True)),
        aux_contrastive_weight=float(getattr(cfg.training, "aux_contrastive_weight", 0.1)),
        aux_contrastive_margin=float(getattr(cfg.training, "aux_contrastive_margin", 1.0e-3)),
    )
    return Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_cfg,
        device=str(device),
    )


def _distributed_context(cfg: DictConfig) -> tuple[bool, int, int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        return True, rank, world_size, local_rank, device
    return False, 0, 1, 0, str(cfg.training.device)


def run_train(cfg: DictConfig) -> None:
    """Wire data, model, and trainer from Hydra ``cfg``; save checkpoints on completion."""
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    is_main_process = True
    seed = int(cfg.seed)
    try:
        # build DDP
        distributed, rank, world_size, local_rank, resolved_device = _distributed_context(cfg)
        is_main_process = rank == 0
        full_seed = seed + rank
        torch.manual_seed(full_seed)
        matmul_precision = str(getattr(cfg.training, "matmul_precision", "high"))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_precision)
        if bool(getattr(cfg.training, "cudnn_benchmark", True)) and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(full_seed)

        # Encoder: frozen (or LoRA) FlowMultitaskEncoder from flow policy ckpt.
        encoder = build_flow_encoder(cfg, device_override=resolved_device)
        skip_validation = not bool(getattr(cfg.training, "run_validation", False))

        # HDF5 demo refs: expert + success_rollout -> positive; fail_rollout -> negative.
        split_refs, split_summary, task_to_index = build_split_refs(
            cfg_data=cfg.data,
            seed=seed,
            include_val=not skip_validation,
        )
        # Dataset: cached refs (default) or full tensors; windowed transition samples.
        datasets = build_training_datasets(
            cfg=cfg,
            split_refs=split_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            build_val_dataset=not skip_validation,
            replica_count=world_size,
        )
        train_dataset = datasets.train_dataset
        val_dataset = datasets.val_dataset
        train_refs = datasets.train_refs
        val_refs = datasets.val_refs
        window_size = resolve_window_size(cfg)
        task_names = _task_names_from_index(task_to_index)
        lora_cfg = getattr(cfg.policy, "lora", None)
        lora_rank = int(_cfg_value(lora_cfg, "rank", 8))
        lora_enabled = bool(_cfg_value(lora_cfg, "enabled", False))
        kernel_size = int(_cfg_value(cfg.model, "kernel_size", 3))

        if is_main_process:
            print(
                f"[lpb_score] train_windows={len(train_dataset)} "
                f"positive_samples={train_dataset.num_positive_samples} "
                f"negative_samples={train_dataset.num_negative_samples} "
                f"num_train_trajectories={len(train_refs)} "
                f"latent_dim={encoder.latent_dim} window_size={window_size} "
                f"storage={'in_memory' if train_dataset.uses_in_memory_trajectories else 'cached_refs'} "
                f"resident={train_dataset.num_resident_trajectories} "
                f"trainable_encoder={bool(getattr(cfg.policy, 'trainable_encoder', False))} "
                f"lora_enabled={lora_enabled} lora_rank={lora_rank} "
                f"kernel_size={kernel_size} "
                f"distributed={distributed} world_size={world_size} "
                f"validation={'enabled' if not skip_validation else 'disabled'}"
            )
            if val_dataset is not None:
                print(
                    f"[lpb_score] val_windows={len(val_dataset)} "
                    f"positive_samples={val_dataset.num_positive_samples} "
                    f"negative_samples={val_dataset.num_negative_samples} "
                    f"num_val_trajectories={len(val_refs)} "
                    f"storage={'in_memory' if val_dataset.uses_in_memory_trajectories else 'cached_refs'} "
                    f"resident={val_dataset.num_resident_trajectories}"
                )

        # DSM: temporal conv denoiser on latent chunks + shared encoder.
        model = build_dsm_model(
            latent_dim=int(encoder.latent_dim),
            num_tasks=int(len(task_to_index)),
            cfg_model=cfg.model,
            window_size=window_size,
            policy_encoder=encoder,
            task_names=task_names,
            inference_seed=int(cfg.seed),
        )
        model_total, model_trainable = _parameter_counts(model)
        encoder_total, encoder_trainable = _parameter_counts(model.policy_encoder)
        model_ratio = 0.0 if model_total <= 0 else float(model_trainable) / float(model_total)
        encoder_ratio = 0.0 if encoder_total <= 0 else float(encoder_trainable) / float(encoder_total)

        if is_main_process:
            print(
                f"[lpb_score] trainable_params model={model_trainable}/{model_total} "
                f"({100.0 * model_ratio:.4f}%) encoder={encoder_trainable}/{encoder_total} "
                f"({100.0 * encoder_ratio:.4f}%)"
            )

        # Training loop: sigma-sampled DSM loss + optional contrastive margin; warmup stats first.
        trainer = _build_trainer(
            cfg=cfg,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=resolved_device,
            distributed=distributed,
            distributed_rank=rank,
            distributed_world_size=world_size,
            distributed_local_rank=local_rank,
        )

        save_dir = to_absolute_path(str(cfg.save_dir))
        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        if str(cfg.save_name) in {"", "None", "null"}:
            save_name = f"lpb_score_dsm_{now_tag()}.pt"
        else:
            save_name = str(cfg.save_name)
        save_path_final = os.path.join(save_dir, save_name)
        stem, ext = os.path.splitext(save_name)
        if ext == "":
            ext = ".pt"

        policy_checkpoint_payload = model.policy_encoder.export_policy_checkpoint_payload()

        def _save_periodic(epoch: int, history: dict[str, dict[str, dict[str, float]]]) -> None:
            periodic_name = f"{stem}_ep{epoch:04d}{ext}"
            periodic_path = os.path.join(save_dir, periodic_name)
            primary_sd, online_sd = trainer.checkpoint_state_dicts()
            payload = _build_payload(
                model_state=primary_sd,
                history=history,
                cfg=cfg,
                model=model,
                task_to_index=task_to_index,
                split_summary=split_summary,
                epoch=epoch,
                policy_checkpoint_payload=policy_checkpoint_payload,
                model_online_state=online_sd,
            )
            torch.save(payload, periodic_path)
            print(f"[lpb_score] Saved checkpoint to: {periodic_path}")

        history = trainer.fit(
            save_freq=int(cfg.training.save_freq),
            save_callback=_save_periodic,
        )

        if distributed:
            dist.barrier()
        if is_main_process:
            primary_sd, online_sd = trainer.checkpoint_state_dicts()
            final_payload = _build_payload(
                model_state=primary_sd,
                history=history,
                cfg=cfg,
                model=model,
                task_to_index=task_to_index,
                split_summary=split_summary,
                epoch=int(cfg.training.epochs),
                policy_checkpoint_payload=policy_checkpoint_payload,
                model_online_state=online_sd,
            )
            torch.save(final_payload, save_path_final)
            print(f"[lpb_score] Saved final checkpoint to: {save_path_final}")
            
    finally:
        if "encoder" in locals():
            encoder.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
