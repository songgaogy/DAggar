"""Hydra training entry for the joint encoder + chunk DSM pipeline."""

from __future__ import annotations

import os

import numpy as np
import torch
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
    print_split_summary,
    resolve_window_size,
)
from robosuite.discriminator.lpb_score.core.model import MODEL_ARCHITECTURE, DSMModel, build_dsm_model
from robosuite.discriminator.lpb_score.core.trainer import Trainer, TrainerConfig


def _task_names_from_index(task_to_index: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(task_to_index.items(), key=lambda item: int(item[1]))]


def _normalize_stats_to_numpy(normalization_stats: dict[str, torch.Tensor | np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in normalization_stats.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


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
    normalization_stats: dict[str, torch.Tensor | np.ndarray],
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
        "normalization_stats": _normalize_stats_to_numpy(normalization_stats),
        "policy_checkpoint_payload": policy_checkpoint_payload,
    }
    if model_online_state is not None:
        payload["model_online"] = model_online_state
    return payload


def _build_trainer(
    cfg: DictConfig,
    model: DSMModel,
    train_dataset: LatentTransitionDataset,
    val_dataset: LatentTransitionDataset | None,
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
        device=str(cfg.training.device),
        use_ema=bool(getattr(cfg.training, "use_ema", False)),
        ema_decay=float(getattr(cfg.training, "ema_decay", 0.999)),
        val_use_ema=bool(getattr(cfg.training, "val_use_ema", True)),
        save_ema_in_checkpoint=bool(getattr(cfg.training, "save_ema_in_checkpoint", True)),
        shared_lr_multiplier=float(getattr(cfg.training, "shared_lr_multiplier", 1.0)),
        chunk_branch_lr_multiplier=float(getattr(cfg.training, "chunk_branch_lr_multiplier", 1.0)),
        encoder_branch_lr_multiplier=float(getattr(cfg.training, "encoder_branch_lr_multiplier", 1.0)),
        normalization_batch_size=int(getattr(cfg.training, "normalization_batch_size", 256)),
        pin_memory=bool(getattr(cfg.training, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 4)),
    )
    return Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_cfg,
        device=str(cfg.training.device),
    )


def run_train(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if bool(getattr(cfg.training, "cudnn_benchmark", True)) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = build_flow_encoder(cfg)

    try:
        split_refs, split_summary, task_to_index = build_split_refs(
            cfg_data=cfg.data,
            seed=seed,
        )
        print_split_summary(split_summary)
        datasets = build_training_datasets(
            cfg=cfg,
            split_refs=split_refs,
            encoder=encoder,
            task_to_index=task_to_index,
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

        print(
            f"[lpb_score] train_windows={len(train_dataset)} "
            f"positive_samples={train_dataset.num_positive_samples} "
            f"negative_samples={train_dataset.num_negative_samples} "
            f"num_train_trajectories={len(train_refs)} "
            f"latent_dim={encoder.latent_dim} window_size={window_size} "
            f"trainable_encoder={bool(getattr(cfg.policy, 'trainable_encoder', False))} "
            f"lora_enabled={lora_enabled} lora_rank={lora_rank} "
            f"kernel_size={kernel_size}"
        )
        if val_dataset is not None:
            print(
                f"[lpb_score] val_windows={len(val_dataset)} "
                f"positive_samples={val_dataset.num_positive_samples} "
                f"negative_samples={val_dataset.num_negative_samples} "
                f"num_val_trajectories={len(val_refs)}"
            )

        model = build_dsm_model(
            latent_dim=int(encoder.latent_dim),
            num_tasks=int(len(task_to_index)),
            cfg_model=cfg.model,
            window_size=window_size,
            policy_encoder=encoder,
            task_names=task_names,
        )
        model_total, model_trainable = _parameter_counts(model)
        encoder_total, encoder_trainable = _parameter_counts(model.policy_encoder)
        model_ratio = 0.0 if model_total <= 0 else float(model_trainable) / float(model_total)
        encoder_ratio = 0.0 if encoder_total <= 0 else float(encoder_trainable) / float(encoder_total)
        print(
            f"[lpb_score] trainable_params model={model_trainable}/{model_total} "
            f"({100.0 * model_ratio:.4f}%) encoder={encoder_trainable}/{encoder_total} "
            f"({100.0 * encoder_ratio:.4f}%)"
        )
        trainer = _build_trainer(
            cfg=cfg,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        if bool(getattr(cfg.training, "use_ema", False)):
            print(
                f"[lpb_score] EMA enabled decay={float(getattr(cfg.training, 'ema_decay', 0.999))} "
                f"val_use_ema={bool(getattr(cfg.training, 'val_use_ema', True))} "
                f"save_ema_in_checkpoint={bool(getattr(cfg.training, 'save_ema_in_checkpoint', True))}"
            )

        save_dir = to_absolute_path(str(cfg.save_dir))
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
                normalization_stats=trainer.checkpoint_normalization_stats(),
                policy_checkpoint_payload=policy_checkpoint_payload,
                model_online_state=online_sd,
            )
            torch.save(payload, periodic_path)
            print(f"[lpb_score] Saved checkpoint to: {periodic_path}")

        history = trainer.fit(
            save_freq=int(cfg.training.save_freq),
            save_callback=_save_periodic,
        )

        primary_sd, online_sd = trainer.checkpoint_state_dicts()
        final_payload = _build_payload(
            model_state=primary_sd,
            history=history,
            cfg=cfg,
            model=model,
            task_to_index=task_to_index,
            split_summary=split_summary,
            epoch=int(cfg.training.epochs),
            normalization_stats=trainer.checkpoint_normalization_stats(),
            policy_checkpoint_payload=policy_checkpoint_payload,
            model_online_state=online_sd,
        )
        torch.save(final_payload, save_path_final)
        print(f"[lpb_score] Saved final checkpoint to: {save_path_final}")
    finally:
        encoder.close()
