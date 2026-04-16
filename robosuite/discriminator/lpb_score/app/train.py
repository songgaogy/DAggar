"""Hydra training entry: latent caches, positive-pool normalization, DSM optimization, checkpoints."""

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
    build_cached_splits,
    load_cached_latent_trajectory,
    resolve_window_size,
)
from robosuite.discriminator.lpb_score.core.model import MODEL_ARCHITECTURE, DSMModel, build_dsm_model
from robosuite.discriminator.lpb_score.core.trainer import Trainer, TrainerConfig


def _compute_positive_normalization_stats(
    *,
    positive_refs,
    min_variance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Per-dimension mean and variance over all timesteps from non-failure train trajectories only."""
    if not positive_refs:
        raise RuntimeError("Expected non-empty positive refs to compute normalization stats.")

    latent_sum = None
    latent_sq_sum = None
    latent_count = 0

    for ref in positive_refs:
        traj = load_cached_latent_trajectory(ref)
        latents = np.asarray(traj.latents, dtype=np.float64)
        if latent_sum is None:
            latent_sum = np.zeros((latents.shape[1],), dtype=np.float64)
            latent_sq_sum = np.zeros((latents.shape[1],), dtype=np.float64)
        latent_sum += latents.sum(axis=0)
        latent_sq_sum += np.square(latents).sum(axis=0)
        latent_count += int(latents.shape[0])

    if latent_count <= 0:
        raise RuntimeError("Positive normalization stats require positive latent counts.")

    latent_mean = latent_sum / float(latent_count)
    latent_var = np.maximum(latent_sq_sum / float(latent_count) - np.square(latent_mean), float(min_variance))
    return {
        "latent_mean": latent_mean.astype(np.float32),
        "latent_var": latent_var.astype(np.float32),
        "latent_count": np.asarray(latent_count, dtype=np.int64),
    }


def _build_payload(
    model_state: dict,
    history: dict[str, dict[str, dict[str, float]]],
    cfg: DictConfig,
    latent_dim: int,
    window_size: int,
    chunk_dim: int,
    task_to_index: dict[str, int],
    split_summary: dict[str, dict[str, dict[str, int]]],
    epoch: int,
    normalization_stats: dict[str, np.ndarray],
    model_online_state: dict | None = None,
) -> dict:
    """Serialize weights, Hydra config, ``task_to_index``, and normalization stats for inference."""
    payload: dict = {
        "model": model_state,
        "history": history,
        "cfg": cfg,
        "model_architecture": MODEL_ARCHITECTURE,
        "latent_dim": int(latent_dim),
        "window_size": int(window_size),
        "chunk_dim": int(chunk_dim),
        "num_tasks": int(len(task_to_index)),
        "task_to_index": dict(task_to_index),
        "split_summary": split_summary,
        "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
        "image_size": int(cfg.data.image_size),
        "epoch": int(epoch),
        "normalization_stats": {
            key: np.asarray(value)
            for key, value in normalization_stats.items()
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
) -> Trainer:
    """Build the trainer so Hydra config stays out of the main loop."""
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
    )
    return Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_cfg,
        device=str(cfg.training.device),
    )


def run_train(cfg: DictConfig) -> None:
    """Run cached splits, build ``DSMModel`` with ``num_tasks=len(task_to_index)``, train, and save."""
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = build_flow_encoder(cfg)

    try:
        cached_splits, split_summary, task_to_index = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=seed,
        )
        datasets = build_training_datasets(
            cfg=cfg,
            cached_splits=cached_splits,
        )
        train_dataset = datasets.train_dataset
        val_dataset = datasets.val_dataset
        train_refs = datasets.train_refs
        val_refs = datasets.val_refs
        positive_train_refs = [ref for ref in train_refs if ref.data_type != "fail_rollout"]
        window_size = resolve_window_size(cfg)

        print(
            f"[lpb_score] train_windows={len(train_dataset)} "
            f"positive_samples={train_dataset.num_positive_samples} "
            f"negative_samples={train_dataset.num_negative_samples} "
            f"num_train_trajectories={len(train_refs)} "
            f"latent_dim={train_dataset.latent_dim} window_size={window_size}"
        )
        if val_dataset is not None:
            print(
                f"[lpb_score] val_windows={len(val_dataset)} "
                f"positive_samples={val_dataset.num_positive_samples} "
                f"negative_samples={val_dataset.num_negative_samples} "
                f"num_val_trajectories={len(val_refs)}"
            )

        model = build_dsm_model(
            latent_dim=int(train_dataset.latent_dim),
            num_tasks=int(len(task_to_index)),
            cfg_model=cfg.model,
            window_size=window_size,
        )
        normalization_stats = _compute_positive_normalization_stats(positive_refs=positive_train_refs)
        model.set_normalization_stats(
            latent_mean=normalization_stats["latent_mean"],
            latent_var=normalization_stats["latent_var"],
        )
        print(f"[lpb_score] positive_norm_stats latent_count={int(normalization_stats['latent_count'])}")

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

        def _save_periodic(epoch: int, history: dict[str, dict[str, dict[str, float]]]) -> None:
            """Write intermediate checkpoint when ``training.save_freq`` divides ``epoch``."""
            periodic_name = f"{stem}_ep{epoch:04d}{ext}"
            periodic_path = os.path.join(save_dir, periodic_name)
            primary_sd, online_sd = trainer.checkpoint_state_dicts()
            payload = _build_payload(
                model_state=primary_sd,
                history=history,
                cfg=cfg,
                latent_dim=train_dataset.latent_dim,
                window_size=window_size,
                chunk_dim=model.chunk_dim,
                task_to_index=task_to_index,
                split_summary=split_summary,
                epoch=epoch,
                normalization_stats=normalization_stats,
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
            latent_dim=train_dataset.latent_dim,
            window_size=window_size,
            chunk_dim=model.chunk_dim,
            task_to_index=task_to_index,
            split_summary=split_summary,
            epoch=int(cfg.training.epochs),
            normalization_stats=normalization_stats,
            model_online_state=online_sd,
        )
        torch.save(final_payload, save_path_final)
        print(f"[lpb_score] Saved final checkpoint to: {save_path_final}")
    finally:
        encoder.close()
