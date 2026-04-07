from __future__ import annotations

import os

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
)
from robosuite.discriminator.lpb_score.core.model import DSMModel, build_dsm_model
from robosuite.discriminator.lpb_score.core.trainer import Trainer, TrainerConfig


def _build_payload(
    model: DSMModel,
    history: dict[str, dict[str, dict[str, float]]],
    cfg: DictConfig,
    latent_dim: int,
    action_dim: int,
    task_to_index: dict[str, int],
    split_summary: dict[str, dict[str, dict[str, int]]],
    epoch: int,
) -> dict:
    """Package model state and run metadata into a checkpoint payload."""
    return {
        "model": model.state_dict(),
        "history": history,
        "cfg": cfg,
        "latent_dim": int(latent_dim),
        "action_dim": int(action_dim),
        "horizon": int(cfg.data.transition_horizon),
        "tau_dim": int(model.tau_dim),
        "task_to_index": dict(task_to_index),
        "split_summary": split_summary,
        "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
        "image_size": int(cfg.data.image_size),
        "epoch": int(epoch),
    }


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
        expert_sampling_ratio=float(cfg.training.expert_ratio),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        log_every=int(cfg.training.log_every),
        device=str(cfg.training.device),
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

        print(
            f"[lpb_score] train_transitions={len(train_dataset)} "
            f"expert_samples={train_dataset.num_expert_samples} "
            f"rollout_samples={train_dataset.num_rollout_samples} "
            f"num_train_trajectories={len(train_refs)} "
            f"latent_dim={train_dataset.latent_dim} action_dim={train_dataset.action_dim}"
        )
        if val_dataset is not None:
            print(
                f"[lpb_score] val_transitions={len(val_dataset)} "
                f"expert_samples={val_dataset.num_expert_samples} "
                f"rollout_samples={val_dataset.num_rollout_samples} "
                f"num_val_trajectories={len(val_refs)}"
            )

        model = build_dsm_model(
            latent_dim=int(train_dataset.latent_dim),
            action_dim=int(train_dataset.action_dim),
            cfg_model=cfg.model,
            transition_horizon=int(cfg.data.transition_horizon),
        )

        trainer = _build_trainer(
            cfg=cfg,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
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
            periodic_name = f"{stem}_ep{epoch:04d}{ext}"
            periodic_path = os.path.join(save_dir, periodic_name)
            payload = _build_payload(
                model=model,
                history=history,
                cfg=cfg,
                latent_dim=train_dataset.latent_dim,
                action_dim=train_dataset.action_dim,
                task_to_index=task_to_index,
                split_summary=split_summary,
                epoch=epoch,
            )
            torch.save(payload, periodic_path)
            print(f"[lpb_score] Saved checkpoint to: {periodic_path}")

        history = trainer.fit(
            save_freq=int(cfg.training.save_freq),
            save_callback=_save_periodic,
        )

        final_payload = _build_payload(
            model=model,
            history=history,
            cfg=cfg,
            latent_dim=train_dataset.latent_dim,
            action_dim=train_dataset.action_dim,
            task_to_index=task_to_index,
            split_summary=split_summary,
            epoch=int(cfg.training.epochs),
        )
        torch.save(final_payload, save_path_final)
        print(f"[lpb_score] Saved final checkpoint to: {save_path_final}")
    finally:
        encoder.close()
