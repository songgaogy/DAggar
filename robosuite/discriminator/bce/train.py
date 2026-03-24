from __future__ import annotations

import os
from datetime import datetime

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.bce.dataset import (
    TemporalTransitionDataset,
    build_cached_splits,
    parse_task_beta_map,
)
from robosuite.discriminator.bce.model import build_temporal_pu_discriminator
from robosuite.discriminator.bce.trainer import TPUDTrainer, TrainerConfig
from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_payload(
    model,
    history: dict[str, dict[str, dict[str, float]]],
    cfg: DictConfig,
    latent_dim: int,
    action_dim: int,
    task_to_index: dict[str, int],
    split_summary: dict[str, dict[str, dict[str, int]]],
    beta_by_task: dict[str, float],
    epoch: int,
) -> dict:
    return {
        "model": model.state_dict(),
        "history": history,
        "cfg": cfg,
        "latent_dim": int(latent_dim),
        "action_dim": int(action_dim),
        "horizon": int(cfg.data.transition_horizon),
        "task_to_index": dict(task_to_index),
        "split_summary": split_summary,
        "beta_by_task": dict(beta_by_task),
        "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
        "image_size": int(cfg.data.image_size),
        "epoch": int(epoch),
    }


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = FrozenFlowMultitaskEncoder(
        checkpoint_path=str(cfg.policy.ckpt),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
    )

    try:
        beta_by_task = parse_task_beta_map(
            cfg_data=cfg.data,
            default_beta=float(cfg.data.labels.default_beta),
        )
        cached_splits, split_summary, task_to_index = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=seed,
        )

        train_dataset = TemporalTransitionDataset(
            trajectory_refs=cached_splits["train"],
            beta_by_task=beta_by_task,
            horizon=int(cfg.data.transition_horizon),
            preload_to_memory=bool(cfg.data.preload_train_to_memory),
        )
        val_dataset = None
        if len(cached_splits["val"]) > 0:
            val_dataset = TemporalTransitionDataset(
                trajectory_refs=cached_splits["val"],
                beta_by_task=beta_by_task,
                horizon=int(cfg.data.transition_horizon),
                preload_to_memory=bool(cfg.data.preload_eval_to_memory),
            )

        print(
            f"[tpud] train_transitions={len(train_dataset)} "
            f"positive_samples={train_dataset.num_positive_samples} "
            f"failure_samples={train_dataset.num_failure_samples} "
            f"latent_dim={train_dataset.latent_dim} action_dim={train_dataset.action_dim}"
        )
        print(f"[tpud] train_task_positive_counts={train_dataset.task_positive_counts}")
        print(f"[tpud] train_task_failure_counts={train_dataset.task_failure_counts}")
        if val_dataset is not None:
            print(
                f"[tpud] val_transitions={len(val_dataset)} "
                f"positive_samples={val_dataset.num_positive_samples} "
                f"failure_samples={val_dataset.num_failure_samples}"
            )

        model = build_temporal_pu_discriminator(
            latent_dim=int(train_dataset.latent_dim),
            action_dim=int(train_dataset.action_dim),
            num_tasks=int(len(task_to_index)),
            cfg_model=cfg.model,
            transition_horizon=int(cfg.data.transition_horizon),
        )

        trainer = TPUDTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=TrainerConfig(
                batch_size=int(cfg.training.batch_size),
                num_workers=int(cfg.training.num_workers),
                prefetch_factor=int(cfg.training.prefetch_factor),
                persistent_workers=bool(cfg.training.persistent_workers),
                learning_rate=float(cfg.training.lr),
                weight_decay=float(cfg.training.weight_decay),
                epochs=int(cfg.training.epochs),
                positive_sampling_ratio=float(cfg.training.positive_ratio),
                grad_clip_norm=float(cfg.training.grad_clip_norm),
                log_every=int(cfg.training.log_every),
                device=str(cfg.training.device),
                amp=bool(cfg.training.amp),
            ),
            device=str(cfg.training.device),
            cfg=cfg,
            metadata={
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "split_summary": split_summary,
                "beta_by_task": beta_by_task,
            },
        )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        if str(cfg.save_name) in {"", "None", "null"}:
            save_name = f"tpud_{_now_tag()}.pt"
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
                beta_by_task=beta_by_task,
                epoch=epoch,
            )
            torch.save(payload, periodic_path)
            print(f"[tpud] Saved checkpoint to: {periodic_path}")

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
            beta_by_task=beta_by_task,
            epoch=int(cfg.training.epochs),
        )
        torch.save(final_payload, save_path_final)
        print(f"[tpud] Saved final checkpoint to: {save_path_final}")
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
