from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.lpb_new.dataset import (
    LatentTransitionDataset,
    build_cached_splits,
    filter_refs_by_data_types,
)
from robosuite.discriminator.lpb_new.model import LatentDynamicsModel, build_latent_dynamics_predictor
from robosuite.discriminator.lpb_new.trainer import Trainer, TrainerConfig


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_payload(
    model: LatentDynamicsModel,
    history: dict[str, dict[str, dict[str, float]]],
    cfg: DictConfig,
    latent_dim: int,
    action_dim: int,
    task_to_index: dict[str, int],
    split_summary: dict[str, dict[str, dict[str, int]]],
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
        cached_splits, split_summary, task_to_index = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=seed,
        )
        train_refs = filter_refs_by_data_types(
            cached_splits["train"],
            list(getattr(cfg.data, "train_data_types", ["expert", "success_rollout"])),
        )
        train_dataset = LatentTransitionDataset(
            trajectory_refs=train_refs,
            horizon=int(cfg.data.transition_horizon),
            preload_to_memory=bool(cfg.data.preload_train_to_memory),
        )
        val_dataset = None
        val_refs = filter_refs_by_data_types(
            cached_splits["val"],
            list(getattr(cfg.data, "val_data_types", ["expert", "success_rollout"])),
        )
        if len(val_refs) > 0:
            val_dataset = LatentTransitionDataset(
                trajectory_refs=val_refs,
                horizon=int(cfg.data.transition_horizon),
                preload_to_memory=bool(cfg.data.preload_eval_to_memory),
            )

        print(
            f"[lpb_new] train_transitions={len(train_dataset)} "
            f"expert_samples={train_dataset.num_expert_samples} "
            f"rollout_samples={train_dataset.num_rollout_samples} "
            f"num_train_trajectories={len(train_refs)} "
            f"latent_dim={train_dataset.latent_dim} action_dim={train_dataset.action_dim}"
        )
        if val_dataset is not None:
            print(
                f"[lpb_new] val_transitions={len(val_dataset)} "
                f"expert_samples={val_dataset.num_expert_samples} "
                f"rollout_samples={val_dataset.num_rollout_samples} "
                f"num_val_trajectories={len(val_refs)}"
            )

        predictor = build_latent_dynamics_predictor(
            latent_dim=int(train_dataset.latent_dim),
            action_dim=int(train_dataset.action_dim),
            cfg_model=cfg.model,
            transition_horizon=int(cfg.data.transition_horizon),
        )
        model = LatentDynamicsModel(predictor=predictor)

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=TrainerConfig(
                batch_size=int(cfg.training.batch_size),
                num_workers=int(cfg.training.num_workers),
                learning_rate=float(cfg.training.lr),
                weight_decay=float(cfg.training.weight_decay),
                epochs=int(cfg.training.epochs),
                expert_sampling_ratio=float(cfg.training.expert_ratio),
                grad_clip_norm=float(cfg.training.grad_clip_norm),
                log_every=int(cfg.training.log_every),
                device=str(cfg.training.device),
            ),
            device=str(cfg.training.device),
        )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        if str(cfg.save_name) in {"", "None", "null"}:
            save_name = f"lpb_new_{_now_tag()}.pt"
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
            print(f"[lpb_new] Saved checkpoint to: {periodic_path}")

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
        print(f"[lpb_new] Saved final checkpoint to: {save_path_final}")
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
