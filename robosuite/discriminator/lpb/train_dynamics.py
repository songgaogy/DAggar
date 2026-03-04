from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import random_split

from robosuite.discriminator.lpb.dataset import LatentDynamicsDataset
from robosuite.discriminator.lpb.model import DynamicsModel, DynamicsPredictor, Encoder
from robosuite.discriminator.lpb.trainer import Trainer, TrainerConfig


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_ckpt_path(path: Optional[str]) -> Optional[str]:
    if path in (None, ""):
        return None
    abs_path = to_absolute_path(str(path))
    if os.path.exists(abs_path):
        return abs_path
    return None


def _build_payload(
    model: DynamicsModel,
    history: Dict[str, Dict[str, Dict[str, float]]],
    cfg: DictConfig,
    latent_dim: int,
    action_dim: int,
    proprio_dim: int,
    epoch: int,
) -> dict:
    return {
        "model": model.state_dict(),
        "history": history,
        "cfg": cfg,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "camera_name": str(cfg.data.camera_name),
        "horizon": int(cfg.data.horizon),
        "epoch": int(epoch),
    }


@hydra.main(version_base="1.2", config_path="./config", config_name="train_dynamics")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    expert_paths = [to_absolute_path(str(p)) for p in cfg.data.expert_paths]
    rollout_paths = [to_absolute_path(str(p)) for p in cfg.data.rollout_paths]
    proprio_indices = list(cfg.data.proprio_indices) if cfg.data.proprio_indices else None

    dataset = LatentDynamicsDataset(
        expert_paths=expert_paths,
        rollout_paths=rollout_paths,
        camera_name=str(cfg.data.camera_name),
        horizon=int(cfg.data.horizon),
        proprio_indices=proprio_indices,
        image_size=(int(cfg.data.image_size), int(cfg.data.image_size)),
        max_trajectories=(
            None if int(cfg.data.max_trajectories) <= 0 else int(cfg.data.max_trajectories)
        ),
    )
    print(
        f"dataset size={len(dataset)} expert_samples={dataset.num_expert_samples} "
        f"rollout_samples={dataset.num_rollout_samples} action_dim={dataset.action_dim} "
        f"proprio_dim={dataset.proprio_dim}"
    )

    n_total = len(dataset)
    n_val = int(round(float(cfg.split.val_ratio) * n_total))
    n_val = min(max(n_val, 0), n_total - 1) if n_total > 1 else 0
    n_train = n_total - n_val
    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=split_gen)

    resnet_ckpt = _resolve_ckpt_path(cfg.encoder.ckpt_path)
    encoder = Encoder(
        checkpoint_path=resnet_ckpt,
        pretrained=bool(cfg.encoder.pretrained) if resnet_ckpt is None else False,
        freeze=not bool(cfg.encoder.trainable),
        normalize_input=bool(cfg.encoder.normalize_input),
    )
    predictor = DynamicsPredictor(
        latent_dim=encoder.latent_dim,
        proprio_dim=dataset.proprio_dim,
        action_dim=dataset.action_dim,
        d_model=int(cfg.model.d_model),
        num_layers=int(cfg.model.num_layers),
        nhead=int(cfg.model.num_heads),
        dropout=float(cfg.model.dropout),
        max_action_horizon=max(int(cfg.model.max_action_horizon), int(cfg.data.horizon)),
    )
    model = DynamicsModel(encoder=encoder, predictor=predictor)

    trainer_cfg = TrainerConfig(
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        learning_rate=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        epochs=int(cfg.training.epochs),
        expert_sampling_ratio=float(cfg.training.expert_ratio),
        proprio_loss_weight=float(cfg.training.proprio_loss_weight),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        log_every=int(cfg.training.log_every),
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_cfg,
        device=str(cfg.training.device),
    )
    save_dir = to_absolute_path(str(cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if str(cfg.save_name) in ("", "null", "None"):
        save_name = f"lpb_dynamics_{_now_tag()}.pt"
    else:
        save_name = str(cfg.save_name)
    save_path_final = os.path.join(save_dir, save_name)
    stem, ext = os.path.splitext(save_name)
    if ext == "":
        ext = ".pt"

    def _save_periodic(epoch: int, hist: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        periodic_name = f"{stem}_ep{epoch:04d}{ext}"
        periodic_path = os.path.join(save_dir, periodic_name)
        payload = _build_payload(
            model=model,
            history=hist,
            cfg=cfg,
            latent_dim=encoder.latent_dim,
            action_dim=dataset.action_dim,
            proprio_dim=dataset.proprio_dim,
            epoch=epoch,
        )
        torch.save(payload, periodic_path)
        print(f"Saved checkpoint to: {periodic_path}")

    history = trainer.fit(
        save_freq=int(cfg.training.save_freq),
        save_callback=_save_periodic,
    )

    final_payload = _build_payload(
        model=model,
        history=history,
        cfg=cfg,
        latent_dim=encoder.latent_dim,
        action_dim=dataset.action_dim,
        proprio_dim=dataset.proprio_dim,
        epoch=int(cfg.training.epochs),
    )
    torch.save(final_payload, save_path_final)
    print(f"Saved final checkpoint to: {save_path_final}")


if __name__ == "__main__":
    main()
