from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .utils.replay_buffer import HILSERLReplayBuffer
from .models.sac import HILSERLSAC
from .common.types import EncoderConfig, ReplayBatch, ReplayBufferConfig, SACConfig, TrainerConfig, Transition
from .common.utils import (
    cfg_get,
    concat_replay_batches,
    infer_action_bounds,
    infer_action_dim,
    infer_observation_example,
)


class HILSERLAgent:
    def __init__(
        self,
        observation_example,
        encoder_config: EncoderConfig,
        sac_config: SACConfig,
        online_buffer_config: ReplayBufferConfig | None = None,
        demo_buffer_config: ReplayBufferConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        action_low=None,
        action_high=None,
    ) -> None:
        self.encoder_config = encoder_config
        self.sac_config = sac_config
        self.trainer_config = trainer_config or TrainerConfig(batch_size=online_buffer_config.batch_size if online_buffer_config else 256)
        self.online_buffer = HILSERLReplayBuffer(
            online_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="online_buffer",
        )
        self.demo_buffer = HILSERLReplayBuffer(
            demo_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="demo_buffer",
        )
        self.core = HILSERLSAC(
            observation_example=observation_example,
            encoder_config=encoder_config,
            config=sac_config,
            action_low=action_low,
            action_high=action_high,
        )

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        observation_space: Any = None,
        action_space: Any = None,
        observation_example: Any = None,
        sample_action: Any = None,
        action_low: Any = None,
        action_high: Any = None,
        device: str | None = None,
    ) -> "HILSERLAgent":
        observation_example = infer_observation_example(
            observation_space=observation_space,
            observation_example=observation_example,
        )
        action_dim = infer_action_dim(action_space=action_space, sample_action=sample_action, cfg=cfg)
        if action_low is None or action_high is None:
            action_low, action_high = infer_action_bounds(action_space=action_space, action_dim=action_dim)

        encoder_cfg = cfg_get(cfg, "encoder", None)
        sac_cfg = cfg_get(cfg, "sac", None)
        online_buffer_cfg = cfg_get(cfg, "online_buffer", None)
        demo_buffer_cfg = cfg_get(cfg, "demo_buffer", None)
        trainer_cfg = cfg_get(cfg, "trainer", None)

        encoder_config = EncoderConfig(
            encoder_type=str(cfg_get(encoder_cfg, "encoder_type", cfg_get(cfg, "encoder_type", "resnet-pretrained"))),
            image_keys=tuple(cfg_get(encoder_cfg, "image_keys", cfg_get(cfg, "image_keys", ()))),
            proprio_keys=tuple(cfg_get(encoder_cfg, "proprio_keys", cfg_get(cfg, "proprio_keys", ()))),
            feature_dim=int(cfg_get(encoder_cfg, "feature_dim", 256)),
            image_size=int(cfg_get(encoder_cfg, "image_size", 84)),
            cnn_channels=tuple(cfg_get(encoder_cfg, "cnn_channels", (32, 64, 64, 64))),
            use_layer_norm=bool(cfg_get(encoder_cfg, "use_layer_norm", True)),
            resnet_name=str(cfg_get(encoder_cfg, "resnet_name", "resnet10")),
            pretrained=bool(cfg_get(encoder_cfg, "pretrained", True)),
            freeze_backbone=bool(cfg_get(encoder_cfg, "freeze_backbone", True)),
            share_image_encoder=bool(cfg_get(encoder_cfg, "share_image_encoder", False)),
            proprio_feature_dim=int(cfg_get(encoder_cfg, "proprio_feature_dim", 64)),
            num_spatial_blocks=int(cfg_get(encoder_cfg, "num_spatial_blocks", 8)),
            pretrained_path=cfg_get(encoder_cfg, "pretrained_path", None),
        )
        batch_size = int(cfg_get(trainer_cfg, "batch_size", cfg_get(cfg, "batch_size", 256)))
        sac_config = SACConfig(
            action_dim=action_dim,
            actor_hidden_dims=tuple(cfg_get(sac_cfg, "actor_hidden_dims", (256, 256))),
            critic_hidden_dims=tuple(cfg_get(sac_cfg, "critic_hidden_dims", (256, 256))),
            discount=float(cfg_get(sac_cfg, "discount", cfg_get(cfg, "discount", 0.97))),
            tau=float(cfg_get(sac_cfg, "tau", cfg_get(cfg, "tau", 0.005))),
            actor_lr=float(cfg_get(sac_cfg, "actor_lr", 3e-4)),
            critic_lr=float(cfg_get(sac_cfg, "critic_lr", 3e-4)),
            alpha_lr=float(cfg_get(sac_cfg, "alpha_lr", 3e-4)),
            init_temperature=float(cfg_get(sac_cfg, "init_temperature", 1e-2)),
            target_entropy=cfg_get(sac_cfg, "target_entropy", None),
            auto_entropy_tuning=bool(cfg_get(sac_cfg, "auto_entropy_tuning", True)),
            backup_entropy=bool(cfg_get(sac_cfg, "backup_entropy", False)),
            reward_bias=float(cfg_get(sac_cfg, "reward_bias", 0.0)),
            critic_ensemble_size=int(cfg_get(sac_cfg, "critic_ensemble_size", 2)),
            critic_subsample_size=cfg_get(sac_cfg, "critic_subsample_size", None),
            log_std_min=float(cfg_get(sac_cfg, "log_std_min", -5.0)),
            log_std_max=float(cfg_get(sac_cfg, "log_std_max", 2.0)),
            device=str(device or cfg_get(sac_cfg, "device", cfg_get(cfg, "device", "cpu"))),
            inference_device=cfg_get(sac_cfg, "inference_device", None),
        )
        online_buffer_config = ReplayBufferConfig(
            capacity=int(cfg_get(online_buffer_cfg, "capacity", cfg_get(cfg, "online_buffer_capacity", 200_000))),
            batch_size=batch_size,
        )
        demo_buffer_config = ReplayBufferConfig(
            capacity=int(cfg_get(demo_buffer_cfg, "capacity", cfg_get(cfg, "demo_buffer_capacity", 200_000))),
            batch_size=batch_size,
        )
        trainer_config = TrainerConfig(
            batch_size=batch_size,
            cta_ratio=int(cfg_get(trainer_cfg, "cta_ratio", cfg_get(cfg, "cta_ratio", 2))),
            warmup_steps=int(cfg_get(trainer_cfg, "warmup_steps", cfg_get(cfg, "warmup_steps", 100))),
            updates_per_step=int(cfg_get(trainer_cfg, "updates_per_step", 1)),
            steps_per_update=int(cfg_get(trainer_cfg, "steps_per_update", 50)),
            random_steps=int(cfg_get(trainer_cfg, "random_steps", 0)),
            online_fraction=float(cfg_get(trainer_cfg, "online_fraction", 0.5)),
        )
        return cls(
            observation_example=observation_example,
            encoder_config=encoder_config,
            sac_config=sac_config,
            online_buffer_config=online_buffer_config,
            demo_buffer_config=demo_buffer_config,
            trainer_config=trainer_config,
            action_low=action_low,
            action_high=action_high,
        )

    def select_action(self, obs, deterministic: bool = False):
        return self.core.select_action(obs=obs, deterministic=deterministic)

    def sync_inference_policy(self) -> None:
        self.core.sync_inference_policy()

    def store_online_transition(self, transition: Transition) -> None:
        self.online_buffer.add(transition)

    def store_demo_transition(self, transition: Transition) -> None:
        self.demo_buffer.add(transition)

    def store_transition(self, transition: Transition) -> None:
        self.store_online_transition(transition)
        if transition.is_intervention:
            self.store_demo_transition(transition)

    def sample_mixed_batch(self, batch_size: int | None = None) -> ReplayBatch:
        batch_size = int(batch_size or self.trainer_config.batch_size)
        online_batch_size, demo_batch_size = self._split_batch_sizes(batch_size)
        online_batch = self.online_buffer.sample(batch_size=online_batch_size, device=self.core.device)
        demo_batch = self.demo_buffer.sample(batch_size=demo_batch_size, device=self.core.device)
        return concat_replay_batches(online_batch, demo_batch)

    def ready_for_update(self, batch_size: int | None = None) -> bool:
        batch_size = int(batch_size or self.trainer_config.batch_size)
        online_batch_size, demo_batch_size = self._split_batch_sizes(batch_size)
        return len(self.online_buffer) >= online_batch_size and len(self.demo_buffer) >= demo_batch_size

    def update(
        self,
        *,
        batch: ReplayBatch | None = None,
        batch_size: int | None = None,
        critic_only: bool = False,
    ) -> dict[str, float]:
        batch = batch or self.sample_mixed_batch(batch_size=batch_size)
        return self.core.update(
            batch=batch,
            update_actor=not critic_only,
            update_temperature=not critic_only,
        )

    def save_checkpoint(self, path: str | Path, include_buffers: bool = True, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder_config": asdict(self.encoder_config),
            "sac_config": asdict(self.sac_config),
            "trainer_config": asdict(self.trainer_config),
            "core": self.core.state_dict(),
        }
        if include_buffers:
            payload["online_buffer"] = self.online_buffer.state_dict()
            payload["demo_buffer"] = self.demo_buffer.state_dict()
        if extra is not None:
            payload["extra"] = extra
        tmp_path = path.with_name(f".{path.name}.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str | Path, load_buffers: bool = True) -> dict[str, Any]:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc
        self.core.load_state_dict(payload["core"])
        if load_buffers:
            if "online_buffer" in payload:
                self.online_buffer.load_state_dict(payload["online_buffer"])
            if "demo_buffer" in payload:
                self.demo_buffer.load_state_dict(payload["demo_buffer"])
        return payload.get("extra", {})

    def _split_batch_sizes(self, batch_size: int) -> tuple[int, int]:
        online_fraction = float(self.trainer_config.online_fraction)
        online_batch_size = int(round(batch_size * online_fraction))
        online_batch_size = min(max(1, online_batch_size), batch_size - 1)
        return online_batch_size, batch_size - online_batch_size
