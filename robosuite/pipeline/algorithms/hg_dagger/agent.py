from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from robosuite.pipeline.algorithms.hil_serl.common.types import EncoderConfig, ReplayBufferConfig, Transition
from robosuite.pipeline.algorithms.hil_serl.common.utils import (
    cfg_get,
    infer_action_bounds,
    infer_action_dim,
    infer_observation_example,
)
from robosuite.pipeline.algorithms.hil_serl.utils.replay_buffer import HILSERLReplayBuffer

from .common import BCConfig, TrainerConfig
from .models import HGDaggerBC


class HGDaggerAgent:
    def __init__(
        self,
        observation_example,
        encoder_config: EncoderConfig,
        bc_config: BCConfig,
        online_buffer_config: ReplayBufferConfig | None = None,
        demo_buffer_config: ReplayBufferConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        action_low=None,
        action_high=None,
    ) -> None:
        self.encoder_config = encoder_config
        self.bc_config = bc_config
        self.trainer_config = trainer_config or TrainerConfig(
            batch_size=online_buffer_config.batch_size if online_buffer_config else 256
        )
        self.online_buffer = HILSERLReplayBuffer(
            online_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="online_buffer",
        )
        self.demo_buffer = HILSERLReplayBuffer(
            demo_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="demo_buffer",
        )
        self.core = HGDaggerBC(
            observation_example=observation_example,
            encoder_config=encoder_config,
            config=bc_config,
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
    ) -> "HGDaggerAgent":
        observation_example = infer_observation_example(
            observation_space=observation_space,
            observation_example=observation_example,
        )
        action_dim = infer_action_dim(action_space=action_space, sample_action=sample_action, cfg=cfg)
        if action_low is None or action_high is None:
            action_low, action_high = infer_action_bounds(action_space=action_space, action_dim=action_dim)

        encoder_cfg = cfg_get(cfg, "encoder", None)
        bc_cfg = cfg_get(cfg, "bc", None)
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
        inference_device = cfg_get(bc_cfg, "inference_device", None)
        if inference_device is not None and str(inference_device).strip().lower() == "auto":
            inference_device = None

        bc_config = BCConfig(
            action_dim=action_dim,
            hidden_dims=tuple(cfg_get(bc_cfg, "hidden_dims", (256, 256))),
            learning_rate=float(cfg_get(bc_cfg, "learning_rate", 3e-4)),
            log_std_min=float(cfg_get(bc_cfg, "log_std_min", -5.0)),
            log_std_max=float(cfg_get(bc_cfg, "log_std_max", 2.0)),
            tanh_squash_distribution=bool(cfg_get(bc_cfg, "tanh_squash_distribution", True)),
            device=str(device or cfg_get(bc_cfg, "device", cfg_get(cfg, "device", "cpu"))),
            inference_device=inference_device,
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
            warmup_steps=int(cfg_get(trainer_cfg, "warmup_steps", cfg_get(cfg, "warmup_steps", 0))),
            updates_per_step=int(cfg_get(trainer_cfg, "updates_per_step", 1)),
            steps_per_update=int(cfg_get(trainer_cfg, "steps_per_update", 50)),
            random_steps=int(cfg_get(trainer_cfg, "random_steps", 0)),
            pretrain_steps=int(cfg_get(trainer_cfg, "pretrain_steps", 20_000)),
        )
        return cls(
            observation_example=observation_example,
            encoder_config=encoder_config,
            bc_config=bc_config,
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

    def sample_demo_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return self.demo_buffer.sample(batch_size=batch_size, device=self.core.device)

    def ready_for_update(self, batch_size: int | None = None) -> bool:
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return len(self.demo_buffer) >= batch_size

    def update(self, *, batch=None, batch_size: int | None = None) -> dict[str, float]:
        batch = batch or self.sample_demo_batch(batch_size=batch_size)
        return self.core.update(batch=batch)

    def save_checkpoint(self, path: str | Path, include_buffers: bool = True, extra: dict[str, Any] | None = None) -> None:
        payload = self.build_checkpoint_payload(include_buffers=include_buffers, extra=extra)
        self.write_checkpoint_payload(path, payload)

    def build_checkpoint_payload(
        self,
        *,
        include_buffers: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "encoder_config": asdict(self.encoder_config),
            "bc_config": asdict(self.bc_config),
            "trainer_config": asdict(self.trainer_config),
            "core": self.core.state_dict(),
        }
        if include_buffers:
            payload["online_buffer"] = self.online_buffer.state_dict()
            payload["demo_buffer"] = self.demo_buffer.state_dict()
        if extra is not None:
            payload["extra"] = extra
        return payload

    def write_checkpoint_payload(self, path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
