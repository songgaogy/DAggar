from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from robosuite.pipeline.src.data.transitions import Transition

from .encoders import EncoderConfig
from .replay import (
    HILSERLReplayBuffer,
    ReplayBatch,
    ReplayBufferConfig,
    concat_replay_batches,
)
from .sac import HILSERLSAC, SACConfig


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class TrainerConfig:
    batch_size: int = 256
    cta_ratio: int = 2
    warmup_steps: int = 100
    steps_per_update: int = 50
    random_steps: int = 0
    online_fraction: float = 0.5
    max_learner_steps: int = 1_000_000

    def split_batch_sizes(self) -> tuple[int, int]:
        if self.batch_size < 2 or self.batch_size % 2 != 0:
            raise ValueError("HIL-SERL requires an even batch_size of at least 2.")
        if self.online_fraction != 0.5:
            raise ValueError("HIL-SERL uses a fixed 50/50 online/demo replay mixture.")
        online_batch = int(round(self.batch_size * self.online_fraction))
        demo_batch = self.batch_size - online_batch
        return online_batch, demo_batch


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
        default_batch_size = online_buffer_config.batch_size if online_buffer_config else 256
        self.trainer_config = trainer_config or TrainerConfig(batch_size=default_batch_size)
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
        self._checkpoint_lock = threading.RLock()
        self._replay_transaction_lock = threading.RLock()

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        observation_example: Any,
        action_low: Any,
        action_high: Any,
    ) -> "HILSERLAgent":
        encoder_cfg = cfg_get(cfg, "encoder", None)
        sac_cfg = cfg_get(cfg, "sac", None)
        online_buffer_cfg = cfg_get(cfg, "online_buffer", None)
        demo_buffer_cfg = cfg_get(cfg, "demo_buffer", None)
        trainer_cfg = cfg_get(cfg, "trainer", None)
        if any(item is None for item in (encoder_cfg, sac_cfg, online_buffer_cfg, demo_buffer_cfg, trainer_cfg)):
            raise ValueError("HIL-SERL config requires encoder, sac, online_buffer, demo_buffer, and trainer sections.")
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if action_low.shape != action_high.shape:
            raise ValueError("action_low and action_high must have matching shapes.")
        action_dim = int(action_low.size)
        if action_dim != 7:
            raise ValueError(f"HIL-SERL expects six arm actions plus one gripper action, got {action_dim}.")

        encoder_config = EncoderConfig(
            encoder_type=str(cfg_get(encoder_cfg, "encoder_type", "resnet-pretrained")),
            image_keys=tuple(cfg_get(encoder_cfg, "image_keys", ())),
            proprio_keys=tuple(cfg_get(encoder_cfg, "proprio_keys", ())),
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
        batch_size = int(cfg_get(trainer_cfg, "batch_size", 256))
        sac_config = SACConfig(
            action_dim=action_dim,
            actor_hidden_dims=tuple(cfg_get(sac_cfg, "actor_hidden_dims", (256, 256))),
            critic_hidden_dims=tuple(cfg_get(sac_cfg, "critic_hidden_dims", (256, 256))),
            discount=float(cfg_get(sac_cfg, "discount", 0.97)),
            tau=float(cfg_get(sac_cfg, "tau", 0.005)),
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
            std_min=float(cfg_get(sac_cfg, "std_min", 1e-5)),
            std_max=float(cfg_get(sac_cfg, "std_max", 5.0)),
            augmentation_padding=int(cfg_get(sac_cfg, "augmentation_padding", 4)),
            device=str(cfg_get(sac_cfg, "device", "cuda:0")),
            inference_device=cfg_get(sac_cfg, "inference_device", "cuda:1"),
        )
        online_buffer_config = ReplayBufferConfig(
            capacity=int(cfg_get(online_buffer_cfg, "capacity", 200_000)),
            batch_size=batch_size,
        )
        demo_buffer_config = ReplayBufferConfig(
            capacity=int(cfg_get(demo_buffer_cfg, "capacity", 200_000)),
            batch_size=batch_size,
        )
        trainer_config = TrainerConfig(
            batch_size=batch_size,
            cta_ratio=int(cfg_get(trainer_cfg, "cta_ratio", 2)),
            warmup_steps=int(cfg_get(trainer_cfg, "warmup_steps", 100)),
            steps_per_update=int(cfg_get(trainer_cfg, "steps_per_update", 50)),
            random_steps=int(cfg_get(trainer_cfg, "random_steps", 0)),
            online_fraction=float(cfg_get(trainer_cfg, "online_fraction", 0.5)),
            max_learner_steps=int(cfg_get(trainer_cfg, "max_learner_steps", 1_000_000)),
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
        with self._replay_transaction_lock:
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
        self._split_batch_sizes(int(batch_size or self.trainer_config.batch_size))
        return len(self.online_buffer) >= int(self.trainer_config.warmup_steps) and len(self.demo_buffer) > 0

    def update(
        self,
        *,
        batch: ReplayBatch | None = None,
        batch_size: int | None = None,
        critic_only: bool = False,
    ) -> dict[str, float]:
        with self._checkpoint_lock:
            batch = batch or self.sample_mixed_batch(batch_size=batch_size)
            return self.core.update(
                batch=batch,
                update_actor=not critic_only,
                update_temperature=not critic_only,
            )

    def save_checkpoint(
        self,
        path: str | Path,
        include_buffers: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._checkpoint_lock, self._replay_transaction_lock:
            payload = self.build_checkpoint_payload(include_buffers=include_buffers, extra=extra)
            self.write_checkpoint_payload(path, payload)

    def build_checkpoint_payload(
        self,
        *,
        include_buffers: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._checkpoint_lock, self._replay_transaction_lock:
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
        with self._checkpoint_lock, self._replay_transaction_lock:
            self.core.load_state_dict(payload["core"])
            if load_buffers:
                if "online_buffer" in payload:
                    self.online_buffer.load_state_dict(payload["online_buffer"])
                if "demo_buffer" in payload:
                    self.demo_buffer.load_state_dict(payload["demo_buffer"])
        return payload.get("extra", {})

    def _split_batch_sizes(self, batch_size: int) -> tuple[int, int]:
        if batch_size != int(self.trainer_config.batch_size):
            if batch_size < 2 or batch_size % 2 != 0:
                raise ValueError("HIL-SERL requires an even batch_size of at least 2.")
            if float(self.trainer_config.online_fraction) != 0.5:
                raise ValueError("HIL-SERL uses a fixed 50/50 online/demo replay mixture.")
            return batch_size // 2, batch_size // 2
        return self.trainer_config.split_batch_sizes()
