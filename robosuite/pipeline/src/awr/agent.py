from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robosuite.pipeline.src.data.transitions import ReplayBufferConfig, Transition
from robosuite.policy.flow_multi.utils.datasets import DEFAULT_TASK_PROMPTS

from .batches import AWRActorBatch, AWRStepBatch
from .config import (
    AWRConfig,
    EncoderConfig,
    FlowAugmentationConfig,
    TrainerConfig,
    cfg_get,
)
from .model import AWRFlowPolicy
from .replay_buffer import AWRReplayBuffer


CHECKPOINT_FORMAT = "robosuite-awr"
CHECKPOINT_VERSION = 1


def _language_instruction(task_name: str, prompt_map: Any = None) -> str:
    prompts = {} if prompt_map is None else dict(prompt_map)
    value = prompts.get(task_name, DEFAULT_TASK_PROMPTS.get(task_name, task_name))
    return str(value[0] if isinstance(value, (list, tuple)) else value)


class AWRAgent:
    def __init__(
        self,
        *,
        observation_example: Any,
        encoder_config: EncoderConfig,
        awr_config: AWRConfig,
        model_cfg: dict[str, Any],
        camera_names: list[str],
        online_buffer_config: ReplayBufferConfig,
        demo_buffer_config: ReplayBufferConfig,
        trainer_config: TrainerConfig,
    ) -> None:
        if not isinstance(observation_example, dict) or "state" not in observation_example:
            raise KeyError("AWR observations must contain a 'state' entry.")
        self.encoder_config = encoder_config
        self.awr_config = awr_config
        self.model_cfg = copy.deepcopy(model_cfg)
        self.camera_names = list(camera_names)
        self.task_name = awr_config.task_name
        self.language_instruction = str(
            awr_config.language_instruction or awr_config.task_name
        )
        self.trainer_config = trainer_config
        buffer_kwargs = {
            "camera_names": self.camera_names,
            "action_horizon": awr_config.action_horizon,
            "image_size": awr_config.image_size,
            "augmentation_config": awr_config.augmentation,
        }
        self.online_buffer = AWRReplayBuffer(
            online_buffer_config, name="online_buffer", **buffer_kwargs
        )
        self.demo_buffer = AWRReplayBuffer(
            demo_buffer_config, name="demo_buffer", **buffer_kwargs
        )
        self.core = AWRFlowPolicy(
            model_cfg=self.model_cfg,
            config=awr_config,
            camera_names=self.camera_names,
        )
        self.core.set_language_instruction(self.language_instruction)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        observation_example: Any,
        action_low: Any,
        action_high: Any,
    ) -> "AWRAgent":
        _ = action_high
        if not isinstance(observation_example, dict) or "state" not in observation_example:
            raise KeyError("AWR observations must contain a 'state' entry.")
        awr = cfg_get(cfg, "awr", {})
        trainer = cfg_get(cfg, "trainer", {})
        online = cfg_get(cfg, "online_buffer", {})
        demo = cfg_get(cfg, "demo_buffer", {})
        camera_names = [str(name) for name in cfg_get(cfg, "camera_names", ())]
        task_name = str(cfg_get(cfg, "task_name", "PickPlaceCereal"))
        action_dim = int(np.asarray(action_low).reshape(-1).shape[0])
        proprio_dim = int(
            np.asarray(observation_example["state"]).reshape(-1).shape[0]
        )
        action_horizon = int(cfg_get(awr, "action_horizon", 8))
        device = str(cfg_get(awr, "device", "cuda:0"))
        augmentation = cfg_get(awr, "augmentation", {})
        awr_config = AWRConfig(
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_horizon=action_horizon,
            execute_horizon=int(
                cfg_get(awr, "execute_horizon", action_horizon)
            ),
            image_size=int(cfg_get(awr, "image_size", 128)),
            actor_learning_rate=float(
                cfg_get(awr, "actor_learning_rate", 1e-4)
            ),
            critic_learning_rate=float(
                cfg_get(awr, "critic_learning_rate", 3e-4)
            ),
            weight_decay=float(cfg_get(awr, "weight_decay", 1e-6)),
            grad_clip_norm=float(cfg_get(awr, "grad_clip_norm", 1.0)),
            lambda_endpoint=float(cfg_get(awr, "lambda_endpoint", 0.5)),
            lambda_smooth=float(cfg_get(awr, "lambda_smooth", 0.05)),
            beta=float(cfg_get(awr, "beta", 3.0)),
            max_adv_weight=float(cfg_get(awr, "max_adv_weight", 100.0)),
            discount=float(cfg_get(awr, "discount", 0.99)),
            expectile=float(cfg_get(awr, "expectile", 0.7)),
            critic_hidden_dims=tuple(
                cfg_get(awr, "critic_hidden_dims", (512, 512))
            ),
            n_ode_steps=int(cfg_get(awr, "n_ode_steps", 8)),
            device=device,
            inference_device=str(
                cfg_get(awr, "inference_device", device)
            ),
            task_name=task_name,
            language_instruction=str(
                cfg_get(
                    awr,
                    "language_instruction",
                    _language_instruction(
                        task_name, cfg_get(awr, "task_prompt_map", None)
                    ),
                )
            ),
            augmentation=FlowAugmentationConfig(
                minimal_shift_pad=int(
                    cfg_get(augmentation, "minimal_shift_pad", 2)
                ),
                eye_in_hand_crop_scale=float(
                    cfg_get(augmentation, "eye_in_hand_crop_scale", 0.88)
                ),
            ),
        )
        value_batch_size = int(
            cfg_get(
                trainer,
                "value_batch_size",
                cfg_get(trainer, "batch_size", 256),
            )
        )
        actor_batch_size = int(
            cfg_get(
                trainer,
                "actor_batch_size",
                cfg_get(trainer, "batch_size", 128),
            )
        )
        buffer_batch_size = max(value_batch_size, actor_batch_size)
        trainer_config = TrainerConfig(
            value_batch_size=value_batch_size,
            actor_batch_size=actor_batch_size,
            warmup_steps=int(cfg_get(trainer, "warmup_steps", 0)),
            episodes_per_train=int(
                cfg_get(trainer, "episodes_per_train", 10)
            ),
            updates_per_train=int(
                cfg_get(trainer, "updates_per_train", 2000)
            ),
            inference_sync_interval=int(
                cfg_get(trainer, "inference_sync_interval", 50)
            ),
            value_warmup_steps=int(
                cfg_get(trainer, "value_warmup_steps", 20_000)
            ),
        )
        encoder_config = EncoderConfig(
            encoder_type=str(
                cfg_get(cfg_get(cfg, "encoder", {}), "encoder_type", "flow-multi")
            ),
            image_keys=tuple(camera_names),
            image_size=awr_config.image_size,
        )
        return cls(
            observation_example=observation_example,
            encoder_config=encoder_config,
            awr_config=awr_config,
            model_cfg=copy.deepcopy(cfg_get(awr, "model", {})),
            camera_names=camera_names,
            online_buffer_config=ReplayBufferConfig(
                capacity=int(cfg_get(online, "capacity", 200_000)),
                batch_size=buffer_batch_size,
            ),
            demo_buffer_config=ReplayBufferConfig(
                capacity=int(cfg_get(demo, "capacity", 200_000)),
                batch_size=buffer_batch_size,
            ),
            trainer_config=trainer_config,
        )

    def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        return self.core.select_action(obs, deterministic)

    def reset_policy_state(self) -> None:
        self.core.reset_action_chunk()

    def notify_intervention(self) -> None:
        self.core.reset_action_chunk()

    def sync_inference_policy(self) -> None:
        self.core.sync_inference_policy()

    def has_normalizers(self) -> bool:
        return self.core.has_normalizers()

    def clone_observation(self, obs: Any) -> Any:
        return self.core.clone_observation(obs)

    def fit_normalizers_from_transitions(
        self, transitions: list[Transition]
    ) -> None:
        trajectories: list[list[Transition]] = []
        current: list[Transition] = []
        for transition in transitions:
            current.append(transition)
            if transition.done:
                trajectories.append(current)
                current = []
        if current:
            trajectories.append(current)
        actions, proprio = [], []
        horizon = self.awr_config.action_horizon
        for trajectory in trajectories:
            for start in range(len(trajectory) - horizon + 1):
                actions.append(
                    np.stack(
                        [
                            np.asarray(trajectory[start + offset].action)
                            for offset in range(horizon)
                        ]
                    )
                )
                proprio.append(np.asarray(trajectory[start].obs["state"]))
        if not actions:
            raise RuntimeError(
                f"No complete action_horizon={horizon} window is available."
            )
        action_array = np.asarray(actions, dtype=np.float32)
        proprio_array = np.asarray(proprio, dtype=np.float32)
        self.core.set_normalizers(
            action_mean=action_array.mean(axis=0),
            action_std=action_array.std(axis=0) + 1e-6,
            proprio_mean=proprio_array.mean(axis=0),
            proprio_std=proprio_array.std(axis=0) + 1e-6,
        )

    def store_transition(self, transition: Transition) -> None:
        self.online_buffer.add(transition)
        if transition.is_intervention:
            self.demo_buffer.add(transition)

    def store_online_transition(self, transition: Transition) -> None:
        self.online_buffer.add(transition)

    def store_demo_transition(self, transition: Transition) -> None:
        self.demo_buffer.add(transition)

    def _actor_batch(
        self, buffer: AWRReplayBuffer, role: str, batch_size: int
    ) -> AWRActorBatch:
        return buffer.sample_actor_batch(
            batch_size,
            action_mean=self.core.act_mean,
            action_std=self.core.act_std,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.awr_config.device,
            buffer_role=role,
        )

    def _step_batch(
        self,
        buffer: AWRReplayBuffer,
        role: str,
        batch_size: int,
        *,
        augment: bool = True,
    ) -> AWRStepBatch:
        return buffer.sample_step_batch(
            batch_size,
            discount=self.awr_config.discount,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.awr_config.device,
            augment=augment,
            buffer_role=role,
        )

    def sample_mixed_actor_batch(
        self, batch_size: int | None = None
    ) -> AWRActorBatch:
        size = int(batch_size or self.trainer_config.actor_batch_size)
        if size < 2:
            raise ValueError("Mixed AWR batches require batch_size >= 2.")
        demo_size = size // 2
        return AWRActorBatch.concat(
            [
                self._actor_batch(self.demo_buffer, "demo", demo_size),
                self._actor_batch(self.online_buffer, "online", size - demo_size),
            ]
        )

    def sample_mixed_step_batch(
        self, batch_size: int | None = None
    ) -> AWRStepBatch:
        size = int(batch_size or self.trainer_config.value_batch_size)
        if size < 2:
            raise ValueError("Mixed AWR batches require batch_size >= 2.")
        demo_size = size // 2
        return AWRStepBatch.concat(
            [
                self._step_batch(self.demo_buffer, "demo", demo_size),
                self._step_batch(
                    self.online_buffer, "online", size - demo_size
                ),
            ]
        )

    def ready_for_value_warmup(self, batch_size: int | None = None) -> bool:
        _ = batch_size
        return self.online_buffer.num_valid_sequences() > 0 and self.has_normalizers()

    def ready_for_update(self, batch_size: int | None = None) -> bool:
        _ = batch_size
        return (
            self.online_buffer.num_valid_sequences() > 0
            and self.demo_buffer.num_valid_sequences() > 0
            and self.has_normalizers()
        )

    def update_value_only(
        self, batch_size: int | None = None
    ) -> dict[str, float]:
        size = int(batch_size or self.trainer_config.value_batch_size)
        return self.core.update_value(
            self._step_batch(
                self.online_buffer, "online", size, augment=False
            )
        )

    def update(self, batch_size: int | None = None) -> dict[str, float]:
        value_size = int(batch_size or self.trainer_config.value_batch_size)
        actor_size = int(batch_size or self.trainer_config.actor_batch_size)
        return self.core.update(
            self.sample_mixed_step_batch(value_size),
            self.sample_mixed_actor_batch(actor_size),
        )

    def build_checkpoint_payload(
        self,
        *,
        include_buffers: bool = True,
        trainer_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "encoder_config": asdict(self.encoder_config),
            "awr_config": asdict(self.awr_config),
            "trainer_config": asdict(self.trainer_config),
            "model_cfg": copy.deepcopy(self.model_cfg),
            "camera_names": self.camera_names,
            "core": self.core.state_dict(),
            "trainer_state": copy.deepcopy(trainer_state or {}),
        }
        if include_buffers:
            payload["buffers"] = {
                "online": self.online_buffer.state_dict(),
                "demo": self.demo_buffer.state_dict(),
            }
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_buffers: bool = True,
        trainer_state: dict[str, Any] | None = None,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(
            self.build_checkpoint_payload(
                include_buffers=include_buffers, trainer_state=trainer_state
            ),
            temporary,
        )
        temporary.replace(target)

    def load_checkpoint(
        self, path: str | Path, *, load_buffers: bool = True
    ) -> dict[str, Any]:
        payload = torch.load(
            Path(path), map_location=self.awr_config.device, weights_only=False
        )
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or int(payload.get("version", -1)) != CHECKPOINT_VERSION
        ):
            raise ValueError("Unsupported AWR checkpoint format.")
        self.core.load_state_dict(payload["core"])
        if load_buffers:
            self.online_buffer.load_state_dict(payload["buffers"]["online"])
            self.demo_buffer.load_state_dict(payload["buffers"]["demo"])
        return copy.deepcopy(payload.get("trainer_state", {}))

    def load_flow_policy_checkpoint(
        self, path: str | Path, *, task_name: str | None = None
    ) -> dict[str, Any]:
        payload = torch.load(
            Path(path), map_location=self.awr_config.device, weights_only=False
        )
        state = payload.get("ema_model", payload.get("model"))
        if state is None:
            raise KeyError("Flow checkpoint has neither 'ema_model' nor 'model'.")
        self.core.load_actor_model_state(state)
        self.core.set_normalizers(
            action_mean=payload.get("act_mean"),
            action_std=payload.get("act_std"),
            proprio_mean=payload.get("prop_mean"),
            proprio_std=payload.get("prop_std"),
        )
        self.task_name = str(task_name or self.task_name)
        self.language_instruction = _language_instruction(
            self.task_name, payload.get("task_prompt_map")
        )
        self.core.set_language_instruction(self.language_instruction)
        return payload


__all__ = [
    "AWRAgent",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
]
