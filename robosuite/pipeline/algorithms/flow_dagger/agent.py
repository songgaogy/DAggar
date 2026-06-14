from __future__ import annotations

import copy
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robosuite.pipeline.common.types import EncoderConfig, ReplayBufferConfig, Transition
from robosuite.pipeline.common.utils import (
    cfg_get,
    infer_action_dim,
    infer_observation_example,
)
from robosuite.policy.flow_multi.utils.datasets import DEFAULT_TASK_PROMPTS

from .common import FlowAugmentationConfig, FlowDaggerConfig, TrainerConfig
from .models import FlowDaggerPolicy
from .replay_buffer import FlowDaggerReplayBuffer


def _resolve_language_instruction(task_name: str, task_prompt_map: dict[str, Any] | None = None) -> str:
    prompt_map = {} if task_prompt_map is None else dict(task_prompt_map)
    prompt_value = prompt_map.get(task_name, DEFAULT_TASK_PROMPTS.get(task_name, task_name))
    if isinstance(prompt_value, str):
        return str(prompt_value)
    if isinstance(prompt_value, (list, tuple)) and len(prompt_value) > 0:
        return str(prompt_value[0])
    return str(task_name)


class FlowDaggerAgent:
    def __init__(
        self,
        *,
        observation_example,
        encoder_config: EncoderConfig,
        flow_config: FlowDaggerConfig,
        model_cfg: dict[str, Any],
        camera_names: list[str],
        online_buffer_config: ReplayBufferConfig | None = None,
        demo_buffer_config: ReplayBufferConfig | None = None,
        trainer_config: TrainerConfig | None = None,
    ) -> None:
        self.encoder_config = encoder_config
        self.flow_config = flow_config
        self.model_cfg = copy.deepcopy(model_cfg)
        self.camera_names = [str(name) for name in camera_names]
        self.task_name = str(flow_config.task_name or "task")
        self.language_instruction = str(flow_config.language_instruction or self.task_name)
        self.trainer_config = trainer_config or TrainerConfig(
            batch_size=online_buffer_config.batch_size if online_buffer_config else 64
        )
        self.online_buffer = FlowDaggerReplayBuffer(
            online_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="online_buffer",
            camera_names=self.camera_names,
            action_horizon=int(flow_config.action_horizon),
            image_size=int(flow_config.image_size),
            augmentation_config=flow_config.augmentation,
        )
        self.demo_buffer = FlowDaggerReplayBuffer(
            demo_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="demo_buffer",
            camera_names=self.camera_names,
            action_horizon=int(flow_config.action_horizon),
            image_size=int(flow_config.image_size),
            augmentation_config=flow_config.augmentation,
        )
        self.core = FlowDaggerPolicy(
            model_cfg=self.model_cfg,
            config=flow_config,
            camera_names=self.camera_names,
        )
        self.core.set_language_instruction(self.language_instruction)
        _ = observation_example

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
    ) -> "FlowDaggerAgent":
        observation_example = infer_observation_example(
            observation_space=observation_space,
            observation_example=observation_example,
        )
        _ = action_low
        _ = action_high
        action_dim = infer_action_dim(action_space=action_space, sample_action=sample_action, cfg=cfg)
        if not isinstance(observation_example, dict) or "state" not in observation_example:
            raise KeyError("flow-dagger requires observation_example to contain a 'state' entry.")
        proprio_dim = int(np.asarray(observation_example["state"]).reshape(-1).shape[0])

        encoder_cfg = cfg_get(cfg, "encoder", None)
        flow_cfg = cfg_get(cfg, "flow", None)
        online_buffer_cfg = cfg_get(cfg, "online_buffer", None)
        demo_buffer_cfg = cfg_get(cfg, "demo_buffer", None)
        trainer_cfg = cfg_get(cfg, "trainer", None)

        encoder_config = EncoderConfig(
            encoder_type=str(cfg_get(encoder_cfg, "encoder_type", "flow-multi")),
            image_keys=tuple(cfg_get(cfg, "camera_names", ())),
            proprio_keys=("state",),
            image_size=int(cfg_get(flow_cfg, "image_size", 128)),
        )
        model_cfg = copy.deepcopy(cfg_get(flow_cfg, "model", {}))
        augmentation_cfg = cfg_get(flow_cfg, "augmentation", None)
        batch_size = int(cfg_get(trainer_cfg, "batch_size", cfg_get(cfg, "batch_size", 64)))
        inference_device = cfg_get(flow_cfg, "inference_device", None)
        if inference_device is not None and str(inference_device).strip().lower() == "auto":
            inference_device = None

        task_name = str(cfg_get(cfg, "task_name", "task"))
        task_prompt_map = cfg_get(flow_cfg, "task_prompt_map", None)
        flow_config = FlowDaggerConfig(
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_horizon=int(cfg_get(flow_cfg, "action_horizon", 8)),
            execute_horizon=int(cfg_get(flow_cfg, "execute_horizon", 1)),
            image_size=int(cfg_get(flow_cfg, "image_size", 128)),
            learning_rate=float(cfg_get(flow_cfg, "learning_rate", 1e-4)),
            weight_decay=float(cfg_get(flow_cfg, "weight_decay", 1e-6)),
            grad_clip_norm=float(cfg_get(flow_cfg, "grad_clip_norm", 1.0)),
            lambda_endpoint=float(cfg_get(flow_cfg, "lambda_endpoint", 0.5)),
            lambda_smooth=float(cfg_get(flow_cfg, "lambda_smooth", 0.05)),
            n_ode_steps=int(cfg_get(flow_cfg, "n_ode_steps", 8)),
            device=str(device or cfg_get(flow_cfg, "device", cfg_get(cfg, "device", "cpu"))),
            inference_device=inference_device,
            task_name=task_name,
            language_instruction=_resolve_language_instruction(task_name, task_prompt_map),
            augmentation=FlowAugmentationConfig(
                minimal_shift_pad=int(cfg_get(augmentation_cfg, "minimal_shift_pad", 2)),
                eye_in_hand_crop_scale=float(cfg_get(augmentation_cfg, "eye_in_hand_crop_scale", 0.88)),
            ),
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
            max_pending_updates=int(cfg_get(trainer_cfg, "max_pending_updates", 8)),
        )
        return cls(
            observation_example=observation_example,
            encoder_config=encoder_config,
            flow_config=flow_config,
            model_cfg=model_cfg,
            camera_names=[str(name) for name in cfg_get(cfg, "camera_names", ())],
            online_buffer_config=online_buffer_config,
            demo_buffer_config=demo_buffer_config,
            trainer_config=trainer_config,
        )

    def select_action(self, obs, deterministic: bool = False):
        return self.core.select_action(obs=obs, deterministic=deterministic)

    def reset_policy_state(self) -> None:
        self.core.reset_action_chunk()

    def notify_intervention(self) -> None:
        self.core.notify_intervention()

    def sync_inference_policy(self) -> None:
        self.core.sync_inference_policy()

    def has_normalizers(self) -> bool:
        return self.core.has_normalizers()

    def fit_normalizers_from_transitions(self, transitions: list[Transition]) -> None:
        trajectories: list[list[Transition]] = []
        current_trajectory: list[Transition] = []
        for transition in transitions:
            current_trajectory.append(transition)
            if bool(transition.done):
                trajectories.append(current_trajectory)
                current_trajectory = []
        if current_trajectory:
            trajectories.append(current_trajectory)

        action_windows = []
        proprio_values = []
        horizon = int(self.flow_config.action_horizon)
        for trajectory in trajectories:
            if len(trajectory) < horizon:
                continue
            for start in range(len(trajectory) - horizon + 1):
                action_windows.append(
                    np.stack(
                        [np.asarray(trajectory[start + offset].action, dtype=np.float32) for offset in range(horizon)],
                        axis=0,
                    )
                )
                proprio_values.append(np.asarray(trajectory[start].obs["state"], dtype=np.float32))
        if len(action_windows) == 0 or len(proprio_values) == 0:
            raise RuntimeError(
                f"Unable to fit flow-dagger normalizers because no valid action_horizon={horizon} windows were found."
            )
        action_array = np.stack(action_windows, axis=0)
        proprio_array = np.stack(proprio_values, axis=0)
        self.core.set_normalizers(
            action_mean=action_array.mean(axis=0),
            action_std=action_array.std(axis=0) + 1e-6,
            proprio_mean=proprio_array.mean(axis=0),
            proprio_std=proprio_array.std(axis=0) + 1e-6,
        )

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
        return self.demo_buffer.sample(
            batch_size=batch_size,
            action_mean=self.core.act_mean,
            action_std=self.core.act_std,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.core.device,
            augment=True,
        )

    def ready_for_update(self, batch_size: int | None = None) -> bool:
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return self.demo_buffer.num_valid_sequences() >= batch_size and self.has_normalizers()

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
            "flow_config": asdict(self.flow_config),
            "trainer_config": asdict(self.trainer_config),
            "model_cfg": copy.deepcopy(self.model_cfg),
            "camera_names": list(self.camera_names),
            "task_name": self.task_name,
            "language_instruction": self.language_instruction,
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

    def load_flow_policy_checkpoint(self, path: str | Path, *, task_name: str | None = None) -> dict[str, Any]:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load flow policy checkpoint from {path}: {exc}") from exc
        model_state = payload.get("ema_model", payload.get("model"))
        if model_state is None:
            raise KeyError(f"Checkpoint {path} is missing both 'ema_model' and 'model'.")
        self.core.load_model_state(model_state, strict=True)
        self.core.set_normalizers(
            action_mean=payload.get("act_mean"),
            action_std=payload.get("act_std"),
            proprio_mean=payload.get("prop_mean"),
            proprio_std=payload.get("prop_std"),
        )
        resolved_task_name = str(task_name or self.task_name)
        task_prompt_map = payload.get("task_prompt_map", None)
        self.task_name = resolved_task_name
        self.language_instruction = _resolve_language_instruction(resolved_task_name, task_prompt_map)
        self.core.set_language_instruction(self.language_instruction)
        self.reset_policy_state()
        return payload

    def load_checkpoint(self, path: str | Path, load_buffers: bool = True) -> dict[str, Any]:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc
        self.model_cfg = copy.deepcopy(payload.get("model_cfg", self.model_cfg))
        self.camera_names = [str(name) for name in payload.get("camera_names", self.camera_names)]
        self.task_name = str(payload.get("task_name", self.task_name))
        self.language_instruction = str(payload.get("language_instruction", self.language_instruction))
        self.core.set_language_instruction(self.language_instruction)
        self.core.load_state_dict(payload["core"])
        if load_buffers:
            if "online_buffer" in payload:
                self.online_buffer.load_state_dict(payload["online_buffer"])
            if "demo_buffer" in payload:
                self.demo_buffer.load_state_dict(payload["demo_buffer"])
        return payload.get("extra", {})
