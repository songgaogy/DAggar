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

from .common import AWRActorBatch, AWRConfig, AWRStepBatch, FlowAugmentationConfig, TrainerConfig
from .models import AWRFlowPolicy
from .replay_buffer import AWRReplayBuffer


def _resolve_language_instruction(task_name: str, task_prompt_map: dict[str, Any] | None = None) -> str:
    prompt_map = {} if task_prompt_map is None else dict(task_prompt_map)
    prompt_value = prompt_map.get(task_name, DEFAULT_TASK_PROMPTS.get(task_name, task_name))
    if isinstance(prompt_value, str):
        return str(prompt_value)
    if isinstance(prompt_value, (list, tuple)) and len(prompt_value) > 0:
        return str(prompt_value[0])
    return str(task_name)


class AWRAgent:
    def __init__(
        self,
        *,
        observation_example,
        encoder_config: EncoderConfig,
        awr_config: AWRConfig,
        model_cfg: dict[str, Any],
        camera_names: list[str],
        online_buffer_config: ReplayBufferConfig | None = None,
        demo_buffer_config: ReplayBufferConfig | None = None,
        trainer_config: TrainerConfig | None = None,
    ) -> None:
        self.encoder_config = encoder_config
        self.awr_config = awr_config
        self.model_cfg = copy.deepcopy(model_cfg)
        self.camera_names = [str(name) for name in camera_names]
        self.task_name = str(awr_config.task_name or "task")
        self.language_instruction = str(awr_config.language_instruction or self.task_name)
        self.trainer_config = trainer_config or TrainerConfig(
            batch_size=online_buffer_config.batch_size if online_buffer_config else 64
        )
        self.online_buffer = AWRReplayBuffer(
            online_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="online_buffer",
            camera_names=self.camera_names,
            action_horizon=int(awr_config.action_horizon),
            image_size=int(awr_config.image_size),
            augmentation_config=awr_config.augmentation,
            await_discriminator_labels=bool(awr_config.reward_from_discriminator),
        )
        self.demo_buffer = AWRReplayBuffer(
            demo_buffer_config or ReplayBufferConfig(batch_size=self.trainer_config.batch_size),
            name="demo_buffer",
            camera_names=self.camera_names,
            action_horizon=int(awr_config.action_horizon),
            image_size=int(awr_config.image_size),
            augmentation_config=awr_config.augmentation,
            await_discriminator_labels=bool(awr_config.reward_from_discriminator),
        )
        self.core = AWRFlowPolicy(
            model_cfg=self.model_cfg,
            config=awr_config,
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
    ) -> "AWRAgent":
        observation_example = infer_observation_example(
            observation_space=observation_space,
            observation_example=observation_example,
        )
        _ = action_low
        _ = action_high
        action_dim = infer_action_dim(action_space=action_space, sample_action=sample_action, cfg=cfg)
        if not isinstance(observation_example, dict) or "state" not in observation_example:
            raise KeyError("awr requires observation_example to contain a 'state' entry.")
        proprio_dim = int(np.asarray(observation_example["state"]).reshape(-1).shape[0])

        encoder_cfg = cfg_get(cfg, "encoder", None)
        awr_cfg = cfg_get(cfg, "awr", None)
        online_buffer_cfg = cfg_get(cfg, "online_buffer", None)
        demo_buffer_cfg = cfg_get(cfg, "demo_buffer", None)
        trainer_cfg = cfg_get(cfg, "trainer", None)

        encoder_config = EncoderConfig(
            encoder_type=str(cfg_get(encoder_cfg, "encoder_type", "flow-multi")),
            image_keys=tuple(cfg_get(cfg, "camera_names", ())),
            proprio_keys=("state",),
            image_size=int(cfg_get(awr_cfg, "image_size", 128)),
        )
        model_cfg = copy.deepcopy(cfg_get(awr_cfg, "model", {}))
        augmentation_cfg = cfg_get(awr_cfg, "augmentation", None)
        batch_size = int(cfg_get(trainer_cfg, "batch_size", cfg_get(cfg, "batch_size", 64)))
        inference_device = cfg_get(awr_cfg, "inference_device", None)
        if inference_device is not None and str(inference_device).strip().lower() == "auto":
            inference_device = None

        task_name = str(cfg_get(cfg, "task_name", "task"))
        task_prompt_map = cfg_get(awr_cfg, "task_prompt_map", None)
        resolved_device = str(device or cfg_get(awr_cfg, "device", cfg_get(cfg, "device", "cpu")))
        awr_config = AWRConfig(
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_horizon=int(cfg_get(awr_cfg, "action_horizon", 8)),
            execute_horizon=int(cfg_get(awr_cfg, "execute_horizon", 1)),
            image_size=int(cfg_get(awr_cfg, "image_size", 128)),
            actor_learning_rate=float(cfg_get(awr_cfg, "actor_learning_rate", 1e-4)),
            critic_learning_rate=float(cfg_get(awr_cfg, "critic_learning_rate", 3e-4)),
            weight_decay=float(cfg_get(awr_cfg, "weight_decay", 1e-6)),
            grad_clip_norm=float(cfg_get(awr_cfg, "grad_clip_norm", 1.0)),
            lambda_endpoint=float(cfg_get(awr_cfg, "lambda_endpoint", 0.5)),
            lambda_smooth=float(cfg_get(awr_cfg, "lambda_smooth", 0.05)),
            beta=float(cfg_get(awr_cfg, "beta", 3.0)),
            max_adv_weight=float(cfg_get(awr_cfg, "max_adv_weight", 100.0)),
            discount=float(cfg_get(awr_cfg, "discount", 0.99)),
            expectile=float(cfg_get(awr_cfg, "expectile", 0.7)),
            critic_hidden_dims=tuple(cfg_get(awr_cfg, "critic_hidden_dims", (512, 512))),
            margin_scale_floor=float(cfg_get(awr_cfg, "margin_scale_floor", 0.1)),
            success_reward_scale=float(cfg_get(awr_cfg, "success_reward_scale", 1.0)),
            discriminator_reward_scale=float(cfg_get(awr_cfg, "discriminator_reward_scale", 1.0)),
            discriminator_reward_clip=float(cfg_get(awr_cfg, "discriminator_reward_clip", 5.0)),
            n_ode_steps=int(cfg_get(awr_cfg, "n_ode_steps", 8)),
            reward_from_discriminator=bool(cfg_get(awr_cfg, "reward_from_discriminator", True)),
            device=resolved_device,
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
            value_warmup_steps=int(cfg_get(trainer_cfg, "value_warmup_steps", 20_000)),
        )
        return cls(
            observation_example=observation_example,
            encoder_config=encoder_config,
            awr_config=awr_config,
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

    def clone_observation(self, obs):
        return self.core.clone_observation(obs)

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
        horizon = int(self.awr_config.action_horizon)
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
                f"Unable to fit awr normalizers because no valid action_horizon={horizon} windows were found."
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

    def patch_transition_reward(
        self,
        *,
        episode_namespace: str,
        episode_index: int,
        episode_step: int,
        total_reward: float,
        env_reward: float,
        discriminator_reward: float,
        score: float,
        threshold: float,
        normalized_margin: float,
        source: str,
        metadata: dict[str, Any] | None = None,
        patch_demo: bool = True,
    ) -> tuple[bool, bool]:
        online_patched = self.online_buffer.patch_awr_reward(
            episode_namespace=episode_namespace,
            episode_index=episode_index,
            episode_step=episode_step,
            total_reward=total_reward,
            env_reward=env_reward,
            discriminator_reward=discriminator_reward,
            score=score,
            threshold=threshold,
            normalized_margin=normalized_margin,
            source=source,
            metadata=metadata,
        )
        demo_patched = False
        if patch_demo:
            demo_patched = self.demo_buffer.patch_awr_reward(
                episode_namespace=episode_namespace,
                episode_index=episode_index,
                episode_step=episode_step,
                total_reward=total_reward,
                env_reward=env_reward,
                discriminator_reward=discriminator_reward,
                score=score,
                threshold=threshold,
                normalized_margin=normalized_margin,
                source=source,
                metadata=metadata,
            )
        return online_patched, demo_patched

    def get_online_transition_awr_fields(
        self,
        *,
        episode_namespace: str,
        episode_index: int,
        episode_step: int,
    ) -> dict[str, Any] | None:
        return self.online_buffer.get_transition_awr_fields(
            episode_namespace=episode_namespace,
            episode_index=episode_index,
            episode_step=episode_step,
        )

    def sample_demo_actor_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return self.demo_buffer.sample_actor_batch(
            batch_size=batch_size,
            action_mean=self.core.act_mean,
            action_std=self.core.act_std,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.core.device,
            augment=True,
            buffer_role="demo",
        )

    def sample_online_actor_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return self.online_buffer.sample_actor_batch(
            batch_size=batch_size,
            action_mean=self.core.act_mean,
            action_std=self.core.act_std,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.core.device,
            augment=True,
            buffer_role="online",
        )

    def sample_demo_step_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return self.demo_buffer.sample_step_batch(
            batch_size=batch_size,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.core.device,
            augment=True,
            buffer_role="demo",
        )

    def sample_online_step_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        return self.online_buffer.sample_step_batch(
            batch_size=batch_size,
            proprio_mean=self.core.prop_mean,
            proprio_std=self.core.prop_std,
            device=self.core.device,
            augment=True,
            buffer_role="online",
        )

    def sample_mixed_actor_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        if batch_size < 2:
            raise ValueError("AWR mixed actor batch size must be at least 2 for strict 1:1 sampling.")
        if self.demo_buffer.num_valid_sequences() <= 0 or self.online_buffer.num_valid_sequences() <= 0:
            raise RuntimeError("AWR mixed actor updates require both demo and online valid sequences.")
        demo_batch_size = batch_size // 2
        online_batch_size = batch_size - demo_batch_size
        return AWRActorBatch.concat(
            [
                self.sample_demo_actor_batch(batch_size=demo_batch_size),
                self.sample_online_actor_batch(batch_size=online_batch_size),
            ]
        )

    def sample_mixed_step_batch(self, batch_size: int | None = None):
        batch_size = int(batch_size or self.trainer_config.batch_size)
        if batch_size < 2:
            raise ValueError("AWR mixed critic batch size must be at least 2 for strict 1:1 sampling.")
        if self.demo_buffer.num_ready_steps() <= 0 or self.online_buffer.num_ready_steps() <= 0:
            raise RuntimeError("AWR mixed critic updates require both demo and online ready steps.")
        demo_batch_size = batch_size // 2
        online_batch_size = batch_size - demo_batch_size
        return AWRStepBatch.concat(
            [
                self.sample_demo_step_batch(batch_size=demo_batch_size),
                self.sample_online_step_batch(batch_size=online_batch_size),
            ]
        )

    def ready_for_value_warmup(self, batch_size: int | None = None) -> bool:
        _ = batch_size
        return self.online_buffer.num_ready_steps() > 0 and self.has_normalizers()

    def ready_for_update(self, batch_size: int | None = None) -> bool:
        _ = batch_size
        return (
            self.demo_buffer.num_valid_sequences() > 0
            and self.online_buffer.num_valid_sequences() > 0
            and self.demo_buffer.num_ready_steps() > 0
            and self.online_buffer.num_ready_steps() > 0
            and self.has_normalizers()
        )

    def update_value_only(self, *, step_batch=None, batch_size: int | None = None, use_online_only: bool = True) -> dict[str, float]:
        if step_batch is None:
            step_batch = self.sample_online_step_batch(batch_size=batch_size) if use_online_only else self.sample_mixed_step_batch(batch_size=batch_size)
        return self.core.update_value(step_batch)

    def update(self, *, actor_batch=None, step_batch=None, batch_size: int | None = None) -> dict[str, float]:
        if step_batch is None:
            step_batch = self.sample_mixed_step_batch(batch_size=batch_size)
        if actor_batch is None:
            actor_batch = self.sample_mixed_actor_batch(batch_size=batch_size)
        return self.core.update(step_batch=step_batch, actor_batch=actor_batch)

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
            "awr_config": asdict(self.awr_config),
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

    def build_qv_cache_payload(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        trainer_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "encoder_config": asdict(self.encoder_config),
            "awr_config": asdict(self.awr_config),
            "trainer_config": asdict(self.trainer_config),
            "model_cfg": copy.deepcopy(self.model_cfg),
            "camera_names": list(self.camera_names),
            "task_name": self.task_name,
            "language_instruction": self.language_instruction,
            "qv_core": self.core.qv_state_dict(),
        }
        if metadata is not None:
            payload["metadata"] = copy.deepcopy(metadata)
        if trainer_state is not None:
            payload["trainer_state"] = copy.deepcopy(trainer_state)
        return payload

    def load_qv_cache_payload(self, payload: dict[str, Any], *, load_optimizers: bool = True) -> dict[str, Any]:
        self.core.load_qv_state_dict(payload["qv_core"], load_optimizers=load_optimizers)
        return {
            "metadata": copy.deepcopy(payload.get("metadata", {})),
            "trainer_state": copy.deepcopy(payload.get("trainer_state", {})),
        }

    def load_flow_policy_checkpoint(self, path: str | Path, *, task_name: str | None = None) -> dict[str, Any]:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load flow policy checkpoint from {path}: {exc}") from exc
        model_state = payload.get("ema_model", payload.get("model"))
        if model_state is None:
            raise KeyError(f"Checkpoint {path} is missing both 'ema_model' and 'model'.")
        self.core.load_actor_model_state(model_state, strict=True)
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
