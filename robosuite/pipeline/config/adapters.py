"""Adapters from the public batch-online config to internal stage contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf


def _plain(node: Any) -> dict[str, Any]:
    value = OmegaConf.to_container(node, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Batch-online configuration must resolve to a mapping.")
    return value


def _flow_config(cfg: DictConfig) -> dict[str, Any]:
    flow = deepcopy(_plain(cfg.task.policy.flow))
    flow["n_ode_steps"] = flow.pop("ode_steps")
    flow["lambda_endpoint"] = flow.pop("endpoint_loss_weight")
    flow["lambda_smooth"] = flow.pop("smoothness_loss_weight")
    head = flow["model"]["head"]
    head["cond_dim"] = head.pop("condition_dim")
    head["channel_mults"] = head.pop("channel_multipliers")
    cross = head["cross_attention"]
    cross["attn_dim"] = cross.pop("attention_dim")
    cross["mid"] = cross.pop("middle")
    return flow


def _environment_config(cfg: DictConfig) -> dict[str, Any]:
    env = cfg.environment
    return {
        "environment": str(env.name),
        "robots": list(env.robots),
        "config": str(env.configuration),
        "controller": env.controller,
        "renderer": str(env.renderer),
        "render_camera": str(env.render_camera),
        "camera_names": list(env.camera_names),
        "proprio_keys": list(env.proprio_keys),
        "img_height": int(env.image_height),
        "img_width": int(env.image_width),
        "control_freq": int(env.control_frequency),
        "horizon": env.horizon,
    }


def _discriminator_runtime(
    cfg: DictConfig,
    *,
    checkpoint: str,
    encoder_checkpoint: str,
    learner_device: str,
    inference_device: str,
    intervene_env: bool,
    hud_enabled: bool,
) -> dict[str, Any]:
    disc = cfg.task.discriminator
    return {
        "enabled": True,
        "task_name": str(cfg.environment.name),
        "checkpoint": str(checkpoint),
        "encoder_ckpt": str(encoder_checkpoint),
        "learner_device": str(learner_device),
        "camera_to_view": _plain(disc.camera_to_view),
        "inference": {
            "device": str(inference_device),
            "fps": float(cfg.collection.discriminator_fps),
            "intervene_env": bool(intervene_env),
        },
        "pause": {
            "consecutive_fail_frames": int(disc.pause.consecutive_failure_frames),
            "require_safe_to_rearm": bool(disc.pause.require_safe_to_rearm),
        },
        "hud": {"enabled": bool(hud_enabled)},
    }


def _algorithm_config(
    cfg: DictConfig,
    *,
    policy_device: str,
    inference_device: str,
    discriminator_checkpoint: str,
    encoder_checkpoint: str,
) -> dict[str, Any]:
    flow = _flow_config(cfg)
    flow.update({"device": policy_device, "inference_device": inference_device})
    return {
        "type": "dipole",
        "camera_names": list(cfg.environment.camera_names),
        "task_name": str(cfg.environment.name),
        "encoder": {"encoder_type": "flow-multi"},
        "flow": flow,
        "dipole": {
            "condition_scale": 1.0,
            "beta": float(cfg.task.policy.branch_weight.beta),
            "k": float(cfg.task.policy.branch_weight.k),
            "guidance_omega": 2.0,
            "g_mode": str(cfg.task.policy.dipole.g_mode),
        },
        "replay_buffer": {"capacity": int(cfg.task.policy.replay_capacity)},
        "trainer": {
            "batch_size": int(cfg.task.policy.batch_size),
        },
        "discriminator": _discriminator_runtime(
            cfg,
            checkpoint=discriminator_checkpoint,
            encoder_checkpoint=encoder_checkpoint,
            learner_device=policy_device,
            inference_device=str(cfg.cuda.collection.discriminator_device),
            intervene_env=bool(cfg.collection.discriminator_intervene_env),
            hud_enabled=bool(cfg.task.discriminator.hud_enabled),
        ),
        "advantage_g_provider": {
            "alpha": float(cfg.task.policy.advantage_g.alpha),
            "beta": float(cfg.task.policy.advantage_g.discriminator_beta),
        },
    }


def _vast_config(cfg: DictConfig) -> dict[str, Any]:
    vast = cfg.task.vast
    return {
        "vast_v_mode": str(vast.value_mode),
        "vast_max_k": int(vast.max_k),
        "vast_comp_coef": float(vast.composition_coefficient),
        "vast_sampling_seed": int(vast.sampling_seed),
        "reward_mode": str(vast.reward_mode),
        "action_horizon": int(cfg.task.policy.flow.action_horizon),
        "discount": float(vast.discount),
        "expectile_tau": float(vast.expectile_tau),
        "v_ensemble_size": int(vast.value_ensemble_size),
        "g_lr": float(vast.g_learning_rate),
        "v_lr": float(vast.v_learning_rate),
        "target_polyak": float(vast.target_polyak),
        "n_step_aggregate": bool(vast.n_step_aggregate),
        "hidden_dims": list(vast.hidden_dims),
        "state_proj_dim": int(vast.state_projection_dim),
        "proprio_proj_dim": int(vast.proprio_projection_dim),
        "proj_activation": str(vast.projection_activation),
        "grad_clip_norm": float(vast.grad_clip_norm),
        "weight_decay": float(vast.weight_decay),
        "device": str(cfg.cuda.training_device),
        "output_reward_coef": float(vast.output_reward_coefficient),
        "disc_reward_coef": float(vast.discriminator_reward_coefficient),
        "update_freq": int(vast.update_frequency),
    }


def vast_warmup_stage_config(cfg: DictConfig) -> DictConfig:
    """Adapt the public config for the retained standalone VAST warmup utility."""
    inputs = cfg.task.inputs
    algorithm = _algorithm_config(
        cfg,
        policy_device=str(cfg.cuda.training_device),
        inference_device=str(cfg.cuda.training_device),
        discriminator_checkpoint=str(inputs.parent_discriminator_checkpoint),
        encoder_checkpoint=str(inputs.dynamics_encoder_checkpoint),
    )
    raw_warmup = OmegaConf.select(cfg, "warmup", default={})
    warmup = (
        OmegaConf.to_container(raw_warmup, resolve=True)
        if raw_warmup is not None
        else {}
    )
    if not isinstance(warmup, dict):
        raise TypeError("warmup must resolve to a mapping.")
    algorithm["vast"] = {
        "enabled": True,
        "warmup_joint_steps": int(warmup.get("num_steps", cfg.task.vast.num_steps)),
        "config": _vast_config(cfg),
    }
    expert_path = Path(str(inputs.expert_data))
    return OmegaConf.create(
        {
            "seed": int(cfg.seed),
            "env": _environment_config(cfg),
            "algorithm": algorithm,
            "runtime": {
                "init_checkpoint": str(inputs.base_policy_checkpoint),
                "use_init_checkpoint_camera_names": True,
                "use_init_checkpoint_model": True,
            },
            "data": {
                "demo_root": str(expert_path.parent.parent),
                "task_name": str(cfg.environment.name),
                "demo_paths": [],
                "demo_split": "expert",
                "num_trajectories": None,
            },
            "warmup": warmup,
            "logging": {"use_tensorboard": True},
        }
    )


def collection_stage_config(
    cfg: DictConfig,
    *,
    policy_checkpoint: str | None = None,
    discriminator_checkpoint: str | None = None,
    encoder_checkpoint: str | None = None,
    output_path: str | None = None,
) -> DictConfig:
    inputs = cfg.task.inputs
    policy_checkpoint = str(policy_checkpoint or inputs.base_policy_checkpoint)
    discriminator_checkpoint = str(
        discriminator_checkpoint or inputs.parent_discriminator_checkpoint
    )
    encoder_checkpoint = str(
        encoder_checkpoint or inputs.dynamics_encoder_checkpoint
    )
    algorithm = _algorithm_config(
        cfg,
        policy_device=str(cfg.cuda.collection.policy_device),
        inference_device=str(cfg.cuda.collection.policy_device),
        discriminator_checkpoint=discriminator_checkpoint,
        encoder_checkpoint=encoder_checkpoint,
    )
    collection = cfg.collection
    output = output_path or "offline_episodes.pt"
    return OmegaConf.create(
        {
            "seed": int(cfg.seed),
            "env": _environment_config(cfg),
            "algorithm": algorithm,
            "runtime": {
                "interactive": bool(collection.interactive),
                "viewer_enabled": bool(collection.viewer_enabled),
                "unthrottled": False,
                "viewer_backend": str(collection.viewer_backend),
                "viewer_async": bool(collection.viewer_async),
                "viewer_startup_delay": float(collection.viewer_startup_delay),
                "viewer_reset_warmup_frames": int(collection.viewer_reset_warmup_frames),
                "visualize_gripper_markers": bool(collection.visualize_gripper_markers),
                "control_fps": int(cfg.environment.control_frequency),
                "render_fps": int(cfg.environment.control_frequency),
                "image_obs_fps": int(collection.image_obs_fps),
                "spacemouse_fps": int(collection.spacemouse_fps),
                "policy_fps": int(collection.policy_fps),
                "fps_log_interval": float(collection.fps_log_interval),
                "eval_episode_max_steps": int(collection.episode_max_steps),
                "init_checkpoint": policy_checkpoint,
                "use_init_checkpoint_camera_names": True,
                "use_init_checkpoint_model": True,
                "episode_pause_sec": float(collection.episode_pause_seconds),
            },
            "intervention": {
                "enabled": bool(collection.intervention_enabled),
                "device": str(collection.intervention.device),
                "pos_sensitivity": float(collection.intervention.position_sensitivity),
                "rot_sensitivity": float(collection.intervention.rotation_sensitivity),
                "reverse_xy": bool(collection.intervention.reverse_xy),
                "goal_update_mode": str(collection.intervention.goal_update_mode),
                "device_reset_as_episode_reset": bool(
                    collection.intervention.device_reset_as_episode_reset
                ),
            },
            "data": {"demo_root": "data", "num_trajectories": None},
            "offline_collect": {
                "num_episodes": int(collection.num_episodes),
                "episode_max_steps": int(collection.episode_max_steps),
                "deterministic": bool(collection.deterministic),
                "save_every_episode": bool(collection.save_every_episode),
                "save_interval_episodes": 1,
                "resume": True,
                "output_dir": str(Path(output).parent),
                "output_file": str(Path(output).name),
            },
            "logging": {"use_tensorboard": False},
        }
    )


def discriminator_stage_config(
    cfg: DictConfig,
    *,
    parent_checkpoint: str | None = None,
    encoder_checkpoint: str | None = None,
    pretrain_dir: str | None = None,
    episodes_paths: Sequence[str] | None = None,
    run_dir: str | None = None,
    round_index: int = 0,
    feature_cache_dir: str | None = None,
) -> DictConfig:
    inputs = cfg.task.inputs
    episodes = list(episodes_paths or [])
    if not episodes:
        raise ValueError("discriminator stage requires at least one episodes path.")
    disc = cfg.task.discriminator
    objective = {
        "steps_per_epoch": disc.steps_per_epoch,
        "logit_normalization": _plain(disc.logit_normalization),
        "quadratic_logit_cap": _plain(disc.quadratic_logit_cap),
        "terms": _plain(disc.losses),
    }
    return OmegaConf.create(
        {
            "seed": int(OmegaConf.select(cfg, "task.discriminator.seed", default=0)),
            "env": {"environment": str(cfg.environment.name)},
            "algorithm": {
                "discriminator": _discriminator_runtime(
                    cfg,
                    checkpoint=str(parent_checkpoint or inputs.parent_discriminator_checkpoint),
                    encoder_checkpoint=str(encoder_checkpoint or inputs.dynamics_encoder_checkpoint),
                    learner_device=str(cfg.cuda.training_device),
                    inference_device=str(cfg.cuda.training_device),
                    intervene_env=False,
                    hud_enabled=False,
                )
            },
            "offline": {
                "discriminator_finetune": {
                    "parent_checkpoint": str(parent_checkpoint or inputs.parent_discriminator_checkpoint),
                    "episodes_path": episodes[-1],
                    "episodes_paths": episodes,
                    "pretrain_dir": str(pretrain_dir or inputs.discriminator_pretrain_data),
                    "epochs": int(disc.epochs),
                    "scheduler_horizon_epochs": int(disc.scheduler_horizon_epochs),
                    "lr": float(disc.learning_rate),
                    "weight_decay": float(disc.weight_decay),
                    "encode_batch_size": int(disc.encode_batch_size),
                    "log_interval": int(disc.log_interval),
                    "run_root": str(Path(run_dir or ".").parent),
                    "run_dir": run_dir,
                    "round_index": int(round_index),
                    "history_mix_beta": float(cfg.task.history_mix_beta),
                    "action_horizon": int(cfg.task.policy.flow.action_horizon),
                    "feature_cache_dir": feature_cache_dir,
                    "gt_negative": _plain(disc.gt_negative_window),
                    "objective": objective,
                }
            },
            "logging": {
                "use_tensorboard": bool(cfg.tensorboard.enabled),
                "tensorboard_dir": str(cfg.tensorboard.directory),
            },
        }
    )


def offline_stage_config(
    cfg: DictConfig,
    *,
    stage: str,
    policy_checkpoint: str | None = None,
    discriminator_checkpoint: str | None = None,
    encoder_checkpoint: str | None = None,
    vast_checkpoint: str | None = None,
    episodes_paths: Sequence[str] | None = None,
    expert_data: str | None = None,
    vast_warmup_transitions: str | None = None,
    run_dir: str | None = None,
) -> DictConfig:
    if stage not in {"all", "vast", "policy"}:
        raise ValueError(f"Unsupported offline stage {stage!r}.")
    inputs = cfg.task.inputs
    episodes = list(episodes_paths or [])
    if not episodes:
        raise ValueError("offline stage requires at least one episodes path.")
    policy_checkpoint = str(policy_checkpoint or inputs.base_policy_checkpoint)
    discriminator_checkpoint = str(
        discriminator_checkpoint or inputs.parent_discriminator_checkpoint
    )
    encoder_checkpoint = str(
        encoder_checkpoint or inputs.dynamics_encoder_checkpoint
    )
    vast_checkpoint = str(vast_checkpoint or inputs.initial_vast_checkpoint)
    algorithm = _algorithm_config(
        cfg,
        policy_device=str(cfg.cuda.training_device),
        inference_device=str(cfg.cuda.training_device),
        discriminator_checkpoint=discriminator_checkpoint,
        encoder_checkpoint=encoder_checkpoint,
    )
    algorithm["vast"] = {
        "enabled": True,
        "warmup_ckpt": vast_checkpoint,
        "config": _vast_config(cfg),
    }
    offline = {
        "data_root": str(Path(expert_data or inputs.expert_data).parent.parent),
        "episodes_path": episodes[-1],
        "episodes_paths": episodes,
        "pretrain_data_path": str(expert_data or inputs.expert_data),
        "max_pretrain_trajectories": None,
        "vast_warmup_transitions_path": str(
            vast_warmup_transitions or inputs.vast_warmup_transitions
        ),
        "use_online_success": bool(cfg.task.policy.use_online_success),
        "reward_success": float(cfg.task.policy.reward_success),
        "reward_fail": float(cfg.task.policy.reward_failure),
        "include_policy_action_neg": bool(cfg.task.policy.include_policy_action_negative),
        "num_train_steps": int(cfg.task.policy.num_steps),
        "log_interval": int(cfg.task.policy.log_interval),
        "plot_log_interval": None,
        "checkpoint_interval": int(cfg.task.policy.checkpoint_interval),
        "preencode_batch_size": int(cfg.task.policy.preencode_batch_size),
        "run_root": str(Path(run_dir or ".").parent),
        "run_dir": run_dir,
        "execution_stage": stage,
        "skip_rl": stage == "policy",
        "vast_finetuned_path": vast_checkpoint if stage == "policy" else None,
        "vast_finetune": {
            "num_steps": int(cfg.task.vast.num_steps),
            "batch_size": int(cfg.task.vast.batch_size),
            "preencode_cache": bool(cfg.task.vast.preencode_cache),
            "relabel_disc_reward": bool(cfg.task.vast.relabel_discriminator_reward),
        },
        "advantage": _plain(cfg.task.policy.advantage),
        "branch_weight": _plain(cfg.task.policy.branch_weight),
    }
    return OmegaConf.create(
        {
            "seed": int(cfg.seed),
            "env": _environment_config(cfg),
            "algorithm": algorithm,
            "runtime": {
                "interactive": False,
                "viewer_enabled": False,
                "init_checkpoint": policy_checkpoint,
                "use_init_checkpoint_camera_names": True,
                "use_init_checkpoint_model": True,
            },
            "offline": offline,
            "logging": {
                "use_tensorboard": bool(cfg.tensorboard.enabled),
                "tensorboard_dir": str(cfg.tensorboard.directory),
            },
        }
    )


__all__ = [
    "collection_stage_config",
    "discriminator_stage_config",
    "offline_stage_config",
    "vast_warmup_stage_config",
]
