from __future__ import annotations

import builtins
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.awr import AWRTrainer
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.train_awr import (
    _build_transition_loader,
    _load_split_transitions,
    build_qv_cache_metadata,
    maybe_load_qv_cache,
    resolve_algorithm_devices,
    resolve_qv_cache_path,
    save_qv_cache,
)
from robosuite.pipeline.train_flow_dagger import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    load_init_checkpoint_payload,
    maybe_set_seed,
    reset_flow_policy_observation,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import now_readable, resolve_camera_names
from robosuite.pipeline.utils.io import resolve_task_demo_paths
from robosuite.pipeline.utils.train_utils import resolve_demo_task_name

print = partial(builtins.print, flush=True)


@hydra.main(version_base="1.2", config_path="./config", config_name="train_awr")
def main(cfg: DictConfig) -> None:
    maybe_set_seed(getattr(cfg, "seed", None))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None:
        raise FileNotFoundError("AWR Q/V cache build requires runtime.init_checkpoint to be set.")

    task_name = str(cfg.env.environment)
    task_data_name = resolve_demo_task_name(cfg)
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    build_dir = output_root / "_qv_cache_build" / f"{task_data_name}_{now_readable()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    qv_cache_path = resolve_qv_cache_path(cfg, task_data_name=task_data_name)

    requested_camera_names = resolve_camera_names(cfg)
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    cfg.algorithm.camera_names = list(policy_camera_names)
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.awr, "camera_aliases", {}) or {}).items()
    }

    runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer=str(cfg.env.renderer),
    )
    env = build_robosuite_env(runtime_cfg)
    flow_proprio_extractor = None
    try:
        flow_proprio_extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
        initial_obs, _ = reset_flow_policy_observation(
            env,
            preserve_mjviewer=False,
            extractor=flow_proprio_extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
        )
        action_low, action_high = env.action_spec
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        if not isinstance(algorithm_cfg, dict):
            raise TypeError("Expected cfg.algorithm to resolve to a dictionary.")
        algorithm_cfg["camera_names"] = list(policy_camera_names)
        algorithm_cfg["task_name"] = task_name
        awr_cfg = algorithm_cfg.setdefault("awr", {})
        awr_cfg.setdefault("image_size", int(cfg.env.img_height))
        awr_cfg["reward_from_discriminator"] = bool(getattr(cfg.discriminator, "enabled", False))
        if bool(getattr(cfg.runtime, "use_init_checkpoint_model", True)) and init_payload is not None:
            if "model_cfg" in init_payload:
                awr_cfg["model"] = init_payload["model_cfg"]
            if "task_prompt_map" in init_payload:
                awr_cfg["task_prompt_map"] = init_payload["task_prompt_map"]
            if init_payload.get("act_mean") is not None:
                awr_cfg["action_horizon"] = int(np.asarray(init_payload["act_mean"]).shape[0])
                awr_cfg.setdefault("execute_horizon", 1)
        model_cfg = awr_cfg.setdefault("model", {})
        image_encoder_cfg = model_cfg.get("image_encoder", None)
        if isinstance(image_encoder_cfg, dict) and image_encoder_cfg.get("pretrained_path"):
            image_encoder_cfg["pretrained_path"] = to_absolute_path(str(image_encoder_cfg["pretrained_path"]))
        language_encoder_cfg = model_cfg.get("language_encoder", None)
        if isinstance(language_encoder_cfg, dict) and language_encoder_cfg.get("pretrained_name"):
            language_encoder_cfg["pretrained_name"] = to_absolute_path(str(language_encoder_cfg["pretrained_name"]))
        learner_device, inference_device = resolve_algorithm_devices(algorithm_cfg)
        print(f"[qv_cache_build] task={task_data_name} learner_device={learner_device} inference_device={inference_device}")

        agent = build_algorithm(
            algorithm_cfg,
            observation_example=initial_obs,
            sample_action=np.zeros_like(action_low, dtype=np.float32),
            action_low=action_low,
            action_high=action_high,
        )
        agent.load_flow_policy_checkpoint(init_checkpoint, task_name=task_name)
        trainer = AWRTrainer(agent)

        qv_cache_metadata = build_qv_cache_metadata(
            cfg,
            task_name=task_name,
            task_data_name=task_data_name,
            policy_camera_names=policy_camera_names,
            init_checkpoint=init_checkpoint,
        )
        qv_cache_cfg = getattr(cfg.runtime, "qv_cache", None)
        qv_cache_enabled = bool(getattr(qv_cache_cfg, "enabled", True)) if qv_cache_cfg is not None else True
        qv_cache_force_rebuild = bool(getattr(qv_cache_cfg, "force_rebuild", False)) if qv_cache_cfg is not None else False
        qv_cache_save_enabled = bool(getattr(qv_cache_cfg, "save", True)) if qv_cache_cfg is not None else True

        if qv_cache_enabled and (not qv_cache_force_rebuild):
            if maybe_load_qv_cache(
                cfg=cfg,
                agent=agent,
                trainer=trainer,
                cache_path=qv_cache_path,
                expected_metadata=qv_cache_metadata,
            ):
                print(f"[qv_cache_build] reused_existing_cache path={qv_cache_path}")
                return

        success_paths = resolve_task_demo_paths(
            task_data_name,
            data_root=to_absolute_path(str(cfg.data.demo_root)),
            split="success_rollout",
        )
        fail_paths = resolve_task_demo_paths(
            task_data_name,
            data_root=to_absolute_path(str(cfg.data.demo_root)),
            split="fail_rollout",
        )
        if not success_paths or not fail_paths:
            raise FileNotFoundError(
                f"AWR value warmup requires both success_rollout and fail_rollout. "
                f"task='{task_data_name}', success_files={len(success_paths)}, fail_files={len(fail_paths)}"
            )

        proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
        cache_key_parts = [
            f"h{int(cfg.env.img_height)}",
            f"w{int(cfg.env.img_width)}",
            f"cams-{'_'.join(policy_camera_names)}",
            f"state-{'_'.join(proprio_keys) if proprio_keys else 'auto'}",
        ]
        transition_loader = _build_transition_loader(
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            state_extractor=flow_proprio_extractor,
        )

        expert_transitions = []
        if not agent.has_normalizers():
            expert_paths = resolve_task_demo_paths(
                task_data_name,
                data_root=to_absolute_path(str(cfg.data.demo_root)),
                split="expert",
            )
            if not expert_paths:
                raise FileNotFoundError(
                    f"No expert demos found for task '{task_data_name}', but expert data is required to fit normalizers."
                )
            print(f"[qv_cache_build] loading_expert_for_normalizers task={task_data_name}")
            expert_transitions = _load_split_transitions(
                split_name="offline_demo",
                demo_paths=expert_paths,
                max_num_trajectories=cfg.data.expert_num_trajectories,
                output_root=output_root,
                checkpoint_dir=build_dir,
                loader=transition_loader,
                cache_key_parts=cache_key_parts,
            )
        print(f"[qv_cache_build] loading_success_rollout task={task_data_name}")
        success_transitions = _load_split_transitions(
            split_name="success_rollout",
            demo_paths=success_paths,
            max_num_trajectories=cfg.data.success_num_trajectories,
            output_root=output_root,
            checkpoint_dir=build_dir,
            loader=transition_loader,
            cache_key_parts=cache_key_parts,
        )
        print(f"[qv_cache_build] loading_fail_rollout task={task_data_name}")
        fail_transitions = _load_split_transitions(
            split_name="fail_rollout",
            demo_paths=fail_paths,
            max_num_trajectories=cfg.data.fail_num_trajectories,
            output_root=output_root,
            checkpoint_dir=build_dir,
            loader=transition_loader,
            cache_key_parts=cache_key_parts,
        )
        if len(success_transitions) == 0 or len(fail_transitions) == 0:
            raise RuntimeError("Success/fail rollout loading completed but produced zero transitions.")

        if not agent.has_normalizers():
            if len(expert_transitions) == 0:
                raise RuntimeError("Expert loading completed but produced zero transitions.")
            agent.fit_normalizers_from_transitions(expert_transitions)
            print("[qv_cache_build] fitted_normalizers_from_expert=true")
        trainer.bootstrap_online_buffer(
            success_transitions,
            demo_source="success_rollout",
            episode_namespace="success_rollout",
        )
        trainer.bootstrap_online_buffer(
            fail_transitions,
            demo_source="fail_rollout",
            episode_namespace="fail_rollout",
        )
        print(
            f"[qv_cache_build] loaded_transitions "
            f"expert={len(expert_transitions)} success={len(success_transitions)} fail={len(fail_transitions)}"
        )

        requested_value_warmup_steps = max(0, int(cfg.algorithm.trainer.value_warmup_steps))
        completed_warmup_steps = int(trainer.total_value_warmup_updates)
        remaining_warmup_steps = max(0, requested_value_warmup_steps - completed_warmup_steps)
        print(
            f"[qv_cache_build] remaining_value_steps={remaining_warmup_steps} "
            f"(completed={completed_warmup_steps}, target={requested_value_warmup_steps})"
        )
        for local_step in range(remaining_warmup_steps):
            metrics = trainer.pretrain_value(1)[-1]
            absolute_warmup_step = completed_warmup_steps + local_step + 1
            if absolute_warmup_step % max(1, int(cfg.logging.log_interval)) == 0:
                print(
                    "[qv_cache_build] "
                    f"step={absolute_warmup_step} q={metrics.get('q_loss', float('nan')):.4f} "
                    f"v={metrics.get('value_loss', float('nan')):.4f}"
                )

        if qv_cache_enabled and qv_cache_save_enabled and int(trainer.total_value_warmup_updates) > 0:
            save_qv_cache(
                agent=agent,
                trainer=trainer,
                cache_path=qv_cache_path,
                metadata=qv_cache_metadata,
            )
        print(f"[qv_cache_build] done task={task_data_name} cache={qv_cache_path}")
    finally:
        try:
            env.close()
        finally:
            if flow_proprio_extractor is not None:
                flow_proprio_extractor.close()


if __name__ == "__main__":
    main()
