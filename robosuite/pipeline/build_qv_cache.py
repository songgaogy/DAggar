"""Build the PickPlaceCereal AWR Q/V initialization cache."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.pipeline.bootstrap import (
    build_agent_config,
    build_qv_cache_metadata,
    resolve_qv_cache_path,
    write_qv_cache,
)
from robosuite.pipeline.src.awr import AWRAgent, AWRTrainer
from robosuite.pipeline.src.data import load_demos
from robosuite.pipeline.src.environment import (
    bind_proprio_extractor,
    build_robosuite_env,
    build_runtime_config,
    flow_checkpoint_settings,
    load_flow_checkpoint,
    load_task_metadata,
    reset_policy_observation,
)
from robosuite.pipeline.utils import require_cuda, set_seed


def _run(cfg: DictConfig, resources: ExitStack) -> None:
    require_cuda()
    set_seed(int(cfg.seed))
    init_checkpoint, payload = load_flow_checkpoint(cfg.checkpoint.init_path)
    if init_checkpoint is None or payload is None:
        raise FileNotFoundError("checkpoint.init_path must reference a flow checkpoint.")
    settings = flow_checkpoint_settings(payload, str(cfg.task.name))
    env_metadata = load_task_metadata(payload, str(cfg.task.name))
    if env_metadata is None:
        raise KeyError(f"Flow checkpoint has no environment metadata for {cfg.task.name}.")
    requested_cameras = [str(name) for name in cfg.env.camera_names]
    checkpoint_cameras = [str(name) for name in settings.get("camera_names", [])]
    camera_names = (
        checkpoint_cameras
        if bool(cfg.checkpoint.use_init_camera_names) and checkpoint_cameras
        else requested_cameras
    )
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.awr, "camera_aliases", {}) or {}).items()
    }
    env = build_robosuite_env(
        build_runtime_config(
            env_metadata,
            camera_names=camera_names,
            image_height=int(cfg.env.img_height),
            image_width=int(cfg.env.img_width),
            control_freq=int(cfg.env.control_freq),
            horizon=int(cfg.env.horizon),
            interactive=False,
        )
    )
    resources.callback(env.close)
    extractor = bind_proprio_extractor(env, env_metadata)
    resources.callback(extractor.close)
    observation, _ = reset_policy_observation(
        env,
        preserve_mjviewer=False,
        extractor=extractor,
        camera_names=camera_names,
        camera_aliases=camera_aliases,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
    )
    action_low, action_high = (
        np.asarray(value, dtype=np.float32) for value in env.action_spec
    )
    agent = AWRAgent.from_config(
        build_agent_config(cfg, camera_names=camera_names, flow_settings=settings),
        observation_example=observation,
        action_low=action_low,
        action_high=action_high,
    )
    agent.load_flow_policy_checkpoint(init_checkpoint, task_name=str(cfg.task.name))
    trainer = AWRTrainer(agent)
    demos = load_demos(
        data_root=to_absolute_path(str(cfg.data.demo_root)),
        task_name=str(cfg.data.task_name),
        camera_names=camera_names,
        camera_aliases=camera_aliases,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
        state_extractor=extractor,
        control_freq=int(cfg.env.control_freq),
        horizon=int(cfg.env.horizon),
        trajectory_limits={
            "expert": int(cfg.data.expert_num_trajectories),
            "success_rollout": int(cfg.data.success_num_trajectories),
            "fail_rollout": int(cfg.data.fail_num_trajectories),
        },
        split_directories={
            "expert": to_absolute_path(str(cfg.data.expert_dir)),
            "success_rollout": to_absolute_path(str(cfg.data.success_dir)),
            "fail_rollout": to_absolute_path(str(cfg.data.fail_dir)),
        },
        cache_dir=(
            Path(to_absolute_path(str(cfg.logging.output_root)))
            / str(cfg.task.name)
            / "_demo_cache"
        ),
        seed=int(cfg.seed),
    )
    if not all(demos[split] for split in ("expert", "success_rollout", "fail_rollout")):
        raise RuntimeError("Q/V cache construction requires expert, success, and fail demos.")
    trainer.bootstrap_demo_buffer(demos["expert"])
    trainer.bootstrap_online_buffer(
        demos["success_rollout"],
        demo_source="success_rollout",
        episode_namespace="success_rollout",
    )
    trainer.bootstrap_online_buffer(
        demos["fail_rollout"],
        demo_source="fail_rollout",
        episode_namespace="fail_rollout",
    )
    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(demos["expert"])
    target = int(cfg.trainer.value_warmup_steps)
    for step in range(target):
        metrics = trainer.pretrain_value(1)[0]
        if (step + 1) % int(cfg.logging.log_interval_updates) == 0:
            print(
                f"[warmup] step={step + 1}/{target} "
                f"q={metrics.get('q_loss', float('nan')):.4f} "
                f"v={metrics.get('value_loss', float('nan')):.4f}"
            )
    path = resolve_qv_cache_path(cfg)
    write_qv_cache(
        agent,
        trainer,
        path=path,
        metadata=build_qv_cache_metadata(
            cfg,
            camera_names=camera_names,
            init_checkpoint=init_checkpoint,
            agent=agent,
        ),
    )


@hydra.main(version_base="1.3", config_path="./config", config_name="overall")
def main(cfg: DictConfig) -> None:
    with ExitStack() as resources:
        _run(cfg, resources)


if __name__ == "__main__":
    main()
