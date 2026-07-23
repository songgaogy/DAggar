"""Evaluate a new-format AWR checkpoint in a headless robosuite environment."""

from __future__ import annotations

import json
from datetime import datetime
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
import imageio.v2 as imageio
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.bootstrap import build_agent_config
from robosuite.pipeline.src.awr import AWRAgent
from robosuite.pipeline.src.environment import (
    bind_proprio_extractor,
    build_policy_observation,
    build_robosuite_env,
    build_runtime_config,
    flow_checkpoint_settings,
    load_flow_checkpoint,
    load_task_metadata,
    reset_policy_observation,
    sparse_success_reward,
)
from robosuite.pipeline.utils import require_cuda, resolve_path, set_seed


def _frame(env, *, camera: str, size: int = 512) -> np.ndarray:
    image = env.sim.render(height=size, width=size, camera_name=camera)
    return np.ascontiguousarray(np.flipud(np.asarray(image, dtype=np.uint8)))


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _run(cfg: DictConfig, resources: ExitStack) -> None:
    require_cuda()
    set_seed(int(cfg.evaluation.seed))
    checkpoint = resolve_path(cfg.evaluation.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("Set evaluation.checkpoint to a new-format AWR checkpoint.")

    init_checkpoint, init_payload = load_flow_checkpoint(cfg.checkpoint.init_path)
    if init_checkpoint is None or init_payload is None:
        raise FileNotFoundError("checkpoint.init_path must reference the flow initialization checkpoint.")
    settings = flow_checkpoint_settings(init_payload, str(cfg.task.name))
    env_metadata = load_task_metadata(init_payload, str(cfg.task.name))
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

    runtime_config = build_runtime_config(
        env_metadata,
        camera_names=camera_names,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
        control_freq=int(cfg.env.control_freq),
        horizon=int(cfg.evaluation.max_episode_steps),
        interactive=False,
    )
    env = build_robosuite_env(runtime_config)
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
    agent.load_checkpoint(checkpoint, load_buffers=False)
    agent.reset_policy_state()

    output_dir = (
        Path(to_absolute_path(str(cfg.logging.output_root)))
        / str(cfg.task.name)
        / f"eval_{checkpoint.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / str(cfg.logging.resolved_config_filename))

    episodes: list[dict[str, Any]] = []
    for episode_index in range(int(cfg.evaluation.num_episodes)):
        if episode_index > 0:
            observation, _ = reset_policy_observation(
                env,
                preserve_mjviewer=False,
                extractor=extractor,
                camera_names=camera_names,
                camera_aliases=camera_aliases,
                image_height=int(cfg.env.img_height),
                image_width=int(cfg.env.img_width),
            )
            agent.reset_policy_state()
        frames: list[np.ndarray] = []
        episode_return = 0.0
        success = False
        length = 0
        for length in range(1, int(cfg.evaluation.max_episode_steps) + 1):
            if bool(cfg.evaluation.save_video):
                frames.append(_frame(env, camera=str(cfg.evaluation.video_camera)))
            action = agent.select_action(
                observation,
                deterministic=bool(cfg.evaluation.deterministic),
            )
            step_result = env.step(np.asarray(action, dtype=np.float32))
            if len(step_result) == 5:
                raw_next_obs, _, terminated, truncated, info = step_result
            else:
                raw_next_obs, _, terminated, info = step_result
                truncated = False
            reward, success = sparse_success_reward(
                env,
                info if isinstance(info, dict) else None,
            )
            episode_return += float(reward)
            observation = build_policy_observation(
                env,
                extractor=extractor,
                camera_names=camera_names,
                camera_aliases=camera_aliases,
                image_height=int(cfg.env.img_height),
                image_width=int(cfg.env.img_width),
                raw_observation=raw_next_obs,
            )
            if bool(terminated or truncated or success):
                break
        if bool(cfg.evaluation.save_video):
            frames.append(_frame(env, camera=str(cfg.evaluation.video_camera)))
            _write_video(
                output_dir / "videos" / f"episode_{episode_index:04d}.mp4",
                frames,
                int(cfg.evaluation.video_fps),
            )
        result = {
            "episode": episode_index,
            "return": episode_return,
            "length": length,
            "success": bool(success),
        }
        episodes.append(result)
        print(
            f"[eval] episode={episode_index + 1}/{cfg.evaluation.num_episodes} "
            f"success={int(success)} length={length} return={episode_return:.1f}"
        )

    successes = sum(int(item["success"]) for item in episodes)
    summary = {
        "format": "awr_eval_v1",
        "task": str(cfg.task.name),
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(init_checkpoint),
        "seed": int(cfg.evaluation.seed),
        "deterministic": bool(cfg.evaluation.deterministic),
        "num_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / max(1, len(episodes)),
        "episodes": episodes,
    }
    (output_dir / str(cfg.evaluation.output_filename)).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[output] {output_dir}")


@hydra.main(version_base="1.3", config_path="./config", config_name="overall")
def main(cfg: DictConfig) -> None:
    with ExitStack() as resources:
        _run(cfg, resources)


if __name__ == "__main__":
    main()
