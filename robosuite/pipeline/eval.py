"""Evaluate a HIL-SERL checkpoint in robosuite."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.src.environment import (
    RobosuiteObservationAdapter,
    build_robosuite_env,
    sparse_success_reward,
    unpack_robosuite_step,
)
from robosuite.pipeline.src.hil_serl import HILSERLAgent
from robosuite.pipeline.utils.runtime import (
    build_runtime_cfg,
    now_readable,
    reset_observation_adapter,
    resolve_camera_names,
    set_seed,
    write_resolved_config,
)


def _algorithm_config(cfg: DictConfig) -> dict[str, Any]:
    algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if not isinstance(algorithm, dict):
        raise TypeError("algorithm must resolve to a mapping.")
    algorithm["encoder"]["pretrained_path"] = to_absolute_path(str(algorithm["encoder"]["pretrained_path"]))
    algorithm["sac"]["device"] = str(cfg.runtime.learner_device)
    algorithm["sac"]["inference_device"] = str(cfg.runtime.inference_device)
    trainer = algorithm["trainer"]
    trainer["warmup_steps"] = int(trainer.pop("training_starts"))
    trainer["steps_per_update"] = int(trainer.pop("policy_publish_interval"))
    trainer["max_learner_steps"] = int(cfg.runtime.max_learner_steps)
    return algorithm


def _run(cfg: DictConfig, resources: ExitStack) -> None:
    set_seed(int(cfg.evaluation.seed))
    if cfg.evaluation.checkpoint is None:
        raise ValueError("Set evaluation.checkpoint to a HIL-SERL checkpoint.")
    checkpoint = Path(to_absolute_path(str(cfg.evaluation.checkpoint)))
    if checkpoint.is_dir():
        checkpoint = checkpoint / str(cfg.checkpoint.directory) / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    camera_names = resolve_camera_names(cfg)
    interactive = bool(cfg.evaluation.viewer_enabled)
    env = build_robosuite_env(
        build_runtime_cfg(
            cfg,
            camera_names,
            has_renderer=interactive,
            has_offscreen_renderer=False,
            renderer="mjviewer" if interactive else str(cfg.env.renderer),
        )
    )
    resources.callback(env.close)
    render_env = build_robosuite_env(
        build_runtime_cfg(
            cfg,
            camera_names,
            has_renderer=False,
            has_offscreen_renderer=True,
        )
    )
    resources.callback(render_env.close)
    render_env.reset()
    adapter = RobosuiteObservationAdapter(
        env,
        render_env=render_env,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        image_obs_fps=float(cfg.runtime.image_obs_fps),
    )
    observation, _ = reset_observation_adapter(adapter, preserve_mjviewer=interactive)
    action_low, action_high = adapter.action_spec()
    agent = HILSERLAgent.from_config(
        _algorithm_config(cfg),
        observation_example=observation,
        action_low=action_low,
        action_high=action_high,
    )
    agent.load_checkpoint(checkpoint, load_buffers=False)

    episodes: list[dict[str, Any]] = []
    for episode_index in range(int(cfg.evaluation.num_episodes)):
        if episode_index > 0:
            observation, _ = reset_observation_adapter(adapter, preserve_mjviewer=interactive)
        episode_return = 0.0
        success = False
        length = 0
        for length in range(1, int(cfg.evaluation.max_episode_steps) + 1):
            action = agent.select_action(
                observation,
                deterministic=bool(cfg.evaluation.deterministic),
            )
            raw_next_obs, _, terminated, truncated, info = unpack_robosuite_step(env.step(action))
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            episode_return += reward
            observation = adapter.transform(raw_next_obs)
            if bool(terminated or truncated or success):
                break
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
        "task": str(cfg.task.name),
        "checkpoint": str(checkpoint),
        "seed": int(cfg.evaluation.seed),
        "deterministic": bool(cfg.evaluation.deterministic),
        "num_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / max(1, len(episodes)),
        "episodes": episodes,
    }
    output_dir = (
        Path(to_absolute_path(str(cfg.logging.output_root)))
        / str(cfg.task.name)
        / f"eval_{checkpoint.stem}_{now_readable()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config(cfg, output_dir)
    metadata = {
        "run_type": "evaluation",
        "task": str(cfg.task.name),
        "checkpoint": str(checkpoint),
        "seed": int(cfg.evaluation.seed),
        "deterministic": bool(cfg.evaluation.deterministic),
    }
    (output_dir / str(cfg.logging.metadata_filename)).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_path = output_dir / str(cfg.evaluation.output_filename)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eval] success_rate={summary['success_rate']:.4f}")
    print(f"[eval] results={output_path}")


@hydra.main(version_base="1.2", config_path="config", config_name="overall")
def main(cfg: DictConfig) -> None:
    with ExitStack() as resources:
        _run(cfg, resources)


if __name__ == "__main__":
    main()
