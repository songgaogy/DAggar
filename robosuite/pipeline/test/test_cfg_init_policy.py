"""Integration eval entry for CFG-initialized DIPOLE policy (omega=0, pos-only).

This script evaluates the policy initialized from `runtime.init_checkpoint`
without online human-in-the-loop updates. It always enforces CFG guidance
omega to 0.0 at inference time to measure the pos-branch-only behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Default direct Python invocations to headless EGL before robosuite imports MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from robosuite.pipeline.envs import build_robosuite_env, sparse_success_reward
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import resolve_camera_names, resolve_requested_device, set_seed


DEFAULT_OUTPUT_ROOT = "./outputs/DIPOLE/test_cfg_init_policy"
DEFAULT_VIDEO_CAMERA = "agentview"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_SIZE = 512
ENFORCED_OMEGA = 0.0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True, help="Path to the base flow init checkpoint.")
    parser.add_argument("--env-name", required=True, help="Robosuite env, e.g. PickPlaceBread.")
    parser.add_argument("--task-name", required=True, help="Task name used for language instruction.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--episode-max-steps", type=int, default=300, help="Per-episode step cap.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory for summaries and videos.")
    parser.add_argument("--video-output", default="true", help="Whether to save mp4 videos.")
    parser.add_argument("--video-camera", default=DEFAULT_VIDEO_CAMERA, help="Camera for video recording.")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video frame rate.")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_SIZE, help="Video frame height.")
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_SIZE, help="Video frame width.")
    parser.add_argument("--max-videos", type=int, default=0, help="Cap videos saved (0 = all).")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic action sampling.")
    parser.add_argument("--device", default=None, help="Override eval device (e.g. cuda:0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-tag", default=None, help="Optional tag appended to output directory name.")
    return parser.parse_args()


def _reset_env(env) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        return reset_output
    return reset_output, {}


def _capture_frame(env, *, video_camera: str, video_height: int, video_width: int) -> np.ndarray:
    frame = env.sim.render(
        height=int(video_height),
        width=int(video_width),
        camera_name=str(video_camera),
    )
    return np.ascontiguousarray(np.flipud(np.asarray(frame, dtype=np.uint8)))


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if len(frames) == 0:
        return
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _build_eval_output_dir(
    output_root: Path,
    init_checkpoint: Path,
    task_name: str,
    run_tag: str | None,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [task_name, init_checkpoint.stem, "cfg_pos_only", timestamp]
    if run_tag:
        parts.insert(0, str(run_tag))
    output_dir = output_root / "__".join(parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _wilson_ci95(success_count: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = float(success_count) / float(total)
    z = 1.96
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float(max(0.0, (center - spread) / denom)), float(min(1.0, (center + spread) / denom))


def run_cfg_init_policy_eval(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be > 0.")
    if int(args.episode_max_steps) <= 0:
        raise ValueError("--episode-max-steps must be > 0.")
    if int(args.video_fps) <= 0:
        raise ValueError("--video-fps must be > 0.")
    if int(args.video_height) <= 0:
        raise ValueError("--video-height must be > 0.")
    if int(args.video_width) <= 0:
        raise ValueError("--video-width must be > 0.")
    if int(args.max_videos) < 0:
        raise ValueError("--max-videos must be >= 0; use 0 to save all videos.")

    set_seed(int(args.seed))
    init_checkpoint = Path(to_absolute_path(str(args.init_checkpoint))).resolve()
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint does not exist: {init_checkpoint}")

    init_cfg = OmegaConf.create({"runtime": {"init_checkpoint": str(init_checkpoint)}})
    _, init_payload = load_init_checkpoint_payload(init_cfg)
    if init_payload is None:
        raise RuntimeError(f"Failed to load init checkpoint payload from {init_checkpoint}")

    env_metadata = resolve_flow_task_metadata(init_payload, str(args.task_name))
    if env_metadata is None:
        raise ValueError(
            f"Could not find env metadata for task '{args.task_name}' in init checkpoint {init_checkpoint}."
        )

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "train_dipole.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.env.environment = str(args.env_name)
    cfg.algorithm.task_name = str(args.task_name)
    cfg.runtime.init_checkpoint = str(init_checkpoint)
    cfg.runtime.use_init_checkpoint_camera_names = True
    cfg.runtime.use_init_checkpoint_model = True

    requested_camera_names = resolve_camera_names(cfg)
    if bool(cfg.runtime.use_init_checkpoint_camera_names):
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    cfg.algorithm.camera_names = list(policy_camera_names)

    if bool(cfg.runtime.use_init_checkpoint_model):
        if "model_cfg" in init_payload:
            cfg.algorithm.flow.model = init_payload["model_cfg"]
        if "task_prompt_map" in init_payload:
            cfg.algorithm.flow.task_prompt_map = init_payload["task_prompt_map"]
        if init_payload.get("act_mean") is not None:
            cfg.algorithm.flow.action_horizon = int(np.asarray(init_payload["act_mean"]).shape[0])
            cfg.algorithm.flow.execute_horizon = 1

    eval_device = resolve_requested_device(
        str(args.device) if args.device else str(cfg.algorithm.flow.inference_device),
        fallback="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    cfg.algorithm.flow.device = str(eval_device)
    cfg.algorithm.flow.inference_device = str(eval_device)
    cfg.algorithm.dipole.guidance_omega = float(ENFORCED_OMEGA)

    runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer=str(cfg.env.renderer),
    )
    env = build_robosuite_env(runtime_cfg)
    proprio_extractor = bind_flow_proprio_extractor(env, env_metadata)
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.flow, "camera_aliases", {}) or {}).items()
    }

    output_root = Path(to_absolute_path(str(args.output_root)))
    output_dir = _build_eval_output_dir(
        output_root=output_root,
        init_checkpoint=init_checkpoint,
        task_name=str(args.task_name),
        run_tag=args.run_tag,
    )
    video_output = _parse_bool(args.video_output)
    video_dir = output_dir / "videos"
    if video_output:
        video_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    success_count = 0
    videos_saved = 0

    try:
        raw_obs, _ = _reset_env(env)
        obs_example = convert_env_camera_observation(
            raw_obs,
            env=env,
            extractor=proprio_extractor,
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
            raise TypeError("Expected cfg.algorithm to resolve into a dict.")

        # add CFG guidance embedding
        agent = build_algorithm(
            algorithm_cfg,
            observation_example=obs_example,
            sample_action=np.zeros_like(action_low, dtype=np.float32),
            action_low=action_low,
            action_high=action_high,
        )
        agent.load_flow_policy_checkpoint(init_checkpoint, task_name=str(args.task_name))
        agent.core.config.guidance_omega = float(ENFORCED_OMEGA)
        agent.reset_policy_state()

        progress = tqdm(
            range(int(args.episodes)),
            desc=f"Eval cfg-init pos-only {args.task_name}",
            dynamic_ncols=True,
        )
        for episode_idx in progress:
            raw_obs, _ = _reset_env(env)
            obs = convert_env_camera_observation(
                raw_obs,
                env=env,
                extractor=proprio_extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
            )
            agent.reset_policy_state()

            record_this_episode = video_output and (
                int(args.max_videos) == 0 or videos_saved < int(args.max_videos)
            )
            frames: list[np.ndarray] = []
            if record_this_episode:
                frames.append(
                    _capture_frame(
                        env,
                        video_camera=args.video_camera,
                        video_height=int(args.video_height),
                        video_width=int(args.video_width),
                    )
                )

            episode_return = 0.0
            episode_steps = 0
            episode_success = False

            for _ in range(int(args.episode_max_steps)):
                action = agent.select_action(obs, deterministic=bool(args.deterministic))
                step_output = env.step(np.asarray(action, dtype=np.float32))
                if len(step_output) == 5:
                    raw_next_obs, _, done, truncated, info = step_output
                    done = bool(done or truncated)
                else:
                    raw_next_obs, _, done, info = step_output
                reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
                obs = convert_env_camera_observation(
                    raw_next_obs,
                    env=env,
                    extractor=proprio_extractor,
                    policy_camera_names=policy_camera_names,
                    camera_aliases=camera_aliases,
                    img_height=int(cfg.env.img_height),
                    img_width=int(cfg.env.img_width),
                )

                episode_return += float(reward)
                episode_steps += 1
                episode_success = bool(episode_success or success)

                if record_this_episode:
                    frames.append(
                        _capture_frame(
                            env,
                            video_camera=args.video_camera,
                            video_height=int(args.video_height),
                            video_width=int(args.video_width),
                        )
                    )

                if episode_success or done:
                    break

            success_count += int(episode_success)
            episode_result = {
                "episode_index": int(episode_idx),
                "return": float(episode_return),
                "steps": int(episode_steps),
                "success": bool(episode_success),
            }
            episode_results.append(episode_result)
            progress.set_postfix(
                success=f"{success_count}/{episode_idx + 1}",
                last_return=f"{episode_return:.2f}",
                refresh=False,
            )

            if record_this_episode and len(frames) > 0:
                video_path = video_dir / (
                    f"ep_{int(episode_idx):03d}_success_{int(episode_success)}_return_{episode_return:.2f}.mp4"
                )
                _write_video(video_path, frames, fps=int(args.video_fps))
                videos_saved += 1
                episode_result["video_path"] = str(video_path)

        mean_return = float(np.mean([item["return"] for item in episode_results])) if episode_results else 0.0
        mean_steps = float(np.mean([item["steps"] for item in episode_results])) if episode_results else 0.0
        success_rate = float(success_count) / float(len(episode_results)) if episode_results else 0.0
        ci_low, ci_high = _wilson_ci95(success_count, len(episode_results))

        summary = {
            "init_checkpoint": str(init_checkpoint),
            "env_name": str(args.env_name),
            "task_name": str(args.task_name),
            "omega": float(ENFORCED_OMEGA),
            "deterministic": bool(args.deterministic),
            "episodes": int(args.episodes),
            "episode_max_steps": int(args.episode_max_steps),
            "seed": int(args.seed),
            "device": str(eval_device),
            "video_output": bool(video_output),
            "video_camera": str(args.video_camera),
            "video_fps": int(args.video_fps),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "max_videos": int(args.max_videos),
            "headless": True,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "output_dir": str(output_dir),
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "success_rate": success_rate,
            "success_count": int(success_count),
            "success_rate_ci95_low": ci_low,
            "success_rate_ci95_high": ci_high,
            "policy_camera_names": list(policy_camera_names),
            "flow_config": asdict(agent.core.config),
            "episode_results": episode_results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print(f"init_checkpoint: {init_checkpoint}")
        print(f"output_dir: {output_dir}")
        print(
            f"omega={ENFORCED_OMEGA:.3f} episodes={len(episode_results)} "
            f"success_rate={success_rate:.3f} ({success_count}/{len(episode_results)}) "
            f"ci95=[{ci_low:.3f}, {ci_high:.3f}] "
            f"mean_return={mean_return:.3f} mean_steps={mean_steps:.1f}"
        )
        if video_output:
            print(f"videos: {video_dir} ({videos_saved} saved)")
        return summary
    finally:
        try:
            env.close()
        finally:
            proprio_extractor.close()


@pytest.mark.integration
@pytest.mark.slow
def test_cfg_init_policy_integration_eval() -> None:
    init_checkpoint = os.getenv("CFG_INIT_CHECKPOINT")
    if not init_checkpoint:
        pytest.skip("Set CFG_INIT_CHECKPOINT to enable integration eval.")

    args = argparse.Namespace(
        init_checkpoint=init_checkpoint,
        env_name=os.getenv("CFG_EVAL_ENV_NAME", "PickPlaceBread"),
        task_name=os.getenv("CFG_EVAL_TASK_NAME", "PickPlaceBread"),
        episodes=int(os.getenv("CFG_EVAL_EPISODES", "1")),
        episode_max_steps=int(os.getenv("CFG_EVAL_MAX_STEPS", "100")),
        output_root=os.getenv("CFG_EVAL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
        video_output=os.getenv("CFG_EVAL_VIDEO_OUTPUT", "false"),
        video_camera=os.getenv("CFG_EVAL_VIDEO_CAMERA", DEFAULT_VIDEO_CAMERA),
        video_fps=int(os.getenv("CFG_EVAL_VIDEO_FPS", str(DEFAULT_VIDEO_FPS))),
        video_height=int(os.getenv("CFG_EVAL_VIDEO_HEIGHT", str(DEFAULT_VIDEO_SIZE))),
        video_width=int(os.getenv("CFG_EVAL_VIDEO_WIDTH", str(DEFAULT_VIDEO_SIZE))),
        max_videos=int(os.getenv("CFG_EVAL_MAX_VIDEOS", "0")),
        deterministic=_parse_bool(os.getenv("CFG_EVAL_DETERMINISTIC", "true")),
        device=os.getenv("CFG_EVAL_DEVICE", None),
        seed=int(os.getenv("CFG_EVAL_SEED", "42")),
        run_tag=os.getenv("CFG_EVAL_RUN_TAG", "pytest"),
    )
    summary = run_cfg_init_policy_eval(args)
    assert summary["omega"] == ENFORCED_OMEGA
    assert int(summary["episodes"]) > 0
    assert "success_rate" in summary


def main() -> None:
    args = _parse_args()
    run_cfg_init_policy_eval(args)


if __name__ == "__main__":
    main()
