from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from robosuite.pipeline.envs import (
    RobosuiteObservationAdapter,
    build_robosuite_env,
    sparse_success_reward,
)
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.utils import (
    build_runtime_cfg,
    reset_observation_adapter,
    resolve_algorithm_devices,
    resolve_camera_names,
)

DEFAULT_OUTPUT_ROOT = "./outputs/hil_serl/eval"
DEFAULT_VIDEO_CAMERA = "agentview"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_SIZE = 512


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HIL-SERL checkpoint and report success rate.")
    parser.add_argument("--checkpoint", required=True, help="Path to a HIL-SERL checkpoint.")
    parser.add_argument("--env-name", required=True, help="Robosuite env name, e.g. Stack or PickPlaceCan.")
    parser.add_argument("--task-name", default=None, help="Task label used in the output directory and summary.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--episode-max-steps", type=int, default=500, help="Per-episode step cap.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory for eval summaries and videos.")
    parser.add_argument("--video-output", default="false", help="Whether to save mp4 videos.")
    parser.add_argument("--video-camera", default=DEFAULT_VIDEO_CAMERA, help="Camera to save for videos.")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS, help="FPS for saved videos.")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_SIZE, help="Saved video frame height.")
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_SIZE, help="Saved video frame width.")
    parser.add_argument("--interactive", action="store_true", help="Open the robosuite viewer during eval.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action sampling.")
    parser.add_argument("--device", default=None, help="Optional learner/model device override.")
    parser.add_argument("--inference-device", default=None, help="Optional inference device override.")
    parser.add_argument("--camera-names", default=None, help="Comma-separated policy camera names override.")
    parser.add_argument("--image-size", type=int, default=None, help="Policy image size fallback when run config is absent.")
    return parser.parse_args()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}.")


def _resolve_run_dir(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_resolved_config(run_dir: Path | None):
    if run_dir is None:
        return None
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return None
    return OmegaConf.load(config_path)


def _split_camera_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    return names or None


def _checkpoint_algorithm_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "hil-serl",
        "encoder": dict(payload.get("encoder_config", {})),
        "sac": dict(payload.get("sac_config", {})),
        "trainer": dict(payload.get("trainer_config", {})),
        "online_buffer": {"capacity": 1},
        "demo_buffer": {"capacity": 1},
    }


def _checkpoint_camera_names(payload: dict[str, Any]) -> list[str]:
    encoder_cfg = dict(payload.get("encoder_config", {}))
    image_keys = [str(name) for name in list(encoder_cfg.get("image_keys", []) or [])]
    return image_keys or ["agentview"]


def _checkpoint_proprio_keys(payload: dict[str, Any]) -> list[str]:
    encoder_cfg = dict(payload.get("encoder_config", {}))
    proprio_keys = [str(name) for name in list(encoder_cfg.get("proprio_keys", []) or [])]
    return [key for key in proprio_keys if key != "state"]


def _load_eval_config(
    *,
    checkpoint_payload: dict[str, Any],
    checkpoint_path: Path,
    env_name: str,
    episode_max_steps: int,
    camera_names_override: list[str] | None,
    image_size_override: int | None,
):
    run_dir = _resolve_run_dir(checkpoint_path)
    resolved_cfg = _load_resolved_config(run_dir)
    if resolved_cfg is None:
        image_size = int(
            image_size_override
            or dict(checkpoint_payload.get("encoder_config", {})).get("image_size", 128)
            or 128
        )
        resolved_cfg = OmegaConf.create(
            {
                "env": {
                    "environment": str(env_name),
                    "robots": ["Panda"],
                    "config": "default",
                    "controller": None,
                    "renderer": "mjviewer",
                    "render_camera": DEFAULT_VIDEO_CAMERA,
                    "camera_names": camera_names_override or _checkpoint_camera_names(checkpoint_payload),
                    "proprio_keys": _checkpoint_proprio_keys(checkpoint_payload),
                    "img_height": image_size,
                    "img_width": image_size,
                    "control_freq": 20,
                    "horizon": int(episode_max_steps),
                },
                "algorithm": _checkpoint_algorithm_cfg(checkpoint_payload),
            }
        )
    else:
        resolved_cfg.env.environment = str(env_name)
        resolved_cfg.env.horizon = int(episode_max_steps)
        if camera_names_override is not None:
            resolved_cfg.env.camera_names = camera_names_override
        if image_size_override is not None:
            resolved_cfg.env.img_height = int(image_size_override)
            resolved_cfg.env.img_width = int(image_size_override)
    return resolved_cfg


def _prepare_algorithm_cfg(resolved_cfg, checkpoint_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not hasattr(resolved_cfg, "algorithm") or resolved_cfg.algorithm is None:
        algorithm_cfg = _checkpoint_algorithm_cfg(checkpoint_payload)
    else:
        algorithm_cfg = OmegaConf.to_container(resolved_cfg.algorithm, resolve=True)
    if not isinstance(algorithm_cfg, dict):
        raise TypeError("Resolved algorithm config must be a dictionary.")

    encoder_cfg = algorithm_cfg.get("encoder", None)
    if isinstance(encoder_cfg, dict) and encoder_cfg.get("pretrained_path"):
        encoder_cfg["pretrained_path"] = to_absolute_path(str(encoder_cfg["pretrained_path"]))

    sac_cfg = algorithm_cfg.setdefault("sac", {})
    if args.device is not None:
        sac_cfg["device"] = str(args.device)
    if args.inference_device is not None:
        sac_cfg["inference_device"] = str(args.inference_device)
    resolve_algorithm_devices(algorithm_cfg)
    return algorithm_cfg


def _build_eval_output_dir(output_root: Path, checkpoint_path: Path, task_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_root / f"{task_name}__{checkpoint_path.stem}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    args = _parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.episode_max_steps) <= 0:
        raise ValueError("--episode-max-steps must be positive.")

    checkpoint_path = Path(to_absolute_path(args.checkpoint)).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task_name = str(args.task_name or args.env_name)
    video_output = _parse_bool(args.video_output)

    resolved_cfg = _load_eval_config(
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
        env_name=str(args.env_name),
        episode_max_steps=int(args.episode_max_steps),
        camera_names_override=_split_camera_names(args.camera_names),
        image_size_override=args.image_size,
    )
    camera_names = resolve_camera_names(resolved_cfg)
    algorithm_cfg = _prepare_algorithm_cfg(resolved_cfg, checkpoint_payload, args)

    runtime_cfg = build_runtime_cfg(
        resolved_cfg,
        camera_names=camera_names,
        has_renderer=bool(args.interactive),
        has_offscreen_renderer=True,
        renderer=str(resolved_cfg.env.renderer),
    )
    env = build_robosuite_env(runtime_cfg)
    adapter = RobosuiteObservationAdapter(
        env,
        camera_names=camera_names,
        img_height=int(resolved_cfg.env.img_height),
        img_width=int(resolved_cfg.env.img_width),
        proprio_keys=tuple(resolved_cfg.env.proprio_keys or []),
        image_obs_fps=None,
    )

    output_root = Path(to_absolute_path(args.output_root))
    output_dir = _build_eval_output_dir(output_root, checkpoint_path, task_name)
    video_dir = output_dir / "videos"
    if video_output:
        video_dir.mkdir(parents=True, exist_ok=True)

    try:
        initial_obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=bool(args.interactive))
        action_low, action_high = adapter.action_spec()
        agent = build_algorithm(
            algorithm_cfg,
            observation_example=initial_obs,
            sample_action=np.zeros_like(action_low, dtype=np.float32),
            action_low=action_low,
            action_high=action_high,
        )
        agent.core.load_state_dict(checkpoint_payload["core"])

        episode_results: list[dict[str, Any]] = []
        success_count = 0

        progress = tqdm(range(int(args.episodes)), desc=f"Eval {task_name}", dynamic_ncols=True)
        for episode_idx in progress:
            obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=bool(args.interactive))
            frames: list[np.ndarray] = []
            if video_output:
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
                obs = adapter.transform(raw_next_obs, force_render=True)

                episode_return += float(reward)
                episode_steps += 1
                episode_success = bool(episode_success or success)

                if video_output:
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

            if video_output and len(frames) > 0:
                video_path = video_dir / (
                    f"ep_{int(episode_idx):03d}_success_{int(episode_success)}_return_{episode_return:.2f}.mp4"
                )
                _write_video(video_path, frames, fps=int(args.video_fps))
                episode_result["video_path"] = str(video_path)

        mean_return = float(np.mean([item["return"] for item in episode_results])) if episode_results else 0.0
        mean_steps = float(np.mean([item["steps"] for item in episode_results])) if episode_results else 0.0
        success_rate = float(success_count) / float(len(episode_results)) if episode_results else 0.0
        run_dir = _resolve_run_dir(checkpoint_path)
        run_info = _load_json(run_dir / "run_info.json") if run_dir is not None else None
        summary = {
            "checkpoint": str(checkpoint_path),
            "source_run_dir": None if run_dir is None else str(run_dir),
            "env_name": str(args.env_name),
            "task_name": task_name,
            "episodes": int(args.episodes),
            "episode_max_steps": int(args.episode_max_steps),
            "deterministic": bool(args.deterministic),
            "video_output": bool(video_output),
            "video_camera": str(args.video_camera),
            "video_fps": int(args.video_fps),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "output_dir": str(output_dir),
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "success_rate": success_rate,
            "success_count": int(success_count),
            "camera_names": list(camera_names),
            "algorithm_devices": {
                "device": str(algorithm_cfg.get("sac", {}).get("device")),
                "inference_device": str(algorithm_cfg.get("sac", {}).get("inference_device")),
            },
            "checkpoint_extra": dict(checkpoint_payload.get("extra", {}) or {}),
            "source_run_info": run_info,
            "episode_results": episode_results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print(f"checkpoint: {checkpoint_path}")
        print(f"output_dir: {output_dir}")
        print(
            f"episodes={len(episode_results)} success_rate={success_rate:.3f} "
            f"mean_return={mean_return:.3f} mean_steps={mean_steps:.1f}"
        )
        if video_output:
            print(f"videos: {video_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
