"""Headless eval driver for a trained DIPOLE checkpoint.

Uses the DIPOLE polarity flow policy and exposes the CFG guidance omega as a
CLI argument so a caller can sweep aggressiveness without retraining.

Outputs per run:
- summary.json with success rate, per-episode results, omega
- videos/ep_*.mp4 (offscreen-rendered) when --video-output true
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from robosuite.pipeline.algorithms.dipole.common import (
    DipoleConfig,
    FlowAugmentationConfig,
)
from robosuite.pipeline.algorithms.dipole.models import DipoleFlowPolicy
from robosuite.pipeline.envs import build_robosuite_env, sparse_success_reward
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import resolve_requested_device


DEFAULT_OUTPUT_ROOT = "./outputs/DIPOLE/eval"
DEFAULT_VIDEO_CAMERA = "agentview"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_SIZE = 512


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a DIPOLE checkpoint.")
    parser.add_argument("--env-name", required=True, help="Robosuite env, e.g. PickPlaceBread.")
    parser.add_argument("--task-name", required=True, help="Demo / task name used for language instruction.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per omega.")
    parser.add_argument("--episode-max-steps", type=int, default=300, help="Per-episode step cap.")
    parser.add_argument("--omega", type=float, default=2.0, help="CFG guidance omega used at sampling time.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory for eval summaries and videos.")
    parser.add_argument("--video-output", default="true", help="Whether to save mp4 videos.")
    parser.add_argument("--video-camera", default=DEFAULT_VIDEO_CAMERA, help="Camera to record.")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video frame rate.")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_SIZE, help="Video frame height.")
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_SIZE, help="Video frame width.")
    parser.add_argument("--max-videos", type=int, default=0, help="Cap videos saved (<=0 = all).")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic action sampling.")
    parser.add_argument("--init-checkpoint", default=None, help="Optional base flow checkpoint for env metadata.")
    parser.add_argument("--device", default=None, help="Override eval device (e.g. cuda:0).")
    parser.add_argument("--run-tag", default=None, help="Optional tag appended to the output dir name.")
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


def _resolve_init_checkpoint(
    *,
    requested: str | None,
    run_info: dict[str, Any] | None,
    resolved_cfg,
) -> Path | None:
    if requested:
        path = Path(to_absolute_path(str(requested)))
        return path if path.exists() else None

    if run_info is not None:
        initialized = run_info.get("initialized_checkpoint")
        if initialized:
            path = Path(str(initialized))
            if path.exists():
                return path

    if resolved_cfg is not None:
        init_checkpoint_cfg = getattr(resolved_cfg.runtime, "init_checkpoint", None)
        if init_checkpoint_cfg:
            path = Path(to_absolute_path(str(init_checkpoint_cfg)))
            if path.exists():
                return path
    return None


def _build_eval_output_dir(
    output_root: Path,
    checkpoint_path: Path,
    task_name: str,
    omega: float,
    run_tag: str | None,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    omega_tag = f"omega_{omega:.2f}".replace(".", "p")
    parts = [task_name, checkpoint_path.stem, omega_tag, timestamp]
    if run_tag:
        parts.insert(0, run_tag)
    output_dir = output_root / "__".join(parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_eval_device(payload: dict[str, Any], override: str | None) -> str:
    if override:
        return resolve_requested_device(override, fallback="cpu")
    flow_cfg = dict(payload.get("flow_config", {}))
    requested = flow_cfg.get("inference_device") or flow_cfg.get("device") or "cpu"
    fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
    return resolve_requested_device(requested, fallback=fallback)


def _build_dipole_policy(
    payload: dict[str, Any],
    *,
    task_name: str,
    device: str,
    omega: float,
) -> DipoleFlowPolicy:
    flow_cfg = dict(payload["flow_config"])
    aug_cfg = dict(flow_cfg.get("augmentation", {}) or {})
    config = DipoleConfig(
        action_dim=int(flow_cfg["action_dim"]),
        proprio_dim=int(flow_cfg["proprio_dim"]),
        action_horizon=int(flow_cfg.get("action_horizon", 8)),
        execute_horizon=int(flow_cfg.get("execute_horizon", 1)),
        image_size=int(flow_cfg.get("image_size", 128)),
        learning_rate=float(flow_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(flow_cfg.get("weight_decay", 1e-6)),
        grad_clip_norm=float(flow_cfg.get("grad_clip_norm", 1.0)),
        lambda_endpoint=float(flow_cfg.get("lambda_endpoint", 0.5)),
        lambda_smooth=float(flow_cfg.get("lambda_smooth", 0.05)),
        n_ode_steps=int(flow_cfg.get("n_ode_steps", 8)),
        device=str(device),
        inference_device=str(device),
        task_name=str(payload.get("task_name", task_name)),
        language_instruction=str(payload.get("language_instruction", task_name)),
        augmentation=FlowAugmentationConfig(
            minimal_shift_pad=int(aug_cfg.get("minimal_shift_pad", 2)),
            eye_in_hand_crop_scale=float(aug_cfg.get("eye_in_hand_crop_scale", 0.88)),
        ),
        beta=float(flow_cfg.get("beta", 2.0)),
        k=float(flow_cfg.get("k", 0.0)),
        guidance_omega=float(omega),
        g_sign=str(flow_cfg.get("g_sign", "negate_raw")),
        g_normalization=str(flow_cfg.get("g_normalization", "batch_zscore")),
        g_clip=float(flow_cfg.get("g_clip", 10.0)),
        polarity_embedding_init=str(flow_cfg.get("polarity_embedding_init", "zero_pos")),
        polarity_embedding_init_scale=float(flow_cfg.get("polarity_embedding_init_scale", 1e-3)),
    )
    policy = DipoleFlowPolicy(
        model_cfg=dict(payload["model_cfg"]),
        config=config,
        camera_names=[str(name) for name in payload["camera_names"]],
    )
    policy.load_state_dict(payload["core"])
    policy.set_language_instruction(str(payload.get("language_instruction", task_name)))
    policy.reset_action_chunk()
    return policy


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


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(to_absolute_path(args.checkpoint)).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    video_output = _parse_bool(args.video_output)
    run_dir = _resolve_run_dir(checkpoint_path)
    run_info = _load_json(run_dir / "run_info.json") if run_dir is not None else None
    resolved_cfg = _load_resolved_config(run_dir)
    init_checkpoint = _resolve_init_checkpoint(
        requested=args.init_checkpoint,
        run_info=run_info,
        resolved_cfg=resolved_cfg,
    )
    if init_checkpoint is None:
        raise FileNotFoundError(
            "Unable to resolve the base flow checkpoint for env metadata. "
            "Pass --init-checkpoint explicitly or evaluate from a run dir containing run_info.json."
        )

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    init_cfg = OmegaConf.create({"runtime": {"init_checkpoint": str(init_checkpoint)}})
    _, init_payload = load_init_checkpoint_payload(init_cfg)
    env_metadata = resolve_flow_task_metadata(init_payload, args.task_name)
    if env_metadata is None:
        raise ValueError(
            f"Could not find env metadata for task '{args.task_name}' in init checkpoint {init_checkpoint}."
        )

    if resolved_cfg is None:
        image_size = int(checkpoint_payload["flow_config"].get("image_size", 128))
        resolved_cfg = OmegaConf.create(
            {
                "env": {
                    "environment": args.env_name,
                    "renderer": "mjviewer",
                    "img_height": image_size,
                    "img_width": image_size,
                    "proprio_keys": [],
                    "control_freq": 20,
                    "horizon": None,
                }
            }
        )
    resolved_cfg.env.environment = str(args.env_name)

    policy_camera_names = [str(name) for name in checkpoint_payload["camera_names"]]
    runtime_cfg = build_flow_runtime_cfg(
        resolved_cfg,
        env_metadata=env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer=str(resolved_cfg.env.renderer),
    )
    env = build_robosuite_env(runtime_cfg)
    proprio_extractor = bind_flow_proprio_extractor(env, env_metadata)
    eval_device = _resolve_eval_device(checkpoint_payload, args.device)
    policy = _build_dipole_policy(
        checkpoint_payload,
        task_name=args.task_name,
        device=eval_device,
        omega=float(args.omega),
    )

    output_root = Path(to_absolute_path(args.output_root))
    output_dir = _build_eval_output_dir(
        output_root,
        checkpoint_path,
        args.task_name,
        omega=float(args.omega),
        run_tag=args.run_tag,
    )
    video_dir = output_dir / "videos"
    if video_output:
        video_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    success_count = 0
    videos_saved = 0

    try:
        progress = tqdm(
            range(int(args.episodes)),
            desc=f"Eval {args.task_name} omega={args.omega:.2f}",
            dynamic_ncols=True,
        )
        for episode_idx in progress:
            raw_obs, _ = _reset_env(env)
            obs = convert_env_camera_observation(
                raw_obs,
                env=env,
                extractor=proprio_extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases={},
                img_height=int(resolved_cfg.env.img_height),
                img_width=int(resolved_cfg.env.img_width),
            )
            policy.reset_action_chunk()

            record_this_episode = video_output and (
                int(args.max_videos) <= 0 or videos_saved < int(args.max_videos)
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
                action = policy.select_action(obs, deterministic=bool(args.deterministic))
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
                    camera_aliases={},
                    img_height=int(resolved_cfg.env.img_height),
                    img_width=int(resolved_cfg.env.img_width),
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
        # Wilson 95% CI for the binomial success rate; informative when episodes is small.
        if episode_results:
            n = len(episode_results)
            p = success_rate
            z = 1.96
            denom = 1.0 + z * z / n
            center = p + z * z / (2.0 * n)
            spread = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
            success_rate_ci_low = float(max(0.0, (center - spread) / denom))
            success_rate_ci_high = float(min(1.0, (center + spread) / denom))
        else:
            success_rate_ci_low = 0.0
            success_rate_ci_high = 0.0

        summary = {
            "checkpoint": str(checkpoint_path),
            "init_checkpoint": str(init_checkpoint),
            "env_name": str(args.env_name),
            "task_name": str(args.task_name),
            "omega": float(args.omega),
            "deterministic": bool(args.deterministic),
            "episodes": int(args.episodes),
            "episode_max_steps": int(args.episode_max_steps),
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
            "success_rate_ci95_low": success_rate_ci_low,
            "success_rate_ci95_high": success_rate_ci_high,
            "policy_camera_names": list(policy_camera_names),
            "flow_config": asdict(policy.config),
            "episode_results": episode_results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print(f"checkpoint: {checkpoint_path}")
        print(f"output_dir: {output_dir}")
        print(
            f"omega={args.omega:.3f} episodes={len(episode_results)} "
            f"success_rate={success_rate:.3f} ({success_count}/{len(episode_results)}) "
            f"ci95=[{success_rate_ci_low:.3f}, {success_rate_ci_high:.3f}] "
            f"mean_return={mean_return:.3f} mean_steps={mean_steps:.1f}"
        )
        if video_output:
            print(f"videos: {video_dir} ({videos_saved} saved)")
    finally:
        try:
            env.close()
        finally:
            proprio_extractor.close()


if __name__ == "__main__":
    main()
