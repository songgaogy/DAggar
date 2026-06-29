"""Headless success-rate + video eval for an offline-finetuned DIPOLE policy.

Reuses the leaf helpers from :mod:`robosuite.pipeline.eval_dipole` (policy build,
env build, frame capture, MP4 writer) so the offline-finetuned checkpoint is
evaluated exactly like an online DIPOLE checkpoint, with DIPOLE two-branch
guidance ``(1 + omega) * v_pos - omega * v_neg``.

Output layout follows the per-run convention (cf. ``pipeline/sft``): results land
under the training run directory by default::

    <run_dir>/eval/<ckpt_stem>__omega_<w>__<timestamp>/
        summary.json
        videos/ep_NNN_success_S_return_R.mp4

Example::

    MUJOCO_GL=egl python -m robosuite.pipeline.offline.eval_offline_dipole \\
        --checkpoint outputs/dipole_offline/PickPlaceCereal/<ts>/checkpoints/latest.pt \\
        --env-name PickPlaceCereal --task-name PickPlaceCereal \\
        --omega 0.2 --episodes 50 --episode-max-steps 500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from robosuite.pipeline.envs import build_robosuite_env, sparse_success_reward
from robosuite.pipeline.eval_dipole import (
    DEFAULT_EVAL_SEED,
    DEFAULT_VIDEO_CAMERA,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_SIZE,
    _assert_eval_seeds_disjoint,
    _build_dipole_policy,
    _capture_frame,
    _load_json,
    _load_resolved_config,
    _parse_bool,
    _reset_env,
    _resolve_eval_device,
    _resolve_init_checkpoint,
    _resolve_run_dir,
    _write_video,
)
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import EnvRandomReducer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Offline-finetuned DIPOLE checkpoint.")
    parser.add_argument("--env-name", required=True, help="Robosuite env, e.g. PickPlaceCereal.")
    parser.add_argument("--task-name", required=True, help="Task name for the language instruction.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--episode-max-steps", type=int, default=500, help="Per-episode step cap.")
    parser.add_argument("--omega", type=float, default=0.2, help="DIPOLE guidance omega at sampling time.")
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=8,
        help="Actions executed from each planned chunk before replanning (default: 8; use 1 for step-wise replanning).",
    )
    parser.add_argument("--output-root", default=None, help="Override eval output root (defaults to <run_dir>/eval).")
    parser.add_argument("--video-output", default="true", help="Whether to save mp4 videos.")
    parser.add_argument("--video-camera", default=DEFAULT_VIDEO_CAMERA, help="Camera to record.")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video frame rate.")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_SIZE, help="Video frame height.")
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_SIZE, help="Video frame width.")
    parser.add_argument("--max-videos", type=int, default=0, help="Cap videos saved (<=0 = all).")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic action sampling.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=(
            "Base seed for deterministic episode layouts. Episode i is reset under seed+i. "
            "Defaults to a high band disjoint from training seeds."
        ),
    )
    parser.add_argument("--init-checkpoint", default=None, help="Optional base flow checkpoint for env metadata.")
    parser.add_argument("--device", default=None, help="Override eval device (e.g. cuda:0).")
    return parser.parse_args()


def _resolve_execute_horizon(requested: int, action_horizon: int) -> int:
    execute_horizon = int(requested)
    action_horizon = int(action_horizon)
    if not 1 <= execute_horizon <= action_horizon:
        raise ValueError(
            f"--execute-horizon must be in [1, {action_horizon}], got {execute_horizon}."
        )
    return execute_horizon


def _build_eval_output_dir(
    *,
    output_root: Path,
    checkpoint_path: Path,
    omega: float,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    omega_tag = f"omega_{omega:.2f}".replace(".", "p")
    output_dir = output_root / f"{checkpoint_path.stem}__{omega_tag}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
            "Pass --init-checkpoint or evaluate from a run dir with run_info.json."
        )

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    init_cfg = OmegaConf.create({"runtime": {"init_checkpoint": str(init_checkpoint)}})
    _, init_payload = load_init_checkpoint_payload(init_cfg)
    env_metadata = resolve_flow_task_metadata(init_payload, args.task_name)
    if env_metadata is None:
        raise ValueError(
            f"Could not find env metadata for task '{args.task_name}' in {init_checkpoint}."
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
    checkpoint_execute_horizon = int(policy.config.execute_horizon)
    policy.config.execute_horizon = _resolve_execute_horizon(
        int(args.execute_horizon),
        int(policy.config.action_horizon),
    )
    print(
        f"[eval] execute_horizon={policy.config.execute_horizon} "
        f"(checkpoint={checkpoint_execute_horizon}, action_horizon={policy.config.action_horizon})"
    )

    if args.output_root is not None:
        output_root = Path(to_absolute_path(args.output_root))
    elif run_dir is not None:
        output_root = run_dir / "eval"
    else:
        output_root = checkpoint_path.parent / "eval"
    output_dir = _build_eval_output_dir(
        output_root=output_root, checkpoint_path=checkpoint_path, omega=float(args.omega)
    )
    video_dir = output_dir / "videos"
    if video_output:
        video_dir.mkdir(parents=True, exist_ok=True)

    # Fixed-seed eval (sft-style): episode i uses base_seed+i to seed the env
    # layout + global RNG, so all checkpoints see identical task layouts. The
    # policy still samples stochastically, but from a seeded noise stream.
    env_random_reducer = EnvRandomReducer(int(args.seed))
    _assert_eval_seeds_disjoint(run_info, eval_base=int(args.seed), eval_count=int(args.episodes))
    print(f"[eval] fixed-seed layouts: base_seed={args.seed} (episode i -> seed {args.seed}+i)")

    episode_results: list[dict[str, Any]] = []
    success_count = 0
    videos_saved = 0

    try:
        progress = tqdm(
            range(int(args.episodes)),
            desc=f"Eval(offline) {args.task_name} omega={args.omega:.2f}",
            dynamic_ncols=True,
        )
        for episode_idx in progress:
            # Seed env layout + global RNG before reset so the placement sampler is
            # deterministic; re-seed global RNG after reset so the policy noise stream
            # starts deterministically (reset consumes RNG).
            episode_seed = env_random_reducer.prepare_episode(env, int(episode_idx))
            raw_obs, _ = _reset_env(env)
            if episode_seed is not None:
                EnvRandomReducer.seed_global(int(episode_seed))
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
                "episode_seed": int(episode_seed) if episode_seed is not None else None,
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
            "run_dir": str(run_dir) if run_dir is not None else None,
            "env_name": str(args.env_name),
            "task_name": str(args.task_name),
            "omega": float(args.omega),
            "checkpoint_execute_horizon": checkpoint_execute_horizon,
            "execute_horizon": int(policy.config.execute_horizon),
            "deterministic": bool(args.deterministic),
            "seed": int(args.seed),
            "seed_rule": "base_seed+episode_index",
            "env_reset_seed_rule": "base_seed+episode_index",
            "policy_seed_rule": "base_seed+episode_index",
            "episodes": int(args.episodes),
            "episode_max_steps": int(args.episode_max_steps),
            "video_output": bool(video_output),
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
            f"omega={args.omega:.3f} execute_horizon={policy.config.execute_horizon} "
            f"episodes={len(episode_results)} "
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
