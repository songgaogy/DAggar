"""Success-rate evaluation for offline-finetuned flow-dagger checkpoints.

Runs closed-loop rollouts of one or more checkpoints in a headless robosuite env
and reports task success rate. No video is recorded (offscreen camera rendering
is used only to feed the policy; nothing is written to disk except a small JSON
results file).

This reuses the exact env / agent construction from ``train_flow_offline.py`` and
the same rollout primitives (``select_action`` / ``sparse_success_reward``) as
``train_flow_dagger.py`` so the eval matches deployment behaviour.

Pass several checkpoints at once (e.g. the ``_at0002000.pt`` ... interim
snapshots) to trace the success-rate-vs-training-step curve in one run, which is
the decisive test for overfitting: if success peaks early then declines while the
training loss keeps dropping, the model is overfitting.

Usage
-----
    # original multitask flow checkpoint
    python -m robosuite.pipeline.temp.eval_flow_offline \
        checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt \
        --env PickPlaceBread --episodes 50 --device cuda:1

    # single checkpoint
    python -m robosuite.pipeline.temp.eval_flow_offline \
        outputs/flow_dagger_no-hil/PickPlaceBread/flow_offline_PickPlaceBread_traj00020_steps00010000.pt \
        --env PickPlaceBread --episodes 50 --device cuda:1

    # the whole overfitting curve (all interim snapshots, sorted)
    python -m robosuite.pipeline.temp.eval_flow_offline \
        outputs/flow_dagger_no-hil/PickPlaceBread/*_at*.pt \
        outputs/flow_dagger_no-hil/PickPlaceBread/flow_offline_PickPlaceBread_traj00020_steps00010000.pt \
        --env PickPlaceBread --episodes 50 --device cuda:1
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path

from robosuite.pipeline.envs import sparse_success_reward
from robosuite.pipeline.train_flow_dagger import (
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    reset_flow_policy_observation,
)

# Reuse the offline trainer's config + env/agent builders so eval setup cannot drift.
from robosuite.pipeline.temp.train_flow_offline import build_agent_and_env, build_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="One or more checkpoint .pt files to evaluate.")
    parser.add_argument("--env", default="PickPlaceBread", help="Robosuite environment / task name.")
    parser.add_argument("--episodes", type=int, default=50, help="Rollout episodes per checkpoint.")
    parser.add_argument("--max-steps", type=int, default=400, help="Max env steps per episode (eval horizon).")
    parser.add_argument("--device", default="cuda:0", help="Inference/learner device.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Episode i is reset under seed+i so all checkpoints see identical task layouts.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic flow sampling (default stochastic, matching flow-dagger eval).",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=None,
        help="Override flow execute_horizon (actions per inference). Default: from config.",
    )
    parser.add_argument(
        "--n-ode-steps",
        type=int,
        default=None,
        help="Override flow n_ode_steps for action sampling. Default: from config.",
    )
    # --- video options (off by default; enable to inspect failure modes) ---------
    parser.add_argument("--save-video", action="store_true", help="Save an mp4 per episode.")
    parser.add_argument(
        "--video-mode",
        choices=["all", "failures", "successes"],
        default="all",
        help="Which episodes to save when --save-video is set (default: all).",
    )
    parser.add_argument("--video-camera", default="agentview", help="Camera rendered into the video.")
    parser.add_argument("--video-fps", type=int, default=20, help="Saved video FPS.")
    parser.add_argument("--video-size", type=int, default=512, help="Saved video frame size (square).")
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory for videos (default: <first-ckpt-parent>/videos).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Base flow-dagger config (defaults to train_flow_offline's canonical config).",
    )
    parser.add_argument("--output", default=None, help="Optional path to write JSON results (default: alongside first ckpt).")
    return parser.parse_args()


def _capture_frame(env, *, video_camera: str, video_size: int) -> np.ndarray:
    # Dedicated high-res render (independent of the 128x128 policy obs). MuJoCo renders
    # bottom-up, so flip vertically. Matches robosuite/pipeline/eval_flow_dagger.py.
    frame = env.sim.render(height=int(video_size), width=int(video_size), camera_name=str(video_camera))
    return np.ascontiguousarray(np.flipud(np.asarray(frame, dtype=np.uint8)))


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if len(frames) == 0:
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


def _load_checkpoint_for_eval(agent, checkpoint: Path, *, task_name: str) -> dict[str, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "core" in payload:
        agent.load_checkpoint(checkpoint, load_buffers=False)
        core = payload.get("core", {})
        if isinstance(core, dict) and "ema_model" in core:
            weight_type = "ema"
        else:
            weight_type = str(payload.get("checkpoint_weight_type", "raw"))
        return {"checkpoint_format": "flow_dagger", "weight_type": weight_type}
    if "ema_model" in payload or "model" in payload:
        agent.load_flow_policy_checkpoint(checkpoint, task_name=task_name)
        return {
            "checkpoint_format": "flow_multi",
            "weight_type": "ema" if "ema_model" in payload else "raw",
        }
    raise KeyError(
        f"Unsupported checkpoint format: {checkpoint}. Expected either a flow-dagger "
        "checkpoint with key 'core' or an original flow checkpoint with 'model'/'ema_model'."
    )


def run_episode(
    agent,
    env,
    *,
    proprio_extractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    max_steps: int,
    deterministic: bool,
    episode_seed: int,
    capture_video: bool = False,
    video_camera: str = "agentview",
    video_size: int = 512,
) -> tuple[bool, int, list[np.ndarray]]:
    """Run one rollout. Returns (success, episode_length, frames)."""
    # Seed before reset so object placement is reproducible across checkpoints.
    np.random.seed(episode_seed)
    obs, _ = reset_flow_policy_observation(
        env,
        preserve_mjviewer=False,
        extractor=proprio_extractor,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=img_height,
        img_width=img_width,
    )
    agent.reset_policy_state()

    frames: list[np.ndarray] = []
    if capture_video:
        frames.append(_capture_frame(env, video_camera=video_camera, video_size=video_size))

    success = False
    steps = 0
    for _ in range(max_steps):
        action = agent.select_action(obs, deterministic=deterministic)
        env_action = np.asarray(action, dtype=np.float32)
        step_output = env.step(env_action)
        if len(step_output) == 5:
            raw_next_obs, _, done, truncated, info = step_output
            done = bool(done or truncated)
        else:
            raw_next_obs, _, done, info = step_output
        _, step_success = sparse_success_reward(env, info if isinstance(info, dict) else None)
        steps += 1
        if step_success:
            success = True
        if capture_video:
            frames.append(_capture_frame(env, video_camera=video_camera, video_size=video_size))
        obs = convert_env_camera_observation(
            raw_next_obs,
            env=env,
            extractor=proprio_extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=img_height,
            img_width=img_width,
        )
        if success or done:
            break
    return success, steps, frames


def main() -> None:
    args = parse_args()

    # Build a config mirroring flow-dagger; reuse the offline trainer's overrides path.
    cfg_args = argparse.Namespace(
        env=args.env,
        num_trajectories=20,  # unused for eval (no demos loaded)
        steps=0,
        device=args.device,
        seed=args.seed,
        config=args.config if args.config is not None else None,
    )
    if cfg_args.config is None:
        from robosuite.pipeline.temp.train_flow_offline import _DEFAULT_CONFIG

        cfg_args.config = str(_DEFAULT_CONFIG)
    cfg = build_cfg(cfg_args)
    if args.execute_horizon is not None:
        cfg.algorithm.flow.execute_horizon = int(args.execute_horizon)
    if args.n_ode_steps is not None:
        cfg.algorithm.flow.n_ode_steps = int(args.n_ode_steps)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # The original multitask checkpoint provides env metadata + model_cfg to build env/agent.
    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None or init_payload is None:
        raise FileNotFoundError(
            "Could not load runtime.init_checkpoint; it is needed to build the eval env/agent."
        )

    agent, env, proprio_extractor, policy_camera_names, camera_aliases = build_agent_and_env(cfg, init_payload)
    img_height = int(cfg.env.img_height)
    img_width = int(cfg.env.img_width)
    execute_horizon = int(cfg.algorithm.flow.execute_horizon)
    action_horizon = int(cfg.algorithm.flow.action_horizon)
    n_ode_steps = int(cfg.algorithm.flow.n_ode_steps)
    print(
        f"[env] task={args.env} cameras={policy_camera_names} action_horizon={action_horizon} "
        f"execute_horizon={execute_horizon} n_ode_steps={n_ode_steps} deterministic={args.deterministic} "
        f"episodes={args.episodes} max_steps={args.max_steps}"
    )

    checkpoints = [Path(to_absolute_path(c)) for c in args.checkpoints]
    for ckpt in checkpoints:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    video_root = (
        Path(to_absolute_path(args.video_dir))
        if args.video_dir is not None
        else checkpoints[0].parent / "videos"
    )
    if args.save_video:
        print(f"[video] saving '{args.video_mode}' episodes ({args.video_camera}, {args.video_size}px) -> {video_root}")

    results: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        ckpt_info = _load_checkpoint_for_eval(agent, ckpt, task_name=str(args.env))
        agent.reset_policy_state()
        ckpt_video_dir = video_root / ckpt.stem
        n_success = 0
        lengths: list[int] = []
        n_videos = 0
        started = time.monotonic()
        for ep in range(args.episodes):
            success, length, frames = run_episode(
                agent,
                env,
                proprio_extractor=proprio_extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=img_height,
                img_width=img_width,
                max_steps=int(args.max_steps),
                deterministic=bool(args.deterministic),
                episode_seed=int(args.seed) + ep,
                capture_video=bool(args.save_video),
                video_camera=str(args.video_camera),
                video_size=int(args.video_size),
            )
            n_success += int(success)
            lengths.append(length)
            if args.save_video:
                keep = (
                    args.video_mode == "all"
                    or (args.video_mode == "failures" and not success)
                    or (args.video_mode == "successes" and success)
                )
                if keep:
                    video_path = ckpt_video_dir / f"ep_{ep:03d}_success_{int(success)}_len_{length:03d}.mp4"
                    _write_video(video_path, frames, fps=int(args.video_fps))
                    n_videos += 1
            print(
                f"  [{ckpt.name}] ep={ep + 1}/{args.episodes} "
                f"success={int(success)} len={length} running_sr={n_success / (ep + 1):.3f}",
                flush=True,
            )
        success_rate = n_success / max(1, args.episodes)
        elapsed = time.monotonic() - started
        record = {
            "checkpoint": str(ckpt),
            "checkpoint_name": ckpt.name,
            "checkpoint_format": ckpt_info["checkpoint_format"],
            "weight_type": ckpt_info["weight_type"],
            "episodes": int(args.episodes),
            "successes": int(n_success),
            "success_rate": float(success_rate),
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "elapsed_sec": float(elapsed),
            "videos_saved": int(n_videos),
            "video_dir": str(ckpt_video_dir) if args.save_video and n_videos > 0 else None,
        }
        results.append(record)
        print(
            f"[result] {ckpt.name}: success_rate={success_rate:.3f} "
            f"({n_success}/{args.episodes}) mean_len={record['mean_episode_length']:.1f} "
            f"({elapsed:.0f}s)"
        )

    print("\n=== SUCCESS RATE SUMMARY ===")
    print(f"{'checkpoint':<60} {'success_rate':>12} {'successes':>10}")
    for r in results:
        print(f"{r['checkpoint_name']:<60} {r['success_rate']:>12.3f} {str(r['successes']) + '/' + str(r['episodes']):>10}")

    out_path = (
        Path(to_absolute_path(args.output))
        if args.output is not None
        else checkpoints[0].parent / "eval_success_rate.json"
    )
    payload = {
        "env": args.env,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "deterministic": bool(args.deterministic),
        "action_horizon": action_horizon,
        "execute_horizon": execute_horizon,
        "n_ode_steps": n_ode_steps,
        "seed": int(args.seed),
        "save_video": bool(args.save_video),
        "video_mode": str(args.video_mode) if args.save_video else None,
        "init_checkpoint": str(init_checkpoint),
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[done] wrote results -> {out_path}")

    try:
        if proprio_extractor is not None and hasattr(proprio_extractor, "close"):
            proprio_extractor.close()
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
