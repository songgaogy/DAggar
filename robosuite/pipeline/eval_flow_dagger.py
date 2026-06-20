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

from robosuite.pipeline.algorithms.flow_dagger.common import FlowAugmentationConfig, FlowDaggerConfig
from robosuite.pipeline.algorithms.flow_dagger.models import FlowDaggerPolicy
from robosuite.pipeline.envs import build_robosuite_env, sparse_success_reward
from robosuite.pipeline.train_flow_dagger import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import (
    EnvRandomReducer,
    assert_disjoint_seed_ranges,
    resolve_requested_device,
)
DEFAULT_OUTPUT_ROOT = "./outputs/flow-DAgger/eval"
DEFAULT_VIDEO_CAMERA = "agentview"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_SIZE = 512
# Eval layout seeds live in a band deliberately disjoint from training (which uses the low base_seed
# from cfg.seed, e.g. 42). This prevents benchmarking the policy on layouts it trained on. The
# overlap is still hard-asserted at runtime against the training run_info when available.
DEFAULT_EVAL_SEED = 900000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a flow-dagger checkpoint with minimal logging.")
    parser.add_argument("--checkpoint", required=True, help="Path to a flow-dagger checkpoint.")
    parser.add_argument("--env-name", required=True, help="Robosuite env name, e.g. PickPlaceBread.")
    parser.add_argument("--task-name", required=True, help="Demo / task name used for language instruction.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--episode-max-steps", type=int, default=500, help="Per-episode step cap.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory for eval summaries and videos.")
    parser.add_argument("--video-output", default="false", help="Whether to save mp4 videos.")
    parser.add_argument("--video-camera", default=DEFAULT_VIDEO_CAMERA, help="Camera to save for videos.")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS, help="FPS for saved videos.")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_SIZE, help="Saved video frame height.")
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_SIZE, help="Saved video frame width.")
    parser.add_argument("--interactive", action="store_true", help="Open the robosuite viewer during eval.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=(
            "Base seed for deterministic episode layouts. Defaults to a high band disjoint from "
            "training seeds; overlap with the training seed range is hard-asserted at runtime."
        ),
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=False,
        help="Use zero-latent deterministic action sampling instead of seeded stochastic sampling.",
    )
    action_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use seeded stochastic action sampling (default).",
    )
    parser.add_argument("--init-checkpoint", default=None, help="Optional base flow checkpoint for env metadata.")
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=None,
        help="Override checkpoint flow_config.execute_horizon for eval.",
    )
    parser.add_argument(
        "--n-ode-steps",
        type=int,
        default=None,
        help="Override checkpoint flow_config.n_ode_steps for eval.",
    )
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


def _assert_eval_seeds_disjoint(
    run_info: dict[str, Any] | None,
    *,
    eval_base: int,
    eval_count: int,
) -> None:
    """Hard-assert eval layout seeds do not reuse training layout seeds.

    Reads the training run's ``env_reset_seed`` and episode count from run_info.json. If they are
    unavailable (e.g. eval run from a bare checkpoint), warn and skip rather than fail hard.
    """
    if run_info is None:
        print("[determinism][warn] no run_info.json found; skipping train/eval seed-disjoint check.")
        return
    train_base = run_info.get("env_reset_seed")
    if train_base is None:
        print(
            "[determinism][warn] run_info has no env_reset_seed (training was not deterministic); "
            "skipping seed-disjoint check."
        )
        return
    # episode_index is the next/total episode count; layout seeds used = [base, base + count).
    train_count = int(run_info.get("episode_index", 0) or 0)
    if train_count <= 0:
        print("[determinism][warn] run_info episode_index is 0/missing; skipping seed-disjoint check.")
        return
    assert_disjoint_seed_ranges(
        int(train_base), train_count, int(eval_base), int(eval_count), context="train vs eval layouts"
    )
    print(
        f"[determinism] seed-disjoint OK: train=[{train_base}, {int(train_base) + train_count}) "
        f"eval=[{eval_base}, {int(eval_base) + int(eval_count)})"
    )


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
    checkpoint_path: Path,
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

    _ = checkpoint_path
    return None


def _build_eval_output_dir(output_root: Path, checkpoint_path: Path, task_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_tag = checkpoint_path.stem
    output_dir = output_root / f"{task_name}__{checkpoint_tag}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_eval_device(payload: dict[str, Any]) -> str:
    flow_cfg = dict(payload.get("flow_config", {}))
    requested = flow_cfg.get("inference_device") or flow_cfg.get("device") or "cpu"
    fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
    return resolve_requested_device(requested, fallback=fallback)


def _build_policy(
    payload: dict[str, Any],
    *,
    task_name: str,
    device: str,
    execute_horizon: int | None,
    n_ode_steps: int | None,
) -> FlowDaggerPolicy:
    flow_cfg = dict(payload["flow_config"])
    aug_cfg = dict(flow_cfg.get("augmentation", {}) or {})
    config = FlowDaggerConfig(
        action_dim=int(flow_cfg["action_dim"]),
        proprio_dim=int(flow_cfg["proprio_dim"]),
        action_horizon=int(flow_cfg.get("action_horizon", 8)),
        execute_horizon=int(execute_horizon if execute_horizon is not None else flow_cfg.get("execute_horizon", 1)),
        image_size=int(flow_cfg.get("image_size", 128)),
        learning_rate=float(flow_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(flow_cfg.get("weight_decay", 1e-6)),
        grad_clip_norm=float(flow_cfg.get("grad_clip_norm", 1.0)),
        lambda_endpoint=float(flow_cfg.get("lambda_endpoint", 0.5)),
        lambda_smooth=float(flow_cfg.get("lambda_smooth", 0.05)),
        n_ode_steps=int(n_ode_steps if n_ode_steps is not None else flow_cfg.get("n_ode_steps", 8)),
        device=str(device),
        inference_device=str(device),
        task_name=str(payload.get("task_name", task_name)),
        language_instruction=str(payload.get("language_instruction", task_name)),
        augmentation=FlowAugmentationConfig(
            minimal_shift_pad=int(aug_cfg.get("minimal_shift_pad", 2)),
            eye_in_hand_crop_scale=float(aug_cfg.get("eye_in_hand_crop_scale", 0.88)),
        ),
    )
    policy = FlowDaggerPolicy(
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
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
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
        checkpoint_path=checkpoint_path,
        run_info=run_info,
        resolved_cfg=resolved_cfg,
    )
    if init_checkpoint is None:
        raise FileNotFoundError(
            "Unable to resolve the base flow checkpoint for env metadata. "
            "Pass --init-checkpoint explicitly or evaluate from a run directory containing run_info.json."
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
        resolved_cfg = OmegaConf.create(
            {
                "env": {
                    "environment": args.env_name,
                    "renderer": "mjviewer",
                    "img_height": int(checkpoint_payload["flow_config"].get("image_size", 128)),
                    "img_width": int(checkpoint_payload["flow_config"].get("image_size", 128)),
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
        has_renderer=bool(args.interactive),
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer=str(resolved_cfg.env.renderer),
    )
    env = build_robosuite_env(runtime_cfg)
    env_random_reducer = EnvRandomReducer(int(args.seed))
    _assert_eval_seeds_disjoint(run_info, eval_base=int(args.seed), eval_count=int(args.episodes))
    proprio_extractor = bind_flow_proprio_extractor(env, env_metadata)
    eval_device = _resolve_eval_device(checkpoint_payload)
    policy = _build_policy(
        checkpoint_payload,
        task_name=args.task_name,
        device=eval_device,
        execute_horizon=args.execute_horizon,
        n_ode_steps=args.n_ode_steps,
    )

    output_root = Path(to_absolute_path(args.output_root))
    output_dir = _build_eval_output_dir(output_root, checkpoint_path, args.task_name)
    video_dir = output_dir / "videos"
    if video_output:
        video_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    success_count = 0

    try:
        progress = tqdm(range(int(args.episodes)), desc=f"Eval {args.task_name}", dynamic_ncols=True)
        for episode_idx in progress:
            episode_seed = env_random_reducer.prepare_episode(env, int(episode_idx))
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
            if episode_seed is not None:
                EnvRandomReducer.seed_global(int(episode_seed))
            policy.reset_action_chunk()

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

            if video_output and len(frames) > 0:
                video_path = video_dir / (
                    f"ep_{int(episode_idx):03d}_success_{int(episode_success)}_return_{episode_return:.2f}.mp4"
                )
                _write_video(video_path, frames, fps=int(args.video_fps))
                episode_result["video_path"] = str(video_path)

        mean_return = float(np.mean([item["return"] for item in episode_results])) if episode_results else 0.0
        mean_steps = float(np.mean([item["steps"] for item in episode_results])) if episode_results else 0.0
        success_rate = float(success_count) / float(len(episode_results)) if episode_results else 0.0
        summary = {
            "checkpoint": str(checkpoint_path),
            "init_checkpoint": str(init_checkpoint),
            "env_name": str(args.env_name),
            "task_name": str(args.task_name),
            "episodes": int(args.episodes),
            "episode_max_steps": int(args.episode_max_steps),
            "seed": int(args.seed),
            "env_reset_seed_rule": "base_seed+episode_index",
            "policy_seed_rule": "base_seed+episode_index",
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
            "policy_camera_names": list(policy_camera_names),
            "flow_config": asdict(policy.config),
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
        try:
            env.close()
        finally:
            proprio_extractor.close()


if __name__ == "__main__":
    main()
