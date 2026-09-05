"""Evaluate a DSRL checkpoint in one headless robosuite environment."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from robosuite.pipeline.src.dsrl import DSRLInferencePolicy, NetworkConfig
from robosuite.pipeline.src.environment import (
    FlowObservation,
    FlowPolicyAdapter,
    bind_proprio_extractor,
    build_policy_observation,
    build_robosuite_env,
    build_runtime_config,
    observation_batch_to_cuda,
    reset_policy_observation,
)
from robosuite.pipeline.src.vision import DinoV2Encoder
from robosuite.pipeline.utils import file_identity, resolve_cuda_device, set_seed, write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VIDEO_CAMERA = "agentview"
VIDEO_SIZE = 512
VIDEO_FPS = 20


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    episode_return: float
    primitive_length: int
    macro_length: int
    success: bool
    termination_reason: str
    video: str


def _run_directory_for_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.parent.name != "checkpoints":
        raise ValueError(
            "DSRL checkpoint must be stored inside its run's checkpoints directory."
        )
    return checkpoint.parent.parent


def _artifact_path(
    payload: Mapping[str, Any],
    *,
    fingerprint_key: str,
    config_key: str,
) -> Path:
    recorded = payload.get("fingerprints", {}).get(fingerprint_key, {})
    candidates = [recorded.get("path")]
    configured = payload.get("config", {}).get("inputs", {}).get(config_key)
    if configured is not None:
        configured_path = Path(str(configured)).expanduser()
        candidates.append(
            configured_path if configured_path.is_absolute() else REPOSITORY_ROOT / configured_path
        )
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            continue
        if recorded:
            identity = file_identity(path)
            if int(identity["size"]) != int(recorded["size"]):
                continue
            expected_mtime = recorded.get("mtime_ns")
            if expected_mtime is not None and int(identity["mtime_ns"]) != int(expected_mtime):
                continue
            expected_digest = recorded.get("prefix_sha256")
            if expected_digest is not None and identity["prefix_sha256"] != expected_digest:
                continue
        return path
    raise FileNotFoundError(
        f"Cannot locate a checkpoint-compatible {fingerprint_key} artifact."
    )


def _checkpoint_spec(
    checkpoint: str | Path, device_override: str | None
) -> tuple[Path, Mapping[str, Any], str, int, torch.device, NetworkConfig, Path, Path]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DSRL checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("DSRL checkpoint must contain a mapping.")
    config = payload.get("config")
    counters = payload.get("counters")
    agent = payload.get("trainer", {}).get("agent")
    if (
        not isinstance(config, Mapping)
        or not isinstance(counters, Mapping)
        or not isinstance(agent, Mapping)
    ):
        raise KeyError("DSRL checkpoint is missing config, counters, or trainer.agent.")
    task = config.get("task", {}).get("name")
    if not task:
        raise KeyError("DSRL checkpoint config is missing task.name.")
    completed_episodes = int(counters.get("completed_episodes", -1))
    if completed_episodes < 0:
        raise ValueError("DSRL checkpoint has an invalid completed episode counter.")
    agent_config = agent.get("config")
    if not isinstance(agent_config, Mapping) or not isinstance(
        agent_config.get("network"), Mapping
    ):
        raise KeyError("DSRL checkpoint is missing the network configuration.")
    missing_inference_state = [
        name for name in ("bottleneck", "actor") if not isinstance(agent.get(name), Mapping)
    ]
    if missing_inference_state:
        raise KeyError(
            f"DSRL checkpoint is missing inference state: {missing_inference_state}."
        )
    network = NetworkConfig(**dict(agent_config["network"]))
    network.validate()
    requested_device = device_override or config.get("runtime", {}).get("inference_device")
    if not requested_device:
        raise KeyError("DSRL checkpoint config is missing runtime.inference_device.")
    device = resolve_cuda_device(str(requested_device))
    flow_path = _artifact_path(
        payload, fingerprint_key="flow", config_key="base_policy_checkpoint"
    )
    dino_path = _artifact_path(
        payload, fingerprint_key="dinov2", config_key="dinov2_checkpoint"
    )
    return path, payload, str(task), completed_episodes, device, network, flow_path, dino_path


def _flow_task_settings(
    flow_checkpoint: Path, task: str, prompt_index: int
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    payload = torch.load(flow_checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("task_metadata_map", {}).get(task)
    prompts = payload.get("task_prompt_map", {}).get(task)
    cameras = payload.get("camera_names")
    if not isinstance(metadata, Mapping):
        raise KeyError(f"Flow checkpoint has no environment metadata for {task}.")
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, Sequence) or not prompts:
        raise KeyError(f"Flow checkpoint has no prompt for {task}.")
    if not 0 <= int(prompt_index) < len(prompts):
        raise IndexError(f"Flow prompt index {prompt_index} is invalid for {task}.")
    if not isinstance(cameras, Sequence) or not cameras:
        raise KeyError("Flow checkpoint has no camera names.")
    return (
        dict(metadata),
        tuple(str(name) for name in cameras),
        str(prompts[int(prompt_index)]),
    )


def _success(env: Any, info: Mapping[str, Any] | None) -> bool:
    if info and bool(info.get("success", info.get("is_success", False))):
        return True
    check = getattr(env, "_check_success", None)
    return bool(check()) if callable(check) else False


def _frame(env: Any, *, camera: str, size: int = 512) -> np.ndarray:
    image = env.sim.render(height=int(size), width=int(size), camera_name=camera)
    return np.ascontiguousarray(np.flipud(np.asarray(image, dtype=np.uint8)))


def _video_writer(path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        macro_block_size=1,
    )


def rollout_episode(
    *,
    env: Any,
    initial_observation: Mapping[str, np.ndarray],
    select_action_chunk: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    build_observation: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
    max_steps: int,
    append_frame: Callable[[], None],
) -> tuple[float, int, int, bool, str]:
    """Run one episode while preserving action-chunk termination semantics."""

    observation = initial_observation
    episode_return = 0.0
    primitive_steps = 0
    macro_steps = 0
    success = False
    reason = "horizon"
    while primitive_steps < int(max_steps):
        action_chunk = np.asarray(select_action_chunk(observation), dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"Policy action chunk must be rank two, got {action_chunk.shape}.")
        if action_chunk.shape[0] == 0:
            raise ValueError("Policy action chunk must contain at least one action.")
        macro_steps += 1
        for action in action_chunk:
            append_frame()
            result = env.step(action)
            if len(result) == 5:
                raw_observation, _, terminated, truncated, raw_info = result
            else:
                raw_observation, _, terminated, raw_info = result
                truncated = False
            info = raw_info if isinstance(raw_info, Mapping) else None
            primitive_steps += 1
            success = _success(env, info)
            episode_return += 0.0 if success else -1.0
            observation = build_observation(raw_observation)
            if success:
                reason = "success"
            elif bool(truncated) or primitive_steps >= int(max_steps):
                reason = "horizon"
            elif bool(terminated):
                reason = "environment"
            else:
                continue
            append_frame()
            return episode_return, primitive_steps, macro_steps, success, reason
    append_frame()
    return episode_return, primitive_steps, macro_steps, success, reason


def _publish_directory(temporary: Path, destination: Path) -> None:
    if destination.parent.is_symlink():
        raise RuntimeError(
            f"Refusing to publish through symlinked eval directory: {destination.parent}"
        )
    if destination.parent.name != "eval" or re.fullmatch(
        r"episode_\d{8}", destination.name
    ) is None:
        raise ValueError(f"Unsafe evaluation output path: {destination}")
    if temporary.parent != destination.parent or not temporary.name.startswith(
        f".{destination.name}.tmp-"
    ):
        raise ValueError(f"Unsafe temporary evaluation path: {temporary}")
    if destination.is_symlink():
        raise RuntimeError(f"Refusing to overwrite symlinked evaluation directory: {destination}")
    stale = None
    if destination.exists():
        stale = destination.parent / f".{destination.name}.stale-{uuid.uuid4().hex}"
        destination.replace(stale)
    try:
        temporary.replace(destination)
    except BaseException:
        if stale is not None and not destination.exists():
            stale.replace(destination)
        raise
    if stale is not None:
        shutil.rmtree(stale, ignore_errors=True)


def evaluate(args: argparse.Namespace) -> Path:
    if int(args.num_episodes) <= 0 or int(args.max_steps) <= 0:
        raise ValueError("num-episodes and max-steps must be positive.")
    checkpoint, payload, task, checkpoint_episode, device, network, flow_path, dino_path = (
        _checkpoint_spec(args.checkpoint, args.device)
    )
    config = payload["config"]
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompt_index = int(config.get("flow", {}).get("prompt_index", 0))
    env_metadata, camera_names, prompt = _flow_task_settings(
        flow_path, task, prompt_index
    )
    configured_environment = config.get("environment", {}).get("name")
    if configured_environment is not None and str(configured_environment) != task:
        raise ValueError(
            f"Checkpoint task mismatch: task.name is {task}, "
            f"environment.name is {configured_environment}."
        )
    flow_environment = env_metadata.get("env_name")
    if flow_environment is not None and str(flow_environment) != task:
        raise ValueError(
            f"Checkpoint task mismatch: task.name is {task}, "
            f"flow env_name is {flow_environment}."
        )
    configured_cameras = tuple(
        str(name) for name in config.get("environment", {}).get("camera_names", ())
    )
    if configured_cameras and configured_cameras != camera_names:
        raise ValueError(
            f"Checkpoint camera mismatch: config has {configured_cameras}, "
            f"flow has {camera_names}."
        )
    flow_config = config.get("flow", {})
    if int(flow_config.get("action_horizon", network.action_horizon)) != network.action_horizon:
        raise ValueError("Checkpoint flow and actor action horizons do not match.")
    if int(flow_config.get("action_dim", network.action_dim)) != network.action_dim:
        raise ValueError("Checkpoint flow and actor action dimensions do not match.")
    image_height = int(config.get("environment", {}).get("image_height", 128))
    image_width = int(config.get("environment", {}).get("image_width", 128))
    control_frequency = int(config.get("environment", {}).get("control_frequency", 20))
    ode_steps = int(config.get("flow", {}).get("ode_steps", 10))
    run_directory = _run_directory_for_checkpoint(checkpoint)
    eval_directory = run_directory / "eval"
    if eval_directory.is_symlink():
        raise RuntimeError(f"Refusing to use symlinked eval directory: {eval_directory}")
    eval_directory.mkdir(parents=True, exist_ok=True)
    output_directory = eval_directory / f"episode_{checkpoint_episode:08d}"
    temporary = output_directory.parent / f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()

    started = time.time()
    episodes: list[EpisodeResult] = []
    try:
        with ExitStack() as resources:
            runtime_config = build_runtime_config(
                env_metadata,
                camera_names=camera_names,
                image_height=image_height,
                image_width=image_width,
                control_freq=control_frequency,
                horizon=int(args.max_steps),
                interactive=False,
            )
            env = build_robosuite_env(runtime_config)
            resources.callback(env.close)
            extractor = bind_proprio_extractor(env, env_metadata)
            resources.callback(extractor.close)
            action_low, action_high = (
                np.asarray(value, dtype=np.float32) for value in env.action_spec
            )
            flow = FlowPolicyAdapter(
                flow_path,
                device,
                expected_camera_names=camera_names,
                expected_action_horizon=network.action_horizon,
                expected_action_dim=network.action_dim,
                ode_steps=ode_steps,
                image_size=image_height,
                action_low=action_low,
                action_high=action_high,
            )
            dino = DinoV2Encoder(dino_path, device)
            policy = DSRLInferencePolicy(network, device)
            policy.load_inference_state(payload["trainer"]["agent"])

            def select_action_chunk(observation: Mapping[str, np.ndarray]) -> np.ndarray:
                images, proprio = observation_batch_to_cuda(
                    observation, camera_names, "state", device
                )
                dino_features = dino(images)
                normalized_proprio = flow.normalize_proprio(proprio)
                context = flow.encode_context(FlowObservation(images, proprio, prompt))
                latent = policy.latent(
                    dino_features, normalized_proprio, deterministic=True
                )
                return flow.decode_noise(context, latent).cpu().numpy()[0]

            def build_observation(raw: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
                return build_policy_observation(
                    env,
                    extractor=extractor,
                    camera_names=camera_names,
                    camera_aliases={},
                    image_height=image_height,
                    image_width=image_width,
                    raw_observation=raw,
                )

            for episode_index in range(int(args.num_episodes)):
                observation, _ = reset_policy_observation(
                    env,
                    preserve_mjviewer=False,
                    extractor=extractor,
                    camera_names=camera_names,
                    camera_aliases={},
                    image_height=image_height,
                    image_width=image_width,
                )
                relative_video = Path("videos") / f"episode_{episode_index:04d}.mp4"
                with _video_writer(temporary / relative_video, VIDEO_FPS) as writer:
                    result = rollout_episode(
                        env=env,
                        initial_observation=observation,
                        select_action_chunk=select_action_chunk,
                        build_observation=build_observation,
                        max_steps=int(args.max_steps),
                        append_frame=lambda: writer.append_data(
                            _frame(env, camera=VIDEO_CAMERA, size=VIDEO_SIZE)
                        ),
                    )
                episode_return, primitive_length, macro_length, success, reason = result
                episodes.append(
                    EpisodeResult(
                        episode=episode_index,
                        episode_return=episode_return,
                        primitive_length=primitive_length,
                        macro_length=macro_length,
                        success=success,
                        termination_reason=reason,
                        video=os.fspath(relative_video),
                    )
                )
                print(
                    f"[eval] episode={episode_index + 1}/{args.num_episodes} "
                    f"success={int(success)} length={primitive_length} "
                    f"return={episode_return:.1f}"
                )

        successes = sum(int(item.success) for item in episodes)
        summary = {
            "format": "dsrl_eval_v1",
            "task": task,
            "checkpoint": os.fspath(checkpoint),
            "checkpoint_episode": checkpoint_episode,
            "flow_checkpoint": os.fspath(flow_path),
            "dinov2_checkpoint": os.fspath(dino_path),
            "device": str(device),
            "seed": seed,
            "deterministic": True,
            "num_episodes": len(episodes),
            "max_episode_steps": int(args.max_steps),
            "evaluation": {
                "deterministic": True,
                "num_episodes": len(episodes),
                "max_episode_steps": int(args.max_steps),
                "video_camera": VIDEO_CAMERA,
                "video_height": VIDEO_SIZE,
                "video_width": VIDEO_SIZE,
                "video_fps": VIDEO_FPS,
                "video_codec": "h264",
            },
            "success_count": successes,
            "success_rate": successes / max(1, len(episodes)),
            "started_at": started,
            "finished_at": time.time(),
            "episodes": [
                {
                    "episode": item.episode,
                    "return": item.episode_return,
                    "primitive_length": item.primitive_length,
                    "macro_length": item.macro_length,
                    "success": item.success,
                    "termination_reason": item.termination_reason,
                    "video": item.video,
                }
                for item in episodes
            ],
        }
        write_json(temporary / "summary.json", summary)
        _publish_directory(temporary, output_directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"[output] {output_directory}")
    return output_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.num_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("num-episodes and max-steps must be positive.")
    evaluate(args)


if __name__ == "__main__":
    main()


__all__ = ["EpisodeResult", "evaluate", "rollout_episode"]
