"""Helper functions and classes for the Flow-DAgger online training entry point.

These are the top-level helpers that used to live in ``train_flow_dagger.py``. They
are extracted here verbatim to keep the training script focused on its ``main()``
control loop. Logic is unchanged; only the location moved.
"""

from __future__ import annotations

import datetime
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import robosuite.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.envs.robosuite import build_runtime_config_from_env_info
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, camera_obs_key
from robosuite.pipeline.utils.train_utils import (
    checkpoint_path,
    now_readable,
    reset_robosuite_env,
    resolve_checkpoint_reference,
    resolve_requested_device,
    set_seed,
)


def resolve_base_policy_directory(output_root: Path) -> Path:
    return output_root / "base_policy"


def make_flow_run_directory(root_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(root_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_set_seed(seed_value: Any) -> None:
    if seed_value is None:
        print("[INFO] Running without a fixed random seed.")
        return
    normalized = str(seed_value).strip().lower()
    if normalized in {"", "none", "null"}:
        print("[INFO] Running without a fixed random seed.")
        return
    set_seed(int(seed_value))


def serialize_seed(seed_value: Any) -> int | None:
    if seed_value is None:
        return None
    normalized = str(seed_value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(seed_value)


def format_base_policy_trajectory_tag(max_num_trajectories: int | None) -> str:
    if max_num_trajectories is None:
        return "all"
    return f"{int(max_num_trajectories):05d}"


def resolve_base_policy_checkpoint_path(
    output_root: Path,
    *,
    env_name: str,
    demo_source_name: str,
    max_num_trajectories: int | None,
    pretrain_steps: int,
) -> Path:
    trajectory_tag = format_base_policy_trajectory_tag(max_num_trajectories)
    checkpoint_name = (
        f"{env_name}__{demo_source_name}__traj_{trajectory_tag}__pretrain_{int(pretrain_steps):08d}.pt"
    )
    return resolve_base_policy_directory(output_root) / checkpoint_name


def resolve_run_directory(cfg: DictConfig) -> tuple[str, Path]:
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    explicit_run_name = cfg.logging.run_name

    if explicit_run_name is not None:
        run_name = str(explicit_run_name)
        return run_name, make_flow_run_directory(output_root, run_name)

    run_name = f"flow_dagger_{cfg.env.environment}_{now_readable()}"
    run_name_suffix = getattr(cfg.logging, "run_name_suffix", None)
    if run_name_suffix is not None and str(run_name_suffix).strip():
        run_name = f"{run_name}_{str(run_name_suffix).strip().lstrip('_')}"
    return run_name, make_flow_run_directory(output_root, run_name)


def find_latest_resumable_run(output_root: Path, env_name: str) -> Path | None:
    env_prefix = f"flow_dagger_{env_name}_"
    candidates = []
    for latest_checkpoint in output_root.glob("*/checkpoints/latest.pt"):
        run_dir = latest_checkpoint.parent.parent
        if run_dir.name.startswith(env_prefix):
            candidates.append((latest_checkpoint.stat().st_mtime, run_dir))
    if not candidates:
        for latest_checkpoint in output_root.glob("*/checkpoints/latest.pt"):
            candidates.append((latest_checkpoint.stat().st_mtime, latest_checkpoint.parent.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resume_checkpoint_candidates(cfg: DictConfig, output_root: Path) -> list[Path]:
    candidates: list[Path] = []

    explicit_checkpoint = resolve_checkpoint_reference(cfg.runtime.checkpoint)
    if explicit_checkpoint is not None:
        candidates.append(explicit_checkpoint)
    elif bool(cfg.runtime.resume):
        resumable_run = find_latest_resumable_run(output_root, str(cfg.env.environment))
        if resumable_run is not None:
            latest = checkpoint_path(resumable_run, "latest")
            if latest.exists():
                candidates.append(latest)
            candidates.extend(sorted((resumable_run / "checkpoints").glob("step_*.pt"), reverse=True))
            candidates.extend(sorted((resumable_run / "checkpoints").glob("pretrain_*.pt"), reverse=True))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def resolve_algorithm_devices(algorithm_cfg: dict[str, Any]) -> tuple[str, str]:
    flow_cfg = algorithm_cfg.setdefault("flow", {})
    learner_requested = flow_cfg.get("device", "cpu")
    inference_requested = flow_cfg.get("inference_device", None)
    normalized_learner_request = "cpu" if learner_requested is None else str(learner_requested).strip().lower()

    default_learner = "cuda:0" if torch.cuda.is_available() else "cpu"
    learner_device = resolve_requested_device(learner_requested, fallback=default_learner)

    if inference_requested is None or str(inference_requested).lower() == "auto":
        if normalized_learner_request in {"cuda", "cuda:0"} and torch.cuda.device_count() >= 2:
            learner_device = "cuda:1"
            inference_device = "cuda:0"
        elif learner_device.startswith("cuda"):
            inference_device = "cpu"
        else:
            inference_device = learner_device
    else:
        explicit_inference_request = str(inference_requested).strip().lower()
        inference_fallback = "cpu" if learner_device.startswith("cuda") else learner_device
        inference_device = resolve_requested_device(inference_requested, fallback=inference_fallback)
        if explicit_inference_request.startswith("cuda") and inference_device == learner_device and learner_device.startswith("cuda"):
            inference_device = "cpu"

    flow_cfg["device"] = learner_device
    flow_cfg["inference_device"] = inference_device
    return learner_device, inference_device


def format_runtime_line(
    *,
    step: int,
    episode_index: int,
    overall_fps: float,
    learner_progress: dict[str, int],
    pending_updates: int,
) -> str:
    return (
        f"[runtime] step={step} ep={episode_index} fps={overall_fps:5.1f} "
        f"flow_updates={learner_progress['actor_updates']} "
        f"next_publish_in={learner_progress['updates_until_publish']} pending={pending_updates}"
    )


def format_train_line(step: int, metrics: dict[str, float], pending_updates: int) -> str:
    return (
        f"[train] step={step} "
        f"loss={metrics.get('actor_loss', float('nan')):.4f} "
        f"flow={metrics.get('flow_loss', float('nan')):.4f} "
        f"endpoint={metrics.get('endpoint_loss', float('nan')):.4f} "
        f"smooth={metrics.get('smooth_loss', float('nan')):.4f} "
        f"flow_updates={int(metrics.get('learner_actor_updates', 0.0))} "
        f"next_publish_in={int(metrics.get('learner_updates_until_publish', 0.0))} pending={pending_updates}"
    )


def format_publish_line(metrics: dict[str, float]) -> str:
    return (
        f"[publish] policy #{int(metrics.get('learner_publish_count', 0.0))} synced "
        f"at learner_update={int(metrics.get('learner_last_published_update', 0.0))}"
    )


class AsyncDemoTransitionChunkWriter:
    def __init__(self, output_dir: Path, *, chunk_size: int, event_logger: callable | None = None) -> None:
        self.output_dir = output_dir
        self.chunk_size = max(1, int(chunk_size))
        self.event_logger = event_logger
        self._condition = threading.Condition()
        self._pending: deque[Any] = deque()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._flush_requested = False
        self._busy = False
        self._error: BaseException | None = None
        self._chunk_index = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._worker_loop, name="flow_dagger_demo_writer", daemon=True)
            self._thread.start()

    def request_transition(self, transition) -> None:
        self._raise_error()
        with self._condition:
            self._pending.append(transition)
            self._condition.notify_all()

    def flush(self, timeout: float | None = None) -> None:
        self._raise_error()
        with self._condition:
            if self._thread is None:
                return
            self._flush_requested = True
            self._condition.notify_all()
            end_time = None if timeout is None else time.monotonic() + float(timeout)
            while self._pending or self._busy or self._flush_requested:
                self._raise_error()
                remaining = None if end_time is None else max(0.0, end_time - time.monotonic())
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError("Timed out while waiting for demo transition chunk writer.")
                self._condition.wait(timeout=0.1 if remaining is None else min(0.1, remaining))
        self._raise_error()

    def close(self) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stop_requested = True
            self._condition.notify_all()
        thread.join(timeout=10.0)
        with self._condition:
            self._thread = None
        self._raise_error()

    def _worker_loop(self) -> None:
        batch: list[Any] = []
        while True:
            with self._condition:
                while len(self._pending) == 0 and not self._flush_requested and not self._stop_requested:
                    self._condition.wait(timeout=0.1)
                while self._pending:
                    batch.append(self._pending.popleft())
                should_flush = self._flush_requested or self._stop_requested
                if self._stop_requested and len(batch) == 0:
                    return
                if should_flush and len(batch) == 0:
                    self._flush_requested = False
                    self._condition.notify_all()
                    continue
                should_write = len(batch) >= self.chunk_size or (should_flush and len(batch) > 0)
                if not should_write:
                    continue
                self._busy = True
                self._condition.notify_all()
            try:
                while len(batch) >= self.chunk_size or (should_flush and len(batch) > 0):
                    chunk = batch[: self.chunk_size]
                    del batch[: len(chunk)]
                    self._write_chunk(self._chunk_index, chunk)
                    self._chunk_index += 1
                    if not should_flush:
                        break
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._stop_requested = True
                    self._busy = False
                    self._condition.notify_all()
                return
            finally:
                with self._condition:
                    if should_flush and len(batch) == 0:
                        self._flush_requested = False
                    self._busy = False
                    self._condition.notify_all()

    def _write_chunk(self, chunk_index: int, transitions: list[Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = self.output_dir / f"chunk_{int(chunk_index):08d}.pt"
        tmp_path = chunk_path.with_name(f".{chunk_path.name}.tmp")
        payload = {
            "chunk_index": int(chunk_index),
            "transition_count": int(len(transitions)),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "transitions": list(transitions),
        }
        torch.save(payload, tmp_path)
        tmp_path.replace(chunk_path)
        if self.event_logger is not None:
            self.event_logger(
                {
                    "event": "buffer_chunk_written",
                    "stream": "demo_chunks",
                    "chunk_index": int(chunk_index),
                    "transition_count": int(len(transitions)),
                    "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
                }
            )

    def _raise_error(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Demo transition chunk writer failed: {error}") from error


def load_init_checkpoint_payload(cfg: DictConfig) -> tuple[Path | None, dict[str, Any] | None]:
    init_checkpoint_cfg = getattr(cfg.runtime, "init_checkpoint", None)
    init_checkpoint = resolve_checkpoint_reference(init_checkpoint_cfg)
    if init_checkpoint is None or not init_checkpoint.exists():
        return None, None
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    return init_checkpoint, payload


def resolve_flow_task_metadata(init_payload: dict[str, Any] | None, task_name: str) -> dict[str, Any] | None:
    if not init_payload:
        return None
    task_metadata_map = init_payload.get("task_metadata_map", None)
    if isinstance(task_metadata_map, dict):
        if task_name in task_metadata_map:
            return dict(task_metadata_map[task_name])
        if len(task_metadata_map) == 1:
            return dict(next(iter(task_metadata_map.values())))
    env_metadata = init_payload.get("env_metadata", None)
    if env_metadata is not None:
        return dict(env_metadata)
    return None


def bind_flow_proprio_extractor(env, env_metadata: dict[str, Any] | None = None) -> RobosuiteProprioExtractor:
    _ = env_metadata
    extractor = RobosuiteProprioExtractor.__new__(RobosuiteProprioExtractor)
    extractor.env = env
    extractor.sim = env.sim
    extractor._build_robot_joint_indices()
    return extractor


def normalize_policy_observation(
    obs: dict[str, Any],
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
) -> dict[str, Any]:
    normalized_obs = {"state": np.asarray(obs["state"], dtype=np.float32)}
    for camera_name in policy_camera_names:
        if camera_name in obs:
            normalized_obs[camera_name] = np.asarray(obs[camera_name], dtype=np.uint8)
            continue
        alias_source = camera_aliases.get(camera_name)
        if alias_source is None or alias_source not in obs:
            raise KeyError(
                f"Unable to resolve observation camera '{camera_name}'. "
                f"Available keys: {sorted(obs.keys())}, aliases: {camera_aliases}"
            )
        normalized_obs[camera_name] = np.asarray(obs[alias_source], dtype=np.uint8)
    return normalized_obs


def convert_env_camera_observation(
    raw_obs: dict[str, Any],
    *,
    env,
    extractor: RobosuiteProprioExtractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
) -> dict[str, Any]:
    converted_obs: dict[str, Any] = {}
    for camera_name in policy_camera_names:
        source_camera = camera_aliases.get(camera_name, camera_name)
        source_key = camera_obs_key(source_camera)
        if source_key not in raw_obs:
            raise KeyError(
                f"Unable to resolve env camera observation for '{camera_name}'. "
                f"Expected key '{source_key}' in env obs keys {sorted(raw_obs.keys())}."
            )
        converted_obs[camera_name] = _center_crop_resize_image(
            np.asarray(raw_obs[source_key], dtype=np.uint8),
            img_height=img_height,
            img_width=img_width,
        )
    converted_obs["state"] = extractor.extract(env.sim.get_state().flatten()).astype(np.float32)
    return converted_obs


def extract_flow_state(env, extractor: RobosuiteProprioExtractor) -> np.ndarray:
    return extractor.extract(env.sim.get_state().flatten()).astype(np.float32)


def render_policy_camera_images(
    env,
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
) -> dict[str, np.ndarray]:
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    images: dict[str, np.ndarray] = {}
    for camera_name in policy_camera_names:
        source_camera = camera_aliases.get(camera_name, camera_name)
        frame = env.sim.render(
            height=int(img_height),
            width=int(img_width),
            camera_name=source_camera,
        )
        frame = np.asarray(frame[::convention], dtype=np.uint8)
        images[camera_name] = _center_crop_resize_image(
            frame,
            img_height=img_height,
            img_width=img_width,
        )
    return images


def build_live_policy_observation(
    env,
    *,
    extractor: RobosuiteProprioExtractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    include_images: bool,
) -> dict[str, Any]:
    obs: dict[str, Any] = {"state": extract_flow_state(env, extractor)}
    if include_images:
        obs.update(
            render_policy_camera_images(
                env,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=img_height,
                img_width=img_width,
            )
        )
    return obs


def reset_flow_policy_observation(
    env,
    *,
    preserve_mjviewer: bool,
    extractor: RobosuiteProprioExtractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_obs, info = reset_robosuite_env(env, preserve_mjviewer=preserve_mjviewer)
    return (
        convert_env_camera_observation(
            raw_obs,
            env=env,
            extractor=extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=img_height,
            img_width=img_width,
        ),
        info,
    )


def build_flow_runtime_cfg(
    cfg: DictConfig,
    *,
    env_metadata: dict[str, Any] | None,
    camera_names: list[str],
    has_renderer: bool,
    has_offscreen_renderer: bool,
    use_camera_obs: bool = False,
    renderer: str | None = None,
):
    if env_metadata is None:
        raise ValueError("flow-dagger requires env metadata to build a flow-aligned runtime config.")

    runtime_cfg = build_runtime_config_from_env_info(
        env_metadata,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        has_renderer=has_renderer,
        renderer=str(cfg.env.renderer) if renderer is None else str(renderer),
        reward_shaping=False,
        control_freq=int(cfg.env.control_freq),
    )
    if cfg.env.horizon is not None:
        runtime_cfg.horizon = int(cfg.env.horizon)
    runtime_cfg.use_camera_obs = bool(use_camera_obs)
    if use_camera_obs:
        runtime_cfg.has_offscreen_renderer = True
    return runtime_cfg


def _resolve_hdf5_demo_group(file_handle: h5py.File | h5py.Group) -> h5py.Group:
    if "demos" in file_handle:
        return file_handle["demos"]
    if "data" in file_handle:
        return file_handle["data"]
    raise KeyError("Expected the demo file to contain either a 'demos' group or a 'data' group.")


def _center_crop_resize_image(image: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop.shape[0] == img_height and crop.shape[1] == img_width:
        return np.asarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, img_height).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, img_width).astype(np.int32)
    return np.asarray(crop[ys][:, xs], dtype=np.uint8)


def _center_crop_resize_batch(images: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    # Vectorized version of _center_crop_resize_image over a leading time axis (T, H, W, C).
    height, width = images.shape[1:3]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = images[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop.shape[1] == img_height and crop.shape[2] == img_width:
        return np.ascontiguousarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, img_height).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, img_width).astype(np.int32)
    return np.ascontiguousarray(crop[:, ys][:, :, xs], dtype=np.uint8)


def load_hdf5_demos_into_flow_transitions(
    path: str | Path,
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    proprio_keys: tuple[str, ...],
    renderer: str,
    control_freq: int,
    demo_names: list[str] | None,
    state_extractor: RobosuiteProprioExtractor,
) -> list:
    _ = proprio_keys
    _ = renderer
    _ = control_freq
    transitions: list[Transition] = []
    with h5py.File(Path(path), "r") as file_handle:
        demos_group = _resolve_hdf5_demo_group(file_handle)
        available_demo_names = {str(name) for name in demos_group.keys()}
        if demo_names is None:
            selected_demo_names = sorted(available_demo_names)
        else:
            selected_demo_names = [str(name) for name in demo_names if str(name) in available_demo_names]
        for demo_name in selected_demo_names:
            demo_group = demos_group[demo_name]
            states = np.asarray(demo_group["states"])
            actions = np.asarray(demo_group["actions"], dtype=np.float32)
            if len(states) == 0 or len(actions) == 0:
                continue
            successful = bool(demo_group.attrs.get("successful", False))
            if "intervention_labels" in demo_group:
                intervention_labels = np.asarray(demo_group["intervention_labels"], dtype=np.bool_)
            else:
                intervention_labels = np.zeros(len(actions), dtype=np.bool_)
            obs_group = demo_group["observations"]
            required_hdf5_camera_names = {
                camera_aliases.get(camera_name, camera_name) for camera_name in policy_camera_names
            }
            for camera_name in required_hdf5_camera_names:
                if camera_name not in obs_group:
                    raise KeyError(f"Missing camera '{camera_name}' in {path}:{demo_name}")

            num_steps = len(actions)
            # Vectorized read: pull each camera stream once and crop/resize all frames in one shot,
            # instead of per-frame HDF5 random reads (the dominant bottleneck).
            cropped_images = {
                camera_name: _center_crop_resize_batch(
                    np.asarray(obs_group[camera_name]["images"][:], dtype=np.uint8),
                    img_height=img_height,
                    img_width=img_width,
                )
                for camera_name in required_hdf5_camera_names
            }
            # Extract + normalize each frame exactly once; next_obs reuses the next frame's obs.
            frame_obs = []
            for step_idx in range(num_steps):
                obs_state = state_extractor.extract(states[step_idx]).astype(np.float32)
                frame_obs.append(
                    normalize_policy_observation(
                        {
                            **{name: cropped_images[name][step_idx] for name in required_hdf5_camera_names},
                            "state": obs_state,
                        },
                        policy_camera_names=policy_camera_names,
                        camera_aliases=camera_aliases,
                    )
                )

            for step_idx in range(num_steps):
                next_idx = min(step_idx + 1, num_steps - 1)
                is_last_step = step_idx == num_steps - 1
                reward = 0.0 if successful and is_last_step else -1.0
                transitions.append(
                    Transition(
                        obs=frame_obs[step_idx],
                        action=np.asarray(actions[step_idx], dtype=np.float32),
                        reward=float(reward),
                        next_obs=frame_obs[next_idx],
                        done=bool(is_last_step),
                        grasp_penalty=None,
                        is_intervention=bool(intervention_labels[step_idx]),
                        info={
                            "success": bool(successful),
                            "demo_name": str(demo_name),
                            "reward_convention": "sparse_success_-1_0",
                        },
                        reward_source="offline_sparse_success",
                        demo_source="offline_demo",
                    )
                )
    return transitions
