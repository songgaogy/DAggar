from __future__ import annotations

import datetime
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import h5py
import numpy as np
import torch
import robosuite.macros as macros
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

from robosuite.pipeline.flow_dagger import FlowDaggerTrainer
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.envs import (
    RobosuiteInterventionRuntime,
    RobosuiteViewerRuntime,
    build_device,
    build_robosuite_env,
    choose_viewer_backend,
    snapshot_env_state,
    sparse_success_reward,
)
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, camera_obs_key
from robosuite.pipeline.utils import (
    AsyncCheckpointWriter,
    ConsoleLogCapture,
    EMAFpsTracker,
    EnvRandomReducer,
    FixedRateLimiter,
    IntervalGate,
    JsonlEventLogger,
    checkpoint_path,
    checkpoint_step_path,
    format_episode_line,
    load_demo_paths,
    load_transition_chunks,
    maybe_build_wandb,
    maybe_log,
    maybe_wrap_visualization,
    now_readable,
    resolve_buffer_chunk_dirs,
    resolve_buffer_snapshot_paths,
    resolve_camera_names,
    resolve_checkpoint_reference,
    resolve_checkpoint_run_dir,
    resolve_demo_inputs,
    resolve_render_camera,
    resolve_requested_device,
    resolve_runtime_fps,
    set_seed,
    write_resolved_config,
    write_run_info,
)
from robosuite.pipeline.utils.train_utils import reset_robosuite_env
from robosuite.pipeline.envs.robosuite import build_runtime_config_from_env_info


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


@hydra.main(version_base="1.2", config_path="./config", config_name="train_flow_dagger")
def main(cfg: DictConfig) -> None:
    maybe_set_seed(getattr(cfg, "seed", None))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    task_name = str(cfg.env.environment)
    requested_camera_names = resolve_camera_names(cfg)
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    render_camera_names = list(policy_camera_names)
    cfg.algorithm.camera_names = list(policy_camera_names)
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.flow, "camera_aliases", {}) or {}).items()
    }
    run_name, checkpoint_dir = resolve_run_directory(cfg)
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    run_started_at = datetime.datetime.now().isoformat(timespec="seconds")
    run_started_monotonic = time.monotonic()
    console_log_path = checkpoint_dir / "console.log"
    runtime_log_path = checkpoint_dir / "metrics_runtime.jsonl"
    console_capture = ConsoleLogCapture(console_log_path)
    console_capture.start()
    runtime_logger = JsonlEventLogger(runtime_log_path)
    runtime_logger.start()
    runtime_logger.log(
        {
            "event": "run_start",
            "run_name": run_name,
            "run_dir": str(checkpoint_dir),
            "started_at": run_started_at,
            "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "run_elapsed_sec": 0.0,
            "training_elapsed_sec": 0.0,
        }
    )
    save_demo_buffer = bool(getattr(cfg.runtime, "save_demo_buffer", True))
    demo_buffer_writer: AsyncDemoTransitionChunkWriter | None = None
    if save_demo_buffer:
        buffer_save_interval = max(1, int(getattr(cfg.runtime, "buffer_save_interval", 200)))
        demo_buffer_writer = AsyncDemoTransitionChunkWriter(
            checkpoint_dir / "buffers" / "demo_chunks",
            chunk_size=buffer_save_interval,
            event_logger=runtime_logger.log,
        )
        demo_buffer_writer.start()
    else:
        print("[INFO] Intervention demo buffer saving is disabled; only checkpoints will be saved.")
    print(f"[run] {run_name}")
    print(f"[path] {checkpoint_dir}")
    print(
        f"[env] task={cfg.env.environment} robots={list(cfg.env.robots)} "
        f"render_cameras={render_camera_names} policy_cameras={policy_camera_names}"
    )
    print(f"[view] render_camera={resolve_render_camera(cfg, render_camera_names)}")
    visualize_gripper_markers = bool(getattr(cfg.runtime, "visualize_gripper_markers", True))
    frozen_eval_mode = (
        (not bool(getattr(cfg.runtime, "online_updates_enabled", True)))
        and (not bool(cfg.intervention.enabled))
    )
    if len(render_camera_names) == 0:
        raise ValueError("flow-dagger requires at least one policy camera.")

    online_updates_enabled = bool(getattr(cfg.runtime, "online_updates_enabled", True))
    viewer_enabled = bool(cfg.runtime.interactive) and bool(cfg.runtime.viewer_enabled)
    on_demand_image_obs = bool(getattr(cfg.runtime, "on_demand_image_obs", True))
    store_non_intervention_images = bool(getattr(cfg.runtime, "store_non_intervention_images", False))
    store_next_obs_images = bool(getattr(cfg.runtime, "store_next_obs_images", False))
    decoupled_viewer_enabled = viewer_enabled and online_updates_enabled
    rollout_has_renderer = viewer_enabled and (not decoupled_viewer_enabled)
    main_renderer = str(cfg.env.renderer)
    if viewer_enabled and main_renderer != "mjviewer":
        print(
            f"[WARN] Overriding env.renderer='{main_renderer}' to 'mjviewer' so the main training env uses "
            "robosuite's native window renderer."
        )
        main_renderer = "mjviewer"
    main_runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=render_camera_names,
        has_renderer=rollout_has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=not on_demand_image_obs,
        renderer=main_renderer,
    )
    if visualize_gripper_markers:
        print("[INFO] Disabling gripper visualization markers on the main env because policy images use its cameras.")
    main_env = build_robosuite_env(main_runtime_cfg)
    main_env = maybe_wrap_visualization(
        main_env,
        enabled=False,
        label="training env",
    )
    if on_demand_image_obs:
        print("[INFO] Main env camera observations are disabled; policy images will be rendered on demand.")
    else:
        print("[INFO] Using camera observations directly from the main env.")

    flow_proprio_extractor = bind_flow_proprio_extractor(main_env, flow_env_metadata)
    if on_demand_image_obs:
        _, _ = reset_robosuite_env(main_env, preserve_mjviewer=rollout_has_renderer)
        initial_obs = build_live_policy_observation(
            main_env,
            extractor=flow_proprio_extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            include_images=True,
        )
    else:
        initial_obs, _ = reset_flow_policy_observation(
            main_env,
            preserve_mjviewer=rollout_has_renderer,
            extractor=flow_proprio_extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
        )
    if rollout_has_renderer and getattr(main_env, "viewer", None) is not None and hasattr(main_env.viewer, "update"):
        main_env.viewer.update()

    action_low, action_high = main_env.action_spec
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if isinstance(algorithm_cfg, dict):
        algorithm_cfg["camera_names"] = list(policy_camera_names)
        algorithm_cfg["task_name"] = task_name
        flow_cfg = algorithm_cfg.setdefault("flow", {})
        flow_cfg.setdefault("image_size", int(cfg.env.img_height))
        if bool(getattr(cfg.runtime, "use_init_checkpoint_model", True)) and init_payload is not None:
            if "model_cfg" in init_payload:
                flow_cfg["model"] = init_payload["model_cfg"]
            if "task_prompt_map" in init_payload:
                flow_cfg["task_prompt_map"] = init_payload["task_prompt_map"]
            if init_payload.get("act_mean") is not None:
                flow_cfg["action_horizon"] = int(np.asarray(init_payload["act_mean"]).shape[0])
                flow_cfg.setdefault("execute_horizon", 1)
        model_cfg = flow_cfg.setdefault("model", {})
        image_encoder_cfg = model_cfg.get("image_encoder", None)
        if isinstance(image_encoder_cfg, dict) and image_encoder_cfg.get("pretrained_path"):
            image_encoder_cfg["pretrained_path"] = to_absolute_path(str(image_encoder_cfg["pretrained_path"]))
        language_encoder_cfg = model_cfg.get("language_encoder", None)
        if isinstance(language_encoder_cfg, dict) and language_encoder_cfg.get("pretrained_name"):
            language_encoder_cfg["pretrained_name"] = to_absolute_path(str(language_encoder_cfg["pretrained_name"]))
        learner_device, inference_device = resolve_algorithm_devices(algorithm_cfg)
        print(f"[INFO] Learner device: {learner_device}")
        print(f"[INFO] Inference device: {inference_device}")
        if learner_device.startswith("cuda") and inference_device == "cpu":
            print("[INFO] Inference policy is running on CPU to reduce render-time stutter on single-GPU setups.")

    agent = build_algorithm(
        algorithm_cfg,
        observation_example=initial_obs,
        sample_action=np.zeros_like(action_low, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    checkpoint_writer = AsyncCheckpointWriter(agent, checkpoint_dir)
    checkpoint_writer.start()
    trainer = FlowDaggerTrainer(agent)

    demo_source_name, demo_paths, max_num_trajectories = resolve_demo_inputs(cfg)
    if not demo_paths:
        raise FileNotFoundError(
            "flow-dagger requires offline demos before training starts. "
            f"No demo files were found for '{demo_source_name}'. "
            "Place demos under ./data/<task>/expert or set data.demo_paths explicitly."
        )

    env = main_env
    env_random_reducer = EnvRandomReducer(serialize_seed(getattr(cfg, "seed", None)))
    if env_random_reducer.enabled:
        print(f"[determinism] env_reset_seed={env_random_reducer.base_seed} rule=base_seed+episode_index")
    obs = initial_obs
    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    render_fps = resolve_runtime_fps(cfg, "render_fps", control_fps)
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    image_obs_fps = resolve_runtime_fps(cfg, "image_obs_fps", control_fps)
    fps_log_interval = max(0.1, float(getattr(cfg.runtime, "fps_log_interval", 1.0)))
    unthrottled_runtime = bool(getattr(cfg.runtime, "unthrottled", False))
    async_updates = bool(getattr(cfg.runtime, "async_updates", False))
    eval_episode_max_steps = int(getattr(cfg.runtime, "eval_episode_max_steps", 300))
    viewer_runtime: RobosuiteViewerRuntime | None = None
    if decoupled_viewer_enabled:
        viewer_requested_backend = str(getattr(cfg.runtime, "viewer_backend", "auto"))
        if viewer_requested_backend.lower() == "auto" and main_renderer == "mjviewer":
            viewer_backend = "mjviewer"
        else:
            viewer_backend = choose_viewer_backend(
                main_renderer,
                render_camera_names,
                requested_backend=viewer_requested_backend,
            )
        viewer_runtime_cfg = build_flow_runtime_cfg(
            cfg,
            env_metadata=flow_env_metadata,
            camera_names=render_camera_names,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            renderer=main_renderer,
        )
        viewer_runtime_cfg.render_camera = resolve_render_camera(cfg, render_camera_names)
        viewer_env = build_robosuite_env(viewer_runtime_cfg)
        viewer_env = maybe_wrap_visualization(
            viewer_env,
            enabled=visualize_gripper_markers,
            label="viewer env",
        )
        viewer_runtime = RobosuiteViewerRuntime(
            viewer_env,
            render_fps=render_fps,
            async_mode=bool(getattr(cfg.runtime, "viewer_async", False)),
            preview_camera=resolve_render_camera(cfg, render_camera_names),
            backend=viewer_backend,
        )
        viewer_runtime.start()
        viewer_startup_delay = max(0.0, float(getattr(cfg.runtime, "viewer_startup_delay", 0.0)))
        if viewer_startup_delay > 0.0:
            time.sleep(viewer_startup_delay)
    if unthrottled_runtime and bool(cfg.intervention.enabled):
        print("[WARN] runtime.unthrottled=true is incompatible with human intervention. Falling back to throttled mode.")
        unthrottled_runtime = False
    if abs(control_fps - float(cfg.env.control_freq)) > 1e-6:
        print(
            "[WARN] runtime.control_fps does not match env.control_freq. "
            f"Actor loop will target {control_fps:.2f}Hz while the env is configured for {int(cfg.env.control_freq)}Hz."
        )
    if image_obs_fps < control_fps:
        print(
            "[INFO] Image observations are rate-limited below control frequency "
            f"({image_obs_fps:.2f}Hz images vs {control_fps:.2f}Hz control)."
        )
    if not online_updates_enabled:
        print("[INFO] Online learner updates are disabled. Policy parameters will stay frozen during rollout.")
    if on_demand_image_obs:
        print("[INFO] Policy camera observations are rendered on demand to reduce rollout latency.")
    else:
        print("[INFO] Main env camera observations are being used for policy inference.")
    if decoupled_viewer_enabled:
        print(
            "[INFO] Training viewer is decoupled from rollout. "
            f"Window refresh runs at {render_fps:.2f}Hz while policy observations still come from the rollout env."
        )
    if frozen_eval_mode:
        print(
            f"[INFO] Frozen-policy eval mode will reset episodes after {eval_episode_max_steps} steps "
            "if not terminated earlier."
        )
    if unthrottled_runtime:
        print("[INFO] Running without real-time control throttling.")

    loaded_checkpoint = None
    initialized_checkpoint = None
    extra: dict[str, Any] = {}
    for candidate in resume_checkpoint_candidates(cfg, output_root):
        try:
            extra = agent.load_checkpoint(candidate, load_buffers=bool(cfg.runtime.load_buffers))
            loaded_checkpoint = candidate
            print(f"[load] checkpoint={candidate}")
            break
        except Exception as exc:
            print(f"[WARN] Skipping invalid checkpoint {candidate}: {exc}")

    if loaded_checkpoint is None and init_checkpoint is not None:
        init_payload = agent.load_flow_policy_checkpoint(init_checkpoint, task_name=task_name)
        initialized_checkpoint = init_checkpoint
        print(f"[init] flow_policy_checkpoint={init_checkpoint}")

    trainer.load_state_dict(extra.get("trainer_state"))
    if bool(cfg.runtime.load_buffers):
        online_buffer_path, demo_buffer_path, _ = resolve_buffer_snapshot_paths(loaded_checkpoint)
        if online_buffer_path is not None and demo_buffer_path is not None:
            agent.online_buffer.load(online_buffer_path)
            agent.demo_buffer.load(demo_buffer_path)
            print(f"[load] buffers={online_buffer_path.parent}")
        elif loaded_checkpoint is not None:
            print("[load] no legacy buffer snapshot found, checking chunk buffers after demo bootstrap")

    if loaded_checkpoint is not None:
        start_step = int(extra.get("global_step", -1)) + 1
        episode_index = int(extra.get("episode_index", 0))
    else:
        start_step = 0
        episode_index = 0

    write_resolved_config(cfg, checkpoint_dir)
    write_run_info(
        checkpoint_dir,
        {
            "run_name": run_name,
            "run_dir": str(checkpoint_dir),
            "started_at": run_started_at,
            "env_name": str(cfg.env.environment),
            "robots": [str(name) for name in list(cfg.env.robots)],
            "camera_names": policy_camera_names,
            "policy_camera_names": policy_camera_names,
            "render_camera_names": render_camera_names,
            "render_camera": resolve_render_camera(cfg, render_camera_names),
            "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
            "initialized_checkpoint": None if initialized_checkpoint is None else str(initialized_checkpoint),
            "resume_enabled": bool(cfg.runtime.resume),
            "load_buffers": bool(cfg.runtime.load_buffers),
            "seed": serialize_seed(getattr(cfg, "seed", None)),
            "env_reset_seed": env_random_reducer.base_seed,
            "env_reset_seed_rule": "base_seed+episode_index" if env_random_reducer.enabled else None,
            "console_log": str(console_log_path),
            "runtime_log": str(runtime_log_path),
            "buffer_dir": str(checkpoint_dir / "buffers") if save_demo_buffer else None,
            "online_chunk_dir": None,
            "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks") if save_demo_buffer else None,
        },
    )

    print(f"[INFO] Loading offline demos from {demo_source_name}...")
    proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
    # flow_dagger is imitation learning (not RL): no demo cache (re-read each run),
    # randomly sample trajectories per run, and skip the -1/0 reward guard.
    transitions = load_demo_paths(
        demo_paths,
        hdf5_loader=lambda path, demo_names=None: load_hdf5_demos_into_flow_transitions(
            path,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            demo_names=demo_names,
            state_extractor=flow_proprio_extractor,
        ),
        max_num_trajectories=max_num_trajectories,
        check_legacy_rewards=False,
        random_sample=True,
    )
    if len(transitions) == 0:
        raise RuntimeError(
            "Offline demo loading completed but produced zero transitions. "
            "Check that the expert files contain valid trajectories before starting training."
        )

    if not agent.demo_buffer.is_compatible(transitions[0].obs, transitions[0].action):
        print("[WARN] Loaded checkpoint demo_buffer is incompatible with current observation shape. Rebuilding demo buffer.")
        agent.demo_buffer.clear()

    if len(agent.demo_buffer) == 0:
        trainer.bootstrap_demo_buffer(transitions, demo_source="offline_demo")
        if max_num_trajectories is None:
            print(f"[INFO] Loaded offline demo transitions: {len(transitions)}")
        else:
            print(f"[INFO] Loaded offline demo transitions: {len(transitions)} from up to {max_num_trajectories} trajectories")
    else:
        print(f"[INFO] Reusing checkpoint demo buffer with {len(agent.demo_buffer)} transitions")

    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(transitions)
        print("[INFO] Fitted flow normalizers from offline demos.")
    else:
        print("[INFO] Reusing normalizers from checkpoint.")

    requested_pretrain_steps = max(0, int(cfg.algorithm.trainer.pretrain_steps))
    base_policy_checkpoint = resolve_base_policy_checkpoint_path(
        output_root,
        env_name=str(cfg.env.environment),
        demo_source_name=demo_source_name,
        max_num_trajectories=max_num_trajectories,
        pretrain_steps=requested_pretrain_steps,
    )
    base_policy_metadata = {
        "env_name": str(cfg.env.environment),
        "demo_source_name": demo_source_name,
        "num_trajectories": None if max_num_trajectories is None else int(max_num_trajectories),
        "offline_transition_count": int(len(transitions)),
        "pretrain_steps": int(requested_pretrain_steps),
        "checkpoint_path": str(base_policy_checkpoint),
    }
    base_policy_reused = False
    if loaded_checkpoint is None and initialized_checkpoint is None:
        if base_policy_checkpoint.exists():
            try:
                extra = agent.load_checkpoint(base_policy_checkpoint, load_buffers=False)
                trainer.load_state_dict(extra.get("trainer_state"))
                loaded_checkpoint = base_policy_checkpoint
                base_policy_reused = True
                print(
                    "[base_policy] reusing "
                    f"{base_policy_checkpoint.name} "
                    f"(trajectories={format_base_policy_trajectory_tag(max_num_trajectories)}, "
                    f"pretrain_steps={requested_pretrain_steps})"
                )
            except Exception as exc:
                print(f"[WARN] Failed to load reusable base policy {base_policy_checkpoint}: {exc}")
        elif requested_pretrain_steps <= 0:
            raise ValueError(
                "Base policy checkpoint is missing and algorithm.trainer.pretrain_steps <= 0. "
                "Set a positive pretrain_steps value to build the base policy from offline demos, "
                "or provide an existing checkpoint."
            )
    elif initialized_checkpoint is not None:
        base_policy_reused = True
        print(f"[base_policy] initialized from multitask flow checkpoint={initialized_checkpoint}")

    if bool(cfg.runtime.load_buffers):
        _, demo_chunk_dir = resolve_buffer_chunk_dirs(loaded_checkpoint)
        loaded_online_transitions = 0
        loaded_demo_transitions = 0
        if demo_chunk_dir is not None:
            loaded_demo_transitions = load_transition_chunks(agent.demo_buffer, demo_chunk_dir)
        if loaded_online_transitions > 0 or loaded_demo_transitions > 0:
            print(
                f"[load] chunk_buffers online={loaded_online_transitions} demo={loaded_demo_transitions} "
                f"from {resolve_checkpoint_run_dir(loaded_checkpoint) / 'buffers'}"
            )

    if not store_non_intervention_images and len(agent.online_buffer) > 0:
        agent.online_buffer.clear()
        print("[INFO] Cleared online buffer for state-only Flow-DAgger rollout storage.")

    wandb_run = maybe_build_wandb(cfg, run_name=run_name, run_dir=checkpoint_dir)
    device = None
    intervention_runtime = None

    episode_return = 0.0
    episode_length = 0
    episode_step_index = 0
    success_count = 0
    last_step = start_step - 1
    control_limiter = None if unthrottled_runtime else FixedRateLimiter(control_fps)
    policy_gate = IntervalGate(policy_fps)
    spacemouse_gate = IntervalGate(spacemouse_fps)
    overall_fps_tracker = EMAFpsTracker()
    last_fps_log_time = time.monotonic()
    training_started_monotonic = last_fps_log_time
    cached_policy_action = np.zeros_like(action_low, dtype=np.float32)
    cached_override_action: np.ndarray | None = None
    cached_is_intervention = False
    last_reported_publish_count = int(trainer.progress_snapshot()["publish_count"])
    episode_pause_sec = max(0.0, float(getattr(cfg.runtime, "episode_pause_sec", 1.0)))
    total_transition_count = max(0, int(start_step))
    total_intervention_transitions = 0
    episode_transition_count = 0
    episode_intervention_transitions = 0

    def event_time_fields() -> dict[str, Any]:
        now = datetime.datetime.now()
        return {
            "wall_time": now.isoformat(timespec="milliseconds"),
            "run_elapsed_sec": round(time.monotonic() - run_started_monotonic, 6),
            "training_elapsed_sec": round(time.monotonic() - training_started_monotonic, 6),
        }

    def refresh_main_viewer() -> None:
        if viewer_runtime is not None:
            viewer_runtime.publish_from_env(env)
            viewer_runtime.render_if_due()
            return
        if not rollout_has_renderer:
            return
        if getattr(env, "viewer", None) is None and hasattr(env, "initialize_renderer"):
            env.initialize_renderer()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
            env.viewer.update()

    def reset_viewer_preview() -> None:
        if viewer_runtime is None:
            refresh_main_viewer()
            return
        viewer_runtime.reset_preview(
            snapshot_env_state(env),
            warmup_frames=int(getattr(cfg.runtime, "viewer_reset_warmup_frames", 2)),
        )

    def maybe_print_publish_events(metrics_list: list[dict[str, float]]) -> None:
        nonlocal last_reported_publish_count
        for metrics in metrics_list:
            publish_count = int(metrics.get("learner_publish_count", 0.0))
            if publish_count <= last_reported_publish_count:
                continue
            runtime_logger.log(
                {
                    "event": "publish",
                    "step": int(last_step),
                    "episode_index": int(episode_index),
                    "publish_count": publish_count,
                    "learner_total_updates": int(metrics.get("learner_total_updates", 0.0)),
                    "learner_actor_updates": int(metrics.get("learner_actor_updates", 0.0)),
                    **event_time_fields(),
                }
            )
            print(format_publish_line(metrics))
            last_reported_publish_count = publish_count

    def maybe_report_runtime(step: int) -> None:
        nonlocal last_fps_log_time
        now = time.monotonic()
        elapsed = now - last_fps_log_time
        if elapsed < fps_log_interval:
            return
        learner_progress = trainer.progress_snapshot()
        overall_fps = overall_fps_tracker.snapshot(elapsed)
        pending_updates = trainer.pending_async_updates() if async_updates else 0
        runtime_payload = {
            "overall_fps": overall_fps,
            "learner_actor_updates": float(learner_progress["actor_updates"]),
            "learner_updates_until_publish": float(learner_progress["updates_until_publish"]),
            "learner_publish_count": float(learner_progress["publish_count"]),
        }
        if async_updates:
            runtime_payload["pending_updates"] = float(pending_updates)
        maybe_log(wandb_run, runtime_payload, step=step)
        runtime_logger.log(
            {
                "event": "runtime",
                "step": int(step),
                "episode_index": int(episode_index),
                "overall_fps": float(overall_fps),
                "pending_updates": int(pending_updates),
                "learner_total_updates": int(learner_progress["total_updates"]),
                "learner_actor_updates": int(learner_progress["actor_updates"]),
                "learner_publish_count": int(learner_progress["publish_count"]),
                "learner_updates_until_publish": int(learner_progress["updates_until_publish"]),
                "total_transition_count": int(total_transition_count),
                "total_intervention_transitions": int(total_intervention_transitions),
                "global_intervention_ratio": (
                    float(total_intervention_transitions) / float(total_transition_count)
                    if total_transition_count > 0
                    else 0.0
                ),
                "episode_transition_count": int(episode_transition_count),
                "episode_intervention_transitions": int(episode_intervention_transitions),
                "episode_intervention_ratio": (
                    float(episode_intervention_transitions) / float(episode_transition_count)
                    if episode_transition_count > 0
                    else 0.0
                ),
                **event_time_fields(),
            }
        )
        print(
            format_runtime_line(
                step=step,
                episode_index=episode_index,
                overall_fps=overall_fps,
                learner_progress=learner_progress,
                pending_updates=pending_updates,
            )
        )
        last_fps_log_time = now

    def request_checkpoint_save(step: int, tag: str | None = None) -> None:
        step_checkpoint = (
            checkpoint_path(checkpoint_dir, tag)
            if tag is not None
            else checkpoint_step_path(
                checkpoint_dir,
                step=step,
                learner_updates=trainer.total_updates,
                episode_index=episode_index,
            )
        )
        checkpoint_extra = {
            "global_step": int(step),
            "episode_index": int(episode_index),
            "success_count": int(success_count),
            "trainer_state": trainer.state_dict(),
        }
        checkpoint_writer.request_save(
            paths=[step_checkpoint, checkpoint_path(checkpoint_dir, "latest")],
            include_buffers=False,
            extra=checkpoint_extra,
            metadata={"step": int(step)},
        )
        pending_updates = trainer.pending_async_updates() if async_updates else 0
        runtime_logger.log(
            {
                "event": "checkpoint_queued",
                "step": int(step),
                "episode_index": int(episode_index),
                "pending_updates": int(pending_updates),
                "learner_total_updates": int(trainer.total_updates),
                "checkpoint_name": step_checkpoint.name,
                **event_time_fields(),
            }
        )
        print(f"[ckpt] step={step} pending={pending_updates} queued={step_checkpoint.name}")

    cached_policy_images: dict[str, np.ndarray] | None = None

    def make_live_obs(*, include_images: bool, refresh_images: bool = True) -> dict[str, Any]:
        nonlocal cached_policy_images
        obs = {"state": extract_flow_state(env, flow_proprio_extractor)}
        if not include_images:
            return obs
        if refresh_images or cached_policy_images is None:
            cached_policy_images = render_policy_camera_images(
                env,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
            )
        obs.update(cached_policy_images)
        return obs

    def reset_rollout_observation() -> dict[str, Any]:
        episode_seed = env_random_reducer.prepare_episode(env, episode_index, seed_global=False)
        runtime_logger.log(
            {
                "event": "episode_reset",
                "step": int(last_step),
                "episode_index": int(episode_index),
                "episode_seed": None if episode_seed is None else int(episode_seed),
                **event_time_fields(),
            }
        )
        if on_demand_image_obs:
            reset_robosuite_env(env, preserve_mjviewer=rollout_has_renderer)
            return make_live_obs(include_images=True, refresh_images=True)
        reset_obs, _ = reset_flow_policy_observation(
            env,
            preserve_mjviewer=rollout_has_renderer,
            extractor=flow_proprio_extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
        )
        return reset_obs

    def convert_step_observation(raw_obs: dict[str, Any], *, include_images: bool) -> dict[str, Any]:
        if on_demand_image_obs:
            return make_live_obs(include_images=include_images, refresh_images=True)
        if include_images:
            return convert_env_camera_observation(
                raw_obs,
                env=env,
                extractor=flow_proprio_extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
            )
        return make_live_obs(include_images=False)

    demo_action_horizon = max(1, int(agent.flow_config.action_horizon))
    pending_demo_chunk: list[Transition] = []
    completed_intervention_demo_chunks = 0

    def replace_demo_transition(
        transition: Transition,
        *,
        action=None,
        reward: float | None = None,
        next_obs=None,
        done: bool | None = None,
        is_intervention: bool | None = None,
        info: dict[str, Any] | None = None,
        reward_source: str | None = None,
        demo_source: str | None = None,
    ) -> Transition:
        return Transition(
            obs=transition.obs,
            action=transition.action if action is None else action,
            reward=transition.reward if reward is None else float(reward),
            next_obs=transition.next_obs if next_obs is None else next_obs,
            done=bool(transition.done if done is None else done),
            grasp_penalty=transition.grasp_penalty,
            is_intervention=bool(transition.is_intervention if is_intervention is None else is_intervention),
            info=dict(transition.info) if info is None and transition.info is not None else info,
            reward_source=transition.reward_source if reward_source is None else reward_source,
            demo_source=transition.demo_source if demo_source is None else demo_source,
        )

    def emit_completed_demo_chunks(*, step: int, reason: str) -> None:
        nonlocal completed_intervention_demo_chunks
        while len(pending_demo_chunk) >= demo_action_horizon:
            chunk = pending_demo_chunk[:demo_action_horizon]
            del pending_demo_chunk[:demo_action_horizon]
            for demo_transition in chunk:
                agent.store_demo_transition(demo_transition)
                if demo_buffer_writer is not None:
                    demo_buffer_writer.request_transition(demo_transition)
            completed_intervention_demo_chunks += 1
            runtime_logger.log(
                {
                    "event": "intervention_demo_chunk_committed",
                    "step": int(step),
                    "reason": str(reason),
                    "chunk_size": int(demo_action_horizon),
                    "completed_intervention_demo_chunks": int(completed_intervention_demo_chunks),
                    "demo_buffer_size": len(agent.demo_buffer),
                    **event_time_fields(),
                }
            )

    def queue_demo_chunk_transition(transition: Transition, *, step: int, reason: str) -> None:
        pending_demo_chunk.append(agent.demo_buffer.snapshot_transition(transition))
        emit_completed_demo_chunks(step=step, reason=reason)

    def pad_pending_demo_chunk(*, step: int, reason: str, episode_index: int, episode_step: int) -> None:
        if len(pending_demo_chunk) == 0:
            return
        pad_count = (-len(pending_demo_chunk)) % demo_action_horizon
        if pad_count == 0:
            emit_completed_demo_chunks(step=step, reason=reason)
            return

        for index, transition in enumerate(pending_demo_chunk):
            if bool(transition.done):
                pending_demo_chunk[index] = replace_demo_transition(transition, done=False)

        last_transition = pending_demo_chunk[-1]
        last_info = dict(last_transition.info) if last_transition.info is not None else {}
        pad_episode_index = int(last_info.get("episode_index", episode_index))
        last_episode_step = int(last_info.get("episode_step", episode_step - 1))
        zero_action = np.zeros_like(np.asarray(last_transition.action, dtype=np.float32))
        pad_obs = last_transition.obs
        for pad_offset in range(pad_count):
            pad_info = dict(last_info)
            pad_info["episode_index"] = int(pad_episode_index)
            pad_info["episode_step"] = int(last_episode_step + pad_offset + 1)
            pad_info["flow_dagger_padding"] = True
            pad_info["flow_dagger_padding_reason"] = str(reason)
            pending_demo_chunk.append(
                agent.demo_buffer.snapshot_transition(
                    Transition(
                        obs=pad_obs,
                        action=zero_action,
                        reward=0.0,
                        next_obs=None,
                        done=pad_offset == pad_count - 1,
                        grasp_penalty=None,
                        is_intervention=False,
                        info=pad_info,
                        reward_source="padding",
                        demo_source="intervention_zero_pad",
                    )
                )
            )

        runtime_logger.log(
            {
                "event": "intervention_demo_chunk_padded",
                "step": int(step),
                "reason": str(reason),
                "pad_count": int(pad_count),
                "chunk_size": int(demo_action_horizon),
                **event_time_fields(),
            }
        )
        emit_completed_demo_chunks(step=step, reason=reason)

    if loaded_checkpoint is None and initialized_checkpoint is None and start_step == 0:
        completed_pretrain_steps = int(trainer.total_pretrain_updates)
        remaining_pretrain_steps = requested_pretrain_steps - completed_pretrain_steps
        print(
            f"[base_policy] training remaining_pretrain_steps={remaining_pretrain_steps} "
            f"(completed={completed_pretrain_steps}, target={requested_pretrain_steps})"
        )
        for local_step in range(remaining_pretrain_steps):
            metrics = trainer.pretrain(1)[-1]
            absolute_pretrain_step = completed_pretrain_steps + local_step + 1
            if absolute_pretrain_step % int(cfg.logging.log_interval) == 0:
                maybe_log(
                    wandb_run,
                    {f"pretrain/{key}": value for key, value in metrics.items()},
                    step=absolute_pretrain_step,
                )
                print(
                    "[base_policy] "
                    f"step={absolute_pretrain_step} loss={metrics.get('actor_loss', float('nan')):.4f} "
                    f"flow={metrics.get('flow_loss', float('nan')):.4f} "
                    f"endpoint={metrics.get('endpoint_loss', float('nan')):.4f}"
                )
            maybe_print_publish_events([metrics])
        base_policy_extra = {
            "global_step": -1,
            "episode_index": 0,
            "success_count": 0,
            "trainer_state": trainer.state_dict(),
            "base_policy_metadata": dict(base_policy_metadata),
        }
        agent.save_checkpoint(base_policy_checkpoint, include_buffers=False, extra=base_policy_extra)
        print(f"[base_policy] saved {base_policy_checkpoint}")
    elif base_policy_reused:
        print(f"[base_policy] ready from checkpoint={base_policy_checkpoint}")

    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention_runtime = RobosuiteInterventionRuntime(
            env=env,
            device=device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )

    # Refresh the rollout episode after model / demo bootstrap so the first episode starts from the
    # same phase as the original flow_multi eval path, which resets immediately before inference.
    obs = reset_rollout_observation()
    reset_viewer_preview()
    agent.reset_policy_state()
    policy_gate.force_ready()
    spacemouse_gate.force_ready()
    if intervention_runtime is not None:
        intervention_runtime.start_episode()

    if async_updates and online_updates_enabled:
        trainer.start_async_worker()

    interrupted = False
    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = time.monotonic() if control_limiter is None else control_limiter.wait()
            overall_fps_tracker.mark()

            if (not cached_is_intervention) and (unthrottled_runtime or policy_gate.ready(loop_start)):
                if step < int(cfg.algorithm.trainer.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    if on_demand_image_obs and agent.needs_action_chunk():
                        obs = make_live_obs(include_images=True, refresh_images=True)
                    cached_policy_action = agent.select_action(obs, deterministic=bool(cfg.runtime.eval_deterministic))
                    if on_demand_image_obs:
                        obs = make_live_obs(include_images=False)

            env_action = np.asarray(cached_policy_action, dtype=np.float32)
            is_intervention = False
            reset_requested = False
            if intervention_runtime is not None and (unthrottled_runtime or spacemouse_gate.ready(loop_start)):
                was_intervening = cached_is_intervention
                override_action, sampled_is_intervention, reset_requested = intervention_runtime.maybe_override_action(
                    cached_policy_action
                )
                if reset_requested:
                    cached_override_action = None
                    cached_is_intervention = False
                    agent.reset_policy_state()
                elif sampled_is_intervention:
                    cached_override_action = np.asarray(override_action, dtype=np.float32)
                    cached_is_intervention = True
                    if not was_intervening:
                        agent.notify_intervention()
                else:
                    cached_override_action = None
                    cached_is_intervention = False
                    if was_intervening:
                        agent.reset_policy_state()
                        policy_gate.force_ready()

            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[INFO] Device reset requested. Exiting training loop.")
                    break
                pad_pending_demo_chunk(
                    step=step,
                    reason="device_reset",
                    episode_index=episode_index,
                    episode_step=episode_step_index,
                )
                reset_episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": 0,
                    "is_success": False,
                    "episode_seed": env_random_reducer.seed_for_episode(episode_index),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                    "end_reason": "device_reset",
                }
                maybe_log(wandb_run, reset_episode_payload, step=step)
                runtime_logger.log(
                    {
                        "event": "episode_end",
                        "end_reason": "device_reset",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        "episode_success": False,
                        "is_success": False,
                        "episode_seed": env_random_reducer.seed_for_episode(episode_index),
                        "episode_transition_count": int(episode_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        "episode_intervention_ratio": (
                            float(episode_intervention_transitions) / float(episode_transition_count)
                            if episode_transition_count > 0
                            else 0.0
                        ),
                        "total_transition_count": int(total_transition_count),
                        "total_intervention_transitions": int(total_intervention_transitions),
                        "global_intervention_ratio": (
                            float(total_intervention_transitions) / float(total_transition_count)
                            if total_transition_count > 0
                            else 0.0
                        ),
                        **event_time_fields(),
                    }
                )
                print(
                    format_episode_line(
                        step=step,
                        episode_index=episode_index,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        success=False,
                        online_buffer_size=len(agent.online_buffer),
                        demo_buffer_size=len(agent.demo_buffer),
                    )
                )
                obs = reset_rollout_observation()
                reset_viewer_preview()
                agent.reset_policy_state()
                episode_return = 0.0
                episode_length = 0
                episode_step_index = 0
                episode_index += 1
                episode_transition_count = 0
                episode_intervention_transitions = 0
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
                maybe_report_runtime(step)
                continue

            if cached_is_intervention and cached_override_action is not None:
                env_action = np.asarray(cached_override_action, dtype=np.float32)
                is_intervention = True

            needs_demo_chunk_transition = is_intervention or len(pending_demo_chunk) > 0
            online_obs = make_live_obs(include_images=store_non_intervention_images)
            demo_obs = make_live_obs(include_images=True, refresh_images=True) if needs_demo_chunk_transition else None
            step_output = env.step(env_action)
            if len(step_output) == 5:
                raw_next_obs, _, done, truncated, info = step_output
                done = bool(done or truncated)
            else:
                raw_next_obs, _, done, info = step_output
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            next_obs = convert_step_observation(
                raw_next_obs,
                include_images=store_next_obs_images,
            )
            demo_next_obs = (
                convert_step_observation(raw_next_obs, include_images=True)
                if needs_demo_chunk_transition and store_next_obs_images
                else None
            )
            refresh_main_viewer()
            if frozen_eval_mode and (episode_length + 1) >= eval_episode_max_steps:
                done = True
            done = bool(done or success)
            info_payload = dict(info) if isinstance(info, dict) else {"raw_info": info}
            info_payload.setdefault("episode_index", int(episode_index))
            info_payload.setdefault("episode_step", int(episode_step_index))
            info_payload.setdefault("episode_seed", env_random_reducer.seed_for_episode(episode_index))
            trainer.record_transition(
                obs=online_obs,
                action=env_action,
                next_obs=next_obs,
                done=done,
                reward=reward,
                grasp_penalty=None,
                is_intervention=is_intervention,
                info=info_payload,
                reward_source="env_success",
                demo_source="intervention" if is_intervention else None,
                demo_obs=demo_obs,
                demo_next_obs=demo_next_obs,
                store_demo_transition=False,
                episode_index=episode_index,
                episode_step=episode_step_index,
            )
            if needs_demo_chunk_transition and demo_obs is not None:
                demo_transition = Transition(
                    obs=demo_obs,
                    action=env_action,
                    reward=reward,
                    next_obs=demo_next_obs,
                    done=done,
                    grasp_penalty=None,
                    is_intervention=is_intervention,
                    info=info_payload,
                    reward_source="env_success",
                    demo_source="intervention" if is_intervention else "intervention_policy_tail",
                )
                queue_demo_chunk_transition(
                    demo_transition,
                    step=step,
                    reason="intervention" if is_intervention else "policy_tail",
                )
            total_transition_count += 1
            episode_transition_count += 1
            episode_step_index += 1
            if is_intervention:
                total_intervention_transitions += 1
                episode_intervention_transitions += 1
                runtime_logger.log(
                    {
                        "event": "intervention_transition",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_transition_index": int(episode_transition_count),
                        "total_transition_count": int(total_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        "total_intervention_transitions": int(total_intervention_transitions),
                        "episode_intervention_ratio": float(episode_intervention_transitions)
                        / float(episode_transition_count),
                        "global_intervention_ratio": float(total_intervention_transitions)
                        / float(total_transition_count),
                        **event_time_fields(),
                    }
                )
            episode_return += reward
            episode_length += 1
            success_count += int(success)

            if online_updates_enabled:
                if async_updates:
                    update_metrics_list = trainer.maybe_update_async()
                else:
                    update_metrics_list = trainer.maybe_update()
            else:
                update_metrics_list = []
            if update_metrics_list:
                maybe_print_publish_events(update_metrics_list)
                update_metrics = update_metrics_list[-1]
                if step % int(cfg.logging.log_interval) == 0:
                    maybe_log(wandb_run, update_metrics, step=step)
                    print(
                        format_train_line(
                            step=step,
                            metrics=update_metrics,
                            pending_updates=trainer.pending_async_updates() if async_updates else 0,
                        )
                    )

            if done:
                pad_pending_demo_chunk(
                    step=step,
                    reason="episode_end",
                    episode_index=episode_index,
                    episode_step=episode_step_index,
                )
                episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": int(success),
                    "is_success": bool(success),
                    "episode_seed": env_random_reducer.seed_for_episode(episode_index),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                }
                maybe_log(wandb_run, episode_payload, step=step)
                runtime_logger.log(
                    {
                        "event": "episode_end",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        "episode_success": bool(success),
                        "is_success": bool(success),
                        "episode_seed": env_random_reducer.seed_for_episode(episode_index),
                        "episode_transition_count": int(episode_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        "episode_intervention_ratio": (
                            float(episode_intervention_transitions) / float(episode_transition_count)
                            if episode_transition_count > 0
                            else 0.0
                        ),
                        "total_transition_count": int(total_transition_count),
                        "total_intervention_transitions": int(total_intervention_transitions),
                        "global_intervention_ratio": (
                            float(total_intervention_transitions) / float(total_transition_count)
                            if total_transition_count > 0
                            else 0.0
                        ),
                        **event_time_fields(),
                    }
                )
                print(
                    format_episode_line(
                        step=step,
                        episode_index=episode_index,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        success=bool(success),
                        online_buffer_size=len(agent.online_buffer),
                        demo_buffer_size=len(agent.demo_buffer),
                    )
                )
                if episode_pause_sec > 0.0:
                    time.sleep(episode_pause_sec)
                obs = reset_rollout_observation()
                reset_viewer_preview()
                agent.reset_policy_state()
                episode_return = 0.0
                episode_length = 0
                episode_step_index = 0
                episode_index += 1
                episode_transition_count = 0
                episode_intervention_transitions = 0
                cached_override_action = None
                cached_is_intervention = False
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
            else:
                obs = next_obs

            if step > 0 and step % int(cfg.logging.checkpoint_interval) == 0:
                request_checkpoint_save(step)
            maybe_report_runtime(step)
    except KeyboardInterrupt:
        interrupted = True
        print("[INFO] Ctrl+C received. Flushing intervention demo data and latest checkpoint before exit.")
    finally:
        pad_pending_demo_chunk(
            step=last_step if last_step >= 0 else 0,
            reason="shutdown",
            episode_index=episode_index,
            episode_step=episode_step_index,
        )
        if demo_buffer_writer is not None:
            demo_buffer_writer.flush(timeout=120.0)
        if async_updates:
            trainer.flush_async_updates()
            flushed_metrics = trainer.drain_async_metrics()
            if flushed_metrics:
                maybe_print_publish_events(flushed_metrics)
        maybe_report_runtime(last_step if last_step >= 0 else 0)
        checkpoint_writer.request_save(
            paths=[checkpoint_path(checkpoint_dir, "latest")],
            include_buffers=False,
            extra={
                "global_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "trainer_state": trainer.state_dict(),
            },
            metadata={"step": int(last_step)},
        )
        checkpoint_writer.flush(timeout=120.0)
        runtime_logger.log(
            {
                "event": "run_end",
                "step": int(last_step),
                "episode_index": int(episode_index),
                "interrupted": bool(interrupted),
                "success_count": int(success_count),
                "total_transition_count": int(total_transition_count),
                "total_intervention_transitions": int(total_intervention_transitions),
                "global_intervention_ratio": (
                    float(total_intervention_transitions) / float(total_transition_count)
                    if total_transition_count > 0
                    else 0.0
                ),
                **event_time_fields(),
            }
        )
        write_run_info(
            checkpoint_dir,
            {
                "run_name": run_name,
                "run_dir": str(checkpoint_dir),
                "started_at": run_started_at,
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "env_name": str(cfg.env.environment),
                "robots": [str(name) for name in list(cfg.env.robots)],
                "policy_camera_names": policy_camera_names,
                "render_camera_names": render_camera_names,
                "render_camera": resolve_render_camera(cfg, render_camera_names),
                "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
                "initialized_checkpoint": None if initialized_checkpoint is None else str(initialized_checkpoint),
                "resume_enabled": bool(cfg.runtime.resume),
                "load_buffers": bool(cfg.runtime.load_buffers),
                "seed": serialize_seed(getattr(cfg, "seed", None)),
                "env_reset_seed": env_random_reducer.base_seed,
                "env_reset_seed_rule": "base_seed+episode_index" if env_random_reducer.enabled else None,
                "console_log": str(console_log_path),
                "runtime_log": str(runtime_log_path),
                "buffer_dir": str(checkpoint_dir / "buffers") if save_demo_buffer else None,
                "online_chunk_dir": None,
                "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks") if save_demo_buffer else None,
                "last_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "latest_checkpoint": str(checkpoint_path(checkpoint_dir, "latest")),
                "trainer_state": trainer.state_dict(),
                "base_policy_checkpoint": str(base_policy_checkpoint),
                "base_policy_reused": bool(base_policy_reused),
                "base_policy_metadata": dict(base_policy_metadata),
            },
        )
        if async_updates:
            trainer.close_async_worker(wait=False)
        if intervention_runtime is not None:
            intervention_runtime.close()
        if wandb_run is not None:
            wandb_run.finish()
        try:
            env.close()
        finally:
            try:
                if viewer_runtime is not None:
                    viewer_runtime.close()
            finally:
                try:
                    if flow_proprio_extractor is not None:
                        flow_proprio_extractor.close()
                finally:
                    try:
                        checkpoint_writer.close()
                    finally:
                        try:
                            runtime_logger.close()
                        finally:
                            try:
                                if demo_buffer_writer is not None:
                                    demo_buffer_writer.close()
                            finally:
                                console_capture.stop()


if __name__ == "__main__":
    main()
