from __future__ import annotations

import datetime
import json
import os
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import VisualizationWrapper

from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.algorithms.hil_serl import HILSERLTrainer
from robosuite.pipeline.algorithms.hil_serl.utils import load_demo_paths, resolve_task_demo_paths
from robosuite.pipeline.algorithms.hil_serl.envs import (
    RobosuiteInterventionRuntime,
    RobosuiteObservationAdapter,
    RobosuiteRuntimeConfig,
    build_device,
    build_robosuite_env,
    load_hdf5_demos_into_transitions,
    make_checkpoint_directory,
    sparse_success_reward,
)


def now_readable() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_camera_names(cfg: DictConfig) -> list[str]:
    camera_names = [str(name) for name in list(cfg.env.camera_names or [])]
    if len(camera_names) == 0:
        camera_names = ["agentview"]
    return camera_names


def resolve_render_camera(cfg: DictConfig, camera_names: list[str]) -> str | None:
    render_camera = getattr(cfg.env, "render_camera", None)
    if render_camera is not None:
        return str(render_camera)
    if len(camera_names) > 0:
        return str(camera_names[0])
    return None


def resolve_demo_task_name(cfg: DictConfig) -> str:
    explicit_task_name = cfg.data.task_name
    if explicit_task_name is not None:
        return str(explicit_task_name)

    robots = [str(name) for name in list(cfg.env.robots or [])]
    if len(robots) == 1:
        return f"{robots[0]}{cfg.env.environment}"
    return str(cfg.env.environment)


def resolve_demo_inputs(cfg: DictConfig) -> tuple[str, list[Path], int | None]:
    max_num_trajectories = cfg.data.num_trajectories
    if max_num_trajectories is not None:
        max_num_trajectories = int(max_num_trajectories)
        if max_num_trajectories <= 0:
            raise ValueError("data.num_trajectories must be positive when provided.")

    explicit_demo_paths = [Path(to_absolute_path(str(path))) for path in list(cfg.data.demo_paths or [])]
    if explicit_demo_paths:
        return "explicit_paths", explicit_demo_paths, max_num_trajectories

    task_name = resolve_demo_task_name(cfg)
    demo_paths = resolve_task_demo_paths(
        task_name=task_name,
        data_root=to_absolute_path(str(cfg.data.demo_root)),
        split=str(cfg.data.demo_split),
    )
    return task_name, demo_paths, max_num_trajectories


def maybe_build_wandb(cfg: DictConfig, run_name: str | None = None, run_dir: Path | None = None):
    if not bool(cfg.logging.use_wandb):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Falling back to stdout logging.")
        return None

    os.environ.setdefault("WANDB_MODE", str(cfg.logging.wandb_mode))
    run = wandb.init(
        project=str(cfg.logging.project),
        entity=str(cfg.logging.entity),
        mode=str(cfg.logging.wandb_mode),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=str(cfg.logging.run_name or run_name) if (cfg.logging.run_name is not None or run_name is not None) else None,
        dir=str(run_dir if run_dir is not None else Path(to_absolute_path(str(cfg.logging.output_root)))),
    )
    return run


def resolve_checkpoint_reference(reference: Any) -> Path | None:
    if reference is None:
        return None
    normalized = str(reference).strip()
    if normalized == "" or normalized.lower() == "null":
        return None

    candidate = Path(to_absolute_path(normalized))
    if candidate.is_dir():
        latest_inside_checkpoints = candidate / "latest.pt" if candidate.name == "checkpoints" else candidate / "checkpoints" / "latest.pt"
        if latest_inside_checkpoints.exists():
            return latest_inside_checkpoints
        raise FileNotFoundError(f"Could not find latest checkpoint under directory: {candidate}")
    return candidate


def resolve_checkpoint_run_dir(checkpoint: Path | None) -> Path | None:
    if checkpoint is None:
        return None
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return None


def resolve_buffer_snapshot_paths(checkpoint: Path | None) -> tuple[Path | None, Path | None, Path | None]:
    run_dir = resolve_checkpoint_run_dir(checkpoint)
    if run_dir is None:
        return None, None, None
    latest_dir = run_dir / "buffers" / "latest"
    online_path = latest_dir / "online_buffer.pt"
    demo_path = latest_dir / "demo_buffer.pt"
    metadata_path = latest_dir / "metadata.json"
    if not online_path.exists() or not demo_path.exists():
        return None, None, metadata_path if metadata_path.exists() else None
    return online_path, demo_path, metadata_path if metadata_path.exists() else None


def resolve_buffer_chunk_dirs(checkpoint: Path | None) -> tuple[Path | None, Path | None]:
    run_dir = resolve_checkpoint_run_dir(checkpoint)
    if run_dir is None:
        return None, None
    online_dir = run_dir / "buffers" / "online_chunks"
    demo_dir = run_dir / "buffers" / "demo_chunks"
    return online_dir if online_dir.exists() else None, demo_dir if demo_dir.exists() else None


def load_transition_chunks(buffer, chunk_dir: Path) -> int:
    total_loaded = 0
    for chunk_path in sorted(chunk_dir.glob("chunk_*.pt")):
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        transitions = payload.get("transitions", [])
        if len(transitions) == 0:
            continue
        buffer.extend(list(transitions))
        total_loaded += int(len(transitions))
    return total_loaded


def build_runtime_cfg(
    cfg: DictConfig,
    camera_names: list[str],
    *,
    has_renderer: bool,
    has_offscreen_renderer: bool,
    renderer: str | None = None,
) -> RobosuiteRuntimeConfig:
    render_camera = resolve_render_camera(cfg, camera_names)
    return RobosuiteRuntimeConfig(
        env_name=str(cfg.env.environment),
        robots=[str(name) for name in list(cfg.env.robots)],
        env_configuration=str(cfg.env.config) if cfg.env.config is not None else None,
        controller=str(cfg.env.controller) if cfg.env.controller is not None else None,
        controller_configs=None,
        renderer=str(cfg.env.renderer) if renderer is None else str(renderer),
        render_camera=render_camera,
        camera_names=tuple(camera_names),
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        reward_shaping=False,
        control_freq=int(cfg.env.control_freq),
        has_renderer=bool(has_renderer),
        has_offscreen_renderer=bool(has_offscreen_renderer),
        ignore_done=False,
        use_camera_obs=False,
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        horizon=int(cfg.env.horizon) if cfg.env.horizon is not None else None,
    )


def maybe_log(run, payload: dict[str, Any], step: int) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def checkpoint_path(checkpoint_dir: Path, tag: str) -> Path:
    return checkpoint_dir / "checkpoints" / f"{tag}.pt"


def checkpoint_step_path(checkpoint_dir: Path, *, step: int, learner_updates: int, episode_index: int) -> Path:
    return checkpoint_path(
        checkpoint_dir,
        f"step_{int(step):08d}_updates_{int(learner_updates):08d}_ep_{int(episode_index):05d}",
    )


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

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def find_latest_resumable_run(output_root: Path, env_name: str) -> Path | None:
    env_prefix = f"hil_serl_{env_name}_"
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


def resolve_run_directory(cfg: DictConfig) -> tuple[str, Path]:
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    explicit_run_name = cfg.logging.run_name

    if explicit_run_name is not None:
        run_name = str(explicit_run_name)
        return run_name, make_checkpoint_directory(output_root, run_name)

    run_name = f"hil_serl_{cfg.env.environment}_{now_readable()}"
    return run_name, make_checkpoint_directory(output_root, run_name)


class FixedRateLimiter:
    def __init__(self, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError("FixedRateLimiter requires a positive fps.")
        self.period = 1.0 / float(fps)
        self._next_tick_time: float | None = None

    def wait(self) -> float:
        now = time.monotonic()
        if self._next_tick_time is None:
            self._next_tick_time = now + self.period
            return now
        if now < self._next_tick_time:
            time.sleep(self._next_tick_time - now)
            now = time.monotonic()
        while self._next_tick_time <= now:
            self._next_tick_time += self.period
        return now


class IntervalGate:
    def __init__(self, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError("IntervalGate requires a positive fps.")
        self.period = 1.0 / float(fps)
        self._next_fire_time: float | None = None

    def ready(self, now: float) -> bool:
        if self._next_fire_time is None:
            self._next_fire_time = now + self.period
            return True
        if now < self._next_fire_time:
            return False
        while self._next_fire_time <= now:
            self._next_fire_time += self.period
        return True

    def force_ready(self) -> None:
        self._next_fire_time = None


class EMAFpsTracker:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = float(alpha)
        self._pending_count = 0
        self._ema = 0.0
        self._initialized = False

    def mark(self, count: int = 1) -> None:
        self._pending_count += int(count)

    def snapshot(self, elapsed: float) -> float:
        instantaneous_fps = float(self._pending_count) / max(float(elapsed), 1e-6)
        self._pending_count = 0
        if not self._initialized:
            self._ema = instantaneous_fps
            self._initialized = True
        else:
            self._ema = self.alpha * instantaneous_fps + (1.0 - self.alpha) * self._ema
        return self._ema


def resolve_runtime_fps(cfg: DictConfig, key: str, default_value: float) -> float:
    runtime_value = getattr(cfg.runtime, key, None)
    if runtime_value is None:
        return float(default_value)
    return float(runtime_value)


def resolve_requested_device(requested: Any, *, fallback: str) -> str:
    if requested is None:
        return fallback
    normalized = str(requested).strip()
    if normalized == "":
        return fallback
    if not normalized.startswith("cuda"):
        return normalized
    if not torch.cuda.is_available():
        print(f"[WARN] Requested CUDA device '{normalized}' but CUDA is unavailable. Falling back to cpu.")
        return "cpu"
    if normalized == "cuda":
        return "cuda:0"
    try:
        device_index = int(normalized.split(":", 1)[1])
    except (IndexError, ValueError):
        print(f"[WARN] Invalid CUDA device '{normalized}'. Falling back to {fallback}.")
        return fallback
    if device_index >= torch.cuda.device_count():
        print(
            f"[WARN] Requested CUDA device '{normalized}' but only {torch.cuda.device_count()} visible GPU(s) exist. "
            f"Falling back to {fallback}."
        )
        return fallback
    return normalized


def resolve_algorithm_devices(algorithm_cfg: dict[str, Any]) -> tuple[str, str]:
    sac_cfg = algorithm_cfg.setdefault("sac", {})
    learner_requested = sac_cfg.get("device", "cpu")
    inference_requested = sac_cfg.get("inference_device", None)
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

    sac_cfg["device"] = learner_device
    sac_cfg["inference_device"] = inference_device
    return learner_device, inference_device


def maybe_wrap_visualization(env, *, enabled: bool, label: str):
    if not enabled:
        return env
    wrapped_env = VisualizationWrapper(env)
    print(f"[INFO] Enabled robosuite gripper visualization markers on {label}.")
    return wrapped_env


def reset_robosuite_env(env, *, preserve_mjviewer: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if not preserve_mjviewer:
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            raw_obs, info = reset_output
        else:
            raw_obs, info = reset_output, {}
        return raw_obs, info

    if str(getattr(env, "renderer", "")).lower() != "mjviewer":
        raise ValueError("preserve_mjviewer=True is only supported when env.renderer == 'mjviewer'.")

    original_hard_reset = bool(getattr(env, "hard_reset", False))
    env.hard_reset = False
    try:
        env.sim.reset()
        env._reset_internal()
        env.sim.forward()
        env._obs_cache = {}
        env._reset_observables()
        env.visualize(vis_settings={vis: False for vis in env._visualizations})
        env.update_state()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "reset"):
            env.viewer.reset()
        raw_obs = env.viewer._get_observations(force_update=True) if env.viewer_get_obs else env._get_observations(force_update=True)
        return raw_obs, {}
    finally:
        env.hard_reset = original_hard_reset


def reset_observation_adapter(
    adapter: RobosuiteObservationAdapter,
    *,
    preserve_mjviewer: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_obs, info = reset_robosuite_env(adapter.env, preserve_mjviewer=preserve_mjviewer)
    adapter._cached_images = None
    adapter._last_image_time = None
    return adapter.transform(raw_obs, force_render=True), info


def write_resolved_config(cfg: DictConfig, run_dir: Path) -> None:
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    (run_dir / "config_resolved.yaml").write_text(resolved_yaml, encoding="utf-8")


def write_run_info(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "run_info.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


class ConsoleLogCapture:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._stdout = None
        self._stderr = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = TeeStream(self._stdout, self._file)
        sys.stderr = TeeStream(self._stderr, self._file)

    def stop(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class AsyncTransitionChunkWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        chunk_size: int,
        event_logger: callable | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.chunk_size = max(1, int(chunk_size))
        self.event_logger = event_logger
        self._condition = threading.Condition()
        self._pending_online: deque[Any] = deque()
        self._pending_demo: deque[Any] = deque()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._error: BaseException | None = None
        self._busy = False
        self._flush_requested = False
        self._online_chunk_index = 0
        self._demo_chunk_index = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "online_chunks").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "demo_chunks").mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._worker_loop, name="hil_serl_buffer_writer", daemon=True)
            self._thread.start()

    def request_transition(
        self,
        *,
        online_transition,
        demo_transition=None,
    ) -> None:
        self._raise_error()
        with self._condition:
            self._pending_online.append(online_transition)
            if demo_transition is not None:
                self._pending_demo.append(demo_transition)
            self._condition.notify_all()

    def flush(self, timeout: float | None = None) -> None:
        self._raise_error()
        with self._condition:
            if self._thread is None:
                return
            self._flush_requested = True
            self._condition.notify_all()
            end_time = None if timeout is None else time.monotonic() + float(timeout)
            while self._pending_online or self._pending_demo or self._busy or self._flush_requested:
                self._raise_error()
                remaining = None if end_time is None else max(0.0, end_time - time.monotonic())
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError("Timed out while waiting for transition chunk writer.")
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

    def latest_metadata_path(self) -> Path:
        return self.output_dir / "latest" / "metadata.json"

    def online_buffer_path(self) -> Path:
        return self.output_dir / "online_chunks"

    def demo_buffer_path(self) -> Path:
        return self.output_dir / "demo_chunks"

    def _worker_loop(self) -> None:
        online_batch: list[Any] = []
        demo_batch: list[Any] = []
        while True:
            with self._condition:
                while (
                    len(self._pending_online) == 0
                    and len(self._pending_demo) == 0
                    and not self._flush_requested
                    and not self._stop_requested
                ):
                    self._condition.wait(timeout=0.1)
                while self._pending_online:
                    online_batch.append(self._pending_online.popleft())
                while self._pending_demo:
                    demo_batch.append(self._pending_demo.popleft())
                should_flush = self._flush_requested or self._stop_requested
                if self._stop_requested and len(online_batch) == 0 and len(demo_batch) == 0:
                    return
                should_write = (
                    len(online_batch) >= self.chunk_size
                    or len(demo_batch) >= self.chunk_size
                    or (should_flush and (len(online_batch) > 0 or len(demo_batch) > 0))
                )
                if not should_write:
                    continue
                self._busy = True
                self._condition.notify_all()
            try:
                while len(online_batch) >= self.chunk_size or (should_flush and len(online_batch) > 0):
                    chunk = online_batch[: self.chunk_size]
                    del online_batch[: len(chunk)]
                    self._write_chunk("online_chunks", self._online_chunk_index, chunk)
                    self._online_chunk_index += 1
                    if not should_flush:
                        break
                while len(demo_batch) >= self.chunk_size or (should_flush and len(demo_batch) > 0):
                    chunk = demo_batch[: self.chunk_size]
                    del demo_batch[: len(chunk)]
                    self._write_chunk("demo_chunks", self._demo_chunk_index, chunk)
                    self._demo_chunk_index += 1
                    if not should_flush:
                        break
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._busy = False
                    self._stop_requested = True
                    self._condition.notify_all()
                return
            finally:
                with self._condition:
                    if should_flush and len(online_batch) == 0 and len(demo_batch) == 0:
                        self._flush_requested = False
                    self._busy = False
                    self._condition.notify_all()

    def _write_chunk(self, directory_name: str, chunk_index: int, transitions: list[Any]) -> None:
        chunk_dir = self.output_dir / directory_name
        chunk_path = chunk_dir / f"chunk_{int(chunk_index):08d}.pt"
        payload = {
            "chunk_index": int(chunk_index),
            "transition_count": int(len(transitions)),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "transitions": list(transitions),
        }
        _atomic_torch_save(payload, chunk_path)
        if self.event_logger is not None:
            self.event_logger(
                {
                    "event": "buffer_chunk_written",
                    "stream": directory_name,
                    "chunk_index": int(chunk_index),
                    "transition_count": int(len(transitions)),
                    "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
                }
            )

    def _raise_error(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Transition chunk writer failed: {error}") from error


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def log(self, payload: dict[str, Any]) -> None:
        if self._file is None:
            return
        with self._lock:
            self._file.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.flush()
            self._file.close()
            self._file = None


class AsyncCheckpointWriter:
    def __init__(self, agent, checkpoint_dir: Path) -> None:
        self.agent = agent
        self.checkpoint_dir = checkpoint_dir
        self._condition = threading.Condition()
        self._pending_requests: deque[dict[str, Any]] = deque()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._busy = False
        self._error: BaseException | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            (self.checkpoint_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._worker_loop, name="hil_serl_checkpoint_writer", daemon=True)
            self._thread.start()

    def request_save(
        self,
        *,
        paths: list[Path],
        include_buffers: bool,
        extra: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._raise_error()
        with self._condition:
            self._pending_requests.append(
                {
                    "paths": list(paths),
                    "include_buffers": bool(include_buffers),
                    "extra": None if extra is None else dict(extra),
                    "metadata": {} if metadata is None else dict(metadata),
                }
            )
            self._condition.notify_all()

    def flush(self, timeout: float | None = None) -> None:
        self._raise_error()
        with self._condition:
            if self._thread is None:
                return
            end_time = None if timeout is None else time.monotonic() + float(timeout)
            while self._pending_requests or self._busy:
                self._raise_error()
                remaining = None if end_time is None else max(0.0, end_time - time.monotonic())
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError("Timed out while waiting for checkpoint writer.")
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
        while True:
            with self._condition:
                while not self._pending_requests and not self._stop_requested:
                    self._condition.wait(timeout=0.1)
                if self._stop_requested and not self._pending_requests:
                    return
                request = self._pending_requests.popleft()
                self._busy = True
                self._condition.notify_all()
            try:
                payload = self.agent.build_checkpoint_payload(
                    include_buffers=bool(request["include_buffers"]),
                    extra=request["extra"],
                )
                for path in request["paths"]:
                    self.agent.write_checkpoint_payload(path, payload)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._busy = False
                    self._stop_requested = True
                    self._condition.notify_all()
                return
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def _raise_error(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Checkpoint writer failed: {error}") from error


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


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
        f"learn(c/a/t)={learner_progress['critic_updates']}/{learner_progress['actor_updates']}/{learner_progress['temperature_updates']} "
        f"next_publish_in={learner_progress['updates_until_publish']} pending={pending_updates}"
    )


def format_train_line(step: int, metrics: dict[str, float], pending_updates: int) -> str:
    return (
        f"[train] step={step} "
        f"critic_loss={metrics.get('critic_loss', float('nan')):.4f} "
        f"actor_loss={metrics.get('actor_loss', float('nan')):.4f} "
        f"alpha={metrics.get('alpha', float('nan')):.4f} "
        f"learn(c/a/t)={int(metrics.get('learner_critic_updates', 0.0))}/"
        f"{int(metrics.get('learner_actor_updates', 0.0))}/"
        f"{int(metrics.get('learner_temperature_updates', 0.0))} "
        f"next_publish_in={int(metrics.get('learner_updates_until_publish', 0.0))} pending={pending_updates}"
    )


def format_publish_line(metrics: dict[str, float]) -> str:
    return (
        f"[publish] policy #{int(metrics.get('learner_publish_count', 0.0))} synced "
        f"at learner_update={int(metrics.get('learner_last_published_update', 0.0))} "
        f"(critic={int(metrics.get('learner_critic_updates', 0.0))}, "
        f"actor={int(metrics.get('learner_actor_updates', 0.0))}, "
        f"temp={int(metrics.get('learner_temperature_updates', 0.0))})"
    )


def format_episode_line(
    *,
    step: int,
    episode_index: int,
    episode_return: float,
    episode_length: int,
    success: bool,
    online_buffer_size: int,
    demo_buffer_size: int,
) -> str:
    status = "success" if success else "done"
    return (
        f"[episode] ep={episode_index} step={step} {status} "
        f"return={episode_return:.2f} len={episode_length} "
        f"buffers(on/demo)={online_buffer_size}/{demo_buffer_size}"
    )


@hydra.main(version_base="1.2", config_path="./config", config_name="train_hil_serl")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    camera_names = resolve_camera_names(cfg)
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
    buffer_save_interval = max(1, int(getattr(cfg.runtime, "buffer_save_interval", 200)))
    buffer_writer = AsyncTransitionChunkWriter(
        checkpoint_dir / "buffers",
        chunk_size=buffer_save_interval,
        event_logger=runtime_logger.log,
    )
    buffer_writer.start()
    print(f"[run] {run_name}")
    print(f"[path] {checkpoint_dir}")
    print(f"[env] task={cfg.env.environment} robots={list(cfg.env.robots)} cameras={camera_names}")
    print(f"[view] render_camera={resolve_render_camera(cfg, camera_names)}")
    visualize_gripper_markers = bool(getattr(cfg.runtime, "visualize_gripper_markers", True))

    main_has_renderer = bool(cfg.runtime.interactive) and bool(cfg.runtime.viewer_enabled)
    main_renderer = str(cfg.env.renderer)
    if main_has_renderer and main_renderer != "mjviewer":
        print(
            f"[WARN] Overriding env.renderer='{main_renderer}' to 'mjviewer' so the main training env uses "
            "robosuite's native window renderer."
        )
        main_renderer = "mjviewer"

    main_runtime_cfg = build_runtime_cfg(
        cfg,
        camera_names=camera_names,
        has_renderer=main_has_renderer,
        has_offscreen_renderer=False,
        renderer=main_renderer,
    )
    main_env = build_robosuite_env(main_runtime_cfg)
    main_env = maybe_wrap_visualization(
        main_env,
        enabled=visualize_gripper_markers,
        label="training env",
    )

    obs_runtime_cfg = build_runtime_cfg(
        cfg,
        camera_names=camera_names,
        has_renderer=False,
        has_offscreen_renderer=bool(len(camera_names) > 0),
    )
    obs_render_env = build_robosuite_env(obs_runtime_cfg)
    obs_render_env = maybe_wrap_visualization(
        obs_render_env,
        enabled=visualize_gripper_markers,
        label="observation render env",
    )
    obs_render_env.reset()

    bootstrap_adapter = RobosuiteObservationAdapter(
        main_env,
        render_env=obs_render_env,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        image_obs_fps=float(getattr(cfg.runtime, "image_obs_fps", cfg.runtime.control_fps)),
    )
    initial_obs, _ = reset_observation_adapter(bootstrap_adapter, preserve_mjviewer=main_has_renderer)
    if main_has_renderer and getattr(main_env, "viewer", None) is not None and hasattr(main_env.viewer, "update"):
        main_env.viewer.update()

    action_low, action_high = bootstrap_adapter.action_spec()
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if isinstance(algorithm_cfg, dict):
        encoder_cfg = algorithm_cfg.get("encoder", None)
        if isinstance(encoder_cfg, dict) and encoder_cfg.get("pretrained_path"):
            encoder_cfg["pretrained_path"] = to_absolute_path(str(encoder_cfg["pretrained_path"]))
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
    trainer = HILSERLTrainer(agent)

    demo_source_name, demo_paths, max_num_trajectories = resolve_demo_inputs(cfg)
    if not demo_paths:
        raise FileNotFoundError(
            "HIL-SERL requires offline demos before training starts. "
            f"No demo files were found for '{demo_source_name}'. "
            "Place demos under ./data/<task>/expert or set data.demo_paths explicitly."
        )

    env = main_env
    adapter = bootstrap_adapter
    obs = initial_obs
    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    image_obs_fps = resolve_runtime_fps(cfg, "image_obs_fps", control_fps)
    fps_log_interval = max(0.1, float(getattr(cfg.runtime, "fps_log_interval", 1.0)))
    async_updates = bool(getattr(cfg.runtime, "async_updates", False))
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

    if main_has_renderer:
        print(
            "[INFO] Main training env owns the robosuite mjviewer window; "
            "image observations are rendered from a separate headless env."
        )

    # set resume path
    loaded_checkpoint = None
    extra: dict[str, Any] = {}
    for candidate in resume_checkpoint_candidates(cfg, output_root):
        try:
            extra = agent.load_checkpoint(candidate, load_buffers=bool(cfg.runtime.load_buffers))
            loaded_checkpoint = candidate
            print(f"[load] checkpoint={candidate}")
            break
        except Exception as exc:
            print(f"[WARN] Skipping invalid checkpoint {candidate}: {exc}")

    trainer.load_state_dict(extra.get("trainer_state"))
    if bool(cfg.runtime.load_buffers):
        online_buffer_path, demo_buffer_path, buffer_metadata_path = resolve_buffer_snapshot_paths(loaded_checkpoint)
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
            "camera_names": camera_names,
            "render_camera": resolve_render_camera(cfg, camera_names),
            "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
            "resume_enabled": bool(cfg.runtime.resume),
            "load_buffers": bool(cfg.runtime.load_buffers),
            "seed": int(cfg.seed),
            "console_log": str(console_log_path),
            "runtime_log": str(runtime_log_path),
            "buffer_dir": str(checkpoint_dir / "buffers"),
            "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
            "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
        },
    )

    # load offline demos from `./data/<task>/expert`
    print(f"[INFO] Loading offline demos from {demo_source_name}...")
    proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
    shared_demo_cache_dir = output_root / "_demo_cache"
    cache_key_parts = [
        f"h{int(cfg.env.img_height)}",
        f"w{int(cfg.env.img_width)}",
        f"cams-{'_'.join(camera_names)}",
        f"state-{'_'.join(proprio_keys) if proprio_keys else 'auto'}",
    ]
    transitions = load_demo_paths(
        demo_paths,
        cache_dir=shared_demo_cache_dir,
        mirror_cache_dir=checkpoint_dir / "demo_cache",
        hdf5_loader=lambda path, demo_names=None: load_hdf5_demos_into_transitions(
            path,
            camera_names=camera_names,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            demo_names=demo_names,
        ),
        max_num_trajectories=max_num_trajectories,
        cache_key="__".join(cache_key_parts),
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

    if bool(cfg.runtime.load_buffers):
        online_chunk_dir, demo_chunk_dir = resolve_buffer_chunk_dirs(loaded_checkpoint)
        loaded_online_transitions = 0
        loaded_demo_transitions = 0
        if online_chunk_dir is not None:
            agent.online_buffer.clear()
            loaded_online_transitions = load_transition_chunks(agent.online_buffer, online_chunk_dir)
        if demo_chunk_dir is not None:
            loaded_demo_transitions = load_transition_chunks(agent.demo_buffer, demo_chunk_dir)
        if loaded_online_transitions > 0 or loaded_demo_transitions > 0:
            print(
                f"[load] chunk_buffers online={loaded_online_transitions} demo={loaded_demo_transitions} "
                f"from {resolve_checkpoint_run_dir(loaded_checkpoint) / 'buffers'}"
            )

    wandb_run = maybe_build_wandb(cfg, run_name=run_name, run_dir=checkpoint_dir)
    device = None
    intervention_runtime = None
    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention_runtime = RobosuiteInterventionRuntime(
            env=env,
            device=device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )
        intervention_runtime.start_episode()
    if async_updates:
        trainer.start_async_worker()

    episode_return = 0.0
    episode_length = 0
    success_count = 0
    last_step = start_step - 1
    control_limiter = FixedRateLimiter(control_fps)
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

    def request_checkpoint_save(step: int) -> None:
        step_checkpoint = checkpoint_step_path(
            checkpoint_dir,
            step=step,
            learner_updates=trainer.total_updates,
            episode_index=episode_index,
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

    def refresh_main_viewer() -> None:
        if not main_has_renderer:
            return
        if getattr(env, "viewer", None) is None and hasattr(env, "initialize_renderer"):
            env.initialize_renderer()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
            env.viewer.update()

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
                    "learner_critic_updates": int(metrics.get("learner_critic_updates", 0.0)),
                    "learner_actor_updates": int(metrics.get("learner_actor_updates", 0.0)),
                    "learner_temperature_updates": int(metrics.get("learner_temperature_updates", 0.0)),
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
            "learner_critic_updates": float(learner_progress["critic_updates"]),
            "learner_actor_updates": float(learner_progress["actor_updates"]),
            "learner_temperature_updates": float(learner_progress["temperature_updates"]),
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
                "learner_critic_updates": int(learner_progress["critic_updates"]),
                "learner_actor_updates": int(learner_progress["actor_updates"]),
                "learner_temperature_updates": int(learner_progress["temperature_updates"]),
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

    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = control_limiter.wait()
            overall_fps_tracker.mark()

            if policy_gate.ready(loop_start):
                if step < int(cfg.algorithm.trainer.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    cached_policy_action = agent.select_action(obs, deterministic=bool(cfg.runtime.eval_deterministic))

            env_action = np.asarray(cached_policy_action, dtype=np.float32)
            is_intervention = False
            reset_requested = False
            if intervention_runtime is not None and spacemouse_gate.ready(loop_start):
                # Keep device polling on its own cadence and reuse the latest override between polls.
                override_action, sampled_is_intervention, reset_requested = intervention_runtime.maybe_override_action(
                    cached_policy_action
                )
                if reset_requested:
                    cached_override_action = None
                    cached_is_intervention = False
                elif sampled_is_intervention:
                    cached_override_action = np.asarray(override_action, dtype=np.float32)
                    cached_is_intervention = True
                else:
                    cached_override_action = None
                    cached_is_intervention = False

            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[INFO] Device reset requested. Exiting training loop.")
                    break
                obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=main_has_renderer)
                refresh_main_viewer()
                episode_return = 0.0
                episode_length = 0
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

            # collect current transition
            step_output = env.step(env_action)
            if len(step_output) == 5:
                raw_next_obs, _, done, truncated, info = step_output
                done = bool(done or truncated)
            else:
                raw_next_obs, _, done, info = step_output
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            next_obs = adapter.transform(raw_next_obs)
            done = bool(done or success)
            recorded_transition = trainer.record_transition(
                obs=obs,
                action=env_action,
                next_obs=next_obs,
                done=done,
                reward=reward,
                grasp_penalty=info.get("grasp_penalty") if isinstance(info, dict) else None,
                is_intervention=is_intervention,
                info=dict(info) if isinstance(info, dict) else {"raw_info": info},
                reward_source="env_success",
                demo_source="intervention" if is_intervention else None,
            )
            serialized_transition = agent.online_buffer.snapshot_transition(recorded_transition)
            buffer_writer.request_transition(
                online_transition=serialized_transition,
                demo_transition=serialized_transition if is_intervention else None,
            )
            total_transition_count += 1
            episode_transition_count += 1
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

            # update when necessary
            if async_updates:
                update_metrics_list = trainer.maybe_update_async()
            else:
                update_metrics_list = trainer.maybe_update()
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
                episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": int(success),
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
                obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=main_has_renderer)
                refresh_main_viewer()
                episode_return = 0.0
                episode_length = 0
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
    finally:
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
        buffer_writer.flush(timeout=120.0)
        runtime_logger.log(
            {
                "event": "run_end",
                "step": int(last_step),
                "episode_index": int(episode_index),
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
                "camera_names": camera_names,
                "render_camera": resolve_render_camera(cfg, camera_names),
                "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
                "resume_enabled": bool(cfg.runtime.resume),
                "load_buffers": bool(cfg.runtime.load_buffers),
                "seed": int(cfg.seed),
                "console_log": str(console_log_path),
                "runtime_log": str(runtime_log_path),
                "buffer_dir": str(checkpoint_dir / "buffers"),
                "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
                "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
                "last_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "latest_checkpoint": str(checkpoint_path(checkpoint_dir, "latest")),
                "trainer_state": trainer.state_dict(),
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
                obs_render_env.close()
            finally:
                try:
                    checkpoint_writer.close()
                finally:
                    try:
                        runtime_logger.close()
                    finally:
                        try:
                            buffer_writer.close()
                        finally:
                            console_capture.stop()


if __name__ == "__main__":
    main()
