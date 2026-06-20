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

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import VisualizationWrapper

from robosuite.pipeline.envs import (
    RobosuiteObservationAdapter,
    RobosuiteRuntimeConfig,
)
from robosuite.pipeline.utils import resolve_task_demo_paths


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
        if data and ("\n" in data or "\r" in data):
            for stream in self._streams:
                stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


class ConsoleLogCapture:
    def __init__(self, path: Path, *, file_only: bool = False) -> None:
        self.path = path
        # When True, stdout/stderr are redirected to the log file only (no terminal
        # echo). The original terminal stdout is preserved on ``terminal_stdout`` so
        # callers (e.g. the discriminator HUD) can still draw to the real console.
        self.file_only = bool(file_only)
        self._file = None
        self._stdout = None
        self._stderr = None
        self.terminal_stdout = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self.terminal_stdout = self._stdout
        if self.file_only:
            sys.stdout = self._file
            sys.stderr = self._file
        else:
            sys.stdout = TeeStream(self._stdout, self._file)
            sys.stderr = TeeStream(self._stderr, self._file)

    def set_file_only(self, file_only: bool) -> None:
        """Toggle file-only redirection at runtime (e.g. once rollout starts)."""
        file_only = bool(file_only)
        if file_only == self.file_only or self._file is None:
            self.file_only = file_only
            return
        self.file_only = file_only
        if file_only:
            sys.stdout = self._file
            sys.stderr = self._file
        else:
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
        f"[episode] ep={episode_index} step={step} {status} is_success={str(bool(success)).lower()} "
        f"return={episode_return:.2f} len={episode_length} "
        f"buffers(on/demo)={online_buffer_size}/{demo_buffer_size}"
    )
