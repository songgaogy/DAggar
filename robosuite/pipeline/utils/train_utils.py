from __future__ import annotations

import datetime
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import VisualizationWrapper

from robosuite.pipeline.common.environment import (
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


def _as_scalar(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.reshape(-1)[0])
    return None


def _tensorboard_tag(key: str) -> str:
    return str(key).strip().replace(" ", "_")


def plt_close(fig) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


class TensorBoardMetricLogger:
    def __init__(self, writer, log_dir: Path) -> None:
        self.writer = writer
        self.log_dir = log_dir

    def log(self, payload: dict[str, Any], step: int) -> None:
        for key, value in payload.items():
            scalar = _as_scalar(value)
            if scalar is None:
                continue
            self.writer.add_scalar(_tensorboard_tag(key), scalar, int(step))

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(_tensorboard_tag(tag), text, int(step))

    def log_figure(self, tag: str, fig, step: int) -> None:
        self.writer.add_figure(_tensorboard_tag(tag), fig, global_step=int(step))
        plt_close(fig)

    def log_histogram(self, tag: str, values: Any, step: int, *, bins: int = 30) -> None:
        self.writer.add_histogram(_tensorboard_tag(tag), values, global_step=int(step), bins=int(bins))

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def maybe_build_tensorboard(cfg: DictConfig, run_name: str | None = None, run_dir: Path | None = None):
    if not bool(getattr(cfg.logging, "use_tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard is not installed. Falling back to stdout and JSONL logging.")
        return None

    base_dir = run_dir if run_dir is not None else Path(to_absolute_path(str(cfg.logging.output_root)))
    tensorboard_dir = Path(str(getattr(cfg.logging, "tensorboard_dir", "tensorboard")))
    log_dir = tensorboard_dir if tensorboard_dir.is_absolute() else base_dir / tensorboard_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    logger = TensorBoardMetricLogger(writer, log_dir)
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)
    logger.log_text("run/config_resolved", f"```\n{resolved_cfg}\n```", step=0)
    if run_name is not None:
        logger.log_text("run/name", str(run_name), step=0)
    print(f"[tensorboard] log_dir={log_dir}")
    return logger


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


def maybe_log(logger, payload: dict[str, Any], step: int) -> None:
    if logger is None:
        return
    logger.log(payload, step=step)


def maybe_log_figure(logger, tag: str, fig, step: int) -> None:
    if logger is None:
        plt_close(fig)
        return
    if hasattr(logger, "log_figure"):
        logger.log_figure(tag, fig, step)
    else:
        plt_close(fig)


def checkpoint_path(checkpoint_dir: Path, tag: str) -> Path:
    return checkpoint_dir / "checkpoints" / f"{tag}.pt"


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
    selected = fallback if requested is None else str(requested).strip()
    if selected == "":
        selected = fallback
    normalized = str(selected).strip().lower()
    if not normalized.startswith("cuda"):
        raise RuntimeError(
            f"Pipeline tensor computation requires CUDA, got device {selected!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device {normalized!r}, but CUDA is unavailable."
        )
    if normalized == "cuda":
        normalized = "cuda:0"
    try:
        device_index = int(normalized.split(":", 1)[1])
    except (IndexError, ValueError):
        raise RuntimeError(f"Invalid CUDA device {normalized!r}.") from None
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device {normalized!r}, but only "
            f"{torch.cuda.device_count()} GPU(s) are visible."
        )
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
def maybe_build_metric_logger(
    cfg: DictConfig,
    run_name: str | None = None,
    run_dir: Path | None = None,
):
    """Build the pipeline's sole supported training logger: TensorBoard."""
    return maybe_build_tensorboard(cfg, run_name=run_name, run_dir=run_dir)
