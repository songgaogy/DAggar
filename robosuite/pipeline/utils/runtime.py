from __future__ import annotations

import datetime
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from robosuite.wrappers import VisualizationWrapper

from robosuite.pipeline.src.environment import RobosuiteObservationAdapter, RobosuiteRuntimeConfig


def now_readable() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def set_seed(seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HIL-SERL requires CUDA; CPU tensor execution is not supported.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_camera_names(cfg: DictConfig) -> list[str]:
    names = [str(name) for name in list(cfg.env.camera_names or [])]
    return names or ["agentview"]


def build_runtime_cfg(
    cfg: DictConfig,
    camera_names: list[str],
    *,
    has_renderer: bool,
    has_offscreen_renderer: bool,
    renderer: str | None = None,
) -> RobosuiteRuntimeConfig:
    return RobosuiteRuntimeConfig(
        env_name=str(cfg.env.environment),
        robots=[str(name) for name in list(cfg.env.robots)],
        env_configuration=str(cfg.env.config) if cfg.env.config is not None else None,
        controller=str(cfg.env.controller) if cfg.env.controller is not None else None,
        renderer=str(cfg.env.renderer) if renderer is None else str(renderer),
        render_camera=str(cfg.env.render_camera),
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


def maybe_wrap_visualization(env, *, enabled: bool, label: str):
    if not enabled:
        return env
    print(f"[INFO] Enabled robosuite gripper visualization markers on {label}.")
    return VisualizationWrapper(env)


def reset_observation_adapter(
    adapter: RobosuiteObservationAdapter,
    *,
    preserve_mjviewer: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env = adapter.env
    if not preserve_mjviewer:
        reset_output = env.reset()
        raw_obs, info = reset_output if isinstance(reset_output, tuple) else (reset_output, {})
    else:
        original_hard_reset = bool(getattr(env, "hard_reset", False))
        env.hard_reset = False
        try:
            env.sim.reset()
            env._reset_internal()
            env.sim.forward()
            env._obs_cache = {}
            env._reset_observables()
            env.visualize(vis_settings={name: False for name in env._visualizations})
            env.update_state()
            if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "reset"):
                env.viewer.reset()
            raw_obs = (
                env.viewer._get_observations(force_update=True)
                if env.viewer_get_obs
                else env._get_observations(force_update=True)
            )
            info = {}
        finally:
            env.hard_reset = original_hard_reset
    adapter._cached_images = None
    adapter._last_image_time = None
    return adapter.transform(raw_obs, force_render=True), info


def write_resolved_config(cfg: DictConfig, run_dir: Path) -> None:
    (run_dir / str(cfg.logging.resolved_config_filename)).write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )


class FixedRateLimiter:
    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.period = 1.0 / float(fps)
        self._next: float | None = None

    def wait(self) -> float:
        now = time.monotonic()
        if self._next is None:
            self._next = now + self.period
            return now
        if now < self._next:
            time.sleep(self._next - now)
            now = time.monotonic()
        while self._next <= now:
            self._next += self.period
        return now


class IntervalGate:
    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.period = 1.0 / float(fps)
        self._next: float | None = None

    def ready(self, now: float) -> bool:
        if self._next is None:
            self._next = now + self.period
            return True
        if now < self._next:
            return False
        while self._next <= now:
            self._next += self.period
        return True

    def force_ready(self) -> None:
        self._next = None


class EMAFpsTracker:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = float(alpha)
        self._count = 0
        self._value: float | None = None

    def mark(self, count: int = 1) -> None:
        self._count += int(count)

    def snapshot(self, elapsed: float) -> float:
        current = self._count / max(float(elapsed), 1e-6)
        self._count = 0
        self._value = current if self._value is None else self.alpha * current + (1.0 - self.alpha) * self._value
        return self._value
