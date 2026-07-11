"""Non-blocking nnPU HUD scorer and pause state machine for HIL rollout."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .encoder import SharedDynamicsEncoder
from .nnpu import FrozenNNPUDiscriminator


@dataclass
class NNPURuntimeConfig:
    enabled: bool = True
    task_name: str = ""
    checkpoint: str = ""
    encoder_ckpt: str | None = None
    camera_to_view: dict[str, str] | None = None
    device: str = "cuda:0"
    fps: float = 0.0  # <= 0 means score once for each newly planned chunk
    intervene_env: bool = False
    consecutive_fail_frames: int = 3
    require_safe_to_rearm: bool = True
    hud_enabled: bool = True

    @classmethod
    def from_node(cls, node: Any) -> "NNPURuntimeConfig":
        if node is None:
            return cls(enabled=False)

        def get(path: str, default: Any) -> Any:
            value = node
            for key in path.split("."):
                try:
                    value = value[key] if key in value else getattr(value, key, None)
                except (KeyError, TypeError, AttributeError):
                    value = getattr(value, key, None)
                if value is None:
                    return default
            return value

        encoder_ckpt = get("encoder_ckpt", None)
        camera_to_view = get("camera_to_view", None)
        return cls(
            enabled=bool(get("enabled", True)),
            task_name=str(get("task_name", "")),
            checkpoint=str(get("checkpoint", get("nnpu_ckpt", ""))),
            encoder_ckpt=None if encoder_ckpt is None else str(encoder_ckpt),
            camera_to_view=(
                None if camera_to_view is None else {str(k): str(v) for k, v in dict(camera_to_view).items()}
            ),
            device=str(get("inference.device", get("device", "cuda:0"))),
            fps=float(get("inference.fps", 0.0)),
            intervene_env=bool(get("inference.intervene_env", False)),
            consecutive_fail_frames=int(get("pause.consecutive_fail_frames", 3)),
            require_safe_to_rearm=bool(get("pause.require_safe_to_rearm", True)),
            hud_enabled=bool(get("hud.enabled", True)),
        )


@dataclass
class NNPUStatus:
    pred: int = -1
    score: float = float("nan")
    threshold: float = float("nan")
    paused: bool = False
    armed: bool = True
    scored_chunks: int = 0
    error: str | None = None


class NNPUDiscriminatorRuntime:
    """Latest-snapshot background scorer; the main thread remains MuJoCo owner."""

    def __init__(
        self,
        cfg: NNPURuntimeConfig,
        encoder: SharedDynamicsEncoder,
        discriminator: FrozenNNPUDiscriminator,
        *,
        policy_camera_names: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.discriminator = discriminator
        self.policy_camera_names = [
            str(name) for name in (policy_camera_names or encoder.view_names)
        ]
        self.encoder.bind_policy_cameras(self.policy_camera_names)

        self._condition = threading.Condition()
        self._latest: dict[str, Any] | None = None
        self._sequence = 0
        self._consumed_sequence = -1
        self._state_lock = threading.Lock()
        self._status = NNPUStatus(threshold=float(discriminator.threshold))
        self._consecutive_failures = 0
        self._armed = True
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._score_loop, name="dipole_nnpu_scorer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def publish(
        self,
        *,
        images_per_view: Mapping[str, np.ndarray],
        proprio: np.ndarray,
        executed_action: np.ndarray,
        planned_chunk: np.ndarray | None = None,
        is_new_chunk: bool = True,
    ) -> None:
        """Publish a snapshot; chunk mode ignores non-boundary publications."""
        if self.cfg.fps <= 0.0 and not is_new_chunk:
            return
        snapshot = {
            "images": {name: np.asarray(images_per_view[name]).copy() for name in self.policy_camera_names},
            "proprio": np.asarray(proprio, dtype=np.float32).copy(),
            "actions": np.asarray(
                planned_chunk if planned_chunk is not None else executed_action,
                dtype=np.float32,
            ).copy(),
        }
        with self._condition:
            self._latest = snapshot
            self._sequence += 1
            self._condition.notify_all()

    def on_episode_reset(self) -> None:
        with self._condition:
            self._latest = None
        with self._state_lock:
            self._consecutive_failures = 0
            self._armed = True
            self._status = NNPUStatus(threshold=float(self.discriminator.threshold))
        self._pause_event.clear()

    def pause_requested(self) -> bool:
        return bool(self.cfg.intervene_env and self._pause_event.is_set())

    def resume(self) -> None:
        """Resume from ENTER or SpaceMouse and require a fresh failure streak."""
        self._pause_event.clear()
        with self._state_lock:
            self._consecutive_failures = 0
            self._armed = True
            self._status.paused = False

    def status(self) -> NNPUStatus:
        with self._state_lock:
            return NNPUStatus(
                pred=self._status.pred,
                score=self._status.score,
                threshold=self._status.threshold,
                paused=self._pause_event.is_set(),
                armed=self._armed,
                scored_chunks=self._status.scored_chunks,
                error=self._status.error,
            )

    def _score_loop(self) -> None:
        minimum_period = 0.0 if self.cfg.fps <= 0.0 else 1.0 / self.cfg.fps
        last_score_time = 0.0
        while not self._stop_event.is_set():
            with self._condition:
                while (
                    not self._stop_event.is_set()
                    and (self._latest is None or self._sequence == self._consumed_sequence)
                ):
                    self._condition.wait(timeout=0.2)
                if self._stop_event.is_set():
                    return
                snapshot = self._latest
                sequence = self._sequence
            delay = minimum_period - (time.monotonic() - last_score_time)
            if delay > 0.0 and self._stop_event.wait(delay):
                return
            with self._condition:
                snapshot = self._latest
                sequence = self._sequence
                self._consumed_sequence = sequence
            if snapshot is None:
                continue
            try:
                pred, score = self._score_snapshot(snapshot)
            except Exception as exc:  # HUD failure must not terminate training
                with self._state_lock:
                    self._status.error = f"{type(exc).__name__}: {exc}"
                print(f"[nnPU HUD] scoring error: {type(exc).__name__}: {exc}", flush=True)
                continue
            last_score_time = time.monotonic()
            self._update_debounce(pred, score)

    @torch.no_grad()
    def _score_snapshot(self, snapshot: dict[str, Any]) -> tuple[int, float]:
        views = []
        for name in self.policy_camera_names:
            image = np.asarray(snapshot["images"][name])
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
                if image.size and image.max() > 1.5:
                    image = image / 255.0
            views.append(np.transpose(image, (2, 0, 1)))
        image_tensor = torch.from_numpy(np.stack(views, axis=0)).unsqueeze(0)
        proprio_tensor = torch.from_numpy(snapshot["proprio"]).reshape(1, -1)
        action_tensor = torch.from_numpy(snapshot["actions"])
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor.unsqueeze(0)
        feature = self.encoder.encode_chunk(
            image_obs_raw=image_tensor,
            proprio_raw=proprio_tensor,
            action_chunk=action_tensor,
        )
        output = self.discriminator.score(chunk_feature=feature)
        return int(output.decision.reshape(-1)[0]), float(output.logit.reshape(-1)[0])

    def _update_debounce(self, pred: int, score: float) -> None:
        trigger = False
        with self._state_lock:
            self._status.pred = int(pred)
            self._status.score = float(score)
            self._status.threshold = float(self.discriminator.threshold)
            self._status.scored_chunks += 1
            self._status.error = None
            if pred:
                self._consecutive_failures += 1
                if (
                    self.cfg.intervene_env
                    and self._armed
                    and self._consecutive_failures >= self.cfg.consecutive_fail_frames
                ):
                    self._armed = False
                    trigger = True
            else:
                self._consecutive_failures = 0
                if self.cfg.require_safe_to_rearm:
                    self._armed = True
        if trigger:
            self._pause_event.set()


def build_nnpu_runtime(
    node: Any,
    *,
    policy_camera_names: Sequence[str] | None = None,
    shared_encoder: SharedDynamicsEncoder | None = None,
    discriminator: FrozenNNPUDiscriminator | None = None,
) -> NNPUDiscriminatorRuntime | None:
    """Build the optional HUD runtime; callers may catch errors and disable it."""
    cfg = NNPURuntimeConfig.from_node(node)
    if not cfg.enabled:
        return None
    if not cfg.checkpoint:
        raise ValueError("discriminator.checkpoint / nnpu_ckpt is required")
    if not Path(cfg.checkpoint).expanduser().exists():
        raise FileNotFoundError(f"nnPU checkpoint not found: {cfg.checkpoint}")
    encoder = shared_encoder or SharedDynamicsEncoder(
        cfg.checkpoint,
        encoder_ckpt=cfg.encoder_ckpt,
        device=cfg.device,
        camera_to_view=cfg.camera_to_view,
    )
    frozen = discriminator or FrozenNNPUDiscriminator(
        cfg.checkpoint, task_name=cfg.task_name, device=cfg.device, encoder=encoder
    )
    return NNPUDiscriminatorRuntime(
        cfg, encoder, frozen, policy_camera_names=policy_camera_names
    )


def render_nnpu_hud(out: Any, status: NNPUStatus, *, step: int, episode_step: int) -> None:
    """Render one compact terminal line; all output failures are ignored."""
    red = "\033[1;31m"
    green = "\033[1;32m"
    yellow = "\033[1;33m"
    reset = "\033[0m"
    if status.paused:
        label = "FAIL · PAUSED (ENTER/SpaceMouse to resume)"
        color = yellow
    elif status.pred == 1:
        label = "FAIL"
        color = red
    elif status.pred == 0:
        label = "SAFE"
        color = green
    else:
        label = "...."
        color = yellow
    score = "n/a" if not np.isfinite(status.score) else f"{status.score:+.3f}"
    try:
        out.write(
            f"\r[nnPU {color}{label}{reset}] step={step} epstep={episode_step} "
            f"score={score} tau={status.threshold:+.3f}\033[K"
        )
        out.flush()
    except Exception:
        pass


class EnterKeyListener:
    """Portable background ENTER event source for pause/resume loops."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or not sys.stdin.isatty():
            return

        def listen() -> None:
            while True:
                if sys.stdin.readline() == "":
                    return
                self.event.set()

        self._thread = threading.Thread(target=listen, name="nnpu_enter_listener", daemon=True)
        self._thread.start()

    def consume(self) -> bool:
        value = self.event.is_set()
        self.event.clear()
        return value


__all__ = [
    "EnterKeyListener",
    "NNPUDiscriminatorRuntime",
    "NNPURuntimeConfig",
    "NNPUStatus",
    "build_nnpu_runtime",
    "render_nnpu_hud",
]
