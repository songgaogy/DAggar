"""PU-BCE failure discriminator runtime for the Flow-DAgger online loop.

This module wraps the frozen ``dyn_disc`` PU-BCE discriminator so it can run as a
*background indicator* during Flow-DAgger rollout. It does **not** influence policy
learning in any way; it only:

  * scores the latest published frame as ``fail`` / ``safe`` at ``inference.fps``;
  * exposes a thread-safe status for a single-line console HUD;
  * (optionally, when ``intervene_env=true``) raises a pause request once the policy
    has been judged ``fail`` for ``consecutive_fail_frames`` consecutive scored
    frames, so the main loop can pause the env and wait for a human.

Threading model (mirrors ``RobosuiteViewerRuntime``): the **main thread** owns the
MuJoCo sim. It renders the discriminator's views, extracts proprio, and calls
:meth:`DiscriminatorRuntime.publish`. The **background thread** only consumes the
latest published snapshot (CPU/GPU tensors) and runs the heavy DINOv3 encode +
head scoring, so the control loop is never blocked by inference.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from robosuite.discriminator.dyn_disc.detectors import DynEncoder, PUBCEDiscriminator


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DiscriminatorConfig:
    """Typed view over ``config/discriminator.yaml`` (all fields overridable)."""

    enabled: bool = True
    task_name: str = ""
    checkpoint: str = ""
    # Optional overrides; ``None`` => use the value embedded in the head checkpoint.
    encoder_ckpt: Optional[str] = None
    feature_source: Optional[str] = None
    transformer_layer: Optional[int] = None
    # Inference
    device: str = "cuda:0"
    fps: float = 2.0
    intervene_env: bool = False
    # Pause debounce / re-arm
    consecutive_fail_frames: int = 3
    require_safe_to_rearm: bool = True
    # HUD
    hud_enabled: bool = True

    @classmethod
    def from_node(cls, node: Any) -> "DiscriminatorConfig":
        """Build from an OmegaConf node (or ``None``)."""
        if node is None:
            return cls(enabled=False)

        def g(path: str, default: Any) -> Any:
            cur: Any = node
            for key in path.split("."):
                if cur is None:
                    return default
                try:
                    cur = cur[key] if key in cur else getattr(cur, key, None)
                except Exception:
                    cur = getattr(cur, key, None)
            return default if cur is None else cur

        return cls(
            enabled=bool(g("enabled", True)),
            task_name=str(g("task_name", "")),
            checkpoint=str(g("checkpoint", "")),
            encoder_ckpt=(
                None
                if g("encoder_ckpt", g("model_ckpt", None)) is None
                else str(g("encoder_ckpt", g("model_ckpt", None)))
            ),
            feature_source=(None if g("feature_source", None) is None else str(g("feature_source", None))),
            transformer_layer=(
                None if g("transformer_layer", None) is None else int(g("transformer_layer", None))
            ),
            device=str(g("inference.device", "cuda:0")),
            fps=float(g("inference.fps", 2.0)),
            intervene_env=bool(g("inference.intervene_env", False)),
            consecutive_fail_frames=int(g("pause.consecutive_fail_frames", 3)),
            require_safe_to_rearm=bool(g("pause.require_safe_to_rearm", True)),
            hud_enabled=bool(g("hud.enabled", True)),
        )


# --------------------------------------------------------------------------- #
# Status snapshot for the HUD
# --------------------------------------------------------------------------- #
@dataclass
class DiscriminatorStatus:
    pred: int = -1  # -1 unknown, 0 safe, 1 fail
    score: float = float("nan")
    threshold: float = float("nan")
    paused: bool = False
    armed: bool = True
    scored_frames: int = 0


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class DiscriminatorRuntime:
    """Background PU-BCE scorer + pause-request state machine."""

    def __init__(self, cfg: DiscriminatorConfig, encoder: DynEncoder, detector: PUBCEDiscriminator) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.detector = detector
        self.task_name = cfg.task_name

        # proprio_map indices into sim.get_state().flatten(), keyed by task.
        proprio_map = getattr(encoder.cfg, "proprio_map", None)
        self.proprio_indices: Optional[List[int]] = None
        if proprio_map is not None and cfg.task_name in proprio_map:
            self.proprio_indices = list(proprio_map[cfg.task_name]["indices"])

        self.view_names: List[str] = list(encoder.view_names)
        self.original_img_size: int = int(encoder.original_img_size)
        self.frameskip: int = int(encoder.frameskip)
        self.action_dim_per_step: int = int(encoder.action_dim_per_step)

        # Shared snapshot (written by main thread, consumed by worker).
        self._cond = threading.Condition()
        self._latest: Optional[Dict[str, Any]] = None
        self._seq = 0
        self._consumed_seq = -1

        # Rolling buffer of executed actions (fallback when no planned chunk).
        self._action_buffer: deque[np.ndarray] = deque(maxlen=self.frameskip)

        # Status / debounce state (guarded by _state_lock).
        self._state_lock = threading.Lock()
        self._status = DiscriminatorStatus()
        self._consecutive_fail = 0
        self._armed = True
        self._pause_event = threading.Event()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----------------------- lifecycle ----------------------- #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._score_loop, name="flow_dagger_discriminator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None

    # ----------------------- main-thread API ----------------------- #
    def extract_proprio(self, flat_sim_state: np.ndarray) -> np.ndarray:
        """Select the 14-dim proprio from a flattened ``sim.get_state()`` (main thread)."""
        if self.proprio_indices is None:
            raise KeyError(
                f"Task {self.task_name!r} not in dynamics proprio_map "
                f"{sorted(getattr(self.encoder.cfg, 'proprio_map', {}) or {})}. "
                "Set discriminator.task_name to match the dynamics model's tasks."
            )
        flat = np.asarray(flat_sim_state, dtype=np.float32)
        return flat[self.proprio_indices].astype(np.float32)

    def publish(
        self,
        *,
        images_per_view: Dict[str, np.ndarray],
        proprio: np.ndarray,
        executed_action: np.ndarray,
        planned_chunk: Optional[np.ndarray] = None,
    ) -> None:
        """Hand the latest frame to the worker. Cheap; never blocks on inference."""
        self._action_buffer.append(np.asarray(executed_action, dtype=np.float32).reshape(-1))
        snapshot = {
            "images_per_view": {k: np.asarray(v) for k, v in images_per_view.items()},
            "proprio": np.asarray(proprio, dtype=np.float32),
            "planned_chunk": None if planned_chunk is None else np.asarray(planned_chunk, dtype=np.float32),
            "action_buffer": np.stack(list(self._action_buffer), axis=0) if self._action_buffer else None,
        }
        with self._cond:
            self._latest = snapshot
            self._seq += 1
            self._cond.notify_all()

    def on_episode_reset(self) -> None:
        self._action_buffer.clear()
        with self._cond:
            self._latest = None
        with self._state_lock:
            self._consecutive_fail = 0
            self._armed = True
            self._status = DiscriminatorStatus(armed=True)
        self._pause_event.clear()

    # ----------------------- pause API ----------------------- #
    def pause_requested(self) -> bool:
        return self.cfg.intervene_env and self._pause_event.is_set()

    def resume(self) -> None:
        """Clear a pending pause and re-arm so it triggers again only after re-failing."""
        self._pause_event.clear()
        with self._state_lock:
            self._consecutive_fail = 0
            # On manual resume, require the discriminator to fail anew before re-pausing.
            self._armed = True
            self._status.paused = False

    def status(self) -> DiscriminatorStatus:
        with self._state_lock:
            st = DiscriminatorStatus(
                pred=self._status.pred,
                score=self._status.score,
                threshold=self._status.threshold,
                paused=self._pause_event.is_set(),
                armed=self._armed,
                scored_frames=self._status.scored_frames,
            )
        return st

    # ----------------------- worker ----------------------- #
    def _score_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._cond:
                while (
                    not self._stop_event.is_set()
                    and (self._latest is None or self._seq == self._consumed_seq)
                ):
                    self._cond.wait(timeout=0.2)
                if self._stop_event.is_set():
                    return
                snapshot = self._latest
                self._consumed_seq = self._seq
            if snapshot is None:
                continue
            try:
                pred, score, tau = self._score_snapshot(snapshot)
            except Exception as exc:  # never let the indicator kill the run
                print(f"[discriminator] scoring error: {type(exc).__name__}: {exc}", flush=True)
                continue
            self._update_debounce(pred, score, tau)

    def _score_snapshot(self, snapshot: Dict[str, Any]) -> tuple[int, float, float]:
        device = self.encoder.device
        size = self.original_img_size

        images_per_view: Dict[str, torch.Tensor] = {}
        for view in self.view_names:
            img = np.asarray(snapshot["images_per_view"][view])
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
                if img.max() > 1.5:
                    img = img / 255.0
            # HWC -> CHW, add batch dim -> (1, 3, H, W)
            t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
            images_per_view[view] = t

        proprio = torch.from_numpy(snapshot["proprio"].reshape(1, -1)).to(device)

        # Action window: prefer the policy's planned future chunk (matches the
        # dynamics model's source = action[t:t+frameskip]); else last executed actions.
        chunk = snapshot.get("planned_chunk")
        if chunk is None:
            chunk = snapshot.get("action_buffer")
        if chunk is None:
            chunk = np.zeros((self.frameskip, self.action_dim_per_step), dtype=np.float32)
        actions_flat = self.encoder.prepare_actions(np.asarray(chunk, dtype=np.float32), t_len=1)
        actions = torch.from_numpy(actions_flat).to(device)

        feat = self.encoder.encode_batch(images_per_view=images_per_view, proprio=proprio, actions=actions)
        result = self.detector.score(feat, task=self.task_name)
        return int(result.preds[0]), float(result.step_scores[0]), float(result.thresholds[0])

    def _update_debounce(self, pred: int, score: float, tau: float) -> None:
        trigger = False
        with self._state_lock:
            self._status.pred = int(pred)
            self._status.score = float(score)
            self._status.threshold = float(tau)
            self._status.scored_frames += 1

            if pred == 1:
                self._consecutive_fail += 1
                if (
                    self.cfg.intervene_env
                    and self._armed
                    and self._consecutive_fail >= int(self.cfg.consecutive_fail_frames)
                ):
                    trigger = True
                    self._armed = False
            else:  # safe -> reset counter, re-arm
                self._consecutive_fail = 0
                if self.cfg.require_safe_to_rearm:
                    self._armed = True
        if trigger:
            self._pause_event.set()


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_discriminator_runtime(node: Any) -> Optional[DiscriminatorRuntime]:
    """Construct a :class:`DiscriminatorRuntime` from an OmegaConf node, or ``None``."""
    cfg = DiscriminatorConfig.from_node(node)
    if not cfg.enabled:
        return None

    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Discriminator checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_ckpt = cfg.encoder_ckpt or str(payload["model_ckpt"])
    feature_source = cfg.feature_source or str(payload.get("feature_source", "transformer"))
    transformer_layer = (
        cfg.transformer_layer
        if cfg.transformer_layer is not None
        else int(payload.get("transformer_layer", 1))
    )

    encoder = DynEncoder(
        model_ckpt=model_ckpt,
        device=cfg.device,
        feature_source=feature_source,
        transformer_layer=transformer_layer,
    )
    detector = PUBCEDiscriminator(
        in_dim=int(payload["in_dim"]),
        hidden=int(payload["hidden"]),
        num_layers=int(payload["num_layers"]),
        device=cfg.device,
    )
    detector.load_state_dict(payload["pu_bce_detector"])

    print(
        f"[discriminator] loaded head={ckpt_path.name} task={cfg.task_name} "
        f"feature_source={feature_source} layer={transformer_layer} "
        f"fps={cfg.fps} intervene_env={cfg.intervene_env}",
        flush=True,
    )
    return DiscriminatorRuntime(cfg, encoder, detector)


# --------------------------------------------------------------------------- #
# Console HUD + ENTER listener
# --------------------------------------------------------------------------- #
_ANSI = {
    "fail": "\033[1;31m",  # bold red
    "safe": "\033[1;32m",  # bold green
    "paused": "\033[1;33m",  # bold yellow
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def render_hud(
    out,
    *,
    status: DiscriminatorStatus,
    episode_index: int,
    step: int,
    episode_step: int,
    pending_updates: int,
) -> None:
    """Single-line, in-place (``\\r``) status bar written to the real terminal."""
    if status.paused:
        tag = f"{_ANSI['paused']}[ FAIL · PAUSED ]{_ANSI['reset']}"
        hint = f" {_ANSI['dim']}ENTER dagger/bin{_ANSI['reset']}"
    elif status.pred == 1:
        tag = f"{_ANSI['fail']}[ FAIL ]{_ANSI['reset']}"
        hint = ""
    elif status.pred == 0:
        tag = f"{_ANSI['safe']}[ SAFE ]{_ANSI['reset']}"
        hint = ""
    else:
        tag = f"{_ANSI['dim']}[ .... ]{_ANSI['reset']}"
        hint = ""
    score = "n/a" if status.score != status.score else f"{status.score:+.3f}"
    line = (
        f"\r{tag} ep={episode_index:>3d} step={step:>7d} "
        f"epstep={episode_step:>4d} pending={pending_updates:>2d} score={score}{hint}"
    )
    try:
        out.write(line + "\033[K")  # clear to end of line
        out.flush()
    except Exception:
        pass


class EnterKeyListener:
    """Background thread that sets an event whenever the user presses ENTER on stdin."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="flow_dagger_enter_listener", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":  # EOF
                return
            self._event.set()

    def pressed(self) -> bool:
        """Consume and return whether ENTER was pressed since the last check."""
        if self._event.is_set():
            self._event.clear()
            return True
        return False

    def clear(self) -> None:
        self._event.clear()

    def stop(self) -> None:
        self._stop.set()
