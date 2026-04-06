from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.bce.model import build_temporal_pu_discriminator
from robosuite.discriminator.lpb_new.core.dataset import LatentTrajectory
from robosuite.discriminator.utils.base import OfflineTrajectoryDiscriminator
from robosuite.discriminator.utils.types import DetectorCalibrationSummary, TrajectoryDetectionResult


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            try:
                cur = cur[key]
            except Exception:
                try:
                    cur = getattr(cur, key)
                except Exception:
                    return default
    return default if cur is None else cur


def _resolve_device(device: str) -> torch.device:
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _torch_load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _moving_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return vals
    window = max(1, int(window_size))
    csum = np.cumsum(vals, dtype=np.float64)
    idx = np.arange(vals.shape[0], dtype=np.int64)
    start = np.maximum(0, idx - window + 1)
    start_minus = start - 1
    left = np.where(start_minus >= 0, csum[start_minus], 0.0)
    win_sum = csum - left
    denom = (idx - start + 1).astype(np.float64)
    return (win_sum / denom).astype(np.float32)


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return vals
    a = float(np.clip(alpha, 1e-4, 1.0))
    out = np.empty_like(vals, dtype=np.float32)
    out[0] = vals[0]
    for idx in range(1, vals.shape[0]):
        out[idx] = a * vals[idx] + (1.0 - a) * out[idx - 1]
    return out


def _aggregate_scores(values: np.ndarray, mode: str, moving_mean_window: int, ema_alpha: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if mode == "identity":
        return vals
    if mode == "mean":
        if vals.size == 0:
            return vals
        csum = np.cumsum(vals, dtype=np.float64)
        denom = np.arange(1, vals.shape[0] + 1, dtype=np.float64)
        return (csum / denom).astype(np.float32)
    if mode == "max":
        return np.maximum.accumulate(vals)
    if mode == "moving_mean":
        return _moving_mean(vals, window_size=int(moving_mean_window))
    if mode == "ema":
        return _ema(vals, alpha=float(ema_alpha))
    raise ValueError(f"Unsupported aggregate_mode: {mode}")


def _compute_threshold(values: np.ndarray, delta: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot calibrate threshold from empty values")
    q = 100.0 * (1.0 - float(delta) / 100.0)
    return float(np.percentile(values.astype(np.float64), q=q))


@dataclass
class _ScoreBundle:
    step_scores: np.ndarray
    aggregate_scores: np.ndarray


class TPUDDiscriminator(OfflineTrajectoryDiscriminator[LatentTrajectory]):
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        batch_size: int = 512,
        action_horizon: int = -1,
        aggregate_mode: str = "identity",
        default_delta: float = 10.0,
        task_delta: Optional[dict[str, float]] = None,
        delta_step: float = 0.5,
        moving_mean_window: int = 5,
        ema_alpha: float = 0.15,
        min_persistence: int = 1,
        decision_warmup_steps: int = 0,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.aggregate_mode = str(aggregate_mode)
        self.default_delta = float(default_delta)
        self.task_delta = {str(k): float(v) for k, v in (task_delta or {}).items()}
        self.delta_step = float(delta_step)
        self.moving_mean_window = max(1, int(moving_mean_window))
        self.ema_alpha = float(ema_alpha)
        self.min_persistence = max(1, int(min_persistence))
        self.decision_warmup_steps = max(0, int(decision_warmup_steps))
        self.threshold_by_task: dict[str, float] = {}
        self.calibration_scores_by_task: dict[str, np.ndarray] = {}

        payload = _torch_load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.payload = payload
        self.task_to_index = {
            str(task_name): int(task_index)
            for task_name, task_index in dict(payload["task_to_index"]).items()
        }
        self.index_to_task = {
            int(task_index): str(task_name)
            for task_name, task_index in self.task_to_index.items()
        }
        self.horizon_ckpt = int(payload.get("horizon", 1))
        self.action_horizon = self.horizon_ckpt if int(action_horizon) <= 0 else int(action_horizon)
        self.model = build_temporal_pu_discriminator(
            latent_dim=int(payload["latent_dim"]),
            action_dim=int(payload["action_dim"]),
            num_tasks=int(len(self.task_to_index)),
            cfg_model=_cfg_get(payload.get("cfg", None), "model", {}),
            transition_horizon=int(self.action_horizon),
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def name(self) -> str:
        return "tpud_bce"

    def _resolve_delta(self, task_name: str) -> float:
        return float(self.task_delta.get(task_name, self.default_delta))

    def _prepare_actions(self, actions: np.ndarray, t_len: int) -> np.ndarray:
        actions = np.asarray(actions[:t_len], dtype=np.float32)
        action_dim = int(self.payload["action_dim"])
        if actions.shape[1] < action_dim:
            pad = np.zeros((actions.shape[0], action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > action_dim:
            actions = actions[:, :action_dim]

        horizon = int(self.action_horizon)
        valid_len = t_len - horizon
        chunks = np.zeros((valid_len, horizon, action_dim), dtype=np.float32)
        for t in range(valid_len):
            chunk = actions[t : t + horizon]
            chunks[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < horizon:
                fill = chunk[-1] if chunk.shape[0] > 0 else np.zeros((action_dim,), dtype=np.float32)
                chunks[t, chunk.shape[0] :] = fill
        return chunks

    @torch.no_grad()
    def _score_trajectory(self, trajectory: LatentTrajectory) -> _ScoreBundle:
        t_len = min(int(trajectory.latents.shape[0]), int(trajectory.actions.shape[0]))
        valid_len = t_len - int(self.action_horizon)
        if valid_len <= 0:
            raise ValueError(
                f"Trajectory length {t_len} is too short for action_horizon={self.action_horizon}"
            )

        latents = np.asarray(trajectory.latents[:valid_len], dtype=np.float32)
        action_chunks = self._prepare_actions(trajectory.actions, t_len=t_len)
        task_index = int(trajectory.task_index)

        step_scores: list[np.ndarray] = []
        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            latent_batch = torch.from_numpy(latents[start:end]).to(self.device)
            action_batch = torch.from_numpy(action_chunks[start:end]).to(self.device)
            task_batch = torch.full(
                (end - start,),
                task_index,
                dtype=torch.int64,
                device=self.device,
            )
            outputs = self.model(
                current_latent=latent_batch,
                action_sequence=action_batch,
                task_index=task_batch,
            )
            ood_scores = 1.0 - outputs["probs"]
            step_scores.append(ood_scores.detach().cpu().numpy().astype(np.float32))

        step_scores_np = np.concatenate(step_scores, axis=0)
        return _ScoreBundle(
            step_scores=step_scores_np,
            aggregate_scores=_aggregate_scores(
                step_scores_np,
                mode=self.aggregate_mode,
                moving_mean_window=self.moving_mean_window,
                ema_alpha=self.ema_alpha,
            ),
        )

    @torch.no_grad()
    def fit(
        self,
        normal_bank_trajectories: Sequence[LatentTrajectory],
        calibration_trajectories: Optional[Sequence[LatentTrajectory]] = None,
    ) -> DetectorCalibrationSummary:
        bank_by_task: dict[str, list[np.ndarray]] = {}
        calib_by_task: dict[str, list[np.ndarray]] = {}

        for traj in normal_bank_trajectories:
            bundle = self._score_trajectory(traj)
            bank_by_task.setdefault(str(traj.task_name), []).append(bundle.aggregate_scores)

        source_trajectories = calibration_trajectories if calibration_trajectories is not None else normal_bank_trajectories
        for traj in source_trajectories:
            bundle = self._score_trajectory(traj)
            calib_by_task.setdefault(str(traj.task_name), []).append(bundle.aggregate_scores)

        self.calibration_scores_by_task = {}
        self.threshold_by_task = {}
        all_tasks = sorted(set(bank_by_task.keys()) | set(calib_by_task.keys()))
        for task_name in all_tasks:
            values = calib_by_task.get(task_name)
            if values is None or len(values) == 0:
                values = bank_by_task.get(task_name, [])
            if len(values) == 0:
                continue
            merged = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
            self.calibration_scores_by_task[task_name] = merged.astype(np.float32)
            self.threshold_by_task[task_name] = _compute_threshold(
                merged,
                delta=self._resolve_delta(task_name),
            )

        if not self.threshold_by_task:
            raise RuntimeError("TPUD calibration failed: no task-specific thresholds were produced.")

        return DetectorCalibrationSummary(
            detector_name=self.name,
            threshold=None,
            metadata={
                "threshold_by_task": dict(self.threshold_by_task),
                "delta_by_task": {
                    task_name: self._resolve_delta(task_name)
                    for task_name in sorted(self.threshold_by_task.keys())
                },
                "aggregate_mode": self.aggregate_mode,
                "moving_mean_window": int(self.moving_mean_window),
                "ema_alpha": float(self.ema_alpha),
                "min_persistence": int(self.min_persistence),
                "decision_warmup_steps": int(self.decision_warmup_steps),
                "checkpoint_path": self.checkpoint_path,
            },
        )

    @torch.no_grad()
    def detect_trajectory(
        self,
        trajectory: LatentTrajectory,
        *,
        labels: Optional[np.ndarray] = None,
        adaptive_threshold: bool = False,
        delta_min: float = 0.0,
        delta_max: float = 100.0,
        warmup_steps: int = 0,
        update_interval: int = 1,
    ) -> TrajectoryDetectionResult:
        task_name = str(trajectory.task_name)
        if task_name not in self.threshold_by_task:
            raise KeyError(f"Task '{task_name}' was not calibrated. Available tasks: {sorted(self.threshold_by_task.keys())}")

        bundle = self._score_trajectory(trajectory)
        aggregate_scores = bundle.aggregate_scores.astype(np.float32)
        calibration_scores = self.calibration_scores_by_task[task_name]
        current_delta = float(self._resolve_delta(task_name))
        current_threshold = float(self.threshold_by_task[task_name])
        thresholds = np.zeros_like(aggregate_scores, dtype=np.float32)
        predictions = np.zeros_like(aggregate_scores, dtype=np.int64)
        dmin = float(np.clip(delta_min, 0.0, 100.0))
        dmax = float(np.clip(delta_max, 0.0, 100.0))
        if dmin > dmax:
            dmin, dmax = dmax, dmin
        warmup = max(0, int(warmup_steps))
        update_every = max(1, int(update_interval))
        decision_warmup = int(self.decision_warmup_steps)
        persistence = int(self.min_persistence)
        raw_predictions = np.zeros_like(aggregate_scores, dtype=np.int64)
        active_count = 0

        labels_np = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_np is not None and labels_np.shape[0] != aggregate_scores.shape[0]:
            raise ValueError(f"labels shape mismatch: {labels_np.shape} vs {aggregate_scores.shape}")

        for t in range(aggregate_scores.shape[0]):
            raw_pred = int(aggregate_scores[t] >= current_threshold)
            raw_predictions[t] = raw_pred
            if (t + 1) <= decision_warmup:
                active_count = 0
                predictions[t] = 0
            else:
                active_count = (active_count + 1) if raw_pred == 1 else 0
                predictions[t] = int(active_count >= persistence)
            thresholds[t] = float(current_threshold)

            if adaptive_threshold and labels_np is not None:
                should_update = (t + 1) > warmup and ((t + 1 - warmup) % update_every == 0)
                if should_update:
                    label = int(labels_np[t])
                    pred = int(predictions[t])
                    if label == 1 and pred == 0:
                        current_delta += self.delta_step
                    elif label == 0 and pred == 1:
                        current_delta -= self.delta_step
                    current_delta = float(np.clip(current_delta, dmin, dmax))
                    current_threshold = _compute_threshold(calibration_scores, delta=current_delta)

        return TrajectoryDetectionResult(
            detector_name=self.name,
            step_scores=bundle.step_scores.astype(np.float32),
            aggregate_scores=aggregate_scores,
            thresholds=thresholds,
            predictions=predictions,
            labels=labels_np,
            metadata={
                "task_name": task_name,
                "threshold_final": float(current_threshold),
                "delta_final": float(current_delta),
                "threshold_task_init": float(self.threshold_by_task[task_name]),
                "delta_task_init": float(self._resolve_delta(task_name)),
                "aggregate_mode": self.aggregate_mode,
                "moving_mean_window": int(self.moving_mean_window),
                "ema_alpha": float(self.ema_alpha),
                "min_persistence": int(self.min_persistence),
                "decision_warmup_steps": int(self.decision_warmup_steps),
                "raw_positive_count": int(raw_predictions.sum()),
            },
        )

    def close(self) -> None:
        return None


__all__ = [
    "TPUDDiscriminator",
]
