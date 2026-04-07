from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.utils.base import OfflineTrajectoryDiscriminator
from robosuite.discriminator.utils.types import DetectorCalibrationSummary, TrajectoryDetectionResult

from .dataset import LatentTrajectory
from .model import DSMModel, build_dsm_model


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


@dataclass
class TrajectoryScoreBundle:
    step_scores: np.ndarray
    state_error_scores: np.ndarray
    action_error_scores: np.ndarray
    next_state_error_scores: np.ndarray
    state_contrib_scores: np.ndarray
    action_contrib_scores: np.ndarray
    next_state_contrib_scores: np.ndarray


class DSMTransitionScorer:
    """Score trajectories with a trained DSM denoiser."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        action_horizon: int = -1,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.model, self.action_horizon, self.action_dim, self.latent_dim = self._load_model(
            checkpoint_path=checkpoint_path,
            override_action_horizon=action_horizon,
        )

    def _load_model(
        self,
        checkpoint_path: str,
        override_action_horizon: int,
    ) -> tuple[DSMModel, int, int, int]:
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing key `model`: {checkpoint_path}")

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim"))
        action_dim = int(payload.get("action_dim"))
        horizon_ckpt = int(payload.get("horizon", 1))
        if int(override_action_horizon) > 0 and int(override_action_horizon) != horizon_ckpt:
            raise ValueError(
                "Inference action_horizon must match the training horizon stored in the checkpoint: "
                f"got override={int(override_action_horizon)}, checkpoint={horizon_ckpt}"
            )
        action_horizon = horizon_ckpt if int(override_action_horizon) <= 0 else int(override_action_horizon)

        model = build_dsm_model(
            latent_dim=latent_dim,
            action_dim=action_dim,
            cfg_model=_cfg_get(cfg, "model", {}),
            transition_horizon=action_horizon,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(self.device)
        model.eval()
        return model, action_horizon, action_dim, latent_dim

    def _prepare_latents(self, latents: np.ndarray, t_len: int) -> np.ndarray:
        latents = np.asarray(latents[:t_len], dtype=np.float32)
        if latents.shape[1] < self.latent_dim:
            pad = np.zeros((latents.shape[0], self.latent_dim - latents.shape[1]), dtype=np.float32)
            latents = np.concatenate([latents, pad], axis=1)
        elif latents.shape[1] > self.latent_dim:
            latents = latents[:, : self.latent_dim]
        return latents

    def _prepare_actions(self, actions: Optional[np.ndarray], t_len: int) -> np.ndarray:
        if actions is None:
            actions = np.zeros((t_len, self.action_dim), dtype=np.float32)
        else:
            actions = np.asarray(actions[:t_len], dtype=np.float32)
        if actions.shape[1] < self.action_dim:
            pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]

        horizon = self.action_horizon
        out = np.zeros((t_len, horizon, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + horizon)
            chunk = actions[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < horizon:
                pad_value = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad_value
        return out

    @torch.no_grad()
    def score_trajectory(self, traj: LatentTrajectory) -> TrajectoryScoreBundle:
        t_len = min(int(traj.latents.shape[0]), int(traj.actions.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")
        horizon = int(self.action_horizon)
        valid_len = t_len - horizon
        if valid_len <= 0:
            raise ValueError(f"Trajectory length {t_len} is too short for action_horizon={horizon}")

        latents = self._prepare_latents(traj.latents, t_len=t_len)
        action_chunks = self._prepare_actions(traj.actions, t_len=t_len)
        latents_t = torch.from_numpy(latents)
        action_t = torch.from_numpy(action_chunks)

        current_t = latents_t[:valid_len]
        action_chunk_t = action_t[:valid_len]
        target_t = latents_t[horizon : horizon + valid_len]
        tau = self.model.build_tau(
            current_latent=current_t,
            action_sequence=action_chunk_t,
            target_latent=target_t,
        )

        step_scores: list[torch.Tensor] = []
        state_scores: list[torch.Tensor] = []
        action_scores: list[torch.Tensor] = []
        next_state_scores: list[torch.Tensor] = []
        state_contrib: list[torch.Tensor] = []
        action_contrib: list[torch.Tensor] = []
        next_state_contrib: list[torch.Tensor] = []

        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            tau_b = tau[start:end].to(self.device)
            out = self.model.denoise_tau(tau=tau_b, add_noise=False)
            recon = self.model.reconstruction_components(tau=out["tau"], tau_hat=out["tau_hat"])
            step_scores.append(recon["tau_sse_per_sample"].detach().cpu())
            state_scores.append(recon["state_sse_per_sample"].detach().cpu())
            action_scores.append(recon["action_sse_per_sample"].detach().cpu())
            next_state_scores.append(recon["next_state_sse_per_sample"].detach().cpu())
            state_contrib.append(recon["state_sse_per_sample"].detach().cpu())
            action_contrib.append(recon["action_sse_per_sample"].detach().cpu())
            next_state_contrib.append(recon["next_state_sse_per_sample"].detach().cpu())

        return TrajectoryScoreBundle(
            step_scores=torch.cat(step_scores, dim=0).numpy().astype(np.float32),
            state_error_scores=torch.cat(state_scores, dim=0).numpy().astype(np.float32),
            action_error_scores=torch.cat(action_scores, dim=0).numpy().astype(np.float32),
            next_state_error_scores=torch.cat(next_state_scores, dim=0).numpy().astype(np.float32),
            state_contrib_scores=torch.cat(state_contrib, dim=0).numpy().astype(np.float32),
            action_contrib_scores=torch.cat(action_contrib, dim=0).numpy().astype(np.float32),
            next_state_contrib_scores=torch.cat(next_state_contrib, dim=0).numpy().astype(np.float32),
        )


class DSMDiscriminator(OfflineTrajectoryDiscriminator[LatentTrajectory]):
    COMPONENT_ORDER: tuple[str, ...] = (
        "state_error",
        "action_error",
        "next_state_error",
    )

    def __init__(
        self,
        checkpoint_path: str,
        *,
        feature_device: str = "cuda",
        feature_batch_size: int = 256,
        action_horizon: int = -1,
        detector_device: str = "cuda",
        delta: float = 10.0,
        delta_step: float = 0.5,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.extractor = DSMTransitionScorer(
            checkpoint_path=self.checkpoint_path,
            device=str(feature_device),
            batch_size=int(feature_batch_size),
            action_horizon=int(action_horizon),
        )
        self.device = _resolve_device(detector_device)
        self.delta = float(delta)
        self.delta_step = float(delta_step)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        if self.lambda_mode not in {"mean", "max"}:
            raise ValueError("lambda_mode must be 'mean' or 'max'")
        if self.lambda_window_size == 0:
            raise ValueError("lambda_window_size must be -1 or >=1")

        self._calib_lambdas: Optional[np.ndarray] = None
        self._calib_lambdas_by_task: dict[str, np.ndarray] = {}
        self.threshold: Optional[float] = None
        self.thresholds_by_task: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "lpb_score_dsm"

    @staticmethod
    def _compute_threshold(values: np.ndarray, delta: float) -> float:
        if values.size == 0:
            raise ValueError("Cannot calibrate threshold from empty values")
        clipped_delta = float(np.clip(delta, 0.0, 100.0))
        q = 100.0 * (1.0 - clipped_delta / 100.0)
        return float(np.percentile(values.astype(np.float64), q=q))

    @staticmethod
    def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
        n = values.shape[0]
        out = np.empty(n, dtype=np.float32)
        dq: deque[int] = deque()
        for i in range(n):
            while dq and dq[0] <= i - window:
                dq.popleft()
            while dq and values[dq[-1]] <= values[i]:
                dq.pop()
            dq.append(i)
            out[i] = float(values[dq[0]])
        return out

    def _aggregate_lambda(self, step_scores: np.ndarray) -> np.ndarray:
        vals = np.asarray(step_scores, dtype=np.float32).reshape(-1)
        n = vals.shape[0]
        if n == 0:
            return vals

        window = int(self.lambda_window_size)
        full_prefix = window <= 0

        if self.lambda_mode == "mean":
            if full_prefix:
                csum = np.cumsum(vals, dtype=np.float64)
                denom = np.arange(1, n + 1, dtype=np.float64)
                return (csum / denom).astype(np.float32)

            csum = np.cumsum(vals, dtype=np.float64)
            idx = np.arange(n, dtype=np.int64)
            start = np.maximum(0, idx - window + 1)
            start_minus = start - 1
            left = np.where(start_minus >= 0, csum[start_minus], 0.0)
            win_sum = csum - left
            denom = (idx - start + 1).astype(np.float64)
            return (win_sum / denom).astype(np.float32)

        if full_prefix:
            return np.maximum.accumulate(vals)
        return self._rolling_max(vals, window=window)

    def _compute_lambda_support_indices(
        self,
        step_scores: np.ndarray,
    ) -> np.ndarray:
        vals = np.asarray(step_scores, dtype=np.float32).reshape(-1)
        n = int(vals.shape[0])
        support = np.zeros((n,), dtype=np.int64)
        if n == 0:
            return support

        window = int(self.lambda_window_size)
        full_prefix = window <= 0

        if full_prefix:
            best_idx = 0
            best_val = float(vals[0])
            for idx in range(n):
                cur = float(vals[idx])
                if cur >= best_val:
                    best_val = cur
                    best_idx = idx
                support[idx] = int(best_idx)
            return support

        dq: deque[int] = deque()
        for idx in range(n):
            while dq and dq[0] <= idx - window:
                dq.popleft()
            while dq and float(vals[dq[-1]]) <= float(vals[idx]):
                dq.pop()
            dq.append(idx)
            support[idx] = int(dq[0])
        return support

    @staticmethod
    def _compute_component_shares(
        contributions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if not contributions:
            return {}
        keys = list(contributions.keys())
        total = np.zeros_like(np.asarray(contributions[keys[0]], dtype=np.float32), dtype=np.float32)
        for values in contributions.values():
            total = total + np.asarray(values, dtype=np.float32)
        safe_total = np.where(np.abs(total) > 1e-8, total, 1.0).astype(np.float32)
        return {
            key: (np.asarray(values, dtype=np.float32) / safe_total).astype(np.float32, copy=False)
            for key, values in contributions.items()
        }

    def _aggregate_component_contributions(
        self,
        contributions: dict[str, np.ndarray],
        step_scores: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self.lambda_mode == "mean":
            return {
                key: self._aggregate_lambda(values)
                for key, values in contributions.items()
            }
        support = self._compute_lambda_support_indices(step_scores)
        return {
            key: np.asarray(values, dtype=np.float32)[support].astype(np.float32, copy=False)
            for key, values in contributions.items()
        }

    def _dominant_component_terms(
        self,
        contributions: dict[str, np.ndarray],
    ) -> list[str]:
        if not contributions:
            return []
        keys = [key for key in self.COMPONENT_ORDER if key in contributions]
        if not keys:
            keys = list(contributions.keys())
        stacked = np.stack([np.asarray(contributions[key], dtype=np.float32) for key in keys], axis=0)
        dominant_idx = np.argmax(stacked, axis=0)
        return [str(keys[int(idx)]) for idx in dominant_idx.tolist()]

    def fit(
        self,
        normal_bank_trajectories: Sequence[LatentTrajectory],
        calibration_trajectories: Optional[Sequence[LatentTrajectory]] = None,
    ) -> DetectorCalibrationSummary:
        calibration_set = (
            list(calibration_trajectories)
            if calibration_trajectories is not None
            else list(normal_bank_trajectories)
        )
        if len(calibration_set) == 0:
            raise ValueError("Calibration requires at least one trajectory.")

        calib_lambdas: list[np.ndarray] = []
        calib_lambdas_by_task: dict[str, list[np.ndarray]] = {}
        for traj in calibration_set:
            lambdas = self._aggregate_lambda(self.extractor.score_trajectory(traj).step_scores)
            calib_lambdas.append(lambdas)
            calib_lambdas_by_task.setdefault(str(traj.task_name), []).append(lambdas)

        self._calib_lambdas = np.concatenate(calib_lambdas, axis=0).astype(np.float32)
        self._calib_lambdas_by_task = {
            task_name: np.concatenate(task_values, axis=0).astype(np.float32)
            for task_name, task_values in calib_lambdas_by_task.items()
            if len(task_values) > 0
        }
        self.threshold = self._compute_threshold(self._calib_lambdas, self.delta)
        self.thresholds_by_task = {
            task_name: self._compute_threshold(task_values, self.delta)
            for task_name, task_values in self._calib_lambdas_by_task.items()
        }
        return DetectorCalibrationSummary(
            detector_name=self.name,
            threshold=float(self.threshold),
            metadata={
                "dsm_ckpt": self.checkpoint_path,
                "noise_sigma": float(self.extractor.model.noise_sigma),
                "latent_dim": int(self.extractor.latent_dim),
                "action_dim": int(self.extractor.action_dim),
                "horizon": int(self.extractor.action_horizon),
                "tau_dim": int(self.extractor.model.tau_dim),
                "delta_init": float(self.delta),
                "delta_final": float(self.delta),
                "lambda_mode": str(self.lambda_mode),
                "lambda_window_size": int(self.lambda_window_size),
                "num_calibration_trajectories": int(len(calibration_set)),
                "num_calibration_trajectories_by_task": {
                    task_name: int(len(task_values))
                    for task_name, task_values in calib_lambdas_by_task.items()
                },
                "thresholds_by_task": {
                    task_name: float(threshold)
                    for task_name, threshold in self.thresholds_by_task.items()
                },
            },
        )

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
        if self.threshold is None or self._calib_lambdas is None:
            raise RuntimeError("Call fit(...) before detect_trajectory(...)")

        bundle = self.extractor.score_trajectory(trajectory)
        step_scores = np.asarray(bundle.step_scores, dtype=np.float32)
        lamb = self._aggregate_lambda(step_scores)

        component_scores = {
            "state_error": np.asarray(bundle.state_error_scores, dtype=np.float32),
            "action_error": np.asarray(bundle.action_error_scores, dtype=np.float32),
            "next_state_error": np.asarray(bundle.next_state_error_scores, dtype=np.float32),
        }
        component_contrib = {
            "state_error": np.asarray(bundle.state_contrib_scores, dtype=np.float32),
            "action_error": np.asarray(bundle.action_contrib_scores, dtype=np.float32),
            "next_state_error": np.asarray(bundle.next_state_contrib_scores, dtype=np.float32),
        }
        step_contribution_shares = self._compute_component_shares(component_contrib)
        aggregate_contributions = self._aggregate_component_contributions(
            component_contrib,
            step_scores=step_scores,
        )
        aggregate_contribution_shares = self._compute_component_shares(aggregate_contributions)
        dominant_step_terms = self._dominant_component_terms(component_contrib)
        dominant_aggregate_terms = self._dominant_component_terms(aggregate_contributions)

        preds = np.zeros_like(lamb, dtype=np.int64)
        ths = np.zeros_like(lamb, dtype=np.float32)

        task_name = str(trajectory.task_name)
        task_calib_lambdas = self._calib_lambdas_by_task.get(task_name, None)
        task_threshold = self.thresholds_by_task.get(task_name, None)
        cur_delta = float(self.delta)
        cur_threshold = float(task_threshold if task_threshold is not None else self.threshold)
        dmin = float(np.clip(delta_min, 0.0, 100.0))
        dmax = float(np.clip(delta_max, 0.0, 100.0))
        if dmin > dmax:
            dmin, dmax = dmax, dmin
        warmup = max(0, int(warmup_steps))
        update_every = max(1, int(update_interval))

        labels_np = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
        if adaptive_threshold and labels_np is not None and labels_np.shape[0] != lamb.shape[0]:
            raise ValueError(
                "labels length must match the number of aggregate scores when adaptive_threshold=True: "
                f"got labels={labels_np.shape[0]}, scores={lamb.shape[0]}"
            )
        for t in range(lamb.shape[0]):
            pred = int(lamb[t] >= cur_threshold)
            preds[t] = pred
            ths[t] = float(cur_threshold)

            if adaptive_threshold and labels_np is not None:
                should_update = (t + 1) > warmup and ((t + 1 - warmup) % update_every == 0)
                if should_update:
                    label = int(labels_np[t])
                    if label == 1 and pred == 0:
                        cur_delta += self.delta_step
                    elif label == 0 and pred == 1:
                        cur_delta -= self.delta_step
                    cur_delta = float(np.clip(cur_delta, dmin, dmax))
                    threshold_source_values = task_calib_lambdas if task_calib_lambdas is not None else self._calib_lambdas
                    cur_threshold = self._compute_threshold(threshold_source_values, cur_delta)

        first_crossing_idx = np.where(preds == 1)[0]
        first_crossing_index = int(first_crossing_idx[0]) if first_crossing_idx.size > 0 else None
        first_crossing_dominant_term = (
            str(dominant_aggregate_terms[first_crossing_index])
            if first_crossing_index is not None and first_crossing_index < len(dominant_aggregate_terms)
            else None
        )
        first_crossing_term_shares = (
            {
                key: float(np.asarray(values, dtype=np.float32)[first_crossing_index])
                for key, values in aggregate_contribution_shares.items()
            }
            if first_crossing_index is not None
            else {}
        )

        return TrajectoryDetectionResult(
            detector_name=self.name,
            step_scores=step_scores,
            aggregate_scores=np.asarray(lamb, dtype=np.float32),
            thresholds=np.asarray(ths, dtype=np.float32),
            predictions=np.asarray(preds, dtype=np.int64),
            aux_scores=None,
            labels=labels_np,
            metadata={
                "task_name": task_name,
                "threshold_source": "task" if task_threshold is not None else "global",
                "threshold_init": float(task_threshold if task_threshold is not None else self.threshold),
                "delta_final": float(cur_delta),
                "threshold_final": float(cur_threshold),
                "state_error_scores": component_scores["state_error"],
                "action_error_scores": component_scores["action_error"],
                "next_state_error_scores": component_scores["next_state_error"],
                "weighted_step_contributions": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in component_contrib.items()
                },
                "aggregate_contributions": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in aggregate_contributions.items()
                },
                "step_contribution_shares": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in step_contribution_shares.items()
                },
                "aggregate_contribution_shares": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in aggregate_contribution_shares.items()
                },
                "dominant_step_terms": list(dominant_step_terms),
                "dominant_aggregate_terms": list(dominant_aggregate_terms),
                "first_crossing_index": first_crossing_index,
                "first_crossing_dominant_term": first_crossing_dominant_term,
                "first_crossing_term_shares": dict(first_crossing_term_shares),
                "state_error_mean": float(np.mean(component_scores["state_error"])),
                "action_error_mean": float(np.mean(component_scores["action_error"])),
                "next_state_error_mean": float(np.mean(component_scores["next_state_error"])),
            },
        )

    def close(self) -> None:
        return None
