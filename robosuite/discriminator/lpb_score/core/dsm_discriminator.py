"""Offline SσDC trajectory discriminator: scoring, calibration, and detection API.

Implements the Single-σ Diffusion Classifier (SσDC) step score

    u(x) = Σ_b [ α_b · z(E_b^+) + β_b · (z(E_b^+) − z(E_b^-)) ],   b ∈ {state, dynamics}

where ``E_b^(c)(x) = ||g_θ(x, c) - x||^2`` is the per-factor single-sample
reconstruction energy under class ``c`` and ``z(·)`` is per-task, per-branch
z-score standardization estimated from the success bank.
"""

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
    """Dot-path lookup for nested checkpoint or Hydra config blobs."""
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
    """Per-step per-factor conditional energies and signed margins for one trajectory."""

    state_positive_scores: np.ndarray
    state_negative_scores: np.ndarray
    state_margin_scores: np.ndarray
    dynamics_positive_scores: np.ndarray
    dynamics_negative_scores: np.ndarray
    dynamics_margin_scores: np.ndarray


class DSMTransitionScorer:
    """Loads a trained ``DSMModel`` checkpoint and runs ``compute_conditional_energies`` along sliding windows."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        action_horizon: int = -1,
    ) -> None:
        """Restore weights from ``checkpoint_path``; horizon must match the checkpoint when overridden."""
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
        """Rebuild ``DSMModel`` from checkpoint metadata and load weights.

        ``DSMModel`` inference requires:
        - architecture hyperparameters (latent/action dims, horizon, num_tasks)
        - normalization stats (latent/action mean/var) if stored in the checkpoint

        Note: ``num_tasks`` defines the discrete task embedding vocabulary size. It must match the
        dataset's ``task_index`` range used during training and benchmarking.
        """
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing key `model`: {checkpoint_path}")

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim"))
        action_dim = int(payload.get("action_dim"))
        num_tasks = int(payload.get("num_tasks", 0))
        if num_tasks <= 0:
            task_to_index = payload.get("task_to_index", None)
            if isinstance(task_to_index, dict) and task_to_index:
                num_tasks = int(len(task_to_index))
        if num_tasks <= 0:
            cfg_tasks = _cfg_get(cfg, "data.tasks", None)
            if cfg_tasks is not None:
                try:
                    num_tasks = int(len(cfg_tasks))
                except Exception:
                    num_tasks = 0
        if num_tasks <= 0:
            raise ValueError(
                "Checkpoint is missing task vocabulary metadata. "
                f"Expected `num_tasks` or `task_to_index` in {checkpoint_path}."
            )
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
            num_tasks=num_tasks,
            cfg_model=_cfg_get(cfg, "model", {}),
            transition_horizon=action_horizon,
        )
        normalization_stats = payload.get("normalization_stats", None)
        if normalization_stats is not None:
            model.set_normalization_stats(
                latent_mean=normalization_stats["latent_mean"],
                latent_var=normalization_stats["latent_var"],
                action_mean=normalization_stats["action_mean"],
                action_var=normalization_stats["action_var"],
            )
        load_result = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {"latent_mean", "latent_var", "action_mean", "action_var"}
        missing = set(load_result.missing_keys)
        unexpected = set(load_result.unexpected_keys)
        if unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Expected a SσDC (state + dynamics) lpb_score checkpoint, got incompatible weights from "
                f"{checkpoint_path}. "
                "Likely a pre-refactor uni-dsm three-head checkpoint (with action head) — retrain under the new recipe. "
                f"Unexpected keys: {sorted(unexpected)}"
            )
        if missing and not missing.issubset(allowed_missing):
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Expected a SσDC (state + dynamics) lpb_score checkpoint, got incompatible weights from "
                f"{checkpoint_path}. "
                f"Missing keys: {sorted(missing)}"
            )
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
                # Repeat the last action when the rollout tail is shorter than H.
                # This keeps the conditioning tensor shape (H, A) fixed for every t.
                pad_value = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad_value
        return out

    @torch.no_grad()
    def score_trajectory(self, traj: LatentTrajectory) -> TrajectoryScoreBundle:
        """Compute per-step per-factor energies for a trajectory.

        For a trajectory of length T and horizon H, SσDC scores the valid windows:

        - current latent: ``z_t`` for t = 0 .. (T - H - 1)
        - action chunk:  ``a_{t:t+H}`` as a fixed-size (H, A) tensor
        - target latent: ``z_{t+H}``

        The underlying ``DSMModel.compute_conditional_energies`` returns energies under the
        success-conditioned (traj_type=0) and failure-conditioned (traj_type=1) denoisers.
        """
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

        # Align the transition targets so each index i corresponds to (z_i, a_{i:i+H}, z_{i+H}).
        current_t = latents_t[:valid_len]
        action_chunk_t = action_t[:valid_len]
        target_t = latents_t[horizon : horizon + valid_len]

        state_positive_scores: list[torch.Tensor] = []
        state_negative_scores: list[torch.Tensor] = []
        state_margin_scores: list[torch.Tensor] = []
        dynamics_positive_scores: list[torch.Tensor] = []
        dynamics_negative_scores: list[torch.Tensor] = []
        dynamics_margin_scores: list[torch.Tensor] = []

        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            current_b = current_t[start:end].to(self.device)
            action_b = action_chunk_t[start:end].to(self.device)
            target_b = target_t[start:end].to(self.device)
            # Inference uses add_noise=False inside compute_conditional_energies, producing the
            # single-sample per-class reconstruction energies E_b^(c)(x) = ||g(x,c) - x||^2.
            energies = self.model.compute_conditional_energies(
                current_latent=current_b,
                action_sequence=action_b,
                target_latent=target_b,
                task_index=int(traj.task_index),
            )
            state_positive_scores.append(energies["state_positive_energy_per_sample"].detach().cpu())
            state_negative_scores.append(energies["state_negative_energy_per_sample"].detach().cpu())
            state_margin_scores.append(energies["state_margin_per_sample"].detach().cpu())
            dynamics_positive_scores.append(energies["dynamics_positive_energy_per_sample"].detach().cpu())
            dynamics_negative_scores.append(energies["dynamics_negative_energy_per_sample"].detach().cpu())
            dynamics_margin_scores.append(energies["dynamics_margin_per_sample"].detach().cpu())

        return TrajectoryScoreBundle(
            state_positive_scores=torch.cat(state_positive_scores, dim=0).numpy().astype(np.float32),
            state_negative_scores=torch.cat(state_negative_scores, dim=0).numpy().astype(np.float32),
            state_margin_scores=torch.cat(state_margin_scores, dim=0).numpy().astype(np.float32),
            dynamics_positive_scores=torch.cat(dynamics_positive_scores, dim=0).numpy().astype(np.float32),
            dynamics_negative_scores=torch.cat(dynamics_negative_scores, dim=0).numpy().astype(np.float32),
            dynamics_margin_scores=torch.cat(dynamics_margin_scores, dim=0).numpy().astype(np.float32),
        )


class DSMDiscriminator(OfflineTrajectoryDiscriminator[LatentTrajectory]):
    """SσDC detector: single score formula over state + dynamics conditional energies."""

    COMPONENT_ORDER: tuple[str, ...] = ("state", "dynamics")

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
        alpha_state: float = 0.0,
        alpha_dynamics: float = 0.0,
        beta_state: float = 1.0,
        beta_dynamics: float = 1.0,
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

        self.alpha = {
            "state": float(alpha_state),
            "dynamics": float(alpha_dynamics),
        }
        self.beta = {
            "state": float(beta_state),
            "dynamics": float(beta_dynamics),
        }

        self._calib_lambdas: Optional[np.ndarray] = None
        self._calib_lambdas_by_task: dict[str, np.ndarray] = {}
        self.threshold: Optional[float] = None
        self.thresholds_by_task: dict[str, float] = {}
        self._norm_stats_global: dict[str, float] = {}
        self._norm_stats_by_task: dict[str, dict[str, float]] = {}

    @property
    def name(self) -> str:
        return "lpb_score_dsm"

    def get_calibration_lambdas_by_task(self) -> dict[str, np.ndarray]:
        return {
            str(task_name): np.asarray(values, dtype=np.float32).copy()
            for task_name, values in self._calib_lambdas_by_task.items()
        }

    def get_thresholds_by_task(self) -> dict[str, float]:
        return {
            str(task_name): float(threshold)
            for task_name, threshold in self.thresholds_by_task.items()
        }

    def collect_lambdas_by_task(
        self,
        trajectories: Sequence[LatentTrajectory],
    ) -> dict[str, np.ndarray]:
        lambdas_by_task: dict[str, list[np.ndarray]] = {}
        for traj in trajectories:
            bundle = self.extractor.score_trajectory(traj)
            step_scores = self._compute_step_scores(bundle, task_name=str(traj.task_name))
            lambdas = self._aggregate_lambda(step_scores)
            lambdas_by_task.setdefault(str(traj.task_name), []).append(lambdas.astype(np.float32, copy=False))
        return {
            str(task_name): np.concatenate(values, axis=0).astype(np.float32)
            for task_name, values in lambdas_by_task.items()
            if values
        }

    @staticmethod
    def _standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
        denom = max(float(std), 1e-6)
        return ((np.asarray(values, dtype=np.float32) - float(mean)) / denom).astype(np.float32)

    @staticmethod
    def _bundle_raw_terms(bundle: TrajectoryScoreBundle) -> dict[str, np.ndarray]:
        return {
            "state_positive": np.asarray(bundle.state_positive_scores, dtype=np.float32),
            "state_negative": np.asarray(bundle.state_negative_scores, dtype=np.float32),
            "dynamics_positive": np.asarray(bundle.dynamics_positive_scores, dtype=np.float32),
            "dynamics_negative": np.asarray(bundle.dynamics_negative_scores, dtype=np.float32),
        }

    def _compute_norm_stats(
        self,
        scored_bundles: Sequence[tuple[str, TrajectoryScoreBundle]],
    ) -> None:
        """Estimate z-score normalization stats from the success bank.

        Normalization is intentionally derived only from trajectories treated as "normal"
        (success bank). These stats define ``z(·)`` in the SσDC score, per task when available,
        otherwise falling back to global.
        """
        global_terms: dict[str, list[np.ndarray]] = {}
        task_terms: dict[str, dict[str, list[np.ndarray]]] = {}

        for task_name, bundle in scored_bundles:
            raw_terms = self._bundle_raw_terms(bundle)
            bucket = task_terms.setdefault(str(task_name), {})
            for key, values in raw_terms.items():
                global_terms.setdefault(key, []).append(values)
                bucket.setdefault(key, []).append(values)

        def _reduce(term_map: dict[str, list[np.ndarray]]) -> dict[str, float]:
            stats: dict[str, float] = {}
            for key, values in term_map.items():
                arr = np.concatenate(values, axis=0).astype(np.float32) if values else np.zeros((0,), dtype=np.float32)
                if arr.size == 0:
                    stats[f"{key}_mean"] = 0.0
                    stats[f"{key}_std"] = 1.0
                else:
                    stats[f"{key}_mean"] = float(np.mean(arr))
                    stats[f"{key}_std"] = float(max(float(np.std(arr)), 1e-6))
            return stats

        self._norm_stats_global = _reduce(global_terms)
        self._norm_stats_by_task = {
            str(task_name): _reduce(term_map)
            for task_name, term_map in task_terms.items()
        }

    def _resolve_norm_stats(self, task_name: str) -> dict[str, float]:
        task_stats = self._norm_stats_by_task.get(str(task_name), None)
        if task_stats is not None:
            return task_stats
        return self._norm_stats_global

    def _compute_step_scores(
        self,
        bundle: TrajectoryScoreBundle,
        *,
        task_name: str,
    ) -> np.ndarray:
        """SσDC per-step score.

        Given per-step per-branch energies under the two class-conditional denoisers:
        - ``E_b^+``: traj_type=0 (success-conditioned)
        - ``E_b^-``: traj_type=1 (failure-conditioned)

        compute:

            u(x) = Σ_b [ α_b · z(E_b^+) + β_b · ( z(E_b^+) − z(E_b^-) ) ].

        ``z(·)`` is a per-task (or global fallback) z-score using success-bank mean/std.
        """
        stats = self._resolve_norm_stats(task_name)
        branches: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "state": (bundle.state_positive_scores, bundle.state_negative_scores),
            "dynamics": (bundle.dynamics_positive_scores, bundle.dynamics_negative_scores),
        }
        score: Optional[np.ndarray] = None
        for branch, (pos, neg) in branches.items():
            z_pos = self._standardize(
                pos,
                stats.get(f"{branch}_positive_mean", 0.0),
                stats.get(f"{branch}_positive_std", 1.0),
            )
            z_neg = self._standardize(
                neg,
                stats.get(f"{branch}_negative_mean", 0.0),
                stats.get(f"{branch}_negative_std", 1.0),
            )
            contribution = self.alpha[branch] * z_pos + self.beta[branch] * (z_pos - z_neg)
            if score is None:
                score = contribution.astype(np.float32, copy=True)
            else:
                score = score + contribution
        assert score is not None
        return score.astype(np.float32, copy=False)

    def _step_components(
        self,
        bundle: TrajectoryScoreBundle,
        *,
        task_name: str,
    ) -> dict[str, np.ndarray]:
        """Per-branch additive contribution to the step score (for attribution)."""
        stats = self._resolve_norm_stats(task_name)
        components: dict[str, np.ndarray] = {}
        for branch, (pos, neg) in (
            ("state", (bundle.state_positive_scores, bundle.state_negative_scores)),
            ("dynamics", (bundle.dynamics_positive_scores, bundle.dynamics_negative_scores)),
        ):
            z_pos = self._standardize(
                pos,
                stats.get(f"{branch}_positive_mean", 0.0),
                stats.get(f"{branch}_positive_std", 1.0),
            )
            z_neg = self._standardize(
                neg,
                stats.get(f"{branch}_negative_mean", 0.0),
                stats.get(f"{branch}_negative_std", 1.0),
            )
            components[branch] = (self.alpha[branch] * z_pos + self.beta[branch] * (z_pos - z_neg)).astype(np.float32)
        return components

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

        # Aggregate step scores u_t into a prefix score λ_t.
        #
        # - mean: rolling mean (or full-prefix mean when window <= 0)
        # - max: rolling max  (or full-prefix max  when window <= 0)
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
        # Two-stage calibration:
        # 1) success bank -> per-branch mean/std used for z-score normalization z(·)
        # 2) calibration set -> λ values whose percentile defines the detection threshold
        normal_bank_set = list(normal_bank_trajectories)
        calibration_set = (
            list(calibration_trajectories)
            if calibration_trajectories is not None
            else list(normal_bank_set)
        )
        if len(calibration_set) == 0:
            raise ValueError("Calibration requires at least one trajectory.")
        if len(normal_bank_set) == 0:
            raise ValueError("SσDC normalization requires at least one success-bank trajectory.")

        bank_scored_bundles: list[tuple[str, TrajectoryScoreBundle]] = []
        for traj in normal_bank_set:
            bank_scored_bundles.append((str(traj.task_name), self.extractor.score_trajectory(traj)))

        # Normalization stats always come from the success bank.
        self._compute_norm_stats(bank_scored_bundles)

        scored_bundles: list[tuple[str, TrajectoryScoreBundle]] = []
        for traj in calibration_set:
            scored_bundles.append((str(traj.task_name), self.extractor.score_trajectory(traj)))

        calib_lambdas: list[np.ndarray] = []
        calib_lambdas_by_task: dict[str, list[np.ndarray]] = {}
        for task_name, bundle in scored_bundles:
            step_scores = self._compute_step_scores(bundle, task_name=task_name)
            lambdas = self._aggregate_lambda(step_scores)
            calib_lambdas.append(lambdas)
            calib_lambdas_by_task.setdefault(str(task_name), []).append(lambdas)

        self._calib_lambdas = (
            np.concatenate(calib_lambdas, axis=0).astype(np.float32)
            if calib_lambdas
            else np.zeros((0,), dtype=np.float32)
        )
        self._calib_lambdas_by_task = {
            str(task_name): np.concatenate(values, axis=0).astype(np.float32)
            for task_name, values in calib_lambdas_by_task.items()
            if values
        }
        # Threshold is the (1 - delta/100) quantile of calibration λ values.
        self.threshold = self._compute_threshold(self._calib_lambdas, self.delta)
        self.thresholds_by_task = {
            task_name: self._compute_threshold(values, self.delta)
            for task_name, values in self._calib_lambdas_by_task.items()
        }

        return DetectorCalibrationSummary(
            detector_name=self.name,
            threshold=float(self.threshold),
            metadata={
                "dsm_ckpt": self.checkpoint_path,
                "noise_scale": float(self.extractor.model.noise_scale),
                "noise_sigma": float(self.extractor.model.noise_scale),
                "std_clamp_min": float(self.extractor.model.std_clamp_min),
                "model_architecture": "sigma_diffusion_classifier",
                "score_formula": "alpha*z(E+) + beta*(z(E+) - z(E-))",
                "latent_dim": int(self.extractor.latent_dim),
                "action_dim": int(self.extractor.action_dim),
                "horizon": int(self.extractor.action_horizon),
                "tau_dim": int(self.extractor.model.tau_dim),
                "delta_init": float(self.delta),
                "delta_final": float(self.delta),
                "alpha": {key: float(value) for key, value in self.alpha.items()},
                "beta": {key: float(value) for key, value in self.beta.items()},
                "norm_source": "success_bank",
                "num_norm_bank_trajectories": int(len(normal_bank_set)),
                "num_norm_bank_trajectories_by_task": {
                    task_name: int(sum(1 for bundle_task_name, _ in bank_scored_bundles if bundle_task_name == task_name))
                    for task_name in sorted({task_name for task_name, _ in bank_scored_bundles})
                },
                "norm_stats_global": {
                    key: float(value)
                    for key, value in self._norm_stats_global.items()
                },
                "norm_stats_by_task": {
                    task_name: {
                        key: float(value)
                        for key, value in stats.items()
                    }
                    for task_name, stats in self._norm_stats_by_task.items()
                },
                "lambda_mode": str(self.lambda_mode),
                "lambda_window_size": int(self.lambda_window_size),
                "num_calibration_trajectories": int(len(calibration_set)),
                "num_calibration_trajectories_by_task": {
                    task_name: int(sum(1 for bundle_task_name, _ in scored_bundles if bundle_task_name == task_name))
                    for task_name in sorted({task_name for task_name, _ in scored_bundles})
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

        # 1) Compute per-step energies under the two class-conditional denoisers.
        bundle = self.extractor.score_trajectory(trajectory)
        task_name = str(trajectory.task_name)
        # 2) Convert energies -> z-scored step scores u_t -> aggregated prefix scores λ_t.
        step_scores = self._compute_step_scores(bundle, task_name=task_name)
        lamb = self._aggregate_lambda(step_scores)

        # Attribution helpers: decompose u_t and λ_t into per-branch additive contributions.
        component_contrib = self._step_components(bundle, task_name=task_name)
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
        # 3) Threshold λ_t into per-step predictions. Optionally adapt δ online using labels.
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
                "state_positive_scores": np.asarray(bundle.state_positive_scores, dtype=np.float32),
                "state_negative_scores": np.asarray(bundle.state_negative_scores, dtype=np.float32),
                "state_margin_scores": np.asarray(bundle.state_margin_scores, dtype=np.float32),
                "dynamics_positive_scores": np.asarray(bundle.dynamics_positive_scores, dtype=np.float32),
                "dynamics_negative_scores": np.asarray(bundle.dynamics_negative_scores, dtype=np.float32),
                "dynamics_margin_scores": np.asarray(bundle.dynamics_margin_scores, dtype=np.float32),
                "alpha": {key: float(value) for key, value in self.alpha.items()},
                "beta": {key: float(value) for key, value in self.beta.items()},
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
                "state_positive_mean": float(np.mean(bundle.state_positive_scores)),
                "state_negative_mean": float(np.mean(bundle.state_negative_scores)),
                "state_margin_mean": float(np.mean(bundle.state_margin_scores)),
                "dynamics_positive_mean": float(np.mean(bundle.dynamics_positive_scores)),
                "dynamics_negative_mean": float(np.mean(bundle.dynamics_negative_scores)),
                "dynamics_margin_mean": float(np.mean(bundle.dynamics_margin_scores)),
            },
        )

    def close(self) -> None:
        return None
