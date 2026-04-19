"""Offline DSM-based trajectory discriminator for joint encoder + chunk scoring."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FlowMultitaskEncoder
from robosuite.discriminator.utils.base import OfflineTrajectoryDiscriminator
from robosuite.discriminator.utils.types import DetectorCalibrationSummary, TrajectoryDetectionResult

from .dataset import PreparedTrajectory, resolve_window_size
from .model import MODEL_ARCHITECTURE, DSMModel, build_dsm_model


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
    chunk_positive_scores: np.ndarray
    chunk_negative_scores: np.ndarray
    chunk_margin_scores: np.ndarray
    t3_energy_terms: np.ndarray
    t3_margin_terms: np.ndarray


class DSMTransitionScorer:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        window_size: int = -1,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.model, self.window_size, self.latent_dim = self._load_model(
            checkpoint_path=self.checkpoint_path,
            override_window_size=window_size,
        )
        self.encoder = self.model.policy_encoder

    def _load_model(
        self,
        checkpoint_path: str,
        override_window_size: int,
    ) -> tuple[DSMModel, int, int]:
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing key `model`: {checkpoint_path}")
        checkpoint_architecture = str(payload.get("model_architecture", "") or "")
        if checkpoint_architecture and checkpoint_architecture != MODEL_ARCHITECTURE:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Expected `{MODEL_ARCHITECTURE}` but got `{checkpoint_architecture}` from {checkpoint_path}."
            )
        if "policy_checkpoint_payload" not in payload:
            raise RuntimeError(
                "Checkpoint is missing `policy_checkpoint_payload`. "
                "Old latent-only DSM checkpoints are not load-compatible with the joint encoder model."
            )

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim"))
        task_to_index = dict(payload.get("task_to_index", {}) or {})
        num_tasks = int(payload.get("num_tasks", len(task_to_index)))
        if num_tasks <= 0 or not task_to_index:
            raise ValueError(
                "Checkpoint is missing task vocabulary metadata. "
                f"Expected `num_tasks` and `task_to_index` in {checkpoint_path}."
            )

        task_names = payload.get("task_names", None)
        if not task_names:
            task_names = [name for name, _ in sorted(task_to_index.items(), key=lambda item: int(item[1]))]
        task_names = [str(name) for name in task_names]

        if "window_size" in payload:
            window_size_ckpt = int(payload["window_size"])
        else:
            window_size_ckpt = int(resolve_window_size(cfg, default=0))
        if window_size_ckpt <= 0:
            raise ValueError(f"Checkpoint has invalid window_size={window_size_ckpt}: {checkpoint_path}")
        if int(override_window_size) > 0 and int(override_window_size) != window_size_ckpt:
            raise ValueError(
                "Inference window_size must match the training window_size stored in the checkpoint: "
                f"got override={int(override_window_size)}, checkpoint={window_size_ckpt}"
            )
        window_size = window_size_ckpt if int(override_window_size) <= 0 else int(override_window_size)

        encoder = FlowMultitaskEncoder(
            checkpoint_payload=dict(payload["policy_checkpoint_payload"]),
            device=str(self.device),
            image_size=int(payload.get("image_size", _cfg_get(cfg, "data.image_size", 128))),
            batch_size=self.batch_size,
            trainable=False,
        )
        model = build_dsm_model(
            latent_dim=latent_dim,
            num_tasks=num_tasks,
            cfg_model=_cfg_get(cfg, "model", {}),
            window_size=window_size,
            policy_encoder=encoder,
            task_names=task_names,
        )
        normalization_stats = payload.get("normalization_stats", None)
        if normalization_stats is not None:
            model.set_normalization_stats(
                latent_mean=normalization_stats["latent_mean"],
                latent_var=normalization_stats["latent_var"],
            )
        load_result = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {"latent_mean", "latent_var"}
        missing = set(load_result.missing_keys)
        unexpected = set(load_result.unexpected_keys)
        if unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Unexpected keys from {checkpoint_path}: {sorted(unexpected)}"
            )
        if missing and not missing.issubset(allowed_missing):
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Missing keys from {checkpoint_path}: {sorted(missing)}"
            )
        model.to(self.device)
        model.eval()
        return model, window_size, latent_dim

    def _build_latent_windows(self, latents: np.ndarray) -> np.ndarray:
        t_len = int(latents.shape[0])
        windows = np.zeros((t_len, self.window_size, self.latent_dim), dtype=np.float32)
        for t in range(t_len):
            # Build causal windows: each step only sees current and past latents.
            start = max(0, t - self.window_size + 1)
            chunk = latents[start : t + 1]
            windows[t, -chunk.shape[0] :] = chunk
            if chunk.shape[0] < self.window_size:
                # Left-pad the prefix with the first latent to keep fixed-length windows.
                windows[t, : self.window_size - chunk.shape[0]] = latents[0]
        return windows

    @torch.no_grad()
    def score_trajectory(self, traj: PreparedTrajectory) -> TrajectoryScoreBundle:
        t_len = int(traj.images.shape[0])
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")

        latents = self.model.encode_trajectory(
            images=traj.images,
            proprio=traj.proprio,
            task_index=int(traj.task_index),
            batch_size=self.batch_size,
        ).detach().cpu().numpy().astype(np.float32)
        latent_windows = self._build_latent_windows(latents)
        latent_window_t = torch.from_numpy(latent_windows)

        chunk_positive_scores: list[torch.Tensor] = []
        chunk_negative_scores: list[torch.Tensor] = []
        chunk_margin_scores: list[torch.Tensor] = []

        for start in range(0, t_len, self.batch_size):
            end = min(start + self.batch_size, t_len)
            latent_window_b = latent_window_t[start:end].to(self.device)
            fisher = self.model.compute_fisher_score_from_latent_window(
                latent_window=latent_window_b,
                task_index=int(traj.task_index),
            )
            chunk_positive_scores.append(fisher["chunk_positive_energy_per_sample"].detach().cpu())
            chunk_negative_scores.append(fisher["chunk_negative_energy_per_sample"].detach().cpu())
            chunk_margin_scores.append(fisher["chunk_margin_per_sample"].detach().cpu())

        chunk_positive = torch.cat(chunk_positive_scores, dim=0).numpy().astype(np.float32)
        chunk_negative = torch.cat(chunk_negative_scores, dim=0).numpy().astype(np.float32)
        chunk_margin = torch.cat(chunk_margin_scores, dim=0).numpy().astype(np.float32)
        return TrajectoryScoreBundle(
            chunk_positive_scores=chunk_positive,
            chunk_negative_scores=chunk_negative,
            chunk_margin_scores=chunk_margin,
            t3_energy_terms=chunk_positive.copy(),
            t3_margin_terms=chunk_margin.copy(),
        )

    def close(self) -> None:
        self.encoder.close()


class DSMDiscriminator(OfflineTrajectoryDiscriminator[PreparedTrajectory]):
    SCORE_T3 = "t3_weighted_combo"
    COMPONENT_ORDER: tuple[str, ...] = ("chunk_energy", "chunk_margin")

    def __init__(
        self,
        checkpoint_path: str,
        *,
        feature_device: str = "cuda",
        feature_batch_size: int = 256,
        window_size: int = -1,
        detector_device: str = "cuda",
        delta: float = 10.0,
        delta_step: float = 0.5,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.extractor = DSMTransitionScorer(
            checkpoint_path=self.checkpoint_path,
            device=str(feature_device),
            batch_size=int(feature_batch_size),
            window_size=int(window_size),
        )
        self.device = _resolve_device(detector_device)
        self.delta = float(delta)
        self.delta_step = float(delta_step)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.score_mode = self.SCORE_T3
        self.alpha = float(alpha)
        self.beta = float(beta)
        if self.lambda_mode not in {"mean", "max"}:
            raise ValueError("lambda_mode must be 'mean' or 'max'")
        if self.lambda_window_size == 0:
            raise ValueError("lambda_window_size must be -1 or >=1")
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

        self.t3_alpha = {"chunk_energy": self.alpha}
        self.t3_beta = {"chunk_margin": self.beta}

        self._calib_lambdas: Optional[np.ndarray] = None
        self._calib_lambdas_by_task: dict[str, np.ndarray] = {}
        self.threshold: Optional[float] = None
        self.thresholds_by_task: dict[str, float] = {}
        self._t3_norm_stats_global: dict[str, float] = {}
        self._t3_norm_stats_by_task: dict[str, dict[str, float]] = {}

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

    def get_thresholds_by_score(self) -> dict[str, float]:
        if self.threshold is None:
            return {}
        return {self.SCORE_T3: float(self.threshold)}

    def collect_lambdas_by_task(
        self,
        trajectories: Sequence[PreparedTrajectory],
    ) -> dict[str, np.ndarray]:
        lambdas_by_task: dict[str, list[np.ndarray]] = {}
        for traj in trajectories:
            bundle = self.extractor.score_trajectory(traj)
            step_scores = self._bundle_score_family(bundle, task_name=str(traj.task_name))["step_scores"]
            lambdas = self._aggregate_lambda(step_scores)
            lambdas_by_task.setdefault(str(traj.task_name), []).append(lambdas.astype(np.float32, copy=False))
        return {
            str(task_name): np.concatenate(values, axis=0).astype(np.float32)
            for task_name, values in lambdas_by_task.items()
            if values
        }

    @staticmethod
    def _safe_standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
        denom = max(float(std), 1e-6)
        return ((np.asarray(values, dtype=np.float32) - float(mean)) / denom).astype(np.float32)

    @staticmethod
    def _bundle_raw_terms(bundle: TrajectoryScoreBundle) -> dict[str, np.ndarray]:
        return {
            "chunk_positive": np.asarray(bundle.chunk_positive_scores, dtype=np.float32),
            "chunk_margin": np.asarray(bundle.chunk_margin_scores, dtype=np.float32),
        }

    def _compute_t3_norm_stats(
        self,
        scored_bundles: Sequence[tuple[str, TrajectoryScoreBundle]],
    ) -> None:
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

        self._t3_norm_stats_global = _reduce(global_terms)
        self._t3_norm_stats_by_task = {
            str(task_name): _reduce(term_map)
            for task_name, term_map in task_terms.items()
        }

    def _resolve_t3_norm_stats(self, task_name: str) -> dict[str, float]:
        task_stats = self._t3_norm_stats_by_task.get(str(task_name), None)
        if task_stats is not None:
            return task_stats
        return self._t3_norm_stats_global

    def _bundle_score_family(
        self,
        bundle: TrajectoryScoreBundle,
        *,
        task_name: str,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        t3_stats = self._resolve_t3_norm_stats(task_name)
        # T3 score = alpha * z(positive_energy) - beta * z(margin).
        alpha_terms = {
            "chunk_energy": (
                self.t3_alpha["chunk_energy"]
                * self._safe_standardize(
                    bundle.t3_energy_terms,
                    t3_stats.get("chunk_positive_mean", 0.0),
                    t3_stats.get("chunk_positive_std", 1.0),
                )
            ).astype(np.float32),
        }
        beta_terms = {
            "chunk_margin": (
                self.t3_beta["chunk_margin"]
                * self._safe_standardize(
                    bundle.t3_margin_terms,
                    t3_stats.get("chunk_margin_mean", 0.0),
                    t3_stats.get("chunk_margin_std", 1.0),
                )
            ).astype(np.float32),
        }
        components = {
            "chunk_energy": np.asarray(alpha_terms["chunk_energy"], dtype=np.float32),
            "chunk_margin": (-np.asarray(beta_terms["chunk_margin"], dtype=np.float32)).astype(np.float32),
        }
        step_scores = (
            np.asarray(components["chunk_energy"], dtype=np.float32)
            + np.asarray(components["chunk_margin"], dtype=np.float32)
        ).astype(np.float32)
        return {
            "step_scores": step_scores,
            "components": components,
            "alpha_terms": alpha_terms,
            "beta_terms": beta_terms,
        }

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
            # Mean mode smooths step scores into a running average alarm signal.
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

        # Max mode tracks worst-case evidence over the configured support window.
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
            total = total + np.abs(np.asarray(values, dtype=np.float32))
        safe_total = np.where(np.abs(total) > 1e-8, total, 1.0).astype(np.float32)
        return {
            key: (np.abs(np.asarray(values, dtype=np.float32)) / safe_total).astype(np.float32, copy=False)
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
        stacked = np.stack([np.abs(np.asarray(contributions[key], dtype=np.float32)) for key in keys], axis=0)
        dominant_idx = np.argmax(stacked, axis=0)
        return [str(keys[int(idx)]) for idx in dominant_idx.tolist()]

    def fit(
        self,
        normal_bank_trajectories: Sequence[PreparedTrajectory],
        calibration_trajectories: Optional[Sequence[PreparedTrajectory]] = None,
    ) -> DetectorCalibrationSummary:
        normal_bank_set = list(normal_bank_trajectories)
        calibration_set = (
            list(calibration_trajectories)
            if calibration_trajectories is not None
            else list(normal_bank_set)
        )
        if len(calibration_set) == 0:
            raise ValueError("Calibration requires at least one trajectory.")
        if len(normal_bank_set) == 0:
            raise ValueError("T3 normalization requires at least one success-bank trajectory.")

        bank_scored_bundles: list[tuple[str, TrajectoryScoreBundle]] = []
        for traj in normal_bank_set:
            bank_scored_bundles.append((str(traj.task_name), self.extractor.score_trajectory(traj)))

        scored_bundles: list[tuple[str, TrajectoryScoreBundle]] = []
        for traj in calibration_set:
            scored_bundles.append((str(traj.task_name), self.extractor.score_trajectory(traj)))

        self._compute_t3_norm_stats(bank_scored_bundles)
        calib_lambdas: list[np.ndarray] = []
        calib_lambdas_by_task: dict[str, list[np.ndarray]] = {}
        for task_name, bundle in scored_bundles:
            family = self._bundle_score_family(bundle, task_name=task_name)
            lambdas = self._aggregate_lambda(np.asarray(family["step_scores"], dtype=np.float32))
            calib_lambdas.append(lambdas)
            calib_lambdas_by_task.setdefault(str(task_name), []).append(lambdas)

        self._calib_lambdas = np.concatenate(calib_lambdas, axis=0).astype(np.float32)
        self._calib_lambdas_by_task = {
            task_name: np.concatenate(values, axis=0).astype(np.float32)
            for task_name, values in calib_lambdas_by_task.items()
            if values
        }
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
                "encoder_source": self.extractor.encoder.source_description,
                "noise_scale": float(self.extractor.model.noise_scale),
                "noise_sigma": float(self.extractor.model.noise_scale),
                "std_clamp_min": float(self.extractor.model.std_clamp_min),
                "model_architecture": MODEL_ARCHITECTURE,
                "score_semantics": self.SCORE_T3,
                "score_mode": self.SCORE_T3,
                "supported_score_modes": [self.SCORE_T3],
                "t3_formula": "alpha * z(chunk_positive) - beta * z(chunk_margin)",
                "latent_dim": int(self.extractor.latent_dim),
                "window_size": int(self.extractor.window_size),
                "chunk_dim": int(self.extractor.model.chunk_dim),
                "delta_init": float(self.delta),
                "delta_final": float(self.delta),
                "alpha": float(self.alpha),
                "beta": float(self.beta),
                "t3_alpha": {key: float(value) for key, value in self.t3_alpha.items()},
                "t3_beta": {key: float(value) for key, value in self.t3_beta.items()},
                "t3_norm_source": "success_bank",
                "num_t3_norm_bank_trajectories": int(len(normal_bank_set)),
                "num_t3_norm_bank_trajectories_by_task": {
                    task_name: int(sum(1 for bundle_task_name, _ in bank_scored_bundles if bundle_task_name == task_name))
                    for task_name in sorted({task_name for task_name, _ in bank_scored_bundles})
                },
                "t3_norm_stats_global": {
                    key: float(value)
                    for key, value in self._t3_norm_stats_global.items()
                },
                "t3_norm_stats_by_task": {
                    task_name: {
                        key: float(value)
                        for key, value in stats.items()
                    }
                    for task_name, stats in self._t3_norm_stats_by_task.items()
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
                "threshold_by_score": {self.SCORE_T3: float(self.threshold)},
                "thresholds_by_task_and_score": {
                    self.SCORE_T3: {
                        task_name: float(threshold)
                        for task_name, threshold in self.thresholds_by_task.items()
                    }
                },
            },
        )

    def detect_trajectory(
        self,
        trajectory: PreparedTrajectory,
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
        task_name = str(trajectory.task_name)
        family = self._bundle_score_family(bundle, task_name=task_name)
        step_scores = np.asarray(family["step_scores"], dtype=np.float32)
        lamb = self._aggregate_lambda(step_scores)

        component_scores = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in family["components"].items()
        }
        step_contribution_shares = self._compute_component_shares(component_scores)
        aggregate_contributions = self._aggregate_component_contributions(
            component_scores,
            step_scores=step_scores,
        )
        aggregate_contribution_shares = self._compute_component_shares(aggregate_contributions)
        dominant_step_terms = self._dominant_component_terms(component_scores)
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
        for t in range(lamb.shape[0]):
            pred = int(lamb[t] >= cur_threshold)
            preds[t] = pred
            ths[t] = float(cur_threshold)

            if adaptive_threshold and labels_np is not None:
                should_update = (t + 1) > warmup and ((t + 1 - warmup) % update_every == 0)
                if should_update:
                    label = int(labels_np[t])
                    # Increase delta when missing failures, decrease when over-triggering.
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
                "model_architecture": MODEL_ARCHITECTURE,
                "score_mode": self.score_mode,
                "supported_score_modes": [self.SCORE_T3],
                "threshold_source": "task" if task_threshold is not None else "global",
                "threshold_init": float(task_threshold if task_threshold is not None else self.threshold),
                "delta_final": float(cur_delta),
                "threshold_final": float(cur_threshold),
                "window_size": int(self.extractor.window_size),
                "aggregate_scores_by_mode": {self.SCORE_T3: np.asarray(lamb, dtype=np.float32)},
                "threshold_by_score": {
                    self.SCORE_T3: float(task_threshold if task_threshold is not None else self.threshold)
                },
                "step_contributions": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in component_scores.items()
                },
                "step_contribution_shares": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in step_contribution_shares.items()
                },
                "aggregate_contributions": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in aggregate_contributions.items()
                },
                "aggregate_contribution_shares": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in aggregate_contribution_shares.items()
                },
                "dominant_step_terms": list(dominant_step_terms),
                "dominant_aggregate_terms": list(dominant_aggregate_terms),
                "first_crossing_index": first_crossing_index,
                "first_crossing_dominant_term": first_crossing_dominant_term,
                "first_crossing_term_shares": first_crossing_term_shares,
                "t3_alpha_terms": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in family["alpha_terms"].items()
                },
                "t3_beta_terms": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in family["beta_terms"].items()
                },
                "t3_alpha": {key: float(value) for key, value in self.t3_alpha.items()},
                "t3_beta": {key: float(value) for key, value in self.t3_beta.items()},
            },
        )

    def close(self) -> None:
        self.extractor.close()
