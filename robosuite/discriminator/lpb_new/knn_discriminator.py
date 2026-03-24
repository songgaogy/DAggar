from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.discriminator.utils.base import OfflineTrajectoryDiscriminator
from robosuite.discriminator.utils.types import DetectorCalibrationSummary, TrajectoryDetectionResult

from robosuite.discriminator.lpb_new.dataset import LatentTrajectory
from robosuite.discriminator.lpb_new.model import LatentDynamicsModel, build_latent_dynamics_predictor


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


@torch.no_grad()
def knn_min_sqdist(
    query: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 8192,
    exclude_range: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if bank.shape[0] == 0:
        raise ValueError("bank cannot be empty")

    q = query
    b = bank.to(device=q.device, dtype=q.dtype)
    best = torch.full((q.shape[0],), float("inf"), device=q.device, dtype=q.dtype)

    ex_start, ex_end = (-1, -1) if exclude_range is None else exclude_range
    csz = int(chunk_size)

    for start in range(0, b.shape[0], csz):
        end = min(start + csz, b.shape[0])
        chunk = b[start:end]
        d2 = torch.cdist(q, chunk, p=2.0).pow(2)

        if exclude_range is not None:
            lo = max(start, ex_start)
            hi = min(end, ex_end)
            if hi > lo:
                d2[:, (lo - start) : (hi - start)] = float("inf")

        cur = torch.min(d2, dim=1).values
        best = torch.minimum(best, cur)
    return best


@dataclass
class DetectionResult:
    step_scores: np.ndarray
    lambda_values: np.ndarray
    thresholds: np.ndarray
    preds: np.ndarray
    delta_final: float


class LPBFeatureExtractor:
    """
    Build LPB features directly from frozen policy latents and the learned transition model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        action_horizon: int = -1,
        normalize_feature: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.normalize_feature = bool(normalize_feature)
        self.model, self.action_horizon, self.action_dim, self.latent_dim = self._load_model(
            checkpoint_path=checkpoint_path,
            override_action_horizon=action_horizon,
        )

    def _load_model(
        self,
        checkpoint_path: str,
        override_action_horizon: int,
    ) -> tuple[LatentDynamicsModel, int, int, int]:
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing key `model`: {checkpoint_path}")

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim"))
        action_dim = int(payload.get("action_dim"))
        horizon_ckpt = int(payload.get("horizon", 1))
        action_horizon = horizon_ckpt if int(override_action_horizon) <= 0 else int(override_action_horizon)

        predictor = build_latent_dynamics_predictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            cfg_model=_cfg_get(cfg, "model", {}),
            transition_horizon=action_horizon,
        )
        model = LatentDynamicsModel(predictor=predictor)
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
    def encode_trajectory(self, traj: LatentTrajectory) -> torch.Tensor:
        t_len = min(int(traj.latents.shape[0]), int(traj.actions.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")

        latents = self._prepare_latents(traj.latents, t_len=t_len)
        act_chunks = self._prepare_actions(traj.actions, t_len=t_len)
        latents_t = torch.from_numpy(latents)
        act_t = torch.from_numpy(act_chunks)

        feats = []
        for start in range(0, t_len, self.batch_size):
            end = min(start + self.batch_size, t_len)
            latent_b = latents_t[start:end].to(self.device)
            act_b = act_t[start:end].to(self.device)

            feat = self.model.extract_feature(
                current_latent=latent_b,
                action_sequence=act_b,
            )
            if self.normalize_feature:
                feat = F.normalize(feat, p=2.0, dim=-1)
            feats.append(feat.detach().cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_trajectory_with_transition_error(
        self,
        traj: LatentTrajectory,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_len = min(int(traj.latents.shape[0]), int(traj.actions.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")

        horizon = int(self.action_horizon)
        valid_len = t_len - horizon
        if valid_len <= 0:
            raise ValueError(f"Trajectory length {t_len} is too short for action_horizon={horizon}")

        latents = self._prepare_latents(traj.latents, t_len=t_len)
        act_chunks = self._prepare_actions(traj.actions, t_len=t_len)
        latents_t = torch.from_numpy(latents)
        act_t = torch.from_numpy(act_chunks)

        feat_all = []
        for start in range(0, t_len, self.batch_size):
            end = min(start + self.batch_size, t_len)
            latent_b = latents_t[start:end].to(self.device)
            act_b = act_t[start:end].to(self.device)

            feat = self.model.extract_feature(
                current_latent=latent_b,
                action_sequence=act_b,
            )
            if self.normalize_feature:
                feat = F.normalize(feat, p=2.0, dim=-1)
            feat_all.append(feat.detach().cpu())

        latents_dev = latents_t.to(self.device)
        act_dev = act_t.to(self.device)
        errs = []
        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            pred = self.model(
                current_latent=latents_dev[start:end],
                action_sequence=act_dev[start:end],
            )
            target = latents_dev[start + horizon : end + horizon]
            err = (pred["pred_latent"] - target).pow(2).mean(dim=-1)
            errs.append(err.detach().cpu())

        feat_all_t = torch.cat(feat_all, dim=0)
        err_t = torch.cat(errs, dim=0)
        return feat_all_t[:valid_len], err_t


class AdaptiveKNNDiscriminator:
    def __init__(
        self,
        delta: float = 10.0,
        delta_step: float = 1.0,
        knn_chunk_size: int = 8192,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        aux_weight: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self.delta = float(delta)
        self.delta_step = float(delta_step)
        self.knn_chunk_size = int(knn_chunk_size)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.aux_weight = float(aux_weight)
        if self.lambda_mode not in {"mean", "max"}:
            raise ValueError("lambda_mode must be 'mean' or 'max'")
        if self.lambda_window_size == 0:
            raise ValueError("lambda_window_size must be -1 or >=1")

        self.device = _resolve_device(device)
        self.bank: Optional[torch.Tensor] = None
        self._offsets: list[tuple[int, int]] = []
        self._calib_lambdas: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self._aux_scale: float = 1.0

    @staticmethod
    def _compute_threshold(values: np.ndarray, delta: float) -> float:
        if values.size == 0:
            raise ValueError("Cannot calibrate threshold from empty values")
        q = 100.0 * (1.0 - float(delta) / 100.0)
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
        vals = step_scores.astype(np.float32, copy=False)
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

    def _combine_scores(self, knn_scores: np.ndarray, aux_scores: Optional[np.ndarray]) -> np.ndarray:
        combined = knn_scores.astype(np.float32, copy=False)
        if self.aux_weight <= 0.0 or aux_scores is None:
            return combined
        aux = aux_scores.astype(np.float32, copy=False)
        if aux.shape != combined.shape:
            raise ValueError(f"aux shape mismatch: {aux.shape} vs {combined.shape}")
        scale = float(self._aux_scale) if self._aux_scale > 0 else 1.0
        return combined + float(self.aux_weight) * (aux / scale)

    @torch.no_grad()
    def fit(
        self,
        expert_sequences: Sequence[torch.Tensor],
        calibration_sequences: Optional[Sequence[torch.Tensor]] = None,
        calibration_aux: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        if len(expert_sequences) == 0:
            raise ValueError("fit requires non-empty expert_sequences")

        seqs = [seq.detach().to(self.device, dtype=torch.float32) for seq in expert_sequences if seq.numel() > 0]
        if len(seqs) == 0:
            raise ValueError("All expert_sequences are empty")

        offsets: list[tuple[int, int]] = []
        start = 0
        for seq in seqs:
            end = start + seq.shape[0]
            offsets.append((start, end))
            start = end
        bank = torch.cat(seqs, dim=0)

        self.bank = bank
        self._offsets = offsets

        if calibration_aux is not None and len(calibration_aux) > 0 and self.aux_weight > 0.0:
            aux_cat = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in calibration_aux], axis=0)
            self._aux_scale = float(np.mean(aux_cat)) if aux_cat.size > 0 else 1.0
            if self._aux_scale <= 1e-8:
                self._aux_scale = 1.0

        calib_lambdas: list[np.ndarray] = []
        if calibration_sequences is None:
            if calibration_aux is not None and len(calibration_aux) != len(seqs):
                raise ValueError(f"calibration_aux length mismatch: {len(calibration_aux)} vs {len(seqs)}")
            for idx, seq in enumerate(seqs):
                ex = offsets[idx]
                step = knn_min_sqdist(
                    query=seq,
                    bank=bank,
                    chunk_size=self.knn_chunk_size,
                    exclude_range=ex,
                )
                step_np = step.detach().cpu().numpy().astype(np.float32)
                combined = self._combine_scores(
                    step_np,
                    None if calibration_aux is None else np.asarray(calibration_aux[idx], dtype=np.float32),
                )
                calib_lambdas.append(self._aggregate_lambda(combined))
        else:
            cseqs = [
                seq.detach().to(self.device, dtype=torch.float32)
                for seq in calibration_sequences
                if seq.numel() > 0
            ]
            if len(cseqs) == 0:
                raise ValueError("All calibration_sequences are empty")
            if calibration_aux is not None and len(calibration_aux) != len(cseqs):
                raise ValueError(f"calibration_aux length mismatch: {len(calibration_aux)} vs {len(cseqs)}")
            for idx, seq in enumerate(cseqs):
                step = knn_min_sqdist(
                    query=seq,
                    bank=bank,
                    chunk_size=self.knn_chunk_size,
                    exclude_range=None,
                )
                step_np = step.detach().cpu().numpy().astype(np.float32)
                combined = self._combine_scores(
                    step_np,
                    None if calibration_aux is None else np.asarray(calibration_aux[idx], dtype=np.float32),
                )
                calib_lambdas.append(self._aggregate_lambda(combined))

        self._calib_lambdas = np.concatenate(calib_lambdas, axis=0).astype(np.float32)
        self.threshold = self._compute_threshold(self._calib_lambdas, self.delta)
        return float(self.threshold)

    @torch.no_grad()
    def detect_sequence(
        self,
        features: torch.Tensor,
        labels: Optional[np.ndarray] = None,
        adaptive_delta: bool = False,
        aux_scores: Optional[np.ndarray] = None,
        delta_min: float = 0.0,
        delta_max: float = 100.0,
        warmup_steps: int = 0,
        update_interval: int = 1,
    ) -> DetectionResult:
        if self.bank is None or self.threshold is None or self._calib_lambdas is None:
            raise RuntimeError("Call fit(...) before detect_sequence(...)")

        feat = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)
        step_knn = knn_min_sqdist(
            feat,
            self.bank,
            chunk_size=self.knn_chunk_size,
        ).detach().cpu().numpy().astype(np.float32)
        step = self._combine_scores(step_knn, aux_scores)
        lamb = self._aggregate_lambda(step)

        preds = np.zeros_like(lamb, dtype=np.int64)
        ths = np.zeros_like(lamb, dtype=np.float32)

        cur_delta = float(self.delta)
        cur_threshold = float(self.threshold)
        dmin = float(np.clip(delta_min, 0.0, 100.0))
        dmax = float(np.clip(delta_max, 0.0, 100.0))
        if dmin > dmax:
            dmin, dmax = dmax, dmin
        warmup = max(0, int(warmup_steps))
        update_every = max(1, int(update_interval))

        for t in range(lamb.shape[0]):
            pred = int(lamb[t] >= cur_threshold)
            preds[t] = pred
            ths[t] = float(cur_threshold)

            if adaptive_delta and labels is not None:
                should_update = (t + 1) > warmup and ((t + 1 - warmup) % update_every == 0)
                if should_update:
                    label = int(labels[t])
                    if label == 1 and pred == 0:
                        cur_delta += self.delta_step
                    elif label == 0 and pred == 1:
                        cur_delta -= self.delta_step
                    cur_delta = float(np.clip(cur_delta, dmin, dmax))
                    cur_threshold = self._compute_threshold(self._calib_lambdas, cur_delta)

        self.delta = float(cur_delta)
        self.threshold = float(cur_threshold)
        return DetectionResult(
            step_scores=step,
            lambda_values=lamb,
            thresholds=ths,
            preds=preds,
            delta_final=float(self.delta),
        )


class LPBKNNDiscriminator(OfflineTrajectoryDiscriminator[LatentTrajectory]):
    def __init__(
        self,
        checkpoint_path: str,
        *,
        feature_device: str = "cuda",
        feature_batch_size: int = 256,
        action_horizon: int = -1,
        normalize_feature: bool = True,
        use_transition_error: bool = True,
        detector_device: str = "cuda",
        delta: float = 10.0,
        delta_step: float = 0.5,
        knn_chunk_size: int = 8192,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        transition_aux_weight: float = 0.0,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.use_transition_error = bool(use_transition_error)
        self.extractor = LPBFeatureExtractor(
            checkpoint_path=self.checkpoint_path,
            device=str(feature_device),
            batch_size=int(feature_batch_size),
            action_horizon=int(action_horizon),
            normalize_feature=bool(normalize_feature),
        )
        self.detector = AdaptiveKNNDiscriminator(
            delta=float(delta),
            delta_step=float(delta_step),
            knn_chunk_size=int(knn_chunk_size),
            lambda_mode=str(lambda_mode),
            lambda_window_size=int(lambda_window_size),
            aux_weight=float(transition_aux_weight),
            device=str(detector_device),
        )

    @property
    def name(self) -> str:
        return "lpb_new_knn"

    def _encode_set(
        self,
        trajectories: Sequence[LatentTrajectory],
    ) -> tuple[list[torch.Tensor], list[Optional[np.ndarray]]]:
        features: list[torch.Tensor] = []
        aux_values: list[Optional[np.ndarray]] = []
        for traj in trajectories:
            if self.use_transition_error:
                feat, aux = self.extractor.encode_trajectory_with_transition_error(traj)
                features.append(feat)
                aux_values.append(aux.detach().cpu().numpy().astype(np.float32))
            else:
                features.append(self.extractor.encode_trajectory(traj))
                aux_values.append(None)
        return features, aux_values

    def fit(
        self,
        normal_bank_trajectories: Sequence[LatentTrajectory],
        calibration_trajectories: Optional[Sequence[LatentTrajectory]] = None,
    ) -> DetectorCalibrationSummary:
        if calibration_trajectories is None:
            calibration_trajectories = normal_bank_trajectories

        bank_features, _ = self._encode_set(normal_bank_trajectories)
        calib_features, calib_aux = self._encode_set(calibration_trajectories)
        threshold = self.detector.fit(
            expert_sequences=bank_features,
            calibration_sequences=calib_features,
            calibration_aux=None if not self.use_transition_error else calib_aux,
        )
        return DetectorCalibrationSummary(
            detector_name=self.name,
            threshold=float(threshold),
            metadata={
                "lpb_ckpt": self.checkpoint_path,
                "use_transition_error": bool(self.use_transition_error),
                "transition_aux_weight": float(self.detector.aux_weight),
                "delta_init": float(self.detector.delta),
                "delta_final": float(self.detector.delta),
                "knn_chunk_size": int(self.detector.knn_chunk_size),
                "lambda_mode": str(self.detector.lambda_mode),
                "lambda_window_size": int(self.detector.lambda_window_size),
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
        if self.use_transition_error:
            features, aux = self.extractor.encode_trajectory_with_transition_error(trajectory)
            aux_np = aux.detach().cpu().numpy().astype(np.float32)
        else:
            features = self.extractor.encode_trajectory(trajectory)
            aux_np = None
        output = self.detector.detect_sequence(
            features=features,
            labels=None if labels is None else np.asarray(labels, dtype=np.int64),
            adaptive_delta=bool(adaptive_threshold),
            aux_scores=aux_np,
            delta_min=float(delta_min),
            delta_max=float(delta_max),
            warmup_steps=int(warmup_steps),
            update_interval=int(update_interval),
        )
        return TrajectoryDetectionResult(
            detector_name=self.name,
            step_scores=np.asarray(output.step_scores, dtype=np.float32),
            aggregate_scores=np.asarray(output.lambda_values, dtype=np.float32),
            thresholds=np.asarray(output.thresholds, dtype=np.float32),
            predictions=np.asarray(output.preds, dtype=np.int64),
            aux_scores=None if aux_np is None else np.asarray(aux_np, dtype=np.float32),
            labels=None if labels is None else np.asarray(labels, dtype=np.int64),
            metadata={
                "delta_final": float(output.delta_final),
                "threshold_final": float(self.detector.threshold if self.detector.threshold is not None else np.nan),
            },
        )
