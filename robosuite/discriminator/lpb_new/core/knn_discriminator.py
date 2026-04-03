from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.discriminator.utils.base import OfflineTrajectoryDiscriminator
from robosuite.discriminator.utils.types import DetectorCalibrationSummary, TrajectoryDetectionResult

from .dataset import LatentTrajectory
from .model import LatentDynamicsModel, build_latent_dynamics_predictor


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


@torch.no_grad()
def knn_topk_sqdist(
    query: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    chunk_size: int = 8192,
    exclude_range: Optional[tuple[int, int]] = None,
    allowed_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if bank.shape[0] == 0:
        raise ValueError("bank cannot be empty")
    if allowed_mask is not None:
        if allowed_mask.ndim != 1 or allowed_mask.shape[0] != bank.shape[0]:
            raise ValueError(
                f"allowed_mask must be shape ({bank.shape[0]},), got {tuple(allowed_mask.shape)}"
            )
        allowed_mask = allowed_mask.to(device=query.device, dtype=torch.bool)
        valid_total = int(allowed_mask.sum().item())
        if exclude_range is not None:
            ex_start, ex_end = exclude_range
            lo = max(0, int(ex_start))
            hi = min(int(bank.shape[0]), int(ex_end))
            if hi > lo:
                valid_total -= int(allowed_mask[lo:hi].sum().item())
        if valid_total <= 0:
            raise ValueError("No valid bank entries remain after applying allowed_mask/exclude_range.")
        k_eff = max(1, min(int(k), valid_total))
    else:
        k_eff = max(1, min(int(k), int(bank.shape[0])))

    q = query
    b = bank.to(device=q.device, dtype=q.dtype)
    best_d2 = torch.full((q.shape[0], k_eff), float("inf"), device=q.device, dtype=q.dtype)
    best_idx = torch.full((q.shape[0], k_eff), -1, device=q.device, dtype=torch.long)

    ex_start, ex_end = (-1, -1) if exclude_range is None else exclude_range
    csz = int(chunk_size)

    for start in range(0, b.shape[0], csz):
        end = min(start + csz, b.shape[0])
        chunk = b[start:end]
        d2 = torch.cdist(q, chunk, p=2.0).pow(2)

        if allowed_mask is not None:
            chunk_mask = allowed_mask[start:end]
            if not bool(chunk_mask.any().item()):
                continue
            d2[:, ~chunk_mask] = float("inf")

        if exclude_range is not None:
            lo = max(start, ex_start)
            hi = min(end, ex_end)
            if hi > lo:
                d2[:, (lo - start) : (hi - start)] = float("inf")

        local_k = min(k_eff, int(chunk.shape[0]))
        local_d2, local_idx = torch.topk(d2, k=local_k, dim=1, largest=False)
        local_idx = local_idx + int(start)

        merged_d2 = torch.cat([best_d2, local_d2], dim=1)
        merged_idx = torch.cat([best_idx, local_idx], dim=1)
        best_d2, order = torch.topk(merged_d2, k=k_eff, dim=1, largest=False)
        best_idx = torch.gather(merged_idx, dim=1, index=order)

    return best_d2, best_idx


@dataclass
class DetectionResult:
    step_scores: np.ndarray
    lambda_values: np.ndarray
    thresholds: np.ndarray
    preds: np.ndarray
    delta_final: float


@dataclass
class EncodedTrajectoryBundle:
    learned_features: torch.Tensor
    policy_chunk_features: torch.Tensor
    target_deltas: torch.Tensor
    transition_errors: torch.Tensor
    task_index: int


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
        normalize_policy_chunk: bool = True,
        policy_history_steps: int = 4,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.normalize_feature = bool(normalize_feature)
        self.normalize_policy_chunk = bool(normalize_policy_chunk)
        self.policy_history_steps = max(1, int(policy_history_steps))
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

    def _build_policy_state_history(self, latents: torch.Tensor, valid_len: int) -> torch.Tensor:
        """Flatten a short latent history so policy neighborhoods retain temporal context."""
        hist = max(1, int(self.policy_history_steps))
        history_chunks = []
        for t in range(valid_len):
            start = max(0, t - hist + 1)
            chunk = latents[start : t + 1]
            if int(chunk.shape[0]) < hist:
                pad = chunk[:1].expand(hist - int(chunk.shape[0]), -1)
                chunk = torch.cat([pad, chunk], dim=0)
            history_chunks.append(chunk.reshape(-1))
        return torch.stack(history_chunks, dim=0)

    @torch.no_grad()
    def encode_trajectory_bundle(self, traj: LatentTrajectory) -> EncodedTrajectoryBundle:
        """Encode one trajectory into all feature banks used by the discriminator."""
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

        valid_latents_t = latents_t[:valid_len]
        valid_act_t = act_t[:valid_len]
        target_delta_t = latents_t[horizon : horizon + valid_len] - valid_latents_t

        feats: list[torch.Tensor] = []
        errs: list[torch.Tensor] = []
        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            latent_b = valid_latents_t[start:end].to(self.device)
            act_b = valid_act_t[start:end].to(self.device)

            # Reuse the frozen dynamics model for both KNN features and prediction error.
            feat = self.model.extract_feature(
                current_latent=latent_b,
                action_sequence=act_b,
            )
            if self.normalize_feature:
                feat = F.normalize(feat, p=2.0, dim=-1)
            feats.append(feat.detach().cpu())
            pred = self.model(
                current_latent=latent_b,
                action_sequence=act_b,
            )
            target = latents_t[horizon + start : horizon + end].to(self.device)
            err = (pred["pred_latent"] - target).pow(2).mean(dim=-1)
            errs.append(err.detach().cpu())

        policy_state_history = self._build_policy_state_history(valid_latents_t, valid_len=valid_len)
        policy_chunk = torch.cat(
            [
                policy_state_history,
                valid_act_t.reshape(valid_len, -1),
            ],
            dim=-1,
        )
        if self.normalize_policy_chunk:
            policy_chunk = F.normalize(policy_chunk, p=2.0, dim=-1)

        return EncodedTrajectoryBundle(
            learned_features=torch.cat(feats, dim=0),
            policy_chunk_features=policy_chunk,
            target_deltas=target_delta_t,
            transition_errors=torch.cat(errs, dim=0),
            task_index=int(traj.task_index),
        )

    @torch.no_grad()
    def encode_trajectory(self, traj: LatentTrajectory) -> torch.Tensor:
        bundle = self.encode_trajectory_bundle(traj)
        return bundle.learned_features

    @torch.no_grad()
    def encode_trajectory_with_transition_error(
        self,
        traj: LatentTrajectory,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bundle = self.encode_trajectory_bundle(traj)
        return bundle.learned_features, bundle.transition_errors


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
    COMPONENT_ORDER: tuple[str, ...] = (
        "feature_knn",
        "transition_error",
        "policy_chunk",
        "neighbor_dynamics",
    )

    def __init__(
        self,
        checkpoint_path: str,
        *,
        feature_device: str = "cuda",
        feature_batch_size: int = 256,
        action_horizon: int = -1,
        normalize_feature: bool = True,
        normalize_policy_chunk: bool = True,
        policy_history_steps: int = 4,
        use_transition_error: bool = True,
        detector_device: str = "cuda",
        delta: float = 10.0,
        delta_step: float = 0.5,
        knn_chunk_size: int = 8192,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        feature_knn_weight: float = 1.0,
        transition_aux_weight: float = 0.0,
        policy_chunk_weight: float = 1.0,
        dynamics_weight: float = 1.0,
        neighbor_topk: int = 8,
        dynamics_temperature: float = 1.0,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.use_transition_error = bool(use_transition_error)
        self.extractor = LPBFeatureExtractor(
            checkpoint_path=self.checkpoint_path,
            device=str(feature_device),
            batch_size=int(feature_batch_size),
            action_horizon=int(action_horizon),
            normalize_feature=bool(normalize_feature),
            normalize_policy_chunk=bool(normalize_policy_chunk),
            policy_history_steps=int(policy_history_steps),
        )
        self.feature_knn_weight = float(feature_knn_weight)
        self.transition_aux_weight = float(transition_aux_weight)
        self.policy_chunk_weight = float(policy_chunk_weight)
        self.dynamics_weight = float(dynamics_weight)
        self.neighbor_topk = max(1, int(neighbor_topk))
        self.dynamics_temperature = max(float(dynamics_temperature), 1e-6)
        if (
            self.feature_knn_weight <= 0.0
            and self.transition_aux_weight <= 0.0
            and self.policy_chunk_weight <= 0.0
            and self.dynamics_weight <= 0.0
        ):
            raise ValueError("At least one score weight must be positive.")
        self.detector = AdaptiveKNNDiscriminator(
            delta=float(delta),
            delta_step=float(delta_step),
            knn_chunk_size=int(knn_chunk_size),
            lambda_mode=str(lambda_mode),
            lambda_window_size=int(lambda_window_size),
            aux_weight=0.0,
            device=str(detector_device),
        )
        self.learned_bank: Optional[torch.Tensor] = None
        self.policy_chunk_bank: Optional[torch.Tensor] = None
        self.target_delta_bank: Optional[torch.Tensor] = None
        self.policy_chunk_task_index_bank: Optional[torch.Tensor] = None
        self._offsets: list[tuple[int, int]] = []
        self._score_scales: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "lpb_new_knn"

    def _encode_set(
        self,
        trajectories: Sequence[LatentTrajectory],
    ) -> list[EncodedTrajectoryBundle]:
        bundles: list[EncodedTrajectoryBundle] = []
        for traj in trajectories:
            bundles.append(self.extractor.encode_trajectory_bundle(traj))
        return bundles

    @staticmethod
    def _build_offsets(bundles: Sequence[EncodedTrajectoryBundle]) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        start = 0
        for bundle in bundles:
            end = start + int(bundle.learned_features.shape[0])
            offsets.append((start, end))
            start = end
        return offsets

    def _set_banks(self, bundles: Sequence[EncodedTrajectoryBundle]) -> None:
        if not bundles:
            raise ValueError("Cannot build banks from an empty bundle list.")
        self._offsets = self._build_offsets(bundles)
        self.learned_bank = torch.cat(
            [bundle.learned_features for bundle in bundles],
            dim=0,
        ).to(self.detector.device, dtype=torch.float32)
        self.policy_chunk_bank = torch.cat(
            [bundle.policy_chunk_features for bundle in bundles],
            dim=0,
        ).to(self.detector.device, dtype=torch.float32)
        self.target_delta_bank = torch.cat(
            [bundle.target_deltas for bundle in bundles],
            dim=0,
        ).to(self.detector.device, dtype=torch.float32)
        self.policy_chunk_task_index_bank = torch.cat(
            [
                torch.full(
                    (int(bundle.policy_chunk_features.shape[0]),),
                    int(bundle.task_index),
                    dtype=torch.int64,
                )
                for bundle in bundles
            ],
            dim=0,
        ).to(self.detector.device)

    @staticmethod
    def _compute_scale(values: np.ndarray) -> float:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return 1.0
        scale = float(np.mean(finite))
        return scale if scale > 1e-8 else 1.0

    def _component_weights(self) -> list[tuple[str, float]]:
        return [
            ("feature_knn", float(self.feature_knn_weight)),
            ("transition_error", float(self.transition_aux_weight)),
            ("policy_chunk", float(self.policy_chunk_weight)),
            ("neighbor_dynamics", float(self.dynamics_weight)),
        ]

    def _compute_weighted_component_scores(
        self,
        scores: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Apply calibration scaling and configured weights to each raw component."""
        weighted: dict[str, np.ndarray] = {}
        for key, weight in self._component_weights():
            if weight <= 0.0 or key not in scores:
                continue
            scale = float(self._score_scales.get(key, 1.0))
            if scale <= 1e-8:
                scale = 1.0
            weighted[key] = (
                float(weight) * (np.asarray(scores[key], dtype=np.float32) / scale)
            ).astype(np.float32, copy=False)
        return weighted

    def _compute_lambda_support_indices(
        self,
        step_scores: np.ndarray,
    ) -> np.ndarray:
        vals = np.asarray(step_scores, dtype=np.float32).reshape(-1)
        n = int(vals.shape[0])
        support = np.zeros((n,), dtype=np.int64)
        if n == 0:
            return support

        window = int(self.detector.lambda_window_size)
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

    def _aggregate_component_contributions(
        self,
        weighted_scores: dict[str, np.ndarray],
        step_scores: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Project step-wise contributions into the same lambda space as the detector."""
        if self.detector.lambda_mode == "mean":
            return {
                key: self.detector._aggregate_lambda(values)
                for key, values in weighted_scores.items()
            }

        support = self._compute_lambda_support_indices(step_scores)
        return {
            key: np.asarray(values, dtype=np.float32)[support].astype(np.float32, copy=False)
            for key, values in weighted_scores.items()
        }

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

    def _combine_component_scores(
        self,
        scores: dict[str, np.ndarray],
    ) -> np.ndarray:
        if not scores:
            raise ValueError("scores cannot be empty")
        lengths = {int(np.asarray(values).shape[0]) for values in scores.values()}
        if len(lengths) != 1:
            raise ValueError(f"Component length mismatch: {sorted(lengths)}")

        weighted_scores = self._compute_weighted_component_scores(scores)
        step = np.zeros((next(iter(lengths)),), dtype=np.float32)
        for values in weighted_scores.values():
            step = step + np.asarray(values, dtype=np.float32)
        return step.astype(np.float32, copy=False)

    def _compute_component_scores(
        self,
        bundle: EncodedTrajectoryBundle,    # a set for all features
        *,
        exclude_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, np.ndarray]:
        """Compute the four discriminator components before weighting and aggregation."""
        # check whether all banks exist
        if (
            self.learned_bank is None
            or self.policy_chunk_bank is None
            or self.target_delta_bank is None
            or self.policy_chunk_task_index_bank is None
        ):
            raise RuntimeError("Call fit(...) before computing scores.")

        scores: dict[str, np.ndarray] = {}
        detector_device = self.detector.device

        # 1) KNN using `bundle.learned_features` -> z = f(s,a)
        if self.feature_knn_weight > 0.0:
            learned_query = bundle.learned_features.to(detector_device, dtype=torch.float32)    # NOTE: using all tasks features
            scores["feature_knn"] = (
                knn_min_sqdist(
                    query=learned_query,
                    bank=self.learned_bank,
                    chunk_size=self.detector.knn_chunk_size,
                    exclude_range=exclude_range,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        # 2) transition error
        if self.transition_aux_weight > 0.0 and self.use_transition_error:      # NOTE: using all tasks features
            scores["transition_error"] = (
                bundle.transition_errors.detach().cpu().numpy().astype(np.float32)
            )
        
        # 3) & 4) 
        if self.policy_chunk_weight > 0.0 or self.dynamics_weight > 0.0:
            # retrive policy encoded features
            policy_query = bundle.policy_chunk_features.to(detector_device, dtype=torch.float32)
            # restrict policy-neighbor lookup to the same task to avoid trivial multitask leakage.
            same_task_mask = self.policy_chunk_task_index_bank == int(bundle.task_index)

            topk_d2, topk_idx = knn_topk_sqdist(
                query=policy_query,
                bank=self.policy_chunk_bank,
                k=self.neighbor_topk,
                chunk_size=self.detector.knn_chunk_size,
                exclude_range=exclude_range,
                allowed_mask=same_task_mask,
            )

            # 3) policy encoded features top-k square distance
            if self.policy_chunk_weight > 0.0:
                scores["policy_chunk"] = (
                    topk_d2.mean(dim=1).detach().cpu().numpy().astype(np.float32)
                )

            #  4) dynamic difference between current and expert
            if self.dynamics_weight > 0.0:
                neighbor_deltas = self.target_delta_bank[topk_idx]
                query_delta = bundle.target_deltas.to(detector_device, dtype=torch.float32).unsqueeze(1)
                dyn_err = (query_delta - neighbor_deltas).pow(2).mean(dim=-1)
                weights = torch.softmax(-topk_d2 / float(self.dynamics_temperature), dim=1)
                dyn_score = (weights * dyn_err).sum(dim=1)
                scores["neighbor_dynamics"] = (
                    dyn_score.detach().cpu().numpy().astype(np.float32)
                )

        return scores

    def fit(
        self,
        normal_bank_trajectories: Sequence[LatentTrajectory],
        calibration_trajectories: Optional[Sequence[LatentTrajectory]] = None,
    ) -> DetectorCalibrationSummary:
        """Encode the normal bank and calibrate score scales plus the lambda threshold."""
        if len(normal_bank_trajectories) == 0:
            raise ValueError("normal_bank_trajectories cannot be empty")

        bank_bundles = self._encode_set(normal_bank_trajectories)
        self._set_banks(bank_bundles)

        calibration_bundles = bank_bundles if calibration_trajectories is None else self._encode_set(calibration_trajectories)
        calibration_ranges = self._offsets if calibration_trajectories is None else [None] * len(calibration_bundles)

        calibration_component_scores = [
            self._compute_component_scores(bundle, exclude_range=exclude_range)
            for bundle, exclude_range in zip(calibration_bundles, calibration_ranges)
        ]
        if not calibration_component_scores:
            raise ValueError("No calibration bundles available.")

        merged_component_scores: dict[str, np.ndarray] = {}
        for key in calibration_component_scores[0].keys():
            merged_component_scores[key] = np.concatenate(
                [scores[key] for scores in calibration_component_scores if key in scores],
                axis=0,
            ).astype(np.float32)
        self._score_scales = {
            key: self._compute_scale(values)
            for key, values in merged_component_scores.items()
        }

        calib_lambdas = [
            self.detector._aggregate_lambda(self._combine_component_scores(scores))
            for scores in calibration_component_scores
        ]
        self.detector._calib_lambdas = np.concatenate(calib_lambdas, axis=0).astype(np.float32)
        self.detector.threshold = self.detector._compute_threshold(
            self.detector._calib_lambdas,
            self.detector.delta,
        )
        threshold = float(self.detector.threshold)
        return DetectorCalibrationSummary(
            detector_name=self.name,
            threshold=float(threshold),
            metadata={
                "lpb_ckpt": self.checkpoint_path,
                "use_transition_error": bool(self.use_transition_error),
                "feature_knn_weight": float(self.feature_knn_weight),
                "transition_aux_weight": float(self.transition_aux_weight),
                "policy_chunk_weight": float(self.policy_chunk_weight),
                "dynamics_weight": float(self.dynamics_weight),
                "neighbor_topk": int(self.neighbor_topk),
                "dynamics_temperature": float(self.dynamics_temperature),
                "score_scales": dict(self._score_scales),
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
        """
        Run detection on one trajectory and attach per-term attribution metadata.
        """
        # make sure dectector has been initialized
        if self.detector.threshold is None or self.detector._calib_lambdas is None:
            raise RuntimeError("Call fit(...) before detect_trajectory(...)")

        # pre-process, create bundle for fast operation later
        bundle = self.extractor.encode_trajectory_bundle(trajectory)

        # [MAIN] lambda computing ######################################
        # compute 4 components
        component_scores = self._compute_component_scores(bundle)
        # normalize @ add weight
        step_scores = self._combine_component_scores(component_scores)
        # aggregate and post-process (like sliding windows)
        lamb = self.detector._aggregate_lambda(step_scores)
        ################################################################

        # [test] for anaylising importance for different components
        weighted_component_scores = self._compute_weighted_component_scores(component_scores)
        aggregate_contributions = self._aggregate_component_contributions(
            weighted_component_scores,
            step_scores=step_scores,
        )
        step_contribution_shares = self._compute_component_shares(weighted_component_scores)
        aggregate_contribution_shares = self._compute_component_shares(aggregate_contributions)
        dominant_step_terms = self._dominant_component_terms(weighted_component_scores)
        dominant_aggregate_terms = self._dominant_component_terms(aggregate_contributions)

        # [MAIN] failure discriminating ############################################################################
        preds = np.zeros_like(lamb, dtype=np.int64)
        ths = np.zeros_like(lamb, dtype=np.float32)

        cur_delta = float(self.detector.delta)
        cur_threshold = float(self.detector.threshold)
        dmin = float(np.clip(delta_min, 0.0, 100.0))
        dmax = float(np.clip(delta_max, 0.0, 100.0))
        if dmin > dmax:
            dmin, dmax = dmax, dmin
        warmup = max(0, int(warmup_steps))
        update_every = max(1, int(update_interval))

        labels_np = None if labels is None else np.asarray(labels, dtype=np.int64)
        for t in range(lamb.shape[0]):
            pred = int(lamb[t] >= cur_threshold)
            preds[t] = pred
            ths[t] = float(cur_threshold)

            if adaptive_threshold and labels_np is not None:
                # Online delta adaptation only nudges the threshold every few steps.
                should_update = (t + 1) > warmup and ((t + 1 - warmup) % update_every == 0)
                if should_update:
                    label = int(labels_np[t])
                    if label == 1 and pred == 0:
                        cur_delta += self.detector.delta_step
                    elif label == 0 and pred == 1:
                        cur_delta -= self.detector.delta_step
                    cur_delta = float(np.clip(cur_delta, dmin, dmax))
                    cur_threshold = self.detector._compute_threshold(self.detector._calib_lambdas, cur_delta)

        ######################################################################################################

        self.detector.delta = float(cur_delta)
        self.detector.threshold = float(cur_threshold)
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
        mean_aggregate_share = {
            key: float(np.mean(np.asarray(values, dtype=np.float32)))
            for key, values in aggregate_contribution_shares.items()
        }

        return TrajectoryDetectionResult(
            detector_name=self.name,
            step_scores=np.asarray(step_scores, dtype=np.float32),
            aggregate_scores=np.asarray(lamb, dtype=np.float32),
            thresholds=np.asarray(ths, dtype=np.float32),
            predictions=np.asarray(preds, dtype=np.int64),
            aux_scores=(
                None
                if "transition_error" not in component_scores
                else np.asarray(component_scores["transition_error"], dtype=np.float32)
            ),
            labels=labels_np,
            metadata={
                "delta_final": float(self.detector.delta),
                "threshold_final": float(self.detector.threshold if self.detector.threshold is not None else np.nan),
                "score_scales": dict(self._score_scales),
                "weighted_step_contributions": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in weighted_component_scores.items()
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
                "mean_aggregate_share": dict(mean_aggregate_share),
                "feature_knn_scores": (
                    np.asarray(component_scores["feature_knn"], dtype=np.float32)
                    if "feature_knn" in component_scores
                    else None
                ),
                "transition_error_scores": (
                    np.asarray(component_scores["transition_error"], dtype=np.float32)
                    if "transition_error" in component_scores
                    else None
                ),
                "policy_chunk_scores": (
                    np.asarray(component_scores["policy_chunk"], dtype=np.float32)
                    if "policy_chunk" in component_scores
                    else None
                ),
                "neighbor_dynamics_scores": (
                    np.asarray(component_scores["neighbor_dynamics"], dtype=np.float32)
                    if "neighbor_dynamics" in component_scores
                    else None
                ),
                "feature_knn_mean": float(np.mean(component_scores["feature_knn"])) if "feature_knn" in component_scores else float("nan"),
                "transition_error_mean": float(np.mean(component_scores["transition_error"])) if "transition_error" in component_scores else float("nan"),
                "policy_chunk_mean": float(np.mean(component_scores["policy_chunk"])) if "policy_chunk" in component_scores else float("nan"),
                "neighbor_dynamics_mean": float(np.mean(component_scores["neighbor_dynamics"])) if "neighbor_dynamics" in component_scores else float("nan"),
            },
        )
