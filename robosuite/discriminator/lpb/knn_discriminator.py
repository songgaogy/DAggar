from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.discriminator.float.float_data import PolicyTrajectory
from robosuite.discriminator.lpb.model import DynamicsModel, DynamicsPredictor, Encoder


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
    """
    Compatibility loader for PyTorch >=2.6 (weights_only default changed to True).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions may not support `weights_only`.
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def knn_min_sqdist(
    query: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 8192,
    exclude_range: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Compute squared L2 nearest-neighbor distance from query to bank.

    Args:
        query: (B, D)
        bank: (N, D)
        exclude_range: optional [start, end) indices in bank to ignore.
    Returns:
        (B,) tensor of min squared distances.
    """
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
    Build (o, a) latent feature from LPB dynamics checkpoint.

    Feature definition (per timestep):
      feat_t = [obs_proj(h(o_t)), proprio_proj(s_t), mean(action_proj(a_{t:t+h-1}))].
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        action_horizon: int = -1,
        proprio_indices: Optional[Sequence[int]] = None,
        normalize_feature: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.proprio_indices = None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        self.normalize_feature = bool(normalize_feature)
        self.model, self.action_horizon, self.action_dim, self.proprio_dim = self._load_model(
            checkpoint_path=checkpoint_path,
            override_action_horizon=action_horizon,
        )

    def _load_model(
        self,
        checkpoint_path: str,
        override_action_horizon: int,
    ) -> tuple[DynamicsModel, int, int, int]:
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing key `model`: {checkpoint_path}")

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim", 512))
        action_dim = int(payload.get("action_dim", 7))
        proprio_dim = int(payload.get("proprio_dim", 0))
        horizon_ckpt = int(payload.get("horizon", 1))
        action_horizon = horizon_ckpt if int(override_action_horizon) <= 0 else int(override_action_horizon)

        d_model = int(_cfg_get(cfg, "model.d_model", 512))
        num_layers = int(_cfg_get(cfg, "model.num_layers", 6))
        num_heads = int(_cfg_get(cfg, "model.num_heads", 8))
        dropout = float(_cfg_get(cfg, "model.dropout", 0.1))
        max_action_horizon = int(_cfg_get(cfg, "model.max_action_horizon", action_horizon))
        fusion_hidden_dim = int(_cfg_get(cfg, "model.fusion_hidden_dim", d_model))
        projection_dim = int(_cfg_get(cfg, "model.projection_dim", 128))
        normalize_input = bool(_cfg_get(cfg, "encoder.normalize_input", True))

        encoder = Encoder(
            checkpoint_path=None,
            pretrained=False,
            freeze=True,
            normalize_input=normalize_input,
        )
        predictor = DynamicsPredictor(
            latent_dim=latent_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=num_heads,
            dropout=dropout,
            max_action_horizon=max(max_action_horizon, action_horizon),
            fusion_hidden_dim=fusion_hidden_dim,
        )
        model = DynamicsModel(
            encoder=encoder,
            predictor=predictor,
            projection_dim=projection_dim,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(self.device)
        model.eval()
        return model, action_horizon, action_dim, proprio_dim

    def _prepare_images(self, images: np.ndarray) -> torch.Tensor:
        # images: (T,H,W,3), uint8 or float
        img = images.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        chw = np.transpose(img, (0, 3, 1, 2))
        return torch.from_numpy(chw)

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

        h = self.action_horizon
        out = np.zeros((t_len, h, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + h)
            chunk = actions[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < h:
                pad = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad
        return out

    def _prepare_proprio(self, states: np.ndarray, t_len: int) -> np.ndarray:
        prop = np.asarray(states[:t_len], dtype=np.float32)
        if self.proprio_indices is not None and self.proprio_indices.size > 0:
            prop = prop[:, self.proprio_indices]
        if prop.shape[1] < self.proprio_dim:
            pad = np.zeros((prop.shape[0], self.proprio_dim - prop.shape[1]), dtype=np.float32)
            prop = np.concatenate([prop, pad], axis=1)
        elif prop.shape[1] > self.proprio_dim:
            prop = prop[:, : self.proprio_dim]
        return prop

    @torch.no_grad()
    def encode_trajectory(self, traj: PolicyTrajectory) -> torch.Tensor:
        t_len = int(traj.images.shape[0])
        if traj.actions is not None:
            t_len = min(t_len, int(traj.actions.shape[0]))
        t_len = min(t_len, int(traj.states.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")

        imgs = self._prepare_images(traj.images[:t_len])
        prop = self._prepare_proprio(traj.states, t_len=t_len)
        act_chunks = self._prepare_actions(traj.actions, t_len=t_len)

        prop_t = torch.from_numpy(prop)
        act_t = torch.from_numpy(act_chunks)

        feats = []
        for start in range(0, t_len, self.batch_size):
            end = min(start + self.batch_size, t_len)
            img_b = imgs[start:end].to(self.device)
            prop_b = prop_t[start:end].to(self.device)
            act_b = act_t[start:end].to(self.device)

            z = self.model.encode_observation(img_b)
            obs_tok = self.model.predictor.obs_proj(z)
            prop_tok = self.model.predictor.proprio_proj(prop_b)
            act_tok = self.model.predictor.action_proj(act_b).mean(dim=1)
            f = torch.cat([obs_tok, prop_tok, act_tok], dim=-1)
            if self.normalize_feature:
                f = F.normalize(f, p=2.0, dim=-1)
            feats.append(f.detach().cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_trajectory_with_transition_error(
        self,
        traj: PolicyTrajectory,
        proprio_error_weight: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            features: (T-H, D)
            transition_error: (T-H,)
        where H = action_horizon.
        """
        t_len = int(traj.images.shape[0])
        if traj.actions is not None:
            t_len = min(t_len, int(traj.actions.shape[0]))
        t_len = min(t_len, int(traj.states.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")

        h = int(self.action_horizon)
        valid_len = t_len - h
        if valid_len <= 0:
            raise ValueError(
                f"Trajectory length {t_len} is too short for action_horizon={h}"
            )

        imgs = self._prepare_images(traj.images[:t_len])
        prop = self._prepare_proprio(traj.states, t_len=t_len)
        act_chunks = self._prepare_actions(traj.actions, t_len=t_len)

        prop_t = torch.from_numpy(prop)
        act_t = torch.from_numpy(act_chunks)

        z_all = []
        feat_all = []
        for start in range(0, t_len, self.batch_size):
            end = min(start + self.batch_size, t_len)
            img_b = imgs[start:end].to(self.device)
            prop_b = prop_t[start:end].to(self.device)
            act_b = act_t[start:end].to(self.device)

            z = self.model.encode_observation(img_b)
            obs_tok = self.model.predictor.obs_proj(z)
            prop_tok = self.model.predictor.proprio_proj(prop_b)
            act_tok = self.model.predictor.action_proj(act_b).mean(dim=1)
            f = torch.cat([obs_tok, prop_tok, act_tok], dim=-1)
            if self.normalize_feature:
                f = F.normalize(f, p=2.0, dim=-1)

            z_all.append(z.detach().cpu())
            feat_all.append(f.detach().cpu())

        z_all_t = torch.cat(z_all, dim=0).to(self.device)
        feat_all_t = torch.cat(feat_all, dim=0)
        prop_all_t = prop_t.to(self.device)
        act_all_t = act_t.to(self.device)

        errs = []
        w_prop = float(proprio_error_weight)
        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            pred = self.model.predictor(
                obs_token=z_all_t[start:end],
                proprio_token=prop_all_t[start:end],
                action_tokens=act_all_t[start:end],
            )
            tgt_z = z_all_t[start + h : end + h]
            tgt_p = prop_all_t[start + h : end + h]

            z_err = (pred["pred_latent"] - tgt_z).pow(2).mean(dim=-1)
            p_err = (pred["pred_proprio"] - tgt_p).pow(2).mean(dim=-1)
            e = z_err + w_prop * p_err
            errs.append(e.detach().cpu())

        err_t = torch.cat(errs, dim=0)
        return feat_all_t[:valid_len], err_t


class AdaptiveKNNDiscriminator:
    """
    KNN OOD / failure detector with adaptive percentile threshold.
    """

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
            raise ValueError("lambda_window_size must be -1 (full prefix) or >=1")

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

        w = int(self.lambda_window_size)
        full_prefix = w <= 0

        if self.lambda_mode == "mean":
            if full_prefix:
                csum = np.cumsum(vals, dtype=np.float64)
                denom = np.arange(1, n + 1, dtype=np.float64)
                return (csum / denom).astype(np.float32)

            csum = np.cumsum(vals, dtype=np.float64)
            idx = np.arange(n, dtype=np.int64)
            start = np.maximum(0, idx - w + 1)
            start_minus = start - 1
            left = np.where(start_minus >= 0, csum[start_minus], 0.0)
            win_sum = csum - left
            denom = (idx - start + 1).astype(np.float64)
            return (win_sum / denom).astype(np.float32)

        # lambda_mode == "max"
        if full_prefix:
            return np.maximum.accumulate(vals)
        return self._rolling_max(vals, window=w)

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

        seqs = [x.detach().to(self.device, dtype=torch.float32) for x in expert_sequences if x.numel() > 0]
        if len(seqs) == 0:
            raise ValueError("All expert sequences are empty")

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
                raise ValueError(
                    f"calibration_aux length mismatch: {len(calibration_aux)} vs {len(seqs)}"
                )
            for i, seq in enumerate(seqs):
                ex = offsets[i]
                step = knn_min_sqdist(
                    query=seq,
                    bank=bank,
                    chunk_size=self.knn_chunk_size,
                    exclude_range=ex,
                )
                step_np = step.detach().cpu().numpy().astype(np.float32)
                combined = self._combine_scores(step_np, None if calibration_aux is None else np.asarray(calibration_aux[i], dtype=np.float32))
                calib_lambdas.append(self._aggregate_lambda(combined))
        else:
            cseqs = [x.detach().to(self.device, dtype=torch.float32) for x in calibration_sequences if x.numel() > 0]
            if len(cseqs) == 0:
                raise ValueError("All calibration sequences are empty")
            if calibration_aux is not None and len(calibration_aux) != len(cseqs):
                raise ValueError(
                    f"calibration_aux length mismatch: {len(calibration_aux)} vs {len(cseqs)}"
                )
            for i, seq in enumerate(cseqs):
                step = knn_min_sqdist(
                    query=seq,
                    bank=bank,
                    chunk_size=self.knn_chunk_size,
                    exclude_range=None,
                )
                step_np = step.detach().cpu().numpy().astype(np.float32)
                combined = self._combine_scores(step_np, None if calibration_aux is None else np.asarray(calibration_aux[i], dtype=np.float32))
                calib_lambdas.append(self._aggregate_lambda(combined))

        self._calib_lambdas = np.concatenate(calib_lambdas, axis=0).astype(np.float32)
        self.threshold = self._compute_threshold(self._calib_lambdas, self.delta)
        return float(self.threshold)

    @torch.no_grad()
    def score_oa(self, feature: torch.Tensor) -> float:
        if self.bank is None:
            raise RuntimeError("Call fit(...) before score_oa(...)")
        f = feature.reshape(1, -1).to(self.device, dtype=torch.float32)
        s = knn_min_sqdist(f, self.bank, chunk_size=self.knn_chunk_size)
        return float(s.item())

    @torch.no_grad()
    def score_chunk(self, features: torch.Tensor) -> float:
        if self.bank is None:
            raise RuntimeError("Call fit(...) before score_chunk(...)")
        f = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)
        step = knn_min_sqdist(f, self.bank, chunk_size=self.knn_chunk_size).detach().cpu().numpy().astype(np.float32)
        return float(self._aggregate_lambda(step)[-1])

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

        f = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)
        step_knn = knn_min_sqdist(f, self.bank, chunk_size=self.knn_chunk_size).detach().cpu().numpy().astype(np.float32)
        step = self._combine_scores(step_knn, aux_scores)
        lamb = self._aggregate_lambda(step)

        preds = np.zeros_like(lamb, dtype=np.int64)
        ths = np.zeros_like(lamb, dtype=np.float32)

        cur_delta = float(self.delta)
        cur_threshold = float(self.threshold)
        dmin = float(delta_min)
        dmax = float(delta_max)
        if dmin < 0.0:
            dmin = 0.0
        if dmax > 100.0:
            dmax = 100.0
        if dmin > dmax:
            dmin, dmax = dmax, dmin
        warmup = max(0, int(warmup_steps))
        upd_int = max(1, int(update_interval))

        for t in range(lamb.shape[0]):
            pred = int(lamb[t] >= cur_threshold)
            preds[t] = pred
            ths[t] = float(cur_threshold)

            if adaptive_delta and labels is not None:
                should_update = (t + 1) > warmup and ((t + 1 - warmup) % upd_int == 0)
                if should_update:
                    label = int(labels[t])
                    # Correct direction:
                    # - FN (label=1, pred=0): lower threshold => increase delta
                    # - FP (label=0, pred=1): raise threshold => decrease delta
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
