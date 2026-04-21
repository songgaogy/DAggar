"""D3Detector: pooled success + F3-cleaned fail KNN density-ratio scoring.

Structural mirror of ``lpb.knn_discriminator.AdaptiveKNNDiscriminator`` but
split into two phases so callers can share one bank across tasks while still
calibrating a per-task percentile threshold ``tau``:

    fit_banks(pos_sequences, neg_sequences)   # build D+ and F3-cleaned D-
    calibrate(calib_sequences) -> tau         # per-task percentile of lambdas
    score(features, tau=tau)                  # evaluate on a query trajectory

Per-step score (IRL-optimal reward style):

    lambda_t = (1 + omega) * d^2(phi_t, D+) - omega * d^2_weighted(phi_t, D-)

When ``omega == 0`` the neg branch is skipped entirely and the detector
reduces to LPB's (1-NN sq-distance on unnormalized features).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .filter import compute_f3_weights, knn_sqdist, weighted_knn_sqdist


_LOG_EPS = 1e-8


def _resolve_device(device: str) -> torch.device:
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _percentile(values: np.ndarray, delta: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute percentile from empty array")
    q = 100.0 * (1.0 - float(delta) / 100.0)
    return float(np.percentile(values.astype(np.float64), q=q))


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


def _aggregate_lambda(step_scores: np.ndarray, mode: str, window: int) -> np.ndarray:
    """Prefix/rolling aggregation of the per-step score, matching LPB semantics."""
    vals = step_scores.astype(np.float32, copy=False)
    n = vals.shape[0]
    if n == 0:
        return vals
    w = int(window)
    full_prefix = w <= 0

    if mode == "mean":
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

    if mode == "max":
        if full_prefix:
            return np.maximum.accumulate(vals)
        return _rolling_max(vals, window=w)

    raise ValueError(f"lambda_mode must be 'mean' or 'max', got {mode!r}")


@dataclass
class DetectionResult:
    step_scores: np.ndarray   # (T,) raw per-step lambda
    lambda_values: np.ndarray # (T,) prefix-aggregated lambda
    thresholds: np.ndarray    # (T,) constant tau broadcast along time
    preds: np.ndarray         # (T,) int64 in {0, 1}
    d_pos_sq: np.ndarray      # (T,) per-step squared distance to D+
    d_neg_sq: Optional[np.ndarray]  # (T,) per-step weighted sq-dist to D- (None if omega==0)


class D3Detector:
    """Dichotomous density-ratio detector with F3 static soft-weight filter."""

    def __init__(
        self,
        *,
        omega: float = 0.5,
        k: int = 1,
        beta: Optional[float] = None,
        kappa: Optional[float] = None,
        sigma_sq: float = 0.5,
        delta: float = 10.0,
        knn_chunk_size: int = 8192,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        device: str = "cuda",
    ) -> None:
        if float(omega) < 0.0:
            raise ValueError(f"omega must be >= 0, got {omega}")
        if int(k) <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        if float(delta) < 0.0 or float(delta) > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if lambda_mode not in {"mean", "max"}:
            raise ValueError("lambda_mode must be 'mean' or 'max'")

        self.omega = float(omega)
        self.k = int(k)
        self._beta_req = None if beta is None else float(beta)
        self._kappa_req = None if kappa is None else float(kappa)
        self.sigma_sq = float(sigma_sq)
        self.delta = float(delta)
        self.knn_chunk_size = int(knn_chunk_size)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.device = _resolve_device(device)

        self.pos_bank: Optional[torch.Tensor] = None
        self.neg_bank: Optional[torch.Tensor] = None
        self._neg_log_weights: Optional[torch.Tensor] = None
        self.beta_used: Optional[float] = None
        self.kappa_used: Optional[float] = None
        self.num_pos_used: int = 0
        self.num_neg_used: int = 0

    # ------------------------------------------------------------------ #
    # Bank construction                                                  #
    # ------------------------------------------------------------------ #

    def _cat_sequences(self, sequences: Iterable[torch.Tensor]) -> torch.Tensor:
        tensors = [x.detach().to(self.device, dtype=torch.float32) for x in sequences if x.numel() > 0]
        if len(tensors) == 0:
            raise ValueError("Expected at least one non-empty sequence")
        return torch.cat(tensors, dim=0)

    @torch.no_grad()
    def fit_banks(
        self,
        pos_sequences: Sequence[torch.Tensor],
        neg_sequences: Optional[Sequence[torch.Tensor]] = None,
    ) -> None:
        """Build the pooled success bank and the F3-cleaned fail bank."""
        if len(pos_sequences) == 0:
            raise ValueError("pos_sequences must be non-empty")
        self.pos_bank = self._cat_sequences(pos_sequences)
        self.num_pos_used = int(self.pos_bank.shape[0])

        if self.omega <= 0.0 or not neg_sequences:
            self.neg_bank = None
            self._neg_log_weights = None
            self.beta_used = None
            self.kappa_used = None
            self.num_neg_used = 0
            return

        self.neg_bank = self._cat_sequences(neg_sequences)
        self.num_neg_used = int(self.neg_bank.shape[0])

        weights, beta_used, kappa_used = compute_f3_weights(
            fail_feats=self.neg_bank,
            pos_bank=self.pos_bank,
            beta=self._beta_req,
            kappa=self._kappa_req,
            k=self.k,
            chunk_size=self.knn_chunk_size,
        )
        self.beta_used = beta_used
        self.kappa_used = kappa_used
        # Precompute log(w + eps) so weighted_knn_sqdist can fold it as a
        # const per-bank penalty across all queries.
        self._neg_log_weights = torch.log(weights.clamp_min(_LOG_EPS))

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _step_score(self, features: torch.Tensor) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Returns (lambda_t, d_pos_sq, d_neg_sq_or_None) as numpy arrays."""
        if self.pos_bank is None:
            raise RuntimeError("Call fit_banks(...) before scoring")
        f = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)

        d_pos_sq = knn_sqdist(
            f, self.pos_bank, k=self.k, chunk_size=self.knn_chunk_size, exclude_self=False
        )

        if self.omega <= 0.0 or self.neg_bank is None:
            lam = d_pos_sq
            lam_np = lam.detach().cpu().numpy().astype(np.float32)
            return lam_np, d_pos_sq.detach().cpu().numpy().astype(np.float32), None

        d_neg_sq = weighted_knn_sqdist(
            f,
            self.neg_bank,
            self._neg_log_weights,
            sigma_sq=self.sigma_sq,
            k=self.k,
            chunk_size=self.knn_chunk_size,
        )
        lam = (1.0 + self.omega) * d_pos_sq - self.omega * d_neg_sq
        return (
            lam.detach().cpu().numpy().astype(np.float32),
            d_pos_sq.detach().cpu().numpy().astype(np.float32),
            d_neg_sq.detach().cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def calibrate(self, calib_sequences: Sequence[torch.Tensor]) -> float:
        """Compute tau = percentile of aggregated lambdas on held-out success frames."""
        if self.pos_bank is None:
            raise RuntimeError("Call fit_banks(...) before calibrate(...)")
        seqs = [x for x in calib_sequences if x.numel() > 0]
        if len(seqs) == 0:
            raise ValueError("calib_sequences must contain at least one non-empty trajectory")
        lambdas_all: list[np.ndarray] = []
        for seq in seqs:
            step, _, _ = self._step_score(seq)
            lambdas_all.append(_aggregate_lambda(step, self.lambda_mode, self.lambda_window_size))
        calib_concat = np.concatenate(lambdas_all, axis=0).astype(np.float32)
        return _percentile(calib_concat, self.delta)

    @torch.no_grad()
    def score(self, features: torch.Tensor, tau: float) -> DetectionResult:
        step, d_pos_sq, d_neg_sq = self._step_score(features)
        lam = _aggregate_lambda(step, self.lambda_mode, self.lambda_window_size)
        ths = np.full_like(lam, float(tau), dtype=np.float32)
        preds = (lam >= float(tau)).astype(np.int64)
        return DetectionResult(
            step_scores=step,
            lambda_values=lam,
            thresholds=ths,
            preds=preds,
            d_pos_sq=d_pos_sq,
            d_neg_sq=d_neg_sq,
        )
