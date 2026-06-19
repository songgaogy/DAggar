"""Two-bank KNN OOD discriminator on top of the frozen dyn_disc encoder.

Per-frame score:

    d_succ(t) = min_j ||f(x_t) - s_j||_2     over success bank S
    d_fail(t) = min_k ||f(x_t) - r_k||_2     over failure bank R
    score(t)  = {
        difference : d_succ(t) - alpha * d_fail(t)
        ratio      : d_succ(t) / (d_succ(t) + d_fail(t) + eps)
        dsucc_only : d_succ(t)
    }

Failure prediction: ``pred_t = 1 iff score_t >= tau``.

Calibration of ``tau``:
  * ``success_percentile`` -- ``tau = percentile(success-calib scores, 100 - delta)``,
    mirroring ``SingleBankKNN.fit``. With ``score_mode='dsucc_only'`` and matching
    delta/calib_fraction, the result is bitwise-equivalent to ``SingleBankKNN``.
  * ``two_class_youden`` -- maximise TPR(tau) - FPR(tau) on a held-out frame
    set that combines success-calib (negatives) and fail-calib (positives).

This file deliberately mirrors the structure of ``SingleBankKNN`` rather than
subclassing it: bank handling is different enough that a subclass would
obscure the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from .single_bank_knn import DetectionResult, knn_min_l2_dist


_VALID_SCORE_MODES = ("difference", "ratio", "dsucc_only")
_VALID_CALIB_MODES = ("success_percentile", "two_class_youden")
_RATIO_EPS = 1e-8


@dataclass
class _CalibrationStats:
    threshold: float
    score_stats: dict  # min/max/mean of calibration scores
    method: str        # actually-used calib method (may differ from request if fallback)


class TwoBankKNN:
    """Two-bank KNN failure detector.

    The success bank ``S`` is built from expert trajectories (all frames). The
    failure bank ``R`` is built from frames at-or-after each failure trajectory's
    first_gt_failure_frame (handled by the caller; this class just receives the
    pre-selected feature tensors).
    """

    def __init__(
        self,
        visual_dim: int,
        proprio_dim: int,
        action_dim: int,
        *,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        action_weight: float = 1.0,
        alpha: float = 1.0,
        score_mode: str = "difference",
        delta: float = 10.0,
        calib_mode: str = "success_percentile",
        chunk_size: int = 2048,
        device: str = "cuda",
    ) -> None:
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if score_mode not in _VALID_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {_VALID_SCORE_MODES}, got {score_mode!r}")
        if calib_mode not in _VALID_CALIB_MODES:
            raise ValueError(f"calib_mode must be one of {_VALID_CALIB_MODES}, got {calib_mode!r}")

        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.visual_weight = float(visual_weight)
        self.proprio_weight = float(proprio_weight)
        self.action_weight = float(action_weight)
        self.alpha = float(alpha)
        self.score_mode = str(score_mode)
        self.delta = float(delta)
        self.calib_mode = str(calib_mode)
        self.chunk_size = int(chunk_size)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        self.success_bank: Optional[torch.Tensor] = None
        self.failure_bank: Optional[torch.Tensor] = None
        self.threshold: Optional[float] = None
        self._weights: Optional[torch.Tensor] = None
        self._calib_scores: Optional[np.ndarray] = None
        self._calib_method_used: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Per-block weighting                                                #
    # ------------------------------------------------------------------ #

    def _make_weights(self) -> torch.Tensor:
        if self._weights is not None:
            return self._weights
        w = torch.cat([
            torch.full((self.visual_dim,), self.visual_weight, device=self.device, dtype=torch.float32),
            torch.full((self.proprio_dim,), self.proprio_weight, device=self.device, dtype=torch.float32),
            torch.full((self.action_dim,), self.action_weight, device=self.device, dtype=torch.float32),
        ])
        self._weights = w
        return w

    def _apply_weights(self, feats: torch.Tensor) -> torch.Tensor:
        w = self._make_weights()
        if feats.shape[-1] != w.shape[0]:
            raise ValueError(
                f"feature dim mismatch: feat.shape[-1]={feats.shape[-1]}, "
                f"expected visual_dim+proprio_dim+action_dim={w.shape[0]}"
            )
        return feats * w.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # Distance + score primitives                                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _d_succ(self, feats_w: torch.Tensor) -> torch.Tensor:
        assert self.success_bank is not None
        return knn_min_l2_dist(feats_w, self.success_bank, chunk_size=self.chunk_size)

    @torch.no_grad()
    def _d_fail(self, feats_w: torch.Tensor) -> torch.Tensor:
        assert self.failure_bank is not None
        return knn_min_l2_dist(feats_w, self.failure_bank, chunk_size=self.chunk_size)

    @torch.no_grad()
    def _score_from_distances(
        self,
        d_succ: torch.Tensor,
        d_fail: torch.Tensor,
    ) -> torch.Tensor:
        if self.score_mode == "difference":
            return d_succ - self.alpha * d_fail
        if self.score_mode == "ratio":
            return d_succ / (d_succ + d_fail + _RATIO_EPS)
        # dsucc_only: pure single-bank baseline; failure bank distance unused
        return d_succ

    @torch.no_grad()
    def _score_features(self, features: torch.Tensor) -> np.ndarray:
        f = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)
        f_w = self._apply_weights(f)
        d_s = self._d_succ(f_w)
        if self.score_mode == "dsucc_only":
            # avoid the wasted failure-bank cdist when it cannot affect the score
            d_f = torch.zeros_like(d_s)
        else:
            d_f = self._d_fail(f_w)
        s = self._score_from_distances(d_s, d_f)
        return s.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # Fit                                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(
        self,
        expert_features: Sequence[torch.Tensor],
        fail_features: Sequence[torch.Tensor],
        success_calib_features: Sequence[torch.Tensor],
        fail_calib_features: Optional[Sequence[torch.Tensor]] = None,
    ) -> float:
        """Build success + failure banks, score calibration frames, calibrate ``tau``.

        Args:
            expert_features: list of (T_i, D) per-trajectory feature tensors (success bank).
            fail_features: list of (T_j, D) feature tensors for failure-bank trajectories
                (each already restricted to the last-K frames at/after first_gt_failure_frame).
            success_calib_features: held-out success trajectories (disjoint from expert_features).
            fail_calib_features: held-out failure trajectories for ``two_class_youden``; ignored
                otherwise.
        Returns:
            The calibrated threshold ``tau``.
        """
        s_seqs = [x.detach().to(self.device, dtype=torch.float32) for x in expert_features if x.numel() > 0]
        if not s_seqs:
            raise ValueError("expert_features (success bank) is empty")
        self.success_bank = self._apply_weights(torch.cat(s_seqs, dim=0))

        if self.score_mode != "dsucc_only":
            f_seqs = [x.detach().to(self.device, dtype=torch.float32) for x in fail_features if x.numel() > 0]
            if not f_seqs:
                raise ValueError("fail_features (failure bank) is empty")
            self.failure_bank = self._apply_weights(torch.cat(f_seqs, dim=0))
        else:
            # Build a zero-row placeholder so attribute is always set; never queried.
            self.failure_bank = None

        cs_seqs = [x.detach().to(self.device, dtype=torch.float32) for x in success_calib_features if x.numel() > 0]
        if not cs_seqs:
            raise ValueError("success_calib_features required for calibration")

        succ_calib_scores: List[np.ndarray] = []
        for seq in cs_seqs:
            succ_calib_scores.append(self._score_features(seq))
        succ_calib = np.concatenate(succ_calib_scores, axis=0)
        self._calib_scores = succ_calib

        method = self.calib_mode
        if method == "two_class_youden":
            if fail_calib_features is None or len(list(fail_calib_features)) == 0:
                print(
                    "[two_bank_knn] WARNING: calib_mode='two_class_youden' but no fail_calib_features "
                    "provided; falling back to 'success_percentile'."
                )
                method = "success_percentile"
            else:
                fail_calib_scores: List[np.ndarray] = []
                for seq in fail_calib_features:
                    if seq.numel() == 0:
                        continue
                    fail_calib_scores.append(self._score_features(seq))
                if not fail_calib_scores:
                    print(
                        "[two_bank_knn] WARNING: all fail_calib_features were empty; "
                        "falling back to 'success_percentile'."
                    )
                    method = "success_percentile"
                else:
                    fail_calib = np.concatenate(fail_calib_scores, axis=0)
                    self.threshold = float(_youden_threshold(succ_calib, fail_calib))

        if method == "success_percentile":
            # tau = percentile(values, 100 - delta).
            # delta=10 -> 90th percentile -> ~10% of calib frames flagged as failure.
            q = 100.0 * (1.0 - self.delta / 100.0)
            self.threshold = float(np.percentile(succ_calib.astype(np.float64), q=q))

        self._calib_method_used = method
        assert self.threshold is not None
        return self.threshold

    # ------------------------------------------------------------------ #
    # Score                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> DetectionResult:
        if self.success_bank is None or self.threshold is None:
            raise RuntimeError("Call fit(...) before score(...)")
        if self.score_mode != "dsucc_only" and self.failure_bank is None:
            raise RuntimeError("failure_bank is required for score_mode != 'dsucc_only'")
        s = self._score_features(features)
        ths = np.full_like(s, float(self.threshold), dtype=np.float32)
        preds = (s >= float(self.threshold)).astype(np.int64)
        return DetectionResult(step_scores=s, thresholds=ths, preds=preds)

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    def calib_summary(self) -> dict:
        if self._calib_scores is None or self.threshold is None:
            return {}
        arr = np.asarray(self._calib_scores, dtype=np.float64)
        return {
            "calib_method_used": str(self._calib_method_used or self.calib_mode),
            "threshold": float(self.threshold),
            "num_calib_frames": int(arr.size),
            "calib_score_min": float(arr.min()),
            "calib_score_max": float(arr.max()),
            "calib_score_mean": float(arr.mean()),
            "calib_score_std": float(arr.std()),
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "delta": self.delta,
            "success_bank_size": int(self.success_bank.shape[0]) if self.success_bank is not None else 0,
            "failure_bank_size": int(self.failure_bank.shape[0]) if self.failure_bank is not None else 0,
        }


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #


def _youden_threshold(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """Pick tau maximising J = TPR(tau) - FPR(tau) over the union of observed scores.

    Convention: prediction is positive (failure) iff score >= tau, matching `score`.
    """
    neg = np.asarray(neg_scores, dtype=np.float64).reshape(-1)
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    if neg.size == 0 or pos.size == 0:
        raise ValueError("Youden threshold requires non-empty neg and pos score arrays")

    # Candidate thresholds: midpoints between consecutive unique sorted scores,
    # plus the extremes; this covers every distinct decision boundary.
    all_scores = np.unique(np.concatenate([neg, pos], axis=0))
    if all_scores.size == 1:
        return float(all_scores[0])
    mids = 0.5 * (all_scores[:-1] + all_scores[1:])
    candidates = np.concatenate([[all_scores[0] - 1e-6], mids, [all_scores[-1] + 1e-6]], axis=0)

    npos = float(pos.size)
    nneg = float(neg.size)
    best_j = -np.inf
    best_tau = float(all_scores[0])
    for tau in candidates:
        tpr = float((pos >= tau).sum()) / npos
        fpr = float((neg >= tau).sum()) / nneg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_tau = float(tau)
    return best_tau
