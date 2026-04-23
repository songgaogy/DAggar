"""D4Detector: CFG-guided per-frame scoring + per-task conformal tau.

Per-step score (residual form, design §8.4 / §A.3):
    f_omega = (1 + omega) * f(.|c=+) - omega * f(.|c=-)
    lambda_t = || f_omega - z_{t+h} ||^2 / (2 * sigma^2)

This is rank-equivalent to the linear-log-density CFG score. Aggregation
via ``lambda_mode`` ("mean" or "max") over a prefix/rolling window uses the
exact same helper as ``D3Detector._aggregate_lambda`` (imported verbatim).
Conformal tau is the delta-percentile of aggregated lambdas on held-out
success frames (Proposition 10.1 of design).

CFG stability cap: omega is clamped to 2.0 at inference; see plan §7
(CFG divergence at large omega produces NaNs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.d3disc.detector import (
    DetectionResult,
    _aggregate_lambda,
    _percentile,
    _resolve_device,
)

from ..models.adaln import ConditionEmbedder
from ..models.dynamics import ConditionalDynamicsPredictor
from .feature import D4Frames


_OMEGA_MAX = 2.0
_VALID_SCORE_MODES = ("rel", "abs")


@dataclass
class D4StepOutput:
    step_scores: np.ndarray
    r_plus: np.ndarray
    r_minus: np.ndarray
    advantage: np.ndarray


class D4Detector:
    """CFG-guided D4-Disc scorer. One trained predictor serves all omega.

    score_mode:
        "rel" (default): lambda_t = (||f_omega - z_target||^2 - ||z_t - z_target||^2) / (2 sigma^2).
            With a residual latent head (pred = z_t + Delta), the raw abs residual
            is ||Delta - d||^2 where d = z_target - z_t. If the predictor is only
            a modest improvement over identity (Phase-A only, or early bootstrap),
            ||d||^2 (motion magnitude) dominates, so the score is confounded by
            per-task motion statistics (e.g. fail rollouts that stall have lower
            |d| than success rollouts, inverting AUROC). Subtracting the identity
            baseline isolates the model's delta-prediction quality and removes the
            motion confound.
        "abs": matches d4disc_0423.md §8.2 verbatim: lambda_t = ||f_omega - z_target||^2 / (2 sigma^2).
            Useful as a reference/ablation. Prefer "rel" unless the predictor has
            learned transitions to near-zero abs error (post full Phase-B).
    """

    def __init__(
        self,
        predictor: ConditionalDynamicsPredictor,
        *,
        omega: float = 0.5,
        sigma_sq: float = 0.5,
        delta: float = 10.0,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        device: str = "cuda",
        batch_size: int = 512,
        score_mode: str = "rel",
    ) -> None:
        if float(omega) < 0.0:
            raise ValueError(f"omega must be >= 0, got {omega}")
        if float(omega) > _OMEGA_MAX:
            print(f"[d4_detector] WARNING: omega={omega} > {_OMEGA_MAX}; clamping to {_OMEGA_MAX}")
        if float(delta) < 0.0 or float(delta) > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if lambda_mode not in {"mean", "max"}:
            raise ValueError("lambda_mode must be 'mean' or 'max'")
        if str(score_mode) not in _VALID_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {_VALID_SCORE_MODES}, got {score_mode!r}")

        self.predictor = predictor
        self.omega = float(min(max(float(omega), 0.0), _OMEGA_MAX))
        self.sigma_sq = float(sigma_sq)
        self.delta = float(delta)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.score_mode = str(score_mode)
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.predictor.to(self.device)
        self.predictor.eval()

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _score_frames(self, frames: D4Frames) -> D4StepOutput:
        T = int(frames.length)
        two_sigma_sq = 2.0 * float(self.sigma_sq)

        step = np.empty((T,), dtype=np.float32)
        rp = np.empty((T,), dtype=np.float32)
        rm = np.empty((T,), dtype=np.float32)
        adv = np.empty((T,), dtype=np.float32)

        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            B = end - start
            obs = frames.z_current[start:end].to(self.device, dtype=torch.float32)
            prop = frames.proprio[start:end].to(self.device, dtype=torch.float32)
            act = frames.action_chunks[start:end].to(self.device, dtype=torch.float32)
            tgt = frames.z_target[start:end].to(self.device, dtype=torch.float32)

            c_plus = torch.full((B,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=self.device)
            c_minus = torch.full((B,), ConditionEmbedder.COND_MINUS, dtype=torch.long, device=self.device)

            out_p = self.predictor(obs, prop, act, cond_idx=c_plus)
            out_m = self.predictor(obs, prop, act, cond_idx=c_minus)

            f_plus = out_p["pred_latent"]
            f_minus = out_m["pred_latent"]
            f_omega = (1.0 + self.omega) * f_plus - self.omega * f_minus
            lam = ((f_omega - tgt) ** 2).sum(dim=-1) / two_sigma_sq
            if self.score_mode == "rel":
                # Remove motion-magnitude confound; see class docstring.
                id_err = ((obs - tgt) ** 2).sum(dim=-1) / two_sigma_sq
                lam = lam - id_err
            r_plus = ((f_plus - tgt) ** 2).sum(dim=-1)
            r_minus = ((f_minus - tgt) ** 2).sum(dim=-1)
            advantage = (r_minus - r_plus) / two_sigma_sq

            step[start:end] = lam.detach().cpu().numpy().astype(np.float32)
            rp[start:end] = r_plus.detach().cpu().numpy().astype(np.float32)
            rm[start:end] = r_minus.detach().cpu().numpy().astype(np.float32)
            adv[start:end] = advantage.detach().cpu().numpy().astype(np.float32)

        return D4StepOutput(step_scores=step, r_plus=rp, r_minus=rm, advantage=adv)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def calibrate(self, calib_frames: Sequence[D4Frames]) -> float:
        if not calib_frames:
            raise ValueError("calib_frames must be non-empty")
        lambdas_all: list[np.ndarray] = []
        for fr in calib_frames:
            step = self._score_frames(fr)
            lambdas_all.append(
                _aggregate_lambda(step.step_scores, self.lambda_mode, self.lambda_window_size)
            )
        calib_concat = np.concatenate(lambdas_all, axis=0).astype(np.float32)
        return _percentile(calib_concat, self.delta)

    @torch.no_grad()
    def score(self, frames: D4Frames, tau: float) -> DetectionResult:
        step = self._score_frames(frames)
        lam = _aggregate_lambda(step.step_scores, self.lambda_mode, self.lambda_window_size)
        ths = np.full_like(lam, float(tau), dtype=np.float32)
        preds = (lam >= float(tau)).astype(np.int64)
        # Reuse DetectionResult; d_pos_sq := r_plus, d_neg_sq := r_minus for
        # downstream visualization compatibility.
        return DetectionResult(
            step_scores=step.step_scores,
            lambda_values=lam,
            thresholds=ths,
            preds=preds,
            d_pos_sq=step.r_plus,
            d_neg_sq=step.r_minus,
        )

    def score_advantage(self, frames: D4Frames) -> D4StepOutput:
        """Raw per-step (r_plus, r_minus, advantage, lambda); used by visualize."""
        return self._score_frames(frames)
