"""FAILDetectMonitor: core monitor for the logpZO variant of FAIL-Detect.

Stage 1 — density estimation: fit a normalizing flow on success-state features.
Stage 2 — threshold calibration: time-varying Conformal Prediction band over
          success step-level scores (functional CP, Diquigiovanni 2024 [14]),
          giving a per-step failure threshold eta_t (paper Sec. IV-B).

Trigger rule at runtime: flag OOD if score(s_t) > eta_t, where
    score(s)   = - log p_Z(f^{-1}(s))                 (higher => more anomalous)
    eta_t      = mu_t + eta * sigma_t                 (one-sided upper band)
    eta        = ceil((N+1)(1-alpha))/N quantile of   r_i = max_t (s_i_t - mu_t)/sigma_t
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .flow_model import RealNVPFlow


@dataclass
class FitStats:
    """Summary returned by `fit_score_model`."""

    epochs: int
    final_train_nll: float
    best_val_nll: float
    num_train: int
    num_val: int


class FAILDetectMonitor:
    """Runtime OOD monitor for imitation-learning policies (paper arXiv:2503.08558).

    Parameters
    ----------
    feature_dim
        Dimensionality of the input observation feature `s` (e.g. DINOv2 embedding).
    num_layers, hidden_dim
        Flow capacity knobs.
    device
        "cuda" / "cpu".
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_layers: int = 8,
        hidden_dim: int = 512,
        scale_clamp: float = 3.0,
        device: str = "cuda",
    ) -> None:
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        self.flow = RealNVPFlow(
            dim=self.feature_dim,
            num_layers=int(num_layers),
            hidden_dim=int(hidden_dim),
            scale_clamp=float(scale_clamp),
        ).to(self.device)

        self._threshold: Optional[float] = None
        self._calibration_scores: Optional[np.ndarray] = None
        self._alpha: Optional[float] = None
        self._fit_stats: Optional[FitStats] = None

        # Functional CP band (paper Sec. IV-B, Eq. eta_t = mu_t + eta * sigma_t).
        self._mu_t: Optional[np.ndarray] = None
        self._sigma_t: Optional[np.ndarray] = None
        self._eta: Optional[float] = None
        self._cp_band_calib_T: Optional[int] = None
        self._cp_band_num_calib: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Stage 1: density estimation                                        #
    # ------------------------------------------------------------------ #

    def fit_score_model(
        self,
        features: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_fraction: float = 0.1,
        early_stop_patience: int = 8,
        seed: int = 0,
        verbose: bool = True,
    ) -> FitStats:
        """Train the normalizing flow on `features` (N, D) from success data."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be (N, {self.feature_dim}); got {arr.shape}"
            )
        if arr.shape[0] < 10:
            raise ValueError(f"need >=10 training samples, got {arr.shape[0]}")

        # Standardize using training-set statistics.
        mean = torch.from_numpy(arr.mean(axis=0))
        std = torch.from_numpy(arr.std(axis=0) + 1e-6)
        self.flow.set_standardization(mean, std)

        # Train/val split.
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(arr.shape[0])
        n_val = max(1, int(round(float(val_fraction) * arr.shape[0])))
        n_val = min(n_val, arr.shape[0] - 1)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        x_train = torch.from_numpy(arr[train_idx])
        x_val = torch.from_numpy(arr[val_idx]).to(self.device)

        loader = DataLoader(
            TensorDataset(x_train),
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
        )
        optim = torch.optim.Adam(self.flow.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        best_val = math.inf
        best_state = {k: v.detach().clone() for k, v in self.flow.state_dict().items()}
        patience = 0
        last_train = math.nan

        self.flow.train()
        for ep in range(int(epochs)):
            total = 0.0
            n = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                loss = self.flow.nll(batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=5.0)
                optim.step()
                bs = int(batch.shape[0])
                total += float(loss.item()) * bs
                n += bs
            last_train = total / max(n, 1)

            self.flow.eval()
            with torch.no_grad():
                val_nll = float(self.flow.nll(x_val).item())
            self.flow.train()

            if verbose:
                print(f"[logpZO][fit] epoch={ep + 1}/{int(epochs)} train_nll={last_train:.4f} val_nll={val_nll:.4f}")

            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_state = {k: v.detach().clone() for k, v in self.flow.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= int(early_stop_patience):
                    if verbose:
                        print(f"[logpZO][fit] early stop at epoch {ep + 1}")
                    break

        self.flow.load_state_dict(best_state)
        self.flow.eval()

        self._fit_stats = FitStats(
            epochs=ep + 1,
            final_train_nll=float(last_train),
            best_val_nll=float(best_val),
            num_train=int(x_train.shape[0]),
            num_val=int(x_val.shape[0]),
        )
        return self._fit_stats

    # ------------------------------------------------------------------ #
    # Stage 2: conformal prediction threshold                            #
    # ------------------------------------------------------------------ #

    def calibrate_conformal_threshold(
        self,
        calibration_features: np.ndarray,
        *,
        alpha: float = 0.1,
    ) -> float:
        """Compute tau from success-only features via empirical quantile.

        Following Eq. from prompt:
            tau = Quantile(S, ceil((N+1)*(1-alpha))/N)
        where S = { score(s) } and score(s) = - log p_Z(f^{-1}(s)).
        """
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        arr = np.asarray(calibration_features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.feature_dim:
            raise ValueError(
                f"calibration_features must be (N, {self.feature_dim}); got {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("need at least one calibration sample")

        scores = self.score_features(arr)  # higher => more OOD
        n = int(scores.shape[0])
        k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
        k = max(1, min(n, k))
        # Sort ascending; take the k-th smallest (1-indexed) as the strict threshold.
        tau = float(np.partition(scores, k - 1)[k - 1])

        self._alpha = float(alpha)
        self._threshold = float(tau)
        self._calibration_scores = scores.astype(np.float64)
        return self._threshold

    # ------------------------------------------------------------------ #
    # Stage 2 (paper-faithful): functional CP band, time-varying eta_t   #
    # ------------------------------------------------------------------ #

    def calibrate_functional_cp_band(
        self,
        per_trajectory_features: Sequence[np.ndarray],
        *,
        alpha: float = 0.1,
        sigma_floor: float = 1e-3,
    ) -> None:
        """Time-varying one-sided CP band over success calibration trajectories.

        Implements FAIL-Detect Sec. IV-B (Diquigiovanni 2024 functional CP):
            mu_t   = mean_i s_i_t                                  per-step mean
            sigma_t= std_i s_i_t + sigma_floor                     per-step modulation
            r_i    = max_t (s_i_t - mu_t) / sigma_t                one-sided nonconformity
            eta    = ceil((N+1)(1-alpha))/N quantile of {r_i}      conformal multiplier
            eta_t  = mu_t + eta * sigma_t                          upper band

        Variable-length trajectories are right-padded with their final value to
        the longest calibration length; tests longer than that reuse the last
        eta_t (Sec. IV-B uses identically-shaped rollouts).
        """
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if len(per_trajectory_features) < 2:
            raise ValueError(
                f"functional CP needs >=2 calibration trajectories, got {len(per_trajectory_features)}"
            )

        per_traj_scores: list[np.ndarray] = []
        max_T = 0
        for feats in per_trajectory_features:
            arr = np.asarray(feats, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.feature_dim:
                raise ValueError(
                    f"calibration features must be (T, {self.feature_dim}); got {arr.shape}"
                )
            if arr.shape[0] <= 0:
                raise ValueError("calibration trajectory is empty")
            s = self.score_features(arr).astype(np.float64)
            per_traj_scores.append(s)
            if int(s.shape[0]) > max_T:
                max_T = int(s.shape[0])

        N = len(per_traj_scores)
        padded = np.empty((N, max_T), dtype=np.float64)
        for i, s in enumerate(per_traj_scores):
            T_i = int(s.shape[0])
            padded[i, :T_i] = s
            if T_i < max_T:
                padded[i, T_i:] = s[-1]

        mu_t = padded.mean(axis=0)
        sigma_t = padded.std(axis=0) + float(sigma_floor)

        deviations = (padded - mu_t[None, :]) / sigma_t[None, :]
        r = deviations.max(axis=1)  # (N,) one-sided trajectory-level nonconformity

        k = int(math.ceil((N + 1) * (1.0 - float(alpha))))
        k = max(1, min(N, k))
        eta = float(np.partition(r, k - 1)[k - 1])

        self._alpha = float(alpha)
        self._mu_t = mu_t.astype(np.float64)
        self._sigma_t = sigma_t.astype(np.float64)
        self._eta = float(eta)
        self._cp_band_calib_T = int(max_T)
        self._cp_band_num_calib = int(N)
        # Flatten padded scores for summary stats only.
        self._calibration_scores = padded.reshape(-1).astype(np.float64)
        # Mirror the band-mean to the legacy scalar slot so older callers still
        # see "a threshold" and stats remain populated.
        self._threshold = float((mu_t + eta * sigma_t).mean())

    def threshold_per_step(self, num_steps: int) -> np.ndarray:
        """Return eta_t over `num_steps` frames; extends with the last band value."""
        if self._mu_t is None or self._sigma_t is None or self._eta is None:
            raise RuntimeError(
                "functional CP band not calibrated; call calibrate_functional_cp_band first"
            )
        band = self._mu_t + self._eta * self._sigma_t
        if int(num_steps) <= int(band.shape[0]):
            return band[: int(num_steps)].astype(np.float64).copy()
        out = np.empty(int(num_steps), dtype=np.float64)
        out[: int(band.shape[0])] = band
        out[int(band.shape[0]) :] = band[-1]
        return out

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score_features(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Per-sample anomaly score: higher = more failure-like.

        score = - log p_Z(f^{-1}(s))   (logpZO variant)
        """
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be (N, {self.feature_dim}); got {arr.shape}"
            )
        self.flow.eval()
        outs: list[np.ndarray] = []
        for start in range(0, arr.shape[0], int(batch_size)):
            stop = min(arr.shape[0], start + int(batch_size))
            x = torch.from_numpy(arr[start:stop]).to(self.device)
            logpz = self.flow.log_pz_only(x)
            outs.append((-logpz).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @property
    def alpha(self) -> Optional[float]:
        return self._alpha

    @property
    def fit_stats(self) -> Optional[FitStats]:
        return self._fit_stats

    def calibration_summary(self) -> dict:
        if self._calibration_scores is None or self._threshold is None or self._alpha is None:
            return {}
        s = self._calibration_scores
        out = {
            "alpha": float(self._alpha),
            "threshold": float(self._threshold),
            "num_calibration": int(s.shape[0]),
            "calib_score_mean": float(np.mean(s)),
            "calib_score_std": float(np.std(s)),
            "calib_score_min": float(np.min(s)),
            "calib_score_max": float(np.max(s)),
        }
        if self._eta is not None and self._mu_t is not None and self._sigma_t is not None:
            band = self._mu_t + self._eta * self._sigma_t
            out.update({
                "cp_band_eta": float(self._eta),
                "cp_band_T": int(self._cp_band_calib_T or 0),
                "cp_band_num_calib_trajs": int(self._cp_band_num_calib or 0),
                "cp_band_threshold_mean": float(band.mean()),
                "cp_band_threshold_min": float(band.min()),
                "cp_band_threshold_max": float(band.max()),
            })
        return out
