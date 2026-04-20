"""FAILDetectMonitor: core monitor for the logpZO variant of FAIL-Detect.

Stage 1 — density estimation: fit a normalizing flow on success-state features.
Stage 2 — threshold calibration: Conformal Prediction quantile over success
          step-level scores, giving a strict failure threshold tau.

Trigger rule at runtime: flag OOD if score(s_t) > tau, where
    score(s) = - log p_Z(f^{-1}(s))   (higher => more anomalous)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

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
        return {
            "alpha": float(self._alpha),
            "threshold": float(self._threshold),
            "num_calibration": int(s.shape[0]),
            "calib_score_mean": float(np.mean(s)),
            "calib_score_std": float(np.std(s)),
            "calib_score_min": float(np.min(s)),
            "calib_score_max": float(np.max(s)),
        }
