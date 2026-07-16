"""Non-negative PU (nnPU) failure discriminator on the frozen dynamics latent.

This is the **no-GT-failure-timing** sibling of the GT-split BCE head. Instead of
slicing each failure trajectory at ``first_gt_failure_frame()`` into a clean
(success-like prefix, failure suffix), this head treats every failure-rollout
frame as **unlabeled** and learns a positive-vs-unlabeled classifier with the
non-negative PU risk estimator of Kiryo et al. (2017).

Labels
------
  * Positives (labeled, ``P``)  : all frames from SUCCESS trajectories.
  * Unlabeled (``U``)           : all frames from failure-rollout trajectories,
    taken as a WHOLE (no prefix/suffix split, no GT timing).

The unlabeled set is a mixture ``pi_p * P + (1 - pi_p) * N`` where ``pi_p`` is
the (unknown) fraction of "success-like" frames inside failure rollouts and is
supplied as the hyperparameter ``pi_p`` (class prior).

Head + score convention (mirrors the GT BCE head so the benchmark JSON layout is
unchanged):

    g(z)           = head(z)          # "success/positive-likeness" logit
    failure_score  = -g(z)            # step_scores; larger = more failure
    tau_task       = percentile(failure_score over success-calib frames, 100 - delta)
    pred_t = 1     iff failure_score_t >= tau_task

nnPU risk (logistic surrogate, see ``pu_risk`` for the exact form)
-----------------------------------------------------------------
    R_pu = pi_p * E_p[ ell(+1, g) ]
           + max( 0,  E_u[ ell(-1, g) ] - pi_p * E_p[ ell(-1, g) ] )

with the **non-negative correction** clamping the second (negative-risk) term at
0. We use the clamped variant; the canonical Kiryo nnPU additionally performs a
gradient-ascent step on ``-gamma * (negative-risk term)`` when it goes negative
(see ``pu_risk`` note). The approved surrogate is logistic loss,
``ell(y, g) = softplus(-y * g)``. The sigmoid surrogate remains available for
legacy compatibility.

**Hard constraint:** ``fit(...)`` does not run any evaluation or compute AUROC.
It trains for a fixed number of epochs against the nnPU objective and then
calibrates per-task thresholds on the disjoint success-calib split using the
success_percentile rule only (no failure labels are ever used).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .single_bank_knn import DetectionResult


# --------------------------------------------------------------------------- #
# Head                                                                        #
# --------------------------------------------------------------------------- #


class BCEHead(nn.Module):
    """MLP scalar-logit head on top of a frozen latent.

    Architecture (num_layers=3, hidden=512):
        Linear(in_dim, hidden) -> LayerNorm -> GELU
        Linear(hidden,  hidden) -> LayerNorm -> GELU
        Linear(hidden,  hidden) -> LayerNorm -> GELU
        Linear(hidden, 1)

    Reused verbatim from the GT-split BCE head so checkpoints / geometry match.
    """

    def __init__(self, in_dim: int, hidden: int = 512, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        layers: List[nn.Module] = []
        d = int(in_dim)
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(d, int(hidden)))
            layers.append(nn.LayerNorm(int(hidden)))
            layers.append(nn.GELU())
            d = int(hidden)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("logit_center", torch.zeros((), dtype=torch.float32))
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)

    def raw_forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D)  ->  (B,)
        return self.net(z).squeeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.raw_forward(z) - self.logit_center

    @torch.no_grad()
    def set_logit_center(self, value: float | torch.Tensor) -> None:
        center = torch.as_tensor(
            value,
            device=self.logit_center.device,
            dtype=self.logit_center.dtype,
        )
        if center.numel() != 1 or not bool(torch.isfinite(center).item()):
            raise ValueError("logit center must be one finite scalar")
        self.logit_center.copy_(center.reshape(()))


# --------------------------------------------------------------------------- #
# nnPU risk                                                                   #
# --------------------------------------------------------------------------- #


def _surrogate_loss(g: torch.Tensor, positive: bool, surrogate: str) -> torch.Tensor:
    """Per-sample surrogate loss ``ell(y, g)`` for ``y = +1`` (positive=True) or
    ``y = -1`` (positive=False).

    surrogate='sigmoid'  : ell(y, g) = sigmoid(-y * g)          (Kiryo nnPU default)
    surrogate='logistic' : ell(y, g) = softplus(-y * g) = log(1 + exp(-y*g))
    """
    y = 1.0 if positive else -1.0
    z = -y * g
    if surrogate == "sigmoid":
        return torch.sigmoid(z)
    if surrogate == "logistic":
        return torch.nn.functional.softplus(z)
    raise ValueError(f"unknown loss surrogate {surrogate!r}; expected 'sigmoid' or 'logistic'")


def soft_logit_cap_penalty(
    logits: torch.Tensor,
    cap: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the linear soft-hinge penalty outside ``[-cap, cap]``."""
    if float(cap) <= 0.0:
        raise ValueError(f"cap must be positive, got {cap}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if logits.numel() == 0:
        raise ValueError("soft logit cap requires at least one logit")
    t = torch.as_tensor(float(temperature), device=logits.device, dtype=logits.dtype)
    return (t * torch.nn.functional.softplus((logits.abs() - float(cap)) / t)).mean()


def pu_risk(
    g_p: torch.Tensor,
    g_u: torch.Tensor,
    *,
    pi_p: float,
    surrogate: str = "logistic",
    nn_correction: bool = True,
    beta: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute the (non-negative) PU risk and its components.

    Args:
        g_p: (Np,) head logits on labeled-positive frames.
        g_u: (Nu,) head logits on unlabeled frames.
        pi_p: class prior P(y=+1), the fraction of true positives in the
            unlabeled mixture.
        surrogate: 'logistic' (default) or legacy 'sigmoid'.
        nn_correction: when True, clamp the negative-risk term at ``-beta``
            (nnPU); when False, use the plain uPU estimator (may go negative).
        beta: lower clamp for the negative-risk term (Kiryo uses beta=0).

    Returns:
        dict with keys:
          'risk'          : scalar loss to minimize.
          'pos_risk'      : pi_p * E_p[ell(+1, g)].
          'neg_risk'      : E_u[ell(-1, g)] - pi_p * E_p[ell(-1, g)] (pre-clamp).
          'neg_risk_used' : the value actually added to 'risk' (post-clamp).

    Note (variant): the canonical Kiryo nnPU additionally replaces the gradient
    by ``-gamma * d(neg_risk)/d(theta)`` (gradient ascent) when ``neg_risk`` dips
    below ``-beta``. We implement the simpler clamped objective (no ascent step),
    which is the common practical default and keeps the loss a plain scalar.
    """
    pi = float(pi_p)
    # Positive-class risk (treat positives as +1).
    pos_risk = pi * _surrogate_loss(g_p, positive=True, surrogate=surrogate).mean()
    # Negative-class risk estimated via the PU identity:
    #   E_u[ell(-1)] = pi * E_p[ell(-1)] + (1-pi) * E_n[ell(-1)]
    #   => (1-pi) E_n[ell(-1)] = E_u[ell(-1)] - pi * E_p[ell(-1)]
    risk_u_neg = _surrogate_loss(g_u, positive=False, surrogate=surrogate).mean()
    risk_p_neg = _surrogate_loss(g_p, positive=False, surrogate=surrogate).mean()
    neg_risk = risk_u_neg - pi * risk_p_neg

    if nn_correction:
        neg_risk_used = torch.clamp(neg_risk, min=float(-beta))
    else:
        neg_risk_used = neg_risk

    risk = pos_risk + neg_risk_used
    return {
        "risk": risk,
        "pos_risk": pos_risk.detach(),
        "neg_risk": neg_risk.detach(),
        "neg_risk_used": neg_risk_used.detach(),
    }


# --------------------------------------------------------------------------- #
# Calibration stats                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class PUCalibStats:
    threshold: float
    num_calib_frames: int
    calib_score_min: float
    calib_score_max: float
    calib_score_mean: float
    calib_score_std: float


# --------------------------------------------------------------------------- #
# Discriminator                                                               #
# --------------------------------------------------------------------------- #


class PUBCEDiscriminator:
    """Shared-head nnPU discriminator with per-task threshold table.

    Workflow:
        det = PUBCEDiscriminator(in_dim, ...)
        det.fit(positive_features, unlabeled_features, success_calib_per_task,
                pi_p=..., epochs=..., lr=..., delta=..., seed=...)
        det.score(features, task) -> DetectionResult
    """

    def __init__(
        self,
        in_dim: int,
        *,
        hidden: int = 512,
        num_layers: int = 3,
        device: str = "cuda",
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PUBCEDiscriminator requires CUDA, but CUDA is unavailable")
        with torch.device(self.device):
            self.head = BCEHead(
                in_dim=self.in_dim,
                hidden=self.hidden,
                num_layers=self.num_layers,
            )
        self.thresholds: Dict[str, float] = {}
        self.calib_stats: Dict[str, PUCalibStats] = {}
        self._delta: Optional[float] = None
        self._pi_p: Optional[float] = None
        self._surrogate: str = "logistic"
        self._train_history: List[Dict[str, Any]] = []
        self._threshold_normalization: str = "none"
        self._soft_cap_c: Optional[float] = 5.0
        self._soft_cap_lambda: float = 1e-2
        self._soft_cap_temperature: float = 1.0
        self._scheduler_horizon_epochs: int = 20
        self._completed_epochs: int = 0

    # ------------------------------------------------------------------ #
    # Internal forward helpers                                           #
    # ------------------------------------------------------------------ #

    def logits_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Return success-likeness logits while preserving tensor/device flow."""
        if features.shape[-1] != self.in_dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.in_dim}, got {features.shape[-1]}"
            )
        leading_shape = features.shape[:-1]
        flat = features.reshape(-1, self.in_dim).to(self.device, dtype=torch.float32)
        logits = self.head(flat)
        return logits.reshape(leading_shape)

    def raw_logits_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Return the pre-normalization head output while preserving device flow."""
        if features.shape[-1] != self.in_dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.in_dim}, got {features.shape[-1]}"
            )
        leading_shape = features.shape[:-1]
        flat = features.reshape(-1, self.in_dim).to(self.device, dtype=torch.float32)
        return self.head.raw_forward(flat).reshape(leading_shape)

    def failure_score_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``-g(z)``; larger values are more failure-like."""
        return -self.logits_tensor(features)

    # Short aliases keep tensor inference ergonomic without changing the
    # existing numpy ``score(...)`` benchmark API.
    logits = logits_tensor
    failure_scores = failure_score_tensor

    def threshold_tensor(self, features: torch.Tensor, task: str) -> torch.Tensor:
        """Return the calibrated task threshold on ``features``' output device."""
        if task not in self.thresholds:
            raise KeyError(
                f"Task {task!r} has no calibrated threshold. "
                f"Available: {sorted(self.thresholds)}"
            )
        return torch.as_tensor(
            self.thresholds[task],
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def _logits_np(self, features: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
        """Run the head in eval mode and return (N,) numpy g(z) values."""
        self.head.eval()
        if features.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        f = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)
        outs: List[np.ndarray] = []
        for start in range(0, f.shape[0], batch_size):
            chunk = f[start : start + batch_size]
            g = self.logits_tensor(chunk)
            outs.append(g.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def _pool_logits_cuda(
        self,
        features: Sequence[torch.Tensor],
        *,
        raw: bool,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        was_training = self.head.training
        self.head.eval()
        for sequence in features:
            flat = sequence.reshape(-1, sequence.shape[-1])
            for start in range(0, int(flat.shape[0]), int(batch_size)):
                chunk = flat[start : start + int(batch_size)].to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                value = self.head.raw_forward(chunk) if raw else self.head(chunk)
                outputs.append(value.detach())
        if was_training:
            self.head.train()
        if not outputs:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _logit_summary(values: torch.Tensor) -> Dict[str, float]:
        if values.device.type != "cuda":
            raise ValueError("logit diagnostics must run on CUDA")
        flat = values.reshape(-1).to(torch.float32)
        if flat.numel() == 0:
            raise ValueError("cannot summarize an empty logit pool")
        finite = torch.isfinite(flat)
        if not bool(finite.all().item()):
            raise FloatingPointError("non-finite logits detected")
        quantiles = torch.quantile(
            flat,
            torch.tensor([0.01, 0.5, 0.99], device=flat.device, dtype=flat.dtype),
        )
        abs_p99 = torch.quantile(flat.abs(), 0.99)
        saturation = (flat.abs() > 9.21).to(torch.float32).mean()
        return {
            "num_frames": float(flat.numel()),
            "all_finite": True,
            "finite_fraction": 1.0,
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "p01": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "abs_p99": float(abs_p99.item()),
            "max_abs": float(flat.abs().max().item()),
            "saturation_fraction": float(saturation.item()),
        }

    @torch.no_grad()
    def _calibrate_epoch(
        self,
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        delta: float,
        verbose: bool,
    ) -> None:
        q = 1.0 - float(delta) / 100.0
        self.thresholds = {}
        self.calib_stats = {}
        for task, calib_sequences in success_calib_per_task.items():
            sequences = [value for value in calib_sequences if value.numel() > 0]
            if not sequences:
                raise ValueError(
                    f"Task {task!r}: success_calib_per_task[{task!r}] is empty; "
                    "cannot calibrate threshold."
                )
            effective = self._pool_logits_cuda(sequences, raw=False)
            failure_score = -effective
            tau_tensor = torch.quantile(failure_score, q)
            tau = float(tau_tensor.item())
            self.thresholds[task] = tau
            self.calib_stats[task] = PUCalibStats(
                threshold=tau,
                num_calib_frames=int(failure_score.numel()),
                calib_score_min=float(failure_score.min().item()),
                calib_score_max=float(failure_score.max().item()),
                calib_score_mean=float(failure_score.mean().item()),
                calib_score_std=float(failure_score.std(unbiased=False).item()),
            )
            if verbose:
                print(
                    f"[pu_bce][calib] task={task} tau={tau:.5f} "
                    f"center={float(self.head.logit_center.item()):.5f} "
                    f"n_calib={failure_score.numel()} "
                    f"mean={failure_score.mean().item():.5f} "
                    f"std={failure_score.std(unbiased=False).item():.5f}",
                    flush=True,
                )

    # ------------------------------------------------------------------ #
    # Fit                                                                #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        positive_features: Sequence[torch.Tensor],
        unlabeled_features: Sequence[torch.Tensor],
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        pi_p: float = 0.3,
        epochs: int = 1,
        scheduler_horizon_epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        delta: float = 10.0,
        seed: int = 0,
        loss_surrogate: str = "logistic",
        nn_correction: bool = True,
        beta: float = 0.0,
        soft_cap_c: Optional[float] = 5.0,
        soft_cap_lambda: float = 1e-2,
        soft_cap_temperature: float = 1.0,
        metric_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train the shared head with the nnPU risk; calibrate per-task thresholds.

        Args:
            positive_features: list of (T_i, D) tensors. Pooled into ``P``
                (frames from success trajectories).
            unlabeled_features: list of (T_j, D) tensors. Pooled into ``U``
                (frames from WHOLE failure trajectories; no GT split).
            success_calib_per_task: ``{task_name: [(T_k, D), ...]}`` of disjoint
                success-calib trajectories per task. Threshold is calibrated
                per-task on these via the success_percentile rule.
            pi_p: class prior P(y=+1) for the unlabeled mixture.
            epochs: fixed epoch budget. No early stopping, no validation eval.
            scheduler_horizon_epochs: cosine schedule horizon. The approved
                one-epoch fit retains the 20-epoch sweep schedule at epoch 1.
            delta: percentile-based false-alarm budget in [0, 100]. ``tau =
                percentile(failure_score on success-calib, 100 - delta)``.
            loss_surrogate: 'logistic' (approved default) or legacy 'sigmoid'.
            nn_correction: enable the non-negative correction (clamp neg-risk).
            beta: lower clamp for the negative-risk term (Kiryo default 0).
        Returns:
            Per-task threshold dict.
        """
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if scheduler_horizon_epochs < epochs:
            raise ValueError(
                "scheduler_horizon_epochs must be >= epochs, got "
                f"{scheduler_horizon_epochs} < {epochs}"
            )
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if not (0.0 < float(pi_p) < 1.0):
            raise ValueError(f"pi_p (class prior) must be in (0, 1), got {pi_p}")
        if loss_surrogate not in ("sigmoid", "logistic"):
            raise ValueError(f"loss_surrogate must be 'sigmoid' or 'logistic', got {loss_surrogate!r}")
        if float(soft_cap_lambda) < 0.0:
            raise ValueError("soft_cap_lambda must be non-negative")
        if float(soft_cap_lambda) > 0.0 and (soft_cap_c is None or float(soft_cap_c) <= 0.0):
            raise ValueError("positive soft_cap_lambda requires positive soft_cap_c")
        if float(soft_cap_temperature) <= 0.0:
            raise ValueError("soft_cap_temperature must be positive")

        self._delta = float(delta)
        self._pi_p = float(pi_p)
        self._surrogate = str(loss_surrogate)
        self._threshold_normalization = "none"
        self._soft_cap_c = None if soft_cap_c is None else float(soft_cap_c)
        self._soft_cap_lambda = float(soft_cap_lambda)
        self._soft_cap_temperature = float(soft_cap_temperature)
        self._scheduler_horizon_epochs = int(scheduler_horizon_epochs)
        self._completed_epochs = 0

        if abs(float(pi_p) - 0.5) < 1e-9 and verbose:
            print(
                "[pu_bce][fit] WARNING: pi_p=0.5 differs from the approved 0.3 "
                "default.",
                flush=True,
            )

        # ------- assemble pooled (P, U) frames ----------
        p_seqs = [t.detach() for t in positive_features if t.numel() > 0]
        u_seqs = [t.detach() for t in unlabeled_features if t.numel() > 0]
        if not p_seqs:
            raise ValueError("positive_features is empty after filtering")
        if not u_seqs:
            raise ValueError("unlabeled_features is empty after filtering")

        Z_p = torch.cat([t.to(torch.float32).reshape(-1, t.shape[-1]) for t in p_seqs], dim=0)
        Z_u = torch.cat([t.to(torch.float32).reshape(-1, t.shape[-1]) for t in u_seqs], dim=0)
        if Z_p.shape[1] != self.in_dim or Z_u.shape[1] != self.in_dim:
            raise ValueError(
                f"feature dim mismatch: in_dim={self.in_dim}, "
                f"P={Z_p.shape[1]}, U={Z_u.shape[1]}"
            )
        Np = int(Z_p.shape[0])
        Nu = int(Z_u.shape[0])

        # group=0 -> positive, group=1 -> unlabeled
        Z = torch.cat([Z_p, Z_u], dim=0)
        grp = torch.cat([
            torch.zeros(Np, dtype=torch.int64),
            torch.ones(Nu, dtype=torch.int64),
        ], dim=0)

        # ------- balanced sampling: P and U each carry equal total weight ----------
        # Without this, mini-batches can contain only-positive or only-unlabeled
        # samples, which makes the nnPU empirical risk components ill-defined.
        w_p = 1.0 / float(Np)
        w_u = 1.0 / float(Nu)
        weights = torch.cat([
            torch.full((Np,), w_p, dtype=torch.float64),
            torch.full((Nu,), w_u, dtype=torch.float64),
        ], dim=0)
        num_samples = int(2 * (Np + Nu))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        dataset = TensorDataset(Z, grp)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        # ------- deterministic head init + optim & schedule ----------
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        with torch.device(self.device):
            self.head = BCEHead(
                in_dim=self.in_dim, hidden=self.hidden, num_layers=self.num_layers,
            )

        optim = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        total_steps = int(scheduler_horizon_epochs) * max(1, len(loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=0.0
        )

        calib_sequences = [
            sequence
            for sequences in success_calib_per_task.values()
            for sequence in sequences
            if sequence.numel() > 0
        ]

        # ------- train loop (NO benchmark eval, NO AUROC) ----------
        self.head.train()
        self._train_history = []
        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            epoch_risk = 0.0
            epoch_cap = 0.0
            epoch_neg = 0.0
            n_correct = 0  # batches where the nn-correction fired (neg_risk < -beta)
            n_batches = 0
            for z_batch, grp_batch in loader:
                z_batch = z_batch.to(self.device, non_blocking=True)
                grp_batch = grp_batch.to(self.device, non_blocking=True)
                p_mask = grp_batch == 0
                u_mask = grp_batch == 1
                # Skip degenerate batches lacking one of the two groups.
                if int(p_mask.sum()) == 0 or int(u_mask.sum()) == 0:
                    continue
                g = self.head(z_batch)
                g_p = g[p_mask]
                g_u = g[u_mask]
                parts = pu_risk(
                    g_p, g_u,
                    pi_p=float(pi_p),
                    surrogate=self._surrogate,
                    nn_correction=bool(nn_correction),
                    beta=float(beta),
                )
                if self._soft_cap_lambda > 0.0:
                    cap_penalty = soft_logit_cap_penalty(
                        g,
                        cap=float(self._soft_cap_c),
                        temperature=self._soft_cap_temperature,
                    )
                else:
                    cap_penalty = torch.zeros((), device=g.device, dtype=g.dtype)
                loss = parts["risk"] + self._soft_cap_lambda * cap_penalty
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={epoch + 1}, batch={n_batches + 1}"
                    )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                scheduler.step()
                epoch_loss += float(loss.detach().item())
                epoch_risk += float(parts["risk"].detach().item())
                epoch_cap += float(cap_penalty.detach().item())
                epoch_neg += float(parts["neg_risk"].item())
                if float(parts["neg_risk"].item()) < float(-beta):
                    n_correct += 1
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            avg_risk = epoch_risk / max(1, n_batches)
            avg_cap = epoch_cap / max(1, n_batches)
            avg_neg = epoch_neg / max(1, n_batches)
            cur_lr = float(optim.param_groups[0]["lr"])
            self._calibrate_epoch(
                success_calib_per_task,
                delta=float(delta),
                verbose=False,
            )
            pool_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
            for pool_name, sequences in (
                ("train_positive", p_seqs),
                ("unlabeled_failure", u_seqs),
                ("success_calib", calib_sequences),
            ):
                pool_stats[pool_name] = {
                    "raw": self._logit_summary(self._pool_logits_cuda(sequences, raw=True)),
                    "effective": self._logit_summary(self._pool_logits_cuda(sequences, raw=False)),
                }
            entry: Dict[str, Any] = {
                "epoch": int(epoch + 1),
                "loss": avg,
                "nnpu_risk": avg_risk,
                "soft_cap_penalty": avg_cap,
                "soft_cap_weighted": self._soft_cap_lambda * avg_cap,
                "neg_risk": avg_neg,
                "nn_correction_batches": int(n_correct),
                "num_batches": int(n_batches),
                "nn_correction_fraction": float(n_correct) / float(max(1, n_batches)),
                "lr": cur_lr,
                "scheduler_horizon_epochs": int(self._scheduler_horizon_epochs),
                "logit_center": float(self.head.logit_center.item()),
                "thresholds": dict(self.thresholds),
                "pools": pool_stats,
            }
            self._train_history.append(entry)
            self._completed_epochs = int(epoch + 1)
            if metric_callback is not None:
                metric_callback(int(epoch + 1), entry)
            if verbose:
                print(
                    f"[pu_bce][fit] epoch={epoch + 1}/{int(epochs)} "
                    f"loss={avg:.5f} risk={avg_risk:.5f} cap={avg_cap:.5f} "
                    f"neg_risk={avg_neg:+.5f} center={self.head.logit_center.item():+.5f} "
                    f"nn_corr_batches={n_correct}/{n_batches} "
                    f"lr={cur_lr:.2e} Np={Np} Nu={Nu} pi_p={float(pi_p):.3f}",
                    flush=True,
                )
        self.head.eval()

        return dict(self.thresholds)

    # ------------------------------------------------------------------ #
    # Score                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score(self, features: torch.Tensor, task: str) -> DetectionResult:
        """Per-frame failure score + binary prediction for one task.

        Returns:
            DetectionResult with:
              step_scores = -g(z)               (higher = more failure)
              thresholds  = full tau_task array (constant)
              preds       = (step_scores >= tau_task).astype(int64)
        """
        if not self.thresholds:
            raise RuntimeError("Call fit(...) before score(...)")
        if task not in self.thresholds:
            raise KeyError(
                f"Task {task!r} has no calibrated threshold. "
                f"Available: {sorted(self.thresholds)}"
            )
        tau = float(self.thresholds[task])
        g = self._logits_np(features)
        failure_score = (-g).astype(np.float32)
        thresholds = np.full_like(failure_score, tau, dtype=np.float32)
        preds = (failure_score >= tau).astype(np.int64)
        return DetectionResult(step_scores=failure_score, thresholds=thresholds, preds=preds)

    # ------------------------------------------------------------------ #
    # Checkpoint                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> Dict[str, object]:
        return {
            "head": self.head.state_dict(),
            "in_dim": int(self.in_dim),
            "hidden": int(self.hidden),
            "num_layers": int(self.num_layers),
            "thresholds": {str(k): float(v) for k, v in self.thresholds.items()},
            "delta": None if self._delta is None else float(self._delta),
            "pi_p": None if self._pi_p is None else float(self._pi_p),
            "loss_surrogate": str(self._surrogate),
            "threshold_normalization": str(self._threshold_normalization),
            "soft_cap_c": self._soft_cap_c,
            "soft_cap_lambda": float(self._soft_cap_lambda),
            "soft_cap_temperature": float(self._soft_cap_temperature),
            "scheduler_horizon_epochs": int(self._scheduler_horizon_epochs),
            "completed_epochs": int(self._completed_epochs),
            "calib_stats": {
                str(k): {
                    "threshold": float(v.threshold),
                    "num_calib_frames": int(v.num_calib_frames),
                    "calib_score_min": float(v.calib_score_min),
                    "calib_score_max": float(v.calib_score_max),
                    "calib_score_mean": float(v.calib_score_mean),
                    "calib_score_std": float(v.calib_score_std),
                }
                for k, v in self.calib_stats.items()
            },
            "train_history": list(self._train_history),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        in_dim = int(state["in_dim"])  # type: ignore[arg-type]
        hidden = int(state["hidden"])  # type: ignore[arg-type]
        num_layers = int(state["num_layers"])  # type: ignore[arg-type]
        if (in_dim, hidden, num_layers) != (self.in_dim, self.hidden, self.num_layers):
            self.in_dim = in_dim
            self.hidden = hidden
            self.num_layers = num_layers
            with torch.device(self.device):
                self.head = BCEHead(in_dim=in_dim, hidden=hidden, num_layers=num_layers)
        self.head.set_logit_center(0.0)
        incompatible = self.head.load_state_dict(state["head"], strict=False)  # type: ignore[arg-type]
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing - {"logit_center"} or unexpected:
            raise RuntimeError(
                "Incompatible PU-BCE head state: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        self.thresholds = {str(k): float(v) for k, v in dict(state.get("thresholds", {})).items()}  # type: ignore[arg-type]
        self._delta = None if state.get("delta") is None else float(state["delta"])  # type: ignore[arg-type]
        self._pi_p = None if state.get("pi_p") is None else float(state["pi_p"])  # type: ignore[arg-type]
        self._surrogate = str(state.get("loss_surrogate", "sigmoid"))
        self._threshold_normalization = str(state.get("threshold_normalization", "none"))
        raw_cap = state.get("soft_cap_c")
        self._soft_cap_c = None if raw_cap is None else float(raw_cap)  # type: ignore[arg-type]
        self._soft_cap_lambda = float(state.get("soft_cap_lambda", 0.0))  # type: ignore[arg-type]
        self._soft_cap_temperature = float(state.get("soft_cap_temperature", 1.0))  # type: ignore[arg-type]
        self._scheduler_horizon_epochs = int(state.get("scheduler_horizon_epochs", 20))  # type: ignore[arg-type]
        self._completed_epochs = int(state.get("completed_epochs", 0))  # type: ignore[arg-type]
        cs = state.get("calib_stats", {}) or {}
        self.calib_stats = {
            str(k): PUCalibStats(
                threshold=float(v["threshold"]),
                num_calib_frames=int(v["num_calib_frames"]),
                calib_score_min=float(v["calib_score_min"]),
                calib_score_max=float(v["calib_score_max"]),
                calib_score_mean=float(v["calib_score_mean"]),
                calib_score_std=float(v["calib_score_std"]),
            )
            for k, v in dict(cs).items()  # type: ignore[arg-type]
        }
        self._train_history = list(state.get("train_history", []))  # type: ignore[arg-type]
