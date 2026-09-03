"""Non-negative PU (nnPU) failure discriminator on the frozen RPT latent.

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

nnPU risk (sigmoid surrogate, see ``_pu_risk`` for the exact form)
------------------------------------------------------------------
    R_pu = pi_p * E_p[ ell(+1, g) ]
           + max( 0,  E_u[ ell(-1, g) ] - pi_p * E_p[ ell(-1, g) ] )

with the **non-negative correction** clamping the second (negative-risk) term at
0. We use the clamped variant; the canonical Kiryo nnPU additionally performs a
gradient-ascent step on ``-gamma * (negative-risk term)`` when it goes negative
(see ``_pu_risk`` note). ``ell`` is the sigmoid surrogate
``ell(y, g) = sigmoid(-y * g)`` (a.k.a. the "ramp"/sigmoid loss used in the
original nnPU paper). Logistic loss is available via ``loss_surrogate='logistic'``.

**Hard constraint:** ``fit(...)`` does not run any evaluation or compute AUROC.
It trains for a fixed number of epochs against the nnPU objective and then
calibrates per-task thresholds on the disjoint success-calib split using the
success_percentile rule only (no failure labels are ever used).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler



@dataclass
class DetectionResult:
    """Per-frame discriminator output."""

    step_scores: np.ndarray
    thresholds: np.ndarray
    preds: np.ndarray


# --------------------------------------------------------------------------- #
# Head                                                                        #
# --------------------------------------------------------------------------- #


class BCEHead(nn.Module):
    """MLP scalar-logit head on top of a frozen latent.

    Architecture (num_layers=2, hidden=256):
        Linear(in_dim, hidden) -> LayerNorm -> GELU
        Linear(hidden,  hidden) -> LayerNorm -> GELU
        Linear(hidden, 1)

    Reused verbatim from the GT-split BCE head so checkpoints / geometry match.
    """

    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 2) -> None:
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
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D)  ->  (B,)
        return self.net(z).squeeze(-1)


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


def pu_risk(
    g_p: torch.Tensor,
    g_u: torch.Tensor,
    *,
    pi_p: float,
    surrogate: str = "sigmoid",
    nn_correction: bool = True,
    beta: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute the (non-negative) PU risk and its components.

    Args:
        g_p: (Np,) head logits on labeled-positive frames.
        g_u: (Nu,) head logits on unlabeled frames.
        pi_p: class prior P(y=+1), the fraction of true positives in the
            unlabeled mixture.
        surrogate: 'sigmoid' (default) or 'logistic'.
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
        hidden: int = 256,
        num_layers: int = 2,
        device: str = "cuda",
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError(
                f"PUBCEDiscriminator is CUDA-only; got device={device!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for nnPU training and scoring")
        self.device = requested_device

        self.head: BCEHead = BCEHead(in_dim=self.in_dim, hidden=self.hidden, num_layers=self.num_layers).to(self.device)
        self.thresholds: Dict[str, float] = {}
        self.calib_stats: Dict[str, PUCalibStats] = {}
        self._delta: Optional[float] = None
        self._pi_p: Optional[float] = None
        self._surrogate: str = "sigmoid"
        self._train_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ #
    # Internal forward helpers                                           #
    # ------------------------------------------------------------------ #

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
            g = self.head(chunk)
            outs.append(g.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    # ------------------------------------------------------------------ #
    # Fit                                                                #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        positive_features: Sequence[torch.Tensor],
        unlabeled_features: Sequence[torch.Tensor],
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        pi_p: float = 0.5,
        epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        delta: float = 10.0,
        seed: int = 0,
        loss_surrogate: str = "sigmoid",
        nn_correction: bool = True,
        beta: float = 0.0,
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
            pi_p: class prior P(y=+1) for the unlabeled mixture. Defaults to 0.5
                with a logged warning -- it should be set from domain knowledge.
            epochs: fixed epoch budget. No early stopping, no validation eval.
            delta: percentile-based false-alarm budget in [0, 100]. ``tau =
                percentile(failure_score on success-calib, 100 - delta)``.
            loss_surrogate: 'sigmoid' (nnPU default) or 'logistic'.
            nn_correction: enable the non-negative correction (clamp neg-risk).
            beta: lower clamp for the negative-risk term (Kiryo default 0).
        Returns:
            Per-task threshold dict.
        """
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if not (0.0 < float(pi_p) < 1.0):
            raise ValueError(f"pi_p (class prior) must be in (0, 1), got {pi_p}")
        if loss_surrogate not in ("sigmoid", "logistic"):
            raise ValueError(f"loss_surrogate must be 'sigmoid' or 'logistic', got {loss_surrogate!r}")

        self._delta = float(delta)
        self._pi_p = float(pi_p)
        self._surrogate = str(loss_surrogate)

        if abs(float(pi_p) - 0.5) < 1e-9 and verbose:
            print(
                "[pu_bce][fit] WARNING: pi_p left at the 0.5 default. The class "
                "prior (fraction of success-like frames inside failure rollouts) "
                "should be set from domain knowledge via --pi-p.",
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
            pin_memory=not Z.is_cuda,
            drop_last=True,
        )

        # ------- deterministic head init + optim & schedule ----------
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        self.head = BCEHead(
            in_dim=self.in_dim, hidden=self.hidden, num_layers=self.num_layers,
        ).to(self.device)

        optim = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        total_steps = int(epochs) * max(1, len(loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=0.0
        )

        # ------- train loop (NO eval, NO AUROC) ----------
        self.head.train()
        self._train_history = []
        for epoch in range(int(epochs)):
            epoch_loss = 0.0
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
                loss = parts["risk"]
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                scheduler.step()
                epoch_loss += float(loss.detach().item())
                epoch_neg += float(parts["neg_risk"].item())
                if float(parts["neg_risk"].item()) < float(-beta):
                    n_correct += 1
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            avg_neg = epoch_neg / max(1, n_batches)
            cur_lr = float(optim.param_groups[0]["lr"])
            self._train_history.append({
                "epoch": float(epoch),
                "loss": avg,
                "neg_risk": avg_neg,
                "nn_correction_batches": float(n_correct),
                "lr": cur_lr,
            })
            if verbose:
                print(
                    f"[pu_bce][fit] epoch={epoch + 1}/{int(epochs)} "
                    f"risk={avg:.5f} neg_risk={avg_neg:+.5f} "
                    f"nn_corr_batches={n_correct}/{n_batches} "
                    f"lr={cur_lr:.2e} Np={Np} Nu={Nu} pi_p={float(pi_p):.3f}",
                    flush=True,
                )

        # ------- per-task threshold calibration on disjoint success-calib ----------
        # success_percentile only (no failure labels available in this branch).
        self.head.eval()
        self.thresholds = {}
        self.calib_stats = {}
        q = 100.0 * (1.0 - float(delta) / 100.0)
        for task, calib_seqs in success_calib_per_task.items():
            seqs = [t for t in calib_seqs if t.numel() > 0]
            if not seqs:
                raise ValueError(
                    f"Task {task!r}: success_calib_per_task[{task!r}] is empty; "
                    "cannot calibrate threshold."
                )
            calib_feats = torch.cat([s.to(torch.float32).reshape(-1, s.shape[-1]) for s in seqs], dim=0)
            g_calib = self._logits_np(calib_feats)
            failure_score = -g_calib  # higher = more failure
            tau = float(np.percentile(failure_score.astype(np.float64), q=q))
            self.thresholds[task] = tau
            self.calib_stats[task] = PUCalibStats(
                threshold=tau,
                num_calib_frames=int(failure_score.size),
                calib_score_min=float(failure_score.min()),
                calib_score_max=float(failure_score.max()),
                calib_score_mean=float(failure_score.mean()),
                calib_score_std=float(failure_score.std()),
            )
            if verbose:
                print(
                    f"[pu_bce][calib] task={task} tau={tau:.5f} "
                    f"n_calib={failure_score.size} "
                    f"mean={failure_score.mean():.5f} std={failure_score.std():.5f}",
                    flush=True,
                )

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
            self.head = BCEHead(in_dim=in_dim, hidden=hidden, num_layers=num_layers).to(self.device)
        self.head.load_state_dict(state["head"])  # type: ignore[arg-type]
        self.thresholds = {str(k): float(v) for k, v in dict(state.get("thresholds", {})).items()}  # type: ignore[arg-type]
        self._delta = None if state.get("delta") is None else float(state["delta"])  # type: ignore[arg-type]
        self._pi_p = None if state.get("pi_p") is None else float(state["pi_p"])  # type: ignore[arg-type]
        self._surrogate = str(state.get("loss_surrogate", "sigmoid"))
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
