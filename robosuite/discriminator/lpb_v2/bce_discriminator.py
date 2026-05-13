"""BCE-style density-ratio failure discriminator on frozen WAM latent (GT split).

``D_e`` pools success-rollout frames plus, from each failure bank trajectory, the
prefix before ``first_gt_failure_frame()``; ``D_o`` pools suffix frames after
that cut. One shared MLP head ``g_θ(z)`` is trained with ``BCEWithLogitsLoss``.

Score convention (matches ``LPBV2KNN`` so the benchmark JSON layout is unchanged):

    g(z)            = head(z)                # expert-likeness logit, higher = more expert
    failure_score   = -g(z)                  # step_scores; higher = more failure
    tau_task        = percentile(failure_score over success-calib frames, 100 - delta)
    pred_t = 1      iff failure_score_t >= tau_task

A single ``BCEDiscriminator`` holds one shared head **and** a ``Dict[str, float]``
of per-task thresholds. ``score(features, task=...)`` does the per-task lookup.

**Hard constraint:** ``fit(...)`` does not run any evaluation or compute AUROC.
It trains for a fixed number of epochs against the BCE objective and then
calibrates per-task thresholds on the disjoint success-calib split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .knn import DetectionResult


# --------------------------------------------------------------------------- #
# Head                                                                        #
# --------------------------------------------------------------------------- #


class BCEHead(nn.Module):
    """MLP scalar-logit head on top of a frozen latent.

    Architecture (num_layers=2, hidden=256):
        Linear(in_dim, hidden) → LayerNorm → GELU
        Linear(hidden,  hidden) → LayerNorm → GELU
        Linear(hidden, 1)
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
# Calibration stats                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class BCECalibStats:
    threshold: float
    num_calib_frames: int
    calib_score_min: float
    calib_score_max: float
    calib_score_mean: float
    calib_score_std: float


# --------------------------------------------------------------------------- #
# Discriminator                                                               #
# --------------------------------------------------------------------------- #


class BCEDiscriminator:
    """Shared-head BCE discriminator with per-task threshold table.

    Workflow:
        det = BCEDiscriminator(in_dim, ...)
        det.fit(expert_features, other_features, expert_calib_per_task,
                epochs=..., lr=..., delta=..., seed=...)
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
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        self.head: BCEHead = BCEHead(in_dim=self.in_dim, hidden=self.hidden, num_layers=self.num_layers).to(self.device)
        self.thresholds: Dict[str, float] = {}
        self.calib_stats: Dict[str, BCECalibStats] = {}
        # populated by fit; used by score()
        self._delta: Optional[float] = None
        # bookkeeping
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
        expert_features: Sequence[torch.Tensor],
        other_features: Sequence[torch.Tensor],
        expert_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        delta: float = 10.0,
        seed: int = 0,
        verbose: bool = True,
        max_expert_other_ratio: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Train the shared head with BCE; then calibrate per-task thresholds.

        Args:
            expert_features: list of (T_i, D) tensors. Pooled into ``D_e``.
            other_features:  list of (T_j, D) tensors. Pooled into ``D_o``.
            expert_calib_per_task: ``{task_name: [(T_k, D), ...]}`` of disjoint
                success-calib trajectories per task. Threshold is calibrated
                per-task on these features.
            epochs: fixed epoch budget. No early stopping, no validation eval.
            delta: percentile-based false-alarm budget in [0, 100]. ``tau =
                percentile(failure_score on success-calib, 100 - delta)``.
            max_expert_other_ratio: cap ``|D_e| <= max_expert_other_ratio * |D_o|``
                via random subsampling. ``None`` disables the cap. Default 1.0 (1:1).
        Returns:
            Per-task threshold dict.
        """
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if max_expert_other_ratio is not None and float(max_expert_other_ratio) <= 0.0:
            raise ValueError(
                f"max_expert_other_ratio must be > 0 or None, got {max_expert_other_ratio}"
            )

        self._delta = float(delta)

        # ------- assemble pooled (D_e, D_o) frames ----------
        e_seqs = [t.detach() for t in expert_features if t.numel() > 0]
        o_seqs = [t.detach() for t in other_features if t.numel() > 0]
        if not e_seqs:
            raise ValueError("expert_features is empty after filtering")
        if not o_seqs:
            raise ValueError("other_features is empty after filtering")

        Z_e = torch.cat([t.to(torch.float32).reshape(-1, t.shape[-1]) for t in e_seqs], dim=0)
        Z_o = torch.cat([t.to(torch.float32).reshape(-1, t.shape[-1]) for t in o_seqs], dim=0)
        if Z_e.shape[1] != self.in_dim or Z_o.shape[1] != self.in_dim:
            raise ValueError(
                f"feature dim mismatch: in_dim={self.in_dim}, "
                f"D_e={Z_e.shape[1]}, D_o={Z_o.shape[1]}"
            )
        Ne_raw = int(Z_e.shape[0])
        No_raw = int(Z_o.shape[0])

        # ------- cap |D_e| relative to |D_o| (avoid the success side dwarfing failures) ----------
        if max_expert_other_ratio is not None:
            cap = int(float(max_expert_other_ratio) * float(No_raw))
            if cap > 0 and Ne_raw > cap:
                g_sub = torch.Generator(device="cpu").manual_seed(int(seed))
                idx = torch.randperm(Ne_raw, generator=g_sub)[:cap]
                Z_e = Z_e[idx]
                if verbose:
                    print(
                        f"[bce][balance] subsampled D_e: {Ne_raw} -> {cap} frames "
                        f"(ratio={float(max_expert_other_ratio):.2f} * No={No_raw})",
                        flush=True,
                    )
        Ne = int(Z_e.shape[0])
        No = int(Z_o.shape[0])
        Z = torch.cat([Z_e, Z_o], dim=0)
        # Labels: 1 = expert, 0 = other. BCEWithLogitsLoss wants float targets.
        Y = torch.cat([
            torch.ones(Ne, dtype=torch.float32),
            torch.zeros(No, dtype=torch.float32),
        ], dim=0)

        # ------- balanced sampling: each class has equal total weight ----------
        w_e = 1.0 / float(Ne)
        w_o = 1.0 / float(No)
        weights = torch.cat([
            torch.full((Ne,), w_e, dtype=torch.float64),
            torch.full((No,), w_o, dtype=torch.float64),
        ], dim=0)
        num_samples = int(2 * (Ne + No))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        dataset = TensorDataset(Z, Y)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        # ------- deterministic head init + optim & schedule ----------
        # Re-init the head from a fresh seed so that two calls to fit(seed=k)
        # produce identical thresholds regardless of the prior global RNG state
        # at __init__ time.
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
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
        bce_loss_fn = nn.BCEWithLogitsLoss()

        # ------- train loop (NO eval, NO AUROC) ----------
        self.head.train()
        self._train_history = []
        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            n_batches = 0
            for z_batch, y_batch in loader:
                z_batch = z_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                g = self.head(z_batch)
                loss = bce_loss_fn(g, y_batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                scheduler.step()
                epoch_loss += float(loss.detach().item())
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            cur_lr = float(optim.param_groups[0]["lr"])
            self._train_history.append({"epoch": float(epoch), "loss": avg, "lr": cur_lr})
            if verbose:
                print(
                    f"[bce][fit] epoch={epoch + 1}/{int(epochs)} "
                    f"loss={avg:.5f} lr={cur_lr:.2e} "
                    f"batches={n_batches} Ne={Ne} No={No}",
                    flush=True,
                )

        # ------- per-task threshold calibration on disjoint success-calib ----------
        self.head.eval()
        self.thresholds = {}
        self.calib_stats = {}
        q = 100.0 * (1.0 - float(delta) / 100.0)
        for task, calib_seqs in expert_calib_per_task.items():
            seqs = [t for t in calib_seqs if t.numel() > 0]
            if not seqs:
                raise ValueError(
                    f"Task {task!r}: expert_calib_per_task[{task!r}] is empty; "
                    "cannot calibrate threshold."
                )
            calib_feats = torch.cat([s.to(torch.float32).reshape(-1, s.shape[-1]) for s in seqs], dim=0)
            g_calib = self._logits_np(calib_feats)
            failure_score = -g_calib  # higher = more failure
            tau = float(np.percentile(failure_score.astype(np.float64), q=q))
            self.thresholds[task] = tau
            self.calib_stats[task] = BCECalibStats(
                threshold=tau,
                num_calib_frames=int(failure_score.size),
                calib_score_min=float(failure_score.min()),
                calib_score_max=float(failure_score.max()),
                calib_score_mean=float(failure_score.mean()),
                calib_score_std=float(failure_score.std()),
            )
            if verbose:
                print(
                    f"[bce][calib] task={task} tau={tau:.5f} "
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
            # Re-init head with the saved geometry.
            self.in_dim = in_dim
            self.hidden = hidden
            self.num_layers = num_layers
            self.head = BCEHead(in_dim=in_dim, hidden=hidden, num_layers=num_layers).to(self.device)
        self.head.load_state_dict(state["head"])  # type: ignore[arg-type]
        self.thresholds = {str(k): float(v) for k, v in dict(state.get("thresholds", {})).items()}  # type: ignore[arg-type]
        self._delta = None if state.get("delta") is None else float(state["delta"])  # type: ignore[arg-type]
        cs = state.get("calib_stats", {}) or {}
        self.calib_stats = {
            str(k): BCECalibStats(
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
