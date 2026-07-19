"""Per-sample branch weights for discriminator-weighted offline DIPOLE.

PROMPT.md §Policy-update point 4 requires the ``w_pos`` / ``w_neg`` scheme to be
modular so different weight formats can be tested (e.g. adding a discriminator
output scale). ``DipoleFlowPolicy._compute_branch_weights`` delegates to whatever
object was set via ``DipoleFlowPolicy.set_branch_weight_policy`` when one is
attached, passing:

    policy(batch, g_provider=<attached provider or None>,
           sigmoid_fn=<DipoleFlowPolicy._g_weights_from_g>,
           device=<policy device>, want_metrics=<bool>)
        -> (w_pos: (B,) float tensor, w_neg: (B,) float tensor, metrics: dict)

The default :class:`RoutedSigmoidBranchWeightPolicy` reads the per-frame
``batch.metadata["route"]`` tag emitted by the episode dataset (see
``episode_dataset.py``) and routes each row:

- ``pos_only`` (human intervention → positive branch): ``w_pos=1, w_neg=0``.
- ``neg_only`` (policy action during intervention → negative branch): ``w_pos=0,
  w_neg=1``.
- ``disc_weighted`` (policy rollout sections): ``w_pos=σ(β·G)``,
  ``w_neg=1-w_pos``, using the cached calibrated discriminator margin.

:class:`DiscriminatorScaledBranchWeightPolicy` is an optional power transform
for discriminator-weighted rows.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch, select_dipole_batch
from robosuite.pipeline.offline.utils.episode_dataset import (
    ROUTE_DISC_WEIGHTED,
    ROUTE_NEG_ONLY,
    ROUTE_POS_ONLY,
)

SigmoidFn = Callable[..., tuple[torch.Tensor, torch.Tensor, dict[str, float]]]


class BranchWeightPolicy(Protocol):
    def __call__(
        self,
        batch: DipoleBatch,
        *,
        g_provider: Any,
        sigmoid_fn: SigmoidFn,
        device: torch.device | str,
        want_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        ...


def _route_masks(
    batch: DipoleBatch, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Boolean per-row masks keyed by route (missing rows use discriminator G)."""
    B = batch.batch_size
    routes = batch.metadata.get("route")
    if routes is None or len(routes) != B:
        routes = [ROUTE_DISC_WEIGHTED] * B
    routes = [str(r) for r in routes]
    return {
        ROUTE_POS_ONLY: torch.tensor(
            [r == ROUTE_POS_ONLY for r in routes], device=device, dtype=torch.bool
        ),
        ROUTE_NEG_ONLY: torch.tensor(
            [r == ROUTE_NEG_ONLY for r in routes], device=device, dtype=torch.bool
        ),
        ROUTE_DISC_WEIGHTED: torch.tensor(
            [r not in (ROUTE_POS_ONLY, ROUTE_NEG_ONLY) for r in routes],
            device=device,
            dtype=torch.bool,
        ),
    }


class RoutedSigmoidBranchWeightPolicy:
    """Route human/negative rows hard and policy rows by discriminator G."""

    def __call__(
        self,
        batch: DipoleBatch,
        *,
        g_provider: Any,
        sigmoid_fn: SigmoidFn,
        device: torch.device | str,
        want_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        B = batch.batch_size
        masks = _route_masks(batch, device)
        w_pos = torch.zeros(B, dtype=torch.float32, device=device)
        w_neg = torch.zeros(B, dtype=torch.float32, device=device)

        w_pos[masks[ROUTE_POS_ONLY]] = 1.0
        w_neg[masks[ROUTE_NEG_ONLY]] = 1.0

        disc_mask = masks[ROUTE_DISC_WEIGHTED]
        g_metrics: dict[str, float] = {}
        if bool(disc_mask.any().item()):
            disc_idx = torch.nonzero(disc_mask, as_tuple=False).squeeze(1)
            g_values, score_metrics = self._discriminator_g(
                batch, disc_idx, g_provider, device, want_metrics=want_metrics
            )
            w_adv_pos, w_adv_neg, g_metrics = sigmoid_fn(
                g_values, want_metrics=want_metrics
            )
            g_metrics.update(score_metrics)
            w_pos[disc_mask] = w_adv_pos.to(device=device, dtype=torch.float32)
            w_neg[disc_mask] = w_adv_neg.to(device=device, dtype=torch.float32)

        metrics: dict[str, float] = {}
        if want_metrics:
            metrics.update(g_metrics)
            metrics["frac_pos_only"] = float(masks[ROUTE_POS_ONLY].float().mean().item())
            metrics["frac_neg_only"] = float(masks[ROUTE_NEG_ONLY].float().mean().item())
            metrics["frac_disc_weighted"] = float(disc_mask.float().mean().item())
            metrics["w_pos_mean"] = float(w_pos.mean().item())
            metrics["w_neg_mean"] = float(w_neg.mean().item())
        return w_pos, w_neg, metrics

    @staticmethod
    def _discriminator_g(
        batch: DipoleBatch,
        disc_idx: torch.Tensor,
        g_provider: Any,
        device: torch.device | str,
        *,
        want_metrics: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if g_provider is None:
            return (
                torch.zeros(int(disc_idx.numel()), dtype=torch.float32, device=device),
                {},
            )
        with torch.no_grad():
            sub = select_dipole_batch(batch, disc_idx)
            g_values = g_provider.compute_g_for_batch(sub).to(device).reshape(-1)
            metrics: dict[str, float] = {}
            if want_metrics and hasattr(g_provider, "raw_scores_for_batch"):
                raw_score = g_provider.raw_scores_for_batch(sub).to(device).reshape(-1)
                metrics = {
                    "raw_score_mean": float(raw_score.mean().item()),
                    "raw_score_std": float(
                        raw_score.std().item() if raw_score.numel() > 1 else 0.0
                    ),
                    "threshold": float(g_provider.threshold),
                }
            return g_values, metrics


class DiscriminatorScaledBranchWeightPolicy(RoutedSigmoidBranchWeightPolicy):
    """Optionally sharpen discriminator-weighted ``w_pos`` by a power."""

    def __init__(self, *, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def __call__(
        self,
        batch: DipoleBatch,
        *,
        g_provider: Any,
        sigmoid_fn: SigmoidFn,
        device: torch.device | str,
        want_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        w_pos, w_neg, metrics = super().__call__(
            batch,
            g_provider=g_provider,
            sigmoid_fn=sigmoid_fn,
            device=device,
            want_metrics=want_metrics,
        )
        masks = _route_masks(batch, device)
        disc_mask = masks[ROUTE_DISC_WEIGHTED]
        if bool(disc_mask.any().item()) and self.scale != 1.0:
            scaled = (w_pos[disc_mask] ** self.scale)
            w_pos[disc_mask] = scaled
            w_neg[disc_mask] = 1.0 - scaled
            if want_metrics:
                metrics["w_pos_mean"] = float(w_pos.mean().item())
                metrics["w_neg_mean"] = float(w_neg.mean().item())
        return w_pos, w_neg, metrics


def build_branch_weight_policy(cfg: Any) -> BranchWeightPolicy:
    """Factory keyed on ``offline.branch_weight.type`` (default routed_sigmoid)."""
    weight_type = str(getattr(cfg, "type", "routed_sigmoid")).strip().lower()
    if weight_type in ("routed_sigmoid", "", "default"):
        return RoutedSigmoidBranchWeightPolicy()
    if weight_type in ("disc_scaled", "discriminator_scaled"):
        return DiscriminatorScaledBranchWeightPolicy(
            scale=float(getattr(cfg, "scale", 1.0))
        )
    raise ValueError(f"Unknown offline.branch_weight.type={weight_type!r}")


__all__ = [
    "BranchWeightPolicy",
    "RoutedSigmoidBranchWeightPolicy",
    "DiscriminatorScaledBranchWeightPolicy",
    "build_branch_weight_policy",
]
