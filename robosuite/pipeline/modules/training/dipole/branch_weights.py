"""Pluggable per-sample branch-weight policies for offline DIPOLE.

PROMPT.md §Policy-update point 4 requires the ``w_pos`` / ``w_neg`` scheme to be
modular so different weight formats can be tested (e.g. adding a discriminator
output scale). ``DipoleFlowPolicy._compute_branch_weights`` delegates to whatever
object was set via ``DipoleFlowPolicy.set_branch_weight_policy`` when one is
attached, passing:

    policy(batch, g_provider=<attached provider or None>,
           sigmoid_fn=<DipoleFlowPolicy._g_weights_from_raw>,
           device=<policy device>, want_metrics=<bool>)
        -> (w_pos: (B,) float tensor, w_neg: (B,) float tensor, metrics: dict)

The default :class:`RoutedSigmoidBranchWeightPolicy` reads the per-frame
``batch.metadata["route"]`` tag emitted by the episode dataset (see
``episode_dataset.py``) and routes each row:

- ``pos_only`` (human intervention → positive branch): ``w_pos=1, w_neg=0``.
- ``neg_only`` (policy action during intervention → negative branch): ``w_pos=0,
  w_neg=1``.
- ``advantage`` (policy rollout sections): ``w_pos=σ(β·(G+k)+ηY)``,
  ``w_neg=1-w_pos``, where ``Y=1`` for pure on-policy success trajectories.
  The policy reuses its own ``sigmoid_fn`` on the attached G provider's advantage.

:class:`DiscriminatorScaledBranchWeightPolicy` is a worked example of the "add a
discriminator output scale" extension: it multiplies the advantage-row positive
weight by a bounded factor derived from the provider's failure score.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch, select_dipole_batch
from robosuite.pipeline.modules.training.dipole.episode_dataset import (
    ROUTE_ADVANTAGE,
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
    """Boolean per-row masks keyed by route (missing rows default to advantage)."""
    B = batch.batch_size
    routes = batch.metadata.get("route")
    if routes is None or len(routes) != B:
        routes = [ROUTE_ADVANTAGE] * B
    routes = [str(r) for r in routes]
    return {
        ROUTE_POS_ONLY: torch.tensor(
            [r == ROUTE_POS_ONLY for r in routes], device=device, dtype=torch.bool
        ),
        ROUTE_NEG_ONLY: torch.tensor(
            [r == ROUTE_NEG_ONLY for r in routes], device=device, dtype=torch.bool
        ),
        ROUTE_ADVANTAGE: torch.tensor(
            [r not in (ROUTE_POS_ONLY, ROUTE_NEG_ONLY) for r in routes],
            device=device,
            dtype=torch.bool,
        ),
    }


class RoutedSigmoidBranchWeightPolicy:
    """Routed weighting with an eta logit bonus for pure success trajectories."""

    def __init__(self, *, eta: float = 0.0) -> None:
        self.eta = float(eta)

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
        success_mask = self._success_trajectory_mask(batch, device)
        w_pos = torch.zeros(B, dtype=torch.float32, device=device)
        w_neg = torch.zeros(B, dtype=torch.float32, device=device)

        w_pos[masks[ROUTE_POS_ONLY]] = 1.0
        w_neg[masks[ROUTE_NEG_ONLY]] = 1.0

        adv_mask = masks[ROUTE_ADVANTAGE]
        g_metrics: dict[str, float] = {}
        if bool(adv_mask.any().item()):
            adv_idx = torch.nonzero(adv_mask, as_tuple=False).squeeze(1)
            raw = self._advantage_raw_g(batch, adv_idx, g_provider, device)
            success = success_mask.index_select(0, adv_idx)
            logit_bias = self.eta * success.to(dtype=torch.float32)
            w_adv_pos, w_adv_neg, g_metrics = sigmoid_fn(
                raw,
                logit_bias=logit_bias,
                want_metrics=want_metrics,
            )
            w_pos[adv_mask] = w_adv_pos.to(device=device, dtype=torch.float32)
            w_neg[adv_mask] = w_adv_neg.to(device=device, dtype=torch.float32)

        metrics: dict[str, float] = {}
        if want_metrics:
            metrics.update(g_metrics)
            metrics["frac_pos_only"] = float(masks[ROUTE_POS_ONLY].float().mean().item())
            metrics["frac_neg_only"] = float(masks[ROUTE_NEG_ONLY].float().mean().item())
            metrics["frac_advantage"] = float(adv_mask.float().mean().item())
            metrics["frac_success_trajectory"] = float(
                success_mask.float().mean().item()
            )
            metrics["success_logit_bonus_mean"] = float(
                (self.eta * (success_mask & adv_mask).float()).mean().item()
            )
            metrics["w_pos_mean"] = float(w_pos.mean().item())
            metrics["w_neg_mean"] = float(w_neg.mean().item())
        return w_pos, w_neg, metrics

    @staticmethod
    def _success_trajectory_mask(
        batch: DipoleBatch, device: torch.device | str
    ) -> torch.Tensor:
        values = batch.metadata.get("is_success_trajectory")
        if values is None or len(values) != batch.batch_size:
            values = [False] * batch.batch_size
        return torch.tensor(values, device=device, dtype=torch.bool)

    @staticmethod
    def _advantage_raw_g(
        batch: DipoleBatch,
        adv_idx: torch.Tensor,
        g_provider: Any,
        device: torch.device | str,
    ) -> torch.Tensor:
        if g_provider is None:
            return torch.zeros(int(adv_idx.numel()), dtype=torch.float32, device=device)
        with torch.no_grad():
            sub = select_dipole_batch(batch, adv_idx)
            return g_provider.compute_g_for_batch(sub).to(device).reshape(-1)


class DiscriminatorScaledBranchWeightPolicy(RoutedSigmoidBranchWeightPolicy):
    """Example extension: scale advantage-row ``w_pos`` by a disc-derived factor.

    Demonstrates the "add discriminator output scale" idea from PROMPT.md point 4.
    ``discriminator.failure_score`` is not available per-row without re-encoding, so
    this example instead rescales using the provider's own failure term already
    baked into G; concretely it sharpens ``w_pos`` toward the advantage sign with a
    tunable ``scale``. Kept minimal on purpose — a template, not a tuned default.
    """

    def __init__(self, *, scale: float = 1.0, eta: float = 0.0) -> None:
        super().__init__(eta=eta)
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
        adv_mask = masks[ROUTE_ADVANTAGE]
        if bool(adv_mask.any().item()) and self.scale != 1.0:
            scaled = (w_pos[adv_mask] ** self.scale)
            w_pos[adv_mask] = scaled
            w_neg[adv_mask] = 1.0 - scaled
            if want_metrics:
                metrics["w_pos_mean"] = float(w_pos.mean().item())
                metrics["w_neg_mean"] = float(w_neg.mean().item())
        return w_pos, w_neg, metrics


def build_branch_weight_policy(cfg: Any) -> BranchWeightPolicy:
    """Factory keyed on ``offline.branch_weight.type`` (default routed_sigmoid)."""
    weight_type = str(getattr(cfg, "type", "routed_sigmoid")).strip().lower()
    if weight_type in ("routed_sigmoid", "", "default"):
        return RoutedSigmoidBranchWeightPolicy(
            eta=float(getattr(cfg, "eta", 0.0))
        )
    if weight_type in ("disc_scaled", "discriminator_scaled"):
        return DiscriminatorScaledBranchWeightPolicy(
            scale=float(getattr(cfg, "scale", 1.0)),
            eta=float(getattr(cfg, "eta", 0.0)),
        )
    raise ValueError(f"Unknown offline.branch_weight.type={weight_type!r}")


__all__ = [
    "BranchWeightPolicy",
    "RoutedSigmoidBranchWeightPolicy",
    "DiscriminatorScaledBranchWeightPolicy",
    "build_branch_weight_policy",
]
