"""Constant-negative G provider for the naive offline-DIPOLE hard-label split.

Used when ``offline.mode == "naive"``. In that mode the replay buffer is
assembled so that:

- expert (pretrain_dir) and ``success_rollout`` frames carry
  ``is_intervention=True`` -> the policy forces ``w_pos=1, w_neg=0`` on them
  (positive branch), and
- ``fail_rollout`` frames carry ``is_intervention=False`` -> they are the ONLY
  frames whose weight is still decided by the G provider.

So a provider that returns a large *negative* constant G for the whole batch
sends exactly those fail frames to the negative branch:
``w_pos = sigmoid(beta * (-BIG) + k) ~= 0`` -> ``w_neg ~= 1`` (see
``DipoleFlowPolicy._g_weights_from_raw`` / ``_compute_branch_weights``). The
success/expert frames get the same raw G but are overridden to ``w_pos=1`` by
the intervention mask, so the constant only affects fail frames.

This keeps the naive split entirely inside ``offline/`` — no IQL/advantage/
discriminator stack and no change to the DIPOLE core. It implements the same
``compute_g_for_batch(batch) -> (B,)`` contract as
``algorithms/dipole/advantage_g_provider.py::AdvantageGProvider``.
"""

from __future__ import annotations

from typing import Any

import torch

# Large enough that sigmoid(beta * G + k) saturates to ~0 for any sane
# (positive) beta/k, i.e. fail frames get w_neg ~= 1.
_NEG_G: float = -1.0e6


class NaiveNegativeGProvider:
    """Return a large constant negative G for every sample in the batch."""

    def __init__(self, neg_g: float = _NEG_G) -> None:
        self.neg_g = float(neg_g)

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        """Args:
            batch: ``DipoleBatch``. Only its size/device are used.
        Returns:
            (B,) tensor filled with ``neg_g`` on the batch's device.
        """
        device = batch.action_sequences_raw.device
        return torch.full((int(batch.batch_size),), self.neg_g, dtype=torch.float32, device=device)

    @torch.no_grad()
    def compute_g_for_observation(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Single-observation variant. Not supported: naive mode is offline-only."""
        raise NotImplementedError(
            "NaiveNegativeGProvider does not support single-observation scoring; "
            "it is only used for offline hard-label training."
        )

    def bind_policy_cameras(self, policy_cameras: list[str]) -> None:
        """No-op: naive mode has no encoder to bind cameras to."""
        return None


__all__ = ["NaiveNegativeGProvider"]
