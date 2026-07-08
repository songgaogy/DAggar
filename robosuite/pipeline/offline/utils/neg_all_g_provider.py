"""Per-frame negative-branch membership provider for offline.mode == "neg_all".

In naive/normal modes the negative branch is driven by ``1 - w_pos``. The
"neg_all" mode instead trains the negative branch on ALL ``offline_data`` rows
(success_rollout + fail_rollout) with weight 1, while the positive branch trains
on success data only (expert + success_rollout, via ``is_intervention``).

That needs a per-frame *neg membership* signal that is decoupled from ``w_pos``:
1 for any ``offline_data`` row, 0 for expert (pretrain) rows. This provider
precomputes that membership once per valid chunk window (keyed on the buffer
start index) and serves it by ``batch.metadata["start_indices"]`` — the same
lookup mechanism as ``OfflineAdvantageGProvider``, but with no IQL/encoder.

``DipoleFlowPolicy._neg_all_branch_weights`` reads this as ``w_neg`` directly
(clamped to [0, 1]); see ``branch_weight_mode == "neg_all"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from robosuite.pipeline.offline.utils.buffer import _trajectory_kind

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer


def precompute_neg_all_membership(
    base_buffer: "FlowDaggerReplayBuffer",
) -> tuple[torch.Tensor, dict[int, int]]:
    """Classify every valid chunk window as offline_data (1) vs expert (0).

    Returns ``(neg_raw, start_to_row)``:
      - ``neg_raw``: CPU float tensor of shape ``(N,)`` in
        ``_get_valid_start_indices_locked()`` order — 1.0 for success/fail
        rollout windows, 0.0 for expert (pretrain) windows.
      - ``start_to_row``: maps a storage start index -> row in ``neg_raw`` so the
        provider can gather by the start indices ``sample`` reports.

    Each materialized episode is single-kind, so the window's start frame kind
    (from ``info["episode_namespace"]``) is the window kind.
    """
    with base_buffer._lock:  # noqa: SLF001 — read valid windows under the buffer lock
        valid_starts = list(base_buffer._get_valid_start_indices_locked())  # noqa: SLF001
        if not valid_starts:
            raise ValueError(
                "precompute_neg_all_membership: base buffer has no valid chunk windows."
            )
        neg_vals: list[float] = []
        for start in valid_starts:
            transition = base_buffer._storage[int(start)]  # noqa: SLF001
            namespace = (transition.info or {}).get("episode_namespace", "")
            kind = _trajectory_kind(namespace)
            # offline_data rows (success + fail rollouts) train the negative
            # branch; expert / unknown rows do not.
            neg_vals.append(1.0 if kind in ("success", "fail") else 0.0)

    neg_raw = torch.tensor(neg_vals, dtype=torch.float32)
    start_to_row = {int(start): row for row, start in enumerate(valid_starts)}
    return neg_raw, start_to_row


class NegAllGProvider:
    """Serve precomputed neg-branch membership (0/1) by chunk start index."""

    def __init__(self, neg_raw: torch.Tensor, start_to_row: dict[int, int]) -> None:
        self._neg_raw = neg_raw.detach().to("cpu").reshape(-1)
        self._start_to_row = dict(start_to_row)

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        start_indices = batch.metadata.get("start_indices")
        if start_indices is None:
            raise KeyError(
                "NegAllGProvider requires batch.metadata['start_indices']; "
                "sample directly from the replay buffer (DipoleReplayBuffer.sample)."
            )
        try:
            rows = [self._start_to_row[int(s)] for s in start_indices]
        except KeyError as exc:  # pragma: no cover — buffer mutated after precompute
            raise KeyError(
                f"NegAllGProvider: start index {exc} has no precomputed membership. "
                "The replay buffer must stay static after precompute."
            ) from exc

        row_idx = torch.tensor(rows, dtype=torch.long)
        device = batch.action_sequences_raw.device
        return self._neg_raw.index_select(0, row_idx).to(device).reshape(-1)

    @torch.no_grad()
    def compute_g_for_observation(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Single-observation variant. Not supported: neg_all is offline-only."""
        raise NotImplementedError(
            "NegAllGProvider does not support single-observation scoring; "
            "it is only used for offline neg_all training."
        )

    def bind_policy_cameras(self, policy_cameras: list[str]) -> None:
        """No-op: neg_all mode has no encoder to bind cameras to."""
        return None


__all__ = ["precompute_neg_all_membership", "NegAllGProvider"]
