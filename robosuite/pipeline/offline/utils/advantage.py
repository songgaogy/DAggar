"""Offline TD-advantage G provider for offline DIPOLE training.

Online DIPOLE-RL weights the two flow branches with
``G = alpha * A - beta * disc`` where ``A = mean(Q) - V``
(:class:`AdvantageGProvider`). README step-4 requires the offline variant to use
the **TD residual** instead::

    A = r + gamma^H * V_lcb_target(s') - V_lcb(s)

with ``V_lcb`` the soft-LCB ensemble value (``iql.v_lcb`` / ``iql.target_v_lcb``,
``mean_k − β·std_k``), ``gamma^H`` (H=action_horizon) and ``r`` the
chunk-aggregated reward — a drop-in replacement for ``Q - V`` with matching
sign: larger A (better-than-V transition) raises ``w_pos``.

Because the encoder, IQL critics and nnPU head are all **frozen** in offline
training, each valid chunk window's ``A`` and failure score are constant for the
whole run. We therefore pre-encode every window once
(:func:`precompute_offline_advantage`) and, at training time, look the values up
by the window start index that :meth:`DipoleReplayBuffer.sample` already returns
in ``batch.metadata["start_indices"]`` — no per-step re-encoding, and no need to
carry ``next_obs`` / ``reward`` through the BC batch.

Note: the BC images are augmented per step, but the advantage uses the
un-augmented pre-encoded features. That is intentional — the advantage is a
property of the underlying transition, not of a particular augmentation, and the
frozen-critic value is more stable computed on the clean observation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from robosuite.pipeline.algorithms.dipole.advantage_g_provider import AdvantageGProvider

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
    from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
    from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
    from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner

logger = logging.getLogger(__name__)


@torch.no_grad()
def precompute_offline_advantage(
    *,
    base_buffer: "FlowDaggerReplayBuffer",
    iql_learner: "IQLLearner",
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator",
    iql_cfg: "IQLConfig",
    device: str,
    encode_batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Encode every valid chunk window once and compute its TD advantage.

    Returns ``(advantage_raw, failure_raw, start_to_row)``:
      - ``advantage_raw`` / ``failure_raw``: CPU float tensors of shape ``(N,)``
        in ``_get_valid_start_indices_locked()`` order;
      - ``start_to_row``: maps a storage start index -> row in those tensors,
        so :class:`OfflineAdvantageGProvider` can gather by the start indices
        that ``DipoleReplayBuffer.sample`` reports.
    """
    # Local import to avoid a heavy import at module load time.
    from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer

    replay = IQLReplayBuffer(base_buffer, iql_cfg)
    with base_buffer._lock:  # noqa: SLF001 — read valid windows under the buffer lock
        valid_starts = list(base_buffer._get_valid_start_indices_locked())  # noqa: SLF001
    if not valid_starts:
        raise ValueError(
            "precompute_offline_advantage: base buffer has no valid chunk windows."
        )

    H = int(iql_cfg.action_horizon)
    gamma_h = float(iql_cfg.discount) ** H
    encode_bs = max(1, int(encode_batch_size))

    adv_parts: list[torch.Tensor] = []
    fail_parts: list[torch.Tensor] = []
    for offset in range(0, len(valid_starts), encode_bs):
        batch_starts = valid_starts[offset : offset + encode_bs]
        with base_buffer._lock:  # noqa: SLF001
            sequences = [base_buffer._storage[s : s + H] for s in batch_starts]  # noqa: SLF001
        step_batch = replay._build_step_batch(  # noqa: SLF001 — reuse the exact warmup encoding
            sequences,
            batch_starts,
            encoder=encoder,
            discriminator=discriminator,
            device=device,
        )
        v_s = iql_learner.v_lcb(step_batch.v_state_feature)
        v_sp = iql_learner.target_v_lcb(step_batch.next_v_state_feature)
        advantage = (step_batch.rewards + gamma_h * v_sp - v_s).reshape(-1)
        failure = discriminator.failure_score(
            chunk_feature=step_batch.q_chunk_feature
        ).reshape(-1)
        adv_parts.append(advantage.detach().to("cpu"))
        fail_parts.append(failure.detach().to("cpu"))

    advantage_raw = torch.cat(adv_parts, dim=0)
    failure_raw = torch.cat(fail_parts, dim=0)
    start_to_row = {int(start): row for row, start in enumerate(valid_starts)}
    logger.info(
        "[offline] precomputed TD advantage for %d windows "
        "(gamma^H=%.4f, adv_mean=%.4f adv_std=%.4f, fail_mean=%.4f)",
        advantage_raw.numel(),
        gamma_h,
        float(advantage_raw.mean().item()),
        float(advantage_raw.std().item()) if advantage_raw.numel() > 1 else 0.0,
        float(failure_raw.mean().item()),
    )
    return advantage_raw, failure_raw, start_to_row


class OfflineAdvantageGProvider(AdvantageGProvider):
    """``AdvantageGProvider`` that serves precomputed TD advantage by start index.

    Reuses the parent's ``bind_policy_cameras`` but replaces
    ``compute_g_for_batch`` with a cache lookup keyed on
    ``batch.metadata["start_indices"]``.
    """

    def __init__(
        self,
        *,
        iql_learner: "IQLLearner",
        discriminator: "FrozenNNPUDiscriminator",
        encoder: "SharedDynamicsEncoder",
        alpha: float,
        beta: float,
        advantage_raw: torch.Tensor,
        failure_raw: torch.Tensor,
        start_to_row: dict[int, int],
    ) -> None:
        super().__init__(
            iql_learner=iql_learner,
            discriminator=discriminator,
            encoder=encoder,
            alpha=alpha,
            beta=beta,
        )
        self._advantage_raw = advantage_raw.detach().to("cpu").reshape(-1)
        self._failure_raw = failure_raw.detach().to("cpu").reshape(-1)
        self._start_to_row = dict(start_to_row)

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        start_indices = batch.metadata.get("start_indices")
        if start_indices is None:
            raise KeyError(
                "OfflineAdvantageGProvider requires batch.metadata['start_indices']; "
                "sample directly from the replay buffer (DipoleReplayBuffer.sample)."
            )
        try:
            rows = [self._start_to_row[int(s)] for s in start_indices]
        except KeyError as exc:  # pragma: no cover — buffer mutated after precompute
            raise KeyError(
                f"OfflineAdvantageGProvider: start index {exc} has no precomputed "
                "advantage. The replay buffer must stay static after precompute."
            ) from exc

        row_idx = torch.tensor(rows, dtype=torch.long)
        device = batch.action_sequences_raw.device
        advantage = self._advantage_raw.index_select(0, row_idx).to(device)
        failure = self._failure_raw.index_select(0, row_idx).to(device)

        g = self.alpha * advantage - self.beta * failure
        return g.reshape(-1)


__all__ = ["precompute_offline_advantage", "OfflineAdvantageGProvider"]
