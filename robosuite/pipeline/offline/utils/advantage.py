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
from tqdm import tqdm

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
    estimator: str = "td1",
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Encode every valid chunk window once and compute its advantage.

    Two ``estimator`` modes (both read the soft-LCB ensemble value via
    :meth:`IQLLearner.compute_td_advantage`, which masks the bootstrap at
    terminal windows with ``(1 - done)``):

    - ``"td1"``: the 1-step (macro-step) TD residual
      ``delta_t = r_chunk + gamma^H*(1-done)*V_lcb_target(s') - V_lcb(s)``.
    - ``"gae"``: GAE(``gae_lambda``) accumulated backward over each episode's
      chunk chain, ``A_t = delta_t + gamma^H*lambda*(1-done_t)*A_{t+H}``. This
      mirrors the read-out in ``q_learning/utils/vis_qv.py`` exactly. Offline
      policy sections are stored as standalone short episodes with ``done=True``
      on the last frame, so the recursion stops at the section boundary (no
      cross-section bootstrap).

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

    estimator = str(estimator).lower()
    if estimator not in ("td1", "gae"):
        raise ValueError(
            f"precompute_offline_advantage: estimator must be 'td1' or 'gae', got {estimator!r}."
        )

    H = int(iql_cfg.action_horizon)
    gamma_h = float(iql_cfg.discount) ** H
    lam = float(gae_lambda)
    encode_bs = max(1, int(encode_batch_size))

    # episode index per valid window (aligned with valid_starts) so GAE chains
    # only within an episode / policy section, never across a section boundary.
    with base_buffer._lock:  # noqa: SLF001
        episode_indices = [
            int((base_buffer._storage[s].info or {}).get("episode_index", -1))  # noqa: SLF001
            for s in valid_starts
        ]

    delta_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    fail_parts: list[torch.Tensor] = []
    # Bottleneck is frozen-encoder encode + V/disc forward per window batch;
    # the subsequent GAE pass is a cheap CPU O(N) recursion.
    n_windows = len(valid_starts)
    batch_offsets = range(0, n_windows, encode_bs)
    for offset in tqdm(
        batch_offsets,
        total=(n_windows + encode_bs - 1) // encode_bs,
        desc=f"[offline] precompute {estimator.upper()} advantage",
        unit="batch",
    ):
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
        # 1-step (macro-step) TD residual on the soft-LCB value, done-masked —
        # identical definition to the online IQLLearner.compute_td_advantage and
        # the vis_qv.py read-out delta.
        delta = iql_learner.compute_td_advantage(
            step_batch.v_state_feature,
            step_batch.next_v_state_feature,
            step_batch.rewards,
            step_batch.dones,
        ).reshape(-1)
        failure = discriminator.failure_score(
            chunk_feature=step_batch.q_chunk_feature
        ).reshape(-1)
        delta_parts.append(delta.detach().to("cpu"))
        done_parts.append(step_batch.dones.detach().to("cpu").reshape(-1))
        fail_parts.append(failure.detach().to("cpu"))

    delta_raw = torch.cat(delta_parts, dim=0).reshape(-1)
    done_raw = torch.cat(done_parts, dim=0).reshape(-1)
    failure_raw = torch.cat(fail_parts, dim=0).reshape(-1)
    start_to_row = {int(start): row for row, start in enumerate(valid_starts)}

    if estimator == "gae":
        advantage_raw = _gae_over_episodes(
            delta=delta_raw,
            done=done_raw,
            valid_starts=valid_starts,
            episode_indices=episode_indices,
            start_to_row=start_to_row,
            horizon=H,
            gamma_h=gamma_h,
            lam=lam,
        )
    else:
        advantage_raw = delta_raw.clone()

    logger.info(
        "[offline] precomputed %s advantage for %d windows "
        "(gamma^H=%.4f, lambda=%.3f, adv_mean=%.4f adv_std=%.4f, fail_mean=%.4f)",
        estimator.upper(),
        advantage_raw.numel(),
        gamma_h,
        lam if estimator == "gae" else float("nan"),
        float(advantage_raw.mean().item()),
        float(advantage_raw.std().item()) if advantage_raw.numel() > 1 else 0.0,
        float(failure_raw.mean().item()),
    )
    return advantage_raw, failure_raw, start_to_row


def _gae_over_episodes(
    *,
    delta: torch.Tensor,
    done: torch.Tensor,
    valid_starts: list[int],
    episode_indices: list[int],
    start_to_row: dict[int, int],
    horizon: int,
    gamma_h: float,
    lam: float,
) -> torch.Tensor:
    """GAE(lambda) accumulated backward over each episode's H-strided chunk chain.

    ``A_t = delta_t + gamma^H * lambda * (1 - done_t) * A_{t+H}``, where the
    ``t+H`` successor is the window ``horizon`` storage steps ahead **iff it is a
    valid window in the same episode** (else the chain terminates and ``A_t``
    falls back to the 1-step ``delta_t`` — matching vis_qv's boundary ``0``).
    Rows are processed in descending start order so each successor is finalized
    before its predecessor.
    """
    delta_l = delta.tolist()
    done_l = done.tolist()
    adv = list(delta_l)
    coef = float(gamma_h) * float(lam)
    order = sorted(range(len(valid_starts)), key=lambda r: valid_starts[r], reverse=True)
    for r in order:
        if done_l[r] > 0.5:
            continue  # terminal window: no bootstrap, no propagation
        nb = start_to_row.get(int(valid_starts[r]) + int(horizon))
        if nb is None or episode_indices[nb] != episode_indices[r]:
            continue  # no in-episode successor window -> 1-step delta
        adv[r] = delta_l[r] + coef * adv[nb]
    return torch.tensor(adv, dtype=delta.dtype)


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
