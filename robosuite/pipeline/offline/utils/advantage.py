"""Offline advantage precomputation for DIPOLE policy training.

Online DIPOLE-RL weights the two flow branches with
``G = alpha * A + disc_weight * r_disc`` where ``A`` is the configured value
signal (:class:`AdvantageGProvider`). The offline variant builds GAE from the
VAST one-macro-step TD residual::

    delta_t = r + gamma^H * (1 - done_t) * V_target(s') - V(s)
    A_t = delta_t + gamma^H * lambda * (1 - done_t) * A_{t+H}

with V read as the single head or independent-ensemble mean, ``gamma^H``
(H=action_horizon) and ``r`` the
chunk-aggregated reward — a drop-in replacement for ``Q - V`` with matching
sign: larger A (better-than-V transition) raises ``w_pos``.

Because the encoder, VAST critics and nnPU head are all **frozen** in offline
training, each valid chunk window's ``A`` and discriminator reward are constant
for the whole run. We therefore pre-encode every window once
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from robosuite.pipeline.algorithms.dipole.advantage_g_provider import AdvantageGProvider

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
    from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
    from robosuite.pipeline.algorithms.vast.common import VASTConfig
    from robosuite.pipeline.algorithms.vast.vast import VASTLearner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VASTAdvantageDiagnostics:
    """Per-window VAST diagnostics aligned with the returned advantage tensor."""

    sampled_k: torch.Tensor
    future_indices: torch.Tensor
    g_values: torch.Tensor
    bootstrap_values: torch.Tensor
    stitched_targets: torch.Tensor
    td1_advantage: torch.Tensor
    gae_advantage: torch.Tensor
    mc_errors: torch.Tensor
    composition_residuals: torch.Tensor
    td1_fallback: torch.Tensor

    @property
    def fallback_fraction(self) -> float:
        if self.td1_fallback.numel() == 0:
            return 0.0
        return float(self.td1_fallback.float().mean().item())


def sample_vast_macro_horizons(
    *,
    valid_starts: list[int],
    episode_indices: list[int],
    horizon: int,
    max_k: int,
    seed: int,
    terminal_starts: set[int] | None = None,
) -> tuple[list[int], list[int], list[bool]]:
    """Sample one reproducible VAST horizon per valid window.

    A stitched policy target requires at least two complete macro chunks.  A
    candidate ``k`` is legal only when every macro start ``s + mH`` for
    ``m=0..k-1`` is a valid window in the same episode.  Windows without a
    legal ``k >= 2`` are marked for the TD1 fallback.
    """
    if len(valid_starts) != len(episode_indices):
        raise ValueError("valid_starts and episode_indices must have equal length.")
    H = int(horizon)
    K = int(max_k)
    if H < 1 or K < 1:
        raise ValueError(f"horizon and max_k must be positive, got H={H}, K={K}.")

    start_to_episode = {
        int(start): int(episode) for start, episode in zip(valid_starts, episode_indices)
    }
    terminal_starts = set() if terminal_starts is None else {int(s) for s in terminal_starts}
    rng = np.random.default_rng(int(seed))
    sampled_k: list[int] = []
    future_indices: list[int] = []
    fallback: list[bool] = []
    for start, episode in zip(valid_starts, episode_indices):
        max_legal = 1
        for k in range(2, K + 1):
            previous_macro_start = int(start) + (k - 2) * H
            if previous_macro_start in terminal_starts:
                break
            last_macro_start = int(start) + (k - 1) * H
            if start_to_episode.get(last_macro_start) != int(episode):
                break
            max_legal = k
        if max_legal < 2:
            sampled_k.append(1)
            future_indices.append(int(start) + H)
            fallback.append(True)
        else:
            k = int(rng.integers(2, max_legal + 1))
            sampled_k.append(k)
            future_indices.append(int(start) + k * H)
            fallback.append(False)
    return sampled_k, future_indices, fallback


@torch.no_grad()
def precompute_offline_advantage(
    *,
    base_buffer: "FlowDaggerReplayBuffer",
    vast_learner: "VASTLearner",
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator",
    vast_cfg: "VASTConfig",
    device: str,
    encode_batch_size: int = 64,
    estimator: str = "td1",
    gae_lambda: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Encode every valid chunk window once and compute its advantage.

    Two ``estimator`` modes (both read the scalar V value via
    :meth:`VASTLearner.compute_td_advantage`, which masks the bootstrap at
    terminal windows with ``(1 - done)``):

    - ``"td1"``: the 1-step (macro-step) TD residual
      ``delta_t = r_chunk + gamma^H*(1-done)*V_target(s') - V(s)``.
    - ``"gae"``: GAE(``gae_lambda``) accumulated backward over each episode's
      chunk chain, ``A_t = delta_t + gamma^H*lambda*(1-done_t)*A_{t+H}``. This
      mirrors the read-out in ``vast/utils/vis_vast.py`` exactly. Offline
      policy sections are stored as standalone short episodes with ``done=True``
      on the last frame, so the recursion stops at the section boundary (no
      cross-section bootstrap).

    Returns ``(advantage_raw, disc_reward_raw, start_to_row)``:
      - ``advantage_raw`` / ``disc_reward_raw``: CUDA float tensors of shape ``(N,)``
        in ``_get_valid_start_indices_locked()`` order;
      - ``start_to_row``: maps a storage start index -> row in those tensors,
        so :class:`OfflineAdvantageGProvider` can gather by the start indices
        that ``DipoleReplayBuffer.sample`` reports.
    """
    # Local import to avoid a heavy import at module load time.
    from robosuite.pipeline.algorithms.vast.replay import VASTReplayBuffer

    replay = VASTReplayBuffer(base_buffer, vast_cfg)
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

    H = int(vast_cfg.action_horizon)
    gamma_h = float(vast_cfg.discount) ** H
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
    disc_reward_parts: list[torch.Tensor] = []
    # Bottleneck is frozen-encoder encode + V/disc forward per window batch.
    # Keep the scalar caches on CUDA so GAE and Phase-B lookups never fall back
    # to CPU tensor computation.
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
        # 1-step (macro-step) TD residual on the scalar V readout, done-masked —
        # identical definition to the online VASTLearner.compute_td_advantage and
        # the vis_vast.py read-out delta.
        delta = vast_learner.compute_td_advantage(
            step_batch.v_state_feature,
            step_batch.next_v_state_feature,
            step_batch.rewards,
            step_batch.dones,
        ).reshape(-1)
        disc_reward = discriminator.intrinsic_reward(
            chunk_feature=step_batch.chunk_feature
        ).reshape(-1)
        delta_parts.append(delta.detach())
        done_parts.append(step_batch.dones.detach().reshape(-1))
        disc_reward_parts.append(disc_reward.detach())

    delta_raw = torch.cat(delta_parts, dim=0).reshape(-1)
    done_raw = torch.cat(done_parts, dim=0).reshape(-1)
    disc_reward_raw = torch.cat(disc_reward_parts, dim=0).reshape(-1)
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
        "(gamma^H=%.4f, lambda=%.3f, adv_mean=%.4f adv_std=%.4f, "
        "disc_reward_mean=%.4f)",
        estimator.upper(),
        advantage_raw.numel(),
        gamma_h,
        lam if estimator == "gae" else float("nan"),
        float(advantage_raw.mean().item()),
        float(advantage_raw.std().item()) if advantage_raw.numel() > 1 else 0.0,
        float(disc_reward_raw.mean().item()),
    )
    return advantage_raw, disc_reward_raw, start_to_row


@torch.no_grad()
def precompute_vast_offline_advantage(
    *,
    base_buffer: "FlowDaggerReplayBuffer",
    vast_learner: "VASTLearner",
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator",
    vast_cfg: "VASTConfig",
    device: str,
    encode_batch_size: int = 64,
    sampling_seed: int | None = None,
    gae_lambda: float = 0.6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[int, int],
    VASTAdvantageDiagnostics,
]:
    """Precompute fixed-seed stitched advantages with a TD1 tail fallback.

    The sampled horizon is fixed for the lifetime of the static policy replay.
    Stitched windows use ``G(s, s_k, k) + gamma^(kH) V_target(s_k) - V(s)``;
    windows with fewer than two complete macro chunks retain the legacy TD1
    residual.  TD1 and GAE are always retained as diagnostics.
    """
    from robosuite.pipeline.algorithms.vast.replay import (
        VASTReplayBuffer,
        _stack_views_uint8,
        _to_image_tensor,
    )

    replay = VASTReplayBuffer(base_buffer, vast_cfg)
    with base_buffer._lock:  # noqa: SLF001
        valid_starts = list(base_buffer._get_valid_start_indices_locked())  # noqa: SLF001
        episode_indices = [
            int((base_buffer._storage[s].info or {}).get("episode_index", -1))  # noqa: SLF001
            for s in valid_starts
        ]
    if not valid_starts:
        raise ValueError("precompute_vast_offline_advantage: no valid chunk windows.")

    H = int(vast_cfg.action_horizon)
    K = int(vast_cfg.vast_max_k)
    seed = int(vast_cfg.vast_sampling_seed if sampling_seed is None else sampling_seed)
    gamma_h = float(vast_cfg.discount) ** H
    encode_bs = max(1, int(encode_batch_size))
    with base_buffer._lock:  # noqa: SLF001
        terminal_starts = {
            start
            for start in valid_starts
            if replay._raw_chunk_done_locked(start)  # noqa: SLF001
        }
    sampled_k, future_indices, fallback_flags = sample_vast_macro_horizons(
        valid_starts=valid_starts,
        episode_indices=episode_indices,
        horizon=H,
        max_k=K,
        seed=seed,
        terminal_starts=terminal_starts,
    )
    start_to_row = {int(start): row for row, start in enumerate(valid_starts)}

    state_parts: list[torch.Tensor] = []
    next_state_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    disc_reward_parts: list[torch.Tensor] = []
    for offset in tqdm(
        range(0, len(valid_starts), encode_bs),
        total=(len(valid_starts) + encode_bs - 1) // encode_bs,
        desc="[offline] precompute VAST base windows",
        unit="batch",
    ):
        starts = valid_starts[offset : offset + encode_bs]
        with base_buffer._lock:  # noqa: SLF001
            sequences = [base_buffer._storage[s : s + H] for s in starts]  # noqa: SLF001
        batch = replay._build_step_batch(  # noqa: SLF001
            sequences,
            starts,
            encoder=encoder,
            discriminator=discriminator,
            device=device,
        )
        state_parts.append(batch.v_state_feature.detach().cpu())
        next_state_parts.append(batch.next_v_state_feature.detach().cpu())
        reward_parts.append(batch.rewards.detach().cpu().reshape(-1))
        done_parts.append(batch.dones.detach().cpu().reshape(-1))
        disc_reward_parts.append(
            discriminator.intrinsic_reward(chunk_feature=batch.chunk_feature)
            .detach()
            .cpu()
            .reshape(-1)
        )

    state_raw = torch.cat(state_parts, dim=0)
    next_state_raw = torch.cat(next_state_parts, dim=0)
    macro_rewards = torch.cat(reward_parts, dim=0)
    macro_dones = torch.cat(done_parts, dim=0)
    disc_reward_raw = torch.cat(disc_reward_parts, dim=0)

    # Encode only the selected future state for each row.  `_next_obs_for` on
    # the last macro chunk also handles a trajectory endpoint consistently with
    # the learner replay semantics.
    camera_names = list(base_buffer.camera_names)
    future_parts: list[torch.Tensor] = []
    for offset in tqdm(
        range(0, len(valid_starts), encode_bs),
        total=(len(valid_starts) + encode_bs - 1) // encode_bs,
        desc="[offline] encode VAST future states",
        unit="batch",
    ):
        rows = range(offset, min(offset + encode_bs, len(valid_starts)))
        future_obs: list[Any] = []
        with base_buffer._lock:  # noqa: SLF001
            for row in rows:
                last_macro_start = valid_starts[row] + (sampled_k[row] - 1) * H
                obs, _ = replay._next_obs_for(last_macro_start)  # noqa: SLF001
                future_obs.append(obs)
        image_np = np.stack(
            [_stack_views_uint8(obs, camera_names) for obs in future_obs], axis=0
        )
        proprio_np = np.stack(
            [np.asarray(obs["state"], dtype=np.float32) for obs in future_obs], axis=0
        )
        future_feature = encoder.encode_state(
            image_obs_raw=_to_image_tensor(image_np, device),
            proprio_raw=torch.from_numpy(np.ascontiguousarray(proprio_np)).float(),
        )
        future_parts.append(future_feature.detach().cpu())
    future_state_raw = torch.cat(future_parts, dim=0)

    td1_parts: list[torch.Tensor] = []
    stitched_adv_parts: list[torch.Tensor] = []
    g_parts: list[torch.Tensor] = []
    bootstrap_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    mc_error_parts: list[torch.Tensor] = []
    comp_parts: list[torch.Tensor] = []
    future_done_values: list[float] = []
    k_return_values: list[float] = []
    for row, (start, k) in enumerate(zip(valid_starts, sampled_k)):
        span_rows = [start_to_row[start + m * H] for m in range(k)]
        future_done = float(macro_dones[span_rows].max().item())
        future_done_values.append(future_done)
        k_return_values.append(
            sum((gamma_h**m) * float(macro_rewards[r].item()) for m, r in enumerate(span_rows))
        )

    future_done_raw = torch.tensor(future_done_values, dtype=torch.float32)
    k_return_raw = torch.tensor(k_return_values, dtype=torch.float32)
    sampled_k_raw = torch.tensor(sampled_k, dtype=torch.long)
    fallback_raw = torch.tensor(fallback_flags, dtype=torch.bool)

    for offset in range(0, len(valid_starts), encode_bs):
        sl = slice(offset, min(offset + encode_bs, len(valid_starts)))
        s = state_raw[sl].to(device)
        sk = future_state_raw[sl].to(device)
        k = sampled_k_raw[sl].to(device=device, dtype=torch.float32).unsqueeze(-1)
        done = future_done_raw[sl].to(device).unsqueeze(-1)
        rewards = macro_rewards[sl].to(device)
        one_done = macro_dones[sl].to(device)

        td1 = vast_learner.compute_td_advantage(
            s,
            next_state_raw[sl].to(device),
            rewards,
            one_done,
        )
        stitched = vast_learner.compute_stitched_advantage(s, sk, k, done)
        g = vast_learner.g_value(s, sk, k).reshape(-1)
        bootstrap = vast_learner.target_v_value(sk).reshape(-1)
        discount = torch.pow(
            torch.full_like(k.reshape(-1), float(vast_cfg.discount)),
            k.reshape(-1) * H,
        )
        target = g + discount * (1.0 - done.reshape(-1)) * bootstrap

        # Composition diagnostics use deterministic j=floor(k/2); fallback
        # rows have k=1 and therefore no valid composition residual.
        comp = torch.full_like(g, float("nan"))
        for local, global_row in enumerate(range(sl.start, sl.stop)):
            ki = sampled_k[global_row]
            if ki < 2:
                continue
            ji = ki // 2
            mid_row = start_to_row[valid_starts[global_row] + ji * H]
            mid = state_raw[mid_row : mid_row + 1].to(device)
            lhs = g[local : local + 1]
            rhs = vast_learner.g_value(
                s[local : local + 1],
                mid,
                torch.tensor([[float(ji)]], device=device),
            ).reshape(-1)
            rhs = rhs + (float(vast_cfg.discount) ** (ji * H)) * vast_learner.g_value(
                mid,
                sk[local : local + 1],
                torch.tensor([[float(ki - ji)]], device=device),
            ).reshape(-1)
            comp[local] = (lhs - rhs).item()

        td1_parts.append(td1.detach().cpu())
        stitched_adv_parts.append(stitched.detach().cpu())
        g_parts.append(g.detach().cpu())
        bootstrap_parts.append(bootstrap.detach().cpu())
        target_parts.append(target.detach().cpu())
        mc_error_parts.append((g - k_return_raw[sl].to(device)).detach().cpu())
        comp_parts.append(comp.detach().cpu())

    td1_raw = torch.cat(td1_parts).reshape(-1)
    stitched_raw = torch.cat(stitched_adv_parts).reshape(-1)
    advantage_raw = torch.where(fallback_raw, td1_raw, stitched_raw)
    gae_raw = _gae_over_episodes(
        delta=td1_raw.to(device),
        done=macro_dones.to(device),
        valid_starts=valid_starts,
        episode_indices=episode_indices,
        start_to_row=start_to_row,
        horizon=H,
        gamma_h=gamma_h,
        lam=float(gae_lambda),
    )
    diagnostics = VASTAdvantageDiagnostics(
        sampled_k=sampled_k_raw,
        future_indices=torch.tensor(future_indices, dtype=torch.long),
        g_values=torch.cat(g_parts).reshape(-1),
        bootstrap_values=torch.cat(bootstrap_parts).reshape(-1),
        stitched_targets=torch.cat(target_parts).reshape(-1),
        td1_advantage=td1_raw,
        gae_advantage=gae_raw.cpu(),
        mc_errors=torch.cat(mc_error_parts).reshape(-1),
        composition_residuals=torch.cat(comp_parts).reshape(-1),
        td1_fallback=fallback_raw,
    )
    logger.info(
        "[offline] precomputed VAST stitched advantage for %d windows "
        "(K=%d seed=%d fallback=%.2f%% adv_mean=%.4f "
        "disc_reward_mean=%.4f)",
        len(valid_starts),
        K,
        seed,
        100.0 * diagnostics.fallback_fraction,
        float(advantage_raw.mean().item()),
        float(disc_reward_raw.mean().item()),
    )
    return advantage_raw, disc_reward_raw, start_to_row, diagnostics


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
    falls back to the 1-step ``delta_t`` — matching vis_vast's boundary ``0``).
    Rows are grouped by reverse dependency depth so every successor is finalized
    before its predecessors. Arithmetic and indexing stay on CUDA.
    """
    delta_flat = delta.reshape(-1)
    done_flat = done.reshape(-1)
    if not delta_flat.is_cuda or not done_flat.is_cuda:
        raise RuntimeError("Offline GAE requires CUDA tensors; CPU fallback is disabled.")
    if done_flat.device != delta_flat.device:
        raise ValueError(
            f"GAE delta/done device mismatch: {delta_flat.device} vs {done_flat.device}."
        )
    if delta_flat.numel() != len(valid_starts) or done_flat.numel() != len(valid_starts):
        raise ValueError(
            "GAE delta, done, valid_starts, and episode_indices must have equal length."
        )
    if len(episode_indices) != len(valid_starts):
        raise ValueError(
            "GAE delta, done, valid_starts, and episode_indices must have equal length."
        )

    successors = [-1] * len(valid_starts)
    depths = [0] * len(valid_starts)
    order = sorted(range(len(valid_starts)), key=lambda r: valid_starts[r], reverse=True)
    for r in order:
        nb = start_to_row.get(int(valid_starts[r]) + int(horizon))
        if nb is None or episode_indices[nb] != episode_indices[r]:
            continue
        successors[r] = int(nb)
        depths[r] = depths[nb] + 1

    adv = delta_flat.clone()
    coef = float(gamma_h) * float(lam)
    rows_by_depth: list[list[int]] = [
        [] for _ in range(max(depths, default=0) + 1)
    ]
    for row, depth in enumerate(depths):
        if depth > 0:
            rows_by_depth[depth].append(row)
    for rows in rows_by_depth[1:]:
        row_idx = torch.tensor(rows, dtype=torch.long, device=delta_flat.device)
        successor_idx = torch.tensor(
            [successors[r] for r in rows],
            dtype=torch.long,
            device=delta_flat.device,
        )
        values = delta_flat.index_select(0, row_idx) + coef * (
            1.0 - done_flat.index_select(0, row_idx)
        ) * adv.index_select(0, successor_idx)
        adv.index_copy_(0, row_idx, values)
    return adv


class OfflineAdvantageGProvider(AdvantageGProvider):
    """Serve precomputed TD1/GAE advantage by static-buffer start index.

    Reuses the parent's ``bind_policy_cameras`` but replaces
    ``compute_g_for_batch`` with a cache lookup keyed on
    ``batch.metadata["start_indices"]``.
    """

    def __init__(
        self,
        *,
        vast_learner: "VASTLearner",
        discriminator: "FrozenNNPUDiscriminator",
        encoder: "SharedDynamicsEncoder",
        alpha: float,
        disc_weight: float,
        advantage_raw: torch.Tensor,
        disc_reward_raw: torch.Tensor,
        start_to_row: dict[int, int],
    ) -> None:
        super().__init__(
            vast_learner=vast_learner,
            discriminator=discriminator,
            encoder=encoder,
            alpha=alpha,
            disc_weight=disc_weight,
        )
        advantage_raw = advantage_raw.detach().reshape(-1)
        disc_reward_raw = disc_reward_raw.detach().reshape(-1)
        if not advantage_raw.is_cuda or not disc_reward_raw.is_cuda:
            raise RuntimeError(
                "OfflineAdvantageGProvider requires CUDA advantage/disc-reward caches; "
                "CPU fallback is disabled."
            )
        if advantage_raw.device != disc_reward_raw.device:
            raise ValueError(
                "OfflineAdvantageGProvider advantage/disc-reward device mismatch: "
                f"{advantage_raw.device} vs {disc_reward_raw.device}."
            )
        self._advantage_raw = advantage_raw
        self._disc_reward_raw = disc_reward_raw
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

        device = batch.action_sequences_raw.device
        if device != self._advantage_raw.device:
            raise RuntimeError(
                "Offline advantage cache and policy batch must share one CUDA device: "
                f"cache={self._advantage_raw.device}, batch={device}."
            )
        row_idx = torch.tensor(rows, dtype=torch.long, device=device)
        advantage = self._advantage_raw.index_select(0, row_idx)
        disc_reward = self._disc_reward_raw.index_select(0, row_idx)

        g = self.alpha * advantage + self.disc_weight * disc_reward
        return g.reshape(-1)


__all__ = [
    "OfflineAdvantageGProvider",
    "VASTAdvantageDiagnostics",
    "precompute_offline_advantage",
    "precompute_vast_offline_advantage",
    "sample_vast_macro_horizons",
]
