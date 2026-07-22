"""Unit tests for the offline GAE(lambda) advantage recursion.

CUDA-only: exercises ``_gae_over_episodes`` (the pure backward
recursion that turns per-window 1-step TD residuals into a GAE advantage) in
isolation — no encoder, discriminator, buffer, or VAST learner. Covers:
    - a hand-computed closed-form chain (multi-hop accumulation),
    - the ``done`` terminal window stopping propagation,
    - the episode-boundary guard: a numeric ``start + H`` successor that belongs
      to a *different* episode must not be chained,
    - agreement with an independent forward-expansion brute force on a random
      contiguous multi-episode scenario.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.offline import (
    nnpu_intrinsic_from_failure_score,
)
from robosuite.pipeline.offline.utils.advantage import (
    OfflineAdvantageGProvider,
    _gae_over_episodes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Offline tensor tests require CUDA and never fall back to CPU.",
)
DEVICE = torch.device("cuda")


def test_discriminator_reward_is_negative_failure_probability_cuda():
    failure_score = torch.tensor([-1.0, 0.5, 2.0], device=DEVICE)
    expected = -torch.sigmoid(failure_score - 0.5)

    reward = nnpu_intrinsic_from_failure_score(failure_score, threshold=0.5)

    torch.testing.assert_close(reward, expected)


def _brute_gae(delta, done, starts, ep, horizon, coef):
    """Forward-expansion reference: A_t = sum_k coef^k * delta_{t+kH}.

    Expands ``A_t = delta_t + coef*(1-done_t)*A_{t+H}`` by walking successors
    until a terminal window (``done``) or no in-episode ``+H`` successor.
    """
    row = {int(s): i for i, s in enumerate(starts)}
    out = []
    for i in range(len(starts)):
        acc = torch.zeros((), dtype=delta.dtype, device=delta.device)
        c = 1.0
        cur = i
        while True:
            acc = acc + c * delta[cur]
            if done[cur] > 0.5:
                break
            nb = row.get(int(starts[cur]) + int(horizon))
            if nb is None or ep[nb] != ep[cur]:
                break
            c *= coef
            cur = nb
        out.append(acc)
    return torch.stack(out)


def test_gae_closed_form_terminal_and_episode_boundary():
    # Two episodes, H=2. ep0 windows at starts 10,12,14; ep1 at 16,18.
    # Note 14 + H == 16 is present numerically but belongs to ep1 -> the guard
    # must exclude it (14 has no valid in-episode successor).
    starts = [10, 12, 14, 16, 18]
    ep = [0, 0, 0, 1, 1]
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)
    done = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], device=DEVICE)
    start_to_row = {s: i for i, s in enumerate(starts)}
    gamma_h, lam = 0.5, 0.6
    coef = gamma_h * lam  # 0.3

    adv = _gae_over_episodes(
        delta=delta,
        done=done,
        valid_starts=starts,
        episode_indices=ep,
        start_to_row=start_to_row,
        horizon=2,
        gamma_h=gamma_h,
        lam=lam,
    )

    # A[18]=5 ; A[16]=4+0.3*5=5.5 ; A[14]=3 (successor 16 is ep1 -> excluded)
    # A[12]=2+0.3*3=2.9 ; A[10]=1+0.3*2.9=1.87
    expected = torch.tensor([1.87, 2.9, 3.0, 5.5, 5.0], device=DEVICE)
    assert adv.device.type == "cuda"
    assert torch.allclose(adv, expected, atol=1e-6), adv


def test_gae_missing_horizon_successor_does_not_recurse():
    # With H=2, start=0 would recurse only through a start=2 row. A later row at
    # start=3 must not be treated as a successor even though it is in the same
    # episode.
    starts = [0, 1, 3]
    delta = torch.tensor([1.0, 2.0, 100.0], device=DEVICE)
    done = torch.zeros(3, device=DEVICE)

    adv = _gae_over_episodes(
        delta=delta,
        done=done,
        valid_starts=starts,
        episode_indices=[0, 0, 0],
        start_to_row={s: i for i, s in enumerate(starts)},
        horizon=2,
        gamma_h=0.9,
        lam=0.8,
    )

    assert adv.device.type == "cuda"
    expected = torch.tensor([1.0, 74.0, 100.0], device=DEVICE)
    assert torch.allclose(adv, expected, atol=1e-7), adv


def test_gae_matches_brute_force_random_scenario():
    torch.manual_seed(0)
    horizon = 3
    gamma_h, lam = 0.9227, 0.6
    coef = gamma_h * lam

    # Three contiguous episodes of varying length; done=1 on the last window of
    # each episode (terminal), 0 elsewhere.
    starts: list[int] = []
    ep: list[int] = []
    done_list: list[float] = []
    cursor = 0
    for e, length in enumerate((7, 4, 9)):
        for j in range(length):
            starts.append(cursor)
            ep.append(e)
            done_list.append(1.0 if j == length - 1 else 0.0)
            cursor += 1
        cursor += 5  # gap so episodes are not storage-adjacent
    delta = torch.randn(len(starts), device=DEVICE)
    done = torch.tensor(done_list, device=DEVICE)
    start_to_row = {s: i for i, s in enumerate(starts)}

    adv = _gae_over_episodes(
        delta=delta,
        done=done,
        valid_starts=starts,
        episode_indices=ep,
        start_to_row=start_to_row,
        horizon=horizon,
        gamma_h=gamma_h,
        lam=lam,
    )
    ref = _brute_gae(delta, done_list, starts, ep, horizon, coef)
    assert torch.allclose(adv, ref, atol=1e-5), (adv - ref).abs().max()


def test_gae_lambda_zero_recovers_one_step_delta():
    starts = [0, 1, 2]
    ep = [0, 0, 0]
    delta = torch.tensor([1.0, -2.0, 3.0], device=DEVICE)
    done = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
    adv = _gae_over_episodes(
        delta=delta,
        done=done,
        valid_starts=starts,
        episode_indices=ep,
        start_to_row={s: i for i, s in enumerate(starts)},
        horizon=1,
        gamma_h=0.9,
        lam=0.0,
    )
    assert torch.allclose(adv, delta, atol=1e-7), adv


def test_offline_advantage_provider_cuda_cache_lookup_stays_on_cuda():
    provider = OfflineAdvantageGProvider(
        vast_learner=None,
        discriminator=None,
        encoder=None,
        alpha=2.0,
        disc_weight=0.25,
        advantage_raw=torch.tensor([1.0, -2.0, 3.0], device=DEVICE),
        disc_reward_raw=torch.tensor([-0.5, -1.0, -0.25], device=DEVICE),
        start_to_row={10: 0, 20: 1, 30: 2},
    )
    batch = SimpleNamespace(
        metadata={"start_indices": [30, 10]},
        action_sequences_raw=torch.empty(2, 1, 1, device=DEVICE),
    )

    g = provider.compute_g_for_batch(batch)

    assert provider._advantage_raw.device.type == "cuda"
    assert provider._disc_reward_raw.device.type == "cuda"
    assert g.device.type == "cuda"
    expected = torch.tensor([5.9375, 1.875], device=DEVICE)
    assert torch.allclose(g, expected, atol=1e-7), g


def test_offline_advantage_provider_disc_weight_zero_is_pure_advantage():
    provider = OfflineAdvantageGProvider(
        vast_learner=None,
        discriminator=None,
        encoder=None,
        alpha=1.5,
        disc_weight=0.0,
        advantage_raw=torch.tensor([2.0], device=DEVICE),
        disc_reward_raw=torch.tensor([-1.0], device=DEVICE),
        start_to_row={7: 0},
    )
    batch = SimpleNamespace(
        metadata={"start_indices": [7]},
        action_sequences_raw=torch.empty(1, 1, 1, device=DEVICE),
    )

    g = provider.compute_g_for_batch(batch)

    torch.testing.assert_close(g, torch.tensor([3.0], device=DEVICE))
