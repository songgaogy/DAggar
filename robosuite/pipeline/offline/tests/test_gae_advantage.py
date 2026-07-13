"""Unit tests for the offline GAE(lambda) advantage recursion.

CPU-only, dependency-free: exercises ``_gae_over_episodes`` (the pure backward
recursion that turns per-window 1-step TD residuals into a GAE advantage) in
isolation — no encoder, discriminator, buffer, or IQL learner. Covers:
    - a hand-computed closed-form chain (multi-hop accumulation),
    - the ``done`` terminal window stopping propagation,
    - the episode-boundary guard: a numeric ``start + H`` successor that belongs
      to a *different* episode must not be chained,
    - agreement with an independent forward-expansion brute force on a random
      contiguous multi-episode scenario.
"""

from __future__ import annotations

import torch

from robosuite.pipeline.offline.utils.advantage import _gae_over_episodes


def _brute_gae(delta, done, starts, ep, horizon, coef):
    """Forward-expansion reference: A_t = sum_k coef^k * delta_{t+kH}.

    Expands ``A_t = delta_t + coef*(1-done_t)*A_{t+H}`` by walking successors
    until a terminal window (``done``) or no in-episode ``+H`` successor.
    """
    row = {int(s): i for i, s in enumerate(starts)}
    out = []
    for i in range(len(starts)):
        acc = 0.0
        c = 1.0
        cur = i
        while True:
            acc += c * float(delta[cur])
            if float(done[cur]) > 0.5:
                break
            nb = row.get(int(starts[cur]) + int(horizon))
            if nb is None or ep[nb] != ep[cur]:
                break
            c *= coef
            cur = nb
        out.append(acc)
    return out


def test_gae_closed_form_terminal_and_episode_boundary():
    # Two episodes, H=2. ep0 windows at starts 10,12,14; ep1 at 16,18.
    # Note 14 + H == 16 is present numerically but belongs to ep1 -> the guard
    # must exclude it (14 has no valid in-episode successor).
    starts = [10, 12, 14, 16, 18]
    ep = [0, 0, 0, 1, 1]
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    done = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
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
    expected = torch.tensor([1.87, 2.9, 3.0, 5.5, 5.0])
    assert torch.allclose(adv, expected, atol=1e-6), adv


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
    delta = torch.randn(len(starts))
    done = torch.tensor(done_list)
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
    ref = torch.tensor(_brute_gae(delta, done, starts, ep, horizon, coef))
    assert torch.allclose(adv, ref, atol=1e-5), (adv - ref).abs().max()


def test_gae_lambda_zero_recovers_one_step_delta():
    starts = [0, 1, 2]
    ep = [0, 0, 0]
    delta = torch.tensor([1.0, -2.0, 3.0])
    done = torch.tensor([0.0, 0.0, 0.0])
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
