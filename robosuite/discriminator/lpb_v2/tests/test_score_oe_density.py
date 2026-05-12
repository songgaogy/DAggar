"""Unit tests for the OE-density score discriminator.

Covers the five required cases from ``PROMPT_oe_density.md`` §6:

1. Synthetic 2-class Gaussians with a planted direction (sanity of the
   end-to-end joint optimisation: the projection should rotate to the planted
   axis and frame AUROC should clear 0.95).
2. ``lam = 0`` regression: passing arbitrary fail data with ``lam = 0`` must
   produce bit-identical gradients to passing ``None`` -- the fail term must
   contribute zero gradient.
3. State-dict round-trip: ``save -> load -> identical .score()``.
4. Disjointness assertion: ``_assert_disjoint`` must raise with overlapping
   ``video_id`` named for every pair.
5. ``first_gt_failure_frame is None`` skip: ``_encode_fail_subset`` skips the
   trajectory and prints a warning that names ``video_id``.

These tests are CPU-only and avoid the encoder pipeline. Tests 4/5 build
lightweight mocks instead of real ``BenchmarkTrajectory`` objects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch

from robosuite.discriminator.lpb_v2.benchmark_oe_density import (
    OEDensityBenchmarkDiscriminator,
)
from robosuite.discriminator.lpb_v2.models.score_oe_density import OEDensityScore


# --------------------------------------------------------------------------- #
# Test 1 — synthetic 2-class Gaussians, planted direction                     #
# --------------------------------------------------------------------------- #


def _frame_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (positive label = 1, larger score = more positive)."""
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Number of positives above each negative.
    cum_pos = np.cumsum(ranked_labels == 1)
    # Sum of ranks of positives.
    auroc = (cum_pos[ranked_labels == 0]).sum() / (n_pos * n_neg)
    # The above counts positives ranked above each negative; equivalent to
    # the classical AUC formula since ties are broken by mergesort stability.
    return float(auroc)


def test_synthetic_planted_direction_recovers_axis():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    D = 16
    T_tasks = 1  # in_dim = D + T = 17
    in_dim = D + T_tasks
    k = 1

    # Success ~ N(0, I_D), fail ~ N(4 e_1, I_D). One-hot column is constant 1.
    # NB: the score s(z) = -log p_theta(f(z)) is symmetric around mu (quadratic
    # distance from the success mean), so the Bayes-optimal AUROC on this
    # setup is bounded by ``2 Phi(|shift| / (sigma sqrt(2))) - 1`` *after*
    # folding -- a 2 sigma shift caps near AUROC=0.85, which is below the
    # 0.95 acceptance threshold. We use a 4 sigma planted shift to put the
    # ceiling above 0.97 while keeping the rest of the setup identical to
    # PROMPT_oe_density.md §6 test 1.
    N_succ = 5000
    N_fail = 200
    z_succ = rng.standard_normal((N_succ, D)).astype(np.float32)
    z_fail = rng.standard_normal((N_fail, D)).astype(np.float32) + np.array(
        [4.0] + [0.0] * (D - 1), dtype=np.float32
    )
    onehot = np.ones((1, T_tasks), dtype=np.float32)
    z_succ_t = np.concatenate([z_succ, np.tile(onehot, (N_succ, 1))], axis=1)
    z_fail_t = np.concatenate([z_fail, np.tile(onehot, (N_fail, 1))], axis=1)

    held_succ = rng.standard_normal((500, D)).astype(np.float32)
    held_fail = rng.standard_normal((500, D)).astype(np.float32) + np.array(
        [4.0] + [0.0] * (D - 1), dtype=np.float32
    )
    held_succ_t = np.concatenate(
        [held_succ, np.tile(onehot, (held_succ.shape[0], 1))], axis=1
    )
    held_fail_t = np.concatenate(
        [held_fail, np.tile(onehot, (held_fail.shape[0], 1))], axis=1
    )

    z_succ_train = torch.from_numpy(z_succ_t)
    z_fail_train = torch.from_numpy(z_fail_t)

    model = OEDensityScore(
        in_dim=in_dim,
        score_k=k,
        density="gaussian",
        weight_decay=1e-4,
        device="cpu",
    )
    model.init_density_from_batch(z_succ_train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)

    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(z_succ_train, z_fail_train, lam=1.0)
        losses["total"].backward()
        optimizer.step()

    # Check alignment of the (only) projection row with e_1 in the original
    # latent (the one-hot dim is task-constant, so it should not dominate).
    W1_row = model.W1.detach().cpu().numpy()[0, :D]
    e1 = np.zeros_like(W1_row)
    e1[0] = 1.0
    cos = abs(float(np.dot(W1_row, e1) / (np.linalg.norm(W1_row) + 1e-12)))
    assert cos > 0.9, f"projection should align with planted direction; got cos={cos:.3f}"

    # Held-out frame AUROC.
    with torch.no_grad():
        s_held = torch.cat(
            [
                model.score(torch.from_numpy(held_succ_t)),
                model.score(torch.from_numpy(held_fail_t)),
            ]
        ).numpy()
    labels = np.concatenate(
        [
            np.zeros((held_succ_t.shape[0],), dtype=np.int64),
            np.ones((held_fail_t.shape[0],), dtype=np.int64),
        ]
    )
    auroc = _frame_auroc(s_held, labels)
    assert auroc > 0.95, f"frame AUROC should clear 0.95 on synthetic; got {auroc:.3f}"


# --------------------------------------------------------------------------- #
# Test 1b — loss stays bounded under long training (regression guard)         #
# --------------------------------------------------------------------------- #


def test_loss_stays_bounded_under_long_training():
    """Regression guard against the anti-density / log-likelihood-ratio form.

    The prior anti-density loss ``L = -E_s log p + lam E_f log p`` is
    mathematically unbounded under a Gaussian (run_20260512_211335 diverged
    to L_total ~ -2e21 by epoch 200). The current logistic loss is bounded
    below by 0; both succ and fail components must stay bounded throughout
    training.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    D, T_tasks, k = 8, 1, 4
    in_dim = D + T_tasks
    onehot = np.ones((1, T_tasks), dtype=np.float32)
    z_succ = rng.standard_normal((512, D)).astype(np.float32)
    z_fail = rng.standard_normal((128, D)).astype(np.float32) + np.array(
        [3.0] + [0.0] * (D - 1), dtype=np.float32
    )
    z_succ_t = torch.from_numpy(
        np.concatenate([z_succ, np.tile(onehot, (z_succ.shape[0], 1))], axis=1)
    )
    z_fail_t = torch.from_numpy(
        np.concatenate([z_fail, np.tile(onehot, (z_fail.shape[0], 1))], axis=1)
    )

    model = OEDensityScore(
        in_dim=in_dim, score_k=k, density="gaussian", weight_decay=1e-4, device="cpu"
    )
    model.init_density_from_batch(z_succ_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)

    L_BOUND = 1e3  # logistic loss should never grow past this on 8-D toy data
    for step in range(500):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(z_succ_t, z_fail_t, lam=1.0)
        losses["total"].backward()
        optimizer.step()
        assert torch.isfinite(losses["total"]).item(), f"L_total NaN/Inf at step {step}"
        assert losses["succ"].item() >= 0.0, "L_succ (softplus) must be >= 0"
        assert losses["fail"].item() >= 0.0, "L_fail (lam * softplus) must be >= 0 for lam >= 0"
        assert abs(losses["total"].item()) < L_BOUND, (
            f"loss should stay bounded; got |L_total|={abs(losses['total'].item()):.2e} "
            f"at step {step}"
        )


# --------------------------------------------------------------------------- #
# Test 2 — lam = 0 yields zero gradient from fail data                        #
# --------------------------------------------------------------------------- #


def test_lam_zero_ignores_fail_data_bitwise():
    torch.manual_seed(0)
    in_dim = 8
    k = 4
    model_a = OEDensityScore(
        in_dim=in_dim, score_k=k, density="gaussian", weight_decay=1e-4, device="cpu"
    )
    z_succ = torch.randn(64, in_dim)
    model_a.init_density_from_batch(z_succ)

    # Two deep-copied models with identical state.
    model_b = copy.deepcopy(model_a)

    # Run A with z_tilde_fail = None.
    losses_a = model_a.loss(z_succ, None, lam=0.0)
    losses_a["total"].backward()
    grads_a = {n: p.grad.detach().clone() for n, p in model_a.named_parameters()}

    # Run B with arbitrary fail data but lam = 0.
    z_fail = torch.randn(20, in_dim) * 10.0  # deliberately huge
    losses_b = model_b.loss(z_succ, z_fail, lam=0.0)
    losses_b["total"].backward()
    grads_b = {n: p.grad.detach().clone() for n, p in model_b.named_parameters()}

    # Bit-identical gradients on every parameter.
    for name in grads_a:
        assert torch.equal(
            grads_a[name], grads_b[name]
        ), f"gradient mismatch on {name!r} between None-fail and arbitrary-fail with lam=0"

    # And the losses themselves are bit-identical.
    assert torch.equal(losses_a["total"].detach(), losses_b["total"].detach())


# --------------------------------------------------------------------------- #
# Test 3 — state-dict round-trip                                              #
# --------------------------------------------------------------------------- #


def test_state_dict_round_trip():
    torch.manual_seed(0)
    in_dim = 12
    k = 4
    model = OEDensityScore(
        in_dim=in_dim, score_k=k, density="gaussian", weight_decay=1e-4, device="cpu"
    )
    z = torch.randn(64, in_dim)
    model.init_density_from_batch(z)

    # Do a few optimizer steps to move parameters away from init.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(z, torch.randn(8, in_dim) * 3.0, lam=1.0)["total"]
        loss.backward()
        optimizer.step()

    z_eval = torch.randn(32, in_dim)
    s_before = model.score(z_eval).clone()

    state = model.state_dict()
    model_loaded = OEDensityScore(
        in_dim=in_dim, score_k=k, density="gaussian", weight_decay=1e-4, device="cpu"
    )
    model_loaded.load_state_dict(state)
    s_after = model_loaded.score(z_eval)
    assert torch.allclose(s_before, s_after, atol=1e-6), (
        f"state-dict round-trip changed score; max abs diff = "
        f"{(s_before - s_after).abs().max().item()}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — disjointness assertion                                             #
# --------------------------------------------------------------------------- #


@dataclass
class _MockTraj:
    video_id: str
    task_name: str = "candy_in_plate"
    is_failure: bool = False
    num_frames: int = 100
    _first_gt: Optional[int] = None

    def first_gt_failure_frame(self) -> Optional[int]:
        return self._first_gt


def test_assert_disjoint_names_all_overlaps():
    eval_trajs = [_MockTraj("E1"), _MockTraj("E2"), _MockTraj("X1"), _MockTraj("X2")]
    ftrain = [_MockTraj("X1"), _MockTraj("F2"), _MockTraj("Y1")]
    fcalib = [_MockTraj("X2"), _MockTraj("Y1"), _MockTraj("C3")]

    with pytest.raises(RuntimeError) as exc:
        OEDensityBenchmarkDiscriminator._assert_disjoint(
            eval_trajs=eval_trajs, ftrain_trajs=ftrain, fcalib_trajs=fcalib
        )
    msg = str(exc.value)
    assert "eval ∩ ftrain" in msg and "X1" in msg
    assert "eval ∩ fail_calib" in msg and "X2" in msg
    assert "ftrain ∩ fail_calib" in msg and "Y1" in msg


def test_assert_disjoint_passes_when_disjoint():
    eval_trajs = [_MockTraj("E1"), _MockTraj("E2")]
    ftrain = [_MockTraj("F1"), _MockTraj("F2")]
    fcalib = [_MockTraj("C1")]
    # Should not raise.
    OEDensityBenchmarkDiscriminator._assert_disjoint(
        eval_trajs=eval_trajs, ftrain_trajs=ftrain, fcalib_trajs=fcalib
    )


# --------------------------------------------------------------------------- #
# Test 5 — _encode_fail_subset skips trajectories with first_gt=None          #
# --------------------------------------------------------------------------- #


class _FakeAdapter:
    """Minimal stub exposing what ``_encode_fail_subset`` needs.

    Avoids the real encoder pipeline; ``_encode`` returns a deterministic
    feature tensor of size ``(num_frames, feat_dim)``.
    """

    def __init__(self, feat_dim: int = 4):
        self.feat_dim = int(feat_dim)

    def _encode(self, trajectory):
        return torch.arange(
            int(trajectory.num_frames) * self.feat_dim, dtype=torch.float32
        ).reshape(int(trajectory.num_frames), self.feat_dim)


def test_encode_fail_subset_skips_when_first_gt_is_none(capsys):
    adapter = _FakeAdapter(feat_dim=4)
    method = OEDensityBenchmarkDiscriminator._encode_fail_subset.__get__(adapter)

    traj_skip = _MockTraj("SKIP_ME", num_frames=100, _first_gt=None)
    out = method(traj_skip, last_k=60, index_ranges=None, warn_tag="ftrain")
    captured = capsys.readouterr().out
    assert out is None
    assert "WARNING" in captured
    assert "SKIP_ME" in captured
    assert "no first_gt_failure_frame" in captured

    # And a trajectory with a valid first_gt produces a non-empty slice.
    index_ranges: dict = {}
    traj_keep = _MockTraj(
        "KEEP_ME", task_name="duck_in_bowl", num_frames=100, _first_gt=50
    )
    sub = method(
        traj_keep, last_k=60, index_ranges=index_ranges, warn_tag="ftrain"
    )
    assert sub is not None
    assert tuple(sub.shape) == (50, 4)  # min(100, 50+60) - 50 = 50 frames
    assert index_ranges == {"duck_in_bowl": {"KEEP_ME": [50, 100]}}
