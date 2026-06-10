"""Synthetic-tensor tests for the IQL Q-chunking module.

CPU-only; no encoder, no env. Exercises:
    - `compute_advantage` shape and value.
    - `expectile_v_loss` at tau=0.5 reducing to 0.5·MSE.
    - `bellman_q_loss` matches MSE.
    - One `IQLLearner.update` step and one `warmup_value_only` step run
      forward+backward without NaNs and actually move parameters.
    - `state_dict / load_state_dict` round-trip on a fresh learner.
    - `aggregate_chunk_reward` matches the closed-form geometric sum.
    - `chunk_done_mask` reduces along H.
"""

from __future__ import annotations

import math

import pytest
import torch

from robosuite.pipeline.algorithms.q_learning.common import IQLConfig, IQLStepBatch
from robosuite.pipeline.algorithms.q_learning.data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
)
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.losses import (
    bellman_q_loss,
    compute_advantage,
    compute_ensemble_advantage,
    expectile_v_loss,
)


def _make_cfg(action_horizon: int = 2) -> IQLConfig:
    return IQLConfig(
        action_horizon=action_horizon,
        discount=0.9,
        expectile_tau=0.7,
        q_lr=1e-3,
        v_lr=1e-3,
        target_polyak=0.1,
        n_step_aggregate=True,
        hidden_dims=(32, 32),
        q_ensemble_size=5,
        v_subset_size=2,
        grad_clip_norm=1.0,
        weight_decay=0.0,
        device="cpu",
        disc_reward_coef=0.0,
        output_reward_coef=1.0,
    )


def _make_step_batch(B: int = 8, D_ctx: int = 16, D_a: int = 4, H: int = 2) -> IQLStepBatch:
    torch.manual_seed(0)
    return IQLStepBatch(
        context=torch.randn(B, D_ctx),
        next_context=torch.randn(B, D_ctx),
        action_chunk=torch.randn(B, H, D_a),
        rewards=torch.randn(B, 1),
        dones=torch.zeros(B, 1),
        is_online=torch.zeros(B, 1),
        is_intervention=torch.zeros(B, 1),
        metadata={},
    )


def test_compute_advantage_shape_and_value() -> None:
    q1 = torch.tensor([[1.0], [2.0], [3.0]])
    q2 = torch.tensor([[2.0], [1.0], [4.0]])
    v = torch.tensor([[0.5], [0.5], [0.5]])
    adv = compute_advantage(q1, q2, v)
    assert adv.shape == (3,)
    assert torch.allclose(adv, torch.tensor([0.5, 0.5, 2.5]))


def test_compute_ensemble_advantage_uses_q_mean() -> None:
    q_values = torch.tensor(
        [
            [[1.0], [3.0]],
            [[2.0], [5.0]],
            [[4.0], [7.0]],
        ]
    )
    v = torch.tensor([[0.5], [1.0]])
    adv = compute_ensemble_advantage(q_values, v)
    assert adv.shape == (2,)
    assert torch.allclose(adv, torch.tensor([7.0 / 3.0 - 0.5, 5.0 - 1.0]))


def test_expectile_v_loss_tau_half_is_quarter_mse() -> None:
    diff = torch.tensor([[1.0], [-2.0], [0.5]])
    # tau = 0.5 -> weight = 0.5 everywhere; loss = 0.5 * mean(diff^2)
    expected = 0.5 * diff.square().mean()
    actual = expectile_v_loss(diff, tau=0.5)
    assert torch.allclose(actual, expected)


def test_expectile_v_loss_asymmetry() -> None:
    diff = torch.tensor([[1.0], [-1.0]])
    tau = 0.7
    # positive diff contributes 0.7 * 1; negative diff contributes 0.3 * 1; mean = 0.5
    actual = expectile_v_loss(diff, tau=tau)
    assert torch.allclose(actual, torch.tensor(0.5))


def test_bellman_q_loss_matches_mse() -> None:
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.5], [1.0], [4.0]])
    assert torch.allclose(bellman_q_loss(pred, target), torch.nn.functional.mse_loss(pred, target))


def test_bellman_q_loss_sums_ensemble_critics() -> None:
    pred = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[2.0], [4.0], [6.0]],
        ]
    )
    target = torch.tensor([[1.5], [1.0], [4.0]])
    expected = (
        torch.nn.functional.mse_loss(pred[0], target)
        + torch.nn.functional.mse_loss(pred[1], target)
    )
    assert torch.allclose(bellman_q_loss(pred, target), expected)


def test_bellman_q_loss_weighting_matches_manual() -> None:
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.5], [1.0], [4.0]])
    weights = torch.tensor([[1.0], [1.0], [10.0]])
    sq = (pred - target).square()
    expected = (sq * weights).sum() / weights.sum()
    assert torch.allclose(bellman_q_loss(pred, target, weights=weights), expected)
    # All-ones weights must reduce to the plain (unweighted) mean.
    ones = torch.ones_like(target)
    assert torch.allclose(
        bellman_q_loss(pred, target, weights=ones), bellman_q_loss(pred, target)
    )


def test_value_target_blends_td_and_mc() -> None:
    cfg = _make_cfg(action_horizon=2)
    cfg.mc_blend_lambda = 0.25
    iql = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    batch = _make_step_batch(B=8, D_ctx=16, D_a=4, H=2)
    batch.mc_return = torch.full((8, 1), 0.5)
    td = iql._bootstrap_target(batch)
    expected = 0.75 * td + 0.25 * batch.mc_return
    assert torch.allclose(iql._value_target(batch), expected)
    # lambda == 0 => pure TD; mc_return None => pure TD.
    cfg.mc_blend_lambda = 0.0
    assert torch.allclose(iql._value_target(batch), td)
    cfg.mc_blend_lambda = 0.25
    batch.mc_return = None
    assert torch.allclose(iql._value_target(batch), td)


def test_aggregate_chunk_reward_closed_form() -> None:
    H = 4
    discount = 0.9
    step_rewards = torch.ones(1, H)  # constant reward 1
    out = aggregate_chunk_reward(step_rewards, discount)
    expected = sum(discount ** i for i in range(H))
    assert out.shape == (1, 1)
    assert math.isclose(float(out.item()), expected, rel_tol=1e-6)


def test_chunk_done_mask() -> None:
    step_dones = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    out = chunk_done_mask(step_dones)
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[1.0], [0.0]]))


def test_iql_update_step_runs_and_moves_params() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    batch = _make_step_batch(B=8, D_ctx=16, D_a=4, H=2)

    # Step v away from its zero-init final layer using one warmup pass so the
    # initial degenerate case (q_min == v_pred == 0 => expectile_v_loss == 0)
    # doesn't mask the V update.
    iql.warmup_value_only(batch)

    snapshot_q = [
        [p.detach().clone() for p in critic.parameters()]
        for critic in iql.q_ensemble
    ]
    snapshot_v = [p.detach().clone() for p in iql.v.parameters()]

    metrics = iql.update(batch)
    assert all(math.isfinite(v) for v in metrics.values()), f"non-finite metric: {metrics}"

    moved_q = [
        any(
            not torch.equal(before, after)
            for before, after in zip(snapshot, critic.parameters())
        )
        for snapshot, critic in zip(snapshot_q, iql.q_ensemble)
    ]
    moved_v = any(
        not torch.equal(before, after)
        for before, after in zip(snapshot_v, iql.v.parameters())
    )
    assert all(moved_q), f"not every Q critic moved after update: {moved_q}"
    assert moved_v, "v parameters did not move after update"


def test_iql_warmup_value_only_runs() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    batch = _make_step_batch(B=8, D_ctx=16, D_a=4, H=2)
    snapshot_q = [
        [p.detach().clone() for p in critic.parameters()]
        for critic in iql.q_ensemble
    ]
    snapshot_v = [p.detach().clone() for p in iql.v.parameters()]
    metrics = iql.warmup_value_only(batch)
    assert all(math.isfinite(v) for v in metrics.values()), f"non-finite metric: {metrics}"
    moved_v = any(
        not torch.equal(before, after)
        for before, after in zip(snapshot_v, iql.v.parameters())
    )
    assert moved_v, "v parameters did not move during warmup"
    # Q must NOT move during warmup-value-only.
    untouched_q = [
        all(
            torch.equal(before, after)
            for before, after in zip(snapshot, critic.parameters())
        )
        for snapshot, critic in zip(snapshot_q, iql.q_ensemble)
    ]
    assert all(untouched_q), "Q parameters moved during warmup_value_only (should be V-only)"


def test_iql_state_dict_roundtrip() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql_a = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    iql_b = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    # Run one step on A so its weights diverge from B's fresh init.
    batch = _make_step_batch(B=4, D_ctx=16, D_a=4, H=2)
    iql_a.update(batch)
    sd = iql_a.state_dict()
    iql_b.load_state_dict(sd, strict=True)
    for qa, qb in zip(iql_a.q_ensemble, iql_b.q_ensemble):
        for pa, pb in zip(qa.parameters(), qb.parameters()):
            assert torch.equal(pa, pb)
    for pa, pb in zip(iql_a.v.parameters(), iql_b.v.parameters()):
        assert torch.equal(pa, pb)


def test_iql_load_state_dict_mismatched_context_dim_raises() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql_a = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    iql_c = IQLLearner(cfg=cfg, context_dim=8, action_dim=4)  # mismatched context_dim
    sd = iql_a.state_dict()
    with pytest.raises(ValueError, match="context_dim mismatch"):
        iql_c.load_state_dict(sd, strict=True)


def test_iql_old_two_q_checkpoint_schema_raises() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql_a = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    iql_b = IQLLearner(cfg=cfg, context_dim=16, action_dim=4)
    sd = iql_a.state_dict()
    old_sd = dict(sd)
    old_sd.pop("q_ensemble")
    old_sd.pop("q_ensemble_size")
    old_sd.pop("v_subset_size")
    old_sd["q1"] = iql_a.q1.state_dict()
    old_sd["q2"] = iql_a.q2.state_dict()
    with pytest.raises(ValueError, match="old two-Q schema"):
        iql_b.load_state_dict(old_sd, strict=True)
