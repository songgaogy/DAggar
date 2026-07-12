"""Synthetic-tensor tests for the V-only IQL module.

CPU-only; no encoder, no env. Exercises:
    - `expectile_v_loss` at tau=0.5 reducing to 0.5·MSE.
    - `VEnsemble` forward shape and the `V_lcb = mean - beta*std` reduction
      (N=1 ⇒ std=0 ⇒ lcb==mean; N=2 ⇒ lcb below mean by beta*std).
    - One `IQLLearner.update` (V-only expectile-TD) step runs forward+backward
      without NaNs, moves V, and diversifies the heads (v_std_mean > 0).
    - `_bootstrap_target` / `compute_td_advantage` use the soft-LCB value.
    - `state_dict / load_state_dict` round-trip on a fresh learner.
    - `load_state_dict` rejects legacy Q-containing checkpoints and
      `v_ensemble_size` mismatches (old single-head v4).
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
from robosuite.pipeline.algorithms.q_learning.losses import expectile_v_loss


# Synthetic token layout for the Token/Group projector: small dims that stay
# divisible by n_tokens=4 (state visual 12-4=8 -> 4x2).
_N_TOKENS = 4
_PROPRIO_DIM = 4


def _make_cfg(action_horizon: int = 2, v_ensemble_size: int = 2) -> IQLConfig:
    return IQLConfig(
        action_horizon=action_horizon,
        discount=0.9,
        expectile_tau=0.7,
        v_ensemble_size=v_ensemble_size,
        ensemble_lcb_beta=0.5,
        ensemble_bootstrap_prob=0.5,
        v_lr=1e-3,
        target_polyak=0.1,
        n_step_aggregate=True,
        hidden_dims=(32, 32),
        state_proj_dim=8,
        proprio_proj_dim=4,
        proj_activation="mish",
        grad_clip_norm=1.0,
        weight_decay=0.0,
        device="cpu",
        disc_reward_coef=0.0,
        output_reward_coef=1.0,
    )


def _make_learner(
    cfg: IQLConfig,
    state_feature_dim: int = 12,
    chunk_feature_dim: int = 16,
    action_dim: int = 4,
) -> IQLLearner:
    return IQLLearner(
        cfg=cfg,
        state_feature_dim=state_feature_dim,
        chunk_feature_dim=chunk_feature_dim,
        action_dim=action_dim,
        n_tokens=_N_TOKENS,
        proprio_dim=_PROPRIO_DIM,
    )


def _make_step_batch(
    B: int = 8,
    D_state: int = 12,
    D_chunk: int = 16,
    D_a: int = 4,
    H: int = 2,
) -> IQLStepBatch:
    torch.manual_seed(0)
    return IQLStepBatch(
        q_chunk_feature=torch.randn(B, D_chunk),
        v_state_feature=torch.randn(B, D_state),
        next_v_state_feature=torch.randn(B, D_state),
        action_chunk=torch.randn(B, H, D_a),
        rewards=torch.randn(B, 1),
        dones=torch.zeros(B, 1),
        is_online=torch.zeros(B, 1),
        is_intervention=torch.zeros(B, 1),
        metadata={},
    )


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


def test_iql_update_step_runs_and_moves_v() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql = _make_learner(cfg)
    batch = _make_step_batch(B=8, D_state=12, D_chunk=16, D_a=4, H=2)

    snapshot_v = [p.detach().clone() for p in iql.v.parameters()]

    # Heads are zero-init at the final layer, so V_lcb reduces to 0 and the
    # bootstrap target reduces to `rewards` (nonzero); the expectile-TD loss
    # produces gradients that move V.
    metrics = iql.update(batch)
    assert all(math.isfinite(v) for v in metrics.values()), f"non-finite metric: {metrics}"
    assert set(metrics) == {
        "v_loss",
        "v_mean",
        "target_mean",
        "td_error_abs_mean",
        "v_std_mean",
    }

    moved_v = any(
        not torch.equal(before, after)
        for before, after in zip(snapshot_v, iql.v.parameters())
    )
    assert moved_v, "v parameters did not move after update"


def test_compute_td_advantage_matches_formula() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql = _make_learner(cfg)
    batch = _make_step_batch(B=6, D_state=12, D_chunk=16, D_a=4, H=2)
    # Move V away from its zero init so v(s) != target_v(s').
    iql.update(batch)

    adv = iql.compute_td_advantage(
        batch.v_state_feature, batch.next_v_state_feature, batch.rewards, batch.dones
    )
    gamma_h = float(cfg.discount) ** int(cfg.action_horizon)
    v_s = iql.v_lcb(batch.v_state_feature)
    v_sp = iql.target_v_lcb(batch.next_v_state_feature)
    expected = (batch.rewards + gamma_h * (1.0 - batch.dones) * v_sp - v_s).reshape(-1)
    assert adv.shape == (6,)
    assert torch.allclose(adv, expected, atol=1e-6)


def test_iql_state_dict_roundtrip() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql_a = _make_learner(cfg)
    iql_b = _make_learner(cfg)
    # Run one step on A so its weights diverge from B's fresh init.
    batch = _make_step_batch(B=4, D_state=12, D_chunk=16, D_a=4, H=2)
    iql_a.update(batch)
    sd = iql_a.state_dict()
    assert "q_ensemble" not in sd and "q_optim" not in sd
    iql_b.load_state_dict(sd, strict=True)
    for pa, pb in zip(iql_a.v.parameters(), iql_b.v.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(iql_a.target_v.parameters(), iql_b.target_v.parameters()):
        assert torch.equal(pa, pb)


def test_iql_load_state_dict_mismatched_feature_dim_raises() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql_a = _make_learner(cfg)
    iql_c = _make_learner(cfg, state_feature_dim=8)
    sd = iql_a.state_dict()
    with pytest.raises(ValueError, match="state_feature_dim mismatch"):
        iql_c.load_state_dict(sd, strict=True)


def test_iql_old_q_checkpoint_schema_raises() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql_a = _make_learner(cfg)
    iql_b = _make_learner(cfg)
    sd = dict(iql_a.state_dict())
    # Simulate a legacy Q-containing checkpoint.
    sd["q_ensemble"] = {"dummy": torch.zeros(1)}
    with pytest.raises(ValueError, match="Q head"):
        iql_b.load_state_dict(sd, strict=True)


def test_iql_legacy_checkpoint_requires_new_warmup() -> None:
    cfg = _make_cfg(action_horizon=2)
    iql = _make_learner(cfg)
    legacy = dict(iql.state_dict())
    legacy.pop("state_feature_dim")
    with pytest.raises(ValueError, match="legacy schema.*Re-run offline V warmup"):
        iql.load_state_dict(legacy, strict=True)


# --------------------------------------------------------------------------- #
# V-ensemble + soft-LCB (§4 finalized method)                                 #
# --------------------------------------------------------------------------- #


def test_v_ensemble_forward_shape() -> None:
    cfg = _make_cfg(v_ensemble_size=3)
    iql = _make_learner(cfg)
    out = iql.v(torch.randn(5, 12))
    assert out.shape == (5, 3), "VEnsemble.forward must return (B, N)"
    assert iql.v_lcb(torch.randn(5, 12)).shape == (5, 1)


def test_v_lcb_single_head_reduces_to_mean() -> None:
    # N=1 -> population std == 0 -> V_lcb == the single head (beta inert).
    cfg = _make_cfg(v_ensemble_size=1)
    iql = _make_learner(cfg)
    iql.update(_make_step_batch(B=8))  # move off zero-init
    feat = torch.randn(6, 12)
    assert torch.allclose(iql.v_lcb(feat), iql.v(feat), atol=1e-6)


def test_v_lcb_is_mean_minus_beta_std() -> None:
    cfg = _make_cfg(v_ensemble_size=2)
    iql = _make_learner(cfg)
    iql.update(_make_step_batch(B=8))  # diverge the heads
    feat = torch.randn(7, 12)
    per_head = iql.v(feat)  # (B, 2)
    mean = per_head.mean(dim=-1, keepdim=True)
    std = per_head.std(dim=-1, unbiased=False, keepdim=True)
    expected = mean - float(cfg.ensemble_lcb_beta) * std
    assert torch.allclose(iql.v_lcb(feat), expected, atol=1e-6)


def test_bootstrap_target_uses_lcb() -> None:
    cfg = _make_cfg(v_ensemble_size=2)
    iql = _make_learner(cfg)
    batch = _make_step_batch(B=5)
    iql.update(batch)  # move target heads off zero-init
    target = iql._bootstrap_target(batch)
    gamma_h = float(cfg.discount) ** int(cfg.action_horizon)
    v_next = iql.target_v_lcb(batch.next_v_state_feature)
    expected = batch.rewards + gamma_h * (1.0 - batch.dones) * v_next
    assert target.shape == (5, 1)
    assert torch.allclose(target, expected, atol=1e-6)


def test_update_diversifies_heads() -> None:
    # Independent init + per-head bootstrap mask must drive the ensemble std
    # away from 0 so the soft-LCB is not inert.
    cfg = _make_cfg(v_ensemble_size=2)
    iql = _make_learner(cfg)
    last = 0.0
    for _ in range(40):
        last = iql.update(_make_step_batch(B=16))["v_std_mean"]
    assert last > 0.0, "ensemble heads did not diverge (LCB std collapsed)"


def test_state_dict_records_ensemble_size_and_rejects_mismatch() -> None:
    cfg2 = _make_cfg(v_ensemble_size=2)
    iql2 = _make_learner(cfg2)
    sd = iql2.state_dict()
    assert sd["v_ensemble_size"] == 2
    # A single-head learner must refuse an N=2 checkpoint before shape errors.
    iql1 = _make_learner(_make_cfg(v_ensemble_size=1))
    with pytest.raises(ValueError, match="v_ensemble_size mismatch"):
        iql1.load_state_dict(sd, strict=True)
