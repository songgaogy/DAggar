"""CUDA synthetic-tensor tests for the VAST learner.

No encoder or environment is constructed. Exercises:
    - `expectile_v_loss` at tau=0.5 reducing to 0.5·MSE.
    - `VEnsemble` forward shape and the `V_lcb = mean - beta*std` reduction
      (N=1 ⇒ std=0 ⇒ lcb==mean; N=2 ⇒ lcb below mean by beta*std).
    - One `VASTLearner.update` joint G/V step runs forward+backward
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

from robosuite.pipeline.algorithms.vast.common import VASTConfig, VASTStepBatch
from robosuite.pipeline.algorithms.vast.data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
)
from robosuite.pipeline.algorithms.vast.vast import VASTLearner
from robosuite.pipeline.algorithms.vast.losses import expectile_v_loss


# Synthetic token layout for the Token/Group projector: small dims that stay
# divisible by n_tokens=4 (state visual 12-4=8 -> 4x2).
_N_TOKENS = 4
_PROPRIO_DIM = 4
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VAST tensor tests require CUDA and never fall back to CPU.",
)
DEVICE = "cuda:0"


def _make_cfg(
    action_horizon: int = 2,
    v_ensemble_size: int = 2,
    vast_v_mode: str | None = None,
) -> VASTConfig:
    return VASTConfig(
        vast_v_mode=(
            vast_v_mode
            if vast_v_mode is not None
            else ("single_vast" if v_ensemble_size == 1 else "ensemble_lcb")
        ),
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
        device=DEVICE,
        disc_reward_coef=0.0,
        output_reward_coef=1.0,
    )


def _make_learner(
    cfg: VASTConfig,
    state_feature_dim: int = 12,
    chunk_feature_dim: int = 16,
    action_dim: int = 4,
) -> VASTLearner:
    return VASTLearner(
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
) -> VASTStepBatch:
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    current = torch.randn(B, D_state, device=DEVICE, generator=generator)
    future = torch.randn(B, D_state, device=DEVICE, generator=generator)
    intermediate = torch.randn(B, D_state, device=DEVICE, generator=generator)
    return VASTStepBatch(
        chunk_feature=torch.randn(B, D_chunk, device=DEVICE, generator=generator),
        v_state_feature=current,
        next_v_state_feature=torch.randn(B, D_state, device=DEVICE, generator=generator),
        action_chunk=torch.randn(B, H, D_a, device=DEVICE, generator=generator),
        rewards=torch.randn(B, 1, device=DEVICE, generator=generator),
        dones=torch.zeros(B, 1, device=DEVICE),
        is_online=torch.zeros(B, 1, device=DEVICE),
        is_intervention=torch.zeros(B, 1, device=DEVICE),
        metadata={},
        future_v_state_feature=future,
        intermediate_v_state_feature=intermediate,
        k=torch.full((B, 1), 2.0, device=DEVICE),
        j=torch.ones(B, 1, device=DEVICE),
        k_step_returns=torch.randn(B, 1, device=DEVICE, generator=generator),
        mc_mask=torch.ones(B, 1, device=DEVICE),
        future_dones=torch.zeros(B, 1, device=DEVICE),
    )


def test_expectile_v_loss_tau_half_is_quarter_mse() -> None:
    diff = torch.tensor([[1.0], [-2.0], [0.5]], device=DEVICE)
    # tau = 0.5 -> weight = 0.5 everywhere; loss = 0.5 * mean(diff^2)
    expected = 0.5 * diff.square().mean()
    actual = expectile_v_loss(diff, tau=0.5)
    assert torch.allclose(actual, expected)


def test_expectile_v_loss_asymmetry() -> None:
    diff = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    tau = 0.7
    # positive diff contributes 0.7 * 1; negative diff contributes 0.3 * 1; mean = 0.5
    actual = expectile_v_loss(diff, tau=tau)
    assert torch.allclose(actual, torch.tensor(0.5, device=DEVICE))


def test_aggregate_chunk_reward_closed_form() -> None:
    H = 4
    discount = 0.9
    step_rewards = torch.ones(1, H, device=DEVICE)  # constant reward 1
    out = aggregate_chunk_reward(step_rewards, discount)
    expected = sum(discount ** i for i in range(H))
    assert out.shape == (1, 1)
    assert math.isclose(float(out.item()), expected, rel_tol=1e-6)


def test_chunk_done_mask() -> None:
    step_dones = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], device=DEVICE
    )
    out = chunk_done_mask(step_dones)
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[1.0], [0.0]], device=DEVICE))


def test_vast_update_step_runs_and_moves_v() -> None:
    cfg = _make_cfg(action_horizon=2)
    vast = _make_learner(cfg)
    batch = _make_step_batch(B=8, D_state=12, D_chunk=16, D_a=4, H=2)

    snapshot_v = [p.detach().clone() for p in vast.v.parameters()]

    # The first coherent-snapshot step moves zero-initialized G. The second
    # step observes the updated G in the detached stitched target and moves V.
    vast.update(batch)
    metrics = vast.update(batch)
    assert all(math.isfinite(v) for v in metrics.values()), f"non-finite metric: {metrics}"
    assert {
        "g_loss",
        "g_mc_loss",
        "g_comp_loss",
        "v_loss",
        "v_mean",
        "target_mean",
        "td_error_abs_mean",
        "v_std_mean",
    }.issubset(metrics)

    moved_v = any(
        not torch.equal(before, after)
        for before, after in zip(snapshot_v, vast.v.parameters())
    )
    assert moved_v, "v parameters did not move after update"


def test_compute_td_advantage_matches_formula() -> None:
    cfg = _make_cfg(action_horizon=2)
    vast = _make_learner(cfg)
    batch = _make_step_batch(B=6, D_state=12, D_chunk=16, D_a=4, H=2)
    # Move V away from its zero init so v(s) != target_v(s').
    vast.update(batch)

    adv = vast.compute_td_advantage(
        batch.v_state_feature, batch.next_v_state_feature, batch.rewards, batch.dones
    )
    gamma_h = float(cfg.discount) ** int(cfg.action_horizon)
    v_s = vast.v_lcb(batch.v_state_feature)
    v_sp = vast.target_v_lcb(batch.next_v_state_feature)
    expected = (batch.rewards + gamma_h * (1.0 - batch.dones) * v_sp - v_s).reshape(-1)
    assert adv.shape == (6,)
    assert torch.allclose(adv, expected, atol=1e-6)


def test_vast_state_dict_roundtrip() -> None:
    cfg = _make_cfg(action_horizon=2)
    vast_a = _make_learner(cfg)
    vast_b = _make_learner(cfg)
    # Run one step on A so its weights diverge from B's fresh init.
    batch = _make_step_batch(B=4, D_state=12, D_chunk=16, D_a=4, H=2)
    vast_a.update(batch)
    sd = vast_a.state_dict()
    assert "q_ensemble" not in sd and "q_optim" not in sd
    vast_b.load_state_dict(sd, strict=True)
    for pa, pb in zip(vast_a.v.parameters(), vast_b.v.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(vast_a.target_v.parameters(), vast_b.target_v.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(vast_a.g.parameters(), vast_b.g.parameters()):
        assert torch.equal(pa, pb)


def test_vast_load_state_dict_mismatched_feature_dim_raises() -> None:
    cfg = _make_cfg(action_horizon=2)
    vast_a = _make_learner(cfg)
    vast_c = _make_learner(cfg, state_feature_dim=8)
    sd = vast_a.state_dict()
    with pytest.raises(ValueError, match="state_feature_dim mismatch"):
        vast_c.load_state_dict(sd, strict=True)


def test_vast_old_q_checkpoint_schema_raises() -> None:
    cfg = _make_cfg(action_horizon=2)
    vast_a = _make_learner(cfg)
    vast_b = _make_learner(cfg)
    sd = dict(vast_a.state_dict())
    # Simulate a legacy Q-containing checkpoint.
    sd["q_ensemble"] = {"dummy": torch.zeros(1, device=DEVICE)}
    with pytest.raises(ValueError, match="Q head"):
        vast_b.load_state_dict(sd, strict=True)


def test_vast_legacy_checkpoint_requires_new_warmup() -> None:
    cfg = _make_cfg(action_horizon=2)
    vast = _make_learner(cfg)
    legacy = dict(vast.state_dict())
    legacy["learner_schema_version"] = 5
    with pytest.raises(ValueError, match="unsupported learner schema"):
        vast.load_state_dict(legacy, strict=True)


# --------------------------------------------------------------------------- #
# V-ensemble + soft-LCB (§4 finalized method)                                 #
# --------------------------------------------------------------------------- #


def test_v_ensemble_forward_shape() -> None:
    cfg = _make_cfg(v_ensemble_size=3)
    vast = _make_learner(cfg)
    out = vast.v(torch.randn(5, 12, device=DEVICE))
    assert out.shape == (5, 3), "VEnsemble.forward must return (B, N)"
    assert vast.v_lcb(torch.randn(5, 12, device=DEVICE)).shape == (5, 1)


def test_v_lcb_single_head_reduces_to_mean() -> None:
    # N=1 -> population std == 0 -> V_lcb == the single head (beta inert).
    cfg = _make_cfg(v_ensemble_size=1)
    vast = _make_learner(cfg)
    vast.update(_make_step_batch(B=8))  # move off zero-init
    feat = torch.randn(6, 12, device=DEVICE)
    assert torch.allclose(vast.v_lcb(feat), vast.v(feat), atol=1e-6)


def test_v_lcb_is_mean_minus_beta_std() -> None:
    cfg = _make_cfg(v_ensemble_size=2)
    vast = _make_learner(cfg)
    vast.update(_make_step_batch(B=8))  # diverge the heads
    feat = torch.randn(7, 12, device=DEVICE)
    per_head = vast.v(feat)  # (B, 2)
    mean = per_head.mean(dim=-1, keepdim=True)
    std = per_head.std(dim=-1, unbiased=False, keepdim=True)
    expected = mean - float(cfg.ensemble_lcb_beta) * std
    assert torch.allclose(vast.v_lcb(feat), expected, atol=1e-6)


def test_bootstrap_target_uses_lcb() -> None:
    cfg = _make_cfg(v_ensemble_size=2)
    vast = _make_learner(cfg)
    batch = _make_step_batch(B=5)
    vast.update(batch)  # move target heads off zero-init
    target = vast._bootstrap_target(batch)
    gamma_h = float(cfg.discount) ** int(cfg.action_horizon)
    v_next = vast.target_v_lcb(batch.next_v_state_feature)
    expected = batch.rewards + gamma_h * (1.0 - batch.dones) * v_next
    assert target.shape == (5, 1)
    assert torch.allclose(target, expected, atol=1e-6)


def test_update_diversifies_heads() -> None:
    # Independent init + per-head bootstrap mask must drive the ensemble std
    # away from 0 so the soft-LCB is not inert.
    cfg = _make_cfg(v_ensemble_size=2)
    vast = _make_learner(cfg)
    last = 0.0
    for _ in range(40):
        last = vast.update(_make_step_batch(B=16))["v_std_mean"]
    assert last > 0.0, "ensemble heads did not diverge (LCB std collapsed)"


def test_state_dict_records_ensemble_size_and_rejects_mismatch() -> None:
    cfg2 = _make_cfg(v_ensemble_size=2)
    vast2 = _make_learner(cfg2)
    sd = vast2.state_dict()
    assert sd["v_ensemble_size"] == 2
    # A single-head learner must refuse an N=2 checkpoint before shape errors.
    vast1 = _make_learner(_make_cfg(v_ensemble_size=1))
    with pytest.raises(ValueError, match="v_ensemble_size mismatch"):
        vast1.load_state_dict(sd, strict=True)
