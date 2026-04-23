"""AdaLN and LPB-degeneration sanity checks."""

from __future__ import annotations

import torch

from robosuite.discriminator.d4disc.models.adaln import AdaLNDecoderBlock, AdaLNModulation, ConditionEmbedder
from robosuite.discriminator.d4disc.models.dynamics import ConditionalDynamicsPredictor
from robosuite.discriminator.d4disc.training.trainer import D4Trainer, D4TrainerConfig


def test_modulation_zero_init_outputs_zero() -> None:
    torch.manual_seed(0)
    mod = AdaLNModulation(d_cond=32, d_model=64)
    cond = torch.randn(4, 32)
    shift, scale, gate = mod(cond)
    assert torch.all(shift == 0)
    assert torch.all(scale == 0)
    assert torch.all(gate == 0)


def test_adaln_block_identity_at_init() -> None:
    torch.manual_seed(0)
    block = AdaLNDecoderBlock(d_model=64, nhead=4, d_cond=32).eval()
    cond_embedder = ConditionEmbedder(d_cond=32).eval()
    x = torch.randn(2, 5, 64)
    mask = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)
    for c in (ConditionEmbedder.COND_PLUS, ConditionEmbedder.COND_MINUS):
        cond = cond_embedder(torch.full((2,), c, dtype=torch.long))
        out = block(x, cond=cond, attn_mask=mask)
        assert torch.allclose(out, x, atol=1e-6), f"c={c}: max abs diff={(out - x).abs().max()}"


def test_predictor_c_invariant_at_init() -> None:
    torch.manual_seed(0)
    predictor = ConditionalDynamicsPredictor(
        latent_dim=32,
        proprio_dim=6,
        action_dim=4,
        d_model=32,
        num_layers=2,
        nhead=4,
        dropout=0.0,
        max_action_horizon=4,
        d_cond=16,
    ).eval()

    obs = torch.randn(3, 32)
    prop = torch.randn(3, 6)
    act = torch.randn(3, 2, 4)

    outs = []
    for c in (ConditionEmbedder.COND_PLUS, ConditionEmbedder.COND_MINUS):
        cond_idx = torch.full((3,), c, dtype=torch.long)
        outs.append(predictor(obs, prop, act, cond_idx=cond_idx))

    for key in ("pred_latent", "pred_proprio"):
        ref = outs[0][key]
        for other in outs[1:]:
            assert torch.allclose(ref, other[key], atol=1e-6), (
                f"{key}: max abs diff={(ref - other[key]).abs().max()}"
            )


def test_predictor_plus_branch_matches_identity_at_init() -> None:
    torch.manual_seed(0)
    predictor = ConditionalDynamicsPredictor(
        latent_dim=32,
        proprio_dim=6,
        action_dim=4,
        d_model=32,
        num_layers=2,
        nhead=4,
        dropout=0.0,
        max_action_horizon=4,
        d_cond=16,
        adaln_init_std=0.0,
        residual_latent_head=True,
    ).eval()

    obs = torch.randn(5, 32)
    prop = torch.randn(5, 6)
    act = torch.randn(5, 2, 4)
    cond_idx = torch.full((5,), ConditionEmbedder.COND_PLUS, dtype=torch.long)
    out = predictor(obs, prop, act, cond_idx=cond_idx)
    assert torch.allclose(out["pred_latent"], obs, atol=1e-6)


def test_trainer_condition_routing_matches_two_branch_design() -> None:
    trainer = D4Trainer.__new__(D4Trainer)
    trainer.cfg = D4TrainerConfig(min_batch_balance=0.0)

    is_fail_raw = torch.tensor([False, True, True, True], dtype=torch.bool)
    gamma = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    cond_phase_a = trainer._sample_condition(is_fail_raw, gamma, phase="A")
    assert torch.all(cond_phase_a == ConditionEmbedder.COND_PLUS)

    cond_phase_b = trainer._sample_condition(is_fail_raw, gamma, phase="B")
    expected = torch.tensor(
        [
            ConditionEmbedder.COND_PLUS,
            ConditionEmbedder.COND_PLUS,
            ConditionEmbedder.COND_MINUS,
            ConditionEmbedder.COND_MINUS,
        ],
        dtype=torch.long,
    )
    assert torch.equal(cond_phase_b.cpu(), expected)
