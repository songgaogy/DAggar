"""Zero-init identity test for AdaLN-Zero.

At initialization every AdaLNModulation head is zero-weighted, so every
block behaves as identity regardless of the condition token. The predictor
therefore produces numerically identical outputs for c in {+, -, null}.
This is load-bearing: Phase A of the trainer never routes fail_raw samples
to c=-, so c=- must stay identity until Phase B gradients flow through the
modulation heads.
"""

from __future__ import annotations

import torch

from robosuite.discriminator.d4disc.adaln import (
    AdaLNDecoderBlock,
    AdaLNModulation,
    ConditionEmbedder,
)
from robosuite.discriminator.d4disc.model import ConditionalDynamicsPredictor


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
    d_model, d_cond = 64, 32
    block = AdaLNDecoderBlock(d_model=d_model, nhead=4, d_cond=d_cond)
    # Disable attention dropout + mlp dropout to get byte-exact identity.
    block.eval()
    cond_embedder = ConditionEmbedder(d_cond=d_cond).eval()
    x = torch.randn(2, 5, d_model)
    L = x.size(1)
    mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    for c in (ConditionEmbedder.COND_PLUS, ConditionEmbedder.COND_MINUS, ConditionEmbedder.COND_NULL):
        cond_idx = torch.full((2,), c, dtype=torch.long)
        cond = cond_embedder(cond_idx)
        out = block(x, cond=cond, attn_mask=mask)
        # With zero-gate, residual branch contributes 0 -> output == input.
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

    B, H = 3, 2
    obs = torch.randn(B, 32)
    prop = torch.randn(B, 6)
    act = torch.randn(B, H, 4)

    outs = []
    for c in (ConditionEmbedder.COND_PLUS, ConditionEmbedder.COND_MINUS, ConditionEmbedder.COND_NULL):
        cond_idx = torch.full((B,), c, dtype=torch.long)
        outs.append(predictor(obs, prop, act, cond_idx=cond_idx))

    for key in ("pred_latent", "pred_proprio"):
        ref = outs[0][key]
        for other in outs[1:]:
            assert torch.allclose(ref, other[key], atol=1e-6), (
                f"{key}: max abs diff={(ref - other[key]).abs().max()}"
            )
