"""AdaLN-Zero conditioning primitives for D4-Disc.

Adopts the DiT AdaLN-Zero block (Peebles & Xie, 2023) as the per-layer
conditioning primitive on top of the LPB decoder backbone. The load-bearing
stability property is the ``zero-init`` of every ``AdaLNModulation.linear``
head: at step 0 the block collapses to identity regardless of the condition
token ``c in {+, -, null}``, so a freshly-instantiated conditional predictor
behaves as an unconditional network. This is what makes Phase A training
(c in {+, null} only) numerically safe and what keeps CFG well-defined at
inference.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConditionEmbedder(nn.Module):
    """3-token embedding for c in {+, -, null}.

    Constants match the routing table in ``filter.py`` / ``trainer.py``:
        COND_PLUS  = 0
        COND_MINUS = 1
        COND_NULL  = 2
    """

    COND_PLUS: int = 0
    COND_MINUS: int = 1
    COND_NULL: int = 2

    def __init__(self, d_cond: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(3, d_cond)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.act = nn.SiLU()
        self.d_cond = int(d_cond)

    def forward(self, cond_idx: torch.Tensor) -> torch.Tensor:
        if cond_idx.dtype != torch.long:
            cond_idx = cond_idx.long()
        return self.act(self.embed(cond_idx))


class AdaLNModulation(nn.Module):
    """Produces (shift, scale, gate) per sub-block.

    ``init_std=0.0`` (default) reproduces the DiT AdaLN-Zero behavior — the
    block is identity at step 0 for all conditions. Setting ``init_std`` >0
    initializes the modulation head's weights with small Gaussian noise so
    that c=+ and c=- immediately produce different (shift, scale, gate)
    vectors. Useful for escaping the symmetric fixed point (A=0, gamma=0.5)
    when fail data alone cannot grow the gate fast enough during bootstrap.
    """

    def __init__(self, d_cond: int, d_model: int, init_std: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(d_cond, 3 * d_model)
        if float(init_std) <= 0.0:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.normal_(self.linear.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.linear.bias)
        self.d_model = int(d_model)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(cond).chunk(3, dim=-1)
        return shift, scale, gate


class AdaLNDecoderBlock(nn.Module):
    """Drop-in replacement for ``lpb.model.DecoderBlock`` with AdaLN-Zero.

    Keeps pre-norm residual structure of the LPB backbone; swaps affine part of
    LayerNorm for (shift, scale) derived from the condition token, and gates
    each residual contribution by a zero-init gate vector.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_cond: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        adaln_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.mod_attn = AdaLNModulation(d_cond, d_model, init_std=float(adaln_init_std))
        self.mod_mlp = AdaLNModulation(d_cond, d_model, init_std=float(adaln_init_std))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, L, D)   cond: (B, d_cond)
        shift_a, scale_a, gate_a = self.mod_attn(cond)
        shift_m, scale_m, gate_m = self.mod_mlp(cond)

        h = self.ln_1(x) * (1.0 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate_a.unsqueeze(1) * h

        h = self.ln_2(x) * (1.0 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_m.unsqueeze(1) * h
        return x
