"""CPU-only smoke tests for AdvantageGProvider.

The provider is exercised against tiny fakes for SharedFrozenEncoder /
IQLLearner / OnlineBCEDiscriminator so the test runs without CUDA or the
heavy LPB v2 / Q-network machinery. The mixing math is what we care about.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

from robosuite.pipeline.algorithms.dipole.advantage_g_provider import (
    AdvantageGProvider,
)


class _FakeEncoder:
    """Stand-in for SharedFrozenEncoder. Returns a zero context."""

    def __init__(self, context_dim: int) -> None:
        self.context_dim = int(context_dim)
        self.device = "cpu"
        self.bind_calls: list[list[str]] = []
        self.last_action_real: torch.Tensor | None = None

    def bind_policy_cameras(self, cams: Sequence[str]) -> None:
        self.bind_calls.append(list(cams))

    @torch.no_grad()
    def encode(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_real: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = int(image_obs_raw.shape[0])
        if action_real is not None:
            self.last_action_real = action_real.detach().clone()
        return torch.zeros(B, self.context_dim)


class _FakeIQL:
    """Returns a deterministic advantage tensor."""

    def __init__(self, advantage_values: torch.Tensor | float) -> None:
        self.advantage_values = advantage_values

    @torch.no_grad()
    def compute_advantage_for_batch(self, actor_batch) -> torch.Tensor:
        B = int(actor_batch.context.shape[0])
        if isinstance(self.advantage_values, torch.Tensor):
            assert self.advantage_values.shape == (B,), (
                f"fake advantage shape {tuple(self.advantage_values.shape)} != (B={B},)"
            )
            return self.advantage_values.clone()
        return torch.full((B,), float(self.advantage_values))


class _FakeDisc:
    """Returns a deterministic logit tensor."""

    def __init__(self, logit_values: torch.Tensor | float) -> None:
        self.logit_values = logit_values

    @torch.no_grad()
    def score(self, *, context: torch.Tensor):
        B = int(context.shape[0])
        if isinstance(self.logit_values, torch.Tensor):
            assert self.logit_values.shape == (B,), (
                f"fake logit shape {tuple(self.logit_values.shape)} != (B={B},)"
            )
            logit = self.logit_values.clone()
        else:
            logit = torch.full((B,), float(self.logit_values))
        return SimpleNamespace(
            logit=logit,
            prob_failure=torch.sigmoid(logit),
            decision=logit > 0,
            metadata={},
        )


def _make_batch(B: int = 4, H: int = 3, D_a: int = 2, D_s: int = 5):
    return SimpleNamespace(
        image_obs_raw=torch.zeros(B, 1, 3, 4, 4),
        proprio_raw=torch.zeros(B, D_s),
        action_sequences_raw=torch.zeros(B, H, D_a),
    )


def _zscore(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + 1e-6)


# --------------------------------------------------------------------------- #
# 1. beta=0 -> G == zscore(advantage)                                          #
# --------------------------------------------------------------------------- #


def test_g_reduces_to_pure_advantage_when_beta_zero() -> None:
    torch.manual_seed(0)
    B = 8
    adv = torch.linspace(-1.0, 1.0, B)
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(adv),
        discriminator=_FakeDisc(0.0),
        encoder=_FakeEncoder(context_dim=4),
        alpha=1.0,
        beta=0.0,
        advantage_normalization="batch_zscore",
        disc_normalization="batch_zscore",
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    assert g.shape == (B,)
    assert torch.allclose(g, _zscore(adv), atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. alpha=0 -> G == -zscore(disc_logit)                                       #
# --------------------------------------------------------------------------- #


def test_g_reduces_to_negated_disc_when_alpha_zero() -> None:
    torch.manual_seed(0)
    B = 6
    logits = torch.linspace(-2.0, 3.0, B)
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(0.0),
        discriminator=_FakeDisc(logits),
        encoder=_FakeEncoder(context_dim=4),
        alpha=0.0,
        beta=1.0,
        advantage_normalization="batch_zscore",
        disc_normalization="batch_zscore",
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    assert g.shape == (B,)
    assert torch.allclose(g, -_zscore(logits), atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. mode="none" -> G == alpha*adv + beta*(-logit)                             #
# --------------------------------------------------------------------------- #


def test_linear_mixing_with_none_mode() -> None:
    B = 5
    adv = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.5])
    logits = torch.tensor([-0.5, 0.5, 1.0, -1.0, 2.0])
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(adv),
        discriminator=_FakeDisc(logits),
        encoder=_FakeEncoder(context_dim=4),
        alpha=2.0,
        beta=3.0,
        advantage_normalization="none",
        disc_normalization="none",
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    expected = 2.0 * adv + 3.0 * (-logits)
    assert torch.allclose(g, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. Shape + no grad                                                           #
# --------------------------------------------------------------------------- #


def test_output_shape_and_no_grad() -> None:
    B = 4
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(0.1),
        discriminator=_FakeDisc(0.2),
        encoder=_FakeEncoder(context_dim=4),
        alpha=1.0,
        beta=0.0,
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    assert g.dim() == 1
    assert g.shape == (B,)
    assert not g.requires_grad


# --------------------------------------------------------------------------- #
# 5. compute_g_for_observation raises                                          #
# --------------------------------------------------------------------------- #


def test_compute_g_for_observation_raises() -> None:
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(0.0),
        discriminator=_FakeDisc(0.0),
        encoder=_FakeEncoder(context_dim=4),
        alpha=1.0,
        beta=0.0,
    )
    with pytest.raises(NotImplementedError, match="single-observation"):
        provider.compute_g_for_observation({}, None)


# --------------------------------------------------------------------------- #
# 6. bind_policy_cameras proxies to encoder                                    #
# --------------------------------------------------------------------------- #


def test_bind_policy_cameras_proxies_to_encoder() -> None:
    encoder = _FakeEncoder(context_dim=4)
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(0.0),
        discriminator=_FakeDisc(0.0),
        encoder=encoder,
        alpha=1.0,
        beta=0.0,
    )
    cams = ["frontview_image", "wristview_image"]
    provider.bind_policy_cameras(cams)
    assert encoder.bind_calls == [cams]


# --------------------------------------------------------------------------- #
# 7. running_zscore advances EMA state                                         #
# --------------------------------------------------------------------------- #


def test_running_zscore_state_advances() -> None:
    B = 4
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(adv),
        discriminator=_FakeDisc(torch.zeros(B)),
        encoder=_FakeEncoder(context_dim=4),
        alpha=1.0,
        beta=1.0,
        advantage_normalization="running_zscore",
        disc_normalization="running_zscore",
    )
    assert provider._adv_running_count == 0
    assert provider._disc_running_count == 0
    g1 = provider.compute_g_for_batch(_make_batch(B=B))
    assert provider._adv_running_count == B
    assert provider._disc_running_count == B
    assert torch.isfinite(g1).all()
    g2 = provider.compute_g_for_batch(_make_batch(B=B))
    assert provider._adv_running_count == 2 * B
    assert torch.isfinite(g2).all()


# --------------------------------------------------------------------------- #
# 8. Unknown normalization mode rejected at construction                       #
# --------------------------------------------------------------------------- #


def test_unknown_normalization_mode_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        AdvantageGProvider(
            iql_learner=_FakeIQL(0.0),
            discriminator=_FakeDisc(0.0),
            encoder=_FakeEncoder(context_dim=4),
            alpha=1.0,
            beta=0.0,
            advantage_normalization="ema",
        )
