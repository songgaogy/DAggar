"""CPU-only direction and raw-mixing tests for AdvantageGProvider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

from robosuite.pipeline.algorithms.dipole.advantage_g_provider import (
    AdvantageGProvider,
)
from robosuite.pipeline.algorithms.dipole.agent import DipoleAgent


class _FakeEncoder:
    """Return deterministic state and action-conditioned chunk features."""

    def __init__(self, context_dim: int) -> None:
        self.context_dim = int(context_dim)
        self.device = "cpu"
        self.bind_calls: list[list[str]] = []
        self.last_action_real: torch.Tensor | None = None

    def bind_policy_cameras(self, cams: Sequence[str]) -> None:
        self.bind_calls.append(list(cams))

    @torch.no_grad()
    def encode_state_and_chunk(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = int(image_obs_raw.shape[0])
        self.last_action_real = action_chunk.detach().clone()
        state = torch.zeros(B, self.context_dim)
        chunk = torch.zeros(B, self.context_dim)
        return state, chunk


class _FakeIQL:
    """Returns a deterministic advantage tensor."""

    def __init__(self, advantage_values: torch.Tensor | float) -> None:
        self.advantage_values = advantage_values

    @torch.no_grad()
    def compute_advantage_for_batch(self, actor_batch) -> torch.Tensor:
        B = int(actor_batch.q_chunk_feature.shape[0])
        if isinstance(self.advantage_values, torch.Tensor):
            assert self.advantage_values.shape == (B,), (
                f"fake advantage shape {tuple(self.advantage_values.shape)} != (B={B},)"
            )
            return self.advantage_values.clone()
        return torch.full((B,), float(self.advantage_values))


class _FakeDisc:
    """Return a deterministic nnPU failure-score tensor."""

    def __init__(self, logit_values: torch.Tensor | float) -> None:
        self.logit_values = logit_values

    @torch.no_grad()
    def failure_score(self, *, chunk_feature: torch.Tensor) -> torch.Tensor:
        B = int(chunk_feature.shape[0])
        if isinstance(self.logit_values, torch.Tensor):
            assert self.logit_values.shape == (B,), (
                f"fake logit shape {tuple(self.logit_values.shape)} != (B={B},)"
            )
            score = self.logit_values.clone()
        else:
            score = torch.full((B,), float(self.logit_values))
        return score


def _make_batch(B: int = 4, H: int = 3, D_a: int = 2, D_s: int = 5):
    return SimpleNamespace(
        image_obs_raw=torch.zeros(B, 1, 3, 4, 4),
        proprio_raw=torch.zeros(B, D_s),
        action_sequences_raw=torch.zeros(B, H, D_a),
    )


# --------------------------------------------------------------------------- #
# 1. beta=0 -> g == advantage                                                  #
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
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    assert g.shape == (B,)
    assert torch.allclose(g, adv, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. alpha=0 -> g == -failure                                                  #
# --------------------------------------------------------------------------- #


def test_g_reduces_to_failure_when_alpha_zero() -> None:
    torch.manual_seed(0)
    B = 6
    logits = torch.linspace(-2.0, 3.0, B)
    provider = AdvantageGProvider(
        iql_learner=_FakeIQL(0.0),
        discriminator=_FakeDisc(logits),
        encoder=_FakeEncoder(context_dim=4),
        alpha=0.0,
        beta=1.0,
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    assert g.shape == (B,)
    assert torch.allclose(g, -logits, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. g == alpha*adv - beta*failure                                             #
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
    )
    g = provider.compute_g_for_batch(_make_batch(B=B))
    expected = 2.0 * adv - 3.0 * logits
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

def test_agent_attach_g_provider_is_a_method() -> None:
    class Core:
        def set_g_provider(self, provider) -> None:
            self.provider = provider

    agent = object.__new__(DipoleAgent)
    agent.core = Core()
    provider = object()
    agent.attach_g_provider(provider)
    assert agent.core.provider is provider
