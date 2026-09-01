"""CPU-only tests for AdvantageGProvider.

The online TD-advantage path is currently a stub (the online DipoleBatch carries
no next state / chunk reward), so ``compute_g_for_batch`` raises
``NotImplementedError``. These tests cover the stub contract plus the wiring that
is still live (single-observation guard, camera binding, agent attach).
"""

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
    """Minimal encoder stub: only camera binding is exercised now."""

    def __init__(self, context_dim: int) -> None:
        self.context_dim = int(context_dim)
        self.device = "cpu"
        self.bind_calls: list[list[str]] = []

    def bind_policy_cameras(self, cams: Sequence[str]) -> None:
        self.bind_calls.append(list(cams))


class _FakeVAST:
    """Placeholder VAST learner (the stub provider never calls into it)."""


def _make_batch(B: int = 4, H: int = 3, D_a: int = 2, D_s: int = 5):
    return SimpleNamespace(
        image_obs_raw=torch.zeros(B, 1, 3, 4, 4),
        proprio_raw=torch.zeros(B, D_s),
        action_sequences_raw=torch.zeros(B, H, D_a),
    )


def _make_provider() -> AdvantageGProvider:
    return AdvantageGProvider(
        vast_learner=_FakeVAST(),
        encoder=_FakeEncoder(context_dim=4),
        alpha=1.0,
    )


def test_compute_g_for_batch_raises_not_implemented() -> None:
    provider = _make_provider()
    with pytest.raises(NotImplementedError, match="not implemented for"):
        provider.compute_g_for_batch(_make_batch(B=4))


def test_compute_g_for_observation_raises() -> None:
    provider = _make_provider()
    with pytest.raises(NotImplementedError, match="single-observation"):
        provider.compute_g_for_observation({}, None)


def test_bind_policy_cameras_proxies_to_encoder() -> None:
    encoder = _FakeEncoder(context_dim=4)
    provider = AdvantageGProvider(
        vast_learner=_FakeVAST(),
        encoder=encoder,
        alpha=1.0,
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
