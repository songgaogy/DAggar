"""CPU-only tests for DIPOLE guided action sampling (two-policy architecture)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from robosuite.pipeline.algorithms.dipole.agent import DipoleAgent
from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.algorithms.dipole.models.flow import (
    DipoleFlowPolicy,
    _sample_guided_action_sequence,
)
from robosuite.pipeline.modules.evaluation.policy import (
    _resolve_execute_horizon,
)


class _CountingFlowHead(nn.Module):
    def __init__(self, cond_offset: float = 0.0) -> None:
        super().__init__()
        self.calls = 0
        self.cond_offset = float(cond_offset)

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        task_scene_cond: torch.Tensor,
        context_tokens: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        context_scale = context_tokens.masked_fill(
            context_padding_mask.unsqueeze(-1), 0.0
        ).mean(dim=(1, 2))
        scale = task_scene_cond.mean(dim=1) + context_scale + timesteps + self.cond_offset
        return torch.tanh(x_t + scale[:, None, None])


class _FakeFlowModel(nn.Module):
    """Stub with the minimal ``MultiModalFlowPolicy`` surface the sampler uses.

    Each policy (positive / negative) is now a fully independent model; the two
    differ by ``cond_offset`` (standing in for the two diverged full-tune policies),
    so the guided combination ``v=(1+omega)v_pos - omega v_neg`` is non-trivial.
    """

    def __init__(self, action_dim: int = 3, context_dim: int = 5, cond_offset: float = 0.0) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.flow_head = _CountingFlowHead(cond_offset=cond_offset)

    def encode_multimodal_context(
        self,
        *,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str],
    ) -> dict[str, torch.Tensor]:
        batch_size = proprio.shape[0]
        assert len(language) == batch_size
        base = proprio.mean(dim=1, keepdim=True)
        task_scene_cond = base.repeat(1, self.context_dim)
        context_tokens = images.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        context_tokens = context_tokens.repeat(1, 1, self.context_dim)
        context_padding_mask = torch.zeros(
            batch_size, context_tokens.shape[1], dtype=torch.bool
        )
        return {
            "task_scene_cond": task_scene_cond,
            "context_tokens": context_tokens,
            "context_padding_mask": context_padding_mask,
        }


def _serial_sample(
    model_pos: _FakeFlowModel,
    model_neg: _FakeFlowModel,
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
    n_steps: int,
    omega: float,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    x = torch.zeros(batch_size, model_pos.action_dim, action_horizon)
    language = ["task"] * batch_size
    context_pos = model_pos.encode_multimodal_context(images=images, proprio=proprio, language=language)
    context_neg = model_neg.encode_multimodal_context(images=images, proprio=proprio, language=language)
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps))
        v_pos = model_pos.flow_head(
            x_t=x,
            timesteps=t,
            task_scene_cond=context_pos["task_scene_cond"],
            context_tokens=context_pos["context_tokens"],
            context_padding_mask=context_pos["context_padding_mask"],
        )
        v_neg = model_neg.flow_head(
            x_t=x,
            timesteps=t,
            task_scene_cond=context_neg["task_scene_cond"],
            context_tokens=context_neg["context_tokens"],
            context_padding_mask=context_neg["context_padding_mask"],
        )
        x = x + ((1.0 + omega) * v_pos - omega * v_neg) / float(n_steps)
    return x.transpose(1, 2)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("omega", [0.2, 1.0])
def test_guidance_matches_serial_reference(batch_size: int, omega: float) -> None:
    torch.manual_seed(0)
    images = torch.randn(batch_size, 2, 3, 4, 4)
    proprio = torch.randn(batch_size, 4)
    model_pos = _FakeFlowModel(cond_offset=0.0)
    model_neg = _FakeFlowModel(cond_offset=0.5)

    expected = _serial_sample(
        model_pos,
        model_neg,
        images=images,
        proprio=proprio,
        action_horizon=4,
        n_steps=5,
        omega=omega,
    )
    # Reset call counters consumed by the reference pass.
    model_pos.flow_head.calls = 0
    model_neg.flow_head.calls = 0
    actual = _sample_guided_action_sequence(
        model_pos,
        model_neg,
        images=images,
        proprio=proprio,
        language=["task"] * batch_size,
        action_horizon=4,
        n_steps=5,
        omega=omega,
        deterministic=True,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert model_pos.flow_head.calls == 5
    assert model_neg.flow_head.calls == 5


def test_zero_omega_uses_only_positive_policy() -> None:
    model_pos = _FakeFlowModel(cond_offset=0.0)
    model_neg = _FakeFlowModel(cond_offset=0.5)
    output = _sample_guided_action_sequence(
        model_pos,
        model_neg,
        images=torch.randn(2, 2, 3, 4, 4),
        proprio=torch.randn(2, 4),
        language=["task", "task"],
        action_horizon=4,
        n_steps=6,
        omega=0.0,
        deterministic=True,
    )

    assert output.shape == (2, 4, model_pos.action_dim)
    # omega=0 => positive policy only; the negative policy is never touched.
    assert model_pos.flow_head.calls == 6
    assert model_neg.flow_head.calls == 0


def test_agent_online_actions_enable_configured_guidance() -> None:
    class _Core:
        def __init__(self) -> None:
            self.calls = []

        def select_action(self, **kwargs):
            self.calls.append(("select", kwargs))
            return "action"

        def plan_action_chunk(self, **kwargs):
            self.calls.append(("plan", kwargs))
            return "chunk"

    agent = DipoleAgent.__new__(DipoleAgent)
    agent.core = _Core()

    assert agent.select_action("obs", deterministic=True) == "action"
    assert agent.plan_action_chunk("obs", deterministic=False) == "chunk"
    assert agent.core.calls == [
        ("select", {"obs": "obs", "deterministic": True, "guided": True}),
        ("plan", {"obs": "obs", "deterministic": False, "guided": True}),
    ]


class _FakeUpdateModel(nn.Module):
    """Trainable stub with the ``MultiModalFlowPolicy.forward`` signature."""

    def __init__(self, scale: float = 0.1) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor(float(scale)))
        self.calls = 0

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str],
    ) -> torch.Tensor:
        self.calls += 1
        _ = t, images, proprio, language
        # x_t is (B, action_dim, horizon); return same shape, dependent on param.
        return x_t * self.param + self.param


def test_update_trains_both_policies_independently() -> None:
    policy = object.__new__(DipoleFlowPolicy)
    policy.device = torch.device("cpu")
    policy.model_pos = _FakeUpdateModel(scale=0.1)
    policy.model_neg = _FakeUpdateModel(scale=-0.1)
    policy.config = SimpleNamespace(
        beta=0.0,
        k=0.0,
        lambda_endpoint=0.0,
        lambda_smooth=0.0,
        grad_clip_norm=10.0,
        branch_weight_mode="coupled",
    )
    policy.language_instruction = "task"
    policy.g_provider = None
    policy.optimizer_pos = torch.optim.SGD(policy.model_pos.parameters(), lr=0.01)
    policy.optimizer_neg = torch.optim.SGD(policy.model_neg.parameters(), lr=0.01)
    policy.scaler_pos = torch.amp.GradScaler(enabled=False, device=policy.device)
    policy.scaler_neg = torch.amp.GradScaler(enabled=False, device=policy.device)

    batch = DipoleBatch(
        image_obs=torch.zeros(3, 1, 3, 4, 4),
        image_obs_raw=torch.zeros(3, 1, 3, 4, 4),
        proprio=torch.zeros(3, 2),
        proprio_raw=torch.zeros(3, 2),
        action_sequences=torch.randn(3, 2, 2),
        action_sequences_raw=torch.zeros(3, 2, 2),
        is_intervention=torch.zeros(3, dtype=torch.bool),
        metadata={},
    )

    metrics = policy.update(batch, want_metrics=True)

    # One forward per policy (no fused 2x pass anymore).
    assert policy.model_pos.calls == 1
    assert policy.model_neg.calls == 1
    assert "loss_pos" in metrics
    assert "loss_neg" in metrics
    # Both policies received gradients and are optimized independently.
    assert policy.model_pos.param.grad is not None
    assert policy.model_neg.param.grad is not None


def test_update_skips_metrics_when_not_requested() -> None:
    policy = object.__new__(DipoleFlowPolicy)
    policy.device = torch.device("cpu")
    policy.model_pos = _FakeUpdateModel(scale=0.1)
    policy.model_neg = _FakeUpdateModel(scale=-0.1)
    policy.config = SimpleNamespace(
        beta=0.0,
        k=0.0,
        lambda_endpoint=0.0,
        lambda_smooth=0.0,
        grad_clip_norm=10.0,
        branch_weight_mode="coupled",
    )
    policy.language_instruction = "task"
    policy.g_provider = None
    policy.optimizer_pos = torch.optim.SGD(policy.model_pos.parameters(), lr=0.01)
    policy.optimizer_neg = torch.optim.SGD(policy.model_neg.parameters(), lr=0.01)
    policy.scaler_pos = torch.amp.GradScaler(enabled=False, device=policy.device)
    policy.scaler_neg = torch.amp.GradScaler(enabled=False, device=policy.device)

    batch = DipoleBatch(
        image_obs=torch.zeros(3, 1, 3, 4, 4),
        image_obs_raw=torch.zeros(3, 1, 3, 4, 4),
        proprio=torch.zeros(3, 2),
        proprio_raw=torch.zeros(3, 2),
        action_sequences=torch.randn(3, 2, 2),
        action_sequences_raw=torch.zeros(3, 2, 2),
        is_intervention=torch.zeros(3, dtype=torch.bool),
        metadata={},
    )

    # want_metrics=False still trains (grads present) but returns no scalar metrics.
    metrics = policy.update(batch, want_metrics=False)
    assert metrics == {}
    assert policy.model_pos.param.grad is not None
    assert policy.model_neg.param.grad is not None


@pytest.mark.parametrize("requested", [1, 4, 8])
def test_execute_horizon_accepts_valid_values(requested: int) -> None:
    assert _resolve_execute_horizon(requested, action_horizon=8) == requested


@pytest.mark.parametrize("requested", [0, 9])
def test_execute_horizon_rejects_out_of_range_values(requested: int) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _resolve_execute_horizon(requested, action_horizon=8)
