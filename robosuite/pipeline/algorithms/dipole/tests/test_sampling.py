"""CPU-only tests for DIPOLE guided action sampling."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from robosuite.pipeline.algorithms.dipole.models.flow import (
    _sample_guided_action_sequence,
)
from robosuite.pipeline.offline.eval_offline_dipole import (
    _resolve_execute_horizon,
)


class _CountingFlowHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

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
        scale = task_scene_cond.mean(dim=1) + context_scale + timesteps
        return torch.tanh(x_t + scale[:, None, None])


class _FakeDipoleModel(nn.Module):
    def __init__(self, action_dim: int = 3, context_dim: int = 5) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.polarity_embedding = nn.Embedding(2, context_dim)
        self.flow_head = _CountingFlowHead()
        with torch.no_grad():
            self.polarity_embedding.weight.copy_(
                torch.tensor(
                    [
                        [-0.4, 0.1, 0.2, -0.3, 0.5],
                        [0.3, -0.2, 0.4, 0.1, -0.1],
                    ]
                )
            )

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
        task_scene_cond = base.repeat(1, self.polarity_embedding.embedding_dim)
        context_tokens = images.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        context_tokens = context_tokens.repeat(1, 1, self.polarity_embedding.embedding_dim)
        context_padding_mask = torch.zeros(
            batch_size, context_tokens.shape[1], dtype=torch.bool
        )
        return {
            "task_scene_cond": task_scene_cond,
            "context_tokens": context_tokens,
            "context_padding_mask": context_padding_mask,
        }

    def forward_from_context(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
        polarity_idx: int,
    ) -> torch.Tensor:
        return self.flow_head(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=(
                context["task_scene_cond"]
                + self.polarity_embedding.weight[int(polarity_idx)]
            ),
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )


def _serial_sample(
    model: _FakeDipoleModel,
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
    n_steps: int,
    omega: float,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    x = torch.zeros(batch_size, model.action_dim, action_horizon)
    context = model.encode_multimodal_context(
        images=images,
        proprio=proprio,
        language=["task"] * batch_size,
    )
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps))
        v_pos = model.forward_from_context(
            x_t=x, t=t, context=context, polarity_idx=1
        )
        v_neg = model.forward_from_context(
            x_t=x, t=t, context=context, polarity_idx=0
        )
        x = x + ((1.0 + omega) * v_pos - omega * v_neg) / float(n_steps)
    return x.transpose(1, 2)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("omega", [0.2, 1.0])
def test_batched_guidance_matches_serial_reference(
    batch_size: int, omega: float
) -> None:
    torch.manual_seed(0)
    images = torch.randn(batch_size, 2, 3, 4, 4)
    proprio = torch.randn(batch_size, 4)
    reference_model = _FakeDipoleModel()
    batched_model = _FakeDipoleModel()
    batched_model.load_state_dict(reference_model.state_dict())

    expected = _serial_sample(
        reference_model,
        images=images,
        proprio=proprio,
        action_horizon=4,
        n_steps=5,
        omega=omega,
    )
    actual = _sample_guided_action_sequence(
        batched_model,
        images=images,
        proprio=proprio,
        language=["task"] * batch_size,
        action_horizon=4,
        n_steps=5,
        omega=omega,
        deterministic=True,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert batched_model.flow_head.calls == 5


def test_zero_omega_uses_only_positive_branch() -> None:
    model = _FakeDipoleModel()
    output = _sample_guided_action_sequence(
        model,
        images=torch.randn(2, 2, 3, 4, 4),
        proprio=torch.randn(2, 4),
        language=["task", "task"],
        action_horizon=4,
        n_steps=6,
        omega=0.0,
        deterministic=True,
    )

    assert output.shape == (2, 4, model.action_dim)
    assert model.flow_head.calls == 6


@pytest.mark.parametrize("requested", [1, 4, 8])
def test_execute_horizon_accepts_valid_values(requested: int) -> None:
    assert _resolve_execute_horizon(requested, action_horizon=8) == requested


@pytest.mark.parametrize("requested", [0, 9])
def test_execute_horizon_rejects_out_of_range_values(requested: int) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _resolve_execute_horizon(requested, action_horizon=8)
