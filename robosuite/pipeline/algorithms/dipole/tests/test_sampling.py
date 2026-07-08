"""CPU-only tests for DIPOLE guided action sampling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.algorithms.dipole.models.flow import (
    DipoleFlowPolicy,
    _sample_guided_action_sequence,
)
from robosuite.pipeline.algorithms.dipole.models.lora import LoRARuntime
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
    """Stub mirroring the dual-LoRA DIPOLE model API used by sampling.

    The negative branch differs from the positive branch through
    ``branch_task_scene_cond`` (emulating the aggregator-side neg LoRA delta; the
    pos adapter is a no-op here, mirroring its zero-init at start); the stub flow
    head itself is LoRA-agnostic, so this exercises the sampling orchestration
    (2x-batch build, row mask plumbing, branch combination).
    """

    def __init__(self, action_dim: int = 3, context_dim: int = 5) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.lora_runtime = LoRARuntime()
        self.flow_head = _CountingFlowHead()
        # Fixed negative-branch condition delta (stands in for the neg aggregator LoRA).
        self.register_buffer(
            "neg_cond_delta",
            torch.tensor([0.3, -0.2, 0.4, 0.1, -0.1])[:context_dim],
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
            # Aggregator inputs (unused by the stub aggregator, present for parity).
            "fused_tokens": context_tokens,
            "token_padding_mask": context_padding_mask,
            "language_global": task_scene_cond,
        }

    def branch_task_scene_cond(
        self, context: dict[str, torch.Tensor], *, branch: str
    ) -> torch.Tensor:
        # pos adapter is a no-op (zero delta); neg adapter adds a fixed delta.
        if branch == "neg":
            return context["task_scene_cond"] + self.neg_cond_delta
        return context["task_scene_cond"]

    def negative_task_scene_cond(
        self, context: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.branch_task_scene_cond(context, branch="neg")

    def forward_from_context(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
        negative: bool,
    ) -> torch.Tensor:
        return self.flow_head(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=context["task_scene_cond"],
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
    neg_context = dict(context)
    neg_context["task_scene_cond"] = model.negative_task_scene_cond(context)
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps))
        v_pos = model.forward_from_context(
            x_t=x, t=t, context=context, negative=False
        )
        v_neg = model.forward_from_context(
            x_t=x, t=t, context=neg_context, negative=True
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


class _FakeMaskedFlowHead(nn.Module):
    def __init__(self, owner: "_FakeUpdateModel") -> None:
        super().__init__()
        object.__setattr__(self, "owner", owner)
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
        _ = timesteps, task_scene_cond, context_tokens, context_padding_mask
        row_mask = self.owner.lora_runtime.row_mask
        assert row_mask is not None
        scale = torch.where(row_mask, self.owner.neg_lora_param, self.owner.pos_lora_param)
        return torch.ones_like(x_t) * scale.view(-1, 1, 1)


class _FakeUpdateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_runtime = LoRARuntime()
        self.pos_lora_param = nn.Parameter(torch.tensor(0.1))
        self.neg_lora_param = nn.Parameter(torch.tensor(-0.1))
        self.flow_head = _FakeMaskedFlowHead(self)

    def encode_multimodal_context(
        self,
        *,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str],
    ) -> dict[str, torch.Tensor]:
        batch_size = int(proprio.shape[0])
        assert len(language) == batch_size
        return {
            "task_scene_cond": torch.zeros(batch_size, 1),
            "context_tokens": torch.zeros(batch_size, 1, 1),
            "context_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool),
            "fused_tokens": torch.zeros(batch_size, 1, 1),
            "token_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool),
            "language_global": torch.zeros(batch_size, 1),
        }

    def branch_task_scene_cond(
        self, context: dict[str, torch.Tensor], *, branch: str
    ) -> torch.Tensor:
        assert branch in ("pos", "neg")
        return context["task_scene_cond"]


def test_update_uses_single_masked_branch_forward() -> None:
    policy = object.__new__(DipoleFlowPolicy)
    policy.device = torch.device("cpu")
    policy.model = _FakeUpdateModel()
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
    policy._lora_params = [policy.model.pos_lora_param, policy.model.neg_lora_param]
    policy.optimizer = torch.optim.SGD(policy._lora_params, lr=0.01)
    policy.scaler = torch.amp.GradScaler(enabled=False, device=policy.device)

    batch = DipoleBatch(
        image_obs=torch.zeros(3, 1, 3, 4, 4),
        image_obs_raw=torch.zeros(3, 1, 3, 4, 4),
        proprio=torch.zeros(3, 2),
        proprio_raw=torch.zeros(3, 2),
        action_sequences=torch.zeros(3, 2, 2),
        action_sequences_raw=torch.zeros(3, 2, 2),
        is_intervention=torch.zeros(3, dtype=torch.bool),
        metadata={},
    )

    metrics = policy.update(batch)

    assert policy.model.flow_head.calls == 1
    assert "loss_pos" in metrics
    assert "loss_neg" in metrics
    assert policy.model.pos_lora_param.grad is not None
    assert policy.model.neg_lora_param.grad is not None


@pytest.mark.parametrize("requested", [1, 4, 8])
def test_execute_horizon_accepts_valid_values(requested: int) -> None:
    assert _resolve_execute_horizon(requested, action_horizon=8) == requested


@pytest.mark.parametrize("requested", [0, 9])
def test_execute_horizon_rejects_out_of_range_values(requested: int) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _resolve_execute_horizon(requested, action_horizon=8)
