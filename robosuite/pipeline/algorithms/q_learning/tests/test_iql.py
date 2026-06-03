"""CPU-only tests for ResNet-50 IQL Q-chunking."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from robosuite.pipeline.algorithms.q_learning import networks
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig, IQLStepBatch
from robosuite.pipeline.algorithms.q_learning.data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
)
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.losses import (
    bellman_q_loss,
    compute_advantage,
    expectile_v_loss,
)


class _TinyBackbone(nn.Module):
    """Small torchvision-ResNet stand-in with a 2048-d output."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(4, networks.RESNET50_OUTPUT_DIM)
        self.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(images)).flatten(1)
        return self.fc(self.proj(x))


@pytest.fixture()
def resnet_ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(networks, "resnet50", lambda weights=None: _TinyBackbone())
    path = tmp_path / "resnet50.pth"
    torch.save(_TinyBackbone().state_dict(), path)
    return str(path)


def _make_cfg(resnet_ckpt: str, action_horizon: int = 2) -> IQLConfig:
    return IQLConfig(
        action_horizon=action_horizon,
        discount=0.9,
        expectile_tau=0.7,
        q_lr=1e-3,
        v_lr=1e-3,
        target_polyak=0.1,
        n_step_aggregate=True,
        hidden_dims=(32, 32),
        state_feature_dim=16,
        action_feature_dim=16,
        resnet_pretrained_path=resnet_ckpt,
        grad_clip_norm=1.0,
        weight_decay=0.0,
        device="cpu",
        disc_reward_coef=0.0,
        output_reward_coef=1.0,
    )


def _make_iql(resnet_ckpt: str, *, camera_names: tuple[str, ...] = ("front", "wrist")) -> IQLLearner:
    return IQLLearner(
        cfg=_make_cfg(resnet_ckpt),
        camera_names=camera_names,
        proprio_dim=5,
        action_dim=4,
    )


def _make_step_batch(B: int = 4, D_a: int = 4, H: int = 2) -> IQLStepBatch:
    torch.manual_seed(0)
    return IQLStepBatch(
        image_obs_raw=torch.rand(B, 2, 3, 8, 8),
        proprio_raw=torch.randn(B, 5),
        next_image_obs_raw=torch.rand(B, 2, 3, 8, 8),
        next_proprio_raw=torch.randn(B, 5),
        action_chunk=torch.randn(B, H, D_a),
        rewards=torch.randn(B, 1),
        dones=torch.zeros(B, 1),
        is_online=torch.zeros(B, 1),
        is_intervention=torch.zeros(B, 1),
        metadata={},
    )


def test_compute_advantage_shape_and_value() -> None:
    q1 = torch.tensor([[1.0], [2.0], [3.0]])
    q2 = torch.tensor([[2.0], [1.0], [4.0]])
    v = torch.tensor([[0.5], [0.5], [0.5]])
    adv = compute_advantage(q1, q2, v)
    assert adv.shape == (3,)
    assert torch.allclose(adv, torch.tensor([0.5, 0.5, 2.5]))


def test_expectile_v_loss_tau_half_is_half_mse() -> None:
    diff = torch.tensor([[1.0], [-2.0], [0.5]])
    assert torch.allclose(expectile_v_loss(diff, tau=0.5), 0.5 * diff.square().mean())


def test_expectile_v_loss_asymmetry() -> None:
    assert torch.allclose(expectile_v_loss(torch.tensor([[1.0], [-1.0]]), tau=0.7), torch.tensor(0.5))


def test_bellman_q_loss_matches_mse() -> None:
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.5], [1.0], [4.0]])
    assert torch.allclose(bellman_q_loss(pred, target), torch.nn.functional.mse_loss(pred, target))


def test_aggregate_chunk_reward_closed_form() -> None:
    out = aggregate_chunk_reward(torch.ones(1, 4), 0.9)
    assert out.shape == (1, 1)
    assert math.isclose(float(out.item()), sum(0.9 ** i for i in range(4)), rel_tol=1e-6)


def test_chunk_done_mask() -> None:
    step_dones = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert torch.allclose(chunk_done_mask(step_dones), torch.tensor([[1.0], [0.0]]))


def test_iql_update_and_warmup_move_trainable_params(resnet_ckpt: str) -> None:
    iql = _make_iql(resnet_ckpt)
    batch = _make_step_batch()
    q_before = [param.detach().clone() for param in iql._q_trainable]
    v_before = [param.detach().clone() for param in iql._v_trainable]
    iql.warmup_value_only(batch)
    metrics = iql.update(batch)
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(q_before, iql._q_trainable))
    assert any(not torch.equal(a, b) for a, b in zip(v_before, iql._v_trainable))


def test_backbones_are_frozen_and_compact_state_excludes_them(resnet_ckpt: str) -> None:
    iql = _make_iql(resnet_ckpt)
    for network in (iql.q1, iql.q2, iql.v, iql.target_v):
        assert all(not param.requires_grad for param in network.vis_encoder.backbone.parameters())
        network.train()
        assert not network.vis_encoder.backbone.training
    state = iql.state_dict()
    assert state["schema_version"] == 2
    for key in ("q1", "q2", "v", "target_v"):
        assert not any(name.startswith("vis_encoder.backbone.") for name in state[key])


def test_iql_state_dict_roundtrip(resnet_ckpt: str) -> None:
    iql_a = _make_iql(resnet_ckpt)
    iql_b = _make_iql(resnet_ckpt)
    iql_a.warmup_value_only(_make_step_batch())
    iql_b.load_state_dict(iql_a.state_dict(), strict=True)
    for param_a, param_b in zip(iql_a._v_trainable, iql_b._v_trainable):
        assert torch.equal(param_a, param_b)


def test_iql_load_state_dict_mismatched_camera_names_raises(resnet_ckpt: str) -> None:
    iql_a = _make_iql(resnet_ckpt)
    iql_b = _make_iql(resnet_ckpt, camera_names=("wrist", "front"))
    with pytest.raises(ValueError, match="camera_names mismatch"):
        iql_b.load_state_dict(iql_a.state_dict(), strict=True)


def test_iql_rejects_legacy_state(resnet_ckpt: str) -> None:
    with pytest.raises(ValueError, match="Re-run Q/V warmup"):
        _make_iql(resnet_ckpt).load_state_dict({"context_dim": 16}, strict=True)
