from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.offline.src.train_offline_cfgrl import (
    _branch_only_update,
    _compute_cfgrl_neg_weights,
    _normalize_cfg_mode,
)
from robosuite.pipeline.offline.utils.branch_weights import RoutedSigmoidBranchWeightPolicy


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("CFGRL tests require CUDA; CPU fallback is forbidden.")
    return torch.device("cuda:0")


class _FakeFlowModel(nn.Module):
    def __init__(self, value: float, *, device: torch.device) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor(value, device=device))
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
        _ = t, images, proprio, language
        self.calls += 1
        return x_t * self.param + self.param


def _make_batch(device: torch.device) -> DipoleBatch:
    batch_size = 3
    return DipoleBatch(
        image_obs=torch.zeros(batch_size, 1, 3, 4, 4, device=device),
        image_obs_raw=torch.zeros(batch_size, 1, 3, 4, 4, device=device),
        proprio=torch.ones(batch_size, 2, device=device),
        proprio_raw=torch.ones(batch_size, 2, device=device),
        action_sequences=torch.ones(batch_size, 2, 2, device=device),
        action_sequences_raw=torch.ones(batch_size, 2, 2, device=device),
        is_intervention=torch.tensor([True, False, False], device=device),
        metadata={"route": ["pos_only", "neg_only", "advantage"]},
    )


def _make_core(device: torch.device) -> SimpleNamespace:
    model_pos = _FakeFlowModel(0.1, device=device)
    model_neg = _FakeFlowModel(-0.1, device=device)
    return SimpleNamespace(
        device=device,
        model_pos=model_pos,
        model_neg=model_neg,
        optimizer_pos=torch.optim.SGD(model_pos.parameters(), lr=0.1),
        optimizer_neg=torch.optim.SGD(model_neg.parameters(), lr=0.1),
        scaler_pos=torch.amp.GradScaler(device=device.type, enabled=False),
        scaler_neg=torch.amp.GradScaler(device=device.type, enabled=False),
        language_instruction="task",
        config=SimpleNamespace(
            lambda_endpoint=0.0,
            lambda_smooth=0.0,
            grad_clip_norm=10.0,
            beta=2.0,
            k=-0.25,
        ),
    )


@pytest.mark.parametrize("branch", ["pos", "neg"])
def test_branch_only_update_isolated(branch: str, cuda_device: torch.device) -> None:
    core = _make_core(cuda_device)
    batch = _make_batch(cuda_device)
    before_pos = core.model_pos.param.detach().clone()
    before_neg = core.model_neg.param.detach().clone()
    torch.cuda.manual_seed_all(0)

    metrics = _branch_only_update(
        core,
        batch,
        branch=branch,
        weights=torch.ones(batch.batch_size, device=cuda_device),
        want_metrics=True,
    )
    torch.cuda.synchronize(cuda_device)

    if branch == "pos":
        assert not torch.equal(core.model_pos.param.detach(), before_pos)
        assert torch.equal(core.model_neg.param.detach(), before_neg)
        assert core.model_pos.calls == 1
        assert core.model_neg.calls == 0
        assert core.model_pos.param.grad is not None
        assert core.model_neg.param.grad is None
        assert core.optimizer_neg.state == {}
    else:
        assert torch.equal(core.model_pos.param.detach(), before_pos)
        assert not torch.equal(core.model_neg.param.detach(), before_neg)
        assert core.model_pos.calls == 0
        assert core.model_neg.calls == 1
        assert core.model_pos.param.grad is None
        assert core.model_neg.param.grad is not None
        assert core.optimizer_pos.state == {}
    assert metrics[f"loss_{branch}"] > 0.0


def test_all_mode_uses_uniform_negative_weights(cuda_device: torch.device) -> None:
    core = _make_core(cuda_device)
    batch = _make_batch(cuda_device)

    class _RaiseIfCalled:
        def compute_g_for_batch(self, batch: DipoleBatch) -> torch.Tensor:
            raise AssertionError("all mode must not call the G provider")

    weights, metrics = _compute_cfgrl_neg_weights(
        core,
        batch,
        mode="all",
        branch_policy=RoutedSigmoidBranchWeightPolicy(),
        g_provider=_RaiseIfCalled(),
        want_metrics=True,
    )

    torch.testing.assert_close(weights, torch.ones(3, device=cuda_device))
    assert metrics["w_neg_mean"] == 1.0


def test_neg_weighted_matches_routed_dipole_weights(cuda_device: torch.device) -> None:
    core = _make_core(cuda_device)
    batch = _make_batch(cuda_device)

    class _FixedProvider:
        calls = 0

        def compute_g_for_batch(self, sub_batch: DipoleBatch) -> torch.Tensor:
            self.calls += 1
            assert sub_batch.batch_size == 1
            assert sub_batch.metadata["route"] == ["advantage"]
            return torch.tensor([0.5], dtype=torch.float32, device=cuda_device)

    provider = _FixedProvider()

    def sigmoid_fn(
        raw: torch.Tensor,
        *,
        want_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        logit = float(core.config.beta) * (raw + float(core.config.k))
        w_pos = torch.sigmoid(logit)
        metrics = {"G_mean": float(raw.mean().item())} if want_metrics else {}
        return w_pos, 1.0 - w_pos, metrics

    core._g_weights_from_raw = sigmoid_fn
    weights, metrics = _compute_cfgrl_neg_weights(
        core,
        batch,
        mode="neg-weighted",
        branch_policy=RoutedSigmoidBranchWeightPolicy(),
        g_provider=provider,
        want_metrics=True,
    )
    expected = torch.stack(
        (
            torch.tensor(0.0, device=cuda_device),
            torch.tensor(1.0, device=cuda_device),
            1.0 - torch.sigmoid(torch.tensor(0.5, device=cuda_device)),
        )
    )

    torch.testing.assert_close(weights, expected)
    assert provider.calls == 1
    assert metrics["frac_pos_only"] == pytest.approx(1.0 / 3.0)
    assert metrics["frac_neg_only"] == pytest.approx(1.0 / 3.0)
    assert metrics["frac_advantage"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("mode", ["", "bad", "neg_weighted", "ALLL"])
def test_invalid_cfg_mode_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="all.*neg-weighted|neg-weighted.*all"):
        _normalize_cfg_mode(mode)
