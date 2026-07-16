"""CUDA-only tests for composable discriminator objectives."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from robosuite.discriminator.dyn_disc.detectors.pu_bce import pu_risk
from robosuite.pipeline.offline.discriminator.objectives import (
    CUDAPoolSampler,
    LossTermConfig,
    NNPUParameters,
    OFFLINE_GT_NEGATIVE,
    OFFLINE_POSITIVE,
    PRETRAIN_POSITIVE,
    PRETRAIN_UNLABELED,
    build_objective,
)


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for discriminator objective tests.")
    return torch.device("cuda:0")


def _pools(device: torch.device, *, dim: int = 3) -> dict[str, torch.Tensor]:
    return {
        PRETRAIN_POSITIVE: torch.randn((20, dim), device=device),
        PRETRAIN_UNLABELED: torch.randn((30, dim), device=device),
        OFFLINE_POSITIVE: torch.randn((9, dim), device=device),
        OFFLINE_GT_NEGATIVE: torch.randn((2, dim), device=device),
    }


def _config(*, steps_per_epoch: int | None = None) -> dict:
    return {
        "steps_per_epoch": steps_per_epoch,
        "terms": {
            "nnpu_replay": {
                "type": "nnpu",
                "enabled": True,
                "weight": 1.0,
                "batch_size": 8,
                "positive_fraction": 0.5,
            },
            "supervised_gt_bce": {
                "type": "supervised_bce",
                "enabled": True,
                "weight": 2.0,
                "batch_size": 10,
                "positive_fraction": 0.4,
                "class_weights": {"positive": 0.25, "negative": 0.75},
            },
        },
    }


def _parameters() -> NNPUParameters:
    return NNPUParameters(
        pi_p=0.3,
        surrogate="logistic",
        nn_correction=True,
        beta=0.0,
    )


def test_default_ratios_sample_each_named_pool_independently_with_replacement() -> None:
    device = _cuda()
    pools = _pools(device)
    objective = build_objective(
        _config(), pools=pools, nnpu_parameters=_parameters(), device=device, seed=17
    )
    batches = objective.sample_batches()

    assert batches[PRETRAIN_POSITIVE].shape == (4, 3)
    assert batches[PRETRAIN_UNLABELED].shape == (4, 3)
    assert batches[OFFLINE_POSITIVE].shape == (4, 3)
    assert batches[OFFLINE_GT_NEGATIVE].shape == (6, 3)
    # GT-N has only two frames, so a six-sample batch proves replacement is allowed.
    assert torch.unique(batches[OFFLINE_GT_NEGATIVE], dim=0).shape[0] <= 2

    replay = build_objective(
        _config(), pools=pools, nnpu_parameters=_parameters(), device=device, seed=17
    )
    replay_batches = replay.sample_batches()
    for name, expected in batches.items():
        torch.testing.assert_close(replay_batches[name], expected)


def test_baseline_batches_draw_1024_samples_across_four_pools() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["nnpu_replay"].update(
        {"batch_size": 512, "positive_fraction": 0.5}
    )
    config["terms"]["supervised_gt_bce"].update(
        {"batch_size": 512, "positive_fraction": 0.5}
    )
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
    )

    batches = objective.sample_batches()

    assert {name: int(batch.shape[0]) for name, batch in batches.items()} == {
        PRETRAIN_POSITIVE: 256,
        PRETRAIN_UNLABELED: 256,
        OFFLINE_POSITIVE: 256,
        OFFLINE_GT_NEGATIVE: 256,
    }
    assert sum(int(batch.shape[0]) for batch in batches.values()) == 1024


def test_composite_loss_matches_manual_nnpu_and_weighted_gt_bce() -> None:
    device = _cuda()
    objective = build_objective(
        _config(steps_per_epoch=3),
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=1,
    )
    logits = {
        PRETRAIN_POSITIVE: torch.tensor([1.0, -0.5], device=device),
        PRETRAIN_UNLABELED: torch.tensor([-1.0, 0.25, 0.5], device=device),
        OFFLINE_POSITIVE: torch.tensor([0.5, 1.5], device=device),
        OFFLINE_GT_NEGATIVE: torch.tensor([-0.5, 0.0, 1.0], device=device),
    }
    result = objective.compute(logits)

    expected_nnpu = pu_risk(
        logits[PRETRAIN_POSITIVE],
        logits[PRETRAIN_UNLABELED],
        pi_p=0.3,
        surrogate="logistic",
        nn_correction=True,
        beta=0.0,
    )["risk"]
    expected_gt = (
        0.25 * F.softplus(-logits[OFFLINE_POSITIVE]).mean()
        + 0.75 * F.softplus(logits[OFFLINE_GT_NEGATIVE]).mean()
    )
    torch.testing.assert_close(result.raw_losses["nnpu_replay"], expected_nnpu)
    torch.testing.assert_close(result.raw_losses["supervised_gt_bce"], expected_gt)
    torch.testing.assert_close(result.loss, expected_nnpu + 2.0 * expected_gt)
    torch.testing.assert_close(
        result.metrics["gt_bce/positive"],
        F.softplus(-logits[OFFLINE_POSITIVE]).mean(),
    )
    torch.testing.assert_close(
        result.metrics["gt_bce/negative"],
        F.softplus(logits[OFFLINE_GT_NEGATIVE]).mean(),
    )


def test_gt_negative_gradient_survives_nnpu_clamp_and_lowers_logit() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["nnpu_replay"]["batch_size"] = 4
    config["terms"]["supervised_gt_bce"].update(
        {
            "weight": 1.0,
            "batch_size": 4,
            "positive_fraction": 0.5,
            "class_weights": {"positive": 0.5, "negative": 0.5},
        }
    )
    parameters = NNPUParameters(
        pi_p=0.5,
        surrogate="sigmoid",
        nn_correction=True,
        beta=0.0,
    )
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=parameters,
        device=device,
        seed=0,
    )
    gt_negative = torch.nn.Parameter(torch.full((2,), 1.0, device=device))
    optimizer = torch.optim.SGD([gt_negative], lr=0.1)
    before = gt_negative.detach().clone()
    result = objective.compute(
        {
            PRETRAIN_POSITIVE: torch.full((2,), 10.0, device=device),
            PRETRAIN_UNLABELED: torch.full((2,), -10.0, device=device),
            OFFLINE_POSITIVE: torch.full((2,), 1.0, device=device),
            OFFLINE_GT_NEGATIVE: gt_negative,
        }
    )
    assert result.metrics["nnpu/clamped"].item() == 1.0
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    assert gt_negative.grad is not None
    assert torch.all(gt_negative.grad > 0.0)
    optimizer.step()
    assert torch.all(gt_negative.detach() < before)


def test_head_path_uses_joint_loss_and_expected_batch_sizes() -> None:
    device = _cuda()
    objective = build_objective(
        _config(steps_per_epoch=2),
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=3,
    )
    head = torch.nn.Linear(3, 1, device=device)
    result = objective(head)
    assert result.loss.is_cuda and result.loss.ndim == 0
    assert result.batch_sizes == {
        PRETRAIN_POSITIVE: 4,
        PRETRAIN_UNLABELED: 4,
        OFFLINE_POSITIVE: 4,
        OFFLINE_GT_NEGATIVE: 6,
    }
    result.loss.backward()
    assert head.weight.grad is not None


def test_steps_per_epoch_derivation_and_explicit_override() -> None:
    device = _cuda()
    pools = _pools(device)
    derived = build_objective(
        _config(), pools=pools, nnpu_parameters=_parameters(), device=device, seed=0
    )
    assert derived.steps_per_epoch == 12  # floor(20 / 4 + 30 / 4)
    explicit = build_objective(
        _config(steps_per_epoch=7),
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
    )
    assert explicit.steps_per_epoch == 7


def test_zero_gt_weight_reduces_to_pretrain_nnpu_replay() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["supervised_gt_bce"]["weight"] = 0.0
    pools = _pools(device)
    pools.pop(OFFLINE_POSITIVE)
    pools.pop(OFFLINE_GT_NEGATIVE)
    objective = build_objective(
        config, pools=pools, nnpu_parameters=_parameters(), device=device, seed=5
    )
    result = objective.compute(
        {
            PRETRAIN_POSITIVE: torch.tensor([1.0, 2.0], device=device),
            PRETRAIN_UNLABELED: torch.tensor([-1.0, 0.0], device=device),
        }
    )
    assert set(result.raw_losses) == {"nnpu_replay"}
    torch.testing.assert_close(result.loss, result.raw_losses["nnpu_replay"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"batch_size": 3, "positive_fraction": 0.5}, "must be an integer"),
        ({"batch_size": 4, "positive_fraction": 0.0}, "non-empty"),
        ({"weight": -1.0}, "non-negative"),
        (
            {
                "type": "supervised_bce",
                "class_weights": {"positive": 0.6, "negative": 0.5},
            },
            "sum to 1",
        ),
    ],
)
def test_loss_term_validation(kwargs: dict, message: str) -> None:
    base = {"name": "term", "type": "nnpu"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        LossTermConfig(**base)


def test_unknown_loss_and_missing_active_pool_fail_fast() -> None:
    with pytest.raises(ValueError, match="Unknown loss type"):
        LossTermConfig(name="bad", type="focal")

    device = _cuda()
    pools = _pools(device)
    pools.pop(OFFLINE_GT_NEGATIVE)
    with pytest.raises(ValueError, match="missing pools"):
        build_objective(
            _config(steps_per_epoch=1),
            pools=pools,
            nnpu_parameters=_parameters(),
            device=device,
            seed=0,
        )


def test_cuda_guards_reject_cpu_without_fallback() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        CUDAPoolSampler({}, device="cpu", seed=0)
