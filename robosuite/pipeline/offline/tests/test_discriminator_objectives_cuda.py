"""CUDA-only tests for composable discriminator objectives."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from robosuite.discriminator.dyn_disc.detectors.pu_bce import pu_risk
import robosuite.pipeline.offline.discriminator.objectives as objectives_module
from robosuite.pipeline.offline.discriminator.objectives import (
    CUDAPoolSampler,
    FixedLogitNormalizer,
    LossTermConfig,
    NNPUParameters,
    OFFLINE_GT_NEGATIVE,
    OFFLINE_POSITIVE,
    PRETRAIN_POSITIVE,
    PRETRAIN_UNLABELED,
    build_objective,
)


POSITIVE_SAFETY_BOUNDARY = 1.25
SAFETY_MARGIN_WEIGHT = 0.75
MARGIN_DELTA = 0.4
TEMPERATURE = 0.5


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
            "gt_positive": {
                "type": "positive_safety_margin",
                "enabled": True,
                "weight": 2.0,
                "batch_size": 4,
                "safety_margin_weight": SAFETY_MARGIN_WEIGHT,
                "margin_delta": MARGIN_DELTA,
                "temperature": TEMPERATURE,
                "boundary_source": "parent_checkpoint",
            },
            "gt_negative": {
                "type": "negative_logistic",
                "enabled": True,
                "weight": 3.0,
                "batch_size": 6,
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
        _config(),
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=17,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    batches = objective.sample_batches()

    assert batches[PRETRAIN_POSITIVE].shape == (4, 3)
    assert batches[PRETRAIN_UNLABELED].shape == (4, 3)
    assert batches[OFFLINE_POSITIVE].shape == (4, 3)
    assert batches[OFFLINE_GT_NEGATIVE].shape == (6, 3)
    # GT-N has only two frames, so a six-sample batch proves replacement is allowed.
    assert torch.unique(batches[OFFLINE_GT_NEGATIVE], dim=0).shape[0] <= 2

    replay = build_objective(
        _config(),
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=17,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
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
    config["terms"]["gt_positive"]["batch_size"] = 256
    config["terms"]["gt_negative"]["batch_size"] = 256
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )

    batches = objective.sample_batches()

    assert {name: int(batch.shape[0]) for name, batch in batches.items()} == {
        PRETRAIN_POSITIVE: 256,
        PRETRAIN_UNLABELED: 256,
        OFFLINE_POSITIVE: 256,
        OFFLINE_GT_NEGATIVE: 256,
    }
    assert sum(int(batch.shape[0]) for batch in batches.values()) == 1024


def test_composite_loss_matches_manual_three_risk_objective() -> None:
    device = _cuda()
    objective = build_objective(
        _config(steps_per_epoch=3),
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=1,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
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
    expected_positive_bce = F.softplus(-logits[OFFLINE_POSITIVE]).mean()
    expected_positive_safety = F.softplus(
        (
            POSITIVE_SAFETY_BOUNDARY
            + MARGIN_DELTA
            - logits[OFFLINE_POSITIVE]
        )
        / TEMPERATURE
    ).mean()
    expected_positive = (
        expected_positive_bce
        + SAFETY_MARGIN_WEIGHT * expected_positive_safety
    )
    expected_negative = F.softplus(logits[OFFLINE_GT_NEGATIVE]).mean()
    torch.testing.assert_close(result.raw_losses["nnpu_replay"], expected_nnpu)
    torch.testing.assert_close(result.raw_losses["gt_positive"], expected_positive)
    torch.testing.assert_close(result.raw_losses["gt_negative"], expected_negative)
    torch.testing.assert_close(
        result.loss, expected_nnpu + 2.0 * expected_positive + 3.0 * expected_negative
    )
    torch.testing.assert_close(
        result.metrics["gt/positive_bce"], expected_positive_bce
    )
    torch.testing.assert_close(
        result.metrics["gt/positive_safety_margin"], expected_positive_safety
    )
    torch.testing.assert_close(
        result.metrics["gt/positive_combined"], expected_positive
    )
    torch.testing.assert_close(
        result.metrics["safety/m_k"],
        logits[OFFLINE_POSITIVE].new_tensor(POSITIVE_SAFETY_BOUNDARY),
    )
    torch.testing.assert_close(
        result.metrics["safety/target_logit"],
        logits[OFFLINE_POSITIVE].new_tensor(
            POSITIVE_SAFETY_BOUNDARY + MARGIN_DELTA
        ),
    )
    torch.testing.assert_close(
        result.metrics["safety/margin_violation_fraction"],
        (
            logits[OFFLINE_POSITIVE]
            < POSITIVE_SAFETY_BOUNDARY + MARGIN_DELTA
        )
        .to(dtype=torch.float32)
        .mean(),
    )
    torch.testing.assert_close(
        result.metrics["gt/negative_logistic"], expected_negative
    )


def test_quadratic_logit_cap_matches_dyn_disc_on_normalized_replay_logits() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["gt_positive"] = {
        "type": "positive_logistic",
        "enabled": True,
        "weight": 2.0,
        "batch_size": 4,
    }
    config["quadratic_logit_cap"] = {
        "enabled": True,
        "scope": "nnpu_replay",
        "cap": 2.0,
        "weight": 0.25,
    }
    normalizer = FixedLogitNormalizer(enabled=True, center=1.0, scale=2.0)
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
        logit_normalizer=normalizer,
    )
    logits = {
        PRETRAIN_POSITIVE: torch.tensor([4.0, -3.0], device=device),
        PRETRAIN_UNLABELED: torch.tensor([0.5, -0.25], device=device),
        OFFLINE_POSITIVE: torch.tensor([100.0], device=device),
        OFFLINE_GT_NEGATIVE: torch.tensor([-100.0], device=device),
    }

    result = objective.compute(logits)
    effective = {name: normalizer(value) for name, value in logits.items()}
    replay_logits = torch.cat(
        [effective[PRETRAIN_POSITIVE], effective[PRETRAIN_UNLABELED]], dim=0
    )
    expected_cap = torch.relu(replay_logits.abs() - 2.0).square().mean()
    expected_nnpu = pu_risk(
        effective[PRETRAIN_POSITIVE],
        effective[PRETRAIN_UNLABELED],
        pi_p=0.3,
        surrogate="logistic",
        nn_correction=True,
        beta=0.0,
    )["risk"]
    expected_positive = F.softplus(-effective[OFFLINE_POSITIVE]).mean()
    expected_negative = F.softplus(effective[OFFLINE_GT_NEGATIVE]).mean()
    three_terms = sum(result.weighted_losses.values())

    torch.testing.assert_close(result.raw_losses["nnpu_replay"], expected_nnpu)
    torch.testing.assert_close(result.raw_losses["gt_positive"], expected_positive)
    torch.testing.assert_close(result.raw_losses["gt_negative"], expected_negative)
    torch.testing.assert_close(
        result.metrics["regularization/quadratic_logit_cap"], expected_cap
    )
    torch.testing.assert_close(
        result.metrics["regularization/quadratic_logit_cap_weighted"],
        0.25 * expected_cap,
    )
    torch.testing.assert_close(result.loss, three_terms + 0.25 * expected_cap)
    torch.testing.assert_close(
        result.metrics["regularization/quadratic_logit_cap_fraction_outside"],
        (replay_logits.abs() > 2.0).to(dtype=torch.float32).mean(),
    )


def test_zero_safety_margin_weight_reduces_exactly_to_positive_bce() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["gt_positive"]["safety_margin_weight"] = 0.0
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    logits = {
        PRETRAIN_POSITIVE: torch.tensor([1.0, -0.5], device=device),
        PRETRAIN_UNLABELED: torch.tensor([-1.0, 0.25], device=device),
        OFFLINE_POSITIVE: torch.tensor([-2.0, 0.5, 3.0], device=device),
        OFFLINE_GT_NEGATIVE: torch.tensor([-0.5, 1.0], device=device),
    }

    result = objective.compute(logits)
    expected_bce = F.softplus(-logits[OFFLINE_POSITIVE]).mean()

    torch.testing.assert_close(
        result.raw_losses["gt_positive"], expected_bce, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.metrics["gt/positive_combined"], expected_bce, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.weighted_losses["gt_positive"], 2.0 * expected_bce
    )


def test_gt_gradients_survive_nnpu_clamp_and_move_logits_in_correct_directions() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["nnpu_replay"]["batch_size"] = 4
    config["terms"]["gt_positive"].update({"weight": 1.0, "batch_size": 2})
    config["terms"]["gt_negative"].update({"weight": 1.0, "batch_size": 2})
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
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    gt_positive = torch.nn.Parameter(torch.full((2,), -1.0, device=device))
    gt_negative = torch.nn.Parameter(torch.full((2,), 1.0, device=device))
    optimizer = torch.optim.SGD([gt_positive, gt_negative], lr=0.1)
    positive_before = gt_positive.detach().clone()
    negative_before = gt_negative.detach().clone()
    result = objective.compute(
        {
            PRETRAIN_POSITIVE: torch.full((2,), 10.0, device=device),
            PRETRAIN_UNLABELED: torch.full((2,), -10.0, device=device),
            OFFLINE_POSITIVE: gt_positive,
            OFFLINE_GT_NEGATIVE: gt_negative,
        }
    )
    assert result.metrics["nnpu/clamped"].item() == 1.0
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    assert gt_positive.grad is not None
    assert gt_negative.grad is not None
    expected_positive_grad = -(
        torch.sigmoid(-gt_positive.detach())
        + SAFETY_MARGIN_WEIGHT
        / TEMPERATURE
        * torch.sigmoid(
            (
                POSITIVE_SAFETY_BOUNDARY
                + MARGIN_DELTA
                - gt_positive.detach()
            )
            / TEMPERATURE
        )
    ) / gt_positive.numel()
    torch.testing.assert_close(gt_positive.grad, expected_positive_grad)
    assert torch.all(gt_positive.grad < 0.0)
    assert torch.all(gt_negative.grad > 0.0)
    optimizer.step()
    assert torch.all(gt_positive.detach() > positive_before)
    assert torch.all(gt_negative.detach() < negative_before)


def test_head_path_uses_joint_loss_and_expected_batch_sizes() -> None:
    device = _cuda()
    objective = build_objective(
        _config(steps_per_epoch=2),
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=3,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
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
        _config(),
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    assert derived.steps_per_epoch == 12  # floor(20 / 4 + 30 / 4)
    explicit = build_objective(
        _config(steps_per_epoch=7),
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    assert explicit.steps_per_epoch == 7


@pytest.mark.parametrize(
    "term_name,pool_name",
    [
        ("gt_positive", OFFLINE_POSITIVE),
        ("gt_negative", OFFLINE_GT_NEGATIVE),
    ],
)
def test_zero_gt_weight_does_not_require_its_pool(
    term_name: str, pool_name: str
) -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"][term_name]["weight"] = 0.0
    pools = _pools(device)
    pools.pop(pool_name)
    objective = build_objective(
        config,
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=5,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    result = objective.compute(
        {
            PRETRAIN_POSITIVE: torch.tensor([1.0, 2.0], device=device),
            PRETRAIN_UNLABELED: torch.tensor([-1.0, 0.0], device=device),
            **(
                {}
                if pool_name == OFFLINE_POSITIVE
                else {OFFLINE_POSITIVE: torch.tensor([0.5], device=device)}
            ),
            **(
                {}
                if pool_name == OFFLINE_GT_NEGATIVE
                else {OFFLINE_GT_NEGATIVE: torch.tensor([-0.5], device=device)}
            ),
        }
    )
    assert term_name not in result.raw_losses


def test_nnpu_risk_receives_only_pretrain_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _cuda()
    objective = build_objective(
        _config(steps_per_epoch=1),
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
    )
    logits = {
        PRETRAIN_POSITIVE: torch.tensor([11.0, 12.0], device=device),
        PRETRAIN_UNLABELED: torch.tensor([21.0, 22.0], device=device),
        OFFLINE_POSITIVE: torch.tensor([31.0, 32.0], device=device),
        OFFLINE_GT_NEGATIVE: torch.tensor([41.0, 42.0], device=device),
    }
    called = False

    def recording_pu_risk(g_p: torch.Tensor, g_u: torch.Tensor, **kwargs):
        nonlocal called
        called = True
        torch.testing.assert_close(g_p, logits[PRETRAIN_POSITIVE])
        torch.testing.assert_close(g_u, logits[PRETRAIN_UNLABELED])
        return pu_risk(g_p, g_u, **kwargs)

    monkeypatch.setattr(objectives_module, "pu_risk", recording_pu_risk)
    objective.compute(logits)
    assert called


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
        (
            {"type": "positive_logistic", "positive_fraction": 0.5},
            "only valid for paired losses",
        ),
    ],
)
def test_loss_term_validation(kwargs: dict, message: str) -> None:
    base = {"name": "term", "type": "nnpu"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        LossTermConfig(**base)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("safety_margin_weight", -0.1, "safety_margin_weight.*non-negative"),
        ("margin_delta", -0.1, "margin_delta.*non-negative"),
        ("temperature", 0.0, "temperature.*positive"),
        ("temperature", float("inf"), "temperature.*finite"),
        ("boundary_source", "current_head", "boundary_source"),
    ],
)
def test_positive_safety_margin_term_rejects_invalid_parameters(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "name": "gt_positive",
        "type": "positive_safety_margin",
        "enabled": True,
        "weight": 2.0,
        "batch_size": 4,
        "safety_margin_weight": SAFETY_MARGIN_WEIGHT,
        "margin_delta": MARGIN_DELTA,
        "temperature": TEMPERATURE,
        "boundary_source": "parent_checkpoint",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        LossTermConfig(**kwargs)


@pytest.mark.parametrize(
    "boundary,message",
    [
        (None, "positive_safety_boundary"),
        (float("nan"), "positive_safety_boundary.*finite"),
    ],
)
def test_positive_safety_margin_requires_finite_parent_boundary(
    boundary: float | None,
    message: str,
) -> None:
    device = _cuda()
    kwargs = {}
    if boundary is not None:
        kwargs["positive_safety_boundary"] = boundary

    with pytest.raises(ValueError, match=message):
        build_objective(
            _config(steps_per_epoch=1),
            pools=_pools(device),
            nnpu_parameters=_parameters(),
            device=device,
            seed=0,
            **kwargs,
        )


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
            positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
        )


def test_legacy_supervised_bce_cannot_overlap_separate_gt_risks() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["legacy_gt_bce"] = {
        "type": "supervised_bce",
        "enabled": True,
        "weight": 1.0,
        "batch_size": 4,
        "positive_fraction": 0.5,
        "class_weights": {"positive": 0.5, "negative": 0.5},
    }
    with pytest.raises(ValueError, match="disjoint pools"):
        build_objective(
            config,
            pools=_pools(device),
            nnpu_parameters=_parameters(),
            device=device,
            seed=0,
            positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
        )


def test_legacy_supervised_bce_remains_supported_as_an_alternative() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"].pop("gt_positive")
    config["terms"].pop("gt_negative")
    config["terms"]["supervised_gt_bce"] = {
        "type": "supervised_bce",
        "enabled": True,
        "weight": 0.025,
        "batch_size": 8,
        "positive_fraction": 0.5,
        "class_weights": {"positive": 0.5, "negative": 0.5},
    }
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
    )
    batches = objective.sample_batches()
    assert batches[OFFLINE_POSITIVE].shape[0] == 4
    assert batches[OFFLINE_GT_NEGATIVE].shape[0] == 4
    assert "supervised_gt_bce" in objective(torch.nn.Linear(3, 1, device=device)).raw_losses


def test_cuda_guards_reject_cpu_without_fallback() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        CUDAPoolSampler({}, device="cpu", seed=0)
