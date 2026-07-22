"""CUDA-only tests for composable discriminator objectives."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from robosuite.discriminator.dyn_disc.detectors.pu_bce import pu_risk
import robosuite.pipeline.modules.training.discriminator.objectives as objectives_module
from robosuite.pipeline.modules.training.discriminator.objectives import (
    CUDAPoolSampler,
    FixedLogitNormalizer,
    LossTermConfig,
    NNPUParameters,
    OFFLINE_GT_NEGATIVE,
    OFFLINE_POSITIVE,
    PRETRAIN_POSITIVE,
    PRETRAIN_UNLABELED,
    QuadraticLogitCapConfig,
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
            "gt_positive": {
                "type": "positive_logistic",
                "enabled": True,
                "weight": 2.0,
                "batch_size": 4,
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
    )
    replay_batches = replay.sample_batches()
    for name, expected in batches.items():
        torch.testing.assert_close(replay_batches[name], expected)


def test_active_batches_draw_512_samples_across_four_pools() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["nnpu_replay"].update(
        {"batch_size": 256, "positive_fraction": 0.5}
    )
    config["terms"]["gt_positive"]["batch_size"] = 128
    config["terms"]["gt_negative"]["batch_size"] = 128
    objective = build_objective(
        config,
        pools=_pools(device),
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
    )

    batches = objective.sample_batches()

    assert {name: int(batch.shape[0]) for name, batch in batches.items()} == {
        PRETRAIN_POSITIVE: 128,
        PRETRAIN_UNLABELED: 128,
        OFFLINE_POSITIVE: 128,
        OFFLINE_GT_NEGATIVE: 128,
    }
    assert sum(int(batch.shape[0]) for batch in batches.values()) == 512


def test_composite_loss_matches_manual_three_risk_objective() -> None:
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
    expected_positive_bce = F.softplus(-logits[OFFLINE_POSITIVE]).mean()
    expected_positive = expected_positive_bce
    expected_negative = F.softplus(logits[OFFLINE_GT_NEGATIVE]).mean()
    torch.testing.assert_close(result.raw_losses["nnpu_replay"], expected_nnpu)
    torch.testing.assert_close(result.raw_losses["gt_positive"], expected_positive)
    torch.testing.assert_close(result.raw_losses["gt_negative"], expected_negative)
    torch.testing.assert_close(
        result.loss, expected_nnpu + 2.0 * expected_positive + 3.0 * expected_negative
    )
    torch.testing.assert_close(result.metrics["gt/positive_logistic"], expected_positive)
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
    torch.testing.assert_close(
        result.metrics["regularization/quadratic_logit_cap/nnpu_replay"],
        expected_cap,
    )
    assert "regularization/quadratic_logit_cap/gt_positive" not in result.metrics
    assert "regularization/quadratic_logit_cap/gt_negative" not in result.metrics


def test_all_terms_quadratic_cap_averages_normalized_term_penalties() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["quadratic_logit_cap"] = {
        "enabled": True,
        "scope": "all_terms",
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
        PRETRAIN_POSITIVE: torch.nn.Parameter(
            torch.tensor([7.0, -7.0], device=device)
        ),
        PRETRAIN_UNLABELED: torch.nn.Parameter(
            torch.tensor([9.0, -9.0], device=device)
        ),
        OFFLINE_POSITIVE: torch.nn.Parameter(
            torch.tensor([11.0, 7.0], device=device)
        ),
        OFFLINE_GT_NEGATIVE: torch.nn.Parameter(
            torch.tensor([-11.0, -7.0], device=device)
        ),
    }

    result = objective.compute(logits)
    effective = {name: normalizer(value) for name, value in logits.items()}
    term_logits = {
        "nnpu_replay": torch.cat(
            [effective[PRETRAIN_POSITIVE], effective[PRETRAIN_UNLABELED]], dim=0
        ),
        "gt_positive": effective[OFFLINE_POSITIVE],
        "gt_negative": effective[OFFLINE_GT_NEGATIVE],
    }
    term_caps = {
        name: torch.relu(values.abs() - 2.0).square().mean()
        for name, values in term_logits.items()
    }
    term_fractions = {
        name: (values.abs() > 2.0).to(dtype=torch.float32).mean()
        for name, values in term_logits.items()
    }
    expected_cap = torch.stack(list(term_caps.values())).mean()
    expected_fraction = torch.stack(list(term_fractions.values())).mean()
    three_terms = sum(result.weighted_losses.values())

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
        expected_fraction,
    )
    for term_name in term_caps:
        torch.testing.assert_close(
            result.metrics[f"regularization/quadratic_logit_cap/{term_name}"],
            term_caps[term_name],
        )
        torch.testing.assert_close(
            result.metrics[
                "regularization/quadratic_logit_cap_fraction_outside/"
                f"{term_name}"
            ],
            term_fractions[term_name],
        )

    cap_only = result.loss - three_terms
    cap_gradients = torch.autograd.grad(cap_only, tuple(logits.values()))
    assert all(torch.count_nonzero(gradient).item() > 0 for gradient in cap_gradients)


def test_all_terms_quadratic_cap_averages_only_active_terms() -> None:
    device = _cuda()
    config = _config(steps_per_epoch=1)
    config["terms"]["gt_negative"]["weight"] = 0.0
    config["quadratic_logit_cap"] = {
        "enabled": True,
        "scope": "all_terms",
        "cap": 2.0,
        "weight": 0.5,
    }
    pools = _pools(device)
    pools.pop(OFFLINE_GT_NEGATIVE)
    objective = build_objective(
        config,
        pools=pools,
        nnpu_parameters=_parameters(),
        device=device,
        seed=0,
    )
    logits = {
        PRETRAIN_POSITIVE: torch.tensor([4.0], device=device),
        PRETRAIN_UNLABELED: torch.tensor([-4.0], device=device),
        OFFLINE_POSITIVE: torch.tensor([6.0], device=device),
    }

    result = objective.compute(logits)
    replay_cap = torch.tensor(4.0, device=device)
    positive_cap = torch.tensor(16.0, device=device)
    expected_cap = (replay_cap + positive_cap) / 2.0

    torch.testing.assert_close(
        result.metrics["regularization/quadratic_logit_cap"], expected_cap
    )
    assert "regularization/quadratic_logit_cap/gt_negative" not in result.metrics


@pytest.mark.parametrize("scope", ["nnpu_replay", "all_terms"])
def test_quadratic_logit_cap_accepts_supported_scopes(scope: str) -> None:
    assert QuadraticLogitCapConfig(scope=scope).scope == scope


def test_quadratic_logit_cap_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="nnpu_replay.*all_terms"):
        QuadraticLogitCapConfig(scope="unknown")


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
    expected_positive_grad = -torch.sigmoid(-gt_positive.detach()) / gt_positive.numel()
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

