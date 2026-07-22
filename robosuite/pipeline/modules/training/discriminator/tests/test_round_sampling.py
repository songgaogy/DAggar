"""Batch-online discriminator round sampling and feature-cache tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from robosuite.pipeline.modules.training.discriminator.objectives import (
    CUDAPoolSampler,
    OFFLINE_GT_NEGATIVE,
    OFFLINE_POSITIVE,
    PRETRAIN_POSITIVE,
    PRETRAIN_UNLABELED,
)
from robosuite.pipeline.modules.training.discriminator.pools import LatentTrajectory
from robosuite.pipeline.modules.training.discriminator.round_feature_cache import (
    load_round_feature_cache,
    save_round_feature_cache,
)
from robosuite.pipeline.modules.training.discriminator.round_sampling import (
    BatchOnlineCUDAPoolSampler,
    RecursiveRoundCUDASampler,
    recursive_round_weights,
    resolve_round_mixture_plan,
)
from robosuite.pipeline.modules.training.discriminator.runner import _feature_cache_key


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for discriminator round-sampling tests.")
    return torch.device("cuda:0")


def test_recursive_round_weights_match_moving_average() -> None:
    assert recursive_round_weights(
        current_round=0, history_mix_beta=0.5
    ) == {0: 1.0}
    assert recursive_round_weights(
        current_round=3, history_mix_beta=0.5
    ) == {0: 0.125, 1: 0.125, 2: 0.25, 3: 0.5}
    near_one = recursive_round_weights(
        current_round=2, history_mix_beta=0.999
    )
    assert math.isclose(sum(near_one.values()), 1.0)
    assert near_one[0] > near_one[2] > near_one[1]


@pytest.mark.parametrize("beta", [-0.1, 1.0, float("inf")])
def test_recursive_round_weights_reject_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="history_mix_beta"):
        recursive_round_weights(current_round=1, history_mix_beta=beta)


def test_missing_history_is_renormalized_inside_old_distribution() -> None:
    plan = resolve_round_mixture_plan(
        pool_name=OFFLINE_POSITIVE,
        current_round=3,
        history_mix_beta=0.5,
        round_pool_sizes={0: 10, 1: 0, 2: 20, 3: 30},
        configured_loss_weight=2.0,
    )

    assert plan.active
    assert plan.configured_round_weights == {
        0: 0.125,
        1: 0.125,
        2: 0.25,
        3: 0.5,
    }
    assert plan.effective_round_weights == pytest.approx(
        {0: 1.0 / 6.0, 2: 1.0 / 3.0, 3: 0.5}
    )
    assert plan.effective_loss_weight == 2.0


def test_positive_and_gt_fail_plans_skip_independently() -> None:
    positive = resolve_round_mixture_plan(
        pool_name=OFFLINE_POSITIVE,
        current_round=1,
        history_mix_beta=0.5,
        round_pool_sizes={0: 20, 1: 0},
        configured_loss_weight=2.0,
    )
    gt_fail = resolve_round_mixture_plan(
        pool_name=OFFLINE_GT_NEGATIVE,
        current_round=1,
        history_mix_beta=0.5,
        round_pool_sizes={0: 0, 1: 4},
        configured_loss_weight=3.0,
    )

    assert not positive.active
    assert positive.skip_reason == "current_round_pool_empty"
    assert positive.effective_loss_weight == 0.0
    assert positive.effective_round_weights == {}
    assert gt_fail.active
    assert gt_fail.skip_reason is None
    assert gt_fail.effective_loss_weight == 3.0
    assert gt_fail.effective_round_weights == {1: 1.0}
    assert positive.to_record()["skip_reason"] == "current_round_pool_empty"


def test_round_zero_named_sampler_matches_original_sampler_exactly() -> None:
    device = _cuda()
    seed = 41
    pools = {
        PRETRAIN_POSITIVE: torch.arange(24, device=device).reshape(8, 3).float(),
        PRETRAIN_UNLABELED: torch.arange(30, device=device).reshape(10, 3).float(),
        OFFLINE_POSITIVE: torch.arange(15, device=device).reshape(5, 3).float(),
        OFFLINE_GT_NEGATIVE: torch.arange(6, device=device).reshape(2, 3).float(),
    }
    legacy = CUDAPoolSampler(pools, device=device, seed=seed)
    plans = {
        name: resolve_round_mixture_plan(
            pool_name=name,
            current_round=0,
            history_mix_beta=0.5,
            round_pool_sizes={0: int(pools[name].shape[0])},
            configured_loss_weight=1.0,
        )
        for name in (OFFLINE_POSITIVE, OFFLINE_GT_NEGATIVE)
    }
    online = BatchOnlineCUDAPoolSampler(
        {
            PRETRAIN_POSITIVE: pools[PRETRAIN_POSITIVE],
            PRETRAIN_UNLABELED: pools[PRETRAIN_UNLABELED],
        },
        round_pools={
            OFFLINE_POSITIVE: {0: pools[OFFLINE_POSITIVE]},
            OFFLINE_GT_NEGATIVE: {0: pools[OFFLINE_GT_NEGATIVE]},
        },
        plans=plans,
        device=device,
        seed=seed,
    )

    for pool_name, batch_size in (
        (PRETRAIN_POSITIVE, 4),
        (PRETRAIN_UNLABELED, 4),
        (OFFLINE_POSITIVE, 5),
        (OFFLINE_GT_NEGATIVE, 6),
    ):
        torch.testing.assert_close(
            online.sample(pool_name, batch_size),
            legacy.sample(pool_name, batch_size),
        )


def test_multi_round_sampling_is_seeded_cuda_and_with_replacement() -> None:
    device = _cuda()
    plan = resolve_round_mixture_plan(
        pool_name=OFFLINE_POSITIVE,
        current_round=2,
        history_mix_beta=0.5,
        round_pool_sizes={0: 1, 1: 1, 2: 1},
        configured_loss_weight=1.0,
    )
    round_pools = {
        0: torch.full((1, 2), 0.0, device=device),
        1: torch.full((1, 2), 1.0, device=device),
        2: torch.full((1, 2), 2.0, device=device),
    }
    first = RecursiveRoundCUDASampler(
        plan, round_pools, device=device, seed=7
    ).sample(4096)
    second = RecursiveRoundCUDASampler(
        plan, round_pools, device=device, seed=7
    ).sample(4096)

    assert first.is_cuda
    torch.testing.assert_close(first, second)
    observed = torch.stack([(first[:, 0] == value).float().mean() for value in range(3)])
    torch.testing.assert_close(
        observed,
        torch.tensor([0.25, 0.25, 0.5], device=device),
        atol=0.03,
        rtol=0.0,
    )
    assert first.shape[0] > sum(plan.round_pool_sizes.values())


def test_nnpu_static_pools_do_not_depend_on_gt_round_sampling() -> None:
    device = _cuda()
    static = {
        PRETRAIN_POSITIVE: torch.randn((7, 3), device=device),
        PRETRAIN_UNLABELED: torch.randn((9, 3), device=device),
    }
    plan = resolve_round_mixture_plan(
        pool_name=OFFLINE_POSITIVE,
        current_round=1,
        history_mix_beta=0.5,
        round_pool_sizes={0: 2, 1: 2},
        configured_loss_weight=1.0,
    )
    kwargs = dict(
        static_pools=static,
        round_pools={
            OFFLINE_POSITIVE: {
                0: torch.randn((2, 3), device=device),
                1: torch.randn((2, 3), device=device),
            }
        },
        plans={OFFLINE_POSITIVE: plan},
        device=device,
        seed=23,
    )
    sampled_gt_first = BatchOnlineCUDAPoolSampler(**kwargs)
    untouched_gt = BatchOnlineCUDAPoolSampler(**kwargs)
    sampled_gt_first.sample(OFFLINE_POSITIVE, 128)

    torch.testing.assert_close(
        sampled_gt_first.sample(PRETRAIN_POSITIVE, 5),
        untouched_gt.sample(PRETRAIN_POSITIVE, 5),
    )
    torch.testing.assert_close(
        sampled_gt_first.sample(PRETRAIN_UNLABELED, 5),
        untouched_gt.sample(PRETRAIN_UNLABELED, 5),
    )


def test_round_feature_cache_roundtrip_and_external_key_miss(tmp_path) -> None:
    device = _cuda()
    path = tmp_path / "round-001-features.pt"
    positive = LatentTrajectory(
        features=torch.randn((3, 4), device=device),
        pool=OFFLINE_POSITIVE,
        source="online",
        identifier="positive-1",
        metadata={"episode": 1},
    )
    gt_fail = LatentTrajectory(
        features=torch.randn((2, 4), device=device),
        pool=OFFLINE_GT_NEGATIVE,
        source="online",
        identifier="gt-fail-1",
        metadata={"episode": 2},
    )
    save_round_feature_cache(
        path,
        round_index=1,
        cache_key="external-contract-sha256",
        offline_positive=[positive],
        offline_gt_negative=[gt_fail],
    )

    assert load_round_feature_cache(
        path,
        expected_round_index=1,
        expected_cache_key="changed-contract",
        device=device,
    ) is None
    loaded = load_round_feature_cache(
        path,
        expected_round_index=1,
        expected_cache_key="external-contract-sha256",
        device=device,
    )
    assert loaded is not None
    assert loaded.round_index == 1
    assert loaded.cache_key == "external-contract-sha256"
    assert loaded.offline_positive[0].identifier == "positive-1"
    assert loaded.offline_gt_negative[0].identifier == "gt-fail-1"
    torch.testing.assert_close(loaded.offline_positive[0].features, positive.features)
    torch.testing.assert_close(loaded.offline_gt_negative[0].features, gt_fail.features)


def test_round_feature_cache_load_rejects_cpu_device(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        load_round_feature_cache(
            tmp_path / "missing.pt",
            expected_round_index=0,
            expected_cache_key="key",
            device="cpu",
        )


def test_feature_cache_key_tracks_the_full_frozen_encoder_contract(tmp_path) -> None:
    episodes = tmp_path / "episodes.pt"
    encoder_checkpoint = tmp_path / "encoder.pth"
    normalizer_checkpoint = tmp_path / "normalizer.pth"
    episodes.write_bytes(b"episodes-v1")
    encoder_checkpoint.write_bytes(b"encoder-v1")
    normalizer_checkpoint.write_bytes(b"normalizer-v1")
    encoder = SimpleNamespace(
        encoder_checkpoint=str(encoder_checkpoint),
        normalizer_checkpoint=str(normalizer_checkpoint),
        feature_source="transformer",
        transformer_layer=-1,
        proprio_indices=[1, 2],
        use_chunk=True,
        inner_encoder=SimpleNamespace(frameskip=8),
    )
    kwargs = {
        "episodes_path": episodes,
        "encoder": encoder,
        "camera_names": ["agentview"],
        "camera_to_view": {"agentview": "agentview"},
        "action_horizon": 8,
        "image_height": 128,
        "image_width": 128,
        "gt_config": {"action_source": "policy_action"},
    }
    baseline = _feature_cache_key(**kwargs)

    for changed in (
        {"action_horizon": 4},
        {"camera_to_view": {"agentview": "robot0_eye_in_hand"}},
        {"image_height": 256},
        {"gt_config": {"action_source": "executed_action"}},
    ):
        assert _feature_cache_key(**(kwargs | changed)) != baseline
    encoder.transformer_layer = -2
    assert _feature_cache_key(**kwargs) != baseline
    encoder.transformer_layer = -1
    normalizer_checkpoint.write_bytes(b"normalizer-v2")
    assert _feature_cache_key(**kwargs) != baseline

