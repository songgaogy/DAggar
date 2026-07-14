"""CUDA-only tests for the VAST value-stitching adaptation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.vast.common import VASTConfig, VASTStepBatch
from robosuite.pipeline.algorithms.vast.vast import VASTLearner
from robosuite.pipeline.algorithms.vast.replay import VASTPreencodedReplayCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VAST tensor tests require CUDA and never fall back to CPU.",
)
DEVICE = "cuda:0"


def _cfg(**overrides) -> VASTConfig:
    values = {
        "vast_v_mode": "single_vast",
        "vast_max_k": 3,
        "vast_comp_coef": 0.5,
        "vast_sampling_seed": 7,
        "action_horizon": 2,
        "discount": 0.9,
        "expectile_tau": 0.9,
        "v_ensemble_size": 2,
        "hidden_dims": (32, 32),
        "state_proj_dim": 8,
        "proprio_proj_dim": 4,
        "device": DEVICE,
        "disc_reward_coef": 0.0,
    }
    values.update(overrides)
    return VASTConfig(**values)


def _learner(cfg: VASTConfig | None = None) -> VASTLearner:
    return VASTLearner(
        cfg=cfg or _cfg(),
        state_feature_dim=12,
        chunk_feature_dim=16,
        action_dim=4,
        n_tokens=4,
        proprio_dim=4,
    )


def _batch(batch_size: int = 8) -> VASTStepBatch:
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    state = torch.randn(batch_size, 12, device=DEVICE, generator=generator)
    future = torch.randn(batch_size, 12, device=DEVICE, generator=generator)
    middle = torch.randn(batch_size, 12, device=DEVICE, generator=generator)
    k = torch.tensor([[2], [3]] * (batch_size // 2), device=DEVICE, dtype=torch.float32)
    j = torch.ones_like(k)
    return VASTStepBatch(
        chunk_feature=torch.randn(batch_size, 16, device=DEVICE, generator=generator),
        v_state_feature=state,
        next_v_state_feature=future,
        action_chunk=torch.randn(batch_size, 2, 4, device=DEVICE, generator=generator),
        rewards=torch.randn(batch_size, 1, device=DEVICE, generator=generator),
        dones=torch.zeros(batch_size, 1, device=DEVICE),
        is_online=torch.zeros(batch_size, 1, device=DEVICE),
        is_intervention=torch.zeros(batch_size, 1, device=DEVICE),
        future_v_state_feature=future,
        intermediate_v_state_feature=middle,
        k=k,
        j=j,
        k_step_returns=torch.randn(batch_size, 1, device=DEVICE, generator=generator),
        mc_mask=torch.ones(batch_size, 1, device=DEVICE),
        future_dones=torch.zeros(batch_size, 1, device=DEVICE),
    )


def test_vast_joint_update_moves_g_and_v() -> None:
    learner = _learner()
    batch = _batch()
    assert learner.ensemble_size == 1
    assert learner.g is not None
    g_before = [parameter.detach().clone() for parameter in learner.g.parameters()]
    v_before = [parameter.detach().clone() for parameter in learner.v.parameters()]
    metrics = learner.update(batch)
    # G and V heads are zero-output initialized. The first coherent joint
    # snapshot therefore gives V a zero stitched target while moving G; the
    # second update exposes the learned non-zero G target to V.
    metrics = learner.update(batch)
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(g_before, learner.g.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(v_before, learner.v.parameters()))
    assert {"g_mc_loss", "g_comp_loss", "stitched_target_mean", "k_mean"} <= set(metrics)


def test_stitched_advantage_uses_macro_discount_and_done_mask() -> None:
    learner = _learner()
    batch = _batch(8)
    assert batch.future_v_state_feature is not None and batch.k is not None
    assert batch.future_dones is not None
    advantage = learner.compute_stitched_advantage(
        batch.v_state_feature,
        batch.future_v_state_feature,
        batch.k,
        batch.future_dones,
    )
    g = learner.g_value(batch.v_state_feature, batch.future_v_state_feature, batch.k)
    gamma = torch.full_like(g, learner.cfg.discount)
    discount = gamma.pow(batch.k * learner.cfg.action_horizon)
    expected = (
        g
        + discount
        * (1.0 - batch.future_dones)
        * learner.target_v_lcb(batch.future_v_state_feature)
        - learner.v_lcb(batch.v_state_feature)
    ).reshape(-1)
    torch.testing.assert_close(advantage, expected)


def test_ensemble_lcb_vast_mode_preserves_configured_ensemble() -> None:
    learner = _learner(_cfg(vast_v_mode="ensemble_lcb", v_ensemble_size=3))
    assert learner.ensemble_size == 3
    assert learner.v(torch.randn(5, 12, device=DEVICE)).shape == (5, 3)


def test_vast_checkpoint_roundtrip_and_schema6_compatibility() -> None:
    source = _learner()
    source.update(_batch())
    state = source.state_dict()
    assert state["learner_schema_version"] == 7
    assert "g" in state and "g_optim" in state
    target = _learner()
    target.load_state_dict(state)
    assert target.g is not None and source.g is not None
    for left, right in zip(source.g.parameters(), target.g.parameters()):
        assert torch.equal(left, right)
    legacy = dict(state)
    legacy["learner_schema_version"] = 6
    legacy["method"] = "vast_value_stitching"
    legacy.pop("algorithm")
    target.load_state_dict(legacy)


def _cache(seed: int = 11) -> VASTPreencodedReplayCache:
    cfg = _cfg(vast_sampling_seed=seed, vast_max_k=3)
    n = 5
    state = torch.arange(n * 12, device=DEVICE, dtype=torch.float32).reshape(n, 12)
    return VASTPreencodedReplayCache(
        chunk_feature=torch.zeros(n, 16, device=DEVICE),
        v_state_feature=state,
        next_v_state_feature=state + 0.5,
        action_chunk=torch.zeros(n, 2, 4, device=DEVICE),
        rewards=torch.arange(1, n + 1, device=DEVICE, dtype=torch.float32).view(n, 1),
        dones=torch.tensor([[0], [0], [1], [0], [0]], device=DEVICE, dtype=torch.float32),
        is_online=torch.zeros(n, 1, device=DEVICE),
        is_intervention=torch.zeros(n, 1, device=DEVICE),
        source_size=n,
        cfg=cfg,
        valid_starts=[0, 2, 4, 10, 12],
        episode_ids=[0, 0, 0, 1, 1],
    )


def test_macro_sampling_is_seeded_and_never_crosses_episode_or_terminal() -> None:
    first = _cache(seed=19)
    second = _cache(seed=19)
    selected = np.asarray([0, 0, 3, 3])
    batch_a = first._sample_vast_step_batch(selected, device=DEVICE)
    batch_b = second._sample_vast_step_batch(selected, device=DEVICE)
    torch.testing.assert_close(batch_a.k, batch_b.k)
    torch.testing.assert_close(batch_a.j, batch_b.j)
    assert batch_a.k is not None and batch_a.j is not None
    assert torch.all(batch_a.k >= 1)
    assert torch.all(batch_a.k <= torch.tensor([[3], [3], [2], [2]], device=DEVICE))
    active = batch_a.k >= 2
    assert torch.all(batch_a.j[active] >= 1)
    assert torch.all(batch_a.j[active] < batch_a.k[active])


def _legacy_vast_cache_sample(cache, sampled_rows, rng):
    """Reference implementation from before the vectorized cache sampler."""
    H = int(cache.cfg.action_horizon)
    paths = []
    ks = []
    js = []
    for raw_row in sampled_rows:
        row = int(raw_row)
        start = cache.valid_starts[row]
        episode = cache.episode_ids[row]
        candidates = []
        for macro_idx in range(int(cache.cfg.vast_max_k)):
            candidate = cache._start_to_row.get(start + macro_idx * H)
            if candidate is None or cache.episode_ids[candidate] != episode:
                break
            candidates.append(candidate)
            if bool(cache.dones[candidate].max().item()):
                break
        k = int(rng.integers(1, len(candidates) + 1))
        j = int(rng.integers(1, k)) if k >= 2 else 1
        paths.append(candidates[:k])
        ks.append(k)
        js.append(j)

    gamma_h = float(cache.cfg.discount) ** H
    returns = torch.stack(
        [
            sum(cache.rewards[row] * (gamma_h**m) for m, row in enumerate(path))
            for path in paths
        ],
        dim=0,
    )
    future_dones = torch.stack(
        [cache.dones[path].amax(dim=0) for path in paths], dim=0
    )
    return {
        "current": torch.tensor([path[0] for path in paths], device=DEVICE),
        "last": torch.tensor([path[-1] for path in paths], device=DEVICE),
        "middle": torch.tensor(
            [path[j] if k >= 2 else path[0] for path, k, j in zip(paths, ks, js)],
            device=DEVICE,
        ),
        "returns": returns,
        "future_dones": future_dones,
        "ks": ks,
        "js": js,
    }


def test_vectorized_cache_sampler_is_exactly_legacy_equivalent() -> None:
    seed = 23
    cache = _cache(seed=seed)
    reference_rng = np.random.default_rng(seed)
    selections = (
        np.asarray([0, 0, 1, 2, 3, 4]),
        np.asarray([4, 3, 2, 1, 0, 0]),
        np.asarray([0, 3, 0, 3, 1, 4]),
    )
    for selected in selections:
        expected = _legacy_vast_cache_sample(cache, selected, reference_rng)
        actual = cache._sample_vast_step_batch(selected, device=DEVICE)
        current = expected["current"]
        torch.testing.assert_close(actual.chunk_feature, cache.chunk_feature[current], rtol=0, atol=0)
        torch.testing.assert_close(actual.v_state_feature, cache.v_state_feature[current], rtol=0, atol=0)
        torch.testing.assert_close(
            actual.future_v_state_feature,
            cache.next_v_state_feature[expected["last"]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.intermediate_v_state_feature,
            cache.v_state_feature[expected["middle"]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual.k_step_returns, expected["returns"], rtol=0, atol=0)
        torch.testing.assert_close(actual.future_dones, expected["future_dones"], rtol=0, atol=0)
        assert actual.metadata["sampled_k"] == expected["ks"]
        assert actual.metadata["sampled_j"] == expected["js"]
    assert cache._vast_rng.bit_generator.state == reference_rng.bit_generator.state
