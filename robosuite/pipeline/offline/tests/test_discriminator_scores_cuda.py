"""CUDA contracts for frozen-discriminator offline DIPOLE weighting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.algorithms.dipole.models.flow import DipoleFlowPolicy
from robosuite.pipeline.offline.utils.branch_weights import (
    RoutedSigmoidBranchWeightPolicy,
)
from robosuite.pipeline.offline.utils.discriminator_scores import (
    OfflineDiscriminatorGProvider,
    precompute_discriminator_scores,
)
from robosuite.pipeline.offline.utils.episode_dataset import (
    ROUTE_DISC_WEIGHTED,
    ROUTE_NEG_ONLY,
    ROUTE_POS_ONLY,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Offline discriminator weighting requires CUDA.",
)


class _StaticCache:
    def __init__(self, raw_scores: torch.Tensor) -> None:
        device = raw_scores.device
        count = int(raw_scores.numel())
        self.images = torch.zeros(count, 1, 3, 2, 2, device=device, dtype=torch.uint8)
        self.proprio_raw = torch.zeros(count, 3, device=device, dtype=torch.float32)
        self.actions_raw = raw_scores.reshape(count, 1, 1)
        self.start_indices = torch.arange(
            100, 100 + count, device=device, dtype=torch.long
        )
        self.pin_memory = False

    def __len__(self) -> int:
        return int(self.start_indices.numel())


class _RawScoreEncoder:
    def encode_chunk(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        assert image_obs_raw.is_cuda
        assert proprio_raw.is_cuda
        assert action_chunk.is_cuda
        return action_chunk[:, 0, :1]


class _IdentityRawScoreDiscriminator:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def failure_score(self, *, chunk_feature: torch.Tensor) -> torch.Tensor:
        assert chunk_feature.is_cuda
        return chunk_feature[:, 0]


def _batch(
    *,
    start_indices: list[int],
    routes: list[str] | None = None,
) -> DipoleBatch:
    device = torch.device("cuda:0")
    count = len(start_indices)
    return DipoleBatch(
        image_obs=torch.zeros(count, 1, 3, 2, 2, device=device),
        image_obs_raw=torch.zeros(count, 1, 3, 2, 2, device=device),
        proprio=torch.zeros(count, 3, device=device),
        proprio_raw=torch.zeros(count, 3, device=device),
        action_sequences=torch.zeros(count, 1, 1, device=device),
        action_sequences_raw=torch.zeros(count, 1, 1, device=device),
        is_intervention=torch.zeros(count, device=device, dtype=torch.bool),
        metadata={
            "start_indices": start_indices,
            **({"route": routes} if routes is not None else {}),
        },
    )


def test_precompute_score_margin_has_calibrated_direction_and_cuda_contract() -> None:
    device = torch.device("cuda:0")
    threshold = 0.75
    delta = 0.4
    raw_scores = torch.tensor(
        [threshold - delta, threshold, threshold + delta],
        device=device,
        dtype=torch.float32,
    )
    cache = _StaticCache(raw_scores)

    cached_raw_scores, g_values, start_to_row = precompute_discriminator_scores(
        static_cache=cache,
        encoder=_RawScoreEncoder(),
        discriminator=_IdentityRawScoreDiscriminator(threshold),
        device=device,
        batch_size=2,
    )

    expected_g = torch.tensor([delta, 0.0, -delta], device=device)
    torch.testing.assert_close(cached_raw_scores, raw_scores)
    torch.testing.assert_close(g_values, expected_g)
    positive_weight = torch.sigmoid(2.0 * g_values)
    torch.testing.assert_close(
        positive_weight,
        torch.sigmoid(2.0 * expected_g),
    )
    assert positive_weight[0].item() > 0.5
    assert positive_weight[1].item() == pytest.approx(0.5)
    assert positive_weight[2].item() < 0.5
    assert cached_raw_scores.shape == g_values.shape == (3,)
    assert cached_raw_scores.dtype == g_values.dtype == torch.float32
    assert cached_raw_scores.device == g_values.device == device
    assert start_to_row == {100: 0, 101: 1, 102: 2}


def test_cached_provider_looks_up_reordered_starts_and_rejects_missing_rows() -> None:
    device = torch.device("cuda:0")
    raw_scores = torch.tensor([-0.95, 1.05, 0.25], device=device)
    g_values = torch.tensor([0.5, -0.8, 0.0], device=device)
    provider = OfflineDiscriminatorGProvider(
        raw_scores=raw_scores,
        g_values=g_values,
        start_to_row={5: 0, 9: 1, 13: 2},
        threshold=0.25,
    )
    batch = _batch(start_indices=[13, 5, 9])

    torch.testing.assert_close(
        provider.compute_g_for_batch(batch),
        torch.tensor([0.0, 0.5, -0.8], device=device),
    )
    torch.testing.assert_close(
        provider.raw_scores_for_batch(batch),
        torch.tensor([0.25, -0.95, 1.05], device=device),
    )
    assert provider.compute_g_for_batch(batch).shape == (3,)
    assert provider.compute_g_for_batch(batch).dtype == torch.float32
    assert provider.compute_g_for_batch(batch).is_cuda
    assert provider.threshold == pytest.approx(0.25)

    missing_metadata = _batch(start_indices=[5])
    missing_metadata.metadata.clear()
    with pytest.raises(KeyError, match="start_indices"):
        provider.compute_g_for_batch(missing_metadata)

    with pytest.raises(KeyError, match="no precomputed discriminator score"):
        provider.compute_g_for_batch(_batch(start_indices=[999]))


def test_routed_branch_weights_mix_discriminator_soft_and_hard_rows() -> None:
    device = torch.device("cuda:0")
    delta = 0.6
    provider = OfflineDiscriminatorGProvider(
        raw_scores=torch.tensor([-0.35, 0.85], device=device),
        g_values=torch.tensor([delta, -delta], device=device),
        start_to_row={10: 0, 40: 1},
        threshold=0.25,
    )
    batch = _batch(
        start_indices=[10, 20, 30, 40],
        routes=[
            ROUTE_DISC_WEIGHTED,
            ROUTE_POS_ONLY,
            ROUTE_NEG_ONLY,
            ROUTE_DISC_WEIGHTED,
        ],
    )
    policy = object.__new__(DipoleFlowPolicy)
    policy.device = device
    policy.config = SimpleNamespace(beta=2.0)
    policy.g_provider = provider
    policy.branch_weight_policy = RoutedSigmoidBranchWeightPolicy()

    w_pos, w_neg, metrics = policy._compute_branch_weights(batch)
    diagnostic_g = policy._provider_g_for_batch(batch)
    expected_pos = torch.sigmoid(
        2.0 * torch.tensor([delta, 0.0, 0.0, -delta], device=device)
    )
    expected_pos[1:3] = torch.tensor([1.0, 0.0], device=device)

    torch.testing.assert_close(w_pos, expected_pos)
    torch.testing.assert_close(w_neg, 1.0 - expected_pos)
    torch.testing.assert_close(w_pos + w_neg, torch.ones_like(w_pos))
    torch.testing.assert_close(
        diagnostic_g,
        torch.tensor([delta, 0.0, 0.0, -delta], device=device),
    )
    assert w_pos.shape == w_neg.shape == (4,)
    assert w_pos.dtype == w_neg.dtype == torch.float32
    assert w_pos.device == w_neg.device == device
    assert metrics["frac_disc_weighted"] == pytest.approx(0.5)
    assert metrics["frac_pos_only"] == pytest.approx(0.25)
    assert metrics["frac_neg_only"] == pytest.approx(0.25)
    assert metrics["threshold"] == pytest.approx(0.25)
    assert metrics["raw_score_mean"] == pytest.approx(0.25)
    assert metrics[f"effective_pos_mass/{ROUTE_POS_ONLY}"] == pytest.approx(
        1.0 / float(expected_pos.sum().item())
    )
    assert metrics[f"effective_neg_mass/{ROUTE_NEG_ONLY}"] == pytest.approx(
        1.0 / float((1.0 - expected_pos).sum().item())
    )
    assert sum(
        metrics[f"effective_pos_mass/{route}"]
        for route in (ROUTE_DISC_WEIGHTED, ROUTE_POS_ONLY, ROUTE_NEG_ONLY)
    ) == pytest.approx(1.0)
