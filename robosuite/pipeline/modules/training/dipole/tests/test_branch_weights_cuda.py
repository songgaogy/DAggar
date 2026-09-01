"""CUDA tests for routed DIPOLE branch weights."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch, select_dipole_batch
from robosuite.pipeline.algorithms.dipole.models.flow import DipoleFlowPolicy
from robosuite.pipeline.modules.training.dipole.branch_weights import (
    RoutedSigmoidBranchWeightPolicy,
)


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DIPOLE branch-weight tests.")
    return torch.device("cuda:0")


def _batch(device: torch.device) -> DipoleBatch:
    values = torch.tensor([0.25, -0.5, 0.75, -0.25], device=device)
    zeros = torch.zeros((4, 1), device=device)
    return DipoleBatch(
        image_obs=zeros,
        image_obs_raw=zeros,
        proprio=values[:, None],
        proprio_raw=zeros,
        action_sequences=zeros[:, None, :],
        action_sequences_raw=zeros[:, None, :],
        is_intervention=torch.zeros(4, dtype=torch.bool, device=device),
        metadata={
            "route": ["advantage", "advantage", "pos_only", "neg_only"],
            "is_success_trajectory": [True, False, True, True],
        },
    )


class _ProprioProvider:
    @staticmethod
    def compute_g_for_batch(batch: DipoleBatch) -> torch.Tensor:
        return batch.proprio[:, 0]


def _sigmoid_fn(beta: float, k: float):
    owner = SimpleNamespace(config=SimpleNamespace(beta=beta, k=k))

    def compute(raw: torch.Tensor, **kwargs):
        return DipoleFlowPolicy._g_weights_from_raw(owner, raw, **kwargs)

    return compute


def test_success_trajectory_adds_eta_to_advantage_logit() -> None:
    device = _cuda()
    batch = _batch(device)
    policy = RoutedSigmoidBranchWeightPolicy(eta=0.5)

    w_pos, w_neg, metrics = policy(
        batch,
        g_provider=_ProprioProvider(),
        sigmoid_fn=_sigmoid_fn(beta=2.0, k=0.1),
        device=device,
    )

    expected = torch.stack(
        (
            torch.sigmoid(torch.tensor(2.0 * (0.25 + 0.1) + 0.5, device=device)),
            torch.sigmoid(torch.tensor(2.0 * (-0.5 + 0.1), device=device)),
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
    )
    torch.testing.assert_close(w_pos, expected)
    torch.testing.assert_close(w_neg, 1.0 - expected)
    assert metrics["frac_success_trajectory"] == 0.75
    assert metrics["success_logit_bonus_mean"] == 0.125


def test_eta_zero_matches_original_formula_and_selection_preserves_marker() -> None:
    device = _cuda()
    batch = _batch(device)
    selected = select_dipole_batch(
        batch, torch.tensor([0, 1], dtype=torch.long, device=device)
    )
    assert selected.metadata["is_success_trajectory"] == [True, False]

    w_pos, w_neg, _ = RoutedSigmoidBranchWeightPolicy(eta=0.0)(
        selected,
        g_provider=_ProprioProvider(),
        sigmoid_fn=_sigmoid_fn(beta=2.0, k=0.1),
        device=device,
        want_metrics=False,
    )
    expected = torch.sigmoid(2.0 * (selected.proprio[:, 0] + 0.1))
    torch.testing.assert_close(w_pos, expected)
    torch.testing.assert_close(w_neg, 1.0 - expected)


def test_missing_success_marker_defaults_to_zero_bonus() -> None:
    device = _cuda()
    batch = _batch(device)
    batch.metadata.pop("is_success_trajectory")

    w_pos, _, _ = RoutedSigmoidBranchWeightPolicy(eta=0.5)(
        batch,
        g_provider=_ProprioProvider(),
        sigmoid_fn=_sigmoid_fn(beta=2.0, k=0.1),
        device=device,
        want_metrics=False,
    )
    expected_advantage = torch.sigmoid(2.0 * (batch.proprio[:2, 0] + 0.1))
    torch.testing.assert_close(w_pos[:2], expected_advantage)
