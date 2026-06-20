"""Frozen nnPU failure-score provider for DIPOLE."""

from __future__ import annotations

from typing import Any

import torch


class NNPUGProvider:
    """Return calibrated nnPU failure scores through the DIPOLE G contract.

    The policy applies ``g_sign=negate_raw`` after this provider, so larger
    failure scores reduce the positive-branch weight.
    """

    def __init__(self, *, encoder: Any, discriminator: Any) -> None:
        self.encoder = encoder
        self.discriminator = discriminator
        self.device = torch.device(encoder.device)
        self.view_names = list(encoder.view_names)
        self.threshold = float(discriminator.threshold)

    def bind_policy_cameras(self, policy_camera_names: list[str]) -> None:
        self.encoder.bind_policy_cameras(policy_camera_names)

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        chunk_feature = self.encoder.encode_chunk(
            image_obs_raw=batch.image_obs_raw,
            proprio_raw=batch.proprio_raw,
            action_chunk=batch.action_sequences_raw,
        )
        score = self.discriminator.failure_score(chunk_feature=chunk_feature)
        return score.to(batch.action_sequences_raw.device).reshape(-1)


__all__ = ["NNPUGProvider"]
