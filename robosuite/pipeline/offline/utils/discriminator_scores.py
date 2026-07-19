"""CUDA-only discriminator score cache for offline DIPOLE."""

from __future__ import annotations

import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)


@torch.no_grad()
def precompute_discriminator_scores(
    *,
    static_cache: Any,
    encoder: Any,
    discriminator: Any,
    device: str | torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Precompute raw discriminator scores and calibrated DIPOLE G values.

    This follows the finetuned discriminator visualization convention: the raw
    score ``s = -head_logit`` is larger for failure-like windows and the stored
    threshold is in the same score space. Offline DIPOLE uses::

        G = threshold - s

    All output tensors stay on the requested CUDA device.
    """
    score_device = torch.device(device)
    if score_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Offline discriminator score precomputation requires CUDA; "
            f"requested device={score_device}, cuda_available={torch.cuda.is_available()}."
        )
    encode_batch_size = max(1, int(batch_size))
    if len(static_cache) <= 0:
        raise ValueError("Cannot precompute discriminator scores for an empty cache.")

    raw_score_parts: list[torch.Tensor] = []
    g_parts: list[torch.Tensor] = []
    threshold = torch.as_tensor(
        float(discriminator.threshold), device=score_device, dtype=torch.float32
    )
    for offset in range(0, len(static_cache), encode_batch_size):
        stop = min(offset + encode_batch_size, len(static_cache))
        images = static_cache.images[offset:stop].to(
            score_device, dtype=torch.float32, non_blocking=bool(static_cache.pin_memory)
        ).div_(255.0)
        proprio = static_cache.proprio_raw[offset:stop].to(
            score_device, dtype=torch.float32, non_blocking=bool(static_cache.pin_memory)
        )
        actions = static_cache.actions_raw[offset:stop].to(
            score_device, dtype=torch.float32, non_blocking=bool(static_cache.pin_memory)
        )
        chunk_feature = encoder.encode_chunk(
            image_obs_raw=images,
            proprio_raw=proprio,
            action_chunk=actions,
        )
        raw_score = discriminator.failure_score(
            chunk_feature=chunk_feature
        ).reshape(-1)
        g_value = threshold - raw_score
        raw_score_parts.append(raw_score.detach())
        g_parts.append(g_value.detach())

    raw_scores = torch.cat(raw_score_parts, dim=0).reshape(-1)
    g_values = torch.cat(g_parts, dim=0).reshape(-1)
    if not raw_scores.is_cuda or not g_values.is_cuda:
        raise RuntimeError("Discriminator score cache unexpectedly left CUDA.")
    start_indices = [int(value) for value in static_cache.start_indices.tolist()]
    start_to_row = {start: row for row, start in enumerate(start_indices)}
    logger.info(
        "[offline] precomputed discriminator scores for %d windows "
        "(raw_score_mean=%+.4f G_mean=%+.4f threshold=%+.4f)",
        len(start_indices),
        float(raw_scores.mean().item()),
        float(g_values.mean().item()),
        float(discriminator.threshold),
    )
    return raw_scores, g_values, start_to_row


class OfflineDiscriminatorGProvider:
    """Serve cached ``raw_score`` and ``G = threshold - raw_score``."""

    def __init__(
        self,
        *,
        raw_scores: torch.Tensor,
        g_values: torch.Tensor,
        start_to_row: dict[int, int],
        threshold: float,
    ) -> None:
        raw_scores = raw_scores.detach().reshape(-1)
        g_values = g_values.detach().reshape(-1)
        if not raw_scores.is_cuda or not g_values.is_cuda:
            raise RuntimeError(
                "OfflineDiscriminatorGProvider requires CUDA score caches; "
                "CPU fallback is disabled."
            )
        if raw_scores.device != g_values.device:
            raise ValueError(
                "Raw-score and G caches must share one CUDA device: "
                f"{raw_scores.device} vs {g_values.device}."
            )
        if raw_scores.numel() != g_values.numel():
            raise ValueError("Raw-score and G caches must have equal length.")
        if len(start_to_row) != int(g_values.numel()):
            raise ValueError("start_to_row size must match the score cache length.")
        self._raw_scores = raw_scores
        self._g_values = g_values
        self._start_to_row = dict(start_to_row)
        self.threshold = float(threshold)

    def bind_policy_cameras(self, policy_camera_names: list[str]) -> None:
        del policy_camera_names

    def _rows_for_batch(self, batch: Any) -> torch.Tensor:
        start_indices = batch.metadata.get("start_indices")
        if start_indices is None:
            raise KeyError(
                "OfflineDiscriminatorGProvider requires "
                "batch.metadata['start_indices']."
            )
        try:
            rows = [self._start_to_row[int(start)] for start in start_indices]
        except KeyError as exc:
            raise KeyError(
                f"Start index {exc} has no precomputed discriminator score; "
                "the offline replay buffer must remain static."
            ) from exc
        device = batch.action_sequences_raw.device
        if device != self._g_values.device:
            raise RuntimeError(
                "Discriminator cache and policy batch must share one CUDA device: "
                f"cache={self._g_values.device}, batch={device}."
            )
        return torch.tensor(rows, dtype=torch.long, device=device)

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        rows = self._rows_for_batch(batch)
        return self._g_values.index_select(0, rows).reshape(-1)

    @torch.no_grad()
    def raw_scores_for_batch(self, batch: Any) -> torch.Tensor:
        rows = self._rows_for_batch(batch)
        return self._raw_scores.index_select(0, rows).reshape(-1)


__all__ = [
    "OfflineDiscriminatorGProvider",
    "precompute_discriminator_scores",
]
