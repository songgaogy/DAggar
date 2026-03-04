from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def latent_ood_score(
    z: torch.Tensor,
    expert_latents: torch.Tensor,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    """
    Compute latent OOD score:
      delta(z) = || z - z_NN ||_2^2
    where z_NN is nearest neighbor in expert latent bank.

    Args:
        z: Query latents, shape (B, D) or (..., D).
        expert_latents: Expert bank, shape (N, D).
    Returns:
        dict:
            score: (B,) squared L2 to nearest expert latent
            nn_index: (B,) nearest-neighbor indices in expert_latents
    """
    if expert_latents.ndim != 2:
        raise ValueError(f"expert_latents must be shape (N,D), got {tuple(expert_latents.shape)}")
    if expert_latents.shape[0] == 0:
        raise ValueError("expert_latents cannot be empty")

    q = z.reshape(-1, z.shape[-1])
    if q.shape[-1] != expert_latents.shape[-1]:
        raise ValueError(
            "latent dim mismatch: "
            f"query={q.shape[-1]} expert={expert_latents.shape[-1]}"
        )

    device = q.device
    expert_latents = expert_latents.to(device=device, dtype=q.dtype)

    best_dist = None
    best_idx = None
    for start in range(0, expert_latents.shape[0], int(chunk_size)):
        chunk = expert_latents[start : start + int(chunk_size)]
        # cdist returns L2 distance; square it for delta(z).
        d2 = torch.cdist(q, chunk, p=2.0).pow(2)
        cur_dist, cur_idx = torch.min(d2, dim=1)
        cur_idx = cur_idx + start
        if best_dist is None:
            best_dist, best_idx = cur_dist, cur_idx
        else:
            mask = cur_dist < best_dist
            best_dist = torch.where(mask, cur_dist, best_dist)
            best_idx = torch.where(mask, cur_idx, best_idx)

    assert best_dist is not None and best_idx is not None
    return {
        "score": best_dist,
        "nn_index": best_idx,
    }
