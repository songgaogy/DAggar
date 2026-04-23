"""KNN distance utilities + F3 soft-weight filter.

All three functions are stateless and torch-native. The chunked cdist loop is
copied in spirit from ``lpb.knn_discriminator.knn_min_sqdist`` but generalized
to k-NN and to a weighted variant.

F3 filter definition (static, no EM):
    w^- = sigmoid(beta * (d^2_knn(phi, D_+) - kappa))
Low-weight fail frames (expert-like prefix) are softly rejected from rho_-;
high-weight frames (genuine off-support) are fully used.

Weighted-KNN trick: folds soft weights as additive log-penalties on the
squared distance, so a low-weight bank point is pushed "further" in effective
distance and is unlikely to be selected as a nearest neighbor.
    d^2_eff(q, b_j) = ||q - b_j||^2 - 2 * sigma^2 * log(w_j)
"""

from __future__ import annotations

from typing import Optional

import torch


_LOG_EPS = 1e-8


@torch.no_grad()
def knn_sqdist(
    query: torch.Tensor,
    bank: torch.Tensor,
    *,
    k: int = 1,
    chunk_size: int = 8192,
    query_chunk_size: int = 4096,
    exclude_self: bool = False,
) -> torch.Tensor:
    """k-NN mean squared L2 distance from each query to the bank.

    Args:
        query: (B, D) — query features.
        bank:  (N, D) — bank features.
        k:     number of nearest neighbors to average.
        chunk_size: bank-side chunk size. Peak intermediate tensor is
            ``qcsz × chunk_size × 4 B``.
        query_chunk_size: query-side chunk size. Needed because cdist allocates
            a full ``Q × C`` matrix; without query chunking 399k × 8192 cdist
            is 13 GB and OOMs on 24 GB cards.
        exclude_self: when ``bank is query``, drop the trivial self match
            (diagonal) before taking the min/top-k.
    Returns:
        (B,) tensor of squared distances averaged over the top-k.
    """
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if bank.shape[0] == 0:
        raise ValueError("bank cannot be empty")

    n_bank = int(bank.shape[0])
    n_query = int(query.shape[0])
    eff_k = max(1, min(int(k), n_bank))
    csz = int(chunk_size)
    qcsz = max(1, int(query_chunk_size))

    out = torch.empty(n_query, device=query.device, dtype=query.dtype)

    for qstart in range(0, n_query, qcsz):
        qend = min(qstart + qcsz, n_query)
        q = query[qstart:qend]
        best = torch.full((q.shape[0], eff_k), float("inf"), device=q.device, dtype=q.dtype)

        for start in range(0, n_bank, csz):
            end = min(start + csz, n_bank)
            chunk = bank[start:end].to(device=q.device, dtype=q.dtype)
            d2 = torch.cdist(q, chunk, p=2.0).pow(2)

            if exclude_self:
                # Diagonal i == j in the global (query, bank) frame; shift into
                # the (qstart..qend, start..end) sub-block.
                rows_in_chunk = torch.arange(start, end, device=q.device)
                query_rows = torch.arange(qstart, qend, device=q.device).unsqueeze(1)
                mask = query_rows == rows_in_chunk.unsqueeze(0)
                d2 = d2.masked_fill(mask, float("inf"))

            merged = torch.cat([best, d2], dim=1)
            best = torch.topk(merged, k=eff_k, dim=1, largest=False).values

        out[qstart:qend] = best.mean(dim=1)

    return out


@torch.no_grad()
def compute_f3_weights(
    fail_feats: torch.Tensor,
    pos_bank: torch.Tensor,
    *,
    beta: Optional[float] = None,
    kappa: Optional[float] = None,
    k: int = 1,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, float, float]:
    """F3 static soft-rejection weights for fail-bank points.

    Args:
        fail_feats: (N_-, D)
        pos_bank:   (N_+, D)
        beta: slope; when ``None``, use 1 / MAD of the positive self-k-NN distances.
        kappa: offset; when ``None``, use the median of those self-k-NN distances.
        k: k-NN order (must match what the detector will use).
    Returns:
        (weights (N_-,) in (0,1), beta_used, kappa_used).
    """
    if beta is None or kappa is None:
        self_d2 = knn_sqdist(pos_bank, pos_bank, k=k, chunk_size=chunk_size, exclude_self=True)
        self_d2 = self_d2[torch.isfinite(self_d2)]
        if self_d2.numel() == 0:
            raise ValueError("pos_bank too small to derive auto beta/kappa")
        med = float(self_d2.median().item())
        mad = float((self_d2 - med).abs().median().item())
        if kappa is None:
            kappa = med
        if beta is None:
            beta = 1.0 / max(mad, _LOG_EPS)

    d2_fail = knn_sqdist(fail_feats, pos_bank, k=k, chunk_size=chunk_size, exclude_self=False)
    weights = torch.sigmoid(float(beta) * (d2_fail - float(kappa)))
    return weights, float(beta), float(kappa)


@torch.no_grad()
def weighted_knn_sqdist(
    query: torch.Tensor,
    bank: torch.Tensor,
    bank_log_weights: torch.Tensor,
    *,
    sigma_sq: float = 0.5,
    k: int = 1,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Weighted k-NN where soft weights fold into distances as log-penalties.

    ``d^2_eff(q, b_j) = ||q - b_j||^2 - 2 * sigma_sq * log(w_j)``.

    ``bank_log_weights`` must be precomputed as ``log(w + eps)`` so that zero
    weights degrade gracefully. Returns the same shape as ``knn_sqdist``: the
    *effective* top-k mean squared distance.

    Note: when all weights are 1 (log_weights == 0) this function is
    numerically identical to ``knn_sqdist``.
    """
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if bank.shape[0] == 0:
        raise ValueError("bank cannot be empty")
    if bank_log_weights.shape != (bank.shape[0],):
        raise ValueError(
            f"bank_log_weights must have shape ({bank.shape[0]},), got {tuple(bank_log_weights.shape)}"
        )

    n_bank = int(bank.shape[0])
    eff_k = max(1, min(int(k), n_bank))
    penalty_all = -2.0 * float(sigma_sq) * bank_log_weights.to(device=query.device, dtype=query.dtype)

    best = torch.full((query.shape[0], eff_k), float("inf"), device=query.device, dtype=query.dtype)

    csz = int(chunk_size)
    for start in range(0, n_bank, csz):
        end = min(start + csz, n_bank)
        chunk = bank[start:end].to(device=query.device, dtype=query.dtype)
        d2 = torch.cdist(query, chunk, p=2.0).pow(2)
        d2 = d2 + penalty_all[start:end].unsqueeze(0)

        merged = torch.cat([best, d2], dim=1)
        best = torch.topk(merged, k=eff_k, dim=1, largest=False).values

    return best.mean(dim=1)
