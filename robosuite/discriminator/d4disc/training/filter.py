"""Advantage-gate update for the image-based D4 pipeline.

This module implements the Phase-B "gate" (outer-loop) update that maintains
the per-fail-sample routing probabilities gamma_i stored in the Dataset.

Given a conditional predictor f(·|c) and a fail transition i, we compute:
    r_+(i), r_-(i)   (branch-specific scores)
    A(i) = (r_-(i) - r_+(i)) / (2 * sigma_sq)   (advantage, scaled)
    gamma_i <- sigmoid(alpha_k * ( -A(i) - kappa_k ))

Intuition:
    - If the (-) branch looks worse than the (+) branch on sample i, then A>0
      and gamma decreases (route more mass to +).
    - If the (-) branch looks better, A<0 and gamma increases (route more mass to -).

The gate is always run with EMA model weights in the trainer for stability.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from robosuite.discriminator.d3disc.filter import knn_sqdist

from ..data.dataset import LatentFlowDynamicsDatasetD4
from ..models.adaln import ConditionEmbedder
from ..models.dynamics import ConditionalDynamicsPredictor
from ..models.encoder import Encoder


def _collate(batch: list[dict]) -> dict:
    out: Dict[str, torch.Tensor] = {}
    if not batch:
        return out
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
    return out


def _resolve_device(device: str) -> torch.device:
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _gamma_quantile_diag(gamma: torch.Tensor) -> Dict[str, float]:
    if gamma.numel() == 0:
        return {
            "mean_gamma": 0.0,
            "gamma_q05": 0.0,
            "gamma_q25": 0.0,
            "gamma_q50": 0.0,
            "gamma_q75": 0.0,
            "gamma_q95": 0.0,
        }
    g = gamma.detach().float().cpu()
    return {
        "mean_gamma": float(g.mean().item()),
        "gamma_q05": float(g.quantile(0.05).item()),
        "gamma_q25": float(g.quantile(0.25).item()),
        "gamma_q50": float(g.quantile(0.50).item()),
        "gamma_q75": float(g.quantile(0.75).item()),
        "gamma_q95": float(g.quantile(0.95).item()),
    }


@torch.no_grad()
def build_expert_target_bank(
    dataset: LatentFlowDynamicsDatasetD4,
    encoder: Encoder,
    *,
    max_bank_size: Optional[int] = 100_000,
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Encode target_image of clean positive transitions into a KNN bank.

    Returns (N, latent_dim) tensor on ``device``. Used by the KNN-mode
    advantage gate (compute_advantage_gate with advantage_mode='knn') as the
    expert z_{t+h} distribution that pred_latent is scored against.
    """
    dev = _resolve_device(device)
    encoder_was_training = encoder.training
    encoder.eval()
    encoder.to(dev)

    clean_idx = (~dataset.is_fail_raw).nonzero(as_tuple=False).squeeze(-1)
    n_clean = int(clean_idx.numel())
    if n_clean == 0:
        if encoder_was_training:
            encoder.train()
        raise RuntimeError("expert target bank requires clean positive transitions")

    if max_bank_size is not None and n_clean > int(max_bank_size):
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n_clean, generator=gen)[: int(max_bank_size)]
        clean_idx = clean_idx[perm]

    subset = Subset(dataset, clean_idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate,
        pin_memory=(dev.type == "cuda"),
    )
    chunks: list[torch.Tensor] = []
    for batch in loader:
        target_images = batch["target_image"].to(dev, non_blocking=True)
        chunks.append(encoder(target_images).detach().float())
    bank = torch.cat(chunks, dim=0).to(dev)

    if encoder_was_training:
        encoder.train()
    return bank


@torch.no_grad()
def compute_advantage_gate(
    predictor: ConditionalDynamicsPredictor,
    encoder: Encoder,
    dataset: LatentFlowDynamicsDatasetD4,
    *,
    alpha_k: float,
    kappa_k: float = 0.0,
    sigma_sq: float = 0.5,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 0,
    ema_alpha: float = 0.5,
    advantage_mode: str = "knn",
    expert_z_bank: Optional[torch.Tensor] = None,
    knn_k: int = 1,
    knn_chunk_size: int = 8192,
) -> Dict[str, float]:
    """Compute per-sample advantage and update gamma for fail samples.

    The update is applied only to `dataset.fail_indices()`; clean positives
    retain their default gamma values but are never routed via gamma.

    advantage_mode:
        "residual": r_c = ||f(c) - z_target||² (raw next-latent MSE;
            d4disc_0423.md §5 verbatim). Score is dominated by motion
            magnitude when the predictor is modestly trained.
        "knn" (default): r_c = knn_sqdist(f(c), expert_z_bank) — min
            squared L2 distance from the conditional prediction to a bank
            of expert target latents. Under a Gaussian-KNN interpretation
            this equals -log p_c(z_{t+h}|o,a) up to a constant, so the
            advantage is exactly log p_+ - log p_- over the expert
            distribution. Matches the inference-time score_mode='knn'
            paradigm (LPB-parity density estimation).

    `expert_z_bank` must be provided (and pre-encoded) when
    ``advantage_mode='knn'``; typically built once per Phase-B loop by
    ``build_expert_target_bank``.
    """
    if str(advantage_mode) not in {"residual", "knn"}:
        raise ValueError(f"advantage_mode must be 'residual' or 'knn', got {advantage_mode!r}")
    if advantage_mode == "knn" and (expert_z_bank is None or expert_z_bank.numel() == 0):
        raise ValueError("advantage_mode='knn' requires non-empty expert_z_bank")

    predictor_was_training = predictor.training
    encoder_was_training = encoder.training
    predictor.eval()
    encoder.eval()

    dev = _resolve_device(device)
    predictor.to(dev)
    encoder.to(dev)
    bank_dev: Optional[torch.Tensor] = None
    if advantage_mode == "knn":
        bank_dev = expert_z_bank.to(dev, dtype=torch.float32, non_blocking=True)

    fail_idx = dataset.fail_indices()
    n_fail = int(fail_idx.numel())
    if n_fail == 0:
        if predictor_was_training:
            predictor.train()
        if encoder_was_training:
            encoder.train()
        return {
            "mean_gamma": 0.0,
            "mean_r_plus": 0.0,
            "mean_r_minus": 0.0,
            "separation_gap": 0.0,
            "alpha_used": float(alpha_k),
            "kappa_used": float(kappa_k),
            "num_fail_raw": 0,
            "advantage_mode": advantage_mode,
        }

    subset = Subset(dataset, fail_idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
        collate_fn=_collate,
    )

    r_plus_all: list[torch.Tensor] = []
    r_minus_all: list[torch.Tensor] = []
    gamma_all: list[torch.Tensor] = []
    sample_idx_all: list[torch.Tensor] = []

    two_sigma_sq = 2.0 * float(sigma_sq)
    for batch in loader:
        images = batch["current_image"].to(dev, non_blocking=True)
        target_images = batch["target_image"].to(dev, non_blocking=True)
        prop = batch["current_proprio"].to(dev, non_blocking=True)
        act = batch["action_sequence"].to(dev, non_blocking=True)

        z_t = encoder(images)
        z_target = encoder(target_images).detach()

        batch_size_now = int(z_t.shape[0])
        c_plus = torch.full((batch_size_now,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=dev)
        c_minus = torch.full((batch_size_now,), ConditionEmbedder.COND_MINUS, dtype=torch.long, device=dev)

        out_plus = predictor(z_t, prop, act, cond_idx=c_plus)
        out_minus = predictor(z_t, prop, act, cond_idx=c_minus)
        pred_plus = out_plus["pred_latent"]
        pred_minus = out_minus["pred_latent"]

        if advantage_mode == "residual":
            r_plus = ((pred_plus - z_target) ** 2).sum(dim=-1)
            r_minus = ((pred_minus - z_target) ** 2).sum(dim=-1)
        else:
            # KNN: score each prediction against the expert target-latent bank.
            r_plus = knn_sqdist(pred_plus, bank_dev, k=int(knn_k), chunk_size=int(knn_chunk_size))
            r_minus = knn_sqdist(pred_minus, bank_dev, k=int(knn_k), chunk_size=int(knn_chunk_size))
        # Advantage A := (r_- - r_+) / (2 sigma^2).
        # Gamma update uses sigmoid(alpha_k * ( -A - kappa_k )).
        advantage = (r_minus - r_plus) / two_sigma_sq
        gamma_new = torch.sigmoid(float(alpha_k) * (-advantage - float(kappa_k)))

        r_plus_all.append(r_plus.detach().cpu())
        r_minus_all.append(r_minus.detach().cpu())
        gamma_all.append(gamma_new.detach().cpu())
        sample_idx_all.append(batch["sample_idx"].detach().cpu())

    r_plus_cat = torch.cat(r_plus_all, dim=0) if r_plus_all else torch.zeros(0)
    r_minus_cat = torch.cat(r_minus_all, dim=0) if r_minus_all else torch.zeros(0)
    gamma_cat = torch.cat(gamma_all, dim=0) if gamma_all else torch.zeros(0)
    idx_cat = torch.cat(sample_idx_all, dim=0) if sample_idx_all else torch.zeros(0, dtype=torch.long)

    dataset.update_gamma(idx_cat, gamma_cat, ema_alpha=float(ema_alpha))

    if predictor_was_training:
        predictor.train()
    if encoder_was_training:
        encoder.train()

    diag = _gamma_quantile_diag(dataset.gamma_buffer[fail_idx])
    diag.update(
        {
            "mean_r_plus": float(r_plus_cat.mean().item()) if r_plus_cat.numel() > 0 else 0.0,
            "mean_r_minus": float(r_minus_cat.mean().item()) if r_minus_cat.numel() > 0 else 0.0,
            "separation_gap": float((r_minus_cat - r_plus_cat).mean().item())
            if r_plus_cat.numel() > 0
            else 0.0,
            "alpha_used": float(alpha_k),
            "kappa_used": float(kappa_k),
            "num_fail_raw": int(n_fail),
            "advantage_mode": advantage_mode,
            "bank_size": int(bank_dev.shape[0]) if bank_dev is not None else 0,
        }
    )
    return diag


@torch.no_grad()
def warm_start_gamma_from_knn(
    dataset: LatentFlowDynamicsDatasetD4,
    encoder: Encoder,
    *,
    k: int = 1,
    mode: str = "rank",
    beta: Optional[float] = None,
    kappa: Optional[float] = None,
    chunk_size: int = 8192,
    max_clean_bank: Optional[int] = 100_000,
    max_fail_queries: Optional[int] = None,
    device: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 0,
    seed: int = 0,
) -> Dict[str, float]:
    dev = _resolve_device(device)
    encoder_was_training = encoder.training
    encoder.eval()
    encoder.to(dev)

    clean_idx_all = (~dataset.is_fail_raw).nonzero(as_tuple=False).squeeze(-1)
    fail_idx_all = dataset.fail_indices()
    n_clean = int(clean_idx_all.numel())
    n_fail = int(fail_idx_all.numel())
    if n_clean == 0 or n_fail == 0:
        if encoder_was_training:
            encoder.train()
        return {"warm_start": 0, "n_clean": n_clean, "n_fail": n_fail}

    rng = torch.Generator().manual_seed(int(seed))
    if max_clean_bank is not None and n_clean > int(max_clean_bank):
        perm = torch.randperm(n_clean, generator=rng)[: int(max_clean_bank)]
        clean_idx = clean_idx_all[perm]
    else:
        clean_idx = clean_idx_all

    if max_fail_queries is not None and n_fail > int(max_fail_queries):
        perm = torch.randperm(n_fail, generator=rng)[: int(max_fail_queries)]
        fail_idx = fail_idx_all[perm]
    else:
        fail_idx = fail_idx_all

    def _gather_z(indices: torch.Tensor) -> torch.Tensor:
        subset = Subset(dataset, indices.tolist())
        loader = DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=_collate,
            pin_memory=(dev.type == "cuda"),
        )
        chunks: list[torch.Tensor] = []
        for batch in loader:
            images = batch["current_image"].to(dev, non_blocking=True)
            chunks.append(encoder(images).detach().cpu().float())
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0)

    pos_bank = _gather_z(clean_idx).to(dev)
    fail_feats = _gather_z(fail_idx).to(dev)

    d2_fail = knn_sqdist(fail_feats, pos_bank, k=int(k), chunk_size=int(chunk_size))
    d2_cpu = d2_fail.detach().float().cpu()

    mode_lower = str(mode).lower()
    if mode_lower == "rank":
        ranks = d2_cpu.argsort().argsort().float()
        weights_cpu = (ranks + 0.5) / float(max(ranks.numel(), 1))
        beta_used = float("nan")
        kappa_used = float("nan")
    elif mode_lower == "sigmoid":
        med = float(d2_cpu.median().item())
        mad = float((d2_cpu - med).abs().median().item())
        kappa_v = float(kappa) if kappa is not None else med
        beta_v = float(beta) if beta is not None else 1.0 / max(mad, 1e-8)
        weights_cpu = torch.sigmoid(beta_v * (d2_cpu - kappa_v))
        beta_used = float(beta_v)
        kappa_used = float(kappa_v)
    else:
        raise ValueError(f"unknown warm-start mode={mode!r}; expected 'rank' or 'sigmoid'")

    q = torch.tensor([0.05, 0.5, 0.95])
    d2_q = torch.quantile(d2_cpu, q).tolist()
    raw_q = torch.quantile(weights_cpu, q).tolist()
    saturated_hi = float((weights_cpu >= 1.0 - 1e-3).float().mean().item())
    saturated_lo = float((weights_cpu <= 1e-3).float().mean().item())

    dataset.update_gamma(fail_idx, weights_cpu, ema_alpha=0.0)

    if encoder_was_training:
        encoder.train()

    diag = _gamma_quantile_diag(dataset.gamma_buffer[fail_idx_all])
    diag.update(
        {
            "warm_start": 1,
            "mode": mode_lower,
            "beta_used": beta_used,
            "kappa_used": kappa_used,
            "n_clean": int(pos_bank.shape[0]),
            "n_fail": int(fail_feats.shape[0]),
            "d2_mean": float(d2_cpu.mean().item()),
            "d2_q05": float(d2_q[0]),
            "d2_q50": float(d2_q[1]),
            "d2_q95": float(d2_q[2]),
            "raw_gamma_q05": float(raw_q[0]),
            "raw_gamma_q50": float(raw_q[1]),
            "raw_gamma_q95": float(raw_q[2]),
            "raw_frac_saturated_hi": saturated_hi,
            "raw_frac_saturated_lo": saturated_lo,
        }
    )
    return diag
