"""Advantage-gate update: the D4 replacement for the static F3 filter.

Equation (flat, design §3.3):

    gamma_j = sigmoid(alpha_k * ( -A_theta(phi_j, a_j, phi_j+h) - kappa_k ))

with A_theta = Q^+ - Q^- = (r^- - r^+) / (2 sigma^2), where r^c is the
per-sample squared latent residual under condition c. Called once per
outer bootstrap epoch over the full fail_raw population. EMA-smoothed by
the dataset's ``update_gamma`` (no hard replacement, no per-batch change).

Stability contract (from plan §7):
- Runs under torch.no_grad and in eval mode on the EMA-weight predictor.
- Operates on the dataset's fail_raw indices only — clean positives do
  not participate in the gate.
- Diagnostics reported: mean_gamma, separation_gap, mean_r_plus,
  mean_r_minus; consumed by the collapse monitor.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from robosuite.discriminator.d3disc.filter import knn_sqdist

from .adaln import ConditionEmbedder
from .dataset import LatentFlowDynamicsDatasetD4
from .model import ConditionalDynamicsPredictor


def _collate(batch: list[dict]) -> dict:
    out: Dict[str, torch.Tensor] = {}
    if not batch:
        return out
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
    return out


@torch.no_grad()
def compute_advantage_gate(
    predictor: ConditionalDynamicsPredictor,
    dataset: LatentFlowDynamicsDatasetD4,
    *,
    alpha_k: float,
    kappa_k: float = 0.0,
    sigma_sq: float = 0.5,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 0,
    ema_alpha: float = 0.5,
) -> Dict[str, float]:
    was_training = predictor.training
    predictor.eval()
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    predictor.to(dev)

    fail_idx = dataset.fail_indices()
    n_fail = int(fail_idx.numel())
    if n_fail == 0:
        if was_training:
            predictor.train()
        return {
            "mean_gamma": 0.0,
            "mean_r_plus": 0.0,
            "mean_r_minus": 0.0,
            "separation_gap": 0.0,
            "alpha_used": float(alpha_k),
            "kappa_used": float(kappa_k),
            "num_fail_raw": 0,
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
        obs = batch["current_latent"].to(dev, non_blocking=True)
        prop = batch["current_proprio"].to(dev, non_blocking=True)
        act = batch["action_sequence"].to(dev, non_blocking=True)
        tgt = batch["target_latent"].to(dev, non_blocking=True)
        B = obs.shape[0]

        c_plus = torch.full((B,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=dev)
        c_minus = torch.full((B,), ConditionEmbedder.COND_MINUS, dtype=torch.long, device=dev)

        out_plus = predictor(obs, prop, act, cond_idx=c_plus)
        out_minus = predictor(obs, prop, act, cond_idx=c_minus)

        r_plus = ((out_plus["pred_latent"] - tgt) ** 2).sum(dim=-1)
        r_minus = ((out_minus["pred_latent"] - tgt) ** 2).sum(dim=-1)
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

    if was_training:
        predictor.train()

    diag = _gamma_quantile_diag(dataset.gamma_buffer[fail_idx])
    diag.update({
        "mean_r_plus": float(r_plus_cat.mean().item()) if r_plus_cat.numel() > 0 else 0.0,
        "mean_r_minus": float(r_minus_cat.mean().item()) if r_minus_cat.numel() > 0 else 0.0,
        "separation_gap": float((r_minus_cat - r_plus_cat).mean().item()) if r_plus_cat.numel() > 0 else 0.0,
        "alpha_used": float(alpha_k),
        "kappa_used": float(kappa_k),
        "num_fail_raw": int(n_fail),
    })
    return diag


def _gamma_quantile_diag(gamma: torch.Tensor) -> Dict[str, float]:
    if gamma.numel() == 0:
        return {
            "mean_gamma": 0.0,
            "gamma_q05": 0.0, "gamma_q25": 0.0, "gamma_q50": 0.0,
            "gamma_q75": 0.0, "gamma_q95": 0.0,
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
def warm_start_gamma_from_knn(
    dataset: LatentFlowDynamicsDatasetD4,
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
    """Break the γ=0.5 symmetric fixed point by initializing γ from KNN
    distance to the clean success bank.

    Rationale: at end of Phase A, f(+) and f(-) are numerically close because
    AdaLN-Zero kept c=- at identity and phase-A never trained c=-. Starting
    Phase B with γ≡0.5 routes fail samples 50/50, so both branches see the
    same data and converge to the same function — a stable fixed point. A
    one-shot data-driven γ warm-start (high γ on frames far from clean bank,
    low γ near it) produces the asymmetry needed for the bootstrap to
    sharpen instead of collapse.

    Writes γ directly to ``dataset.gamma_buffer`` (no EMA blending).

    Args:
        k: kNN order for the fail→clean-bank query.
        mode: "rank" (default) assigns ``γ_i = (rank_i + 0.5) / N`` so γ is
            uniform on (0,1) regardless of the d² scale — robust to the
            common failure where fail→clean distances dwarf the clean
            self-kNN scale used by D3's sigmoid and every γ saturates to 1.
            "sigmoid" auto-calibrates β = 1/MAD(d²_fail) and κ = median(d²_fail)
            from the fail→clean distance distribution itself (NOT clean-self
            as in D3), which prevents saturation when the two populations
            have different scales.
        beta / kappa: override auto calibration (sigmoid mode only).
        max_clean_bank: subsample the clean bank if larger (memory/time cap).
        max_fail_queries: None => all fail_raw samples.
        device: "cuda" / "cpu".
    """
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    clean_idx_all = (~dataset.is_fail_raw).nonzero(as_tuple=False).squeeze(-1)
    fail_idx_all = dataset.fail_indices()
    n_clean = int(clean_idx_all.numel())
    n_fail = int(fail_idx_all.numel())
    if n_clean == 0 or n_fail == 0:
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
            pin_memory=False,
        )
        chunks: list[torch.Tensor] = []
        for batch in loader:
            chunks.append(batch["current_latent"].detach().float())
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0)

    pos_bank = _gather_z(clean_idx).to(dev)
    fail_feats = _gather_z(fail_idx).to(dev)

    # Raw fail→clean kNN distances.
    d2_fail = knn_sqdist(
        fail_feats, pos_bank, k=int(k), chunk_size=int(chunk_size)
    )
    d2_cpu = d2_fail.detach().float().cpu()

    mode_lower = str(mode).lower()
    if mode_lower == "rank":
        # γ_i = (rank_i + 0.5) / N — uniform on (0,1); no scale calibration.
        ranks = d2_cpu.argsort().argsort().float()
        weights_cpu = (ranks + 0.5) / float(max(ranks.numel(), 1))
        beta_used = float("nan")
        kappa_used = float("nan")
    elif mode_lower == "sigmoid":
        # Calibrate β, κ from the fail→clean distribution itself, not the
        # clean self-kNN (which caused saturation in run d4dyn_20260422_054915).
        med = float(d2_cpu.median().item())
        mad = float((d2_cpu - med).abs().median().item())
        kappa_v = float(kappa) if kappa is not None else med
        beta_v = float(beta) if beta is not None else 1.0 / max(mad, 1e-8)
        weights_cpu = torch.sigmoid(beta_v * (d2_cpu - kappa_v))
        beta_used = float(beta_v)
        kappa_used = float(kappa_v)
    else:
        raise ValueError(
            f"unknown warm-start mode={mode!r}; expected 'rank' or 'sigmoid'"
        )

    # Diagnostics on raw d² and raw weights (pre-clamp, pre-buffer-write).
    q = torch.tensor([0.05, 0.5, 0.95])
    d2_q = torch.quantile(d2_cpu, q).tolist()
    raw_q = torch.quantile(weights_cpu, q).tolist()
    saturated_hi = float((weights_cpu >= 1.0 - 1e-3).float().mean().item())
    saturated_lo = float((weights_cpu <= 1e-3).float().mean().item())

    # One-shot replace (ema_alpha=0 means 100% new, 0% old).
    dataset.update_gamma(fail_idx, weights_cpu, ema_alpha=0.0)

    diag = _gamma_quantile_diag(dataset.gamma_buffer[fail_idx_all])
    diag.update({
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
    })
    return diag
