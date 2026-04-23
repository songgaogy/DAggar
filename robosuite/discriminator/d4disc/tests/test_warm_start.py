"""Sanity test for F3-based gamma warm-start.

Build a toy dataset whose fail_raw samples split into:
  - on-support block: current_latent near origin (close to clean bank) -> low gamma
  - off-support block: current_latent far from origin -> high gamma

Verifies the warm-start breaks the gamma=0.5 symmetric fixed point as
required for Phase B bootstrap.
"""

from __future__ import annotations

import numpy as np
import torch

from robosuite.discriminator.d4disc.filter import warm_start_gamma_from_knn


class _WarmStartToyDataset(torch.utils.data.Dataset):
    def __init__(self, n_clean: int = 128, n_on: int = 48, n_off: int = 48, dim: int = 8,
                 shift: float = 5.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        clean = rng.normal(scale=0.1, size=(n_clean, dim)).astype(np.float32)
        on_support = rng.normal(scale=0.1, size=(n_on, dim)).astype(np.float32)
        off_support = (rng.normal(scale=0.1, size=(n_off, dim)) + shift).astype(np.float32)
        self._z = np.concatenate([clean, on_support, off_support], axis=0)
        self._is_fail_raw = torch.tensor(
            [False] * n_clean + [True] * (n_on + n_off), dtype=torch.bool
        )
        self.gamma_buffer = torch.full((self._is_fail_raw.numel(),), 0.5, dtype=torch.float32)
        self.n_clean = n_clean
        self.n_on = n_on
        self.n_off = n_off

    def __len__(self) -> int:
        return int(self._is_fail_raw.numel())

    @property
    def is_fail_raw(self) -> torch.Tensor:
        return self._is_fail_raw

    def fail_indices(self) -> torch.Tensor:
        return self._is_fail_raw.nonzero(as_tuple=False).squeeze(-1)

    def update_gamma(self, indices: torch.Tensor, new_gamma: torch.Tensor,
                     ema_alpha: float = 0.5, clamp_eps: float = 1e-3) -> None:
        idx = indices.detach().cpu().long().view(-1)
        ng = new_gamma.detach().cpu().float().view(-1)
        old = self.gamma_buffer[idx]
        mixed = float(ema_alpha) * old + (1.0 - float(ema_alpha)) * ng
        self.gamma_buffer[idx] = mixed.clamp(float(clamp_eps), 1.0 - float(clamp_eps))

    def __getitem__(self, i: int) -> dict:
        z = torch.from_numpy(self._z[i].copy())
        return {
            "current_latent": z,
            "current_proprio": torch.zeros(2, dtype=torch.float32),
            "action_sequence": torch.zeros(1, 2, dtype=torch.float32),
            "target_latent": torch.zeros_like(z),
            "target_proprio": torch.zeros(2, dtype=torch.float32),
            "is_expert": torch.tensor(0, dtype=torch.int64),
            "sample_idx": torch.tensor(i, dtype=torch.long),
            "is_fail_raw": self._is_fail_raw[i].clone(),
            "gamma": self.gamma_buffer[i].clone(),
        }


def _run_warm_start(mode: str):
    ds = _WarmStartToyDataset(n_clean=128, n_on=48, n_off=48, dim=8, shift=5.0, seed=0)
    diag = warm_start_gamma_from_knn(
        ds,
        k=1,
        mode=mode,
        max_clean_bank=None,
        device="cpu",
        batch_size=32,
        num_workers=0,
    )
    assert diag["warm_start"] == 1
    assert diag["n_clean"] == 128
    assert diag["n_fail"] == 96
    fail_idx = ds.fail_indices()
    gamma = ds.gamma_buffer[fail_idx]
    on_mean = float(gamma[: ds.n_on].mean())
    off_mean = float(gamma[ds.n_on :].mean())
    return diag, on_mean, off_mean


def test_warm_start_rank_splits_on_and_off_support() -> None:
    diag, on_mean, off_mean = _run_warm_start(mode="rank")
    # Rank-normalized γ: on-support frames fill the lower-rank half
    # (→ γ near 0.25), off-support fills the upper half (→ γ near 0.75).
    assert diag["mode"] == "rank"
    assert off_mean > on_mean, f"rank warm-start ordering broken: on={on_mean} off={off_mean}"
    assert (off_mean - on_mean) > 0.3, (
        f"rank warm-start gap too small ({off_mean - on_mean}) — would not break symmetric fixed point"
    )
    # Rank mode is scale-invariant: no saturation on either tail beyond clamp.
    assert diag["raw_frac_saturated_hi"] < 0.02
    assert diag["raw_frac_saturated_lo"] < 0.02


def test_warm_start_sigmoid_fail_calibrated_splits() -> None:
    diag, on_mean, off_mean = _run_warm_start(mode="sigmoid")
    # Sigmoid mode auto-calibrated on fail→clean d² (NOT the D3 clean-self
    # calibration): on samples land below 0.5, off samples above 0.5.
    assert diag["mode"] == "sigmoid"
    assert off_mean > on_mean, f"sigmoid warm-start ordering broken: on={on_mean} off={off_mean}"
    assert (off_mean - on_mean) > 0.3, (
        f"sigmoid warm-start gap too small ({off_mean - on_mean})"
    )
