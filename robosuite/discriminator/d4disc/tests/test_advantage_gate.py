"""Synthetic sanity test for the image-based advantage gate."""

from __future__ import annotations

import numpy as np
import torch

from robosuite.discriminator.d4disc.models.adaln import ConditionEmbedder
from robosuite.discriminator.d4disc.training.filter import compute_advantage_gate


def _vec_to_image(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(values.astype(np.float32).reshape(2, 2, 2))


class _FlattenEncoder:
    def __init__(self) -> None:
        self.training = False
        self.latent_dim = 8

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.float().reshape(image.shape[0], -1)


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n_fail: int = 64, dim: int = 8, shift: float = 2.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        n_half = n_fail // 2
        zeros = np.zeros((n_half, dim), dtype=np.float32)
        offs = np.full((n_fail - n_half, dim), shift, dtype=np.float32)
        self.targets = np.concatenate([zeros, offs], axis=0)
        self.n = int(n_fail)
        self.is_fail_raw_flags = np.ones(self.n, dtype=bool)
        self.gamma_buffer = torch.full((self.n,), 0.5, dtype=torch.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        target = _vec_to_image(self.targets[idx])
        return {
            "current_image": torch.zeros_like(target),
            "current_proprio": torch.zeros(2, dtype=torch.float32),
            "action_sequence": torch.zeros(1, 2, dtype=torch.float32),
            "target_image": target,
            "target_proprio": torch.zeros(2, dtype=torch.float32),
            "is_expert": torch.tensor(0, dtype=torch.int64),
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "is_fail_raw": torch.tensor(True, dtype=torch.bool),
            "gamma": torch.tensor(float(self.gamma_buffer[idx].item()), dtype=torch.float32),
        }

    def fail_indices(self) -> torch.Tensor:
        return torch.arange(self.n, dtype=torch.long)

    @property
    def is_fail_raw(self) -> torch.Tensor:
        return torch.from_numpy(self.is_fail_raw_flags).bool()

    def update_gamma(
        self,
        indices: torch.Tensor,
        new_gamma: torch.Tensor,
        ema_alpha: float = 0.5,
        clamp_eps: float = 1e-3,
    ) -> None:
        idx = indices.detach().cpu().long().view(-1)
        ng = new_gamma.detach().cpu().float().view(-1)
        old = self.gamma_buffer[idx]
        mixed = float(ema_alpha) * old + (1.0 - float(ema_alpha)) * ng
        self.gamma_buffer[idx] = mixed.clamp(float(clamp_eps), 1.0 - float(clamp_eps))


class _ToyPredictor:
    def __init__(self, dim: int, shift: float) -> None:
        self.dim = int(dim)
        self.shift = float(shift)
        self.training = False

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def __call__(self, obs, prop, act, cond_idx):
        pred = torch.zeros(obs.shape[0], self.dim, dtype=torch.float32, device=obs.device)
        pred[cond_idx == ConditionEmbedder.COND_MINUS] = self.shift
        return {"pred_latent": pred, "pred_proprio": torch.zeros(obs.shape[0], 2, device=obs.device)}


def test_advantage_gate_routes_by_advantage_sign() -> None:
    ds = _ToyDataset(n_fail=64, dim=8, shift=2.0, seed=0)
    pred = _ToyPredictor(dim=8, shift=2.0)
    enc = _FlattenEncoder()

    diag = compute_advantage_gate(
        pred,
        enc,
        ds,
        alpha_k=50.0,
        kappa_k=0.0,
        sigma_sq=0.5,
        device="cpu",
        batch_size=16,
        num_workers=0,
        ema_alpha=0.0,
        advantage_mode="residual",
    )

    gamma = ds.gamma_buffer.numpy()
    on_support = gamma[: ds.n // 2]
    off_support = gamma[ds.n // 2 :]
    assert on_support.mean() < 0.05, f"on-support mean gamma too high: {on_support.mean()}"
    assert off_support.mean() > 0.95, f"off-support mean gamma too low: {off_support.mean()}"
    assert "mean_gamma" in diag and "separation_gap" in diag
