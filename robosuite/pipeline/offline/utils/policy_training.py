"""Controlled branch updates for offline discriminator-policy ablations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch


POLICY_TRAINING_COUPLED = "coupled"
POLICY_TRAINING_INDEPENDENT_SOFT = "independent_soft"
POLICY_TRAINING_FILTERED_BC = "filtered_bc"
POLICY_TRAINING_MODES = frozenset(
    {
        POLICY_TRAINING_COUPLED,
        POLICY_TRAINING_INDEPENDENT_SOFT,
        POLICY_TRAINING_FILTERED_BC,
    }
)


def normalize_policy_training_mode(raw: Any) -> str:
    mode = str(raw).strip().lower().replace("-", "_")
    if mode not in POLICY_TRAINING_MODES:
        expected = ", ".join(sorted(POLICY_TRAINING_MODES))
        raise ValueError(
            f"offline.positive_training.mode must be one of {{{expected}}}, "
            f"got {raw!r}."
        )
    return mode


def branch_seed(base_seed: int, *, step: int, branch: str, phase: str) -> int:
    """Return a stable, disjoint seed for one branch and stochastic phase."""
    branch_offsets = {"pos": 100_000_003, "neg": 200_000_033}
    phase_offsets = {"sample": 10_007, "update": 20_011}
    if branch not in branch_offsets:
        raise ValueError(f"Unknown branch {branch!r}.")
    if phase not in phase_offsets:
        raise ValueError(f"Unknown stochastic phase {phase!r}.")
    modulus = 2**63 - 1
    return int(
        (
            int(base_seed) * 1_000_003
            + int(step) * 10_000_019
            + branch_offsets[branch]
            + phase_offsets[phase]
        )
        % modulus
    )


@contextmanager
def isolated_cuda_rng(device: torch.device | str, seed: int) -> Iterator[None]:
    """Isolate CUDA randomness without changing the caller's global RNG stream."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Independent offline branch updates require CUDA; "
            f"device={resolved}, cuda_available={torch.cuda.is_available()}."
        )
    device_index = (
        torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    )
    with torch.random.fork_rng(devices=[device_index], enabled=True):
        torch.cuda.manual_seed(int(seed))
        yield


def sample_static_cache(
    cache: Any,
    batch_size: int,
    *,
    sample_kwargs: dict[str, Any],
    numpy_rng: np.random.Generator,
    torch_seed: int,
) -> DipoleBatch:
    """Sample rows and augment images with explicitly isolated RNG streams."""
    device = sample_kwargs.get("device")
    with isolated_cuda_rng(device, torch_seed):
        return cache.sample(
            int(batch_size),
            rng=numpy_rng,
            **sample_kwargs,
        )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    values = values.reshape(-1)
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(1.0e-6)


def branch_only_update(
    core: Any,
    batch: DipoleBatch,
    *,
    branch: str,
    weights: torch.Tensor,
    torch_seed: int,
    want_metrics: bool = True,
) -> dict[str, Any]:
    """Update one independent flow branch with a reproducible CUDA RNG stream."""
    if branch not in {"pos", "neg"}:
        raise ValueError(f"branch must be 'pos' or 'neg', got {branch!r}.")

    batch = batch.to(core.device)
    weights = weights.to(device=core.device, dtype=torch.float32).reshape(-1)
    if weights.numel() != batch.batch_size:
        raise ValueError(
            f"weights has {weights.numel()} rows, expected {batch.batch_size}."
        )
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0).any().item()):
        raise ValueError("Branch weights must be finite and non-negative.")
    if float(weights.sum().item()) <= 0.0:
        raise ValueError(f"The {branch} branch batch has zero total weight.")

    model = core.model_pos if branch == "pos" else core.model_neg
    optimizer = core.optimizer_pos if branch == "pos" else core.optimizer_neg
    scaler = core.scaler_pos if branch == "pos" else core.scaler_neg
    batch_size = batch.batch_size

    with isolated_cuda_rng(core.device, torch_seed):
        noise = torch.randn_like(batch.action_sequences)
        timesteps = torch.rand(batch_size, device=core.device)
        x_t = (
            (1.0 - timesteps).view(-1, 1, 1) * noise
            + timesteps.view(-1, 1, 1) * batch.action_sequences
        )
        v_target = batch.action_sequences - noise
        language = [core.language_instruction] * batch_size

        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            enabled=(core.device.type == "cuda"),
            device_type=core.device.type,
        ):
            v_pred = model(
                x_t=x_t.transpose(1, 2),
                t=timesteps,
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            ).transpose(1, 2)
            flow_per_row = torch.mean((v_pred - v_target) ** 2, dim=(1, 2))
            x1 = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
            endpoint_per_row = torch.mean(
                (x1 - batch.action_sequences) ** 2,
                dim=(1, 2),
            )
            if batch.action_sequences.shape[1] > 1:
                smooth_per_row = torch.mean(
                    (x1[:, 1:] - x1[:, :-1]) ** 2,
                    dim=(1, 2),
                )
            else:
                smooth_per_row = torch.zeros_like(flow_per_row)
            flow = _weighted_mean(flow_per_row, weights)
            endpoint = _weighted_mean(endpoint_per_row, weights)
            smooth = _weighted_mean(smooth_per_row, weights)
            loss = (
                flow
                + float(core.config.lambda_endpoint) * endpoint
                + float(core.config.lambda_smooth) * smooth
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(core.config.grad_clip_norm),
        )
        scaler.step(optimizer)
        scaler.update()

    if not want_metrics:
        return {}
    return {
        f"loss_{branch}": float(loss.detach().item()),
        f"flow_loss_{branch}": float(flow.detach().item()),
        f"endpoint_loss_{branch}": float(endpoint.detach().item()),
        f"smooth_loss_{branch}": float(smooth.detach().item()),
        f"grad_norm_{branch}": float(grad_norm.detach().item()),
        f"w_{branch}_mean": float(weights.mean().item()),
        f"w_{branch}_std": float(
            weights.std().item() if batch_size > 1 else 0.0
        ),
        f"frac_w_{branch}_saturated_high": float(
            (weights > 0.99).float().mean().item()
        ),
        f"frac_w_{branch}_saturated_low": float(
            (weights < 0.01).float().mean().item()
        ),
        f"v_{branch}_mse": flow_per_row.detach(),
    }


def trainable_parameter_snapshot(core: Any) -> dict[str, list[torch.Tensor]]:
    """Clone initial trainable parameters on their CUDA device."""
    snapshots: dict[str, list[torch.Tensor]] = {}
    for branch, model in (("pos", core.model_pos), ("neg", core.model_neg)):
        snapshots[branch] = [
            parameter.detach().clone()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    return snapshots


@torch.no_grad()
def parameter_distance_metrics(
    core: Any,
    snapshots: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    """Compute relative trainable-parameter L2 movement entirely on CUDA."""
    metrics: dict[str, float] = {}
    for branch, model in (("pos", core.model_pos), ("neg", core.model_neg)):
        current = [parameter for parameter in model.parameters() if parameter.requires_grad]
        initial = snapshots[branch]
        if len(current) != len(initial):
            raise RuntimeError(f"Trainable parameter structure changed for {branch}.")
        delta_sq = torch.zeros((), device=core.device, dtype=torch.float64)
        initial_sq = torch.zeros((), device=core.device, dtype=torch.float64)
        for parameter, reference in zip(current, initial, strict=True):
            delta_sq += torch.sum((parameter.detach() - reference).double().square())
            initial_sq += torch.sum(reference.double().square())
        delta = torch.sqrt(delta_sq)
        metrics[f"parameter_delta_l2/{branch}"] = float(delta.item())
        metrics[f"parameter_delta_relative/{branch}"] = float(
            (delta / torch.sqrt(initial_sq).clamp_min(1.0e-12)).item()
        )
    return metrics


__all__ = [
    "POLICY_TRAINING_COUPLED",
    "POLICY_TRAINING_FILTERED_BC",
    "POLICY_TRAINING_INDEPENDENT_SOFT",
    "POLICY_TRAINING_MODES",
    "branch_only_update",
    "branch_seed",
    "normalize_policy_training_mode",
    "parameter_distance_metrics",
    "sample_static_cache",
    "trainable_parameter_snapshot",
]
