"""CUDA-only recursive round sampling for batch-online GT risks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .objectives import CUDAPoolSampler


TensorPool = torch.Tensor | Sequence[torch.Tensor]


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, got bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


@dataclass(frozen=True)
class RoundMixturePlan:
    """Resolved per-loss sampling distribution and logging metadata."""

    pool_name: str
    current_round: int
    history_mix_beta: float
    configured_loss_weight: float
    effective_loss_weight: float
    round_pool_sizes: dict[int, int]
    configured_round_weights: dict[int, float]
    effective_round_weights: dict[int, float]
    skip_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.skip_reason is None and self.effective_loss_weight > 0.0

    @property
    def direct_current_round(self) -> bool:
        return self.active and self.effective_round_weights == {
            self.current_round: 1.0
        }

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable state/TensorBoard companion record."""
        return {
            "pool_name": self.pool_name,
            "current_round": int(self.current_round),
            "history_mix_beta": float(self.history_mix_beta),
            "configured_loss_weight": float(self.configured_loss_weight),
            "effective_loss_weight": float(self.effective_loss_weight),
            "round_pool_sizes": {
                str(index): int(size)
                for index, size in self.round_pool_sizes.items()
            },
            "configured_round_weights": {
                str(index): float(weight)
                for index, weight in self.configured_round_weights.items()
            },
            "effective_round_weights": {
                str(index): float(weight)
                for index, weight in self.effective_round_weights.items()
            },
            "skip_reason": self.skip_reason,
        }


def recursive_round_weights(
    *, current_round: int, history_mix_beta: float
) -> dict[int, float]:
    """Resolve ``beta * D_old + (1-beta) * D_new`` recursively."""
    if isinstance(current_round, bool) or not isinstance(current_round, int):
        raise TypeError("current_round must be an integer.")
    if current_round < 0:
        raise ValueError("current_round must be non-negative.")
    beta = _finite_float(history_mix_beta, name="history_mix_beta")
    if not 0.0 <= beta < 1.0:
        raise ValueError("history_mix_beta must be in [0, 1).")
    if current_round == 0:
        return {0: 1.0}

    weights = {0: beta**current_round}
    for round_index in range(1, current_round + 1):
        weights[round_index] = (1.0 - beta) * beta ** (
            current_round - round_index
        )
    return weights


def resolve_round_mixture_plan(
    *,
    pool_name: str,
    current_round: int,
    history_mix_beta: float,
    round_pool_sizes: Mapping[int, int],
    configured_loss_weight: float,
) -> RoundMixturePlan:
    """Resolve one GT loss independently, including empty-new-pool skipping."""
    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("pool_name must be a non-empty string.")
    configured_weight = _finite_float(
        configured_loss_weight, name="configured_loss_weight"
    )
    if configured_weight < 0.0:
        raise ValueError("configured_loss_weight must be non-negative.")
    configured_round_weights = recursive_round_weights(
        current_round=current_round,
        history_mix_beta=history_mix_beta,
    )
    sizes: dict[int, int] = {}
    for raw_round, raw_size in round_pool_sizes.items():
        if isinstance(raw_round, bool) or not isinstance(raw_round, int):
            raise TypeError("round_pool_sizes keys must be integer round indices.")
        if raw_round < 0 or raw_round > current_round:
            raise ValueError(
                f"round_pool_sizes contains out-of-range round {raw_round}."
            )
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise TypeError("round pool sizes must be integers.")
        if raw_size < 0:
            raise ValueError("round pool sizes must be non-negative.")
        sizes[raw_round] = raw_size
    sizes = {index: sizes.get(index, 0) for index in range(current_round + 1)}

    skip_reason: str | None = None
    if configured_weight == 0.0:
        skip_reason = "configured_loss_weight_zero"
    elif sizes[current_round] == 0:
        skip_reason = "current_round_pool_empty"

    effective_round_weights: dict[int, float] = {}
    if skip_reason is None:
        beta = float(history_mix_beta)
        if current_round == 0 or beta == 0.0:
            effective_round_weights = {current_round: 1.0}
        else:
            available_history = [
                index for index in range(current_round) if sizes[index] > 0
            ]
            if not available_history:
                effective_round_weights = {current_round: 1.0}
            else:
                historical_mass = sum(
                    configured_round_weights[index] for index in available_history
                )
                if historical_mass == 0.0:
                    effective_round_weights = {current_round: 1.0}
                else:
                    effective_round_weights = {
                        index: beta
                        * configured_round_weights[index]
                        / historical_mass
                        for index in available_history
                    }
                    effective_round_weights[current_round] = 1.0 - beta

    return RoundMixturePlan(
        pool_name=pool_name,
        current_round=current_round,
        history_mix_beta=float(history_mix_beta),
        configured_loss_weight=configured_weight,
        effective_loss_weight=(0.0 if skip_reason else configured_weight),
        round_pool_sizes=sizes,
        configured_round_weights=configured_round_weights,
        effective_round_weights=effective_round_weights,
        skip_reason=skip_reason,
    )


def _pool_seed(seed: int, pool_name: str) -> int:
    digest = hashlib.sha256(pool_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return (int(seed) + offset) % (2**63 - 1)


def _validated_cuda_pool(
    value: TensorPool,
    *,
    pool_name: str,
    round_index: int,
    device: torch.device,
) -> torch.Tensor:
    tensors = [value] if torch.is_tensor(value) else list(value)
    if not tensors:
        raise ValueError(f"Pool {pool_name!r} round {round_index} is empty.")
    latent_dim: int | None = None
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Pool {pool_name!r} round {round_index} contains a non-tensor."
            )
        if tensor.device.type != "cuda" or tensor.device != device:
            raise ValueError(
                f"Pool {pool_name!r} round {round_index} must be on {device}, "
                f"got {tensor.device}."
            )
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            raise ValueError(
                f"Pool {pool_name!r} round {round_index} tensors must have "
                f"non-empty shape (N, D), got {tuple(tensor.shape)}."
            )
        if latent_dim is None:
            latent_dim = int(tensor.shape[1])
        elif int(tensor.shape[1]) != latent_dim:
            raise ValueError(
                f"Pool {pool_name!r} round {round_index} latent dimensions differ."
            )
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


class RecursiveRoundCUDASampler:
    """Sample an active round-mixture plan on CUDA with replacement."""

    def __init__(
        self,
        plan: RoundMixturePlan,
        round_pools: Mapping[int, TensorPool],
        *,
        device: str | torch.device,
        seed: int,
    ) -> None:
        if not plan.active:
            raise ValueError(
                f"Cannot sample inactive pool {plan.pool_name!r}: {plan.skip_reason}."
            )
        self.plan = plan
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"RecursiveRoundCUDASampler requires CUDA, got {self.device}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RecursiveRoundCUDASampler requires CUDA, but CUDA is unavailable."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        self._pools: dict[int, torch.Tensor] = {}
        latent_dim: int | None = None
        for round_index in plan.effective_round_weights:
            if round_index not in round_pools:
                raise KeyError(
                    f"Missing {plan.pool_name!r} tensor pool for round {round_index}."
                )
            tensor = _validated_cuda_pool(
                round_pools[round_index],
                pool_name=plan.pool_name,
                round_index=round_index,
                device=self.device,
            )
            expected_size = plan.round_pool_sizes[round_index]
            if int(tensor.shape[0]) != expected_size:
                raise ValueError(
                    f"Pool {plan.pool_name!r} round {round_index} has "
                    f"{tensor.shape[0]} rows, expected {expected_size}."
                )
            if latent_dim is None:
                latent_dim = int(tensor.shape[1])
            elif int(tensor.shape[1]) != latent_dim:
                raise ValueError(
                    f"Pool {plan.pool_name!r} latent dimensions differ by round."
                )
            self._pools[round_index] = tensor

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @property
    def pool_size(self) -> int:
        return sum(int(pool.shape[0]) for pool in self._pools.values())

    def sample(self, batch_size: int) -> torch.Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.plan.direct_current_round:
            pool = self._pools[self.plan.current_round]
            indices = torch.randint(
                int(pool.shape[0]),
                (batch_size,),
                device=self.device,
                generator=self.generator,
            )
            return pool.index_select(0, indices)

        round_indices = tuple(self.plan.effective_round_weights)
        probabilities = torch.tensor(
            [self.plan.effective_round_weights[index] for index in round_indices],
            dtype=torch.float64,
            device=self.device,
        )
        choices = torch.multinomial(
            probabilities,
            batch_size,
            replacement=True,
            generator=self.generator,
        )
        first_pool = self._pools[round_indices[0]]
        result = torch.empty(
            (batch_size, int(first_pool.shape[1])),
            dtype=first_pool.dtype,
            device=self.device,
        )
        for choice_index, round_index in enumerate(round_indices):
            output_rows = torch.nonzero(
                choices == choice_index, as_tuple=False
            ).reshape(-1)
            count = int(output_rows.numel())
            if count == 0:
                continue
            pool = self._pools[round_index]
            source_rows = torch.randint(
                int(pool.shape[0]),
                (count,),
                device=self.device,
                generator=self.generator,
            )
            result.index_copy_(0, output_rows, pool.index_select(0, source_rows))
        return result


class BatchOnlineCUDAPoolSampler:
    """Named sampler compatible with ``CompositeDiscriminatorObjective``."""

    def __init__(
        self,
        static_pools: Mapping[str, TensorPool],
        *,
        round_pools: Mapping[str, Mapping[int, TensorPool]],
        plans: Mapping[str, RoundMixturePlan],
        device: str | torch.device,
        seed: int,
    ) -> None:
        self.device = torch.device(device)
        direct_pools = dict(static_pools)
        for pool_name, plan in plans.items():
            if plan.pool_name != pool_name:
                raise ValueError(
                    f"Plan key {pool_name!r} does not match {plan.pool_name!r}."
                )
            if plan.direct_current_round:
                direct_pools[pool_name] = round_pools[pool_name][plan.current_round]
        self._direct = CUDAPoolSampler(direct_pools, device=self.device, seed=seed)
        self._round_samplers = {
            pool_name: RecursiveRoundCUDASampler(
                plan,
                round_pools[pool_name],
                device=self.device,
                seed=_pool_seed(seed, pool_name),
            )
            for pool_name, plan in plans.items()
            if plan.active and not plan.direct_current_round
        }
        self.plans = dict(plans)

    @property
    def pool_sizes(self) -> dict[str, int]:
        sizes = self._direct.pool_sizes
        sizes.update(
            {
                name: sampler.pool_size
                for name, sampler in self._round_samplers.items()
            }
        )
        return sizes

    def sample(self, pool_name: str, batch_size: int) -> torch.Tensor:
        if pool_name in self._round_samplers:
            return self._round_samplers[pool_name].sample(batch_size)
        return self._direct.sample(pool_name, batch_size)


__all__ = [
    "BatchOnlineCUDAPoolSampler",
    "RecursiveRoundCUDASampler",
    "RoundMixturePlan",
    "recursive_round_weights",
    "resolve_round_mixture_plan",
]
