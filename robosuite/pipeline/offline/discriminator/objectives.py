"""Composable CUDA-only objectives for offline discriminator finetuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from robosuite.discriminator.dyn_disc.detectors.pu_bce import pu_risk


PRETRAIN_POSITIVE = "pretrain_positive"
PRETRAIN_UNLABELED = "pretrain_unlabeled"
OFFLINE_POSITIVE = "offline_positive"
OFFLINE_GT_NEGATIVE = "offline_gt_negative"

_TERM_POOLS = {
    "nnpu": (PRETRAIN_POSITIVE, PRETRAIN_UNLABELED),
    "supervised_bce": (OFFLINE_POSITIVE, OFFLINE_GT_NEGATIVE),
}
_TERM_TYPE_ALIASES = {
    "nnpu": "nnpu",
    "nnpu_replay": "nnpu",
    "supervised_bce": "supervised_bce",
    "supervised_gt_bce": "supervised_bce",
}


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
class LossTermConfig:
    """Validated configuration for one independently sampled loss term."""

    name: str
    type: str
    enabled: bool = True
    weight: float = 1.0
    batch_size: int = 512
    positive_fraction: float = 0.5
    class_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Loss term name must be a non-empty string.")
        if not isinstance(self.type, str) or self.type not in _TERM_TYPE_ALIASES:
            raise ValueError(
                f"Unknown loss type {self.type!r}; expected one of "
                f"{sorted(_TERM_TYPE_ALIASES)}."
            )
        object.__setattr__(self, "type", _TERM_TYPE_ALIASES[self.type])
        if not isinstance(self.enabled, bool):
            raise TypeError(f"{self.name}.enabled must be bool.")
        weight = _finite_float(self.weight, name=f"{self.name}.weight")
        if weight < 0.0:
            raise ValueError(f"{self.name}.weight must be non-negative.")
        object.__setattr__(self, "weight", weight)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError(f"{self.name}.batch_size must be an integer.")
        if self.batch_size < 2:
            raise ValueError(f"{self.name}.batch_size must be at least 2.")
        fraction = _finite_float(
            self.positive_fraction,
            name=f"{self.name}.positive_fraction",
        )
        positive_size = self.batch_size * fraction
        rounded_size = round(positive_size)
        if not math.isclose(positive_size, rounded_size, abs_tol=1.0e-9):
            raise ValueError(
                f"{self.name}.batch_size * positive_fraction must be an integer, "
                f"got {positive_size}."
            )
        if rounded_size <= 0 or rounded_size >= self.batch_size:
            raise ValueError(
                f"{self.name}.positive_fraction must produce non-empty positive "
                "and negative/unlabeled batches."
            )
        object.__setattr__(self, "positive_fraction", fraction)

        if self.type == "nnpu":
            if self.class_weights is not None:
                raise ValueError(
                    f"{self.name}.class_weights is only valid for supervised_bce."
                )
            return
        weights = self.class_weights or {"positive": 0.5, "negative": 0.5}
        if not isinstance(weights, Mapping):
            raise TypeError(f"{self.name}.class_weights must be a mapping.")
        if set(weights) != {"positive", "negative"}:
            raise ValueError(
                f"{self.name}.class_weights must contain exactly positive and negative."
            )
        positive_weight = _finite_float(
            weights["positive"], name=f"{self.name}.class_weights.positive"
        )
        negative_weight = _finite_float(
            weights["negative"], name=f"{self.name}.class_weights.negative"
        )
        if positive_weight < 0.0 or negative_weight < 0.0:
            raise ValueError(f"{self.name}.class_weights must be non-negative.")
        if not math.isclose(positive_weight + negative_weight, 1.0, abs_tol=1.0e-9):
            raise ValueError(f"{self.name}.class_weights must sum to 1.")
        object.__setattr__(
            self,
            "class_weights",
            {"positive": positive_weight, "negative": negative_weight},
        )

    @property
    def positive_batch_size(self) -> int:
        return int(round(self.batch_size * self.positive_fraction))

    @property
    def negative_batch_size(self) -> int:
        return int(self.batch_size - self.positive_batch_size)

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.weight > 0.0)


@dataclass(frozen=True)
class NNPUParameters:
    """Parent-checkpoint nnPU parameters, supplied explicitly by the caller."""

    pi_p: float
    surrogate: str
    nn_correction: bool
    beta: float

    def __post_init__(self) -> None:
        pi_p = _finite_float(self.pi_p, name="pi_p")
        if not 0.0 < pi_p < 1.0:
            raise ValueError(f"pi_p must be in (0, 1), got {pi_p}.")
        if not isinstance(self.surrogate, str):
            raise TypeError("surrogate must be a string.")
        if self.surrogate not in {"sigmoid", "logistic"}:
            raise ValueError(
                "surrogate must be inherited as 'sigmoid' or 'logistic', "
                f"got {self.surrogate!r}."
            )
        if not isinstance(self.nn_correction, bool):
            raise TypeError("nn_correction must be bool.")
        beta = _finite_float(self.beta, name="beta")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        object.__setattr__(self, "pi_p", pi_p)
        object.__setattr__(self, "beta", beta)


@dataclass
class ObjectiveResult:
    """Differentiable total loss plus detached logging components."""

    total_loss: torch.Tensor
    raw_losses: dict[str, torch.Tensor]
    weighted_losses: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]
    batch_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def loss(self) -> torch.Tensor:
        """Alias used by optimizer loops."""
        return self.total_loss


class CUDAPoolSampler:
    """Uniform CUDA sampler with replacement for named latent pools."""

    def __init__(
        self,
        pools: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
        *,
        device: str | torch.device,
        seed: int,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"CUDAPoolSampler requires CUDA, got {self.device}.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDAPoolSampler requires CUDA, but torch.cuda.is_available() is False."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if not isinstance(pools, Mapping) or not pools:
            raise ValueError("pools must be a non-empty mapping.")

        self._pools: dict[str, torch.Tensor] = {}
        latent_dim: int | None = None
        for name, value in pools.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Pool names must be non-empty strings.")
            tensors = [value] if torch.is_tensor(value) else list(value)
            if not tensors:
                raise ValueError(f"Pool {name!r} is empty.")
            for tensor in tensors:
                if not torch.is_tensor(tensor):
                    raise TypeError(f"Pool {name!r} contains a non-tensor value.")
                if tensor.device.type != "cuda" or tensor.device != self.device:
                    raise ValueError(
                        f"Pool {name!r} must already be on {self.device}, "
                        f"got {tensor.device}."
                    )
                if tensor.ndim != 2:
                    raise ValueError(
                        f"Pool {name!r} tensors must have shape (N, D), "
                        f"got {tuple(tensor.shape)}."
                    )
                if tensor.shape[0] == 0:
                    raise ValueError(f"Pool {name!r} contains an empty tensor.")
                if latent_dim is None:
                    latent_dim = int(tensor.shape[1])
                elif int(tensor.shape[1]) != latent_dim:
                    raise ValueError(
                        f"Pool {name!r} latent dim {tensor.shape[1]} does not match "
                        f"the expected dim {latent_dim}."
                    )
            self._pools[name] = (
                tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {name: int(pool.shape[0]) for name, pool in self._pools.items()}

    def sample(self, pool_name: str, batch_size: int) -> torch.Tensor:
        if pool_name not in self._pools:
            raise KeyError(f"Unknown CUDA pool {pool_name!r}.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        pool = self._pools[pool_name]
        indices = torch.randint(
            int(pool.shape[0]),
            (batch_size,),
            device=self.device,
            generator=self.generator,
        )
        return pool.index_select(0, indices)


class CompositeDiscriminatorObjective:
    """Independent sampling and joint backward loss for configured terms."""

    def __init__(
        self,
        terms: Sequence[LossTermConfig],
        *,
        sampler: CUDAPoolSampler,
        nnpu_parameters: NNPUParameters,
        steps_per_epoch: int | None = None,
    ) -> None:
        if not terms:
            raise ValueError("At least one loss term must be configured.")
        names = [term.name for term in terms]
        if len(names) != len(set(names)):
            raise ValueError(f"Loss term names must be unique, got {names}.")
        active = [term for term in terms if term.active]
        if not active:
            raise ValueError("At least one enabled loss term must have positive weight.")
        active_types = [term.type for term in active]
        if len(active_types) != len(set(active_types)):
            raise ValueError("Only one active term of each loss type is supported.")
        required_pools = {
            pool_name for term in active for pool_name in _TERM_POOLS[term.type]
        }
        missing = sorted(required_pools - set(sampler.pool_sizes))
        if missing:
            raise ValueError(f"Active objectives require missing pools: {missing}.")

        self.terms = tuple(terms)
        self.active_terms = tuple(active)
        self.sampler = sampler
        self.nnpu_parameters = nnpu_parameters
        self.steps_per_epoch = self._resolve_steps_per_epoch(steps_per_epoch)

    def _resolve_steps_per_epoch(self, override: int | None) -> int:
        if override is not None:
            if isinstance(override, bool) or not isinstance(override, int):
                raise TypeError("steps_per_epoch must be null or an integer.")
            if override < 1:
                raise ValueError("steps_per_epoch must be at least 1.")
            return override
        replay_terms = [term for term in self.active_terms if term.type == "nnpu"]
        if not replay_terms:
            raise ValueError(
                "steps_per_epoch must be set explicitly when nnpu replay is inactive."
            )
        replay = replay_terms[0]
        sizes = self.sampler.pool_sizes
        derived = math.floor(
            sizes[PRETRAIN_POSITIVE] / replay.positive_batch_size
            + sizes[PRETRAIN_UNLABELED] / replay.negative_batch_size
        )
        return max(1, int(derived))

    def sample_batches(self) -> dict[str, torch.Tensor]:
        batches: dict[str, torch.Tensor] = {}
        for term in self.active_terms:
            positive_pool, negative_pool = _TERM_POOLS[term.type]
            batches[positive_pool] = self.sampler.sample(
                positive_pool, term.positive_batch_size
            )
            batches[negative_pool] = self.sampler.sample(
                negative_pool, term.negative_batch_size
            )
        return batches

    def compute(self, batch_logits: Mapping[str, torch.Tensor]) -> ObjectiveResult:
        if not isinstance(batch_logits, Mapping):
            raise TypeError("batch_logits must be a mapping keyed by pool name.")
        raw_losses: dict[str, torch.Tensor] = {}
        weighted_losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        batch_sizes: dict[str, int] = {}

        for term in self.active_terms:
            positive_pool, negative_pool = _TERM_POOLS[term.type]
            positive_logits = self._validated_logits(batch_logits, positive_pool)
            negative_logits = self._validated_logits(batch_logits, negative_pool)
            batch_sizes[positive_pool] = int(positive_logits.numel())
            batch_sizes[negative_pool] = int(negative_logits.numel())
            metrics[f"batch/{positive_pool}"] = positive_logits.new_tensor(
                float(positive_logits.numel())
            )
            metrics[f"batch/{negative_pool}"] = negative_logits.new_tensor(
                float(negative_logits.numel())
            )

            if term.type == "nnpu":
                parts = pu_risk(
                    positive_logits,
                    negative_logits,
                    pi_p=self.nnpu_parameters.pi_p,
                    surrogate=self.nnpu_parameters.surrogate,
                    nn_correction=self.nnpu_parameters.nn_correction,
                    beta=self.nnpu_parameters.beta,
                )
                raw_loss = parts["risk"]
                clamped = (
                    parts["neg_risk"] < -float(self.nnpu_parameters.beta)
                    if self.nnpu_parameters.nn_correction
                    else parts["neg_risk"].new_tensor(False)
                )
                metrics.update(
                    {
                        "nnpu/positive_risk": parts["pos_risk"],
                        "nnpu/estimated_negative_risk": parts["neg_risk"],
                        "nnpu/used_negative_risk": parts["neg_risk_used"],
                        "nnpu/clamped": clamped.to(dtype=torch.float32),
                    }
                )
            else:
                assert term.class_weights is not None
                positive_bce = F.softplus(-positive_logits).mean()
                negative_bce = F.softplus(negative_logits).mean()
                raw_loss = (
                    term.class_weights["positive"] * positive_bce
                    + term.class_weights["negative"] * negative_bce
                )
                metrics.update(
                    {
                        "gt_bce/positive": positive_bce.detach(),
                        "gt_bce/negative": negative_bce.detach(),
                    }
                )

            weighted_loss = term.weight * raw_loss
            raw_losses[term.name] = raw_loss
            weighted_losses[term.name] = weighted_loss
            metrics[f"loss/{term.name}/raw"] = raw_loss.detach()
            metrics[f"loss/{term.name}/weighted"] = weighted_loss.detach()

        total_loss = sum(weighted_losses.values())
        if total_loss.ndim != 0 or total_loss.device.type != "cuda":
            raise RuntimeError("Composite objective must produce a scalar CUDA loss.")
        metrics["loss/total"] = total_loss.detach()
        return ObjectiveResult(
            total_loss=total_loss,
            raw_losses=raw_losses,
            weighted_losses=weighted_losses,
            metrics=metrics,
            batch_sizes=batch_sizes,
        )

    def __call__(self, head: torch.nn.Module) -> ObjectiveResult:
        batches = self.sample_batches()
        logits: dict[str, torch.Tensor] = {}
        for pool_name, features in batches.items():
            output = head(features)
            if not torch.is_tensor(output) or output.numel() != features.shape[0]:
                raise ValueError(
                    f"Head output for {pool_name!r} must contain one logit per sample."
                )
            logits[pool_name] = output.reshape(-1)
        return self.compute(logits)

    def _validated_logits(
        self,
        batch_logits: Mapping[str, torch.Tensor],
        pool_name: str,
    ) -> torch.Tensor:
        if pool_name not in batch_logits:
            raise KeyError(f"Missing logits for required pool {pool_name!r}.")
        logits = batch_logits[pool_name]
        if not torch.is_tensor(logits):
            raise TypeError(f"Logits for {pool_name!r} must be a tensor.")
        if logits.device != self.sampler.device or logits.device.type != "cuda":
            raise ValueError(
                f"Logits for {pool_name!r} must be on {self.sampler.device}, "
                f"got {logits.device}."
            )
        if logits.ndim != 1 or logits.numel() == 0:
            raise ValueError(
                f"Logits for {pool_name!r} must be a non-empty 1D tensor."
            )
        return logits


def build_objective(
    config: Mapping[str, Any],
    *,
    pools: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    nnpu_parameters: NNPUParameters | Mapping[str, Any],
    device: str | torch.device,
    seed: int,
) -> CompositeDiscriminatorObjective:
    """Build a validated objective from the ``objective`` config mapping."""
    if not isinstance(config, Mapping):
        raise TypeError("objective config must be a mapping.")
    raw_terms = config.get("terms")
    if not isinstance(raw_terms, Mapping) or not raw_terms:
        raise ValueError("objective.terms must be a non-empty mapping.")

    terms: list[LossTermConfig] = []
    for name, raw_term in raw_terms.items():
        if not isinstance(raw_term, Mapping):
            raise TypeError(f"objective term {name!r} must be a mapping.")
        unknown = set(raw_term) - {
            "type",
            "enabled",
            "weight",
            "batch_size",
            "positive_fraction",
            "class_weights",
        }
        if unknown:
            raise ValueError(f"Unknown options for objective term {name!r}: {sorted(unknown)}.")
        if "type" not in raw_term:
            raise ValueError(f"objective term {name!r} is missing type.")
        terms.append(LossTermConfig(name=str(name), **dict(raw_term)))

    parameters = (
        nnpu_parameters
        if isinstance(nnpu_parameters, NNPUParameters)
        else NNPUParameters(**dict(nnpu_parameters))
    )
    active_terms = [term for term in terms if term.active]
    if not active_terms:
        raise ValueError("At least one enabled loss term must have positive weight.")
    required_pools = {
        pool_name
        for term in active_terms
        for pool_name in _TERM_POOLS[term.type]
    }
    missing = sorted(required_pools - set(pools))
    if missing:
        raise ValueError(f"Active objectives require missing pools: {missing}.")
    active_pools = {
        pool_name: pools[pool_name] for pool_name in required_pools
    }
    sampler = CUDAPoolSampler(active_pools, device=device, seed=seed)
    return CompositeDiscriminatorObjective(
        terms,
        sampler=sampler,
        nnpu_parameters=parameters,
        steps_per_epoch=config.get("steps_per_epoch"),
    )


__all__ = [
    "CUDAPoolSampler",
    "CompositeDiscriminatorObjective",
    "LossTermConfig",
    "NNPUParameters",
    "ObjectiveResult",
    "OFFLINE_GT_NEGATIVE",
    "OFFLINE_POSITIVE",
    "PRETRAIN_POSITIVE",
    "PRETRAIN_UNLABELED",
    "build_objective",
]
