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
    "positive_logistic": (OFFLINE_POSITIVE,),
    "positive_safety_margin": (OFFLINE_POSITIVE,),
    "negative_logistic": (OFFLINE_GT_NEGATIVE,),
}
_TERM_TYPE_ALIASES = {
    "nnpu": "nnpu",
    "nnpu_replay": "nnpu",
    "supervised_bce": "supervised_bce",
    "supervised_gt_bce": "supervised_bce",
    "positive_logistic": "positive_logistic",
    "gt_positive": "positive_logistic",
    "positive_safety_margin": "positive_safety_margin",
    "negative_logistic": "negative_logistic",
    "gt_negative": "negative_logistic",
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
    positive_fraction: float | None = None
    class_weights: Mapping[str, float] | None = None
    safety_margin_weight: float | None = None
    margin_delta: float | None = None
    temperature: float | None = None
    boundary_source: str | None = None

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
        paired = self.type in {"nnpu", "supervised_bce"}
        minimum_batch_size = 2 if paired else 1
        if self.batch_size < minimum_batch_size:
            raise ValueError(
                f"{self.name}.batch_size must be at least {minimum_batch_size}."
            )
        if not paired:
            if self.positive_fraction is not None:
                raise ValueError(
                    f"{self.name}.positive_fraction is only valid for paired losses."
                )
            if self.class_weights is not None:
                raise ValueError(
                    f"{self.name}.class_weights is only valid for supervised_bce."
                )
            safety_values = {
                "safety_margin_weight": self.safety_margin_weight,
                "margin_delta": self.margin_delta,
                "temperature": self.temperature,
                "boundary_source": self.boundary_source,
            }
            if self.type != "positive_safety_margin":
                configured = sorted(
                    key for key, value in safety_values.items() if value is not None
                )
                if configured:
                    raise ValueError(
                        f"{self.name} safety-margin options are only valid for "
                        f"positive_safety_margin, got {configured}."
                    )
                return
            missing = sorted(
                key for key, value in safety_values.items() if value is None
            )
            if missing:
                raise ValueError(
                    f"{self.name} positive_safety_margin is missing {missing}."
                )
            safety_weight = _finite_float(
                self.safety_margin_weight,
                name=f"{self.name}.safety_margin_weight",
            )
            margin_delta = _finite_float(
                self.margin_delta,
                name=f"{self.name}.margin_delta",
            )
            temperature = _finite_float(
                self.temperature,
                name=f"{self.name}.temperature",
            )
            if safety_weight < 0.0:
                raise ValueError(
                    f"{self.name}.safety_margin_weight must be non-negative."
                )
            if margin_delta < 0.0:
                raise ValueError(f"{self.name}.margin_delta must be non-negative.")
            if temperature <= 0.0:
                raise ValueError(f"{self.name}.temperature must be positive.")
            if self.boundary_source != "parent_checkpoint":
                raise ValueError(
                    f"{self.name}.boundary_source must be 'parent_checkpoint', "
                    f"got {self.boundary_source!r}."
                )
            object.__setattr__(self, "safety_margin_weight", safety_weight)
            object.__setattr__(self, "margin_delta", margin_delta)
            object.__setattr__(self, "temperature", temperature)
            return

        configured_safety = sorted(
            key
            for key, value in {
                "safety_margin_weight": self.safety_margin_weight,
                "margin_delta": self.margin_delta,
                "temperature": self.temperature,
                "boundary_source": self.boundary_source,
            }.items()
            if value is not None
        )
        if configured_safety:
            raise ValueError(
                f"{self.name} safety-margin options are only valid for "
                f"positive_safety_margin, got {configured_safety}."
            )

        fraction = _finite_float(
            0.5 if self.positive_fraction is None else self.positive_fraction,
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
        if self.positive_fraction is None:
            raise ValueError(f"{self.name} does not define a positive batch split.")
        return int(round(self.batch_size * self.positive_fraction))

    @property
    def negative_batch_size(self) -> int:
        return int(self.batch_size - self.positive_batch_size)

    @property
    def pool_batch_sizes(self) -> dict[str, int]:
        pools = _TERM_POOLS[self.type]
        if len(pools) == 1:
            return {pools[0]: int(self.batch_size)}
        return {
            pools[0]: int(self.positive_batch_size),
            pools[1]: int(self.negative_batch_size),
        }

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
        positive_safety_boundary: float | None = None,
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
        active_pools = [pool_name for term in active for pool_name in term.pool_batch_sizes]
        if len(active_pools) != len(set(active_pools)):
            raise ValueError(
                "Active loss terms must use disjoint pools; do not combine legacy "
                "supervised_bce with separate GT risks."
            )
        required_pools = {pool_name for term in active for pool_name in term.pool_batch_sizes}
        missing = sorted(required_pools - set(sampler.pool_sizes))
        if missing:
            raise ValueError(f"Active objectives require missing pools: {missing}.")

        self.terms = tuple(terms)
        self.active_terms = tuple(active)
        self.sampler = sampler
        self.nnpu_parameters = nnpu_parameters
        safety_terms = [
            term for term in self.active_terms if term.type == "positive_safety_margin"
        ]
        if safety_terms:
            if positive_safety_boundary is None:
                raise ValueError(
                    "positive_safety_boundary is required for positive_safety_margin."
                )
            self.positive_safety_boundary = _finite_float(
                positive_safety_boundary,
                name="positive_safety_boundary",
            )
        else:
            self.positive_safety_boundary = None
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
            for pool_name, batch_size in term.pool_batch_sizes.items():
                batches[pool_name] = self.sampler.sample(pool_name, batch_size)
        return batches

    def compute(self, batch_logits: Mapping[str, torch.Tensor]) -> ObjectiveResult:
        if not isinstance(batch_logits, Mapping):
            raise TypeError("batch_logits must be a mapping keyed by pool name.")
        raw_losses: dict[str, torch.Tensor] = {}
        weighted_losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        batch_sizes: dict[str, int] = {}

        for term in self.active_terms:
            term_logits = {
                pool_name: self._validated_logits(batch_logits, pool_name)
                for pool_name in term.pool_batch_sizes
            }
            for pool_name, logits in term_logits.items():
                batch_sizes[pool_name] = int(logits.numel())
                metrics[f"batch/{pool_name}"] = logits.new_tensor(
                    float(logits.numel())
                )

            if term.type == "nnpu":
                positive_logits = term_logits[PRETRAIN_POSITIVE]
                negative_logits = term_logits[PRETRAIN_UNLABELED]
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
            elif term.type == "supervised_bce":
                assert term.class_weights is not None
                positive_logits = term_logits[OFFLINE_POSITIVE]
                negative_logits = term_logits[OFFLINE_GT_NEGATIVE]
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
            elif term.type == "positive_logistic":
                raw_loss = F.softplus(-term_logits[OFFLINE_POSITIVE]).mean()
                metrics["gt/positive_logistic"] = raw_loss.detach()
            elif term.type == "positive_safety_margin":
                assert term.safety_margin_weight is not None
                assert term.margin_delta is not None
                assert term.temperature is not None
                assert self.positive_safety_boundary is not None
                positive_logits = term_logits[OFFLINE_POSITIVE]
                positive_bce = F.softplus(-positive_logits).mean()
                boundary = positive_logits.new_tensor(self.positive_safety_boundary)
                target_logit = boundary + float(term.margin_delta)
                positive_safety = F.softplus(
                    (target_logit - positive_logits) / float(term.temperature)
                ).mean()
                raw_loss = (
                    positive_bce
                    + float(term.safety_margin_weight) * positive_safety
                )
                metrics.update(
                    {
                        "gt/positive_bce": positive_bce.detach(),
                        "gt/positive_safety_margin": positive_safety.detach(),
                        "gt/positive_combined": raw_loss.detach(),
                        "safety/m_k": boundary.detach(),
                        "safety/target_logit": target_logit.detach(),
                        "safety/margin_violation_fraction": (
                            positive_logits.detach() < target_logit
                        )
                        .to(dtype=torch.float32)
                        .mean(),
                    }
                )
            else:
                assert term.type == "negative_logistic"
                raw_loss = F.softplus(term_logits[OFFLINE_GT_NEGATIVE]).mean()
                metrics["gt/negative_logistic"] = raw_loss.detach()

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
    positive_safety_boundary: float | None = None,
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
            "safety_margin_weight",
            "margin_delta",
            "temperature",
            "boundary_source",
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
        for pool_name in term.pool_batch_sizes
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
        positive_safety_boundary=positive_safety_boundary,
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
