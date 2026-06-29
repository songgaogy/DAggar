"""LoRA condition-injection for the DIPOLE negative branch.

Replaces the additive polarity-embedding with a low-rank adapter on the
condition-pathway ``nn.Linear`` modules of the flow head (and, optionally, the
condition aggregator). The positive branch is the original pretrained path
(LoRA OFF); the negative branch is ``base + LoRA delta``.

A single :class:`LoRARuntime` object is shared by every :class:`LoRALinear` of a
model and carries the per-forward branch state, so the existing single
``flow_head(...)`` call on a 2x batch can apply the LoRA delta to only the
negative (second) half via a per-row mask -- no forward-signature changes.
"""

from __future__ import annotations

import contextlib
import math
from typing import Callable, Iterable

import torch
import torch.nn as nn


# Condition-pathway Linear suffixes (relative module paths under flow_head /
# condition_aggregator). ``conditioner.time_mlp.*`` is intentionally excluded:
# the time embedding is not part of polarity conditioning.
_DEFAULT_TARGET_SUFFIXES: tuple[str, ...] = (
    # TimeConditioner condition MLPs (task_scene + joint cond)
    "task_scene_mlp.0",
    "task_scene_mlp.2",
    "cond_mlp.0",
    "cond_mlp.2",
    # FiLMResidualBlock1D condition projection (every down/mid/up/merge block)
    "cond_proj",
    # GatedCrossAttention1D projections
    "query_proj",
    "key_value_proj",
    "out_proj",
    "gate_proj",
    # AttentionConditionAggregator projections
    "key_proj",
    "value_proj",
    "output_proj.1",
    "output_proj.3",
)


def default_selector() -> Callable[[str], bool]:
    """Return a selector matching the default condition-pathway suffixes."""

    suffixes = _DEFAULT_TARGET_SUFFIXES

    def _selector(path: str) -> bool:
        return any(path == s or path.endswith("." + s) for s in suffixes)

    return _selector


class LoRARuntime:
    """Shared, per-forward branch state referenced by every LoRALinear of a model.

    - ``enabled=False``               -> positive/base-only forward (zero overhead)
    - ``enabled=True, row_mask=None`` -> pure-negative forward (delta on all rows)
    - ``enabled=True, row_mask=(N,)`` -> 2x-batch forward (delta only on True rows)
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.row_mask: torch.Tensor | None = None


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with an additive low-rank adapter.

    The original Linear is kept as the child ``self.base`` so its pretrained
    ``weight``/``bias`` Parameters are preserved bit-for-bit (their state_dict
    keys become ``<path>.base.weight`` -- see :func:`remap_legacy_cond_keys`).
    ``lora_B`` is zero-initialized so the delta is exactly 0 at construction
    (negative branch == positive branch at start).
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        runtime: LoRARuntime,
    ) -> None:
        super().__init__()
        self.base = base
        self.runtime = runtime
        self.r = int(rank)
        self.scaling = float(alpha) / float(rank) if rank else 0.0
        in_features = int(base.in_features)
        out_features = int(base.out_features)
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> delta == 0 at init.
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        lora_a = self.lora_A.to(x.dtype)
        lora_b = self.lora_B.to(x.dtype)
        return (self.dropout(x) @ lora_a.t() @ lora_b.t()) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        runtime = self.runtime
        if (not runtime.enabled) or (self.r <= 0):
            return out
        if runtime.row_mask is None:
            return out + self._delta(x)
        # Masked: add delta only to True rows along dim 0. The condition-pathway
        # Linears all broadcast the per-row cond, so dim 0 is always the batch.
        mask = runtime.row_mask
        view = (int(mask.shape[0]),) + (1,) * (x.dim() - 1)
        m = mask.view(view).to(out.dtype)
        return out + m * self._delta(x)


@contextlib.contextmanager
def lora_disabled(runtime: LoRARuntime):
    """Positive-only forward (online rollout, omega==0 path)."""
    prev = (runtime.enabled, runtime.row_mask)
    runtime.enabled, runtime.row_mask = False, None
    try:
        yield
    finally:
        runtime.enabled, runtime.row_mask = prev


@contextlib.contextmanager
def lora_full(runtime: LoRARuntime):
    """Pure-negative forward (delta on every row)."""
    prev = (runtime.enabled, runtime.row_mask)
    runtime.enabled, runtime.row_mask = True, None
    try:
        yield
    finally:
        runtime.enabled, runtime.row_mask = prev


@contextlib.contextmanager
def lora_masked(runtime: LoRARuntime, row_mask: torch.Tensor):
    """2x-batch forward: delta only on True rows of ``row_mask``."""
    prev = (runtime.enabled, runtime.row_mask)
    runtime.enabled, runtime.row_mask = True, row_mask
    try:
        yield
    finally:
        runtime.enabled, runtime.row_mask = prev


def _iter_target_linears(
    root: nn.Module, selector: Callable[[str], bool]
) -> list[tuple[nn.Module, str, nn.Linear]]:
    found: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in root.named_modules():
        # Never wrap the internal projections of an nn.MultiheadAttention: its
        # ``out_proj`` is an nn.Linear subclass that MHA accesses by ``.weight``
        # directly, and the attention path is intentionally out of LoRA scope.
        if isinstance(module, nn.MultiheadAttention):
            continue
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            path = f"{name}.{child_name}" if name else child_name
            if selector(path):
                found.append((module, child_name, child))
    return found


def apply_lora(
    flow_head: nn.Module,
    condition_aggregator: nn.Module | None,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    selector: Callable[[str], bool],
    runtime: LoRARuntime,
) -> int:
    """In-place replace target ``nn.Linear`` modules with ``LoRALinear``.

    Reuses the same ``nn.Linear`` objects (pretrained weights untouched).
    Returns the number of wrapped modules.
    """
    count = 0
    roots: list[nn.Module] = [flow_head]
    if condition_aggregator is not None:
        roots.append(condition_aggregator)
    for root in roots:
        for parent, attr, linear in _iter_target_linears(root, selector):
            setattr(
                parent,
                attr,
                LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout, runtime=runtime),
            )
            count += 1
    return count


def iter_lora_linears(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_target_paths(model: nn.Module) -> list[str]:
    """Module paths (e.g. ``flow_head...cond_proj``) of every wrapped Linear."""
    return [name for name, _ in iter_lora_linears(model)]


def remap_legacy_cond_keys(state_dict: dict, model: nn.Module) -> dict:
    """Rewrite pre-LoRA checkpoint keys for wrapped Linears.

    Wrapping renames ``<path>.weight`` -> ``<path>.base.weight``. Old pretrained
    checkpoints store the un-wrapped key, so without this remap a ``strict=False``
    load silently drops the pretrained condition-pathway weights. Idempotent:
    checkpoints already containing ``.base.`` keys are left untouched.
    """
    remapped = dict(state_dict)
    for path in lora_target_paths(model):
        for suffix in ("weight", "bias"):
            old_key = f"{path}.{suffix}"
            new_key = f"{path}.base.{suffix}"
            if old_key in remapped and new_key not in remapped:
                remapped[new_key] = remapped.pop(old_key)
    return remapped


@torch.no_grad()
def lora_health_metrics(model: nn.Module) -> dict[str, float]:
    """Aggregate LoRA-delta magnitude metrics (replaces polarity-embedding stats)."""
    b_norms: list[float] = []
    delta_norms: list[float] = []
    for _, lora in iter_lora_linears(model):
        b_norms.append(float(torch.linalg.matrix_norm(lora.lora_B.detach()).item()))
        delta = (lora.lora_B.detach() @ lora.lora_A.detach()) * lora.scaling
        delta_norms.append(float(torch.linalg.matrix_norm(delta).item()))
    if not b_norms:
        return {"lora_B_norm_mean": 0.0, "lora_delta_norm_mean": 0.0, "lora_num_layers": 0.0}
    return {
        "lora_B_norm_mean": float(sum(b_norms) / len(b_norms)),
        "lora_delta_norm_mean": float(sum(delta_norms) / len(delta_norms)),
        "lora_num_layers": float(len(b_norms)),
    }


__all__ = [
    "LoRARuntime",
    "LoRALinear",
    "apply_lora",
    "default_selector",
    "iter_lora_linears",
    "lora_target_paths",
    "remap_legacy_cond_keys",
    "lora_health_metrics",
    "lora_disabled",
    "lora_full",
    "lora_masked",
]
