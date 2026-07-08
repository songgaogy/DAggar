"""Dual-LoRA condition-injection for the DIPOLE positive/negative branches.

The shared flow backbone is **frozen**; polarity is expressed by two low-rank
adapters that cover both the condition-pathway ``nn.Linear`` modules of the flow
head (and, optionally, the condition aggregator) via :class:`LoRALinear`, and the
UNet denoising-pathway ``nn.Conv1d`` modules via :class:`LoRAConv1d`:

- positive policy = ``base + pos_LoRA``
- negative policy = ``base + neg_LoRA``

Covering the Conv1d denoiser is essential under a frozen base: it holds most of
the flow head's capacity, so condition-only adapters cannot fit the data.

Both adapters are zero-initialized on their ``B`` matrix, so at construction the
delta is exactly 0 and ``pos == neg == base``. The two adapters are symmetric
(they share the same rank / alpha / dropout / adapter_lr).

A single :class:`LoRARuntime` object is shared by every :class:`LoRALinear` of a
model and carries the per-forward branch state, so the existing single
``flow_head(...)`` call on a 2x batch can apply the pos adapter to the positive
(first) half and the neg adapter to the negative (second) half via a per-row
mask -- no forward-signature changes.
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

    - ``active=False``                        -> base-only forward (zero overhead)
    - ``active=True, row_mask=None``          -> whole-batch forward with the
      ``branch`` adapter ("pos" or "neg") applied to every row
    - ``active=True, row_mask=(N,)``          -> 2x-batch forward: the pos adapter
      on False rows, the neg adapter on True rows
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.branch: str = "pos"
        self.row_mask: torch.Tensor | None = None


class LoRALinear(nn.Module):
    """Wraps a (frozen) ``nn.Linear`` with two additive low-rank adapters.

    The original Linear is kept as the child ``self.base`` so its pretrained
    ``weight``/``bias`` Parameters are preserved bit-for-bit (their state_dict
    keys become ``<path>.base.weight`` -- see :func:`remap_legacy_cond_keys`).
    Two symmetric adapters ``(pos_lora_A, pos_lora_B)`` and
    ``(neg_lora_A, neg_lora_B)`` express polarity. Both ``*_B`` are
    zero-initialized so the delta is exactly 0 at construction
    (``pos == neg == base`` at start).
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
        self.pos_lora_A = nn.Parameter(torch.zeros(self.r, in_features))
        self.pos_lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        self.neg_lora_A = nn.Parameter(torch.zeros(self.r, in_features))
        self.neg_lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.pos_lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.neg_lora_A, a=math.sqrt(5))
        # Both *_B stay zero -> delta == 0 at init.
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def _delta_for(self, x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        lora_a = lora_A.to(x.dtype)
        lora_b = lora_B.to(x.dtype)
        return (self.dropout(x) @ lora_a.t() @ lora_b.t()) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        runtime = self.runtime
        if (not runtime.active) or (self.r <= 0):
            return out
        if runtime.row_mask is None:
            if runtime.branch == "neg":
                return out + self._delta_for(x, self.neg_lora_A, self.neg_lora_B)
            return out + self._delta_for(x, self.pos_lora_A, self.pos_lora_B)
        # Masked 2x-batch: pos adapter on False rows, neg adapter on True rows.
        # The condition-pathway Linears all broadcast the per-row cond, so dim 0
        # is always the batch.
        mask = runtime.row_mask
        view = (int(mask.shape[0]),) + (1,) * (x.dim() - 1)
        m = mask.view(view).to(out.dtype)  # 1.0 on neg rows, 0.0 on pos rows
        pos_delta = self._delta_for(x, self.pos_lora_A, self.pos_lora_B)
        neg_delta = self._delta_for(x, self.neg_lora_A, self.neg_lora_B)
        return out + (1.0 - m) * pos_delta + m * neg_delta

    # --- health-metric helpers (uniform across LoRALinear / LoRAConv1d) ---
    def b_tensor(self, branch: str) -> torch.Tensor:
        return self.pos_lora_B if branch == "pos" else self.neg_lora_B

    def delta_weight(self, branch: str) -> torch.Tensor:
        if branch == "pos":
            return (self.pos_lora_B @ self.pos_lora_A) * self.scaling
        return (self.neg_lora_B @ self.neg_lora_A) * self.scaling


class LoRAConv1d(nn.Module):
    """Wraps a (frozen) ``nn.Conv1d`` with two additive low-rank adapters.

    Each adapter factorizes the conv delta as ``B @ A``: ``A`` is an
    ``in->rank`` conv reusing the base kernel geometry (kernel/stride/padding/
    dilation) so its output length matches the base, and ``B`` is a ``rank->out``
    1x1 conv. ``B`` is zero-initialized so the delta is 0 at construction
    (``pos == neg == base`` at start). This is what gives the frozen UNet
    denoising pathway (input_proj / conv1 / conv2 / up-down samples / output_conv)
    real branch-specific adaptation capacity -- LoRALinear only covers the
    condition-injection Linears, which is not enough on its own.
    """

    def __init__(
        self,
        base: nn.Conv1d,
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
        in_ch = int(base.in_channels)
        out_ch = int(base.out_channels)
        self.pos_lora_A = nn.Conv1d(
            in_ch, self.r, kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, dilation=base.dilation, groups=1, bias=False,
        )
        self.pos_lora_B = nn.Conv1d(self.r, out_ch, kernel_size=1, bias=False)
        self.neg_lora_A = nn.Conv1d(
            in_ch, self.r, kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, dilation=base.dilation, groups=1, bias=False,
        )
        self.neg_lora_B = nn.Conv1d(self.r, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.pos_lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.neg_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.pos_lora_B.weight)
        nn.init.zeros_(self.neg_lora_B.weight)
        # Both B convs stay zero -> delta == 0 at init.
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def _delta_for(self, x: torch.Tensor, lora_A: nn.Conv1d, lora_B: nn.Conv1d) -> torch.Tensor:
        return lora_B(lora_A(self.dropout(x))) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        runtime = self.runtime
        if (not runtime.active) or (self.r <= 0):
            return out
        if runtime.row_mask is None:
            if runtime.branch == "neg":
                return out + self._delta_for(x, self.neg_lora_A, self.neg_lora_B)
            return out + self._delta_for(x, self.pos_lora_A, self.pos_lora_B)
        # Masked 2x-batch: pos adapter on False rows, neg adapter on True rows.
        # Conv inputs/outputs are (B, C, L), so dim 0 is always the batch.
        mask = runtime.row_mask
        view = (int(mask.shape[0]),) + (1,) * (out.dim() - 1)
        m = mask.view(view).to(out.dtype)  # 1.0 on neg rows, 0.0 on pos rows
        pos_delta = self._delta_for(x, self.pos_lora_A, self.pos_lora_B)
        neg_delta = self._delta_for(x, self.neg_lora_A, self.neg_lora_B)
        return out + (1.0 - m) * pos_delta + m * neg_delta

    # --- health-metric helpers (uniform across LoRALinear / LoRAConv1d) ---
    def b_tensor(self, branch: str) -> torch.Tensor:
        conv = self.pos_lora_B if branch == "pos" else self.neg_lora_B
        return conv.weight

    def delta_weight(self, branch: str) -> torch.Tensor:
        lora_A = self.pos_lora_A if branch == "pos" else self.neg_lora_A
        lora_B = self.pos_lora_B if branch == "pos" else self.neg_lora_B
        # delta_W[o,i,k] = sum_r B[o,r,0] * A[r,i,k]
        return torch.einsum("or,rik->oik", lora_B.weight[:, :, 0], lora_A.weight) * self.scaling


@contextlib.contextmanager
def lora_base_only(runtime: LoRARuntime):
    """Base-only forward (no adapter delta; diagnostics)."""
    prev = (runtime.active, runtime.branch, runtime.row_mask)
    runtime.active, runtime.row_mask = False, None
    try:
        yield
    finally:
        runtime.active, runtime.branch, runtime.row_mask = prev


@contextlib.contextmanager
def lora_branch(runtime: LoRARuntime, branch: str):
    """Whole-batch forward with the ``branch`` ("pos"|"neg") adapter on every row."""
    if branch not in ("pos", "neg"):
        raise ValueError(f"lora_branch expects 'pos' or 'neg', got {branch!r}")
    prev = (runtime.active, runtime.branch, runtime.row_mask)
    runtime.active, runtime.branch, runtime.row_mask = True, branch, None
    try:
        yield
    finally:
        runtime.active, runtime.branch, runtime.row_mask = prev


def lora_positive(runtime: LoRARuntime):
    """Whole-batch positive forward (``base + pos_LoRA``)."""
    return lora_branch(runtime, "pos")


def lora_negative(runtime: LoRARuntime):
    """Whole-batch negative forward (``base + neg_LoRA``)."""
    return lora_branch(runtime, "neg")


@contextlib.contextmanager
def lora_masked(runtime: LoRARuntime, row_mask: torch.Tensor):
    """2x-batch forward: pos adapter on False rows, neg adapter on True rows."""
    prev = (runtime.active, runtime.branch, runtime.row_mask)
    runtime.active, runtime.row_mask = True, row_mask
    try:
        yield
    finally:
        runtime.active, runtime.branch, runtime.row_mask = prev


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


def _iter_target_convs(root: nn.Module) -> list[tuple[nn.Module, str, nn.Conv1d]]:
    """Every ``nn.Conv1d`` child (groups==1) -- the UNet denoising pathway."""
    found: list[tuple[nn.Module, str, nn.Conv1d]] = []
    for _, module in root.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRAConv1d):
                continue  # already wrapped
            if isinstance(child, nn.Conv1d) and int(child.groups) == 1:
                found.append((module, child_name, child))
    return found


def apply_lora_conv1d(
    flow_head: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    runtime: LoRARuntime,
) -> int:
    """In-place wrap every ``nn.Conv1d`` in ``flow_head`` with ``LoRAConv1d``.

    Reuses the same ``nn.Conv1d`` objects (pretrained weights untouched). Returns
    the number of wrapped modules. This adapts the actual denoising pathway
    (input_proj / conv1 / conv2 / up-down samples / output_conv) which the
    condition-pathway ``LoRALinear`` set does not cover.
    """
    count = 0
    for parent, attr, conv in _iter_target_convs(flow_head):
        setattr(
            parent,
            attr,
            LoRAConv1d(conv, rank=rank, alpha=alpha, dropout=dropout, runtime=runtime),
        )
        count += 1
    return count


def is_lora_param(name: str) -> bool:
    """True for a parameter name belonging to either adapter."""
    return (".pos_lora_" in name) or (".neg_lora_" in name)


def freeze_base_params(model: nn.Module) -> int:
    """Freeze every parameter that is not a pos/neg LoRA adapter.

    Only ``pos_lora_A/B`` and ``neg_lora_A/B`` stay trainable; the whole backbone
    (encoders, fusion, aggregator base, flow-head base) is frozen. Returns the
    number of frozen parameter tensors.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if not is_lora_param(name):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def iter_lora_linears(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Every LoRA-wrapped module (Linear condition pathway + Conv1d denoiser)."""
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            yield name, module


def lora_target_paths(model: nn.Module) -> list[str]:
    """Module paths of every wrapped module (Linear + Conv1d)."""
    return [name for name, _ in iter_lora_modules(model)]


def remap_legacy_cond_keys(state_dict: dict, model: nn.Module) -> dict:
    """Rewrite legacy checkpoint keys for wrapped Linears onto the dual-LoRA layout.

    Two legacy forms are handled, idempotently (checkpoints already in the new
    layout are left untouched):

    1. **Pre-LoRA pretrained** -- wrapping renames ``<path>.weight`` ->
       ``<path>.base.weight``. Without this remap a ``strict=False`` load silently
       drops the pretrained condition-pathway weights.
    2. **Single-LoRA** -- the old ``<path>.lora_A``/``<path>.lora_B`` (the negative
       branch) is seeded into ``<path>.neg_lora_A``/``<path>.neg_lora_B``; the
       positive adapter is left at its zero-init (matching the old positive branch
       which was base-only). Legacy keys are popped so ``strict=False`` reports no
       unexpected keys.
    """
    remapped = dict(state_dict)
    for path in lora_target_paths(model):
        # (1) base weights.
        for suffix in ("weight", "bias"):
            old_key = f"{path}.{suffix}"
            new_key = f"{path}.base.{suffix}"
            if old_key in remapped and new_key not in remapped:
                remapped[new_key] = remapped.pop(old_key)
        # (2) legacy single adapter -> negative adapter.
        for suffix in ("lora_A", "lora_B"):
            old_key = f"{path}.{suffix}"
            new_key = f"{path}.neg_{suffix}"
            if old_key in remapped:
                if new_key not in remapped:
                    remapped[new_key] = remapped.pop(old_key)
                else:
                    remapped.pop(old_key)
    return remapped


@torch.no_grad()
def lora_health_metrics(model: nn.Module) -> dict[str, float]:
    """Aggregate per-adapter LoRA magnitude metrics over Linear + Conv1d adapters.

    Norms are Frobenius (flattened), so they are uniform across the 2D Linear
    deltas and 3D Conv1d deltas.
    """
    stats: dict[str, list[float]] = {
        "pos_lora_B_norm": [],
        "pos_lora_delta_norm": [],
        "neg_lora_B_norm": [],
        "neg_lora_delta_norm": [],
    }

    def _fro(t: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(t.detach().reshape(-1)).item())

    for _, lora in iter_lora_modules(model):
        for prefix in ("pos", "neg"):
            stats[f"{prefix}_lora_B_norm"].append(_fro(lora.b_tensor(prefix)))
            stats[f"{prefix}_lora_delta_norm"].append(_fro(lora.delta_weight(prefix)))
    num_layers = len(stats["pos_lora_B_norm"])
    if num_layers == 0:
        return {
            "pos_lora_B_norm_mean": 0.0,
            "pos_lora_delta_norm_mean": 0.0,
            "neg_lora_B_norm_mean": 0.0,
            "neg_lora_delta_norm_mean": 0.0,
            "lora_num_layers": 0.0,
        }
    return {
        "pos_lora_B_norm_mean": float(sum(stats["pos_lora_B_norm"]) / num_layers),
        "pos_lora_delta_norm_mean": float(sum(stats["pos_lora_delta_norm"]) / num_layers),
        "neg_lora_B_norm_mean": float(sum(stats["neg_lora_B_norm"]) / num_layers),
        "neg_lora_delta_norm_mean": float(sum(stats["neg_lora_delta_norm"]) / num_layers),
        "lora_num_layers": float(num_layers),
    }


__all__ = [
    "LoRARuntime",
    "LoRALinear",
    "LoRAConv1d",
    "apply_lora",
    "apply_lora_conv1d",
    "default_selector",
    "freeze_base_params",
    "is_lora_param",
    "iter_lora_linears",
    "iter_lora_modules",
    "lora_target_paths",
    "remap_legacy_cond_keys",
    "lora_health_metrics",
    "lora_base_only",
    "lora_branch",
    "lora_positive",
    "lora_negative",
    "lora_masked",
]
