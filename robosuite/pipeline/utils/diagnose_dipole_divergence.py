"""Static branch-divergence diagnostic for a trained DIPOLE checkpoint.

Reads the LoRA adapters from a DIPOLE `.pt` and reports how strongly the
negative branch's condition-pathway adapters have grown away from the base
(positive) branch. No env / dataset required; meant to be called before any
rollout-eval as a quick "did the negative LoRA branch actually specialize"
check.

Usage:
    python -m robosuite.pipeline.utils.diagnose_dipole_divergence \
        --checkpoint outputs/DIPOLE/<run>/checkpoints/latest.pt

The diagnostic returns nonzero exit code when LoRA growth is below the warning
threshold, so it can short-circuit a sweep when the model has clearly collapsed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


def _load_model_state(checkpoint_path: Path) -> dict[str, Any]:
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint at {checkpoint_path} is not a dict payload.")
    core = payload.get("core")
    if not isinstance(core, dict):
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'core' field; not a DIPOLE checkpoint."
        )
    model_state = core.get("model")
    if not isinstance(model_state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} is missing 'core.model' state dict.")
    return model_state


def _lora_summary(model_state: dict[str, Any]) -> dict[str, float]:
    """Aggregate dual-LoRA branch-divergence stats from a model state dict.

    Each wrapped condition Linear stores ``<path>.pos_lora_A/pos_lora_B`` and
    ``<path>.neg_lora_A/neg_lora_B``. What matters for CFG guidance is how far the
    two branches diverge, so the per-layer delta reported here is the Frobenius
    norm of ``(neg_B @ neg_A) - (pos_B @ pos_A)`` -- the positive/negative branch
    difference. The scaling factor lives on the live module, not the state dict,
    so it is omitted here.
    """

    def _delta(prefix: str) -> torch.Tensor | None:
        b = model_state.get(prefix + "_B")
        a = model_state.get(prefix + "_A")
        if not isinstance(b, torch.Tensor) or not isinstance(a, torch.Tensor):
            return None
        return b.detach().float().cpu() @ a.detach().float().cpu()

    neg_b_norms: list[float] = []
    diff_norms: list[float] = []
    for key, value in model_state.items():
        if not key.endswith(".neg_lora_B") or not isinstance(value, torch.Tensor):
            continue
        path = key[: -len(".neg_lora_B")]
        neg_delta = _delta(path + ".neg_lora")
        pos_delta = _delta(path + ".pos_lora")
        if neg_delta is None:
            raise KeyError(f"State dict has '{key}' but is missing its neg_lora_A pair.")
        if pos_delta is None:
            pos_delta = torch.zeros_like(neg_delta)
        neg_b_norms.append(float(torch.linalg.matrix_norm(value.detach().float().cpu()).item()))
        diff_norms.append(float(torch.linalg.matrix_norm(neg_delta - pos_delta).item()))
    if not neg_b_norms:
        raise KeyError(
            "State dict has no '.neg_lora_B' adapter keys. Either this is not a "
            "dual-LoRA DIPOLE checkpoint or the adapters were renamed."
        )
    return {
        "num_lora_layers": float(len(neg_b_norms)),
        "lora_B_norm_mean": float(sum(neg_b_norms) / len(neg_b_norms)),
        "lora_delta_norm_mean": float(sum(diff_norms) / len(diff_norms)),
        "lora_delta_norm_max": float(max(diff_norms)),
    }


def _verdict(delta_max: float, delta_mean: float, *, delta_good: float, delta_warn: float) -> str:
    if delta_max < delta_warn:
        return "BAD: pos/neg adapters never diverged — the two branches are still ~identical."
    if delta_mean < delta_good:
        return "OK-ish: partial pos/neg divergence. Worth scanning omega; watch action quality."
    return "GOOD: pos/neg branches clearly diverge; CFG has room to move policy aggressiveness."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to DIPOLE .pt checkpoint")
    parser.add_argument(
        "--delta-good",
        type=float,
        default=0.1,
        help="Mean per-layer LoRA delta norm above which branches are well-diverged.",
    )
    parser.add_argument(
        "--delta-warn",
        type=float,
        default=1e-4,
        help="Max per-layer LoRA delta norm below which the adapters never escaped init.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if verdict is BAD (useful to short-circuit a sweep).",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit a JSON summary on stdout in addition to the human-readable lines.",
    )
    args = parser.parse_args()

    model_state = _load_model_state(args.checkpoint)
    stats = _lora_summary(model_state)
    num_layers = int(stats["num_lora_layers"])
    b_norm_mean = stats["lora_B_norm_mean"]
    delta_mean = stats["lora_delta_norm_mean"]
    delta_max = stats["lora_delta_norm_max"]

    verdict = _verdict(
        delta_max=delta_max,
        delta_mean=delta_mean,
        delta_good=args.delta_good,
        delta_warn=args.delta_warn,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_lora_layers": num_layers,
        "lora_B_norm_mean": b_norm_mean,
        "lora_delta_norm_mean": delta_mean,
        "lora_delta_norm_max": delta_max,
        "thresholds": {
            "delta_good": args.delta_good,
            "delta_warn": args.delta_warn,
        },
        "verdict": verdict,
    }

    print(f"[dipole-divergence] checkpoint = {args.checkpoint}")
    print(f"[dipole-divergence] num_lora_layers     = {num_layers}")
    print(f"[dipole-divergence] mean ||lora_B||      = {b_norm_mean:.4f}")
    print(f"[dipole-divergence] mean ||lora_B@lora_A|| = {delta_mean:.4f}")
    print(f"[dipole-divergence] max  ||lora_B@lora_A|| = {delta_max:.4f}")
    print(
        f"[dipole-divergence] thresholds: delta_good>{args.delta_good}, "
        f"delta_warn<{args.delta_warn}"
    )
    print(f"[dipole-divergence] {verdict}")
    if args.emit_json:
        print(json.dumps(summary, indent=2))

    if args.strict and verdict.startswith("BAD"):
        return 2
    if not math.isfinite(delta_mean):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
