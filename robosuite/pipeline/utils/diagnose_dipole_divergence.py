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
    """Aggregate LoRA-adapter magnitude stats from a model state dict.

    For each wrapped condition Linear the state dict stores ``<path>.lora_A``
    (r, in) and ``<path>.lora_B`` (out, r). Per-layer delta (unscaled) is the
    Frobenius norm of ``lora_B @ lora_A`` -- the scaling factor lives on the live
    module, not the state dict, so it is omitted here.
    """
    b_norms: list[float] = []
    delta_norms: list[float] = []
    for key, value in model_state.items():
        if not key.endswith(".lora_B"):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        lora_b = value.detach().float().cpu()
        a_key = key[: -len(".lora_B")] + ".lora_A"
        lora_a = model_state.get(a_key)
        if not isinstance(lora_a, torch.Tensor):
            raise KeyError(f"State dict has '{key}' but no matching '{a_key}'.")
        lora_a = lora_a.detach().float().cpu()
        b_norms.append(float(torch.linalg.matrix_norm(lora_b).item()))
        delta = lora_b @ lora_a
        delta_norms.append(float(torch.linalg.matrix_norm(delta).item()))
    if not b_norms:
        raise KeyError(
            "State dict has no '.lora_B' adapter keys. Either this is not a "
            "LoRA-based DIPOLE checkpoint or the adapters were renamed."
        )
    return {
        "num_lora_layers": float(len(b_norms)),
        "lora_B_norm_mean": float(sum(b_norms) / len(b_norms)),
        "lora_delta_norm_mean": float(sum(delta_norms) / len(delta_norms)),
        "lora_delta_norm_max": float(max(delta_norms)),
    }


def _verdict(delta_max: float, delta_mean: float, *, delta_good: float, delta_warn: float) -> str:
    if delta_max < delta_warn:
        return "BAD: LoRA adapters never grew — negative branch is still ~identical to base."
    if delta_mean < delta_good:
        return "OK-ish: partial LoRA growth. Worth scanning omega; watch action quality."
    return "GOOD: LoRA adapters are clearly active; CFG has room to move policy aggressiveness."


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
