"""Static divergence diagnostic for a trained DIPOLE checkpoint.

DIPOLE trains two independent full-tune flow policies (positive + negative) that
start from the same pretrained init. This reads a DIPOLE `.pt` and reports how far
the negative policy's weights have diverged from the positive policy's -- the
analog of the old dual-LoRA branch-divergence check. No env / dataset required;
meant to be called before any rollout-eval as a quick "did the negative policy
actually specialize" check.

Usage:
    python -m robosuite.pipeline.utils.diagnose_dipole_divergence \
        --checkpoint outputs/DIPOLE/<run>/checkpoints/latest.pt

Exits nonzero when divergence is below the warning threshold, so it can
short-circuit a sweep when the two policies are still ~identical.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


import torch


def _load_core_models(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint at {checkpoint_path} is not a dict payload.")
    core = payload.get("core")
    if not isinstance(core, dict):
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'core' field; not a DIPOLE checkpoint."
        )
    core_pos = core.get("core_pos")
    core_neg = core.get("core_neg")
    if not isinstance(core_pos, dict) or not isinstance(core_neg, dict):
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing 'core.core_pos'/'core.core_neg'; "
            "not a two-policy DIPOLE checkpoint."
        )
    model_pos = core_pos.get("model")
    model_neg = core_neg.get("model")
    if not isinstance(model_pos, dict) or not isinstance(model_neg, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} is missing a 'model' state dict.")
    return model_pos, model_neg


def _divergence_summary(
    model_pos: dict[str, Any], model_neg: dict[str, Any]
) -> dict[str, float]:
    """Per-parameter Frobenius divergence ``||neg - pos||`` between the policies.

    Reports absolute and pos-normalized (relative) divergence, averaged over all
    matched floating-point parameters. The relative measure drives the verdict so
    it is scale-independent across layers.
    """
    abs_norms: list[float] = []
    rel_norms: list[float] = []
    for key, pos_val in model_pos.items():
        neg_val = model_neg.get(key)
        if not isinstance(pos_val, torch.Tensor) or not isinstance(neg_val, torch.Tensor):
            continue
        if not pos_val.is_floating_point() or pos_val.shape != neg_val.shape:
            continue
        pos_t = pos_val.detach().float()
        neg_t = neg_val.detach().float()
        diff = float(torch.linalg.vector_norm(neg_t - pos_t).item())
        base = float(torch.linalg.vector_norm(pos_t).item())
        abs_norms.append(diff)
        rel_norms.append(diff / (base + 1e-8))
    if not abs_norms:
        raise KeyError(
            "No matched floating-point parameters between core_pos and core_neg."
        )
    return {
        "num_params": float(len(abs_norms)),
        "abs_divergence_mean": float(sum(abs_norms) / len(abs_norms)),
        "abs_divergence_max": float(max(abs_norms)),
        "rel_divergence_mean": float(sum(rel_norms) / len(rel_norms)),
        "rel_divergence_max": float(max(rel_norms)),
    }


def _verdict(rel_max: float, rel_mean: float, *, rel_good: float, rel_warn: float) -> str:
    if rel_max < rel_warn:
        return "BAD: pos/neg policies never diverged — the two are still ~identical."
    if rel_mean < rel_good:
        return "OK-ish: partial pos/neg divergence. Worth scanning omega; watch action quality."
    return "GOOD: pos/neg policies clearly diverge; CFG has room to move policy aggressiveness."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to DIPOLE .pt checkpoint")
    parser.add_argument(
        "--rel-good",
        type=float,
        default=0.02,
        help="Mean relative divergence above which the two policies are well-separated.",
    )
    parser.add_argument(
        "--rel-warn",
        type=float,
        default=1e-5,
        help="Max relative divergence below which the two policies never diverged.",
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

    model_pos, model_neg = _load_core_models(args.checkpoint)
    stats = _divergence_summary(model_pos, model_neg)
    num_params = int(stats["num_params"])
    rel_mean = stats["rel_divergence_mean"]
    rel_max = stats["rel_divergence_max"]

    verdict = _verdict(
        rel_max=rel_max,
        rel_mean=rel_mean,
        rel_good=args.rel_good,
        rel_warn=args.rel_warn,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_params": num_params,
        "abs_divergence_mean": stats["abs_divergence_mean"],
        "abs_divergence_max": stats["abs_divergence_max"],
        "rel_divergence_mean": rel_mean,
        "rel_divergence_max": rel_max,
        "thresholds": {
            "rel_good": args.rel_good,
            "rel_warn": args.rel_warn,
        },
        "verdict": verdict,
    }

    print(f"[dipole-divergence] checkpoint = {args.checkpoint}")
    print(f"[dipole-divergence] num_params           = {num_params}")
    print(f"[dipole-divergence] mean ||neg-pos||      = {stats['abs_divergence_mean']:.4f}")
    print(f"[dipole-divergence] max  ||neg-pos||      = {stats['abs_divergence_max']:.4f}")
    print(f"[dipole-divergence] mean rel divergence   = {rel_mean:.4f}")
    print(f"[dipole-divergence] max  rel divergence   = {rel_max:.4f}")
    print(
        f"[dipole-divergence] thresholds: rel_good>{args.rel_good}, "
        f"rel_warn<{args.rel_warn}"
    )
    print(f"[dipole-divergence] {verdict}")
    if args.emit_json:
        print(json.dumps(summary, indent=2))

    if args.strict and verdict.startswith("BAD"):
        return 2
    if not math.isfinite(rel_mean):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
