"""Static branch-divergence diagnostic for a trained DIPOLE checkpoint.

Reads `core.model.polarity_embedding.weight` from a DIPOLE `.pt` and reports
how far the positive and negative branches have separated. No env / dataset
required; meant to be called before any rollout-eval as a quick "did the two
CFG branches actually specialize" check.

Usage:
    python -m robosuite.pipeline.tools.diagnose_dipole_divergence \
        --checkpoint outputs/DIPOLE/<run>/checkpoints/latest.pt

The diagnostic returns nonzero exit code when divergence is below the warning
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


POLARITY_KEY = "polarity_embedding.weight"


def _load_polarity_weight(checkpoint_path: Path) -> torch.Tensor:
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
    if POLARITY_KEY not in model_state:
        raise KeyError(
            f"State dict has no '{POLARITY_KEY}'. Either this is not a DIPOLE checkpoint "
            "or the polarity embedding was renamed."
        )
    weight = model_state[POLARITY_KEY]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != 2:
        raise ValueError(
            f"'{POLARITY_KEY}' must be a (2, cond_dim) tensor; got shape {tuple(weight.shape)}."
        )
    return weight.detach().float().cpu()


def _verdict(cos: float, max_norm: float, *, cos_good: float, cos_warn: float, norm_warn: float) -> str:
    if max_norm < norm_warn:
        return "BAD: polarity embedding never grew — branches are still at init scale."
    if cos > cos_warn:
        return (
            "BAD: pos/neg are near-collinear; CFG combine will mostly amplify noise. "
            "Consider zero_both + omega warmup, or upgrade to adaLN-Zero modulation."
        )
    if cos > cos_good:
        return "OK-ish: partial divergence. Worth scanning omega; watch action quality."
    return "GOOD: branches are clearly separated; CFG has room to move policy aggressiveness."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to DIPOLE .pt checkpoint")
    parser.add_argument(
        "--cos-good",
        type=float,
        default=0.7,
        help="Cosine threshold below which branches are considered well-diverged.",
    )
    parser.add_argument(
        "--cos-warn",
        type=float,
        default=0.95,
        help="Cosine threshold above which branches are considered collapsed.",
    )
    parser.add_argument(
        "--norm-warn",
        type=float,
        default=0.05,
        help="Min branch-embedding norm; below this the embedding never escaped init.",
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

    weight = _load_polarity_weight(args.checkpoint)
    neg = weight[0]
    pos = weight[1]
    pos_norm = float(torch.linalg.vector_norm(pos).item())
    neg_norm = float(torch.linalg.vector_norm(neg).item())
    diff_norm = float(torch.linalg.vector_norm(pos - neg).item())
    cos = float((pos * neg).sum().item() / (pos_norm * neg_norm + 1e-12))
    norm_ratio = max(pos_norm, neg_norm) / max(1e-12, min(pos_norm, neg_norm))
    max_norm = max(pos_norm, neg_norm)

    verdict = _verdict(
        cos=cos,
        max_norm=max_norm,
        cos_good=args.cos_good,
        cos_warn=args.cos_warn,
        norm_warn=args.norm_warn,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "polarity_embedding_pos_norm": pos_norm,
        "polarity_embedding_neg_norm": neg_norm,
        "polarity_embedding_l2_distance": diff_norm,
        "polarity_embedding_cos": cos,
        "polarity_norm_ratio_max_over_min": norm_ratio,
        "thresholds": {
            "cos_good": args.cos_good,
            "cos_warn": args.cos_warn,
            "norm_warn": args.norm_warn,
        },
        "verdict": verdict,
    }

    print(f"[dipole-divergence] checkpoint = {args.checkpoint}")
    print(f"[dipole-divergence] pos_norm   = {pos_norm:.4f}")
    print(f"[dipole-divergence] neg_norm   = {neg_norm:.4f}")
    print(f"[dipole-divergence] ||pos-neg|| = {diff_norm:.4f}")
    print(f"[dipole-divergence] cos(pos, neg) = {cos:+.4f}")
    print(f"[dipole-divergence] norm ratio (max/min) = {norm_ratio:.2f}")
    print(
        f"[dipole-divergence] thresholds: cos_good<{args.cos_good}, cos_warn>{args.cos_warn}, "
        f"norm_warn<{args.norm_warn}"
    )
    print(f"[dipole-divergence] {verdict}")
    if args.emit_json:
        print(json.dumps(summary, indent=2))

    if args.strict and verdict.startswith("BAD"):
        return 2
    if not math.isfinite(cos):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
