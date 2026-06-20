"""Small plotting helpers for frozen nnPU discriminator traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator


@torch.no_grad()
def score_chunk_features(
    discriminator: FrozenNNPUDiscriminator,
    chunk_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return tensor-native failure, probability, decision, and reward traces."""
    failure = discriminator.failure_score(chunk_feature=chunk_features)
    probability = torch.sigmoid(failure - float(discriminator.threshold))
    return {
        "failure_score": failure,
        "prob_failure": probability,
        "decision": failure >= float(discriminator.threshold),
        "intrinsic_reward": -probability,
    }


def write_discriminator_trace(
    output_dir: str | Path,
    trace: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a CSV, PNG, and JSON summary from a one-dimensional trace."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cpu = {name: value.detach().reshape(-1).cpu() for name, value in trace.items()}
    lengths = {int(value.numel()) for value in cpu.values()}
    if len(lengths) != 1:
        raise ValueError(f"Trace lengths differ: {sorted(lengths)}")
    count = lengths.pop()
    csv_path = directory / "nnpu_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *cpu.keys()])
        for step in range(count):
            writer.writerow([step, *[float(value[step].item()) for value in cpu.values()]])
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(cpu["failure_score"].numpy(), label="failure score")
    axes[0].plot(cpu["prob_failure"].numpy(), label="failure probability")
    axes[1].plot(cpu["intrinsic_reward"].numpy(), label="intrinsic reward")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    axes[1].set_xlabel("step")
    figure.tight_layout()
    png_path = directory / "nnpu_scores.png"
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps({"num_steps": count, **(metadata or {})}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"csv": csv_path, "plot": png_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        help="Optional torch file containing a chunk_features tensor.",
    )
    parser.add_argument("--nnpu-ckpt")
    parser.add_argument("--task-name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/nnpu_trace")
    args = parser.parse_args()
    if not args.features:
        return
    if not args.nnpu_ckpt or not args.task_name:
        parser.error("--nnpu-ckpt and --task-name are required with --features")
    features = torch.load(args.features, map_location=args.device, weights_only=False)
    if isinstance(features, dict):
        features = features["chunk_features"]
    discriminator = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=args.nnpu_ckpt,
        task_name=args.task_name,
        device=args.device,
    )
    trace = score_chunk_features(discriminator, features)
    outputs = write_discriminator_trace(
        args.output_dir,
        trace,
        metadata={
            "nnpu_checkpoint": discriminator.ckpt_path,
            "task_name": discriminator.task_name,
            "threshold": discriminator.threshold,
            "threshold_source": "checkpoint",
        },
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["score_chunk_features", "write_discriminator_trace"]
