"""Plotting + HUD-video helpers for the frozen nnPU discriminator.

Two layers live here:
  * ``score_chunk_features`` / ``write_discriminator_trace`` — generic 1-D trace
    dump used by the standalone ``main`` CLI.
  * ``visualize_selected_trajectory_discriminator_nnpu`` — per-frame failure
    timeseries (CSV + plot) and a HUD/red-border MP4 for one rollout, consumed
    by ``vis_vast``. This mirrors the ``dipole-rl/v0-kingback`` discriminator
    visualization, adapted to nnPU per-frame failure scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.common.types import Transition


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


# --------------------------------------------------------------------------- #
# Per-trajectory discriminator visualization (CSV + plot + HUD video).
# --------------------------------------------------------------------------- #


@dataclass
class DiscriminatorVizResult:
    output_dir: Path
    scores_csv: Path
    plot_png: Path
    plot_pdf: Path
    video: Path
    summary_json: Path


def _safe_id(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def select_camera(camera_names: Sequence[str], camera_name: str | None) -> str:
    """Resolve which camera to render: explicit override, else agentview, else first."""
    names = [str(name) for name in camera_names]
    if not names:
        raise ValueError("camera_names is empty")
    if camera_name is not None:
        if str(camera_name) not in names:
            raise KeyError(
                f"camera {camera_name!r} not available; choices: {names}"
            )
        return str(camera_name)
    return "agentview" if "agentview" in names else names[0]


def _extract_frames(transitions: list[Transition], camera_name: str) -> np.ndarray:
    """Stack a single camera's RGB frames from transition obs: (T, H, W, 3) uint8."""
    return np.stack(
        [np.asarray(item.obs[camera_name], dtype=np.uint8) for item in transitions], axis=0
    )


def _draw_red_border(frame: np.ndarray, thickness: int) -> np.ndarray:
    out = frame.copy()
    t = max(1, int(thickness))
    red = np.array([255, 0, 0], dtype=np.uint8)
    out[:t, :, :] = red
    out[-t:, :, :] = red
    out[:, :t, :] = red
    out[:, -t:, :] = red
    return out


def _overlay_hud(
    image: np.ndarray,
    *,
    score: float,
    threshold: float,
    pred_failure: bool,
    intrinsic_reward: float,
    frame_idx: int,
    total: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    """Draw a translucent HUD bar with frame index, score/tau/r_disc, and verdict."""
    scale = max(1, int(image.shape[0]) // 128)
    pil = Image.fromarray(np.ascontiguousarray(image)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bar_height = 46 * scale
    draw.rectangle([0, 0, pil.size[0], bar_height], fill=(0, 0, 0, 160))
    draw.text((4 * scale, 2 * scale), f"frame {int(frame_idx)}/{int(total)}", fill=(255, 255, 255, 255), font=font)
    draw.text(
        (4 * scale, 15 * scale),
        f"score={float(score):+.3f}  tau={float(threshold):+.3f}  r_disc={float(intrinsic_reward):+.3f}",
        fill=(255, 255, 255, 255),
        font=font,
    )
    verdict = "FAIL" if pred_failure else "OK"
    verdict_color = (255, 80, 80, 255) if pred_failure else (80, 255, 80, 255)
    draw.text((4 * scale, 30 * scale), f"PRED: {verdict}", fill=verdict_color, font=font)
    composited = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.asarray(composited, dtype=np.uint8)


def _write_video(
    path: Path,
    frames: np.ndarray,
    *,
    scores: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
    intrinsic: np.ndarray,
    fps: int,
    border_thickness: int,
    flip_vertical: bool,
) -> None:
    """Render an MP4 with a per-frame HUD and a red border on predicted failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(frames.shape[1]) // 128)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8 * scale)
    except OSError:
        font = ImageFont.load_default()
    total = int(frames.shape[0])
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for idx in range(total):
            frame = np.asarray(frames[idx], dtype=np.uint8)
            if flip_vertical:
                frame = np.flipud(frame)
            pred = bool(predictions[idx])
            if pred:
                frame = _draw_red_border(frame, border_thickness * scale)
            frame = _overlay_hud(
                frame,
                score=float(scores[idx]),
                threshold=float(threshold),
                pred_failure=pred,
                intrinsic_reward=float(intrinsic[idx]),
                frame_idx=idx,
                total=total,
                font=font,
            )
            writer.append_data(frame)


def _plot_disc_scores(
    path_base: Path,
    *,
    scores: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
    intrinsic: np.ndarray,
    title: str,
) -> tuple[Path, Path]:
    """2-subplot discriminator timeseries: failure score (+tau) and intrinsic reward."""
    steps = np.arange(int(scores.shape[0]), dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(48, 28), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, scores, label="nnPU failure score", color="tab:blue")
    axes[0].axhline(float(threshold), color="tab:red", linestyle="--", label=f"tau={float(threshold):.3f}")
    if bool(predictions.any()):
        axes[0].fill_between(
            steps,
            scores.min(),
            scores.max(),
            where=predictions.astype(bool),
            color="tab:red",
            alpha=0.12,
            label="pred failure",
        )
    axes[0].set_ylabel("failure score")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, intrinsic, label="intrinsic reward (-sigmoid(score-tau))", color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(-1.0, color="gray", linewidth=1, linestyle=":")
    axes[1].set_ylabel("r_disc")
    axes[1].set_xlabel("step")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _write_scores_csv(
    csv_path: Path,
    *,
    scores: np.ndarray,
    intrinsic: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "failure_score", "intrinsic_reward", "threshold", "pred_failure"])
        for step in range(int(scores.shape[0])):
            writer.writerow(
                [
                    step,
                    float(scores[step]),
                    float(intrinsic[step]),
                    float(threshold),
                    int(predictions[step]),
                ]
            )


def write_rollout_video(
    path: Path,
    transitions: list[Transition],
    *,
    camera_names: Sequence[str],
    camera_name: str | None = None,
    fps: int = 20,
    flip_vertical: bool = True,
) -> Path | None:
    """Write a plain single-camera rollout MP4 (no HUD)."""
    if not transitions:
        return None
    selected = select_camera(camera_names, camera_name)
    frames = _extract_frames(transitions, selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for idx in range(int(frames.shape[0])):
            frame = np.asarray(frames[idx], dtype=np.uint8)
            if flip_vertical:
                frame = np.flipud(frame)
            writer.append_data(np.ascontiguousarray(frame))
    return path


def visualize_selected_trajectory_discriminator_nnpu(
    *,
    output_dir: Path,
    transitions: list[Transition],
    camera_names: Sequence[str],
    failure_score: np.ndarray,
    intrinsic_reward: np.ndarray,
    pred_failure: np.ndarray,
    threshold: float,
    ckpt_path: str,
    task_name: str,
    video_fps: int,
    camera_name: str | None = None,
    border_thickness: int = 10,
    flip_vertical: bool = True,
) -> DiscriminatorVizResult:
    """Render per-frame nnPU failure CSV + plot + HUD video for one trajectory.

    Consumes the already-computed per-frame failure/intrinsic series (no model
    rebuild). Frames are pulled from ``transitions[i].obs[camera]`` and truncated
    to the score length when a window cap shortened the series.
    """
    if not transitions:
        raise RuntimeError("Cannot visualize discriminator on an empty trajectory.")
    scores = np.asarray(failure_score, dtype=np.float32).reshape(-1)
    intrinsic = np.asarray(intrinsic_reward, dtype=np.float32).reshape(-1)
    predictions = np.asarray(pred_failure).reshape(-1).astype(np.int64)
    length = int(scores.shape[0])
    if not (intrinsic.shape[0] == length and predictions.shape[0] == length):
        raise ValueError(
            "failure_score, intrinsic_reward, pred_failure must share length; got "
            f"{scores.shape[0]}, {intrinsic.shape[0]}, {predictions.shape[0]}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_camera = select_camera(camera_names, camera_name)
    frames = _extract_frames(transitions, selected_camera)
    if int(frames.shape[0]) > length:
        frames = frames[:length]
    elif int(frames.shape[0]) < length:
        scores = scores[: int(frames.shape[0])]
        intrinsic = intrinsic[: int(frames.shape[0])]
        predictions = predictions[: int(frames.shape[0])]
        length = int(frames.shape[0])

    scores_csv = out_dir / "bce_scores.csv"
    plot_base = out_dir / "discriminator_timeseries"
    video_path = out_dir / f"rollout_discriminator_{_safe_id(selected_camera)}.mp4"
    summary_path = out_dir / "summary.json"

    _write_scores_csv(
        scores_csv,
        scores=scores,
        intrinsic=intrinsic,
        threshold=float(threshold),
        predictions=predictions,
    )
    plot_png, plot_pdf = _plot_disc_scores(
        plot_base,
        scores=scores,
        threshold=float(threshold),
        predictions=predictions,
        intrinsic=intrinsic,
        title=f"{task_name} discriminator ({selected_camera})",
    )
    _write_video(
        video_path,
        frames,
        scores=scores,
        threshold=float(threshold),
        predictions=predictions,
        intrinsic=intrinsic,
        fps=int(video_fps),
        border_thickness=int(border_thickness),
        flip_vertical=bool(flip_vertical),
    )

    first_pred = np.where(predictions.astype(bool))[0]
    summary = {
        "nnpu_checkpoint": str(ckpt_path),
        "task_name": str(task_name),
        "threshold": float(threshold),
        "threshold_source": "checkpoint",
        "camera_name": selected_camera,
        "num_frames": int(length),
        "predicted_failure_frames": int(predictions.sum()),
        "first_pred_failure_frame": None if first_pred.size == 0 else int(first_pred[0]),
        "score_min": float(np.min(scores)) if length else None,
        "score_mean": float(np.mean(scores)) if length else None,
        "score_max": float(np.max(scores)) if length else None,
        "outputs": {
            "scores_csv": str(scores_csv),
            "plot_png": str(plot_png),
            "plot_pdf": str(plot_pdf),
            "video": str(video_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return DiscriminatorVizResult(
        output_dir=out_dir,
        scores_csv=scores_csv,
        plot_png=plot_png,
        plot_pdf=plot_pdf,
        video=video_path,
        summary_json=summary_path,
    )


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


__all__ = [
    "score_chunk_features",
    "write_discriminator_trace",
    "DiscriminatorVizResult",
    "select_camera",
    "write_rollout_video",
    "visualize_selected_trajectory_discriminator_nnpu",
]
