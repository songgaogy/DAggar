"""Per-frame BCE discriminator visualization for the Q/V vis pipeline.

The reusable LPB v2 scoring primitives (detector loader, BCE-benchmark
discriminator builder, Youden threshold helpers, trajectory wrapper) now
live in `robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer`. This
module is a thin visualization layer on top of those primitives — the
warmup reward path consumes the SAME `LPBV2OfflineScorer`, so the scorer
and the visualizer can no longer drift apart (the bug captured in
`pipeline/docs/IQL_DISCRIMINATOR_REWARD_DEBUG.md`).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import to_absolute_path
from PIL import Image, ImageDraw, ImageFont

from robosuite.discriminator.lpb_v2.visualization.visualize import (
    _draw_border,
    _load_font,
    _pad_to_even,
)
from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    LPBV2OfflineScorer,
    _build_bce_discriminator_from_ckpt,
    _build_selected_trajectory,
    _compute_youden_threshold_for_task,
    _load_failure_mask_and_segments,
    _resolve_task_for_detector,
    SafeRobosuiteBenchmarkTrajectory,
)
from robosuite.pipeline.common import Transition


__all__ = [
    "DiscriminatorVizResult",
    "visualize_selected_trajectory_discriminator",
    # Re-exported so existing callers do not break:
    "LPBV2OfflineScorer",
    "SafeRobosuiteBenchmarkTrajectory",
    "_build_bce_discriminator_from_ckpt",
    "_build_selected_trajectory",
    "_compute_youden_threshold_for_task",
    "_load_failure_mask_and_segments",
    "_resolve_task_for_detector",
]


@dataclass
class DiscriminatorVizResult:
    output_dir: Path
    scores_csv: Path
    plot_png: Path
    plot_pdf: Path
    video: Path
    summary_json: Path


def _safe_id(value: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(value))


def _write_scores_csv(
    path: Path,
    *,
    scores: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    gt_mask: np.ndarray | None,
    bce_logits: np.ndarray | None = None,
    intrinsic_rewards: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    use_training_cols = bce_logits is not None and intrinsic_rewards is not None
    if use_training_cols:
        fieldnames = [
            "step",
            "bce_logit",
            "intrinsic_reward",
            "threshold",
            "pred_failure",
            "gt_failure",
        ]
    else:
        fieldnames = ["step", "failure_score", "threshold", "pred_failure", "gt_failure"]
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(int(scores.shape[0])):
            if use_training_cols:
                writer.writerow(
                    {
                        "step": idx,
                        "bce_logit": float(bce_logits[idx]),
                        "intrinsic_reward": float(intrinsic_rewards[idx]),
                        "threshold": float(thresholds[idx]),
                        "pred_failure": int(predictions[idx]),
                        "gt_failure": "" if gt_mask is None else int(gt_mask[idx]),
                    }
                )
            else:
                writer.writerow(
                    {
                        "step": idx,
                        "failure_score": float(scores[idx]),
                        "threshold": float(thresholds[idx]),
                        "pred_failure": int(predictions[idx]),
                        "gt_failure": "" if gt_mask is None else int(gt_mask[idx]),
                    }
                )


def _plot_scores(
    path_base: Path,
    *,
    scores: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    gt_mask: np.ndarray | None,
    segments: list[dict[str, Any]],
    title: str,
    bce_logits: np.ndarray | None = None,
    intrinsic_rewards: np.ndarray | None = None,
) -> None:
    steps = np.arange(int(scores.shape[0]))
    use_training_cols = bce_logits is not None and intrinsic_rewards is not None

    if use_training_cols:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
        ax_logit, ax_reward = axes
        ax_logit.plot(steps, bce_logits, color="tab:blue", linewidth=1.4, label="BCE logit")
        if thresholds.size:
            ax_logit.plot(
                steps,
                thresholds,
                color="tab:red",
                linestyle="--",
                linewidth=1.1,
                label="threshold",
            )
        ax_logit.set_ylabel("BCE logit")
        ax_reward.plot(
            steps,
            intrinsic_rewards,
            color="tab:pink",
            linewidth=1.4,
            label="intrinsic r_disc",
        )
        ax_reward.axhline(-1.0, color="gray", linestyle=":", linewidth=1)
        ax_reward.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax_reward.set_ylabel("r_disc")
        ax_reward.set_xlabel("Step")
        plot_axes = [ax_logit, ax_reward]
    else:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.plot(steps, scores, color="tab:blue", linewidth=1.4, label="BCE failure score (-logit)")
        if thresholds.size:
            ax.plot(steps, thresholds, color="tab:red", linestyle="--", linewidth=1.1, label="threshold")
        ax.set_ylabel("Failure score")
        ax.set_xlabel("Step")
        plot_axes = [ax]

    for segment in segments:
        try:
            start = int(segment["start"])
            end = int(segment["end"])
        except Exception:
            continue
        for axis in plot_axes:
            axis.axvspan(max(0, start), min(int(scores.shape[0]) - 1, end), color="tab:red", alpha=0.12)

    if gt_mask is not None and bool(gt_mask.any()):
        ymin = float(np.nanmin(scores))
        gt_band = np.where(gt_mask.astype(bool), ymin, np.nan)
        plot_axes[0].plot(steps, gt_band, color="darkred", linewidth=5, alpha=0.35, label="GT failure")

    if bool(predictions.any()):
        if use_training_cols:
            ymin = float(np.nanmin(intrinsic_rewards))
            pred_band = np.where(predictions.astype(bool), ymin, np.nan)
            plot_axes[1].plot(
                steps, pred_band, color="tab:red", linewidth=3, alpha=0.75, label="predicted failure"
            )
        else:
            ymin = float(np.nanmin(scores))
            pred_band = np.where(predictions.astype(bool), ymin, np.nan)
            plot_axes[0].plot(
                steps, pred_band, color="tab:red", linewidth=3, alpha=0.75, label="predicted failure"
            )

    plot_axes[0].set_title(title, loc="left", fontsize=11)
    plot_axes[0].set_xlim(0, max(int(scores.shape[0]) - 1, 1))
    for axis in plot_axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=160)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _overlay_hud(
    image: np.ndarray,
    *,
    score: float,
    threshold: float,
    pred_failure: bool,
    gt_failure: bool | None,
    frame_idx: int,
    total: int,
    font: ImageFont.ImageFont,
    bce_logit: float | None = None,
    intrinsic_reward: float | None = None,
) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, mode="RGBA")
    width, _height = pil.size
    hud_height = 120 if intrinsic_reward is not None else 92
    draw.rectangle([(0, 0), (width, hud_height)], fill=(0, 0, 0, 140))
    gt_text = "" if gt_failure is None else f"   GT: {'FAIL' if gt_failure else 'OK'}"
    if intrinsic_reward is not None and bce_logit is not None:
        lines = [
            f"frame {frame_idx + 1}/{total}",
            f"logit={bce_logit:.3f}  tau={threshold:.3f}  r_disc={intrinsic_reward:.3f}",
            f"PRED: {'FAIL' if pred_failure else 'OK'}{gt_text}",
        ]
    else:
        lines = [
            f"frame {frame_idx + 1}/{total}",
            f"bce_score={score:.3f}  tau={threshold:.3f}",
            f"PRED: {'FAIL' if pred_failure else 'OK'}{gt_text}",
        ]
    colors = [
        (255, 255, 255, 255),
        (255, 255, 255, 255),
        (255, 80, 80, 255) if pred_failure else (80, 255, 80, 255),
    ]
    y = 4
    for text, color in zip(lines, colors):
        draw.text((8, y), text, fill=color, font=font)
        y += 28
    return np.asarray(pil)


def _write_video(
    path: Path,
    frames: np.ndarray,
    *,
    camera_name: str,
    scores: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    gt_mask: np.ndarray | None,
    fps: int,
    border_thickness: int,
    flip_vertical: bool,
    bce_logits: np.ndarray | None = None,
    intrinsic_rewards: np.ndarray | None = None,
) -> None:
    font = _load_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        format="ffmpeg",
        fps=int(fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        T = min(int(frames.shape[0]), int(scores.shape[0]))
        for idx in range(T):
            image = np.asarray(frames[idx], dtype=np.uint8)
            if flip_vertical:
                image = image[::-1, ...]
            pred_failure = bool(predictions[idx])
            if pred_failure:
                image = _draw_border(image, (255, 0, 0), int(border_thickness))
            gt_failure = None if gt_mask is None else bool(gt_mask[idx])
            canvas = _overlay_hud(
                image,
                score=float(scores[idx]),
                threshold=float(thresholds[idx]),
                pred_failure=pred_failure,
                gt_failure=gt_failure,
                frame_idx=idx,
                total=int(scores.shape[0]),
                font=font,
                bce_logit=None if bce_logits is None else float(bce_logits[idx]),
                intrinsic_reward=None if intrinsic_rewards is None else float(intrinsic_rewards[idx]),
            )
            writer.append_data(_pad_to_even(canvas).astype(np.uint8))


def visualize_selected_trajectory_training_disc(
    *,
    task_name: str,
    selected_hdf5_path: Path,
    selected_demo_key: str,
    transitions: list[Transition],
    camera_names: list[str],
    output_dir: Path,
    bce_logits: np.ndarray,
    intrinsic_rewards: np.ndarray,
    threshold: float,
    video_fps: int,
    camera_name: str | None = None,
    border_thickness: int = 10,
    flip_vertical: bool = True,
) -> DiscriminatorVizResult:
    """HUD + timeseries using OnlineBCEDiscriminator (same path as IQL vis Q/V)."""
    if not transitions:
        raise RuntimeError("Cannot visualize discriminator on an empty trajectory.")
    if int(bce_logits.shape[0]) != len(transitions):
        raise ValueError(
            f"bce_logits length {bce_logits.shape[0]} != transitions {len(transitions)}"
        )

    logits = np.asarray(bce_logits, dtype=np.float32).reshape(-1)
    intrinsic = np.asarray(intrinsic_rewards, dtype=np.float32).reshape(-1)
    threshold_f = float(threshold)
    thresholds = np.full_like(logits, threshold_f, dtype=np.float32)
    predictions = (logits >= threshold_f).astype(np.int64)
    gt_mask, segments = _load_failure_mask_and_segments(
        Path(selected_hdf5_path),
        str(selected_demo_key),
        length=int(logits.shape[0]),
    )
    if gt_mask is not None and gt_mask.shape[0] != logits.shape[0]:
        aligned = np.zeros((int(logits.shape[0]),), dtype=np.uint8)
        n_copy = min(int(gt_mask.shape[0]), int(logits.shape[0]))
        aligned[:n_copy] = gt_mask[:n_copy]
        gt_mask = aligned

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = out_dir / "bce_scores.csv"
    plot_base = out_dir / "discriminator_timeseries"
    selected_camera = str(
        camera_name or (camera_names[0] if camera_names else "agentview")
    )
    video_path = out_dir / f"rollout_discriminator_{_safe_id(selected_camera)}.mp4"
    summary_path = out_dir / "summary.json"

    _write_scores_csv(
        scores_csv,
        scores=logits,
        thresholds=thresholds,
        predictions=predictions,
        gt_mask=gt_mask,
        bce_logits=logits,
        intrinsic_rewards=intrinsic,
    )
    _plot_scores(
        plot_base,
        scores=logits,
        thresholds=thresholds,
        predictions=predictions,
        gt_mask=gt_mask,
        segments=segments,
        title=f"{task_name} {Path(selected_hdf5_path).name}::{selected_demo_key} (training BCE)",
        bce_logits=logits,
        intrinsic_rewards=intrinsic,
    )

    frames_list = []
    for transition in transitions:
        if selected_camera not in transition.obs:
            raise KeyError(
                f"Transition is missing camera '{selected_camera}' for discriminator video."
            )
        frame = np.asarray(transition.obs[selected_camera], dtype=np.uint8)
        frames_list.append(frame)
    frames = np.stack(frames_list, axis=0)
    _write_video(
        video_path,
        frames,
        camera_name=selected_camera,
        scores=logits,
        thresholds=thresholds,
        predictions=predictions,
        gt_mask=gt_mask,
        fps=int(video_fps),
        border_thickness=int(border_thickness),
        flip_vertical=bool(flip_vertical),
        bce_logits=logits,
        intrinsic_rewards=intrinsic,
    )

    first_pred = np.where(predictions.astype(bool))[0]
    first_gt = None if gt_mask is None or not bool(gt_mask.any()) else int(np.where(gt_mask.astype(bool))[0][0])
    summary = {
        "task_name": str(task_name),
        "selected_hdf5": str(selected_hdf5_path),
        "selected_demo_key": str(selected_demo_key),
        "num_frames": int(logits.shape[0]),
        "threshold": threshold_f,
        "scoring_path": "OnlineBCEDiscriminator.intrinsic_reward(encode_chunk_frames)",
        "bce_logit_min": float(np.min(logits)),
        "bce_logit_mean": float(np.mean(logits)),
        "bce_logit_max": float(np.max(logits)),
        "intrinsic_min": float(np.min(intrinsic)),
        "intrinsic_mean": float(np.mean(intrinsic)),
        "intrinsic_max": float(np.max(intrinsic)),
        "predicted_failure_frames": int(predictions.sum()),
        "first_pred_failure_frame": None if first_pred.size == 0 else int(first_pred[0]),
        "first_gt_failure_frame": first_gt,
        "camera_name": selected_camera,
        "outputs": {
            "scores_csv": str(scores_csv),
            "plot_png": str(plot_base.with_suffix(".png")),
            "plot_pdf": str(plot_base.with_suffix(".pdf")),
            "video": str(video_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return DiscriminatorVizResult(
        output_dir=out_dir,
        scores_csv=scores_csv,
        plot_png=plot_base.with_suffix(".png"),
        plot_pdf=plot_base.with_suffix(".pdf"),
        video=video_path,
        summary_json=summary_path,
    )


def visualize_selected_trajectory_discriminator(
    *,
    disc_ckpt: str,
    task_name: str,
    selected_hdf5_path: Path,
    selected_demo_key: str,
    transitions: list[Transition],
    camera_names: list[str],
    image_size: int,
    output_dir: Path,
    device: str,
    batch_size: int,
    video_fps: int,
    camera_name: str | None = None,
    border_thickness: int = 10,
    flip_vertical: bool = True,
    scorer: LPBV2OfflineScorer | None = None,
) -> DiscriminatorVizResult:
    """Render BCE-discriminator score timeseries + video HUD for one demo.

    If `scorer` is provided, it is reused (saves a redundant model build);
    otherwise a fresh `LPBV2OfflineScorer` is constructed from `disc_ckpt`.
    Either way, the scoring path is identical to the warmup reward path.
    """
    if not transitions:
        raise RuntimeError("Cannot visualize discriminator on an empty trajectory.")

    disc_ckpt_path = Path(to_absolute_path(str(disc_ckpt))).resolve()
    if scorer is None:
        scorer = LPBV2OfflineScorer(
            bce_ckpt_path=disc_ckpt_path,
            task_name=str(task_name),
            device=str(device),
            batch_size=max(1, int(batch_size)),
        )
    detector_task = scorer.detector_task
    trajectory = _build_selected_trajectory(
        hdf5_path=Path(selected_hdf5_path),
        demo_key=str(selected_demo_key),
        task_name=detector_task,
        fps=int(video_fps),
    )
    scored = scorer._discriminator.score_trajectory(trajectory)  # noqa: SLF001
    scores = np.asarray(scored.step_scores, dtype=np.float32)
    threshold = float(scorer.tau)
    threshold_source = str(scorer.tau_source)
    checkpoint_threshold = float(scorer.checkpoint_detector_threshold)
    thresholds = np.full_like(scores, float(threshold), dtype=np.float32)
    predictions = (scores >= float(threshold)).astype(np.int64)
    gt_mask, segments = _load_failure_mask_and_segments(
        Path(selected_hdf5_path),
        str(selected_demo_key),
        length=int(scores.shape[0]),
    )
    if gt_mask is not None and gt_mask.shape[0] != scores.shape[0]:
        aligned = np.zeros((int(scores.shape[0]),), dtype=np.uint8)
        n_copy = min(int(gt_mask.shape[0]), int(scores.shape[0]))
        aligned[:n_copy] = gt_mask[:n_copy]
        gt_mask = aligned

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = out_dir / "bce_scores.csv"
    plot_base = out_dir / "discriminator_timeseries"
    selected_camera = str(camera_name or ("agentview" if "agentview" in trajectory.available_cameras else trajectory.available_cameras[0]))
    if selected_camera not in trajectory.available_cameras:
        raise KeyError(
            f"Discriminator visualization camera {selected_camera!r} is not available in "
            f"{selected_hdf5_path}::{selected_demo_key}. Available: {trajectory.available_cameras}."
        )
    video_path = out_dir / f"rollout_discriminator_{_safe_id(selected_camera)}.mp4"
    summary_path = out_dir / "summary.json"

    _write_scores_csv(
        scores_csv,
        scores=scores,
        thresholds=thresholds,
        predictions=predictions,
        gt_mask=gt_mask,
    )
    _plot_scores(
        plot_base,
        scores=scores,
        thresholds=thresholds,
        predictions=predictions,
        gt_mask=gt_mask,
        segments=segments,
        title=f"{task_name} {Path(selected_hdf5_path).name}::{selected_demo_key}",
    )
    frames = trajectory.load_images(cameras=[selected_camera])[selected_camera]
    _write_video(
        video_path,
        frames,
        camera_name=selected_camera,
        scores=scores,
        thresholds=thresholds,
        predictions=predictions,
        gt_mask=gt_mask,
        fps=int(video_fps),
        border_thickness=int(border_thickness),
        flip_vertical=bool(flip_vertical),
    )

    first_pred = np.where(predictions.astype(bool))[0]
    first_gt = None if gt_mask is None or not bool(gt_mask.any()) else int(np.where(gt_mask.astype(bool))[0][0])
    summary = {
        "disc_ckpt": str(disc_ckpt_path),
        "model_ckpt": str(scorer._payload.get("model_ckpt", "")),  # noqa: SLF001
        "task_name": str(task_name),
        "detector_task": str(detector_task),
        "threshold": float(threshold),
        "checkpoint_detector_threshold": checkpoint_threshold,
        "threshold_source": str(threshold_source),
        "scoring_path": "LPBV2OfflineScorer.score_hdf5_demo(BCEBenchmarkDiscriminator.score_trajectory)",
        "feature_source": str(scored.aux.get("feature_source", "")),
        "transformer_layer": int(scored.aux.get("transformer_layer", -1)),
        "view_names": list(scored.aux.get("view_names", [])),
        "available_cameras": list(trajectory.available_cameras),
        "selected_hdf5": str(selected_hdf5_path),
        "selected_demo_key": str(selected_demo_key),
        "num_frames": int(scores.shape[0]),
        "raw_video_frame_shape": list(np.asarray(frames[0]).shape) if int(frames.shape[0]) > 0 else [],
        "predicted_failure_frames": int(predictions.sum()),
        "first_pred_failure_frame": None if first_pred.size == 0 else int(first_pred[0]),
        "first_gt_failure_frame": first_gt,
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "camera_name": selected_camera,
        "outputs": {
            "scores_csv": str(scores_csv),
            "plot_png": str(plot_base.with_suffix(".png")),
            "plot_pdf": str(plot_base.with_suffix(".pdf")),
            "video": str(video_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return DiscriminatorVizResult(
        output_dir=out_dir,
        scores_csv=scores_csv,
        plot_png=plot_base.with_suffix(".png"),
        plot_pdf=plot_base.with_suffix(".pdf"),
        video=video_path,
        summary_json=summary_path,
    )
