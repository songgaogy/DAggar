"""Failure-detector visualization for the LPB KNN discriminator.

Usage (from repo root):
    python -m robosuite.discriminator.lpb.visualize \
        --lpb-ckpt /abs/path/checkpoints/lpb/dynamics/dynamics_model.pt \
        --fail-root /abs/path/data/utils/fail_rollout \
        --success-root /abs/path/data/utils/success_rollout \
        --task PickPlaceBread \
        --num-trajs 4 \
        --out-dir /tmp/lpb_viz

Outputs:
    <out_dir>/videos/<video_id>.mp4  — H.264 / yuv420p, VS Code friendly
    <out_dir>/lpb_scores.pdf         — multi-page score plots
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from data.utils.benchmark import BenchmarkTrajectory, FailureBenchmark

from .lpb_benchmark import LPBBenchmarkDiscriminator


# ---------------------------------------------------------------------- #
# Rendering helpers                                                      #
# ---------------------------------------------------------------------- #


def _pad_to_even(img: np.ndarray) -> np.ndarray:
    """libx264 prefers even H/W. Pad by 1 bottom/right when needed."""
    h, w = int(img.shape[0]), int(img.shape[1])
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _draw_border(img: np.ndarray, color: tuple[int, int, int], thickness: int) -> np.ndarray:
    out = img.copy()
    t = int(thickness)
    out[:t, :, :] = color
    out[-t:, :, :] = color
    out[:, :t, :] = color
    out[:, -t:, :] = color
    return out


def _load_font() -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, 18)
            except Exception:
                continue
    return ImageFont.load_default()


def _overlay_hud(
    img: np.ndarray,
    *,
    score: float,
    threshold: float,
    pred_fail: bool,
    gt_fail: Optional[bool],
    frame_idx: int,
    total: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil, mode="RGBA")

    w, _h = pil.size
    box_h = 92
    draw.rectangle([(0, 0), (w, box_h)], fill=(0, 0, 0, 140))

    lines = [
        f"frame {frame_idx + 1}/{total}",
        f"lambda={score:.3f}  tau={threshold:.3f}",
        f"PRED: {'FAIL' if pred_fail else 'OK  '}   "
        + (f"GT: {'FAIL' if gt_fail else 'OK  '}" if gt_fail is not None else ""),
    ]
    colors = [
        (255, 255, 255, 255),
        (255, 255, 255, 255),
        (255, 80, 80, 255) if pred_fail else (80, 255, 80, 255),
    ]
    y = 4
    for text, color in zip(lines, colors):
        draw.text((8, y), text, fill=color, font=font)
        y += 28

    return np.asarray(pil)


# ---------------------------------------------------------------------- #
# Data containers                                                        #
# ---------------------------------------------------------------------- #


@dataclass
class PerTrajectoryViz:
    video_id: str
    task_name: str
    num_frames: int
    step_scores: np.ndarray          # (T,) aggregated lambda per frame
    predictions: np.ndarray          # (T,) {0,1}
    gt_mask: Optional[np.ndarray]    # (T,) uint8 or None
    failure_segments: list[dict]
    first_gt_failure_frame: Optional[int]
    first_pred_failure_frame: Optional[int]


# ---------------------------------------------------------------------- #
# Main class                                                             #
# ---------------------------------------------------------------------- #


class LPBVisualizer:
    """Combine a fitted LPBBenchmarkDiscriminator with video + PDF renderers."""

    def __init__(
        self,
        discriminator: LPBBenchmarkDiscriminator,
        *,
        camera_name: str = "agentview",
        fps: int = 20,
        border_thickness: int = 10,
        border_color_fail: tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        self.discriminator = discriminator
        self.camera_name = str(camera_name)
        self.fps = int(fps)
        self.border_thickness = int(border_thickness)
        self.border_color_fail = tuple(int(c) for c in border_color_fail)
        self._font = _load_font()

    def _task_threshold(self, task: str) -> float:
        detector = self.discriminator._detectors_per_task.get(task, None)
        if detector is None or detector.threshold is None:
            return float("nan")
        return float(detector.threshold)

    # ------------------------------------------------------------------ #
    # Per-trajectory scoring                                             #
    # ------------------------------------------------------------------ #

    def _score_trajectory(self, traj: BenchmarkTrajectory) -> PerTrajectoryViz:
        out = self.discriminator.score_trajectory(traj)
        gt_mask = traj.load_failure_mask()
        first_gt = traj.first_gt_failure_frame()
        return PerTrajectoryViz(
            video_id=str(traj.video_id),
            task_name=str(traj.task_name),
            num_frames=int(traj.num_frames),
            step_scores=np.asarray(out.step_scores, dtype=np.float32),
            predictions=np.asarray(out.predictions, dtype=np.int64),
            gt_mask=None if gt_mask is None else np.asarray(gt_mask, dtype=np.uint8),
            failure_segments=list(traj.failure_segments),
            first_gt_failure_frame=first_gt,
            first_pred_failure_frame=out.first_failure_frame,
        )

    # ------------------------------------------------------------------ #
    # Video writer                                                       #
    # ------------------------------------------------------------------ #

    def render_video(
        self,
        traj: BenchmarkTrajectory,
        viz: PerTrajectoryViz,
        out_path: str,
    ) -> None:
        threshold = self._task_threshold(viz.task_name)
        images_by_cam = traj.load_images(cameras=[self.camera_name])
        frames = np.asarray(images_by_cam[self.camera_name], dtype=np.uint8)
        # robosuite off-screen renders are vertically flipped; flip for display.
        frames = frames[:, ::-1, :, :]
        T = min(int(frames.shape[0]), int(viz.num_frames))
        if T <= 0:
            raise RuntimeError(f"Empty frames for {viz.video_id}")

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        writer = imageio.get_writer(
            out_path,
            format="ffmpeg",
            fps=self.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        )
        try:
            for t in range(T):
                img = frames[t]
                pred_fail = bool(viz.predictions[t])
                gt_fail = None if viz.gt_mask is None else bool(viz.gt_mask[t])

                canvas = img
                if pred_fail:
                    canvas = _draw_border(canvas, self.border_color_fail, self.border_thickness)

                canvas = _overlay_hud(
                    canvas,
                    score=float(viz.step_scores[t]),
                    threshold=float(threshold),
                    pred_fail=pred_fail,
                    gt_fail=gt_fail,
                    frame_idx=t,
                    total=T,
                    font=self._font,
                )
                canvas = _pad_to_even(canvas)
                writer.append_data(canvas.astype(np.uint8))
        finally:
            writer.close()

    # ------------------------------------------------------------------ #
    # PDF writer                                                         #
    # ------------------------------------------------------------------ #

    def render_pdf(
        self,
        vizs: list[PerTrajectoryViz],
        out_path: str,
    ) -> None:
        if not vizs:
            raise RuntimeError("No trajectories to plot.")
        task = vizs[0].task_name
        threshold = self._task_threshold(task)
        summary = self.discriminator.calibration_summary()
        per_task = summary.get("per_task", {}).get(task, {})

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        with PdfPages(out_path) as pdf:
            # ---- Summary page ---- #
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.axis("off")
            ax.set_title(f"LPB KNN summary — task={task}", fontsize=14, loc="left")

            def _g(key, default="n/a"):
                val = per_task.get(key, default)
                return val

            lines = [
                f"discriminator: {self.discriminator.name}",
                f"checkpoint: {summary.get('checkpoint_path', 'n/a')}",
                f"action_horizon: {summary.get('action_horizon', 'n/a')}  "
                f"lambda_mode: {summary.get('lambda_mode', 'n/a')}  "
                f"lambda_window_size: {summary.get('lambda_window_size', 'n/a')}",
                f"delta (FA budget %): {summary.get('delta', 'n/a')}  "
                f"use_transition_error: {summary.get('use_transition_error', False)}  "
                f"transition_aux_weight: {summary.get('transition_aux_weight', 0.0)}",
                f"threshold tau: {threshold:.4f}",
                "",
                "Calibration (success pool, per task):",
                f"  num_success_trajectories = {_g('num_success_trajectories')}",
                f"  num_bank_trajectories    = {_g('num_bank_trajectories')}  "
                f"  num_calib_trajectories = {_g('num_calib_trajectories')}",
                f"  num_bank_steps           = {_g('num_bank_steps')}  "
                f"  num_calib_steps        = {_g('num_calib_steps')}",
                f"  threshold_init           = {_g('threshold_init')}",
                "",
                f"Sampled failure trajectories: {len(vizs)}",
                "",
                "Trigger rule: flag when aggregated lambda_t (= KNN min sq-dist to"
                " expert bank, aggregated) exceeds tau.",
            ]
            ax.text(0.01, 0.95, "\n".join(lines), fontsize=11, family="monospace", va="top", ha="left")
            pdf.savefig(fig)
            plt.close(fig)

            # ---- One page per trajectory ---- #
            for viz in vizs:
                self._plot_trajectory(pdf, viz, threshold=threshold)

    def _plot_trajectory(self, pdf: PdfPages, viz: PerTrajectoryViz, *, threshold: float) -> None:
        T = int(viz.num_frames)
        t = np.arange(T)

        fig, ax = plt.subplots(figsize=(10.0, 4.5))
        ax.plot(t, viz.step_scores, color="#1f77b4", lw=1.4, label="lambda (aggregate)")
        if np.isfinite(threshold):
            ax.axhline(threshold, color="red", lw=1.2, ls="--", label=f"tau={threshold:.3f}")

        # Shade GT failure segments.
        for seg in viz.failure_segments:
            try:
                s = int(seg["start"])
                e = int(seg["end"])
            except Exception:
                continue
            ax.axvspan(max(0, s), min(T - 1, e), color="red", alpha=0.12, zorder=0)

        # Predicted-positive frames band.
        pred_mask = viz.predictions.astype(bool)
        if pred_mask.any():
            ymin = float(viz.step_scores.min())
            band = np.where(pred_mask, ymin, np.nan)
            ax.plot(t, band, color="red", lw=4, alpha=0.6, label="predicted fail frames")

        # First-failure markers.
        if viz.first_gt_failure_frame is not None:
            ax.axvline(int(viz.first_gt_failure_frame), color="darkred", lw=0.9, ls=":", label="first GT fail")
        if viz.first_pred_failure_frame is not None:
            ax.axvline(int(viz.first_pred_failure_frame), color="orange", lw=0.9, ls=":", label="first PRED fail")

        ax.set_xlim(0, max(T - 1, 1))
        ax.set_xlabel("frame")
        ax.set_ylabel("lambda (KNN aggregate)")
        ax.set_title(f"[{viz.task_name}] {viz.video_id}  (T={T})", fontsize=11, loc="left")
        ax.grid(True, alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        uniq_h.append(Patch(facecolor="red", alpha=0.12, label="GT failure segment"))
        uniq_l.append("GT failure segment")
        ax.legend(uniq_h, uniq_l, loc="upper left", fontsize=8, framealpha=0.85)

        n_pred = int(viz.predictions.sum())
        n_gt = int(viz.gt_mask.sum()) if viz.gt_mask is not None else -1
        delay_txt = "n/a"
        if viz.first_gt_failure_frame is not None and viz.first_pred_failure_frame is not None:
            delay_txt = f"{int(viz.first_pred_failure_frame) - int(viz.first_gt_failure_frame):+d}"
        stats = (
            f"pred_frames={n_pred}  gt_frames={n_gt}  "
            f"first_pred={viz.first_pred_failure_frame}  first_gt={viz.first_gt_failure_frame}  "
            f"delay(pred-gt)={delay_txt}"
        )
        ax.text(0.01, -0.23, stats, transform=ax.transAxes, fontsize=9, family="monospace")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Top-level entry                                                    #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        fail_trajectories: list[BenchmarkTrajectory],
        *,
        out_dir: str,
        pdf_name: str = "lpb_scores.pdf",
    ) -> dict:
        if not fail_trajectories:
            raise RuntimeError("No failure trajectories provided for visualization.")

        videos_dir = os.path.join(out_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        vizs: list[PerTrajectoryViz] = []
        video_paths: list[str] = []
        for traj in fail_trajectories:
            viz = self._score_trajectory(traj)
            safe_id = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(traj.video_id))
            video_path = os.path.join(videos_dir, f"{safe_id}.mp4")
            self.render_video(traj, viz, video_path)
            print(
                f"[viz] {traj.task_name}/{traj.video_id}  T={viz.num_frames}  "
                f"pred_frames={int(viz.predictions.sum())}  -> {video_path}"
            )
            vizs.append(viz)
            video_paths.append(video_path)

        pdf_path = os.path.join(out_dir, pdf_name)
        self.render_pdf(vizs, pdf_path)
        print(f"[viz] wrote PDF -> {pdf_path}")

        return {"videos": video_paths, "pdf": pdf_path}


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpb-ckpt", required=True, help="Path to LPB dynamics checkpoint (.pt)")
    parser.add_argument("--fail-root", required=True, help="data/utils/fail_rollout")
    parser.add_argument("--success-root", required=True, help="data/utils/success_rollout")
    parser.add_argument("--task", required=True, help="Single task name, e.g. PickPlaceBread")
    parser.add_argument("--num-trajs", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", type=str, default="lpb_scores.pdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)

    # Feature extractor.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--action-horizon", type=int, default=-1)
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--no-normalize-feature", action="store_true")
    parser.add_argument("--use-transition-error", action="store_true")
    parser.add_argument("--transition-proprio-error-weight", type=float, default=0.1)

    # Detector.
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--delta-step", type=float, default=1.0)
    parser.add_argument("--knn-chunk-size", type=int, default=8192)
    parser.add_argument("--lambda-mode", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--lambda-window-size", type=int, default=-1)
    parser.add_argument("--transition-aux-weight", type=float, default=0.0)

    # Calibration / misc.
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--quiet-fit", action="store_true")

    # Data caps.
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=[str(args.task)],
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    trajs = bench.trajectories()
    if not trajs:
        raise RuntimeError(f"No trajectories discovered for task {args.task!r}.")

    fail_trajs = [t for t in trajs if bool(t.is_failure)]
    succ_trajs = [t for t in trajs if not bool(t.is_failure)]
    if not fail_trajs:
        raise RuntimeError(f"No failure trajectories for task {args.task!r}.")
    if not succ_trajs:
        raise RuntimeError(f"No success trajectories for task {args.task!r} (needed for fit).")

    rng = random.Random(int(args.seed))
    n = min(int(args.num_trajs), len(fail_trajs))
    sampled = rng.sample(fail_trajs, n)
    print(f"[viz] sampled {n}/{len(fail_trajs)} failure trajectories from {args.task}")

    discriminator = LPBBenchmarkDiscriminator(
        checkpoint_path=str(args.lpb_ckpt),
        device=str(args.device),
        feature_batch_size=int(args.feature_batch_size),
        action_horizon=int(args.action_horizon),
        camera_name=str(args.camera_name),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        normalize_feature=not bool(args.no_normalize_feature),
        use_transition_error=bool(args.use_transition_error),
        transition_proprio_error_weight=float(args.transition_proprio_error_weight),
        delta=float(args.delta),
        delta_step=float(args.delta_step),
        knn_chunk_size=int(args.knn_chunk_size),
        lambda_mode=str(args.lambda_mode),
        lambda_window_size=int(args.lambda_window_size),
        transition_aux_weight=float(args.transition_aux_weight),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        discriminator.fit_on_benchmark(trajs)
        visualizer = LPBVisualizer(
            discriminator,
            camera_name=str(args.camera_name),
            fps=int(args.fps),
            border_thickness=int(args.border_thickness),
        )
        out_paths = visualizer.visualize(
            sampled,
            out_dir=str(args.out_dir),
            pdf_name=str(args.pdf_name),
        )
        print(f"[viz] done. videos: {len(out_paths['videos'])}  pdf: {out_paths['pdf']}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
