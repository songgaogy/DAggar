"""Failure-detector visualization for logpZO.

Usage (from repo root):
    python -m robosuite.discriminator.logpZO.visualize \
        --fail-root /abs/path/data/utils/fail_rollout \
        --success-root /abs/path/data/utils/success_rollout \
        --task PickPlaceBread \
        --num-trajs 4 \
        --out-dir /tmp/logpZO_viz

Outputs:
    <out_dir>/videos/<video_id>.mp4  — H.264 / yuv420p, VS Code friendly
    <out_dir>/logpZO_scores.pdf      — multi-page score plots
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

from .logpZO_benchmark import LogpZOBenchmarkDiscriminator


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
    """Stamp a solid-color border `thickness` px wide in-place (copy-safe)."""
    out = img.copy()
    t = int(thickness)
    out[:t, :, :] = color
    out[-t:, :, :] = color
    out[:, :t, :] = color
    out[:, -t:, :] = color
    return out


def _load_font() -> ImageFont.ImageFont:
    """Prefer a truetype font when available; fall back to PIL bitmap."""
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
    """Draw a translucent HUD with per-frame score / threshold / pred / gt."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil, mode="RGBA")

    # Semi-transparent background box.
    w, h = pil.size
    box_h = 92
    draw.rectangle([(0, 0), (w, box_h)], fill=(0, 0, 0, 140))

    lines = [
        f"frame {frame_idx + 1}/{total}",
        f"score={score:.2f}  tau={threshold:.2f}",
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
    step_scores: np.ndarray          # (T,)
    predictions: np.ndarray          # (T,) {0,1}
    gt_mask: Optional[np.ndarray]    # (T,) uint8 or None
    failure_segments: list[dict]
    first_gt_failure_frame: Optional[int]
    first_pred_failure_frame: Optional[int]


# ---------------------------------------------------------------------- #
# Main class                                                             #
# ---------------------------------------------------------------------- #


class LogpZOVisualizer:
    """Combine a trained LogpZO discriminator with video + PDF renderers."""

    def __init__(
        self,
        discriminator: LogpZOBenchmarkDiscriminator,
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
        """Write an H.264/yuv420p mp4 with per-frame HUD + red border on fail."""
        threshold = float(self.discriminator._thresholds_per_task[viz.task_name])
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
        """One summary page + one score page per trajectory."""
        if not vizs:
            raise RuntimeError("No trajectories to plot.")
        task = vizs[0].task_name
        threshold = float(self.discriminator._thresholds_per_task[task])
        calib_summary = self.discriminator.calibration_summary()["per_task"].get(task, {})

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        with PdfPages(out_path) as pdf:
            # ---- Summary page ---- #
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.axis("off")
            ax.set_title(f"logpZO summary — task={task}", fontsize=14, loc="left")

            lines = [
                f"discriminator: {self.discriminator.name}",
                f"encoder: {calib_summary.get('encoder', 'n/a')}  (dim={calib_summary.get('embedding_dim', 'n/a')})",
                f"alpha (CP significance): {calib_summary.get('alpha', 'n/a')}",
                f"threshold tau: {threshold:.4f}",
                "",
                "Calibration (success-pool step scores):",
                f"  num frames  = {calib_summary.get('num_calibration', 'n/a')}",
                f"  score mean  = {calib_summary.get('calib_score_mean', float('nan')):.3f}",
                f"  score std   = {calib_summary.get('calib_score_std', float('nan')):.3f}",
                f"  score range = [{calib_summary.get('calib_score_min', float('nan')):.3f}, "
                f"{calib_summary.get('calib_score_max', float('nan')):.3f}]",
                "",
                f"Sampled failure trajectories: {len(vizs)}",
                "",
                "Trigger rule: step score = -log p_Z(f^(-1)(s));  flag when score > tau.",
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
        ax.plot(t, viz.step_scores, color="#1f77b4", lw=1.4, label="step score")
        ax.axhline(threshold, color="red", lw=1.2, ls="--", label=f"threshold={threshold:.2f}")

        # Shade GT failure segments.
        for seg in viz.failure_segments:
            try:
                s = int(seg["start"])
                e = int(seg["end"])
            except Exception:
                continue
            ax.axvspan(max(0, s), min(T - 1, e), color="red", alpha=0.12, zorder=0)

        # Mark predicted-positive frames with a red step trace along the bottom.
        pred_mask = viz.predictions.astype(bool)
        if pred_mask.any():
            ymin = float(viz.step_scores.min())
            band = np.where(pred_mask, ymin, np.nan)
            ax.plot(t, band, color="red", lw=4, alpha=0.6, label="predicted fail frames")

        # First frames markers.
        if viz.first_gt_failure_frame is not None:
            ax.axvline(int(viz.first_gt_failure_frame), color="darkred", lw=0.9, ls=":", label="first GT fail")
        if viz.first_pred_failure_frame is not None:
            ax.axvline(int(viz.first_pred_failure_frame), color="orange", lw=0.9, ls=":", label="first PRED fail")

        ax.set_xlim(0, max(T - 1, 1))
        ax.set_xlabel("frame")
        ax.set_ylabel("score  (= -log p_Z(z))")
        ax.set_title(f"[{viz.task_name}] {viz.video_id}  (T={T})", fontsize=11, loc="left")
        ax.grid(True, alpha=0.2)

        # Legend (dedup).
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
        pdf_name: str = "logpZO_scores.pdf",
    ) -> dict:
        """Render videos + aggregate PDF. Returns a dict of written paths."""
        if not fail_trajectories:
            raise RuntimeError("No failure trajectories provided for visualization.")

        videos_dir = os.path.join(out_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        vizs: list[PerTrajectoryViz] = []
        video_paths: list[str] = []
        for traj in fail_trajectories:
            viz = self._score_trajectory(traj)
            # Sanitize filename.
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
    parser.add_argument("--fail-root", required=True, help="data/utils/fail_rollout")
    parser.add_argument("--success-root", required=True, help="data/utils/success_rollout")
    parser.add_argument("--task", required=True, help="Single task name, e.g. PickPlaceBread")
    parser.add_argument("--num-trajs", type=int, default=4, help="# failure trajectories to render")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", type=str, default="logpZO_scores.pdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)
    # Encoder
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="agentview")
    # Flow / training
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--scale-clamp", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calib-fraction", type=float, default=0.2,
                        help="Fraction of success trajectories held out for CP calibration.")
    parser.add_argument("--quiet-fit", action="store_true")
    # Data caps
    parser.add_argument("--max-fail-per-task", type=int, default=None,
                        help="Cap fail trajectories discovered; visualization samples from this pool")
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
        raise RuntimeError(f"No success trajectories for task {args.task!r} (need them for fit).")

    # Sample fail trajectories.
    rng = random.Random(int(args.seed))
    n = min(int(args.num_trajs), len(fail_trajs))
    sampled = rng.sample(fail_trajs, n)
    print(f"[viz] sampled {n}/{len(fail_trajs)} failure trajectories from {args.task}")

    discriminator = LogpZOBenchmarkDiscriminator(
        device=args.device,
        image_size=int(args.image_size),
        encoder_batch_size=int(args.encoder_batch_size),
        camera_name=str(args.camera_name),
        num_layers=int(args.num_layers),
        hidden_dim=int(args.hidden_dim),
        scale_clamp=float(args.scale_clamp),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        early_stop_patience=int(args.early_stop_patience),
        alpha=float(args.alpha),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        discriminator.fit_on_benchmark(trajs)
        visualizer = LogpZOVisualizer(
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
