"""Failure-detector visualization for the LPB v2 KNN discriminator.

Usage (from repo root):
    python -m robosuite.discriminator.dyn_disc.visualization.visualize \
        --model-ckpt /abs/path/checkpoints/dyn_disc/dynamics/<run>/checkpoints/model_49.pth \
        --data-root /abs/path/data \
        --task PickPlaceBread \
        --num-trajs 4 \
        --out-dir /tmp/dyn_disc_viz

Outputs:
    <out_dir>/videos/<video_id>.mp4
    <out_dir>/dyn_disc_scores.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lpb-v2"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from benchmark.core import BenchmarkTrajectory
from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark

from robosuite.discriminator.dyn_disc.adapters.single_bank import LPBV2BenchmarkDiscriminator


def _percentile_summary(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "empty"
    qs = np.percentile(arr, [0, 50, 90, 95, 99, 100])
    return (
        f"min={qs[0]:.3f} p50={qs[1]:.3f} p90={qs[2]:.3f} "
        f"p95={qs[3]:.3f} p99={qs[4]:.3f} max={qs[5]:.3f}"
    )


# ---------------------------------------------------------------------- #
# Rendering helpers                                                      #
# ---------------------------------------------------------------------- #


def _pad_to_even(img: np.ndarray) -> np.ndarray:
    """Pad bottom/right when needed because libx264 prefers even H/W."""
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
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, 18)
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
        f"knn_dist={score:.3f}  tau={threshold:.3f}",
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


def _parse_camera_to_view(s: Optional[str]) -> Optional[dict[str, str]]:
    if not s:
        return None
    out: dict[str, str] = {}
    for chunk in s.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def _compute_success_percentile_thresholds(
    discriminator: LPBV2BenchmarkDiscriminator,
    trajectories: list[BenchmarkTrajectory],
    percentile: float,
) -> dict[str, float]:
    task_to_scores: dict[str, list[np.ndarray]] = {}
    for traj in trajectories:
        if bool(traj.is_failure):
            continue
        out = discriminator.score_trajectory(traj)
        task_to_scores.setdefault(str(traj.task_name), []).append(
            np.asarray(out.step_scores, dtype=np.float64).reshape(-1)
        )

    thresholds: dict[str, float] = {}
    for task, seqs in task_to_scores.items():
        pooled = np.concatenate(seqs) if seqs else np.zeros((0,), dtype=np.float64)
        if pooled.size == 0:
            continue
        thresholds[task] = float(np.percentile(pooled, float(percentile)))
        print(
            f"[dyn_disc][viz][debug] success scores task={task}: "
            f"{_percentile_summary(pooled)}  p{float(percentile):.1f}={thresholds[task]:.4f}",
            flush=True,
        )
    return thresholds


def _load_benchmark_traj_best_f1_thresholds(path: str) -> dict[str, float]:
    with open(path, "r") as fp:
        data = json.load(fp)
    per_task = data.get("trajectory_level_per_task", {})
    thresholds: dict[str, float] = {}
    for task, metrics in per_task.items():
        if "best_f1_threshold" not in metrics:
            continue
        thresholds[str(task)] = float(metrics["best_f1_threshold"])
    if not thresholds and "best_f1_threshold" in data.get("trajectory_level", {}):
        thresholds["__global__"] = float(data["trajectory_level"]["best_f1_threshold"])
    if not thresholds:
        raise ValueError(f"No best_f1_threshold found in benchmark JSON: {path}")
    print(
        f"[dyn_disc][viz][debug] loaded benchmark trajectory best-F1 thresholds: {thresholds}",
        flush=True,
    )
    return thresholds


# ---------------------------------------------------------------------- #
# Data containers                                                        #
# ---------------------------------------------------------------------- #


@dataclass
class PerTrajectoryViz:
    video_id: str
    task_name: str
    num_frames: int
    step_scores: np.ndarray
    thresholds: np.ndarray
    predictions: np.ndarray
    gt_mask: Optional[np.ndarray]
    failure_segments: list[dict]
    first_gt_failure_frame: Optional[int]
    first_pred_failure_frame: Optional[int]


# ---------------------------------------------------------------------- #
# Main class                                                             #
# ---------------------------------------------------------------------- #


class LPBV2Visualizer:
    """Combine a fitted LPB v2 discriminator with video and PDF renderers."""

    def __init__(
        self,
        discriminator: LPBV2BenchmarkDiscriminator,
        *,
        camera_name: str = "agentview",
        fps: int = 20,
        border_thickness: int = 10,
        border_color_fail: tuple[int, int, int] = (255, 0, 0),
        threshold_overrides: Optional[dict[str, float]] = None,
        threshold_source: str = "detector",
        debug_score_stats: bool = True,
    ) -> None:
        self.discriminator = discriminator
        self.camera_name = str(camera_name)
        self.fps = int(fps)
        self.border_thickness = int(border_thickness)
        self.border_color_fail = tuple(int(c) for c in border_color_fail)
        self.threshold_overrides = dict(threshold_overrides) if threshold_overrides else {}
        self.threshold_source = str(threshold_source)
        self.debug_score_stats = bool(debug_score_stats)
        self._font = _load_font()

    def _task_threshold(self, task: str) -> float:
        if task in self.threshold_overrides:
            return float(self.threshold_overrides[task])
        if "__global__" in self.threshold_overrides:
            return float(self.threshold_overrides["__global__"])
        detector = self.discriminator._detectors_per_task.get(task, None)
        if detector is None or detector.threshold is None:
            return float("nan")
        return float(detector.threshold)

    def _detector_threshold(self, task: str) -> float:
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
        thresholds = out.aux.get("thresholds", None)
        if thresholds is None:
            thresholds = np.full_like(out.step_scores, out.aux.get("threshold", float("nan")))
        threshold = self._task_threshold(str(traj.task_name))
        if np.isfinite(threshold):
            predictions = (np.asarray(out.step_scores, dtype=np.float32) >= threshold).astype(np.int64)
        else:
            predictions = np.asarray(out.predictions, dtype=np.int64)
        positive = np.where(predictions == 1)[0]
        first_pred = int(positive[0]) if positive.size > 0 else None

        return PerTrajectoryViz(
            video_id=str(traj.video_id),
            task_name=str(traj.task_name),
            num_frames=int(traj.num_frames),
            step_scores=np.asarray(out.step_scores, dtype=np.float32),
            thresholds=np.asarray(thresholds, dtype=np.float32),
            predictions=predictions,
            gt_mask=None if gt_mask is None else np.asarray(gt_mask, dtype=np.uint8),
            failure_segments=list(traj.failure_segments),
            first_gt_failure_frame=first_gt,
            first_pred_failure_frame=first_pred,
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
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.axis("off")
            ax.set_title(f"LPB v2 KNN summary - task={task}", fontsize=14, loc="left")

            def _g(key: str, default: str = "n/a"):
                return per_task.get(key, default)

            lines = [
                f"discriminator: {self.discriminator.name}",
                f"model_ckpt: {summary.get('model_ckpt', 'n/a')}",
                f"view_names: {summary.get('view_names', 'n/a')}",
                f"camera_to_view: {summary.get('camera_to_view', 'n/a')}",
                f"visual_weight: {summary.get('visual_weight', 'n/a')}  "
                f"proprio_weight: {summary.get('proprio_weight', 'n/a')}",
                f"delta (FA budget %): {summary.get('delta', 'n/a')}  "
                f"calib_fraction: {summary.get('calib_fraction', 'n/a')}",
                f"encode_batch_size: {summary.get('encode_batch_size', 'n/a')}  "
                f"knn_chunk_size: {summary.get('knn_chunk_size', 'n/a')}",
                f"visualized threshold source: {self.threshold_source}",
                f"visualized threshold tau: {threshold:.4f}",
                f"detector threshold tau: {self._detector_threshold(task):.4f}",
                "",
                "Calibration (success pool, per task):",
                f"  num_success_trajectories = {_g('num_success_trajectories')}",
                f"  num_bank_trajectories    = {_g('num_bank_trajectories')}  "
                f"num_calib_trajectories = {_g('num_calib_trajectories')}",
                f"  num_bank_steps           = {_g('num_bank_steps')}  "
                f"num_calib_steps        = {_g('num_calib_steps')}",
                f"  feat_dim                 = {_g('feat_dim')}  "
                f"visual_dim = {_g('visual_dim')}  proprio_dim = {_g('proprio_dim')}",
                f"  threshold_init           = {_g('threshold_init')}",
                "",
                f"Sampled failure trajectories: {len(vizs)}",
                "",
                "Trigger rule: flag when per-frame LPB v2 weighted KNN min L2 "
                "distance is >= tau.",
            ]
            ax.text(
                0.01,
                0.95,
                "\n".join(lines),
                fontsize=10,
                family="monospace",
                va="top",
                ha="left",
            )
            pdf.savefig(fig)
            plt.close(fig)

            for viz in vizs:
                self._plot_trajectory(pdf, viz, threshold=threshold)

    def _plot_trajectory(self, pdf: PdfPages, viz: PerTrajectoryViz, *, threshold: float) -> None:
        T = int(viz.num_frames)
        t = np.arange(T)

        fig, ax = plt.subplots(figsize=(10.0, 4.5))
        ax.plot(t, viz.step_scores, color="#1f77b4", lw=1.4, label="KNN min L2 distance")
        if np.isfinite(threshold):
            ax.axhline(threshold, color="red", lw=1.2, ls="--", label=f"tau={threshold:.3f}")

        for seg in viz.failure_segments:
            try:
                s = int(seg["start"])
                e = int(seg["end"])
            except Exception:
                continue
            ax.axvspan(max(0, s), min(T - 1, e), color="red", alpha=0.12, zorder=0)

        pred_mask = viz.predictions.astype(bool)
        if pred_mask.any():
            ymin = float(np.nanmin(viz.step_scores))
            band = np.where(pred_mask, ymin, np.nan)
            ax.plot(t, band, color="red", lw=4, alpha=0.6, label="predicted fail frames")

        if viz.first_gt_failure_frame is not None:
            ax.axvline(
                int(viz.first_gt_failure_frame),
                color="darkred",
                lw=0.9,
                ls=":",
                label="first GT fail",
            )
        if viz.first_pred_failure_frame is not None:
            ax.axvline(
                int(viz.first_pred_failure_frame),
                color="orange",
                lw=0.9,
                ls=":",
                label="first PRED fail",
            )

        ax.set_xlim(0, max(T - 1, 1))
        ax.set_xlabel("frame")
        ax.set_ylabel("weighted KNN min L2 distance")
        ax.set_title(f"[{viz.task_name}] {viz.video_id}  (T={T})", fontsize=11, loc="left")
        ax.grid(True, alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            uniq_h.append(h)
            uniq_l.append(label)
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
        pdf_name: str = "dyn_disc_scores.pdf",
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
                f"[dyn_disc][viz] {traj.task_name}/{traj.video_id}  T={viz.num_frames}  "
                f"pred_frames={int(viz.predictions.sum())}  -> {video_path}",
                flush=True,
            )
            if self.debug_score_stats:
                print(f"[dyn_disc][viz][debug] {traj.task_name}/{traj.video_id} score_all: {_percentile_summary(viz.step_scores)}", flush=True)
                if viz.gt_mask is not None:
                    normal_scores = viz.step_scores[viz.gt_mask == 0]
                    failure_scores = viz.step_scores[viz.gt_mask == 1]
                    print(
                        f"[dyn_disc][viz][debug] {traj.task_name}/{traj.video_id} score_normal_gt0: "
                        f"{_percentile_summary(normal_scores)}",
                        flush=True,
                    )
                    print(
                        f"[dyn_disc][viz][debug] {traj.task_name}/{traj.video_id} score_failure_gt1: "
                        f"{_percentile_summary(failure_scores)}",
                        flush=True,
                    )
            vizs.append(viz)
            video_paths.append(video_path)

        pdf_path = os.path.join(out_dir, pdf_name)
        self.render_pdf(vizs, pdf_path)
        print(f"[dyn_disc][viz] wrote PDF -> {pdf_path}", flush=True)

        return {"videos": video_paths, "pdf": pdf_path}


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True, help="Path to an LPB v2 dynamics checkpoint.")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root containing data/<task>/<split> directories.")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", type=str, default="success_rollout-val")
    parser.add_argument("--fail-root", type=str, default=None,
                        help="Deprecated; use --data-root/--fail-split.")
    parser.add_argument("--success-root", type=str, default=None,
                        help="Deprecated; use --data-root/--success-split.")
    parser.add_argument("--success-cache-root", type=str, default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--metadata-cache-root", type=str, default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--cache-camera-names", nargs="*", default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--task", required=True, help="Single task name, e.g. PickPlaceBread")
    parser.add_argument("--num-trajs", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", type=str, default="dyn_disc_scores.pdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", type=str, default=None)
    parser.add_argument("--camera-name", type=str, default="agentview")

    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--quiet-fit", action="store_true")
    parser.add_argument(
        "--threshold-source",
        type=str,
        default="detector",
        choices=["detector", "success_percentile", "fixed"],
        help="Threshold used for visualization borders/predicted frames.",
    )
    parser.add_argument(
        "--benchmark-json",
        type=str,
        default=None,
        help="Use per-task trajectory best-F1 thresholds from a benchmark.json file.",
    )
    parser.add_argument(
        "--step-success-percentile",
        type=float,
        default=95.0,
        help="Success score percentile for threshold-source=success_percentile.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--no-debug-score-stats", action="store_true")

    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bench = FailureBenchmark(
        data_root=args.data_root,
        tasks=[str(args.task)],
        fail_split=args.fail_split,
        success_split=args.success_split,
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
    print(f"[dyn_disc][viz] sampled {n}/{len(fail_trajs)} failure trajectories from {args.task}", flush=True)

    discriminator = LPBV2BenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        visual_weight=float(args.visual_weight),
        proprio_weight=float(args.proprio_weight),
        delta=float(args.delta),
        knn_chunk_size=int(args.knn_chunk_size),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        discriminator.fit_on_benchmark(trajs)
        for task, detector in sorted(discriminator._detectors_per_task.items()):
            tau = float(detector.threshold) if detector.threshold is not None else float("nan")
            print(f"[dyn_disc][viz][debug] detector threshold task={task}: tau={tau:.4f}", flush=True)
        threshold_overrides = None
        threshold_source = str(args.threshold_source)
        if threshold_source == "success_percentile":
            threshold_overrides = _compute_success_percentile_thresholds(
                discriminator,
                trajs,
                percentile=float(args.step_success_percentile),
            )
            threshold_source = f"success_percentile_p{float(args.step_success_percentile):.1f}"
        elif threshold_source == "fixed":
            if args.fixed_threshold is None:
                raise ValueError("--threshold-source fixed requires --fixed-threshold")
            threshold_overrides = {str(args.task): float(args.fixed_threshold)}
            threshold_source = f"fixed_{float(args.fixed_threshold):.4f}"
        if args.benchmark_json:
            threshold_overrides = _load_benchmark_traj_best_f1_thresholds(str(args.benchmark_json))
            threshold_source = f"benchmark_json_traj_best_f1:{args.benchmark_json}"

        visualizer = LPBV2Visualizer(
            discriminator,
            camera_name=str(args.camera_name),
            fps=int(args.fps),
            border_thickness=int(args.border_thickness),
            threshold_overrides=threshold_overrides,
            threshold_source=threshold_source,
            debug_score_stats=not bool(args.no_debug_score_stats),
        )
        out_paths = visualizer.visualize(
            sampled,
            out_dir=str(args.out_dir),
            pdf_name=str(args.pdf_name),
        )
        print(
            f"[dyn_disc][viz] done. videos: {len(out_paths['videos'])}  pdf: {out_paths['pdf']}",
            flush=True,
        )
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
