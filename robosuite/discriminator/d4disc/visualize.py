"""Failure-detector visualization for D4-Disc.

Mirrors ``robosuite.discriminator.d3disc.visualize`` and adds a second
diagnostics panel with the CFG raw residuals (r+, r-) and the learned
advantage A. The aux dict on ``DiscriminatorOutput`` already carries
``r_plus`` and ``r_minus`` padded to the trajectory length (D4-specific
extension of the D3 schema).

Usage:
    python -m robosuite.discriminator.d4disc.visualize \
        --policy-ckpt /abs/flow_multi.pt \
        --d4-ckpt    /abs/d4_dynamics.pt \
        --fail-root  /abs/data/utils/fail_rollout \
        --success-root /abs/data/utils/success_rollout \
        --task PickPlaceBread --num-trajs 4 --omega 0.5 \
        --out-dir /tmp/d4_viz
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from data.utils.benchmark import BenchmarkTrajectory, FailureBenchmark

from .d4_benchmark import D4BenchmarkDiscriminator


def _pad_to_even(img: np.ndarray) -> np.ndarray:
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
    r_plus: Optional[float],
    r_minus: Optional[float],
    pred_fail: bool,
    gt_fail: Optional[bool],
    frame_idx: int,
    total: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil, mode="RGBA")

    w, _h = pil.size
    box_h = 116 if r_minus is not None else 92
    draw.rectangle([(0, 0), (w, box_h)], fill=(0, 0, 0, 140))

    lines = [
        f"frame {frame_idx + 1}/{total}",
        f"lambda={score:.3f}  tau={threshold:.3f}",
    ]
    colors = [(255, 255, 255, 255), (255, 255, 255, 255)]
    if r_minus is not None:
        lines.append(f"r+={r_plus:.3f}  r-={r_minus:.3f}  A={(r_minus - (r_plus or 0.0)):.3f}")
        colors.append((200, 200, 255, 255))
    else:
        lines.append(f"r+={r_plus:.3f}")
        colors.append((200, 200, 255, 255))
    lines.append(
        f"PRED: {'FAIL' if pred_fail else 'OK  '}   "
        + (f"GT: {'FAIL' if gt_fail else 'OK  '}" if gt_fail is not None else "")
    )
    colors.append((255, 80, 80, 255) if pred_fail else (80, 255, 80, 255))

    y = 4
    for text, color in zip(lines, colors):
        draw.text((8, y), text, fill=color, font=font)
        y += 28
    return np.asarray(pil)


@dataclass
class PerTrajectoryViz:
    video_id: str
    task_name: str
    num_frames: int
    step_scores: np.ndarray
    predictions: np.ndarray
    r_plus: np.ndarray
    r_minus: Optional[np.ndarray]
    threshold: float
    gt_mask: Optional[np.ndarray]
    failure_segments: list[dict]
    first_gt_failure_frame: Optional[int]
    first_pred_failure_frame: Optional[int]


class D4Visualizer:
    def __init__(
        self,
        discriminator: D4BenchmarkDiscriminator,
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
        tau = self.discriminator._tau_per_task.get(task, None)
        return float("nan") if tau is None else float(tau)

    def _score_trajectory(self, traj: BenchmarkTrajectory) -> PerTrajectoryViz:
        out = self.discriminator.score_trajectory(traj)
        gt_mask = traj.load_failure_mask()
        first_gt = traj.first_gt_failure_frame()
        aux = out.aux or {}
        T = int(traj.num_frames)
        r_plus = np.asarray(aux.get("r_plus", np.zeros(T)), dtype=np.float32)
        r_minus_raw = aux.get("r_minus")
        r_minus = None if r_minus_raw is None else np.asarray(r_minus_raw, dtype=np.float32)
        return PerTrajectoryViz(
            video_id=str(traj.video_id),
            task_name=str(traj.task_name),
            num_frames=T,
            step_scores=np.asarray(out.step_scores, dtype=np.float32),
            predictions=np.asarray(out.predictions, dtype=np.int64),
            r_plus=r_plus,
            r_minus=r_minus,
            threshold=float(aux.get("threshold", float("nan"))),
            gt_mask=None if gt_mask is None else np.asarray(gt_mask, dtype=np.uint8),
            failure_segments=list(traj.failure_segments),
            first_gt_failure_frame=first_gt,
            first_pred_failure_frame=out.first_failure_frame,
        )

    # ------------------------------------------------------------------ #
    # Video                                                              #
    # ------------------------------------------------------------------ #

    def render_video(
        self,
        traj: BenchmarkTrajectory,
        viz: PerTrajectoryViz,
        out_path: str,
    ) -> None:
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
                r_plus_val = float(viz.r_plus[t]) if viz.r_plus is not None else 0.0
                r_minus_val = None if viz.r_minus is None else float(viz.r_minus[t])
                canvas = _overlay_hud(
                    canvas,
                    score=float(viz.step_scores[t]),
                    threshold=float(viz.threshold),
                    r_plus=r_plus_val,
                    r_minus=r_minus_val,
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
    # PDF                                                                #
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
            ax.set_title(f"D4-Disc summary — task={task}", fontsize=14, loc="left")

            def _g(key, default="n/a"):
                return per_task.get(key, default)

            lines = [
                f"discriminator: {self.discriminator.name}",
                f"policy_ckpt: {summary.get('policy_ckpt_path', 'n/a')}",
                f"d4_ckpt:     {summary.get('d4_ckpt_path', 'n/a')}",
                f"omega: {summary.get('omega', 'n/a')}   "
                f"sigma_sq: {summary.get('sigma_sq', 'n/a')}",
                f"lambda_mode: {summary.get('lambda_mode', 'n/a')}   "
                f"lambda_window_size: {summary.get('lambda_window_size', 'n/a')}",
                f"per_task_calibration: {summary.get('per_task_calibration', True)}   "
                f"delta (FA %): {summary.get('delta', 'n/a')}",
                f"threshold tau (task): {threshold:.4f}",
                "",
                "Calibration (per task):",
                f"  num_success_trajectories = {_g('num_success_trajectories')}",
                f"  num_bank_trajectories    = {_g('num_bank_trajectories')}   "
                f"num_calib_trajectories = {_g('num_calib_trajectories')}",
                f"  num_fail_trajectories    = {_g('num_fail_trajectories')}",
                "",
                f"Sampled failure trajectories: {len(vizs)}",
                "",
                "Score: lambda_t = ||f_omega - z_{t+h}||^2 / (2 sigma^2).",
                "  f_omega = (1+w) f(.|c=+) - w f(.|c=-)   (CFG guidance).",
            ]
            ax.text(0.01, 0.95, "\n".join(lines), fontsize=10, family="monospace", va="top", ha="left")
            pdf.savefig(fig)
            plt.close(fig)

            for viz in vizs:
                self._plot_trajectory(pdf, viz, threshold=threshold)

    def _plot_trajectory(self, pdf: PdfPages, viz: PerTrajectoryViz, *, threshold: float) -> None:
        T = int(viz.num_frames)
        t = np.arange(T)
        has_neg = viz.r_minus is not None

        fig, axes = plt.subplots(
            nrows=3 if has_neg else 2,
            ncols=1,
            figsize=(10.0, 8.5 if has_neg else 6.0),
            sharex=True,
        )
        ax_top = axes[0]
        ax_mid = axes[1]
        ax_bot = axes[2] if has_neg else None

        ax_top.plot(t, viz.step_scores, color="#1f77b4", lw=1.4, label="lambda (agg)")
        if np.isfinite(threshold):
            ax_top.axhline(threshold, color="red", lw=1.2, ls="--", label=f"tau={threshold:.3f}")
        for seg in viz.failure_segments:
            try:
                s = int(seg["start"]); e = int(seg["end"])
            except Exception:
                continue
            ax_top.axvspan(max(0, s), min(T - 1, e), color="red", alpha=0.12, zorder=0)
        pred_mask = viz.predictions.astype(bool)
        if pred_mask.any():
            ymin = float(viz.step_scores.min())
            band = np.where(pred_mask, ymin, np.nan)
            ax_top.plot(t, band, color="red", lw=4, alpha=0.6, label="predicted fail frames")
        if viz.first_gt_failure_frame is not None:
            ax_top.axvline(int(viz.first_gt_failure_frame), color="darkred", lw=0.9, ls=":", label="first GT fail")
        if viz.first_pred_failure_frame is not None:
            ax_top.axvline(int(viz.first_pred_failure_frame), color="orange", lw=0.9, ls=":", label="first PRED fail")

        ax_top.set_xlim(0, max(T - 1, 1))
        ax_top.set_ylabel("lambda")
        ax_top.set_title(f"[{viz.task_name}] {viz.video_id}  (T={T})", fontsize=11, loc="left")
        ax_top.grid(True, alpha=0.2)

        handles, labels = ax_top.get_legend_handles_labels()
        seen, uh, ul = set(), [], []
        for h, l in zip(handles, labels):
            if l in seen: continue
            seen.add(l); uh.append(h); ul.append(l)
        uh.append(Patch(facecolor="red", alpha=0.12, label="GT failure segment"))
        ul.append("GT failure segment")
        ax_top.legend(uh, ul, loc="upper left", fontsize=8, framealpha=0.85)

        ax_mid.plot(t, viz.r_plus, color="#2ca02c", lw=1.2, label="r+ (|f(+) - z_t+h|^2)")
        if has_neg:
            ax_mid.plot(t, viz.r_minus, color="#d62728", lw=1.2, label="r-")
        ax_mid.set_ylabel("residual")
        ax_mid.grid(True, alpha=0.2)
        ax_mid.legend(loc="upper left", fontsize=8, framealpha=0.85)

        if has_neg and ax_bot is not None:
            advantage = viz.r_minus - viz.r_plus
            ax_bot.plot(t, advantage, color="#9467bd", lw=1.2, label="A (r- - r+)")
            ax_bot.axhline(0.0, color="gray", lw=0.8, ls=":")
            ax_bot.set_xlabel("frame")
            ax_bot.set_ylabel("advantage")
            ax_bot.grid(True, alpha=0.2)
            ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.85)
        else:
            ax_mid.set_xlabel("frame")

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
        ax_anchor = ax_bot if ax_bot is not None else ax_mid
        ax_anchor.text(0.01, -0.23, stats, transform=ax_anchor.transAxes, fontsize=9, family="monospace")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def visualize(
        self,
        fail_trajectories: list[BenchmarkTrajectory],
        *,
        out_dir: str,
        pdf_name: str = "d4_scores.pdf",
    ) -> dict:
        if not fail_trajectories:
            raise RuntimeError("No failure trajectories provided.")
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
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--d4-ckpt", required=True)
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-trajs", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", type=str, default="d4_scores.pdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--cache-root", type=str, default="data/.lpb_score_cache")
    parser.add_argument("--camera-name", type=str, default="agentview")

    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--sigma-sq", type=float, default=None)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--lambda-mode", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--lambda-window-size", type=int, default=-1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)

    parser.add_argument("--quiet-fit", action="store_true")
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

    discriminator = D4BenchmarkDiscriminator(
        policy_ckpt_path=str(args.policy_ckpt),
        d4_ckpt_path=str(args.d4_ckpt),
        cache_root=str(args.cache_root),
        device=str(args.device),
        encoder_batch_size=int(args.encoder_batch_size),
        image_size=int(args.image_size),
        omega=float(args.omega),
        sigma_sq=(None if args.sigma_sq is None else float(args.sigma_sq)),
        delta=float(args.delta),
        lambda_mode=str(args.lambda_mode),
        lambda_window_size=int(args.lambda_window_size),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        discriminator.fit_on_benchmark(trajs)
        visualizer = D4Visualizer(
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
