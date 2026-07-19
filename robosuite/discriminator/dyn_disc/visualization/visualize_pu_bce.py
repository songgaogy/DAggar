"""Reusable nnPU failure-detector visualization helpers and load-only CLI.

Renders per-trajectory MP4 (with HUD + red border on predicted-failure frames)
and a multi-page PDF summary for a fitted
:class:`PUBCEBenchmarkDiscriminator`. Training and benchmark orchestration live
in their respective pipeline entry points. The standalone CLI only loads an
existing discriminator checkpoint; it never fits or modifies a model.

Threshold: the visualizer uses the detector's own per-task **success_percentile**
threshold (``tau = percentile(success-calib failure scores, 100 - delta)``). No
GT failure timing / two-class Youden is available in this branch (it would need
failure labels). The HUD shows ``failure_score = -g(z)`` and that tau.

Outputs:
    <out_dir>/videos/<video_id>.mp4
    <out_dir>/<pdf-name>.pdf
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-dyn-disc"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from benchmark.core import BenchmarkTrajectory

from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator
from robosuite.discriminator.dyn_disc.detectors.pu_bce import PUBCEDiscriminator


# ---------------------------------------------------------------------- #
# Self-contained rendering helpers                                        #
# ---------------------------------------------------------------------- #


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


def _pad_to_even(img: np.ndarray) -> np.ndarray:
    """Pad bottom/right when needed because libx264 prefers even H/W."""
    h, w = int(img.shape[0]), int(img.shape[1])
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _draw_border(img: np.ndarray, color: tuple, thickness: int) -> np.ndarray:
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


def _parse_camera_to_view(s: Optional[str]) -> Optional[Dict[str, str]]:
    if not s:
        return None
    out: Dict[str, str] = {}
    for chunk in s.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def _safe_id(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))


def _overlay_hud_pu(
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
        f"failure_score=-g(z)={score:.3f}  tau={threshold:.3f}",
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
# Data container                                                         #
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
    failure_segments: list
    first_gt_failure_frame: Optional[int]
    first_pred_failure_frame: Optional[int]


# ---------------------------------------------------------------------- #
# Visualizer                                                             #
# ---------------------------------------------------------------------- #


class PUBCEVisualizer:
    """Combine a fitted PU-BCE benchmark discriminator with video + PDF renderers."""

    def __init__(
        self,
        discriminator: PUBCEBenchmarkDiscriminator,
        *,
        camera_name: str = "agentview",
        fps: int = 20,
        border_thickness: int = 10,
        border_color_fail: tuple = (255, 0, 0),
        debug_score_stats: bool = True,
        flip_vertical: bool = True,
    ) -> None:
        self.discriminator = discriminator
        self.camera_name = str(camera_name)
        self.fps = int(fps)
        self.border_thickness = int(border_thickness)
        self.border_color_fail = tuple(int(c) for c in border_color_fail)
        self.debug_score_stats = bool(debug_score_stats)
        self.flip_vertical = bool(flip_vertical)
        self._font = _load_font()

    def _task_threshold(self, task: str) -> float:
        detector = self.discriminator._detectors_per_task.get(task, None)
        if detector is None:
            return float("nan")
        return float(detector.thresholds.get(task, float("nan")))

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #

    def _score_trajectory(self, traj: BenchmarkTrajectory) -> PerTrajectoryViz:
        out = self.discriminator.score_trajectory(traj)
        gt_mask = traj.load_failure_mask()
        first_gt = traj.first_gt_failure_frame()
        thresholds = out.aux.get("thresholds", None)
        if thresholds is None:
            thresholds = np.full_like(out.step_scores, out.aux.get("threshold", float("nan")))

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
    # Video                                                              #
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
        if self.flip_vertical:
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
                canvas = _overlay_hud_pu(
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
    # PDF                                                                #
    # ------------------------------------------------------------------ #

    def _split_summary_label(self, split: str, vizs: List[PerTrajectoryViz]) -> str:
        if split != "both":
            return f"{split} (n={len(vizs)})"
        n_fail = sum(1 for v in vizs if v.gt_mask is not None)
        n_succ = len(vizs) - n_fail
        return f"fail_rollout={n_fail}, success_rollout={n_succ} (total={len(vizs)})"

    def render_pdf(
        self,
        vizs: List[PerTrajectoryViz],
        out_path: str,
        *,
        split: str = "fail_rollout",
    ) -> None:
        if not vizs:
            raise RuntimeError("No trajectories to plot.")
        task = vizs[0].task_name
        threshold = self._task_threshold(task)
        summary = self.discriminator.calibration_summary()
        per_task = summary.get("per_task", {}).get(task, {})
        glob = summary.get("global", {}) or {}

        def _g(d: dict, key: str, default: str = "n/a"):
            v = d.get(key, default)
            return default if v is None else v

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with PdfPages(out_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 6.0))
            ax.axis("off")
            ax.set_title(f"dyn_disc PU-BCE (nnPU) summary - task={task}", fontsize=14, loc="left")

            lines = [
                f"discriminator: {self.discriminator.name}",
                f"model_ckpt: {summary.get('model_ckpt', 'n/a')}",
                f"view_names: {summary.get('view_names', 'n/a')}",
                f"camera_to_view: {summary.get('camera_to_view', 'n/a')}",
                f"feature_source: {summary.get('feature_source', 'n/a')}  "
                f"transformer_layer: {summary.get('transformer_layer', 'n/a')}",
                f"delta (FA budget %): {summary.get('delta', 'n/a')}  "
                f"calib_fraction: {summary.get('calib_fraction', 'n/a')}",
                f"calib_mode: {summary.get('calib_mode', 'success_percentile')}",
                "",
                "nnPU head:",
                f"  feat_dim   = {_g(glob, 'feat_dim')}  "
                f"head_hidden = {_g(glob, 'head_hidden')}  "
                f"head_layers = {_g(glob, 'head_layers')}",
                f"  pi_p       = {_g(glob, 'pi_p')}  "
                f"loss_surrogate = {_g(glob, 'loss_surrogate')}  "
                f"nn_correction = {_g(glob, 'nn_correction')}",
                f"  epochs     = {_g(glob, 'epochs')}  lr = {_g(glob, 'lr')}  "
                f"batch_size = {_g(glob, 'batch_size')}",
                f"  num_unlabeled_fail_trajectories = {_g(glob, 'num_unlabeled_fail_trajectories')}",
                f"  loaded_from_ckpt = {_g(glob, 'loaded_from_ckpt')}",
                "",
                f"threshold tau (success_percentile) = {threshold:.4f}",
                "",
                "Calibration (success pool, per task):",
                f"  num_success_trajectories       = {_g(per_task, 'num_success_trajectories')}",
                f"  num_train_success_trajectories = {_g(per_task, 'num_train_success_trajectories')}  "
                f"num_calib_success_trajectories = {_g(per_task, 'num_calib_success_trajectories')}",
                f"  num_train_success_frames       = {_g(per_task, 'num_train_success_frames')}  "
                f"num_calib_success_frames       = {_g(per_task, 'num_calib_success_frames')}",
                f"  num_unlabeled_fail_frames      = {_g(per_task, 'num_unlabeled_fail_frames')}",
                f"  calib_score_min  = {_g(per_task, 'calib_score_min')}  "
                f"calib_score_max  = {_g(per_task, 'calib_score_max')}",
                f"  calib_score_mean = {_g(per_task, 'calib_score_mean')}  "
                f"calib_score_std  = {_g(per_task, 'calib_score_std')}",
                "",
                f"Sampled trajectories: {self._split_summary_label(split, vizs)}",
                "",
                "Trigger rule: flag when failure_score = -head_logit(z) >= tau.",
            ]
            ax.text(0.01, 0.97, "\n".join(lines), fontsize=9, family="monospace", va="top", ha="left")
            pdf.savefig(fig)
            plt.close(fig)

            for viz in vizs:
                self._plot_trajectory(pdf, viz, threshold=threshold)

    def _plot_trajectory(self, pdf: PdfPages, viz: PerTrajectoryViz, *, threshold: float) -> None:
        T = int(viz.num_frames)
        t = np.arange(T)
        fig, ax = plt.subplots(figsize=(10.0, 4.5))
        ax.plot(t, viz.step_scores, color="#1f77b4", lw=1.4, label="PU failure score (-logit)")
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
            ax.axvline(int(viz.first_gt_failure_frame), color="darkred", lw=0.9, ls=":",
                       label="first GT fail")
        if viz.first_pred_failure_frame is not None:
            ax.axvline(int(viz.first_pred_failure_frame), color="orange", lw=0.9, ls=":",
                       label="first PRED fail")

        ax.set_xlim(0, max(T - 1, 1))
        ax.set_xlabel("frame")
        ax.set_ylabel("PU failure score (-logit)")
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
    # Entry                                                              #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        trajectories: List[BenchmarkTrajectory],
        *,
        out_dir: str,
        pdf_name: str,
        split: str = "fail_rollout",
    ) -> dict:
        if not trajectories:
            raise RuntimeError("No trajectories provided for visualization.")
        videos_dir = os.path.join(out_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        vizs: List[PerTrajectoryViz] = []
        video_paths: List[str] = []
        for traj in trajectories:
            viz = self._score_trajectory(traj)
            if split == "both":
                kind = "fail_rollout" if bool(traj.is_failure) else "success_rollout"
                traj_videos_dir = os.path.join(videos_dir, kind)
            else:
                traj_videos_dir = videos_dir
            os.makedirs(traj_videos_dir, exist_ok=True)
            video_path = os.path.join(traj_videos_dir, f"{_safe_id(traj.video_id)}.mp4")
            self.render_video(traj, viz, video_path)
            print(
                f"[pu_bce][viz] {traj.task_name}/{traj.video_id}  T={viz.num_frames}  "
                f"pred_frames={int(viz.predictions.sum())}  -> {video_path}",
                flush=True,
            )
            if self.debug_score_stats:
                print(
                    f"[pu_bce][viz][debug] {traj.task_name}/{traj.video_id} score_all: "
                    f"{_percentile_summary(viz.step_scores)}",
                    flush=True,
                )
                if viz.gt_mask is not None:
                    normal = viz.step_scores[viz.gt_mask == 0]
                    failure = viz.step_scores[viz.gt_mask == 1]
                    print(
                        f"[pu_bce][viz][debug] {traj.task_name}/{traj.video_id} score_normal_gt0: "
                        f"{_percentile_summary(normal)}",
                        flush=True,
                    )
                    print(
                        f"[pu_bce][viz][debug] {traj.task_name}/{traj.video_id} score_failure_gt1: "
                        f"{_percentile_summary(failure)}",
                        flush=True,
                    )
            vizs.append(viz)
            video_paths.append(video_path)

        pdf_path = os.path.join(out_dir, pdf_name)
        self.render_pdf(vizs, pdf_path, split=split)
        print(f"[pu_bce][viz] wrote PDF -> {pdf_path}", flush=True)
        return {"videos": video_paths, "pdf": pdf_path}


# ---------------------------------------------------------------------- #
# Eval-trajectory sampling                                                #
# ---------------------------------------------------------------------- #


def _sample_trajectories_for_viz(
    trajs: Sequence[BenchmarkTrajectory],
    *,
    task: str,
    split: str,
    num_trajs: int,
    seed: int,
) -> List[BenchmarkTrajectory]:
    if split == "both":
        fail = _sample_trajectories_for_viz(
            trajs,
            task=task,
            split="fail_rollout",
            num_trajs=num_trajs,
            seed=seed,
        )
        success = _sample_trajectories_for_viz(
            trajs,
            task=task,
            split="success_rollout",
            num_trajs=num_trajs,
            seed=int(seed) + 1,
        )
        return fail + success

    if split == "success_rollout":
        pool = [t for t in trajs if not bool(t.is_failure) and str(t.task_name) == str(task)]
        kind = "success"
    elif split == "fail_rollout":
        pool = [t for t in trajs if bool(t.is_failure) and str(t.task_name) == str(task)]
        kind = "failure"
    else:
        raise ValueError(
            f"unknown split {split!r}; expected fail_rollout, success_rollout, or both"
        )

    if not pool:
        raise RuntimeError(
            f"No {kind} trajectories in eval set for task {task!r} (split={split!r})."
        )
    rng = random.Random(int(seed))
    n = min(int(num_trajs), len(pool))
    sampled = rng.sample(pool, n)
    print(
        f"[pu_bce][viz] sampled {n}/{len(pool)} {split} trajectories from {task}",
        flush=True,
    )
    return sampled


# ---------------------------------------------------------------------- #
# Load nnPU head from disk without re-fitting                            #
# ---------------------------------------------------------------------- #


def _bootstrap_from_ckpt(disc: PUBCEBenchmarkDiscriminator, ckpt_path: str) -> None:
    payload = torch.load(
        ckpt_path,
        map_location=torch.device(disc.device),
        weights_only=False,
    )
    disc.use_chunk = bool(payload.get("use_chunk", False))
    state = payload["pu_bce_detector"]
    detector = PUBCEDiscriminator(
        in_dim=int(payload["in_dim"]),
        hidden=int(payload["hidden"]),
        num_layers=int(payload["num_layers"]),
        device=str(disc.device),
    )
    detector.load_state_dict(state)
    if not detector.thresholds:
        raise RuntimeError(
            f"PU-BCE checkpoint at {ckpt_path} has no per-task thresholds; cannot score."
        )
    if detector._delta is not None:
        disc.delta = float(detector._delta)

    disc._shared_detector = detector
    disc._detectors_per_task = {task: detector for task in detector.thresholds}
    disc._global_stats = {
        "feat_dim": int(payload["in_dim"]),
        "epochs": int(payload.get("epoch", 0)),
        "head_hidden": int(payload["hidden"]),
        "head_layers": int(payload["num_layers"]),
        "feature_source": str(payload.get("feature_source", disc.feature_source)),
        "transformer_layer": int(payload.get("transformer_layer", disc.transformer_layer)),
        "use_chunk": bool(disc.use_chunk),
        "pi_p": payload.get("pi_p"),
        "loss_surrogate": payload.get("loss_surrogate"),
        "nn_correction": payload.get("nn_correction"),
        "num_unlabeled_fail_trajectories": len(payload.get("unlabeled_fail_video_ids", []) or []),
        "loaded_from_ckpt": str(ckpt_path),
    }
    disc._calibration_stats = {}
    for task, tau in detector.thresholds.items():
        cs = detector.calib_stats.get(task)
        disc._calibration_stats[task] = {
            "threshold": float(tau),
            "calib_score_min": None if cs is None else float(cs.calib_score_min),
            "calib_score_max": None if cs is None else float(cs.calib_score_max),
            "calib_score_mean": None if cs is None else float(cs.calib_score_mean),
            "calib_score_std": None if cs is None else float(cs.calib_score_std),
            "num_calib_success_frames": None if cs is None else int(cs.num_calib_frames),
        }
    print(
        f"[pu_bce][viz] loaded nnPU head from {ckpt_path}; tasks="
        f"{sorted(detector.thresholds)} "
        f"tau="
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(detector.thresholds.items())),
        flush=True,
    )


# ---------------------------------------------------------------------- #
# Load-only CLI                                                          #
# ---------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an existing robosuite nnPU checkpoint (load-only)."
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["success_rollout", "fail_rollout", "both"],
        help="Evaluation pool to visualize.",
    )
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--load-ckpt", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fail-split", default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", default="success_rollout-val")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-trajs", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", default="pu_bce_scores.pdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)
    parser.add_argument("--max-fail-per-task", type=int, default=50)
    parser.add_argument("--max-success-per-task", type=int, default=50)
    parser.add_argument("--no-flip-vertical", action="store_true")
    parser.add_argument("--no-debug-score-stats", action="store_true")
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--camera-to-view", default=None)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument(
        "--delta",
        type=float,
        default=5.0,
        help=(
            "Fallback delta for checkpoints without calibration metadata. "
            "Stored checkpoint delta and threshold take precedence."
        ),
    )
    parser.add_argument(
        "--feature-source",
        default="transformer",
        choices=["encoder", "transformer"],
    )
    parser.add_argument("--transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    return parser.parse_args()


def _require_cuda(device_value: str) -> torch.device:
    device = torch.device(str(device_value))
    if device.type != "cuda":
        raise ValueError(f"nnPU visualization requires CUDA, got {device_value!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("nnPU visualization requires CUDA, but CUDA is unavailable")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {device} is unavailable; visible count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0 if device.index is None else device.index)
    return device


def _build_eval_trajectories(args: argparse.Namespace) -> List[BenchmarkTrajectory]:
    from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark

    benchmark = FailureBenchmark(
        data_root=str(args.data_root),
        tasks=[str(args.task)],
        fail_split=str(args.fail_split),
        success_split=str(args.success_split),
        max_fail_per_task=int(args.max_fail_per_task),
        max_success_per_task=int(args.max_success_per_task),
    )
    trajectories = benchmark.trajectories()
    if not trajectories:
        raise RuntimeError(f"No trajectories discovered for task {args.task!r}.")
    return trajectories


def main() -> None:
    args = _parse_args()
    device = _require_cuda(str(args.device))
    if not os.path.isfile(args.load_ckpt):
        raise FileNotFoundError(f"--load-ckpt not found: {args.load_ckpt}")

    trajectories = _build_eval_trajectories(args)
    sampled = _sample_trajectories_for_viz(
        trajectories,
        task=str(args.task),
        split=str(args.split),
        num_trajs=int(args.num_trajs),
        seed=int(args.seed),
    )

    with torch.device(device):
        discriminator = PUBCEBenchmarkDiscriminator(
            model_ckpt=str(args.model_ckpt),
            unlabeled_fail_trajectories=[],
            save_ckpt_dir=None,
            device=str(device),
            encode_batch_size=int(args.encode_batch_size),
            proprio_indices=(
                list(args.proprio_indices) if args.proprio_indices else None
            ),
            camera_to_view=_parse_camera_to_view(args.camera_to_view),
            visual_weight=float(args.visual_weight),
            proprio_weight=float(args.proprio_weight),
            action_weight=float(args.action_weight),
            delta=float(args.delta),
            feature_source=str(args.feature_source),
            transformer_layer=int(args.transformer_layer),
            calib_fraction=float(args.calib_fraction),
            seed=int(args.seed),
            verbose_fit=False,
        )

    try:
        _bootstrap_from_ckpt(discriminator, str(args.load_ckpt))
        if str(args.task) not in discriminator._detectors_per_task:
            raise RuntimeError(
                f"Task {args.task!r} is absent from the checkpoint; available tasks: "
                f"{sorted(discriminator._detectors_per_task)}"
            )
        visualizer = PUBCEVisualizer(
            discriminator,
            camera_name=str(args.camera_name),
            fps=int(args.fps),
            border_thickness=int(args.border_thickness),
            debug_score_stats=not bool(args.no_debug_score_stats),
            flip_vertical=not bool(args.no_flip_vertical),
        )
        paths = visualizer.visualize(
            sampled,
            out_dir=str(args.out_dir),
            pdf_name=str(args.pdf_name),
            split=str(args.split),
        )
        print(
            f"[pu_bce][viz] done. videos={len(paths['videos'])} pdf={paths['pdf']}",
            flush=True,
        )
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
