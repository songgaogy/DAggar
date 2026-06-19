"""Failure-detector visualization for the dyn_disc BCE discriminator.

Renders per-trajectory MP4 (with HUD + red border on predicted-failure frames)
and a multi-page PDF summary, driven by a fitted
:class:`BCEBenchmarkDiscriminator`.

Threshold: the visualizer always reports the **two-class Youden** operating
point. ``tau`` is recomputed for the sampled task as ``argmax (TPR - FPR)`` on
the empirical failure-score distributions of (success-calib frames, fail-bank
GT-suffix frames). It is independent of the detector's own per-task threshold
(which is set by the runner's ``calib_mode`` choice). To inspect a different
operating point, change the runner's ``calib_mode`` and re-fit.

Robosuite layout (mirrors ``robosuite_bce.py``): eval trajectories from
fail_rollout-val-labeled / success_rollout-val, BCE bank from
fail_rollout-labeled.

Use ``--split`` to choose which eval trajectories are rendered:

  * ``fail_rollout``    - sample from ``--fail-split`` (default)
  * ``success_rollout`` - sample from ``--success-split`` (``is_failure=False``)

Outputs:
    <out_dir>/videos/<video_id>.mp4
    <out_dir>/<pdf-name>.pdf
    <out_dir>/checkpoints/bce_head.pth  (only when fitting; absent under --load-ckpt)
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

from robosuite.discriminator.dyn_disc.adapters.bce import BCEBenchmarkDiscriminator
from robosuite.discriminator.dyn_disc.detectors.bce import (
    BCEDiscriminator,
    two_class_youden_threshold,
)


# ---------------------------------------------------------------------- #
# Rendering helpers                                                      #
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


def _overlay_hud_bce(
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
        f"bce_score={score:.3f}  tau={threshold:.3f}",
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
# Per-task failure-score arrays used by Youden                           #
# ---------------------------------------------------------------------- #


def _compute_fail_suffix_failure_scores_per_task(
    discriminator: BCEBenchmarkDiscriminator,
    fail_bank_trajs: Sequence[BenchmarkTrajectory],
) -> Dict[str, np.ndarray]:
    """Score each fail-bank trajectory's GT-failure suffix and concat per task.

    Mirrors the suffix slice used by ``BCEBenchmarkDiscriminator.fit_on_benchmark``:
    ``suffix = features[first_gt_failure_frame:]``. Skips trajectories with no
    usable GT cut (mirrors the adapter's ``skipped_no_gt`` rule).
    """
    if discriminator._shared_detector is None:
        raise RuntimeError(
            "Cannot compute fail-suffix scores: BCE head is not fitted/loaded."
        )
    per_task: Dict[str, List[np.ndarray]] = {}
    for t in fail_bank_trajs:
        t_star = t.first_gt_failure_frame()
        if t_star is None:
            continue
        feat = discriminator._encode(t)
        T = int(feat.shape[0])
        t_star_int = int(t_star)
        if t_star_int <= 0 or t_star_int >= T:
            continue
        suffix = feat[t_star_int:]
        g = discriminator._shared_detector._logits_np(suffix)
        per_task.setdefault(str(t.task_name), []).append((-g).astype(np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in per_task.items() if v}


def _compute_success_calib_failure_scores_per_task(
    discriminator: BCEBenchmarkDiscriminator,
    eval_trajs: Sequence[BenchmarkTrajectory],
    *,
    seed: int,
    calib_fraction: float,
) -> Dict[str, np.ndarray]:
    """Re-derive the success train/calib split used by the adapter and score
    the calib slice through the fitted head.

    Replication note: must match the rng + split rule in
    ``BCEBenchmarkDiscriminator.fit_on_benchmark`` (NumPy ``default_rng(seed)``,
    ``calib_fraction``, per-task ``permutation``, ``n_calib =
    max(1, min(N-1, round(calib_fraction*N)))``). Changes to either side must
    be mirrored here, otherwise ``youden`` will mix train + calib frames.
    """
    if discriminator._shared_detector is None:
        raise RuntimeError(
            "Cannot compute success-calib scores: BCE head is not fitted/loaded."
        )
    task_to_success: Dict[str, List[BenchmarkTrajectory]] = {}
    for t in eval_trajs:
        if bool(t.is_failure):
            continue
        task_to_success.setdefault(str(t.task_name), []).append(t)

    rng = np.random.default_rng(int(seed))
    out: Dict[str, np.ndarray] = {}
    for task in sorted(task_to_success):
        succ_list = task_to_success[task]
        if len(succ_list) < 2:
            continue
        perm = rng.permutation(len(succ_list))
        n_calib = int(round(float(calib_fraction) * len(succ_list)))
        n_calib = max(1, min(len(succ_list) - 1, n_calib))
        calib_idx = set(perm[:n_calib].tolist())
        calib_trajs = [t for i, t in enumerate(succ_list) if i in calib_idx]
        if not calib_trajs:
            continue
        feats_list = [discriminator._encode(t) for t in calib_trajs]
        catf = torch.cat(
            [f.to(torch.float32).reshape(-1, f.shape[-1]) for f in feats_list], dim=0,
        )
        g = discriminator._shared_detector._logits_np(catf)
        out[task] = (-g).astype(np.float32)
    return out


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


class BCEVisualizer:
    """Combine a fitted BCE benchmark discriminator with video + PDF renderers."""

    def __init__(
        self,
        discriminator: BCEBenchmarkDiscriminator,
        *,
        camera_name: str = "agentview",
        fps: int = 20,
        border_thickness: int = 10,
        border_color_fail: tuple = (255, 0, 0),
        threshold_overrides: Optional[Dict[str, float]] = None,
        threshold_source: str = "youden",
        debug_score_stats: bool = True,
        flip_vertical: bool = True,
    ) -> None:
        self.discriminator = discriminator
        self.camera_name = str(camera_name)
        self.fps = int(fps)
        self.border_thickness = int(border_thickness)
        self.border_color_fail = tuple(int(c) for c in border_color_fail)
        self.threshold_overrides = dict(threshold_overrides) if threshold_overrides else {}
        self.threshold_source = str(threshold_source)
        self.debug_score_stats = bool(debug_score_stats)
        self.flip_vertical = bool(flip_vertical)
        self._font = _load_font()

    def _task_threshold(self, task: str) -> float:
        if task in self.threshold_overrides:
            return float(self.threshold_overrides[task])
        if "__global__" in self.threshold_overrides:
            return float(self.threshold_overrides["__global__"])
        detector = self.discriminator._detectors_per_task.get(task, None)
        if detector is None:
            return float("nan")
        return float(detector.thresholds.get(task, float("nan")))

    def _detector_threshold(self, task: str) -> float:
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
                canvas = _overlay_hud_bce(
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
            ax.set_title(f"dyn_disc BCE summary - task={task}", fontsize=14, loc="left")

            lines = [
                f"discriminator: {self.discriminator.name}",
                f"model_ckpt: {summary.get('model_ckpt', 'n/a')}",
                f"view_names: {summary.get('view_names', 'n/a')}",
                f"camera_to_view: {summary.get('camera_to_view', 'n/a')}",
                f"feature_source: {summary.get('feature_source', 'n/a')}  "
                f"transformer_layer: {summary.get('transformer_layer', 'n/a')}",
                f"delta (FA budget %): {summary.get('delta', 'n/a')}  "
                f"calib_fraction: {summary.get('calib_fraction', 'n/a')}",
                f"encode_batch_size: {summary.get('encode_batch_size', 'n/a')}",
                "",
                "BCE head:",
                f"  feat_dim     = {_g(glob, 'feat_dim')}  "
                f"head_hidden = {_g(glob, 'head_hidden')}  "
                f"head_layers = {_g(glob, 'head_layers')}",
                f"  epochs       = {_g(glob, 'epochs')}  lr = {_g(glob, 'lr')}  "
                f"weight_decay = {_g(glob, 'weight_decay')}  "
                f"batch_size = {_g(glob, 'batch_size')}",
                f"  max_expert_other_ratio = {_g(glob, 'max_expert_other_ratio')}",
                f"  num_fail_bank_trajectories = {_g(glob, 'num_fail_bank_trajectories')}  "
                f"skipped_no_gt = {_g(glob, 'num_fail_bank_skipped_no_gt')}",
                f"  loaded_from_ckpt = {_g(glob, 'loaded_from_ckpt')}",
                "",
                f"visualized threshold source: {self.threshold_source}",
                f"visualized threshold tau   : {threshold:.4f}",
                f"detector threshold tau     : {self._detector_threshold(task):.4f}",
                "",
                "Calibration (success pool, per task):",
                f"  num_success_trajectories       = {_g(per_task, 'num_success_trajectories')}",
                f"  num_train_success_trajectories = {_g(per_task, 'num_train_success_trajectories')}  "
                f"num_calib_success_trajectories = {_g(per_task, 'num_calib_success_trajectories')}",
                f"  num_train_success_frames       = {_g(per_task, 'num_train_success_frames')}  "
                f"num_calib_success_frames       = {_g(per_task, 'num_calib_success_frames')}",
                f"  num_fail_prefix_frames         = {_g(per_task, 'num_fail_prefix_frames')}  "
                f"num_fail_other_frames          = {_g(per_task, 'num_fail_other_frames')}",
                f"  calib_score_min  = {_g(per_task, 'calib_score_min')}  "
                f"calib_score_max  = {_g(per_task, 'calib_score_max')}",
                f"  calib_score_mean = {_g(per_task, 'calib_score_mean')}  "
                f"calib_score_std  = {_g(per_task, 'calib_score_std')}",
                "",
                f"Sampled {split} trajectories: {len(vizs)}",
                "",
                "Trigger rule: flag when bce_failure_score = -head_logit(z) >= tau.",
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
        ax.plot(t, viz.step_scores, color="#1f77b4", lw=1.4, label="BCE failure score (-logit)")
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
        ax.set_ylabel("BCE failure score (-logit)")
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
            video_path = os.path.join(videos_dir, f"{_safe_id(traj.video_id)}.mp4")
            self.render_video(traj, viz, video_path)
            print(
                f"[bce][viz] {traj.task_name}/{traj.video_id}  T={viz.num_frames}  "
                f"pred_frames={int(viz.predictions.sum())}  -> {video_path}",
                flush=True,
            )
            if self.debug_score_stats:
                print(
                    f"[bce][viz][debug] {traj.task_name}/{traj.video_id} score_all: "
                    f"{_percentile_summary(viz.step_scores)}",
                    flush=True,
                )
                if viz.gt_mask is not None:
                    normal = viz.step_scores[viz.gt_mask == 0]
                    failure = viz.step_scores[viz.gt_mask == 1]
                    print(
                        f"[bce][viz][debug] {traj.task_name}/{traj.video_id} score_normal_gt0: "
                        f"{_percentile_summary(normal)}",
                        flush=True,
                    )
                    print(
                        f"[bce][viz][debug] {traj.task_name}/{traj.video_id} score_failure_gt1: "
                        f"{_percentile_summary(failure)}",
                        flush=True,
                    )
            vizs.append(viz)
            video_paths.append(video_path)

        pdf_path = os.path.join(out_dir, pdf_name)
        self.render_pdf(vizs, pdf_path, split=split)
        print(f"[bce][viz] wrote PDF -> {pdf_path}", flush=True)
        return {"videos": video_paths, "pdf": pdf_path}


# ---------------------------------------------------------------------- #
# Eval-trajectory sampling for visualization                               #
# ---------------------------------------------------------------------- #


def _sample_trajectories_for_viz(
    trajs: Sequence[BenchmarkTrajectory],
    *,
    task: str,
    split: str,
    num_trajs: int,
    seed: int,
) -> List[BenchmarkTrajectory]:
    """Sample trajectories from the eval benchmark for MP4/PDF rendering."""
    if split == "success_rollout":
        pool = [t for t in trajs if not bool(t.is_failure) and str(t.task_name) == str(task)]
        kind = "success"
    elif split == "fail_rollout":
        pool = [t for t in trajs if bool(t.is_failure) and str(t.task_name) == str(task)]
        kind = "failure"
    else:
        raise ValueError(f"unknown split {split!r}; expected success_rollout or fail_rollout")

    if not pool:
        raise RuntimeError(
            f"No {kind} trajectories in eval set for task {task!r} (split={split!r})."
        )
    rng = random.Random(int(seed))
    n = min(int(num_trajs), len(pool))
    sampled = rng.sample(pool, n)
    print(
        f"[bce][viz] sampled {n}/{len(pool)} {split} trajectories from {task}",
        flush=True,
    )
    return sampled


# ---------------------------------------------------------------------- #
# Bank-pool selection (mirrors the runner helpers)                       #
# ---------------------------------------------------------------------- #


def _select_bank_robosuite(
    bank_pool_by_task: Dict[str, List],
    eval_tasks: Sequence[str],
    fail_bank_per_task: int,
) -> List:
    out: List = []
    for task in sorted(set(eval_tasks)):
        pool = sorted(bank_pool_by_task.get(task, []), key=lambda t: str(t.video_id))
        n_take = min(len(pool), int(fail_bank_per_task))
        if n_take == 0:
            raise RuntimeError(
                f"Task {task!r}: zero GT-labeled failure trajectories available; "
                f"check --fail-train-root and --tasks."
            )
        if n_take < int(fail_bank_per_task):
            print(
                f"[bce][viz] task={task}: only {n_take} GT failure trajectories "
                f"(< --fail-bank-per-task={fail_bank_per_task}); using all.",
                flush=True,
            )
        out.extend(pool[:n_take])
    return out


# ---------------------------------------------------------------------- #
# Benchmark + bank-pool construction (robosuite)                         #
# ---------------------------------------------------------------------- #


def _build_benchmark_and_bank(args: argparse.Namespace):
    """Build the eval FailureBenchmark plus the disjoint failure bank pool.

    Returns (bench, eval_trajs, fail_bank_trajs).
    """
    from robosuite.discriminator.utils.robosuite_benchmark import (
        FailureBenchmark,
        discover_failure_bank,
    )

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

    eval_fail_keys = {str(t.video_id) for t in trajs if bool(t.is_failure)}
    all_fail = discover_failure_bank(
        data_root=args.data_root,
        tasks=[str(args.task)],
        split=args.fail_train_split,
        max_fail_per_task=None,
    )
    all_fail = [t for t in all_fail if bool(t.is_failure)]
    bank_pool = [t for t in all_fail if str(t.video_id) not in eval_fail_keys]
    bank_pool_by_task: Dict[str, List] = {}
    for t in bank_pool:
        bank_pool_by_task.setdefault(str(t.task_name), []).append(t)
    eval_tasks = sorted({str(t.task_name) for t in trajs})
    fail_bank = _select_bank_robosuite(bank_pool_by_task, eval_tasks, int(args.fail_bank_per_task))
    print(
        f"[bce][viz] robosuite eval={len(trajs)} (fail={len(eval_fail_keys)}) "
        f"bank_pool={len(bank_pool)} bank_used={len(fail_bank)}",
        flush=True,
    )
    return bench, trajs, fail_bank


# ---------------------------------------------------------------------- #
# Load BCE head from disk without re-fitting                             #
# ---------------------------------------------------------------------- #


def _bootstrap_from_ckpt(disc: BCEBenchmarkDiscriminator, ckpt_path: str) -> None:
    """Restore a fitted BCE head into ``disc`` without running fit_on_benchmark.

    Mirrors :meth:`BCEBenchmarkDiscriminator._save_checkpoint` reverse.
    Populates ``_shared_detector``, ``_detectors_per_task`` (aliased per task in
    the saved threshold table), ``_global_stats``, and ``_calibration_stats``
    so :meth:`score_trajectory` and :meth:`calibration_summary` both work.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["bce_detector"]
    detector = BCEDiscriminator(
        in_dim=int(payload["in_dim"]),
        hidden=int(payload["hidden"]),
        num_layers=int(payload["num_layers"]),
        device=str(disc.device),
    )
    detector.load_state_dict(state)
    if not detector.thresholds:
        raise RuntimeError(
            f"BCE checkpoint at {ckpt_path} has no per-task thresholds; cannot score."
        )

    disc._shared_detector = detector
    disc._detectors_per_task = {task: detector for task in detector.thresholds}

    disc._global_stats = {
        "feat_dim": int(payload["in_dim"]),
        "epochs": int(payload.get("epoch", 0)),
        "head_hidden": int(payload["hidden"]),
        "head_layers": int(payload["num_layers"]),
        "feature_source": str(payload.get("feature_source", disc.feature_source)),
        "transformer_layer": int(payload.get("transformer_layer", disc.transformer_layer)),
        "max_expert_other_ratio": payload.get("max_expert_other_ratio"),
        "num_fail_bank_trajectories": len(payload.get("fail_bank_video_ids", []) or []),
        "num_fail_bank_skipped_no_gt": None,
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
        f"[bce][viz] loaded BCE head from {ckpt_path}; tasks="
        f"{sorted(detector.thresholds)} "
        f"tau="
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(detector.thresholds.items())),
        flush=True,
    )


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="fail_rollout",
        choices=["success_rollout", "fail_rollout"],
        help="Eval split to visualize: fail_rollout (--fail-split) or "
             "success_rollout (--success-split).",
    )
    parser.add_argument("--model-ckpt", required=True, help="dyn_disc DINOv3 dynamics checkpoint .pth")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Robosuite data root containing data/<task>/<split> directories.")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", type=str, default="success_rollout-val")
    parser.add_argument("--fail-train-split", type=str, default="fail_rollout-labeled")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-trajs", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", type=str, default="bce_scores.pdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--no-flip-vertical", action="store_true",
                        help="Disable the default top/bottom flip applied to rendered frames.")

    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)

    # encoder / scoring knobs
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--camera-to-view", type=str, default=None)
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--feature-source", type=str, default="transformer",
                        choices=["encoder", "transformer"])
    parser.add_argument("--transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--quiet-fit", action="store_true")

    # BCE-specific
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-expert-other-ratio", type=float, default=1.0)
    parser.add_argument("--fail-bank-per-task", type=int, default=25)
    parser.add_argument("--save-ckpt-dir", type=str, default=None,
                        help="Where bce_head.pth is written when fitting. "
                             "Defaults to <out-dir>/checkpoints.")
    parser.add_argument("--load-ckpt", type=str, default=None,
                        help="Path to an existing bce_head.pth; skip fit and restore "
                             "the head + per-task thresholds from disk.")

    parser.add_argument("--no-debug-score-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bench, trajs, fail_bank_trajs = _build_benchmark_and_bank(args)
    sampled = _sample_trajectories_for_viz(
        trajs,
        task=str(args.task),
        split=str(args.split),
        num_trajs=int(args.num_trajs),
        seed=int(args.seed),
    )

    save_ckpt_dir = args.save_ckpt_dir or os.path.join(str(args.out_dir), "checkpoints")
    max_eo_ratio = (
        None if float(args.max_expert_other_ratio) <= 0.0 else float(args.max_expert_other_ratio)
    )
    discriminator = BCEBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        fail_bank_trajectories=fail_bank_trajs if not args.load_ckpt else [],
        fail_calib_trajectories=[],
        max_expert_other_ratio=max_eo_ratio,
        head_hidden=int(args.head_hidden),
        head_layers=int(args.head_layers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        save_ckpt_dir=None if args.load_ckpt else save_ckpt_dir,
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        visual_weight=float(args.visual_weight),
        proprio_weight=float(args.proprio_weight),
        action_weight=float(args.action_weight),
        delta=float(args.delta),
        feature_source=str(args.feature_source),
        transformer_layer=int(args.transformer_layer),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )

    try:
        if args.load_ckpt:
            if not os.path.isfile(args.load_ckpt):
                raise FileNotFoundError(f"--load-ckpt not found: {args.load_ckpt}")
            print(f"[bce][viz] loading BCE head from {args.load_ckpt}; skipping fit.", flush=True)
            _bootstrap_from_ckpt(discriminator, str(args.load_ckpt))
        else:
            print("[bce][viz] fitting BCE head on benchmark...", flush=True)
            discriminator.fit_on_benchmark(trajs)

        if str(args.task) not in discriminator._detectors_per_task:
            available = sorted(discriminator._detectors_per_task)
            raise RuntimeError(
                f"Task {args.task!r} is not present in the fitted/loaded discriminator. "
                f"Available tasks: {available}"
            )

        # Threshold: always two-class Youden on (success-calib, fail-suffix).
        if not fail_bank_trajs:
            raise RuntimeError(
                "Youden threshold requires a non-empty fail bank pool; "
                "check --fail-train-root / --fail-root."
            )
        fail_suffix_scores = _compute_fail_suffix_failure_scores_per_task(
            discriminator, fail_bank_trajs,
        )
        calib_scores = _compute_success_calib_failure_scores_per_task(
            discriminator, trajs,
            seed=int(args.seed),
            calib_fraction=float(args.calib_fraction),
        )
        s_fail = fail_suffix_scores.get(str(args.task))
        s_succ = calib_scores.get(str(args.task))
        if s_fail is None or s_fail.size == 0:
            raise RuntimeError(
                f"No usable GT-failure-suffix frames for task {args.task!r}; "
                f"cannot compute Youden threshold."
            )
        if s_succ is None or s_succ.size == 0:
            raise RuntimeError(
                f"No success-calib frames for task {args.task!r}; cannot compute "
                f"Youden threshold. Under --load-ckpt this requires the same "
                f"eval set / seed / calib_fraction as the fit run."
            )
        tau = two_class_youden_threshold(s_succ, s_fail)
        threshold_overrides = {str(args.task): tau}
        threshold_source = f"youden(n_succ={s_succ.size}, n_fail={s_fail.size})"
        print(
            f"[bce][viz][thr] youden  n_succ_calib={s_succ.size}  "
            f"n_fail_suffix={s_fail.size}  "
            f"succ[min/mean/max]=[{s_succ.min():.3f}/{s_succ.mean():.3f}/{s_succ.max():.3f}]  "
            f"fail[min/mean/max]=[{s_fail.min():.3f}/{s_fail.mean():.3f}/{s_fail.max():.3f}]  "
            f"tau={tau:.4f}",
            flush=True,
        )

        visualizer = BCEVisualizer(
            discriminator,
            camera_name=str(args.camera_name),
            fps=int(args.fps),
            border_thickness=int(args.border_thickness),
            threshold_overrides=threshold_overrides,
            threshold_source=threshold_source,
            debug_score_stats=not bool(args.no_debug_score_stats),
            flip_vertical=not bool(args.no_flip_vertical),
        )
        out_paths = visualizer.visualize(
            sampled,
            out_dir=str(args.out_dir),
            pdf_name=str(args.pdf_name),
            split=str(args.split),
        )
        print(
            f"[bce][viz] done. videos: {len(out_paths['videos'])}  pdf: {out_paths['pdf']}",
            flush=True,
        )
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
