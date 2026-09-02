"""Intervention-aware rendering layered on the existing nnPU visualizer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from benchmark.core import BenchmarkTrajectory

from robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce import (
    PUBCEVisualizer,
    PerTrajectoryViz,
    _draw_border,
    _overlay_hud_pu,
    _pad_to_even,
    _percentile_summary,
    _safe_id,
)


@dataclass
class FinetunedTrajectoryViz(PerTrajectoryViz):
    """Per-trajectory scores plus offline intervention metadata."""

    source_kind: str = "benchmark"
    intervention_frames: int = 0
    terminal_reason: Optional[str] = None
    intervention_mask: Optional[np.ndarray] = None


class FinetunedPUBCEVisualizer(PUBCEVisualizer):
    """Render benchmark and collected episodes with human frames unannotated."""

    def _score_trajectory(
        self,
        traj: BenchmarkTrajectory,
    ) -> FinetunedTrajectoryViz:
        base = super()._score_trajectory(traj)
        raw_mask = getattr(traj, "intervention_mask", None)
        intervention_mask = (
            None
            if raw_mask is None
            else np.asarray(raw_mask, dtype=np.bool_).reshape(-1)
        )
        if intervention_mask is not None and int(intervention_mask.shape[0]) != int(
            traj.num_frames
        ):
            raise ValueError(
                f"Intervention mask length {intervention_mask.shape[0]} does not match "
                f"trajectory length {traj.num_frames} for {traj.video_id}."
            )
        return FinetunedTrajectoryViz(
            **base.__dict__,
            source_kind=str(getattr(traj, "source_kind", "benchmark")),
            intervention_frames=(
                0 if intervention_mask is None else int(intervention_mask.sum())
            ),
            terminal_reason=getattr(traj, "terminal_reason", None),
            intervention_mask=intervention_mask,
        )

    def render_video(
        self,
        traj: BenchmarkTrajectory,
        viz: FinetunedTrajectoryViz,
        out_path: str,
    ) -> None:
        threshold = self._task_threshold(viz.task_name)
        images = traj.load_images(cameras=[self.camera_name])
        frames = np.asarray(images[self.camera_name], dtype=np.uint8)
        if self.flip_vertical:
            frames = frames[:, ::-1, :, :]
        frame_count = min(int(frames.shape[0]), int(viz.num_frames))
        if frame_count <= 0:
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
            for index in range(frame_count):
                image = frames[index]
                target_size = getattr(traj, "video_frame_size", None)
                if target_size is not None:
                    target_height, target_width = target_size
                    if image.shape[:2] != (int(target_height), int(target_width)):
                        image = np.asarray(
                            Image.fromarray(np.ascontiguousarray(image)).resize(
                                (int(target_width), int(target_height)),
                                Image.BILINEAR,
                            ),
                            dtype=np.uint8,
                        )
                canvas = image
                is_intervention = (
                    viz.intervention_mask is not None
                    and bool(viz.intervention_mask[index])
                )
                if not is_intervention:
                    pred_fail = bool(viz.predictions[index])
                    gt_fail = (
                        None if viz.gt_mask is None else bool(viz.gt_mask[index])
                    )
                    if pred_fail:
                        canvas = _draw_border(
                            canvas,
                            self.border_color_fail,
                            self.border_thickness,
                        )
                    canvas = _overlay_hud_pu(
                        canvas,
                        score=float(viz.step_scores[index]),
                        threshold=float(threshold),
                        pred_fail=pred_fail,
                        gt_fail=gt_fail,
                        frame_idx=index,
                        total=frame_count,
                        font=self._font,
                    )
                writer.append_data(_pad_to_even(canvas).astype(np.uint8))
        finally:
            writer.close()

    def _plot_trajectory(
        self,
        pdf: PdfPages,
        viz: FinetunedTrajectoryViz,
        *,
        threshold: float,
    ) -> None:
        frame_count = int(viz.num_frames)
        frame_indices = np.arange(frame_count)
        fig, axis = plt.subplots(figsize=(10.0, 4.5))
        axis.plot(
            frame_indices,
            viz.step_scores,
            color="#1f77b4",
            lw=1.4,
            label="PU failure score (-logit)",
        )
        if np.isfinite(threshold):
            axis.axhline(
                threshold,
                color="red",
                lw=1.2,
                ls="--",
                label=f"tau={threshold:.3f}",
            )
        for segment in viz.failure_segments:
            try:
                start = int(segment["start"])
                end = int(segment["end"])
            except Exception:
                continue
            axis.axvspan(
                max(0, start),
                min(frame_count - 1, end),
                color="red",
                alpha=0.12,
                zorder=0,
            )
        pred_mask = viz.predictions.astype(bool)
        if pred_mask.any():
            score_minimum = float(np.nanmin(viz.step_scores))
            axis.plot(
                frame_indices,
                np.where(pred_mask, score_minimum, np.nan),
                color="red",
                lw=4,
                alpha=0.6,
                label="predicted fail frames",
            )
        if viz.first_gt_failure_frame is not None:
            axis.axvline(
                int(viz.first_gt_failure_frame),
                color="darkred",
                lw=0.9,
                ls=":",
                label="first GT fail",
            )
        if viz.first_pred_failure_frame is not None:
            axis.axvline(
                int(viz.first_pred_failure_frame),
                color="orange",
                lw=0.9,
                ls=":",
                label="first PRED fail",
            )
        axis.set_xlim(0, max(frame_count - 1, 1))
        axis.set_xlabel("frame")
        axis.set_ylabel("PU failure score (-logit)")
        axis.set_title(
            f"[{viz.task_name}] {viz.video_id}  (T={frame_count})",
            fontsize=11,
            loc="left",
        )
        axis.grid(True, alpha=0.2)

        handles, labels = axis.get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        if viz.gt_mask is not None or viz.failure_segments:
            unique_handles.append(
                Patch(facecolor="red", alpha=0.12, label="GT failure segment")
            )
            unique_labels.append("GT failure segment")
        axis.legend(
            unique_handles,
            unique_labels,
            loc="upper left",
            fontsize=8,
            framealpha=0.85,
        )

        ground_truth_frames = (
            str(int(viz.gt_mask.sum())) if viz.gt_mask is not None else "n/a"
        )
        delay = "n/a"
        if (
            viz.first_gt_failure_frame is not None
            and viz.first_pred_failure_frame is not None
        ):
            delay = (
                f"{int(viz.first_pred_failure_frame) - int(viz.first_gt_failure_frame):+d}"
            )
        stats = (
            f"pred_frames={int(viz.predictions.sum())}  gt_frames={ground_truth_frames}  "
            f"first_pred={viz.first_pred_failure_frame}  "
            f"first_gt={viz.first_gt_failure_frame}  delay(pred-gt)={delay}"
        )
        if viz.source_kind == "offline":
            stats += (
                f"  interventions={viz.intervention_frames}  "
                f"terminal_reason={viz.terminal_reason or 'unknown'}  "
                "video_annotations=hidden_on_intervention_frames"
            )
        axis.text(
            0.01,
            -0.23,
            stats,
            transform=axis.transAxes,
            fontsize=9,
            family="monospace",
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def visualize(
        self,
        trajectories: list[BenchmarkTrajectory],
        *,
        out_dir: str,
        pdf_name: str,
        split: str = "fail_rollout",
    ) -> dict[str, object]:
        if not trajectories:
            raise RuntimeError("No trajectories provided for visualization.")
        videos_dir = os.path.join(out_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        vizs: list[FinetunedTrajectoryViz] = []
        video_paths: list[str] = []
        for trajectory in trajectories:
            viz = self._score_trajectory(trajectory)
            if split == "both":
                kind = "fail_rollout" if bool(trajectory.is_failure) else "success_rollout"
                target_dir = os.path.join(videos_dir, kind)
            elif split in {"online", "online-success"}:
                target_dir = os.path.join(videos_dir, split)
            else:
                target_dir = videos_dir
            os.makedirs(target_dir, exist_ok=True)
            video_path = os.path.join(target_dir, f"{_safe_id(trajectory.video_id)}.mp4")
            self.render_video(trajectory, viz, video_path)
            annotation_mode = (
                "hidden_on_intervention_frames"
                if viz.intervention_frames > 0
                else "shown"
            )
            print(
                f"[pu_bce][viz] {trajectory.task_name}/{trajectory.video_id} "
                f"T={viz.num_frames} pred_frames={int(viz.predictions.sum())} "
                f"video_annotations={annotation_mode} -> {video_path}",
                flush=True,
            )
            if self.debug_score_stats:
                print(
                    f"[pu_bce][viz][debug] {trajectory.task_name}/{trajectory.video_id} "
                    f"score_all: {_percentile_summary(viz.step_scores)}",
                    flush=True,
                )
                if viz.gt_mask is not None:
                    normal_scores = viz.step_scores[viz.gt_mask == 0]
                    failure_scores = viz.step_scores[viz.gt_mask == 1]
                    print(
                        f"[pu_bce][viz][debug] {trajectory.task_name}/"
                        f"{trajectory.video_id} score_normal_gt0: "
                        f"{_percentile_summary(normal_scores)}",
                        flush=True,
                    )
                    print(
                        f"[pu_bce][viz][debug] {trajectory.task_name}/"
                        f"{trajectory.video_id} score_failure_gt1: "
                        f"{_percentile_summary(failure_scores)}",
                        flush=True,
                    )
            vizs.append(viz)
            video_paths.append(video_path)

        pdf_path = os.path.join(out_dir, pdf_name)
        self.render_pdf(vizs, pdf_path, split=split)
        print(f"[pu_bce][viz] wrote PDF -> {pdf_path}", flush=True)
        return {"videos": video_paths, "pdf": pdf_path}


__all__ = ["FinetunedPUBCEVisualizer", "FinetunedTrajectoryViz"]
