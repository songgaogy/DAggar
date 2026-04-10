from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass

import h5py
import matplotlib
import numpy as np
from hydra.utils import to_absolute_path

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.task_registry import normalize_task_name, resolve_checkpoint_task_name
from robosuite.discriminator.lpb_score.analysis.attribution import (
    TERM_COLORS,
    TERM_LABELS,
    TERM_SHORT_LABELS,
    map_term_values_to_frames,
    ordered_term_keys,
    summarize_trajectory_term_attribution,
)
from robosuite.discriminator.lpb_score.app.pipeline import (
    build_dsm_discriminator,
    build_flow_encoder,
    now_tag,
    select_split_refs,
)
from robosuite.discriminator.lpb_score.core.dataset import (
    EncodedTrajectoryRef,
    LatentTrajectory,
    build_cached_splits,
    load_latent_trajectories,
)
from robosuite.discriminator.utils.types import VideoRenderRecord
from robosuite.discriminator.utils.video_io import create_vscode_mp4_writer
from robosuite.discriminator.utils.visualization import (
    draw_detection_overlay,
    fix_robosuite_frame_orientation,
    map_step_values_to_frames,
    select_camera_frame,
)


@dataclass(frozen=True)
class SuboptimalDemoRef:
    task_name: str
    split: str
    file_path: str
    demo_key: str
    sub_start: int
    sub_stop: int


@dataclass(frozen=True)
class LabeledLatentTrajectory:
    trajectory: LatentTrajectory
    labels: np.ndarray
    split: str


SCORE_MODE_ORDER = (
    "t1_positive_energy",
    "t2_negative_margin",
    "t3_weighted_combo",
)
SCORE_MODE_SHORT_LABELS = {
    "t1_positive_energy": "T1",
    "t2_negative_margin": "T2",
    "t3_weighted_combo": "T3",
}
SCORE_MODE_COLORS = {
    "t1_positive_energy": "#1f77b4",
    "t2_negative_margin": "#2ca02c",
    "t3_weighted_combo": "#9467bd",
}


def _sample_refs(
    refs,
    num_samples: int,
    seed: int,
):
    """Choose a reproducible subset of trajectories to render."""
    if not refs:
        return []
    num = min(len(refs), max(1, int(num_samples)))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(refs), size=num, replace=False)
    indices = np.sort(indices)
    return [refs[int(idx)] for idx in indices.tolist()]


def _select_thumbnail_indices(
    num_frames: int,
    num_thumbnails: int,
) -> list[int]:
    """Spread thumbnails across the video timeline."""
    if int(num_frames) <= 0:
        return []

    target = min(int(num_frames), max(1, int(num_thumbnails)))
    if target == 1:
        return [int(num_frames) // 2]

    return np.linspace(0, int(num_frames) - 1, num=target, dtype=np.int64).astype(int).tolist()


def _build_plot_frames(
    prepared,
    *,
    frame_indices: list[int],
    camera_names: list[str],
    vis_camera_name: str | None,
    flip_vertical: bool,
) -> list[np.ndarray]:
    """Collect a few representative frames for the PDF summary."""
    frames: list[np.ndarray] = []
    for frame_id in frame_indices:
        frame_rgb = select_camera_frame(
            prepared.images_hwc,
            frame_id=int(frame_id),
            camera_names=camera_names,
            vis_camera_name=vis_camera_name,
        )
        frame_rgb = fix_robosuite_frame_orientation(
            frame_rgb=frame_rgb,
            flip_vertical=bool(flip_vertical),
        )
        frames.append(np.asarray(frame_rgb, dtype=np.uint8))
    return frames


def _save_failure_plot_pdf(
    *,
    pdf_path: str,
    report_pdf: PdfPages | None,
    ref,
    trajectory_id: int,
    fps: int,
    frame_scores: np.ndarray,
    frame_thresholds: np.ndarray,
    frame_scores_by_mode: dict[str, np.ndarray],
    frame_thresholds_by_mode: dict[str, np.ndarray],
    active_score_mode: str,
    thumbnail_frames: list[np.ndarray],
    thumbnail_indices: list[int],
    delta_final: float,
    threshold_final: float,
    aggregate_contribution_shares: dict[str, np.ndarray],
    first_crossing_index: int | None,
    first_crossing_dominant_term: str | None,
    gt_fail_mask: np.ndarray | None = None,
) -> None:
    """Save a PDF page with active score, T1/T2/T3 traces, and term attributions."""
    num_frames = int(np.asarray(frame_scores).shape[0])
    if num_frames <= 0:
        return

    fps_value = max(1, int(fps))
    times = np.arange(num_frames, dtype=np.float32) / float(fps_value)
    num_cols = max(1, len(thumbnail_frames))

    fig = plt.figure(figsize=(4.2 * num_cols, 11.4), constrained_layout=True)
    grid = fig.add_gridspec(4, num_cols, height_ratios=[1.0, 1.5, 1.5, 1.7])
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(num_cols)]
    score_ax = fig.add_subplot(grid[1, :])
    mode_ax = fig.add_subplot(grid[2, :])
    contrib_ax = fig.add_subplot(grid[3, :])

    for idx, (thumb_ax, image_rgb, frame_id) in enumerate(zip(top_axes, thumbnail_frames, thumbnail_indices)):
        thumb_ax.imshow(image_rgb)
        thumb_ax.set_axis_off()
        for spine in thumb_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("#444444")
        thumb_ax.set_title(
            f"frame {int(frame_id) + 1}\nt={times[int(frame_id)]:.2f}s",
            fontsize=10,
            color="#444444",
            pad=6,
        )
        if idx < len(top_axes):
            score_ax.axvline(times[int(frame_id)], color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)
            mode_ax.axvline(times[int(frame_id)], color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)
            contrib_ax.axvline(times[int(frame_id)], color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)

    score_ax.plot(times, np.asarray(frame_scores, dtype=np.float32), color="#1f77b4", linewidth=2.2, label="lambda")
    score_ax.plot(
        times,
        np.asarray(frame_thresholds, dtype=np.float32),
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="--",
        label="threshold",
    )
    if gt_fail_mask is not None:
        gt_mask = np.asarray(gt_fail_mask, dtype=bool)
        if gt_mask.shape[0] == num_frames and np.any(gt_mask):
            in_region = False
            start_idx = 0
            for frame_idx, flag in enumerate(gt_mask.tolist()):
                if flag and not in_region:
                    start_idx = int(frame_idx)
                    in_region = True
                if in_region and ((not flag) or frame_idx == num_frames - 1):
                    end_idx = int(frame_idx if not flag else frame_idx + 1)
                    score_ax.axvspan(
                        times[start_idx],
                        times[min(end_idx - 1, num_frames - 1)],
                        color="#d62728",
                        alpha=0.12,
                    )
                    mode_ax.axvspan(
                        times[start_idx],
                        times[min(end_idx - 1, num_frames - 1)],
                        color="#d62728",
                        alpha=0.10,
                    )
                    contrib_ax.axvspan(
                        times[start_idx],
                        times[min(end_idx - 1, num_frames - 1)],
                        color="#d62728",
                        alpha=0.10,
                    )
                    in_region = False
    if first_crossing_index is not None and 0 <= int(first_crossing_index) < num_frames:
        score_ax.axvline(
            times[int(first_crossing_index)],
            color="#d62728",
            linewidth=1.5,
            linestyle="-.",
            alpha=0.95,
            label="first crossing",
        )
        mode_ax.axvline(
            times[int(first_crossing_index)],
            color="#d62728",
            linewidth=1.4,
            linestyle="-.",
            alpha=0.95,
        )

    score_ax.set_title(
        f"Trajectory {int(trajectory_id):02d} | {ref.task_name} | {ref.demo_key}",
        fontsize=13,
        pad=10,
    )
    score_ax.set_xlabel("Time (s)")
    score_ax.set_ylabel("Lambda")
    score_ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    score_ax.legend(loc="upper left")
    score_ax.text(
        0.995,
        0.02,
        f"delta_final={float(delta_final):.2f}  threshold_final={float(threshold_final):.3f}",
        transform=score_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )

    for score_mode in SCORE_MODE_ORDER:
        score_values = np.asarray(frame_scores_by_mode.get(score_mode, []), dtype=np.float32).reshape(-1)
        if score_values.size != num_frames:
            continue
        threshold_values = np.asarray(frame_thresholds_by_mode.get(score_mode, []), dtype=np.float32).reshape(-1)
        color = SCORE_MODE_COLORS.get(score_mode, "#555555")
        short_label = SCORE_MODE_SHORT_LABELS.get(score_mode, score_mode)
        mode_ax.plot(
            times,
            score_values,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"{short_label} score",
        )
        if threshold_values.size == num_frames:
            mode_ax.plot(
                times,
                threshold_values,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label=f"{short_label} th",
            )
    mode_ax.set_title(
        f"T1/T2/T3 Aggregate Scores | active={SCORE_MODE_SHORT_LABELS.get(active_score_mode, active_score_mode)}",
        fontsize=12,
        pad=8,
    )
    mode_ax.set_xlabel("Time (s)")
    mode_ax.set_ylabel("Aggregate Score")
    mode_ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    mode_ax.legend(loc="upper left", ncol=3)

    term_keys = ordered_term_keys(aggregate_contribution_shares)
    if term_keys:
        share_stack = [
            100.0 * np.asarray(aggregate_contribution_shares[key], dtype=np.float32)
            for key in term_keys
        ]
        contrib_ax.stackplot(
            times,
            *share_stack,
            labels=[TERM_LABELS.get(key, key) for key in term_keys],
            colors=[TERM_COLORS.get(key, "#777777") for key in term_keys],
            alpha=0.86,
        )
        if first_crossing_index is not None and 0 <= int(first_crossing_index) < num_frames:
            contrib_ax.axvline(
                times[int(first_crossing_index)],
                color="#111111",
                linewidth=1.3,
                linestyle="-.",
                alpha=0.95,
            )
        contrib_ax.set_ylim(0.0, 100.0)
        contrib_ax.set_ylabel("Contribution Share (%)")
        contrib_ax.set_xlabel("Time (s)")
        contrib_ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
        contrib_ax.legend(loc="upper left", ncol=max(1, min(2, len(term_keys))))
        if first_crossing_dominant_term:
            contrib_ax.text(
                0.995,
                0.98,
                f"first crossing dominant: {TERM_LABELS.get(first_crossing_dominant_term, first_crossing_dominant_term)}",
                transform=contrib_ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
            )
    else:
        contrib_ax.text(
            0.5,
            0.5,
            "No term attribution available",
            transform=contrib_ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
        )
        contrib_ax.set_axis_off()

    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if report_pdf is not None:
        report_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _save_threshold_distribution_plots(
    *,
    threshold_dir: str,
    calibration_lambdas_by_task: dict[str, np.ndarray],
    fail_lambdas_by_task: dict[str, np.ndarray],
    thresholds_by_task: dict[str, float],
) -> dict[str, str]:
    """Save one calibration-threshold distribution plot per task."""
    os.makedirs(threshold_dir, exist_ok=True)
    saved_paths: dict[str, str] = {}

    task_names = sorted(set(calibration_lambdas_by_task.keys()) | set(fail_lambdas_by_task.keys()))
    for task_name in task_names:
        calibration_values = np.asarray(
            calibration_lambdas_by_task.get(task_name, []),
            dtype=np.float32,
        ).reshape(-1)
        fail_values = np.asarray(
            fail_lambdas_by_task.get(task_name, []),
            dtype=np.float32,
        ).reshape(-1)
        if calibration_values.size == 0 and fail_values.size == 0:
            continue

        threshold = float(thresholds_by_task.get(task_name, np.nan))
        combined_values = np.concatenate(
            [
                values
                for values in [calibration_values, fail_values]
                if values.size > 0
            ],
            axis=0,
        ).astype(np.float32)
        num_bins = max(10, min(60, int(np.sqrt(combined_values.size))))

        fig = plt.figure(figsize=(8.4, 6.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.3])
        hist_ax = fig.add_subplot(grid[0, 0])
        cdf_ax = fig.add_subplot(grid[1, 0])

        if calibration_values.size > 0:
            hist_ax.hist(
                calibration_values,
                bins=num_bins,
                color="#4c78a8",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.62,
                label="calibration",
            )
        if fail_values.size > 0:
            hist_ax.hist(
                fail_values,
                bins=num_bins,
                color="#f58518",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.48,
                label="fail_rollout",
            )
        if np.isfinite(threshold):
            hist_ax.axvline(
                threshold,
                color="#d62728",
                linewidth=2.0,
                linestyle="--",
                label=f"threshold={threshold:.3f}",
            )
        if calibration_values.size > 0 or fail_values.size > 0 or np.isfinite(threshold):
            hist_ax.legend(loc="upper right")
        hist_ax.set_title(f"{task_name} Lambda Distribution Comparison", fontsize=13, pad=10)
        hist_ax.set_ylabel("Count")
        hist_ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.8)
        calibration_mean = float(np.mean(calibration_values)) if calibration_values.size > 0 else float("nan")
        fail_mean = float(np.mean(fail_values)) if fail_values.size > 0 else float("nan")
        hist_ax.text(
            0.995,
            0.98,
            (
                f"cal_n={int(calibration_values.size)}  "
                f"cal_mean={calibration_mean:.3f}  "
                f"fail_n={int(fail_values.size)}"
                + (
                    f"  fail_mean={fail_mean:.3f}"
                    if fail_values.size > 0
                    else ""
                )
            ),
            transform=hist_ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
        )

        if calibration_values.size > 0:
            sorted_calibration = np.sort(calibration_values.astype(np.float64))
            calibration_cdf = np.arange(1, sorted_calibration.size + 1, dtype=np.float64) / float(sorted_calibration.size)
            cdf_ax.plot(sorted_calibration, calibration_cdf, color="#1f77b4", linewidth=2.0, label="calibration")
        if fail_values.size > 0:
            sorted_fail = np.sort(fail_values.astype(np.float64))
            fail_cdf = np.arange(1, sorted_fail.size + 1, dtype=np.float64) / float(sorted_fail.size)
            cdf_ax.plot(sorted_fail, fail_cdf, color="#f58518", linewidth=2.0, label="fail_rollout")
        if np.isfinite(threshold):
            cdf_ax.axvline(threshold, color="#d62728", linewidth=1.8, linestyle="--")
            calibration_quantile = (
                float(np.mean(calibration_values.astype(np.float64) <= threshold))
                if calibration_values.size > 0
                else float("nan")
            )
            fail_quantile = (
                float(np.mean(fail_values.astype(np.float64) <= threshold))
                if fail_values.size > 0
                else float("nan")
            )
            cdf_ax.text(
                0.995,
                0.08,
                (
                    f"cal_q={calibration_quantile:.3f}"
                    + (f"  fail_q={fail_quantile:.3f}" if np.isfinite(fail_quantile) else "")
                ),
                transform=cdf_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
            )
        cdf_ax.set_xlabel("Lambda")
        cdf_ax.set_ylabel("ECDF")
        cdf_ax.set_ylim(0.0, 1.02)
        cdf_ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.8)
        if calibration_values.size > 0 or fail_values.size > 0:
            cdf_ax.legend(loc="lower right")

        safe_task_name = str(task_name).replace(os.sep, "_")
        plot_path = os.path.join(threshold_dir, f"{safe_task_name}_threshold_distribution.png")
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths[str(task_name)] = plot_path

    return saved_paths


def _list_suboptimal_demo_refs(cfg: DictConfig) -> tuple[list[SuboptimalDemoRef], dict[str, dict[str, int]]]:
    root_dir = to_absolute_path(str(cfg.suboptimal.root_dir))
    train_count = int(cfg.suboptimal.train_count_per_task)
    eval_count = int(cfg.suboptimal.eval_count_per_task)
    total_required = train_count + eval_count
    seed = int(cfg.suboptimal.seed)

    refs: list[SuboptimalDemoRef] = []
    split_summary: dict[str, dict[str, int]] = {}

    for task_offset, task_name_raw in enumerate(list(cfg.suboptimal.tasks)):
        task_name = normalize_task_name(str(task_name_raw))
        sub_dir = os.path.join(root_dir, f"{resolve_checkpoint_task_name(task_name)}_allview")
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"Missing suboptimal directory for {task_name}: {sub_dir}")

        task_refs: list[SuboptimalDemoRef] = []
        for file_path in sorted(glob.glob(os.path.join(sub_dir, "*.hdf5"))):
            with h5py.File(file_path, "r") as file_handle:
                if "demos" not in file_handle:
                    continue
                for demo_key in sorted(file_handle["demos"].keys()):
                    demo = file_handle["demos"][demo_key]
                    task_refs.append(
                        SuboptimalDemoRef(
                            task_name=task_name,
                            split="",
                            file_path=file_path,
                            demo_key=demo_key,
                            sub_start=int(np.asarray(demo["sub_start"])[()]),
                            sub_stop=int(np.asarray(demo["sub_stop"])[()]),
                        )
                    )

        if len(task_refs) < total_required:
            raise ValueError(
                f"Task {task_name} requires at least {total_required} suboptimal demos, found {len(task_refs)}."
            )

        rng = np.random.default_rng(seed + task_offset * 97)
        indices = np.arange(len(task_refs))
        rng.shuffle(indices)
        selected_refs = [task_refs[int(idx)] for idx in indices[:total_required]]

        split_summary[task_name] = {"train": 0, "eval": 0}
        for local_idx, ref in enumerate(selected_refs):
            split_name = "train" if local_idx < train_count else "eval"
            refs.append(
                SuboptimalDemoRef(
                    task_name=ref.task_name,
                    split=split_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                    sub_start=int(ref.sub_start),
                    sub_stop=int(ref.sub_stop),
                )
            )
            split_summary[task_name][split_name] += 1

    return refs, split_summary


def _encode_suboptimal_refs(
    refs: list[SuboptimalDemoRef],
    *,
    encoder,
    task_to_index: dict[str, int],
    horizon: int,
    batch_size: int,
) -> tuple[list[LabeledLatentTrajectory], dict[str, object]]:
    encoded: list[LabeledLatentTrajectory] = []
    dropped: list[dict[str, object]] = []
    prepared_batch = []
    ref_batch: list[SuboptimalDemoRef] = []

    def flush_batch() -> None:
        nonlocal prepared_batch, ref_batch
        if not prepared_batch:
            return
        encoded_batch = encoder.encode_prepared_demos(prepared_batch)
        encoded_lookup = {
            (item.file_path, item.demo_key): item
            for item in encoded_batch
        }
        for ref in ref_batch:
            key = (ref.file_path, ref.demo_key)
            item = encoded_lookup.get(key)
            if item is None:
                raise KeyError(f"Missing encoded suboptimal demo for {key}")
            length = min(int(item.latents.shape[0]), int(item.actions.shape[0]))
            valid_len = length - int(horizon)
            if valid_len <= 0:
                dropped.append(
                    {
                        "task_name": ref.task_name,
                        "split": ref.split,
                        "file_path": ref.file_path,
                        "demo_key": ref.demo_key,
                        "reason": "too_short_for_horizon",
                    }
                )
                continue

            start = int(np.clip(ref.sub_start, 0, valid_len))
            stop = int(np.clip(ref.sub_stop, 0, valid_len))
            if stop <= start:
                dropped.append(
                    {
                        "task_name": ref.task_name,
                        "split": ref.split,
                        "file_path": ref.file_path,
                        "demo_key": ref.demo_key,
                        "reason": "empty_positive_interval_after_clip",
                        "sub_start": int(ref.sub_start),
                        "sub_stop": int(ref.sub_stop),
                        "valid_len": int(valid_len),
                    }
                )
                continue

            labels = np.zeros((valid_len,), dtype=np.int64)
            labels[start:stop] = 1
            encoded.append(
                LabeledLatentTrajectory(
                    trajectory=LatentTrajectory(
                        latents=np.asarray(item.latents, dtype=np.float32),
                        actions=np.asarray(item.actions, dtype=np.float32),
                        task_name=ref.task_name,
                        task_index=int(task_to_index[ref.task_name]),
                        data_type="suboptimal",
                        data_type_index=-1,
                        split=ref.split,
                        file_path=ref.file_path,
                        demo_key=ref.demo_key,
                    ),
                    labels=labels,
                    split=ref.split,
                )
            )
        prepared_batch = []
        ref_batch = []

    for idx, ref in enumerate(refs):
        prepared_batch.append(
            encoder.load_demo_raw(
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
        )
        ref_batch.append(ref)
        if len(prepared_batch) >= batch_size:
            flush_batch()
        if (idx + 1) % 20 == 0 or (idx + 1) == len(refs):
            print(f"[lpb_score visualize] encoded_suboptimal {idx + 1}/{len(refs)}")

    flush_batch()
    summary = {
        "num_requested": int(len(refs)),
        "num_encoded": int(len(encoded)),
        "num_dropped": int(len(dropped)),
        "dropped": dropped,
    }
    return encoded, summary


def _load_visualization_targets(
    cfg: DictConfig,
    *,
    encoder,
    cached_splits: dict[str, list[EncodedTrajectoryRef]],
    task_to_index: dict[str, int],
    seed: int,
    num_videos: int,
    horizon: int,
) -> tuple[list[object], list[LatentTrajectory], list[np.ndarray | None], dict[str, object]]:
    data_source = str(getattr(cfg.visualization, "data_source", "suboptimal"))
    if data_source != "suboptimal":
        fail_refs = select_split_refs(
            cached_splits=cached_splits,
            split_name=str(cfg.eval.fail_eval_split),
            data_types=list(cfg.eval.fail_eval_data_types),
        )
        selected_fail_refs = _sample_refs(
            refs=fail_refs,
            num_samples=int(num_videos),
            seed=int(seed),
        )
        if not selected_fail_refs:
            raise RuntimeError("No fail trajectories selected for visualization.")
        fail_trajectories = load_latent_trajectories(selected_fail_refs)
        return (
            list(selected_fail_refs),
            list(fail_trajectories),
            [None for _ in range(len(fail_trajectories))],
            {
                "data_source": data_source,
                "split_name": str(cfg.eval.fail_eval_split),
                "num_available": int(len(fail_refs)),
                "num_selected": int(len(selected_fail_refs)),
            },
        )

    requested_split = str(cfg.suboptimal.source_split)
    suboptimal_refs, suboptimal_split_summary = _list_suboptimal_demo_refs(cfg)
    source_refs = [ref for ref in suboptimal_refs if str(ref.split) == requested_split]
    selected_refs = _sample_refs(
        refs=source_refs,
        num_samples=int(num_videos),
        seed=int(seed),
    )
    if not selected_refs:
        raise RuntimeError(f"No suboptimal trajectories selected for visualization from split={requested_split}.")

    labeled_trajectories, encode_summary = _encode_suboptimal_refs(
        refs=selected_refs,
        encoder=encoder,
        task_to_index=task_to_index,
        horizon=int(horizon),
        batch_size=int(cfg.data.encode_demo_batch_size),
    )
    selected_lookup = {
        (ref.file_path, ref.demo_key): ref
        for ref in selected_refs
    }
    labeled_lookup = {
        (item.trajectory.file_path, item.trajectory.demo_key): item
        for item in labeled_trajectories
    }

    ordered_refs: list[object] = []
    ordered_trajectories: list[LatentTrajectory] = []
    ordered_labels: list[np.ndarray | None] = []
    for ref in selected_refs:
        key = (ref.file_path, ref.demo_key)
        labeled = labeled_lookup.get(key)
        if labeled is None:
            continue
        ordered_refs.append(selected_lookup[key])
        ordered_trajectories.append(labeled.trajectory)
        ordered_labels.append(np.asarray(labeled.labels, dtype=np.int64))

    if not ordered_refs:
        raise RuntimeError("No encoded suboptimal trajectories remain for visualization.")

    return (
        ordered_refs,
        ordered_trajectories,
        ordered_labels,
        {
            "data_source": data_source,
            "split_name": requested_split,
            "num_available": int(len(source_refs)),
            "num_selected": int(len(selected_refs)),
            "suboptimal_split_summary": suboptimal_split_summary,
            "suboptimal_encode_summary": encode_summary,
        },
    )


def run_visualize(cfg: DictConfig) -> None:
    detector = None
    encoder = build_flow_encoder(cfg)

    try:
        cached_splits, split_summary, task_to_index = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=int(cfg.seed),
            build_missing_cache=False,
        )
        bank_refs = select_split_refs(
            cached_splits=cached_splits,
            split_name=str(cfg.eval.bank_split),
            data_types=list(cfg.eval.bank_data_types),
        )
        bank_size = int(getattr(cfg.visualization, "bank_size", -1))
        calibration_seed = int(getattr(cfg.visualization, "calibration_seed", cfg.seed))
        sampled_bank_refs = (
            _sample_refs(
                refs=bank_refs,
                num_samples=bank_size,
                seed=calibration_seed,
            )
            if bank_size > 0
            else list(bank_refs)
        )
        bank_trajectories = load_latent_trajectories(sampled_bank_refs)
        calibration_trajectories = list(bank_trajectories)
        if not sampled_bank_refs:
            raise RuntimeError("No bank trajectories found for visualization.")
        fail_distribution_refs = select_split_refs(
            cached_splits=cached_splits,
            split_name=str(cfg.eval.fail_eval_split),
            data_types=list(cfg.eval.fail_eval_data_types),
        )
        fail_distribution_trajectories = load_latent_trajectories(fail_distribution_refs)

        detector = build_dsm_discriminator(cfg)
        selected_refs, target_trajectories, gt_label_sequences, target_summary = _load_visualization_targets(
            cfg,
            encoder=encoder,
            cached_splits=cached_splits,
            task_to_index=task_to_index,
            seed=int(cfg.seed),
            num_videos=int(cfg.visualization.num_videos),
            horizon=int(detector.extractor.action_horizon),
        )
        calibration_summary = detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )

        run_dir = os.path.join(to_absolute_path(str(cfg.save_dir)), f"run_{now_tag()}")
        os.makedirs(run_dir, exist_ok=True)
        threshold_dir = os.path.join(run_dir, "threshold")
        threshold_plot_paths = _save_threshold_distribution_plots(
            threshold_dir=threshold_dir,
            calibration_lambdas_by_task=detector.get_calibration_lambdas_by_task(),
            fail_lambdas_by_task=detector.collect_lambdas_by_task(fail_distribution_trajectories),
            thresholds_by_task=detector.get_thresholds_by_task(),
        )
        report_pdf_path = os.path.join(run_dir, "failure_report.pdf")
        report_pdf = PdfPages(report_pdf_path) if bool(cfg.visualization.save_pdf) else None

        print("\n[INFO] start discriminating:\n")
        records: list[VideoRenderRecord] = []
        try:
            for traj_idx, (ref, latent_traj, gt_labels) in enumerate(
                zip(selected_refs, target_trajectories, gt_label_sequences)
            ):
                use_adaptive = bool(getattr(cfg.online, "adaptive_delta", False)) and gt_labels is not None
                result = detector.detect_trajectory(
                    latent_traj,
                    labels=gt_labels if use_adaptive else None,
                    adaptive_threshold=use_adaptive,
                    delta_min=float(getattr(cfg.online, "delta_min", 0.0)),
                    delta_max=float(getattr(cfg.online, "delta_max", 100.0)),
                    warmup_steps=int(getattr(cfg.online, "warmup_steps", 0)),
                    update_interval=int(getattr(cfg.online, "update_interval", 1)),
                )

                prepared = encoder.load_demo_raw(
                    task_name=ref.task_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                )
                num_frames = int(prepared.images_hwc.shape[0])
                frame_preds = map_step_values_to_frames(
                    result.predictions,
                    num_frames=num_frames,
                    tail_fill=float(result.predictions[-1]) if result.predictions.size > 0 else 0.0,
                ).astype(np.int64)
                frame_scores = map_step_values_to_frames(
                    result.aggregate_scores,
                    num_frames=num_frames,
                    tail_fill=float(result.aggregate_scores[-1]) if result.aggregate_scores.size > 0 else 0.0,
                )
                frame_thresholds = map_step_values_to_frames(
                    result.thresholds,
                    num_frames=num_frames,
                    tail_fill=float(result.thresholds[-1]) if result.thresholds.size > 0 else 0.0,
                )
                frame_gt_fail = (
                    map_step_values_to_frames(
                        np.asarray(gt_labels, dtype=np.float32),
                        num_frames=num_frames,
                        tail_fill=float(np.asarray(gt_labels, dtype=np.float32)[-1]),
                    ).astype(np.int64)
                    if gt_labels is not None and np.asarray(gt_labels).size > 0
                    else np.zeros((num_frames,), dtype=np.int64)
                )
                aggregate_contribution_shares = result.metadata.get("aggregate_contribution_shares", {}) or {}
                frame_aggregate_share_by_term = map_term_values_to_frames(
                    aggregate_contribution_shares,
                    num_frames=num_frames,
                )
                aggregate_scores_by_mode = result.metadata.get("aggregate_scores_by_mode", {}) or {}
                threshold_by_score = result.metadata.get("threshold_by_score", {}) or {}
                active_score_mode = str(result.metadata.get("score_mode", detector.score_mode))
                frame_scores_by_mode = {
                    score_mode: map_step_values_to_frames(
                        np.asarray(values, dtype=np.float32),
                        num_frames=num_frames,
                        tail_fill=float(np.asarray(values, dtype=np.float32)[-1]),
                    )
                    for score_mode, values in aggregate_scores_by_mode.items()
                    if np.asarray(values).size > 0
                }
                frame_thresholds_by_mode: dict[str, np.ndarray] = {}
                for score_mode in SCORE_MODE_ORDER:
                    if score_mode == active_score_mode:
                        threshold_steps = np.asarray(result.thresholds, dtype=np.float32)
                    else:
                        scalar_threshold = float(threshold_by_score.get(score_mode, np.nan))
                        ref_steps = np.asarray(aggregate_scores_by_mode.get(score_mode, []), dtype=np.float32)
                        if ref_steps.size == 0:
                            continue
                        threshold_steps = np.full(ref_steps.shape, scalar_threshold, dtype=np.float32)
                    if threshold_steps.size == 0:
                        continue
                    frame_thresholds_by_mode[score_mode] = map_step_values_to_frames(
                        threshold_steps,
                        num_frames=num_frames,
                        tail_fill=float(threshold_steps[-1]),
                    )
                term_summary = summarize_trajectory_term_attribution(result)

                stem = (
                    f"traj{traj_idx:02d}_{ref.task_name}_"
                    f"{os.path.basename(ref.file_path).replace('.hdf5', '')}_{ref.demo_key}"
                )
                video_path = os.path.join(run_dir, f"{stem}.mp4")
                writer = create_vscode_mp4_writer(video_path, fps=int(cfg.visualization.fps))
                try:
                    for frame_id in range(num_frames):
                        frame_rgb = select_camera_frame(
                            prepared.images_hwc,
                            frame_id=frame_id,
                            camera_names=encoder.camera_names,
                            vis_camera_name=str(cfg.visualization.camera_name) if str(cfg.visualization.camera_name) else None,
                        )
                        frame_rgb = fix_robosuite_frame_orientation(
                            frame_rgb=frame_rgb,
                            flip_vertical=bool(cfg.visualization.flip_vertical),
                        )
                        footer_lines = [
                            " ".join(
                                [
                                    (
                                        f"{SCORE_MODE_SHORT_LABELS.get(score_mode, score_mode)} "
                                        f"{float(frame_scores_by_mode[score_mode][frame_id]):.2f}"
                                    )
                                    for score_mode in SCORE_MODE_ORDER
                                    if score_mode in frame_scores_by_mode
                                ]
                            ),
                            (
                                f"{SCORE_MODE_SHORT_LABELS.get(active_score_mode, active_score_mode)} "
                                f"{float(frame_scores[frame_id]):.2f}|{float(frame_thresholds[frame_id]):.2f}"
                            ),
                        ]
                        frame = draw_detection_overlay(
                            frame_rgb,
                            frame_id=frame_id,
                            pred_fail_flag=bool(frame_preds[frame_id]),
                            gt_fail_flag=bool(frame_gt_fail[frame_id]),
                            aggregate_score=float(frame_scores[frame_id]),
                            threshold=float(frame_thresholds[frame_id]),
                            detector_name=detector.name,
                            footer_lines=footer_lines,
                            border_thickness=int(cfg.visualization.border_thickness),
                            banner_font_scale=float(cfg.visualization.banner_font_scale),
                            banner_thickness=int(cfg.visualization.banner_thickness),
                            footer_font_scale=float(cfg.visualization.footer_font_scale),
                            footer_thickness=int(cfg.visualization.footer_thickness),
                        )
                        writer.append_data(frame)
                finally:
                    writer.close()

                plot_pdf_path = None
                if bool(cfg.visualization.save_pdf):
                    thumbnail_indices = _select_thumbnail_indices(
                        num_frames=num_frames,
                        num_thumbnails=int(cfg.visualization.num_plot_frames),
                    )
                    thumbnail_frames = _build_plot_frames(
                        prepared,
                        frame_indices=thumbnail_indices,
                        camera_names=[str(name) for name in encoder.camera_names],
                        vis_camera_name=str(cfg.visualization.camera_name) if str(cfg.visualization.camera_name) else None,
                        flip_vertical=bool(cfg.visualization.flip_vertical),
                    )
                    plot_pdf_path = os.path.join(run_dir, f"{stem}.pdf")
                    _save_failure_plot_pdf(
                        pdf_path=plot_pdf_path,
                        report_pdf=report_pdf,
                        ref=ref,
                        trajectory_id=int(traj_idx),
                        fps=int(cfg.visualization.fps),
                        frame_scores=frame_scores,
                        frame_thresholds=frame_thresholds,
                        thumbnail_frames=thumbnail_frames,
                        thumbnail_indices=thumbnail_indices,
                        delta_final=float(result.metadata.get("delta_final", np.nan)),
                        threshold_final=float(result.metadata.get("threshold_final", np.nan)),
                        frame_scores_by_mode={
                            key: np.asarray(values, dtype=np.float32)
                            for key, values in frame_scores_by_mode.items()
                        },
                        frame_thresholds_by_mode={
                            key: np.asarray(values, dtype=np.float32)
                            for key, values in frame_thresholds_by_mode.items()
                        },
                        active_score_mode=active_score_mode,
                        aggregate_contribution_shares={
                            key: np.asarray(values, dtype=np.float32)
                            for key, values in frame_aggregate_share_by_term.items()
                        },
                        first_crossing_index=result.metadata.get("first_crossing_index", None),
                        first_crossing_dominant_term=result.metadata.get("first_crossing_dominant_term", None),
                        gt_fail_mask=np.asarray(frame_gt_fail, dtype=np.int64),
                    )

                first_pred_failure = np.where(frame_preds == 1)[0]
                first_gt_failure = np.where(frame_gt_fail == 1)[0]
                records.append(
                    VideoRenderRecord(
                        trajectory_id=int(traj_idx),
                        source_file=str(ref.file_path),
                        demo_key=str(ref.demo_key),
                        num_frames=num_frames,
                        first_pred_failure_frame=(
                            int(first_pred_failure[0] + 1) if first_pred_failure.size > 0 else None
                        ),
                        pred_failure_frame_count=int(np.sum(frame_preds)),
                        gt_failure_frame_count=int(np.sum(frame_gt_fail)),
                        video_path=str(video_path),
                        metadata={
                            "task_name": ref.task_name,
                            "camera_name": (
                                str(cfg.visualization.camera_name)
                                if str(cfg.visualization.camera_name)
                                else str(encoder.camera_names[0])
                            ),
                            "threshold_final": float(result.metadata.get("threshold_final", np.nan)),
                            "delta_final": float(result.metadata.get("delta_final", np.nan)),
                            "first_gt_failure_frame": (
                                int(first_gt_failure[0] + 1) if first_gt_failure.size > 0 else None
                            ),
                            "plot_pdf_path": plot_pdf_path,
                            "term_attribution": term_summary,
                        },
                    )
                )
        finally:
            if report_pdf is not None:
                report_pdf.close()

        summary = {
            "timestamp": now_tag(),
            "split_summary": split_summary,
            "runtime": {
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "dsm_ckpt": to_absolute_path(str(cfg.model.dsm_ckpt)),
                "threshold_init": (
                    float(calibration_summary.threshold)
                    if calibration_summary.threshold is not None
                    else float("nan")
                ),
                "detector_hparams": {
                    "score_mode": str(getattr(cfg.detector, "score_mode", detector.score_mode)),
                    "alpha_state": float(getattr(cfg.detector, "alpha_state", 1.0)),
                    "alpha_action": float(getattr(cfg.detector, "alpha_action", 1.0)),
                    "alpha_dynamics": float(getattr(cfg.detector, "alpha_dynamics", 1.0)),
                    "beta_state": float(getattr(cfg.detector, "beta_state", 1.0)),
                    "beta_action": float(getattr(cfg.detector, "beta_action", 1.0)),
                    "beta_dynamics": float(getattr(cfg.detector, "beta_dynamics", 1.0)),
                },
                **dict(calibration_summary.metadata),
            },
            "visualization": {
                "data_source": str(target_summary.get("data_source")),
                "source_split": str(target_summary.get("split_name")),
                "num_requested": int(cfg.visualization.num_videos),
                "num_rendered": len(records),
                "num_available": int(target_summary.get("num_available", len(records))),
                "num_selected": int(target_summary.get("num_selected", len(records))),
                "fps": int(cfg.visualization.fps),
                "flip_vertical": bool(cfg.visualization.flip_vertical),
                "save_pdf": bool(cfg.visualization.save_pdf),
                "report_pdf_path": report_pdf_path if bool(cfg.visualization.save_pdf) else None,
                "camera_name": (
                    str(cfg.visualization.camera_name)
                    if str(cfg.visualization.camera_name)
                    else str(encoder.camera_names[0])
                ),
            },
            "target_summary": target_summary,
            "calibration_summary": {
                "source_split": str(cfg.eval.bank_split),
                "source_data_types": list(cfg.eval.bank_data_types),
                "num_available": int(len(bank_refs)),
                "num_selected": int(len(sampled_bank_refs)),
                "bank_size": int(bank_size),
                "calibration_seed": int(calibration_seed),
                "fail_distribution_split": str(cfg.eval.fail_eval_split),
                "fail_distribution_data_types": list(cfg.eval.fail_eval_data_types),
                "num_fail_distribution_trajectories": int(len(fail_distribution_trajectories)),
                "threshold_plot_dir": threshold_dir,
                "threshold_plots_by_task": threshold_plot_paths,
            },
            "videos": [asdict(record) for record in records],
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[lpb_score] Saved visualization summary to: {summary_path}")

    finally:
        if detector is not None:
            detector.close()
        encoder.close()
