from __future__ import annotations

import json
import os
from dataclasses import asdict

import matplotlib
import numpy as np
from hydra.utils import to_absolute_path

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from robosuite.discriminator.lpb_dice.app.pipeline import (
    build_flow_encoder,
    now_tag,
)
from robosuite.discriminator.lpb_dice.app.suboptimal import (
    encode_suboptimal_refs,
    list_suboptimal_demo_refs,
)
from robosuite.discriminator.lpb_dice.core.dataset import (
    EncodedTrajectoryRef,
    LatentTrajectory,
    build_split_refs,
    list_raw_demo_refs,
    load_latent_trajectories,
    prepare_cached_trajectories,
)
from robosuite.discriminator.lpb_dice.core.pu_detector import LPBDiceDiscriminator
from robosuite.discriminator.utils.types import VideoRenderRecord
from robosuite.discriminator.utils.video_io import create_vscode_mp4_writer
from robosuite.discriminator.utils.visualization import (
    draw_detection_overlay,
    fix_robosuite_frame_orientation,
    map_step_values_to_frames,
    select_camera_frame,
)


def _sample_refs(
    refs,
    num_samples: int,
    seed: int,
):
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
    frame_raw_scores: np.ndarray,
    frame_corrected_scores: np.ndarray,
    frame_support_penalty: np.ndarray,
    thumbnail_frames: list[np.ndarray],
    thumbnail_indices: list[int],
    threshold_final: float,
    support_penalty_weight: float,
    first_crossing_index: int | None,
    gt_fail_mask: np.ndarray | None = None,
) -> None:
    num_frames = int(np.asarray(frame_scores).shape[0])
    if num_frames <= 0:
        return

    fps_value = max(1, int(fps))
    times = np.arange(num_frames, dtype=np.float32) / float(fps_value)
    num_cols = max(1, len(thumbnail_frames))

    fig = plt.figure(figsize=(4.2 * num_cols, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(3, num_cols, height_ratios=[1.0, 1.7, 1.7])
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(num_cols)]
    score_ax = fig.add_subplot(grid[1, :])
    component_ax = fig.add_subplot(grid[2, :])

    for thumb_ax, image_rgb, frame_id in zip(top_axes, thumbnail_frames, thumbnail_indices):
        thumb_ax.imshow(image_rgb)
        thumb_ax.set_axis_off()
        thumb_ax.set_title(
            f"frame {int(frame_id) + 1}\nt={times[int(frame_id)]:.2f}s",
            fontsize=10,
            color="#444444",
            pad=6,
        )
        score_ax.axvline(times[int(frame_id)], color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)
        component_ax.axvline(times[int(frame_id)], color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)

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
                    component_ax.axvspan(
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
        component_ax.axvline(
            times[int(first_crossing_index)],
            color="#d62728",
            linewidth=1.2,
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
        f"threshold={float(threshold_final):.3f}  support_w={float(support_penalty_weight):.3f}",
        transform=score_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )

    component_ax.plot(times, np.asarray(frame_raw_scores, dtype=np.float32), color="#7f7f7f", linewidth=1.2, label="raw")
    component_ax.plot(times, np.asarray(frame_corrected_scores, dtype=np.float32), color="#2ca02c", linewidth=1.6, label="corrected")
    component_ax.plot(times, np.asarray(frame_support_penalty, dtype=np.float32), color="#d62728", linewidth=1.4, label="support")
    component_ax.set_xlabel("Time (s)")
    component_ax.set_ylabel("Component")
    component_ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
    component_ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if report_pdf is not None:
        report_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _load_visualization_targets(
    cfg: DictConfig,
    *,
    encoder,
    task_to_index: dict[str, int],
    seed: int,
    num_videos: int,
    horizon: int,
) -> tuple[list[object], list[LatentTrajectory], list[np.ndarray | None], dict[str, object]]:
    data_source = str(getattr(cfg.visualization, "data_source", "fail_rollout"))
    if data_source != "suboptimal":
        fail_refs, fail_pool_summary = list_raw_demo_refs(
            cfg.data,
            data_types=list(getattr(cfg.visualization, "fail_data_types", ["fail_rollout"])),
        )
        selected_fail_refs = _sample_refs(refs=fail_refs, num_samples=int(num_videos), seed=int(seed))
        if not selected_fail_refs:
            raise RuntimeError("No fail trajectories selected for visualization.")

        encoded_fail_refs = prepare_cached_trajectories(
            refs=selected_fail_refs,
            encoder=encoder,
            cache_root=str(cfg.data.cache_dir),
            task_to_index=task_to_index,
            encode_demo_batch_size=int(cfg.data.encode_demo_batch_size),
        )
        ordered_fail_refs: list[EncodedTrajectoryRef] = []
        fail_trajectories: list[LatentTrajectory] = []
        for encoded_ref in encoded_fail_refs:
            trajectories = load_latent_trajectories([encoded_ref])
            if not trajectories:
                continue
            ordered_fail_refs.append(encoded_ref)
            fail_trajectories.append(trajectories[0])
        if not fail_trajectories:
            raise RuntimeError("Selected fail trajectories are too short after latent encoding.")
        return (
            list(ordered_fail_refs),
            list(fail_trajectories),
            [None for _ in range(len(fail_trajectories))],
            {
                "data_source": data_source,
                "split_name": "raw",
                "num_available": int(len(fail_refs)),
                "num_selected": int(len(ordered_fail_refs)),
                "raw_pool_summary": fail_pool_summary,
            },
        )

    requested_split = str(getattr(cfg.suboptimal, "source_split", "all"))
    suboptimal_refs, suboptimal_split_summary = list_suboptimal_demo_refs(cfg)
    if requested_split in {"", "*", "all"}:
        source_refs = list(suboptimal_refs)
    else:
        source_refs = [ref for ref in suboptimal_refs if str(ref.split) == requested_split]
    selected_refs = _sample_refs(refs=source_refs, num_samples=int(num_videos), seed=int(seed))
    if not selected_refs:
        raise RuntimeError(f"No suboptimal trajectories selected from split={requested_split}.")

    labeled_trajectories, encode_summary = encode_suboptimal_refs(
        refs=selected_refs,
        encoder=encoder,
        task_to_index=task_to_index,
        horizon=int(horizon),
        batch_size=int(cfg.data.encode_demo_batch_size),
    )
    labeled_lookup = {
        (item.trajectory.file_path, item.trajectory.demo_key): item
        for item in labeled_trajectories
    }

    ordered_refs: list[object] = []
    ordered_trajectories: list[LatentTrajectory] = []
    ordered_labels: list[np.ndarray | None] = []
    for ref in selected_refs:
        labeled = labeled_lookup.get((ref.file_path, ref.demo_key))
        if labeled is None:
            continue
        ordered_refs.append(ref)
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
    detector = LPBDiceDiscriminator.from_checkpoint(
        to_absolute_path(str(cfg.model.dice_ckpt)),
        device=str(cfg.detector.device),
    )
    encoder = build_flow_encoder(cfg)
    try:
        _, split_summary, task_to_index = build_split_refs(
            cfg_data=cfg.data,
            seed=int(cfg.seed),
        )
        selected_refs, trajectories, label_sequences, target_summary = _load_visualization_targets(
            cfg,
            encoder=encoder,
            task_to_index=task_to_index,
            seed=int(cfg.seed),
            num_videos=int(cfg.visualization.num_videos),
            horizon=int(detector.representation.action_horizon),
        )

        run_dir = os.path.join(to_absolute_path(str(cfg.save_dir)), f"run_{now_tag()}")
        os.makedirs(run_dir, exist_ok=True)
        report_pdf_path = os.path.join(run_dir, "failure_report.pdf")
        report_pdf = PdfPages(report_pdf_path) if bool(cfg.visualization.save_pdf) else None
        calibration_summary = detector.calibration_summary()

        records: list[VideoRenderRecord] = []
        try:
            for traj_idx, (ref, latent_traj, gt_labels) in enumerate(
                zip(selected_refs, trajectories, label_sequences)
            ):
                result = detector.detect_trajectory(
                    latent_traj,
                    support_penalty_weight=float(cfg.support_penalty.weight),
                )
                prepared = encoder.load_demo_raw(
                    task_name=ref.task_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                )
                frame_preds = map_step_values_to_frames(
                    result.predictions,
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(result.predictions[-1]) if result.predictions.size > 0 else 0.0,
                ).astype(np.int64)
                frame_scores = map_step_values_to_frames(
                    result.aggregate_scores,
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(result.aggregate_scores[-1]) if result.aggregate_scores.size > 0 else 0.0,
                )
                frame_thresholds = map_step_values_to_frames(
                    result.thresholds,
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(result.thresholds[-1]) if result.thresholds.size > 0 else 0.0,
                )
                frame_raw_scores = map_step_values_to_frames(
                    np.asarray(result.metadata["pu_raw_score"], dtype=np.float32),
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(np.asarray(result.metadata["pu_raw_score"], dtype=np.float32)[-1]),
                )
                frame_corrected_scores = map_step_values_to_frames(
                    np.asarray(result.metadata["pu_corrected_score"], dtype=np.float32),
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(np.asarray(result.metadata["pu_corrected_score"], dtype=np.float32)[-1]),
                )
                frame_support_penalty = map_step_values_to_frames(
                    np.asarray(result.metadata["support_penalty"], dtype=np.float32),
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(np.asarray(result.metadata["support_penalty"], dtype=np.float32)[-1]),
                )
                frame_gt_fail = (
                    map_step_values_to_frames(
                        np.asarray(gt_labels, dtype=np.float32),
                        num_frames=int(prepared.images_hwc.shape[0]),
                        tail_fill=float(np.asarray(gt_labels, dtype=np.float32)[-1]),
                    ).astype(np.int64)
                    if gt_labels is not None and np.asarray(gt_labels).size > 0
                    else np.zeros((int(prepared.images_hwc.shape[0]),), dtype=np.int64)
                )

                stem = (
                    f"traj{traj_idx:02d}_{ref.task_name}_"
                    f"{os.path.basename(ref.file_path).replace('.hdf5', '')}_{ref.demo_key}"
                )
                video_path = os.path.join(run_dir, f"{stem}.mp4")
                writer = create_vscode_mp4_writer(video_path, fps=int(cfg.visualization.fps))
                try:
                    for frame_id in range(int(prepared.images_hwc.shape[0])):
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
                            f"det={detector.name} c={float(result.metadata.get('c_estimate', np.nan)):.3f}",
                            f"lam={float(frame_scores[frame_id]):.3f} th={float(frame_thresholds[frame_id]):.3f}",
                            f"corr={float(frame_corrected_scores[frame_id]):.3f} pen={float(frame_support_penalty[frame_id]):.3f}",
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
                        num_frames=int(prepared.images_hwc.shape[0]),
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
                        frame_raw_scores=frame_raw_scores,
                        frame_corrected_scores=frame_corrected_scores,
                        frame_support_penalty=frame_support_penalty,
                        thumbnail_frames=thumbnail_frames,
                        thumbnail_indices=thumbnail_indices,
                        threshold_final=float(result.metadata.get("threshold_final", np.nan)),
                        support_penalty_weight=float(result.metadata.get("support_penalty_weight", 0.0)),
                        first_crossing_index=result.metadata.get("first_crossing_index", None),
                        gt_fail_mask=np.asarray(frame_gt_fail, dtype=np.int64),
                    )

                first_pred_failure = np.where(frame_preds == 1)[0]
                first_gt_failure = np.where(frame_gt_fail == 1)[0]
                records.append(
                    VideoRenderRecord(
                        trajectory_id=int(traj_idx),
                        source_file=str(ref.file_path),
                        demo_key=str(ref.demo_key),
                        num_frames=int(prepared.images_hwc.shape[0]),
                        first_pred_failure_frame=(
                            int(first_pred_failure[0] + 1) if first_pred_failure.size > 0 else None
                        ),
                        pred_failure_frame_count=int(np.sum(frame_preds)),
                        gt_failure_frame_count=int(np.sum(frame_gt_fail)),
                        video_path=str(video_path),
                        metadata={
                            "task_name": ref.task_name,
                            "plot_pdf_path": plot_pdf_path,
                            "first_gt_failure_frame": (
                                int(first_gt_failure[0] + 1) if first_gt_failure.size > 0 else None
                            ),
                            "threshold_final": float(result.metadata.get("threshold_final", np.nan)),
                            "c_estimate": float(result.metadata.get("c_estimate", np.nan)),
                            "support_penalty_weight": float(result.metadata.get("support_penalty_weight", 0.0)),
                            "pu_raw_score_mean": float(result.metadata.get("pu_raw_score_mean", np.nan)),
                            "pu_corrected_score_mean": float(result.metadata.get("pu_corrected_score_mean", np.nan)),
                            "support_penalty_mean": float(result.metadata.get("support_penalty_mean_sequence", np.nan)),
                            "final_step_score_mean": float(result.metadata.get("final_step_score_mean", np.nan)),
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
                "dice_ckpt": to_absolute_path(str(cfg.model.dice_ckpt)),
                "threshold_init": (
                    float(calibration_summary.threshold)
                    if calibration_summary.threshold is not None
                    else float("nan")
                ),
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
            "videos": [asdict(record) for record in records],
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[lpb_dice] Saved visualization summary to: {summary_path}")
    finally:
        detector.close()
        encoder.close()
