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

from robosuite.discriminator.lpb_new.analysis.attribution import (
    TERM_COLORS,
    TERM_LABELS,
    TERM_SHORT_LABELS,
    map_term_values_to_frames,
    ordered_term_keys,
    summarize_trajectory_term_attribution,
)
from robosuite.discriminator.lpb_new.app.pipeline import (
    build_flow_encoder,
    build_lpb_knn_discriminator,
    now_tag,
    select_split_refs,
)
from robosuite.discriminator.lpb_new.core.dataset import (
    EncodedTrajectoryRef,
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


def _sample_refs(
    refs: list[EncodedTrajectoryRef],
    num_samples: int,
    seed: int,
) -> list[EncodedTrajectoryRef]:
    """Choose a reproducible subset of fail trajectories to render."""
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
    ref: EncodedTrajectoryRef,
    trajectory_id: int,
    fps: int,
    frame_scores: np.ndarray,
    frame_thresholds: np.ndarray,
    thumbnail_frames: list[np.ndarray],
    thumbnail_indices: list[int],
    delta_final: float,
    threshold_final: float,
    aggregate_contribution_shares: dict[str, np.ndarray],
    first_crossing_index: int | None,
    first_crossing_dominant_term: str | None,
) -> None:
    """Save a PDF page with both score traces and term attributions."""
    num_frames = int(np.asarray(frame_scores).shape[0])
    if num_frames <= 0:
        return

    fps_value = max(1, int(fps))
    times = np.arange(num_frames, dtype=np.float32) / float(fps_value)
    num_cols = max(1, len(thumbnail_frames))

    fig = plt.figure(figsize=(4.2 * num_cols, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(3, num_cols, height_ratios=[1.0, 1.6, 1.7])
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(num_cols)]
    score_ax = fig.add_subplot(grid[1, :])
    contrib_ax = fig.add_subplot(grid[2, :])

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
    if first_crossing_index is not None and 0 <= int(first_crossing_index) < num_frames:
        score_ax.axvline(
            times[int(first_crossing_index)],
            color="#d62728",
            linewidth=1.5,
            linestyle="-.",
            alpha=0.95,
            label="first crossing",
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


def run_visualize(cfg: DictConfig) -> None:
    detector = None
    encoder = build_flow_encoder(cfg)   # reuse policy encoder

    try:
        # 1) create banks
        cached_splits, split_summary, _ = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=int(cfg.seed),
        )
        bank_refs = select_split_refs(  # normal trajs from expert + success rollout
            cached_splits=cached_splits,
            split_name=str(cfg.eval.bank_split),
            data_types=list(cfg.eval.bank_data_types),
        )
        calibration_refs = select_split_refs(   # for testing threshold
            cached_splits=cached_splits,
            split_name=str(cfg.eval.calibration_split),
            data_types=list(cfg.eval.calibration_data_types),
        )
        fail_refs = select_split_refs(
            cached_splits=cached_splits,
            split_name=str(cfg.eval.fail_eval_split),
            data_types=list(cfg.eval.fail_eval_data_types),
        )
        selected_fail_refs = _sample_refs(      # for output visualization
            refs=fail_refs,
            num_samples=int(cfg.visualization.num_videos),
            seed=int(cfg.seed),
        )
        if not selected_fail_refs:
            raise RuntimeError("No fail trajectories selected for visualization.")
        
        # 2) create / load latent after dynamic model
        bank_trajectories = load_latent_trajectories(bank_refs)
        calibration_trajectories = load_latent_trajectories(calibration_refs)
        fail_trajectories = load_latent_trajectories(selected_fail_refs)
        if not bank_trajectories:
            raise RuntimeError("No bank trajectories found for visualization.")
        if not calibration_trajectories:
            raise RuntimeError("No calibration trajectories found for visualization.")

        # 3) init detector, get threshold
        detector = build_lpb_knn_discriminator(cfg)
        calibration_summary = detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )

        run_dir = os.path.join(to_absolute_path(str(cfg.save_dir)), f"run_{now_tag()}")
        os.makedirs(run_dir, exist_ok=True)
        report_pdf_path = os.path.join(run_dir, "failure_report.pdf")
        report_pdf = PdfPages(report_pdf_path) if bool(cfg.visualization.save_pdf) else None

        # 4) run detector
        print("\n[INFO] start discriminating:\n")
        records: list[VideoRenderRecord] = []
        try:
            for traj_idx, (ref, latent_traj) in enumerate(zip(selected_fail_refs, fail_trajectories)):
                # run detector
                result = detector.detect_trajectory(
                    latent_traj,
                    adaptive_threshold=False,
                )

                # --- all logics below are helper functions ---
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
                aggregate_contribution_shares = (
                    result.metadata.get("aggregate_contribution_shares", {}) or {}
                )
                frame_aggregate_share_by_term = map_term_values_to_frames(
                    aggregate_contribution_shares,
                    num_frames=int(prepared.images_hwc.shape[0]),
                )
                feature_knn_scores = result.metadata.get("feature_knn_scores", None)
                transition_error_scores = result.metadata.get("transition_error_scores", None)
                policy_chunk_scores = result.metadata.get("policy_chunk_scores", None)
                neighbor_dynamics_scores = result.metadata.get("neighbor_dynamics_scores", None)
                frame_feature_knn = (
                    map_step_values_to_frames(
                        np.asarray(feature_knn_scores, dtype=np.float32),
                        num_frames=int(prepared.images_hwc.shape[0]),
                        tail_fill=float(np.asarray(feature_knn_scores, dtype=np.float32)[-1]),
                    )
                    if feature_knn_scores is not None
                    else None
                )
                frame_transition_error = (
                    map_step_values_to_frames(
                        np.asarray(transition_error_scores, dtype=np.float32),
                        num_frames=int(prepared.images_hwc.shape[0]),
                        tail_fill=float(np.asarray(transition_error_scores, dtype=np.float32)[-1]),
                    )
                    if transition_error_scores is not None
                    else None
                )
                frame_policy_chunk = (
                    map_step_values_to_frames(
                        np.asarray(policy_chunk_scores, dtype=np.float32),
                        num_frames=int(prepared.images_hwc.shape[0]),
                        tail_fill=float(np.asarray(policy_chunk_scores, dtype=np.float32)[-1]),
                    )
                    if policy_chunk_scores is not None
                    else None
                )
                frame_neighbor_dynamics = (
                    map_step_values_to_frames(
                        np.asarray(neighbor_dynamics_scores, dtype=np.float32),
                        num_frames=int(prepared.images_hwc.shape[0]),
                        tail_fill=float(np.asarray(neighbor_dynamics_scores, dtype=np.float32)[-1]),
                    )
                    if neighbor_dynamics_scores is not None
                    else None
                )
                term_summary = summarize_trajectory_term_attribution(result)

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
                            f"det={detector.name} dl={float(result.metadata.get('delta_final', np.nan)):.1f}",
                            f"lam={float(frame_scores[frame_id]):.3f} th={float(frame_thresholds[frame_id]):.3f}",
                        ]
                        if frame_aggregate_share_by_term:
                            # Surface the winning attribution term directly in the render footer.
                            dominant_term = max(
                                frame_aggregate_share_by_term,
                                key=lambda key: float(frame_aggregate_share_by_term[key][frame_id]),
                            )
                            dominant_share = float(frame_aggregate_share_by_term[dominant_term][frame_id])
                            footer_lines.append(
                                f"dom={TERM_SHORT_LABELS.get(dominant_term, dominant_term)} {100.0 * dominant_share:.0f}%"
                            )
                        if frame_feature_knn is not None or frame_transition_error is not None:
                            footer_lines.append(
                                " ".join(
                                    [
                                        f"fk={float(frame_feature_knn[frame_id]):.3f}" if frame_feature_knn is not None else "",
                                        f"te={float(frame_transition_error[frame_id]):.3f}" if frame_transition_error is not None else "",
                                    ]
                                ).strip()
                            )
                        if frame_policy_chunk is not None or frame_neighbor_dynamics is not None:
                            footer_lines.append(
                                " ".join(
                                    [
                                        f"pc={float(frame_policy_chunk[frame_id]):.3f}" if frame_policy_chunk is not None else "",
                                        f"dy={float(frame_neighbor_dynamics[frame_id]):.3f}" if frame_neighbor_dynamics is not None else "",
                                    ]
                                ).strip()
                            )
                        footer_lines = [line for line in footer_lines if line]
                        frame = draw_detection_overlay(
                            frame_rgb,
                            frame_id=frame_id,
                            pred_fail_flag=bool(frame_preds[frame_id]),
                            gt_fail_flag=False,
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
                        thumbnail_frames=thumbnail_frames,
                        thumbnail_indices=thumbnail_indices,
                        delta_final=float(result.metadata.get("delta_final", np.nan)),
                        threshold_final=float(result.metadata.get("threshold_final", np.nan)),
                        aggregate_contribution_shares={
                            key: np.asarray(values, dtype=np.float32)
                            for key, values in aggregate_contribution_shares.items()
                        },
                        first_crossing_index=result.metadata.get("first_crossing_index", None),
                        first_crossing_dominant_term=result.metadata.get("first_crossing_dominant_term", None),
                    )

                first_pred_failure = np.where(frame_preds == 1)[0]
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
                        gt_failure_frame_count=0,
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
                "lpb_ckpt": to_absolute_path(str(cfg.model.lpb_ckpt)),
                "threshold_init": (
                    float(calibration_summary.threshold)
                    if calibration_summary.threshold is not None
                    else float("nan")
                ),
                **dict(calibration_summary.metadata),
            },
            "visualization": {
                "num_requested": int(cfg.visualization.num_videos),
                "num_rendered": len(records),
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
            "videos": [asdict(record) for record in records],
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[lpb_new] Saved visualization summary to: {summary_path}")

    finally:
        if detector is not None:
            detector.close()
        encoder.close()
