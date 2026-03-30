from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

import hydra
import matplotlib
import numpy as np
from hydra.utils import to_absolute_path

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.float.float_data import fail_prefix_labels
from robosuite.discriminator.lpb_new.dataset import (
    EncodedTrajectoryRef,
    build_cached_splits,
    filter_refs_by_data_types,
    load_latent_trajectories,
)
from robosuite.discriminator.lpb_new.knn_discriminator import LPBKNNDiscriminator
from robosuite.discriminator.utils.types import VideoRenderRecord
from robosuite.discriminator.utils.video_io import create_vscode_mp4_writer
from robosuite.discriminator.utils.visualization import (
    draw_detection_overlay,
    fix_robosuite_frame_orientation,
    map_step_values_to_frames,
    select_camera_frame,
)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sample_refs(
    refs: list[EncodedTrajectoryRef],
    num_samples: int,
    seed: int,
) -> list[EncodedTrajectoryRef]:
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
    ref: EncodedTrajectoryRef,
    trajectory_id: int,
    fps: int,
    frame_scores: np.ndarray,
    frame_thresholds: np.ndarray,
    thumbnail_frames: list[np.ndarray],
    thumbnail_indices: list[int],
    delta_final: float,
    threshold_final: float,
) -> None:
    num_frames = int(np.asarray(frame_scores).shape[0])
    if num_frames <= 0:
        return

    fps_value = max(1, int(fps))
    times = np.arange(num_frames, dtype=np.float32) / float(fps_value)
    num_cols = max(1, len(thumbnail_frames))

    fig = plt.figure(figsize=(4.2 * num_cols, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, num_cols, height_ratios=[1.0, 2.2])
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(num_cols)]
    ax = fig.add_subplot(grid[1, :])

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
            ax.axvline(times[int(frame_id)], color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)

    ax.plot(times, np.asarray(frame_scores, dtype=np.float32), color="#1f77b4", linewidth=2.2, label="lambda")
    ax.plot(
        times,
        np.asarray(frame_thresholds, dtype=np.float32),
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="--",
        label="threshold",
    )

    ax.set_title(
        f"Trajectory {int(trajectory_id):02d} | {ref.task_name} | {ref.demo_key}",
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lambda")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(loc="upper left")
    ax.text(
        0.995,
        0.02,
        f"delta_final={float(delta_final):.2f}  threshold_final={float(threshold_final):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )

    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if report_pdf is not None:
        report_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base="1.2", config_path="./config", config_name="visualize")
def main(cfg: DictConfig) -> None:
    detector = None
    encoder = FrozenFlowMultitaskEncoder(
        checkpoint_path=str(cfg.policy.ckpt),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
    )

    try:
        cached_splits, split_summary, _ = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=int(cfg.seed),
        )
        bank_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.bank_split)],
            list(cfg.eval.bank_data_types),
        )
        calibration_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.calibration_split)],
            list(cfg.eval.calibration_data_types),
        )
        fail_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.fail_eval_split)],
            list(cfg.eval.fail_eval_data_types),
        )
        selected_fail_refs = _sample_refs(
            refs=fail_refs,
            num_samples=int(cfg.visualization.num_videos),
            seed=int(cfg.seed),
        )
        if not selected_fail_refs:
            raise RuntimeError("No fail trajectories selected for visualization.")

        bank_trajectories = load_latent_trajectories(bank_refs)
        calibration_trajectories = load_latent_trajectories(calibration_refs)
        fail_trajectories = load_latent_trajectories(selected_fail_refs)
        if not bank_trajectories:
            raise RuntimeError("No bank trajectories found for visualization.")
        if not calibration_trajectories:
            raise RuntimeError("No calibration trajectories found for visualization.")

        detector = LPBKNNDiscriminator(
            checkpoint_path=to_absolute_path(str(cfg.model.lpb_ckpt)),
            feature_device=str(cfg.feature.device),
            feature_batch_size=int(cfg.feature.batch_size),
            action_horizon=int(cfg.feature.action_horizon),
            normalize_feature=bool(cfg.feature.normalize_feature),
            normalize_policy_chunk=bool(cfg.feature.normalize_policy_chunk),
            policy_history_steps=int(cfg.feature.policy_history_steps),
            use_transition_error=bool(cfg.feature.use_transition_error),
            detector_device=str(cfg.detector.device),
            delta=float(cfg.detector.delta),
            delta_step=float(cfg.detector.delta_step),
            knn_chunk_size=int(cfg.detector.knn_chunk_size),
            lambda_mode=str(cfg.detector.lambda_mode),
            lambda_window_size=int(cfg.detector.lambda_window_size),
            feature_knn_weight=float(cfg.detector.feature_knn_weight),
            transition_aux_weight=float(cfg.detector.transition_aux_weight),
            policy_chunk_weight=float(cfg.detector.policy_chunk_weight),
            dynamics_weight=float(cfg.detector.dynamics_weight),
            neighbor_topk=int(cfg.detector.neighbor_topk),
            dynamics_temperature=float(cfg.detector.dynamics_temperature),
        )
        calibration_summary = detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )

        run_dir = os.path.join(to_absolute_path(str(cfg.save_dir)), f"run_{_now_tag()}")
        os.makedirs(run_dir, exist_ok=True)
        report_pdf_path = os.path.join(run_dir, "failure_report.pdf")
        report_pdf = PdfPages(report_pdf_path) if bool(cfg.visualization.save_pdf) else None

        records: list[VideoRenderRecord] = []
        try:
            for traj_idx, (ref, latent_traj) in enumerate(zip(selected_fail_refs, fail_trajectories)):
                draft_result = detector.detect_trajectory(latent_traj, adaptive_threshold=False)
                labels = fail_prefix_labels(
                    length=int(draft_result.aggregate_scores.shape[0]),
                    fail_tail_ratio=float(cfg.labels.fail_tail_ratio),
                )
                result = detector.detect_trajectory(
                    latent_traj,
                    labels=labels if bool(cfg.online.adaptive_delta) else None,
                    adaptive_threshold=bool(cfg.online.adaptive_delta),
                    delta_min=float(cfg.online.delta_min),
                    delta_max=float(cfg.online.delta_max),
                    warmup_steps=int(cfg.online.warmup_steps),
                    update_interval=int(cfg.online.update_interval),
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
                frame_labels = map_step_values_to_frames(
                    labels,
                    num_frames=int(prepared.images_hwc.shape[0]),
                    tail_fill=float(labels[-1]) if labels.size > 0 else 0.0,
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
                            gt_fail_flag=bool(frame_labels[frame_id]),
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
                        gt_failure_frame_count=int(np.sum(frame_labels)),
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
                        },
                    )
                )
        finally:
            if report_pdf is not None:
                report_pdf.close()

        summary = {
            "timestamp": _now_tag(),
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


if __name__ == "__main__":
    main()
