from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.bce.dataset import (
    build_cached_splits,
    build_temporal_binary_labels,
    filter_refs_by_data_types,
    load_latent_trajectories,
    parse_task_beta_map,
)
from robosuite.discriminator.bce.tpud_discriminator import TPUDDiscriminator
from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
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


def _sample_refs(refs, num_samples: int, seed: int):
    if not refs:
        return []
    num = min(len(refs), max(1, int(num_samples)))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(refs), size=num, replace=False)
    indices = np.sort(indices)
    return [refs[int(idx)] for idx in indices.tolist()]


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
        beta_by_task = parse_task_beta_map(
            cfg_data=cfg.data,
            default_beta=float(cfg.data.labels.default_beta),
        )
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

        detector = TPUDDiscriminator(
            checkpoint_path=to_absolute_path(str(cfg.model.tpud_ckpt)),
            device=str(cfg.detector.device),
            batch_size=int(cfg.detector.batch_size),
            action_horizon=int(cfg.detector.action_horizon),
            aggregate_mode=str(cfg.detector.aggregate_mode),
            default_delta=float(cfg.detector.default_delta),
            task_delta={str(k): float(v) for k, v in cfg.detector.task_delta.items()},
            delta_step=float(cfg.detector.delta_step),
            moving_mean_window=int(cfg.detector.moving_mean_window),
            ema_alpha=float(cfg.detector.ema_alpha),
            min_persistence=int(cfg.detector.min_persistence),
            decision_warmup_steps=int(cfg.detector.decision_warmup_steps),
        )
        calibration_summary = detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )

        run_dir = os.path.join(to_absolute_path(str(cfg.save_dir)), f"run_{_now_tag()}")
        os.makedirs(run_dir, exist_ok=True)

        records: list[VideoRenderRecord] = []
        for traj_idx, (ref, latent_traj) in enumerate(zip(selected_fail_refs, fail_trajectories)):
            draft_result = detector.detect_trajectory(latent_traj, adaptive_threshold=False)
            labels = build_temporal_binary_labels(
                num_steps=int(draft_result.aggregate_scores.shape[0]),
                beta=float(beta_by_task.get(ref.task_name, cfg.data.labels.default_beta)),
                inlier_cutoff=float(cfg.visualization.inlier_label_cutoff),
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

            video_path = os.path.join(
                run_dir,
                f"traj{traj_idx:02d}_{ref.task_name}_{os.path.basename(ref.file_path).replace('.hdf5', '')}_{ref.demo_key}.mp4",
            )
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
                    extra_text = (
                        f"task={ref.task_name} beta={float(beta_by_task.get(ref.task_name, cfg.data.labels.default_beta)):.2f} "
                        f"delta={float(result.metadata.get('delta_final', np.nan)):.2f}"
                    )
                    frame = draw_detection_overlay(
                        frame_rgb,
                        frame_id=frame_id,
                        pred_fail_flag=bool(frame_preds[frame_id]),
                        gt_fail_flag=bool(frame_labels[frame_id]),
                        aggregate_score=float(frame_scores[frame_id]),
                        threshold=float(frame_thresholds[frame_id]),
                        detector_name=detector.name,
                        extra_text=extra_text,
                        border_thickness=int(cfg.visualization.border_thickness),
                        banner_font_scale=float(cfg.visualization.banner_font_scale),
                        banner_thickness=int(cfg.visualization.banner_thickness),
                        footer_font_scale=float(cfg.visualization.footer_font_scale),
                        footer_thickness=int(cfg.visualization.footer_thickness),
                    )
                    writer.append_data(frame)
            finally:
                writer.close()

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
                        "beta": float(beta_by_task.get(ref.task_name, cfg.data.labels.default_beta)),
                        "threshold_final": float(result.metadata.get("threshold_final", np.nan)),
                        "delta_final": float(result.metadata.get("delta_final", np.nan)),
                    },
                )
            )

        summary = {
            "timestamp": _now_tag(),
            "split_summary": split_summary,
            "runtime": {
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "tpud_ckpt": to_absolute_path(str(cfg.model.tpud_ckpt)),
                **dict(calibration_summary.metadata),
            },
            "beta_by_task": beta_by_task,
            "visualization": {
                "num_requested": int(cfg.visualization.num_videos),
                "num_rendered": len(records),
                "fps": int(cfg.visualization.fps),
                "flip_vertical": bool(cfg.visualization.flip_vertical),
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
        print(f"[tpud] Saved visualization summary to: {summary_path}")
    finally:
        if detector is not None:
            detector.close()
        encoder.close()


if __name__ == "__main__":
    main()
