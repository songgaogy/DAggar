from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.float.float_data import fail_prefix_labels
from robosuite.discriminator.utils.evaluation import evaluate_trajectory_discriminator
from robosuite.discriminator.lpb_new.dataset import (
    DATA_TYPE_ORDER,
    filter_refs_by_data_types,
    load_latent_trajectories,
)
from robosuite.discriminator.lpb_new.dataset import build_cached_splits
from robosuite.discriminator.lpb_new.knn_discriminator import LPBKNNDiscriminator


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_knn_discriminator")
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
        print(f"[lpb_new] eval split_summary={split_summary}")

        bank_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.bank_split)],
            list(cfg.eval.bank_data_types),
        )
        calib_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.calibration_split)],
            list(cfg.eval.calibration_data_types),
        )
        expert_eval_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.expert_eval_split)],
            list(cfg.eval.expert_eval_data_types),
        )
        success_eval_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.success_eval_split)],
            list(cfg.eval.success_eval_data_types),
        )
        fail_eval_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.fail_eval_split)],
            list(cfg.eval.fail_eval_data_types),
        )

        bank_trajectories = load_latent_trajectories(bank_refs)
        calib_trajectories = load_latent_trajectories(calib_refs)
        expert_eval_trajectories = load_latent_trajectories(expert_eval_refs)
        success_eval_trajectories = load_latent_trajectories(success_eval_refs)
        fail_eval_trajectories = load_latent_trajectories(fail_eval_refs)

        if not bank_trajectories:
            raise RuntimeError("No bank trajectories found for KNN fitting.")
        if not calib_trajectories:
            raise RuntimeError("No calibration trajectories found for KNN fitting.")
        if not fail_eval_trajectories:
            raise RuntimeError("No fail trajectories found for KNN evaluation.")

        detector = LPBKNNDiscriminator(
            checkpoint_path=to_absolute_path(str(cfg.model.lpb_ckpt)),
            feature_device=str(cfg.feature.device),
            feature_batch_size=int(cfg.feature.batch_size),
            action_horizon=int(cfg.feature.action_horizon),
            normalize_feature=bool(cfg.feature.normalize_feature),
            normalize_policy_chunk=bool(cfg.feature.normalize_policy_chunk),
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
        summary, calibration_summary, _ = evaluate_trajectory_discriminator(
            detector=detector,
            bank_trajectories=bank_trajectories,
            calibration_trajectories=calib_trajectories,
            expert_eval_trajectories=expert_eval_trajectories,
            success_eval_trajectories=success_eval_trajectories,
            fail_eval_trajectories=fail_eval_trajectories,
            fail_label_builder=lambda _traj, result: fail_prefix_labels(
                length=int(result.aggregate_scores.shape[0]),
                fail_tail_ratio=float(cfg.labels.fail_tail_ratio),
            ),
            adaptive_threshold=bool(cfg.online.adaptive_delta),
            delta_min=float(cfg.online.delta_min),
            delta_max=float(cfg.online.delta_max),
            warmup_steps=int(cfg.online.warmup_steps),
            update_interval=int(cfg.online.update_interval),
        )
        summary["split_summary"] = split_summary
        runtime = dict(summary.get("runtime", {}))
        runtime.update(
            {
                "lpb_ckpt": to_absolute_path(str(cfg.model.lpb_ckpt)),
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "threshold_init": (
                    float(calibration_summary.threshold)
                    if calibration_summary.threshold is not None
                    else float("nan")
                ),
                "bank_split": str(cfg.eval.bank_split),
                "calibration_split": str(cfg.eval.calibration_split),
                "expert_eval_split": str(cfg.eval.expert_eval_split),
                "success_eval_split": str(cfg.eval.success_eval_split),
                "fail_eval_split": str(cfg.eval.fail_eval_split),
                "threshold_final": float(summary["fail_metrics"]["threshold_final"]),
                "delta_final": float(summary["fail_metrics"]["delta_final"]),
            }
        )
        summary["runtime"] = runtime
        summary["data_types"] = DATA_TYPE_ORDER

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"lpb_new_knn_eval_{_now_tag()}.json")
        with open(out_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[lpb_new] Saved evaluation summary to: {out_path}")
    finally:
        if detector is not None:
            detector.close()
        encoder.close()


if __name__ == "__main__":
    main()
