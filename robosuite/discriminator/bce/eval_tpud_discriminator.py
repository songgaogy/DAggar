from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.bce.dataset import (
    DATA_TYPE_ORDER,
    build_cached_splits,
    build_temporal_binary_labels,
    filter_refs_by_data_types,
    load_latent_trajectories,
    parse_task_beta_map,
)
from robosuite.discriminator.bce.tpud_discriminator import TPUDDiscriminator
from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.utils.evaluation import evaluate_trajectory_discriminator


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_tpud_discriminator")
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
        print(f"[tpud] eval split_summary={split_summary}")

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
            raise RuntimeError("No bank trajectories found for TPUD fitting.")
        if not calib_trajectories:
            raise RuntimeError("No calibration trajectories found for TPUD fitting.")
        if not fail_eval_trajectories:
            raise RuntimeError("No fail trajectories found for TPUD evaluation.")

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
        summary, calibration_summary, _ = evaluate_trajectory_discriminator(
            detector=detector,
            bank_trajectories=bank_trajectories,
            calibration_trajectories=calib_trajectories,
            expert_eval_trajectories=expert_eval_trajectories,
            success_eval_trajectories=success_eval_trajectories,
            fail_eval_trajectories=fail_eval_trajectories,
            fail_label_builder=lambda traj, result: build_temporal_binary_labels(
                num_steps=int(result.aggregate_scores.shape[0]),
                beta=float(beta_by_task.get(traj.task_name, cfg.data.labels.default_beta)),
                inlier_cutoff=float(cfg.eval.inlier_label_cutoff),
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
                "tpud_ckpt": to_absolute_path(str(cfg.model.tpud_ckpt)),
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "threshold_by_task": dict(calibration_summary.metadata.get("threshold_by_task", {})),
                "delta_by_task": dict(calibration_summary.metadata.get("delta_by_task", {})),
                "bank_split": str(cfg.eval.bank_split),
                "calibration_split": str(cfg.eval.calibration_split),
                "expert_eval_split": str(cfg.eval.expert_eval_split),
                "success_eval_split": str(cfg.eval.success_eval_split),
                "fail_eval_split": str(cfg.eval.fail_eval_split),
            }
        )
        summary["runtime"] = runtime
        summary["beta_by_task"] = beta_by_task
        summary["data_types"] = DATA_TYPE_ORDER

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"tpud_eval_{_now_tag()}.json")
        with open(out_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[tpud] Saved evaluation summary to: {out_path}")
    finally:
        if detector is not None:
            detector.close()
        encoder.close()


if __name__ == "__main__":
    main()
