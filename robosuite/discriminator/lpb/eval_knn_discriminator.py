from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.float.float_data import (
    fail_prefix_labels,
    load_policy_trajectories,
    split_train_val,
)
from robosuite.discriminator.float.float_eval import classification_metrics
from robosuite.discriminator.lpb.knn_discriminator import AdaptiveKNNDiscriminator, LPBFeatureExtractor


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_knn_discriminator")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))

    expert_dir = to_absolute_path(str(cfg.data.expert_dir))
    fail_dir = to_absolute_path(str(cfg.data.fail_rollout_dir))
    camera_name = str(cfg.data.camera_name)
    lpb_ckpt = to_absolute_path(str(cfg.model.lpb_ckpt))

    experts = load_policy_trajectories(
        data_dir=expert_dir,
        camera_name=camera_name,
        max_trajectories=(None if int(cfg.data.max_expert_trajectories) <= 0 else int(cfg.data.max_expert_trajectories)),
    )
    fails = load_policy_trajectories(
        data_dir=fail_dir,
        camera_name=camera_name,
        max_trajectories=(None if int(cfg.data.max_fail_trajectories) <= 0 else int(cfg.data.max_fail_trajectories)),
    )
    print("finish loading data")

    if len(experts) == 0:
        raise RuntimeError("No expert trajectories found.")
    if len(fails) == 0:
        raise RuntimeError("No fail trajectories found.")

    # Split expert trajectories:
    # - bank: KNN expert memory
    # - holdout -> (calibration + expert validation)
    bank_ratio_holdout = float(cfg.split.holdout_ratio)
    expert_bank, expert_holdout = split_train_val(
        experts,
        val_ratio=bank_ratio_holdout,
        seed=int(cfg.seed),
    )
    if len(expert_bank) == 0:
        raise RuntimeError("No expert bank trajectories after split.")
    if len(expert_holdout) == 0:
        expert_holdout = expert_bank

    calib_ratio = float(cfg.split.calib_ratio_within_holdout)
    expert_calib, expert_eval = split_train_val(
        expert_holdout,
        val_ratio=calib_ratio,
        seed=int(cfg.seed) + 1,
    )
    if len(expert_calib) == 0:
        expert_calib = expert_holdout
    if len(expert_eval) == 0:
        expert_eval = expert_calib

    proprio_indices = list(cfg.feature.proprio_indices) if cfg.feature.proprio_indices else None
    extractor = LPBFeatureExtractor(
        checkpoint_path=lpb_ckpt,
        device=str(cfg.feature.device),
        batch_size=int(cfg.feature.batch_size),
        action_horizon=int(cfg.feature.action_horizon),
        proprio_indices=proprio_indices,
        normalize_feature=bool(cfg.feature.normalize_feature),
    )

    use_transition_score = bool(cfg.feature.use_transition_error)
    proprio_err_w = float(cfg.feature.transition_proprio_error_weight)

    def _encode_set(trajs):
        feat_list = []
        aux_list = []
        for tr in trajs:
            if use_transition_score:
                f, e = extractor.encode_trajectory_with_transition_error(
                    tr,
                    proprio_error_weight=proprio_err_w,
                )
                feat_list.append(f)
                aux_list.append(e.detach().cpu().numpy().astype(np.float32))
            else:
                feat_list.append(extractor.encode_trajectory(tr))
                aux_list.append(None)
        return feat_list, aux_list

    bank_features, _ = _encode_set(expert_bank)
    calib_features, calib_aux = _encode_set(expert_calib)
    eval_features, eval_aux = _encode_set(expert_eval)

    detector = AdaptiveKNNDiscriminator(
        delta=float(cfg.detector.delta),
        delta_step=float(cfg.detector.delta_step),
        knn_chunk_size=int(cfg.detector.knn_chunk_size),
        lambda_mode=str(cfg.detector.lambda_mode),
        lambda_window_size=int(cfg.detector.lambda_window_size),
        aux_weight=float(cfg.detector.transition_aux_weight),
        device=str(cfg.detector.device),
    )
    threshold_init = detector.fit(
        expert_sequences=bank_features,
        calibration_sequences=calib_features,
        calibration_aux=None if not use_transition_score else calib_aux,
    )

    # Evaluate expert false alarm on held-out expert eval set (all labels=0).
    expert_false_alarm = 0
    expert_total = 0
    expert_lambda_values = []
    for i, feat in enumerate(eval_features):
        aux = None if (not use_transition_score or eval_aux[i] is None) else eval_aux[i]
        out = detector.detect_sequence(feat, labels=None, adaptive_delta=False, aux_scores=aux)
        expert_false_alarm += int(np.sum(out.preds))
        expert_total += int(out.preds.size)
        expert_lambda_values.extend(out.lambda_values.tolist())

    # Evaluate fail trajectories (last 20% labeled as fail).
    # NOTE: here 20% is human-intuitive percentage
    y_true: list[int] = []
    y_pred: list[int] = []
    fail_lambda_values: list[float] = []
    pre_fail_lambda_values: list[float] = []

    for traj in fails:
        if use_transition_score:
            feat, aux = extractor.encode_trajectory_with_transition_error(
                traj,
                proprio_error_weight=proprio_err_w,
            )
            aux_np = aux.detach().cpu().numpy().astype(np.float32)
        else:
            feat = extractor.encode_trajectory(traj)
            aux_np = None
        labels = fail_prefix_labels(length=feat.shape[0], fail_tail_ratio=float(cfg.labels.fail_tail_ratio))
        out = detector.detect_sequence(
            features=feat,
            labels=labels if bool(cfg.online.adaptive_delta) else None,
            adaptive_delta=bool(cfg.online.adaptive_delta),
            aux_scores=aux_np,
            delta_min=float(cfg.online.delta_min),
            delta_max=float(cfg.online.delta_max),
            warmup_steps=int(cfg.online.warmup_steps),
            update_interval=int(cfg.online.update_interval),
        )
        y_true.extend(labels.tolist())
        y_pred.extend(out.preds.tolist())

        for lv, lb in zip(out.lambda_values.tolist(), labels.tolist()):
            if int(lb) == 1:
                fail_lambda_values.append(float(lv))
            else:
                pre_fail_lambda_values.append(float(lv))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    fail_metrics = classification_metrics(y_true=y_true_np, y_pred=y_pred_np)
    fail_metrics["delta_final"] = float(detector.delta)
    fail_metrics["threshold_final"] = float(detector.threshold if detector.threshold is not None else np.nan)

    expert_metrics = {
        "expert_prefixes": float(expert_total),
        "false_alarms": float(expert_false_alarm),
        "false_alarm_rate": float(expert_false_alarm / expert_total) if expert_total > 0 else float("nan"),
        "lambda_expert_mean": float(np.mean(expert_lambda_values)) if expert_lambda_values else float("nan"),
    }

    lambda_stats = {
        "lambda_fail_mean": float(np.mean(fail_lambda_values)) if fail_lambda_values else float("nan"),
        "lambda_prefail_mean": float(np.mean(pre_fail_lambda_values)) if pre_fail_lambda_values else float("nan"),
        "lambda_gap": (
            float(np.mean(fail_lambda_values) - np.mean(pre_fail_lambda_values))
            if fail_lambda_values and pre_fail_lambda_values
            else float("nan")
        ),
    }

    summary = {
        "counts": {
            "expert_total": len(experts),
            "expert_bank": len(expert_bank),
            "expert_calib": len(expert_calib),
            "expert_eval": len(expert_eval),
            "fail_total": len(fails),
        },
        "knn_runtime": {
            "lpb_ckpt": lpb_ckpt,
            "camera_name": camera_name,
            "delta_init": float(cfg.detector.delta),
            "delta_final": float(detector.delta),
            "delta_step": float(cfg.detector.delta_step),
            "knn_chunk_size": int(cfg.detector.knn_chunk_size),
            "lambda_mode": str(cfg.detector.lambda_mode),
            "lambda_window_size": int(cfg.detector.lambda_window_size),
            "transition_aux_weight": float(cfg.detector.transition_aux_weight),
            "use_transition_error": bool(cfg.feature.use_transition_error),
            "transition_proprio_error_weight": float(cfg.feature.transition_proprio_error_weight),
            "adaptive_delta": bool(cfg.online.adaptive_delta),
            "delta_min": float(cfg.online.delta_min),
            "delta_max": float(cfg.online.delta_max),
            "warmup_steps": int(cfg.online.warmup_steps),
            "update_interval": int(cfg.online.update_interval),
        },
        "threshold_init": float(threshold_init),
        "threshold_final": float(detector.threshold if detector.threshold is not None else np.nan),
        "expert_validation": expert_metrics,
        "fail_validation": fail_metrics,
        "lambda_separation": lambda_stats,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    save_dir = to_absolute_path(str(cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    tag = _now_tag()
    summary_path = os.path.join(save_dir, f"lpb_knn_summary_{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Saved KNN summary to: {summary_path}")


if __name__ == "__main__":
    main()
