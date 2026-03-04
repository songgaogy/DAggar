from __future__ import annotations

import json
import os
from dataclasses import dataclass

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.float_data import fail_prefix_labels, load_policy_trajectories, split_train_val
from robosuite.discriminator.float_eval import classification_metrics
from robosuite.discriminator.float_official import OfficialFloatOfflineEvaluator
from robosuite.discriminator.float_policy_latent import FlowPolicyLatentExtractor


@dataclass
class PrefixRecord:
    trajectory_id: int
    step: int
    cumulative_cost: float
    label: int
    pred: int


def _sample_tail_indices(length: int, start_ratio: float, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if start_ratio < 0.0 or start_ratio >= 1.0:
        raise ValueError(f"start_ratio must be in [0,1), got {start_ratio}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be >=1, got {num_samples}")

    start = int(np.floor(start_ratio * length))
    start = int(np.clip(start, 0, length - 1))
    candidates = np.arange(start, length, dtype=np.int64)
    if candidates.size == 0:
        candidates = np.asarray([length - 1], dtype=np.int64)

    replace = candidates.size < num_samples
    return rng.choice(candidates, size=num_samples, replace=replace)


def _evaluate_random_truncation_success_rate(
    evaluator: OfficialFloatOfflineEvaluator,
    threshold: float,
    fail_embeddings: list[np.ndarray],
    expert_embeddings: list[np.ndarray],
    fail_start_ratio: float,
    expert_start_ratio: float,
    samples_per_trajectory: int,
    seed: int,
) -> dict[str, float]:
    """
    Build a random truncation validation set:
    - fail class: prefixes sampled from the last 20% (or configured tail) of fail trajectories
    - success class: prefixes sampled from expert trajectories
    """
    rng = np.random.default_rng(int(seed))

    y_true: list[int] = []
    y_pred: list[int] = []

    # Fail samples (label=1): sample 80%+n% prefixes
    for emb in fail_embeddings:
        out = evaluator.run_episode(rollout_embeddings=emb, threshold=threshold)
        sampled_idx = _sample_tail_indices(
            length=emb.shape[0],
            start_ratio=float(fail_start_ratio),
            num_samples=int(samples_per_trajectory),
            rng=rng,
        )
        for idx in sampled_idx.tolist():
            y_true.append(1)
            y_pred.append(int(out.step_failure_flags[int(idx)]))

    # Expert samples (label=0): sample late prefixes to match truncation regime
    for emb in expert_embeddings:
        out = evaluator.run_episode(rollout_embeddings=emb, threshold=threshold)
        sampled_idx = _sample_tail_indices(
            length=emb.shape[0],
            start_ratio=float(expert_start_ratio),
            num_samples=int(samples_per_trajectory),
            rng=rng,
        )
        for idx in sampled_idx.tolist():
            y_true.append(0)
            y_pred.append(int(out.step_failure_flags[int(idx)]))

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    metrics = classification_metrics(y_true=y_true_arr, y_pred=y_pred_arr)
    metrics["success_rate"] = float(metrics["accuracy"])
    metrics["num_samples"] = float(y_true_arr.size)
    metrics["fail_samples"] = float(np.sum(y_true_arr == 1))
    metrics["expert_samples"] = float(np.sum(y_true_arr == 0))
    metrics["fail_start_ratio"] = float(fail_start_ratio)
    metrics["expert_start_ratio"] = float(expert_start_ratio)
    metrics["samples_per_trajectory"] = float(samples_per_trajectory)
    return metrics


def _compute_threshold(scores: list[float], delta: float) -> float:
    if not scores:
        raise ValueError("Cannot compute threshold from empty calibration scores")
    if delta < 0 or delta > 100:
        raise ValueError(f"delta must be in [0,100], got {delta}")
    q = 100.0 * (1.0 - float(delta) / 100.0)
    return float(np.percentile(np.asarray(scores, dtype=np.float64), q=q))


def _encode_rollouts(extractor: FlowPolicyLatentExtractor, trajectories: list) -> list[np.ndarray]:
    return [extractor.encode_trajectory(t) for t in trajectories]


def _resolve_torch_device(preferred: str) -> str:
    import warnings

    import torch

    pref = str(preferred).lower()
    if pref.startswith("cuda"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA initialization:.*")
            try:
                if not torch.cuda.is_available():
                    return "cpu"
            except Exception:
                return "cpu"
    return preferred


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_discriminator")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))

    if str(cfg.data.obs_key) != "states":
        raise ValueError("Official-like evaluator requires states for proprio extraction")
    if cfg.data.camera_name in (None, ""):
        raise ValueError("data.camera_name is required to read image observations")
    if cfg.policy.ckpt in (None, ""):
        raise ValueError("policy.ckpt is required for policy latent extraction")

    expert_dir = to_absolute_path(cfg.data.expert_dir)
    fail_dir = to_absolute_path(cfg.data.fail_rollout_dir)
    success_dir = to_absolute_path(cfg.data.success_rollout_dir) if cfg.data.success_rollout_dir else ""
    policy_ckpt = to_absolute_path(cfg.policy.ckpt)
    policy_device = _resolve_torch_device(str(cfg.policy.device))
    camera_name = str(cfg.data.camera_name)

    expert_rollouts = load_policy_trajectories(
        data_dir=expert_dir,
        camera_name=camera_name,
        max_trajectories=(None if int(cfg.data.max_expert_trajectories) <= 0 else int(cfg.data.max_expert_trajectories)),
    )
    fail_rollouts = load_policy_trajectories(
        data_dir=fail_dir,
        camera_name=camera_name,
        max_trajectories=(None if int(cfg.data.max_fail_trajectories) <= 0 else int(cfg.data.max_fail_trajectories)),
    )

    success_rollouts = []
    if success_dir and os.path.isdir(success_dir):
        success_rollouts = load_policy_trajectories(
            data_dir=success_dir,
            camera_name=camera_name,
            max_trajectories=(
                None if int(cfg.data.max_success_trajectories) <= 0 else int(cfg.data.max_success_trajectories)
            ),
        )

    expert_train, expert_val = split_train_val(expert_rollouts, val_ratio=float(cfg.split.val_ratio), seed=int(cfg.seed))
    fail_train, fail_val = split_train_val(fail_rollouts, val_ratio=float(cfg.split.val_ratio), seed=int(cfg.seed) + 1)

    if success_rollouts:
        success_train, success_val = split_train_val(success_rollouts, val_ratio=float(cfg.split.val_ratio), seed=int(cfg.seed) + 2)
    else:
        success_train, success_val = [], []

    if not expert_train:
        raise RuntimeError("No expert training trajectories available after split")
    if not fail_val:
        raise RuntimeError("No fail validation trajectories available after split")

    calibration_rollouts = success_train if success_train else expert_train
    if not success_train:
        print("No rollout success data provided. Calibrating threshold with expert trajectories only (surrogate).")

    extractor = FlowPolicyLatentExtractor(
        ckpt_path=policy_ckpt,
        camera_name=camera_name,
        image_size=int(cfg.policy.image_size),
        ta=int(cfg.float.ta),
        to=int(cfg.float.to),
        device=policy_device,
        batch_size=int(cfg.policy.latent_batch_size),
        history_len=int(cfg.policy.history_len),
        robots=str(cfg.policy.robots),
        env_name=str(cfg.policy.env_name),
    )

    try:
        expert_train_emb = _encode_rollouts(extractor, expert_train)
        expert_val_emb = _encode_rollouts(extractor, expert_val)
        fail_val_emb = _encode_rollouts(extractor, fail_val)
        calibration_emb = _encode_rollouts(extractor, calibration_rollouts)
        success_eval_emb = _encode_rollouts(extractor, success_val if success_val else expert_val)
    finally:
        extractor.close()

    inferred_max_steps = max(max(x.shape[0] for x in expert_train_emb), 1)
    max_steps = inferred_max_steps if int(cfg.float.max_steps) <= 0 else int(cfg.float.max_steps)

    evaluator = OfficialFloatOfflineEvaluator(
        expert_embeddings=expert_train_emb,
        sinkhorn_reg=float(cfg.float.sinkhorn_reg),
        max_iter=int(cfg.float.max_iter),
        tol=float(cfg.float.tol),
        num_expert_candidates=int(cfg.float.num_expert_candidates),
        max_steps=max_steps,
        use_similarity_cost=bool(cfg.float.use_similarity_cost),
    )

    calibration_scores = [evaluator.episode_score(x) for x in calibration_emb]
    delta = float(cfg.calibration.delta)
    delta_step = float(cfg.calibration.delta_step)
    threshold = _compute_threshold(calibration_scores, delta)

    fail_records: list[PrefixRecord] = []

    for traj_id, emb in enumerate(fail_val_emb):
        out = evaluator.run_episode(rollout_embeddings=emb, threshold=threshold)
        labels = fail_prefix_labels(length=emb.shape[0], fail_tail_ratio=float(cfg.labels.fail_tail_ratio))

        for step in range(emb.shape[0]):
            pred = int(out.step_failure_flags[step])
            label = int(labels[step])
            fail_records.append(
                PrefixRecord(
                    trajectory_id=traj_id,
                    step=step + 1,
                    cumulative_cost=float(out.cumulative_costs[step]),
                    label=label,
                    pred=pred,
                )
            )

            if bool(cfg.online.adaptive_delta):
                if label == 1 and pred == 0:
                    delta -= delta_step
                elif label == 0 and pred == 1:
                    delta += delta_step
                delta = float(np.clip(delta, 0.0, 100.0))
                threshold = _compute_threshold(calibration_scores, delta)

    y_true = np.asarray([r.label for r in fail_records], dtype=np.int64)
    y_pred = np.asarray([r.pred for r in fail_records], dtype=np.int64)
    fail_metrics = classification_metrics(y_true=y_true, y_pred=y_pred)
    fail_metrics["delta_final"] = float(delta)
    fail_metrics["threshold_final"] = float(threshold)

    fail_costs = [r.cumulative_cost for r in fail_records if r.label == 1]
    pre_fail_costs = [r.cumulative_cost for r in fail_records if r.label == 0]
    lambda_stats = {
        "lambda_fail_mean": float(np.mean(fail_costs)) if fail_costs else float("nan"),
        "lambda_success_mean": float(np.mean(pre_fail_costs)) if pre_fail_costs else float("nan"),
        "lambda_gap": (
            float(np.mean(fail_costs) - np.mean(pre_fail_costs)) if fail_costs and pre_fail_costs else float("nan")
        ),
    }

    # success validation (all labels=0)
    success_total = 0
    success_false_alarm = 0
    success_costs: list[float] = []

    for emb in success_eval_emb:
        out = evaluator.run_episode(rollout_embeddings=emb, threshold=threshold)
        success_total += emb.shape[0]
        success_false_alarm += int(np.sum(out.step_failure_flags))
        success_costs.extend([float(x) for x in out.cumulative_costs])

    success_metrics = {
        "success_prefixes": float(success_total),
        "false_alarms": float(success_false_alarm),
        "false_alarm_rate": float(success_false_alarm / success_total) if success_total > 0 else float("nan"),
        "lambda_success_mean": float(np.mean(success_costs)) if success_costs else float("nan"),
    }

    random_truncation_metrics = {}
    if bool(cfg.random_truncation_eval.enabled):
        random_truncation_metrics = _evaluate_random_truncation_success_rate(
            evaluator=evaluator,
            threshold=threshold,
            fail_embeddings=fail_val_emb,
            expert_embeddings=expert_val_emb,
            fail_start_ratio=float(cfg.random_truncation_eval.fail_start_ratio),
            expert_start_ratio=float(cfg.random_truncation_eval.expert_start_ratio),
            samples_per_trajectory=int(cfg.random_truncation_eval.samples_per_trajectory),
            seed=int(cfg.random_truncation_eval.seed),
        )

    result = {
        "threshold": float(threshold),
        "delta_init": float(cfg.calibration.delta),
        "delta_final": float(delta),
        "counts": {
            "expert_total": len(expert_rollouts),
            "expert_train": len(expert_train),
            "expert_val": len(expert_val),
            "fail_total": len(fail_rollouts),
            "fail_train": len(fail_train),
            "fail_val": len(fail_val),
            "success_total": len(success_rollouts),
            "success_train": len(success_train),
            "success_val": len(success_val),
        },
        "float_runtime": {
            "ta": int(cfg.float.ta),
            "to": int(cfg.float.to),
            "num_expert_candidates": int(cfg.float.num_expert_candidates),
            "max_steps": int(max_steps),
            "policy_ckpt": policy_ckpt,
            "policy_device": policy_device,
            "policy_latent_source": "flow_cond_token",
        },
        "calibration_scores": {
            "num_scores": len(calibration_scores),
            "mean": float(np.mean(calibration_scores)) if calibration_scores else float("nan"),
            "std": float(np.std(calibration_scores)) if calibration_scores else float("nan"),
        },
        "fail_validation": fail_metrics,
        "success_validation": success_metrics,
        "lambda_separation": lambda_stats,
        "random_truncation_eval": random_truncation_metrics,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    save_path = str(cfg.output.save_json_path)
    if save_path:
        out_fp = to_absolute_path(save_path)
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Saved evaluation summary to: {out_fp}")


if __name__ == "__main__":
    main()
