from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.float_data import fail_prefix_labels, load_policy_trajectories, split_train_val
from robosuite.discriminator.float_eval import classification_metrics
from robosuite.discriminator.float_official import OfficialFloatOfflineEvaluator
from robosuite.discriminator.float_policy_latent import FlowPolicyLatentExtractor


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _threshold(scores: list[float], delta: float) -> float:
    q = 100.0 * (1.0 - float(delta) / 100.0)
    return float(np.percentile(np.asarray(scores, dtype=np.float64), q=q))


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


@hydra.main(version_base="1.2", config_path="./config", config_name="train_discriminator_intervention")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))

    if str(cfg.data.obs_key) != "states":
        raise ValueError("Official-like training requires states for proprio extraction")
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

    successes = []
    if success_dir and os.path.isdir(success_dir):
        successes = load_policy_trajectories(
            data_dir=success_dir,
            camera_name=camera_name,
            max_trajectories=(
                None if int(cfg.data.max_success_trajectories) <= 0 else int(cfg.data.max_success_trajectories)
            ),
        )

    expert_train, expert_val = split_train_val(experts, val_ratio=float(cfg.split.val_ratio), seed=int(cfg.seed))
    fail_train, fail_val = split_train_val(fails, val_ratio=float(cfg.split.val_ratio), seed=int(cfg.seed) + 1)

    if successes:
        success_train, success_val = split_train_val(successes, val_ratio=float(cfg.split.val_ratio), seed=int(cfg.seed) + 2)
    else:
        success_train, success_val = [], []

    calibration_rollouts = success_train if success_train else expert_train
    if not success_train:
        print("No rollout success data provided. Calibrating threshold using expert trajectories only.")

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
        expert_emb = [extractor.encode_trajectory(t) for t in expert_train]
        calibration_emb = [extractor.encode_trajectory(t) for t in calibration_rollouts]
        fail_train_emb = [extractor.encode_trajectory(t) for t in fail_train]
        fail_val_emb = [extractor.encode_trajectory(t) for t in fail_val]
    finally:
        extractor.close()

    max_steps = max(max(x.shape[0] for x in expert_emb), 1)
    if int(cfg.float.max_steps) > 0:
        max_steps = int(cfg.float.max_steps)

    evaluator = OfficialFloatOfflineEvaluator(
        expert_embeddings=expert_emb,
        sinkhorn_reg=float(cfg.float.sinkhorn_reg),
        max_iter=int(cfg.float.max_iter),
        tol=float(cfg.float.tol),
        num_expert_candidates=int(cfg.float.num_expert_candidates),
        max_steps=max_steps,
        use_similarity_cost=bool(cfg.float.use_similarity_cost),
    )

    calibration_scores = [evaluator.episode_score(x) for x in calibration_emb]
    threshold = _threshold(calibration_scores, float(cfg.calibration.delta))

    def eval_split(rollout_embeddings: list[np.ndarray]):
        if not rollout_embeddings:
            return {}
        y_true = []
        y_pred = []
        for emb in rollout_embeddings:
            out = evaluator.run_episode(emb, threshold=threshold)
            labels = fail_prefix_labels(len(emb), float(cfg.labels.fail_tail_ratio))
            y_true.extend(labels.tolist())
            y_pred.extend(out.step_failure_flags.tolist())
        m = classification_metrics(np.asarray(y_true), np.asarray(y_pred))
        m["threshold_final"] = float(threshold)
        return m

    train_fail_metrics = eval_split(fail_train_emb)
    val_fail_metrics = eval_split(fail_val_emb)

    summary = {
        "threshold": float(threshold),
        "delta": float(cfg.calibration.delta),
        "counts": {
            "expert_total": len(experts),
            "expert_train": len(expert_train),
            "expert_val": len(expert_val),
            "fail_total": len(fails),
            "fail_train": len(fail_train),
            "fail_val": len(fail_val),
            "success_total": len(successes),
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
        "train_fail_metrics": train_fail_metrics,
        "val_fail_metrics": val_fail_metrics,
    }

    save_dir = to_absolute_path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    tag = _now_tag()
    summary_path = os.path.join(save_dir, f"float_summary_{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    calib_path = os.path.join(save_dir, f"float_calibration_{tag}.npz")
    np.savez(
        calib_path,
        threshold=np.float64(threshold),
        delta=np.float64(cfg.calibration.delta),
        calibration_scores=np.asarray(calibration_scores, dtype=np.float64),
        sinkhorn_reg=np.float64(cfg.float.sinkhorn_reg),
        max_iter=np.int64(cfg.float.max_iter),
        tol=np.float64(cfg.float.tol),
        ta=np.int64(cfg.float.ta),
        to=np.int64(cfg.float.to),
        num_expert_candidates=np.int64(cfg.float.num_expert_candidates),
        max_steps=np.int64(max_steps),
        use_similarity_cost=np.int64(bool(cfg.float.use_similarity_cost)),
        policy_ckpt=np.asarray(policy_ckpt),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved FLOAT summary to: {summary_path}")
    print(f"Saved FLOAT calibration to: {calib_path}")


if __name__ == "__main__":
    main()
