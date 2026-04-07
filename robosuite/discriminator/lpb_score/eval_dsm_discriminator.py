from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.lpb_score.app.pipeline import (
    build_dsm_discriminator,
    build_flow_encoder,
    load_eval_trajectory_splits,
)
from robosuite.discriminator.lpb_score.core.dataset import build_cached_splits
from robosuite.discriminator.utils.evaluation import evaluate_trajectory_discriminator


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_fail_tail_labels(length: int, fail_tail_ratio: float) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    tail = max(1, int(np.ceil(float(length) * float(fail_tail_ratio))))
    labels = np.zeros((length,), dtype=np.int64)
    labels[-tail:] = 1
    return labels


def run_eval(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = build_flow_encoder(cfg)
    try:
        cached_splits, split_summary, _ = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=seed,
        )
        splits = load_eval_trajectory_splits(cfg, cached_splits)
        detector = build_dsm_discriminator(cfg)
        summary, calibration_summary, _ = evaluate_trajectory_discriminator(
            detector=detector,
            bank_trajectories=splits.bank,
            calibration_trajectories=splits.calibration,
            expert_eval_trajectories=splits.expert_eval,
            success_eval_trajectories=splits.success_eval,
            fail_eval_trajectories=splits.fail_eval,
            fail_label_builder=lambda _traj, result: _build_fail_tail_labels(
                len(result.aggregate_scores),
                fail_tail_ratio=float(cfg.labels.fail_tail_ratio),
            ),
            adaptive_threshold=bool(cfg.online.adaptive_delta),
            delta_min=float(cfg.online.delta_min),
            delta_max=float(cfg.online.delta_max),
            warmup_steps=int(cfg.online.warmup_steps),
            update_interval=int(cfg.online.update_interval),
        )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, f"summary_{_now_tag()}.json")
        payload = {
            **summary,
            "split_summary": split_summary,
            "runtime": {
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "dsm_ckpt": to_absolute_path(str(cfg.model.dsm_ckpt)),
                **dict(summary.get("runtime", {})),
                "threshold_init": (
                    float(calibration_summary.threshold)
                    if calibration_summary.threshold is not None
                    else float("nan")
                ),
            },
        }
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)

        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[lpb_score eval] Saved summary to: {summary_path}")
    finally:
        encoder.close()


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_dsm_discriminator")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
