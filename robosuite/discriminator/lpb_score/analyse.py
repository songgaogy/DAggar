from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.lpb_score.app.pipeline import build_dsm_discriminator, build_flow_encoder
from robosuite.discriminator.lpb_score.core.dataset import (
    build_cached_splits,
    filter_refs_by_data_types,
    load_latent_trajectories,
)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Expected non-empty values for summary stats.")
    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        **{f"p{q:02d}": float(np.percentile(arr, q)) for q in quantiles},
    }


def run_analyse(cfg: DictConfig) -> None:
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
        bank_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.bank_split)],
            list(cfg.eval.bank_data_types),
        )
        calibration_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.calibration_split)],
            list(cfg.eval.calibration_data_types),
        )
        bank_trajectories = load_latent_trajectories(bank_refs)
        calibration_trajectories = load_latent_trajectories(calibration_refs)
        if not bank_trajectories:
            raise RuntimeError("Analysis requires non-empty clean bank trajectories.")
        if not calibration_trajectories:
            raise RuntimeError("Analysis requires non-empty clean calibration trajectories.")

        detector = build_dsm_discriminator(cfg)
        calibration_summary = detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )
        lambda_values = np.asarray(detector._calib_lambdas, dtype=np.float32)
        threshold = float(calibration_summary.threshold if calibration_summary.threshold is not None else np.nan)

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        save_name = str(getattr(cfg, "save_name", "") or "").strip()
        if save_name in {"", "None", "null"}:
            save_name = f"lpb_score_dsm_threshold_{_now_tag()}.json"
        if not save_name.endswith(".json"):
            save_name = f"{save_name}.json"
        save_path = os.path.join(save_dir, save_name)

        artifact = {
            "detector_name": detector.name,
            "source_dsm_ckpt": str(cfg.model.dsm_ckpt),
            "delta": float(cfg.detector.delta),
            "threshold": float(threshold),
            "lambda_stats": _summary_stats(lambda_values),
            "counts": {
                "num_bank_trajectories": int(len(bank_trajectories)),
                "num_calibration_trajectories": int(len(calibration_trajectories)),
                "num_lambda_steps": int(lambda_values.size),
            },
            "calibration": {
                "bank_split": str(cfg.eval.bank_split),
                "bank_data_types": list(cfg.eval.bank_data_types),
                "calibration_split": str(cfg.eval.calibration_split),
                "calibration_data_types": list(cfg.eval.calibration_data_types),
                "lambda_mode": str(cfg.detector.lambda_mode),
                "lambda_window_size": int(cfg.detector.lambda_window_size),
            },
            "runtime": dict(calibration_summary.metadata),
            "clean_split_summary": split_summary,
        }

        with open(save_path, "w", encoding="utf-8") as file_handle:
            json.dump(artifact, file_handle, indent=2, sort_keys=True, default=_json_default)

        print(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default))
        print(f"[lpb_score analyse] Saved threshold artifact to: {save_path}")
    finally:
        encoder.close()


@hydra.main(version_base="1.2", config_path="./config", config_name="analyse")
def main(cfg: DictConfig) -> None:
    run_analyse(cfg)


if __name__ == "__main__":
    main()
