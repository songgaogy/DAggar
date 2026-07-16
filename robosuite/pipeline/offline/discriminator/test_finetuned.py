"""CUDA-only diagnostic: does a finetuned nnPU head label GT-fail frames "fail"?

This entry point reproduces the exact GT-negative (GT-fail) frames used during
offline finetuning, scores them with the finetuned checkpoint, and reports how
many are predicted "fail". The decision rule mirrors the detector
(``robosuite/discriminator/dyn_disc/detectors/pu_bce.py``):

    failure_score = -head_logit
    pred_fail     = failure_score >= tau_task

Run with::

    python -m robosuite.pipeline.offline.discriminator.test_finetuned \
        --load-ckpt <pu_bce_head_finetuned.pth> \
        --model-ckpt <dynamics model.pth> \
        --offline-episodes <offline_episodes.pt> \
        --task PickPlaceCereal --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce import (
    _parse_camera_to_view,
)

from .checkpoint import load_warmstart_detector
from .encoder import FinetuneDynamicsEncoder
from .episodes import build_gt_negative_windows, load_offline_episodes
from .features import encode_gt_negative_windows
from .trainer import require_cuda_device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a finetuned nnPU head labels GT-fail frames 'fail'."
    )
    parser.add_argument("--load-ckpt", required=True, help="pu_bce_head_finetuned.pth")
    parser.add_argument("--model-ckpt", required=True, help="frozen dynamics encoder")
    parser.add_argument("--offline-episodes", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report-name", default="gt_fail_detection.json")
    parser.add_argument("--camera-to-view", default=None)
    parser.add_argument("--seed", type=int, default=0)
    # GT-window overrides; when omitted the checkpoint's finetune config is used
    # so the evaluated frames match training exactly.
    parser.add_argument("--pre-intervention-chunks", type=int, default=None)
    parser.add_argument("--post-intervention-chunks", type=int, default=None)
    parser.add_argument("--pre-end-chunks", type=int, default=None)
    return parser.parse_args()


def _resolved_gt_config(
    payload: dict[str, Any], args: argparse.Namespace
) -> tuple[int, int, int]:
    """Resolve GT-window chunk counts from CLI overrides or the checkpoint."""
    finetune_config = payload.get("finetune_config", {}) or {}
    gt_config = finetune_config.get("gt_negative", {}) or {}
    pre_intervention = (
        int(args.pre_intervention_chunks)
        if args.pre_intervention_chunks is not None
        else int(gt_config.get("pre_intervention_chunks", 3))
    )
    post_intervention = (
        int(args.post_intervention_chunks)
        if args.post_intervention_chunks is not None
        else int(gt_config.get("post_intervention_chunks", 3))
    )
    pre_end = (
        int(args.pre_end_chunks)
        if args.pre_end_chunks is not None
        else int(gt_config.get("pre_end_chunk", 0))
    )
    if pre_intervention < 0 or post_intervention <= 0 or pre_end < 0:
        raise ValueError(
            "GT-window chunks must satisfy pre_intervention>=0, post_intervention>0, "
            f"pre_end>=0; got {pre_intervention}, {post_intervention}, {pre_end}."
        )
    return pre_intervention, post_intervention, pre_end


def _score_summary(failure_scores: np.ndarray, tau: float) -> dict[str, float]:
    """Detection rate and score statistics for one group of GT-fail frames."""
    n_frames = int(failure_scores.size)
    if n_frames == 0:
        return {
            "n_frames": 0,
            "n_fail": 0,
            "detection_rate": float("nan"),
            "score_mean": float("nan"),
            "score_std": float("nan"),
            "score_min": float("nan"),
            "score_median": float("nan"),
            "score_max": float("nan"),
            "mean_margin": float("nan"),
        }
    predicted_fail = failure_scores >= float(tau)
    scores = failure_scores.astype(np.float64)
    return {
        "n_frames": n_frames,
        "n_fail": int(predicted_fail.sum()),
        "detection_rate": float(predicted_fail.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
        "mean_margin": float((scores - float(tau)).mean()),
    }


def main() -> None:
    args = _parse_args()
    device = require_cuda_device(str(args.device))
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    task_name = str(args.task)
    load_ckpt = str(Path(args.load_ckpt).expanduser().resolve())
    model_ckpt = str(Path(args.model_ckpt).expanduser().resolve())

    detector, payload = load_warmstart_detector(
        load_ckpt,
        device=device,
        expected_task=task_name,
    )
    detector.head.eval()
    tau = float(detector.thresholds[task_name])

    requested_camera_map = _parse_camera_to_view(args.camera_to_view)
    with torch.device(device):
        encoder = FinetuneDynamicsEncoder(
            nnpu_ckpt_path=load_ckpt,
            encoder_ckpt=model_ckpt,
            device=device,
            camera_to_view=requested_camera_map,
            proprio_indices=None,
        )
    frameskip = int(encoder.inner_encoder.frameskip)

    pre_intervention, post_intervention, pre_end = _resolved_gt_config(payload, args)

    offline_payload = load_offline_episodes(args.offline_episodes)
    windows, window_stats = build_gt_negative_windows(
        offline_payload,
        pre_intervention_chunks=pre_intervention,
        post_intervention_chunks=post_intervention,
        frameskip=frameskip,
        pre_end_chunks=pre_end,
    )
    if not windows:
        raise RuntimeError(
            "No GT-fail windows were produced from the offline episodes; nothing to test."
        )
    trajectories = encode_gt_negative_windows(
        windows,
        encoder=encoder,
        camera_names=list(offline_payload["camera_names"]),
        batch_size=int(args.encode_batch_size),
    )

    all_scores: list[np.ndarray] = []
    kind_scores: dict[str, list[np.ndarray]] = {"intervention": [], "pre_end": []}
    position_scores: dict[str, list[np.ndarray]] = {
        "pre": [],
        "post": [],
        "pre_end": [],
    }
    per_window: list[dict[str, Any]] = []

    for trajectory in trajectories:
        metadata = trajectory.metadata
        kind = str(metadata.get("gt_negative_kind", "intervention"))
        logits = detector._logits_np(trajectory.features)  # noqa: SLF001
        failure_scores = (-logits).astype(np.float32)
        all_scores.append(failure_scores)
        kind_scores.setdefault(kind, []).append(failure_scores)

        frame_indices = np.asarray(
            metadata.get("frame_indices", []), dtype=np.int64
        )
        onset = int(metadata.get("intervention_start", 0))
        if kind == "pre_end":
            position_scores["pre_end"].append(failure_scores)
        else:
            pre_mask = frame_indices < onset
            position_scores["pre"].append(failure_scores[pre_mask])
            position_scores["post"].append(failure_scores[~pre_mask])

        window_summary = _score_summary(failure_scores, tau)
        per_window.append(
            {
                "identifier": trajectory.identifier,
                "kind": kind,
                "source_episode_index": int(
                    metadata.get("source_episode_index", -1)
                ),
                "n_frames": window_summary["n_frames"],
                "n_fail": window_summary["n_fail"],
                "detection_rate": window_summary["detection_rate"],
                "mean_failure_score": window_summary["score_mean"],
            }
        )

    def _pool(groups: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(groups, axis=0)
            if groups
            else np.zeros((0,), dtype=np.float32)
        )

    overall = _score_summary(_pool(all_scores), tau)
    by_kind = {
        name: _score_summary(_pool(groups), tau)
        for name, groups in kind_scores.items()
    }
    by_position = {
        name: _score_summary(_pool(groups), tau)
        for name, groups in position_scores.items()
    }

    report: dict[str, Any] = {
        "task": task_name,
        "tau": tau,
        "decision_rule": "pred_fail = (failure_score >= tau); failure_score = -logit",
        "note": "train-set diagnostic: GT-fail frames come from the offline episodes.",
        "data_source": {
            "finetuned_checkpoint": load_ckpt,
            "model_checkpoint": model_ckpt,
            "offline_episodes": str(offline_payload.get("_resolved_path", "")),
        },
        "gt_window_config": {
            "pre_intervention_chunks": pre_intervention,
            "post_intervention_chunks": post_intervention,
            "pre_end_chunks": pre_end,
            "frameskip": frameskip,
            "source": "cli_override" if any(
                value is not None
                for value in (
                    args.pre_intervention_chunks,
                    args.post_intervention_chunks,
                    args.pre_end_chunks,
                )
            ) else "finetuned_checkpoint",
        },
        "gt_window_stats": {
            "intervention_events": int(window_stats["intervention_events"]),
            "gt_negative_windows": int(window_stats["gt_negative_windows"]),
            "non_success_episodes": int(window_stats.get("non_success_episodes", 0)),
            "pre_end_windows": int(window_stats.get("pre_end_windows", 0)),
        },
        "overall": overall,
        "by_kind": by_kind,
        "by_position": by_position,
        "per_window": per_window,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / str(args.report_name)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    def _fmt(summary: dict[str, float]) -> str:
        if int(summary["n_frames"]) == 0:
            return "n=0"
        return (
            f"det_rate={summary['detection_rate']:.3f} "
            f"({summary['n_fail']}/{summary['n_frames']}) "
            f"score_mean={summary['score_mean']:+.4f} "
            f"margin={summary['mean_margin']:+.4f}"
        )

    print(f"[gt_fail_test] task={task_name} tau={tau:+.5f}", flush=True)
    print(
        f"[gt_fail_test] frameskip={frameskip} "
        f"gt_windows: pre_int={pre_intervention} post_int={post_intervention} "
        f"pre_end={pre_end}",
        flush=True,
    )
    print(f"[gt_fail_test] overall     {_fmt(overall)}", flush=True)
    for name in ("intervention", "pre_end"):
        print(f"[gt_fail_test] kind:{name:<9} {_fmt(by_kind[name])}", flush=True)
    for name in ("pre", "post", "pre_end"):
        print(f"[gt_fail_test] pos:{name:<10} {_fmt(by_position[name])}", flush=True)
    print(f"[gt_fail_test] wrote report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
