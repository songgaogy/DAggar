"""CUDA-only GT-fail training-set evaluation for finetuned nnPU heads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce import (
    _parse_camera_to_view,
)

from .checkpoint import load_warmstart_detector
from .encoder import FinetuneDynamicsEncoder
from .episodes import build_gt_negative_windows, load_offline_episodes
from .features import encode_gt_negative_windows
from .trainer import require_cuda_device


REPORT_SCHEMA_VERSION = 2
_SUMMARY_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the exact GT-negative training frames stored by a finetuned "
            "nnPU checkpoint."
        )
    )
    parser.add_argument("--load-ckpt", required=True)
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--offline-episodes", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report-name", default="gt_fail_detection.json")
    parser.add_argument("--camera-to-view", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _resolve_training_contract(
    payload: Mapping[str, Any], *, task_name: str
) -> tuple[dict[str, int], Mapping[str, Any], Mapping[str, Any]]:
    """Read the exact GT-window rule and provenance from a finetuned checkpoint."""
    if not bool(payload.get("finetuned_offline", False)):
        raise ValueError(
            "GT-fail training-set evaluation is not applicable to a parent checkpoint."
        )
    if int(payload.get("finetune_schema_version", -1)) < 3:
        raise ValueError("Unsupported finetuned checkpoint schema for GT-fail evaluation.")
    if str(payload.get("finetune_task")) != str(task_name):
        raise ValueError(
            "Finetuned checkpoint task differs from the GT-fail evaluation task: "
            f"checkpoint={payload.get('finetune_task')!r}, runtime={task_name!r}."
        )
    finetune_config = _required_mapping(
        payload.get("finetune_config"), name="finetune_config"
    )
    gt_config = _required_mapping(
        finetune_config.get("gt_negative"), name="finetune_config.gt_negative"
    )
    required_semantics = {
        "action_source": "policy_action",
        "post_boundary": "intervention_block",
        "chunk_boundary": "continuous_window",
    }
    for key, expected in required_semantics.items():
        if gt_config.get(key) != expected:
            raise ValueError(
                f"finetune_config.gt_negative.{key} must be {expected!r}, "
                f"got {gt_config.get(key)!r}."
            )
    resolved = {
        "pre_intervention_chunks": int(gt_config["pre_intervention_chunks"]),
        "post_intervention_chunks": int(gt_config["post_intervention_chunks"]),
        "pre_end_chunks": int(gt_config["pre_end_chunk"]),
    }
    if (
        resolved["pre_intervention_chunks"] < 0
        or resolved["post_intervention_chunks"] <= 0
        or resolved["pre_end_chunks"] < 0
    ):
        raise ValueError(f"Invalid checkpoint GT-negative window config: {resolved}.")

    finetune_data = _required_mapping(
        payload.get("finetune_data"), name="finetune_data"
    )
    expected_rule = _required_mapping(
        finetune_data.get("gt_negative_rule"),
        name="finetune_data.gt_negative_rule",
    )
    named_pool_stats = _required_mapping(
        finetune_data.get("named_pool_stats"),
        name="finetune_data.named_pool_stats",
    )
    expected_pool = _required_mapping(
        named_pool_stats.get("offline_gt_negative"),
        name="finetune_data.named_pool_stats.offline_gt_negative",
    )
    return resolved, expected_rule, expected_pool


def _canonical_frame_keys(value: Any, *, name: str) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of [episode, frame] pairs.")
    result: list[list[int]] = []
    for item in value:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise ValueError(f"{name} contains an invalid frame key: {item!r}.")
        result.append([int(item[0]), int(item[1])])
    return result


def validate_rebuilt_gt_provenance(
    rebuilt: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Require the reconstructed evaluation pool to equal the training pool."""
    scalar_fields = (
        "intervention_events",
        "gt_negative_windows",
        "gt_negative_frames",
        "pre_frames",
        "post_frames",
        "pre_end_windows",
        "pre_end_frames",
        "deduplicated_frames",
    )
    for field in scalar_fields:
        if field not in expected:
            raise KeyError(f"Checkpoint GT-negative provenance is missing {field!r}.")
        if int(rebuilt.get(field, -1)) != int(expected[field]):
            raise ValueError(
                "Rebuilt GT-negative provenance differs from training: "
                f"{field}={rebuilt.get(field)!r}, expected={expected[field]!r}."
            )
    rebuilt_keys = _canonical_frame_keys(
        rebuilt.get("selected_frame_keys"), name="rebuilt.selected_frame_keys"
    )
    expected_keys = _canonical_frame_keys(
        expected.get("selected_frame_keys"),
        name="checkpoint.selected_frame_keys",
    )
    if rebuilt_keys != expected_keys:
        raise ValueError(
            "Rebuilt GT-negative selected_frame_keys differ from the training checkpoint."
        )


def _empty_summary() -> dict[str, int | float | None]:
    return {
        "n_frames": 0,
        "n_fail": 0,
        "n_missed": 0,
        "detection_rate": None,
        "miss_rate": None,
        "score_mean": None,
        "score_std": None,
        "score_min": None,
        "score_q01": None,
        "score_q05": None,
        "score_median": None,
        "score_q95": None,
        "score_q99": None,
        "score_max": None,
        "margin_mean": None,
        "margin_min": None,
        "margin_q01": None,
        "margin_q05": None,
        "margin_median": None,
    }


def score_summary_cuda(
    failure_scores: torch.Tensor, threshold: torch.Tensor | float
) -> dict[str, int | float | None]:
    """Summarize failure decisions without moving tensors to CPU."""
    if failure_scores.device.type != "cuda":
        raise ValueError("GT-fail score summaries require CUDA tensors.")
    scores = failure_scores.detach().reshape(-1).to(dtype=torch.float32)
    if scores.numel() == 0:
        return _empty_summary()
    if not bool(torch.isfinite(scores).all().item()):
        raise FloatingPointError("GT-fail failure scores contain non-finite values.")
    tau = torch.as_tensor(
        threshold, device=scores.device, dtype=scores.dtype
    ).reshape(())
    if not bool(torch.isfinite(tau).item()):
        raise FloatingPointError("GT-fail threshold is non-finite.")
    predicted_fail = scores >= tau
    margins = scores - tau
    if not bool(torch.isfinite(margins).all().item()):
        raise FloatingPointError("GT-fail score margins contain non-finite values.")
    quantile_levels = scores.new_tensor(_SUMMARY_QUANTILES)
    score_quantiles = torch.quantile(scores, quantile_levels)
    margin_quantiles = torch.quantile(margins, quantile_levels[:3])
    n_frames = int(scores.numel())
    n_fail = int(predicted_fail.sum().item())
    return {
        "n_frames": n_frames,
        "n_fail": n_fail,
        "n_missed": n_frames - n_fail,
        "detection_rate": float(predicted_fail.float().mean().item()),
        "miss_rate": float((~predicted_fail).float().mean().item()),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std(unbiased=False).item()),
        "score_min": float(scores.min().item()),
        "score_q01": float(score_quantiles[0].item()),
        "score_q05": float(score_quantiles[1].item()),
        "score_median": float(score_quantiles[2].item()),
        "score_q95": float(score_quantiles[3].item()),
        "score_q99": float(score_quantiles[4].item()),
        "score_max": float(scores.max().item()),
        "margin_mean": float(margins.mean().item()),
        "margin_min": float(margins.min().item()),
        "margin_q01": float(margin_quantiles[0].item()),
        "margin_q05": float(margin_quantiles[1].item()),
        "margin_median": float(margin_quantiles[2].item()),
    }


def _pool_cuda(groups: Sequence[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    if not groups:
        return torch.empty((0,), device=device, dtype=torch.float32)
    if any(group.device.type != "cuda" for group in groups):
        raise ValueError("GT-fail score pooling requires CUDA tensors.")
    return torch.cat([group.reshape(-1) for group in groups], dim=0)


@torch.inference_mode()
def build_gt_fail_training_report(
    *,
    detector: Any,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_path: str | Path,
    model_checkpoint: str | Path,
    offline_episodes_path: str | Path,
    task_name: str,
    device: torch.device,
    encode_batch_size: int,
    camera_to_view: Mapping[str, str] | None,
    proprio_indices: Sequence[int] | None,
    seed: int,
) -> dict[str, Any]:
    """Rebuild and score the exact GT-negative training pool on CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GT-fail training-set evaluation requires CUDA.")
    gt_config, expected_rule, expected_pool = _resolve_training_contract(
        checkpoint_payload, task_name=str(task_name)
    )
    if str(task_name) not in detector.thresholds:
        raise KeyError(f"Checkpoint has no threshold for task {task_name!r}.")
    tau = detector.threshold_tensor(
        torch.empty((0,), device=device, dtype=torch.float32), str(task_name)
    )

    with torch.device(device):
        encoder = FinetuneDynamicsEncoder(
            nnpu_ckpt_path=str(Path(checkpoint_path).expanduser().resolve()),
            encoder_ckpt=str(Path(model_checkpoint).expanduser().resolve()),
            device=device,
            camera_to_view=(
                None
                if camera_to_view is None
                else {str(key): str(value) for key, value in camera_to_view.items()}
            ),
            proprio_indices=(
                None
                if proprio_indices is None
                else [int(value) for value in proprio_indices]
            ),
        )
    offline_payload = load_offline_episodes(offline_episodes_path)
    windows, window_stats = build_gt_negative_windows(
        offline_payload,
        pre_intervention_chunks=gt_config["pre_intervention_chunks"],
        post_intervention_chunks=gt_config["post_intervention_chunks"],
        frameskip=int(encoder.inner_encoder.frameskip),
        pre_end_chunks=gt_config["pre_end_chunks"],
    )
    validate_rebuilt_gt_provenance(window_stats, expected_rule)
    if not windows:
        raise RuntimeError("The checkpoint training provenance contains no GT-fail windows.")
    trajectories = encode_gt_negative_windows(
        windows,
        encoder=encoder,
        camera_names=list(offline_payload["camera_names"]),
        batch_size=int(encode_batch_size),
    )
    if len(trajectories) != int(window_stats["gt_negative_windows"]):
        raise ValueError("Encoded GT-negative window count differs from provenance.")
    expected_trajectories = int(expected_pool.get("trajectories", -1))
    expected_frames = int(expected_pool.get("frames", -1))
    expected_latent_dim = int(expected_pool.get("latent_dim", -1))
    if expected_trajectories != len(trajectories):
        raise ValueError("Encoded GT-negative trajectory count differs from training.")
    if expected_frames != int(window_stats["gt_negative_frames"]):
        raise ValueError("GT-negative pool frame count differs from training.")
    if expected_latent_dim != int(detector.in_dim):
        raise ValueError("GT-negative pool latent dimension differs from the checkpoint head.")
    for trajectory in trajectories:
        if int(trajectory.features.shape[-1]) != expected_latent_dim:
            raise ValueError(
                f"Window {trajectory.identifier!r} latent dimension differs from training."
            )

    all_scores: list[torch.Tensor] = []
    kind_scores: dict[str, list[torch.Tensor]] = {
        "intervention": [],
        "pre_end": [],
    }
    position_scores: dict[str, list[torch.Tensor]] = {
        "pre": [],
        "post": [],
        "pre_end": [],
    }
    per_window: list[dict[str, Any]] = []
    detector.head.eval()
    for trajectory in trajectories:
        metadata = trajectory.metadata
        features = trajectory.features.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        failure_scores = detector.failure_score_tensor(features).reshape(-1)
        if failure_scores.device.type != "cuda":
            raise RuntimeError("Detector returned non-CUDA GT-fail scores.")
        kind = str(metadata.get("gt_negative_kind", "intervention"))
        if kind not in kind_scores:
            raise ValueError(f"Unknown GT-negative kind {kind!r}.")
        all_scores.append(failure_scores)
        kind_scores[kind].append(failure_scores)
        frame_indices = torch.as_tensor(
            metadata.get("frame_indices", []),
            device=device,
            dtype=torch.long,
        )
        if frame_indices.numel() != failure_scores.numel():
            raise ValueError(
                f"Window {trajectory.identifier!r} frame/score lengths differ."
            )

        if kind == "pre_end":
            position_scores["pre_end"].append(failure_scores)
        else:
            onset = int(metadata["intervention_start"])
            pre_mask = frame_indices < onset
            position_scores["pre"].append(failure_scores[pre_mask])
            position_scores["post"].append(failure_scores[~pre_mask])

        summary = score_summary_cuda(failure_scores, tau)
        per_window.append(
            {
                "identifier": trajectory.identifier,
                "kind": kind,
                "source_episode_index": int(
                    metadata.get("source_episode_index", -1)
                ),
                "frame_indices": frame_indices.detach().cpu().tolist(),
                "missed_frame_indices": frame_indices[
                    failure_scores < tau
                ].detach().cpu().tolist(),
                **summary,
            }
        )

    overall = score_summary_cuda(_pool_cuda(all_scores, device=device), tau)
    if int(overall["n_frames"]) != int(window_stats["gt_negative_frames"]):
        raise ValueError("Scored GT-negative frame count differs from provenance.")
    by_kind = {
        name: score_summary_cuda(_pool_cuda(groups, device=device), tau)
        for name, groups in kind_scores.items()
    }
    by_position = {
        name: score_summary_cuda(_pool_cuda(groups, device=device), tau)
        for name, groups in position_scores.items()
    }

    threshold_value = float(tau.item())
    if not math.isfinite(threshold_value):
        raise FloatingPointError("GT-fail threshold is non-finite.")
    expected_offline_path = checkpoint_payload.get("finetune_data", {}).get(
        "offline_episodes_path"
    )
    resolved_offline_path = str(Path(offline_episodes_path).expanduser().resolve())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "applicable": True,
        "task": str(task_name),
        "evaluation_role": "training_set_diagnostic_only",
        "is_held_out": False,
        "hard_gate": False,
        "threshold": threshold_value,
        "decision_rule": (
            "pred_fail = (failure_score >= threshold); failure_score = -logit"
        ),
        "device": str(device),
        "seed": int(seed),
        "data_source": {
            "finetuned_checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
            "model_checkpoint": str(Path(model_checkpoint).expanduser().resolve()),
            "offline_episodes": resolved_offline_path,
            "checkpoint_offline_episodes": expected_offline_path,
            "path_matches_checkpoint": (
                expected_offline_path is not None
                and Path(str(expected_offline_path)).expanduser().resolve()
                == Path(resolved_offline_path)
            ),
        },
        "gt_window_config": {
            **gt_config,
            "frameskip": int(encoder.inner_encoder.frameskip),
            "source": "finetuned_checkpoint",
        },
        "provenance_validation": {
            "selected_frame_keys_match": True,
            "window_counts_match": True,
            "frame_counts_match": True,
        },
        "gt_window_stats": {
            key: window_stats[key]
            for key in (
                "intervention_events",
                "gt_negative_windows",
                "gt_negative_frames",
                "pre_frames",
                "post_frames",
                "pre_end_windows",
                "pre_end_frames",
                "deduplicated_frames",
            )
        },
        "overall": overall,
        "by_kind": by_kind,
        "by_position": by_position,
        "per_window": per_window,
    }


def build_not_applicable_report(
    *, checkpoint_path: str | Path, task_name: str
) -> dict[str, Any]:
    """Represent the parent-checkpoint branch without inventing training metrics."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "applicable": False,
        "task": str(task_name),
        "evaluation_role": "training_set_diagnostic_only",
        "is_held_out": False,
        "hard_gate": False,
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "reason": "parent_checkpoint_has_no_offline_gt_negative_training_pool",
    }


def _write_report(report: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, allow_nan=False)
    return path


def main() -> None:
    args = _parse_args()
    device = require_cuda_device(str(args.device))
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    checkpoint_path = str(Path(args.load_ckpt).expanduser().resolve())
    detector, payload = load_warmstart_detector(
        checkpoint_path,
        device=device,
        expected_task=str(args.task),
    )
    report = build_gt_fail_training_report(
        detector=detector,
        checkpoint_payload=payload,
        checkpoint_path=checkpoint_path,
        model_checkpoint=args.model_ckpt,
        offline_episodes_path=args.offline_episodes,
        task_name=str(args.task),
        device=device,
        encode_batch_size=int(args.encode_batch_size),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        proprio_indices=None,
        seed=int(args.seed),
    )
    report_path = _write_report(
        report, Path(args.out_dir) / str(args.report_name)
    )
    overall = report["overall"]
    print(
        "[gt_fail_eval] "
        f"task={args.task} detection_rate={overall['detection_rate']:.6f} "
        f"n_fail={overall['n_fail']}/{overall['n_frames']} "
        f"n_missed={overall['n_missed']} "
        f"margin_mean={overall['margin_mean']:+.6f}",
        flush=True,
    )
    print(f"[gt_fail_eval] wrote report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_gt_fail_training_report",
    "build_not_applicable_report",
    "score_summary_cuda",
    "validate_rebuilt_gt_provenance",
]
