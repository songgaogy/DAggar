"""CUDA-only load-only benchmark for parent or offline-finetuned nnPU heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from benchmark.core import DiscriminatorOutput, EvalConfig
from robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce import (
    _bootstrap_from_ckpt,
    _parse_camera_to_view,
)
from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark
from robosuite.pipeline.offline.discriminator.gt_fail_evaluation import (
    build_gt_fail_training_report,
    build_not_applicable_report,
)
from robosuite.pipeline.offline.visualization import (
    FinetunedPUBCEBenchmarkDiscriminator,
    FinetunedVisualizationContract,
    load_offline_success_trajectories,
    load_finetuned_visualization_contract,
    validate_runtime_normalizer,
)


@dataclass(frozen=True)
class CheckpointContract:
    """Validated checkpoint metadata and effective encoder selectors."""

    payload: dict[str, Any]
    checkpoint_kind: str
    camera_to_view: dict[str, str] | None
    proprio_indices: list[int] | None


class _RecordingDiscriminator:
    """Record benchmark outputs without scoring trajectories twice."""

    def __init__(self, discriminator: Any) -> None:
        self._discriminator = discriminator
        self.name = str(discriminator.name)
        self.outputs: dict[int, DiscriminatorOutput] = {}

    def score_trajectory(self, trajectory: Any) -> DiscriminatorOutput:
        output = self._discriminator.score_trajectory(trajectory)
        self.outputs[id(trajectory)] = output
        return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a parent or offline-finetuned nnPU checkpoint without fitting."
    )
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--load-ckpt", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--offline-episodes", required=True)
    parser.add_argument("--offline-fps", type=int, default=20)
    parser.add_argument("--offline-video-size", type=int, default=256)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fail-split", default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", default="success_rollout-val")
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--camera-to-view", default=None)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument(
        "--delta",
        type=float,
        default=5.0,
        help=(
            "Fallback delta for checkpoints without calibration metadata. "
            "Stored checkpoint delta and threshold take precedence."
        ),
    )
    parser.add_argument(
        "--feature-source", default="transformer", choices=["encoder", "transformer"]
    )
    parser.add_argument("--transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _require_cuda(device_value: str) -> torch.device:
    device = torch.device(str(device_value))
    if device.type != "cuda":
        raise ValueError(f"nnPU checkpoint evaluation requires CUDA, got {device_value!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "nnPU checkpoint evaluation requires CUDA, but "
            "torch.cuda.is_available() is False."
        )
    index = 0 if device.index is None else int(device.index)
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {device} is unavailable; visible device count is "
            f"{torch.cuda.device_count()}."
        )
    torch.cuda.set_device(index)
    return device


def _sha256_file(path_value: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path_value).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_camera_map(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    return {str(key): str(item) for key, item in dict(value).items()}


def _optional_indices(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _select_contract_value(
    *, checkpoint_value: Any, requested_value: Any, field_name: str
) -> Any:
    if (
        checkpoint_value is not None
        and requested_value is not None
        and checkpoint_value != requested_value
    ):
        raise ValueError(
            f"Runtime {field_name} differs from the loaded checkpoint: "
            f"runtime={requested_value!r}, checkpoint={checkpoint_value!r}."
        )
    return requested_value if requested_value is not None else checkpoint_value


def load_checkpoint_contract(
    checkpoint: str | Path,
    *,
    model_checkpoint: str | Path,
    feature_source: str,
    transformer_layer: int,
    camera_to_view: Mapping[str, str] | None,
    proprio_indices: Sequence[int] | None,
) -> CheckpointContract:
    """Validate parent/finetuned checkpoint provenance and latent selectors."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"nnPU checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "pu_bce_detector" not in payload:
        raise KeyError(f"{checkpoint_path} is not a pu_bce_head checkpoint.")

    if bool(payload.get("finetuned_offline", False)):
        contract = load_finetuned_visualization_contract(
            checkpoint_path,
            model_checkpoint=model_checkpoint,
            feature_source=feature_source,
            transformer_layer=transformer_layer,
            camera_to_view=camera_to_view,
            proprio_indices=proprio_indices,
        )
        return CheckpointContract(
            payload=contract.payload,
            checkpoint_kind="finetuned",
            camera_to_view=contract.camera_to_view,
            proprio_indices=contract.proprio_indices,
        )

    if str(payload.get("feature_source")) != str(feature_source):
        raise ValueError(
            "Runtime feature_source differs from the parent checkpoint: "
            f"runtime={feature_source!r}, checkpoint={payload.get('feature_source')!r}."
        )
    if int(payload.get("transformer_layer", -1)) != int(transformer_layer):
        raise ValueError(
            "Runtime transformer layer differs from the parent checkpoint: "
            f"runtime={transformer_layer}, "
            f"checkpoint={payload.get('transformer_layer')!r}."
        )

    requested_camera = _optional_camera_map(camera_to_view)
    checkpoint_camera = _optional_camera_map(payload.get("camera_to_view"))
    effective_camera = _select_contract_value(
        checkpoint_value=checkpoint_camera,
        requested_value=requested_camera,
        field_name="camera_to_view",
    )
    requested_proprio = _optional_indices(proprio_indices)
    checkpoint_proprio = _optional_indices(payload.get("proprio_indices"))
    effective_proprio = _select_contract_value(
        checkpoint_value=checkpoint_proprio,
        requested_value=requested_proprio,
        field_name="proprio_indices",
    )

    model_path = Path(model_checkpoint).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Dynamics checkpoint not found: {model_path}")
    expected_sha = payload.get("model_ckpt_sha256")
    if expected_sha is not None and str(expected_sha) != _sha256_file(model_path):
        raise ValueError("Dynamics checkpoint SHA-256 differs from the parent checkpoint.")
    recorded_model = payload.get("model_ckpt")
    if expected_sha is None and recorded_model is not None:
        recorded_path = Path(str(recorded_model)).expanduser().resolve()
        if recorded_path != model_path:
            raise ValueError(
                "Dynamics checkpoint path differs from the legacy parent checkpoint: "
                f"runtime={model_path}, checkpoint={recorded_path}."
            )

    return CheckpointContract(
        payload=payload,
        checkpoint_kind="parent",
        camera_to_view=effective_camera,
        proprio_indices=effective_proprio,
    )


def build_success_false_alarm_report(
    trajectories: Sequence[Any],
    outputs: Mapping[int, DiscriminatorOutput],
    *,
    task: str,
    checkpoint: str,
    checkpoint_kind: str,
) -> dict[str, Any]:
    """Aggregate false alarms over valid pre-completion success frames only."""
    per_trajectory: list[dict[str, Any]] = []
    total_frames = 0
    false_positive_frames = 0
    trajectories_with_alarm = 0

    for trajectory in trajectories:
        if bool(trajectory.is_failure):
            continue
        output = outputs.get(id(trajectory))
        if output is None:
            raise KeyError(f"Missing recorded output for {trajectory.video_id!r}.")
        if output.predictions is None:
            raise ValueError(
                f"Checkpoint output for {trajectory.video_id!r} has no predictions."
            )
        predictions = np.asarray(output.predictions, dtype=np.int64).reshape(-1)
        num_frames = int(trajectory.num_frames)
        if int(predictions.shape[0]) != num_frames:
            raise ValueError(
                f"Predictions for {trajectory.video_id!r} have length "
                f"{predictions.shape[0]}, expected {num_frames}."
            )
        prefix_fn = getattr(trajectory, "prefix_frames_before_done", None)
        valid_frames = num_frames if prefix_fn is None else int(prefix_fn())
        if not 0 <= valid_frames <= num_frames:
            raise ValueError(
                f"Invalid success prefix for {trajectory.video_id!r}: "
                f"{valid_frames} not in [0, {num_frames}]."
            )
        valid_predictions = predictions[:valid_frames]
        num_false_positive = int(np.count_nonzero(valid_predictions == 1))
        positive = np.flatnonzero(valid_predictions == 1)
        any_alarm = bool(positive.size)
        total_frames += valid_frames
        false_positive_frames += num_false_positive
        trajectories_with_alarm += int(any_alarm)
        per_trajectory.append(
            {
                "task_name": str(trajectory.task_name),
                "video_id": str(trajectory.video_id),
                "num_frames": num_frames,
                "valid_frames": valid_frames,
                "false_positive_frames": num_false_positive,
                "frame_fpr": (
                    float(num_false_positive / valid_frames)
                    if valid_frames > 0
                    else None
                ),
                "any_alarm": any_alarm,
                "first_alarm_frame": int(positive[0]) if any_alarm else None,
            }
        )

    num_trajectories = len(per_trajectory)
    if num_trajectories == 0:
        raise RuntimeError("No success trajectories were evaluated.")
    if total_frames == 0:
        raise RuntimeError("Success trajectories contain zero pre-completion frames.")
    return {
        "schema_version": 1,
        "task": str(task),
        "checkpoint": str(checkpoint),
        "checkpoint_kind": str(checkpoint_kind),
        "valid_frame_policy": "frames_before_first_is_success_true",
        "num_trajectories": num_trajectories,
        "valid_frames": total_frames,
        "false_positive_frames": false_positive_frames,
        "frame_fpr": float(false_positive_frames / total_frames),
        "trajectories_with_alarm": trajectories_with_alarm,
        "trajectory_alarm_rate": float(trajectories_with_alarm / num_trajectories),
        "per_trajectory": per_trajectory,
    }


def build_offline_success_false_alarm_report(
    trajectories: Sequence[Any],
    outputs: Mapping[int, DiscriminatorOutput],
    *,
    task: str,
    checkpoint: str,
    checkpoint_kind: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate alarms over all policy-only offline success prefixes."""
    if not trajectories:
        raise RuntimeError("No offline success trajectories were provided.")

    per_episode: list[dict[str, Any]] = []
    total_frames = 0
    false_alarm_frames = 0
    episodes_with_alarm = 0
    first_alarm_frames: list[int] = []
    for trajectory in trajectories:
        if str(getattr(trajectory, "terminal_reason", "")) != "success":
            raise ValueError(
                f"Offline episode {trajectory.video_id!r} is not terminal_reason='success'."
            )
        intervention_mask = np.asarray(
            getattr(trajectory, "intervention_mask", []), dtype=np.bool_
        ).reshape(-1)
        if int(intervention_mask.shape[0]) != int(trajectory.num_frames):
            raise ValueError(
                f"Intervention mask for {trajectory.video_id!r} has length "
                f"{intervention_mask.shape[0]}, expected {trajectory.num_frames}."
            )
        if bool(intervention_mask.any()):
            raise ValueError(
                f"Offline success episode {trajectory.video_id!r} contains intervention frames."
            )

        output = outputs.get(id(trajectory))
        if output is None:
            raise KeyError(f"Missing recorded output for {trajectory.video_id!r}.")
        if output.predictions is None:
            raise ValueError(
                f"Checkpoint output for {trajectory.video_id!r} has no predictions."
            )
        predictions = np.asarray(output.predictions, dtype=np.int64).reshape(-1)
        num_frames = int(trajectory.num_frames)
        if int(predictions.shape[0]) != num_frames:
            raise ValueError(
                f"Predictions for {trajectory.video_id!r} have length "
                f"{predictions.shape[0]}, expected {num_frames}."
            )
        valid_frames = int(trajectory.prefix_frames_before_done())
        if not 0 <= valid_frames <= num_frames:
            raise ValueError(
                f"Invalid success prefix for {trajectory.video_id!r}: "
                f"{valid_frames} not in [0, {num_frames}]."
            )
        valid_predictions = predictions[:valid_frames]
        alarm_indices = np.flatnonzero(valid_predictions == 1)
        num_false_alarms = int(alarm_indices.size)
        any_alarm = bool(alarm_indices.size)
        first_alarm = int(alarm_indices[0]) if any_alarm else None
        if first_alarm is not None:
            first_alarm_frames.append(first_alarm)

        total_frames += valid_frames
        false_alarm_frames += num_false_alarms
        episodes_with_alarm += int(any_alarm)
        per_episode.append(
            {
                "episode_index": int(trajectory.episode_index),
                "video_id": str(trajectory.video_id),
                "num_frames": num_frames,
                "first_success_frame": valid_frames,
                "valid_frames": valid_frames,
                "false_alarm_frames": num_false_alarms,
                "frame_false_alarm_rate": (
                    float(num_false_alarms / valid_frames)
                    if valid_frames > 0
                    else None
                ),
                "any_alarm": any_alarm,
                "first_alarm_frame": first_alarm,
            }
        )

    if total_frames == 0:
        raise RuntimeError("Offline success trajectories contain zero pre-success frames.")
    num_episodes = len(per_episode)
    source_path = Path(str(source_metadata["source_path"])).expanduser().resolve()
    return {
        "schema_version": 1,
        "task": str(task),
        "valid_frame_policy": "frames_before_first_success_true",
        "alarm_definition": "checkpoint_prediction_equals_one",
        "aggregate": {
            "source_episode_count": int(source_metadata["source_episode_count"]),
            "eligible_episode_count": num_episodes,
            "valid_frames": total_frames,
            "false_alarm_frames": false_alarm_frames,
            "frame_false_alarm_rate": float(false_alarm_frames / total_frames),
            "episodes_with_alarm": episodes_with_alarm,
            "trajectory_alarm_rate": float(episodes_with_alarm / num_episodes),
        },
        "first_alarm": {
            "frame_index_basis": "zero_based_episode_frame",
            "count": len(first_alarm_frames),
            "frames": first_alarm_frames,
            "min_frame": min(first_alarm_frames) if first_alarm_frames else None,
            "max_frame": max(first_alarm_frames) if first_alarm_frames else None,
            "mean_frame": (
                float(np.mean(first_alarm_frames)) if first_alarm_frames else None
            ),
            "median_frame": (
                float(np.median(first_alarm_frames)) if first_alarm_frames else None
            ),
        },
        "per_episode": per_episode,
        "provenance": {
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_kind": str(checkpoint_kind),
            "offline_episodes": str(source_path),
            "offline_episodes_sha256": _sha256_file(source_path),
            "offline_schema_version": int(source_metadata["schema_version"]),
            "offline_task_name": source_metadata.get("payload_task_name"),
            "camera_names": list(source_metadata["camera_names"]),
            "selection": "terminal_reason=success AND no intervention frames",
            "eligible_episode_indices": list(
                source_metadata["eligible_episode_indices"]
            ),
        },
    }


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    args = _parse_args()
    device = _require_cuda(str(args.device))
    torch.cuda.manual_seed_all(int(args.seed))

    load_ckpt = str(Path(args.load_ckpt).expanduser().resolve())
    model_ckpt = str(Path(args.model_ckpt).expanduser().resolve())
    requested_camera = _parse_camera_to_view(args.camera_to_view)
    contract = load_checkpoint_contract(
        load_ckpt,
        model_checkpoint=model_ckpt,
        feature_source=str(args.feature_source),
        transformer_layer=int(args.transformer_layer),
        camera_to_view=requested_camera,
        proprio_indices=args.proprio_indices,
    )

    benchmark = FailureBenchmark(
        data_root=str(args.data_root),
        tasks=[str(args.task)],
        fail_split=str(args.fail_split),
        success_split=str(args.success_split),
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    trajectories = benchmark.trajectories()
    if not trajectories:
        raise RuntimeError(f"No trajectories discovered for task {args.task!r}.")

    with torch.device(device):
        discriminator = FinetunedPUBCEBenchmarkDiscriminator(
            model_ckpt=model_ckpt,
            unlabeled_fail_trajectories=[],
            save_ckpt_dir=None,
            device=str(device),
            encode_batch_size=int(args.encode_batch_size),
            proprio_indices=contract.proprio_indices,
            camera_to_view=contract.camera_to_view,
            visual_weight=float(args.visual_weight),
            proprio_weight=float(args.proprio_weight),
            action_weight=float(args.action_weight),
            delta=float(args.delta),
            feature_source=str(args.feature_source),
            transformer_layer=int(args.transformer_layer),
            calib_fraction=float(args.calib_fraction),
            seed=int(args.seed),
            verbose_fit=False,
        )

    try:
        if contract.payload.get("normalizer_ckpt_sha256") is not None:
            validate_runtime_normalizer(
                discriminator,
                FinetunedVisualizationContract(
                    payload=contract.payload,
                    camera_to_view=contract.camera_to_view,
                    proprio_indices=contract.proprio_indices,
                ),
            )
        _bootstrap_from_ckpt(discriminator, load_ckpt)
        task_name = str(args.task)
        if task_name not in discriminator._detectors_per_task:
            raise RuntimeError(
                f"Task {task_name!r} is absent from the checkpoint; available tasks: "
                f"{sorted(discriminator._detectors_per_task)}."
            )

        recording = _RecordingDiscriminator(discriminator)
        result = benchmark.evaluate(
            recording,
            EvalConfig(step_binarize_strategy="provided"),
        )
        result.config.update(
            {
                "checkpoint": load_ckpt,
                "checkpoint_kind": contract.checkpoint_kind,
                "model_checkpoint": model_ckpt,
            }
        )
        success_report = build_success_false_alarm_report(
            trajectories,
            recording.outputs,
            task=task_name,
            checkpoint=load_ckpt,
            checkpoint_kind=contract.checkpoint_kind,
        )
        offline_success_trajectories, offline_source = (
            load_offline_success_trajectories(
                args.offline_episodes,
                task=task_name,
                fps=int(args.offline_fps),
                video_size=int(args.offline_video_size),
            )
        )
        offline_outputs = {
            id(trajectory): discriminator.score_trajectory(trajectory)
            for trajectory in offline_success_trajectories
        }
        offline_success_report = build_offline_success_false_alarm_report(
            offline_success_trajectories,
            offline_outputs,
            task=task_name,
            checkpoint=load_ckpt,
            checkpoint_kind=contract.checkpoint_kind,
            source_metadata=offline_source,
        )

        out_dir = Path(args.out_dir).expanduser().resolve()
        if contract.checkpoint_kind == "finetuned":
            gt_fail_report = build_gt_fail_training_report(
                detector=discriminator._detectors_per_task[task_name],  # noqa: SLF001
                checkpoint_payload=contract.payload,
                checkpoint_path=load_ckpt,
                model_checkpoint=model_ckpt,
                offline_episodes_path=args.offline_episodes,
                task_name=task_name,
                device=device,
                encode_batch_size=int(args.encode_batch_size),
                camera_to_view=contract.camera_to_view,
                proprio_indices=contract.proprio_indices,
                seed=int(args.seed),
            )
        else:
            gt_fail_report = build_not_applicable_report(
                checkpoint_path=load_ckpt,
                task_name=task_name,
            )
        result.save_json(str(out_dir / "benchmark.json"))
        _save_json(success_report, out_dir / "success_false_alarm.json")
        _save_json(
            offline_success_report,
            out_dir / "offline_success_false_alarm.json",
        )
        _save_json(gt_fail_report, out_dir / "gt_fail_detection.json")
        gt_fail_log = (
            "gt_fail_train_detection_rate="
            f"{gt_fail_report['overall']['detection_rate']:.6f}"
            if gt_fail_report["applicable"]
            else "gt_fail_train_detection_rate=not_applicable"
        )
        print(result.summary(), flush=True)
        print(
            "[nnpu][load-only] "
            f"success_frame_fpr={success_report['frame_fpr']:.6f} "
            f"success_trajectory_alarm_rate="
            f"{success_report['trajectory_alarm_rate']:.6f} "
            f"offline_success_frame_far="
            f"{offline_success_report['aggregate']['frame_false_alarm_rate']:.6f} "
            f"offline_success_trajectory_alarm_rate="
            f"{offline_success_report['aggregate']['trajectory_alarm_rate']:.6f} "
            f"gt_fail_train_applicable={gt_fail_report['applicable']} "
            f"{gt_fail_log} "
            f"out_dir={out_dir}",
            flush=True,
        )
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
