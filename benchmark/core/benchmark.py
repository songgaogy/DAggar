"""FailureBenchmark: orchestrate discriminator scoring + metric computation.

This core class is data-source agnostic: it takes a pre-discovered list of
BenchmarkTrajectory subclass instances at construction time. Source-specific
skins (benchmark.robosuite, benchmark.real_world) provide convenience
constructors that wrap their respective discoverers.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .discriminator import Discriminator, DiscriminatorOutput
from .metrics import (
    aggregate_trajectory_score,
    event_level_metrics,
    frame_binary_metrics,
    frame_level_auroc_auprc,
    trajectory_level_metrics,
)
from .trajectory import BenchmarkTrajectory


try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(x, **_kw):
        return x


# ---------------------------------------------------------------------- #
# Result container                                                       #
# ---------------------------------------------------------------------- #


@dataclass
class BenchmarkResult:
    discriminator_name: str
    trajectory_level: dict = field(default_factory=dict)
    trajectory_level_per_task: dict = field(default_factory=dict)
    step_level: dict = field(default_factory=dict)
    step_level_per_task: dict = field(default_factory=dict)
    per_trajectory: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Benchmark: {self.discriminator_name} ===",
            f"num_trajectories: {self.trajectory_level.get('num_trajectories', 0)} "
            f"(success={self.trajectory_level.get('num_success', 0)}, "
            f"failure={self.trajectory_level.get('num_failure', 0)})",
            "",
            "[Trajectory level]",
        ]
        for k, v in self.trajectory_level.items():
            if k in {"num_trajectories", "num_failure", "num_success"}:
                continue
            lines.append(f"  {k}: {_fmt(v)}")
        if self.trajectory_level_per_task:
            lines.append("  per task:")
            for task, metrics in self.trajectory_level_per_task.items():
                extra = ", ".join(
                    f"{k}={_fmt(metrics[k])}"
                    for k in ("auroc", "auprc", "best_f1")
                    if k in metrics
                )
                lines.append(f"    {task}: {extra}")

        lines.append("")
        lines.append("[Step level]")
        for k, v in self.step_level.items():
            lines.append(f"  {k}: {_fmt(v)}")
        if self.step_level_per_task:
            lines.append("  per task:")
            for task, metrics in self.step_level_per_task.items():
                extra = ", ".join(
                    f"{k}={_fmt(metrics[k])}"
                    for k in ("frame_auroc", "event_recall", "detection_delay_median")
                    if k in metrics
                )
                lines.append(f"    {task}: {extra}")

        lines.append("")
        lines.append(f"runtime_seconds: {self.runtime.get('total_seconds', 0):.1f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def _coerce(obj):
            if isinstance(obj, dict):
                return {str(k): _coerce(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_coerce(x) for x in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        return _coerce(dataclasses.asdict(self))

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=2, default=_json_default)


def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


# ---------------------------------------------------------------------- #
# Benchmark                                                              #
# ---------------------------------------------------------------------- #


@dataclass
class EvalConfig:
    """Tunable knobs; passed to FailureBenchmark.evaluate()."""

    traj_score_aggregator: str = "max"       # "max" | "mean" | "topk_mean"
    traj_score_topk: int = 20
    event_iou_threshold: float = 0.1
    step_binarize_strategy: str = "success_calibrated"  # or "provided" | "fixed"
    step_success_percentile: float = 95.0    # for "success_calibrated"
    step_fixed_threshold: Optional[float] = None  # for "fixed"


class FailureBenchmark:
    """Source-agnostic benchmark orchestrator.

    Accepts a pre-discovered list of BenchmarkTrajectory subclass instances.
    Source-specific subclasses (benchmark.robosuite.FailureBenchmark, etc.)
    wrap their loaders and pass the discovered trajectories to super().__init__.

    Usage:
        bench = FailureBenchmark(trajectories=trajs)
        result = bench.evaluate(my_discriminator)
        print(result.summary())
        result.save_json("/tmp/bench.json")
    """

    def __init__(
        self,
        trajectories: list[BenchmarkTrajectory],
        source_metadata: Optional[dict] = None,
    ) -> None:
        self._trajectories = list(trajectories) if trajectories is not None else []
        self.source_metadata: dict = dict(source_metadata) if source_metadata else {}

    def trajectories(self) -> list[BenchmarkTrajectory]:
        return list(self._trajectories)

    # ------------------------------------------------------------------ #
    # Main entry                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        discriminator: Discriminator,
        config: Optional[EvalConfig] = None,
        *,
        progress: bool = True,
    ) -> BenchmarkResult:
        cfg = config or EvalConfig()
        trajs = self.trajectories()
        if not trajs:
            raise RuntimeError(
                "No trajectories supplied to FailureBenchmark; "
                f"source_metadata={self.source_metadata!r}"
            )

        t_start = time.perf_counter()
        # 1) Run the discriminator on every trajectory.
        outputs: list[DiscriminatorOutput] = []
        iterator = _tqdm(trajs, desc=f"scoring[{discriminator.name}]") if progress else trajs
        for traj in iterator:
            out = discriminator.score_trajectory(traj)
            out.validate(expected_T=int(traj.num_frames))
            outputs.append(out)

        # 2) Trajectory-level aggregation.
        traj_scores = np.asarray(
            [
                aggregate_trajectory_score(
                    o.step_scores, mode=cfg.traj_score_aggregator, topk=cfg.traj_score_topk
                )
                for o in outputs
            ],
            dtype=np.float64,
        )
        traj_labels = np.asarray([int(t.is_failure) for t in trajs], dtype=np.int64)
        traj_tasks = [t.task_name for t in trajs]

        traj_metrics = trajectory_level_metrics(traj_scores, traj_labels)
        traj_metrics_per_task = {}
        for task_name in sorted(set(traj_tasks)):
            idx = np.asarray([i for i, t in enumerate(traj_tasks) if t == task_name], dtype=np.int64)
            if idx.size == 0:
                continue
            traj_metrics_per_task[task_name] = trajectory_level_metrics(
                traj_scores[idx], traj_labels[idx]
            )

        # 3) Step-level (failure trajectories only).
        fail_idx = [i for i, t in enumerate(trajs) if t.is_failure]
        success_idx = [i for i, t in enumerate(trajs) if not t.is_failure]

        success_step_scores = [
            np.asarray(outputs[i].step_scores, dtype=np.float64).reshape(-1)
            for i in success_idx
        ]
        fail_step_scores = [
            np.asarray(outputs[i].step_scores, dtype=np.float64).reshape(-1)
            for i in fail_idx
        ]
        fail_masks = [
            np.asarray(trajs[i].load_failure_mask(), dtype=np.int64).reshape(-1)
            for i in fail_idx
        ]

        # 3a) Threshold-free frame scoring.
        frame_scores_metrics = frame_level_auroc_auprc(fail_step_scores, fail_masks)

        # 3b) Binary predictions per traj.
        fail_preds = _materialize_predictions(
            fail_trajs=[trajs[i] for i in fail_idx],
            fail_outputs=[outputs[i] for i in fail_idx],
            success_step_scores=success_step_scores,
            cfg=cfg,
        )
        frame_bin = frame_binary_metrics(fail_preds, fail_masks)

        # 3c) Event-level.
        event_metrics = event_level_metrics(
            per_traj_predictions=fail_preds,
            per_traj_gt_segments=[trajs[i].failure_segments for i in fail_idx],
            per_traj_num_frames=[trajs[i].num_frames for i in fail_idx],
            iou_threshold=cfg.event_iou_threshold,
        )

        step_metrics = {**frame_scores_metrics, **frame_bin, **event_metrics}

        # 3d) Per-task step metrics.
        step_per_task: dict[str, dict] = {}
        fail_tasks = [trajs[i].task_name for i in fail_idx]
        for task_name in sorted(set(fail_tasks)):
            task_mask = [j for j, name in enumerate(fail_tasks) if name == task_name]
            if not task_mask:
                continue
            task_scores = [fail_step_scores[j] for j in task_mask]
            task_masks = [fail_masks[j] for j in task_mask]
            task_preds = [fail_preds[j] for j in task_mask]
            task_segs = [trajs[fail_idx[j]].failure_segments for j in task_mask]
            task_T = [trajs[fail_idx[j]].num_frames for j in task_mask]
            sub = {
                **frame_level_auroc_auprc(task_scores, task_masks),
                **frame_binary_metrics(task_preds, task_masks),
                **event_level_metrics(
                    per_traj_predictions=task_preds,
                    per_traj_gt_segments=task_segs,
                    per_traj_num_frames=task_T,
                    iou_threshold=cfg.event_iou_threshold,
                ),
            }
            step_per_task[task_name] = sub

        # 4) Per-trajectory details for downstream analysis.
        per_traj = []
        for i, (traj, out) in enumerate(zip(trajs, outputs)):
            rec = {
                "task_name": traj.task_name,
                "video_id": traj.video_id,
                "is_failure": bool(traj.is_failure),
                "num_frames": int(traj.num_frames),
                "trajectory_score": float(traj_scores[i]),
                "first_gt_failure_frame": traj.first_gt_failure_frame(),
                "first_pred_failure_frame": (
                    int(out.first_failure_frame)
                    if out.first_failure_frame is not None
                    else None
                ),
                "failure_segments": list(traj.failure_segments),
            }
            per_traj.append(rec)

        total_seconds = float(time.perf_counter() - t_start)
        return BenchmarkResult(
            discriminator_name=str(getattr(discriminator, "name", "unknown")),
            trajectory_level=traj_metrics,
            trajectory_level_per_task=traj_metrics_per_task,
            step_level=step_metrics,
            step_level_per_task=step_per_task,
            per_trajectory=per_traj,
            config={
                **dataclasses.asdict(cfg),
                **self.source_metadata,
            },
            runtime={
                "total_seconds": total_seconds,
                "num_trajectories": int(len(trajs)),
            },
        )


# ---------------------------------------------------------------------- #
# Thresholding helpers                                                   #
# ---------------------------------------------------------------------- #


def _materialize_predictions(
    fail_trajs: list[BenchmarkTrajectory],
    fail_outputs: list[DiscriminatorOutput],
    success_step_scores: list[np.ndarray],
    cfg: EvalConfig,
) -> list[np.ndarray]:
    """Decide per-frame binary predictions for failure trajectories."""
    strategy = cfg.step_binarize_strategy
    if strategy == "provided":
        if all(o.predictions is not None for o in fail_outputs):
            return [
                np.asarray(o.predictions, dtype=np.int64).reshape(-1)
                for o in fail_outputs
            ]
        print(
            "[benchmark] step_binarize_strategy=provided but some outputs are "
            "missing predictions; falling back to success_calibrated."
        )
        strategy = "success_calibrated"

    if strategy == "fixed":
        thr = cfg.step_fixed_threshold
        if thr is None:
            raise ValueError(
                "step_binarize_strategy='fixed' requires cfg.step_fixed_threshold"
            )
        return [(np.asarray(o.step_scores) >= float(thr)).astype(np.int64)
                for o in fail_outputs]

    if strategy == "success_calibrated":
        if len(success_step_scores) == 0:
            preds = []
            for o in fail_outputs:
                s = np.asarray(o.step_scores, dtype=np.float64)
                thr = float(np.percentile(s, cfg.step_success_percentile))
                preds.append((s >= thr).astype(np.int64))
            return preds
        pooled = np.concatenate(success_step_scores) if success_step_scores else np.zeros(0)
        thr = float(np.percentile(pooled, float(cfg.step_success_percentile)))
        return [
            (np.asarray(o.step_scores, dtype=np.float64) >= thr).astype(np.int64)
            for o in fail_outputs
        ]

    raise ValueError(f"Unknown step_binarize_strategy={strategy!r}")
