"""Robosuite discriminator benchmark for the data/<task>/<split> layout."""

from __future__ import annotations

from typing import Optional

from benchmark.core import (
    BenchmarkResult,
    Discriminator,
    DiscriminatorOutput,
    EvalConfig,
)
from benchmark.core.benchmark import FailureBenchmark as _CoreFailureBenchmark

from .loader import (
    DEFAULT_BANK_SPLIT,
    DEFAULT_FAIL_SPLIT,
    DEFAULT_SUCCESS_SPLIT,
    DEFAULT_SUCCESS_TRAIN_SPLIT,
    canonical_task_name,
    discover_failure_bank,
    discover_success_rollouts,
    discover_success_training_dirs,
    discover_trajectories,
)
from .trajectory import RobosuiteBenchmarkTrajectory


BenchmarkTrajectory = RobosuiteBenchmarkTrajectory


class FailureBenchmark(_CoreFailureBenchmark):
    """Robosuite benchmark over val-labeled failures and val success rollouts."""

    def __init__(
        self,
        data_root: str = "data",
        tasks: Optional[list[str]] = None,
        fail_split: str = DEFAULT_FAIL_SPLIT,
        success_split: str = DEFAULT_SUCCESS_SPLIT,
        max_fail_per_task: Optional[int] = None,
        max_success_per_task: Optional[int] = None,
    ) -> None:
        self.data_root = str(data_root)
        self.tasks = list(tasks) if tasks is not None else None
        self.fail_split = str(fail_split)
        self.success_split = str(success_split)
        self.max_fail_per_task = max_fail_per_task
        self.max_success_per_task = max_success_per_task

        trajs = discover_trajectories(
            data_root=self.data_root,
            tasks=self.tasks,
            fail_split=self.fail_split,
            success_split=self.success_split,
            max_fail_per_task=self.max_fail_per_task,
            max_success_per_task=self.max_success_per_task,
        )
        super().__init__(
            trajectories=trajs,
            source_metadata={
                "data_root": self.data_root,
                "tasks": self.tasks,
                "fail_split": self.fail_split,
                "success_split": self.success_split,
                "max_fail_per_task": self.max_fail_per_task,
                "max_success_per_task": self.max_success_per_task,
            },
        )


__all__ = [
    "BenchmarkTrajectory",
    "RobosuiteBenchmarkTrajectory",
    "FailureBenchmark",
    "BenchmarkResult",
    "EvalConfig",
    "Discriminator",
    "DiscriminatorOutput",
    "DEFAULT_BANK_SPLIT",
    "DEFAULT_FAIL_SPLIT",
    "DEFAULT_SUCCESS_SPLIT",
    "DEFAULT_SUCCESS_TRAIN_SPLIT",
    "canonical_task_name",
    "discover_failure_bank",
    "discover_success_rollouts",
    "discover_success_training_dirs",
    "discover_trajectories",
]
