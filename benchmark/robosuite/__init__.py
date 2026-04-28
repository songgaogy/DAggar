"""Robosuite-specific skin of the benchmark.

Re-exports the core API + adds a robosuite-flavored FailureBenchmark whose
constructor matches the original `data/utils/benchmark` signature
(fail_labeled_root / success_root / tasks / caps).
"""

from benchmark.core import (
    BenchmarkResult,
    Discriminator,
    DiscriminatorOutput,
    EvalConfig,
)
from benchmark.core.benchmark import FailureBenchmark as _CoreFailureBenchmark

from .trajectory import RobosuiteBenchmarkTrajectory
from .loader import discover_trajectories


# Keep the public name `BenchmarkTrajectory` pointing at the robosuite subclass
# so callers that imported it from data.utils.benchmark see a drop-in type.
BenchmarkTrajectory = RobosuiteBenchmarkTrajectory


class FailureBenchmark(_CoreFailureBenchmark):
    """Robosuite benchmark: discovers trajectories from fail/success HDF5 layouts."""

    def __init__(
        self,
        fail_labeled_root: str,
        success_root: str,
        tasks=None,
        max_fail_per_task=None,
        max_success_per_task=None,
        *,
        success_cache_root: str | None = None,
        metadata_cache_root: str | None = None,
        cache_camera_names=None,
    ) -> None:
        self.fail_labeled_root = str(fail_labeled_root)
        self.success_root = str(success_root)
        self.success_cache_root = None if success_cache_root is None else str(success_cache_root)
        self.metadata_cache_root = None if metadata_cache_root is None else str(metadata_cache_root)
        self.tasks = list(tasks) if tasks is not None else None
        self.max_fail_per_task = max_fail_per_task
        self.max_success_per_task = max_success_per_task
        self.cache_camera_names = (
            None if cache_camera_names is None else tuple(str(x) for x in cache_camera_names)
        )

        trajs = discover_trajectories(
            fail_labeled_root=self.fail_labeled_root,
            success_root=self.success_root,
            tasks=self.tasks,
            max_fail_per_task=self.max_fail_per_task,
            max_success_per_task=self.max_success_per_task,
            success_cache_root=self.success_cache_root,
            metadata_cache_root=self.metadata_cache_root,
            cache_camera_names=self.cache_camera_names,
        )
        super().__init__(
            trajectories=trajs,
            source_metadata={
                "fail_labeled_root": self.fail_labeled_root,
                "success_root": self.success_root,
                "success_cache_root": self.success_cache_root,
                "metadata_cache_root": self.metadata_cache_root,
                "tasks": self.tasks,
                "max_fail_per_task": self.max_fail_per_task,
                "max_success_per_task": self.max_success_per_task,
                "cache_camera_names": self.cache_camera_names,
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
    "discover_trajectories",
]
