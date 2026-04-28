"""Real-world (agilex) skin of the benchmark.

The Agilex annotated-failure dataset uses a different HDF5 layout from
robosuite (per-episode groups under ``episodes/<split>/<episode_N>``,
images under ``observations/images/<cam>``, action shape ``(T, 14)``).
Default behaviour for the LPB-style pipeline:
    * ``load_actions``  -> ``action[:, 7:14]``  (right arm 7-D).
    * ``load_states``   -> ``observations/qpos[:, 7:14]`` (right arm 7-D).
    * default camera    -> ``cam_high``.
"""

from benchmark.core import (
    BenchmarkResult,
    Discriminator,
    DiscriminatorOutput,
    EvalConfig,
)
from benchmark.core.benchmark import FailureBenchmark as _CoreFailureBenchmark

from .trajectory import AgilexBenchmarkTrajectory
from .loader import discover_agilex_trajectories, discover_cached_agilex_trajectories


# Public alias for parity with benchmark.robosuite.
BenchmarkTrajectory = AgilexBenchmarkTrajectory


class FailureBenchmark(_CoreFailureBenchmark):
    """Real-world (agilex) benchmark.

    Args:
        fail_labeled_root: directory containing ``<task>/out.hdf5`` annotated
            files (e.g. ``data/agilex/failure_annotations/out_by_task``).
        success_root: directory containing ``<task>/success_rollout/episode_*.hdf5``
            (e.g. ``data/agilex``).
        tasks: optional task filter, e.g. ``["candy_in_plate"]``.
        proprio_field / proprio_slice: how to extract per-step state from
            the observation group (default: ``qpos[:, 7:14]``).
        action_slice: which slots of the (T, 14) action vector to expose
            via ``load_actions`` (default: right arm ``[:, 7:14]``).
    """

    def __init__(
        self,
        fail_labeled_root: str,
        success_root: str,
        tasks=None,
        max_fail_per_task=None,
        max_success_per_task=None,
        *,
        cache_root: str | None = None,
        proprio_field: str = "qpos",
        proprio_slice: slice = slice(7, 14),
        action_slice: slice = slice(7, 14),
    ) -> None:
        self.fail_labeled_root = str(fail_labeled_root)
        self.success_root = str(success_root)
        self.cache_root = None if cache_root is None else str(cache_root)
        self.tasks = list(tasks) if tasks is not None else None
        self.max_fail_per_task = max_fail_per_task
        self.max_success_per_task = max_success_per_task

        if self.cache_root:
            trajs = discover_cached_agilex_trajectories(
                cache_root=self.cache_root,
                tasks=self.tasks,
                max_fail_per_task=self.max_fail_per_task,
                max_success_per_task=self.max_success_per_task,
            )
        else:
            trajs = discover_agilex_trajectories(
                fail_labeled_root=self.fail_labeled_root,
                success_root=self.success_root,
                tasks=self.tasks,
                max_fail_per_task=self.max_fail_per_task,
                max_success_per_task=self.max_success_per_task,
                proprio_field=proprio_field,
                proprio_slice=proprio_slice,
                action_slice=action_slice,
            )
        super().__init__(
            trajectories=trajs,
            source_metadata={
                "fail_labeled_root": self.fail_labeled_root,
                "success_root": self.success_root,
                "cache_root": self.cache_root,
                "tasks": self.tasks,
                "max_fail_per_task": self.max_fail_per_task,
                "max_success_per_task": self.max_success_per_task,
                "proprio_field": proprio_field,
                "proprio_slice": [proprio_slice.start, proprio_slice.stop],
                "action_slice": [action_slice.start, action_slice.stop],
            },
        )


__all__ = [
    "BenchmarkTrajectory",
    "AgilexBenchmarkTrajectory",
    "FailureBenchmark",
    "BenchmarkResult",
    "EvalConfig",
    "Discriminator",
    "DiscriminatorOutput",
    "discover_agilex_trajectories",
    "discover_cached_agilex_trajectories",
]
