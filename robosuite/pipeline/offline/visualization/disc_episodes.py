"""Collected offline episode adapters for discriminator visualization."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from benchmark.core import BenchmarkTrajectory

from robosuite.pipeline.offline.discriminator.episodes import load_offline_episodes


@dataclass
class OfflineEpisodeTrajectory(BenchmarkTrajectory):
    """In-memory benchmark-compatible view of one collected episode."""

    episode_index: int = 0
    terminal_reason: str = ""
    intervention_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.bool_)
    )
    observations: Mapping[str, Any] = field(default_factory=dict, repr=False)
    states: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32), repr=False
    )
    actions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32), repr=False
    )
    source_kind: str = "offline"
    states_are_preselected_proprio: bool = True
    video_frame_size: Optional[tuple[int, int]] = None

    @property
    def intervention_frames(self) -> int:
        return int(np.asarray(self.intervention_mask, dtype=np.bool_).sum())

    def load_images(
        self,
        cameras: Optional[Sequence[str]] = None,
    ) -> dict[str, np.ndarray]:
        requested = tuple(cameras) if cameras is not None else self.available_cameras
        missing = [name for name in requested if name not in self.available_cameras]
        if missing:
            raise KeyError(
                f"cameras {missing} not available for {self.video_id}; "
                f"available: {self.available_cameras}"
            )
        return {
            str(name): np.asarray(self.observations[str(name)], dtype=np.uint8)
            for name in requested
        }

    def load_states(self) -> np.ndarray:
        return np.asarray(self.states, dtype=np.float32)

    def load_actions(self) -> np.ndarray:
        return np.asarray(self.actions, dtype=np.float32)

    def load_failure_mask(self) -> None:
        return None

    def load_failure_segment_index(self) -> None:
        return None


def offline_trajectory(
    episode: Mapping[str, Any],
    *,
    episode_index: int,
    task: str,
    camera_names: Sequence[str],
    fps: int,
    source_path: str,
    video_size: int = 256,
) -> OfflineEpisodeTrajectory:
    """Map validated recorded fields without relabeling policy or human actions."""
    actions = np.asarray(episode["executed_action"], dtype=np.float32)
    observations = episode["obs"]
    return OfflineEpisodeTrajectory(
        task_name=str(task),
        num_frames=int(actions.shape[0]),
        is_failure=True,
        video_id=f"offline_episode_{int(episode_index):06d}",
        fps=int(fps),
        available_cameras=tuple(str(name) for name in camera_names),
        failure_segments=[],
        source_hdf5_path=str(source_path),
        source_demo_key=f"episode_{int(episode_index):06d}",
        episode_index=int(episode_index),
        terminal_reason=str(episode["terminal_reason"]),
        intervention_mask=np.asarray(episode["is_intervention"], dtype=np.bool_),
        observations=observations,
        states=np.asarray(observations["state"], dtype=np.float32),
        actions=actions,
        video_frame_size=(int(video_size), int(video_size)),
    )


def sample_offline_trajectories(
    episodes_path: str | Path,
    *,
    task: str,
    num_trajs: int,
    seed: int,
    fps: int,
    video_size: int = 256,
) -> list[OfflineEpisodeTrajectory]:
    """Deterministically sample collected episodes without replacement."""
    payload = load_offline_episodes(episodes_path)
    payload_task = payload.get("task_name")
    if payload_task is not None and str(payload_task) != str(task):
        raise ValueError(
            f"Offline episodes task mismatch: payload={payload_task!r}, cli={task!r}."
        )
    episodes = list(payload["episodes"])
    count = min(int(num_trajs), len(episodes))
    if count <= 0:
        raise ValueError(f"--num-trajs must be positive, got {num_trajs}.")
    indices = random.Random(int(seed)).sample(range(len(episodes)), count)
    sampled = [
        offline_trajectory(
            episodes[index],
            episode_index=index,
            task=str(task),
            camera_names=payload["camera_names"],
            fps=int(fps),
            source_path=str(payload["_resolved_path"]),
            video_size=int(video_size),
        )
        for index in indices
    ]
    print(
        f"[pu_bce][viz] sampled {len(sampled)}/{len(episodes)} offline trajectories "
        f"from {payload['_resolved_path']} (seed={int(seed)})",
        flush=True,
    )
    return sampled


__all__ = [
    "OfflineEpisodeTrajectory",
    "offline_trajectory",
    "sample_offline_trajectories",
]
