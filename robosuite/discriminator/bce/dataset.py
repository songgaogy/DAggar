from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.lpb_new.core.dataset import (
    DATA_TYPE_ORDER,
    EncodedTrajectoryRef,
    LatentTrajectory,
    build_cached_splits as _build_cached_splits_impl,
    filter_refs_by_data_types,
    load_cached_latent_trajectory,
    load_latent_trajectories,
)


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_cached_splits(
    cfg_data: Any,
    encoder: FrozenFlowMultitaskEncoder,
    seed: int,
) -> tuple[dict[str, list[EncodedTrajectoryRef]], dict[str, dict[str, dict[str, int]]], dict[str, int]]:
    cached_splits, split_summary, task_to_index = _build_cached_splits_impl(
        cfg_data=cfg_data,
        encoder=encoder,
        seed=seed,
    )
    beta_by_task = parse_task_beta_map(
        cfg_data=cfg_data,
        default_beta=float(_cfg_get(_cfg_get(cfg_data, "labels", None), "default_beta", 2.0)),
    )
    for task_name, beta in beta_by_task.items():
        print(f"[tpud] task={task_name} beta={beta:.4f}")
    return cached_splits, split_summary, task_to_index


def parse_task_beta_map(cfg_data: Any, default_beta: float = 2.0) -> dict[str, float]:
    tasks_cfg = _cfg_get(cfg_data, "tasks", {})
    labels_cfg = _cfg_get(cfg_data, "labels", None)
    shared_default = float(_cfg_get(labels_cfg, "default_beta", default_beta))
    beta_by_task: dict[str, float] = {}
    for task_name in tasks_cfg.keys():
        task_cfg = _cfg_get(tasks_cfg, task_name)
        beta = float(_cfg_get(task_cfg, "beta", shared_default))
        beta_by_task[str(task_name)] = beta
    return beta_by_task


def build_temporal_soft_targets(num_steps: int, beta: float) -> np.ndarray:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if int(num_steps) == 1:
        return np.asarray([1.0], dtype=np.float32)
    progress = np.arange(int(num_steps), dtype=np.float32) / float(max(int(num_steps) - 1, 1))
    targets = 1.0 - np.power(progress, float(beta), dtype=np.float32)
    return np.clip(targets.astype(np.float32), 0.0, 1.0)


def build_temporal_binary_labels(
    num_steps: int,
    beta: float,
    inlier_cutoff: float = 0.5,
) -> np.ndarray:
    soft_targets = build_temporal_soft_targets(num_steps=num_steps, beta=beta)
    return (soft_targets <= float(inlier_cutoff)).astype(np.int64)


@dataclass(frozen=True)
class TransitionRef:
    trajectory_index: int
    t: int
    horizon: int
    task_name: str
    is_positive: bool
    soft_target: float


class TemporalTransitionDataset(Dataset):
    def __init__(
        self,
        trajectory_refs: Sequence[EncodedTrajectoryRef],
        beta_by_task: dict[str, float],
        horizon: int = 1,
        preload_to_memory: bool = False,
    ) -> None:
        super().__init__()
        self.trajectory_refs = list(trajectory_refs)
        self.beta_by_task = {str(k): float(v) for k, v in beta_by_task.items()}
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if not self.trajectory_refs:
            raise ValueError("trajectory_refs cannot be empty")

        self.preload_to_memory = bool(preload_to_memory)
        self._trajectory_cache: dict[int, LatentTrajectory] = {}
        self._action_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None
        self._transition_refs: list[TransitionRef] = []
        self._sample_is_positive: list[bool] = []
        self._task_sample_counts: dict[str, int] = {}
        self._task_positive_counts: dict[str, int] = {}
        self._task_failure_counts: dict[str, int] = {}

        for traj_idx, ref in enumerate(self.trajectory_refs):
            traj = load_cached_latent_trajectory(ref)
            length = min(int(traj.latents.shape[0]), int(traj.actions.shape[0]))
            usable = length - self.horizon
            if usable <= 0:
                continue
            if self._action_dim is None:
                self._action_dim = int(traj.actions.shape[-1])
                self._latent_dim = int(traj.latents.shape[-1])
            else:
                if int(traj.actions.shape[-1]) != self._action_dim:
                    raise ValueError(
                        f"Action dim mismatch in {traj.file_path}:{traj.demo_key}. "
                        f"expected={self._action_dim}, got={traj.actions.shape[-1]}"
                    )
                if int(traj.latents.shape[-1]) != self._latent_dim:
                    raise ValueError(
                        f"Latent dim mismatch in {traj.file_path}:{traj.demo_key}. "
                        f"expected={self._latent_dim}, got={traj.latents.shape[-1]}"
                    )

            is_positive = ref.data_type in {"expert", "success_rollout"}
            if is_positive:
                soft_targets = np.ones(usable, dtype=np.float32)
            else:
                beta = float(self.beta_by_task.get(ref.task_name, 2.0))
                soft_targets = build_temporal_soft_targets(num_steps=usable, beta=beta)

            self._task_sample_counts[ref.task_name] = self._task_sample_counts.get(ref.task_name, 0) + int(usable)
            if is_positive:
                self._task_positive_counts[ref.task_name] = self._task_positive_counts.get(ref.task_name, 0) + int(usable)
            else:
                self._task_failure_counts[ref.task_name] = self._task_failure_counts.get(ref.task_name, 0) + int(usable)

            for t, soft_target in enumerate(soft_targets.tolist()):
                self._transition_refs.append(
                    TransitionRef(
                        trajectory_index=traj_idx,
                        t=int(t),
                        horizon=self.horizon,
                        task_name=ref.task_name,
                        is_positive=bool(is_positive),
                        soft_target=float(soft_target),
                    )
                )
                self._sample_is_positive.append(bool(is_positive))

            if self.preload_to_memory:
                self._trajectory_cache[traj_idx] = traj

        if not self._transition_refs:
            raise RuntimeError("No valid transitions found for the provided trajectory refs.")

    @property
    def action_dim(self) -> int:
        assert self._action_dim is not None
        return self._action_dim

    @property
    def latent_dim(self) -> int:
        assert self._latent_dim is not None
        return self._latent_dim

    @property
    def sample_is_positive(self) -> list[bool]:
        return self._sample_is_positive

    @property
    def num_positive_samples(self) -> int:
        return int(sum(self._sample_is_positive))

    @property
    def num_failure_samples(self) -> int:
        return len(self._sample_is_positive) - self.num_positive_samples

    @property
    def task_sample_counts(self) -> dict[str, int]:
        return dict(self._task_sample_counts)

    @property
    def task_positive_counts(self) -> dict[str, int]:
        return dict(self._task_positive_counts)

    @property
    def task_failure_counts(self) -> dict[str, int]:
        return dict(self._task_failure_counts)

    def __len__(self) -> int:
        return len(self._transition_refs)

    def _get_trajectory(self, trajectory_index: int) -> LatentTrajectory:
        traj = self._trajectory_cache.get(trajectory_index)
        if traj is not None:
            return traj
        traj = load_cached_latent_trajectory(self.trajectory_refs[trajectory_index])
        self._trajectory_cache[trajectory_index] = traj
        return traj

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self._transition_refs[index]
        traj = self._get_trajectory(ref.trajectory_index)
        t0 = int(ref.t)
        th = t0 + int(ref.horizon)
        action_sequence = traj.actions[t0:th]
        progress = float(t0 / max((min(int(traj.latents.shape[0]), int(traj.actions.shape[0])) - self.horizon - 1), 1))
        return {
            "current_latent": torch.from_numpy(np.asarray(traj.latents[t0], dtype=np.float32)),
            "action_sequence": torch.from_numpy(np.asarray(action_sequence, dtype=np.float32)),
            "task_index": torch.tensor(int(traj.task_index), dtype=torch.int64),
            "data_type_index": torch.tensor(int(traj.data_type_index), dtype=torch.int64),
            "soft_target": torch.tensor(float(ref.soft_target), dtype=torch.float32),
            "ood_target": torch.tensor(float(1.0 - ref.soft_target), dtype=torch.float32),
            "is_positive": torch.tensor(1 if ref.is_positive else 0, dtype=torch.int64),
            "progress": torch.tensor(progress, dtype=torch.float32),
        }


__all__ = [
    "DATA_TYPE_ORDER",
    "EncodedTrajectoryRef",
    "LatentTrajectory",
    "TemporalTransitionDataset",
    "build_cached_splits",
    "build_temporal_binary_labels",
    "build_temporal_soft_targets",
    "filter_refs_by_data_types",
    "load_latent_trajectories",
    "parse_task_beta_map",
]
