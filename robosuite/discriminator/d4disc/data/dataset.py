from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import PreprocessedCacheReader, PreprocessedDemo


@dataclass(frozen=True)
class _TransitionRef:
    file_path: str
    demo_key: str
    task_name: str
    t: int
    horizon: int
    is_expert: bool
    is_fail_raw: bool


def _expand_hdf5_inputs(paths: Optional[Sequence[str]]) -> List[str]:
    if not paths:
        return []
    out: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            out.extend(sorted(glob.glob(os.path.join(path, "*.hdf5"))))
        elif os.path.isfile(path) and path.endswith(".hdf5"):
            out.append(path)
    return sorted(set(out))


def _infer_task_name(file_path: str) -> str:
    parts = Path(file_path).resolve().parts
    if len(parts) < 3:
        raise ValueError(f"Cannot infer task name from path: {file_path}")
    return str(parts[-3])


class LatentFlowDynamicsDatasetD4(Dataset):
    def __init__(
        self,
        cache_reader: PreprocessedCacheReader,
        *,
        expert_paths: Optional[Sequence[str]] = None,
        rollout_paths: Optional[Sequence[str]] = None,
        fail_rollout_paths: Optional[Sequence[str]] = None,
        horizon: int = 1,
        proprio_indices: Optional[Sequence[int]] = None,
        max_trajectories_per_kind: Optional[int] = None,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.cache_reader = cache_reader
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self.image_size = int(image_size)
        self.proprio_indices = (
            None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        )
        if self.proprio_indices is not None and self.proprio_indices.size == 0:
            self.proprio_indices = None

        expert_files = _expand_hdf5_inputs(expert_paths)
        rollout_files = _expand_hdf5_inputs(rollout_paths)
        fail_files = _expand_hdf5_inputs(fail_rollout_paths)
        if not expert_files and not rollout_files and not fail_files:
            raise FileNotFoundError("No expert/rollout/fail HDF5 files found.")

        self._max_traj_per_kind = (
            None if max_trajectories_per_kind in (None, 0) else int(max_trajectories_per_kind)
        )

        self._refs: List[_TransitionRef] = []
        self._sample_is_expert: List[bool] = []
        self._demo_cache: Dict[Tuple[str, str, str], PreprocessedDemo] = {}
        self._coverage: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._action_dim: Optional[int] = None
        self._proprio_dim: Optional[int] = None
        self._num_expert_traj = 0
        self._num_rollout_traj = 0

        self._scan_files(
            expert_files,
            data_type="expert",
            is_expert=True,
            is_fail_raw=False,
        )
        self._scan_files(
            rollout_files,
            data_type="success_rollout",
            is_expert=False,
            is_fail_raw=False,
        )
        self._scan_files(
            fail_files,
            data_type="fail_rollout",
            is_expert=False,
            is_fail_raw=True,
        )

        if not self._refs:
            raise RuntimeError("No cached transitions found for the provided inputs.")

        self._is_fail_raw = torch.tensor([ref.is_fail_raw for ref in self._refs], dtype=torch.bool)
        self.gamma_buffer = torch.full((len(self._refs),), 0.5, dtype=torch.float32)
        self._print_coverage()

    def _bucket(self, task_name: str, data_type: str) -> Dict[str, int]:
        return self._coverage.setdefault(task_name, {}).setdefault(
            data_type,
            {"total": 0, "cached": 0, "selected": 0},
        )

    def _scan_files(
        self,
        files: Sequence[str],
        *,
        data_type: str,
        is_expert: bool,
        is_fail_raw: bool,
    ) -> None:
        task_counts: Dict[str, int] = {}
        kept = 0
        for file_path in files:
            task_name = _infer_task_name(file_path)
            try:
                with h5py.File(file_path, "r") as handle:
                    if "demos" not in handle:
                        continue
                    for demo_key in sorted(handle["demos"].keys()):
                        bucket = self._bucket(task_name, data_type)
                        bucket["total"] += 1

                        demo = handle["demos"][demo_key]
                        if "states" not in demo or "actions" not in demo:
                            continue

                        cache_path = self.cache_reader.cache_path(task_name, file_path, demo_key)
                        if not os.path.isfile(cache_path):
                            continue
                        bucket["cached"] += 1

                        if self._max_traj_per_kind is not None:
                            if task_counts.get(task_name, 0) >= self._max_traj_per_kind:
                                continue

                        with np.load(cache_path) as cached:
                            images = cached["images_chw"]
                            proprio = cached["proprio"]
                            actions = cached["actions"]
                            length = min(
                                int(images.shape[0]),
                                int(proprio.shape[0]),
                                int(actions.shape[0]),
                            )
                            action_dim = int(actions.shape[1])
                            proprio_dim = int(proprio.shape[1])

                        if length <= self.horizon:
                            continue
                        if self.proprio_indices is not None and np.max(self.proprio_indices) >= proprio_dim:
                            raise ValueError(
                                f"proprio_indices out of range in cache for {file_path}:{demo_key} "
                                f"with proprio_dim={proprio_dim}"
                            )
                        if self._action_dim is None:
                            self._action_dim = action_dim
                        elif action_dim != self._action_dim:
                            raise ValueError(
                                f"Action dim mismatch in {file_path}:{demo_key}. "
                                f"expected={self._action_dim}, got={action_dim}"
                            )

                        effective_proprio = (
                            int(self.proprio_indices.shape[0])
                            if self.proprio_indices is not None
                            else proprio_dim
                        )
                        if self._proprio_dim is None or effective_proprio > self._proprio_dim:
                            self._proprio_dim = effective_proprio

                        max_t = length - self.horizon
                        for t in range(max_t):
                            self._refs.append(
                                _TransitionRef(
                                    file_path=file_path,
                                    demo_key=demo_key,
                                    task_name=task_name,
                                    t=t,
                                    horizon=self.horizon,
                                    is_expert=is_expert,
                                    is_fail_raw=is_fail_raw,
                                )
                            )
                            self._sample_is_expert.append(is_expert)

                        bucket["selected"] += 1
                        task_counts[task_name] = task_counts.get(task_name, 0) + 1
                        kept += 1
            except OSError as exc:
                print(f"[d4_disc][dataset] skip {file_path}: {exc}")

        if is_expert:
            self._num_expert_traj += kept
        else:
            self._num_rollout_traj += kept

    def _print_coverage(self) -> None:
        for task_name in sorted(self._coverage):
            for data_type in ("expert", "success_rollout", "fail_rollout"):
                bucket = self._coverage[task_name].get(data_type)
                if bucket is None:
                    continue
                print(
                    f"[d4_disc][dataset] task={task_name} kind={data_type} "
                    f"total={bucket['total']} cached={bucket['cached']} selected={bucket['selected']}"
                )

    @property
    def action_dim(self) -> int:
        assert self._action_dim is not None
        return self._action_dim

    @property
    def latent_dim(self) -> int:
        return 512

    @property
    def proprio_dim(self) -> int:
        assert self._proprio_dim is not None
        return self._proprio_dim

    @property
    def sample_is_expert(self) -> List[bool]:
        return self._sample_is_expert

    @property
    def num_expert_samples(self) -> int:
        return int(sum(self._sample_is_expert))

    @property
    def num_rollout_samples(self) -> int:
        return len(self._sample_is_expert) - self.num_expert_samples

    @property
    def num_expert_trajectories(self) -> int:
        return self._num_expert_traj

    @property
    def num_rollout_trajectories(self) -> int:
        return self._num_rollout_traj

    @property
    def is_fail_raw(self) -> torch.Tensor:
        return self._is_fail_raw

    @property
    def num_fail_raw_samples(self) -> int:
        return int(self._is_fail_raw.sum().item())

    def __len__(self) -> int:
        return len(self._refs)

    def _demo_cache_key(self, ref: _TransitionRef) -> Tuple[str, str, str]:
        return (ref.task_name, ref.file_path, ref.demo_key)

    def _load_demo(self, ref: _TransitionRef) -> PreprocessedDemo:
        key = self._demo_cache_key(ref)
        cached = self._demo_cache.get(key)
        if cached is not None:
            return cached
        demo = self.cache_reader.load(ref.task_name, ref.file_path, ref.demo_key)
        self._demo_cache[key] = demo
        return demo

    def preload_preprocessed(self) -> int:
        loaded = 0
        seen = set()
        for ref in self._refs:
            key = self._demo_cache_key(ref)
            if key in seen:
                continue
            seen.add(key)
            if key not in self._demo_cache:
                self._demo_cache[key] = self.cache_reader.load(*key)
                loaded += 1
        return int(loaded)

    def _maybe_slice_proprio(self, proprio: np.ndarray) -> np.ndarray:
        if self.proprio_indices is not None:
            return proprio[self.proprio_indices]
        target = int(self._proprio_dim) if self._proprio_dim is not None else int(proprio.shape[0])
        dim = int(proprio.shape[0])
        if dim == target:
            return proprio
        if dim > target:
            return proprio[:target]
        pad = np.zeros((target - dim,), dtype=proprio.dtype)
        return np.concatenate([proprio, pad], axis=0)

    def _slice_action_sequence(self, actions: np.ndarray, t0: int) -> np.ndarray:
        end = min(int(actions.shape[0]), int(t0 + self.horizon))
        chunk = np.asarray(actions[t0:end], dtype=np.float32)
        if chunk.shape[0] == 0:
            return np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        if chunk.shape[1] < self.action_dim:
            pad = np.zeros((chunk.shape[0], self.action_dim - chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=1)
        elif chunk.shape[1] > self.action_dim:
            chunk = chunk[:, : self.action_dim]
        if chunk.shape[0] < self.horizon:
            tail = np.repeat(chunk[-1:], self.horizon - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, tail], axis=0)
        return chunk

    @torch.no_grad()
    def update_gamma(
        self,
        indices: torch.Tensor,
        new_gamma: torch.Tensor,
        ema_alpha: float = 0.5,
        clamp_eps: float = 1e-3,
    ) -> None:
        idx = indices.detach().cpu().long().view(-1)
        ng = new_gamma.detach().cpu().float().view(-1)
        if idx.numel() != ng.numel():
            raise ValueError(f"indices/new_gamma length mismatch: {idx.numel()} vs {ng.numel()}")
        if idx.numel() == 0:
            return
        old = self.gamma_buffer[idx]
        mixed = float(ema_alpha) * old + (1.0 - float(ema_alpha)) * ng
        self.gamma_buffer[idx] = mixed.clamp(float(clamp_eps), 1.0 - float(clamp_eps))

    def fail_indices(self) -> torch.Tensor:
        return self._is_fail_raw.nonzero(as_tuple=False).squeeze(-1)

    def __getitem__(self, index: int) -> dict:
        ref = self._refs[index]
        demo = self._load_demo(ref)
        t0 = int(ref.t)
        th = min(t0 + int(ref.horizon), int(demo.length) - 1)

        cur_img = torch.from_numpy(np.ascontiguousarray(demo.images_chw[t0]))
        tgt_img = torch.from_numpy(np.ascontiguousarray(demo.images_chw[th]))

        cur_prop = self._maybe_slice_proprio(np.asarray(demo.proprio[t0], dtype=np.float32))
        tgt_prop = self._maybe_slice_proprio(np.asarray(demo.proprio[th], dtype=np.float32))
        act_seq = self._slice_action_sequence(demo.actions, t0=t0)

        return {
            "current_image": cur_img,
            "current_proprio": torch.from_numpy(cur_prop),
            "action_sequence": torch.from_numpy(act_seq),
            "target_image": tgt_img,
            "target_proprio": torch.from_numpy(tgt_prop),
            "is_expert": torch.tensor(1 if ref.is_expert else 0, dtype=torch.int64),
            "sample_idx": torch.tensor(int(index), dtype=torch.long),
            "is_fail_raw": torch.tensor(bool(ref.is_fail_raw), dtype=torch.bool),
            "gamma": torch.tensor(float(self.gamma_buffer[index].item()), dtype=torch.float32),
        }
