"""Latent-based dynamics dataset for the D3-Disc predictor.

Key idea: the flow_multi encoder is frozen, and per-demo 256-D latents are
already on disk under ``data/.lpb_score_cache/<task>/<sha1>.npz``. So the
training dataset reads ``z_t`` directly from the cache and only needs the
source HDF5 for ``states`` and ``actions``. This avoids running the
flow_multi image encoder per batch.

Each training sample is a single (t, t+h) transition:
    z_t  in R^{256},   s_t  in R^{proprio_dim}
    a_{t:t+h}  in R^{H x action_dim}
    z_{t+h},  s_{t+h}   (targets)

Task name is inferred from the HDF5 path (data/<task>/<kind>/<file>.hdf5).
Proprio handling mirrors ``lpb.dataset.LatentDynamicsDataset`` — raw state is
right-padded / truncated so that demos from tasks with different state_dims
stack cleanly.
"""

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

from .encoder import FlowMultiEncoderWrapper


def _expand_hdf5_inputs(paths: Optional[Sequence[str]]) -> List[str]:
    if not paths:
        return []
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.hdf5"))))
        elif os.path.isfile(p) and p.endswith(".hdf5"):
            out.append(p)
    return sorted(set(out))


def _infer_task_name(file_path: str) -> str:
    """Infer task from path ``<...>/data/<task>/<kind>/<file>.hdf5``."""
    parts = Path(file_path).resolve().parts
    # Grandparent = task directory.
    if len(parts) < 3:
        raise ValueError(f"Cannot infer task name from path: {file_path}")
    return str(parts[-3])


@dataclass(frozen=True)
class _TransitionRef:
    file_path: str
    demo_key: str
    task_name: str
    cache_path: str
    t: int
    horizon: int
    is_expert: bool


class LatentFlowDynamicsDataset(Dataset):
    """Dataset over (z_t, s_t, a_{t:t+h}, z_{t+h}, s_{t+h}) transitions.

    Latents are read via ``FlowMultiEncoderWrapper.load_or_encode_demo`` so the
    on-disk cache is reused byte-exactly (and only populated on miss).

    ``max_trajectories_per_kind`` caps the number of trajectories kept
    **per task per kind** (kind = expert vs rollout). A single global cap
    across all files causes the first-alphabetical task to exhaust the quota,
    producing a single-task dataset despite multi-task inputs.
    """

    def __init__(
        self,
        encoder: FlowMultiEncoderWrapper,
        *,
        expert_paths: Optional[Sequence[str]] = None,
        rollout_paths: Optional[Sequence[str]] = None,
        horizon: int = 1,
        proprio_indices: Optional[Sequence[int]] = None,
        max_trajectories_per_kind: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.proprio_indices = (
            None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        )
        if self.proprio_indices is not None and self.proprio_indices.size == 0:
            self.proprio_indices = None

        expert_files = _expand_hdf5_inputs(expert_paths)
        rollout_files = _expand_hdf5_inputs(rollout_paths)
        if not expert_files and not rollout_files:
            raise FileNotFoundError("No expert/rollout HDF5 files found.")

        max_traj = None if max_trajectories_per_kind in (None, 0) else int(max_trajectories_per_kind)

        self._refs: List[_TransitionRef] = []
        self._sample_is_expert: List[bool] = []
        self._proprio_dim: Optional[int] = None
        self._action_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None
        self._num_expert_traj = 0
        self._num_rollout_traj = 0

        # Cache of per-demo encoded latents: path -> (L, 256) ndarray.
        # Keeps RAM bounded; warm-on-first-access. Most runs will have <= a few
        # thousand demos x ~400 frames x 256 dims = O(100 MB) which is fine.
        self._latent_cache: Dict[str, np.ndarray] = {}

        # Per-demo states + actions cache, keyed by (file_path, demo_key).
        # Opening the hdf5 on EVERY __getitem__ was the dataloader bottleneck
        # (1M+ samples × h5py.File() open/close → GPU starves). Preloaded once
        # after scan; ~700 MB RAM for 1M transitions (states: 71 × 4 B,
        # actions: horizon × 7 × 4 B per transition).
        self._states_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._actions_cache: Dict[Tuple[str, str], np.ndarray] = {}

        self._scan_files(expert_files, is_expert=True, max_traj=max_traj)
        self._scan_files(rollout_files, is_expert=False, max_traj=max_traj)

        if not self._refs:
            raise RuntimeError("No valid transitions found.")

    # ------------------------------------------------------------------ #
    # Indexing                                                           #
    # ------------------------------------------------------------------ #

    def _scan_files(self, files: Sequence[str], is_expert: bool, max_traj: Optional[int]) -> None:
        # max_traj is per-task (and per-kind, via the is_expert split). A global
        # counter would let the first-alphabetical task's files exhaust the quota
        # before any other task is scanned, producing a single-task dataset.
        task_counts: Dict[str, int] = {}
        total = 0
        for fp in files:
            task_name = _infer_task_name(fp)
            if max_traj is not None and task_counts.get(task_name, 0) >= max_traj:
                continue
            try:
                with h5py.File(fp, "r") as f:
                    if "demos" not in f:
                        continue
                    for demo_key in sorted(f["demos"].keys()):
                        if max_traj is not None and task_counts.get(task_name, 0) >= max_traj:
                            break
                        demo = f["demos"][demo_key]
                        if "states" not in demo or "actions" not in demo:
                            continue
                        states = demo["states"]
                        actions = demo["actions"]
                        length = int(min(states.shape[0], actions.shape[0]))
                        if length <= self.horizon:
                            continue
                        state_dim = int(states.shape[1])
                        action_dim = int(actions.shape[1])

                        if self.proprio_indices is not None and np.max(self.proprio_indices) >= state_dim:
                            raise ValueError(
                                f"proprio_indices out of range in {fp}:{demo_key} "
                                f"for state_dim={state_dim}"
                            )
                        if self._action_dim is None:
                            self._action_dim = action_dim
                        elif action_dim != self._action_dim:
                            raise ValueError(
                                f"Action dim mismatch in {fp}:{demo_key}. "
                                f"expected={self._action_dim}, got={action_dim}"
                            )

                        if self.proprio_indices is not None:
                            effective_proprio = int(self.proprio_indices.shape[0])
                        else:
                            effective_proprio = state_dim
                        if self._proprio_dim is None or effective_proprio > self._proprio_dim:
                            self._proprio_dim = effective_proprio

                        cache_path = self.encoder.cache_path(
                            task_name=task_name, file_path=fp, demo_key=demo_key
                        )

                        # Truncate to shared length of latents and HDF5 data.
                        max_t = length - self.horizon
                        for t in range(max_t):
                            self._refs.append(
                                _TransitionRef(
                                    file_path=fp,
                                    demo_key=demo_key,
                                    task_name=task_name,
                                    cache_path=cache_path,
                                    t=t,
                                    horizon=self.horizon,
                                    is_expert=is_expert,
                                )
                            )
                            self._sample_is_expert.append(is_expert)
                        task_counts[task_name] = task_counts.get(task_name, 0) + 1
                        total += 1
            except OSError as exc:
                print(f"[d3_disc][dataset] skip {fp}: {exc}")

        if is_expert:
            self._num_expert_traj += total
        else:
            self._num_rollout_traj += total

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def action_dim(self) -> int:
        assert self._action_dim is not None
        return self._action_dim

    @property
    def proprio_dim(self) -> int:
        assert self._proprio_dim is not None
        return self._proprio_dim

    @property
    def latent_dim(self) -> int:
        # Peek at the encoder's output dim (populated at construction).
        return int(self.encoder.latent_dim)

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

    def __len__(self) -> int:
        return len(self._refs)

    def materialize_missing_latent_caches(self) -> int:
        """Encode demos whose ``.npz`` cache is absent, on the current process only.

        When ``DataLoader`` uses ``num_workers>0``, workers are forked on Linux. If the
        parent has already initialized CUDA, a cache miss in ``__getitem__`` would call
        ``FlowMultiEncoderWrapper`` in the worker and trigger
        ``Cannot re-initialize CUDA in forked subprocess``. Running this once in the
        main process before the dataloader exists ensures workers only read numpy from
        disk.

        Returns:
            Number of distinct demos (cache keys) that required encoding.
        """
        seen: set[str] = set()
        n_encoded = 0
        for ref in self._refs:
            if ref.cache_path in seen:
                continue
            seen.add(ref.cache_path)
            if os.path.isfile(ref.cache_path):
                continue
            self.encoder.load_or_encode_demo(
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
            n_encoded += 1
        return int(n_encoded)

    # ------------------------------------------------------------------ #
    # Sampling                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_slice_proprio(self, state: np.ndarray) -> np.ndarray:
        if self.proprio_indices is not None:
            return state[self.proprio_indices]
        target = int(self._proprio_dim) if self._proprio_dim is not None else int(state.shape[0])
        d = int(state.shape[0])
        if d == target:
            return state
        if d > target:
            return state[:target]
        pad = np.zeros((target - d,), dtype=state.dtype)
        return np.concatenate([state, pad], axis=0)

    def _get_states_actions(self, ref: _TransitionRef) -> Tuple[np.ndarray, np.ndarray]:
        key = (ref.file_path, ref.demo_key)
        s = self._states_cache.get(key)
        a = self._actions_cache.get(key)
        if s is not None and a is not None:
            return s, a
        with h5py.File(ref.file_path, "r") as f:
            demo = f["demos"][ref.demo_key]
            s = np.asarray(demo["states"][:], dtype=np.float32)
            a = np.asarray(demo["actions"][:], dtype=np.float32)
        self._states_cache[key] = s
        self._actions_cache[key] = a
        return s, a

    def preload_states_actions(self) -> int:
        """Warm the per-demo states/actions cache from disk.

        Call before constructing the DataLoader so worker processes inherit
        the populated dict via copy-on-write (Linux fork). Skips the per-
        sample hdf5 open that was dominating __getitem__ cost.
        """
        seen: set[Tuple[str, str]] = set()
        for ref in self._refs:
            key = (ref.file_path, ref.demo_key)
            if key in seen:
                continue
            seen.add(key)
            self._get_states_actions(ref)
        return len(seen)

    def _get_latents(self, ref: _TransitionRef) -> np.ndarray:
        cached = self._latent_cache.get(ref.cache_path)
        if cached is not None:
            return cached
        if os.path.isfile(ref.cache_path):
            with np.load(ref.cache_path) as data:
                latents = np.asarray(data["latents"], dtype=np.float32)
        else:
            encoded = self.encoder.load_or_encode_demo(
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
            latents = np.asarray(encoded.latents, dtype=np.float32)
        self._latent_cache[ref.cache_path] = latents
        return latents

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self._refs[index]
        latents = self._get_latents(ref)
        t0 = ref.t
        th = t0 + ref.horizon
        # The latents cache may be shorter than the raw HDF5 (alignment of
        # images/states/actions); clamp th defensively.
        if th >= latents.shape[0]:
            th = int(latents.shape[0] - 1)
            t0 = max(0, th - ref.horizon)

        z_t = latents[t0]
        z_tp = latents[th]

        states_arr, actions_arr = self._get_states_actions(ref)
        s_t = states_arr[t0]
        s_tp = states_arr[th]
        a_seq = actions_arr[t0:th]
        if a_seq.shape[0] < ref.horizon:
            pad = np.repeat(a_seq[-1:] if a_seq.shape[0] > 0 else
                            np.zeros((1, self.action_dim), dtype=np.float32),
                            ref.horizon - a_seq.shape[0], axis=0)
            a_seq = np.concatenate([a_seq, pad], axis=0)

        s_t = self._maybe_slice_proprio(s_t)
        s_tp = self._maybe_slice_proprio(s_tp)

        return {
            "current_latent": torch.from_numpy(z_t.astype(np.float32)),
            "current_proprio": torch.from_numpy(s_t.astype(np.float32)),
            "action_sequence": torch.from_numpy(a_seq.astype(np.float32)),
            "target_latent": torch.from_numpy(z_tp.astype(np.float32)),
            "target_proprio": torch.from_numpy(s_tp.astype(np.float32)),
            "is_expert": torch.tensor(1 if ref.is_expert else 0, dtype=torch.int64),
        }
