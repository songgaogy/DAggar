"""Extends ``d3disc.LatentFlowDynamicsDataset`` with a contamination-aware
``fail_rollout`` branch + a mutable per-sample gamma buffer.

Key additions:

- ``fail_rollout_paths`` kwarg: same scan logic as rollout_paths, but every
  transition coming from these HDF5s is tagged ``is_fail_raw=True``.
- ``gamma_buffer: (N,) float32`` on CPU, initialized to 0.5. Updated in-place
  each bootstrap epoch by ``compute_advantage_gate``. EMA-smoothed and clamped
  to (1e-3, 1 - 1e-3) — design §9.3 rationale: hard 0/1 kills routing
  gradients on one branch.
- ``sample_idx``, ``is_fail_raw``, ``gamma`` added to ``__getitem__`` output.

Clean positives (``expert_paths`` + ``rollout_paths``) retain
``is_fail_raw=False``. They are used as pure success frames in both phases,
never routed to the fail branch.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch

from robosuite.discriminator.d3disc.dataset import (
    LatentFlowDynamicsDataset,
    _TransitionRef,
    _expand_hdf5_inputs,
    _infer_task_name,
)
from robosuite.discriminator.d3disc.encoder import FlowMultiEncoderWrapper


class LatentFlowDynamicsDatasetD4(LatentFlowDynamicsDataset):
    def __init__(
        self,
        encoder: FlowMultiEncoderWrapper,
        *,
        expert_paths: Optional[Sequence[str]] = None,
        rollout_paths: Optional[Sequence[str]] = None,
        fail_rollout_paths: Optional[Sequence[str]] = None,
        horizon: int = 1,
        proprio_indices: Optional[Sequence[int]] = None,
        max_trajectories_per_kind: Optional[int] = None,
    ) -> None:
        # Allow empty expert+rollout if fail paths are present; the parent raises
        # otherwise, so we pre-seed with an empty fail-scan context.
        fail_files = _expand_hdf5_inputs(fail_rollout_paths) if fail_rollout_paths else []

        # The parent __init__ will reject "no expert and no rollout" as an
        # input error. We always want to support fail-only datasets for
        # replay/scoring, so we scan expert+rollout first (possibly empty)
        # and append fail afterwards. When both parent lists are empty, we
        # temporarily pass a sentinel then trim.
        self._fail_files_for_init = list(fail_files)
        parent_expert = list(expert_paths) if expert_paths else []
        parent_rollout = list(rollout_paths) if rollout_paths else []
        if (not parent_expert) and (not parent_rollout):
            if not fail_files:
                raise FileNotFoundError("No expert/rollout/fail HDF5 files found.")
            # Use fail as rollout for the parent scan; we will retag afterwards.
            parent_rollout = list(fail_files)
            self._fail_files_for_init = []
            self._bootstrap_from_fail = True
        else:
            self._bootstrap_from_fail = False

        super().__init__(
            encoder=encoder,
            expert_paths=parent_expert,
            rollout_paths=parent_rollout,
            horizon=horizon,
            proprio_indices=proprio_indices,
            max_trajectories_per_kind=max_trajectories_per_kind,
        )

        self._max_traj_per_kind = (
            None if max_trajectories_per_kind in (None, 0) else int(max_trajectories_per_kind)
        )

        # Retag the bootstrap-from-fail case (parent treated them as rollouts).
        is_fail_raw = [False] * len(self._refs)
        if self._bootstrap_from_fail:
            fail_set = set(os.path.abspath(p) for p in parent_rollout)
            for i, ref in enumerate(self._refs):
                if os.path.abspath(ref.file_path) in fail_set:
                    is_fail_raw[i] = True

        # Scan explicit fail_rollout_paths as a separate population.
        if self._fail_files_for_init:
            self._scan_fail_rollout(self._fail_files_for_init, is_fail_raw)

        self._is_fail_raw = torch.tensor(is_fail_raw, dtype=torch.bool)
        self.gamma_buffer = torch.full(
            (len(self._refs),), 0.5, dtype=torch.float32
        )

    # ------------------------------------------------------------------ #
    # Scanning                                                           #
    # ------------------------------------------------------------------ #

    def _scan_fail_rollout(
        self,
        files: Sequence[str],
        is_fail_raw: List[bool],
    ) -> None:
        # max_traj is per-task; a single global counter would drain on the first
        # alphabetical task's files before any other task is scanned.
        max_traj = self._max_traj_per_kind
        task_counts: dict[str, int] = {}
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
                                    is_expert=False,
                                )
                            )
                            self._sample_is_expert.append(False)
                            is_fail_raw.append(True)
                        task_counts[task_name] = task_counts.get(task_name, 0) + 1
                        total += 1
            except OSError as exc:
                print(f"[d4_disc][dataset] skip {fp}: {exc}")
        # Fail rollouts count as "rollout" trajectories for book-keeping.
        self._num_rollout_traj += total

    # ------------------------------------------------------------------ #
    # Gamma buffer interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def is_fail_raw(self) -> torch.Tensor:
        return self._is_fail_raw

    @property
    def num_fail_raw_samples(self) -> int:
        return int(self._is_fail_raw.sum().item())

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

    # ------------------------------------------------------------------ #
    # Sampling                                                           #
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        item["sample_idx"] = torch.tensor(int(index), dtype=torch.long)
        item["is_fail_raw"] = torch.tensor(bool(self._is_fail_raw[index].item()), dtype=torch.bool)
        item["gamma"] = torch.tensor(float(self.gamma_buffer[index].item()), dtype=torch.float32)
        return item
