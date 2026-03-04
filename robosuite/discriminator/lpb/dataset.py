from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise ImportError("LatentDynamicsDataset requires h5py (`pip install h5py`).") from exc


@dataclass(frozen=True)
class _TransitionRef:
    file_path: str
    demo_key: str
    t: int
    horizon: int
    is_expert: bool


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


class LatentDynamicsDataset(Dataset):
    """
    Build transitions (O_t, A_t, O_{t+h}) from expert + rollout HDF5 data.

    Expected HDF5 layout per trajectory:
      demos/<demo_key>/states                        (T, state_dim)
      demos/<demo_key>/actions                       (T, action_dim)
      demos/<demo_key>/observations/<camera>/images  (T, H, W, 3)
    """

    def __init__(
        self,
        expert_paths: Optional[Sequence[str]] = None,
        rollout_paths: Optional[Sequence[str]] = None,
        camera_name: str = "agentview",
        horizon: int = 1,
        proprio_indices: Optional[Sequence[int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        max_trajectories: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.camera_name = str(camera_name)
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self.proprio_indices = None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        if self.proprio_indices is not None and self.proprio_indices.size == 0:
            self.proprio_indices = None
        self.image_size = image_size

        expert_files = _expand_hdf5_inputs(expert_paths)
        rollout_files = _expand_hdf5_inputs(rollout_paths)
        if not expert_files and not rollout_files:
            raise FileNotFoundError("No expert/rollout HDF5 files found.")

        if max_trajectories is not None and max_trajectories > 0:
            max_traj = int(max_trajectories)
        else:
            max_traj = None

        self._refs: List[_TransitionRef] = []
        self._sample_is_expert: List[bool] = []
        self._proprio_dim: Optional[int] = None
        self._action_dim: Optional[int] = None
        self._image_shape: Optional[Tuple[int, int, int]] = None
        self._num_expert_traj = 0
        self._num_rollout_traj = 0

        self._scan_files(expert_files, is_expert=True, max_traj=max_traj)
        self._scan_files(rollout_files, is_expert=False, max_traj=max_traj)

        if not self._refs:
            raise RuntimeError("No valid transitions found in provided HDF5 files.")

    def _scan_files(self, files: Sequence[str], is_expert: bool, max_traj: Optional[int]) -> None:
        traj_count = 0
        for fp in files:
            with h5py.File(fp, "r") as f:
                if "demos" not in f:
                    continue
                for demo_key in sorted(f["demos"].keys()):
                    demo = f["demos"][demo_key]
                    if "states" not in demo or "actions" not in demo:
                        continue
                    if "observations" not in demo or self.camera_name not in demo["observations"]:
                        continue
                    cam = demo["observations"][self.camera_name]
                    if "images" not in cam:
                        continue

                    states = demo["states"]
                    actions = demo["actions"]
                    images = cam["images"]
                    length = int(min(states.shape[0], actions.shape[0], images.shape[0]))
                    if length <= self.horizon:
                        continue

                    if self._proprio_dim is None:
                        state_dim = int(states.shape[1])
                        if self.proprio_indices is not None:
                            if np.max(self.proprio_indices) >= state_dim:
                                raise ValueError(
                                    f"proprio_indices out of range for state_dim={state_dim}"
                                )
                            self._proprio_dim = int(self.proprio_indices.shape[0])
                        else:
                            self._proprio_dim = state_dim
                        self._action_dim = int(actions.shape[1])
                        self._image_shape = tuple(images.shape[1:])  # (H, W, C)
                    else:
                        state_dim = int(states.shape[1])
                        action_dim = int(actions.shape[1])
                        image_shape = tuple(images.shape[1:])
                        if self.proprio_indices is not None and np.max(self.proprio_indices) >= state_dim:
                            raise ValueError(
                                f"proprio_indices out of range in {fp}:{demo_key} for state_dim={state_dim}"
                            )
                        if action_dim != self._action_dim:
                            raise ValueError(
                                f"Action dim mismatch in {fp}:{demo_key}. "
                                f"expected={self._action_dim}, got={action_dim}"
                            )
                        if image_shape != self._image_shape:
                            raise ValueError(
                                f"Image shape mismatch in {fp}:{demo_key}. "
                                f"expected={self._image_shape}, got={image_shape}"
                            )

                    max_t = length - self.horizon
                    for t in range(max_t):
                        self._refs.append(
                            _TransitionRef(
                                file_path=fp,
                                demo_key=demo_key,
                                t=t,
                                horizon=self.horizon,
                                is_expert=is_expert,
                            )
                        )
                        self._sample_is_expert.append(is_expert)

                    traj_count += 1
                    if max_traj is not None and traj_count >= max_traj:
                        break
                if max_traj is not None and traj_count >= max_traj:
                    break
        if is_expert:
            self._num_expert_traj += traj_count
        else:
            self._num_rollout_traj += traj_count

    @property
    def action_dim(self) -> int:
        assert self._action_dim is not None
        return self._action_dim

    @property
    def proprio_dim(self) -> int:
        assert self._proprio_dim is not None
        return self._proprio_dim

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        assert self._image_shape is not None
        return self._image_shape

    @property
    def sample_is_expert(self) -> List[bool]:
        return self._sample_is_expert

    @property
    def num_expert_samples(self) -> int:
        return int(sum(self._sample_is_expert))

    @property
    def num_rollout_samples(self) -> int:
        return len(self._sample_is_expert) - self.num_expert_samples

    def __len__(self) -> int:
        return len(self._refs)

    def _maybe_slice_proprio(self, state: np.ndarray) -> np.ndarray:
        if self.proprio_indices is None:
            return state
        return state[self.proprio_indices]

    def _to_chw_tensor(self, image_hwc: np.ndarray) -> torch.Tensor:
        img = image_hwc.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        chw = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(chw)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self._refs[index]
        with h5py.File(ref.file_path, "r") as f:
            demo = f["demos"][ref.demo_key]
            states = demo["states"]
            actions = demo["actions"]
            images = demo["observations"][self.camera_name]["images"]

            t0 = ref.t
            th = t0 + ref.horizon

            cur_img = self._to_chw_tensor(images[t0])
            tgt_img = self._to_chw_tensor(images[th])

            cur_prop = np.asarray(states[t0], dtype=np.float32)
            tgt_prop = np.asarray(states[th], dtype=np.float32)
            cur_prop = self._maybe_slice_proprio(cur_prop)
            tgt_prop = self._maybe_slice_proprio(tgt_prop)

            act_seq = np.asarray(actions[t0:th], dtype=np.float32)  # (H, A)

        if self.image_size is not None:
            h, w = self.image_size
            cur_img = torch.nn.functional.interpolate(
                cur_img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            tgt_img = torch.nn.functional.interpolate(
                tgt_img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)

        return {
            "current_image": cur_img,
            "current_proprio": torch.from_numpy(cur_prop),
            "action_sequence": torch.from_numpy(act_seq),
            "target_image": tgt_img,
            "target_proprio": torch.from_numpy(tgt_prop),
            "is_expert": torch.tensor(1 if ref.is_expert else 0, dtype=torch.int64),
        }
