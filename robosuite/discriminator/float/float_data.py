from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import numpy as np

from .float_core import Trajectory

try:  # pragma: no cover - optional dependency guard
    import h5py
except Exception:  # pragma: no cover
    h5py = None


@dataclass(frozen=True)
class TrajectoryRef:
    """Reference to one trajectory inside an HDF5 dataset."""

    file_path: str
    demo_key: str


@dataclass
class PolicyTrajectory:
    """Trajectory container for policy latent extraction."""

    states: np.ndarray
    images: np.ndarray
    actions: Optional[np.ndarray] = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states)
        self.images = np.asarray(self.images)
        if self.states.ndim != 2:
            raise ValueError(f"PolicyTrajectory.states must be (T, D), got shape={self.states.shape}")
        if self.images.ndim != 4:
            raise ValueError(f"PolicyTrajectory.images must be (T, H, W, C), got shape={self.images.shape}")
        if self.images.shape[-1] != 3:
            raise ValueError(f"PolicyTrajectory.images must have 3 channels, got shape={self.images.shape}")
        if self.states.shape[0] != self.images.shape[0]:
            raise ValueError(
                "states and images length mismatch: "
                f"states={self.states.shape[0]} images={self.images.shape[0]}"
            )
        if self.actions is not None:
            self.actions = np.asarray(self.actions)
            if self.actions.ndim != 2:
                raise ValueError(f"PolicyTrajectory.actions must be (T, A), got shape={self.actions.shape}")
            if self.actions.shape[0] != self.states.shape[0]:
                raise ValueError(
                    "actions length mismatch: "
                    f"actions={self.actions.shape[0]} states={self.states.shape[0]}"
                )


def require_h5py() -> None:
    if h5py is None:  # pragma: no cover
        raise ImportError("h5py is required for loading HDF5 trajectories. Install with `pip install h5py`.")


def list_hdf5_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in directory: {data_dir}")
    return files


def scan_hdf5_trajectories(data_dir: str, max_trajectories: Optional[int] = None) -> list[TrajectoryRef]:
    require_h5py()
    refs: list[TrajectoryRef] = []

    for fp in list_hdf5_files(data_dir):
        with h5py.File(fp, "r") as f:
            if "demos" not in f:
                continue
            for demo_key in sorted(f["demos"].keys()):
                refs.append(TrajectoryRef(file_path=fp, demo_key=demo_key))
                if max_trajectories is not None and len(refs) >= int(max_trajectories):
                    return refs

    if not refs:
        raise RuntimeError(f"No trajectories found under HDF5 demos groups in: {data_dir}")
    return refs


def _read_obs(demo, obs_key: str, camera_name: Optional[str]) -> np.ndarray:
    if obs_key == "states":
        if "states" not in demo:
            raise KeyError("states dataset is missing in demo")
        return np.asarray(demo["states"][:])

    if obs_key == "actions":
        if "actions" not in demo:
            raise KeyError("actions dataset is missing in demo")
        return np.asarray(demo["actions"][:])

    if obs_key == "images":
        if "observations" not in demo:
            raise KeyError("observations group is missing in demo")
        if camera_name is None:
            raise ValueError("camera_name must be provided when obs_key='images'")
        if camera_name not in demo["observations"]:
            raise KeyError(f"camera '{camera_name}' is missing in demo['observations']")
        cam = demo["observations"][camera_name]
        if "images" not in cam:
            raise KeyError(f"images dataset missing in observations/{camera_name}")
        return np.asarray(cam["images"][:])

    raise ValueError(f"Unsupported obs_key='{obs_key}'. Supported: states, actions, images")


def load_trajectory_from_ref(
    ref: TrajectoryRef,
    obs_key: str = "states",
    camera_name: Optional[str] = None,
) -> Trajectory:
    require_h5py()
    with h5py.File(ref.file_path, "r") as f:
        demo = f["demos"][ref.demo_key]

        obs = _read_obs(demo=demo, obs_key=obs_key, camera_name=camera_name)
        actions = np.asarray(demo["actions"][:]) if "actions" in demo else None

        t = int(obs.shape[0])
        if actions is not None:
            t = min(t, int(actions.shape[0]))
        if t <= 0:
            raise ValueError(f"Trajectory {ref.demo_key} in {ref.file_path} has zero valid timesteps")

        obs = np.asarray(obs[:t])
        actions = np.asarray(actions[:t]) if actions is not None else None

        meta = {
            "file_path": ref.file_path,
            "demo_key": ref.demo_key,
            "successful": bool(demo.attrs.get("successful", False)),
            "length": int(t),
        }

    return Trajectory(obs=obs, actions=actions, meta=meta)


def load_trajectories(
    data_dir: str,
    obs_key: str = "states",
    camera_name: Optional[str] = None,
    max_trajectories: Optional[int] = None,
) -> list[Trajectory]:
    refs = scan_hdf5_trajectories(data_dir=data_dir, max_trajectories=max_trajectories)
    return [load_trajectory_from_ref(ref=r, obs_key=obs_key, camera_name=camera_name) for r in refs]


def load_policy_trajectory_from_ref(ref: TrajectoryRef, camera_name: str) -> PolicyTrajectory:
    """
    Load one trajectory containing state + image observations for policy-latent encoding.
    """
    require_h5py()
    with h5py.File(ref.file_path, "r") as f:
        demo = f["demos"][ref.demo_key]
        if "states" not in demo:
            raise KeyError("states dataset is missing in demo")
        if "actions" not in demo:
            raise KeyError("actions dataset is missing in demo")
        if "observations" not in demo:
            raise KeyError("observations group is missing in demo")
        if camera_name not in demo["observations"]:
            raise KeyError(f"camera '{camera_name}' is missing in demo observations")
        if "images" not in demo["observations"][camera_name]:
            raise KeyError(f"images dataset missing under observations/{camera_name}")

        states = np.asarray(demo["states"][:], dtype=np.float32)
        actions = np.asarray(demo["actions"][:], dtype=np.float32)
        images = np.asarray(demo["observations"][camera_name]["images"][:], dtype=np.uint8)

        t = min(int(states.shape[0]), int(actions.shape[0]), int(images.shape[0]))
        if t <= 0:
            raise ValueError(f"Trajectory {ref.demo_key} in {ref.file_path} has zero valid timesteps")

        meta = {
            "file_path": ref.file_path,
            "demo_key": ref.demo_key,
            "successful": bool(demo.attrs.get("successful", False)),
            "length": int(t),
            "camera_name": camera_name,
        }

    return PolicyTrajectory(
        states=states[:t],
        images=images[:t],
        actions=actions[:t],
        meta=meta,
    )


def load_policy_trajectories(
    data_dir: str,
    camera_name: str,
    max_trajectories: Optional[int] = None,
) -> list[PolicyTrajectory]:
    refs = scan_hdf5_trajectories(data_dir=data_dir, max_trajectories=max_trajectories)
    trajectories: list[PolicyTrajectory] = []
    for ref in tqdm(refs, desc="loading trajectories"):
        try:
            trajectories.append(load_policy_trajectory_from_ref(ref=ref, camera_name=camera_name))
        except Exception:
            continue
    if not trajectories:
        raise RuntimeError(
            f"No valid policy trajectories found in {data_dir} for camera '{camera_name}'. "
            "Expected states/actions/observations/<camera>/images in each demo."
        )
    return trajectories


def split_train_val(
    items: list,
    val_ratio: float,
    seed: int,
) -> tuple[list, list]:
    if not items:
        return [], []

    ratio = float(val_ratio)
    if ratio < 0 or ratio >= 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(items))
    rng.shuffle(idx)

    n_val = int(round(len(items) * ratio))
    if len(items) >= 2:
        n_val = min(max(n_val, 1), len(items) - 1)
    else:
        n_val = 0

    val_ids = set(idx[:n_val].tolist())
    train = [x for i, x in enumerate(items) if i not in val_ids]
    val = [x for i, x in enumerate(items) if i in val_ids]
    return train, val


def fail_prefix_labels(length: int, fail_tail_ratio: float = 0.2) -> np.ndarray:
    """
    Create prefix labels for one fail trajectory.

    Label convention:
    - 0: success-like prefix
    - 1: fail prefix

    The last `fail_tail_ratio` portion is labeled as fail.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if fail_tail_ratio <= 0 or fail_tail_ratio >= 1:
        raise ValueError(f"fail_tail_ratio must be in (0, 1), got {fail_tail_ratio}")

    labels = np.zeros(length, dtype=np.int64)
    fail_start = int(np.floor((1.0 - float(fail_tail_ratio)) * length))
    fail_start = int(np.clip(fail_start, 0, length - 1))
    labels[fail_start:] = 1
    return labels
