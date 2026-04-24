"""Preprocessed cache reader for D4-Disc.

This module provides a light-weight IO layer over an on-disk `.npz` cache
produced by the legacy LPB/D3 preprocessing pipeline. D4-Disc training and
inference consume only the cached tensors, not the original HDF5 demos.

Cache file schema (legacy `lpb_score_preprocessed_v2`):
    - `images_chw`: uint8 or float array of shape (T, V, 3, H, W)
        T: episode length, V: number of cameras/views. We select one view via
        `camera_index` and expose images as (T, 3, H, W).
    - `proprio`: float array of shape (T, P)
    - `actions`: float array of shape (T, A)

Notes:
    - This reader is intentionally conservative: it validates shapes, clamps
      image dtype/range to uint8, and memoizes decoded episodes in-memory to
      avoid repeated disk reads during dataset indexing.
    - Cache keys include a global token + task metadata + demo identity + source
      file mtime so stale cache entries are automatically invalidated.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


_LEGACY_CACHE_VERSION = "lpb_score_preprocessed_v2"
_LEGACY_CAMERA_NAMES: Tuple[str, ...] = (
    "agentview",
    "robot0_robotview",
    "robot0_eye_in_hand",
)
_LEGACY_PROP_MEAN_SHA1 = "d42f3c06cffef9a91d1bdbe7cca70e2329adba99"
_LEGACY_PROP_STD_SHA1 = "3620749c11ed8f06d3eef4d02ba27ccdd3b732ec"
_TASK_TO_CKPT: Dict[str, str] = {
    "PandaLift": "Lift",
    "Lift": "Lift",
    "PandaStack": "Stack",
    "Stack": "Stack",
    "PandaPickPlaceCan": "PickPlaceCan",
    "PickPlaceCan": "PickPlaceCan",
    "PickPlaceBread": "PickPlaceBread",
    "PickPlaceCereal": "PickPlaceCereal",
    "PickPlaceMilk": "PickPlaceMilk",
}
_LEGACY_TASK_METADATA_SHA1: Dict[str, str] = {
    "Lift": "bfb38787161ae540355ab5d20c5c7ba75c890dbe",
    "PickPlaceBread": "4e14627ee350b61316cf5c1c51b1f7ac012efa04",
    "PickPlaceCan": "0c72b4c72b613d81a3341472a265a47ed3e8c0c4",
    "PickPlaceCereal": "8fc84f3ce9cb3b857c62d155e3186018c522a6a3",
    "PickPlaceMilk": "4c549aa6b8357a28f3e370efb3db30ca197d8e56",
    "Stack": "939119cbc8db761963fa85539397129dd62b0767",
}


def _normalize_path_component(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _resolve_checkpoint_task_name(task_name: str) -> str:
    if task_name not in _TASK_TO_CKPT:
        raise KeyError(f"Unsupported task name: {task_name!r}")
    return _TASK_TO_CKPT[task_name]


@dataclass(frozen=True)
class PreprocessedDemo:
    """A single cached episode decoded from disk.

    Attributes:
        images_chw: (T, 3, H, W) uint8 images for the selected camera.
        proprio: (T, P) float32 proprioceptive state.
        actions: (T, A) float32 actions.
        length: Effective trajectory length after aligning modalities.
    """
    images_chw: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    length: int


class PreprocessedCacheReader:
    def __init__(
        self,
        cache_root: str = "data/.lpb_score_preprocessed_cache",
        *,
        image_size: int = 128,
        camera_index: int = 0,
        cache_key_fn: Callable[[str, str, str], str] | None = None,
    ) -> None:
        self.cache_root = str(Path(cache_root).resolve())
        self.image_size = int(image_size)
        self.camera_index = int(camera_index)
        self.cache_key_fn = cache_key_fn
        self.camera_names = tuple(_LEGACY_CAMERA_NAMES)
        self._global_token = "|".join(
            [
                _LEGACY_CACHE_VERSION,
                str(self.image_size),
                ",".join(self.camera_names),
                _LEGACY_PROP_MEAN_SHA1,
                _LEGACY_PROP_STD_SHA1,
            ]
        )
        # In-process memoization keyed by cache path. This is safe because cache
        # files are immutable for a fixed preprocessing token.
        self._memo: Dict[str, PreprocessedDemo] = {}

    def cache_key(self, task: str, file_path: str, demo_key: str) -> str:
        """Return a stable hash key for a demo's cached `.npz`.

        The key ties together:
            - preprocessing version + image size + camera list + normalization stats
            - task metadata fingerprint (legacy compat)
            - absolute source file path + demo key + source file mtime
        """
        if self.cache_key_fn is not None:
            return str(self.cache_key_fn(str(task), str(file_path), str(demo_key)))
        ckpt_task = _resolve_checkpoint_task_name(str(task))
        task_token = _LEGACY_TASK_METADATA_SHA1.get(ckpt_task, "missing_task_metadata")
        key = "|".join(
            [
                self._global_token,
                str(task),
                str(task_token),
                str(Path(file_path).resolve()),
                str(demo_key),
                str(os.path.getmtime(file_path)),
            ]
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def cache_path(self, task: str, file_path: str, demo_key: str) -> str:
        cache_dir = os.path.join(self.cache_root, _normalize_path_component(str(task)))
        return os.path.join(cache_dir, f"{self.cache_key(task, file_path, demo_key)}.npz")

    def exists(self, task: str, file_path: str, demo_key: str) -> bool:
        return os.path.isfile(self.cache_path(task, file_path, demo_key))

    def load(self, task: str, file_path: str, demo_key: str) -> PreprocessedDemo:
        """Load (and memoize) a cached episode for `(task, file_path, demo_key)`."""
        path = self.cache_path(task, file_path, demo_key)
        cached = self._memo.get(path)
        if cached is not None:
            return cached
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        # The `.npz` is expected to hold three arrays: images_chw, proprio, actions.
        with np.load(path) as data:
            images = np.asarray(data["images_chw"])
            proprio = np.asarray(data["proprio"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)

        if images.ndim != 5:
            raise ValueError(f"Expected cached images_chw shape (T,V,3,H,W), got {images.shape}")
        if not (0 <= self.camera_index < int(images.shape[1])):
            raise ValueError(
                f"camera_index={self.camera_index} out of range for cached cameras={images.shape[1]}"
            )

        # Select a single camera and ensure contiguous (T, 3, H, W).
        cam = np.ascontiguousarray(images[:, self.camera_index])
        if cam.dtype != np.uint8:
            if np.issubdtype(cam.dtype, np.floating):
                if float(np.nanmax(cam)) <= 1.5:
                    cam = np.clip(np.rint(cam * 255.0), 0.0, 255.0).astype(np.uint8)
                else:
                    cam = np.clip(np.rint(cam), 0.0, 255.0).astype(np.uint8)
            else:
                cam = cam.astype(np.uint8)

        # Modalities occasionally mismatch by 1 frame; use the safe aligned prefix.
        length = min(int(cam.shape[0]), int(proprio.shape[0]), int(actions.shape[0]))
        demo = PreprocessedDemo(
            images_chw=np.ascontiguousarray(cam[:length]),
            proprio=np.ascontiguousarray(proprio[:length]),
            actions=np.ascontiguousarray(actions[:length]),
            length=int(length),
        )
        self._memo[path] = demo
        return demo
