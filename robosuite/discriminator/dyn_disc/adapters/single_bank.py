"""Shared benchmark-adapter base (``DynBenchmarkDiscriminator``).

This is the encoder + trajectory-feature-cache backbone shared by every head in
this package. It owns:

  * a :class:`DynEncoder` (frozen DINOv3 dynamics model) built from ``model_ckpt``;
  * camera->view resolution, proprio/action slicing, image preprocessing;
  * a per-trajectory feature cache keyed by (file_path, demo/cache path,
    feature_source, transformer_layer).

It does **not** define a discriminator head of its own anymore. Concrete heads
subclass it and implement ``fit_on_benchmark`` / ``score_trajectory`` /
``calibration_summary`` (see ``adapters/pu_bce.py``).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput  # noqa: F401

from robosuite.discriminator.dyn_disc.detectors.single_bank_knn import DynEncoder


def _pad_to_length(values: np.ndarray, target_len: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    n = int(arr.shape[0])
    T = int(target_len)
    if n == T:
        return arr
    if n > T:
        return arr[:T].copy()
    if n == 0:
        return np.zeros((T,), dtype=dtype)
    pad = np.full((T - n,), arr[-1], dtype=dtype)
    return np.concatenate([arr, pad], axis=0)


class DynBenchmarkDiscriminator:
    """Encoder + feature-cache backbone exposed through the BenchmarkTrajectory API.

    Concrete heads subclass this and override ``fit_on_benchmark`` /
    ``score_trajectory`` / ``calibration_summary``.
    """

    name = "dyn_disc"

    def __init__(
        self,
        *,
        model_ckpt: str,
        device: str = "cuda",
        encode_batch_size: int = 32,
        proprio_indices: Optional[Sequence[int]] = None,
        camera_to_view: Optional[Dict[str, str]] = None,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        action_weight: float = 1.0,
        delta: float = 10.0,
        knn_chunk_size: int = 2048,
        feature_source: str = "encoder",
        transformer_layer: int = -1,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")

        self.model_ckpt = str(model_ckpt)
        self.device = str(device)
        self.encode_batch_size = int(encode_batch_size)
        self.proprio_indices = None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        self.camera_to_view = dict(camera_to_view) if camera_to_view else None
        self.visual_weight = float(visual_weight)
        self.proprio_weight = float(proprio_weight)
        self.action_weight = float(action_weight)
        self.delta = float(delta)
        self.knn_chunk_size = int(knn_chunk_size)
        self.feature_source = str(feature_source)
        self.transformer_layer = int(transformer_layer)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.verbose_fit = bool(verbose_fit)

        self.encoder = DynEncoder(
            model_ckpt=self.model_ckpt,
            device=self.device,
            feature_source=self.feature_source,
            transformer_layer=self.transformer_layer,
        )
        cfg_map = getattr(self.encoder.cfg, "proprio_map", None)
        self.proprio_map = (
            OmegaConf.to_container(cfg_map, resolve=True) if cfg_map is not None else {}
        ) or {}

        # If user did not provide a camera->view map, default to a 1:1 identity
        # over the encoder's view_names (assume cameras are named the same).
        if self.camera_to_view is None:
            self.camera_to_view = {v: v for v in self.encoder.view_names}

        # Per-task state (populated by concrete subclasses).
        self._detectors_per_task: Dict[str, Any] = {}
        self._calibration_stats: Dict[str, dict] = {}
        # Cache encoded trajectories: key -> (T, D) tensor.
        self._feature_cache: Dict[tuple, torch.Tensor] = {}
        # Optional list populated with per-call encode_batch() latencies (seconds).
        self.batch_infer_times: Optional[List[float]] = None

    # ------------------------------------------------------------------ #
    # Encoding helpers                                                   #
    # ------------------------------------------------------------------ #

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str, str, str]:
        group_key = getattr(trajectory, "demo_path", None)
        if group_key is None:
            group_key = getattr(trajectory, "episode_path", "")
        cache_key = getattr(trajectory, "cache_npz_path", "")
        return (
            str(getattr(trajectory, "file_path", "")),
            str(cache_key or group_key),
            self.feature_source,
            str(self.transformer_layer),
        )

    def _resolve_camera(self, view: str) -> str:
        # Reverse-lookup camera name from view_name (we stored cam->view map).
        for cam, vname in self.camera_to_view.items():
            if vname == view:
                return cam
        raise KeyError(
            f"No camera mapped to view {view!r}. Camera->view map: {self.camera_to_view}"
        )

    def _slice_proprio(
        self,
        states: np.ndarray,
        target_dim: Optional[int],
        task_name: Optional[str] = None,
    ) -> np.ndarray:
        if self.proprio_indices is not None:
            states = states[:, self.proprio_indices]
        elif task_name and self.proprio_map:
            task_cfg = self.proprio_map.get(str(task_name), {})
            indices = task_cfg.get("indices") if isinstance(task_cfg, dict) else None
            if indices:
                states = states[:, np.asarray(indices, dtype=np.int64)]
        if target_dim is None:
            return states.astype(np.float32, copy=False)
        d = int(states.shape[1])
        if d == target_dim:
            return states.astype(np.float32, copy=False)
        if d > target_dim:
            return states[:, :target_dim].astype(np.float32, copy=False)
        pad = np.zeros((states.shape[0], target_dim - d), dtype=np.float32)
        return np.concatenate([states.astype(np.float32, copy=False), pad], axis=1)

    def _slice_action(self, actions: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
        if target_dim is None:
            return actions.astype(np.float32, copy=False)
        d = int(actions.shape[1])
        if d == target_dim:
            return actions.astype(np.float32, copy=False)
        if d > target_dim:
            return actions[:, :target_dim].astype(np.float32, copy=False)
        pad = np.zeros((actions.shape[0], target_dim - d), dtype=np.float32)
        return np.concatenate([actions.astype(np.float32, copy=False), pad], axis=1)

    def _prepare_trajectory_tensors(self, trajectory: BenchmarkTrajectory) -> Dict[str, Any]:
        view_names = self.encoder.view_names
        cameras_needed = [self._resolve_camera(v) for v in view_names]
        images_by_cam = trajectory.load_images(cameras=cameras_needed)
        states = np.asarray(trajectory.load_states(), dtype=np.float32)
        actions = np.asarray(trajectory.load_actions(), dtype=np.float32)

        cam_lens = [int(images_by_cam[c].shape[0]) for c in cameras_needed]
        t_len = int(min([states.shape[0], actions.shape[0]] + cam_lens))
        if t_len <= 0:
            raise ValueError(f"Empty trajectory: {trajectory.describe()}")

        target_proprio_dim = None
        try:
            target_proprio_dim = int(self.encoder.model.proprio_encoder.in_chans)
        except Exception:
            target_proprio_dim = None

        prop = self._slice_proprio(
            states[:t_len],
            target_dim=target_proprio_dim,
            task_name=str(trajectory.task_name),
        )
        if self.feature_source == "encoder":
            act = self.encoder.prepare_actions(actions[:t_len], t_len=t_len)
        else:
            target_action_dim = None
            try:
                target_action_dim = int(self.encoder.model.action_encoder.in_chans)
            except Exception:
                target_action_dim = None
            act = self._slice_action(actions[:t_len], target_dim=target_action_dim)

        h = self.encoder.original_img_size
        per_view_chw: Dict[str, np.ndarray] = {}
        for view in view_names:
            cam = self._resolve_camera(view)
            imgs = np.asarray(images_by_cam[cam][:t_len])
            if imgs.shape[1] != h or imgs.shape[2] != h:
                t_imgs = torch.from_numpy(imgs.astype(np.float32))
                if t_imgs.max() > 1.5:
                    t_imgs = t_imgs / 255.0
                t_imgs = t_imgs.permute(0, 3, 1, 2)
                t_imgs = torch.nn.functional.interpolate(
                    t_imgs, size=(h, h), mode="bilinear", align_corners=False
                )
                per_view_chw[view] = t_imgs.numpy()
            else:
                arr = imgs.astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                per_view_chw[view] = np.transpose(arr, (0, 3, 1, 2))

        return {
            "view_names": view_names,
            "per_view_chw": per_view_chw,
            "prop": prop,
            "act": act,
            "t_len": t_len,
        }

    @staticmethod
    def _truncate_prepared(prepared: Dict[str, Any], frame_end: int) -> Dict[str, Any]:
        t_end = min(int(frame_end), int(prepared["t_len"]))
        if t_end <= 0:
            raise ValueError(f"frame_end must be positive, got {t_end}")
        return {
            "view_names": prepared["view_names"],
            "per_view_chw": {k: v[:t_end] for k, v in prepared["per_view_chw"].items()},
            "prop": prepared["prop"][:t_end],
            "act": prepared["act"][:t_end],
            "t_len": t_end,
        }

    def _encode_tensors(self, key: tuple, prepared: Dict[str, Any]) -> torch.Tensor:
        view_names: List[str] = prepared["view_names"]
        per_view_chw: Dict[str, np.ndarray] = prepared["per_view_chw"]
        prop: np.ndarray = prepared["prop"]
        act: np.ndarray = prepared["act"]
        t_len: int = int(prepared["t_len"])

        feats: List[torch.Tensor] = []
        bs = self.encode_batch_size
        for start in range(0, t_len, bs):
            end = min(start + bs, t_len)
            batch_imgs = {
                v: torch.from_numpy(per_view_chw[v][start:end]) for v in view_names
            }
            batch_prop = torch.from_numpy(prop[start:end])
            batch_act = torch.from_numpy(act[start:end])
            if self.batch_infer_times is not None:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            f = self.encoder.encode_batch(batch_imgs, batch_prop, actions=batch_act)
            if self.batch_infer_times is not None:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.batch_infer_times.append(float(time.perf_counter() - t0))
            feats.append(f.detach().cpu())

        out = torch.cat(feats, dim=0)
        self._feature_cache[key] = out
        return out

    def preload_trajectory(self, trajectory: BenchmarkTrajectory) -> Dict[str, Any]:
        return self._prepare_trajectory_tensors(trajectory)

    @torch.no_grad()
    def encode_preloaded(
        self,
        trajectory: BenchmarkTrajectory,
        prepared: Dict[str, Any],
    ) -> torch.Tensor:
        key = self._trajectory_key(trajectory)
        cached = self._feature_cache.get(key, None)
        if cached is not None:
            return cached
        return self._encode_tensors(key, prepared)

    @torch.no_grad()
    def _encode(
        self,
        trajectory: BenchmarkTrajectory,
        *,
        frame_end: Optional[int] = None,
    ) -> torch.Tensor:
        key = self._trajectory_key(trajectory)
        if frame_end is not None:
            key = key + (f"end{int(frame_end)}",)
        cached = self._feature_cache.get(key, None)
        if cached is not None:
            return cached

        prepared = self._prepare_trajectory_tensors(trajectory)
        if frame_end is not None:
            prepared = self._truncate_prepared(prepared, frame_end)
        return self._encode_tensors(key, prepared)

    @torch.no_grad()
    def _encode_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        desc: str = "encode",
    ) -> List[torch.Tensor]:
        """Encode trajectories sequentially; show a tqdm bar when verbose_fit."""
        if not trajectories:
            return []

        iterable: Iterable[BenchmarkTrajectory] = trajectories
        if self.verbose_fit:
            try:
                from tqdm import tqdm

                iterable = tqdm(
                    trajectories,
                    desc=desc,
                    unit="traj",
                    dynamic_ncols=True,
                )
            except ImportError:
                pass

        out: List[torch.Tensor] = []
        for traj in iterable:
            out.append(self._encode(traj))
        return out

    # ------------------------------------------------------------------ #
    # Public API (overridden by concrete heads)                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        raise NotImplementedError(
            "DynBenchmarkDiscriminator is an encoder/cache backbone; subclass it "
            "and implement fit_on_benchmark (see adapters/pu_bce.py)."
        )

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        raise NotImplementedError(
            "DynBenchmarkDiscriminator is an encoder/cache backbone; subclass it "
            "and implement score_trajectory (see adapters/pu_bce.py)."
        )

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "model_ckpt": self.model_ckpt,
            "view_names": list(self.encoder.view_names),
            "camera_to_view": dict(self.camera_to_view),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "delta": float(self.delta),
            "calib_fraction": float(self.calib_fraction),
            "encode_batch_size": int(self.encode_batch_size),
        }

    def close(self) -> None:
        try:
            self.encoder.model.to("cpu")
        except Exception:
            pass
