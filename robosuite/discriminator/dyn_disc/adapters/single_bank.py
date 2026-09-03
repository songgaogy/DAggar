"""Shared benchmark-adapter base (``DynBenchmarkDiscriminator``).

This is the encoder + trajectory-feature-cache backbone shared by every head in
this package. It owns:

  * a :class:`DynEncoder` (frozen DINOv3 TACO model) built from ``model_ckpt``;
  * camera->view resolution, proprio/action slicing, image preprocessing;
  * a per-trajectory TACO encoder-feature cache.

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
from torch.utils.data import DataLoader, Dataset

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput  # noqa: F401

from robosuite.discriminator.dyn_disc.detectors.single_bank_knn import DynEncoder


def _load_trajectory_inputs(
    trajectory: BenchmarkTrajectory,
    cameras: Sequence[str],
    frame_end: Optional[int],
    action_lookahead: int,
) -> Dict[str, Any]:
    if frame_end is not None and int(frame_end) <= 0:
        raise ValueError(f"frame_end must be positive, got {frame_end}")
    action_frame_end = (
        None
        if frame_end is None
        else min(int(trajectory.num_frames), int(frame_end) + int(action_lookahead))
    )
    combined_loader = getattr(trajectory, "load_model_inputs", None)
    if combined_loader is not None:
        return combined_loader(
            cameras,
            frame_end=frame_end,
            action_frame_end=action_frame_end,
        )

    images = trajectory.load_images(cameras=cameras)
    states = trajectory.load_states()
    actions = trajectory.load_actions()
    if frame_end is not None:
        end = int(frame_end)
        images = {camera: values[:end] for camera, values in images.items()}
        states = states[:end]
    if action_frame_end is not None:
        actions = actions[:action_frame_end]
    return {"images": images, "states": states, "actions": actions}


class _TrajectoryInputsDataset(Dataset):
    """Load raw trajectory arrays in spawned DataLoader workers."""

    def __init__(
        self,
        requests: Sequence[tuple[int, BenchmarkTrajectory, Optional[int]]],
        cameras: Sequence[str],
        action_lookahead: int,
    ) -> None:
        self.requests = list(requests)
        self.cameras = tuple(cameras)
        self.action_lookahead = int(action_lookahead)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        output_index, trajectory, frame_end = self.requests[index]
        loaded = _load_trajectory_inputs(
            trajectory,
            self.cameras,
            frame_end,
            self.action_lookahead,
        )
        loaded["output_index"] = int(output_index)
        loaded["frame_end"] = frame_end
        return loaded


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
        preload_workers: int = 0,
        prefetch_factor: int = 1,
        pin_memory: bool = True,
        proprio_indices: Optional[Sequence[int]] = None,
        camera_to_view: Optional[Dict[str, str]] = None,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        action_weight: float = 1.0,
        delta: float = 10.0,
        knn_chunk_size: int = 2048,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")

        self.model_ckpt = str(model_ckpt)
        self.device = str(device)
        self.encode_batch_size = int(encode_batch_size)
        self.preload_workers = int(preload_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory)
        if self.preload_workers < 0:
            raise ValueError(f"preload_workers must be >= 0, got {self.preload_workers}")
        if self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be > 0, got {self.prefetch_factor}")
        self.proprio_indices = None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        self.camera_to_view = dict(camera_to_view) if camera_to_view else None
        self.visual_weight = float(visual_weight)
        self.proprio_weight = float(proprio_weight)
        self.action_weight = float(action_weight)
        self.delta = float(delta)
        self.knn_chunk_size = int(knn_chunk_size)
        self.feature_source = "encoder"
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.verbose_fit = bool(verbose_fit)

        self.encoder = DynEncoder(
            model_ckpt=self.model_ckpt,
            device=self.device,
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

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str, str]:
        group_key = getattr(trajectory, "demo_path", None)
        if group_key is None:
            group_key = getattr(trajectory, "episode_path", "")
        cache_key = getattr(trajectory, "cache_npz_path", "")
        return (
            str(getattr(trajectory, "file_path", "")),
            str(cache_key or group_key),
            self.feature_source,
        )

    def _trajectory_frame_key(
        self,
        trajectory: BenchmarkTrajectory,
        frame_end: Optional[int],
    ) -> tuple:
        key: tuple = self._trajectory_key(trajectory)
        if frame_end is not None:
            key = key + (f"end{int(frame_end)}",)
        return key

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

    def _prepare_loaded_inputs(
        self,
        trajectory: BenchmarkTrajectory,
        loaded: Dict[str, Any],
        frame_end: Optional[int],
    ) -> Dict[str, Any]:
        view_names = self.encoder.view_names
        cameras_needed = [self._resolve_camera(v) for v in view_names]
        images_by_cam = loaded["images"]
        states_raw = loaded["states"]
        actions_raw = loaded["actions"]
        states = np.asarray(
            states_raw.numpy() if torch.is_tensor(states_raw) else states_raw,
            dtype=np.float32,
        )
        actions = np.asarray(
            actions_raw.numpy() if torch.is_tensor(actions_raw) else actions_raw,
            dtype=np.float32,
        )

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
        action_windows = self.encoder.prepare_actions(actions, t_len=int(actions.shape[0]))
        act = action_windows[:t_len]

        per_view_images: Dict[str, torch.Tensor] = {}
        for view in view_names:
            cam = self._resolve_camera(view)
            imgs = images_by_cam[cam][:t_len]
            per_view_images[view] = (
                imgs if torch.is_tensor(imgs) else torch.from_numpy(np.asarray(imgs))
            )

        return {
            "view_names": view_names,
            "per_view_images": per_view_images,
            "prop": prop,
            "act": act,
            "t_len": t_len,
            "frame_end": frame_end,
        }

    def _prepare_trajectory_tensors(
        self,
        trajectory: BenchmarkTrajectory,
        *,
        frame_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        cameras_needed = [self._resolve_camera(v) for v in self.encoder.view_names]
        loaded = _load_trajectory_inputs(
            trajectory,
            cameras_needed,
            frame_end,
            max(0, int(self.encoder.frameskip) - 1),
        )
        return self._prepare_loaded_inputs(trajectory, loaded, frame_end)

    @staticmethod
    def _truncate_prepared(prepared: Dict[str, Any], frame_end: int) -> Dict[str, Any]:
        t_end = min(int(frame_end), int(prepared["t_len"]))
        if t_end <= 0:
            raise ValueError(f"frame_end must be positive, got {t_end}")
        return {
            "view_names": prepared["view_names"],
            "per_view_images": {
                k: v[:t_end] for k, v in prepared["per_view_images"].items()
            },
            "prop": prepared["prop"][:t_end],
            "act": prepared["act"][:t_end],
            "t_len": t_end,
            "frame_end": t_end,
        }

    def _encode_tensors(self, key: tuple, prepared: Dict[str, Any]) -> torch.Tensor:
        view_names: List[str] = prepared["view_names"]
        per_view_images: Dict[str, torch.Tensor] = prepared["per_view_images"]
        prop: np.ndarray = prepared["prop"]
        act: np.ndarray = prepared["act"]
        t_len: int = int(prepared["t_len"])

        feats: List[torch.Tensor] = []
        infer_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        bs = self.encode_batch_size
        for start in range(0, t_len, bs):
            end = min(start + bs, t_len)
            batch_imgs = {v: per_view_images[v][start:end] for v in view_names}
            batch_prop = torch.from_numpy(prop[start:end])
            batch_act = torch.from_numpy(act[start:end])
            if self.batch_infer_times is not None:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    t0 = time.perf_counter()
            f = self.encoder.encode_batch(batch_imgs, batch_prop, actions=batch_act)
            if self.batch_infer_times is not None:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    end_event.record()
                    infer_events.append((start_event, end_event))
                else:
                    self.batch_infer_times.append(float(time.perf_counter() - t0))
            feats.append(f.detach())

        out = torch.cat(feats, dim=0).cpu()
        if self.batch_infer_times is not None:
            self.batch_infer_times.extend(
                float(start.elapsed_time(end)) / 1000.0
                for start, end in infer_events
            )
        self._feature_cache[key] = out
        return out

    def preload_trajectory(
        self,
        trajectory: BenchmarkTrajectory,
        *,
        frame_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._prepare_trajectory_tensors(trajectory, frame_end=frame_end)

    @torch.no_grad()
    def encode_preloaded(
        self,
        trajectory: BenchmarkTrajectory,
        prepared: Dict[str, Any],
    ) -> torch.Tensor:
        key = self._trajectory_frame_key(trajectory, prepared.get("frame_end"))
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
        key = self._trajectory_frame_key(trajectory, frame_end)
        cached = self._feature_cache.get(key, None)
        if cached is not None:
            return cached

        prepared = self._prepare_trajectory_tensors(trajectory, frame_end=frame_end)
        return self._encode_tensors(key, prepared)

    @torch.no_grad()
    def _encode_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        desc: str = "encode",
        frame_ends: Optional[Sequence[Optional[int]]] = None,
    ) -> List[torch.Tensor]:
        """Encode trajectories in order while workers preload raw HDF5 arrays."""
        if not trajectories:
            return []
        if frame_ends is None:
            frame_ends = [None] * len(trajectories)
        if len(frame_ends) != len(trajectories):
            raise ValueError("frame_ends must match trajectories length")

        outputs: List[Optional[torch.Tensor]] = [None] * len(trajectories)
        requests: List[tuple[int, BenchmarkTrajectory, Optional[int]]] = []
        for index, (trajectory, frame_end) in enumerate(zip(trajectories, frame_ends)):
            cached = self._feature_cache.get(
                self._trajectory_frame_key(trajectory, frame_end)
            )
            if cached is None:
                requests.append((index, trajectory, frame_end))
            else:
                outputs[index] = cached

        if not requests:
            return [value for value in outputs if value is not None]

        cameras_needed = [self._resolve_camera(v) for v in self.encoder.view_names]
        dataset = _TrajectoryInputsDataset(
            requests,
            cameras_needed,
            action_lookahead=max(0, int(self.encoder.frameskip) - 1),
        )
        loader_kwargs: Dict[str, Any] = {
            "batch_size": None,
            "num_workers": self.preload_workers,
            "pin_memory": self.pin_memory and self.device.startswith("cuda"),
        }
        if self.preload_workers > 0:
            loader_kwargs.update(
                multiprocessing_context="spawn",
                prefetch_factor=self.prefetch_factor,
            )
        loader = DataLoader(dataset, **loader_kwargs)

        iterable: Iterable[Dict[str, Any]] = loader
        progress = None
        if self.verbose_fit:
            try:
                from tqdm import tqdm

                progress = tqdm(
                    total=len(trajectories),
                    desc=desc,
                    unit="traj",
                    dynamic_ncols=True,
                )
                progress.update(len(trajectories) - len(requests))
            except ImportError:
                pass

        try:
            for loaded in iterable:
                output_index = int(loaded.pop("output_index"))
                frame_end = loaded.pop("frame_end")
                if torch.is_tensor(frame_end):
                    frame_end = int(frame_end.item())
                trajectory = trajectories[output_index]
                prepared = self._prepare_loaded_inputs(trajectory, loaded, frame_end)
                key = self._trajectory_frame_key(trajectory, frame_end)
                outputs[output_index] = self._encode_tensors(key, prepared)
                if progress is not None:
                    progress.update()
        finally:
            if progress is not None:
                progress.close()

        if any(value is None for value in outputs):
            raise RuntimeError("Trajectory preloader did not produce every requested item")
        return [value for value in outputs if value is not None]

    def preencode_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        success_prefix: bool = False,
        desc: str = "encode",
    ) -> List[torch.Tensor]:
        frame_ends: List[Optional[int]] = []
        for trajectory in trajectories:
            if success_prefix and not bool(trajectory.is_failure):
                prefix_fn = getattr(trajectory, "prefix_frames_before_done", None)
                frame_ends.append(
                    int(prefix_fn()) if prefix_fn is not None else int(trajectory.num_frames)
                )
            else:
                frame_ends.append(None)
        return self._encode_trajectories(
            trajectories,
            desc=desc,
            frame_ends=frame_ends,
        )

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
            "delta": float(self.delta),
            "calib_fraction": float(self.calib_fraction),
            "encode_batch_size": int(self.encode_batch_size),
            "preload_workers": int(self.preload_workers),
            "prefetch_factor": int(self.prefetch_factor),
            "pin_memory": bool(self.pin_memory),
        }

    def close(self) -> None:
        try:
            self.encoder.model.to("cpu")
        except Exception:
            pass
