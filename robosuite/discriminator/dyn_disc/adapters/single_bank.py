"""Policy-encoder backbone shared by benchmark discriminator heads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput  # noqa: F401

from robosuite.discriminator.dyn_disc.detectors.policy_encoder import PolicyFeatureEncoder


def _pad_to_length(values: np.ndarray, target_len: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    n = int(arr.shape[0])
    target_len = int(target_len)
    if n == target_len:
        return arr
    if n > target_len:
        return arr[:target_len].copy()
    if n == 0:
        return np.zeros((target_len,), dtype=dtype)
    return np.concatenate(
        [arr, np.full((target_len - n,), arr[-1], dtype=dtype)], axis=0
    )


class PolicyBenchmarkDiscriminator:
    """Frozen policy encoder plus memory/disk trajectory feature caches."""

    name = "policy_disc"

    def __init__(
        self,
        *,
        policy_ckpt: str,
        device: str = "cuda",
        encode_batch_size: int = 128,
        preload_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        feature_cache_dir: Optional[str] = None,
        reuse_feature_cache: bool = True,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")
        if int(encode_batch_size) <= 0:
            raise ValueError("encode_batch_size must be positive")
        if int(preload_workers) < 0 or int(prefetch_factor) <= 0:
            raise ValueError("preload_workers must be >= 0 and prefetch_factor must be positive")

        self.policy_ckpt = str(Path(policy_ckpt).resolve())
        self.device = str(device)
        self.encode_batch_size = int(encode_batch_size)
        self.preload_workers = int(preload_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory)
        self.feature_cache_dir = (
            None if feature_cache_dir is None else Path(feature_cache_dir).resolve()
        )
        self.reuse_feature_cache = bool(reuse_feature_cache)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.verbose_fit = bool(verbose_fit)
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = PolicyFeatureEncoder(policy_ckpt=self.policy_ckpt, device=self.device)
        self.camera_names = list(self.encoder.camera_names)
        self.policy_ckpt_hash = self.encoder.checkpoint_sha256
        self.feature_dim = self.encoder.feature_dim
        self.feature_source = "policy_task_scene_cond"
        self.policy_weight_source = "ema_model"
        self._detectors_per_task: Dict[str, Any] = {}
        self._calibration_stats: Dict[str, dict] = {}
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self.batch_infer_times: Optional[List[float]] = None

    def prompt_for_task(self, task_name: str) -> str:
        return self.encoder.prompt_for_task(task_name)

    def feature_metadata(self) -> dict[str, Any]:
        return self.encoder.metadata()

    def _trajectory_identity(
        self,
        trajectory: BenchmarkTrajectory,
        frame_end: Optional[int],
    ) -> dict[str, Any]:
        file_path = Path(str(getattr(trajectory, "file_path", ""))).resolve()
        file_stat = file_path.stat() if file_path.is_file() else None
        group_key = getattr(trajectory, "demo_path", None)
        if group_key is None:
            group_key = getattr(trajectory, "episode_path", "")
        cache_path = getattr(trajectory, "cache_npz_path", "")
        task_name = str(trajectory.task_name)
        return {
            "policy_sha256": self.policy_ckpt_hash,
            "state_dict": "ema_model",
            "latent": self.encoder.latent_name,
            "task": task_name,
            "prompt": self.prompt_for_task(task_name),
            "camera_names": self.camera_names,
            "preprocess_version": self.encoder.preprocess_version,
            "file_path": str(file_path),
            "file_size": None if file_stat is None else int(file_stat.st_size),
            "file_mtime_ns": None if file_stat is None else int(file_stat.st_mtime_ns),
            "trajectory_path": str(cache_path or group_key),
            "success_prefix_end": None if frame_end is None else int(frame_end),
        }

    def _trajectory_key(
        self,
        trajectory: BenchmarkTrajectory,
        frame_end: Optional[int] = None,
    ) -> str:
        payload = json.dumps(
            self._trajectory_identity(trajectory, frame_end),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _disk_cache_path(self, key: str) -> Optional[Path]:
        if self.feature_cache_dir is None:
            return None
        return self.feature_cache_dir / key[:2] / f"{key}.npy"

    def _load_cached(self, key: str) -> Optional[torch.Tensor]:
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        path = self._disk_cache_path(key)
        if not self.reuse_feature_cache or path is None or not path.is_file():
            return None
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if array.ndim != 2 or array.shape[1] != self.feature_dim:
            return None
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        self._feature_cache[key] = tensor
        return tensor

    def _store_cached(self, key: str, features: torch.Tensor) -> None:
        features = features.detach().cpu().to(dtype=torch.float32).contiguous()
        self._feature_cache[key] = features
        path = self._disk_cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".npy", delete=False
            ) as handle:
                temp_path = Path(handle.name)
                np.save(handle, features.numpy(), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _prepare_trajectory_tensors(
        self,
        trajectory: BenchmarkTrajectory,
        frame_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        load_all = getattr(trajectory, "load_model_inputs", None)
        if load_all is not None:
            images_by_camera, states, actions = load_all(
                self.camera_names, frame_end=frame_end
            )
        else:
            images_by_camera = trajectory.load_images(cameras=self.camera_names)
            states = trajectory.load_states()
            actions = trajectory.load_actions()
        states = np.asarray(states)
        actions = np.asarray(actions)
        camera_lengths = [len(images_by_camera[name]) for name in self.camera_names]
        length = min([len(states), len(actions), *camera_lengths])
        if frame_end is not None:
            length = min(length, int(frame_end))
        if length <= 0:
            raise ValueError(f"Empty trajectory: {trajectory.describe()}")
        proprio = self.encoder.extract_proprio(
            states[:length], task_name=str(trajectory.task_name)
        )
        images = {
            name: np.asarray(images_by_camera[name][:length])
            for name in self.camera_names
        }
        return {
            "images": images,
            "proprio": proprio,
            "task_name": str(trajectory.task_name),
            "t_len": int(length),
        }

    def _to_host_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if self.pin_memory:
            tensor = tensor.pin_memory()
        return tensor

    def _encode_tensors(self, key: str, prepared: Dict[str, Any]) -> torch.Tensor:
        images_by_camera: Dict[str, np.ndarray] = prepared["images"]
        proprio: np.ndarray = prepared["proprio"]
        task_name = str(prepared["task_name"])
        length = int(prepared["t_len"])
        features: List[torch.Tensor] = []
        for start in range(0, length, self.encode_batch_size):
            end = min(start + self.encode_batch_size, length)
            images = np.stack(
                [images_by_camera[name][start:end] for name in self.camera_names], axis=1
            )
            image_tensor = self._to_host_tensor(images)
            proprio_tensor = self._to_host_tensor(proprio[start:end])
            if self.batch_infer_times is not None:
                torch.cuda.synchronize(self.encoder.device)
                started = time.perf_counter()
            encoded = self.encoder.encode_batch(image_tensor, proprio_tensor, task_name)
            if self.batch_infer_times is not None:
                torch.cuda.synchronize(self.encoder.device)
                self.batch_infer_times.append(time.perf_counter() - started)
            features.append(encoded.detach())
        output = torch.cat(features, dim=0).cpu()
        self._store_cached(key, output)
        return output

    def preload_trajectory(
        self,
        trajectory: BenchmarkTrajectory,
        *,
        frame_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._prepare_trajectory_tensors(trajectory, frame_end=frame_end)

    @torch.inference_mode()
    def encode_preloaded(
        self,
        trajectory: BenchmarkTrajectory,
        prepared: Dict[str, Any],
        *,
        frame_end: Optional[int] = None,
    ) -> torch.Tensor:
        key = self._trajectory_key(trajectory, frame_end=frame_end)
        cached = self._load_cached(key)
        return cached if cached is not None else self._encode_tensors(key, prepared)

    @torch.inference_mode()
    def _encode(
        self,
        trajectory: BenchmarkTrajectory,
        *,
        frame_end: Optional[int] = None,
    ) -> torch.Tensor:
        key = self._trajectory_key(trajectory, frame_end=frame_end)
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        prepared = self._prepare_trajectory_tensors(trajectory, frame_end=frame_end)
        return self._encode_tensors(key, prepared)

    def _preload_job(
        self,
        trajectory: BenchmarkTrajectory,
        frame_end: Optional[int],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        key = self._trajectory_key(trajectory, frame_end=frame_end)
        if self._load_cached(key) is not None:
            return key, None
        return key, self._prepare_trajectory_tensors(trajectory, frame_end=frame_end)

    @torch.inference_mode()
    def _encode_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        desc: str = "encode",
        frame_ends: Optional[Sequence[Optional[int]]] = None,
    ) -> List[torch.Tensor]:
        if not trajectories:
            return []
        if frame_ends is None:
            frame_ends = [None] * len(trajectories)
        if len(frame_ends) != len(trajectories):
            raise ValueError("frame_ends must have one entry per trajectory")

        items: Iterable[tuple[BenchmarkTrajectory, Optional[int]]] = zip(
            trajectories, frame_ends
        )
        progress = None
        if self.verbose_fit:
            try:
                from tqdm import tqdm

                progress = tqdm(total=len(trajectories), desc=desc, unit="traj", dynamic_ncols=True)
            except ImportError:
                pass

        output: List[torch.Tensor] = []
        if self.preload_workers == 0:
            for trajectory, frame_end in items:
                output.append(self._encode(trajectory, frame_end=frame_end))
                if progress is not None:
                    progress.update(1)
        else:
            max_pending = max(1, self.preload_workers * self.prefetch_factor)
            iterator = iter(items)
            pending: deque[tuple[BenchmarkTrajectory, Optional[int], Future]] = deque()
            with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
                for _ in range(max_pending):
                    try:
                        trajectory, frame_end = next(iterator)
                    except StopIteration:
                        break
                    pending.append(
                        (
                            trajectory,
                            frame_end,
                            executor.submit(self._preload_job, trajectory, frame_end),
                        )
                    )
                while pending:
                    trajectory, frame_end, future = pending.popleft()
                    key, prepared = future.result()
                    cached = self._load_cached(key)
                    if cached is None:
                        assert prepared is not None
                        cached = self._encode_tensors(key, prepared)
                    output.append(cached)
                    if progress is not None:
                        progress.update(1)
                    try:
                        next_trajectory, next_end = next(iterator)
                    except StopIteration:
                        continue
                    pending.append(
                        (
                            next_trajectory,
                            next_end,
                            executor.submit(self._preload_job, next_trajectory, next_end),
                        )
                    )
        if progress is not None:
            progress.close()
        return output

    def preencode_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        desc: str = "encode",
        success_prefix: bool = False,
    ) -> List[torch.Tensor]:
        frame_ends: Optional[List[Optional[int]]] = None
        if success_prefix:
            frame_ends = []
            for trajectory in trajectories:
                if bool(trajectory.is_failure):
                    frame_ends.append(None)
                    continue
                prefix_fn = getattr(trajectory, "prefix_frames_before_done", None)
                frame_end = (
                    int(prefix_fn()) if prefix_fn is not None else int(trajectory.num_frames)
                )
                if frame_end <= 0:
                    raise ValueError(
                        f"Success trajectory has no pre-done frames: {trajectory.describe()}"
                    )
                frame_ends.append(frame_end)
        return self._encode_trajectories(
            trajectories, desc=desc, frame_ends=frame_ends
        )

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        raise NotImplementedError

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        raise NotImplementedError

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            **self.feature_metadata(),
            "calib_fraction": self.calib_fraction,
            "encode_batch_size": self.encode_batch_size,
            "preload_workers": self.preload_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "feature_cache_dir": (
                None if self.feature_cache_dir is None else str(self.feature_cache_dir)
            ),
        }

    def close(self) -> None:
        self.encoder.close()
