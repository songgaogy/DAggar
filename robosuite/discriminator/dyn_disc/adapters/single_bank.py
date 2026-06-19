"""Benchmark adapter for the cleaned LPB v2 KNN OOD discriminator.

Drop-in alternative to `robosuite.discriminator.lpb.lpb_benchmark` but built on
the original LPB dynamics model + KNN feature definition (see `knn.py`).

Per-task workflow (matches `LPBBenchmarkDiscriminator` in lpb/):
  1. fit_on_benchmark(trajectories): split per-task success demos into
     bank + disjoint calibration, encode all frames via the original LPB encoder,
     build the KNN bank, calibrate the percentile threshold.
  2. score_trajectory(traj): encode the trajectory frame-by-frame, compute
     per-frame min L2 distance to the bank, threshold to produce binary preds.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from robosuite.discriminator.dyn_disc.detectors.single_bank_knn import LPBV2Encoder, LPBV2KNN


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


class LPBV2BenchmarkDiscriminator:
    """LPB v2 KNN OOD detector exposed through BenchmarkTrajectory API."""

    name = "dyn_disc_knn"

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

        self.encoder = LPBV2Encoder(
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

        # Per-task state.
        self._detectors_per_task: Dict[str, LPBV2KNN] = {}
        self._calibration_stats: Dict[str, dict] = {}
        # Cache encoded trajectories: (file_path, demo_path) -> (T, D) tensor.
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
    def _encode(self, trajectory: BenchmarkTrajectory) -> torch.Tensor:
        key = self._trajectory_key(trajectory)
        cached = self._feature_cache.get(key, None)
        if cached is not None:
            return cached

        prepared = self._prepare_trajectory_tensors(trajectory)
        return self._encode_tensors(key, prepared)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        task_to_success: Dict[str, List[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)

        if not task_to_success:
            raise RuntimeError(
                "LPB-original KNN calibration requires success trajectories per task."
            )

        for task, succ_list in task_to_success.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "need at least 2 for disjoint bank + calibration split."
                )
            rng = np.random.default_rng(int(self.seed))
            perm = rng.permutation(len(succ_list))
            n_calib = int(round(self.calib_fraction * len(succ_list)))
            n_calib = max(1, min(len(succ_list) - 1, n_calib))
            calib_idx = set(perm[:n_calib].tolist())
            bank_trajs = [t for i, t in enumerate(succ_list) if i not in calib_idx]
            calib_trajs = [t for i, t in enumerate(succ_list) if i in calib_idx]

            bank_feats: List[torch.Tensor] = []
            calib_feats: List[torch.Tensor] = []
            bank_total = 0
            calib_total = 0
            for t in bank_trajs:
                f = self._encode(t)
                bank_feats.append(f)
                bank_total += int(f.shape[0])
            for t in calib_trajs:
                f = self._encode(t)
                calib_feats.append(f)
                calib_total += int(f.shape[0])

            feat_dim = int(bank_feats[0].shape[1])
            if self.feature_source == "encoder":
                visual_dim = int(self.encoder.visual_emb_dim_total)
                proprio_dim = int(self.encoder.proprio_emb_dim)
                action_dim = int(self.encoder.action_emb_dim)
                visual_weight = float(self.visual_weight)
                proprio_weight = float(self.proprio_weight)
                action_weight = float(self.action_weight)
            else:
                visual_dim = int(feat_dim)
                proprio_dim = 0
                action_dim = 0
                visual_weight = 1.0
                proprio_weight = 1.0
                action_weight = 1.0

            if self.verbose_fit:
                print(
                    f"[dyn_disc][fit] task={task} "
                    f"bank_trajs={len(bank_trajs)} ({bank_total} steps)  "
                    f"calib_trajs={len(calib_trajs)} ({calib_total} steps)  "
                    f"feature_source={self.feature_source} layer={self.transformer_layer}  "
                    f"feat_dim={feat_dim} (visual={visual_dim} + proprio={proprio_dim} + action={action_dim})"
                )

            det = LPBV2KNN(
                visual_dim=visual_dim,
                proprio_dim=proprio_dim,
                action_dim=action_dim,
                visual_weight=visual_weight,
                proprio_weight=proprio_weight,
                action_weight=action_weight,
                delta=self.delta,
                chunk_size=self.knn_chunk_size,
                device=self.device,
            )
            threshold = det.fit(expert_features=bank_feats, calibration_features=calib_feats)
            self._detectors_per_task[task] = det
            self._calibration_stats[task] = {
                "num_success_trajectories": int(len(succ_list)),
                "num_bank_trajectories": int(len(bank_trajs)),
                "num_calib_trajectories": int(len(calib_trajs)),
                "num_bank_steps": int(bank_total),
                "num_calib_steps": int(calib_total),
                "threshold_init": float(threshold),
                "delta_init": float(self.delta),
                "feat_dim": feat_dim,
                "visual_dim": visual_dim,
                "proprio_dim": proprio_dim,
                "action_dim": action_dim,
                "action_weight": float(self.action_weight),
                "feature_source": self.feature_source,
                "transformer_layer": int(self.transformer_layer),
                "effective_visual_weight": float(visual_weight),
                "effective_proprio_weight": float(proprio_weight),
                "effective_action_weight": float(action_weight),
            }

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        det = self._detectors_per_task.get(task, None)
        if det is None:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._detectors_per_task)}"
            )

        T = int(trajectory.num_frames)
        feat = self._encode(trajectory)
        result = det.score(feat)

        step_scores = _pad_to_length(result.step_scores, target_len=T, dtype=np.float32)
        thresholds = _pad_to_length(result.thresholds, target_len=T, dtype=np.float32)
        preds = _pad_to_length(result.preds, target_len=T, dtype=np.int64).astype(np.int64)

        positive = np.where(preds == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: Dict[str, Any] = {
            "task": task,
            "threshold": float(det.threshold) if det.threshold is not None else float("nan"),
            "delta": float(det.delta),
            "thresholds": thresholds,
            "step_scores_raw": step_scores,
            "feature_len": int(feat.shape[0]),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "visual_weight": float(self.visual_weight),
            "proprio_weight": float(self.proprio_weight),
            "effective_visual_weight": float(det.visual_weight),
            "effective_proprio_weight": float(det.proprio_weight),
            "view_names": list(self.encoder.view_names),
        }
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=preds,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "model_ckpt": self.model_ckpt,
            "view_names": list(self.encoder.view_names),
            "camera_to_view": dict(self.camera_to_view),
            "visual_weight": float(self.visual_weight),
            "proprio_weight": float(self.proprio_weight),
            "transformer_metric": "uniform_l2" if self.feature_source == "transformer" else "block_weighted_l2",
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "delta": float(self.delta),
            "knn_chunk_size": int(self.knn_chunk_size),
            "calib_fraction": float(self.calib_fraction),
            "encode_batch_size": int(self.encode_batch_size),
        }

    def close(self) -> None:
        try:
            self.encoder.model.to("cpu")
        except Exception:
            pass


LPBOriginalBenchmarkDiscriminator = LPBV2BenchmarkDiscriminator
