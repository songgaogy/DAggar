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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from .knn import LPBV2Encoder, LPBV2KNN


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

    name = "lpb_v2_knn"

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

        # If user did not provide a camera->view map, default to a 1:1 identity
        # over the encoder's view_names (assume cameras are named the same).
        if self.camera_to_view is None:
            self.camera_to_view = {v: v for v in self.encoder.view_names}

        # Per-task state.
        self._detectors_per_task: Dict[str, LPBV2KNN] = {}
        self._calibration_stats: Dict[str, dict] = {}
        # Cache encoded trajectories: (file_path, demo_path) -> (T, D) tensor.
        self._feature_cache: Dict[tuple, torch.Tensor] = {}

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

    def _slice_proprio(self, states: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
        if self.proprio_indices is not None:
            states = states[:, self.proprio_indices]
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

    @torch.no_grad()
    def _encode(self, trajectory: BenchmarkTrajectory) -> torch.Tensor:
        key = self._trajectory_key(trajectory)
        cached = self._feature_cache.get(key, None)
        if cached is not None:
            return cached

        view_names = self.encoder.view_names
        cameras_needed = [self._resolve_camera(v) for v in view_names]
        images_by_cam = trajectory.load_images(cameras=cameras_needed)
        states = np.asarray(trajectory.load_states(), dtype=np.float32)
        actions = None
        if self.feature_source == "transformer":
            actions = np.asarray(trajectory.load_actions(), dtype=np.float32)

        # IMPORTANT: benchmark trajectories can have minor length mismatches across
        # (states, actions, per-camera images). We always truncate to the shortest so each
        # encoded feature corresponds to a real frame for all modalities.
        cam_lens = [int(images_by_cam[c].shape[0]) for c in cameras_needed]
        lens = [states.shape[0]] + cam_lens
        if actions is not None:
            lens.append(actions.shape[0])
        T = int(min(lens))
        if T <= 0:
            raise ValueError(f"Empty trajectory: {trajectory.describe()}")

        # Determine target proprio dim from the encoder's proprio_encoder.
        # The proprio_encoder.in_chans is set at load time to match training.
        target_proprio_dim = None
        try:
            target_proprio_dim = int(self.encoder.model.proprio_encoder.in_chans)
        except Exception:
            target_proprio_dim = None

        # Proprio slicing/padding must match the training-time proprio encoder input.
        prop = self._slice_proprio(states[:T], target_dim=target_proprio_dim)
        act = None
        if actions is not None:
            target_action_dim = None
            try:
                target_action_dim = int(self.encoder.model.action_encoder.in_chans)
            except Exception:
                target_action_dim = None
            act = self._slice_action(actions[:T], target_dim=target_action_dim)

        # Pre-build per-view image arrays (T, 3, H, W) at original_img_size,
        # converted to float in [0, 1].
        H = self.encoder.original_img_size
        per_view_chw: Dict[str, np.ndarray] = {}
        for v in view_names:
            cam = self._resolve_camera(v)
            imgs = np.asarray(images_by_cam[cam][:T])
            if imgs.shape[1] != H or imgs.shape[2] != H:
                # Resize via torch later in a tight batched loop; do it now per traj
                # keeping things simple: fall back to torch interpolate.
                t_imgs = torch.from_numpy(imgs.astype(np.float32))
                if t_imgs.max() > 1.5:
                    t_imgs = t_imgs / 255.0
                t_imgs = t_imgs.permute(0, 3, 1, 2)
                t_imgs = torch.nn.functional.interpolate(
                    t_imgs, size=(H, H), mode="bilinear", align_corners=False
                )
                per_view_chw[v] = t_imgs.numpy()
            else:
                arr = imgs.astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                per_view_chw[v] = np.transpose(arr, (0, 3, 1, 2))

        feats: List[torch.Tensor] = []
        bs = self.encode_batch_size
        for start in range(0, T, bs):
            end = min(start + bs, T)
            batch_imgs = {
                v: torch.from_numpy(per_view_chw[v][start:end]) for v in view_names
            }
            batch_prop = torch.from_numpy(prop[start:end])
            batch_act = None if act is None else torch.from_numpy(act[start:end])
            # encode_batch applies the same normalization/cropping used during dynamics training.
            f = self.encoder.encode_batch(batch_imgs, batch_prop, actions=batch_act)
            feats.append(f.detach().cpu())

        out = torch.cat(feats, dim=0)
        self._feature_cache[key] = out
        return out

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
                visual_weight = float(self.visual_weight)
                proprio_weight = float(self.proprio_weight)
            else:
                visual_dim = int(feat_dim)
                proprio_dim = 0
                visual_weight = 1.0
                proprio_weight = 1.0

            if self.verbose_fit:
                print(
                    f"[lpb_v2][fit] task={task} "
                    f"bank_trajs={len(bank_trajs)} ({bank_total} steps)  "
                    f"calib_trajs={len(calib_trajs)} ({calib_total} steps)  "
                    f"feature_source={self.feature_source} layer={self.transformer_layer}  "
                    f"feat_dim={feat_dim} (visual={visual_dim} + proprio={proprio_dim})"
                )

            det = LPBV2KNN(
                visual_dim=visual_dim,
                proprio_dim=proprio_dim,
                visual_weight=visual_weight,
                proprio_weight=proprio_weight,
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
                "feature_source": self.feature_source,
                "transformer_layer": int(self.transformer_layer),
                "effective_visual_weight": float(visual_weight),
                "effective_proprio_weight": float(proprio_weight),
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
