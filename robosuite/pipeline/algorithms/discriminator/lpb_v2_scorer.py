"""LPB v2 reference scorer for IQL offline warmup and visualization.

This module wraps the legacy `BCEBenchmarkDiscriminator.score_trajectory(...)` path
so it can be reused as the **reward source** during IQL Q/V warmup (replacing the
broken `OnlineBCEDiscriminator` reward described in
`pipeline/docs/IQL_DISCRIMINATOR_REWARD_DEBUG.md`) AND as the **scorer** behind the
Q/V visualization (`pipeline/algorithms/q_learning/utils/vis_qv.py`).

Why a new module:
- Previously the same helpers (`_load_lpb_bce_detector`, `_build_bce_discriminator_from_ckpt`,
  `_compute_youden_threshold_for_task`, `SafeRobosuiteBenchmarkTrajectory`,
  `_build_selected_trajectory`) lived only inside `vis_discriminator_util.py`.
  Sharing them here means the warmup reward path and the visualization path use
  ONE scorer, eliminating the OnlineBCE vs LPB-v2 semantic split that caused
  the bug.

Public surface:
- `LPBV2OfflineScorer`: build once per warmup/vis run; resolves tau from
  `meta.json["bce_youden_threshold"]` first (operational threshold), falling
  back to a recomputed Youden threshold and finally to the checkpoint
  detector threshold with a clear warning.
- `score_hdf5_demo(hdf5_path, demo_key)`: per-frame failure_score `(T,)` via
  `BCEBenchmarkDiscriminator.score_trajectory(...)`. Success-rollout HDF5 demos
  first resolve to the LPB preprocessed cache by action fingerprint so the score
  matches `visualize_bce.py`; other demos use raw HDF5 images/states/actions.
- `failure_score_to_margin_reward(scores)`: returns `tau - failure_score`
  (per user's `r_disc` convention: higher = more expert-like).
- `annotate_transitions_with_lpb_scores(transitions, failure_scores, tau)`:
  writes `lpb_failure_score / lpb_margin_reward / lpb_tau` onto each
  transition's `info` dict for the replay sampler to read at sample time.
"""

from __future__ import annotations

import hashlib
import h5py
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hydra.utils import to_absolute_path

from benchmark.robosuite.trajectory import RobosuiteBenchmarkTrajectory
from robosuite.discriminator.dyn_bce.task_registry import resolve_checkpoint_task_name
from robosuite.discriminator.lpb_v2.adapters.bce import BCEBenchmarkDiscriminator
from robosuite.discriminator.lpb_v2.detectors.bce import (
    BCEDiscriminator,
    two_class_youden_threshold,
)
from robosuite.discriminator.lpb_v2.visualization.visualize_bce import (
    _build_benchmark_and_bank,
    _compute_fail_suffix_failure_scores_per_task,
    _compute_success_calib_failure_scores_per_task,
)
from robosuite.pipeline.common import Transition


DEFAULT_FAIL_ROOT = "data/utils/fail_rollout"
DEFAULT_SUCCESS_ROOT = "data/utils/success_rollout"
DEFAULT_FAIL_TRAIN_ROOT = "data/utils/fail_labeled_train"
DEFAULT_SUCCESS_CACHE_ROOT = "data/.lpb_score_preprocessed_cache"
DEFAULT_METADATA_CACHE_ROOT = "data/.lpb_score_cache"
DEFAULT_CACHE_CAMERA_NAMES = ("agentview", "birdview", "frontview")

# Common ckpt layout: <root>/<task>/bce_head.pth lives next to a meta.json
# that carries the operational Youden threshold ("bce_youden_threshold").
META_JSON_FILENAME = "meta.json"


def _action_fingerprint(actions: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    hasher = hashlib.sha1()
    hasher.update(str(tuple(arr.shape)).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _cache_camera_names(num_views: int) -> tuple[str, ...]:
    names = tuple(DEFAULT_CACHE_CAMERA_NAMES)
    if len(names) >= int(num_views):
        return names[: int(num_views)]
    extra = tuple(f"view_{i}" for i in range(len(names), int(num_views)))
    return names + extra


# --------------------------------------------------------------------------- #
# Detector loading helpers (moved from vis_discriminator_util.py so the scorer #
# and the visualizer share a single source of truth).                          #
# --------------------------------------------------------------------------- #

def _load_lpb_bce_detector(
    disc_ckpt: Path, *, device: str
) -> tuple[BCEDiscriminator, dict[str, Any]]:
    if not disc_ckpt.exists():
        raise FileNotFoundError(f"Discriminator checkpoint does not exist: {disc_ckpt}")
    payload = torch.load(str(disc_ckpt), map_location="cpu", weights_only=False)
    state = payload.get("bce_detector", None)
    if not isinstance(state, dict):
        raise KeyError(f"BCE checkpoint {disc_ckpt} is missing 'bce_detector'.")
    in_dim = int(payload.get("in_dim", state.get("in_dim", 0)))
    hidden = int(payload.get("hidden", state.get("hidden", 256)))
    num_layers = int(payload.get("num_layers", state.get("num_layers", 2)))
    if in_dim <= 0:
        raise KeyError(f"BCE checkpoint {disc_ckpt} does not define a positive in_dim.")

    detector = BCEDiscriminator(
        in_dim=in_dim,
        hidden=hidden,
        num_layers=num_layers,
        device=str(device),
    )
    detector.load_state_dict(state)
    detector.head.eval()
    return detector, payload


def _build_bce_discriminator_from_ckpt(
    disc_ckpt: Path,
    *,
    device: str,
    batch_size: int,
) -> tuple[BCEBenchmarkDiscriminator, BCEDiscriminator, dict[str, Any]]:
    detector, payload = _load_lpb_bce_detector(disc_ckpt, device=device)
    model_ckpt_raw = payload.get("model_ckpt", None)
    if not model_ckpt_raw:
        raise KeyError(f"BCE checkpoint {disc_ckpt} is missing 'model_ckpt'.")
    model_ckpt = Path(to_absolute_path(str(model_ckpt_raw))).resolve()
    if not model_ckpt.exists():
        raise FileNotFoundError(f"LPB dynamics checkpoint does not exist: {model_ckpt}")

    discriminator = BCEBenchmarkDiscriminator(
        model_ckpt=str(model_ckpt),
        fail_bank_trajectories=[],
        fail_calib_trajectories=[],
        max_expert_other_ratio=None,
        head_hidden=int(detector.hidden),
        head_layers=int(detector.num_layers),
        epochs=int(payload.get("epoch", 0) or 1),
        save_ckpt_dir=None,
        device=str(device),
        encode_batch_size=max(1, int(batch_size)),
        feature_source=str(payload.get("feature_source", "transformer")),
        transformer_layer=int(payload.get("transformer_layer", 1)),
        verbose_fit=False,
    )
    discriminator._shared_detector = detector
    discriminator._detectors_per_task = {task: detector for task in detector.thresholds}
    discriminator._global_stats = {
        "feat_dim": int(detector.in_dim),
        "epochs": int(payload.get("epoch", 0) or 0),
        "head_hidden": int(detector.hidden),
        "head_layers": int(detector.num_layers),
        "feature_source": str(payload.get("feature_source", discriminator.feature_source)),
        "transformer_layer": int(payload.get("transformer_layer", discriminator.transformer_layer)),
        "max_expert_other_ratio": payload.get("max_expert_other_ratio"),
        "num_fail_bank_trajectories": len(payload.get("fail_bank_video_ids", []) or []),
        "num_fail_bank_skipped_no_gt": None,
        "loaded_from_ckpt": str(disc_ckpt),
    }
    discriminator._calibration_stats = {}
    for task, tau in detector.thresholds.items():
        cs = detector.calib_stats.get(task)
        discriminator._calibration_stats[task] = {
            "threshold": float(tau),
            "calib_score_min": None if cs is None else float(cs.calib_score_min),
            "calib_score_max": None if cs is None else float(cs.calib_score_max),
            "calib_score_mean": None if cs is None else float(cs.calib_score_mean),
            "calib_score_std": None if cs is None else float(cs.calib_score_std),
            "num_calib_success_frames": None if cs is None else int(cs.num_calib_frames),
        }
    return discriminator, detector, payload


def _resolve_task_for_detector(task_name: str, detector: BCEDiscriminator) -> str:
    candidates = [str(task_name)]
    try:
        candidates.append(resolve_checkpoint_task_name(str(task_name)))
    except KeyError:
        pass
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in detector.thresholds:
            return candidate
    raise KeyError(
        f"Task {task_name!r} has no threshold in discriminator checkpoint. "
        f"Available tasks: {sorted(detector.thresholds)}"
    )


def _compute_youden_threshold_for_task(
    discriminator: BCEBenchmarkDiscriminator,
    *,
    task_name: str,
    seed: int = 0,
    calib_fraction: float = 0.2,
    fail_bank_per_task: int = 25,
) -> tuple[float, str]:
    """Recompute Youden tau from success_calib + fail_bank if meta.json is missing."""
    success_cache_root = Path(to_absolute_path(DEFAULT_SUCCESS_CACHE_ROOT)).resolve()
    metadata_cache_root = Path(to_absolute_path(DEFAULT_METADATA_CACHE_ROOT)).resolve()
    use_success_cache = success_cache_root.is_dir() and metadata_cache_root.is_dir()
    args = Namespace(
        kind="robosuite",
        fail_root=str(Path(to_absolute_path(DEFAULT_FAIL_ROOT)).resolve()),
        success_root=str(Path(to_absolute_path(DEFAULT_SUCCESS_ROOT)).resolve()),
        fail_train_root=str(Path(to_absolute_path(DEFAULT_FAIL_TRAIN_ROOT)).resolve()),
        task=str(task_name),
        max_fail_per_task=100,
        max_success_per_task=100,
        success_cache_root=str(success_cache_root) if use_success_cache else None,
        metadata_cache_root=str(metadata_cache_root) if use_success_cache else None,
        cache_camera_names=list(DEFAULT_CACHE_CAMERA_NAMES) if use_success_cache else None,
        fail_bank_per_task=int(fail_bank_per_task),
    )
    _bench, trajs, fail_bank_trajs = _build_benchmark_and_bank(args)
    fail_suffix_scores = _compute_fail_suffix_failure_scores_per_task(
        discriminator,
        fail_bank_trajs,
    )
    calib_scores = _compute_success_calib_failure_scores_per_task(
        discriminator,
        trajs,
        seed=int(seed),
        calib_fraction=float(calib_fraction),
    )
    s_fail = fail_suffix_scores.get(str(task_name))
    s_succ = calib_scores.get(str(task_name))
    if s_fail is None or s_fail.size == 0:
        raise RuntimeError(f"No fail-bank suffix scores for task {task_name!r}.")
    if s_succ is None or s_succ.size == 0:
        raise RuntimeError(f"No success-calib scores for task {task_name!r}.")
    tau = float(two_class_youden_threshold(s_succ, s_fail))
    source = f"youden(n_succ={int(s_succ.size)}, n_fail={int(s_fail.size)}, seed={int(seed)})"
    return tau, source


# --------------------------------------------------------------------------- #
# Trajectory subclass used by both the scorer and the visualizer.             #
# --------------------------------------------------------------------------- #

class SafeRobosuiteBenchmarkTrajectory(RobosuiteBenchmarkTrajectory):
    """Tolerant of missing failure_frame_mask / failure_segment_index annotations.

    The base RobosuiteBenchmarkTrajectory raises if these annotations are
    absent; warmup data may not have them, so we degrade gracefully.
    """

    def load_failure_mask(self) -> np.ndarray | None:
        if not self.is_failure:
            return None
        if self._has_cache():
            try:
                return np.asarray(self._cache_array("failure_mask"), dtype=np.uint8)
            except Exception:
                return None
        with h5py.File(self.file_path, "r") as file_handle:
            group = file_handle[self.demo_path]
            annotations = group.get("annotations", None)
            if annotations is None or "failure_frame_mask" not in annotations:
                return None
            return np.asarray(annotations["failure_frame_mask"][:], dtype=np.uint8)

    def load_failure_segment_index(self) -> np.ndarray | None:
        if not self.is_failure:
            return None
        if self._has_cache():
            try:
                return np.asarray(self._cache_array("failure_segment_index"), dtype=np.int32)
            except Exception:
                return None
        with h5py.File(self.file_path, "r") as file_handle:
            group = file_handle[self.demo_path]
            annotations = group.get("annotations", None)
            if annotations is None or "failure_segment_index" not in annotations:
                return None
            return np.asarray(annotations["failure_segment_index"][:], dtype=np.int32)


class TransitionBackedRobosuiteBenchmarkTrajectory(RobosuiteBenchmarkTrajectory):
    """Benchmark trajectory backed by already preprocessed pipeline transitions."""

    def __init__(
        self,
        *,
        task_name: str,
        transitions: Sequence[Transition],
        camera_names: Sequence[str],
        video_id: str,
        fps: int,
        is_failure: bool,
        failure_segments: list[dict[str, Any]],
        source_hdf5_path: str = "",
        source_demo_key: str = "",
        failure_mask: np.ndarray | None = None,
    ) -> None:
        if not transitions:
            raise ValueError("TransitionBackedRobosuiteBenchmarkTrajectory requires transitions.")
        self._transitions = list(transitions)
        self._camera_names = tuple(str(name) for name in camera_names)
        self._failure_mask = None if failure_mask is None else np.asarray(failure_mask, dtype=np.uint8)
        super().__init__(
            task_name=str(task_name),
            num_frames=len(self._transitions),
            is_failure=bool(is_failure),
            video_id=str(video_id),
            fps=int(fps),
            available_cameras=self._camera_names,
            failure_segments=list(failure_segments),
            source_hdf5_path=str(source_hdf5_path),
            source_demo_key=str(source_demo_key),
            file_path="",
            demo_path="",
            cache_npz_path="",
        )

    def load_images(self, cameras: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        req = tuple(str(c) for c in (cameras if cameras is not None else self._camera_names))
        missing = [c for c in req if c not in self._camera_names]
        if missing:
            raise KeyError(
                f"cameras {missing} not available for {self.video_id}; "
                f"available: {self._camera_names}"
            )
        return {
            cam: np.stack(
                [np.asarray(trans.obs[cam], dtype=np.uint8) for trans in self._transitions],
                axis=0,
            )
            for cam in req
        }

    def load_states(self) -> np.ndarray:
        return np.stack(
            [np.asarray(trans.obs["state"], dtype=np.float32) for trans in self._transitions],
            axis=0,
        )

    def load_actions(self) -> np.ndarray:
        return np.stack(
            [np.asarray(trans.action, dtype=np.float32) for trans in self._transitions],
            axis=0,
        )

    def load_failure_mask(self) -> np.ndarray | None:
        if not self.is_failure:
            return None
        return None if self._failure_mask is None else self._failure_mask.copy()

    def load_failure_segment_index(self) -> np.ndarray | None:
        return None


def _load_failure_mask_and_segments(
    hdf5_path: Path, demo_key: str, length: int
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    with h5py.File(hdf5_path, "r") as file_handle:
        root = "demos" if "demos" in file_handle else "data"
        group = file_handle[root][str(demo_key)]
        annotations = group.get("annotations", None)
        if annotations is not None and "failure_frame_mask" in annotations:
            mask = np.asarray(annotations["failure_frame_mask"][:], dtype=np.uint8)
            mask = mask[: int(length)]
        else:
            mask = None

        segments_raw = group.attrs.get("failure_segments_json", "[]")
        if isinstance(segments_raw, bytes):
            segments_raw = segments_raw.decode("utf-8")
        try:
            segments = json.loads(str(segments_raw))
            if not isinstance(segments, list):
                segments = []
        except Exception:
            segments = []
    return mask, segments


def _build_selected_trajectory(
    *,
    hdf5_path: Path,
    demo_key: str,
    task_name: str,
    fps: int,
) -> SafeRobosuiteBenchmarkTrajectory:
    with h5py.File(hdf5_path, "r") as file_handle:
        root = "demos" if "demos" in file_handle else "data"
        demo_path = f"{root}/{demo_key}"
        group = file_handle[demo_path]
        actions_len = int(group["actions"].shape[0])
        states_len = int(group["states"].shape[0])
        obs = group["observations"]
        cameras = tuple(sorted(str(k) for k in obs.keys()))
        image_lens = [int(obs[camera]["images"].shape[0]) for camera in cameras]
        length = int(min([actions_len, states_len] + image_lens))
        successful = bool(group.attrs.get("successful", False))
        video_id = str(group.attrs.get("video_id", f"{hdf5_path.stem}_{demo_key}"))

    gt_mask, segments = _load_failure_mask_and_segments(hdf5_path, demo_key, length)
    return SafeRobosuiteBenchmarkTrajectory(
        task_name=str(task_name),
        num_frames=int(length),
        is_failure=(not successful) or bool(segments) or (gt_mask is not None and bool(gt_mask.any())),
        video_id=video_id,
        fps=int(fps),
        available_cameras=cameras,
        failure_segments=list(segments),
        source_hdf5_path=str(hdf5_path),
        source_demo_key=str(demo_key),
        file_path=str(hdf5_path),
        demo_path=demo_path,
    )


def _build_transition_trajectory(
    *,
    transitions: Sequence[Transition],
    task_name: str,
    camera_names: Sequence[str] | None,
    fps: int,
    hdf5_path: Path | None = None,
    demo_key: str | None = None,
    successful: bool | None = None,
) -> TransitionBackedRobosuiteBenchmarkTrajectory:
    """Build a benchmark trajectory from pipeline-preprocessed transitions."""
    if not transitions:
        raise ValueError("_build_transition_trajectory requires non-empty transitions.")

    if camera_names is None:
        camera_tuple = tuple(
            sorted(str(k) for k in transitions[0].obs.keys() if str(k) != "state")
        )
    else:
        camera_tuple = tuple(str(name) for name in camera_names)
    if not camera_tuple:
        raise ValueError("_build_transition_trajectory could not resolve any cameras.")

    for camera_name in camera_tuple:
        if camera_name not in transitions[0].obs:
            raise KeyError(
                f"Transition observation is missing camera '{camera_name}'. "
                f"Available: {sorted(transitions[0].obs.keys())}"
            )

    gt_mask: np.ndarray | None = None
    segments: list[dict[str, Any]] = []
    if hdf5_path is not None and demo_key:
        gt_mask, segments = _load_failure_mask_and_segments(
            Path(hdf5_path),
            str(demo_key),
            length=len(transitions),
        )
        if gt_mask is not None and gt_mask.shape[0] != len(transitions):
            aligned = np.zeros((len(transitions),), dtype=np.uint8)
            n_copy = min(int(gt_mask.shape[0]), len(transitions))
            aligned[:n_copy] = gt_mask[:n_copy]
            gt_mask = aligned

    if successful is None:
        demo_success_values = [
            bool((trans.info or {}).get("demo_success_attr", False))
            for trans in transitions
            if "demo_success_attr" in (trans.info or {})
        ]
        successful = bool(all(demo_success_values)) if demo_success_values else None

    is_failure = (
        (successful is False)
        or bool(segments)
        or (gt_mask is not None and bool(gt_mask.any()))
    )
    if hdf5_path is not None and demo_key:
        video_id = f"{Path(hdf5_path).stem}_{demo_key}"
    else:
        video_id = str((transitions[0].info or {}).get("demo_name", "transition_demo"))

    return TransitionBackedRobosuiteBenchmarkTrajectory(
        task_name=str(task_name),
        transitions=transitions,
        camera_names=camera_tuple,
        video_id=video_id,
        fps=int(fps),
        is_failure=bool(is_failure),
        failure_segments=segments,
        source_hdf5_path="" if hdf5_path is None else str(hdf5_path),
        source_demo_key="" if demo_key is None else str(demo_key),
        failure_mask=gt_mask,
    )


def _is_success_rollout_hdf5(path: Path) -> bool:
    return "success_rollout" in {str(part) for part in path.parts}


def _read_hdf5_demo_actions(hdf5_path: Path, demo_key: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as file_handle:
        root = "demos" if "demos" in file_handle else "data"
        group = file_handle[root][str(demo_key)]
        return np.asarray(group["actions"][:], dtype=np.float32)


def _build_cache_trajectory(
    *,
    cache_path: Path,
    task_name: str,
    fps: int,
) -> RobosuiteBenchmarkTrajectory:
    with np.load(str(cache_path), allow_pickle=False) as data:
        images = data["images_chw"]
        proprio = data["proprio"]
        actions = data["actions"]
        length = int(
            min(int(images.shape[0]), int(proprio.shape[0]), int(actions.shape[0]))
        )
        cameras = _cache_camera_names(int(images.shape[1]))
    return RobosuiteBenchmarkTrajectory(
        task_name=str(task_name),
        num_frames=int(length),
        is_failure=False,
        video_id=str(cache_path.stem),
        fps=int(fps),
        available_cameras=cameras,
        failure_segments=[],
        cache_npz_path=str(cache_path),
    )


# --------------------------------------------------------------------------- #
# LPBV2OfflineScorer                                                          #
# --------------------------------------------------------------------------- #

def load_bce_youden_threshold(
    bce_ckpt_path: str | Path,
    *,
    meta_json_path: str | Path | None = None,
    required: bool = True,
) -> tuple[float, Path]:
    """Load operational Youden threshold ``bce_youden_threshold`` from meta.json.

    Default lookup: ``<bce_ckpt.parent>/meta.json`` (canonical ``bce_ckeckpoints``
    layout). Raises when ``required=True`` and the value cannot be read.
    """
    ckpt_path = Path(bce_ckpt_path).resolve()
    explicit = (
        Path(to_absolute_path(str(meta_json_path))).resolve()
        if meta_json_path is not None
        else None
    )
    tau, used = _resolve_meta_json_tau(ckpt_path, meta_json_path=explicit)
    if tau is not None and used is not None:
        return float(tau), used
    if required:
        hint = explicit if explicit is not None else ckpt_path.parent / META_JSON_FILENAME
        raise FileNotFoundError(
            f"Missing or invalid bce_youden_threshold in {hint}. "
            "Export meta.json next to bce_head.pth (see visualize_bce / "
            "checkpoints/lpb_v2/bce_ckeckpoints/<TASK>/meta.json)."
        )
    raise ValueError("load_bce_youden_threshold called with required=False but tau missing")


def _resolve_meta_json_tau(
    ckpt_path: Path, *, meta_json_path: Path | None
) -> tuple[float | None, Path | None]:
    """Return (tau, path) read from `meta.json["bce_youden_threshold"]`, or
    `(None, None)` if the file is missing/malformed."""
    candidate = meta_json_path
    if candidate is None:
        candidate = ckpt_path.parent / META_JSON_FILENAME
    if not candidate.exists():
        return None, None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    value = payload.get("bce_youden_threshold", None)
    if value is None:
        return None, candidate
    try:
        return float(value), candidate
    except (TypeError, ValueError):
        return None, candidate


class LPBV2OfflineScorer:
    """Scores robosuite HDF5 demos with the reference LPB v2 BCE detector.

    Resolves the operational Youden threshold tau in this order:
        1. `<ckpt.parent>/meta.json["bce_youden_threshold"]` (or an explicit
           `meta_json_path`) — this is what `visualize_bce.py` writes by default
           and what the user has flagged as the canonical operational value.
        2. Recomputed two-class Youden from `_compute_youden_threshold_for_task`
           (requires success + fail_bank data on disk).
        3. The per-task threshold inside the ckpt's `bce_detector.thresholds`
           dict — printed as a loud warning since this is the detection-time
           threshold, not the visualized one (they differ).
    """

    def __init__(
        self,
        *,
        bce_ckpt_path: str | Path,
        task_name: str,
        device: str,
        batch_size: int = 32,
        meta_json_path: str | Path | None = None,
        allow_youden_recompute: bool = True,
    ) -> None:
        self.ckpt_path = Path(to_absolute_path(str(bce_ckpt_path))).resolve()
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.task_name = str(task_name)
        self._success_cache_root = Path(to_absolute_path(DEFAULT_SUCCESS_CACHE_ROOT)).resolve()
        self._success_cache_index: dict[str, Path] | None = None
        self._success_cache_warned_unavailable = False
        self._success_cache_misses: set[str] = set()

        discriminator, detector, payload = _build_bce_discriminator_from_ckpt(
            self.ckpt_path,
            device=self.device,
            batch_size=self.batch_size,
        )
        self._discriminator: BCEBenchmarkDiscriminator = discriminator
        self._detector: BCEDiscriminator = detector
        self._payload: dict[str, Any] = payload
        self.detector_task: str = _resolve_task_for_detector(self.task_name, detector)
        self.checkpoint_detector_threshold: float = float(
            detector.thresholds.get(self.detector_task, float("nan"))
        )

        explicit_meta = (
            Path(to_absolute_path(str(meta_json_path))).resolve()
            if meta_json_path is not None
            else None
        )
        meta_tau, meta_used = _resolve_meta_json_tau(
            self.ckpt_path, meta_json_path=explicit_meta
        )

        if meta_tau is not None:
            self.tau = float(meta_tau)
            self.tau_source = f"meta_json({meta_used})"
        elif allow_youden_recompute:
            try:
                tau, source = _compute_youden_threshold_for_task(
                    self._discriminator,
                    task_name=self.detector_task,
                )
                self.tau = float(tau)
                self.tau_source = source
            except Exception as exc:
                self.tau = float(self.checkpoint_detector_threshold)
                self.tau_source = f"checkpoint_detector(fallback_after:{type(exc).__name__})"
                print(
                    "[LPBV2OfflineScorer][WARN] meta.json missing and Youden "
                    f"recompute failed ({type(exc).__name__}: {exc}); "
                    f"falling back to checkpoint detector tau={self.tau:.6f}.",
                    flush=True,
                )
        else:
            self.tau = float(self.checkpoint_detector_threshold)
            self.tau_source = "checkpoint_detector"
            print(
                "[LPBV2OfflineScorer][WARN] meta.json missing and Youden "
                f"recompute disabled; falling back to checkpoint detector "
                f"tau={self.tau:.6f}.",
                flush=True,
            )

        print(
            f"[LPBV2OfflineScorer] ready: ckpt={self.ckpt_path.name} "
            f"task={self.task_name} detector_task={self.detector_task} "
            f"tau={self.tau:.6f} (source={self.tau_source}) "
            f"checkpoint_detector_tau={self.checkpoint_detector_threshold:.6f}",
            flush=True,
        )

    def _get_success_cache_index(self) -> dict[str, Path]:
        if self._success_cache_index is not None:
            return self._success_cache_index

        task_dir = self._success_cache_root / str(self.detector_task)
        index: dict[str, Path] = {}
        if not task_dir.is_dir():
            if not self._success_cache_warned_unavailable:
                print(
                    f"[LPBV2OfflineScorer][WARN] success cache dir not found: {task_dir}; "
                    "success_rollout scoring will use raw HDF5.",
                    flush=True,
                )
                self._success_cache_warned_unavailable = True
            self._success_cache_index = index
            return index

        for cache_path in sorted(task_dir.glob("*.npz")):
            try:
                with np.load(str(cache_path), allow_pickle=False) as data:
                    if "actions" not in data:
                        continue
                    fp = _action_fingerprint(np.asarray(data["actions"], dtype=np.float32))
            except Exception:
                continue
            index.setdefault(fp, cache_path)

        print(
            f"[LPBV2OfflineScorer] indexed {len(index)} success cache demos from {task_dir}",
            flush=True,
        )
        self._success_cache_index = index
        return index

    def _maybe_build_success_cache_trajectory(
        self,
        hdf5_path: Path,
        demo_key: str,
        *,
        fps: int,
    ) -> RobosuiteBenchmarkTrajectory | None:
        if not _is_success_rollout_hdf5(hdf5_path):
            return None

        index = self._get_success_cache_index()
        if not index:
            return None

        actions = _read_hdf5_demo_actions(hdf5_path, str(demo_key))
        fp = _action_fingerprint(actions)
        cache_path = index.get(fp)
        if cache_path is None:
            miss_key = f"{hdf5_path}:{demo_key}"
            if miss_key not in self._success_cache_misses:
                print(
                    f"[LPBV2OfflineScorer][WARN] no LPB success cache match for "
                    f"{miss_key}; using raw HDF5.",
                    flush=True,
                )
                self._success_cache_misses.add(miss_key)
            return None

        return _build_cache_trajectory(
            cache_path=cache_path,
            task_name=self.detector_task,
            fps=int(fps),
        )

    # ------------------------------------------------------------------ #
    # Public scoring API                                                  #
    # ------------------------------------------------------------------ #

    def score_hdf5_demo(
        self,
        hdf5_path: str | Path,
        demo_key: str,
        *,
        fps: int = 20,
    ) -> np.ndarray:
        """Score one HDF5 demo and return per-frame failure_score `(T,)`.

        `T` is the demo's frame count after the encoder's internal truncation
        (min length across actions/states/cameras). Higher = more failure-like.
        """
        hdf5_path_obj = Path(hdf5_path)
        trajectory = self._maybe_build_success_cache_trajectory(
            hdf5_path_obj,
            str(demo_key),
            fps=int(fps),
        )
        if trajectory is None:
            trajectory = _build_selected_trajectory(
                hdf5_path=hdf5_path_obj,
                demo_key=str(demo_key),
                task_name=self.detector_task,
                fps=int(fps),
            )
        scored = self._discriminator.score_trajectory(trajectory)
        return np.asarray(scored.step_scores, dtype=np.float32)

    def score_transitions(
        self,
        transitions: Sequence[Transition],
        *,
        camera_names: Sequence[str] | None = None,
        fps: int = 20,
        hdf5_path: str | Path | None = None,
        demo_key: str | None = None,
        successful: bool | None = None,
    ) -> np.ndarray:
        """Score already-loaded pipeline transitions.

        This path preserves the Flow/IQL preprocessing done by
        ``load_hdf5_demos_into_flow_transitions``: images are the policy frames
        and states are the extracted 14-D robot proprio, not raw simulator state.
        """
        trajectory = _build_transition_trajectory(
            transitions=transitions,
            task_name=self.detector_task,
            camera_names=camera_names,
            fps=int(fps),
            hdf5_path=None if hdf5_path is None else Path(hdf5_path),
            demo_key=demo_key,
            successful=successful,
        )
        scored = self._discriminator.score_trajectory(trajectory)
        return np.asarray(scored.step_scores, dtype=np.float32)

    def failure_score_to_margin_reward(
        self, failure_score: np.ndarray | float
    ) -> np.ndarray | float:
        """Map failure_score -> r_disc with the Youden margin convention:

            r_disc = tau - failure_score

        so frames more failure-like than tau get a negative reward and frames
        more expert-like get a positive reward.
        """
        if isinstance(failure_score, np.ndarray):
            return (float(self.tau) - failure_score.astype(np.float32))
        return float(self.tau) - float(failure_score)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "bce_ckpt": str(self.ckpt_path),
            "task": str(self.task_name),
            "detector_task": str(self.detector_task),
            "tau": float(self.tau),
            "tau_source": str(self.tau_source),
            "checkpoint_detector_threshold": float(self.checkpoint_detector_threshold),
            "device": str(self.device),
            "batch_size": int(self.batch_size),
            "feature_source": str(self._payload.get("feature_source", "")),
            "transformer_layer": int(self._payload.get("transformer_layer", -1)),
        }


# --------------------------------------------------------------------------- #
# Transition annotation helper                                                #
# --------------------------------------------------------------------------- #

def lpb_disc_intrinsic_from_failure_score(
    failure_score: float | np.ndarray, tau: float
) -> np.ndarray | float:
    """``r_disc = -sigmoid(failure_score - tau)`` ∈ (-1, 0) (LPB / vis convention)."""
    fs = np.asarray(failure_score, dtype=np.float64)
    t = float(tau)
    out = -1.0 / (1.0 + np.exp(-(fs - t)))
    if np.ndim(fs) == 0:
        return float(out)
    return out.astype(np.float32)


def annotate_transitions_lpb_by_demo(
    transitions: Sequence[Transition],
    scorer: LPBV2OfflineScorer,
    *,
    fps: int = 20,
) -> int:
    """Score each HDF5 demo in ``transitions`` and write LPB fields into ``info``.

    Groups by ``(source_hdf5_path, demo_name)``. Returns the number of demos scored.
    """
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[Transition]] = defaultdict(list)
    for trans in transitions:
        info = trans.info or {}
        hdf5_path = info.get("source_hdf5_path")
        demo_key = info.get("demo_name")
        if hdf5_path is None or demo_key is None:
            continue
        grouped[(str(hdf5_path), str(demo_key))].append(trans)

    scored = 0
    for (hdf5_path, demo_key), demo_transitions in grouped.items():
        failure_scores = scorer.score_hdf5_demo(
            hdf5_path,
            demo_key,
            fps=int(fps),
        )
        annotate_transitions_with_lpb_scores(
            demo_transitions,
            failure_scores=failure_scores,
            tau=float(scorer.tau),
        )
        scored += 1
    return scored


def annotate_transitions_with_lpb_scores(
    transitions: Sequence[Transition],
    *,
    failure_scores: np.ndarray,
    tau: float,
) -> None:
    """Write `lpb_failure_score / lpb_margin_reward / lpb_tau` onto each
    transition's `info` dict.

    `failure_scores` may be slightly shorter than `transitions` because the
    LPB encoder truncates to the min length across modalities; we repeat the
    last available score to pad. This is rare (off-by-one), but happens for
    some HDF5 demos with mismatched action/state/image counts.
    """
    if not transitions:
        return
    n = len(transitions)
    scores = np.asarray(failure_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError(
            "annotate_transitions_with_lpb_scores got empty failure_scores."
        )
    if scores.size < n:
        # Pad by repeating the last score (encoder min-length truncation case).
        pad = np.full((n - scores.size,), float(scores[-1]), dtype=np.float32)
        scores = np.concatenate([scores, pad], axis=0)
    elif scores.size > n:
        scores = scores[:n]
    margins = float(tau) - scores
    intrinsics = lpb_disc_intrinsic_from_failure_score(scores, tau)

    for trans, fail, margin, intrinsic in zip(transitions, scores, margins, intrinsics):
        info = dict(trans.info or {})
        info["lpb_failure_score"] = float(fail)
        info["lpb_margin_reward"] = float(margin)
        info["lpb_disc_intrinsic"] = float(intrinsic)
        info["lpb_tau"] = float(tau)
        trans.info = info
