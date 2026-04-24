"""D4 benchmark adapter for the cached-image LPB-parity refactor."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from data.utils.benchmark import BenchmarkTrajectory, DiscriminatorOutput

from ..data.cache import PreprocessedCacheReader
from ..models.dynamics import ConditionalDynamicsPredictor
from ..models.encoder import Encoder
from .detector import D4Detector
from .feature import D4FeatureExtractor, D4Frames, D4LPBFeatureExtractor
from robosuite.discriminator.lpb.knn_discriminator import AdaptiveKNNDiscriminator


_VALID_SCORE_MODES = ("rel", "abs", "knn")


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


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class D4BenchmarkDiscriminator:
    name = "d4_disc"

    def __init__(
        self,
        *,
        d4_ckpt_path: str,
        preprocessed_cache_root: str = "data/.lpb_score_preprocessed_cache",
        device: str = "cuda",
        encoder_batch_size: int = 256,
        image_size: int = 128,
        encoder_pretrained: bool = False,
        encoder_freeze: bool = True,
        omega: float = 0.5,
        sigma_sq: Optional[float] = None,
        horizon: int = 1,
        delta: float = 10.0,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        calib_fraction: float = 0.2,
        seed: int = 0,
        per_task_calibration: bool = True,
        scoring_batch_size: int = 512,
        verbose_fit: bool = True,
        score_mode: str = "knn",
        knn_chunk_size: int = 8192,
        knn_normalize_feature: bool = True,
    ) -> None:
        """Load a D4 checkpoint and expose a `FailureBenchmark` discriminator API.

        Modes:
            - score_mode in {"rel","abs"}: use `D4Detector` (CFG scoring) and
              conformal calibration tau via held-out success trajectories.
            - score_mode == "knn": bypass CFG and instead run LPB-style adaptive
              KNN on features extracted from the D4 predictor's input projections.

        Important:
            - For stable inference we always load `ema_state_dict` (not raw weights).
            - The checkpoint is expected to be the 512-d LPB-parity refactor.
        """
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0,1), got {calib_fraction}")
        if str(score_mode) not in _VALID_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {_VALID_SCORE_MODES}, got {score_mode!r}")

        self.d4_ckpt_path = str(d4_ckpt_path)
        self.preprocessed_cache_root = str(preprocessed_cache_root)
        self.device = str(device)
        self.omega = float(omega)
        self.delta = float(delta)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.per_task_calibration = bool(per_task_calibration)
        self.verbose_fit = bool(verbose_fit)
        self.horizon = int(horizon)
        self.score_mode = str(score_mode)
        self.knn_chunk_size = int(knn_chunk_size)
        self.knn_normalize_feature = bool(knn_normalize_feature)

        payload = _torch_load(self.d4_ckpt_path)
        if "encoder_state_dict" not in payload:
            raise ValueError(
                f"D4 ckpt missing 'encoder_state_dict': {self.d4_ckpt_path}. "
                "Old 256-d flow-latent checkpoints are not supported."
            )
        for required in ("arch_args", "ema_state_dict"):
            if required not in payload:
                raise ValueError(f"D4 ckpt missing key {required!r}: {self.d4_ckpt_path}")

        arch_args = dict(payload["arch_args"])
        arch_args.setdefault("residual_latent_head", False)
        if int(arch_args.get("latent_dim", -1)) != 512:
            raise ValueError(
                f"Unsupported D4 checkpoint latent_dim={arch_args.get('latent_dim')}. "
                "Only the refactored 512-d LPB-parity checkpoints are supported."
            )
        self.arch_args = arch_args

        encoder_cfg = dict(payload.get("encoder_config", {}) or {})
        self.encoder = Encoder(
            checkpoint_path=None,
            pretrained=bool(encoder_pretrained),
            freeze=bool(encoder_freeze),
            normalize_input=bool(encoder_cfg.get("normalize_input", True)),
        )
        self.encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.predictor = ConditionalDynamicsPredictor(**arch_args)
        # Inference always uses EMA weights for stability and best validation metrics.
        self.predictor.load_state_dict(payload["ema_state_dict"], strict=True)
        self.predictor.eval()
        for param in self.predictor.parameters():
            param.requires_grad = False

        self.cache_reader = PreprocessedCacheReader(
            cache_root=self.preprocessed_cache_root,
            image_size=int(image_size),
            camera_index=0,
        )

        ckpt_sigma = float(payload.get("sigma_sq", 0.5))
        self.sigma_sq = float(sigma_sq) if sigma_sq is not None else ckpt_sigma

        ckpt_cfg = payload.get("config", {}) or {}
        ckpt_horizon = int(ckpt_cfg.get("horizon", self.horizon))
        self.horizon = int(ckpt_horizon) if self.horizon == 1 else int(self.horizon)
        proprio_indices = ckpt_cfg.get("proprio_indices")

        self.feature_extractor = D4FeatureExtractor(
            encoder=self.encoder,
            cache_reader=self.cache_reader,
            latent_dim=int(arch_args["latent_dim"]),
            proprio_dim=int(arch_args["proprio_dim"]),
            action_dim=int(arch_args["action_dim"]),
            action_horizon=int(arch_args["max_action_horizon"]),
            proprio_indices=None if proprio_indices is None else list(proprio_indices),
            encoder_batch_size=int(encoder_batch_size),
            device=self.device,
        )
        # KNN path shares the D4Frames pipeline but bypasses the CFG detector;
        # in that mode D4Detector is not constructed (omega/sigma_sq unused).
        self.detector: Optional[D4Detector]
        self.lpb_feature_extractor: Optional[D4LPBFeatureExtractor]
        if self.score_mode == "knn":
            self.detector = None
            self.lpb_feature_extractor = D4LPBFeatureExtractor(
                frame_extractor=self.feature_extractor,
                predictor=self.predictor,
                device=self.device,
                batch_size=int(scoring_batch_size),
                normalize_feature=self.knn_normalize_feature,
            )
        else:
            self.detector = D4Detector(
                self.predictor,
                omega=self.omega,
                sigma_sq=self.sigma_sq,
                delta=self.delta,
                lambda_mode=self.lambda_mode,
                lambda_window_size=self.lambda_window_size,
                device=self.device,
                batch_size=int(scoring_batch_size),
                score_mode=self.score_mode,
            )
            self.lpb_feature_extractor = None

        self._tau_per_task: dict[str, float] = {}
        self._calibration_stats: dict[str, dict[str, Any]] = {}
        self._frames_cache: dict[tuple[str, str], D4Frames] = {}
        # KNN per-task detector + per-trajectory feature cache.
        self._knn_per_task: dict[str, AdaptiveKNNDiscriminator] = {}
        self._knn_feature_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        return (str(trajectory.file_path), str(trajectory.demo_path))

    def _source_ids(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        src_file = trajectory.source_hdf5_path or trajectory.file_path
        src_key = trajectory.source_demo_key or trajectory.demo_path.split("/")[-1]
        return str(src_file), str(src_key)

    def has_preprocessed_cache(self, trajectory: BenchmarkTrajectory) -> bool:
        src_file, src_key = self._source_ids(trajectory)
        return self.cache_reader.exists(str(trajectory.task_name), src_file, src_key)

    def filter_cached_trajectories(
        self,
        trajectories: list[BenchmarkTrajectory],
    ) -> tuple[list[BenchmarkTrajectory], dict[str, dict[str, int]]]:
        kept: list[BenchmarkTrajectory] = []
        stats: dict[str, dict[str, int]] = {}

        for traj in trajectories:
            task = str(traj.task_name)
            bucket = stats.setdefault(
                task,
                {
                    "kept_success": 0,
                    "kept_fail": 0,
                    "skipped_success": 0,
                    "skipped_fail": 0,
                },
            )
            is_fail = bool(traj.is_failure)
            if self.has_preprocessed_cache(traj):
                kept.append(traj)
                bucket["kept_fail" if is_fail else "kept_success"] += 1
            else:
                bucket["skipped_fail" if is_fail else "skipped_success"] += 1

        return kept, stats

    def _knn_feature(self, trajectory: BenchmarkTrajectory) -> torch.Tensor:
        assert self.lpb_feature_extractor is not None
        key = self._trajectory_key(trajectory)
        cached = self._knn_feature_cache.get(key)
        if cached is not None:
            return cached
        frames = self._frames(trajectory)
        feat = self.lpb_feature_extractor.encode_frames(frames)
        self._knn_feature_cache[key] = feat
        return feat

    def _frames(self, trajectory: BenchmarkTrajectory) -> D4Frames:
        key = self._trajectory_key(trajectory)
        cached = self._frames_cache.get(key)
        if cached is not None:
            return cached
        src_file, src_key = self._source_ids(trajectory)
        if not self.cache_reader.exists(str(trajectory.task_name), src_file, src_key):
            raise FileNotFoundError(
                f"No preprocessed cache for task={trajectory.task_name} file={src_file} demo={src_key}"
            )
        frames = self.feature_extractor.extract(
            task_name=str(trajectory.task_name),
            source_file_path=src_file,
            source_demo_key=src_key,
            trajectory_length=int(trajectory.num_frames),
            horizon=self.horizon,
        )
        self._frames_cache[key] = frames
        return frames

    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None:
        """Fit per-task (or pooled) conformal thresholds on success trajectories.

        Calibration splits each task's success trajectories into:
            - bank set: used to build the "expert" reference (KNN mode only)
            - calibration set: used to pick tau at the `delta` percentile
        """
        succ_per_task: dict[str, list[BenchmarkTrajectory]] = {}
        fail_per_task: dict[str, list[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            bucket = fail_per_task if bool(traj.is_failure) else succ_per_task
            bucket.setdefault(str(traj.task_name), []).append(traj)

        if not succ_per_task:
            raise RuntimeError("D4-Disc needs success trajectories for calibration tau.")

        rng = np.random.default_rng(self.seed)
        bank_succ_per_task: dict[str, list[BenchmarkTrajectory]] = {}
        calib_succ_per_task: dict[str, list[BenchmarkTrajectory]] = {}
        for task, succ_list in succ_per_task.items():
            if len(succ_list) < 2:
                raise RuntimeError(f"Task {task!r} has only {len(succ_list)} success trajectory; need >= 2.")
            perm = rng.permutation(len(succ_list))
            n_calib = max(1, min(len(succ_list) - 1, int(round(self.calib_fraction * len(succ_list)))))
            calib_set = set(perm[:n_calib].tolist())
            bank_succ_per_task[task] = [t for i, t in enumerate(succ_list) if i not in calib_set]
            calib_succ_per_task[task] = [t for i, t in enumerate(succ_list) if i in calib_set]

        tasks_seen = sorted(set(bank_succ_per_task) | set(fail_per_task))

        if self.score_mode == "knn":
            if not self.per_task_calibration:
                raise ValueError(
                    "score_mode='knn' requires per_task_calibration=True "
                    "(each task owns its own expert bank, matching LPB)."
                )
            for task in tasks_seen:
                bank_list = bank_succ_per_task.get(task, [])
                calib_list = calib_succ_per_task.get(task, [])
                if not bank_list or not calib_list:
                    if self.verbose_fit:
                        print(f"[d4_disc][fit] task={task} SKIP knn — insufficient success trajectories")
                    continue
                bank_feats = [self._knn_feature(t) for t in bank_list]
                calib_feats = [self._knn_feature(t) for t in calib_list]
                knn = AdaptiveKNNDiscriminator(
                    delta=self.delta,
                    delta_step=1.0,
                    knn_chunk_size=self.knn_chunk_size,
                    lambda_mode=self.lambda_mode,
                    lambda_window_size=self.lambda_window_size,
                    aux_weight=0.0,
                    device=self.device,
                )
                threshold = knn.fit(
                    expert_sequences=bank_feats,
                    calibration_sequences=calib_feats,
                    calibration_aux=None,
                )
                self._knn_per_task[task] = knn
                bank_steps = int(sum(int(f.shape[0]) for f in bank_feats))
                calib_steps = int(sum(int(f.shape[0]) for f in calib_feats))
                self._calibration_stats[task] = {
                    "num_success_trajectories": int(len(succ_per_task.get(task, []))),
                    "num_bank_trajectories": int(len(bank_list)),
                    "num_calib_trajectories": int(len(calib_list)),
                    "num_fail_trajectories": int(len(fail_per_task.get(task, []))),
                    "num_bank_steps": bank_steps,
                    "num_calib_steps": calib_steps,
                    "threshold": float(threshold),
                    "delta": self.delta,
                    "score_mode": "knn",
                    "feature_dim": int(self.lpb_feature_extractor.feature_dim),
                }
                if self.verbose_fit:
                    print(
                        f"[d4_disc][fit] task={task} mode=knn "
                        f"bank_trajs={len(bank_list)} ({bank_steps} steps)  "
                        f"calib_trajs={len(calib_list)} ({calib_steps} steps)  "
                        f"tau={threshold:.6f}"
                    )
            return

        if self.per_task_calibration:
            for task in tasks_seen:
                calib_list = calib_succ_per_task.get(task, [])
                if not calib_list:
                    if self.verbose_fit:
                        print(f"[d4_disc][fit] task={task} SKIP tau — no calibration trajectories")
                    continue
                calib_frames = [self._frames(t) for t in calib_list]
                tau = self.detector.calibrate(calib_frames)
                self._tau_per_task[task] = float(tau)
                self._calibration_stats[task] = {
                    "num_success_trajectories": int(len(succ_per_task.get(task, []))),
                    "num_bank_trajectories": int(len(bank_succ_per_task.get(task, []))),
                    "num_calib_trajectories": int(len(calib_list)),
                    "num_fail_trajectories": int(len(fail_per_task.get(task, []))),
                    "threshold": float(tau),
                    "omega": self.omega,
                    "sigma_sq": self.sigma_sq,
                    "delta": self.delta,
                }
                if self.verbose_fit:
                    print(f"[d4_disc][fit] task={task}  calib_trajs={len(calib_list)}  tau={tau:.6f}")
        else:
            pooled_calib: list[D4Frames] = []
            for calib_list in calib_succ_per_task.values():
                pooled_calib.extend(self._frames(t) for t in calib_list)
            if not pooled_calib:
                raise RuntimeError("no calibration trajectories found")
            tau = self.detector.calibrate(pooled_calib)
            for task in tasks_seen:
                self._tau_per_task[task] = float(tau)
                self._calibration_stats[task] = {
                    "num_success_trajectories": int(len(succ_per_task.get(task, []))),
                    "num_bank_trajectories": int(len(bank_succ_per_task.get(task, []))),
                    "num_calib_trajectories": int(len(calib_succ_per_task.get(task, []))),
                    "num_fail_trajectories": int(len(fail_per_task.get(task, []))),
                    "threshold": float(tau),
                    "omega": self.omega,
                    "sigma_sq": self.sigma_sq,
                    "delta": self.delta,
                }
            if self.verbose_fit:
                print(f"[d4_disc][fit] shared tau={tau:.6f} across {len(tasks_seen)} tasks")

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        """Score one trajectory and return per-frame predictions + aux diagnostics."""
        task = str(trajectory.task_name)
        T = int(trajectory.num_frames)

        if self.score_mode == "knn":
            knn = self._knn_per_task.get(task)
            if knn is None:
                raise KeyError(
                    f"Task {task!r} not calibrated (knn). Available: {sorted(self._knn_per_task)}"
                )
            feat = self._knn_feature(trajectory)
            det = knn.detect_sequence(feat, labels=None, adaptive_delta=False)

            step_scores = _pad_to_length(det.lambda_values, T, dtype=np.float32)
            predictions = _pad_to_length(det.preds, T, dtype=np.int64).astype(np.int64)
            thresholds = _pad_to_length(det.thresholds, T, dtype=np.float32)
            raw_step = _pad_to_length(det.step_scores, T, dtype=np.float32)

            positive = np.where(predictions == 1)[0]
            first_failure_frame = int(positive[0]) if positive.size > 0 else None

            aux_knn: dict[str, Any] = {
                "task": task,
                "threshold": float(knn.threshold) if knn.threshold is not None else float("nan"),
                "delta": float(knn.delta),
                "aggregated_lambda": step_scores,
                "raw_step_scores": raw_step,
                "thresholds": thresholds,
                "lambda_mode": self.lambda_mode,
                "lambda_window_size": self.lambda_window_size,
                "feature_len": int(feat.shape[0]),
                "feature_dim": int(feat.shape[1]),
                "score_mode": "knn",
            }
            return DiscriminatorOutput(
                step_scores=step_scores,
                predictions=predictions,
                first_failure_frame=first_failure_frame,
                aux=aux_knn,
            )

        tau = self._tau_per_task.get(task)
        if tau is None:
            raise KeyError(f"Task {task!r} not calibrated. Available: {sorted(self._tau_per_task)}")

        assert self.detector is not None
        frames = self._frames(trajectory)
        result = self.detector.score(frames, tau=float(tau))

        T = int(trajectory.num_frames)
        step_scores = _pad_to_length(result.lambda_values, T, dtype=np.float32)
        aggregated_lambda = _pad_to_length(result.lambda_values, T, dtype=np.float32)
        predictions = _pad_to_length(result.preds, T, dtype=np.int64).astype(np.int64)
        thresholds = _pad_to_length(result.thresholds, T, dtype=np.float32)
        raw_step = _pad_to_length(result.step_scores, T, dtype=np.float32)
        r_plus_padded = _pad_to_length(result.d_pos_sq, T, dtype=np.float32)
        r_minus_padded = _pad_to_length(result.d_neg_sq, T, dtype=np.float32) if result.d_neg_sq is not None else None

        positive = np.where(predictions == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: dict[str, Any] = {
            "task": task,
            "threshold": float(tau),
            "omega": self.omega,
            "sigma_sq": self.sigma_sq,
            "aggregated_lambda": aggregated_lambda,
            "raw_step_scores": raw_step,
            "thresholds": thresholds,
            "d_pos_sq": r_plus_padded,
            "d_neg_sq": r_minus_padded,
            "r_plus": r_plus_padded,
            "r_minus": r_minus_padded,
            "lambda_mode": self.lambda_mode,
            "lambda_window_size": self.lambda_window_size,
            "feature_len": int(frames.length),
        }
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=predictions,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "d4_ckpt_path": self.d4_ckpt_path,
            "preprocessed_cache_root": self.preprocessed_cache_root,
            "omega": self.omega,
            "sigma_sq": self.sigma_sq,
            "delta": self.delta,
            "lambda_mode": self.lambda_mode,
            "lambda_window_size": self.lambda_window_size,
            "calib_fraction": self.calib_fraction,
            "per_task_calibration": self.per_task_calibration,
            "seed": self.seed,
            "arch_args": dict(self.arch_args),
            "horizon": self.horizon,
            "score_mode": self.score_mode,
            "knn_chunk_size": self.knn_chunk_size if self.score_mode == "knn" else None,
            "knn_normalize_feature": self.knn_normalize_feature if self.score_mode == "knn" else None,
        }

    def close(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
