"""D4-Disc adapter for ``data.utils.benchmark.FailureBenchmark``.

Mirrors ``D3BenchmarkDiscriminator`` in surface area, so the same
``run_*.py`` benchmark driver + visualizer APIs apply. Differences:

- A single trained ConditionalDynamicsPredictor (loaded from the
  ``ema_state_dict`` of the checkpoint) replaces the KDE/KNN banks.
- There is no fail bank at inference: fail information is already
  distilled into the c=- branch of the critic via phase-B training.
- omega is read at benchmark time (no retraining for the omega sweep).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from data.utils.benchmark import BenchmarkTrajectory, DiscriminatorOutput

from robosuite.discriminator.d3disc.encoder import FlowMultiEncoderWrapper

from .detector import D4Detector
from .dynamics_feature import D4FeatureExtractor, D4Frames
from .model import ConditionalDynamicsPredictor


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
        policy_ckpt_path: str,
        d4_ckpt_path: str,
        cache_root: str = "data/.lpb_score_cache",
        device: str = "cuda",
        encoder_batch_size: int = 256,
        image_size: int = 128,
        # D4 hyperparameters at inference time.
        omega: float = 0.5,
        sigma_sq: Optional[float] = None,
        horizon: int = 1,
        # Calibration.
        delta: float = 10.0,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        calib_fraction: float = 0.2,
        seed: int = 0,
        per_task_calibration: bool = True,
        # Inference batching.
        scoring_batch_size: int = 512,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0,1), got {calib_fraction}")

        self.policy_ckpt_path = str(policy_ckpt_path)
        self.d4_ckpt_path = str(d4_ckpt_path)
        self.cache_root = str(cache_root)
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

        self.encoder = FlowMultiEncoderWrapper(
            policy_ckpt_path=self.policy_ckpt_path,
            cache_root=self.cache_root,
            device=self.device,
            image_size=int(image_size),
            encoder_batch_size=int(encoder_batch_size),
        )

        payload = _torch_load(self.d4_ckpt_path)
        for required in ("arch_args", "ema_state_dict"):
            if required not in payload:
                raise ValueError(
                    f"D4 ckpt missing key {required!r}: {self.d4_ckpt_path}"
                )
        arch_args = dict(payload["arch_args"])
        # Ckpts trained before the residual_latent_head flag existed used the
        # non-residual head. Default to False for backward compatibility —
        # new ckpts record the flag explicitly.
        arch_args.setdefault("residual_latent_head", False)
        self.arch_args = arch_args
        self.predictor = ConditionalDynamicsPredictor(**arch_args)
        self.predictor.load_state_dict(payload["ema_state_dict"], strict=True)
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad = False

        # sigma_sq default: inherit from checkpoint unless overridden.
        ckpt_sigma = float(payload.get("sigma_sq", 0.5))
        self.sigma_sq = float(sigma_sq) if sigma_sq is not None else ckpt_sigma

        # Inference horizon: prefer trainer horizon recorded in ckpt config.
        ckpt_cfg = payload.get("config", {}) or {}
        ckpt_horizon = int(ckpt_cfg.get("horizon", self.horizon))
        self.horizon = int(ckpt_horizon) if self.horizon == 1 else int(self.horizon)

        self.feature_extractor = D4FeatureExtractor(
            encoder=self.encoder,
            latent_dim=int(arch_args["latent_dim"]),
            proprio_dim=int(arch_args["proprio_dim"]),
            action_dim=int(arch_args["action_dim"]),
            action_horizon=int(arch_args["max_action_horizon"]),
        )

        self.detector = D4Detector(
            self.predictor,
            omega=self.omega,
            sigma_sq=self.sigma_sq,
            delta=self.delta,
            lambda_mode=self.lambda_mode,
            lambda_window_size=self.lambda_window_size,
            device=self.device,
            batch_size=int(scoring_batch_size),
        )

        self._tau_per_task: dict[str, float] = {}
        self._calibration_stats: dict[str, dict[str, Any]] = {}
        self._frames_cache: dict[tuple[str, str], D4Frames] = {}

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        return (str(trajectory.file_path), str(trajectory.demo_path))

    def _source_ids(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        src_file = trajectory.source_hdf5_path or trajectory.file_path
        src_key = trajectory.source_demo_key or trajectory.demo_path.split("/")[-1]
        return str(src_file), str(src_key)

    def _frames(self, trajectory: BenchmarkTrajectory) -> D4Frames:
        key = self._trajectory_key(trajectory)
        cached = self._frames_cache.get(key)
        if cached is not None:
            return cached
        src_file, src_key = self._source_ids(trajectory)
        states = np.asarray(trajectory.load_states(), dtype=np.float32)
        actions = np.asarray(trajectory.load_actions(), dtype=np.float32)
        frames = self.feature_extractor.extract(
            task_name=str(trajectory.task_name),
            source_file_path=src_file,
            source_demo_key=src_key,
            states=states,
            actions=actions,
            horizon=self.horizon,
        )
        self._frames_cache[key] = frames
        return frames

    # ------------------------------------------------------------------ #
    # Fit / score                                                        #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None:
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
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; need >= 2."
                )
            perm = rng.permutation(len(succ_list))
            n_calib = max(
                1, min(len(succ_list) - 1, int(round(self.calib_fraction * len(succ_list))))
            )
            calib_set = set(perm[:n_calib].tolist())
            bank_succ_per_task[task] = [t for i, t in enumerate(succ_list) if i not in calib_set]
            calib_succ_per_task[task] = [t for i, t in enumerate(succ_list) if i in calib_set]

        tasks_seen = sorted(set(bank_succ_per_task) | set(fail_per_task))

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
                    print(
                        f"[d4_disc][fit] task={task}  calib_trajs={len(calib_list)}  tau={tau:.6f}"
                    )
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
        task = str(trajectory.task_name)
        tau = self._tau_per_task.get(task)
        if tau is None:
            raise KeyError(
                f"Task {task!r} not calibrated. Available: {sorted(self._tau_per_task)}"
            )

        frames = self._frames(trajectory)
        result = self.detector.score(frames, tau=float(tau))

        T = int(trajectory.num_frames)
        # step_scores = raw per-frame λ (not the prefix/rolling-aggregated view).
        # The benchmark pipeline re-aggregates for trajectory score and uses
        # step_scores directly for per-frame AUROC and for
        # `_materialize_predictions(success_calibrated)`; feeding it the
        # already-aggregated signal double-aggregates at the trajectory stage
        # (harmless only in "max+full-prefix" mode) and makes frame AUROC
        # structurally inflated via prefix-monotone ordering.
        step_scores = _pad_to_length(result.step_scores, T, dtype=np.float32)
        aggregated_lambda = _pad_to_length(result.lambda_values, T, dtype=np.float32)
        # predictions must stay consistent with step_scores: threshold the raw
        # per-frame λ rather than the aggregated λ. For full-prefix modes this
        # yields the same first-firing time the detector previously reported.
        predictions = (step_scores >= float(tau)).astype(np.int64)
        thresholds = _pad_to_length(result.thresholds, T, dtype=np.float32)
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
            "thresholds": thresholds,
            # Map D4 residuals into the D3 aux schema so D3's viz is reusable.
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
            "policy_ckpt_path": self.policy_ckpt_path,
            "d4_ckpt_path": self.d4_ckpt_path,
            "cache_root": self.cache_root,
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
        }

    def close(self) -> None:
        try:
            self.encoder.close()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
