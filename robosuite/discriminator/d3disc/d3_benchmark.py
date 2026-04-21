"""D3-Disc adapter for the shared ``data.utils.benchmark`` framework.

Mirrors ``LPBBenchmarkDiscriminator`` in surface area so ``FailureBenchmark``
can use either one interchangeably. Differences from LPB:

- Features come from the frozen flow_multi policy encoder (256-D), reusing
  the pre-computed ``data/.lpb_score_cache`` NPZs.
- A single success bank + (optionally) a single F3-cleaned fail bank are
  shared across all tasks (D8), while tau is calibrated per task.
- Fail rollouts are incorporated only when ``omega > 0``; PandaLift has no
  fail_rollout and degrades gracefully.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from data.utils.benchmark import BenchmarkTrajectory, DiscriminatorOutput

from .detector import D3Detector
from .dynamics_feature import D3FeatureExtractor
from .encoder import FlowMultiEncoderWrapper


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


class D3BenchmarkDiscriminator:
    """D3-Disc plugged into ``FailureBenchmark.evaluate(...)``."""

    name = "d3_disc"

    def __init__(
        self,
        *,
        policy_ckpt_path: str,
        cache_root: str = "data/.lpb_score_cache",
        device: str = "cuda",
        encoder_batch_size: int = 256,
        image_size: int = 128,
        # Dynamics feature extractor (optional): if provided, scoring uses
        # LPB-style projected feature [obs_proj(z), proprio_proj(s), mean(action_proj(a))].
        dynamics_ckpt_path: Optional[str] = None,
        normalize_feature: bool = True,
        # D3 hyperparameters.
        omega: float = 0.5,
        k: int = 1,
        beta: Optional[float] = None,
        kappa: Optional[float] = None,
        sigma_sq: float = 0.5,
        # Threshold calibration.
        delta: float = 10.0,
        knn_chunk_size: int = 8192,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        calib_fraction: float = 0.2,
        # Miscellaneous.
        seed: int = 0,
        share_banks_across_tasks: bool = True,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0,1), got {calib_fraction}")

        self.policy_ckpt_path = str(policy_ckpt_path)
        self.cache_root = str(cache_root)
        self.device = str(device)
        self.omega = float(omega)
        self.k = int(k)
        self.beta = None if beta is None else float(beta)
        self.kappa = None if kappa is None else float(kappa)
        self.sigma_sq = float(sigma_sq)
        self.delta = float(delta)
        self.knn_chunk_size = int(knn_chunk_size)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.share_banks_across_tasks = bool(share_banks_across_tasks)
        self.verbose_fit = bool(verbose_fit)

        self.encoder = FlowMultiEncoderWrapper(
            policy_ckpt_path=self.policy_ckpt_path,
            cache_root=self.cache_root,
            device=self.device,
            image_size=int(image_size),
            encoder_batch_size=int(encoder_batch_size),
        )

        self.dynamics_ckpt_path = None if dynamics_ckpt_path in (None, "") else str(dynamics_ckpt_path)
        self.normalize_feature = bool(normalize_feature)
        self.feature_extractor: Optional[D3FeatureExtractor] = None
        if self.dynamics_ckpt_path is not None:
            self.feature_extractor = D3FeatureExtractor(
                dynamics_ckpt_path=self.dynamics_ckpt_path,
                encoder=self.encoder,
                device=self.device,
                normalize_feature=self.normalize_feature,
            )

        # Detector(s) + calibration state.
        self._detectors_per_task: dict[str, D3Detector] = {}
        self._tau_per_task: dict[str, float] = {}
        self._calibration_stats: dict[str, dict[str, Any]] = {}
        self._feature_cache: dict[tuple[str, str], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        return (str(trajectory.file_path), str(trajectory.demo_path))

    def _source_ids(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        """Return (source_hdf5_path, source_demo_key) used for cache key."""
        src_file = trajectory.source_hdf5_path or trajectory.file_path
        src_key = trajectory.source_demo_key or trajectory.demo_path.split("/")[-1]
        return str(src_file), str(src_key)

    def _encode(self, trajectory: BenchmarkTrajectory) -> torch.Tensor:
        key = self._trajectory_key(trajectory)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        src_file, src_key = self._source_ids(trajectory)

        if self.feature_extractor is not None:
            # Dynamics-projected feature: needs states + actions alongside latents.
            states = np.asarray(trajectory.load_states(), dtype=np.float32)
            actions = np.asarray(trajectory.load_actions(), dtype=np.float32)
            feat = self.feature_extractor.extract_from_trajectory(
                task_name=str(trajectory.task_name),
                source_file_path=src_file,
                source_demo_key=src_key,
                states=states,
                actions=actions,
            )
        else:
            # Raw flow_multi z_t (no dynamics projection).
            encoded = self.encoder.load_or_encode_demo(
                task_name=str(trajectory.task_name),
                file_path=src_file,
                demo_key=src_key,
            )
            feat = torch.from_numpy(np.asarray(encoded.latents, dtype=np.float32))

        self._feature_cache[key] = feat
        return feat

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
            raise RuntimeError("D3-Disc needs success trajectories to build the positive bank.")

        rng = np.random.default_rng(self.seed)

        # Per-task success split: (bank, calib).
        bank_succ_per_task: dict[str, list[BenchmarkTrajectory]] = {}
        calib_succ_per_task: dict[str, list[BenchmarkTrajectory]] = {}
        for task, succ_list in succ_per_task.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; need >= 2."
                )
            perm = rng.permutation(len(succ_list))
            n_calib = max(1, min(len(succ_list) - 1, int(round(self.calib_fraction * len(succ_list)))))
            calib_set = set(perm[:n_calib].tolist())
            bank_succ_per_task[task] = [t for i, t in enumerate(succ_list) if i not in calib_set]
            calib_succ_per_task[task] = [t for i, t in enumerate(succ_list) if i in calib_set]

        # Encode everything once. Pre-warm cache for visibility.
        bank_succ_feats: dict[str, list[torch.Tensor]] = {
            task: [self._encode(t) for t in trajs] for task, trajs in bank_succ_per_task.items()
        }
        calib_succ_feats: dict[str, list[torch.Tensor]] = {
            task: [self._encode(t) for t in trajs] for task, trajs in calib_succ_per_task.items()
        }
        fail_feats: dict[str, list[torch.Tensor]] = {
            task: [self._encode(t) for t in trajs] for task, trajs in fail_per_task.items()
        }

        tasks_seen = sorted(set(bank_succ_per_task) | set(fail_per_task))

        if self.share_banks_across_tasks:
            # Single shared D+ / D-, per-task tau.
            pooled_pos = [f for feats in bank_succ_feats.values() for f in feats]
            pooled_neg: list[torch.Tensor] = (
                [f for feats in fail_feats.values() for f in feats] if self.omega > 0.0 else []
            )
            detector = D3Detector(
                omega=self.omega,
                k=self.k,
                beta=self.beta,
                kappa=self.kappa,
                sigma_sq=self.sigma_sq,
                delta=self.delta,
                knn_chunk_size=self.knn_chunk_size,
                lambda_mode=self.lambda_mode,
                lambda_window_size=self.lambda_window_size,
                device=self.device,
            )
            detector.fit_banks(
                pos_sequences=pooled_pos,
                neg_sequences=pooled_neg if pooled_neg else None,
            )
            if self.verbose_fit:
                print(
                    f"[d3_disc][fit] SHARED banks  pos_steps={detector.num_pos_used}  "
                    f"neg_steps={detector.num_neg_used}  omega={self.omega}  k={self.k}  "
                    f"beta={detector.beta_used}  kappa={detector.kappa_used}"
                )

            for task in tasks_seen:
                calib_seqs = calib_succ_feats.get(task, [])
                if len(calib_seqs) == 0:
                    if self.verbose_fit:
                        print(f"[d3_disc][fit] task={task} SKIP tau — no calibration trajectories")
                    continue
                tau = detector.calibrate(calib_seqs)
                self._detectors_per_task[task] = detector  # same reference is fine
                self._tau_per_task[task] = tau
                self._calibration_stats[task] = {
                    "num_success_trajectories": int(len(succ_per_task.get(task, []))),
                    "num_bank_trajectories": int(len(bank_succ_per_task.get(task, []))),
                    "num_calib_trajectories": int(len(calib_seqs)),
                    "num_fail_trajectories": int(len(fail_per_task.get(task, []))),
                    "threshold": float(tau),
                    "omega": self.omega,
                    "k": self.k,
                    "beta_used": float(detector.beta_used) if detector.beta_used is not None else None,
                    "kappa_used": float(detector.kappa_used) if detector.kappa_used is not None else None,
                    "num_pos_bank_steps": int(detector.num_pos_used),
                    "num_neg_bank_steps": int(detector.num_neg_used),
                }
                if self.verbose_fit:
                    print(
                        f"[d3_disc][fit] task={task}  calib_trajs={len(calib_seqs)}  tau={tau:.6f}"
                    )
        else:
            # Fallback: per-task banks. Useful as a degraded baseline/ablation.
            for task in tasks_seen:
                if task not in bank_succ_feats:
                    continue
                detector = D3Detector(
                    omega=self.omega,
                    k=self.k,
                    beta=self.beta,
                    kappa=self.kappa,
                    sigma_sq=self.sigma_sq,
                    delta=self.delta,
                    knn_chunk_size=self.knn_chunk_size,
                    lambda_mode=self.lambda_mode,
                    lambda_window_size=self.lambda_window_size,
                    device=self.device,
                )
                detector.fit_banks(
                    pos_sequences=bank_succ_feats[task],
                    neg_sequences=fail_feats.get(task, None) if self.omega > 0.0 else None,
                )
                calib_seqs = calib_succ_feats.get(task, [])
                if len(calib_seqs) == 0:
                    if self.verbose_fit:
                        print(f"[d3_disc][fit] task={task} SKIP tau — no calibration trajectories")
                    continue
                tau = detector.calibrate(calib_seqs)
                self._detectors_per_task[task] = detector
                self._tau_per_task[task] = tau
                self._calibration_stats[task] = {
                    "num_success_trajectories": int(len(succ_per_task.get(task, []))),
                    "num_bank_trajectories": int(len(bank_succ_per_task.get(task, []))),
                    "num_calib_trajectories": int(len(calib_seqs)),
                    "num_fail_trajectories": int(len(fail_per_task.get(task, []))),
                    "threshold": float(tau),
                    "omega": self.omega,
                    "k": self.k,
                    "beta_used": float(detector.beta_used) if detector.beta_used is not None else None,
                    "kappa_used": float(detector.kappa_used) if detector.kappa_used is not None else None,
                    "num_pos_bank_steps": int(detector.num_pos_used),
                    "num_neg_bank_steps": int(detector.num_neg_used),
                }
                if self.verbose_fit:
                    print(
                        f"[d3_disc][fit] task={task}  pos_steps={detector.num_pos_used}  "
                        f"neg_steps={detector.num_neg_used}  tau={tau:.6f}"
                    )

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        detector = self._detectors_per_task.get(task)
        tau = self._tau_per_task.get(task)
        if detector is None or tau is None:
            raise KeyError(
                f"Task {task!r} not calibrated. Available: {sorted(self._tau_per_task)}"
            )

        feat = self._encode(trajectory)
        result = detector.score(feat, tau=float(tau))

        T = int(trajectory.num_frames)
        step_scores = _pad_to_length(result.lambda_values, T, dtype=np.float32)
        predictions = _pad_to_length(result.preds, T, dtype=np.int64).astype(np.int64)
        thresholds = _pad_to_length(result.thresholds, T, dtype=np.float32)
        raw_step = _pad_to_length(result.step_scores, T, dtype=np.float32)
        d_pos_sq_padded = _pad_to_length(result.d_pos_sq, T, dtype=np.float32)
        d_neg_sq_padded = (
            None if result.d_neg_sq is None else _pad_to_length(result.d_neg_sq, T, dtype=np.float32)
        )

        positive = np.where(predictions == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: dict[str, Any] = {
            "task": task,
            "threshold": float(tau),
            "omega": self.omega,
            "k": self.k,
            "beta_used": self._calibration_stats.get(task, {}).get("beta_used"),
            "kappa_used": self._calibration_stats.get(task, {}).get("kappa_used"),
            "raw_step_scores": raw_step,
            "thresholds": thresholds,
            "d_pos_sq": d_pos_sq_padded,
            "d_neg_sq": d_neg_sq_padded,
            "feature_len": int(feat.shape[0]),
            "lambda_mode": self.lambda_mode,
            "lambda_window_size": self.lambda_window_size,
        }
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=predictions,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        summary = {
            "per_task": dict(self._calibration_stats),
            "policy_ckpt_path": self.policy_ckpt_path,
            "cache_root": self.cache_root,
            "omega": self.omega,
            "k": self.k,
            "beta": self.beta,
            "kappa": self.kappa,
            "sigma_sq": self.sigma_sq,
            "delta": self.delta,
            "lambda_mode": self.lambda_mode,
            "lambda_window_size": self.lambda_window_size,
            "calib_fraction": self.calib_fraction,
            "share_banks_across_tasks": self.share_banks_across_tasks,
            "seed": self.seed,
            "dynamics_ckpt_path": self.dynamics_ckpt_path,
            "uses_dynamics_feature": self.feature_extractor is not None,
        }
        if self.feature_extractor is not None:
            summary["dynamics_feature"] = self.feature_extractor.summary()
        return summary

    def close(self) -> None:
        try:
            self.encoder.close()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
