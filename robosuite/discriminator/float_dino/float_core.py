from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


@dataclass
class Trajectory:
    """Container for one trajectory."""

    obs: np.ndarray
    actions: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.obs = np.asarray(self.obs)
        if self.obs.ndim < 2:
            raise ValueError(f"Trajectory.obs must have at least 2 dims (T, ...), got shape={self.obs.shape}")
        if self.actions is not None:
            self.actions = np.asarray(self.actions)
            if self.actions.ndim != 2:
                raise ValueError(f"Trajectory.actions must have shape (T, A), got shape={self.actions.shape}")
            if self.actions.shape[0] != self.obs.shape[0]:
                raise ValueError(
                    "Trajectory.actions first dim must match obs length: "
                    f"actions={self.actions.shape[0]} obs={self.obs.shape[0]}"
                )


class EmbeddingEncoder(Protocol):
    """Protocol for observation-to-embedding encoders."""

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode observations with shape (T, ...) into embeddings (T, E)."""


class IdentityEncoder:
    """Return observations as embeddings (flattened per timestep when needed)."""

    def encode(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError(f"IdentityEncoder expects obs shape (T, ...), got {arr.shape}")
        if arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
        t = int(arr.shape[0])
        return arr.reshape(t, -1).astype(np.float32, copy=False)


class TorchEncoderWrapper:
    """Optional torch-backed encoder wrapper for policy/image encoders."""

    def __init__(
        self,
        module: Any,
        device: str = "cpu",
        batch_size: int = 256,
        output_key: Optional[str] = None,
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("TorchEncoderWrapper requires torch to be installed") from exc

        self._torch = torch
        self.module = module
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.output_key = output_key
        self.module.to(self.device)
        self.module.eval()

    def _extract_output(self, output: Any) -> Any:
        if self.output_key is not None:
            if not isinstance(output, dict):
                raise ValueError("output_key was provided but module output is not a dict")
            if self.output_key not in output:
                raise ValueError(f"output_key '{self.output_key}' not found in module output keys={list(output.keys())}")
            return output[self.output_key]
        if isinstance(output, dict):
            if "embedding" in output:
                return output["embedding"]
            if "embeddings" in output:
                return output["embeddings"]
            raise ValueError(
                "Module returned dict output but no output_key was provided and no default key "
                "('embedding'/'embeddings') exists"
            )
        return output

    def encode(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError(f"TorchEncoderWrapper expects obs shape (T, ...), got {arr.shape}")

        embeddings: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, arr.shape[0], self.batch_size):
                stop = min(arr.shape[0], start + self.batch_size)
                batch_np = arr[start:stop]
                batch = self._torch.from_numpy(batch_np).to(self.device)
                output = self.module(batch)
                output = self._extract_output(output)
                if output.ndim < 2:
                    raise ValueError(f"Encoder output must have shape (B, ...), got {tuple(output.shape)}")
                out = output.reshape(output.shape[0], -1).detach().cpu().numpy().astype(np.float32)
                embeddings.append(out)

        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(embeddings, axis=0)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(y, axis=axis)


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    reg: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Compute an entropic regularized OT plan with a log-domain Sinkhorn solver.

    Args:
        a: Source marginal of shape (N,), sums to 1.
        b: Target marginal of shape (M,), sums to 1.
        c: Cost matrix of shape (N, M).
        reg: Entropic regularization strength (>0).
        max_iter: Maximum Sinkhorn iterations.
        tol: Stop when max marginal error <= tol.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Sinkhorn marginals must be 1D, got a.ndim={a.ndim}, b.ndim={b.ndim}")
    if c.shape != (a.shape[0], b.shape[0]):
        raise ValueError(
            "Cost shape mismatch for Sinkhorn: "
            f"c.shape={c.shape}, expected={(a.shape[0], b.shape[0])}"
        )
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Sinkhorn marginals must be strictly positive")

    a = a / np.sum(a)
    b = b / np.sum(b)

    log_a = np.log(a)
    log_b = np.log(b)
    u = np.zeros_like(a)
    v = np.zeros_like(b)

    for _ in range(int(max_iter)):
        u = reg * (log_a - _logsumexp((-c + v[None, :]) / reg, axis=1))
        v = reg * (log_b - _logsumexp((-c + u[:, None]) / reg, axis=0))

        if tol > 0:
            p = np.exp((-c + u[:, None] + v[None, :]) / reg)
            err = max(np.max(np.abs(p.sum(axis=1) - a)), np.max(np.abs(p.sum(axis=0) - b)))
            if err <= tol:
                return p

    return np.exp((-c + u[:, None] + v[None, :]) / reg)


def cosine_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    use_similarity_cost: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute pairwise cosine-based cost matrix between embeddings x and y."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2D embeddings, got x.shape={x.shape}, y.shape={y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Embedding dims must match, got x.shape[1]={x.shape[1]} and y.shape[1]={y.shape[1]}"
        )

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    denom = np.maximum(x_norm, eps) * np.maximum(y_norm.T, eps)
    sim = (x @ y.T) / denom
    sim = np.clip(sim, -1.0, 1.0)

    if use_similarity_cost:
        return sim
    return 1.0 - sim


def _pad_embeddings(embeddings: np.ndarray, target_len: int) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be shape (T, E), got shape={embeddings.shape}")
    t, e = embeddings.shape
    if t == 0:
        raise ValueError("Embeddings cannot be empty")
    if target_len < t:
        raise ValueError(f"target_len={target_len} is smaller than current length={t}")
    if target_len == t:
        return embeddings

    pad_count = target_len - t
    tail = np.repeat(embeddings[-1:, :], repeats=pad_count, axis=0)
    return np.concatenate([embeddings, tail], axis=0)


class FLOATComputer:
    """Compute FLOAT index values using OT between rollout prefixes and expert trajectories."""

    def __init__(
        self,
        experts: list[Trajectory],
        encoder: EmbeddingEncoder,
        sinkhorn_reg: float,
        max_iter: int,
        tol: float,
        pad_to: Optional[int] = None,
        use_similarity_cost: bool = False,
        pad_rollout_to_bound: bool = True,
    ) -> None:
        if not experts:
            raise ValueError("FLOATComputer requires at least one expert trajectory")

        self.encoder = encoder
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.pad_to = int(pad_to) if pad_to is not None else None
        self.use_similarity_cost = bool(use_similarity_cost)
        self.pad_rollout_to_bound = bool(pad_rollout_to_bound)

        self._expert_embeddings: list[np.ndarray] = []
        for idx, traj in enumerate(experts):
            emb = np.asarray(self.encoder.encode(traj.obs), dtype=np.float32)
            if emb.ndim != 2:
                raise ValueError(f"Expert encoder output must be 2D, got shape={emb.shape} for expert idx={idx}")
            if emb.shape[0] == 0:
                raise ValueError(f"Expert idx={idx} has empty embedding sequence")
            self._expert_embeddings.append(emb)

        if self.pad_to is not None:
            self._expert_embeddings = [_pad_embeddings(e, self.pad_to) for e in self._expert_embeddings]

    @property
    def expert_embeddings(self) -> list[np.ndarray]:
        return self._expert_embeddings

    def _prepare_rollout_embeddings(self, rollout: Trajectory, t0: Optional[int]) -> np.ndarray:
        emb = np.asarray(self.encoder.encode(rollout.obs), dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"Rollout encoder output must be 2D, got shape={emb.shape}")
        if emb.shape[0] == 0:
            raise ValueError("Rollout has empty embedding sequence")

        if t0 is None:
            t_end = emb.shape[0]
        else:
            t_end = int(t0)
            if t_end <= 0 or t_end > emb.shape[0]:
                raise ValueError(f"t0 must be in [1, {emb.shape[0]}], got t0={t0}")

        prefix = emb[:t_end]

        if self.pad_to is not None and self.pad_rollout_to_bound:
            if prefix.shape[0] > self.pad_to:
                raise ValueError(
                    f"Rollout prefix length {prefix.shape[0]} exceeds pad_to {self.pad_to}; "
                    "set pad_to=None, increase pad_to, or truncate t0"
                )
            prefix = _pad_embeddings(prefix, self.pad_to)

        return prefix

    def _lambda_single_expert(self, expert_emb: np.ndarray, rollout_prefix_emb: np.ndarray) -> float:
        c = cosine_cost_matrix(
            expert_emb,
            rollout_prefix_emb,
            use_similarity_cost=self.use_similarity_cost,
        )
        a = np.full(expert_emb.shape[0], 1.0 / float(expert_emb.shape[0]), dtype=np.float64)
        b = np.full(rollout_prefix_emb.shape[0], 1.0 / float(rollout_prefix_emb.shape[0]), dtype=np.float64)
        p = sinkhorn(a=a, b=b, c=c, reg=self.sinkhorn_reg, max_iter=self.max_iter, tol=self.tol)
        return float(np.sum(p * c))

    def lambda_per_expert(self, rollout: Trajectory, t0: Optional[int] = None) -> np.ndarray:
        """Compute FLOAT lambda_n(T_b[1:t0]) for each expert trajectory."""
        rollout_prefix_emb = self._prepare_rollout_embeddings(rollout=rollout, t0=t0)
        vals = [self._lambda_single_expert(expert_emb=e, rollout_prefix_emb=rollout_prefix_emb) for e in self._expert_embeddings]
        return np.asarray(vals, dtype=np.float64)

    def lambda_index(self, rollout: Trajectory, t0: Optional[int] = None) -> float:
        """Compute FLOAT index lambda(T_b[1:t0]) = min_n lambda_n(T_b[1:t0])."""
        vals = self.lambda_per_expert(rollout=rollout, t0=t0)
        return float(np.min(vals))

    def rewind_timestep(self, rollout: Trajectory, t0: int, eps: float = 0.2) -> int:
        """
        Compute adaptive rewind timestep:
            argmax_{t>=1} 1[lambda(T_b[1:t]) <= eps * lambda(T_b[1:t0])] * t
        """
        if eps <= 0 or eps >= 1:
            raise ValueError(f"eps must be in (0, 1), got {eps}")

        t0 = int(t0)
        if t0 <= 0 or t0 > rollout.obs.shape[0]:
            raise ValueError(f"t0 must be in [1, {rollout.obs.shape[0]}], got {t0}")

        lambda_t0 = self.lambda_index(rollout=rollout, t0=t0)
        threshold = eps * lambda_t0

        best_t = 1
        for t in range(1, t0 + 1):
            if self.lambda_index(rollout=rollout, t0=t) <= threshold:
                best_t = t
        return best_t


class ThresholdCalibrator:
    """Calibrate universal FLOAT threshold with percentile over successful rollout lambdas."""

    def __init__(self) -> None:
        self.success_lambdas: list[float] = []
        self.threshold: Optional[float] = None
        self.delta: Optional[float] = None

    def _compute_threshold(self, delta: float) -> float:
        if not self.success_lambdas:
            raise ValueError("No success lambdas available for threshold calibration")
        if delta < 0 or delta > 100:
            raise ValueError(f"delta must be in [0, 100], got {delta}")

        q = 100.0 * (1.0 - float(delta) / 100.0)
        return float(np.percentile(np.asarray(self.success_lambdas, dtype=np.float64), q=q))

    def fit(self, success_rollouts: list[Trajectory], float_computer: FLOATComputer, delta: float) -> float:
        if not success_rollouts:
            raise ValueError("fit requires non-empty success_rollouts")

        self.success_lambdas = [float(float_computer.lambda_index(r)) for r in success_rollouts]
        self.delta = float(delta)
        self.threshold = self._compute_threshold(delta=self.delta)
        return self.threshold

    def update(self, delta: float, lambdas: Optional[list[float]] = None) -> float:
        """
        Update calibrated threshold.

        Args:
            delta: Percent parameter in [0, 100]. Paper default is 10.
            lambdas: Optional new success lambdas to add to the calibration set.
        """
        if lambdas is not None:
            self.success_lambdas.extend(float(x) for x in lambdas)
        self.delta = float(delta)
        self.threshold = self._compute_threshold(delta=self.delta)
        return self.threshold


class OnlineDetector:
    """Online FLOAT detector over a growing rollout prefix."""

    def __init__(
        self,
        float_computer: FLOATComputer,
        calibrator: ThresholdCalibrator,
        delta: float,
        delta_step: float,
        stride: int = 1,
        greater_is_failure: bool = True,
    ) -> None:
        self.float_computer = float_computer
        self.calibrator = calibrator
        self.delta = float(delta)
        self.delta_step = float(delta_step)
        self.stride = int(stride)
        self.greater_is_failure = bool(greater_is_failure)

        if self.stride <= 0:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self._obs_buffer: list[np.ndarray] = []
        self._last_lambda: Optional[float] = None

    def reset(self) -> None:
        self._obs_buffer.clear()
        self._last_lambda = None

    def _is_failure(self, lambda_value: Optional[float], threshold: float) -> bool:
        if lambda_value is None:
            return False
        return lambda_value >= threshold if self.greater_is_failure else lambda_value <= threshold

    def step(self, obs_t: np.ndarray) -> dict[str, Any]:
        if self.calibrator.threshold is None:
            raise RuntimeError("Threshold is not calibrated. Call calibrator.fit(...) before online detection.")

        obs = np.asarray(obs_t)
        if obs.ndim == 0:
            raise ValueError("obs_t must have at least one dimension")
        self._obs_buffer.append(obs)

        t = len(self._obs_buffer)
        if t % self.stride == 0 or self._last_lambda is None:
            rollout_obs = np.stack(self._obs_buffer, axis=0)
            rollout = Trajectory(obs=rollout_obs)
            self._last_lambda = self.float_computer.lambda_index(rollout=rollout)

        threshold = float(self.calibrator.threshold)
        is_failure = self._is_failure(self._last_lambda, threshold)

        return {
            "t": t,
            "lambda": self._last_lambda,
            "threshold": threshold,
            "is_failure": is_failure,
        }

    def feedback(self, was_failure: bool, detector_raised: bool) -> float:
        """Update delta based on FN / FP feedback and refresh threshold."""
        if bool(was_failure) and not bool(detector_raised):
            self.delta -= self.delta_step
        elif not bool(was_failure) and bool(detector_raised):
            self.delta += self.delta_step

        self.delta = float(np.clip(self.delta, 0.0, 100.0))
        return self.calibrator.update(delta=self.delta)
