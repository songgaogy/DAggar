from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .float_core import cosine_cost_matrix, sinkhorn


@dataclass
class EpisodeFloatResult:
    """Per-episode FLOAT outputs in ARMADA-style online matching."""

    cumulative_costs: np.ndarray
    step_failure_flags: np.ndarray
    matched_expert_indices: np.ndarray


class StateWindowEmbeddingBuilder:
    """
    Build To-window state embeddings sampled every Ta steps.

    This approximates policy latent stacking in official FLOAT when policy latent
    extraction is unavailable in this package.
    """

    def __init__(self, to: int, ta: int, normalize: bool = True) -> None:
        self.to = int(to)
        self.ta = int(ta)
        self.normalize = bool(normalize)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        if self.to <= 0:
            raise ValueError(f"to must be >= 1, got {to}")
        if self.ta <= 0:
            raise ValueError(f"ta must be >= 1, got {ta}")

    def fit(self, trajectories: list[np.ndarray]) -> None:
        if not trajectories:
            raise ValueError("Cannot fit state embedding builder with empty trajectories")
        stacked = np.concatenate([np.asarray(x, dtype=np.float32) for x in trajectories], axis=0)
        self.mean = stacked.mean(axis=0)
        self.std = stacked.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def _normalize_states(self, states: np.ndarray) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"states must be shape (T, D), got {arr.shape}")
        if self.normalize:
            if self.mean is None or self.std is None:
                raise RuntimeError("StateWindowEmbeddingBuilder is not fitted")
            if arr.shape[1] != self.mean.shape[0]:
                raise ValueError(
                    "State dimension mismatch for normalization: "
                    f"states dim={arr.shape[1]} fitted dim={self.mean.shape[0]}"
                )
            arr = (arr - self.mean) / self.std
        return arr

    def encode(self, states: np.ndarray) -> np.ndarray:
        arr = self._normalize_states(states)
        t = arr.shape[0]
        if t == 0:
            raise ValueError("Cannot encode empty state trajectory")

        num_steps = max(1, t // self.ta)
        embeddings = []

        for step in range(num_steps):
            idx = min(step * self.ta, t - 1)
            start = idx - self.to + 1
            indices = np.arange(start, idx + 1)
            indices = np.clip(indices, 0, t - 1)
            window = arr[indices]
            embeddings.append(window.reshape(-1))

        return np.asarray(embeddings, dtype=np.float32)


def _pad_last(embeddings: np.ndarray, target_len: int) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be (T, E), got {embeddings.shape}")
    t = embeddings.shape[0]
    if t > target_len:
        return embeddings[:target_len]
    if t == target_len:
        return embeddings
    if t == 0:
        raise ValueError("Cannot pad empty embeddings")
    return np.concatenate([embeddings, np.repeat(embeddings[-1:, :], target_len - t, axis=0)], axis=0)


class OfficialFloatOfflineEvaluator:
    """
    ARMADA-style FLOAT evaluator.

    Key mechanics matched to official implementation:
    - candidate expert retrieval using rollout init latent
    - expert rematching on rollout prefix
    - partial OT on padded rollout and cumulative greedy OT costs
    - failure when cumulative cost exceeds calibrated threshold
    """

    def __init__(
        self,
        expert_embeddings: list[np.ndarray],
        sinkhorn_reg: float,
        max_iter: int,
        tol: float,
        num_expert_candidates: int,
        max_steps: Optional[int] = None,
        use_similarity_cost: bool = False,
    ) -> None:
        if not expert_embeddings:
            raise ValueError("OfficialFloatOfflineEvaluator requires non-empty expert embeddings")

        self.sinkhorn_reg = float(sinkhorn_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.num_expert_candidates = int(num_expert_candidates)
        self.use_similarity_cost = bool(use_similarity_cost)

        if self.num_expert_candidates <= 0:
            raise ValueError(f"num_expert_candidates must be >= 1, got {num_expert_candidates}")

        lengths = [int(e.shape[0]) for e in expert_embeddings]
        inferred_max = max(lengths)
        self.max_steps = int(max_steps) if max_steps is not None else inferred_max
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be >=1, got {self.max_steps}")

        self.expert_embeddings = [_pad_last(np.asarray(e, dtype=np.float32), self.max_steps) for e in expert_embeddings]

    def _ot_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        c = cosine_cost_matrix(x, y, use_similarity_cost=self.use_similarity_cost)
        a = np.full(x.shape[0], 1.0 / float(x.shape[0]), dtype=np.float64)
        b = np.full(y.shape[0], 1.0 / float(y.shape[0]), dtype=np.float64)
        p = sinkhorn(a=a, b=b, c=c, reg=self.sinkhorn_reg, max_iter=self.max_iter, tol=self.tol)
        return float(np.sum(p * c))

    def _partial_ot_cost_vector(self, expert: np.ndarray, rollout_prefix: np.ndarray) -> np.ndarray:
        idx = rollout_prefix.shape[0] - 1
        le = expert.shape[0]

        partial_dist = np.concatenate(
            [
                cosine_cost_matrix(expert, rollout_prefix, use_similarity_cost=self.use_similarity_cost),
                np.zeros((le, self.max_steps - idx - 1), dtype=np.float64),
            ],
            axis=1,
        )

        rollout_padded = np.concatenate(
            [
                rollout_prefix,
                np.zeros((self.max_steps - idx - 1, rollout_prefix.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )

        a = np.full(le, 1.0 / float(le), dtype=np.float64)
        b = np.full(self.max_steps, 1.0 / float(self.max_steps), dtype=np.float64)
        p = sinkhorn(a=a, b=b, c=partial_dist, reg=self.sinkhorn_reg, max_iter=self.max_iter, tol=self.tol)

        observed_col_cost = np.sum(p[:, : idx + 1] * partial_dist[:, : idx + 1], axis=0)
        return np.concatenate([observed_col_cost, np.zeros(self.max_steps - idx - 1, dtype=np.float64)], axis=0)

    def find_matching_expert_demo(self, rollout_init: np.ndarray) -> np.ndarray:
        costs = []
        for expert in self.expert_embeddings:
            costs.append(self._ot_cost(expert, rollout_init))
        order = np.argsort(np.asarray(costs, dtype=np.float64))
        k = min(self.num_expert_candidates, len(order))
        return order[:k]

    def rematch_expert_episode(self, candidate_indices: np.ndarray, rollout_prefix: np.ndarray) -> np.ndarray:
        costs = []
        for idx in candidate_indices:
            expert = self.expert_embeddings[int(idx)]
            costs.append(self._ot_cost(expert, rollout_prefix))
        order = np.argsort(np.asarray(costs, dtype=np.float64))
        return candidate_indices[order]

    def run_episode(self, rollout_embeddings: np.ndarray, threshold: Optional[float]) -> EpisodeFloatResult:
        rollout = np.asarray(rollout_embeddings, dtype=np.float32)
        if rollout.ndim != 2:
            raise ValueError(f"rollout_embeddings must be (T,E), got {rollout.shape}")
        if rollout.shape[0] == 0:
            raise ValueError("rollout_embeddings cannot be empty")

        if rollout.shape[0] > self.max_steps:
            rollout = rollout[: self.max_steps]

        candidates = self.find_matching_expert_demo(rollout_init=rollout[:1])
        cumulative = np.zeros(rollout.shape[0], dtype=np.float64)
        flags = np.zeros(rollout.shape[0], dtype=np.int64)
        matched = np.zeros(rollout.shape[0], dtype=np.int64)

        for idx in range(rollout.shape[0]):
            prefix = rollout[: idx + 1]
            candidates = self.rematch_expert_episode(candidates, prefix)
            best_idx = int(candidates[0])
            matched[idx] = best_idx

            expert = self.expert_embeddings[best_idx]
            greedy_cost = self._partial_ot_cost_vector(expert=expert, rollout_prefix=prefix)
            cumulative[idx] = float(np.sum(greedy_cost[: idx + 1]))
            if threshold is not None:
                flags[idx] = int(cumulative[idx] > float(threshold))

        return EpisodeFloatResult(
            cumulative_costs=cumulative,
            step_failure_flags=flags,
            matched_expert_indices=matched,
        )

    def episode_score(self, rollout_embeddings: np.ndarray) -> float:
        result = self.run_episode(rollout_embeddings=rollout_embeddings, threshold=None)
        return float(result.cumulative_costs[-1])

    def calibrate_threshold(self, success_embeddings: list[np.ndarray], delta: float) -> float:
        if not success_embeddings:
            raise ValueError("calibrate_threshold requires non-empty success embeddings")
        if delta < 0 or delta > 100:
            raise ValueError(f"delta must be in [0, 100], got {delta}")

        scores = [self.episode_score(x) for x in success_embeddings]
        q = 100.0 * (1.0 - float(delta) / 100.0)
        return float(np.percentile(np.asarray(scores, dtype=np.float64), q=q))
