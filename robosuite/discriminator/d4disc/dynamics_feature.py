"""Inference-time feature assembly for D4-Disc.

D4 scoring is CFG-guided prediction in latent space, so per-frame we need
(z_t, s_t, a_{t:t+h}, z_{t+h}) to feed the conditional predictor twice
(c=+, c=-) and compare its omega-mixed prediction to z_{t+h}.

This module assembles those tensors from a ``BenchmarkTrajectory`` using:
- the frozen flow_multi encoder cache (z_t via encoder.load_or_encode_demo),
- per-step state slicing matching the trained predictor's proprio_dim,
- edge-padded action chunks of length ``action_horizon``,
- target latents z_{t+h}, with the last ``h`` frames edge-clamped
  (z_{T-1} reused as target for t >= T-h so scoring is defined over the
  full trajectory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from robosuite.discriminator.d3disc.encoder import FlowMultiEncoderWrapper


@dataclass
class D4Frames:
    z_current: torch.Tensor      # (T, latent_dim)
    z_target: torch.Tensor       # (T, latent_dim) -- z[t+h] with edge clamp
    proprio: torch.Tensor        # (T, proprio_dim)
    action_chunks: torch.Tensor  # (T, action_horizon, action_dim)
    length: int


class D4FeatureExtractor:
    """Assemble (z_t, s_t, a_chunks, z_{t+h}) for inference-time scoring.

    No model is loaded here — the trained ConditionalDynamicsPredictor lives
    in ``D4Detector``; this class only reads the on-disk cache and slices
    proprio / actions to match ``arch_args``.
    """

    def __init__(
        self,
        encoder: FlowMultiEncoderWrapper,
        *,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        action_horizon: int,
        proprio_indices: Optional[list[int]] = None,
    ) -> None:
        self.encoder = encoder
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.proprio_indices = (
            None if proprio_indices is None or len(proprio_indices) == 0
            else np.asarray(proprio_indices, dtype=np.int64)
        )
        if self.latent_dim != int(self.encoder.latent_dim):
            raise ValueError(
                f"latent_dim={self.latent_dim} != encoder.latent_dim={self.encoder.latent_dim}"
            )

    # ------------------------------------------------------------------ #
    # Slicing helpers                                                    #
    # ------------------------------------------------------------------ #

    def _slice_proprio(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if self.proprio_indices is not None:
            return states[:, self.proprio_indices]
        d = int(states.shape[1])
        target = int(self.proprio_dim)
        if d == target:
            return states
        if d > target:
            return states[:, :target]
        pad = np.zeros((states.shape[0], target - d), dtype=np.float32)
        return np.concatenate([states, pad], axis=1)

    def _action_chunks(self, actions: np.ndarray, t_len: int, horizon: int) -> np.ndarray:
        """Build (T, horizon, A) action windows matching the training horizon.

        MUST use the actual training horizon (typically 1) rather than
        ``max_action_horizon`` (the architectural upper bound, typically 32).
        A mismatched horizon changes the transformer sequence length, shifts
        positional embeddings, and displaces the pred_latent/pred_proprio
        tokens, which makes the trained predictor produce garbage residuals
        at inference. See benchmark regression observed on run d4dyn_20260422_063722.
        """
        actions = np.asarray(actions[:t_len], dtype=np.float32)
        # Align action_dim.
        if actions.shape[1] < self.action_dim:
            pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]

        h = int(horizon)
        if h <= 0:
            raise ValueError(f"horizon must be >= 1, got {h}")
        if h > self.action_horizon:
            raise ValueError(
                f"horizon={h} exceeds max_action_horizon={self.action_horizon}"
            )
        out = np.zeros((t_len, h, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + h)
            chunk = actions[t:end]
            k = chunk.shape[0]
            if k == 0:
                continue
            out[t, :k] = chunk
            if k < h:
                out[t, k:] = chunk[-1]
        return out

    def _target_latents(self, latents: np.ndarray, horizon: int) -> np.ndarray:
        T = int(latents.shape[0])
        out = np.empty_like(latents)
        for t in range(T):
            tp = min(t + int(horizon), T - 1)
            out[t] = latents[tp]
        return out

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        task_name: str,
        source_file_path: str,
        source_demo_key: str,
        states: np.ndarray,
        actions: np.ndarray,
        horizon: int = 1,
    ) -> D4Frames:
        encoded = self.encoder.load_or_encode_demo(
            task_name=task_name,
            file_path=source_file_path,
            demo_key=source_demo_key,
        )
        latents = np.asarray(encoded.latents, dtype=np.float32)
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        t_len = int(min(latents.shape[0], states_np.shape[0], actions_np.shape[0]))
        if t_len <= 0:
            raise ValueError(f"empty trajectory {task_name}/{source_demo_key}")

        proprio_np = self._slice_proprio(states_np[:t_len])
        action_chunks_np = self._action_chunks(actions_np, t_len, horizon=int(horizon))
        latents_cur = latents[:t_len]
        latents_tgt = self._target_latents(latents_cur, horizon=horizon)

        return D4Frames(
            z_current=torch.from_numpy(latents_cur),
            z_target=torch.from_numpy(latents_tgt),
            proprio=torch.from_numpy(proprio_np),
            action_chunks=torch.from_numpy(action_chunks_np),
            length=t_len,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "proprio_indices": None if self.proprio_indices is None else self.proprio_indices.tolist(),
        }
