"""Inference-time feature extractor that mirrors LPB's 3-concat layout but
uses flow_multi's encoder + a predictor checkpoint trained by
``train_dynamics.py``:

    feat_t = [ obs_proj(z_t),  proprio_proj(s_t),  mean(action_proj(a_{t:t+h})) ]

Where ``z_t`` comes from ``FlowMultiEncoderWrapper`` (cache-aware), and
``obs_proj`` / ``proprio_proj`` / ``action_proj`` are the projection layers
inside the trained ``DynamicsPredictor`` (shared projection space, d_model).

Optional L2 normalization (default on) matches LPB's KNN convention, so F3
auto-kappa/beta heuristics live on the unit sphere.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.discriminator.lpb.model import DynamicsPredictor

from .encoder import FlowMultiEncoderWrapper


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class D3FeatureExtractor:
    """Turn (z_t, s_t, a_chunk) -> LPB-style concatenated feature phi_t."""

    def __init__(
        self,
        dynamics_ckpt_path: str,
        encoder: FlowMultiEncoderWrapper,
        *,
        device: str = "cuda",
        normalize_feature: bool = True,
    ) -> None:
        self.ckpt_path = str(dynamics_ckpt_path)
        self.encoder = encoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.normalize_feature = bool(normalize_feature)

        payload = _torch_load(self.ckpt_path)
        for key in ("latent_dim", "proprio_dim", "action_dim", "d_model",
                    "num_layers", "num_heads", "dropout", "max_action_horizon",
                    "horizon", "model"):
            if key not in payload:
                raise ValueError(f"dynamics ckpt missing key {key!r}: {self.ckpt_path}")

        self.latent_dim = int(payload["latent_dim"])
        self.proprio_dim = int(payload["proprio_dim"])
        self.action_dim = int(payload["action_dim"])
        self.action_horizon = int(payload["horizon"])
        self.d_model = int(payload["d_model"])
        self.proprio_indices: Optional[np.ndarray] = None
        indices = payload.get("proprio_indices")
        if indices is not None and len(indices) > 0:
            self.proprio_indices = np.asarray(list(indices), dtype=np.int64)

        if self.latent_dim != int(self.encoder.latent_dim):
            raise ValueError(
                f"dynamics ckpt latent_dim={self.latent_dim} != encoder.latent_dim={self.encoder.latent_dim}"
            )

        self.predictor = DynamicsPredictor(
            latent_dim=self.latent_dim,
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
            d_model=self.d_model,
            num_layers=int(payload["num_layers"]),
            nhead=int(payload["num_heads"]),
            dropout=float(payload["dropout"]),
            max_action_horizon=int(payload["max_action_horizon"]),
        )
        self.predictor.load_state_dict(payload["model"], strict=True)
        self.predictor.to(self.device)
        self.predictor.eval()
        for param in self.predictor.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return int(3 * self.d_model)

    # ------------------------------------------------------------------ #
    # Proprio / action prep (matches LPB conventions for mixed-task)     #
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

    def _action_chunks(self, actions: np.ndarray, t_len: int) -> np.ndarray:
        """(T, action_horizon, action_dim) edge-padded."""
        actions = np.asarray(actions[:t_len], dtype=np.float32)
        if actions.shape[1] < self.action_dim:
            pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]

        h = self.action_horizon
        out = np.zeros((t_len, h, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + h)
            chunk = actions[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < h:
                pad_val = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad_val
        return out

    # ------------------------------------------------------------------ #
    # Feature computation                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def extract_from_latents(
        self,
        latents: np.ndarray,      # (T, latent_dim)
        states: np.ndarray,       # (T, state_dim_raw)
        actions: np.ndarray,      # (T, action_dim_raw)
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """Returns (T, 3*d_model) CPU float32 tensor."""
        t_len = min(int(latents.shape[0]), int(states.shape[0]), int(actions.shape[0]))
        if t_len <= 0:
            raise ValueError("extract_from_latents got zero-length trajectory")

        z = torch.as_tensor(np.asarray(latents[:t_len], dtype=np.float32))
        s_np = self._slice_proprio(states[:t_len])
        a_chunks = self._action_chunks(actions, t_len)

        s = torch.from_numpy(s_np)
        a = torch.from_numpy(a_chunks)

        feats: list[torch.Tensor] = []
        for start in range(0, t_len, int(batch_size)):
            end = min(start + int(batch_size), t_len)
            zb = z[start:end].to(self.device)
            sb = s[start:end].to(self.device)
            ab = a[start:end].to(self.device)

            obs_tok = self.predictor.obs_proj(zb)              # (B, d_model)
            prop_tok = self.predictor.proprio_proj(sb)          # (B, d_model)
            act_tok = self.predictor.action_proj(ab).mean(dim=1)  # (B, d_model) — chunk mean
            f = torch.cat([obs_tok, prop_tok, act_tok], dim=-1)
            if self.normalize_feature:
                f = F.normalize(f, p=2.0, dim=-1)
            feats.append(f.detach().cpu())
        return torch.cat(feats, dim=0)

    def extract_from_trajectory(
        self,
        task_name: str,
        source_file_path: str,
        source_demo_key: str,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> torch.Tensor:
        """Look up cached z_t via the encoder, then compute phi."""
        encoded = self.encoder.load_or_encode_demo(
            task_name=task_name,
            file_path=source_file_path,
            demo_key=source_demo_key,
        )
        return self.extract_from_latents(
            latents=encoded.latents,
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
        )

    # ------------------------------------------------------------------ #
    # Summary (for calibration_summary() embed)                          #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        return {
            "dynamics_ckpt_path": self.ckpt_path,
            "latent_dim": self.latent_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "d_model": self.d_model,
            "feature_dim": self.feature_dim,
            "normalize_feature": bool(self.normalize_feature),
        }
