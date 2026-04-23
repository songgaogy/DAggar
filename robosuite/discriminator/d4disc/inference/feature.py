"""Inference-time feature assembly using cached images + LPB encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from ..data.cache import PreprocessedCacheReader
from ..models.encoder import Encoder


@dataclass
class D4Frames:
    z_current: torch.Tensor
    z_target: torch.Tensor
    proprio: torch.Tensor
    action_chunks: torch.Tensor
    length: int


class D4FeatureExtractor:
    def __init__(
        self,
        encoder: Encoder,
        cache_reader: PreprocessedCacheReader,
        *,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        action_horizon: int,
        proprio_indices: Optional[list[int]] = None,
        encoder_batch_size: int = 256,
        device: str = "cuda",
    ) -> None:
        self.encoder = encoder
        self.cache_reader = cache_reader
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.encoder_batch_size = int(encoder_batch_size)
        self.device = (
            torch.device(device)
            if str(device).lower().startswith("cuda") and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.proprio_indices = (
            None if proprio_indices is None or len(proprio_indices) == 0 else np.asarray(proprio_indices, dtype=np.int64)
        )
        if self.latent_dim != int(self.encoder.latent_dim):
            raise ValueError(f"latent_dim={self.latent_dim} != encoder.latent_dim={self.encoder.latent_dim}")

    def _slice_proprio(self, proprio: np.ndarray) -> np.ndarray:
        arr = np.asarray(proprio, dtype=np.float32)
        if self.proprio_indices is not None:
            return arr[:, self.proprio_indices]
        dim = int(arr.shape[1])
        target = int(self.proprio_dim)
        if dim == target:
            return arr
        if dim > target:
            return arr[:, :target]
        pad = np.zeros((arr.shape[0], target - dim), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1)

    def _action_chunks(self, actions: np.ndarray, t_len: int, horizon: int) -> np.ndarray:
        actions = np.asarray(actions[:t_len], dtype=np.float32)
        if actions.shape[1] < self.action_dim:
            pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]

        h = int(horizon)
        if h <= 0:
            raise ValueError(f"horizon must be >= 1, got {h}")
        if h > self.action_horizon:
            raise ValueError(f"horizon={h} exceeds max_action_horizon={self.action_horizon}")
        out = np.zeros((t_len, h, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + h)
            chunk = actions[t:end]
            k = int(chunk.shape[0])
            if k == 0:
                continue
            out[t, :k] = chunk
            if k < h:
                out[t, k:] = chunk[-1]
        return out

    def _target_latents(self, latents: np.ndarray, horizon: int) -> np.ndarray:
        out = np.empty_like(latents)
        T = int(latents.shape[0])
        for t in range(T):
            out[t] = latents[min(t + int(horizon), T - 1)]
        return out

    @torch.no_grad()
    def _encode_images(self, images_chw: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        self.encoder.to(self.device)
        image_tensor = torch.from_numpy(np.ascontiguousarray(images_chw))
        chunks: list[torch.Tensor] = []
        batch = max(1, int(self.encoder_batch_size))
        for start in range(0, int(image_tensor.shape[0]), batch):
            end = min(start + batch, int(image_tensor.shape[0]))
            chunks.append(self.encoder(image_tensor[start:end].to(self.device)).detach().cpu())
        return torch.cat(chunks, dim=0).numpy().astype(np.float32)

    def extract(
        self,
        task_name: str,
        source_file_path: str,
        source_demo_key: str,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        trajectory_length: Optional[int] = None,
        horizon: int = 1,
    ) -> D4Frames:
        demo = self.cache_reader.load(task_name, source_file_path, source_demo_key)
        limits = [int(demo.length)]
        if trajectory_length is not None:
            limits.append(int(trajectory_length))
        if states is not None:
            limits.append(int(np.asarray(states).shape[0]))
        if actions is not None:
            limits.append(int(np.asarray(actions).shape[0]))
        t_len = int(min(limits))
        if t_len <= 0:
            raise ValueError(f"empty trajectory {task_name}/{source_demo_key}")

        z_current = self._encode_images(demo.images_chw[:t_len])
        z_target = self._target_latents(z_current, horizon=int(horizon))
        proprio = self._slice_proprio(demo.proprio[:t_len])
        action_chunks = self._action_chunks(demo.actions, t_len=t_len, horizon=int(horizon))
        return D4Frames(
            z_current=torch.from_numpy(z_current),
            z_target=torch.from_numpy(z_target),
            proprio=torch.from_numpy(proprio),
            action_chunks=torch.from_numpy(action_chunks),
            length=t_len,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "proprio_indices": None if self.proprio_indices is None else self.proprio_indices.tolist(),
            "encoder_batch_size": self.encoder_batch_size,
            "cache_root": self.cache_reader.cache_root,
        }
