"""Offline nnPU scoring helpers shared by VAST warmup and visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .encoder import SharedDynamicsEncoder
from .nnpu import FrozenNNPUDiscriminator


def nnpu_intrinsic_from_failure_score(
    failure_score: float | np.ndarray | torch.Tensor,
    threshold: float,
) -> float | np.ndarray | torch.Tensor:
    """Map nnPU failure score to the DIPOLE intrinsic reward convention."""
    if torch.is_tensor(failure_score):
        tau = torch.as_tensor(
            threshold, dtype=failure_score.dtype, device=failure_score.device
        )
        return -torch.sigmoid(failure_score - tau)
    score = np.asarray(failure_score, dtype=np.float64)
    reward = -1.0 / (1.0 + np.exp(-(score - float(threshold))))
    if np.isscalar(failure_score):
        return float(reward)
    return reward.astype(np.float32)


def annotate_transitions_with_nnpu_scores(
    transitions: Sequence[Any],
    *,
    failure_scores: np.ndarray,
    threshold: float,
) -> None:
    """Write immutable nnPU score/reward provenance on pipeline transitions."""
    scores = np.asarray(failure_scores, dtype=np.float32).reshape(-1)
    if not transitions:
        return
    if scores.size == 0:
        raise ValueError("failure_scores cannot be empty")
    if scores.size < len(transitions):
        scores = np.pad(scores, (0, len(transitions) - scores.size), mode="edge")
    rewards = np.asarray(
        nnpu_intrinsic_from_failure_score(scores[: len(transitions)], threshold),
        dtype=np.float32,
    )
    for transition, score, reward in zip(transitions, scores, rewards):
        info = dict(transition.info or {})
        info["nnpu_failure_score"] = float(score)
        info["nnpu_threshold"] = float(threshold)
        info["nnpu_disc_intrinsic"] = float(reward)
        transition.info = info


class NNPUOfflineScorer:
    """Batch scorer for already-loaded pipeline transitions."""

    def __init__(
        self,
        *,
        nnpu_ckpt_path: str,
        task_name: str,
        device: str = "cuda:1",
        batch_size: int = 32,
        camera_names: Sequence[str] | None = None,
        camera_to_view: dict[str, str] | None = None,
        encoder_ckpt: str | None = None,
        encoder: SharedDynamicsEncoder | None = None,
        discriminator: FrozenNNPUDiscriminator | None = None,
    ) -> None:
        self.encoder = encoder or SharedDynamicsEncoder(
            nnpu_ckpt_path,
            encoder_ckpt=encoder_ckpt,
            device=device,
            camera_to_view=camera_to_view,
        )
        self.discriminator = discriminator or FrozenNNPUDiscriminator(
            nnpu_ckpt_path, task_name=task_name, device=device, encoder=self.encoder
        )
        self.task_name = str(task_name)
        self.batch_size = max(1, int(batch_size))
        self.tau = float(self.discriminator.threshold)
        self.tau_source = f"pu_bce_detector.thresholds[{self.task_name!r}]"
        if camera_names is not None:
            self.encoder.bind_policy_cameras(camera_names)

    @torch.no_grad()
    def score_features(self, chunk_features: torch.Tensor) -> torch.Tensor:
        return self.discriminator.failure_score(chunk_features)

    @torch.no_grad()
    def score_transitions(
        self,
        transitions: Sequence[Any],
        *,
        camera_names: Sequence[str],
    ) -> np.ndarray:
        """Score transition states using their future executed-action windows."""
        if not transitions:
            return np.zeros((0,), dtype=np.float32)
        self.encoder.bind_policy_cameras(camera_names)
        images: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        actions = [np.asarray(item.action, dtype=np.float32).reshape(-1) for item in transitions]
        for item in transitions:
            obs = item.obs
            views = []
            for camera in camera_names:
                image = np.asarray(obs[camera])
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"obs[{camera!r}] must be HxWx3, got {image.shape}")
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                else:
                    image = image.astype(np.float32)
                    if image.size and image.max() > 1.5:
                        image = image / 255.0
                views.append(np.transpose(image, (2, 0, 1)))
            images.append(np.stack(views, axis=0))
            proprio.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))

        score_parts: list[torch.Tensor] = []
        horizon = len(transitions)
        for start in range(0, horizon, self.batch_size):
            end = min(horizon, start + self.batch_size)
            windows = []
            for step in range(start, end):
                rows = actions[step : step + self.encoder.frameskip]
                if len(rows) < self.encoder.frameskip:
                    rows = rows + [rows[-1]] * (self.encoder.frameskip - len(rows))
                windows.append(np.stack(rows, axis=0))
            features = self.encoder.encode_chunk(
                image_obs_raw=torch.from_numpy(np.stack(images[start:end], axis=0)),
                proprio_raw=torch.from_numpy(np.stack(proprio[start:end], axis=0)),
                action_chunk=torch.from_numpy(np.stack(windows, axis=0)),
            )
            score_parts.append(self.score_features(features).detach().cpu())
        return torch.cat(score_parts).numpy().astype(np.float32)

    def annotate_transitions(
        self,
        transitions: Sequence[Any],
        *,
        camera_names: Sequence[str],
    ) -> np.ndarray:
        scores = self.score_transitions(transitions, camera_names=camera_names)
        annotate_transitions_with_nnpu_scores(
            transitions, failure_scores=scores, threshold=self.tau
        )
        return scores

    def summary_dict(self) -> dict[str, Any]:
        return {
            "nnpu_ckpt": self.discriminator.ckpt_path,
            "task": self.task_name,
            "threshold": self.tau,
            "threshold_source": self.tau_source,
            "state_feature_dim": self.encoder.state_feature_dim,
            "chunk_feature_dim": self.encoder.chunk_feature_dim,
            "device": str(self.encoder.device),
        }


__all__ = [
    "NNPUOfflineScorer",
    "annotate_transitions_with_nnpu_scores",
    "nnpu_intrinsic_from_failure_score",
]
