"""Frozen nnPU discriminator used by DIPOLE and VAST."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from robosuite.discriminator.dyn_disc.detectors import PUBCEDiscriminator

from .base import DiscriminatorOutput


class FrozenNNPUDiscriminator:
    """Tensor-native adapter around a calibrated ``PUBCEDiscriminator``."""

    def __init__(
        self,
        nnpu_ckpt_path: str | Path | None = None,
        *,
        ckpt_path: str | Path | None = None,
        task_name: str,
        device: str | torch.device = "cuda:1",
        encoder: Any | None = None,
    ) -> None:
        supplied = nnpu_ckpt_path or ckpt_path
        if supplied is None:
            raise ValueError("nnpu_ckpt_path is required")
        self.ckpt_path = str(Path(supplied).expanduser().resolve())
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"nnPU checkpoint not found: {self.ckpt_path}")
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.task_name = str(task_name)
        self.encoder = encoder

        payload = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if "pu_bce_detector" not in payload:
            legacy = "bce_detector" in payload
            hint = " Legacy BCE checkpoints are not compatible." if legacy else ""
            raise KeyError(
                f"{self.ckpt_path} has no pu_bce_detector state.{hint}"
            )
        state = dict(payload["pu_bce_detector"])
        in_dim = int(state.get("in_dim", payload.get("in_dim", 0)))
        hidden = int(state.get("hidden", payload.get("hidden", 256)))
        num_layers = int(state.get("num_layers", payload.get("num_layers", 2)))
        self.detector = PUBCEDiscriminator(
            in_dim=in_dim,
            hidden=hidden,
            num_layers=num_layers,
            device=str(self.device),
        )
        self.detector.load_state_dict(state)
        self.detector.head.eval()
        for parameter in self.detector.head.parameters():
            parameter.requires_grad_(False)
        if self.task_name not in self.detector.thresholds:
            raise KeyError(
                f"Task {self.task_name!r} has no nnPU threshold; available: "
                f"{sorted(self.detector.thresholds)}"
            )
        self.threshold = float(self.detector.thresholds[self.task_name])
        self.chunk_feature_dim = int(self.detector.in_dim)

    @staticmethod
    def _select_feature(
        chunk_feature: torch.Tensor | None,
        context: torch.Tensor | None,
        features: torch.Tensor | None,
    ) -> torch.Tensor:
        selected = [value for value in (chunk_feature, context, features) if value is not None]
        if len(selected) != 1:
            raise ValueError("Provide exactly one of chunk_feature, context, or features")
        return selected[0]

    @torch.no_grad()
    def failure_score(
        self,
        chunk_feature: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature = self._select_feature(chunk_feature, context, features)
        return self.detector.failure_score_tensor(feature)

    @torch.no_grad()
    def score(
        self,
        chunk_feature: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> DiscriminatorOutput:
        failure_score = self.failure_score(
            chunk_feature, context=context, features=features
        )
        threshold = torch.full_like(failure_score, self.threshold)
        margin = failure_score - threshold
        return DiscriminatorOutput(
            logit=failure_score,
            prob_failure=torch.sigmoid(margin),
            decision=failure_score >= threshold,
            metadata={
                "failure_score": failure_score,
                "threshold": threshold,
                "normalized_margin": margin.abs(),
                "task_name": self.task_name,
            },
        )

    @torch.no_grad()
    def intrinsic_reward(
        self,
        chunk_feature: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``-sigmoid(failure_score - threshold)`` with leading shape intact."""
        failure_score = self.failure_score(
            chunk_feature, context=context, features=features
        )
        threshold = torch.as_tensor(
            self.threshold, device=failure_score.device, dtype=failure_score.dtype
        )
        return -torch.sigmoid(failure_score - threshold)

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint provenance only; the immutable weights remain external."""
        return {
            "schema": "frozen_nnpu_v1",
            "nnpu_ckpt_path": self.ckpt_path,
            "task_name": self.task_name,
            "threshold": self.threshold,
            "chunk_feature_dim": self.chunk_feature_dim,
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        if strict:
            expected = {
                "task_name": self.task_name,
                "threshold": self.threshold,
                "chunk_feature_dim": self.chunk_feature_dim,
            }
            for key, current in expected.items():
                if key in state and state[key] != current:
                    raise ValueError(
                        f"Frozen nnPU {key} mismatch: saved={state[key]!r}, current={current!r}"
                    )


__all__ = ["FrozenNNPUDiscriminator"]
