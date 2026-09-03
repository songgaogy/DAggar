"""Causal trajectory feature extraction from a frozen RPT checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

from robosuite.discriminator.dyn_disc.core.model_loader import load_rpt_checkpoint
from robosuite.discriminator.dyn_disc.models.dinov3_encoder import DINOv3Encoder
from robosuite.discriminator.dyn_disc.utils.normalizer import LinearNormalizer


class RPTTrajectoryEncoder:
    """Encode each trajectory frame using an eight-step, past-only RPT window."""

    feature_source = "rpt_action_token"

    def __init__(
        self,
        model_ckpt: str,
        *,
        device: str = "cuda",
        image_batch_size: int = 128,
        window_batch_size: int = 2048,
    ) -> None:
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError(f"RPTTrajectoryEncoder is CUDA-only; got device={device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for RPT trajectory encoding")
        self.device = requested_device
        self.image_batch_size = int(image_batch_size)
        self.window_batch_size = int(window_batch_size)
        if self.image_batch_size < 1 or self.window_batch_size < 1:
            raise ValueError("Encoding batch sizes must be positive")

        self.model, self.cfg, self.metadata = load_rpt_checkpoint(
            model_ckpt, device=self.device
        )
        architecture = self.metadata["architecture"]
        self.view_names: List[str] = list(architecture["view_names"])
        self.context_length = int(architecture["context_length"])
        self.proprio_dim = int(architecture["proprio_dim"])
        self.action_dim = int(architecture["action_dim"])
        self.hidden_dim = int(architecture["hidden_dim"])
        self.original_img_size = int(self.cfg.encoder.image_size)
        cfg_proprio_map = getattr(self.cfg, "proprio_map", {}) or {}
        self.proprio_map = dict(
            OmegaConf.to_container(cfg_proprio_map, resolve=True) or {}
        )

        self.visual_encoder = DINOv3Encoder(
            model_path=str(self.cfg.encoder.model_path),
            view_names=self.view_names,
            image_size=self.original_img_size,
        ).to(self.device)
        self.visual_encoder.eval()
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad_(False)

        run_dir = Path(self.metadata["checkpoint_path"]).parent.parent
        normalizer_path = run_dir / "normalizer.pth"
        if not normalizer_path.is_file():
            raise FileNotFoundError(
                f"RPT normalizer not found next to checkpoint: {normalizer_path}"
            )
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(
            torch.load(normalizer_path, map_location=self.device, weights_only=False)
        )
        self.normalizer.to(self.device)

    @property
    def checkpoint_fingerprint(self) -> str:
        return str(self.metadata["checkpoint_fingerprint"])

    @torch.inference_mode()
    def _encode_images(self, images_per_view: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        visual_latents: Dict[str, torch.Tensor] = {}
        for view in self.view_names:
            if view not in images_per_view:
                raise KeyError(f"Missing RPT camera view {view!r}")
            images = images_per_view[view]
            chunks: List[torch.Tensor] = []
            for start in range(0, images.shape[0], self.image_batch_size):
                batch = images[start : start + self.image_batch_size].to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    latent = self.visual_encoder.encode_images(batch)
                chunks.append(latent.to(dtype=torch.float32))
            visual_latents[view] = torch.cat(chunks, dim=0)
        return visual_latents

    @torch.inference_mode()
    def encode_trajectory(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Return one 192-D current-action-token feature per input frame."""
        proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if proprio_tensor.ndim != 2 or action_tensor.ndim != 2:
            raise ValueError("proprio and actions must have shape (T, D)")
        t_len = int(proprio_tensor.shape[0])
        if t_len < 1 or int(action_tensor.shape[0]) != t_len:
            raise ValueError("Trajectory modalities must have the same non-zero length")
        if proprio_tensor.shape[1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio dim {self.proprio_dim}, got {proprio_tensor.shape[1]}"
            )
        if action_tensor.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {action_tensor.shape[1]}"
            )
        for view in self.view_names:
            if int(images_per_view[view].shape[0]) != t_len:
                raise ValueError(f"View {view!r} length does not match state/action length")

        visual_latents = self._encode_images(images_per_view)
        proprio_tensor = self.normalizer["state"].normalize(proprio_tensor)
        action_tensor = self.normalizer["act"].normalize(action_tensor)

        offsets = torch.arange(
            1 - self.context_length, 1, device=self.device, dtype=torch.long
        )
        indices = torch.arange(t_len, device=self.device)[:, None] + offsets[None, :]
        indices.clamp_(min=0)

        features: List[torch.Tensor] = []
        for start in range(0, t_len, self.window_batch_size):
            index = indices[start : start + self.window_batch_size]
            visual_windows = torch.stack(
                [visual_latents[view][index] for view in self.view_names], dim=2
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feature = self.model.extract_features(
                    visual_windows,
                    proprio_tensor[index],
                    action_tensor[index],
                )
            features.append(feature.to(dtype=torch.float32))
        output = torch.cat(features, dim=0)
        if output.shape != (t_len, self.hidden_dim):
            raise RuntimeError(
                f"RPT feature shape invariant failed: expected {(t_len, self.hidden_dim)}, "
                f"got {tuple(output.shape)}"
            )
        return output

    def close(self) -> None:
        del self.model
        del self.visual_encoder
        torch.cuda.empty_cache()


__all__ = ["RPTTrajectoryEncoder"]
