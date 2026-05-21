"""Shared frozen LPB v2 encoder for DIPOLE / IQL / OnlineBCE.

Single instance per process. The encoder is loaded from the pre-fitted
LPB v2 dynamics ckpt referenced by `bce_head.pth["model_ckpt"]` and lives
on the learner GPU (cuda:1). Inference-time copies live on cuda:0 in the
flow policy itself — those are NOT this object.

Degraded f(o, s) := encode_batch(images, proprio, actions := tile(proprio))
so the latent is action-independent at the encoder side. Q networks then
re-introduce the action via concat(context, flatten(action_chunk)).

This module ABSORBS the encoder-handling code currently embedded in
`robosuite/pipeline/algorithms/dipole/g_provider.py:84-219`. The original
g_provider should be refactored to consume an externally-built
SharedFrozenEncoder rather than building its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from robosuite.discriminator.lpb_v2 import LPBV2Encoder

from ..dipole.g_provider import _resolve_encoder_ckpt


def _tile_proprio_as_action(
    proprio: torch.Tensor,
    feature_source: str,
    frameskip: int,
    action_dim_per_step: int,
    action_input_dim: int,
) -> torch.Tensor:
    """Build the degraded `a := tile(s)` tensor.

    Returns shape:
        feature_source == "transformer": (B, action_dim_per_step)
        feature_source == "encoder":     (B, action_input_dim)
                                         == (B, frameskip * action_dim_per_step)
    """
    B, D_s = proprio.shape
    if D_s == 0:
        raise ValueError("proprio has zero feature dim; cannot tile as action")
    reps = (action_dim_per_step + D_s - 1) // D_s
    one = proprio.repeat(1, reps)[:, :action_dim_per_step]  # (B, action_dim_per_step)
    if feature_source == "transformer":
        return one.contiguous()
    expanded = one.unsqueeze(1).expand(B, frameskip, action_dim_per_step)
    flat = expanded.reshape(B, frameskip * action_dim_per_step)
    if flat.shape[1] != action_input_dim:
        # Pad/truncate defensively to honor the encoder's declared action_input_dim.
        if flat.shape[1] < action_input_dim:
            pad = torch.zeros(
                B, action_input_dim - flat.shape[1], device=flat.device, dtype=flat.dtype
            )
            flat = torch.cat([flat, pad], dim=-1)
        else:
            flat = flat[:, :action_input_dim]
    return flat.contiguous()


class SharedFrozenEncoder:
    """Thin shared wrapper around `LPBV2Encoder` (lpb_v2 single_bank_knn).

    Args:
        bce_ckpt_path:  path to the pre-fitted `bce_head.pth` (used only to
                        recover the upstream `model_ckpt` + feature_source
                        + transformer_layer fields).
        device:         CUDA device the encoder lives on (default cuda:1).
        camera_to_view: optional `{policy_camera: encoder_view}` remap.
    """

    def __init__(
        self,
        bce_ckpt_path: str,
        device: str = "cuda:1",
        camera_to_view: dict[str, str] | None = None,
    ) -> None:
        self.bce_ckpt_path = str(Path(bce_ckpt_path).resolve())
        if not Path(self.bce_ckpt_path).exists():
            raise FileNotFoundError(f"BCE checkpoint not found: {self.bce_ckpt_path}")
        self.device = device
        self.camera_to_view = dict(camera_to_view or {})

        ckpt = torch.load(self.bce_ckpt_path, map_location="cpu", weights_only=False)
        for required in ("bce_detector", "feature_source", "transformer_layer", "model_ckpt"):
            if required not in ckpt:
                raise KeyError(
                    f"BCE checkpoint {self.bce_ckpt_path} is missing required key "
                    f"'{required}'. Expected the schema produced by "
                    "run_bce_robosuite_benchmark.sh / visualize_bce_robosuite.sh."
                )
        self.feature_source: str = str(ckpt["feature_source"])
        self.transformer_layer: int = int(ckpt["transformer_layer"])

        encoder_ckpt_path = _resolve_encoder_ckpt(
            str(ckpt["model_ckpt"]), hint_root=Path(self.bce_ckpt_path).parent
        )
        self.inner_encoder = LPBV2Encoder(
            model_ckpt=str(encoder_ckpt_path),
            device=str(device),
            feature_source=self.feature_source,
            transformer_layer=self.transformer_layer,
        )

        # Hard-freeze the encoder.
        self.inner_encoder.model.eval()
        for p in self.inner_encoder.model.parameters():
            p.requires_grad_(False)
        for p in self.inner_encoder.model.parameters():
            assert not p.requires_grad, (
                "SharedFrozenEncoder requires every encoder parameter to have "
                "requires_grad=False"
            )

        self._view_names: list[str] = [str(v) for v in self.inner_encoder.view_names]
        self.frameskip: int = int(self.inner_encoder.frameskip)
        self.action_dim_per_step: int = int(self.inner_encoder.action_dim_per_step)
        self._action_input_dim: int = int(self.inner_encoder.action_input_dim)
        S = int(self.inner_encoder.original_img_size)
        self._original_img_size: tuple[int, int] = (S, S)
        self._proprio_input_dim: int = self._discover_proprio_input_dim()

        ckpt_in_dim = int(ckpt.get("in_dim", 0) or 0)
        if ckpt_in_dim <= 0:
            bce_state = ckpt.get("bce_detector", {})
            if isinstance(bce_state, dict):
                ckpt_in_dim = int(bce_state.get("in_dim", 0) or 0)
        if ckpt_in_dim > 0:
            self._context_dim: int = ckpt_in_dim
        else:
            self._context_dim = self._probe_context_dim()

        self._policy_camera_idx: list[int] | None = None
        self._policy_cameras_seen: list[str] | None = None

    def _discover_proprio_input_dim(self) -> int:
        """Discover the proprio input channel count from the encoder model."""
        proprio_encoder = getattr(self.inner_encoder.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            in_chans = getattr(proprio_encoder, "in_chans", None)
            if isinstance(in_chans, int) and in_chans > 0:
                return int(in_chans)
            for module in proprio_encoder.modules():
                if isinstance(module, torch.nn.Conv1d):
                    return int(module.in_channels)
        raise RuntimeError(
            "SharedFrozenEncoder could not infer proprio_input_dim from the "
            "wrapped LPBV2Encoder (no proprio_encoder Conv1d found)."
        )

    @torch.no_grad()
    def _probe_context_dim(self) -> int:
        """Run a tiny forward to discover D_ctx when ckpt['in_dim'] is missing."""
        H, W = self._original_img_size
        device = torch.device(self.device)
        images_per_view = {
            v: torch.zeros((1, 3, H, W), device=device, dtype=torch.float32)
            for v in self._view_names
        }
        proprio = torch.zeros((1, self._proprio_input_dim), device=device)
        actions = _tile_proprio_as_action(
            proprio,
            self.feature_source,
            self.frameskip,
            self.action_dim_per_step,
            self._action_input_dim,
        )
        feat = self.inner_encoder.encode_batch(images_per_view, proprio, actions)
        return int(feat.shape[-1])

    # ------------------------------------------------------------------ #
    # Wiring                                                              #
    # ------------------------------------------------------------------ #

    def bind_policy_cameras(self, policy_cameras: Sequence[str]) -> None:
        """Resolve encoder-view → policy-camera-index map. Raises if a
        required encoder view is missing in policy_cameras (after applying
        camera_to_view remap)."""
        new = [str(c) for c in policy_cameras]
        if self._policy_cameras_seen is not None:
            if new == self._policy_cameras_seen:
                return
            raise RuntimeError(
                "SharedFrozenEncoder.bind_policy_cameras was already called with "
                f"{self._policy_cameras_seen}; refusing to re-bind to {new}. "
                "Rebuild a fresh SharedFrozenEncoder if the camera tuple changes."
            )

        forward_map = {str(c): str(v) for c, v in self.camera_to_view.items()}
        reverse_map: dict[str, str] = {view: cam for cam, view in forward_map.items()}

        idx_list: list[int] = []
        for view in self._view_names:
            policy_camera = reverse_map.get(view, view)
            if policy_camera not in new:
                raise KeyError(
                    f"LPB encoder requires view '{view}' but the policy's camera tuple is "
                    f"{new}. Either include this camera in env.camera_names or pass "
                    f"camera_to_view={{policy_camera_name: '{view}'}} in the config."
                )
            idx_list.append(new.index(policy_camera))
        self._policy_camera_idx = idx_list
        self._policy_cameras_seen = new

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen encoder on (image, proprio).

        Args:
            image_obs_raw: (B, V, 3, H, W) floats in [0, 1] at the policy's
                           image size (the wrapper resizes per-view to the
                           encoder's `_original_img_size`).
            proprio_raw:   (B, D_s) un-normalized proprio.
        Returns:
            (B, D_ctx) frozen latent. action_input := tile(proprio_raw).
        """
        if self._policy_camera_idx is None:
            raise RuntimeError(
                "SharedFrozenEncoder.bind_policy_cameras(...) must be called before encode()."
            )
        device = torch.device(self.device)
        images = image_obs_raw.to(device, dtype=torch.float32)
        if images.dim() != 5:
            raise ValueError(
                f"image_obs_raw expected (B, V, 3, H, W); got shape {tuple(images.shape)}"
            )
        H, W = self._original_img_size
        images_per_view: dict[str, torch.Tensor] = {}
        for view, idx in zip(self._view_names, self._policy_camera_idx):
            view_tensor = images[:, idx]  # (B, 3, h, w)
            if view_tensor.shape[-1] != W or view_tensor.shape[-2] != H:
                view_tensor = F.interpolate(
                    view_tensor, size=(H, W), mode="bilinear", align_corners=False
                )
            images_per_view[view] = view_tensor.contiguous()

        proprio = proprio_raw.to(device, dtype=torch.float32)
        if proprio.dim() != 2:
            raise ValueError(
                f"proprio_raw expected (B, D_s); got shape {tuple(proprio.shape)}"
            )
        actions = _tile_proprio_as_action(
            proprio,
            self.feature_source,
            self.frameskip,
            self.action_dim_per_step,
            self._action_input_dim,
        )
        return self.inner_encoder.encode_batch(images_per_view, proprio, actions)

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def context_dim(self) -> int:
        """Dimensionality of the encoder output (= D_ctx)."""
        if self._context_dim < 0:
            raise RuntimeError(
                "SharedFrozenEncoder.context_dim accessed before initialization"
            )
        return self._context_dim

    @property
    def action_input_dim(self) -> int:
        return self._action_input_dim

    @property
    def view_names(self) -> list[str]:
        return list(self._view_names)

    @property
    def original_img_size(self) -> tuple[int, int]:
        return self._original_img_size

    @property
    def proprio_input_dim(self) -> int:
        """Channel count the wrapped LPBV2Encoder expects for proprio."""
        return self._proprio_input_dim
