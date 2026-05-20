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

from typing import Sequence

import torch


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
        self.bce_ckpt_path = bce_ckpt_path
        self.device = device
        self.camera_to_view = dict(camera_to_view or {})
        # In implementation:
        #   - load bce_ckpt to extract feature_source / transformer_layer /
        #     model_ckpt;
        #   - build LPBV2Encoder(model_ckpt, device, feature_source, layer);
        #   - record self._context_dim, self._action_input_dim,
        #     self._original_img_size, self._view_names.
        self._context_dim: int = -1
        self._action_input_dim: int = -1
        self._original_img_size: tuple[int, int] = (0, 0)
        self._view_names: list[str] = []
        self._policy_camera_idx: list[int] | None = None

    # ------------------------------------------------------------------ #
    # Wiring                                                              #
    # ------------------------------------------------------------------ #

    def bind_policy_cameras(self, policy_cameras: Sequence[str]) -> None:
        """Resolve encoder-view → policy-camera-index map. Raises if a
        required encoder view is missing in policy_cameras (after applying
        camera_to_view remap)."""
        raise NotImplementedError

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
        raise NotImplementedError

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
