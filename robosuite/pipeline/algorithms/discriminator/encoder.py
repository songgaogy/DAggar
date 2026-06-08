"""Shared frozen LPB v2 encoder for DIPOLE / IQL / OnlineBCE.

Single instance per process. The encoder is loaded from the pre-fitted
LPB v2 dynamics ckpt referenced by `bce_head.pth["model_ckpt"]` and lives
on the learner GPU (cuda:1). Inference-time copies live on cuda:0 in the
flow policy itself — those are NOT this object.

Encoding modes (see ``encode`` / ``encode_chunk_frames``):
    * **lpb-aligned (default for IQL / disc replay):** pass real per-step or
      per-frame actions. For an H-step chunk, ``encode_chunk_frames`` builds
      the same frameskip action window as ``LPBV2GProvider._prepare_action_input``.
    * **degraded fallback:** ``action_real=None`` uses ``a := tile(proprio)``
      (legacy ``LPBV2GProvider`` / display paths without actions).

Q-chunk critics still consume ``concat(context, flatten(action_chunk))``;
the frozen encoder only produces ``context`` (D_ctx).

This module ABSORBS the encoder-handling code currently embedded in
`robosuite/pipeline/algorithms/dipole/g_provider.py:84-219`.
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


def _pad_actions_to_encoder_step_dim(
    actions: torch.Tensor, action_dim_per_step: int
) -> torch.Tensor:
    """Pad/truncate the last dim of ``(..., D_a)`` to ``action_dim_per_step``."""
    D_a = int(actions.shape[-1])
    if D_a == action_dim_per_step:
        return actions
    if D_a < action_dim_per_step:
        pad_shape = list(actions.shape[:-1]) + [action_dim_per_step - D_a]
        pad = torch.zeros(pad_shape, device=actions.device, dtype=actions.dtype)
        return torch.cat([actions, pad], dim=-1)
    return actions[..., :action_dim_per_step]


def _action_input_from_sequence_window(
    actions: torch.Tensor,
    *,
    feature_source: str,
    frameskip: int,
    action_dim_per_step: int,
    action_input_dim: int,
) -> torch.Tensor:
    """Build encoder action input from ``actions`` shaped ``(B, T, D_a)``.

    Uses the first ``frameskip`` steps of the sequence (padding with the last
    step if ``T < frameskip``), matching ``LPBV2GProvider._prepare_action_input``.
    """
    if actions.dim() != 3:
        raise ValueError(f"actions must be (B, T, D_a); got {tuple(actions.shape)}")
    B, horizon, _ = actions.shape
    acts = _pad_actions_to_encoder_step_dim(actions, action_dim_per_step)
    fs = int(frameskip)
    if horizon >= fs:
        window = acts[:, :fs, :]
    else:
        pad_steps = fs - horizon
        last = acts[:, -1:, :].expand(-1, pad_steps, -1)
        window = torch.cat([acts, last], dim=1)
    if feature_source == "transformer":
        return window[:, 0, :].contiguous()
    flat = window.reshape(B, fs * action_dim_per_step)
    if flat.shape[1] < action_input_dim:
        pad = torch.zeros(
            B, action_input_dim - flat.shape[1], device=flat.device, dtype=flat.dtype
        )
        flat = torch.cat([flat, pad], dim=-1)
    elif flat.shape[1] > action_input_dim:
        flat = flat[:, :action_input_dim]
    return flat.contiguous()


def _action_inputs_for_chunk_frames(
    chunk_actions: torch.Tensor,
    *,
    feature_source: str,
    frameskip: int,
    action_dim_per_step: int,
    action_input_dim: int,
) -> torch.Tensor:
    """Per-frame action inputs for an H-step chunk; returns ``(B*H, D_act)``."""
    B, H, _ = chunk_actions.shape
    rows: list[torch.Tensor] = []
    for h in range(H):
        window = chunk_actions[:, h:, :]
        rows.append(
            _action_input_from_sequence_window(
                window,
                feature_source=feature_source,
                frameskip=frameskip,
                action_dim_per_step=action_dim_per_step,
                action_input_dim=action_input_dim,
            )
        )
    return torch.stack(rows, dim=1).reshape(B * H, -1).contiguous()


def _adapt_action_to_input_dim(
    action_real: torch.Tensor,
    feature_source: str,
    frameskip: int,
    action_dim_per_step: int,
    action_input_dim: int,
) -> torch.Tensor:
    """Adapt a single-step real action ``(B, D_a)`` via a length-1 action window."""
    if action_real.dim() != 2:
        raise ValueError(
            f"action_real must be (B, D_a); got shape {tuple(action_real.shape)}"
        )
    if action_real.shape[-1] == 0:
        raise ValueError("action_real has zero feature dim; cannot adapt")
    return _action_input_from_sequence_window(
        action_real.unsqueeze(1),
        feature_source=feature_source,
        frameskip=frameskip,
        action_dim_per_step=action_dim_per_step,
        action_input_dim=action_input_dim,
    )


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
        action_real: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run frozen encoder on (image, proprio[, real_action]).

        This is the single owner of the LPB v2 image protocol. Callers
        (IQL replay, DiscriminatorReplayBuffer, AdvantageGProvider) MUST
        NOT pre-crop or pre-resize: this method (a) F.interpolate-s each
        view to ``_original_img_size``, then (b) hands off to
        ``LPBV2Encoder.encode_batch`` which applies the per-view
        ``LinearNormalizer`` and the eval CenterCrop.

        Action handling:
            action_real is None → degraded ``a := tile(proprio)``. Kept for
                backward compatibility (LPBV2GProvider, AdvantageG when the
                policy's chunk is unavailable) but yields a latent at a
                distribution offset from the lpb v2 BCE head's training
                domain — the head's first Linear was trained against
                (image, proprio, REAL action) concat.
            action_real is given → builds a length-1 frameskip window (same
                contract as ``LPBV2GProvider._prepare_action_input``) and
                feeds `encode_batch`. Prefer ``encode_chunk_frames`` when
                scoring every step in an H-step chunk.

        Args:
            image_obs_raw: (B, V, 3, H, W) floats in [0, 1] at the policy's
                           rollout image size (any HxW is fine — resized
                           internally).
            proprio_raw:   (B, D_s) un-normalized proprio.
            action_real:   optional (B, D_a) per-step real action. When
                           supplied the encoder uses it; otherwise falls
                           back to tile(proprio).
        Returns:
            (B, D_ctx) frozen latent.
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
        if action_real is None:
            actions = _tile_proprio_as_action(
                proprio,
                self.feature_source,
                self.frameskip,
                self.action_dim_per_step,
                self._action_input_dim,
            )
        else:
            action_real_t = action_real.to(device, dtype=torch.float32)
            if action_real_t.shape[0] != proprio.shape[0]:
                raise ValueError(
                    f"action_real batch size {action_real_t.shape[0]} does not match "
                    f"proprio batch size {proprio.shape[0]}"
                )
            actions = _adapt_action_to_input_dim(
                action_real_t,
                self.feature_source,
                self.frameskip,
                self.action_dim_per_step,
                self._action_input_dim,
            )
        return self.inner_encoder.encode_batch(images_per_view, proprio, actions)

    @torch.no_grad()
    def encode_chunk_frames(
        self,
        *,
        chunk_images: torch.Tensor,
        chunk_proprio: torch.Tensor,
        chunk_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode every frame in an H-step chunk with lpb-aligned action windows.

        Args:
            chunk_images:  (B, H, V, 3, h, w) float in [0, 1].
            chunk_proprio: (B, H, D_s).
            chunk_actions: (B, H, D_a) real per-step actions from the buffer.
        Returns:
            (B, H, D_ctx) latents — one row per frame, same order as the chunk.
        """
        if self._policy_camera_idx is None:
            raise RuntimeError(
                "SharedFrozenEncoder.bind_policy_cameras(...) must be called "
                "before encode_chunk_frames()."
            )
        if chunk_images.dim() != 6:
            raise ValueError(
                f"chunk_images expected (B, H, V, 3, h, w); got {tuple(chunk_images.shape)}"
            )
        B, H = int(chunk_images.shape[0]), int(chunk_images.shape[1])
        if chunk_proprio.shape[:2] != (B, H):
            raise ValueError(
                f"chunk_proprio must be (B, H, D_s); got {tuple(chunk_proprio.shape)}"
            )
        if chunk_actions.shape[:2] != (B, H):
            raise ValueError(
                f"chunk_actions must be (B, H, D_a); got {tuple(chunk_actions.shape)}"
            )

        device = torch.device(self.device)
        flat_images = chunk_images.to(device, dtype=torch.float32).reshape(
            B * H, chunk_images.shape[2], chunk_images.shape[3],
            chunk_images.shape[4], chunk_images.shape[5],
        )
        flat_proprio = chunk_proprio.to(device, dtype=torch.float32).reshape(B * H, -1)
        chunk_actions_d = chunk_actions.to(device, dtype=torch.float32)
        flat_actions = _action_inputs_for_chunk_frames(
            chunk_actions_d,
            feature_source=self.feature_source,
            frameskip=self.frameskip,
            action_dim_per_step=self.action_dim_per_step,
            action_input_dim=self._action_input_dim,
        )

        H_img, W_img = self._original_img_size
        images_per_view: dict[str, torch.Tensor] = {}
        for view, idx in zip(self._view_names, self._policy_camera_idx):
            view_tensor = flat_images[:, idx]
            if view_tensor.shape[-1] != W_img or view_tensor.shape[-2] != H_img:
                view_tensor = F.interpolate(
                    view_tensor, size=(H_img, W_img), mode="bilinear", align_corners=False
                )
            images_per_view[view] = view_tensor.contiguous()

        flat_ctx = self.inner_encoder.encode_batch(
            images_per_view, flat_proprio, flat_actions
        )
        D_ctx = int(flat_ctx.shape[-1])
        return flat_ctx.view(B, H, D_ctx)

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
