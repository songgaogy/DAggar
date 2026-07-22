"""Shared frozen dynamics encoder for DIPOLE state and action-chunk features."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from robosuite.discriminator.dyn_disc.detectors import DynEncoder


def resolve_encoder_checkpoint(model_ckpt: str, *, hint_root: Path) -> Path:
    """Resolve an encoder path embedded in a portable nnPU checkpoint."""
    path = Path(model_ckpt).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([hint_root / path, Path.cwd() / path])
        repo_root = Path(__file__).resolve().parents[4]
        candidates.append(repo_root / path)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate dynamics checkpoint {model_ckpt!r}; tried "
        f"{[str(p.resolve()) for p in candidates]}"
    )


class SharedDynamicsEncoder:
    """One frozen ``DynEncoder`` shared by nnPU, Q, and V.

    ``encode_state`` produces an action-free visual/proprio latent for V.
    ``encode_chunk`` produces the action-conditioned latent used to train the
    nnPU head for Q and discriminator rewards.
    """

    def __init__(
        self,
        nnpu_ckpt_path: str | Path | None = None,
        *,
        ckpt_path: str | Path | None = None,
        bce_ckpt_path: str | Path | None = None,
        encoder_ckpt: str | Path | None = None,
        device: str | torch.device = "cuda:1",
        camera_to_view: dict[str, str] | None = None,
    ) -> None:
        supplied = nnpu_ckpt_path or ckpt_path or bce_ckpt_path
        if supplied is None:
            raise ValueError("nnpu_ckpt_path is required")
        self.nnpu_ckpt_path = str(Path(supplied).expanduser().resolve())
        self.bce_ckpt_path = self.nnpu_ckpt_path  # transitional read-only alias
        if not Path(self.nnpu_ckpt_path).exists():
            raise FileNotFoundError(f"nnPU checkpoint not found: {self.nnpu_ckpt_path}")

        payload = torch.load(self.nnpu_ckpt_path, map_location="cpu", weights_only=False)
        required = ("pu_bce_detector", "feature_source", "transformer_layer", "model_ckpt")
        missing = [key for key in required if key not in payload]
        if missing:
            legacy = "bce_detector" in payload or "model" in payload
            hint = " This appears to be a legacy LPB/BCE checkpoint." if legacy else ""
            raise KeyError(
                f"nnPU checkpoint is missing {missing}; expected pu_bce_head.pth schema.{hint}"
            )

        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise RuntimeError(
                f"SharedDynamicsEncoder requires CUDA, got {requested_device}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SharedDynamicsEncoder requires CUDA, but CUDA is unavailable."
            )
        self.device = requested_device
        self.camera_to_view = dict(camera_to_view or {})
        self.feature_source = str(payload["feature_source"])
        self.transformer_layer = int(payload["transformer_layer"])
        model_ckpt = str(encoder_ckpt or payload["model_ckpt"])
        model_path = resolve_encoder_checkpoint(
            model_ckpt, hint_root=Path(self.nnpu_ckpt_path).parent
        )
        self.encoder_checkpoint = str(model_path)

        self.inner_encoder = DynEncoder(
            model_ckpt=str(model_path),
            device=str(self.device),
            feature_source=self.feature_source,
            transformer_layer=self.transformer_layer,
        )
        self.inner_encoder.model.eval()
        for parameter in self.inner_encoder.model.parameters():
            parameter.requires_grad_(False)

        self._view_names = [str(view) for view in self.inner_encoder.view_names]
        self.frameskip = int(self.inner_encoder.frameskip)
        self.action_dim_per_step = int(self.inner_encoder.action_dim_per_step)
        self.action_input_dim = int(self.inner_encoder.action_input_dim)
        size = int(self.inner_encoder.original_img_size)
        self.original_img_size = (size, size)
        self.state_feature_dim = int(
            self.inner_encoder.visual_emb_dim_total + self.inner_encoder.proprio_emb_dim
        )
        detector_state = dict(payload["pu_bce_detector"])
        self.chunk_feature_dim = int(detector_state.get("in_dim", payload.get("in_dim", 0)))
        if self.chunk_feature_dim <= 0:
            raise ValueError("nnPU checkpoint has an invalid or missing in_dim")

        self._policy_camera_indices: list[int] | None = None
        self._policy_cameras: list[str] | None = None

    def bind_policy_cameras(self, policy_cameras: Sequence[str]) -> None:
        """Bind encoder view order to the immutable policy camera tuple."""
        cameras = [str(camera) for camera in policy_cameras]
        if self._policy_cameras is not None:
            if cameras == self._policy_cameras:
                return
            raise RuntimeError(
                f"Encoder already bound to {self._policy_cameras}; cannot rebind to {cameras}"
            )
        reverse_map = {str(view): str(camera) for camera, view in self.camera_to_view.items()}
        indices: list[int] = []
        for view in self._view_names:
            camera = reverse_map.get(view, view)
            if camera not in cameras:
                raise KeyError(
                    f"Dynamics view {view!r} has no policy camera in {cameras}; "
                    "configure camera_to_view if their names differ."
                )
            indices.append(cameras.index(camera))
        self._policy_cameras = cameras
        self._policy_camera_indices = indices

    def _prepare_images(self, image_obs_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._policy_camera_indices is None:
            raise RuntimeError("bind_policy_cameras(...) must be called before encoding")
        images = image_obs_raw.to(self.device, dtype=torch.float32, non_blocking=True)
        if images.ndim != 5:
            raise ValueError(f"image_obs_raw must be (B, V, 3, H, W); got {tuple(images.shape)}")
        height, width = self.original_img_size
        out: dict[str, torch.Tensor] = {}
        for view, index in zip(self._view_names, self._policy_camera_indices):
            tensor = images[:, index]
            if tensor.shape[-2:] != (height, width):
                tensor = F.interpolate(
                    tensor, size=(height, width), mode="bilinear", align_corners=False
                )
            out[view] = tensor.contiguous()
        return out

    @torch.no_grad()
    def encode_state(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(B, state_feature_dim)`` action-free features."""
        features = self.inner_encoder.encode_state_batch(
            self._prepare_images(image_obs_raw),
            proprio_raw.to(self.device, dtype=torch.float32, non_blocking=True),
        )
        if features.shape[-1] != self.state_feature_dim:
            raise RuntimeError(
                f"State feature dim changed: expected {self.state_feature_dim}, got {features.shape[-1]}"
            )
        return features

    @torch.no_grad()
    def encode_chunk(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
        action_sequences_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``(B, chunk_feature_dim)`` action-conditioned features."""
        actions = action_chunk if action_chunk is not None else action_sequences_raw
        if actions is None:
            raise ValueError("action_chunk is required")
        features = self.inner_encoder.encode_chunk_batch(
            self._prepare_images(image_obs_raw),
            proprio_raw.to(self.device, dtype=torch.float32, non_blocking=True),
            actions.to(self.device, dtype=torch.float32, non_blocking=True),
        )
        if features.shape[-1] != self.chunk_feature_dim:
            raise RuntimeError(
                f"Chunk feature dim changed: expected {self.chunk_feature_dim}, got {features.shape[-1]}"
            )
        return features

    @torch.no_grad()
    def encode_state_and_chunk(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return state and action-conditioned chunk features in one forward."""
        state, conditioned = self.inner_encoder.encode_state_and_chunk_batch(
            self._prepare_images(image_obs_raw),
            proprio_raw.to(self.device, dtype=torch.float32, non_blocking=True),
            action_chunk.to(self.device, dtype=torch.float32, non_blocking=True),
        )
        if state.shape[-1] != self.state_feature_dim:
            raise RuntimeError(
                f"State feature dim changed: expected {self.state_feature_dim}, got {state.shape[-1]}"
            )
        if conditioned.shape[-1] != self.chunk_feature_dim:
            raise RuntimeError(
                f"Chunk feature dim changed: expected {self.chunk_feature_dim}, got {conditioned.shape[-1]}"
            )
        return state, conditioned

    @torch.no_grad()
    def encode(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_real: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper: state-only if action is absent, chunk otherwise."""
        if action_real is None:
            return self.encode_state(image_obs_raw=image_obs_raw, proprio_raw=proprio_raw)
        return self.encode_chunk(
            image_obs_raw=image_obs_raw,
            proprio_raw=proprio_raw,
            action_chunk=action_real,
        )

    @torch.no_grad()
    def encode_features(
        self,
        *,
        chunk_images: torch.Tensor,
        chunk_proprio: torch.Tensor,
        chunk_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode every chunk frame into aligned state and action-window features."""
        if chunk_images.ndim != 6:
            raise ValueError(
                f"chunk_images must be (B, H, V, 3, h, w); got {tuple(chunk_images.shape)}"
            )
        batch, horizon = chunk_images.shape[:2]
        if chunk_proprio.shape[:2] != (batch, horizon):
            raise ValueError("chunk_proprio leading dimensions must match chunk_images")
        if chunk_actions.shape[:2] != (batch, horizon):
            raise ValueError("chunk_actions leading dimensions must match chunk_images")

        flat_images = chunk_images.reshape(batch * horizon, *chunk_images.shape[2:])
        flat_proprio = chunk_proprio.reshape(batch * horizon, -1)
        windows = []
        for step in range(horizon):
            suffix = chunk_actions[:, step:, :]
            windows.append(
                self.inner_encoder._prepare_action_chunks_tensor(suffix, batch)  # noqa: SLF001
            )
        flat_windows = torch.stack(windows, dim=1).reshape(batch * horizon, -1)
        state, conditioned = self.inner_encoder.encode_state_and_chunk_batch(
            self._prepare_images(flat_images),
            flat_proprio.to(self.device, dtype=torch.float32, non_blocking=True),
            flat_windows,
        )
        if state.shape[-1] != self.state_feature_dim:
            raise RuntimeError(
                f"State feature dim changed: expected {self.state_feature_dim}, got {state.shape[-1]}"
            )
        if conditioned.shape[-1] != self.chunk_feature_dim:
            raise RuntimeError(
                f"Chunk feature dim changed: expected {self.chunk_feature_dim}, got {conditioned.shape[-1]}"
            )
        return (
            state.reshape(batch, horizon, self.state_feature_dim),
            conditioned.reshape(batch, horizon, self.chunk_feature_dim),
        )

    @torch.no_grad()
    def encode_chunk_frames(
        self,
        *,
        chunk_images: torch.Tensor,
        chunk_proprio: torch.Tensor,
        chunk_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the per-frame action-conditioned component."""
        return self.encode_features(
            chunk_images=chunk_images,
            chunk_proprio=chunk_proprio,
            chunk_actions=chunk_actions,
        )[1]

    @property
    def context_dim(self) -> int:
        """Transitional alias for callers migrating from LPB context features."""
        return self.chunk_feature_dim

    @property
    def view_names(self) -> list[str]:
        return list(self._view_names)


SharedFrozenEncoder = SharedDynamicsEncoder

__all__ = ["SharedDynamicsEncoder", "SharedFrozenEncoder", "resolve_encoder_checkpoint"]
