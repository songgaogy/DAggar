"""LPB v2 BCE G provider for DIPOLE.

Loads a pre-fitted BCE checkpoint (the artefact produced by
``robosuite/discriminator/lpb_v2/scripts/run_bce_robosuite_benchmark.sh`` and
``robosuite/discriminator/lpb_v2/scripts/visualize_bce_robosuite.sh``) and runs
the frozen LPB encoder + BCE head on a DipoleBatch.

The checkpoint schema this loader expects (see
``robosuite.discriminator.lpb_v2.detectors.bce.BCEDiscriminator.state_dict``
plus the runner wrapper):

```
{
    "epoch":             int,
    "in_dim":            int,
    "hidden":            int,
    "num_layers":        int,
    "bce_detector":      dict,        # BCEDiscriminator.state_dict()
    "feature_source":    str,         # "encoder" | "transformer"
    "transformer_layer": int,
    "model_ckpt":        str,         # path to LPB v2 dynamics .pth
    "calib_mode":        str,
    "fail_bank_video_ids": list,
    "fail_calib_video_ids": list,
}
```

Score convention (matches BCEDiscriminator.score): ``raw = -g(z)`` where ``g(z)``
is the BCE head's expert-likeness logit, so larger raw means more failure-like.
``DipoleFlowPolicy`` flips this with ``g_sign='negate_raw'`` so larger G means
"more expert/safe".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.discriminator.lpb_v2 import BCEDiscriminator, LPBV2Encoder


def _resolve_encoder_ckpt(model_ckpt: str, *, hint_root: Path | None = None) -> Path:
    """Best-effort resolve a possibly relative encoder ckpt path."""
    candidate = Path(model_ckpt)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    # Try relative to the BCE artefact directory, then the repo root.
    candidates = []
    if hint_root is not None:
        candidates.append((hint_root / model_ckpt).resolve())
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append((repo_root / model_ckpt).resolve())
    candidates.append(Path.cwd().joinpath(model_ckpt).resolve())
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not locate LPB encoder checkpoint '{model_ckpt}' referenced by the "
        f"BCE artefact. Tried: {[str(c) for c in candidates]}"
    )


class LPBV2GProvider:
    """BCE-only provider tied to the pre-fitted ``bce_head.pth`` schema.

    Args:
        ckpt_path: path to the BCE checkpoint (e.g. ``checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth``).
        task_name: name of the task whose calibrated threshold to use for the
            BCE score lookup. Must be a key in ``state['bce_detector']['thresholds']``.
            Currently DIPOLE only consumes raw step scores (not thresholded preds),
            so this serves as both a sanity check and the BCEDiscriminator.score lookup key.
        device: torch device string.
        camera_to_view: optional mapping ``{policy_camera_name: encoder_view_name}``
            used to pick the right camera channels out of the policy's DipoleBatch.
            Defaults to identity (encoder view name = policy camera name).
    """

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        task_name: str,
        device: str | torch.device = "cuda:0",
        camera_to_view: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path).resolve()
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"BCE checkpoint not found: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.task_name = str(task_name)

        for required in ("bce_detector", "feature_source", "transformer_layer", "model_ckpt"):
            if required not in ckpt:
                raise KeyError(
                    f"BCE checkpoint {self.ckpt_path} is missing required key '{required}'. "
                    "Expected the schema produced by run_bce_robosuite_benchmark.sh / "
                    "visualize_bce_robosuite.sh."
                )

        self.feature_source: str = str(ckpt["feature_source"])
        self.transformer_layer: int = int(ckpt["transformer_layer"])

        encoder_ckpt_path = _resolve_encoder_ckpt(
            str(ckpt["model_ckpt"]), hint_root=self.ckpt_path.parent
        )
        self.encoder = LPBV2Encoder(
            model_ckpt=str(encoder_ckpt_path),
            device=str(self.device),
            feature_source=self.feature_source,
            transformer_layer=self.transformer_layer,
        )
        self.view_names: list[str] = [str(v) for v in self.encoder.view_names]
        self.frameskip: int = int(self.encoder.frameskip)
        self.action_dim_per_step: int = int(self.encoder.action_dim_per_step)
        self.action_input_dim: int = int(self.encoder.action_input_dim)
        self.original_img_size: int = int(self.encoder.original_img_size)

        # Build + load the BCE head.
        bce_state = dict(ckpt["bce_detector"])
        in_dim = int(bce_state.get("in_dim", ckpt.get("in_dim", 0)))
        hidden = int(bce_state.get("hidden", ckpt.get("hidden", 256)))
        num_layers = int(bce_state.get("num_layers", ckpt.get("num_layers", 2)))
        if in_dim <= 0:
            raise ValueError(f"BCE checkpoint {self.ckpt_path} has invalid in_dim={in_dim}")
        self.detector = BCEDiscriminator(
            in_dim=in_dim,
            hidden=hidden,
            num_layers=num_layers,
            device=str(self.device),
        )
        self.detector.load_state_dict(bce_state)
        self.detector.head.eval()

        if not self.detector.thresholds:
            raise RuntimeError(
                f"BCE checkpoint {self.ckpt_path} has no calibrated thresholds."
            )
        if self.task_name not in self.detector.thresholds:
            raise KeyError(
                f"task_name={self.task_name!r} not found in BCE thresholds "
                f"(available: {sorted(self.detector.thresholds)}). "
                "Set algorithm.task_name to one of the calibrated tasks."
            )
        self.threshold: float = float(self.detector.thresholds[self.task_name])

        self._camera_to_view: dict[str, str] = dict(camera_to_view or {})
        self._policy_camera_names: list[str] | None = None
        self._view_index_in_batch: list[int] | None = None

    def bind_policy_cameras(self, policy_camera_names: list[str]) -> None:
        """Tell the provider which camera order the DipoleBatch will use.

        For each view the LPB encoder expects (self.view_names), resolve the
        index in the policy camera tuple via ``camera_to_view`` (defaulting to
        identity). Raises if a required view is missing.
        """
        self._policy_camera_names = [str(c) for c in policy_camera_names]
        forward_map = {str(c): str(v) for c, v in self._camera_to_view.items()}
        # We need view -> policy camera; invert when camera_to_view is given.
        reverse_map: dict[str, str] = {}
        for camera, view in forward_map.items():
            reverse_map[view] = camera

        view_to_index: list[int] = []
        for view in self.view_names:
            # If user supplied a reverse mapping use it; else fall back to identity.
            policy_camera = reverse_map.get(view, view)
            if policy_camera not in self._policy_camera_names:
                raise KeyError(
                    f"LPB encoder requires view '{view}' but the policy's camera tuple is "
                    f"{self._policy_camera_names}. Either include this camera in env.camera_names "
                    f"or pass camera_to_view={{policy_camera_name: '{view}'}} in the lpb_detector config."
                )
            view_to_index.append(self._policy_camera_names.index(policy_camera))
        self._view_index_in_batch = view_to_index

    def _resize_to_encoder(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> (B, 3, H_enc, W_enc) bilinear if needed."""
        target = self.original_img_size
        if images.shape[-1] == target and images.shape[-2] == target:
            return images
        return F.interpolate(images, size=(target, target), mode="bilinear", align_corners=False)

    def _prepare_action_input(self, actions_raw: torch.Tensor) -> torch.Tensor:
        B, horizon, A = actions_raw.shape
        A_d = self.action_dim_per_step
        if A < A_d:
            pad = torch.zeros(B, horizon, A_d - A, device=actions_raw.device, dtype=actions_raw.dtype)
            acts = torch.cat([actions_raw, pad], dim=-1)
        elif A > A_d:
            acts = actions_raw[..., :A_d]
        else:
            acts = actions_raw

        if self.feature_source == "transformer":
            return acts[:, 0, :].contiguous()

        fs = self.frameskip
        if horizon >= fs:
            window = acts[:, :fs, :]
        else:
            pad_steps = fs - horizon
            last = acts[:, -1:, :].expand(-1, pad_steps, -1)
            window = torch.cat([acts, last], dim=1)
        flat = window.reshape(B, fs * A_d)
        target = self.action_input_dim
        if flat.shape[1] < target:
            pad = torch.zeros(B, target - flat.shape[1], device=flat.device, dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[1] > target:
            flat = flat[:, :target]
        return flat.contiguous()

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        if self._view_index_in_batch is None:
            raise RuntimeError(
                "LPBV2GProvider.bind_policy_cameras(...) must be called before compute_g_for_batch."
            )
        images_raw = batch.image_obs_raw.to(self.device, dtype=torch.float32)
        if images_raw.dim() != 5:
            raise ValueError(
                f"DipoleBatch.image_obs_raw expected (B, V, 3, H, W), got shape {tuple(images_raw.shape)}"
            )
        B = images_raw.shape[0]
        images_per_view: dict[str, torch.Tensor] = {}
        for view, idx in zip(self.view_names, self._view_index_in_batch):
            view_tensor = images_raw[:, idx]  # (B, 3, H_pol, W_pol)
            view_tensor = self._resize_to_encoder(view_tensor)
            images_per_view[view] = view_tensor.contiguous()

        proprio = batch.proprio_raw.to(self.device, dtype=torch.float32)
        actions = self._prepare_action_input(
            batch.action_sequences_raw.to(self.device, dtype=torch.float32)
        )
        feat = self.encoder.encode_batch(images_per_view, proprio, actions)
        result = self.detector.score(feat, task=self.task_name)
        return torch.as_tensor(result.step_scores, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def compute_g_for_observation(
        self,
        obs: Mapping[str, Any],
        action_sequence: np.ndarray,
    ) -> dict[str, float]:
        """Single-frame BCE score for live discriminator display.

        Args:
            obs: dict keyed by policy camera name (HxWx3 uint8 in [0, 255]) plus
                ``"state"`` (proprio_dim,) float.
            action_sequence: (horizon, action_dim) un-normalized policy actions.

        Returns:
            dict with ``raw`` (-g(z), higher = more failure-like), ``tau``
            (per-task BCE threshold in raw space) and ``is_failure``
            (``1`` if ``raw >= tau`` else ``0``).
        """
        if self._view_index_in_batch is None:
            raise RuntimeError(
                "LPBV2GProvider.bind_policy_cameras(...) must be called before compute_g_for_observation."
            )
        images_per_view: dict[str, torch.Tensor] = {}
        for view, camera_idx in zip(self.view_names, self._view_index_in_batch):
            camera_name = self._policy_camera_names[camera_idx]
            arr = np.asarray(obs[camera_name])
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"Expected obs[{camera_name!r}] to be HxWx3, got shape {tuple(arr.shape)}"
                )
            if arr.dtype == np.uint8:
                tensor = torch.from_numpy(arr).to(self.device, dtype=torch.float32) / 255.0
            else:
                tensor = torch.from_numpy(arr.astype(np.float32)).to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            images_per_view[view] = self._resize_to_encoder(tensor).contiguous()

        proprio = torch.as_tensor(
            np.asarray(obs["state"], dtype=np.float32), device=self.device
        ).reshape(1, -1)
        actions_np = np.asarray(action_sequence, dtype=np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        actions_raw = torch.from_numpy(actions_np).to(self.device).unsqueeze(0)
        actions_input = self._prepare_action_input(actions_raw)

        feat = self.encoder.encode_batch(images_per_view, proprio, actions_input)
        result = self.detector.score(feat, task=self.task_name)
        raw = float(np.asarray(result.step_scores).reshape(-1)[0])
        return {
            "raw": raw,
            "tau": float(self.threshold),
            "is_failure": int(raw >= self.threshold),
        }


__all__ = ["LPBV2GProvider"]
