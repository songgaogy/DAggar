"""Flow-Multi policy encoder wrapper with cache reuse.

Strip-down of the historical ``lpb_score.core.policy_encoder.FlowMultitaskEncoder``
(git 5056929) that keeps only the bits needed to produce per-frame 256-D latents
for a demo trajectory, and to reuse the existing ``data/.lpb_score_cache/`` npz
cache byte-exactly. No LoRA, no trainable path, no profiling, no raw-tensor
bundle cache.

Cache file format (unchanged):
    ``<cache_root>/<task_name>/<sha1>.npz`` with keys
    ``latents (T,256) float32``, ``actions (T,7) float32``,
    ``task_name``, ``file_path``, ``demo_key`` (string scalars).
"""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn

from robosuite.policy.flow_multi.eval_flow import resolve_language_instruction
from robosuite.policy.flow_multi.model import build_flow_policy
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor


# Alias table copied from historical lpb_score so cache_key strings match byte-for-byte.
# Canonical (benchmark) name -> checkpoint-side task name (used for task_metadata_map /
# task_prompt_map lookups). Extra entries kept only for robustness.
_TASK_TO_CKPT: dict[str, str] = {
    "PandaLift": "Lift",
    "Lift": "Lift",
    "PandaStack": "Stack",
    "Stack": "Stack",
    "PandaPickPlaceCan": "PickPlaceCan",
    "PickPlaceCan": "PickPlaceCan",
    "PickPlaceBread": "PickPlaceBread",
    "PickPlaceCereal": "PickPlaceCereal",
    "PickPlaceMilk": "PickPlaceMilk",
}


def _resolve_checkpoint_task_name(task_name: str) -> str:
    if task_name not in _TASK_TO_CKPT:
        raise KeyError(f"Unsupported task name: {task_name!r}")
    return _TASK_TO_CKPT[task_name]


def _normalize_path_component(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class EncodedDemo:
    """Per-demo output: per-frame 256-D latent + raw action sequence."""

    latents: np.ndarray       # (T, 256) float32
    actions: np.ndarray       # (T, 7) float32
    task_name: str
    file_path: str
    demo_key: str


class FlowMultiEncoderWrapper(nn.Module):
    """Loads the flow_multi policy and emits per-frame condition embeddings.

    Only the forward pass through ``MultiModalFlowPolicy.encode_context`` is used
    (no action sampling). Images/proprio preprocessing must match the historical
    encoder that populated the on-disk cache — any drift breaks bit-exact reuse.
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        *,
        cache_root: str = "data/.lpb_score_cache",
        device: str = "cuda",
        image_size: int = 128,
        encoder_batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.checkpoint_path = str(Path(policy_ckpt_path).resolve())
        self.cache_root = str(cache_root)
        self.image_size = int(image_size)
        self.encoder_batch_size = int(encoder_batch_size)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = _torch_load(self.checkpoint_path)
        for required in ("camera_names", "model_cfg", "act_mean", "prop_mean"):
            if required not in checkpoint:
                raise ValueError(f"Checkpoint missing required key {required!r}: {self.checkpoint_path}")

        self.camera_names: list[str] = [str(name) for name in checkpoint["camera_names"]]
        self.task_prompt_map: dict = dict(checkpoint.get("task_prompt_map", {}))
        self.task_metadata_map: dict = dict(checkpoint.get("task_metadata_map", {}))
        self.model_cfg = copy.deepcopy(checkpoint["model_cfg"])
        self.act_mean = _to_numpy(checkpoint.get("act_mean"))
        self.act_std = _to_numpy(checkpoint.get("act_std"))
        self.prop_mean = _to_numpy(checkpoint.get("prop_mean"))
        self.prop_std = _to_numpy(checkpoint.get("prop_std"))

        action_dim = int(np.asarray(self.act_mean).shape[-1])
        proprio_dim = int(np.asarray(self.prop_mean).shape[-1])
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.latent_dim = int(self.model_cfg["aggregator"]["output_dim"])

        self.model = build_flow_policy(
            self.model_cfg,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            camera_names=self.camera_names,
        )
        state_dict = checkpoint.get("ema_model", checkpoint.get("model"))
        if state_dict is None:
            raise ValueError("Checkpoint must contain 'ema_model' or 'model' state_dict")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # ImageNet normalization (broadcast to (1,1,3,1,1)). Placed on the same
        # device as the encoder weights so no per-batch host->device copy is needed.
        self.register_buffer(
            "_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )
        self._image_mean = self._image_mean.to(self.device)
        self._image_std = self._image_std.to(self.device)

        # Per-task extractor cache (spawns a robosuite env once per task — expensive).
        self._extractors: dict[str, RobosuiteProprioExtractor] = {}
        self._language_cache: dict[str, str] = {}
        self._resize_index_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    # Task-side helpers                                                  #
    # ------------------------------------------------------------------ #

    def _get_extractor(self, task_name: str) -> RobosuiteProprioExtractor:
        ckpt_task = _resolve_checkpoint_task_name(task_name)
        if ckpt_task not in self._extractors:
            if ckpt_task not in self.task_metadata_map:
                raise KeyError(f"Task {ckpt_task!r} missing from checkpoint task_metadata_map")
            env_kwargs = dict(self.task_metadata_map[ckpt_task])
            self._extractors[ckpt_task] = RobosuiteProprioExtractor(
                env_kwargs=env_kwargs,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                camera_names=None,
                reward_shaping=False,
            )
        return self._extractors[ckpt_task]

    def _resolve_language(self, task_name: str) -> str:
        ckpt_task = _resolve_checkpoint_task_name(task_name)
        cached = self._language_cache.get(ckpt_task)
        if cached is not None:
            return cached
        resolved = resolve_language_instruction(self.task_prompt_map, ckpt_task)
        self._language_cache[ckpt_task] = resolved
        return resolved

    # ------------------------------------------------------------------ #
    # Preprocessing                                                      #
    # ------------------------------------------------------------------ #

    def _resize_center_crop_video(self, images: np.ndarray) -> np.ndarray:
        """Center crop to min(H,W) then nearest-index resize to ``image_size``.

        Must match historical implementation exactly (np.linspace int32 indices).
        """
        height, width = images.shape[1:3]
        crop_size = min(height, width)
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        crop = images[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
        if crop_size == self.image_size:
            return crop
        if crop_size not in self._resize_index_cache:
            ys = np.linspace(0, crop_size - 1, self.image_size).astype(np.int32)
            xs = np.linspace(0, crop_size - 1, self.image_size).astype(np.int32)
            self._resize_index_cache[crop_size] = (ys, xs)
        ys, xs = self._resize_index_cache[crop_size]
        return crop[:, ys][:, :, xs]

    def _prepare_images(self, images_hwc: np.ndarray) -> np.ndarray:
        """(T,V,H,W,3) uint8 -> (T,V,3,S,S) float32 in [0,1]."""
        if images_hwc.ndim != 5:
            raise ValueError(f"Expected image tensor (T,V,H,W,C), got {images_hwc.shape}")
        timesteps, num_cameras = int(images_hwc.shape[0]), int(images_hwc.shape[1])
        if num_cameras != len(self.camera_names):
            raise ValueError(
                f"Expected {len(self.camera_names)} cameras from checkpoint, got {num_cameras}"
            )
        views: list[np.ndarray] = []
        for view_idx in range(num_cameras):
            resized = self._resize_center_crop_video(images_hwc[:, view_idx])
            views.append(np.transpose(resized.astype(np.float32) / 255.0, (0, 3, 1, 2)))
        stacked = np.stack(views, axis=1)
        return np.ascontiguousarray(
            stacked.reshape(timesteps, num_cameras, 3, self.image_size, self.image_size)
        )

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        x = np.asarray(proprio, dtype=np.float32)
        if self.prop_mean is not None and self.prop_std is not None:
            x = (x - self.prop_mean.reshape(1, -1)) / (self.prop_std.reshape(1, -1) + 1e-6)
        return x.astype(np.float32, copy=False)

    def _compute_proprio(self, task_name: str, states: np.ndarray) -> np.ndarray:
        """Slice qpos/qvel from flattened mujoco state via per-task extractor."""
        extractor = self._get_extractor(task_name)
        state_array = np.asarray(states, dtype=np.float32)
        if state_array.ndim != 2:
            raise ValueError(f"Expected states (T,S), got {state_array.shape}")
        nq = int(extractor.sim.model.nq)
        nv = int(extractor.sim.model.nv)
        na = int(getattr(extractor.sim.model, "na", 0))
        core0 = nq + nv
        core1 = nq + nv + na
        size = int(state_array.shape[1])

        if size == core0:
            base = state_array
        elif size == core1:
            base = state_array[:, :core0]
        elif size == core0 + 1:
            base = state_array[:, 1 : core0 + 1]
        elif size == core1 + 1:
            base = state_array[:, 1 : core0 + 1]
        else:
            raise ValueError(f"Unexpected flattened state length: {size}")

        qpos = base[:, :nq]
        qvel = base[:, nq : nq + nv]
        proprio = np.concatenate(
            [qpos[:, extractor.qpos_indices], qvel[:, extractor.qvel_indices]],
            axis=1,
        ).astype(np.float32)
        return self._normalize_proprio(proprio)

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _encode_context_tensors(
        self,
        images: torch.Tensor,   # (B, V, 3, S, S) float in [0,1]
        proprio: torch.Tensor,  # (B, P) float32
        task_names: Sequence[str],
    ) -> torch.Tensor:
        target_device = self._image_mean.device
        image_batch = images.to(device=target_device, dtype=torch.float32)
        prop_batch = proprio.to(device=target_device, dtype=torch.float32)
        image_batch = (image_batch - self._image_mean.to(dtype=image_batch.dtype)) / self._image_std.to(
            dtype=image_batch.dtype
        )
        language = [self._resolve_language(str(name)) for name in task_names]
        return self.model.encode_context(
            images=image_batch,
            proprio=prop_batch,
            language=language,
        )

    @torch.no_grad()
    def _encode_sequence(
        self,
        images_chw: np.ndarray,     # (T, V, 3, S, S)
        proprio: np.ndarray,        # (T, P)
        task_name: str,
    ) -> np.ndarray:
        image_tensor = torch.as_tensor(images_chw)
        proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32)
        batch = max(1, int(self.encoder_batch_size))
        outputs: list[torch.Tensor] = []
        for start in range(0, int(image_tensor.shape[0]), batch):
            end = min(start + batch, int(image_tensor.shape[0]))
            n = int(end - start)
            latent = self._encode_context_tensors(
                images=image_tensor[start:end],
                proprio=proprio_tensor[start:end],
                task_names=[task_name] * n,
            ).detach().cpu()
            outputs.append(latent)
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # Cache I/O                                                          #
    # ------------------------------------------------------------------ #

    def cache_key(self, task_name: str, file_path: str, demo_key: str) -> str:
        """Byte-match historical convention — any drift here breaks cache reuse."""
        key = "|".join(
            [
                str(task_name),
                str(Path(file_path).resolve()),
                str(demo_key),
                str(self.image_size),
                ",".join(self.camera_names),
                str(os.path.getmtime(file_path)),
                str(self.checkpoint_path),
            ]
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def cache_path(self, task_name: str, file_path: str, demo_key: str) -> str:
        cache_dir = os.path.join(self.cache_root, _normalize_path_component(task_name))
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.cache_key(task_name, file_path, demo_key)}.npz")

    def load_cached(self, task_name: str, file_path: str, demo_key: str) -> EncodedDemo | None:
        path = self.cache_path(task_name, file_path, demo_key)
        if not os.path.isfile(path):
            return None
        with np.load(path) as cached:
            return EncodedDemo(
                latents=np.asarray(cached["latents"], dtype=np.float32),
                actions=np.asarray(cached["actions"], dtype=np.float32),
                task_name=str(cached["task_name"].item()),
                file_path=str(cached["file_path"].item()),
                demo_key=str(cached["demo_key"].item()),
            )

    def save_encoded(self, encoded: EncodedDemo) -> str:
        path = self.cache_path(encoded.task_name, encoded.file_path, encoded.demo_key)
        np.savez_compressed(
            path,
            latents=np.asarray(encoded.latents, dtype=np.float32),
            actions=np.asarray(encoded.actions, dtype=np.float32),
            task_name=np.asarray(encoded.task_name),
            file_path=np.asarray(encoded.file_path),
            demo_key=np.asarray(encoded.demo_key),
        )
        return path

    # ------------------------------------------------------------------ #
    # Public: load demo from hdf5 + encode (with cache)                  #
    # ------------------------------------------------------------------ #

    def _load_demo_from_hdf5(self, file_path: str, demo_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read raw (states, actions, images_hwc) from the source hdf5."""
        with h5py.File(file_path, "r") as fh:
            demo = fh["demos"][demo_key]
            states = np.asarray(demo["states"][:], dtype=np.float32)
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
            image_views = []
            lengths = [int(states.shape[0]), int(actions.shape[0])]
            for camera_name in self.camera_names:
                cam_images = np.asarray(
                    demo["observations"][camera_name]["images"][:], dtype=np.uint8
                )
                image_views.append(cam_images)
                lengths.append(int(cam_images.shape[0]))
        length = min(lengths)
        if length <= 0:
            raise ValueError(f"Demo {demo_key} from {file_path} is empty.")
        stacked = np.stack([imgs[:length] for imgs in image_views], axis=1)
        return states[:length], actions[:length], stacked

    @torch.no_grad()
    def encode_fresh(self, task_name: str, file_path: str, demo_key: str) -> EncodedDemo:
        """Encode from scratch (no cache lookup). Used by Phase A verification."""
        states, actions, images_hwc = self._load_demo_from_hdf5(file_path, demo_key)
        proprio = self._compute_proprio(task_name=task_name, states=states)
        images_chw = self._prepare_images(images_hwc)
        length = min(int(proprio.shape[0]), int(images_chw.shape[0]), int(actions.shape[0]))
        latents = self._encode_sequence(
            images_chw=images_chw[:length],
            proprio=proprio[:length],
            task_name=task_name,
        )
        return EncodedDemo(
            latents=np.asarray(latents[:length], dtype=np.float32),
            actions=np.asarray(actions[:length], dtype=np.float32),
            task_name=str(task_name),
            file_path=str(file_path),
            demo_key=str(demo_key),
        )

    def load_or_encode_demo(self, task_name: str, file_path: str, demo_key: str) -> EncodedDemo:
        """Cache-first: load from ``<cache_root>/<task>/<sha1>.npz`` if present;
        otherwise encode from the source hdf5 and write the npz for future reuse."""
        cached = self.load_cached(task_name=task_name, file_path=file_path, demo_key=demo_key)
        if cached is not None:
            return cached
        encoded = self.encode_fresh(task_name=task_name, file_path=file_path, demo_key=demo_key)
        self.save_encoded(encoded)
        return encoded

    def close(self) -> None:
        for extractor in self._extractors.values():
            try:
                extractor.close()
            except Exception:
                pass
        self._extractors.clear()
