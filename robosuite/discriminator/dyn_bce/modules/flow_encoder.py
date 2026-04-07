from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path

from robosuite.discriminator.dyn_bce.task_registry import resolve_checkpoint_task_name
from robosuite.policy.flow_multi.eval_flow import center_crop_resize, resolve_language_instruction
from robosuite.policy.flow_multi.model import build_flow_policy
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor


@dataclass(frozen=True)
class EncodedDemo:
    latents: np.ndarray
    actions: np.ndarray
    task_name: str
    file_path: str
    demo_key: str


@dataclass(frozen=True)
class PreparedDemo:
    states: np.ndarray
    actions: np.ndarray
    images_hwc: np.ndarray
    task_name: str
    file_path: str
    demo_key: str


def _torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class FrozenFlowMultitaskEncoder:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 128,
        batch_size: int = 96,
    ) -> None:
        self.checkpoint_path = to_absolute_path(str(checkpoint_path))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

        checkpoint = _torch_load_checkpoint(self.checkpoint_path)
        self.checkpoint = checkpoint
        if "camera_names" not in checkpoint or "model_cfg" not in checkpoint:
            raise ValueError(
                "Invalid flow policy checkpoint: "
                f"{self.checkpoint_path}. Expected keys like `camera_names` and `model_cfg`, "
                "but they were missing. This usually means policy.ckpt was set to a DSM "
                "checkpoint instead of the frozen flow policy checkpoint."
            )
        self.camera_names = [str(name) for name in checkpoint["camera_names"]]
        self.task_prompt_map = dict(checkpoint.get("task_prompt_map", {}))
        self.task_metadata_map = dict(checkpoint.get("task_metadata_map", {}))
        self.prop_mean = checkpoint.get("prop_mean")
        self.prop_std = checkpoint.get("prop_std")
        if torch.is_tensor(self.prop_mean):
            self.prop_mean = self.prop_mean.detach().cpu().numpy()
        if torch.is_tensor(self.prop_std):
            self.prop_std = self.prop_std.detach().cpu().numpy()
        self.prop_mean = None if self.prop_mean is None else np.asarray(self.prop_mean, dtype=np.float32)
        self.prop_std = None if self.prop_std is None else np.asarray(self.prop_std, dtype=np.float32)

        model_cfg = checkpoint["model_cfg"]
        action_dim = int(np.asarray(checkpoint["act_mean"]).shape[-1])
        proprio_dim = int(np.asarray(checkpoint["prop_mean"]).shape[-1])
        self.latent_dim = int(model_cfg["aggregator"]["output_dim"])
        self.action_dim = action_dim

        self.model = build_flow_policy(
            model_cfg,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            camera_names=self.camera_names,
        ).to(self.device)
        state_dict = checkpoint.get("ema_model", checkpoint["model"])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._extractors: dict[str, RobosuiteProprioExtractor] = {}
        self._image_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)
        self._image_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)

    def close(self) -> None:
        for extractor in self._extractors.values():
            extractor.close()
        self._extractors.clear()

    def _get_extractor(self, task_name: str) -> RobosuiteProprioExtractor:
        ckpt_task_name = resolve_checkpoint_task_name(task_name)
        if ckpt_task_name not in self._extractors:
            if ckpt_task_name not in self.task_metadata_map:
                raise KeyError(f"Task '{ckpt_task_name}' missing from checkpoint task_metadata_map")
            env_metadata = dict(self.task_metadata_map[ckpt_task_name])
            self._extractors[ckpt_task_name] = RobosuiteProprioExtractor(
                env_kwargs=env_metadata,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                camera_names=None,
                reward_shaping=False,
            )
        return self._extractors[ckpt_task_name]

    def resolve_language_instruction(self, task_name: str) -> str:
        ckpt_task_name = resolve_checkpoint_task_name(task_name)
        return resolve_language_instruction(self.task_prompt_map, ckpt_task_name)

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        x = np.asarray(proprio, dtype=np.float32)
        if self.prop_mean is not None and self.prop_std is not None:
            x = (x - self.prop_mean.reshape(1, -1)) / (self.prop_std.reshape(1, -1) + 1e-6)
        return x

    def _prepare_images(self, images_hwc: np.ndarray) -> np.ndarray:
        if not images_hwc.ndim == 5:
            raise ValueError(f"Expected image tensor (T,V,H,W,C), got {images_hwc.shape}")
        timesteps, num_cameras = int(images_hwc.shape[0]), int(images_hwc.shape[1])
        if not num_cameras == len(self.camera_names):
            raise ValueError(
                f"Expected {len(self.camera_names)} cameras from checkpoint, got {num_cameras}"
            )
        prepared = np.empty(
            (timesteps, num_cameras, 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        for t in range(timesteps):
            for view_idx in range(num_cameras):
                cropped = center_crop_resize(images_hwc[t, view_idx], self.image_size)
                prepared[t, view_idx] = np.transpose(cropped.astype(np.float32) / 255.0, (2, 0, 1))
        return prepared

    def _compute_proprio(self, task_name: str, states: np.ndarray) -> np.ndarray:
        extractor = self._get_extractor(task_name)
        proprio = np.stack(
            [extractor.extract(np.asarray(state, dtype=np.float32)) for state in states],
            axis=0,
        ).astype(np.float32)
        return self._normalize_proprio(proprio)

    @torch.no_grad()
    def encode_arrays(
        self,
        task_name: str,
        states: np.ndarray,
        images_hwc: np.ndarray,
    ) -> np.ndarray:
        language_instruction = self.resolve_language_instruction(task_name)
        proprio = self._compute_proprio(task_name=task_name, states=states)
        images_chw = self._prepare_images(images_hwc)

        outputs: list[np.ndarray] = []
        for start in range(0, images_chw.shape[0], self.batch_size):
            stop = min(images_chw.shape[0], start + self.batch_size)
            image_batch = torch.from_numpy(images_chw[start:stop]).to(self.device)
            prop_batch = torch.from_numpy(proprio[start:stop]).to(self.device)
            image_batch = (image_batch - self._image_mean) / self._image_std
            embedding = self.model.encode_context(
                images=image_batch,
                proprio=prop_batch,
                language=[language_instruction] * int(image_batch.shape[0]),
            )
            outputs.append(embedding.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def encode_demo(self, task_name: str, file_path: str, demo_key: str) -> EncodedDemo:
        prepared = self.load_demo_raw(task_name=task_name, file_path=file_path, demo_key=demo_key)
        return self.encode_prepared_demos([prepared])[0]

    def load_demo_raw(self, task_name: str, file_path: str, demo_key: str) -> PreparedDemo:
        with h5py.File(file_path, "r") as file_handle:
            demo = file_handle["demos"][demo_key]
            states = np.asarray(demo["states"][:], dtype=np.float32)
            actions = np.asarray(demo["actions"][:], dtype=np.float32)

            image_views = []
            lengths = [int(states.shape[0]), int(actions.shape[0])]
            for camera_name in self.camera_names:
                cam_images = np.asarray(
                    demo["observations"][camera_name]["images"][:],
                    dtype=np.uint8,
                )
                image_views.append(cam_images)
                lengths.append(int(cam_images.shape[0]))

        length = min(lengths)
        if length <= 1:
            raise ValueError(f"Demo {demo_key} from {file_path} is too short: {length}")

        stacked_images = np.stack([images[:length] for images in image_views], axis=1)
        return PreparedDemo(
            states=states[:length],
            actions=actions[:length],
            images_hwc=stacked_images,
            task_name=task_name,
            file_path=file_path,
            demo_key=demo_key,
        )

    @torch.no_grad()
    def encode_prepared_demos(self, prepared_demos: list[PreparedDemo]) -> list[EncodedDemo]:
        if not prepared_demos:
            return []

        image_batches: list[np.ndarray] = []
        proprio_batches: list[np.ndarray] = []
        action_batches: list[np.ndarray] = []
        language_batches: list[list[str]] = []
        metadata: list[tuple[str, str, str, int]] = []

        for prepared in prepared_demos:
            proprio = self._compute_proprio(task_name=prepared.task_name, states=prepared.states)
            images_chw = self._prepare_images(prepared.images_hwc)
            length = min(int(proprio.shape[0]), int(images_chw.shape[0]), int(prepared.actions.shape[0]))
            if length <= 1:
                continue
            image_batches.append(images_chw[:length])
            proprio_batches.append(proprio[:length].astype(np.float32))
            action_batches.append(prepared.actions[:length].astype(np.float32))
            language_batches.append([self.resolve_language_instruction(prepared.task_name)] * length)
            metadata.append((prepared.task_name, prepared.file_path, prepared.demo_key, length))

        if not metadata:
            return []

        all_images = np.concatenate(image_batches, axis=0)
        all_proprio = np.concatenate(proprio_batches, axis=0)
        all_languages = [lang for batch in language_batches for lang in batch]

        outputs: list[np.ndarray] = []
        for start in range(0, all_images.shape[0], self.batch_size):
            stop = min(all_images.shape[0], start + self.batch_size)
            image_batch = torch.from_numpy(all_images[start:stop]).to(self.device)
            prop_batch = torch.from_numpy(all_proprio[start:stop]).to(self.device)
            image_batch = (image_batch - self._image_mean) / self._image_std
            embedding = self.model.encode_context(
                images=image_batch,
                proprio=prop_batch,
                language=all_languages[start:stop],
            )
            outputs.append(embedding.detach().cpu().numpy().astype(np.float32))
        all_latents = np.concatenate(outputs, axis=0)

        encoded_demos: list[EncodedDemo] = []
        offset = 0
        for (task_name, file_path, demo_key, length), actions in zip(metadata, action_batches):
            latents = all_latents[offset : offset + length]
            encoded_demos.append(
                EncodedDemo(
                    latents=latents,
                    actions=actions[:length],
                    task_name=task_name,
                    file_path=file_path,
                    demo_key=demo_key,
                )
            )
            offset += length
        return encoded_demos

    def cache_key(self, task_name: str, file_path: str, demo_key: str) -> str:
        key_str = "|".join(
            [
                str(task_name),
                str(Path(file_path).resolve()),
                str(demo_key),
                str(self.image_size),
                ",".join(self.camera_names),
                str(os.path.getmtime(file_path)),
                str(Path(self.checkpoint_path).resolve()),
            ]
        )
        return hashlib.sha1(key_str.encode("utf-8")).hexdigest()

    def cache_path(self, cache_root: str, task_name: str, file_path: str, demo_key: str) -> str:
        cache_dir = os.path.join(to_absolute_path(str(cache_root)), normalize_path_component(task_name))
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.cache_key(task_name, file_path, demo_key)}.npz")

    def load_or_encode_demo(
        self,
        cache_root: str,
        task_name: str,
        file_path: str,
        demo_key: str,
    ) -> EncodedDemo:
        cache_path = self.cache_path(
            cache_root=cache_root,
            task_name=task_name,
            file_path=file_path,
            demo_key=demo_key,
        )
        if os.path.isfile(cache_path):
            with np.load(cache_path) as cached:
                return EncodedDemo(
                    latents=np.asarray(cached["latents"], dtype=np.float32),
                    actions=np.asarray(cached["actions"], dtype=np.float32),
                    task_name=str(cached["task_name"].item()),
                    file_path=str(cached["file_path"].item()),
                    demo_key=str(cached["demo_key"].item()),
                )

        encoded = self.encode_demo(task_name=task_name, file_path=file_path, demo_key=demo_key)
        self.save_encoded_demo(cache_root=cache_root, encoded=encoded)
        return encoded

    def save_encoded_demo(self, cache_root: str, encoded: EncodedDemo) -> str:
        cache_path = self.cache_path(
            cache_root=cache_root,
            task_name=encoded.task_name,
            file_path=encoded.file_path,
            demo_key=encoded.demo_key,
        )
        np.savez_compressed(
            cache_path,
            latents=encoded.latents.astype(np.float32),
            actions=encoded.actions.astype(np.float32),
            task_name=np.asarray(encoded.task_name),
            file_path=np.asarray(encoded.file_path),
            demo_key=np.asarray(encoded.demo_key),
        )
        return cache_path


def normalize_path_component(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")
