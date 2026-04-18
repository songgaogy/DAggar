from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import to_absolute_path

from robosuite.discriminator.dyn_bce.task_registry import resolve_checkpoint_task_name
from robosuite.policy.flow_multi.eval_flow import resolve_language_instruction
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


@dataclass(frozen=True)
class PreparedEncoderDemo:
    images_chw: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    task_name: str
    file_path: str
    demo_key: str


def _torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_path_component(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_linear = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base_linear.parameters():
            param.requires_grad = False

    @property
    def in_features(self) -> int:
        return int(self.base_linear.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base_linear.out_features)

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base_linear.bias

    @property
    def weight(self) -> torch.Tensor:
        delta = torch.matmul(self.lora_b.weight, self.lora_a.weight) * self.scaling
        return self.base_linear.weight + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base_linear.weight, self.base_linear.bias)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


class FlowMultitaskEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        checkpoint_payload: dict[str, Any] | None = None,
        device: str = "cuda",
        image_size: int = 128,
        batch_size: int = 96,
        trainable: bool = False,
        lora_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if checkpoint_path is None and checkpoint_payload is None:
            raise ValueError("Provide checkpoint_path or checkpoint_payload.")
        if checkpoint_path is not None and checkpoint_payload is not None:
            raise ValueError("Provide only one of checkpoint_path or checkpoint_payload.")

        self.checkpoint_path = None if checkpoint_path is None else to_absolute_path(str(checkpoint_path))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.trainable = bool(trainable)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

        checkpoint = (
            copy.deepcopy(dict(checkpoint_payload))
            if checkpoint_payload is not None
            else _torch_load_checkpoint(str(self.checkpoint_path))
        )
        self._validate_checkpoint_payload(checkpoint)
        self.policy_checkpoint_payload = self._export_policy_payload(checkpoint)
        self.checkpoint = copy.deepcopy(self.policy_checkpoint_payload)
        self.source_description = (
            "embedded_dsm_checkpoint_payload"
            if checkpoint_payload is not None
            else str(self.checkpoint_path)
        )

        self.camera_names = [str(name) for name in checkpoint["camera_names"]]
        self.task_prompt_map = dict(checkpoint.get("task_prompt_map", {}))
        self.task_metadata_map = dict(checkpoint.get("task_metadata_map", {}))
        self.model_cfg = copy.deepcopy(checkpoint["model_cfg"])
        self.act_mean = self._to_numpy(checkpoint.get("act_mean"))
        self.act_std = self._to_numpy(checkpoint.get("act_std"))
        self.prop_mean = self._to_numpy(checkpoint.get("prop_mean"))
        self.prop_std = self._to_numpy(checkpoint.get("prop_std"))

        if self.act_mean is None or self.prop_mean is None:
            raise ValueError("Flow encoder checkpoint must include act_mean and prop_mean.")

        action_dim = int(np.asarray(self.act_mean).shape[-1])
        proprio_dim = int(np.asarray(self.prop_mean).shape[-1])
        self.latent_dim = int(self.model_cfg["aggregator"]["output_dim"])
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)

        self.model = build_flow_policy(
            self.model_cfg,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            camera_names=self.camera_names,
        )
        state_dict = checkpoint.get("ema_model", checkpoint.get("model"))
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=True)
        self.lora_cfg = self._resolve_lora_cfg(
            checkpoint_lora_cfg=checkpoint.get("lora_cfg", None),
            override_lora_cfg=lora_cfg,
            trainable=self.trainable,
        )
        self._apply_lora_if_needed()
        self.model.to(self.device)

        self.register_buffer(
            "_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )

        self._extractors: dict[str, RobosuiteProprioExtractor] = {}
        self._resize_index_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._preprocessed_cache_version = "lpb_score_preprocessed_v2"
        prop_mean_token = (
            "none"
            if self.prop_mean is None
            else hashlib.sha1(np.asarray(self.prop_mean, dtype=np.float32).tobytes()).hexdigest()
        )
        prop_std_token = (
            "none"
            if self.prop_std is None
            else hashlib.sha1(np.asarray(self.prop_std, dtype=np.float32).tobytes()).hexdigest()
        )
        self._preprocessed_global_token = "|".join(
            [
                self._preprocessed_cache_version,
                str(self.image_size),
                ",".join(self.camera_names),
                prop_mean_token,
                prop_std_token,
            ]
        )
        self._preprocessed_task_tokens = {
            str(task_name): hashlib.sha1(
                json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            for task_name, metadata in self.task_metadata_map.items()
        }
        self._set_trainable(self.trainable)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _validate_checkpoint_payload(payload: dict[str, Any]) -> None:
        required = {"camera_names", "model_cfg", "act_mean", "prop_mean"}
        missing = sorted(required.difference(payload.keys()))
        if missing:
            raise ValueError(
                "Invalid flow policy checkpoint payload. "
                f"Missing keys: {missing}"
            )

    @staticmethod
    def _export_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "camera_names": [str(name) for name in payload["camera_names"]],
            "model_cfg": copy.deepcopy(payload["model_cfg"]),
            "task_prompt_map": dict(payload.get("task_prompt_map", {})),
            "task_metadata_map": dict(payload.get("task_metadata_map", {})),
            "prop_mean": None if payload.get("prop_mean") is None else FlowMultitaskEncoder._to_numpy(payload["prop_mean"]),
            "prop_std": None if payload.get("prop_std") is None else FlowMultitaskEncoder._to_numpy(payload["prop_std"]),
            "act_mean": None if payload.get("act_mean") is None else FlowMultitaskEncoder._to_numpy(payload["act_mean"]),
            "act_std": None if payload.get("act_std") is None else FlowMultitaskEncoder._to_numpy(payload["act_std"]),
            "lora_cfg": copy.deepcopy(payload.get("lora_cfg", {})),
        }

    def export_policy_checkpoint_payload(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.policy_checkpoint_payload)
        payload["lora_cfg"] = copy.deepcopy(self.lora_cfg)
        return payload

    @staticmethod
    def _resolve_lora_cfg(
        *,
        checkpoint_lora_cfg: dict[str, Any] | None,
        override_lora_cfg: dict[str, Any] | None,
        trainable: bool,
    ) -> dict[str, Any]:
        cfg = copy.deepcopy(dict(checkpoint_lora_cfg or {}))
        if override_lora_cfg is not None:
            cfg.update(copy.deepcopy(dict(override_lora_cfg)))
        enabled = bool(cfg.get("enabled", False)) or bool(trainable)
        rank = int(cfg.get("rank", 8))
        alpha = float(cfg.get("alpha", float(rank)))
        dropout = float(cfg.get("dropout", 0.0))
        if enabled and rank <= 0:
            raise ValueError(f"LoRA rank must be positive when enabled, got {rank}")
        if dropout < 0.0:
            raise ValueError(f"LoRA dropout must be non-negative, got {dropout}")
        return {
            "enabled": bool(enabled),
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
        }

    def _apply_lora_if_needed(self) -> None:
        if not bool(self.lora_cfg.get("enabled", False)):
            return
        target_modules = [
            self.model.image_encoder,
            self.model.proprio_tokenizer,
            self.model.language_encoder,
            self.model.language_guided_modulation,
            self.model.fusion,
            self.model.condition_aggregator,
        ]
        num_replaced = 0
        for module in target_modules:
            num_replaced += self._replace_linear_layers_with_lora(module)
        if num_replaced <= 0:
            raise RuntimeError("LoRA was enabled, but no Linear layers were replaced in the encoder path.")

    def _replace_linear_layers_with_lora(self, module: nn.Module) -> int:
        replaced = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        child,
                        rank=int(self.lora_cfg["rank"]),
                        alpha=float(self.lora_cfg["alpha"]),
                        dropout=float(self.lora_cfg["dropout"]),
                    ),
                )
                replaced += 1
                continue
            replaced += self._replace_linear_layers_with_lora(child)
        return replaced

    def _set_trainable(self, trainable: bool) -> None:
        self.trainable = bool(trainable)
        for name, param in self.model.named_parameters():
            if name.startswith("flow_head."):
                param.requires_grad = False
            elif ".lora_a." in name or ".lora_b." in name:
                param.requires_grad = self.trainable
            else:
                param.requires_grad = False
        if self.trainable:
            self.model.train()
            self.model.flow_head.eval()
        else:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.trainable:
            self.model.train(mode)
            self.model.flow_head.eval()
        else:
            self.model.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

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
        return x.astype(np.float32, copy=False)

    def _prepare_images(self, images_hwc: np.ndarray) -> np.ndarray:
        if images_hwc.ndim != 5:
            raise ValueError(f"Expected image tensor (T,V,H,W,C), got {images_hwc.shape}")
        timesteps, num_cameras = int(images_hwc.shape[0]), int(images_hwc.shape[1])
        if num_cameras != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} cameras from checkpoint, got {num_cameras}")
        prepared_views: list[np.ndarray] = []
        for view_idx in range(num_cameras):
            resized = self._resize_center_crop_video(images_hwc[:, view_idx])
            prepared_views.append(np.transpose(resized.astype(np.float32) / 255.0, (0, 3, 1, 2)))
        prepared = np.stack(prepared_views, axis=1)
        return np.ascontiguousarray(prepared.reshape(timesteps, num_cameras, 3, self.image_size, self.image_size))

    def _resize_center_crop_video(self, images: np.ndarray) -> np.ndarray:
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

    def _compute_proprio(self, task_name: str, states: np.ndarray) -> np.ndarray:
        extractor = self._get_extractor(task_name)
        state_array = np.asarray(states, dtype=np.float32)
        if state_array.ndim != 2:
            raise ValueError(f"Expected states shape (T,S), got {state_array.shape}")
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
            [
                qpos[:, extractor.qpos_indices],
                qvel[:, extractor.qvel_indices],
            ],
            axis=1,
        ).astype(np.float32)
        return self._normalize_proprio(proprio)

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
        if length <= 0:
            raise ValueError(f"Demo {demo_key} from {file_path} is empty.")

        stacked_images = np.stack([images[:length] for images in image_views], axis=1)
        return PreparedDemo(
            states=states[:length],
            actions=actions[:length],
            images_hwc=stacked_images,
            task_name=task_name,
            file_path=file_path,
            demo_key=demo_key,
        )

    def materialize_prepared_demo(self, prepared: PreparedDemo) -> PreparedEncoderDemo:
        proprio = self._compute_proprio(task_name=prepared.task_name, states=prepared.states)
        images_chw = self._prepare_images(prepared.images_hwc)
        length = min(int(proprio.shape[0]), int(images_chw.shape[0]), int(prepared.actions.shape[0]))
        if length <= 0:
            raise ValueError(f"Prepared demo {prepared.file_path}:{prepared.demo_key} is empty after alignment.")
        return PreparedEncoderDemo(
            images_chw=np.asarray(images_chw[:length], dtype=np.float32),
            proprio=np.asarray(proprio[:length], dtype=np.float32),
            actions=np.asarray(prepared.actions[:length], dtype=np.float32),
            task_name=prepared.task_name,
            file_path=prepared.file_path,
            demo_key=prepared.demo_key,
        )

    def preprocessed_cache_key(self, task_name: str, file_path: str, demo_key: str) -> str:
        ckpt_task_name = resolve_checkpoint_task_name(task_name)
        key_str = "|".join(
            [
                self._preprocessed_global_token,
                str(task_name),
                self._preprocessed_task_tokens.get(ckpt_task_name, "missing_task_metadata"),
                str(Path(file_path).resolve()),
                str(demo_key),
                str(os.path.getmtime(file_path)),
            ]
        )
        return hashlib.sha1(key_str.encode("utf-8")).hexdigest()

    def preprocessed_cache_path(self, cache_root: str, task_name: str, file_path: str, demo_key: str) -> str:
        cache_dir = os.path.join(
            to_absolute_path(str(cache_root)),
            normalize_path_component(task_name),
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.preprocessed_cache_key(task_name, file_path, demo_key)}.npz")

    def load_preprocessed_demo(
        self,
        cache_root: str,
        task_name: str,
        file_path: str,
        demo_key: str,
        *,
        normalize_images: bool = True,
    ) -> PreparedEncoderDemo:
        cache_path = self.preprocessed_cache_path(
            cache_root=cache_root,
            task_name=task_name,
            file_path=file_path,
            demo_key=demo_key,
        )
        with np.load(cache_path) as cached:
            images = np.asarray(cached["images_chw"])
            if normalize_images:
                if images.dtype == np.uint8:
                    images = images.astype(np.float32) / 255.0
                else:
                    images = np.asarray(images, dtype=np.float32)
            return PreparedEncoderDemo(
                images_chw=images,
                proprio=np.asarray(cached["proprio"], dtype=np.float32),
                actions=np.asarray(cached["actions"], dtype=np.float32),
                task_name=str(task_name),
                file_path=str(file_path),
                demo_key=str(demo_key),
            )

    def save_preprocessed_demo(
        self,
        cache_root: str,
        prepared: PreparedEncoderDemo,
    ) -> str:
        cache_path = self.preprocessed_cache_path(
            cache_root=cache_root,
            task_name=prepared.task_name,
            file_path=prepared.file_path,
            demo_key=prepared.demo_key,
        )
        tmp_path = f"{cache_path}.tmp"
        images = np.asarray(prepared.images_chw)
        if images.dtype != np.uint8:
            images = np.clip(np.rint(images * 255.0), 0.0, 255.0).astype(np.uint8)
        with open(tmp_path, "wb") as file_handle:
            np.savez(
                file_handle,
                images_chw=images,
                proprio=np.asarray(prepared.proprio, dtype=np.float32),
                actions=np.asarray(prepared.actions, dtype=np.float32),
            )
        os.replace(tmp_path, cache_path)
        return cache_path

    def materialize_demo_inputs(
        self,
        task_name: str,
        file_path: str,
        demo_key: str,
        *,
        cache_root: str | None = None,
        use_cache: bool = False,
        refresh_cache: bool = False,
    ) -> PreparedEncoderDemo:
        use_preprocessed_cache = bool(use_cache) and cache_root is not None and str(cache_root) != ""
        if use_preprocessed_cache and not bool(refresh_cache):
            cache_path = self.preprocessed_cache_path(
                cache_root=cache_root,
                task_name=task_name,
                file_path=file_path,
                demo_key=demo_key,
            )
            if os.path.isfile(cache_path):
                return self.load_preprocessed_demo(
                    cache_root=cache_root,
                    task_name=task_name,
                    file_path=file_path,
                    demo_key=demo_key,
                    normalize_images=True,
                )

        materialized = self.materialize_prepared_demo(
            self.load_demo_raw(task_name=task_name, file_path=file_path, demo_key=demo_key)
        )
        if use_preprocessed_cache:
            self.save_preprocessed_demo(cache_root=cache_root, prepared=materialized)
        return materialized

    def encode_context_tensors(
        self,
        *,
        images: torch.Tensor,
        proprio: torch.Tensor,
        task_names: Sequence[str] | str,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images shape (B,V,3,H,W), got {tuple(images.shape)}")
        if proprio.ndim != 2:
            raise ValueError(f"Expected proprio shape (B,P), got {tuple(proprio.shape)}")
        batch_size = int(images.shape[0])
        if int(images.shape[1]) != len(self.camera_names):
            raise ValueError(
                f"Expected {len(self.camera_names)} cameras from checkpoint, got {images.shape[1]}"
            )
        if int(proprio.shape[0]) != batch_size:
            raise ValueError("images and proprio batch sizes must match.")

        if isinstance(task_names, str):
            task_names = [task_names] * batch_size
        if len(task_names) != batch_size:
            raise ValueError(f"Expected {batch_size} task names, got {len(task_names)}")

        target_device = self._image_mean.device
        image_batch = images.to(device=target_device, dtype=torch.float32)
        prop_batch = proprio.to(device=target_device, dtype=torch.float32)
        image_batch = (image_batch - self._image_mean.to(dtype=image_batch.dtype)) / self._image_std.to(
            dtype=image_batch.dtype
        )
        language = [self.resolve_language_instruction(str(task_name)) for task_name in task_names]
        return self.model.encode_context(
            images=image_batch,
            proprio=prop_batch,
            language=language,
        )

    @torch.no_grad()
    def encode_sequence(
        self,
        *,
        images: np.ndarray | torch.Tensor,
        proprio: np.ndarray | torch.Tensor,
        task_name: str,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        image_tensor = torch.as_tensor(images, dtype=torch.float32)
        proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32)
        if image_tensor.ndim != 5 or proprio_tensor.ndim != 2:
            raise ValueError("encode_sequence expects images (T,V,3,H,W) and proprio (T,P).")
        batch = max(1, int(batch_size or self.batch_size))
        outputs: list[torch.Tensor] = []
        for start in range(0, int(image_tensor.shape[0]), batch):
            end = min(start + batch, int(image_tensor.shape[0]))
            outputs.append(
                self.encode_context_tensors(
                    images=image_tensor[start:end],
                    proprio=proprio_tensor[start:end],
                    task_names=[task_name] * int(end - start),
                ).detach().cpu()
            )
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_arrays(
        self,
        task_name: str,
        states: np.ndarray,
        images_hwc: np.ndarray,
    ) -> np.ndarray:
        prepared = PreparedDemo(
            states=np.asarray(states, dtype=np.float32),
            actions=np.zeros((int(states.shape[0]), self.action_dim), dtype=np.float32),
            images_hwc=np.asarray(images_hwc),
            task_name=task_name,
            file_path="",
            demo_key="",
        )
        materialized = self.materialize_prepared_demo(prepared)
        latents = self.encode_sequence(
            images=materialized.images_chw,
            proprio=materialized.proprio,
            task_name=task_name,
        )
        return latents.numpy().astype(np.float32)

    def encode_demo(self, task_name: str, file_path: str, demo_key: str) -> EncodedDemo:
        prepared = self.load_demo_raw(task_name=task_name, file_path=file_path, demo_key=demo_key)
        return self.encode_prepared_demos([prepared])[0]

    @torch.no_grad()
    def encode_prepared_demos(self, prepared_demos: list[PreparedDemo]) -> list[EncodedDemo]:
        if not prepared_demos:
            return []

        encoded_demos: list[EncodedDemo] = []
        for prepared in prepared_demos:
            materialized = self.materialize_prepared_demo(prepared)
            latents = self.encode_sequence(
                images=materialized.images_chw,
                proprio=materialized.proprio,
                task_name=materialized.task_name,
            ).numpy()
            encoded_demos.append(
                EncodedDemo(
                    latents=np.asarray(latents, dtype=np.float32),
                    actions=np.asarray(materialized.actions, dtype=np.float32),
                    task_name=materialized.task_name,
                    file_path=materialized.file_path,
                    demo_key=materialized.demo_key,
                )
            )
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
                str(Path(self.checkpoint_path).resolve()) if self.checkpoint_path is not None else "embedded_payload",
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
            latents=np.asarray(encoded.latents, dtype=np.float32),
            actions=np.asarray(encoded.actions, dtype=np.float32),
            task_name=np.asarray(encoded.task_name),
            file_path=np.asarray(encoded.file_path),
            demo_key=np.asarray(encoded.demo_key),
        )
        return cache_path


class FrozenFlowMultitaskEncoder(FlowMultitaskEncoder):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 128,
        batch_size: int = 96,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            trainable=False,
        )
