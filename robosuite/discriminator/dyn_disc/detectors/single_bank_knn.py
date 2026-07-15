"""Shared frozen-dynamics encoder (``DynEncoder``) + small helpers.

``DynEncoder`` loads the frozen DINOv3 dynamics model and exposes a
``(visual + proprio + action)`` per-frame encoding in the configured feature
space (encoder embeddings or a transformer-layer feature). It is the shared
encoder backbone used by every benchmark adapter in this package
(``DynBenchmarkDiscriminator`` and its subclasses).

The dynamics model that produces the encoder + proprio_encoder is loaded from a
checkpoint via ``robosuite.discriminator.dyn_disc.core.model_loader.load_model``.

This module also defines:
  * :class:`DetectionResult` -- the shared per-frame score/threshold/pred struct.
  * :func:`knn_min_l2_dist`  -- chunked nearest-neighbour L2 helper (kept for
    diagnostics / potential future KNN-style heads).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

from robosuite.discriminator.dyn_disc.core.model_loader import load_model
from robosuite.discriminator.dyn_disc.utils.normalizer import LinearNormalizer


# --------------------------------------------------------------------------- #
# KNN distance helper                                                         #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def knn_min_l2_dist(
    query: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Chunked nearest-neighbor non-squared L2 distance from query to bank.

    Args:
        query: (B, D) tensor.
        bank:  (N, D) tensor.
        chunk_size: number of bank rows processed per cdist call (memory-bound).
    Returns:
        (B,) tensor of min L2 distances (non-squared).
    """
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if bank.shape[0] == 0:
        raise ValueError("bank cannot be empty")

    q = query
    best = torch.full((q.shape[0],), float("inf"), device=q.device, dtype=q.dtype)
    csz = int(chunk_size)
    for start in range(0, bank.shape[0], csz):
        end = min(start + csz, bank.shape[0])
        chunk = bank[start:end].to(device=q.device, dtype=q.dtype, non_blocking=True)
        d = torch.cdist(q, chunk, p=2.0)  # (B, chunk)
        cur = torch.min(d, dim=1).values
        best = torch.minimum(best, cur)
    return best


# --------------------------------------------------------------------------- #
# Result struct                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class DetectionResult:
    step_scores: np.ndarray   # per-frame min L2 distance (positive; bigger = more OOD)
    thresholds: np.ndarray    # broadcasted threshold per frame (constant unless adaptive)
    preds: np.ndarray         # (T,) int: 1 if failure (min_dist >= threshold), else 0


def _resolve_dataset_class_path(path: str) -> str:
    mapping = {
        "robosuite.discriminator.lpb_original.datasets.HDF5DynamicsModelDataset":
            "robosuite.discriminator.dyn_disc.data.hdf5_dynamics_dataset.HDF5DynamicsModelDataset",
        "robosuite.discriminator.lpb_original.datasets.PreprocessedCacheDynamicsModelDataset":
            "robosuite.discriminator.dyn_disc.data.preprocessed_cache_dataset.PreprocessedCacheDynamicsModelDataset",
        "robosuite.discriminator.lpb_original.datasets.hdf5_dynamics_dataset.HDF5DynamicsModelDataset":
            "robosuite.discriminator.dyn_disc.data.hdf5_dynamics_dataset.HDF5DynamicsModelDataset",
        "robosuite.discriminator.lpb_original.datasets.preprocessed_cache_dataset.PreprocessedCacheDynamicsModelDataset":
            "robosuite.discriminator.dyn_disc.data.preprocessed_cache_dataset.PreprocessedCacheDynamicsModelDataset",
    }
    return mapping.get(str(path), str(path))


# --------------------------------------------------------------------------- #
# Encoder wrapper                                                             #
# --------------------------------------------------------------------------- #


class DynEncoder:
    """Loads the frozen dynamics model and exposes (visual+proprio+action) encoding.

    Reads:
      <ckpt_dir>/hydra.yaml       # full training config (used by dyn_model.plan.load_model)
      <ckpt_dir>/normalizer.pth   # saved LinearNormalizer state_dict (image + state stats)

    `model_ckpt` is the actual `.pth` produced by the training loop (e.g. checkpoints/model_50.pth).
    """

    def __init__(
        self,
        model_ckpt: str,
        device: str = "cuda",
        feature_source: str = "encoder",
        transformer_layer: int = -1,
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        feature_source = str(feature_source)
        if feature_source not in {"encoder", "transformer"}:
            raise ValueError(f"feature_source must be 'encoder' or 'transformer', got {feature_source!r}")
        self.feature_source = feature_source
        self.transformer_layer = int(transformer_layer)

        ckpt_path = Path(model_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LPB dynamics ckpt not found: {ckpt_path}")

        # ------------------------------------------------------------------ #
        # Locate saved Hydra config + normalizer across common layouts.
        #
        # New (this repo's `lpb_original/train.py`):
        #   <run_dir>/{hydra.yaml, normalizer.pth, checkpoints/model_*.pth}
        #
        # Legacy (Hydra default output dir):
        #   <run_dir>/.hydra/{hydra.yaml, config.yaml, overrides.yaml}
        #   <run_dir>/checkpoints/model_*.pth
        #
        # Some older runs store `model_*.pth` directly under <run_dir>.
        # ------------------------------------------------------------------ #
        run_dir_candidates: List[Path] = [
            ckpt_path.parent,
            ckpt_path.parent.parent,
        ]
        # If there's a single preprocessing subdir that contains `.hydra/hydra.yaml`,
        # prefer it as a config source.
        try:
            for child in ckpt_path.parent.iterdir():
                if child.is_dir() and (child / ".hydra" / "hydra.yaml").exists():
                    run_dir_candidates.insert(0, child)
        except Exception:
            pass

        cfg_path: Optional[Path] = None
        cfg_candidates: List[Path] = []
        for rd in run_dir_candidates:
            cfg_candidates.extend([
                rd / "hydra.yaml",
                rd / ".hydra" / "config.yaml",
                rd / ".hydra" / "hydra.yaml",
            ])
        for p in cfg_candidates:
            if p.exists():
                cfg_path = p
                break
        if cfg_path is None:
            tried = "\n  - " + "\n  - ".join(str(p) for p in cfg_candidates[:12])
            raise FileNotFoundError(
                "Could not locate a Hydra config for the LPB dynamics checkpoint.\n"
                f"Checkpoint: {ckpt_path}\n"
                f"Tried (first few):{tried}"
            )

        self.cfg = OmegaConf.load(cfg_path)

        # Normalizer: prefer a saved `normalizer.pth` next to the run dir; if missing,
        # rebuild it from the dataset specified in the saved config.
        norm_path: Optional[Path] = None
        norm_candidates: List[Path] = []
        for rd in run_dir_candidates:
            norm_candidates.extend([
                rd / "normalizer.pth",
                rd / ".hydra" / "normalizer.pth",
            ])
        for p in norm_candidates:
            if p.exists():
                norm_path = p
                break
        self.normalizer_checkpoint = (
            None if norm_path is None else str(norm_path.expanduser().resolve())
        )

        self.model = load_model(ckpt_path, self.cfg, device=self.device)
        self.model.eval()

        normalizer = LinearNormalizer()
        if norm_path is not None:
            normalizer.load_state_dict(torch.load(norm_path, map_location=self.device))
        else:
            # Fallback: rebuild the normalizer from the training dataset config.
            # This is deterministic for the same dataset and matches the original training code.
            try:
                from hydra.utils import get_class, to_absolute_path

                env = self.cfg.env
                dataset_class_path = str(getattr(env, "dataset_class"))
                DatasetCls = get_class(_resolve_dataset_class_path(dataset_class_path))
                train_data_path = getattr(env, "train_data_path")
                if isinstance(train_data_path, ListConfig):
                    train_data_path = OmegaConf.to_container(train_data_path, resolve=True)
                else:
                    train_data_path = str(train_data_path)

                    # Mirror the dataset kwargs used in original LPB training.
                kwargs: Dict[str, Any] = dict(
                    zarr_path=train_data_path,
                    num_hist=int(getattr(self.cfg, "num_hist")),
                    num_pred=int(getattr(self.cfg, "num_pred")),
                    frameskip=int(getattr(self.cfg, "frameskip")),
                    view_names=list(getattr(env, "view_names")),
                    abs_action=bool(getattr(self.cfg, "abs_action")),
                    use_crop=bool(getattr(self.cfg, "use_crop", True)),
                    train=True,
                    original_img_size=int(getattr(env, "original_img_size")),
                    cropped_img_size=int(getattr(env, "cropped_img_size")),
                    action_dim=int(getattr(env, "action_dim")),
                )
                # Optional knobs present in some dataset classes / configs.
                if hasattr(env, "use_cache"):
                    kwargs["use_cache"] = bool(getattr(env, "use_cache"))
                if hasattr(env, "cache_dir") and getattr(env, "cache_dir") is not None:
                    kwargs["cache_dir"] = to_absolute_path(str(getattr(env, "cache_dir")))
                if hasattr(env, "shape_obs") and getattr(env, "shape_obs") is not None:
                    kwargs["shape_obs"] = OmegaConf.to_container(getattr(env, "shape_obs"), resolve=True)
                if hasattr(env, "cache_root") and getattr(env, "cache_root") is not None:
                    kwargs["cache_root"] = to_absolute_path(str(getattr(env, "cache_root")))
                if hasattr(env, "tasks") and getattr(env, "tasks") is not None:
                    kwargs["tasks"] = list(getattr(env, "tasks"))
                if hasattr(env, "train_sources") and getattr(env, "train_sources") is not None:
                    kwargs["train_sources"] = list(getattr(env, "train_sources"))
                if hasattr(env, "camera_to_view") and getattr(env, "camera_to_view") is not None:
                    kwargs["camera_to_view"] = OmegaConf.to_container(
                        getattr(env, "camera_to_view"), resolve=True
                    )
                if hasattr(env, "max_cached_episodes") and getattr(env, "max_cached_episodes") is not None:
                    kwargs["max_cached_episodes"] = int(getattr(env, "max_cached_episodes"))
                if hasattr(env, "load_all_into_ram"):
                    kwargs["load_all_into_ram"] = bool(getattr(env, "load_all_into_ram"))
                if hasattr(env, "metadata_cache_root") and getattr(env, "metadata_cache_root") is not None:
                    kwargs["metadata_cache_root"] = to_absolute_path(str(getattr(env, "metadata_cache_root")))
                if hasattr(env, "num_expert"):
                    kwargs["num_expert"] = int(getattr(env, "num_expert"))
                if hasattr(env, "num_success"):
                    kwargs["num_success"] = int(getattr(env, "num_success"))
                if hasattr(env, "num_fail"):
                    kwargs["num_fail"] = int(getattr(env, "num_fail"))
                if getattr(self.cfg, "proprio_indices", None):
                    kwargs["proprio_indices"] = list(getattr(self.cfg, "proprio_indices"))
                if getattr(self.cfg, "proprio_map", None):
                    kwargs["proprio_map"] = OmegaConf.to_container(
                        getattr(self.cfg, "proprio_map"), resolve=True
                    )
                if getattr(self.cfg, "max_trajectories", None):
                    kwargs["max_trajectories"] = int(getattr(self.cfg, "max_trajectories"))

                ds = DatasetCls(**kwargs)
                rebuilt = ds.get_normalizer()
                normalizer.load_state_dict(rebuilt.state_dict())
                print(
                    "[dyn_disc] WARNING: normalizer.pth not found; rebuilt normalizer from dataset config. "
                    "For exact reproducibility, re-export `normalizer.pth` next to the checkpoint."
                )
            except Exception as e:
                raise FileNotFoundError(
                    "normalizer.pth not found near the checkpoint, and rebuilding the normalizer from the "
                    f"saved Hydra config failed.\nCheckpoint: {ckpt_path}\nConfig: {cfg_path}\n"
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
        self.normalizer = normalizer.to(self.device)

        self.view_names: List[str] = list(self.cfg.env.view_names)
        self.source_view_names: List[str] = list(getattr(self.cfg, "source_view_names", self.view_names))
        self.original_img_size: int = int(self.cfg.env.original_img_size)
        self.cropped_img_size: int = int(self.cfg.env.cropped_img_size)
        self.use_crop: bool = bool(getattr(self.cfg, "use_crop", True))
        self.proprio_emb_dim: int = int(self.cfg.env.proprio_emb_dim)
        self.action_emb_dim: int = int(self.cfg.env.action_emb_dim)
        self.num_patches: int = int(getattr(self.model.encoder, "num_patches", 1))
        self.visual_emb_dim_total: int = int(self.model.encoder.emb_dim) * len(self.source_view_names) * self.num_patches
        self.frameskip: int = int(getattr(self.cfg, "frameskip", 1))
        self.action_dim_per_step: int = int(
            getattr(self.cfg, "action_dim_per_step", getattr(self.cfg.env, "action_dim", 7))
        )
        self.action_input_dim: int = int(
            getattr(self.model.action_encoder, "in_chans", self.action_dim_per_step * self.frameskip)
        )

        if self.use_crop:
            from robosuite.discriminator.dyn_disc.data.img_transforms import get_eval_crop_transform_resnet
            self.img_transform = get_eval_crop_transform_resnet(
                original_img_size=self.original_img_size,
                cropped_img_size=self.cropped_img_size,
            )
        else:
            self.img_transform = lambda x: x

    def prepare_actions(self, actions: np.ndarray, t_len: int) -> np.ndarray:
        """Build flattened action windows matching the train-time action encoder input."""
        act = np.asarray(actions[:t_len], dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(-1, 1)
        if act.shape[1] < self.action_dim_per_step:
            pad = np.zeros((act.shape[0], self.action_dim_per_step - act.shape[1]), dtype=np.float32)
            act = np.concatenate([act, pad], axis=1)
        elif act.shape[1] > self.action_dim_per_step:
            act = act[:, : self.action_dim_per_step]

        out = np.zeros((int(t_len), self.frameskip, self.action_dim_per_step), dtype=np.float32)
        for t in range(int(t_len)):
            end = min(int(t_len), t + self.frameskip)
            chunk = act[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < self.frameskip:
                pad_value = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim_per_step,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad_value

        flat = out.reshape(int(t_len), -1)
        if flat.shape[1] < self.action_input_dim:
            pad = np.zeros((flat.shape[0], self.action_input_dim - flat.shape[1]), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=1)
        elif flat.shape[1] > self.action_input_dim:
            flat = flat[:, : self.action_input_dim]
        return flat.astype(np.float32, copy=False)

    def _normalize_flat_actions(self, actions: torch.Tensor, batch_size: int) -> torch.Tensor:
        action_in = actions.to(self.device, dtype=torch.float32, non_blocking=True)
        action_in = action_in.view(batch_size, self.frameskip, self.action_dim_per_step)
        action_in = self.normalizer["act"].normalize(action_in)
        return action_in.reshape(batch_size, 1, self.action_input_dim)

    def _prepare_action_chunks_tensor(self, actions: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Return flattened, train-aligned action windows without leaving torch."""
        action = actions.to(self.device, dtype=torch.float32, non_blocking=True)
        if action.ndim == 2 and action.shape[-1] == self.action_input_dim:
            return action.contiguous()
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3 or action.shape[0] != batch_size:
            raise ValueError(
                "actions must be (B, H, D_a) or (B, action_input_dim); "
                f"got {tuple(action.shape)}"
            )
        if action.shape[1] == 0:
            raise ValueError("action chunk horizon cannot be zero")

        if action.shape[-1] < self.action_dim_per_step:
            pad = torch.zeros(
                (*action.shape[:-1], self.action_dim_per_step - action.shape[-1]),
                device=action.device,
                dtype=action.dtype,
            )
            action = torch.cat([action, pad], dim=-1)
        elif action.shape[-1] > self.action_dim_per_step:
            action = action[..., : self.action_dim_per_step]

        if action.shape[1] < self.frameskip:
            tail = action[:, -1:, :].expand(-1, self.frameskip - action.shape[1], -1)
            action = torch.cat([action, tail], dim=1)
        else:
            action = action[:, : self.frameskip, :]

        flat = action.reshape(batch_size, -1)
        if flat.shape[-1] < self.action_input_dim:
            pad = torch.zeros(
                (batch_size, self.action_input_dim - flat.shape[-1]),
                device=flat.device,
                dtype=flat.dtype,
            )
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[-1] > self.action_input_dim:
            flat = flat[:, : self.action_input_dim]
        return flat.contiguous()

    def _encode_observation_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Normalize and encode the action-free observation once."""
        if not images_per_view:
            raise ValueError("images_per_view cannot be empty")
        B = next(iter(images_per_view.values())).shape[0]
        missing = [v for v in self.view_names if v not in images_per_view]
        if missing:
            raise KeyError(f"Missing dynamics encoder views: {missing}")

        visual_in: Dict[str, torch.Tensor] = {}
        for view in self.view_names:
            x = images_per_view[view].to(self.device, dtype=torch.float32, non_blocking=True)
            if x.ndim != 4 or x.shape[0] != B or x.shape[1] != 3:
                raise ValueError(
                    f"images_per_view[{view!r}] must be (B, 3, H, W); got {tuple(x.shape)}"
                )
            if not bool(getattr(self.model.encoder, "normalizes_images", False)):
                x = self.normalizer[view].normalize(x)
                x = self.img_transform(
                    x.view(-1, 3, self.original_img_size, self.original_img_size)
                )
            visual_in[view] = x.view(B, 1, 3, x.shape[-2], x.shape[-1])

        proprio_in = self.normalizer["state"].normalize(
            proprio.to(self.device, dtype=torch.float32, non_blocking=True)
        )
        if proprio_in.ndim == 2:
            proprio_in = proprio_in.unsqueeze(1)
        if proprio_in.ndim != 3 or proprio_in.shape[0] != B:
            raise ValueError(f"proprio must be (B, D); got {tuple(proprio.shape)}")
        return self.model.encode_obs({"visual": visual_in, "proprio": proprio_in})

    @staticmethod
    def _flatten_state_encoding(enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten visual and proprio embeddings into an action-free state latent."""
        visual = enc["visual"]
        proprio = enc["proprio"]
        if visual.ndim >= 3 and visual.shape[1] == 1:
            visual = visual.squeeze(1)
        if proprio.ndim >= 3 and proprio.shape[1] == 1:
            proprio = proprio.squeeze(1)
        return torch.cat(
            [visual.reshape(visual.shape[0], -1), proprio.reshape(proprio.shape[0], -1)],
            dim=-1,
        )

    @torch.no_grad()
    def encode_state_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Encode visual + proprio state without injecting an action."""
        enc = self._encode_observation_batch(images_per_view, proprio)
        return self._flatten_state_encoding(enc)

    @torch.no_grad()
    def encode_chunk_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a state fused with the complete train-aligned action window."""
        enc = self._encode_observation_batch(images_per_view, proprio)
        return self._encode_chunk_from_encoding(enc, action_chunk)

    def _encode_chunk_from_encoding(
        self,
        enc: Dict[str, torch.Tensor],
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse a precomputed observation encoding with an action window."""
        B = int(enc["visual"].shape[0])
        flat_actions = self._prepare_action_chunks_tensor(action_chunk, B)
        action_in = self._normalize_flat_actions(flat_actions, B)
        action_emb = self.model.encode_act(action_in)

        if self.feature_source == "transformer":
            visual_emb = enc["visual"]
            proprio_emb = enc["proprio"]
            if visual_emb.ndim == 4:
                num_patches = visual_emb.shape[2]
                proprio_emb = proprio_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
                action_emb = action_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
            z = torch.cat([visual_emb, proprio_emb, action_emb], dim=-1)
            if z.ndim == 4:
                z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3])
            feat = self.model.predictor.extract_transformer_features(
                z, layer_index=self.transformer_layer
            )
            return feat.reshape(feat.shape[0], -1)

        state = self._flatten_state_encoding(enc)
        action_emb = action_emb.reshape(action_emb.shape[0], -1)
        return torch.cat([state, action_emb], dim=-1)

    @torch.no_grad()
    def encode_state_and_chunk_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state and chunk features with one visual/proprio forward."""
        enc = self._encode_observation_batch(images_per_view, proprio)
        state = self._flatten_state_encoding(enc)
        chunk = self._encode_chunk_from_encoding(enc, action_chunk)
        return state, chunk

    @torch.no_grad()
    def encode_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of timesteps into the configured KNN feature space.

        Args:
            images_per_view[view]: (B, 3, H, W) float tensor in [0, 1]; H=W=original_img_size.
            proprio: (B, proprio_dim) float tensor (same layout as the train-time concat).
            actions:
              - feature_source="encoder": required flattened action windows shaped (B, action_dim_per_step * frameskip)
                (typically produced by `prepare_actions`).
              - feature_source="transformer": required per-step actions shaped (B, action_dim_per_step).
        """
        if actions is None:
            raise ValueError("actions are required for chunk encoding")
        return self.encode_chunk_batch(images_per_view, proprio, actions)
