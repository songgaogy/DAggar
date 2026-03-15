from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.float.float_data import PolicyTrajectory
from robosuite.discriminator.lpb.knn_discriminator import (
    AdaptiveKNNDiscriminator,
    DetectionResult,
    _cfg_get,
    _resolve_device,
    _torch_load_checkpoint,
    knn_min_sqdist,
)
from robosuite.discriminator.contrastive.model import (
    ContrastiveContextEncoder,
    Encoder,
    PureContrastiveModel,
)


class PureContrastiveFeatureExtractor:
    """
    Build (o, a) features from the pure contrastive checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 256,
        action_horizon: int = -1,
        proprio_indices: Optional[Sequence[int]] = None,
        normalize_feature: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.proprio_indices = (
            None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        )
        self.normalize_feature = bool(normalize_feature)
        self.model, self.action_horizon, self.action_dim, self.proprio_dim = self._load_model(
            checkpoint_path=checkpoint_path,
            override_action_horizon=action_horizon,
        )

    def _load_model(
        self,
        checkpoint_path: str,
        override_action_horizon: int,
    ) -> tuple[PureContrastiveModel, int, int, int]:
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing key `model`: {checkpoint_path}")

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim", 512))
        action_dim = int(payload.get("action_dim", 7))
        proprio_dim = int(payload.get("proprio_dim", 0))
        horizon_ckpt = int(payload.get("horizon", 1))
        action_horizon = horizon_ckpt if int(override_action_horizon) <= 0 else int(override_action_horizon)

        d_model = int(_cfg_get(cfg, "model.d_model", 512))
        num_layers = int(_cfg_get(cfg, "model.num_layers", 6))
        num_heads = int(_cfg_get(cfg, "model.num_heads", 8))
        dropout = float(_cfg_get(cfg, "model.dropout", 0.1))
        max_action_horizon = int(_cfg_get(cfg, "model.max_action_horizon", action_horizon))
        fusion_hidden_dim = int(_cfg_get(cfg, "model.fusion_hidden_dim", d_model))
        projection_dim = int(_cfg_get(cfg, "model.projection_dim", 128))
        normalize_input = bool(_cfg_get(cfg, "encoder.normalize_input", True))

        encoder = Encoder(
            checkpoint_path=None,
            pretrained=False,
            freeze=True,
            normalize_input=normalize_input,
        )
        context_encoder = ContrastiveContextEncoder(
            latent_dim=latent_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=num_heads,
            dropout=dropout,
            max_action_horizon=max(max_action_horizon, action_horizon),
            fusion_hidden_dim=fusion_hidden_dim,
        )
        model = PureContrastiveModel(
            encoder=encoder,
            context_encoder=context_encoder,
            projection_dim=projection_dim,
        )
        incompatible = model.load_state_dict(payload["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected or missing:
            print(
                "Pure contrastive checkpoint load warning:",
                {
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
            )
        model.to(self.device)
        model.eval()
        return model, action_horizon, action_dim, proprio_dim

    def _prepare_images(self, images: np.ndarray) -> torch.Tensor:
        img = images.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        chw = np.transpose(img, (0, 3, 1, 2))
        return torch.from_numpy(chw)

    def _prepare_actions(self, actions: Optional[np.ndarray], t_len: int) -> np.ndarray:
        if actions is None:
            actions = np.zeros((t_len, self.action_dim), dtype=np.float32)
        else:
            actions = np.asarray(actions[:t_len], dtype=np.float32)
        if actions.shape[1] < self.action_dim:
            pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]

        h = self.action_horizon
        out = np.zeros((t_len, h, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + h)
            chunk = actions[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < h:
                pad = (
                    chunk[-1]
                    if chunk.shape[0] > 0
                    else np.zeros((self.action_dim,), dtype=np.float32)
                )
                out[t, chunk.shape[0] :] = pad
        return out

    def _prepare_proprio(self, states: np.ndarray, t_len: int) -> np.ndarray:
        prop = np.asarray(states[:t_len], dtype=np.float32)
        if self.proprio_indices is not None and self.proprio_indices.size > 0:
            prop = prop[:, self.proprio_indices]
        if prop.shape[1] < self.proprio_dim:
            pad = np.zeros((prop.shape[0], self.proprio_dim - prop.shape[1]), dtype=np.float32)
            prop = np.concatenate([prop, pad], axis=1)
        elif prop.shape[1] > self.proprio_dim:
            prop = prop[:, : self.proprio_dim]
        return prop

    @torch.no_grad()
    def encode_trajectory(self, traj: PolicyTrajectory) -> torch.Tensor:
        t_len = int(traj.images.shape[0])
        if traj.actions is not None:
            t_len = min(t_len, int(traj.actions.shape[0]))
        t_len = min(t_len, int(traj.states.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps")

        imgs = self._prepare_images(traj.images[:t_len])
        prop = self._prepare_proprio(traj.states, t_len=t_len)
        act_chunks = self._prepare_actions(traj.actions, t_len=t_len)

        prop_t = torch.from_numpy(prop)
        act_t = torch.from_numpy(act_chunks)

        feats = []
        for start in range(0, t_len, self.batch_size):
            end = min(start + self.batch_size, t_len)
            img_b = imgs[start:end].to(self.device)
            prop_b = prop_t[start:end].to(self.device)
            act_b = act_t[start:end].to(self.device)

            f = self.model.encode_shared_feature(
                current_image=img_b,
                current_proprio=prop_b,
                action_sequence=act_b,
                normalize=self.normalize_feature,
            )
            feats.append(f.detach().cpu())
        return torch.cat(feats, dim=0)


__all__ = [
    "AdaptiveKNNDiscriminator",
    "DetectionResult",
    "PureContrastiveFeatureExtractor",
    "knn_min_sqdist",
]
