from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from .dataset import LatentTrajectory
from .model import LatentDynamicsModel, build_latent_dynamics_predictor


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            try:
                cur = cur[key]
            except Exception:
                try:
                    cur = getattr(cur, key)
                except Exception:
                    return default
    return default if cur is None else cur


def _resolve_device(device: str) -> torch.device:
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _torch_load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class TransitionFeatureSequence:
    features: np.ndarray
    support_penalty_raw: np.ndarray
    task_name: str
    task_index: int
    data_type: str
    data_type_index: int
    split: str
    file_path: str
    demo_key: str


class FrozenTransitionRepresentation:
    """
    Frozen transition encoder built from the pretrained latent dynamics checkpoint.

    The PU detector consumes a concatenation of:
    1. policy encoder latent z_t
    2. world-model transition feature extract_feature(z_t, a_{t:t+H-1})
    """

    def __init__(
        self,
        model: LatentDynamicsModel,
        *,
        action_horizon: int,
        action_dim: int,
        latent_dim: int,
        cfg_model: dict[str, Any],
        source_ckpt: str | None,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        self.model = model
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.cfg_model = dict(cfg_model)
        self.source_ckpt = None if source_ckpt in {None, "", "None", "null"} else str(source_ckpt)
        self.device = _resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.feature_dim = self._infer_feature_dim()

    @classmethod
    def from_backbone_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> "FrozenTransitionRepresentation":
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        return cls.from_backbone_payload(
            payload=payload,
            device=device,
            batch_size=batch_size,
            source_ckpt=checkpoint_path,
        )

    @classmethod
    def from_backbone_payload(
        cls,
        *,
        payload: dict[str, Any],
        device: str = "cuda",
        batch_size: int = 256,
        source_ckpt: str | None = None,
    ) -> "FrozenTransitionRepresentation":
        if "model" not in payload:
            raise ValueError("Backbone payload is missing `model`.")

        cfg = payload.get("cfg", None)
        latent_dim = int(payload.get("latent_dim"))
        action_dim = int(payload.get("action_dim"))
        horizon = int(payload.get("horizon", 1))
        cfg_model = dict(payload.get("cfg_model", _cfg_get(cfg, "model", {})))
        predictor = build_latent_dynamics_predictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            cfg_model=cfg_model,
            transition_horizon=horizon,
        )
        model = LatentDynamicsModel(predictor=predictor)
        model.load_state_dict(payload["model"], strict=True)
        return cls(
            model=model,
            action_horizon=horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            cfg_model=dict(cfg_model),
            source_ckpt=source_ckpt if source_ckpt is not None else payload.get("source_ckpt", None),
            device=device,
            batch_size=batch_size,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "latent_dim": int(self.latent_dim),
            "action_dim": int(self.action_dim),
            "horizon": int(self.action_horizon),
            "cfg_model": dict(self.cfg_model),
            "source_ckpt": self.source_ckpt,
        }

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            current = torch.zeros((1, self.latent_dim), dtype=torch.float32, device=self.device)
            action = torch.zeros(
                (1, self.action_horizon, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            dyn_feature = self.model.extract_feature(current_latent=current, action_sequence=action)
        return int(self.latent_dim + dyn_feature.shape[-1])

    def _prepare_latents(self, latents: np.ndarray, t_len: int) -> np.ndarray:
        latents = np.asarray(latents[:t_len], dtype=np.float32)
        if latents.shape[1] < self.latent_dim:
            pad = np.zeros((latents.shape[0], self.latent_dim - latents.shape[1]), dtype=np.float32)
            latents = np.concatenate([latents, pad], axis=1)
        elif latents.shape[1] > self.latent_dim:
            latents = latents[:, : self.latent_dim]
        return latents

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

        horizon = self.action_horizon
        out = np.zeros((t_len, horizon, self.action_dim), dtype=np.float32)
        for t in range(t_len):
            end = min(t_len, t + horizon)
            chunk = actions[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < horizon:
                pad_value = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad_value
        return out

    @torch.no_grad()
    def encode_trajectory(self, traj: LatentTrajectory) -> TransitionFeatureSequence:
        t_len = min(int(traj.latents.shape[0]), int(traj.actions.shape[0]))
        if t_len <= 0:
            raise ValueError("Trajectory has zero valid timesteps.")
        valid_len = t_len - self.action_horizon
        if valid_len <= 0:
            raise ValueError(
                f"Trajectory length {t_len} is too short for action_horizon={self.action_horizon}."
            )

        latents = self._prepare_latents(traj.latents, t_len=t_len)
        action_chunks = self._prepare_actions(traj.actions, t_len=t_len)
        latents_t = torch.from_numpy(latents)
        action_t = torch.from_numpy(action_chunks)
        target_t = latents_t[self.action_horizon : self.action_horizon + valid_len]

        features: list[torch.Tensor] = []
        penalties: list[torch.Tensor] = []
        for start in range(0, valid_len, self.batch_size):
            end = min(start + self.batch_size, valid_len)
            current_b = latents_t[start:end].to(self.device)
            action_b = action_t[start:end].to(self.device)
            target_b = target_t[start:end].to(self.device)

            dyn_feature = self.model.extract_feature(
                current_latent=current_b,
                action_sequence=action_b,
            )
            pred = self.model(
                current_latent=current_b,
                action_sequence=action_b,
            )
            error = pred["pred_latent"] - target_b
            if "pred_logvar" in pred:
                inv_var = torch.exp(-pred["pred_logvar"])
                penalty = 0.5 * (error.pow(2) * inv_var + pred["pred_logvar"]).mean(dim=-1)
            else:
                penalty = error.pow(2).mean(dim=-1)

            feature = torch.cat([current_b, dyn_feature], dim=-1)
            features.append(feature.detach().cpu())
            penalties.append(penalty.detach().cpu())

        return TransitionFeatureSequence(
            features=torch.cat(features, dim=0).numpy().astype(np.float32),
            support_penalty_raw=torch.cat(penalties, dim=0).numpy().astype(np.float32),
            task_name=str(traj.task_name),
            task_index=int(traj.task_index),
            data_type=str(traj.data_type),
            data_type_index=int(traj.data_type_index),
            split=str(traj.split),
            file_path=str(traj.file_path),
            demo_key=str(traj.demo_key),
        )

    @torch.no_grad()
    def encode_trajectories(
        self,
        trajectories: list[LatentTrajectory],
    ) -> list[TransitionFeatureSequence]:
        return [self.encode_trajectory(traj) for traj in trajectories]
