from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import Normalize

try:
    from robosuite.policy.flow import FlowPolicy, sinusoidal_time_embedding
    from robosuite.policy.utils.env_util import PandaLiftProprioExtractor
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name not in {"mujoco", "robosuite"}:
        raise
    # Fallback import path for environments without robosuite top-level package deps.
    from policy.flow import FlowPolicy, sinusoidal_time_embedding
    from policy.utils.env_util import PandaLiftProprioExtractor

from .float_data import PolicyTrajectory


@dataclass
class FlowBackboneBuildResult:
    model: FlowPolicy
    history_len: int
    token_dim: int
    loaded_params: int
    prop_mean: Optional[np.ndarray]
    prop_std: Optional[np.ndarray]


def _choose_heads(dim: int) -> int:
    for h in (8, 4, 2, 1):
        if dim % h == 0:
            return h
    return 1


def _infer_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    ids = set()
    needle = prefix + ".enc.layers."
    for key in state_dict.keys():
        if key.startswith(needle):
            rest = key[len(needle) :]
            ids.add(int(rest.split(".")[0]))
    return (max(ids) + 1) if ids else 0


def _infer_vel_layers(state_dict: dict[str, torch.Tensor]) -> int:
    ids = set()
    for key in state_dict.keys():
        if key.startswith("blocks."):
            rest = key[len("blocks.") :]
            ids.add(int(rest.split(".")[0]))
    return (max(ids) + 1) if ids else 1


def _center_crop_resize(img: np.ndarray, out_size: int) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0 : y0 + side, x0 : x0 + side, :]
    if side == out_size:
        return crop
    ys = np.linspace(0, side - 1, out_size).astype(np.int32)
    xs = np.linspace(0, side - 1, out_size).astype(np.int32)
    return crop[ys][:, xs]


def build_flow_backbone_from_ckpt(
    ckpt_path: str,
    device: str,
    history_len: int = -1,
) -> FlowBackboneBuildResult:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("ema_model", ckpt["model"])

    token_dim = int(state_dict["img_to_token.weight"].shape[0])
    img_dim = int(state_dict["img_enc.proj.weight"].shape[0])
    prop_dim = int(state_dict["prop_enc.0.weight"].shape[0])
    proprio_in_dim = int(state_dict["prop_enc.0.weight"].shape[1])
    time_dim = int(state_dict["time_mlp.0.weight"].shape[1])
    ckpt_history = int(state_dict["pos_emb"].shape[1]) - 3
    resolved_history = ckpt_history if int(history_len) <= 0 else int(history_len)

    temporal_layers = _infer_layers(state_dict, "temporal")
    temporal_layers = max(1, temporal_layers)
    temporal_heads = _choose_heads(token_dim)

    vel_hidden = int(state_dict["in_proj.weight"].shape[0])
    vel_layers = _infer_vel_layers(state_dict)

    chunk_size = int(state_dict["action_pos_emb"].shape[1])
    step_act_dim = int(state_dict["out.weight"].shape[0])
    act_dim = int(chunk_size * step_act_dim)

    action_temporal_layers = _infer_layers(state_dict, "action_temporal")
    action_temporal_heads = _choose_heads(vel_hidden)

    model = FlowPolicy(
        act_dim=act_dim,
        proprio_in_dim=proprio_in_dim,
        img_dim=img_dim,
        prop_dim=prop_dim,
        time_dim=time_dim,
        token_dim=token_dim,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        vel_hidden=vel_hidden,
        vel_layers=vel_layers,
        dropout=0.0,
        history_len=resolved_history,
        action_chunk_size=chunk_size,
        action_temporal_layers=action_temporal_layers,
        action_temporal_heads=action_temporal_heads,
        pretrained_resnet=False,
        freeze_resnet=True,
    )

    own = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
    own.update(filtered)
    model.load_state_dict(own, strict=False)
    model.to(device)
    model.eval()

    prop_mean = ckpt.get("prop_mean")
    prop_std = ckpt.get("prop_std")
    if torch.is_tensor(prop_mean):
        prop_mean = prop_mean.detach().cpu().numpy()
    if torch.is_tensor(prop_std):
        prop_std = prop_std.detach().cpu().numpy()
    if prop_mean is not None:
        prop_mean = np.asarray(prop_mean, dtype=np.float32)
    if prop_std is not None:
        prop_std = np.asarray(prop_std, dtype=np.float32)

    return FlowBackboneBuildResult(
        model=model,
        history_len=resolved_history,
        token_dim=token_dim,
        loaded_params=len(filtered),
        prop_mean=prop_mean,
        prop_std=prop_std,
    )


class FlowPolicyLatentExtractor:
    """
    Extract policy observation embeddings from Flow checkpoint.

    Embedding phi(o_t) is the conditional token used by the policy backbone.
    """

    def __init__(
        self,
        ckpt_path: str,
        camera_name: str,
        image_size: int,
        ta: int,
        to: int,
        device: str,
        batch_size: int = 128,
        history_len: int = -1,
        robots: str = "Panda",
        env_name: str = "Lift",
    ) -> None:
        self.camera_name = str(camera_name)
        self.image_size = int(image_size)
        self.ta = int(ta)
        self.to = int(to)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        if self.ta <= 0:
            raise ValueError(f"ta must be >=1, got {ta}")
        if self.to <= 0:
            raise ValueError(f"to must be >=1, got {to}")

        build = build_flow_backbone_from_ckpt(
            ckpt_path=ckpt_path,
            device=device,
            history_len=history_len,
        )
        self.model = build.model
        self.history_len = int(build.history_len)
        self.prop_mean = build.prop_mean
        self.prop_std = build.prop_std

        self.img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.proprio_extractor = PandaLiftProprioExtractor(
            robots=robots,
            env_name=env_name,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names=None,
            reward_shaping=False,
        )

    def close(self) -> None:
        self.proprio_extractor.close()

    def _compute_proprio(self, states: np.ndarray) -> np.ndarray:
        props = np.stack([self.proprio_extractor.extract(s) for s in states], axis=0).astype(np.float32)
        return props

    def _build_windows(self, images: np.ndarray, proprio: np.ndarray, sampled_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img_windows: list[np.ndarray] = []
        prop_windows: list[np.ndarray] = []

        for idx in sampled_indices.tolist():
            start = idx - self.history_len + 1
            window_idx = np.arange(start, idx + 1)
            window_idx = np.clip(window_idx, 0, images.shape[0] - 1)

            imgs = images[window_idx]
            props = proprio[window_idx]

            img_windows.append(imgs)
            prop_windows.append(props.reshape(-1))

        return np.asarray(img_windows, dtype=np.float32), np.asarray(prop_windows, dtype=np.float32)

    def _policy_cond_embedding(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        tokens_base = self.model.get_cond_features(images=images, proprio=proprio)
        batch_size = int(tokens_base.shape[0])

        t = torch.zeros(batch_size, device=images.device, dtype=images.dtype)
        t_emb = sinusoidal_time_embedding(t, self.model.time_dim)
        t_cond = self.model.time_mlp(t_emb)
        t_tok = self.model.time_token + t_cond.unsqueeze(1)

        tokens = torch.cat([t_tok, tokens_base], dim=1)
        tokens = tokens + self.model.pos_emb[:, : tokens.shape[1], :]
        tokens = self.model.token_time_adaln(tokens, t_cond)
        tokens = self.model.temporal(tokens)

        cond = tokens[:, 1]
        cond = self.model.cond_ln(cond)
        return cond

    @torch.no_grad()
    def encode_trajectory_with_indices(self, traj: PolicyTrajectory) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(traj.states, dtype=np.float32)
        images_raw = np.asarray(traj.images, dtype=np.uint8)

        if states.shape[0] != images_raw.shape[0]:
            raise ValueError(
                "state and image length mismatch in policy trajectory: "
                f"states={states.shape[0]} images={images_raw.shape[0]}"
            )
        if states.shape[0] == 0:
            raise ValueError("trajectory is empty")

        # preprocess full trajectory once
        images_chw = []
        for i in range(images_raw.shape[0]):
            img = _center_crop_resize(images_raw[i], self.image_size)
            img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
            images_chw.append(img)
        images_chw = np.asarray(images_chw, dtype=np.float32)

        proprio = self._compute_proprio(states)

        sampled_indices = np.arange(self.ta - 1, states.shape[0], self.ta)
        if sampled_indices.size == 0:
            sampled_indices = np.asarray([states.shape[0] - 1], dtype=np.int64)

        img_windows, prop_windows = self._build_windows(images=images_chw, proprio=proprio, sampled_indices=sampled_indices)

        if self.prop_mean is not None and self.prop_std is not None:
            if prop_windows.shape[1] == self.prop_mean.size:
                prop_windows = (prop_windows - self.prop_mean.reshape(1, -1)) / (
                    self.prop_std.reshape(1, -1) + 1e-6
                )

        outputs: list[np.ndarray] = []
        for start in range(0, img_windows.shape[0], self.batch_size):
            stop = min(img_windows.shape[0], start + self.batch_size)

            img_batch = torch.from_numpy(img_windows[start:stop]).to(self.device)
            prop_batch = torch.from_numpy(prop_windows[start:stop]).to(self.device)

            b, k, c, h, w = img_batch.shape
            img_flat = img_batch.view(b * k, c, h, w)
            img_norm = self.img_normalize(img_flat).view(b, k, c, h, w)

            emb = self._policy_cond_embedding(images=img_norm, proprio=prop_batch)
            outputs.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0), sampled_indices.astype(np.int64)

    @torch.no_grad()
    def encode_trajectory(self, traj: PolicyTrajectory) -> np.ndarray:
        emb, _ = self.encode_trajectory_with_indices(traj)
        return emb
