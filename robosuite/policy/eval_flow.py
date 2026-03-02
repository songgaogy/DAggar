import os
import numpy as np
import torch
import robosuite as suite
import imageio
from collections import deque
from torchvision.transforms import Normalize

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from utils.env_util import PandaLiftProprioExtractor
from flow import FlowPolicy


def center_crop_resize(img, out_size):
    H, W = img.shape[0], img.shape[1]
    s = min(H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    crop = img[y0 : y0 + s, x0 : x0 + s, :]
    if s == out_size:
        return crop
    ys = np.linspace(0, s - 1, out_size).astype(np.int32)
    xs = np.linspace(0, s - 1, out_size).astype(np.int32)
    out = crop[ys][:, xs]
    return out


@torch.no_grad()
def sample_action_fast(model, images, proprio, n_steps):
    """Euler integrate the flow field from t=0 -> 1.

    Keep the same calling convention as training:
        v_pred = model(x_t=..., t=..., images=..., proprio=...)
    This avoids relying on optional helper APIs that may not exist.
    """
    device = proprio.device
    B = proprio.shape[0]

    x = torch.randn(B, model.act_dim, device=device)
    dt = 1.0 / float(n_steps)

    for i in range(n_steps):
        t = torch.full((B,), float(i) / float(n_steps), device=device)
        v = model(x_t=x, t=t, images=images, proprio=proprio)
        x = x + dt * v

    return x


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_flow")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt_path = to_absolute_path(cfg.ckpt)
    video_dir = to_absolute_path(cfg.video_dir) if cfg.video_dir else None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    def _to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    act_mean = _to_numpy(ckpt.get("act_mean"))
    act_std = _to_numpy(ckpt.get("act_std"))
    prop_mean = _to_numpy(ckpt.get("prop_mean"))
    prop_std = _to_numpy(ckpt.get("prop_std"))

    extractor = PandaLiftProprioExtractor(
        robots="Panda",
        env_name="Lift",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[cfg.camera],
        reward_shaping=False,
    )

    env = extractor.env
    
    state_dict = ckpt.get("ema_model", ckpt["model"])

    # Prefer dims from saved normalization stats (most robust across refactors)
    if act_mean is not None:
        act_dim = int(np.prod(act_mean.shape))
    else:
        out_w = None
        for k in ["out.weight", "head.weight", "vel_head.weight"]:
            if k in state_dict:
                out_w = state_dict[k]
                break
        if out_w is None:
            raise KeyError(
                "Cannot infer act_dim: missing act_mean and cannot find output weight key in checkpoint state_dict."
            )
        act_dim = int(out_w.shape[0])

    if prop_mean is not None:
        prop_dim = int(np.prod(prop_mean.shape))
    else:
        prop_w = None
        for k in ["prop_enc.0.weight", "proprio_enc.0.weight", "prop_encoder.0.weight"]:
            if k in state_dict:
                prop_w = state_dict[k]
                break
        if prop_w is None:
            raise KeyError(
                "Cannot infer prop_dim: missing prop_mean and cannot find proprio encoder weight key in checkpoint state_dict."
            )
        prop_dim = int(prop_w.shape[1])

    base_act_dim = act_dim // cfg.chunk_size

    model = FlowPolicy(
        act_dim=act_dim,
        proprio_in_dim=prop_dim,
        img_dim=cfg.flow.image_dim,
        prop_dim=cfg.flow.propior_dim,
        time_dim=cfg.flow.time_dim,
        token_dim=cfg.flow.token_dim,
        pretrained_resnet=True,
        freeze_resnet=True,
        temporal_layers=cfg.flow.temporal_layers,
        temporal_heads=cfg.flow.temporal_heads,
        vel_hidden=cfg.flow.vel_hidden,
        vel_layers=cfg.flow.vel_layers,
        history_len=cfg.history_len
    ).to(device)

    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"], strict=True)
        print("Successfully loaded EMA model weights.")
    else:
        model.load_state_dict(ckpt["model"], strict=True)
        print("Loaded standard model weights (EMA not found).")
        
    model.eval()

    if cfg.video_dir is not None:
        os.makedirs(cfg.video_dir, exist_ok=True)

    # torchvision Normalize is not a nn.Module; do not call .to(device)
    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for ep in range(cfg.episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        frames = []
        step_count = 0
        
        img_history = deque(maxlen=cfg.history_len)
        prop_history = deque(maxlen=cfg.history_len)
        
        initial_prop = extractor.extract(env.sim.get_state().flatten()).reshape(-1)
        raw_img = obs[cfg.camera + "_image"]
        img = center_crop_resize(raw_img, cfg.image_size)
        img_processed = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))

        for _ in range(cfg.history_len):
            prop_history.append(initial_prop)
            img_history.append(img_processed)

        while step_count < cfg.max_steps and not done:
            imgs_stacked = np.stack(img_history, axis=0) # [K, C, H, W]
            images = torch.from_numpy(imgs_stacked).unsqueeze(0).to(device) # [1, K, C, H, W]

            # Match train_flow.py normalization path
            B, K, C, H, W = images.shape
            images_flat = images.view(B * K, C, H, W)
            images_norm = img_normalize(images_flat).view(B, K, C, H, W)
            
            prop_stacked = np.concatenate(prop_history, axis=0).astype(np.float32)
            if prop_mean is not None:
                if prop_std is None:
                    raise ValueError("prop_mean is present but prop_std is missing in checkpoint.")
                prop_stacked = (prop_stacked - prop_mean) / prop_std
            proprio = torch.from_numpy(prop_stacked).unsqueeze(0).to(device)

            a_norm = sample_action_fast(model, images_norm, proprio, n_steps=cfg.n_ode_steps).cpu().numpy().reshape(-1)
            
            if act_mean is not None:
                if act_std is None:
                    raise ValueError("act_mean is present but act_std is missing in checkpoint.")
                a_flat = a_norm * act_std + act_mean
            else:
                a_flat = a_norm

            a_chunk = a_flat.reshape(cfg.chunk_size, base_act_dim)

            execute_steps = min(cfg.action_horizon, cfg.chunk_size)
            for i in range(execute_steps):
                if cfg.video_dir is not None:
                    frames.append(np.flipud(obs[cfg.camera + "_image"]))

                action_step = a_chunk[i]
                obs, r, done, info = env.step(action_step)
                total_r += float(r)
                step_count += 1

                new_prop = extractor.extract(env.sim.get_state().flatten()).reshape(-1)
                prop_history.append(new_prop)
                
                new_raw_img = obs[cfg.camera + "_image"]
                new_img = center_crop_resize(new_raw_img, cfg.image_size)
                new_img_processed = np.transpose(new_img.astype(np.float32) / 255.0, (2, 0, 1))
                img_history.append(new_img_processed)

                if done or step_count >= cfg.max_steps:
                    break

        print(f"episode={ep} return={total_r:.4f} steps={step_count}")

        if video_dir is not None and len(frames) > 0:
            video_path = os.path.join(cfg.video_dir, f"episode_{ep}_return_{total_r:.2f}.mp4")
            imageio.mimsave(video_path, frames, fps=20)
            print(f"Saved video to {video_path}")

    extractor.close()


if __name__ == "__main__":
    main()