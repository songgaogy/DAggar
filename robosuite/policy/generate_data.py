import os
import json
import datetime
import h5py
import numpy as np
import torch
import robosuite as suite
from tqdm import tqdm
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
    device = proprio.device
    B = proprio.shape[0]

    x = torch.randn(B, model.act_dim, device=device)
    dt = 1.0 / float(n_steps)

    for i in range(n_steps):
        t = torch.full((B,), float(i) / float(n_steps), device=device)
        v = model(x_t=x, t=t, images=images, proprio=proprio)
        x = x + dt * v

    return x


def append_demo_to_hdf5(
    hdf5_path: str,
    demo_id: int,
    states: list,
    actions: list,
    images_dict: dict,
    success: bool,
    env_name: str,
    env_info: str,
    camera_names: list,
    xml_str: str,
):
    """
    Append a single demonstration to the unified HDF5 file.
    The data structure strictly aligns with the output of collect_human_demonstrations.py.
    """
    f = h5py.File(hdf5_path, "a")

    if "demos" not in f:
        demos_grp = f.create_group("demos")
        f.attrs["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        f.attrs["repository_version"] = suite.__version__
        f.attrs["env"] = env_name
        f.attrs["env_info"] = env_info
        f.attrs["camera_names"] = json.dumps(camera_names)
    else:
        demos_grp = f["demos"]

    demo_grp = demos_grp.create_group(f"demo_{demo_id:06d}")

    if xml_str:
        demo_grp.attrs["model_file"] = xml_str

    states_np = np.array(states)
    actions_np = np.array(actions)

    # Align sequence lengths
    T = min(states_np.shape[0], actions_np.shape[0])
    states_np = states_np[:T]
    actions_np = actions_np[:T]

    demo_grp.attrs["length"] = int(T)
    demo_grp.attrs["successful"] = bool(success)

    demo_grp.create_dataset("states", data=states_np)
    demo_grp.create_dataset("actions", data=actions_np)

    obs_grp = demo_grp.create_group("observations")

    for cam in camera_names:
        cam_grp = obs_grp.create_group(cam)
        imgs = images_dict.get(cam, [])
        if len(imgs) == 0:
            cam_grp.create_dataset("images", data=np.zeros((0,), dtype=np.uint8))
            continue
            
        imgs_np = np.stack(imgs, axis=0).astype(np.uint8)[:T]

        cam_grp.create_dataset(
            "images",
            data=imgs_np,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

    f.close()


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_flow")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = to_absolute_path(cfg.ckpt)
    
    # Set HDF5 output directory and max sequence length
    output_dir = to_absolute_path(cfg.generate.output_dir)
    max_len = cfg.max_steps
    
    os.makedirs(output_dir, exist_ok=True)
    base_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hdf5_path = os.path.join(output_dir, f"policy_rollout_{base_time}_ep{cfg.generate.episodes}.hdf5")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    def _to_numpy(x):
        if x is None: return None
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return x

    act_mean, act_std = _to_numpy(ckpt.get("act_mean")), _to_numpy(ckpt.get("act_std"))
    prop_mean, prop_std = _to_numpy(ckpt.get("prop_mean")), _to_numpy(ckpt.get("prop_std"))

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

    # Infer dimensions automatically based on checkpoint keys
    if act_mean is not None: 
        act_dim = int(np.prod(act_mean.shape))
    else: 
        act_dim = int([state_dict[k] for k in ["out.weight", "head.weight", "vel_head.weight"] if k in state_dict][0].shape[0])

    if prop_mean is not None: 
        prop_dim = int(np.prod(prop_mean.shape))
    else: 
        prop_dim = int([state_dict[k] for k in ["prop_enc.0.weight", "proprio_enc.0.weight", "prop_encoder.0.weight"] if k in state_dict][0].shape[1])

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

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Extract simulation XML to store as an HDF5 attribute
    try:
        xml_str = env.sim.model.get_xml()
    except Exception:
        xml_str = ""

    env_info = json.dumps({"env_name": "Lift", "robots": ["Panda"]})

    saved_count = 0

    for ep in range(cfg.generate.episodes):
        obs = env.reset()
        done = False
        step_count = 0
        
        # Containers for episode trajectory data
        ep_states = []
        ep_actions = []
        ep_images = {cfg.camera: []}
        success = False
        
        img_history = deque(maxlen=cfg.history_len)
        prop_history = deque(maxlen=cfg.history_len)
        
        initial_prop = extractor.extract(env.sim.get_state().flatten()).reshape(-1)
        raw_img = obs[cfg.camera + "_image"]
        
        img = center_crop_resize(raw_img, cfg.image_size)
        img_processed = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))

        for _ in range(cfg.history_len):
            prop_history.append(initial_prop)
            img_history.append(img_processed)

        # Execution loop strictly bounded by max_len
        while step_count < max_len and not done:
            imgs_stacked = np.stack(img_history, axis=0)
            images = torch.from_numpy(imgs_stacked).unsqueeze(0).to(device)

            B, K, C, H, W = images.shape
            images_flat = images.view(B * K, C, H, W)
            images_norm = img_normalize(images_flat).view(B, K, C, H, W)
            
            prop_stacked = np.concatenate(prop_history, axis=0).astype(np.float32)
            if prop_mean is not None:
                prop_stacked = (prop_stacked - prop_mean) / prop_std
            proprio = torch.from_numpy(prop_stacked).unsqueeze(0).to(device)

            a_norm = sample_action_fast(model, images_norm, proprio, n_steps=cfg.n_ode_steps).cpu().numpy().reshape(-1)
            
            a_flat = a_norm * act_std + act_mean if act_mean is not None else a_norm
            a_chunk = a_flat.reshape(cfg.chunk_size, base_act_dim)

            execute_steps = min(cfg.action_horizon, cfg.chunk_size)
            for i in range(execute_steps):
                action_step = a_chunk[i]
                
                # Record state (pre-action) and raw image (flipped vertically to match visualizers)
                ep_states.append(env.sim.get_state().flatten())
                ep_actions.append(action_step)
                ep_images[cfg.camera].append(np.flipud(obs[cfg.camera + "_image"]))

                # Step the environment
                obs, r, d, info = env.step(action_step)
                step_count += 1
                
                # Check for task success
                if env._check_success():
                    success = True

                new_prop = extractor.extract(env.sim.get_state().flatten()).reshape(-1)
                prop_history.append(new_prop)
                
                new_raw_img = obs[cfg.camera + "_image"]
                new_img = center_crop_resize(new_raw_img, cfg.image_size)
                new_img_processed = np.transpose(new_img.astype(np.float32) / 255.0, (2, 0, 1))
                img_history.append(new_img_processed)

                if d or step_count >= max_len:
                    done = True
                    break

        print(f"episode={ep} success={success} steps={step_count}")

        # Save the episode data
        saved_count += 1
        append_demo_to_hdf5(
            hdf5_path=hdf5_path,
            demo_id=saved_count,
            states=ep_states,
            actions=ep_actions,
            images_dict=ep_images,
            success=success,
            env_name="Lift",
            env_info=env_info,
            camera_names=[cfg.camera],
            xml_str=xml_str
        )
        print(f"Saved data_{saved_count:06d} to {hdf5_path}")

    extractor.close()
    print(f"\nFinished. All data saved to {hdf5_path}")

if __name__ == "__main__":
    main()