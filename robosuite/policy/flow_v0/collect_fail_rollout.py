"""
A script to collect failed rollouts from a base policy.

Behavior:
- Run the policy in the environment.
- Press keyboard SPACE to mark current episode as failure and save it.
- If the rollout succeeds, it is discarded automatically.
"""

import datetime
import json
import os
import shutil
import time
from collections import deque
from glob import glob

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pynput.keyboard import Key
from torchvision.transforms import Normalize

import robosuite as suite
from eval_flow import center_crop_resize, sample_action_fast
from flow import FlowPolicy
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.devices import Keyboard
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from utils.env_util import PandaLiftProprioExtractor


class BasePolicy:
    def reset(self, initial_obs, initial_images, env):
        pass

    def update_history(self, obs, step_images, env):
        pass

    def get_action(self):
        raise NotImplementedError


class DummyPolicy(BasePolicy):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_action(self):
        return np.random.normal(0, 0.02, size=self.action_dim)


class FlowMatchWrapper(BasePolicy):
    def __init__(self, cfg: DictConfig, env):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        if isinstance(cfg.camera, str):
            self.camera_names_list = [cfg.camera]
        else:
            self.camera_names_list = list(cfg.camera)

        self.extractor = PandaLiftProprioExtractor(
            robots="Panda",
            env_name="Lift",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.camera_names_list,
            reward_shaping=False,
        )
        self.extractor.env = env

        ckpt_path = to_absolute_path(cfg.ckpt)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        def _to_numpy(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return x

        self.act_mean = _to_numpy(ckpt.get("act_mean"))
        self.act_std = _to_numpy(ckpt.get("act_std"))
        self.prop_mean = _to_numpy(ckpt.get("prop_mean"))
        self.prop_std = _to_numpy(ckpt.get("prop_std"))

        state_dict = ckpt.get("ema_model", ckpt["model"])

        if self.act_mean is not None:
            act_dim = int(np.prod(self.act_mean.shape))
        else:
            out_w = next((state_dict[k] for k in ["out.weight", "head.weight", "vel_head.weight"] if k in state_dict), None)
            act_dim = int(out_w.shape[0])

        if self.prop_mean is not None:
            prop_dim = int(np.prod(self.prop_mean.shape))
        else:
            prop_w = next(
                (state_dict[k] for k in ["prop_enc.0.weight", "proprio_enc.0.weight", "prop_encoder.0.weight"] if k in state_dict),
                None,
            )
            prop_dim = int(prop_w.shape[1])

        self.base_act_dim = act_dim // cfg.chunk_size

        self.model = FlowPolicy(
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
            history_len=cfg.history_len,
            action_chunk_size=cfg.chunk_size,
            action_temporal_layers=cfg.flow.action_temporal_layers,
            action_temporal_heads=cfg.flow.action_temporal_heads,
        ).to(self.device)

        try:
            if "ema_model" in ckpt:
                self.model.load_state_dict(ckpt["ema_model"], strict=True)
                print("Successfully loaded EMA model weights for Flow Policy.")
            else:
                self.model.load_state_dict(ckpt["model"], strict=True)
                print("Loaded standard model weights for Flow Policy.")
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint is incompatible with current FlowPolicy architecture. "
                "Please retrain with the updated model or switch to a matching code version."
            ) from e

        self.model.eval()
        self.img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_history = deque(maxlen=cfg.history_len)
        self.prop_history = deque(maxlen=cfg.history_len)
        self.current_chunk = None
        self.step_in_chunk = 0

    def reset(self, initial_obs, initial_images, env):
        self.img_history.clear()
        self.prop_history.clear()

        prop = self.extractor.extract(env.sim.get_state().flatten()).reshape(-1)
        cam_name = self.camera_names_list[0]
        raw_img = initial_images[cam_name]
        img = center_crop_resize(raw_img, self.cfg.image_size)
        img_processed = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))

        for _ in range(self.cfg.history_len):
            self.prop_history.append(prop)
            self.img_history.append(img_processed)

        self.current_chunk = None
        self.step_in_chunk = 0

    def update_history(self, obs, step_images, env):
        prop = self.extractor.extract(env.sim.get_state().flatten()).reshape(-1)
        self.prop_history.append(prop)

        cam_name = self.camera_names_list[0]
        raw_img = step_images[cam_name]
        img = center_crop_resize(raw_img, self.cfg.image_size)
        img_processed = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        self.img_history.append(img_processed)

    def get_action(self):
        execute_steps = min(self.cfg.action_horizon, self.cfg.chunk_size)

        if self.current_chunk is None or self.step_in_chunk >= execute_steps:
            imgs_stacked = np.stack(self.img_history, axis=0)
            images = torch.from_numpy(imgs_stacked).unsqueeze(0).to(self.device)
            bsz, ksz, csz, hsz, wsz = images.shape
            images_flat = images.view(bsz * ksz, csz, hsz, wsz)
            images_norm = self.img_normalize(images_flat).view(bsz, ksz, csz, hsz, wsz)

            prop_stacked = np.concatenate(self.prop_history, axis=0).astype(np.float32)
            if self.prop_mean is not None:
                prop_stacked = (prop_stacked - self.prop_mean) / self.prop_std
            proprio = torch.from_numpy(prop_stacked).unsqueeze(0).to(self.device)

            a_norm = sample_action_fast(self.model, images_norm, proprio, n_steps=self.cfg.n_ode_steps).cpu().numpy().reshape(-1)
            if self.act_mean is not None:
                a_flat = a_norm * self.act_std + self.act_mean
            else:
                a_flat = a_norm

            self.current_chunk = a_flat.reshape(self.cfg.chunk_size, self.base_act_dim)
            self.step_in_chunk = 0

        action_step = self.current_chunk[self.step_in_chunk]
        self.step_in_chunk += 1
        return action_step


class EpisodeStopKeyboard(Keyboard):
    """Keyboard device that also exposes a one-shot SPACE stop signal."""

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__(env=env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        self._stop_requested = False

    def on_release(self, key):
        if key == Key.space:
            self._stop_requested = True
        super().on_release(key)

    def consume_stop_signal(self) -> bool:
        if self._stop_requested:
            self._stop_requested = False
            return True
        return False


def now_readable(ts: datetime.datetime | None = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()
    return ts.strftime("%Y-%m-%d_%H-%M-%S")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_ep_dirs(tmp_root: str) -> list[str]:
    if not os.path.isdir(tmp_root):
        return []
    ep_dirs = [os.path.join(tmp_root, d) for d in os.listdir(tmp_root) if d.startswith("ep_")]
    ep_dirs = [d for d in ep_dirs if os.path.isdir(d)]
    ep_dirs.sort(key=lambda p: os.path.getmtime(p))
    return ep_dirs


def get_current_ep_dir(env_wrapped, tmp_root: str) -> str | None:
    ep_dir = getattr(env_wrapped, "ep_directory", None)
    if isinstance(ep_dir, str) and len(ep_dir) > 0:
        return ep_dir
    eps = list_ep_dirs(tmp_root)
    return eps[-1] if len(eps) > 0 else None


def read_single_demo_from_ep(ep_dir: str):
    state_paths = os.path.join(ep_dir, "state_*.npz")
    states = []
    actions = []
    success = False
    env_name = None

    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file, allow_pickle=True)
        env_name = str(dic["env"])
        states.extend(dic["states"])
        for ai in dic["action_infos"]:
            actions.append(ai["actions"])
        success = success or dic["successful"]

    if len(states) == 0:
        return None

    del states[-1]

    states = np.array(states)
    actions = np.array(actions)

    return {
        "env_name": env_name,
        "states": states,
        "actions": actions,
        "success": bool(success),
        "ep_dir": ep_dir,
    }


def append_demo_to_hdf5(
    hdf5_path: str,
    demo_id: int,
    demo_payload: dict,
    env_info: str,
    camera_names: list[str],
    images_dict: dict,
):
    """
    Save format intentionally matches scripts/collect_human_demonstrations.py.
    """
    camera_names = list(camera_names)
    f = h5py.File(hdf5_path, "a")

    if "demos" not in f:
        demos_grp = f.create_group("demos")
        f.attrs["created_at"] = now_readable()
        f.attrs["repository_version"] = suite.__version__
        f.attrs["env"] = demo_payload.get("env_name", "")
        f.attrs["env_info"] = env_info
        f.attrs["camera_names"] = json.dumps(camera_names)
    else:
        demos_grp = f["demos"]

    demo_grp = demos_grp.create_group(f"demo_{demo_id:06d}")

    xml_path = os.path.join(demo_payload["ep_dir"], "model.xml")
    if os.path.isfile(xml_path):
        with open(xml_path, "r") as fx:
            xml_str = fx.read()
        demo_grp.attrs["model_file"] = xml_str

    states = demo_payload["states"]
    actions = demo_payload["actions"]

    tlen = min(states.shape[0], actions.shape[0])
    for cam in camera_names:
        imgs = images_dict.get(cam, None)
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            tlen = min(tlen, imgs.shape[0])

    states = states[:tlen]
    actions = actions[:tlen]

    demo_grp.attrs["length"] = int(tlen)
    demo_grp.attrs["successful"] = bool(demo_payload.get("success", False))

    demo_grp.create_dataset("states", data=states)
    demo_grp.create_dataset("actions", data=actions)

    obs_grp = demo_grp.create_group("observations")
    for cam in camera_names:
        cam_grp = obs_grp.create_group(cam)
        imgs = images_dict.get(cam, None)
        if imgs is None or not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[0] == 0:
            cam_grp.create_dataset("images", data=np.zeros((0,), dtype=np.uint8))
            continue

        imgs = imgs[:tlen].astype(np.uint8)
        cam_grp.create_dataset(
            "images",
            data=imgs,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

    f.close()


def build_env(env_config: dict, cfg):
    camera_names = cfg.camera if len(cfg.camera) > 0 else ["agentview"]
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=cfg.renderer,
        has_offscreen_renderer=True,
        render_camera=camera_names[0],
        camera_names=camera_names,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    return env


def build_keyboard_device(env, cfg):
    if cfg.device != "keyboard":
        raise ValueError("collect_fail_rollout supports keyboard only, because SPACE is used as failure trigger.")
    return EpisodeStopKeyboard(env=env, pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)


def collect_policy_rollout_until_event(
    env,
    device: EpisodeStopKeyboard,
    policy: BasePolicy,
    camera_names: list[str],
    img_height: int,
    img_width: int,
    max_fr: int | None,
    max_steps: int | None,
):
    env.render()
    device.start_control()

    images = {cam: [] for cam in camera_names}

    obs = env.unwrapped._get_observations()
    initial_images = {cam: env.sim.render(height=img_height, width=img_width, camera_name=cam) for cam in camera_names}
    policy.reset(obs, initial_images, env.unwrapped)

    stop_by_space = False
    success = False
    user_quit = False
    step_count = 0

    while True:
        start = time.time()

        if device.consume_stop_signal():
            stop_by_space = True
            break

        if getattr(device, "_reset_state", 0) == 1:
            user_quit = True
            break

        if max_steps is not None and step_count >= max_steps:
            break

        env_action = policy.get_action()
        obs, _, _, _ = env.step(env_action)
        step_count += 1

        step_images = {}
        for cam in camera_names:
            img = env.sim.render(height=img_height, width=img_width, camera_name=cam)
            step_images[cam] = img
            images[cam].append(img)

        policy.update_history(obs, step_images, env.unwrapped)
        env.render()

        if env._check_success():
            success = True
            break

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    images_np = {}
    for cam in camera_names:
        if len(images[cam]) == 0:
            images_np[cam] = np.zeros((0,), dtype=np.uint8)
        else:
            images_np[cam] = np.stack(images[cam], axis=0).astype(np.uint8)

    return images_np, stop_by_space, success, user_quit, step_count


@hydra.main(version_base="1.2", config_path="./config", config_name="collect_fail_rollout")
def main(cfg: DictConfig):
    if isinstance(cfg.camera, str):
        camera_names = [cfg.camera]
    else:
        camera_names = list(cfg.camera)
    if len(camera_names) == 0:
        camera_names = ["agentview"]

    rcfg = cfg.fail_rollout

    controller_config = load_composite_controller_config(
        controller=rcfg.controller,
        robot=rcfg.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401

    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(rcfg.robots) == 1, "Whole Body IK only supports one robot"

    env_config = {
        "env_name": rcfg.environment,
        "robots": list(rcfg.robots),
        "controller_configs": controller_config,
    }
    if "TwoArm" in rcfg.environment:
        env_config["env_configuration"] = rcfg.config

    env_info = json.dumps(env_config)

    safe_mkdir(rcfg.directory)
    base_time = now_readable()
    tmp_root = os.path.join("/tmp", f"robosuite_fail_rollout_{base_time}")
    safe_mkdir(tmp_root)

    tmp_hdf5_path = os.path.join(rcfg.directory, f"fail_rollout_{base_time}_0.hdf5")
    print(f"Output HDF5: {tmp_hdf5_path}")

    class DummyArgs:
        pass

    args = DummyArgs()
    for k, v in rcfg.items():
        setattr(args, k, v)
    setattr(args, "camera", camera_names)

    env = build_env(env_config, args)
    device = build_keyboard_device(env, args)

    if cfg.policy_type == "flow":
        print("Using FlowMatching as the base policy.")
        policy = FlowMatchWrapper(cfg, env.unwrapped)
    elif cfg.policy_type == "dummy":
        print("Using Dummy Policy (Random Noise).")
        action_dim = env.action_spec[0].shape[0]
        policy = DummyPolicy(action_dim)
    else:
        raise ValueError(f"Unknown policy_type: {cfg.policy_type}")

    env_wrapped = DataCollectionWrapper(env, tmp_root)
    saved_count = 0
    attempt_count = 0
    env_wrapped.reset()

    try:
        while True:
            if int(rcfg.target_failures) > 0 and saved_count >= int(rcfg.target_failures):
                print(f"Reached target_failures={int(rcfg.target_failures)}.")
                break

            attempt_count += 1
            images_dict, stop_by_space, success, user_quit, step_count = collect_policy_rollout_until_event(
                env=env_wrapped,
                device=device,
                policy=policy,
                camera_names=camera_names,
                img_height=rcfg.img_height,
                img_width=rcfg.img_width,
                max_fr=rcfg.max_fr,
                max_steps=rcfg.max_steps,
            )

            ep_dir = get_current_ep_dir(env_wrapped, tmp_root)
            env_wrapped.reset()

            if ep_dir is None or not os.path.isdir(ep_dir):
                print("Empty demo. Discarded.")
                if user_quit:
                    print("Quit requested.")
                    break
                continue

            demo_payload = read_single_demo_from_ep(ep_dir)
            if demo_payload is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Empty demo. Discarded.")
                if user_quit:
                    print("Quit requested.")
                    break
                continue

            # Keep successful rollouts discarded even if user requested stop.
            is_success = bool(success or demo_payload.get("success", False))
            should_save = bool(stop_by_space and (not is_success))

            print(
                f"attempt={attempt_count} steps={step_count} stop_by_space={stop_by_space} "
                f"success={is_success} save={should_save}"
            )

            if should_save:
                saved_count += 1
                demo_payload["success"] = False
                append_demo_to_hdf5(
                    hdf5_path=tmp_hdf5_path,
                    demo_id=saved_count,
                    demo_payload=demo_payload,
                    env_info=env_info,
                    camera_names=camera_names,
                    images_dict=images_dict,
                )
                print(f"Saved demo_{saved_count:06d}.")
            else:
                if is_success:
                    print("Skipped save: successful rollout.")
                elif not stop_by_space:
                    print("Skipped save: rollout was not terminated by SPACE.")

            shutil.rmtree(ep_dir, ignore_errors=True)

            if user_quit:
                print("Quit requested.")
                break

    finally:
        try:
            if hasattr(device, "stop_control"):
                device.stop_control()
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass

    final_hdf5_path = os.path.join(rcfg.directory, f"fail_rollout_{base_time}_{saved_count}.hdf5")
    print(f"\nFinal episodes: {saved_count}")
    try:
        if os.path.isfile(tmp_hdf5_path):
            os.replace(tmp_hdf5_path, final_hdf5_path)
            print(f"Final HDF5: {final_hdf5_path}")
        else:
            print("No HDF5 file created.")
    except Exception as e:
        print(f"Failed to rename HDF5: {e}")
        print(f"Kept temporary file: {tmp_hdf5_path}")


if __name__ == "__main__":
    main()
