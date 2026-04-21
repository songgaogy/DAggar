import datetime
import os
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from robosuite.policy.flow_multi.utils.datasets import RobosuiteMultiViewFlowDataset
from robosuite.policy.flow_multi.model import build_flow_policy


def _cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return images
    bsz, channels, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    padded_height = height + 2 * pad
    padded_width = width + 2 * pad

    eps_y = 1.0 / padded_height
    eps_x = 1.0 / padded_width
    base_y = torch.linspace(
        -1.0 + eps_y,
        1.0 - eps_y,
        padded_height,
        device=images.device,
        dtype=images.dtype,
    )[:height]
    base_x = torch.linspace(
        -1.0 + eps_x,
        1.0 - eps_x,
        padded_width,
        device=images.device,
        dtype=images.dtype,
    )[:width]
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)

    shift_x = torch.randint(0, 2 * pad + 1, (bsz, 1, 1), device=images.device)
    shift_y = torch.randint(0, 2 * pad + 1, (bsz, 1, 1), device=images.device)
    shift = torch.stack(
        [
            shift_x.to(dtype=images.dtype) * (2.0 / padded_width),
            shift_y.to(dtype=images.dtype) * (2.0 / padded_height),
        ],
        dim=-1,
    )
    grid = base_grid + shift
    return F.grid_sample(padded, grid, padding_mode="zeros", align_corners=False)


def random_crop_resize(images: torch.Tensor, crop_scale: float) -> torch.Tensor:
    if crop_scale >= 0.999:
        return images
    bsz, channels, height, width = images.shape
    crop_h = max(1, int(round(height * crop_scale)))
    crop_w = max(1, int(round(width * crop_scale)))
    y0 = torch.randint(0, height - crop_h + 1, (bsz,), device=images.device)
    x0 = torch.randint(0, width - crop_w + 1, (bsz,), device=images.device)

    center_x = ((x0.to(images.dtype) + 0.5 * crop_w) * 2.0 / width) - 1.0
    center_y = ((y0.to(images.dtype) + 0.5 * crop_h) * 2.0 / height) - 1.0
    scale_x = torch.full((bsz,), float(crop_w) / float(width), device=images.device, dtype=images.dtype)
    scale_y = torch.full((bsz,), float(crop_h) / float(height), device=images.device, dtype=images.dtype)

    theta = torch.zeros((bsz, 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y
    theta[:, 0, 2] = center_x
    theta[:, 1, 2] = center_y

    grid = F.affine_grid(theta, size=images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


class CUDAPrefetcher:
    def __init__(self, loader, device: torch.device, camera_names: list[str], cfg: DictConfig):
        self.loader = loader
        self.device = device
        self.camera_names = camera_names
        self.cfg = cfg
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.loader_iter = None
        self.next_batch = None

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.stream is not None:
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_stream(self.stream)
            for value in self.next_batch.values():
                if torch.is_tensor(value):
                    value.record_stream(current_stream)
        batch = self.next_batch
        self._preload()
        return batch

    def _move_batch(self, batch):
        images = batch["images"].to(self.device, non_blocking=True)
        proprio = batch["proprio"].to(self.device, non_blocking=True)
        actions = batch["actions"].to(self.device, non_blocking=True)
        images = preprocess_images(images, self.camera_names, self.cfg)
        return {
            "images": images,
            "proprio": proprio,
            "actions": actions,
            "language": list(batch["language"]),
            "task_name": list(batch["task_name"]),
        }

    def _preload(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = self._move_batch(batch)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._move_batch(batch)


class DemoBatchSampler:
    def __init__(self, dataset: RobosuiteMultiViewFlowDataset, batch_size: int, drop_last: bool = True, seed: int = 0):
        self.indices_by_demo = {
            meta_id: np.asarray(sample_indices, dtype=np.int64)
            for meta_id, sample_indices in dataset.sample_indices_by_meta_id.items()
        }
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        total_samples = int(sum(len(sample_indices) for sample_indices in self.indices_by_demo.values()))
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        demo_ids = list(self.indices_by_demo.keys())
        rng.shuffle(demo_ids)

        batch = []
        for meta_id in demo_ids:
            sample_indices = self.indices_by_demo[meta_id].copy()
            rng.shuffle(sample_indices)
            for sample_idx in sample_indices.tolist():
                batch.append(int(sample_idx))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


def preprocess_images(images: torch.Tensor, camera_names: list[str], cfg: DictConfig) -> torch.Tensor:
    images = images.to(dtype=torch.float32).div_(255.0)
    augmented = images
    minimal_shift_pad = int(cfg.augmentation.minimal_shift_pad)
    eye_in_hand_crop_scale = float(cfg.augmentation.eye_in_hand_crop_scale)

    for view_idx, camera_name in enumerate(camera_names):
        if camera_name == "robot0_eye_in_hand":
            augmented[:, view_idx] = random_crop_resize(augmented[:, view_idx], crop_scale=eye_in_hand_crop_scale)
        else:
            augmented[:, view_idx] = random_shift(augmented[:, view_idx], pad=minimal_shift_pad)

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    return (augmented - mean) / std


def maybe_init_wandb(cfg: DictConfig):
    if not bool(cfg.logging.enable_wandb):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed, skipping wandb logging.")
        return None

    wandb.init(
        project=str(cfg.logging.project),
        entity=str(cfg.logging.entity),
        mode=str(cfg.logging.mode),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return wandb


@hydra.main(version_base="1.2", config_path="./config", config_name="train_flow")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    save_dir = to_absolute_path(cfg.checkpoint.save_dir)
    data_dir_cfg = _cfg_get(cfg.data, "data_dir", None)
    data_dir = None if data_dir_cfg in (None, "null") else to_absolute_path(str(data_dir_cfg))
    data_dirs_cfg = _cfg_get(cfg.data, "data_dirs", None)
    data_dirs = None if data_dirs_cfg is None else [to_absolute_path(str(path)) for path in data_dirs_cfg]
    num_traj_cfg = _cfg_get(cfg.data, "num_traj", None)
    if num_traj_cfg in (None, "null"):
        num_traj_cfg = _cfg_get(cfg.data, "num_train_traj", None)
    os.makedirs(save_dir, exist_ok=True)

    dataset = RobosuiteMultiViewFlowDataset(
        data_dir=data_dir if data_dirs is None else None,
        data_dirs=data_dirs,
        camera_names=list(cfg.data.camera_names),
        action_horizon=int(cfg.data.action_horizon),
        image_size=int(cfg.data.image_size),
        stride=int(cfg.data.stride),
        normalize=bool(cfg.data.normalize),
        max_demos_per_file=cfg.data.max_demos_per_file,
        num_train_traj=num_traj_cfg,
        cache_proprio=bool(cfg.data.cache_proprio),
        use_demo_cache=bool(cfg.data.use_demo_cache),
        max_cached_demos_per_worker=int(cfg.data.max_cached_demos_per_worker),
        preload_all_demos_to_ram=bool(cfg.data.preload_all_demos_to_ram),
        use_disk_cache=bool(cfg.data.use_disk_cache),
        task_prompt_map=_cfg_get(cfg.data, "task_prompt_map", None),
    )

    batch_sampler = DemoBatchSampler(
        dataset=dataset,
        batch_size=int(cfg.train.batch_size),
        drop_last=True,
        seed=int(cfg.seed),
    )
    data_loader_kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        persistent_workers=bool(cfg.train.num_workers > 0),
    )
    if int(cfg.train.num_workers) > 0:
        data_loader_kwargs["prefetch_factor"] = int(cfg.train.prefetch_factor)
    data_loader = DataLoader(**data_loader_kwargs)

    sample = dataset[0]
    action_dim = int(sample["actions"].shape[-1])
    proprio_dim = int(sample["proprio"].numel())

    model = build_flow_policy(
        cfg.flow,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_names=list(cfg.data.camera_names),
    ).to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(cfg.train.ema_decay)))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"), device=device)
    total_steps = int(cfg.train.epochs) * len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=float(cfg.train.min_lr),
    )

    wandb = maybe_init_wandb(cfg)
    global_step = 0
    print("start training flow_multi...")

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        batch_iter = CUDAPrefetcher(
            loader=data_loader,
            device=device,
            camera_names=list(cfg.data.camera_names),
            cfg=cfg,
        )
        running_loss = 0.0
        running_flow = 0.0
        running_endpoint = 0.0
        running_smooth = 0.0
        loop_start = time.time()
        num_batches = 0

        for iteration, batch in enumerate(batch_iter):
            num_batches += 1
            images = batch["images"]
            proprio = batch["proprio"]
            x1 = batch["actions"]
            language = batch["language"]
            noise = torch.randn_like(x1)
            timesteps = torch.rand(x1.shape[0], device=device)
            x_t = (1.0 - timesteps).view(-1, 1, 1) * noise + timesteps.view(-1, 1, 1) * x1
            v_target = x1 - noise

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=(device.type == "cuda"), device_type=device.type):
                v_pred = model(
                    x_t=x_t.transpose(1, 2),
                    t=timesteps,
                    images=images,
                    proprio=proprio,
                    language=language,
                ).transpose(1, 2)
                flow_loss = torch.mean((v_pred - v_target) ** 2)
                x1_pred = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
                endpoint_loss = torch.mean((x1_pred - x1) ** 2)
                if x1.shape[1] > 1:
                    smooth_loss = torch.mean((x1_pred[:, 1:] - x1_pred[:, :-1]) ** 2)
                else:
                    smooth_loss = torch.zeros((), device=device, dtype=x1.dtype)
                loss = (
                    flow_loss
                    + float(cfg.train.lambda_endpoint) * endpoint_loss
                    + float(cfg.train.lambda_smooth) * smooth_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_model.update_parameters(model)

            running_loss += float(loss.item())
            running_flow += float(flow_loss.item())
            running_endpoint += float(endpoint_loss.item())
            running_smooth += float(smooth_loss.item())
            global_step += 1

            if (iteration + 1) % int(cfg.logging.log_every) == 0:
                avg_loss = running_loss / int(cfg.logging.log_every)
                avg_flow = running_flow / int(cfg.logging.log_every)
                avg_endpoint = running_endpoint / int(cfg.logging.log_every)
                avg_smooth = running_smooth / int(cfg.logging.log_every)
                elapsed = max(1e-6, time.time() - loop_start)
                samples_per_second = (int(cfg.train.batch_size) * int(cfg.logging.log_every)) / elapsed
                print(
                    f"epoch={epoch:04d} iter={iteration:05d} step={global_step:07d} "
                    f"loss={avg_loss:.6f} flow={avg_flow:.6f} "
                    f"endpoint={avg_endpoint:.6f} smooth={avg_smooth:.6f} "
                    f"samples_per_sec={samples_per_second:.1f}"
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/flow_loss": avg_flow,
                            "train/endpoint_loss": avg_endpoint,
                            "train/smooth_loss": avg_smooth,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/samples_per_sec": samples_per_second,
                        },
                        step=global_step,
                    )
                running_loss = 0.0
                running_flow = 0.0
                running_endpoint = 0.0
                running_smooth = 0.0
                loop_start = time.time()

        if num_batches == 0:
            raise RuntimeError(
                "DataLoader yielded zero training batches. "
                "Reduce batch_size or change the sampler configuration."
            )

        checkpoint = {
            "model": model.state_dict(),
            "ema_model": ema_model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "model_cfg": OmegaConf.to_container(cfg.flow, resolve=True),
            "camera_names": list(cfg.data.camera_names),
            "env_metadata": dataset.env_metadata,
            "task_metadata_map": dataset.task_metadata_map,
            "task_prompt_map": dataset.task_prompt_map,
            "task_roots": dataset.task_roots,
            "act_mean": dataset.act_mean,
            "act_std": dataset.act_std,
            "prop_mean": dataset.prop_mean,
            "prop_std": dataset.prop_std,
        }

        if (epoch + 1) % int(cfg.checkpoint.save_freq) == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(save_dir, f"flow_multi_ep{epoch + 1:04d}_{timestamp}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}")

    if wandb is not None:
        wandb.finish()
    dataset.close()


if __name__ == "__main__":
    main()
