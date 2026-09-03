"""CUDA training entry for TACO InfoNCE representation pretraining."""

from __future__ import annotations

import logging
import os
import random
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from robosuite.discriminator.dyn_disc.core.model_loader import instantiate_local

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_paths(data_path):
    """Resolve dataset paths against Hydra launch cwd (safe when job.chdir=true)."""
    if isinstance(data_path, (list, tuple)):
        return [hydra.utils.to_absolute_path(str(p)) for p in data_path]
    return hydra.utils.to_absolute_path(str(data_path))


def _instantiate_dataset(cfg: DictConfig, train: bool):
    """Hydra-instantiate the dataset class named in cfg.env.dataset_class."""
    DatasetCls = hydra.utils.get_class(cfg.env.dataset_class)
    data_path = cfg.env.train_data_path if train else cfg.env.val_data_path
    if isinstance(data_path, ListConfig):
        data_path = OmegaConf.to_container(data_path, resolve=True)
    data_path = _resolve_data_paths(data_path)
    # HDF5 dataset contract:
    #   __getitem__ -> (obs, act, state)
    #     obs["visual"][view]: (F, 3, H, W) float in [0, 1]
    #     obs["proprio"]:      (F, P) float
    #     act:                (F * frameskip, A) float
    # where F = num_hist + num_pred.
    kwargs = dict(
        zarr_path=data_path,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
        view_names=list(cfg.env.view_names),
        abs_action=cfg.abs_action,
        use_crop=cfg.use_crop,
        train=train,
        original_img_size=cfg.env.original_img_size,
        cropped_img_size=cfg.env.cropped_img_size,
        action_dim=cfg.env.action_dim,
    )
    if "use_cache" in cfg.env:
        kwargs["use_cache"] = bool(cfg.env.use_cache)
    if "cache_dir" in cfg.env and cfg.env.cache_dir is not None:
        kwargs["cache_dir"] = hydra.utils.to_absolute_path(str(cfg.env.cache_dir))
    if cfg.get("proprio_indices", None):
        kwargs["proprio_indices"] = list(cfg.proprio_indices)
    if cfg.get("proprio_map", None):
        kwargs["proprio_map"] = OmegaConf.to_container(cfg.proprio_map, resolve=True)
    if cfg.get("max_trajectories", None):
        kwargs["max_trajectories"] = int(cfg.max_trajectories)
    if cfg.get("causal_action_chunks", None) is not None:
        kwargs["causal_action_chunks"] = bool(cfg.causal_action_chunks)
    if cfg.get("return_uint8_images", None) is not None:
        kwargs["return_uint8_images"] = bool(cfg.return_uint8_images)
    if cfg.get("view_frame_counts", None) is not None:
        kwargs["view_frame_counts"] = OmegaConf.to_container(
            cfg.view_frame_counts,
            resolve=True,
        )
    return DatasetCls(**kwargs)


def _instantiate_encoder(cfg: DictConfig):
    encoder_cfg = getattr(cfg, "encoder", None)
    if encoder_cfg is None:
        raise ValueError(
            "dyn_disc is DINOv3-only: cfg.encoder must be set "
            "(e.g. config/encoder/dinov3.yaml). No ResNet fallback is provided."
        )
    return instantiate_local(encoder_cfg, view_names=list(cfg.view_names))


def _build_model(cfg: DictConfig, dataset, device: torch.device):
    """Build the TACO representation model."""
    encoder = _instantiate_encoder(cfg)
    train_encoder_flag = bool(getattr(cfg.model, "train_encoder", False))
    if train_encoder_flag:
        raise ValueError("TACO pretraining requires a frozen DINOv3 backbone.")
    if hasattr(encoder, "set_trainable"):
        train_projection = bool(getattr(cfg.encoder, "train_projection", True))
        encoder.set_trainable(train_backbone=train_encoder_flag, train_projection=train_projection)
    else:
        for p in encoder.parameters():
            p.requires_grad = train_encoder_flag
    log.info(
        "encoder: frozen DINOv3 backbone with trainable projection=%s",
        bool(getattr(cfg.encoder, "train_projection", True)),
    )

    proprio_encoder = instantiate_local(
        cfg.proprio_encoder,
        in_chans=dataset.proprio_dim,
        emb_dim=cfg.env.proprio_emb_dim,
    )
    source_view_names = list(getattr(cfg, "source_view_names", cfg.view_names))
    target_view_names = list(getattr(cfg, "target_view_names", cfg.view_names))

    model = instantiate_local(
        cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        proprio_dim=cfg.env.proprio_emb_dim,
        action_dim_per_step=cfg.env.action_dim,
        frameskip=cfg.frameskip,
        view_names=list(cfg.view_names),
        source_view_names=source_view_names,
        target_view_names=target_view_names,
    )
    configured_action_emb = int(cfg.env.action_emb_dim)
    actual_action_emb = int(model.action_encoder.emb_dim)
    if configured_action_emb != actual_action_emb:
        raise ValueError(
            "env.action_emb_dim must equal action_step_emb_dim * frameskip; "
            f"got env.action_emb_dim={configured_action_emb}, "
            f"action encoder emb_dim={actual_action_emb}"
        )
    log.info(
        "TACO capacity: encoder.emb_dim=%s state_dim=%s proprio_emb_dim=%s "
        "action_emb_dim=%s transition_hidden=%s",
        int(encoder.emb_dim),
        int(model.state_dim),
        int(cfg.env.proprio_emb_dim),
        actual_action_emb,
        int(model.transition_hidden),
    )
    model = model.to(device)
    return model, encoder, proprio_encoder, model.action_encoder


def _prepare_action_chunks(act: torch.Tensor, num_frames: int, frameskip: int) -> torch.Tensor:
    if act.dim() == 3 and act.shape[1] == num_frames and frameskip > 1:
        return act
    if act.dim() != 3:
        raise ValueError(f"Expected action tensor with shape (B, F*frameskip, A), got {tuple(act.shape)}")
    b, raw_t, action_dim = act.shape
    expected = int(num_frames) * int(frameskip)
    if raw_t < expected:
        pad = act[:, -1:, :].expand(b, expected - raw_t, action_dim)
        act = torch.cat([act, pad], dim=1)
    elif raw_t > expected:
        act = act[:, :expected, :]
    return act.reshape(b, int(num_frames), int(frameskip) * action_dim)


def _normalize_batch(
    batch,
    normalizer,
    view_names,
    device,
    num_frames: int,
    frameskip: int,
    normalize_images: bool,
):
    obs, act, _state = batch
    visual = obs["visual"]
    for v in view_names:
        x = visual[v]
        x = x.to(device, non_blocking=True)
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div_(255.0)
        if normalize_images:
            # x: (B, num_frames, 3, H, W)
            B, F, C, H, W = x.shape
            x_flat = x.view(B * F, C, H, W)
            x_flat = normalizer[v].normalize(x_flat)
            x = x_flat.view(B, F, C, H, W)
        visual[v] = x
    proprio = obs["proprio"].to(device, non_blocking=True)
    proprio = normalizer["state"].normalize(proprio)
    act = act.to(device, non_blocking=True)
    act = normalizer["act"].normalize(act)
    act = _prepare_action_chunks(act, num_frames=num_frames, frameskip=frameskip)
    return {"visual": visual, "proprio": proprio}, act


def _save_ckpt(out_dir: Path, epoch: int, parts: dict, ckpt_subdir: str = "checkpoints") -> Path:
    ckpt_dir = out_dir / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch}
    for k, v in parts.items():
        if isinstance(v, nn.Module):
            payload[k] = v.state_dict()
        else:
            payload[k] = v
    fp = ckpt_dir / f"model_{epoch}.pth"
    torch.save(payload, fp)
    return fp


def _count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _log_trainable_parameters(model: nn.Module, **named_modules: nn.Module) -> None:
    lines = []
    subtotal = 0
    for name, module in named_modules.items():
        n = _count_trainable_params(module)
        subtotal += n
        lines.append(f"  {name}: {_format_param_count(n)} ({n:,})")
    total = _count_trainable_params(model)
    other = total - subtotal
    if other > 0:
        lines.append(f"  other (norms, etc.): {_format_param_count(other)} ({other:,})")
    lines.append(f"  total: {_format_param_count(total)} ({total:,})")
    log.info("Trainable parameters:\n%s", "\n".join(lines))


def _distributed_context() -> tuple[bool, int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("dyn_disc representation pretraining requires CUDA; CPU fallback is disabled.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        torch.cuda.set_device(0)
        return False, 0, 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True, local_rank, rank, world_size


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().float().clone()
    if int(world_size) > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(int(world_size))
    return value


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    if str(cfg.pretraining_method).lower() != "taco":
        raise ValueError("This branch only supports pretraining_method=taco.")
    is_dist, local_rank, rank, world_size = _distributed_context()
    is_main = rank == 0
    _seed_all(int(cfg.training.seed) + rank)

    device = torch.device("cuda", local_rank if is_dist else 0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    if is_main:
        log.info(f"Output dir: {out_dir}")

    # Hydra @main does not write hydra.yaml by default; do it ourselves so the
    # discriminator can pick the saved config up.
    train_ds = _instantiate_dataset(cfg, train=True)
    if is_main:
        log.info(
            f"train: {len(train_ds)} samples; "
            f"proprio_dim={train_ds.proprio_dim}, action_dim={train_ds.action_dim}"
        )

    with open_dict(cfg):
        cfg.train_data_path = cfg.env.train_data_path
        cfg.val_data_path = cfg.env.val_data_path
        cfg.prior_in_chans = int(train_ds.proprio_dim)
        cfg.action_dim_per_step = int(cfg.env.action_dim)
    if is_main:
        OmegaConf.save(cfg, out_dir / "hydra.yaml", resolve=True)

    normalizer = train_ds.get_normalizer().to(device)
    # Persist normalizer alongside the checkpoint so benchmark/KNN encoding matches training.
    if is_main:
        torch.save(normalizer.state_dict(), out_dir / "normalizer.pth")

    sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if is_dist else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=(sampler is None),
        num_workers=cfg.training.num_workers, drop_last=True,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=int(cfg.training.num_workers) > 0,
        prefetch_factor=(
            int(OmegaConf.select(cfg, "training.prefetch_factor", default=4))
            if int(cfg.training.num_workers) > 0 else None
        ),
    )

    model, encoder, proprio_encoder, action_encoder = _build_model(cfg, train_ds, device)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        static_graph=True,
    ) if is_dist else model

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(cfg.training.encoder_lr),
        weight_decay=0.0,
        fused=True,
    )

    if is_main:
        _log_trainable_parameters(
            model,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
        )

    save_every = max(1, int(cfg.training.save_every_x_epoch))
    ckpt_subdir = str(OmegaConf.select(cfg, "training.checkpoint_subdir", default="checkpoints"))
    ckpt_epoch_offset = int(OmegaConf.select(cfg, "training.checkpoint_epoch_offset", default=0))
    log_every = max(1, int(OmegaConf.select(cfg, "training.log_every_steps", default=50)))
    view_names = list(cfg.view_names)
    writer = SummaryWriter(str(out_dir / "tensorboard")) if is_main else None
    global_step = 0
    last_logged_step = 0
    last_log_time = time.perf_counter()
    interval_data_time = 0.0
    last_step_end = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(int(cfg.training.epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            ddp_model.train()
            epoch_loss = torch.zeros((), device=device, dtype=torch.float32)
            n_batches = 0
            for batch in train_loader:
                interval_data_time += time.perf_counter() - last_step_end
                obs, act = _normalize_batch(
                    batch,
                    normalizer,
                    view_names,
                    device,
                    num_frames=int(cfg.num_hist) + int(cfg.num_pred),
                    frameskip=int(cfg.frameskip),
                    normalize_images=not bool(getattr(model.encoder, "normalizes_images", False)),
                )
                optimizer.zero_grad(set_to_none=True)
                measure_step = (global_step + 1) % log_every == 0
                if measure_step:
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)
                    backward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    loss, _components = ddp_model(obs, act)
                if measure_step:
                    forward_end.record()
                loss.backward()
                optimizer.step()
                if measure_step:
                    backward_end.record()
                epoch_loss.add_(loss.detach().float())
                n_batches += 1
                global_step += 1

                if measure_step:
                    torch.cuda.synchronize(device)
                    now = time.perf_counter()
                    elapsed = max(now - last_log_time, 1e-9)
                    logged_steps = global_step - last_logged_step
                    mean_loss = _distributed_mean(loss, world_size)
                    if is_main and writer is not None:
                        global_batch = int(cfg.training.batch_size) * int(world_size)
                        writer.add_scalar("train/taco_loss", float(mean_loss.item()), global_step)
                        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                        writer.add_scalar("performance/samples_per_second", global_batch * logged_steps / elapsed, global_step)
                        writer.add_scalar("performance/steps_per_second", logged_steps / elapsed, global_step)
                        writer.add_scalar("performance/data_time_seconds", interval_data_time / logged_steps, global_step)
                        writer.add_scalar("performance/forward_time_ms", forward_start.elapsed_time(forward_end), global_step)
                        writer.add_scalar("performance/backward_optimizer_time_ms", forward_end.elapsed_time(backward_end), global_step)
                        writer.add_scalar("performance/step_time_ms", forward_start.elapsed_time(backward_end), global_step)
                        writer.add_scalar("performance/global_batch_size", global_batch, global_step)
                        writer.add_scalar("performance/negative_count", global_batch - 1, global_step)
                        writer.add_scalar("memory/allocated_gib", torch.cuda.memory_allocated(device) / 2**30, global_step)
                        writer.add_scalar("memory/reserved_gib", torch.cuda.memory_reserved(device) / 2**30, global_step)
                        writer.add_scalar("memory/peak_allocated_gib", torch.cuda.max_memory_allocated(device) / 2**30, global_step)
                    last_log_time = now
                    last_logged_step = global_step
                    interval_data_time = 0.0
                last_step_end = time.perf_counter()

            avg_train_tensor = _distributed_mean(epoch_loss / max(1, n_batches), world_size)
            avg_train = float(avg_train_tensor.item())
            if is_main:
                log.info(f"epoch={epoch} train_loss={avg_train:.5f}")
                if writer is not None:
                    writer.add_scalar("epoch/train_loss", avg_train, epoch + 1)

            if is_main and ((epoch + 1) % save_every == 0 or epoch == int(cfg.training.epochs) - 1):
                parts = {"pretraining_method": "taco", "model": model}
                fp = _save_ckpt(out_dir, epoch + ckpt_epoch_offset, parts, ckpt_subdir=ckpt_subdir)
                log.info(f"saved checkpoint to {fp}")
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        _cleanup_distributed()


if __name__ == "__main__":
    main()
