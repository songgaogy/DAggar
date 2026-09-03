"""Two-GPU CUDA training entry for RPT masked sensorimotor pretraining."""

from __future__ import annotations

import logging
import math
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from robosuite.discriminator.dyn_disc.core.model_loader import CHECKPOINT_VERSION
from robosuite.discriminator.dyn_disc.data.rpt_cache_dataset import RPTCacheDataset
from robosuite.discriminator.dyn_disc.utils.normalize_util import (
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from robosuite.discriminator.dyn_disc.utils.normalizer import LinearNormalizer


log = logging.getLogger(__name__)


def _distributed_context() -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("RPT pretraining requires CUDA; CPU fallback is disabled")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalizer(dataset: RPTCacheDataset, device: torch.device) -> LinearNormalizer:
    proprio_min = dataset.proprio_min.numpy()
    proprio_max = dataset.proprio_max.numpy()
    state_stat = {
        "min": proprio_min,
        "max": proprio_max,
        "mean": (proprio_min + proprio_max) / 2.0,
        "std": np.maximum((proprio_max - proprio_min) / math.sqrt(12.0), 1e-6),
    }
    action_stat = {
        key: value
        for key, value in {
            "min": np.zeros(7, dtype=np.float32),
            "max": np.zeros(7, dtype=np.float32),
            "mean": np.zeros(7, dtype=np.float32),
            "std": np.ones(7, dtype=np.float32),
        }.items()
    }
    normalizer = LinearNormalizer()
    normalizer["state"] = get_range_normalizer_from_stat(state_stat)
    normalizer["act"] = get_identity_normalizer_from_stat(action_stat)
    return normalizer.to(device)


def _mean_across_ranks(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().float().clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result.div_(world_size)
    return result


def _save_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    *,
    epoch: int,
    global_step: int,
    cache_fingerprint: str,
    subdir: str,
) -> Path:
    checkpoint_dir = out_dir / subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pretraining_method": "rpt",
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rpt_model": model.state_dict(),
        "cache_manifest_fingerprint": str(cache_fingerprint),
        "architecture": model.architecture_metadata(),
    }
    path = checkpoint_dir / f"model_{epoch}.pth"
    torch.save(payload, path)
    return path


def _configure_cuda() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    if str(cfg.pretraining_method).lower() != "rpt":
        raise ValueError("This branch only supports pretraining_method=rpt")
    local_rank, rank, world_size = _distributed_context()
    if world_size != 2:
        raise RuntimeError(f"RPT training requires exactly two CUDA ranks, got {world_size}")
    _seed_all(int(cfg.training.seed) + rank)
    _configure_cuda()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    cache_root = Path(hydra.utils.to_absolute_path(str(cfg.cache.root)))
    dataset = RPTCacheDataset(cache_root, context_length=int(cfg.context_length))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(cfg.training.seed),
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        sampler=sampler,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.training.num_workers) > 0,
        prefetch_factor=(
            int(cfg.training.prefetch_factor)
            if int(cfg.training.num_workers) > 0
            else None
        ),
        drop_last=True,
    )
    if not len(loader):
        raise RuntimeError("RPT cache is too small for the configured global batch")

    model = hydra.utils.instantiate(cfg.model).to(device)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.peak_lr),
        weight_decay=float(cfg.training.weight_decay),
        fused=True,
    )
    total_steps = int(cfg.training.epochs) * len(loader)
    warmup_steps = int(cfg.training.warmup_epochs) * len(loader)

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    normalizer = _normalizer(dataset, device)

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    writer = SummaryWriter(str(out_dir / "tensorboard")) if is_main else None
    if is_main:
        OmegaConf.save(cfg, out_dir / "hydra.yaml", resolve=True)
        torch.save(normalizer.state_dict(), out_dir / "normalizer.pth")
        trainable = sum(parameter.numel() for parameter in model.parameters())
        log.info(
            "RPT train samples=%d batches=%d global_batch=%d parameters=%d cache=%s",
            len(dataset), len(loader), int(cfg.training.batch_size) * world_size,
            trainable, cache_root,
        )

    global_step = 0
    log_every = max(1, int(cfg.training.log_every_steps))
    save_every = max(1, int(cfg.training.save_every_x_epoch))
    last_log_step = 0
    last_log_time = time.perf_counter()
    interval_data_time = 0.0
    last_step_end = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(1, int(cfg.training.epochs) + 1):
            sampler.set_epoch(epoch - 1)
            ddp_model.train()
            epoch_sums = {name: torch.zeros((), device=device) for name in ("total", "visual", "proprio", "action", "mask_ratio")}
            for batch in loader:
                interval_data_time += time.perf_counter() - last_step_end
                visual = batch["visual_latents"].to(device, non_blocking=True)
                proprio = normalizer["state"].normalize(
                    batch["proprio"].to(device, non_blocking=True)
                )
                actions = normalizer["act"].normalize(
                    batch["actions"].to(device, non_blocking=True)
                )
                measure = (global_step + 1) % log_every == 0
                if measure:
                    start_event = torch.cuda.Event(enable_timing=True)
                    forward_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, components = ddp_model(visual, proprio, actions)
                if measure:
                    forward_event.record()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if measure:
                    end_event.record()
                for name in epoch_sums:
                    epoch_sums[name].add_(components[name].detach().float())
                global_step += 1

                if measure:
                    torch.cuda.synchronize(device)
                    now = time.perf_counter()
                    elapsed = max(now - last_log_time, 1e-9)
                    steps = global_step - last_log_step
                    if is_main and writer is not None:
                        for name in ("total", "visual", "proprio", "action"):
                            writer.add_scalar(f"loss/{name}", float(_mean_across_ranks(components[name], world_size).item()), global_step)
                        writer.add_scalar("mask/ratio", float(_mean_across_ranks(components["mask_ratio"], world_size).item()), global_step)
                        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                        writer.add_scalar("performance/samples_per_second", int(cfg.training.batch_size) * world_size * steps / elapsed, global_step)
                        writer.add_scalar("performance/steps_per_second", steps / elapsed, global_step)
                        writer.add_scalar("performance/data_time_seconds", interval_data_time / steps, global_step)
                        writer.add_scalar("performance/forward_time_ms", start_event.elapsed_time(forward_event), global_step)
                        writer.add_scalar("performance/backward_optimizer_time_ms", forward_event.elapsed_time(end_event), global_step)
                        writer.add_scalar("memory/allocated_gib", torch.cuda.memory_allocated(device) / 2**30, global_step)
                        writer.add_scalar("memory/reserved_gib", torch.cuda.memory_reserved(device) / 2**30, global_step)
                        writer.add_scalar("memory/peak_allocated_gib", torch.cuda.max_memory_allocated(device) / 2**30, global_step)
                    elif world_size > 1:
                        # All ranks must participate in metric reductions.
                        for name in ("total", "visual", "proprio", "action", "mask_ratio"):
                            _mean_across_ranks(components[name], world_size)
                    last_log_time = now
                    last_log_step = global_step
                    interval_data_time = 0.0
                last_step_end = time.perf_counter()

            means = {
                name: float(_mean_across_ranks(value / len(loader), world_size).item())
                for name, value in epoch_sums.items()
            }
            if is_main:
                log.info(
                    "epoch=%d loss=%.6f visual=%.6f proprio=%.6f action=%.6f mask=%.4f",
                    epoch, means["total"], means["visual"], means["proprio"],
                    means["action"], means["mask_ratio"],
                )
                if writer is not None:
                    for name, value in means.items():
                        writer.add_scalar(f"epoch/{name}", value, epoch)
                if epoch % save_every == 0 or epoch == int(cfg.training.epochs):
                    path = _save_checkpoint(
                        out_dir,
                        model,
                        epoch=epoch,
                        global_step=global_step,
                        cache_fingerprint=dataset.fingerprint,
                        subdir=str(cfg.training.checkpoint_subdir),
                    )
                    log.info("Saved RPT checkpoint to %s", path)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
