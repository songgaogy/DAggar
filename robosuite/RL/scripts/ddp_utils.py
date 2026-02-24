import os
import torch
import torch.distributed as dist
import numpy as np


def init_distributed():
    """Initialize torch.distributed from torchrun environment variables.

    Returns:
        ddp (bool): whether distributed is initialized
        rank (int): global rank
        local_rank (int): local rank on node
        world_size (int): world size
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        ddp = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        ddp = False
    return ddp, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap(m):
    return m.module if hasattr(m, "module") else m


def to_device(x, device, dtype=None):
    if torch.is_tensor(x):
        if dtype is None:
            return x.to(device=device, non_blocking=True)
        return x.to(device=device, dtype=dtype, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        y = [to_device(v, device, dtype) for v in x]
        return type(x)(y)
    # numpy arrays
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t.to(device=device, non_blocking=True)
    return x
