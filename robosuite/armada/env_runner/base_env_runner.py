from omegaconf import DictConfig
from typing import List
import torch
import torch.distributed as dist


class BaseEnvRunner:
    def __init__(
            self, 
            cfg: DictConfig,
            rank: int,
            device_ids: List[int]
        ):
        self.cfg = cfg
        self.rank = rank
        self.device_ids = device_ids
        self.world_size = len(device_ids)
        self.device_id = device_ids[rank]
        self.device = f"cuda:{self.device_id}"
        self.fps = cfg.train.task.env.max_fr

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=self.world_size,
                init_method="env://",
            )

    def run_rollout(self):
        raise NotImplementedError()