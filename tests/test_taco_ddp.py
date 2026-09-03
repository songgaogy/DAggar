from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from robosuite.discriminator.dyn_disc.models.taco import TACORepresentationModel


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gather_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        local = torch.full((2, 50), float(rank), device=f"cuda:{rank}")
        gathered, offset = TACORepresentationModel.gather_future_keys(local)
        assert gathered.shape == (4, 50)
        assert offset == rank * 2
        torch.testing.assert_close(gathered[:2], torch.zeros_like(gathered[:2]))
        torch.testing.assert_close(gathered[2:], torch.ones_like(gathered[2:]))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
def test_two_gpu_future_key_gather_and_positive_offset() -> None:
    mp.spawn(_gather_worker, args=(2, _free_port()), nprocs=2, join=True)
