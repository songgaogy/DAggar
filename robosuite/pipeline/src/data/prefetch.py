from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from robosuite.pipeline.utils.tensor import resolve_cuda_device

from .replay import UniformReplay


class CudaBatchPrefetcher:
    """Gather replay batches in a thread and asynchronously copy them to CUDA."""

    def __init__(
        self,
        replay: UniformReplay,
        *,
        batch_size: int,
        device: str | torch.device,
        depth: int = 2,
    ) -> None:
        if depth <= 0:
            raise ValueError("Prefetch depth must be positive.")
        self.replay = replay
        self.batch_size = int(batch_size)
        self.device = resolve_cuda_device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=int(depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="dsrl-replay-prefetch", daemon=True)
        self._thread.start()
        self._staged = self._stage()

    @staticmethod
    def _pin(batch: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(value)).pin_memory()
            for key, value in batch.items()
        }

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                pinned = self._pin(self.replay.sample(self.batch_size))
                while not self._stop.is_set():
                    try:
                        self._queue.put(pinned, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:
            while not self._stop.is_set():
                try:
                    self._queue.put(error, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _stage(self) -> dict[str, torch.Tensor]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise RuntimeError("Replay prefetch worker failed.") from item
        with torch.cuda.stream(self.stream):
            return {
                key: value.to(self.device, non_blocking=True)
                for key, value in item.items()
            }

    def next(self) -> dict[str, torch.Tensor]:
        batch = self._staged
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        for value in batch.values():
            value.record_stream(torch.cuda.current_stream(self.device))
        self._staged = self._stage()
        return batch

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "CudaBatchPrefetcher":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
