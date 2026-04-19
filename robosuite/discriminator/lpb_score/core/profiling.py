"""Lightweight wall-clock profiling helpers for training bottleneck diagnosis."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch


class StepProfiler:
    def __init__(
        self,
        *,
        enabled: bool = False,
        sync_cuda: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda) and torch.cuda.is_available()
        self._timings_s: dict[str, float] = defaultdict(float)

    def _sync(self) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._timings_s[str(name)] += time.perf_counter() - start

    def snapshot_ms(self) -> dict[str, float]:
        return {key: float(value * 1000.0) for key, value in self._timings_s.items()}
