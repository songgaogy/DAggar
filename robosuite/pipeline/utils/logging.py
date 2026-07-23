"""Thread-safe TensorBoard scalar logging for the HIL-SERL pipeline."""

from __future__ import annotations

import json
import math
import sys
import threading
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


def _scalar(value: object) -> float | None:
    if isinstance(value, Real):
        result = float(value)
    else:
        numel = getattr(value, "numel", None)
        if callable(numel) and int(numel()) != 1:
            return None
        item = getattr(value, "item", None)
        if not callable(item):
            return None
        try:
            result = float(item())
        except (TypeError, ValueError):
            return None
    return result if math.isfinite(result) else None


class TensorBoardLogger:
    """Write finite scalar metrics to TensorBoard."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        enabled: bool = True,
        flush_secs: int = 10,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir)
        self._lock = threading.Lock()
        self._writer = None
        if self.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=int(flush_secs),
            )

    def log(
        self,
        metrics: Mapping[str, object],
        *,
        step: int,
        prefix: str | None = None,
    ) -> int:
        """Log all finite scalar values and return the number written."""
        if self._writer is None:
            return 0
        normalized_prefix = "" if not prefix else f"{prefix.strip('/')}/"
        scalars = [
            (f"{normalized_prefix}{name.strip('/')}", scalar)
            for name, value in metrics.items()
            if (scalar := _scalar(value)) is not None
        ]
        with self._lock:
            for tag, value in scalars:
                self._writer.add_scalar(tag, value, int(step))
        return len(scalars)

    def flush(self) -> None:
        if self._writer is not None:
            with self._lock:
                self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            with self._lock:
                self._writer.close()
                self._writer = None

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            if "\n" in data or "\r" in data:
                stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class ConsoleLogCapture:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._stdout = None
        self._stderr = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._stdout, self._file)
        sys.stderr = _Tee(self._stderr, self._file)

    def stop(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.close()
            self._file = None


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def log(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._file is not None:
                self._file.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
