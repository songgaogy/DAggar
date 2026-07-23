"""Thread-safe console, JSONL, and TensorBoard logging."""

from __future__ import annotations

import json
import math
import sys
import threading
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, TextIO


def _finite_scalar(value: object) -> float | None:
    if isinstance(value, Real):
        scalar = float(value)
    else:
        numel = getattr(value, "numel", None)
        if callable(numel) and int(numel()) != 1:
            return None
        item = getattr(value, "item", None)
        if not callable(item):
            return None
        try:
            scalar = float(item())
        except (TypeError, ValueError):
            return None
    return scalar if math.isfinite(scalar) else None


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
        """Log finite scalar values and return the number written."""
        normalized_prefix = "" if not prefix else f"{prefix.strip('/')}/"
        scalars = [
            (f"{normalized_prefix}{name.strip('/')}", scalar)
            for name, value in metrics.items()
            if (scalar := _finite_scalar(value)) is not None
        ]
        with self._lock:
            if self._writer is None:
                return 0
            for tag, value in scalars:
                self._writer.add_scalar(tag, value, int(step))
        return len(scalars)

    def flush(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.flush()

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _ThreadSafeTee:
    def __init__(self, *streams: TextIO, lock: threading.Lock | None = None) -> None:
        self.streams = streams
        self._lock = lock or threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            for stream in self.streams:
                stream.write(data)
                if "\n" in data or "\r" in data:
                    stream.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            for stream in self.streams:
                stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class ConsoleLogCapture:
    """Mirror stdout and stderr to a line-buffered file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None

    def start(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        shared_lock = threading.Lock()
        sys.stdout = _ThreadSafeTee(self._stdout, self._file, lock=shared_lock)
        sys.stderr = _ThreadSafeTee(self._stderr, self._file, lock=shared_lock)

    def stop(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
            self._stdout = None
        if self._stderr is not None:
            sys.stderr = self._stderr
            self._stderr = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ConsoleLogCapture":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


class JsonlEventLogger:
    """Append structured events to one JSON-lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._file is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def log(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if self._file is None:
                raise RuntimeError("JsonlEventLogger.start() must be called before log().")
            self._file.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "JsonlEventLogger":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
