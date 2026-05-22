"""Benchmark-facing LPB v2 adapters (lazy imports to avoid pulling ``benchmark``)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BCEBenchmarkDiscriminator",
    "LPBV2BenchmarkDiscriminator",
    "TwoBankBenchmarkDiscriminator",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BCEBenchmarkDiscriminator": (".bce", "BCEBenchmarkDiscriminator"),
    "LPBV2BenchmarkDiscriminator": (".single_bank", "LPBV2BenchmarkDiscriminator"),
    "TwoBankBenchmarkDiscriminator": (".two_bank", "TwoBankBenchmarkDiscriminator"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
