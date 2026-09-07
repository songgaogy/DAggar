"""Offline cache and replay support for DSRL."""

from .cache import (
    CACHE_SCHEMA_VERSION,
    CacheDataset,
    CacheValidationError,
    build_cache_fingerprint,
    build_feature_cache,
)
from .legacy import LegacyBuffer, LegacyTransition, load_legacy_buffer
from .prefetch import CudaBatchPrefetcher
from .replay import CompactTransition, OnlineReplay, ReplaySource, UniformReplay
from .warmup import (
    WARMUP_ARRAYS,
    WARMUP_SCHEMA_VERSION,
    WarmupReplay,
    WarmupValidationError,
    build_warmup_fingerprint,
    load_warmup_cache,
    save_warmup_cache,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheDataset",
    "CacheValidationError",
    "CompactTransition",
    "CudaBatchPrefetcher",
    "LegacyBuffer",
    "LegacyTransition",
    "OnlineReplay",
    "ReplaySource",
    "UniformReplay",
    "WARMUP_ARRAYS",
    "WARMUP_SCHEMA_VERSION",
    "WarmupReplay",
    "WarmupValidationError",
    "build_cache_fingerprint",
    "build_feature_cache",
    "build_warmup_fingerprint",
    "load_legacy_buffer",
    "load_warmup_cache",
    "save_warmup_cache",
]
