"""Per-transition frozen observation-encoding cache.

The frozen dynamics encoder splits into an expensive action-free observation
forward (``SharedDynamicsEncoder.encode_obs_batch``) and a cheap action fusion
(``chunk_from_obs_enc``).  VAST windows are stride-1 overlapping, so the same
frame is re-encoded ``action_horizon`` times per pass, and every round re-encodes
the whole static warmup buffer from scratch.

This module caches the observation encoding once per source frame.  Reuse is
exact: the encoder is frozen and evaluated under ``no_grad``, so a cached row is
bit-identical to a live forward.  Rows that carry no stable provenance key
(synthetic absorbing tails, legacy data) are never cached and fall back to live
encoding, which is also what makes runs without a cache work unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


logger = logging.getLogger(__name__)

OBS_ENCODING_CACHE_SCHEMA_VERSION = 1

FIELD_OBS = "obs"
FIELD_NEXT_OBS = "next_obs"

WARMUP_STREAM = "warmup"


# --------------------------------------------------------------------------- #
# Keys                                                                          #
# --------------------------------------------------------------------------- #


def transition_stream_and_key(transition: Any, field: str) -> tuple[str, str] | None:
    """Return ``(stream, row_key)`` for a transition frame, or ``None``.

    ``None`` means the frame has no stable identity (synthetic tails, legacy
    payloads) and must be encoded live every time.
    """
    if field not in (FIELD_OBS, FIELD_NEXT_OBS):
        raise ValueError(f"field must be {FIELD_OBS!r} or {FIELD_NEXT_OBS!r}, got {field!r}")
    info = getattr(transition, "info", None) or {}
    # Frames synthesized by clone_with_vast_absorbing_tails duplicate a source
    # frame but carry VAST-only bookkeeping; keep them out of the cache.
    if info.get("synthetic_vast_success_tail") or info.get("synthetic_vast_failure_tail"):
        return None

    warmup_row = info.get("warmup_row")
    if warmup_row is not None:
        return WARMUP_STREAM, f"w{int(warmup_row)}:{field}"

    source_round = info.get("source_round")
    episode_index = info.get("round_episode_index")
    frame_index = info.get("source_frame_index")
    if source_round is None or episode_index is None or frame_index is None:
        return None
    return (
        f"round_{int(source_round):03d}",
        f"r{int(source_round)}:e{int(episode_index)}:f{int(frame_index)}:{field}",
    )


def frame_digest(obs: Mapping[str, Any], camera_names: Sequence[str]) -> int:
    """Stable 64-bit content digest of one observation frame."""
    hasher = hashlib.blake2b(digest_size=8)
    for camera in camera_names:
        array = np.ascontiguousarray(obs[camera])
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(memoryview(array).cast("B"))
    state = np.ascontiguousarray(np.asarray(obs["state"], dtype=np.float32))
    hasher.update(memoryview(state).cast("B"))
    return int.from_bytes(hasher.digest(), "little", signed=False)


# --------------------------------------------------------------------------- #
# Config / storage                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObsEncodingCacheConfig:
    enabled: bool = True
    verify_content_hash: bool = True
    encode_batch_size: int = 64
    round_cache_dir: str | None = None
    warmup_cache_dir: str | None = None


@dataclass
class _StreamStore:
    stream: str
    stream_key: str
    path: Path | None
    key_to_row: dict[str, int] = field(default_factory=dict)
    visual: torch.Tensor | None = None       # (N, P, D_v) float32, cpu
    proprio: torch.Tensor | None = None      # (N, D_p) float32, cpu
    digests: torch.Tensor | None = None      # (N,) int64 (bit-cast uint64)
    dirty: bool = False

    def __len__(self) -> int:
        return 0 if self.visual is None else int(self.visual.shape[0])


def encode_in_fixed_batches(
    encode_fn: Callable[[Sequence[int]], dict[str, torch.Tensor]],
    num_rows: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Encode ``num_rows`` rows using batches of exactly ``batch_size``.

    The frozen encoder is deterministic at a fixed batch size but NOT
    batch-invariant: cuBLAS picks a different reduction order per GEMM shape, so
    the same frame encoded at batch 1 differs from batch 64 by ~5e-3 relative
    (~2e-6 between batches of 24 and above).  Padding the trailing partial batch
    up to ``batch_size`` and discarding the padding makes every row see the same
    shape, so a row's encoding no longer depends on the dataset size or on how
    rows were grouped into requests.
    """
    if num_rows <= 0:
        raise ValueError("encode_in_fixed_batches requires at least one row.")
    size = max(1, int(batch_size))
    visual_parts: list[torch.Tensor] = []
    proprio_parts: list[torch.Tensor] = []
    for start in range(0, num_rows, size):
        rows = list(range(start, min(start + size, num_rows)))
        real = len(rows)
        if real < size:
            rows = rows + [rows[-1]] * (size - real)
        encoded = encode_fn(rows)
        visual = encoded["visual"]
        proprio = encoded["proprio"]
        if visual.ndim == 4 and visual.shape[1] == 1:
            visual = visual.squeeze(1)
        if proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio.squeeze(1)
        visual_parts.append(visual[:real])
        proprio_parts.append(proprio[:real])
    return {
        "visual": torch.cat(visual_parts, dim=0),
        "proprio": torch.cat(proprio_parts, dim=0),
    }


# --------------------------------------------------------------------------- #
# Cache                                                                         #
# --------------------------------------------------------------------------- #


class ObsEncodingCache:
    """Disk-backed store of frozen observation encodings, keyed per source frame."""

    def __init__(
        self,
        *,
        encoder: Any,
        config: ObsEncodingCacheConfig,
        camera_names: Sequence[str],
        image_size: int,
        stream_keys: Mapping[str, str] | None = None,
    ) -> None:
        self._encoder = encoder
        self._config = config
        self._camera_names = [str(name) for name in camera_names]
        self._image_size = int(image_size)
        self._spec = dict(encoder.obs_encoding_spec)
        self._stream_keys = {str(k): str(v) for k, v in (stream_keys or {}).items()}
        self._streams: dict[str, _StreamStore] = {}
        self.stats = {
            "hits": 0,
            "encoded": 0,
            "uncacheable": 0,
            "digest_mismatch": 0,
            "loaded_rows": 0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    def register_stream_key(self, stream: str, stream_key: str) -> None:
        """Bind a stream's content identity before its shard is first touched."""
        if stream in self._streams:
            raise RuntimeError(f"Stream {stream!r} was already opened; set its key earlier.")
        self._stream_keys[str(stream)] = str(stream_key)

    # -- shard paths ------------------------------------------------------ #

    def _shard_path(self, stream: str) -> Path | None:
        key = self._spec["encoder_key"][:16]
        if stream == WARMUP_STREAM:
            root = self._config.warmup_cache_dir
            return None if root is None else Path(root).expanduser() / f"obs_encoding-{key}.pt"
        root = self._config.round_cache_dir
        return None if root is None else Path(root).expanduser() / f"{stream}-{key}.pt"

    # -- header ----------------------------------------------------------- #

    def _header(self, stream: str) -> dict[str, Any]:
        return {
            "schema_version": OBS_ENCODING_CACHE_SCHEMA_VERSION,
            "stream": stream,
            "stream_key": self._stream_keys.get(stream, ""),
            "camera_names": list(self._camera_names),
            "image_size": int(self._image_size),
            **{k: v for k, v in self._spec.items() if k != "encoder_checkpoint"},
        }

    def _header_matches(self, cached: Mapping[str, Any], stream: str) -> tuple[bool, str]:
        expected = self._header(stream)
        for key, value in expected.items():
            got = cached.get(key)
            if isinstance(value, list):
                got = list(got) if got is not None else None
            if got != value:
                return False, f"{key}: cached={got!r} expected={value!r}"
        return True, ""

    # -- load / save ------------------------------------------------------ #

    def _stream_store(self, stream: str) -> _StreamStore:
        store = self._streams.get(stream)
        if store is not None:
            return store
        store = _StreamStore(
            stream=stream,
            stream_key=self._stream_keys.get(stream, ""),
            path=self._shard_path(stream),
        )
        if self.enabled and store.path is not None and store.path.exists():
            self._load_shard(store)
        self._streams[stream] = store
        return store

    def _load_shard(self, store: _StreamStore) -> None:
        assert store.path is not None
        try:
            payload = torch.load(store.path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - corrupt shard
            logger.warning(
                "[obs-cache] failed to read %s (%s); rebuilding this stream.", store.path, exc
            )
            return
        ok, reason = self._header_matches(payload.get("header", {}), store.stream)
        if not ok:
            logger.warning(
                "[obs-cache] stale shard %s (%s); rebuilding this stream.", store.path, reason
            )
            return
        keys = list(payload["keys"])
        store.key_to_row = {str(key): row for row, key in enumerate(keys)}
        store.visual = payload["visual"]
        store.proprio = payload["proprio"]
        store.digests = payload.get("digests")
        store.dirty = False
        self.stats["loaded_rows"] += len(keys)
        logger.info(
            "[obs-cache] loaded %d rows for stream %r from %s (%.2f GB)",
            len(keys),
            store.stream,
            store.path,
            (store.visual.numel() + store.proprio.numel()) * 4 / 1e9,
        )

    def flush(self) -> None:
        """Atomically persist every dirty stream shard."""
        if not self.enabled:
            return
        for store in self._streams.values():
            if not store.dirty or store.path is None or store.visual is None:
                continue
            store.path.parent.mkdir(parents=True, exist_ok=True)
            keys = [""] * len(store.key_to_row)
            for key, row in store.key_to_row.items():
                keys[row] = key
            payload = {
                "header": self._header(store.stream),
                "keys": keys,
                "visual": store.visual,
                "proprio": store.proprio,
                "digests": store.digests,
            }
            temporary = store.path.with_name(f".{store.path.name}.tmp-{os.getpid()}")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, store.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            store.dirty = False
            logger.info(
                "[obs-cache] wrote %d rows for stream %r to %s (%.2f GB)",
                len(keys),
                store.stream,
                store.path,
                (store.visual.numel() + store.proprio.numel()) * 4 / 1e9,
            )

    # -- growth ----------------------------------------------------------- #

    def _append(
        self,
        store: _StreamStore,
        keys: Sequence[str],
        visual: torch.Tensor,
        proprio: torch.Tensor,
        digests: Sequence[int],
    ) -> None:
        visual = visual.detach().to("cpu", torch.float32).contiguous()
        proprio = proprio.detach().to("cpu", torch.float32).contiguous()
        digest_tensor = torch.tensor(
            [int(np.int64(np.uint64(d))) for d in digests], dtype=torch.int64
        )
        if store.visual is None:
            store.visual = visual
            store.proprio = proprio
            store.digests = digest_tensor
            base = 0
        else:
            base = int(store.visual.shape[0])
            store.visual = torch.cat([store.visual, visual], dim=0)
            store.proprio = torch.cat([store.proprio, proprio], dim=0)
            store.digests = torch.cat([store.digests, digest_tensor], dim=0)
        for offset, key in enumerate(keys):
            store.key_to_row[str(key)] = base + offset
        store.dirty = True

    # -- main API --------------------------------------------------------- #

    @staticmethod
    def _obs_of(item: tuple[Any, str]) -> Mapping[str, Any]:
        transition, field = item
        return transition.obs if field == FIELD_OBS else transition.next_obs

    def gather(
        self,
        items: Sequence[tuple[Any, str]],
        *,
        encode_missing: Callable[[Sequence[int]], dict[str, torch.Tensor]],
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return the observation encoding for ``items`` as one ``enc`` dict.

        ``items`` are ``(transition, field)`` pairs.  ``encode_missing`` is called
        at most once, with the local indices whose encodings are not cached; it
        must return an ``enc`` dict for exactly those rows, in that order.
        """
        total = len(items)
        if total == 0:
            raise ValueError("ObsEncodingCache.gather requires at least one item.")

        placement: list[tuple[_StreamStore, int] | None] = [None] * total
        missing_local: list[int] = []
        missing_stream: list[tuple[str, str] | None] = []

        if self.enabled:
            for local, item in enumerate(items):
                resolved = transition_stream_and_key(*item)
                if resolved is None:
                    self.stats["uncacheable"] += 1
                    missing_local.append(local)
                    missing_stream.append(None)
                    continue
                stream, key = resolved
                store = self._stream_store(stream)
                row = store.key_to_row.get(key)
                if row is None:
                    missing_local.append(local)
                    missing_stream.append((stream, key))
                else:
                    placement[local] = (store, row)
                    self.stats["hits"] += 1
        else:
            missing_local = list(range(total))
            missing_stream = [None] * total

        target = torch.device(device)
        visual_out: torch.Tensor | None = None
        proprio_out: torch.Tensor | None = None

        if missing_local:
            # Overlapping windows request the same frame many times per call, so
            # dedupe by key first; then run every forward at exactly
            # `encode_batch_size` (see encode_in_fixed_batches) so a row's
            # encoding does not depend on how rows were grouped.
            unique_offsets: list[int] = []       # -> offset into missing_local
            slot_of_missing: list[int] = []      # missing offset -> encoded row
            first_slot: dict[tuple[str, str], int] = {}
            for offset, spec in enumerate(missing_stream):
                if spec is None:
                    slot_of_missing.append(len(unique_offsets))
                    unique_offsets.append(offset)
                    continue
                seen = first_slot.get(spec)
                if seen is None:
                    first_slot[spec] = len(unique_offsets)
                    slot_of_missing.append(len(unique_offsets))
                    unique_offsets.append(offset)
                else:
                    slot_of_missing.append(seen)

            def _encode_unique(rows: Sequence[int]) -> dict[str, torch.Tensor]:
                return encode_missing([missing_local[unique_offsets[r]] for r in rows])

            encoded = encode_in_fixed_batches(
                _encode_unique,
                len(unique_offsets),
                int(self._config.encode_batch_size),
            )
            enc_visual = encoded["visual"]
            enc_proprio = encoded["proprio"]
            if int(enc_visual.shape[0]) != len(unique_offsets):
                raise RuntimeError(
                    "encode_missing returned "
                    f"{int(enc_visual.shape[0])} rows for {len(unique_offsets)} requests."
                )
            self.stats["encoded"] += len(unique_offsets)
            visual_out = torch.empty(
                (total, *enc_visual.shape[1:]), dtype=enc_visual.dtype, device=target
            )
            proprio_out = torch.empty(
                (total, *enc_proprio.shape[1:]), dtype=enc_proprio.dtype, device=target
            )
            miss_idx = torch.tensor(missing_local, dtype=torch.long, device=target)
            slot_idx = torch.tensor(
                slot_of_missing, dtype=torch.long, device=enc_visual.device
            )
            visual_out.index_copy_(
                0, miss_idx, enc_visual.index_select(0, slot_idx).to(target)
            )
            proprio_out.index_copy_(
                0, miss_idx, enc_proprio.index_select(0, slot_idx).to(target)
            )

            by_stream: dict[str, list[tuple[int, str]]] = {}
            for spec, slot in first_slot.items():
                stream, key = spec
                by_stream.setdefault(stream, []).append((slot, key))
            for stream, entries in by_stream.items():
                store = self._stream_store(stream)
                slots = [slot for slot, _ in entries]
                keys = [key for _, key in entries]
                digests = [
                    frame_digest(
                        self._obs_of(items[missing_local[unique_offsets[slot]]]),
                        self._camera_names,
                    )
                    for slot in slots
                ]
                sel = torch.tensor(slots, dtype=torch.long, device=enc_visual.device)
                self._append(
                    store,
                    keys,
                    enc_visual.index_select(0, sel),
                    enc_proprio.index_select(0, sel),
                    digests,
                )

        if visual_out is None:
            store, _row = placement[0]  # type: ignore[misc]
            assert store.visual is not None and store.proprio is not None
            visual_out = torch.empty(
                (total, *store.visual.shape[1:]), dtype=store.visual.dtype, device=target
            )
            proprio_out = torch.empty(
                (total, *store.proprio.shape[1:]), dtype=store.proprio.dtype, device=target
            )

        assert proprio_out is not None
        hit_by_store: dict[int, tuple[_StreamStore, list[int], list[int]]] = {}
        for local, slot in enumerate(placement):
            if slot is None:
                continue
            store, row = slot
            bucket = hit_by_store.setdefault(id(store), (store, [], []))
            bucket[1].append(local)
            bucket[2].append(row)
        for store, locals_, rows in hit_by_store.values():
            assert store.visual is not None and store.proprio is not None
            rows_t = torch.tensor(rows, dtype=torch.long)
            locals_t = torch.tensor(locals_, dtype=torch.long, device=target)
            visual_out.index_copy_(
                0, locals_t, store.visual.index_select(0, rows_t).to(target)
            )
            proprio_out.index_copy_(
                0, locals_t, store.proprio.index_select(0, rows_t).to(target)
            )

        return {"visual": visual_out.unsqueeze(1), "proprio": proprio_out.unsqueeze(1)}

    # -- verification ----------------------------------------------------- #

    def verify_and_prune(self, items: Iterable[tuple[Any, str]]) -> int:
        """Drop cached rows whose source frame content no longer matches.

        Returns the number of pruned rows.  A pruned key simply misses on the
        next ``gather`` and is re-encoded, so this is always safe.
        """
        if not (self.enabled and self._config.verify_content_hash):
            return 0
        pruned = 0
        for item in items:
            resolved = transition_stream_and_key(*item)
            if resolved is None:
                continue
            stream, key = resolved
            store = self._stream_store(stream)
            row = store.key_to_row.get(key)
            if row is None or store.digests is None:
                continue
            expected = int(np.uint64(np.int64(int(store.digests[row]))))
            if expected != frame_digest(self._obs_of(item), self._camera_names):
                del store.key_to_row[key]
                pruned += 1
        if pruned:
            self.stats["digest_mismatch"] += pruned
            logger.warning(
                "[obs-cache] pruned %d rows with a content-hash mismatch; "
                "they will be re-encoded.",
                pruned,
            )
        return pruned

    def summary(self) -> str:
        return (
            "[obs-cache] hits={hits} encoded={encoded} uncacheable={uncacheable} "
            "loaded={loaded_rows} pruned={digest_mismatch}".format(**self.stats)
        )


def stream_key_for_file(path: str | Path) -> str:
    """Content identity of a source file, cheap enough to compute every run."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return hashlib.sha256(
        f"{resolved}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}".encode("utf-8")
    ).hexdigest()


__all__ = [
    "FIELD_NEXT_OBS",
    "FIELD_OBS",
    "OBS_ENCODING_CACHE_SCHEMA_VERSION",
    "ObsEncodingCache",
    "ObsEncodingCacheConfig",
    "WARMUP_STREAM",
    "frame_digest",
    "stream_key_for_file",
    "transition_stream_and_key",
]
