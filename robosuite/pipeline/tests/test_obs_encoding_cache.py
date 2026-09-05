"""Equivalence and invalidation tests for the frozen observation-encoding cache.

The cache is only useful if reusing a row is indistinguishable from encoding it
again.  The frozen encoder is deterministic at a fixed batch size but *not*
batch-invariant, so the cache runs every forward through
``encode_in_fixed_batches``; these tests pin that behaviour down.

The GPU tests need the real dynamics checkpoint and are skipped without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.obs_encoding_cache import (
    FIELD_NEXT_OBS,
    FIELD_OBS,
    ObsEncodingCache,
    ObsEncodingCacheConfig,
    encode_in_fixed_batches,
    frame_digest,
    stream_key_for_file,
    transition_stream_and_key,
)


class _FakeTransition:
    def __init__(self, info, obs=None, next_obs=None):
        self.info = info
        self.obs = obs or {}
        self.next_obs = next_obs or {}


# --------------------------------------------------------------------------- #
# Keys (no GPU needed)                                                          #
# --------------------------------------------------------------------------- #


def test_round_key_uses_source_provenance():
    transition = _FakeTransition(
        {"source_round": 2, "round_episode_index": 5, "source_frame_index": 17}
    )
    assert transition_stream_and_key(transition, FIELD_OBS) == (
        "round_002",
        "r2:e5:f17:obs",
    )
    assert transition_stream_and_key(transition, FIELD_NEXT_OBS) == (
        "round_002",
        "r2:e5:f17:next_obs",
    )


def test_warmup_key_takes_precedence():
    transition = _FakeTransition({"warmup_row": 9, "source_round": 0})
    assert transition_stream_and_key(transition, FIELD_OBS) == ("warmup", "w9:obs")


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"source_round": 1},                              # incomplete provenance
        {"source_round": 1, "round_episode_index": 0},    # still incomplete
        {
            "source_round": 1,
            "round_episode_index": 0,
            "source_frame_index": 3,
            "synthetic_vast_success_tail": True,
        },
        {
            "source_round": 1,
            "round_episode_index": 0,
            "source_frame_index": 3,
            "synthetic_vast_failure_tail": True,
        },
    ],
)
def test_uncacheable_frames_return_none(info):
    """Legacy rows and synthetic tails must fall back to live encoding."""
    assert transition_stream_and_key(_FakeTransition(info), FIELD_OBS) is None


def test_frame_digest_detects_content_change():
    obs = {
        "cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.zeros(3, dtype=np.float32),
    }
    before = frame_digest(obs, ["cam"])
    assert before == frame_digest(obs, ["cam"])
    obs["cam"][0, 0, 0] = 1
    assert frame_digest(obs, ["cam"]) != before


def test_stream_key_changes_with_file(tmp_path: Path):
    target = tmp_path / "transitions.pt"
    target.write_bytes(b"a")
    first = stream_key_for_file(target)
    target.write_bytes(b"bb")
    assert stream_key_for_file(target) != first


# --------------------------------------------------------------------------- #
# Fixed-batch encoding (no GPU needed)                                          #
# --------------------------------------------------------------------------- #


def test_encode_in_fixed_batches_pads_the_tail_and_preserves_order():
    seen: list[list[int]] = []

    def encode(rows):
        seen.append(list(rows))
        values = torch.tensor(rows, dtype=torch.float32).view(-1, 1, 1)
        return {"visual": values.expand(-1, 1, 2).clone(), "proprio": values.clone()}

    out = encode_in_fixed_batches(encode, num_rows=10, batch_size=4)
    assert [len(batch) for batch in seen] == [4, 4, 4]      # tail padded 2 -> 4
    assert seen[-1] == [8, 9, 9, 9]                          # padding repeats last row
    assert out["visual"].shape[0] == 10
    assert torch.equal(out["proprio"].view(-1), torch.arange(10, dtype=torch.float32))


def test_encode_in_fixed_batches_pads_when_smaller_than_one_batch():
    sizes: list[int] = []

    def encode(rows):
        sizes.append(len(rows))
        values = torch.tensor(rows, dtype=torch.float32).view(-1, 1)
        return {"visual": values.unsqueeze(-1), "proprio": values}

    out = encode_in_fixed_batches(encode, num_rows=3, batch_size=64)
    assert sizes == [64]
    assert out["visual"].shape[0] == 3


# --------------------------------------------------------------------------- #
# Cache round-trip (no GPU needed: a stub encoder stands in for DINOv3)          #
# --------------------------------------------------------------------------- #


class _StubEncoder:
    """Deterministic stand-in whose output depends only on the frame content."""

    obs_encoding_spec = {
        "encoder_key": "stub-key",
        "encoder_checkpoint": "/dev/null",
        "feature_source": "transformer",
        "transformer_layer": 1,
        "original_img_size": [256, 256],
        "view_names": ["cam"],
        "policy_cameras": ["cam"],
        "state_feature_dim": 8,
        "chunk_feature_dim": 8,
    }


def _make_items(count: int, *, round_index: int = 0):
    items = []
    for frame in range(count):
        obs = {
            "cam": np.full((2, 2, 3), frame, dtype=np.uint8),
            "state": np.full(2, frame, dtype=np.float32),
        }
        transition = _FakeTransition(
            {
                "source_round": round_index,
                "round_episode_index": 0,
                "source_frame_index": frame,
            },
            obs=obs,
            next_obs=obs,
        )
        items.append((transition, FIELD_OBS))
    return items


def _encode_from_items(items):
    def encode_missing(local_indices):
        rows = list(local_indices)
        values = torch.tensor(
            [float(items[row][0].info["source_frame_index"]) for row in rows]
        ).view(-1, 1)
        return {
            "visual": values.expand(-1, 4).reshape(-1, 1, 2, 2).clone(),
            "proprio": values.expand(-1, 2).unsqueeze(1).clone(),
        }

    return encode_missing


def _cache(tmp_path: Path, **overrides):
    config = ObsEncodingCacheConfig(
        round_cache_dir=str(tmp_path),
        warmup_cache_dir=str(tmp_path),
        encode_batch_size=4,
        **overrides,
    )
    return ObsEncodingCache(
        encoder=_StubEncoder(),
        config=config,
        camera_names=["cam"],
        image_size=128,
        stream_keys={"round_000": "stream-key-v1"},
    )


def test_cache_hit_matches_live_encoding_and_persists(tmp_path: Path):
    items = _make_items(10)
    cold = _cache(tmp_path)
    first = cold.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert cold.stats["encoded"] == 10 and cold.stats["hits"] == 0

    warm = cold.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert cold.stats["hits"] == 10
    assert torch.equal(first["visual"], warm["visual"])
    assert torch.equal(first["proprio"], warm["proprio"])

    cold.flush()
    reloaded = _cache(tmp_path)
    from_disk = reloaded.gather(
        items, encode_missing=_encode_from_items(items), device="cpu"
    )
    assert reloaded.stats["encoded"] == 0, "the shard must serve every row"
    assert torch.equal(first["visual"], from_disk["visual"])


def test_repeated_frames_are_encoded_once(tmp_path: Path):
    """Overlapping windows request the same frame many times per call."""
    items = _make_items(4)
    duplicated = items * 5
    cache = _cache(tmp_path)
    out = cache.gather(
        duplicated, encode_missing=_encode_from_items(duplicated), device="cpu"
    )
    assert cache.stats["encoded"] == 4, "duplicates must collapse to unique keys"
    for repeat in range(1, 5):
        assert torch.equal(out["visual"][:4], out["visual"][repeat * 4 : (repeat + 1) * 4])


def test_stale_stream_key_rebuilds(tmp_path: Path):
    items = _make_items(6)
    cache = _cache(tmp_path)
    reference = cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    cache.flush()

    stale = ObsEncodingCache(
        encoder=_StubEncoder(),
        config=ObsEncodingCacheConfig(
            round_cache_dir=str(tmp_path),
            warmup_cache_dir=str(tmp_path),
            encode_batch_size=4,
        ),
        camera_names=["cam"],
        image_size=128,
        stream_keys={"round_000": "stream-key-v2"},
    )
    out = stale.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert stale.stats["loaded_rows"] == 0, "a changed stream key must invalidate"
    assert stale.stats["encoded"] == 6
    assert torch.equal(reference["visual"], out["visual"])


def test_content_hash_mismatch_is_pruned(tmp_path: Path):
    items = _make_items(5)
    cache = _cache(tmp_path)
    cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")

    items[2][0].obs["cam"][0, 0, 0] = 99  # source frame changed underneath the cache
    assert cache.verify_and_prune(items) == 1
    cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert cache.stats["digest_mismatch"] == 1


def test_uncacheable_rows_still_resolve(tmp_path: Path):
    items = _make_items(4)
    items[1][0].info.pop("source_round")          # legacy row, no provenance
    cache = _cache(tmp_path)
    out = cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert cache.stats["uncacheable"] == 1
    assert out["visual"].shape[0] == 4
    again = cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert torch.equal(out["visual"], again["visual"])


def test_disabled_cache_encodes_everything(tmp_path: Path):
    items = _make_items(4)
    cache = _cache(tmp_path, enabled=False)
    cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    cache.gather(items, encode_missing=_encode_from_items(items), device="cpu")
    assert cache.stats["hits"] == 0
    cache.flush()
    assert not list(tmp_path.iterdir()), "a disabled cache must not write shards"
