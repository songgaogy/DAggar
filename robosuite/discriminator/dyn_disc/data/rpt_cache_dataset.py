"""Memory-mapped RPT latent cache dataset."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


CACHE_VERSION = "rpt_dinov3_mean_v1"
MANIFEST_NAME = "manifest.json"


def manifest_fingerprint(payload: Dict[str, Any]) -> str:
    """Return a stable SHA-256 over manifest content excluding its fingerprint."""
    canonical = {key: value for key, value in payload.items() if key != "fingerprint"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cache_manifest(cache_dir: str | Path) -> Dict[str, Any]:
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"RPT cache manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported RPT cache version {manifest.get('cache_version')!r}; expected {CACHE_VERSION!r}"
        )
    actual = manifest_fingerprint(manifest)
    if manifest.get("fingerprint") != actual:
        raise ValueError("RPT cache manifest fingerprint mismatch")
    return manifest


class RPTCacheDataset(Dataset):
    """Serve contiguous, within-episode RPT windows from float32 shards.

    Each item contains ``visual_latents [T,2,768]``, ``proprio [T,14]``,
    and ``actions [T,7]``. Shards use mmap-backed ``torch.load`` where the
    installed PyTorch version supports it.
    """

    def __init__(self, cache_dir: str | Path, context_length: int = 8) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest = load_cache_manifest(self.cache_dir)
        self.fingerprint = str(self.manifest["fingerprint"])
        self.context_length = int(context_length)
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if tuple(self.manifest["view_names"]) != ("agentview", "robot0_eye_in_hand"):
            raise ValueError("RPT cache must contain agentview then robot0_eye_in_hand")
        if int(self.manifest["latent_dim"]) != 768:
            raise ValueError("RPT cache visual latent dimension must be 768")
        if int(self.manifest["proprio_dim"]) != 14 or int(self.manifest["action_dim"]) != 7:
            raise ValueError("RPT cache requires 14D proprio and 7D actions")

        self._shards: List[Dict[str, Any]] = []
        self._episodes: List[tuple[int, int, int]] = []
        self._window_ends: List[int] = []
        total_windows = 0
        for shard_index, shard_entry in enumerate(self.manifest["shards"]):
            shard_path = self.cache_dir / shard_entry["path"]
            if not shard_path.is_file():
                raise FileNotFoundError(f"RPT cache shard not found: {shard_path}")
            try:
                shard = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=True)
            except TypeError:
                shard = torch.load(shard_path, map_location="cpu")
            self._validate_shard(shard, shard_path)
            self._shards.append(shard)
            episode_start = 0
            for episode_end in shard["episode_ends"].tolist():
                episode_end = int(episode_end)
                count = max(0, episode_end - episode_start - self.context_length + 1)
                if count:
                    total_windows += count
                    self._episodes.append((shard_index, episode_start, count))
                    self._window_ends.append(total_windows)
                episode_start = episode_end
        self._length = total_windows
        if self._length == 0:
            raise RuntimeError(f"RPT cache has no {self.context_length}-frame windows")

        statistics = self.manifest["statistics"]
        self.proprio_min = torch.tensor(statistics["proprio_min"], dtype=torch.float32)
        self.proprio_max = torch.tensor(statistics["proprio_max"], dtype=torch.float32)

    @staticmethod
    def _validate_shard(shard: Dict[str, Any], path: Path) -> None:
        required = {"visual_latents", "proprio", "actions", "episode_ends", "source"}
        missing = required.difference(shard)
        if missing:
            raise ValueError(f"RPT shard {path} misses keys: {sorted(missing)}")
        visual = shard["visual_latents"]
        proprio = shard["proprio"]
        actions = shard["actions"]
        if visual.dtype != torch.float32 or visual.ndim != 3 or visual.shape[1:] != (2, 768):
            raise ValueError(f"Invalid visual latent tensor in {path}: {tuple(visual.shape)} {visual.dtype}")
        length = int(visual.shape[0])
        if proprio.shape != (length, 14) or proprio.dtype != torch.float32:
            raise ValueError(f"Invalid proprio tensor in {path}: {tuple(proprio.shape)} {proprio.dtype}")
        if actions.shape != (length, 7) or actions.dtype != torch.float32:
            raise ValueError(f"Invalid action tensor in {path}: {tuple(actions.shape)} {actions.dtype}")
        episode_ends = shard["episode_ends"]
        if episode_ends.dtype != torch.int64 or episode_ends.ndim != 1:
            raise ValueError(f"Invalid episode_ends in {path}")
        if length <= 0 or not len(episode_ends) or int(episode_ends[-1]) != length:
            raise ValueError(f"episode_ends in {path} do not terminate at shard length {length}")
        if not bool(torch.all(episode_ends[1:] > episode_ends[:-1])):
            raise ValueError(f"episode_ends in {path} must be strictly increasing")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._window_ends, index)
        previous_end = 0 if episode_index == 0 else self._window_ends[episode_index - 1]
        shard_index, episode_start, _ = self._episodes[episode_index]
        start = episode_start + index - previous_end
        stop = start + self.context_length
        shard = self._shards[shard_index]
        return {
            "visual_latents": shard["visual_latents"][start:stop],
            "proprio": shard["proprio"][start:stop],
            "actions": shard["actions"][start:stop],
        }
