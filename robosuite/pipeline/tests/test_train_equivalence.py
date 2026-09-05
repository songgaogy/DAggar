"""The offline training hot path must not change what it computes.

These cover the optimizations that move work onto the GPU or drop redundant
copies: where the static cache lives, how route masks are built, and how the
offline advantage provider is indexed.  All of them must be bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.dipole.replay_buffer import DipoleReplayBuffer
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowAugmentationConfig
from robosuite.pipeline.common.types import ReplayBufferConfig, Transition
from robosuite.pipeline.modules.training.dipole.branch_weights import _route_masks
from robosuite.pipeline.modules.training.dipole.episode_dataset import (
    ROUTE_ADVANTAGE,
    ROUTE_NEG_ONLY,
    ROUTE_POS_ONLY,
)

CAMERAS = ["agentview", "robot0_robotview", "robot0_eye_in_hand"]
H = 4
IMAGE = 32
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="offline training requires CUDA"
)


def _buffer(num_transitions: int = 40) -> DipoleReplayBuffer:
    buffer = DipoleReplayBuffer(
        config=ReplayBufferConfig(capacity=1000, batch_size=1),
        name="equivalence",
        camera_names=list(CAMERAS),
        action_horizon=H,
        image_size=IMAGE,
        augmentation_config=FlowAugmentationConfig(),
    )
    rng = np.random.default_rng(0)
    routes = (ROUTE_ADVANTAGE, ROUTE_POS_ONLY, ROUTE_NEG_ONLY)
    for index in range(num_transitions):
        obs = {
            camera: rng.integers(0, 255, (IMAGE, IMAGE, 3), dtype=np.uint8)
            for camera in CAMERAS
        }
        obs["state"] = rng.standard_normal(14).astype(np.float32)
        buffer.add(
            Transition(
                obs=obs,
                action=rng.standard_normal(7).astype(np.float32),
                reward=-1.0,
                next_obs=obs,
                done=(index == num_transitions - 1),
                is_intervention=bool(index % 5 == 0),
                info={
                    "episode_index": 0,
                    "episode_step": index,
                    "route": routes[index % 3],
                },
            )
        )
    return buffer


def _sample(cache, *, seed: int, device: str):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return cache.sample(8, device=device, augment=True)


@requires_cuda
def test_static_cache_on_gpu_matches_host_sampling():
    """Where the cache lives must not change the batch it produces."""
    buffer = _buffer()
    host = buffer.build_static_cache(pin_memory=True, device="cpu")
    gpu = buffer.build_static_cache(pin_memory=False, device="cuda")
    assert host.device.type == "cpu" and gpu.device.type == "cuda"

    host_batch = _sample(host, seed=7, device="cuda")
    gpu_batch = _sample(gpu, seed=7, device="cuda")
    for field in (
        "image_obs",
        "image_obs_raw",
        "proprio",
        "proprio_raw",
        "action_sequences",
        "action_sequences_raw",
        "is_intervention",
    ):
        got, want = getattr(gpu_batch, field), getattr(host_batch, field)
        assert torch.equal(got, want), field
    assert gpu_batch.metadata["start_indices"] == host_batch.metadata["start_indices"]
    assert gpu_batch.metadata["route"] == host_batch.metadata["route"]


@requires_cuda
def test_route_codes_match_the_string_masks():
    buffer = _buffer()
    cache = buffer.build_static_cache(pin_memory=False, device="cuda")
    batch = _sample(cache, seed=3, device="cuda")

    from_codes = _route_masks(batch, "cuda")
    stripped = type(batch)(
        image_obs=batch.image_obs,
        image_obs_raw=batch.image_obs_raw,
        proprio=batch.proprio,
        proprio_raw=batch.proprio_raw,
        action_sequences=batch.action_sequences,
        action_sequences_raw=batch.action_sequences_raw,
        is_intervention=batch.is_intervention,
        metadata={"route": batch.metadata["route"]},
    )
    from_strings = _route_masks(stripped, "cuda")
    for route in (ROUTE_ADVANTAGE, ROUTE_POS_ONLY, ROUTE_NEG_ONLY):
        assert torch.equal(from_codes[route], from_strings[route]), route
    total = sum(int(mask.sum()) for mask in from_codes.values())
    assert total == batch.batch_size, "routes must partition the batch"


@requires_cuda
def test_static_cache_exposes_device_side_metadata():
    buffer = _buffer()
    cache = buffer.build_static_cache(pin_memory=False, device="cuda")
    batch = _sample(cache, seed=5, device="cuda")
    starts = batch.metadata["start_index_tensor"]
    assert starts.device.type == "cuda"
    assert starts.tolist() == batch.metadata["start_indices"]
    assert batch.metadata["route_codes"].shape[0] == batch.batch_size


@requires_cuda
def test_advantage_lookup_matches_the_dict_path():
    from robosuite.pipeline.modules.training.dipole.advantage import (
        OfflineAdvantageGProvider,
    )

    torch.manual_seed(0)
    starts = [0, 3, 9, 27, 81]
    start_to_row = {start: row for row, start in enumerate(starts)}
    advantage = torch.randn(len(starts), device="cuda")
    provider = OfflineAdvantageGProvider.__new__(OfflineAdvantageGProvider)
    provider.alpha = 1.5
    provider._advantage_raw = advantage
    provider._start_to_row = dict(start_to_row)
    # Rebuild the dense lookup the constructor would have made.
    largest = max(start_to_row)
    lookup = torch.full((largest + 2,), -1, dtype=torch.long, device=advantage.device)
    lookup.index_copy_(
        0,
        torch.tensor(list(start_to_row), dtype=torch.long, device=advantage.device),
        torch.tensor(list(start_to_row.values()), dtype=torch.long, device=advantage.device),
    )
    provider._start_lookup = lookup

    query = torch.tensor([81, 0, 27, 3, 9, 9], device="cuda")
    got = provider.compute_g_for_start_indices(query)
    want = torch.stack([provider.alpha * advantage[start_to_row[int(s)]] for s in query])
    assert torch.equal(got, want)

    with pytest.raises(KeyError):
        provider.compute_g_for_start_indices(torch.tensor([5], device="cuda"))
    with pytest.raises(KeyError):
        provider.compute_g_for_start_indices(torch.tensor([10_000], device="cuda"))
