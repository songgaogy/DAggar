import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from robosuite.discriminator.dyn_disc.data.rpt_cache_dataset import RPTCacheDataset
from robosuite.discriminator.dyn_disc.training.build_rpt_cache import PROPRIO_MAP, _task_name, build_rpt_cache


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="RPT cache tests require CUDA")


def test_task_name_resolves_panda_prefix_and_filename_substring():
    path = Path(
        "/data/gy/robosuite/raw/PandaPickPlaceCan/success_rollout/"
        "flow_multi_rollout_PickPlaceCan_success_2026-06-16_03-20-08_100.hdf5"
    )
    assert _task_name(path, PROPRIO_MAP) == "PickPlaceCan"
    assert _task_name(Path("/data/PandaStack/success_rollout/rollouts.hdf5"), PROPRIO_MAP) == "Stack"
    assert _task_name(Path("/data/PickPlaceCereal/success_rollout/rollouts.hdf5"), PROPRIO_MAP) == "PickPlaceCereal"
    assert (
        _task_name(Path("/data/PandaPickPlaceCereal/success_rollout/rollouts.hdf5"), PROPRIO_MAP)
        == "PickPlaceCereal"
    )


class _DummyCUDAEncoder(nn.Module):
    latent_dim = 768

    def __init__(self):
        super().__init__()
        self.call_sizes = []

    def encode_images(self, images):
        assert images.is_cuda
        self.call_sizes.append(int(images.shape[0]))
        values = images.float().mean(dim=(1, 2, 3), keepdim=False)
        return values[:, None].expand(-1, self.latent_dim).contiguous()

    def manifest_config(self):
        return {
            "model_path": "dummy",
            "image_size": 8,
            "latent_dim": 768,
            "pooling": "test_mean",
            "inference_dtype": "float32",
            "output_dtype": "float32",
        }


def _write_hdf5(path: Path, task: str):
    state_dim = max(PROPRIO_MAP[task]) + 1
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        demos = handle.create_group("demos")
        for episode, length in enumerate((9, 10)):
            demo = demos.create_group(f"demo_{episode:06d}")
            states = np.arange(length * state_dim, dtype=np.float32).reshape(length, state_dim)
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=np.full((length, 7), episode, dtype=np.float32))
            observations = demo.create_group("observations")
            for view_index, view in enumerate(("agentview", "robot0_eye_in_hand")):
                group = observations.create_group(view)
                values = np.arange(length, dtype=np.uint8) + 10 + view_index + 20 * episode
                images = np.broadcast_to(values[:, None, None, None], (length, 8, 8, 3)).copy()
                group.create_dataset("images", data=images, chunks=(2, 4, 4, 1))


def _make_inputs(tmp_path: Path):
    inputs = []
    for task in PROPRIO_MAP:
        directory = tmp_path / "data" / task / "success_rollout"
        _write_hdf5(directory / "rollouts.hdf5", task)
        inputs.append(str(directory))
    return inputs


@requires_cuda
def test_build_cache_and_dataset_windows_do_not_cross_episodes(tmp_path):
    cache_dir = tmp_path / "cache"
    manifest = build_rpt_cache(
        inputs=_make_inputs(tmp_path),
        cache_dir=str(cache_dir),
        encoder=_DummyCUDAEncoder(),
        batch_size=4,
        max_trajectories_per_task=2,
        num_workers=0,
        io_chunk_frames=2,
    )
    assert manifest is not None
    assert manifest["total_episodes"] == 14
    dataset = RPTCacheDataset(cache_dir, context_length=8)
    assert len(dataset) == 7 * ((9 - 8 + 1) + (10 - 8 + 1))
    item = dataset[0]
    assert item["visual_latents"].shape == (8, 2, 768)
    assert item["proprio"].shape == (8, 14)
    assert item["actions"].shape == (8, 7)
    assert all(tensor.dtype == torch.float32 for tensor in item.values())
    assert torch.equal(dataset[-1]["actions"], torch.ones(8, 7))


@requires_cuda
def test_cache_manifest_invalidates_changed_source(tmp_path):
    inputs = _make_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    build_rpt_cache(
        inputs=inputs,
        cache_dir=str(cache_dir),
        encoder=_DummyCUDAEncoder(),
        max_trajectories_per_task=2,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="inference_batching"):
        build_rpt_cache(
            inputs=inputs,
            cache_dir=str(cache_dir),
            encoder=_DummyCUDAEncoder(),
            batch_size=400,
            max_trajectories_per_task=2,
            num_workers=0,
        )
    source = next((tmp_path / "data").glob("*/success_rollout/*.hdf5"))
    source.touch()
    with pytest.raises(ValueError, match="does not match current sources"):
        build_rpt_cache(
            inputs=inputs,
            cache_dir=str(cache_dir),
            encoder=_DummyCUDAEncoder(),
            max_trajectories_per_task=2,
            num_workers=0,
        )


@requires_cuda
def test_cache_rejects_tampered_manifest(tmp_path):
    inputs = _make_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    build_rpt_cache(
        inputs=inputs,
        cache_dir=str(cache_dir),
        encoder=_DummyCUDAEncoder(),
        max_trajectories_per_task=2,
        num_workers=0,
    )
    path = cache_dir / "manifest.json"
    payload = json.loads(path.read_text())
    payload["latent_dim"] = 123
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        RPTCacheDataset(cache_dir)


@requires_cuda
def test_parallel_loader_preserves_frame_view_order_and_combines_views(tmp_path):
    task = "Stack"
    source = tmp_path / "data" / task / "success_rollout" / "rollouts.hdf5"
    _write_hdf5(source, task)
    cache_dir = tmp_path / "parallel_cache"
    encoder = _DummyCUDAEncoder().cuda()
    manifest = build_rpt_cache(
        inputs=[str(source.parent)],
        cache_dir=str(cache_dir),
        encoder=encoder,
        batch_size=8,
        max_trajectories_per_task=2,
        num_workers=2,
        prefetch_factor=2,
        io_chunk_frames=2,
        proprio_map={task: PROPRIO_MAP[task]},
    )
    shard = torch.load(
        cache_dir / manifest["shards"][0]["path"],
        map_location="cuda",
        weights_only=True,
    )
    expected_agent = torch.cat(
        [torch.arange(9, device="cuda") + 10, torch.arange(10, device="cuda") + 30]
    ).float()
    expected_hand = expected_agent + 1
    assert torch.equal(shard["visual_latents"][:, 0, 0], expected_agent)
    assert torch.equal(shard["visual_latents"][:, 1, 0], expected_hand)
    assert torch.equal(shard["episode_ends"], torch.tensor([9, 19], device="cuda"))
    assert sum(encoder.call_sizes) == 2 * 19
    assert len(encoder.call_sizes) == 6
    assert max(encoder.call_sizes) == 8
