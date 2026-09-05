from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline import eval_policy
from robosuite.pipeline.eval_policy import (
    _checkpoint_spec,
    _publish_directory,
    _run_directory_for_checkpoint,
    rollout_episode,
)
from robosuite.pipeline.src.dsrl import DSRLInferencePolicy, NetworkConfig
from robosuite.pipeline.utils import file_identity


class MockEnv:
    def __init__(
        self,
        *,
        success_step: int | None = None,
        terminated_step: int | None = None,
        truncated_step: int | None = None,
    ) -> None:
        self.success_step = success_step
        self.terminated_step = terminated_step
        self.truncated_step = truncated_step
        self.steps = 0

    def step(self, action: np.ndarray):
        del action
        self.steps += 1
        terminated = self.steps == self.terminated_step
        truncated = self.steps == self.truncated_step
        return {}, 123.0, terminated, truncated, {
            "success": self.steps == self.success_step
        }


def _rollout(
    env: MockEnv, max_steps: int, horizon: int = 8
) -> tuple[tuple[float, int, int, bool, str], int]:
    frames = 0

    def append_frame() -> None:
        nonlocal frames
        frames += 1

    result = rollout_episode(
        env=env,
        initial_observation={},
        select_action_chunk=lambda _: np.zeros((horizon, 7), dtype=np.float32),
        build_observation=lambda raw: raw,
        max_steps=max_steps,
        append_frame=append_frame,
    )
    return result, frames


@pytest.mark.parametrize(
    ("env", "max_steps", "expected"),
    [
        (MockEnv(success_step=11), 50, (-10.0, 11, 2, True, "success")),
        (MockEnv(success_step=3), 50, (-2.0, 3, 1, True, "success")),
        (MockEnv(), 10, (-10.0, 10, 2, False, "horizon")),
        (MockEnv(terminated_step=4), 50, (-4.0, 4, 1, False, "environment")),
        (MockEnv(truncated_step=5), 50, (-5.0, 5, 1, False, "horizon")),
    ],
)
def test_rollout_chunk_termination_and_sparse_reward(
    env: MockEnv,
    max_steps: int,
    expected: tuple[float, int, int, bool, str],
) -> None:
    result, frames = _rollout(env, max_steps)
    assert result == expected
    assert env.steps == expected[1]
    assert frames == expected[1] + 1


def test_rollout_count_is_exactly_fifty() -> None:
    results = [_rollout(MockEnv(success_step=1), 500)[0] for _ in range(50)]
    assert len(results) == 50
    assert all(item == (0.0, 1, 1, True, "success") for item in results)


def test_video_frame_and_encoder_settings(tmp_path: Path, monkeypatch) -> None:
    rendered = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(3, 2, 3)

    class Simulation:
        def render(self, **kwargs):
            assert kwargs == {"height": 512, "width": 512, "camera_name": "agentview"}
            return rendered

    captured: dict[str, object] = {}

    def fake_writer(path: Path, **kwargs):
        captured.update(path=path, **kwargs)
        return object()

    monkeypatch.setattr(eval_policy.imageio, "get_writer", fake_writer)
    frame = eval_policy._frame(type("Env", (), {"sim": Simulation()})(), camera="agentview")
    writer = eval_policy._video_writer(tmp_path / "episode_0000.mp4", 20)

    assert writer is not None
    assert np.array_equal(frame, np.flipud(rendered))
    assert frame.flags.c_contiguous
    assert captured["fps"] == 20
    assert captured["codec"] == "libx264"
    assert captured["ffmpeg_params"] == [
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def test_publish_directory_replaces_only_safe_episode_target(tmp_path: Path) -> None:
    eval_directory = tmp_path / "run" / "eval"
    destination = eval_directory / "episode_00000020"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("old", encoding="utf-8")
    temporary = eval_directory / ".episode_00000020.tmp-test"
    temporary.mkdir()
    (temporary / "new.txt").write_text("new", encoding="utf-8")

    _publish_directory(temporary, destination)

    assert not (destination / "old.txt").exists()
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    unsafe = tmp_path / "outside"
    unsafe.mkdir()
    with pytest.raises(ValueError, match="Unsafe evaluation output"):
        _publish_directory(unsafe, tmp_path / "episode_00000020")


def test_publish_directory_rejects_symlinked_eval_parent(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    eval_link = tmp_path / "run" / "eval"
    eval_link.parent.mkdir()
    eval_link.symlink_to(external, target_is_directory=True)
    destination = eval_link / "episode_00000020"
    temporary = eval_link / ".episode_00000020.tmp-test"
    temporary.mkdir()

    with pytest.raises(RuntimeError, match="symlinked eval directory"):
        _publish_directory(temporary, destination)

    assert temporary.is_dir()
    assert not destination.exists()


def test_checkpoint_metadata_and_artifact_identity(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required by the DSRL evaluation test suite.")
    run_directory = tmp_path / "PickPlaceCereal_run"
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    flow_path = tmp_path / "flow.pt"
    dino_path = tmp_path / "dino.pt"
    flow_path.write_bytes(b"flow")
    dino_path.write_bytes(b"dino")
    network = NetworkConfig(
        visual_dim=12,
        proprio_dim=3,
        state_dim=8,
        action_horizon=8,
        action_dim=7,
        hidden_dims=(16, 16, 16),
    )
    payload = {
        "config": {
            "task": {"name": "PickPlaceCereal"},
            "runtime": {"inference_device": "cuda:0"},
            "inputs": {
                "base_policy_checkpoint": str(flow_path),
                "dinov2_checkpoint": str(dino_path),
            },
        },
        "counters": {"completed_episodes": 20},
        "trainer": {
            "agent": {
                "config": {"network": asdict(network)},
                "bottleneck": {},
                "actor": {},
            }
        },
        "fingerprints": {
            "flow": file_identity(flow_path),
            "dinov2": file_identity(dino_path),
        },
    }
    checkpoint = checkpoint_directory / "latest.pt"
    torch.save(payload, checkpoint)

    spec = _checkpoint_spec(checkpoint, None)

    assert _run_directory_for_checkpoint(checkpoint) == run_directory
    assert spec[2:4] == ("PickPlaceCereal", 20)
    assert spec[4] == torch.device("cuda:0")
    assert spec[6:] == (flow_path.resolve(), dino_path.resolve())
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _checkpoint_spec(checkpoint, "cpu")
    flow_path.write_bytes(b"replaced")
    with pytest.raises(FileNotFoundError, match="checkpoint-compatible flow"):
        _checkpoint_spec(checkpoint, None)


def test_checkpoint_must_be_inside_checkpoints_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoints directory"):
        _run_directory_for_checkpoint(tmp_path / "latest.pt")


def test_inference_policy_is_strict_frozen_and_deterministic() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required by the DSRL evaluation test suite.")
    config = NetworkConfig(
        visual_dim=12,
        proprio_dim=3,
        state_dim=8,
        action_horizon=8,
        action_dim=7,
        hidden_dims=(16, 16, 16),
    )
    source = DSRLInferencePolicy(config, "cuda:0")
    restored = DSRLInferencePolicy(config, "cuda:0")
    restored.load_inference_state(
        {"bottleneck": source.bottleneck.state_dict(), "actor": source.actor.state_dict()}
    )
    dino = torch.randn(2, 3, 4, device="cuda:0")
    proprio = torch.randn(2, 3, device="cuda:0")

    first = restored.latent(dino, proprio, deterministic=True)
    torch.randn(100, device="cuda:0")
    second = restored.latent(dino, proprio, deterministic=True)

    assert torch.equal(first, second)
    assert first.shape == (2, 8, 7)
    assert first.device.type == "cuda"
    assert torch.all(first.abs() <= config.latent_limit)
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    with pytest.raises(RuntimeError):
        restored.load_inference_state({"bottleneck": {}, "actor": {}})
