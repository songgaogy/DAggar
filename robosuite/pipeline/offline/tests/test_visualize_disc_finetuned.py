"""Offline-episode sampling and rendering tests for the nnPU visualizer."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robosuite.pipeline.offline.visualization import disc_renderer as viz_module
from robosuite.pipeline.offline.visualization.disc_adapter import (
    FinetunedPUBCEBenchmarkDiscriminator,
)
from robosuite.pipeline.offline.visualization.disc_episodes import (
    offline_trajectory as _offline_trajectory,
    sample_offline_trajectories as _sample_offline_trajectories,
)
from robosuite.pipeline.offline.visualization.disc_renderer import (
    FinetunedPUBCEVisualizer as PUBCEVisualizer,
    FinetunedTrajectoryViz as PerTrajectoryViz,
)
from robosuite.pipeline.offline.visualization.disc_contract import (
    load_finetuned_visualization_contract,
    validate_runtime_normalizer,
)


def _episode(index: int, *, interventions: list[bool]) -> dict[str, object]:
    length = len(interventions)
    policy = np.full((length, 2), float(index), dtype=np.float32)
    executed = policy.copy()
    for step, is_intervention in enumerate(interventions):
        if is_intervention:
            executed[step] = -float(index + step + 1)
    state = np.full((length, 3), float(index), dtype=np.float32)
    frames = np.full((length, 4, 4, 3), index, dtype=np.uint8)
    done = np.zeros((length,), dtype=np.bool_)
    done[-1] = True
    return {
        "obs": {"state": state, "agentview": frames},
        "next_obs": {"state": state + 1.0, "agentview": frames},
        "executed_action": executed,
        "policy_action": policy,
        "is_intervention": np.asarray(interventions, dtype=np.bool_),
        "success": np.zeros((length,), dtype=np.bool_),
        "done": done,
        "terminal_reason": "manual_reset",
    }


def _payload(*episodes: dict[str, object], task: str = "Task") -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_name": task,
        "camera_names": ["agentview"],
        "episodes": list(episodes),
    }


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_finetuned_contract_pins_encoder_inputs(tmp_path) -> None:
    model = tmp_path / "model.pth"
    normalizer = tmp_path / "normalizer.pth"
    model.write_bytes(b"model")
    normalizer.write_bytes(b"normalizer")
    checkpoint = tmp_path / "pu_bce_head_finetuned.pth"
    torch.save(
        {
            "pu_bce_detector": {},
            "finetuned_offline": True,
            "feature_source": "transformer",
            "transformer_layer": 1,
            "camera_to_view": {"agentview": "front"},
            "proprio_indices": [0, 2],
            "model_ckpt_sha256": _sha256(model),
            "normalizer_ckpt_sha256": _sha256(normalizer),
        },
        checkpoint,
    )

    contract = load_finetuned_visualization_contract(
        checkpoint,
        model_checkpoint=model,
        feature_source="transformer",
        transformer_layer=1,
        camera_to_view=None,
        proprio_indices=None,
    )

    assert contract.camera_to_view == {"agentview": "front"}
    assert contract.proprio_indices == [0, 2]
    discriminator = SimpleNamespace(
        encoder=SimpleNamespace(normalizer_checkpoint=str(normalizer))
    )
    validate_runtime_normalizer(discriminator, contract)


def test_finetuned_contract_rejects_parent_checkpoint(tmp_path) -> None:
    model = tmp_path / "model.pth"
    model.write_bytes(b"model")
    checkpoint = tmp_path / "pu_bce_head.pth"
    torch.save(
        {
            "pu_bce_detector": {},
            "feature_source": "transformer",
            "transformer_layer": 1,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="rejects parent"):
        load_finetuned_visualization_contract(
            checkpoint,
            model_checkpoint=model,
            feature_source="transformer",
            transformer_layer=1,
            camera_to_view=None,
            proprio_indices=None,
        )


def test_offline_sampling_is_reproducible_and_maps_recorded_fields(tmp_path) -> None:
    episodes = [
        _episode(index, interventions=[False, index == 2, False])
        for index in range(5)
    ]
    path = tmp_path / "offline_episodes.pt"
    torch.save(_payload(*episodes), path)

    first = _sample_offline_trajectories(
        path, task="Task", num_trajs=3, seed=7, fps=17
    )
    second = _sample_offline_trajectories(
        path, task="Task", num_trajs=3, seed=7, fps=17
    )

    assert [item.episode_index for item in first] == [
        item.episode_index for item in second
    ]
    assert len(first) == 3
    for trajectory in first:
        source = episodes[trajectory.episode_index]
        np.testing.assert_array_equal(
            trajectory.load_actions(), source["executed_action"]
        )
        np.testing.assert_array_equal(trajectory.load_states(), source["obs"]["state"])
        np.testing.assert_array_equal(
            trajectory.load_images()["agentview"], source["obs"]["agentview"]
        )
        np.testing.assert_array_equal(
            trajectory.intervention_mask,
            np.asarray(source["is_intervention"], dtype=np.bool_),
        )
        assert trajectory.video_frame_size == (256, 256)
        assert trajectory.load_failure_mask() is None
        assert trajectory.fps == 17


def test_offline_sampling_caps_at_episode_count(tmp_path) -> None:
    path = tmp_path / "offline_episodes.pt"
    torch.save(
        _payload(
            _episode(0, interventions=[False]),
            _episode(1, interventions=[False]),
        ),
        path,
    )

    sampled = _sample_offline_trajectories(
        path, task="Task", num_trajs=10, seed=0, fps=20
    )

    assert sorted(item.episode_index for item in sampled) == [0, 1]


def test_offline_sampling_rejects_task_mismatch_and_invalid_payload(tmp_path) -> None:
    mismatch = tmp_path / "mismatch.pt"
    torch.save(_payload(_episode(0, interventions=[False]), task="OtherTask"), mismatch)
    with pytest.raises(ValueError, match="task mismatch"):
        _sample_offline_trajectories(
            mismatch, task="Task", num_trajs=1, seed=0, fps=20
        )

    invalid = tmp_path / "invalid.pt"
    episode = _episode(0, interventions=[False])
    episode.pop("executed_action")
    torch.save(_payload(episode), invalid)
    with pytest.raises(KeyError, match="missing fields"):
        _sample_offline_trajectories(
            invalid, task="Task", num_trajs=1, seed=0, fps=20
        )


class _Writer:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame))

    def close(self) -> None:
        self.closed = True


def _per_trajectory_viz(*, interventions: list[bool]) -> PerTrajectoryViz:
    length = len(interventions)
    intervention_mask = np.asarray(interventions, dtype=np.bool_)
    return PerTrajectoryViz(
        video_id="offline_episode_000000",
        task_name="Task",
        num_frames=length,
        step_scores=np.ones((length,), dtype=np.float32),
        thresholds=np.full((length,), 0.5, dtype=np.float32),
        predictions=np.ones((length,), dtype=np.int64),
        gt_mask=None,
        failure_segments=[],
        first_gt_failure_frame=None,
        first_pred_failure_frame=0,
        source_kind="offline",
        intervention_frames=int(intervention_mask.sum()),
        terminal_reason="manual_reset",
        intervention_mask=intervention_mask,
    )


def _trajectory(*, intervention: bool):
    return _offline_trajectory(
        _episode(0, interventions=[intervention]),
        episode_index=0,
        task="Task",
        camera_names=["agentview"],
        fps=20,
        source_path="offline_episodes.pt",
    )


def test_scoring_propagates_offline_intervention_render_policy() -> None:
    output = SimpleNamespace(
        step_scores=np.asarray([0.1], dtype=np.float32),
        predictions=np.asarray([0], dtype=np.int64),
        aux={"threshold": 0.5},
    )
    discriminator = SimpleNamespace(score_trajectory=lambda trajectory: output)
    visualizer = PUBCEVisualizer(discriminator, flip_vertical=False)

    hidden = visualizer._score_trajectory(_trajectory(intervention=True))
    shown = visualizer._score_trajectory(_trajectory(intervention=False))

    np.testing.assert_array_equal(hidden.intervention_mask, [True])
    assert hidden.intervention_frames == 1
    np.testing.assert_array_equal(shown.intervention_mask, [False])
    assert shown.intervention_frames == 0


def test_offline_preselected_proprio_skips_checkpoint_indices() -> None:
    trajectory = _offline_trajectory(
        _episode(0, interventions=[False]),
        episode_index=0,
        task="Task",
        camera_names=["agentview"],
        fps=20,
        source_path="offline_episodes.pt",
    )
    proprio_encoder = SimpleNamespace(in_chans=3)
    action_encoder = SimpleNamespace(in_chans=2)
    model = SimpleNamespace(
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
    )
    encoder = SimpleNamespace(
        view_names=["agentview"],
        model=model,
        original_img_size=4,
    )
    adapter = object.__new__(FinetunedPUBCEBenchmarkDiscriminator)
    adapter.encoder = encoder
    adapter.camera_to_view = {"agentview": "agentview"}
    adapter.proprio_indices = np.asarray([0, 38], dtype=np.int64)
    adapter.proprio_map = {}
    adapter.feature_source = "transformer"
    adapter.use_chunk = False

    prepared = adapter._prepare_trajectory_tensors(trajectory)

    np.testing.assert_array_equal(prepared["prop"], trajectory.load_states())
    assert prepared["prop"].shape == (1, 3)


def test_offline_preselected_proprio_dimension_mismatch_is_rejected() -> None:
    trajectory = _trajectory(intervention=False)
    proprio_encoder = SimpleNamespace(in_chans=4)
    action_encoder = SimpleNamespace(in_chans=2)
    encoder = SimpleNamespace(
        view_names=["agentview"],
        model=SimpleNamespace(
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
        ),
        original_img_size=4,
    )
    adapter = object.__new__(FinetunedPUBCEBenchmarkDiscriminator)
    adapter.encoder = encoder
    adapter.camera_to_view = {"agentview": "agentview"}
    adapter.proprio_indices = np.asarray([0, 38], dtype=np.int64)
    adapter.proprio_map = {}
    adapter.feature_source = "transformer"
    adapter.use_chunk = False

    with pytest.raises(ValueError, match="must exactly match"):
        adapter._prepare_trajectory_tensors(trajectory)


def test_intervention_video_skips_discriminator_annotations(monkeypatch, tmp_path) -> None:
    writer = _Writer()
    monkeypatch.setattr(viz_module.imageio, "get_writer", lambda *args, **kwargs: writer)
    monkeypatch.setattr(
        viz_module,
        "_draw_border",
        lambda *args, **kwargs: pytest.fail("intervention video drew a border"),
    )
    monkeypatch.setattr(
        viz_module,
        "_overlay_hud_pu",
        lambda *args, **kwargs: pytest.fail("intervention video drew a HUD"),
    )
    detector = SimpleNamespace(thresholds={"Task": 0.5})
    discriminator = SimpleNamespace(_detectors_per_task={"Task": detector})
    visualizer = PUBCEVisualizer(discriminator, flip_vertical=False)

    visualizer.render_video(
        _trajectory(intervention=True),
        _per_trajectory_viz(interventions=[True]),
        str(tmp_path / "offline.mp4"),
    )

    assert len(writer.frames) == 1
    assert writer.frames[0].shape == (256, 256, 3)
    assert writer.closed


def test_non_intervention_video_keeps_discriminator_annotations(monkeypatch, tmp_path) -> None:
    writer = _Writer()
    calls = {"border": 0, "hud": 0}

    def draw_border(image, *args, **kwargs):
        calls["border"] += 1
        return image

    def draw_hud(image, *args, **kwargs):
        calls["hud"] += 1
        return image

    monkeypatch.setattr(viz_module.imageio, "get_writer", lambda *args, **kwargs: writer)
    monkeypatch.setattr(viz_module, "_draw_border", draw_border)
    monkeypatch.setattr(viz_module, "_overlay_hud_pu", draw_hud)
    detector = SimpleNamespace(thresholds={"Task": 0.5})
    discriminator = SimpleNamespace(_detectors_per_task={"Task": detector})
    visualizer = PUBCEVisualizer(discriminator, flip_vertical=False)

    visualizer.render_video(
        _trajectory(intervention=False),
        _per_trajectory_viz(interventions=[False]),
        str(tmp_path / "offline.mp4"),
    )

    assert calls == {"border": 1, "hud": 1}
    assert len(writer.frames) == 1
    assert writer.frames[0].shape == (256, 256, 3)
    assert writer.closed


def test_mixed_episode_annotates_only_non_intervention_frames(monkeypatch, tmp_path) -> None:
    writer = _Writer()
    calls = {"border": 0, "hud": 0}

    def draw_border(image, *args, **kwargs):
        calls["border"] += 1
        return image

    def draw_hud(image, *args, **kwargs):
        calls["hud"] += 1
        return image

    trajectory = _offline_trajectory(
        _episode(0, interventions=[False, True]),
        episode_index=0,
        task="Task",
        camera_names=["agentview"],
        fps=20,
        source_path="offline_episodes.pt",
    )
    monkeypatch.setattr(viz_module.imageio, "get_writer", lambda *args, **kwargs: writer)
    monkeypatch.setattr(viz_module, "_draw_border", draw_border)
    monkeypatch.setattr(viz_module, "_overlay_hud_pu", draw_hud)
    detector = SimpleNamespace(thresholds={"Task": 0.5})
    discriminator = SimpleNamespace(_detectors_per_task={"Task": detector})
    visualizer = PUBCEVisualizer(discriminator, flip_vertical=False)

    visualizer.render_video(
        trajectory,
        _per_trajectory_viz(interventions=[False, True]),
        str(tmp_path / "offline.mp4"),
    )

    assert calls == {"border": 1, "hud": 1}
    assert len(writer.frames) == 2
    assert all(frame.shape == (256, 256, 3) for frame in writer.frames)
    assert writer.closed
