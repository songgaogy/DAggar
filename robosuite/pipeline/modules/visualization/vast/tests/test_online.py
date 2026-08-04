from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.data_util import (
    clone_with_absorbing_success_tail,
)
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.common.types import ReplayBufferConfig
from robosuite.pipeline.modules.training.dipole.advantage import (
    precompute_offline_advantage,
)
from robosuite.pipeline.modules.visualization.vast import renderer
from robosuite.pipeline.modules.visualization.vast import discriminator as disc_viz
from robosuite.pipeline.modules.visualization.vast.online import (
    OnlineMetricAssembly,
    SelectedOnlineEpisode,
    _write_online_metrics_csv,
    assemble_online_metrics,
    materialize_policy_sections,
    select_online_episode,
)


def _episode(
    index: int,
    interventions: list[bool],
    *,
    terminal_reason: str = "success",
) -> dict:
    length = len(interventions)
    obs = {
        "state": np.zeros((length, 3), dtype=np.float32),
        "agentview": np.zeros((length, 4, 4, 3), dtype=np.uint8),
    }
    next_obs = {key: value.copy() for key, value in obs.items()}
    actions = np.zeros((length, 2), dtype=np.float32)
    success = np.zeros(length, dtype=np.bool_)
    if terminal_reason == "success":
        success[-1] = True
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    return {
        "episode_index": index,
        "obs": obs,
        "next_obs": next_obs,
        "executed_action": actions,
        "policy_action": actions.copy(),
        "is_intervention": np.asarray(interventions, dtype=np.bool_),
        "success": success,
        "done": done,
        "terminal_reason": terminal_reason,
    }


def _payload(*episodes: dict) -> dict:
    return {
        "schema_version": 1,
        "task_name": "Task",
        "camera_names": ["agentview"],
        "img_height": 4,
        "img_width": 4,
        "action_dim": 2,
        "episodes": list(episodes),
    }


def _result_rows(count: int, threshold: float = 0.5):
    rows = []
    for step in range(count):
        rows.append(
            {
                "step": float(step),
                "window_start": float(step),
                "v": 1.0 + step,
                "v_std": 0.0,
                "next_v": 2.0 + step,
                "bootstrap_v": 1.0,
                "advantage_td1": 3.0 + step,
                "advantage_gae": 4.0 + step,
                "td_target": 5.0 + step,
                "env_reward_horizon": -1.0,
                "disc_reward_horizon": -0.5,
                "total_reward_horizon": -1.5,
                "disc_intrinsic_step0": -0.25,
                "failure_score_start": 0.75,
                "done_chunk": float(step == count - 1),
                "has_valid_next": float(step != count - 1),
                "sampled_k": 1.0,
                "future_frame_index": float(step + 2),
                "advantage_stitched": 6.0 + step,
            }
        )
    disc = renderer.PerStepNNPUDisc(
        failure_score=np.full(count + 1, 0.75, dtype=np.float32),
        intrinsic_reward=np.full(count + 1, -0.25, dtype=np.float32),
        threshold=threshold,
        pred_failure=np.ones(count + 1, dtype=np.float32),
    )
    return rows, disc


def test_select_online_episode_filters_ineligible_cumulative_data(tmp_path: Path) -> None:
    first = tmp_path / "round0.pt"
    second = tmp_path / "round1.pt"
    torch.save(_payload(_episode(0, [False], terminal_reason="success")), first)
    torch.save(
        _payload(_episode(0, [False, False, True], terminal_reason="success")),
        second,
    )

    selected = select_online_episode([first, second], seed=7, action_horizon=2)

    assert selected.merged_episode_index == 1
    assert selected.source_round == 1
    assert selected.round_episode_index == 0
    assert selected.source_paths == [str(first.resolve()), str(second.resolve())]


def test_online_assembly_preserves_global_gaps_and_section_local_stride() -> None:
    episode = _episode(
        0,
        [False, False, False, True, True, False, False, False],
        terminal_reason="success",
    )
    episode["source_round"] = 2
    episode["source_episode_index"] = 4
    selected = SelectedOnlineEpisode(
        merged_episode_index=9,
        source_round=2,
        round_episode_index=4,
        terminal_reason="success",
        episode=episode,
        camera_names=["agentview"],
        source_paths=["r0", "r1", "r2"],
    )
    sections = materialize_policy_sections(
        selected,
        action_horizon=2,
        reward_success=0.0,
        reward_fail=-1.0,
    )
    source_frames = [
        [int((item.info or {})["source_frame_index"]) for item in section]
        for section in sections
    ]
    assert source_frames == [
        [0, 1, 2],
        [5, 6, 7, 7],
    ]
    assert not any(
        bool((item.info or {}).get("synthetic_vast_success_tail", False))
        for item in sections[0]
    )
    assert bool(
        (sections[1][-1].info or {}).get("synthetic_vast_success_tail", False)
    )

    assembly = assemble_online_metrics(
        selected,
        sections,
        [_result_rows(2), _result_rows(3)],
        advantage_estimator="td1",
        threshold=0.5,
    )

    assert isinstance(assembly, OnlineMetricAssembly)
    assert len(assembly.rows) == 8
    assert [int(row["step"]) for row in assembly.rows] == list(range(8))
    assert assembly.computed_windows == 5
    assert [index for index, value in enumerate(assembly.annotation_mask) if value] == [
        0,
        1,
        5,
        6,
        7,
    ]
    for index in (2, 3, 4):
        assert np.isnan(assembly.rows[index]["advantage_policy"])
        assert np.isnan(assembly.per_step_disc.failure_score[index])
    assert assembly.rows[5]["section_step"] == 0.0
    assert assembly.rows[7]["section_step"] == 2.0
    assert assembly.rows[5]["future_frame_index"] == 7.0
    assert np.isnan(assembly.rows[6]["future_frame_index"])
    assert np.isnan(assembly.rows[7]["future_frame_index"])
    assert len(assembly.per_step_disc.failure_score) == len(assembly.rows)

    nonoverlap = renderer.filter_nonoverlap_chunk_metrics(assembly.rows, 2)
    kept = [
        int(row["step"])
        for row in nonoverlap
        if float(row["valid_window"]) > 0.5
    ]
    assert kept == [0, 5, 7]
    assert len(nonoverlap) == len(assembly.rows)


def test_online_csv_writes_nan_metrics_as_blank(tmp_path: Path) -> None:
    rows = [
        {"step": 0.0, "is_intervention": 0.0, "advantage_policy": 1.0},
        {"step": 1.0, "is_intervention": 1.0, "advantage_policy": float("nan")},
    ]
    path = tmp_path / "steps.csv"

    _write_online_metrics_csv(path, rows)

    with path.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["advantage_policy"] == "1.0"
    assert loaded[1]["advantage_policy"] == ""


def test_discriminator_annotation_mask_blanks_human_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transitions = [
        Transition(
            obs={"state": np.zeros(1), "agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            action=np.zeros(1),
            reward=0.0,
            next_obs={"state": np.zeros(1), "agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            done=index == 1,
        )
        for index in range(2)
    ]
    video_masks: list[np.ndarray] = []

    def fake_video(path, frames, **kwargs):
        video_masks.append(np.asarray(kwargs["annotation_mask"]))

    def fake_plot(path_base, **kwargs):
        return path_base.with_suffix(".png"), path_base.with_suffix(".pdf")

    monkeypatch.setattr(disc_viz, "_write_video", fake_video)
    monkeypatch.setattr(disc_viz, "_plot_disc_scores", fake_plot)

    result = disc_viz.visualize_selected_trajectory_discriminator_nnpu(
        output_dir=tmp_path,
        transitions=transitions,
        camera_names=["agentview"],
        failure_score=np.asarray([1.0, 9.0], dtype=np.float32),
        intrinsic_reward=np.asarray([-0.5, -1.0], dtype=np.float32),
        pred_failure=np.asarray([1.0, 1.0], dtype=np.float32),
        threshold=0.0,
        ckpt_path="disc.pth",
        task_name="Task",
        video_fps=20,
        annotation_mask=np.asarray([True, False]),
    )

    np.testing.assert_array_equal(video_masks[0], [True, False])
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["annotated_frames"] == 1
    assert summary["blank_frames"] == 1
    assert summary["predicted_failure_frames"] == 1
    with result.scores_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1] == {
        "step": "1",
        "failure_score": "",
        "intrinsic_reward": "",
        "threshold": "",
        "pred_failure": "",
    }
    with pytest.raises(ValueError, match="does not match scores"):
        disc_viz.visualize_selected_trajectory_discriminator_nnpu(
            output_dir=tmp_path / "mismatch",
            transitions=transitions,
            camera_names=["agentview"],
            failure_score=np.asarray([1.0, 9.0], dtype=np.float32),
            intrinsic_reward=np.asarray([-0.5, -1.0], dtype=np.float32),
            pred_failure=np.asarray([1.0, 1.0], dtype=np.float32),
            threshold=0.0,
            ckpt_path="disc.pth",
            task_name="Task",
            video_fps=20,
            annotation_mask=np.asarray([True, False, True]),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Online advantage parity requires CUDA and never falls back to CPU.",
)
@pytest.mark.parametrize("success_terminal", [False, True])
def test_policy_boundary_metrics_match_training_td1_and_gae(
    success_terminal: bool,
) -> None:
    device = "cuda"
    horizon = 2
    cfg = VASTConfig(
        action_horizon=horizon,
        discount=0.9,
        vast_max_k=1,
        disc_reward_coef=0.1,
        device=device,
    )

    class FakeEncoder:
        def encode_features(self, *, chunk_images, chunk_proprio, chunk_actions):
            state = chunk_proprio.to(device)[..., :1]
            chunk = state + chunk_actions.to(device)[..., :1]
            return state, chunk

        def encode_state(self, *, image_obs_raw, proprio_raw):
            return proprio_raw.to(device)[..., :1]

        def encode_state_and_chunk(self, *, image_obs_raw, proprio_raw, action_chunk):
            state = proprio_raw.to(device)[..., :1]
            return state, state + action_chunk.to(device)[:, 0, :1]

    class FakeDiscriminator:
        threshold = 0.0

        def failure_score(self, *, chunk_feature):
            return chunk_feature[..., 0]

        def intrinsic_reward(self, *, chunk_feature):
            return -torch.sigmoid(self.failure_score(chunk_feature=chunk_feature))

    class FakeLearner:
        def __init__(self):
            self.cfg = SimpleNamespace(device=device)
            self.g = object()

        def v(self, state):
            base = state[..., :1]
            return torch.cat([base, base + 0.2], dim=-1)

        @staticmethod
        def _head_mean(values):
            return values.mean(dim=-1, keepdim=True)

        def v_value(self, state):
            return self._head_mean(self.v(state))

        @staticmethod
        def target_v_value(state):
            return 0.5 * state[..., :1]

        def compute_td_advantage(self, state, next_state, rewards, dones):
            return (
                rewards
                + (cfg.discount ** cfg.action_horizon)
                * (1.0 - dones)
                * self.target_v_value(next_state)
                - self.v_value(state)
            )

        @staticmethod
        def g_value(current_state, future_state, k):
            return torch.zeros_like(current_state[..., :1])

    transitions: list[Transition] = []
    for step in range(5):
        obs = {
            "state": np.asarray([float(step)], dtype=np.float32),
            "agentview": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        next_obs = {
            "state": np.asarray([float(step + 1)], dtype=np.float32),
            "agentview": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        transitions.append(
            Transition(
                obs=obs,
                action=np.asarray([0.25], dtype=np.float32),
                reward=0.0 if success_terminal and step == 4 else -1.0,
                next_obs=next_obs,
                done=step == 4,
                info={
                    "episode_index": 0,
                    "episode_step": step,
                    "success": bool(success_terminal and step == 4),
                },
            )
        )
    if success_terminal:
        transitions, synthetic_count, _ = clone_with_absorbing_success_tail(
            transitions,
            horizon,
        )
        assert synthetic_count == horizon - 1
    encoder = FakeEncoder()
    discriminator = FakeDiscriminator()
    learner = FakeLearner()
    rows, _ = renderer.compute_metrics(
        transitions,
        learner=learner,
        encoder=encoder,
        discriminator=discriminator,
        cfg=cfg,
        camera_names=["agentview"],
        batch_size=8,
        max_windows=None,
        use_disc_reward=True,
        gae_lambda=0.6,
        boundary_semantics="policy_update",
    )
    buffer = FlowDaggerReplayBuffer(
        ReplayBufferConfig(capacity=100, batch_size=8),
        name="online-parity",
        camera_names=["agentview"],
        action_horizon=horizon,
        image_size=2,
    )
    buffer.extend(transitions)

    td1, _, _ = precompute_offline_advantage(
        base_buffer=buffer,
        vast_learner=learner,
        encoder=encoder,
        discriminator=discriminator,
        vast_cfg=cfg,
        device=device,
        encode_batch_size=8,
        estimator="td1",
        gae_lambda=0.6,
    )
    gae, _, _ = precompute_offline_advantage(
        base_buffer=buffer,
        vast_learner=learner,
        encoder=encoder,
        discriminator=discriminator,
        vast_cfg=cfg,
        device=device,
        encode_batch_size=8,
        estimator="gae",
        gae_lambda=0.6,
    )

    actual_td1 = torch.tensor(
        [row["advantage_td1"] for row in rows],
        device=device,
    )
    actual_gae = torch.tensor(
        [row["advantage_gae"] for row in rows],
        device=device,
    )
    assert torch.allclose(actual_td1, td1, atol=1e-6)
    assert torch.allclose(actual_gae, gae, atol=1e-6)
    for row in rows:
        assert row["total_reward_horizon"] == pytest.approx(
            row["env_reward_horizon"] + 0.1 * row["disc_reward_horizon"]
        )
        assert row["g_mc_error"] == pytest.approx(
            -row["total_reward_horizon"]
        )
    if success_terminal:
        assert rows[-1]["done_chunk"] == 1.0
        assert rows[-1]["bootstrap_v"] == 0.0
        assert rows[-1]["has_valid_next"] == 0.0
    else:
        assert rows[-1]["done_chunk"] == 0.0
        assert rows[-1]["bootstrap_v"] != 0.0
        assert rows[-1]["has_valid_next"] == 1.0
