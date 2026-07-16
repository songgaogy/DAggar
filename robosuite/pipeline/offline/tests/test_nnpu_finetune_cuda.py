"""CUDA-only tests for warm-start nnPU finetuning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robosuite.discriminator.dyn_disc.detectors.pu_bce import PUBCEDiscriminator
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
import robosuite.pipeline.offline.discriminator.objectives as objectives_module
from robosuite.pipeline.offline.discriminator import (
    PUBCEDiscriminatorFT,
    build_finetuned_checkpoint_payload,
    encode_policy_segments,
    load_pretrain_pools,
    load_warmstart_detector,
    require_cuda_device,
    save_finetuned_checkpoint,
    split_policy_segments,
    validate_offline_payload,
)
from robosuite.pipeline.offline.discriminator.episodes import build_gt_negative_windows
from robosuite.pipeline.offline.discriminator.features import (
    encode_gt_negative_windows,
    exclude_gt_frames_from_unlabeled,
)
from robosuite.pipeline.offline.discriminator.pools import LatentTrajectory


POSITIVE_SAFETY_BOUNDARY = 1.25


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nnPU finetune tests.")
    return torch.device("cuda:0")


def _feature_pools(
    device: torch.device,
) -> tuple[dict[str, list[torch.Tensor]], list[torch.Tensor]]:
    positive = torch.randn((24, 6), device=device) + 1.0
    unlabeled = torch.randn((24, 6), device=device) - 0.5
    calibration = [torch.randn((12, 6), device=device) + 1.0]
    return {
        "pretrain_positive": [positive],
        "pretrain_unlabeled": [unlabeled],
        "offline_positive": [positive.clone()],
        "offline_gt_negative": [unlabeled.clone()],
    }, calibration


def _objective_config(batch_size: int = 8) -> dict[str, object]:
    return {
        "steps_per_epoch": 1,
        "terms": {
            "nnpu_replay": {
                "type": "nnpu",
                "enabled": True,
                "weight": 1.0,
                "batch_size": batch_size,
                "positive_fraction": 0.5,
            },
            "gt_positive": {
                "type": "positive_safety_margin",
                "enabled": True,
                "weight": 1.0,
                "batch_size": batch_size // 2,
                "safety_margin_weight": 1.0,
                "margin_delta": 1.0,
                "temperature": 1.0,
                "boundary_source": "parent_checkpoint",
            },
            "gt_negative": {
                "type": "negative_logistic",
                "enabled": True,
                "weight": 1.0,
                "batch_size": batch_size // 2,
            },
        },
    }


def test_finetune_keeps_loaded_head_when_lr_is_zero() -> None:
    device = _cuda()
    feature_pools, calibration = _feature_pools(device)
    with torch.device(device):
        detector = PUBCEDiscriminatorFT(
            in_dim=6, hidden=8, num_layers=1, device=str(device)
        )
    detector.thresholds = {"Task": 1.0e9}
    before = {
        name: parameter.detach().clone()
        for name, parameter in detector.head.state_dict().items()
    }
    logged: list[dict[str, float]] = []

    thresholds = detector.finetune(
        feature_pools,
        {"Task": calibration},
        objective_config=_objective_config(),
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
        pi_p=0.3,
        epochs=1,
        lr=0.0,
        seed=7,
        loss_surrogate="logistic",
        metric_callback=logged.append,
        verbose=False,
    )

    for name, parameter in detector.head.state_dict().items():
        torch.testing.assert_close(parameter, before[name])
    assert "Task" in thresholds
    assert thresholds["Task"] != 1.0e9
    assert len(logged) == 1
    assert logged[0]["data/pretrain_positive_frames"] == 24.0
    assert logged[0]["data/pretrain_unlabeled_frames"] == 24.0
    assert logged[0]["data/offline_gt_negative_frames"] == 24.0
    assert logged[0]["safety/m_k"] == pytest.approx(POSITIVE_SAFETY_BOUNDARY)
    assert logged[0]["safety/target_logit"] == pytest.approx(
        POSITIVE_SAFETY_BOUNDARY + 1.0
    )


def test_log_interval_reports_running_mean() -> None:
    device = _cuda()
    feature_pools, calibration = _feature_pools(device)
    with torch.device(device):
        detector = PUBCEDiscriminatorFT(
            in_dim=6, hidden=8, num_layers=1, device=str(device)
        )
    config = _objective_config()
    config["steps_per_epoch"] = 2
    interval_logs: list[dict[str, float]] = []
    epoch_logs: list[dict[str, float]] = []

    detector.finetune(
        feature_pools,
        {"Task": calibration},
        objective_config=config,
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
        pi_p=0.3,
        epochs=1,
        lr=0.0,
        seed=3,
        loss_surrogate="logistic",
        log_interval=2,
        metric_callback=epoch_logs.append,
        step_metric_callback=interval_logs.append,
        verbose=False,
    )

    assert len(interval_logs) == 1
    assert len(epoch_logs) == 1
    assert interval_logs[0]["loss/total"] == pytest.approx(
        epoch_logs[0]["loss/total"]
    )


def test_finetune_updates_head_and_checkpoint_remains_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _cuda()
    feature_pools, calibration = _feature_pools(device)
    with torch.device(device):
        detector = PUBCEDiscriminatorFT(
            in_dim=6, hidden=8, num_layers=1, device=str(device)
        )
    detector.thresholds = {"Task": 0.0}
    parent_state = detector.state_dict()
    before = {
        name: parameter.detach().clone()
        for name, parameter in detector.head.state_dict().items()
    }

    sampled_sizes: list[tuple[int, int]] = []
    original_pu_risk = objectives_module.pu_risk

    def recording_pu_risk(g_p: torch.Tensor, g_u: torch.Tensor, **kwargs):
        sampled_sizes.append((int(g_p.shape[0]), int(g_u.shape[0])))
        return original_pu_risk(g_p, g_u, **kwargs)

    monkeypatch.setattr(objectives_module, "pu_risk", recording_pu_risk)
    detector.finetune(
        feature_pools,
        {"Task": calibration},
        objective_config=_objective_config(batch_size=10),
        positive_safety_boundary=POSITIVE_SAFETY_BOUNDARY,
        pi_p=0.3,
        epochs=1,
        lr=1e-3,
        seed=11,
        loss_surrogate="logistic",
        verbose=False,
    )
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in detector.head.state_dict().items()
    )
    assert sampled_sizes
    assert set(sampled_sizes) == {(5, 5)}

    parent_payload = {
        "in_dim": 6,
        "hidden": 8,
        "num_layers": 1,
        "pu_bce_detector": parent_state,
        "feature_source": "transformer",
        "transformer_layer": 1,
        "use_chunk": True,
        "model_ckpt": "unused.pth",
        "pi_p": 0.3,
        "loss_surrogate": "logistic",
        "nn_correction": True,
        "beta": 0.0,
    }
    payload = build_finetuned_checkpoint_payload(
        detector,
        parent_payload=parent_payload,
        parent_checkpoint=tmp_path / "parent.pth",
        task_name="Task",
        finetune_config={
            "method": "nnpu_replay_positive_safety_margin_gt_negative",
            "epochs": 1,
            "lr": 1e-3,
            "resolved_positive_safety_margin": {
                "task": "Task",
                "boundary_source": "parent_checkpoint",
                "parent_failure_threshold": -POSITIVE_SAFETY_BOUNDARY,
                "m_k": POSITIVE_SAFETY_BOUNDARY,
                "margin_delta": 1.0,
                "target_logit": POSITIVE_SAFETY_BOUNDARY + 1.0,
                "safety_margin_weight": 1.0,
                "temperature": 1.0,
            },
        },
        data_provenance={
            "positive_frames": 24,
            "unlabeled_frames": 24,
            "pretrain_manifest": {
                "checkpoint": {
                    "model_ckpt_sha256": "model-sha",
                    "normalizer_ckpt": "/tmp/normalizer.pth",
                    "normalizer_ckpt_sha256": "normalizer-sha",
                },
                "feature_contract": {
                    "camera_to_view": {"camera": "view"},
                    "proprio_indices": [1, 2],
                },
                "split_config": {"seed": 0, "calibration_fraction": 0.2},
                "trajectory_ids": {
                    "positive_train": ["positive-train"],
                    "positive_calib": ["positive-calib"],
                    "unlabeled_train": ["unlabeled"],
                },
            },
        },
    )
    assert payload["model_ckpt_sha256"] == "model-sha"
    assert payload["normalizer_ckpt_sha256"] == "normalizer-sha"
    assert payload["camera_to_view"] == {"camera": "view"}
    assert payload["proprio_indices"] == [1, 2]
    assert payload["success_train_video_ids"] == {"Task": ["positive-train"]}
    assert payload["success_calib_video_ids"] == {"Task": ["positive-calib"]}
    assert payload["finetune_schema_version"] == 4
    assert payload["finetune_method"] == (
        "nnpu_replay_positive_safety_margin_gt_negative"
    )
    assert payload["finetune_config"]["resolved_positive_safety_margin"]["m_k"] == (
        POSITIVE_SAFETY_BOUNDARY
    )
    checkpoint = save_finetuned_checkpoint(
        payload,
        tmp_path / "pu_bce_head_finetuned.pth",
    )
    restored, restored_payload = load_warmstart_detector(
        checkpoint,
        device=device,
        expected_task="Task",
    )
    assert isinstance(restored, PUBCEDiscriminatorFT)
    assert restored_payload["finetuned_offline"] is True
    assert restored.thresholds["Task"] == pytest.approx(detector.thresholds["Task"])
    probe = torch.randn((5, 6), device=device)
    torch.testing.assert_close(
        restored.failure_score_tensor(probe),
        detector.failure_score_tensor(probe),
    )
    frozen = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=checkpoint,
        task_name="Task",
        device=device,
    )
    torch.testing.assert_close(
        frozen.failure_score(probe),
        detector.failure_score_tensor(probe),
    )

    legacy_payload = dict(payload)
    legacy_payload["finetune_schema_version"] = 2
    legacy_payload["finetune_method"] = "nnpu_replay_gt_bce"
    legacy_checkpoint = save_finetuned_checkpoint(
        legacy_payload,
        tmp_path / "pu_bce_head_finetuned_v2.pth",
    )
    legacy_detector, legacy_restored_payload = load_warmstart_detector(
        legacy_checkpoint,
        device=device,
        expected_task="Task",
    )
    assert legacy_restored_payload["finetune_schema_version"] == 2
    torch.testing.assert_close(
        legacy_detector.failure_score_tensor(probe),
        detector.failure_score_tensor(probe),
    )


def test_finetune_rejects_cpu_before_touching_features() -> None:
    detector = object.__new__(PUBCEDiscriminatorFT)
    detector.device = torch.device("cpu")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        detector.finetune({}, {}, objective_config={}, pi_p=0.3)


def test_public_device_guards_reject_cpu() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        require_cuda_device("cpu")


class _FakeFrozenEncoder:
    def __init__(self, device: torch.device) -> None:
        model = torch.nn.Linear(1, 1, bias=False, device=device)
        model.requires_grad_(False)
        self.device = device
        self.feature_source = "transformer"
        self.transformer_layer = 1
        self.use_chunk = True
        self.inner_encoder = SimpleNamespace(
            model=model,
            frameskip=3,
            action_dim_per_step=2,
            action_input_dim=6,
        )
        self.bound_cameras: list[str] | None = None
        self.seen_actions: list[torch.Tensor] = []

    @staticmethod
    def prepare_proprio(states: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(states, dtype=np.float32)

    def bind_policy_cameras(self, cameras: list[str]) -> None:
        self.bound_cameras = list(cameras)

    def encode_chunk(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        assert image_obs_raw.device.type == "cuda"
        assert proprio_raw.device.type == "cuda"
        assert action_chunk.device.type == "cuda"
        self.seen_actions.append(action_chunk.detach().clone())
        return action_chunk


def _encoding_payload() -> dict[str, object]:
    length = 4
    actions = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [50.0, 60.0], [7.0, 8.0]],
        dtype=np.float32,
    )
    policy = actions.copy()
    policy[2] = [5.0, 6.0]
    interventions = np.asarray([False, False, True, False], dtype=np.bool_)
    success = np.asarray([False, False, False, True], dtype=np.bool_)
    done = np.asarray([False, False, False, True], dtype=np.bool_)
    state = np.zeros((length, 3), dtype=np.float32)
    image = np.zeros((length, 4, 5, 3), dtype=np.uint8)
    episode = {
        "obs": {"state": state, "agentview": image},
        "next_obs": {"state": state, "agentview": image},
        "executed_action": actions,
        "policy_action": policy,
        "is_intervention": interventions,
        "success": success,
        "done": done,
        "terminal_reason": "success",
    }
    return {
        "schema_version": 1,
        "task_name": "Task",
        "camera_names": ["agentview"],
        "episodes": [episode],
    }


def test_cuda_encoding_freezes_encoder_and_never_crosses_segment_boundaries() -> None:
    device = _cuda()
    payload = validate_offline_payload(_encoding_payload())
    segments, _ = split_policy_segments(payload)
    encoder = _FakeFrozenEncoder(device)
    before = encoder.inner_encoder.model.weight.detach().clone()

    pools = encode_policy_segments(
        segments,
        encoder=encoder,
        camera_names=["agentview"],
        batch_size=16,
    )

    assert encoder.bound_cameras == ["agentview"]
    assert len(pools.unlabeled) == 1
    assert len(pools.positive) == 1
    assert pools.unlabeled[0].source == "offline"
    assert pools.positive[0].source == "offline"
    torch.testing.assert_close(encoder.inner_encoder.model.weight, before)
    assert encoder.inner_encoder.model.weight.grad is None
    assert all(
        not parameter.requires_grad
        for parameter in encoder.inner_encoder.model.parameters()
    )
    expected_first = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
            [3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    expected_second = torch.tensor(
        [[7.0, 8.0, 7.0, 8.0, 7.0, 8.0]],
        dtype=torch.float32,
        device=device,
    )
    torch.testing.assert_close(encoder.seen_actions[0], expected_first)
    torch.testing.assert_close(encoder.seen_actions[1], expected_second)


def test_cuda_gt_negative_encoding_uses_policy_actions_across_onset() -> None:
    device = _cuda()
    payload = validate_offline_payload(_encoding_payload())
    windows, _ = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=1,
        post_intervention_chunks=1,
        frameskip=3,
    )
    encoder = _FakeFrozenEncoder(device)

    trajectories = encode_gt_negative_windows(
        windows,
        encoder=encoder,
        camera_names=["agentview"],
        batch_size=16,
    )

    assert len(trajectories) == 1
    assert trajectories[0].pool == "offline_gt_negative"
    assert trajectories[0].source == "offline"
    assert trajectories[0].metadata["action_source"] == "policy_action"
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 5.0, 6.0],
            [5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    torch.testing.assert_close(encoder.seen_actions[0], expected)
    torch.testing.assert_close(trajectories[0].features.to(device), expected)
    assert not torch.any(encoder.seen_actions[0] == 50.0)


def test_cuda_reserved_unlabeled_excludes_selected_pre_onset_frames() -> None:
    device = _cuda()
    payload = validate_offline_payload(
        {
            **_encoding_payload(),
            "episodes": [
                {
                    **_encoding_payload()["episodes"][0],  # type: ignore[index]
                    "is_intervention": np.asarray(
                        [False, False, True, True], dtype=np.bool_
                    ),
                    "success": np.asarray([False] * 4, dtype=np.bool_),
                    "terminal_reason": "manual_reset",
                }
            ],
        }
    )
    windows, _ = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=1,
        post_intervention_chunks=1,
        frameskip=1,
    )
    features = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 6)
    trajectory = LatentTrajectory(
        features=features,
        pool="unlabeled",
        source="offline",
        identifier="episode-0-policy-prefix",
        metadata={
            "source_episode_index": 0,
            "frame_start": 0,
            "frame_end": 2,
        },
    )

    reserved = exclude_gt_frames_from_unlabeled([trajectory], windows)

    assert len(reserved) == 1
    assert reserved[0].pool == "offline_unlabeled_reserved"
    assert reserved[0].metadata["frame_start"] == 0
    assert reserved[0].metadata["frame_end"] == 1
    assert reserved[0].features.device.type == "cuda"
    torch.testing.assert_close(reserved[0].features, features[:1])


def test_pretrain_manifest_and_shards_round_trip_from_cuda(tmp_path: Path) -> None:
    device = _cuda()
    splits: dict[str, list[dict[str, object]]] = {}
    for split_name in ("positive_train", "positive_calib", "unlabeled_train"):
        split_dir = tmp_path / split_name
        split_dir.mkdir()
        latent = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        shard_path = split_dir / "trajectory.pt"
        torch.save(
            {
                "schema_version": 1,
                "latent": latent,
                "frame_indices": np.arange(3, dtype=np.int64),
                "provenance": {"split": split_name},
            },
            shard_path,
        )
        splits[split_name] = [
            {
                "path": f"{split_name}/trajectory.pt",
                "video_id": f"{split_name}-id",
                "num_frames": 3,
                "latent_dim": 4,
                "provenance": {"split": split_name},
            }
        ]
    manifest = {
        "schema_version": 1,
        "task": "Task",
        "splits": splits,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pools, restored_manifest = load_pretrain_pools(tmp_path)

    assert restored_manifest == manifest
    assert pools.stats == {
        "positive_trajectories": 1,
        "positive_frames": 3,
        "unlabeled_trajectories": 1,
        "unlabeled_frames": 3,
        "calibration_trajectories": 1,
        "calibration_frames": 3,
        "latent_dim": 4,
    }
    assert pools.positive[0].features.device.type == "cpu"


def test_legacy_checkpoint_without_new_provenance_remains_loadable(tmp_path: Path) -> None:
    device = _cuda()
    with torch.device(device):
        detector = PUBCEDiscriminator(in_dim=4, hidden=8, num_layers=1, device=str(device))
    detector.thresholds = {"Task": 0.25}
    legacy_payload = {
        "in_dim": 4,
        "hidden": 8,
        "num_layers": 1,
        "feature_source": "transformer",
        "transformer_layer": 1,
        "model_ckpt": "model.pth",
        "pi_p": 0.3,
        "loss_surrogate": "logistic",
        "nn_correction": True,
        "beta": 0.0,
        "pu_bce_detector": detector.state_dict(),
    }
    checkpoint = tmp_path / "legacy.pth"
    torch.save(legacy_payload, checkpoint)

    restored, payload = load_warmstart_detector(
        checkpoint,
        device=device,
        expected_task="Task",
    )

    assert "finetune_schema_version" not in payload
    assert restored.thresholds == {"Task": pytest.approx(0.25)}


def test_single_task_finetune_rejects_multi_task_parent(tmp_path: Path) -> None:
    device = _cuda()
    with torch.device(device):
        detector = PUBCEDiscriminator(in_dim=4, hidden=8, num_layers=1, device=str(device))
    detector.thresholds = {"Task": 0.25, "OtherTask": 0.5}
    checkpoint = tmp_path / "multi-task.pth"
    torch.save(
        {
            "in_dim": 4,
            "hidden": 8,
            "num_layers": 1,
            "feature_source": "transformer",
            "transformer_layer": 1,
            "model_ckpt": "model.pth",
            "pi_p": 0.3,
            "pu_bce_detector": detector.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="single-task"):
        load_warmstart_detector(checkpoint, device=device, expected_task="Task")
