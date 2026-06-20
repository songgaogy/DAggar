from __future__ import annotations

from pathlib import Path

import pytest
import torch

from robosuite.discriminator.dyn_disc.detectors import DynEncoder, PUBCEDiscriminator
from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.discriminator.offline import (
    nnpu_intrinsic_from_failure_score,
)
from robosuite.pipeline.algorithms.discriminator.runtime import (
    NNPUDiscriminatorRuntime,
    NNPURuntimeConfig,
)


def _write_checkpoint(path: Path) -> PUBCEDiscriminator:
    detector = PUBCEDiscriminator(in_dim=4, hidden=8, num_layers=1, device="cpu")
    detector.thresholds = {"Task": 0.25}
    torch.save(
        {
            "in_dim": 4,
            "hidden": 8,
            "num_layers": 1,
            "pu_bce_detector": detector.state_dict(),
            "feature_source": "transformer",
            "transformer_layer": 1,
            "model_ckpt": "unused.pth",
        },
        path,
    )
    return detector


def test_frozen_nnpu_uses_checkpoint_threshold_and_tensor_scores(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pu_bce_head.pth"
    source = _write_checkpoint(checkpoint)
    frozen = FrozenNNPUDiscriminator(checkpoint, task_name="Task", device="cpu")
    features = torch.randn(2, 3, 4)

    expected_score = source.failure_score_tensor(features)
    output = frozen.score(chunk_feature=features)

    assert frozen.threshold == pytest.approx(0.25)
    assert output.logit.shape == (2, 3)
    torch.testing.assert_close(output.logit, expected_score)
    torch.testing.assert_close(
        frozen.intrinsic_reward(chunk_feature=features),
        -torch.sigmoid(expected_score - 0.25),
    )
    assert all(not parameter.requires_grad for parameter in frozen.detector.head.parameters())


def test_frozen_nnpu_rejects_legacy_bce_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bce_head.pth"
    torch.save({"bce_detector": {}}, checkpoint)
    with pytest.raises(KeyError, match="Legacy BCE"):
        FrozenNNPUDiscriminator(checkpoint, task_name="Task", device="cpu")


def test_frozen_nnpu_requires_calibrated_task_threshold(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pu_bce_head.pth"
    _write_checkpoint(checkpoint)
    with pytest.raises(KeyError, match="has no nnPU threshold"):
        FrozenNNPUDiscriminator(checkpoint, task_name="OtherTask", device="cpu")


def test_shared_encoder_rejects_incomplete_nnpu_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pu_bce_head.pth"
    torch.save({"pu_bce_detector": {}}, checkpoint)
    with pytest.raises(KeyError, match="feature_source.*transformer_layer.*model_ckpt"):
        SharedDynamicsEncoder(checkpoint, device="cpu")


def test_nnpu_intrinsic_formula_numpy_and_tensor() -> None:
    score = torch.tensor([-1.0, 0.5, 2.0])
    expected = -torch.sigmoid(score - 0.5)
    torch.testing.assert_close(nnpu_intrinsic_from_failure_score(score, 0.5), expected)
    assert nnpu_intrinsic_from_failure_score(0.5, 0.5) == pytest.approx(-0.5)


def test_pubce_tensor_api_preserves_leading_shape() -> None:
    detector = PUBCEDiscriminator(in_dim=5, hidden=8, num_layers=1, device="cpu")
    features = torch.randn(2, 3, 5)
    logits = detector.logits_tensor(features)
    failure = detector.failure_score_tensor(features)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(failure, -logits)


def test_action_window_padding_and_truncation_match_training_contract() -> None:
    encoder = object.__new__(DynEncoder)
    encoder.device = torch.device("cpu")
    encoder.frameskip = 3
    encoder.action_dim_per_step = 2
    encoder.action_input_dim = 6

    short = torch.tensor([[[1.0], [2.0]]])
    padded = encoder._prepare_action_chunks_tensor(short, batch_size=1)
    torch.testing.assert_close(
        padded,
        torch.tensor([[1.0, 0.0, 2.0, 0.0, 2.0, 0.0]]),
    )

    long = torch.tensor([[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0], [7.0, 8.0, 9.0]]])
    truncated = encoder._prepare_action_chunks_tensor(long, batch_size=1)
    torch.testing.assert_close(
        truncated,
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )


def test_shared_encoder_camera_binding_is_stable() -> None:
    encoder = object.__new__(SharedDynamicsEncoder)
    encoder._view_names = ["front", "wrist"]
    encoder.camera_to_view = {"agentview": "front", "eye_in_hand": "wrist"}
    encoder._policy_cameras = None
    encoder._policy_camera_indices = None

    encoder.bind_policy_cameras(["eye_in_hand", "agentview", "unused"])
    assert encoder._policy_camera_indices == [1, 0]
    encoder.bind_policy_cameras(["eye_in_hand", "agentview", "unused"])
    with pytest.raises(RuntimeError, match="cannot rebind"):
        encoder.bind_policy_cameras(["agentview", "eye_in_hand"])


def test_runtime_chunk_cadence_and_pause_rearm() -> None:
    class Encoder:
        view_names = ["camera"]

        def bind_policy_cameras(self, names):
            self.bound = list(names)

    class Discriminator:
        threshold = 0.5

    cfg = NNPURuntimeConfig(
        fps=0.0, intervene_env=True, consecutive_fail_frames=2
    )
    runtime = NNPUDiscriminatorRuntime(
        cfg, Encoder(), Discriminator(), policy_camera_names=["camera"]
    )
    kwargs = {
        "images_per_view": {"camera": torch.zeros(4, 4, 3).numpy()},
        "proprio": torch.zeros(2).numpy(),
        "executed_action": torch.zeros(2).numpy(),
    }
    runtime.publish(**kwargs, is_new_chunk=False)
    assert runtime._sequence == 0
    runtime.publish(**kwargs, is_new_chunk=True)
    assert runtime._sequence == 1

    runtime._update_debounce(1, 0.8)
    assert not runtime.pause_requested()
    runtime._update_debounce(1, 0.9)
    assert runtime.pause_requested()
    runtime.resume()
    assert not runtime.pause_requested()
    assert runtime.status().armed
