from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from robosuite.pipeline.src.environment.flow import (
    FlowContext,
    FlowObservation,
    FlowPolicyAdapter,
)
from robosuite.pipeline.src.vision import DinoV2Encoder


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class TinyDino(nn.Module):
    embed_dim = 768
    num_register_tokens = 0

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.embed_dim, bias=False)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = images.mean(dim=(-2, -1))
        return {"x_norm_clstoken": self.projection(pooled)}


class ConstantFlowHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.ones_like(x_t) * self.velocity


class TinyFlow(nn.Module):
    def __init__(self, camera_names: list[str], action_dim: int) -> None:
        super().__init__()
        self.camera_names = camera_names
        self.action_dim = action_dim
        self.feature_dim = 4
        self.image_encoder = TinyImageEncoder()
        self.context_projection = nn.Linear(2, 4)
        self.flow_head = ConstantFlowHead()

    def encode_multimodal_context(
        self, images: torch.Tensor, proprio: torch.Tensor, language: list[str]
    ) -> dict[str, torch.Tensor]:
        del language
        context = self.context_projection(proprio)
        batch_size, camera_count = images.shape[:2]
        flat_images = images.flatten(0, 1)
        image_tokens = self.image_encoder(flat_images).reshape(
            batch_size, camera_count * self.image_encoder.num_tokens, self.feature_dim
        )
        return {
            "task_scene_cond": context,
            "context_tokens": context[:, None],
            "context_padding_mask": torch.zeros(
                (context.shape[0], 1), dtype=torch.bool, device=context.device
            ),
            "image_tokens": image_tokens,
        }


class TinyImageEncoder(nn.Module):
    num_tokens = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1))
        token = self.projection(pooled)
        return torch.stack((token, token + 1.0), dim=1)


def _flow_builder(
    _cfg: dict, proprio_dim: int, action_dim: int, camera_names: list[str]
) -> TinyFlow:
    assert proprio_dim == 2
    return TinyFlow(camera_names, action_dim)


def _write_flow_checkpoint(path: Path, *, corrupt_state: bool = False) -> None:
    model = TinyFlow(["a", "b", "c"], action_dim=2)
    state = model.state_dict()
    if corrupt_state:
        state.pop("flow_head.velocity")
    torch.save(
        {
            "ema_model": state,
            "model_cfg": {},
            "camera_names": ["a", "b", "c"],
            "act_mean": np.full((3, 2), 2.0, dtype=np.float32),
            "act_std": np.full((3, 2), 3.0, dtype=np.float32),
            "prop_mean": np.array([1.0, 2.0], dtype=np.float32),
            "prop_std": np.array([2.0, 4.0], dtype=np.float32),
            "cfg": {"data": {"image_size": 16}},
        },
        path,
    )


def test_dinov2_strict_load_resize_normalize_and_freeze(tmp_path: Path) -> None:
    weights = tmp_path / "tiny_dino.pt"
    torch.save(TinyDino().state_dict(), weights)
    encoder = DinoV2Encoder(
        weights,
        "cuda:0",
        backbone_factory=TinyDino,
    )
    images = torch.randint(0, 256, (2, 3, 3, 31, 47), dtype=torch.uint8, device="cuda:0")
    features = encoder(images)

    assert features.shape == (2, 3, 768)
    assert features.device.type == "cuda"
    assert not features.requires_grad
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_dinov2_rejects_register_tokens_and_cpu_inputs(tmp_path: Path) -> None:
    weights = tmp_path / "tiny_dino.pt"
    torch.save(TinyDino().state_dict(), weights)
    encoder = DinoV2Encoder(weights, "cuda:0", backbone_factory=TinyDino)
    with pytest.raises(ValueError, match="must be on CUDA"):
        encoder(torch.zeros(1, 3, 3, 16, 16, dtype=torch.uint8))

    class RegisteredDino(TinyDino):
        num_register_tokens = 4

    with pytest.raises(ValueError, match="must not use register tokens"):
        DinoV2Encoder(weights, "cuda:0", backbone_factory=RegisteredDino)


def test_flow_external_noise_context_cache_and_action_clamp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow.pt"
    _write_flow_checkpoint(checkpoint)
    adapter = FlowPolicyAdapter(
        checkpoint,
        "cuda:0",
        expected_camera_names=("a", "b", "c"),
        expected_action_horizon=3,
        expected_action_dim=2,
        ode_steps=2,
        action_low=-4.0,
        action_high=4.0,
        model_builder=_flow_builder,
    )
    observation = FlowObservation(
        images=torch.zeros((2, 3, 3, 20, 16), dtype=torch.uint8, device="cuda:0"),
        proprio=torch.tensor([[1.0, 2.0], [3.0, 6.0]], device="cuda:0"),
        language=["first", "second"],
    )
    context, combined_image_tokens = adapter.encode_context_with_image_tokens(observation)
    image_tokens = adapter.encode_image_tokens(observation.images)
    preprocessed_tokens = adapter.encode_image_tokens(
        adapter.preprocess_images(observation.images), images_preprocessed=True
    )
    restored = FlowContext.from_mapping(context.to_mapping())
    stacked = FlowContext.stack(
        [
            FlowContext(*(value[:1] for value in restored.to_mapping().values())),
            FlowContext(*(value[1:] for value in restored.to_mapping().values())),
        ]
    )
    noise = torch.zeros((2, 3, 2), device="cuda:0")

    normalized = adapter.decode_noise(stacked, noise, denormalize=False)
    actions = adapter.decode_noise(stacked, noise)
    projected = adapter.project_normalized_actions(normalized)

    torch.testing.assert_close(normalized, torch.ones_like(normalized))
    torch.testing.assert_close(actions, torch.full_like(actions, 4.0))
    torch.testing.assert_close(projected, torch.full_like(projected, 2.0 / 3.0))
    torch.testing.assert_close(image_tokens, preprocessed_tokens)
    torch.testing.assert_close(image_tokens, combined_image_tokens)
    assert image_tokens.shape == (2, 3, 2, 4)
    assert image_tokens.device.type == "cuda"
    assert not image_tokens.requires_grad
    assert adapter.image_tokens_per_camera == 2
    assert adapter.image_token_dim == 4
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())


def test_flow_strict_checkpoint_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow.pt"
    _write_flow_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="camera mismatch"):
        FlowPolicyAdapter(
            checkpoint,
            "cuda:0",
            expected_camera_names=("wrong",),
            model_builder=_flow_builder,
        )
    with pytest.raises(ValueError, match="horizon mismatch"):
        FlowPolicyAdapter(
            checkpoint,
            "cuda:0",
            expected_action_horizon=8,
            model_builder=_flow_builder,
        )

    corrupt = tmp_path / "corrupt.pt"
    _write_flow_checkpoint(corrupt, corrupt_state=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        FlowPolicyAdapter(corrupt, "cuda:0", model_builder=_flow_builder)


def test_cuda_only_device_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow.pt"
    _write_flow_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        FlowPolicyAdapter(checkpoint, "cpu", model_builder=_flow_builder)
