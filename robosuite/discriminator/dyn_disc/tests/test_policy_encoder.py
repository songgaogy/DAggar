"""CUDA-only unit tests for the frozen policy feature encoder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from robosuite.discriminator.dyn_disc.detectors import policy_encoder as policy_module
from robosuite.discriminator.dyn_disc.detectors.policy_encoder import (
    LATENT_NAME,
    PREPROCESS_VERSION,
    PolicyFeatureEncoder,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Policy encoder tensor tests require CUDA.",
)
CUDA = torch.device("cuda")


class _DummyAggregator:
    output_dim = 256

    def __init__(self, model: "_DummyPolicy") -> None:
        self.model = model

    def __call__(
        self,
        *,
        fused_tokens: torch.Tensor,
        token_padding_mask: torch.Tensor,
        language_global: torch.Tensor,
    ) -> torch.Tensor:
        del token_padding_mask
        return fused_tokens.mean(dim=1) + language_global + self.model.scale


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), device=CUDA))
        self.condition_aggregator = _DummyAggregator(self)
        self.language_calls = 0
        self.strict_load = None

    def load_state_dict(self, state_dict, strict: bool = True):
        self.strict_load = bool(strict)
        return super().load_state_dict(state_dict, strict=strict)

    def image_encoder(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1)).mean(dim=1, keepdim=True)
        return pooled.unsqueeze(-1).expand(-1, 1, 256)

    def proprio_tokenizer(self, proprio: torch.Tensor) -> torch.Tensor:
        pooled = proprio.mean(dim=1, keepdim=True)
        return pooled.unsqueeze(-1).expand(-1, 1, 256)

    def language_encoder(self, prompts: list[str]):
        self.language_calls += 1
        batch_size = len(prompts)
        tokens = torch.full((batch_size, 2, 256), 0.25, device=CUDA)
        global_feature = torch.full((batch_size, 256), 0.5, device=CUDA)
        mask = torch.ones((batch_size, 2), dtype=torch.bool, device=CUDA)
        return tokens, global_feature, mask

    @staticmethod
    def language_guided_modulation(
        *,
        visual_tokens: torch.Tensor,
        proprio_tokens: torch.Tensor,
        language_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del language_global
        return visual_tokens, proprio_tokens

    @staticmethod
    def fusion(
        *,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        proprio_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        language_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del language_global
        fused = torch.cat([language_tokens, proprio_tokens, image_tokens], dim=1)
        extra_mask = torch.ones(
            (fused.shape[0], fused.shape[1] - language_mask.shape[1]),
            dtype=torch.bool,
            device=fused.device,
        )
        return fused, torch.cat([language_mask, extra_mask], dim=1)

    def get_cond_features(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str],
    ) -> torch.Tensor:
        batch_size, num_cameras, channels, height, width = images.shape
        image_tokens = self.image_encoder(
            images.reshape(batch_size * num_cameras, channels, height, width)
        )
        image_tokens = image_tokens.reshape(batch_size, num_cameras, 256)
        proprio_tokens = self.proprio_tokenizer(proprio)
        language_tokens, language_global, language_mask = self.language_encoder(language)
        image_tokens, proprio_tokens = self.language_guided_modulation(
            visual_tokens=image_tokens,
            proprio_tokens=proprio_tokens,
            language_global=language_global,
        )
        fused_tokens, padding_mask = self.fusion(
            language_tokens=language_tokens,
            language_mask=language_mask,
            proprio_tokens=proprio_tokens,
            image_tokens=image_tokens,
            language_global=language_global,
        )
        return self.condition_aggregator(
            fused_tokens=fused_tokens,
            token_padding_mask=padding_mask,
            language_global=language_global,
        )


class _DummyExtractor:
    closed = False

    def __init__(self, **kwargs) -> None:
        del kwargs
        model = SimpleNamespace(nq=4, nv=3, na=0)
        self.sim = SimpleNamespace(model=model)
        self.qpos_indices = np.asarray([0, 2], dtype=np.int64)
        self.qvel_indices = np.asarray([1], dtype=np.int64)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch, tmp_path) -> PolicyFeatureEncoder:
    ckpt_path = tmp_path / "policy.pt"
    ckpt_path.touch()
    checkpoint = {
        "ema_model": {"scale": torch.ones((), device=CUDA)},
        "model": {"ignored_non_ema_state": "sentinel"},
        "model_cfg": {"name": "dummy"},
        "camera_names": ["agentview", "robot0_eye_in_hand", "frontview"],
        "prop_mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "prop_std": np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        "act_mean": np.zeros((2, 7), dtype=np.float32),
        "task_prompt_map": {"Stack": ["first prompt", "unused prompt"]},
        "task_metadata_map": {"Stack": {"robots": "Panda"}},
    }
    built_models: list[_DummyPolicy] = []

    def _build_policy(cfg, proprio_dim, action_dim, camera_names):
        assert cfg == {"name": "dummy"}
        assert proprio_dim == 3
        assert action_dim == 7
        assert camera_names == checkpoint["camera_names"]
        model = _DummyPolicy()
        built_models.append(model)
        return model

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        policy_module,
        "_load_policy_builders",
        lambda: (_build_policy, _DummyExtractor),
    )
    result = PolicyFeatureEncoder(str(ckpt_path), device="cuda")
    assert result.model is built_models[0]
    yield result
    result.close()


def test_cuda_only_guard_runs_before_checkpoint_loading() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        PolicyFeatureEncoder("does-not-matter.pt", device="cpu")


def test_checkpoint_schema_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    ckpt_path = tmp_path / "missing.pt"
    ckpt_path.touch()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"ema_model": {}})
    with pytest.raises(KeyError, match="missing required keys"):
        PolicyFeatureEncoder(str(ckpt_path), device="cuda")


def test_checkpoint_uses_ema_and_first_prompt(encoder: PolicyFeatureEncoder) -> None:
    assert encoder.model.strict_load is True
    assert encoder.camera_names == ["agentview", "robot0_eye_in_hand", "frontview"]
    assert encoder.prompt_for_task("Stack") == "first prompt"
    assert not encoder.model.training
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert all(parameter.device.type == "cuda" for parameter in encoder.model.parameters())

    metadata = encoder.metadata()
    assert metadata["policy_weight_source"] == "ema_model"
    assert metadata["feature_source"] == "policy_task_scene_cond"
    assert metadata["latent"] == LATENT_NAME
    assert metadata["feature_dim"] == 256
    assert metadata["dtype"] == "float32"
    assert metadata["preprocess_version"] == PREPROCESS_VERSION


def test_proprio_extraction_and_normalization(encoder: PolicyFeatureEncoder) -> None:
    states = np.asarray(
        [
            [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0, 33.0, 40.0, 41.0, 42.0],
        ],
        dtype=np.float32,
    )
    proprio = encoder.extract_proprio(states, "Stack")
    expected_raw = np.asarray([[10.0, 12.0, 21.0], [30.0, 32.0, 41.0]])
    expected = (expected_raw - np.asarray([1.0, 2.0, 3.0])) / np.asarray(
        [1.0, 2.0, 4.0]
    )
    np.testing.assert_allclose(proprio, expected.astype(np.float32))


def test_preprocess_and_cached_language_match_policy_path(
    encoder: PolicyFeatureEncoder,
) -> None:
    images = torch.zeros((2, 3, 128, 160, 3), dtype=torch.uint8, device=CUDA)
    images[:, :, :, 16:144, :] = 255
    proprio = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        dtype=torch.float32,
        device=CUDA,
    )

    preprocessed = encoder.preprocess_images(images)
    assert preprocessed.shape == (2, 3, 3, 128, 128)
    channel_mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32, device=CUDA
    ).view(1, 1, 3, 1, 1)
    channel_std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=torch.float32, device=CUDA
    ).view(1, 1, 3, 1, 1)
    expected_images = (torch.ones_like(preprocessed) - channel_mean) / channel_std
    torch.testing.assert_close(preprocessed, expected_images)

    direct = encoder.model.get_cond_features(
        preprocessed,
        proprio,
        [encoder.prompt_for_task("Stack")] * images.shape[0],
    )
    encoded = encoder.encode_batch(images, proprio, "Stack")
    encoded_again = encoder.encode_batch(images, proprio, "Stack")
    assert encoded.shape == (2, 256)
    assert encoded.dtype == torch.float32
    assert encoded.device.type == "cuda"
    assert bool(torch.isfinite(encoded).all())
    torch.testing.assert_close(encoded, direct)
    torch.testing.assert_close(encoded_again, direct)
    assert encoder.model.language_calls == 2
