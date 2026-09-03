import types

import pytest
import torch
from omegaconf import OmegaConf

from robosuite.discriminator.dyn_disc.core.model_loader import load_rpt_checkpoint
from robosuite.discriminator.dyn_disc.core.rpt_encoder import RPTTrajectoryEncoder
from robosuite.discriminator.dyn_disc.models.rpt import RPTModel


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="RPT trajectory tests require CUDA"
)


class _IdentityField:
    @staticmethod
    def normalize(value):
        return value


def _encoder_without_dino() -> RPTTrajectoryEncoder:
    encoder = RPTTrajectoryEncoder.__new__(RPTTrajectoryEncoder)
    encoder.device = torch.device("cuda")
    encoder.image_batch_size = 16
    encoder.window_batch_size = 16
    encoder.model = RPTModel().cuda().eval()
    encoder.view_names = ["agentview", "robot0_eye_in_hand"]
    encoder.context_length = 8
    encoder.proprio_dim = 14
    encoder.action_dim = 7
    encoder.hidden_dim = 192
    encoder.normalizer = {"state": _IdentityField(), "act": _IdentityField()}
    encoder._encode_images = types.MethodType(
        lambda self, images: {view: images[view].cuda() for view in self.view_names},
        encoder,
    )
    return encoder


def test_rpt_trajectory_encoder_is_past_only_and_left_padded():
    torch.cuda.manual_seed_all(7)
    encoder = _encoder_without_dino()
    length = 10
    visual = {
        view: torch.randn(length, 768, device="cuda")
        for view in encoder.view_names
    }
    proprio = torch.randn(length, 14, device="cuda")
    actions = torch.randn(length, 7, device="cuda")

    baseline = encoder.encode_trajectory(visual, proprio, actions)
    perturbed_visual = {view: value.clone() for view, value in visual.items()}
    perturbed_visual["agentview"][6:] += 100.0
    perturbed_proprio = proprio.clone()
    perturbed_actions = actions.clone()
    perturbed_proprio[6:] -= 100.0
    perturbed_actions[6:] += 100.0
    perturbed = encoder.encode_trajectory(
        perturbed_visual, perturbed_proprio, perturbed_actions
    )

    assert baseline.shape == (length, 192)
    assert torch.equal(baseline[:6], perturbed[:6])
    assert not torch.equal(baseline[6:], perturbed[6:])


def test_rpt_checkpoint_roundtrip_and_legacy_rejection(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            "pretraining_method": "rpt",
            "env": {
                "view_names": ["agentview", "robot0_eye_in_hand"],
                "proprio_dim": 14,
                "action_dim": 7,
            },
            "context_length": 8,
        }
    )
    OmegaConf.save(cfg, run_dir / "hydra.yaml")
    source = RPTModel().cuda().eval()
    checkpoint = checkpoint_dir / "model_10.pth"
    payload = {
        "pretraining_method": "rpt",
        "checkpoint_version": 1,
        "epoch": 10,
        "global_step": 12,
        "rpt_model": source.state_dict(),
        "architecture": source.architecture_metadata(),
        "cache_manifest_fingerprint": "cache-fingerprint",
    }
    torch.save(payload, checkpoint)

    restored, _, metadata = load_rpt_checkpoint(checkpoint, device="cuda")
    inputs = (
        torch.randn(1, 8, 2, 768, device="cuda"),
        torch.randn(1, 8, 14, device="cuda"),
        torch.randn(1, 8, 7, device="cuda"),
    )
    with torch.inference_mode():
        expected = source.extract_features(*inputs)
        actual = restored.extract_features(*inputs)
    assert torch.equal(expected, actual)
    assert metadata["cache_manifest_fingerprint"] == "cache-fingerprint"
    assert len(metadata["checkpoint_fingerprint"]) == 64

    payload["pretraining_method"] = "dynamics"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="pretraining_method='rpt'"):
        load_rpt_checkpoint(checkpoint, device="cuda")
