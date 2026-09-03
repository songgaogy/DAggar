import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from robosuite.discriminator.dyn_disc.core import model_loader
from robosuite.discriminator.dyn_disc.detectors.single_bank_knn import DynEncoder
from robosuite.discriminator.dyn_disc.models.taco import (
    RandomShiftsAug,
    TACOActionEncoder,
    TACORepresentationModel,
)
from robosuite.discriminator.dyn_disc.training.train import _save_ckpt
from robosuite.discriminator.dyn_disc.utils.normalizer import LinearNormalizer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
DEVICE = torch.device("cuda")


class DummyVisualEncoder(nn.Module):
    emb_dim = 6
    num_patches = 4
    normalizes_images = True

    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.proj = nn.Linear(3, self.emb_dim)
        self.freeze_backbone = True
        self.train_projection = True
        self.set_trainable(False, True)

    def set_trainable(self, train_backbone, train_projection=True):
        self.freeze_backbone = not train_backbone
        self.train_projection = train_projection
        self.backbone.requires_grad_(train_backbone)
        self.proj.requires_grad_(train_projection)

    def _encode_flat(self, images):
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone(images)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (2, 2))
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.proj(tokens)

    def forward(self, visual):
        encoded = {}
        for name, images in visual.items():
            batch, time = images.shape[:2]
            tokens = self._encode_flat(images.flatten(0, 1))
            encoded[name] = tokens.unflatten(0, (batch, time))
        return encoded


class DummyProprioEncoder(nn.Module):
    emb_dim = 5

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, self.emb_dim)

    def forward(self, proprio):
        return self.net(proprio)


def make_model():
    return TACORepresentationModel(
        image_size=8,
        num_hist=1,
        num_pred=1,
        encoder=DummyVisualEncoder(),
        proprio_encoder=DummyProprioEncoder(),
        proprio_dim=5,
        action_dim_per_step=7,
        frameskip=8,
        view_names=["agentview", "robot0_eye_in_hand"],
        source_view_names=["agentview", "robot0_eye_in_hand"],
        target_view_names=["agentview"],
        encoder_micro_batch_size=2,
    ).to(DEVICE)


def make_batch(batch_size=4):
    visual = {
        "agentview": torch.randint(0, 256, (batch_size, 2, 3, 8, 8), dtype=torch.uint8, device=DEVICE),
        "robot0_eye_in_hand": torch.randint(
            0, 256, (batch_size, 2, 3, 8, 8), dtype=torch.uint8, device=DEVICE
        ),
    }
    proprio = torch.randn(batch_size, 2, 4, device=DEVICE)
    actions = torch.randn(batch_size, 2, 56, device=DEVICE)
    return {"visual": visual, "proprio": proprio}, actions


def _test_config():
    return OmegaConf.create(
        {
            "pretraining_method": "taco",
            "view_names": ["agentview", "robot0_eye_in_hand"],
            "source_view_names": ["agentview", "robot0_eye_in_hand"],
            "target_view_names": ["agentview"],
            "frameskip": 8,
            "num_hist": 1,
            "num_pred": 1,
            "abs_action": False,
            "use_crop": False,
            "prior_in_chans": 4,
            "action_dim_per_step": 7,
            "proprio_emb_dim": 5,
            "action_emb_dim": 72,
            "policy_ckpt_path": None,
            "train_data_path": "unused",
            "encoder": {"_target_": "fake.encoder", "train_projection": True},
            "proprio_encoder": {"_target_": "fake.proprio"},
            "model": {
                "_target_": "fake.taco",
                "image_size": 8,
                "num_hist": 1,
                "num_pred": 1,
                "train_encoder": False,
                "state_dim": 50,
                "encoder_micro_batch_size": 2,
                "random_shift_pad": 4,
            },
            "env": {
                "view_names": ["agentview", "robot0_eye_in_hand"],
                "proprio_dim": 4,
                "action_dim": 7,
                "proprio_emb_dim": 5,
                "action_emb_dim": 72,
                "original_img_size": 8,
                "cropped_img_size": 8,
            },
        }
    )


def _fake_instantiate(node, **kwargs):
    target = str(node._target_)
    if target == "fake.encoder":
        return DummyVisualEncoder()
    if target == "fake.proprio":
        return DummyProprioEncoder()
    if target == "fake.taco":
        return TACORepresentationModel(
            image_size=8,
            num_hist=1,
            num_pred=1,
            state_dim=50,
            encoder_micro_batch_size=2,
            random_shift_pad=4,
            **kwargs,
        )
    raise AssertionError(f"Unexpected test target: {target}")


def _save_test_normalizer(run_dir):
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "act": torch.linspace(-1, 1, 56, device=DEVICE).reshape(8, 7),
            "state": torch.linspace(-1, 1, 32, device=DEVICE).reshape(8, 4),
        }
    )
    torch.save(normalizer.state_dict(), run_dir / "normalizer.pth")


def test_action_encoder_shapes_and_sequence_slicing():
    encoder = TACOActionEncoder(action_dim=7, sequence_length=8).to(DEVICE)
    flat = torch.randn(3, 2, 56, device=DEVICE)
    sequence = flat.unflatten(-1, (8, 7))
    flat_result = encoder(flat)
    sequence_result = encoder(sequence)
    assert flat_result.shape == (3, 2, 72)
    torch.testing.assert_close(flat_result, sequence_result)


def test_action_encoder_respects_configured_width():
    encoder = TACOActionEncoder(
        action_dim=7,
        sequence_length=8,
        step_hidden=128,
        step_emb_dim=32,
    ).to(DEVICE)
    out = encoder(torch.randn(2, 56, device=DEVICE))
    assert encoder.emb_dim == 256
    assert out.shape == (2, 256)


def test_scaled_taco_capacity_forward_shapes():
    model = TACORepresentationModel(
        image_size=8,
        num_hist=1,
        num_pred=1,
        encoder=DummyVisualEncoder(),
        proprio_encoder=DummyProprioEncoder(),
        proprio_dim=5,
        action_dim_per_step=7,
        frameskip=8,
        view_names=["agentview", "robot0_eye_in_hand"],
        source_view_names=["agentview", "robot0_eye_in_hand"],
        target_view_names=["agentview"],
        state_dim=256,
        transition_hidden=1024,
        action_step_hidden=128,
        action_step_emb_dim=32,
        encoder_micro_batch_size=2,
    ).to(DEVICE)
    obs, actions = make_batch(batch_size=2)
    loss, components = model(obs, actions)
    assert model.state_projector.out_features == 256
    assert model.action_encoder.emb_dim == 256
    assert model.transition[0].in_features == 512
    assert model.transition[0].out_features == 1024
    assert model.W.shape == (256, 256)
    assert components["logits"].shape == (2, 2)
    assert torch.isfinite(loss)


def test_info_nce_uses_global_keys_and_positive_offset_in_fp32():
    model = make_model()
    model.W.data.copy_(torch.eye(50, device=DEVICE))
    keys = torch.eye(50, device=DEVICE, dtype=torch.bfloat16)[:4] * 10
    predictions = keys[2:4]
    loss, logits = model.info_nce(predictions, keys, positive_offset=2)
    assert loss.item() < 1e-4
    assert logits.dtype == torch.float32
    assert logits.argmax(dim=1).tolist() == [2, 3]


def test_forward_shapes_stop_gradient_and_trainable_gradients():
    model = make_model().train()
    obs, actions = make_batch()
    loss, components = model(obs, actions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert components["logits"].shape == (4, 4)
    assert components["global_batch_size"] == 4
    assert components["negative_count"] == 3
    assert not components["future_keys"].requires_grad

    loss.backward()
    assert model.encoder.proj.weight.grad is not None
    assert model.encoder.backbone.weight.grad is None
    assert model.proprio_encoder.net.weight.grad is not None
    assert model.action_encoder.step_encoder[0].weight.grad is not None
    assert model.state_projector.weight.grad is not None
    assert model.transition[0].weight.grad is not None
    assert model.W.grad is not None


def test_future_projection_matches_shared_projector_with_missing_view_zero_filled():
    model = make_model()
    batch_size = 3
    agentview = torch.randn(batch_size, 4, 6, device=DEVICE)
    proprio = torch.randn(batch_size, 5, device=DEVICE)

    projected = model._project_future({"agentview": agentview}, proprio)
    missing_wrist = torch.zeros_like(agentview)
    shared_input = torch.cat(
        [agentview.flatten(1), missing_wrist.flatten(1), proprio], dim=-1
    )
    expected = model.state_norm(model.state_projector(shared_input))

    torch.testing.assert_close(projected, expected)


def test_random_shift_is_shared_across_views_and_resampled_for_future():
    model = make_model()
    ramp = torch.arange(64, device=DEVICE).view(1, 1, 1, 8, 8).expand(2, 2, 3, 8, 8).float()
    visual = {"agentview": ramp.clone(), "robot0_eye_in_hand": ramp.clone()}
    calls = []

    def fixed_shifts(batch_size, device):
        shift = torch.tensor([[0, 0], [1, 1]], device=device) if not calls else torch.tensor(
            [[8, 8], [7, 7]], device=device
        )
        calls.append(shift)
        return shift

    model.random_shift.sample_shifts = fixed_shifts
    query = model._augment_views(visual, model.source_view_names, time_index=0)
    future = model._augment_views(visual, model.target_view_names, time_index=1)
    torch.testing.assert_close(query["agentview"], query["robot0_eye_in_hand"])
    assert len(calls) == 2
    assert not torch.equal(query["agentview"], future["agentview"])


def test_bfloat16_autocast_forward_backward_keeps_loss_fp32():
    model = make_model().train()
    obs, actions = make_batch(batch_size=2)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, components = model(obs, actions)
    assert loss.dtype == torch.float32
    assert components["logits"].dtype == torch.float32
    loss.backward()
    assert torch.isfinite(model.W.grad).all()


def test_encode_obs_and_encode_act_keep_downstream_contract():
    model = make_model().eval()
    obs, actions = make_batch(batch_size=2)
    with torch.no_grad():
        encoded = model.encode_obs(obs)
        action_latent = model.encode_act(actions)
    assert encoded["visual"].shape == (2, 2, 4, 12)
    assert encoded["proprio"].shape == (2, 2, 5)
    assert action_latent.shape == (2, 2, 72)


def test_taco_checkpoint_round_trip_and_dyn_encoder_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "instantiate_local", _fake_instantiate)
    run_dir = tmp_path / "taco_run"
    run_dir.mkdir()
    cfg = _test_config()
    OmegaConf.save(cfg, run_dir / "hydra.yaml", resolve=True)
    _save_test_normalizer(run_dir)

    original = make_model().eval()
    checkpoint = _save_ckpt(
        run_dir,
        epoch=10,
        parts={"pretraining_method": "taco", "model": original},
        ckpt_subdir="checkpoint",
    )
    restored = model_loader.load_model(checkpoint, cfg, device=DEVICE).eval()
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

    dyn_encoder = DynEncoder(
        model_ckpt=str(checkpoint),
        device="cuda",
    )
    batch_size = 3
    images = {
        name: torch.rand(batch_size, 3, 8, 8, device=DEVICE)
        for name in cfg.view_names
    }
    proprio = torch.randn(batch_size, 4, device=DEVICE)
    actions = torch.randn(batch_size, 56, device=DEVICE)
    latent = dyn_encoder.encode_batch(images, proprio, actions)
    assert latent.ndim == 2
    assert latent.shape == (batch_size, 4 * 6 * 2 + 5 + 72)
    assert torch.isfinite(latent).all()


@pytest.mark.parametrize(
    ("checkpoint_method", "config_method"),
    [
        (None, "taco"),
        ("dynamics", "taco"),
        ("taco", "dynamics"),
    ],
)
def test_model_loader_rejects_non_taco_checkpoint_or_config(
    tmp_path,
    monkeypatch,
    checkpoint_method,
    config_method,
):
    monkeypatch.setattr(model_loader, "instantiate_local", _fake_instantiate)
    cfg = _test_config()
    cfg.pretraining_method = config_method
    parts = {"model": make_model()}
    if checkpoint_method is not None:
        parts["pretraining_method"] = checkpoint_method
    checkpoint = _save_ckpt(
        tmp_path,
        epoch=10,
        parts=parts,
    )

    with pytest.raises(ValueError, match="TACO|taco"):
        model_loader.load_model(checkpoint, cfg, device=DEVICE)
