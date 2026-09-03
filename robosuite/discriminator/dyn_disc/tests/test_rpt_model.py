import pytest
import torch

from robosuite.discriminator.dyn_disc.models.rpt import RPTModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="RPT tensor tests require CUDA")


def _inputs(batch_size=3):
    device = torch.device("cuda")
    return (
        torch.randn(batch_size, 8, 2, 768, device=device),
        torch.randn(batch_size, 8, 14, device=device),
        torch.randn(batch_size, 8, 7, device=device),
    )


def test_rpt_forward_and_action_token_features_are_cuda_float32():
    model = RPTModel().cuda()
    visual, proprio, actions = _inputs()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, components = model(visual, proprio, actions)
        features = model.extract_features(visual, proprio, actions)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.is_cuda and torch.isfinite(loss)
    assert features.shape == (3, 192)
    assert features.is_cuda
    assert set(components) == {"total", "visual", "proprio", "action", "mask_ratio"}
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_rpt_mask_is_reproducible_and_repairs_degenerate_sampling():
    model = RPTModel(mask_ratio_min=1.0, mask_ratio_max=1.0).cuda()
    torch.cuda.manual_seed_all(11)
    first = model.sample_mask(16, device=torch.device("cuda"))
    torch.cuda.manual_seed_all(11)
    second = model.sample_mask(16, device=torch.device("cuda"))
    assert torch.equal(first, second)
    assert first.flatten(1).any(dim=1).all()
    assert (~first).flatten(1).any(dim=1).all()


def test_rpt_sampled_mask_ratio_stays_in_configured_range_per_sample():
    model = RPTModel().cuda()
    mask = model.sample_mask(256, device=torch.device("cuda"))
    ratios = mask.float().mean(dim=(1, 2))
    # Bernoulli draws fluctuate around each sampled p; the hard guarantees and
    # aggregate range are the stable properties of the finite 32-token mask.
    assert mask.flatten(1).any(dim=1).all()
    assert (~mask).flatten(1).any(dim=1).all()
    assert 0.65 <= ratios.mean().item() <= 0.95


def test_rpt_shared_visual_modules_and_equal_modality_loss_reduction():
    model = RPTModel().cuda()
    assert not hasattr(model, "agentview_projection")
    assert model.visual_projection.in_features == 768
    assert model.slot_position.shape == (4, 192)
    visual, proprio, actions = _inputs(batch_size=1)
    mask = torch.zeros(1, 8, 4, dtype=torch.bool, device="cuda")
    mask[:, 0, 0] = True
    mask[:, 1, 2] = True
    loss, components = model(visual, proprio, actions, mask=mask)
    assert components["action"].item() == 0.0
    assert torch.allclose(loss, (components["visual"] + components["proprio"]) / 2)


def test_slot_position_makes_camera_order_observable():
    torch.cuda.manual_seed_all(5)
    model = RPTModel().cuda().eval()
    visual, proprio, actions = _inputs(batch_size=2)
    with torch.inference_mode():
        baseline = model.extract_features(visual, proprio, actions)
        swapped = model.extract_features(visual.flip(dims=(2,)), proprio, actions)
    assert not torch.allclose(baseline, swapped, atol=1e-6, rtol=1e-6)


def test_unmasked_targets_do_not_enter_reconstruction_loss():
    prediction = torch.randn(2, 8, 7, device="cuda")
    target = torch.randn_like(prediction)
    mask = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
    mask[:, 3] = True
    baseline, valid = RPTModel._masked_mse(prediction, target, mask)
    changed = target.clone()
    changed[:, :3] += 1000.0
    changed[:, 4:] -= 1000.0
    updated, updated_valid = RPTModel._masked_mse(prediction, changed, mask)
    assert valid.item() == updated_valid.item() == 1.0
    assert torch.equal(baseline, updated)


def test_rpt_rejects_mask_without_visible_token():
    model = RPTModel().cuda()
    visual, proprio, actions = _inputs(batch_size=1)
    mask = torch.ones(1, 8, 4, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="at least one masked and one visible"):
        model(visual, proprio, actions, mask=mask)
