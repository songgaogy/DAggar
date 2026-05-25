"""Smoke tests for SharedFrozenEncoder.

Picks up a BCE checkpoint either from the DIPOLE_BCE_CKPT env var or by
globbing the newest robosuite eval run. Skips cleanly when neither is
available or CUDA is missing.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder

REPO_ROOT = Path(__file__).resolve().parents[5]
CKPT_GLOB = str(
    REPO_ROOT / "checkpoints" / "lpb_v2" / "bce_eval_robosuite"
    / "run_*" / "checkpoints" / "bce_head.pth"
)


def _resolve_bce_ckpt() -> str | None:
    env = os.environ.get("DIPOLE_BCE_CKPT")
    if env and Path(env).exists():
        return env
    matches = sorted(glob.glob(CKPT_GLOB), key=os.path.getmtime)
    return matches[-1] if matches else None


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def bce_ckpt() -> str:
    p = _resolve_bce_ckpt()
    if p is None:
        pytest.skip(
            "No BCE checkpoint available "
            "(set DIPOLE_BCE_CKPT or place one under "
            "checkpoints/lpb_v2/bce_eval_robosuite/...)."
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for LPBV2Encoder")
    return p


@pytest.fixture(scope="module")
def enc(bce_ckpt: str) -> SharedFrozenEncoder:
    return SharedFrozenEncoder(bce_ckpt, device=_device())


def test_metadata_available_after_init(enc: SharedFrozenEncoder) -> None:
    assert enc.context_dim > 0
    assert enc.action_input_dim > 0
    assert len(enc.view_names) > 0
    H, W = enc.original_img_size
    assert H > 0 and W > 0


def test_all_params_frozen(enc: SharedFrozenEncoder) -> None:
    params = list(enc.inner_encoder.model.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)


def test_encode_shape(enc: SharedFrozenEncoder) -> None:
    enc.bind_policy_cameras(enc.view_names)
    B, V = 2, len(enc.view_names)
    H, W = enc.original_img_size
    images = torch.rand(B, V, 3, H, W)
    proprio = torch.zeros(B, enc.proprio_input_dim)
    out = enc.encode(image_obs_raw=images, proprio_raw=proprio)
    assert out.shape == (B, enc.context_dim), (
        f"expected (B, D_ctx) = ({B}, {enc.context_dim}); got {tuple(out.shape)}"
    )
    assert not out.requires_grad


def test_rebind_same_cameras_is_noop(enc: SharedFrozenEncoder) -> None:
    enc.bind_policy_cameras(enc.view_names)
    enc.bind_policy_cameras(list(enc.view_names))  # same content, fresh list


def test_bind_missing_view_raises(bce_ckpt: str) -> None:
    fresh = SharedFrozenEncoder(bce_ckpt, device=_device())
    with pytest.raises(KeyError):
        fresh.bind_policy_cameras(["definitely_not_a_view"])


def test_rebind_different_cameras_raises(bce_ckpt: str) -> None:
    fresh = SharedFrozenEncoder(bce_ckpt, device=_device())
    fresh.bind_policy_cameras(fresh.view_names)
    bogus = list(fresh.view_names) + ["bogus_extra_camera"]
    with pytest.raises(RuntimeError):
        fresh.bind_policy_cameras(bogus)


def test_action_real_affects_latent(enc: SharedFrozenEncoder) -> None:
    """With `action_real` supplied, the encoder must produce a different
    latent than the default `tile(proprio)` path. This pins down that the
    new kwarg actually reaches `LPBV2Encoder.encode_batch(...)` and isn't
    silently dropped.
    """
    enc.bind_policy_cameras(enc.view_names)
    torch.manual_seed(0)
    B, V = 3, len(enc.view_names)
    H, W = enc.original_img_size
    images = torch.rand(B, V, 3, H, W)
    proprio = torch.zeros(B, enc.proprio_input_dim)
    # Random action; the tile(proprio) fallback would produce a different
    # bit pattern, so the two latents must disagree element-wise.
    action_real = torch.randn(B, enc.action_dim_per_step)

    out_tile = enc.encode(image_obs_raw=images, proprio_raw=proprio)
    out_real = enc.encode(
        image_obs_raw=images, proprio_raw=proprio, action_real=action_real
    )
    assert out_real.shape == out_tile.shape == (B, enc.context_dim)
    assert not torch.allclose(out_tile, out_real, atol=1e-5), (
        "encoder produced identical latents with and without `action_real`; "
        "the kwarg may not be threaded through to inner_encoder.encode_batch."
    )


def test_encode_chunk_frames_uses_frameskip_window(enc: SharedFrozenEncoder) -> None:
    """Frame h>0 must use actions[h:h+fs], not repeat only action_h in isolation."""
    enc.bind_policy_cameras(enc.view_names)
    if enc.frameskip <= 1:
        pytest.skip("frameskip=1: window collapse makes this test uninformative.")
    torch.manual_seed(1)
    B, H = 1, min(4, enc.frameskip + 1)
    V = len(enc.view_names)
    Hi, Wi = enc.original_img_size
    chunk_images = torch.rand(B, H, V, 3, Hi, Wi)
    chunk_proprio = torch.zeros(B, H, enc.proprio_input_dim)
    chunk_actions = torch.randn(B, H, enc.action_dim_per_step)
    # Deliberately different actions across steps.
    chunk_actions[0, 1] = chunk_actions[0, 0] + 3.0

    out_chunk = enc.encode_chunk_frames(
        chunk_images=chunk_images,
        chunk_proprio=chunk_proprio,
        chunk_actions=chunk_actions,
    )
    # Old path: per-row encode with only action_h (no future steps in window).
    flat_images = chunk_images.reshape(B * H, V, 3, Hi, Wi)
    flat_proprio = chunk_proprio.reshape(B * H, -1)
    flat_actions = chunk_actions.reshape(B * H, -1)
    out_repeat = enc.encode(
        image_obs_raw=flat_images,
        proprio_raw=flat_proprio,
        action_real=flat_actions,
    ).view(B, H, -1)

    assert out_chunk.shape == out_repeat.shape
    # At h=1 with fs>1 the windows differ when more chunk steps exist.
    if H > 1 and enc.frameskip > 1:
        assert not torch.allclose(out_chunk[0, 1], out_repeat[0, 1], atol=1e-5)
