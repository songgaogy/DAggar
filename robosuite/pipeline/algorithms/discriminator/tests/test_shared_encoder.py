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
