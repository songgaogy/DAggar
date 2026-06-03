"""Verify LPB encoder ownership after Q/V move to independent ResNet-50.

We assert:
1. OnlineBCEDiscriminator and AdvantageGProvider reuse one SharedFrozenEncoder.
2. IQL owns independent frozen ResNet-50 visual encoders.

Skips when no CUDA / no LPB BCE checkpoint is available.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
from robosuite.pipeline.algorithms.discriminator.online_bce import (
    DiscriminatorConfig,
    OnlineBCEDiscriminator,
)
from robosuite.pipeline.algorithms.dipole.advantage_g_provider import AdvantageGProvider
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner

REPO_ROOT = Path(__file__).resolve().parents[5]
RESNET50_CKPT = REPO_ROOT / "data" / "pretrained" / "resnet50.pth"
CKPT_GLOBS = [
    str(REPO_ROOT / "checkpoints" / "lpb_v2" / "bce_viz_robosuite"
        / "*" / "checkpoints" / "bce_head.pth"),
    str(REPO_ROOT / "checkpoints" / "lpb_v2" / "bce_eval_robosuite"
        / "run_*" / "checkpoints" / "bce_head.pth"),
]


def _resolve_bce_ckpt() -> str | None:
    env = os.environ.get("DIPOLE_BCE_CKPT")
    if env and Path(env).exists():
        return env
    matches: list[str] = []
    for pattern in CKPT_GLOBS:
        matches.extend(glob.glob(pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


@pytest.fixture(scope="module")
def bce_ckpt() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 CUDA devices for cuda:1 layout test")
    p = _resolve_bce_ckpt()
    if p is None:
        pytest.skip("No lpb_v2 BCE checkpoint available")
    return p


def test_encoder_id_shared_across_modules(bce_ckpt: str) -> None:
    device = "cuda:1"
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)

    enc = SharedFrozenEncoder(bce_ckpt, device=device)
    enc.bind_policy_cameras(enc.view_names)

    iql_cfg = IQLConfig(
        action_horizon=8, hidden_dims=(64, 64), device=device,
        resnet_pretrained_path=str(RESNET50_CKPT),
    )
    action_dim = 7
    iql = IQLLearner(
        iql_cfg, camera_names=enc.view_names, proprio_dim=8, action_dim=action_dim
    )

    disc_cfg = DiscriminatorConfig(
        device=device, hidden=64, num_layers=2,
        warm_start_ckpt=None,
    )
    disc = OnlineBCEDiscriminator(
        cfg=disc_cfg,
        encoder=enc,
        context_dim=enc.context_dim,
        action_dim=action_dim,
    )

    adv = AdvantageGProvider(
        iql_learner=iql,
        discriminator=disc,
        encoder=enc,
        alpha=1.0,
        beta=0.5,
        advantage_normalization="batch_zscore",
        disc_normalization="batch_zscore",
    )

    # 1. Identity: same encoder object reached through each downstream.
    assert id(disc._encoder_ref) == id(enc)
    assert id(adv.encoder) == id(enc)

    assert iql.q1.vis_encoder.backbone is not iql.q2.vis_encoder.backbone
    assert iql.q1.vis_encoder.backbone is not iql.v.vis_encoder.backbone
    assert all(not p.requires_grad for p in iql.q1.vis_encoder.backbone.parameters())


def test_memory_overhead_under_threshold(bce_ckpt: str) -> None:
    device = "cuda:1"
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=device)
    base_alloc = torch.cuda.memory_allocated(device=device)

    enc = SharedFrozenEncoder(bce_ckpt, device=device)
    enc.bind_policy_cameras(enc.view_names)
    torch.cuda.synchronize(device=device)
    encoder_alloc = torch.cuda.memory_allocated(device=device) - base_alloc

    iql_cfg = IQLConfig(
        action_horizon=8,
        hidden_dims=(64, 64),
        device=device,
        resnet_pretrained_path=str(RESNET50_CKPT),
    )
    action_dim = 7
    iql = IQLLearner(
        iql_cfg, camera_names=enc.view_names, proprio_dim=8, action_dim=action_dim
    )

    disc_cfg = DiscriminatorConfig(
        device=device, hidden=64, num_layers=2, warm_start_ckpt=None
    )
    disc = OnlineBCEDiscriminator(
        cfg=disc_cfg,
        encoder=enc,
        context_dim=enc.context_dim,
        action_dim=action_dim,
    )
    adv = AdvantageGProvider(
        iql_learner=iql, discriminator=disc, encoder=enc,
        alpha=1.0, beta=0.5,
        advantage_normalization="batch_zscore",
        disc_normalization="batch_zscore",
    )
    torch.cuda.synchronize(device=device)
    full_alloc = torch.cuda.memory_allocated(device=device) - base_alloc

    overhead = full_alloc - encoder_alloc
    # Four frozen ResNet-50 copies plus trainable heads should stay bounded.
    assert overhead < 600 * 1024 * 1024, (
        f"ResNet Q/V + head overhead {overhead / 1e6:.1f} MB exceeds 600 MB; "
        f"encoder_alloc={encoder_alloc / 1e6:.1f} MB, full_alloc={full_alloc / 1e6:.1f} MB"
    )

    pct = 100.0 * overhead / max(encoder_alloc, 1)
    print(
        f"[encoder_sharing] encoder={encoder_alloc / 1e6:.1f}MB, "
        f"overhead={overhead / 1e6:.1f}MB ({pct:.1f}% of encoder)"
    )
    # Quiet hold the locals in scope so pyflakes does not warn.
    _ = (adv,)
