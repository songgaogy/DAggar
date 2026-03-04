from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_float_core_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "robosuite" / "discriminator" / "float_core.py"
    module_name = "float_core_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


float_core = _load_float_core_module()
FLOATComputer = float_core.FLOATComputer
IdentityEncoder = float_core.IdentityEncoder
ThresholdCalibrator = float_core.ThresholdCalibrator
Trajectory = float_core.Trajectory
sinkhorn = float_core.sinkhorn


def test_sinkhorn_plan_respects_marginals() -> None:
    a = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    b = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4], dtype=np.float64)
    c = np.array(
        [
            [0.1, 0.3, 0.8, 0.9],
            [0.2, 0.1, 0.7, 0.6],
            [0.9, 0.8, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    p = sinkhorn(a=a, b=b, c=c, reg=0.05, max_iter=500, tol=1e-8)

    assert p.shape == (3, 4)
    assert np.allclose(p.sum(axis=1), a, atol=1e-3)
    assert np.allclose(p.sum(axis=0), b, atol=1e-3)


def test_lambda_is_lower_for_matching_rollout() -> None:
    expert_1 = Trajectory(
        obs=np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.90, 0.10],
            ],
            dtype=np.float32,
        )
    )
    expert_2 = Trajectory(
        obs=np.array(
            [
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
    )

    match_rollout = Trajectory(
        obs=np.array(
            [
                [1.0, 0.0],
                [0.93, 0.07],
                [0.89, 0.11],
            ],
            dtype=np.float32,
        )
    )
    mismatch_rollout = Trajectory(
        obs=np.array(
            [
                [-1.0, 0.0],
                [-0.9, -0.1],
                [-0.8, -0.2],
            ],
            dtype=np.float32,
        )
    )

    fc = FLOATComputer(
        experts=[expert_1, expert_2],
        encoder=IdentityEncoder(),
        sinkhorn_reg=0.05,
        max_iter=300,
        tol=1e-6,
    )

    l_match = fc.lambda_index(match_rollout)
    l_mismatch = fc.lambda_index(mismatch_rollout)

    assert l_match < l_mismatch


def test_threshold_percentile_behavior() -> None:
    cal = ThresholdCalibrator()
    threshold = cal.update(delta=10.0, lambdas=[1.0, 2.0, 3.0, 4.0, 5.0])
    expected = float(np.percentile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), q=90.0))
    assert np.isclose(threshold, expected)


def test_rewind_timestep_returns_latest_valid_prefix() -> None:
    expert = Trajectory(obs=np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1)))
    rollout = Trajectory(
        obs=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    fc = FLOATComputer(
        experts=[expert],
        encoder=IdentityEncoder(),
        sinkhorn_reg=0.05,
        max_iter=300,
        tol=1e-6,
    )

    t0 = 5
    eps = 0.5
    l0 = fc.lambda_index(rollout=rollout, t0=t0)
    expected = 1
    for t in range(1, t0 + 1):
        if fc.lambda_index(rollout=rollout, t0=t) <= eps * l0:
            expected = t

    rewind_t = fc.rewind_timestep(rollout=rollout, t0=t0, eps=eps)
    assert rewind_t == expected


if __name__ == "__main__":
    test_sinkhorn_plan_respects_marginals()
    test_lambda_is_lower_for_matching_rollout()
    test_threshold_percentile_behavior()
    test_rewind_timestep_returns_latest_valid_prefix()
