"""CPU tests for LPB-annotated disc rewards in IQLReplayBuffer."""

from __future__ import annotations

import numpy as np

from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    lpb_disc_intrinsic_from_failure_score,
)
from robosuite.pipeline.algorithms.q_learning.replay import _lpb_disc_steps_for_sequence
from robosuite.pipeline.common import Transition


def _trans(intrinsic: float) -> Transition:
    return Transition(
        obs={"agentview": np.zeros((4, 4, 3), dtype=np.uint8), "state": np.zeros(3)},
        action=np.zeros(2, dtype=np.float32),
        reward=-1.0,
        next_obs={"agentview": np.zeros((4, 4, 3), dtype=np.uint8), "state": np.zeros(3)},
        done=False,
        info={"lpb_disc_intrinsic": float(intrinsic)},
    )


def test_lpb_disc_steps_for_sequence_reads_cached_intrinsic() -> None:
    seq = [_trans(-0.1), _trans(-0.5), _trans(-0.9)]
    out = _lpb_disc_steps_for_sequence(seq, horizon=3)
    assert out is not None
    np.testing.assert_allclose(out, [-0.1, -0.5, -0.9], rtol=0, atol=1e-6)


def test_lpb_disc_steps_missing_annotation_returns_none() -> None:
    seq = [_trans(-0.1), Transition(
        obs={"agentview": np.zeros((4, 4, 3), dtype=np.uint8), "state": np.zeros(3)},
        action=np.zeros(2, dtype=np.float32),
        reward=-1.0,
        next_obs={"agentview": np.zeros((4, 4, 3), dtype=np.uint8), "state": np.zeros(3)},
        done=False,
        info={},
    )]
    assert _lpb_disc_steps_for_sequence(seq, horizon=2) is None


def test_intrinsic_matches_sigmoid_formula() -> None:
    tau = 3.82
    score = 7.0
    expected = float(lpb_disc_intrinsic_from_failure_score(score, tau))
    assert expected < -0.9
