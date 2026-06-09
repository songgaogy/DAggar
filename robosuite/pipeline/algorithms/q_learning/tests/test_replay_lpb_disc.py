"""CPU tests for LPB-annotated disc rewards in IQLReplayBuffer."""

from __future__ import annotations

import threading

import numpy as np

from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    lpb_disc_intrinsic_from_failure_score,
)
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.replay import (
    IQLReplayBuffer,
    _lpb_disc_steps_for_sequence,
)
from robosuite.pipeline.common import Transition


def _trans(
    intrinsic: float = -0.1,
    *,
    done: bool = False,
    info: dict | None = None,
) -> Transition:
    info_payload = {"lpb_disc_intrinsic": float(intrinsic)}
    if info is not None:
        info_payload.update(info)
    return Transition(
        obs={"agentview": np.zeros((4, 4, 3), dtype=np.uint8), "state": np.zeros(3)},
        action=np.zeros(2, dtype=np.float32),
        reward=-1.0,
        next_obs={"agentview": np.zeros((4, 4, 3), dtype=np.uint8), "state": np.zeros(3)},
        done=bool(done),
        info=info_payload,
    )


class _FakeBase:
    def __init__(self, storage: list[Transition], horizon: int) -> None:
        self._storage = storage
        self._lock = threading.RLock()
        self.camera_names = ["agentview"]
        self.action_horizon = int(horizon)

    def _get_valid_start_indices_locked(self) -> list[int]:
        return list(range(len(self._storage) - self.action_horizon + 1))


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


def test_iql_replay_drops_truncated_boundary_chunk() -> None:
    horizon = 4
    storage = []
    for step in range(10):
        storage.append(
            _trans(
                info={
                    "episode_index": 0,
                    "episode_step": step,
                    "is_truncated_boundary": step == 9,
                    "episode_terminal_reason": "truncated" if step == 9 else "",
                }
            )
        )
    replay = IQLReplayBuffer(
        _FakeBase(storage, horizon),
        IQLConfig(action_horizon=horizon, device="cpu"),
    )
    with replay._base._lock:  # noqa: SLF001
        valid = replay._get_iql_valid_start_indices_locked()  # noqa: SLF001
    assert valid == [0, 1, 2, 3, 4, 5]


def test_iql_replay_keeps_success_terminal_chunk() -> None:
    horizon = 4
    storage = []
    for step in range(7):
        storage.append(
            _trans(
                done=step == 6,
                info={
                    "episode_index": 0,
                    "episode_step": step,
                    "episode_terminal_reason": "success" if step == 6 else "",
                    "is_truncated_boundary": False,
                },
            )
        )
    replay = IQLReplayBuffer(
        _FakeBase(storage, horizon),
        IQLConfig(action_horizon=horizon, device="cpu"),
    )
    with replay._base._lock:  # noqa: SLF001
        valid = replay._get_iql_valid_start_indices_locked()  # noqa: SLF001
    assert valid == [0, 1, 2, 3]
    _, forced_done = replay._next_obs_for(3)  # noqa: SLF001
    assert forced_done is True
