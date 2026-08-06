from types import SimpleNamespace

import numpy as np
import pytest

from robosuite.pipeline.src.environment import compute_grasp_penalty, estimate_gripper_openness


class FakePandaRobot:
    arms = ["right"]
    has_gripper = {"right": True}
    _ref_gripper_joint_pos_indexes = {"right": [0, 1]}
    _ref_joints_indexes_dict = {"PandaGripper": [0, 1]}

    @staticmethod
    def get_gripper_name(arm: str) -> str:
        assert arm == "right"
        return "PandaGripper"


def _fake_env(qpos: tuple[float, float]):
    sim = SimpleNamespace(
        data=SimpleNamespace(qpos=np.asarray(qpos, dtype=np.float32)),
        model=SimpleNamespace(
            jnt_range=np.asarray(
                [
                    [0.0, 0.04],
                    [-0.04, 0.0],
                ],
                dtype=np.float32,
            )
        ),
    )
    return SimpleNamespace(robots=[FakePandaRobot()], sim=sim)


@pytest.mark.parametrize(
    ("qpos", "expected"),
    [
        ((0.0, 0.0), 0.0),
        ((0.02, -0.02), 0.5),
        ((0.04, -0.04), 1.0),
    ],
)
def test_panda_mirrored_finger_ranges_produce_physical_openness(qpos, expected) -> None:
    assert estimate_gripper_openness(_fake_env(qpos)) == pytest.approx(expected)


def test_grasp_penalty_rejects_redundant_open_and_close_commands() -> None:
    closed_env = _fake_env((0.0, 0.0))
    open_env = _fake_env((0.04, -0.04))
    close_action = np.asarray([0.0] * 6 + [1.0], dtype=np.float32)
    open_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)

    assert compute_grasp_penalty(closed_env, close_action) == pytest.approx(-0.02)
    assert compute_grasp_penalty(open_env, open_action) == pytest.approx(-0.02)
    assert compute_grasp_penalty(closed_env, open_action) == pytest.approx(0.0)
    assert compute_grasp_penalty(open_env, close_action) == pytest.approx(0.0)
