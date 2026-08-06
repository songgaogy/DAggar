import pytest

from robosuite.pipeline.src.environment.robosuite import render_mjviewer


class _FakeViewer:
    def __init__(self, events: list[object], *, fail: bool = False) -> None:
        self.events = events
        self.fail = bool(fail)

    def update(self) -> None:
        self.events.append("update")
        if self.fail:
            raise RuntimeError("viewer update failed")


class _FakeEnv:
    renderer = "mjviewer"
    _visualizations = {"env", "robots", "grippers"}

    def __init__(self, *, viewer_fails: bool = False) -> None:
        self.events: list[object] = []
        self.viewer = _FakeViewer(self.events, fail=viewer_fails)

    def visualize(self, *, vis_settings: dict[str, bool]) -> None:
        self.events.append(("visualize", vis_settings.copy()))


def test_gripper_markers_are_visible_only_during_viewer_update() -> None:
    env = _FakeEnv()

    render_mjviewer(env, visualize_gripper_markers=True)

    assert env.events == [
        (
            "visualize",
            {"env": False, "robots": False, "grippers": True},
        ),
        "update",
        (
            "visualize",
            {"env": False, "robots": False, "grippers": False},
        ),
    ]


def test_disabled_gripper_markers_leave_visualization_unchanged() -> None:
    env = _FakeEnv()

    render_mjviewer(env, visualize_gripper_markers=False)

    assert env.events == ["update"]


def test_gripper_markers_are_hidden_when_viewer_update_fails() -> None:
    env = _FakeEnv(viewer_fails=True)

    with pytest.raises(RuntimeError, match="viewer update failed"):
        render_mjviewer(env, visualize_gripper_markers=True)

    assert env.events[-1] == (
        "visualize",
        {"env": False, "robots": False, "grippers": False},
    )
