from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.src.data.transitions import Transition, TransitionChunkWriter


def _transition(index: int, *, intervention: bool = False) -> Transition:
    return Transition(
        obs=np.asarray([index], dtype=np.float32),
        action=np.asarray([index], dtype=np.float32),
        reward=float(index),
        next_obs=np.asarray([index + 1], dtype=np.float32),
        done=False,
        is_intervention=intervention,
    )


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def test_async_writer_preserves_chunks_roles_and_payload(tmp_path: Path) -> None:
    writer = TransitionChunkWriter(tmp_path, chunk_size=2)
    transitions = [
        _transition(0, intervention=True),
        _transition(1),
        _transition(2, intervention=True),
    ]
    for transition in transitions:
        writer.append(transition)
    transitions[0].reward = 99.0
    writer.close()

    online_paths = sorted((tmp_path / "online_chunks").glob("chunk_*.pt"))
    demo_paths = sorted((tmp_path / "demo_chunks").glob("chunk_*.pt"))
    assert [path.name for path in online_paths] == ["chunk_000000.pt", "chunk_000001.pt"]
    assert [path.name for path in demo_paths] == ["chunk_000000.pt"]

    online = [_load(path) for path in online_paths]
    demo = _load(demo_paths[0])
    assert online[0]["version"] == 1
    assert online[0]["role"] == "online"
    assert [item.reward for chunk in online for item in chunk["transitions"]] == [0.0, 1.0, 2.0]
    assert demo["role"] == "demo"
    assert [item.reward for item in demo["transitions"]] == [0.0, 2.0]
    assert not list(tmp_path.rglob("*.tmp"))


def test_flush_waits_for_pending_write_and_close_is_idempotent(tmp_path: Path) -> None:
    writer = TransitionChunkWriter(tmp_path, chunk_size=10)
    writer.append(_transition(0))
    writer.flush("online")

    assert (tmp_path / "online_chunks" / "chunk_000000.pt").is_file()
    assert not list((tmp_path / "demo_chunks").glob("chunk_*.pt"))
    writer.close()
    writer.close()


def test_writer_continues_numbering_across_restarts(tmp_path: Path) -> None:
    with TransitionChunkWriter(tmp_path, chunk_size=1) as writer:
        writer.append(_transition(0))
    with TransitionChunkWriter(tmp_path, chunk_size=1) as writer:
        writer.append(_transition(1))

    paths = sorted((tmp_path / "online_chunks").glob("chunk_*.pt"))
    assert [path.name for path in paths] == ["chunk_000000.pt", "chunk_000001.pt"]


def test_bounded_queue_applies_backpressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_started = threading.Event()
    allow_save = threading.Event()
    original_save = torch.save

    def blocked_save(*args, **kwargs):
        save_started.set()
        assert allow_save.wait(timeout=2.0)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", blocked_save)
    writer = TransitionChunkWriter(tmp_path, chunk_size=1)
    writer.append(_transition(0))
    assert save_started.wait(timeout=2.0)
    writer.append(_transition(1))

    append_finished = threading.Event()
    thread = threading.Thread(
        target=lambda: (writer.append(_transition(2)), append_finished.set())
    )
    thread.start()
    time.sleep(0.1)
    assert not append_finished.is_set()

    allow_save.set()
    thread.join(timeout=2.0)
    assert append_finished.is_set()
    writer.close()


def test_background_failure_is_propagated_by_flush_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = OSError("disk full")

    def failing_save(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(torch, "save", failing_save)
    writer = TransitionChunkWriter(tmp_path, chunk_size=1)
    writer.append(_transition(0))

    with pytest.raises(RuntimeError, match="Transition chunk writer failed") as flush_error:
        writer.flush()
    assert flush_error.value.__cause__ is failure
    with pytest.raises(RuntimeError, match="Transition chunk writer failed") as close_error:
        writer.close()
    assert close_error.value.__cause__ is failure


def test_unknown_flush_role_is_rejected(tmp_path: Path) -> None:
    writer = TransitionChunkWriter(tmp_path)
    with pytest.raises(ValueError, match="Unknown transition role"):
        writer.flush("invalid")
    writer.close()
