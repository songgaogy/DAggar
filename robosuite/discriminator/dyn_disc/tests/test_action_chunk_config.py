"""NumPy-only tests for nnPU action chunk configuration."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite.discriminator.dyn_disc.adapters.single_bank import (
    DynBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors.single_bank_knn import DynEncoder
from robosuite.discriminator.dyn_disc.robosuite_pu_bce import _parse_args, _parse_bool


def _adapter(*, use_chunk: bool) -> DynBenchmarkDiscriminator:
    encoder = object.__new__(DynEncoder)
    encoder.frameskip = 3
    encoder.action_dim_per_step = 2
    encoder.action_input_dim = 6
    encoder.model = SimpleNamespace(
        action_encoder=SimpleNamespace(in_chans=encoder.action_input_dim)
    )

    adapter = object.__new__(DynBenchmarkDiscriminator)
    adapter.feature_source = "transformer"
    adapter.transformer_layer = 1
    adapter.use_chunk = use_chunk
    adapter.encoder = encoder
    return adapter


def test_transformer_actions_use_zero_padding_when_chunk_is_disabled() -> None:
    adapter = _adapter(use_chunk=False)
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    prepared = adapter._prepare_actions(actions, t_len=3)

    np.testing.assert_array_equal(
        prepared,
        np.asarray(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_transformer_actions_use_forward_chunks_and_repeat_tail_when_enabled() -> None:
    adapter = _adapter(use_chunk=True)
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    prepared = adapter._prepare_actions(actions, t_len=3)

    np.testing.assert_array_equal(
        prepared,
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 5.0, 6.0],
                [5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_chunk_mode_is_part_of_trajectory_cache_key() -> None:
    trajectory = SimpleNamespace(
        file_path="rollout.hdf5",
        cache_npz_path="cache.npz",
    )
    zero_pad_key = _adapter(use_chunk=False)._trajectory_key(trajectory)
    chunk_key = _adapter(use_chunk=True)._trajectory_key(trajectory)

    assert zero_pad_key != chunk_key
    assert zero_pad_key[-1] is False
    assert chunk_key[-1] is True


@pytest.mark.parametrize("value", ["True", "true", "1", "yes", "on"])
def test_parse_bool_accepts_true_values(value: str) -> None:
    assert _parse_bool(value) is True


@pytest.mark.parametrize("value", ["False", "false", "0", "no", "off"])
def test_parse_bool_accepts_false_values(value: str) -> None:
    assert _parse_bool(value) is False


def test_parse_bool_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Expected a boolean value"):
        _parse_bool("maybe")


def test_cli_defaults_to_zero_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["robosuite_pu_bce", "--model-ckpt", "model.pth"])
    assert _parse_args().use_chunk is False


def test_cli_accepts_explicit_chunk_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["robosuite_pu_bce", "--model-ckpt", "model.pth", "--use-chunk", "True"],
    )
    assert _parse_args().use_chunk is True
