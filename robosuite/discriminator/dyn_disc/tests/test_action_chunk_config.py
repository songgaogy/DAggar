"""NumPy-only tests for nnPU action chunk configuration."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator
from robosuite.discriminator.dyn_disc.adapters.single_bank import (
    DynBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors.pu_bce import (
    PUBCEDiscriminator,
    pu_risk,
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


def test_cli_defaults_match_selected_health_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["robosuite_pu_bce", "--model-ckpt", "model.pth"])
    args = _parse_args()
    assert args.use_chunk is True
    assert args.pi_p == pytest.approx(0.3)
    assert args.head_hidden == 512
    assert args.head_layers == 3
    assert args.epochs == 1
    assert args.scheduler_horizon_epochs == 20
    assert args.loss_surrogate == "logistic"
    assert args.beta == pytest.approx(0.0)
    assert args.lr == pytest.approx(3e-4)
    assert args.weight_decay == pytest.approx(1e-4)
    assert args.batch_size == 512
    assert args.quadratic_cap_c == pytest.approx(2.0)
    assert args.quadratic_cap_lambda == pytest.approx(1e-2)
    assert args.max_fail_per_task == 50
    assert args.max_success_per_task == 50
    assert args.train_max_success_per_task == 50
    assert args.train_max_fail_per_task == 50
    assert args.delta == pytest.approx(10.0)
    assert args.calib_fraction == pytest.approx(0.2)
    assert args.knn_transformer_layer == 1
    assert args.seed == 0


def test_adapter_and_launcher_defaults_match_selected_health_config() -> None:
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            PUBCEBenchmarkDiscriminator.__init__
        ).parameters.items()
    }
    assert defaults["pi_p"] == pytest.approx(0.3)
    assert defaults["head_hidden"] == 512
    assert defaults["head_layers"] == 3
    assert defaults["epochs"] == 1
    assert defaults["scheduler_horizon_epochs"] == 20
    assert defaults["use_chunk"] is True
    assert defaults["loss_surrogate"] == "logistic"
    assert defaults["quadratic_cap_c"] == pytest.approx(2.0)
    assert defaults["quadratic_cap_lambda"] == pytest.approx(1e-2)
    assert defaults["lr"] == pytest.approx(3e-4)
    assert defaults["weight_decay"] == pytest.approx(1e-4)
    assert defaults["batch_size"] == 512
    assert defaults["delta"] == pytest.approx(10.0)
    assert defaults["calib_fraction"] == pytest.approx(0.2)
    assert defaults["transformer_layer"] == 1
    assert defaults["seed"] == 0

    fit_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(PUBCEDiscriminator.fit).parameters.items()
    }
    assert fit_defaults["pi_p"] == pytest.approx(0.3)
    assert fit_defaults["epochs"] == 1
    assert fit_defaults["scheduler_horizon_epochs"] == 20
    assert fit_defaults["loss_surrogate"] == "logistic"
    assert fit_defaults["quadratic_cap_c"] == pytest.approx(2.0)
    assert fit_defaults["quadratic_cap_lambda"] == pytest.approx(1e-2)
    assert inspect.signature(pu_risk).parameters["surrogate"].default == "logistic"

    launcher = Path(
        "robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh"
    ).read_text()
    for expected in (
        'EPOCHS="${EPOCHS:-1}"',
        'SCHEDULER_HORIZON_EPOCHS="${SCHEDULER_HORIZON_EPOCHS:-20}"',
        'QUADRATIC_CAP_C="${QUADRATIC_CAP_C:-2.0}"',
        'QUADRATIC_CAP_LAMBDA="${QUADRATIC_CAP_LAMBDA:-1e-2}"',
    ):
        assert expected in launcher


def test_cli_accepts_explicit_chunk_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["robosuite_pu_bce", "--model-ckpt", "model.pth", "--use-chunk", "True"],
    )
    assert _parse_args().use_chunk is True
