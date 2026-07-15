"""Contract tests for offline discriminator finetuning inputs."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from robosuite.pipeline.offline.discriminator import (
    sha256_file,
    validate_finetune_contract,
)


def _contract_fixture(tmp_path):
    parent = tmp_path / "pu_bce_head.pth"
    model = tmp_path / "model.pth"
    normalizer = tmp_path / "normalizer.pth"
    parent.write_bytes(b"parent")
    model.write_bytes(b"model")
    normalizer.write_bytes(b"normalizer")

    parent_payload = {
        "feature_source": "transformer",
        "transformer_layer": 1,
        "use_chunk": True,
        "pu_bce_detector": {"in_dim": 6},
    }
    feature_contract = {
        "feature_source": "transformer",
        "transformer_layer": 1,
        "use_chunk": True,
        "latent_dim": 6,
        "view_names": ["front"],
        "camera_to_view": {"agentview": "front"},
        "proprio_indices": [0, 2],
        "proprio_input_dim": 2,
        "frameskip": 3,
        "action_dim_per_step": 2,
        "action_input_dim": 6,
    }
    manifest = {
        "task": "Task",
        "checkpoint": {
            "sha256": sha256_file(parent),
            "model_ckpt_sha256": sha256_file(model),
            "normalizer_ckpt_sha256": sha256_file(normalizer),
        },
        "feature_contract": feature_contract,
    }
    encoder = SimpleNamespace(
        encoder_checkpoint=str(model),
        normalizer_checkpoint=str(normalizer),
        feature_source="transformer",
        transformer_layer=1,
        use_chunk=True,
        chunk_feature_dim=6,
        inner_encoder=SimpleNamespace(view_names=["front"]),
        camera_to_view={"agentview": "front"},
        proprio_indices=[0, 2],
        proprio_input_dim=2,
        frameskip=3,
        action_dim_per_step=2,
        action_input_dim=6,
    )
    return parent, parent_payload, manifest, encoder


def test_finetune_contract_accepts_matching_manifest(tmp_path) -> None:
    parent, payload, manifest, encoder = _contract_fixture(tmp_path)

    validate_finetune_contract(
        task_name="Task",
        parent_checkpoint=parent,
        parent_payload=payload,
        pretrain_manifest=manifest,
        offline_payload={"task_name": "Task"},
        encoder=encoder,
    )


def test_finetune_contract_rejects_parent_checkpoint_mismatch(tmp_path) -> None:
    parent, payload, manifest, encoder = _contract_fixture(tmp_path)
    manifest = deepcopy(manifest)
    manifest["checkpoint"]["sha256"] = "wrong"

    with pytest.raises(ValueError, match="selected parent nnPU checkpoint"):
        validate_finetune_contract(
            task_name="Task",
            parent_checkpoint=parent,
            parent_payload=payload,
            pretrain_manifest=manifest,
            offline_payload={"task_name": "Task"},
            encoder=encoder,
        )


def test_finetune_contract_rejects_camera_contract_mismatch(tmp_path) -> None:
    parent, payload, manifest, encoder = _contract_fixture(tmp_path)
    encoder.camera_to_view = {"robot0_eye_in_hand": "front"}

    with pytest.raises(ValueError, match="encoder contract differs"):
        validate_finetune_contract(
            task_name="Task",
            parent_checkpoint=parent,
            parent_payload=payload,
            pretrain_manifest=manifest,
            offline_payload={"task_name": "Task"},
            encoder=encoder,
        )
