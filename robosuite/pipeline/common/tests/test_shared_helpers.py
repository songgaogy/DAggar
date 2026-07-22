"""Import smoke tests for helpers extracted from legacy entrypoints."""

import pytest

from robosuite.pipeline.common import flow
from robosuite.pipeline.modules.evaluation import policy_runtime
from robosuite.pipeline.utils import resolve_requested_device


def test_shared_helper_modules_export_batch_pipeline_dependencies() -> None:
    assert callable(flow.load_hdf5_demos_into_flow_transitions)
    assert callable(flow.build_flow_runtime_cfg)
    assert callable(flow.load_init_checkpoint_payload)
    assert callable(flow.merge_checkpoint_model_config)
    assert callable(policy_runtime._build_dipole_policy)
    assert callable(policy_runtime._write_video)


def test_checkpoint_model_merge_preserves_snapshotted_assets() -> None:
    flow_cfg = {
        "model": {
            "image_encoder": {"pretrained_path": "/run/inputs/resnet.pth"},
            "language_encoder": {"pretrained_name": "/run/inputs/clip"},
        }
    }
    payload = {
        "model_cfg": {
            "image_encoder": {
                "type": "resnet18",
                "pretrained_path": "/external/resnet.pth",
            },
            "language_encoder": {
                "type": "clip_text",
                "pretrained_name": "/external/clip",
            },
        }
    }

    flow.merge_checkpoint_model_config(flow_cfg, payload)

    assert flow_cfg["model"]["image_encoder"]["type"] == "resnet18"
    assert flow_cfg["model"]["image_encoder"]["pretrained_path"] == "/run/inputs/resnet.pth"
    assert flow_cfg["model"]["language_encoder"]["pretrained_name"] == "/run/inputs/clip"


def test_device_resolution_has_no_cpu_fallback(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_requested_device("cpu", fallback="cuda:0")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_requested_device("cuda:0", fallback="cuda:0")
