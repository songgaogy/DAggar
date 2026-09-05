from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from robosuite.pipeline.utils.tensor import resolve_cuda_device


def camera_obs_key(camera_name: str) -> str:
    return f"{camera_name}_image"


def _resolve_camera_key(observation: Mapping[str, object], camera_name: str) -> str:
    live_key = camera_obs_key(camera_name)
    if live_key in observation:
        return live_key
    if camera_name in observation:
        return camera_name
    raise KeyError(
        f"Observation is missing camera key '{live_key}' or legacy key '{camera_name}'"
    )


def observation_batch_to_cuda(
    observations: Mapping[str, object] | Sequence[Mapping[str, object]],
    camera_names: Sequence[str],
    proprio_key: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather robosuite HWC images and proprio without CPU numerical transforms."""
    cuda_device = resolve_cuda_device(device)
    batch = [observations] if isinstance(observations, Mapping) else list(observations)
    if not batch:
        raise ValueError("Observation batch must not be empty")
    if not camera_names:
        raise ValueError("camera_names must not be empty")

    image_rows = []
    proprio_rows = []
    for observation in batch:
        cameras = []
        for camera_name in camera_names:
            key = _resolve_camera_key(observation, str(camera_name))
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Camera '{key}' must be HWC RGB, got {image.shape}")
            cameras.append(torch.as_tensor(np.ascontiguousarray(image)).permute(2, 0, 1))
        if proprio_key not in observation:
            raise KeyError(f"Observation is missing proprio key '{proprio_key}'")
        proprio = np.asarray(observation[proprio_key], dtype=np.float32).reshape(-1)
        image_rows.append(torch.stack(cameras))
        proprio_rows.append(torch.from_numpy(proprio))

    images = torch.stack(image_rows).to(cuda_device, non_blocking=True)
    proprio = torch.stack(proprio_rows).to(cuda_device, non_blocking=True)
    return images, proprio
