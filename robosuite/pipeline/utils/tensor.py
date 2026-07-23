from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def clone_array_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clone_array_tree(item) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (np.generic, float, int, bool)):
        return np.asarray(value).copy()
    return copy.deepcopy(value)


def to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def stack_tree(items: list[Any]) -> Any:
    if len(items) == 0:
        raise ValueError("Cannot stack an empty list.")
    first = items[0]
    if isinstance(first, Mapping):
        keys = list(first.keys())
        return {key: stack_tree([item[key] for item in items]) for key in keys}
    arrays = [to_numpy(item) for item in items]
    return np.stack(arrays, axis=0)


def tree_shapes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: tree_shapes(item) for key, item in value.items()}
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    return ()


def assert_same_structure(reference: Any, value: Any, path: str = "root") -> None:
    if isinstance(reference, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} expected mapping, got {type(value)!r}.")
        if set(reference.keys()) != set(value.keys()):
            raise ValueError(
                f"{path} keys mismatch. Expected {sorted(reference.keys())}, got {sorted(value.keys())}."
            )
        for key in reference:
            assert_same_structure(reference[key], value[key], path=f"{path}.{key}")
        return

    ref_shape = tree_shapes(reference)
    value_shape = tree_shapes(value)
    if ref_shape != value_shape:
        raise ValueError(f"{path} shape mismatch. Expected {ref_shape}, got {value_shape}.")


def nested_to_torch(value: Any, device: torch.device | str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {str(key): nested_to_torch(item, device=device) for key, item in value.items()}
    tensor = torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def require_cuda_device(device: torch.device | str, *, name: str = "device") -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"{name} must be a CUDA device, got {resolved}.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{name}={resolved} requested, but CUDA is unavailable.")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{name}={resolved} does not exist; detected {torch.cuda.device_count()} CUDA device(s)."
        )
    return torch.device("cuda", index)


def random_crop_observations(
    observations: Any,
    image_keys: Sequence[str],
    *,
    padding: int = 4,
) -> Any:
    if not isinstance(observations, Mapping):
        return observations
    augmented = dict(observations)
    for key in image_keys:
        if key in augmented:
            augmented[key] = _batched_random_crop(augmented[key], padding=padding)
    return augmented


def _batched_random_crop(images: torch.Tensor, *, padding: int) -> torch.Tensor:
    if padding <= 0:
        return images
    if images.ndim < 4:
        raise ValueError(f"Expected a batched image tensor, got shape {tuple(images.shape)}.")

    channels_last = images.shape[-1] in (1, 3, 4)
    if channels_last:
        leading_shape = images.shape[:-3]
        height, width, channels = images.shape[-3:]
        flattened = images.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    elif images.shape[-3] in (1, 3, 4):
        leading_shape = images.shape[:-3]
        channels, height, width = images.shape[-3:]
        flattened = images.reshape(-1, channels, height, width)
    else:
        raise ValueError(f"Cannot identify image channels in shape {tuple(images.shape)}.")

    padded = torch.nn.functional.pad(flattened, (padding, padding, padding, padding), mode="replicate")
    offsets = torch.randint(
        0,
        2 * padding + 1,
        (flattened.shape[0], 2),
        device=images.device,
    )
    padded_nhwc = padded.permute(0, 2, 3, 1)
    rows = offsets[:, :1] + torch.arange(height, device=images.device).view(1, -1)
    columns = offsets[:, 1:] + torch.arange(width, device=images.device).view(1, -1)
    batch_indices = torch.arange(flattened.shape[0], device=images.device).view(-1, 1, 1)
    cropped_nhwc = padded_nhwc[
        batch_indices,
        rows.unsqueeze(-1),
        columns.unsqueeze(-2),
    ]
    if channels_last:
        return cropped_nhwc.reshape(*leading_shape, height, width, channels)
    return cropped_nhwc.permute(0, 3, 1, 2).reshape(*leading_shape, channels, height, width)


def standardize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim < 3:
        raise ValueError(f"Expected image tensor with at least 3 dims, got shape {tuple(image.shape)}.")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.ndim == 4:
        if image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)
    elif image.ndim == 5:
        if image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 1, 4, 2, 3)
        image = image.flatten(1, 2)
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}.")

    scale_uint8 = not image.is_floating_point()
    image = image.float()
    if scale_uint8:
        image = image / 255.0
    return image


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
