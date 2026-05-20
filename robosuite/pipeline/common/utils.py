from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .types import ReplayBatch


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot convert value of type {type(value)!r} to dict.")


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

    image = image.float()
    if image.max().item() > 1.0:
        image = image / 255.0
    return image


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def concat_replay_batches(first: ReplayBatch, second: ReplayBatch) -> ReplayBatch:
    return ReplayBatch(
        obs=_concat_tree(first.obs, second.obs),
        actions=torch.cat([first.actions, second.actions], dim=0),
        rewards=torch.cat([first.rewards, second.rewards], dim=0),
        next_obs=_concat_tree(first.next_obs, second.next_obs),
        dones=torch.cat([first.dones, second.dones], dim=0),
        grasp_penalty=torch.cat([first.grasp_penalty, second.grasp_penalty], dim=0),
        is_intervention=torch.cat([first.is_intervention, second.is_intervention], dim=0),
        metadata={
            "infos": list(first.metadata.get("infos", [])) + list(second.metadata.get("infos", [])),
            "reward_source": list(first.metadata.get("reward_source", []))
            + list(second.metadata.get("reward_source", [])),
            "demo_source": list(first.metadata.get("demo_source", []))
            + list(second.metadata.get("demo_source", [])),
            "indices": list(first.metadata.get("indices", [])) + list(second.metadata.get("indices", [])),
        },
    )


def _concat_tree(first: Any, second: Any) -> Any:
    if isinstance(first, Mapping):
        return {key: _concat_tree(first[key], second[key]) for key in first.keys()}
    return torch.cat([first, second], dim=0)


def infer_action_bounds(action_space: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if action_space is not None and hasattr(action_space, "low") and hasattr(action_space, "high"):
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if low.size == action_dim and high.size == action_dim:
            return low, high
    low = -np.ones(action_dim, dtype=np.float32)
    high = np.ones(action_dim, dtype=np.float32)
    return low, high


def infer_action_dim(action_space: Any = None, sample_action: Any = None, cfg: Any = None) -> int:
    if action_space is not None and hasattr(action_space, "shape") and action_space.shape is not None:
        return int(np.prod(action_space.shape))
    if sample_action is not None:
        return int(np.asarray(sample_action).reshape(-1).shape[0])
    action_dim = cfg_get(cfg, "action_dim", None)
    if action_dim is None:
        raise ValueError("Unable to infer action_dim. Provide an action space, sample action, or cfg.action_dim.")
    return int(action_dim)


def infer_observation_example(observation_space: Any = None, observation_example: Any = None) -> Any:
    if observation_example is not None:
        return clone_array_tree(observation_example)
    if observation_space is None:
        raise ValueError("Provide observation_example or observation_space to build pipeline modules.")
    if hasattr(observation_space, "sample"):
        return clone_array_tree(observation_space.sample())
    raise ValueError("observation_space does not expose a sample() method.")
