from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import robosuite.macros as macros
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, camera_obs_key
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

from .robosuite import reset_robosuite_env


def bind_proprio_extractor(
    env,
    env_metadata: dict[str, Any] | None = None,
) -> RobosuiteProprioExtractor:
    if env_metadata is None:
        extractor = RobosuiteProprioExtractor.__new__(RobosuiteProprioExtractor)
    else:
        extractor = RobosuiteProprioExtractor(
            env_kwargs=env_metadata,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names=None,
            reward_shaping=False,
        )
        extractor.close()
    extractor.env = env
    extractor.sim = env.sim
    extractor._build_robot_joint_indices()
    return extractor


def normalize_policy_observation(
    observation: Mapping[str, Any],
    *,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
) -> dict[str, np.ndarray]:
    normalized = {"state": np.asarray(observation["state"], dtype=np.float32)}
    for camera_name in camera_names:
        source_name = camera_aliases.get(camera_name, camera_name)
        if camera_name in observation:
            normalized[camera_name] = np.asarray(observation[camera_name], dtype=np.uint8)
        elif source_name in observation:
            normalized[camera_name] = np.asarray(observation[source_name], dtype=np.uint8)
        else:
            raise KeyError(f"Missing policy camera '{camera_name}' (source '{source_name}').")
    return normalized


def build_policy_observation(
    env,
    *,
    extractor: RobosuiteProprioExtractor,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
    image_height: int,
    image_width: int,
    raw_observation: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    observation: dict[str, Any] = {
        "state": extractor.extract(env.sim.get_state().flatten()).astype(np.float32)
    }
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    for camera_name in camera_names:
        source_name = camera_aliases.get(camera_name, camera_name)
        source_key = camera_obs_key(source_name)
        if raw_observation is not None and source_key in raw_observation:
            image = np.asarray(raw_observation[source_key], dtype=np.uint8)
        else:
            image = np.asarray(
                env.sim.render(
                    height=int(image_height),
                    width=int(image_width),
                    camera_name=source_name,
                )[::convention],
                dtype=np.uint8,
            )
        observation[camera_name] = center_crop_resize(image, image_height, image_width)
    return observation


def reset_policy_observation(
    env,
    *,
    preserve_mjviewer: bool,
    extractor: RobosuiteProprioExtractor,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
    image_height: int,
    image_width: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_observation, info = reset_robosuite_env(env, preserve_mjviewer=preserve_mjviewer)
    return (
        build_policy_observation(
            env,
            extractor=extractor,
            camera_names=camera_names,
            camera_aliases=camera_aliases,
            image_height=image_height,
            image_width=image_width,
            raw_observation=raw_observation,
        ),
        info,
    )


def center_crop_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    crop_size = min(image_height, image_width)
    y0 = (image_height - crop_size) // 2
    x0 = (image_width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop.shape[:2] == (height, width):
        return np.asarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, int(height)).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, int(width)).astype(np.int32)
    return np.asarray(crop[ys][:, xs], dtype=np.uint8)


def center_crop_resize_batch(images: np.ndarray, height: int, width: int) -> np.ndarray:
    image_height, image_width = images.shape[1:3]
    crop_size = min(image_height, image_width)
    y0 = (image_height - crop_size) // 2
    x0 = (image_width - crop_size) // 2
    crop = images[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop.shape[1:3] == (height, width):
        return np.ascontiguousarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, int(height)).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, int(width)).astype(np.int32)
    return np.ascontiguousarray(crop[:, ys][:, :, xs], dtype=np.uint8)
