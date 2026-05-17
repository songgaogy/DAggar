from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
from hydra.utils import to_absolute_path

# Legacy discriminator imports — these modules were removed on the `dipole` branch
# (init dipole commit). The classes that depend on them (LPBNewOnlineDiscriminator,
# TPUDOnlineDiscriminator, FlowMultiPolicyRuntime) will raise at instantiation time
# instead of breaking module import, which would otherwise block every pipeline entry
# point including train_dipole.py.
try:
    from robosuite.discriminator.bce.dataset import (  # type: ignore
        build_cached_splits as build_bce_cached_splits,
        filter_refs_by_data_types as filter_bce_refs_by_data_types,
        load_latent_trajectories as load_bce_latent_trajectories,
    )
    from robosuite.discriminator.bce.tpud_discriminator import TPUDDiscriminator  # type: ignore
except Exception:
    build_bce_cached_splits = None  # type: ignore
    filter_bce_refs_by_data_types = None  # type: ignore
    load_bce_latent_trajectories = None  # type: ignore
    TPUDDiscriminator = None  # type: ignore

try:
    from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder  # type: ignore
    from robosuite.discriminator.dyn_bce.task_registry import (  # type: ignore
        normalize_task_name,
        resolve_checkpoint_task_name,
    )
except Exception:
    FrozenFlowMultitaskEncoder = None  # type: ignore

    def normalize_task_name(name: str) -> str:  # type: ignore
        return str(name)

    def resolve_checkpoint_task_name(name: str) -> str:  # type: ignore
        return str(name)

try:
    from robosuite.discriminator.lpb_new.core.dataset import (  # type: ignore
        LatentTrajectory,
        build_cached_splits as build_lpb_cached_splits,
        filter_refs_by_data_types as filter_lpb_refs_by_data_types,
        load_latent_trajectories as load_lpb_latent_trajectories,
    )
    from robosuite.discriminator.lpb_new.core.knn_discriminator import LPBKNNDiscriminator  # type: ignore
except Exception:
    LatentTrajectory = None  # type: ignore
    build_lpb_cached_splits = None  # type: ignore
    filter_lpb_refs_by_data_types = None  # type: ignore
    load_lpb_latent_trajectories = None  # type: ignore
    LPBKNNDiscriminator = None  # type: ignore
from robosuite.policy.flow_multi.eval_flow import (
    center_crop_resize,
    resolve_language_instruction,
    sample_action_sequence,
)
from robosuite.policy.flow_multi.model import build_flow_policy
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _torch_load_checkpoint(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_camera_names(checkpoint_path: str) -> list[str]:
    payload = _torch_load_checkpoint(to_absolute_path(str(checkpoint_path)))
    return [str(name) for name in payload.get("camera_names", [])]


class BasePolicyRuntime:
    camera_names: list[str]

    def reset(self, initial_obs: dict, initial_images: dict[str, np.ndarray], env) -> None:
        raise NotImplementedError

    def update_history(self, obs: dict, step_images: dict[str, np.ndarray], env) -> None:
        raise NotImplementedError

    def notify_intervention(self) -> None:
        raise NotImplementedError

    def get_action(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DummyPolicyRuntime(BasePolicyRuntime):
    def __init__(self, action_dim: int, camera_names: Optional[Sequence[str]] = None):
        self.action_dim = int(action_dim)
        self.camera_names = [str(name) for name in (camera_names or [])]

    def reset(self, initial_obs: dict, initial_images: dict[str, np.ndarray], env) -> None:
        return None

    def update_history(self, obs: dict, step_images: dict[str, np.ndarray], env) -> None:
        return None

    def notify_intervention(self) -> None:
        return None

    def get_action(self) -> np.ndarray:
        return np.random.normal(0.0, 0.02, size=self.action_dim).astype(np.float32)


class FlowMultiPolicyRuntime(BasePolicyRuntime):
    def __init__(self, cfg: Any, env_name: str, env) -> None:
        self.cfg = cfg
        self.device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = to_absolute_path(str(cfg.ckpt))
        checkpoint = _torch_load_checkpoint(self.checkpoint_path)
        self.camera_names = [str(name) for name in checkpoint["camera_names"]]
        self.model = self._build_model(checkpoint).to(self.device)
        self.model.eval()

        self.act_mean = self._to_numpy(checkpoint.get("act_mean"))
        self.act_std = self._to_numpy(checkpoint.get("act_std"))
        self.prop_mean = self._to_numpy(checkpoint.get("prop_mean"))
        self.prop_std = self._to_numpy(checkpoint.get("prop_std"))
        self.action_horizon = int(np.asarray(self.act_mean).shape[0])
        self.action_dim = int(np.asarray(self.act_mean).shape[-1])
        requested_task_name = str(_cfg_get(cfg, "task_name", None) or env_name)
        self.canonical_task_name = normalize_task_name(requested_task_name)
        self.checkpoint_task_name = self._resolve_checkpoint_task_name(requested_task_name, checkpoint)
        self.language_instruction = resolve_language_instruction(
            checkpoint.get("task_prompt_map", {}),
            self.checkpoint_task_name,
        )
        self.execute_horizon = int(_cfg_get(cfg, "execute_horizon", self.action_horizon))
        if self.execute_horizon <= 0:
            self.execute_horizon = self.action_horizon
        self.execute_horizon = min(self.execute_horizon, self.action_horizon)
        self.n_ode_steps = int(_cfg_get(cfg, "n_ode_steps", 20))
        self.image_size = int(_cfg_get(cfg, "image_size", 128))
        self._image_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)
        self._image_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)

        self.proprio_extractor = self._bind_proprio_extractor(env=env, checkpoint=checkpoint)
        self.current_images: dict[str, np.ndarray] = {}
        self.current_proprio: Optional[np.ndarray] = None
        self.current_chunk: Optional[np.ndarray] = None
        self.step_in_chunk = 0

    @staticmethod
    def _to_numpy(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _resolve_checkpoint_task_name(self, task_name: str, checkpoint: dict) -> str:
        task_metadata_map = checkpoint.get("task_metadata_map", None)
        if isinstance(task_metadata_map, dict):
            candidate = resolve_checkpoint_task_name(task_name)
            if candidate in task_metadata_map:
                return str(candidate)
            if task_name in task_metadata_map:
                return str(task_name)
            if len(task_metadata_map) == 1:
                return str(next(iter(task_metadata_map.keys())))
        return str(task_name)

    def _resolve_env_metadata(self, checkpoint: dict) -> dict[str, Any]:
        task_metadata_map = checkpoint.get("task_metadata_map", None)
        if isinstance(task_metadata_map, dict) and self.checkpoint_task_name in task_metadata_map:
            return dict(task_metadata_map[self.checkpoint_task_name])
        env_metadata = checkpoint.get("env_metadata", None)
        if env_metadata is None:
            raise KeyError(
                f"Checkpoint {self.checkpoint_path} does not provide env metadata for task "
                f"{self.checkpoint_task_name}."
            )
        return dict(env_metadata)

    def _bind_proprio_extractor(self, env, checkpoint: dict) -> RobosuiteProprioExtractor:
        env_metadata = self._resolve_env_metadata(checkpoint)
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

    def _build_model(self, checkpoint: dict) -> torch.nn.Module:
        model_cfg = checkpoint["model_cfg"]
        action_dim = int(np.asarray(checkpoint["act_mean"]).shape[-1])
        proprio_dim = int(np.asarray(checkpoint["prop_mean"]).shape[-1])
        model = build_flow_policy(
            model_cfg,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            camera_names=[str(name) for name in checkpoint["camera_names"]],
        )
        state_dict = checkpoint.get("ema_model", checkpoint["model"])
        model.load_state_dict(state_dict, strict=True)
        return model

    def _extract_proprio(self, env) -> np.ndarray:
        proprio = self.proprio_extractor.extract(env.sim.get_state().flatten()).astype(np.float32)
        if self.prop_mean is not None and self.prop_std is not None:
            proprio = (proprio - self.prop_mean) / (self.prop_std + 1e-6)
        return proprio

    def _prepare_images(self) -> torch.Tensor:
        images = []
        for camera_name in self.camera_names:
            image = center_crop_resize(self.current_images[camera_name], self.image_size)
            image = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
            images.append(image)
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(self.device)
        return (image_tensor - self._image_mean) / self._image_std

    def reset(self, initial_obs: dict, initial_images: dict[str, np.ndarray], env) -> None:
        self.current_images = {
            camera_name: np.asarray(initial_images[camera_name], dtype=np.uint8)
            for camera_name in self.camera_names
        }
        self.current_proprio = self._extract_proprio(env)
        self.current_chunk = None
        self.step_in_chunk = 0

    def update_history(self, obs: dict, step_images: dict[str, np.ndarray], env) -> None:
        self.current_images = {
            camera_name: np.asarray(step_images[camera_name], dtype=np.uint8)
            for camera_name in self.camera_names
        }
        self.current_proprio = self._extract_proprio(env)

    def notify_intervention(self) -> None:
        self.current_chunk = None
        self.step_in_chunk = 0

    def get_action(self) -> np.ndarray:
        if self.current_proprio is None:
            raise RuntimeError("Call reset(...) before get_action().")
        if self.current_chunk is None or self.step_in_chunk >= self.execute_horizon:
            images = self._prepare_images()
            proprio = torch.from_numpy(self.current_proprio.astype(np.float32)).unsqueeze(0).to(self.device)
            action_seq = sample_action_sequence(
                model=self.model,
                images=images,
                proprio=proprio,
                language=[self.language_instruction],
                action_horizon=self.action_horizon,
                n_steps=self.n_ode_steps,
            )[0].detach().cpu().numpy().astype(np.float32)
            if self.act_mean is not None and self.act_std is not None:
                action_seq = action_seq * self.act_std + self.act_mean
            self.current_chunk = action_seq
            self.step_in_chunk = 0

        action = np.asarray(self.current_chunk[self.step_in_chunk], dtype=np.float32)
        self.step_in_chunk += 1
        return action

    def close(self) -> None:
        if self.proprio_extractor is not None:
            try:
                self.proprio_extractor.close()
            except Exception:
                pass


@dataclass
class OnlineDiscriminatorDecision:
    evaluated: bool = False
    score: float = float("nan")
    threshold: float = float("nan")
    prediction: int = 0
    available_steps: int = 0
    raw_step_score: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseOnlineDiscriminator:
    name: str
    camera_names: list[str]

    def reset(
        self,
        task_name: str,
        initial_state: np.ndarray,
        initial_images: dict[str, np.ndarray],
    ) -> None:
        raise NotImplementedError

    def record_step(
        self,
        action: np.ndarray,
        next_state: np.ndarray,
        next_images: dict[str, np.ndarray],
    ) -> OnlineDiscriminatorDecision:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullOnlineDiscriminator(BaseOnlineDiscriminator):
    def __init__(self) -> None:
        self.name = "none"
        self.camera_names = []

    def reset(
        self,
        task_name: str,
        initial_state: np.ndarray,
        initial_images: dict[str, np.ndarray],
    ) -> None:
        return None

    def record_step(
        self,
        action: np.ndarray,
        next_state: np.ndarray,
        next_images: dict[str, np.ndarray],
    ) -> OnlineDiscriminatorDecision:
        return OnlineDiscriminatorDecision(evaluated=False)


class OfflinePrefixOnlineDiscriminator(BaseOnlineDiscriminator):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.seed = int(_cfg_get(cfg, "seed", 42))
        self.encoder = FrozenFlowMultitaskEncoder(
            checkpoint_path=str(cfg.policy.ckpt),
            device=str(cfg.policy.device),
            image_size=int(cfg.data.image_size),
            batch_size=int(cfg.policy.encoder_batch_size),
        )
        self.camera_names = list(self.encoder.camera_names)
        self.eval_interval = max(1, int(_cfg_get(cfg.monitor, "eval_interval", 1)))
        self.min_history = max(1, int(_cfg_get(cfg.monitor, "min_history", 1)))
        self.adaptive_threshold = bool(_cfg_get(cfg.online, "adaptive_delta", False))
        self.delta_min = float(_cfg_get(cfg.online, "delta_min", 0.0))
        self.delta_max = float(_cfg_get(cfg.online, "delta_max", 100.0))
        self.warmup_steps = int(_cfg_get(cfg.online, "warmup_steps", 0))
        self.update_interval = int(_cfg_get(cfg.online, "update_interval", 1))
        self.detector = self._build_and_fit_detector()
        self.required_actions = max(self.min_history, self._detector_required_actions())

        self.task_name = ""
        self.task_index = -1
        self.state_history: list[np.ndarray] = []
        self.image_history: dict[str, list[np.ndarray]] = {}
        self.action_history: list[np.ndarray] = []
        self.last_eval_action_count = 0
        self.last_decision = OnlineDiscriminatorDecision(evaluated=False)

    def _detector_required_actions(self) -> int:
        horizon = getattr(self.detector, "action_horizon", None)
        if horizon is None and hasattr(self.detector, "extractor"):
            horizon = getattr(self.detector.extractor, "action_horizon", None)
        if horizon is None:
            return 1
        return int(horizon) + 1

    def _build_and_fit_detector(self):
        raise NotImplementedError

    def _task_to_index(self, task_name: str) -> int:
        raise NotImplementedError

    def _detect_prefix(self, trajectory: LatentTrajectory):
        return self.detector.detect_trajectory(
            trajectory,
            adaptive_threshold=self.adaptive_threshold,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            warmup_steps=self.warmup_steps,
            update_interval=self.update_interval,
        )

    def reset(
        self,
        task_name: str,
        initial_state: np.ndarray,
        initial_images: dict[str, np.ndarray],
    ) -> None:
        self.task_name = normalize_task_name(task_name)
        self.task_index = self._task_to_index(self.task_name)
        self.state_history = [np.asarray(initial_state, dtype=np.float32).reshape(-1)]
        self.image_history = {
            camera_name: [np.asarray(initial_images[camera_name], dtype=np.uint8)]
            for camera_name in self.camera_names
        }
        self.action_history = []
        self.last_eval_action_count = 0
        self.last_decision = OnlineDiscriminatorDecision(evaluated=False)

    def _build_prefix_trajectory(self) -> Optional[LatentTrajectory]:
        num_actions = len(self.action_history)
        if num_actions < self.required_actions:
            return None

        states = np.stack(self.state_history[:-1], axis=0).astype(np.float32)
        actions = np.stack(self.action_history, axis=0).astype(np.float32)
        images_hwc = np.stack(
            [
                np.stack(self.image_history[camera_name][:-1], axis=0).astype(np.uint8)
                for camera_name in self.camera_names
            ],
            axis=1,
        )
        latents = self.encoder.encode_arrays(
            task_name=self.task_name,
            states=states,
            images_hwc=images_hwc,
        )
        return LatentTrajectory(
            latents=latents,
            actions=actions,
            task_name=self.task_name,
            task_index=int(self.task_index),
            data_type="online",
            data_type_index=-1,
            split="online",
            file_path="",
            demo_key="",
        )

    def record_step(
        self,
        action: np.ndarray,
        next_state: np.ndarray,
        next_images: dict[str, np.ndarray],
    ) -> OnlineDiscriminatorDecision:
        self.action_history.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.state_history.append(np.asarray(next_state, dtype=np.float32).reshape(-1))
        for camera_name in self.camera_names:
            self.image_history[camera_name].append(np.asarray(next_images[camera_name], dtype=np.uint8))

        num_actions = len(self.action_history)
        should_eval = num_actions >= self.required_actions and (
            self.last_eval_action_count == 0 or (num_actions - self.last_eval_action_count) >= self.eval_interval
        )
        if not should_eval:
            return OnlineDiscriminatorDecision(
                evaluated=False,
                score=float(self.last_decision.score),
                threshold=float(self.last_decision.threshold),
                prediction=int(self.last_decision.prediction),
                available_steps=int(self.last_decision.available_steps),
                raw_step_score=float(self.last_decision.raw_step_score),
                metadata=dict(self.last_decision.metadata),
            )

        trajectory = self._build_prefix_trajectory()
        if trajectory is None:
            return OnlineDiscriminatorDecision(evaluated=False)

        result = self._detect_prefix(trajectory)
        last_index = int(result.aggregate_scores.shape[0]) - 1
        decision = OnlineDiscriminatorDecision(
            evaluated=True,
            score=float(result.aggregate_scores[last_index]),
            threshold=float(result.thresholds[last_index]),
            prediction=int(result.predictions[last_index]),
            available_steps=int(result.aggregate_scores.shape[0]),
            raw_step_score=float(result.step_scores[last_index]),
            metadata={
                "detector_name": str(result.detector_name),
                "task_name": str(self.task_name),
                "threshold_final": float(result.metadata.get("threshold_final", result.thresholds[last_index])),
                "delta_final": float(result.metadata.get("delta_final", np.nan)),
            },
        )
        self.last_decision = decision
        self.last_eval_action_count = num_actions
        return decision

    def close(self) -> None:
        try:
            self.detector.close()
        except Exception:
            pass
        self.encoder.close()


class LPBNewOnlineDiscriminator(OfflinePrefixOnlineDiscriminator):
    def __init__(self, cfg: Any) -> None:
        self.name = "lpb_new"
        self._task_to_index_map: dict[str, int] = {}
        super().__init__(cfg)

    def _build_and_fit_detector(self):
        cached_splits, _, task_to_index = build_lpb_cached_splits(
            cfg_data=self.cfg.data,
            encoder=self.encoder,
            seed=self.seed,
        )
        self._task_to_index_map = {str(task_name): int(idx) for task_name, idx in task_to_index.items()}
        bank_refs = filter_lpb_refs_by_data_types(
            cached_splits[str(self.cfg.eval.bank_split)],
            list(self.cfg.eval.bank_data_types),
        )
        calibration_refs = filter_lpb_refs_by_data_types(
            cached_splits[str(self.cfg.eval.calibration_split)],
            list(self.cfg.eval.calibration_data_types),
        )
        bank_trajectories = load_lpb_latent_trajectories(bank_refs)
        calibration_trajectories = load_lpb_latent_trajectories(calibration_refs)
        if not bank_trajectories:
            raise RuntimeError("No bank trajectories found for online LPB discriminator.")
        if not calibration_trajectories:
            raise RuntimeError("No calibration trajectories found for online LPB discriminator.")

        detector = LPBKNNDiscriminator(
            checkpoint_path=to_absolute_path(str(self.cfg.model.lpb_ckpt)),
            feature_device=str(self.cfg.feature.device),
            feature_batch_size=int(self.cfg.feature.batch_size),
            action_horizon=int(self.cfg.feature.action_horizon),
            normalize_feature=bool(self.cfg.feature.normalize_feature),
            normalize_policy_chunk=bool(self.cfg.feature.normalize_policy_chunk),
            use_transition_error=bool(self.cfg.feature.use_transition_error),
            detector_device=str(self.cfg.detector.device),
            delta=float(self.cfg.detector.delta),
            delta_step=float(self.cfg.detector.delta_step),
            knn_chunk_size=int(self.cfg.detector.knn_chunk_size),
            lambda_mode=str(self.cfg.detector.lambda_mode),
            lambda_window_size=int(self.cfg.detector.lambda_window_size),
            feature_knn_weight=float(self.cfg.detector.feature_knn_weight),
            transition_aux_weight=float(self.cfg.detector.transition_aux_weight),
            policy_chunk_weight=float(self.cfg.detector.policy_chunk_weight),
            dynamics_weight=float(self.cfg.detector.dynamics_weight),
            neighbor_topk=int(self.cfg.detector.neighbor_topk),
            dynamics_temperature=float(self.cfg.detector.dynamics_temperature),
        )
        detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )
        return detector

    def _task_to_index(self, task_name: str) -> int:
        if task_name not in self._task_to_index_map:
            raise KeyError(
                f"Task '{task_name}' was not part of the online LPB build. "
                f"Available tasks: {sorted(self._task_to_index_map.keys())}"
            )
        return int(self._task_to_index_map[task_name])


class TPUDOnlineDiscriminator(OfflinePrefixOnlineDiscriminator):
    def __init__(self, cfg: Any) -> None:
        self.name = "bce"
        self._task_to_index_map: dict[str, int] = {}
        super().__init__(cfg)

    def _build_and_fit_detector(self):
        cached_splits, _, task_to_index = build_bce_cached_splits(
            cfg_data=self.cfg.data,
            encoder=self.encoder,
            seed=self.seed,
        )
        self._task_to_index_map = {str(task_name): int(idx) for task_name, idx in task_to_index.items()}
        bank_refs = filter_bce_refs_by_data_types(
            cached_splits[str(self.cfg.eval.bank_split)],
            list(self.cfg.eval.bank_data_types),
        )
        calibration_refs = filter_bce_refs_by_data_types(
            cached_splits[str(self.cfg.eval.calibration_split)],
            list(self.cfg.eval.calibration_data_types),
        )
        bank_trajectories = load_bce_latent_trajectories(bank_refs)
        calibration_trajectories = load_bce_latent_trajectories(calibration_refs)
        if not bank_trajectories:
            raise RuntimeError("No bank trajectories found for online TPUD discriminator.")
        if not calibration_trajectories:
            raise RuntimeError("No calibration trajectories found for online TPUD discriminator.")

        detector = TPUDDiscriminator(
            checkpoint_path=to_absolute_path(str(self.cfg.model.tpud_ckpt)),
            device=str(self.cfg.detector.device),
            batch_size=int(self.cfg.detector.batch_size),
            action_horizon=int(self.cfg.detector.action_horizon),
            aggregate_mode=str(self.cfg.detector.aggregate_mode),
            default_delta=float(self.cfg.detector.default_delta),
            task_delta={str(k): float(v) for k, v in self.cfg.detector.task_delta.items()},
            delta_step=float(self.cfg.detector.delta_step),
            moving_mean_window=int(self.cfg.detector.moving_mean_window),
            ema_alpha=float(self.cfg.detector.ema_alpha),
            min_persistence=int(self.cfg.detector.min_persistence),
            decision_warmup_steps=int(self.cfg.detector.decision_warmup_steps),
        )
        detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )
        return detector

    def _task_to_index(self, task_name: str) -> int:
        if task_name not in self._task_to_index_map:
            raise KeyError(
                f"Task '{task_name}' was not part of the online TPUD build. "
                f"Available tasks: {sorted(self._task_to_index_map.keys())}"
            )
        return int(self._task_to_index_map[task_name])
