import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, Any, Tuple, Optional, List
from collections import OrderedDict
import time

import numpy as np
import torch

from .async_failure_detector import AsyncFailureDetectionModule
from ..utils.macros import INTV
from .STAC.error_utils import compute_temporal_error


class STAC(AsyncFailureDetectionModule):
    """
    STAC (Statistical measures of Temporal Action Consistency) failure/OOD detector.
    """

    def __init__(
        self,
        max_queue_size: int = 3,
        num_samples: int = 8,
        pred_horizon: Optional[int] = None,
        exec_horizon: Optional[int] = None,
        error_fn: str = "mmd_rbf",
        aggr_fn: str = "mean",
        ignore_gripper: bool = True,
        ignore_rotation: bool = True,
        sim_freq: float = 5.0,
        num_robots: int = 2,
        action_dim: int = 4,
        stac_percentile: float = 95.0,
        fixed_threshold: Optional[float] = None,
        update_stats: bool = False,
        rewind_chunks: int = 3,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)

        # Runtime-initialized.
        self.device: Optional[torch.device] = None
        self.policy: Optional[Any] = None
        self.episode_manager: Optional[Any] = None
        self.max_episode_length: Optional[int] = None

        # STAC config.
        self.num_samples = int(num_samples)
        self.pred_horizon = pred_horizon
        self.exec_horizon = exec_horizon
        self.error_fn = str(error_fn)
        self.aggr_fn = str(aggr_fn)
        self.ignore_gripper = bool(ignore_gripper)
        self.ignore_rotation = bool(ignore_rotation)
        self.sim_freq = float(sim_freq)
        self.num_robots = int(num_robots)
        self.action_dim = int(action_dim)

        # Thresholding.
        self.stac_percentile = float(stac_percentile)
        self.fixed_threshold = fixed_threshold
        self.update_stats = bool(update_stats)

        # Rewind behavior.
        self.rewind_chunks = int(rewind_chunks)
        self._rewind_start_timestep: Optional[int] = None

        # Online state.
        self._prev_samples: Optional[np.ndarray] = None
        self._prev_boundary_idx: Optional[int] = None

        self.latest_result_idx: int = 0
        self.failure_logs: "OrderedDict[int, str]" = OrderedDict()
        self.score_history: "OrderedDict[int, float]" = OrderedDict()

        # Success statistics for adaptive thresholding.
        self.success_scores: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.stac_threshold: Optional[float] = None

    def runtime_initialize(
        self,
        device: torch.device,
        policy: torch.nn.Module,
        replay_buffer: Any,
        episode_manager: Any,
        max_episode_length: int,
    ) -> None:
        self.device = device
        self.policy = policy
        self.episode_manager = episode_manager
        self.max_episode_length = int(max_episode_length)

        if self.exec_horizon is None:
            if hasattr(policy, "Ta"):
                self.exec_horizon = int(policy.Ta)
            else:
                self.exec_horizon = 8

        if self.fixed_threshold is not None:
            self.stac_threshold = float(self.fixed_threshold)

        self.start_async_processing()

    def cleanup(self) -> None:
        self.stop_async_processing()

    # Async task handler
    def handle_async_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task_type = task.get("task_type")

        if task_type == "stac_compute":
            if self.policy is None or self.device is None:
                return None

            idx: int = int(task["idx"])
            policy_obs: Dict[str, torch.Tensor] = task["policy_obs"]

            samples = self._sample_action_sequences(policy_obs, self.num_samples)

            pred_horizon = int(self.pred_horizon) if self.pred_horizon is not None else int(samples.shape[1])
            pred_horizon = min(pred_horizon, int(samples.shape[1]))
            if pred_horizon <= exec_horizon:
                print(f"[STAC Warning] Prediction horizon ({pred_horizon}) <= Execution horizon ({exec_horizon}). "
                    "STAC requires overlap. Skipping check.")
                return None

            exec_horizon = int(self.exec_horizon) if self.exec_horizon is not None else 8

            score: Optional[float] = None
            if self._prev_samples is not None:
                err = compute_temporal_error(
                    error_fn=self.error_fn,
                    curr_action=samples,
                    prev_action=self._prev_samples,
                    pred_horizon=pred_horizon,
                    exec_horizon=exec_horizon,
                    ignore_gripper=self.ignore_gripper,
                    ignore_rotation=self.ignore_rotation,
                    sim_freq=self.sim_freq,
                    num_robots=self.num_robots,
                    action_dim=self.action_dim,
                )
                score = float(self._aggregate_error(err, self.aggr_fn))

            self._prev_samples = samples
            self._prev_boundary_idx = idx

            if score is None:
                return {"task_type": "stac_score", "idx": idx, "score": None}

            return {"task_type": "stac_score", "idx": idx, "score": score}

        return None

    # Public API used by runner
    def process_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        step_type = step_data["step_type"]

        if step_type == "episode_start":
            self.failure_logs = OrderedDict()
            self.score_history = OrderedDict()
            self.latest_result_idx = 0
            self._prev_samples = None
            self._prev_boundary_idx = None
            self._rewind_start_timestep = None
            return {}

        if step_type == "policy_step":
            timestep = int(step_data.get("timestep", 0))
            exec_horizon = int(self.exec_horizon) if self.exec_horizon is not None else 8

            if timestep < exec_horizon or (timestep % exec_horizon) != 0:
                return {}

            if self.episode_manager is None:
                return {}

            policy_obs = self.episode_manager.get_policy_observation()
            policy_obs = {k: v.clone() for k, v in policy_obs.items()}

            idx = timestep // exec_horizon - 1
            self.submit_task({"task_type": "stac_compute", "idx": idx, "policy_obs": policy_obs})
            return {}

        return {}

    def detect_failure(self, **kwargs) -> Tuple[bool, Optional[str], int]:
        timestep = int(kwargs.get("timestep", 0))
        max_episode_length = int(kwargs.get("max_episode_length", self.max_episode_length or 0))
        exec_horizon = int(self.exec_horizon) if self.exec_horizon is not None else 8
        idx = timestep // exec_horizon - 1

        results = self.get_results()

        failure_flag = False
        failure_reason: Optional[str] = None

        for result in results:
            if result.get("task_type") != "stac_score":
                continue

            res_idx = int(result["idx"])
            if res_idx > idx:
                continue

            self.latest_result_idx = max(self.latest_result_idx, res_idx)

            score = result.get("score", None)
            if score is None:
                continue

            self.score_history[res_idx] = float(score)

            threshold = self._get_threshold()
            if threshold is not None and score > threshold:
                failure_flag = True
                failure_reason = f"STAC score {score:.6f} > threshold {threshold:.6f}"
                self.failure_logs[res_idx] = "STAC"
                break

        if not failure_flag and timestep >= max_episode_length - exec_horizon:
            failure_reason = "maximum episode length reached"

        return failure_flag, failure_reason, self.latest_result_idx

    def wait_for_final_results(self, j: int, timeout_s: float = 2.0) -> Tuple[bool, str, int]:
        exec_horizon = int(self.exec_horizon) if self.exec_horizon is not None else 8
        target_idx = j // exec_horizon - 1

        start = time.time()
        while time.time() - start < float(timeout_s):
            _ = self.detect_failure(timestep=j, max_episode_length=j)
            if self.latest_result_idx >= target_idx:
                return True, "ok", self.latest_result_idx
            time.sleep(0.02)
        return False, "timeout", self.latest_result_idx

    def finalize_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        exec_horizon = int(self.exec_horizon) if self.exec_horizon is not None else 8
        num_chunks = int(episode["action_mode"].shape[0] // exec_horizon)
        success = INTV not in episode["action_mode"]

        if success:
            self.wait_for_final_results(int(episode["action_mode"].shape[0]))

        failure_signal = np.zeros((num_chunks,), dtype=np.bool_)
        if len(self.failure_logs) > 0:
            failure_signal[list(self.failure_logs.keys())] = True
        failure_indices = np.repeat(failure_signal, exec_horizon)

        if self.update_stats and success:
            episode_scores = self._collect_episode_scores()
            if episode_scores.size > 0:
                self.success_scores = np.concatenate([self.success_scores, np.array([episode_scores.max()])])
                self.stac_threshold = float(np.percentile(self.success_scores, self.stac_percentile))

        return {"failure_indices": failure_indices}

    # Rewind support
    def rewind_step(self, j: int, episode_buffers: Dict[str, List], curr_timestep: int) -> bool:
        exec_horizon = int(self.exec_horizon) if self.exec_horizon is not None else 8

        if self._rewind_start_timestep is None:
            self._rewind_start_timestep = int(curr_timestep)

        rewound = self._rewind_start_timestep - int(j)
        if rewound >= self.rewind_chunks * exec_horizon:
            self._reset_after_rewind()
            return False

        if len(episode_buffers.get("action_mode", [])) > 0 and episode_buffers["action_mode"][-1] == INTV:
            self._reset_after_rewind()
            return False

        if j % exec_horizon == 0:
            self._reset_after_rewind()

        return True

    def should_stop_rewinding(self, j: int, episode_buffers: Dict[str, List], *args, **kwargs) -> bool:
        return False
    
    def _reset_after_rewind(self) -> None:
        self._prev_samples = None
        self._prev_boundary_idx = None

    def _collect_episode_scores(self) -> np.ndarray:
        if len(self.score_history) == 0:
            return np.array([], dtype=np.float64)
        return np.asarray(list(self.score_history.values()), dtype=np.float64)

    def _get_threshold(self) -> Optional[float]:
        if self.fixed_threshold is not None:
            return float(self.fixed_threshold)
        if self.stac_threshold is not None:
            return float(self.stac_threshold)
        if self.success_scores.size > 0:
            return float(np.percentile(self.success_scores, self.stac_percentile))
        return None

    @staticmethod
    def _aggregate_error(err: Any, aggr_fn: str) -> float:
        if isinstance(err, (float, int, np.floating)):
            return float(err)
        arr = np.asarray(err).astype(np.float64)
        if arr.size == 0:
            return float("nan")

        aggr_fn = aggr_fn.lower()
        if aggr_fn == "mean":
            return float(np.mean(arr))
        if aggr_fn == "max":
            return float(np.max(arr))
        if aggr_fn == "min":
            return float(np.min(arr))
        if aggr_fn == "std":
            return float(np.std(arr))
        if aggr_fn == "var":
            return float(np.var(arr))
        raise ValueError(f"Unsupported aggr_fn: {aggr_fn}")

    def _sample_action_sequences(self, policy_obs: Dict[str, torch.Tensor], k: int) -> np.ndarray:
        """
        batch infer

        TODO(gaoyuan): check if diffusion policy support this
        """
        try:
            obs_batch = {key: val.repeat(k, *([1] * (val.ndim - 1))) for key, val in policy_obs.items()}
            
            with torch.no_grad():
                action_seq = self.policy.predict_action(obs_batch)
            
            if isinstance(action_seq, dict): action_seq = action_seq['action']
            samples = action_seq.detach().float().cpu().numpy() # Shape (k, T, D)
            return samples
            
        except Exception as e:
            return super()._sample_action_sequences(policy_obs, k)
