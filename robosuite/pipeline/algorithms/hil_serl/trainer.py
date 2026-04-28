from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from robosuite.pipeline.common import RewardProvider, TrainerConfig, Transition

from .agent import HILSERLAgent


class HILSERLTrainer:
    def __init__(
        self,
        agent: HILSERLAgent,
        config: TrainerConfig | None = None,
        reward_provider: RewardProvider | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or agent.trainer_config
        self.reward_provider = reward_provider
        self.total_env_steps = 0
        self.total_updates = 0
        self.total_critic_updates = 0
        self.total_actor_updates = 0
        self.total_temperature_updates = 0
        self.total_publishes = 0
        self.last_published_update = 0
        self._async_condition = threading.Condition()
        self._async_metrics: deque[dict[str, float]] = deque()
        self._async_thread: threading.Thread | None = None
        self._async_stop_requested = False
        self._async_pending_updates = 0
        self._async_pending_batch_size: int | None = None
        self._async_busy = False
        self._async_error: BaseException | None = None
        self._async_dropped_updates = 0

    def bootstrap_demo_buffer(self, transitions: list[Transition], demo_source: str = "offline_demo") -> None:
        for transition in transitions:
            if transition.reward is None:
                raise ValueError("Offline demo transitions must include reward values before bootstrapping.")
            normalized = Transition(
                obs=transition.obs,
                action=transition.action,
                reward=transition.reward,
                next_obs=transition.next_obs,
                done=transition.done,
                grasp_penalty=transition.grasp_penalty,
                is_intervention=bool(transition.is_intervention),
                info=transition.info,
                reward_source=transition.reward_source or "precomputed",
                demo_source=transition.demo_source or demo_source,
            )
            self.agent.store_demo_transition(normalized)

    def record_transition(
        self,
        *,
        obs,
        action,
        next_obs,
        done: bool,
        env_reward: float | None = None,
        reward: float | None = None,
        grasp_penalty: float | None = None,
        is_intervention: bool = False,
        info: dict[str, Any] | None = None,
        reward_source: str | None = None,
        demo_source: str | None = None,
    ) -> Transition:
        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            grasp_penalty=grasp_penalty,
            is_intervention=is_intervention,
            info=info,
            reward_source=reward_source,
            demo_source=demo_source or ("intervention" if is_intervention else None),
        )
        resolved_reward, resolved_source = self.resolve_reward(transition=transition, env_reward=env_reward)
        transition.reward = resolved_reward
        transition.reward_source = transition.reward_source or resolved_source
        self.agent.store_transition(transition)
        self.total_env_steps += 1
        return transition

    def resolve_reward(self, transition: Transition, env_reward: float | None = None) -> tuple[float, str]:
        if transition.reward is not None:
            return float(transition.reward), transition.reward_source or "precomputed"
        if self.reward_provider is not None:
            resolved = self.reward_provider(transition, env_reward=env_reward)
            if isinstance(resolved, tuple):
                reward, source = resolved
                return float(reward), str(source)
            return float(resolved), "reward_provider"
        if env_reward is None:
            raise ValueError("No reward available. Provide env_reward, transition.reward, or reward_provider.")
        return float(env_reward), "env"

    def train_step(self, batch_size: int | None = None) -> dict[str, float]:
        if not self.agent.ready_for_update(batch_size=batch_size):
            raise RuntimeError("Not enough data in the online/demo buffers for a mixed HIL-SERL update.")

        metrics = {}
        critic_only_updates_this_step = 0
        for _ in range(max(0, int(self.config.cta_ratio) - 1)):
            critic_batch = self.agent.sample_mixed_batch(batch_size=batch_size)
            metrics = self.agent.update(batch=critic_batch, critic_only=True)
            self.total_updates += 1
            self.total_critic_updates += 1
            critic_only_updates_this_step += 1

        full_batch = self.agent.sample_mixed_batch(batch_size=batch_size)
        metrics = self.agent.update(batch=full_batch, critic_only=False)
        self.total_updates += 1
        self.total_critic_updates += 1
        self.total_actor_updates += 1
        if "alpha_loss" in metrics:
            self.total_temperature_updates += 1

        summarized_metrics = dict(metrics)
        summarized_metrics["learner_critic_only_updates_this_step"] = float(critic_only_updates_this_step)
        summarized_metrics["learner_full_updates_this_step"] = 1.0
        return summarized_metrics

    def maybe_update(self, *, env_step: int | None = None, batch_size: int | None = None) -> list[dict[str, float]]:
        self._raise_async_error()
        current_step = self.total_env_steps if env_step is None else int(env_step)
        if current_step < int(self.config.warmup_steps):
            return self.drain_async_metrics()
        if not self.agent.ready_for_update(batch_size=batch_size):
            return self.drain_async_metrics()
        metrics_list = []
        for _ in range(int(self.config.updates_per_step)):
            metrics = self.train_step(batch_size=batch_size)
            published = self._maybe_publish_inference_policy()
            metrics_list.append(self._attach_progress_metrics(metrics, published=published))
        return metrics_list

    def maybe_update_async(self, *, env_step: int | None = None, batch_size: int | None = None) -> list[dict[str, float]]:
        self._raise_async_error()
        current_step = self.total_env_steps if env_step is None else int(env_step)
        if current_step >= int(self.config.warmup_steps) and self.agent.ready_for_update(batch_size=batch_size):
            self.start_async_worker()
            with self._async_condition:
                requested = int(self.config.updates_per_step)
                # Cap the queue so a slow learner cannot accumulate an unbounded
                # backlog. Excess requests are dropped (the env already moved on,
                # so stale updates have low value).
                max_pending = max(1, int(self.config.max_pending_updates))
                in_flight = int(self._async_pending_updates) + (1 if self._async_busy else 0)
                room = max(0, max_pending - in_flight)
                accepted = min(requested, room)
                dropped = requested - accepted
                if accepted > 0:
                    self._async_pending_updates += accepted
                    self._async_pending_batch_size = None if batch_size is None else int(batch_size)
                    self._async_condition.notify_all()
                if dropped > 0:
                    self._async_dropped_updates += dropped
        return self.drain_async_metrics()

    def start_async_worker(self) -> None:
        self._raise_async_error()
        with self._async_condition:
            if self._async_thread is not None:
                return
            self._async_stop_requested = False
            self._async_thread = threading.Thread(target=self._async_update_loop, name="hil_serl_learner", daemon=True)
            self._async_thread.start()

    def pending_async_updates(self) -> int:
        with self._async_condition:
            return int(self._async_pending_updates) + int(self._async_busy)

    def dropped_async_updates(self) -> int:
        with self._async_condition:
            return int(self._async_dropped_updates)

    def drain_async_metrics(self) -> list[dict[str, float]]:
        self._raise_async_error()
        with self._async_condition:
            metrics = list(self._async_metrics)
            self._async_metrics.clear()
        return metrics

    def flush_async_updates(self, timeout: float | None = None) -> None:
        self._raise_async_error()
        with self._async_condition:
            if self._async_thread is None:
                return
            if timeout is None:
                while self._async_pending_updates > 0 or self._async_busy:
                    self._async_condition.wait(timeout=0.1)
                    self._raise_async_error()
                return

        with self._async_condition:
            end_time = time.monotonic() + float(timeout)
            remaining_timeout = timeout
            while self._async_pending_updates > 0 or self._async_busy:
                self._raise_async_error()
                if remaining_timeout is not None and remaining_timeout <= 0.0:
                    raise TimeoutError("Timed out while waiting for async learner updates to finish.")
                wait_timeout = 0.1 if remaining_timeout is None else min(0.1, remaining_timeout)
                self._async_condition.wait(timeout=wait_timeout)
                if remaining_timeout is not None:
                    remaining_timeout = end_time - time.monotonic()

    def close_async_worker(self, wait: bool = True) -> None:
        if wait:
            self.flush_async_updates()
        with self._async_condition:
            thread = self._async_thread
            if thread is None:
                return
            self._async_stop_requested = True
            self._async_condition.notify_all()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("Async learner worker did not stop cleanly.")
        with self._async_condition:
            if self._async_thread is thread:
                self._async_thread = None
        self._raise_async_error()

    def _async_update_loop(self) -> None:
        while True:
            with self._async_condition:
                while self._async_pending_updates == 0 and not self._async_stop_requested:
                    self._async_condition.wait(timeout=0.1)
                if self._async_stop_requested and self._async_pending_updates == 0:
                    self._async_thread = None
                    self._async_condition.notify_all()
                    return
                self._async_pending_updates -= 1
                batch_size = self._async_pending_batch_size
                self._async_busy = True

            try:
                metrics = self.train_step(batch_size=batch_size)
                published = self._maybe_publish_inference_policy()
                with self._async_condition:
                    self._async_metrics.append(self._attach_progress_metrics(metrics, published=published))
            except BaseException as exc:
                with self._async_condition:
                    self._async_error = exc
                    self._async_stop_requested = True
                    self._async_condition.notify_all()
                return
            finally:
                with self._async_condition:
                    self._async_busy = False
                    self._async_condition.notify_all()

    def _raise_async_error(self) -> None:
        with self._async_condition:
            error = self._async_error
        if error is not None:
            raise RuntimeError(f"Async learner worker failed: {error}") from error

    def state_dict(self) -> dict[str, int]:
        return {
            "total_env_steps": int(self.total_env_steps),
            "total_updates": int(self.total_updates),
            "total_critic_updates": int(self.total_critic_updates),
            "total_actor_updates": int(self.total_actor_updates),
            "total_temperature_updates": int(self.total_temperature_updates),
            "total_publishes": int(self.total_publishes),
            "last_published_update": int(self.last_published_update),
        }

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return
        self.total_env_steps = int(state_dict.get("total_env_steps", self.total_env_steps))
        self.total_updates = int(state_dict.get("total_updates", self.total_updates))
        self.total_critic_updates = int(state_dict.get("total_critic_updates", self.total_updates))
        self.total_actor_updates = int(state_dict.get("total_actor_updates", self.total_actor_updates))
        self.total_temperature_updates = int(
            state_dict.get("total_temperature_updates", self.total_temperature_updates)
        )
        self.total_publishes = int(state_dict.get("total_publishes", self.total_publishes))
        self.last_published_update = int(state_dict.get("last_published_update", self.last_published_update))

    def progress_snapshot(self) -> dict[str, int]:
        return {
            "env_steps": int(self.total_env_steps),
            "total_updates": int(self.total_updates),
            "critic_updates": int(self.total_critic_updates),
            "actor_updates": int(self.total_actor_updates),
            "temperature_updates": int(self.total_temperature_updates),
            "publish_count": int(self.total_publishes),
            "last_published_update": int(self.last_published_update),
            "updates_until_publish": int(self.updates_until_next_publish()),
        }

    def updates_until_next_publish(self) -> int:
        publish_interval = max(1, int(self.config.steps_per_update))
        remainder = int(self.total_updates) % publish_interval
        if remainder == 0:
            return publish_interval
        return publish_interval - remainder

    def _maybe_publish_inference_policy(self) -> bool:
        publish_interval = max(1, int(self.config.steps_per_update))
        if self.total_updates % publish_interval != 0:
            return False
        self.agent.sync_inference_policy()
        self.total_publishes += 1
        self.last_published_update = int(self.total_updates)
        return True

    def _attach_progress_metrics(self, metrics: dict[str, float], *, published: bool) -> dict[str, float]:
        progress = self.progress_snapshot()
        summarized_metrics = dict(metrics)
        summarized_metrics["learner_total_updates"] = float(progress["total_updates"])
        summarized_metrics["learner_critic_updates"] = float(progress["critic_updates"])
        summarized_metrics["learner_actor_updates"] = float(progress["actor_updates"])
        summarized_metrics["learner_temperature_updates"] = float(progress["temperature_updates"])
        summarized_metrics["learner_publish_count"] = float(progress["publish_count"])
        summarized_metrics["learner_last_published_update"] = float(progress["last_published_update"])
        summarized_metrics["learner_updates_until_publish"] = float(progress["updates_until_publish"])
        summarized_metrics["learner_published"] = 1.0 if published else 0.0
        return summarized_metrics

    def fit(
        self,
        env,
        num_steps: int,
        *,
        intervention_callback: Callable[[Any, np.ndarray, int], np.ndarray | None] | None = None,
        deterministic: bool = False,
        reset_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, float]]:
        metrics_history: list[dict[str, float]] = []
        reset_kwargs = reset_kwargs or {}
        reset_output = env.reset(**reset_kwargs)
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output

        for step in range(int(num_steps)):
            if step < int(self.config.random_steps) and hasattr(env, "action_space"):
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                action = self.agent.select_action(obs, deterministic=deterministic)

            executed_action = action
            is_intervention = False
            if intervention_callback is not None:
                intervention_action = intervention_callback(obs, action, step)
                if intervention_action is not None:
                    executed_action = np.asarray(intervention_action, dtype=np.float32)
                    is_intervention = True

            step_output = env.step(executed_action)
            if len(step_output) == 5:
                next_obs, env_reward, terminated, truncated, info = step_output
                done = bool(terminated or truncated)
            else:
                next_obs, env_reward, done, info = step_output
                truncated = False

            if isinstance(info, dict) and "intervene_action" in info:
                executed_action = np.asarray(info["intervene_action"], dtype=np.float32)
                is_intervention = True

            self.record_transition(
                obs=obs,
                action=executed_action,
                next_obs=next_obs,
                done=done,
                env_reward=env_reward,
                grasp_penalty=info.get("grasp_penalty") if isinstance(info, dict) else None,
                is_intervention=is_intervention,
                info=info if isinstance(info, dict) else {"raw_info": info},
            )
            metrics_history.extend(self.maybe_update())

            if done:
                reset_output = env.reset(**reset_kwargs)
                obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            else:
                obs = next_obs

        return metrics_history
