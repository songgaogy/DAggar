from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from robosuite.pipeline.src.data.transitions import Transition

from .agent import HILSERLAgent, TrainerConfig


class HILSERLTrainer:
    """Coordinates environment collection and a continuously running learner."""

    def __init__(
        self,
        agent: HILSERLAgent,
        config: TrainerConfig | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or agent.trainer_config
        self.total_env_steps = 0
        self.total_updates = 0
        self.total_critic_updates = 0
        self.total_actor_updates = 0
        self.total_temperature_updates = 0
        self.total_publishes = 0
        self.last_published_update = 0

        self._condition = threading.Condition()
        self._metrics: deque[dict[str, float]] = deque()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._pause_requested = False
        self._busy = False
        self._error: BaseException | None = None
        self._batch_size: int | None = None

    def bootstrap_demo_buffer(self, transitions: list[Transition], demo_source: str = "offline_demo") -> None:
        for transition in transitions:
            if transition.reward is None:
                raise ValueError("Offline demo transitions must include 0/1 success rewards.")
            if float(transition.reward) not in (0.0, 1.0):
                raise ValueError(f"Offline demo reward must be 0 or 1, got {transition.reward}.")
            normalized = Transition(
                obs=transition.obs,
                action=transition.action,
                reward=float(bool(transition.reward)),
                next_obs=transition.next_obs,
                done=bool(transition.done),
                grasp_penalty=transition.grasp_penalty,
                is_intervention=bool(transition.is_intervention),
                info=transition.info,
                reward_source=transition.reward_source or "precomputed_success",
                demo_source=transition.demo_source or demo_source,
                terminated=transition.terminated,
                truncated=bool(transition.truncated),
            )
            self.agent.store_demo_transition(normalized)
        self.notify_data_available()

    def record_transition(
        self,
        *,
        obs,
        action,
        next_obs,
        done: bool | None = None,
        terminated: bool | None = None,
        truncated: bool = False,
        is_success: bool | None = None,
        reward: float | None = None,
        grasp_penalty: float | None = None,
        is_intervention: bool = False,
        info: dict[str, Any] | None = None,
        reward_source: str | None = None,
        demo_source: str | None = None,
    ) -> Transition:
        info = info or {}
        if terminated is None:
            terminated = bool(done) and not bool(truncated)
        success = is_success
        if success is None and "is_success" in info:
            success = bool(info["is_success"])
        terminal = bool(terminated) or bool(success)

        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=terminal,
            grasp_penalty=grasp_penalty,
            is_intervention=is_intervention,
            info=info,
            reward_source=reward_source,
            demo_source=demo_source or ("intervention" if is_intervention else None),
            terminated=terminal,
            truncated=bool(truncated),
        )
        resolved_reward, resolved_source = self.resolve_reward(transition=transition, is_success=success)
        transition.reward = resolved_reward
        transition.reward_source = transition.reward_source or resolved_source
        self.agent.store_transition(transition)
        with self._condition:
            self.total_env_steps += 1
            self._condition.notify_all()
        return transition

    def resolve_reward(
        self,
        transition: Transition,
        *,
        is_success: bool | None = None,
    ) -> tuple[float, str]:
        if transition.reward is not None:
            reward = float(transition.reward)
            if reward not in (0.0, 1.0):
                raise ValueError(f"HIL-SERL requires a binary success reward, got {reward}.")
            return reward, transition.reward_source or "precomputed_success"
        if is_success is None:
            raise ValueError("Missing robosuite is_success for HIL-SERL transition reward.")
        return float(bool(is_success)), "is_success"

    def train_step(self, batch_size: int | None = None) -> dict[str, float]:
        if not self.agent.ready_for_update(batch_size=batch_size):
            raise RuntimeError("Online replay warmup or demonstration replay is not ready.")

        update_started = time.perf_counter()
        metrics: dict[str, float] = {}
        critic_only_updates = max(0, int(self.config.cta_ratio) - 1)
        for _ in range(critic_only_updates):
            metrics = self.agent.update(
                batch=self.agent.sample_mixed_batch(batch_size=batch_size),
                critic_only=True,
            )

        metrics = self.agent.update(
            batch=self.agent.sample_mixed_batch(batch_size=batch_size),
            critic_only=False,
        )
        with self._condition:
            self.total_updates += 1
            self.total_critic_updates += int(self.config.cta_ratio)
            self.total_actor_updates += 1
            if "alpha_loss" in metrics:
                self.total_temperature_updates += 1

        summarized = dict(metrics)
        summarized["learner_critic_only_updates_this_step"] = float(critic_only_updates)
        summarized["learner_full_updates_this_step"] = 1.0
        summarized["learner_update_seconds"] = time.perf_counter() - update_started
        return summarized

    @property
    def learner_finished(self) -> bool:
        with self._condition:
            return self.total_updates >= int(self.config.max_learner_steps)

    def start_async_worker(self, batch_size: int | None = None) -> None:
        self.raise_if_failed()
        with self._condition:
            self._batch_size = None if batch_size is None else int(batch_size)
            self._pause_requested = False
            if self._thread is None:
                self._stop_requested = False
                self._thread = threading.Thread(
                    target=self._async_update_loop,
                    name="hil_serl_learner",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()

    def notify_data_available(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def pending_async_updates(self) -> int:
        with self._condition:
            return int(self._busy)

    def dropped_async_updates(self) -> int:
        return 0

    def drain_async_metrics(self) -> list[dict[str, float]]:
        self.raise_if_failed()
        with self._condition:
            metrics = list(self._metrics)
            self._metrics.clear()
        return metrics

    def flush_async_updates(self, timeout: float | None = None) -> None:
        """Pause after the current learner step, yielding a checkpoint-safe point."""
        self.raise_if_failed()
        with self._condition:
            if self._thread is None:
                return
            self._pause_requested = True
            self._condition.notify_all()
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            while self._busy:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out while pausing the HIL-SERL learner.")
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                self._condition.wait(timeout=0.1 if remaining is None else min(0.1, remaining))
                self.raise_if_failed()

    def close_async_worker(self, wait: bool = True) -> None:
        del wait
        with self._condition:
            thread = self._thread
            if thread is None:
                self.raise_if_failed()
                return
            self._stop_requested = True
            self._pause_requested = False
            self._condition.notify_all()
        thread.join(timeout=30.0)
        if thread.is_alive():
            raise RuntimeError("HIL-SERL learner did not stop after its current update.")
        self.raise_if_failed()

    def _async_update_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    while (
                        not self._stop_requested
                        and (
                            self._pause_requested
                            or self.total_updates >= int(self.config.max_learner_steps)
                        )
                    ):
                        self._condition.wait(timeout=0.1)
                    if self._stop_requested:
                        return

                if not self.agent.ready_for_update(batch_size=self._batch_size):
                    with self._condition:
                        self._condition.wait(timeout=0.1)
                    continue

                with self._condition:
                    if self._stop_requested:
                        return
                    if self._pause_requested:
                        continue
                    self._busy = True

                metrics = self.train_step(batch_size=self._batch_size)
                published = self._maybe_publish_inference_policy()
                with self._condition:
                    self._metrics.append(self._attach_progress_metrics(metrics, published=published))
                    self._busy = False
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._stop_requested = True
                self._busy = False
                self._condition.notify_all()
        finally:
            with self._condition:
                self._thread = None
                self._busy = False
                self._condition.notify_all()

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError(f"HIL-SERL learner failed: {error}") from error

    def state_dict(self) -> dict[str, int]:
        with self._condition:
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
        with self._condition:
            if self._busy:
                raise RuntimeError("Pause the learner before restoring trainer state.")
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
        state = self.state_dict()
        return {
            "env_steps": state["total_env_steps"],
            "total_updates": state["total_updates"],
            "critic_updates": state["total_critic_updates"],
            "actor_updates": state["total_actor_updates"],
            "temperature_updates": state["total_temperature_updates"],
            "publish_count": state["total_publishes"],
            "last_published_update": state["last_published_update"],
            "updates_until_publish": self.updates_until_next_publish(),
        }

    def updates_until_next_publish(self) -> int:
        interval = max(1, int(self.config.steps_per_update))
        remainder = int(self.total_updates) % interval
        return interval if remainder == 0 else interval - remainder

    def _maybe_publish_inference_policy(self) -> bool:
        interval = max(1, int(self.config.steps_per_update))
        if self.total_updates == 0 or self.total_updates % interval != 0:
            return False
        self.agent.sync_inference_policy()
        with self._condition:
            self.total_publishes += 1
            self.last_published_update = int(self.total_updates)
        return True

    def _attach_progress_metrics(self, metrics: dict[str, float], *, published: bool) -> dict[str, float]:
        progress = self.progress_snapshot()
        summarized = dict(metrics)
        summarized.update(
            {
                "learner_total_updates": float(progress["total_updates"]),
                "learner_critic_updates": float(progress["critic_updates"]),
                "learner_actor_updates": float(progress["actor_updates"]),
                "learner_temperature_updates": float(progress["temperature_updates"]),
                "learner_publish_count": float(progress["publish_count"]),
                "learner_last_published_update": float(progress["last_published_update"]),
                "learner_updates_until_publish": float(progress["updates_until_publish"]),
                "learner_published": 1.0 if published else 0.0,
            }
        )
        return summarized
