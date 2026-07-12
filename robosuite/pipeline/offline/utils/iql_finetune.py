"""Phase-A IQL finetuning for offline DIPOLE.

PROMPT.md §Q/V update: load the pretrained IQL critics (``init_iql_qv.sh`` output),
**continue training** them (unfrozen) on the collected policy-rollout sections
mixed with the exact transitions the warmup consumed, then save the finetuned
checkpoint. The critics are frozen afterwards (in the training entry) and used to
supply the TD advantage for the weighted-BC policy update.

Building blocks reused verbatim:
- :class:`FlowDaggerReplayBuffer` (+ ``.load``) for the mixed transition store,
- :class:`IQLReplayBuffer` + :meth:`preencode_step_cache` for a one-shot frozen-
  encoder feature cache (the finetune loop never re-runs the encoder),
- :meth:`IQLLearner.update` for the V-only MSE-TD optimization step.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    ReplayBufferConfig,
)
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer
from robosuite.pipeline.common.types import Transition

if TYPE_CHECKING:
    import torch

    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
    from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner

logger = logging.getLogger(__name__)


def _max_episode_index(transitions: list[Transition]) -> int:
    hi = -1
    for t in transitions:
        info = t.info or {}
        hi = max(hi, int(info.get("episode_index", -1)))
    return hi


def build_iql_finetune_buffer(
    policy_bc_transitions: list[Transition],
    *,
    camera_names: list[str],
    image_size: int,
    action_horizon: int,
    warmup_transitions_path: str | Path | None,
    capacity: int = 10_000_000,
) -> tuple[FlowDaggerReplayBuffer, dict[str, int]]:
    """Mix collected policy sections with the warmup transitions into one buffer.

    ``warmup_transitions_path`` points at the ``iql_offline_transitions.pt`` the
    IQL warmup exported (``FlowDaggerReplayBuffer``-serialized). Its transitions
    are reloaded verbatim and their ``episode_index`` re-stamped into a disjoint
    range so chunk windows stay bounded to temporally-adjacent frames.
    """
    buffer = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=int(capacity), batch_size=1),
        name="iql_finetune_buffer",
        camera_names=list(camera_names),
        action_horizon=int(action_horizon),
        image_size=int(image_size),
        augmentation_config=FlowAugmentationConfig(),
    )
    for t in policy_bc_transitions:
        buffer.add(t)

    stats = {
        "policy_bc_transitions": len(policy_bc_transitions),
        "warmup_transitions": 0,
    }

    if warmup_transitions_path is not None:
        path = Path(warmup_transitions_path)
        if not path.exists():
            raise FileNotFoundError(
                f"IQL warmup transitions not found: {path}. Run init_iql_qv.sh with "
                "warmup.num_trajectories.save_data=true first, or set "
                "offline.iql_warmup_transitions_dir correctly."
            )
        loader = FlowDaggerReplayBuffer(
            config=ReplayBufferConfig(capacity=int(capacity), batch_size=1),
            name="iql_warmup_loader",
            camera_names=list(camera_names),
            action_horizon=int(action_horizon),
            image_size=int(image_size),
            augmentation_config=FlowAugmentationConfig(),
        )
        loader.load(path)
        base = _max_episode_index(policy_bc_transitions) + 1
        warmup_transitions = list(loader._storage)  # noqa: SLF001 - read-only access
        for src in warmup_transitions:
            info = dict(src.info or {})
            info["episode_index"] = base + int(info.get("episode_index", 0))
            info.setdefault("buffer_role", "offline")
            buffer.add(
                Transition(
                    obs=src.obs,
                    action=src.action,
                    reward=src.reward,
                    next_obs=src.next_obs,
                    done=bool(src.done),
                    grasp_penalty=src.grasp_penalty,
                    is_intervention=bool(src.is_intervention),
                    info=info,
                    reward_source=src.reward_source,
                    demo_source=src.demo_source,
                )
            )
        stats["warmup_transitions"] = len(warmup_transitions)

    stats["valid_windows"] = int(buffer.num_valid_sequences())
    logger.info(
        "[offline][iql] finetune buffer: policy_bc=%d + warmup=%d transitions -> %d valid windows",
        stats["policy_bc_transitions"],
        stats["warmup_transitions"],
        stats["valid_windows"],
    )
    return buffer, stats


def finetune_iql(
    iql_learner: "IQLLearner",
    base_buffer: FlowDaggerReplayBuffer,
    iql_cfg: IQLConfig,
    *,
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator | None",
    num_steps: int,
    batch_size: int,
    value_only_steps: int = 0,
    preencode: bool = True,
    device: str,
    encode_batch_size: int = 64,
    metric_logger: Any = None,
    log_interval: int = 200,
    log_prefix: str = "iql_finetune",
) -> dict[str, float]:
    """Continue training the (unfrozen) IQL critics on the mixed buffer.

    Returns the last logged metrics dict. When ``preencode`` is set, every valid
    chunk window is encoded once into an in-memory cache (frozen encoder) and the
    loop samples from the cache — no per-step encoder forward.
    """
    from robosuite.pipeline.utils import maybe_log

    replay = IQLReplayBuffer(base_buffer, iql_cfg)
    if not replay.ready(batch_size):
        raise RuntimeError(
            f"[offline][iql] finetune buffer has < batch_size={batch_size} valid windows."
        )

    if preencode:
        train_replay: Any = replay.preencode_step_cache(
            encoder=encoder,
            discriminator=discriminator,
            device=device,
            encode_batch_size=int(encode_batch_size),
            cache_device="cpu",
            progress_desc="[offline][iql] preencode",
        )
        logger.info("[offline][iql] pre-encoded %d windows into cache", len(train_replay))
    else:
        train_replay = replay

    last_metrics: dict[str, float] = {}
    total = int(value_only_steps) + int(num_steps)
    for step in range(total):
        batch = train_replay.sample_step_batch(
            batch_size, encoder=encoder, discriminator=discriminator, device=device
        )
        # V-only MSE-TD (no Q phase). value_only_steps + num_steps are summed
        # into one loop; the phase label is kept only for logging continuity.
        metrics = iql_learner.update(batch)
        phase = "value_only" if step < int(value_only_steps) else "full"
        last_metrics = metrics
        is_log = (step % max(1, int(log_interval)) == 0) or (step == total - 1)
        if is_log:
            maybe_log(
                metric_logger,
                {f"{log_prefix}/{k}": float(v) for k, v in metrics.items()},
                step=step,
            )
            print(
                f"[offline][iql][{phase} {step:6d}/{total}] "
                f"v_loss={metrics.get('v_loss', 0.0):.4f} "
                f"v_mean={metrics.get('v_mean', 0.0):+.3f} "
                f"target_mean={metrics.get('target_mean', 0.0):+.3f} "
                f"td={metrics.get('td_error_abs_mean', 0.0):.4f}"
            )
    return last_metrics


def save_finetuned_iql(
    iql_learner: "IQLLearner",
    path: str | Path,
    *,
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator | None",
    iql_cfg: IQLConfig,
    nnpu_ckpt: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist the finetuned IQL in the same schema-v3 payload as the warmup.

    Reloadable by ``train_dipole_rl._load_iql_warmup_state``.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder_meta: dict[str, Any] = {
        "nnpu_checkpoint": nnpu_ckpt,
        "state_feature_dim": int(encoder.state_feature_dim),
        "chunk_feature_dim": int(encoder.chunk_feature_dim),
        "policy_action_dim": int(iql_learner.action_dim),
        "n_tokens": int(iql_learner.n_tokens),
        "proprio_dim": int(iql_learner.proprio_dim),
        "state_proj_dim": int(iql_cfg.state_proj_dim),
        "proprio_proj_dim": int(iql_cfg.proprio_proj_dim),
        "proj_activation": str(iql_cfg.proj_activation),
        "disc_reward_coef": float(iql_cfg.disc_reward_coef),
        "output_reward_coef": float(iql_cfg.output_reward_coef),
        "threshold": (
            float(discriminator.threshold) if discriminator is not None else float("nan")
        ),
        "finetuned_offline": True,
    }
    if extra_meta:
        encoder_meta.update(extra_meta)
    payload = {
        "iql_state": iql_learner.state_dict(),
        "cfg": asdict(iql_cfg),
        "encoder_meta": encoder_meta,
        "schema_version": 4,
    }
    torch.save(payload, path)
    logger.info("[offline][iql] wrote finetuned IQL state to %s", path)
    return path


__all__ = ["build_iql_finetune_buffer", "finetune_iql", "save_finetuned_iql"]
