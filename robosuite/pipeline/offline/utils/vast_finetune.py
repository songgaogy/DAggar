"""Phase-A VAST value-stitching finetuning for offline DIPOLE.

Load the pretrained VAST modules (``init_vast.sh`` output),
**continue training** them (unfrozen) on the collected policy-rollout sections
mixed with the exact transitions the warmup consumed, then save the finetuned
checkpoint. The modules are frozen afterwards (in the training entry) and used
to supply stitched advantage with TD1 tail fallback for weighted-BC.

Building blocks reused verbatim:
- :class:`FlowDaggerReplayBuffer` (+ ``.load``) for the mixed transition store,
- :class:`VASTReplayBuffer` + :meth:`preencode_step_cache` for a one-shot frozen-
  encoder feature cache (the finetune loop never re-runs the encoder),
- :meth:`VASTLearner.update` for the joint G/V value-stitching step.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    ReplayBufferConfig,
)
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.vast.checkpoint import (
    VAST_ALGORITHM,
    normalize_vast_payload,
)
from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.replay import VASTReplayBuffer
from robosuite.pipeline.common.types import Transition

if TYPE_CHECKING:
    import torch

    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
    from robosuite.pipeline.algorithms.vast.vast import VASTLearner

logger = logging.getLogger(__name__)


def _max_episode_index(transitions: list[Transition]) -> int:
    hi = -1
    for t in transitions:
        info = t.info or {}
        hi = max(hi, int(info.get("episode_index", -1)))
    return hi


def build_vast_finetune_buffer(
    policy_bc_transitions: list[Transition],
    *,
    camera_names: list[str],
    image_size: int,
    action_horizon: int,
    warmup_transitions_path: str | Path | None,
    capacity: int = 10_000_000,
) -> tuple[FlowDaggerReplayBuffer, dict[str, int]]:
    """Mix collected policy sections with the warmup transitions into one buffer.

    ``warmup_transitions_path`` points at the ``vast_offline_transitions.pt`` the
    VAST warmup exported (``FlowDaggerReplayBuffer``-serialized). Its transitions
    are reloaded verbatim and their ``episode_index`` re-stamped into a disjoint
    range so chunk windows stay bounded to temporally-adjacent frames.
    """
    buffer = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=int(capacity), batch_size=1),
        name="vast_finetune_buffer",
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
            legacy_path = path.with_name("iql_offline_transitions.pt")
            if path.name == "vast_offline_transitions.pt" and legacy_path.exists():
                warnings.warn(
                    f"Loading deprecated warmup buffer {legacy_path}; re-save as "
                    "vast_offline_transitions.pt.",
                    FutureWarning,
                    stacklevel=2,
                )
                path = legacy_path
            else:
                raise FileNotFoundError(
                    f"VAST warmup transitions not found: {path}. Run init_vast.sh with "
                    "warmup.num_trajectories.save_data=true first, or set "
                    "offline.vast_warmup_transitions_dir correctly."
                )
        loader = FlowDaggerReplayBuffer(
            config=ReplayBufferConfig(capacity=int(capacity), batch_size=1),
            name="vast_warmup_loader",
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
        "[offline][vast] finetune buffer: policy_bc=%d + warmup=%d transitions -> %d valid windows",
        stats["policy_bc_transitions"],
        stats["warmup_transitions"],
        stats["valid_windows"],
    )
    return buffer, stats


def finetune_vast(
    vast_learner: "VASTLearner",
    base_buffer: FlowDaggerReplayBuffer,
    vast_cfg: VASTConfig,
    *,
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator | None",
    num_steps: int,
    batch_size: int,
    preencode: bool = True,
    device: str,
    encode_batch_size: int = 64,
    metric_logger: Any = None,
    log_interval: int = 200,
    log_prefix: str = "vast_value_stitching_finetune",
) -> dict[str, float]:
    """Jointly continue training the unfrozen VAST G/V modules.

    Returns the last logged metrics dict. When ``preencode`` is set, every valid
    chunk window is encoded once into an in-memory cache (frozen encoder) and the
    loop samples from the cache — no per-step encoder forward.
    """
    from robosuite.pipeline.utils import maybe_log

    replay = VASTReplayBuffer(base_buffer, vast_cfg)
    if not replay.ready(batch_size):
        raise RuntimeError(
            f"[offline][vast] finetune buffer has < batch_size={batch_size} valid windows."
        )

    if preencode:
        train_replay: Any = replay.preencode_step_cache(
            encoder=encoder,
            discriminator=discriminator,
            device=device,
            encode_batch_size=int(encode_batch_size),
            cache_device="cpu",
            progress_desc="[offline][vast] preencode",
        )
        logger.info("[offline][vast] pre-encoded %d windows into cache", len(train_replay))
    else:
        train_replay = replay

    last_metrics: dict[str, float] = {}
    total = int(num_steps)
    for step in range(total):
        batch = train_replay.sample_step_batch(
            batch_size, encoder=encoder, discriminator=discriminator, device=device
        )
        metrics = vast_learner.update(batch)
        last_metrics = metrics
        is_log = (step % max(1, int(log_interval)) == 0) or (step == total - 1)
        if is_log:
            maybe_log(
                metric_logger,
                {f"{log_prefix}/{k}": float(v) for k, v in metrics.items()},
                step=step,
            )
            print(
                f"[offline][vast][joint {step:6d}/{total}] "
                f"v_loss={metrics.get('v_loss', 0.0):.4f} "
                f"g_loss={metrics.get('g_loss', 0.0):.4f} "
                f"mc={metrics.get('g_mc_loss', 0.0):.4f} "
                f"comp={metrics.get('g_comp_loss', 0.0):.4f} "
                f"v_mean={metrics.get('v_mean', 0.0):+.3f} "
                f"target_mean={metrics.get('target_mean', 0.0):+.3f}"
            )
    return last_metrics


def save_finetuned_vast(
    vast_learner: "VASTLearner",
    path: str | Path,
    *,
    encoder: "SharedDynamicsEncoder",
    discriminator: "FrozenNNPUDiscriminator | None",
    vast_cfg: VASTConfig,
    nnpu_ckpt: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist the finetuned VAST G/V learner as a schema-v7 checkpoint."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder_meta: dict[str, Any] = {
        "nnpu_checkpoint": nnpu_ckpt,
        "state_feature_dim": int(encoder.state_feature_dim),
        "chunk_feature_dim": int(encoder.chunk_feature_dim),
        "policy_action_dim": int(vast_learner.action_dim),
        "n_tokens": int(vast_learner.n_tokens),
        "proprio_dim": int(vast_learner.proprio_dim),
        "state_proj_dim": int(vast_cfg.state_proj_dim),
        "proprio_proj_dim": int(vast_cfg.proprio_proj_dim),
        "proj_activation": str(vast_cfg.proj_activation),
        "algorithm": "vast_value_stitching_adaptation",
        "vast_v_mode": str(vast_cfg.vast_v_mode),
        "vast_max_k": int(vast_cfg.vast_max_k),
        "vast_comp_coef": float(vast_cfg.vast_comp_coef),
        "vast_sampling_seed": int(vast_cfg.vast_sampling_seed),
        "action_horizon": int(vast_cfg.action_horizon),
        "v_ensemble_size": int(vast_learner.ensemble_size),
        "ensemble_lcb_beta": float(vast_cfg.ensemble_lcb_beta),
        "ensemble_bootstrap_prob": float(vast_cfg.ensemble_bootstrap_prob),
        "expectile_tau": float(vast_cfg.expectile_tau),
        "disc_reward_coef": float(vast_cfg.disc_reward_coef),
        "output_reward_coef": float(vast_cfg.output_reward_coef),
        "threshold": (
            float(discriminator.threshold) if discriminator is not None else float("nan")
        ),
        "finetuned_offline": True,
    }
    if extra_meta:
        encoder_meta.update(extra_meta)
    payload = {
        "vast_state": vast_learner.state_dict(),
        "cfg": asdict(vast_cfg),
        "encoder_meta": encoder_meta,
        "schema_version": 7,
        "algorithm": "vast_value_stitching_adaptation",
    }
    torch.save(payload, path)
    logger.info("[offline][vast] wrote finetuned VAST state to %s", path)
    return path


def validate_vast_checkpoint_payload(
    payload: Any,
    vast_cfg: VASTConfig,
    *,
    require_finetuned: bool,
) -> None:
    """Reject legacy or semantically incompatible VAST checkpoints."""
    if not isinstance(payload, dict):
        raise ValueError("VAST checkpoint payload must be a dictionary.")
    normalized = normalize_vast_payload(payload)
    payload.clear()
    payload.update(normalized)
    schema = int(payload["schema_version"])
    ckpt_cfg = payload.get("cfg", {})
    meta = payload.get("encoder_meta", {})
    if schema == 7:
        algorithm = payload.get("algorithm", meta.get("algorithm"))
        if algorithm != VAST_ALGORITHM:
            raise ValueError(
                f"Checkpoint algorithm={algorithm!r} is not "
                f"{VAST_ALGORITHM!r}. Re-run init_vast.sh."
            )
    else:
        method = ckpt_cfg.get("method", meta.get("method"))
        if method != "vast_value_stitching":
            raise ValueError(
                f"Checkpoint method={method!r} is not 'vast_value_stitching'. "
                "Re-run init_vast.sh with the VAST warmup configuration."
            )
    if require_finetuned and not bool(meta.get("finetuned_offline", False)):
        raise ValueError(
            "offline.skip_rl=true requires a schema-v7 checkpoint (or deprecated "
            "schema-v6 checkpoint) with "
            "encoder_meta.finetuned_offline=true. Run Phase A first."
        )

    fields = (
        "vast_v_mode",
        "vast_max_k",
        "action_horizon",
        "vast_sampling_seed",
        "discount",
        "expectile_tau",
        "vast_comp_coef",
        "output_reward_coef",
        "disc_reward_coef",
    )
    for field in fields:
        if field not in ckpt_cfg:
            raise ValueError(f"VAST checkpoint cfg is missing required field {field!r}.")
        runtime = getattr(vast_cfg, field)
        checkpoint = ckpt_cfg[field]
        if isinstance(runtime, float):
            matches = abs(float(runtime) - float(checkpoint)) <= 1e-12
        else:
            matches = runtime == type(runtime)(checkpoint)
        if not matches:
            raise ValueError(
                f"VAST checkpoint config mismatch for {field}: "
                f"ckpt={checkpoint!r}, runtime={runtime!r}."
            )


__all__ = [
    "build_vast_finetune_buffer",
    "finetune_vast",
    "save_finetuned_vast",
    "validate_vast_checkpoint_payload",
]
