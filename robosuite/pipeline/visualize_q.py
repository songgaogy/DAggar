"""Visualize AWR Q/V estimates on one successful PickPlaceCereal rollout."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import hydra
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robosuite.pipeline.bootstrap import (
    build_agent_config,
    resolve_qv_cache_path,
)
from robosuite.pipeline.src.awr import AWRAgent, load_qv_cache
from robosuite.pipeline.src.data import list_demo_names, load_hdf5_demos, resolve_demo_paths
from robosuite.pipeline.src.environment import (
    bind_proprio_extractor,
    build_robosuite_env,
    build_runtime_config,
    flow_checkpoint_settings,
    load_flow_checkpoint,
    load_task_metadata,
)
from robosuite.pipeline.utils import require_cuda, resolve_path, set_seed


def _image_tensor(agent: AWRAgent, observations: list[dict]) -> torch.Tensor:
    images = np.stack(
        [
            np.stack(
                [
                    np.transpose(
                        np.asarray(observation[name], dtype=np.float32) / 255.0,
                        (2, 0, 1),
                    )
                    for name in agent.camera_names
                ]
            )
            for observation in observations
        ]
    )
    tensor = torch.from_numpy(images).to(agent.awr_config.device)
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=tensor.device,
    ).view(1, 1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=tensor.device,
    ).view(1, 1, 3, 1, 1)
    return (tensor - mean) / std


def _proprio_tensor(agent: AWRAgent, observations: list[dict]) -> torch.Tensor:
    values = np.stack(
        [np.asarray(observation["state"], dtype=np.float32) for observation in observations]
    )
    if agent.core.prop_mean is not None:
        values = (values - agent.core.prop_mean) / agent.core.prop_std
    return torch.from_numpy(values.astype(np.float32)).to(agent.awr_config.device)


def _compute_metrics(agent: AWRAgent, transitions: list, max_steps: int | None) -> list[dict]:
    horizon = int(agent.awr_config.action_horizon)
    valid_starts = []
    for start in range(len(transitions) - horizon + 1):
        sequence = transitions[start : start + horizon]
        if any(item.done for item in sequence[:-1]):
            continue
        valid_starts.append(start)
        if max_steps is not None and len(valid_starts) >= int(max_steps):
            break
    if not valid_starts:
        raise RuntimeError(f"Rollout has no complete action_horizon={horizon} windows.")

    metrics: list[dict] = []
    discount = float(agent.awr_config.discount)
    powers = np.power(discount, np.arange(horizon, dtype=np.float32))
    for offset in range(0, len(valid_starts), int(agent.trainer_config.value_batch_size)):
        starts = valid_starts[offset : offset + int(agent.trainer_config.value_batch_size)]
        sequences = [transitions[start : start + horizon] for start in starts]
        observations = [sequence[0].obs for sequence in sequences]
        next_observations = [sequence[-1].next_obs for sequence in sequences]
        actions = torch.from_numpy(
            np.stack(
                [
                    np.stack(
                        [np.asarray(item.action, dtype=np.float32) for item in sequence]
                    )
                    for sequence in sequences
                ]
            )
        ).to(agent.awr_config.device)
        rewards = torch.tensor(
            [
                float(
                    np.sum(
                        np.asarray([item.reward for item in sequence], dtype=np.float32)
                        * powers
                    )
                )
                for sequence in sequences
            ],
            device=agent.awr_config.device,
        ).view(-1, 1)
        dones = torch.tensor(
            [float(any(item.done for item in sequence)) for sequence in sequences],
            device=agent.awr_config.device,
        ).view(-1, 1)
        language = [agent.language_instruction] * len(sequences)
        with torch.no_grad():
            context = agent.core.model.encode_context(
                _image_tensor(agent, observations),
                _proprio_tensor(agent, observations),
                language,
            )["task_scene_cond"]
            next_context = agent.core.model.encode_context(
                _image_tensor(agent, next_observations),
                _proprio_tensor(agent, next_observations),
                language,
            )["task_scene_cond"]
            q1, q2 = agent.core.model.qs(context, actions)
            q_min = torch.minimum(q1, q2)
            value = agent.core.model.value(context)
            next_value = agent.core.model.value(next_context)
            target = rewards + (discount**horizon) * (1.0 - dones) * next_value
        for index, start in enumerate(starts):
            metrics.append(
                {
                    "step": int(start),
                    "reward_horizon": float(rewards[index].item()),
                    "done_horizon": float(dones[index].item()),
                    "q1": float(q1[index].item()),
                    "q2": float(q2[index].item()),
                    "q_min": float(q_min[index].item()),
                    "v": float(value[index].item()),
                    "next_v": float(next_value[index].item()),
                    "td_target": float(target[index].item()),
                    "td_residual": float((target[index] - q_min[index]).item()),
                    "advantage": float((q_min[index] - value[index]).item()),
                }
            )
    return metrics


def _write_outputs(output_dir: Path, metrics: list[dict], transitions: list, cameras: list[str], fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "qv.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    steps = np.asarray([row["step"] for row in metrics])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(steps, [row["q_min"] for row in metrics], label="Q min")
    axes[0].plot(steps, [row["v"] for row in metrics], label="V")
    axes[0].legend()
    axes[1].plot(steps, [row["td_target"] for row in metrics], label="TD target")
    axes[1].plot(steps, [row["td_residual"] for row in metrics], label="TD residual")
    axes[1].legend()
    axes[2].plot(steps, [row["advantage"] for row in metrics], label="Q - V")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].legend()
    axes[2].set_xlabel("rollout step")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "qv.png", dpi=160)
    plt.close(fig)
    with imageio.get_writer(
        output_dir / "rollout.mp4",
        fps=int(fps),
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        for transition in transitions:
            frames = [np.asarray(transition.obs[name], dtype=np.uint8) for name in cameras]
            writer.append_data(np.concatenate(frames, axis=1))
    summary = {
        "num_windows": len(metrics),
        "q_min_mean": float(np.mean([row["q_min"] for row in metrics])),
        "v_mean": float(np.mean([row["v"] for row in metrics])),
        "mean_abs_td_residual": float(
            np.mean(np.abs([row["td_residual"] for row in metrics]))
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(cfg: DictConfig, resources: ExitStack) -> None:
    require_cuda()
    set_seed(int(cfg.visualization.seed))
    init_checkpoint, payload = load_flow_checkpoint(cfg.checkpoint.init_path)
    if init_checkpoint is None or payload is None:
        raise FileNotFoundError("checkpoint.init_path must reference a flow checkpoint.")
    settings = flow_checkpoint_settings(payload, str(cfg.task.name))
    env_metadata = load_task_metadata(payload, str(cfg.task.name))
    if env_metadata is None:
        raise KeyError(f"Flow checkpoint has no environment metadata for {cfg.task.name}.")
    checkpoint_cameras = [str(name) for name in settings.get("camera_names", [])]
    camera_names = checkpoint_cameras or [str(name) for name in cfg.env.camera_names]
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.awr, "camera_aliases", {}) or {}).items()
    }
    env = build_robosuite_env(
        build_runtime_config(
            env_metadata,
            camera_names=camera_names,
            image_height=int(cfg.env.img_height),
            image_width=int(cfg.env.img_width),
            control_freq=int(cfg.env.control_freq),
            horizon=int(cfg.env.horizon),
            interactive=False,
        )
    )
    resources.callback(env.close)
    extractor = bind_proprio_extractor(env, env_metadata)
    resources.callback(extractor.close)
    env.reset()
    observation = {
        "state": extractor.extract(env.sim.get_state().flatten()).astype(np.float32),
        **{
            name: np.asarray(
                env.sim.render(
                    height=int(cfg.env.img_height),
                    width=int(cfg.env.img_width),
                    camera_name=camera_aliases.get(name, name),
                ),
                dtype=np.uint8,
            )
            for name in camera_names
        },
    }
    action_low, action_high = (
        np.asarray(value, dtype=np.float32) for value in env.action_spec
    )
    agent = AWRAgent.from_config(
        build_agent_config(cfg, camera_names=camera_names, flow_settings=settings),
        observation_example=observation,
        action_low=action_low,
        action_high=action_high,
    )
    checkpoint = resolve_path(cfg.visualization.checkpoint)
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"AWR checkpoint does not exist: {checkpoint}")
        agent.load_checkpoint(checkpoint, load_buffers=False)
    else:
        agent.load_flow_policy_checkpoint(init_checkpoint, task_name=str(cfg.task.name))
        cache_path = resolve_qv_cache_path(cfg)
        load_qv_cache(cache_path, agent=agent, load_optimizers=False)
    demo_paths = resolve_demo_paths(
        to_absolute_path(str(cfg.data.demo_root)),
        str(cfg.data.task_name),
        "success_rollout",
        directory=to_absolute_path(
            str(
                cfg.visualization.success_rollout_dir
                if cfg.visualization.success_rollout_dir is not None
                else cfg.data.success_dir
            )
        ),
    )
    if not demo_paths:
        raise FileNotFoundError("No successful rollout was found for Q/V visualization.")
    names = list_demo_names(demo_paths[0])
    if not names:
        raise RuntimeError(f"No demonstrations were found in {demo_paths[0]}.")
    transitions = load_hdf5_demos(
        demo_paths[0],
        split="success_rollout",
        camera_names=camera_names,
        camera_aliases=camera_aliases,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
        state_extractor=extractor,
        control_freq=int(cfg.env.control_freq),
        horizon=int(cfg.env.horizon),
        demo_names=[names[int(cfg.visualization.seed) % len(names)]],
    )
    metrics = _compute_metrics(agent, transitions, cfg.visualization.max_steps)
    output_dir = (
        Path(to_absolute_path(str(cfg.logging.output_root)))
        / str(cfg.task.name)
        / str(cfg.visualization.output_directory)
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    _write_outputs(
        output_dir,
        metrics,
        transitions,
        camera_names,
        int(cfg.visualization.video_fps),
    )
    print(f"[output] {output_dir}")


@hydra.main(version_base="1.3", config_path="./config", config_name="overall")
def main(cfg: DictConfig) -> None:
    with ExitStack() as resources:
        _run(cfg, resources)


if __name__ == "__main__":
    main()
