"""Visualize cumulative human-in-the-loop episodes with policy-update boundaries."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from robosuite.pipeline.algorithms.vast.data_util import (
    clone_with_absorbing_success_tail,
)
from robosuite.pipeline.common.episodes import load_round_episode_payloads
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.modules.training.dipole.episode_dataset import (
    _keep_policy_section,
    _split_sections,
    build_offline_transitions,
)
from robosuite.pipeline.modules.visualization.vast import renderer
from robosuite.pipeline.modules.visualization.vast.cli import (
    augment_payload_metadata,
    infer_run_dir,
    load_json,
    load_resolved_config,
    require_cuda_device,
)
from robosuite.pipeline.modules.visualization.vast.discriminator import (
    visualize_selected_trajectory_discriminator_nnpu,
    write_rollout_video,
)


@dataclass(frozen=True)
class SelectedOnlineEpisode:
    merged_episode_index: int
    source_round: int
    round_episode_index: int
    terminal_reason: str
    episode: dict[str, Any]
    camera_names: list[str]
    source_paths: list[str]


@dataclass(frozen=True)
class OnlineMetricAssembly:
    rows: list[dict[str, Any]]
    valid_rows: list[dict[str, Any]]
    per_step_disc: renderer.PerStepNNPUDisc
    annotation_mask: np.ndarray
    policy_section_count: int
    computed_windows: int


def _episode_has_valid_policy_window(
    episode: dict[str, Any],
    *,
    action_horizon: int,
) -> bool:
    interventions = np.asarray(episode["is_intervention"], dtype=np.bool_)
    sections = _split_sections(interventions)
    terminal_reason = str(episode.get("terminal_reason", ""))
    for index, section in enumerate(sections):
        if section.kind != "policy":
            continue
        keep, _ = _keep_policy_section(sections, index, terminal_reason)
        if keep and int(section.hi - section.lo) >= int(action_horizon):
            return True
    return False


def select_online_episode(
    episodes_paths: Sequence[str | Path],
    *,
    seed: int,
    action_horizon: int,
) -> SelectedOnlineEpisode:
    """Select one eligible original episode from the cumulative round pool."""
    payload = load_round_episode_payloads(episodes_paths)
    episodes = list(payload["episodes"])
    eligible = [
        index
        for index, episode in enumerate(episodes)
        if _episode_has_valid_policy_window(
            episode,
            action_horizon=int(action_horizon),
        )
    ]
    if not eligible:
        raise RuntimeError(
            "Cumulative online episodes contain no policy section with at least "
            f"action_horizon={int(action_horizon)} frames."
        )
    selected_index = random.Random(int(seed)).choice(eligible)
    episode = dict(episodes[selected_index])
    return SelectedOnlineEpisode(
        merged_episode_index=int(selected_index),
        source_round=int(episode.get("source_round", 0)),
        round_episode_index=int(episode.get("source_episode_index", 0)),
        terminal_reason=str(episode.get("terminal_reason", "")),
        episode=episode,
        camera_names=[str(name) for name in payload["camera_names"]],
        source_paths=[str(path) for path in payload.get("_resolved_paths", [])],
    )


def _frame_obs(
    arrays: dict[str, Any],
    index: int,
    camera_names: Sequence[str],
) -> dict[str, np.ndarray]:
    obs = {"state": np.asarray(arrays["state"][index], dtype=np.float32)}
    for camera in camera_names:
        obs[str(camera)] = np.asarray(arrays[str(camera)][index], dtype=np.uint8)
    return obs


def materialize_original_episode(
    selected: SelectedOnlineEpisode,
) -> list[Transition]:
    """Build the full recorded episode for raw videos and global coordinates."""
    episode = selected.episode
    actions = np.asarray(episode["executed_action"], dtype=np.float32)
    interventions = np.asarray(episode["is_intervention"], dtype=np.bool_)
    success = np.asarray(episode["success"], dtype=np.bool_)
    done = np.asarray(episode["done"], dtype=np.bool_)
    reward_values = episode.get("reward")
    rewards = (
        np.zeros(len(actions), dtype=np.float32)
        if reward_values is None
        else np.asarray(reward_values, dtype=np.float32)
    )
    transitions: list[Transition] = []
    for index in range(len(actions)):
        transitions.append(
            Transition(
                obs=_frame_obs(episode["obs"], index, selected.camera_names),
                action=np.asarray(actions[index], dtype=np.float32),
                reward=float(rewards[index]),
                next_obs=_frame_obs(
                    episode["next_obs"], index, selected.camera_names
                ),
                done=bool(done[index]),
                is_intervention=bool(interventions[index]),
                info={
                    "episode_index": int(selected.merged_episode_index),
                    "episode_step": int(index),
                    "source_round": int(selected.source_round),
                    "round_episode_index": int(selected.round_episode_index),
                    "success": bool(success[index]),
                },
                reward_source="online_recorded",
                demo_source="online",
            )
        )
    return transitions


def materialize_policy_sections(
    selected: SelectedOnlineEpisode,
    *,
    action_horizon: int,
    reward_success: float,
    reward_fail: float,
) -> list[list[Transition]]:
    """Reuse policy training's section, reward, and terminal construction."""
    payload = {
        "camera_names": list(selected.camera_names),
        "episodes": [selected.episode],
    }
    streams = build_offline_transitions(
        payload,
        action_horizon=int(action_horizon),
        reward_success=float(reward_success),
        reward_fail=float(reward_fail),
        include_policy_action_neg=False,
    )
    grouped: dict[int, list[Transition]] = {}
    for transition in streams.policy_bc:
        section_index = int((transition.info or {})["episode_index"])
        grouped.setdefault(section_index, []).append(transition)
    sections: list[list[Transition]] = []
    for index in sorted(grouped):
        section, _, _ = clone_with_absorbing_success_tail(
            grouped[index],
            int(action_horizon),
        )
        sections.append(section)
    return sections


def _blank_rows(
    selected: SelectedOnlineEpisode,
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    interventions = np.asarray(selected.episode["is_intervention"], dtype=np.bool_)
    rows: list[dict[str, Any]] = []
    for step, intervention in enumerate(interventions):
        row: dict[str, Any] = {
            "step": float(step),
            "window_start": float(step),
            "source_round": float(selected.source_round),
            "source_episode_index": float(selected.round_episode_index),
            "episode_index": float(selected.merged_episode_index),
            "section_index": float("nan"),
            "section_step": float("nan"),
            "is_intervention": float(intervention),
            "valid_window": 0.0,
        }
        row.update({key: float("nan") for key in metric_keys})
        rows.append(row)
    return rows


def assemble_online_metrics(
    selected: SelectedOnlineEpisode,
    sections: Sequence[list[Transition]],
    section_results: Sequence[
        tuple[list[dict[str, float]], renderer.PerStepNNPUDisc]
    ],
    *,
    advantage_estimator: str,
    threshold: float,
) -> OnlineMetricAssembly:
    """Map section-local window results onto the original episode timeline."""
    if len(sections) != len(section_results):
        raise ValueError("sections and section_results must have equal length")
    metric_keys: set[str] = {"advantage_policy"}
    for rows, _ in section_results:
        for row in rows:
            metric_keys.update(row)
    metric_keys.difference_update({"step", "window_start"})
    rows = _blank_rows(selected, sorted(metric_keys))
    length = len(rows)
    failure = np.full(length, np.nan, dtype=np.float32)
    intrinsic = np.full(length, np.nan, dtype=np.float32)
    predictions = np.full(length, np.nan, dtype=np.float32)
    annotations = np.zeros(length, dtype=np.bool_)
    valid_rows: list[dict[str, Any]] = []

    for section_index, (section, result) in enumerate(
        zip(sections, section_results)
    ):
        local_rows, _ = result
        for section_step, transition in enumerate(section):
            if bool(
                (transition.info or {}).get("synthetic_vast_success_tail", False)
            ):
                continue
            source_step = int((transition.info or {})["source_frame_index"])
            rows[source_step]["section_index"] = float(section_index)
            rows[source_step]["section_step"] = float(section_step)
        for local_row in local_rows:
            section_step = int(local_row["step"])
            source_step = int(
                (section[section_step].info or {})["source_frame_index"]
            )
            mapped = dict(rows[source_step])
            mapped.update(local_row)
            mapped["step"] = float(source_step)
            mapped["window_start"] = float(source_step)
            mapped["section_index"] = float(section_index)
            mapped["section_step"] = float(section_step)
            mapped["valid_window"] = 1.0
            mapped["advantage_policy"] = float(
                local_row[
                    "advantage_gae"
                    if advantage_estimator == "gae"
                    else "advantage_td1"
                ]
            )
            future_local = float(local_row.get("future_frame_index", np.nan))
            if np.isfinite(future_local) and int(future_local) < len(section):
                future_transition = section[int(future_local)]
                if bool(
                    (future_transition.info or {}).get(
                        "synthetic_vast_success_tail", False
                    )
                ):
                    mapped["future_frame_index"] = float("nan")
                else:
                    mapped["future_frame_index"] = float(
                        (future_transition.info or {})["source_frame_index"]
                    )
            else:
                mapped["future_frame_index"] = float("nan")
            rows[source_step] = mapped
            valid_rows.append(mapped)
            failure[source_step] = float(local_row["failure_score_start"])
            intrinsic[source_step] = float(local_row["disc_intrinsic_step0"])
            predictions[source_step] = float(failure[source_step] >= threshold)
            annotations[source_step] = True

    return OnlineMetricAssembly(
        rows=rows,
        valid_rows=valid_rows,
        per_step_disc=renderer.PerStepNNPUDisc(
            failure_score=failure,
            intrinsic_reward=intrinsic,
            threshold=float(threshold),
            pred_failure=predictions,
        ),
        annotation_mask=annotations,
        policy_section_count=len(sections),
        computed_windows=len(valid_rows),
    )


def _write_online_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, (float, np.floating))
                        and not np.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vast-ckpt", required=True)
    parser.add_argument("--disc-ckpt", required=True)
    parser.add_argument("--online-episodes", nargs="+", required=True)
    parser.add_argument("--task-data-name", required=True)
    parser.add_argument("--split", default="online")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--advantage-estimator", choices=("td1", "gae"), required=True)
    parser.add_argument("--gae-lambda", type=float, default=0.6)
    parser.add_argument("--reward-success", type=float, default=0.0)
    parser.add_argument("--reward-fail", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--vast-sampling-seed", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--disc-viz-image-size", type=int, default=256)
    parser.add_argument("--disc-viz-camera", default=None)
    parser.add_argument("--disc-viz-border-thickness", type=int, default=10)
    parser.add_argument("--no-disc-reward", action="store_true")
    parser.add_argument("--no-disc-viz", action="store_true")
    parser.add_argument("--no-flip-vertical", action="store_true")
    args = parser.parse_args()
    if str(args.split).strip().lower() != "online":
        parser.error("online renderer requires --split online")
    if args.max_windows is not None and int(args.max_windows) <= 0:
        parser.error("--max-windows must be positive")
    return args


def main() -> None:
    args = _parse_args()
    device = require_cuda_device(args.device)
    checkpoint = renderer.resolve_cli_path(args.vast_ckpt)
    run_dir = infer_run_dir(checkpoint)
    run_info = load_json(run_dir / "run_info.json")
    resolved_cfg = load_resolved_config(run_dir / "config_resolved.yaml")
    payload = augment_payload_metadata(
        renderer.load_vast_payload(checkpoint),
        checkpoint=checkpoint,
        run_info=run_info,
        resolved_cfg=resolved_cfg,
        task_data_name=args.task_data_name,
        disc_ckpt=args.disc_ckpt,
    )
    learner, encoder, discriminator, cfg, meta = renderer.build_models(
        payload,
        device=device,
        disc_override=args.disc_ckpt,
    )
    camera_names = [str(name) for name in meta["policy_camera_names"]]
    selected = select_online_episode(
        args.online_episodes,
        seed=int(args.seed),
        action_horizon=int(cfg.action_horizon),
    )
    if set(selected.camera_names) != set(camera_names):
        raise RuntimeError(
            f"Online episode cameras {selected.camera_names} do not match "
            f"checkpoint {camera_names}."
        )
    original = materialize_original_episode(selected)
    sections = materialize_policy_sections(
        selected,
        action_horizon=int(cfg.action_horizon),
        reward_success=float(args.reward_success),
        reward_fail=float(args.reward_fail),
    )

    remaining = args.max_windows
    section_results: list[
        tuple[list[dict[str, float]], renderer.PerStepNNPUDisc]
    ] = []
    computed_sections: list[list[Transition]] = []
    for section in sections:
        if remaining is not None and remaining <= 0:
            break
        max_windows = None if remaining is None else int(remaining)
        result = renderer.compute_metrics(
            section,
            learner=learner,
            encoder=encoder,
            discriminator=discriminator,
            cfg=cfg,
            camera_names=camera_names,
            batch_size=int(args.batch_size),
            max_windows=max_windows,
            use_disc_reward=not bool(args.no_disc_reward),
            gae_lambda=float(args.gae_lambda),
            vast_sampling_seed=args.vast_sampling_seed,
            boundary_semantics="policy_update",
        )
        computed_sections.append(section)
        section_results.append(result)
        if remaining is not None:
            remaining -= len(result[0])

    assembly = assemble_online_metrics(
        selected,
        computed_sections,
        section_results,
        advantage_estimator=str(args.advantage_estimator),
        threshold=float(discriminator.threshold),
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        renderer.resolve_cli_path(args.output_root)
        / f"online_seed{int(args.seed)}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{args.task_data_name} online round {selected.source_round:03d} "
        f"episode {selected.round_episode_index} "
        f"(policy {args.advantage_estimator.upper()})"
    )
    _write_online_metrics_csv(output_dir / "steps.csv", assembly.rows)
    plot_paths = renderer.plot_vast(
        output_dir / "vast_timeseries",
        assembly.rows,
        title=title,
        action_horizon=int(cfg.action_horizon),
        per_step_disc=assembly.per_step_disc,
        gae_lambda=float(args.gae_lambda),
    )
    flip_vertical = not bool(args.no_flip_vertical)
    rollout_video = write_rollout_video(
        output_dir / "rollout_policy_obs.mp4",
        original,
        camera_names=camera_names,
        camera_name=args.disc_viz_camera,
        fps=int(args.video_fps),
        flip_vertical=flip_vertical,
    )
    pair_video = renderer.write_vast_current_future_video(
        output_dir / "vast_current_future.mp4",
        original,
        assembly.valid_rows,
        camera_names=camera_names,
        camera_name=args.disc_viz_camera,
        fps=int(args.video_fps),
        flip_vertical=flip_vertical,
    )

    disc_outputs: dict[str, str] | None = None
    if not bool(args.no_disc_viz):
        disc_transitions = renderer.upscale_transition_images_for_disc_viz(
            original,
            image_size=int(args.disc_viz_image_size),
        )
        disc_result = visualize_selected_trajectory_discriminator_nnpu(
            output_dir=output_dir / "discriminator",
            transitions=disc_transitions,
            camera_names=camera_names,
            failure_score=assembly.per_step_disc.failure_score,
            intrinsic_reward=assembly.per_step_disc.intrinsic_reward,
            pred_failure=assembly.per_step_disc.pred_failure,
            threshold=assembly.per_step_disc.threshold,
            ckpt_path=discriminator.ckpt_path,
            task_name=discriminator.task_name,
            video_fps=int(args.video_fps),
            camera_name=args.disc_viz_camera,
            border_thickness=int(args.disc_viz_border_thickness),
            flip_vertical=flip_vertical,
            annotation_mask=assembly.annotation_mask,
        )
        disc_outputs = {
            "scores_csv": str(disc_result.scores_csv),
            "plot_png": str(disc_result.plot_png),
            "plot_pdf": str(disc_result.plot_pdf),
            "video": str(disc_result.video),
            "summary": str(disc_result.summary_json),
        }

    summary = {
        "schema_version": 1,
        "split": "online",
        "selection_seed": int(args.seed),
        "source_episode_paths": selected.source_paths,
        "source_rounds": list(range(len(selected.source_paths))),
        "selected_merged_episode_index": int(selected.merged_episode_index),
        "selected_source_round": int(selected.source_round),
        "selected_round_episode_index": int(selected.round_episode_index),
        "terminal_reason": selected.terminal_reason,
        "num_frames": len(original),
        "intervention_frames": int(
            np.asarray(selected.episode["is_intervention"], dtype=np.bool_).sum()
        ),
        "blank_score_frames": int(len(original) - assembly.annotation_mask.sum()),
        "policy_section_count": int(len(sections)),
        "computed_windows": int(assembly.computed_windows),
        "synthetic_success_tail_transitions": int(
            sum(
                bool(
                    (transition.info or {}).get(
                        "synthetic_vast_success_tail", False
                    )
                )
                for section in sections
                for transition in section
            )
        ),
        "padded_success_sections": int(
            sum(
                any(
                    bool(
                        (transition.info or {}).get(
                            "synthetic_vast_success_tail", False
                        )
                    )
                    for transition in section
                )
                for section in sections
            )
        ),
        "absorbing_reward_mask_transitions": int(
            sum(
                bool((transition.info or {}).get("frozen_post_success", False))
                for section in sections
                for transition in section
            )
        ),
        "policy_advantage": {
            "estimator": str(args.advantage_estimator),
            "gae_lambda": (
                float(args.gae_lambda)
                if str(args.advantage_estimator) == "gae"
                else None
            ),
            "reward_success": float(args.reward_success),
            "reward_fail": float(args.reward_fail),
        },
        "vast_checkpoint": str(checkpoint),
        "nnpu_checkpoint": discriminator.ckpt_path,
        "device": device,
        "outputs": {
            "steps_csv": str(output_dir / "steps.csv"),
            "plot_png": str(plot_paths["overlapping"]),
            "plot_png_nonoverlap": str(plot_paths["nonoverlap"]),
            "vast_plot_png": str(plot_paths.get("vast", "")) or None,
            "video": None if rollout_video is None else str(rollout_video),
            "vast_current_future_video": (
                None if pair_video is None else str(pair_video)
            ),
            "discriminator": disc_outputs,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[vis_vast][online] selected round={selected.source_round:03d} "
        f"episode={selected.round_episode_index} seed={args.seed}"
    )
    print(f"[vis_vast][online] output_dir={output_dir}")


if __name__ == "__main__":
    main()
