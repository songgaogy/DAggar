from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.task_registry import normalize_task_name, ordered_task_names
from robosuite.discriminator.lpb_score.app.pipeline import build_flow_encoder
from robosuite.discriminator.tsne.data import (
    _build_point_dataframe,
    _encode_annotated_failure_sequences,
    _encode_standard_sequences,
    _encode_suboptimal_sequences,
    _limit_annotated_failure_sequences,
    _limit_sequences_by_total_points,
    _list_annotated_failure_demo_refs,
    _list_standard_demo_refs,
    _list_suboptimal_demo_refs,
    _sorted_task_names,
    _summary_counts,
)
from robosuite.discriminator.tsne.embedding import _run_embedding
from robosuite.discriminator.tsne.plotting import (
    _annotated_failure_modes,
    _configure_plot_style,
    _plot_annotated_failure_embedding,
    _plot_annotated_failure_embedding_by_task,
    _plot_annotated_failure_embedding_per_mode,
    _plot_by_task,
    _plot_failure_pattern,
    _plot_per_task_failure_patterns,
)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _run_annotated_failure_analysis(cfg: DictConfig, encoder) -> None:
    failure_refs, failure_ref_summary, task_names = _list_annotated_failure_demo_refs(cfg)
    if not task_names:
        raise RuntimeError("No annotated failure tasks were discovered for embedding analysis.")
    task_to_index = {
        task_name: idx
        for idx, task_name in enumerate(task_names)
    }
    success_max_rollouts = int(cfg.annotated_failures.max_success_rollouts_per_task)

    if success_max_rollouts == 0:
        success_refs = []
        success_sequences = []
        success_ref_summary = {
            task_name: {
                "success_rollout": {
                    "available": 0,
                    "selected": 0,
                }
            }
            for task_name in task_names
        }
        success_encode_summary = {
            "num_requested": 0,
            "num_cached_refs": 0,
            "num_sequences": 0,
            "skipped": True,
            "reason": "annotated_failures.max_success_rollouts_per_task=0",
        }
    else:
        success_refs, success_ref_summary, _ = _list_standard_demo_refs(
            cfg,
            task_names_override=task_names,
            source_names_override=["success_rollout"],
            max_rollouts_override=success_max_rollouts,
        )
        success_sequences, success_encode_summary = _encode_standard_sequences(
            refs=success_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cfg=cfg,
        )
    failure_sequences, failure_encode_summary = _encode_annotated_failure_sequences(
        refs=failure_refs,
        encoder=encoder,
        task_to_index=task_to_index,
        cfg=cfg,
    )

    sequences = list(success_sequences) + list(failure_sequences)
    if not sequences:
        raise RuntimeError("No trajectories were selected for annotated failure embedding analysis.")

    sequences, point_limit_summary = _limit_annotated_failure_sequences(
        sequences,
        timestep_stride=int(cfg.data.timestep_stride),
        max_total_points=int(cfg.analysis.max_total_points),
        seed=int(cfg.seed),
    )
    features, frame = _build_point_dataframe(
        sequences,
        timestep_stride=int(cfg.data.timestep_stride),
    )
    embedding, embedding_summary = _run_embedding(features, cfg)
    frame["embedding_x"] = embedding[:, 0]
    frame["embedding_y"] = embedding[:, 1]

    success_task_names = set(frame.loc[frame["source_name"] == "success_rollout", "task_name"].astype(str).tolist())
    failure_task_names = set(frame.loc[frame["source_name"] == "fail_rollout", "task_name"].astype(str).tolist())
    missing_success_tasks = _sorted_task_names(list(set(task_names) - success_task_names))
    missing_failure_tasks = _sorted_task_names(list(set(task_names) - failure_task_names))
    if success_max_rollouts != 0 and missing_success_tasks:
        raise RuntimeError(
            "Annotated failure plots require success_rollout points for every task. "
            f"Missing success data for: {missing_success_tasks}"
        )
    if missing_failure_tasks:
        raise RuntimeError(
            "Annotated failure plots require annotated failure points for every task. "
            f"Missing failure data for: {missing_failure_tasks}"
        )

    save_dir = to_absolute_path(str(cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    run_dir = os.path.join(save_dir, f"run_{_now_tag()}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "embedding_points.csv")
    frame.to_csv(csv_path, index=False)

    npz_path = os.path.join(run_dir, "embedding_features.npz")
    np.savez_compressed(
        npz_path,
        latents=features.astype(np.float32),
        embedding_xy=embedding.astype(np.float32),
        rollout_index=frame["rollout_index"].to_numpy(dtype=np.int64),
        task_index=frame["task_index"].to_numpy(dtype=np.int64),
        timestep=frame["timestep"].to_numpy(dtype=np.int64),
        failure_label=frame["failure_label"].to_numpy(dtype=np.int64),
        success_label=frame["success_label"].to_numpy(dtype=np.int64),
        task_name=frame["task_name"].astype(str).to_numpy(),
        source_name=frame["source_name"].astype(str).to_numpy(),
        split=frame["split"].astype(str).to_numpy(),
        rollout_id=frame["rollout_id"].astype(str).to_numpy(),
        file_path=frame["file_path"].astype(str).to_numpy(),
        demo_key=frame["demo_key"].astype(str).to_numpy(),
        display_group=frame["display_group"].astype(str).to_numpy(),
        display_color=frame["display_color"].astype(str).to_numpy(),
        failure_mode=frame["failure_mode"].astype(str).to_numpy(),
        ood_progress=frame["ood_progress"].to_numpy(dtype=np.float32),
        segment_progress=frame["ood_progress"].to_numpy(dtype=np.float32),
    )

    annotated_failure_plot_paths = _plot_annotated_failure_embedding(frame, cfg, run_dir)
    by_task_plot_paths = _plot_annotated_failure_embedding_by_task(frame, cfg, run_dir)
    per_mode_plot_paths = _plot_annotated_failure_embedding_per_mode(frame, cfg, run_dir)
    summary = {
        "timestamp": _now_tag(),
        "analysis_mode": "annotated_failures",
        "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
        "save_dir": run_dir,
        "task_names": task_names,
        "feature_dim": int(features.shape[1]),
        "num_sequences": int(len(sequences)),
        "num_points": int(features.shape[0]),
        "data": {
            "root_dir": to_absolute_path(str(cfg.data.root_dir)),
            "cache_dir": to_absolute_path(str(cfg.data.cache_dir)),
            "image_size": int(cfg.data.image_size),
            "source_data_types": ["success_rollout", "annotated_fail_rollout"],
            "max_rollouts_per_task_source": int(cfg.data.max_rollouts_per_task_source),
            "success_max_rollouts_per_task": int(success_max_rollouts),
            "timestep_stride": int(cfg.data.timestep_stride),
            "build_missing_cache": bool(cfg.data.build_missing_cache),
        },
        "annotated_failures": {
            "enabled": bool(cfg.annotated_failures.enabled),
            "root_dir": to_absolute_path(str(cfg.annotated_failures.root_dir)),
            "tasks": [normalize_task_name(str(name)) for name in task_names],
            "hdf5_name": str(cfg.annotated_failures.hdf5_name),
            "max_rollouts_per_task": int(cfg.annotated_failures.max_rollouts_per_task),
            "max_success_rollouts_per_task": int(success_max_rollouts),
            "mode_handling": "pooled_overview_plus_one_plot_per_mode",
            "failure_modes": _annotated_failure_modes(frame),
            "ref_summary": failure_ref_summary,
            "encode_summary": failure_encode_summary,
        },
        "success_rollout": {
            "ref_summary": success_ref_summary,
            "encode_summary": success_encode_summary,
        },
        "sampling": point_limit_summary,
        "embedding": embedding_summary,
        "counts": _summary_counts(frame),
        "outputs": {
            "csv_path": csv_path,
            "npz_path": npz_path,
            "annotated_failure_embedding": annotated_failure_plot_paths,
            "annotated_failure_embedding_by_task": by_task_plot_paths,
            "per_mode": per_mode_plot_paths,
        },
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, ensure_ascii=False, default=_json_default)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    print(f"[embedding] Saved analysis summary to: {summary_path}")


def run_analyse(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    _configure_plot_style()
    encoder = build_flow_encoder(cfg)
    try:
        if bool(cfg.annotated_failures.enabled):
            _run_annotated_failure_analysis(cfg, encoder)
            return

        standard_refs, standard_ref_summary, task_names = _list_standard_demo_refs(cfg)
        task_to_index = {
            task_name: idx
            for idx, task_name in enumerate(task_names)
        }
        standard_sequences, standard_encode_summary = _encode_standard_sequences(
            refs=standard_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cfg=cfg,
        )

        suboptimal_refs, suboptimal_ref_summary = _list_suboptimal_demo_refs(cfg)
        suboptimal_sequences, suboptimal_encode_summary = _encode_suboptimal_sequences(
            refs=suboptimal_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cfg=cfg,
        )

        sequences = list(standard_sequences) + list(suboptimal_sequences)
        if not sequences:
            raise RuntimeError("No trajectories were selected for embedding analysis.")

        sequences, point_limit_summary = _limit_sequences_by_total_points(
            sequences,
            timestep_stride=int(cfg.data.timestep_stride),
            max_total_points=int(cfg.analysis.max_total_points),
            seed=int(cfg.seed),
        )
        features, frame = _build_point_dataframe(
            sequences,
            timestep_stride=int(cfg.data.timestep_stride),
        )
        embedding, embedding_summary = _run_embedding(features, cfg)
        frame["embedding_x"] = embedding[:, 0]
        frame["embedding_y"] = embedding[:, 1]

        expert_task_names = set(frame.loc[frame["source_name"] == "expert", "task_name"].astype(str).tolist())
        suboptimal_task_names = set(frame.loc[frame["source_name"] == "suboptimal", "task_name"].astype(str).tolist())
        missing_expert_tasks = ordered_task_names(list(set(task_names) - expert_task_names))
        missing_suboptimal_tasks = ordered_task_names(list(set(task_names) - suboptimal_task_names))
        if missing_expert_tasks:
            raise RuntimeError(
                "Failure-pattern plots require expert points for every task. "
                f"Missing expert data for: {missing_expert_tasks}"
            )
        if missing_suboptimal_tasks:
            raise RuntimeError(
                "Failure-pattern plots require suboptimal points for every task. "
                f"Missing suboptimal data for: {missing_suboptimal_tasks}"
            )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        run_dir = os.path.join(save_dir, f"run_{_now_tag()}")
        os.makedirs(run_dir, exist_ok=True)

        csv_path = os.path.join(run_dir, "embedding_points.csv")
        frame.to_csv(csv_path, index=False)

        npz_path = os.path.join(run_dir, "embedding_features.npz")
        np.savez_compressed(
            npz_path,
            latents=features.astype(np.float32),
            embedding_xy=embedding.astype(np.float32),
            rollout_index=frame["rollout_index"].to_numpy(dtype=np.int64),
            task_index=frame["task_index"].to_numpy(dtype=np.int64),
            timestep=frame["timestep"].to_numpy(dtype=np.int64),
            failure_label=frame["failure_label"].to_numpy(dtype=np.int64),
            success_label=frame["success_label"].to_numpy(dtype=np.int64),
            task_name=frame["task_name"].astype(str).to_numpy(),
            source_name=frame["source_name"].astype(str).to_numpy(),
            split=frame["split"].astype(str).to_numpy(),
            rollout_id=frame["rollout_id"].astype(str).to_numpy(),
            file_path=frame["file_path"].astype(str).to_numpy(),
            demo_key=frame["demo_key"].astype(str).to_numpy(),
            display_group=frame["display_group"].astype(str).to_numpy(),
            display_color=frame["display_color"].astype(str).to_numpy(),
            failure_mode=frame["failure_mode"].astype(str).to_numpy(),
            ood_progress=frame["ood_progress"].to_numpy(dtype=np.float32),
        )

        plot_b_paths = _plot_by_task(frame, cfg, run_dir)
        failure_pattern_paths = _plot_failure_pattern(frame, cfg, run_dir)
        per_task_paths = _plot_per_task_failure_patterns(frame, cfg, run_dir)

        summary = {
            "timestamp": _now_tag(),
            "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
            "save_dir": run_dir,
            "task_names": task_names,
            "feature_dim": int(features.shape[1]),
            "num_sequences": int(len(sequences)),
            "num_points": int(features.shape[0]),
            "data": {
                "root_dir": to_absolute_path(str(cfg.data.root_dir)),
                "cache_dir": to_absolute_path(str(cfg.data.cache_dir)),
                "image_size": int(cfg.data.image_size),
                "source_data_types": [str(name) for name in list(cfg.data.source_data_types)],
                "max_rollouts_per_task_source": int(cfg.data.max_rollouts_per_task_source),
                "timestep_stride": int(cfg.data.timestep_stride),
                "build_missing_cache": bool(cfg.data.build_missing_cache),
            },
            "suboptimal": {
                "enabled": bool(cfg.suboptimal.enabled),
                "root_dir": to_absolute_path(str(cfg.suboptimal.root_dir)),
                "tasks": [normalize_task_name(str(name)) for name in list(cfg.suboptimal.tasks)],
                "max_rollouts_per_task": int(cfg.suboptimal.max_rollouts_per_task),
                "ref_summary": suboptimal_ref_summary,
                "encode_summary": suboptimal_encode_summary,
            },
            "standard": {
                "ref_summary": standard_ref_summary,
                "encode_summary": standard_encode_summary,
            },
            "sampling": point_limit_summary,
            "embedding": embedding_summary,
            "counts": _summary_counts(frame),
            "outputs": {
                "csv_path": csv_path,
                "npz_path": npz_path,
                "plot_b_task": plot_b_paths,
                "failure_pattern": failure_pattern_paths,
                "per_task": per_task_paths,
            },
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False, default=_json_default)

        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        print(f"[embedding] Saved analysis summary to: {summary_path}")
    finally:
        encoder.close()


@hydra.main(version_base="1.2", config_path="./config", config_name="analyse")
def main(cfg: DictConfig) -> None:
    run_analyse(cfg)


if __name__ == "__main__":
    main()
