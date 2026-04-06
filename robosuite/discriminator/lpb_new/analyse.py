from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any, Optional

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.task_registry import normalize_task_name, resolve_checkpoint_task_name
from robosuite.discriminator.lpb_new.app.pipeline import build_flow_encoder
from robosuite.discriminator.lpb_new.core.dataset import (
    LatentTrajectory,
    build_cached_splits,
    filter_refs_by_data_types,
    load_latent_trajectories,
)
from robosuite.discriminator.lpb_new.core.knn_discriminator import LPBKNNDiscriminator
from robosuite.discriminator.utils.evaluation import classification_metrics


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class SuboptimalDemoRef:
    task_name: str
    split: str
    file_path: str
    demo_key: str
    sub_start: int
    sub_stop: int


@dataclass(frozen=True)
class LabeledLatentTrajectory:
    trajectory: LatentTrajectory
    labels: np.ndarray
    sub_start: int
    sub_stop: int
    split: str


@dataclass(frozen=True)
class MetricSequenceSample:
    task_name: str
    split: str
    file_path: str
    demo_key: str
    components: dict[str, np.ndarray]
    labels: Optional[np.ndarray]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _build_analysis_detector(cfg: DictConfig) -> LPBKNNDiscriminator:
    transition_weight = 1.0 if bool(cfg.feature.use_transition_error) else 0.0
    return LPBKNNDiscriminator(
        checkpoint_path=to_absolute_path(str(cfg.model.lpb_ckpt)),
        feature_device=str(cfg.feature.device),
        feature_batch_size=int(cfg.feature.batch_size),
        action_horizon=int(cfg.feature.action_horizon),
        normalize_feature=bool(cfg.feature.normalize_feature),
        normalize_policy_chunk=bool(cfg.feature.normalize_policy_chunk),
        policy_history_steps=int(cfg.feature.policy_history_steps),
        use_transition_error=bool(cfg.feature.use_transition_error),
        detector_device=str(cfg.detector.device),
        delta=float(cfg.detector.delta),
        delta_step=float(cfg.detector.delta_step),
        knn_chunk_size=int(cfg.detector.knn_chunk_size),
        lambda_mode=str(cfg.detector.lambda_mode),
        lambda_window_size=int(cfg.detector.lambda_window_size),
        feature_knn_weight=1.0,
        transition_aux_weight=float(transition_weight),
        policy_chunk_weight=1.0,
        dynamics_weight=1.0,
        neighbor_topk=int(cfg.detector.neighbor_topk),
        dynamics_temperature=float(cfg.detector.dynamics_temperature),
        weight_json_path=None,
    )


def _list_suboptimal_demo_refs(cfg: DictConfig) -> tuple[list[SuboptimalDemoRef], dict[str, dict[str, int]]]:
    root_dir = to_absolute_path(str(cfg.suboptimal.root_dir))
    train_count = int(cfg.suboptimal.train_count_per_task)
    eval_count = int(cfg.suboptimal.eval_count_per_task)
    total_required = train_count + eval_count
    seed = int(cfg.suboptimal.seed)

    refs: list[SuboptimalDemoRef] = []
    split_summary: dict[str, dict[str, int]] = {}

    for task_offset, task_name_raw in enumerate(list(cfg.suboptimal.tasks)):
        task_name = normalize_task_name(str(task_name_raw))
        sub_dir = os.path.join(root_dir, f"{resolve_checkpoint_task_name(task_name)}_allview")
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"Missing suboptimal directory for {task_name}: {sub_dir}")

        task_refs: list[SuboptimalDemoRef] = []
        for file_path in sorted(glob.glob(os.path.join(sub_dir, "*.hdf5"))):
            with h5py.File(file_path, "r") as file_handle:
                if "demos" not in file_handle:
                    continue
                for demo_key in sorted(file_handle["demos"].keys()):
                    demo = file_handle["demos"][demo_key]
                    task_refs.append(
                        SuboptimalDemoRef(
                            task_name=task_name,
                            split="",
                            file_path=file_path,
                            demo_key=demo_key,
                            sub_start=int(np.asarray(demo["sub_start"])[()]),
                            sub_stop=int(np.asarray(demo["sub_stop"])[()]),
                        )
                    )

        if len(task_refs) < total_required:
            raise ValueError(
                f"Task {task_name} requires at least {total_required} suboptimal demos, found {len(task_refs)}."
            )

        rng = np.random.default_rng(seed + task_offset * 97)
        indices = np.arange(len(task_refs))
        rng.shuffle(indices)
        selected_refs = [task_refs[int(idx)] for idx in indices[:total_required]]

        split_summary[task_name] = {"train": 0, "eval": 0}
        for local_idx, ref in enumerate(selected_refs):
            split_name = "train" if local_idx < train_count else "eval"
            refs.append(
                SuboptimalDemoRef(
                    task_name=ref.task_name,
                    split=split_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                    sub_start=int(ref.sub_start),
                    sub_stop=int(ref.sub_stop),
                )
            )
            split_summary[task_name][split_name] += 1

    return refs, split_summary


def _encode_suboptimal_refs(
    refs: list[SuboptimalDemoRef],
    *,
    encoder,
    task_to_index: dict[str, int],
    horizon: int,
    batch_size: int,
) -> tuple[list[LabeledLatentTrajectory], dict[str, Any]]:
    encoded: list[LabeledLatentTrajectory] = []
    dropped: list[dict[str, Any]] = []
    prepared_batch = []
    ref_batch: list[SuboptimalDemoRef] = []

    def flush_batch() -> None:
        nonlocal prepared_batch, ref_batch
        if not prepared_batch:
            return
        encoded_batch = encoder.encode_prepared_demos(prepared_batch)
        encoded_lookup = {
            (item.file_path, item.demo_key): item
            for item in encoded_batch
        }
        for ref in ref_batch:
            key = (ref.file_path, ref.demo_key)
            item = encoded_lookup.get(key)
            if item is None:
                raise KeyError(f"Missing encoded suboptimal demo for {key}")
            length = min(int(item.latents.shape[0]), int(item.actions.shape[0]))
            valid_len = length - int(horizon)
            if valid_len <= 0:
                dropped.append(
                    {
                        "task_name": ref.task_name,
                        "split": ref.split,
                        "file_path": ref.file_path,
                        "demo_key": ref.demo_key,
                        "reason": "too_short_for_horizon",
                    }
                )
                continue

            start = int(np.clip(ref.sub_start, 0, valid_len))
            stop = int(np.clip(ref.sub_stop, 0, valid_len))
            if stop <= start:
                dropped.append(
                    {
                        "task_name": ref.task_name,
                        "split": ref.split,
                        "file_path": ref.file_path,
                        "demo_key": ref.demo_key,
                        "reason": "empty_positive_interval_after_clip",
                        "sub_start": int(ref.sub_start),
                        "sub_stop": int(ref.sub_stop),
                        "valid_len": int(valid_len),
                    }
                )
                continue

            labels = np.zeros((valid_len,), dtype=np.int64)
            labels[start:stop] = 1
            encoded.append(
                LabeledLatentTrajectory(
                    trajectory=LatentTrajectory(
                        latents=np.asarray(item.latents, dtype=np.float32),
                        actions=np.asarray(item.actions, dtype=np.float32),
                        task_name=ref.task_name,
                        task_index=int(task_to_index[ref.task_name]),
                        data_type="suboptimal",
                        data_type_index=-1,
                        split=ref.split,
                        file_path=ref.file_path,
                        demo_key=ref.demo_key,
                    ),
                    labels=labels,
                    sub_start=int(ref.sub_start),
                    sub_stop=int(ref.sub_stop),
                    split=ref.split,
                )
            )
        prepared_batch = []
        ref_batch = []

    for idx, ref in enumerate(refs):
        prepared_batch.append(
            encoder.load_demo_raw(
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
        )
        ref_batch.append(ref)
        if len(prepared_batch) >= batch_size:
            flush_batch()
        if (idx + 1) % 20 == 0 or (idx + 1) == len(refs):
            print(f"[lpb_new analyse] encoded_suboptimal {idx + 1}/{len(refs)}")

    flush_batch()
    summary = {
        "num_requested": int(len(refs)),
        "num_encoded": int(len(encoded)),
        "num_dropped": int(len(dropped)),
        "dropped": dropped,
    }
    return encoded, summary


def _build_metric_samples(
    detector: LPBKNNDiscriminator,
    trajectories: list[LatentTrajectory] | list[LabeledLatentTrajectory],
    *,
    has_labels: bool,
) -> list[MetricSequenceSample]:
    samples: list[MetricSequenceSample] = []
    for item in trajectories:
        if has_labels:
            assert isinstance(item, LabeledLatentTrajectory)
            trajectory = item.trajectory
            labels = np.asarray(item.labels, dtype=np.int64)
        else:
            assert isinstance(item, LatentTrajectory)
            trajectory = item
            labels = None

        bundle = detector.extractor.encode_trajectory_bundle(trajectory)
        raw_components = detector._compute_component_scores(bundle)
        normalized_components = {}
        for key, values in raw_components.items():
            scale = float(detector._score_scales.get(key, 1.0))
            if scale <= 1e-8:
                scale = 1.0
            normalized_components[key] = (
                np.asarray(values, dtype=np.float32) / scale
            ).astype(np.float32)

        if not normalized_components:
            raise RuntimeError("No metric components available for analysis.")

        valid_len = int(next(iter(normalized_components.values())).shape[0])
        if labels is not None and int(labels.shape[0]) != valid_len:
            raise ValueError(
                f"Label length mismatch for {trajectory.file_path}:{trajectory.demo_key}: "
                f"{labels.shape[0]} vs {valid_len}"
            )

        samples.append(
            MetricSequenceSample(
                task_name=str(trajectory.task_name),
                split=str(trajectory.split),
                file_path=str(trajectory.file_path),
                demo_key=str(trajectory.demo_key),
                components=normalized_components,
                labels=None if labels is None else labels.copy(),
            )
        )
    return samples


def _ordered_candidate_metrics(detector: LPBKNNDiscriminator, samples: list[MetricSequenceSample]) -> list[str]:
    available = set(samples[0].components.keys())
    return [key for key in detector.COMPONENT_ORDER if key in available]


def _combine_sample_components(sample: MetricSequenceSample, weights: dict[str, float]) -> np.ndarray:
    step_scores = np.zeros_like(next(iter(sample.components.values())), dtype=np.float32)
    for key, values in sample.components.items():
        weight = float(weights.get(key, 0.0))
        if weight > 0.0:
            step_scores = step_scores + weight * np.asarray(values, dtype=np.float32)
    return step_scores.astype(np.float32, copy=False)


def _calibrate_threshold(
    detector: LPBKNNDiscriminator,
    clean_samples: list[MetricSequenceSample],
    weights: dict[str, float],
) -> float:
    calib_lambdas = [
        detector.detector._aggregate_lambda(_combine_sample_components(sample, weights))
        for sample in clean_samples
    ]
    merged = np.concatenate(calib_lambdas, axis=0).astype(np.float32)
    return float(detector.detector._compute_threshold(merged, detector.detector.delta))


def _evaluate_labeled_samples(
    detector: LPBKNNDiscriminator,
    samples: list[MetricSequenceSample],
    weights: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for sample in samples:
        if sample.labels is None:
            raise ValueError("Expected labels for evaluation samples.")
        lamb = detector.detector._aggregate_lambda(_combine_sample_components(sample, weights))
        preds = (lamb >= float(threshold)).astype(np.int64)
        y_true.extend(np.asarray(sample.labels, dtype=np.int64).tolist())
        y_pred.extend(preds.tolist())

    metrics = classification_metrics(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
    )
    metrics["num_trajectories"] = float(len(samples))
    metrics["num_steps"] = float(len(y_true))
    metrics["positive_steps"] = float(int(np.sum(np.asarray(y_true, dtype=np.int64))))
    metrics["pred_positive_steps"] = float(int(np.sum(np.asarray(y_pred, dtype=np.int64))))
    metrics["threshold"] = float(threshold)
    return metrics


def _clean_false_alarm_rate(
    detector: LPBKNNDiscriminator,
    clean_samples: list[MetricSequenceSample],
    weights: dict[str, float],
    threshold: float,
) -> float:
    total = 0
    positives = 0
    for sample in clean_samples:
        lamb = detector.detector._aggregate_lambda(_combine_sample_components(sample, weights))
        preds = (lamb >= float(threshold)).astype(np.int64)
        total += int(preds.size)
        positives += int(preds.sum())
    return float(positives / total) if total > 0 else float("nan")


def _tensorize_subset(
    samples: list[MetricSequenceSample],
    subset: tuple[str, ...],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    matrices: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for sample in samples:
        matrix = np.stack([np.asarray(sample.components[key], dtype=np.float32) for key in subset], axis=1)
        matrices.append(torch.from_numpy(matrix).to(device=device, dtype=torch.float32))
        if sample.labels is None:
            raise ValueError("Expected labels when tensorizing subset.")
        labels.append(torch.from_numpy(np.asarray(sample.labels, dtype=np.float32)).to(device=device))
    return matrices, labels


def _fit_subset_weights(
    train_samples: list[MetricSequenceSample],
    subset: tuple[str, ...],
    *,
    fit_lr: float,
    fit_steps: int,
    device: torch.device,
) -> dict[str, float]:
    if len(subset) == 1:
        return {subset[0]: 1.0}

    matrices, labels = _tensorize_subset(train_samples, subset, device=device)
    raw_weights = torch.zeros((len(subset),), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw_weights], lr=float(fit_lr))

    for _ in range(max(1, int(fit_steps))):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(raw_weights, dim=0)
        score_center = torch.cat([matrix @ weights for matrix in matrices], dim=0).mean().detach()
        loss = torch.zeros((), device=device, dtype=torch.float32)
        total_steps = 0
        for matrix, label in zip(matrices, labels):
            logits = (matrix @ weights) - score_center
            loss = loss + F.binary_cross_entropy_with_logits(logits, label, reduction="sum")
            total_steps += int(label.numel())
        loss = loss / max(1, total_steps)
        loss.backward()
        optimizer.step()

    final_weights = torch.softmax(raw_weights.detach(), dim=0).cpu().numpy().astype(np.float64)
    return {
        key: float(value)
        for key, value in zip(subset, final_weights.tolist())
    }


def _full_weight_dict(
    detector: LPBKNNDiscriminator,
    subset_weights: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(subset_weights.get(key, 0.0))
        for key in detector.COMPONENT_ORDER
    }


def _candidate_is_better(
    candidate: dict[str, Any],
    best: Optional[dict[str, Any]],
    epsilon: float,
) -> bool:
    if best is None:
        return True
    cand_f1 = float(candidate["train_summary"]["f1"])
    best_f1 = float(best["train_summary"]["f1"])
    if cand_f1 > best_f1 + float(epsilon):
        return True
    if best_f1 > cand_f1 + float(epsilon):
        return False

    cand_active = int(len(candidate["active_metrics"]))
    best_active = int(len(best["active_metrics"]))
    if cand_active < best_active:
        return True
    if cand_active > best_active:
        return False

    cand_far = float(candidate["clean_false_alarm_rate"])
    best_far = float(best["clean_false_alarm_rate"])
    if cand_far < best_far - 1e-12:
        return True
    if best_far < cand_far - 1e-12:
        return False

    return tuple(candidate["active_metrics"]) < tuple(best["active_metrics"])


def _search_sparse_weights(
    detector: LPBKNNDiscriminator,
    *,
    clean_samples: list[MetricSequenceSample],
    train_samples: list[MetricSequenceSample],
    fit_lr: float,
    fit_steps: int,
    epsilon: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_metrics = _ordered_candidate_metrics(detector, clean_samples)
    best_result: Optional[dict[str, Any]] = None
    all_results: list[dict[str, Any]] = []

    for subset_size in range(1, len(candidate_metrics) + 1):
        for subset in combinations(candidate_metrics, subset_size):
            subset_weights = _fit_subset_weights(
                train_samples=train_samples,
                subset=tuple(subset),
                fit_lr=float(fit_lr),
                fit_steps=int(fit_steps),
                device=device,
            )
            weights = _full_weight_dict(detector, subset_weights)
            threshold = _calibrate_threshold(detector, clean_samples, weights)
            train_summary = _evaluate_labeled_samples(detector, train_samples, weights, threshold)
            clean_far = _clean_false_alarm_rate(detector, clean_samples, weights, threshold)
            result = {
                "active_metrics": list(subset),
                "weights": weights,
                "train_summary": train_summary,
                "clean_false_alarm_rate": float(clean_far),
            }
            all_results.append(result)
            if _candidate_is_better(result, best_result, epsilon=float(epsilon)):
                best_result = result

    if best_result is None:
        raise RuntimeError("Sparse metric search failed to produce a candidate.")
    return best_result, all_results


def run_analyse(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = build_flow_encoder(cfg)
    try:
        cached_splits, split_summary, task_to_index = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=seed,
        )
        bank_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.bank_split)],
            list(cfg.eval.bank_data_types),
        )
        calibration_refs = filter_refs_by_data_types(
            cached_splits[str(cfg.eval.calibration_split)],
            list(cfg.eval.calibration_data_types),
        )
        bank_trajectories = load_latent_trajectories(bank_refs)
        calibration_trajectories = load_latent_trajectories(calibration_refs)
        if not bank_trajectories or not calibration_trajectories:
            raise RuntimeError("Analysis requires non-empty clean bank and calibration trajectories.")

        detector = _build_analysis_detector(cfg)
        calibration_summary = detector.fit(
            normal_bank_trajectories=bank_trajectories,
            calibration_trajectories=calibration_trajectories,
        )

        clean_samples = _build_metric_samples(
            detector=detector,
            trajectories=calibration_trajectories,
            has_labels=False,
        )

        suboptimal_refs, suboptimal_split_summary = _list_suboptimal_demo_refs(cfg)
        labeled_trajectories, encode_summary = _encode_suboptimal_refs(
            refs=suboptimal_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            horizon=int(detector.extractor.action_horizon),
            batch_size=int(cfg.data.encode_demo_batch_size),
        )
        train_labeled = [item for item in labeled_trajectories if item.split == "train"]
        eval_labeled = [item for item in labeled_trajectories if item.split == "eval"]
        if not train_labeled:
            raise RuntimeError("No suboptimal training trajectories remain after encoding/filtering.")
        if not eval_labeled:
            raise RuntimeError("No suboptimal eval trajectories remain after encoding/filtering.")

        train_samples = _build_metric_samples(
            detector=detector,
            trajectories=train_labeled,
            has_labels=True,
        )
        eval_samples = _build_metric_samples(
            detector=detector,
            trajectories=eval_labeled,
            has_labels=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.detector.device).startswith("cuda") else "cpu")
        best_candidate, candidate_results = _search_sparse_weights(
            detector=detector,
            clean_samples=clean_samples,
            train_samples=train_samples,
            fit_lr=float(cfg.suboptimal.fit_lr),
            fit_steps=int(cfg.suboptimal.fit_steps),
            epsilon=float(cfg.suboptimal.selection_train_f1_epsilon),
            device=device,
        )
        final_weights = dict(best_candidate["weights"])
        final_threshold = float(best_candidate["train_summary"]["threshold"])
        train_summary = _evaluate_labeled_samples(detector, train_samples, final_weights, final_threshold)
        eval_summary = _evaluate_labeled_samples(detector, eval_samples, final_weights, final_threshold)
        clean_far = _clean_false_alarm_rate(detector, clean_samples, final_weights, final_threshold)

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        save_name = str(getattr(cfg, "save_name", "") or "").strip()
        if save_name in {"", "None", "null"}:
            save_name = f"lpb_new_metric_weights_{_now_tag()}.json"
        if not save_name.endswith(".json"):
            save_name = f"{save_name}.json"
        save_path = os.path.join(save_dir, save_name)

        artifact = {
            "detector_name": detector.name,
            "source_lpb_ckpt": str(cfg.model.lpb_ckpt),
            "active_metrics": list(best_candidate["active_metrics"]),
            "weights": dict(final_weights),
            "train_summary": dict(train_summary),
            "eval_summary": dict(eval_summary),
            "suboptimal_tasks": [normalize_task_name(str(task_name)) for task_name in list(cfg.suboptimal.tasks)],
            "split": {
                "train_count_per_task": int(cfg.suboptimal.train_count_per_task),
                "eval_count_per_task": int(cfg.suboptimal.eval_count_per_task),
            },
            "seed": int(cfg.suboptimal.seed),
            "selection_summary": {
                "selection_train_f1_epsilon": float(cfg.suboptimal.selection_train_f1_epsilon),
                "clean_false_alarm_rate": float(clean_far),
                "threshold": float(final_threshold),
                "candidate_results": candidate_results,
            },
            "clean_calibration_summary": {
                "threshold_init": float(calibration_summary.threshold if calibration_summary.threshold is not None else np.nan),
                "score_scales": dict(detector._score_scales),
                "num_bank_trajectories": int(len(bank_trajectories)),
                "num_calibration_trajectories": int(len(calibration_trajectories)),
            },
            "suboptimal_split_summary": suboptimal_split_summary,
            "suboptimal_encode_summary": encode_summary,
            "clean_split_summary": split_summary,
        }

        with open(save_path, "w", encoding="utf-8") as file_handle:
            json.dump(artifact, file_handle, indent=2, sort_keys=True, default=_json_default)

        print(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default))
        print(f"[lpb_new analyse] Saved metric weight artifact to: {save_path}")
    finally:
        encoder.close()


@hydra.main(version_base="1.2", config_path="./config", config_name="analyse")
def main(cfg: DictConfig) -> None:
    run_analyse(cfg)


if __name__ == "__main__":
    main()
