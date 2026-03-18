from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore[no-redef]
        return iterable

try:
    from robosuite.discriminator.float_dino.dino_v2_latent import DinoV2ImageLatentExtractor
    from robosuite.discriminator.float_dino.float_data import fail_prefix_labels, load_policy_trajectories
    from robosuite.discriminator.float_dino.float_official import OfficialFloatOfflineEvaluator
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name not in {"mujoco", "robosuite"}:
        raise
    import sys

    pkg_root = Path(__file__).resolve().parents[3]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from discriminator.float_dino.dino_v2_latent import DinoV2ImageLatentExtractor
    from discriminator.float_dino.float_data import fail_prefix_labels, load_policy_trajectories
    from discriminator.float_dino.float_official import OfficialFloatOfflineEvaluator

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class EvalTrajectoryRecord:
    trajectory_id: int
    source_file: str
    demo_key: str
    num_frames: int
    num_latent_steps: int
    first_pred_failure_frame: Optional[int]
    pred_failure_frame_count: int
    gt_failure_frame_count: int
    video_path: str


def _resolve_torch_device(preferred: str) -> str:
    import torch

    pref = str(preferred).lower()
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("policy-device is CUDA but torch.cuda.is_available() is False.")
    return preferred


def _compute_threshold(scores: list[float], delta: float) -> float:
    if not scores:
        raise ValueError("Cannot compute threshold from empty calibration scores")
    if delta < 0 or delta > 100:
        raise ValueError(f"delta must be in [0,100], got {delta}")
    q = 100.0 * (1.0 - float(delta) / 100.0)
    return float(np.percentile(np.asarray(scores, dtype=np.float64), q=q))


def _split_fail_train_eval(fail_rollouts: list, num_vis: int, seed: int) -> tuple[list, list]:
    if not fail_rollouts:
        raise ValueError("No fail rollouts provided")
    if num_vis <= 0:
        raise ValueError(f"num_vis must be >=1, got {num_vis}")
    if len(fail_rollouts) < 2:
        raise ValueError(f"Need at least 2 fail trajectories, got {len(fail_rollouts)}")

    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(fail_rollouts))
    rng.shuffle(indices)
    k = min(int(num_vis), len(fail_rollouts) - 1)
    eval_ids = set(indices[:k].tolist())
    fail_eval = [x for i, x in enumerate(fail_rollouts) if i in eval_ids]
    fail_train = [x for i, x in enumerate(fail_rollouts) if i not in eval_ids]
    return fail_train, fail_eval


def _map_step_values_to_frames(
    step_values: np.ndarray,
    sampled_indices: np.ndarray,
    num_frames: int,
    default_value: float,
) -> np.ndarray:
    values = np.full(num_frames, default_value, dtype=np.float64)
    sampled = np.asarray(sampled_indices, dtype=np.int64).reshape(-1)
    steps = np.asarray(step_values).reshape(-1)
    if sampled.size != steps.size:
        raise ValueError(f"sampled_indices and step_values must have same length: {sampled.size} vs {steps.size}")
    if sampled.size == 0:
        return values

    for i in range(sampled.size):
        start = int(sampled[i])
        end = int(sampled[i + 1]) if (i + 1) < sampled.size else num_frames
        start = int(np.clip(start, 0, num_frames))
        end = int(np.clip(end, start, num_frames))
        values[start:end] = float(steps[i])
    return values


def _draw_overlay(
    frame_rgb: np.ndarray,
    frame_id: int,
    pred_fail_flag: bool,
    gt_fail_flag: bool,
    cumulative_cost: float,
    threshold: float,
    label_hint: int,
) -> np.ndarray:
    frame = np.asarray(frame_rgb, dtype=np.uint8).copy()

    if cv2 is None:
        if gt_fail_flag:
            frame[:5, :, 1] = 255
            frame[-5:, :, 1] = 255
            frame[:, :5, 1] = 255
            frame[:, -5:, 1] = 255
        if pred_fail_flag:
            frame[:8, :, 0] = 255
            frame[-8:, :, 0] = 255
            frame[:, :8, 0] = 255
            frame[:, -8:, 0] = 255
        return frame

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    if gt_fail_flag:
        cv2.rectangle(bgr, (8, 8), (w - 9, h - 9), (0, 255, 255), 2)
    if pred_fail_flag:
        cv2.rectangle(bgr, (3, 3), (w - 4, h - 4), (0, 0, 255), 5)
        cv2.putText(bgr, "FAILURE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    text_1 = f"frame={frame_id + 1} label_hint={label_hint}"
    text_2 = f"lambda={cumulative_cost:.5f} threshold={threshold:.5f}"
    cv2.putText(bgr, text_1, (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, text_2, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fix_robosuite_frame_orientation(frame_rgb: np.ndarray, flip_vertical: bool) -> np.ndarray:
    return np.flipud(frame_rgb) if bool(flip_vertical) else frame_rgb


def _adapt_delta_on_fail_train(
    evaluator: OfficialFloatOfflineEvaluator,
    fail_train_embeddings: list[np.ndarray],
    calibration_scores: list[float],
    delta: float,
    delta_step: float,
    fail_tail_ratio: float,
    show_progress: bool = True,
) -> tuple[float, float]:
    threshold = _compute_threshold(calibration_scores, delta)
    if not fail_train_embeddings:
        return delta, threshold

    for emb in tqdm(fail_train_embeddings, desc="Adaptive delta on fail-train", leave=False, disable=not show_progress):
        out = evaluator.run_episode(rollout_embeddings=emb, threshold=threshold)
        labels = fail_prefix_labels(length=int(out.step_failure_flags.shape[0]), fail_tail_ratio=fail_tail_ratio)
        preds = out.step_failure_flags.astype(np.int64)

        for i in range(labels.shape[0]):
            label = int(labels[i])
            pred = int(preds[i])
            if label == 1 and pred == 0:
                delta += delta_step
            elif label == 0 and pred == 1:
                delta -= delta_step
            delta = float(np.clip(delta, 0.0, 100.0))
            threshold = _compute_threshold(calibration_scores, delta)

    return float(delta), float(threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample fail trajectories, run FLOAT-DINO evaluation, and export failure-marked videos."
    )
    parser.add_argument("--expert-dir", type=str, default="data/PandaPickPlaceCan/expert_recover")
    parser.add_argument("--fail-dir", type=str, default="data/PandaPickPlaceCan/fail_rollout")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--policy-device", type=str, default="cuda")
    parser.add_argument("--pretrained-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="vit_base")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--latent-batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-register-tokens", type=int, default=4)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--num-expert-candidates", type=int, default=20)
    parser.add_argument("--use-similarity-cost", action="store_true")
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--delta-step", type=float, default=1.0)
    parser.add_argument("--adaptive-delta-on-fail-train", action="store_true")
    parser.add_argument("--fail-tail-ratio", type=float, default=0.2)
    parser.add_argument("--num-vis", type=int, default=2)
    parser.add_argument("--max-expert-trajectories", type=int, default=0)
    parser.add_argument("--max-fail-trajectories", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-vertical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=str, default="outputs/float_dino_visualize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    expert_rollouts = load_policy_trajectories(
        data_dir=os.path.abspath(args.expert_dir),
        camera_name=args.camera_name,
        max_trajectories=(None if int(args.max_expert_trajectories) <= 0 else int(args.max_expert_trajectories)),
    )
    fail_rollouts = load_policy_trajectories(
        data_dir=os.path.abspath(args.fail_dir),
        camera_name=args.camera_name,
        max_trajectories=(None if int(args.max_fail_trajectories) <= 0 else int(args.max_fail_trajectories)),
    )
    fail_train, fail_eval = _split_fail_train_eval(fail_rollouts=fail_rollouts, num_vis=int(args.num_vis), seed=int(args.seed))

    extractor = DinoV2ImageLatentExtractor(
        camera_name=args.camera_name,
        image_size=int(args.image_size),
        device=_resolve_torch_device(args.policy_device),
        pretrained_path=args.pretrained_path,
        model_name=args.model_name,
        batch_size=int(args.latent_batch_size),
        patch_size=int(args.patch_size),
        num_register_tokens=int(args.num_register_tokens),
        normalize_embedding=bool(args.normalize_embedding),
    )

    eval_assets: list[tuple] = []
    try:
        expert_embeddings = [extractor.encode_trajectory(t) for t in tqdm(expert_rollouts, desc="Encoding expert", disable=not bool(args.progress))]
        fail_train_embeddings = [extractor.encode_trajectory(t) for t in tqdm(fail_train, desc="Encoding fail-train", disable=not bool(args.progress))]
        for traj in tqdm(fail_eval, desc="Encoding fail-eval", disable=not bool(args.progress)):
            emb, sampled_idx = extractor.encode_trajectory_with_indices(traj)
            eval_assets.append((traj, emb, sampled_idx))
    finally:
        extractor.close()

    max_steps = max(int(x.shape[0]) for x in expert_embeddings)
    evaluator = OfficialFloatOfflineEvaluator(
        expert_embeddings=expert_embeddings,
        sinkhorn_reg=float(args.sinkhorn_reg),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        num_expert_candidates=int(args.num_expert_candidates),
        max_steps=max_steps,
        use_similarity_cost=bool(args.use_similarity_cost),
    )

    calibration_scores = [evaluator.episode_score(x) for x in expert_embeddings]
    delta_before = float(args.delta)
    threshold = _compute_threshold(scores=calibration_scores, delta=delta_before)
    delta_after = delta_before
    if bool(args.adaptive_delta_on_fail_train):
        delta_after, threshold = _adapt_delta_on_fail_train(
            evaluator=evaluator,
            fail_train_embeddings=fail_train_embeddings,
            calibration_scores=calibration_scores,
            delta=delta_before,
            delta_step=float(args.delta_step),
            fail_tail_ratio=float(args.fail_tail_ratio),
            show_progress=bool(args.progress),
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.abspath(args.output_dir), f"run_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)

    records: list[EvalTrajectoryRecord] = []
    for traj_idx, (traj, emb, sampled_idx) in enumerate(tqdm(eval_assets, desc="Rendering videos", disable=not bool(args.progress))):
        out = evaluator.run_episode(rollout_embeddings=emb, threshold=threshold)
        n_steps = int(out.step_failure_flags.shape[0])
        labels = fail_prefix_labels(length=n_steps, fail_tail_ratio=float(args.fail_tail_ratio))
        sampled_idx = np.asarray(sampled_idx[:n_steps], dtype=np.int64)

        frame_flags = _map_step_values_to_frames(out.step_failure_flags, sampled_idx, traj.images.shape[0], 0.0).astype(np.int64)
        frame_costs = _map_step_values_to_frames(out.cumulative_costs, sampled_idx, traj.images.shape[0], 0.0)
        frame_labels = _map_step_values_to_frames(labels, sampled_idx, traj.images.shape[0], 0.0).astype(np.int64)

        src = Path(str(traj.meta.get("file_path", "unknown")))
        demo_key = str(traj.meta.get("demo_key", f"traj_{traj_idx}"))
        video_path = os.path.join(run_dir, f"traj{traj_idx:02d}_{src.stem}_{demo_key}.mp4")

        writer = imageio.get_writer(video_path, fps=int(args.fps))
        try:
            for frame_id in range(traj.images.shape[0]):
                frame = _fix_robosuite_frame_orientation(traj.images[frame_id], bool(args.flip_vertical))
                frame = _draw_overlay(
                    frame_rgb=frame,
                    frame_id=frame_id,
                    pred_fail_flag=bool(frame_flags[frame_id]),
                    gt_fail_flag=bool(frame_labels[frame_id]),
                    cumulative_cost=float(frame_costs[frame_id]),
                    threshold=float(threshold),
                    label_hint=int(frame_labels[frame_id]),
                )
                writer.append_data(frame)
        finally:
            writer.close()

        first_pred_failure = np.where(frame_flags == 1)[0]
        records.append(
            EvalTrajectoryRecord(
                trajectory_id=traj_idx,
                source_file=str(traj.meta.get("file_path", "")),
                demo_key=demo_key,
                num_frames=int(traj.images.shape[0]),
                num_latent_steps=int(n_steps),
                first_pred_failure_frame=(int(first_pred_failure[0] + 1) if first_pred_failure.size > 0 else None),
                pred_failure_frame_count=int(np.sum(frame_flags)),
                gt_failure_frame_count=int(np.sum(frame_labels)),
                video_path=video_path,
            )
        )

    summary = {
        "timestamp": run_stamp,
        "input": {
            "expert_dir": os.path.abspath(args.expert_dir),
            "fail_dir": os.path.abspath(args.fail_dir),
            "camera_name": args.camera_name,
            "pretrained_path": args.pretrained_path,
            "model_name": args.model_name,
        },
        "split": {
            "fail_total": len(fail_rollouts),
            "fail_train": len(fail_train),
            "fail_eval": len(fail_eval),
            "num_vis": int(args.num_vis),
            "seed": int(args.seed),
        },
        "float_runtime": {
            "num_expert_candidates": int(args.num_expert_candidates),
            "sinkhorn_reg": float(args.sinkhorn_reg),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "policy_type": "dino_v2",
            "policy_latent_source": "dino_v2_cls_token",
            "policy_device": args.policy_device,
            "image_size": int(args.image_size),
            "loaded_params": int(extractor.loaded_params),
        },
        "calibration": {
            "mode": "expert_only",
            "num_scores": len(calibration_scores),
            "score_mean": float(np.mean(calibration_scores)),
            "score_std": float(np.std(calibration_scores)),
            "delta_before": float(delta_before),
            "delta_after": float(delta_after),
            "threshold": float(threshold),
            "adaptive_delta_on_fail_train": bool(args.adaptive_delta_on_fail_train),
        },
        "videos": [
            {
                "trajectory_id": r.trajectory_id,
                "source_file": r.source_file,
                "demo_key": r.demo_key,
                "num_frames": r.num_frames,
                "num_latent_steps": r.num_latent_steps,
                "first_pred_failure_frame": r.first_pred_failure_frame,
                "pred_failure_frame_count": r.pred_failure_frame_count,
                "gt_failure_frame_count": r.gt_failure_frame_count,
                "video_path": r.video_path,
            }
            for r in records
        ],
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
