"""Run the two-bank KNN discriminator through the robosuite benchmark API.

Evaluation uses ``data/<task>/fail_rollout-val-labeled`` (failure eval) and
``data/<task>/success_rollout-val`` (success eval + success bank). The GT
failure bank is discovered from ``data/<task>/fail_rollout-labeled`` and is
hard-asserted ``video_id``-disjoint from the failure eval set. Each failure-bank
trajectory is sliced from ``first_gt_failure_frame()`` onward by the adapter.

Example:
    python -m robosuite.discriminator.dyn_disc.sim_benchmark_two_bank \
        --model-ckpt /abs/path/checkpoint/model_50.pth \
        --data-root data \
        --tasks PickPlaceCereal \
        --fail-bank-per-task 25 \
        --score-mode difference --alpha 1.0 \
        --calib-mode success_percentile \
        --save-json /tmp/dyn_disc_two_bank_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from benchmark.core import EvalConfig
from robosuite.discriminator.utils.robosuite_benchmark import (
    FailureBenchmark,
    RobosuiteBenchmarkTrajectory,
    discover_failure_bank,
)

from robosuite.discriminator.dyn_disc.adapters.two_bank import TwoBankBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True,
                        help="Path to a dyn_disc dynamics checkpoint.")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root containing data/<task>/<split> directories.")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled",
                        help="Failure eval split.")
    parser.add_argument("--success-split", type=str, default="success_rollout-val",
                        help="Success eval split (also builds the success bank).")
    parser.add_argument("--fail-train-split", type=str, default="fail_rollout-labeled",
                        help="GT-labeled failure split used to build the disjoint failure bank.")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None,
                        help="State indices to slice as proprio (default: use all).")
    parser.add_argument("--camera-to-view", type=str, default=None,
                        help="Comma-separated camera:view pairs, e.g. agentview:agentview")

    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=10.0,
                        help="Percentile-based false-alarm budget (0-100).")
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--knn-feature-source", type=str, default="transformer",
                        choices=["encoder", "transformer"])
    parser.add_argument("--knn-transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")

    # Two-bank knobs.
    parser.add_argument("--fail-bank-per-task", type=int, default=25,
                        help="Target number of GT-labeled failure trajectories per task to "
                             "use as the failure bank. If fewer are available, all are used.")
    parser.add_argument("--fail-bank-last-k", type=int, default=60,
                        help="Frames kept per failure-bank trajectory starting at first_gt_failure_frame.")
    parser.add_argument(
        "--fail-bank-ids-json",
        type=str,
        default=None,
        help="Optional JSON path mapping task_name -> [video_id, ...] to override auto-selection.",
    )
    parser.add_argument("--fail-calib-per-task", type=int, default=0,
                        help="Optional extra failure trajectories reserved for calib "
                             "(used by --calib-mode two_class_youden).")
    parser.add_argument(
        "--score-mode",
        type=str,
        default="difference",
        choices=["difference", "ratio", "dsucc_only"],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--calib-mode",
        type=str,
        default="success_percentile",
        choices=["success_percentile", "two_class_youden"],
    )
    return parser.parse_args()


def _parse_camera_to_view(value: Optional[str]):
    if not value:
        return None
    out = {}
    for chunk in value.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def _select_bank_and_calib(
    bank_pool_by_task: Dict[str, List[RobosuiteBenchmarkTrajectory]],
    eval_tasks: Sequence[str],
    fail_bank_per_task: int,
    fail_calib_per_task: int,
    fail_bank_ids_override: Optional[Dict[str, List[str]]],
) -> tuple[List[RobosuiteBenchmarkTrajectory], List[RobosuiteBenchmarkTrajectory]]:
    """Deterministically pick fail-bank + fail-calib trajectories per task.

    Picks the first N by sorted video_id from the disjoint bank pool. If a task
    has fewer GT-labeled failures than ``fail_bank_per_task``, all available are
    used (no error).
    """
    bank_out: List[RobosuiteBenchmarkTrajectory] = []
    calib_out: List[RobosuiteBenchmarkTrajectory] = []

    for task in sorted(set(eval_tasks)):
        pool = sorted(bank_pool_by_task.get(task, []), key=lambda t: str(t.video_id))
        pool_by_id = {str(t.video_id): t for t in pool}

        if fail_bank_ids_override and task in fail_bank_ids_override:
            chosen_ids = list(fail_bank_ids_override[task])
            missing = [v for v in chosen_ids if v not in pool_by_id]
            if missing:
                raise RuntimeError(
                    f"Task {task!r}: --fail-bank-ids-json names video_ids not present in the "
                    f"disjoint bank pool: {missing}"
                )
            bank_trajs = [pool_by_id[v] for v in chosen_ids]
            remaining = [t for t in pool if str(t.video_id) not in set(chosen_ids)]
        else:
            n_take = min(len(pool), int(fail_bank_per_task))
            if n_take == 0:
                raise RuntimeError(
                    f"Task {task!r}: zero GT-labeled failure trajectories available in "
                    f"fail-train-split. Check --fail-train-split and --tasks."
                )
            if n_take < int(fail_bank_per_task):
                print(
                    f"[robosuite][two_bank] task={task}: only {n_take} GT failure trajectories "
                    f"available (< --fail-bank-per-task={fail_bank_per_task}); using all.",
                    flush=True,
                )
            bank_trajs = pool[:n_take]
            remaining = pool[n_take:]

        bank_out.extend(bank_trajs)

        if fail_calib_per_task > 0:
            if len(remaining) < fail_calib_per_task:
                raise RuntimeError(
                    f"Task {task!r}: only {len(remaining)} failure trajectories remain after "
                    f"taking the bank, but --fail-calib-per-task={fail_calib_per_task}."
                )
            calib_out.extend(remaining[:fail_calib_per_task])

    return bank_out, calib_out


def main() -> None:
    args = _parse_args()
    print("[robosuite][two_bank] building FailureBenchmark...", flush=True)
    bench = FailureBenchmark(
        data_root=args.data_root,
        tasks=args.tasks,
        fail_split=args.fail_split,
        success_split=args.success_split,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    trajs = bench.trajectories()
    eval_fail_keys = {str(t.video_id) for t in trajs if bool(t.is_failure)}
    eval_tasks = sorted({str(t.task_name) for t in trajs})
    n_fail = len(eval_fail_keys)
    n_succ = len(trajs) - n_fail
    print(
        f"[robosuite][two_bank] eval set: {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={eval_tasks}",
        flush=True,
    )

    # Discover ALL GT-labeled failure trajectories from the train labeled split.
    all_fail = discover_failure_bank(
        data_root=args.data_root,
        tasks=args.tasks,
        split=args.fail_train_split,
        max_fail_per_task=None,
    )
    all_fail = [t for t in all_fail if bool(t.is_failure)]
    # video_id disjointness against eval set (defence-in-depth: train split should
    # already be disjoint from the eval fail split by construction).
    bank_pool = [t for t in all_fail if str(t.video_id) not in eval_fail_keys]
    dropped = len(all_fail) - len(bank_pool)
    print(
        f"[robosuite][two_bank] GT-failure discovery from {args.fail_train_split}: "
        f"all_fail={len(all_fail)} eval_fail={len(eval_fail_keys)} "
        f"bank_pool={len(bank_pool)} (dropped {dropped} as video_id overlap)",
        flush=True,
    )

    bank_pool_by_task: Dict[str, List[RobosuiteBenchmarkTrajectory]] = {}
    for t in bank_pool:
        bank_pool_by_task.setdefault(str(t.task_name), []).append(t)

    fail_bank_ids_override = None
    if args.fail_bank_ids_json:
        with open(args.fail_bank_ids_json, "r") as fh:
            fail_bank_ids_override = json.load(fh)
        if not isinstance(fail_bank_ids_override, dict):
            raise RuntimeError(
                "--fail-bank-ids-json must be a JSON object mapping task -> [video_id, ...]"
            )

    fail_bank_trajs, fail_calib_trajs = _select_bank_and_calib(
        bank_pool_by_task=bank_pool_by_task,
        eval_tasks=eval_tasks,
        fail_bank_per_task=int(args.fail_bank_per_task),
        fail_calib_per_task=int(args.fail_calib_per_task),
        fail_bank_ids_override=fail_bank_ids_override,
    )

    # Hard-assert disjointness (defence-in-depth; the adapter also checks).
    bank_keys = {str(t.video_id) for t in fail_bank_trajs}
    calib_keys = {str(t.video_id) for t in fail_calib_trajs}
    overlap = sorted((bank_keys | calib_keys) & eval_fail_keys)
    if overlap:
        raise RuntimeError(
            f"Disjointness invariant violated: fail-bank/calib video_ids appear in eval set: {overlap}"
        )
    bank_calib_overlap = sorted(bank_keys & calib_keys)
    if bank_calib_overlap:
        raise RuntimeError(
            f"Disjointness invariant violated: fail-bank and fail-calib share video_ids: {bank_calib_overlap}"
        )

    bank_by_task: Dict[str, List[str]] = {}
    for t in fail_bank_trajs:
        bank_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    calib_by_task: Dict[str, List[str]] = {}
    for t in fail_calib_trajs:
        calib_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    print(
        "[robosuite][two_bank] DISJOINTNESS OK | fail bank sizes per task: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(bank_by_task.items())),
        flush=True,
    )
    if fail_calib_trajs:
        print(
            "[robosuite][two_bank] fail calib sizes per task: "
            + ", ".join(f"{k}={len(v)}" for k, v in sorted(calib_by_task.items())),
            flush=True,
        )

    discriminator = TwoBankBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        fail_bank_trajectories=fail_bank_trajs,
        fail_calib_trajectories=fail_calib_trajs,
        fail_bank_last_k=int(args.fail_bank_last_k),
        alpha=float(args.alpha),
        score_mode=str(args.score_mode),
        calib_mode=str(args.calib_mode),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        visual_weight=float(args.visual_weight),
        proprio_weight=float(args.proprio_weight),
        action_weight=float(args.action_weight),
        delta=float(args.delta),
        knn_chunk_size=int(args.knn_chunk_size),
        feature_source=str(args.knn_feature_source),
        transformer_layer=int(args.knn_transformer_layer),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        print("[robosuite][two_bank] fitting two-bank detector...", flush=True)
        discriminator.fit_on_benchmark(trajs)
        print("[robosuite][two_bank] fit done; starting evaluate()...", flush=True)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        calib_summary = discriminator.calibration_summary()
        print("[robosuite][two_bank] calibration summary:", calib_summary)

        if args.save_json:
            out_dir = os.path.dirname(os.path.abspath(args.save_json))
            os.makedirs(out_dir, exist_ok=True)
            result.save_json(args.save_json)
            manifest_path = os.path.join(out_dir, "fail_bank_manifest.json")
            manifest = {
                "labeling": "gt_failure_split",
                "method": "two_bank_knn",
                "data_root": str(args.data_root),
                "fail_train_split": str(args.fail_train_split),
                "fail_eval_split": str(args.fail_split),
                "success_split": str(args.success_split),
                "fail_bank": bank_by_task,
                "fail_calib": calib_by_task,
                "fail_bank_per_task": int(args.fail_bank_per_task),
                "fail_calib_per_task": int(args.fail_calib_per_task),
                "fail_bank_last_k": int(args.fail_bank_last_k),
                "fail_bank_index_ranges": calib_summary.get("fail_bank_index_ranges", {}),
                "eval_fail_video_ids": sorted(eval_fail_keys),
                "score_mode": str(args.score_mode),
                "alpha": float(args.alpha),
                "calib_mode": str(args.calib_mode),
                "delta": float(args.delta),
                "knn_feature_source": str(args.knn_feature_source),
                "knn_transformer_layer": int(args.knn_transformer_layer),
            }
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            print(f"[robosuite][two_bank] wrote {args.save_json}")
            print(f"[robosuite][two_bank] wrote {manifest_path}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
