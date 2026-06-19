"""Run the BCE-WAM discriminator on the new robosuite benchmark layout.

Evaluation uses ``data/<task>/fail_rollout-val-labeled`` and
``data/<task>/success_rollout-val``. The BCE failure bank is built from
``data/<task>/fail_rollout-labeled``.

Hard constraint: training and evaluation are separated -- ``fit_on_benchmark``
does no evaluation, and ``bench.evaluate`` is invoked here *after* the head is
trained.
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
    discover_success_rollouts,
)

from robosuite.discriminator.dyn_disc.adapters.bce import BCEBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root containing data/<task>/<split> directories.")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", type=str, default="success_rollout-val",
                        help="Eval success split (benchmark test only).")
    parser.add_argument("--success-train-split", type=str, default="success_rollout",
                        help="Train success split for BCE positives + calibration. "
                             "Kept disjoint from --success-split.")
    parser.add_argument("--fail-train-split", type=str, default="fail_rollout-labeled")
    parser.add_argument("--train-max-success-per-task", type=int, default=None,
                        help="Max success trajectories per task from --success-train-split.")
    parser.add_argument("--fail-root", type=str, default=None,
                        help="Deprecated; use --data-root/--fail-split.")
    parser.add_argument("--success-root", type=str, default=None,
                        help="Deprecated; use --data-root/--success-split.")
    parser.add_argument("--fail-train-root", type=str, default=None,
                        help="Deprecated; use --data-root/--fail-train-split.")
    parser.add_argument("--success-cache-root", type=str, default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--metadata-cache-root", type=str, default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--cache-camera-names", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-ckpt-dir", type=str, default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", type=str, default=None)

    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--knn-feature-source", type=str, default="transformer",
                        choices=["encoder", "transformer"])
    parser.add_argument("--knn-transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")

    # BCE knobs.
    parser.add_argument("--max-expert-other-ratio", type=float, default=1.0,
                        help="Cap |D_e| <= ratio * |D_o| by random subsampling. "
                             "Use a value <= 0 to disable.")
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--calib-mode",
        type=str,
        default="two_class_youden",
        choices=["success_percentile", "two_class_youden"],
        help="Per-task threshold calibration. success_percentile follows the "
             "original delta-percentile rule on success-calib frames. "
             "two_class_youden (default) overrides tau with argmax(TPR-FPR) on "
             "(success-calib, fail-suffix) failure scores. Either choice leaves "
             "step_scores untouched, so AUROC/AUPRC are unaffected.",
    )

    # Failure-pool selection.
    parser.add_argument("--fail-bank-per-task", type=int, default=25,
                        help="Target number of GT-labeled failure trajectories per task to use "
                             "as D_o source. If fewer are available, all are used.")
    parser.add_argument(
        "--fail-bank-ids-json",
        type=str,
        default=None,
        help="Optional JSON path mapping task_name -> [video_id, ...] to override auto-selection.",
    )
    parser.add_argument("--fail-calib-per-task", type=int, default=0,
                        help="Optional extra failure trajectories reserved for calib (rarely used).")
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

    If a task has fewer GT-labeled failure trajectories than
    ``fail_bank_per_task``, all available are used (no error).
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
                    f"fail-train-root. Check --fail-train-root and --tasks."
                )
            if n_take < int(fail_bank_per_task):
                print(
                    f"[robosuite][bce] task={task}: only {n_take} GT failure trajectories "
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
    print("[robosuite][bce] building FailureBenchmark...", flush=True)
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
        f"[robosuite][bce] eval set: {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={eval_tasks}",
        flush=True,
    )

    # Discover the disjoint TRAIN success pool (BCE positives + calibration).
    train_max_success = (
        None
        if args.train_max_success_per_task is None or int(args.train_max_success_per_task) <= 0
        else int(args.train_max_success_per_task)
    )
    train_success_trajs = discover_success_rollouts(
        data_root=args.data_root,
        tasks=eval_tasks,
        split=args.success_train_split,
        max_success_per_task=train_max_success,
    )
    train_by_task: Dict[str, int] = {}
    for t in train_success_trajs:
        train_by_task[str(t.task_name)] = train_by_task.get(str(t.task_name), 0) + 1
    print(
        f"[robosuite][bce] train success from {args.success_train_split}: "
        f"{len(train_success_trajs)} trajectories "
        + ", ".join(f"{k}={v}" for k, v in sorted(train_by_task.items())),
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
    # video_id disjointness against eval set (defence-in-depth: train root should
    # already be disjoint from the eval fail root by construction).
    bank_pool = [t for t in all_fail if str(t.video_id) not in eval_fail_keys]
    dropped = len(all_fail) - len(bank_pool)
    print(
        f"[robosuite][bce] GT-failure discovery from {args.fail_train_split}: "
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

    # Defence-in-depth disjoint check (adapter also asserts inside fit_on_benchmark).
    bank_keys = {str(t.video_id) for t in fail_bank_trajs}
    calib_keys = {str(t.video_id) for t in fail_calib_trajs}
    overlap = sorted((bank_keys | calib_keys) & eval_fail_keys)
    if overlap:
        raise RuntimeError(
            f"Disjointness invariant violated: fail-bank/calib video_ids appear in eval set: {overlap}"
        )

    bank_by_task: Dict[str, List[str]] = {}
    for t in fail_bank_trajs:
        bank_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    calib_by_task: Dict[str, List[str]] = {}
    for t in fail_calib_trajs:
        calib_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    print(
        "[robosuite][bce] fail bank sizes per task: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(bank_by_task.items())),
        flush=True,
    )

    max_eo_ratio: Optional[float] = (
        None if float(args.max_expert_other_ratio) <= 0.0 else float(args.max_expert_other_ratio)
    )
    discriminator = BCEBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),        # freezed WAM model
        fail_bank_trajectories=fail_bank_trajs,
        fail_calib_trajectories=fail_calib_trajs,
        max_expert_other_ratio=max_eo_ratio,
        head_hidden=int(args.head_hidden),
        head_layers=int(args.head_layers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        calib_mode=str(args.calib_mode),
        save_ckpt_dir=str(args.save_ckpt_dir) if args.save_ckpt_dir else None,
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
        # fit bce model
        print("[robosuite][bce] training BCE head (GT split; no eval during training)...",
              flush=True)
        discriminator.fit_on_benchmark(
            trajs,
            train_success_trajectories=train_success_trajs,
        )

        print("[robosuite][bce] training complete; running bench.evaluate(...)", flush=True)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        calib_summary = discriminator.calibration_summary()
        print("[robosuite][bce] calibration summary:", calib_summary)

        if args.save_json:
            out_dir = os.path.dirname(os.path.abspath(args.save_json))
            os.makedirs(out_dir, exist_ok=True)
            result.save_json(args.save_json)
            manifest_path = os.path.join(out_dir, "fail_bank_manifest.json")
            manifest = {
                "labeling": "gt_failure_split",
                "loss": "bce",
                "data_root": str(args.data_root),
                "fail_train_split": str(args.fail_train_split),
                "fail_eval_split": str(args.fail_split),
                "success_eval_split": str(args.success_split),
                "success_train_split": str(args.success_train_split),
                "train_max_success_per_task": train_max_success,
                "max_expert_other_ratio": (
                    None if float(args.max_expert_other_ratio) <= 0.0
                    else float(args.max_expert_other_ratio)
                ),
                "fail_bank": bank_by_task,
                "fail_calib": calib_by_task,
                "fail_bank_per_task": int(args.fail_bank_per_task),
                "fail_calib_per_task": int(args.fail_calib_per_task),
                "eval_fail_video_ids": sorted(eval_fail_keys),
                "delta": float(args.delta),
                "calib_mode": str(args.calib_mode),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "head_hidden": int(args.head_hidden),
                "head_layers": int(args.head_layers),
                "knn_feature_source": str(args.knn_feature_source),
                "knn_transformer_layer": int(args.knn_transformer_layer),
            }
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            print(f"[robosuite][bce] wrote {args.save_json}")
            print(f"[robosuite][bce] wrote {manifest_path}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
