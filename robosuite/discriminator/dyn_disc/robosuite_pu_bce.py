"""Run the nnPU (PU-BCE) discriminator on the robosuite benchmark layout.

Evaluation uses ``data/<task>/fail_rollout-val-labeled`` and
``data/<task>/success_rollout-val``. Both training and eval success trajectories
keep only pre-done frames from the per-step ``is_success`` label (success
rollouts may continue after task completion). The unlabeled failure pool is
built from ``data/<task>/fail_rollout`` -- each failure trajectory is used as a
WHOLE (no ``first_gt_failure_frame`` timing; GT failure timing is forbidden in
this branch).

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
    discover_success_rollouts,
    discover_unlabeled_failures,
)

from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root containing data/<task>/<split> directories.")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", type=str, default="success_rollout-val",
                        help="Eval success split (benchmark test only).")
    parser.add_argument("--success-train-split", type=str, default="success_rollout",
                        help="Train success split for nnPU positives + calibration.")
    parser.add_argument("--fail-train-split", type=str, default="fail_rollout",
                        help="Split that supplies the UNLABELED failure pool "
                             "(whole trajectories; no GT timing or mask required).")
    parser.add_argument("--cache-camera-names", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-ckpt-dir", type=str, default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None,
                        help="Eval failure cap per task (benchmark test set only).")
    parser.add_argument("--max-success-per-task", type=int, default=None,
                        help="Eval success cap per task (benchmark test set only).")
    parser.add_argument("--train-max-success-per-task", type=int, default=None,
                        help="Max success trajectories per task from --success-train-split "
                             "(nnPU positives + calibration).")
    parser.add_argument("--train-max-fail-per-task", type=int, default=None,
                        help="Max failure trajectories per task from --fail-train-split "
                             "(unlabeled pool). Overrides --unlabeled-per-task when set.")
    parser.add_argument("--train-max-per-task", type=int, default=None,
                        help="Shorthand for setting both --train-max-success-per-task and "
                             "--train-max-fail-per-task when those are omitted.")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", type=str, default=None)

    parser.add_argument("--delta", type=float, default=10.0,
                        help="False-alarm budget %% for success_percentile calib: "
                             "tau = percentile(success-calib failure scores, 100 - delta).")
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")

    # nnPU / head knobs.
    parser.add_argument("--pi-p", type=float, default=0.5,
                        help="Class prior P(y=+1): fraction of success-like frames inside "
                             "failure rollouts. Default 0.5 (logged warning); set from "
                             "domain knowledge.")
    parser.add_argument("--loss-surrogate", type=str, default="logistic",
                        choices=["sigmoid", "logistic"],
                        help="nnPU surrogate loss. 'logistic' (softplus) is the "
                             "robust default; 'sigmoid' (Kiryo) saturates to zero "
                             "gradient and collapses under low pi_p + weak features.")
    parser.add_argument("--no-nn-correction", action="store_true",
                        help="Disable the non-negative correction (use plain uPU).")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Lower clamp for the negative-risk term (Kiryo default 0).")
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)

    # Unlabeled failure-pool selection.
    parser.add_argument("--unlabeled-per-task", type=int, default=25,
                        help="Target number of failure trajectories per task to use as the "
                             "UNLABELED pool. If fewer are available, all are used.")
    parser.add_argument(
        "--unlabeled-ids-json",
        type=str,
        default=None,
        help="Optional JSON path mapping task_name -> [video_id, ...] to override auto-selection.",
    )
    return parser.parse_args()


def _positive_int_cap(value: Optional[int]) -> Optional[int]:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _effective_train_caps(
    args: argparse.Namespace,
) -> tuple[Optional[int], Optional[int], int]:
    """Return (success cap, fail cap, unlabeled selection cap per task)."""
    shared = _positive_int_cap(args.train_max_per_task)
    train_success_cap = _positive_int_cap(args.train_max_success_per_task)
    train_fail_cap = _positive_int_cap(args.train_max_fail_per_task)
    if train_success_cap is None:
        train_success_cap = shared
    if train_fail_cap is None:
        train_fail_cap = shared
    unlabeled_cap = (
        int(train_fail_cap)
        if train_fail_cap is not None
        else int(args.unlabeled_per_task)
    )
    return train_success_cap, train_fail_cap, unlabeled_cap


def _parse_camera_to_view(value: Optional[str]):
    if not value:
        return None
    out = {}
    for chunk in value.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def _select_unlabeled(
    pool_by_task: Dict[str, List[RobosuiteBenchmarkTrajectory]],
    eval_tasks: Sequence[str],
    unlabeled_per_task: int,
    unlabeled_ids_override: Optional[Dict[str, List[str]]],
) -> List[RobosuiteBenchmarkTrajectory]:
    """Deterministically pick the unlabeled failure trajectories per task.

    If a task has fewer disjoint failure trajectories than ``unlabeled_per_task``,
    all available are used (no error).
    """
    out: List[RobosuiteBenchmarkTrajectory] = []
    for task in sorted(set(eval_tasks)):
        pool = sorted(pool_by_task.get(task, []), key=lambda t: str(t.video_id))
        pool_by_id = {str(t.video_id): t for t in pool}

        if unlabeled_ids_override and task in unlabeled_ids_override:
            chosen_ids = list(unlabeled_ids_override[task])
            missing = [v for v in chosen_ids if v not in pool_by_id]
            if missing:
                raise RuntimeError(
                    f"Task {task!r}: --unlabeled-ids-json names video_ids not present in the "
                    f"disjoint pool: {missing}"
                )
            out.extend(pool_by_id[v] for v in chosen_ids)
        else:
            n_take = min(len(pool), int(unlabeled_per_task))
            if n_take == 0:
                raise RuntimeError(
                    f"Task {task!r}: zero failure trajectories available in "
                    f"fail-train-split. Check --fail-train-split and --tasks."
                )
            if n_take < int(unlabeled_per_task):
                print(
                    f"[robosuite][pu_bce] task={task}: only {n_take} failure trajectories "
                    f"available (< --unlabeled-per-task={unlabeled_per_task}); using all.",
                    flush=True,
                )
            out.extend(pool[:n_take])
    return out


def main() -> None:
    args = _parse_args()
    train_max_success_per_task, train_max_fail_per_task, unlabeled_per_task = (
        _effective_train_caps(args)
    )
    print("[robosuite][pu_bce] building FailureBenchmark...", flush=True)
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
        f"[robosuite][pu_bce] eval set: {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={eval_tasks}",
        flush=True,
    )

    print(
        f"[robosuite][pu_bce] train caps per task: "
        f"success<={train_max_success_per_task} fail_unlabeled<={unlabeled_per_task}",
        flush=True,
    )

    # Discover failure rollouts for the unlabeled pool (mask optional).
    all_fail = discover_unlabeled_failures(
        data_root=args.data_root,
        tasks=args.tasks,
        split=args.fail_train_split,
        max_fail_per_task=train_max_fail_per_task,
    )
    all_fail = [t for t in all_fail if bool(t.is_failure)]
    # video_id disjointness against eval set (defence-in-depth).
    pool = [t for t in all_fail if str(t.video_id) not in eval_fail_keys]
    dropped = len(all_fail) - len(pool)
    print(
        f"[robosuite][pu_bce] failure discovery from {args.fail_train_split}: "
        f"all_fail={len(all_fail)} eval_fail={len(eval_fail_keys)} "
        f"pool={len(pool)} (dropped {dropped} as video_id overlap)",
        flush=True,
    )

    pool_by_task: Dict[str, List[RobosuiteBenchmarkTrajectory]] = {}
    for t in pool:
        pool_by_task.setdefault(str(t.task_name), []).append(t)

    unlabeled_ids_override = None
    if args.unlabeled_ids_json:
        with open(args.unlabeled_ids_json, "r") as fh:
            unlabeled_ids_override = json.load(fh)
        if not isinstance(unlabeled_ids_override, dict):
            raise RuntimeError(
                "--unlabeled-ids-json must be a JSON object mapping task -> [video_id, ...]"
            )

    unlabeled_trajs = _select_unlabeled(
        pool_by_task=pool_by_task,
        eval_tasks=eval_tasks,
        unlabeled_per_task=unlabeled_per_task,
        unlabeled_ids_override=unlabeled_ids_override,
    )

    # Defence-in-depth disjoint check (adapter also HARD-asserts inside fit).
    unlabeled_keys = {str(t.video_id) for t in unlabeled_trajs}
    overlap = sorted(unlabeled_keys & eval_fail_keys)
    if overlap:
        raise RuntimeError(
            f"Disjointness invariant violated: unlabeled video_ids appear in eval set: {overlap}"
        )

    unlabeled_by_task: Dict[str, List[str]] = {}
    for t in unlabeled_trajs:
        unlabeled_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    print(
        "[robosuite][pu_bce] unlabeled pool sizes per task: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(unlabeled_by_task.items())),
        flush=True,
    )

    train_success_trajs = discover_success_rollouts(
        data_root=args.data_root,
        tasks=eval_tasks,
        split=args.success_train_split,
        max_success_per_task=train_max_success_per_task,
    )
    train_by_task: Dict[str, int] = {}
    for t in train_success_trajs:
        train_by_task[str(t.task_name)] = train_by_task.get(str(t.task_name), 0) + 1
    print(
        f"[robosuite][pu_bce] train success from {args.success_train_split}: "
        f"{len(train_success_trajs)} trajectories "
        + ", ".join(f"{k}={v}" for k, v in sorted(train_by_task.items())),
        flush=True,
    )

    discriminator = PUBCEBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        unlabeled_fail_trajectories=unlabeled_trajs,
        pi_p=float(args.pi_p),
        loss_surrogate=str(args.loss_surrogate),
        nn_correction=not bool(args.no_nn_correction),
        beta=float(args.beta),
        head_hidden=int(args.head_hidden),
        head_layers=int(args.head_layers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        save_ckpt_dir=str(args.save_ckpt_dir) if args.save_ckpt_dir else None,
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        delta=float(args.delta),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )

    try:
        print("[robosuite][pu_bce] training nnPU head (no GT timing; no eval during training)...",
              flush=True)
        discriminator.fit_on_benchmark(
            trajs,
            train_success_trajectories=train_success_trajs,
        )

        print("[robosuite][pu_bce] training complete; running bench.evaluate(...)", flush=True)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        calib_summary = discriminator.calibration_summary()
        print("[robosuite][pu_bce] calibration summary:", calib_summary)

        if args.save_json:
            out_dir = os.path.dirname(os.path.abspath(args.save_json))
            os.makedirs(out_dir, exist_ok=True)
            result.save_json(args.save_json)
            manifest_path = os.path.join(out_dir, "unlabeled_pool_manifest.json")
            manifest = {
                "pretraining_method": "rpt",
                "feature_source": "rpt_action_token",
                "representation_fingerprint": discriminator.representation_fingerprint,
                "labeling": "pu_no_gt_timing",
                "loss": "nnpu",
                "loss_surrogate": str(args.loss_surrogate),
                "nn_correction": not bool(args.no_nn_correction),
                "beta": float(args.beta),
                "pi_p": float(args.pi_p),
                "data_root": str(args.data_root),
                "fail_train_split": str(args.fail_train_split),
                "fail_eval_split": str(args.fail_split),
                "success_eval_split": str(args.success_split),
                "success_train_split": str(args.success_train_split),
                "train_max_success_per_task": train_max_success_per_task,
                "train_max_fail_per_task": train_max_fail_per_task,
                "unlabeled_pool": unlabeled_by_task,
                "unlabeled_per_task": int(unlabeled_per_task),
                "eval_fail_video_ids": sorted(eval_fail_keys),
                "delta": float(args.delta),
                "calib_mode": "success_percentile",
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "head_hidden": int(args.head_hidden),
                "head_layers": int(args.head_layers),
            }
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            print(f"[robosuite][pu_bce] wrote {args.save_json}")
            print(f"[robosuite][pu_bce] wrote {manifest_path}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
