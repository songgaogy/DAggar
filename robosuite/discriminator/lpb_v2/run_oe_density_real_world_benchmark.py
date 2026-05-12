"""Run the OE-density score discriminator through the real-world Agilex benchmark."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from benchmark.core import EvalConfig
from benchmark.real_world import FailureBenchmark
from benchmark.real_world.loader import discover_agilex_trajectories
from benchmark.real_world.trajectory import AgilexBenchmarkTrajectory
from robosuite.discriminator.lpb_v2.benchmark_oe_density import (
    OEDensityBenchmarkDiscriminator,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)

    parser.add_argument("--proprio-field", type=str, default="qpos")
    parser.add_argument("--proprio-start", type=int, default=7)
    parser.add_argument("--proprio-stop", type=int, default=14)
    parser.add_argument("--action-start", type=int, default=7)
    parser.add_argument("--action-stop", type=int, default=14)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", type=str, default=None)

    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--knn-feature-source",
        type=str,
        default="transformer",
        choices=["encoder", "transformer"],
    )
    parser.add_argument("--knn-transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")

    # OE-density / fail-train knobs.
    parser.add_argument("--ftrain-per-task", type=int, default=10)
    parser.add_argument("--ftrain-last-k", type=int, default=60)
    parser.add_argument(
        "--fail-use-all-after-gt",
        dest="fail_use_all_after_gt",
        action="store_true",
        default=True,
        help="Use all frames [first_gt, T) as fail; ignores --ftrain-last-k.",
    )
    parser.add_argument(
        "--no-fail-use-all-after-gt",
        dest="fail_use_all_after_gt",
        action="store_false",
        help="Restrict fail to [first_gt, first_gt + last_k).",
    )
    parser.add_argument(
        "--fail-prefix-to-succ",
        dest="fail_prefix_to_succ",
        action="store_true",
        default=True,
        help="Add the [0, first_gt) prefix of each fail trajectory to the succ training pool.",
    )
    parser.add_argument(
        "--no-fail-prefix-to-succ",
        dest="fail_prefix_to_succ",
        action="store_false",
    )
    parser.add_argument(
        "--ftrain-ids-json",
        type=str,
        default=None,
        help="Optional JSON path mapping task_name -> [video_id, ...] to override auto-selection.",
    )
    parser.add_argument("--fail-calib-per-task", type=int, default=0)

    parser.add_argument("--score-k", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--density",
        type=str,
        default="gaussian",
        choices=["gaussian", "gmm2", "gmm4"],
    )
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
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


def _select_ftrain_and_calib(
    pool_by_task: Dict[str, List[AgilexBenchmarkTrajectory]],
    eval_tasks: Sequence[str],
    ftrain_per_task: int,
    fail_calib_per_task: int,
    ftrain_ids_override: Optional[Dict[str, List[str]]],
) -> tuple[List[AgilexBenchmarkTrajectory], List[AgilexBenchmarkTrajectory]]:
    """Deterministically pick ftrain + fail-calib trajectories per task.

    Picks the first N by sorted video_id from the disjoint pool. If
    ``ftrain_ids_override`` is provided, those ids are used verbatim and must
    already be present in the pool. Mirrors ``_select_bank_and_calib`` in the
    two-bank CLI.
    """
    ftrain_out: List[AgilexBenchmarkTrajectory] = []
    calib_out: List[AgilexBenchmarkTrajectory] = []

    for task in sorted(set(eval_tasks)):
        pool = sorted(pool_by_task.get(task, []), key=lambda t: str(t.video_id))
        pool_by_id = {str(t.video_id): t for t in pool}

        if ftrain_ids_override and task in ftrain_ids_override:
            chosen_ids = list(ftrain_ids_override[task])
            missing = [v for v in chosen_ids if v not in pool_by_id]
            if missing:
                raise RuntimeError(
                    f"Task {task!r}: --ftrain-ids-json names video_ids not present in the "
                    f"disjoint pool: {missing}"
                )
            ftrain_trajs = [pool_by_id[v] for v in chosen_ids]
            remaining = [t for t in pool if str(t.video_id) not in set(chosen_ids)]
        else:
            if len(pool) < ftrain_per_task:
                raise RuntimeError(
                    f"Task {task!r}: only {len(pool)} disjoint failure trajectories available "
                    f"in the pool, but --ftrain-per-task={ftrain_per_task}. "
                    "Raise MAX_FAIL_PER_TASK in EVAL, lower ftrain-per-task, or annotate more failures."
                )
            ftrain_trajs = pool[:ftrain_per_task]
            remaining = pool[ftrain_per_task:]

        ftrain_out.extend(ftrain_trajs)

        if fail_calib_per_task > 0:
            if len(remaining) < fail_calib_per_task:
                raise RuntimeError(
                    f"Task {task!r}: only {len(remaining)} failure trajectories remain after "
                    f"taking ftrain, but --fail-calib-per-task={fail_calib_per_task}."
                )
            calib_out.extend(remaining[:fail_calib_per_task])

    return ftrain_out, calib_out


def main() -> None:
    args = _parse_args()
    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        cache_root=args.cache_root,
        proprio_field=args.proprio_field,
        proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
        action_slice=slice(int(args.action_start), int(args.action_stop)),
    )
    trajs = bench.trajectories()
    eval_fail_keys = {str(t.video_id) for t in trajs if bool(t.is_failure)}
    eval_tasks = sorted({str(t.task_name) for t in trajs})
    n_fail = len(eval_fail_keys)
    n_succ = len(trajs) - n_fail
    print(
        f"[real_world][oe_density] eval set: {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={eval_tasks}",
        flush=True,
    )

    # ALL labeled failure trajectories (no cap), regardless of cache_root.
    all_fail = discover_agilex_trajectories(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=None,
        max_success_per_task=0,
        proprio_field=args.proprio_field,
        proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
        action_slice=slice(int(args.action_start), int(args.action_stop)),
    )
    all_fail = [t for t in all_fail if bool(t.is_failure)]
    ftrain_pool = [t for t in all_fail if str(t.video_id) not in eval_fail_keys]
    print(
        f"[real_world][oe_density] failure discovery: all_fail={len(all_fail)} "
        f"eval_fail={len(eval_fail_keys)} ftrain_pool={len(ftrain_pool)}",
        flush=True,
    )

    pool_by_task: Dict[str, List[AgilexBenchmarkTrajectory]] = {}
    for t in ftrain_pool:
        pool_by_task.setdefault(str(t.task_name), []).append(t)

    ftrain_ids_override = None
    if args.ftrain_ids_json:
        with open(args.ftrain_ids_json, "r") as fh:
            ftrain_ids_override = json.load(fh)
        if not isinstance(ftrain_ids_override, dict):
            raise RuntimeError(
                "--ftrain-ids-json must be a JSON object mapping task -> [video_id, ...]"
            )

    if float(args.lam) == 0.0:
        # In the lam=0 ablation we still want the same fail data layout for
        # bookkeeping (disjointness check, manifest, calibration_summary), but
        # the model itself ignores it.
        ftrain_trajs, fail_calib_trajs = _select_ftrain_and_calib(
            pool_by_task=pool_by_task,
            eval_tasks=eval_tasks,
            ftrain_per_task=int(args.ftrain_per_task),
            fail_calib_per_task=int(args.fail_calib_per_task),
            ftrain_ids_override=ftrain_ids_override,
        )
    else:
        ftrain_trajs, fail_calib_trajs = _select_ftrain_and_calib(
            pool_by_task=pool_by_task,
            eval_tasks=eval_tasks,
            ftrain_per_task=int(args.ftrain_per_task),
            fail_calib_per_task=int(args.fail_calib_per_task),
            ftrain_ids_override=ftrain_ids_override,
        )

    # Hard-assert disjointness (defence-in-depth; the adapter also checks).
    ftrain_keys = {str(t.video_id) for t in ftrain_trajs}
    calib_keys = {str(t.video_id) for t in fail_calib_trajs}
    overlap = sorted((ftrain_keys | calib_keys) & eval_fail_keys)
    if overlap:
        raise RuntimeError(
            f"Disjointness invariant violated: ftrain/fail-calib video_ids appear in eval set: {overlap}"
        )

    ftrain_by_task: Dict[str, List[str]] = {}
    for t in ftrain_trajs:
        ftrain_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    calib_by_task: Dict[str, List[str]] = {}
    for t in fail_calib_trajs:
        calib_by_task.setdefault(str(t.task_name), []).append(str(t.video_id))
    print(
        "[real_world][oe_density] ftrain sizes per task: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(ftrain_by_task.items())),
        flush=True,
    )
    if fail_calib_trajs:
        print(
            "[real_world][oe_density] fail calib sizes per task: "
            + ", ".join(f"{k}={len(v)}" for k, v in sorted(calib_by_task.items())),
            flush=True,
        )

    discriminator = OEDensityBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        fail_train_trajectories=ftrain_trajs,
        fail_calib_trajectories=fail_calib_trajs,
        ftrain_last_k=int(args.ftrain_last_k),
        fail_use_all_after_gt=bool(args.fail_use_all_after_gt),
        fail_prefix_to_succ=bool(args.fail_prefix_to_succ),
        score_k=int(args.score_k),
        lam=float(args.lam),
        weight_decay=float(args.weight_decay),
        density=str(args.density),
        calib_mode=str(args.calib_mode),
        num_epochs=int(args.num_epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
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
        print("[real_world][oe_density] fitting OE-density detector...", flush=True)
        discriminator.fit_on_benchmark(trajs)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        calib_summary = discriminator.calibration_summary()
        print("[real_world][oe_density] calibration summary:", calib_summary)

        if args.save_json:
            out_dir = os.path.dirname(os.path.abspath(args.save_json))
            os.makedirs(out_dir, exist_ok=True)
            result.save_json(args.save_json)
            manifest_path = os.path.join(out_dir, "ftrain_manifest.json")
            manifest = {
                "ftrain": ftrain_by_task,
                "fail_calib": calib_by_task,
                "ftrain_per_task": int(args.ftrain_per_task),
                "fail_calib_per_task": int(args.fail_calib_per_task),
                "ftrain_last_k": int(args.ftrain_last_k),
                "fail_use_all_after_gt": bool(args.fail_use_all_after_gt),
                "fail_prefix_to_succ": bool(args.fail_prefix_to_succ),
                "eval_fail_video_ids": sorted(eval_fail_keys),
                "ftrain_index_ranges": calib_summary.get("ftrain_index_ranges", {}),
                "ftrain_prefix_index_ranges": calib_summary.get("ftrain_prefix_index_ranges", {}),
                "score_k": int(args.score_k),
                "lam": float(args.lam),
                "fail_loss_form": "logistic",
                "weight_decay": float(args.weight_decay),
                "density": str(args.density),
                "num_epochs": int(args.num_epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "calib_mode": str(args.calib_mode),
                "delta": float(args.delta),
                "knn_feature_source": str(args.knn_feature_source),
                "knn_transformer_layer": int(args.knn_transformer_layer),
                "discriminator_config": vars(args),
                "train_history": discriminator.train_history,
                "tau_per_task": dict(discriminator._tau_per_task),
            }
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
            print(f"[real_world][oe_density] wrote {args.save_json}")
            print(f"[real_world][oe_density] wrote {manifest_path}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
