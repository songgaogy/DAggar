"""Run the BCE-WAM discriminator (GT failure split) on the real-world Agilex benchmark.

Mirrors `run_two_bank_real_world_benchmark.py` but swaps in
:class:`BCEBenchmarkDiscriminator`. Hard constraint: training and evaluation
are separated -- ``fit_on_benchmark`` does no evaluation, and ``bench.evaluate``
is invoked here *after* the head is trained.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from benchmark.core import EvalConfig
from benchmark.real_world import FailureBenchmark
from benchmark.real_world.loader import discover_agilex_trajectories
from benchmark.real_world.trajectory import AgilexBenchmarkTrajectory
from robosuite.discriminator.lpb_v2.adapters.bce import BCEBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-ckpt-dir", type=str, default=None)
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

    # Failure-pool selection (mirrors two-bank script).
    parser.add_argument("--fail-bank-per-task", type=int, default=25,
                        help="How many disjoint failure trajectories per task to use as D_o source.")
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
    bank_pool_by_task: Dict[str, List[AgilexBenchmarkTrajectory]],
    eval_tasks: Sequence[str],
    fail_bank_per_task: int,
    fail_calib_per_task: int,
    fail_bank_ids_override: Optional[Dict[str, List[str]]],
) -> tuple[List[AgilexBenchmarkTrajectory], List[AgilexBenchmarkTrajectory]]:
    """Deterministically pick fail-bank + fail-calib trajectories per task.

    Same shape as :func:`run_two_bank_real_world_benchmark._select_bank_and_calib`.
    Picks the first N by sorted video_id from the disjoint bank pool. If
    `fail_bank_ids_override` is provided, those ids are used verbatim and
    must already be present in the pool.
    """
    bank_out: List[AgilexBenchmarkTrajectory] = []
    calib_out: List[AgilexBenchmarkTrajectory] = []

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
            if len(pool) < fail_bank_per_task:
                raise RuntimeError(
                    f"Task {task!r}: only {len(pool)} disjoint failure trajectories available "
                    f"in the bank pool, but --fail-bank-per-task={fail_bank_per_task}. "
                    "Raise MAX_FAIL_PER_TASK in EVAL, lower fail-bank-per-task, or annotate more failures."
                )
            bank_trajs = pool[:fail_bank_per_task]
            remaining = pool[fail_bank_per_task:]

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
        f"[real_world][bce] eval set: {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={eval_tasks}",
        flush=True,
    )

    # Discover ALL labeled failure trajectories (no cap), regardless of cache_root.
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
    bank_pool = [t for t in all_fail if str(t.video_id) not in eval_fail_keys]
    print(
        f"[real_world][bce] failure discovery: all_fail={len(all_fail)} "
        f"eval_fail={len(eval_fail_keys)} bank_pool={len(bank_pool)}",
        flush=True,
    )

    bank_pool_by_task: Dict[str, List[AgilexBenchmarkTrajectory]] = {}
    for t in bank_pool:
        bank_pool_by_task.setdefault(str(t.task_name), []).append(t)

    fail_bank_ids_override = None
    if args.fail_bank_ids_json:
        with open(args.fail_bank_ids_json, "r") as fh:
            fail_bank_ids_override = json.load(fh)
        if not isinstance(fail_bank_ids_override, dict):
            raise RuntimeError("--fail-bank-ids-json must be a JSON object mapping task -> [video_id, ...]")

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
        "[real_world][bce] fail bank sizes per task: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(bank_by_task.items())),
        flush=True,
    )

    # ratio<=0 means "disable cap"
    max_eo_ratio: Optional[float] = (
        None if float(args.max_expert_other_ratio) <= 0.0 else float(args.max_expert_other_ratio)
    )
    discriminator = BCEBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
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
        print("[real_world][bce] training BCE head (GT split; no eval during training)...", flush=True)
        discriminator.fit_on_benchmark(trajs)
        # Eval happens here, AFTER training finishes. Single eval pass, no
        # mid-training evaluation has been (or will be) executed.
        print("[real_world][bce] training complete; running bench.evaluate(...)", flush=True)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        calib_summary = discriminator.calibration_summary()
        print("[real_world][bce] calibration summary:", calib_summary)

        if args.save_json:
            out_dir = os.path.dirname(os.path.abspath(args.save_json))
            os.makedirs(out_dir, exist_ok=True)
            result.save_json(args.save_json)
            manifest_path = os.path.join(out_dir, "fail_bank_manifest.json")
            manifest = {
                "labeling": "gt_failure_split",
                "loss": "bce",
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
            print(f"[real_world][bce] wrote {args.save_json}")
            print(f"[real_world][bce] wrote {manifest_path}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
