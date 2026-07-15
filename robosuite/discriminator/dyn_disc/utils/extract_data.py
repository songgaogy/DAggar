"""Extract reproducible CUDA nnPU pretraining latent shards."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robosuite.discriminator.dyn_disc.adapters.single_bank import DynBenchmarkDiscriminator
from robosuite.discriminator.utils.robosuite_benchmark import (
    canonical_task_name,
    discover_success_rollouts,
    discover_unlabeled_failures,
)

from .pretrain_extract_contract import (
    ids_for_task,
    load_checkpoint_contract,
    require_cuda,
    require_saved_normalizer,
    resolve_model_checkpoint,
    select_trajectories_by_id,
    select_unlabeled_trajectories,
    sha256_file,
    split_success_trajectories,
    validate_saved_split_ids,
)
from .pretrain_extract_shards import (
    SCHEMA_VERSION,
    atomic_json_save,
    effective_proprio_indices,
    path_exists,
    publish_staging,
    split_statistics,
    write_pool,
)


LEGACY_SEED = 0
LEGACY_CALIBRATION_FRACTION = 0.2
# Compatibility alias used by device-guard tests and external callers.
_require_cuda = require_cuda


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CUDA-encoded latent shards from a trained nnPU checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to pu_bce_head.pth.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--success-split", default="success_rollout")
    parser.add_argument("--failure-split", default="fail_rollout")
    parser.add_argument("--success-cap", type=_positive_int, default=50)
    parser.add_argument("--failure-cap", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=_positive_int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <data-root>/<task>/discriminator-pretrain.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_success_pools(args, contract, task: str, data_root: Path):
    seed = int(
        args.seed
        if args.seed is not None
        else contract.seed if contract.seed is not None else LEGACY_SEED
    )
    calibration_fraction = float(
        args.calibration_fraction
        if args.calibration_fraction is not None
        else contract.calibration_fraction
        if contract.calibration_fraction is not None
        else LEGACY_CALIBRATION_FRACTION
    )
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )

    success = discover_success_rollouts(
        data_root=str(data_root),
        tasks=[task],
        split=str(args.success_split),
        max_success_per_task=int(args.success_cap),
    )
    success = [trajectory for trajectory in success if str(trajectory.task_name) == task]
    reconstructed_train, reconstructed_calib = split_success_trajectories(
        success,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    saved_train_ids = ids_for_task(
        contract.success_train_video_ids,
        task,
        field_name="success_train_video_ids",
    )
    saved_calib_ids = ids_for_task(
        contract.success_calib_video_ids,
        task,
        field_name="success_calib_video_ids",
    )
    if (saved_train_ids is None) != (saved_calib_ids is None):
        raise RuntimeError(
            "Checkpoint success split provenance is incomplete; train and calibration "
            "trajectory IDs must either both be present or both be absent."
        )
    if saved_train_ids is not None and contract.seed is not None:
        if args.seed is not None and int(args.seed) != int(contract.seed):
            raise ValueError(
                f"--seed={args.seed} conflicts with checkpoint split seed={contract.seed}."
            )
    if saved_calib_ids is not None and contract.calibration_fraction is not None:
        if args.calibration_fraction is not None and not np.isclose(
            float(args.calibration_fraction), float(contract.calibration_fraction)
        ):
            raise ValueError(
                "--calibration-fraction conflicts with checkpoint split provenance: "
                f"cli={args.calibration_fraction}, checkpoint={contract.calibration_fraction}."
            )

    if saved_train_ids is not None and saved_calib_ids is not None:
        overlap = sorted(set(saved_train_ids) & set(saved_calib_ids))
        if overlap:
            raise RuntimeError(f"Checkpoint success train/calibration IDs overlap: {overlap}")
        positive_train = select_trajectories_by_id(
            success,
            saved_train_ids,
            field_name="success_train_video_ids",
        )
        positive_calib = select_trajectories_by_id(
            success,
            saved_calib_ids,
            field_name="success_calib_video_ids",
        )
    else:
        positive_train = reconstructed_train
        positive_calib = reconstructed_calib
        validate_saved_split_ids(
            positive_train, saved_train_ids, field_name="success_train_video_ids"
        )
        validate_saved_split_ids(
            positive_calib, saved_calib_ids, field_name="success_calib_video_ids"
        )
    return positive_train, positive_calib, saved_train_ids, saved_calib_ids, seed, calibration_fraction


def _build_manifest(
    *,
    args,
    task: str,
    contract,
    model_ckpt: Path,
    model_ckpt_sha256: str,
    normalizer_ckpt: Path,
    normalizer_ckpt_sha256: str,
    adapter: DynBenchmarkDiscriminator,
    effective_proprio: list[int] | None,
    seed: int,
    calibration_fraction: float,
    saved_train_ids: list[str] | None,
    saved_calib_ids: list[str] | None,
    splits: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "checkpoint": {
            "path": str(contract.checkpoint_path),
            "sha256": contract.checkpoint_sha256,
            "model_ckpt": str(model_ckpt),
            "model_ckpt_sha256": model_ckpt_sha256,
            "normalizer_ckpt": str(normalizer_ckpt),
            "normalizer_ckpt_sha256": normalizer_ckpt_sha256,
        },
        "feature_contract": {
            "feature_source": contract.feature_source,
            "transformer_layer": int(contract.transformer_layer),
            "use_chunk": bool(contract.use_chunk),
            "device": str(adapter.encoder.device),
            "latent_dtype": "float32",
            "latent_dim": int(contract.in_dim),
            "view_names": [str(value) for value in adapter.encoder.view_names],
            "camera_to_view": dict(adapter.camera_to_view),
            "proprio_indices": effective_proprio,
            "proprio_input_dim": int(adapter.encoder.model.proprio_encoder.in_chans),
            "frameskip": int(adapter.encoder.frameskip),
            "action_dim_per_step": int(adapter.encoder.action_dim_per_step),
            "action_input_dim": int(adapter.encoder.action_input_dim),
        },
        "split_config": {
            "success_split": str(args.success_split),
            "failure_split": str(args.failure_split),
            "success_cap": int(args.success_cap),
            "failure_cap": int(args.failure_cap),
            "seed": int(seed),
            "seed_source": (
                "cli" if args.seed is not None
                else "checkpoint" if contract.seed is not None else "legacy_default"
            ),
            "calibration_fraction": float(calibration_fraction),
            "calibration_fraction_source": (
                "cli" if args.calibration_fraction is not None
                else "checkpoint"
                if contract.calibration_fraction is not None
                else "legacy_default"
            ),
            "success_ids_validated_from_checkpoint": bool(
                saved_train_ids is not None and saved_calib_ids is not None
            ),
            "unlabeled_ids_source": "checkpoint",
            "success_split_source": (
                "checkpoint_ids" if saved_train_ids is not None else "seeded_reconstruction"
            ),
            "checkpoint_seed": contract.seed,
            "checkpoint_calibration_fraction": contract.calibration_fraction,
        },
        "splits": splits,
        "trajectory_ids": {
            name: [str(entry["video_id"]) for entry in entries]
            for name, entries in splits.items()
        },
        "statistics": {
            name: split_statistics(entries) for name, entries in splits.items()
        },
    }


def main() -> None:
    args = _parse_args()
    device = require_cuda(args.device)
    task = canonical_task_name(args.task)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else data_root / task / "discriminator-pretrain"
    )
    if path_exists(output_dir) and not bool(args.overwrite):
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --overwrite to replace it."
        )
    if output_dir == data_root or data_root.is_relative_to(output_dir):
        raise ValueError(f"Unsafe output directory would replace data_root: {output_dir}")

    contract = load_checkpoint_contract(args.checkpoint, device)
    model_ckpt = resolve_model_checkpoint(contract.model_ckpt)
    model_ckpt_sha256 = sha256_file(model_ckpt)
    if contract.model_ckpt_sha256 is not None and contract.model_ckpt_sha256 != model_ckpt_sha256:
        raise ValueError(
            "Resolved dynamics checkpoint does not match parent nnPU provenance: "
            f"expected_sha256={contract.model_ckpt_sha256}, actual_sha256={model_ckpt_sha256}."
        )
    normalizer_ckpt = require_saved_normalizer(model_ckpt)
    normalizer_ckpt_sha256 = sha256_file(normalizer_ckpt)
    if (
        contract.normalizer_ckpt_sha256 is not None
        and contract.normalizer_ckpt_sha256 != normalizer_ckpt_sha256
    ):
        raise ValueError(
            "Resolved normalizer does not match parent nnPU provenance: "
            f"expected_sha256={contract.normalizer_ckpt_sha256}, "
            f"actual_sha256={normalizer_ckpt_sha256}."
        )
    if contract.checkpoint_path.is_relative_to(output_dir) or model_ckpt.is_relative_to(output_dir):
        raise ValueError("Output directory must not contain an input checkpoint.")

    (
        positive_train,
        positive_calib,
        saved_train_ids,
        saved_calib_ids,
        seed,
        calibration_fraction,
    ) = _resolve_success_pools(args, contract, task, data_root)
    u_ids = ids_for_task(
        contract.unlabeled_fail_video_ids,
        task,
        field_name="unlabeled_fail_video_ids",
    )
    if u_ids is None:
        raise AssertionError("unlabeled_fail_video_ids unexpectedly resolved to None")
    failure_candidates = discover_unlabeled_failures(
        data_root=str(data_root),
        tasks=[task],
        split=str(args.failure_split),
        max_fail_per_task=int(args.failure_cap),
    )
    unlabeled_train = select_unlabeled_trajectories(failure_candidates, u_ids)
    source_files = {
        Path(trajectory.file_path).expanduser().resolve()
        for trajectory in (*positive_train, *positive_calib, *unlabeled_train)
    }
    contained_sources = sorted(
        str(path) for path in source_files if path.is_relative_to(output_dir)
    )
    if contained_sources:
        raise ValueError(
            "Output directory must not contain source rollout files; overwrite could "
            f"destroy inputs. Conflicting sources: {contained_sources}"
        )

    encode_batch_size = int(
        args.encode_batch_size
        if args.encode_batch_size is not None
        else contract.encode_batch_size if contract.encode_batch_size is not None else 32
    )
    print(
        f"[extract_data] task={task} positive_train={len(positive_train)} "
        f"positive_calib={len(positive_calib)} unlabeled_train={len(unlabeled_train)} "
        f"seed={seed} calibration_fraction={calibration_fraction} device={device}",
        flush=True,
    )

    with torch.device(device):
        adapter = DynBenchmarkDiscriminator(
            model_ckpt=str(model_ckpt),
            device=str(device),
            encode_batch_size=encode_batch_size,
            proprio_indices=contract.proprio_indices,
            camera_to_view=contract.camera_to_view,
            feature_source=contract.feature_source,
            transformer_layer=contract.transformer_layer,
            use_chunk=contract.use_chunk,
            calib_fraction=calibration_fraction,
            seed=seed,
            verbose_fit=False,
        )
    if adapter.encoder.device.type != "cuda":
        raise RuntimeError(f"Dynamics encoder initialized on non-CUDA device {adapter.encoder.device}.")
    adapter.encoder.model.eval()
    adapter.encoder.model.requires_grad_(False)
    effective_proprio = effective_proprio_indices(adapter, task)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        splits = {
            "positive_train": write_pool(
                adapter=adapter,
                trajectories=positive_train,
                pool="positive_train",
                success_prefix_only=True,
                staging_dir=staging_dir,
                contract=contract,
                model_ckpt=model_ckpt,
                device=device,
                batch_size=encode_batch_size,
            ),
            "positive_calib": write_pool(
                adapter=adapter,
                trajectories=positive_calib,
                pool="positive_calib",
                success_prefix_only=True,
                staging_dir=staging_dir,
                contract=contract,
                model_ckpt=model_ckpt,
                device=device,
                batch_size=encode_batch_size,
            ),
            "unlabeled_train": write_pool(
                adapter=adapter,
                trajectories=unlabeled_train,
                pool="unlabeled_train",
                success_prefix_only=False,
                staging_dir=staging_dir,
                contract=contract,
                model_ckpt=model_ckpt,
                device=device,
                batch_size=encode_batch_size,
            ),
        }
        manifest = _build_manifest(
            args=args,
            task=task,
            contract=contract,
            model_ckpt=model_ckpt,
            model_ckpt_sha256=model_ckpt_sha256,
            normalizer_ckpt=normalizer_ckpt,
            normalizer_ckpt_sha256=normalizer_ckpt_sha256,
            adapter=adapter,
            effective_proprio=effective_proprio,
            seed=seed,
            calibration_fraction=calibration_fraction,
            saved_train_ids=saved_train_ids,
            saved_calib_ids=saved_calib_ids,
            splits=splits,
        )
        atomic_json_save(manifest, staging_dir / "manifest.json")
        publish_staging(staging_dir, output_dir, overwrite=bool(args.overwrite))
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    print(f"[extract_data] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
