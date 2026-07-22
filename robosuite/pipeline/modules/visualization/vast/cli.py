"""Visualize finetuned offline VAST critics with the standard value visualizer.

This module is intentionally a thin wrapper around
``robosuite.pipeline.modules.visualization.vast.renderer``. Offline finetune
checkpoints can miss visualization metadata that warmup checkpoints persist, so
we patch the loaded payload in memory from the offline run files and then delegate
all plotting / CSV / video work to the existing visualizer.
"""

from __future__ import annotations

import argparse
import json
import warnings
import sys
from pathlib import Path
from typing import Any

import torch


def resolve_cli_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def infer_run_dir(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def load_resolved_config(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - project runtime has OmegaConf.
        raise RuntimeError(
            f"Cannot read {path}: omegaconf is unavailable in this environment."
        ) from exc
    return OmegaConf.load(path)


def cfg_select(cfg: Any | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, key, default=default)


def cfg_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


def as_str_list(value: Any) -> list[str]:
    value = cfg_container(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            continue
        return value
    return None


def require_cuda_device(device: str | None) -> str:
    value = str(device or "cuda").strip()
    if not torch.cuda.is_available():
        raise RuntimeError("vis_vast_finetuned requires CUDA, but torch.cuda.is_available() is False.")
    if not value.startswith("cuda"):
        raise ValueError(f"vis_vast_finetuned requires a CUDA device, got {value!r}.")
    parsed = torch.device(value)
    if parsed.type != "cuda":
        raise ValueError(f"vis_vast_finetuned requires a CUDA device, got {value!r}.")
    if parsed.index is not None and parsed.index >= torch.cuda.device_count():
        raise ValueError(
            f"device={value!r} is unavailable; visible CUDA device count is "
            f"{torch.cuda.device_count()}."
        )
    return value


def resolve_offline_buffer(
    *,
    explicit: str | None,
    run_info: dict[str, Any],
    task_name: str,
) -> Path:
    candidates: list[Path] = []
    if explicit is not None and str(explicit).strip():
        candidates.append(resolve_cli_path(explicit))
    warmup_path = run_info.get("warmup_transitions_path")
    if warmup_path is not None and str(warmup_path).strip():
        candidates.append(resolve_cli_path(str(warmup_path)))
    candidates.extend(
        [
            Path.cwd() / "data" / task_name / "offline_data-vast" / "vast_offline_transitions.pt",
            Path.cwd() / "data" / task_name / "offline_data" / "vast_offline_transitions.pt",
            Path.cwd() / "data" / task_name / "offline_data-vast" / "iql_offline_transitions.pt",
            Path.cwd() / "data" / task_name / "offline_data-iql" / "iql_offline_transitions.pt",
        ]
    )

    for candidate in candidates:
        if candidate.is_dir():
            canonical = candidate / "vast_offline_transitions.pt"
            if canonical.is_file():
                return canonical
        elif candidate.is_file():
            return candidate
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve VAST offline transitions for visualization. Tried:\n"
        f"  {attempted}"
    )


def augment_payload_metadata(
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    run_info: dict[str, Any],
    resolved_cfg: Any | None,
    task_data_name: str | None,
    disc_ckpt: str | None,
) -> dict[str, Any]:
    payload = dict(payload)
    meta = dict(payload.get("encoder_meta", {}) or {})
    cfg_dict = dict(payload.get("cfg", {}) or {})

    task_name = first_nonempty(
        task_data_name,
        meta.get("task"),
        meta.get("task_env"),
        meta.get("nnpu_task"),
        run_info.get("task_name"),
        cfg_select(resolved_cfg, "algorithm.task_name"),
        cfg_select(resolved_cfg, "env.environment"),
    )
    if not task_name:
        raise KeyError(
            f"Cannot infer task name for {checkpoint}. Pass --task-data-name."
        )
    task_name = str(task_name)

    camera_names = as_str_list(meta.get("policy_camera_names"))
    if not camera_names:
        camera_names = as_str_list(run_info.get("policy_camera_names"))
    if not camera_names:
        camera_names = as_str_list(cfg_select(resolved_cfg, "algorithm.camera_names"))
    if not camera_names:
        camera_names = as_str_list(cfg_select(resolved_cfg, "env.camera_names"))
    if not camera_names:
        raise KeyError(
            f"Cannot infer policy_camera_names for {checkpoint}. "
            "Expected run_info.json or config_resolved.yaml next to the run."
        )

    nnpu_ckpt = first_nonempty(
        disc_ckpt,
        meta.get("nnpu_checkpoint"),
        run_info.get("nnpu_checkpoint"),
        cfg_select(resolved_cfg, "algorithm.discriminator.checkpoint"),
    )
    if not nnpu_ckpt:
        raise KeyError(f"Cannot infer nnPU checkpoint for {checkpoint}; pass --disc-ckpt.")

    img_height = first_nonempty(
        meta.get("img_height"),
        meta.get("image_size"),
        cfg_select(resolved_cfg, "env.img_height"),
        cfg_select(resolved_cfg, "algorithm.flow.image_size"),
    )
    img_width = first_nonempty(
        meta.get("img_width"),
        meta.get("image_size"),
        cfg_select(resolved_cfg, "env.img_width"),
        img_height,
    )
    control_freq = first_nonempty(
        meta.get("control_freq"),
        cfg_select(resolved_cfg, "env.control_freq"),
        cfg_select(resolved_cfg, "runtime.control_fps"),
    )
    renderer = first_nonempty(meta.get("renderer"), cfg_select(resolved_cfg, "env.renderer"))
    reward_mode = first_nonempty(meta.get("reward_mode"), cfg_dict.get("reward_mode"))
    vast_fields = {
        "vast_v_mode": first_nonempty(
            cfg_dict.get("vast_v_mode"),
            meta.get("vast_v_mode"),
            run_info.get("vast_v_mode"),
            cfg_select(resolved_cfg, "algorithm.vast.config.vast_v_mode"),
        ),
        "vast_max_k": first_nonempty(
            cfg_dict.get("vast_max_k"),
            meta.get("vast_max_k"),
            run_info.get("vast_max_k"),
            cfg_select(resolved_cfg, "algorithm.vast.config.vast_max_k"),
        ),
        "vast_comp_coef": first_nonempty(
            cfg_dict.get("vast_comp_coef"),
            meta.get("vast_comp_coef"),
            run_info.get("vast_comp_coef"),
            cfg_select(resolved_cfg, "algorithm.vast.config.vast_comp_coef"),
        ),
        "vast_sampling_seed": first_nonempty(
            cfg_dict.get("vast_sampling_seed"),
            meta.get("vast_sampling_seed"),
            run_info.get("vast_sampling_seed"),
            cfg_select(resolved_cfg, "algorithm.vast.config.vast_sampling_seed"),
        ),
    }
    camera_aliases = cfg_container(
        first_nonempty(
            meta.get("camera_aliases"),
            cfg_select(resolved_cfg, "algorithm.flow.camera_aliases"),
            {},
        )
    )

    meta.update(
        {
            "algorithm": "vast_value_stitching_adaptation",
            "task": task_name,
            "task_env": task_name,
            "nnpu_task": task_name,
            "policy_camera_names": camera_names,
            "nnpu_checkpoint": str(nnpu_ckpt),
        }
    )
    if img_height is not None:
        meta["img_height"] = int(img_height)
    if img_width is not None:
        meta["img_width"] = int(img_width)
    if control_freq is not None:
        meta["control_freq"] = int(control_freq)
    if renderer is not None:
        meta["renderer"] = str(renderer)
    if reward_mode is not None:
        meta["reward_mode"] = str(reward_mode)
    meta["camera_aliases"] = dict(camera_aliases or {})
    for key, value in vast_fields.items():
        if value is not None:
            cfg_dict[key] = value
            meta[key] = value

    payload["encoder_meta"] = meta
    payload["cfg"] = cfg_dict
    payload["algorithm"] = "vast_value_stitching_adaptation"
    return payload


def drop_forwarded_option(args: list[str], option: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item == option:
            skip_next = True
            continue
        if item.startswith(f"{option}="):
            continue
        out.append(item)
    return out


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a finetuned offline VAST checkpoint by delegating to the "
            "standard VAST renderer."
        )
    )
    parser.add_argument("--checkpoint", "--vast-ckpt", dest="checkpoint", default=None)
    parser.add_argument("--iql-ckpt", dest="legacy_iql_ckpt", default=None)
    parser.add_argument("--disc-ckpt", default=None)
    parser.add_argument("--task-data-name", default=None)
    parser.add_argument("--split", default="fail")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--offline-buffer",
        default=None,
        help="Optional vast_offline_transitions.pt override. Defaults to run_info.json.",
    )
    args, forwarded = parser.parse_known_args()
    if args.checkpoint and args.legacy_iql_ckpt:
        parser.error("set only one of --vast-ckpt and deprecated --iql-ckpt")
    if args.legacy_iql_ckpt:
        warnings.warn(
            "--iql-ckpt is deprecated; use --vast-ckpt.",
            FutureWarning,
            stacklevel=2,
        )
        args.checkpoint = args.legacy_iql_ckpt
    if not args.checkpoint:
        parser.error("--vast-ckpt is required")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return args, forwarded


def main() -> None:
    args, forwarded = parse_args()
    device = require_cuda_device(args.device)
    checkpoint = resolve_cli_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Finetuned VAST checkpoint not found: {checkpoint}")

    run_dir = infer_run_dir(checkpoint)
    run_info = load_json(run_dir / "run_info.json")
    resolved_cfg = load_resolved_config(run_dir / "config_resolved.yaml")
    task_name = str(
        first_nonempty(
            args.task_data_name,
            run_info.get("task_name"),
            cfg_select(resolved_cfg, "algorithm.task_name"),
            cfg_select(resolved_cfg, "env.environment"),
        )
        or ""
    )
    if not task_name:
        raise KeyError(f"Cannot infer task name for {checkpoint}; pass --task-data-name.")

    offline_buffer = resolve_offline_buffer(
        explicit=args.offline_buffer,
        run_info=run_info,
        task_name=task_name,
    )
    output_root = (
        resolve_cli_path(args.output_root)
        if args.output_root
        else run_dir / "vis_vast"
    )
    forwarded = drop_forwarded_option(forwarded, "--output-root")

    from robosuite.pipeline.modules.visualization.vast import renderer

    original_loader = renderer.load_vast_payload

    def patched_load_vast_payload(path: Path) -> dict[str, Any]:
        loaded = original_loader(path)
        return augment_payload_metadata(
            loaded,
            checkpoint=checkpoint,
            run_info=run_info,
            resolved_cfg=resolved_cfg,
            task_data_name=task_name,
            disc_ckpt=args.disc_ckpt,
        )

    renderer.load_vast_payload = patched_load_vast_payload
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            old_argv[0],
            "--vast-ckpt",
            str(checkpoint),
            "--output-root",
            str(output_root),
            "--task-data-name",
            task_name,
            "--offline-buffer",
            str(offline_buffer),
            "--split",
            str(args.split),
            "--seed",
            str(args.seed),
            "--device",
            device,
        ]
        if args.disc_ckpt:
            sys.argv.extend(["--disc-ckpt", str(args.disc_ckpt)])
        sys.argv.extend(forwarded)

        print(f"[vis_vast_finetuned] checkpoint={checkpoint}")
        print(f"[vis_vast_finetuned] run_dir={run_dir}")
        print(f"[vis_vast_finetuned] output_root={output_root}")
        print(f"[vis_vast_finetuned] offline_buffer={offline_buffer}")
        print(f"[vis_vast_finetuned] task={task_name} split={args.split} seed={args.seed} device={device}")
        renderer.main()
    finally:
        sys.argv = old_argv
        renderer.load_vast_payload = original_loader


if __name__ == "__main__":
    main()
