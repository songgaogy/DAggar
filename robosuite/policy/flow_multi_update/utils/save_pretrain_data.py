"""Save the exact demos selected by the dataset for flow_multi pretraining.

The HDF5 copy/manifest logic mirrors temp/extract_dataset.py but is rewritten here
so the training pipeline does not import that standalone script.
"""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import h5py


def _copy_root_attrs(src_handle, dst_handle):
    for key, value in src_handle.attrs.items():
        dst_handle.attrs[key] = value


def _group_by_data_dir(demo_meta: list[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for item in demo_meta:
        grouped.setdefault(item["data_dir"], []).append(item)
    return grouped


def _write_task_dataset(output_path: Path, selected: list[dict]):
    # Use the first selected demo's source file to seed root attrs.
    with h5py.File(selected[0]["file_path"], "r") as first_handle:
        with h5py.File(output_path, "w") as dst_handle:
            _copy_root_attrs(first_handle, dst_handle)
            dst_demo_root = dst_handle.create_group("demos")
            for dst_idx, item in enumerate(selected, start=1):
                dst_key = f"demo_{dst_idx:06d}"
                with h5py.File(item["file_path"], "r") as src_handle:
                    src_handle.copy(src_handle["demos"][item["demo_key"]], dst_demo_root, name=dst_key)
                item["target_demo_key"] = dst_key


def _write_manifest(
    manifest_path: Path,
    task_name: str,
    data_dir: str,
    output_path: Path,
    selected: list[dict],
    camera_names: list[str],
    seed: int,
    action_horizon: int,
    stride: int,
):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_name": task_name,
        "source_data_dir": str(data_dir),
        "output_hdf5": str(output_path),
        "seed": int(seed),
        "num_traj": len(selected),
        "action_horizon": int(action_horizon),
        "stride": int(stride),
        "camera_names": list(camera_names),
        "selection_rule": "seeded random subset of valid demos used for training",
        "demos": [
            {
                "source_file": item["file_path"],
                "source_demo_key": item["demo_key"],
                "num_steps": int(item["num_steps"]),
                "target_demo_key": item.get("target_demo_key"),
                "task_name": item["task_name"],
            }
            for item in selected
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def save_pretrain_demos(
    demo_meta: list[dict],
    camera_names: list[str],
    seed: int,
    action_horizon: int,
    stride: int,
    timestamp: str,
    output_name: str = "expert_pretrain_data.hdf5",
):
    """Write per-task HDF5 + manifest under data_root/<task>/pretrain_data-<timestamp>/.

    `demo_meta` items must contain file_path, demo_key, num_steps, task_name, data_dir
    (i.e. the dataset's selected_demos).
    """
    if len(demo_meta) == 0:
        print("[save_pretrain_data] no demos to save, skipping.")
        return

    grouped = _group_by_data_dir(demo_meta)
    for data_dir, selected in grouped.items():
        task_name = selected[0]["task_name"]
        # data_dir is .../<task>/<split>; save next to the task folder.
        output_dir = Path(data_dir).parent / f"pretrain_data-{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        _write_task_dataset(output_path, selected)
        _write_manifest(
            manifest_path=output_dir / "manifest.json",
            task_name=task_name,
            data_dir=data_dir,
            output_path=output_path,
            selected=selected,
            camera_names=camera_names,
            seed=seed,
            action_horizon=action_horizon,
            stride=stride,
        )
        print(f"[save_pretrain_data] {task_name}: saved {len(selected)} demos to {output_path}")
