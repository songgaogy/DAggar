import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_DATA_DIRS = [
    REPO_ROOT / "data" / "PickPlaceBread" / "expert",
    REPO_ROOT / "data" / "PickPlaceCereal" / "expert",
    REPO_ROOT / "data" / "PickPlaceMilk" / "expert",
    REPO_ROOT / "data" / "PandaPickPlaceCan" / "expert",
    REPO_ROOT / "data" / "PandaStack" / "expert",
    REPO_ROOT / "data" / "PandaLift" / "expert",
]

DEFAULT_CAMERA_NAMES = ["agentview", "robot0_robotview", "robot0_eye_in_hand"]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the expert demos used by flow_multi pretraining into "
            "data/<task_dir>/expert_pretrain_data."
        )
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_DATA_DIRS],
        help="Expert data directories. Defaults match scripts/train_flow_multi.sh.",
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=20,
        help="Number of valid demos to extract per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed recorded in manifest. Demo selection itself is deterministic.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=8,
        help="Minimum demo length required by flow_multi training.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Training stride recorded in manifest.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="expert_pretrain_data.hdf5",
        help="Output HDF5 filename inside each expert_pretrain_data directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing expert_pretrain_data HDF5 files.",
    )
    return parser.parse_args()


def _list_hdf5_files(data_dir: Path):
    files = sorted(data_dir.glob("*.hdf5"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def _copy_root_attrs(src_handle, dst_handle):
    for key, value in src_handle.attrs.items():
        dst_handle.attrs[key] = value


def _select_demos(data_dir: Path, num_traj: int, action_horizon: int, camera_names: list[str]):
    selected = []
    files = _list_hdf5_files(data_dir)
    task_name = None

    for file_path in files:
        if len(selected) >= num_traj:
            break
        with h5py.File(file_path, "r") as handle:
            if task_name is None:
                task_name = str(handle.attrs.get("env", data_dir.parent.name))
            demo_root = handle["demos"]
            for demo_key in sorted(demo_root.keys()):
                if len(selected) >= num_traj:
                    break
                demo_group = demo_root[demo_key]
                obs_group = demo_group["observations"]
                missing = [name for name in camera_names if name not in obs_group]
                if len(missing) > 0:
                    raise KeyError(f"Missing cameras {missing} in {file_path}:{demo_key}")
                num_steps = int(demo_group["actions"].shape[0])
                if num_steps < action_horizon:
                    continue
                selected.append(
                    {
                        "source_file": str(file_path),
                        "source_demo_key": str(demo_key),
                        "num_steps": num_steps,
                    }
                )

    if len(selected) < num_traj:
        raise RuntimeError(
            f"Only found {len(selected)} valid demos in {data_dir}, expected {num_traj}."
        )

    return task_name, selected


def _write_task_dataset(
    data_dir: Path,
    selected: list[dict],
    output_name: str,
    overwrite: bool,
):
    output_dir = data_dir.parent / "expert_pretrain_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    with h5py.File(selected[0]["source_file"], "r") as first_handle:
        with h5py.File(output_path, "w") as dst_handle:
            _copy_root_attrs(first_handle, dst_handle)
            dst_demo_root = dst_handle.create_group("demos")
            for dst_idx, item in enumerate(selected, start=1):
                dst_key = f"demo_{dst_idx:06d}"
                with h5py.File(item["source_file"], "r") as src_handle:
                    src_handle.copy(src_handle["demos"][item["source_demo_key"]], dst_demo_root, name=dst_key)
                item["target_demo_key"] = dst_key

    return output_dir, output_path


def _write_manifest(
    output_dir: Path,
    task_dir_name: str,
    task_name: str,
    data_dir: Path,
    output_path: Path,
    selected: list[dict],
    args,
):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "task_dir_name": task_dir_name,
        "task_name": task_name,
        "source_data_dir": str(data_dir),
        "output_hdf5": str(output_path),
        "seed": int(args.seed),
        "num_traj": int(args.num_traj),
        "action_horizon": int(args.action_horizon),
        "stride": int(args.stride),
        "camera_names": list(DEFAULT_CAMERA_NAMES),
        "selection_rule": "sorted hdf5 files, sorted demo keys, skip demos shorter than action_horizon",
        "demos": selected,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


def main():
    args = _parse_args()
    data_dirs = []
    for path in args.data_dirs:
        data_dir = Path(path).expanduser()
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        data_dirs.append(data_dir)

    print("Extracting flow_multi expert pretrain data")
    print(f"seed={args.seed} num_traj={args.num_traj} action_horizon={args.action_horizon}")

    for data_dir in data_dirs:
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        task_dir_name = data_dir.parent.name
        task_name, selected = _select_demos(
            data_dir=data_dir,
            num_traj=int(args.num_traj),
            action_horizon=int(args.action_horizon),
            camera_names=DEFAULT_CAMERA_NAMES,
        )
        output_dir, output_path = _write_task_dataset(
            data_dir=data_dir,
            selected=selected,
            output_name=str(args.output_name),
            overwrite=bool(args.overwrite),
        )
        manifest_path = _write_manifest(
            output_dir=output_dir,
            task_dir_name=task_dir_name,
            task_name=task_name,
            data_dir=data_dir,
            output_path=output_path,
            selected=selected,
            args=args,
        )
        print(
            f"{task_dir_name}: saved {len(selected)} demos "
            f"from env={task_name} to {output_path}"
        )
        print(f"{task_dir_name}: manifest {manifest_path}")


if __name__ == "__main__":
    main()
