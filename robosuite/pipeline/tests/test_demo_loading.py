import random
from pathlib import Path

import h5py
import pytest

from robosuite.pipeline.src.data.demos import load_demo_paths


def _write_demo_file(path: Path, demo_names: list[str]) -> None:
    with h5py.File(path, "w") as file_handle:
        demos = file_handle.create_group("data")
        for demo_name in demo_names:
            demos.create_group(demo_name)


def test_random_sampling_uses_global_hdf5_catalog_and_is_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _write_demo_file(first_path, ["demo_0", "demo_1", "demo_2"])
    _write_demo_file(second_path, ["demo_3", "demo_4", "demo_5"])
    paths = [first_path, second_path]
    selected_runs: list[list[tuple[Path, str]]] = []

    for _ in range(2):
        selected: list[tuple[Path, str]] = []
        load_demo_paths(
            paths,
            hdf5_loader=lambda path, demo_names: [],
            max_num_trajectories=4,
            random_sample=True,
            random_seed=42,
            selected_demo_callback=lambda path, names: selected.extend(
                (path, name) for name in names
            ),
        )
        selected_runs.append(selected)

    catalog = [
        (first_path, "demo_0"),
        (first_path, "demo_1"),
        (first_path, "demo_2"),
        (second_path, "demo_3"),
        (second_path, "demo_4"),
        (second_path, "demo_5"),
    ]
    expected = random.Random(42).sample(catalog, 4)
    assert set(selected_runs[0]) == set(expected)
    assert selected_runs[0] == selected_runs[1]
    assert {path for path, _ in selected_runs[0]} == {first_path, second_path}


@pytest.mark.parametrize("requested", [0, -1])
def test_trajectory_limit_must_be_positive(tmp_path: Path, requested: int) -> None:
    demo_path = tmp_path / "expert.hdf5"
    _write_demo_file(demo_path, ["demo_0"])

    with pytest.raises(ValueError, match="at least 1"):
        load_demo_paths(
            [demo_path],
            hdf5_loader=lambda path, demo_names: [],
            max_num_trajectories=requested,
        )


def test_trajectory_limit_cannot_exceed_global_availability(tmp_path: Path) -> None:
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _write_demo_file(first_path, ["demo_0"])
    _write_demo_file(second_path, ["demo_1"])

    with pytest.raises(ValueError, match="requested 3, available 2"):
        load_demo_paths(
            [first_path, second_path],
            hdf5_loader=lambda path, demo_names: [],
            max_num_trajectories=3,
        )


def test_cache_key_distinguishes_selected_demo_ids(tmp_path: Path) -> None:
    demo_path = tmp_path / "expert.hdf5"
    cache_dir = tmp_path / "cache"
    _write_demo_file(demo_path, [f"demo_{index}" for index in range(6)])
    loader_calls: list[list[str]] = []

    def loader(path: Path, demo_names: list[str]):
        loader_calls.append(list(demo_names))
        return []

    for seed in (1, 1, 2):
        load_demo_paths(
            [demo_path],
            cache_dir=cache_dir,
            hdf5_loader=loader,
            max_num_trajectories=3,
            random_sample=True,
            random_seed=seed,
            cache_key="conversion_v2",
        )

    assert loader_calls[0] != loader_calls[1]
    assert len(loader_calls) == 2
    assert len(list(cache_dir.glob("expert_conversion_v2_selected_*.pt"))) == 2
