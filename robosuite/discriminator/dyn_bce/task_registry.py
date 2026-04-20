from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskAlias:
    canonical_name: str
    checkpoint_name: str


TASK_ALIASES: dict[str, TaskAlias] = {
    "PandaLift": TaskAlias(canonical_name="PandaLift", checkpoint_name="Lift"),
    "Lift": TaskAlias(canonical_name="PandaLift", checkpoint_name="Lift"),
    "PandaStack": TaskAlias(canonical_name="PandaStack", checkpoint_name="Stack"),
    "Stack": TaskAlias(canonical_name="PandaStack", checkpoint_name="Stack"),
    "PandaPickPlaceCan": TaskAlias(canonical_name="PandaPickPlaceCan", checkpoint_name="PickPlaceCan"),
    "PickPlaceCan": TaskAlias(canonical_name="PandaPickPlaceCan", checkpoint_name="PickPlaceCan"),
    "PickPlaceBread": TaskAlias(canonical_name="PickPlaceBread", checkpoint_name="PickPlaceBread"),
    "PickPlaceCereal": TaskAlias(canonical_name="PickPlaceCereal", checkpoint_name="PickPlaceCereal"),
    "PickPlaceMilk": TaskAlias(canonical_name="PickPlaceMilk", checkpoint_name="PickPlaceMilk"),
}


DEFAULT_TASK_ORDER: list[str] = [
    "PandaLift",
    "PandaPickPlaceCan",
    "PandaStack",
    "PickPlaceBread",
    "PickPlaceCereal",
    "PickPlaceMilk",
]


def normalize_task_name(task_name: str) -> str:
    key = str(task_name).strip()
    if key not in TASK_ALIASES:
        raise KeyError(f"Unsupported task name: {task_name}")
    return TASK_ALIASES[key].canonical_name


def resolve_checkpoint_task_name(task_name: str) -> str:
    canonical_name = normalize_task_name(task_name)
    return TASK_ALIASES[canonical_name].checkpoint_name


def ordered_task_names(task_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if task_names is None:
        return list(DEFAULT_TASK_ORDER)
    canonical = [normalize_task_name(name) for name in task_names]
    seen = set()
    out = []
    for name in canonical:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out
