"""Core-Knowledge-aligned symbolic transition programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .encoder import ObjectSet


DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "ACTION1": (-1, 0),
    "ACTION2": (1, 0),
    "ACTION3": (0, -1),
    "ACTION4": (0, 1),
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass(frozen=True)
class RuleProgram:
    """A compact symbolic transition hypothesis.

    The program vocabulary is intentionally small. It encodes objectness,
    geometry, numerosity, and action-triggered causality without game-specific
    shortcuts.
    """

    name: str
    primitive: str
    description_length: float
    trigger_action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def applies_to(self, action: str) -> bool:
        return self.trigger_action is None or str(action) == self.trigger_action

    def predict(self, objects: ObjectSet, action: str) -> ObjectSet:
        if not self.applies_to(str(action)):
            return objects

        if self.primitive == "identity":
            return objects

        if self.primitive == "translate_all":
            dy = int(self.params.get("dy", 0))
            dx = int(self.params.get("dx", 0))
            return objects.translated(dy=dy, dx=dx)

        if self.primitive == "translate_color":
            dy = int(self.params.get("dy", 0))
            dx = int(self.params.get("dx", 0))
            color = int(self.params["color"])
            return objects.translated(dy=dy, dx=dx, color=color)

        if self.primitive == "delete_color":
            color = int(self.params["color"])
            return objects.with_objects(obj for obj in objects.objects if obj.color != color)

        if self.primitive == "keep_largest":
            if not objects.objects:
                return objects
            largest = max(objects.objects, key=lambda obj: obj.size)
            return objects.with_objects([largest])

        return objects

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "primitive": self.primitive,
            "description_length": float(self.description_length),
            "trigger_action": self.trigger_action,
            "params": dict(self.params),
        }


def normalize_actions(actions: Iterable[Any]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(action) for action in actions}))
    if not normalized:
        raise ValueError("At least one alive action is required.")
    return normalized


def generate_initial_programs(actions: Iterable[Any], objects: ObjectSet) -> list[RuleProgram]:
    """Generate the bounded initial DSL beam from alive actions and objects."""

    alive = normalize_actions(actions)
    programs: list[RuleProgram] = [
        RuleProgram(
            name="identity",
            primitive="identity",
            description_length=1.0,
        )
    ]

    colors = objects.colors
    for action in alive:
        if action in DIRECTION_DELTAS:
            dy, dx = DIRECTION_DELTAS[action]
            programs.append(
                RuleProgram(
                    name=f"{action}:translate_all({dy},{dx})",
                    primitive="translate_all",
                    trigger_action=action,
                    params={"dy": dy, "dx": dx},
                    description_length=2.0,
                )
            )
            for color in colors:
                programs.append(
                    RuleProgram(
                        name=f"{action}:translate_color({color},{dy},{dx})",
                        primitive="translate_color",
                        trigger_action=action,
                        params={"color": int(color), "dy": dy, "dx": dx},
                        description_length=3.0,
                    )
                )

        if action in {"ACTION5", "INTERACT"}:
            for color in colors:
                programs.append(
                    RuleProgram(
                        name=f"{action}:delete_color({color})",
                        primitive="delete_color",
                        trigger_action=action,
                        params={"color": int(color)},
                        description_length=4.0,
                    )
                )
            programs.append(
                RuleProgram(
                    name=f"{action}:keep_largest",
                    primitive="keep_largest",
                    trigger_action=action,
                    description_length=4.5,
                )
            )

    return programs


def compose_programs(programs: Iterable[RuleProgram], limit: int) -> list[RuleProgram]:
    """Return deterministic composite candidates when the beam is ambiguous.

    This proof-of-concept keeps composition conservative: it creates only
    explicit sequence names and assigns a longer description length. Execution
    remains the first program's prediction because full program sequencing
    requires official transition semantics that are not present in this folder.
    """

    base = list(programs)
    composites: list[RuleProgram] = []
    for left in base:
        for right in base:
            if left.name == right.name:
                continue
            composites.append(
                RuleProgram(
                    name=f"seq({left.name};{right.name})",
                    primitive=left.primitive,
                    trigger_action=left.trigger_action,
                    params=dict(left.params),
                    description_length=left.description_length + right.description_length + 1.0,
                )
            )
            if len(composites) >= limit:
                return composites
    return composites
