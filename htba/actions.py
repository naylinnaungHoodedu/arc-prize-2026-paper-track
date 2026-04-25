"""Canonical ARC action handling and coordinate candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from .encoder import ObjectSet


ACTION_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "INTERACT": "ACTION5",
    "ACT": "ACTION5",
    "CLICK": "ACTION6",
    "UNDO": "ACTION7",
}

CANONICAL_ACTIONS = {
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
}
COORDINATE_ACTIONS = {"ACTION6"}
ACTION_WITH_COORDS_RE = re.compile(r"^(ACTION6|CLICK)\s*[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]$", re.I)


@dataclass(frozen=True)
class ActionCandidate:
    """A reviewer-readable action candidate.

    The public agent still returns a canonical string so unknown official
    toolkit action classes are not invented. Coordinate candidates keep the
    click point explicit in the string representation and in traces.
    """

    name: str
    coords: tuple[int, int] | None = None
    source: str | None = None

    @property
    def key(self) -> str:
        if self.coords is None:
            return self.name
        x, y = self.coords
        return f"{self.name}({x},{y})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coords": list(self.coords) if self.coords is not None else None,
            "key": self.key,
            "source": self.source,
        }

    def __str__(self) -> str:
        return self.key


def canonical_action_name(action: Any) -> str:
    """Return the toolkit-independent action name for a raw action value."""

    if isinstance(action, ActionCandidate):
        return action.name
    if isinstance(action, dict):
        raw = action.get("action", action.get("name", action.get("type", "")))
    elif hasattr(action, "value") and str(getattr(action, "value")).upper().startswith("ACTION"):
        raw = getattr(action, "value")
    elif isinstance(action, (tuple, list)) and action:
        raw = action[0]
    else:
        raw = getattr(action, "name", action)

    text = str(raw).strip()
    match = ACTION_WITH_COORDS_RE.match(text)
    if match:
        text = match.group(1)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.upper().replace("-", "_").replace(" ", "_")
    text = ACTION_ALIASES.get(text, text)
    return text


def parse_action_candidate(action: Any) -> ActionCandidate:
    """Convert a raw action into an ActionCandidate if coordinates are known."""

    if isinstance(action, ActionCandidate):
        return action
    if isinstance(action, dict):
        name = canonical_action_name(action)
        if "x" in action and "y" in action:
            return ActionCandidate(name=name, coords=(int(action["x"]), int(action["y"])), source=repr(action))
        return ActionCandidate(name=name, source=repr(action))
    if isinstance(action, (tuple, list)) and action:
        name = canonical_action_name(action)
        if len(action) >= 3:
            return ActionCandidate(name=name, coords=(int(action[1]), int(action[2])), source=repr(action))
        return ActionCandidate(name=name, source=repr(action))

    text = str(getattr(action, "name", action)).strip()
    match = ACTION_WITH_COORDS_RE.match(text)
    if match:
        return ActionCandidate(
            name=canonical_action_name(match.group(1)),
            coords=(int(match.group(2)), int(match.group(3))),
            source=text,
        )
    return ActionCandidate(name=canonical_action_name(action), source=text)


def normalize_actions(actions: Iterable[Any]) -> tuple[str, ...]:
    """Normalize a toolkit action collection into canonical action names."""

    normalized = tuple(
        action
        for action in sorted({canonical_action_name(action) for action in actions})
        if action in CANONICAL_ACTIONS
    )
    if not normalized:
        raise ValueError("At least one alive action is required.")
    return normalized


def action_candidates(
    actions: Iterable[Any],
    objects: ObjectSet | None = None,
    max_coordinate_candidates: int = 30,
) -> tuple[ActionCandidate, ...]:
    """Generate bounded action candidates, including ACTION6 click points."""

    base = normalize_actions(actions)
    candidates: list[ActionCandidate] = []
    for name in base:
        if name in COORDINATE_ACTIONS:
            for coords in _coordinate_candidates(objects, max_coordinate_candidates):
                candidates.append(ActionCandidate(name=name, coords=coords, source="object/frame candidate"))
        else:
            candidates.append(ActionCandidate(name=name, source="alive action"))
    if not candidates:
        raise ValueError("No action candidates could be generated.")
    return tuple(candidates)


def _coordinate_candidates(
    objects: ObjectSet | None,
    max_coordinate_candidates: int,
) -> tuple[tuple[int, int], ...]:
    if objects is None:
        return ((32, 32),)

    height, width = objects.frame_shape
    points: list[tuple[int, int]] = []
    for obj in objects.objects:
        row, col = obj.centroid
        points.append((round(col), round(row)))
        min_r, min_c, max_r, max_c = obj.bbox
        points.append((round((min_c + max_c) / 2), round((min_r + max_r) / 2)))
    points.append((width // 2, height // 2))

    clipped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in points:
        point = (int(max(0, min(width - 1, x))), int(max(0, min(height - 1, y))))
        if point in seen:
            continue
        seen.add(point)
        clipped.append(point)
        if len(clipped) >= max_coordinate_candidates:
            break
    return tuple(clipped)
