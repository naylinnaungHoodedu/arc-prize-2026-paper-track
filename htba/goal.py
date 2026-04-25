"""Reward inference from frame deltas and WIN-linked trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .encoder import ObjectSet


@dataclass(frozen=True)
class DeltaSummary:
    object_count_delta: int
    colors_added: tuple[int, ...]
    colors_removed: tuple[int, ...]
    moved_objects: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_count_delta": self.object_count_delta,
            "colors_added": list(self.colors_added),
            "colors_removed": list(self.colors_removed),
            "moved_objects": self.moved_objects,
        }


def summarize_delta(before: ObjectSet, after: ObjectSet) -> DeltaSummary:
    before_colors = set(before.colors)
    after_colors = set(after.colors)
    moved = sum(1 for obj in after.objects if obj.motion_delta is not None)
    return DeltaSummary(
        object_count_delta=after.object_count - before.object_count,
        colors_added=tuple(sorted(after_colors - before_colors)),
        colors_removed=tuple(sorted(before_colors - after_colors)),
        moved_objects=moved,
    )


@dataclass
class GoalInferer:
    """Online estimate of goal-relevant deltas.

    The model does not assert a reward before evidence exists. It stores deltas
    associated with WIN and uses exact feature overlap as a conservative
    expected-progress score.
    """

    positive_deltas: list[DeltaSummary] = field(default_factory=list)
    observations: int = 0
    wins: int = 0

    def update(self, before: ObjectSet, after: ObjectSet, win: bool) -> None:
        self.observations += 1
        if win:
            self.wins += 1
            self.positive_deltas.append(summarize_delta(before, after))

    def score_transition(self, before: ObjectSet, after: ObjectSet) -> float:
        if not self.positive_deltas:
            return 0.0
        delta = summarize_delta(before, after)
        score = 0.0
        for positive in self.positive_deltas:
            local = 0.0
            if delta.object_count_delta == positive.object_count_delta:
                local += 1.0
            if delta.colors_added == positive.colors_added:
                local += 1.0
            if delta.colors_removed == positive.colors_removed:
                local += 1.0
            if delta.moved_objects == positive.moved_objects:
                local += 1.0
            score = max(score, local / 4.0)
        return score

    def estimate(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "wins": self.wins,
            "positive_deltas": [delta.to_dict() for delta in self.positive_deltas],
            "status": "uninferred" if not self.positive_deltas else "win_condition_delta_observed",
        }
