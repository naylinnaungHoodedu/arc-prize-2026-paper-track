"""Information-gain-aware action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .actions import ActionCandidate, action_candidates, canonical_action_name, normalize_actions
from .encoder import ObjectSet
from .goal import GoalInferer
from .hypothesis import HypothesisBeam


@dataclass(frozen=True)
class ActionDecision:
    action: str
    mode: str
    rationale: str
    eig_by_action: dict[str, float]
    progress_by_action: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "mode": self.mode,
            "rationale": self.rationale,
            "eig_by_action": self.eig_by_action,
            "progress_by_action": self.progress_by_action,
        }


class Planner:
    def __init__(
        self,
        alive_actions: Iterable[Any],
        entropy_threshold: float,
        eig_samples: int,
        seed: int,
        plan_depth: int = 6,
        max_coordinate_candidates: int = 30,
    ) -> None:
        self.alive_actions = normalize_actions(alive_actions)
        self.entropy_threshold = float(entropy_threshold)
        self.eig_samples = int(eig_samples)
        self.plan_depth = max(1, int(plan_depth))
        self.max_coordinate_candidates = max(1, int(max_coordinate_candidates))
        self.rng = np.random.default_rng(seed)

    def information_gain(self, action: Any, current: ObjectSet, beam: HypothesisBeam) -> float:
        action_name = canonical_action_name(action)
        if action_name not in self.alive_actions:
            return float("-inf")

        prior_entropy = beam.entropy()
        distribution = beam.prediction_distribution(current, action_name)
        if len(distribution) <= 1:
            return 0.0

        outcomes = sorted(distribution.values(), key=lambda item: item[0], reverse=True)
        outcomes = outcomes[: self.eig_samples]
        expected_entropy = 0.0
        mass = sum(probability for probability, _ in outcomes)
        if mass <= 0:
            return 0.0
        for probability, predicted in outcomes:
            after = beam.posterior_after(current, action_name, predicted)
            expected_entropy += (probability / mass) * after.entropy()
        return max(0.0, prior_entropy - expected_entropy)

    def expected_progress(
        self,
        action: Any,
        current: ObjectSet,
        beam: HypothesisBeam,
        goal: GoalInferer,
        depth: int | None = None,
    ) -> float:
        return self._expected_progress(
            action=action,
            current=current,
            beam=beam,
            goal=goal,
            depth=self.plan_depth if depth is None else int(depth),
            memo={},
        )

    def _expected_progress(
        self,
        action: Any,
        current: ObjectSet,
        beam: HypothesisBeam,
        goal: GoalInferer,
        depth: int,
        memo: dict[tuple[str, tuple[Any, ...], int], float],
    ) -> float:
        action_name = canonical_action_name(action)
        if action_name not in self.alive_actions:
            return float("-inf")
        if not goal.positive_deltas:
            return 0.0
        key = (str(action), current.signature(), depth)
        if key in memo:
            return memo[key]
        map_prediction = beam.map_entry().program.predict(current, action_name)
        immediate = goal.score_transition(current, map_prediction)
        if depth <= 1:
            memo[key] = immediate
            return immediate

        future_candidates = self._candidates(map_prediction)
        future = max(
            self._expected_progress(candidate, map_prediction, beam, goal, depth - 1, memo)
            for candidate in future_candidates
        )
        memo[key] = immediate + 0.8 * future
        return memo[key]

    def _candidates(self, current: ObjectSet) -> tuple[ActionCandidate, ...]:
        return action_candidates(
            self.alive_actions,
            objects=current,
            max_coordinate_candidates=self.max_coordinate_candidates,
        )

    def choose_action(self, current: ObjectSet, beam: HypothesisBeam, goal: GoalInferer) -> ActionDecision:
        entropy = beam.entropy()
        candidates = self._candidates(current)
        eig_by_action = {
            candidate.key: round(self.information_gain(candidate, current, beam), 6)
            for candidate in candidates
        }
        progress_by_action = {
            candidate.key: round(self.expected_progress(candidate, current, beam, goal), 6)
            for candidate in candidates
        }

        if entropy > self.entropy_threshold:
            action = max(candidates, key=lambda item: (eig_by_action[item.key], -candidates.index(item)))
            return ActionDecision(
                action=action.key,
                mode="explore",
                eig_by_action=eig_by_action,
                progress_by_action=progress_by_action,
                rationale=(
                    "posterior entropy exceeds threshold; selected action maximizes "
                    "expected information gain"
                ),
            )

        action = max(candidates, key=lambda item: (progress_by_action[item.key], -candidates.index(item)))
        return ActionDecision(
            action=action.key,
            mode="exploit",
            eig_by_action=eig_by_action,
            progress_by_action=progress_by_action,
            rationale=(
                "posterior entropy is below threshold; selected action maximizes "
                "expected progress under inferred reward"
            ),
        )
