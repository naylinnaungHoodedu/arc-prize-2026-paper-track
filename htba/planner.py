"""Information-gain-aware action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .dsl import normalize_actions
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
    ) -> None:
        self.alive_actions = normalize_actions(alive_actions)
        self.entropy_threshold = float(entropy_threshold)
        self.eig_samples = int(eig_samples)
        self.rng = np.random.default_rng(seed)

    def information_gain(self, action: Any, current: ObjectSet, beam: HypothesisBeam) -> float:
        action_name = str(action)
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

    def expected_progress(self, action: Any, current: ObjectSet, beam: HypothesisBeam, goal: GoalInferer) -> float:
        action_name = str(action)
        if action_name not in self.alive_actions:
            return float("-inf")
        map_prediction = beam.map_entry().program.predict(current, action_name)
        return goal.score_transition(current, map_prediction)

    def choose_action(self, current: ObjectSet, beam: HypothesisBeam, goal: GoalInferer) -> ActionDecision:
        entropy = beam.entropy()
        eig_by_action = {
            action: round(self.information_gain(action, current, beam), 6)
            for action in self.alive_actions
        }
        progress_by_action = {
            action: round(self.expected_progress(action, current, beam, goal), 6)
            for action in self.alive_actions
        }

        if entropy > self.entropy_threshold:
            action = max(self.alive_actions, key=lambda item: (eig_by_action[item], -self.alive_actions.index(item)))
            return ActionDecision(
                action=action,
                mode="explore",
                eig_by_action=eig_by_action,
                progress_by_action=progress_by_action,
                rationale=(
                    "posterior entropy exceeds threshold; selected action maximizes "
                    "expected information gain"
                ),
            )

        action = max(self.alive_actions, key=lambda item: (progress_by_action[item], -self.alive_actions.index(item)))
        return ActionDecision(
            action=action,
            mode="exploit",
            eig_by_action=eig_by_action,
            progress_by_action=progress_by_action,
            rationale=(
                "posterior entropy is below threshold; selected action maximizes "
                "expected progress under inferred reward"
            ),
        )
