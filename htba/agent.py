"""Public HTBA agent interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .arc_adapter import probe_alive_actions
from .encoder import ObjectSet, encode_frame, object_distance
from .goal import GoalInferer
from .hypothesis import HypothesisBeam
from .memory import CrossGameMemory
from .planner import Planner
from .trace import ReasoningTrace, ReasoningTraceRecord


class HTBAAgent:
    """Hybrid Theory-Based Agent proof-of-concept implementation."""

    def __init__(
        self,
        seed: int = 0xA6C16E26,
        beam_width: int = 64,
        eig_samples: int = 8,
        entropy_threshold: float = 0.25,
        memory_dir: str | Path = "out/memory",
    ) -> None:
        self.seed = int(seed)
        self.beam_width = int(beam_width)
        self.eig_samples = int(eig_samples)
        self.entropy_threshold = float(entropy_threshold)
        self.rng = np.random.default_rng(self.seed)
        self.memory = CrossGameMemory(Path(memory_dir))
        self.trace = ReasoningTrace()
        self.goal = GoalInferer()
        self.alive_actions: tuple[str, ...] = ()
        self.planner: Planner | None = None
        self.beam: HypothesisBeam | None = None
        self.current_objects: ObjectSet | None = None
        self.actions_taken: list[str] = []
        self.wins: int = 0
        self.game: Any = None

    def reset(self, game: Any) -> None:
        self.game = game
        self.alive_actions = probe_alive_actions(game)
        self.planner = Planner(
            alive_actions=self.alive_actions,
            entropy_threshold=self.entropy_threshold,
            eig_samples=self.eig_samples,
            seed=self.seed,
        )
        self.beam = None
        self.current_objects = None
        self.trace = ReasoningTrace()
        self.goal = GoalInferer()
        self.actions_taken = []
        self.wins = 0

    def act(self, frame: Any) -> str:
        if self.planner is None:
            raise RuntimeError("Call reset(game) before act(frame).")

        previous = self.current_objects
        current = encode_frame(frame, previous=previous)
        if self.beam is None:
            self.beam = HypothesisBeam.initialize(
                actions=self.alive_actions,
                objects=current,
                beam_width=self.beam_width,
            )
        self.current_objects = current

        decision = self.planner.choose_action(current=current, beam=self.beam, goal=self.goal)
        self.actions_taken.append(decision.action)
        step = len(self.actions_taken)
        map_program = self.beam.map_entry().program
        predicted = map_program.predict(current, decision.action)
        self.trace.append(
            ReasoningTraceRecord(
                step=step,
                observation={
                    "frame": current.to_dict(),
                    "alive_actions": list(self.alive_actions),
                },
                hypothesis={
                    "map_program": map_program.to_dict(),
                    "beam": self.beam.summary(top_n=5),
                },
                transformation={
                    "candidate_action": decision.action,
                    "map_prediction": predicted.to_dict(),
                    "mode": decision.mode,
                },
                validation={"status": "pending_observation"},
                posterior=self.beam.summary(top_n=5),
                selected_action=decision.action,
                rationale=decision.rationale,
                failure_flags=[],
            )
        )
        return decision.action

    def observe(self, action: Any, next_frame: Any, win: bool = False) -> None:
        if self.current_objects is None or self.beam is None:
            raise RuntimeError("Call act(frame) before observe(action, next_frame, win).")
        observed_next = encode_frame(next_frame, previous=self.current_objects)
        before = self.current_objects
        predicted_next = self.beam.map_entry().program.predict(before, str(action))
        mismatch = object_distance(predicted_next, observed_next)
        self.beam = self.beam.update(before, str(action), observed_next)
        self.goal.update(before, observed_next, bool(win))
        self.current_objects = observed_next
        if win:
            self.wins += 1
        self.trace.update_last_validation(
            validation={
                "status": "observed",
                "win": bool(win),
                "map_prediction_mismatch": round(mismatch, 6),
                "goal_estimate": self.goal.estimate(),
            },
            posterior=self.beam.summary(top_n=5),
        )

    def scorecard(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "actions": len(self.actions_taken),
            "wins": self.wins,
            "alive_actions": list(self.alive_actions),
            "posterior_entropy": None if self.beam is None else round(self.beam.entropy(), 6),
            "trace_records": len(self.trace.records),
        }

    def reasoning_trace(self) -> list[dict[str, Any]]:
        return self.trace.to_list()
