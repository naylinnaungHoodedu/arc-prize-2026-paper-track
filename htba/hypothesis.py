"""Bayesian/MDL hypothesis management."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .actions import canonical_action_name
from .dsl import RuleProgram, compose_programs, generate_initial_programs
from .encoder import ObjectSet, object_distance


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("-inf")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


@dataclass(frozen=True)
class HypothesisEntry:
    program: RuleProgram
    cumulative_log_likelihood: float = 0.0
    log_weight: float = 0.0

    def posterior_score(self, tau: float) -> float:
        return self.cumulative_log_likelihood - self.program.description_length / tau

    def to_dict(self) -> dict[str, Any]:
        payload = self.program.to_dict()
        payload.update(
            {
                "cumulative_log_likelihood": round(self.cumulative_log_likelihood, 6),
                "log_weight": round(self.log_weight, 6),
                "probability": round(math.exp(self.log_weight), 6),
            }
        )
        return payload


class HypothesisBeam:
    """A bounded posterior over symbolic rule programs."""

    def __init__(
        self,
        entries: Iterable[HypothesisEntry],
        beam_width: int = 64,
        tau: float = 1.0,
        prune_ratio: float = 1e-6,
        ambiguity_delta: float = 0.05,
        noise_scale: float = 1.0,
    ) -> None:
        self.beam_width = int(beam_width)
        self.tau = float(tau)
        self.prune_ratio = float(prune_ratio)
        self.ambiguity_delta = float(ambiguity_delta)
        self.noise_scale = float(noise_scale)
        self.entries = self._normalize(list(entries))[: self.beam_width]

    @classmethod
    def initialize(
        cls,
        actions: Iterable[Any],
        objects: ObjectSet,
        beam_width: int = 64,
        tau: float = 1.0,
    ) -> "HypothesisBeam":
        programs = generate_initial_programs(actions, objects)
        entries = [HypothesisEntry(program=program) for program in programs]
        return cls(entries=entries, beam_width=beam_width, tau=tau)

    def _normalize(self, entries: list[HypothesisEntry]) -> list[HypothesisEntry]:
        if not entries:
            raise ValueError("HypothesisBeam requires at least one entry.")

        scored = [
            HypothesisEntry(
                program=entry.program,
                cumulative_log_likelihood=entry.cumulative_log_likelihood,
                log_weight=entry.posterior_score(self.tau),
            )
            for entry in entries
        ]
        normalizer = _logsumexp(entry.log_weight for entry in scored)
        normalized = [
            HypothesisEntry(
                program=entry.program,
                cumulative_log_likelihood=entry.cumulative_log_likelihood,
                log_weight=entry.log_weight - normalizer,
            )
            for entry in scored
        ]
        normalized.sort(key=lambda entry: entry.log_weight, reverse=True)
        if normalized:
            map_weight = normalized[0].log_weight
            cutoff = map_weight + math.log(self.prune_ratio)
            normalized = [entry for entry in normalized if entry.log_weight >= cutoff]
        return normalized[: self.beam_width]

    def update(self, current: ObjectSet, action: Any, observed_next: ObjectSet) -> "HypothesisBeam":
        action_name = canonical_action_name(action)
        updated: list[HypothesisEntry] = []
        for entry in self.entries:
            prediction = entry.program.predict(current, action_name)
            mismatch = object_distance(prediction, observed_next)
            incremental_ll = -mismatch / self.noise_scale
            updated.append(
                HypothesisEntry(
                    program=entry.program,
                    cumulative_log_likelihood=entry.cumulative_log_likelihood + incremental_ll,
                )
            )

        candidate = HypothesisBeam(
            entries=updated,
            beam_width=self.beam_width,
            tau=self.tau,
            prune_ratio=self.prune_ratio,
            ambiguity_delta=self.ambiguity_delta,
            noise_scale=self.noise_scale,
        )
        if candidate.is_ambiguous() and len(candidate.entries) < self.beam_width:
            expansion = compose_programs(
                (entry.program for entry in candidate.entries),
                limit=self.beam_width - len(candidate.entries),
            )
            if expansion:
                candidate = HypothesisBeam(
                    entries=list(candidate.entries)
                    + [HypothesisEntry(program=program) for program in expansion],
                    beam_width=self.beam_width,
                    tau=self.tau,
                    prune_ratio=self.prune_ratio,
                    ambiguity_delta=self.ambiguity_delta,
                    noise_scale=self.noise_scale,
                )
        return candidate

    def posterior_after(self, current: ObjectSet, action: Any, observed_next: ObjectSet) -> "HypothesisBeam":
        return self.update(current=current, action=action, observed_next=observed_next)

    def entropy(self) -> float:
        return -sum(math.exp(entry.log_weight) * entry.log_weight for entry in self.entries)

    def probabilities(self) -> list[float]:
        return [math.exp(entry.log_weight) for entry in self.entries]

    def map_entry(self) -> HypothesisEntry:
        return max(self.entries, key=lambda entry: entry.log_weight)

    def is_ambiguous(self) -> bool:
        if len(self.entries) < 2:
            return False
        return abs(self.entries[0].log_weight - self.entries[1].log_weight) < self.ambiguity_delta

    def prediction_distribution(self, current: ObjectSet, action: Any) -> dict[tuple[Any, ...], tuple[float, ObjectSet]]:
        distribution: dict[tuple[Any, ...], tuple[float, ObjectSet]] = {}
        action_name = canonical_action_name(action)
        for entry in self.entries:
            probability = math.exp(entry.log_weight)
            prediction = entry.program.predict(current, action_name)
            signature = prediction.signature()
            old_probability, _ = distribution.get(signature, (0.0, prediction))
            distribution[signature] = (old_probability + probability, prediction)
        return distribution

    def summary(self, top_n: int = 5) -> dict[str, Any]:
        return {
            "beam_width": len(self.entries),
            "entropy": round(self.entropy(), 6),
            "top": [entry.to_dict() for entry in self.entries[:top_n]],
        }
