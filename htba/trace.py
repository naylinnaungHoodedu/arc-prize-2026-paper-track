"""Auditable reasoning trace records."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class ReasoningTraceRecord:
    step: int
    observation: dict[str, Any]
    hypothesis: dict[str, Any]
    transformation: dict[str, Any]
    validation: dict[str, Any]
    posterior: dict[str, Any]
    selected_action: str
    rationale: str
    failure_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "hypothesis": self.hypothesis,
            "transformation": self.transformation,
            "validation": self.validation,
            "posterior": self.posterior,
            "selected_action": self.selected_action,
            "rationale": self.rationale,
            "failure_flags": list(self.failure_flags),
        }


class ReasoningTrace:
    def __init__(self) -> None:
        self.records: list[ReasoningTraceRecord] = []

    def append(self, record: ReasoningTraceRecord) -> None:
        self.records.append(record)

    def update_last_validation(self, validation: dict[str, Any], posterior: dict[str, Any]) -> None:
        if not self.records:
            return
        self.records[-1].validation = validation
        self.records[-1].posterior = posterior

    def to_list(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_list(), indent=indent, sort_keys=True)
