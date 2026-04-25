"""Hybrid Theory-Based Agent package for the ARC Prize 2026 Paper Track.

The package implements the proof-of-concept scaffold described in the local
strategic blueprint. It is deliberately offline-first and exposes auditable
reasoning traces for every action decision.
"""

from .agent import HTBAAgent
from .actions import ActionCandidate, action_candidates, canonical_action_name
from .arc_adapter import (
    PreflightError,
    offline_preflight,
    probe_alive_actions,
    run_required_official_evaluation,
)

DEFAULT_SEED = 0xA6C16E26
DEFAULT_BEAM_WIDTH = 64
DEFAULT_EIG_SAMPLES = 8
DEFAULT_PLAN_DEPTH = 6

__all__ = [
    "ActionCandidate",
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_EIG_SAMPLES",
    "DEFAULT_PLAN_DEPTH",
    "DEFAULT_SEED",
    "HTBAAgent",
    "PreflightError",
    "action_candidates",
    "canonical_action_name",
    "offline_preflight",
    "probe_alive_actions",
    "run_required_official_evaluation",
]
