import math

import numpy as np

from htba.dsl import RuleProgram
from htba.encoder import encode_frame
from htba.hypothesis import HypothesisBeam, HypothesisEntry
from htba.goal import GoalInferer
from htba.planner import Planner


def _frame_with_square(row=2, col=2):
    frame = np.zeros((8, 8), dtype=int)
    frame[row : row + 2, col : col + 2] = 1
    return frame


def test_mdl_prefers_shorter_equivalent_program():
    current = encode_frame(_frame_with_square())
    unchanged = encode_frame(_frame_with_square(), previous=current)
    short = RuleProgram("short_identity", "identity", description_length=1.0)
    long = RuleProgram("long_identity", "identity", description_length=5.0)
    beam = HypothesisBeam(
        entries=[HypothesisEntry(short), HypothesisEntry(long)],
        beam_width=2,
        tau=1.0,
        prune_ratio=1e-12,
    )

    for _ in range(20):
        beam = beam.update(current, "ACTION1", unchanged)

    probabilities = {entry.program.name: math.exp(entry.log_weight) for entry in beam.entries}
    assert probabilities["short_identity"] > 0.95


def test_posterior_renormalizes_to_one():
    current = encode_frame(_frame_with_square())
    next_frame = encode_frame(_frame_with_square(row=1), previous=current)
    beam = HypothesisBeam.initialize(["ACTION1"], current, beam_width=64)

    beam = beam.update(current, "ACTION1", next_frame)

    assert abs(sum(beam.probabilities()) - 1.0) < 1e-9


def test_beam_prune_keeps_bounded_width_and_does_not_resurrect():
    current = encode_frame(_frame_with_square())
    unchanged = encode_frame(_frame_with_square(), previous=current)
    entries = [
        HypothesisEntry(RuleProgram(f"identity_{idx}", "identity", description_length=1.0 + idx))
        for idx in range(100)
    ]
    beam = HypothesisBeam(entries=entries, beam_width=64, prune_ratio=1e-6)

    beam = beam.update(current, "ACTION1", unchanged)
    names_after_first = {entry.program.name for entry in beam.entries}
    beam = beam.update(current, "ACTION1", unchanged)
    names_after_second = {entry.program.name for entry in beam.entries}

    assert len(beam.entries) <= 64
    assert names_after_second.issubset(names_after_first)


def test_eig_is_zero_when_all_hypotheses_agree():
    current = encode_frame(_frame_with_square())
    entries = [
        HypothesisEntry(RuleProgram(f"identity_{idx}", "identity", description_length=1.0 + idx * 0.1))
        for idx in range(3)
    ]
    beam = HypothesisBeam(entries=entries, beam_width=3, prune_ratio=1e-12)
    planner = Planner(["ACTION1"], entropy_threshold=0.0, eig_samples=8, seed=123)

    assert planner.information_gain("ACTION1", current, beam) <= 1e-6


def test_planner_is_deterministic_for_same_seed():
    current = encode_frame(_frame_with_square())
    beam = HypothesisBeam.initialize(["ACTION1", "ACTION2"], current, beam_width=64)
    goal = GoalInferer()
    first = Planner(["ACTION1", "ACTION2"], entropy_threshold=0.0, eig_samples=8, seed=123)
    second = Planner(["ACTION1", "ACTION2"], entropy_threshold=0.0, eig_samples=8, seed=123)

    assert first.choose_action(current, beam, goal).action == second.choose_action(current, beam, goal).action
