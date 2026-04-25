import numpy as np

from htba.actions import action_candidates, canonical_action_name
from htba.encoder import encode_frame
from htba.goal import GoalInferer
from htba.hypothesis import HypothesisBeam
from htba.planner import Planner


def test_action_canonicalization_for_arc_variants():
    assert canonical_action_name("up") == "ACTION1"
    assert canonical_action_name("Action.ACTION2") == "ACTION2"
    assert canonical_action_name("interact") == "ACTION5"
    assert canonical_action_name("CLICK(7, 9)") == "ACTION6"
    assert canonical_action_name("undo") == "ACTION7"


def test_action6_coordinate_candidates_use_objects_and_frame_center():
    frame = np.zeros((64, 64), dtype=int)
    frame[10:14, 20:24] = 2
    objects = encode_frame(frame)

    candidates = action_candidates(["ACTION6"], objects=objects, max_coordinate_candidates=30)
    keys = {candidate.key for candidate in candidates}

    assert "ACTION6(22,12)" in keys
    assert "ACTION6(32,32)" in keys
    assert len(candidates) <= 30


def test_planner_never_selects_unavailable_action():
    frame = np.zeros((8, 8), dtype=int)
    frame[2:4, 2:4] = 1
    current = encode_frame(frame)
    beam = HypothesisBeam.initialize(["ACTION1"], current)
    planner = Planner(["ACTION1"], entropy_threshold=0.0, eig_samples=8, seed=123)

    decision = planner.choose_action(current, beam, GoalInferer())

    assert canonical_action_name(decision.action) == "ACTION1"
    assert "ACTION2" not in decision.eig_by_action
