import os

import numpy as np
import pytest

from htba import HTBAAgent
from htba.arc_adapter import PreflightError, offline_preflight
from htba.encoder import encode_frame
from htba.goal import GoalInferer


class DummyOfficialGameShape:
    action_space = ["ACTION1", "ACTION2"]


def test_reward_inference_records_only_win_linked_deltas():
    before_frame = np.zeros((6, 6), dtype=int)
    before_frame[2, 2] = 1
    after_frame = np.zeros((6, 6), dtype=int)
    after_frame[1, 2] = 1

    before = encode_frame(before_frame)
    after = encode_frame(after_frame, previous=before)
    goal = GoalInferer()
    goal.update(before, after, win=False)
    assert goal.estimate()["status"] == "uninferred"

    goal.update(before, after, win=True)
    assert goal.estimate()["status"] == "win_condition_delta_observed"
    assert goal.score_transition(before, after) > 0.0


def test_agent_public_interface_emits_required_trace_fields():
    frame = np.zeros((8, 8), dtype=int)
    frame[2:4, 2:4] = 1
    next_frame = np.zeros((8, 8), dtype=int)
    next_frame[1:3, 2:4] = 1
    agent = HTBAAgent(seed=0xA6C16E26, beam_width=64, eig_samples=8)
    agent.reset(DummyOfficialGameShape())

    action = agent.act(frame)
    agent.observe(action, next_frame, win=False)
    trace = agent.reasoning_trace()

    assert trace
    for key in [
        "observation",
        "hypothesis",
        "transformation",
        "validation",
        "posterior",
        "selected_action",
        "rationale",
        "failure_flags",
    ]:
        assert key in trace[0]


def test_offline_preflight_fails_clearly_without_required_resources(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_AGI_DATA_DIR", raising=False)
    with pytest.raises(PreflightError, match="Official ARC-AGI-3 data directory is required"):
        offline_preflight(root=tmp_path, require_toolkit=True, require_data=True)


@pytest.mark.integration
def test_official_preflight_when_local_resources_are_configured():
    if "ARC_AGI_DATA_DIR" not in os.environ:
        pytest.skip("Set ARC_AGI_DATA_DIR and install local arc_agi to run official integration.")
    result = offline_preflight(require_toolkit=True, require_data=True)
    assert result["arc_agi_available"] is True
