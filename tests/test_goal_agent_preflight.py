import json
import os
import importlib.util
import importlib.machinery
import sys
import types

import numpy as np
import pytest

from htba import HTBAAgent
from htba.arc_adapter import PreflightError, offline_preflight, run_required_official_evaluation
from htba.encoder import encode_frame
from htba.goal import GoalInferer


class DummyOfficialGameShape:
    action_space = ["ACTION1", "ACTION2"]


def _deterministic_agent_payload():
    frame = np.zeros((8, 8), dtype=int)
    frame[2:4, 2:4] = 1
    next_frame = np.zeros((8, 8), dtype=int)
    next_frame[1:3, 2:4] = 1

    agent = HTBAAgent(seed=0xA6C16E26, beam_width=64, eig_samples=8, plan_depth=6)
    agent.reset(DummyOfficialGameShape())
    for _ in range(2):
        action = agent.act(frame)
        agent.observe(action, next_frame, win=False)
        frame, next_frame = next_frame, frame
    return {
        "scorecard": agent.scorecard(),
        "trace": agent.reasoning_trace(),
    }


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


def test_seeded_dry_run_is_byte_identical():
    first = json.dumps(_deterministic_agent_payload(), sort_keys=True)
    second = json.dumps(_deterministic_agent_payload(), sort_keys=True)

    assert first == second


def test_offline_preflight_fails_clearly_without_required_resources(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_AGI_DATA_DIR", raising=False)
    with pytest.raises(PreflightError, match="Official ARC-AGI-3 data directory is required"):
        offline_preflight(root=tmp_path, require_toolkit=True, require_data=True)


@pytest.mark.integration
def test_official_preflight_when_local_resources_are_configured():
    if importlib.util.find_spec("arc_agi") is None:
        pytest.skip("Install or attach local arc_agi to run official integration.")
    result = offline_preflight(require_toolkit=True, require_data=True)
    assert result["arc_agi_available"] is True


@pytest.mark.integration
def test_official_three_game_run_when_local_resources_are_configured():
    if importlib.util.find_spec("arc_agi") is None:
        pytest.skip("Install or attach local arc_agi to run official integration.")
    agent = HTBAAgent(seed=0xA6C16E26, beam_width=64, eig_samples=8, plan_depth=6)
    result = run_required_official_evaluation(agent=agent, game_count=None, max_actions_per_game=1000)
    assert result["game_count"] >= 1


def test_competition_mode_adapter_with_mocked_arcade(tmp_path, monkeypatch):
    class FakeOperationMode:
        COMPETITION = "competition"

    class FakeEnvInfo:
        game_id = "fake"
        title = "Fake ARC environment"

    class FakeObservation:
        def __init__(self, frame, state="NOT_FINISHED"):
            self.frame = frame
            self.state = state

    class FakeEnv:
        def __init__(self):
            self.frame = np.zeros((8, 8), dtype=int)
            self.frame[2:4, 2:4] = 1
            self._has_observation = False

        @property
        def action_space(self):
            return ["ACTION1"] if self._has_observation else []

        def reset(self):
            self._has_observation = True
            return FakeObservation(self.frame)

        def step(self, action, data=None):
            next_frame = np.zeros((8, 8), dtype=int)
            next_frame[1:3, 2:4] = 1
            return FakeObservation(next_frame, state="GAME_OVER")

    class FakeArcade:
        def __init__(self, operation_mode):
            self.operation_mode = operation_mode

        def get_environments(self):
            return [FakeEnvInfo()]

        def make(self, game_id):
            assert game_id == "fake"
            return FakeEnv()

        def get_scorecard(self):
            raise RuntimeError("in-flight scorecard unavailable")

    fake_arc_agi = types.ModuleType("arc_agi")
    fake_arc_agi.__spec__ = importlib.machinery.ModuleSpec("arc_agi", loader=None)
    fake_arc_agi.Arcade = FakeArcade
    fake_arc_agi.OperationMode = FakeOperationMode

    fake_arcengine = types.ModuleType("arcengine")
    fake_arcengine.__spec__ = importlib.machinery.ModuleSpec("arcengine", loader=None)
    fake_arcengine.GameAction = types.SimpleNamespace(
        RESET="RESET",
        ACTION1="ACTION1",
        ACTION2="ACTION2",
        ACTION3="ACTION3",
        ACTION4="ACTION4",
        ACTION5="ACTION5",
        ACTION6="ACTION6",
        ACTION7="ACTION7",
    )

    monkeypatch.setitem(sys.modules, "arc_agi", fake_arc_agi)
    monkeypatch.setitem(sys.modules, "arcengine", fake_arcengine)

    agent = HTBAAgent(seed=0xA6C16E26, beam_width=64, eig_samples=8, plan_depth=6)
    result = run_required_official_evaluation(
        agent=agent,
        root=tmp_path,
        game_count=1,
        max_actions_per_game=3,
    )

    assert result["mode"] == "arcade_competition"
    assert result["game_count"] == 1
    assert (tmp_path / "out" / "scorecard.json").exists()
    assert (tmp_path / "out" / "reasoning_trace.json").exists()
