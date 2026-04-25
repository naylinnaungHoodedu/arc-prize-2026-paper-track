"""Offline preflight and official ARC-AGI-3 adapter seams."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import socket
import json
from typing import Any, Callable, Iterable


class PreflightError(RuntimeError):
    """Raised when required offline official ARC resources are unavailable."""


LOCALHOST_NAMES = {"localhost", "127.0.0.1", "::1", ""}


def _resolve_data_dir(root: Path | None = None) -> Path:
    env_dir = os.environ.get("ARC_AGI_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    base = root or Path.cwd()
    return (base / "data" / "arc_agi_3").resolve()


def _install_socket_guard() -> None:
    if getattr(socket, "_htba_guard_installed", False):
        return
    original_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else ""
        if str(host) not in LOCALHOST_NAMES:
            raise PreflightError(f"Network access blocked by HTBA offline guard: {host!r}")
        return original_connect(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket._htba_guard_installed = True  # type: ignore[attr-defined]


def offline_preflight(
    root: Path | str | None = None,
    require_toolkit: bool = True,
    require_data: bool = True,
    block_network: bool = True,
) -> dict[str, Any]:
    """Require local official ARC resources and install the offline guard."""

    root_path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    if block_network:
        _install_socket_guard()

    data_dir = _resolve_data_dir(root_path)
    if require_data and not data_dir.exists():
        raise PreflightError(
            "Official ARC-AGI-3 data directory is required. Set ARC_AGI_DATA_DIR "
            f"or create {data_dir}."
        )

    toolkit_found = importlib.util.find_spec("arc_agi") is not None
    if require_toolkit and not toolkit_found:
        raise PreflightError(
            "Local official arc_agi toolkit import is required. Install or vendor "
            "the toolkit before executing the notebook."
        )

    return {
        "root": str(root_path),
        "data_dir": str(data_dir),
        "arc_agi_available": toolkit_found,
        "network_guard": bool(block_network),
    }


def probe_alive_actions(game: Any) -> tuple[str, ...]:
    """Read the alive action subset from official toolkit metadata."""

    candidates = []
    for attr in ("action_space", "available_actions", "actions", "alive_actions"):
        if hasattr(game, attr):
            value = getattr(game, attr)
            value = value() if callable(value) else value
            candidates.extend(_flatten_actions(value))
            if candidates:
                return tuple(sorted({str(action) for action in candidates}))

    if hasattr(game, "get_action_space") and callable(game.get_action_space):
        candidates.extend(_flatten_actions(game.get_action_space()))
    if candidates:
        return tuple(sorted({str(action) for action in candidates}))

    raise PreflightError(
        "Could not discover alive actions from official game metadata. Expected "
        "action_space, available_actions, actions, alive_actions, or get_action_space()."
    )


def _flatten_actions(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "n"):
        return [f"ACTION{i}" for i in range(int(value.n))]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


@dataclass(frozen=True)
class OfficialRunConfig:
    root: Path = Path.cwd()
    game_count: int = 3


def discover_official_loader(data_dir: Path | str | None = None) -> Callable[..., Any]:
    """Find a local game loader exposed by the official toolkit.

    The official toolkit is not present in this folder, so this function avoids
    asserting a single API shape. It tries common local-loader entry points and
    raises a clear error if none are available.
    """

    if importlib.util.find_spec("arc_agi") is None:
        raise PreflightError("arc_agi is not importable; official loader discovery cannot run.")

    module_names = [
        "arc_agi",
        "arc_agi.local",
        "arc_agi.games",
        "arc_agi.envs",
        "arc_agi.benchmark",
    ]
    function_names = [
        "load_games",
        "load_local_games",
        "make_games",
        "make_envs",
        "load_environments",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for function_name in function_names:
            fn = getattr(module, function_name, None)
            if callable(fn):
                return _wrap_loader(fn, data_dir)

    raise PreflightError(
        "arc_agi import succeeded, but no known local game loader was found. "
        "Expose load_games(data_dir=...) or adapt htba.arc_adapter.discover_official_loader."
    )


def _wrap_loader(fn: Callable[..., Any], data_dir: Path | str | None) -> Callable[..., Any]:
    def loader(count: int = 3) -> Any:
        signature = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        if data_dir is not None:
            for name in ("data_dir", "root", "path", "dataset_dir"):
                if name in signature.parameters:
                    kwargs[name] = str(data_dir)
                    break
        for name in ("count", "limit", "n_games"):
            if name in signature.parameters:
                kwargs[name] = count
                break
        return fn(**kwargs)

    return loader


def iter_first_games(games: Any, count: int) -> Iterable[Any]:
    if isinstance(games, dict):
        iterator = iter(games.values())
    else:
        iterator = iter(games)
    yielded = 0
    for game in iterator:
        yield game
        yielded += 1
        if yielded >= count:
            return


def run_required_official_evaluation(
    agent: Any,
    root: Path | str | None = None,
    game_count: int = 3,
    max_actions_per_game: int = 1000,
) -> dict[str, Any]:
    """Run the agent on required local official games and write artifacts.

    This function intentionally has no synthetic fallback. If the official
    toolkit exposes a direct evaluation hook, it uses that. Otherwise it tries
    a minimal reset/step/done loop common to interactive environments.
    """

    preflight = offline_preflight(root=root, require_toolkit=True, require_data=True)
    data_dir = Path(preflight["data_dir"])
    loader = discover_official_loader(data_dir)
    games = list(iter_first_games(loader(count=game_count), game_count))
    if len(games) < game_count:
        raise PreflightError(
            f"Official loader returned {len(games)} games; {game_count} are required."
        )

    rows: list[dict[str, Any]] = []
    for index, game in enumerate(games, start=1):
        rows.append(_run_one_game(agent, game, index, max_actions_per_game))

    scorecard = {
        "preflight": preflight,
        "game_count": len(rows),
        "rows": rows,
    }
    out_dir = Path(root or Path.cwd()) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return scorecard


def _run_one_game(agent: Any, game: Any, index: int, max_actions: int) -> dict[str, Any]:
    if hasattr(game, "evaluate_agent") and callable(game.evaluate_agent):
        result = game.evaluate_agent(agent)
        return {
            "game_index": index,
            "mode": "official_evaluate_agent",
            "result": _jsonable(result),
            "agent_scorecard": agent.scorecard() if hasattr(agent, "scorecard") else {},
        }

    if hasattr(game, "run_agent") and callable(game.run_agent):
        result = game.run_agent(agent)
        return {
            "game_index": index,
            "mode": "official_run_agent",
            "result": _jsonable(result),
            "agent_scorecard": agent.scorecard() if hasattr(agent, "scorecard") else {},
        }

    if not all(hasattr(game, attr) for attr in ("reset", "step")):
        raise PreflightError(
            "Loaded official game does not expose evaluate_agent, run_agent, "
            "or reset/step methods."
        )

    agent.reset(game)
    frame = game.reset()
    actions = 0
    wins = 0
    done = False
    while not done and actions < max_actions:
        action = agent.act(frame)
        outcome = game.step(action)
        frame, win, done = _parse_step_outcome(outcome)
        agent.observe(action, frame, win=win)
        actions += 1
        wins += int(bool(win))
        if hasattr(game, "done"):
            value = getattr(game, "done")
            done = bool(value() if callable(value) else value)

    result = game.scorecard() if hasattr(game, "scorecard") and callable(game.scorecard) else {}
    return {
        "game_index": index,
        "mode": "generic_reset_step_loop",
        "actions": actions,
        "wins": wins,
        "done": bool(done),
        "result": _jsonable(result),
        "agent_scorecard": agent.scorecard(),
    }


def _parse_step_outcome(outcome: Any) -> tuple[Any, bool, bool]:
    if isinstance(outcome, dict):
        frame = outcome.get("frame", outcome.get("observation", outcome.get("state")))
        win = bool(outcome.get("win", outcome.get("won", False)))
        done = bool(outcome.get("done", outcome.get("terminated", win)))
        return frame, win, done
    if isinstance(outcome, tuple):
        if len(outcome) >= 5:
            frame, _reward, terminated, truncated, info = outcome[:5]
            win = bool(getattr(info, "get", lambda _key, default=None: default)("win", False))
            return frame, win, bool(terminated or truncated)
        if len(outcome) >= 4:
            frame, _reward, done, info = outcome[:4]
            win = bool(getattr(info, "get", lambda _key, default=None: default)("win", False))
            return frame, win, bool(done)
        if len(outcome) == 3:
            frame, win, done = outcome
            return frame, bool(win), bool(done)
        if len(outcome) == 2:
            frame, win = outcome
            return frame, bool(win), bool(win)
    raise PreflightError("Could not parse official game step outcome.")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        return repr(value)
