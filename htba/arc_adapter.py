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
import sys
from typing import Any, Callable, Iterable

from .actions import parse_action_candidate, normalize_actions


class PreflightError(RuntimeError):
    """Raised when required offline official ARC resources are unavailable."""


LOCALHOST_NAMES = {"localhost", "127.0.0.1", "::1", ""}
OFFICIAL_ARC_HOSTS = {"three.arcprize.org", "arcprize.org", "www.arcprize.org"}


def _candidate_kaggle_dirs() -> list[Path]:
    base = Path("/kaggle/input")
    candidates = [
        base / "arc-agi-3",
        base / "arc-agi3",
        base / "arc-agi-3-data",
        base / "arc-prize-2026",
        base / "arc-prize-2026-paper-track",
    ]
    if base.exists():
        for child in sorted(path for path in base.iterdir() if path.is_dir()):
            if "arc" not in child.name.lower():
                continue
            candidates.extend(
                [
                    child,
                    child / "data",
                    child / "arc_agi_3",
                    child / "data" / "arc_agi_3",
                ]
            )
    return candidates


def _first_existing_dir(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_data_dir(root: Path | None = None) -> Path:
    env_dir = os.environ.get("ARC_AGI_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    base = root or Path.cwd()
    candidates = _candidate_kaggle_dirs()
    candidates.append(base / "data" / "arc_agi_3")
    found = _first_existing_dir(candidates)
    return found if found is not None else (base / "data" / "arc_agi_3").resolve()


def _candidate_toolkit_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("ARC_AGI_TOOLKIT_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            Path("/kaggle/input/arc-agi-toolkit"),
            Path("/kaggle/input/arc-agi-3-toolkit"),
            Path("/kaggle/input/arc-agi"),
            root / "arc_agi",
            root / "vendor" / "arc_agi",
        ]
    )
    for kaggle_dir in _candidate_kaggle_dirs():
        candidates.extend([kaggle_dir, kaggle_dir / "src", kaggle_dir / "python"])
    return candidates


def _ensure_toolkit_on_path(root: Path) -> str | None:
    for candidate in _candidate_toolkit_dirs(root):
        if not candidate.exists() or not candidate.is_dir():
            continue
        if candidate.name == "arc_agi" and (candidate / "__init__.py").exists():
            resolved_parent = str(candidate.parent.resolve())
            if resolved_parent not in sys.path:
                sys.path.insert(0, resolved_parent)
            return str(candidate.resolve())
        import_roots = [candidate, candidate / "src", candidate / "python"]
        for import_root in import_roots:
            if not import_root.exists() or not import_root.is_dir():
                continue
            if (import_root / "arc_agi").exists() or (import_root / "arc_agi.py").exists():
                resolved = str(import_root.resolve())
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)
                return str(candidate.resolve())
    return None


def _installed_toolkit_path() -> str | None:
    spec = importlib.util.find_spec("arc_agi")
    if spec is None:
        return None
    if spec.origin:
        return str(Path(spec.origin).resolve().parent)
    if spec.submodule_search_locations:
        first = next(iter(spec.submodule_search_locations), None)
        if first:
            return str(Path(first).resolve())
    return None


def _allowed_socket_hosts() -> set[str]:
    configured = {
        host.strip()
        for host in os.environ.get("HTBA_ALLOWED_SOCKET_HOSTS", "").split(",")
        if host.strip()
    }
    return LOCALHOST_NAMES | OFFICIAL_ARC_HOSTS | configured


def _install_socket_guard() -> None:
    """Install a conservative socket guard for local offline checks.

    Official Kaggle ARC-AGI-3 evaluation is forced through toolkit competition
    mode, which may use ARC-managed endpoints. The production submission path
    therefore does not enable this guard by default. The guard remains available
    for local audits that want to prove no arbitrary network host is contacted.
    """

    if getattr(socket, "_htba_guard_installed", False):
        return
    original_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else ""
        if str(host) not in _allowed_socket_hosts():
            raise PreflightError(f"Network access blocked by HTBA offline guard: {host!r}")
        return original_connect(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket._htba_guard_installed = True  # type: ignore[attr-defined]


def offline_preflight(
    root: Path | str | None = None,
    require_toolkit: bool = True,
    require_data: bool = True,
    block_network: bool = False,
) -> dict[str, Any]:
    """Check official ARC resources and optionally install the local guard."""

    root_path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    if block_network:
        _install_socket_guard()

    data_dir = _resolve_data_dir(root_path)
    if require_data and not data_dir.exists():
        raise PreflightError(
            "Official ARC-AGI-3 data directory is required. Set ARC_AGI_DATA_DIR "
            f"or create {data_dir}."
        )

    toolkit_path = _ensure_toolkit_on_path(root_path)
    toolkit_found = importlib.util.find_spec("arc_agi") is not None
    if toolkit_found and toolkit_path is None:
        toolkit_path = _installed_toolkit_path()
    if require_toolkit and not toolkit_found:
        raise PreflightError(
            "Local official arc_agi toolkit import is required. Install or vendor "
            "the toolkit before executing the notebook, or attach it as a Kaggle input."
        )

    return {
        "root": str(root_path),
        "data_dir": str(data_dir),
        "toolkit_path": toolkit_path,
        "arc_agi_available": toolkit_found,
        "network_guard": bool(block_network),
        "allowed_socket_hosts": sorted(_allowed_socket_hosts()) if block_network else [],
        "official_resource_backed": bool(data_dir.exists() and toolkit_found),
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
                return normalize_actions(candidates)

    if hasattr(game, "get_action_space") and callable(game.get_action_space):
        candidates.extend(_flatten_actions(game.get_action_space()))
    if candidates:
        return normalize_actions(candidates)

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
        return [f"ACTION{i + 1}" for i in range(int(value.n))]
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
    game_count: int | None = None,
    max_actions_per_game: int = 1000,
) -> dict[str, Any]:
    """Run the agent through the official ARC-AGI-3 toolkit and write artifacts.

    The Kaggle competition is forced into ``OperationMode.COMPETITION``. This
    function uses that path when the official toolkit exposes ``Arcade`` and
    ``OperationMode``. A generic local loader remains as a fallback for toolkit
    API drift and for local integration tests. There is no synthetic fallback.
    """

    preflight = offline_preflight(
        root=root,
        require_toolkit=True,
        require_data=False,
        block_network=False,
    )
    if _official_arcade_available():
        return _run_arcade_competition(
            agent=agent,
            root=Path(root or Path.cwd()),
            preflight=preflight,
            game_count=game_count,
            max_actions_per_game=max_actions_per_game,
        )

    data_dir = Path(preflight["data_dir"])
    loader = discover_official_loader(data_dir)
    required_count = 3 if game_count is None else int(game_count)
    games = list(iter_first_games(loader(count=required_count), required_count))
    if len(games) < required_count:
        raise PreflightError(
            f"Official loader returned {len(games)} games; {required_count} are required."
        )

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index, game in enumerate(games, start=1):
        row = _run_one_game(agent, game, index, max_actions_per_game)
        traces.append(
            {
                "game_index": index,
                "records": row.pop("reasoning_trace", []),
            }
        )
        rows.append(row)

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
    (out_dir / "reasoning_trace.json").write_text(
        json.dumps({"games": traces}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return scorecard


def _official_arcade_available() -> bool:
    module = importlib.import_module("arc_agi")
    return hasattr(module, "Arcade") and hasattr(module, "OperationMode")


def _run_arcade_competition(
    agent: Any,
    root: Path,
    preflight: dict[str, Any],
    game_count: int | None,
    max_actions_per_game: int,
) -> dict[str, Any]:
    arc_agi = importlib.import_module("arc_agi")
    operation_mode = getattr(arc_agi, "OperationMode")
    arcade_cls = getattr(arc_agi, "Arcade")
    arc = arcade_cls(operation_mode=operation_mode.COMPETITION)

    environments = list(arc.get_environments())
    if game_count is not None:
        environments = environments[: int(game_count)]
    if not environments:
        raise PreflightError("Official ARC toolkit returned no environments.")

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index, info in enumerate(environments, start=1):
        row = _run_one_arcade_environment(
            agent=agent,
            arc=arc,
            environment_info=info,
            index=index,
            max_actions=max_actions_per_game,
        )
        traces.append(
            {
                "game_index": index,
                "game_id": row.get("game_id"),
                "records": row.pop("reasoning_trace", []),
            }
        )
        rows.append(row)

    scorecard_payload = _jsonable(_safe_call(getattr(arc, "get_scorecard", None)))
    scorecard = {
        "preflight": preflight,
        "mode": "arcade_competition",
        "game_count": len(rows),
        "rows": rows,
        "arc_scorecard": scorecard_payload,
        "note": (
            "In official Kaggle competition mode, the ARC toolkit may withhold "
            "in-flight score details until the platform records the submission."
        ),
    }
    _write_run_artifacts(root, scorecard, traces)
    return scorecard


def _run_one_arcade_environment(
    agent: Any,
    arc: Any,
    environment_info: Any,
    index: int,
    max_actions: int,
) -> dict[str, Any]:
    game_id = _game_id(environment_info)
    env = arc.make(game_id)
    if env is None:
        raise PreflightError(f"Official ARC toolkit could not create environment {game_id!r}.")

    first = _initial_observation(env)
    frame = _extract_frame(first)
    agent.reset(env)
    actions = 0
    wins = 0
    done = _is_done(first)
    last_state = _state_name(first)

    while not done and actions < max_actions:
        action = agent.act(frame)
        toolkit_action, data = _format_toolkit_action(action)
        observation = _step_toolkit_environment(env, toolkit_action, data)
        next_frame = _extract_frame(observation)
        win = _is_win(observation)
        done = _is_done(observation)
        agent.observe(action, next_frame, win=win)
        frame = next_frame
        actions += 1
        wins += int(win)
        last_state = _state_name(observation)

    return {
        "game_index": index,
        "game_id": game_id,
        "title": getattr(environment_info, "title", None),
        "mode": "arcade_competition_make_step",
        "actions": actions,
        "wins": wins,
        "done": bool(done),
        "last_state": last_state,
        "agent_scorecard": agent.scorecard(),
        "reasoning_trace": agent.reasoning_trace(),
    }


def _write_run_artifacts(
    root: Path,
    scorecard: dict[str, Any],
    traces: list[dict[str, Any]],
) -> None:
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "reasoning_trace.json").write_text(
        json.dumps({"games": traces}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _game_id(environment_info: Any) -> str:
    for attr in ("game_id", "id", "name"):
        value = getattr(environment_info, attr, None)
        if value:
            return str(value)
    if isinstance(environment_info, dict):
        for key in ("game_id", "id", "name"):
            if environment_info.get(key):
                return str(environment_info[key])
    return str(environment_info)


def _safe_call(fn: Any) -> Any:
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception as exc:
        return {"unavailable": type(exc).__name__, "message": str(exc)}


def _initial_observation(env: Any) -> Any:
    for attr in ("last_observation", "observation_space", "observation", "obs", "state"):
        value = getattr(env, attr, None)
        if value is not None and _extract_frame(value, raise_on_missing=False) is not None:
            return value
    if hasattr(env, "reset") and callable(env.reset):
        value = env.reset()
        if _extract_frame(value, raise_on_missing=False) is not None:
            return value
    reset_action, data = _format_toolkit_action("RESET")
    return _step_toolkit_environment(env, reset_action, data)


def _format_toolkit_action(action: Any) -> tuple[Any, dict[str, int]]:
    candidate = parse_action_candidate(action)
    data: dict[str, int] = {}
    if candidate.coords is not None:
        x, y = candidate.coords
        data = {"x": int(x), "y": int(y)}

    try:
        from arcengine import GameAction  # type: ignore

        return getattr(GameAction, candidate.name), data
    except Exception:
        return candidate.name, data


def _step_toolkit_environment(env: Any, action: Any, data: dict[str, int]) -> Any:
    try:
        return env.step(action, data=data)
    except TypeError:
        if data:
            try:
                return env.step(action, data)
            except TypeError:
                return env.step({"action": str(action), **data})
        return env.step(action)


def _extract_frame(value: Any, raise_on_missing: bool = True) -> Any:
    if value is None:
        if raise_on_missing:
            raise PreflightError("Official ARC observation did not include frame data.")
        return None
    if isinstance(value, dict):
        for key in ("frame", "observation", "state", "grid", "screen", "frame_data", "data"):
            if key in value:
                nested = _extract_frame(value[key], raise_on_missing=False)
                if nested is not None:
                    return nested
    for attr in ("frame", "observation", "grid", "screen", "frame_data", "data"):
        if hasattr(value, attr):
            nested = _extract_frame(getattr(value, attr), raise_on_missing=False)
            if nested is not None:
                return nested
    if hasattr(value, "to_numpy") and callable(value.to_numpy):
        return value.to_numpy()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return value
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) in (2, 3):
        return value
    if raise_on_missing:
        raise PreflightError("Official ARC observation did not include frame data.")
    return None


def _state_name(value: Any) -> str | None:
    state = None
    if isinstance(value, dict):
        state = value.get("state") or value.get("game_state")
    if state is None:
        state = getattr(value, "state", getattr(value, "game_state", None))
    if state is None:
        return None
    return str(getattr(state, "name", state)).upper()


def _is_win(value: Any) -> bool:
    state = _state_name(value)
    if state == "WIN":
        return True
    if isinstance(value, dict):
        return bool(value.get("win", value.get("won", False)))
    return bool(getattr(value, "win", getattr(value, "won", False)))


def _is_done(value: Any) -> bool:
    state = _state_name(value)
    if state in {"WIN", "GAME_OVER", "DONE", "TERMINATED"}:
        return True
    if isinstance(value, dict):
        return bool(value.get("done", value.get("terminated", value.get("truncated", False))))
    return bool(getattr(value, "done", getattr(value, "terminated", False)))


def _run_one_game(agent: Any, game: Any, index: int, max_actions: int) -> dict[str, Any]:
    if hasattr(game, "evaluate_agent") and callable(game.evaluate_agent):
        result = game.evaluate_agent(agent)
        return {
            "game_index": index,
            "mode": "official_evaluate_agent",
            "result": _jsonable(result),
            "agent_scorecard": agent.scorecard() if hasattr(agent, "scorecard") else {},
            "reasoning_trace": agent.reasoning_trace() if hasattr(agent, "reasoning_trace") else [],
        }

    if hasattr(game, "run_agent") and callable(game.run_agent):
        result = game.run_agent(agent)
        return {
            "game_index": index,
            "mode": "official_run_agent",
            "result": _jsonable(result),
            "agent_scorecard": agent.scorecard() if hasattr(agent, "scorecard") else {},
            "reasoning_trace": agent.reasoning_trace() if hasattr(agent, "reasoning_trace") else [],
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
        env_action = _format_env_action(game, action)
        outcome = game.step(env_action)
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
        "reasoning_trace": agent.reasoning_trace(),
    }


def _format_env_action(game: Any, action: Any) -> Any:
    candidate = parse_action_candidate(action)
    if hasattr(game, "format_action") and callable(game.format_action):
        return game.format_action(candidate.name, *(candidate.coords or ()))
    if hasattr(game, "make_action") and callable(game.make_action):
        return game.make_action(candidate.name, *(candidate.coords or ()))
    if candidate.coords is None:
        return candidate.name
    x, y = candidate.coords
    return {"action": candidate.name, "x": x, "y": y}


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
    frame = _extract_frame(outcome, raise_on_missing=False)
    if frame is not None:
        return frame, _is_win(outcome), _is_done(outcome)
    raise PreflightError("Could not parse official game step outcome.")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        return repr(value)
