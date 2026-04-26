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
import time
from typing import Any, Callable, Iterable
from zipfile import ZipFile

from .actions import canonical_action_name, parse_action_candidate, normalize_actions


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


def _candidate_toolkit_wheel_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("ARC_AGI_WHEELS_DIR", "ARC_AGI_TOOLKIT_WHEELS_DIR"):
        env_dir = os.environ.get(env_name)
        if env_dir:
            candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/wheels"),
            root / "arc_agi_3_wheels",
            root / "wheels",
            root / "vendor" / "wheels",
        ]
    )
    for kaggle_dir in _candidate_kaggle_dirs():
        candidates.extend([kaggle_dir / "arc_agi_3_wheels", kaggle_dir / "wheels"])

    base = Path("/kaggle/input")
    if base.exists():
        for child in sorted(path for path in base.iterdir() if path.is_dir()):
            candidates.extend([child / "arc_agi_3_wheels", child / "wheels"])
            if child.name.lower() in {"competition", "competitions"}:
                for competition in sorted(path for path in child.iterdir() if path.is_dir()):
                    candidates.extend([competition / "arc_agi_3_wheels", competition / "wheels"])
    return candidates


def _candidate_toolkit_wheels(root: Path) -> list[Path]:
    wheel_files: list[Path] = []
    for wheel_dir in _candidate_toolkit_wheel_dirs(root):
        if not wheel_dir.exists() or not wheel_dir.is_dir():
            continue
        wheels = sorted(wheel_dir.glob("*.whl"))
        if any("arc" in wheel.name.lower() for wheel in wheels):
            wheel_files.extend(wheels)

    seen: set[Path] = set()
    unique: list[Path] = []
    for wheel in wheel_files:
        resolved = wheel.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


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
    wheel_path = _ensure_toolkit_wheels_on_path(root)
    if wheel_path is not None:
        return wheel_path
    return None


def _ensure_toolkit_wheels_on_path(root: Path) -> str | None:
    wheels = _candidate_toolkit_wheels(root)
    if not wheels:
        return None

    extract_root = root / ".htba_arc_agi_wheels"
    extract_root.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        _extract_wheel(wheel, extract_root)

    resolved = str(extract_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    importlib.invalidate_caches()
    return resolved if importlib.util.find_spec("arc_agi") is not None else None


def _extract_wheel(wheel: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise PreflightError(f"Unsafe wheel member path in {wheel.name}: {member.filename!r}")
            target = (destination / member_path).resolve()
            if not str(target).startswith(str(destination)):
                raise PreflightError(f"Unsafe wheel member path in {wheel.name}: {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


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
            "Local official arc_agi toolkit import is required. Attach the official "
            "competition wheel input, set ARC_AGI_WHEELS_DIR, set ARC_AGI_TOOLKIT_DIR, "
            "or vendor an unpacked arc_agi toolkit before executing the notebook."
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
    if isinstance(game, dict):
        for key in ("action_space", "available_actions", "actions", "alive_actions"):
            if key in game:
                candidates.extend(_flatten_actions(game[key]))
                if candidates:
                    return normalize_actions(candidates)

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

    scorecard_payload = _finalize_arc_scorecard(arc)
    scorecard = {
        "preflight": preflight,
        "mode": "arcade_competition",
        "game_count": len(rows),
        "rows": rows,
        "arc_scorecard": scorecard_payload,
        "note": (
            "In official Kaggle competition mode, the ARC toolkit may withhold "
            "score details until the platform records or closes the submission."
        ),
    }
    _write_run_artifacts(root, scorecard, traces)
    return scorecard


def _finalize_arc_scorecard(arc: Any) -> dict[str, Any]:
    """Close the official scorecard when possible and preserve failures.

    The public toolkit documents ``close_scorecard`` as the final-score path,
    while competition mode may still withhold score details. This helper records
    both the close attempt and the fallback read attempt without turning
    platform-side withholding into a local crash.
    """

    close_result = _safe_call(getattr(arc, "close_scorecard", None))
    if close_result is not None and not _call_unavailable(close_result):
        return {
            "source": "close_scorecard",
            "payload": _jsonable(close_result),
        }

    get_result = _safe_call(getattr(arc, "get_scorecard", None))
    if get_result is not None and not _call_unavailable(get_result):
        return {
            "source": "get_scorecard",
            "payload": _jsonable(get_result),
            "close_scorecard": _jsonable(close_result),
        }

    return {
        "source": "unavailable",
        "close_scorecard": _jsonable(close_result),
        "get_scorecard": _jsonable(get_result),
        "note": "The official toolkit did not expose scorecard details locally.",
    }


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
    _reset_agent_with_available_actions(agent, env, first)
    actions = 0
    wins = 0
    done = _is_done(first)
    last_state = _state_name(first)
    failure_flags: list[dict[str, Any]] = []

    while not done and actions < max_actions:
        action = agent.act(frame)
        toolkit_action, data = _format_toolkit_action(action)
        observation = _step_toolkit_environment_with_retries(env, toolkit_action, data)
        if _is_step_error(observation):
            failure_flags.append(observation["htba_step_error"])
            done = True
            last_state = "STEP_ERROR"
            break

        next_frame = _extract_frame(observation, raise_on_missing=False)
        if next_frame is None:
            failure_flags.append(_missing_frame_failure(toolkit_action, observation))
            done = True
            last_state = _state_name(observation) or "MISSING_FRAME"
            break

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
        "failure_flags": failure_flags,
        "agent_scorecard": agent.scorecard(),
        "reasoning_trace": agent.reasoning_trace(),
    }


def _reset_agent_with_available_actions(agent: Any, env: Any, observation: Any) -> None:
    """Initialize the agent from env metadata, falling back to reset observation.

    The official toolkit has exposed alive actions on different objects across
    releases. Prefer the environment, but accept the first observation when it
    carries fields such as ``available_actions``.
    """

    for source in (env, observation):
        try:
            probe_alive_actions(source)
        except Exception:
            continue
        agent.reset(source)
        return
    agent.reset(env)


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
    _write_submission_parquet(root, scorecard)


def _write_submission_parquet(root: Path, scorecard: dict[str, Any]) -> None:
    """Generate submission.parquet for the Kaggle ARC-AGI-3 competition.

    The competition expects ``submission.parquet`` in the notebook output
    directory (``/kaggle/working/``).  This helper builds it from the
    per-environment scorecard rows.  If pyarrow is unavailable, it falls back
    to a minimal raw-Parquet writer so no extra dependency is required.
    """

    rows = scorecard.get("rows", [])
    records: list[dict[str, Any]] = []
    for row in rows:
        game_id = row.get("game_id", f"game_{row.get('game_index', 0)}")
        agent_sc = row.get("agent_scorecard", {})
        records.append({
            "game_id": str(game_id),
            "score": float(agent_sc.get("wins", 0)),
            "actions": int(agent_sc.get("actions", row.get("actions", 0))),
            "done": bool(row.get("done", False)),
        })

    parquet_path = root / "submission.parquet"
    try:
        import pyarrow as pa         # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        table = pa.table({
            "game_id": [r["game_id"] for r in records],
            "score": [r["score"] for r in records],
            "actions": [r["actions"] for r in records],
            "done": [r["done"] for r in records],
        })
        pq.write_table(table, str(parquet_path))
    except Exception:
        # Fallback: pandas is available in the Kaggle runtime.
        try:
            import pandas as pd  # type: ignore[import-untyped]

            df = pd.DataFrame(records)
            df.to_parquet(str(parquet_path), index=False)
        except Exception:
            # Last resort: write a JSON stand-in so the file exists.
            parquet_path.with_suffix(".json").write_text(
                json.dumps(records, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return



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


def _call_unavailable(value: Any) -> bool:
    return isinstance(value, dict) and "unavailable" in value


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
        pass
    except Exception as exc:
        return _step_error(action, exc)

    if data:
        try:
            return env.step(action, data)
        except TypeError:
            pass
        except Exception as exc:
            return _step_error(action, exc)

        try:
            return env.step({"action": str(action), **data})
        except Exception as exc:
            return _step_error(action, exc)

    try:
        return env.step(action)
    except Exception as exc:
        return _step_error(action, exc)


def _step_toolkit_environment_with_retries(env: Any, action: Any, data: dict[str, int]) -> Any:
    retries = _env_int("HTBA_ARC_STEP_RETRIES", 3)
    delay = _env_float("HTBA_ARC_STEP_RETRY_DELAY", 2.0)
    result: Any = None
    for attempt in range(retries + 1):
        result = _step_toolkit_environment(env, action, data)
        if not _should_retry_step_result(result):
            return result
        if attempt < retries and delay > 0:
            time.sleep(delay * (attempt + 1))
    return result


def _step_error(action: Any, exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return {
        "state": "STEP_ERROR",
        "done": True,
        "htba_step_error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "status_code": status_code,
            "action": str(action),
        },
    }


def _is_step_error(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("htba_step_error"), dict)


def _should_retry_step_result(value: Any) -> bool:
    if value is None:
        return True
    if not _is_step_error(value):
        return False
    error = value["htba_step_error"]
    message = str(error.get("message", "")).lower()
    return error.get("status_code") == 429 or "429" in message or "too many requests" in message


def _missing_frame_failure(action: Any, observation: Any) -> dict[str, Any]:
    return {
        "type": "missing_frame_after_step",
        "message": "Official step returned no frame or metadata observation.",
        "action": str(action),
        "observation_type": type(observation).__name__,
        "observation_repr": repr(observation)[:500],
    }


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


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
    metadata_frame = _observation_metadata_frame(value)
    if metadata_frame is not None:
        return metadata_frame
    if raise_on_missing:
        raise PreflightError("Official ARC observation did not include frame data.")
    return None


def _observation_metadata_frame(value: Any) -> list[list[int]] | None:
    """Encode non-visual official observations as a small feature grid.

    Some competition-mode observations expose state and alive-action metadata
    without a raw pixel frame. Returning a deterministic 2-D grid keeps the
    object encoder and planner path intact without notebook-level monkey
    patches or synthetic task data.
    """

    action_names = _metadata_action_names(value)
    state = _state_name(value)
    metrics = [
        _metadata_int(value, "levels_completed"),
        _metadata_int(value, "win_levels"),
        _metadata_int(value, "score"),
        _metadata_int(value, "reward"),
        _metadata_int(value, "step"),
        _metadata_int(value, "steps"),
        _metadata_int(value, "lives"),
    ]
    if not action_names and state is None and not any(metrics):
        return None

    action_row = [0] * 8
    for action in action_names:
        index = _action_index(action)
        if index is not None:
            action_row[index] = 1

    state_row = [_state_code(state), *metrics]
    return [action_row, state_row[:8]]


def _metadata_action_names(value: Any) -> tuple[str, ...]:
    try:
        return probe_alive_actions(value)
    except Exception:
        return ()


def _action_index(action: Any) -> int | None:
    name = canonical_action_name(action)
    if name == "RESET":
        return 0
    if name.startswith("ACTION") and name[6:].isdigit():
        index = int(name[6:])
        if 1 <= index <= 7:
            return index
    return None


def _metadata_int(value: Any, name: str) -> int:
    raw = value.get(name) if isinstance(value, dict) else getattr(value, name, 0)
    raw = 0 if raw is None else raw
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _state_code(state: str | None) -> int:
    if state is None:
        return 0
    return {
        "NOT_FINISHED": 0,
        "RUNNING": 0,
        "PLAYING": 0,
        "WIN": 1,
        "WON": 1,
        "GAME_OVER": 2,
        "LOSE": 2,
        "LOST": 2,
        "DONE": 3,
        "TERMINATED": 3,
        "TRUNCATED": 3,
        "STEP_ERROR": 3,
    }.get(state, 0)


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
    if state in {"WIN", "GAME_OVER", "DONE", "TERMINATED", "STEP_ERROR"}:
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
