#!/usr/bin/env python3
"""Codex ``PostToolUse`` command hook that prints one ``[clock]`` line.

Codex runs this as a bare file path under an interpreter named in the hook's
command string, so it imports nothing but the standard library and the one
shared formatter.  Codex hands it ``session_id``, ``turn_id``, ``cwd``,
``transcript_path``, ``model``, ``tool_name``, ``tool_input`` and
``tool_response`` on stdin -- no turn start time and no budget.  Both of those
come from a small state file vNext writes when it starts a turn, whose path is
argv[1]; argv[2] carries the budget as a fallback.

Contract with the turn: never raise, never exit non-zero, and print nothing at
all when there is no state to print from.  A hook that failed in the live probe
left the turn intact, and this keeps it that way by construction.

Usage:
    python clock_hook.py <state-file> [<limit-seconds>]
"""

from __future__ import annotations

import json
import os
import sys
import time

try:  # Normal case: the package is importable.
    from vnext.clock_format import mid_turn_line
except ImportError:  # Run as a loose file path with no package on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vnext.clock_format import mid_turn_line


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _fallback_start(path: str, turn_id: str, now: float, monotonic_now: float) -> tuple[float, bool] | None:
    """Record and return the first moment this turn was seen.

    Codex's hook payload carries the turn id on every call of a turn, and the
    live probe confirmed it is stable across them.  So when the state file has
    no entry for this session, the first tool call of the turn becomes the
    zero point.  The number is then a lower bound on the turn's real age, which
    is the honest thing to report when vNext's own record is missing.
    """

    if not turn_id:
        return None
    sidecar = path + ".turns.json"
    try:
        seen = _read_json(sidecar)
    except Exception:
        seen = {}
    entry = seen.get(turn_id)
    if isinstance(entry, dict) and isinstance(entry.get("started_monotonic"), (int, float)):
        return float(entry["started_monotonic"]), True
    if isinstance(entry, (int, float)):
        return float(entry), False
    seen[turn_id] = {"started": now, "started_monotonic": monotonic_now}
    try:
        temporary = sidecar + f".{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(seen, handle)
        os.replace(temporary, sidecar)
    except Exception:
        # A sidecar we could not write still gives this call a sane "0s".
        pass
    return monotonic_now, True


def resolve_turn(payload: dict, state: dict, *, limit: float | None):
    """Return start, turn id, limit, and start clock for a PostToolUse payload.

    A missing start lets the caller use the first-seen turn fallback.
    """

    session_id = str(payload.get("session_id") or "")
    turn_id = str(payload.get("turn_id") or "")
    entry = state.get(session_id) if session_id else None
    started = None
    monotonic_start = False
    if isinstance(entry, dict):
        if isinstance(entry.get("started_monotonic"), (int, float)):
            started = float(entry["started_monotonic"])
            monotonic_start = True
        elif isinstance(entry.get("started"), (int, float)):
            # State from an older manager still supplies a useful fallback.
            started = float(entry["started"])
        budget = entry.get("limit")
        if isinstance(budget, (int, float)):
            limit = float(budget)
    return started, turn_id, limit, monotonic_start


def main(argv: list[str], stdin) -> int:
    if len(argv) < 2:
        return 0
    state_path = argv[1]
    limit: float | None = None
    if len(argv) > 2:
        try:
            limit = float(argv[2])
        except ValueError:
            limit = None
    try:
        payload = json.loads(stdin.read())
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        state = _read_json(state_path)
    except Exception:
        # No state file means this is not a vNext-managed turn -- the user's
        # own Codex sessions share ~/.codex and must stay untouched.
        return 0
    now = time.time()
    monotonic_now = time.monotonic()
    started, turn_id, limit, monotonic_start = resolve_turn(payload, state, limit=limit)
    if started is None:
        fallback = _fallback_start(state_path, turn_id, now, monotonic_now)
        if fallback is None:
            return 0
        started, monotonic_start = fallback
    elapsed = (monotonic_now if monotonic_start else now) - started
    line = mid_turn_line(now, elapsed=max(0.0, elapsed), limit=limit)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": line,
        }
    }))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    try:
        raise SystemExit(main(sys.argv, sys.stdin))
    except SystemExit:
        raise
    except Exception:
        # A clock is never worth a turn.
        raise SystemExit(0)
