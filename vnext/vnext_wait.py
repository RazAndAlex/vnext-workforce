"""Wait for vNext workers so a background command can wake their manager.

On 2026-10-05 a Claude Code manager ended its turn while a researcher was still
running, and the result sat unread because MCP cannot wake an idle client.
This read-only command exits when the named agents settle or its deadline passes,
so a client can use its background shell's completion notification.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vnext_orchestration import (
    OUTCOME_TEXT_LIMIT, SETTLED_STATUSES, _STOPPED_WITHOUT_A_WORD,
)
from .vnext_report import _parse_row, _run_logs


_SETTLED = frozenset(status.value for status in SETTLED_STATUSES)

# A worker that ends its turn by messaging its parent stays "ready" in the run
# log.  That is a stop only if the turn really ended and no new one begins, so
# the waiter holds the condition this long.  A queued message that starts the
# next turn at once then never causes a wake.
READY_GRACE_SECONDS = 5.0


def _positive(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive finite number") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _duration(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?|\.\d+)([smh])", value)
    if match is None:
        raise argparse.ArgumentTypeError("use a duration such as 30s, 20m or 1h")
    seconds = _positive(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError("duration must be finite")
    return seconds


def _agent(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("agent ID must not be empty")
    return value


def _offset(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError("must be a whole number of bytes")
    return int(value)


def _session(value: str) -> str:
    if not value.strip() or value in {".", ".."} or any(char in value for char in '/\\\0<>:"|?*'):
        raise argparse.ArgumentTypeError("session ID must be a nonempty file name")
    return value


@dataclass
class _LogCursor:
    offset: int = 0
    pending: bytes = b""
    line_number: int = 0

    def rows(self, path: Path) -> Iterator[dict[str, Any]]:
        """Read each complete JSONL line once, holding a write still in flight."""
        try:
            size = path.stat().st_size
            if self.offset > size:
                self.offset = 0
                self.pending = b""
                self.line_number = 0
            if self.offset == size:
                return
            with path.open("rb") as handle:
                handle.seek(self.offset)
                # Bound this poll to a snapshot, even if the writer stays busy.
                remaining = size - self.offset
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.offset += len(chunk)
                    remaining -= len(chunk)
                    lines = (self.pending + chunk).split(b"\n")
                    self.pending = lines.pop()
                    for raw in lines:
                        self.line_number += 1
                        row = _parse_row(raw, self.line_number)
                        if row is not None:
                            yield row
        except OSError as exc:
            yield {"type": "unreadable.record", "where": "the whole file", "cause": str(exc)}


@dataclass
class _AgentState:
    status: str
    stamp: str
    payload: dict[str, Any]
    session_id: str

    def print_report(self, agent: str, trail: "_Trail | None" = None) -> None:
        result = self.payload.get("result")
        result = result if isinstance(result, dict) else {}
        if self.status == "ready":
            # Reached only through the ended-turn rule: no final report exists.
            when = trail.last_turn.get(agent, ("", self.stamp))[1] if trail else self.stamp
            print(f"{agent} ready {when}")
            text = ("This worker ended its turn without a final report and is waiting "
                    "for a message.")
            message = (trail.last_message.get(agent, "") if trail else "").strip()
            if message:
                text += f" Its latest message to its parent: {message}"
            else:
                text += " It sent no message to its parent that this log holds."
        else:
            print(f"{agent} {self.status} {self.stamp}")
            text = str(result.get("outcome") or "").strip()
            if not text:
                text = str(self.payload.get("blocker") or "").strip()
            if not text:
                text = _STOPPED_WITHOUT_A_WORD.get(self.status, "")
        cut_mark = (
            "… (cut; the full text is in "
            f".vnext/outcomes/{self.session_id}.jsonl)"
        )
        if len(text) > OUTCOME_TEXT_LIMIT:
            text = text[: OUTCOME_TEXT_LIMIT - len(cut_mark)].rstrip() + cut_mark
        print(text)
        verified = "yes" if result.get("verified") else "no"
        evidence = result.get("evidence") or ()
        print(f"verified: {verified}, evidence: {len(evidence)} item(s)")
        usage = self.payload.get("usage")
        if isinstance(usage, dict):
            tokens = usage.get("cost_tokens")
            if isinstance(tokens, dict):
                tokens = tokens.get("total_tokens")
            parts = [f"{label}: {value}" for label, value in (
                ("tokens", tokens), ("cost_usd", usage.get("cost_usd"))
            ) if value is not None]
            if parts:
                print(", ".join(parts))


@dataclass
class _Trail:
    """What the rows read so far say about each agent's turns and messages."""

    turn_started: set[str] = field(default_factory=set)
    last_turn: dict[str, tuple[str, str]] = field(default_factory=dict)
    last_message: dict[str, str] = field(default_factory=dict)
    ready_since: dict[str, float] = field(default_factory=dict)

    def turn_ended(self, agent: str, states: dict[str, "_AgentState"]) -> bool:
        state = states.get(agent)
        return (
            state is not None and state.status == "ready"
            and agent in self.turn_started
            and self.last_turn.get(agent, ("", ""))[0] == "turn.completed"
        )


def _states(
    workspace: Path, wanted: set[str], warned: set[tuple],
    states: dict[str, _AgentState], cursors: dict[Path, _LogCursor],
    session: str | None = None, trail: _Trail | None = None,
) -> dict[str, _AgentState]:
    trail = trail if trail is not None else _Trail()
    logs = [workspace / ".vnext" / "runs" / f"{session}.jsonl"] if session is not None else (
        _run_logs(workspace / ".vnext"))
    for log in logs:
        for row in cursors.setdefault(log, _LogCursor()).rows(log):
            if row.get("type") == "unreadable.record":
                warning = (log, row.get("where"), row.get("cause"))
                if warning not in warned:
                    print(f"vnext-wait: skipped {log}, {warning[1]}: {warning[2]}", file=sys.stderr)
                    warned.add(warning)
                continue
            agent = row.get("agent_id")
            payload = row.get("payload")
            kind = row.get("type")
            if kind in {"turn.started", "turn.completed"} and agent in wanted:
                if kind == "turn.started":
                    trail.turn_started.add(agent)
                trail.last_turn[agent] = (kind, str(row.get("timestamp") or "time not recorded"))
                continue
            if (
                kind == "command.acknowledged" and isinstance(payload, dict)
                and payload.get("command") == "send_message"
                and payload.get("sender_id") in wanted
            ):
                trail.last_message[payload["sender_id"]] = str(payload.get("message") or "")
                continue
            if row.get("type") != "agent.upsert" or not isinstance(agent, str) or agent not in wanted:
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
                continue
            status = payload["status"]
            previous = states.get(agent)
            # Later usage snapshots can repeat a stopped status. Keep when it moved.
            stamp = previous.stamp if previous and previous.status == status else str(
                row.get("timestamp") or "time not recorded")
            states[agent] = _AgentState(status, stamp, payload, str(
                row.get("session_id") or session or log.stem))
    return states


def main(
    argv: Sequence[str] | None = None, *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--agent", action="append", type=_agent, required=True)
    parser.add_argument("--session", type=_session, metavar="SESSION_ID")
    parser.add_argument(
        "--since-offset", type=_offset, default=0, metavar="BYTES",
        help="with --session, ignore run-log bytes before this offset (a retried "
             "agent's earlier settled row)",
    )
    parser.add_argument("--deadline", default="30m", metavar="DURATION")
    parser.add_argument("--poll", type=_positive, default=2.0, metavar="SECONDS")
    try:
        args = parser.parse_args(argv)
        try:
            deadline = _duration(args.deadline)
        except argparse.ArgumentTypeError as exc:
            parser.error(f"--deadline: {exc}")
    except SystemExit as exc:
        return int(exc.code)
    # Never lose a finished worker's report to a console that lacks one of its characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"vnext-wait: workspace is not a folder: {workspace}", file=sys.stderr)
        return 2
    agents = list(dict.fromkeys(args.agent))
    wanted = set(agents)
    seen: set[str] = set()
    warned: set[tuple] = set()
    states: dict[str, _AgentState] = {}
    trail = _Trail()
    cursors: dict[Path, _LogCursor] = {}
    if args.since_offset and args.session is not None:
        cursors[workspace / ".vnext" / "runs" / f"{args.session}.jsonl"] = _LogCursor(
            offset=args.since_offset)
    started = clock()
    while True:
        _states(workspace, wanted, warned, states, cursors, args.session, trail)
        seen.update(states)
        now = clock()
        for agent in agents:
            if trail.turn_ended(agent, states):
                trail.ready_since.setdefault(agent, now)
            else:
                trail.ready_since.pop(agent, None)

        def stopped(agent: str) -> bool:
            if agent not in states:
                return False
            if states[agent].status in _SETTLED:
                return True
            since = trail.ready_since.get(agent)
            return since is not None and now - since >= READY_GRACE_SECONDS

        if all(stopped(agent) for agent in agents):
            for index, agent in enumerate(agents):
                if index:
                    print()
                states[agent].print_report(agent, trail)
            return 0
        elapsed = clock() - started
        unseen = wanted - seen
        if unseen and elapsed >= min(60.0, deadline):
            search = str(workspace / ".vnext" / "runs" / f"{args.session}.jsonl") if (
                args.session is not None) else (
                f"{workspace / '.vnext' / 'runs' / '*.jsonl'} and "
                f"{workspace / '.vnext' / 'events.jsonl'}")
            print(f"vnext-wait: agent ID(s) never seen: {', '.join(sorted(unseen))}; "
                  f"searched {search} for {elapsed:g}s", file=sys.stderr)
            return 2
        if elapsed >= deadline:
            for agent in agents:
                status = states[agent].status if agent in states else "unknown"
                if not stopped(agent):
                    print(f"{agent} {status} still running after {args.deadline}")
            return 3
        remaining = deadline - elapsed
        if unseen:
            remaining = min(remaining, 60.0 - elapsed)
        pause = min(args.poll, remaining)
        # Wake as soon as the grace ends instead of up to a poll later.
        for since in trail.ready_since.values():
            pause = min(pause, max(since + READY_GRACE_SECONDS - clock(), 0.01))
        sleep(pause)


if __name__ == "__main__":
    raise SystemExit(main())
