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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vnext_orchestration import (
    OUTCOME_TEXT_LIMIT, SETTLED_STATUSES, _STOPPED_WITHOUT_A_WORD,
)
from .vnext_report import _parse_row, _run_logs


_SETTLED = frozenset(status.value for status in SETTLED_STATUSES)


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

    def print_report(self, agent: str) -> None:
        print(f"{agent} {self.status} {self.stamp}")
        result = self.payload.get("result")
        result = result if isinstance(result, dict) else {}
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


def _states(
    workspace: Path, wanted: set[str], warned: set[tuple],
    states: dict[str, _AgentState], cursors: dict[Path, _LogCursor],
    session: str | None = None,
) -> dict[str, _AgentState]:
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
    cursors: dict[Path, _LogCursor] = {}
    started = clock()
    while True:
        _states(workspace, wanted, warned, states, cursors, args.session)
        seen.update(states)
        if all(agent in states and states[agent].status in _SETTLED for agent in agents):
            for index, agent in enumerate(agents):
                if index:
                    print()
                states[agent].print_report(agent)
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
                if status not in _SETTLED:
                    print(f"{agent} {status} still running after {args.deadline}")
            return 3
        remaining = deadline - elapsed
        if unseen:
            remaining = min(remaining, 60.0 - elapsed)
        sleep(min(args.poll, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
