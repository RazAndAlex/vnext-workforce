"""Read back what the vNext workforce did, and why it stopped.

The run records are one JSONL file per session per workspace, which is the right
shape to write and the wrong shape to read: finding out why vNext had failed in
other chats meant opening several event logs by hand and guessing which of them
was current.  This reads them.

    vnext-report "C:/path/to/project" ...

``python -m vnext.vnext_report`` runs the same thing.

With no path it reads the current directory, and says which folder that is.
``--failures`` prints only the workspaces that went wrong, which is the question
worth asking first, and exits non-zero when any did, so a watcher can ask it in
a script.  A record the reader could not parse counts with the failures.  An error recorded against a worker that had already finished is shown
as a note instead: that is the server's own quit closing a stream, and counting
it made every clean session read as a failure.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

# A traceback belongs on disk and not on screen unless asked for; the cause line
# and the first few stderr lines are what identify a failure.
_STDERR_LINES = 6
_FAILURE_EVENTS = {"session.error", "provider.error"}
# What the SDK answers when vNext's own quit closes a worker's still-open
# stream.  Nothing else a provider says about a finished worker is our doing.
_SHUTDOWN_REASONS = frozenset({"aborted_streaming", "aborted_tools"})


def closed_by_shutdown(failure: dict[str, Any]) -> bool:
    """Whether a provider error is a stream that vNext's own stop closed.

    A completed worker's turn can stay open while its usage is provisional.
    Quitting then closes that stream and the SDK answers aborted_streaming,
    which is noise.  A provider that refuses the same turn, with a 401 or a
    failed Codex turn, has said something an operator needs to hear, so only
    the abort shape is read as ours.
    """

    error = failure.get("error")
    if not isinstance(error, dict) or provider_refused(failure):
        return False
    return error.get("terminal_reason") in _SHUTDOWN_REASONS


# Where a Codex-route turn error carries the HTTP status: only in its sentence.
_STATUS_SAID = re.compile(r"\bstatus (\d{3})\b")


# The codes a Claude assistant message carries when the provider refused the
# turn.  "unknown" is left out: it names nothing the provider decided.
_REFUSAL_CODES = frozenset({
    "authentication_failed", "billing_error", "rate_limit", "invalid_request", "server_error",
})


def provider_refused(failure: dict[str, Any]) -> bool:
    """Whether the provider itself refused or failed this turn.

    A stream our own stop closed carries no status and no refusal code.  A
    refusal carries one of them: api_error_status or a refusal code on the
    Claude route, and the status in the error sentence on the Codex route.
    That is the provider's verdict whatever vNext was doing.
    """

    error = failure.get("error")
    if isinstance(error, dict):
        status = error.get("api_error_status")
        if isinstance(status, int) and not isinstance(status, bool):
            return True
        code = error.get("code")
        return isinstance(code, str) and code in _REFUSAL_CODES
    return isinstance(error, str) and _STATUS_SAID.search(error) is not None


def _recorded_at(row: dict[str, Any]) -> float | None:
    """When a record was written, as seconds, or None when it cannot be read."""

    stamp = row.get("timestamp")
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


def _came_after(failure: dict[str, Any], finished_at: float | None) -> bool:
    """True when this failure was recorded after the agent had finished.

    Both stamps come out of the same log written by the same server, so when
    either is unreadable the order of the lines in the file answers the same
    question: the completion was already read before this record was reached.
    """

    stamp = _recorded_at(failure)
    if stamp is None or finished_at is None:
        return True
    return stamp >= finished_at


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    for _line, row in _numbered_rows(path):
        yield row


def _numbered_rows(path: Path) -> Iterator[tuple[int | str, dict[str, Any]]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        # A log nobody can open is a record nobody read, and "no failures
        # recorded" over it told a watching script the run was clean.
        yield "the whole file", {"type": "unreadable.record", "where": "the whole file",
                                 "cause": str(exc)}
        return
    for line_number, raw in enumerate(data.splitlines(), 1):
        row = _parse_row(raw, line_number)
        if row is not None:
            yield line_number, row


def _parse_row(raw: bytes, line_number: int) -> dict[str, Any] | None:
    """Parse one line, sharing report diagnostics with incremental readers."""
    if not raw.strip():
        return None
    # Bytes that are not UTF-8 were once replaced in silence, and a damaged
    # "provider.error" became a word nothing looked for.
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {"type": "unreadable.record", "where": f"line {line_number}",
                "cause": f"the line is not UTF-8 ({exc.reason} at byte {exc.start})"}
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        # Keep reading later lines and report the parse error without
        # guessing whether the cause was a truncated write or interleaving.
        return {"type": "unreadable.record", "where": f"line {line_number}",
                "cause": str(exc)}
    if isinstance(row, dict):
        return row
    held = "null" if row is None else type(row).__name__
    return {"type": "unreadable.record", "where": f"line {line_number}",
            "cause": f"the line holds JSON {held} where a record object belongs"}


def _run_logs(home: Path) -> list[Path]:
    logs = sorted((home / "runs").glob("*.jsonl"))
    legacy = home / "events.jsonl"
    if legacy.exists():
        logs.append(legacy)
    return logs


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _readable_session(session_id: str) -> str:
    """Enough of a session id to recognise it.

    Ids chosen by hand share a prefix, so eight characters of "session-a" came
    out as "session-", which names nothing.  A short id travels whole and a long
    one keeps the first eight, where a generated id already differs.
    """

    return session_id if len(session_id) <= 12 else session_id[:8]

def _status_sessions(home: Path) -> dict[str, dict[str, Any]]:
    """Session id -> its status snapshot, including nested-server context.

    A Claude worker runs a real Claude Code, so the vnext plugin starts a
    second server inside the worker, in the same project.  That server writes
    its own records beside the outer session's.  The outer session puts its own
    id in its environment, and any server that starts under it copies the id
    into its status file, so the pair can be read back here.

    The marker travels in the environment, which every descendant process
    inherits: a shell, an editor, any program started from that session.  So it
    attests where a server started and nothing more.  Reading it as proof of a
    worker labelled unrelated programs as somebody's worker.
    """

    found: dict[str, dict[str, Any]] = {}
    for path in sorted((home / "status").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        found[str(data.get("session_id") or path.stem)] = data
    return found


def read_workspace(workspace: Path) -> dict[str, Any]:
    """Every session this workspace has run, oldest first."""

    home = workspace / ".vnext"
    sessions: dict[str, dict[str, Any]] = {}
    for log in _run_logs(home):
        for row in _rows(log):
            kind = row.get("type")
            key = str(row.get("session_id") or log.stem)
            session = sessions.setdefault(key, {
                "session_id": key,
                "log": log,
                "records": 0,
                "unreadable": 0,
                "unreadable_causes": [],
                "status": "unknown",
                "agents": set(),
                "agent_states": set(),
                "agent_status": {},
                "agent_finished": {},
                # The session's root when a client vNext does not own holds it.
                # Its record is about the client, not about work this workforce
                # ran, so the shown status does not read it.
                "external_root": None,
                "failures": [],
                # One failure can be recorded twice: a Z.ai worker's refused
                # key was written by its turn and again by its release, and
                # the report printed the same line twice.
                "failures_seen": set(),
                "notes": [],
            })
            session["records"] += 1
            if kind == "unreadable.record":
                session["unreadable"] += 1
                session["unreadable_causes"].append(
                    f"unreadable record at {row['where']}: {row['cause']}"
                )
                continue
            payload = row.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            if kind == "session.upsert" and payload.get("status"):
                session["status"] = str(payload["status"])
            elif kind in {"agent.spawned", "agent.delegated"}:
                session["agents"].add(str(row.get("agent_id") or len(session["agents"])))
            elif kind == "agent.upsert" and payload.get("status"):
                # One agent writes a status every time it moves, so the log holds
                # a worker's whole history.  Keep the last word per agent: an
                # earlier "blocked" is where that worker was, not where it ended.
                agent = str(row.get("agent_id") or f"#{session['records']}")
                session["agent_status"][agent] = str(payload["status"])
                if payload["status"] == "completed":
                    session["agent_finished"][agent] = _recorded_at(row)
                else:
                    session["agent_finished"].pop(agent, None)
                if str(payload.get("provider") or "") == "external":
                    session["external_root"] = agent
            elif kind in _FAILURE_EVENTS:
                failure = {"type": kind, **payload}
                # Quitting the server closes a finished Claude worker's still
                # open stream, and the SDK answers an error for a turn nobody
                # was waiting on.  Older logs already hold that record, so the
                # reading happens here as well as where it is written: an error
                # against an agent that had already completed, and recorded
                # after it completed, is a note under the session.
                agent = str(row.get("agent_id") or "")
                if agent in session["agent_finished"] and _came_after(
                    row, session["agent_finished"][agent]
                ) and closed_by_shutdown(failure):
                    session["notes"].append(failure)
                elif (agent, str(failure.get("error"))) not in session["failures_seen"]:
                    session["failures_seen"].add((agent, str(failure.get("error"))))
                    session["failures"].append(failure)
    snapshots = _status_sessions(home)
    for key, session in sessions.items():
        snapshot = snapshots.get(key, {})
        parent = snapshot.get("parent_session")
        session["parent"] = parent if isinstance(parent, str) and parent else None
        session["server_pid"] = snapshot.get("pid")
        # The states each worker finished in, which is what the display reads.
        # The external root is left out: a server quit writes `cancelled` on
        # that record too, and reading it back made every clean run with an
        # external primary cancel itself.
        session["agent_states"] = {
            state
            for agent, state in session["agent_status"].items()
            if agent != session["external_root"]
        }
    outcomes: list[dict[str, Any]] = []
    # An outcome row that cannot be read is a price and a verdict nobody saw;
    # dropped here, the run read clean and --failures exited 0 over it.
    unreadable_outcomes: list[str] = []
    for path in sorted((home / "outcomes").glob("*.jsonl")):
        for line_number, row in _numbered_rows(path):
            if row.get("type") == "unreadable.record" and not row.get("agent_id"):
                unreadable_outcomes.append(
                    f"outcomes/{path.name}: unreadable record at {row['where']}: {row['cause']}"
                )
                continue
            fault = _outcome_fault(row)
            if fault:
                unreadable_outcomes.append(
                    f"outcomes/{path.name}: unreadable record at line {line_number}: {fault}"
                )
            else:
                outcomes.append(row)
    # Each row is finite on its own and the sum can still overflow; printed,
    # it read "$inf" on a run that exited clean.
    priced = [
        row["cost_usd"] for row in outcomes
        if isinstance(row.get("cost_usd"), (int, float)) and not isinstance(row.get("cost_usd"), bool)
    ]
    try:
        overflowed = not math.isfinite(sum(priced))
    except OverflowError:
        overflowed = True
    if overflowed:
        unreadable_outcomes.append(
            "outcomes: the recorded costs add up to more than the report can count"
        )
        outcomes = [{key: value for key, value in row.items() if key != "cost_usd"} for row in outcomes]
    return {
        "workspace": workspace,
        "exists": home.is_dir(),
        "sessions": sorted(sessions.values(), key=lambda s: s["log"].stat().st_mtime),
        "outcomes": outcomes,
        "unreadable_outcomes": unreadable_outcomes,
    }


def _finite(number: int | float) -> bool:
    # An integer past what a float holds raises here instead of reading as inf.
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


def _shown(value: object) -> str:
    shown = repr(value)
    return shown if len(shown) <= 40 else shown[:39] + "\u2026"


def _outcome_fault(row: dict[str, Any]) -> str | None:
    """Why an outcome row cannot be counted, or None when it can.

    A row with no agent was skipped and the run read clean; a list in
    cost_source crashed the total, a NaN price printed as $nan, a price too
    large for a float raised, and "false" in claimed_verified counted as a
    claim.
    """

    agent = row.get("agent_id")
    if not isinstance(agent, str) or not agent:
        return "the row names no agent"
    source = row.get("cost_source")
    if source is not None and not isinstance(source, str):
        return f"cost_source holds JSON {type(source).__name__} where a word belongs"
    cost = row.get("cost_usd")
    if cost is not None and (
        isinstance(cost, bool) or not isinstance(cost, (int, float)) or not _finite(cost)
    ):
        return f"cost_usd holds {_shown(cost)} where a price belongs"
    # The claim fields are counted, so a damaged one must not read as a claim.
    status = row.get("status")
    if status is not None and not isinstance(status, str):
        return f"status holds JSON {type(status).__name__} where a word belongs"
    claimed = row.get("claimed_verified")
    if claimed is not None and not isinstance(claimed, bool):
        return f"claimed_verified holds {_shown(claimed)} where true or false belongs"
    return None


# A worker in one of these states is why a session would read "cancelled".  With
# none of them the word came from the quit itself.
_UNFINISHED_AGENTS = frozenset({"cancelled", "failed", "blocked"})


_TERMINAL_SESSION_STATUSES = frozenset({"completed", "failed", "cancelled", "stopped"})


def _pid_alive(pid: int) -> bool:
    """Treat an inaccessible process as alive; only report a known exit."""

    if sys.platform == "win32":
        # os.kill(pid, 0) terminates the process on Windows.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        open_process.restype = ctypes.c_void_p
        exit_code = kernel32.GetExitCodeProcess
        exit_code.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        exit_code.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        handle = open_process(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no process
        try:
            code = ctypes.c_ulong()
            return not exit_code(handle, ctypes.byref(code)) or code.value == 259
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _shown_status(session: dict[str, Any]) -> str:
    """The session's status as a word a reader can act on.

    Quitting the server writes ``cancelled`` on the session, whatever its
    workers did, so a run where every worker completed read as a cancellation
    and sent people looking for a failure that never happened.  When no worker
    was cancelled, failed or left blocked, say what really happened instead.

    Only the state each worker ended in counts.  Reading the whole history made
    a worker that was blocked, retried and then completed cancel a clean quit.

    The status file keeps its own value: another program reads it, and this
    changes the printed word alone.
    """

    status = str(session.get("status") or "unknown")
    if status not in _TERMINAL_SESSION_STATUSES:
        pid = session.get("server_pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            if not _pid_alive(pid):
                return "ended (server gone)"
    if status != "cancelled":
        return status
    states = session.get("agent_states") or set()
    if states & _UNFINISHED_AGENTS:
        return status
    return "ended (server quit)"


def _said(error: Any) -> Any:
    """A provider's structured error as the words in it.

    The Claude route records an object: a code or a subtype, a terminal
    reason and an HTTP status.  Printed whole, it read as a Python dict.
    """

    if not isinstance(error, dict):
        return error
    words = [str(error[key]) for key in ("code", "subtype", "terminal_reason")
             if isinstance(error.get(key), str) and error[key]]
    status = error.get("api_error_status")
    if isinstance(status, int) and not isinstance(status, bool):
        words.append(f"HTTP {status}")
    return ", ".join(dict.fromkeys(words)) or "the provider reported an error with no code"


def _print_failure(failure: dict[str, Any], *, full: bool) -> None:
    where = failure.get("model_id") or failure.get("provider")
    head = f"{failure['type']}" + (f" [{where}]" if where else "")
    print(f"      {head}: {_said(failure.get('error'))}")
    if failure.get("cause"):
        print(f"        cause: {failure['cause']}")
    stderr = failure.get("provider_stderr") or ()
    for line in list(stderr)[-_STDERR_LINES:]:
        print(f"        stderr: {line}")
    if full and failure.get("traceback"):
        for line in str(failure["traceback"]).splitlines():
            print(f"        {line}")


def _print_workspace(report: dict[str, Any], *, failures_only: bool, full: bool) -> bool:
    """Print one workspace.  True when something in it had failed."""

    sessions = report["sessions"]
    # A record the reader could not parse counts with the failures: a torn
    # last line can be the provider error that ended the session, and a
    # script asking --failures read "no failures recorded" over it.
    broken = [s for s in sessions if s["failures"] or s["unreadable"]]
    torn_outcomes = report.get("unreadable_outcomes") or []
    if failures_only and not broken and not torn_outcomes:
        return False
    print("")
    print(str(report["workspace"]))
    if not report["exists"]:
        print("  no .vnext here -- this workspace has never run a workforce")
        return False
    for session in (broken if failures_only else sessions):
        age = _age(time.time() - session["log"].stat().st_mtime)
        mark = (
            "FAILED" if session["failures"]
            else "PART UNREADABLE" if session["unreadable"]
            else _shown_status(session)
        )
        parent = session.get("parent")
        whose = f"  (started inside session {_readable_session(parent)})" if parent else ""
        print(
            f"  {session['session_id'][:8]}  {mark:<19} "
            f"{len(session['agents'])} agent{'s' if len(session['agents']) != 1 else ''}  "
            f"{session['records']} record{'s' if session['records'] != 1 else ''}  {age}"
            f"{whose}"
        )
        for cause in session["unreadable_causes"]:
            print(f"      {cause}")
        for failure in session["failures"]:
            _print_failure(failure, full=full)
        for note in session["notes"]:
            where = note.get("model_id") or note.get("provider")
            said = note.get("error") or {}
            reason = said.get("terminal_reason") if isinstance(said, dict) else said
            print(
                f"      note [{where}]: the stream closed after this agent had finished"
                + (f" ({reason})" if reason else "")
            )
    for cause in torn_outcomes:
        print(f"  {cause}")
    rows = report["outcomes"]
    if rows and not failures_only:
        billable = [row for row in rows if row.get("cost_source") != "in_parent"]
        priced = [
            row for row in billable
            if isinstance(row.get("cost_usd"), (int, float))
            and not isinstance(row.get("cost_usd"), bool)
        ]
        # A row with no price is a run whose bill never arrived, and adding it in
        # as zero printed $0.00 for a worker that had really cost $0.238.  Say
        # how many are unknown instead.
        unknown = len(billable) - len(priced)
        runs = f"{unknown} run{'s' if unknown != 1 else ''}"
        # Each part of the total keeps the label of where its price came from:
        # a provider's own figure printed as "at API rates" read as our estimate.
        parts: dict[str, float] = {}
        for row in priced:
            label = _COST_LABELS.get(row.get("cost_source"), _COST_LABELS[None])
            parts[label] = parts.get(label, 0.0) + row["cost_usd"]
        total = sum(parts.values())
        if len(parts) == 1:
            [(label, _)] = parts.items()
            money = f"${total:.2f} {label}"
        elif parts:
            money = f"${total:.2f} (" + ", ".join(
                f"${amount:.2f} {label}" for label, amount in parts.items()
            ) + ")"
        if not priced and not unknown:
            money = "cost included in parent"
        elif not priced:
            money = f"cost unknown for {runs}"
        elif unknown:
            money = f"{money}; cost unknown for {runs}"
        done = sum(1 for row in rows if row.get("status") == "completed")
        claimed = sum(1 for row in rows if row.get("claimed_verified"))
        print(
            f"  {len(rows)} finished agent{'s' if len(rows) != 1 else ''}, {done} completed, "
            f"{claimed} claiming verification, {money}"
        )
    return bool(broken or torn_outcomes)


# Where an outcome row's price came from, as the report prints it.  Rows
# written before cost_source existed say nothing, so the report says so too.
_COST_LABELS = {
    "provider": "as the provider reported",
    "api_rates": "at API rates",
    None: "from an older record that does not say",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vnext-report",
        description="What the vNext workforce did in each workspace, and why it stopped.",
    )
    parser.add_argument(
        "workspace", nargs="*",
        help="project directories to read (default: the current one)",
    )
    parser.add_argument(
        "--failures", action="store_true",
        help="print only the workspaces that went wrong, and exit non-zero if any did",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="print each failure's whole traceback",
    )
    args = parser.parse_args(argv)
    # Provider text can hold a lone surrogate the record keeps on purpose.
    # Printed raw to a UTF-8 terminal it stopped the report mid-failure, so
    # anything the stream cannot encode prints as its escape.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")

    # A folder that is not there and a folder that never ran a workforce used to
    # print the same line and exit 0, so a watcher kept passing after the project
    # was renamed or the path lost a letter.  Exit 2 says the question was wrong,
    # which is a different answer from "no failures".
    targets: list[Path] = []
    unusable: list[tuple[str, Path]] = []
    for given in args.workspace or ["."]:
        path = Path(given).expanduser().resolve()
        if path.is_dir():
            targets.append(path)
        else:
            unusable.append((given, path))
    if unusable:
        for given, path in unusable:
            reason = "is not a folder" if path.exists() else "does not exist"
            where = "" if str(path) == given else f" ({path})"
            print(f"vnext-report: {given} {reason}{where}", file=sys.stderr)
        return 2

    if not args.workspace:
        # With no path this read the folder it happened to be started in and
        # said nothing about which one, so a report from the wrong directory
        # looked like a report about the right one.
        print(f"reporting {targets[0]} (the current folder)")

    broken = False
    for target in targets:
        report = read_workspace(target)
        if _print_workspace(report, failures_only=args.failures, full=args.full):
            broken = True
    if args.failures and not broken:
        print("")
        print("no failures recorded")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
