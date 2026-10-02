"""One place that decides what a ``[clock]`` line looks like.

Three separate code paths print this line: the scheduler writes it into a
turn-start prompt, the Claude bridge returns it from a ``PostToolUse``
callback hook, and ``vnext.clock_hook`` prints it from a Codex
command hook.  Three copies of the same ``strftime`` call would drift, so
they all call in here.

Standard library only, and no import from the rest of the package: the Codex
hook runs as a bare script under whatever interpreter the command string
names.
"""

from __future__ import annotations

import datetime

PREFIX = "[clock]"


def format_time(moment: float) -> str:
    """Render a POSIX timestamp as local wall-clock time, e.g. ``21:14 CEST``."""

    return datetime.datetime.fromtimestamp(moment).astimezone().strftime("%H:%M %Z")


def format_duration(seconds: float) -> str:
    """Render an elapsed span as ``45s``, ``2m 03s`` or ``2h 14m``.

    The unit pair changes with the magnitude so the line stays short.  A span
    under a minute prints seconds alone; a minute or more prints minutes and
    padded seconds; an hour or more drops the seconds, because at that scale
    they say nothing a reader acts on.
    """

    total = int(max(0.0, float(seconds)))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def _budget(limit: float | None) -> str:
    return f"{int(limit)}s" if limit is not None else "?"


def turn_start_line(
    moment: float,
    *,
    elapsed: float = 0.0,
    limit: float | None,
    agent_age: float | None = None,
) -> str:
    """The line a worker reads at the top of its turn.

    The turn is the figure the worker has to act on, so it comes first.  The
    agent's age rides behind it: it answers a different question, and an
    agent on its fifth turn would otherwise read hours where the turn is
    seconds old.
    """

    line = f"{PREFIX} {format_time(moment)} · turn {format_duration(elapsed)} / {_budget(limit)}"
    if agent_age is not None:
        line += f" · agent running {format_duration(agent_age)}"
    return line


def mid_turn_line(moment: float, *, elapsed: float | None, limit: float | None) -> str:
    """The line a worker reads after a tool call.

    ``elapsed`` is ``None`` when the turn clock is not known -- the Claude
    bridge loses it between a resume and the next ``start_turn``.  The line
    then carries the wall-clock half alone rather than raising or inventing a
    number.
    """

    if elapsed is None:
        return f"{PREFIX} {format_time(moment)}"
    return f"{PREFIX} {format_time(moment)} · turn {format_duration(elapsed)} / {_budget(limit)}"
