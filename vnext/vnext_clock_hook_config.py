"""Build the Codex ``PostToolUse`` clock hook: its command, its flag, its table.

Three shapes of the same hook, kept together so they cannot disagree:

* ``hook_command_string`` -- what Codex runs after a tool call.
* ``hook_cli_args`` -- the ``-c`` flag the app-server is launched with, used
  once at startup so ``hooks/list`` can report the hook's ``currentHash``.
* ``hook_thread_config`` -- the ``hooks`` table merged into the ``config`` map
  of ``thread/start``, carrying the same hook plus the trust entry that lets
  Codex actually run it.

The trust hash is never hardcoded.  It covers the event, the matcher and the
normalized command string, so it changes whenever the command does -- the live
probe of 2026-09-24 watched ``67457c…`` become ``9f5646…`` after one edit to
the command.  ``--dangerously-bypass-hook-trust`` is never used: it would also
switch on every untrusted hook in the user's own ``~/.codex/hooks.json``.

Nothing here writes to ``~/.codex``.  The hook and its trust live only in the
thread's session-flags layer, which is what the probe demonstrated.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Mapping

# The synthetic source path Codex gives a hook supplied through session flags.
# The ``0:0`` tail is the group and handler index inside the PostToolUse list,
# so this key is right only while the clock is the one hook vNext passes.
TRUST_KEY = "/<session-flags>/config.toml:post_tool_use:0:0"
HOOK_TIMEOUT = 10


def hook_command_string(*, python: str | Path, script: str | Path, state_path: str | Path, limit: float) -> str:
    """Quote every path in the command Codex will run.

    The probe's first live turn failed with exit 126 because the interpreter
    path was split at a space in the repository path.  A repository path can
    still have a space in it, so the interpreter, the script and the state
    file are all quoted.
    """

    return " ".join([
        shlex.quote(str(python)),
        shlex.quote(str(script)),
        shlex.quote(str(state_path)),
        str(int(limit)),
    ])


def _handler(command: str) -> dict[str, Any]:
    return {"type": "command", "command": command, "timeout": HOOK_TIMEOUT}


def _toml_inline(value: Any) -> str:
    """Render a value as TOML inline syntax for a ``-c`` override.

    The ``-c`` parser reads TOML, not JSON.  Under ``--strict-config`` a JSON
    array of objects is rejected outright -- measured on the pinned 0.156.0
    binary, which answered `invalid type: string "[{\\"hooks\\"…}]", expected a
    sequence` and exited 1.  JSON string escaping matches TOML basic strings,
    so ``json.dumps`` still renders the leaves.
    """

    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}={_toml_inline(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_inline(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def hook_cli_args(command: str) -> list[str]:
    """Render the startup ``-c`` flag as two argv entries, never a shell string.

    The dotted key stops at ``hooks.PostToolUse``, which the ``-c`` parser
    handles.  The trust entry cannot go this way: its key contains
    ``config.toml`` and the parser splits the key at that dot, which is why the
    trust travels in the thread config instead.
    """

    return ["-c", "hooks.PostToolUse=" + _toml_inline([{"hooks": [_handler(command)]}])]


def hook_thread_config(command: str, trusted_hash: str) -> dict[str, Any]:
    """The ``hooks`` table for ``ThreadStartParams.config``, hook plus trust."""

    return {
        "hooks": {
            "PostToolUse": [{"hooks": [_handler(command)]}],
            "state": {TRUST_KEY: {"trusted_hash": trusted_hash}},
        }
    }


def merge_config(base: Mapping[str, Any] | None, addition: Mapping[str, Any]) -> dict[str, Any]:
    """Merge one config map into another without losing what is already there.

    Command Code rides this same ``config`` map and carries the reseller's
    bearer token in it.  A replace would break its authentication outright, so
    the merge is recursive and the existing value wins on a leaf collision.
    """

    merged: dict[str, Any] = dict(base or {})
    for key, value in addition.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_config(existing, value)
        elif key not in merged:
            merged[key] = value
    return merged


def trusted_hash_from_hooks_list(result: Mapping[str, Any], command: str) -> str | None:
    """Pick this hook's ``currentHash`` out of a ``hooks/list`` result.

    The hash is trusted under ``TRUST_KEY``, so only the hook listed under that
    key with exactly this command qualifies.  A looser match could hand back
    another hook's hash, and Codex would then refuse to run the clock.
    """

    data = result.get("data")
    if not isinstance(data, (list, tuple)):
        return None
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        hooks = entry.get("hooks")
        if not isinstance(hooks, (list, tuple)):
            continue
        for hook in hooks:
            if not isinstance(hook, Mapping):
                continue
            if str(hook.get("eventName", "")).lower() != "posttooluse":
                continue
            if hook.get("command") != command:
                continue
            if hook.get("key", TRUST_KEY) != TRUST_KEY:
                continue
            current = hook.get("currentHash")
            if isinstance(current, str) and current:
                return current
    return None
