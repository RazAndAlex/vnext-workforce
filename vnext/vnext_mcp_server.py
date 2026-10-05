"""Serve one vNext control tree to a coding client that vNext does not run.

Claude Code is the first such client.  It keeps its own conversation, its own
permission system and its own user; what it does not have is a workforce.  This
module gives it one: a long-lived loopback MCP endpoint whose tools are the
vNext manager tools, backed by a session whose root is the client itself.

The endpoint is deliberately boring.  A client a person configures by hand needs
an address and a credential that survive a restart, so both are arguments rather
than per-run secrets, and the service holds exactly one session for one
workspace.  Everything interesting -- delegation, waiting, steering, evidence --
already exists in the scheduler and is simply exposed here.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .clock_format import format_time
from .host_contract import SessionStartRequest
from .vnext_model_identity import UNKNOWN as EXACT_MODEL_UNKNOWN, resolve_claude_alias
from .release_check import RELEASE_CODEX_MODEL_COMPATIBILITY, RELEASE_CODEX_VERSION
from .vnext_codex_mcp import LOOPBACK_HOSTS, CodexMcpRelay, McpProtocol
from .vnext_orchestration import SESSION_CLOSING_CODE, session_closing_message
from .vnext_provider_config import (
    describe_provider_config_problem,
    load_commandcode_provider,
    load_zai_provider,
    provider_config_problems,
)
from .vnext_runtime_effects import (
    BUILT_IN_EFFECT_READER_PROVIDERS,
    effect_reader_hint,
)
from .vnext_runtime_types import ToolCallResult
from .vnext_runtimes import (
    NO_UPDATE_CHECK_ENV,
    auto_update,
    auto_update_needed,
    promote_staged,
    register_runtime_in_use,
    missing_codex_catalog_models,
    is_runtime_command,
    active_codex_slugs,
    check_for_updates,
    claude_runtime_selection,
    codex_runtime_selection,
    read_active_runtime,
    runtime_notice_lines,
    read_staged_runtime,
    runtime_view,
    session_runtime_config,
    untested_catalog_entries,
)
from .vnext_session_runtime import PROVIDERS, VNextRuntimeSession, _catalog_claims


# A starting point rather than a policy: these are the models this project has
# priced and exercised.  Pass --catalog to serve a different set; the root
# cannot delegate to a model that is not in the catalog it was started with.
#
# The GPT-6 models are here because the pinned 0.160.0 runtime runs them.  The
# catalog remains a claim that the pinned runtime can run every listed model;
# release_check.py owns the explicit compatibility table used to enforce that
# claim before a session starts.
DEFAULT_CATALOG: tuple[dict[str, Any], ...] = (
    {"provider": "codex", "model": "gpt-6.1-sol"},
    {"provider": "codex", "model": "gpt-6-sol"},
    {
        "provider": "codex",
        "model": "gpt-6-luna",
        "claims": [{
            "kind": "observation",
            "statement": (
                "At effort low or medium gpt-6-luna often answers read-and-report tasks at once, "
                "with no reasoning and no tool call. At effort high it used tools in 26 of 26 "
                "measured runs at about the same token cost. Delegate it with effort high."
            ),
            "evidence_pointer": "26 measured runs of the gpt-6-luna effort comparison",
        }],
    },
    {"provider": "codex", "model": "gpt-6-astra"},
    {"provider": "codex", "model": "gpt-5.6-sol"},
    {"provider": "codex", "model": "gpt-5.6-terra"},
    {"provider": "codex", "model": "gpt-5.6-luna"},
    {"provider": "claude", "model": "sonnet"},
    {"provider": "claude", "model": "opus"},
    {"provider": "claude", "model": "fable"},
    {"provider": "zai", "model": "glm-5.3-flash"},
    {"provider": "zai", "model": "glm-5.3"},
    {"provider": "zai", "model": "glm-5.2"},
    {"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"},
)

SERVER_INFO = {"name": "vnext", "version": "1"}

# "<session id>:<pid>" of the session that owns the environment.  A server that
# finds another process's id here started somewhere under that session, and says
# so in its status file.  The value sits in os.environ, so every descendant
# process inherits it, including programs that are nobody's worker: the marker
# attests where a server started and never whose worker it is.
_PARENT_SESSION_ENV = "VNEXT_PARENT_SESSION"

# Sentinel so "no status file" can be said explicitly and still leave the
# default -- a file beside the workspace -- as what happens when nobody asks.
_DEFAULT = object()
_TERMINAL = frozenset({"completed", "failed", "cancelled", "replaced"})

# The tools that put a provider agent into a turn: delegate and replace build a
# worker outright, retry starts one again, and send_message to an idle worker
# wakes it into one.  Everything else reads the roster or stops something, and
# stays available while a close runs.
_TOOLS_THAT_START_A_TURN = frozenset({"delegate", "replace", "retry", "send_message"})
_BUILT_IN_PROVIDERS = frozenset(PROVIDERS)
# The harness name PROVIDERS gives a provider that runs through the Claude Agent
# SDK.  Reading it from PROVIDERS keeps the two in step: a provider added there
# with this harness is covered without a second list to remember.
_CLAUDE_SDK_HARNESS = "claude-agent-sdk"
_CLAUDE_SDK_MODULE = "claude_agent_sdk"
CLAUDE_SDK_PROVIDERS = frozenset(
    name for name, (harness, _credential) in PROVIDERS.items()
    if harness == _CLAUDE_SDK_HARNESS
)
# Said the way the README's install step says it, because that is the fix.
CLAUDE_SDK_MISSING = (
    "the Claude SDK is not installed; reinstall with uv tool install \".[claude]\""
)


def _claude_sdk_installed() -> bool:
    """Is the Claude Agent SDK importable, asked without importing it.

    A login was the whole availability test, so an install without the
    ``[claude]`` extra still offered opus, sonnet and the glm models: delegate
    was accepted and the worker died with ModuleNotFoundError inside the
    provider.  find_spec answers at a fraction of the import's cost and leaves
    the import to the bridge that needs it.
    """

    try:
        return importlib.util.find_spec(_CLAUDE_SDK_MODULE) is not None
    except (ImportError, ValueError):
        return False
_PROVIDER_REGISTRY_LOCK = threading.Lock()
# How many stdio tool calls may run at once.  One long await_children plus the
# handful of calls a client makes while it waits is the case this serves; past
# that, answering on the reading thread is the backpressure.
_MAX_CONCURRENT_STDIO_CALLS = 16

# The manager tool descriptions are written for a manager vNext runs, and one of
# them describes a mechanic that does not apply to a client it does not run: a
# provider manager yields its turn to wait, while an external one is answered by
# a call that blocks.  Correct that where it is served rather than in the
# scheduler, because both statements are true of their own caller.
EXTERNAL_TOOL_DESCRIPTIONS = {
    "await_children": (
        "Block until the selected direct children settle, then return their "
        "outcomes: every child that has stopped carries the text it reported in "
        "outcome, with verified and evidence beside it. A long wait returns "
        "settled=false with the children still "
        "running; call this again with the same agent_ids to keep waiting. A "
        "blocked child is not waiting on anything except a decision from you, "
        "so waiting only for stopped children is refused."
    ),
}


# A client vNext runs is handed the catalog in its developer instructions.  A
# client vNext does not run receives the tool listing and nothing else, so the
# only place it can read a model name is here.  Without it a manager spells the
# id out of the words its user typed: a ZCode session asked for "deepseek v4.1
# flash" spent three delegate calls on deepseek-v4.1-flash, deepseek-v4.1 and
# deepseek-v4-flash, never reached deepseek/deepseek-v4.1-flash, and told its
# user DeepSeek was unavailable while the card was live the whole time.
_MODEL_ROSTER_PREAMBLE = (
    " Spell model_id exactly as this session names it below; a name assembled "
    "from how a person said it is refused. Available model_id values, with the "
    "provider and harness each one runs on: "
)
# A client keeps the tool listing it read at its own start, so the roster below
# can be older than the server answering a delegate.
_MODEL_ROSTER_EPILOGUE = (
    " This list was read when this session started; a refusal for an unknown "
    "model lists what the server offers now."
)


def _model_roster(entries: Sequence[Mapping[str, Any]]) -> str:
    """Name every delegable model, so no manager has to invent one."""

    named = []
    for entry in entries:
        model = entry.get("model") or entry.get("model_id")
        provider = str(entry.get("provider") or "")
        harness = PROVIDERS.get(provider, (None, None))[0]
        runs_on = f"{provider} via {harness}" if harness else provider
        claims = entry.get("claims")
        annotations = []
        if isinstance(claims, list):
            annotations = [
                f"{claim['kind']}: {claim['statement']}"
                for claim in claims
                if isinstance(claim, Mapping)
                and claim.get("kind")
                and claim.get("statement")
            ]
        suffix = f"; {'; '.join(annotations)}" if annotations else ""
        named.append(f"{model} ({runs_on}{suffix})")
    return _MODEL_ROSTER_PREAMBLE + "; ".join(named) + "." + _MODEL_ROSTER_EPILOGUE


def _external_tools(
    tools: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    roster = _model_roster(entries) if entries else ""
    projected = []
    for tool in tools:
        entry = dict(tool)
        name = str(entry.get("name"))
        replacement = EXTERNAL_TOOL_DESCRIPTIONS.get(name)
        if replacement is not None:
            entry["description"] = replacement
        if name == "delegate":
            entry["description"] = str(entry.get("description", "")) + (
                " Claude Code with the vNext plugin wakes the manager automatically when the child stops."
                " Other clients can run the returned wake_command in the background "
                "to be woken when the child stops or its deadline passes."
            )
        # Both tools that choose a model carry the roster.
        if roster and name in {"delegate", "replace"}:
            entry["description"] = str(entry.get("description", "")) + roster
        projected.append(entry)
    return tuple(projected)

# The folder this process imported its vNext code from.  A server keeps the
# code it started with until restart_server, so a release merged into this
# folder afterwards is on disk and absent from the process.
_PACKAGE_FOLDER = Path(__file__).resolve().parent
# How long one look at the folder is trusted.  The look happens only when an
# answer needs it (inspect self, an unknown-model refusal), never on a timer.
_CODE_RECHECK_SECONDS = 5.0


def _code_fingerprint(folder: Path) -> tuple[float, int] | None:
    """Newest modification time and count of the ``.py`` files in ``folder``.

    One stat per file and no hashing.  ``None`` when the folder cannot be read,
    such as an installed wheel laid out some other way: an unknown answer is
    treated as "not newer", so the check never invents a restart.
    """

    try:
        newest = 0.0
        count = 0
        for entry in os.scandir(folder):
            if entry.name.endswith(".py") and entry.is_file():
                newest = max(newest, entry.stat().st_mtime)
                count += 1
    except OSError:
        return None
    return (newest, count) if count else None


class CodeFreshness:
    """Answer whether the vNext code on disk is newer than what this process loaded.

    The fingerprint is taken once at construction and compared, at most once
    per ``recheck_after`` seconds, with a fresh one.  It only says what
    changed and which call loads it; restarting stays the caller's decision,
    because a restart stops the workers that are running.
    """

    def __init__(
        self,
        folder: Path,
        *,
        recheck_after: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._folder = Path(folder)
        self._recheck_after = (
            _CODE_RECHECK_SECONDS if recheck_after is None else recheck_after
        )
        self._clock = clock
        self.started_at = time.time()
        self._loaded = self._read()
        self._checked_at: float | None = None
        self._newer = False
        self._lock = threading.Lock()

    def _read(self) -> tuple[float, int] | None:
        try:
            return _code_fingerprint(self._folder)
        except Exception:
            return None

    def newer_on_disk(self) -> bool:
        if self._loaded is None:
            return False
        with self._lock:
            now = self._clock()
            if (
                self._checked_at is None
                or now - self._checked_at >= self._recheck_after
            ):
                self._checked_at = now
                current = self._read()
                self._newer = current is not None and (
                    current[0] > self._loaded[0] or current[1] != self._loaded[1]
                )
            return self._newer

    def stale_notice(self) -> str | None:
        if not self.newer_on_disk():
            return None
        return (
            "newer vNext code is on disk than this server loaded at "
            f"{format_time(self.started_at)}; restart_server loads it, "
            "and running workers stop when it does"
        )


ROOT_INSTRUCTIONS = (
    "You are the Root Manager of a vNext workforce. Delegate bounded work to "
    "worker models, await their results, and judge the evidence they return."
)


class VNextMcpServiceError(RuntimeError):
    pass


def _missing_workspace(path: Path) -> str:
    """Name why a workspace path cannot be used: absent, or present as something other than a folder."""

    if path.exists():
        return f"workspace is not a folder: {path}"
    return f"workspace does not exist: {path}"


# How long a close gets in total before it gives up and reports what is left.
#
# The restart proxy that keeps a client connected across a server restart is
# reloaderoo, pinned to 1.1.5 in plugins/vnext/reload/package-lock.json.  Its
# terminateChild sends SIGTERM and arms a 5000 ms timer to follow with SIGKILL
# (dist/process-manager.js).  Measured against that pinned build on 2026-09-30
# with a child that ignores SIGTERM: the timer fires at 5 s and then does
# nothing, because it is guarded by `!child.killed` and Node sets `killed` the
# moment kill() is called, so the escalation never arrives and the proxy waits
# without end.  Five seconds is therefore the grace the proxy means to give,
# and this close has to finish inside it on its own.
CLOSE_GRACE_SECONDS = 5.0

# How much of that grace the stdio quit may spend letting open tool calls
# finish before it abandons them.
#
# A client closes the pipe to stop this server, and the calls it left open are
# answered on their own threads.  Joining them without a bound made the quit as
# long as the longest call: measured on 2026-09-30, EOF with one 50 s
# `await_children` open returned from serve_stdio after 50.06 s and the process
# exited after 50.11 s, ten times the grace the restart proxy gives.  The
# client that would have read those answers is the one that just went away, so
# a call still running at the drain deadline is left to its daemon thread and
# what remains of the grace goes to the close.
_EOF_DRAIN_SECONDS = 1.0


class _CloseStep:
    """One step of a close, run off the caller's thread so a hang cannot hold it.

    A step that overruns keeps its thread, so a later close attempt waits on
    the same work again rather than starting a second copy of it.

    Deciding whether that thread exists is itself guarded.  Two callers that
    asked at once both read ``_thread`` as None and both started one, so the
    endpoint was stopped twice; and a caller that read the attribute after the
    other had assigned it but before it had started it joined a thread that had
    never run, which raises RuntimeError.  Both were measured on 2026-09-30.
    The guard covers the decision only: the join happens outside it, so a step
    that hangs holds no lock.
    """

    def __init__(self, label: str, work: Callable[[], None]) -> None:
        self.label = label
        self.finished = False
        self.error: BaseException | None = None
        self._work = work
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run(self, budget: float) -> bool:
        """Wait up to ``budget`` seconds; True once the step has run to an end."""

        with self._lock:
            if self.finished and self.error is None:
                return True
            if self.finished:
                # A step that raised is tried again by the next close attempt.
                self.finished = False
                self.error = None
                self._thread = None
            if self._thread is None:
                thread = threading.Thread(
                    target=self._call, name=f"vnext-close-{self.label}", daemon=True
                )
                thread.start()
                self._thread = thread
            running = self._thread
        running.join(timeout=max(0.0, budget))
        if running.is_alive():
            return False
        self.finished = True
        return True

    def _call(self) -> None:
        try:
            self._work()
        except BaseException as exc:  # noqa: BLE001 - carried to the caller
            self.error = exc


# The credential shapes the pinned Codex CLI writes into auth.json.  An
# API-key login writes "OPENAI_API_KEY"; a ChatGPT login writes a "tokens"
# object carrying "access_token", "refresh_token" and "id_token".  Both are
# read as key NAMES alone: nothing here ever reports a credential value.
_CODEX_API_KEY_FIELD = "OPENAI_API_KEY"
_CODEX_TOKEN_FIELDS = ("access_token", "refresh_token", "id_token")


def _codex_login_problem() -> str | None:
    """The reason the local Codex login is unusable, or None when it is usable.

    The bare existence of auth.json used to count, so an empty or half-written
    file advertised six Codex models that could only fail after a manager had
    delegated to them.
    """

    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    path = home / "auth.json"
    if not path.is_file():
        return f"{path} does not exist; run 'codex login'"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{path} could not be read: {exc}"
    try:
        value = json.loads(raw)
    except ValueError:
        return f"{path} does not parse as JSON; run 'codex login' again"
    if not isinstance(value, Mapping):
        return f"{path} does not hold a JSON object; run 'codex login' again"
    if isinstance(value.get(_CODEX_API_KEY_FIELD), str) and value[_CODEX_API_KEY_FIELD].strip():
        return None
    tokens = value.get("tokens")
    if isinstance(tokens, Mapping):
        for field in _CODEX_TOKEN_FIELDS:
            held = tokens.get(field)
            if isinstance(held, str) and held.strip():
                return None
    return (
        f"{path} holds no credential: it needs {_CODEX_API_KEY_FIELD} or a "
        f"tokens object with one of {', '.join(_CODEX_TOKEN_FIELDS)}; run 'codex login'"
    )


def _codex_login_available() -> bool:
    return _codex_login_problem() is None


_CLAUDE_LOGIN_TIMEOUT_SECONDS = 15.0

CLAUDE_LOGIN_AVAILABLE = "available"
CLAUDE_LOGIN_ABSENT = "absent"
CLAUDE_LOGIN_UNCHECKED = "unchecked"


def _is_windows() -> bool:
    """The platform decision, in one place a test can simulate.

    ``os.name`` itself cannot stay patched while these functions run: pathlib
    dispatches ``Path`` on it and refuses to build a ``WindowsPath`` on a POSIX
    host, so a test that patched it could not construct the very paths it was
    checking.
    """

    return os.name == "nt"


def _pathext() -> tuple[str, ...]:
    """The extensions Windows appends to an extensionless command name."""

    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    return tuple(part for part in raw.split(";") if part.strip())


def _claude_install_locations() -> tuple[Path, ...]:
    """Where the official Claude Code installers put the CLI.

    PATH is asked first and this list second, for the case where PATH holds
    none of these folders.

    The native installer writes ``~/.local/bin/claude`` on macOS -- a symlink
    into ``~/.local/share/claude/versions/`` -- and
    ``%USERPROFILE%\\.local\\bin\\claude.exe`` on Windows. A global npm
    install writes into ``{prefix}/bin`` on macOS and straight into the prefix
    on Windows, where the default prefix is ``%APPDATA%\\npm`` and the shim is
    a ``.cmd``. ``~/.claude/local`` is the older local install.
    """

    try:
        home: Path | None = Path.home()
    except RuntimeError:
        # No HOME, USERPROFILE or password entry: only the folders that do
        # not hang off the home folder are left to look in.
        home = None
    if _is_windows():
        candidates = [] if home is None else [
            home / ".local" / "bin" / "claude.exe",
            home / ".local" / "bin" / "claude.cmd",
            home / ".claude" / "local" / "claude.exe",
            home / ".claude" / "local" / "claude.cmd",
        ]
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates += [
                Path(appdata) / "npm" / "claude.cmd",
                Path(appdata) / "npm" / "claude.exe",
            ]
        return tuple(candidates)
    in_home = () if home is None else (
        home / ".local" / "bin" / "claude",
        home / ".claude" / "local" / "claude",
    )
    return (
        *in_home,
        Path("/opt/homebrew/bin/claude"),
        Path("/usr/local/bin/claude"),
    )


def _claude_executable() -> str | None:
    """Resolve the ``claude`` CLI without trusting the inherited PATH.

    A bare ``subprocess`` call with ``shell=False`` reaches ``CreateProcess``
    on Windows, which appends ``.exe`` to an extensionless name and walks no
    further, so the installer's ``claude.cmd`` shim stayed invisible.
    ``shutil.which`` is the repo's own answer to that
    (``native_terminal_process.py:126``); each PATHEXT entry is then named in
    turn, which keeps the same walk reachable on a simulated Windows host
    where CPython's win32 branch cannot run.
    """

    found = shutil.which("claude")
    if found:
        return found
    if _is_windows():
        for extension in _pathext():
            found = shutil.which("claude" + extension.lower())
            if found:
                return found
    for candidate in _claude_install_locations():
        if candidate.is_file():
            return str(candidate)
    return None


def _claude_login_state() -> str:
    """"available", "absent", or "unchecked" when the CLI did not answer.

    "unchecked" was being reported as "not signed in": a loaded machine cost a
    signed-in user their whole Claude fleet and told them to sign in to a CLI
    they were already signed in to.
    """

    executable = _claude_executable()
    if executable is None:
        return CLAUDE_LOGIN_ABSENT
    try:
        result = subprocess.run(
            [executable, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_CLAUDE_LOGIN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CLAUDE_LOGIN_UNCHECKED
    except (OSError, subprocess.SubprocessError):
        return CLAUDE_LOGIN_ABSENT
    try:
        signed_in = json.loads(result.stdout).get("loggedIn") is True
    except (ValueError, AttributeError):
        return CLAUDE_LOGIN_ABSENT
    if result.returncode == 0 and signed_in:
        return CLAUDE_LOGIN_AVAILABLE
    return CLAUDE_LOGIN_ABSENT


def _claude_login_available() -> bool:
    return _claude_login_state() == CLAUDE_LOGIN_AVAILABLE


class _LoginProbe:
    """Asks each built-in provider about its local login at most once.

    Startup used to ask twice -- ``_load_catalog`` validated the default
    catalog and ``VNextMcpService.__init__`` validated it again -- so a slow
    ``claude`` was paid for twice. One probe handed to both call sites asks
    once.
    """

    def __init__(self) -> None:
        self._codex: bool | None = None
        self._claude: str | None = None

    def codex_available(self) -> bool:
        if self._codex is None:
            self._codex = _codex_login_available()
        return self._codex

    def claude_state(self) -> str:
        if self._claude is None:
            self._claude = _claude_login_state()
        return self._claude

    def claude_available(self) -> bool:
        return self.claude_state() == CLAUDE_LOGIN_AVAILABLE


def _validate_catalog(
    catalog: Sequence[Mapping[str, Any]],
    adapter_factories: Mapping[str, Any] | None = None,
    login: "_LoginProbe | None" = None,
    catalog_path: str | None = None,
) -> list[dict[str, Any]]:
    factories = adapter_factories or {}
    entries: list[dict[str, Any]] = []
    backed = _BUILT_IN_PROVIDERS | set(factories)
    location = f'--catalog "{catalog_path}"' if catalog_path is not None else "catalog"
    for position, raw in enumerate(catalog, start=1):
        prefix = f"{location}: entry {position}"
        if not isinstance(raw, Mapping):
            raise VNextMcpServiceError(
                f"{prefix}: each entry must be an object with provider and model"
            )
        entry = dict(raw)
        # The runtime reads model_id as another name for model, so both pass.
        model = entry.get("model") or entry.get("model_id")
        if not isinstance(model, str) or not model.strip():
            raise VNextMcpServiceError(f"{prefix} requires a non-empty string model")
        provider = entry.get("provider")
        if not isinstance(provider, str) or provider not in backed:
            raise VNextMcpServiceError(
                f"{prefix} (model {model!r}) names provider {provider!r}, which is neither "
                f"built in ({', '.join(sorted(_BUILT_IN_PROVIDERS))}) nor registered; "
                f"pass --provider NAME=module:attribute"
            )
        if "harness" in entry:
            harness = PROVIDERS.get(provider, ("registered-adapter",))[0]
            if entry["harness"] != harness:
                raise VNextMcpServiceError(f"{prefix} has an unsupported harness for {provider}")
        try:
            _catalog_claims(entry)
        except ValueError as exc:
            raise VNextMcpServiceError(f"{prefix}: {exc}") from exc
        entries.append(entry)
    probe = login or _LoginProbe()
    # A provider that runs on the Claude Agent SDK needs the SDK present as much
    # as it needs a credential.  Without the [claude] extra the import fails
    # inside the worker, after a manager has spent a delegate call on it.
    if not _claude_sdk_installed():
        entries = [
            entry
            for entry in entries
            if entry.get("provider") not in CLAUDE_SDK_PROVIDERS
            or entry.get("provider") in factories
        ]
    # Only advertise built-in providers that have a usable local login.
    # Explicit adapter factories own their own credential checks.
    if (
        "codex" not in factories
        and any(entry.get("provider") == "codex" for entry in entries)
        and not probe.codex_available()
    ):
        entries = [entry for entry in entries if entry.get("provider") != "codex"]
    if (
        "claude" not in factories
        and any(entry.get("provider") == "claude" for entry in entries)
        and not probe.claude_available()
    ):
        entries = [entry for entry in entries if entry.get("provider") != "claude"]
        if probe.claude_state() == CLAUDE_LOGIN_UNCHECKED:
            # stdout carries the protocol in stdio mode, so this goes to stderr,
            # which the run record picks up.
            print(
                "vnext: the Claude login could not be checked -- "
                f"'claude auth status' did not answer within "
                f"{_CLAUDE_LOGIN_TIMEOUT_SECONDS:.0f}s. Claude worker models are "
                "left out of this session's roster; you may well still be signed in.",
                file=sys.stderr,
            )
    # A missing local provider credential removes z.ai cards rather than
    # leaving models that can only fail after a manager delegates to them.
    if (
        any(entry.get("provider") == "zai" for entry in entries)
        and load_zai_provider() is None
    ):
        entries = [entry for entry in entries if entry.get("provider") != "zai"]
    # Command Code answers for its own catalog, so its cards go the same way
    # when the account key is absent.
    if (
        any(entry.get("provider") == "commandcode" for entry in entries)
        and load_commandcode_provider() is None
    ):
        entries = [entry for entry in entries if entry.get("provider") != "commandcode"]
    if not entries:
        message = (
            "a workforce needs at least one available worker model; sign in to "
            "Codex or Claude, or configure a Z.ai or Command Code provider key"
        )
        if not _claude_sdk_installed():
            message += f"; claude and zai models are left out because {CLAUDE_SDK_MISSING}"
        # A file that exists and is wrong used to read as a file that is absent,
        # so the one message a person is sent to told them to configure the key
        # they had already written.
        for fault in provider_config_problems():
            message += f"; {fault}"
        if probe.claude_state() == CLAUDE_LOGIN_UNCHECKED:
            message += (
                "; the Claude login could not be checked, because "
                f"'claude auth status' did not answer within "
                f"{_CLAUDE_LOGIN_TIMEOUT_SECONDS:.0f}s"
            )
        raise VNextMcpServiceError(message)

    # A direct Python caller may deliberately replace the built-in Codex
    # adapter (the test harness does).  In that case the pinned CLI is not the
    # runtime that will receive the model, so its allowlist does not apply.
    #
    # The allowlist names the models the pinned CLI has built in, so it is
    # asked about "codex" cards alone.  A "commandcode" card names a model the
    # CLI has never heard of and reaches through a configured provider, which
    # is a path every Codex release since custom providers landed supports.
    uses_pinned_codex = "codex" not in (adapter_factories or {})
    supported = RELEASE_CODEX_MODEL_COMPATIBILITY[RELEASE_CODEX_VERSION]
    runtime_label = f"pinned Codex CLI {RELEASE_CODEX_VERSION}"
    # A side runtime installed by --update-runtimes answers for itself: the
    # models its own model/list named when it was installed.
    active = read_active_runtime()
    side_slugs = active_codex_slugs(active)
    if side_slugs is not None:
        supported = side_slugs
        runtime_label = f"side Codex runtime ({runtime_view(active)['codex']})"
    if uses_pinned_codex:
        for model in missing_codex_catalog_models(entries, supported):
            raise VNextMcpServiceError(
                f"Codex model {model!r} cannot be run by the {runtime_label}"
            )
    return entries


def _load_provider_factories(
    registrations: Sequence[str],
) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {}
    expected = "NAME=module:attribute"
    for registration in registrations:
        name, separator, target = registration.partition("=")
        module_name, target_separator, attribute = target.partition(":")
        if (
            not separator
            or not target_separator
            or not name.strip()
            or not module_name.strip()
            or not attribute.strip()
            or ":" in attribute
        ):
            raise VNextMcpServiceError(
                f"malformed --provider value {registration!r}; expected {expected}"
            )
        name = name.strip()
        module_name = module_name.strip()
        attribute = attribute.strip()
        if name in _BUILT_IN_PROVIDERS:
            raise VNextMcpServiceError(
                f"registered provider {name!r} collides with a built-in provider"
            )
        if name in factories:
            raise VNextMcpServiceError(f"registered provider {name!r} was supplied more than once")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise VNextMcpServiceError(
                f"registered provider {name!r} module {module_name!r} could not be imported; "
                f"expected {expected}"
            ) from exc
        try:
            factory = getattr(module, attribute)
        except AttributeError as exc:
            raise VNextMcpServiceError(
                f"registered provider {name!r} module {module_name!r} has no attribute "
                f"{attribute!r}; expected {expected}"
            ) from exc
        if not callable(factory):
            raise VNextMcpServiceError(
                f"registered provider {name!r} attribute {module_name}:{attribute} is not "
                f"callable; expected a zero-argument adapter factory"
            )
        # A registered provider has no built-in effect reader, and a worker
        # whose effects cannot be decoded blocks on its first turn.  The two
        # ways an adapter answers for its own effects are both class
        # attributes, so the refusal is cheap and legible here rather than a
        # blocker an hour into a session.
        if effect_reader_hint(factory) is None:
            raise VNextMcpServiceError(
                f"registered provider {name!r} has no runtime effect reader; give "
                f"{module_name}:{attribute} a 'runtime_effect_reader' attribute, or a "
                f"'provider' naming the built-in provider whose event shape it speaks "
                f"({', '.join(sorted(BUILT_IN_EFFECT_READER_PROVIDERS))})"
            )
        factories[name] = factory
    return factories


@dataclass(frozen=True)
class ServiceAddress:
    """Everything a client needs in order to be pointed at this service."""

    endpoint: str
    token: str

    def claude_mcp_add(self, name: str = "vnext") -> str:
        return (
            f"claude mcp add --transport http {name} {self.endpoint} "
            f'--header "Authorization: Bearer {self.token}"'
        )


class VNextMcpService:
    """One workspace, one external-primary session, one MCP endpoint."""

    # The session this one started inside, when it started under another.  Named
    # on the class so a probe that builds the object without __init__ still has
    # it.
    _parent_session: str | None = None

    def __init__(
        self,
        *,
        workspace: str | Path,
        catalog: Sequence[Mapping[str, Any]] = DEFAULT_CATALOG,
        client: str = "claude-code",
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        session_id: str | None = None,
        event_log: str | Path | None = _DEFAULT,  # type: ignore[assignment]
        status_file: str | Path | None = _DEFAULT,  # type: ignore[assignment]
        outcome_log: str | Path | None = _DEFAULT,  # type: ignore[assignment]
        instructions: str = ROOT_INSTRUCTIONS,
        adapter_factories: Mapping[str, Any] | None = None,
        login_probe: "_LoginProbe | None" = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise VNextMcpServiceError(_missing_workspace(self.workspace))
        entries = _validate_catalog(catalog, adapter_factories, login=login_probe)
        self._client = client
        self._log_lock = threading.Lock()
        # Records whose last write failed.  Each writer tries again on the next
        # change, so a disk freed a second later or a reader that let go of the
        # file costs only the writes in between.
        self._failing_writes: set[Path] = set()
        self._closed = False
        # Set the moment a close starts and never cleared: a workforce step that
        # overruns its budget keeps emitting events, and each one used to
        # republish the roster as live after the close had published it as
        # stopped.  Whatever draws the status file then shows a live session
        # nobody can reach.
        self._closing = False
        # Set by a close that ran out of time or failed, so a later agent event
        # republishes the roster as stopping and keeps counting its workers.
        self._stopping = False
        self._close_steps: list[_CloseStep] | None = None
        # One close at a time.  Concurrent callers wait on this one rather than
        # each starting a copy of the work.
        self._close_lock = threading.Lock()
        # What the stdio quit left of the grace, once it has drained its open
        # calls.  Set by serve_stdio, spent by the first close after it.
        self._stdio_quit_budget: float | None = None
        self._session_id = session_id or str(uuid.uuid4())
        # A Claude worker runs a real Claude Code, which loads the user's own
        # plugins and starts a second vNext server inside the worker, in the
        # same project.  Its records used to sit beside this session's with
        # nothing to tell them apart.  The marker travels in the environment the
        # worker inherits, so the nested server can name the session it started
        # inside in its own status file.  Every other descendant of this process
        # inherits it too, which is why the claim stops at where the server
        # started.  The pid is part of it: two sessions in one process started
        # side by side, so neither is inside the other.
        inherited = os.environ.get(_PARENT_SESSION_ENV, "")
        parent, _, pid = inherited.rpartition(":")
        self._parent_session = parent if parent and pid != str(os.getpid()) else None
        os.environ[_PARENT_SESSION_ENV] = f"{self._session_id}:{os.getpid()}"

        # One file per session, never one per project.  Several Claude Code
        # windows open the same project at once, each with a server of its own,
        # and they were all appending to one log: 4 of 4328 records in this
        # project's first log were left truncated and NUL-padded by overlapping
        # appends, and no record said which session it came from.
        home: Path | None = self.workspace / ".vnext"
        try:
            home.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            # A regular file named .vnext is the case that found this, and the
            # answer is the read-only one: the session is worth more than the
            # records, so it starts and says once where the records would have
            # gone.  Only this session's own defaults go; a caller that named
            # its own paths still gets them.
            print(
                f"vnext: {home} is not a folder this session can make "
                f"({getattr(error, 'strerror', None) or error}); this session keeps no run records.",
                file=sys.stderr,
            )
            home = None
        else:
            try:
                with (home / ".gitignore").open("x", encoding="utf-8") as ignore:
                    ignore.write("*\n")
            except OSError:
                # A convenience file, so every reason it cannot be written is
                # the same reason: the folder is the user's. An existing
                # .gitignore is theirs to keep, and a read-only .vnext is their
                # arrangement too.  Refusing to start the session over either
                # one was worse than leaving the records where the user put
                # them.
                pass
        self._event_log = self._resolve(
            event_log, None if home is None else home / "runs" / f"{self._session_id}.jsonl"
        )
        self._status_file = self._resolve(
            status_file, None if home is None else home / "status" / f"{self._session_id}.json"
        )
        self._outcome_log = self._resolve(
            outcome_log,
            None if home is None else home / "outcomes" / f"{self._session_id}.jsonl",
        )
        self._roster: dict[str, dict[str, Any]] = {}
        # The outcome file is a materialized one-row-per-agent view, not an
        # event stream.  Provider usage can keep growing after completion.
        self._outcome_rows: dict[str, dict[str, Any]] = {}
        self._latest_usage: dict[str, Mapping[str, Any]] = {}
        # Why a row has no tokens, when something knows.
        self._usage_unavailable: dict[str, str] = {}
        # Children report from the scheduler's threads, not the caller's.
        self._roster_lock = threading.Lock()

        self._code_freshness = CodeFreshness(_PACKAGE_FOLDER)
        active_runtime = read_active_runtime()
        # Shown on inspect when a side runtime runs or a newer release is out.
        # A start never waits on PyPI: it reads the day's cached answer, and
        # when there is none a daemon thread asks and fills the lines in later.
        newest = check_for_updates(cached_only=True)
        self._runtime_notice = runtime_notice_lines(
            entries, active=active_runtime, newest=newest, always=False,
        )
        self._update_check_thread: threading.Thread | None = None
        if os.environ.get(NO_UPDATE_CHECK_ENV) != "1" and (
            newest is None or (
                auto_update_needed(newest) and read_staged_runtime() is None
            )
        ):
            def refresh_runtime_notice() -> None:
                answer = newest if newest is not None else check_for_updates()
                # A server that closed meanwhile has no inspect left to show it.
                if answer and not getattr(self, "_closing", False):
                    self._runtime_notice = runtime_notice_lines(
                        entries, active=active_runtime, newest=answer, always=False,
                    )
                    auto_update(answer)
                    self._runtime_notice = runtime_notice_lines(
                        entries, active=active_runtime, newest=answer, always=False,
                    )

            self._update_check_thread = threading.Thread(
                target=refresh_runtime_notice, name="vnext-update-check", daemon=True,
            )
            self._update_check_thread.start()

        request = SessionStartRequest(
            session_id=self._session_id,
            workspace=str(self.workspace),
            primary_agent_id="primary",
            primary={"provider": "external", "model": client, "effort": "high"},
            main_preset={"instructions": instructions},
            catalog_config={"models": [{"provider": "external", "model": client}, *entries]},
            # The selected side runtime is held for this session; automatic
            # updates only stage a new pair for the next process start.
            config={"external_client": client, **session_runtime_config(active_runtime)},
        )
        registered = set(adapter_factories or {}) - _BUILT_IN_PROVIDERS
        # session_registry reads provider execution metadata while the runtime
        # is constructed.  Admit only names backed by an explicit factory, and
        # remove the temporary metadata immediately; the session keeps its own
        # model cards and _adapter() keeps its own factory mapping afterwards.
        with _PROVIDER_REGISTRY_LOCK:
            PROVIDERS.update({
                name: ("registered-adapter", "provider-owned") for name in registered
            })
            try:
                self.session = VNextRuntimeSession(
                    request, self._record, adapter_factories=adapter_factories
                )
            finally:
                for name in registered:
                    PROVIDERS.pop(name, None)
        self.session.start()
        self._tools = _external_tools(self.session.external_tools(), entries)
        self._publish_status()
        self._host, self._port = host, port
        self._token = token or secrets.token_urlsafe(32)
        self.relay: CodexMcpRelay | None = None

    def _resolve(self, given: Any, default: Path | None) -> Path | None:
        """A caller's path, this session's own, or nothing at all.

        ``default`` is None when the run folder could not be made: the session
        keeps no record of its own, and has already said so once.
        """

        if given is None or (given is _DEFAULT and default is None):
            return None
        chosen = default if given is _DEFAULT else Path(given)
        chosen = chosen.resolve()
        # A client that launches this process names a folder it does not create.
        # Failing there would take the whole workforce down over a directory.
        # A path refused here is kept: the writers try again on every change,
        # and print "written again" once it frees, as they do mid-session.
        try:
            chosen.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            # A read-only .vnext is the case that found this. The session is
            # worth more than the file, so the session starts and says so.
            self._write_failed(chosen, error)
            return chosen
        # A writable parent was the whole test before, and it passed for an
        # existing read-only file and for a directory standing where the log
        # belongs.  Ask about the path that actually gets opened.
        if chosen.exists():
            if chosen.is_dir():
                self._write_failed(chosen, IsADirectoryError(chosen.name))
            elif not os.access(chosen, os.W_OK):
                self._write_failed(chosen, PermissionError(chosen.name))
        elif not os.access(chosen.parent, os.W_OK):
            self._write_failed(chosen, PermissionError(chosen.parent.name))
        return chosen

    def _write_failed(self, path: Path, error: BaseException) -> None:
        """Say once per run of failures which record could not be written."""

        if path in self._failing_writes:
            return
        self._failing_writes.add(path)
        print(
            f"vnext: {path} could not be written ({getattr(error, 'strerror', None) or error}); the next change tries again.",
            file=sys.stderr,
        )

    def _make_room(self, path: Path) -> None:
        """Make a failing record's folder again before the next attempt.

        A folder that could not be made at startup, or was removed since, would
        otherwise refuse every later write for the rest of the session.
        """

        if path in self._failing_writes:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _write_succeeded(self, path: Path, *, gap: str = "") -> None:
        if path not in self._failing_writes:
            return
        self._failing_writes.discard(path)
        print(f"vnext: {path} is written again{gap}.", file=sys.stderr)

    @property
    def status_file(self) -> Path | None:
        """Where this session publishes its roster, for whatever draws it."""

        return self._status_file

    @property
    def outcome_log(self) -> Path | None:
        """Where this session appends one row per finished agent."""

        # Later cumulative usage rewrites that one row in place rather than
        # appending a second row that an external rating pass would count twice.

        return self._outcome_log

    @property
    def address(self) -> ServiceAddress:
        if self.relay is None:
            raise VNextMcpServiceError("the HTTP endpoint has not been started")
        return ServiceAddress(self.relay.endpoint, self.relay.bearer_token)

    def start(self) -> ServiceAddress:
        """Listen for a client the user configures by hand."""

        if self.relay is None:
            self.relay = CodexMcpRelay(
                tools=self._tools,
                dispatcher=self._dispatch,
                host=self._host,
                port=self._port,
                token=self._token,
                server_info=SERVER_INFO,
            )
        self.relay.start()
        return self.address

    def serve_stdio(self, stdin: Any = None, stdout: Any = None) -> int:
        """Answer a client that launched this process and owns its lifetime.

        The plugin path takes this one.  It removes the two steps a person would
        otherwise repeat every day -- start a server, paste a token -- and the
        client stops the workforce by closing the pipe, which is the same thing
        it already does for every other MCP server it runs.

        Nothing but protocol messages may reach stdout here; anything this
        service wants to say goes to stderr.

        One tool call at a time was the shape this started with, and an
        ``await_children`` from an external primary waits for real -- up to
        ``external_await_budget`` seconds.  Every other request queued behind
        it, so a client could not even `inspect` the children it was waiting
        on.  Each tool call now runs on its own thread.  Nothing here holds a
        scheduler lock across a wait: ``await_children_blocking`` reads the
        roster and sleeps in 0.1s slices, and the approval poll takes the
        scheduler lock and gives it back per pass, so a second call answers
        while the first waits.  The handshake and `tools/list` stay on this
        thread, in arrival order, because a client reads their answers before
        it sends anything else.
        """

        protocol = McpProtocol(
            tools=self._tools, dispatcher=self._dispatch, server_info=SERVER_INFO
        )
        source = stdin if stdin is not None else sys.stdin
        sink = stdout if stdout is not None else sys.stdout
        write_lock = threading.Lock()
        inflight: list[threading.Thread] = []

        def answer_for(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
            # The id belongs in the answer even when the answer is a failure.
            # These errors carried `id: null`, and a client matches a response
            # to the request it sent by that id alone: it could not tell which
            # of its open calls had failed, and an id-less error is not an
            # answer to any of them.  A notification has no id and keeps None.
            request_id = payload.get("id")
            try:
                return protocol.handle(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                return McpProtocol.error(request_id, -32700, str(exc))
            except Exception as exc:
                return McpProtocol.error(
                    request_id, -32603, f"{type(exc).__name__}: {exc}"
                )

        def respond(request_id: object, answer: Mapping[str, Any] | None) -> None:
            if answer is None:
                return
            try:
                line = json.dumps(answer, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                # A value that will not serialise used to raise here, on a
                # daemon thread with nobody to catch it, and the call stayed
                # open forever with the client waiting on it.  The id comes
                # from the request rather than the answer, because a broken
                # answer is exactly the one that may have lost it.
                line = json.dumps(
                    McpProtocol.error(
                        request_id,
                        -32603,
                        f"the answer could not be encoded: {type(exc).__name__}: {exc}",
                    ),
                    ensure_ascii=False,
                )
            # A lone surrogate in provider text made the write below raise on
            # a thread with nobody to catch it.  Its JSON escape reads back
            # as the same text.
            line = line.encode("utf-8", "backslashreplace").decode("utf-8")
            # Two calls can finish at once, and half a line of JSON mixed into
            # another is unreadable to the client.
            with write_lock:
                sink.write(line + chr(10))
                sink.flush()

        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("an MCP message must be an object")
            except (ValueError, json.JSONDecodeError) as exc:
                respond(None, McpProtocol.error(None, -32700, str(exc)))
                continue
            concurrent = (
                payload.get("method") == "tools/call" and payload.get("id") is not None
            )
            inflight = [thread for thread in inflight if thread.is_alive()]
            if not concurrent or len(inflight) >= _MAX_CONCURRENT_STDIO_CALLS:
                # A client with that many calls open is not waiting on one
                # answer any more, and unbounded threads are worse than a
                # queue.  Answer this one here.
                respond(payload.get("id"), answer_for(payload))
                continue
            thread = threading.Thread(
                target=lambda held=payload: respond(
                    held.get("id"), answer_for(held)
                ),
                name="vnext-mcp-call",
                daemon=True,
            )
            inflight.append(thread)
            thread.start()
        # The pipe is closed: the client that owns this process has gone.
        # Calls still open get a share of the close grace to finish in, and
        # whatever is left of it goes to the close that follows this return.
        eof_at = time.monotonic()
        drain_ends_at = eof_at + _EOF_DRAIN_SECONDS
        for thread in inflight:
            thread.join(timeout=max(0.0, drain_ends_at - time.monotonic()))
        abandoned = [thread for thread in inflight if thread.is_alive()]
        if abandoned:
            # Their answers have nowhere to go anyway, and they are daemon
            # threads, so none of them can hold the process open.
            print(
                f"vNext left {len(abandoned)} tool call(s) unanswered: the client "
                f"closed the pipe and they had not finished "
                f"{_EOF_DRAIN_SECONDS:g}s later",
                file=sys.stderr,
            )
        self._stdio_quit_budget = max(
            0.0, CLOSE_GRACE_SECONDS - (time.monotonic() - eof_at)
        )
        return 0

    def close(self, deadline: float | None = None, *, announce: bool = True) -> None:
        """Stop the endpoint and the workforce inside the proxy's kill grace.

        Two faults are answered here.  ``_closed`` was set before any of the
        work, so a close that raised was never retried: the second attempt saw
        the flag and returned, leaving the bridge and every worker behind it
        running.  And the work itself had no overall deadline -- a worker whose
        stop never answers held the quit for as long as the provider took,
        past the grace the restart proxy gives this process.

        So the steps run off this thread under one shared budget, the flag is
        set only once every step has finished, and an unfinished close says
        which step is still going and stays retryable.
        """

        if self._closed:
            return
        # The session's own flag goes first.  It is the one every path reads,
        # and the mirror below only buys the client a refusal before its call
        # reaches the scheduler: a client call that arrives between these two
        # lines is still refused, one layer further in.
        self._begin_session_closing()
        self._closing = True
        if deadline is not None:
            budget = max(0.0, float(deadline))
        else:
            # A quit that came through the pipe has already spent part of the
            # grace draining its open calls, and the whole quit has to fit
            # inside it.  The leftover is spent once; a close that comes later
            # is a close of its own and gets the whole grace.
            leftover, self._stdio_quit_budget = self._stdio_quit_budget, None
            budget = CLOSE_GRACE_SECONDS if leftover is None else leftover
        ends_at = time.monotonic() + budget
        if not self._close_lock.acquire(timeout=max(0.0, ends_at - time.monotonic())):
            # Another caller is already doing this work.  Same reasoning as an
            # overrun below: report it and stay retryable rather than raise.
            print(
                f"vNext close ran out of its {budget:g}s budget waiting for a "
                "close already under way",
                file=sys.stderr,
            )
            return
        try:
            self._close_under_lock(budget, ends_at, announce=announce)
        finally:
            self._close_lock.release()

    def _close_under_lock(self, budget: float, ends_at: float, *, announce: bool = True) -> None:
        if self._closed:
            # The caller that held the lock finished the job while this one
            # waited for it.
            return
        if self._close_steps is None:
            # The relay goes first, because a call that arrives while it is
            # still accepting has nowhere good to land once the workforce is
            # going down.  A step that overruns keeps its thread, so this is
            # the order the close prefers rather than one it can promise: a
            # hung endpoint leaves the relay accepting while the workforce
            # stops behind it.  What makes that window safe is the external
            # adapter, which refuses every call from the moment the session
            # closed and does so before it reads the tool name.  Measured over
            # real HTTP on 2026-09-30 with the endpoint step hung: a delegate
            # arriving then was refused and started no provider child, and
            # delegate, send_message, steer, cancel_agent, retry, replace,
            # interrupt_agent and inspect were all refused alike.  See
            # test_a_delegate_after_the_workforce_closed_starts_no_worker.
            self._close_steps = [
                _CloseStep("endpoint", self._close_relay),
                _CloseStep("workforce", self.session.close),
            ]
        unfinished = [
            step.label
            for step in self._close_steps
            if not step.run(ends_at - time.monotonic())
        ]
        failed = [step for step in self._close_steps if step.error is not None]
        # A close that ran out of time or failed has workers that may still be
        # running under this pid, so the roster keeps counting them.
        self._stopping = bool(unfinished or failed)
        self._publish_status(live=False, stopping=self._stopping)
        if unfinished:
            # Saying this on stderr rather than raising: being asked to stop and
            # running out of time is not a crash, and the restart proxy reads a
            # nonzero exit as one.
            print(
                "vNext close ran out of its "
                f"{budget:g}s budget with {', '.join(unfinished)} still stopping",
                file=sys.stderr,
            )
            return
        if failed:
            raise failed[0].error  # type: ignore[misc]
        self._closed = True
        # A clean close printed nothing at all, so a reader of the log could
        # not tell the client closing the pipe from the process dying.  Both
        # the signal path and the stdin-EOF path arrive here, and the overrun
        # and failure paths above return or raise before it, so this one line
        # means the workforce stopped on purpose.  stderr only: on stdio,
        # stdout carries the protocol.
        with self._roster_lock:
            recorded = len(self._roster)
        if announce:
            print(
                f"vnext: closed cleanly ({recorded} agent{'s' if recorded != 1 else ''} recorded)",
                file=sys.stderr,
                flush=True,
            )

    def _begin_session_closing(self) -> None:
        """Tell the workforce it is closing, if one was ever built.

        A close that cannot reach the session still has to run its steps, so a
        service that never started, or whose session is already gone, is not an
        error here.
        """

        session = getattr(self, "session", None)
        if session is None:
            return
        try:
            session.begin_closing()
        except Exception as exc:  # pragma: no cover - a close never fails here
            print(
                f"vnext: could not mark the session closing: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _close_relay(self) -> None:
        if self.relay is not None:
            self.relay.close()

    def __enter__(self) -> "VNextMcpService":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- the two seams -----------------------------------------------------

    def _dispatch(
        self, tool: str, arguments: Mapping[str, Any], _metadata: Mapping[str, Any]
    ) -> ToolCallResult | Mapping[str, Any]:
        """Run one manager tool for the client that owns this session's root.

        The scheduler already turns a rejected tool call into a failed result
        with a reason the caller can act on.  What it cannot answer for is a
        call that never reached it -- an unknown tool, or a tree not yet bound.
        Those become failed results here too, because a client reads an error
        result and corrects itself, while an HTTP 502 only tells it the server
        is broken.
        """

        if self._closing and tool in _TOOLS_THAT_START_A_TURN:
            # close() stops the endpoint first and a step that overruns keeps
            # its thread, so a hung endpoint leaves this session accepting
            # while the workforce stops behind it.  A call that only reads or
            # stops something is still worth answering in that window; one that
            # would start a provider turn builds a worker nothing is left to
            # stop, so it is refused here, under the caller's own request id.
            return {
                "success": False,
                "error": session_closing_message(tool),
                "error_code": SESSION_CLOSING_CODE,
            }
        try:
            result = self.session.external_tool_call(tool=tool, arguments=arguments)
        except Exception as exc:
            return {"success": False, "error": str(exc), "error_code": "external-dispatch"}
        if tool == "delegate" and isinstance(result, ToolCallResult) and result.success:
            command = [
                sys.executable, "-m", "vnext.vnext_wait",
                "--workspace", str(self.workspace.resolve()),
                "--agent", str(result.value["agent_id"]),
                "--session", self._session_id, "--deadline", "30m",
            ]
            wake_command = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
            result = ToolCallResult(True, {**dict(result.value), "wake_command": wake_command})
        notice = list(getattr(self, "_runtime_notice", None) or ())
        # The disk is read only for the two answers that carry the sentence.
        freshness = (
            getattr(self, "_code_freshness", None)
            if isinstance(result, ToolCallResult) else None
        )
        if (
            freshness is not None
            and tool == "inspect"
            and result.success
            and arguments.get("agent_id") == "self"
        ):
            stale = freshness.stale_notice()
            if stale:
                notice.append(stale)
        if (
            freshness is not None
            and tool in {"delegate", "replace"}
            and not result.success
            and isinstance(result.value, Mapping)
            and result.value.get("error_code") == "unknown-model"
        ):
            stale = freshness.stale_notice()
            if stale:
                error = str(result.value.get("error", ""))
                return ToolCallResult(
                    False, {**dict(result.value), "error": f"{error}; {stale}"}
                )
        if tool == "inspect" and notice and isinstance(result, ToolCallResult) and result.success:
            return ToolCallResult(True, {**dict(result.value), "runtime": notice})
        return result

    def _record(self, event: Any) -> None:
        """Keep a durable trace of the run, if one was asked for.

        Real use is the point of this service, so the events are worth keeping;
        but they are the session's own record and belong beside the workspace,
        not in a shared store.
        """

        payload = getattr(event, "payload", None)
        event_type = getattr(event, "type", None)
        agent_id = getattr(event, "agent_id", None)
        if (
            event_type == "agent.upsert"
            and isinstance(payload, Mapping)
            and payload.get("agent_id") not in (None, "primary")
        ):
            self._remember(payload)
            self._publish_status()
        elif (
            event_type == "usage.updated"
            and agent_id not in (None, "primary")
            and isinstance(payload, Mapping)
        ):
            # Unlike agent.upsert, usage.updated carries its identity on the
            # event envelope.  Its payload is the usage block itself.
            self._refresh_outcome_usage(str(agent_id), payload)
        elif (
            event_type == "usage.unavailable"
            and agent_id not in (None, "primary")
            and isinstance(payload, Mapping)
        ):
            self._note_usage_unavailable(str(agent_id), payload.get("reason"))
        if self._event_log is None:
            return
        with self._log_lock:
            line = _record_text(
                {
                    # Without this a merged read cannot tell two sessions
                    # apart, and the first log this server wrote had no way to say
                    # which of four runs any record belonged to.
                    "session_id": self._session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": getattr(event, "type", None),
                    "agent_id": getattr(event, "agent_id", None),
                    "turn_id": getattr(event, "turn_id", None),
                    "payload": getattr(event, "payload", {}),
                },
            )
            try:
                self._make_room(self._event_log)
                with self._event_log.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as error:
                # os.access said yes and the open said no: a read-only mount, a
                # full disk, a revoked permission mid-session.  Name it once,
                # let the workforce carry on, and try again on the next event.
                self._write_failed(self._event_log, error)
            else:
                self._write_succeeded(
                    self._event_log, gap="; the events in between are missing from it"
                )

    # -- the roster the user can see --------------------------------------

    def _remember(self, view: Mapping[str, Any]) -> None:
        """Keep the last seen shape of one delegated agent."""

        agent_id = str(view.get("agent_id"))
        status = view.get("status")
        with self._roster_lock:
            previous = self._roster.get(agent_id) or {}
            started = previous.get("started", time.time())
            terminal_change = status in _TERMINAL and (
                not previous.get("recorded") or previous.get("status") != status
            )
            row = {
                "agent_id": agent_id,
                "role": view.get("role"),
                "model": view.get("model"),
                "provider": view.get("provider"),
                "status": status,
                "effort": view.get("effort"),
                "parent_agent_id": view.get("parent_agent_id"),
                "started": started,
                "recorded": previous.get("recorded") or terminal_change,
            }
            # A blocked row that does not say why leaves the reader to guess.
            # Carried under the name the record uses, and only when the
            # runtime actually supplied a sentence.
            blocker = view.get("blocker")
            if isinstance(blocker, str) and blocker:
                row["blocker"] = blocker
            # The alias in "model" is what was asked for; these name the exact
            # model behind it and the one that answered, as the outcome row
            # does.  An id once known stays on the row.
            for key in ("model_exact", "model_exact_source", "model_ran", "model_mismatch", "model_note"):
                if key in view:
                    row[key] = view[key]
                elif key in previous:
                    row[key] = previous[key]
            self._roster[agent_id] = row
        if terminal_change:
            self._record_outcome(view, started)
        elif row["recorded"]:
            usage = view.get("usage")
            if isinstance(usage, Mapping):
                self._refresh_outcome_usage(agent_id, usage)

    def _record_outcome(self, view: Mapping[str, Any], started: float) -> None:
        """One row per finished agent: what it cost and whether it worked.

        The events hold this already, but spread over thousands of records and
        only in whichever log the run happened to write.  A row per agent is
        what a rating pass reads, and it is written once, when the agent is
        terminal and its own account of itself is final.

        Live providers can report more cumulative usage after that terminal
        account.  The row is therefore created once, then atomically refreshed
        when a later observation is larger.
        """

        if self._outcome_log is None:
            return
        usage = view.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        tokens = usage.get("tokens")
        tokens = tokens if isinstance(tokens, Mapping) else {}
        # Every provider fills the neutral block, and the Codex names below fill
        # for one of them.  Reading the neutral block first is what stopped a
        # Claude worker's row carrying no counts and no price at all.
        counts = usage.get("cost_tokens")
        counts = counts if isinstance(counts, Mapping) else {}
        result = view.get("result")
        result = result if isinstance(result, Mapping) else {}
        reported_cost = _reported_cost(usage)
        recorded_at = time.time()
        api_cost = _api_equivalent(view.get("model"), tokens, at=recorded_at)
        native_identity = view.get("native_identity")
        # Only the SDK reports a parent query whose total covers its Agent-tool
        # children; a Codex native child keeps a thread and a count of its own.
        native_child = (
            bool(view.get("parent_agent_id"))
            and view.get("harness") == _CLAUDE_SDK_HARNESS
            and isinstance(native_identity, Mapping)
            and native_identity.get("origin") == "native"
        )
        row = {
            "ts": recorded_at,
            "session_id": self._session_id,
            "workspace": str(self.workspace),
            "client": self._client,
            "agent_id": view.get("agent_id"),
            "parent_agent_id": view.get("parent_agent_id"),
            "role": view.get("role"),
            "provider": view.get("provider"),
            "model": view.get("model"),
            "effort": view.get("effort"),
            "status": view.get("status"),
            "duration_s": round(time.time() - started, 1),
            # The agent's own account of itself, named as a claim: a worker
            # saying it verified its work is not a verdict on that work.
            "claimed_verified": bool(result.get("verified")),
            "claimed_outcome": result.get("outcome"),
            "evidence_count": len(result.get("evidence") or ()),
            "tokens": {
                "total": counts.get("total_tokens", tokens.get("totalTokens")),
                "input": counts.get("input_tokens", _fresh_codex_input(tokens)),
                "cached_input": counts.get("cache_read_tokens", tokens.get("cachedInputTokens")),
                "output": counts.get("output_tokens", tokens.get("outputTokens")),
                # Codex alone separates the reasoning share out.
                "reasoning_output": tokens.get("reasoningOutputTokens"),
                # "thread" means the provider already added the agent's calls up.
                # "call" means vNext did the adding.
                "basis": counts.get("basis"),
            },
            # A provider that states a figure is believed: the SDK documents
            # total_cost_usd as a running session total, so the last one it sent
            # is the agent's bill.  Otherwise this is what the run would have
            # cost at published API rates, and None when the price table does
            # not carry the model, rather than a guessed number.
            "cost_usd": reported_cost or api_cost,
            # Say whether the figure came from the provider or our rate table;
            # an unlabeled estimate reads too much like a provider bill.
            "cost_source": (
                "provider" if reported_cost is not None
                else "api_rates" if api_cost is not None else None
            ),
        }
        # The alias in "model" is what was asked for.  These name the exact
        # model the provider resolved it to and the one that answered, so no
        # row is left holding only the alias.
        for key in ("model_exact", "model_exact_source", "model_ran", "model_mismatch",
                    "model_note", "model_reroutes", "model_exact_history"):
            if key in view:
                row[key] = view[key]
        if not row.get("model_exact"):
            # No identity reached this row: the worker never got as far as a
            # provider that reports one.  Say which, so the row is not bare.
            reason = (
                "the provider reported no model for this worker"
                if view.get("status") == "completed"
                else "the worker stopped before its provider named a model"
            )
            row["model_exact"] = f"{EXACT_MODEL_UNKNOWN}: {reason}"
            row.setdefault("model_exact_source", None)
        if native_child:
            # The provider's parent query already includes this Agent-tool
            # child's usage. Its own row is for the outcome, not another bill.
            row["tokens"] = {key: None for key in row["tokens"]}
            row["tokens"]["basis"] = "in_parent"
            row["cost_usd"] = None
            row["cost_source"] = "in_parent"
        with self._log_lock:
            previous = self._outcome_rows.get(str(view.get("agent_id")))
            if previous is not None:
                # A failed worker can be retried. Keep one row, but make its
                # status and claim describe the latest terminal attempt.
                for field in (
                    "ts", "status", "duration_s", "claimed_verified",
                    "claimed_outcome", "evidence_count",
                ):
                    previous[field] = row[field]
                self._apply_outcome_usage(previous, usage)
            else:
                previous = row
                latest = self._latest_usage.get(str(view.get("agent_id")))
                if latest is not None:
                    self._apply_outcome_usage(previous, latest)
                if latest is not None and _usage_is_real(latest):
                    self._usage_unavailable.pop(str(view.get("agent_id")), None)
                missing = self._usage_unavailable.get(str(view.get("agent_id")))
                if missing is not None:
                    previous["usage_unavailable"] = missing
                self._outcome_rows[str(view.get("agent_id"))] = previous
            self._rewrite_outcomes_locked()

    def _note_usage_unavailable(self, agent_id: str, reason: Any) -> None:
        """Say in the row why a worker's tokens and price are missing.

        A null price alone reads as free.  The quit that outran the provider's
        last usage message is a fact worth carrying, so whoever reads the row
        later knows the number was lost rather than zero.
        """

        text = str(reason or "the final usage message never arrived")
        with self._log_lock:
            self._usage_unavailable[agent_id] = text
            row = self._outcome_rows.get(agent_id)
            if row is None or row.get("usage_unavailable") == text:
                return
            row["usage_unavailable"] = text
            self._rewrite_outcomes_locked()

    def _refresh_outcome_usage(self, agent_id: str, usage: Mapping[str, Any]) -> None:
        """Retain cumulative usage and refresh a finished agent's row."""

        with self._log_lock:
            latest = self._latest_usage.get(agent_id)
            if latest is None or _usage_total(usage) > _usage_total(latest):
                self._latest_usage[agent_id] = usage
            # A message that was late rather than lost: the reason the row gave
            # for the missing figure is now false, so drop it. Keeping both made
            # one row say the cost was $0.25 and unknown at the same time.
            cleared = _usage_is_real(usage) and (
                self._usage_unavailable.pop(agent_id, None) is not None
            )
            row = self._outcome_rows.get(agent_id)
            if row is None:
                return
            if cleared:
                row.pop("usage_unavailable", None)
            if not self._apply_outcome_usage(row, usage) and not cleared:
                return
            self._rewrite_outcomes_locked()

    def _apply_outcome_usage(
        self, row: dict[str, Any], usage: Mapping[str, Any]
    ) -> bool:
        """Replace a newer cumulative token observation, or a newer price.

        Cost attribution can arrive after the counts it belongs to, and did:
        a Claude worker's last usage event repeated the token total and added
        the dollar figure, so a token-only test dropped the price.
        """

        if row.get("cost_source") == "in_parent":
            return False
        total = _usage_total(usage)
        previous = row.get("tokens")
        previous = previous if isinstance(previous, Mapping) else {}
        previous_total = previous.get("total")
        stale_tokens = total < 0 or (
            isinstance(previous_total, int)
            and not isinstance(previous_total, bool)
            and total <= previous_total
        )
        tokens = usage.get("tokens")
        tokens = tokens if isinstance(tokens, Mapping) else {}
        reported_cost = _reported_cost(usage)
        api_cost = _api_equivalent(row.get("model"), tokens, at=row.get("ts"))
        refreshed_cost = reported_cost or api_cost
        refreshed_source = (
            "provider" if reported_cost is not None
            else "api_rates" if api_cost is not None else None
        )
        new_price = refreshed_cost is not None and (
            refreshed_cost != row.get("cost_usd")
            or refreshed_source != row.get("cost_source")
        )
        if stale_tokens:
            # A late price with no counts, or with the same total, is the
            # provider's own figure. A smaller total is an older observation:
            # pricing its counts once cut a 1000-token row's cost to the
            # price of 100 tokens.
            same_counts = total < 0 or total == previous_total
            if not same_counts or reported_cost is None or not new_price:
                return False
            row["cost_usd"] = reported_cost
            row["cost_source"] = "provider"
            return True
        counts = usage.get("cost_tokens")
        counts = counts if isinstance(counts, Mapping) else {}
        row["tokens"] = {
            "total": counts.get("total_tokens", tokens.get("totalTokens")),
            "input": counts.get("input_tokens", _fresh_codex_input(tokens)),
            "cached_input": counts.get("cache_read_tokens", tokens.get("cachedInputTokens")),
            "output": counts.get("output_tokens", tokens.get("outputTokens")),
            "reasoning_output": tokens.get("reasoningOutputTokens"),
            "basis": counts.get("basis"),
        }
        if refreshed_cost is not None:
            row["cost_usd"] = refreshed_cost
            row["cost_source"] = refreshed_source
        return True

    def _rewrite_outcomes_locked(self) -> None:
        """Publish the in-memory outcome view without exposing a torn file."""

        if self._outcome_log is None:
            return
        lines = [
            _record_text(self._outcome_rows[agent_id])
            for agent_id in sorted(self._outcome_rows)
        ]
        temp = self._outcome_log.with_name(
            f".{self._outcome_log.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            self._make_room(self._outcome_log)
            temp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            os.replace(temp, self._outcome_log)
        except OSError as error:
            try:
                temp.unlink()
            except OSError:
                pass
            # Silence here left a rating pass reading a file that had stopped
            # being updated, with nothing anywhere saying why.  The whole file
            # is rewritten each time, so the next success repairs it.
            self._write_failed(self._outcome_log, error)
        else:
            self._write_succeeded(self._outcome_log)

    def _publish_status(self, *, live: bool = True, stopping: bool = False) -> None:
        """Rewrite the whole roster so something outside can show it.

        A whole small file rather than a tail of the event log: the log grows
        without bound and is appended across sessions, while whatever draws this
        redraws often and must be able to read it in one go.  Written to a
        sibling and renamed, so a reader never catches a half-written file.
        """

        target = self._status_file
        if target is None:
            return
        # Read inside the lock, not before it: a publish that had already read
        # a False here could sit at the lock while close() ran to completion,
        # then write live=True over the stopped status close() had just
        # written.  A gated probe reproduced exactly that.  The lock close()
        # takes for its own final write is this one, so asking the question
        # inside it makes stopped the last word.
        # Held across the write as well as the read for a second reason: two
        # agents can report at once, and both would otherwise rename over the
        # same temporary file.
        with self._roster_lock:
            live = live and not self._closing
            stopping = stopping or self._stopping
            agents = sorted(self._roster.values(), key=lambda agent: agent["agent_id"])
            working = [a for a in agents if a["status"] not in _TERMINAL]
            counted = live or stopping
            snapshot = {
                "session_id": self._session_id,
                "workspace": str(self.workspace),
                "client": self._client,
                "pid": os.getpid(),
                "updated": time.time(),
                **({"parent_session": self._parent_session} if self._parent_session else {}),
                "live": live,
                **({"stopping": True} if stopping and not live else {}),
                "working": len(working) if counted else 0,
                "blocked": sum(1 for a in working if a["status"] == "blocked") if counted else 0,
                "finished": len(agents) - len(working),
                "agents": agents,
            }
            temp = target.with_suffix(".json.tmp")
            try:
                self._make_room(target)
                temp.write_text(
                    _record_text(snapshot), encoding="utf-8"
                )
                os.replace(temp, target)
            except OSError as error:
                # A roster nobody can write is not a reason to stop the workforce.
                # On Windows a reader holding the file refuses the rename, so
                # the next change tries again.
                try:
                    temp.unlink()
                except OSError:
                    pass
                self._write_failed(target, error)
            else:
                self._write_succeeded(target)


def _record_text(value: Any) -> str:
    """JSON text that UTF-8 can always carry.

    Provider text can hold a lone surrogate, which ``ensure_ascii=False``
    passes through and the UTF-8 write then refuses.  Inside a JSON string the
    surrogate becomes its ``\\uXXXX`` escape, which reads back as the same text.
    """

    text = json.dumps(value, ensure_ascii=False, default=str)
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _usage_is_real(usage: Mapping[str, Any]) -> bool:
    """Say whether a usage message carries a figure worth believing.

    An empty block is the provider saying nothing, so it cannot contradict an
    earlier "the bill never arrived".  Counts or a price can.
    """

    if _usage_total(usage) > 0:
        return True
    return _reported_cost(usage) is not None


def _fresh_codex_input(tokens: Mapping[str, Any]) -> int | None:
    """Codex inputTokens includes cache reads; outcome rows store fresh input."""

    total = tokens.get("inputTokens")
    if not isinstance(total, int) or isinstance(total, bool):
        return None
    cached = tokens.get("cachedInputTokens")
    cached = cached if isinstance(cached, int) and not isinstance(cached, bool) else 0
    return max(0, total - cached)


def _usage_total(usage: Mapping[str, Any]) -> int:
    """Comparable cumulative total, or -1 when the event has no total."""

    counts = usage.get("cost_tokens")
    counts = counts if isinstance(counts, Mapping) else {}
    tokens = usage.get("tokens")
    tokens = tokens if isinstance(tokens, Mapping) else {}
    total = counts.get("total_tokens", tokens.get("totalTokens"))
    return total if isinstance(total, int) and not isinstance(total, bool) else -1


def _reported_cost(usage: Mapping[str, Any]) -> float | None:
    """The dollar figure the provider itself stated, when it stated one.

    The Claude Agent SDK reports a running session total on every result, and
    documents that only a conversation reset zeroes it, so the newest figure is
    the whole agent rather than its last turn.  Codex states no figure.
    """

    reported = usage.get("cost_usd")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        return None
    try:
        reported = float(reported)
    except OverflowError:
        # An integer past what a float holds raised here, mid-recording.
        return None
    if not math.isfinite(reported):
        # NaN was written as bare NaN, which strict JSON readers refuse.
        return None
    return reported or None


def _api_equivalent(
    model: Any, tokens: Mapping[str, Any], *, at: float | None = None
) -> float | None:
    """What this agent would have cost at published API rates, or nothing.

    A model the price table does not carry returns None rather than a number
    nobody can trace, because an outcome row is meant to be rated against.
    """

    try:
        from .pricing import api_equivalent
    except ImportError:  # pragma: no cover - pricing is part of the package
        return None
    total = tokens.get("inputTokens")
    if not isinstance(model, str) or not isinstance(total, int):
        return None
    cached = tokens.get("cachedInputTokens")
    output = tokens.get("outputTokens")
    # Absurd counts overflow the arithmetic, which either raises or gives
    # inf, and strict JSON has no word for inf.
    try:
        cost = api_equivalent(
            model, total, cached if isinstance(cached, int) else 0,
            output if isinstance(output, int) else 0,
            at=at,
        )
    except OverflowError:
        return None
    return cost if cost is None or math.isfinite(cost) else None


def _load_catalog(
    path: str | None,
    login: "_LoginProbe | None" = None,
    adapter_factories: Mapping[str, Any] | None = None,
) -> Sequence[Mapping[str, Any]]:
    """Read the roster, from the default or from a file, and guard both alike.

    A --catalog file used to be returned whole.  Every guard lives in
    ``_validate_catalog`` -- the pinned-CLI Codex model gate, the per-provider
    login and key filters, the non-empty check -- so a file could offer a model
    that can only fail long after a manager delegates to it, which is the
    outcome those filters exist to prevent.  A card for a provider registered
    through --provider still passes untouched: that factory owns its own
    credential check.
    """

    if path is None:
        # Models the side runtime lists and this catalog does not are offered
        # too, each marked "not tested by vNext".
        extra = untested_catalog_entries(read_active_runtime(), DEFAULT_CATALOG)
        return _validate_catalog((*DEFAULT_CATALOG, *extra), adapter_factories, login=login)
    target = Path(path)
    # The bare OSError read as "vnext cannot start: [Errno 2] No such file or
    # directory: 'opus'", which names neither the flag nor what it takes.
    if not target.is_file():
        raise VNextMcpServiceError(f"--catalog: no such file: {path}")
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise VNextMcpServiceError(
            f"--catalog: {path} could not be read: {exc.strerror or exc}"
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VNextMcpServiceError(f"--catalog: {path} holds invalid JSON: {exc}") from exc
    models = value.get("models") if isinstance(value, Mapping) else value
    if not isinstance(models, list) or not models:
        raise VNextMcpServiceError("a catalog file must hold a non-empty models list")
    return _validate_catalog(models, adapter_factories, login=login, catalog_path=path)


@dataclass(frozen=True)
class _StartupPlan:
    """What a start needs, once everything that can refuse one has had its say."""

    workspace: Path
    entries: list[dict[str, Any]]
    adapter_factories: dict[str, Any]
    login: "_LoginProbe"
    # The providers a --catalog file names, before any login filter; None for
    # the built-in catalog.  --check reads it so that a provider the file never
    # names is reported as such, ahead of a login it does not need.
    named_providers: frozenset[str] | None = None


def resolve_startup(
    *,
    workspace: str | Path = ".",
    catalog: str | None = None,
    registrations: Sequence[str] = (),
    login: "_LoginProbe | None" = None,
) -> _StartupPlan:
    """Every question "will this server start?" asks, asked in one place.

    ``--check`` read --workspace and --catalog alone, so a catalog card for a
    provider registered through --provider was reported as offered while a real
    start refused it: with no --provider the runtime knows no such provider, and
    with a --provider naming a class that answers for no effect shape the
    registration itself is refused.  Both paths now reach both refusals through
    this function, so the check and the start cannot drift apart again.
    """

    resolved = Path(workspace).resolve()
    if not resolved.is_dir():
        raise VNextMcpServiceError(_missing_workspace(resolved))
    factories = _load_provider_factories(list(registrations))
    probe = login or _LoginProbe()
    entries = [
        dict(entry)
        for entry in _load_catalog(catalog, login=probe, adapter_factories=factories)
    ]
    named = _catalog_file_providers(catalog) if catalog is not None else None
    return _StartupPlan(resolved, entries, factories, probe, named)


def _catalog_file_providers(path: str) -> frozenset[str]:
    """The providers a catalog file names; ``_load_catalog`` has already accepted it."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    models = value.get("models") if isinstance(value, Mapping) else value
    return frozenset(
        entry["provider"]
        for entry in models
        if isinstance(entry, Mapping) and isinstance(entry.get("provider"), str)
    )


def _chosen_path(given: str | None) -> Any:
    """An unset flag keeps the session default; "none" turns the file off.

    Passing the bare None through meant an omitted flag silenced the file, and
    dropping --event-log from the plugin manifest would then have thrown the run
    record away without saying so.
    """

    if given is None:
        return _DEFAULT
    return None if given == "none" else given


# SIGTERM is what the restart proxy and a service manager send.  SIGBREAK is
# the CTRL_BREAK the Windows launcher sends to the proxy's group when its client
# quits: Windows has no group SIGTERM, and Python's default for SIGBREAK exits
# without unwinding, so the close that writes the run record never ran.
_STOP_SIGNAL_NAMES = ("SIGTERM", "SIGBREAK")


def _install_stop_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for name in _STOP_SIGNAL_NAMES:
        number = getattr(signal, name, None)
        if number is not None:
            previous[number] = signal.signal(number, _unwind_on_sigterm)
    return previous


def _restore_stop_handlers(previous: dict[int, Any]) -> None:
    for number, handler in previous.items():
        signal.signal(number, handler)


def _unwind_on_sigterm(_signum: int, _frame: Any) -> None:
    # A second signal must not cut short the close the first one started.
    for name in _STOP_SIGNAL_NAMES:
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.SIG_IGN)
    # Exit 0: being asked to stop and stopping is not a crash, and the restart
    # proxy treats any other code as one.
    raise SystemExit(0)


def _address_refusal(host: str, token: str | None) -> str | None:
    """The argument refusals main makes before it starts, worded once for --check too."""

    if host not in LOOPBACK_HOSTS:
        return "--host must be a loopback host (127.0.0.1 or localhost)"
    if token is not None and len(token.strip()) < 16:
        return f"--token must be at least 16 characters (it was {len(token.strip())})"
    return None


def run_startup_check(
    argv: Sequence[str] | None = None,
    *,
    out: Any = None,
    err: Any = None,
) -> int:
    """Answer "will the server start, and with which models?" and exit.

    The startup failure a client shows is "Connection closed" with no cause,
    so this runs the same roster and login resolution the server runs and says
    the answer in words.  It opens no MCP transport.  To name the exact model
    behind each alias it starts the Claude CLI once without sending a prompt
    and asks one Codex app-server for ``model/list``, each bounded by 20 s; a
    probe that does not answer prints "exact model unknown".
    """

    stream = out if out is not None else sys.stdout
    errors = err if err is not None else sys.stderr
    # The check answers for the command a person actually runs, so it reads
    # that command with the server's own parser.  A parser of its own dropped
    # --provider once and a mistyped --catlog later, and each time passed a
    # command line the server then refused.
    parser = _server_parser("vnext-mcp --check")
    try:
        with contextlib.redirect_stderr(errors):
            parsed = parser.parse_args(list(argv or []))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    refusal = _address_refusal(parsed.host, parsed.token)
    if refusal is not None:
        # The server exits 2 on this refusal, so the check answers the same.
        print(f"vnext cannot start: {refusal}", file=errors)
        return 2
    try:
        plan = resolve_startup(
            workspace=parsed.workspace,
            catalog=parsed.catalog,
            registrations=parsed.provider,
        )
    except (VNextMcpServiceError, OSError, ValueError) as exc:
        print(f"vnext cannot start: {exc}", file=errors)
        return 1
    workspace, entries, probe = plan.workspace, plan.entries, plan.login

    print(f"checked from: {workspace} (Claude Code starts the server in your project folder)", file=stream)
    active = read_active_runtime()
    for line in runtime_notice_lines(entries, active=active, newest=check_for_updates(retry_failed=True)):
        print(line, file=stream)
    broken = _side_runtime_problem(active)
    if broken is not None:
        print(
            f"vnext cannot start workers: the side runtime is broken: {broken}. "
            "Run vnext-mcp --update-runtimes again, or vnext-mcp --update-runtimes --rollback to go back "
            "to the pinned runtime",
            file=errors,
        )
        return 1
    print(f"worker models offered: {len(entries)}", file=stream)
    exact = _exact_models_for_check(entries, workspace)
    unprobed = (
        "the model probe is switched off (VNEXT_CHECK_SKIP_MODEL_PROBE=1)"
        if os.environ.get("VNEXT_CHECK_SKIP_MODEL_PROBE") == "1"
        else "vNext has no model probe for this provider"
    )
    for entry in entries:
        model = entry.get("model") or entry.get("model_id") or "?"
        provider = entry.get("provider") or "unknown provider"
        named = exact.get((str(provider), str(model))) or f"{EXACT_MODEL_UNKNOWN}: {unprobed}"
        print(f"{model} ({provider}) -> {named}", file=stream)
    for line in startup_check_notes(entries, login=probe, named=plan.named_providers):
        print(line, file=stream)
    return 0


def _side_runtime_problem(active: Mapping[str, Any] | None) -> str | None:
    """Why the active side runtime cannot run a worker, or None when it can.

    The same checks a worker start makes: the Codex executable's recorded
    hash and version, and the side Python and Claude CLI.
    """

    if active is None:
        return None
    from .live_runtime import resolve_session_runtime
    from .vnext_runtimes import verify_claude_runtime

    try:
        resolve_session_runtime(codex_runtime_selection(active))
    except Exception as exc:  # noqa: BLE001 - reported to the person
        return f"its Codex executable cannot run ({str(exc) or type(exc).__name__})"
    try:
        verify_claude_runtime(claude_runtime_selection(active) or {})
    except Exception as exc:  # noqa: BLE001 - reported to the person
        # The caller names the fix once, so the reason's own hint is dropped.
        return (str(exc) or type(exc).__name__).split("; run vnext-mcp")[0]
    return None


_EXACT_PROBE_TIMEOUT_SECONDS = 20.0
_NO_MODEL_TABLE_PROVIDERS = frozenset({"zai", "commandcode"})
_NO_MODEL_TABLE_REASON = "until the first reply: this provider publishes no model table"


def _exact_models_for_check(
    entries: Sequence[Mapping[str, Any]], workspace: Path,
) -> dict[tuple[str, str], str]:
    """Name the exact model behind each listed alias, or say why it cannot.

    The test suite sets VNEXT_CHECK_SKIP_MODEL_PROBE=1 so a check run there
    never starts a provider process.
    """

    models = [(str(entry.get("provider")), str(entry.get("model") or entry.get("model_id") or "")) for entry in entries]
    answer: dict[tuple[str, str], str] = {}
    # These providers publish no model table: the id a worker's first reply
    # names is the first exact id anyone has, and its outcome row records it.
    for provider, model in models:
        if provider in _NO_MODEL_TABLE_PROVIDERS:
            answer[(provider, model)] = f"{EXACT_MODEL_UNKNOWN} {_NO_MODEL_TABLE_REASON}"
    if os.environ.get("VNEXT_CHECK_SKIP_MODEL_PROBE") == "1":
        return answer
    providers = {str(entry.get("provider")) for entry in entries}
    active = read_active_runtime()
    if "claude" in providers:
        if active is not None:
            # The side runtime's table, read when it was installed: the
            # server's own SDK is not the one its workers run on.
            rows, reason = list(active["claude"].get("models") or []), None
        else:
            rows, reason = _probe_claude_models(workspace)
        for provider, model in models:
            if provider != "claude":
                continue
            resolved, _source = resolve_claude_alias(model, rows)
            if not resolved and reason is None and active is None:
                # The CLI lists a row for an alias it was started with, which
                # settles an alias whose family has several models.
                own_rows, _own_reason = _probe_claude_models(workspace, model=model)
                resolved, _source = resolve_claude_alias(model, own_rows)
            if resolved:
                answer[(provider, model)] = resolved
                continue
            family = sorted({
                str(row.get("resolvedModel"))
                for row in rows
                if isinstance(row, Mapping)
                and str(row.get("resolvedModel") or "").startswith(f"claude-{model}-")
            })
            if reason is None and len(family) > 1:
                choices = ", ".join(family)
                answer[(provider, model)] = (
                    f"{EXACT_MODEL_UNKNOWN}: the CLI lists {choices}; "
                    "the worker's first reply names the one that runs"
                )
            else:
                answer[(provider, model)] = f"{EXACT_MODEL_UNKNOWN}: {reason or 'the CLI lists no row for it'}"
    if "codex" in providers:
        catalog, reason = _probe_codex_models(workspace)
        for provider, model in models:
            if provider != "codex":
                continue
            answer[(provider, model)] = catalog.get(model) or f"{EXACT_MODEL_UNKNOWN}: {reason or 'model/list does not list it'}"
    return answer


def _probe_claude_models(workspace: Path, model: str | None = None) -> tuple[list[Any], str | None]:
    """Start the Claude CLI without a prompt and read its model table."""

    try:
        import asyncio

        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    except Exception as exc:  # noqa: BLE001 - the check only reports
        return [], f"Claude SDK unavailable: {type(exc).__name__}"

    async def probe() -> list[Any]:
        options = ClaudeAgentOptions(cwd=str(workspace), **({"model": model} if model else {}))
        client = ClaudeSDKClient(options=options)
        try:
            await client.connect(None)
            info = await client.get_server_info()
        finally:
            await client.disconnect()
        rows = info.get("models") if isinstance(info, Mapping) else None
        return list(rows) if isinstance(rows, list) else []

    try:
        return asyncio.run(asyncio.wait_for(probe(), _EXACT_PROBE_TIMEOUT_SECONDS)), None
    except asyncio.TimeoutError:
        return [], f"the Claude CLI did not answer within {_EXACT_PROBE_TIMEOUT_SECONDS:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return [], f"Claude CLI probe failed: {type(exc).__name__}"


def _probe_codex_models(workspace: Path) -> tuple[dict[str, str], str | None]:
    """Ask one Codex app-server for model/list over the adapter's own connection."""

    adapter = None
    try:
        from .live_runtime import resolve_session_runtime
        from .vnext_app_server import VNextAppServerAdapter

        executable, _identity = resolve_session_runtime(codex_runtime_selection(read_active_runtime()))
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        adapter = VNextAppServerAdapter(
            codex_executable=executable, codex_home=home, workspace=workspace,
            client_name="vnext_check",
        )
        adapter.initialize(timeout=_EXACT_PROBE_TIMEOUT_SECONDS)
        return dict(getattr(adapter, "_model_catalog", {}) or {}), getattr(adapter, "_model_list_error", None)
    except Exception as exc:  # noqa: BLE001
        return {}, f"Codex probe failed: {str(exc) or type(exc).__name__}"
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass


def startup_check_notes(
    entries: Sequence[Mapping[str, Any]],
    *,
    login: "_LoginProbe | None" = None,
    named: frozenset[str] | None = None,
) -> list[str]:
    """One line per built-in provider whose models are missing from the roster.

    The probe from the roster resolution is reused, so the slow Claude check is
    paid for once rather than twice.  ``named`` is the set of providers a
    --catalog file names: a provider outside it is missing because of the file,
    and a login hint for it would send the user to fix the wrong thing.
    """

    probe = login or _LoginProbe()
    offered = {entry.get("provider") for entry in entries}
    notes: list[str] = []
    if named is not None:
        for provider in ("codex", "claude", "zai", "commandcode"):
            if provider not in offered and provider not in named:
                notes.append(f"{provider} models left out: the catalog names none")
                offered.add(provider)
    if "codex" not in offered:
        problem = _codex_login_problem()
        notes.append(f"codex models left out: {problem or 'the catalog names none'}")
    if "claude" not in offered and not _claude_sdk_installed():
        notes.append(f"claude models left out: {CLAUDE_SDK_MISSING}")
    elif "claude" not in offered:
        state = probe.claude_state()
        if state == CLAUDE_LOGIN_AVAILABLE:
            reason = "the catalog names none"
        elif state == CLAUDE_LOGIN_UNCHECKED:
            reason = (
                "'claude auth status' did not answer within "
                f"{_CLAUDE_LOGIN_TIMEOUT_SECONDS:.0f}s; you may well still be signed in"
            )
        else:
            reason = "no Claude Code login was found; run 'claude auth login'"
        notes.append(f"claude models left out: {reason}")
    if "zai" not in offered and not _claude_sdk_installed():
        notes.append(f"zai models left out: {CLAUDE_SDK_MISSING}")
    elif "zai" not in offered:
        if load_zai_provider() is None:
            problem = describe_provider_config_problem("zai")
            reason = problem or "no Z.ai provider key is configured"
        else:
            reason = "the catalog names none"
        notes.append(f"zai models left out: {reason}")
    if "commandcode" not in offered:
        if load_commandcode_provider() is None:
            problem = describe_provider_config_problem("commandcode")
            reason = problem or "no Command Code account key is configured"
        else:
            reason = "the catalog names none"
        notes.append(f"commandcode models left out: {reason}")
    return notes


def _server_parser(prog: str) -> argparse.ArgumentParser:
    """The server's arguments, shared with --check so both refuse the same ones."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Serve a vNext workforce to Claude Code over MCP.",
    )
    parser.add_argument("--workspace", default=".", help="workspace the workforce writes in")
    parser.add_argument("--host", default="127.0.0.1", help="loopback host only (127.0.0.1 or localhost)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--token",
        default=None,
        help="bearer token the client sends; generated and printed when omitted",
    )
    parser.add_argument("--client", default="claude-code", help="name of the client that owns the root")
    parser.add_argument("--catalog", default=None, help="JSON file holding the worker model catalog")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="NAME=module:attribute",
        help="register a zero-argument provider adapter factory (repeatable); the factory must carry a 'runtime_effect_reader', or a 'provider' naming the built-in provider whose event shape it speaks",
    )
    parser.add_argument(
        "--event-log",
        default=None,
        help="append session events to this JSONL file "
             "(default: <workspace>/.vnext/runs/<session>.jsonl; 'none' to write nothing)",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="roster of live workers, rewritten on every change "
             "(default: <workspace>/.vnext/status/<session>.json; 'none' to write nothing)",
    )
    parser.add_argument(
        "--outcome-log",
        default=None,
        help="one row per finished agent, for rating a run afterwards "
             "(default: <workspace>/.vnext/outcomes/<session>.jsonl; 'none' to write nothing)",
    )
    parser.add_argument("--name", default="vnext", help="MCP server name to suggest to the client")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="speak MCP on stdin/stdout instead of listening, for a client that launches this process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if is_runtime_command(raw):
        from .vnext_runtimes import main as update_main

        return update_main(raw)
    parser = _server_parser("python -m vnext.vnext_mcp_server")
    args = parser.parse_args(argv)
    refusal = _address_refusal(args.host, args.token)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    # All three startup reads (catalog load, validation, session config) must
    # see the same promoted record. The separate --check path never promotes.
    promote_staged(args.catalog)
    register_runtime_in_use(read_active_runtime())
    # The same resolution --check reports on, so the check cannot say yes to a
    # start this would refuse.  One probe for the whole startup: the catalog load
    # and the service both ask about the local logins, and asking twice paid the
    # claude timeout twice.
    #
    # A refusal is not a crash.  The traceback this used to print read like one,
    # where --check -- which asks the very same question -- prints a single line.
    # An unexpected exception keeps its traceback, because that one is a bug.
    try:
        plan = resolve_startup(
            workspace=args.workspace, catalog=args.catalog, registrations=args.provider
        )
    except VNextMcpServiceError as exc:
        print(f"vnext cannot start: {exc}", file=sys.stderr)
        return 1
    adapter_factories = plan.adapter_factories
    login_probe = plan.login

    service = VNextMcpService(
        workspace=plan.workspace,
        catalog=plan.entries,
        client=args.client,
        host=args.host,
        port=args.port,
        token=args.token,
        event_log=_chosen_path(args.event_log),
        status_file=_chosen_path(args.status_file),
        outcome_log=_chosen_path(args.outcome_log),
        adapter_factories=adapter_factories,
        login_probe=login_probe,
    )
    if args.stdio:
        # stdout belongs to the protocol from here on.
        print(f"vNext workforce ready for {args.client} in {service.workspace}", file=sys.stderr)
        # The restart proxy stops this process with SIGTERM.  Python's default
        # for it exits without unwinding, so the close below never ran and the
        # Claude bridge outlived the server with every worker still in a turn.
        previous = _install_stop_handlers()
        try:
            return service.serve_stdio()
        finally:
            try:
                service.close()
            finally:
                _restore_stop_handlers(previous)

    try:
        address = service.start()
    except OSError as exc:
        service.close(announce=False)
        print(
            f"vnext cannot start: could not bind {args.host}:{args.port} ({exc}); "
            "another vNext server may hold it; choose another --port, or --port 0 for any free one",
            file=sys.stderr,
        )
        return 1
    print(f"vNext workforce ready for {args.client} in {service.workspace}")
    print(f"  endpoint: {address.endpoint}")
    print(f"  token:    {address.token}")
    print(f"  register: {address.claude_mcp_add(args.name)}")
    # stdout is a pipe whenever a supervisor or a test starts this process, and
    # a pipe is block-buffered: these four lines sat in the buffer and the
    # reader waiting for `endpoint:` waited for as long as the server ran.
    sys.stdout.flush()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    # SIGTERM is how a service manager, a terminal closing and the restart
    # proxy all ask this process to stop.  Only Ctrl-C was handled here, so
    # Python's default for SIGTERM ended the process without unwinding: the
    # close below never ran, and the workforce outlived the endpoint with every
    # worker still in a turn.  This is the same handler the stdio path uses.
    previous = _install_stop_handlers()
    try:
        stop.wait()
    finally:
        try:
            service.close()
        finally:
            _restore_stop_handlers(previous)
    return 0


if __name__ == "__main__":  # pragma: no cover - console entrypoint
    raise SystemExit(main())
