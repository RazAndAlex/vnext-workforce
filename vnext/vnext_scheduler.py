"""Event-driven scheduler for one manager-owned vNext orchestration session.

The scheduler is the deep module above ``OrchestrationControlPlane``.  Managers
choose topology and completion through tools; this module owns only runtime
binding, concurrent turn waiting, wake dispatch, approval rendezvous, and
terminal detection.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .vnext_model_identity import PENDING as PENDING_EXACT_MODEL, cached_exact, read_identity
from .clock_format import format_duration, mid_turn_line, turn_start_line
from .vnext_approval_subject import approval_subject, visible_on_one_line
from .vnext_claude_effort import CLAUDE_ACCEPTED_EFFORTS
from .vnext_app_server import ToolCallContext, VNextAppServerAdapter, function_tool
from .vnext_managed_session import ManagedSessionError, ManagedTurn, VNextManagedSession
from .vnext_runtime_types import (
    CREDENTIAL_ADJACENT_STDERR_PROVIDERS,
    NativeChildBinding,
    NativeChildObservation,
    TurnHandle,
)
from .vnext_orchestration import (
    AgentRecord,
    AgentRole,
    AgentStatus,
    ProtocolError,
    finished_agent_outcome,
    REPLACEABLE_STATUSES,
    SESSION_CLOSING_CODE,
    SETTLED_STATUSES,
    TERMINAL_STATUSES,
    session_closing_message,
)
from .vnext_runtime_types import RuntimePosture, ToolCallResult
from .vnext_worker_tools import (
    ApprovalDecision,
    ApprovalOutcome,
    ApprovalRequest,
    RequiredToolEffect,
    RequiredToolEffectError,
    required_effect_contract_facts,
    validate_required_tool_effect,
)


DEFAULT_TURN_TIMEOUT = 1800

MANAGER_EFFORT = "high"
WORKER_EFFORT = "high"
# How many queued messages one prompt may carry.  The rest stay undelivered and
# are handed over on the next turn rather than being marked read and dropped.
MESSAGE_BATCH = 8
# A provider that hangs up part way through a turn is worth trying again.
# Measured on one swarm: 33 provider faults, every one of them a turn ended
# part way and none of them rate limited.  One worker died that way twice on
# the same packet and cost $1.10 with no files written.  Two retries, so three
# attempts in all for one agent across a whole run, and the wait grows between
# them because a provider that just cut a stream off is often still unwell.
PROVIDER_RETRY_LIMIT = 2
PROVIDER_RETRY_DELAYS = (2.0, 8.0)
# Window between a provider starting a turn and the scheduler publishing its handle.
APPROVAL_ROUTING_GRACE_SECONDS = 5.0
# Stands in for a child record the manager already holds.  Kept short on
# purpose: it is charged against the record it replaces.
_UNCHANGED_CHILD_NOTE = "unchanged; inspect for the full record"

# Effects that take nothing away and change nothing, so the reviewer sees what
# is being read and where it comes from.  "read" covers a file or directory the
# runtime wants to open, including one outside the workspace; "network" covers
# a search or a page fetch, which is a read that also sends its subject out.
# An effect absent here and from the two write classes is still declined.
_READ_ONLY_APPROVALS: dict[str, tuple[str, str, str]] = {
    "read": ("workspace-read", "workspace", "runtime requested read approval"),
    "network": ("network-read", "network", "runtime requested network approval"),
}

# Who said no, in the words the worker reads.  Only a manager's own decision --
# `resolve_approval`, or a standing grant that refuses -- may be reported as a
# manager decline.  Every other decline is this boundary's, and it used to
# borrow the manager's sentence: a worker was told its manager had refused a
# read that no manager was ever asked about.
_MANAGER_DECLINE_REASON = "manager declined this approval"
_BOUNDARY_DECLINE_REASONS: dict[str, str] = {
    "malformed": "vNext declined this approval: the approval envelope was malformed",
    "unrouted": "vNext declined this approval: it named no worker it could be routed to",
    "unnamed-effect": "vNext declined this approval: its boundary named no effect that can be reviewed",
    "no-manager": "vNext declined this approval: the worker has no manager to ask",
    "routing-failed": "vNext declined this approval: routing it to the manager failed",
    "timeout": "vNext declined this approval: the manager did not answer in time",
    "undecided": "vNext declined this approval: the rendezvous ended without a decision",
    "cancelled": "vNext declined this approval: the worker was cancelled while it waited",
    "unknown": "vNext declined this approval and recorded no reason",
}
# The resolver named on an approval the boundary decided by itself, so an
# operator reading runs/<session>.jsonl can tell it from a manager's refusal.
_BOUNDARY_RESOLVER = "vnext-approval-boundary"


class SchedulerError(RuntimeError):
    pass


class SchedulerCancelled(SchedulerError):
    pass


def _noop_agent(agent: AgentRecord) -> None:
    del agent


def _noop_event(agent: AgentRecord, state: str, code: str, message: str) -> None:
    del agent, state, code, message


def _noop_lifecycle(
    event_type: str,
    agent: AgentRecord,
    data: Mapping[str, Any],
) -> None:
    del event_type, agent, data


def _noop_native_approval(
    method: str,
    params: Mapping[str, Any],
    response: Mapping[str, Any],
) -> None:
    del method, params, response


@dataclass(frozen=True)
class SchedulerHooks:
    """Product-owned durability and presentation hooks at the scheduler seam."""

    record_agent: Callable[[AgentRecord], None] = _noop_agent
    emit: Callable[[AgentRecord, str, str, str], None] = _noop_event
    lifecycle: Callable[[str, AgentRecord, Mapping[str, Any]], None] = _noop_lifecycle
    native_approval: Callable[
        [str, Mapping[str, Any], Mapping[str, Any]], None
    ] = _noop_native_approval
    # Handed the adapter being replaced, returns its successor.  A tree
    # can span two providers, so one nullary factory cannot serve it.
    adapter_factory: Callable[[Any], Any] | None = None


@dataclass
class _ApprovalEnvelope:
    approval_id: str
    manager_id: str
    worker_id: str
    request: ApprovalRequest
    resolved: threading.Event
    outcome: ApprovalOutcome | None = None
    resolver: str | None = None


@dataclass(frozen=True)
class _TurnFinished:
    agent_id: str
    turn: ManagedTurn
    result: dict[str, Any] | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _UnsolicitedTurnEnded:
    """A turn the provider started by itself, between vNext turns, ended."""

    agent_id: str
    hook_failures: Mapping[str, Any]
    blocker: str | None = None


@dataclass(frozen=True)
class _BlockReported:
    """A worker called report_blocked; the loop decides what the reason does."""

    agent_id: str
    blocker: str


def _hook_failure_blocker(value: object) -> str | None:
    """The blocker for a turn whose last tool calls failed in a hook, or None.

    The bridge reports the trailing run of tool results that failed because a
    hook timed out or failed.  Such a turn ends normally, so without this the
    worker sat READY with nobody woken while its parent waited on it.
    """

    if not isinstance(value, Mapping):
        return None
    count = value.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return None
    first = str(value.get("first") or "").strip() or "a hook failed"
    return f"{count} tool calls failed in its last turn: {first}"


def _encoded_length(value: Mapping[str, Any]) -> int:
    """How many characters this record costs in a rendered prompt."""

    return len(json.dumps(value, ensure_ascii=False, default=str))


def _child_digest(child: Mapping[str, Any]) -> str:
    """Fingerprint a rendered child so an unchanged one can be recognised.

    Sorted keys make the encoding stable, so an identical record always yields
    an identical digest.  The digest is never shown to a model; it only decides
    whether the record is worth re-sending.
    """

    encoded = json.dumps(child, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# A URL is matched first so its scheme and host survive: read as a path,
# "https://api.openai.com/v1/responses" lost everything up to "responses".
# A path has to start a word, so "src/app", "./src/app" and "../lib" inside a
# sentence are left alone.  A Windows share path such as
# \\server\share\file.txt starts with two backslashes.  A path runs on
# across a space while the next word still holds a separator, so
# "/home/Jane Doe/token.txt" is one path and "/a/b to /c/d" is two.
_ABSOLUTE_PATH = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>|]*"
    r"|(?<![\w.~-])(?:~[\\/]|\\\\|[A-Za-z]:[\\/]|/)[^\s\"'<>|]{2,}"
    r"(?: [^\s\"'<>|\\/]+[\\/][^\s\"'<>|]*)*"
)


# A URL path part shown to a manager: an API version, or a word from the
# routes the providers vNext talks to.  Spelling alone proves nothing, since a
# lowercase token reads as a word, so any other part is hidden.
_URL_VERSION = re.compile(r"v\d+(?:\.\d+)?(?:beta|alpha)?")
_URL_ROUTE_WORDS = frozenset({
    "api", "backend-api", "chat", "codex", "coding", "completions", "count_tokens", "download",
    "embeddings", "health", "messages", "models", "oauth", "paas", "responses", "status", "token",
})


# The home-relative paths vNext itself writes into its hints, which tell a
# user where to correct a key.  Any other home-relative path is somebody's
# folder and is reduced like an absolute one.
_HOME_HINTS = frozenset({"~/.vnext/providers.json", "~/.vnext/providers.json"})


def _plain_url_path(path: str) -> str:
    kept: list[str] = []
    for part in path.split("/")[1:]:
        if part not in _URL_ROUTE_WORDS and not _URL_VERSION.fullmatch(part):
            kept.append("\u2026")
            break
        kept.append(part)
    return "".join("/" + part for part in kept)


def _shorten_path(match: "re.Match[str]") -> str:
    """Keep the file's name and drop the machine it lives on.

    A URL keeps its scheme and host.  Its login, query and fragment are where
    a signed link carries its credential, so they go, and an ellipsis says
    something was there.  A path can carry one too, as in /redeem/<token> or
    /download;token=<token>, so each path part is kept only while it is an API
    version ("v1") or a known provider route word ("chat", "completions"); the
    first other part ends the path with an ellipsis.
    """

    path = match.group(0)
    if path in _HOME_HINTS:
        return path
    scheme, separator, _rest = path.partition("://")
    if separator and scheme.lower() != "file":
        try:
            parts = urlsplit(path)
            host = parts.hostname or ""
            port = parts.port
        except ValueError:
            return f"{scheme}://\u2026"
        if ":" in host:
            host = f"[{host}]"
        if port is not None:
            host = f"{host}:{port}"
        shown = f"{parts.scheme}://{host}{_plain_url_path(parts.path)}"
        if parts.query:
            shown += "?\u2026"
        if parts.fragment:
            shown += "#\u2026"
        return shown
    if separator:
        # A file URL carries a query and fragment the way any URL does.
        try:
            parts = urlsplit(path)
        except ValueError:
            return f"{scheme}://\u2026"
        name = parts.path.rsplit("/", 1)[-1]
        if parts.query:
            name += "?\u2026"
        if parts.fragment:
            name += "#\u2026"
        return name
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    # A question mark in the last part is a link that lost its scheme.  The
    # "\\\\?\\" long-path prefix sits before the last separator, so it never
    # reaches here.
    if "?" in name:
        name = name.partition("?")[0] + "?\u2026"
    return name


# How many times the stop evidence is re-read inside one replacement barrier.
# A handful is enough to catch a process that dies shortly after its transport
# did, and few enough that the waiting costs one bounded call per slice.
_STOP_EVIDENCE_CHECKS = 4


def _wait_gave_up(exc: BaseException | None) -> bool:
    """Whether this turn failure is our own wait expiring.

    Two shapes reach here.  ``TimeoutError`` covers a plain wait and everything
    the standard library folds into it.  The Codex app-server raises its own
    error class for an idle budget that ran out, and the sentence it carries is
    the only thing that distinguishes it from a transport failure, so the text
    is read as well.  Reading the text can at worst send one extra best-effort
    interrupt for a turn that is already over, which costs a refused provider
    call and nothing else.
    """

    if exc is None:
        return False
    if isinstance(exc, TimeoutError):
        return True
    try:
        return "timed out" in str(exc).lower()
    except Exception:
        return False


# The HTTP status inside a Codex turn error's sentence, which is the only place
# the Codex CLI puts it on a turn that has given up retrying.
_HTTP_STATUS_SAID = re.compile(r"\bstatus (\d{3})\b")


def _safe_detail(exc: BaseException | None, *, limit: int = 300) -> str:
    """The one line about a failure that is safe to hand onward.

    Three things can go wrong turning an exception into a sentence, and all
    three have to be survivable here, because this runs on the scheduler's
    own thread while it is already handling a failure.  ``str`` on a broken
    exception raises, and an uncaught raise in the failure reporter kills the
    loop -- the observer destroying what it was built to observe.  A message
    can be unbounded.  And it can carry an absolute path or a secret from a
    provider, which this sentence then writes into a status record the
    manager reads.  So: never raise, bound the length, and reduce absolute
    paths to their last segment.  The unabridged text and traceback still go
    to the local run log, which is where an operator should look.
    """

    if exc is None:
        return ""
    try:
        detail = str(exc)
    except Exception:
        return f"<unprintable {type(exc).__name__}>"
    if not isinstance(detail, str):
        return f"<unprintable {type(exc).__name__}>"
    return _safe_text(detail, limit=limit)


# A credential written into an error sentence without a path around it:
# "Authorization: Bearer sk-...", "api_key=...", or a key whose own prefix
# gives it away.  The word that names it stays, so the line still says what
# was refused; the value goes.  A value is taken for a credential only when it
# holds a digit or a symbol, because "Bearer service unavailable" is a
# provider's sentence and lost its useful word when every word was taken.
_TOKEN_SHAPED = r"(?=[^\s,;\"'&]*[0-9._~+/=-])"
_BEARER_SECRET = re.compile(
    r"\b(Bearer|Basic) +" + _TOKEN_SHAPED + r"[A-Za-z0-9._~+/=-]{6,}", re.IGNORECASE
)
_NAMED_SECRET = re.compile(
    r"\b([A-Za-z0-9_-]*(?:api[_-]?key|token|secret|password|passwd|auth))(\s*[:=]\s*)"
    r"([\"']?)(?!…)" + _TOKEN_SHAPED + r"[^\s,;\"'&]+\3",
    re.IGNORECASE,
)
_PREFIXED_SECRET = re.compile(r"\b(?:sk-|gh[pousr]_|AKIA)[A-Za-z0-9_-]{4,}")


def _safe_text(text: str, *, limit: int = 300) -> str:
    """Provider text made fit for a manager: paths shortened, secrets cut.

    Every sentence a provider writes can reach a manager that runs on another
    vendor's model, so whatever the route -- an exception, a turn's reason,
    a refused stop -- it passes through here first.
    """

    text = _ABSOLUTE_PATH.sub(_shorten_path, text)
    text = _BEARER_SECRET.sub(lambda found: f"{found.group(1)} …", text)
    text = _NAMED_SECRET.sub(
        lambda found: f"{found.group(1)}{found.group(2)}{found.group(3)}…{found.group(3)}", text
    )
    text = _PREFIXED_SECRET.sub("…", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _objective_line(objective: str, *, limit: int = 120) -> str:
    """One line naming a child, not the packet that was sent to it.

    This receipt is returned for every child on every ``await_children`` poll,
    and a manager polls a long build many times.  Echoing the whole objective
    back cost roughly four thousand tokens per call for three children --
    paid over and over, to tell the manager the thing it wrote itself.  The
    first line is enough to tell the children apart, which is all the receipt
    is for; the full text is still one ``inspect`` away.
    """

    first = (objective or "").strip().splitlines()
    if not first:
        return ""
    line = first[0].strip()
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def strict_bool_argument(
    arguments: Mapping[str, Any], name: str, default: bool = False
) -> bool:
    """Read one boolean tool argument, and refuse a value that is not one.

    The MCP protocol layer checks that ``arguments`` is an object and nothing
    else: no field is matched against the declared schema.  So every boolean
    here used to be read with ``bool(arguments.get(name))``, and the string
    ``"false"`` is a non-empty string, which is True.  A client that sent
    ``start_despite_unconfirmed_stop: "false"`` was answered as though it had
    asked for the override by name, and got two writers in one workspace.
    The same reading made ``verified: "false"`` record a verified result.

    A real bool passes.  ``"true"`` and ``"false"`` pass in any case, because
    a client that stringifies its arguments means what the word says.  A
    missing or null argument takes the declared default.  Anything else is a
    bad request: it names the argument and the two words it takes, and the
    caller gets ``success`` false with nothing mutated.
    """

    value = arguments.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word == "true":
            return True
        if word == "false":
            return False
    raise ValueError(
        f"{name} takes true or false, and was {value!r}"
    )


def evidence_argument(arguments: Mapping[str, Any]) -> list[str]:
    """Read the evidence list of a completion, and refuse one that is not text.

    ``list`` on a string made one item per character, and on a number it
    raised a ``TypeError`` the handler does not translate.  A single string is
    taken as a one-item list, since a client that sends one line of evidence
    means that line.
    """

    value = arguments.get("evidence")
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"evidence takes a list of strings, and was {value!r}")


def criteria_argument(arguments: Mapping[str, Any]) -> dict[str, bool]:
    """Read complete_session's criteria: an object of names to true or false.

    A named criterion with no value is refused by name.  Read as the default,
    ``{"tests pass": null}`` recorded a criterion as judged when nobody had.
    """

    value = arguments.get("criteria")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"criteria takes an object of true or false values, and was {value!r}")
    for name, judged in value.items():
        if judged is None:
            raise ValueError(f"{name} takes true or false, and was None")
    return {str(name): strict_bool_argument(value, name) for name in value}


def required_agent_id(arguments: Mapping[str, Any]) -> str:
    """Read a required tool target before permission or handle lookup."""

    value = arguments.get("agent_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("agent_id is required")
    return value


def require_completion_fields(arguments: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    """The MCP transport does not enforce required fields in tool schemas."""

    missing = [field for field in fields if field not in arguments or arguments[field] is None]
    if missing:
        raise ValueError(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} required")


def _dynamic_result(success: bool, value: Mapping[str, Any]) -> ToolCallResult:
    """Answer one manager tool call in the provider-neutral shape.

    This used to build the Codex app-server ``contentItems`` payload directly,
    which quietly made the scheduler a Codex-specific component.  It now
    returns a neutral value and each adapter projects it, so a manager can be
    hosted on any runtime without the control plane knowing which one.
    """

    return ToolCallResult(success=bool(success), value=dict(value))


class VNextScheduler:
    """Run one dynamic manager tree through a small lifecycle interface."""

    def __init__(
        self,
        *,
        managed: VNextManagedSession,
        root: AgentRecord,
        cancellation: Any,
        hooks: SchedulerHooks | None = None,
        max_turns: int | None = None,
        # A reasoning model on a slow endpoint spends most of a long turn
        # thinking rather than stalling.  Measured on 2026-09-18: a
        # glm-5.3-flash worker finished a healthy turn at 598.9s and a
        # sibling on the same work hit the old 600s bound, so the bound was
        # cutting off turns that were still making progress.  The branch
        # approval wait below stays at its own 240s ceiling.
        turn_timeout: float = DEFAULT_TURN_TIMEOUT,
        # Injected so a test can pin the wall clock the worker reads.
        now: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if root.role is not AgentRole.ROOT_MANAGER:
            raise ValueError("scheduler root must be a Root Manager")
        if root.session_id != managed.session_id:
            raise ValueError("scheduler root is outside the managed session")
        self.managed = managed
        self.root = root
        self.cancellation = cancellation
        self.hooks = hooks or SchedulerHooks()
        self.max_turns = max_turns
        self.turn_timeout = turn_timeout
        self._now = now
        self._monotonic = monotonic
        # How long a blocking await_children holds an external client's call
        # before handing back a resumable "still running" answer.  The plugin
        # reaches this server through the reloaderoo restart proxy, whose MCP
        # client drops any call that takes over 60 seconds (-32001).  The stdio
        # loop answers one call at a time, so a longer wait also times out every
        # call queued behind it.  Kept under that ceiling with room to spare.
        self.external_await_budget = 50.0
        self._events: queue.Queue[object] = queue.Queue()
        self._lock = threading.RLock()
        self._active_turns_condition = threading.Condition(self._lock)
        self._active_turns: dict[str, ManagedTurn] = {}
        # When each live turn began, so a mid-turn steer can say how much of
        # the budget is already gone.  ManagedTurn carries no start time and
        # is a shared dataclass, so the clock is kept beside it instead.
        self._turn_started_at: dict[str, float] = {}
        # How many times the provider has been given another go at this agent,
        # and the monotonic deadline before which the next go must not start.
        self._provider_retries: dict[str, int] = {}
        self._retry_not_before: dict[str, float] = {}
        # Replacement agent id -> (predecessor id, monotonic deadline).  A
        # replacement whose predecessor's provider turn is still running shares
        # the workspace with a turn that is still writing to it, so the starter
        # holds it until the predecessor's turn leaves _active_turns.  The
        # deadline bounds that hold: the provider can refuse to stop.
        self._replacement_barriers: dict[str, tuple[str, float]] = {}
        # Agent id -> every turn of that agent vNext has asked to stop and
        # that has not yet said it ended, keyed by the turn's own identity.  A
        # turn our own wait gave up on is gone from _active_turns while the
        # model on the far side may still be running commands in the workspace,
        # and an interrupt is a request rather than a receipt.  The barrier
        # below reads this beside _active_turns, so "we stopped waiting" is
        # never read as "it stopped working".
        #
        # Per turn rather than per agent, because a retry reuses one record: an
        # agent can hold two unconfirmed turns at once, and the late receipt for
        # the older one says nothing about the newer one.  Keyed per agent it
        # erased the hold it was not watching, and the next plain retry started
        # a third writer with no override and nothing written down.
        self._stopping_turns: dict[str, dict[str, ManagedTurn]] = {}
        # Replacement id -> the predecessor whose stop it never confirmed.  The
        # replacement is BLOCKED, the manager has been told why, and a retry of
        # it is the manager taking that risk deliberately.
        self._unconfirmed_blocks: dict[str, str] = {}
        # Agents whose manager named ``start_despite_unconfirmed_stop``.  The
        # hold is re-read on every scan, so the decision has to outlive the one
        # block it cleared or the next scan would simply take it again.
        self._unconfirmed_start_overrides: set[str] = set()
        # How long a replacement waits for its predecessor's turn to end before
        # the predecessor is force-stopped.  Long enough for a provider to
        # honour the interrupt the replace already sent, short enough that a
        # manager waiting on the replacement is not parked for a whole turn.
        self.replacement_barrier_seconds = 30.0
        # A native-terminal owner must hold this before presenting a thread to
        # a provider UI. While held, scheduler mail is durable but never starts
        # a competing provider turn; observed native starts are adopted below.
        self._native_control_leases: set[str] = set()
        # An adopted native child cannot necessarily receive an arbitrary
        # provider-side prompt. Keep the adapter's precise delivery statement
        # beside its lease so a queued vNext message is never mislabeled as a
        # delivered native message.
        self._native_child_delivery_contracts: dict[str, dict[str, str]] = {}
        # Tool reads offer a batch; only its caller can acknowledge it. Native
        # callers additionally need their scoped provider turn below.
        # This host-local cursor does not promise provider restart recovery.
        self._native_messages_offered: dict[str, int] = {}
        self._waiters: set[threading.Thread] = set()
        self._bound: set[str] = set()
        self._pending_approvals: dict[str, _ApprovalEnvelope] = {}
        self._native_approval_records: list[dict[str, Any]] = []
        self._evidence_requests: dict[str, RequiredToolEffect] = {}
        self._children_delivered: dict[str, dict[str, str]] = {}
        self._drained_wakes: dict[str, list[dict[str, Any]]] = {}
        self._cancel_requested = threading.Event()
        self._idle_shutdown_requested = threading.Event()
        self._reconnect_requested = threading.Event()
        self._reconnect_interrupts: set[str] = set()
        self._turns_started = 0
        self._cancelled = False
        # Interrupting a conversational turn is deliberately not cancellation:
        # the target stays READY but does not auto-restart until a new message
        # or explicit resume arrives.
        self._interrupted_agents: set[str] = set()
        # Native children whose provider stop was refused.  The first cancel
        # already moved the record to CANCELLED, so the terminal guard below
        # skipped the provider on every later try and answered success while
        # the child's turn was still running.
        self._unstopped_native_children: set[str] = set()
        # Agents inside _start_turn, between asking the runtime for a turn and
        # reading the pause bit.  That read is the bit's only consumer, so a
        # replace landing in this window must leave the bit alone.
        self._starting_agents: set[str] = set()
        # A native turn may end normally without an explicit lifecycle tool.
        # Keep that live agent available for a follow-up instead of spinning a
        # new turn or treating its descriptive role as an instruction to die.
        self._awaiting_message_agents: set[str] = set()
        # A native Claude Code chat vNext did not start belongs to the user:
        # every turn prompt is written into its transcript as the user's own
        # line.  These agents receive the user's text verbatim and nothing
        # else; vNext guidance rides in the system prompt, peer mail in tool
        # responses, and a wake with no user text starts no turn.
        self._raw_prompt_agents: set[str] = set()
        # A provider-started turn's blocker that arrived while a vNext turn
        # ran.  That turn's end applies it, since its own result knows nothing
        # of a turn it did not run.
        self._deferred_blockers: dict[str, str] = {}
        self._terminal_release_pending: set[str] = set()
        self._idle_notified = False
        self.managed.adapter.native_approval_handler = self.review_approval

    def run(self, *, keep_alive: bool = False) -> dict[str, Any]:
        """Drive one workforce, optionally keeping its primary conversation alive.

        ``keep_alive`` is for the ordinary host conversation service.  A Root
        completion closes one objective, not its native thread or the session;
        the scheduler waits idle until another user message reopens the Root.
        Existing one-shot callers retain the historical default.
        """

        if self._cancel_requested.is_set() or bool(self.cancellation.requested):
            self._cancel_tree()
            raise SchedulerCancelled("vNext scheduler cancellation requested")
        self._bind_agent(self.root)
        self._start_ready_agents()
        while True:
            self._release_terminal_agents()
            if self._idle_shutdown_requested.is_set():
                return dict(self.root.result)
            root = self._session().agents[self.root.agent_id]
            if self._cancel_requested.is_set() or bool(self.cancellation.requested):
                # A persistent conversation commonly sits with a completed
                # objective while it waits for the next user turn.  That idle
                # state still has to observe cancellation; previously this
                # check lived only in the non-terminal branch and close() could
                # wait forever after a successful first objective.
                if keep_alive or root.status not in TERMINAL_STATUSES:
                    self._cancel_tree()
                    raise SchedulerCancelled("vNext scheduler cancellation requested")
            terminal_draining = root.status in TERMINAL_STATUSES
            if terminal_draining:
                # Completion tools run inside app-server handler threads. The
                # control plane can therefore become terminal a moment before
                # the enclosing runtime turn and handler callback have
                # returned. Drain every active turn before closing the adapter;
                # otherwise a successful concurrent tree can be recorded as a
                # handler-thread cleanup failure.
                with self._lock:
                    active_turns = bool(self._active_turns)
                if not active_turns and not keep_alive:
                    break
                if not active_turns and keep_alive:
                    if not self._idle_notified:
                        self._idle_notified = True
                        self.hooks.lifecycle("objective_completed", root, {"result": dict(root.result)})
                        self.hooks.lifecycle("conversation_idle", root, {})
            else:
                if self._cancel_requested.is_set() or bool(self.cancellation.requested):
                    self._cancel_tree()
                    raise SchedulerCancelled("vNext scheduler cancellation requested")
                self._quiesce_yielded_managers_for_reconnect()
                self._perform_reconnect_if_safe()
                self._start_ready_agents()
                # A conversational native turn may answer without invoking an
                # orchestration completion tool.  That leaves the primary
                # READY and deliberately awaiting its next message.  Surface
                # that state to the host once the whole workforce is quiescent,
                # while keeping the same native thread available for followup.
                if keep_alive and self._primary_waits_for_message() and not self._idle_notified:
                    self._idle_notified = True
                    self.hooks.lifecycle("conversation_idle", root, {})
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                if self._is_deadlocked():
                    raise SchedulerError("scheduler has no runnable agent or pending event")
                continue
            if isinstance(event, _TurnFinished):
                self._handle_turn_finished(event)
            elif isinstance(event, _UnsolicitedTurnEnded):
                self._handle_unsolicited_turn_ended(event)
            elif isinstance(event, _BlockReported):
                self._handle_block_reported(event)
            elif event == "cancel":
                continue
            elif event == "reconnect":
                self._quiesce_yielded_managers_for_reconnect()
                self._perform_reconnect_if_safe()
            elif event in {"approval", "control", "user"}:
                pass
            else:  # pragma: no cover - defensive fail-closed branch
                raise SchedulerError("scheduler received an unknown event")
        if root.status is AgentStatus.CANCELLED:
            raise SchedulerCancelled("vNext scheduler was cancelled")
        if root.status is not AgentStatus.COMPLETED:
            raise SchedulerError(f"Root Manager ended as {root.status.value}")
        return dict(root.result)

    def queue_terminal_release(self, agent_id: str) -> None:
        with self._lock:
            self._terminal_release_pending.add(agent_id)
        self._events.put("control")

    def _release_terminal_agents(self) -> None:
        with self._lock:
            pending = tuple(self._terminal_release_pending)
            self._terminal_release_pending.clear()
        for agent_id in pending:
            agent = self._session().agents.get(agent_id)
            if agent is None or agent.status not in SETTLED_STATUSES or not agent.thread_id:
                continue
            with self._lock:
                if agent_id in self._native_control_leases:
                    continue
            adapter = self.managed.adapter_for_agent(agent)
            release = getattr(adapter, "release_terminal_thread", None)
            if callable(release):
                try:
                    release(agent.thread_id, status=agent.status.value)
                except Exception as exc:
                    self._report_binding_failure(agent, adapter, exc, phase="release_agent")

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._events.put("cancel")

    def request_idle_shutdown(self) -> bool:
        """Release a quiescent controller without cancelling its saved graph."""
        with self._lock:
            if self._cancel_requested.is_set() or self.cancellation.requested:
                return False
            if self._active_turns or self._pending_approvals:
                return False
            # Adopted native children retain their scheduler-start exclusion
            # lease after completion. That historical lease is not active work.
            agents = self._session().agents
            if any(agent_id == self.root.agent_id or agent_id not in agents
                   or agents[agent_id].status not in TERMINAL_STATUSES
                   for agent_id in self._native_control_leases):
                return False
            if any(waiter.is_alive() for waiter in self._waiters):
                return False
            if self.root.status is not AgentStatus.COMPLETED and not (
                self.root.status is AgentStatus.READY
                and self.root.agent_id in self._awaiting_message_agents
            ):
                return False
            if any(agent.agent_id != self.root.agent_id and agent.status not in TERMINAL_STATUSES
                   for agent in self._session().agents.values()):
                return False
            self._idle_shutdown_requested.set()
            self._events.put("control")
            return True

    def cancel_agent(self, agent_id: str) -> dict[str, Any]:
        """Cancel one primary-owned subtree without cancelling other work."""

        return self._cancel_agent_for(self.root.agent_id, agent_id)

    def _cancel_agent_for(self, requester_id: str, agent_id: str) -> dict[str, Any]:
        """Interrupt and cancel one authorized subtree, including native turns."""

        session = self._session()
        target = session.agents.get(agent_id)
        if target is None:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        if requester_id not in session.agents:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        if requester_id != agent_id:
            cursor = target.parent_agent_id
            while cursor is not None and cursor != requester_id:
                parent = session.agents.get(cursor)
                cursor = parent.parent_agent_id if parent is not None else None
            if cursor != requester_id:
                raise ProtocolError("not-descendant", "cancellation target is outside the requester's subtree")
        # Control-plane cancellation changes every descendant to terminal.  It
        # cannot, by itself, stop an already running native turn, so interrupt
        # every active member before the status transition.  An unrelated
        # sibling remains untouched.
        targets = self._subtree_agent_ids(agent_id)
        # Cancelling work that already finished answered {"status": "cancelled"}
        # and wrote a cancel into the run record, while steer, send_message,
        # replace and complete_branch all refuse a terminal agent.  A manager
        # that had lost track of a finished child could not tell the two apart,
        # and the record showed a cancellation of an agent that completed.
        live_descendants = [
            target_id
            for target_id in targets
            if target_id != agent_id
            and session.agents[target_id].status not in TERMINAL_STATUSES
        ]
        # A stop the provider refused leaves the record terminal while the
        # native turn runs on, and the retry is a second cancel.  So "finished"
        # has to mean the provider agreed as well, not only that the record
        # says so.
        with self._lock:
            still_stopping = any(
                target_id in self._active_turns
                or target_id in self._stopping_turns
                or target_id in self._unstopped_native_children
                for target_id in targets
            )
        target_was_terminal = target.status in TERMINAL_STATUSES
        if target_was_terminal and not live_descendants and not still_stopping:
            return {
                "status": "already-finished",
                "agent_id": agent_id,
                "agent_status": target.status.value,
            }
        # A worker parked on an approval is a turn that can still write.  Its
        # request is declined here, before any interrupt, so a manager working
        # through its prompts afterwards cannot clear a cancelled worker's
        # write; resolve_approval then finds nothing pending.
        target_set = set(targets)
        with self._lock:
            for envelope in self._pending_approvals.values():
                if envelope.worker_id in target_set and envelope.outcome is None:
                    envelope.outcome = ApprovalOutcome(
                        ApprovalDecision.DECLINE, _BOUNDARY_DECLINE_REASONS["cancelled"]
                    )
                    envelope.resolver = _BOUNDARY_RESOLVER
                    envelope.resolved.set()
        uninterrupted: list[dict[str, str]] = []
        for target_id in targets:
            # The active turn and the held ones are asked separately, so a
            # refusal from one never skips the other and each is reported.
            for interrupt in (self._interrupt_turn, self._interrupt_unconfirmed_stop):
                try:
                    interrupt(target_id)
                except Exception as exc:
                    # Report the provider failure instead of raising: an exception
                    # leaves a manager unable to tell whether anything was cancelled,
                    # while this command must always reach a decision.  A timeout
                    # carries no message of its own, so the class name goes in the
                    # reason and the line still says something.
                    said = _safe_detail(exc)
                    reason = f"{type(exc).__name__}: {said}" if said else type(exc).__name__
                    uninterrupted.append({"agent_id": target_id, "reason": reason})
        previous = {
            target_id: session.agents[target_id].status
            for target_id in targets
            if target_id in session.agents
        }
        self.managed.control.cancel_agent(requester_id=requester_id, agent_id=agent_id)
        self._interrupted_agents.difference_update(targets)
        self._awaiting_message_agents.difference_update(targets)
        self.hooks.lifecycle("command_acknowledged", target, {"command": "cancel_agent", "status": "cancelled"})
        # The control plane cancelled the whole subtree, but only the target was
        # announced.  A descendant left at its last published state reads as
        # blocked or ready for good: a live session showed 10 such rows on 2026-09-23.
        for target_id, status in previous.items():
            agent = session.agents[target_id]
            if target_id != agent_id and status not in TERMINAL_STATUSES and agent.status in TERMINAL_STATUSES:
                self.hooks.lifecycle(
                    "agent_terminal",
                    agent,
                    {"role": agent.role.value, "status": agent.status.value},
                )
        self._events.put("control")
        with self._lock:
            stop_unconfirmed = any(
                target_id in self._stopping_turns for target_id in targets
            )
        result: dict[str, Any] = {"status": "cancelled", "agent_id": agent_id}
        if target_was_terminal and stop_unconfirmed:
            # The record went terminal when our wait on the turn gave up, and
            # the provider has still not said the turn ended.  Answering
            # "already-finished" told the manager the opposite of what it needed
            # and sent nothing; this names what is actually true and what the
            # call did about it.
            result["status"] = "stop-requested"
            result["agent_status"] = previous[agent_id].value
            result["detail"] = (
                f"{agent_id} is already {previous[agent_id].value}, and the provider turn "
                "behind it has not confirmed it stopped. A fresh stop was sent; it may "
                "still be changing shared files until the provider answers."
            )
        if target_was_terminal and live_descendants:
            # The target itself had already finished; what this call stopped is
            # the work still running underneath it, so the reply names it.
            result["agent_status"] = previous[agent_id].value
            result["cancelled_descendants"] = live_descendants
        if uninterrupted:
            result["uninterrupted"] = uninterrupted
        return result

    def interrupt_primary(self) -> dict[str, Any]:
        """Interrupt only the primary's active native turn."""

        return self.interrupt_agent(self.root.agent_id)

    def interrupt_agent(self, agent_id: str) -> dict[str, Any]:
        """Interrupt an active turn and pause it pending an explicit message."""

        agent = self._session().agents.get(agent_id)
        if agent is None:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        self._interrupted_agents.add(agent_id)
        with self._lock:
            active = self._active_turns.get(agent_id)
        if active is None:
            if agent.status in TERMINAL_STATUSES or agent.status is AgentStatus.BLOCKED:
                self._interrupted_agents.discard(agent_id)
                return {"status": "not-running", "agent_id": agent_id}
            # A command can arrive in the narrow window before the scheduler
            # binds/starts this READY agent.  Remember it so the turn does not
            # start behind the user's back; a later message explicitly resumes
            # it.  The same state also makes an idle service safe to command.
            self.hooks.lifecycle(
                "command_acknowledged",
                agent,
                {"command": "interrupt_agent", "status": "paused"},
            )
            self._events.put("control")
            return {"status": "paused", "agent_id": agent_id}
        try:
            self.managed._adapter_for(agent_id).interrupt(active.runtime)
        except Exception:
            self._interrupted_agents.discard(agent_id)
            raise
        self.hooks.lifecycle("command_acknowledged", agent, {"command": "interrupt_agent", "status": "interrupt-requested"})
        self._events.put("control")
        return {"status": "interrupt-requested", "agent_id": agent_id}

    def compact_agent(self, agent_id: str) -> dict[str, Any]:
        """Compact an idle agent's provider session.

        The native-control lease holds scheduler starts for the duration, so
        no managed turn can begin on the thread while the provider compacts.
        A lease another owner already holds (a native terminal) stays held:
        this call releases only a lease it acquired itself.
        """

        with self._lock:
            already_held = agent_id in self._native_control_leases
        self.acquire_native_control_lease(agent_id)
        try:
            agent = self._session().agents[agent_id]
            thread_id = self.managed._threads.get(agent_id)
            compact = getattr(self.managed._adapter_for(agent_id), "compact", None)
            if not callable(compact):
                raise ProtocolError("compact-unsupported", "this provider cannot compact a session")
            if not isinstance(thread_id, str):
                raise SchedulerError("compact requires a bound runtime thread")
            result = dict(compact(thread_id, timeout_seconds=self.turn_timeout))
        finally:
            if not already_held:
                self.release_native_control_lease(agent_id)
        self.hooks.lifecycle("command_acknowledged", agent, {"command": "compact_agent", "status": "compacted", **result})
        return {**result, "agent_id": agent_id}

    def message(self, message: str) -> dict[str, Any]:
        """Deliver a user message to the persistent primary conversation."""

        return self.steer(message)

    def acquire_native_control_lease(self, agent_id: str) -> None:
        """Hold scheduler starts while an external native UI owns this thread.

        The caller must release the lease after the native process exits. This is a
        control contract, not an attachment claim: no terminal surface is
        advertised until the attaching runtime can acquire it first.
        """

        agent = self._session().agents.get(agent_id)
        if agent is None:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        resumable_primary = (
            agent.agent_id == self.root.agent_id
            and agent.status is AgentStatus.COMPLETED
        )
        with self._lock:
            if agent_id in self._active_turns:
                raise ProtocolError(
                    "native-control-active-turn",
                    "native terminal attachment requires an idle agent turn",
                )
            if agent.status in TERMINAL_STATUSES and not resumable_primary:
                raise ProtocolError("terminal-agent", "cannot lease a terminal agent")
            if agent.status is not AgentStatus.READY and not resumable_primary:
                raise ProtocolError(
                    "native-control-not-runnable",
                    "native terminal attachment requires an idle ready agent",
                )
            already_held = agent_id in self._native_control_leases
            # Reserve before provider I/O.  Otherwise a scheduler loop on a
            # second thread can see READY after _bind_agent returns and start
            # a managed turn in the interval before this set is updated.
            self._native_control_leases.add(agent_id)
        try:
            if not self._bind_agent(agent):
                raise SchedulerError("could not bind a native-control lease thread")
        except BaseException:
            if not already_held:
                with self._lock:
                    self._native_control_leases.discard(agent_id)
            raise
        self._events.put("control")

    def release_native_control_lease(self, agent_id: str) -> None:
        with self._lock:
            self._native_control_leases.discard(agent_id)
        agent = self._session().agents.get(agent_id)
        # Settled, not terminal: a release queued while this lease was held is
        # skipped, and BLOCKED is one of the statuses that queues one. Requeue
        # the wider set or that release is lost for good.
        if agent is not None and agent.status in SETTLED_STATUSES:
            self.queue_terminal_release(agent_id)
        self._events.put("control")

    def adopt_native_child(
        self,
        observation: NativeChildObservation | Mapping[str, Any],
    ) -> NativeChildBinding:
        """Make one provider-created, attested child a vNext session agent.

        The child already exists in its provider.  Binding it here must not
        invoke the normal runtime-thread launcher; a native control lease then
        reserves its READY state for provider-observed turns.  The returned
        contract is the adapter's exact statement of how that child can accept
        contextual delivery and control.
        """

        child, binding, created = self.managed.adopt_native_child(
            observation,
            tool_handler_factory=lambda agent: (
                lambda tool, arguments, context, agent_id=agent.agent_id: self._manager_handler(
                    agent_id, tool, arguments, context
                )
            ),
        )
        if not created:
            return binding
        with self._lock:
            self._bound.add(child.agent_id)
            # The native provider owns this child thread.  The same lease that
            # protects an attached primary prevents _start_ready_agents from
            # starting a duplicate scheduler turn on it.
            self._native_control_leases.add(child.agent_id)
            self._native_child_delivery_contracts[child.agent_id] = dict(binding.delivery_contract)
            self._idle_notified = False
        self.hooks.record_agent(child)
        self.hooks.emit(child, "ready", "native_child_adopted", "Native provider child attached")
        self.hooks.lifecycle(
            "agent_spawned",
            child,
            {
                "role": child.role.value,
                "model_id": child.model_id,
                "parent_agent": child.parent_agent_id,
                "native": True,
                "delivery_contract": dict(binding.delivery_contract),
                "unsupported_reason": binding.unsupported_reason,
            },
        )
        self._events.put("control")
        return binding

    def adopt_native_turn(
        self,
        *,
        agent_id: str,
        provider: str,
        thread_id: str,
        turn_id: str,
        cursor: object | None,
    ) -> ManagedTurn:
        """Make a verified externally started native turn scheduler-visible."""

        agent = self._session().agents.get(agent_id)
        if agent is None:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        with self._lock:
            active = self._active_turns.get(agent_id)
            existing_control_turn = self.managed.control_turn_for_native(
                agent_id=agent_id, provider=provider, native_turn_id=turn_id
            )
            scheduler_starting = self.managed.native_turn_starting(
                agent_id=agent_id, provider=provider, thread_id=thread_id
            )
            if active is not None:
                if active.control_turn_id == existing_control_turn:
                    return active
                raise SchedulerError("agent already has a different active turn")
            if (
                existing_control_turn is None
                and not scheduler_starting
                and agent_id not in self._native_control_leases
            ):
                raise ProtocolError(
                    "native-control-not-leased",
                    "external native turns require a held control lease",
                )
            if (
                agent.status is AgentStatus.AWAITING_WORKERS
                and existing_control_turn is None
            ):
                self._resume_awaiting_for_fresh_native_turn(
                    agent=agent,
                    provider=provider,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    scheduler_starting=scheduler_starting,
                )
                agent = self._session().agents[agent_id]
            if (
                existing_control_turn is None
                and not scheduler_starting
                and agent_id == self.root.agent_id
                and agent.status is AgentStatus.COMPLETED
            ):
                # The terminal's next native user message begins a fresh
                # objective on the same primary thread. There is no separate
                # scheduler prompt to queue or duplicate in the inbox.
                self.managed.control.resume_primary_for_native_turn(agent_id)
                agent = self._session().agents[agent_id]
            turn, adopted = self.managed.adopt_native_turn(
                agent_id=agent_id,
                provider=provider,
                runtime=TurnHandle(thread_id=thread_id, turn_id=turn_id, cursor=cursor),
            )
            # A scheduler-started turn can have registered its native id in
            # the narrow interval before it publishes _active_turns. Its own
            # starter will install the waiter, so this event is a harmless
            # duplicate rather than a second external adoption.
            if not adopted:
                if scheduler_starting:
                    self._active_turns[agent_id] = turn
                    self._turn_started_at[agent_id] = self._monotonic()
                    self._active_turns_condition.notify_all()
                return turn
            self._active_turns[agent_id] = turn
            self._turn_started_at[agent_id] = self._monotonic()
            self._active_turns_condition.notify_all()
            self._awaiting_message_agents.discard(agent_id)
            self._idle_notified = False
        self.hooks.emit(agent, "running", "native_turn_adopted", "Native provider turn adopted")
        self.hooks.lifecycle(
            "turn_started",
            agent,
            {"role": agent.role.value, "phase": turn.phase, "turn_id": turn.control_turn_id},
        )
        self._watch_turn(turn)
        return turn

    def _resume_awaiting_for_fresh_native_turn(
        self,
        *,
        agent: AgentRecord,
        provider: str,
        thread_id: str,
        turn_id: str,
        scheduler_starting: bool,
    ) -> None:
        """Reconcile only a fresh, independently observed native follow-up.

        A provider-owned native manager may start its next turn after its
        vNext ``await_children`` call parked the control record. The incoming
        provider turn is sufficient only when it is fresh for this exact agent,
        provider, and already leased thread. A known native turn is handled by
        ``managed.adopt_native_turn`` as an idempotent replay before this
        helper is reached. A scheduler-start overlap, foreign thread, or a
        non-native owner remains a refusal.
        """

        if scheduler_starting:
            raise SchedulerError("awaiting native agent conflicts with a scheduler native start")
        if agent.agent_id not in self._native_control_leases:
            raise ProtocolError(
                "native-control-not-leased",
                "awaiting native turn requires a held control lease",
            )
        expected_thread = self.managed._threads.get(agent.agent_id)
        if expected_thread != thread_id:
            raise ProtocolError(
                "native-control-thread-mismatch",
                "native turn does not belong to the awaiting leased thread",
            )
        if provider != self._native_provider_for_agent(agent.agent_id):
            raise ManagedSessionError("native turn provider does not match the awaiting agent selection")
        if not isinstance(turn_id, str) or not turn_id:
            raise ProtocolError("invalid-turn", "native turn handle is required")
        self.managed.control.resume_awaiting_agent_for_native_turn(
            agent.agent_id,
            thread_id=thread_id,
        )

    def message_user(self, agent_id: str, message: str) -> dict[str, Any]:
        """Deliver a user-authored prompt to one live agent and wake its turn."""

        text = message.strip()
        if not text:
            raise ValueError("User message is required")
        target = self._session().agents.get(agent_id)
        if target is None:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        with self._lock:
            active = self._active_turns.get(agent_id)
        native_delivery = self._native_message_delivery(agent_id)
        if active is None and native_delivery == "unavailable":
            raise ProtocolError(
                "native-message-unavailable",
                "the provider did not attest contextual delivery to this native child",
            )
        # A raw-prompt chat keeps the user's text byte for byte.
        self.managed.control.send_user_message(
            agent_id, message if agent_id in self._raw_prompt_agents else text
        )
        self._interrupted_agents.discard(agent_id)
        self._awaiting_message_agents.discard(agent_id)
        self._idle_notified = False
        delivery = "queued-for-native-tool-read" if native_delivery == "available" else "queued-for-next-turn"
        if active is not None:
            try:
                self.managed._adapter_for(agent_id).steer(active.runtime, text)
                delivery = "delivered-into-active-turn"
            except Exception:
                delivery = "queued-for-next-turn"
        self._events.put("user")
        result = {"status": "queued", "agent_id": agent_id, "delivery": delivery}
        self.hooks.lifecycle("command_acknowledged", target, {"command": "message_user", **result})
        return result

    def steer(self, message: str) -> dict[str, Any]:
        text = message.strip()
        if not text:
            raise ValueError("Steering message is required")
        # message_user strips again unless the root is a raw-prompt chat.
        result = self.message_user(self.root.agent_id, message)
        return {**result, "status": "steered", "target": AgentRole.ROOT_MANAGER.value}

    def send_message(self, *, sender_id: str, target_id: str, text: str) -> dict[str, Any]:
        """Deliver a durable same-session peer message and resume its target."""

        body = text.strip()
        if not body:
            raise ValueError("Message is required")
        if self._native_message_delivery(target_id) == "unavailable":
            raise ProtocolError(
                "native-message-unavailable",
                "the provider did not attest contextual delivery to this native child",
            )
        self.managed.control.message_agent(sender_id, target_id, body, kind="message")
        self._interrupted_agents.discard(target_id)
        self._awaiting_message_agents.discard(target_id)
        self._idle_notified = False
        target = self._session().agents[target_id]
        # Keep both the exact contextual mail and its size in the durable run
        # log so a later review can reconstruct what the target received.
        self.hooks.lifecycle("command_acknowledged", target, {
            "command": "send_message",
            "sender_id": sender_id,
            "message": body,
            "message_characters": len(body),
        })
        self._events.put("control")
        delivery = self._native_message_delivery(target_id)
        return {
            "status": "queued",
            "sender_id": sender_id,
            "target_id": target_id,
            "delivery": "queued-for-native-tool-read" if delivery == "available" else "queued-for-next-turn",
        }

    def message_agent(self, *, sender_id: str, target_id: str, text: str) -> dict[str, Any]:
        """Compatibility-friendly public name for a service command."""

        return self.send_message(sender_id=sender_id, target_id=target_id, text=text)

    def snapshot(self, *, deep: bool = False) -> dict[str, Any]:
        """Return a lock-consistent control snapshot for the session service."""

        session = self._session()
        with session.lock:
            return {
                "session_id": session.session_id,
                "root_agent_id": session.root_agent_id,
                "agents": [
                    self.managed.control.inspect_agent(self.root.agent_id, agent_id, deep=deep)
                    for agent_id in session.agents
                ],
                "interrupted_agent_ids": sorted(self._interrupted_agents),
            }

    def request_reconnect(self) -> dict[str, Any]:
        if self.hooks.adapter_factory is None:
            raise SchedulerError("no reconnect adapter factory is configured")
        self._reconnect_requested.set()
        self._events.put("reconnect")
        return {"status": "reconnect-queued"}

    @staticmethod
    def _delegate_role(value: Any) -> AgentRole:
        """Read the role word, and name the words this tool takes.

        A delegate call with no role used to be answered ``'' is not a valid
        AgentRole``: a Python class name and an empty-string artifact, with no
        list of the roles that would have worked.  The caller reads this reply
        and has to act on it, so it says what is missing and what to put there.
        """

        word = value.strip() if isinstance(value, str) else ""
        try:
            return AgentRole(word)
        except ValueError:
            roles = ", ".join(role.value for role in AgentRole)
            if not word:
                raise ValueError(f"role is required: one of {roles}") from None
            raise ValueError(
                f"role must be one of {roles}, and was {word!r}"
            ) from None

    @staticmethod
    def _approval_decision(decision: Any) -> ApprovalDecision:
        """Read the decision word, and refuse one this tool does not define.

        The tool schema declares ``enum: ["accept", "decline"]`` and the
        external MCP surface does not enforce it, so anything else used to fall
        through to DECLINE.  A manager that says "approve" then watched its
        worker's command be refused and had no way to tell that from a real
        refusal.  Silence is the wrong answer to a word this tool does not
        know: say which words it takes and leave the approval pending, so the
        caller can answer it again.
        """

        word = str(decision).strip().lower()
        if word == "accept":
            return ApprovalDecision.ACCEPT
        if word == "decline":
            return ApprovalDecision.DECLINE
        raise ValueError(
            f"decision must be 'accept' or 'decline', and was {str(decision)!r}. "
            "The approval is still pending; call resolve_approval again."
        )

    def resolve_approval(
        self,
        approval_id: str,
        decision: str,
        rationale: str = "",
        *,
        resolver: str = "user",
    ) -> dict[str, Any]:
        selected = self._approval_decision(decision)
        self._resolve_pending_approval(
            approval_id=str(approval_id),
            decision=selected,
            rationale=str(rationale),
            resolver=resolver,
        )
        return {
            "approval_id": str(approval_id),
            "decision": selected.value,
            "resolver": resolver,
        }

    def evidence_requests(self) -> list[dict[str, Any]]:
        return [
            {
                "agent_id": agent_id,
                "contract": required_effect_contract_facts(
                    contract,
                    workspace=self.managed.workspace_for(agent_id),
                ),
                "status": "recorded-unattributed",
            }
            for agent_id, contract in self._evidence_requests.items()
        ]

    def native_approval_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._native_approval_records]

    # A provider-shaped payload is not trusted for length any more than for
    # shape, so the reviewer's own copy is cut here as well as at the bridge.
    _APPROVAL_SUBJECT_CHARS = 400
    _APPROVAL_SUBJECT_ARGV = 16

    @classmethod
    def _approval_subject(
        cls, params: Mapping[str, Any]
    ) -> tuple[str, tuple[str, ...]]:
        """Name the tool and the file or command one approval is about.

        Verified live: 34 approvals resolved in one day, every one of them
        showing an empty command, so a manager approving a file edit was never
        told which file.  A boundary that does not publish these fields -- an
        older bridge, or Codex and Command Code, which do not -- still gets
        exactly the approval it got before: the neutral tool name and no
        command.  A malformed field is read as an absent one and never raises.
        """

        event = params.get("event")
        if not isinstance(event, Mapping):
            return "runtime-effect", ()
        raw_tool = event.get("tool")
        tool = (
            raw_tool[: cls._APPROVAL_SUBJECT_CHARS]
            if isinstance(raw_tool, str) and raw_tool.strip()
            else "runtime-effect"
        )
        # The tool says which of its fields is the subject.  A boundary that
        # publishes the raw input rather than a resolved "path" therefore gets
        # the right field even when the input carries several candidates: a
        # WebFetch with a real url and a decoy file_path used to record the
        # decoy, because the scan met file_path first.
        named = approval_subject(tool, event, cls._APPROVAL_SUBJECT_CHARS)
        if named is not None:
            return tool, (named,)
        raw_path = event.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            return tool, (raw_path[: cls._APPROVAL_SUBJECT_CHARS],)
        raw_command = event.get("command")
        if isinstance(raw_command, (list, tuple)):
            argv = tuple(
                part[: cls._APPROVAL_SUBJECT_CHARS]
                for part in list(raw_command)[: cls._APPROVAL_SUBJECT_ARGV]
                if isinstance(part, str)
            )
            if argv:
                return tool, argv
        return tool, ()

    @classmethod
    def _approval_subject_value(cls, command: tuple[str, ...]) -> str | None:
        """What one approval is about, exactly as the boundary sent it.

        A read, a fetch and a search each carry one subject -- the file, the
        URL, the query -- and a command carries its argv.  Nothing is returned
        when the boundary named nothing, so the key stays absent rather than
        empty: an operator must not read a silent provider as a read of the
        root directory.
        """

        parts = [part for part in command if part.strip()]
        if not parts:
            return None
        return " ".join(parts)[: cls._APPROVAL_SUBJECT_CHARS]

    @classmethod
    def _approval_subject_line(cls, command: tuple[str, ...]) -> str | None:
        """Render that subject so it stays one line wherever it is shown."""

        value = cls._approval_subject_value(command)
        return None if value is None else visible_on_one_line(value)

    @classmethod
    def _approval_subject_fields(cls, command: tuple[str, ...]) -> dict[str, str]:
        """The subject as an audit record carries it: raw, plus a safe line.

        A provider-shaped subject is whatever the provider sent, and one with a
        newline in it turned a single audit line into two, the second reading
        like a record of its own.  So "subject" keeps the raw value -- an
        operator comparing it against a file name needs the bytes -- and
        "subject_line" appears beside it only when rendering it changed
        something, carrying the version that is safe to print.
        """

        value = cls._approval_subject_value(command)
        if value is None:
            return {}
        line = visible_on_one_line(value)
        return {"subject": value} if line == value else {
            "subject": value,
            "subject_line": line,
        }

    def review_approval(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Route one provider-neutral approval envelope through the rendezvous."""

        approval_reference = params.get("approval_reference")
        provider = params.get("provider")
        if (
            method != "approval/request"
            or not isinstance(approval_reference, str)
            or not approval_reference
            or approval_reference.strip() != approval_reference
            or not isinstance(provider, str)
            or not provider
            or provider.strip() != provider
        ):
            return self._finish_approval(
                method,
                self._unattested_approval(params),
                {
                    "decision": "decline",
                    "reason": _BOUNDARY_DECLINE_REASONS["malformed"],
                },
                "decline",
                None,
            )
        approval_id = f"approval-{uuid.uuid4()}"
        worker_id = self._approval_worker(params, provider)
        if worker_id is None:
            reason = _BOUNDARY_DECLINE_REASONS["unrouted"]
            self._record_boundary_decline(None, approval_id, params, reason)
            response = {"decision": "decline", "reason": reason}
            return self._finish_approval(
                method,
                self._unattested_approval(params),
                response,
                "decline",
                approval_id,
            )
        effect = params.get("effect")
        tool_label, command = self._approval_subject(params)
        if effect == "execute":
            request = ApprovalRequest(
                request_id=approval_id,
                tool=tool_label,
                effect="execute",
                permission="workspace-execute",
                target="workspace",
                justification="runtime requested command approval",
                command=command,
            )
        elif effect == "modify":
            request = ApprovalRequest(
                request_id=approval_id,
                tool=tool_label,
                effect="modify",
                permission="workspace-write",
                target="workspace",
                justification="runtime requested file approval",
                command=command,
            )
        elif effect in _READ_ONLY_APPROVALS:
            permission, target, justification = _READ_ONLY_APPROVALS[effect]
            request = ApprovalRequest(
                request_id=approval_id,
                tool=tool_label,
                effect=effect,
                permission=permission,
                target=target,
                justification=justification,
                command=command,
            )
        else:
            # An effect this reviewer cannot name is still declined, and the
            # decline is now an event of the session: the request is recorded,
            # the boundary names itself as the resolver, and the worker is told
            # what refused it instead of being told its manager did.
            reason = _BOUNDARY_DECLINE_REASONS["unnamed-effect"]
            self._record_boundary_decline(worker_id, approval_id, params, reason)
            response = {"decision": "decline", "reason": reason}
            return self._finish_approval(
                method,
                self._unattested_approval(params),
                response,
                "decline",
                approval_id,
            )
        outcome, decline_source = self._review_worker_approval(worker_id, request)
        if outcome.decision is ApprovalDecision.ACCEPT:
            decision = ApprovalDecision.ACCEPT
            response = {"decision": decision.value}
        else:
            decision = ApprovalDecision.DECLINE
            response = {
                "decision": decision.value,
                "reason": (
                    _MANAGER_DECLINE_REASON
                    if decline_source == "manager"
                    else _BOUNDARY_DECLINE_REASONS.get(
                        decline_source, _BOUNDARY_DECLINE_REASONS["unknown"]
                    )
                ),
            }
        return self._finish_approval(
            method, params, response, decision.value, approval_id
        )

    @staticmethod
    def _unattested_approval(params: Mapping[str, Any]) -> dict[str, Any]:
        """Keep a locally routed decline from claiming provider correlation."""

        value = dict(params)
        value["correlation_attested"] = False
        return value

    def _approval_worker(
        self, params: Mapping[str, Any], provider: str
    ) -> str | None:
        """Resolve an approval through exactly one trusted correlation route."""

        if params.get("correlation_attested") is True:
            if "routing_handle" in params:
                return None
            correlation = params.get("provider_correlation")
            if not self._exact_nonempty_strings(
                correlation, {"session", "turn", "request"}
            ):
                return None
            return self._worker_for_attested_provider_session(
                provider, correlation["session"]
            )
        if params.get("correlation_attested") is False:
            if "provider_correlation" in params:
                return None
            routing_handle = params.get("routing_handle")
            if not self._exact_nonempty_strings(
                routing_handle, {"reservation_id", "turn_reference"}
            ):
                return None
            reservation_id = routing_handle["reservation_id"]
            worker_id = self.managed.agent_for_thread(reservation_id)
            if not isinstance(worker_id, str) or not self._local_reservation_is_owned(
                worker_id, provider, reservation_id
            ):
                return None
            return worker_id
        return None

    @staticmethod
    def _exact_nonempty_strings(value: Any, keys: set[str]) -> bool:
        return bool(
            isinstance(value, Mapping)
            and set(value) == keys
            and all(
                isinstance(value[key], str)
                and value[key]
                and value[key].strip() == value[key]
                for key in keys
            )
        )

    def _worker_for_attested_provider_session(
        self, provider: str, provider_session: str
    ) -> str | None:
        """Find the unique worker attested to the native provider session.

        The cached attestations are a snapshot taken when each turn started.  A
        provider that binds its session identity partway through a turn is not
        in that snapshot yet, so its own worker's first approval would find no
        route and be declined -- the worker asks to write, nothing reaches its
        manager, and the write fails with no approval ever recorded.  When the
        snapshot has no unique match, the live attestations are re-read and the
        match is retried.

        Nothing is assumed on the retry.  A worker still has to be attested,
        bound, and carrying that exact provider session to be routed to, and a
        session that matches two workers is still refused.
        """

        match = self._match_attested_worker(provider, provider_session)
        if match is not None:
            return match
        self._rebind_active_identities(provider)
        match = self._match_attested_worker(provider, provider_session)
        if match is not None:
            return match
        # The provider is already running by the time the scheduler publishes
        # its turn, so an approval can arrive while _active_turns is still
        # empty and there is nothing yet to match against. Wait for the
        # publication rather than declining the worker's first question.
        deadline = time.monotonic() + APPROVAL_ROUTING_GRACE_SECONDS
        while True:
            with self._active_turns_condition:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                # Waiting releases the lock, so the thread publishing the
                # handle can take it and wake this one the moment it lands.
                self._active_turns_condition.wait(timeout=remaining)
            # Re-reading an identity calls the provider, so it happens with no
            # scheduler lock held. A slow re-read must never delay the very
            # publication this is waiting for.
            match = self._match_attested_worker(provider, provider_session)
            if match is not None:
                return match
            self._rebind_active_identities(provider)
            match = self._match_attested_worker(provider, provider_session)
            if match is not None:
                return match

    def _match_attested_worker(
        self, provider: str, provider_session: str
    ) -> str | None:
        with self.managed._lock:
            matches = [
                worker_id
                for worker_id, identity in self.managed._identity_attestations.items()
                if self._native_identity_is_attested(identity, provider, provider_session)
            ]
        return matches[0] if len(matches) == 1 else None

    def _rebind_active_identities(self, provider: str) -> None:
        """Re-read the provider's own identity for every agent holding a turn.

        Scoped to active turns because an idle agent cannot be the source of a
        live approval, and to one provider because the other runtimes have no
        stake in this request.
        """

        with self._lock:
            active = [
                (turn.agent_id, turn.runtime.thread_id)
                for turn in self._active_turns.values()
            ]
        for agent_id, thread_id in active:
            with self.managed._lock:
                cached = self.managed._identity_attestations.get(agent_id)
            if isinstance(cached, Mapping) and cached.get("provider") != provider:
                continue
            try:
                self.managed._refresh_identity(agent_id, thread_id)
            except Exception:
                # A runtime that cannot attest right now simply stays
                # unroutable; it is never given a fabricated identity.
                continue

    @staticmethod
    def _native_identity_is_attested(
        identity: Any,
        provider: str,
        provider_session: str,
    ) -> bool:
        """Check an attested provider fact without aliasing it to a reservation."""

        return bool(
            isinstance(identity, Mapping)
            and identity.get("provider") == provider
            and identity.get("provider_session") == provider_session
            and identity.get("binding_phase") == "attested"
            and identity.get("bound") is True
        )

    def _local_reservation_is_owned(
        self,
        worker_id: str,
        provider: str,
        reservation_id: str,
    ) -> bool:
        """Check a local reservation without treating it as a provider session."""

        with self.managed._lock:
            identity = self.managed._identity_attestations.get(worker_id)
        return bool(
            isinstance(identity, Mapping)
            and identity.get("runtime_thread") == reservation_id
            and identity.get("provider") == provider
        )

    def _finish_approval(
        self,
        method: str,
        params: Mapping[str, Any],
        response: dict[str, Any],
        decision: str,
        approval_id: str | None,
    ) -> dict[str, Any]:
        reason = response.get("reason")
        self._record_approval(
            method,
            params,
            decision,
            approval_id,
            reason if isinstance(reason, str) and reason else None,
        )
        try:
            self.hooks.native_approval(method, params, response)
        except Exception:
            pass
        return response

    def _record_approval(
        self,
        method: str,
        params: Mapping[str, Any],
        decision: str,
        approval_id: str | None,
        reason: str | None = None,
    ) -> None:
        correlation = params.get("provider_correlation")
        correlation = (
            correlation
            if params.get("correlation_attested") is True
            and isinstance(correlation, Mapping)
            else {}
        )
        _tool_label, command = self._approval_subject(params)
        subject_field = self._approval_subject_fields(command)
        with self._lock:
            self._native_approval_records.append(
                {
                    "approval_reference": approval_id,
                    "method": method,
                    "decision": decision,
                    # Which side said no.  A decline with no reason beside it
                    # is the one shape this record must never take again.
                    **({"reason": reason} if reason else {}),
                    # What was read, fetched or run.  An accepted standing
                    # grant never reaches a manager, so this record and the
                    # lifecycle events are the only account of it.
                    **subject_field,
                    "session_correlated": bool(
                        correlation.get("session")
                    ),
                    "turn_correlated": bool(
                        correlation.get("turn")
                    ),
                    "request_correlated": bool(
                        correlation.get("request")
                    ),
                }
            )

    def _record_boundary_decline(
        self,
        worker_id: str | None,
        approval_id: str,
        params: Mapping[str, Any],
        reason: str,
    ) -> None:
        """Leave an approval pair for a decline this boundary made by itself.

        A request the reviewer cannot name an effect for never reached a
        manager, so nothing at all used to appear in runs/<session>.jsonl: the
        attempt survived only as prose inside the worker's own text.  The pair
        below names this boundary as the resolver, which is what tells an
        operator it apart from a manager's refusal.

        A request routed to no worker (`worker_id` None) is recorded against
        the session root, the one agent always present, and names no worker.
        Eleven such declines in one day once left no event at all: every
        escalation of a granted Codex worker was refused and nothing showed it.
        """

        session = self._session()
        if worker_id is None:
            worker = session.agents.get(self.root.agent_id)
            routed: dict[str, Any] = {}
            provider = params.get("provider")
            if isinstance(provider, str) and provider:
                routed["provider"] = provider
            # Only an attested correlation names a provider session; a local
            # reservation is recorded as what it is.
            correlation = params.get("provider_correlation")
            handle = params.get("routing_handle")
            if params.get("correlation_attested") is True and isinstance(
                correlation, Mapping
            ):
                unrouted = ("provider_session", correlation.get("session"))
            elif isinstance(handle, Mapping):
                unrouted = ("routing_reservation", handle.get("reservation_id"))
            else:
                unrouted = ("", None)
            if isinstance(unrouted[1], str) and unrouted[1]:
                routed[unrouted[0]] = unrouted[1]
        else:
            worker = session.agents.get(worker_id)
            routed = {} if worker is None else {
                "worker_agent": worker.agent_id,
                **(
                    {"manager_agent": worker.parent_agent_id}
                    if isinstance(worker.parent_agent_id, str)
                    else {}
                ),
            }
        if worker is None:
            return
        tool_label, command = self._approval_subject(params)
        subject_field = self._approval_subject_fields(command)
        effect = params.get("effect")
        self.hooks.lifecycle(
            "approval_requested",
            worker,
            {
                "approval_id": approval_id,
                **routed,
                "tool": tool_label,
                "effect": effect if isinstance(effect, str) and effect else "unnamed",
                **subject_field,
            },
        )
        self.hooks.lifecycle(
            "approval_resolved",
            worker,
            {
                "approval_id": approval_id,
                "decision": ApprovalDecision.DECLINE.value,
                "resolver": _BOUNDARY_RESOLVER,
                "reason": reason,
                **subject_field,
            },
        )

    def _review_worker_approval(
        self,
        worker_id: str,
        request: ApprovalRequest,
    ) -> tuple[ApprovalOutcome, str]:
        """Rendezvous a blocking native Worker effect with its parent manager.

        Returns the outcome and which side decided it: "manager" for a real
        manager decision, including a standing grant, or a key into
        `_BOUNDARY_DECLINE_REASONS` for a decline this boundary made alone.
        Only the first may be reported to a worker as a manager decline.
        """

        session = self._session()
        worker = session.agents.get(worker_id)
        if worker is None or worker.parent_agent_id is None:
            return (
                ApprovalOutcome(ApprovalDecision.DECLINE, "Worker parent is unavailable"),
                "no-manager",
            )
        subject_field = self._approval_subject_fields(request.command)
        if worker.approvals == "granted":
            self.hooks.lifecycle(
                "approval_requested",
                worker,
                {
                    "approval_id": request.request_id,
                    "worker_agent": worker.agent_id,
                    "manager_agent": worker.parent_agent_id,
                    "tool": request.tool,
                    "effect": request.effect,
                    **subject_field,
                },
            )
            self.hooks.lifecycle(
                "approval_resolved",
                worker,
                {
                    "approval_id": request.request_id,
                    "decision": ApprovalDecision.ACCEPT.value,
                    "resolver": "standing-grant",
                    **subject_field,
                },
            )
            return ApprovalOutcome(ApprovalDecision.ACCEPT, "standing grant"), "manager"
        envelope = _ApprovalEnvelope(
            approval_id=request.request_id,
            manager_id=worker.parent_agent_id,
            worker_id=worker.agent_id,
            request=request,
            resolved=threading.Event(),
        )
        with self._lock:
            self._pending_approvals[envelope.approval_id] = envelope
        try:
            self.managed.control.request_attention(
                manager_id=envelope.manager_id,
                source_agent_id=envelope.worker_id,
                reason="approval-requested",
            )
        except ProtocolError as exc:
            with self._lock:
                self._pending_approvals.pop(envelope.approval_id, None)
            # The run log holds a requested/resolved pair for every other
            # decline, so a failed wake leaves one too.
            self.hooks.lifecycle(
                "approval_requested",
                worker,
                {
                    "approval_id": envelope.approval_id,
                    "worker_agent": worker.agent_id,
                    "manager_agent": envelope.manager_id,
                    "tool": request.tool,
                    "effect": request.effect,
                    **subject_field,
                },
            )
            self.hooks.lifecycle(
                "approval_resolved",
                worker,
                {
                    "approval_id": envelope.approval_id,
                    "decision": ApprovalDecision.DECLINE.value,
                    "resolver": _BOUNDARY_RESOLVER,
                    "reason": _BOUNDARY_DECLINE_REASONS["routing-failed"],
                    **subject_field,
                },
            )
            return (
                ApprovalOutcome(ApprovalDecision.DECLINE, f"approval routing failed: {exc.code}"),
                "routing-failed",
            )
        self.hooks.lifecycle(
            "approval_requested",
            worker,
            {
                "approval_id": envelope.approval_id,
                "worker_agent": worker.agent_id,
                "manager_agent": envelope.manager_id,
                "tool": request.tool,
                "effect": request.effect,
                **subject_field,
            },
        )
        self._events.put("approval")
        with self._lock:
            active_manager_turn = self._active_turns.get(envelope.manager_id)
        if active_manager_turn is not None and envelope.manager_id not in self._raw_prompt_agents:
            try:
                self.managed._adapter_for(envelope.manager_id).steer(
                    active_manager_turn.runtime,
                    self._approval_prompt(envelope),
                )
            except Exception:
                # The ordinary queued-wake path remains available if the
                # manager turn ends before the approval deadline.
                pass
        if not envelope.resolved.wait(timeout=min(self.turn_timeout, 240)):
            with self._lock:
                self._pending_approvals.pop(envelope.approval_id, None)
            outcome = ApprovalOutcome(ApprovalDecision.DECLINE, "Branch approval timed out")
            self.hooks.lifecycle(
                "approval_resolved",
                worker,
                {
                    "approval_id": envelope.approval_id,
                    "decision": outcome.decision.value,
                    "resolver": "timeout",
                    **subject_field,
                },
            )
            return outcome, "timeout"
        with self._lock:
            self._pending_approvals.pop(envelope.approval_id, None)
        decided = envelope.outcome
        outcome = decided or ApprovalOutcome(
            ApprovalDecision.DECLINE,
            "Branch approval ended without a decision",
        )
        self.hooks.lifecycle(
            "approval_resolved",
            worker,
            {
                "approval_id": envelope.approval_id,
                "decision": outcome.decision.value,
                "resolver": envelope.resolver or "system",
                **subject_field,
            },
        )
        if decided is None:
            return outcome, "undecided"
        return outcome, ("cancelled" if envelope.resolver == _BOUNDARY_RESOLVER else "manager")

    @staticmethod
    def manager_tools() -> list[dict[str, Any]]:
        task_contract = {
            "type": "object",
            "properties": {"criteria": {"type": "array", "items": {"type": "string"}}},
            "required": ["criteria"],
            "additionalProperties": True,
        }
        evidence_request = {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["run_command"]},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 64,
                            "items": {"type": "string", "minLength": 1, "maxLength": 4096},
                        },
                        "cwd": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 120},
                    },
                    "required": ["argv", "cwd", "timeout_seconds"],
                    "additionalProperties": False,
                },
                "completion": {
                    "type": "object",
                    "properties": {"exit_code": {"type": "integer"}},
                    "required": ["exit_code"],
                    "additionalProperties": False,
                },
            },
            "required": ["tool", "arguments", "completion"],
            "additionalProperties": False,
        }
        agent_ids = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "maxItems": 32,
        }
        return [
            function_tool(
                "delegate",
                "Create one direct child. Every agent may execute work and delegate when that keeps context, cost, or quality under control.",
                properties={
                    "role": {
                        "type": "string",
                        "enum": [role.value for role in AgentRole],
                    },
                    "model_id": {"type": "string", "minLength": 1},
                    "effort": {
                        "type": "string", "minLength": 1,
                        "description": (
                            "Defaults to high when omitted. Claude and z.ai accept: "
                            + ", ".join(sorted(CLAUDE_ACCEPTED_EFFORTS)) + ". "
                            "Codex and Command Code efforts depend on the selected model."
                        ),
                    },
                    "objective": {"type": "string", "minLength": 1},
                    "task_contract": task_contract,
                    "required_tool_effect": evidence_request,
                    "workspace": {
                        "type": "string",
                        "enum": ["shared", "private", "worktree"],
                        "description": (
                            "shared: this child works in the session workspace, "
                            "as every agent does by default, and can read and "
                            "change the same files as its siblings. private: "
                            "this child gets an empty working directory of its "
                            "own, inside the session workspace, holding none of "
                            "the project. Files it writes by a relative path "
                            "land there, so they cannot collide with a "
                            "sibling's. The directory is a starting point and "
                            "fences nothing: the child's tools can still read "
                            "and change any path they are given, the project "
                            "included. Choose it for work that produces "
                            "something new on bare ground, such as a report or "
                            "a fresh artifact, and for Workers you intend to "
                            "run at the same time on output that would "
                            "collide. Choose shared when children are meant to be "
                            "working on one tree together, and keep their write "
                            "scopes disjoint or sequence them. worktree: this child gets a git "
                            "checkout of its own, on a branch of its own, "
                            "seeded from the last commit. Choose it when two "
                            "children must change the SAME existing code at "
                            "the same time. Two things to know before you do: "
                            "the child cannot see any work that has not been "
                            "committed, so anything uncommitted is invisible "
                            "to it; and nothing merges the branch for you. You "
                            "are handed the path and the branch when the child "
                            "finishes, and what happens to them is your "
                            "decision. Submodules arrive empty and the "
                            "child must initialise them itself. A checkout the child left unchanged is "
                            "removed along with its branch."
                        ),
                    },
                    "approvals": {
                        "type": "string",
                        "enum": ["granted", "ask"],
                        "description": (
                            "granted (the default): this child's requests to edit files "
                            "or run commands are accepted at once and written to the run "
                            "log, so it never waits on you. ask: each such request waits "
                            "for your resolve_approval call; choose it for a child whose "
                            "actions you want to review one by one."
                        ),
                    },
                },
                # The server starts a child whose task_contract is absent and
                # records evidence_request_recorded false, so a schema that
                # called it required made a validating client refuse a call the
                # server accepts.  The schema follows the server.
                required=["role", "model_id", "objective"],
            ),
            function_tool(
                "await_children",
                "Yield this manager turn until one of the selected direct children changes "
                "materially. Every child that has stopped carries the text it reported in "
                "outcome, with verified and evidence beside it. A blocked child is not "
                "waiting on anything except a decision "
                "from you, so waiting only for stopped children is refused.",
                properties={"agent_ids": agent_ids},
                required=["agent_ids"],
            ),
            function_tool(
                "inspect",
                'Inspect yourself with agent_id "self", or inspect another agent in this session. '
                "Native inbox messages repeat until acknowledged: after processing them, "
                'inspect with agent_id "self" and acknowledge_messages_through '
                "set to the returned message_cursor. Acknowledge before completing."
                " The response includes direct children with canonical agent_id, runtime_thread_id, status and terminal. "
                "Use agent_id for vNext controls and waits."
                " An agent that has stopped returns what it reported: outcome, verified and evidence.",
                properties={
                    "agent_id": {"type": "string"},
                    "deep": {"type": "boolean"},
                    "acknowledge_messages_through": {"type": "integer", "minimum": 0},
                },
                required=["deep"],
            ),
            function_tool(
                "steer",
                "Send a bounded instruction to another agent, steering its active turn when possible.",
                properties={
                    "agent_id": {"type": "string", "minLength": 1},
                    "message": {"type": "string", "minLength": 1, "maxLength": 8192},
                },
                required=["agent_id", "message"],
            ),
            function_tool(
                "send_message",
                "Send contextual peer mail to another live agent in this session.",
                properties={
                    "agent_id": {"type": "string", "minLength": 1},
                    "message": {"type": "string", "minLength": 1, "maxLength": 8192},
                },
                required=["agent_id", "message"],
            ),
            function_tool(
                "interrupt_agent",
                "Interrupt one active agent turn without cancelling its work or descendants. A later message resumes it.",
                properties={"agent_id": {"type": "string", "minLength": 1}},
                required=["agent_id"],
            ),
            function_tool(
                "retry",
                "Retry one blocked or failed direct child on its persistent session.",
                properties={
                    "agent_id": {"type": "string"},
                    "task_contract": task_contract,
                    "start_despite_unconfirmed_stop": {
                        "type": "boolean",
                        "description": (
                            "true starts this agent although the turn it replaces "
                            "never confirmed it stopped; both may then write the "
                            "same files"
                        ),
                    },
                },
                required=["agent_id", "task_contract"],
            ),
            function_tool(
                "replace",
                "Replace one direct child with an eligible model selected for its task.",
                properties={
                    "agent_id": {"type": "string"},
                    "model_id": {"type": "string", "minLength": 1},
                    "objective": {"type": "string", "minLength": 1},
                    "task_contract": task_contract,
                },
                required=["agent_id", "model_id", "task_contract"],
            ),
            function_tool(
                "cancel_agent",
                "Cancel one agent subtree owned by this manager. The session root cannot cancel itself because that would leave the session unable to start more work; cancel its children or call complete_session instead.",
                properties={"agent_id": {"type": "string", "minLength": 1}},
                required=["agent_id"],
            ),
            function_tool(
                "resolve_approval",
                "Resolve one pending command approval assigned to this agent.",
                properties={
                    "approval_id": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "enum": ["accept", "decline"]},
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 1000},
                },
                required=["approval_id", "decision", "rationale"],
            ),
            function_tool(
                "complete_branch",
                "Record this agent's final outcome after every direct child is terminal. The primary uses complete_session.",
                properties={
                    "outcome": {"type": "string", "minLength": 1},
                    "verified": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 32},
                },
                required=["outcome", "verified", "evidence"],
            ),
            function_tool(
                "complete_session",
                "Record the primary agent's final judgment after every direct child is terminal.",
                properties={
                    "decision": {
                        "type": "string",
                        "enum": ["accepted", "rejected", "needs-review"],
                    },
                    "summary": {"type": "string", "minLength": 1},
                    "criteria": {"type": "object", "additionalProperties": {"type": "boolean"}},
                },
                required=["decision", "summary", "criteria"],
            ),
            function_tool(
                "complete_agent",
                "Complete this agent's own task after every direct child is terminal. The primary uses complete_session for its final objective.",
                properties={
                    "outcome": {"type": "string", "minLength": 1},
                    "verified": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 32},
                },
                required=["outcome", "verified", "evidence"],
            ),
            function_tool(
                "report_blocked",
                "Use this when you cannot go on without something only your manager or user can give, "
                "such as a token, an access or a decision. You stay alive with your context, and your "
                "manager is told the reason and answers you with a message.",
                properties={"reason": {"type": "string", "minLength": 1}},
                required=["reason"],
            ),
        ]

    def _session(self):
        return self.managed.control.sessions[self.managed.session_id]

    def bind_resumed_primary(self, *, provider_session: str, raw_user_prompts: bool = False) -> None:
        """Attest an idle primary in a fresh controller, without starting a turn.

        ``raw_user_prompts`` marks a native chat vNext did not start: its turns
        carry only the user's text, so the role guidance and any session
        instructions go into the system prompt append instead.
        """
        agent = self.root
        if agent.thread_id is None or agent.status not in {AgentStatus.READY, AgentStatus.COMPLETED}:
            raise SchedulerError("primary has no resumable quiescent thread")
        with self._lock:
            if self._bound or self._active_turns:
                raise SchedulerError("scheduler already owns runtime work")
        adapter = self.managed.adapter_for_agent(agent)
        resume = getattr(adapter, "resume_attested_thread", None)
        if not callable(resume):
            raise SchedulerError("provider does not support attested primary resume")
        handler = lambda tool, arguments, context: self._manager_handler(
            agent.agent_id, tool, arguments, context
        )
        result = resume(
            runtime_thread=agent.thread_id, provider_session=provider_session,
            model=agent.model_id, effort=agent.effort,
            workspace=str(self.managed.ensure_workspace(agent.agent_id)),
            approvals_reviewer="auto_review", tools=self.manager_tools(),
            developer_instructions=self._manager_instructions(agent.role) + (
                self._contract_instructions(agent) if raw_user_prompts else ""
            ),
            tool_handler=handler,
        )
        identity = adapter.thread_identity_attestation(agent.thread_id)
        if (identity.get("bound") is not True
                or identity.get("provider_session") != provider_session
                or identity.get("runtime_thread") != agent.thread_id):
            raise SchedulerError("resumed primary identity differs from saved thread")
        policy = result.get("policy")
        if not isinstance(policy, Mapping):
            raise SchedulerError("resumed primary lacks an effective policy")
        self.managed.bind_thread(agent_id=agent.agent_id, thread_id=agent.thread_id,
                                 start_result=policy, tool_handler=handler, adapter=adapter)
        with self._lock:
            self._bound.add(agent.agent_id)
            self._awaiting_message_agents.add(agent.agent_id)
            if raw_user_prompts:
                self._raw_prompt_agents.add(agent.agent_id)
            # The historical idle/completion fact is already durable. Only a
            # new user prompt may produce a new objective lifecycle here.
            self._idle_notified = True

    def _report_binding_failure(
        self, agent: AgentRecord, adapter: Any, exc: BaseException,
        *, phase: str = "start_thread",
    ) -> None:
        """Put the whole failure on disk, once, and never let that raise.

        A worker that would not bind left one sentence in the blocker and
        nothing else.  When that sentence was wrong -- an ImportError reported
        as a package that was not installed, while pip and a plain import both
        found it -- there was nothing on disk to check it against.
        """

        provider = str(getattr(adapter, "provider", ""))
        try:
            # Credential-adjacent stderr is never persisted; the constant
            # says why, and VNextRuntimeSession._emit_provider_error reads it
            # too so the two paths cannot drift.
            stderr = (
                []
                if provider in CREDENTIAL_ADJACENT_STDERR_PROVIDERS
                else list(adapter.captured_stderr()) if adapter is not None else []
            )
        except Exception:
            stderr = []
        try:
            self.hooks.lifecycle("provider_error", agent, {
                "provider": provider,
                "model_id": agent.model_id,
                "role": agent.role.value,
                "phase": phase,
                "error": str(exc),
                "kind": type(exc).__name__,
                "cause": (
                    f"{type(exc.__cause__).__name__}: {exc.__cause__}"
                    if exc.__cause__ else None
                ),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
                "provider_stderr": stderr,
            })
        except Exception:
            # Observing a failure must never be what destroys it.
            pass

    def _report_turn_failure(self, agent: AgentRecord, event: _TurnFinished) -> str:
        """Record why a turn died, and return the sentence to block it with.

        This used to say "runtime turn failed" and throw the exception away for
        every agent but the root.  Five workers died at once behind that
        sentence and there was nothing on disk to say whether it was one cause
        or five.  The exception is the only thing that distinguishes a provider
        that stopped answering from a turn that simply ran longer than the
        wait, so it goes into the record and into the blocker both.
        """

        exc = event.error
        kind = type(exc).__name__ if exc is not None else "UnknownError"
        detail = _safe_detail(exc)
        reason = f"runtime turn failed: {kind}: {detail}" if detail else f"runtime turn failed: {kind}"
        try:
            self.hooks.lifecycle("provider_error", agent, {
                "provider": event.turn.provider,
                "model_id": agent.model_id,
                "role": agent.role.value,
                "phase": "run_turn",
                "turn_id": event.turn.control_turn_id,
                "native_turn_id": getattr(event.turn.runtime, "turn_id", None),
                "error": detail,
                "kind": kind,
                "cause": (
                    f"{type(exc.__cause__).__name__}: {exc.__cause__}"
                    if exc is not None and exc.__cause__ else None
                ),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ) if exc is not None else "",
            })
        except Exception:
            # Observing a failure must never be what destroys it.
            pass
        return reason

    def _provider_turn_ended(self, turn: ManagedTurn) -> bool:
        """Whether anything can show the far side of this turn really ended.

        The question asked here is the one that can be answered honestly: has
        the provider process for this turn exited?  A dead process cannot
        still be writing the workspace, so that is evidence.  Everything else
        -- a closed socket, a refused call, an adapter that owns no process --
        reads as still running, because unknown is the answer that keeps a
        second writer out of the folder.
        """

        try:
            return self.managed.turn_process_ended(turn) is True
        except BaseException:
            # Asking must never be what turns an unconfirmed stop into a
            # confirmed one, so a query that fails reads as unknown.
            return False

    def _stop_unconfirmed_provider_turn(
        self, agent: AgentRecord, event: _TurnFinished, *, timed_out: bool = True
    ) -> None:
        """Ask the provider to stop a turn whose end nothing has confirmed.

        Two shapes arrive.  Our own wait expiring, and a wait that failed for
        another reason with no evidence the provider process ended -- the
        app-server WebSocket dropping is the common one.  Neither says
        anything about the far side.  The model keeps
        running commands, keeps spending tokens, and keeps writing to the
        shared workspace, while vNext has already marked the agent failed and
        told its manager so -- and a manager that replaces or retries then puts
        a second writer in the same folder as the first.  So the same stop
        cancel uses is sent here too.

        Best-effort on purpose: the provider may be the very thing that stopped
        answering.  An interrupt that fails is written down as evidence and
        never raised, because the failure being recorded above is the fact the
        manager actually needs.
        """

        runtime = getattr(event.turn, "runtime", None)
        if runtime is None:
            return
        # Our wait ended; nothing says the turn did.  The caller registered it as
        # stopping under the lock that took it off the active list, so a manager
        # that reads the failure and replaces or retries this agent in the same
        # prompt finds it on the stopping list however fast it moves.  Written
        # again here for the callers that come straight to this method.
        self._mark_turn_stopping(agent.agent_id, event.turn)
        outcome = "requested"
        detail: str | None = None
        try:
            self.managed._adapter_for(agent.agent_id).interrupt(runtime)
        except BaseException as exc:
            outcome = "refused"
            detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        try:
            self.hooks.lifecycle(
                # A timeout keeps the record it has always had, so logs written
                # before transport failures were held read the same.  The new
                # case carries its own name and reason.
                "turn_timeout_interrupt" if timed_out else "turn_unconfirmed_stop_interrupt",
                agent,
                {
                    "role": agent.role.value,
                    "provider": event.turn.provider,
                    "turn_id": event.turn.control_turn_id,
                    "native_turn_id": getattr(runtime, "turn_id", None),
                    "reason": "wait_timeout" if timed_out else "transport_failure",
                    "outcome": outcome,
                    "error": detail,
                },
            )
        except Exception:
            # Observing the stop must never be what loses the failure record.
            pass
        self._watch_stopping_turn(agent.agent_id, event.turn)

    def _watch_stopping_turn(self, agent_id: str, turn: ManagedTurn) -> None:
        """Wait once more, briefly, for an interrupted turn to say it ended.

        The interrupt is a request.  Only the provider's own turn completion
        proves the far side stopped writing to the workspace, so one more wait
        is spent on the same turn, bounded by the replacement barrier: past
        that deadline the barrier has already made its own decision and a
        longer wait would change nothing.

        Only a normal return clears the stopping record.  An adapter that
        raises has told us nothing about the turn, and reading that as
        "stopped" is the guess that puts two writers in one workspace.  So
        the wait is spent in a few slices instead of one, and between them the
        other question is asked: a turn whose provider process exits part way
        through the barrier releases the hold when it does.
        """

        def confirmed() -> None:
            if self._forget_stopping_turn(agent_id, turn):
                # This watcher's turn was the last unconfirmed one of this
                # agent.  While another is still held, whatever waits behind
                # this agent is still waiting on a live writer.
                self._release_blocks_on_confirmed_stop(agent_id)
            self._events.put("control")

        def wait() -> None:
            deadline = time.monotonic() + self.replacement_barrier_seconds
            slice_seconds = self.replacement_barrier_seconds / _STOP_EVIDENCE_CHECKS
            while True:
                started = time.monotonic()
                try:
                    self.managed.wait_turn(
                        turn,
                        timeout=min(slice_seconds, max(0.0, deadline - started)),
                    )
                except BaseException:
                    # Deliberately silent: the failure the manager needs is
                    # already recorded, and this thread's only job is the stop
                    # evidence.
                    pass
                else:
                    confirmed()
                    return
                if self._provider_turn_ended(turn):
                    confirmed()
                    return
                left = deadline - time.monotonic()
                if left <= 0:
                    return
                # An adapter that refuses at once must not turn the barrier
                # into a spin, so the rest of the slice is slept through.
                time.sleep(min(left, max(0.0, slice_seconds - (time.monotonic() - started))))

        # Not registered among the turn waiters: those decide whether the
        # controller may shut down idle, and a stop that is merely unconfirmed
        # must not hold a finished tree open for the whole barrier.
        threading.Thread(
            target=wait,
            name=f"vnext-stopping-{agent_id[:8]}",
            daemon=True,
        ).start()

    @staticmethod
    def _turn_identity(turn: ManagedTurn) -> str:
        """One turn's own name, so a receipt releases only the turn it watched.

        The control turn id is the identity vNext itself hands out and the one
        that appears in the run log.  A turn built without one still has to be
        tellable apart from its successor, so the object stands for itself.
        """

        control_turn_id = getattr(turn, "control_turn_id", None)
        if isinstance(control_turn_id, str) and control_turn_id:
            return control_turn_id
        return f"turn-{id(turn):x}"

    def _mark_turn_stopping(self, agent_id: str, turn: ManagedTurn) -> None:
        """Record one turn as asked to stop and not yet confirmed stopped."""

        with self._lock:
            self._stopping_turns.setdefault(agent_id, {}).setdefault(
                self._turn_identity(turn), turn
            )

    def _forget_stopping_turn(self, agent_id: str, turn: ManagedTurn) -> bool:
        """Release one turn's hold.  True when none of this agent's are left.

        A caller that holds evidence about one turn may only speak for that
        turn.  Whatever is blocked behind this agent stays blocked while any
        other turn of it is still unconfirmed.
        """

        with self._lock:
            turns = self._stopping_turns.get(agent_id)
            if turns is None:
                return True
            turns.pop(self._turn_identity(turn), None)
            if turns:
                return False
            # An empty entry would read as "still stopping" to every membership
            # test in the barrier, so the key goes with its last turn.
            self._stopping_turns.pop(agent_id, None)
            return True

    def _latest_stopping_turn(self, agent_id: str) -> ManagedTurn | None:
        """The most recently held turn of this agent, or None if it holds none."""

        with self._lock:
            turns = self._stopping_turns.get(agent_id)
            if not turns:
                return None
            return next(reversed(list(turns.values())))

    def _predecessor_chain(self, agent: AgentRecord) -> list[str]:
        """The agent, then everything it replaced or retried, oldest last.

        A replacement records the attempt it came from in ``retry_of_agent_id``,
        so A <- B <- C is already written down and this only reads it.  The
        agent itself opens the list because a retry reuses one record: the turn
        that may still be writing is then its own previous turn.
        """

        session = self._session()
        chain = [agent.agent_id]
        seen = {agent.agent_id}
        cursor = agent.retry_of_agent_id
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            predecessor = session.agents.get(cursor)
            cursor = predecessor.retry_of_agent_id if predecessor is not None else None
        return chain

    def _busy_predecessor(self, agent: AgentRecord) -> str | None:
        """The first agent in this one's chain whose turn may still be writing.

        Three states say "may still be writing", and all three are read here
        rather than once per path that creates an attempt.  A turn on the active
        list is running.  An agent in ``_starting_agents`` has been handed to its
        provider and the native handle has not come back, so there is no turn to
        interrupt yet and the model may already be working.  A turn on the
        stopping list was asked to stop and never said it had.

        Only the predecessors are read for the starting state: the agent itself
        is the one this scan is about to start, and it reaches here before any
        start of its own.
        """

        chain = self._predecessor_chain(agent)
        with self._lock:
            for position, candidate in enumerate(chain):
                if candidate in self._active_turns or candidate in self._stopping_turns:
                    return candidate
                if position and candidate in self._starting_agents:
                    return candidate
        return None

    def _chain_clear_to_start(self, agent: AgentRecord) -> bool:
        """Whether this agent may begin a provider turn yet.

        One rule, at the one place every automatic start passes through: an
        agent starts only when no attempt it came from may still be writing the
        shared workspace.  Four rounds of review each found another way to
        reach a start with a live predecessor -- a replace of a replacement, a
        predecessor still inside its own start, a handoff between two lists --
        because each path carried its own copy of the test.  The paths now only
        record who came before whom, and the question is asked here.

        The timing is unchanged: the predecessor is given the barrier to report
        its end, and past that deadline the agent is blocked with the reason and
        the name of the override in plain words.
        """

        if agent.agent_id in self._unconfirmed_start_overrides:
            # The manager was told the chain was unclear and said start anyway.
            # That is recorded, and it is not asked again on the next scan.
            return True
        predecessor_id = self._busy_predecessor(agent)
        if predecessor_id is None:
            with self._lock:
                self._replacement_barriers.pop(agent.agent_id, None)
            return True
        with self._lock:
            barrier = self._replacement_barriers.get(agent.agent_id)
            if barrier is None:
                # First scan that finds the chain busy starts the clock.
                deadline = time.monotonic() + self.replacement_barrier_seconds
                self._replacement_barriers[agent.agent_id] = (predecessor_id, deadline)
            else:
                deadline = barrier[1]
            if time.monotonic() < deadline:
                # No sleep: this loop is single-threaded and the run loop
                # rescans on its own 200ms timeout, so the hold is re-read well
                # inside the deadline.
                return False
            self._replacement_barriers.pop(agent.agent_id, None)
            self._unconfirmed_blocks[agent.agent_id] = predecessor_id
        self._block_on_unconfirmed_stop(predecessor_id, agent)
        return False

    def _block_on_unconfirmed_stop(
        self, predecessor_id: str, replacement: AgentRecord
    ) -> None:
        """Hold a replacement whose predecessor never confirmed it stopped.

        The wait is bounded because a provider can decline to stop a turn, and
        holding a replacement forever would turn one stuck worker into a stuck
        branch.  What the deadline buys is a decision rather than a start: the
        interrupt is sent again, the old turn moves to the stopping list, and
        the replacement is blocked with the reason in plain words.  Starting it
        here instead would put a second writer in the workspace on no evidence
        at all, which is the fault the barrier exists for.

        The manager decides from there.  A retry that names
        ``start_despite_unconfirmed_stop`` starts it, and that choice is written
        down; a retry without the flag is answered with this same reason and
        changes nothing.  The provider process itself
        is shared -- the manager's own connection is frequently the same object
        -- so closing it would stop the tree rather than the turn, and it is
        left alone and said so.
        """

        with self._lock:
            turn = self._active_turns.get(predecessor_id) or self._latest_stopping_turn(predecessor_id)
        detail: str | None = None
        if turn is not None:
            try:
                self.managed._adapter_for(predecessor_id).interrupt(turn.runtime)
            except BaseException as exc:
                detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        with self._lock:
            stopping = self._active_turns.pop(predecessor_id, None)
            if stopping is not None:
                # Off the active list, because no wait of ours is still on it,
                # and onto the stopping list, because it may still be writing.
                self._mark_turn_stopping(predecessor_id, stopping)
            self._turn_started_at.pop(predecessor_id, None)
            self._active_turns_condition.notify_all()
        blocker = (
            f"the previous turn of {predecessor_id} did not confirm it stopped within "
            f"{self.replacement_barrier_seconds:g}s; it may still be changing shared "
            f"files. Retry with start_despite_unconfirmed_stop true to start it "
            f"anyway, or cancel it."
        )
        try:
            self.managed.control.block_agent(replacement.agent_id, blocker)
        except Exception:
            # A replacement that went terminal by another route needs no block.
            with self._lock:
                self._unconfirmed_blocks.pop(replacement.agent_id, None)
        line = (
            f"replaced turn did not confirm it stopped within "
            f"{self.replacement_barrier_seconds:g}s: replacement "
            f"{replacement.agent_id} is blocked rather than started"
            # The run log keeps the provider's own words; the manager reads this.
            + (f"; second interrupt {_safe_text(detail)}" if detail else "")
        )
        session = self._session()
        predecessor = session.agents.get(predecessor_id)
        if predecessor is not None:
            self._record_stopped_progress(predecessor, line)
            try:
                self.hooks.lifecycle(
                    "replaced_turn_unconfirmed",
                    predecessor,
                    {
                        "role": predecessor.role.value,
                        "replacement_id": replacement.agent_id,
                        "waited_seconds": self.replacement_barrier_seconds,
                        "second_interrupt_error": detail,
                        "replacement_status": "blocked",
                        "provider_process": "left running, shared with the tree",
                    },
                )
            except Exception:
                pass
        self._events.put("control")

    def _unconfirmed_stop_hold(self, agent_id: str) -> str | None:
        """Which predecessor, if any, this agent is held behind right now."""

        with self._lock:
            return self._unconfirmed_blocks.get(agent_id)

    def _held_turns_have_ended(self, predecessor_id: str) -> bool:
        """Ask once more whether every held turn of this agent has ended.

        The barrier watcher stops asking at its deadline, and a provider
        process can exit after that.  Each turn whose process is now known
        to have exited is let go on its own evidence; True only when none is
        left.  An agent holding no turn at all answers False, because a block
        without a turn behind it says nothing either way.
        """

        with self._lock:
            turns = list(self._stopping_turns.get(predecessor_id, {}).values())
        cleared = False
        for turn in turns:
            if self._provider_turn_ended(turn):
                cleared = self._forget_stopping_turn(predecessor_id, turn)
        return cleared

    def _release_blocks_on_confirmed_stop(self, predecessor_id: str) -> None:
        """Free whatever was blocked once this turn reported its own end.

        The block is a decision taken on missing evidence: the predecessor said
        nothing inside the barrier.  When its turn does report it ended, the
        reason for the block is simply gone, so the hold is dropped and the
        starter picks the agent up on its next pass.  Asking the manager to
        retry here would charge it for news it could not have acted on.
        """

        with self._lock:
            released = [
                held_id
                for held_id, held_on in self._unconfirmed_blocks.items()
                if held_on == predecessor_id
            ]
            for held_id in released:
                self._unconfirmed_blocks.pop(held_id, None)
                self._replacement_barriers.pop(held_id, None)
        if not released:
            return
        session = self._session()
        for held_id in released:
            agent = session.agents.get(held_id)
            if agent is None:
                continue
            try:
                self.managed.control.release_unconfirmed_stop_block(held_id)
            except Exception:
                # Terminal by another route, or never blocked at all: there is
                # nothing left to free and nothing to report.
                continue
            try:
                self.hooks.lifecycle(
                    "replacement_released_after_confirmed_stop",
                    agent,
                    {
                        "role": agent.role.value,
                        "predecessor_id": predecessor_id,
                        "released_by": "the replaced turn reporting its end",
                    },
                )
            except Exception:
                pass
        self._events.put("control")

    def _start_despite_unconfirmed_stop(self, agent: AgentRecord) -> bool:
        """Clear a block the manager has chosen to override with a retry.

        The manager was told the predecessor never confirmed its stop and named
        ``start_despite_unconfirmed_stop`` anyway, so the hold is released and
        the choice is written into the run log where the two writers would be
        explained from.
        """

        with self._lock:
            predecessor_id = self._unconfirmed_blocks.pop(agent.agent_id, None)
            barrier = self._replacement_barriers.pop(agent.agent_id, None)
        if predecessor_id is None and barrier is not None:
            # Named inside the barrier, before the block: the manager still said
            # start, and waiting for the block to land first would answer a
            # decision it has already taken.
            predecessor_id = barrier[0]
        if predecessor_id is None:
            predecessor_id = self._busy_predecessor(agent)
        if predecessor_id is None:
            # Nothing to override.  The flag is harmless here and records
            # nothing, because no second writer was ever in question.
            return False
        with self._lock:
            self._unconfirmed_start_overrides.add(agent.agent_id)
        line = (
            f"started on a manager retry with the previous turn of {predecessor_id} "
            "still unconfirmed: both may be writing the same workspace"
        )
        try:
            self.hooks.lifecycle(
                "replacement_started_unconfirmed",
                agent,
                {
                    "role": agent.role.value,
                    "predecessor_id": predecessor_id,
                    "decided_by": "manager retry",
                    "explicit_override": True,
                    "note": line,
                },
            )
        except Exception:
            pass
        return True

    # Providers whose endpoint does not serve the runtime's own review model.
    # "auto_review" asks Codex to judge a child's work with a model named
    # codex-auto-review, and an endpoint that resells other vendors' models
    # answers that the model is unsupported.  Measured against Command Code on
    # 2026-09-18: three turns, every write refused, the file never written, and
    # the child reporting a blocker no manager could clear.  A worker already
    # sends its approvals to the manager above it; a manager on one of these
    # providers does the same, which is the path vNext owns anyway.
    _REVIEWS_THROUGH_THE_MANAGER = frozenset({"commandcode"})

    def _reviewer_for(self, agent: AgentRecord) -> str:
        """Who answers this child's approval requests."""

        if agent.role is AgentRole.WORKER:
            return "user"
        card = self.managed.control.registry.cards.get(agent.model_id)
        provider = str(getattr(card, "provider", "")) if card is not None else ""
        return "user" if provider in self._REVIEWS_THROUGH_THE_MANAGER else "auto_review"

    def _bind_agent(self, agent: AgentRecord) -> bool:
        """Bind one agent to a runtime. False means it was blocked instead.

        The caller has to know, because starting a turn on an agent that never
        bound raises out of the main loop and kills the tree -- which is
        precisely the failure blocking the child was written to prevent.
        """

        with self._lock:
            if agent.agent_id in self._bound:
                return True
        # Native execution and coordination are compatible capabilities.  A
        # worker is not downgraded to a tool-less leaf just because its current
        # role is execution-heavy; it can delegate a context-heavy subtask or
        # message a peer when the work calls for it.
        tools = self.manager_tools()
        instructions = self._manager_instructions(agent.role)
        handler = lambda tool, arguments, context, agent_id=agent.agent_id: self._manager_handler(
            agent_id, tool, arguments, context
        )
        try:
            agent_workspace = self.managed.ensure_workspace(agent.agent_id)
        except (ManagedSessionError, OSError) as exc:
            # A workspace that cannot be created is this child's problem, not
            # the tree's. Raising here killed the root, the manager and every
            # sibling, and the manager that chose the scope was never told
            # anything at all. Blocking the child wakes its manager with a
            # reason it can route around.
            if agent.role is AgentRole.ROOT_MANAGER:
                raise
            self._block(agent, f"workspace could not be prepared: {_safe_detail(exc)}")
            # Every other place the scheduler blocks a child does it from
            # _handle_turn_finished, after the manager has parked. This one
            # runs inside the main loop while the manager's turn may still be
            # in flight, so the loop is nudged to take another pass rather than
            # concluding there is nothing left to do.
            self._events.put("control")
            return False
        runtime_adapter = None
        try:
            runtime_adapter = self.managed.adapter_for_agent(agent)
            runtime_adapter.native_approval_handler = self.review_approval
            thread_id, start_result = runtime_adapter.start_thread(
                model=agent.model_id,
                effort=agent.effort,
                developer_instructions=instructions,
                tools=tools,
                tool_handler=handler,
                requested_posture=RuntimePosture(
                    workspace_writes=True,
                    network="restricted",
                    approvals_requested=True,
                    reviewer=self._reviewer_for(agent),
                    environment_ready=True,
                ),
                workspace=agent_workspace,
            )
        except Exception as exc:
            # A provider that will not start is this child's problem, the same
            # as a workspace that will not open.  It was killing the tree
            # instead: one Claude worker whose bridge timed out ended the whole
            # session, and every later tool call in that chat answered only that
            # the session had failed, until the user restarted the terminal.
            # The manager can route around a blocked child; it cannot route
            # around a dead session.
            if agent.role is AgentRole.ROOT_MANAGER:
                raise
            # The blocker line is what the manager reads and it has to stay one
            # sentence.  The traceback and the provider's own stderr are what a
            # person needs the next morning, so they go to the run record
            # instead of into the manager's context.
            self._report_binding_failure(agent, runtime_adapter, exc)
            self._block(
                agent,
                f"{agent.model_id} runtime could not be started: {_safe_detail(exc)}",
            )
            self._events.put("control")
            return False
        self.managed.bind_thread(
            agent_id=agent.agent_id,
            thread_id=thread_id,
            start_result=start_result,
            tool_handler=handler,
            adapter=runtime_adapter,
        )
        self._refresh_model_identity(agent)
        with self._lock:
            self._bound.add(agent.agent_id)
        self.hooks.record_agent(agent)
        self.hooks.emit(agent, "ready", "agent_bound", f"{agent.role.value} runtime bound")
        self.hooks.lifecycle(
            "agent_spawned",
            agent,
            {
                "role": agent.role.value,
                "model_id": agent.model_id,
                "parent_agent": agent.parent_agent_id,
            },
        )
        return True

    def _record_delegated_agent(self, agent: AgentRecord) -> None:
        """Publish a new child before its handle is returned to the manager."""

        self.hooks.record_agent(agent)
        self.hooks.lifecycle(
            "agent_delegated",
            agent,
            {
                "role": agent.role.value,
                "model_id": agent.model_id,
                "effort": agent.effort,
                "parent_agent": agent.parent_agent_id,
            },
        )

    def _start_ready_agents(self) -> None:
        if self._session_is_closing():
            # Every automatic start goes through this scan: a first turn after
            # binding, a manager woken by a child, a queued peer message
            # delivered at turn end, an agent whose provider backoff expired.
            # One read here covers all of them, and it comes before _bind_agent,
            # so a close also costs no new provider thread.  The agents stay
            # READY with no active turn, which is what the workforce step of the
            # close then cancels.
            return
        session = self._session()
        for agent in list(session.agents.values()):
            if agent.status is not AgentStatus.READY:
                continue
            if agent.agent_id in self._interrupted_agents or agent.agent_id in self._awaiting_message_agents:
                continue
            if agent.agent_id in self._raw_prompt_agents and not self._has_unread_user_text(agent):
                # A wake, a peer message or a provider retry has no user text
                # to send, and anything else would be written into the
                # user's chat as theirs.  Wait for the user instead.  Children
                # stay readable through inspect, and peer mail is offered in
                # the tool responses of the next user turn.
                self._awaiting_message_agents.add(agent.agent_id)
                if not self._has_unread_user_text(agent):
                    continue
                # message_user queues its text before it clears the parked
                # set, so a prompt that landed between the read above and the
                # park is visible now.  Unpark and start it.
                self._awaiting_message_agents.discard(agent.agent_id)
            deadline = self._retry_not_before.get(agent.agent_id)
            if deadline is not None:
                # No sleep here on purpose: this loop is single-threaded and a
                # sleep would stall every other agent in the tree.  The run
                # loop polls on a 200ms timeout, so a held agent is rescanned
                # well inside its own backoff.
                if time.monotonic() < deadline:
                    continue
                self._retry_not_before.pop(agent.agent_id, None)
            if (
                self._reconnect_requested.is_set()
                and self._approval_for_manager(agent.agent_id) is None
            ):
                continue
            with self._lock:
                if agent.agent_id in self._active_turns:
                    continue
                if agent.agent_id in self._native_control_leases:
                    continue
            if not self._chain_clear_to_start(agent):
                # Two writers in one workspace is the thing this prevents, and
                # the hold costs nothing else: the agent stays READY with its
                # wake and its generation untouched.
                continue
            if not self._bind_agent(agent):
                # Binding blocked this child rather than raising. Its manager
                # has a wake carrying the reason; starting a turn on an agent
                # with no runtime thread would raise straight out of this loop.
                continue
            try:
                runtime_ready = self._runtime_can_start_turn(agent)
            except SchedulerError:
                # A bound agent with no runtime thread is vNext contradicting
                # itself, not a provider answering slowly.  Blocking one child
                # would bury the invariant, so it stays fatal.
                raise
            except Exception as exc:
                # The readiness probe is a live provider call carrying its own
                # two-second deadline, and nothing caught it.  Four live
                # sessions died of that: the exception left this loop, left
                # run(), and latched the session as failed, so every later tool
                # call in that chat answered only that the session was gone.
                # Same contract as binding: the manager can route around a
                # blocked child, and it cannot route around a dead session.
                if agent.role is AgentRole.ROOT_MANAGER:
                    raise
                try:
                    probe_adapter = self.managed.adapter_for_agent(agent)
                except Exception:
                    probe_adapter = None
                self._report_binding_failure(
                    agent, probe_adapter, exc, phase="can_start_turn"
                )
                self._block(
                    agent,
                    f"{agent.model_id} readiness probe failed: {_safe_detail(exc)}",
                )
                self._events.put("control")
                continue
            if not runtime_ready:
                # Claude can retain its sole reservation reader after a
                # primary result while native-child lifecycle drains. Keep
                # this agent READY, with its queued wake and generation
                # untouched, until its adapter's local readiness probe says a
                # fresh prompt cannot overlap that reader.
                continue
            self._start_turn(agent)

    def _session_is_closing(self) -> bool:
        """Whether a close of this tree has begun.

        Read under ``session.lock``, the lock ``begin_closing`` takes to set the
        flag, so the answer is never half of that transition.
        """

        session = self._session()
        with session.lock:
            return session.closing

    def _refuse_if_session_closing(self, tool: str) -> None:
        """Refuse a call whose first act would be felt by a provider.

        The authoritative check for a call that creates a record stays inside
        the control plane, under the same lock as the record.  This one is for a
        path that spends something before it gets there.
        """

        if self._session_is_closing():
            raise ProtocolError(SESSION_CLOSING_CODE, session_closing_message(tool))

    def _runtime_can_start_turn(self, agent: AgentRecord) -> bool:
        """Ask an adapter-specific local seam before reusing a bound thread."""

        adapter = self.managed.adapter_for_agent(agent)
        can_start = getattr(adapter, "can_start_turn", None)
        if not callable(can_start):
            return True
        thread_id = self.managed._threads.get(agent.agent_id)
        if not isinstance(thread_id, str) or not thread_id:
            raise SchedulerError("bound agent lacks a runtime thread")
        return can_start(thread_id) is True

    def _primary_waits_for_message(self) -> bool:
        """Whether a persistent conversation is safely idle between turns."""

        session = self._session()
        with self._lock:
            if self.root.agent_id not in self._awaiting_message_agents:
                return False
            if self._active_turns or self._pending_approvals:
                return False
            interrupted = set(self._interrupted_agents)
            awaiting = set(self._awaiting_message_agents)
        for agent in session.agents.values():
            if agent.status is AgentStatus.RUNNING:
                return False
            if (
                agent.status is AgentStatus.READY
                and agent.agent_id not in interrupted
                and agent.agent_id not in awaiting
            ):
                return False
        return True

    def _give_back_unstarted_turn(
        self,
        agent: AgentRecord,
        delivered_before: int,
        children_before: dict[str, Any],
    ) -> None:
        """Undo everything a prompt nobody read had already spent.

        Two callers now: a runtime that refused the turn, and an admission the
        close refused.  Both leave an agent that has to be startable again, so
        its mail, its children ledger and its wakes all go back.
        """

        with self._lock:
            self._starting_agents.discard(agent.agent_id)
            # The override covered the one start the manager asked for, and
            # that start never happened.  The next scan asks the chain question
            # again rather than reusing a decision with nobody asking.
            self._unconfirmed_start_overrides.discard(agent.agent_id)
        agent.delivered_message_count = delivered_before
        if agent.agent_id in self._children_delivered:
            self._children_delivered[agent.agent_id] = children_before
        drained = self._drained_wakes.pop(agent.agent_id, None)
        if drained:
            # Building the prompt drained this manager's wakes. Nobody read
            # that prompt, and a blocker is reported once and never again, so
            # it goes back on the queue with the rest.
            self.managed.control.restore_wakes(agent.agent_id, drained)

    def _start_turn(self, agent: AgentRecord) -> None:
        if self._session_is_closing():
            # The scan above reads the same flag, and a slow runtime binding can
            # span the close: the scan passed its check, then start_thread sat
            # in the provider for as long as it took, and a turn was asked for
            # after the flag was set.  This read is the last one before the
            # adapter is told to work, so it is the one that has to hold.
            return
        if self._skip_start_after_status_change(agent):
            return
        self._turns_started += 1
        if self.max_turns is not None and self._turns_started > self.max_turns:
            raise SchedulerError("scheduler turn ceiling exceeded")
        # Building the prompt is what marks messages delivered, but a runtime
        # that refuses to start the turn never shows the prompt to anybody.  The
        # offset is restored below in that case, so a blocked agent keeps its
        # mail and reads it when it is retried.
        delivered_before = agent.delivered_message_count
        # The children ledger records the same kind of fact at the same moment
        # and has to be undone the same way.  A prompt nobody saw must not leave
        # this manager marked as having been shown a child's full record, or the
        # next prompt would call that record unchanged and the manager would
        # never receive it at all.
        children_before = dict(self._children_delivered.get(agent.agent_id, {}))
        self._drained_wakes.pop(agent.agent_id, None)
        prompt, phase = self._agent_prompt(agent)
        effort = agent.effort or (WORKER_EFFORT if agent.role is AgentRole.WORKER else MANAGER_EFFORT)
        with self._lock:
            self._starting_agents.add(agent.agent_id)
        try:
            turn = self.managed.start_turn(
                agent.agent_id,
                prompt=prompt,
                effort=effort,
                phase=phase,
                turn_timeout=self.turn_timeout,
            )
        except ProtocolError as exc:
            self._give_back_unstarted_turn(agent, delivered_before, children_before)
            if exc.code == SESSION_CLOSING_CODE:
                # Admission was refused because the close had already begun.
                # Nothing was asked of the provider and the agent is still
                # READY with its mail unread, which is the state the workforce
                # step of the close then cancels.  Raising here would leave the
                # run loop and latch the whole session as failed over a close
                # that is working exactly as intended.
                return
            if agent.status in SETTLED_STATUSES:
                # The status check above ran before the prompt was built, and
                # building it renders children and drains wakes, so a manager's
                # replace or cancel can land after that check and before the
                # control plane admits the turn.  The manager was answered
                # success, so the refusal is the same skipped start the check
                # above would have made, and raising it ended the scheduler.
                self._skip_start_after_status_change(agent)
                return
            raise
        except BaseException:
            self._give_back_unstarted_turn(agent, delivered_before, children_before)
            raise
        self._drained_wakes.pop(agent.agent_id, None)
        with self._lock:
            self._active_turns[agent.agent_id] = turn
            self._turn_started_at[agent.agent_id] = self._monotonic()
            # An override covers the one start the manager asked for.  A later
            # turn of the same agent is asked the chain question again.
            self._unconfirmed_start_overrides.discard(agent.agent_id)
            self._active_turns_condition.notify_all()
            paused_before_start = agent.agent_id in self._interrupted_agents
        if paused_before_start:
            # An interrupt can arrive after the READY scan but before this
            # native handle is published.  Stop this just-started turn rather
            # than letting it run until the next user message.
            try:
                self.managed._adapter_for(agent.agent_id).interrupt(turn.runtime)
            except Exception:
                # The pause bit remains set, so an adapter that races us still
                # cannot be auto-started for another turn after it returns.
                pass
            if agent.status in TERMINAL_STATUSES:
                self._interrupted_agents.discard(agent.agent_id)
        with self._lock:
            self._starting_agents.discard(agent.agent_id)
            if agent.status in TERMINAL_STATUSES:
                # A replace landing after the pause read above sets the bit and
                # keeps it, because this agent was still marked as starting.
                # That read was the starter's only one and the agent is terminal
                # now, so nothing will ever consume the bit; left set, the id
                # stayed published under interrupted_agent_ids for the rest of
                # the session.
                self._interrupted_agents.discard(agent.agent_id)
        self.hooks.emit(agent, "running", "turn_started", f"{agent.role.value} turn started")
        self.hooks.lifecycle(
            "turn_started",
            agent,
            {"role": agent.role.value, "phase": phase},
        )

        self._watch_turn(turn)

    def _skip_start_after_status_change(self, agent: AgentRecord) -> bool:
        """Whether this agent's status moved while its runtime was binding.

        The READY check in the scan is old by the time binding returns:
        ``start_thread`` sits in the provider for as long as the provider takes,
        and a manager that replaces or cancels the child inside that window is
        answered success.  Starting the turn anyway asks the control plane for a
        turn on a replaced or cancelled record.  It refuses, and that refusal
        left ``scheduler.run`` and latched the whole session as failed after the
        manager had already been told its command worked.

        So the window is read once more here, the last point before the adapter
        is asked to work.  A start the status has overtaken is an ordinary
        skipped start: it is written into the run log, the thread the start had
        already bound goes back through the release path every settled agent
        uses, and the scan moves on to the next agent.

        BLOCKED is read beside the terminal statuses.  It is not terminal, so
        the record is not finished, but a blocked agent is one its manager has
        been handed a decision about and a turn started under that decision
        would cross it.
        """

        status = agent.status
        if status not in SETTLED_STATUSES:
            return False
        with self._lock:
            # The starter's own bit, cleared here for the callers that set it
            # before reaching this point.
            self._starting_agents.discard(agent.agent_id)
        # The bound thread will never carry a turn now.
        self.queue_terminal_release(agent.agent_id)
        try:
            self.hooks.lifecycle(
                "start_skipped_after_status_change",
                agent,
                {
                    "role": agent.role.value,
                    "status": status.value,
                    "reason": "the status changed while its runtime was binding",
                },
            )
        except Exception:
            # Observing a skipped start must never be what fails the session
            # this skip exists to keep alive.
            pass
        return True

    def _watch_turn(self, turn: ManagedTurn) -> None:
        """Drain one registered turn through the shared completion path."""

        def wait() -> None:
            try:
                result = self.managed.wait_turn(turn, timeout=self.turn_timeout)
                event = _TurnFinished(turn.agent_id, turn, result=dict(result))
            except BaseException as exc:
                event = _TurnFinished(turn.agent_id, turn, error=exc)
            finally:
                with self._lock:
                    self._waiters.discard(threading.current_thread())
            self._events.put(event)

        waiter = threading.Thread(
            target=wait,
            name=f"vnext-turn-{turn.agent_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._waiters.add(waiter)
        waiter.start()

    def _handle_turn_finished(self, event: _TurnFinished) -> None:
        timed_out = event.error is not None and _wait_gave_up(event.error)
        # A wait that ended in an exception is a fact about our transport and
        # not a receipt from the model.  Treat it as unconfirmed unless the
        # provider process behind the turn can be shown to have exited.
        unconfirmed = timed_out or (
            event.error is not None and not self._provider_turn_ended(event.turn)
        )
        stop_fully_confirmed = False
        with self._lock:
            self._active_turns.pop(event.agent_id, None)
            turn_started = self._turn_started_at.pop(event.agent_id, None)
            if unconfirmed and getattr(event.turn, "runtime", None) is not None:
                # The same lock acquisition that takes the turn off the active
                # list puts it on the stopping list.  Registering it a moment
                # later, from the stop itself, left a window in which the turn
                # was on neither list and a replace in that window found an
                # empty chain and started the second writer.  The stop is still
                # sent below, outside this lock, because it is a provider call.
                self._mark_turn_stopping(event.agent_id, event.turn)
            elif not unconfirmed:
                # Either the turn reported its own end or the process behind it
                # is gone.  Both are evidence that nothing of THIS turn is still
                # writing the shared workspace.  Another turn of the same agent
                # keeps its own hold.
                stop_fully_confirmed = self._forget_stopping_turn(
                    event.agent_id, event.turn
                )
            reconnect_yield = event.agent_id in self._reconnect_interrupts
            self._reconnect_interrupts.discard(event.agent_id)
        deferred_blocker = self._deferred_blockers.pop(event.agent_id, None)
        # The turn's own replies name the model that really answered.
        finished_agent = self._session().agents.get(event.agent_id)
        if finished_agent is not None:
            self._refresh_model_identity(finished_agent)
        if stop_fully_confirmed:
            # The receipt arrived after the barrier had already blocked
            # whatever was waiting on this turn, so release it here too.
            self._release_blocks_on_confirmed_stop(event.agent_id)
        agent = self._session().agents[event.agent_id]
        if agent.status in TERMINAL_STATUSES:
            self._interrupted_agents.discard(event.agent_id)
        turn_data = {"role": agent.role.value, "turn_id": event.turn.control_turn_id}
        if event.error is not None:
            if unconfirmed:
                self._stop_unconfirmed_provider_turn(agent, event, timed_out=timed_out)
            reason = self._report_turn_failure(agent, event)
            if timed_out and turn_started is not None and self._monotonic() - turn_started >= self.turn_timeout:
                # The bare transport sentence reads like a fault.  Say that the
                # turn used up its time and what the manager can do about it.
                reason += (
                    f". The turn reached its {format_duration(self.turn_timeout)} limit and was stopped;"
                    " work it wrote stays in the workspace, and retry continues the same session"
                )
            self.hooks.lifecycle(
                "turn_completed",
                agent,
                {**turn_data, "status": "failed"},
            )
            if agent.status not in TERMINAL_STATUSES:
                self._fail_with_evidence(agent, reason)
            if agent.role is AgentRole.ROOT_MANAGER:
                raise SchedulerError("Root Manager runtime turn failed") from event.error
            return
        result = event.result or {}
        runtime_status = str(result.get("status") or "unknown")
        self.hooks.lifecycle(
            "turn_completed",
            agent,
            {**turn_data, "status": runtime_status},
        )
        if runtime_status != "completed":
            if self._cancel_requested.is_set() or bool(self.cancellation.requested):
                return
            if runtime_status == "interrupted" and event.agent_id in self._interrupted_agents:
                if agent.status is AgentStatus.RUNNING:
                    self.managed.control.finish_turn(agent.agent_id)
                if agent.agent_id == self.root.agent_id:
                    # An interrupted primary waits for the next message like
                    # one whose turn ended, so the conversation can go idle.
                    self._awaiting_message_agents.add(agent.agent_id)
                self.hooks.lifecycle("turn_interrupted", agent, turn_data)
                if (
                    deferred_blocker is not None
                    and agent.status is AgentStatus.READY
                    and agent.agent_id != self.root.agent_id
                ):
                    # The failure waited for this turn's end, and this is it:
                    # an interrupted worker rests READY, so block it now or
                    # its parent is never told.
                    self._block_with_evidence(agent, deferred_blocker)
                return
            if runtime_status == "interrupted" and reconnect_yield:
                if deferred_blocker is not None:
                    # The reconnect starts the worker's next turn; its end
                    # applies the failure.
                    self._deferred_blockers.setdefault(event.agent_id, deferred_blocker)
                return
            used = self._provider_retries.get(agent.agent_id, 0)
            if (
                self._provider_hung_up(result)
                and agent.status not in TERMINAL_STATUSES
                and used < PROVIDER_RETRY_LIMIT
            ):
                self._provider_retries[agent.agent_id] = used + 1
                self._retry_not_before[agent.agent_id] = (
                    time.monotonic() + PROVIDER_RETRY_DELAYS[used]
                )
                line = (
                    f"{result.get('terminal_reason')} on turn {agent.turn_count}, "
                    f"retrying (attempt {used + 2} of {PROVIDER_RETRY_LIMIT + 1})"
                )
                if agent.status is AgentStatus.RUNNING:
                    self.managed.record_progress(
                        agent.agent_id,
                        activity="the provider cut the turn off",
                        progress=line,
                        files_touched=[],
                        commands=[],
                        material=False,
                    )
                    self.managed.control.finish_turn(agent.agent_id)
                # The next prompt for an agent past its first turn is built out
                # of drained wakes, so without one the retry would arrive
                # carrying no news at all.  ``request_attention`` is the wake
                # seam for a parent and refuses an agent waking itself, so the
                # smallest public seam that queues a wake for any agent is the
                # one the prompt builder uses to put wakes back.
                self.managed.control.restore_wakes(
                    agent.agent_id,
                    [{
                        "reason": (
                            "the provider cut your last turn off part way; the "
                            "work was not finished, carry on from where you were"
                        ),
                        "source_agent_id": agent.agent_id,
                    }],
                )
                if deferred_blocker is not None:
                    # Carried to the retry turn, whose end applies it.
                    self._deferred_blockers.setdefault(event.agent_id, deferred_blocker)
                self.hooks.emit(agent, "ready", "turn_retried", line)
                self._events.put("control")
                return
            if agent.status not in TERMINAL_STATUSES:
                reason = self._runtime_failure_blocker(
                    runtime_status, result, attempts=used + 1
                )
                if runtime_status == "failed":
                    self._fail_with_evidence(agent, reason)
                else:
                    self._block_with_evidence(agent, reason)
            return
        if (
            result.get("native_task_terminal") is True
            and event.agent_id in self._native_child_delivery_contracts
            and agent.status not in TERMINAL_STATUSES
        ):
            # A built-in provider task can return normally without calling
            # vNext's completion tool. Its observed end is task completion,
            # never evidence that the user's outcome was accepted.
            unfinished = [child_id for child_id in agent.child_ids
                          if self._session().agents[child_id].status not in TERMINAL_STATUSES]
            unread = self._unread_message_count(agent)
            children_blocker = None
            if unfinished:
                named = list(unfinished[:3])
                if len(unfinished) > len(named):
                    named.append(f"and {len(unfinished) - len(named)} more")
                child_label = "child" if len(unfinished) == 1 else "children"
                children_blocker = (
                    f"native task ended with {len(unfinished)} unfinished "
                    f"{child_label}: {', '.join(named)}"
                )
            if deferred_blocker is not None and agent.agent_id != self.root.agent_id:
                # The worker asked for its manager during this turn.  Ending
                # the task normally does not answer that, so completing it
                # here would drop the reason and refuse the manager's reply.
                self._block_with_evidence(agent, "; ".join(
                    part for part in (deferred_blocker, children_blocker) if part
                ))
            elif children_blocker is not None:
                blocker = children_blocker
                if unread > 0:
                    blocker += f", and {unread} unread message(s)"
                self._block_with_evidence(agent, blocker)
            elif unread > 0:
                # Mail on its own was a blocker here, which stranded it: a
                # blocked agent gets no further turn, so the message it was
                # blocked for was the one it could never read. A Worker has no
                # completion tool to refuse, so the guarantee is delivery.
                self._worker_reads_messages(agent)
            else:
                self.managed.complete_agent(agent.agent_id, {
                    "outcome": "completed", "verified": False,
                    "source": "provider-native-task",
                    "summary": str(result.get("summary") or "Native provider task completed"),
                }, require_messages_read=True,
                   progress=self._completion_effect_progress(agent))
                self._hand_over_workspace(agent)
                self.hooks.lifecycle("agent_terminal", agent,
                    {**turn_data, "status": "completed", "source": "provider-native-task"})
            return
        blocker = _hook_failure_blocker(result.get("hook_failures")) or deferred_blocker
        if (
            blocker is not None
            and agent.status is AgentStatus.RUNNING
            and agent.agent_id != self.root.agent_id
        ):
            # Its last tool calls never ran, so the turn ended on work it
            # could not do.  Blocking wakes the parent with the reason.
            self._block_with_evidence(agent, blocker)
            return
        if agent.status is AgentStatus.RUNNING:
            self.managed.control.finish_turn(agent.agent_id)
            # A returned native turn is not an implicit completion.  Pause it
            # pending new mail unless this was the scheduler's short approval
            # rendezvous, whose only purpose is to resume the interrupted work.
            if (
                event.turn.phase != "approval-review"
                and self._unread_message_count(agent) == 0
            ):
                self._awaiting_message_agents.add(agent.agent_id)
            self.hooks.emit(
                agent,
                "ready",
                "agent_turn_finished",
                "Agent turn ended without yielding or completing",
            )

    def observe_unsolicited_turn(self, agent_id: str, payload: Mapping[str, Any]) -> None:
        """Hand the end of a provider-started turn to the scheduler loop.

        Called from the telemetry thread that drains native events, so the
        decision itself waits for the loop, where every other status change
        of an agent is made.
        """

        if payload.get("status") == "reader-ended":
            # The bridge stopped reading the session between turns, so no
            # hook or approval of it will be answered again.
            error = str(payload.get("error") or "stream ended")
            self._events.put(_UnsolicitedTurnEnded(agent_id, {}, blocker=(
                f"the Claude session's message stream stopped between turns ({error}); "
                "nothing answers its hooks or approvals any more"
            )))
            return
        if payload.get("status") != "completed":
            return
        failures = payload.get("hook_failures")
        if _hook_failure_blocker(failures) is None:
            return
        self._events.put(_UnsolicitedTurnEnded(agent_id, dict(failures)))  # type: ignore[arg-type]

    def _handle_unsolicited_turn_ended(self, event: _UnsolicitedTurnEnded) -> None:
        blocker = event.blocker or _hook_failure_blocker(event.hook_failures)
        agent = self._session().agents.get(event.agent_id)
        if blocker is None or agent is None or agent.agent_id == self.root.agent_id:
            return
        with self._lock:
            active = event.agent_id in self._active_turns
        # Only an agent at rest is blocked here.  A vNext turn that started
        # since owns the agent's status, so the blocker waits for its end.
        if active or agent.status is AgentStatus.RUNNING:
            self._deferred_blockers.setdefault(event.agent_id, blocker)
            return
        if agent.status is not AgentStatus.READY:
            return
        self._block_unless_cancelled(agent, blocker)

    def _handle_block_reported(self, event: _BlockReported) -> None:
        agent = self._session().agents.get(event.agent_id)
        if agent is None or agent.agent_id == self.root.agent_id:
            return
        with self._lock:
            active = event.agent_id in self._active_turns
        if active or agent.status is AgentStatus.RUNNING:
            # The turn's end applies it.  A second report in the same turn
            # adds its reason to the first.
            earlier = self._deferred_blockers.get(event.agent_id)
            blocker = event.blocker
            if earlier is not None:
                blocker = earlier if blocker in earlier else f"{earlier}; {blocker}"
            self._deferred_blockers[event.agent_id] = blocker
            return
        if agent.status is not AgentStatus.READY:
            return
        # The turn ended before the loop saw the report, so nothing else
        # will apply it: block now so the parent is woken with the reason.
        self._block_unless_cancelled(agent, event.blocker)

    def _block_unless_cancelled(self, agent: AgentRecord, blocker: str) -> None:
        try:
            self._block_with_evidence(agent, blocker)
        except ProtocolError:
            # A cancel from the handler thread can land after the READY check.
            current = self._session().agents.get(agent.agent_id)
            if current is None or current.status not in TERMINAL_STATUSES:
                raise

    def _manager_handler(
        self,
        agent_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        context: ToolCallContext | None,
    ) -> dict[str, Any]:
        try:
            if context is not None and context.turn_id:
                self.adopt_native_turn(
                    agent_id=agent_id,
                    provider=self._native_provider_for_agent(agent_id),
                    thread_id=context.thread_id,
                    turn_id=context.turn_id,
                    cursor=context.cursor,
                )
            if "acknowledge_messages_through" in arguments:
                self._acknowledge_native_messages(agent_id, tool, arguments, context)
            response = self._apply_manager_tool(agent_id, tool, arguments)
            external_caller = self._agent_is_external(agent_id)
            scoped_native_caller = (
                context is not None
                and bool(context.turn_id)
                and (self._native_message_delivery(agent_id) == "available"
                     or agent_id in self._raw_prompt_agents)
            )
            if response.success and (external_caller or scoped_native_caller):
                session = self._session()
                with session.lock:
                    agent = session.agents[agent_id]
                    messages = self._new_messages(agent, mark_delivered=False)
                    cursor = agent.delivered_message_count + len(messages)
                    self._native_messages_offered[agent_id] = cursor
                if messages:
                    return _dynamic_result(True, {
                        **response.value, "messages": messages,
                        "message_delivery": (
                            "external-tool-response" if external_caller
                            else "native-tool-response"
                        ),
                        "message_cursor": cursor,
                        "message_acknowledgement": (
                            "After processing these messages, call inspect for self with "
                            "acknowledge_messages_through equal to message_cursor. "
                            "Reading alone does not acknowledge them."
                        ),
                    })
            return response
        except ProtocolError as exc:
            return _dynamic_result(False, {"error": str(exc), "error_code": exc.code})
        except (ManagedSessionError, RequiredToolEffectError, ValueError) as exc:
            return _dynamic_result(False, {"error": str(exc), "error_code": "invalid-request"})
        except SchedulerError as exc:
            return _dynamic_result(False, {"error": str(exc), "error_code": "scheduler-error"})

    def _acknowledge_native_messages(
        self, agent_id: str, tool: str, arguments: Mapping[str, Any], context: ToolCallContext | None,
    ) -> None:
        """Acknowledge only messages already offered to this authorized caller."""
        scoped_native_caller = (
            context is not None
            and bool(context.turn_id)
            and (self._native_message_delivery(agent_id) == "available"
                 or agent_id in self._raw_prompt_agents)
        )
        if (tool != "inspect" or arguments.get("agent_id") not in (None, "", "self", agent_id)
                or not (self._agent_is_external(agent_id) or scoped_native_caller)):
            raise ValueError(
                "message acknowledgement requires an authenticated external or scoped native self inspect"
            )
        cursor = arguments["acknowledge_messages_through"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("message acknowledgement cursor must be a nonnegative integer")
        session = self._session()
        with session.lock:
            agent = session.agents[agent_id]
            offered = self._native_messages_offered.get(agent_id, 0)
            if cursor > offered:
                raise ValueError("cannot acknowledge messages not yet offered to this caller")
            # Repeated or older acknowledgements cannot eat a later arrival.
            agent.delivered_message_count = max(agent.delivered_message_count, cursor)

    def _agent_is_external(self, agent_id: str) -> bool:
        """Whether this agent is a client vNext does not run.

        Deliberately tolerant where ``_native_provider_for_agent`` is strict:
        that one identifies a provider in order to adopt a native turn and must
        fail rather than guess, while this only asks a yes/no question about
        the caller and an unidentified provider is simply not external.
        """

        card = self.managed.control.registry.cards.get(
            self._session().agents[agent_id].model_id
        )
        return card is not None and str(getattr(card, "provider", "")) == "external"

    def _native_provider_for_agent(self, agent_id: str) -> str:
        card = self.managed.control.registry.cards[self._session().agents[agent_id].model_id]
        if isinstance(card.provider, str) and card.provider:
            return card.provider
        with self.managed._lock:
            provider = self.managed._identity_attestations.get(agent_id, {}).get("provider")
        if isinstance(provider, str) and provider:
            return provider
        raise ManagedSessionError("native turn provider is not identified")

    def _native_message_delivery(self, agent_id: str) -> str | None:
        """Return inbox delivery for an adopted child or leased native terminal."""

        with self._lock:
            contract = self._native_child_delivery_contracts.get(agent_id)
            if contract is None:
                # A terminal owns continuation of this bound thread. Its scoped
                # tool callbacks must read mail without a scheduler-started turn.
                return "available" if agent_id in self._native_control_leases else None
        return "available" if contract.get("context_messages") == "available" else "unavailable"

    def _note_manager_tool(self, agent_id: str, tool: str) -> None:
        """Record which tool a manager called, against the turn it called it in.

        P2 of the Phase 9 gates asks what a manager did after its last delegate
        and before it yielded. Nothing else records it: a manager reasoning and
        inspecting produces no lifecycle transition at all.
        """

        with self._lock:
            turn = self._active_turns.get(agent_id)
        if turn is None:
            return
        try:
            self.managed.record_turn_tool(turn, tool)
        except Exception:
            # Instrumentation never fails a run.
            pass

    def _active_control_turn_id(self, agent_id: str) -> str | None:
        """Return the live control-turn identity before a tool ends that turn."""

        with self._lock:
            turn = self._active_turns.get(agent_id)
        return turn.control_turn_id if turn is not None else None

    def _delegate_preflight_warnings(self, arguments: Mapping[str, Any]) -> list[str]:
        """Offer route/turn limits without changing whether delegation succeeds."""
        objective = str(arguments.get("objective", ""))
        card = self.managed.control.registry.cards.get(str(arguments.get("model_id", "")))
        provider = card.provider if card is not None else None
        warnings = []
        # Only the Claude SDK routes cut a turn at a fixed wall-clock deadline.
        # The app-server routes (codex, commandcode) renew the same budget on
        # every turn event, so a long but active turn is not cut there.
        if provider in {"claude", "zai"}:
            # "1h30" and "1h30m" are one 90-minute budget. A capital M is a
            # count ("100M rows"), so the bare minute unit is lowercase only.
            budgets = re.finditer(
                r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:(?i:hours?|hrs?|h)(?:(\d+)(?:(?i:minutes?|mins?)|m)?)?"
                r"|((?i:minutes?|mins?)|m))(?!\w)",
                objective,
            )
            minutes = []
            for match in budgets:
                if match[3] is not None:
                    minutes.append(float(match[1]))
                else:
                    minutes.append(float(match[1]) * 60 + float(match[2] or 0))
            if any(value * 60 > self.turn_timeout for value in minutes):
                warnings.append(
                    f"The objective states a time budget above the {self.turn_timeout:g} s"
                    f" ({self.turn_timeout / 60:g} min) wall-clock cap a {provider} route puts on one turn;"
                    " split longer work across turns or children."
                )
        if re.search(
            r"\b(browser|playwright|chrome|chromium|chromedriver|puppeteer)\b"
            r"|\bbrowse\s+the\s+web\b"
            r"|\bscreenshot\s+of\s+(?:the\s+)?(?:page|site)\b",
            objective, re.IGNORECASE,
        ):
            route = f" ({provider} route)" if provider else ""
            warnings.append(
                f"vNext provides no browser of its own{route}; a worker has one only if the user's own"
                " config (Claude settings or ~/.codex) enables a browser MCP server,"
                " and Codex workers run with network restricted."
            )
        return warnings

    def _apply_manager_tool(
        self,
        agent_id: str,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        session = self._session()
        agent = session.agents[agent_id]
        # Check before validation or mutation, including aliases that are no
        # longer advertised by the public catalog and stored retry targets.
        from .vnext_session_runtime import CLAUDE_SUBSCRIPTION_WORKERS, require_worker_provider
        if not CLAUDE_SUBSCRIPTION_WORKERS and tool in {"delegate", "replace", "retry", "send_message"}:
            model_id = str(arguments.get("model_id") or "")
            if tool in {"retry", "send_message"}:
                target = session.agents.get(str(arguments.get("agent_id") or ""))
                model_id = target.model_id if target is not None else ""
            card = self.managed.control.registry.cards.get(model_id)
            provider = card.provider if card is not None else (
                "claude" if model_id in {"sonnet", "opus", "fable"} else ""
            )
            require_worker_provider(provider)
        self._note_manager_tool(agent_id, tool)
        # Tool schemas are the public contract for every manager entry path.
        # Refuse undeclared top-level keys before any handler changes the tree.
        schema = next(
            (item["inputSchema"] for item in self.manager_tools() if item["name"] == tool),
            None,
        )
        if schema is not None and schema.get("additionalProperties") is False:
            accepted = schema["properties"]
            unknown = sorted(key for key in arguments if key not in accepted)
            if unknown:
                hint = (
                    " The wait ends when a selected child changes materially; "
                    "there is no timeout to set."
                    if tool == "await_children" else ""
                )
                raise ValueError(
                    f"{tool} does not take {', '.join(unknown)}; "
                    f"accepted keys: {', '.join(sorted(accepted))}.{hint}"
                )
        if tool == "delegate":
            role = self._delegate_role(arguments.get("role"))
            raw_contract = arguments.get("required_tool_effect")
            evidence_request: RequiredToolEffect | None = None
            if raw_contract is not None:
                # Validate before changing the tree. A malformed optional
                # evidence request must not leave behind a child from a failed
                # delegate call.
                evidence_request = validate_required_tool_effect(raw_contract)
            try:
                warnings = self._delegate_preflight_warnings(arguments)
            except Exception:
                # The warnings are advice; a fault in them must never turn a
                # valid delegate into a failed one.
                warnings = []
            child = self.managed.spawn_from_manager(
                requester_id=agent_id,
                role=role,
                arguments=arguments,
            )
            if evidence_request is not None:
                self._evidence_requests[child.agent_id] = evidence_request
            self._record_delegated_agent(child)
            self._events.put("control")
            return _dynamic_result(
                True,
                {
                    **({"warnings": warnings} if warnings else {}),
                    "agent_id": child.agent_id,
                    "role": child.role.value,
                    "model_id": child.model_id,
                    # The alias above is a selector a manager may pass back
                    # in.  This is the exact id it resolves to, read from the
                    # provider's own table once any worker has connected.
                    "model_exact": self._cached_exact_model(child.model_id),
                    "evidence_request_recorded": raw_contract is not None,
                    # Where this child works.  A private or worktree child
                    # reports its directory relative to the session workspace.
                    # A shared child works in the session workspace itself,
                    # and used to be answered with an empty string, which a
                    # manager cannot open, join or name in a packet.  It gets
                    # the absolute path instead, so every delegate answer
                    # names a directory.
                    "workspace_path": (
                        self.managed.relative_workspace_for(child.agent_id)
                        or str(self.managed.workspace)
                    ),
                },
            )
        if tool == "await_children":
            raw_ids = arguments.get("agent_ids")
            if not isinstance(raw_ids, list) or not raw_ids or not all(
                isinstance(value, str) and value for value in raw_ids
            ):
                raise ValueError("agent_ids must be a non-empty array")
            # A child named twice is one child: the answer lists it once.
            raw_ids = list(dict.fromkeys(raw_ids))
            if self._agent_is_external(agent_id) or agent_id in self._raw_prompt_agents:
                # A client vNext does not run cannot be parked and woken: it is
                # holding this call open and there is no later turn to deliver
                # the wake into.  Wait for real instead.  A resumed native chat
                # is the same case: its only turns are the user's own, so a
                # wake turn would never come.
                return self.await_children_blocking(
                    agent_id, raw_ids, timeout=self.external_await_budget
                )
            active_ids = []
            for child_id in raw_ids:
                child = session.agents.get(child_id)
                if child is None or child.parent_agent_id != agent_id:
                    raise ProtocolError(
                        "not-direct-child",
                        "await targets must be canonical vNext direct-child agent_ids. "
                        'Inspect self with agent_id "self" once and use children[].agent_id; provider thread IDs are separate.',
                    )
                if child.status not in TERMINAL_STATUSES:
                    active_ids.append(child_id)
            if not active_ids:
                return _dynamic_result(True, {
                    "awaiting": [],
                    "settled": True,
                    "children": [self._coordination_child(session.agents[value]) for value in raw_ids],
                    "note": "Selected children are terminal. Continue your work or complete; no wait is needed.",
                })
            # The reason comes from the control plane rather than from the
            # manager's status, because the status is identical whichever way a
            # park is declined. Inferring it meant a manager waiting on one
            # blocked child and one running one was told everything had stopped
            # and not to wait again, when waiting for the running sibling was
            # exactly right -- and once anything reports material progress
            # mid-turn, the same guess would tell a manager to cancel healthy
            # children.
            outcome = self.managed.await_agents(agent_id, active_ids)
            if outcome == "parked":
                return _dynamic_result(True, {"awaiting": list(active_ids)})
            if outcome == "pending-wake":
                return _dynamic_result(
                    True,
                    {
                        "awaiting": [],
                        "note": (
                            "One of these children moved while this turn was "
                            "still open, so there was no need to wait. Your "
                            "next prompt carries what changed."
                        ),
                    },
                )
            # The park was refused because nothing being waited on can still
            # move, and the manager has to be told. Reporting success here was
            # its own bug: the manager believed it had yielded, took another
            # turn, awaited the same stopped child, and did that until the turn
            # ceiling -- roughly two hundred and fifty billed turns to reach
            # the same place a deadlock reached in six. A refusal it can read
            # is the whole fix, and it is smaller than the spin it prevents.
            stopped = [
                {
                    "agent_id": child_id,
                    "status": session.agents[child_id].status.value,
                    "blocker": session.agents[child_id].blocker,
                }
                for child_id in active_ids
                if session.agents[child_id].status in SETTLED_STATUSES
            ]
            return _dynamic_result(
                False,
                {
                    "error_code": "nothing-to-wait-for",
                    "error": (
                        "Every child you asked to wait for has already stopped, "
                        "so waiting cannot end. Each one below is waiting on a "
                        "decision from you: retry it on a revised contract, "
                        "replace it with another model, cancel_agent it, or "
                        "delegate the work afresh. Do not call await_children "
                        "again on the same children without changing something."
                    ),
                    "children_needing_a_decision": stopped,
                },
            )
        if tool == "inspect":
            target_id = str(arguments.get("agent_id") or "")
            if target_id in ("", "self"):
                target_id = agent_id
            view = self._view_with_usage_final(self.managed.control.inspect_agent(
                agent_id,
                target_id,
                deep=strict_bool_argument(arguments, "deep"),
            ))
            with session.lock:
                children = [self._coordination_child(session.agents[value]) for value in view["child_ids"]]
            # A child reported as blocked is often blocked on this manager, and
            # the reason was previously readable only in the raw run log.
            return _dynamic_result(True, {
                "agent": view,
                "children": children,
                "pending_approvals": [
                    self._approval_receipt(value)
                    for value in self._approvals_for_manager(target_id)
                ],
            })
        if tool == "steer":
            target_id = required_agent_id(arguments)
            message = str(arguments.get("message") or "").strip()
            if not message:
                raise ValueError("steering message is required")
            if target_id == agent_id:
                # A message to oneself lands in one's own inbox as unread and
                # then refuses one's own completion on this turn.
                raise ValueError(
                    "an agent cannot steer itself: steer a child by its agent_id"
                )
            with self._lock:
                active = self._active_turns.get(target_id)
            # Queue first, so the message is durable whether or not the target's
            # runtime accepts an in-place steer.  Permission and liveness checks
            # live in message_agent and still refuse before anything is queued.
            self.managed.control.message_agent(agent_id, target_id, message, kind="steer")
            self._interrupted_agents.discard(target_id)
            self._awaiting_message_agents.discard(target_id)
            delivery = "queued-for-next-turn"
            if active is not None and active.agent_id == target_id:
                try:
                    # Codex-family workers receive this.  The Claude bridge
                    # answers every steer with native-steer-not-supported, so
                    # Claude and Z.ai workers read the clock from the
                    # turn-start prompt on the next turn instead -- the same
                    # turn in which they read this queued steer.
                    self.managed._adapter_for(target_id).steer(
                        active.runtime, message + "\n" + self._steer_clock_line(target_id)
                    )
                except Exception:
                    # The message is already queued and the target is already
                    # woken, so a runtime that refuses rather than simulates
                    # in-place steering still receives it on its next turn.
                    delivery = "queued-for-next-turn"
                else:
                    self.managed.mark_check("live_steering", True)
                    delivery = "delivered-into-active-turn"
            self._events.put("control")
            return _dynamic_result(
                True,
                {"target": target_id, "status": "steered", "delivery": delivery},
            )
        if tool == "send_message":
            target_id = required_agent_id(arguments)
            message = str(arguments.get("message") or "").strip()
            if not message:
                raise ValueError("message is required")
            if target_id == agent_id:
                raise ValueError(
                    "an agent cannot send a message to itself: address another "
                    "agent by its agent_id"
                )
            return _dynamic_result(True, self.send_message(
                sender_id=agent_id, target_id=target_id, text=message
            ))
        if tool == "interrupt_agent":
            return _dynamic_result(True, self.interrupt_agent(required_agent_id(arguments)))
        if tool == "retry":
            retried_id = required_agent_id(arguments)
            override = strict_bool_argument(
                arguments, "start_despite_unconfirmed_stop"
            )
            # The override has to be asked for by name.  Retry is what a manager
            # reaches for after any failure, so letting an ordinary one clear
            # this particular block made the habit itself the decision to put a
            # second writer in one workspace.  Nothing is mutated here: the
            # agent stays blocked, the hold stays, and the answer says which
            # argument starts it.
            if not override:
                predecessor_id = self._unconfirmed_stop_hold(retried_id)
                if predecessor_id is not None and self._held_turns_have_ended(predecessor_id):
                    # The old process has exited since the barrier gave up, so
                    # the hold has no reason left.  This agent's own block is
                    # dropped quietly because the retry below restarts it; any
                    # other agent waiting on the same turn is freed as usual.
                    with self._lock:
                        self._unconfirmed_blocks.pop(retried_id, None)
                        self._replacement_barriers.pop(retried_id, None)
                    self._release_blocks_on_confirmed_stop(predecessor_id)
                    predecessor_id = None
                if predecessor_id is not None:
                    return _dynamic_result(False, {
                        "error_code": "unconfirmed-stop",
                        "error": (
                            f"{retried_id} is held: the previous turn of "
                            f"{predecessor_id} has not confirmed it stopped. Retry "
                            "with start_despite_unconfirmed_stop true to start it "
                            "anyway, or cancel it."
                        ),
                    })
            child = self.managed.retry_agent(
                requester_id=agent_id,
                agent_id=retried_id,
                revised_task_contract=dict(arguments.get("task_contract") or {}),
            )
            # The override is read first; otherwise this attempt is itself held
            # behind the attempt it retries, whose provider turn may not have
            # stopped when our own wait on it gave up.  Without a block to
            # clear the flag changes nothing, which is why a manager may send
            # it without reading the agent first.
            # The attempt is held by the starter if the chain it came from may
            # still be writing; nothing is re-tested here.  An override with no
            # hold behind it changes nothing, which is why a manager may send it
            # without reading the agent first.
            if override:
                self._start_despite_unconfirmed_stop(child)
            self.hooks.record_agent(child)
            self._events.put("control")
            return _dynamic_result(True, {"agent_id": child.agent_id, "status": "ready"})
        if tool == "replace":
            replaced_id = required_agent_id(arguments)
            replaced = session.agents.get(replaced_id)
            model_id = str(arguments.get("model_id") or "")
            replacement_objective = arguments.get("objective")
            if replacement_objective is not None and (
                not isinstance(replacement_objective, str) or not replacement_objective
            ):
                raise ManagedSessionError("manager child objective is required")
            if replaced is None:
                raise ProtocolError("invalid-handle", "unknown agent handle")
            if replaced.parent_agent_id != agent_id:
                raise ProtocolError("not-replaceable-agent", "an agent may replace only its direct child")
            # A FAILED attempt passes: a provider refusal of the model is cured
            # by another model.  control.replace_agent applies the same rule, so
            # the MCP tool and a direct call agree.
            if replaced.status in TERMINAL_STATUSES and replaced.status not in REPLACEABLE_STATUSES:
                raise ProtocolError("terminal-agent", f"agent is terminal: {replaced.status.value}")
            self.managed.control.registry.validate_selection(
                preset_id=session.preset_id, model_id=model_id, role=replaced.role,
            )
            if replaced.effort:
                # An agent with no effort of its own starts at the default,
                # which every provider accepts.
                self.managed.validate_effort_for_model(model_id, replaced.effort)
            unfinished = [
                child_id for child_id in replaced.child_ids
                if session.agents[child_id].status not in TERMINAL_STATUSES
            ]
            if unfinished:
                raise ProtocolError(
                    "active-children",
                    f"agent still has active children: {unfinished}",
                )
            # control.replace_agent refuses a closing session under the session
            # lock, and that stays the authority.  By then this had already
            # interrupted the target's provider turn, so a replacement that was
            # never going to happen stopped a turn the manager still owned.  The
            # early read below costs one lock and removes that side effect.
            #
            # A close landing between this read and the interrupt still
            # interrupts, and nothing is lost by it: the workforce step of the
            # same close interrupts every running turn a moment later anyway.
            # So the window is narrower than the close itself and has no effect
            # of its own.
            self._refuse_if_session_closing("replace")
            # A RUNNING control turn may still be inside start_turn, before its
            # native handle reaches _active_turns. The starter consumes this bit
            # after publication; _interrupt_turn covers an already published turn.
            was_running = replaced.status is AgentStatus.RUNNING
            set_here = False
            if was_running:
                with self._lock:
                    set_here = replaced_id not in self._interrupted_agents
                    self._interrupted_agents.add(replaced_id)
            try:
                self._interrupt_turn(replaced_id)
            except Exception as exc:
                if set_here:
                    self._interrupted_agents.discard(replaced_id)
                said = _safe_detail(exc)
                reason = f"{type(exc).__name__}: {said}" if said else type(exc).__name__
                return _dynamic_result(False, {
                    "error_code": "interrupt-failed",
                    "error": f"could not interrupt agent {replaced_id}: {reason}",
                })
            try:
                replacement = self.managed.control.replace_agent(
                    requester_id=agent_id,
                    agent_id=replaced_id,
                    model_id=model_id,
                    objective=replacement_objective,
                    revised_task_contract=dict(arguments.get("task_contract") or {}),
                )
            except Exception:
                # Only a bit this call set is this call's to take back, and only
                # while the agent is still live.  A concurrent replace that won
                # may rely on the same bit to stop a turn inside start_turn,
                # and the success path below clears it once nothing can.
                if set_here and replaced.status not in TERMINAL_STATUSES:
                    self._interrupted_agents.discard(replaced_id)
                raise
            # The interrupt above is a request the provider answers in its own
            # time.  Until it does, the predecessor is still executing in the
            # workspace the replacement is about to edit.  replace_agent wrote
            # the lineage down; the starter reads it and holds the replacement,
            # rather than this call holding the manager.
            with self._lock:
                starting = replaced_id in self._starting_agents
            if was_running and not starting and replaced.status in TERMINAL_STATUSES:
                # The bit exists only so a turn still inside start_turn cannot
                # slip past the replacement.  The old agent is terminal now, so
                # nothing will ever consume it, and it was being published under
                # interrupted_agent_ids for the rest of the session.
                self._interrupted_agents.discard(replaced_id)
            if replaced is not None:
                self.hooks.lifecycle(
                    "agent_terminal",
                    replaced,
                    {"role": replaced.role.value, "status": "replaced"},
                )
            self._record_delegated_agent(replacement)
            self._events.put("control")
            return _dynamic_result(True, {"agent_id": replacement.agent_id, "status": "ready"})
        if tool == "cancel_agent":
            target_id = required_agent_id(arguments)
            if target_id == session.root_agent_id:
                raise ValueError(
                    "the session root cannot cancel itself: to stop its work, "
                    "cancel each child by its agent_id; to end the session, "
                    "call complete_session"
                )
            outcome = self._cancel_agent_for(agent_id, target_id)
            # A stop the provider refused leaves that agent's turn running
            # under a record that now reads cancelled.  This answered success
            # anyway, so a manager read "cancelled" and moved on while the work
            # carried on underneath it.  The subtree is still cancelled -- the
            # call fails so the manager sees that something did not stop, and
            # `uninterrupted` says which agents.
            return _dynamic_result(not outcome.get("uninterrupted"), outcome)
        if tool == "resolve_approval":
            approval_id = str(arguments.get("approval_id") or "")
            selected = self._approval_decision(arguments.get("decision"))
            self._resolve_pending_approval(
                approval_id=approval_id,
                decision=selected,
                rationale=str(arguments.get("rationale") or ""),
                resolver=agent_id,
                expected_manager_id=agent_id,
            )
            return _dynamic_result(True, {"approval_id": approval_id, "decision": selected.value})
        if tool == "complete_branch":
            if agent.role is AgentRole.ROOT_MANAGER:
                raise ValueError("the root ends the session with complete_session")
            require_completion_fields(arguments, ("outcome", "verified", "evidence"))
            self._require_messages_read(agent)
            # The outcome row is written as the agent turns terminal, often
            # before its turn ends, so it must carry the model that answered.
            self._refresh_model_identity(agent)
            turn_id = self._active_control_turn_id(agent_id)
            self.managed.complete_agent(
                agent_id,
                {
                    "outcome": str(arguments.get("outcome") or "unknown"),
                    "verified": strict_bool_argument(arguments, "verified"),
                    "evidence": evidence_argument(arguments),
                },
                progress=self._completion_effect_progress(agent),
            )
            workspace = self._hand_over_workspace(agent)
            self.hooks.lifecycle(
                "agent_terminal",
                agent,
                {"role": agent.role.value, "status": "completed", "turn_id": turn_id},
            )
            return _dynamic_result(True, {"status": "completed", **({"workspace": workspace} if workspace else {})})
        if tool == "complete_session":
            if agent.role is not AgentRole.ROOT_MANAGER:
                raise ProtocolError("not-root", "only the Root Manager completes the session")
            require_completion_fields(arguments, ("decision", "summary", "criteria"))
            if arguments["decision"] not in ("accepted", "rejected", "needs-review"):
                raise ValueError("decision must be one of accepted, rejected, needs-review")
            self._require_messages_read(agent)
            turn_id = self._active_control_turn_id(agent_id)
            self.managed.complete_root(
                agent_id,
                {
                    "decision": str(arguments.get("decision") or "unknown"),
                    "summary": str(arguments.get("summary") or ""),
                    "criteria": criteria_argument(arguments),
                },
                progress=self._completion_effect_progress(agent),
            )
            self.hooks.lifecycle(
                "agent_terminal",
                agent,
                {"role": agent.role.value, "status": "completed", "turn_id": turn_id},
            )
            return _dynamic_result(True, {"status": "completed"})
        if tool == "complete_agent":
            if agent.role is AgentRole.ROOT_MANAGER:
                raise ValueError("the root ends the session with complete_session")
            require_completion_fields(arguments, ("outcome", "verified", "evidence"))
            self._require_messages_read(agent)
            # The outcome row is written as the agent turns terminal, often
            # before its turn ends, so it must carry the model that answered.
            self._refresh_model_identity(agent)
            turn_id = self._active_control_turn_id(agent_id)
            self.managed.complete_agent(
                agent_id,
                {
                    "outcome": str(arguments.get("outcome") or "completed"),
                    "verified": strict_bool_argument(arguments, "verified"),
                    "evidence": evidence_argument(arguments),
                },
                require_messages_read=True,
                progress=self._completion_effect_progress(agent),
            )
            workspace = self._hand_over_workspace(agent)
            self.hooks.lifecycle(
                "agent_terminal",
                agent,
                {"role": agent.role.value, "status": "completed", "turn_id": turn_id},
            )
            return _dynamic_result(True, {"status": "completed", **({"workspace": workspace} if workspace else {})})
        if tool == "report_blocked":
            if agent.role is AgentRole.ROOT_MANAGER:
                raise ProtocolError(
                    "primary-reports-to-user",
                    "the primary has no manager to wake: ask your user directly",
                )
            reason = str(arguments.get("reason") or "").strip()
            if not reason:
                raise ValueError("reason is required")
            # The status stays RUNNING until the turn ends.  Blocking here would
            # release the runtime under a turn that is still answering, so the
            # reason rides the deferred-blocker path, which the turn's end
            # applies and an interrupt, reconnect or provider retry carries on.
            # This runs on a handler thread, so the loop records it: the event
            # is queued before this reply reaches the provider, which puts it
            # ahead of the turn's own end in the queue.
            self._events.put(_BlockReported(agent_id, (
                f"reported by the worker: {reason}. Answer with send_message, "
                "then retry it, to resume it on the same session"
            )))
            return _dynamic_result(True, {
                "status": "block-recorded",
                "message": (
                    "The block is recorded. End your turn now; your manager is told "
                    "the reason and will answer you with a message in your next turn."
                ),
            })
        raise ProtocolError("unknown-tool", f"unknown manager tool: {tool}")

    def _hand_over_workspace(self, agent: AgentRecord) -> dict[str, Any]:
        """Settle a completed child's checkout and put the answer on its result.

        The tools promise a manager the path and the branch when a worktree
        child finishes.  Nothing called the settling step after the session
        backend was reworked, so a manager heard nothing until the session
        closed and an untouched checkout sat on disk all that time.  The
        answer rides on the child's result, which is what ``inspect`` and
        ``await_children`` read back.
        """

        workspace = self._settle_workspace(agent)
        if workspace and isinstance(agent.result, Mapping):
            agent.result = {**agent.result, "workspace": workspace}
        return workspace

    def _settle_workspace(self, agent: AgentRecord) -> dict[str, Any]:
        """Close out a child's own workspace and say what became of it.

        Only a worktree has anything to settle.  A checkout the child left
        untouched is removed with its branch, because keeping it fills the disk
        with empty copies of the project; one it changed is kept, and its path
        and branch are handed to the manager, which is where Claude Code and pi
        both stop.  Nothing here merges anything.
        """

        if agent.workspace_scope != "worktree":
            return {}
        try:
            return self.managed.worktree_outcome(agent.agent_id)
        except (ManagedSessionError, OSError):
            # A checkout that cannot be settled is a tidiness problem.  Failing
            # the child's completion over it would throw away work that is fine.
            return {}

    @staticmethod
    def _provider_hung_up(result: Mapping[str, Any]) -> bool:
        """Say whether the provider cut the turn off part way.

        Only a cut-off stream or a cut-off tool call earns another attempt.
        A rate limit is a wait. It already has its own sentence in
        :meth:`_runtime_failure_blocker`, and going straight back to the
        provider spends the allowance the wait was asking us to save.
        """

        reason = result.get("terminal_reason")
        if reason not in ("aborted_streaming", "aborted_tools"):
            return False
        return result.get("rate_limited") is not True

    @staticmethod
    def _credential_was_rejected(result: Mapping[str, Any]) -> bool:
        """Say whether the provider refused the login rather than the work.

        A 401 or a 403 reproduces on every retry until the key is corrected,
        so it earns its own advice: the sentence that told a manager the
        packet "may still be sound and worth retrying" was pointing it at
        wasted turns.
        """

        status = result.get("api_error_status")
        if isinstance(status, int) and not isinstance(status, bool) and status in (401, 403):
            return True
        # A provider frame is JSON, so any of these can arrive as a list or
        # an object; compared one by one they can never raise.
        return any(
            result.get(key) == "authentication_failed"
            for key in ("terminal_reason", "subtype", "code")
        )

    @staticmethod
    def _runtime_failure_blocker(
        runtime_status: str, result: Mapping[str, Any], *, attempts: int = 1
    ) -> str:
        """Say why the turn ended, in the provider's own terms.

        An Opus worker died 43 seconds into a read-only packet on 2026-09-18 and
        cost $0.80.  Its manager was told ``runtime turn ended as failed`` and
        nothing else, which is the same sentence a model that argued itself into
        a corner produces, so there was no way to tell a provider fault from the
        worker's own doing without opening the raw run log.

        The provider already said which it was.  The turn result carries the
        SDK's ``subtype``, its ``terminal_reason`` and the HTTP status, and this
        method was throwing all three away.  They go in the blocker now, because
        the blocker is the one line a manager reads, and the difference decides
        whether retrying the same packet is sensible or wasteful.

        The line stays one sentence.  Everything else stays in the run record.
        """

        blocker = f"runtime turn ended as {runtime_status}"
        reason = result.get("terminal_reason")
        subtype = result.get("subtype")
        status = result.get("api_error_status")
        # A Codex-route turn puts the provider's own sentence on the turn.  A
        # Command Code worker with a mistyped key was reported with the bare
        # blocker while the turn said "unexpected status 401 Unauthorized".
        error = result.get("error")
        said = error.get("message") if isinstance(error, Mapping) else error
        # A provider's sentence can name a signed link or a local path, and
        # it goes to the manager, so it is scrubbed like an exception is.
        said = _safe_text(said) if isinstance(said, str) else ""
        if said and status is None:
            found = _HTTP_STATUS_SAID.search(said)
            status = int(found.group(1)) if found else None
        if said:
            result = {**result, "api_error_status": status}
        # "success" is the SDK saying its own result message arrived intact.
        # Printed beside "failed" it read as a contradiction and told a
        # manager nothing about the failure, so it stays in the run record
        # and leaves the one line a manager reads.
        if subtype == "success":
            subtype = None
        # The reason and the subtype are provider text too.  Scrubbed copies
        # go in the sentence; the checks below still read the raw result.
        reason = _safe_text(reason) if isinstance(reason, str) else reason
        subtype = _safe_text(subtype) if isinstance(subtype, str) else subtype
        detail = ", ".join(
            value for value in (
                str(reason) if isinstance(reason, str) and reason else "",
                str(subtype) if isinstance(subtype, str) and subtype else "",
                f"HTTP {status}" if isinstance(status, int) and not isinstance(status, bool)
                and not said else "",
                f'"{said}"' if said else "",
            ) if value
        )
        if VNextScheduler._credential_was_rejected(result):
            said = f" ({detail})" if detail else ""
            return (
                f"{blocker}: the provider refused the credential{said} -- the "
                "key or login is wrong, retrying will not help, so correct the "
                "credential before delegating again"
            )
        if not detail:
            return blocker
        if result.get("rate_limited") is True:
            wait = result.get("retry_after_seconds")
            waiting = (
                f" and asked for {int(wait)}s"
                if isinstance(wait, (int, float)) and not isinstance(wait, bool) and wait > 0
                else ""
            )
            return f"{blocker}: the provider rate-limited this turn ({detail}){waiting}"
        if attempts > 1:
            return (
                f"{blocker}: the provider ended the turn ({detail}) after "
                f"{attempts} attempts"
            )
        return (
            f"{blocker}: the provider ended the turn ({detail}), so the packet "
            "itself may still be sound and worth retrying"
        )

    def _completion_effect_progress(self, agent: AgentRecord) -> dict[str, Any] | None:
        """Capture known native effects before completion makes progress immutable.

        The journal normally reads a turn when it ends, but a completion tool
        marks the agent terminal while that provider turn is still open. Read
        the same native event slice now so the first manager view has effects.
        The journal deduplicates the later turn-end read.
        """

        with self._lock:
            turn = self._active_turns.get(agent.agent_id)
        if turn is not None:
            try:
                events_since = getattr(self.managed._adapter_for(agent.agent_id), "events_since", None)
                if callable(events_since):
                    self.managed.runtime_effects.observe_turn(
                        agent_id=agent.agent_id,
                        thread_id=turn.runtime.thread_id,
                        turn_id=turn.runtime.turn_id,
                        events=events_since(turn.runtime.cursor),
                    )
            except Exception:
                # A provider's event reader cannot veto a valid completion.
                pass
        try:
            effects = self.managed.runtime_effect_summary(agent.agent_id)
        except Exception:
            return None
        if not effects["effect_count"]:
            return None
        return {
            "activity": "completed",
            "progress": f"{int(effects['effect_count'])} native tool effect(s) recorded",
            "files_touched": list(effects["files"]),
            "commands": ["recorded command"] if effects["command_count"] else [],
            "material": False,
        }

    def _cached_exact_model(self, model_id: str) -> str:
        card = self.managed.control.registry.cards.get(model_id)
        provider = str(getattr(card, "provider", "") or "")
        return cached_exact(provider, model_id) or PENDING_EXACT_MODEL

    def _refresh_model_identity(self, agent: AgentRecord) -> None:
        """Copy the runtime's exact-model answer onto the agent record."""

        with self.managed._lock:
            adapter = self.managed._adapters.get(agent.agent_id)
            thread_id = self.managed._threads.get(agent.agent_id)
        identity = read_identity(adapter, thread_id or getattr(agent, "thread_id", None))
        if identity:
            agent.model_identity = identity
            stored = self._session().agents.get(agent.agent_id)
            if stored is not None and stored is not agent:
                stored.model_identity = identity

    def _view_with_usage_final(self, view: dict[str, Any]) -> dict[str, Any]:
        """Mark terminal usage provisional until its provider turn has ended."""

        agent_id = str(view["agent_id"])
        with self._lock:
            turn_open = agent_id in self._active_turns or bool(self._stopping_turns.get(agent_id))
        view["usage_final"] = view["status"] in {status.value for status in TERMINAL_STATUSES} and not turn_open
        return view

    def _record_stopped_progress(self, agent: AgentRecord, blocker: str) -> None:
        """Leave behind what a stopped child had managed to do.

        A stopped child wakes its manager with a reason and there is still
        something to steer.  A manager that arrives to a bare status and an
        empty progress line has to spend a turn asking what happened, which is
        the polling the instructions tell it not to do.  So the effects the
        child recorded before it stopped are written down first, in the same
        shape a completion writes them.

        The progress record is deliberately not material: the status change
        that follows is the wake, and recording one event twice would cost the
        manager a second render for no second fact.
        """

        if agent.status is AgentStatus.RUNNING:
            try:
                effects = self.managed.runtime_effect_summary(agent.agent_id)
                self.managed.record_progress(
                    agent.agent_id,
                    activity="stopped before completing",
                    progress=(
                        f"{blocker}; {int(effects['effect_count'])} native tool "
                        "effect(s) recorded before stopping"
                    ),
                    files_touched=list(effects["files"]),
                    commands=["recorded command"] if effects["command_count"] else [],
                    material=False,
                )
            except Exception:
                # Evidence is worth having, never worth losing the blocker for,
                # and this catch has to mean that literally. The journal is read
                # off disk and the counts come from it, so an OSError or a
                # malformed summary is as plausible here as a ProtocolError. Any
                # of them escaping would leave the child RUNNING with a finished
                # turn -- a state nothing restarts -- and its manager never told.
                pass

    def _block_with_evidence(self, agent: AgentRecord, blocker: str) -> None:
        self._record_stopped_progress(agent, blocker)
        self._block(agent, blocker)

    def _fail_with_evidence(self, agent: AgentRecord, reason: str) -> None:
        self._record_stopped_progress(agent, reason)
        self.managed.control.fail_agent(agent.agent_id, reason)
        self.hooks.record_agent(agent)
        self.queue_terminal_release(agent.agent_id)

    def _block(self, agent: AgentRecord, blocker: str) -> None:
        """Block a child and say so where a person can see it.

        block_agent only changes the control plane. The manager is woken and
        can inspect, but nothing outside the tree learned anything: across
        every run record on this machine the string "blocked" had never once
        been written. A worker that died of a 400 went on being drawn as
        running in the status line, and the run record agreed with it.
        """

        self.managed.control.block_agent(agent.agent_id, blocker)
        self.hooks.record_agent(agent)
        # A blocked agent has no turn and cannot start one until its manager
        # decides something, which can be never. Holding its provider process
        # open for that whole time is what kept blocked Claude workers alive
        # until somebody cancelled them. Release the runtime thread; the
        # adapter reconnects the saved session when a later retry gives this
        # agent a turn again. A message alone only queues mail: _wake lifts
        # AWAITING_WORKERS and leaves BLOCKED where it is.
        self.queue_terminal_release(agent.agent_id)

    @staticmethod
    def _is_user_text(message: Any) -> bool:
        return message.kind == "user" and message.sender_id == "user"

    def _has_unread_user_text(self, agent: AgentRecord) -> bool:
        return any(
            self._is_user_text(item) for item in agent.messages[agent.delivered_message_count:]
        )

    def _raw_user_prompt(self, agent: AgentRecord) -> tuple[str, str]:
        """The unread user text of a raw-prompt chat, exactly as typed.

        The unread tail is reordered so the user's messages come first and only
        they are marked delivered.  Peer mail behind them stays unread and is
        offered in the next vNext tool response, as for a native child.  An
        older tool offer no longer describes that tail, so it is withdrawn.
        """

        wakes = self.managed.control.drain_wakes(agent.agent_id)
        self._drained_wakes[agent.agent_id] = list(wakes)
        for wake in wakes:
            self.hooks.lifecycle(
                "wake",
                agent,
                {"reason": wake.get("reason"), "source_agent": wake.get("source_agent_id")},
            )
        session = self._session()
        with session.lock:
            offset = agent.delivered_message_count
            unread = agent.messages[offset:]
            texts = [item for item in unread if self._is_user_text(item)]
            agent.messages[offset:] = texts + [item for item in unread if not self._is_user_text(item)]
            agent.delivered_message_count = offset + len(texts)
            self._native_messages_offered[agent.agent_id] = agent.delivered_message_count
        return "\n\n".join(item.text for item in texts), "user-raw"

    def _agent_prompt(self, agent: AgentRecord) -> tuple[str, str]:
        if agent.agent_id in self._raw_prompt_agents:
            return self._raw_user_prompt(agent)
        self_context = json.dumps(self._self_context(agent), ensure_ascii=False)
        durable_instructions = self._contract_instructions(agent)
        approval = self._approval_for_manager(agent.agent_id)
        if approval is not None:
            return (
                self._approval_prompt(approval)
                + "\nSELF_CONTEXT:\n"
                + self_context
                + durable_instructions
                + "\n"
                + self._clock_line(agent),
                "approval-review",
            )
        if agent.turn_count == 0:
            # A peer can send an agent mail before its first native turn.  The
            # opening prompt must deliver it instead of silently waiting for a
            # resume that may never be needed.
            messages = self._new_messages(agent)
            if agent.role is AgentRole.ROOT_MANAGER:
                return (
                    "[PRIMARY] Own the user outcome below. Execute what needs your judgement and delegate "
                    "token-heavy exploration, broad research, large logs, and bounded execution to the cheapest "
                    "capable agents before they consume your context. Seek contextual cross-provider review when it "
                    "can improve the result. Call await_children only once there is nothing useful to do now. "
                    "Complete only after all direct children are terminal.\nUSER OBJECTIVE:\n"
                    f"{agent.objective}\nTASK CONTRACT:\n{json.dumps(agent.task_contract, ensure_ascii=False)}"
                    f"\nSELF_CONTEXT:\n{self_context}{durable_instructions}\nMESSAGES:\n"
                    f"{json.dumps(messages, ensure_ascii=False)}"
                    f"\n{self._clock_line(agent)}",
                    "root-initial",
                )
            return (
                "[AGENT] Own this task. You may execute it and create any useful direct children. Offload bulky "
                "research, repository mapping, terminal output, and repetitive work early so the coordinating context "
                "holds intent and decisions. Exchange source-backed messages with peers and request contextual review "
                "across providers when useful. Verify child evidence and complete_agent only after every direct child "
                "is terminal.\nTASK OBJECTIVE:\n"
                f"{agent.objective}\nTASK CONTRACT:\n{json.dumps(agent.task_contract, ensure_ascii=False)}"
                f"\nSELF_CONTEXT:\n{self_context}{durable_instructions}\nMESSAGES:\n"
                f"{json.dumps(messages, ensure_ascii=False)}"
                f"\n{self._clock_line(agent)}",
                "agent-initial",
            )
        wakes = self.managed.control.drain_wakes(agent.agent_id)
        self._drained_wakes[agent.agent_id] = list(wakes)
        for wake in wakes:
            self.hooks.lifecycle(
                "wake",
                agent,
                {
                    "reason": wake.get("reason"),
                    "source_agent": wake.get("source_agent_id"),
                },
            )
        children = self._render_children(agent)
        messages = self._new_messages(agent)
        return (
            f"[{agent.role.value.upper()} RESUME] Continue owning your outcome. Execute, inspect evidence, message "
            "peers, steer, interrupt, retry, replace, delegate further, await active children, or complete when justified.\n"
            f"SELF_CONTEXT:\n{self_context}{durable_instructions}\n"
            f"WAKES:\n{json.dumps(wakes, ensure_ascii=False)}\n"
            f"CHILDREN:\n{json.dumps(children, ensure_ascii=False)}\n"
            f"MESSAGES:\n{json.dumps(messages, ensure_ascii=False)}"
            f"\n{self._clock_line(agent)}",
            f"{agent.role.value}-resume",
        )

    # Kept for existing integrations that used the old helper name.  The
    # implementation is deliberately role-neutral now.
    def _manager_prompt(self, agent: AgentRecord) -> tuple[str, str]:
        return self._agent_prompt(agent)

    def _clock_line(self, agent: AgentRecord, *, turn_started: float | None = None) -> str:
        """One line telling an agent the time and how much turn budget is left.

        The turn is the figure the worker acts on.  ``AgentRecord.created_at``
        is written once at birth and nothing else on the record reads it, so
        the agent's age answers a different question: on a fifth turn it would
        read hours while the turn was seconds old.  It rides behind the turn
        as context rather than leading.
        """

        now = self._now()
        elapsed = 0.0 if turn_started is None else max(0.0, self._monotonic() - turn_started)
        return turn_start_line(
            now,
            elapsed=elapsed,
            limit=self.turn_timeout,
            agent_age=max(0.0, now - agent.created_at),
        )

    def _steer_clock_line(self, agent_id: str) -> str:
        """The mid-turn line appended to a steer that lands inside a live turn."""

        with self._lock:
            started = self._turn_started_at.get(agent_id)
        now = self._now()
        return mid_turn_line(
            now,
            elapsed=None if started is None else max(0.0, self._monotonic() - started),
            limit=self.turn_timeout,
        )

    @staticmethod
    def _self_context(agent: AgentRecord) -> dict[str, Any]:
        """Expose native context telemetry without inventing a stopping threshold."""

        usage = dict(agent.usage)
        return {
            "agent_id": agent.agent_id,
            **{
                key: usage[key]
                for key in ("provider", "tokens", "context", "cost", "cost_tokens", "cost_usd", "provider_calls")
                if key in usage
            },
        }

    @staticmethod
    def _contract_instructions(agent: AgentRecord) -> str:
        instructions = agent.task_contract.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            return ""
        return "\nSESSION INSTRUCTIONS:\n" + instructions.strip()

    def _manager_instructions(self, role: AgentRole) -> str:
        selected_catalog = self._selected_model_catalog()
        shared = (
            f"ROLE={role.value}" + chr(10) +
            "Roles describe the current responsibility; every live agent may execute work, "
            "delegate direct children, and collaborate. Choose the cheapest capable agent for token-heavy research, "
            "large repository reads, logs, repetitive tests, and bounded execution. Keep high-judgment context for "
            "intent, trade-offs, integration, and source-backed review. SELECTED MODEL CATALOG (claims retain their "
            "evidence label and pointer): "
            f"{json.dumps(selected_catalog, ensure_ascii=False)}. Missing claims mean capability, quality, and cost are "
            "unknown; do not treat a missing claim as free. The user's preset still enforces model eligibility; there is no fixed "
            "tree depth or active-agent ceiling." + chr(10) +
            "EXPLICIT TASK CONSTRAINTS OVERRIDE DEFAULT ORGANIZATION. When the user or task contract specifies "
            "agent count, direct parentage, descriptive roles, model, effort, or required steps, satisfy it exactly. "
            "Choose the organization only within those constraints; do not add a coordination layer merely to organize work." + chr(10) +
            "ORDER OF WORK IN A TURN. Launch every ready lane before doing further analysis; do not delegate one child "
            "and stop. Do not delegate work you can finish more cheaply yourself. Then do the work that can be done now "
            "with your current context. Then yield with await_children when there is nothing useful left to do. Yielding "
            "is what you do when you have nothing left to do." + chr(10) +
            'VNEXT COORDINATION. Inspect with agent_id "self" once to discover children[].agent_id and terminal states. '
            "Use these canonical vNext IDs for await_children and control tools; runtime_thread_id identifies "
            "the provider thread separately. Cancelled children are terminal: do not wait for them to finish "
            "again or restart them without a new task decision. After await_children, end your turn when parked; "
            "when settled is true, continue or complete. Use the vNext tools exposed by your harness. "
            "In Codex exec, relay tools use tools.mcp__vnext_relay__<tool_name>; native provider wait tools "
            "do not track vNext cancellation. MCP replies expose structuredContent, with a single JSON text "
            "fallback. Native inbox replies give explicit acknowledgement instructions." + chr(10) +
            "Do not poll. Do not call inspect in a loop to see whether anything moved; your next "
            "prompt carries what changed, and a child you were already shown in full is summarised "
            "rather than repeated." + chr(10) +
            "When work stalls, inspect once, steer once with a narrower packet, then decide whether to revise the plan, "
            "message a peer, seek a fresh review, retry, replace, interrupt_agent, or cancel it. There is no fixed retry "
            "cutoff, no short blanket timeout, and no approval gate." + chr(10) +
            "ONE WRITER PER WORKSPACE. Children sharing a workspace must not write the same files at "
            "the same time. Give parallel writers disjoint scopes, or sequence them." + chr(10) +
            "Use inspect, send_message, steer, interrupt_agent, retry, replace, and cancel_agent when evidence warrants "
            "them. Use complete_agent after every direct child is terminal, or complete_session only for the primary's "
            "objective. If it reports unread messages, end the turn instead: the next prompt delivers them under MESSAGES. "
            "Never claim a tool effect from prose alone."
        )
        if role is not AgentRole.WORKER:
            return shared
        return (
            shared + chr(10) +
            "HOW YOU FINISH. Your task is the objective above. The rule about waiting for children "
            "applies to you only if you delegated some yourself. When your task is done, call "
            "complete_agent. It takes three required fields: your summary in outcome, the evidence you "
            "actually gathered in evidence, and verified set to whether you checked the result "
            "yourself. If you did delegate children, wait until every one of them is terminal before "
            "you call it. A written answer at the end of your turn does not finish your task. vNext "
            "parks you and your manager keeps waiting for a result that never arrives. Write the "
            "answer, then call complete_agent."
        )

    def await_children_blocking(
        self, agent_id: str, agent_ids: Sequence[str], *, timeout: float
    ) -> ToolCallResult:
        """Wait for selected children instead of parking their manager.

        The ordinary ``await_children`` parks a manager and wakes it in a fresh
        turn when its children settle.  A client that vNext does not run cannot
        be woken that way: it called a tool and is holding the call open.  So
        for an external primary the wait is real, and a wait that runs out says
        so rather than pretending the children finished.  Calling again after a
        timeout is safe and is the intended continuation.

        The same reasoning decides what ends the wait early.  A worker asks its
        manager before every command batch and every file change, and that
        question reaches a manager vNext runs as a turn prompt.  An external
        primary has no turn to put it in, so until now the question was
        invisible: the worker waited 240 seconds, was auto-declined, and the run
        log was the only place the request ever appeared.  A wait that returns
        the moment an approval is pending puts the question in the one channel
        this client does read.  Nothing is imposed on it: the approval is
        reported, and answering it, ignoring it, or waiting again all remain the
        caller's move.
        """

        session = self._session()
        selected = list(dict.fromkeys(str(value) for value in agent_ids))
        if not selected:
            raise ValueError("agent_ids must be a non-empty array")
        for child_id in selected:
            child = session.agents.get(child_id)
            if child is None or child.parent_agent_id != agent_id:
                raise ProtocolError(
                    "not-direct-child",
                    "await targets must be canonical vNext direct-child agent_ids. "
                    'Inspect with agent_id "self" once and use children[].agent_id.',
                )
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while True:
            # Settled, never terminal. A blocked child is waiting for this
            # caller to decide something and will wait for as long as nobody
            # does, so counting it as still working spends the whole timeout to
            # answer "still running" about the one child that never will be.
            # The parked path already reads it this way; an external primary
            # holding the call open gets the same reading.
            # A child whose turn ended with no complete_agent call is READY
            # and parked until it gets mail, so it counts the same way: only
            # this caller can move it.
            with self._lock:
                idle = [
                    child_id for child_id in selected
                    if session.agents[child_id].status is AgentStatus.READY
                    and child_id in self._awaiting_message_agents
                    and child_id not in self._active_turns
                ]
            pending = [
                child_id for child_id in selected
                if session.agents[child_id].status not in SETTLED_STATUSES
                and child_id not in idle
            ]
            approvals = [
                self._approval_receipt(value)
                for value in self._approvals_for_manager(agent_id)
            ]
            if not pending:
                blocked = [
                    {
                        "agent_id": value,
                        "status": session.agents[value].status.value,
                        "blocker": session.agents[value].blocker,
                    }
                    for value in selected
                    if session.agents[value].status is AgentStatus.BLOCKED
                ]
                outcome: dict[str, Any] = {
                    "awaiting": [],
                    "settled": True,
                    "children": [self._coordination_child(session.agents[value])
                                 for value in selected],
                    "pending_approvals": approvals,
                }
                notes: list[str] = []
                if blocked:
                    outcome["children_needing_a_decision"] = blocked
                    notes.append(
                        "These children stopped on a blocker and cannot go on "
                        "by themselves. Each one is waiting on a decision from "
                        "you: retry it on a revised contract, replace it with "
                        "another model, cancel_agent it, or delegate the work "
                        "afresh. Waiting again on the same children changes "
                        "nothing."
                    )
                if idle:
                    outcome["children_waiting_for_a_message"] = [
                        {"agent_id": value, "status": session.agents[value].status.value}
                        for value in idle
                    ]
                    notes.append(
                        "These children ended their turn without calling "
                        "complete_agent, so they have no result on record and "
                        "will not move again by themselves. Read what each one "
                        "said with inspect, then send_message to tell it to "
                        "finish with complete_agent or to go on, or cancel_agent "
                        "it. Waiting again on the same children changes nothing."
                    )
                if notes:
                    outcome["note"] = " ".join(notes)
                return _dynamic_result(True, outcome)
            if self._cancel_requested.is_set() or bool(self.cancellation.requested):
                raise SchedulerCancelled("vNext scheduler cancellation requested")
            if approvals:
                return _dynamic_result(True, {
                    "awaiting": pending,
                    "settled": False,
                    "children": [self._coordination_child(session.agents[value])
                                 for value in selected],
                    "pending_approvals": approvals,
                    "note": (
                        "A child is waiting on your decision and cannot go on until "
                        "you make it. Call resolve_approval once per approval_id "
                        "listed in pending_approvals, then call await_children again "
                        "with the same agent_ids. An approval nobody answers is "
                        "declined automatically and usually kills the child's turn."
                    ),
                })
            if time.monotonic() >= deadline:
                return _dynamic_result(True, {
                    "awaiting": pending,
                    "settled": False,
                    "children": [self._coordination_child(session.agents[value])
                                 for value in selected],
                    "pending_approvals": approvals,
                    "note": (
                        "The wait ran out while these children were still working. "
                        "They are unaffected: call await_children again with the same "
                        "agent_ids to keep waiting, or inspect them now."
                    ),
                })
            time.sleep(0.1)

    @staticmethod
    def _coordination_child(agent: AgentRecord) -> dict[str, Any]:
        """Small current-state receipt; never consumes the prompt-delivery ledger.

        A child that has stopped also carries what it reported, because this
        receipt is what ``await_children`` hands back and a manager waiting for
        a worker is waiting for its result.  A child still working carries none
        of those fields.
        """
        return {
            "agent_id": agent.agent_id,
            "runtime_thread_id": agent.thread_id,
            "parent_agent_id": agent.parent_agent_id,
            "model_id": agent.model_id,
            "objective_line": _objective_line(agent.objective),
            "status": agent.status.value,
            "terminal": agent.status in TERMINAL_STATUSES,
            **finished_agent_outcome(agent),
        }

    def _selected_model_catalog(self) -> list[dict[str, Any]]:
        session = self._session()
        preset = self.managed.control.registry.preset(session.preset_id)
        return [
            {
                "model_id": model_id,
                "provider": card.provider,
                "eligible_roles": sorted(role.value for role in card.eligible_roles),
                "claims": [
                    {
                        "kind": claim.kind.value,
                        "statement": claim.statement,
                        "evidence_pointer": claim.evidence_pointer,
                    }
                    for claim in card.claims
                ],
            }
            for model_id, card in sorted(self.managed.control.registry.cards.items())
            if model_id in preset.allowed_models
        ]

    def _render_children(self, agent: AgentRecord) -> list[dict[str, Any]]:
        """Render this manager's children, re-sending only what has changed.

        Every wake re-renders the whole child list, and a terminal Worker's
        record carries up to 4000 characters of report.  Ten finished Workers is
        roughly 40 KB re-sent on every wake, about work the manager was already
        shown and cannot influence any more.  That cost is what made waking a
        manager more often look expensive, so it is charged down here first.

        A child whose compact record is byte-identical to the one this manager
        last received is replaced by a short reference.  Nothing is hidden: the
        reference names the child and its status, says the record is unchanged,
        and points at ``inspect``, which returns the full view on demand.  A
        child that moved at all is sent in full, so the manager never has to ask
        for news it should have been handed.

        The ledger is per manager, because two managers are shown different
        subtrees and each has seen its own set.
        """

        delivered = self._children_delivered.setdefault(agent.agent_id, {})
        delivered_before = dict(delivered)
        rendered: list[dict[str, Any]] = []
        for child_id in agent.child_ids:
            child = self._compact_child(child_id)
            digest = _child_digest(child)
            delivered[child_id] = digest
            if delivered_before.get(child_id) == digest:
                reference = {
                    "agent_id": child.get("agent_id"),
                    "role": child.get("role"),
                    "status": child.get("status"),
                    "unchanged_since_last_delivered": True,
                    "note": _UNCHANGED_CHILD_NOTE,
                }
                # A reference is only worth sending when it is smaller than the
                # record it stands in for.  A child that has barely started
                # carries almost nothing, and the sentence explaining its
                # absence costs more than the fields would have.  Substituting
                # there would make the prompt longer in the name of making it
                # shorter, so the guard keeps the saving real in every case
                # rather than on average.
                full_length = _encoded_length(child)
                reference_length = _encoded_length(reference)
                if reference_length < full_length:
                    self.managed.context_saved_characters += (
                        full_length - reference_length
                    )
                    rendered.append(reference)
                    continue
            rendered.append(child)
        return rendered

    def _compact_child(self, child_id: str) -> dict[str, Any]:
        view = self._view_with_usage_final(
            self.managed.control.inspect_agent(self.root.agent_id, child_id, deep=True)
        )
        return {
            key: view.get(key)
            for key in (
                "agent_id",
                "role",
                "model_id",
                "status",
                "latest_progress",
                "blocker",
                "files_touched",
                "commands",
                "usage",
                "usage_final",
                "result",
            )
        }

    def _unread_message_count(self, agent: AgentRecord) -> int:
        return len(agent.messages) - agent.delivered_message_count

    def _require_messages_read(self, agent: AgentRecord) -> None:
        """Refuse a completion that would bury messages nobody ever read.

        A manager may still complete after reading them, change plan, or decide
        the message does not alter the outcome.  What it may not do is finish
        without ever seeing it: the message would be silently dropped and the
        sender would have no way to tell.  The refusal is a tool result the
        manager acts on, and the very next turn delivers the message.
        """

        unread = self._unread_message_count(agent)
        if unread <= 0:
            return
        if (self._agent_is_external(agent.agent_id)
                or self._native_message_delivery(agent.agent_id) == "available"):
            raise ProtocolError(
                "unread-messages",
                'Call inspect with agent_id "self" to read queued messages, then inspect with agent_id "self" and acknowledge_messages_through set to message_cursor before completing.',
            )
        raise ProtocolError(
            "unread-messages",
            f"{unread} message(s) addressed to you have not been read yet. Do not "
            "complete on this turn. End this turn instead; your next prompt "
            "delivers them under MESSAGES. Read them, then complete if the "
            "outcome still stands, or change plan if it does not.",
        )

    def _worker_reads_messages(self, agent: AgentRecord) -> None:
        """Give a Worker one more turn so it actually reads what it was sent.

        A manager can be refused its completion tool and decide what to do about
        it.  A Worker has no completion tool: the scheduler completes it when its
        turn ends, so refusing is not available and the guarantee has to be
        delivery instead.  The Worker returns to READY, the scheduler starts a
        follow-up turn, and ``_worker_prompt`` hands it the messages.  It stays
        free to decide the message changes nothing and finish immediately; what
        it cannot do is finish without seeing it.
        """

        unread = self._unread_message_count(agent)
        try:
            self.managed.control.finish_turn(agent.agent_id)
        except ProtocolError:
            # The Worker reached a terminal state by another route -- a cancel
            # or a replace -- while this turn was being wrapped up.  There is no
            # turn left to give it.  Those paths announce the stranded mail
            # themselves, so this one stops rather than crashing the scheduler.
            self.hooks.lifecycle(
                "worker_messages_stranded",
                agent,
                {"role": agent.role.value, "unread": unread},
            )
            return
        self.hooks.emit(
            agent,
            "ready",
            "worker_messages_pending",
            f"Worker takes another turn to read {unread} message(s)",
        )
        self.hooks.lifecycle(
            "worker_messages_pending",
            agent,
            {"role": agent.role.value, "unread": unread},
        )

    def _new_messages(self, agent: AgentRecord, *, mark_delivered: bool = True) -> list[dict[str, Any]]:
        """Hand over the next batch of undelivered messages, oldest first.

        The batch is capped so one prompt cannot be flooded, and the offset
        advances by exactly what was handed over.  Anything past the cap stays
        undelivered rather than being marked read and dropped, so the unread
        guard keeps the agent taking turns until the queue is empty. Native
        tool responses pass mark_delivered=False and acknowledge separately.
        """

        offset = agent.delivered_message_count
        selected = agent.messages[offset : offset + MESSAGE_BATCH]
        if mark_delivered:
            agent.delivered_message_count = offset + len(selected)
        return [
            {
                "sender_id": item.sender_id,
                "kind": item.kind,
                "text": item.text,
                "direct_override": item.direct_override,
            }
            for item in selected
        ]

    def _approval_for_manager(self, manager_id: str) -> _ApprovalEnvelope | None:
        with self._lock:
            return next(
                (
                    value
                    for value in self._pending_approvals.values()
                    if value.manager_id == manager_id and value.outcome is None
                ),
                None,
            )

    def _approvals_for_manager(self, manager_id: str) -> list[_ApprovalEnvelope]:
        with self._lock:
            return [
                value
                for value in self._pending_approvals.values()
                if value.manager_id == manager_id and value.outcome is None
            ]

    @staticmethod
    def _approval_receipt(approval: _ApprovalEnvelope) -> dict[str, Any]:
        """The same facts the approval prompt carries, as data.

        A manager vNext runs is handed a pending approval as its next turn
        prompt (``_agent_prompt`` returns the ``approval-review`` kind).  A
        client vNext does not run never takes a turn, so the prompt never
        reaches it and the approval times out unseen.  This receipt is how the
        same facts reach that client: through the return value of the call it
        is already holding open.
        """

        request = approval.request
        return {
            "approval_id": approval.approval_id,
            "worker_agent_id": approval.worker_id,
            "tool": request.tool,
            "effect": request.effect,
            "permission": request.permission,
            "target": request.target,
            "justification": request.justification,
            "command": list(request.command),
        }

    @staticmethod
    def _approval_prompt(approval: _ApprovalEnvelope) -> str:
        request = approval.request
        command = json.dumps(list(request.command), ensure_ascii=False)
        return (
            "[APPROVAL] "
            f"approval_id={approval.approval_id} worker_id={approval.worker_id} "
            f"tool={request.tool} effect={request.effect} permission={request.permission} "
            f"target={request.target} justification={request.justification!r} argv={command}. "
            "Call resolve_approval exactly once, then continue or end this turn as appropriate."
        )

    def _interrupt_turn(self, agent_id: str) -> None:
        with self._lock:
            active = self._active_turns.get(agent_id)
            native_child = agent_id in self._native_child_delivery_contracts
        if active is not None:
            self.managed._adapter_for(agent_id).interrupt(active.runtime)
        elif native_child and (
            self._session().agents[agent_id].status not in TERMINAL_STATUSES
            or agent_id in self._unstopped_native_children
        ):
            # The second clause is the retry. A cancel the provider refused
            # leaves the record cancelled and the turn running, so the terminal
            # guard alone turned every later cancel into a silent success.
            adapter = self.managed._adapter_for(agent_id)
            stop = getattr(adapter, "cancel_native_child", None)
            if not callable(stop):
                self._unstopped_native_children.add(agent_id)
                raise SchedulerError("provider cannot stop an unobserved native child turn")
            # A child can already be executing before the collector sees its
            # first turn event. Ask the provider using the attested thread.
            try:
                stop(self._session().agents[agent_id].thread_id)
            except BaseException:
                self._unstopped_native_children.add(agent_id)
                raise
            self._unstopped_native_children.discard(agent_id)

    def _interrupt_unconfirmed_stop(self, agent_id: str) -> None:
        """Ask again for a turn that was told to stop and never confirmed it.

        ``_interrupt_turn`` reads the active list, and this turn left that list
        when our own wait on it gave up.  Cancelling such an agent sent nothing
        at all, so the one command a manager has for "stop writing" was a no-op
        against the exact case it was needed for.  Sending the stop again is
        cheap and the provider is free to refuse it; the caller records a
        refusal the way it records every other one.
        """

        with self._lock:
            # A newer active turn does not mean nothing older is held: an
            # explicit start_despite_unconfirmed_stop runs one beside the held
            # turn, and returning here answered the cancel with success while
            # the held turn was never asked again.  ``_interrupt_turn`` already
            # asked the active turn, so only that one is left out.
            active = self._active_turns.get(agent_id)
            active_runtime = getattr(active, "runtime", None)
            turns = [
                turn
                for turn in self._stopping_turns.get(agent_id, {}).values()
                if getattr(turn, "runtime", None) is not None
                and (active_runtime is None or turn.runtime is not active_runtime)
            ]
        if not turns:
            return
        adapter = self.managed._adapter_for(agent_id)
        first_failure: BaseException | None = None
        for turn in turns:
            # Every held turn is asked, because a retry can leave two of them
            # unconfirmed and stopping only the newest leaves the other writing.
            try:
                adapter.interrupt(turn.runtime)
            except BaseException as exc:
                if first_failure is None:
                    first_failure = exc
        if first_failure is not None:
            raise first_failure

    def _subtree_agent_ids(self, agent_id: str) -> list[str]:
        """Return the target and every descendant from one control snapshot."""

        session = self._session()
        if agent_id not in session.agents:
            raise ProtocolError("invalid-handle", "unknown agent handle")
        result: list[str] = []
        pending = [agent_id]
        while pending:
            current = pending.pop(0)
            result.append(current)
            pending.extend(session.agents[current].child_ids)
        return result

    def _cancel_tree(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        with self._lock:
            targets = set(self._active_turns) | set(self._native_child_delivery_contracts)
            approvals = list(self._pending_approvals.values())
        for agent_id in targets:
            try:
                # Native children may already be running while their first
                # turn event is still queued. Use the same provider-first
                # interruption path as an explicit subtree cancellation.
                self._interrupt_turn(agent_id)
            except Exception:
                pass
        for approval in approvals:
            approval.outcome = ApprovalOutcome(ApprovalDecision.DECLINE, "session cancelled")
            approval.resolver = "system-cancellation"
            approval.resolved.set()
        session = self._session()
        previous = {
            agent_id: agent.status
            for agent_id, agent in session.agents.items()
        }
        try:
            self.managed.control.cancel_agent(
                requester_id=self.root.agent_id,
                agent_id=self.root.agent_id,
            )
        except ProtocolError:
            pass
        for agent_id, agent in session.agents.items():
            if (
                previous.get(agent_id) not in TERMINAL_STATUSES
                and agent.status in TERMINAL_STATUSES
            ):
                self.hooks.lifecycle(
                    "agent_terminal",
                    agent,
                    {"role": agent.role.value, "status": agent.status.value},
                )

    def _resolve_pending_approval(
        self,
        *,
        approval_id: str,
        decision: ApprovalDecision,
        rationale: str,
        resolver: str,
        expected_manager_id: str | None = None,
    ) -> None:
        with self._lock:
            envelope = self._pending_approvals.get(approval_id)
            if (
                envelope is None
                or envelope.outcome is not None
                or (
                    expected_manager_id is not None
                    and envelope.manager_id != expected_manager_id
                )
            ):
                raise ProtocolError(
                    "unknown-approval",
                    "approval is not pending for this resolver",
                )
            envelope.outcome = ApprovalOutcome(decision, rationale)
            envelope.resolver = resolver
            envelope.resolved.set()

    def _perform_reconnect_if_safe(self) -> None:
        if not self._reconnect_requested.is_set():
            return
        with self._lock:
            if self._active_turns:
                return
        factory = self.hooks.adapter_factory
        if factory is None:
            raise SchedulerError("no reconnect adapter factory is configured")
        def arm(adapter: Any) -> None:
            adapter.native_approval_handler = self.review_approval

        self.managed.reconnect(factory, on_adapter_ready=arm)
        for adapter in self.managed.runtime_adapters():
            arm(adapter)
        self._reconnect_requested.clear()
        self.hooks.emit(self.root, "running", "reconnect_completed", "Persistent sessions reconnected")
        self.hooks.lifecycle("reconnect_completed", self.root, {})

    def _quiesce_yielded_managers_for_reconnect(self) -> None:
        if not self._reconnect_requested.is_set():
            return
        session = self._session()
        with self._lock:
            candidates = [
                (agent_id, turn)
                for agent_id, turn in self._active_turns.items()
                if agent_id not in self._reconnect_interrupts
                and session.agents[agent_id].role is not AgentRole.WORKER
                and session.agents[agent_id].active_turn_id is None
                and session.agents[agent_id].status not in TERMINAL_STATUSES
            ]
            self._reconnect_interrupts.update(agent_id for agent_id, _turn in candidates)
        for agent_id, turn in candidates:
            try:
                self.managed._adapter_for(turn.agent_id).interrupt(turn.runtime)
            except Exception:
                with self._lock:
                    self._reconnect_interrupts.discard(agent_id)

    def _is_deadlocked(self) -> bool:
        session = self._session()
        with self._lock:
            if self._active_turns or self._pending_approvals:
                return False
        if any(agent.status is AgentStatus.READY for agent in session.agents.values()):
            return False
        return self.root.status not in TERMINAL_STATUSES
