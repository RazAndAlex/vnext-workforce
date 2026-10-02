"""Content-safe failure classification for the vNext provider leaves.

Decision 3 permits statuses, action-type labels, counts, booleans, and exit
and duration facts in anything that survives a run.  It forbids message text,
prompts, model responses, file contents, credentials, raw identifiers, and
outside-workspace paths.  This module is the one place where a failure is
turned into a *label* from a closed vocabulary, so the reason a run died can
outlive the run without any of the forbidden material travelling with it.

Nothing here ever emits the text it classified.  A bridge traceback may carry
absolute paths; that is precisely why it is consumed by the classifier and
reduced to a category rather than reported.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path

_ENABLE_ENV = "VNEXT_DIAGNOSTICS"
_DIR_ENV = "VNEXT_DIAGNOSTIC_DIR"
_SINK_NAME = "vnext-failures.jsonl"
# `data/` is already declared local runtime output in .gitignore, so a record
# written there cannot be committed by accident and does not live inside any
# caller-supplied (and possibly destroyed) workspace.
_DEFAULT_SUBDIR = ("data", "diagnostics")

# A label token: lowercase category values and CamelCase exception class names
# only.  Deliberately excludes every path separator, drive colon, and space.
# `\Z` rather than `$`: `$` also matches immediately before a trailing newline,
# which would let "leaked_value\n" through the scrubber unchanged.
_TOKEN = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.\-]*\Z")
_PATHISH = re.compile(r"[\\/]|^[A-Za-z]:|\.\.")

# The only two exceptions this module ever lets past its guards.  They mean the
# process is being torn down, and swallowing them would be a worse defect than
# losing one diagnostic record.  Everything else — including a custom
# `BaseException` subclass raised by a hostile `__str__`, which a bare
# `except Exception` would have missed — is absorbed, so the totality promise
# `record_failure` makes in its docstring is literally true rather than
# true-for-ordinary-arguments.
_PROPAGATE: tuple[type[BaseException], ...] = (KeyboardInterrupt, SystemExit)

# A record carries a handful of counts and flags.  A mapping whose `items()`
# never stops would hang the failure path it is meant to describe, which is the
# same defect as raising from it wearing a different hat, so the read is bounded.
_MAX_ITEMS = 256


class FailureCategory(str, Enum):
    """Closed vocabulary of reasons a provider leaf can fail to run."""

    WORKSPACE_UNAVAILABLE = "workspace_unavailable"
    BRIDGE_SPAWN_REFUSED = "bridge_spawn_refused"
    BRIDGE_STREAMS_UNAVAILABLE = "bridge_streams_unavailable"
    BRIDGE_STDIN_UNAVAILABLE = "bridge_stdin_unavailable"
    BRIDGE_WRITE_FAILED = "bridge_write_failed"
    BRIDGE_EXITED_AT_STARTUP = "bridge_exited_at_startup"
    BRIDGE_NOT_RUNNING = "bridge_not_running"
    BRIDGE_PROTOCOL_VIOLATION = "bridge_protocol_violation"
    BRIDGE_REQUEST_TIMEOUT = "bridge_request_timeout"
    SDK_NOT_INSTALLED = "sdk_not_installed"
    SDK_VERSION_MISMATCH = "sdk_version_mismatch"
    CREDENTIAL_OVERRIDE_PRESENT = "credential_override_present"
    PROVIDER_CREDENTIAL_MISSING = "provider_credential_missing"
    # A key that is present and refused is a different repair from a key
    # that is absent: the first is corrected, the second is supplied.
    PROVIDER_CREDENTIAL_REJECTED = "provider_credential_rejected"
    INITIALIZE_ATTESTATION_MISMATCH = "initialize_attestation_mismatch"
    POSTURE_REJECTED = "posture_rejected"
    TOOL_REGISTRATION_REJECTED = "tool_registration_rejected"
    IDENTITY_BINDING_FAILED = "identity_binding_failed"
    APPROVAL_CORRELATION_FAILED = "approval_correlation_failed"
    PROVIDER_CONNECT_FAILED = "provider_connect_failed"
    PROVIDER_QUERY_FAILED = "provider_query_failed"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    LOCAL_RESERVATION_UNKNOWN = "local_reservation_unknown"
    LOCAL_GENERATION_STALE = "local_generation_stale"
    REQUEST_PAYLOAD_INVALID = "request_payload_invalid"
    EFFECT_STATE_EXHAUSTED = "effect_state_exhausted"
    # The SDK and the interactive terminal may own one native session, but
    # never both.  Keep that transfer failure distinct from a missing local
    # reservation: it means the reservation is known and its owner has not
    # yet yielded control.
    TERMINAL_LEASE_CONFLICT = "terminal_lease_conflict"
    ENVIRONMENT_RESTORATION_FAILED = "environment_restoration_failed"
    CONTROL_PLANE_CONSTRUCTION_FAILED = "control_plane_construction_failed"
    # A caller tool handler hosted inside a provider SDK is caught by that
    # SDK: the model is shown an error result and the exception never
    # reaches this process.  Without its own label the failure is invisible.
    MANAGER_TOOL_HANDLER_FAILED = "manager_tool_handler_failed"
    IMPORT_FAILED = "import_failed"
    UNCLASSIFIED = "unclassified"


# Ordered fragment table.  Every left-hand side is a fixed, non-interpolated
# literal already present in the adapter or the bridge; the first match wins,
# so the more specific fragments are listed first.
_FRAGMENTS: tuple[tuple[str, FailureCategory], ...] = (
    ("optional claude-agent-sdk", FailureCategory.SDK_NOT_INSTALLED),
    ("the claude sdk (claude-agent-sdk", FailureCategory.SDK_NOT_INSTALLED),
    ("installed claude sdk version", FailureCategory.SDK_VERSION_MISMATCH),
    ("credential override", FailureCategory.CREDENTIAL_OVERRIDE_PRESENT),
    ("workspace is unavailable", FailureCategory.WORKSPACE_UNAVAILABLE),
    ("workspace does not exist", FailureCategory.WORKSPACE_UNAVAILABLE),
    ("workspace is not a folder", FailureCategory.WORKSPACE_UNAVAILABLE),
    ("workspace differs from owned workspace", FailureCategory.WORKSPACE_UNAVAILABLE),
    ("unable to start owned claude bridge", FailureCategory.BRIDGE_SPAWN_REFUSED),
    ("has no captured streams", FailureCategory.BRIDGE_STREAMS_UNAVAILABLE),
    ("stdin is unavailable", FailureCategory.BRIDGE_STDIN_UNAVAILABLE),
    ("cannot write owned claude bridge", FailureCategory.BRIDGE_WRITE_FAILED),
    ("record exceeds jsonl bound", FailureCategory.BRIDGE_WRITE_FAILED),
    ("stdout closed unexpectedly", FailureCategory.BRIDGE_EXITED_AT_STARTUP),
    ("bridge is not running", FailureCategory.BRIDGE_NOT_RUNNING),
    ("was not initialized", FailureCategory.BRIDGE_NOT_RUNNING),
    ("must initialize before", FailureCategory.BRIDGE_NOT_RUNNING),
    ("timed out waiting for", FailureCategory.BRIDGE_REQUEST_TIMEOUT),
    ("did not attest claude capabilities", FailureCategory.INITIALIZE_ATTESTATION_MISMATCH),
    ("lacks server capability evidence", FailureCategory.PROVIDER_CONNECT_FAILED),
    ("did not attest a connected local reservation", FailureCategory.PROVIDER_CONNECT_FAILED),
    ("lacks a connected client", FailureCategory.PROVIDER_CONNECT_FAILED),
    ("resume lacks exact provider echo", FailureCategory.PROVIDER_CONNECT_FAILED),
    ("fresh tool resume", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("resume developer instructions", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("claude query failed", FailureCategory.PROVIDER_QUERY_FAILED),
    ("bridge operation failed", FailureCategory.PROVIDER_QUERY_FAILED),
    ("posture", FailureCategory.POSTURE_REJECTED),
    ("tool registration", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("refuses caller-provided tools", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("native session identity", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native identity", FailureCategory.IDENTITY_BINDING_FAILED),
    ("attested native session", FailureCategory.IDENTITY_BINDING_FAILED),
    ("permission event", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("interrupt was not exactly correlated", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("attested provider correlation", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("unsupported claude bridge operation", FailureCategory.UNSUPPORTED_OPERATION),
    ("steer", FailureCategory.UNSUPPORTED_OPERATION),
    ("emitted an unsupported event", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("jsonl", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("protocol record", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("non-object result", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("no pending request", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("event cursor", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("event is not an object", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("unattested terminal result", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("did not correlate the local turn", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("lacks exact provider thread correlation", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    # Fragments below were APPENDED to close a measured coverage gap, and new
    # fragments must keep being appended rather than interleaved: a fragment
    # placed at the end can only capture a message the table above already
    # declined, so an addition can never re-route an existing classification.
    # `test_no_source_literal_is_unclassified` parses the real adapter and
    # bridge sources, so a literal added there without a fragment here fails.
    ("did not resolve approval-gated", FailureCategory.POSTURE_REJECTED),
    ("conflicting native session", FailureCategory.IDENTITY_BINDING_FAILED),
    ("stale reservation generation", FailureCategory.LOCAL_GENERATION_STALE),
    ("stale local generation", FailureCategory.LOCAL_GENERATION_STALE),
    ("lacks bounded", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("capacity exceeded", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("incoming request fields", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("turn prompt is invalid", FailureCategory.REQUEST_PAYLOAD_INVALID),
    # Claude keeps model and effort fixed for a connected native session.  A
    # caller mismatch is an invalid request, while an invalid transcript is a
    # provider-boundary protocol defect rather than a missing reservation.
    ("effort is not supported", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("unsupported claude effort", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("turn model drifted", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("turn effort drifted", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("diagnostics timeout", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("invalid transcript page", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("reservation lacks a transcript", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("violates bridge protocol", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("is not json", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("unknown provider thread", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    ("local runtime reservation", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    ("local reservation", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    ("lacks local correlation", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    ("no active local turn", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    # Surfaced by widening the source scan beyond `ClaudeRuntimeError` and
    # `BridgeError`.  `provide only one ...` is a plain `ValueError` the
    # narrower scan was structurally unable to see; `... bridge rejected ...`
    # is the statically visible fallback arm of an `or`, which the scan can now
    # read.  Both classified as `unclassified` before this block existed.
    ("provide only one", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("bridge rejected", FailureCategory.PROVIDER_QUERY_FAILED),
    # Appended for manager tool hosting.  A caller tool handler failing inside
    # the provider SDK is its own reason, distinct from a rejected registration.
    ("caller tool handler", FailureCategory.MANAGER_TOOL_HANDLER_FAILED),
    # Native terminal ownership is a real lifecycle boundary.  These fragments
    # intentionally name the conflict instead of collapsing it into an unknown
    # reservation, which would send recovery down the wrong path.
    ("permission mode is not supported", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("unsupported claude permission mode", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("owned by an active native terminal", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("already owned by a terminal", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("terminal handoff already has an active lease", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("resume lacks an active lease", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("before terminal stop is confirmed", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("while terminal owns the session", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("terminal lease could not be released", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("cannot hand claude session to terminal while an sdk turn is active", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("already has an active turn", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("cannot release claude session while an sdk turn is active", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("with pending control callbacks", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("terminal relay channel is unsupported", FailureCategory.UNSUPPORTED_OPERATION),
    ("terminal manager tools have no registered handler", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("terminal release lacks an attested local session", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    ("terminal release does not match a bound claude session", FailureCategory.IDENTITY_BINDING_FAILED),
    ("terminal release lacks a connected sdk client", FailureCategory.PROVIDER_CONNECT_FAILED),
    ("native task stop lacks provider task identity", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("native task stop lacks an observed child task", FailureCategory.LOCAL_RESERVATION_UNKNOWN),
    ("native task stop has no active sdk controller", FailureCategory.PROVIDER_CONNECT_FAILED),
    ("native task stop was not exactly correlated", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("bridge did not attest idle terminal release", FailureCategory.INITIALIZE_ATTESTATION_MISMATCH),
    ("resume requires the prior sdk client to be released", FailureCategory.EFFECT_STATE_EXHAUSTED),
    # Native Agent enrollment has an exact hook-to-MCP capability boundary.
    # A missing hook server is a registration defect, while an unknown child
    # identity or context proof must fail as an identity binding error.
    ("native child registration lacks enrollment hooks", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("native child tool context lacks enrollment hooks", FailureCategory.TOOL_REGISTRATION_REJECTED),
    ("native child registration was not exactly attested", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child tool context was not exactly attested", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child register hook lacks agent identity", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native task lacks parent tool use", FailureCategory.APPROVAL_CORRELATION_FAILED),
    # Once a child is adopted, its task ID is the only valid control turn.
    # Correlation failures, impossible provider terminal states, and failed
    # attestation stay distinct so the scheduler can choose the right repair.
    ("native child interrupt lacks exact task correlation", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("native child wait lacks exact task correlation", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("native child tool call lacks exact task correlation", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("native child wait timed out", FailureCategory.BRIDGE_REQUEST_TIMEOUT),
    ("native child has an unsupported terminal state", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("native child parent agent is not exactly resolved", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child observer returned an invalid binding", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child identity conflicts with prior observation", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child task conflicts with prior observation", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child tool call is not attested", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child terminal status conflicts with prior terminal", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("native child terminal state conflicts with prior evidence", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("native child origin state is unavailable", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("native child origin lacks an active parent turn", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("native child event lacks immutable parent origin", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("native child event has invalid immutable parent origin", FailureCategory.APPROVAL_CORRELATION_FAILED),
    ("native child cancel lacks an attested task", FailureCategory.IDENTITY_BINDING_FAILED),
    # The terminal relay also owns the local interrupt and tool-controller
    # contracts.  Missing leases and bad callers are lifecycle/request errors;
    # callback failures are provider operations, while a malformed controller
    # result or terminal status breaks the relay contract itself.
    ("terminal interrupt lacks an active lease", FailureCategory.TERMINAL_LEASE_CONFLICT),
    ("terminal interrupt must be callable", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("terminal has no interrupt controller", FailureCategory.PROVIDER_QUERY_FAILED),
    ("terminal interrupt failed", FailureCategory.PROVIDER_QUERY_FAILED),
    ("terminal tool controller failed", FailureCategory.MANAGER_TOOL_HANDLER_FAILED),
    ("terminal tool controller returned an unsupported result", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("terminal turn timed out", FailureCategory.BRIDGE_REQUEST_TIMEOUT),
    ("terminal turn has an unattested status", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    # Automatic native-child identity keeps bounded, ephemeral replay state.
    # A missing state bucket means the bridge cannot safely continue its
    # attested effects; malformed lifecycle data is a broken bridge contract.
    ("reservation lacks metadata retry state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("reservation lacks automatic native-child replay state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("reservation lacks native child identity ledger", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("reservation lacks automatic native-child lifecycle state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("metadata retry cursor is malformed", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("native child lifecycle history is malformed", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("native child lifecycle lacks a source", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    # A nested child cannot be attached without the exact native parent task;
    # a collision with the hook ledger likewise invalidates the identity join.
    ("nested native child lacks an exact parent task", FailureCategory.IDENTITY_BINDING_FAILED),
    ("automatic native child conflicts with hook identity ledger", FailureCategory.IDENTITY_BINDING_FAILED),
    ("native child observed model conflicts with prior metadata", FailureCategory.IDENTITY_BINDING_FAILED),
    ("reservation lacks native child state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    # A primary turn has a separate outcome waiter from the long-lived SDK
    # reader.  Missing local state is unrecoverable bookkeeping; malformed
    # completion or an ended provider stream is a protocol/query boundary.
    ("reservation lacks local turn outcome state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("turn completion is malformed", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("message stream ended before turn terminal result", FailureCategory.PROVIDER_QUERY_FAILED),
    # Reusing an SDK reservation is generation-scoped. A malformed bridge
    # reader cannot safely accept a follow-up turn.
    ("thread lacks a valid generation", FailureCategory.LOCAL_GENERATION_STALE),
    ("readiness lacks exact local correlation", FailureCategory.LOCAL_GENERATION_STALE),
    ("reservation has malformed reader state", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    # A forwarded typed AssistantMessage is retained only behind an exact
    # native origin and promoted after the automatic identity join. Conflicts
    # are identity failures; missing buckets are bounded state defects.
    ("forwarded child effective model", FailureCategory.IDENTITY_BINDING_FAILED),
    ("reservation lacks forwarded child model state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("reservation lacks observed child model state", FailureCategory.EFFECT_STATE_EXHAUSTED),
    ("forwarded child model state is malformed", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("native child observed model conflicts", FailureCategory.IDENTITY_BINDING_FAILED),
    # Optional provider configuration is a provider boundary of its own.  Keep
    # a missing z.ai key distinct from an Anthropic environment override, and
    # classify malformed provider selection as caller input rather than an SDK
    # failure.
    ("z.ai provider is not configured", FailureCategory.PROVIDER_CREDENTIAL_MISSING),
    ("did not attest z.ai capabilities", FailureCategory.INITIALIZE_ATTESTATION_MISMATCH),
    ("adapter provider is not supported", FailureCategory.REQUEST_PAYLOAD_INVALID),
    ("bridge provider is not supported", FailureCategory.REQUEST_PAYLOAD_INVALID),
    # Releasing a finished agent's SDK client echoes the exact reservation it
    # parked; a mismatched echo is a broken bridge contract, and a release
    # asked for a status that is not settled is malformed caller input.
    ("agent release lacks exact reservation ownership", FailureCategory.BRIDGE_PROTOCOL_VIOLATION),
    ("agent release lacks a settled status", FailureCategory.REQUEST_PAYLOAD_INVALID),
    # A terminal handoff also drops the reservation's SDK reader, so it is
    # refused while a native child still needs that reader.  Same busy-state
    # category as the active-turn refusal above: the caller may retry once the
    # child settles.
    ("cannot hand claude session to terminal while ", FailureCategory.EFFECT_STATE_EXHAUSTED),
    # The provider answered a turn by refusing the login.  It reproduces on
    # every retry, so it is ended at once rather than waited out.
    ("refused the credential", FailureCategory.PROVIDER_CREDENTIAL_REJECTED),
)

# Exception classes that identify a category on their own.  Used for the
# spawn/write funnels and for reducing a bridge traceback to one label.
_EXCEPTION_TYPES: Mapping[str, FailureCategory] = {
    "ModuleNotFoundError": FailureCategory.SDK_NOT_INSTALLED,
    "ImportError": FailureCategory.SDK_NOT_INSTALLED,
    "FileNotFoundError": FailureCategory.BRIDGE_SPAWN_REFUSED,
    "NotADirectoryError": FailureCategory.WORKSPACE_UNAVAILABLE,
    "PermissionError": FailureCategory.BRIDGE_SPAWN_REFUSED,
    "BrokenPipeError": FailureCategory.BRIDGE_WRITE_FAILED,
    "ConnectionResetError": FailureCategory.BRIDGE_WRITE_FAILED,
}


def classify_message(message: object) -> FailureCategory:
    """Map one fixed adapter/bridge literal to a category.

    The message is read and discarded.  Only the returned label leaves.
    """

    if not isinstance(message, str) or not message.strip():
        return FailureCategory.UNCLASSIFIED
    lowered = message.lower()
    for fragment, category in _FRAGMENTS:
        if fragment in lowered:
            return category
    return FailureCategory.UNCLASSIFIED


def classify_exception(exc: BaseException | None) -> FailureCategory:
    """Map an exception to a category by class name, then by its message."""

    if exc is None:
        return FailureCategory.UNCLASSIFIED
    named = _EXCEPTION_TYPES.get(type(exc).__name__)
    if named is not None:
        return named
    return classify_message(str(exc))


def classify_stderr(lines: Iterable[str]) -> FailureCategory:
    """Reduce captured bridge stderr to a single category.

    A traceback routinely contains absolute paths, so it is consumed here and
    never reported.  Later lines carry the raised class, so the scan runs
    backwards and the first recognised class wins.
    """

    collected = [line for line in lines if isinstance(line, str)]
    for line in reversed(collected):
        head = line.split(":", 1)[0].strip()
        named = _EXCEPTION_TYPES.get(head.rsplit(".", 1)[-1])
        if named is not None:
            return named
        category = classify_message(line)
        if category is not FailureCategory.UNCLASSIFIED:
            return category
    if collected:
        return FailureCategory.BRIDGE_EXITED_AT_STARTUP
    return FailureCategory.UNCLASSIFIED


def drain_stderr(buffer: object) -> FailureCategory:
    """Classify and empty a bounded stderr buffer, keeping nothing."""

    lines: list[str] = []
    try:
        while True:
            lines.append(buffer.popleft())  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        pass
    return classify_stderr(lines)


def diagnostics_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV) == "1"


def _default_sink_dir() -> Path:
    return Path(__file__).resolve().parents[1].joinpath(*_DEFAULT_SUBDIR)


def override_is_usable(override: str) -> bool:
    """Whether an operator-supplied sink directory may be honoured.

    Two overrides defeat the sink's whole purpose and are rejected here:

    * a RELATIVE path, which resolves against the current working directory.
      When that is the repository it drops an untracked ``vnext-failures.jsonl``
      into the source tree, where nothing gitignores it.
    * a path that is not usable as a directory location at all.

    A caller must additionally never point the override *inside a directory
    the run itself destroys* — a canary's temporary workspace, above all.
    That is the exact failure the sink exists to avoid, and it cannot be
    detected from here, because a legitimate override into a caller-owned
    temporary directory looks identical.  It is the call site's obligation.
    """

    try:
        return bool(override) and Path(override).is_absolute()
    except _PROPAGATE:
        raise
    except BaseException:
        return False


def sink_path() -> Path:
    try:
        override = os.environ.get(_DIR_ENV)
    except Exception:
        override = None
    if override and override_is_usable(override):
        return Path(override) / _SINK_NAME
    return _default_sink_dir() / _SINK_NAME


def _safe_token(value: object) -> str:
    """Reduce a value to a bare label token, or to ``"invalid_label"``.

    CALL-SITE DISCIPLINE — read before adding a caller.  This function is a
    last-resort *shape* filter, not a redaction engine.  It rejects anything
    that is not a bare token, so it does stop paths, prose, and anything
    containing a space or separator.  It does NOT stop a value that already
    has token shape: a credential such as ``sk-ant-api03-1a2b3c4d`` is a legal
    token and passes through verbatim, as does any opaque identifier.

    Therefore every caller must pass a value that is a *fixed literal chosen
    by this codebase* — an operation name, a phase or step name, a
    ``FailureCategory``, or ``type(exc).__name__``.  Nothing attacker-,
    provider-, or environment-controlled may ever reach this function.  That
    property is what makes the sink content-safe; the regex only catches
    mistakes of shape, and cannot catch a mistake of provenance.

    Total by construction: a value whose ``__str__`` raises yields
    ``"invalid_label"`` rather than propagating, so diagnosis can never
    destroy the failure path it is observing.
    """

    try:
        text = value.value if isinstance(value, FailureCategory) else str(value)
        if not isinstance(text, str):
            return "invalid_label"
        if not _TOKEN.match(text) or _PATHISH.search(text):
            return "invalid_label"
        return text
    except _PROPAGATE:
        raise
    except BaseException:
        return "invalid_label"


def _safe_truthy(value: object) -> bool:
    """Evaluate truthiness without letting a hostile ``__bool__`` escape."""

    try:
        return bool(value)
    except _PROPAGATE:
        raise
    except BaseException:
        return False


def _safe_duration(value: object) -> float | None:
    """Coerce a duration to a rounded float, or to ``None``."""

    if value is None:
        return None
    try:
        number = round(float(value), 3)  # type: ignore[arg-type]
    except _PROPAGATE:
        raise
    except BaseException:
        return None
    # NaN and the infinities are not JSON, and `allow_nan=False` would raise
    # at write time; drop them here so the record stays writable.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def record_failure(
    category: FailureCategory,
    *,
    phase: str,
    step: str = "",
    exception_type: object = None,
    duration_ms: float | None = None,
    counts: Mapping[str, int] | None = None,
    flags: Mapping[str, bool] | None = None,
) -> Path | None:
    """Write one content-safe failure record, eagerly, outside any workspace.

    Returns the sink path, or ``None`` when diagnostics are off or the write
    could not be made.  Diagnosis must never change what a run does, so this
    function is TOTAL: no argument it is given can make it raise, and record
    construction is inside the guard rather than in front of it.  A caller on
    a failure path may therefore call it without a try/except and without
    fearing that observing a failure destroys it.

    ``KeyboardInterrupt`` and ``SystemExit`` are deliberately NOT swallowed:
    those mean the process is being torn down, and suppressing them would be
    a worse defect than losing one diagnostic record.  Every OTHER
    ``BaseException`` is absorbed, including one raised by a hostile ``__str__``
    that does not inherit from ``Exception`` — a plain ``except Exception``
    left that case escaping and made this docstring's promise false.
    """

    try:
        if not diagnostics_enabled():
            return None
        record = build_record(
            category,
            phase=phase,
            step=step,
            exception_type=exception_type,
            duration_ms=duration_ms,
            counts=counts,
            flags=flags,
        )
        target = sink_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        # `allow_nan=False`: NaN/Infinity are not JSON and would poison the
        # sink for any reader.  `_safe_duration` already drops them, so this
        # is the backstop rather than the filter.
        line = json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except _PROPAGATE:
        raise
    except BaseException:
        return None
    return target


def build_record(
    category: FailureCategory,
    *,
    phase: str,
    step: str = "",
    exception_type: object = None,
    duration_ms: float | None = None,
    counts: Mapping[str, int] | None = None,
    flags: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Build the record ``record_failure`` writes; every value is scrubbed."""

    record: dict[str, object] = {
        "v": 1,
        "kind": "vnext_failure",
        "category": _safe_token(category),
        "phase": _safe_token(phase),
        "step": _safe_token(step) if _safe_truthy(step) else "",
        "exception_type": (
            _safe_token(exception_type) if _safe_truthy(exception_type) else None
        ),
        "duration_ms": _safe_duration(duration_ms),
        "recorded_monotonic_ms": round(time.monotonic() * 1000.0, 3),
    }
    record["counts"] = _safe_counts(counts)
    record["flags"] = _safe_flags(flags)
    return record


def _safe_items(mapping: object) -> list[tuple[object, object]]:
    """Read a mapping's items as PAIRS, without letting a hostile mapping escape.

    ``items()`` is duck-typed, so it may yield anything at all.  The previous
    version returned whatever it produced and left the two-value unpacking to
    the caller's ``for key, value in ...`` loop, so a mapping whose ``items()``
    yielded a three-element entry raised ``ValueError: too many values to
    unpack`` *out of* ``build_record``.  ``record_failure`` then swallowed it
    and returned ``None``, silently losing the diagnostic — exactly the failure
    totality exists to prevent.

    Every entry is now validated here, so callers may unpack unconditionally.
    Entries that are not two-value sequences are dropped rather than guessed
    at: a malformed count is not worth risking the failure path for.
    """

    if mapping is None:
        return []
    try:
        iterator = iter(mapping.items())  # type: ignore[union-attr]
    except _PROPAGATE:
        raise
    except BaseException:
        return []
    pairs: list[tuple[object, object]] = []
    while len(pairs) < _MAX_ITEMS:
        try:
            entry = next(iterator)
        except StopIteration:
            break
        except _PROPAGATE:
            raise
        except BaseException:
            # A generator that raises mid-iteration keeps whatever it yielded
            # before the fault; a partial record still beats no record.
            break
        try:
            key, value = entry
        except _PROPAGATE:
            raise
        except BaseException:
            continue
        pairs.append((key, value))
    return pairs


def _safe_counts(counts: object) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in _safe_items(counts):
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        result[_safe_token(key)] = value
    return result


def _safe_flags(flags: object) -> dict[str, bool]:
    return {_safe_token(key): _safe_truthy(value) for key, value in _safe_items(flags)}


def record_values(record: Mapping[str, object]) -> list[object]:
    """Flatten a record to every scalar it contains, for leakage assertions.

    Fully recursive over mappings and sequences at any depth.  The previous
    one-level version made the leakage tests weaker than they looked: a value
    nested two deep, or inside a list, was skipped by the very assertion
    meant to catch it.
    """

    values: list[object] = []

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                walk(key)
                walk(nested)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item)
        else:
            values.append(value)

    walk(record)
    return values


__all__ = [
    "FailureCategory",
    "build_record",
    "classify_exception",
    "classify_message",
    "classify_stderr",
    "diagnostics_enabled",
    "drain_stderr",
    "override_is_usable",
    "record_failure",
    "record_values",
    "sink_path",
]
