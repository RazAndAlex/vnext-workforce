"""Owned, bounded JSONL entry point for the optional Claude SDK leaf.

It is intentionally a capability boundary rather than an SDK compatibility
layer.  In particular, it refuses to bind a managed session until the provider
echoes a native session identity.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vnext.clock_format import mid_turn_line
from vnext.vnext_claude_effort import CLAUDE_ACCEPTED_EFFORTS
from vnext.vnext_model_identity import model_table, resolve_claude_alias
from vnext.vnext_claude_native_identity import (
    NativeChildAutomaticResolver,
    NativeChildIdentity,
    NativeChildIdentityError,
    NativeChildIdentityLedger,
)
from vnext.vnext_approval_subject import approval_subject
from vnext.vnext_claude_native_mirror import NativeMetadataMirror
from vnext.vnext_provider_config import ZAI_ENDPOINT, load_zai_provider

# The one claude-agent-sdk release the bridge is written against; pyproject pins it too.
CLAUDE_SDK_VERSION = "0.2.163"
# Set by the side-runtime bridge command (vnext_claude.side_runtime_bridge_command).
_SDK_VERSION_ENV = "VNEXT_CLAUDE_SDK_VERSION"
_CLI_PATH_ENV = "VNEXT_CLAUDE_CLI_PATH"


def _expected_sdk_version() -> str:
    return os.environ.get(_SDK_VERSION_ENV) or CLAUDE_SDK_VERSION
_VERSION = 1
_MAX_JSONL = 65_536
# A tool result that carries an image arrives from the CLI as one base64 JSON
# message, and the SDK refuses any message over its default 1 MiB buffer.
# Anthropic accepts requests up to 32 MB, so one message can approach that size.
# The bridge drops tool result content before its own stdout (_MAX_JSONL).
_SDK_MAX_BUFFER_SIZE = 64 * 1024 * 1024
# Opted-in tool arguments above this size travel as a preview, so one event
# stays well inside _MAX_JSONL after escaping and the event envelope.
_TOOL_INPUT_MAX_CHARS = 32_000
_TOOL_INPUT_PREVIEW_CHARS = 8_000
_MAX_PENDING_EFFECTS = 64
# CLI status/init notifications are not assistant messages or effects. Keep
# their own bounded budget so a startup burst cannot exhaust the 64 slots
# needed by transcript/stream events before assistant/result identity binds.
_MAX_PENDING_SYSTEM_MESSAGES = 1024
_MAX_METADATA_REFRESH_PER_EVENT = 4
# The resolver keeps at most this many unresolved children, so one settling
# pass at a parent terminal can cover all of them and still be bounded.
_MAX_PENDING_NATIVE_CHILDREN = 64
_NATIVE_CHILD_COORDINATION_METADATA_ATTEMPTS = 2
_NATIVE_CHILD_COORDINATION_METADATA_DELAY_SECONDS = 0.05
_CREDENTIAL_OVERRIDES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
_EFFECT_TYPES = {"Bash": "command", "Write": "file"}
# Caller tools are hosted by one in-process SDK MCP server.  The SDK namespaces
# every tool it registers, so a manager tool reaches the model as
# `mcp__vnext__delegate` and reaches this bridge under its bare name.
_TOOL_SERVER_KEY = "vnext"
_TOOL_NAMESPACE = f"mcp__{_TOOL_SERVER_KEY}__"
_TOOL_CALL_TIMEOUT_SECONDS = 300.0
# What a CLI system message may carry out of the bridge, by subtype.  Read
# from the init and compact_boundary writers in Claude Code 2.1.288.  Any
# other subtype forwards its session id alone, since hook and command output
# can hold anything.
_SYSTEM_MESSAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "init": (
        "session_id", "model", "tools", "slash_commands", "terminal_slash_commands",
        "mcp_servers", "permissionMode", "output_style", "agents", "skills",
        "claude_code_version", "uuid",
    ),
    "compact_boundary": ("session_id", "uuid", "compact_metadata"),
}
# The user's own settings load in every worker, and with them the vNext plugin
# this repository ships.  Each worker then started a second, independent vNext
# server of its own beside the bridge's tools: 13 extra proxy and server pairs
# in a live session on 2026-09-23.  The flag-settings layer outranks user settings,
# so this switches off that one plugin and leaves the rest of the setup alone.
# The id is plugins/.claude-plugin/marketplace.json's name after the plugin's.
_WORKER_SETTINGS = json.dumps({"enabledPlugins": {"vnext@vnext": False}})
_NATIVE_CHILD_REGISTER_TOOL = "vnext_register_native_child"
_NATIVE_CHILD_CONTEXT_FIELD = "_vnext_native_child_context"
_PERMISSION_DIAGNOSTIC_KEYS = (
    "callback_count", "hosted_callback_without_agent_id", "hosted_callback_with_agent_id",
    "denied_missing_tool_use_id", "allowed_hosted", "denied_unhosted",
    "allowed_discovery", "allowed_no_effect",
    "review_requested", "review_accepted", "review_declined",
)


def _counter_value(value: Any) -> int:
    """Read a non-negative integer counter.  Anything else reads as zero."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0

def _event_name(event: Any) -> str:
    """Name one held event.  Anything without a usable name reads as unnamed."""

    if isinstance(event, Mapping):
        name = event.get("name")
        if isinstance(name, str) and name:
            return name
    return "unnamed"


def _held_census(pending: list[Any]) -> str:
    """Say what fills a buffer, so a capacity error diagnoses itself."""

    counts: dict[str, int] = {}
    for held in pending:
        name = _event_name(held)
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items())) or "nothing"


def _write_change_descriptor(workspace: Path | None, input_data: object) -> tuple[list[dict[str, Any]], bool]:
    """Project only a bounded workspace-relative Write path.

    Claude's Write input also contains file contents.  The bridge must never
    retain or relay that value, so the descriptor is deliberately built from
    ``file_path`` alone and records the provider's missing change kind as
    ``not_reported``.
    """

    if workspace is None or not isinstance(input_data, Mapping):
        return [], True
    raw_path = input_data.get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return [], True
    root = workspace.resolve()
    try:
        candidate = (root / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        relative = candidate.relative_to(root)
    except (OSError, ValueError):
        return [{"path": "<outside-workspace>", "kind": {"type": "not_reported"}}], True
    return [{"path": relative.as_posix() or ".", "kind": {"type": "not_reported"}}], False


def _nonblank(value: object) -> bool:
    """True for a string that names something. Blank counts as absent."""

    return isinstance(value, str) and bool(value.strip())


def _native_session_id(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value != "default"


def _resolved_posture(reviewer: str) -> dict[str, Any]:
    """Return the neutral posture resolved by one connected SDK client."""

    return {
        "workspace_writes": True,
        "network": "approval_gated",
        "approvals_requested": True,
        "reviewer": reviewer,
        "environment_ready": True,
    }


def _requested_reviewer(value: object) -> str:
    if not isinstance(value, Mapping):
        raise BridgeError("thread request lacks a neutral posture")
    reviewer = value.get("reviewer")
    if (
        set(value) != {
            "workspace_writes",
            "network",
            "approvals_requested",
            "reviewer",
            "environment_ready",
        }
        or value.get("workspace_writes") is not True
        or value.get("network") not in {"restricted", "approval_gated"}
        or value.get("approvals_requested") is not True
        or not isinstance(reviewer, str)
        or not reviewer
        or value.get("environment_ready") is not True
    ):
        raise BridgeError("thread request violates the connected Claude posture")
    return reviewer


def _requested_effort(value: object) -> str:
    if value not in CLAUDE_ACCEPTED_EFFORTS:
        raise BridgeError("thread request has unsupported Claude effort")
    return str(value)


def _requested_permission_mode(value: object) -> str:
    if value in {"bypassPermissions", "auto"}:
        # The CLI approves calls in these modes before it consults
        # can_use_tool (auto through its own permission evaluator), and
        # relay_permission is where vNext asks the reviewer, refuses untracked
        # native children and records effects.  The posture this bridge
        # attests (approvals requested, network approval-gated) would then be
        # false, so the binding fails instead.
        raise BridgeError(f"unsupported Claude permission mode: {value} skips can_use_tool, which carries vNext approvals")
    if value not in {"default", "acceptEdits", "plan"}:
        raise BridgeError("thread request has unsupported Claude permission mode")
    return str(value)


def _policy(reviewer: str) -> dict[str, Any]:
    # This describes the approval route requested by vNext, not a claim that
    # the Claude SDK has installed an OS-level network sandbox.
    return {
        "posture": _resolved_posture(reviewer),
        "capabilities": {
            "native_tools": "standard",
            "settings_sources": ["user", "project", "local"],
            "skills": "all",
            "network_enforcement": "not_attested",
            "native_child_tracking": "attested_enrollment",
        },
    }


def _validated_tools(value: object) -> list[dict[str, Any]]:
    """Accept only tool definitions this bridge can actually host."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise BridgeError("thread request includes a malformed tool registration")
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for definition in value:
        if not isinstance(definition, Mapping):
            raise BridgeError("thread request includes a malformed tool registration")
        name, description, schema = definition.get("name"), definition.get("description"), definition.get("inputSchema")
        if (
            not isinstance(name, str)
            or not name
            or name in seen
            or not isinstance(description, str)
            or not description
            or not isinstance(schema, Mapping)
            or schema.get("type") != "object"
            or not isinstance(schema.get("properties"), Mapping)
        ):
            raise BridgeError("thread request includes a malformed tool registration")
        seen.add(name)
        definitions.append(dict(definition))
    return definitions


def _registration(model: str, definitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Attest what was really bound; an empty payload keeps the leaf shape."""

    digest: str | None = None
    if definitions:
        digest = hashlib.sha256(
            json.dumps([dict(value) for value in definitions], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return {
        "acknowledged": True,
        "model_id": model,
        "tool_count": len(definitions),
        "tool_names": [str(value["name"]) for value in definitions],
        "definition_sha256": digest,
        "handler_registered": bool(definitions),
    }


# The neutral effect each standard Claude tool asks the manager to approve.
# `review_approval` in vnext_scheduler understands these four effects and
# declines every other envelope, so a tool missing from this table cannot run.
# Adding a name here widens what a Claude-family worker may do; removing one
# narrows it.
#
# "read" and "network" were added after a field run: a worker with a standing
# grant was refused WebSearch twice, because a tool absent from this table is
# declined before the grant is ever read.  A file read outside the workspace
# reached the same dead end.  Both classes still go to the reviewer, so a
# manager on `approvals: ask` keeps its say over what a worker reads and what
# it sends to the network.
_TOOL_EFFECTS: dict[str, str] = {
    "Write": "modify",
    "Edit": "modify",
    "MultiEdit": "modify",
    "NotebookEdit": "modify",
    "Bash": "execute",
    "BashOutput": "execute",
    "KillShell": "execute",
    "Read": "read",
    "Grep": "read",
    "Glob": "read",
    "NotebookRead": "read",
    "WebSearch": "network",
    "WebFetch": "network",
}

# Tools that change nothing outside the worker's own turn: a todo list the
# provider keeps in the transcript, and leaving plan mode.  They carry no
# effect a manager could review, so they are allowed here instead of sending
# the manager an approval per item.  Everything with a real effect belongs in
# _TOOL_EFFECTS above.
_LOCAL_NO_EFFECT_TOOLS = frozenset({"TodoWrite", "ExitPlanMode"})


def _tool_effect(tool_name: str) -> dict[str, str]:
    """Name the neutral effect for one tool, or name nothing and fail closed."""

    effect = _TOOL_EFFECTS.get(tool_name)
    return {"effect": effect} if effect is not None else {}


# One path or one command line is what a reviewer needs; a file's contents are
# not, and neither is an argv long enough to push the outgoing record past
# _MAX_JSONL.  Both are cut to this many characters.
_MAX_APPROVAL_FIELD = 400
_MAX_APPROVAL_ARGV = 16


def _approval_detail(
    tool_name: str, workspace: Path | None, input_data: object
) -> dict[str, Any]:
    """Say which file or which command one permission request is about.

    Every approval vNext showed a manager carried an empty command, so the
    manager was asked to approve a file edit without being told which file.
    One ZCode manager wrote that it could not see the command and approved
    anyway.  The path is projected through ``_write_change_descriptor``, which
    reads ``file_path`` alone: file contents never leave this bridge.
    """

    detail: dict[str, Any] = {"tool": tool_name[:_MAX_APPROVAL_FIELD]}
    effect = _TOOL_EFFECTS.get(tool_name)
    if effect == "modify":
        changes, _evidence_limited = _write_change_descriptor(workspace, input_data)
        path = changes[0].get("path") if changes else None
        if isinstance(path, str) and path:
            detail["path"] = path[:_MAX_APPROVAL_FIELD]
        return detail
    if effect in {"read", "network"} and isinstance(input_data, Mapping):
        # `_approval_subject` in vnext_scheduler shows the reviewer one subject
        # line, taken from "path".  For a read that is the file or directory
        # asked for; for WebFetch the URL; for WebSearch the query.  A reviewer
        # told only "WebFetch" cannot tell a docs page from an upload target.
        # Which field to read is the tool's to say, in vnext_approval_subject:
        # a scan of candidate names picked a decoy file_path over a WebFetch's
        # real url, and recorded the decoy.
        subject = approval_subject(tool_name, input_data, _MAX_APPROVAL_FIELD)
        if subject is not None:
            detail["path"] = subject
        return detail
    if effect == "execute" and isinstance(input_data, Mapping):
        # Claude's Bash tool sends one shell string under "command"; a provider
        # that sends a real argv list is carried element by element.  Anything
        # else leaves the tool name to speak alone.
        command = input_data.get("command")
        if isinstance(command, str) and command.strip():
            detail["command"] = [command[:_MAX_APPROVAL_FIELD]]
        elif isinstance(command, (list, tuple)):
            argv = [
                str(part)[:_MAX_APPROVAL_FIELD]
                for part in list(command)[:_MAX_APPROVAL_ARGV]
                if isinstance(part, str)
            ]
            if argv:
                detail["command"] = argv
    return detail


def _assert_tools_not_auto_allowed(allowed_tools: Sequence[str], definitions: Sequence[Mapping[str, Any]]) -> None:
    """Refuse to construct a session that would bypass the approval callback.

    A whole-tool entry in ``allowed_tools`` is auto-approved by the SDK BEFORE
    ``can_use_tool`` runs — the callback is never invoked and the SDK only
    warns.  vNext routes its approval rendezvous through that callback, so a
    manager tool listed there would silently delete the review step.  This is
    a guard, not a hint: it fails the binding.
    """

    listed = set(allowed_tools)
    for definition in definitions:
        name = str(definition.get("name") or "")
        if name in listed or f"{_TOOL_NAMESPACE}{name}" in listed:
            raise BridgeError("caller tool registration must not be auto-allowed")

def _native_canonical_model(value: object) -> bool:
    """Whether a string is a full provider model id rather than a catalog alias.

    Claude answers with ids such as ``claude-opus-5``.  A catalog entry may
    instead name ``opus``, which is a launch selection the provider resolves.
    """

    return isinstance(value, str) and value.startswith("claude-")


def _native_model_agrees(configured: object, observed: object) -> bool:
    """Whether one observed child model agrees with the configured launch model.

    A catalog alias is a request, never an observation, so it cannot be a rival
    observation either: an alias admits the first canonical id the provider
    reports for a child.  A configured canonical id still has to match exactly,
    and two different canonical ids for one child stay a conflict, which
    ``_native_model_conflicts`` keeps.  Agreement only lets the child through
    to the adapter: there a child on another family, such as Sonnet under an
    ``opus`` thread, is named for its own family or refused with a reason.
    """

    if not isinstance(observed, str) or not observed:
        return False
    if not isinstance(configured, str) or not configured:
        return False
    if observed == configured:
        return True
    return not _native_canonical_model(configured) and _native_canonical_model(observed)


def _native_model_conflicts(prior: object, observed: object) -> bool:
    """Whether two recorded models for the same child really disagree.

    Only canonical against canonical is a disagreement.  An alias already in
    the record is earlier, weaker evidence that a canonical id supersedes.
    """

    if not isinstance(prior, str) or not isinstance(observed, str) or prior == observed:
        return False
    return _native_canonical_model(prior) and _native_canonical_model(observed)



class BridgeError(RuntimeError):
    pass


class _StreamEnd:
    """Queued by the pump when the client's message stream ends."""

    def __init__(self, error: Exception | None) -> None:
        self.error = error


# The whole tool result the CLI writes in place of a call it did not run
# because a PreToolUse hook did not answer or failed.  CLI 2.1.286 has two:
# "PreToolUse hook did not respond before its timeout (host client may be
# unreachable). The tool call was not executed; other configured hooks may not
# have completed." and the same with "failed with an unexpected error".  It
# must be the entire result: a command that ran can print the sentence, and
# a pytest run asserting on it did block a worker.
_HOOK_FAILURE = re.compile(
    r"PreToolUse hook [^\n]{1,160}\. The tool call was not executed; "
    r"other configured hooks may not have completed\."
)
_HOOK_FAILURE_TEXT = 300
# Seconds the CLI waits for one of this bridge's hooks before it gives up.
_HOOK_TIMEOUT_SECONDS = 30


def _hook_failure_line(block: Any) -> str | None:
    """The hook-failure sentence of a failed tool result, or None."""

    if getattr(block, "is_error", None) is not True:
        return None
    content = getattr(block, "content", None)
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text") or "") for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        return None
    content = content.strip()
    if _HOOK_FAILURE.fullmatch(content) is None:
        return None
    return content[:_HOOK_FAILURE_TEXT]


def _write(record: Mapping[str, Any]) -> None:
    encoded = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) > _MAX_JSONL:
        raise BridgeError("outgoing JSONL record exceeds bound")
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def _response(request_id: int, *, result: Mapping[str, Any] | None = None, error: str | None = None) -> None:
    record: dict[str, Any] = {"v": _VERSION, "kind": "response", "id": request_id, "ok": error is None}
    if error is None:
        record["result"] = dict(result or {})
    else:
        record["error"] = error
    _write(record)


def _request(raw: str) -> tuple[int, str, Mapping[str, Any]]:
    if len(raw) > _MAX_JSONL:
        raise BridgeError("incoming JSONL record exceeds bound")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BridgeError("incoming record is not JSON") from exc
    if not isinstance(record, Mapping) or record.get("v") != _VERSION or record.get("kind") != "request":
        raise BridgeError("incoming record violates bridge protocol")
    request_id, op, payload = record.get("id"), record.get("op"), record.get("payload")
    if not isinstance(request_id, int) or request_id < 1 or not isinstance(op, str) or not isinstance(payload, Mapping):
        raise BridgeError("incoming request fields are invalid")
    return request_id, op, payload


_REPLAY_FLAG = "replay-user-messages"


def _replays_prompts(client: Any) -> bool:
    extra = getattr(getattr(client, "options", None), "extra_args", None)
    return isinstance(extra, Mapping) and _REPLAY_FLAG in extra


async def _echoed_prompt(prompt: str, echo: str):  # type: ignore[no-untyped-def]
    # The string form of ``query`` cannot carry a uuid; a stream message can.
    yield {
        "type": "user",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
        "uuid": echo,
    }


class _Bridge:
    def __init__(self) -> None:
        self._workspace: Path | None = None
        self._sdk: Any = None
        self._sdk_environment: dict[str, str] = {}
        self._reservations: dict[str, dict[str, Any]] = {}
        # The adapter retains one event deque across turns, so cursors must be
        # bridge-global rather than reset with a reservation or turn.
        self._event_cursor = 0

    async def run(self) -> None:
        active: set[asyncio.Task[None]] = set()
        while True:
            # A synchronous stdin iterator would hold the event loop at its next
            # read while a wait_turn is pending.  Reading lines off-loop lets an
            # interrupt request reach its independently scheduled dispatcher.
            raw = await asyncio.to_thread(sys.stdin.readline)
            if raw == "":
                break
            task = asyncio.create_task(self._serve(raw))
            active.add(task)
            task.add_done_callback(active.discard)
        if active:
            await asyncio.gather(*active)

    async def _serve(self, raw: str) -> None:
        try:
            control = self._control(raw)
        except BridgeError:
            control = None
        if control is not None:
            if control.get("op") == "tool_call_response":
                self._resolve_tool_call(control)
            else:
                self._resolve_permission(control)
            return
        request_id = 0
        try:
            request_id, op, payload = _request(raw)
            result = await self.dispatch(op, payload)
        except BridgeError as exc:
            _response(request_id, error=str(exc))
        except Exception as exc:  # SDK errors must never leak credential material.
            _response(request_id, error=f"Claude bridge operation failed: {type(exc).__name__}")
        else:
            _response(request_id, result=result)

    @staticmethod
    def _control(raw: str) -> Mapping[str, Any] | None:
        """Recognize one manager-to-bridge permission response, if present."""

        if len(raw) > _MAX_JSONL:
            raise BridgeError("incoming JSONL record exceeds bound")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError("incoming record is not JSON") from exc
        if not isinstance(record, Mapping) or record.get("kind") != "control":
            return None
        return record

    def _resolve_permission(self, record: Mapping[str, Any]) -> None:
        """Deliver an exact manager decision to its pending SDK callback only."""

        if (
            record.get("v") != _VERSION
            or record.get("op") != "permission_response"
            or set(record) != {"v", "kind", "op", "reservation_id", "turn_reference", "tool_use_id", "decision"}
        ):
            return
        reservation_id, turn_reference, tool_use_id, decision = (
            record.get("reservation_id"),
            record.get("turn_reference"),
            record.get("tool_use_id"),
            record.get("decision"),
        )
        if (
            not all(isinstance(value, str) and value.strip() == value and value for value in (reservation_id, turn_reference, tool_use_id))
            or not isinstance(decision, Mapping)
            # A decline may carry the reason the reviewer gave for it, so the
            # worker can be told which side refused.  Nothing else is accepted.
            or not set(decision) <= {"decision", "reason"}
            or decision.get("decision") not in {"accept", "decline"}
            or not isinstance(decision.get("reason", ""), str)
        ):
            return
        state = self._reservations.get(reservation_id)
        routed = state.get("permission_turns", {}).get(tool_use_id) if state is not None else None
        if state is None or (routed or state.get("turn_reference")) != turn_reference:
            return
        pending = state.get("pending_permissions")
        future = pending.get(tool_use_id) if isinstance(pending, dict) else None
        if isinstance(future, asyncio.Future) and not future.done():
            future.set_result(
                {"decision": decision["decision"], "reason": decision.get("reason")}
            )

    def _resolve_tool_call(self, record: Mapping[str, Any]) -> None:
        """Deliver one control-plane tool answer to its pending SDK handler.

        Validated exactly as strictly as ``permission_response``: an answer
        that does not name a live reservation, its current turn, and one
        outstanding call id is dropped rather than guessed at.
        """

        if (
            record.get("v") != _VERSION
            or set(record) != {"v", "kind", "op", "reservation_id", "turn_reference", "call_id", "result"}
        ):
            return
        reservation_id, turn_reference, call_id, result = (
            record.get("reservation_id"),
            record.get("turn_reference"),
            record.get("call_id"),
            record.get("result"),
        )
        if (
            not all(isinstance(value, str) and value.strip() == value and value for value in (reservation_id, turn_reference, call_id))
            or not isinstance(result, Mapping)
            or set(result) != {"success", "value"}
            or not isinstance(result.get("success"), bool)
            or not isinstance(result.get("value"), Mapping)
        ):
            return
        state = self._reservations.get(reservation_id)
        pending_turns = state.get("pending_tool_turns") if isinstance(state, Mapping) else None
        expected_turn = (
            pending_turns.get(call_id, state.get("turn_reference"))
            if isinstance(pending_turns, Mapping) and isinstance(state, Mapping)
            else state.get("turn_reference") if isinstance(state, Mapping) else None
        )
        if state is None or expected_turn != turn_reference:
            return
        pending = state.get("pending_tool_calls")
        future = pending.get(call_id) if isinstance(pending, dict) else None
        if isinstance(future, asyncio.Future) and not future.done():
            future.set_result({"success": result["success"], "value": dict(result["value"])})

    async def dispatch(self, op: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if op == "initialize":
            workspace = Path(str(payload.get("workspace", ""))).resolve()
            if not workspace.is_dir():
                raise BridgeError("workspace is unavailable")
            if any(name in os.environ for name in _CREDENTIAL_OVERRIDES):
                raise BridgeError("credential override environment is forbidden")
            self._workspace = workspace
            self._load_sdk()
            requested_provider = payload.get("provider", "claude")
            if requested_provider == "zai":
                config = load_zai_provider()
                if config is None:
                    raise BridgeError("z.ai provider is not configured")
                self._sdk_environment = {
                    "ANTHROPIC_BASE_URL": ZAI_ENDPOINT,
                    "ANTHROPIC_AUTH_TOKEN": config.api_key,
                }
                return {
                    "provider": "zai",
                    "harness": "claude-agent-sdk",
                    "sdk_version": _expected_sdk_version(),
                    "tool_support": {"accepted": True, "requires_empty": False},
                    "credential_override_rejected": True,
                    "endpoint_attestation": {
                        "endpoint_owner": "z.ai",
                        "endpoint": ZAI_ENDPOINT,
                        "endpoint_kind": "named-non-anthropic",
                        "credential_owner": "z.ai provider",
                        "credential_kind": "provider-api-key",
                        "anthropic_subscription_credential": False,
                    },
                }
            if requested_provider != "claude":
                raise BridgeError("bridge provider is not supported")
            return {
                "provider": "claude",
                "harness": "claude-agent-sdk",
                "sdk_version": _expected_sdk_version(),
                "tool_support": {"accepted": True, "requires_empty": False},
                "credential_override_rejected": True,
            }
        if op == "close":
            await self._cancel_all()
            self._sdk_environment.clear()
            return {"closed": True}
        if op == "start_thread":
            thread_workspace = self._reservation_workspace(payload.get("workspace"))
            definitions = _validated_tools(payload.get("tools"))
            reviewer = _requested_reviewer(payload.get("requested_posture"))
            effort = _requested_effort(payload.get("effort", "high"))
            permission_mode = _requested_permission_mode(payload.get("permission_mode", "default"))
            reservation_id = payload.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id or reservation_id in self._reservations:
                raise BridgeError("thread request lacks a fresh local reservation")
            self._reservations[reservation_id] = self._reservation_state(
                reviewer, None, definitions, reservation_id=reservation_id, tool_inputs=payload.get("tool_inputs") is True
            )
            self._reservations[reservation_id]["effort"] = effort
            self._reservations[reservation_id]["permission_mode"] = permission_mode
            self._reservations[reservation_id]["workspace"] = thread_workspace
            client = self._sdk.ClaudeSDKClient(
                options=self._options(
                    model=str(payload.get("model", "")),
                    resume=None,
                    reservation_id=reservation_id,
                    definitions=definitions,
                    developer_instructions=payload.get("developer_instructions"),
                    workspace=thread_workspace,
                    effort=effort,
                    permission_mode=permission_mode,
                )
            )
            try:
                await client.connect(None)
                server_info = await client.get_server_info()
                if not isinstance(server_info, Mapping):
                    raise BridgeError("Claude connection lacks server capability evidence")
            except BaseException:
                self._reservations.pop(reservation_id, None)
                await client.disconnect()
                raise
            self._reservations[reservation_id]["client"] = client
            identity = self._resolve_model_identity(
                self._reservations[reservation_id], str(payload.get("model", "")), server_info, previous=None,
            )
            return {
                "model_identity": identity,
                "reservation_echo": reservation_id,
                "policy": _policy(reviewer),
                "connection_evidence": {"connected": True, "server_info_received": True},
                "tool_registration": _registration(str(payload.get("model", "")), definitions),
                "effort": {"requested": effort, "applied": "sdk-option"},
                "permission_mode": permission_mode,
            }
        if op == "restart_thread":
            reservation_id = payload.get("reservation_id")
            previous = self._reservations.get(reservation_id) if isinstance(reservation_id, str) else None
            if previous is None or previous.get("release_requested") is not True:
                raise BridgeError("restart requires a released local reservation")
            task = previous.get("task")
            if isinstance(task, asyncio.Task) and not task.done():
                await asyncio.gather(task, return_exceptions=True)
            await self._release_agent({"reservation_id": reservation_id})
            self._reservations.pop(reservation_id)
            try:
                return await self.dispatch("start_thread", payload)
            except BaseException:
                self._reservations[reservation_id] = previous
                raise
        if op == "start_turn":
            return await self._start_turn(payload)
        if op == "can_start_turn":
            return self._can_start_turn(payload)
        if op == "wait_turn":
            return await self._wait_turn(payload)
        if op == "interrupt":
            return await self._interrupt(payload)
        if op == "compact":
            return await self._compact(payload)
        if op == "release_for_terminal":
            return await self._release_for_terminal(payload)
        if op == "release_agent":
            return await self._release_agent(payload)
        if op == "stop_native_task":
            return await self._stop_native_task(payload)
        if op == "resume":
            return await self._resume(payload)
        if op == "read_thread":
            return self._read_thread(payload)
        if op == "diagnostics":
            # This is deliberately aggregate-only.  It is used by the opt-in
            # live probe to distinguish a bridge that is still executing a
            # query from one that stopped receiving SDK messages; it must not
            # turn prompts, provider stderr, credentials, or opaque session
            # identities into diagnostic output.
            return self._diagnostics()
        if op == "steer":
            # The documented client can interrupt a query but cannot append a
            # prompt to an active one.  The runtime queues this steer.
            return {
                "reservation_echo": payload.get("reservation_id"),
                "accepted": False,
                "reason": "native-steer-not-supported",
            }
        raise BridgeError(f"unsupported Claude bridge operation: {op}")

    def _can_start_turn(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Report whether this reservation's sole SDK reader has drained.

        This is bridge-local JSONL state only.  It does not contact the
        provider or begin a turn; a scheduler uses it after a native child
        settles before it may reuse the parent reservation.
        """

        reservation_id, generation = payload.get("reservation_id"), payload.get("generation")
        if not isinstance(reservation_id, str) or not isinstance(generation, int):
            raise BridgeError("turn readiness request lacks local correlation")
        state = self._reservations.get(reservation_id)
        if state is None or state.get("generation") != generation:
            raise BridgeError("turn readiness request has stale reservation generation")
        task = state.get("task")
        if task is not None and not isinstance(task, asyncio.Task):
            raise BridgeError("Claude reservation has malformed reader state")
        return {
            "reservation_echo": reservation_id,
            "generation": generation,
            "ready": task is None and state.get("terminal_owner") is not True and state.get("client") is not None,
        }

    def _reservation_state(
        self, reviewer: str, session_id: str | None, definitions: Sequence[Mapping[str, Any]],
        *, reservation_id: str | None = None, tool_inputs: bool = False,
    ) -> dict[str, Any]:
        state = {
            "generation": 0,
            # Session opt-in (claude_tool_inputs) for the host: copy tool_use
            # arguments into message events.  Off by default; see
            # _project_content for why.
            "tool_inputs": tool_inputs is True,
            "session_id": session_id,
            # The turn's reader.  It owns this reservation's turn and may
            # remain active after a primary ResultMessage while exact native
            # child lifecycle frames are still in flight.  It reads frames
            # from ``consumer``, which the pump fills.
            "task": None,
            # The sole owner of ``client.receive_messages()`` for the life of
            # the connected client.  The CLI starts turns of its own between
            # vNext turns (a background task that finished queues a
            # task-notification), and the SDK stops reading stdout, hook
            # callbacks included, once 100 of those frames sit unread.  So the
            # stream is read from the first turn until disconnect, and a frame
            # with no turn to take it is recorded as an unsolicited turn.
            "pump": None,
            "stream": None,
            "consumer": None,
            "unsolicited": None,
            "unsolicited_serial": 0,
            "client": None,
            "release_requested": False,
            "release_task": None,
            "reviewer": reviewer,
            "turn_reference": None,
            # A primary turn is complete at its own ResultMessage, not when
            # the shared session reader drains native child lifecycle.  Keep
            # its immutable local correlation and waiter separately so Stop
            # never has to cancel the reader to release a primary waiter.
            "turn_outcomes": {},
            "pending_permissions": {},
            "permission_diagnostics": {},
            "pending_effects": [],
            "pending_messages": [],
            "dropped_stream_deltas": 0,
            "dropped_stream_deltas_reported": 0,
            "tool_effect_types": {},
            "tool_effect_paths": {},
            "tools": [dict(value) for value in definitions],
            "pending_tool_calls": {},
            "pending_tool_turns": {},
            "tool_call_serial": 0,
            "transcript": [],
            "terminal": None,
            "turn_started_monotonic": None,
            # The scheduler's per-turn budget, carried down so the PostToolUse
            # clock hook can say how much of it is gone.  A resume rebuilds
            # this whole dict, so both keys are absent until the next turn.
            "turn_timeout": None,
            "last_message_monotonic": None,
            "native_children": {},
            # One bridge reservation owns one ephemeral identity ledger.  It
            # has no persistence API: tokens and hook proofs must not outlive
            # this connected SDK session.
            "native_child_ledger": NativeChildIdentityLedger(),
            # This resolver retains only provider IDs.  Unlike the older
            # one-shot MCP enrollment ledger, it does not require a child to
            # follow an instruction before its lifecycle can be joined.
            "native_child_automatic_resolver": None,
            # Kept off in normal service operation. It exists only for old
            # captured sessions whose hook capability route must be replayed;
            # new native children use SDK metadata and never receive an
            # enrollment instruction in their prompt.
            "native_child_legacy_enrollment": False,
            "native_child_hook_sessions": set(),
            # These counters are deliberately aggregate-only.  They identify
            # which enrollment boundary ran in a live service without
            # retaining a prompt, SDK identifier, token, proof, or tool input.
            "native_child_registration": {
                "options_native_children_enabled": False,
                "options_registration_server_exposed": False,
                "options_pretool_hook_configured": False,
                "agent_hook_seen": 0,
                "agent_hook_incomplete": 0,
                "agent_hook_session_conflict": 0,
                "agent_rewrite_applied": 0,
                "agent_rewrite_rejected": 0,
                "registration_hook_seen": 0,
                "registration_hook_incomplete": 0,
                "registration_proof_injected": 0,
                "registration_proof_rejected": 0,
                "registration_permission_allowed": 0,
                "registration_permission_denied": 0,
                "registration_handler_called": 0,
                "registration_handler_accepted": 0,
                "registration_handler_rejected": 0,
                "task_started_seen": 0,
                "task_started_enrollment_found": 0,
                "task_started_joined": 0,
                "task_started_unmatched": 0,
                "subagent_start_hook_seen": 0,
                "subagent_start_hook_incomplete": 0,
                "subagent_start_hook_session_conflict": 0,
                "subagent_stop_hook_seen": 0,
                "subagent_stop_hook_incomplete": 0,
                "subagent_stop_hook_session_conflict": 0,
                "metadata_read_empty": 0,
                "metadata_read_matched": 0,
                "metadata_read_unavailable": 0,
                "metadata_read_error": 0,
                "coordination_metadata_retry": 0,
                "coordination_metadata_unresolved": 0,
                "automatic_identity_joined": 0,
                "agent_result_seen": 0,
                "agent_result_has_status": 0,
                "agent_result_has_agent_id": 0,
                "agent_result_has_task_id": 0,
                "agent_result_status_completed": 0,
                "agent_result_status_failed": 0,
                "agent_result_status_stopped": 0,
                "agent_result_status_other": 0,
            },
            # Parent turn provenance is fixed at the Agent tool-use hook. A
            # child can outlive its parent turn, so later lifecycle facts must
            # never borrow the reservation's mutable current turn.
            "native_child_origins": {},
            "native_child_observed_models": {},
            "native_child_observed_model_sources": {},
            # Forwarded SDK AssistantMessage frames carry the canonical child
            # model and the spawning Agent tool id.  Retain only that bounded
            # pair and immutable origin provenance until the independent
            # automatic identity joins resolve it.
            "native_child_forwarded_models": {},
            "native_child_pending_lifecycle": {},
            "native_child_identity_emitted": set(),
            "native_child_terminal_observed": set(),
            # SubagentStop is an authenticated observation that the child is
            # stopping. It is deliberately separate from a provider task
            # terminal result: a stop hook has no task status or outcome.
            "native_child_stop_observed": set(),
            "native_child_stop_emitted": set(),
            "native_child_metadata_retry_cursor": {},
            # The SDK client is deliberately disconnected during an idle
            # native-terminal lease.  A second SDK client must never race a
            # terminal which has resumed the same native Claude session.
            "terminal_owner": False,
        }

        def record_eviction(child_session: str, agent_id: str, pending: int) -> None:
            self._write_turn_event(state, {
                "name": "native_child",
                "reservation_id": reservation_id,
                "turn_reference": state.get("turn_reference"),
                "generation": state.get("generation"),
                "reported_session_id": child_session,
                "agent_id": agent_id,
                "status": "dropped-from-identity-join",
                "tracking": "unavailable",
                "note": (
                    f"native child {agent_id} dropped from the identity join: "
                    f"{pending} children were waiting for exact identity evidence"
                ),
            })
        state["native_child_automatic_resolver"] = NativeChildAutomaticResolver(on_eviction=record_eviction)
        return state

    def _diagnostics(self) -> Mapping[str, Any]:
        """Return a bounded, content-free view of bridge progress."""

        now = time.monotonic()
        states = tuple(self._reservations.values())
        active = [state for state in states if isinstance(state.get("task"), asyncio.Task) and not state["task"].done()]
        started = [state.get("turn_started_monotonic") for state in active]
        messages = [state.get("last_message_monotonic") for state in active]

        def oldest_age(values: Sequence[object]) -> float | None:
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            return round(max(0.0, now - min(numeric)), 3) if numeric else None

        permissions = {key: 0 for key in _PERMISSION_DIAGNOSTIC_KEYS}
        for state in states:
            counters = state.get("permission_diagnostics", {})
            for key in permissions:
                value = counters.get(key, 0)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    permissions[key] += value

        registration_keys = (
            "agent_hook_seen", "agent_hook_incomplete", "agent_hook_session_conflict",
            "agent_rewrite_applied", "agent_rewrite_rejected", "registration_hook_seen",
            "registration_hook_incomplete", "registration_proof_injected",
            "registration_proof_rejected", "registration_permission_allowed",
            "registration_permission_denied", "registration_handler_called",
            "registration_handler_accepted", "registration_handler_rejected",
            "task_started_seen", "task_started_enrollment_found", "task_started_joined",
            "task_started_unmatched",
            "subagent_start_hook_seen", "subagent_start_hook_incomplete",
            "subagent_start_hook_session_conflict", "subagent_stop_hook_seen",
            "subagent_stop_hook_incomplete", "subagent_stop_hook_session_conflict",
            "metadata_read_empty", "metadata_read_matched", "metadata_read_unavailable",
            "metadata_read_error", "coordination_metadata_retry", "coordination_metadata_unresolved",
            "automatic_identity_joined", "agent_result_seen",
            "agent_result_has_status", "agent_result_has_agent_id", "agent_result_has_task_id",
            "agent_result_status_completed", "agent_result_status_failed",
            "agent_result_status_stopped", "agent_result_status_other",
        )
        registration = {
            "enabled_reservation_count": 0,
            "registration_server_count": 0,
            "pretool_hook_count": 0,
            **{key: 0 for key in registration_keys},
        }
        for state in states:
            counters = state.get("native_child_registration")
            if not isinstance(counters, Mapping):
                continue
            registration["enabled_reservation_count"] += int(counters.get("options_native_children_enabled") is True)
            registration["registration_server_count"] += int(counters.get("options_registration_server_exposed") is True)
            registration["pretool_hook_count"] += int(counters.get("options_pretool_hook_configured") is True)
            for key in registration_keys:
                value = counters.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    registration[key] += value
        return {
            "reservation_count": len(states),
            "permissions": permissions,
            "active_turn_count": len(active),
            "bound_session_count": sum(_native_session_id(state.get("session_id")) for state in states),
            "pending_permission_count": sum(len(state.get("pending_permissions", {})) for state in states),
            "pending_tool_call_count": sum(len(state.get("pending_tool_calls", {})) for state in states),
            "pending_message_count": sum(len(state.get("pending_messages", ())) for state in states),
            "dropped_stream_delta_count": sum(
                _counter_value(state.get("dropped_stream_deltas")) for state in states
            ),
            "transcript_item_count": sum(len(state.get("transcript", ())) for state in states),
            "native_child_observed_model_count": sum(len(state.get("native_child_observed_models", {})) for state in states),
            "native_mirror_appends": sum(getattr(state.get("native_child_mirror"), "appends", 0) for state in states),
            "native_mirror_metadata_entries": sum(getattr(state.get("native_child_mirror"), "metadata_entries", 0) for state in states),
            "native_mirror_metadata_with_model": sum(getattr(state.get("native_child_mirror"), "metadata_with_model", 0) for state in states),
            "native_mirror_callback_errors": sum(getattr(state.get("native_child_mirror"), "callback_errors", 0) for state in states),
            "native_mirror_metadata_conflicts": sum(len(getattr(state.get("native_child_mirror"), "conflicted", ())) for state in states),
            "native_child_pending_lifecycle_count": sum(len(state.get("native_child_pending_lifecycle", {})) for state in states),
            "oldest_active_turn_age_seconds": oldest_age(started),
            "oldest_active_turn_last_message_age_seconds": oldest_age(messages),
            "native_child_registration": registration,
        }

    @staticmethod
    def _native_child_registration_signal(state: Mapping[str, Any] | None, signal: str) -> None:
        """Increment one fixed, content-free native-child enrollment counter."""

        if not isinstance(state, dict):
            return
        counters = state.get("native_child_registration")
        if not isinstance(counters, dict):
            return
        value = counters.get(signal)
        if isinstance(value, int) and not isinstance(value, bool):
            counters[signal] = value + 1

    def _load_sdk(self) -> None:
        if self._sdk is not None:
            return
        try:
            import claude_agent_sdk as sdk
        except ImportError as exc:
            # ImportError also fires when the SDK is there but something it
            # imports is not, and the old wording blamed the SDK for both.  A
            # run in another project reported "not installed" while pip and a
            # plain import both found 0.2.143, and the wrong word cost the
            # whole investigation.  Say what Python actually said.
            raise BridgeError(
                f"the Claude SDK (claude-agent-sdk=={CLAUDE_SDK_VERSION}) could not be imported: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # A side runtime installed by --update-runtimes names its own SDK
        # version; the pinned one is expected otherwise.
        expected = _expected_sdk_version()
        if getattr(sdk, "__version__", None) != expected:
            raise BridgeError(f"installed Claude SDK version is not exactly {expected}")
        self._sdk = sdk

    @staticmethod
    def _turn_outcomes(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
        outcomes = state.get("turn_outcomes")
        if outcomes is None:
            # Direct bridge-message tests predate the generation-scoped
            # protocol state.  Initialise once rather than weakening the
            # validation for a malformed live reservation.
            outcomes = state["turn_outcomes"] = {}
        if not isinstance(outcomes, dict):
            raise BridgeError("Claude reservation lacks local turn outcome state")
        return outcomes

    @classmethod
    def _turn_outcome(
        cls,
        state: dict[str, Any],
        generation: int,
    ) -> dict[str, Any] | None:
        outcome = cls._turn_outcomes(state).get(generation)
        return outcome if isinstance(outcome, dict) else None

    @classmethod
    def _turn_outcome_for_reference(
        cls,
        state: dict[str, Any],
        turn_reference: str,
    ) -> dict[str, Any] | None:
        matches = [
            outcome for outcome in cls._turn_outcomes(state).values()
            if isinstance(outcome, dict) and outcome.get("turn_reference") == turn_reference
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _interrupted_terminal(outcome: Mapping[str, Any], terminal: Mapping[str, Any]) -> bool:
        # The SDK's interrupt acknowledgement says only that the control
        # request was accepted.  It is a primary interruption only when the
        # corresponding provider ResultMessage confirms an abort boundary.
        return (
            outcome.get("interrupt_requested") is True
            and terminal.get("terminal_reason") in {"aborted_streaming", "aborted_tools"}
        )

    def _complete_turn_outcome(
        self,
        state: dict[str, Any],
        generation: int,
        terminal: Mapping[str, Any],
    ) -> None:
        outcome = self._turn_outcome(state, generation)
        if outcome is None:
            return
        prior = outcome.get("terminal")
        if isinstance(prior, Mapping):
            # The SDK can produce a later ResultMessage after a child settles
            # and wakes its parent for a follow-up turn.  Result frames expose
            # no public prompt/turn ID, so the first result is the only safe
            # primary outcome.  Continue reading for child lifecycle without
            # treating a later continuation result as corrupt correlation.
            return
        projected = dict(terminal)
        outcome["terminal"] = projected
        result = dict(projected)
        if self._interrupted_terminal(outcome, projected):
            result["status"] = "interrupted"
        elif projected.get("is_error") is True:
            result["status"] = "failed"
        else:
            result["status"] = "completed"
        hook_failures = self._hook_failure_summary(state)
        if hook_failures is not None:
            result["hook_failures"] = hook_failures
        outcome["result"] = result
        completion = outcome.get("completion")
        if isinstance(completion, asyncio.Future) and not completion.done():
            completion.set_result(dict(result))

    def _fail_turn_outcome(self, state: dict[str, Any], generation: int, exc: BridgeError) -> None:
        outcome = self._turn_outcome(state, generation)
        if outcome is None or isinstance(outcome.get("terminal"), Mapping):
            return
        completion = outcome.get("completion")
        if isinstance(completion, asyncio.Future) and not completion.done():
            completion.set_exception(exc)

    @staticmethod
    def _has_unresolved_native_children(state: Mapping[str, Any]) -> bool:
        children = state.get("native_children", {})
        pending = state.get("native_child_pending_lifecycle", {})
        active_children = (
            isinstance(children, Mapping)
            and any(isinstance(child, Mapping) and child.get("status") == "running" for child in children.values())
        )
        unresolved_pending = (
            isinstance(pending, Mapping)
            and any(isinstance(lifecycle, Mapping) and "terminal" not in lifecycle for lifecycle in pending.values())
        )
        return active_children or unresolved_pending

    async def _start_turn(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        reservation_id, turn_reference, generation = payload.get("reservation_id"), payload.get("turn_reference"), payload.get("generation")
        if not isinstance(reservation_id, str) or not isinstance(turn_reference, str) or not isinstance(generation, int):
            raise BridgeError("turn request lacks local correlation")
        state = self._reservations.get(reservation_id)
        if state is None or generation != state["generation"] + 1 or state["task"] is not None:
            raise BridgeError("turn request has stale reservation generation")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise BridgeError("turn prompt is invalid")
        if payload.get("effort", state.get("effort")) != state.get("effort"):
            raise BridgeError("turn effort drifted from the connected Claude session")
        state["generation"] = generation
        state["prior_turn_reference"] = state.get("turn_reference")
        state["turn_reference"] = turn_reference
        state["turn_started_monotonic"] = time.monotonic()
        state["hook_failures"] = []
        budget = payload.get("turn_timeout")
        state["turn_timeout"] = float(budget) if isinstance(budget, (int, float)) else None
        state["last_message_monotonic"] = None
        completion = asyncio.get_running_loop().create_future()
        # A provider stream can fail after the caller has already abandoned
        # its waiter.  Consume the Future's exception for asyncio's diagnostic
        # bookkeeping; awaiting it still raises the original BridgeError.
        completion.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        self._turn_outcomes(state)[generation] = {
            "turn_reference": turn_reference,
            "interrupt_requested": False,
            "completion": completion,
            # Set by an interrupt that lands before the prompt is sent.
            "interrupted": asyncio.Event(),
        }
        reader = asyncio.create_task(self._run_query(reservation_id, generation, prompt))
        # The completion Future carries a reader failure to the exact primary
        # waiter.  Retrieving the task exception here prevents an abandoned
        # primary waiter from producing a second unobserved-task warning.
        reader.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        state["task"] = reader
        return {
            "reservation_echo": reservation_id,
            "turn_echo": turn_reference,
            "turn_id": turn_reference,
            # `events_since` is inclusive, so expose the first cursor this
            # turn can emit rather than the last cursor already emitted.
            "cursor": self._event_cursor + 1,
        }

    async def _wait_turn(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        reservation_id = payload.get("reservation_id")
        if not isinstance(reservation_id, str) or reservation_id not in self._reservations:
            raise BridgeError("wait request lacks a local reservation")
        state = self._reservations[reservation_id]
        turn_reference = payload.get("turn_reference")
        outcome = self._turn_outcome_for_reference(state, turn_reference) if isinstance(turn_reference, str) else None
        if (
            not isinstance(turn_reference, str)
            or outcome is None
            or outcome.get("turn_reference") != turn_reference
        ):
            raise BridgeError("wait request has no active local turn")
        completion = outcome.get("completion")
        if not isinstance(completion, asyncio.Future):
            raise BridgeError("wait request has no active local turn")
        try:
            # A caller can abandon its wait without cancelling the shared
            # primary outcome or claiming that the provider was interrupted.
            result = await asyncio.shield(completion)
        except asyncio.CancelledError:
            raise
        except BridgeError:
            raise
        except Exception as exc:
            raise BridgeError(f"Claude query failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, Mapping):
            raise BridgeError("Claude turn completion is malformed")
        answer = dict(result)
        for key in ("model_ran", "model_ran_first"):
            value = state.get(key)
            if isinstance(value, str) and value:
                answer[key] = value
        return answer

    def _read_thread(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        reservation_id = payload.get("thread_id")
        after, limit = payload.get("after", 0), payload.get("limit", 100)
        if (
            not isinstance(reservation_id, str)
            or reservation_id not in self._reservations
            or not isinstance(after, int)
            or isinstance(after, bool)
            or after < 0
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise BridgeError("read thread lacks bounded local correlation")
        state = self._reservations[reservation_id]
        transcript = state.get("transcript")
        if not isinstance(transcript, list):
            raise BridgeError("Claude reservation lacks a transcript")
        items = [
            dict(item)
            for item in transcript
            if isinstance(item, Mapping) and item.get("sequence", 0) > after
        ]
        page = items[:limit]
        return {
            "reservation_echo": reservation_id,
            "items": page,
            "next_cursor": page[-1]["sequence"] if page else after,
        }

    async def _interrupt(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        reservation_id = payload.get("reservation_id")
        state = self._reservations.get(reservation_id) if isinstance(reservation_id, str) else None
        if state is None:
            raise BridgeError("interrupt request lacks a local reservation")
        generation = state.get("generation")
        outcome = self._turn_outcome(state, generation) if isinstance(generation, int) else None
        if (
            not isinstance(payload.get("turn_reference"), str)
            or outcome is None
            or outcome.get("turn_reference") != payload.get("turn_reference")
            or isinstance(outcome.get("terminal"), Mapping)
        ):
            raise BridgeError("interrupt request has no active local turn")
        task = state.get("task")
        if not isinstance(task, asyncio.Task) or task.done():
            raise BridgeError("interrupt request has no active local turn")
        # ``ClaudeSDKClient.interrupt`` is a streaming control request.  The
        # reader must continue: native children can still need hook/SDK-MCP
        # responses and report terminal lifecycle after the parent aborts.
        outcome["interrupt_requested"] = True
        try:
            # The reader can receive the provider's abort while this control
            # request is still awaiting its acknowledgement.
            await state["client"].interrupt()
        except BaseException:
            if not isinstance(outcome.get("terminal"), Mapping):
                outcome["interrupt_requested"] = False
            raise
        interrupted = outcome.get("interrupted")
        if isinstance(interrupted, asyncio.Event):
            # A turn still waiting for the CLI's own turn to end has sent
            # nothing; this ends it there.  The interrupt above stopped the
            # CLI's own turn, which is the work running now.
            interrupted.set()
        return {"reservation_echo": reservation_id, "interrupted": True}

    async def _compact(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compact an idle reservation's session by sending ``/compact``.

        claude-agent-sdk has no compact method.  The CLI runs the slash command
        when it arrives as a streaming prompt and answers with a
        ``compact_boundary`` SystemMessage (a live probe showed this; covered by
        ``test_compact_emits_the_boundary_and_the_result_usage_as_events``).  Only aggregate token counts are returned.
        """

        reservation_id = payload.get("reservation_id")
        state = self._reservations.get(reservation_id) if isinstance(reservation_id, str) else None
        if state is None:
            raise BridgeError("compact request lacks a local reservation")
        client = state.get("client")
        if (
            client is None
            or state.get("task") is not None
            or state.get("terminal_owner") is True
            or state.get("release_requested") is True
            # A turn the CLI started by itself is not idle, and its result
            # would end this read before /compact answers.
            or state.get("unsolicited") is not None
        ):
            raise BridgeError("compact requires an idle connected Claude reservation")
        # Hold the sole-reader slot so start_turn and can_start_turn see a busy
        # reservation while this request reads the client's message stream.
        state["task"] = asyncio.current_task()
        metadata: Mapping[str, Any] | None = None
        queue = self._open_consumer(state)
        try:
            await client.query("/compact")
            while True:
                try:
                    message = await self._next_frame(state, queue)
                except StopAsyncIteration:
                    break
                source = type(message).__name__
                if source == "SystemMessage" and getattr(message, "subtype", None) == "compact_boundary":
                    data = getattr(message, "data", None)
                    value = data.get("compact_metadata") if isinstance(data, Mapping) else None
                    metadata = value if isinstance(value, Mapping) else {}
                if source in {"SystemMessage", "ResultMessage"}:
                    # The same path a turn uses, so the host gets the boundary as
                    # provider.system and the result's usage update.  The
                    # previous turn already holds its terminal, so this result
                    # never completes a primary waiter.
                    self._emit_message(reservation_id, state["generation"], message)
                if source == "ResultMessage":
                    break
        finally:
            self._release_consumer(reservation_id, state, queue)
            state["task"] = None
        if metadata is None:
            raise BridgeError("Claude did not report a compact boundary")
        answer: dict[str, Any] = {"reservation_echo": reservation_id, "compacted": True}
        for key in ("trigger", "pre_tokens", "post_tokens"):
            value = metadata.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                answer[key] = value
        return answer

    async def _release_for_terminal(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Disconnect an *idle* SDK client before the CLI resumes its session.

        This is an ownership transfer, not a best-effort close.  The caller
        must name the exact attested native session and a live query, pending
        permission, or pending hosted MCP call makes transfer invalid.
        """

        reservation_id = payload.get("reservation_id")
        session_id = payload.get("session_id")
        if not isinstance(reservation_id, str) or not _native_session_id(session_id):
            raise BridgeError("terminal release lacks an attested local session")
        state = self._reservations.get(reservation_id)
        if state is None or state.get("session_id") != session_id:
            raise BridgeError("terminal release does not match a bound Claude session")
        if state.get("terminal_owner") is True:
            raise BridgeError("Claude session is already owned by a terminal")
        task = state.get("task")
        if isinstance(task, asyncio.Task) and not task.done():
            raise BridgeError("cannot release Claude session while an SDK turn is active")
        if state.get("pending_permissions") or state.get("pending_tool_calls"):
            raise BridgeError("cannot release Claude session with pending control callbacks")
        if state.get("unsolicited") is not None:
            raise BridgeError("cannot release Claude session while an SDK turn is active")
        client = state.get("client")
        if client is None:
            raise BridgeError("Claude terminal release lacks a connected SDK client")
        await self._stop_pump(state)
        await client.disconnect()
        state["client"] = None
        state["terminal_owner"] = True
        return {
            "reservation_echo": reservation_id,
            "session_id": session_id,
            "released": True,
        }

    async def _release_agent(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Release one settled agent after its sole SDK reader has drained.

        ``blocked`` belongs here with the four terminal statuses. It is not
        terminal: the agent keeps its attested native session and a later
        retry reconnects it. A message alone does not, since it only queues
        mail for a turn this agent cannot start. What blocked shares with the
        terminal statuses is that no turn is running and none can start until
        somebody outside decides something, so there is nothing for a
        connected client to do meanwhile.
        """

        reservation_id = payload.get("reservation_id")
        status = payload.get("status", "completed")
        if not isinstance(reservation_id, str) or not reservation_id:
            raise BridgeError("agent release lacks a local reservation")
        if status not in {"completed", "failed", "cancelled", "replaced", "blocked"}:
            raise BridgeError("agent release lacks a settled status")
        state = self._reservations.get(reservation_id)
        if state is None:
            raise BridgeError("agent release has no local reservation")
        if state.get("terminal_owner") is True:
            return {"reservation_echo": reservation_id, "released": False, "terminal_owner": True}
        state["release_requested"] = True
        task = state.get("task")
        if isinstance(task, asyncio.Task) and not task.done():
            if status == "completed":
                return {"reservation_echo": reservation_id, "released": False, "terminal_owner": False}
            if status == "blocked" and self._has_unresolved_native_children(state):
                # This reader is the only observer of the parent's native
                # children, and blocked is not terminal: cancelling it here
                # would throw away the terminal frame of a child that is still
                # running. ``release_requested`` stays set, so the reader
                # releases itself once those children settle.
                return {
                    "reservation_echo": reservation_id,
                    "released": False,
                    "terminal_owner": False,
                    "deferred": "unresolved-native-children",
                }
            # A cancelled or replaced agent may have no future ResultMessage.
            # Its vNext turn is terminal, so stop only this reservation's
            # reader before disconnecting the SDK client.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            outcomes = state.get("turn_outcomes")
            if isinstance(outcomes, Mapping):
                for outcome in outcomes.values():
                    completion = outcome.get("completion") if isinstance(outcome, Mapping) else None
                    if isinstance(completion, asyncio.Future) and not completion.done():
                        completion.cancel()
        await self._stop_pump(state)
        client = state.get("client")
        if client is not None:
            release_task = state.get("release_task")
            if not isinstance(release_task, asyncio.Task):
                release_task = asyncio.create_task(client.disconnect())
                state["release_task"] = release_task
            try:
                await release_task
            except BaseException:
                state["release_task"] = None
                raise
            state["client"] = None
        return {"reservation_echo": reservation_id, "released": True, "terminal_owner": False}

    async def _stop_native_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Stop one SDK-observed task by its provider-issued task ID.

        This is intentionally narrower than vNext child adoption.  The SDK
        documents `stop_task(task_id)` and a terminal notification, while it
        does not document delivery or session-resume control for that child.
        """

        reservation_id, task_id = payload.get("reservation_id"), payload.get("task_id")
        if not isinstance(reservation_id, str) or not isinstance(task_id, str) or not task_id:
            raise BridgeError("native task stop lacks provider task identity")
        state = self._reservations.get(reservation_id)
        children = state.get("native_children") if isinstance(state, Mapping) else None
        if not isinstance(children, Mapping) or task_id not in children:
            raise BridgeError("native task stop lacks an observed child task")
        client = state.get("client")
        if client is None or state.get("terminal_owner") is True:
            raise BridgeError("native task stop has no active SDK controller")
        await client.stop_task(task_id)
        self._write_turn_event(state, {
            "name": "native_child_control",
            "reservation_id": reservation_id,
            "turn_reference": state.get("turn_reference"),
            "generation": state.get("generation"),
            "task_id": task_id,
            "action": "stop-requested",
        })
        return {"reservation_echo": reservation_id, "task_id": task_id, "accepted": True}

    async def _resume(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        session_id = payload.get("session_id")
        if not _native_session_id(session_id):
            raise BridgeError("resume requires an attested native session identity")
        reservation_id = payload.get("reservation_id", session_id)
        if not isinstance(reservation_id, str) or not reservation_id:
            raise BridgeError("resume requires a local reservation")
        reviewer = _requested_reviewer(payload.get("requested_posture"))
        effort = _requested_effort(payload.get("effort", "high"))
        permission_mode = _requested_permission_mode(payload.get("permission_mode", "default"))
        definitions = _validated_tools(payload.get("tools"))
        previous = self._reservations.get(reservation_id)
        if previous is not None and previous.get("release_requested") is True:
            task = previous.get("task")
            if isinstance(task, asyncio.Task) and not task.done():
                await asyncio.gather(task, return_exceptions=True)
            await self._release_agent({"reservation_id": reservation_id})
        if previous is not None and previous.get("client") is not None:
            raise BridgeError("Claude resume requires the prior SDK client to be released")
        resume_workspace = self._reservation_workspace(payload.get("workspace"))
        self._reservations[reservation_id] = self._reservation_state(
            reviewer, session_id, definitions, reservation_id=reservation_id, tool_inputs=payload.get("tool_inputs") is True
        )
        self._reservations[reservation_id]["effort"] = effort
        self._reservations[reservation_id]["permission_mode"] = permission_mode
        self._reservations[reservation_id]["workspace"] = resume_workspace
        client = self._sdk.ClaudeSDKClient(
            options=self._options(
                model=str(payload.get("model", "")),
                resume=session_id,
                reservation_id=reservation_id,
                definitions=definitions,
                developer_instructions=payload.get("developer_instructions"),
                workspace=resume_workspace,
                effort=effort,
                permission_mode=permission_mode,
            )
        )
        try:
            await client.connect(None)
            server_info = await client.get_server_info()
            if not isinstance(server_info, Mapping):
                raise BridgeError("Claude resume lacks server capability evidence")
        except BaseException:
            if previous is None:
                self._reservations.pop(reservation_id, None)
            else:
                self._reservations[reservation_id] = previous
            await client.disconnect()
            raise
        self._reservations[reservation_id]["client"] = client
        self._reservations[reservation_id]["terminal_owner"] = False
        identity = self._resolve_model_identity(
            self._reservations[reservation_id], str(payload.get("model", "")), server_info, previous=previous,
        )
        return {
            "model_identity": identity,
            "provider_echo": True,
            "thread_id": session_id,
            "reservation_echo": reservation_id,
            "policy": _policy(reviewer),
            "connection_evidence": {"connected": True, "server_info_received": True},
            "tool_registration": _registration(str(payload.get("model", "")), definitions),
        }

    @staticmethod
    def _resolve_model_identity(
        state: dict[str, Any],
        alias: str,
        server_info: Mapping[str, Any],
        *,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Turn the configured alias into the exact id the CLI says it means.

        Read again on every connect, resume included, because an alias can
        move between two connects.  What it meant before stays in a history.
        """

        rows = server_info.get("models")
        exact, source = resolve_claude_alias(alias, rows)
        history: list[str] = []
        if isinstance(previous, Mapping):
            history = [str(value) for value in previous.get("model_exact_history") or () if value]
            before = previous.get("model_exact")
            if isinstance(before, str) and before and before != exact:
                history.append(before)
        state["model_exact"] = exact
        state["model_exact_source"] = source
        state["model_exact_history"] = history
        identity: dict[str, Any] = {
            "model_exact": exact,
            "model_exact_source": source,
            "models": model_table(rows),
        }
        if history:
            identity["model_exact_history"] = list(history)
        return identity

    def _stream(self, state: dict[str, Any]) -> Any:
        stream = state.get("stream")
        if stream is None:
            client = state.get("client")
            if client is None:
                raise BridgeError("Claude reservation lacks a connected client")
            stream = state["stream"] = client.receive_messages().__aiter__()
        return stream

    @staticmethod
    def _pump_alive(state: Mapping[str, Any]) -> bool:
        pump = state.get("pump")
        return isinstance(pump, asyncio.Task) and not pump.done()

    def _open_consumer(self, state: dict[str, Any]) -> asyncio.Queue[Any] | None:
        """Take the frames for one turn: from the pump, or straight off the stream.

        A turn reads the stream itself while no pump is reading it, which is
        the first turn of a connected client.  The pump starts on the same
        stream when that turn stops reading.
        """

        if self._pump_alive(state):
            queue: asyncio.Queue[Any] = asyncio.Queue()
            state["consumer"] = queue
            return queue
        if state.get("pump") is not None:
            # A pump that ended, or was cancelled before it ran, leaves no
            # stream this turn can trust to be where the pump stopped.
            state["pump"] = None
            state["stream"] = None
        self._stream(state)
        return None

    async def _next_frame(self, state: dict[str, Any], queue: asyncio.Queue[Any] | None) -> Any:
        """Return the turn's next frame; raise StopAsyncIteration at stream end."""

        if queue is None:
            return await self._stream(state).__anext__()
        message = await queue.get()
        if isinstance(message, _StreamEnd):
            error = message.error
            if error is not None:
                # The same sentence the turn reader writes for a stream that
                # fails under it directly.
                raise BridgeError(f"Claude query failed: {type(error).__name__}: {error}") from error
            raise StopAsyncIteration
        return message

    def _ensure_pump(self, reservation_id: str, state: dict[str, Any]) -> None:
        """Keep reading the stream after a turn stops, for the client's life."""

        if self._pump_alive(state) or state.get("client") is None:
            return
        stream = self._stream(state)
        task = asyncio.create_task(self._pump(reservation_id, state, stream))
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        state["pump"] = task

    async def _pump(self, reservation_id: str, state: dict[str, Any], stream: Any) -> None:
        error: Exception | None = None
        cancelled = False
        try:
            while True:
                try:
                    async for message in stream:
                        self._route_frame(reservation_id, state, message)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if type(exc).__name__ != "MessageParseError" or state.get("client") is None:
                        error = exc
                        break
                    # One frame the SDK could not parse ends the client's
                    # iterator, but not the SDK's read loop under it.  A
                    # reader that stopped here would fill the SDK buffer and
                    # stall the next hook, so it records the frame and reads
                    # on from a fresh iterator.
                    self._write_turn_event(state, {
                        "name": "unsolicited_turn",
                        "reservation_id": reservation_id,
                        "generation": state.get("generation"),
                        "after_turn_reference": state.get("turn_reference"),
                        "status": "frame-dropped",
                        "error": type(exc).__name__,
                    })
                    if state.get("stream") is stream:
                        state["stream"] = None
                    stream = self._stream(state)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            # An ended or cancelled stream cannot be read again; the next turn
            # asks the client for a fresh one, as each turn did before.
            if state.get("stream") is stream:
                state["stream"] = None
            consumer = state.get("consumer")
            if isinstance(consumer, asyncio.Queue):
                consumer.put_nowait(_StreamEnd(error))
            self._end_unsolicited(reservation_id, state, status="stream-ended")
            if not cancelled and state.get("release_requested") is not True:
                # Nothing reads this session any more: no hook, approval or
                # result of it will be answered.  A turn reading through the
                # pump fails on its own; one at rest has only this event.
                self._write_turn_event(state, {
                    "name": "unsolicited_turn",
                    "reservation_id": reservation_id,
                    "generation": state.get("generation"),
                    "after_turn_reference": state.get("turn_reference"),
                    "status": "reader-ended",
                    "error": type(error).__name__ if error is not None else None,
                })

    async def _stop_pump(self, state: dict[str, Any]) -> None:
        pump = state.get("pump")
        state["pump"] = None
        if isinstance(pump, asyncio.Task) and not pump.done() and pump is not asyncio.current_task():
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        state["stream"] = None

    def _release_consumer(
        self, reservation_id: str, state: dict[str, Any], queue: asyncio.Queue[Any] | None,
    ) -> None:
        """Hand frames the turn reader left unread back to the unsolicited path.

        A turn stops reading at its own result, and the pump can have queued
        the next turn's first frames by then.  Dropping them would lose the
        start of a turn the CLI began by itself.
        """

        if queue is not None and state.get("consumer") is queue:
            state["consumer"] = None
        while queue is not None and not queue.empty():
            message = queue.get_nowait()
            if not isinstance(message, _StreamEnd):
                self._route_frame(reservation_id, state, message)
        if state.get("client") is not None and state.get("release_requested") is not True:
            self._ensure_pump(reservation_id, state)

    def _route_frame(self, reservation_id: str, state: dict[str, Any], message: Any) -> None:
        consumer = state.get("consumer")
        if isinstance(consumer, asyncio.Queue):
            consumer.put_nowait(message)
            return
        self._record_unsolicited(reservation_id, state, message)

    def _record_unsolicited(self, reservation_id: str, state: dict[str, Any], message: Any) -> None:
        try:
            self._emit_unsolicited(reservation_id, state, message)
        except Exception as exc:
            # The pump must outlive one frame it cannot project: a pump that
            # stops is exactly the unread stream this reader exists to drain.
            self._write_turn_event(state, {
                "name": "unsolicited_turn",
                "reservation_id": reservation_id,
                "generation": state.get("generation"),
                "after_turn_reference": state.get("turn_reference"),
                "status": "frame-dropped",
                "error": type(exc).__name__,
            })

    def _emit_unsolicited(self, reservation_id: str, state: dict[str, Any], message: Any) -> None:
        """Record a frame that arrived while no vNext turn was reading.

        The turn opens at the first main-thread model output, because every
        model turn ends in a ResultMessage; a stray system frame between turns
        is recorded without opening one.
        """

        source = type(message).__name__
        turn = state.get("unsolicited")
        if (
            turn is None
            and source in {"AssistantMessage", "StreamEvent"}
            and getattr(message, "parent_tool_use_id", None) is None
        ):
            serial = _counter_value(state.get("unsolicited_serial")) + 1
            state["unsolicited_serial"] = serial
            turn = state["unsolicited"] = {
                "serial": serial, "frames": 0, "closed": asyncio.Event(),
                "after_turn_reference": state.get("turn_reference"),
            }
            self._write_turn_event(state, {
                "name": "unsolicited_turn",
                "reservation_id": reservation_id,
                "generation": state.get("generation"),
                "after_turn_reference": turn["after_turn_reference"],
                "unsolicited_serial": serial,
                "status": "started",
            })
        if not isinstance(turn, dict):
            self._emit_message(reservation_id, state["generation"], message, unsolicited=True)
            return
        turn["frames"] += 1
        # start_turn can open the next vNext turn while this one runs, and
        # _emit_message stamps events with the current reference.  The frames
        # belong to the turn this one followed, so stamp them with that.
        current = state.get("turn_reference")
        state["turn_reference"] = turn["after_turn_reference"]
        try:
            self._emit_message(reservation_id, state["generation"], message, unsolicited=True)
        finally:
            state["turn_reference"] = current
        if source == "ResultMessage":
            self._end_unsolicited(
                reservation_id, state, status="completed", is_error=getattr(message, "is_error", None) is True,
            )

    @staticmethod
    async def _await_unsolicited_close(state: dict[str, Any], interrupted: asyncio.Event | None = None) -> None:
        """Let a turn the CLI started by itself end before a vNext turn reads.

        Its ResultMessage carries no prompt correlation.  A vNext turn that
        registered while it was open would take that old result as its own,
        which is how a resumed worker's turn closed on a stale result.  The
        CLI queues a new prompt behind its running turn anyway, so waiting
        here delays nothing the CLI would have answered sooner.  Nothing
        awaits between the return and the caller registering its reader.
        An interrupt of the waiting turn ends the wait too.
        """

        while interrupted is None or not interrupted.is_set():
            turn = state.get("unsolicited")
            closed = turn.get("closed") if isinstance(turn, dict) else None
            if not isinstance(closed, asyncio.Event):
                return
            if interrupted is None:
                await closed.wait()
                continue
            waits = [asyncio.ensure_future(closed.wait()), asyncio.ensure_future(interrupted.wait())]
            try:
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for wait in waits:
                    wait.cancel()

    def _end_undispatched_turn(
        self, state: dict[str, Any], generation: int, reason: str = "interrupted_before_dispatch",
    ) -> None:
        """Settle a turn interrupted before the CLI started its prompt."""

        outcome = self._turn_outcome(state, generation)
        if outcome is None or isinstance(outcome.get("terminal"), Mapping):
            return
        terminal = self._terminal_projection(None)
        terminal["terminal_reason"] = reason
        outcome["terminal"] = terminal
        result = {**terminal, "status": "interrupted"}
        outcome["result"] = result
        completion = outcome.get("completion")
        if isinstance(completion, asyncio.Future) and not completion.done():
            completion.set_result(dict(result))

    def _end_unsolicited(
        self, reservation_id: str, state: dict[str, Any], *, status: str, is_error: bool = False,
    ) -> None:
        turn = state.get("unsolicited")
        if not isinstance(turn, dict):
            return
        state["unsolicited"] = None
        event: dict[str, Any] = {
            "name": "unsolicited_turn",
            "reservation_id": reservation_id,
            "generation": state.get("generation"),
            "after_turn_reference": turn.get("after_turn_reference"),
            "unsolicited_serial": turn.get("serial"),
            "status": status,
            "frames": turn.get("frames"),
            "is_error": is_error,
        }
        hook_failures = self._hook_failure_summary(turn)
        if hook_failures is not None:
            event["hook_failures"] = hook_failures
        self._write_turn_event(state, event)
        closed = turn.get("closed")
        if isinstance(closed, asyncio.Event):
            closed.set()

    async def _run_query(self, reservation_id: str, generation: int, prompt: str) -> None:
        if self._sdk is None or self._workspace is None:
            raise BridgeError("Claude bridge was not initialized")
        state = self._reservations[reservation_id]
        client = state.get("client")
        if client is None:
            raise BridgeError("Claude reservation lacks a connected client")
        counters_written = False
        queue: asyncio.Queue[Any] | None = None
        reading = False
        outcome = self._turn_outcome(state, generation)
        interrupted = outcome.get("interrupted") if outcome is not None else None
        waited = isinstance(state.get("unsolicited"), dict)
        try:
            await self._await_unsolicited_close(
                state, interrupted if isinstance(interrupted, asyncio.Event) else None,
            )
            if waited and outcome is not None and outcome.get("interrupt_requested") is True:
                # Interrupted while it waited behind the CLI's own turn: the
                # interrupt stopped that turn, and this prompt was never sent
                # and never will be.
                self._end_undispatched_turn(state, generation)
                return
            queue = self._open_consumer(state)
            reading = True
            # Under --replay-user-messages the CLI echoes this prompt's uuid
            # when it starts the prompt's turn, and only then.  A turn the CLI
            # started by itself may still be ahead of it with no frame read
            # yet; everything before the echo belongs to that turn.
            echo: str | None = None
            if _replays_prompts(client):
                echo = str(uuid.uuid4())
                await client.query(_echoed_prompt(prompt, echo))
            else:
                await client.query(prompt)
            # Only a CLI process that has echoed a prompt is waited on: a
            # version that takes the flag and never echoes would otherwise
            # hold the turn forever.  Until the first echo the turn keeps the
            # handling it had without the flag, which is safe there because
            # the CLI starts a turn by itself only after a turn of ours.
            # Echoing is fixed for the life of the process, so once seen, a
            # result before the echo is the CLI's own turn and waiting for
            # the echo cannot hang.
            await_echo = echo is not None and state.get("echoing_client") is client
            parent_finished = False
            # ``receive_response`` stops at the parent's ResultMessage.  This
            # shared reader instead remains responsible for exact child
            # lifecycle until the already-known children have settled.  An
            # error/aborted parent result is still a parent terminal, never a
            # reason to abandon its children's control and completion frames.
            while True:
                try:
                    message = await self._next_frame(state, queue)
                except StopAsyncIteration:
                    if await_echo and outcome is not None and outcome.get("interrupt_requested") is True:
                        self._end_undispatched_turn(state, generation, "interrupted_before_echo")
                        return
                    break
                if echo is not None:
                    if (
                        type(message).__name__ == "UserMessage"
                        and getattr(message, "uuid", None) == echo
                        and getattr(message, "parent_tool_use_id", None) is None
                    ):
                        echo = None
                        state["echoing_client"] = client
                        if await_echo:
                            # The CLI's own turn ended before this one started.
                            self._end_unsolicited(reservation_id, state, status="completed")
                        await_echo = False
                        continue
                    if await_echo:
                        # These frames follow the turn before this one.
                        current = state.get("turn_reference")
                        state["turn_reference"] = state.get("prior_turn_reference")
                        try:
                            self._record_unsolicited(reservation_id, state, message)
                        finally:
                            state["turn_reference"] = current
                        if (
                            type(message).__name__ == "ResultMessage"
                            and outcome is not None
                            and outcome.get("interrupt_requested") is True
                        ):
                            # The CLI drops a queued prompt on an interrupt,
                            # so its echo will never come: the abort result
                            # is the last frame this turn gets.
                            self._end_undispatched_turn(state, generation, "interrupted_before_echo")
                            return
                        continue
                terminal = type(message).__name__ == "ResultMessage"
                if terminal and not parent_finished:
                    self._settle_pending_native_children(reservation_id, generation, state)
                    # A ResultMessage settles the turn, and the run record
                    # stops collecting with it, so the snapshot has to be
                    # written before it rather than after.  The reader also
                    # stays on this stream while a child is unresolved, which
                    # can put the ``finally`` snapshot a long way off.
                    self._write_native_child_counters(reservation_id, state)
                    counters_written = True
                self._emit_message(reservation_id, generation, message)
                parent_finished = parent_finished or terminal
                if parent_finished and not self._has_unresolved_native_children(state):
                    break
            if not parent_finished:
                raise BridgeError("Claude message stream ended before turn terminal result")
            if not _native_session_id(state.get("session_id")):
                raise BridgeError("Claude turn completed before native session identity attested")
        except asyncio.CancelledError:
            raise
        except BridgeError as exc:
            self._fail_turn_outcome(state, generation, exc)
            raise
        except Exception as exc:
            # The type name alone drops the only sentence that says which
            # disagreement ended the turn, and the operator never sees the
            # traceback this bridge subprocess holds.
            bridge_error = BridgeError(f"Claude query failed: {type(exc).__name__}: {exc}")
            self._fail_turn_outcome(state, generation, bridge_error)
            raise BridgeError(f"Claude query failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if reading:
                self._release_consumer(reservation_id, state, queue)
            # An operator reading the run record could not tell a hook that
            # never arrived from a join that failed: the enrollment counters
            # lived only inside this process.  One content-free snapshot per
            # turn makes that difference readable without a diagnostics op.
            # A turn that reached its ResultMessage already wrote it there.
            if not counters_written:
                self._write_native_child_counters(reservation_id, state)
            # Start may create another primary only after this sole reader has
            # drained.  While native children remain active it deliberately
            # stays non-null: ResultMessage does not carry enough public
            # correlation to demultiplex a later user prompt from a child-
            # triggered follow-up result.
            if state.get("task") is asyncio.current_task():
                state["task"] = None
                if state.get("release_requested") is True:
                    await self._release_agent({"reservation_id": reservation_id})

    def _reservation_workspace(self, value: object) -> Path:
        """Resolve one reservation's root inside the connected workspace.

        A private Worker root is a directory beneath the workspace this bridge
        connected to.  Equality would refuse exactly that, so containment is the
        rule, and `resolve` is what enforces it against a link or a parent
        segment.  The directory has to exist already: the bridge creates
        nothing, so a path that is not there is a caller defect rather than
        something to paper over by making it.
        """

        if self._workspace is None:
            raise BridgeError("Claude bridge was not initialized")
        if value is None:
            return self._workspace
        resolved = Path(str(value)).resolve()
        if resolved != self._workspace and self._workspace not in resolved.parents:
            raise BridgeError("thread request violates the connected Claude posture")
        if not resolved.is_dir():
            raise BridgeError("thread workspace does not exist")
        return resolved

    def _options(
        self,
        *,
        model: str,
        resume: str | None,
        reservation_id: str,
        definitions: Sequence[Mapping[str, Any]] = (),
        developer_instructions: object = None,
        workspace: Path | None = None,
        effort: str | None = None,
        permission_mode: str = "default",
    ) -> Any:
        if self._sdk is None or self._workspace is None:
            raise BridgeError("Claude bridge was not initialized")
        hosted_names = {str(value["name"]) for value in definitions}
        state = self._reservations.get(reservation_id)
        ledger = state.get("native_child_ledger") if isinstance(state, Mapping) else None
        automatic = state.get("native_child_automatic_resolver") if isinstance(state, Mapping) else None
        native_children_enabled = bool(definitions) and isinstance(ledger, NativeChildIdentityLedger)
        legacy_enrollment_enabled = native_children_enabled and state.get("native_child_legacy_enrollment") is True
        registration = state.get("native_child_registration") if isinstance(state, dict) else None
        if isinstance(registration, dict):
            registration["options_native_children_enabled"] = native_children_enabled
            registration["options_registration_server_exposed"] = legacy_enrollment_enabled
            registration["options_pretool_hook_configured"] = native_children_enabled
        if isinstance(state, dict):
            state["native_child_metadata_directory"] = str(workspace or self._workspace)
            state["native_child_configured_model"] = model

        async def clock_hook(_data: Mapping[str, Any], _tool_use_id: str | None, _context: Any) -> Mapping[str, Any]:
            """Tell the worker the time and how much turn budget it has spent.

            Between a resume and the next ``start_turn`` the reservation dict
            has been rebuilt and carries no turn clock, so the line degrades to
            the wall-clock half alone.  It never raises: a clock is never worth
            a turn.
            """

            elapsed: float | None = None
            limit: float | None = None
            if isinstance(state, Mapping):
                started = state.get("turn_started_monotonic")
                if isinstance(started, (int, float)):
                    elapsed = max(0.0, time.monotonic() - float(started))
                budget = state.get("turn_timeout")
                if isinstance(budget, (int, float)):
                    limit = float(budget)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": mid_turn_line(time.time(), elapsed=elapsed, limit=limit),
                }
            }

        def deny_untracked_agent(tool_name: str, tool_use_id: object, detail: str) -> dict[str, Any]:
            """Refuse an Agent launch this bridge would not be able to track.

            An incomplete PreToolUse record used to answer {"continue": True}
            and leave a counter behind, while `relay_permission` allowed Agent
            on its own: the child ran with no recorded origin, so cancel_agent
            and parentage had no target for it.  The launch is refused here and
            leaves the same `native_child` evidence a refused launch leaves.
            """

            if isinstance(state, dict):
                try:
                    self._write_turn_event(
                        state,
                        {
                            "name": "native_child",
                            "reservation_id": reservation_id,
                            "turn_reference": state.get("turn_reference"),
                            "generation": state.get("generation"),
                            "tool_use_id": tool_use_id if isinstance(tool_use_id, str) else None,
                            "tool": tool_name,
                            "source": "native-child-hook",
                            "parent_tool_use_id": tool_use_id if isinstance(tool_use_id, str) else None,
                            "child_session_id": None,
                            "status": "blocked-untracked",
                            "tracking": "unavailable",
                            "detail": detail,
                        },
                    )
                except Exception:
                    # Evidence is never worth the turn, and the launch is
                    # refused either way.
                    pass
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"vNext could not track this native child: {detail}"
                    ),
                }
            }

        async def native_child_hook(data: Mapping[str, Any], tool_use_id: str | None, _context: Any) -> Mapping[str, Any]:
            """Attach SDK-owned child identity to the only supported MCP path."""

            tool_name, session_id = data.get("tool_name"), data.get("session_id")
            if not native_children_enabled or not isinstance(tool_name, str):
                return {"continue": True}
            is_agent_launch = tool_name in {"Agent", "Task"}
            is_registration = tool_name == _TOOL_NAMESPACE + _NATIVE_CHILD_REGISTER_TOOL
            if is_agent_launch:
                self._native_child_registration_signal(state, "agent_hook_seen")
            elif is_registration:
                self._native_child_registration_signal(state, "registration_hook_seen")
            # An empty or blank identity names nothing, so it counts as
            # missing and takes the audited denial path.
            if not _nonblank(session_id) or not _nonblank(tool_use_id):
                if is_agent_launch:
                    self._native_child_registration_signal(state, "agent_hook_incomplete")
                    return deny_untracked_agent(
                        tool_name,
                        tool_use_id,
                        "the launch record carries no session or tool-use identity",
                    )
                if is_registration:
                    self._native_child_registration_signal(state, "registration_hook_incomplete")
                return {"continue": True}
            current_session = state.get("session_id") if isinstance(state, Mapping) else None
            if _native_session_id(current_session) and current_session != session_id:
                if is_agent_launch:
                    self._native_child_registration_signal(state, "agent_hook_session_conflict")
                elif is_registration:
                    self._native_child_registration_signal(state, "registration_proof_rejected")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "native child hook session conflicts with reservation",
                    }
                }
            hook_sessions = state.get("native_child_hook_sessions") if isinstance(state, Mapping) else None
            if isinstance(hook_sessions, set):
                hook_sessions.add(session_id)
            input_data = data.get("tool_input")
            if not isinstance(input_data, Mapping):
                if is_agent_launch:
                    self._native_child_registration_signal(state, "agent_hook_incomplete")
                    return deny_untracked_agent(
                        tool_name,
                        tool_use_id,
                        "the launch record carries no readable tool input",
                    )
                if is_registration:
                    self._native_child_registration_signal(state, "registration_hook_incomplete")
                return {"continue": True}
            try:
                if is_agent_launch:
                    if legacy_enrollment_enabled and isinstance(data.get("agent_id"), str):
                        # The legacy capability ledger has no exact nested
                        # parent mapping. Keep its historical one-level route
                        # fail-closed; normal automatic mode supports nesting.
                        return {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": "legacy native child enrollment has no nested parent mapping",
                            }
                        }
                    if isinstance(automatic, NativeChildAutomaticResolver):
                        # A PreToolUse hook is authenticated provider origin.
                        # Its absent agent_id is the documented root context;
                        # that is distinct from the later metadata field,
                        # whose absence remains ambiguous until this record
                        # proves a root caller.
                        automatic.record_agent_origin(
                            session_id,
                            tool_use_id,
                            data.get("agent_id"),
                            parent_agent_id_present=True,
                            tool_input=input_data,
                        )
                    origins = state.get("native_child_origins") if isinstance(state, Mapping) else None
                    if not isinstance(origins, dict):
                        raise NativeChildIdentityError("native child origin state is unavailable")
                    origin_turn, origin_generation = state.get("turn_reference"), state.get("generation")
                    if not isinstance(origin_turn, str) or not origin_turn or not isinstance(origin_generation, int):
                        raise NativeChildIdentityError("native child origin lacks an active parent turn")
                    origins.setdefault(
                        (session_id, tool_use_id),
                        {
                            "turn_reference": origin_turn,
                            "generation": origin_generation,
                            "parent_native_agent_id": data.get("agent_id") if isinstance(data.get("agent_id"), str) else None,
                            "background_requested": input_data.get("run_in_background") if isinstance(input_data.get("run_in_background"), bool) else None,
                        },
                    )
                    if legacy_enrollment_enabled:
                        decision = ledger.rewrite_agent_input(
                            session_id,
                            tool_use_id,
                            input_data,
                            registration_tool_name=_TOOL_NAMESPACE + _NATIVE_CHILD_REGISTER_TOOL,
                        )
                        self._native_child_registration_signal(state, "agent_rewrite_applied")
                        return decision
                    return {"continue": True}
                if is_registration:
                    if not legacy_enrollment_enabled:
                        return {"continue": True}
                    agent_id = data.get("agent_id")
                    if not isinstance(agent_id, str):
                        raise NativeChildIdentityError("native child register hook lacks agent identity")
                    decision = ledger.rewrite_register_input(session_id, agent_id, tool_use_id, input_data)
                    self._native_child_registration_signal(state, "registration_proof_injected")
                    return decision
                if tool_name.startswith(_TOOL_NAMESPACE) and tool_name[len(_TOOL_NAMESPACE):] in hosted_names:
                    agent_id = data.get("agent_id")
                    if not isinstance(agent_id, str):
                        return {"continue": True}
                    # A child tool hook is authenticated SDK evidence for
                    # this exact session/agent pair. Refresh before the
                    # ledger consumes a scoped proof so a late first saved
                    # message can join without prompt-side enrollment.
                    if isinstance(automatic, NativeChildAutomaticResolver):
                        automatic.record_subagent_start(session_id, agent_id)
                        # The coordination proof is valid only after the same
                        # independently joined identity has produced its
                        # attested running lifecycle. A child with no explicit
                        # configured launch model must wait for a canonical
                        # child model frame; never borrow this reservation's
                        # configured model to make its tool call routable.
                        await self._refresh_child_for_coordination(
                            reservation_id, state, automatic, session_id, agent_id
                        )
                    return ledger.rewrite_coordination_input(
                        session_id,
                        agent_id,
                        tool_use_id,
                        input_data,
                        context_field=_NATIVE_CHILD_CONTEXT_FIELD,
                    )
            except NativeChildIdentityError as exc:
                if is_agent_launch:
                    self._native_child_registration_signal(state, "agent_rewrite_rejected")
                elif is_registration:
                    self._native_child_registration_signal(state, "registration_proof_rejected")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"native Agent call refused: {exc}",
                    }
                }
            return {"continue": True}

        async def native_subagent_start_hook(data: Mapping[str, Any], _tool_use_id: str | None, _context: Any) -> Mapping[str, Any]:
            """Read only SDK metadata for one exact SubagentStart identity.

            This is deliberately best effort: metadata that is not yet on
            disk produces no inferred child.  A later lifecycle or hook
            delivery may retry the same identifiers safely.
            """

            if not native_children_enabled or not isinstance(automatic, NativeChildAutomaticResolver):
                return {"continue": True}
            self._native_child_registration_signal(state, "subagent_start_hook_seen")
            session_id, agent_id = data.get("session_id"), data.get("agent_id")
            if not isinstance(session_id, str) or not isinstance(agent_id, str):
                self._native_child_registration_signal(state, "subagent_start_hook_incomplete")
                return {"continue": True}
            current_session = state.get("session_id") if isinstance(state, Mapping) else None
            if _native_session_id(current_session) and current_session != session_id:
                self._native_child_registration_signal(state, "subagent_start_hook_session_conflict")
                return {"continue": True}
            try:
                automatic.record_subagent_start(session_id, agent_id)
                self._refresh_automatic_native_metadata(state, automatic, session_id, agent_id)
                self._replay_automatic_native_children(
                    reservation_id, int(state.get("generation", 0)), state, automatic.resolve_ready()
                )
            except NativeChildIdentityError:
                self._native_child_registration_signal(state, "agent_rewrite_rejected")
            return {"continue": True}

        async def native_subagent_stop_hook(data: Mapping[str, Any], _tool_use_id: str | None, _context: Any) -> Mapping[str, Any]:
            """Refresh one authenticated child after a nonterminal stop hook.

            Stop observes an exact SDK child identity but does not carry a
            task outcome. Completion still requires a provider Task* terminal
            message correlated to the already joined task.
            """

            if not native_children_enabled or not isinstance(automatic, NativeChildAutomaticResolver):
                return {"continue": True}
            self._native_child_registration_signal(state, "subagent_stop_hook_seen")
            session_id, agent_id = data.get("session_id"), data.get("agent_id")
            if not isinstance(session_id, str) or not isinstance(agent_id, str):
                self._native_child_registration_signal(state, "subagent_stop_hook_incomplete")
                return {"continue": True}
            current_session = state.get("session_id") if isinstance(state, Mapping) else None
            if _native_session_id(current_session) and current_session != session_id:
                self._native_child_registration_signal(state, "subagent_stop_hook_session_conflict")
                return {"continue": True}
            try:
                automatic.record_subagent_start(session_id, agent_id)
                stops = state.get("native_child_stop_observed") if isinstance(state, Mapping) else None
                if isinstance(stops, set):
                    stops.add((session_id, agent_id))
                self._refresh_automatic_native_metadata(state, automatic, session_id, agent_id)
                self._replay_automatic_native_children(
                    reservation_id, int(state.get("generation", 0)), state,
                    automatic.resolve_ready(),
                )
                try:
                    self._emit_attested_native_child_stop(
                        reservation_id, state, automatic.identity_for_agent(session_id, agent_id)
                    )
                except NativeChildIdentityError:
                    pass
            except NativeChildIdentityError:
                self._native_child_registration_signal(state, "agent_rewrite_rejected")
            return {"continue": True}


        def permission_signal(key: str) -> None:
            current = self._reservations.get(reservation_id)
            if isinstance(current, dict):
                counters = current.setdefault("permission_diagnostics", {})
                counters[key] = counters.get(key, 0) + 1

        async def relay_permission(tool_name: str, input_data: Mapping[str, Any], context: Any) -> Any:
            permission_signal("callback_count")
            if tool_name.startswith(_TOOL_NAMESPACE) and tool_name[len(_TOOL_NAMESPACE):] in hosted_names:
                permission_signal("hosted_callback_with_agent_id" if isinstance(getattr(context, "agent_id", None), str)
                                  else "hosted_callback_without_agent_id")
            tool_use_id = getattr(context, "tool_use_id", None)
            if not isinstance(tool_use_id, str) or not tool_use_id:
                permission_signal("denied_missing_tool_use_id")
                return self._sdk.PermissionResultDeny(message="missing provider tool_use_id", interrupt=True)
            state = self._reservations.get(reservation_id)
            if tool_name.startswith(_TOOL_NAMESPACE):
                # A caller tool is the control plane talking to itself.  It is
                # already routed and recorded by the scheduler, so it is
                # allowed here rather than denied - and it is deliberately NOT
                # sent to the approval reviewer or projected as an effect.
                bare_tool = tool_name[len(_TOOL_NAMESPACE):]
                if bare_tool in hosted_names or (legacy_enrollment_enabled and bare_tool == _NATIVE_CHILD_REGISTER_TOOL):
                    permission_signal("allowed_hosted")
                    if bare_tool == _NATIVE_CHILD_REGISTER_TOOL:
                        self._native_child_registration_signal(state, "registration_permission_allowed")
                    return self._sdk.PermissionResultAllow()
                if bare_tool == _NATIVE_CHILD_REGISTER_TOOL:
                    self._native_child_registration_signal(state, "registration_permission_denied")
                permission_signal("denied_unhosted")
                return self._sdk.PermissionResultDeny(message="tool is outside fixed leaf posture", interrupt=True)
            if tool_name == "ToolSearch" and hosted_names:
                permission_signal("allowed_discovery")
                # The SDK can defer our hosted controls until tool discovery.
                # Loading definitions is not invoking them: each resulting MCP
                # call still passes the exact hosted-name and identity checks.
                return self._sdk.PermissionResultAllow()
            if tool_name in _LOCAL_NO_EFFECT_TOOLS:
                permission_signal("allowed_no_effect")
                return self._sdk.PermissionResultAllow()
            if tool_name in {"Task", "Agent", "TaskOutput"} and native_children_enabled:
                # A backgrounded subagent answers with a task id, and TaskOutput
                # is how its parent reads the result.  Allowing the launch and
                # refusing the read left work billed that nobody could collect.
                return self._sdk.PermissionResultAllow()
            if tool_name in {"Task", "Agent", "TaskOutput", "TeamCreate", "SendMessage"}:
                # The SDK may expose native subagents, but their lifecycle has
                # no reservation, parentage or interrupt receipt in this
                # bridge.  Refuse the launch and leave explicit evidence for
                # the host rather than presenting an invisible child as tracked.
                self._write_turn_event(
                    state,
                    {
                        "name": "native_child",
                        "reservation_id": reservation_id,
                        "turn_reference": state.get("turn_reference"),
                        "generation": state.get("generation"),
                        "tool_use_id": tool_use_id,
                        "tool": tool_name,
                        "source": "native-tool-request",
                        "parent_tool_use_id": tool_use_id,
                        "child_session_id": None,
                        "status": "blocked-untracked",
                        "tracking": "unavailable",
                    },
                )
                if native_children_enabled:
                    # Launches are tracked here.  Continuing one and building a
                    # team are the two native paths with no tracking yet.
                    message = (
                        f"{tool_name} is not tracked by vNext; launch a new subagent "
                        "with the Agent tool or use the hosted delegation tools"
                    )
                else:
                    message = "native subagents are not tracked by vNext; use the hosted delegation tools"
                return self._sdk.PermissionResultDeny(message=message, interrupt=False)
            turn_reference = state.get("turn_reference") if isinstance(state, Mapping) else None
            task = state.get("task") if isinstance(state, Mapping) else None
            if isinstance(state, Mapping) and not isinstance(state.get("consumer"), asyncio.Queue) and self._pump_alive(state):
                # No vNext turn is reading, so this call belongs to a turn the
                # CLI started itself.  It goes to the same reviewer under the
                # turn that one followed; refusing it stopped the CLI's turn.
                unsolicited = state.get("unsolicited")
                if isinstance(unsolicited, dict):
                    turn_reference = unsolicited.get("after_turn_reference")
                task = state.get("pump")
            if not isinstance(turn_reference, str) or not turn_reference or not isinstance(task, asyncio.Task):
                return self._sdk.PermissionResultDeny(message="missing local approval routing handle", interrupt=True)
            pending = state.get("pending_permissions")
            if not isinstance(pending, dict) or tool_use_id in pending:
                return self._sdk.PermissionResultDeny(message="duplicate local approval request", interrupt=True)
            permission_signal("review_requested")
            decision: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
            pending[tool_use_id] = decision
            # The answer names the turn this request went out under, which
            # start_turn may have moved on from by the time it arrives.
            state.setdefault("permission_turns", {})[tool_use_id] = turn_reference
            event: dict[str, Any] = {
                "name": "permission",
                "reservation_id": reservation_id,
                "turn_reference": turn_reference,
                "generation": state["generation"],
                "tool_use_id": tool_use_id,
                # An effect is named only where this adapter has a neutral
                # meaning for it.  A tool left unnamed reaches the reviewer
                # with no effect, and the reviewer declines it, so this table
                # is what decides whether a standard Claude tool can run at
                # all.  Anything absent here fails closed on purpose.
                **_tool_effect(tool_name),
                # What the effect is about.  The reviewer used to be told the
                # effect and nothing else, so it approved a write with no file
                # named.  A path here is workspace-relative and a command is
                # truncated; neither carries file contents.
                **_approval_detail(
                    tool_name,
                    (state.get("workspace") if isinstance(state, Mapping) else None) or self._workspace,
                    input_data,
                ),
            }
            session_id = state.get("session_id")
            if _native_session_id(session_id):
                event["provider_correlation"] = {
                    "session": session_id,
                    "turn": turn_reference,
                    "request": tool_use_id,
                }
                event["correlation_attested"] = True
            else:
                event["routing_handle"] = {
                    "reservation_id": reservation_id,
                    "turn_reference": turn_reference,
                }
                event["correlation_attested"] = False
            self._write_turn_event(state, event)
            try:
                resolved = await decision
            finally:
                pending.pop(tool_use_id, None)
                state["permission_turns"].pop(tool_use_id, None)
            # A decision carries a reason beside it now.  A bare verdict still
            # arrives from anything that resolves this future directly, so it
            # is read as a decision with no reason given.
            answer: Mapping[str, Any] = (
                resolved if isinstance(resolved, Mapping) else {"decision": resolved}
            )
            if answer.get("decision") == "accept":
                permission_signal("review_accepted")
                return self._sdk.PermissionResultAllow()
            permission_signal("review_declined")
            # Who refused, in the reviewer's own words.  The fixed sentence
            # this used to print named the manager on every path, including
            # the ones where no manager had been asked, so a worker reported a
            # decision its manager never took.  A decline that arrives with no
            # reason names nobody.
            reason = answer.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = "vNext declined this approval and recorded no reason"
            # The worker keeps its turn and reads this message, so it can
            # drop the declined effect and still deliver its report with
            # everything it has already read and planned intact.
            return self._sdk.PermissionResultDeny(
                message=(
                    f"{reason.strip()}; asking again gets the same answer, so take"
                    " another route and report what the declined tool was needed for"
                ),
                interrupt=False,
            )

        allowed_tools: list[str] = []
        _assert_tools_not_auto_allowed(allowed_tools, definitions)
        mcp_servers: dict[str, Any] = {}
        if definitions:
            mcp_servers[_TOOL_SERVER_KEY] = self._tool_server(
                reservation_id, definitions, include_native_child_registration=legacy_enrollment_enabled
            )
        side_cli = os.environ.get(_CLI_PATH_ENV)
        options = self._sdk.ClaudeAgentOptions(
            cwd=str(workspace or self._workspace),
            model=model,
            # The side runtime's own Claude CLI, hash-checked by the server
            # before this bridge started.
            **({"cli_path": side_cli} if side_cli else {}),
            # None asks the installed Claude Code client for its normal tool
            # set.  `can_use_tool` below still mediates each invocation.
            tools=None,
            allowed_tools=allowed_tools,
            # Native child creation is refused in `relay_permission`, where
            # vNext emits an explicit untracked-child event.  This avoids a
            # silent loss of parentage while preserving normal Claude tools.
            disallowed_tools=[],
            setting_sources=["user", "project", "local"],
            skills="all",
            include_partial_messages=True,
            # The CLI echoes each prompt when its turn starts, which is the
            # only frame that tells a vNext turn from one the CLI started by
            # itself (_run_query).
            **({"extra_args": {_REPLAY_FLAG: None}}
               if "extra_args" in getattr(self._sdk.ClaudeAgentOptions, "__dataclass_fields__", {}) else {}),
            **({"max_buffer_size": _SDK_MAX_BUFFER_SIZE}
               if "max_buffer_size" in getattr(self._sdk.ClaudeAgentOptions, "__dataclass_fields__", {}) else {}),
            # These are provider observations, not an assertion that every
            # native child can already be adopted.  They give the bridge the
            # task IDs/parent links needed to decide that from evidence.
            include_hook_events=True,
            forward_subagent_text=True,
            # The clock is registered for every worker.  Native-child
            # bookkeeping still needs tool definitions and a ledger, so those
            # three stay gated.
            # Every hook here answers from local state at once, so a bounded
            # wait costs nothing when the bridge is healthy and caps what each
            # tool call loses when it is not (3-4 October: every call waited
            # out the CLI's default).  Approvals that wait on a person go
            # through can_use_tool, which is not a hook and keeps no bound.
            hooks={
                "PostToolUse": [self._sdk.HookMatcher(matcher=None, hooks=[clock_hook], timeout=_HOOK_TIMEOUT_SECONDS)],
                **({
                    "PreToolUse": [self._sdk.HookMatcher(
                        matcher=None, hooks=[native_child_hook], timeout=_HOOK_TIMEOUT_SECONDS,
                    )],
                    "SubagentStart": [self._sdk.HookMatcher(
                        matcher=None, hooks=[native_subagent_start_hook], timeout=_HOOK_TIMEOUT_SECONDS,
                    )],
                    "SubagentStop": [self._sdk.HookMatcher(
                        matcher=None, hooks=[native_subagent_stop_hook], timeout=_HOOK_TIMEOUT_SECONDS,
                    )],
                } if native_children_enabled else {}),
            },
            effort=effort,
            mcp_servers=mcp_servers,
            # Keep the user's normal configured MCP servers available.  The
            # vNext server is still explicitly named and its manager tools are
            # separately correlated by this bridge; strict mode would quietly
            # erase the normal Claude environment promised by setting_sources.
            strict_mcp_config=False,
            settings=_WORKER_SETTINGS,
            permission_mode=permission_mode,
            can_use_tool=relay_permission,
            resume=resume,
            **({"env": dict(self._sdk_environment)} if self._sdk_environment else {}),
            **self._native_mirror_options(state, reservation_id, resume, native_children_enabled),
            **self._system_prompt(developer_instructions),
        )
        return options

    def _native_mirror_options(self, state, reservation_id, resume, enabled):
        if not enabled or resume is not None or os.environ.get("VNEXT_CLAUDE_NATIVE_MIRROR") != "1":
            return {}

        async def appended(key):
            session = key.get("session_id")
            resolver = state.get("native_child_automatic_resolver")
            if not _native_session_id(session) or not isinstance(resolver, NativeChildAutomaticResolver):
                return
            if state.get("session_id") not in {None, session}:
                return
            pending = state.get("native_child_pending_lifecycle", {})
            candidates = tuple(dict.fromkeys((*resolver.pending_agents(session), *(
                identity.agent_id for identity in resolver.snapshot()
                if identity.parent_session_id == session and (session, identity.task_id) in pending
            ))))
            for agent in candidates[:_MAX_METADATA_REFRESH_PER_EVENT]:
                if mirror.is_conflicted(session, agent):
                    continue
                self._record_mirrored_agent_metadata(state, resolver, session, agent, mirror)
                messages = await self._sdk.get_subagent_messages_from_store(
                    mirror, session, agent, directory=state["native_child_metadata_directory"], limit=8,
                )
                self._record_automatic_native_metadata(state, resolver, session, agent, messages)
            self._replay_automatic_native_children(
                reservation_id, state["generation"], state, resolver.resolve_ready(),
            )

        mirror = NativeMetadataMirror(self._sdk, appended)
        state["native_child_mirror"] = mirror
        return {"session_store": mirror, "session_store_flush": "eager"}

    def _record_mirrored_agent_metadata(self, state, resolver, session, agent, mirror):
        metadata = mirror.metadata_for_agent(session, agent)
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("toolUseId"), str):
            return
        resolver.record_metadata(session, agent, metadata["toolUseId"], metadata.get("parentAgentId"))
        model = metadata.get("model")
        if not _native_model_agrees(state.get("native_child_configured_model"), model):
            return
        key = (session, agent)
        models = state["native_child_observed_models"]
        prior = models.get(key)
        if _native_model_conflicts(prior, model):
            raise NativeChildIdentityError("native child observed model conflicts with prior metadata")
        if _native_canonical_model(prior) and not _native_canonical_model(model):
            # Mirror metadata can still echo the launch alias after a canonical
            # id landed. Weaker evidence never overwrites stronger evidence.
            return
        models[key] = model
        state["native_child_observed_model_sources"].setdefault(key, "sdk-agent-metadata")

    @staticmethod
    def _system_prompt(developer_instructions: object) -> dict[str, Any]:
        """Carry the caller's role prompt into the session.

        A root or branch manager that never receives its role prompt cannot
        behave as one; the instructions used to be discarded at the adapter.
        They append to the harness preset rather than replacing it, which is
        the same relationship the Codex boundary gives `developerInstructions`.
        """

        if not isinstance(developer_instructions, str) or not developer_instructions.strip():
            return {}
        return {
            "system_prompt": {
                "type": "preset",
                "preset": "claude_code",
                "append": developer_instructions,
            }
        }

    @staticmethod
    def _native_context_definition(definition: Mapping[str, Any]) -> dict[str, Any]:
        """Declare the hook-injected child proof without changing caller tools."""

        updated = dict(definition)
        schema = dict(updated["inputSchema"])
        properties = dict(schema["properties"])
        properties[_NATIVE_CHILD_CONTEXT_FIELD] = {
            "type": "string",
            "description": "Internal vNext native-child context. Supplied by Claude hook only.",
        }
        schema["properties"] = properties
        updated["inputSchema"] = schema
        return updated

    def _tool_server(
        self,
        reservation_id: str,
        definitions: Sequence[Mapping[str, Any]],
        *,
        include_native_child_registration: bool,
    ) -> Any:
        """Host the caller's tools on one in-process SDK MCP server."""

        hosted = [
            self._sdk.tool(
                str(definition["name"]),
                str(definition["description"]),
                self._native_context_definition(definition)["inputSchema"],
            )(self._tool_handler(reservation_id, str(definition["name"])))
            for definition in definitions
        ]
        if include_native_child_registration:
            hosted.append(
                self._sdk.tool(
                    _NATIVE_CHILD_REGISTER_TOOL,
                    "Register the current native Agent child with vNext exactly once.",
                    {
                        "type": "object",
                        "properties": {
                            "enrollment_token": {"type": "string"},
                            # The child hook injects this after model schema
                            # validation; the endpoint refuses it if absent.
                            "registration_proof": {"type": "string"},
                        },
                        "required": ["enrollment_token"],
                        "additionalProperties": False,
                    },
                )(self._native_child_register_handler(reservation_id))
            )
        return self._sdk.create_sdk_mcp_server(_TOOL_SERVER_KEY, "1.0.0", tools=hosted)

    @staticmethod
    def _native_runtime_thread(identity: NativeChildIdentity) -> str:
        # Provider agent IDs are scoped to their parent session.  The native
        # session UUID is therefore part of the runtime key; the colon is a
        # literal separator, never a parsed provider value.
        return "claude-native:" + identity.parent_session_id + ":" + identity.agent_id

    def _native_identity_event(
        self,
        state: Mapping[str, Any],
        identity: NativeChildIdentity,
        *,
        reservation_id: str,
        source: str,
        status: str,
    ) -> dict[str, Any]:
        origins = state.get("native_child_origins")
        origin = origins.get((identity.parent_session_id, identity.parent_tool_use_id)) if isinstance(origins, Mapping) else None
        if not isinstance(origin, Mapping):
            raise BridgeError("native child event lacks immutable parent origin")
        parent_turn, parent_generation = origin.get("turn_reference"), origin.get("generation")
        if not isinstance(parent_turn, str) or not parent_turn or not isinstance(parent_generation, int):
            raise BridgeError("native child event has invalid immutable parent origin")
        event = {
            "reservation_id": reservation_id,
            # An enrolled child has its own canonical vNext turn.  The parent
            # SDK turn remains explicit for provider correlation and callback
            # routing, never silently substituted.
            "turn_reference": identity.task_id,
            "parent_turn_reference": parent_turn,
            "generation": parent_generation,
            "provider_correlation": {
                "session": identity.parent_session_id,
                "turn": parent_turn,
            },
            "correlation_attested": True,
            "source": source,
            "status": status,
            "tracking": "attested",
            "identity_source": identity.identity_source.replace("_", "-"),
            "native_agent_id": identity.agent_id,
            "native_runtime_thread_id": self._native_runtime_thread(identity),
            "parent_tool_use_id": identity.parent_tool_use_id,
            "task_id": identity.task_id,
            "task_contract": {
                "role": "worker",
                "role_source": "vnext-native-task-mapping",
                "objective": identity.objective,
                "requested_model": identity.requested_model,
                "background_requested": origin.get("background_requested"),
                # Never infer a parent effective model here.  The public
                # Agent input contract did not establish inheritance.
                "model_source": identity.model_source,
            },
        }
        if identity.parent_agent_id is not None:
            if identity.parent_task_id is None:
                raise BridgeError("nested native child lacks an exact parent task")
            event.update({
                "parent_native_agent_id": identity.parent_agent_id,
                "parent_runtime_thread_id": "claude-native:" + identity.parent_session_id + ":" + identity.parent_agent_id,
                "parent_native_task_id": identity.parent_task_id,
                "parent_native_turn_id": identity.parent_task_id,
                # The bridge-local correlation remains the immutable
                # authenticated Agent origin. The separate native fields are
                # what identify a nested provider parent task.
            })
        elif identity.identity_source == "saved_session_metadata":
            # Explicit null is emitted only after the exact PreToolUse root
            # origin closed the otherwise ambiguous metadata gap.
            event["parent_native_agent_id"] = None
        if identity.identity_source == "saved_session_metadata":
            # Timing applies to every automatically joined child, including
            # nested children whose exact parent fields were emitted above.
            terminal_observed = state.get("native_child_terminal_observed")
            stop_observed = state.get("native_child_stop_observed")
            observed_after_identity = (
                isinstance(terminal_observed, set)
                and (identity.parent_session_id, identity.task_id) in terminal_observed
            ) or (
                isinstance(stop_observed, set)
                and (identity.parent_session_id, identity.agent_id) in stop_observed
            )
            event["identity_resolution"] = (
                "after-terminal-observation" if observed_after_identity
                else "before-terminal-observation"
            )
        observed_models = state.get("native_child_observed_models", {})
        observed_model = observed_models.get((identity.parent_session_id, identity.agent_id))
        if isinstance(observed_model, str):
            event["task_contract"].update({
                "observed_model": observed_model,
                "observed_model_source": state.get("native_child_observed_model_sources", {}).get(
                    (identity.parent_session_id, identity.agent_id), "saved-child-assistant-message"),
            })
        return event

    async def _refresh_child_for_coordination(
        self,
        reservation_id: str,
        state: dict[str, Any],
        resolver: NativeChildAutomaticResolver,
        session_id: str,
        agent_id: str,
    ) -> None:
        """Boundedly reread authenticated child metadata before a scoped proof.

        This runs in the SDK hook coroutine, never in the adapter's JSONL
        reader. It only releases a coordination proof after replay emitted the
        independently attested running child lifecycle for the same identity.
        A still-unresolved child is denied by the existing ledger call below;
        no control-plane callback is emitted for it.
        """

        for attempt in range(_NATIVE_CHILD_COORDINATION_METADATA_ATTEMPTS):
            self._refresh_automatic_native_metadata(state, resolver, session_id, agent_id)
            self._replay_automatic_native_children(
                reservation_id, int(state.get("generation", 0)), state, resolver.resolve_ready()
            )
            try:
                identity = resolver.identity_for_agent(session_id, agent_id)
            except NativeChildIdentityError:
                identity = None
            if identity is not None and (
                (identity.parent_session_id, identity.task_id)
                in state.get("native_child_identity_emitted", set())
            ):
                return
            if attempt + 1 < _NATIVE_CHILD_COORDINATION_METADATA_ATTEMPTS:
                self._native_child_registration_signal(state, "coordination_metadata_retry")
                await asyncio.sleep(_NATIVE_CHILD_COORDINATION_METADATA_DELAY_SECONDS)
        self._native_child_registration_signal(state, "coordination_metadata_unresolved")

    def _refresh_automatic_native_metadata(
        self,
        state: Mapping[str, Any],
        resolver: NativeChildAutomaticResolver,
        session_id: str,
        agent_id: str,
    ) -> None:
        """Bounded, read-only metadata refresh for one exact child identity."""

        mirror = state.get("native_child_mirror")
        if isinstance(mirror, NativeMetadataMirror):
            if mirror.is_conflicted(session_id, agent_id):
                return
            self._record_mirrored_agent_metadata(state, resolver, session_id, agent_id, mirror)
        reader = getattr(self._sdk, "get_subagent_messages", None)
        directory = state.get("native_child_metadata_directory")
        if not callable(reader) or not isinstance(directory, str) or not directory:
            self._native_child_registration_signal(state, "metadata_read_unavailable")
            return
        try:
            messages = reader(session_id, agent_id, directory=directory, limit=8)
        except (OSError, ValueError):
            self._native_child_registration_signal(state, "metadata_read_error")
            return
        self._record_automatic_native_metadata(state, resolver, session_id, agent_id, messages)

    def _record_automatic_native_metadata(self, state, resolver, session_id, agent_id, messages):
        if not isinstance(messages, Sequence) or not messages:
            self._native_child_registration_signal(state, "metadata_read_empty")
            return
        first = messages[0]
        parent_tool = getattr(first, "parent_tool_use_id", None)
        parent_agent = getattr(first, "parent_agent_id", None)
        if isinstance(parent_tool, str) and parent_tool:
            resolver.record_metadata(session_id, agent_id, parent_tool, parent_agent)
            self._native_child_registration_signal(state, "metadata_read_matched")
            models = state.get("native_child_observed_models")
            for message in messages:
                payload = getattr(message, "message", None)
                if (getattr(message, "parent_tool_use_id", None) != parent_tool
                        or getattr(message, "parent_agent_id", None) != parent_agent
                        or not isinstance(payload, Mapping) or payload.get("role") != "assistant"):
                    continue
                model = payload.get("model")
                # Metadata commonly reports the configured alias before the
                # provider emits its canonical child AssistantMessage. An
                # alias is not effective-model evidence, and it is not a rival
                # observation either: when the reservation itself was launched
                # with an alias, the first canonical id is admitted and pinned.
                # A second, different canonical id for the same child stays a
                # sticky disagreement rather than a value to silently discard.
                if isinstance(models, dict) and _native_canonical_model(model):
                    key = (session_id, agent_id)
                    prior = models.get(key)
                    if _native_model_conflicts(prior, model):
                        raise NativeChildIdentityError("native child observed model conflicts with prior metadata")
                    if not _native_model_agrees(state.get("native_child_configured_model"), model):
                        raise NativeChildIdentityError("native child observed model conflicts with configured model")
                    if prior != model:
                        sources = state.get("native_child_observed_model_sources")
                        if isinstance(sources, dict):
                            sources[key] = "saved-child-assistant-message"
                    models[key] = model
        else:
            self._native_child_registration_signal(state, "metadata_read_empty")

    def _record_forwarded_child_assistant_model(
        self, state: dict[str, Any], session_id: object, parent_tool_use_id: object, model: object,
    ) -> None:
        """Retain a typed forwarded child model only under its exact SDK origin."""

        if not all(isinstance(value, str) and value for value in (session_id, parent_tool_use_id, model)):
            return
        origins = state.get("native_child_origins")
        origin = origins.get((session_id, parent_tool_use_id)) if isinstance(origins, Mapping) else None
        if not isinstance(origin, Mapping):
            return
        # Claude's configured aliases (for example ``opus``) are not observed
        # effective models. A typed full provider identifier is the only
        # candidate that can be retained, and it must still agree with the
        # configured model before it is promoted below.
        if not model.startswith("claude-"):
            return
        forwarded = state.get("native_child_forwarded_models")
        if not isinstance(forwarded, dict):
            raise BridgeError("Claude reservation lacks forwarded child model state")
        key = (session_id, parent_tool_use_id)
        prior = forwarded.get(key)
        prior_model = prior.get("model") if isinstance(prior, Mapping) else None
        if prior is not None and prior_model != model:
            raise NativeChildIdentityError("forwarded child effective model conflicts with prior evidence")
        # A resolved identity has a durable observed-model cache. Compare
        # later canonical frames with it before considering a new pending
        # record, then return: sequential children must not consume a
        # lifetime pending-model slot after they were already promoted.
        resolver = state.get("native_child_automatic_resolver")
        if isinstance(resolver, NativeChildAutomaticResolver):
            identity = next((item for item in resolver.snapshot() if (
                item.parent_session_id == session_id
                and item.parent_tool_use_id == parent_tool_use_id
            )), None)
            if identity is not None:
                models = state.get("native_child_observed_models")
                observed = models.get((session_id, identity.agent_id)) if isinstance(models, Mapping) else None
                if _native_model_conflicts(observed, model):
                    raise NativeChildIdentityError("forwarded child effective model conflicts with prior evidence")
                if observed == model:
                    return
        if prior is None and len(forwarded) >= _MAX_PENDING_EFFECTS:
            raise BridgeError("Claude forwarded child model capacity exceeded")
        if not _native_model_agrees(state.get("native_child_configured_model"), model):
            raise NativeChildIdentityError("forwarded child effective model is not the configured model")
        turn_reference, generation = origin.get("turn_reference"), origin.get("generation")
        if not isinstance(turn_reference, str) or not turn_reference or not isinstance(generation, int):
            raise BridgeError("native child event has invalid immutable parent origin")
        if prior is None:
            forwarded[key] = {
                "model": model,
                "turn_reference": turn_reference,
                "generation": generation,
            }

    @staticmethod
    def _apply_forwarded_child_assistant_model(
        state: dict[str, Any], identity: NativeChildIdentity,
    ) -> None:
        """Promote a retained model only after the existing three-way identity join."""

        forwarded = state.get("native_child_forwarded_models")
        if not isinstance(forwarded, Mapping):
            raise BridgeError("Claude reservation lacks forwarded child model state")
        key = (identity.parent_session_id, identity.parent_tool_use_id)
        candidate = forwarded.get(key)
        if candidate is None:
            return
        if not isinstance(candidate, Mapping):
            raise BridgeError("Claude forwarded child model state is malformed")
        model = candidate.get("model")
        if not _native_model_agrees(state.get("native_child_configured_model"), model):
            raise NativeChildIdentityError("forwarded child effective model is not the configured model")
        origins = state.get("native_child_origins")
        origin = origins.get(key) if isinstance(origins, Mapping) else None
        if not isinstance(origin, Mapping):
            raise BridgeError("native child event lacks immutable parent origin")
        origin_turn, origin_generation = origin.get("turn_reference"), origin.get("generation")
        if not isinstance(origin_turn, str) or not origin_turn or not isinstance(origin_generation, int):
            raise BridgeError("native child event has invalid immutable parent origin")
        if (
            candidate.get("turn_reference") != origin_turn
            or candidate.get("generation") != origin_generation
        ):
            raise NativeChildIdentityError("forwarded child effective model has stale parent origin")
        models = state.get("native_child_observed_models")
        sources = state.get("native_child_observed_model_sources")
        if not isinstance(models, dict) or not isinstance(sources, dict):
            raise BridgeError("Claude reservation lacks observed child model state")
        observed_key = (identity.parent_session_id, identity.agent_id)
        prior = models.get(observed_key)
        if _native_model_conflicts(prior, model):
            raise NativeChildIdentityError("forwarded child effective model conflicts with prior evidence")
        if prior is None or not _native_canonical_model(prior):
            models[observed_key] = model
            sources[observed_key] = "forwarded-child-assistant-message"
        # The durable observed-model cache now detects any later conflicting
        # canonical evidence. Removing this pair keeps pending capacity about
        # unresolved joins rather than the lifetime number of children.
        if isinstance(forwarded, dict):
            forwarded.pop(key, None)

    def _refresh_pending_automatic_metadata(
        self,
        state: dict[str, Any],
        resolver: NativeChildAutomaticResolver,
        session_id: str,
    ) -> None:
        """Fairly bound retry of unresolved, exact SubagentStart identities."""

        pending = state.get("native_child_pending_lifecycle", {})
        candidates = tuple(dict.fromkeys((*resolver.pending_agents(session_id), *(
            identity.agent_id for identity in resolver.snapshot()
            if identity.parent_session_id == session_id and (session_id, identity.task_id) in pending
        ))))
        if not candidates:
            return
        cursors = state.get("native_child_metadata_retry_cursor")
        if not isinstance(cursors, dict):
            raise BridgeError("Claude reservation lacks metadata retry state")
        start = cursors.get(session_id, 0)
        if not isinstance(start, int) or start < 0:
            raise BridgeError("Claude metadata retry cursor is malformed")
        count = min(len(candidates), _MAX_METADATA_REFRESH_PER_EVENT)
        for offset in range(count):
            self._refresh_automatic_native_metadata(
                state, resolver, session_id, candidates[(start + offset) % len(candidates)]
            )
        cursors[session_id] = (start + count) % len(candidates)

    def _settle_pending_native_children(
        self, reservation_id: str, generation: int, state: dict[str, Any]
    ) -> None:
        """Read metadata once more for every still-pending child, then replay.

        The three refresh points are all earlier than the child's saved
        transcript: SubagentStart and TaskStarted run before the child has
        written anything, and SubagentStop can still race the file.  A child
        whose only read came back empty then stayed pending forever, so its
        outcomes row never said ``in_parent``.  A parent turn cannot end
        before its Agent tool answered, which makes the parent terminal the
        first moment the saved child transcript is certain to be readable.
        """

        automatic = state.get("native_child_automatic_resolver")
        if not isinstance(automatic, NativeChildAutomaticResolver):
            return
        sessions: list[str] = []
        hook_sessions = state.get("native_child_hook_sessions")
        if isinstance(hook_sessions, set):
            sessions.extend(value for value in hook_sessions if isinstance(value, str) and value)
        session_id = state.get("session_id")
        if _native_session_id(session_id) and session_id not in sessions:
            sessions.append(str(session_id))
        for session in dict.fromkeys(sessions):
            try:
                candidates = list(automatic.pending_agents(session))
                for agent_id in self._saved_session_child_ids(state, session):
                    # The saved session index is the same authenticated source
                    # as the metadata read.  A SubagentStart hook that never
                    # arrived left the resolver with no child at all, and a
                    # child it has never heard of can never join.
                    automatic.record_subagent_start(session, agent_id)
                    if agent_id not in candidates:
                        candidates.append(agent_id)
                for agent_id in candidates[:_MAX_PENDING_NATIVE_CHILDREN]:
                    self._refresh_automatic_native_metadata(state, automatic, session, agent_id)
                self._replay_automatic_native_children(
                    reservation_id, generation, state, automatic.resolve_ready()
                )
            except NativeChildIdentityError:
                self._native_child_registration_signal(state, "agent_rewrite_rejected")

    def _saved_session_child_ids(self, state: Mapping[str, Any], session_id: str) -> tuple[str, ...]:
        """List the saved child agent IDs of one session, or nothing."""

        lister = getattr(self._sdk, "list_subagents", None)
        directory = state.get("native_child_metadata_directory")
        if not callable(lister) or not isinstance(directory, str) or not directory:
            self._native_child_registration_signal(state, "metadata_read_unavailable")
            return ()
        try:
            listed = lister(session_id, directory=directory)
        except (OSError, ValueError):
            self._native_child_registration_signal(state, "metadata_read_error")
            return ()
        if not isinstance(listed, Sequence):
            return ()
        return tuple(value for value in listed if isinstance(value, str) and value)

    def _replay_automatic_native_children(
        self,
        reservation_id: str,
        generation: int,
        state: dict[str, Any],
        identities: Sequence[NativeChildIdentity],
    ) -> None:
        """Emit newly resolved native lifecycles after all exact joins exist."""

        pending = state.get("native_child_pending_lifecycle")
        emitted = state.get("native_child_identity_emitted")
        terminal_observed = state.get("native_child_terminal_observed")
        stop_observed = state.get("native_child_stop_observed")
        stop_emitted = state.get("native_child_stop_emitted")
        if (
            not isinstance(pending, Mapping) or not isinstance(emitted, set)
            or not isinstance(terminal_observed, set) or not isinstance(stop_observed, set)
            or not isinstance(stop_emitted, set)
        ):
            raise BridgeError("Claude reservation lacks automatic native-child replay state")
        resolver = state.get("native_child_automatic_resolver")
        # Exact identity can precede the first child message with an effective
        # model. Keep its lifecycle pending and reconsider it on later hooks.
        candidates = list(identities)
        if isinstance(resolver, NativeChildAutomaticResolver):
            candidates.extend(identity for identity in resolver.snapshot() if identity not in candidates)
        for identity in candidates:
            if identity.task_id is None:
                continue
            self._apply_forwarded_child_assistant_model(state, identity)
            key = (identity.parent_session_id, identity.task_id)
            lifecycles = pending.get(key)
            if not isinstance(lifecycles, Mapping):
                continue
            observed_models = state.get("native_child_observed_models", {})
            observed_model = (
                observed_models.get((identity.parent_session_id, identity.agent_id))
                if isinstance(observed_models, Mapping) else None
            )
            # A model is an admission fact, never a fallback to this
            # reservation's configured value. The adapter independently
            # enforces this same selected model before it creates a managed
            # child task, so emitting a lifecycle without one can only race a
            # later proof-backed callback into a fatal rejection.
            selected_model = observed_model if isinstance(observed_model, str) else identity.requested_model
            if not _native_model_agrees(state.get("native_child_configured_model"), selected_model):
                continue
            ledger = state.get("native_child_ledger")
            if not isinstance(ledger, NativeChildIdentityLedger):
                raise BridgeError("Claude reservation lacks native child identity ledger")
            try:
                ledger.adopt_automatic_identity(identity)
            except NativeChildIdentityError as exc:
                raise BridgeError("automatic native child conflicts with hook identity ledger") from exc
            if key not in emitted:
                identity_event = self._native_identity_event(
                    state, identity, reservation_id=reservation_id, source="automatic-sdk-metadata", status="joined"
                )
                identity_event["name"] = "native_child_identity"
                self._write_turn_event(state, identity_event)
                emitted.add(key)
                self._native_child_registration_signal(state, "automatic_identity_joined")
            for lifecycle_kind in ("start", "usage", "latest", "terminal"):
                lifecycle = lifecycles.get(lifecycle_kind)
                if isinstance(lifecycle, Mapping):
                    self._emit_attested_native_child_lifecycle(
                        reservation_id, generation, state, identity, lifecycle
                    )
            self._emit_attested_native_child_stop(reservation_id, state, identity)
            # The durable native child record now owns this history. Retaining
            # it here would turn a pending bound into a lifetime child cap.
            pending.pop(key, None)

    def _emit_attested_native_child_stop(
        self,
        reservation_id: str,
        state: Mapping[str, Any],
        identity: NativeChildIdentity,
    ) -> None:
        """Emit one stop observation after the exact identity already joined."""

        stop_observed = state.get("native_child_stop_observed")
        stop_emitted = state.get("native_child_stop_emitted")
        if not isinstance(stop_observed, set) or not isinstance(stop_emitted, set):
            raise BridgeError("Claude reservation lacks automatic native-child replay state")
        stop_key = (identity.parent_session_id, identity.agent_id)
        task_key = (identity.parent_session_id, identity.task_id)
        if (stop_key not in stop_observed or task_key in stop_emitted
                or task_key not in state.get("native_child_identity_emitted", set())):
            return
        observed_stop = self._native_identity_event(
            state, identity, reservation_id=reservation_id,
            source="authenticated-subagent-stop", status="observed-stop",
        )
        observed_stop.update({"name": "native_child_stop", "stop_observed": True})
        self._write_turn_event(state, observed_stop)
        stop_emitted.add(task_key)

    def _emit_attested_native_child_lifecycle(
        self,
        reservation_id: str,
        generation: int,
        state: dict[str, Any],
        identity: NativeChildIdentity,
        lifecycle: Mapping[str, Any],
    ) -> None:
        """Project one sanitized lifecycle record for an exact child identity."""

        source = lifecycle.get("source")
        if not isinstance(source, str):
            raise BridgeError("native child lifecycle lacks a source")
        native_children = state.setdefault("native_children", {})
        if not isinstance(native_children, dict):
            raise BridgeError("Claude reservation lacks native child state")
        child = native_children.setdefault(
            identity.task_id,
            {
                "task_id": identity.task_id,
                "parent_tool_use_id": identity.parent_tool_use_id,
                "agent_id": identity.agent_id,
                "runtime_thread_id": self._native_runtime_thread(identity),
            },
        )
        if source == "TaskStartedMessage":
            if child.get("terminal_status") is not None:
                # A stale start cannot reopen an attested terminal task.
                return
            child["status"] = "running"
            event = self._native_identity_event(
                state, identity, reservation_id=reservation_id, source=source, status="running"
            )
            event["name"] = "native_child"
            self._write_turn_event(state, event)
        usage = lifecycle.get("usage")
        status = lifecycle.get("status")
        if isinstance(usage, Mapping):
            usage_event = self._native_identity_event(
                state, identity, reservation_id=reservation_id, source=source,
                status=status if isinstance(status, str) else "running",
            )
            usage_event.update({"name": "native_child_usage", "usage": dict(usage), "usage_source": "native_task"})
            self._write_turn_event(state, usage_event)
        if status in {"completed", "failed", "stopped", "killed"}:
            prior_terminal = child.get("terminal_status")
            if prior_terminal is not None:
                if not self._same_native_child_terminal_status(prior_terminal, status):
                    raise BridgeError("Claude native child terminal status conflicts with prior terminal")
                return
            child["terminal_status"] = status
            child["status"] = status
            completed = self._native_identity_event(
                state, identity, reservation_id=reservation_id, source=source, status=status
            )
            completed["name"] = "native_child_completed"
            summary = lifecycle.get("summary")
            if isinstance(summary, str):
                completed["summary"] = summary
            if isinstance(usage, Mapping):
                completed["usage"] = dict(usage)
                completed["usage_source"] = "native_task"
            self._write_turn_event(state, completed)

    @staticmethod
    def _same_native_child_terminal_status(first: object, later: object) -> bool:
        """Return whether two provider terminal facts describe one stop.

        Claude's background task notifications distinguish ``stopped`` and
        ``killed`` in different message families, while vNext maps both to
        the same managed interruption outcome.  Keep the first raw provider
        status for auditability and accept only that one equivalent pair.
        """

        return (
            isinstance(first, str)
            and isinstance(later, str)
            and (first == later or {first, later} == {"stopped", "killed"})
        )

    def _native_child_register_handler(self, reservation_id: str) -> Any:
        async def handler(args: Mapping[str, Any]) -> dict[str, Any]:
            state = self._reservations.get(reservation_id)
            self._native_child_registration_signal(state, "registration_handler_called")
            ledger = state.get("native_child_ledger") if isinstance(state, Mapping) else None
            sessions = state.get("native_child_hook_sessions") if isinstance(state, Mapping) else None
            if not isinstance(ledger, NativeChildIdentityLedger) or not isinstance(sessions, set):
                self._native_child_registration_signal(state, "registration_handler_rejected")
                raise BridgeError("native child registration lacks enrollment hooks")
            identity: NativeChildIdentity | None = None
            for session_id in tuple(sessions):
                try:
                    identity = ledger.register_from_mcp(
                        session_id, args.get("enrollment_token"), args.get("registration_proof")
                    )
                except NativeChildIdentityError:
                    continue
                break
            if identity is None:
                self._native_child_registration_signal(state, "registration_handler_rejected")
                raise BridgeError("native child registration was not exactly attested")
            self._native_child_registration_signal(state, "registration_handler_accepted")
            if identity.task_id is not None:
                self._native_child_registration_signal(state, "task_started_joined")
            child = state.setdefault("native_children", {})
            if identity.task_id is not None:
                child[identity.task_id] = {
                    "task_id": identity.task_id,
                    "parent_tool_use_id": identity.parent_tool_use_id,
                    "agent_id": identity.agent_id,
                    "runtime_thread_id": self._native_runtime_thread(identity),
                    "status": "running",
                }
            event = self._native_identity_event(
                state, identity, reservation_id=reservation_id, source="hook-enrollment", status="registered"
            )
            event["name"] = "native_child_identity"
            self._write_turn_event(state, event)
            if identity.task_id is not None:
                task_event = self._native_identity_event(
                    state, identity, reservation_id=reservation_id, source="hook-enrollment", status="running"
                )
                task_event["name"] = "native_child"
                self._write_turn_event(state, task_event)
            return {
                "content": [{"type": "text", "text": "native child registered"}],
                "is_error": False,
            }

        return handler

    def _tool_handler(self, reservation_id: str, tool: str) -> Any:
        async def handler(args: Mapping[str, Any]) -> dict[str, Any]:
            arguments = dict(args)
            proof = arguments.pop(_NATIVE_CHILD_CONTEXT_FIELD, None)
            identity: NativeChildIdentity | None = None
            if proof is not None:
                state = self._reservations.get(reservation_id)
                ledger = state.get("native_child_ledger") if isinstance(state, Mapping) else None
                sessions = state.get("native_child_hook_sessions") if isinstance(state, Mapping) else None
                if not isinstance(ledger, NativeChildIdentityLedger) or not isinstance(sessions, set):
                    raise BridgeError("native child tool context lacks enrollment hooks")
                for session_id in tuple(sessions):
                    try:
                        identity = ledger.consume_coordination_context(session_id, proof)
                    except NativeChildIdentityError:
                        continue
                    break
                if identity is None:
                    raise BridgeError("native child tool context was not exactly attested")
            answer = await self._call_control_plane(reservation_id, tool, arguments, native_identity=identity)
            # The SDK drops content blocks it does not recognise, so the answer
            # travels as one JSON text block - the same text the Codex boundary
            # puts in its `contentItems`.
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(answer["value"], ensure_ascii=False, separators=(",", ":")),
                    }
                ],
                "is_error": not answer["success"],
            }

        return handler

    async def _call_control_plane(
        self,
        reservation_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        native_identity: NativeChildIdentity | None = None,
    ) -> dict[str, Any]:
        """Block this SDK tool handler until the control plane answers it."""

        state = self._reservations.get(reservation_id)
        turn_reference = state.get("turn_reference") if isinstance(state, Mapping) else None
        if state is None or not isinstance(turn_reference, str) or not turn_reference:
            raise BridgeError("manager tool call lacks a local reservation")
        pending = state.get("pending_tool_calls")
        if not isinstance(pending, dict):
            raise BridgeError("Claude reservation lacks bounded tool call state")
        if len(pending) >= _MAX_PENDING_EFFECTS:
            raise BridgeError("Claude manager tool call capacity exceeded")
        serial = int(state["tool_call_serial"])
        state["tool_call_serial"] = serial + 1
        call_id = "call-" + str(state["generation"]) + "-" + str(serial)
        answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        pending[call_id] = answer
        pending_turns = state.get("pending_tool_turns")
        if not isinstance(pending_turns, dict):
            raise BridgeError("Claude reservation lacks bounded tool turn state")
        event_turn = native_identity.task_id if native_identity is not None and native_identity.task_id is not None else turn_reference
        pending_turns[call_id] = event_turn
        event = {
            "name": "tool_call",
            "reservation_id": reservation_id,
            "turn_reference": event_turn,
            "generation": state["generation"],
            "call_id": call_id,
            "tool": tool,
            "arguments": dict(arguments),
        }
        if native_identity is None and isinstance(state.get("model_ran"), str):
            event["model_ran"] = state["model_ran"]
            event["model_ran_first"] = state.get("model_ran_first") or state["model_ran"]
        if native_identity is not None:
            event.update({
                "parent_turn_reference": turn_reference,
                "native_agent_id": native_identity.agent_id,
                "native_runtime_thread_id": self._native_runtime_thread(native_identity),
                "native_child_task_id": native_identity.task_id,
                "native_parent_tool_use_id": native_identity.parent_tool_use_id,
                "native_session_id": native_identity.parent_session_id,
            })
        _write(
            {
                "v": _VERSION,
                "kind": "event",
                "event": event,
            }
        )
        try:
            return await asyncio.wait_for(answer, _TOOL_CALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise BridgeError("Claude manager tool call timed out waiting for the control plane") from exc
        finally:
            pending.pop(call_id, None)
            pending_turns.pop(call_id, None)

    @staticmethod
    def _track_hook_failures(target: dict[str, Any], message: Any) -> None:
        """Keep the trailing run of tool results that failed in a hook.

        A result that did not fail in a hook ends the run: the model got past
        it.  What is left when the turn ends is what it could not get past.
        """

        if getattr(message, "parent_tool_use_id", None) is not None:
            return
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            line = _hook_failure_line(block)
            if line is None:
                target["hook_failures"] = []
            else:
                failures = target.get("hook_failures")
                if not isinstance(failures, list):
                    failures = target["hook_failures"] = []
                failures.append(line)

    @staticmethod
    def _hook_failure_summary(target: Mapping[str, Any]) -> dict[str, Any] | None:
        failures = target.get("hook_failures")
        if not isinstance(failures, list) or not failures:
            return None
        return {"count": len(failures), "first": failures[0]}

    def _emit_message(
        self, reservation_id: str, generation: int, message: Any, *, unsolicited: bool = False,
    ) -> None:
        state = self._reservations.get(reservation_id)
        if state is None:
            raise BridgeError("Claude emitted a stale local generation")
        source = type(message).__name__
        # Task lifecycle has an immutable parent-tool origin.  Route it before
        # the mutable primary-generation guard: a native child can outlive the
        # primary turn that created it, and its lifecycle must never be lost or
        # reassigned to a later primary generation.
        if source in {"TaskStartedMessage", "TaskProgressMessage", "TaskNotificationMessage", "TaskUpdatedMessage"}:
            state["last_message_monotonic"] = time.monotonic()
            self._emit_native_child_event(reservation_id, generation, state, message)
            return
        if state["generation"] != generation:
            raise BridgeError("Claude emitted a stale local generation")
        state["last_message_monotonic"] = time.monotonic()
        if source == "StreamEvent":
            self._emit_stream_event(reservation_id, generation, state, message)
            return
        if source == "SystemMessage":
            self._emit_system_message(reservation_id, generation, state, message)
            return
        if source not in {"AssistantMessage", "ResultMessage", "UserMessage"}:
            return
        if source == "UserMessage":
            self._observe_native_agent_result(state, message, reservation_id=reservation_id)
            turn = state.get("unsolicited")
            self._track_hook_failures(turn if unsolicited and isinstance(turn, dict) else state, message)
        if source == "AssistantMessage" and getattr(message, "parent_tool_use_id", None) is None:
            # The worker's own reply names the model that really answered.
            # A forwarded Agent-tool child carries a parent tool id and its
            # model is the child's, so it never sets this.
            ran = getattr(message, "model", None)
            if isinstance(ran, str) and ran:
                state["model_ran"] = ran
                state.setdefault("model_ran_first", ran)
        session_id = getattr(message, "session_id", None)
        if source in {"AssistantMessage", "ResultMessage"} and _native_session_id(session_id):
            current = state["session_id"]
            if current is not None and current != session_id:
                raise BridgeError("Claude emitted conflicting native session identities")
            state["session_id"] = session_id
            # The adapter publishes the next generation before the bridge
            # sees start_turn, so an identity event from an unsolicited turn
            # in that window would read as stale and end the adapter.  The
            # session was bound by the vNext turn that came before it.
            if not unsolicited:
                self._write_turn_event(
                    state,
                    {
                        "name": "result" if source == "ResultMessage" else "system",
                        "reservation_id": reservation_id,
                        "generation": generation,
                        "native_identity": {"session_id": session_id, "source": source},
                    },
                )
            self._flush_pending_effects(reservation_id, generation, state)
            self._flush_pending_messages(reservation_id, generation, state)
        if source == "AssistantMessage" and _native_session_id(state.get("session_id")):
            self._record_forwarded_child_assistant_model(
                state, session_id, getattr(message, "parent_tool_use_id", None),
                getattr(message, "model", None),
            )
        if source in {"AssistantMessage", "ResultMessage"} and _native_session_id(state.get("session_id")):
            resolver = state.get("native_child_automatic_resolver")
            if isinstance(resolver, NativeChildAutomaticResolver):
                self._refresh_pending_automatic_metadata(state, resolver, state["session_id"])
                self._replay_automatic_native_children(reservation_id, generation, state, resolver.resolve_ready())
        projected = self._project_message(message, source, tool_inputs=state.get("tool_inputs") is True)
        if projected is not None:
            transcript = state.get("transcript")
            if transcript is None:
                # Direct SDK-message tests and an interrupted old bridge can
                # predate this field.  Initialise it once; malformed values
                # remain a hard protocol defect.
                transcript = state["transcript"] = []
            if not isinstance(transcript, list):
                raise BridgeError("Claude reservation lacks a transcript")
            item = {"sequence": len(transcript) + 1, **projected}
            transcript.append(item)
            event: dict[str, Any] = {
                "name": "message",
                "reservation_id": reservation_id,
                "turn_reference": state.get("turn_reference"),
                "generation": generation,
                "message": projected,
            }
            self._release_or_hold_message(state, event)
        provider_error = self._provider_error_projection(message, source)
        if provider_error is not None:
            error_event: dict[str, Any] = {
                "name": "provider_error",
                "reservation_id": reservation_id,
                "turn_reference": state.get("turn_reference"),
                "generation": generation,
                "provider_error": provider_error,
            }
            if _native_session_id(state.get("session_id")):
                error_event["provider_correlation"] = {
                    "session": state["session_id"],
                    "turn": state.get("turn_reference"),
                }
                error_event["correlation_attested"] = True
            self._write_turn_event(state, error_event)
        if source == "ResultMessage":
            terminal = self._terminal_projection(message)
            if not unsolicited:
                state["terminal"] = terminal
            # A Result may carry the empty/default session sentinel before a
            # real provider identity is attested.  Let the reader turn that
            # into its exact failure before resolving the primary waiter.
            # The result of a turn the CLI started by itself answers no vNext
            # prompt: start_turn may already have opened the next generation,
            # and this result must not settle it.
            if not unsolicited and _native_session_id(state.get("session_id")):
                self._complete_turn_outcome(state, generation, terminal)
            usage_event = {
                "name": "usage",
                "reservation_id": reservation_id,
                "turn_reference": state.get("turn_reference"),
                "generation": generation,
                "usage": terminal.get("usage"),
                "total_cost_usd": terminal.get("total_cost_usd"),
                "model_usage": terminal.get("model_usage"),
            }
            if _native_session_id(state.get("session_id")):
                usage_event["provider_correlation"] = {
                    "session": state["session_id"],
                    "turn": state.get("turn_reference"),
                }
                usage_event["correlation_attested"] = True
            self._write_turn_event(state, usage_event)
        for block in self._content_blocks(message):
            descriptor = self._effect_descriptor(state, block)
            if descriptor is not None:
                self._release_or_hold_effect(reservation_id, generation, state, descriptor)

    def _emit_stream_event(self, reservation_id: str, generation: int, state: dict[str, Any], message: Any) -> None:
        raw = getattr(message, "event", None)
        if not isinstance(raw, Mapping):
            return
        delta = raw.get("delta")
        if not isinstance(delta, Mapping):
            return
        content: dict[str, Any] = {"type": str(delta.get("type") or "delta")}
        if isinstance(delta.get("text"), str):
            content["text"] = delta["text"]
        if isinstance(delta.get("thinking"), str):
            content["thinking"] = delta["thinking"]
        if len(content) == 1:
            return
        self._release_or_hold_message(state, {
            "name": "stream",
            "reservation_id": reservation_id,
            "turn_reference": state.get("turn_reference"),
            "generation": generation,
            "content": content,
        })

    def _emit_system_message(self, reservation_id: str, generation: int, state: dict[str, Any], message: Any) -> None:
        """Forward a CLI system message with only its allowlisted fields.

        The raw data also carries the working directory, plugin and memory
        paths, MCP server sources and error records that can hold URLs with
        tokens.  None of those leave the bridge.  It is never an identity
        source: the session id here is display data, and binding stays with
        the assistant and result frames.
        """

        subtype = getattr(message, "subtype", None)
        raw = getattr(message, "data", None)
        if not isinstance(subtype, str) or not isinstance(raw, Mapping):
            return
        data = {key: raw[key] for key in _SYSTEM_MESSAGE_FIELDS.get(subtype, ("session_id",)) if key in raw}
        if subtype == "init" and isinstance(data.get("mcp_servers"), list):
            data["mcp_servers"] = [
                {key: server[key] for key in ("name", "status") if key in server}
                for server in data["mcp_servers"]
                if isinstance(server, Mapping)
            ]
        if subtype == "compact_boundary" and isinstance(data.get("compact_metadata"), Mapping):
            data["compact_metadata"] = {
                key: data["compact_metadata"][key]
                for key in ("trigger", "pre_tokens")
                if key in data["compact_metadata"]
            }
        self._release_or_hold_message(state, {
            "name": "system_message",
            "reservation_id": reservation_id,
            "turn_reference": state.get("turn_reference"),
            "generation": generation,
            "subtype": subtype,
            "data": data,
        })

    def _release_or_hold_message(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        if _native_session_id(state.get("session_id")):
            emitted = dict(event)
            emitted["provider_correlation"] = {
                "session": state["session_id"],
                "turn": state.get("turn_reference"),
            }
            emitted["correlation_attested"] = True
            self._write_turn_event(state, emitted)
            return
        pending = state.get("pending_messages")
        if pending is None:
            pending = state["pending_messages"] = []
        if not isinstance(pending, list):
            raise BridgeError("Claude reservation lacks bounded pending message state")
        system_count = sum(_event_name(held) == "system_message" for held in pending)
        if _event_name(event) == "system_message":
            if system_count >= _MAX_PENDING_SYSTEM_MESSAGES:
                raise BridgeError(
                    "Claude pending system message capacity exceeded before identity binding; "
                    f"arriving system_message, held {_held_census(pending)}"
                )
            pending.append(dict(event))
            return
        if len(pending) - system_count >= _MAX_PENDING_EFFECTS:
            # A stream delta is incremental display text.  A reasoning model at
            # high effort emits hundreds of them before the first frame that
            # carries a native session id, and ending the reservation over
            # display text loses the whole run.  Discard the oldest delta and
            # count it: a whole message is worth more than a held delta.
            # System notifications have a separate budget above. A buffer
            # full of other whole messages is still a protocol defect.
            if not self._drop_oldest_stream_delta(state, pending):
                raise BridgeError(
                    "Claude pending message capacity exceeded before identity binding; "
                    f"arriving {_event_name(event)}, held {_held_census(pending)}"
                )
        pending.append(dict(event))

    def _drop_oldest_stream_delta(self, state: dict[str, Any], pending: list[Any]) -> bool:
        """Discard the oldest held stream delta.  Say whether one was there."""

        for index, held in enumerate(pending):
            if isinstance(held, Mapping) and held.get("name") == "stream":
                del pending[index]
                state["dropped_stream_deltas"] = _counter_value(state.get("dropped_stream_deltas")) + 1
                return True
        return False

    def _flush_pending_messages(self, reservation_id: str, generation: int, state: dict[str, Any]) -> None:
        pending = state.get("pending_messages")
        if pending is None:
            pending = state["pending_messages"] = []
        if not isinstance(pending, list):
            raise BridgeError("Claude reservation lacks bounded pending message state")
        for event in pending:
            self._release_or_hold_message(state, event)
        pending.clear()
        dropped = _counter_value(state.get("dropped_stream_deltas"))
        reported = _counter_value(state.get("dropped_stream_deltas_reported"))
        if dropped > reported:
            # The run log says what it lost.  A silent discard would make the
            # record claim a complete stream it does not hold.
            state["dropped_stream_deltas_reported"] = dropped
            self._release_or_hold_message(
                state,
                {
                    "name": "stream_deltas_dropped",
                    "reservation_id": reservation_id,
                    "generation": generation,
                    "turn_reference": state.get("turn_reference"),
                    "dropped": dropped - reported,
                    "dropped_total": dropped,
                },
            )

    def _emit_native_child_event(self, reservation_id: str, generation: int, state: dict[str, Any], message: Any) -> None:
        """Buffer lifecycle facts, then project only exact joined children."""

        parent_tool_use_id = getattr(message, "tool_use_id", None)
        provider_session_id = getattr(message, "session_id", None)
        task_id = getattr(message, "task_id", None)
        source = type(message).__name__
        usage = getattr(message, "usage", None)
        status = getattr(message, "status", None)
        if not isinstance(status, str):
            patch = getattr(message, "patch", None)
            status = patch.get("status") if isinstance(patch, Mapping) else None
        if not isinstance(provider_session_id, str) or not isinstance(task_id, str) or not task_id:
            return
        lifecycle: dict[str, Any] = {
            "source": source,
            "parent_tool_use_id": parent_tool_use_id if isinstance(parent_tool_use_id, str) else None,
            "session_id": provider_session_id,
            "task_id": task_id,
            "status": status if isinstance(status, str) else None,
        }
        if isinstance(usage, Mapping):
            lifecycle["usage"] = dict(usage)
        summary = getattr(message, "summary", None)
        if isinstance(summary, str):
            lifecycle["summary"] = summary
        pending = state.get("native_child_pending_lifecycle")
        if not isinstance(pending, dict):
            raise BridgeError("Claude reservation lacks automatic native-child lifecycle state")
        key = (provider_session_id, task_id)
        if key not in pending and len(pending) >= _MAX_PENDING_EFFECTS:
            raise BridgeError("Claude pending native child lifecycle capacity exceeded")
        lifecycles = pending.setdefault(key, {})
        if not isinstance(lifecycles, dict):
            raise BridgeError("Claude native child lifecycle history is malformed")
        terminal = lifecycles.get("terminal")
        prior_terminal = terminal.get("status") if isinstance(terminal, Mapping) else None
        incoming_terminal = lifecycle.get("status") in {"completed", "failed", "stopped", "killed"}
        if prior_terminal is not None and incoming_terminal:
            if not self._same_native_child_terminal_status(prior_terminal, lifecycle.get("status")):
                raise BridgeError("Claude native child terminal status conflicts with prior terminal")
        if source == "TaskStartedMessage":
            lifecycles["start"] = lifecycle
        elif incoming_terminal:
            # Retain the first raw status. A later stopped/killed alias is
            # equivalent for managed completion but must not rewrite the
            # provider fact that was observed first.
            if prior_terminal is None:
                lifecycles["terminal"] = lifecycle
        elif isinstance(usage, Mapping):
            lifecycles["usage"] = lifecycle
        else:
            lifecycles["latest"] = lifecycle
        if lifecycle.get("status") in {"completed", "failed", "stopped", "killed"}:
            terminal_observed = state.get("native_child_terminal_observed")
            if isinstance(terminal_observed, set):
                terminal_observed.add(key)

        identity: NativeChildIdentity | None = None
        automatic_replayed_current = False
        automatic = state.get("native_child_automatic_resolver")
        if isinstance(automatic, NativeChildAutomaticResolver) and source == "TaskStartedMessage":
            self._native_child_registration_signal(state, "task_started_seen")
            try:
                if not isinstance(parent_tool_use_id, str):
                    raise NativeChildIdentityError("native task lacks parent tool use")
                automatic.record_task_started(provider_session_id, parent_tool_use_id, task_id)
                candidate = automatic.pending_agent_for_task(provider_session_id, task_id)
                if candidate is not None:
                    self._refresh_automatic_native_metadata(state, automatic, provider_session_id, candidate)
                else:
                    self._refresh_pending_automatic_metadata(state, automatic, provider_session_id)
                joined = automatic.resolve_ready()
                self._replay_automatic_native_children(reservation_id, generation, state, joined)
                automatic_replayed_current = any(
                    item.parent_session_id == provider_session_id and item.task_id == task_id for item in joined
                )
                identity = automatic.identity_for_task(provider_session_id, task_id)
                self._native_child_registration_signal(state, "task_started_enrollment_found")
                self._native_child_registration_signal(state, "task_started_joined")
            except NativeChildIdentityError:
                self._native_child_registration_signal(state, "task_started_unmatched")
        elif isinstance(automatic, NativeChildAutomaticResolver):
            try:
                candidate = automatic.pending_agent_for_task(provider_session_id, task_id)
                if candidate is not None:
                    self._refresh_automatic_native_metadata(state, automatic, provider_session_id, candidate)
                else:
                    self._refresh_pending_automatic_metadata(state, automatic, provider_session_id)
                joined = automatic.resolve_ready()
                self._replay_automatic_native_children(reservation_id, generation, state, joined)
                automatic_replayed_current = any(
                    item.parent_session_id == provider_session_id and item.task_id == task_id for item in joined
                )
                identity = automatic.identity_for_task(provider_session_id, task_id)
            except NativeChildIdentityError:
                pass

        # Retain the hook-enrollment route as a strictly authenticated legacy
        # fallback.  It is never a requirement for the automatic path above.
        if identity is None:
            ledger = state.get("native_child_ledger")
            if isinstance(ledger, NativeChildIdentityLedger):
                try:
                    if source == "TaskStartedMessage":
                        if not isinstance(parent_tool_use_id, str):
                            raise NativeChildIdentityError("native task lacks parent tool use")
                        identity = ledger.record_task_started(provider_session_id, parent_tool_use_id, task_id)
                    else:
                        identity = ledger.identity_for_task(provider_session_id, task_id)
                except NativeChildIdentityError:
                    identity = None
        if identity is None:
            self._write_turn_event(
                state,
                {
                    "name": "native_child",
                    "reservation_id": reservation_id,
                    "turn_reference": state.get("turn_reference"),
                    "generation": generation,
                    "source": source,
                    "parent_tool_use_id": lifecycle["parent_tool_use_id"],
                    "task_id": task_id,
                    "reported_session_id": provider_session_id,
                    "child_session_id": None,
                    "status": "awaiting-exact-metadata" if source == "TaskStartedMessage" else "unattested",
                    "tracking": "unavailable",
                },
            )
            return
        if automatic_replayed_current:
            return
        if (identity.identity_source == "saved_session_metadata"
                and key not in state["native_child_identity_emitted"]):
            return
        self._emit_attested_native_child_lifecycle(reservation_id, generation, state, identity, lifecycle)
        pending.pop(key, None)

    @classmethod
    def _project_message(cls, message: Any, source: str, *, tool_inputs: bool = False) -> dict[str, Any] | None:
        if source == "UserMessage":
            content = getattr(message, "content", None)
            blocks = cls._project_content(content, tool_inputs=tool_inputs)
            return {"role": "user", "content": blocks}
        if source == "AssistantMessage":
            projected = {
                "role": "assistant",
                "content": cls._project_content(getattr(message, "content", None), tool_inputs=tool_inputs),
                "model": getattr(message, "model", None),
                "stop_reason": getattr(message, "stop_reason", None),
            }
            parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
            if isinstance(parent_tool_use_id, str) and parent_tool_use_id:
                projected["parent_tool_use_id"] = parent_tool_use_id
            return projected
        if source == "ResultMessage":
            result = getattr(message, "result", None)
            if not isinstance(result, str) or not result:
                return None
            return {"role": "assistant", "content": [{"type": "text", "text": result}], "source": "result"}
        return None

    @classmethod
    def _project_content(cls, content: object, *, tool_inputs: bool = False) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return []
        blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = type(block).__name__
            if block_type == "TextBlock" and isinstance(getattr(block, "text", None), str):
                blocks.append({"type": "text", "text": block.text})
            elif block_type == "ThinkingBlock" and isinstance(getattr(block, "thinking", None), str):
                blocks.append({"type": "thinking", "thinking": block.thinking})
            elif block_type == "ToolUseBlock":
                # Tool arguments can contain source files, commands, tokens
                # and credentials.  The event is an observability stream, not
                # a second plaintext command log, so retain the fact and ID
                # without copying that material out of the provider session.
                # A session that set claude_tool_inputs asked for the copy;
                # its run log and the host history then hold the arguments.
                arguments = cls._bounded_tool_input(getattr(block, "input", {})) if tool_inputs else {}
                blocks.append({"type": "tool_use", "id": getattr(block, "id", None), "name": getattr(block, "name", None), "input": arguments})
            elif block_type == "ToolResultBlock":
                blocks.append({"type": "tool_result", "tool_use_id": getattr(block, "tool_use_id", None), "is_error": getattr(block, "is_error", None)})
            elif block_type == "ServerToolUseBlock":
                blocks.append({"type": "server_tool_use", "id": getattr(block, "id", None), "name": getattr(block, "name", None), "input": dict(getattr(block, "input", {}) or {})})
            elif block_type == "ServerToolResultBlock":
                blocks.append({"type": "server_tool_result", "tool_use_id": getattr(block, "tool_use_id", None), "content": getattr(block, "content", None)})
        return blocks

    @staticmethod
    def _bounded_tool_input(value: object) -> dict[str, Any]:
        """Copy tool arguments, or a preview when they would break the JSONL bound.

        A Write of an ordinary large file can exceed _MAX_JSONL by itself, and
        an event that fails _write fails the whole turn.
        """

        arguments = dict(value or {})
        encoded = json.dumps(arguments, ensure_ascii=True, default=str)
        if len(encoded) <= _TOOL_INPUT_MAX_CHARS:
            return arguments
        return {"_vnext_truncated": True, "chars": len(encoded), "preview": encoded[:_TOOL_INPUT_PREVIEW_CHARS]}

    @staticmethod
    def _terminal_projection(message: Any) -> dict[str, Any]:
        def mapping(value: object) -> dict[str, Any] | None:
            return dict(value) if isinstance(value, Mapping) else None

        return {
            "usage": mapping(getattr(message, "usage", None)),
            "total_cost_usd": getattr(message, "total_cost_usd", None),
            "model_usage": mapping(getattr(message, "model_usage", None)),
            "duration_ms": getattr(message, "duration_ms", None),
            "duration_api_ms": getattr(message, "duration_api_ms", None),
            "num_turns": getattr(message, "num_turns", None),
            "is_error": getattr(message, "is_error", None),
            "stop_reason": getattr(message, "stop_reason", None),
            "subtype": getattr(message, "subtype", None),
            "api_error_status": getattr(message, "api_error_status", None),
            "terminal_reason": getattr(message, "terminal_reason", None),
            # Provider error strings can contain user/project content.  Keep
            # the cardinality, never the strings themselves.
            "error_count": len(getattr(message, "errors", ()) or ()),
        }

    @staticmethod
    def _provider_error_projection(message: Any, source: str) -> dict[str, Any] | None:
        """Project only structured provider failure facts, never error prose."""

        if source == "AssistantMessage":
            code = getattr(message, "error", None)
            # A code is text.  Anything else would be hashed by the set test
            # below, and a list or an object raises there.
            if not isinstance(code, str) or code not in {
                "authentication_failed", "billing_error", "rate_limit",
                "invalid_request", "server_error", "unknown",
            }:
                return None
            return {
                "source": source,
                "is_error": True,
                "code": code,
                "api_error_status": None,
                "retry_after_seconds": None,
            }
        if source != "ResultMessage" or getattr(message, "is_error", None) is not True:
            return None
        status = getattr(message, "api_error_status", None)
        safe_status = status if isinstance(status, int) and not isinstance(status, bool) else None
        subtype = getattr(message, "subtype", None)
        terminal_reason = getattr(message, "terminal_reason", None)
        return {
            "source": source,
            "is_error": True,
            "subtype": subtype if isinstance(subtype, str) else None,
            "api_error_status": safe_status,
            "rate_limited": safe_status == 429,
            "terminal_reason": terminal_reason if isinstance(terminal_reason, str) else None,
            "error_count": len(getattr(message, "errors", ()) or ()),
            # claude-agent-sdk 0.2.143 exposes no structured retry/reset
            # field.  Preserve that absence rather than parsing response prose.
            "retry_after_seconds": None,
        }

    @staticmethod
    def _content_blocks(message: Any) -> tuple[Any, ...]:
        content = getattr(message, "content", ())
        return tuple(content) if isinstance(content, list) else ()

    def _effect_descriptor(self, state: dict[str, Any], block: Any) -> dict[str, Any] | None:
        block_type = type(block).__name__
        tool_effect_types = state.get("tool_effect_types")
        if tool_effect_types is None:
            tool_effect_types = state["tool_effect_types"] = {}
        if not isinstance(tool_effect_types, dict):
            raise BridgeError("Claude reservation lacks bounded tool effect state")
        tool_effect_paths = state.get("tool_effect_paths")
        if tool_effect_paths is None:
            tool_effect_paths = state["tool_effect_paths"] = {}
        if not isinstance(tool_effect_paths, dict):
            raise BridgeError("Claude reservation lacks bounded Write effect state")
        if block_type == "ToolUseBlock":
            tool_use_id, tool_name = getattr(block, "id", None), getattr(block, "name", None)
            effect_type = _EFFECT_TYPES.get(tool_name) if isinstance(tool_name, str) else None
            if not isinstance(tool_use_id, str) or not tool_use_id or effect_type is None:
                return None
            if tool_use_id not in tool_effect_types and len(tool_effect_types) >= _MAX_PENDING_EFFECTS:
                raise BridgeError("Claude tool effect correlation capacity exceeded")
            tool_effect_types[tool_use_id] = effect_type
            if effect_type == "file":
                # This reservation's own root, not the connected session
                # root.  A private Worker's client runs with its own cwd, so a
                # relative file_path resolved against the session root records
                # a path the Worker never wrote -- and two isolated siblings
                # writing the same name would record the identical path, hiding
                # exactly the collision a private root exists to prevent.
                reservation_workspace = state.get("workspace") or self._workspace
                changes, evidence_limited = _write_change_descriptor(
                    reservation_workspace, getattr(block, "input", None)
                )
                tool_effect_paths[tool_use_id] = {
                    "changes": changes,
                    "evidence_limited": evidence_limited,
                }
            return {"name": "tool_use", "tool_use_id": tool_use_id, "effect_type": effect_type}
        if block_type == "ToolResultBlock":
            tool_use_id = getattr(block, "tool_use_id", None)
            # The result settles the correlation, so the entry is released
            # here.  The cap bounds tool calls still awaiting a result; kept,
            # it counted every Bash and Write of the reservation's life and
            # stopped a long Worker at its 65th.
            effect_type = tool_effect_types.pop(tool_use_id, None) if isinstance(tool_use_id, str) else None
            path_descriptor = tool_effect_paths.pop(tool_use_id, None) if isinstance(tool_use_id, str) else None
            if not isinstance(tool_use_id, str) or not tool_use_id or effect_type not in set(_EFFECT_TYPES.values()):
                return None
            is_error = getattr(block, "is_error", None)
            status = "failed" if is_error is True else "completed" if is_error is False else "unknown"
            descriptor = {
                "name": "tool_result",
                "tool_use_id": tool_use_id,
                "effect_type": effect_type,
                "status": status,
            }
            if effect_type == "file":
                if not isinstance(path_descriptor, Mapping):
                    descriptor["changes"] = []
                    descriptor["evidence_limited"] = True
                else:
                    descriptor["changes"] = path_descriptor.get("changes", [])
                    descriptor["evidence_limited"] = path_descriptor.get("evidence_limited") is True
            return descriptor
        return None

    def _release_or_hold_effect(self, reservation_id: str, generation: int, state: dict[str, Any], descriptor: Mapping[str, Any]) -> None:
        if _native_session_id(state.get("session_id")):
            self._write_effect_event(reservation_id, generation, state, descriptor)
            return
        pending = state.get("pending_effects")
        if pending is None:
            pending = state["pending_effects"] = []
        if not isinstance(pending, list):
            raise BridgeError("Claude reservation lacks bounded pending effect state")
        if len(pending) >= _MAX_PENDING_EFFECTS:
            raise BridgeError("Claude pending effect capacity exceeded before identity binding")
        pending.append(dict(descriptor))

    def _flush_pending_effects(self, reservation_id: str, generation: int, state: dict[str, Any]) -> None:
        pending = state.get("pending_effects")
        if pending is None:
            pending = state["pending_effects"] = []
        if not isinstance(pending, list):
            raise BridgeError("Claude reservation lacks bounded pending effect state")
        for descriptor in pending:
            self._write_effect_event(reservation_id, generation, state, descriptor)
        pending.clear()

    def _write_effect_event(self, reservation_id: str, generation: int, state: dict[str, Any], descriptor: Mapping[str, Any]) -> None:
        session_id, turn_reference, tool_use_id = state.get("session_id"), state.get("turn_reference"), descriptor.get("tool_use_id")
        if (not _native_session_id(session_id) or not isinstance(turn_reference, str) or not turn_reference or not isinstance(tool_use_id, str) or not tool_use_id):
            raise BridgeError("Claude effect lacks attested provider correlation")
        event = {"name": descriptor["name"], "reservation_id": reservation_id, "turn_reference": turn_reference, "generation": generation, "tool_use_id": tool_use_id, "effect_type": descriptor["effect_type"], "provider_correlation": {"session": session_id, "turn": turn_reference, "request": tool_use_id}, "correlation_attested": True}
        if descriptor["name"] == "tool_result":
            event["status"] = descriptor["status"]
            if descriptor["effect_type"] == "file":
                event["changes"] = descriptor.get("changes", [])
                event["evidence_limited"] = descriptor.get("evidence_limited") is True
        self._write_turn_event(state, event)

    def _observe_native_agent_result(self, state: dict[str, Any], message: Any, *, reservation_id: str | None = None) -> None:
        """Observe a foreground result on its exact authenticated Agent call.

        Saved-session wrappers can omit the live envelope, so this diagnostic
        deliberately observes the in-flight SDK object. It never persists the
        result mapping or unknown keys. Completion requires the exact saved
        child identity as well as an explicit provider result status.
        """

        session_id = state.get("session_id")
        origins = state.get("native_child_origins")
        content = getattr(message, "content", None)
        if (
            not _native_session_id(session_id) or not isinstance(content, Sequence)
            or not isinstance(origins, Mapping)
        ):
            return
        result_tool_ids = [
            tool_use_id
            for block in content
            if isinstance((tool_use_id := getattr(block, "tool_use_id", None)), str)
        ]
        # A UserMessage can carry several results. Only one exact recorded
        # Agent parent can make this a content-free diagnostic observation.
        if len(result_tool_ids) != 1 or (session_id, result_tool_ids[0]) not in origins:
            return
        tool_id = result_tool_ids[0]
        origin = origins[(session_id, tool_id)]
        parent_tool = getattr(message, "parent_tool_use_id", None)
        # parent_tool_use_id is the enclosing context, not this result's ID.
        if parent_tool and origin.get("parent_native_agent_id") is None:
            return
        result = getattr(message, "tool_use_result", None)
        if not isinstance(result, Mapping):
            return
        self._native_child_registration_signal(state, "agent_result_seen")
        status = result.get("status")
        if isinstance(status, str):
            self._native_child_registration_signal(state, "agent_result_has_status")
            if status == "completed":
                self._native_child_registration_signal(state, "agent_result_status_completed")
            elif status == "failed":
                self._native_child_registration_signal(state, "agent_result_status_failed")
            elif status == "stopped":
                self._native_child_registration_signal(state, "agent_result_status_stopped")
            else:
                self._native_child_registration_signal(state, "agent_result_status_other")
        if any(isinstance(result.get(key), str) and result[key] for key in ("agentId", "agent_id")):
            self._native_child_registration_signal(state, "agent_result_has_agent_id")
        if isinstance(result.get("task_id"), str) and result["task_id"]:
            self._native_child_registration_signal(state, "agent_result_has_task_id")
        resolver = state.get("native_child_automatic_resolver")
        agent_id = result.get("agentId") or result.get("agent_id")
        if (reservation_id is None or status not in ("completed", "failed", "stopped") or not isinstance(agent_id, str)
                or not isinstance(resolver, NativeChildAutomaticResolver)):
            return
        if (result.get("agentId") is not None and result.get("agent_id") is not None
                and result["agentId"] != result["agent_id"]):
            return
        try:
            # A result alone cannot invent a child or its task identity.
            identity = resolver.identity_for_agent(session_id, agent_id)
            if origin.get("parent_native_agent_id") is not None:
                parent_identity = resolver.identity_for_agent(session_id, origin["parent_native_agent_id"])
                if parent_tool != parent_identity.parent_tool_use_id:
                    return
        except NativeChildIdentityError:
            return
        if (identity.parent_tool_use_id != tool_id or identity.task_id is None
                or (result.get("task_id") is not None and result["task_id"] != identity.task_id)):
            return
        key = (session_id, identity.task_id)
        state["native_child_terminal_observed"].add(key)
        self._refresh_automatic_native_metadata(state, resolver, session_id, agent_id)
        generation = origin["generation"]
        lifecycle = {"source": "AgentToolResult", "status": status}
        pending = state["native_child_pending_lifecycle"]
        if key in pending:
            lifecycles = pending[key]
            if not isinstance(lifecycles, dict):
                raise BridgeError("Claude native child lifecycle history is malformed")
            terminal = lifecycles.get("terminal")
            prior_terminal = terminal.get("status") if isinstance(terminal, Mapping) else None
            if prior_terminal is None:
                lifecycles["terminal"] = lifecycle
            elif not self._same_native_child_terminal_status(prior_terminal, status):
                raise BridgeError("Claude native child terminal status conflicts with prior terminal")
            self._replay_automatic_native_children(reservation_id, generation, state, (identity,))
        elif key in state["native_child_identity_emitted"]:
            self._emit_attested_native_child_lifecycle(reservation_id, generation, state, identity, lifecycle)

    def _write_native_child_counters(self, reservation_id: str, state: dict[str, Any]) -> None:
        """Write one turn's native-child enrollment counters to the record.

        The counters are aggregate integers and flags only: no prompt, no SDK
        identifier, no token, no tool input.  Only the signals that actually
        fired are written, so a quiet turn costs one small event.
        """

        counters = state.get("native_child_registration")
        if not isinstance(counters, Mapping):
            return
        fired = {
            key: value
            for key, value in counters.items()
            if (value is True) or (isinstance(value, int) and not isinstance(value, bool) and value > 0)
        }
        self._write_turn_event(state, {
            "name": "native_child_counters",
            # The session runtime attributes an event to an agent through its
            # reservation.  Without these the snapshot was dropped on the way
            # to the run record, unattributed and unlogged.
            "reservation_id": reservation_id,
            "generation": state.get("generation"),
            "turn_reference": state.get("turn_reference"),
            "counters": fired,
        })

    def _write_turn_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        cursor = self._event_cursor
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise BridgeError("Claude bridge lacks a valid event cursor")
        event_with_cursor = dict(event)
        event_with_cursor["cursor"] = cursor + 1
        self._event_cursor = cursor + 1
        _write({"v": _VERSION, "kind": "event", "event": event_with_cursor})

    async def _cancel_all(self) -> None:
        for state in self._reservations.values():
            task = state.get("task")
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            # Process shutdown is the only path that cancels the shared
            # reader.  Release any primary waiters that were tied to it; a
            # normal primary Stop leaves these futures and the reader intact.
            outcomes = state.get("turn_outcomes")
            if isinstance(outcomes, Mapping):
                for outcome in outcomes.values():
                    completion = outcome.get("completion") if isinstance(outcome, Mapping) else None
                    if isinstance(completion, asyncio.Future) and not completion.done():
                        completion.cancel()
            state["task"] = None
            await self._stop_pump(state)
            client = state.get("client")
            if client is not None:
                await client.disconnect()
                state["client"] = None


async def _main() -> int:
    bridge = _Bridge()
    try:
        await bridge.run()
    finally:
        await bridge._cancel_all()
    return 0


def main() -> int:
    # Exactly one event-loop entry point is permitted for the owned worker.
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
