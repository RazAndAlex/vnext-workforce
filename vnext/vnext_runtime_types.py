"""Provider-neutral values shared at the managed-runtime boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .process_supervisor import ProcessCleanup


@dataclass(frozen=True)
class TurnHandle:
    """An opaque runtime turn correlation.

    Some providers only expose their native turn cursor after the first
    response.  ``cursor`` is therefore optional at the managed boundary;
    provider adapters remain responsible for translating it when available.
    """

    thread_id: str
    turn_id: str
    # A provider cursor is only meaningful to the adapter that emitted it.  It
    # can be an offset, token, or another opaque checkpoint, so the shared
    # boundary deliberately makes no representation claim.
    cursor: object | None = None


@dataclass(frozen=True)
class RuntimePosture:
    """Provider-neutral execution posture attested by an adapter."""

    workspace_writes: bool | None
    network: str | bool | None
    approvals_requested: bool | None
    reviewer: str | None
    environment_ready: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_writes": self.workspace_writes,
            "network": self.network,
            "approvals_requested": self.approvals_requested,
            "reviewer": self.reviewer,
            "environment_ready": self.environment_ready,
        }


@dataclass(frozen=True)
class RuntimeIdentityAttestation:
    """Binding state for a locally reserved thread handle."""

    runtime_thread: str
    provider: str
    bound: bool
    provider_session: str | None = None
    binding_phase: str = "reserved"
    synthetic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_thread": self.runtime_thread,
            "provider": self.provider,
            "bound": self.bound,
            "provider_session": self.provider_session,
            "binding_phase": self.binding_phase,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class ToolCallContext:
    thread_id: str = ""
    turn_id: str = ""
    call_id: str = ""
    # Adapter-local checkpoint at which this native turn was observed. It is
    # opaque outside that adapter and lets an adopted external turn wait for
    # completion without replaying a whole provider transcript.
    cursor: object | None = None


@dataclass(frozen=True)
class ToolCallResult:
    """Provider-neutral outcome of one caller-hosted dynamic tool call.

    The control plane decides *what* a tool call answered; it must not decide
    how a vendor carries that answer on the wire.  Codex expects an app-server
    ``contentItems`` payload and the Claude SDK expects MCP content blocks, so
    each adapter projects this value into its own shape.  Keeping the two
    projections in the adapters is what lets the scheduler and the
    orchestration layer stay free of provider names.

    ``value`` is always JSON-serialisable and is the identical text every
    provider shows the model, so a manager sees the same answer whichever
    runtime it is hosted on.
    """

    success: bool
    value: Mapping[str, Any]

    def as_json_text(self) -> str:
        return json.dumps(dict(self.value), ensure_ascii=False, separators=(",", ":"))

    def as_dict(self) -> dict[str, Any]:
        return {"success": bool(self.success), "value": dict(self.value)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolCallResult":
        value = payload.get("value")
        if not isinstance(payload.get("success"), bool) or not isinstance(value, Mapping):
            raise ValueError("neutral tool result payload is malformed")
        return cls(success=payload["success"], value=dict(value))


DynamicToolHandler = Callable[[str, Mapping[str, Any], ToolCallContext], "ToolCallResult"]
NativeApprovalHandler = Callable[[str, Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class NativeChildObservation:
    """A provider-attested child thread observed from an owned parent thread.

    The provider adapter establishes the parent/child thread edge before it
    emits this value. Descriptive fields are optional because providers may
    publish the child thread before their parent-side spawn item is complete.
    ``source_cursor`` stays opaque outside that provider adapter.
    """

    provider: str
    parent_agent_id: str
    parent_thread_id: str
    native_child_thread_id: str
    # An adapter sets this only after it has verified that the native child
    # belongs to the named bound parent thread.  Seeing a provider child-shaped
    # event is not enough to create a vNext agent.
    attested: bool
    delivery_contract: Mapping[str, str]
    native_child_id: str | None = None
    model_id: str | None = None
    role: str | None = None
    effort: str | None = None
    objective: str | None = None
    task_contract: Mapping[str, Any] | None = None
    source_cursor: object | None = None
    parent_native_turn_id: str | None = None
    capabilities: Mapping[str, str] | None = None
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class NativeChildBinding:
    """The control-plane result of adopting an attested native child thread."""

    agent_id: str
    provider: str
    native_thread_id: str
    delivery_contract: Mapping[str, str]
    tool_handler: DynamicToolHandler | None = None
    unsupported_reason: str | None = None


NativeChildObserver = Callable[[NativeChildObservation], NativeChildBinding]
NativeParentAgentResolver = Callable[[str], str | None]


@dataclass(frozen=True)
class RuntimeCleanup:
    process: ProcessCleanup
    streams_drained: bool
    handler_threads_drained: bool
    errors: tuple[str, ...]


# A provider credential reaches only the child process: the z.ai key goes to
# the SDK subprocess, a loopback bearer goes to the Command Code child on its
# own command line.  Either child can echo an authorization value while
# failing, so neither one's stderr is ever written to the session run log --
# the file a user attaches to a bug report.  Both paths that persist captured
# stderr read this one set, because they were allowed to drift apart once.
CREDENTIAL_ADJACENT_STDERR_PROVIDERS = frozenset({"zai", "commandcode"})
