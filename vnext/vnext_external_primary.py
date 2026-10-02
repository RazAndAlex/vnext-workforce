"""The runtime seam for a primary this service does not own.

An ordinary vNext primary is a provider process that vNext starts, prompts and
stops.  An *external* primary is not: it is a client that already exists and
already has its own user, its own conversation and its own permission system.
Claude Code is the first one.  vNext holds a record for it, serves it the
manager tool set, and runs the child tree it asks for, but never starts a turn
on it and never claims to be able to stop it.

The adapter therefore implements the lifecycle half of ``RuntimeAdapter`` and
refuses the execution half.  ``start_thread`` is real work: it captures the
scheduler's tool handler, which is the only route an outside caller has into
the control plane.  ``start_turn`` raises, because the turn belongs to the
client's own loop.

Nothing here weakens a boundary.  The posture this adapter attests is the
client's own: the external primary does write to the workspace, does ask its
user for approval, and is reviewed by that user rather than by vNext.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping, Sequence

from .process_supervisor import ProcessCleanup
from .vnext_runtime_types import (
    DynamicToolHandler,
    RuntimeCleanup,
    RuntimePosture,
    ToolCallResult,
    TurnHandle,
)


class ExternalPrimaryError(RuntimeError):
    """An execution control was asked of a primary this service does not own."""


class ExternalPrimaryAdapter:
    """One external client bound as the root of a vNext control tree."""

    provider = "external"
    harness = "external-mcp"

    def __init__(self, *, client: str = "external", workspace: str | None = None) -> None:
        if not isinstance(client, str) or client != client.strip() or not client:
            raise ValueError("external primary client name is required")
        self.native_approval_handler: Any = None
        self._client = client
        self._workspace = workspace
        self._lock = threading.RLock()
        self._thread_id: str | None = None
        self._handler: DynamicToolHandler | None = None
        self._tools: tuple[Mapping[str, Any], ...] = ()
        self._dispatched = 0
        self._closed = False
        self._bound = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, *, timeout: float = 30) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "harness": self.harness,
            "client": self._client,
            "execution": "owned-by-client",
        }

    def start_thread(
        self,
        *,
        model: str,
        developer_instructions: str,
        effort: str = "high",
        tools: Sequence[Mapping[str, Any]],
        tool_handler: DynamicToolHandler | None,
        requested_posture: RuntimePosture,
        workspace: str,
    ) -> tuple[str, Mapping[str, Any]]:
        """Capture the handler an outside caller will reach the tree through."""

        if tool_handler is None:
            raise ExternalPrimaryError("an external primary requires a tool handler")
        with self._lock:
            if self._closed:
                raise ExternalPrimaryError("external primary adapter is closed")
            if self._thread_id is not None:
                raise ExternalPrimaryError("external primary already has a thread")
            self._thread_id = f"external-{self._client}-{uuid.uuid4()}"
            self._handler = tool_handler
            self._tools = tuple(dict(tool) for tool in tools)
            thread_id = self._thread_id
        self._bound.set()
        # The posture is the client's own, not a claim vNext makes on its
        # behalf: it writes in the workspace and its user approves its work.
        return thread_id, {
            "model": model,
            "effort": effort,
            "workspace": workspace,
            "client": self._client,
            "posture": {
                "workspace_writes": True,
                "network": "restricted",
                "approvals_requested": True,
                "reviewer": "user",
                "environment_ready": True,
            },
        }

    def wait_bound(self, timeout: float = 30) -> bool:
        """Whether the scheduler has registered the tool handler yet."""

        if self._closed:
            return False
        return self._bound.wait(timeout)

    def close(self) -> RuntimeCleanup:
        with self._lock:
            self._closed = True
            self._handler = None
        self._bound.clear()
        # vNext owns no process for an external primary, so there is nothing to
        # reap and nothing to drain.  Say so exactly rather than leaving the
        # cleanup record silent about a runtime that did appear in the tree.
        return RuntimeCleanup(
            ProcessCleanup("external-primary", 0, 0, ()), True, True, ()
        )

    # -- execution, which this adapter does not own ------------------------

    def start_turn(self, **_kwargs: Any) -> TurnHandle:
        raise ExternalPrimaryError(
            "an external primary runs its own turns; vNext does not start them"
        )

    def wait_turn(self, handle: TurnHandle, *, timeout: float = 300) -> Mapping[str, Any]:
        raise ExternalPrimaryError("an external primary has no vNext-owned turn to wait on")

    def steer(self, handle: TurnHandle, text: str) -> Mapping[str, Any]:
        raise ExternalPrimaryError("an external primary is steered by its own user")

    def interrupt(self, handle: TurnHandle) -> Mapping[str, Any]:
        raise ExternalPrimaryError("an external primary is stopped by its own client")

    def resume_thread(self, **_kwargs: Any) -> Mapping[str, Any]:
        raise ExternalPrimaryError("an external primary reconnects through its own client")

    def can_start_turn(self, thread_id: str) -> bool:
        """Never.  The scheduler must leave this agent ready and unstarted."""
        return False

    # -- observation -------------------------------------------------------

    def read_thread(self, thread_id: str) -> Mapping[str, Any]:
        return {"thread_id": thread_id, "items": [], "source": "external-client"}

    def events_since(self, cursor: object | None = None) -> tuple[dict[str, Any], ...]:
        return ()

    def thread_identity_attestation(self, thread_id: str) -> Mapping[str, Any]:
        """Attest the binding that exists, and claim no provider session.

        ``provider_session`` is ``None`` on purpose.  An external primary has a
        session, but it belongs to its own client and this service has never
        observed it; naming one here would be the invention the managed layer's
        ``synthetic`` check exists to catch.  The binding itself is real, so it
        is attested as such.
        """

        with self._lock:
            bound = self._thread_id
        return {
            "runtime_thread": thread_id,
            "provider": self.provider,
            "client": self._client,
            "bound": bound is not None and thread_id == bound,
            "provider_session": None,
            "binding_phase": "external-client-bound",
            "synthetic": False,
        }

    def tool_registration_attestation(self, thread_id: str) -> Mapping[str, Any]:
        with self._lock:
            names = tuple(str(tool.get("name") or "") for tool in self._tools)
            registered = self._handler is not None and thread_id == self._thread_id
        return {
            "acknowledged": registered,
            "model_id": self._client,
            "tool_count": len(names),
            "tool_names": names,
            "definition_sha256": None,
            "handler_registered": registered,
        }

    # -- the inbound seam --------------------------------------------------

    @property
    def tools(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return self._tools

    @property
    def closed(self) -> bool:
        """Whether close() has run, so no wait here can ever end in a binding.

        ``wait_bound`` answers False for a closed adapter without sleeping, so a
        caller polling it has nothing to pace itself against.  Saying plainly
        that the adapter is closed lets that caller give up at once rather than
        spin through its whole timeout.
        """

        return self._closed

    @property
    def dispatched_calls(self) -> int:
        with self._lock:
            return self._dispatched

    def dispatch_external_call(
        self, *, tool: str, arguments: Mapping[str, Any]
    ) -> ToolCallResult | Mapping[str, Any]:
        """Run one manager tool on behalf of the client that owns the root.

        The ``None`` context is the point of this adapter.  A provider tool call
        carries a native turn to adopt; this one carries no turn at all, because
        the caller's turn is not a vNext turn.  ``_manager_handler`` already
        accepts that shape.
        """

        if not isinstance(tool, str) or not tool.strip():
            raise ExternalPrimaryError("external tool call requires a tool name")
        if not isinstance(arguments, Mapping):
            raise ExternalPrimaryError("external tool call requires an arguments object")
        with self._lock:
            handler = self._handler
            if self._closed or handler is None:
                raise ExternalPrimaryError("external primary is not bound to a control tree")
            names = {str(item.get("name") or "") for item in self._tools}
            if tool not in names:
                raise ExternalPrimaryError(f"unknown manager tool: {tool}")
            self._dispatched += 1
        return handler(tool, dict(arguments), None)
