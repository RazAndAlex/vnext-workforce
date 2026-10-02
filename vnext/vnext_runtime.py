"""Provider-neutral runtime protocol used by the vNext managed session.

The orchestration layer owns hierarchy and the runtime owns execution.  This
module deliberately contains only the narrow lifecycle surface shared by
runtime adapters; it does not describe any native prompt or event schema.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .vnext_runtime_types import (
    DynamicToolHandler,
    RuntimeCleanup,
    RuntimePosture,
    TurnHandle,
)


class RuntimePolicyError(RuntimeError):
    """A runtime did not attest the provider-neutral workspace posture."""


def effective_thread_policy(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project an adapter response into the neutral receipt policy shape."""

    posture = result.get("posture")
    posture = posture if isinstance(posture, Mapping) else {}
    return {
        "workspace_writes": posture.get("workspace_writes"),
        "network": posture.get("network"),
        "approvals_requested": posture.get("approvals_requested"),
        "reviewer": posture.get("reviewer"),
        "environment_ready": posture.get("environment_ready"),
    }


def require_workspace_policy(
    result: Mapping[str, Any],
    *,
    approvals_reviewer: str | None = None,
) -> None:
    """Require an adapter-attested writable, restricted, reviewable posture."""

    policy = effective_thread_policy(result)
    network = policy.get("network")
    network_ok = network in {"restricted", "approval_gated", "local_only", False}
    reviewer_ok = bool(policy.get("reviewer"))
    if approvals_reviewer is not None:
        reviewer_ok = reviewer_ok and policy.get("reviewer") == approvals_reviewer
    if (
        policy.get("workspace_writes") is not True
        or not network_ok
        or policy.get("approvals_requested") is not True
        or not reviewer_ok
        or policy.get("environment_ready") is not True
    ):
        raise RuntimePolicyError(f"unexpected effective runtime posture: {policy}")


class RuntimeAdapter(Protocol):
    """The lifecycle and attestation seam for one selected runtime."""

    provider: str
    harness: str
    native_approval_handler: Any

    def initialize(self, *, timeout: float = 30) -> Mapping[str, Any]: ...

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
    ) -> tuple[str, Mapping[str, Any]]: ...

    def start_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        model: str,
        effort: str,
        approvals_reviewer: str,
        workspace: str,
        turn_timeout: float | None = None,
    ) -> TurnHandle: ...

    def wait_turn(self, handle: TurnHandle, *, timeout: float = 300) -> Mapping[str, Any]: ...

    def steer(self, handle: TurnHandle, text: str) -> Mapping[str, Any]: ...

    def interrupt(self, handle: TurnHandle) -> Mapping[str, Any]: ...

    def read_thread(self, thread_id: str) -> Mapping[str, Any]: ...

    def turn_process_ended(self, handle: TurnHandle | None = None) -> bool | None:
        """Whether the provider process behind a turn has exited.

        Optional, and best effort.  True means gone, False means still there,
        None means this adapter cannot tell -- which is also what an adapter
        that does not implement it says.
        """
        ...

    def resume_thread(
        self,
        *,
        thread_id: str,
        model: str,
        tool_handler: DynamicToolHandler | None,
        effort: str = "high",
        approvals_reviewer: str,
        workspace: str,
    ) -> Mapping[str, Any]: ...

    def events_since(self, cursor: object | None = None) -> tuple[dict[str, Any], ...]:
        """Return provider-local events after an opaque adapter cursor.

        The managed layer transports this annotation without interpreting or
        slicing provider events; each adapter owns cursor semantics.
        """
        ...

    def thread_identity_attestation(self, thread_id: str) -> Mapping[str, Any]: ...

    def tool_registration_attestation(self, thread_id: str) -> Mapping[str, Any]: ...

    def close(self) -> RuntimeCleanup: ...


# A readable alias makes the boundary discoverable to callers without forcing
# them to depend on the protocol's implementation name.
VNextRuntimeAdapter = RuntimeAdapter
