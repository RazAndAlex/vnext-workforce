"""Provider-neutral local contract between vNext and the host shell.

The contract deliberately describes observed session facts and commands.  It
does not translate a provider transcript or claim that a provider capability is
available before the runtime attests it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence


CapabilityState = Literal["available", "unavailable", "unknown"]
RuntimeEventEmitter = Callable[["RuntimeEvent"], None]


class HostContractError(ValueError):
    """The client sent a malformed or inconsistent contract request."""


@dataclass(frozen=True)
class SessionStartRequest:
    """Stable vNext identity and selected primary for a new session.

    ``config`` is intentionally a provider-neutral extension point for the
    preset catalog and future adapter-specific settings.  It is persisted as
    configuration, never interpreted by the durable host.
    """

    session_id: str
    workspace: str
    primary_agent_id: str
    primary: Mapping[str, Any]
    main_preset: Mapping[str, Any] = field(default_factory=dict)
    catalog_config: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionRestoreRequest:
    """Durable, content-free state supplied to a runtime after host restart.

    The host supplies the exact saved session identity, its ordered agent map
    and the last durable cursor.  It deliberately excludes history and raw
    provider records: adapter-specific reconnect code must obtain those from
    the provider and attest them independently.
    """

    start: SessionStartRequest
    status: str
    cursor: int
    agents: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class RuntimeEvent:
    """One observed runtime fact.  The host supplies durable ordering fields."""

    type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    agent_id: str | None = None
    turn_id: str | None = None
    command_id: str | None = None


@dataclass(frozen=True)
class TerminalAttachment:
    """Result of asking a runtime to attach a terminal presentation channel."""

    terminal_id: str
    agent_id: str
    supported: bool
    mode: str = "unsupported"
    detail: str | None = None
    launch: Mapping[str, Any] = field(default_factory=dict)


class ManagedRuntimeSession(Protocol):
    """Runtime implementation supplied by vNext's Codex/Claude adapters.

    ``start`` is intentionally explicit: creating a durable host record has no
    model-side effect.  Implementations emit only provider-observed facts via
    the factory emitter.  They must not use host persistence as a substitute
    for native terminal or streaming support.
    """

    def start(self) -> Mapping[str, Any] | None: ...

    def restore(self, request: SessionRestoreRequest) -> Mapping[str, Any] | None: ...

    def resume_primary(self) -> Mapping[str, Any] | None: ...

    def prompt(self, agent_id: str, text: str, command_id: str) -> Mapping[str, Any] | None: ...

    def interrupt(
        self, agent_id: str, turn_id: str | None, command_id: str
    ) -> Mapping[str, Any] | None: ...

    def compact(self, agent_id: str, command_id: str) -> Mapping[str, Any] | None: ...

    def cancel(
        self, agent_ids: Sequence[str] | None, command_id: str
    ) -> Mapping[str, Any] | None: ...

    def peer_message(
        self, from_agent_id: str, to_agent_id: str, text: str, command_id: str
    ) -> Mapping[str, Any] | None: ...

    def resolve_approval(
        self, approval_id: str, decision: str, command_id: str
    ) -> Mapping[str, Any] | None: ...

    def snapshot(self) -> Mapping[str, Any] | None: ...

    def history(self, agent_id: str, after: int = 0) -> Sequence[Mapping[str, Any]]: ...

    def attach_terminal(
        self, agent_id: str, terminal_id: str, command_id: str, *, mode: str = "managed"
    ) -> TerminalAttachment | Mapping[str, Any] | None: ...

    def terminal_command(self, terminal_id: str, command_type: str,
                         payload: Mapping[str, Any], command_id: str) -> Mapping[str, Any]: ...

    def detach_terminal(self, terminal_id: str, command_id: str) -> Mapping[str, Any] | None: ...

    def close(self) -> None: ...


class RuntimeSessionFactory(Protocol):
    def __call__(
        self, request: SessionStartRequest, emit: RuntimeEventEmitter
    ) -> ManagedRuntimeSession: ...
