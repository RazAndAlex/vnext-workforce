"""Fail-closed Claude native-child identity and hook-proof mechanics.

The automatic resolver joins a SubagentStart identity to supported saved
session metadata and the exact TaskStarted parent tool-use ID. It never
derives parentage from timing, prompts, roles, or models. The separate
ephemeral ledger issues one-shot hook proofs only after an exact identity has
joined, so child-scoped coordination remains authenticated without making a
registration instruction a condition of adoption.

The ledger retains opaque capabilities in memory only. Its inspection APIs and
automatic replay state exclude prompts, tokens, proofs, and transcript text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import secrets
import threading
from typing import Any, Callable, Mapping


_MAX_PENDING_AUTOMATIC_CHILDREN = 64


class NativeChildIdentityError(ValueError):
    """A native-child enrollment record is missing, malformed, or replayed."""


@dataclass(frozen=True)
class NativeChildIdentity:
    """The non-secret identity vNext may use after enrollment."""

    parent_session_id: str
    parent_tool_use_id: str
    agent_id: str
    task_id: str | None
    objective: str | None = None
    requested_model: str | None = None
    model_source: str = "not_provided"
    # These are provider identity facts, never a prompt-derived topology.
    # ``None`` for ``parent_agent_id`` is only meaningful after the automatic
    # resolver has matched an authenticated root-origin hook.
    parent_agent_id: str | None = None
    parent_task_id: str | None = None
    identity_source: str = "hook_enrollment"


@dataclass
class _Enrollment:
    parent_session_id: str
    parent_tool_use_id: str
    token: str = field(repr=False)
    input_digest: bytes = field(repr=False)
    objective: str | None = None
    requested_model: str | None = None
    model_source: str = "not_provided"
    agent_id: str | None = None
    registration_proof: str | None = field(default=None, repr=False)
    task_id: str | None = None
    consumed: bool = False
    coordination_proofs: set[str] = field(default_factory=set, repr=False)


@dataclass
class _AutomaticChild:
    """One provider child whose independently delivered facts may be late."""

    session_id: str
    agent_id: str
    started: bool = False
    metadata_parent_tool_use_id: str | None = None
    metadata_parent_agent_id: str | None = None
    metadata_seen: bool = False
    task_id: str | None = None


@dataclass(frozen=True)
class _AutomaticOrigin:
    parent_agent_id: str | None
    objective: str | None
    requested_model: str | None
    model_source: str


class NativeChildAutomaticResolver:
    """Fail-closed exact join for Claude-native children.

    The Claude SDK exposes the child identity in ``SubagentStart``, and
    recoverable ``agent_metadata`` identifies the Agent tool-use which created
    it.  Task lifecycle messages identify that same Agent tool-use.  These
    streams are independent and can arrive in any order, so this resolver
    stores only identifiers and resolves a child only after all exact facts
    agree.  It intentionally does not infer parentage from time, prompts,
    roles, models, or a missing ``parentAgentId``.

    ``parentAgentId=None`` is accepted only after an authenticated
    ``PreToolUse`` Agent origin recorded a root caller. The installed SDK's
    hook contract guarantees that absent ``agent_id`` means the main context;
    an unavailable hook observation remains pending.
    """

    def __init__(self, *, on_eviction: Callable[[str, str, int], None] | None = None) -> None:
        self._lock = threading.RLock()
        self._on_eviction = on_eviction
        self._children: dict[tuple[str, str], _AutomaticChild] = {}
        self._origins: dict[tuple[str, str], _AutomaticOrigin] = {}
        self._tasks: dict[tuple[str, str], str] = {}
        self._task_to_parent_tool: dict[tuple[str, str], str] = {}
        self._tool_to_child: dict[tuple[str, str], str] = {}
        self._identities: dict[tuple[str, str], NativeChildIdentity] = {}

    @staticmethod
    def _identifier(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise NativeChildIdentityError(f"{name} must be a non-empty string")
        return value

    def record_agent_origin(
        self,
        session_id: object,
        parent_tool_use_id: object,
        parent_agent_id: object = None,
        *,
        parent_agent_id_present: bool,
        tool_input: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the authenticated caller of an Agent tool invocation.

        The installed SDK documents absent ``agent_id`` as the root/main
        context. ``parent_agent_id_present`` means this method received an
        authenticated PreToolUse origin at all; it does not mean a root hook
        had to carry a literal null key.
        """

        session = self._identifier(session_id, "parent session id")
        parent_tool = self._identifier(parent_tool_use_id, "parent tool-use id")
        if not parent_agent_id_present:
            return
        if parent_agent_id is not None and (not isinstance(parent_agent_id, str) or not parent_agent_id):
            raise NativeChildIdentityError("native parent agent id is invalid")
        parent = parent_agent_id if isinstance(parent_agent_id, str) else None
        objective: str | None = None
        requested_model: str | None = None
        model_source = "not_provided"
        if tool_input is not None:
            description = tool_input.get("description")
            if description is not None:
                if not isinstance(description, str):
                    raise NativeChildIdentityError("native Agent description must be a string")
                if len(description.strip()) > 256:
                    raise NativeChildIdentityError(
                        "native Agent description is longer than 256 characters; shorten it and call again"
                    )
                objective = description.strip() or None
            model = tool_input.get("model")
            if model is not None:
                if not isinstance(model, str) or not model.strip():
                    raise NativeChildIdentityError("native Agent model must be a non-empty model name")
                requested_model, model_source = model, "explicit"
        origin = _AutomaticOrigin(parent, objective, requested_model, model_source)
        key = (session, parent_tool)
        with self._lock:
            prior = self._origins.get(key, _MISSING)
            if prior is not _MISSING and prior != origin:
                raise NativeChildIdentityError("native Agent origin conflicts with prior caller")
            self._origins[key] = origin

    def _make_room_for_child(self, key: tuple[str, str]) -> None:
        """Keep at most 64 unresolved children, preferring the newest evidence."""

        if key in self._children:
            return
        pending = sum(child_key not in self._identities for child_key in self._children)
        if pending < _MAX_PENDING_AUTOMATIC_CHILDREN:
            return
        for old_key, child in self._children.items():
            if old_key in self._identities:
                continue
            del self._children[old_key]
            if child.metadata_parent_tool_use_id is not None:
                tool_key = (child.session_id, child.metadata_parent_tool_use_id)
                if self._tool_to_child.get(tool_key) == child.agent_id:
                    del self._tool_to_child[tool_key]
            if self._on_eviction is not None:
                self._on_eviction(child.session_id, child.agent_id, pending)
            return
        raise NativeChildIdentityError("automatic native child pending capacity is inconsistent")

    def record_subagent_start(self, session_id: object, agent_id: object) -> None:
        session = self._identifier(session_id, "parent session id")
        agent = self._identifier(agent_id, "native agent id")
        with self._lock:
            self._make_room_for_child((session, agent))
            child = self._children.setdefault((session, agent), _AutomaticChild(session, agent))
            child.started = True

    def record_metadata(
        self,
        session_id: object,
        agent_id: object,
        parent_tool_use_id: object,
        parent_agent_id: object = None,
        *,
        metadata_seen: bool = True,
    ) -> None:
        """Record SDK ``agent_metadata`` facts without retaining messages.

        A caller that could not read usable metadata must pass
        ``metadata_seen=False``; this deliberately leaves the child pending.
        """

        session = self._identifier(session_id, "parent session id")
        agent = self._identifier(agent_id, "native agent id")
        if not metadata_seen:
            return
        parent_tool = self._identifier(parent_tool_use_id, "metadata parent tool-use id")
        if parent_agent_id is not None and (not isinstance(parent_agent_id, str) or not parent_agent_id):
            raise NativeChildIdentityError("metadata parent agent id is invalid")
        parent = parent_agent_id if isinstance(parent_agent_id, str) else None
        with self._lock:
            tool_key = (session, parent_tool)
            prior_agent = self._tool_to_child.get(tool_key)
            if prior_agent is not None and prior_agent != agent:
                raise NativeChildIdentityError("native Agent tool use is already bound to another child")
            child = self._children.get((session, agent))
            if child is not None and child.metadata_seen and (
                child.metadata_parent_tool_use_id != parent_tool
                or child.metadata_parent_agent_id != parent
            ):
                raise NativeChildIdentityError("native child metadata conflicts with prior metadata")
            self._make_room_for_child((session, agent))
            child = self._children.setdefault((session, agent), _AutomaticChild(session, agent))
            self._tool_to_child[tool_key] = agent
            child.metadata_seen = True
            child.metadata_parent_tool_use_id = parent_tool
            child.metadata_parent_agent_id = parent

    def record_task_started(self, session_id: object, parent_tool_use_id: object, task_id: object) -> None:
        session = self._identifier(session_id, "parent session id")
        parent_tool = self._identifier(parent_tool_use_id, "parent tool-use id")
        task = self._identifier(task_id, "task id")
        key = (session, parent_tool)
        with self._lock:
            prior = self._tasks.get(key)
            if prior is not None and prior != task:
                raise NativeChildIdentityError("native Agent tool use has conflicting task identities")
            task_key = (session, task)
            prior_tool = self._task_to_parent_tool.get(task_key)
            if prior_tool is not None and prior_tool != parent_tool:
                raise NativeChildIdentityError("native task identity is already bound to another Agent tool use")
            self._tasks[key] = task
            self._task_to_parent_tool[task_key] = parent_tool

    def resolve_ready(self) -> tuple[NativeChildIdentity, ...]:
        """Return newly joined identities in parent-before-child order.

        The result is empty while any required stream is missing or ambiguous.
        Repeated calls are idempotent, which makes reconnect/replay safe.
        """

        with self._lock:
            resolved: list[NativeChildIdentity] = []
            progress = True
            while progress:
                progress = False
                for key, child in tuple(self._children.items()):
                    if key in self._identities or not child.started or not child.metadata_seen:
                        continue
                    parent_tool = child.metadata_parent_tool_use_id
                    if parent_tool is None:
                        continue
                    task_id = self._tasks.get((child.session_id, parent_tool))
                    if task_id is None:
                        continue
                    origin = self._origins.get((child.session_id, parent_tool), _MISSING)
                    if origin is _MISSING or origin.parent_agent_id != child.metadata_parent_agent_id:
                        continue
                    parent_task_id: str | None = None
                    if origin.parent_agent_id is not None:
                        parent_identity = self._identities.get((child.session_id, origin.parent_agent_id))
                        if parent_identity is None or parent_identity.task_id is None:
                            continue
                        parent_task_id = parent_identity.task_id
                    identity = NativeChildIdentity(
                        parent_session_id=child.session_id,
                        parent_tool_use_id=parent_tool,
                        agent_id=child.agent_id,
                        task_id=task_id,
                        objective=origin.objective,
                        requested_model=origin.requested_model,
                        model_source=origin.model_source,
                        parent_agent_id=origin.parent_agent_id,
                        parent_task_id=parent_task_id,
                        identity_source="saved_session_metadata",
                    )
                    self._identities[key] = identity
                    resolved.append(identity)
                    progress = True
            return tuple(resolved)

    def identity_for_task(self, session_id: object, task_id: object) -> NativeChildIdentity:
        session = self._identifier(session_id, "parent session id")
        task = self._identifier(task_id, "task id")
        with self._lock:
            for identity in self._identities.values():
                if identity.parent_session_id == session and identity.task_id == task:
                    return identity
        raise NativeChildIdentityError("native task identity is not automatically joined")

    def identity_for_agent(self, session_id: object, agent_id: object) -> NativeChildIdentity:
        session = self._identifier(session_id, "parent session id")
        agent = self._identifier(agent_id, "native agent id")
        with self._lock:
            identity = self._identities.get((session, agent))
            if identity is not None:
                return identity
        raise NativeChildIdentityError("native agent identity is not automatically joined")

    def pending_agent_for_task(self, session_id: object, task_id: object) -> str | None:
        """Return the exact started child candidate for a task, if singular."""

        session = self._identifier(session_id, "parent session id")
        task = self._identifier(task_id, "task id")
        with self._lock:
            parent_tool = self._task_to_parent_tool.get((session, task))
            return self._tool_to_child.get((session, parent_tool)) if parent_tool is not None else None

    def pending_agents(self, session_id: object) -> tuple[str, ...]:
        """Return started, unresolved child IDs for bounded metadata rereads."""

        session = self._identifier(session_id, "parent session id")
        with self._lock:
            return tuple(
                child.agent_id
                for key, child in self._children.items()
                if key not in self._identities and child.session_id == session
            )

    def snapshot(self) -> tuple[NativeChildIdentity, ...]:
        """Return joined identifier facts only; pending data stays private."""

        with self._lock:
            return tuple(self._identities.values())

    def replay_state(self) -> dict[str, object]:
        """Export strict, content-free resolver state for restart replay.

        This deliberately excludes model text, transcript messages, hook
        capabilities, and coordination proofs.  Rehydration recomputes joins
        rather than trusting a previously projected child.
        """

        with self._lock:
            children: list[dict[str, object]] = []
            for child in self._children.values():
                value: dict[str, object] = {
                    "session_id": child.session_id,
                    "agent_id": child.agent_id,
                    "started": child.started,
                    "metadata_seen": child.metadata_seen,
                }
                if child.metadata_seen:
                    value["parent_tool_use_id"] = child.metadata_parent_tool_use_id
                    value["parent_agent_id"] = child.metadata_parent_agent_id
                children.append(value)
            return {
                "version": 1,
                "origins": [
                    {"session_id": session, "parent_tool_use_id": tool, "parent_agent_id": origin.parent_agent_id}
                    for (session, tool), origin in self._origins.items()
                ],
                "tasks": [
                    {"session_id": session, "parent_tool_use_id": tool, "task_id": task}
                    for (session, tool), task in self._tasks.items()
                ],
                "children": children,
            }

    @classmethod
    def from_replay_state(cls, value: object) -> "NativeChildAutomaticResolver":
        """Restore only a schema-valid content-free state and recompute joins."""

        if not isinstance(value, Mapping) or set(value) != {"version", "origins", "tasks", "children"}:
            raise NativeChildIdentityError("automatic native child replay state has an invalid schema")
        if value.get("version") != 1:
            raise NativeChildIdentityError("automatic native child replay state has an unsupported version")
        origins, tasks, children = value.get("origins"), value.get("tasks"), value.get("children")
        if not isinstance(origins, list) or not isinstance(tasks, list) or not isinstance(children, list):
            raise NativeChildIdentityError("automatic native child replay state has invalid collections")
        resolver = cls()
        for origin in origins:
            if not isinstance(origin, Mapping) or set(origin) != {"session_id", "parent_tool_use_id", "parent_agent_id"}:
                raise NativeChildIdentityError("automatic native child replay origin is invalid")
            resolver.record_agent_origin(
                origin.get("session_id"), origin.get("parent_tool_use_id"), origin.get("parent_agent_id"),
                parent_agent_id_present=True,
            )
        for task in tasks:
            if not isinstance(task, Mapping) or set(task) != {"session_id", "parent_tool_use_id", "task_id"}:
                raise NativeChildIdentityError("automatic native child replay task is invalid")
            resolver.record_task_started(task.get("session_id"), task.get("parent_tool_use_id"), task.get("task_id"))
        for child in children:
            child_keys = set(child) if isinstance(child, Mapping) else set()
            if not isinstance(child, Mapping) or child_keys not in (
                {"session_id", "agent_id", "started", "metadata_seen"},
                {"session_id", "agent_id", "started", "metadata_seen", "parent_tool_use_id", "parent_agent_id"},
            ):
                raise NativeChildIdentityError("automatic native child replay child is invalid")
            if not isinstance(child.get("started"), bool) or not isinstance(child.get("metadata_seen"), bool):
                raise NativeChildIdentityError("automatic native child replay child has invalid flags")
            if child["started"]:
                resolver.record_subagent_start(child.get("session_id"), child.get("agent_id"))
            if child["metadata_seen"]:
                resolver.record_metadata(
                    child.get("session_id"), child.get("agent_id"), child.get("parent_tool_use_id"), child.get("parent_agent_id")
                )
        resolver.resolve_ready()
        return resolver


_MISSING = object()


class NativeChildIdentityLedger:
    """Bind native Agent hook identities to task records without timing guesses.

    Callers feed provider-originated values to the matching methods.  The
    register MCP endpoint must accept only the ``enrollment_token`` and
    ``registration_proof`` injected by these hook decisions; it must never
    accept an agent identifier supplied by the model.
    """

    _INSTRUCTION = (
        "\n\n[vNext native-child enrollment: before any other work, call the "
        "MCP tool {registration_tool_name} exactly once with "
        "enrollment_token={token!r}.]"
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_token: dict[tuple[str, str], _Enrollment] = {}
        self._by_parent_tool: dict[tuple[str, str], _Enrollment] = {}
        self._by_agent: dict[tuple[str, str], _Enrollment] = {}
        self._by_task: dict[tuple[str, str], _Enrollment] = {}

    @staticmethod
    def _identifier(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise NativeChildIdentityError(f"{name} must be a non-empty string")
        return value

    @classmethod
    def _prompt(cls, tool_input: Mapping[str, Any]) -> str:
        prompt = tool_input.get("prompt")
        if not isinstance(prompt, str):
            raise NativeChildIdentityError("native Agent input lacks a string prompt")
        return prompt

    @staticmethod
    def _input_digest(tool_input: Mapping[str, Any]) -> bytes:
        """Compare retried hook input without retaining its potentially raw text."""

        try:
            encoded = json.dumps(dict(tool_input), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise NativeChildIdentityError("native Agent input is not serializable") from exc
        return hashlib.sha256(encoded.encode("utf-8")).digest()

    @staticmethod
    def _task_metadata(tool_input: Mapping[str, Any]) -> tuple[str | None, str | None, str]:
        description = tool_input.get("description")
        objective = description.strip() if isinstance(description, str) and description.strip() else None
        # A task label is normal runtime task metadata, but must stay bounded
        # so a caller cannot smuggle the full prompt through this route.
        if objective is not None and len(objective) > 256:
            raise NativeChildIdentityError("native Agent description is longer than 256 characters; shorten it and call again")
        requested_model = tool_input.get("model")
        if requested_model is None:
            return objective, None, "not_provided"
        if not isinstance(requested_model, str) or not requested_model.strip():
            raise NativeChildIdentityError("native Agent model must be a non-empty model name")
        return objective, requested_model, "explicit"

    @staticmethod
    def _hook_output(updated_input: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": dict(updated_input),
            }
        }

    def rewrite_agent_input(
        self,
        parent_session_id: object,
        parent_tool_use_id: object,
        tool_input: Mapping[str, Any],
        *,
        registration_tool_name: str = "vnext_register_native_child",
    ) -> dict[str, Any]:
        """Create one capability and append its instruction to a native Agent prompt.

        This matches the SDK's documented ``PreToolUse`` ``updatedInput``
        result.  A production caller must invoke it only for the native Agent
        tool and fail closed when the live hook does not expose this prompt
        shape.
        """

        session = self._identifier(parent_session_id, "parent session id")
        parent_tool = self._identifier(parent_tool_use_id, "parent tool-use id")
        prompt = self._prompt(tool_input)
        registration_tool = self._identifier(registration_tool_name, "native child registration tool name")
        key = (session, parent_tool)
        input_digest = self._input_digest(tool_input)
        objective, requested_model, model_source = self._task_metadata(tool_input)
        with self._lock:
            enrollment = self._by_parent_tool.get(key)
            if enrollment is None:
                token = secrets.token_urlsafe(32)
                enrollment = _Enrollment(
                    session, parent_tool, token, input_digest, objective, requested_model, model_source
                )
                self._by_parent_tool[key] = enrollment
                self._by_token[(session, token)] = enrollment
            elif not secrets.compare_digest(enrollment.input_digest, input_digest):
                # Provider delivery can retry a hook.  It is safe to replay
                # an identical invocation, but never to exchange an enrolled
                # parent tool-use ID for a different prompt.
                raise NativeChildIdentityError("native Agent tool use was replayed with conflicting input")
            token = enrollment.token
        updated_input = dict(tool_input)
        updated_input["prompt"] = prompt + self._INSTRUCTION.format(
            registration_tool_name=registration_tool, token=token
        )
        return self._hook_output(updated_input)

    def rewrite_register_input(
        self,
        parent_session_id: object,
        agent_id: object,
        child_tool_use_id: object,
        tool_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind an SDK hook's child identity and inject one MCP registration proof."""

        session = self._identifier(parent_session_id, "parent session id")
        agent = self._identifier(agent_id, "native agent id")
        child_tool = self._identifier(child_tool_use_id, "child tool-use id")
        token = tool_input.get("enrollment_token")
        token = self._identifier(token, "enrollment token")
        with self._lock:
            enrollment = self._by_token.get((session, token))
            if enrollment is None or enrollment.consumed:
                raise NativeChildIdentityError("native child enrollment token is unknown or consumed")
            if enrollment.agent_id is not None:
                raise NativeChildIdentityError("native child enrollment is already bound")
            if (session, agent) in self._by_agent:
                raise NativeChildIdentityError("native agent identity is already bound")
            # This capability is generated from the hook-observed agent and
            # tool-use values.  The MCP endpoint sees the proof, not either
            # model-supplied identity, so concurrent child calls cannot race
            # through a shared 'latest hook' slot.
            proof = secrets.token_urlsafe(32)
            enrollment.agent_id = agent
            enrollment.registration_proof = proof
            self._by_agent[(session, agent)] = enrollment
        updated_input = dict(tool_input)
        updated_input["registration_proof"] = proof
        # Keep the originating hook ID only in the short-lived hook pipeline;
        # it is checked for shape above and deliberately not persisted.
        _ = child_tool
        return self._hook_output(updated_input)

    def register_from_mcp(
        self,
        parent_session_id: object,
        enrollment_token: object,
        registration_proof: object,
    ) -> NativeChildIdentity:
        """Consume the two hook-issued capabilities and return the bound child."""

        session = self._identifier(parent_session_id, "parent session id")
        token = self._identifier(enrollment_token, "enrollment token")
        proof = self._identifier(registration_proof, "registration proof")
        with self._lock:
            enrollment = self._by_token.get((session, token))
            if (
                enrollment is None
                or enrollment.consumed
                or enrollment.agent_id is None
                or not secrets.compare_digest(enrollment.registration_proof or "", proof)
            ):
                raise NativeChildIdentityError("native child registration is not attested")
            enrollment.consumed = True
            return self._identity(enrollment)

    def adopt_automatic_identity(self, identity: NativeChildIdentity) -> None:
        """Authorize hook-scoped coordination proofs for one resolved child.

        This accepts only the resolver's already exact identity object.  It
        does not create an enrollment capability, mutate a prompt, or make
        tools available to a provider child; inherited tool configuration is
        still a separate provider gate.
        """

        session = self._identifier(identity.parent_session_id, "parent session id")
        parent_tool = self._identifier(identity.parent_tool_use_id, "parent tool-use id")
        agent = self._identifier(identity.agent_id, "native agent id")
        key = (session, parent_tool)
        with self._lock:
            existing = self._by_parent_tool.get(key)
            if existing is not None and (
                existing.agent_id not in {None, agent} or existing.task_id not in {None, identity.task_id}
            ):
                raise NativeChildIdentityError("automatic native child conflicts with enrollment identity")
            if existing is None:
                existing = _Enrollment(
                    session, parent_tool, "", b"", identity.objective,
                    identity.requested_model, identity.model_source,
                )
                self._by_parent_tool[key] = existing
            agent_existing = self._by_agent.get((session, agent))
            if agent_existing is not None and agent_existing is not existing:
                raise NativeChildIdentityError("automatic native agent identity is already bound")
            if identity.task_id is not None:
                task_existing = self._by_task.get((session, identity.task_id))
                if task_existing is not None and task_existing is not existing:
                    raise NativeChildIdentityError("automatic native task identity is already bound")
                self._by_task[(session, identity.task_id)] = existing
            existing.agent_id = agent
            existing.task_id = identity.task_id
            existing.consumed = True
            self._by_agent[(session, agent)] = existing

    def rewrite_coordination_input(
        self,
        parent_session_id: object,
        agent_id: object,
        child_tool_use_id: object,
        tool_input: Mapping[str, Any],
        *,
        context_field: str,
    ) -> dict[str, Any]:
        """Inject a one-shot internal proof into a registered child's tool call.

        A coordination tool must declare ``context_field`` as an optional
        string property.  The provider's hook supplies the child identity;
        neither the model nor the MCP endpoint chooses it.
        """

        session = self._identifier(parent_session_id, "parent session id")
        agent = self._identifier(agent_id, "native agent id")
        self._identifier(child_tool_use_id, "child tool-use id")
        if not isinstance(context_field, str) or not context_field:
            raise NativeChildIdentityError("native child context field is invalid")
        with self._lock:
            enrollment = self._by_agent.get((session, agent))
            if enrollment is None or not enrollment.consumed:
                raise NativeChildIdentityError("native coordination caller is not registered")
            proof = secrets.token_urlsafe(32)
            enrollment.coordination_proofs.add(proof)
        updated_input = dict(tool_input)
        # An untrusted model value never survives this overwrite.
        updated_input[context_field] = proof
        return self._hook_output(updated_input)

    def consume_coordination_context(
        self,
        parent_session_id: object,
        proof: object,
    ) -> NativeChildIdentity:
        """Consume one child-scoped tool proof and return its exact identity."""

        session = self._identifier(parent_session_id, "parent session id")
        candidate = self._identifier(proof, "native child context proof")
        with self._lock:
            for enrollment in self._by_parent_tool.values():
                if (
                    enrollment.parent_session_id == session
                    and enrollment.consumed
                    and candidate in enrollment.coordination_proofs
                ):
                    enrollment.coordination_proofs.remove(candidate)
                    return self._identity(enrollment)
        raise NativeChildIdentityError("native child coordination context is not attested")

    def record_task_started(
        self,
        parent_session_id: object,
        parent_tool_use_id: object,
        task_id: object,
    ) -> NativeChildIdentity | None:
        """Attach an SDK task to its enrolled parent tool use, in either event order."""

        session = self._identifier(parent_session_id, "parent session id")
        parent_tool = self._identifier(parent_tool_use_id, "parent tool-use id")
        task = self._identifier(task_id, "task id")
        with self._lock:
            enrollment = self._by_parent_tool.get((session, parent_tool))
            if enrollment is None:
                raise NativeChildIdentityError("native task has no enrolled parent tool use")
            task_key = (session, task)
            existing = self._by_task.get(task_key)
            if existing is not None and existing is not enrollment:
                raise NativeChildIdentityError("native task identity is already bound")
            if enrollment.task_id not in {None, task}:
                raise NativeChildIdentityError("native Agent enrollment already has a different task")
            enrollment.task_id = task
            self._by_task[task_key] = enrollment
            return self._identity(enrollment) if enrollment.consumed else None

    def identity_for_agent(self, parent_session_id: object, agent_id: object) -> NativeChildIdentity:
        """Return a registered child identity; pending enrollments stay unavailable."""

        session = self._identifier(parent_session_id, "parent session id")
        agent = self._identifier(agent_id, "native agent id")
        with self._lock:
            enrollment = self._by_agent.get((session, agent))
            if enrollment is None or not enrollment.consumed:
                raise NativeChildIdentityError("native agent identity is not registered")
            return self._identity(enrollment)

    def identity_for_task(self, parent_session_id: object, task_id: object) -> NativeChildIdentity:
        """Return a registered child by task identity."""

        session = self._identifier(parent_session_id, "parent session id")
        task = self._identifier(task_id, "task id")
        with self._lock:
            enrollment = self._by_task.get((session, task))
            if enrollment is None or not enrollment.consumed:
                raise NativeChildIdentityError("native task identity is not registered")
            return self._identity(enrollment)

    def snapshot(self) -> tuple[NativeChildIdentity, ...]:
        """Return inspectable identities without prompts, tokens, or proofs."""

        with self._lock:
            return tuple(
                self._identity(enrollment)
                for enrollment in self._by_parent_tool.values()
                if enrollment.consumed
            )

    @staticmethod
    def _identity(enrollment: _Enrollment) -> NativeChildIdentity:
        if enrollment.agent_id is None:
            raise NativeChildIdentityError("native child enrollment has no hook identity")
        return NativeChildIdentity(
            parent_session_id=enrollment.parent_session_id,
            parent_tool_use_id=enrollment.parent_tool_use_id,
            agent_id=enrollment.agent_id,
            task_id=enrollment.task_id,
            objective=enrollment.objective,
            requested_model=enrollment.requested_model,
            model_source=enrollment.model_source,
        )
