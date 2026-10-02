"""vNext control-plane authority kernel.

This module proves manager-selected dynamic hierarchy and non-blocking wake
semantics and backs the default-available vNext managed product path.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .vnext_model_identity import identity_view
from collections.abc import Mapping
from typing import Any


class AgentRole(str, Enum):
    ROOT_MANAGER = "root-manager"
    BRANCH_MANAGER = "branch-manager"
    WORKER = "worker"


class AgentStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    AWAITING_WORKERS = "awaiting-workers"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REPLACED = "replaced"


class EvidenceKind(str, Enum):
    FACT = "fact"
    OBSERVATION = "observation"
    INFERENCE = "inference"
    LEARNED_POLICY = "learned-policy"


WORKSPACE_SCOPES = ("shared", "private", "worktree")
"""Where a child works, chosen by its manager.

``shared`` is the session workspace itself: the default, and what every
agent got before there was a choice.  ``private`` is an empty working
directory of the child's own, for producing something new on bare ground; it
gives the child a starting folder and limits nothing its tools can reach.
``worktree`` is a git checkout of the child's own, for changing existing code
without colliding with a sibling.

Nothing here prefers one over another and none is imposed.  A manager asked
how to stop two Workers overwriting each other decides per child; this layer
only supplies the three shapes it may ask for.
"""


TERMINAL_STATUSES = frozenset(
    {
        AgentStatus.COMPLETED,
        AgentStatus.FAILED,
        AgentStatus.CANCELLED,
        AgentStatus.REPLACED,
    }
)
# The one terminal status replace accepts.  BLOCKED is not terminal, so
# retry and replace both already reach it; FAILED needed naming.
REPLACEABLE_STATUSES = frozenset({AgentStatus.FAILED})


# A settled agent has stopped and cannot move again on its own.  That is wider
# than terminal: a BLOCKED agent is waiting for its manager to decide something
# and will sit there for as long as nobody does, so anything asking "can this
# still progress by itself" reads this set.  Terminal keeps its own narrower
# meaning of "finished forever" for the code that depends on it.
SETTLED_STATUSES = frozenset(TERMINAL_STATUSES | {AgentStatus.BLOCKED})


# Runtime-maintained lifetime totals survive a caller's momentary usage snapshot.
RUNNING_TOTAL_USAGE_KEYS = ("cost_tokens", "cost_usd", "provider_calls")


# How much of a finished agent's own account of itself travels back to its
# manager.  The whole text is kept in the outcome log, which is a file on disk;
# what a manager reads arrives inside a tool result or a turn prompt, and one
# worker that wrote a long report must not take the room ten workers need.  The
# cut mark names the file, so nothing is lost without saying where it went.
OUTCOME_TEXT_LIMIT = 4000
OUTCOME_EVIDENCE_LIMIT = 10

# What to say about an agent that stopped and left no words behind.  A manager
# cancels a child itself, so that case needs no explanation beyond the fact.
_STOPPED_WITHOUT_A_WORD = {
    AgentStatus.CANCELLED: "This agent was cancelled before it reported an outcome.",
    AgentStatus.REPLACED: "This agent was replaced before it reported an outcome.",
    AgentStatus.FAILED: "This agent failed and reported no reason.",
    AgentStatus.BLOCKED: "This agent stopped on a blocker and named no reason.",
    AgentStatus.COMPLETED: "This agent completed and reported no outcome text.",
}


def finished_agent_outcome(agent: "AgentRecord") -> dict[str, Any]:
    """What a stopped agent said when it stopped, bounded for a manager to read.

    A manager asks a worker for a result, so the result has to come back
    through the same interface it delegated through.  Until this existed the
    text went only to the outcome log on disk: a manager that vNext does not
    run, which cannot read that log from inside a tool call, could wait for a
    child and then had no way to see what it reported.

    An agent that stopped without completing has no outcome of its own, and the
    manager still needs to know why it stopped, so its blocker or its failure
    reason is reported in that field instead.  An agent still working reports
    nothing here: there is no result yet, and an empty field reads like one.
    """

    if agent.status not in SETTLED_STATUSES:
        return {}
    result = agent.result if isinstance(agent.result, Mapping) else {}
    text = str(result.get("outcome") or "").strip()
    if not text:
        text = str(agent.blocker or "").strip()
    if not text:
        text = _STOPPED_WITHOUT_A_WORD.get(agent.status, "")
    cut_mark = (
        "… (cut; the full text is in "
        f".vnext/outcomes/{agent.session_id}.jsonl)"
    )
    if len(text) > OUTCOME_TEXT_LIMIT:
        text = text[: OUTCOME_TEXT_LIMIT - len(cut_mark)].rstrip() + cut_mark
    evidence = [str(item) for item in (result.get("evidence") or ())]
    summary = {
        "outcome": text,
        "verified": bool(result.get("verified")),
        "evidence": evidence[:OUTCOME_EVIDENCE_LIMIT],
    }
    # A worktree child's checkout: its path, its branch, and whether it was
    # kept.  The manager decides what to do with a kept one.
    if isinstance(result.get("workspace"), Mapping):
        summary["workspace"] = dict(result["workspace"])
    return summary


# The folder name spawn_agent gives a child with a root of its own.
_WORKSPACE_DIR_NAME = re.compile(r"agent-([1-9][0-9]*)")


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RegistryClaim:
    kind: EvidenceKind
    statement: str
    evidence_pointer: str | None = None


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    eligible_roles: frozenset[AgentRole]
    claims: tuple[RegistryClaim, ...] = ()
    # Runtime axes are registry data, not orchestration policy.  Empty values
    # keep existing callers source-compatible; the registry/backend seam owns
    # concrete runtime defaults.
    provider: str | None = None
    harness: str | None = None
    credential_location: str | None = None


@dataclass(frozen=True)
class EconomicPreset:
    """Which models a session may use, and from which providers.

    There is deliberately no ceiling on how many agents may be alive. A cap on
    delegation can force a manager to retain work in a long-lived thread whose
    accumulated context is re-read on every call, but child startup and
    coordination add their own token cost. Whether a particular delegation
    lowers cost requires observed usage and priced model claims; this preset
    does not assert it. The old ``max_active_agents`` also counted every non-terminal
    agent rather than every running one, so a parked manager and a blocked
    child each held a slot while costing nothing; it bounded tree size, not
    concurrent load.

    If this host ever turns out to limit how many provider processes can be
    alive at once, that is a mechanical fact about the machine and belongs
    where runtimes are bound, described as what it is.  It does not return here
    as a number.
    """

    preset_id: str
    allowed_models: frozenset[str]
    allowed_providers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RuntimeSelection:
    """The resolved runtime axes for one role/model choice."""

    model_id: str
    role: AgentRole
    provider: str | None
    harness: str | None
    credential_location: str | None


class ModelRegistry:
    def __init__(
        self,
        cards: list[ModelCard],
        presets: list[EconomicPreset],
    ) -> None:
        self.cards = {card.model_id: card for card in cards}
        self.presets = {preset.preset_id: preset for preset in presets}
        if len(self.cards) != len(cards) or len(self.presets) != len(presets):
            raise ValueError("model and preset identifiers must be unique")

    def preset(self, preset_id: str) -> EconomicPreset:
        try:
            return self.presets[preset_id]
        except KeyError as exc:
            raise ProtocolError("unknown-preset", f"unknown Economic Preset: {preset_id}") from exc

    def validate_selection(
        self,
        *,
        preset_id: str,
        model_id: str,
        role: AgentRole,
        provider: str | None = None,
        harness: str | None = None,
    ) -> ModelCard:
        preset = self.preset(preset_id)
        try:
            card = self.cards[model_id]
        except KeyError as exc:
            # A refusal that only repeats the rejected name leaves the caller
            # guessing again, and a manager that cannot see the catalog guesses
            # from the words its user typed.  Name what this session can run,
            # for the role it was asked about, so the next attempt is informed.
            offered = sorted(
                name for name, offer in self.cards.items()
                if name in preset.allowed_models and role in offer.eligible_roles
            )
            available = (
                "; ".join(f"{name} ({self.cards[name].provider})" for name in offered)
                if offered else "no model in this session"
            )
            raise ProtocolError(
                "unknown-model",
                f"unknown model: {model_id}; "
                f"{role.value} can run on {available}",
            ) from exc
        if model_id not in preset.allowed_models:
            raise ProtocolError(
                "model-not-allowed",
                f"model {model_id} is outside Economic Preset {preset_id}",
            )
        if role not in card.eligible_roles:
            raise ProtocolError(
                "role-not-eligible",
                f"model {model_id} is not eligible for role {role.value}",
            )
        if provider is not None and card.provider != provider:
            raise ProtocolError(
                "provider-mismatch",
                f"model {model_id} resolves to provider {card.provider}, not {provider}",
            )
        if harness is not None and card.harness != harness:
            raise ProtocolError(
                "harness-mismatch",
                f"model {model_id} resolves to harness {card.harness}, not {harness}",
            )
        if preset.allowed_providers and card.provider not in preset.allowed_providers:
            raise ProtocolError(
                "provider-not-allowed",
                f"provider {card.provider} is outside Economic Preset {preset_id}",
            )
        return card

    def resolve_selection(
        self,
        *,
        preset_id: str,
        model_id: str,
        role: AgentRole,
        provider: str | None = None,
        harness: str | None = None,
    ) -> RuntimeSelection:
        card = self.validate_selection(
            preset_id=preset_id,
            model_id=model_id,
            role=role,
            provider=provider,
            harness=harness,
        )
        return RuntimeSelection(
            model_id=card.model_id,
            role=role,
            provider=card.provider,
            harness=card.harness,
            credential_location=card.credential_location,
        )


@dataclass(frozen=True)
class AgentMessage:
    sender_id: str
    target_id: str
    text: str
    kind: str
    created_at: float
    direct_override: bool = False


@dataclass
class AgentRecord:
    session_id: str
    agent_id: str
    parent_agent_id: str | None
    role: AgentRole
    model_id: str
    objective: str
    task_contract: dict[str, Any]
    # Effort is selected with the agent, rather than inferred forever from a
    # descriptive role.  Empty is retained only for old callers; the control
    # plane supplies the compatible high-effort default on creation.
    effort: str = "high"
    status: AgentStatus = AgentStatus.READY
    thread_id: str | None = None
    active_turn_id: str | None = None
    turn_count: int = 0
    current_activity: str = ""
    latest_progress: str = ""
    blocker: str = ""
    files_touched: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    # What the provider said the model alias means and which model answered.
    # Empty until a runtime reports it; ``model_id`` stays the alias.
    model_identity: dict[str, Any] = field(default_factory=dict)
    messages: list[AgentMessage] = field(default_factory=list)
    # How many of `messages` have been handed to the agent.  It lives
    # here rather than in a scheduler-side map so that every lifecycle
    # transition can see, under the session lock, whether it is about to
    # bury something nobody read.
    delivered_message_count: int = 0
    child_ids: list[str] = field(default_factory=list)
    # Empty means this agent works in the session workspace, which is the
    # default and what every agent did before private roots existed.  A
    # name means its manager asked for it to be isolated.  The name is an
    # ordinal, not an identifier, so it can appear inside a
    # workspace-relative path without leaking anything.
    workspace_dir: str = ""
    workspace_scope: str = "shared"
    approvals: str = "granted"
    retry_of_agent_id: str | None = None
    replaced_by_agent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProtocolEvent:
    event_id: int
    session_id: str
    agent_id: str
    event_type: str
    metadata: dict[str, Any]
    created_at: float


@dataclass
class OrchestrationSession:
    session_id: str
    preset_id: str
    workspace: Path
    objective: str
    root_agent_id: str
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    events: list[ProtocolEvent] = field(default_factory=list)
    waits: dict[str, set[str]] = field(default_factory=dict)
    wakes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    event_sequence: int = 0
    private_workspace_count: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)
    # Set once a close of this tree has begun, and never cleared.  It lives here
    # rather than on the service because this is the object ``lock`` guards, so
    # the same lock that makes a spawn atomic makes the check atomic with it:
    # a close cannot slip in between reading this flag and creating the child.
    closing: bool = False


SESSION_CLOSING_CODE = "session-closing"


def session_closing_message(tool: str) -> str:
    """The one sentence every path uses to refuse a call during a close.

    A client call refused at the server's own seam and a running child's call
    refused here describe the same state, so they say it with the same words.
    """

    return (
        f"the vNext session is closing, so {tool} was refused: it starts a "
        "provider turn and nothing would be left to stop it"
    )


class OrchestrationControlPlane:
    """Mechanical protocol host; all strategic choices are explicit inputs."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.sessions: dict[str, OrchestrationSession] = {}
        self.agent_sessions: dict[str, str] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        *,
        preset_id: str,
        root_model_id: str,
        workspace: str | Path,
        objective: str,
        task_contract: dict[str, Any],
        session_id: str | None = None,
        root_agent_id: str | None = None,
    ) -> AgentRecord:
        self.registry.validate_selection(
            preset_id=preset_id,
            model_id=root_model_id,
            role=AgentRole.ROOT_MANAGER,
        )
        resolved_workspace = Path(workspace).resolve()
        if not resolved_workspace.is_dir():
            raise ProtocolError("invalid-workspace", f"workspace is not a directory: {resolved_workspace}")
        with self._lock:
            actual_session_id = session_id or str(uuid.uuid4())
            if actual_session_id in self.sessions:
                raise ProtocolError("duplicate-session", f"session already exists: {actual_session_id}")
            root_id = root_agent_id or str(uuid.uuid4())
            if root_id in self.agent_sessions:
                raise ProtocolError("duplicate-agent", f"agent already belongs to a session: {root_id}")
            requested_effort = task_contract.get("effort")
            root = AgentRecord(
                session_id=actual_session_id,
                agent_id=root_id,
                parent_agent_id=None,
                role=AgentRole.ROOT_MANAGER,
                model_id=root_model_id,
                effort=requested_effort if isinstance(requested_effort, str) and requested_effort else "high",
                objective=objective,
                task_contract=dict(task_contract),
            )
            session = OrchestrationSession(
                session_id=actual_session_id,
                preset_id=preset_id,
                workspace=resolved_workspace,
                objective=objective,
                root_agent_id=root_id,
                agents={root_id: root},
            )
            self.sessions[actual_session_id] = session
            self.agent_sessions[root_id] = actual_session_id
            self._emit(session, root_id, "agent-created", {"role": root.role.value})
            return root

    def spawn_agent(
        self,
        *,
        requester_id: str,
        parent_agent_id: str,
        role: AgentRole,
        model_id: str,
        objective: str,
        task_contract: dict[str, Any],
        effort: str = "high",
        retry_of_agent_id: str | None = None,
        workspace_scope: str = "shared",
        approvals: str = "granted",
    ) -> AgentRecord:
        """Create one direct child of a live agent.

        ``workspace_scope`` is the manager's decision about where this child
        works, and it is one of ``WORKSPACE_SCOPES``.  ``shared`` is the session
        workspace, as every agent had before there was a choice: siblings can
        read and overwrite each other, which is what a manager wants when they
        are working on one tree together.  ``private`` is an empty working
        directory of the child's own, for producing something new on bare
        ground; it gives the child a starting folder and limits nothing its
        tools can reach.
        ``worktree`` is a git checkout of the child's own, for changing the same
        existing code as a sibling without colliding.

        Python supplies the three mechanisms and the runtime sandbox enforces
        the boundary; which one a child gets is not decided here.
        """
        session = self._session_for_agent(requester_id)
        with session.lock:
            self._refuse_if_closing(session, "delegate")
            requester = self._agent(session, requester_id)
            parent = self._agent(session, parent_agent_id)
            if requester.agent_id != parent.agent_id:
                raise ProtocolError("not-parent-agent", "an agent may spawn only its own direct child")
            # A refused delegate must leave the parent exactly as it was, so
            # the reopen happens after every check below and not here: a
            # completed root that asks for an invalid model stays COMPLETED.
            self._require_live_unless_reopenable_root(session, parent)
            # Roles describe the work an agent currently owns.  They do not
            # prescribe a fixed Root -> Branch -> Worker tree: any live agent
            # may both execute its task and create a direct child.  The registry
            # still checks that the selected model is eligible for the child's
            # explicit role, so this does not silently widen a user's preset.
            self.registry.validate_selection(
                preset_id=session.preset_id,
                model_id=model_id,
                role=role,
            )
            if workspace_scope not in WORKSPACE_SCOPES:
                raise ProtocolError(
                    "invalid-workspace-scope",
                    "workspace scope must be shared, private, or worktree",
                )
            if approvals not in {"granted", "ask"}:
                raise ProtocolError(
                    "invalid-approval-mode",
                    "approval mode must be granted or ask",
                )
            self._reopen_root_or_require_live(session, parent)
            workspace_dir = ""
            if workspace_scope != "shared":
                session.private_workspace_count += 1
                workspace_dir = f"agent-{session.private_workspace_count}"
            child_id = str(uuid.uuid4())
            child = AgentRecord(
                session_id=session.session_id,
                agent_id=child_id,
                parent_agent_id=parent.agent_id,
                role=role,
                model_id=model_id,
                effort=effort if isinstance(effort, str) and effort else "high",
                objective=objective,
                task_contract=dict(task_contract),
                workspace_dir=workspace_dir,
                workspace_scope=workspace_scope,
                approvals=approvals,
                retry_of_agent_id=retry_of_agent_id,
            )
            session.agents[child_id] = child
            self.agent_sessions[child_id] = session.session_id
            parent.child_ids.append(child_id)
            parent.updated_at = time.time()
            self._emit(
                session,
                child_id,
                "agent-created",
                {
                    "parent_agent_id": parent.agent_id,
                    "role": role.value,
                    "model_id": model_id,
                    "effort": child.effort,
                    "workspace_scope": workspace_scope,
                    "approvals": approvals,
                    # Which of the parent's turns this delegate call was made
                    # under. Two children carrying the same value is the whole
                    # claim of Phase 9 -- a manager fanning out rather than
                    # yielding after one -- and it is a fact the control plane
                    # holds rather than a judgement about the model.
                    "parent_turn_id": parent.active_turn_id,
                },
            )
            return child

    def restore_terminal_tree(self, session_id: str, agents: list[Mapping[str, Any]]) -> None:
        """Restore an idle or terminal tree without inventing provider work.

        This is intentionally narrower than provider reconnect.  A restarted
        service can reconstruct its durable control graph only when no agent
        has an active turn or a nonterminal wait.  Runtime adapters must later
        attest any provider-thread reconnection before controls are enabled.
        """

        with self._lock:
            try:
                session = self.sessions[session_id]
            except KeyError as exc:
                raise ProtocolError("invalid-session", "unknown session for restore") from exc
            with session.lock:
                if not isinstance(agents, list) or not agents:
                    raise ProtocolError("invalid-restore", "restored agent tree is required")
                supplied: dict[str, Mapping[str, Any]] = {}
                for item in agents:
                    if not isinstance(item, Mapping):
                        raise ProtocolError("invalid-restore", "restored agent must be an object")
                    agent_id = item.get("agent_id")
                    if not isinstance(agent_id, str) or not agent_id or agent_id in supplied:
                        raise ProtocolError("invalid-restore", "restored agent identity is malformed")
                    if item.get("turn_id") is not None:
                        raise ProtocolError("active-restore", "active turns require provider reconciliation")
                    supplied[agent_id] = item
                root_payload = supplied.get(session.root_agent_id)
                if root_payload is None or root_payload.get("parent_agent_id") not in {None, ""}:
                    raise ProtocolError("invalid-restore", "restored tree lacks its exact primary")

                # A child with a root of its own gets that same root back.  Restored
                # without it, a private child resolved to the shared workspace and
                # a worktree child's branch lost its name.  Only the folder names
                # spawn_agent writes are taken, so a damaged record cannot point a
                # child outside the session; a child whose name is missing gets
                # a fresh one past every name in use.
                claimed: dict[str, str] = {}
                highest = 0
                for agent_id, item in supplied.items():
                    if agent_id == session.root_agent_id or item.get("workspace_scope", "shared") == "shared":
                        continue
                    given = item.get("workspace_dir")
                    if not isinstance(given, str) or not given:
                        continue
                    found = _WORKSPACE_DIR_NAME.fullmatch(given)
                    if found is None or given in claimed.values():
                        raise ProtocolError("invalid-restore", "restored workspace folder is invalid")
                    claimed[agent_id] = given
                    highest = max(highest, int(found.group(1)))
                restored: dict[str, AgentRecord] = {}
                pending = dict(supplied)
                while pending:
                    progressed = False
                    for agent_id, item in tuple(pending.items()):
                        parent_id = item.get("parent_agent_id")
                        if agent_id != session.root_agent_id and parent_id not in restored:
                            continue
                        role_value = item.get("role")
                        model_id = item.get("model")
                        status_value = item.get("status")
                        if not isinstance(role_value, str) or not isinstance(model_id, str):
                            raise ProtocolError("invalid-restore", "restored role and model are required")
                        try:
                            role = AgentRole(role_value)
                            status = AgentStatus(status_value)
                        except (TypeError, ValueError) as exc:
                            raise ProtocolError("invalid-restore", "restored role or status is invalid") from exc
                        if status not in TERMINAL_STATUSES and status is not AgentStatus.READY:
                            raise ProtocolError("active-restore", "active agents require provider reconciliation")
                        if agent_id == session.root_agent_id:
                            if role is not AgentRole.ROOT_MANAGER or model_id != self._agent(session, agent_id).model_id:
                                raise ProtocolError("identity-restore", "restored primary identity does not match")
                            record = self._agent(session, agent_id)
                            record.status = status
                            record.thread_id = item.get("runtime_thread_id") if isinstance(item.get("runtime_thread_id"), str) else None
                            record.active_turn_id = None
                            record.effort = item.get("effort") if isinstance(item.get("effort"), str) else record.effort
                            record.child_ids.clear()
                        else:
                            if not isinstance(parent_id, str) or not parent_id:
                                raise ProtocolError("invalid-restore", "restored child parent is required")
                            self.registry.validate_selection(preset_id=session.preset_id, model_id=model_id, role=role)
                            scope = item.get("workspace_scope", "shared")
                            if scope not in WORKSPACE_SCOPES:
                                raise ProtocolError("invalid-restore", "restored workspace scope is invalid")
                            approvals = item.get("approvals", "granted")
                            if approvals not in {"granted", "ask"}:
                                raise ProtocolError("invalid-restore", "restored approval mode is invalid")
                            workspace_dir = ""
                            if scope != "shared":
                                workspace_dir = claimed.get(agent_id, "")
                                if not workspace_dir:
                                    highest += 1
                                    workspace_dir = f"agent-{highest}"
                            record = AgentRecord(
                                session_id=session_id,
                                agent_id=agent_id,
                                parent_agent_id=parent_id,
                                role=role,
                                model_id=model_id,
                                objective="",
                                task_contract={"restored": True},
                                effort=item.get("effort") if isinstance(item.get("effort"), str) else "high",
                                status=status,
                                thread_id=item.get("runtime_thread_id") if isinstance(item.get("runtime_thread_id"), str) else None,
                                workspace_scope=str(scope),
                                workspace_dir=workspace_dir,
                                approvals=str(approvals),
                            )
                            restored[parent_id].child_ids.append(agent_id)
                        restored[agent_id] = record
                        pending.pop(agent_id)
                        progressed = True
                    if not progressed:
                        raise ProtocolError("invalid-restore", "restored tree parentage is incomplete")
                session.agents = restored
                # This control plane can retain independent sessions. Replacing
                # its complete reverse index here would orphan every agent
                # outside the restored tree.
                self.agent_sessions = {
                    agent_id: owner_session_id
                    for agent_id, owner_session_id in self.agent_sessions.items()
                    if owner_session_id != session_id
                }
                self.agent_sessions.update({
                    agent_id: session_id
                    for agent_id in restored
                })
                session.private_workspace_count = highest

    def start_turn(self, agent_id: str, *, thread_id: str | None = None) -> str:
        """Admit one turn, or refuse it because the tree is closing.

        This is where a turn is admitted, and admission is one step: the same
        locked block reads ``closing`` and makes the READY to RUNNING
        transition, so a close cannot land between the two.  The scheduler's own
        reads of the flag come earlier and save a provider thread; they cannot
        carry the guarantee, because the prompt build between them and here can
        span the whole close.  A turn that wins this lock is already a running
        agent with an active turn id, which is what the close's workforce step
        looks at, so it is ended by the close rather than missed by it.  The
        provider is only asked to work after this returns, so the refusal costs
        no provider turn.
        """

        session = self._session_for_agent(agent_id)
        with session.lock:
            self._refuse_if_closing(session, "start_turn")
            agent = self._agent(session, agent_id)
            if agent.status is not AgentStatus.READY:
                raise ProtocolError("invalid-transition", f"cannot start a turn from {agent.status.value}")
            turn_id = str(uuid.uuid4())
            agent.status = AgentStatus.RUNNING
            agent.active_turn_id = turn_id
            agent.thread_id = thread_id or agent.thread_id
            agent.turn_count += 1
            agent.updated_at = time.time()
            self._emit(session, agent_id, "turn-started", {"turn_id": turn_id})
            return turn_id

    def attach_runtime_thread(self, agent_id: str, thread_id: str) -> None:
        """Attach one already-attested provider thread without starting it.

        Native harnesses can create a child before vNext has seen a turn on
        it.  The control plane needs that thread association for inspection and
        later turn correlation, but it must not manufacture a provider turn or
        change the child's READY lifecycle state.
        """

        if not isinstance(thread_id, str) or not thread_id:
            raise ProtocolError("invalid-thread", "runtime thread handle is required")
        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            self._require_live(agent)
            if agent.thread_id is not None and agent.thread_id != thread_id:
                raise ProtocolError("thread-already-bound", "agent already has a different runtime thread")
            agent.thread_id = thread_id
            agent.updated_at = time.time()
            self._emit(session, agent_id, "thread-attached", {})

    def resume_primary_for_native_turn(self, agent_id: str) -> None:
        """Reopen a completed persistent primary for an observed native turn.

        A native terminal supplies its own next user message directly to the
        provider.  It cannot first call ``send_user_message`` without creating
        a second, synthetic inbox entry, so the externally observed turn uses
        this narrow lifecycle-only resume instead.
        """

        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            if agent.agent_id != session.root_agent_id:
                raise ProtocolError("not-primary", "only the persistent primary may resume natively")
            if agent.status is AgentStatus.READY:
                return
            if agent.status is not AgentStatus.COMPLETED:
                raise ProtocolError(
                    "invalid-transition",
                    f"cannot resume native primary from {agent.status.value}",
                )
            agent.status = AgentStatus.READY
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent_id, "conversation-resumed", {"source": "native-terminal"})

    def resume_awaiting_agent_for_native_turn(self, agent_id: str, *, thread_id: str) -> None:
        """Reconcile one attested provider follow-up with a parked native manager.

        The scheduler calls this only after it has proved an unregistered
        provider turn on the agent's already leased native thread. This method
        deliberately does not accept a provider or turn identifier: it owns
        the durable lifecycle transition, while native identity and freshness
        remain scheduler evidence.
        """

        if not isinstance(thread_id, str) or not thread_id:
            raise ProtocolError("invalid-thread", "native runtime thread handle is required")
        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            if agent.status is not AgentStatus.AWAITING_WORKERS:
                raise ProtocolError(
                    "invalid-transition",
                    f"cannot resume native turn from {agent.status.value}",
                )
            if agent.thread_id != thread_id:
                raise ProtocolError(
                    "thread-mismatch",
                    "native turn does not belong to the awaiting agent thread",
                )
            if agent.active_turn_id is not None:
                raise ProtocolError(
                    "invalid-transition",
                    "awaiting agent unexpectedly has an active control turn",
                )
            watched = session.waits.get(agent_id)
            if not watched:
                raise ProtocolError(
                    "invalid-transition",
                    "awaiting agent has no registered child wait",
                )
            agent.status = AgentStatus.READY
            agent.updated_at = time.time()
            session.waits.pop(agent_id, None)
            self._emit(session, agent_id, "native-turn-resumed", {"source": "attested-native-turn"})

    def finish_turn(self, agent_id: str) -> None:
        """Record a runtime turn ending without a lifecycle tool decision."""

        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            if agent.status is not AgentStatus.RUNNING:
                raise ProtocolError(
                    "invalid-transition",
                    f"cannot finish a turn from {agent.status.value}",
                )
            agent.status = AgentStatus.READY
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent_id, "turn-finished", {})

    def await_agents(self, manager_id: str, watched_agent_ids: list[str]) -> str:
        session = self._session_for_agent(manager_id)
        with session.lock:
            manager = self._agent(session, manager_id)
            if manager.status not in {AgentStatus.RUNNING, AgentStatus.READY}:
                raise ProtocolError("invalid-transition", f"cannot wait from {manager.status.value}")
            watched: set[str] = set()
            for child_id in watched_agent_ids:
                child = self._agent(session, child_id)
                if child.parent_agent_id != manager_id:
                    raise ProtocolError("not-direct-child", "wait targets must be direct children")
                watched.add(child_id)
            manager.status = AgentStatus.AWAITING_WORKERS
            manager.active_turn_id = None
            manager.updated_at = time.time()
            session.waits[manager_id] = watched
            self._emit(session, manager_id, "wait-registered", {"watched_agent_ids": sorted(watched)})
            # A child event may arrive while the manager's runtime turn is
            # still active, immediately before it yields with await_children.
            # Preserve that already-queued wake instead of parking the manager
            # after the event that should resume it.
            #
            # Only a wake about a child this manager is now waiting on counts.
            # Wakes reach a manager whether or not it has parked, so its own
            # queue routinely holds news about children it is not waiting for --
            # including ones it cancelled itself moments earlier in this very
            # turn. Refusing to park for those would cost a whole extra turn,
            # with every child re-rendered, to be told something the manager
            # already knew and had not asked to be resumed for.
            pending = session.wakes.get(manager_id) or []
            if any(wake.get("source_agent_id") in watched for wake in pending):
                manager.status = AgentStatus.READY
                manager.updated_at = time.time()
                self._emit(session, manager_id, "wait-skipped-pending-wake", {})
                return "pending-wake"
            # Waiting on a child that is already stopped is a deadlock by
            # construction, and reading the wake queue is not enough to catch
            # it: a child can block while the manager's own turn is still in
            # flight, so the manager drains that wake in its next prompt and
            # then parks anyway, after the only event that could have released
            # it has been consumed. A blocked child is waiting for this manager
            # to decide something; a terminal one has nothing left to send. So
            # the state of the children is what is checked, not the mail.
            settled = [
                child_id
                for child_id in sorted(watched)
                if session.agents[child_id].status in SETTLED_STATUSES
            ]
            # Only when nothing being waited on can still move. One blocked
            # child among four that are still running is not a deadlock: the
            # manager parks, wakes when those four finish, and is refused then,
            # with everything settled and something to decide. Refusing while
            # any sibling could still report denied a legitimate park and cost
            # a turn every time round.
            if len(settled) == len(watched):
                manager.status = AgentStatus.READY
                manager.updated_at = time.time()
                self._emit(
                    session,
                    manager_id,
                    "wait-skipped-settled-child",
                    {"watched_agent_ids": settled},
                )
                return "settled-children"
            return "parked"

    def send_user_message(self, agent_id: str, text: str) -> None:
        """Queue a user-authored message for one live session agent.

        A completed Root remains the one deliberate exception: a new user
        objective reopens that persistent primary conversation. Completed
        children are never revived by a targeted prompt.
        """

        session = self._session_for_agent(agent_id)
        with session.lock:
            target = self._agent(session, agent_id)
            # Completing an objective is not the end of a persistent
            # conversation.  A later user message starts a new Root turn on
            # the same native session; completed children stay terminal.
            self._reopen_root_or_require_live(session, target)
            message = AgentMessage("user", agent_id, text, "user", time.time())
            target.messages.append(message)
            self._emit(session, agent_id, "message-queued", {"sender_id": "user", "kind": "user"})
            self._wake(session, agent_id, "user-message", source_agent_id="user")

    def request_attention(
        self,
        *,
        manager_id: str,
        source_agent_id: str,
        reason: str,
    ) -> None:
        """Wake a direct parent for a mechanical child lifecycle event."""

        session = self._session_for_agent(manager_id)
        with session.lock:
            manager = self._agent(session, manager_id)
            source = self._agent(session, source_agent_id)
            self._require_live(manager)
            self._require_live(source)
            if source.parent_agent_id != manager.agent_id:
                raise ProtocolError(
                    "not-direct-child",
                    "attention requests must come from a direct child",
                )
            self._wake(
                session,
                manager.agent_id,
                reason,
                source_agent_id=source.agent_id,
            )

    def message_agent(self, sender_id: str, target_id: str, text: str, *, kind: str = "steer") -> None:
        session = self._session_for_agent(sender_id)
        with session.lock:
            if kind == "message":
                # Only send_message queues this kind, and it wakes an idle
                # target into a fresh turn.  A steer, an interrupt and a status
                # request reach an agent that is already running and are what a
                # close needs in order to land, so they stay allowed.
                self._refuse_if_closing(session, "send_message")
            sender = self._agent(session, sender_id)
            target = self._agent(session, target_id)
            self._require_live(sender)
            self._require_live(target)
            # A shared vNext session is one collaboration space.  Parentage
            # remains authoritative for lifecycle and workspace ownership, but
            # it must not turn peer review into an impossible routed message.
            direct_override = sender_id != target.parent_agent_id
            message = AgentMessage(sender_id, target_id, text, kind, time.time(), direct_override)
            target.messages.append(message)
            self._emit(
                session,
                target_id,
                "message-queued",
                {"sender_id": sender_id, "kind": kind, "direct_override": direct_override},
            )
            self._wake(session, target_id, "manager-message", source_agent_id=sender_id)

    def request_status(self, manager_id: str, target_id: str) -> None:
        self.message_agent(manager_id, target_id, "Publish current progress or blocker.", kind="status-request")

    def record_progress(
        self,
        agent_id: str,
        *,
        activity: str,
        progress: str,
        files_touched: list[str] | None = None,
        commands: list[str] | None = None,
        usage: dict[str, Any] | None = None,
        material: bool = True,
    ) -> None:
        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            if agent.status is not AgentStatus.RUNNING:
                raise ProtocolError("invalid-transition", "progress requires an active turn")
            self._apply_progress(
                session,
                agent,
                activity=activity,
                progress=progress,
                files_touched=files_touched,
                commands=commands,
                usage=usage,
                material=material,
            )

    def _apply_progress(
        self,
        session: OrchestrationSession,
        agent: AgentRecord,
        *,
        activity: str,
        progress: str,
        files_touched: list[str] | None = None,
        commands: list[str] | None = None,
        usage: dict[str, Any] | None = None,
        material: bool = True,
    ) -> None:
        if agent.status is not AgentStatus.RUNNING:
            raise ProtocolError("invalid-transition", "progress requires an active turn")
        agent.current_activity = activity
        agent.latest_progress = progress
        agent.files_touched = list(files_touched or agent.files_touched)
        agent.commands = list(commands or agent.commands)
        agent.usage = ({**{key: agent.usage[key] for key in RUNNING_TOTAL_USAGE_KEYS
                          if key not in usage and key in agent.usage}, **usage}
                       if usage else dict(agent.usage))
        agent.updated_at = time.time()
        self._emit(session, agent.agent_id, "progress", {"material": material})
        if material:
            self._wake_watchers(session, agent.agent_id, "material-status")

    def block_agent(self, agent_id: str, blocker: str) -> None:
        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            self._require_live(agent)
            agent.status = AgentStatus.BLOCKED
            agent.blocker = blocker
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent_id, "blocked", {})
            self._wake_watchers(session, agent_id, "child-blocked")

    def release_unconfirmed_stop_block(self, agent_id: str) -> None:
        """Return one agent to READY after the turn it waited on reported its end.

        The scheduler blocked it because a predecessor's provider turn had said
        nothing inside the replacement barrier, and that turn has now reported
        it ended.  The reason for the block is gone, so this undoes exactly the
        transition ``block_agent`` made and nothing else: the manager is not
        woken, because there is no decision left for it to take.

        The state is checked rather than assumed.  An agent that went terminal
        by another route, or that a manager has already restarted, must not be
        dragged back to READY by late evidence about somebody else's turn.
        """

        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            if agent.status is not AgentStatus.BLOCKED:
                raise ProtocolError(
                    "invalid-transition",
                    f"cannot release a block from {agent.status.value}",
                )
            agent.status = AgentStatus.READY
            agent.blocker = ""
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(
                session, agent_id, "retry-ready", {"source": "predecessor-stop-confirmed"}
            )

    def fail_agent(self, agent_id: str, reason: str) -> None:
        """Finish an agent whose provider turn errored, preserving the cause."""

        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            self._require_live(agent)
            agent.status = AgentStatus.FAILED
            agent.blocker = reason
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent_id, "failed", {})
            self._wake_watchers(session, agent_id, "child-failed")

    def complete_agent(
        self,
        agent_id: str,
        result: dict[str, Any],
        *,
        require_messages_read: bool = False,
        progress: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a terminal outcome for an agent.

        With ``require_messages_read`` the completion is refused while the
        agent's queue has grown past what was delivered to it: finishing there
        would bury a message nobody ever read, and the sender would have no way
        to tell.  The check happens under the session lock, which is the lock
        ``message_agent`` also takes, so a message queued against a still-live
        agent either lands before the check and refuses the completion, or
        after the agent is terminal and is refused at the sender.  There is no
        window in between.

        Order matters here.  Every validation runs before anything is mutated
        or emitted, so a refused completion leaves no trace: no progress
        record, no wake, no half-applied state.  In particular liveness is
        checked before the message guard, or an agent that went terminal by
        another route would be refused with the wrong reason and the caller
        would retry a recovery that cannot work.
        """

        session = self._session_for_agent(agent_id)
        with session.lock:
            agent = self._agent(session, agent_id)
            unfinished = self._active_descendants(session, agent)
            if unfinished:
                raise ProtocolError(
                    "active-children",
                    f"agent still has active children: {unfinished}",
                )
            self._require_live(agent)
            if require_messages_read:
                unread = len(agent.messages) - agent.delivered_message_count
                if unread > 0:
                    raise ProtocolError(
                        "unread-messages",
                        f"{unread} message(s) addressed to this agent have not "
                        "been delivered yet",
                    )
            if progress is not None:
                # The final progress record belongs to the same transaction as
                # the completion.  Recorded separately it would wake the parent
                # about an agent that a refused completion leaves running.
                #
                # It is never material, whatever the caller passed.  The
                # completion below wakes the parent about this same event, and
                # a material progress record would wake it a second time for
                # the same news: two lines in the next prompt, and before wakes
                # were recorded unconditionally, two chances to un-park.
                final_progress = dict(progress)
                final_progress["material"] = False
                self._apply_progress(session, agent, **final_progress)
            agent.status = AgentStatus.COMPLETED
            agent.result = dict(result)
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent_id, "completed", {})
            self._wake_watchers(session, agent_id, "child-completed")

    def complete_branch(self, branch_manager_id: str, result: dict[str, Any]) -> None:
        session = self._session_for_agent(branch_manager_id)
        # Compatibility name for existing callers.  Branch is descriptive, so
        # completion is the same lifecycle transition any non-root agent uses.
        self.complete_agent(branch_manager_id, result)

    def retry_agent(
        self,
        *,
        requester_id: str,
        agent_id: str,
        revised_task_contract: dict[str, Any],
    ) -> AgentRecord:
        session = self._session_for_agent(requester_id)
        with session.lock:
            self._refuse_if_closing(session, "retry")
            requester = self._agent(session, requester_id)
            agent = self._agent(session, agent_id)
            if agent.parent_agent_id != requester.agent_id:
                raise ProtocolError("not-parent-manager", "only the parent manager may retry an agent")
            if agent.status not in {AgentStatus.BLOCKED, AgentStatus.FAILED}:
                raise ProtocolError("invalid-transition", "retry requires blocked or failed state")
            agent.task_contract = dict(revised_task_contract)
            agent.status = AgentStatus.READY
            agent.blocker = ""
            agent.updated_at = time.time()
            self._emit(session, agent_id, "retry-ready", {})
            return agent

    def replace_agent(
        self,
        *,
        requester_id: str,
        agent_id: str,
        model_id: str,
        objective: str | None = None,
        revised_task_contract: dict[str, Any],
    ) -> AgentRecord:
        session = self._session_for_agent(requester_id)
        with session.lock:
            self._refuse_if_closing(session, "replace")
            requester = self._agent(session, requester_id)
            old = self._agent(session, agent_id)
            if old.parent_agent_id != requester.agent_id:
                raise ProtocolError("not-replaceable-agent", "an agent may replace only its direct child")
            self._require_replaceable(old)
            # Validate the manager's explicit choice before changing the old
            # attempt. A rejected choice must leave useful work intact.
            self.registry.validate_selection(
                preset_id=session.preset_id,
                model_id=model_id,
                role=old.role,
            )
            unfinished = [
                child_id
                for child_id in old.child_ids
                if session.agents[child_id].status not in TERMINAL_STATUSES
            ]
            if unfinished:
                raise ProtocolError(
                    "active-children",
                    f"agent still has active children: {unfinished}",
                )
            # The requester is checked here and not left to ``spawn_agent``
            # below.  A manager that has already completed cannot take a new
            # child, and finding that out after the old attempt had been
            # flipped to REPLACED left a child marked replaced with nothing
            # replacing it.  Every refusal now happens before the first
            # mutation, so a refused replace changes nothing at all.
            self._require_live_unless_reopenable_root(session, requester)
            # Everything ``spawn_agent`` would reject is rejected here too, and
            # every conversion it would do is done here, while the tree is still
            # untouched.  A contract of ``None`` used to reach ``dict()`` inside
            # spawn_agent and raise TypeError with the old attempt already
            # flipped to REPLACED and nothing replacing it.
            if not isinstance(revised_task_contract, Mapping):
                raise ProtocolError(
                    "invalid-task-contract",
                    "a replacement needs a task contract object",
                )
            contract = dict(revised_task_contract)
            if objective is not None and (not isinstance(objective, str) or not objective):
                raise ProtocolError(
                    "invalid-objective",
                    "a replacement objective must be a non-empty string",
                )
            if old.workspace_scope not in WORKSPACE_SCOPES:
                raise ProtocolError(
                    "invalid-workspace-scope",
                    "workspace scope must be shared, private, or worktree",
                )
            if old.approvals not in {"granted", "ask"}:
                raise ProtocolError(
                    "invalid-approval-mode",
                    "approval mode must be granted or ask",
                )
            self._report_undelivered(session, old, "replaced")
            old.status = AgentStatus.REPLACED
            old.active_turn_id = None
            old.updated_at = time.time()
            self._emit(session, old.agent_id, "replaced", {})
            replacement = self.spawn_agent(
                requester_id=requester_id,
                parent_agent_id=requester_id,
                role=old.role,
                model_id=model_id,
                effort=old.effort,
                # An absent objective keeps the old one, which is what the
                # replace tool did for every caller before it could carry a
                # new one. An empty string is refused before it reaches here.
                objective=old.objective if objective is None else objective,
                task_contract=contract,
                retry_of_agent_id=old.agent_id,
                workspace_scope=old.workspace_scope,
                approvals=old.approvals,
            )
            old.replaced_by_agent_id = replacement.agent_id
            return replacement

    def cancel_agent(self, *, requester_id: str, agent_id: str) -> None:
        session = self._session_for_agent(requester_id)
        with session.lock:
            self._agent(session, agent_id)
            if requester_id != agent_id and not self._is_ancestor(session, requester_id, agent_id):
                raise ProtocolError("not-descendant", "cancellation target is outside the requester's subtree")
            self._cancel_subtree(session, agent_id)

    def inspect_agent(self, requester_id: str, agent_id: str, *, deep: bool = False) -> dict[str, Any]:
        session = self._session_for_agent(requester_id)
        with session.lock:
            self._authorize_inspection(session, requester_id, agent_id)
            return self._agent_view(session.agents[agent_id], deep=deep)

    def inspect_subtree(
        self,
        requester_id: str,
        agent_id: str,
        *,
        deep: bool = False,
    ) -> list[dict[str, Any]]:
        session = self._session_for_agent(requester_id)
        with session.lock:
            self._authorize_inspection(session, requester_id, agent_id)
            result: list[dict[str, Any]] = []
            pending = [agent_id]
            while pending:
                current = pending.pop(0)
                result.append(self._agent_view(session.agents[current], deep=deep))
                pending.extend(session.agents[current].child_ids)
            return result

    def drain_wakes(self, manager_id: str) -> list[dict[str, Any]]:
        session = self._session_for_agent(manager_id)
        with session.lock:
            return session.wakes.pop(manager_id, [])

    def restore_wakes(self, manager_id: str, wakes: list[dict[str, Any]]) -> None:
        """Put back wakes drained for a prompt that was never shown to anybody.

        Draining is how a prompt is built, so a runtime that then refuses to
        start the turn would otherwise consume the news and never deliver it.
        The mail offset and the children ledger are already restored for exactly
        this reason; a blocker is the one signal that most needs to survive it,
        since nothing else will report it a second time.

        Restored wakes go in front of anything that arrived meanwhile, because
        they happened first, and duplicates are dropped on the same rule the
        queue uses everywhere else.
        """

        if not wakes:
            return
        session = self._session_for_agent(manager_id)
        with session.lock:
            pending = session.wakes.get(manager_id, [])
            merged = list(wakes)
            for entry in pending:
                if entry not in merged:
                    merged.append(entry)
            session.wakes[manager_id] = merged

    def begin_closing(self) -> None:
        """Mark every session closing, so nothing can start a new turn.

        The service used to hold this state alone and check it where the client
        arrives.  A branch manager already running when the close began never
        passes that seam: its delegate goes from the provider handler into the
        scheduler and straight to a spawn.  So the flag lives on the session
        record and the checks sit inside the locked blocks that do the work,
        which is the one place both routes have to go through.
        """

        with self._lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            with session.lock:
                session.closing = True

    @staticmethod
    def _refuse_if_closing(session: OrchestrationSession, tool: str) -> None:
        """Refuse a call that would start a turn, before anything is changed.

        Callers hold ``session.lock`` when they call this and keep holding it
        through the work, so the answer cannot go stale between the two.
        """

        if session.closing:
            raise ProtocolError(SESSION_CLOSING_CODE, session_closing_message(tool))

    def _session_for_agent(self, agent_id: str) -> OrchestrationSession:
        try:
            return self.sessions[self.agent_sessions[agent_id]]
        except KeyError as exc:
            raise ProtocolError("invalid-handle", f"unknown agent handle: {agent_id}") from exc

    def _agent(self, session: OrchestrationSession, agent_id: str) -> AgentRecord:
        """The agent in this session, or a refusal that says which mistake it was.

        An id another session owns is a cross-session access.  An id no session
        has is a bad handle, the same answer cancel_agent gives, so a manager
        that reads error_code goes back to its own children list.
        """

        try:
            return session.agents[agent_id]
        except KeyError as exc:
            if agent_id not in self.agent_sessions:
                raise ProtocolError("invalid-handle", f"unknown agent handle: {agent_id}") from exc
            raise ProtocolError("cross-session-access", f"agent is not in session: {agent_id}") from exc

    def _report_undelivered(
        self,
        session: OrchestrationSession,
        agent: AgentRecord,
        reason: str,
    ) -> int:
        """Announce mail that a terminal transition is about to strand.

        A completion can be refused and retried, so the guarantee there is
        delivery.  A replacement or a cancellation cannot be: the agent is
        going away and it is a manager's decision that it should.  The mail
        still must not vanish in silence, so it leaves as an event carrying the
        count and the reason.  The text is not republished; the record says
        that something addressed to this agent was never read.
        """

        undelivered = len(agent.messages) - agent.delivered_message_count
        if undelivered <= 0:
            return 0
        self._emit(
            session,
            agent.agent_id,
            "messages-undelivered",
            {"count": undelivered, "reason": reason},
        )
        return undelivered

    def _active_descendants(
        self, session: OrchestrationSession, agent: AgentRecord
    ) -> list[str]:
        """Every agent under this one that has not finished, at any depth.

        FAILED is terminal and it does not stop the work underneath it.  A
        worker whose provider turn errored keeps the children it had, and they
        keep spending turns.  Reading direct children only therefore let a
        manager report its branch done over a grandchild that was still
        running.  Failing an agent does not cancel its subtree -- stopping a
        subtree is a manager's decision -- so the guard is the side that
        widens.
        """

        active: list[str] = []
        pending = list(agent.child_ids)
        while pending:
            current = session.agents[pending.pop(0)]
            if current.status not in TERMINAL_STATUSES:
                active.append(current.agent_id)
            pending.extend(current.child_ids)
        return active

    @staticmethod
    def _require_live(agent: AgentRecord) -> None:
        if agent.status in TERMINAL_STATUSES:
            raise ProtocolError("terminal-agent", f"agent is terminal: {agent.status.value}")

    @staticmethod
    def _require_replaceable(agent: AgentRecord) -> None:
        """Refuse a terminal child, and let a failed attempt change model.

        A provider that will not run the model at all ends the turn FAILED --
        "the 'gpt-6-astra' model requires a newer version of Codex" is the real
        message -- and the cure is another model, which is what replace is for.
        Refusing it left a manager with retry on the model just refused.
        ``retry_agent`` accepts the same pair for the same reason.  FAILED
        stays terminal elsewhere, so a failed child still counts as finished
        for the completion guard.
        """

        if agent.status in TERMINAL_STATUSES and agent.status not in REPLACEABLE_STATUSES:
            raise ProtocolError("terminal-agent", f"agent is terminal: {agent.status.value}")

    def _require_live_unless_reopenable_root(
        self, session: OrchestrationSession, agent: AgentRecord
    ) -> None:
        """Refuse a terminal agent now, and defer the reopen of a root.

        A caller with validation still to run wants the refusal early and the
        state change late, so a rejected request leaves a completed root
        completed instead of READY with a resumed conversation nobody asked
        for. It pairs with ``_reopen_root_or_require_live`` at the point where
        the call is certain to succeed.
        """

        if self._is_reopenable_root(session, agent):
            return
        self._require_live(agent)

    @staticmethod
    def _is_reopenable_root(session: OrchestrationSession, agent: AgentRecord) -> bool:
        return agent.agent_id == session.root_agent_id and agent.status is AgentStatus.COMPLETED

    def _reopen_root_or_require_live(
        self, session: OrchestrationSession, agent: AgentRecord
    ) -> None:
        """Let new work reopen a completed primary conversation.

        Completing an objective ends a turn, and it does not end the persistent
        conversation the primary holds.  A user message already reopened it; so
        does new work the primary decides to hand out, because a refusal there
        is worse than useless: the primary is the one client vNext cannot wake
        by itself, so "agent is terminal: completed" left it holding a task it
        could neither delegate nor hand back.  Only the root reopens.  A
        completed child stays terminal, which is what its manager's record of
        it means.
        """

        if self._is_reopenable_root(session, agent):
            agent.status = AgentStatus.READY
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent.agent_id, "conversation-resumed", {})
            return
        self._require_live(agent)

    @staticmethod
    def _is_ancestor(session: OrchestrationSession, ancestor_id: str, agent_id: str) -> bool:
        current = session.agents[agent_id]
        while current.parent_agent_id is not None:
            if current.parent_agent_id == ancestor_id:
                return True
            current = session.agents[current.parent_agent_id]
        return False

    def _authorize_inspection(
        self,
        session: OrchestrationSession,
        requester_id: str,
        agent_id: str,
    ) -> None:
        self._agent(session, requester_id)
        self._agent(session, agent_id)
        # Agents in one orchestration session can inspect each other to make
        # contextual review possible.  The session lookup above remains the
        # cross-session boundary; lifecycle ownership is still parent-based.
        return

    @staticmethod
    def _agent_view(agent: AgentRecord, *, deep: bool) -> dict[str, Any]:
        view: dict[str, Any] = {
            "session_id": agent.session_id,
            "agent_id": agent.agent_id,
            "parent_agent_id": agent.parent_agent_id,
            "role": agent.role.value,
            "model_id": agent.model_id,
            "effort": agent.effort,
            "objective": agent.objective,
            "status": agent.status.value,
            "current_activity": agent.current_activity,
            "latest_progress": agent.latest_progress,
            "blocker": agent.blocker,
            "workspace_dir": agent.workspace_dir,
            "workspace_scope": agent.workspace_scope,
            "approvals": agent.approvals,
            "files_touched": list(agent.files_touched),
            "commands": list(agent.commands),
            "usage": dict(agent.usage),
            "child_ids": list(agent.child_ids),
            "retry_of_agent_id": agent.retry_of_agent_id,
            "replaced_by_agent_id": agent.replaced_by_agent_id,
            "recent_messages": [
                {
                    "sender_id": message.sender_id,
                    "kind": message.kind,
                    "direct_override": message.direct_override,
                }
                for message in agent.messages[-5:]
            ],
        }
        view.update(identity_view(agent.model_identity))
        # A manager inspecting a finished child reads its report here, in the
        # agent object, whether or not it asked for the deep view.
        view.update(finished_agent_outcome(agent))
        if deep:
            view["task_contract"] = dict(agent.task_contract)
            view["result"] = dict(agent.result)
        return view

    def _wake_watchers(
        self,
        session: OrchestrationSession,
        source_agent_id: str,
        reason: str,
    ) -> None:
        """Tell a manager its child moved, and resume it only if it asked to be.

        These are two different guarantees and they used to be one.  A wake was
        recorded only for a manager that had already registered a wait, so a
        child that blocked while its manager was still working left no trace
        anywhere: by the time the manager's turn ended, the WAKES block was
        empty and the child's stall was invisible until someone thought to
        inspect.  That was tolerable while every manager parked the instant it
        had delegated.  Managers are now told to fan out and keep working, so
        the busy window is the ordinary case and the loss would be routine.

        So the record is unconditional and goes to the parent.  Un-parking
        stays conditional on the watch list, because ``await_children`` names
        the children whose movement is worth resuming for and that is the
        manager's judgement to make, not this layer's.  A child left off the
        list still reports; it just does not drag its manager back.
        """

        source = session.agents.get(source_agent_id)
        parent_id = source.parent_agent_id if source is not None else None
        if parent_id is None or parent_id not in session.agents:
            return
        if session.agents[parent_id].status in TERMINAL_STATUSES:
            # A cancelled subtree wakes every manager in it on the way down.
            # A manager that is already terminal will never take another turn
            # and so will never drain what it is handed; the entry would sit in
            # the session for good and show up as a wake for an agent that no
            # longer runs.
            return
        watched = session.waits.get(parent_id, set())
        self._wake(
            session,
            parent_id,
            reason,
            source_agent_id=source_agent_id,
            resume=source_agent_id in watched,
        )

    def _wake(
        self,
        session: OrchestrationSession,
        agent_id: str,
        reason: str,
        *,
        source_agent_id: str,
        resume: bool = True,
    ) -> None:
        agent = self._agent(session, agent_id)
        pending = session.wakes.setdefault(agent_id, [])
        entry = {"reason": reason, "source_agent_id": source_agent_id}
        # One child saying the same thing twice before its manager has read
        # either is one thing to tell it.  Every pending wake costs a line in
        # the next prompt, and repeats carry no information the first did not.
        if entry not in pending:
            pending.append(entry)
        if resume and agent.status is AgentStatus.AWAITING_WORKERS:
            agent.status = AgentStatus.READY
            agent.updated_at = time.time()
            # The wait is satisfied.  Left in place it would keep resuming this
            # manager for a set of children it has already moved past.
            session.waits.pop(agent_id, None)
        self._emit(session, agent_id, "wake", {"reason": reason, "source_agent_id": source_agent_id})

    def _cancel_subtree(
        self,
        session: OrchestrationSession,
        agent_id: str,
        *,
        notify_parent: bool = True,
    ) -> None:
        """Cancel this agent and everything under it.

        Only the top of the cancelled subtree tells its parent, because that
        parent is the one agent outside the subtree and so the only one that
        will ever take another turn to read it. Every manager inside is going
        away in the same operation; mail addressed to them would sit in the
        session unread for good and appear as wakes for agents that no longer
        run. Cancellation is depth-first, so a manager inside is still live at
        the moment its own child goes, which is why the decision is made here
        rather than by checking status at the point of the wake.
        """

        agent = self._agent(session, agent_id)
        for child_id in list(agent.child_ids):
            self._cancel_subtree(session, child_id, notify_parent=False)
        if agent.status not in TERMINAL_STATUSES:
            self._report_undelivered(session, agent, "cancelled")
            agent.status = AgentStatus.CANCELLED
            agent.active_turn_id = None
            agent.updated_at = time.time()
            self._emit(session, agent_id, "cancelled", {})
            if notify_parent:
                self._wake_watchers(session, agent_id, "child-cancelled")

    @staticmethod
    def _emit(
        session: OrchestrationSession,
        agent_id: str,
        event_type: str,
        metadata: dict[str, Any],
    ) -> None:
        session.event_sequence += 1
        session.events.append(
            ProtocolEvent(
                event_id=session.event_sequence,
                session_id=session.session_id,
                agent_id=agent_id,
                event_type=event_type,
                metadata=dict(metadata),
                created_at=time.time(),
            )
        )
