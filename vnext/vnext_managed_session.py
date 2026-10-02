"""Integrated vNext managed-session mechanics.

The control plane remains authoritative for hierarchy and agent lifecycle. The
selected runtime adapter remains authoritative for runtime threads/turns, native
tools, approvals, and completed-item effects. This module correlates those
facts and emits one unified, content-safe receipt; it contains no planning or
routing policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from vnext.process_supervisor import ProcessCleanup
from vnext.vnext_model_identity import read_identity
from vnext.vnext_runtime import (
    DynamicToolHandler,
    RuntimeAdapter,
    RuntimeCleanup,
    TurnHandle,
    effective_thread_policy,
    require_workspace_policy,
)
from vnext.vnext_orchestration import (
    AgentRecord,
    AgentRole,
    OrchestrationControlPlane,
    ProtocolEvent,
)
from vnext.vnext_claude_effort import CLAUDE_ACCEPTED_EFFORTS
from vnext.vnext_runtime_effects import RuntimeEffectJournal
from vnext.vnext_runtime_types import (
    NativeChildBinding,
    NativeChildObservation,
)

# Where a private Worker root is created, relative to the session workspace.
# Inside it rather than beside it, so a manager can still read what its
# Worker produced and every recorded path stays workspace-relative.
PRIVATE_WORKSPACE_PARENT = ".vnext"
# Measured on a Windows host with core.longpaths unset: a worktree root of
# 215 characters works and 216 fails with "$GIT_DIR too big". Files inside
# the checkout are fine well past it. The ceiling is on the root, and it is
# a MAX_PATH artifact, so worktree_preflight applies it on Windows alone.
_WORKTREE_ROOT_LIMIT = 215
# A separator plus ".vnext" plus a separator (8), a twelve-character token
# (12), a separator plus "agent-" (7), and the ordinal. That is 27 plus the
# ordinal's digits, which matches the measured 28 for a single-digit child --
# and is why the tenth own-workspace child of a session used to pass preflight
# and then fail at bind. The ordinal counts private and worktree children
# together, so four digits is the reservation.
_WORKTREE_ORDINAL_DIGITS = 4
_WORKTREE_PATH_OVERHEAD = 27 + _WORKTREE_ORDINAL_DIGITS


class ManagedSessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManagedTurn:
    agent_id: str
    phase: str
    control_turn_id: str
    provider: str
    runtime: TurnHandle


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class VNextManagedSession:
    """Deep correlation module for one top-level vNext session."""

    schema_version = "VNEXT_MANAGED_SESSION_V3"
    content_safe_schema_version = "VNEXT_CONTENT_SAFE_RECEIPT_V1"

    def __init__(
        self,
        *,
        control: OrchestrationControlPlane,
        adapter: RuntimeAdapter,
        session_id: str,
        runtime_effects: RuntimeEffectJournal | None = None,
        adapter_for_agent: Callable[[AgentRecord], RuntimeAdapter] | None = None,
    ) -> None:
        try:
            session = control.sessions[session_id]
        except KeyError as exc:
            raise ValueError("unknown orchestration session") from exc
        self.control = control
        self.adapter = adapter
        self.session_id = session_id
        self.workspace = session.workspace
        self.runtime_effects = runtime_effects or RuntimeEffectJournal(self.workspace)
        self._adapter_selector = adapter_for_agent
        self._adapters: dict[str, RuntimeAdapter] = {}
        self._threads: dict[str, str] = {}
        self._identity_attestations: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, DynamicToolHandler | None] = {}
        self._policies: dict[str, dict[str, Any]] = {}
        self._tool_registrations: dict[str, dict[str, Any]] = {}
        self._turns: list[dict[str, Any]] = []
        self._turn_indexes: dict[tuple[str, str, str], int] = {}
        # Provider turn identifiers are transport facts. The control-plane
        # turn id is the durable host handle used by lifecycle events and
        # typed commands. Scope the correlation by agent and provider: two
        # harnesses may use the same opaque native identifier.
        self._control_turns_by_native: dict[tuple[str, str, str], str] = {}
        self._managed_turns_by_native: dict[tuple[str, str, str], ManagedTurn] = {}
        # start_turn can synchronously receive a native tool callback before
        # its adapter call returns, so the owning thread must re-enter this
        # narrow registration rendezvous.
        self._turn_registration_locks: dict[str, Any] = {}
        self._pending_native_starts: dict[str, dict[str, str]] = {}
        # Provider thread IDs are opaque and may be reused by a different
        # harness, so native-child identity is scoped by provider and its
        # attested parent thread.  This map also makes duplicate observer
        # callbacks idempotent without ever launching a second thread.
        self._native_children_by_key: dict[tuple[str, str, str], NativeChildBinding] = {}
        self._native_child_registration_locks: dict[tuple[str, str, str], Any] = {}
        self._interruptions: list[dict[str, Any]] = []
        self._reconnects: list[dict[str, Any]] = []
        # True between the moment reconnect closes the old runtimes and the
        # moment it succeeds.  A failure in between leaves nothing live, and a
        # second attempt must say so rather than close them again.
        self._runtimes_discarded = False
        self._cleanups: list[dict[str, Any]] = []
        self._worktrees: dict[str, Path] = {}
        self._worktree_outcomes: list[dict[str, Any]] = []
        self._worktree_bases: dict[str, str] = {}
        self._worktree_creation_locks: dict[str, threading.Lock] = {}
        self._unknown_base_agents: set[str] = set()
        # How many characters the CHILDREN delta kept out of prompts. The
        # saving is the point of that change, so it is measured rather
        # than asserted.
        self.context_saved_characters = 0
        self._checks: dict[str, bool] = {}
        self._outcome: dict[str, Any] = {}
        self._closed = False
        self._lock = threading.RLock()

    def bind_thread(
        self,
        *,
        agent_id: str,
        thread_id: str,
        start_result: Mapping[str, Any],
        tool_handler: DynamicToolHandler | None,
        adapter: RuntimeAdapter | None = None,
    ) -> None:
        self._agent(agent_id)
        require_workspace_policy(start_result)
        runtime_adapter = adapter or self.adapter_for_agent(self._agent(agent_id))
        identity = self._identity_attestation(runtime_adapter, thread_id)
        with self._lock:
            if agent_id in self._threads:
                raise ManagedSessionError("agent already has a runtime thread")
            self._threads[agent_id] = thread_id
            self._handlers[agent_id] = tool_handler
            self._adapters[agent_id] = runtime_adapter
            if identity is not None:
                self._identity_attestations[agent_id] = identity
            self._policies[agent_id] = effective_thread_policy(start_result)
            registration = getattr(runtime_adapter, "tool_registration_attestation", None)
            if callable(registration):
                try:
                    value = registration(thread_id)
                except Exception:
                    value = None
                if isinstance(value, Mapping):
                    self._tool_registrations[agent_id] = {
                        key: value.get(key)
                        for key in (
                            "acknowledged",
                            "model_id",
                            "tool_count",
                            "tool_names",
                            "definition_sha256",
                            "handler_registered",
                        )
                    }

    def adapter_for_agent(self, agent: AgentRecord) -> RuntimeAdapter:
        """Resolve and cache the runtime selected for one agent."""

        with self._lock:
            selected = self._adapters.get(agent.agent_id)
        if selected is not None:
            return selected
        selected = self._adapter_selector(agent) if self._adapter_selector is not None else self.adapter
        if selected is None:
            raise ManagedSessionError("agent runtime selection returned no adapter")
        with self._lock:
            existing = self._adapters.setdefault(agent.agent_id, selected)
        return existing

    def _adapter_for(self, agent_id: str) -> RuntimeAdapter:
        agent = self._agent(agent_id)
        return self.adapter_for_agent(agent)

    def agent_for_thread(self, thread_id: str) -> str | None:
        with self._lock:
            return next(
                (agent_id for agent_id, value in self._threads.items() if value == thread_id),
                None,
            )

    def validate_effort_for_model(self, model_id: str, effort: str) -> None:
        """Refuse tiers the selected Claude-backed runtime cannot start."""

        card = self.control.registry.cards.get(model_id)
        if card is not None and card.provider in {"claude", "zai"} and effort not in CLAUDE_ACCEPTED_EFFORTS:
            accepted = ", ".join(sorted(CLAUDE_ACCEPTED_EFFORTS))
            raise ManagedSessionError(
                f"unsupported {card.provider} effort: {effort}; accepted values: {accepted}"
            )

    def spawn_from_manager(
        self,
        *,
        requester_id: str,
        role: AgentRole,
        arguments: Mapping[str, Any],
    ) -> AgentRecord:
        """Apply a manager's explicit child choice after mechanical validation."""
        objective = arguments.get("objective")
        model_id = arguments.get("model_id")
        task_contract = arguments.get("task_contract") or {}
        if not isinstance(objective, str) or not objective:
            raise ManagedSessionError("manager child objective is required")
        if not isinstance(model_id, str) or not model_id:
            raise ManagedSessionError("manager-selected model_id is required")
        if not isinstance(task_contract, dict):
            raise ManagedSessionError("manager task_contract must be an object")
        workspace_scope = arguments.get("workspace", "shared")
        if not isinstance(workspace_scope, str):
            raise ManagedSessionError("manager workspace scope must be a string")
        approvals = arguments.get("approvals", "granted")
        if not isinstance(approvals, str):
            raise ManagedSessionError("manager approval mode must be a string")
        effort = arguments.get("effort", "high")
        if not isinstance(effort, str) or not effort:
            raise ManagedSessionError("manager child effort must be a non-empty string")
        self.validate_effort_for_model(model_id, effort)
        if workspace_scope == "worktree":
            # Refuse here, where the refusal is a tool result the manager reads
            # and can route around, rather than later in the bind path where it
            # is an exception that kills the whole tree in silence.
            self.worktree_preflight()
        return self.control.spawn_agent(
            requester_id=requester_id,
            parent_agent_id=requester_id,
            role=role,
            model_id=model_id,
            objective=objective,
            task_contract=task_contract,
            effort=effort,
            workspace_scope=workspace_scope,
            approvals=approvals,
        )

    def adopt_native_child(
        self,
        observation: NativeChildObservation | Mapping[str, Any],
        *,
        tool_handler_factory: Callable[[AgentRecord], DynamicToolHandler | None],
    ) -> tuple[AgentRecord, NativeChildBinding, bool]:
        """Register an attested provider-created child without starting it.

        A provider adapter owns the native child thread.  This method creates
        the matching vNext node, attaches that existing thread to the parent
        adapter and returns the adapter's declared delivery capability.  It
        never calls ``start_thread`` or implies that an unavailable native
        message/interrupt path exists.
        """

        observed = self._native_child_observation(observation)
        if not observed.attested:
            raise ManagedSessionError("native child parentage is not attested")
        if not all(
            isinstance(value, str) and value
            for value in (
                observed.provider,
                observed.parent_agent_id,
                observed.parent_thread_id,
                observed.native_child_thread_id,
            )
        ):
            raise ManagedSessionError("native child observation lacks an attested identity")
        if observed.parent_thread_id == observed.native_child_thread_id:
            raise ManagedSessionError("native child thread cannot equal its parent thread")
        if not isinstance(observed.model_id, str) or not observed.model_id:
            raise ManagedSessionError("native child observation lacks an attested model")
        if not isinstance(observed.role, str) or not observed.role:
            raise ManagedSessionError("native child observation lacks an attested role")
        if not isinstance(observed.objective, str) or not observed.objective:
            raise ManagedSessionError("native child observation lacks an attested objective")
        try:
            role = AgentRole(observed.role)
        except ValueError as exc:
            raise ManagedSessionError("native child observation has an unsupported role") from exc
        contract = observed.task_contract or {}
        if not isinstance(contract, Mapping):
            raise ManagedSessionError("native child task contract is malformed")
        effort = observed.effort or "high"
        if not isinstance(effort, str) or not effort:
            raise ManagedSessionError("native child effort is malformed")

        parent = self._agent(observed.parent_agent_id)
        with self._lock:
            if self._threads.get(parent.agent_id) != observed.parent_thread_id:
                raise ManagedSessionError("native child parent thread is not bound to the named agent")
            if self._native_provider_scope(parent.agent_id) != observed.provider:
                raise ManagedSessionError("native child provider does not match its parent selection")
            adapter = self._adapters.get(parent.agent_id)
        if adapter is None:
            raise ManagedSessionError("native child parent has no bound runtime adapter")
        identity = self._identity_attestation(adapter, observed.native_child_thread_id)
        if identity.get("bound") is not True or identity.get("provider") != observed.provider:
            raise ManagedSessionError("native child thread identity is not attested by its provider")

        key = (observed.provider, observed.parent_thread_id, observed.native_child_thread_id)
        with self._native_child_registration_lock(key):
            with self._lock:
                existing = self._native_children_by_key.get(key)
            if existing is not None:
                return self._agent(existing.agent_id), existing, False

            child = self.control.spawn_agent(
                requester_id=parent.agent_id,
                parent_agent_id=parent.agent_id,
                role=role,
                model_id=observed.model_id,
                objective=observed.objective,
                task_contract=dict(contract),
                effort=effort,
            )
            delivery_contract = self._native_delivery_contract(observed)
            # A handler lets the child call the vNext coordination tools. Do
            # not hand one to an adapter that has not attested the two paths
            # those tools need: reading queued context and stopping an exact
            # active native turn. The child still remains visible, with the
            # adapter's unavailable capability stated in its binding.
            handler = (
                tool_handler_factory(child)
                if (
                    delivery_contract.get("context_messages") == "available"
                    and delivery_contract.get("interrupt") == "available"
                )
                else None
            )
            binding = NativeChildBinding(
                agent_id=child.agent_id,
                provider=observed.provider,
                native_thread_id=observed.native_child_thread_id,
                delivery_contract=delivery_contract,
                tool_handler=handler,
                unsupported_reason=observed.unsupported_reason,
            )
            try:
                self.runtime_effects.inherit_reader(parent_agent_id=parent.agent_id, child_agent_id=child.agent_id)
                self.control.attach_runtime_thread(child.agent_id, observed.native_child_thread_id)
                with self._lock:
                    self._threads[child.agent_id] = observed.native_child_thread_id
                    self._adapters[child.agent_id] = adapter
                    self._handlers[child.agent_id] = handler
                    self._identity_attestations[child.agent_id] = identity
                    # This is deliberately empty rather than a copied parent
                    # posture: observation establishes a thread identity, not
                    # a new claim about its sandbox or approvals.
                    self._policies[child.agent_id] = {}
                    self._native_children_by_key[key] = binding
            except BaseException:
                # A native child cannot be retried through a half-attached
                # READY record.  Preserve the agent and a legible lifecycle
                # state instead of silently starting another provider thread.
                self.control.block_agent(child.agent_id, "native child attachment failed")
                raise
            return child, binding, True

    def workspace_for(self, agent_id: str) -> Path:
        """Resolve where an agent works, without creating anything.

        This used to create the directory or the checkout as a side effect of
        being asked where it was, and that was wrong in a way only a real run
        showed. The receipt asks every agent for its path, and the receipt is
        built after the session has closed and settled its checkouts -- so
        writing the report recreated the very worktrees the run had just
        removed, and left them on disk permanently while reporting none
        outstanding. A second caller resurrected them mid-run.

        Resolution is now pure. ``ensure_workspace`` creates, and only the two
        places that are about to hand a root to a runtime call it.

        All roots sit inside the session workspace rather than beside it so a
        manager can still read what its Worker produced and every recorded path
        stays workspace-relative.
        """

        agent = self._agent(agent_id)
        if not agent.workspace_dir:
            return self.workspace
        return (
            self.workspace
            / PRIVATE_WORKSPACE_PARENT
            / self._session_workspace_token()
            / agent.workspace_dir
        )

    def ensure_workspace(self, agent_id: str) -> Path:
        """Resolve where an agent works and make sure it exists.

        Called only where a root is about to be handed to a runtime. A private
        agent gets an empty directory; a worktree agent gets a git checkout.
        """

        agent = self._agent(agent_id)
        root = self.workspace_for(agent_id)
        if root == self.workspace:
            return root
        # Before either scope, because a private child leaves the same
        # directory in the parent repository's sight. It stages plain files
        # rather than gitlinks, which is milder, but it is the same hole.
        self._shield_workspace_parent()
        if agent.workspace_scope == "worktree":
            self._ensure_worktree(agent_id, root)
            return root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def worktree_branch_for(self, agent_id: str) -> str:
        """The branch a worktree child owns.

        Every segment is a fixed name, an opaque digest, or an ordinal, so the
        branch may be reported to a manager without carrying content.
        """

        agent = self._agent(agent_id)
        if agent.workspace_scope != "worktree":
            return ""
        return "vnext/" + self._session_workspace_token() + "/" + agent.workspace_dir

    def worktree_preflight(self) -> None:
        """Refuse a worktree the workspace cannot actually give, and say why.

        Every one of these used to surface as an exception out of the bind
        path, which killed the root, the branch manager and every sibling. The
        manager that asked for the scope heard nothing at all. A manager that
        is told the rule re-routes; a manager whose tree dies cannot.

        Raised from the delegate path, so the text below is a tool result the
        manager reads and acts on.
        """

        # On Windows with core.longpaths unset, a worktree root longer than
        # 215 characters fails with "$GIT_DIR too big". The administrative
        # segments add to that root. This check goes first because it needs no
        # git at all, and because
        # failing inside `worktree add` leaves behind a branch nothing removes
        # and a name the same child can never be retried under.
        budget = _WORKTREE_ROOT_LIMIT - _WORKTREE_PATH_OVERHEAD
        # The ceiling is a Windows MAX_PATH artifact and was measured there.
        # macOS and Linux carry a path limit above a thousand characters, so
        # applying it here refused worktrees git creates without complaint.
        if os.name == "nt" and len(str(self.workspace)) > budget:
            raise ManagedSessionError(
                "the worktree scope needs a shorter workspace path on this "
                f"host: it adds {_WORKTREE_PATH_OVERHEAD} characters and the "
                f"limit leaves {budget}. Use shared or private instead."
            )
        try:
            toplevel = self._git("rev-parse", "--show-toplevel").strip()
        except ManagedSessionError as exc:
            # The remedy is the same either way, but the cause is not, and a
            # reader told "not a repository" goes looking at the repository
            # rather than at PATH.
            if "could not be run" in str(exc):
                raise ManagedSessionError(
                    "the worktree scope needs git, and git could not be run on "
                    "this host. Use shared or private instead."
                ) from exc
            raise ManagedSessionError(
                "the worktree scope needs the session workspace to be a git "
                "repository, and it is not. Use shared or private instead."
            ) from exc
        resolved = Path(toplevel).resolve() if toplevel else None
        if resolved is None or resolved != self.workspace.resolve():
            # git resolves upward, so a workspace nested inside a larger
            # repository would hand the child a checkout of everything above
            # it, and pollute the outer repository's status.
            raise ManagedSessionError(
                "the worktree scope needs the session workspace to be the root "
                "of its git repository; this one sits inside a larger "
                "repository. Use shared or private instead."
            )
        try:
            self._git("rev-parse", "HEAD")
        except ManagedSessionError as exc:
            raise ManagedSessionError(
                "the worktree scope seeds a child from the last commit, and "
                "this repository has none yet. Use shared or private instead."
            ) from exc

    def _is_own_checkout(self, root: Path) -> bool:
        """Whether ``root`` is the top of a checkout of this project's repository.

        The top of a checkout alone is not enough: a repository someone else
        cloned into the folder is a checkout too, and a child adopting it would
        work on another project while its manager believed it had this one.
        Both must share the repository that holds the session's commits.
        """

        try:
            top = self._git("rev-parse", "--show-toplevel", cwd=root).strip()
            common = self._git("rev-parse", "--git-common-dir", cwd=root).strip()
            ours = self._git("rev-parse", "--git-common-dir").strip()
        except ManagedSessionError:
            return False
        try:
            return Path(top).resolve() == root.resolve() and (
                (root / common).resolve() == (self.workspace / ours).resolve()
            )
        except OSError:
            return False

    def _ensure_worktree(self, agent_id: str, root: Path) -> None:
        """Create this child's checkout once, seeded from the committed tip.

        Seeding from a commit rather than from the parent's working tree is the
        sharp edge of this scope and it is deliberate: it is what Claude Code
        and pi both do, and it means the child cannot see work the parent has
        not committed.  That fact belongs in front of the manager choosing the
        scope, which is why the delegate tool says it in the description.

        Git is a hard dependency of this scope and of no other.  A failure is
        reported as what it is rather than falling back to an empty directory,
        which would look like a private workspace and silently lose the project
        the child was sent to work on.
        """

        # One lock per agent, held across the git call. The previous shape
        # checked under the lock, released it, and then shelled out, so two
        # threads routinely both reached `worktree add` for the same child:
        # the scheduler binds a new agent on the main loop while the manager's
        # own handler thread is still resolving that child's path for its tool
        # reply. The loser died on "reference already exists" and took the
        # whole tree with it. It is per agent rather than global because six
        # different children creating checkouts at once is fine.
        with self._lock:
            creation_lock = self._worktree_creation_locks.setdefault(
                agent_id, threading.Lock()
            )
        with creation_lock:
            with self._lock:
                if agent_id in self._worktrees:
                    return
            if root.exists() and not self._is_own_checkout(root):
                # A directory git does not know as a checkout is what a failed
                # `worktree add` or a crash leaves behind.  Adopted, it handed
                # the child an empty folder in place of the project.  An empty
                # one is cleared and the checkout made; one with files in it
                # may be someone's work and is left for a person to look at.
                if any(root.iterdir()):
                    raise ManagedSessionError(
                        f"the worktree folder {root.name} for agent {agent_id} exists "
                        "and is not a git checkout; move it aside and start the child again"
                    )
                root.rmdir()
            if root.exists():
                # An existing checkout is adopted rather than rebuilt, and it
                # needs a base like any other. Without one the head comparison
                # drops out of the change rule entirely, so a child that
                # committed all of its work has a clean tree, nothing to
                # compare against, and is judged to have left nothing --
                # destroying a fully committed deliverable.
                with self._lock:
                    self._worktrees[agent_id] = root
                    if agent_id not in self._worktree_bases:
                        try:
                            self._worktree_bases[agent_id] = self._git(
                                "rev-parse", "HEAD"
                            ).strip()
                        except ManagedSessionError:
                            # No base means the head comparison cannot run, and
                            # the safe reading of "cannot tell" is "changed".
                            self._worktree_bases[agent_id] = ""
                            self._unknown_base_agents.add(agent_id)
                return
            branch = self.worktree_branch_for(agent_id)
            root.parent.mkdir(parents=True, exist_ok=True)
            base = self._git("rev-parse", "HEAD").strip()
            try:
                self._git("worktree", "add", "-b", branch, str(root), base)
            except ManagedSessionError:
                # `worktree add` can fail after creating the branch, and a
                # branch nothing recorded is a branch nothing will ever remove
                # -- and the same child can never be retried, because the name
                # is taken.
                for cleanup in (("worktree", "prune"), ("branch", "-D", branch)):
                    try:
                        self._git(*cleanup)
                    except ManagedSessionError:
                        continue
                raise
            with self._lock:
                self._worktrees[agent_id] = root
                self._worktree_bases[agent_id] = base

    def _shield_workspace_parent(self) -> None:
        """Keep agent roots out of the enclosing repository's sight.

        Nothing ignores the administrative directory, so the parent repository
        reports it as untracked, and a sibling Worker in the shared workspace
        running the ordinary `git add -A` commits live checkouts as submodule
        gitlinks into the user's real history -- pointing at branches this run
        will later delete, and seeding the next child with a checkout nested
        inside its own.

        An ignore file inside the directory does it without touching anything
        the user owns.
        """

        parent = self.workspace / PRIVATE_WORKSPACE_PARENT
        try:
            parent.mkdir(parents=True, exist_ok=True)
            marker = parent / ".gitignore"
            if not marker.exists():
                marker.write_text("*\n", encoding="utf-8")
        except OSError:
            # A shield that cannot be written is a tidiness problem, not a
            # reason to refuse the child its workspace.
            pass

    def worktree_outcome(self, agent_id: str) -> dict[str, Any]:
        """What a worktree child left behind, and whether it survives.

        The rule is the one Claude Code uses and it is cheap to copy: a child
        that changed nothing has its checkout and its branch removed, because
        keeping them fills the disk with empty copies of the project.  A child
        that changed something keeps both, and the path and the branch are
        reported to its manager.

        Merging is not solved here and is not attempted.  The manager is handed
        a path and a branch and decides what to do with them, which is exactly
        where the two systems this was copied from stop.
        """

        with self._lock:
            root = self._worktrees.get(agent_id)
            base = self._worktree_bases.get(agent_id, "")
        if root is None:
            return {}

        def recorded(outcome):
            # A Worker settles its own checkout when it completes, long before
            # the session closes. Recording only what close() settles left the
            # receipt silent about every checkout already dealt with, which in
            # a run that went well is all of them.
            with self._lock:
                self._worktree_outcomes = [
                    item
                    for item in self._worktree_outcomes
                    if item.get("agent_id") != agent_id
                ] + [dict(outcome, agent_id=agent_id)]
            return outcome
        branch = self.worktree_branch_for(agent_id)
        relative = root.relative_to(self.workspace).as_posix()
        try:
            # --ignored and -uall on purpose. Plain `status --porcelain` is
            # built to hide files, and everything it hides is something a
            # Worker may have been told to produce: this project's own ignore
            # list covers reports/, data/ and prototypes/. A child told to
            # write its analysis into reports/ looked untouched, and its whole
            # deliverable was force-deleted while its manager was told it left
            # nothing. -uall also defeats a repository or global
            # status.showUntrackedFiles=no, under which every new file was
            # invisible.
            #
            # The cost of the other error is a checkout kept that could have
            # been removed. That is disk; the other is a child's work.
            dirty = bool(
                self._git(
                    "status",
                    "--porcelain",
                    "--ignored=matching",
                    "--untracked-files=all",
                    cwd=root,
                ).strip()
            )
            head = self._git("rev-parse", "HEAD", cwd=root).strip()
            stashed, unattributable_stash = self._stash_state(branch)
        except ManagedSessionError:
            # An unreadable checkout is not evidence that nothing changed, so it
            # is kept and reported rather than removed.
            return recorded({
                "scope": "worktree",
                "path": relative,
                "branch": branch,
                "changed": True,
                "retained": True,
                "note": "checkout could not be read; kept for inspection",
            })
        # A tidy child that stashed its work leaves a clean tree at base. The
        # stash object survives in the shared git directory, so the work is
        # recoverable, but nothing would have told anyone to look for it.
        # No base is not evidence of no change; it is the absence of the
        # comparison. An unattributable stash is the same shape: work exists
        # somewhere in this repository and nothing can prove it is not this
        # child's. Both resolve to "changed", because the other direction
        # deletes a checkout to save disk.
        cannot_tell = agent_id in self._unknown_base_agents or unattributable_stash
        if base:
            # A base is known now, so whatever made it unknown is over. The
            # discard in _remove_worktree could never run for a flagged agent,
            # because the flag makes it changed and only the unchanged path
            # removes anything.
            with self._lock:
                self._unknown_base_agents.discard(agent_id)
            cannot_tell = unattributable_stash
        changed = dirty or stashed or cannot_tell or (bool(base) and head != base)
        if changed:
            return recorded({
                "scope": "worktree",
                "path": relative,
                "branch": branch,
                "changed": True,
                "retained": True,
                "uncommitted_changes": dirty,
                "stashed_changes": stashed,
                "change_undetermined": cannot_tell,
                "note": (
                    "This checkout and its branch are yours to merge, discard, "
                    "or hand on. Nothing merges it for you."
                    + (" Some of its work is in a git stash." if stashed else "")
                ),
            })
        if self._remove_worktree(agent_id, root, branch):
            return recorded({
                "scope": "worktree",
                "path": relative,
                "branch": branch,
                "changed": False,
                "retained": False,
                "note": "left nothing, so the checkout and its branch were removed",
            })
        return recorded({
            "scope": "worktree",
            "path": relative,
            "branch": branch,
            "changed": False,
            "retained": True,
            "note": (
                "left nothing, but its checkout could not be removed and is "
                "still on disk; its branch was kept so git can still reach it"
            ),
        })

    def _stash_state(self, branch: str) -> tuple[bool, bool]:
        """Whether this branch has a stash, and whether one cannot be placed.

        Stashes live in the shared git directory, not in the checkout, so they
        have to be attributed by the branch each was taken on. Matching the
        branch name anywhere in the line was wrong twice over: "agent-1" is a
        prefix of "agent-10", so the first child inherited the tenth child's
        stash and kept an empty checkout forever; and a stash taken from a
        detached HEAD reads "On (no branch)", belongs to nobody, and was
        silently ignored while its checkout was deleted.

        The second value says a stash exists that cannot be attributed. The
        caller reads that as "cannot tell", which resolves to changed.
        """

        try:
            lines = self._git("stash", "list").splitlines()
        except ManagedSessionError:
            # Every other read in this rule treats a failure as cannot-tell.
            # This one used to read it as no-change, which is the direction
            # that deletes a checkout.
            return False, True
        mine = False
        unattributable = False
        for line in lines:
            # "stash@{0}: WIP on main: a90dc37 docs: On call runbook". Anchor
            # on the entry prefix and then on the form, because the tip
            # commit's subject is part of the line and can contain anything.
            # Splitting on ": On " anywhere in the line matched inside a commit
            # subject and attributed the stash to a branch called "call
            # runbook", which let a real stash fall into the ignore bucket and
            # its checkout be destroyed.
            _, separator, entry = line.partition("}: ")
            if not separator:
                unattributable = True
                continue
            if entry.startswith("WIP on "):
                remainder = entry[len("WIP on "):]
            elif entry.startswith("On "):
                remainder = entry[len("On "):]
            else:
                # A bare reflog message, which `git stash store` writes and
                # nothing else does. The name says nothing, so ask the object:
                # a stash's first parent is the commit it was taken from. If
                # that is not this branch's tip, it is not this child's, and
                # treating it as unattributable would retain every checkout in
                # the repository for one plumbing command nobody typed.
                if not self._stash_may_belong_to(line, branch):
                    continue
                unattributable = True
                continue
            # A git refname may contain neither a space nor a colon, so the
            # first colon ends the branch.
            taken_on = remainder.split(":", 1)[0].strip()
            if taken_on == branch:
                mine = True
            elif taken_on.startswith("(") or not taken_on:
                unattributable = True
        return mine, unattributable

    def _stash_may_belong_to(self, line: str, branch: str) -> bool:
        """Whether a stash whose name says nothing could be this branch's.

        A stash commit's first parent is the HEAD it was taken from. Comparing
        that to the branch tip answers the question the message could not, and
        a read that fails keeps the conservative answer.
        """

        selector = line.split(":", 1)[0].strip()
        if not selector:
            return True
        try:
            taken_from = self._git("rev-parse", selector + "^1").strip()
            tip = self._git("rev-parse", branch).strip()
        except ManagedSessionError:
            return True
        return bool(taken_from) and taken_from == tip

    def _remove_worktree(self, agent_id: str, root: Path, branch: str) -> bool:
        """Remove a checkout and its branch. Says whether it actually managed.

        The three commands used to run unconditionally with every failure
        swallowed, so a checkout still held open by a live process -- which is
        its ordinary state, since a Worker's runtime is not closed until the
        session is -- kept its directory, lost the branch that pointed at it,
        and was reported as removed. Orphaned beyond git's reach and
        misreported at the same time.

        The branch is now only deleted once the checkout is gone.
        """

        try:
            self._git("worktree", "remove", "--force", str(root))
        except ManagedSessionError:
            # Cleanup is best effort by design. A checkout left on disk is a
            # tidiness problem; raising would turn it into a failed run for a
            # child whose work was fine. But the branch stays, because it is
            # the only handle left on the directory.
            return False
        for args in (("branch", "-D", branch), ("worktree", "prune")):
            try:
                self._git(*args)
            except ManagedSessionError:
                continue
        with self._lock:
            self._worktrees.pop(agent_id, None)
            self._worktree_bases.pop(agent_id, None)
            self._worktree_creation_locks.pop(agent_id, None)
            self._unknown_base_agents.discard(agent_id)
        return True

    def close_worktrees(self) -> list[dict[str, Any]]:
        """Settle every checkout still open when the session ends."""

        with self._lock:
            outstanding = list(self._worktrees)
        settled = []
        for agent_id in outstanding:
            try:
                outcome = self.worktree_outcome(agent_id)
            except Exception as exc:
                # A checkout that vanished under us -- a sibling's clean, a
                # user tidying -- must not cost the whole session its receipt,
                # which is the only record of everything else that happened.
                unsettled = {
                    "scope": "worktree",
                    "agent_id": agent_id,
                    "path": self.relative_workspace_for(agent_id),
                    "branch": self.worktree_branch_for(agent_id),
                    "changed": True,
                    "retained": True,
                    "note": f"could not be settled: {type(exc).__name__}",
                }
                with self._lock:
                    self._worktree_outcomes = [
                        item
                        for item in self._worktree_outcomes
                        if item.get("agent_id") != agent_id
                    ] + [unsettled]
                settled.append(unsettled)
                continue
            if outcome:
                settled.append(outcome)
        return settled

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(cwd or self.workspace),
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            # git missing from PATH, or a working directory that no longer
            # exists, both arrive here as OSError. Left unconverted they escape
            # every handler in this module, which all catch ManagedSessionError,
            # and kill the session on the way out.
            raise ManagedSessionError(
                "git " + args[0] + " could not be run in the session workspace"
            ) from exc
        if completed.returncode != 0:
            # The message names the subcommand and never the output, which may
            # carry file content or paths from outside the workspace.
            raise ManagedSessionError("git " + args[0] + " failed in the session workspace")
        return completed.stdout

    def relative_workspace_for(self, agent_id: str) -> str:
        """Where an agent works, as a path relative to the session workspace.

        Empty for a shared agent, which works in the session workspace itself.
        Content-safe by construction: every segment is either a fixed name, an
        opaque digest, or an ordinal.
        """

        root = self.workspace_for(agent_id)
        if root == self.workspace:
            return ""
        return root.relative_to(self.workspace).as_posix()

    @staticmethod
    def _require_same_identity(
        previous: Mapping[str, Any] | None,
        resumed: Mapping[str, Any],
        thread_id: str,
    ) -> None:
        """Refuse a resumed node that is not the node that went away.

        The thread handle and the provider must match what was attested before
        the restart, and the provider session too when there was one.  Reading
        the successor's own attestation and checking only that it is bound
        proves the new runtime attests something, not that it attests the same
        agent.
        """

        if resumed.get("runtime_thread") != thread_id:
            raise ManagedSessionError(
                "resumed runtime identity mismatches persistent attestation"
            )
        if previous is None:
            return
        if resumed.get("provider") != previous.get("provider"):
            raise ManagedSessionError(
                "resumed runtime identity mismatches persistent attestation"
            )
        earlier_session = previous.get("provider_session")
        if earlier_session and resumed.get("provider_session") != earlier_session:
            raise ManagedSessionError(
                "resumed runtime identity mismatches persistent attestation"
            )

    def _session_workspace_token(self) -> str:
        """An opaque, stable per-session segment for private roots.

        Without it the ordinal restarts at one for every run, so the second run
        against a project would hand its first private Worker the first run's
        directory with that run's files still in it -- silent contamination in
        the one place isolation was promised.  A digest rather than the session
        id, because this segment appears in recorded paths and a raw identifier
        may not.
        """

        return hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:12]

    def start_turn(
        self,
        agent_id: str,
        *,
        prompt: str,
        effort: str,
        phase: str,
        turn_timeout: float | None = None,
    ) -> ManagedTurn:
        agent = self._agent(agent_id)
        try:
            thread_id = self._threads[agent_id]
        except KeyError as exc:
            raise ManagedSessionError("agent has no bound runtime thread") from exc
        # ``thread_id`` is a managed local reservation.  A provider may only
        # attest its native identity after this first request, so the initial
        # turn remains runnable against the reservation.
        with self._turn_registration_lock(agent_id):
            control_turn_id = self.control.start_turn(agent_id, thread_id=thread_id)
            approvals_reviewer = str(
                self._policies.get(agent_id, {}).get("reviewer") or "auto_review"
            )
            runtime_adapter = self._adapter_for(agent_id)
            provider = self._native_provider_scope(agent_id)
            with self._lock:
                self._pending_native_starts[agent_id] = {
                    "thread_id": thread_id,
                    "provider": provider,
                    "control_turn_id": control_turn_id,
                    "phase": phase,
                }
            try:
                runtime = runtime_adapter.start_turn(
                    thread_id=thread_id,
                    prompt=prompt,
                    model=agent.model_id,
                    effort=effort,
                    approvals_reviewer=approvals_reviewer,
                    workspace=self.ensure_workspace(agent_id),
                    turn_timeout=turn_timeout,
                )
            except BaseException:
                with self._lock:
                    self._pending_native_starts.pop(agent_id, None)
                self.control.block_agent(agent_id, "runtime turn start failed")
                raise
            self._refresh_identity(agent_id, thread_id)
            key = (agent_id, provider, runtime.turn_id)
            with self._lock:
                provisional = self._managed_turns_by_native.get(key)
                self._pending_native_starts.pop(agent_id, None)
            if provisional is not None:
                if provisional.control_turn_id != control_turn_id:
                    raise ManagedSessionError("native turn start mismatches the control turn")
                return self._finalize_registered_turn(provisional, runtime)
            return self._register_turn(
                agent_id=agent_id,
                provider=provider,
                phase=phase,
                control_turn_id=control_turn_id,
                runtime=runtime,
            )

    def adopt_native_turn(
        self,
        *,
        agent_id: str,
        provider: str,
        runtime: TurnHandle,
        phase: str = "native-external",
    ) -> tuple[ManagedTurn, bool]:
        """Adopt a provider-observed turn on an already owned thread.

        ``True`` means this call made the control-plane turn and callers must
        begin waiting for it. A repeat notification for the exact native turn
        returns the original turn with ``False``. Unknown threads, providers,
        and a second distinct active turn are rejected rather than attributed
        to an arbitrary session agent.
        """

        agent = self._agent(agent_id)
        expected_thread = self._threads.get(agent_id)
        if expected_thread != runtime.thread_id:
            raise ManagedSessionError("native turn does not belong to the agent's bound thread")
        expected_provider = self._native_provider_scope(agent_id)
        if provider != expected_provider:
            raise ManagedSessionError("native turn provider does not match the agent selection")
        key = (agent_id, provider, runtime.turn_id)
        with self._turn_registration_lock(agent_id):
            with self._lock:
                existing = self._managed_turns_by_native.get(key)
                if existing is not None:
                    return existing, False
                pending = self._pending_native_starts.get(agent_id)
            if pending is not None:
                if (
                    pending["thread_id"] != runtime.thread_id
                    or pending["provider"] != provider
                ):
                    raise ManagedSessionError("native turn conflicts with an in-flight scheduler start")
                return self._register_turn(
                    agent_id=agent_id,
                    provider=provider,
                    phase=pending["phase"],
                    control_turn_id=pending["control_turn_id"],
                    runtime=runtime,
                ), False
            control_turn_id = self.control.start_turn(agent_id, thread_id=runtime.thread_id)
            return self._register_turn(
                agent_id=agent_id,
                provider=provider,
                phase=phase,
                control_turn_id=control_turn_id,
                runtime=runtime,
            ), True

    def _register_turn(
        self,
        *,
        agent_id: str,
        provider: str,
        phase: str,
        control_turn_id: str,
        runtime: TurnHandle,
    ) -> ManagedTurn:
        turn = ManagedTurn(agent_id, phase, control_turn_id, provider, runtime)
        key = (agent_id, provider, runtime.turn_id)
        with self._lock:
            self._turn_indexes[key] = len(self._turns)
            self._control_turns_by_native[key] = control_turn_id
            self._managed_turns_by_native[key] = turn
            self._turns.append(
                {
                    "agent_id": agent_id,
                    "phase": phase,
                    "control_turn_id": control_turn_id,
                    "runtime_turn_id": runtime.turn_id,
                    "provider": provider,
                    "status": "running",
                    # Two agents holding turns at the same instant cannot be
                    # read off an ordered list of starts and finishes, because
                    # nothing in it says whether one ended before the other
                    # began. An interval says it directly.
                    "started_at": time.time(),
                    "finished_at": None,
                    "tools": [],
                }
            )
        return turn

    def native_turn_starting(
        self, *, agent_id: str, provider: str, thread_id: str
    ) -> bool:
        """Whether scheduler start has reserved this bound native thread."""

        with self._lock:
            pending = self._pending_native_starts.get(agent_id)
            return bool(
                pending
                and pending["provider"] == provider
                and pending["thread_id"] == thread_id
            )

    def _finalize_registered_turn(self, provisional: ManagedTurn, runtime: TurnHandle) -> ManagedTurn:
        """Replace a callback-created handle with the adapter's final cursor."""

        final = ManagedTurn(
            provisional.agent_id,
            provisional.phase,
            provisional.control_turn_id,
            provisional.provider,
            runtime,
        )
        key = (final.agent_id, final.provider, final.runtime.turn_id)
        with self._lock:
            self._managed_turns_by_native[key] = final
        return final

    def control_turn_for_native(
        self,
        *,
        agent_id: str,
        provider: str,
        native_turn_id: str,
    ) -> str | None:
        """Return the durable control turn for one provider-observed turn.

        This is a read-only lookup for event projection. Service events use
        the returned id while retaining the native id as event evidence.
        """

        with self._lock:
            return self._control_turns_by_native.get(
                (str(agent_id), str(provider), str(native_turn_id))
            )

    def wait_turn(self, turn: ManagedTurn, *, timeout: float = 300) -> dict[str, Any]:
        runtime_adapter = self._adapter_for(turn.agent_id)
        result = runtime_adapter.wait_turn(turn.runtime, timeout=timeout)
        events_since = getattr(runtime_adapter, "events_since", None)
        events = events_since(turn.runtime.cursor) if callable(events_since) else ()
        self.runtime_effects.observe_turn(
            agent_id=turn.agent_id,
            thread_id=turn.runtime.thread_id,
            turn_id=turn.runtime.turn_id,
            # The adapter owns native event slicing and its selected reader
            # owns native correlation.  This generic layer only forwards an
            # opaque cursor slice to the already-bound reader.
            events=events,
        )
        # Effects remain correlated to the local reservation until the
        # provider emits a genuine identity.  Re-reading here binds late
        # identities without relabeling buffered effects.
        self._refresh_identity(turn.agent_id, turn.runtime.thread_id)
        with self._lock:
            index = self._turn_indexes[(turn.agent_id, turn.provider, turn.runtime.turn_id)]
            self._turns[index]["status"] = str(result.get("status") or "unknown")
            self._turns[index]["finished_at"] = time.time()
        return result

    def turn_process_ended(self, turn: ManagedTurn) -> bool | None:
        """Whether the provider process behind one turn has exited.

        Three answers, and the third carries the most weight: True the process
        is gone, False it is still there, None nobody here can tell.  An
        adapter that owns a local process answers from it; one reached over a
        socket it does not own says None rather than guessing.  Callers that
        need evidence a turn stopped read anything but True as still running.
        """

        runtime_adapter = self._adapter_for(turn.agent_id)
        query = getattr(runtime_adapter, "turn_process_ended", None)
        if not callable(query):
            return None
        answer = query(getattr(turn, "runtime", None))
        return None if answer is None else bool(answer)

    def record_turn_tool(self, turn: ManagedTurn, tool: str) -> None:
        """Note that one tool was called inside one turn, in order.

        What a manager did after its last delegate and before it yielded is
        otherwise invisible: only lifecycle transitions are recorded, and a
        manager thinking and inspecting produces none of them. The names are
        fixed strings from the tool surface, so this carries no content.
        """

        with self._lock:
            index = self._turn_indexes.get((turn.agent_id, turn.provider, turn.runtime.turn_id))
            if index is None:
                return
            self._turns[index].setdefault("tools", []).append(str(tool))

    def concurrent_turn_intervals(self) -> list[dict[str, Any]]:
        """Every pair of turns by different agents whose intervals overlapped.

        A turn still running has no finish, and is treated as running to now,
        because a live overlap is exactly the thing worth catching.
        """

        with self._lock:
            turns = [dict(value) for value in self._turns]
        now = time.time()
        overlaps: list[dict[str, Any]] = []
        for index, first in enumerate(turns):
            for second in turns[index + 1:]:
                if first["agent_id"] == second["agent_id"]:
                    continue
                start = max(first["started_at"], second["started_at"])
                end = min(
                    first.get("finished_at") or now,
                    second.get("finished_at") or now,
                )
                if end <= start:
                    continue
                overlaps.append(
                    {
                        "agents": [first["agent_id"], second["agent_id"]],
                        "runtime_turn_ids": [
                            first["runtime_turn_id"],
                            second["runtime_turn_id"],
                        ],
                        "seconds": round(end - start, 6),
                    }
                )
        return overlaps

    def runtime_effect_summary(self, agent_id: str | None = None) -> dict[str, Any]:
        return self.runtime_effects.summary(agent_id)

    def await_agents(self, manager_id: str, watched_agent_ids: list[str]) -> str:
        return self.control.await_agents(manager_id, watched_agent_ids)

    def record_progress(
        self,
        agent_id: str,
        *,
        activity: str,
        progress: str,
        files_touched: list[str] | None = None,
        commands: list[str] | None = None,
        material: bool = True,
    ) -> None:
        self.control.record_progress(
            agent_id,
            activity=activity,
            progress=progress,
            files_touched=files_touched,
            commands=commands,
            material=material,
        )

    def complete_agent(
        self,
        agent_id: str,
        result: Mapping[str, Any],
        *,
        require_messages_read: bool = False,
        progress: Mapping[str, Any] | None = None,
    ) -> None:
        self.control.complete_agent(
            agent_id,
            dict(result),
            require_messages_read=require_messages_read,
            progress=progress,
        )

    def complete_branch(self, branch_id: str, result: Mapping[str, Any]) -> None:
        self.control.complete_branch(branch_id, dict(result))

    def complete_root(
        self, root_id: str, result: Mapping[str, Any], *,
        progress: Mapping[str, Any] | None = None,
    ) -> None:
        session = self.control.sessions[self.session_id]
        if root_id != session.root_agent_id:
            raise ManagedSessionError("session outcome must be declared by the Root Manager")
        self.control.complete_agent(root_id, dict(result), progress=progress)
        with self._lock:
            self._outcome = dict(result)

    def steer(
        self,
        *,
        sender_id: str,
        target_id: str,
        turn: ManagedTurn,
        text: str,
    ) -> dict[str, Any]:
        if turn.agent_id != target_id:
            raise ManagedSessionError("steering target does not match the active turn")
        self.control.message_agent(sender_id, target_id, text, kind="steer")
        result = self._adapter_for(turn.agent_id).steer(turn.runtime, text)
        self.mark_check("live_steering", True)
        return result

    def interrupt_and_block(
        self,
        *,
        requester_id: str,
        target_id: str,
        turn: ManagedTurn,
        reason: str,
        timeout: float = 90,
    ) -> dict[str, Any]:
        if turn.agent_id != target_id:
            raise ManagedSessionError("interruption target does not match the active turn")
        self.request_interrupt(
            requester_id=requester_id,
            target_id=target_id,
            turn=turn,
            reason=reason,
        )
        result = self.wait_turn(turn, timeout=timeout)
        status = str(result.get("status") or "unknown")
        if status != "interrupted":
            raise ManagedSessionError(f"runtime interruption ended as {status}")
        self.control.block_agent(target_id, reason)
        with self._lock:
            self._interruptions[-1]["runtime_status"] = status
        self.mark_check("interruption", True)
        return result

    def request_interrupt(
        self,
        *,
        requester_id: str,
        target_id: str,
        turn: ManagedTurn,
        reason: str,
    ) -> None:
        """Interrupt an active turn and record owned-tool cleanup without waiting."""

        if turn.agent_id != target_id:
            raise ManagedSessionError("interruption target does not match the active turn")
        self.control.message_agent(requester_id, target_id, reason, kind="interrupt")
        self._adapter_for(turn.agent_id).interrupt(turn.runtime)
        with self._lock:
            self._interruptions.append(
                {
                    "requester_id": requester_id,
                    "target_id": target_id,
                    "runtime_status": "requested",
                    "reason": reason,
                    "tool_cleanups": [],
                }
            )
        self.mark_check("interruption", True)

    def observe_interrupt_result(self, target_id: str, status: str) -> None:
        with self._lock:
            for value in reversed(self._interruptions):
                if value["target_id"] == target_id and value["runtime_status"] == "requested":
                    value["runtime_status"] = status
                    return

    def retry_agent(
        self,
        *,
        requester_id: str,
        agent_id: str,
        revised_task_contract: Mapping[str, Any],
    ) -> AgentRecord:
        agent = self.control.retry_agent(
            requester_id=requester_id,
            agent_id=agent_id,
            revised_task_contract=dict(revised_task_contract),
        )
        self.mark_check("clean_recovery", True)
        return agent

    def runtime_adapters(self) -> list[RuntimeAdapter]:
        """Every distinct runtime this session currently holds, default first."""

        with self._lock:
            bound = list(self._adapters.values())
            default = self.adapter
        ordered: list[RuntimeAdapter] = []
        seen: set[int] = set()
        for adapter in [default, *bound]:
            if id(adapter) in seen:
                continue
            seen.add(id(adapter))
            ordered.append(adapter)
        return ordered

    def reconnect(
        self,
        adapter_factory: Callable[[RuntimeAdapter], RuntimeAdapter],
        *,
        on_adapter_ready: Callable[[RuntimeAdapter], None] | None = None,
    ) -> None:
        """Replace every owned runtime and reattach all persistent agent threads.

        A tree can span more than one provider, so reconnect is per runtime
        rather than per session.  The factory is handed the adapter being
        replaced and returns its successor: the caller is the only party that
        knows how to build one of each kind, and keying on the adapter itself
        keeps this layer from having to name a provider.

        Every old runtime is closed before any replacement is built, so a tree
        is never half on the old processes and half on the new ones.  A single
        unclean shutdown fails the whole reconnect, and a failure part-way
        through closes every replacement already built rather than leaking it
        and leaves the agent bindings as it found them.

        Once the closes have run there is no way back: the old processes are
        gone.  A reconnect that then fails leaves the session unusable and says
        so, rather than leaving a caller holding adapters that look live and
        are not.

        ``on_adapter_ready`` is called on each successor as soon as it
        initializes, before any thread resumes on it.  A runtime that is live
        but has no approval handler yet either raises or silently declines, and
        neither is acceptable for the width of a resume loop.
        """

        if self._runtimes_discarded:
            raise ManagedSessionError("this session's runtimes were already discarded")
        with self._lock:
            bound_by_agent = dict(self._adapters)
        outgoing = self.runtime_adapters()
        # Close every runtime even if one of them raises, so a bad shutdown
        # cannot leak the runtimes behind it in the list.
        cleanups: list[Any] = []
        close_errors: list[BaseException] = []
        for adapter in outgoing:
            try:
                cleanups.append(adapter.close())
            except BaseException as exc:
                close_errors.append(exc)
        self._runtimes_discarded = True
        boundary_clean = bool(cleanups) and not close_errors
        for index, cleanup in enumerate(cleanups, start=1):
            self._record_cleanup(f"reconnect-boundary-{index}", cleanup)
            boundary_clean = boundary_clean and (
                cleanup.streams_drained and cleanup.handler_threads_drained
            )
        self.mark_check("reconnect_boundary_clean", boundary_clean)
        if close_errors:
            raise ManagedSessionError(
                "pre-reconnect runtime cleanup raised"
            ) from close_errors[0]
        for cleanup in cleanups:
            if cleanup.process.residual_count or cleanup.errors:
                raise ManagedSessionError("pre-reconnect runtime cleanup was not clean")
        replacements: dict[int, RuntimeAdapter] = {}
        resumed: list[str] = []
        try:
            for adapter in outgoing:
                successor = adapter_factory(adapter)
                replacements[id(adapter)] = successor
                successor.initialize()
                if on_adapter_ready is not None:
                    on_adapter_ready(successor)
            for agent_id in self._ordered_agent_ids():
                thread_id = self._threads.get(agent_id)
                if thread_id is None:
                    continue
                agent = self._agent(agent_id)
                outgoing_adapter = bound_by_agent.get(agent_id, outgoing[0])
                replacement = replacements[id(outgoing_adapter)]
                agent_workspace = self.ensure_workspace(agent_id)
                reviewer = str(
                    self._policies.get(agent_id, {}).get("reviewer")
                    or "auto_review"
                )
                previous_identity = self._identity_attestations.get(agent_id)
                resume_attested = getattr(replacement, "resume_attested_thread", None)
                if callable(resume_attested):
                    identity = previous_identity
                    provider_session = (
                        identity.get("provider_session")
                        if isinstance(identity, Mapping)
                        else None
                    )
                    if (
                        not isinstance(identity, Mapping)
                        or identity.get("runtime_thread") != thread_id
                        or identity.get("bound") is not True
                        or not isinstance(provider_session, str)
                        or not provider_session
                    ):
                        raise ManagedSessionError("runtime identity is not persistently attested")
                    resume = resume_attested(
                        runtime_thread=thread_id,
                        provider_session=provider_session,
                        model=agent.model_id,
                        effort=agent.effort,
                        tool_handler=self._handlers[agent_id],
                        approvals_reviewer=reviewer,
                        workspace=agent_workspace,
                    )
                    identity = self._identity_attestation(replacement, thread_id)
                    if (
                        identity is None
                        or identity.get("bound") is not True
                        or identity.get("provider_session") != provider_session
                    ):
                        raise ManagedSessionError(
                            "resumed runtime identity mismatches persistent attestation"
                        )
                    self._require_same_identity(previous_identity, identity, thread_id)
                else:
                    # This route reads the successor's own attestation, so it
                    # has to be compared with what the node was before.  Left
                    # unchecked, a node could come back bound to a different
                    # provider and a different session and the reconnect would
                    # report success -- and this is the route the Codex adapter
                    # takes, so it was every Codex node in a mixed tree.
                    identity = self._identity_attestation(replacement, thread_id)
                    if identity is None or identity.get("bound") is not True:
                        raise ManagedSessionError("runtime identity is not persistently attested")
                    self._require_same_identity(previous_identity, identity, thread_id)
                    resume = replacement.resume_thread(
                        thread_id=thread_id,
                        model=agent.model_id,
                        effort=agent.effort,
                        tool_handler=self._handlers[agent_id],
                        approvals_reviewer=reviewer,
                        workspace=agent_workspace,
                    )
                resume_policy = (
                    resume.get("policy")
                    if callable(resume_attested) and isinstance(resume.get("policy"), Mapping)
                    else resume
                )
                require_workspace_policy(resume_policy)
                self._policies[agent_id] = effective_thread_policy(resume_policy)
                self._identity_attestations[agent_id] = identity
                resumed.append(agent_id)
                refreshed = read_identity(replacement, thread_id)
                if refreshed:
                    agent.model_identity = refreshed
        except BaseException:
            for successor in replacements.values():
                try:
                    successor.close()
                except BaseException:
                    # One successor refusing to close must not leak the rest,
                    # and must not replace the reason the reconnect failed.
                    continue
            with self._lock:
                self._adapters.clear()
                self._adapters.update(bound_by_agent)
            raise
        self.adapter = replacements[id(outgoing[0])]
        with self._lock:
            for agent_id, previous in bound_by_agent.items():
                self._adapters[agent_id] = replacements[id(previous)]
        self._runtimes_discarded = False
        with self._lock:
            self._reconnects.append({"resumed_agent_ids": list(resumed), "count": len(resumed)})
        self.mark_check("persistent_reconnect", True)

    def mark_check(self, name: str, passed: bool) -> None:
        with self._lock:
            self._checks[name] = bool(passed)

    def close(self) -> None:
        if self._closed:
            return
        # A cancelled or failed Worker never reaches the completion path that
        # settles its checkout, so the sweep happens here too.  An empty one is
        # removed with its branch; one carrying work is kept and stays in the
        # receipt, because a run that ended badly is exactly when someone wants
        # to see what the child had managed to do.
        self.close_worktrees()
        runtimes: list[RuntimeAdapter] = []
        seen: set[int] = set()
        for runtime_adapter in [self.adapter, *self._adapters.values()]:
            if id(runtime_adapter) in seen:
                continue
            seen.add(id(runtime_adapter))
            runtimes.append(runtime_adapter)
        clean = True
        first_error: BaseException | None = None
        for index, runtime_adapter in enumerate(runtimes):
            phase = "final-runtime" if index == 0 else f"final-runtime-{index}"
            try:
                cleanup = runtime_adapter.close()
            except BaseException as exc:
                # Each runtime owns its own processes, so one refusing to
                # close must not leave the next one running.  The failure is
                # recorded in its place and raised once every owner was asked.
                first_error = first_error or exc
                reason = f"{type(exc).__name__}: {exc}"
                cleanup = RuntimeCleanup(
                    ProcessCleanup("close-failed", 0, None, (reason,)), False, False, (reason,)
                )
            self._record_cleanup(phase, cleanup)
            clean = clean and (
                cleanup.process.residual_count == 0
                and cleanup.streams_drained
                and cleanup.handler_threads_drained
                and not cleanup.errors
            )
        self.mark_check("zero_orphan_shutdown", clean)
        if first_error is not None:
            # Left open, so a later close asks the refusing runtime again.
            raise first_error
        self._closed = True

    def receipt(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        session = self.control.sessions[self.session_id]
        aliases = self._aliases()
        hierarchy = []
        permissions = []
        for agent_id in self._ordered_agent_ids():
            agent = session.agents[agent_id]
            policy = dict(self._policies.get(agent_id, {}))
            hierarchy.append(
                {
                    "agent": aliases[agent_id],
                    "parent": aliases.get(agent.parent_agent_id),
                    "role": agent.role.value,
                    "model_id": agent.model_id,
                    "status": agent.status.value,
                    "turn_count": agent.turn_count,
                    "objective_digest": _digest(agent.objective),
                    "thread_ref": self._thread_ref(self._threads.get(agent_id)),
                    "files_touched": list(agent.files_touched),
                    "commands": list(agent.commands),
                    # Which scope a child ran under is the difference between
                    # reading its recorded paths as the project and reading them
                    # as a checkout of its own.  A receipt that omits it cannot
                    # be interpreted, and worktree is the first scope that
                    # changes the reader's git state.
                    "workspace_scope": agent.workspace_scope,
                    "approvals": agent.approvals,
                    "workspace_path": self.relative_workspace_for(agent_id),
                }
            )
            permissions.append(
                {
                    "agent": aliases[agent_id],
                    "thread_ref": self._thread_ref(self._threads.get(agent_id)),
                    "effective": policy,
                }
            )
        turns = [
            {
                "agent": aliases[value["agent_id"]],
                "phase": value["phase"],
                "control_turn_ref": self._thread_ref(value["control_turn_id"]),
                "runtime_turn_ref": self._thread_ref(value["runtime_turn_id"]),
                "status": value["status"],
                # Durations, never wall-clock stamps: a receipt may not carry
                # anything that dates or locates the run.
                "duration_seconds": (
                    round(value["finished_at"] - value["started_at"], 6)
                    if value.get("finished_at")
                    else None
                ),
                "tools": list(value.get("tools") or []),
            }
            for value in self._turns
        ]
        concurrency = [
            {
                "agents": [aliases.get(item, item) for item in overlap["agents"]],
                "runtime_turn_refs": [
                    self._thread_ref(item) for item in overlap["runtime_turn_ids"]
                ],
                "seconds": overlap["seconds"],
            }
            for overlap in self.concurrent_turn_intervals()
        ]
        reconnects = [
            {
                "resumed_agents": [aliases[value] for value in item["resumed_agent_ids"]],
                "count": item["count"],
            }
            for item in self._reconnects
        ]
        tool_registrations = [
            {
                "agent": aliases[agent_id],
                "role": session.agents[agent_id].role.value,
                **dict(value),
            }
            for agent_id, value in self._tool_registrations.items()
        ]
        interruptions = [
            {
                "requester": aliases[item["requester_id"]],
                "target": aliases[item["target_id"]],
                "runtime_status": item["runtime_status"],
                "reason": item["reason"],
                "tool_cleanups": item["tool_cleanups"],
            }
            for item in self._interruptions
        ]
        checks = dict(self._checks)
        return {
            "schema_version": self.schema_version,
            "receipt_id": str(uuid.uuid4()),
            "created_at": time.time(),
            "status": status,
            "error": error,
            "session": {
                "session_id": self.session_id,
                "preset_id": session.preset_id,
                "objective_digest": _digest(session.objective),
                "workspace": ".",
                "confinement": "runtime-native-workspace",
                "native_sandbox_parity_claimed": False,
            },
            "hierarchy": hierarchy,
            "turns": turns,
            "permissions": permissions,
            "tool_registrations": tool_registrations,
            "native_runtime_effects": self.runtime_effects.receipt(aliases),
            "native_runtime_effect_projection": self.runtime_effects.summary(),
            "control_events": [self._event_receipt(value, aliases) for value in session.events],
            "interruptions": interruptions,
            "reconnects": reconnects,
            "cleanup": list(self._cleanups),
            "concurrent_turns": concurrency,
            "context_saved_characters": self.context_saved_characters,
            # Through the alias table, like every other section. The outcomes
            # are keyed by raw agent id so a re-settlement can replace the
            # right row, and a raw identifier may not appear in a receipt.
            "worktrees": [
                {
                    key: (aliases.get(value, value) if key == "agent_id" else value)
                    for key, value in outcome.items()
                }
                for outcome in self._worktree_outcomes
            ],
            "checks": checks,
            "outcome": dict(self._outcome),
        }

    def persist_receipt(
        self,
        path: str | Path,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        receipt = self.receipt(status=status, error=error)
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
            raise
        return receipt

    def content_safe_receipt(
        self,
        *,
        status: str,
        canary_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Project provider-neutral evidence without durable native identifiers.

        This is a separate publication boundary.  The compatibility ``receipt``
        methods above deliberately retain their existing schema and behavior.
        """

        if status not in {
            "blocked",
            "cancelled",
            "completed",
            "failed",
            "passed",
            "pending",
            "running",
        }:
            raise ValueError("content-safe receipt status is invalid")
        evidence = self._content_safe_canary_evidence(canary_evidence)
        session = self.control.sessions[self.session_id]
        aliases = self._aliases()
        provider_aliases: dict[str, str] = {}

        def provider_alias(value: Any) -> str | None:
            if not isinstance(value, str) or not value:
                return None
            if value not in provider_aliases:
                provider_aliases[value] = f"provider-{len(provider_aliases) + 1}"
            return provider_aliases[value]

        hierarchy: list[dict[str, Any]] = []
        permissions: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        for agent_id in self._ordered_agent_ids():
            agent = session.agents[agent_id]
            alias = aliases[agent_id]
            hierarchy.append(
                {
                    "agent": alias,
                    "parent": aliases.get(agent.parent_agent_id),
                    "role": agent.role.value,
                    "status": agent.status.value,
                    "turn_count": agent.turn_count,
                    "objective_digest": _digest(agent.objective),
                    "files_touched": self._content_safe_workspace_paths(
                        agent.files_touched
                    ),
                }
            )
            policy = self._policies.get(agent_id, {})
            permissions.append(
                {
                    "agent": alias,
                    "workspace_writes": self._safe_bool(
                        policy.get("workspace_writes")
                    ),
                    "network": self._safe_choice(
                        policy.get("network"),
                        {
                            "approval_gated",
                            "disabled",
                            "enabled",
                            "none",
                            "restricted",
                            "unrestricted",
                        },
                    ),
                    "approvals_requested": self._safe_bool(
                        policy.get("approvals_requested")
                    ),
                    "reviewer": self._safe_choice(
                        policy.get("reviewer"),
                        {"always", "auto_review", "never", "on_request", "user"},
                    ),
                    "environment_ready": self._safe_bool(
                        policy.get("environment_ready")
                    ),
                }
            )
            identity = self._identity_attestations.get(agent_id)
            if isinstance(identity, Mapping):
                provider_session = identity.get("provider_session")
                identities.append(
                    {
                        "agent": alias,
                        "provider": provider_alias(identity.get("provider")),
                        "bound": identity.get("bound") is True,
                        "binding_phase": self._safe_choice(
                            identity.get("binding_phase"), {"attested", "reserved"}
                        ),
                        "synthetic": False,
                        "runtime_thread_digest": (
                            _digest(identity["runtime_thread"])
                            if isinstance(identity.get("runtime_thread"), str)
                            and identity["runtime_thread"]
                            else None
                        ),
                        "provider_identity_digest": (
                            _digest(provider_session)
                            if isinstance(provider_session, str) and provider_session
                            else None
                        ),
                    }
                )

        effects: list[dict[str, Any]] = []
        for effect in self.runtime_effects.records():
            changes = []
            for value in effect.changes:
                paths = self._content_safe_workspace_paths([value.path])
                if paths:
                    changes.append({"path": paths[0], "kind": value.kind})
            safe_cwd = (
                self._content_safe_workspace_paths([effect.cwd])
                if effect.cwd is not None
                else []
            )
            effects.append(
                {
                    "sequence": effect.sequence,
                    "agent": aliases.get(effect.agent_id, "unknown"),
                    "provider": provider_alias(effect.provider),
                    "item_digest": effect.item_ref,
                    "effect": effect.effect,
                    "status": self._safe_runtime_status(effect.status),
                    "evidence_limited": effect.evidence_limited,
                    "not_reported_fields": list(effect.not_reported_fields),
                    "exit_code": effect.exit_code,
                    "duration_ms": effect.duration_ms,
                    "cwd": safe_cwd[0] if safe_cwd else None,
                    "action_types": list(effect.action_types),
                    "changes": changes,
                }
            )

        turns = [
            {
                "agent": aliases[value["agent_id"]],
                "phase_digest": _digest(value["phase"]),
                "status": self._safe_runtime_status(value["status"]),
                "control_turn_digest": _digest(value["control_turn_id"]),
                "runtime_turn_digest": _digest(value["runtime_turn_id"]),
            }
            for value in self._turns
        ]
        event_counts: dict[str, int] = {}
        safe_event_types = {
            "agent-created",
            "blocked",
            "branch-completed",
            "cancelled",
            "completed",
            "message-queued",
            "progress",
            "replaced",
            "retry-ready",
            "turn-finished",
            "turn-started",
            "wait-registered",
            "wait-skipped-pending-wake",
            "wake",
        }
        for event in session.events:
            if event.event_type in safe_event_types:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        cleanups = []
        for index, item in enumerate(self._cleanups, start=1):
            runtime = item["runtime"]
            process = runtime["process"]
            cleanups.append(
                {
                    "phase": f"cleanup-{index}",
                    "outcome": self._safe_choice(
                        process["outcome"], {"clean", "forced", "not_started"}
                    ),
                    "residual_count": process["residual_count"],
                    "root_exit_code": process["root_exit_code"],
                    "streams_drained": runtime["streams_drained"],
                    "handler_threads_drained": runtime["handler_threads_drained"],
                    "error_count": len(runtime["errors"]) + len(process["errors"]),
                }
            )
        safe_check_names = {
            "clean_recovery",
            "command_proof_route_python_exact_bytes",
            "command_provider_status_not_relied_upon",
            "disposable_home_used_false",
            "effective_workspace_permissions",
            "interruption",
            "live_steering",
            "manager_owned_scheduler",
            "native_runtime_effect_projection",
            "persistent_reconnect",
            "provider_identity_bound_on_first_turn",
            "provider_identity_not_synthetic",
            "python_exact_bytes_verified",
            "reconnect_boundary_clean",
            "zero_orphan_shutdown",
        }
        checks = {
            key: value
            for key, value in self._checks.items()
            if key in safe_check_names
        }
        effect_summary = self.runtime_effects.summary()
        safe_effect_summary = {
            "effect_count": effect_summary.get("effect_count", 0),
            "command_count": effect_summary.get("command_count", 0),
            "file_change_count": effect_summary.get("file_change_count", 0),
            "limited_count": effect_summary.get("limited_count", 0),
            "files": self._content_safe_workspace_paths(
                effect_summary.get("files", [])
                if isinstance(effect_summary.get("files"), list)
                else []
            ),
            "malformed_item_count": effect_summary.get("malformed_item_count", 0),
            "uncorrelated_item_count": effect_summary.get(
                "uncorrelated_item_count", 0
            ),
            # Both are counts of the journal's own bookkeeping, so they carry
            # no workspace content, and a reader needs them to know whether
            # the effects below are the whole run.
            "dropped_effect_count": effect_summary.get("dropped_effect_count", 0),
            "ambiguous_item_count": effect_summary.get("ambiguous_item_count", 0),
            "evidence_complete": bool(
                effect_summary.get("evidence_complete", True)
            ),
        }
        return {
            "schema_version": self.content_safe_schema_version,
            "created_at": time.time(),
            "status": status,
            "session": {
                "session_digest": _digest(self.session_id),
                "objective_digest": _digest(session.objective),
                "workspace": ".",
                "confinement": "runtime-native-workspace",
                "native_sandbox_parity_claimed": False,
            },
            "hierarchy": hierarchy,
            "turns": turns,
            "permissions": permissions,
            "provider_identities": identities,
            "native_runtime_effects": effects,
            "native_runtime_effect_projection": safe_effect_summary,
            "control_event_counts": event_counts,
            "cleanup": cleanups,
            "checks": checks,
            "canary_evidence": evidence,
        }

    def persist_content_safe_receipt(
        self,
        path: str | Path,
        *,
        status: str,
        canary_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist the content-safe provider-neutral projection."""

        receipt = self.content_safe_receipt(
            status=status,
            canary_evidence=canary_evidence,
        )
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
            raise
        return receipt

    @staticmethod
    def _safe_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _safe_choice(value: Any, allowed: set[str]) -> str:
        return value if isinstance(value, str) and value in allowed else "unknown"

    @staticmethod
    def _safe_runtime_status(value: Any) -> str:
        if value in {
            "cancelled",
            "completed",
            "declined",
            "failed",
            "interrupted",
            "pending",
            "running",
        }:
            return value
        return "unknown"

    @staticmethod
    def _content_safe_canary_evidence(
        value: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        expected = {
            "proof_route",
            "provider_status_relied_upon",
            "disposable_home_used",
        }
        if set(value) != expected:
            raise ValueError("content-safe canary evidence has unexpected fields")
        proof_route = value.get("proof_route")
        if proof_route not in {"python_exact_bytes", "structured_exit_zero"}:
            raise ValueError("content-safe canary proof route is invalid")
        if not isinstance(value.get("provider_status_relied_upon"), bool):
            raise ValueError("provider-status reliance must be boolean")
        if not isinstance(value.get("disposable_home_used"), bool):
            raise ValueError("disposable-home use must be boolean")
        return {
            "proof_route": proof_route,
            "provider_status_relied_upon": value["provider_status_relied_upon"],
            "disposable_home_used": value["disposable_home_used"],
        }

    def _content_safe_workspace_paths(self, values: list[str]) -> list[str]:
        safe: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value or value == "<outside-workspace>":
                continue
            candidate = Path(value)
            try:
                if not candidate.is_absolute():
                    candidate = self.workspace / candidate
                candidate = candidate.resolve().relative_to(self.workspace)
            except (OSError, ValueError):
                continue
            normalized = candidate.as_posix()
            if normalized not in {"", "."}:
                safe.append(normalized)
        return safe

    def _record_cleanup(self, phase: str, cleanup: RuntimeCleanup) -> None:
        runtime = asdict(cleanup)
        with self._lock:
            # ``app_server`` is the production manager-service compatibility
            # field. ``runtime`` is the provider-neutral V2 spelling. Emit both
            # atomically until every persisted-receipt consumer is migrated.
            self._cleanups.append(
                {"phase": phase, "runtime": runtime, "app_server": dict(runtime)}
            )

    def _agent(self, agent_id: str) -> AgentRecord:
        session = self.control.sessions[self.session_id]
        try:
            return session.agents[agent_id]
        except KeyError as exc:
            raise ManagedSessionError("agent is outside this managed session") from exc

    @staticmethod
    def _identity_attestation(
        adapter: RuntimeAdapter,
        runtime_thread: str,
    ) -> dict[str, Any]:
        """Read a provider-owned binding without defaulting or relabeling it."""

        read = getattr(adapter, "thread_identity_attestation", None)
        if not callable(read):
            raise ManagedSessionError("runtime does not provide identity attestation")
        value = read(runtime_thread)
        if not isinstance(value, Mapping):
            raise ManagedSessionError("runtime identity attestation is malformed")
        if value.get("runtime_thread") != runtime_thread:
            raise ManagedSessionError("runtime identity attestation mismatches thread handle")
        if value.get("synthetic") is True:
            raise ManagedSessionError("runtime identity attestation is synthetic")
        return {
            "runtime_thread": runtime_thread,
            "provider": value.get("provider"),
            "bound": value.get("bound") is True,
            "provider_session": value.get("provider_session"),
            "binding_phase": value.get("binding_phase"),
        }

    def _refresh_identity(self, agent_id: str, runtime_thread: str) -> dict[str, Any]:
        identity = self._identity_attestation(self._adapter_for(agent_id), runtime_thread)
        with self._lock:
            self._identity_attestations[agent_id] = identity
        return identity

    def _native_provider_scope(self, agent_id: str) -> str:
        """Choose the existing provider identity for native-turn correlation.

        Selected catalog data identifies a concrete provider before execution.
        Legacy callers whose cards omit it retain the provider attested when
        their thread was bound.  The adapter object itself is deliberately not
        another identity source: older compatible adapters never exposed a
        ``provider`` attribute, and a turn map must not add that requirement.
        """

        agent = self._agent(agent_id)
        card = self.control.registry.cards[agent.model_id]
        if isinstance(card.provider, str) and card.provider:
            return card.provider
        with self._lock:
            identity = self._identity_attestations.get(agent_id, {})
            provider = identity.get("provider")
        if isinstance(provider, str) and provider:
            return provider
        raise ManagedSessionError("native turn provider is not identified")

    @staticmethod
    def _native_child_observation(
        value: NativeChildObservation | Mapping[str, Any],
    ) -> NativeChildObservation:
        """Accept the neutral value or an adapter-neutral mapping at the seam."""

        if isinstance(value, NativeChildObservation):
            return value
        if not isinstance(value, Mapping):
            raise ManagedSessionError("native child observation is malformed")
        required = (
            "provider",
            "parent_agent_id",
            "parent_thread_id",
            "native_child_thread_id",
            "attested",
            "delivery_contract",
        )
        if any(name not in value for name in required):
            raise ManagedSessionError("native child observation is incomplete")
        try:
            return NativeChildObservation(
                provider=value["provider"],
                parent_agent_id=value["parent_agent_id"],
                parent_thread_id=value["parent_thread_id"],
                native_child_thread_id=value["native_child_thread_id"],
                attested=value["attested"],
                delivery_contract=value["delivery_contract"],
                native_child_id=value.get("native_child_id"),
                model_id=value.get("model_id"),
                role=value.get("role"),
                effort=value.get("effort"),
                objective=value.get("objective"),
                task_contract=value.get("task_contract"),
                source_cursor=value.get("source_cursor"),
                parent_native_turn_id=value.get("parent_native_turn_id"),
                capabilities=value.get("capabilities"),
                unsupported_reason=value.get("unsupported_reason"),
            )
        except TypeError as exc:
            raise ManagedSessionError("native child observation is malformed") from exc

    @staticmethod
    def _native_delivery_contract(observation: NativeChildObservation) -> dict[str, str]:
        """Retain only adapter-declared, inspectable delivery capability facts."""

        if not isinstance(observation.delivery_contract, Mapping):
            raise ManagedSessionError("native child delivery contract is malformed")
        contract: dict[str, str] = {}
        for key, value in observation.delivery_contract.items():
            if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
                raise ManagedSessionError("native child delivery contract is malformed")
            contract[key] = value
        if not contract:
            raise ManagedSessionError("native child delivery contract is empty")
        return contract

    def _native_child_registration_lock(self, key: tuple[str, str, str]) -> Any:
        with self._lock:
            return self._native_child_registration_locks.setdefault(key, threading.RLock())

    def _turn_registration_lock(self, agent_id: str) -> Any:
        with self._lock:
            return self._turn_registration_locks.setdefault(agent_id, threading.RLock())

    def _ordered_agent_ids(self) -> list[str]:
        session = self.control.sessions[self.session_id]
        ordered: list[str] = []
        pending = [session.root_agent_id]
        while pending:
            agent_id = pending.pop(0)
            ordered.append(agent_id)
            pending.extend(session.agents[agent_id].child_ids)
        return ordered

    def _aliases(self) -> dict[str, str]:
        session = self.control.sessions[self.session_id]
        counts: dict[str, int] = {}
        aliases: dict[str, str] = {}
        for agent_id in self._ordered_agent_ids():
            role = session.agents[agent_id].role.value
            counts[role] = counts.get(role, 0) + 1
            aliases[agent_id] = role if counts[role] == 1 else f"{role}-{counts[role]}"
        return aliases

    @staticmethod
    def _thread_ref(value: str | None) -> str | None:
        return _digest(value)[:16] if value else None

    @staticmethod
    def _event_receipt(event: ProtocolEvent, aliases: Mapping[str, str]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key in ("role", "model_id", "kind", "reason", "material", "direct_override"):
            if key in event.metadata:
                safe[key] = event.metadata[key]
        source = event.metadata.get("source_agent_id")
        if isinstance(source, str):
            safe["source"] = aliases.get(source, source if source == "user" else "unknown")
        return {
            "event_id": event.event_id,
            "agent": aliases[event.agent_id],
            "type": event.event_type,
            "metadata": safe,
            "created_at": event.created_at,
        }
