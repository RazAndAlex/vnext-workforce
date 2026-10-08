from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext.vnext_app_server import TurnHandle, VNextAppServerAdapter
from vnext.vnext_claude import ClaudeRuntimeError
# The two sides of one approval are tested together on purpose: the bridge
# names an effect and this reviewer decides on it, and F17 was the gap between
# them.
from vnext.vnext_claude_bridge import _approval_detail, _tool_effect
from vnext.vnext_managed_session import ManagedSessionError, VNextManagedSession
from vnext.vnext_mcp_server import VNextMcpService
from vnext.vnext_orchestration import (
    AgentRole,
    AgentStatus,
    EconomicPreset,
    EvidenceKind,
    ModelCard,
    ModelRegistry,
    OrchestrationControlPlane,
    ProtocolError,
    RegistryClaim,
)
from vnext.vnext_runtime_effects import ClaudeRuntimeEffectReader, RuntimeEffectJournal
from vnext.vnext_runtime_types import NativeChildObservation, ToolCallContext
from vnext.vnext_scheduler import (
    MESSAGE_BATCH,
    _BOUNDARY_DECLINE_REASONS,
    _BOUNDARY_RESOLVER,
    _MANAGER_DECLINE_REASON,
    PROVIDER_RETRY_DELAYS,
    PROVIDER_RETRY_LIMIT,
    SchedulerCancelled,
    SchedulerError,
    SchedulerHooks,
    VNextScheduler,
    _BlockReported,
    _TurnFinished,
    _UnsolicitedTurnEnded,
    _objective_line,
    _safe_detail,
)
from vnext.vnext_worker_tools import ApprovalDecision
from vnext.workforce_contracts import RunCancellation
from tests.test_vnext_external_primary import ScriptedWorker as ExternalScriptedWorker


MANAGER = "manager-model"
MESSAGES_HEADER = "MESSAGES:\n"
CLOCK_PREFIX = "[clock] "


def _clock_line_of(prompt: str) -> str:
    """Return the trailing clock line of any agent prompt, failing if it is gone."""

    line = prompt.rsplit("\n", 1)[-1]
    if not line.startswith(CLOCK_PREFIX):
        raise AssertionError(f"prompt carries no clock line: {prompt[-200:]!r}")
    return line


def _local_hhmm(moment: float) -> str:
    """The wall-clock half of a clock line, rendered the way the product does."""

    return datetime.datetime.fromtimestamp(moment).astimezone().strftime("%H:%M %Z")


def _messages_of(prompt: str) -> object:
    """Read the MESSAGES block of a prompt whose last line is the clock."""

    _clock_line_of(prompt)
    body = prompt.rsplit("\n", 1)[0]
    return json.loads(body.split(MESSAGES_HEADER, 1)[1])
CHILDREN_HEADER = "CHILDREN:" + chr(10)
WORKER = "worker-model"


def policy() -> dict:
    return {
        "posture": {
            "workspace_writes": True,
            "network": "restricted",
            "approvals_requested": True,
            "reviewer": "auto_review",
            "environment_ready": True,
        }
    }


def payload(result) -> dict:
    """Read a neutral tool result the way a provider adapter's model would."""

    value = json.loads(result.as_json_text())
    value["success"] = result.success
    return value


def _deliver_reported_blocks(scheduler) -> None:
    """Do what the scheduler loop does with the events report_blocked queues.

    Matched by name so this file still runs on a tree without the event,
    where a test that needs it fails on its assertions.
    """

    pending = []
    while not scheduler._events.empty():
        pending.append(scheduler._events.get_nowait())
    for event in pending:
        if type(event).__name__ == "_BlockReported":
            scheduler._handle_block_reported(event)
        else:
            scheduler._events.put(event)


class ScriptedAdapter:
    """In-memory runtime adapter that drives managers only through tools."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_thread = 0
        self._next_turn = 0
        self.handlers: dict[str, object] = {}
        self.tools: dict[str, list[dict]] = {}
        self.requested_reviewers: dict[str, str] = {}
        self.turn_reviewers: list[tuple[str, str]] = []
        self.roles: dict[str, str] = {}
        self.prompts: dict[str, list[str]] = {}
        self.branches: list[str] = []
        self.workers_by_branch: dict[str, list[str]] = {}
        self.invalid_delegation: dict | None = None
        self.invalid_evidence: dict | None = None
        self.early_completion: dict | None = None
        self.inspections = 0
        self.approval_turns = 0
        self.worker_command_once = False
        self.instructions: dict[str, str] = {}
        self.thread_efforts: dict[str, str] = {}
        self.delay_root_terminal_return = False
        self.root_terminal_entered = threading.Event()
        self.root_terminal_release = threading.Event()
        self.events: list[dict] = []
        self.steers: list[tuple[str, str]] = []
        self.steer_observed = threading.Event()
        self.interrupted: list[str] = []
        self.process_ended: bool | None = None
        self.native_approval_handler = None

    def start_thread(
        self,
        *,
        model,
        developer_instructions,
        tools,
        tool_handler,
        requested_posture,
        **_kwargs,
    ):
        del model
        with self._lock:
            self._next_thread += 1
            thread_id = f"thread-{self._next_thread}"
            role = developer_instructions.split("ROLE=", 1)[1].split()[0]
            self.handlers[thread_id] = tool_handler
            self.tools[thread_id] = list(tools)
            self.requested_reviewers[thread_id] = str(requested_posture.reviewer)
            self.roles[thread_id] = role
            self.prompts[thread_id] = []
            self.instructions[thread_id] = developer_instructions
            self.thread_efforts[thread_id] = str(_kwargs.get("effort") or "")
        result = policy()
        result["posture"]["reviewer"] = self.requested_reviewers[thread_id]
        return thread_id, result

    def turn_process_ended(self, handle=None):
        del handle
        return self.process_ended

    def tool_registration_attestation(self, thread_id):
        tools = self.tools[thread_id]
        return {
            "acknowledged": True,
            "model_id": "fixture-model",
            "tool_count": len(tools),
            "tool_names": [str(value.get("name") or "") for value in tools],
            "definition_sha256": "a" * 64,
            "handler_registered": self.handlers[thread_id] is not None,
        }

    def start_turn(self, *, thread_id, prompt, approvals_reviewer, **_kwargs):
        with self._lock:
            self._next_turn += 1
            turn_id = f"turn-{self._next_turn}"
            self.prompts[thread_id].append(prompt)
            self.turn_reviewers.append((thread_id, approvals_reviewer))
            cursor = len(self.events)
        return TurnHandle(thread_id=thread_id, turn_id=turn_id, cursor=cursor)

    def wait_turn(self, handle, *, timeout=300):
        del timeout
        handler = self.handlers[handle.thread_id]
        role = self.roles[handle.thread_id]
        prompt = self.prompts[handle.thread_id][-1]
        if "[APPROVAL]" in prompt:
            approval_id = prompt.split("approval_id=", 1)[1].split()[0]
            decision = payload(
                handler(
                    "resolve_approval",
                    {
                        "approval_id": approval_id,
                        "decision": "accept",
                        "rationale": "bounded test command",
                    },
                    None,
                )
            )
            self.assert_success(decision)
            with self._lock:
                self.approval_turns += 1
            return {"status": "completed", "final_response": "approval resolved"}
        if role == AgentRole.ROOT_MANAGER.value:
            self._run_root(handler)
            status = payload(handler("inspect", {"agent_id": "self", "deep": False}, None))
            if status["agent"]["status"] == AgentStatus.COMPLETED.value:
                self.root_terminal_entered.set()
                if self.delay_root_terminal_return:
                    self.root_terminal_release.wait(5)
            return {"status": "completed", "final_response": "root turn complete"}
        if role == AgentRole.BRANCH_MANAGER.value:
            self._run_branch(handle.thread_id, handler)
            return {"status": "completed", "final_response": "branch turn complete"}
        self._run_worker(handle)
        return {"status": "completed", "final_response": "worker evidence"}
    def _run_root(self, handler) -> None:
        with self._lock:
            branches = list(self.branches)
        if not branches:
            self.invalid_evidence = payload(
                handler(
                    "delegate",
                    {
                        "role": AgentRole.BRANCH_MANAGER.value,
                        "model_id": MANAGER,
                        "objective": "must not be created",
                        "task_contract": {"criteria": []},
                        "required_tool_effect": {
                            "tool": "run_command",
                            "arguments": {
                                "argv": ["python"],
                                "cwd": ".",
                                "timeout_seconds": 60,
                            },
                            "completion": {"exit_code": 0},
                        },
                    },
                    None,
                )
            )
            for index in range(2):
                result = payload(
                    handler(
                        "delegate",
                        {
                            "role": AgentRole.BRANCH_MANAGER.value,
                            "model_id": MANAGER,
                            "objective": f"Own branch {index + 1}",
                            "task_contract": {"criteria": [f"branch {index + 1} complete"]},
                        },
                        None,
                    )
                )
                self.assert_success(result)
                branches.append(result["agent_id"])
            with self._lock:
                self.branches = list(branches)
            self.early_completion = payload(
                handler(
                    "complete_session",
                    {"decision": "accepted", "summary": "too early", "criteria": {}},
                    None,
                )
            )
        active = [agent_id for agent_id in branches if self._status(handler, agent_id) not in {"completed", "cancelled", "failed", "replaced"}]
        if active:
            self.assert_success(payload(handler("await_children", {"agent_ids": active}, None)))
        else:
            self.assert_success(
                payload(
                    handler(
                        "complete_session",
                        {
                            "decision": "accepted",
                            "summary": "all manager-selected branches completed",
                            "criteria": {"dynamic_topology": True},
                        },
                        None,
                    )
                )
            )

    def _run_branch(self, thread_id: str, handler) -> None:
        branch_id = self._self_id(handler)
        with self._lock:
            workers = list(self.workers_by_branch.get(branch_id, []))
            first_branch = bool(self.branches) and branch_id == self.branches[0]
        if first_branch and not workers:
            self.invalid_delegation = payload(
                handler(
                    "delegate",
                    {
                        "role": AgentRole.BRANCH_MANAGER.value,
                        "model_id": MANAGER,
                        "objective": "illegal nested branch",
                        "task_contract": {"criteria": []},
                    },
                    None,
                )
            )
            for index in range(2):
                result = payload(
                    handler(
                        "delegate",
                        {
                            "role": AgentRole.WORKER.value,
                            "model_id": WORKER,
                            "objective": f"Perform worker task {index + 1}",
                            "task_contract": {"criteria": ["return evidence"]},
                            "approvals": "ask",
                            "required_tool_effect": {
                                "tool": "run_command",
                                "arguments": {
                                    "argv": ["python", "-m", "unittest", "-v"],
                                    "cwd": ".",
                                    "timeout_seconds": 60,
                                },
                                "completion": {"exit_code": 0},
                            },
                        },
                        None,
                    )
                )
                self.assert_success(result)
                workers.append(result["agent_id"])
            with self._lock:
                self.workers_by_branch[branch_id] = list(workers)
        active = [agent_id for agent_id in workers if self._status(handler, agent_id) not in {"completed", "cancelled", "failed", "replaced"}]
        if active:
            self.assert_success(payload(handler("await_children", {"agent_ids": active}, None)))
        else:
            self.assert_success(
                payload(
                    handler(
                        "complete_branch",
                        {
                            "outcome": "completed",
                            "verified": True,
                            "evidence": ["children reached terminal state"],
                        },
                        None,
                    )
                )
            )

    def _run_worker(self, handle: TurnHandle) -> None:
        with self._lock:
            should_run = not self.worker_command_once
            if should_run:
                self.worker_command_once = True
        if should_run:
            response = self.native_approval_handler(
                "approval/request",
                {
                    "approval_reference": "fixture:native-command-1",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": handle.thread_id,
                        "turn": handle.turn_id,
                        "request": "native-command-1",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )
            if response != {"decision": "accept"}:
                raise AssertionError(response)
            self.events.append(
                {
                    "name": "tool_result",
                    "reservation_id": handle.thread_id,
                    "turn_reference": handle.turn_id,
                    "tool_use_id": "native-command-1",
                    "effect_type": "command",
                    "status": "completed",
                    "provider_correlation": {
                        "session": handle.thread_id,
                        "turn": handle.turn_id,
                        "request": "native-command-1",
                    },
                    "correlation_attested": True,
                }
            )
        completed = payload(
            self.handlers[handle.thread_id](
                "complete_agent",
                {
                    "outcome": "completed bounded execution",
                    "verified": True,
                    "evidence": ["native effect recorded"],
                },
                None,
            )
        )
        self.assert_success(completed)

    def events_since(self, cursor=None):
        return tuple(self.events[cursor or 0:])

    def thread_identity_attestation(self, thread_id):
        return {
            "runtime_thread": thread_id,
            "provider": "fixture",
            "bound": True,
            "provider_session": thread_id,
            "binding_phase": "attested",
            "synthetic": False,
        }

    def _self_id(self, handler) -> str:
        result = payload(handler("inspect", {"agent_id": "self", "deep": False}, None))
        self.assert_success(result)
        with self._lock:
            self.inspections += 1
        return result["agent"]["agent_id"]

    def _status(self, handler, agent_id: str) -> str:
        result = payload(handler("inspect", {"agent_id": agent_id, "deep": True}, None))
        self.assert_success(result)
        with self._lock:
            self.inspections += 1
        return result["agent"]["status"]

    @staticmethod
    def assert_success(result: dict) -> None:
        if result.get("error"):
            raise AssertionError(result)

    def steer(self, handle, text):
        self.steers.append((handle.thread_id, text))
        self.steer_observed.set()
        return {"ok": True}

    def interrupt(self, handle):
        self.interrupted.append(handle.thread_id)
        return {"ok": True}


class OrdinaryReplyAdapter(ScriptedAdapter):
    """A native chat reply that does not claim orchestration completion."""

    def wait_turn(self, handle, *, timeout=300):
        del handle, timeout
        return {"status": "completed", "final_response": "ordinary native reply"}


class HeldTurnAdapter(ScriptedAdapter):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()
        self.release = threading.Event()

    def wait_turn(self, handle, *, timeout=300):
        del timeout
        self.waiting.set()
        self.release.wait(2)
        return {"status": "interrupted" if handle.thread_id in self.interrupted else "completed"}


class BlockingBindAdapter(ScriptedAdapter):
    """Leaves provider binding in flight to exercise lease/start interleaving."""

    def __init__(self):
        super().__init__()
        self.binding_started = threading.Event()
        self.binding_release = threading.Event()

    def start_thread(self, **kwargs):
        self.binding_started.set()
        self.binding_release.wait(2)
        return super().start_thread(**kwargs)


class DeferredReadinessAdapter(ScriptedAdapter):
    """Models a Claude reservation whose native-child reader drains late."""

    def __init__(self) -> None:
        super().__init__()
        self.readiness = [False, True]
        self.readiness_threads: list[str] = []

    def can_start_turn(self, thread_id: str) -> bool:
        self.readiness_threads.append(thread_id)
        return self.readiness.pop(0)


class RaisingReadinessAdapter(ScriptedAdapter):
    """A readiness probe that dies the way the live Claude bridge died.

    `can_start_turn` reaches the bridge with a two-second deadline of its own
    and raises when it expires.  Nothing caught that, so the exception left the
    scheduler loop and latched the whole session as failed.
    """

    def can_start_turn(self, thread_id: str) -> bool:
        raise RuntimeError("Claude bridge request timed out after 2.0s")


def registry() -> ModelRegistry:
    return ModelRegistry(
        cards=[
            ModelCard(
                MANAGER,
                frozenset(
                    {AgentRole.ROOT_MANAGER, AgentRole.BRANCH_MANAGER, AgentRole.WORKER}
                ),
            ),
            ModelCard(WORKER, frozenset({AgentRole.WORKER})),
        ],
        presets=[EconomicPreset("product-restricted", frozenset({MANAGER, WORKER}))],
    )


class VNextSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="Complete the dynamic scheduler fixture",
            task_contract={"criteria": ["dynamic topology accepted"]},
            session_id="scheduler-session",
        )
        self.adapter = ScriptedAdapter()
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return self.adapter

        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _started_worker_for_completion(self):
        scheduler = VNextScheduler(
            managed=self.managed, root=self.root,
            cancellation=RunCancellation(),
        )
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER,
            objective="Write one note",
            task_contract={"criteria": ["note exists"]},
        )
        self.assertTrue(scheduler._bind_agent(worker))
        turn = self.managed.start_turn(
            worker.agent_id, prompt="Write one note", effort="high",
            phase="worker-execution",
        )
        scheduler._active_turns[worker.agent_id] = turn
        return scheduler, worker, turn

    def test_completed_worker_reports_native_command_and_file_in_both_views(self) -> None:
        scheduler, worker, turn = self._started_worker_for_completion()
        note = self.workspace / "note.txt"
        note.write_text("hello\n", encoding="utf-8")
        def native_effect(item_id, effect_type, changes=None):
            return {
                "name": "tool_result", "reservation_id": turn.runtime.thread_id,
                "turn_reference": turn.runtime.turn_id, "tool_use_id": item_id,
                "effect_type": effect_type, "status": "completed",
                "correlation_attested": True,
                "provider_correlation": {
                    "session": "fixture-session", "turn": turn.runtime.turn_id,
                    "request": item_id,
                },
                **({"changes": changes} if changes is not None else {}),
            }

        self.adapter.events.extend([
            native_effect("command-1", "command"),
            native_effect("file-1", "file", [
                {"path": str(note), "kind": {"type": "not_reported"}},
            ]),
        ])
        completed = payload(scheduler._manager_handler(worker.agent_id, "complete_agent", {
            "outcome": "note written", "verified": True, "evidence": ["note.txt"],
        }, None))
        self.assertTrue(completed["success"], completed)
        inspected = payload(scheduler._manager_handler(self.root.agent_id, "inspect", {
            "agent_id": worker.agent_id, "deep": False,
        }, None))["agent"]
        compact = scheduler._compact_child(worker.agent_id)
        for view in (inspected, compact):
            self.assertEqual("completed", view["status"])
            self.assertEqual(["note.txt"], view["files_touched"])
            self.assertEqual(["recorded command"], view["commands"])

    def test_a_failed_turn_with_an_unhashable_reason_still_fails_the_worker(self) -> None:
        """R17: a JSON list in terminal_reason raised TypeError before the worker was blocked."""

        scheduler, worker, turn = self._started_worker_for_completion()
        scheduler._handle_turn_finished(
            _TurnFinished(worker.agent_id, turn, result={"status": "failed", "terminal_reason": ["x"],
                                                         "subtype": {"a": 1}, "code": None})
        )
        self.assertNotEqual(AgentStatus.RUNNING, worker.status)
        self.assertTrue(worker.blocker)

    def test_completed_worker_usage_is_provisional_until_runtime_turn_ends(self) -> None:
        scheduler, worker, turn = self._started_worker_for_completion()
        self.control.record_progress(
            worker.agent_id, activity="working", progress="tokens observed",
            usage={"tokens": {"totalTokens": 43698}}, material=False,
        )
        completed = payload(scheduler._manager_handler(worker.agent_id, "complete_agent", {
            "outcome": "done", "verified": True, "evidence": ["done"],
        }, None))
        self.assertTrue(completed["success"], completed)
        inspected = payload(scheduler._manager_handler(self.root.agent_id, "inspect", {
            "agent_id": worker.agent_id, "deep": False,
        }, None))["agent"]
        compact = scheduler._compact_child(worker.agent_id)
        for view in (inspected, compact):
            self.assertEqual(43698, view["usage"]["tokens"]["totalTokens"])
            self.assertIs(view["usage_final"], False)
        scheduler._handle_turn_finished(_TurnFinished(
            worker.agent_id, turn, result={"status": "completed"},
        ))
        inspected = payload(scheduler._manager_handler(self.root.agent_id, "inspect", {
            "agent_id": worker.agent_id, "deep": False,
        }, None))["agent"]
        self.assertIs(inspected["usage_final"], True)
        self.assertIs(scheduler._compact_child(worker.agent_id)["usage_final"], True)

    def test_never_bound_retry_and_replacement_are_recorded_before_reply(self) -> None:
        statuses: list[tuple[str, str]] = []
        lifecycle: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                record_agent=lambda agent: statuses.append((agent.agent_id, agent.status.value)),
                lifecycle=lambda kind, agent, data: lifecycle.append(
                    (kind, agent.agent_id, dict(data))
                ),
            ),
        )
        delegated = payload(scheduler._manager_handler(self.root.agent_id, "delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER,
            "objective": "Try the unavailable provider",
            "task_contract": {},
        }, None))
        self.assertTrue(delegated["success"], delegated)
        child_id = delegated["agent_id"]
        self.assertIn((child_id, "ready"), statuses)
        self.assertEqual(1, len([kind for kind, agent_id, _ in lifecycle
                                 if kind == "agent_delegated" and agent_id == child_id]))

        child = self.control.sessions[self.root.session_id].agents[child_id]
        with patch.object(self.adapter, "start_thread", side_effect=RuntimeError("offline")):
            self.assertFalse(scheduler._bind_agent(child))
        self.assertEqual("blocked", statuses[-1][1])

        retried = payload(scheduler._manager_handler(self.root.agent_id, "retry", {
            "agent_id": child_id, "task_contract": {},
        }, None))
        self.assertTrue(retried["success"], retried)
        self.assertEqual((child_id, "ready"), statuses[-1])

        replaced = payload(scheduler._manager_handler(self.root.agent_id, "replace", {
            "agent_id": child_id, "model_id": WORKER, "task_contract": {},
        }, None))
        self.assertTrue(replaced["success"], replaced)
        replacement_id = replaced["agent_id"]
        self.assertEqual((replacement_id, "ready"), statuses[-1])
        announced = [data for kind, agent_id, data in lifecycle
                     if kind == "agent_delegated" and agent_id == replacement_id]
        self.assertEqual(1, len(announced))
        self.assertEqual(self.root.agent_id, announced[0]["parent_agent"])
        self.assertEqual(WORKER, announced[0]["model_id"])
        self.assertEqual("high", announced[0]["effort"])

    def test_manager_selected_tree_wakes_and_terminates_through_scheduler_interface(self) -> None:
        lifecycle: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda event_type, agent, data: lifecycle.append(
                    (event_type, agent.agent_id, dict(data))
                )
            ),
        )

        result = scheduler.run()
        receipt = self.managed.receipt(status="passed")

        session = self.control.sessions[self.root.session_id]
        roles = [agent.role for agent in session.agents.values()]
        self.assertGreaterEqual(roles.count(AgentRole.BRANCH_MANAGER), 2)
        self.assertGreaterEqual(roles.count(AgentRole.WORKER), 2)
        self.assertTrue(all(agent.status is AgentStatus.COMPLETED for agent in session.agents.values()))
        self.assertEqual("accepted", result["decision"])
        self.assertTrue(result["criteria"]["dynamic_topology"])
        self.assertFalse(self.adapter.early_completion["success"])
        self.assertEqual("active-children", self.adapter.early_completion["error_code"])
        self.assertTrue(self.adapter.invalid_delegation["success"])
        self.assertTrue(self.adapter.invalid_evidence["success"])
        self.assertGreaterEqual(self.adapter.inspections, 4)
        worker_threads = [
            thread_id
            for thread_id, role in self.adapter.roles.items()
            if role == AgentRole.WORKER.value
        ]
        self.assertTrue(worker_threads)
        self.assertTrue(all(self.adapter.tools[value] for value in worker_threads))
        self.assertTrue(
            all(self.adapter.requested_reviewers[value] == "user" for value in worker_threads)
        )
        self.assertTrue(
            all(
                self.adapter.requested_reviewers[value] == "auto_review"
                for value, role in self.adapter.roles.items()
                if role != AgentRole.WORKER.value
            )
        )
        self.assertTrue(self.adapter.turn_reviewers)
        self.assertTrue(
            all(
                reviewer == self.adapter.requested_reviewers[thread_id]
                for thread_id, reviewer in self.adapter.turn_reviewers
            )
        )
        worker_registrations = [
            value
            for value in receipt["tool_registrations"]
            if value["role"] == AgentRole.WORKER.value
        ]
        self.assertEqual(len(worker_threads), len(worker_registrations))
        self.assertTrue(
            all(
                value["tool_count"] > 0
                and value["handler_registered"] is True
                for value in worker_registrations
            )
        )
        self.assertGreaterEqual(self.adapter.approval_turns, 1)
        requested = [event for event in lifecycle if event[0] == "approval_requested"]
        resolved = [event for event in lifecycle if event[0] == "approval_resolved"]
        self.assertEqual(1, len(requested))
        self.assertEqual(1, len(resolved))
        self.assertEqual(requested[0][2]["approval_id"], resolved[0][2]["approval_id"])
        self.assertEqual("accept", resolved[0][2]["decision"])
        native_approvals = scheduler.native_approval_records()
        self.assertEqual("accept", native_approvals[0]["decision"])
        self.assertNotIn("native-command-1", json.dumps(native_approvals))
        self.assertGreaterEqual(len(scheduler.evidence_requests()), 2)
        self.assertTrue(
            all(
                request["status"] == "recorded-unattributed"
                for request in scheduler.evidence_requests()
            )
        )
        receipt = self.managed.receipt(status="passed")
        self.assertEqual(1, receipt["native_runtime_effect_projection"]["command_count"])
        hierarchy = receipt["hierarchy"]
        self.assertGreaterEqual(len(hierarchy), 5)
        branches = [item for item in hierarchy if item["role"] == "branch-manager"]
        workers = [item for item in hierarchy if item["role"] == "worker"]
        self.assertIn("root-manager", {item["parent"] for item in branches})
        self.assertTrue(all(item["parent"] for item in workers))
        self.assertTrue({item["parent"] for item in workers} & {item["agent"] for item in branches})
        wake_reasons = [
            event.metadata.get("reason")
            for event in session.events
            if event.event_type == "wake"
        ]
        self.assertIn("approval-requested", wake_reasons)
        self.assertIn("child-completed", wake_reasons)
        manager_instructions = [
            self.adapter.instructions[thread_id]
            for thread_id, role in self.adapter.roles.items()
            if role == AgentRole.ROOT_MANAGER.value
        ]
        branch_instructions = [
            self.adapter.instructions[thread_id]
            for thread_id, role in self.adapter.roles.items()
            if role == AgentRole.BRANCH_MANAGER.value
        ]
        self.assertTrue(all(MANAGER in value for value in manager_instructions))
        self.assertTrue(all(WORKER in value for value in branch_instructions))

    def test_manager_tool_schema_has_repeatable_delegate_without_model_enum(self) -> None:
        tools = VNextScheduler.manager_tools()
        names = [tool["name"] for tool in tools]
        self.assertIn("delegate", names)
        self.assertIn("await_children", names)
        self.assertIn("inspect", names)
        self.assertNotIn("spawn_branch", names)
        self.assertNotIn("spawn_worker", names)
        delegate = next(tool for tool in tools if tool["name"] == "delegate")
        self.assertNotIn("enum", delegate["inputSchema"]["properties"]["model_id"])
        effort = delegate["inputSchema"]["properties"]["effort"]
        self.assertNotIn("enum", effort)
        self.assertIn("high, low, max, medium, xhigh", effort["description"])

    def test_replace_refuses_effort_inherited_by_incompatible_provider(self) -> None:
        self.control.registry.cards[WORKER] = ModelCard(
            WORKER, frozenset({AgentRole.WORKER}), provider="claude"
        )
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id, parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=MANAGER, effort="ultra",
            objective="work", task_contract={},
        )
        self.control.block_agent(child.agent_id, "needs another provider")
        scheduler = VNextScheduler(
            managed=self.managed, root=self.root, cancellation=RunCancellation()
        )
        session = self.control.sessions[self.root.session_id]
        before_agents = set(session.agents)
        result = payload(scheduler._manager_handler(
            self.root.agent_id, "replace",
            {"agent_id": child.agent_id, "model_id": WORKER,
             "task_contract": {"criteria": ["done"]}}, None,
        ))
        self.assertFalse(result["success"])
        self.assertEqual("invalid-request", result["error_code"])
        self.assertIn("high, low, max, medium, xhigh", result["error"])
        self.assertEqual(before_agents, set(session.agents))
        self.assertEqual(AgentStatus.BLOCKED, child.status)

    def test_delegate_returns_invalid_effort_as_a_tool_result_before_spawn(self) -> None:
        card = self.control.registry.cards[WORKER]
        self.control.registry.cards[WORKER] = ModelCard(
            card.model_id, card.eligible_roles, provider="claude"
        )
        scheduler = VNextScheduler(
            managed=self.managed, root=self.root, cancellation=RunCancellation()
        )
        session = self.control.sessions[self.root.session_id]
        before_agents = set(session.agents)
        before_events = len(session.events)
        result = payload(scheduler._manager_handler(
            self.root.agent_id,
            "delegate",
            {"role": "worker", "model_id": WORKER, "objective": "work", "effort": "ludicrous"},
            None,
        ))
        self.assertFalse(result["success"])
        self.assertEqual("invalid-request", result["error_code"])
        self.assertIn("high, low, max, medium, xhigh", result["error"])
        self.assertEqual(before_agents, set(session.agents))
        self.assertEqual(before_events, len(session.events))

    def test_delegate_schema_lists_approval_modes_and_refuses_an_unknown_one(self) -> None:
        delegate = next(
            tool for tool in VNextScheduler.manager_tools() if tool["name"] == "delegate"
        )
        approvals = delegate["inputSchema"]["properties"]["approvals"]
        self.assertEqual(["granted", "ask"], approvals["enum"])
        self.assertNotIn("approvals", delegate["inputSchema"]["required"])
        # The server starts a child with no task_contract, so the schema a
        # client validates against must not refuse that call.
        self.assertNotIn("task_contract", delegate["inputSchema"]["required"])
        self.assertIn("task_contract", delegate["inputSchema"]["properties"])

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        with self.assertRaises(ProtocolError) as refused:
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": "Request an invalid approval policy",
                    "task_contract": {"criteria": ["refused"]},
                    "approvals": "sometimes",
                },
            )
        self.assertEqual("invalid-approval-mode", refused.exception.code)

    def _delegate_for_preflight(self, objective, turn_timeout=1800, provider="claude"):
        self.control.registry.cards[WORKER] = ModelCard(
            WORKER, frozenset({AgentRole.WORKER}), provider=provider
        )
        scheduler = VNextScheduler(
            managed=self.managed, root=self.root,
            cancellation=RunCancellation(), turn_timeout=turn_timeout,
        )
        result = payload(scheduler._apply_manager_tool(self.root.agent_id, "delegate", {
            "role": AgentRole.WORKER.value, "model_id": WORKER, "objective": objective,
        }))
        self.assertTrue(result["success"], result)
        self.assertIn(result["agent_id"], self.control.sessions[self.root.session_id].agents)
        return result

    def test_delegate_warns_when_objective_exceeds_turn_cap(self) -> None:
        for budget in ("96 minutes", "2 hours", "90 min", "1.5h", "31 mins", "0.6 hr", "1h30", "1h30m", "45m"):
            with self.subTest(budget=budget):
                result = self._delegate_for_preflight(f"Spend {budget} investigating")
                self.assertEqual(1, len(result["warnings"]))
                self.assertIn("1800", result["warnings"][0])
                self.assertIn("wall-clock", result["warnings"][0])
        result = self._delegate_for_preflight("Spend 2 minutes investigating", turn_timeout=60)
        self.assertIn("60", result["warnings"][0])

    def test_delegate_time_warning_is_route_aware(self) -> None:
        # Only the Claude SDK routes cut a turn at a wall-clock deadline; the
        # app-server routes renew the budget on every turn event.
        for provider, expected in (("claude", 1), ("zai", 1), ("codex", 0), ("commandcode", 0)):
            with self.subTest(provider=provider):
                result = self._delegate_for_preflight("Spend 2 hours investigating", provider=provider)
                self.assertEqual(expected, len(result.get("warnings", [])))

    def test_delegate_does_not_read_counts_as_minutes(self) -> None:
        for objective in ("process 100M rows", "summarise a 50 M token corpus", "a 64 MB file", "5m of logs"):
            with self.subTest(objective=objective):
                self.assertNotIn("warnings", self._delegate_for_preflight(objective))

    def test_delegate_succeeds_when_preflight_warnings_raise(self) -> None:
        with patch.object(VNextScheduler, "_delegate_preflight_warnings", side_effect=RuntimeError("bug")):
            result = self._delegate_for_preflight("Spend 2 hours investigating")
        self.assertNotIn("warnings", result)

    def test_delegate_warns_when_objective_needs_browser(self) -> None:
        for request in ("browser", "Playwright", "Chrome", "chromium", "puppeteer", "chromedriver",
                        "browse the web", "screenshot of the page", "screenshot of the site"):
            with self.subTest(request=request):
                result = self._delegate_for_preflight(f"Use {request} to inspect the UI")
                self.assertEqual(1, len(result["warnings"]))
                self.assertIn("no browser", result["warnings"][0])
                self.assertIn("network restricted", result["warnings"][0])
                self.assertIn("user", result["warnings"][0])

    def test_delegate_omits_warnings_for_clean_objective(self) -> None:
        result = self._delegate_for_preflight("Spend 30 min reviewing a local Python file")
        self.assertNotIn("warnings", result)

    def test_delegate_defaults_to_standing_approval_and_journals_the_decision(self) -> None:
        lifecycle: list[tuple[str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            turn_timeout=0.1,
            hooks=SchedulerHooks(
                lifecycle=lambda event_type, _agent, data: lifecycle.append(
                    (event_type, dict(data))
                )
            ),
        )
        delegated = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": "Use the standing approval grant",
                    "task_contract": {"criteria": ["approval accepted"]},
                },
            )
        )
        worker = self.control.sessions[self.root.session_id].agents[delegated["agent_id"]]
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="standing-grant-worker-thread",
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle.clear()  # Examine only this approval after delegation.

        outcome = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "fixture:standing-grant",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "standing-grant-worker-thread",
                    "turn": "standing-grant-worker-turn",
                    "request": "standing-grant-item",
                },
                "correlation_attested": True,
                "effect": "execute",
            },
        )

        self.assertEqual({"decision": "accept"}, outcome)
        self.assertEqual({}, scheduler._pending_approvals)
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [event_type for event_type, _data in lifecycle],
        )
        self.assertEqual(
            {"approval_id", "worker_agent", "manager_agent", "tool", "effect"},
            set(lifecycle[0][1]),
        )
        self.assertEqual(
            {"approval_id", "decision", "resolver"},
            set(lifecycle[1][1]),
        )
        self.assertEqual(lifecycle[0][1]["approval_id"], lifecycle[1][1]["approval_id"])
        self.assertEqual("standing-grant", lifecycle[-1][1]["resolver"])

    def test_delegate_can_keep_per_request_approval_review(self) -> None:
        lifecycle: list[tuple[str, dict]] = []
        requested = threading.Event()

        def record(event_type, _agent, data):
            lifecycle.append((event_type, dict(data)))
            if event_type == "approval_requested":
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            turn_timeout=0.1,
            hooks=SchedulerHooks(lifecycle=record),
        )
        delegated = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": "Ask before each protected effect",
                    "task_contract": {"criteria": ["approval reviewed"]},
                    "approvals": "ask",
                },
            )
        )
        worker = self.control.sessions[self.root.session_id].agents[delegated["agent_id"]]
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="ask-worker-thread",
            start_result=policy(),
            tool_handler=None,
        )
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:ask",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "ask-worker-thread",
                        "turn": "ask-worker-turn",
                        "request": "ask-item",
                    },
                    "correlation_attested": True,
                    "effect": "modify",
                },
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        approval_id = lifecycle[-1][1]["approval_id"]
        self.assertIn(approval_id, scheduler._pending_approvals)
        scheduler._apply_manager_tool(
            self.root.agent_id,
            "resolve_approval",
            {
                "approval_id": approval_id,
                "decision": "accept",
                "rationale": "fixture accepted",
            },
        )
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual({"decision": "accept"}, outcome["value"])
        self.assertEqual({}, scheduler._pending_approvals)
        self.assertEqual(self.root.agent_id, lifecycle[-1][1]["resolver"])

        replacement = self.control.replace_agent(
            requester_id=self.root.agent_id,
            agent_id=worker.agent_id,
            model_id=WORKER,
            revised_task_contract={"criteria": ["approval reviewed"]},
        )
        self.assertEqual("ask", replacement.approvals)

    def test_user_can_resolve_a_paired_worker_approval_event(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own approval review",
            task_contract={"criteria": ["approval resolved"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Request one bounded command",
            task_contract={"criteria": ["request reviewed"]},
            approvals="ask",
        )
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="native-worker-thread",
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle: list[tuple[str, dict]] = []
        requested = threading.Event()

        def record(event_type, _agent, data):
            lifecycle.append((event_type, dict(data)))
            if event_type == "approval_requested":
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:native-worker-item",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "native-worker-thread",
                        "turn": "native-worker-turn",
                        "request": "native-worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        approval_id = lifecycle[0][1]["approval_id"]
        result = scheduler.resolve_approval(
            approval_id,
            "decline",
            "user declined the command",
        )
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual("user", result["resolver"])
        self.assertEqual(
            {
                "decision": ApprovalDecision.DECLINE.value,
                "reason": _MANAGER_DECLINE_REASON,
            },
            outcome["value"],
        )
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [event_type for event_type, _data in lifecycle],
        )
        self.assertEqual("user", lifecycle[-1][1]["resolver"])

    def test_cancelling_a_worker_declines_the_approval_it_is_waiting_on(self) -> None:
        """R24: a cancelled worker's parked approval could still be accepted."""

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own approval review",
            task_contract={"criteria": ["approval resolved"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Request one bounded write",
            task_contract={"criteria": ["request reviewed"]},
            approvals="ask",
        )
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="native-worker-thread",
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle: list[tuple[str, dict]] = []
        requested = threading.Event()

        def record(event_type, _agent, data):
            lifecycle.append((event_type, dict(data)))
            if event_type == "approval_requested":
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:native-worker-item",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "native-worker-thread",
                        "turn": "native-worker-turn",
                        "request": "native-worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "modify",
                },
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        approval_id = next(data["approval_id"] for kind, data in lifecycle if kind == "approval_requested")

        scheduler.cancel_agent(worker.agent_id)
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(ApprovalDecision.DECLINE.value, outcome["value"]["decision"])
        self.assertIn("cancelled", outcome["value"]["reason"])
        self.assertEqual({}, scheduler._pending_approvals)
        with self.assertRaises(ProtocolError):
            scheduler.resolve_approval(approval_id, "accept", "late accept")
        resolved = [data for kind, data in lifecycle if kind == "approval_resolved"]
        self.assertEqual(1, len(resolved))
        self.assertEqual(_BOUNDARY_RESOLVER, resolved[0]["resolver"])

    def test_native_approval_rejects_unattested_and_malformed_correlation(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own an approval boundary",
            task_contract={"criteria": ["approval declined"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Request one bounded command",
            task_contract={"criteria": ["approval declined"]},
        )
        reservation_id = "local-worker-reservation"
        native_session = "attested-native-session"
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id=reservation_id,
            start_result=policy(),
            tool_handler=None,
        )
        with self.managed._lock:
            self.managed._identity_attestations[worker.agent_id] = {
                "runtime_thread": reservation_id,
                "provider": "fixture",
                "bound": True,
                "provider_session": native_session,
                "binding_phase": "attested",
            }
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        envelope = {
            "approval_reference": "fixture:native-worker-item",
            "provider": "fixture",
            "provider_correlation": {
                "session": native_session,
                "turn": "native-worker-turn",
                "request": "native-worker-item",
            },
            "correlation_attested": True,
            "effect": "execute",
        }
        malformed = (
            {**envelope, "correlation_attested": False},
            {**envelope, "provider": " "},
            {
                **envelope,
                "provider_correlation": {
                    **envelope["provider_correlation"],
                    "session": " ",
                },
            },
            {
                **envelope,
                "provider_correlation": {
                    **envelope["provider_correlation"],
                    "request": "",
                },
            },
        )
        # A blank provider is rejected before routing is even attempted, so it
        # is the envelope that is named as malformed; the rest reach routing
        # and find no worker to route to.
        for candidate, named in zip(
            malformed, ("unrouted", "malformed", "unrouted", "unrouted")
        ):
            self.assertEqual(
                {
                    "decision": "decline",
                    "reason": _BOUNDARY_DECLINE_REASONS[named],
                },
                scheduler.review_approval("approval/request", candidate),
            )

        with self.managed._lock:
            self.managed._identity_attestations.pop(worker.agent_id)
        self.assertEqual(
            {
                "decision": "decline",
                "reason": _BOUNDARY_DECLINE_REASONS["unrouted"],
            },
            scheduler.review_approval("approval/request", envelope),
        )
        records = scheduler.native_approval_records()
        self.assertEqual(5, len(records))
        self.assertTrue(
            all(
                not record["session_correlated"]
                and not record["turn_correlated"]
                and not record["request_correlated"]
                for record in records
            )
        )

    def test_approval_routes_unbound_reservations_without_accepting_them_as_sessions(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own local approval review",
            task_contract={"criteria": ["approval resolved"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Request one local approval",
            task_contract={"criteria": ["request reviewed"]},
            approvals="ask",
        )
        reservation_id = "local-worker-reservation"
        native_session = "attested-native-session"
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id=reservation_id,
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle: list[tuple[str, dict]] = []
        unrouted: list[tuple[str, dict]] = []
        requested = threading.Event()

        def record(event_type, agent, data):
            # A decline routed to no worker is recorded on the session root.
            if agent.agent_id == self.root.agent_id:
                unrouted.append((event_type, dict(data)))
                return
            lifecycle.append((event_type, dict(data)))
            if event_type == "approval_requested":
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )
        unbound_envelope = {
            "approval_reference": "fixture:unbound-local",
            "provider": "fixture",
            "routing_handle": {
                "reservation_id": reservation_id,
                "turn_reference": "local-worker-turn",
            },
            "correlation_attested": False,
            "effect": "modify",
        }
        # The callback emitted this local route before the SDK disclosed its ID.
        # Binding can happen before the manager consumes the queued envelope.
        with self.managed._lock:
            self.managed._identity_attestations[worker.agent_id] = {
                "runtime_thread": reservation_id,
                "provider": "fixture",
                "bound": True,
                "provider_session": native_session,
                "binding_phase": "attested",
            }
        self.assertEqual(
            {
                "decision": "decline",
                "reason": _BOUNDARY_DECLINE_REASONS["unrouted"],
            },
            scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:bound-mismatch",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": reservation_id,
                        "turn": "native-worker-turn",
                        "request": "native-worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            ),
        )
        self.assertFalse(requested.is_set())
        self.assertEqual(
            [("approval_requested", None), ("approval_resolved", "vnext-approval-boundary")],
            [(event_type, data.get("resolver")) for event_type, data in unrouted],
        )
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                unbound_envelope,
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        approval_id = lifecycle[-1][1]["approval_id"]
        scheduler.resolve_approval(approval_id, "decline", "user declined local effect")
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(
            {"decision": "decline", "reason": _MANAGER_DECLINE_REASON},
            outcome["value"],
        )
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [event_type for event_type, _data in lifecycle],
        )
        self.assertTrue(
            all(
                not record["session_correlated"]
                and not record["turn_correlated"]
                and not record["request_correlated"]
                for record in scheduler.native_approval_records()
            )
        )

    def test_active_branch_turn_receives_native_approval_by_live_steer(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own live approval review",
            task_contract={"criteria": ["approval resolved"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Request one native file change",
            task_contract={"criteria": ["request reviewed"]},
            approvals="ask",
        )
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(branch)
        scheduler._bind_agent(worker)
        branch_turn = self.managed.start_turn(
            branch.agent_id,
            prompt="active branch work",
            effort="high",
            phase="branch-initial",
        )
        scheduler._active_turns[branch.agent_id] = branch_turn
        worker_thread = self.managed._threads[worker.agent_id]
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:native-worker-item",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": worker_thread,
                        "turn": "native-worker-turn",
                        "request": "native-worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "modify",
                },
            )

        review_thread = threading.Thread(target=review)
        review_thread.start()
        self.assertTrue(self.adapter.steer_observed.wait(2))
        steered_thread, prompt = self.adapter.steers[-1]
        self.assertEqual(branch_turn.runtime.thread_id, steered_thread)
        approval_id = prompt.split("approval_id=", 1)[1].split()[0]
        resolved = payload(
            scheduler._manager_handler(
                branch.agent_id,
                "resolve_approval",
                {
                    "approval_id": approval_id,
                    "decision": "accept",
                    "rationale": "bounded native file change",
                },
                None,
            )
        )
        review_thread.join(2)
        scheduler._active_turns.pop(branch.agent_id, None)

        self.assertTrue(resolved["success"])
        self.assertFalse(review_thread.is_alive())
        self.assertEqual({"decision": "accept"}, outcome["value"])

    def test_pending_reconnect_pauses_ordinary_ready_agent_starts(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._reconnect_requested.set()

        scheduler._start_ready_agents()

        self.assertEqual({}, self.adapter.roles)

    def test_reconnect_interrupts_only_a_logically_yielded_manager_turn(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(adapter_factory=lambda outgoing: ScriptedAdapter()),
        )
        scheduler._bind_agent(self.root)
        root_turn = self.managed.start_turn(
            self.root.agent_id,
            prompt="delegate then yield",
            effort="high",
            phase="root-initial",
        )
        branch = self.managed.spawn_from_manager(
            requester_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            arguments={
                "objective": "Own the branch",
                "model_id": MANAGER,
                "task_contract": {"criteria": ["worker verified"]},
            },
        )
        self.managed.await_agents(self.root.agent_id, [branch.agent_id])
        scheduler._active_turns[self.root.agent_id] = root_turn
        scheduler.request_reconnect()

        scheduler._quiesce_yielded_managers_for_reconnect()
        scheduler._handle_turn_finished(
            _TurnFinished(
                self.root.agent_id,
                root_turn,
                result={"status": "interrupted"},
            )
        )

        self.assertEqual([root_turn.runtime.thread_id], self.adapter.interrupted)
        self.assertEqual(AgentStatus.AWAITING_WORKERS, self.root.status)
        self.assertNotEqual(AgentStatus.BLOCKED, self.root.status)

    def _scheduler(self) -> VNextScheduler:
        return VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

    def test_closed_manager_tool_schemas_refuse_unknown_top_level_arguments(self) -> None:
        scheduler = self._scheduler()
        schemas = {tool["name"]: tool["inputSchema"] for tool in scheduler.manager_tools()}
        for tool, schema in schemas.items():
            if schema.get("additionalProperties") is not False:
                continue
            accepted = set(schema["properties"])
            unknown = "workspace_scope" if tool == "delegate" else "totally_made_up"
            answer = scheduler._manager_handler(
                self.root.agent_id, tool, {unknown: "worktree"}, None
            )
            with self.subTest(tool=tool):
                self.assertFalse(answer.success, answer.value)
                self.assertEqual("invalid-request", answer.value["error_code"])
                self.assertIn(tool, answer.value["error"])
                self.assertIn(unknown, answer.value["error"])
                self.assertTrue(all(key in answer.value["error"] for key in accepted))
        self.assertEqual([self.root.agent_id], list(self.control.sessions[self.root.session_id].agents))

    def test_cancel_agent_tool_refuses_root_without_touching_children(self) -> None:
        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id, parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="still live", task_contract={},
        )
        answer = scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": self.root.agent_id}, None
        )
        self.assertFalse(answer.success, answer.value)
        self.assertEqual("invalid-request", answer.value["error_code"])
        self.assertIn("root cannot cancel itself", answer.value["error"])
        self.assertIn("complete_session", answer.value["error"])
        self.assertEqual(AgentStatus.READY, self.root.status)
        self.assertEqual(AgentStatus.READY, child.status)

    def test_an_agent_cannot_steer_or_message_itself(self) -> None:
        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id, parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="talks to itself", task_contract={},
        )
        for caller in (self.root, child):
            for tool in ("steer", "send_message"):
                answer = scheduler._manager_handler(
                    caller.agent_id, tool, {"agent_id": caller.agent_id, "message": "note to self"}, None
                )
                self.assertFalse(answer.success, (caller.role, tool, answer.value))
                self.assertEqual("invalid-request", answer.value["error_code"])
                self.assertIn("itself", answer.value["error"])
                self.assertEqual(0, scheduler._unread_message_count(caller))
        done = scheduler._manager_handler(child.agent_id, "complete_agent", {
            "outcome": "done", "verified": True, "evidence": ["x"],
        }, None)
        self.assertTrue(done.success, done.value)

    def test_await_children_names_a_repeated_child_once(self) -> None:
        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id, parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="waited on twice", task_contract={},
        )
        self.assertTrue(scheduler._bind_agent(child))
        answer = scheduler._manager_handler(
            self.root.agent_id, "await_children", {"agent_ids": [child.agent_id, child.agent_id]}, None
        )
        self.assertTrue(answer.success, answer.value)
        self.assertEqual([child.agent_id], answer.value["awaiting"])

    def test_delegate_contract_keeps_its_open_nested_properties(self) -> None:
        scheduler = self._scheduler()
        answer = scheduler._manager_handler(self.root.agent_id, "delegate", {
            "role": "worker", "model_id": WORKER, "objective": "check nested data",
            "task_contract": {"criteria": ["done"], "owner_note": "retained"},
        }, None)
        self.assertTrue(answer.success, answer.value)
        child = self.control.sessions[self.root.session_id].agents[answer.value["agent_id"]]
        self.assertEqual("retained", child.task_contract["owner_note"])

    def test_non_root_agent_can_still_cancel_itself(self) -> None:
        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id, parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="self cancel", task_contract={},
        )
        answer = scheduler._manager_handler(
            child.agent_id, "cancel_agent", {"agent_id": child.agent_id}, None
        )
        self.assertTrue(answer.success, answer.value)
        self.assertEqual(AgentStatus.CANCELLED, child.status)
        self.assertEqual(AgentStatus.READY, self.root.status)

    def test_completion_tools_require_declared_arguments(self) -> None:
        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id, parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="check", task_contract={},
        )
        for tool in ("complete_agent", "complete_branch"):
            for missing in ("outcome", "verified", "evidence"):
                args = {"outcome": "done", "verified": True, "evidence": []}
                del args[missing]
                answer = scheduler._manager_handler(child.agent_id, tool, args, None)
                self.assertFalse(answer.success, (tool, missing, answer.value))
                self.assertEqual("invalid-request", answer.value["error_code"])
                self.assertIn(missing, answer.value["error"])
        for missing in ("decision", "summary", "criteria"):
            args = {"decision": "accepted", "summary": "done", "criteria": {}}
            del args[missing]
            answer = scheduler._manager_handler(self.root.agent_id, "complete_session", args, None)
            self.assertFalse(answer.success, (missing, answer.value))
            self.assertEqual("invalid-request", answer.value["error_code"])
            self.assertIn(missing, answer.value["error"])
        answer = scheduler._manager_handler(self.root.agent_id, "complete_session", {
            "decision": "unknown", "summary": "done", "criteria": {},
        }, None)
        self.assertFalse(answer.success)
        self.assertEqual("invalid-request", answer.value["error_code"])
        self.assertIn("needs-review", answer.value["error"])
        for tool in ("complete_agent", "complete_branch"):
            answer = scheduler._manager_handler(child.agent_id, tool, {
                "outcome": "done", "verified": "wrong", "evidence": [],
            }, None)
            self.assertFalse(answer.success)
            self.assertIn("verified takes true or false", answer.value["error"])
        for criteria, said in (
            (5, "criteria takes an object"),
            (["tests"], "criteria takes an object"),
            ({"tests pass": "maybe"}, "tests pass takes true or false"),
            ({"tests pass": None}, "tests pass takes true or false"),
        ):
            answer = scheduler._manager_handler(self.root.agent_id, "complete_session", {
                "decision": "accepted", "summary": "done", "criteria": criteria,
            }, None)
            self.assertFalse(answer.success, (criteria, answer.value))
            self.assertEqual("invalid-request", answer.value["error_code"])
            self.assertIn(said, answer.value["error"])
            self.assertNotEqual(AgentStatus.COMPLETED, self.root.status)

    def test_required_agent_id_is_named_before_lookup(self) -> None:
        scheduler = self._scheduler()
        for tool, extra in (
            ("steer", {"message": "hello"}),
            ("send_message", {"message": "hello"}),
            ("interrupt_agent", {}), ("cancel_agent", {}),
            ("retry", {"task_contract": {}}),
            ("replace", {"model_id": WORKER, "task_contract": {}}),
        ):
            for missing in ({}, {"agent_id": ""}):
                answer = scheduler._manager_handler(self.root.agent_id, tool, {**extra, **missing}, None)
                self.assertFalse(answer.success, (tool, answer.value))
                self.assertEqual("invalid-request", answer.value["error_code"])
                self.assertIn("agent_id is required", answer.value["error"])

    def test_primary_must_complete_session(self) -> None:
        scheduler = self._scheduler()
        for tool in ("complete_agent", "complete_branch"):
            answer = scheduler._manager_handler(self.root.agent_id, tool, {
                "outcome": "done", "verified": True, "evidence": [],
            }, None)
            self.assertFalse(answer.success)
            self.assertIn("complete_session", answer.value["error"])
            self.assertNotEqual(AgentStatus.COMPLETED, self.root.status)

    def test_delegate_effort_schema_names_high_default(self) -> None:
        scheduler = self._scheduler()
        delegate = next(tool for tool in scheduler.manager_tools() if tool["name"] == "delegate")
        self.assertIn("high", delegate["inputSchema"]["properties"]["effort"]["description"])

    def test_every_agent_prompt_ends_with_a_pinned_clock_line(self) -> None:
        # 2026-09-24 21:14:00 CEST, with the agent born 2h 14m earlier.
        moment = 1790277240.0
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            now=lambda: moment,
        )
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]
        record.created_at = moment - (2 * 3600 + 14 * 60)

        opening, phase = scheduler._manager_prompt(record)

        self.assertEqual("root-initial", phase)
        expected = (
            f"[clock] {_local_hhmm(moment)} · turn 0s / 1800s · agent running 2h 14m"
        )
        self.assertEqual(expected, _clock_line_of(opening))

        scheduler._bind_agent(self.root)
        turn = self.managed.start_turn(
            self.root.agent_id, prompt="first root turn", effort="high", phase="root-initial"
        )
        scheduler._active_turns[self.root.agent_id] = turn
        scheduler._active_turns.pop(self.root.agent_id, None)
        self.control.finish_turn(self.root.agent_id)
        resumed, resume_phase = scheduler._manager_prompt(record)

        self.assertEqual("root-manager-resume", resume_phase)
        self.assertEqual(expected, _clock_line_of(resumed))

    def test_wall_clock_step_does_not_change_turn_elapsed(self) -> None:
        wall = [1_790_277_360.0]
        monotonic = [100.0]
        scheduler = VNextScheduler(
            managed=self.managed, root=self.root, cancellation=RunCancellation(),
            now=lambda: wall[0], monotonic=lambda: monotonic[0],
        )
        scheduler._turn_started_at[self.root.agent_id] = monotonic[0]
        wall[0] += 3600.0
        monotonic[0] += 10.0
        line = scheduler._steer_clock_line(self.root.agent_id)
        self.assertIn("turn 10s / 1800s", line)
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]
        self.assertIn("turn 10s / 1800s",
                      scheduler._clock_line(record, turn_started=100.0))

    def test_a_mid_turn_steer_carries_the_spent_turn_budget(self) -> None:
        moment = 1790277360.0
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            now=lambda: moment,
            monotonic=lambda: moment,
        )
        scheduler._turn_started_at[self.root.agent_id] = moment - 123.0

        line = scheduler._steer_clock_line(self.root.agent_id)

        self.assertEqual(f"[clock] {_local_hhmm(moment)} · turn 2m 03s / 1800s", line)

    def test_a_steer_with_no_recorded_turn_start_drops_the_elapsed_half(self) -> None:
        moment = 1790277360.0
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            now=lambda: moment,
        )

        self.assertEqual(
            f"[clock] {_local_hhmm(moment)}",
            scheduler._steer_clock_line(self.root.agent_id),
        )

    def test_terminal_release_waits_for_native_control_lease_exit(self) -> None:
        scheduler = self._scheduler()
        scheduler._bind_agent(self.root)
        self.root.thread_id = self.managed._threads[self.root.agent_id]
        self.root.status = AgentStatus.COMPLETED
        released: list[tuple[str, str]] = []
        self.adapter.release_terminal_thread = lambda thread_id, *, status: released.append((thread_id, status))
        scheduler._native_control_leases.add(self.root.agent_id)
        scheduler.queue_terminal_release(self.root.agent_id)
        scheduler._release_terminal_agents()
        self.assertEqual([], released)

        scheduler.release_native_control_lease(self.root.agent_id)
        scheduler._release_terminal_agents()
        self.assertEqual([(self.root.thread_id, "completed")], released)

    def test_blocked_release_waits_for_native_control_lease_exit(self) -> None:
        """Blocked is settled and not terminal. A lease exit that requeued only
        terminal statuses dropped the blocked release, leaving the client
        connected for as long as nobody cancelled the agent."""

        scheduler = self._scheduler()
        scheduler._bind_agent(self.root)
        self.root.thread_id = self.managed._threads[self.root.agent_id]
        released: list[tuple[str, str]] = []
        self.adapter.release_terminal_thread = lambda thread_id, *, status: released.append((thread_id, status))
        scheduler._native_control_leases.add(self.root.agent_id)

        scheduler._block(self.root, "the provider ended the turn")
        scheduler._release_terminal_agents()
        self.assertIs(AgentStatus.BLOCKED, self.root.status)
        self.assertEqual([], released)

        scheduler.release_native_control_lease(self.root.agent_id)
        scheduler._release_terminal_agents()
        self.assertEqual([(self.root.thread_id, "blocked")], released)

    def test_blocking_an_agent_releases_its_runtime_thread(self) -> None:
        """A blocked agent has no turn and cannot start one until somebody
        decides something, which can be never. Holding its provider process
        open for all of that is what kept blocked Claude workers alive until
        they were cancelled."""

        scheduler = self._scheduler()
        scheduler._bind_agent(self.root)
        self.root.thread_id = self.managed._threads[self.root.agent_id]
        released: list[tuple[str, str]] = []
        self.adapter.release_terminal_thread = lambda thread_id, *, status: released.append((thread_id, status))

        scheduler._block(self.root, "the provider ended the turn")
        scheduler._release_terminal_agents()

        self.assertIs(AgentStatus.BLOCKED, self.root.status)
        self.assertEqual([(self.root.thread_id, "blocked")], released)

    def _open_root_turn(self, scheduler: VNextScheduler):
        scheduler._bind_agent(self.root)
        turn = self.managed.start_turn(
            self.root.agent_id,
            prompt="root work in flight",
            effort="high",
            phase="root-initial",
        )
        scheduler._active_turns[self.root.agent_id] = turn
        return turn

    def test_conversational_steer_survives_a_runtime_that_refuses_in_place_steering(self) -> None:
        scheduler = self._scheduler()
        turn = self._open_root_turn(scheduler)

        def refuse(handle, text):
            raise NotImplementedError("this runtime does not steer a live turn")

        self.adapter.steer = refuse

        result = scheduler.steer("stop using Luna, switch to Flash")

        scheduler._active_turns.pop(self.root.agent_id, None)
        session = self.control.sessions[self.root.session_id]
        self.assertEqual("steered", result["status"])
        self.assertEqual("queued-for-next-turn", result["delivery"])
        self.assertEqual(AgentRole.ROOT_MANAGER.value, result["target"])
        queued = session.agents[self.root.agent_id].messages
        self.assertEqual(1, len(queued))
        self.assertEqual("stop using Luna, switch to Flash", queued[-1].text)
        self.assertEqual("user", queued[-1].sender_id)
        self.assertEqual(
            ["user-message"],
            [wake["reason"] for wake in session.wakes[self.root.agent_id]],
        )
        self.assertIsNotNone(turn)

    def test_conversational_steer_reports_delivery_into_an_active_turn(self) -> None:
        scheduler = self._scheduler()
        turn = self._open_root_turn(scheduler)

        result = scheduler.steer("prefer the cheaper branch model")

        scheduler._active_turns.pop(self.root.agent_id, None)
        self.assertEqual("delivered-into-active-turn", result["delivery"])
        self.assertEqual(
            [(turn.runtime.thread_id, "prefer the cheaper branch model")],
            self.adapter.steers,
        )
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(
            ["prefer the cheaper branch model"],
            [item.text for item in session.agents[self.root.agent_id].messages],
        )

    def test_conversational_steer_with_no_active_turn_queues_for_the_next_turn(self) -> None:
        scheduler = self._scheduler()

        result = scheduler.steer("add a verification branch")

        session = self.control.sessions[self.root.session_id]
        self.assertEqual("queued-for-next-turn", result["delivery"])
        self.assertEqual([], self.adapter.steers)
        self.assertEqual(
            ["add a verification branch"],
            [item.text for item in session.agents[self.root.agent_id].messages],
        )
        self.assertIn(self.root.agent_id, session.wakes)

    def test_steered_message_reaches_the_next_root_prompt_exactly_once(self) -> None:
        scheduler = self._scheduler()
        scheduler._bind_agent(self.root)
        turn = self.managed.start_turn(
            self.root.agent_id,
            prompt="first root turn",
            effort="high",
            phase="root-initial",
        )
        scheduler._active_turns[self.root.agent_id] = turn
        scheduler.steer("switch the second branch to Flash")
        scheduler._active_turns.pop(self.root.agent_id, None)
        self.control.finish_turn(self.root.agent_id)
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]
        self.assertNotEqual(0, record.turn_count)

        prompt, phase = scheduler._manager_prompt(record)
        replay, _phase = scheduler._manager_prompt(record)

        self.assertEqual("root-manager-resume", phase)
        messages = _messages_of(prompt)
        self.assertEqual(
            [("user", "switch the second branch to Flash")],
            [(item["sender_id"], item["text"]) for item in messages],
        )
        self.assertEqual([], _messages_of(replay))

    def _resumable_root_with_one_worker(self):
        """A bound Root that has taken a turn and owns one completed child."""

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(self.root)
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own one branch",
            task_contract={"criteria": ["branch done"]},
        )
        turn = self.managed.start_turn(
            self.root.agent_id,
            prompt="first root turn",
            effort="high",
            phase="root-initial",
        )
        scheduler._active_turns.pop(self.root.agent_id, None)
        del turn
        self.control.finish_turn(self.root.agent_id)
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]
        return scheduler, record, child

    @staticmethod
    def _children_of(prompt: str) -> list[dict]:
        body = prompt.split(CHILDREN_HEADER, 1)[1]
        return json.loads(body.split(chr(10) + MESSAGES_HEADER, 1)[0])

    def test_first_render_sends_a_child_in_full(self) -> None:
        scheduler, record, child = self._resumable_root_with_one_worker()

        prompt, _phase = scheduler._manager_prompt(record)

        rendered = self._children_of(prompt)
        self.assertEqual(1, len(rendered))
        self.assertEqual(child.agent_id, rendered[0]["agent_id"])
        self.assertNotIn("unchanged_since_last_delivered", rendered[0])
        self.assertIn("result", rendered[0])

    def test_a_manager_vnext_runs_already_reads_the_outcome_in_its_prompt(self) -> None:
        """Nothing had to be added here: the CHILDREN block carries ``result``.

        The external primary was the side with no route to a finished child's
        report.  A manager vNext runs is handed the whole result record in the
        prompt that tells it the child is terminal, so this pins that route
        rather than widening it.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        self.control.complete_agent(child.agent_id, {
            "outcome": "the word is MANGO",
            "verified": True,
            "evidence": ["said it once"],
        })

        prompt, _phase = scheduler._manager_prompt(record)

        rendered = self._children_of(prompt)
        self.assertEqual("the word is MANGO", rendered[0]["result"]["outcome"])
        self.assertTrue(rendered[0]["result"]["verified"])
        self.assertEqual(["said it once"], rendered[0]["result"]["evidence"])

    def test_an_unchanged_child_is_not_re_sent_on_the_next_wake(self) -> None:
        scheduler, record, child = self._resumable_root_with_one_worker()
        first, _phase = scheduler._manager_prompt(record)

        second, _phase = scheduler._manager_prompt(record)

        rendered = self._children_of(second)
        self.assertEqual(1, len(rendered))
        self.assertTrue(rendered[0]["unchanged_since_last_delivered"])
        self.assertEqual(child.agent_id, rendered[0]["agent_id"])
        self.assertNotIn("result", rendered[0])
        self.assertIn("inspect", rendered[0]["note"])
        self.assertLess(len(second), len(first))

    def test_a_child_that_moves_is_sent_in_full_again(self) -> None:
        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler._manager_prompt(record)
        scheduler._manager_prompt(record)
        self.control.block_agent(child.agent_id, "needs a decision from the manager")

        third, _phase = scheduler._manager_prompt(record)

        rendered = self._children_of(third)
        self.assertNotIn("unchanged_since_last_delivered", rendered[0])
        self.assertEqual("blocked", rendered[0]["status"])
        self.assertEqual("needs a decision from the manager", rendered[0]["blocker"])

    def test_a_reference_is_never_larger_than_the_record_it_replaces(self) -> None:
        scheduler, record, _child = self._resumable_root_with_one_worker()

        first, _phase = scheduler._manager_prompt(record)
        second, _phase = scheduler._manager_prompt(record)

        # The saving has to hold for a child carrying almost nothing, which is
        # the worst case for a substitution that costs a sentence of its own.
        self.assertLessEqual(len(second), len(first))

    def test_a_prompt_nobody_saw_leaves_the_children_ledger_untouched(self) -> None:
        """A refused turn must not mark a child as already delivered.

        The prompt is built before the runtime is asked to start the turn, and
        building it is what records "this manager has now seen child X". If the
        runtime then refuses, nobody read that prompt, but the ledger would say
        otherwise and the next prompt would summarise a record the manager never
        received. The mail offset beside it has always been rolled back for
        exactly this reason.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        self.assertEqual({}, scheduler._children_delivered.get(record.agent_id, {}))

        def refuse(*args, **kwargs):
            raise ManagedSessionError("runtime refused the turn")

        original = self.managed.start_turn
        self.managed.start_turn = refuse
        try:
            with self.assertRaises(ManagedSessionError):
                scheduler._start_turn(record)
        finally:
            self.managed.start_turn = original

        self.assertEqual({}, scheduler._children_delivered.get(record.agent_id, {}))
        prompt, _phase = scheduler._manager_prompt(record)
        rendered = self._children_of(prompt)
        self.assertEqual(child.agent_id, rendered[0]["agent_id"])
        self.assertNotIn("unchanged_since_last_delivered", rendered[0])
        self.assertIn("result", rendered[0])

    def test_a_prompt_nobody_saw_gives_back_the_wakes_it_drained(self) -> None:
        """A blocker is reported once. A refused turn must not consume it.

        Building the prompt is what drains a manager's wakes, exactly as it is
        what marks its messages delivered. Both have to be put back when the
        runtime refuses to start the turn, or the one signal that says a child
        stopped is spent on a prompt nobody read and never reported again.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        self.control.block_agent(child.agent_id, "stopped")
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(1, len(session.wakes[record.agent_id]))

        def refuse(*args, **kwargs):
            raise ManagedSessionError("runtime refused the turn")

        original = self.managed.start_turn
        self.managed.start_turn = refuse
        try:
            with self.assertRaises(ManagedSessionError):
                scheduler._start_turn(record)
        finally:
            self.managed.start_turn = original

        self.assertEqual(1, len(session.wakes.get(record.agent_id, [])))
        prompt, _phase = scheduler._manager_prompt(record)
        wakes = json.loads(prompt.split("WAKES:" + chr(10), 1)[1].split(chr(10) + "CHILDREN:", 1)[0])
        self.assertEqual(["child-blocked"], [wake["reason"] for wake in wakes])

    def test_a_blocker_survives_an_evidence_reader_that_explodes(self) -> None:
        """The catch around evidence has to mean what its comment says.

        The journal is read off disk, so an OSError is as plausible there as a
        protocol error. Any exception escaping would leave the child RUNNING
        with a finished turn, which nothing restarts, and its manager would
        never be told it stopped.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        self.control.start_turn(child.agent_id, thread_id="thread-x")
        self.assertEqual(AgentStatus.RUNNING, child.status)

        def explode(*args, **kwargs):
            raise RuntimeError("effects reader exploded")

        original = self.managed.runtime_effect_summary
        self.managed.runtime_effect_summary = explode
        try:
            scheduler._block_with_evidence(child, "runtime turn failed")
        finally:
            self.managed.runtime_effect_summary = original

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual("runtime turn failed", child.blocker)
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(
            ["child-blocked"],
            [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def test_reporting_a_broken_exception_still_returns_a_blocker(self) -> None:
        class Boom(Exception):
            def __str__(self):
                raise RuntimeError("str exploded")

        scheduler, _session, worker = self._resumable_root_with_one_worker()
        turn = types.SimpleNamespace(
            agent_id=worker.agent_id, provider="codex", control_turn_id="t-1",
            runtime=types.SimpleNamespace(turn_id="native-1"),
        )
        reason = scheduler._report_turn_failure(worker, _TurnFinished(worker.agent_id, turn, error=Boom()))
        self.assertIn("Boom", reason)
        self.assertIn("unprintable", reason)

    def _finish_with(self, failure: BaseException, *, elapsed: float, wall_step: float = 0.0):
        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        self.control.start_turn(child.agent_id, thread_id="thread-y")
        scheduler._now = lambda: 10_000.0 + wall_step
        scheduler._monotonic = lambda: 10_000.0
        scheduler._turn_started_at[child.agent_id] = 10_000.0 - elapsed
        scheduler._handle_turn_finished(_TurnFinished(
            child.agent_id,
            types.SimpleNamespace(
                agent_id=child.agent_id,
                provider="claude",
                control_turn_id="control-turn-1",
                runtime=types.SimpleNamespace(thread_id="thread-y", turn_id="native-1"),
            ),
            error=failure,
        ))
        return scheduler, child

    def test_wall_clock_step_does_not_claim_a_short_timeout_used_the_limit(self) -> None:
        _scheduler, child = self._finish_with(
            RuntimeError("Claude bridge timed out waiting for wait_turn"),
            elapsed=10.0, wall_step=3600.0,
        )
        self.assertNotIn("reached its", child.blocker)

    def test_a_turn_that_used_up_its_limit_says_so_and_how_to_go_on(self) -> None:
        """The bridge's timeout sentence read like a fault after a 30-minute turn."""

        scheduler, child = self._finish_with(
            RuntimeError("Claude bridge timed out waiting for wait_turn"),
            elapsed=1800.0,
        )
        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertIn("timed out waiting for wait_turn", child.blocker)
        self.assertIn("reached its 30m 00s limit", child.blocker)
        self.assertIn("retry continues the same session", child.blocker)

    _HOOK_SENTENCE = (
        "PreToolUse hook did not respond before its timeout (host client may be unreachable)"
    )

    def _finish_turn_with_result(self, scheduler, child, result, number):
        self.control.start_turn(child.agent_id, thread_id="thread-h")
        scheduler._handle_turn_finished(_TurnFinished(
            child.agent_id,
            types.SimpleNamespace(
                agent_id=child.agent_id, provider="claude", phase="work",
                control_turn_id=f"control-turn-{number}",
                runtime=types.SimpleNamespace(thread_id="thread-h", turn_id=f"native-{number}"),
            ),
            result=result,
        ))

    def test_a_turn_that_ends_on_hook_failures_blocks_and_wakes_the_parent(self) -> None:
        """Gap 2 of the 3-4 October stall.

        The worker's last tool calls all failed because its hooks got no
        answer.  The turn still ended normally, so the worker went READY
        waiting for mail and its manager was never told.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        # A turn whose tool calls ran is the ordinary end: nobody is woken.
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 1)
        self.assertEqual(AgentStatus.READY, child.status)
        self.assertEqual([], session.wakes.get(record.agent_id, []))

        self._finish_turn_with_result(scheduler, child, {
            "status": "completed",
            "hook_failures": {"count": 4, "first": self._HOOK_SENTENCE},
        }, 2)

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual(
            f"4 tool calls failed in its last turn: {self._HOOK_SENTENCE}", child.blocker,
        )
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def test_an_unsolicited_turn_that_ends_on_hook_failures_blocks_a_resting_worker(self) -> None:
        """The provider's own turn between vNext turns goes through the same check."""

        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 1)
        self.assertEqual(AgentStatus.READY, child.status)

        failures = {"count": 2, "first": self._HOOK_SENTENCE}
        # A turn that ran its tools, or one that did not complete, is no news.
        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "completed"})
        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "stream-ended", "hook_failures": failures})
        self.assertTrue(scheduler._events.empty())
        # While a vNext turn runs, that turn's own end decides.
        self.control.start_turn(child.agent_id, thread_id="thread-h")
        scheduler._active_turns[child.agent_id] = object()
        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "completed", "hook_failures": failures})
        scheduler._handle_unsolicited_turn_ended(scheduler._events.get_nowait())
        self.assertEqual(AgentStatus.RUNNING, child.status)
        scheduler._active_turns.pop(child.agent_id)
        self.control.finish_turn(child.agent_id)

        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "completed", "hook_failures": failures})
        scheduler._handle_unsolicited_turn_ended(scheduler._events.get_nowait())

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual(f"2 tool calls failed in its last turn: {self._HOOK_SENTENCE}", child.blocker)
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def test_unsolicited_hook_failures_survive_a_vnext_turn_that_started_first(self) -> None:
        """The failure is queued, then a vNext turn starts before it is read.

        Before the fix the handler dropped it, saying the new turn's end would
        check; but that turn reports only its own tool results, so a plain
        status reply ended the worker READY and nobody was woken.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 1)
        failures = {"count": 2, "first": self._HOOK_SENTENCE}
        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "completed", "hook_failures": failures})
        # A new vNext turn starts before the loop reads the event.
        self.control.start_turn(child.agent_id, thread_id="thread-h")
        scheduler._active_turns[child.agent_id] = object()
        scheduler._handle_unsolicited_turn_ended(scheduler._events.get_nowait())
        self.assertEqual(AgentStatus.RUNNING, child.status)
        self.control.finish_turn(child.agent_id)

        # That turn ends on a status reply with no failures of its own.
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 2)

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual(f"2 tool calls failed in its last turn: {self._HOOK_SENTENCE}", child.blocker)
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def _defer_hook_failures_behind_a_vnext_turn(self, scheduler, child):
        failures = {"count": 2, "first": self._HOOK_SENTENCE}
        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "completed", "hook_failures": failures})
        self.control.start_turn(child.agent_id, thread_id="thread-h")
        scheduler._active_turns[child.agent_id] = object()
        scheduler._handle_unsolicited_turn_ended(scheduler._events.get_nowait())
        self.assertEqual(AgentStatus.RUNNING, child.status)
        self.control.finish_turn(child.agent_id)

    def test_deferred_hook_failures_survive_an_interrupted_vnext_turn(self) -> None:
        """The vNext turn the failure waited for is interrupted on purpose.

        Before the fix the interrupt branch returned before the deferred
        blocker was applied, so the worker went READY and its parent was
        never told.
        """

        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 1)
        self._defer_hook_failures_behind_a_vnext_turn(scheduler, child)

        scheduler._interrupted_agents.add(child.agent_id)
        self._finish_turn_with_result(scheduler, child, {
            "status": "interrupted", "terminal_reason": "interrupted_before_dispatch",
        }, 2)

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual(f"2 tool calls failed in its last turn: {self._HOOK_SENTENCE}", child.blocker)
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def test_a_deferred_blocker_applied_on_interrupt_is_not_applied_again(self) -> None:
        """Once spent, the failure does not block the worker's next turn too."""

        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 1)
        self._defer_hook_failures_behind_a_vnext_turn(scheduler, child)
        scheduler._interrupted_agents.add(child.agent_id)
        self._finish_turn_with_result(scheduler, child, {"status": "interrupted"}, 2)
        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertNotIn(child.agent_id, scheduler._deferred_blockers)

        # The manager retries the worker and its next turn ends normally.
        scheduler._interrupted_agents.discard(child.agent_id)
        self.control.retry_agent(
            requester_id=record.agent_id, agent_id=child.agent_id,
            revised_task_contract={"objective": "carry on"},
        )
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 3)

        self.assertEqual(AgentStatus.READY, child.status)
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    _TOKEN_REASON = "needs a GitHub token for the release repository"
    _REPORTED_BLOCKER = (
        "reported by the worker: needs a GitHub token for the release repository. "
        "Answer with send_message, then retry it, to resume it on the same session"
    )

    def _worker_in_a_turn(self, scheduler):
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER, model_id=WORKER,
            objective="Publish the release",
            task_contract={"criteria": ["release published"]},
        )
        self.control.start_turn(worker.agent_id, thread_id="thread-w")
        return worker

    def _end_worker_turn(self, scheduler, worker, result):
        scheduler._handle_turn_finished(_TurnFinished(
            worker.agent_id,
            types.SimpleNamespace(
                agent_id=worker.agent_id, provider="claude", phase="work",
                control_turn_id="control-turn-w",
                runtime=types.SimpleNamespace(thread_id="thread-w", turn_id="native-w"),
            ),
            result=result,
        ))

    def test_a_worker_that_reports_blocked_is_blocked_when_its_turn_ends(self) -> None:
        """A worker that cannot go on says so, and its manager is woken with why."""

        scheduler, record, _branch = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        worker = self._worker_in_a_turn(scheduler)

        answer = payload(scheduler._manager_handler(
            worker.agent_id, "report_blocked", {"reason": self._TOKEN_REASON}, None,
        ))
        _deliver_reported_blocks(scheduler)

        self.assertTrue(answer["success"], answer)
        self.assertIn("End your turn now", answer["message"])
        # The tool only records the reason: the turn is still the worker's.
        self.assertEqual(AgentStatus.RUNNING, worker.status)
        self.assertEqual([], session.wakes.get(record.agent_id, []))

        self._end_worker_turn(scheduler, worker, {"status": "completed"})

        self.assertEqual(AgentStatus.BLOCKED, worker.status)
        self.assertEqual(self._REPORTED_BLOCKER, worker.blocker)
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )
        inspected = payload(scheduler._manager_handler(
            record.agent_id, "inspect", {"agent_id": worker.agent_id, "deep": False}, None,
        ))
        self.assertEqual(self._REPORTED_BLOCKER, inspected["agent"]["blocker"])

    def test_the_primary_cannot_report_itself_blocked(self) -> None:
        """The primary talks to its user directly and has no manager to wake."""

        scheduler, record, _branch = self._resumable_root_with_one_worker()
        self.control.start_turn(record.agent_id)

        answer = payload(scheduler._manager_handler(
            record.agent_id, "report_blocked", {"reason": self._TOKEN_REASON}, None,
        ))

        self.assertFalse(answer["success"], answer)
        self.assertEqual("primary-reports-to-user", answer["error_code"])
        self.assertIn("ask your user directly", answer["error"])
        self.assertNotIn(record.agent_id, scheduler._deferred_blockers)

    def test_a_message_alone_leaves_a_reported_block_in_place(self) -> None:
        """send_message only queues mail for a blocked worker; retry resumes it."""

        scheduler, record, _branch = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        worker = self._worker_in_a_turn(scheduler)
        payload(scheduler._manager_handler(
            worker.agent_id, "report_blocked", {"reason": self._TOKEN_REASON}, None,
        ))
        _deliver_reported_blocks(scheduler)
        self._end_worker_turn(scheduler, worker, {"status": "completed"})

        sent = payload(scheduler._manager_handler(record.agent_id, "send_message", {
            "agent_id": worker.agent_id, "message": "the token is in RELEASE_TOKEN",
        }, None))
        self.assertTrue(sent["success"], sent)
        self.assertEqual(AgentStatus.BLOCKED, worker.status)

        retried = payload(scheduler._manager_handler(record.agent_id, "retry", {
            "agent_id": worker.agent_id, "task_contract": {"criteria": ["release published"]},
        }, None))
        self.assertTrue(retried["success"], retried)
        self.assertEqual(AgentStatus.READY, worker.status)
        self.assertEqual("", worker.blocker)

    def test_a_report_the_loop_sees_after_the_turn_ended_still_blocks(self) -> None:
        """The turn can end between the tool call and the loop's bookkeeping.

        The tool used to read and then write the deferred map from the
        handler thread.  An interrupt that ended the turn between the two
        left the reason in the map after the turn's end had emptied it: the
        worker sat READY and its parent was never woken.
        """

        scheduler, record, _branch = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        worker = self._worker_in_a_turn(scheduler)
        ended = []
        test = self

        class TurnEndsDuringLookup(dict):
            def get(self, key, default=None):
                if not ended:
                    ended.append(True)
                    test._end_worker_turn(scheduler, worker, {"status": "interrupted"})
                return super().get(key, default)

        scheduler._deferred_blockers = TurnEndsDuringLookup()
        scheduler._interrupted_agents.add(worker.agent_id)

        answer = payload(scheduler._manager_handler(
            worker.agent_id, "report_blocked", {"reason": self._TOKEN_REASON}, None,
        ))
        self.assertTrue(answer["success"], answer)
        if not ended:
            ended.append(True)
            self._end_worker_turn(scheduler, worker, {"status": "interrupted"})
        self.assertEqual(AgentStatus.READY, worker.status)
        _deliver_reported_blocks(scheduler)

        self.assertEqual(AgentStatus.BLOCKED, worker.status)
        self.assertEqual(self._REPORTED_BLOCKER, worker.blocker)
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )
        self.assertNotIn(worker.agent_id, scheduler._deferred_blockers)

    def _cancel_during_block(self, handle) -> None:
        """A cancel from the handler thread lands between the READY check and the block."""

        scheduler, record, _branch = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        worker = self._worker_in_a_turn(scheduler)
        self._end_worker_turn(scheduler, worker, {"status": "completed"})
        self.assertEqual(AgentStatus.READY, worker.status)
        wakes_before = list(session.wakes.get(record.agent_id, []))
        original = scheduler._record_stopped_progress

        def cancel_first(*args, **kwargs):
            self.control.cancel_agent(requester_id=record.agent_id, agent_id=worker.agent_id)
            return original(*args, **kwargs)

        scheduler._record_stopped_progress = cancel_first

        handle(scheduler, worker)

        self.assertEqual(AgentStatus.CANCELLED, worker.status)
        self.assertNotIn(
            "child-blocked",
            [wake["reason"] for wake in session.wakes.get(record.agent_id, [])[len(wakes_before):]],
        )

    def test_a_cancel_racing_a_reported_block_does_not_stop_the_loop(self) -> None:
        self._cancel_during_block(lambda scheduler, worker: scheduler._handle_block_reported(
            _BlockReported(worker.agent_id, self._TOKEN_REASON),
        ))

    def test_a_cancel_racing_a_hook_failure_block_does_not_stop_the_loop(self) -> None:
        self._cancel_during_block(lambda scheduler, worker: scheduler._handle_unsolicited_turn_ended(
            _UnsolicitedTurnEnded(worker.agent_id, {}, blocker="hooks failed"),
        ))

    def test_a_reported_block_is_dropped_when_the_turn_fails(self) -> None:
        """A failed turn is the parent's news; the report does not wake it twice."""

        scheduler, record, _branch = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        worker = self._worker_in_a_turn(scheduler)
        answer = payload(scheduler._manager_handler(
            worker.agent_id, "report_blocked", {"reason": self._TOKEN_REASON}, None,
        ))
        self.assertTrue(answer["success"], answer)

        self._end_worker_turn(scheduler, worker, {"status": "failed"})

        self.assertEqual(AgentStatus.FAILED, worker.status)
        self.assertNotIn("reported by the worker", worker.blocker)
        self.assertNotIn(worker.agent_id, scheduler._deferred_blockers)
        self.assertEqual(
            ["child-failed"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def test_a_provider_stream_that_ends_between_turns_blocks_a_resting_worker(self) -> None:
        """The bridge can no longer read the worker's session: say so to the parent."""

        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(lifecycle=lambda *args: None)
        session = self.control.sessions[self.root.session_id]
        self._finish_turn_with_result(scheduler, child, {"status": "completed"}, 1)
        self.assertEqual(AgentStatus.READY, child.status)

        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "frame-dropped", "error": "MessageParseError"})
        self.assertTrue(scheduler._events.empty())
        scheduler.observe_unsolicited_turn(child.agent_id, {"status": "reader-ended", "error": "CLIConnectionError"})
        scheduler._handle_unsolicited_turn_ended(scheduler._events.get_nowait())

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual(
            "the Claude session's message stream stopped between turns (CLIConnectionError); "
            "nothing answers its hooks or approvals any more",
            child.blocker,
        )
        self.assertEqual(
            ["child-blocked"], [wake["reason"] for wake in session.wakes[record.agent_id]],
        )

    def test_an_idle_timeout_inside_the_limit_carries_no_limit_hint(self) -> None:
        scheduler, child = self._finish_with(
            RuntimeError("app-server event wait timed out after 600s without activity"),
            elapsed=600.0,
        )
        self.assertIn("without activity", child.blocker)
        self.assertNotIn("limit", child.blocker)

    def test_a_failed_worker_turn_records_and_names_its_own_cause(self) -> None:
        """"runtime turn failed" was all five workers ever said.

        The exception was kept only for the Root Manager and dropped for every
        worker, so a whole team could die at once and leave nothing on disk to
        say whether that was one cause or five.  The cause belongs in the
        blocker the manager reads and in the record on disk.
        """

        lifecycle: list[tuple[str, dict]] = []
        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(
            lifecycle=lambda event_type, agent, data: lifecycle.append(
                (event_type, dict(data))
            )
        )
        self.control.start_turn(child.agent_id, thread_id="thread-y")

        cause = TimeoutError("the socket went quiet")
        failure = RuntimeError("app-server event wait timed out after 600s without activity")
        failure.__cause__ = cause
        event = _TurnFinished(
            child.agent_id,
            types.SimpleNamespace(
                agent_id=child.agent_id,
                provider="codex",
                control_turn_id="control-turn-1",
                runtime=types.SimpleNamespace(thread_id="thread-y", turn_id="native-1"),
            ),
            error=failure,
        )

        scheduler._handle_turn_finished(event)

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertIn("without activity", child.blocker)
        self.assertIn("RuntimeError", child.blocker)

        errors = [data for event_type, data in lifecycle if event_type == "provider_error"]
        self.assertEqual(1, len(errors))
        reported = errors[0]
        self.assertEqual("run_turn", reported["phase"])
        self.assertEqual("codex", reported["provider"])
        self.assertEqual("RuntimeError", reported["kind"])
        self.assertEqual("control-turn-1", reported["turn_id"])
        self.assertEqual("native-1", reported["native_turn_id"])
        self.assertEqual("TimeoutError: the socket went quiet", reported["cause"])
        self.assertIn("RuntimeError", reported["traceback"])

    def test_a_failed_root_turn_still_raises_with_its_cause_attached(self) -> None:
        scheduler, record, _child = self._resumable_root_with_one_worker()
        failure = RuntimeError("root turn died")
        event = _TurnFinished(
            record.agent_id,
            types.SimpleNamespace(
                agent_id=record.agent_id,
                provider="codex",
                control_turn_id="control-turn-root",
                runtime=types.SimpleNamespace(thread_id="thread-root", turn_id="native-root"),
            ),
            error=failure,
        )

        with self.assertRaises(SchedulerError) as caught:
            scheduler._handle_turn_finished(event)
        self.assertIs(failure, caught.exception.__cause__)

    def test_a_failed_worker_is_told_why_the_provider_ended_its_turn(self) -> None:
        """The reason reaches the manager through the call site, or not at all.

        An Opus worker died 43 seconds into a read-only packet on 2026-09-18 and
        cost $0.80.  The turn result already carried the SDK's subtype and its
        terminal reason, and the blocker threw both away, so the manager could
        not tell a provider fault from the worker arguing itself into a corner.
        Checking the sentence builder alone would leave the call site free to go
        back to the bare status, which is exactly the bug.
        """

        scheduler, _record, child = self._resumable_root_with_one_worker()
        self.control.start_turn(child.agent_id, thread_id="thread-y")
        # A cut-off stream is retried now, so the worker only reaches its
        # blocker once the retry budget is spent.  The detail it carries is
        # still the thing under test.
        scheduler._provider_retries[child.agent_id] = PROVIDER_RETRY_LIMIT

        scheduler._handle_turn_finished(_TurnFinished(
            child.agent_id,
            types.SimpleNamespace(
                agent_id=child.agent_id,
                provider="claude",
                control_turn_id="control-turn-1",
                runtime=types.SimpleNamespace(thread_id="thread-y", turn_id="native-1"),
            ),
            result={
                "status": "failed",
                "terminal_reason": "aborted_streaming",
                "subtype": "error_during_execution",
                "api_error_status": 529,
            },
        ))

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertIn("aborted_streaming", child.blocker)
        self.assertIn("error_during_execution", child.blocker)
        self.assertIn("HTTP 529", child.blocker)
        self.assertIn("3 attempts", child.blocker)

    def _provider_abort(self, child, **extra):
        """A turn the provider cut off part way, in the shape the SDK returns."""

        return _TurnFinished(
            child.agent_id,
            types.SimpleNamespace(
                agent_id=child.agent_id,
                provider="claude",
                control_turn_id="control-turn-abort",
                runtime=types.SimpleNamespace(thread_id="thread-y", turn_id="native-1"),
            ),
            result={
                "status": "failed",
                "terminal_reason": "aborted_tools",
                "subtype": "error_during_execution",
                **extra,
            },
        )

    def test_a_worker_whose_provider_hung_up_is_retried_rather_than_blocked(self) -> None:
        """A cut-off stream is the provider's fault, so the packet gets another go.

        Measured on one swarm: 33 provider faults, every one of them a turn
        ended part way.  One worker died that way twice on the same packet and
        cost $1.10 with no files written, and nothing tried again.
        """

        scheduler, _record, child = self._resumable_root_with_one_worker()
        self.control.start_turn(child.agent_id, thread_id="thread-y")

        scheduler._handle_turn_finished(self._provider_abort(child))

        self.assertEqual(AgentStatus.READY, child.status)
        self.assertEqual(1, scheduler._provider_retries[child.agent_id])
        self.assertNotIn(child.agent_id, scheduler._awaiting_message_agents)

    def test_a_rate_limited_turn_is_never_retried(self) -> None:
        """A rate limit is a wait, and trying again spends what the wait saved."""

        scheduler, _record, child = self._resumable_root_with_one_worker()
        self.control.start_turn(child.agent_id, thread_id="thread-y")

        scheduler._handle_turn_finished(
            self._provider_abort(child, rate_limited=True)
        )

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertEqual({}, scheduler._provider_retries)
        self.assertIn("rate-limited", child.blocker)

    def test_a_failed_turn_with_no_terminal_reason_fails_on_the_first_attempt(self) -> None:
        """Only a cut-off stream or a cut-off tool call earns another go.

        A failed turn can arrive carrying no terminal reason at all: the
        bridge writes the field through only when the SDK reported a string,
        and the Opus worker that died 43 seconds into a read-only packet on
        2026-09-18 ended exactly that way.  A turn like that says nothing
        about a stream cut off part way, so it fails on the first attempt
        and spends no retry on it.
        """

        scheduler, _record, child = self._resumable_root_with_one_worker()
        self.control.start_turn(child.agent_id, thread_id="thread-y")

        scheduler._handle_turn_finished(
            self._provider_abort(child, terminal_reason=None)
        )

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertEqual({}, scheduler._provider_retries)
        self.assertEqual({}, scheduler._retry_not_before)

    def test_the_retries_run_out_and_the_blocker_says_how_many_times_it_tried(self) -> None:
        """The manager reading the blocker has to know the packet was tried."""

        scheduler, _record, child = self._resumable_root_with_one_worker()

        for _attempt in range(PROVIDER_RETRY_LIMIT + 1):
            self.control.start_turn(child.agent_id, thread_id="thread-y")
            scheduler._handle_turn_finished(self._provider_abort(child))

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertEqual(PROVIDER_RETRY_LIMIT, scheduler._provider_retries[child.agent_id])
        self.assertIn("3 attempts", child.blocker)

    def test_a_retried_agent_is_held_back_until_its_delay_has_passed(self) -> None:
        """The backoff is a deadline the start loop reads.

        The loop is single-threaded, so sleeping through one agent's backoff
        would stall every other agent in the tree.
        """

        scheduler, _record, child = self._resumable_root_with_one_worker()
        scheduler._watch_turn = lambda _turn: None
        # Park the Root so this test watches one agent's start decision.
        scheduler._awaiting_message_agents.add(self.root.agent_id)
        self.control.start_turn(child.agent_id, thread_id="thread-y")
        scheduler._handle_turn_finished(self._provider_abort(child))

        scheduler._start_ready_agents()
        held = scheduler._retry_not_before[child.agent_id]

        self.assertGreater(held, time.monotonic())
        self.assertLessEqual(held, time.monotonic() + PROVIDER_RETRY_DELAYS[0])
        self.assertNotIn(child.agent_id, scheduler._active_turns)

        scheduler._retry_not_before[child.agent_id] = time.monotonic() - 0.01
        scheduler._start_ready_agents()

        self.assertIn(child.agent_id, scheduler._active_turns)
        self.assertNotIn(child.agent_id, scheduler._retry_not_before)

    def test_the_delivered_ledger_is_kept_per_manager(self) -> None:
        scheduler, record, child = self._resumable_root_with_one_worker()
        scheduler._manager_prompt(record)
        other_manager = "another-manager"

        self.assertEqual(
            {self.root.agent_id},
            set(scheduler._children_delivered),
        )
        self.assertNotIn(other_manager, scheduler._children_delivered)
        self.assertIn(child.agent_id, scheduler._children_delivered[self.root.agent_id])

    def test_a_manager_is_told_to_fan_out_before_it_yields(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.ROOT_MANAGER)

        # The old clause told every manager to sleep the moment it had
        # delegated, which is what made a tree run one child at a time.
        self.assertNotIn("End nonterminal turns with await_children", instructions)
        self.assertIn("Launch every ready lane before doing further analysis", instructions)
        self.assertIn("do not delegate one child and stop", instructions)
        self.assertIn(
            "Yielding is what you do when you have nothing left to do", instructions
        )

    def test_a_manager_is_told_not_to_poll_and_how_to_handle_a_stall(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.BRANCH_MANAGER)

        self.assertIn("Do not poll", instructions)
        self.assertIn("inspect once, steer once with a narrower packet", instructions)
        self.assertIn("no short blanket timeout", instructions)

    def test_explicit_task_topology_constraints_take_precedence_over_discretion(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.WORKER)

        self.assertIn("EXPLICIT TASK CONSTRAINTS OVERRIDE DEFAULT ORGANIZATION", instructions)
        self.assertIn("agent count, direct parentage, descriptive roles, model, effort", instructions)
        self.assertIn("do not add a coordination layer merely to organize work", instructions)

    def test_a_worker_is_told_to_call_complete_agent_when_its_own_task_is_done(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.WORKER)

        self.assertIn("HOW YOU FINISH", instructions)
        self.assertIn(
            "A written answer at the end of your turn does not finish your task", instructions
        )
        self.assertIn("verified set to whether you checked the result", instructions)
        self.assertIn(
            "If you did delegate children, wait until every one of them is terminal",
            instructions,
        )

    def test_the_closing_worker_block_is_kept_away_from_managers(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        self.assertNotIn(
            "HOW YOU FINISH",
            scheduler._manager_instructions(AgentRole.ROOT_MANAGER),
        )
        self.assertNotIn(
            "HOW YOU FINISH",
            scheduler._manager_instructions(AgentRole.BRANCH_MANAGER),
        )

    def test_a_worker_still_receives_the_shared_instruction_body(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.WORKER)

        self.assertIn("ONE WRITER PER WORKSPACE", instructions)

    def test_selected_catalog_claims_are_labeled_once_in_manager_instructions(self) -> None:
        self.control.registry.cards[MANAGER] = ModelCard(
            MANAGER,
            frozenset({AgentRole.ROOT_MANAGER, AgentRole.BRANCH_MANAGER, AgentRole.WORKER}),
            claims=(RegistryClaim(
                EvidenceKind.OBSERVATION,
                "Local canary measured this model.",
                "docs/research/local-canary.md",
            ),),
        )
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.ROOT_MANAGER)

        self.assertEqual(1, instructions.count('"model_id": "manager-model"'))
        self.assertIn('"kind": "observation"', instructions)
        self.assertIn('"evidence_pointer": "docs/research/local-canary.md"', instructions)
        self.assertIn("Missing claims mean capability, quality, and cost are unknown", instructions)

    def test_every_tool_the_instructions_name_is_a_tool_the_manager_has(self) -> None:
        """Doctrine lifted from another harness can name a tool this one lacks.

        The stall ladder came from a manager asset written against a different
        tool surface, and it told managers to "interrupt" a stalled child. There
        is no interrupt tool here, so the ladder's last rung was a dead end: the
        manager would have to guess which of the tools it does have was meant.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        available = {tool["name"] for tool in scheduler.manager_tools()}

        for role in (AgentRole.ROOT_MANAGER, AgentRole.BRANCH_MANAGER):
            instructions = scheduler._manager_instructions(role)
            # Whole words only: "wait" lives inside "await_children" and would
            # otherwise read as a tool nobody named.
            named = {
                candidate
                for candidate in (
                    "delegate",
                    "await_children",
                    "inspect",
                    "steer",
                    "retry",
                    "replace",
                    "cancel_agent",
                    "resolve_approval",
                    "interrupt",
                    "complete_branch",
                    "complete_session",
                    "pause",
                )
                if re.search(r"\b" + candidate + r"\b", instructions)
            }
            self.assertTrue(named)
            self.assertEqual(set(), named - available, f"{role.value} is told to use a tool it lacks")

    def test_the_order_of_work_puts_fanning_out_before_yielding(self) -> None:
        """The sequence is the whole point, so the sequence is what is checked.

        A prompt could keep every phrase and still be wrong by telling a manager
        to yield before it has launched anything.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        instructions = scheduler._manager_instructions(AgentRole.ROOT_MANAGER)

        launch = instructions.index("Launch every ready lane")
        own_work = instructions.index("the work that can be done now")
        yield_at = instructions.index("Then yield with await_children")
        self.assertLess(launch, own_work)
        self.assertLess(own_work, yield_at)

    def test_the_private_scope_still_says_what_it_is_good_for(self) -> None:
        """Adding the emptiness fact must not cost the reason to choose it."""

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        delegate = next(
            tool for tool in scheduler.manager_tools() if tool["name"] == "delegate"
        )
        described = delegate["inputSchema"]["properties"]["workspace"]["description"]

        self.assertIn("at the same time", described)
        self.assertIn("collide", described)

    def test_a_manager_is_told_the_one_writer_rule(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        instructions = scheduler._manager_instructions(AgentRole.BRANCH_MANAGER)

        self.assertIn("ONE WRITER PER WORKSPACE", instructions)
        self.assertIn("disjoint scopes, or sequence them", instructions)

    def test_the_initial_prompts_no_longer_pair_delegate_with_an_immediate_yield(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]

        prompt, phase = scheduler._manager_prompt(record)

        self.assertEqual("root-initial", phase)
        self.assertNotIn("then call await_children", prompt)
        self.assertIn("cheapest capable agents", prompt)
        self.assertIn("nothing useful to do now", prompt)

    def test_the_private_scope_promises_no_boundary_it_does_not_have(self) -> None:
        """A live private Claude worker read a project file and a file outside
        the workspace.  The directory is a starting point and nothing more, so
        the description says that and stops promising confinement.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        delegate = next(
            tool
            for tool in scheduler.manager_tools()
            if tool["name"] == "delegate"
        )
        described = delegate["inputSchema"]["properties"]["workspace"]["description"]

        self.assertIn("fences nothing", described)
        self.assertNotIn("cannot read", described)
        self.assertNotIn("no sibling can see", described)

    def test_awaiting_only_stopped_children_is_refused_and_says_why(self) -> None:
        """Reporting a refused park as success was its own bug.

        The manager believed it had yielded, took another turn, awaited the
        same stopped child, and did that until the turn ceiling: roughly two
        hundred and fifty billed turns to reach the same place a deadlock
        reached in six. A refusal it can read is the whole fix.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        self.control.block_agent(child.agent_id, "needs a decision")
        # A manager's prompt drains its wakes before its turn, so by the time it
        # calls a tool the block is news it already has. Leaving the wake
        # pending exercises the other refusal entirely.
        self.control.drain_wakes(self.root.agent_id)
        self.control.start_turn(self.root.agent_id)

        result = scheduler._apply_manager_tool(
            self.root.agent_id, "await_children", {"agent_ids": [child.agent_id]}
        )

        read = payload(result)
        self.assertFalse(read["success"])
        self.assertEqual("nothing-to-wait-for", read["error_code"])
        self.assertIn("waiting cannot end", read["error"])
        needing = read["children_needing_a_decision"]
        self.assertEqual([child.agent_id], [item["agent_id"] for item in needing])
        self.assertEqual("blocked", needing[0]["status"])
        self.assertEqual("needs a decision", needing[0]["blocker"])
        # And the escape routes it names really are open to the manager.
        self.assertEqual(
            {"retry", "replace", "cancel_agent"},
            {"retry", "replace", "cancel_agent"}
            & {tool["name"] for tool in scheduler.manager_tools()},
        )

    def test_a_replacement_can_be_given_the_objective_the_old_one_should_have_had(self) -> None:
        """Replacing a child used to copy its goal across word for word.

        The tool took a new model and a new task contract, so a manager whose
        child was pointed at the wrong thing could only send the same wrong
        thing to a better model. Omitting the objective still carries the old
        one, which is what every existing caller does.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        schema = next(tool for tool in scheduler.manager_tools() if tool["name"] == "replace")
        parameters = schema["inputSchema"]
        self.assertIn("objective", parameters["properties"])
        self.assertNotIn("objective", parameters["required"])

        def child_to_replace(objective: str):
            child = self.control.spawn_agent(
                requester_id=self.root.agent_id,
                parent_agent_id=self.root.agent_id,
                role=AgentRole.BRANCH_MANAGER,
                model_id=MANAGER,
                objective=objective,
                task_contract={},
            )
            self.control.block_agent(child.agent_id, "pointed at the wrong thing")
            self.control.drain_wakes(self.root.agent_id)
            if self.control.sessions[self.root.session_id].agents[self.root.agent_id].active_turn_id is None:
                self.control.start_turn(self.root.agent_id)
            return child

        first = child_to_replace("count the rows in the wrong table")
        result = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "replace",
                {
                    "agent_id": first.agent_id,
                    "model_id": MANAGER,
                    "objective": "count the rows in orders",
                    "task_contract": {"criteria": ["one number"]},
                },
            )
        )
        self.assertTrue(result["success"], result)
        agents = self.control.sessions[self.root.session_id].agents
        self.assertEqual("count the rows in orders", agents[result["agent_id"]].objective)

        second = child_to_replace("count the rows in orders")
        kept = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "replace",
                {
                    "agent_id": second.agent_id,
                    "model_id": MANAGER,
                    "task_contract": {"criteria": ["one number"]},
                },
            )
        )
        self.assertTrue(kept["success"], kept)
        self.assertEqual("count the rows in orders", agents[kept["agent_id"]].objective)

    def test_replace_interrupts_old_turn_and_refuses_when_interrupt_fails(self) -> None:
        scheduler = self._scheduler()
        arguments = {"model_id": WORKER, "task_contract": {"criteria": ["done"]}}

        def running_worker():
            worker = self.control.spawn_agent(
                requester_id=self.root.agent_id,
                parent_agent_id=self.root.agent_id,
                role=AgentRole.WORKER,
                model_id=WORKER,
                objective="Work in the shared workspace",
                task_contract={},
            )
            self.control.start_turn(worker.agent_id)
            return worker

        old = running_worker()
        before_count = len(self.control.sessions[self.root.session_id].agents)

        def interrupt_before_replace(target_id: str) -> None:
            self.assertEqual(old.agent_id, target_id)
            self.assertEqual(AgentStatus.RUNNING, old.status)
            self.assertEqual(before_count, len(self.control.sessions[self.root.session_id].agents))

        with patch.object(scheduler, "_interrupt_turn", side_effect=interrupt_before_replace) as interrupt:
            replaced = payload(scheduler._manager_handler(
                self.root.agent_id, "replace", {**arguments, "agent_id": old.agent_id}, None,
            ))
        self.assertTrue(replaced["success"], replaced)
        interrupt.assert_called_once_with(old.agent_id)
        self.assertEqual(AgentStatus.REPLACED, old.status)

        still_running = running_worker()
        before_count = len(self.control.sessions[self.root.session_id].agents)
        with patch.object(scheduler, "_interrupt_turn", side_effect=TimeoutError("provider busy")):
            refused = payload(scheduler._manager_handler(
                self.root.agent_id, "replace", {**arguments, "agent_id": still_running.agent_id}, None,
            ))
        self.assertFalse(refused["success"], refused)
        self.assertEqual("interrupt-failed", refused["error_code"])
        self.assertIn("TimeoutError", refused["error"])
        self.assertEqual(AgentStatus.RUNNING, still_running.status)
        self.assertEqual(before_count, len(self.control.sessions[self.root.session_id].agents))

    def _timed_out_turn_event(self, agent_id: str, *, thread_id: str, message: str):
        """One finished-turn event whose only error is that our wait expired."""

        runtime = types.SimpleNamespace(thread_id=thread_id, turn_id="provider-turn")
        turn = types.SimpleNamespace(
            agent_id=agent_id,
            provider="codex",
            control_turn_id="control-turn",
            runtime=runtime,
        )
        return _TurnFinished(agent_id, turn, error=TimeoutError(message))

    def test_a_timed_out_turn_asks_the_provider_to_stop_that_turn(self) -> None:
        """Ported from the GPT reviewer's TIMEOUT probe.

        The probe printed `provider_interrupt_calls=0`: a wait that expired
        recorded the failure and left the provider turn running, so the model
        kept writing to the shared workspace under a record that already read
        failed, and whatever the manager started next was the second writer.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler, _record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(
            lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
        )
        self.control.start_turn(child.agent_id, thread_id="child-thread")

        scheduler._handle_turn_finished(self._timed_out_turn_event(
            child.agent_id,
            thread_id="child-thread",
            message="app-server event wait timed out after 1s without activity",
        ))

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertEqual(["child-thread"], self.adapter.interrupted)
        stop = [data for kind, _id, data in events if kind == "turn_timeout_interrupt"]
        self.assertEqual([{"outcome": "requested", "error": None}], [
            {"outcome": entry["outcome"], "error": entry["error"]} for entry in stop
        ])

    def test_a_refused_stop_on_a_timed_out_turn_is_recorded_and_not_raised(self) -> None:
        events: list[tuple[str, str, dict]] = []
        scheduler, _record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(
            lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
        )
        self.control.start_turn(child.agent_id, thread_id="child-thread")

        with patch.object(
            self.adapter, "interrupt", side_effect=RuntimeError("app server is gone")
        ):
            scheduler._handle_turn_finished(self._timed_out_turn_event(
                child.agent_id, thread_id="child-thread", message="wait timed out after 1s",
            ))

        self.assertEqual(AgentStatus.FAILED, child.status)
        refused = [data for kind, _id, data in events if kind == "turn_timeout_interrupt"]
        self.assertEqual(1, len(refused), refused)
        self.assertEqual("refused", refused[0]["outcome"])
        self.assertIn("app server is gone", refused[0]["error"])

    def _failed_turn(self, agent_id: str, *, thread_id: str, error: BaseException):
        """One finished-turn event whose wait failed for some other reason."""

        runtime = types.SimpleNamespace(thread_id=thread_id, turn_id="provider-turn")
        turn = types.SimpleNamespace(
            agent_id=agent_id,
            provider="codex",
            control_turn_id="control-turn",
            runtime=runtime,
        )
        return _TurnFinished(agent_id, turn, error=error)

    def test_a_dead_provider_process_is_not_interrupted(self) -> None:
        """The confirmed case: the bridge died, so the turn died with it.

        This test used to assert that any non-timeout error skipped the stop.
        The evidence is what earns that now, and a process whose poll() has an
        exit code is the evidence, so nothing is asked of a provider that is
        already gone and nothing is held back from the manager.
        """

        scheduler, _record, child = self._resumable_root_with_one_worker()
        self.control.start_turn(child.agent_id, thread_id="child-thread")
        self.adapter.process_ended = True

        scheduler._handle_turn_finished(self._failed_turn(
            child.agent_id,
            thread_id="child-thread",
            error=ClaudeRuntimeError("bridge refused the model"),
        ))

        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertEqual([], self.adapter.interrupted)
        self.assertNotIn(child.agent_id, scheduler._stopping_turns)

    def test_a_transport_failure_keeps_the_turn_stopping_and_asks_it_to_stop(self) -> None:
        """Ported from the GPT reviewer's ACTIVE_NON_TIMEOUT probe line.

        The probe printed `stop_requested 0 old_stopping False
        replace_success True new_started True`: the app-server WebSocket
        failing was read as the turn having ended, so no stop was sent, the
        hold was released, and a replacement started in the same workspace as
        a turn that may well have been running.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler, _record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(
            lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
        )
        self.control.start_turn(child.agent_id, thread_id="child-thread")
        event = self._failed_turn(
            child.agent_id,
            thread_id="child-thread",
            error=ConnectionError("remote app-server transport failed"),
        )
        with scheduler._lock:
            scheduler._active_turns[child.agent_id] = event.turn

        scheduler._handle_turn_finished(event)

        # The manager still reads the same failure it read before.
        self.assertEqual(AgentStatus.FAILED, child.status)
        self.assertNotIn(child.agent_id, scheduler._active_turns)
        self.assertIn(child.agent_id, scheduler._stopping_turns)
        self.assertEqual(["child-thread"], self.adapter.interrupted)
        stop = [data for kind, _id, data in events if kind == "turn_unconfirmed_stop_interrupt"]
        self.assertEqual([{"reason": "transport_failure", "outcome": "requested"}], [
            {"reason": entry["reason"], "outcome": entry["outcome"]} for entry in stop
        ])
        # The timeout record keeps its own name so older logs read the same.
        self.assertEqual([], [kind for kind, _id, _d in events if kind == "turn_timeout_interrupt"])

    def test_a_replacement_over_a_transport_failure_waits_behind_the_barrier(self) -> None:
        """Ported from the GPT reviewer's CROSS_ADAPTER_TRANSPORT_FAILURE line.

        The replacement is held by the barrier already there for timeouts, it
        blocks with the text that names the override, and only a manager that
        asks for the override by name starts it.
        """

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        scheduler._bind_agent(old)
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        event = self._failed_turn(
            old.agent_id,
            thread_id="old-thread",
            error=ConnectionError("remote app-server transport failed"),
        )
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = event.turn
        scheduler._handle_turn_finished(event)

        result = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": WORKER, "task_contract": {}},
            None,
        ))
        self.assertTrue(result["success"], result)
        replacement = self.control.sessions[self.root.session_id].agents[result["agent_id"]]
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()

        self.assertNotIn(replacement.agent_id, starts)
        self.assertEqual(AgentStatus.BLOCKED, replacement.status)
        self.assertIn("start_despite_unconfirmed_stop", replacement.blocker)

        plain = payload(scheduler._manager_handler(
            self.root.agent_id, "retry",
            {"agent_id": replacement.agent_id, "task_contract": {}}, None,
        ))
        scheduler._start_ready_agents()
        self.assertFalse(plain["success"], plain)
        self.assertEqual("unconfirmed-stop", plain["error_code"])
        self.assertNotIn(replacement.agent_id, starts)

        forced = payload(scheduler._manager_handler(
            self.root.agent_id, "retry",
            {"agent_id": replacement.agent_id, "task_contract": {},
             "start_despite_unconfirmed_stop": True}, None,
        ))
        scheduler._start_ready_agents()
        self.assertTrue(forced["success"], forced)
        self.assertIn(replacement.agent_id, starts)

    def test_a_dead_provider_process_releases_the_replacement_at_once(self) -> None:
        """The regression guard for the common crash.

        A local provider that died is the case with evidence, so a manager
        replacing the worker it killed waits for nothing: the whole barrier
        would otherwise be spent holding a replacement away from a workspace
        nobody is writing.
        """

        scheduler = self._scheduler()
        self.assertGreater(scheduler.replacement_barrier_seconds, 0.0)
        old = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        scheduler._bind_agent(old)
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        event = self._failed_turn(
            old.agent_id,
            thread_id="old-thread",
            error=ConnectionError("app-server stdin is unavailable"),
        )
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = event.turn
        self.adapter.process_ended = True

        scheduler._handle_turn_finished(event)

        self.assertNotIn(old.agent_id, scheduler._stopping_turns)
        self.assertEqual([], self.adapter.interrupted)

        result = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": WORKER, "task_contract": {}},
            None,
        ))
        replacement = self.control.sessions[self.root.session_id].agents[result["agent_id"]]
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()

        self.assertIn(replacement.agent_id, starts)
        self.assertEqual("", replacement.blocker)

    def test_a_process_that_exits_inside_the_barrier_releases_the_hold(self) -> None:
        """Item 5: the evidence that arrives a moment after the socket broke.

        The interrupt went out and the adapter cannot answer a wait, so the
        stop stays unconfirmed.  The process behind the turn exiting is the
        same evidence as before and is read again inside the barrier, so the
        hold ends when the process does.
        """

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 2.0
        _record, child = self.root, self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        scheduler._bind_agent(child)
        self.control.start_turn(child.agent_id, thread_id="child-thread")
        event = self._failed_turn(
            child.agent_id,
            thread_id="child-thread",
            error=ConnectionError("remote app-server transport failed"),
        )
        with scheduler._lock:
            scheduler._active_turns[child.agent_id] = event.turn
        scheduler._handle_turn_finished(event)
        self.assertIn(child.agent_id, scheduler._stopping_turns)

        # The process the turn ran in exits while the barrier is still open.
        self.adapter.process_ended = True

        deadline = time.monotonic() + 5.0
        while child.agent_id in scheduler._stopping_turns and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertNotIn(child.agent_id, scheduler._stopping_turns)

    def _replace_over_a_live_turn(self, scheduler):
        """Replace a worker whose provider turn is registered and still active."""

        old = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        scheduler._bind_agent(old)
        old_turn = self.managed.start_turn(
            old.agent_id, prompt="edit", effort="high", phase="test",
        )
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = old_turn
        result = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": WORKER, "task_contract": {}},
            None,
        ))
        self.assertTrue(result["success"], result)
        replacement = self.control.sessions[self.root.session_id].agents[result["agent_id"]]
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        return old, replacement, starts

    def _terminal_during_binding(self, action: str):
        """Make a READY worker terminal while its provider binding is in flight.

        Ported from the GPT reviewer's probe_bind_transition.py.  The starter
        passed its own READY check, then sat inside ``start_thread`` for as long
        as the provider took, and a replace or a cancel landed in that window.
        """

        records: list[tuple[str, str, dict]] = []
        self.adapter = BlockingBindAdapter()
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: records.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )
        # The Root holds the pause bit so this scan is about the worker alone.
        scheduler._interrupted_agents.add(self.root.agent_id)
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        errors: list[BaseException] = []

        def start_ready() -> None:
            try:
                scheduler._start_ready_agents()
            except BaseException as exc:  # noqa: BLE001 - the fault under test
                errors.append(exc)

        starter = threading.Thread(target=start_ready)
        starter.start()
        self.assertTrue(self.adapter.binding_started.wait(5))
        args: dict = {"agent_id": worker.agent_id}
        if action == "replace":
            args.update(model_id=WORKER, task_contract={})
        answer = payload(scheduler._manager_handler(
            self.root.agent_id, action, args, None,
        ))
        self.adapter.binding_release.set()
        starter.join(10)
        self.assertFalse(starter.is_alive())
        self.assertTrue(answer["success"], answer)
        return scheduler, worker, errors, records

    def _assert_start_was_skipped(self, scheduler, worker, errors, records) -> None:
        self.assertEqual([], errors, errors and repr(errors[0]))
        self.assertNotIn(worker.agent_id, scheduler._active_turns)
        self.assertNotIn(worker.agent_id, scheduler._starting_agents)
        # The bound thread nobody will use goes back through the release path
        # every terminal agent already uses.
        self.assertIn(worker.agent_id, scheduler._terminal_release_pending)
        skipped = [
            data for kind, agent_id, data in records
            if kind == "start_skipped_after_status_change" and agent_id == worker.agent_id
        ]
        self.assertEqual(1, len(skipped), records)
        self.assertEqual(worker.status.value, skipped[0]["status"])

    def test_a_replace_during_binding_skips_the_start_rather_than_failing_the_session(self) -> None:
        """Finding 1: the replace succeeded and then the scheduler died.

        ``control.start_turn`` refuses a turn from a replaced record, that
        ProtocolError left ``scheduler.run``, and the session was latched as
        failed after the manager had already been told the replace worked.
        """

        scheduler, worker, errors, records = self._terminal_during_binding("replace")

        self.assertIs(AgentStatus.REPLACED, worker.status)
        self._assert_start_was_skipped(scheduler, worker, errors, records)

    def test_a_cancel_during_binding_skips_the_start_rather_than_failing_the_session(self) -> None:
        scheduler, worker, errors, records = self._terminal_during_binding("cancel_agent")

        self.assertIs(AgentStatus.CANCELLED, worker.status)
        self._assert_start_was_skipped(scheduler, worker, errors, records)

    def test_a_block_during_binding_also_skips_the_start(self) -> None:
        """Blocked is the other status a start must not cross.

        A readiness probe or a binding failure on another path can block this
        agent inside the same window, and a blocked record is not a record a
        provider turn may be started from either.
        """

        records: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: records.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        scheduler._bind_agent(worker)
        self.control.block_agent(worker.agent_id, "its runtime never answered")

        scheduler._start_turn(worker)

        self.assertIs(AgentStatus.BLOCKED, worker.status)
        self.assertNotIn(worker.agent_id, scheduler._active_turns)
        self.assertEqual(
            "start_skipped_after_status_change",
            [kind for kind, agent_id, _data in records if agent_id == worker.agent_id][-1],
        )

    def _replace_inside_start(self, where: str):
        """Replace a bound READY worker from inside its own start.

        Ported from the GPT reviewer's round-11 probe.  The starter passed its
        status check, then a manager's replace landed while the prompt was
        being built (``prompt``) or while the control plane was admitting the
        turn (``admission``).  The replace was answered success, then
        ``control.start_turn`` refused the replaced record and the refusal left
        the scheduler.
        """

        records: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: records.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        scheduler._bind_agent(worker)
        answers: list[dict] = []

        def replace_now() -> None:
            answers.append(payload(scheduler._manager_handler(
                self.root.agent_id,
                "replace",
                {"agent_id": worker.agent_id, "model_id": WORKER, "task_contract": {}},
                None,
            )))

        if where == "prompt":
            build = scheduler._agent_prompt

            def prompt_then_replace(agent):
                built = build(agent)
                replace_now()
                return built

            scheduler._agent_prompt = prompt_then_replace
        else:
            admit = self.managed.start_turn

            def replace_then_admit(*args, **kwargs):
                replace_now()
                return admit(*args, **kwargs)

            self.managed.start_turn = replace_then_admit
        delivered_before = worker.delivered_message_count
        scheduler._start_turn(worker)
        self.assertEqual(1, len(answers))
        self.assertTrue(answers[0]["success"], answers[0])
        self.assertIs(AgentStatus.REPLACED, worker.status)
        self.assertEqual(delivered_before, worker.delivered_message_count)
        return scheduler, worker, [], records

    def test_a_replace_while_the_prompt_is_built_skips_the_start(self) -> None:
        self._assert_start_was_skipped(*self._replace_inside_start("prompt"))

    def test_a_replace_inside_turn_admission_skips_the_start(self) -> None:
        self._assert_start_was_skipped(*self._replace_inside_start("admission"))

    def _timed_out_turn(self, agent_id: str, turn_id: str):
        """One provider turn of this agent, identified the way vNext does."""

        runtime = types.SimpleNamespace(thread_id="worker-thread", turn_id=turn_id)
        turn = types.SimpleNamespace(
            agent_id=agent_id,
            provider="codex",
            control_turn_id="control-" + turn_id,
            runtime=runtime,
        )
        return _TurnFinished(agent_id, turn, error=TimeoutError("wait expired"))

    def test_a_late_receipt_for_an_old_turn_leaves_a_newer_stop_unconfirmed(self) -> None:
        """Finding 2: one receipt erased a hold it was never watching.

        Ported from the GPT reviewer's probe_stale_stop.py.  Turn A times out,
        the manager starts turn B on the same record with an explicit override,
        B times out too, and then A's completion finally arrives.  A's watcher
        used to drop the whole agent entry, so B's unconfirmed stop was gone and
        a plain retry put turn C in the workspace beside it with no override and
        no record of the decision.
        """

        scheduler, manager, worker = self._resumable_root_with_one_worker()
        scheduler.replacement_barrier_seconds = 3.0
        old_waiting = threading.Event()
        old_stopped = threading.Event()

        def wait_turn(turn, *, timeout=300):
            if turn.runtime.turn_id == "old":
                old_waiting.set()
                if not old_stopped.wait(10):
                    raise TimeoutError("old stop unconfirmed")
                return {"status": "completed"}
            raise TimeoutError("new turn still running")

        self.managed.wait_turn = wait_turn
        self.control.start_turn(worker.agent_id, thread_id="worker-thread")
        scheduler._handle_turn_finished(self._timed_out_turn(worker.agent_id, "old"))
        self.assertTrue(old_waiting.wait(5))

        first = payload(scheduler._manager_handler(manager.agent_id, "retry", {
            "agent_id": worker.agent_id,
            "task_contract": {},
            "start_despite_unconfirmed_stop": True,
        }, None))
        self.assertTrue(first["success"], first)
        # The real start consumes the named override, and this test starts the
        # second turn through the control plane instead.
        with scheduler._lock:
            scheduler._unconfirmed_start_overrides.discard(worker.agent_id)
        self.control.start_turn(worker.agent_id, thread_id="worker-thread")
        scheduler._handle_turn_finished(self._timed_out_turn(worker.agent_id, "new"))

        self.assertEqual(
            {"control-old", "control-new"},
            set(scheduler._stopping_turns[worker.agent_id]),
        )

        old_stopped.set()
        deadline = time.monotonic() + 5
        while (
            "control-old" in scheduler._stopping_turns.get(worker.agent_id, {})
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        # Only the turn whose receipt arrived is released.
        self.assertEqual(
            {"control-new"}, set(scheduler._stopping_turns[worker.agent_id])
        )
        third = payload(scheduler._manager_handler(manager.agent_id, "retry", {
            "agent_id": worker.agent_id, "task_contract": {},
        }, None))
        self.assertTrue(third["success"], third)
        started: list[str] = []
        scheduler._start_turn = lambda agent: started.append(agent.agent_id)
        scheduler._start_ready_agents()

        # The probe's own line was `worker.agent_id in started`.  The Root of
        # this fixture is READY too, so only the worker is asserted on.
        self.assertNotIn(worker.agent_id, started)
        self.assertEqual(worker.agent_id, scheduler._busy_predecessor(worker))

    def test_both_unconfirmed_turns_of_one_agent_are_asked_again_to_stop(self) -> None:
        """Cancel's second stop reaches every turn that never confirmed."""

        scheduler = self._scheduler()
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared file",
            task_contract={},
        )
        interrupted: list[str] = []
        self.adapter.interrupt = lambda runtime: interrupted.append(runtime.turn_id)
        for turn_id in ("old", "new"):
            event = self._timed_out_turn(worker.agent_id, turn_id)
            with scheduler._lock:
                scheduler._mark_turn_stopping(worker.agent_id, event.turn)

        scheduler._interrupt_unconfirmed_stop(worker.agent_id)

        self.assertEqual({"old", "new"}, set(interrupted))

    def test_a_replacement_waits_for_the_predecessor_turn_to_end(self) -> None:
        """Ported from the GPT reviewer's REPLACE probe.

        The probe printed `old_turn_active=True replacement_start_requested=True`:
        the replacement's first turn started while the turn it replaced was
        still running, with both agents writing the same shared workspace.
        """

        scheduler = self._scheduler()
        old, replacement, starts = self._replace_over_a_live_turn(scheduler)

        scheduler._start_ready_agents()
        self.assertNotIn(replacement.agent_id, starts)
        self.assertIn(old.agent_id, scheduler._active_turns)

        with scheduler._lock:
            scheduler._active_turns.pop(old.agent_id)
        scheduler._start_ready_agents()
        self.assertIn(replacement.agent_id, starts)

    def test_a_predecessor_that_will_not_stop_blocks_the_replacement(self) -> None:
        """Ported from the GPT reviewer's EXPIRED_BARRIER attack.

        The attack printed `replacement_started=True` with no completion from
        the old turn: past the deadline the replacement was admitted on two
        acknowledged interrupts and no evidence either had been honoured, so the
        bound on the wait reopened the two-writer fault it was added to close.
        The deadline now buys a decision and hands it to the manager.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
            ),
        )
        # The provider never answers the interrupt, so the wait has to end.
        scheduler.replacement_barrier_seconds = 0.0
        old, replacement, starts = self._replace_over_a_live_turn(scheduler)

        scheduler._start_ready_agents()

        self.assertNotIn(replacement.agent_id, starts)
        self.assertEqual(AgentStatus.BLOCKED, replacement.status)
        self.assertIn("did not confirm it stopped", replacement.blocker)
        self.assertIn(
            "Retry with start_despite_unconfirmed_stop true to start it anyway",
            replacement.blocker,
        )
        # Off the active list, because no wait of ours is on it any more, and on
        # the stopping list, because it may still be writing the workspace.
        self.assertNotIn(old.agent_id, scheduler._active_turns)
        self.assertIn(old.agent_id, scheduler._stopping_turns)
        unconfirmed = [data for kind, _id, data in events if kind == "replaced_turn_unconfirmed"]
        self.assertEqual(1, len(unconfirmed), unconfirmed)
        self.assertEqual(replacement.agent_id, unconfirmed[0]["replacement_id"])
        self.assertEqual(0.0, unconfirmed[0]["waited_seconds"])
        self.assertEqual("blocked", unconfirmed[0]["replacement_status"])
        # The replace sent one interrupt and the expiry sent the second.
        self.assertEqual(2, len(self.adapter.interrupted), self.adapter.interrupted)

    def test_a_manager_retry_starts_a_replacement_blocked_on_an_unconfirmed_stop(self) -> None:
        """Item 4: the manager was told, so the manager may take the risk.

        The risk is taken by naming the flag.  An ordinary retry is held, so
        the manager that reaches for one out of habit after any failure cannot
        start the second writer without saying it meant to.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
            ),
        )
        scheduler.replacement_barrier_seconds = 0.0
        old, replacement, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()
        self.assertEqual(AgentStatus.BLOCKED, replacement.status)

        retried = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {
                "agent_id": replacement.agent_id,
                "task_contract": {},
                "start_despite_unconfirmed_stop": True,
            },
            None,
        ))
        self.assertTrue(retried["success"], retried)
        scheduler._start_ready_agents()

        self.assertIn(replacement.agent_id, starts)
        self.assertNotIn(replacement.agent_id, scheduler._replacement_barriers)
        chose = [data for kind, _id, data in events if kind == "replacement_started_unconfirmed"]
        self.assertEqual(1, len(chose), chose)
        self.assertEqual(old.agent_id, chose[0]["predecessor_id"])
        self.assertEqual("manager retry", chose[0]["decided_by"])
        self.assertTrue(chose[0]["explicit_override"])

    def test_a_replacement_for_a_timed_out_agent_waits_for_its_stop(self) -> None:
        """Ported from the GPT reviewer's TIMEOUT_REPLACE attack.

        The attack printed `replacement_started=True barrier=False`: the timeout
        path dropped the turn from _active_turns before interrupting it, and the
        barrier only knew about turns on that list, so replacing the agent whose
        wait had expired started the second writer at once.
        """

        scheduler, _record, old = self._resumable_root_with_one_worker()
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        event = self._timed_out_turn_event(
            old.agent_id, thread_id="old-thread", message="wait timed out after 1s",
        )

        scheduler._handle_turn_finished(event)

        self.assertEqual(AgentStatus.FAILED, old.status)
        self.assertNotIn(old.agent_id, scheduler._active_turns)
        self.assertIn(old.agent_id, scheduler._stopping_turns)
        result = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": MANAGER, "task_contract": {}},
            None,
        ))
        self.assertTrue(result["success"], result)
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()

        self.assertNotIn(result["agent_id"], starts)
        self.assertIn(result["agent_id"], scheduler._replacement_barriers)

    def test_a_retry_of_a_timed_out_agent_waits_for_its_stop(self) -> None:
        scheduler, _record, old = self._resumable_root_with_one_worker()
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        scheduler._handle_turn_finished(self._timed_out_turn_event(
            old.agent_id, thread_id="old-thread", message="wait timed out after 1s",
        ))
        self.assertEqual(AgentStatus.FAILED, old.status)

        retried = payload(scheduler._manager_handler(
            self.root.agent_id, "retry", {"agent_id": old.agent_id, "task_contract": {}}, None,
        ))
        self.assertTrue(retried["success"], retried)
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()

        self.assertNotIn(old.agent_id, starts)
        self.assertIn(old.agent_id, scheduler._replacement_barriers)

    def test_a_plain_retry_after_the_old_process_exited_is_not_held(self) -> None:
        """The barrier watcher has given up; the process dies later.

        A dead process cannot still be writing, so a plain retry asks once
        more and starts the agent instead of answering unconfirmed-stop.
        """

        scheduler, _record, old = self._resumable_root_with_one_worker()
        scheduler.replacement_barrier_seconds = 0.0
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        with patch.object(self.adapter, "interrupt", side_effect=RuntimeError("refused stop")), \
                patch.object(self.managed, "wait_turn", side_effect=TimeoutError("never confirms")):
            scheduler._handle_turn_finished(self._timed_out_turn_event(
                old.agent_id, thread_id="old-thread", message="wait timed out after 1s",
            ))
            time.sleep(0.05)
            first = payload(scheduler._manager_handler(
                self.root.agent_id, "retry", {"agent_id": old.agent_id, "task_contract": {}}, None,
            ))
            self.assertTrue(first["success"], first)
            scheduler._start_ready_agents()
        self.assertEqual(AgentStatus.BLOCKED, old.status)
        self.assertNotIn(old.agent_id, starts)

        self.adapter.process_ended = True
        second = payload(scheduler._manager_handler(
            self.root.agent_id, "retry", {"agent_id": old.agent_id, "task_contract": {}}, None,
        ))
        self.assertTrue(second["success"], second)
        scheduler._start_ready_agents()
        self.assertIn(old.agent_id, starts)
        self.assertNotIn(old.agent_id, scheduler._stopping_turns)

    def test_a_confirmed_stop_lets_the_replacement_start_with_no_block(self) -> None:
        """The happy path: the old turn ends, and nothing is blocked at all."""

        scheduler, _record, old = self._resumable_root_with_one_worker()
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        event = self._timed_out_turn_event(
            old.agent_id, thread_id="old-thread", message="wait timed out after 1s",
        )
        scheduler._handle_turn_finished(event)
        result = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": MANAGER, "task_contract": {}},
            None,
        ))
        replacement = self.control.sessions[self.root.session_id].agents[result["agent_id"]]
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()
        self.assertNotIn(replacement.agent_id, starts)

        # The provider answers the interrupt: the same turn reports its end.
        scheduler._handle_turn_finished(
            _TurnFinished(old.agent_id, event.turn, result={"status": "completed"})
        )
        scheduler._start_ready_agents()

        self.assertNotIn(old.agent_id, scheduler._stopping_turns)
        self.assertIn(replacement.agent_id, starts)
        self.assertEqual(AgentStatus.READY, replacement.status)
        self.assertEqual("", replacement.blocker)
        self.assertNotIn(replacement.agent_id, scheduler._replacement_barriers)

    def test_a_plain_retry_leaves_a_blocked_replacement_blocked(self) -> None:
        """The override has to be asked for by name.

        A blind GPT review of round 6 printed `RETRY success=True
        old_stopping=True replacement_started=True`: an ordinary retry, the
        habit a manager reaches for after any failure, doubled as the decision
        to put a second writer in one workspace.  The retry now answers with
        the reason and the name of the flag that overrides it.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
            ),
        )
        scheduler.replacement_barrier_seconds = 0.0
        old, replacement, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()
        self.assertEqual(AgentStatus.BLOCKED, replacement.status)

        retried = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {"agent_id": replacement.agent_id, "task_contract": {}},
            None,
        ))
        scheduler._start_ready_agents()

        self.assertFalse(retried["success"], retried)
        self.assertEqual("unconfirmed-stop", retried["error_code"])
        self.assertIn("start_despite_unconfirmed_stop", retried["error"])
        self.assertIn(old.agent_id, retried["error"])
        self.assertNotIn(replacement.agent_id, starts)
        self.assertEqual(AgentStatus.BLOCKED, replacement.status)
        self.assertIn("start_despite_unconfirmed_stop", replacement.blocker)
        self.assertEqual(
            replacement.agent_id, scheduler._unconfirmed_blocks and
            next(iter(scheduler._unconfirmed_blocks)),
        )
        self.assertEqual(
            [], [data for kind, _id, data in events if kind == "replacement_started_unconfirmed"]
        )

    def test_a_plain_second_retry_of_a_timed_out_agent_is_refused(self) -> None:
        """The same answer on the timed-out path.

        The review's second line was `TIMEOUT_RETRY_OVERRIDE retry_success=True
        old_stopping=True started=True`: the first retry was held and blocked,
        and a manager that simply retried again started the agent while its own
        previous turn was still unaccounted for.
        """

        scheduler, _record, old = self._resumable_root_with_one_worker()
        scheduler.replacement_barrier_seconds = 0.0
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        with patch.object(self.adapter, "interrupt", side_effect=RuntimeError("refused stop")), \
                patch.object(self.managed, "wait_turn", side_effect=TimeoutError("still running")):
            scheduler._handle_turn_finished(self._timed_out_turn_event(
                old.agent_id, thread_id="old-thread", message="wait timed out after 1s",
            ))
            first = payload(scheduler._manager_handler(
                self.root.agent_id, "retry", {"agent_id": old.agent_id, "task_contract": {}}, None,
            ))
            starts: list[str] = []
            scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
            scheduler._start_ready_agents()
            self.assertTrue(first["success"], first)
            self.assertEqual(AgentStatus.BLOCKED, old.status)

            again = payload(scheduler._manager_handler(
                self.root.agent_id, "retry", {"agent_id": old.agent_id, "task_contract": {}}, None,
            ))
            scheduler._start_ready_agents()

        self.assertFalse(again["success"], again)
        self.assertEqual("unconfirmed-stop", again["error_code"])
        self.assertIn("start_despite_unconfirmed_stop", again["error"])
        self.assertNotIn(old.agent_id, starts)
        self.assertEqual(AgentStatus.BLOCKED, old.status)

    def _one_bound_worker(self, scheduler, objective="Edit the shared file"):
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective=objective,
            task_contract={},
        )
        scheduler._bind_agent(worker)
        return worker

    def _replace(self, scheduler, agent_id):
        answer = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": agent_id, "model_id": WORKER, "task_contract": {}},
            None,
        ))
        self.assertTrue(answer["success"], answer)
        return self.control.sessions[self.root.session_id].agents[answer["agent_id"]]

    def test_replacing_a_blocked_replacement_is_still_held_by_the_first_turn(self) -> None:
        """Item 1: the hold belongs to the chain, not to one pair of agents.

        A blind review printed `CHAIN_REPLACE first=blocked second=started
        old_stop=unconfirmed override=absent`: replacing the replacement asked
        only whether B had a turn, and B never ran, so C started while A was
        still on the stopping list.  The question is now asked of everything C
        came from.
        """

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old, held, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()
        self.assertEqual(AgentStatus.BLOCKED, held.status)
        self.assertIn(old.agent_id, scheduler._stopping_turns)

        third = self._replace(scheduler, held.agent_id)
        scheduler._start_ready_agents()

        self.assertNotIn(third.agent_id, starts)
        self.assertEqual(AgentStatus.BLOCKED, third.status)
        self.assertIn("start_despite_unconfirmed_stop", third.blocker)
        self.assertIn(old.agent_id, third.blocker)
        self.assertIn(old.agent_id, scheduler._stopping_turns)

    def test_a_replacement_waits_for_a_predecessor_still_inside_its_own_start(self) -> None:
        """Item 2: a turn handed to the provider that has not answered yet.

        The review printed `INFLIGHT_START_REPLACE replacement=started
        old_turn=active barrier=absent`.  Between the prompt reaching the
        provider and the native handle coming back there is no turn to find on
        any list, so a replace in that window saw an idle predecessor; the model
        was already working, and the replacement started beside it.
        """

        scheduler = self._scheduler()
        old = self._one_bound_worker(scheduler)
        replacements: list = []
        original_start = self.adapter.start_turn

        def replace_mid_start(**kwargs):
            # Exactly the window: the agent has been handed to the provider and
            # the native handle has not come back.
            self.assertIn(old.agent_id, scheduler._starting_agents)
            self.assertNotIn(old.agent_id, scheduler._active_turns)
            replacements.append(self._replace(scheduler, old.agent_id))
            return original_start(**kwargs)

        self.adapter.start_turn = replace_mid_start
        try:
            scheduler._start_turn(old)
        finally:
            self.adapter.start_turn = original_start
        replacement = replacements[0]

        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()

        self.assertNotIn(replacement.agent_id, starts)
        self.assertIn(replacement.agent_id, scheduler._replacement_barriers)
        self.assertIn(old.agent_id, scheduler._active_turns)

    def test_the_timeout_handoff_leaves_no_window_for_a_replace(self) -> None:
        """Item 3: one lock acquisition covers both lists.

        The review printed `TIMEOUT_GAP_REPLACE replacement=started
        old_stop=unconfirmed barrier=absent`.  The turn came off the active list,
        the lock was released, and only then was it put on the stopping list.  A
        replace in between read an empty chain.  The move is now atomic, and the
        provider call that follows is the only thing outside the lock.
        """

        scheduler = self._scheduler()
        old = self._one_bound_worker(scheduler)
        turn = self.managed.start_turn(
            old.agent_id, prompt="edit", effort="high", phase="test",
        )
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = turn
        window: dict = {}
        original_stop = scheduler._stop_unconfirmed_provider_turn

        def replace_in_the_window(agent, event, **kwargs):
            window["active"] = agent.agent_id in scheduler._active_turns
            window["stopping"] = agent.agent_id in scheduler._stopping_turns
            window["replacement"] = self._replace(scheduler, agent.agent_id)
            return original_stop(agent, event, **kwargs)

        scheduler._stop_unconfirmed_provider_turn = replace_in_the_window
        scheduler._handle_turn_finished(
            _TurnFinished(old.agent_id, turn, error=TimeoutError("wait timed out after 1s")),
        )
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)
        scheduler._start_ready_agents()

        self.assertFalse(window["active"])
        self.assertTrue(window["stopping"])
        replacement = window["replacement"]
        self.assertNotIn(replacement.agent_id, starts)
        self.assertIn(replacement.agent_id, scheduler._replacement_barriers)
        self.assertIn(old.agent_id, scheduler._stopping_turns)

    def test_cancelling_an_unconfirmed_stop_sends_another_one(self) -> None:
        """Item 4: a record that reads failed is not a provider that stopped.

        The review printed `CANCEL_TIMED_OUT result=already-finished
        success=true stop=unconfirmed new_interrupt=false`: the one command a
        manager has for "stop writing" answered that there was nothing to stop
        and sent nothing, for the exact case it was needed in.
        """

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old = self._one_bound_worker(scheduler)
        turn = self.managed.start_turn(
            old.agent_id, prompt="edit", effort="high", phase="test",
        )
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = turn
        with patch.object(self.managed, "wait_turn", side_effect=TimeoutError("still running")):
            scheduler._handle_turn_finished(
                _TurnFinished(old.agent_id, turn, error=TimeoutError("wait timed out after 1s")),
            )
            self.assertEqual(AgentStatus.FAILED, old.status)
            self.assertIn(old.agent_id, scheduler._stopping_turns)
            sent = len(self.adapter.interrupted)

            answer = payload(scheduler._manager_handler(
                self.root.agent_id, "cancel_agent", {"agent_id": old.agent_id}, None,
            ))

        self.assertTrue(answer["success"], answer)
        self.assertEqual("stop-requested", answer["status"])
        self.assertEqual(AgentStatus.FAILED.value, answer["agent_status"])
        self.assertIn("has not confirmed it stopped", answer["detail"])
        self.assertGreater(len(self.adapter.interrupted), sent)

        # Once the turn reports its end there is nothing left to stop, and the
        # answer goes back to the one a manager can act on.
        scheduler._handle_turn_finished(
            _TurnFinished(old.agent_id, turn, result={"status": "completed"}),
        )
        again = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": old.agent_id}, None,
        ))
        self.assertEqual("already-finished", again["status"])

    def _held_turn_beside_an_active_one(self, scheduler):
        """An agent with turn A held unconfirmed and turn B active, as an
        explicit ``start_despite_unconfirmed_stop`` retry leaves it."""

        old = self._one_bound_worker(scheduler)
        held = self.managed.start_turn(old.agent_id, prompt="edit", effort="high", phase="test")
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = held
        with patch.object(self.managed, "wait_turn", side_effect=TimeoutError("still running")):
            scheduler._handle_turn_finished(
                _TurnFinished(old.agent_id, held, error=TimeoutError("wait timed out after 1s")),
            )
        self.assertIn(old.agent_id, scheduler._stopping_turns)
        self.control.sessions[self.managed.session_id].agents[old.agent_id].status = AgentStatus.READY
        active = self.managed.start_turn(old.agent_id, prompt="edit again", effort="high", phase="test")
        with scheduler._lock:
            scheduler._active_turns[old.agent_id] = active
        return old, held, active

    def test_cancel_also_stops_a_held_turn_when_a_newer_one_is_active(self) -> None:
        """gpt-code, round 12: with turn B active, cancel asked B to stop and
        skipped turn A, whose stop was never confirmed, then answered success.
        A cancel has to reach every turn that may still be writing."""

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old, held, active = self._held_turn_beside_an_active_one(scheduler)
        asked: list = []
        self.adapter.interrupt = lambda handle: asked.append(handle)

        answer = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": old.agent_id}, None,
        ))

        self.assertTrue(answer["success"], answer)
        self.assertIn(held.runtime, asked)
        self.assertIn(active.runtime, asked)
        self.assertNotIn("uninterrupted", answer)

    def test_a_refused_stop_on_the_active_turn_still_asks_the_held_one(self) -> None:
        """Both refusals and successes are reported per turn: a refusal from the
        active turn must not skip the held one, and it must reach the answer."""

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old, held, active = self._held_turn_beside_an_active_one(scheduler)
        asked: list = []

        def interrupt(handle):
            asked.append(handle)
            if handle is active.runtime:
                raise RuntimeError("provider refused the stop")

        self.adapter.interrupt = interrupt
        answer = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": old.agent_id}, None,
        ))

        self.assertIn(held.runtime, asked)
        reasons = [entry["reason"] for entry in answer.get("uninterrupted", [])]
        self.assertTrue(any("provider refused the stop" in reason for reason in reasons), answer)

    def test_a_refused_stop_keeps_the_provider_s_private_path_out_of_the_reason(self) -> None:
        """R25: the interrupt's exception text went to the manager verbatim."""

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old, held, active = self._held_turn_beside_an_active_one(scheduler)

        def interrupt(handle):
            if handle is active.runtime:
                raise RuntimeError("provider at /home/someone/Private/SECRET/token.json")

        self.adapter.interrupt = interrupt
        answer = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": old.agent_id}, None,
        ))

        reasons = [entry["reason"] for entry in answer.get("uninterrupted", [])]
        self.assertTrue(reasons, answer)
        self.assertTrue(all("SECRET" not in reason for reason in reasons), reasons)
        self.assertTrue(any("token.json" in reason for reason in reasons), reasons)

    def test_a_hold_on_the_first_attempt_reaches_the_fourth(self) -> None:
        """The rule itself, over A <- B <- C <- D.

        Every attempt in the chain is held by the one turn that never confirmed
        it stopped, however many replacements sit between them, and the manager
        still has the one way out it has always had: name the override, and the
        choice is written into the run log.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
            ),
        )
        scheduler.replacement_barrier_seconds = 0.0
        first, second, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()
        # Each attempt is blocked by the same first turn before the manager
        # gives up on it and replaces it in turn.
        self.assertEqual(AgentStatus.BLOCKED, second.status)
        self.assertIn(first.agent_id, second.blocker)
        third = self._replace(scheduler, second.agent_id)
        scheduler._start_ready_agents()
        self.assertEqual(AgentStatus.BLOCKED, third.status)
        self.assertIn(first.agent_id, third.blocker)
        fourth = self._replace(scheduler, third.agent_id)
        scheduler._start_ready_agents()

        self.assertEqual(AgentStatus.BLOCKED, fourth.status)
        self.assertIn(first.agent_id, fourth.blocker)
        self.assertIn("start_despite_unconfirmed_stop", fourth.blocker)
        for attempt in (second, third, fourth):
            self.assertNotIn(attempt.agent_id, starts)
        self.assertEqual(
            [third.agent_id, second.agent_id, first.agent_id],
            scheduler._predecessor_chain(fourth)[1:],
        )

        forced = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {
                "agent_id": fourth.agent_id,
                "task_contract": {},
                "start_despite_unconfirmed_stop": True,
            },
            None,
        ))
        scheduler._start_ready_agents()

        self.assertTrue(forced["success"], forced)
        self.assertIn(fourth.agent_id, starts)
        self.assertNotIn(second.agent_id, starts)
        self.assertNotIn(third.agent_id, starts)
        chose = [
            data for kind, agent_id, data in events
            if kind == "replacement_started_unconfirmed" and agent_id == fourth.agent_id
        ]
        self.assertEqual(1, len(chose), chose)
        self.assertEqual(first.agent_id, chose[0]["predecessor_id"])
        self.assertTrue(chose[0]["explicit_override"])
        # The override covers that one start.  The helper stubs _start_turn, so
        # run the real one: once the turn is live, a later turn is asked again.
        self.assertIn(fourth.agent_id, scheduler._unconfirmed_start_overrides)
        VNextScheduler._start_turn(scheduler, scheduler._session().agents[fourth.agent_id])
        self.assertIn(fourth.agent_id, scheduler._active_turns)
        self.assertNotIn(fourth.agent_id, scheduler._unconfirmed_start_overrides)

    def test_the_refused_one_of_two_concurrent_replaces_keeps_the_stop_the_other_needs(self) -> None:
        """R19 claude-code F3, ported from probe_double_replace.py.

        Two replaces of one agent both pass the status check and both set the
        bit that stops a turn still inside start_turn.  The session lock lets one
        win.  The refused one used to clear the shared bit on its way out, and
        the replaced turn then published and ran on with no stop ever sent.
        """

        scheduler, _records, old = self._resumable_root_with_one_worker()
        scheduler.replacement_barrier_seconds = 30.0
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        # The state the bit exists for: inside start_turn, no handle published.
        scheduler._starting_agents.add(old.agent_id)
        scheduler._active_turns.pop(old.agent_id, None)
        gate = threading.Barrier(2)
        interrupt = scheduler._interrupt_turn

        def both_past_the_check(target_id: str) -> None:
            gate.wait(timeout=10)
            interrupt(target_id)

        scheduler._interrupt_turn = both_past_the_check
        answers: list[dict] = []

        def replace() -> None:
            answers.append(payload(scheduler._manager_handler(
                self.root.agent_id,
                "replace",
                {"agent_id": old.agent_id, "model_id": MANAGER, "task_contract": {}},
                None,
            )))

        callers = [threading.Thread(target=replace) for _ in range(2)]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(15)

        self.assertEqual([False, True], sorted(answer["success"] for answer in answers), answers)
        self.assertEqual(AgentStatus.REPLACED, old.status)
        self.assertIn(old.agent_id, scheduler._starting_agents)
        self.assertIn(old.agent_id, scheduler._interrupted_agents)

    def test_a_refused_start_spends_the_unconfirmed_stop_override(self) -> None:
        """R19 claude-code F4, ported from probe_override_leak.py.

        The override covers the one start the manager asked for.  When the
        provider refuses that start, the next automatic scan used to reuse the
        override with nobody asking and no record of its own.  The chain
        question is asked again, so the barrier holds and then blocks with the
        override named, and the manager decides a second time.
        """

        scheduler, _records, old = self._resumable_root_with_one_worker()
        scheduler.replacement_barrier_seconds = 30.0
        self.control.start_turn(old.agent_id, thread_id="old-thread")
        scheduler._starting_agents.add(old.agent_id)
        scheduler._active_turns.pop(old.agent_id, None)
        replaced = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": MANAGER, "task_contract": {}},
            None,
        ))
        replacement = self.control.sessions[self.root.session_id].agents[replaced["agent_id"]]
        self.assertFalse(scheduler._chain_clear_to_start(replacement))
        scheduler._start_despite_unconfirmed_stop(replacement)
        self.assertIn(replacement.agent_id, scheduler._unconfirmed_start_overrides)

        refused = ProtocolError("runtime-refused", "provider refused the turn")
        with patch.object(scheduler.managed, "start_turn", side_effect=refused):
            with self.assertRaises(ProtocolError):
                scheduler._start_turn(replacement)

        self.assertNotIn(replacement.agent_id, scheduler._unconfirmed_start_overrides)
        self.assertEqual(old.agent_id, scheduler._busy_predecessor(replacement))
        self.assertFalse(scheduler._chain_clear_to_start(replacement))

    def test_a_confirmed_stop_releases_a_blocked_replacement_by_itself(self) -> None:
        """Item 5: the evidence arrives late, and no manager has to act on it.

        The block is a decision taken because the predecessor said nothing
        inside the barrier.  When its turn does report its end, the reason for
        the block is gone, so the hold is dropped and the replacement starts on
        its own: a manager that already read the blocker owes nothing.
        """

        events: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
            ),
        )
        scheduler.replacement_barrier_seconds = 0.0
        old, replacement, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()
        self.assertEqual(AgentStatus.BLOCKED, replacement.status)
        old_turn = next(iter(scheduler._stopping_turns[old.agent_id].values()))

        scheduler._handle_turn_finished(
            _TurnFinished(old.agent_id, old_turn, result={"status": "completed"})
        )
        scheduler._start_ready_agents()

        self.assertNotIn(old.agent_id, scheduler._stopping_turns)
        self.assertEqual({}, scheduler._unconfirmed_blocks)
        self.assertEqual(AgentStatus.READY, replacement.status)
        self.assertEqual("", replacement.blocker)
        self.assertIn(replacement.agent_id, starts)
        released = [
            data for kind, _id, data in events
            if kind == "replacement_released_after_confirmed_stop"
        ]
        self.assertEqual(1, len(released), released)
        self.assertEqual(old.agent_id, released[0]["predecessor_id"])

    def test_the_override_flag_on_an_ordinary_retry_records_nothing(self) -> None:
        """Item 6: the flag is harmless where there is nothing to override."""

        events: list[tuple[str, str, dict]] = []
        scheduler, _record, child = self._resumable_root_with_one_worker()
        scheduler.hooks = SchedulerHooks(
            lifecycle=lambda kind, agent, data: events.append((kind, agent.agent_id, dict(data)))
        )
        self.control.start_turn(child.agent_id, thread_id="child-thread")
        self.control.block_agent(child.agent_id, "needs a decision from the manager")
        starts: list[str] = []
        scheduler._start_turn = lambda agent: starts.append(agent.agent_id)

        retried = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {
                "agent_id": child.agent_id,
                "task_contract": {},
                "start_despite_unconfirmed_stop": True,
            },
            None,
        ))
        scheduler._start_ready_agents()

        self.assertTrue(retried["success"], retried)
        self.assertIn(child.agent_id, starts)
        self.assertEqual(
            [], [data for kind, _id, data in events if kind == "replacement_started_unconfirmed"]
        )

    def test_the_retry_tool_offers_the_unconfirmed_stop_override(self) -> None:
        schema = next(
            tool for tool in VNextScheduler.manager_tools() if tool["name"] == "retry"
        )["inputSchema"]

        override = schema["properties"]["start_despite_unconfirmed_stop"]
        self.assertEqual("boolean", override["type"])
        self.assertIn("never confirmed it stopped", override["description"])
        # Optional on purpose: the safety mechanism is offered, and a manager
        # that does not name it gets the hold.
        self.assertNotIn("start_despite_unconfirmed_stop", schema["required"])

    def test_the_string_false_does_not_start_a_second_writer(self) -> None:
        """A stringified argument meant the opposite of what it said.

        The MCP layer checks that ``arguments`` is an object and matches no
        field against the schema, and ``bool("false")`` is True.  So a client
        that sent the word "false" was read as asking for the override by
        name, and got two writers in one workspace.
        """

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old, new, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()

        answer = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {
                "agent_id": new.agent_id,
                "task_contract": {},
                "start_despite_unconfirmed_stop": "false",
            },
            None,
        ))
        scheduler._start_ready_agents()

        self.assertFalse(answer["success"], answer)
        self.assertIn("start_despite_unconfirmed_stop", answer["error"])
        self.assertNotIn(new.agent_id, starts)
        self.assertIn(old.agent_id, scheduler._stopping_turns)

    def test_the_string_true_still_starts_the_replacement(self) -> None:
        """A client that stringifies its arguments means the word it sent."""

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        _old, new, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()

        answer = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {
                "agent_id": new.agent_id,
                "task_contract": {},
                "start_despite_unconfirmed_stop": "TRUE",
            },
            None,
        ))
        scheduler._start_ready_agents()

        self.assertTrue(answer["success"], answer)
        self.assertIn(new.agent_id, starts)

    def test_a_value_that_is_neither_word_is_refused_by_name(self) -> None:
        """Neither word, so neither answer: say which argument and what it takes."""

        scheduler = self._scheduler()
        scheduler.replacement_barrier_seconds = 0.0
        old, new, starts = self._replace_over_a_live_turn(scheduler)
        scheduler._start_ready_agents()

        answer = payload(scheduler._manager_handler(
            self.root.agent_id,
            "retry",
            {
                "agent_id": new.agent_id,
                "task_contract": {},
                "start_despite_unconfirmed_stop": "yes",
            },
            None,
        ))
        scheduler._start_ready_agents()

        self.assertFalse(answer["success"], answer)
        self.assertEqual("invalid-request", answer["error_code"], answer)
        self.assertIn(
            "start_despite_unconfirmed_stop takes true or false", answer["error"]
        )
        self.assertNotIn(new.agent_id, starts)
        self.assertIn(old.agent_id, scheduler._stopping_turns)

    def test_a_number_or_a_list_is_not_read_as_a_boolean(self) -> None:
        """1 is the habit a schema-less client falls into, and it is not a bool."""

        scheduler = self._scheduler()

        for value in (1, 0, ["true"], {"deep": True}):
            with self.subTest(value=value):
                answer = payload(scheduler._manager_handler(
                    self.root.agent_id, "inspect", {"agent_id": "self", "deep": value}, None
                ))

                self.assertFalse(answer["success"], answer)
                self.assertEqual("invalid-request", answer["error_code"], answer)
                self.assertIn("deep takes true or false", answer["error"])

    def test_a_non_boolean_deep_is_refused_rather_than_read_as_true(self) -> None:
        scheduler = self._scheduler()

        answer = payload(scheduler._manager_handler(
            self.root.agent_id, "inspect", {"agent_id": "self", "deep": "deep"}, None
        ))

        self.assertFalse(answer["success"], answer)
        self.assertIn("deep takes true or false", answer["error"])

    def test_the_string_false_on_verified_records_an_unverified_result(self) -> None:
        """An agent saying it did not verify may not be recorded as having done so."""

        scheduler, _record, child = self._resumable_root_with_one_worker()

        answer = payload(scheduler._manager_handler(
            child.agent_id,
            "complete_agent",
            {
                "outcome": "failed to verify",
                "verified": "false",
                "evidence": ["no tests ran"],
            },
            None,
        ))

        self.assertTrue(answer["success"], answer)
        self.assertIs(False, child.result["verified"])

    def test_a_non_boolean_verified_leaves_the_agent_working(self) -> None:
        scheduler, _record, child = self._resumable_root_with_one_worker()

        answer = payload(scheduler._manager_handler(
            child.agent_id,
            "complete_agent",
            {"outcome": "done", "verified": "probably", "evidence": ["a log"]},
            None,
        ))

        self.assertFalse(answer["success"], answer)
        self.assertIn("verified takes true or false", answer["error"])
        self.assertNotEqual(AgentStatus.COMPLETED, child.status)

    def test_a_single_evidence_string_is_one_item(self) -> None:
        scheduler, _record, child = self._resumable_root_with_one_worker()

        answer = payload(scheduler._manager_handler(
            child.agent_id,
            "complete_agent",
            {"outcome": "done", "verified": True, "evidence": "tests pass"},
            None,
        ))

        self.assertTrue(answer["success"], answer)
        self.assertEqual(["tests pass"], child.result["evidence"])

    def test_evidence_that_is_not_text_is_refused_by_name(self) -> None:
        scheduler, _record, child = self._resumable_root_with_one_worker()
        for tool in ("complete_agent", "complete_branch"):
            for evidence in (7, {"log": "x"}, ["a log", 3]):
                with self.subTest(tool=tool, evidence=evidence):
                    answer = payload(scheduler._manager_handler(
                        child.agent_id,
                        tool,
                        {"outcome": "done", "verified": True, "evidence": evidence},
                        None,
                    ))
                    self.assertFalse(answer["success"], answer)
                    self.assertEqual("invalid-request", answer["error_code"])
                    self.assertIn("evidence takes", answer["error"])
                    self.assertNotEqual(AgentStatus.COMPLETED, child.status)

    def test_delegate_reply_keeps_the_alias_and_names_the_exact_model(self) -> None:
        """model_id stays the selector; model_exact is the id it resolves to."""

        from vnext import vnext_model_identity

        vnext_model_identity.clear_cache()
        self.addCleanup(vnext_model_identity.clear_cache)
        scheduler = self._scheduler()
        arguments = {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER,
            "objective": "bounded work",
            "task_contract": {"criteria": ["work done"]},
        }
        first = payload(scheduler._manager_handler(self.root.agent_id, "delegate", dict(arguments), None))
        self.assertTrue(first["success"], first)
        self.assertEqual(WORKER, first["model_id"])
        self.assertEqual(vnext_model_identity.PENDING, first["model_exact"])
        provider = self.managed.control.registry.cards[WORKER].provider
        vnext_model_identity.remember(provider, WORKER, "worker-model-2026-10-01", "model_list")
        second = payload(scheduler._manager_handler(self.root.agent_id, "delegate", dict(arguments), None))
        self.assertEqual(WORKER, second["model_id"])
        self.assertEqual("worker-model-2026-10-01", second["model_exact"])

    def test_inspect_carries_the_exact_model_and_what_ran(self) -> None:
        child = self.managed.spawn_from_manager(
            requester_id=self.root.agent_id,
            role=AgentRole.WORKER,
            arguments={"model_id": WORKER, "objective": "x", "task_contract": {"criteria": ["x"]}},
        )
        child.model_identity = {
            "model_exact": "worker-model-2026-10-01",
            "model_exact_source": "model_list",
            "model_ran": "worker-model-2026-10-01",
        }
        view = self.managed.control.inspect_agent(self.root.agent_id, child.agent_id)
        self.assertEqual(WORKER, view["model_id"])
        self.assertEqual("worker-model-2026-10-01", view["model_exact"])
        self.assertEqual("worker-model-2026-10-01", view["model_ran"])
        self.assertFalse(view["model_mismatch"])

    def test_a_completion_mid_turn_records_the_model_that_answered(self) -> None:
        """The outcome row is written at complete_agent, before the turn ends."""

        scheduler, worker, _turn = self._started_worker_for_completion()
        identity = {
            "model_exact": "worker-model-2026-10-01",
            "model_exact_source": "server_info",
            "model_ran": "worker-model-2026-10-01",
        }
        with self.managed._lock:
            adapter = self.managed._adapters[worker.agent_id]
        with patch.object(type(adapter), "model_identity", lambda _self, _thread: identity, create=True):
            completed = payload(scheduler._manager_handler(worker.agent_id, "complete_agent", {
                "outcome": "done", "verified": True, "evidence": ["x"],
            }, None))
        self.assertTrue(completed["success"], completed)
        view = self.managed.control.inspect_agent(self.root.agent_id, worker.agent_id)
        self.assertEqual("completed", view["status"])
        self.assertEqual("worker-model-2026-10-01", view["model_exact"])
        self.assertEqual("worker-model-2026-10-01", view["model_ran"])
        self.assertFalse(view["model_mismatch"])

    def test_delegate_names_the_session_workspace_for_a_shared_child(self) -> None:
        """An empty string is not a directory a manager can open or name."""

        scheduler = self._scheduler()

        answer = payload(scheduler._manager_handler(
            self.root.agent_id,
            "delegate",
            {
                "role": AgentRole.WORKER.value,
                "model_id": WORKER,
                "objective": "bounded work in the project itself",
                "task_contract": {"criteria": ["work done"]},
            },
            None,
        ))

        self.assertTrue(answer["success"], answer)
        self.assertEqual(str(self.managed.workspace), answer["workspace_path"])
        self.assertTrue(Path(answer["workspace_path"]).is_absolute())

    def test_delegate_still_reports_a_private_child_relative_to_the_session(self) -> None:
        scheduler = self._scheduler()

        answer = payload(scheduler._manager_handler(
            self.root.agent_id,
            "delegate",
            {
                "role": AgentRole.WORKER.value,
                "model_id": WORKER,
                "objective": "bounded work in a copy of its own",
                "task_contract": {"criteria": ["work done"]},
                "workspace": "private",
            },
            None,
        ))

        self.assertTrue(answer["success"], answer)
        self.assertTrue(answer["workspace_path"])
        self.assertFalse(Path(answer["workspace_path"]).is_absolute())

    def test_replace_refuses_manager_with_live_worker_before_session_can_complete(self) -> None:
        scheduler = self._scheduler()
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own one coherent branch",
            task_contract={},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared workspace",
            task_contract={},
        )
        self.control.start_turn(branch.agent_id)
        self.control.start_turn(worker.agent_id)
        before_count = len(self.control.sessions[self.root.session_id].agents)

        refused = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": branch.agent_id, "model_id": MANAGER, "task_contract": {}},
            None,
        ))

        self.assertFalse(refused["success"], refused)
        self.assertEqual("active-children", refused["error_code"])
        self.assertIn(worker.agent_id, refused["error"])
        self.assertEqual(AgentStatus.RUNNING, branch.status)
        self.assertEqual(AgentStatus.RUNNING, worker.status)
        self.assertEqual(before_count, len(self.control.sessions[self.root.session_id].agents))
        self.control.start_turn(self.root.agent_id)
        completed = payload(scheduler._manager_handler(
            self.root.agent_id,
            "complete_session",
            {"decision": "accepted", "summary": "all done", "criteria": {}},
            None,
        ))
        self.assertFalse(completed["success"], completed)
        self.assertEqual("active-children", completed["error_code"])
        self.assertEqual(AgentStatus.RUNNING, self.root.status)

    def test_replace_interrupts_turn_started_before_active_publication(self) -> None:
        scheduler = self._scheduler()
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Edit the shared workspace",
            task_contract={},
        )
        self.assertTrue(scheduler._bind_agent(worker))
        start_turn = self.adapter.start_turn
        replacement = {}

        def replace_during_start(**kwargs):
            self.assertEqual(AgentStatus.RUNNING, worker.status)
            self.assertNotIn(worker.agent_id, scheduler._active_turns)
            replacement.update(payload(scheduler._manager_handler(
                self.root.agent_id,
                "replace",
                {"agent_id": worker.agent_id, "model_id": WORKER, "task_contract": {}},
                None,
            )))
            return start_turn(**kwargs)

        with patch.object(self.adapter, "start_turn", side_effect=replace_during_start), \
                patch.object(scheduler, "_watch_turn"):
            scheduler._start_turn(worker)

        self.assertTrue(replacement["success"], replacement)
        self.assertEqual(AgentStatus.REPLACED, worker.status)
        self.assertIn(self.managed._threads[worker.agent_id], self.adapter.interrupted)
        # The starter read the bit and the agent is terminal, so nothing is
        # left to publish under interrupted_agent_ids.
        self.assertNotIn(worker.agent_id, scheduler._interrupted_agents)

    def test_replace_accepts_a_failed_worker_and_still_refuses_the_other_ends(self) -> None:
        """A provider refusal is cured by another model, so replace must accept it.

        A worker whose turn died on "this model requires a newer Codex" is
        FAILED, and swapping the model is the whole remedy. Refusing replace
        there left a manager with retry on the same model that had just been
        refused.
        """

        scheduler = self._scheduler()
        arguments = {"model_id": WORKER, "task_contract": {"criteria": ["done"]}}

        def worker(objective: str):
            child = self.control.spawn_agent(
                requester_id=self.root.agent_id,
                parent_agent_id=self.root.agent_id,
                role=AgentRole.WORKER,
                model_id=WORKER,
                objective=objective,
                task_contract={},
            )
            self.control.start_turn(child.agent_id)
            return child

        failed = worker("the model the provider refused")
        self.control.fail_agent(failed.agent_id, "HTTP 400: model requires a newer Codex")
        self.assertEqual(AgentStatus.FAILED, failed.status)

        accepted = payload(scheduler._manager_handler(
            self.root.agent_id, "replace", {**arguments, "agent_id": failed.agent_id}, None,
        ))

        self.assertTrue(accepted["success"], accepted)
        self.assertEqual(AgentStatus.REPLACED, failed.status)
        self.assertEqual(accepted["agent_id"], failed.replaced_by_agent_id)

        # The other terminal ends stay refused: there is nothing left to steer.
        cancelled = worker("the one the manager stopped")
        scheduler.cancel_agent(cancelled.agent_id)
        self.assertEqual(AgentStatus.CANCELLED, cancelled.status)
        refused_cancelled = payload(scheduler._manager_handler(
            self.root.agent_id, "replace", {**arguments, "agent_id": cancelled.agent_id}, None,
        ))
        self.assertFalse(refused_cancelled["success"], refused_cancelled)
        self.assertEqual("terminal-agent", refused_cancelled["error_code"])

        done = worker("the one that finished")
        self.control.complete_agent(done.agent_id, {"outcome": "done"})
        refused_done = payload(scheduler._manager_handler(
            self.root.agent_id, "replace", {**arguments, "agent_id": done.agent_id}, None,
        ))
        self.assertFalse(refused_done["success"], refused_done)
        self.assertEqual("terminal-agent", refused_done["error_code"])

        # And a replaced agent cannot be replaced a second time.
        refused_twice = payload(scheduler._manager_handler(
            self.root.agent_id, "replace", {**arguments, "agent_id": failed.agent_id}, None,
        ))
        self.assertFalse(refused_twice["success"], refused_twice)
        self.assertEqual("terminal-agent", refused_twice["error_code"])

    def test_a_successful_replace_drops_the_old_id_from_the_interrupted_set(self) -> None:
        """The pause bit outlived the agent and kept being published.

        Replacing a RUNNING child sets the bit so a turn already inside
        start_turn cannot slip past the replacement. Once the old agent is
        terminal nothing will ever consume it, so it sat in
        interrupted_agent_ids for the rest of the session.
        """

        scheduler = self._scheduler()
        old = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Work in the shared workspace",
            task_contract={},
        )
        self.control.start_turn(old.agent_id)
        self.assertEqual(AgentStatus.RUNNING, old.status)

        replaced = payload(scheduler._manager_handler(
            self.root.agent_id,
            "replace",
            {"agent_id": old.agent_id, "model_id": WORKER, "task_contract": {}},
            None,
        ))

        self.assertTrue(replaced["success"], replaced)
        self.assertEqual(AgentStatus.REPLACED, old.status)
        self.assertNotIn(old.agent_id, scheduler._interrupted_agents)

    def test_replace_after_the_pause_bit_was_read_leaves_no_stale_interrupted_id(self) -> None:
        """The starter had one chance to consume the bit and replace arrived after it.

        _start_turn publishes the native handle and reads the pause bit in the
        same locked step, then clears the starting mark. A replace landing in
        between sets the bit -- the agent is still marked as starting, so
        replace keeps it -- and the starter has already read it for the last
        time. The old id stayed published under interrupted_agent_ids for the
        rest of the session.

        The interleaving is forced rather than raced: the instrumented set
        records the starter's read, and the lock wrapper runs the replace at the
        exact release that follows it.
        """

        scheduler = self._scheduler()
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Work in the shared workspace",
            task_contract={},
        )
        self.assertTrue(scheduler._bind_agent(worker))
        replacement = {}
        real_lock = scheduler._lock
        self_root = self.root.agent_id

        class ReadRecordingSet(set):
            """Note the starter's read of this worker's pause bit."""

            pending_replace = False

            def __contains__(self, item):
                if (
                    item == worker.agent_id
                    and item in scheduler._active_turns
                    and item in scheduler._starting_agents
                ):
                    self.pending_replace = True
                return super().__contains__(item)

        pause_bits = ReadRecordingSet(scheduler._interrupted_agents)
        scheduler._interrupted_agents = pause_bits

        class ReplaceAtPublication:
            """Run the replace at the release of the publication lock."""

            def __enter__(self):
                real_lock.acquire()
                return self

            def __exit__(self, exc_type, exc, tb):
                real_lock.release()
                if pause_bits.pending_replace:
                    pause_bits.pending_replace = False
                    scheduler._lock = real_lock
                    replacement.update(payload(scheduler._manager_handler(
                        self_root,
                        "replace",
                        {"agent_id": worker.agent_id, "model_id": WORKER, "task_contract": {}},
                        None,
                    )))
                return False

        scheduler._lock = ReplaceAtPublication()
        try:
            with patch.object(scheduler, "_watch_turn"):
                scheduler._start_turn(worker)
        finally:
            scheduler._lock = real_lock

        self.assertTrue(replacement["success"], replacement)
        self.assertEqual(AgentStatus.REPLACED, worker.status)
        self.assertNotIn(worker.agent_id, scheduler._starting_agents)
        self.assertNotIn(worker.agent_id, scheduler._interrupted_agents)

    def test_a_child_that_moved_mid_turn_is_not_reported_as_stopped(self) -> None:
        """The two refusals are different and were told apart by guessing.

        A manager's status is identical whichever way a park is declined, so
        inferring the reason from it told a manager waiting on one blocked child
        and one running one that everything had stopped and not to wait again --
        when waiting for the running sibling was exactly right.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        stopped = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="the one that stopped",
            task_contract={},
        )
        running = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="the one still going",
            task_contract={},
        )
        self.control.start_turn(running.agent_id)
        self.control.start_turn(self.root.agent_id)
        # It blocks during the manager's own turn, so the wake is still pending.
        self.control.block_agent(stopped.agent_id, "needs a decision")

        result = scheduler._apply_manager_tool(
            self.root.agent_id,
            "await_children",
            {"agent_ids": [stopped.agent_id, running.agent_id]},
        )

        read = payload(result)
        self.assertTrue(read["success"])
        self.assertNotIn("children_needing_a_decision", read)
        self.assertIn("moved while this turn was still open", read["note"])

    def test_awaiting_a_running_child_still_parks_and_reports_it(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        self.control.start_turn(self.root.agent_id)

        result = scheduler._apply_manager_tool(
            self.root.agent_id, "await_children", {"agent_ids": [child.agent_id]}
        )

        read = payload(result)
        self.assertTrue(read["success"])
        self.assertEqual([child.agent_id], read["awaiting"])
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]
        self.assertEqual(AgentStatus.AWAITING_WORKERS, record.status)

    def test_an_argument_await_children_does_not_have_is_refused_by_name(self) -> None:
        """The schema and the validation used to disagree.

        await_children advertises additionalProperties false, and the handler
        read agent_ids and ignored the rest, so a client that sent
        timeout_seconds was answered as though the wait had honoured it.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        self.control.start_turn(self.root.agent_id)

        await_tool = next(
            tool for tool in scheduler.manager_tools() if tool["name"] == "await_children"
        )
        self.assertFalse(await_tool["inputSchema"]["additionalProperties"])

        with self.assertRaises(ValueError) as refused:
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "await_children",
                {
                    "agent_ids": [child.agent_id],
                    "agent_id": self.root.agent_id,
                    "timeout_seconds": 60,
                },
            )

        said = str(refused.exception)
        self.assertIn("agent_id", said)
        self.assertIn("timeout_seconds", said)

    def test_the_await_description_says_a_blocked_child_is_not_waited_for(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

        await_tool = next(
            tool for tool in scheduler.manager_tools() if tool["name"] == "await_children"
        )

        self.assertIn("blocked child", await_tool["description"])
        self.assertIn("decision from you", await_tool["description"])

    def test_prestart_cancellation_terminates_without_binding_a_runtime(self) -> None:
        cancellation = RunCancellation()
        cancellation.request()
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=cancellation,
        )

        with self.assertRaises(SchedulerCancelled):
            scheduler.run()

        self.assertEqual(AgentStatus.CANCELLED, self.root.status)
        self.assertEqual({}, self.adapter.roles)

    def test_cancel_agent_interrupts_each_active_descendant_but_not_a_sibling(self) -> None:
        """A subtree cancel must stop native work before control marks it terminal."""

        scheduler = self._scheduler()
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own a subtree",
            task_contract={},
        )
        grandchild = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="active descendant",
            task_contract={},
        )
        sibling = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="unrelated active sibling",
            task_contract={},
        )
        turns = {}
        for record in (branch, grandchild, sibling):
            scheduler._bind_agent(record)
            turn = self.managed.start_turn(
                record.agent_id,
                prompt="active test turn",
                effort="high",
                phase="test",
            )
            scheduler._active_turns[record.agent_id] = turn
            turns[record.agent_id] = turn

        result = scheduler.cancel_agent(branch.agent_id)

        self.assertEqual("cancelled", result["status"])
        self.assertEqual(AgentStatus.CANCELLED, branch.status)
        self.assertEqual(AgentStatus.CANCELLED, grandchild.status)
        self.assertEqual(AgentStatus.RUNNING, sibling.status)
        self.assertCountEqual(
            [turns[branch.agent_id].runtime.thread_id, turns[grandchild.agent_id].runtime.thread_id],
            self.adapter.interrupted,
        )
        self.assertNotIn(turns[sibling.agent_id].runtime.thread_id, self.adapter.interrupted)

    def test_cancel_agent_publishes_every_descendant_it_ends(self) -> None:
        """On 2026-09-23 a live session's status file showed 10 agents still working.

        Each had a cancelled ancestor.  The control plane had ended them, but
        only the cancelled target was announced, so their rows kept saying
        blocked or ready.
        """

        announced: list[tuple[str, str, str]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, _data: announced.append(
                    (kind, agent.agent_id, agent.status.value)
                )
            ),
        )
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own a subtree",
            task_contract={},
        )
        blocked = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="interrupted worker",
            task_contract={},
        )
        done = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="finished worker",
            task_contract={},
        )
        blocked.status = AgentStatus.BLOCKED
        done.status = AgentStatus.COMPLETED

        scheduler.cancel_agent(branch.agent_id)

        self.assertIn(("agent_terminal", blocked.agent_id, "cancelled"), announced)
        self.assertNotIn(done.agent_id, [agent_id for _kind, agent_id, _status in announced])
        self.assertEqual(
            [("command_acknowledged", branch.agent_id, "cancelled")],
            [entry for entry in announced if entry[1] == branch.agent_id],
        )

    def test_cancel_agent_reports_interrupt_failure_but_cancels_the_subtree(self) -> None:
        """A provider that refuses to stop must not hide whether anything was cancelled.

        The claude adapter waits fifteen seconds for a reply and then raises,
        and the raise travelled out of cancel_agent.  The manager saw a failed
        command and could not tell that the subtree was still running.
        """

        scheduler = self._scheduler()
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own a subtree",
            task_contract={},
        )
        grandchild = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="interrupt may fail",
            task_contract={},
        )

        def interrupt(target_id: str) -> None:
            if target_id == grandchild.agent_id:
                raise RuntimeError("provider did not answer interrupt")

        with patch.object(scheduler, "_interrupt_turn", side_effect=interrupt):
            result = scheduler.cancel_agent(branch.agent_id)

        self.assertEqual({"status": "cancelled", "agent_id": branch.agent_id,
                          "uninterrupted": [{"agent_id": grandchild.agent_id,
                                              "reason": "RuntimeError: provider did not answer interrupt"}]}, result)
        self.assertEqual(AgentStatus.CANCELLED, branch.status)
        self.assertEqual(AgentStatus.CANCELLED, grandchild.status)

    def test_cancel_agent_omits_uninterrupted_when_all_interrupts_succeed(self) -> None:
        """The ordinary answer keeps the shape every caller already reads."""

        scheduler = self._scheduler()
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own a subtree",
            task_contract={},
        )
        grandchild = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="interrupt succeeds",
            task_contract={},
        )

        result = scheduler.cancel_agent(branch.agent_id)

        self.assertEqual({"status": "cancelled", "agent_id": branch.agent_id}, result)
        self.assertEqual(AgentStatus.CANCELLED, branch.status)
        self.assertEqual(AgentStatus.CANCELLED, grandchild.status)

    def test_a_silent_failure_still_names_itself(self) -> None:
        """A timeout carries no message, so the reason falls back to its class."""

        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="interrupt times out",
            task_contract={},
        )

        with patch.object(scheduler, "_interrupt_turn", side_effect=TimeoutError()):
            result = scheduler.cancel_agent(child.agent_id)

        self.assertEqual([{"agent_id": child.agent_id, "reason": "TimeoutError"}],
                         result["uninterrupted"])
        self.assertEqual(AgentStatus.CANCELLED, child.status)

    def test_dynamic_cancel_fails_the_call_when_a_stop_did_not_happen(self) -> None:
        """A refused stop leaves a turn running, so the call must not read success.

        The answer already carried an `uninterrupted` list, and it was handed
        back with success=True anyway.  A manager reads the flag first, so it
        saw a clean cancellation while the worker's turn carried on.
        """

        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="the provider refuses to stop this one",
            task_contract={},
        )

        with patch.object(scheduler, "_interrupt_turn", side_effect=TimeoutError()):
            answer = payload(
                scheduler._apply_manager_tool(
                    self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}
                )
            )

        self.assertFalse(answer["success"])
        self.assertEqual(
            [{"agent_id": child.agent_id, "reason": "TimeoutError"}],
            answer["uninterrupted"],
        )
        # The subtree really is cancelled: the failure is about the turn that
        # did not stop, and the record has to say what the control plane did.
        self.assertEqual("cancelled", answer["status"])
        self.assertEqual(AgentStatus.CANCELLED, child.status)

    def test_dynamic_cancel_succeeds_when_every_stop_landed(self) -> None:
        """The ordinary cancellation keeps reading as success."""

        scheduler = self._scheduler()
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="stops when asked",
            task_contract={},
        )

        answer = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}
            )
        )

        self.assertTrue(answer["success"])
        self.assertNotIn("uninterrupted", answer)

    def _announcing_scheduler(self, announced: list) -> VNextScheduler:
        """A scheduler whose lifecycle announcements the test can read back."""

        return VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: announced.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )

    def test_cancelling_a_finished_subtree_says_it_already_finished(self) -> None:
        """Every other control refuses a terminal agent; this one claimed success.

        A manager that had lost track of a completed child was answered
        {"status": "cancelled"} and a cancel went into the run record, so the
        record showed a cancellation of work that had finished and the manager
        could not tell the two cases apart.
        """

        announced: list = []
        scheduler = self._announcing_scheduler(announced)
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="finishes before anybody cancels it",
            task_contract={},
        )
        self.control.complete_agent(child.agent_id, {"outcome": "done"})

        result = scheduler.cancel_agent(child.agent_id)

        self.assertEqual(
            {
                "status": "already-finished",
                "agent_id": child.agent_id,
                "agent_status": "completed",
            },
            result,
        )
        self.assertEqual(AgentStatus.COMPLETED, child.status)
        self.assertEqual([], [entry for entry in announced if entry[0] == "command_acknowledged"])

        # The in-tree manager tool path answers the same way.
        answer = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}
            )
        )
        self.assertTrue(answer["success"])
        self.assertEqual("already-finished", answer["status"])
        self.assertEqual("completed", answer["agent_status"])
        self.assertEqual([], [entry for entry in announced if entry[0] == "command_acknowledged"])

    def test_cancelling_a_finished_manager_stops_the_work_under_it(self) -> None:
        """A terminal target with live descendants is still a real cancellation."""

        announced: list = []
        scheduler = self._announcing_scheduler(announced)
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own a subtree",
            task_contract={},
        )
        grandchild = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="still working",
            task_contract={},
        )
        branch.status = AgentStatus.COMPLETED

        result = scheduler.cancel_agent(branch.agent_id)

        self.assertEqual("cancelled", result["status"])
        self.assertEqual("completed", result["agent_status"])
        self.assertEqual([grandchild.agent_id], result["cancelled_descendants"])
        self.assertEqual(AgentStatus.CANCELLED, grandchild.status)
        self.assertEqual(
            [("command_acknowledged", branch.agent_id)],
            [
                (entry[0], entry[1])
                for entry in announced
                if entry[0] == "command_acknowledged"
            ],
        )

    def test_dynamic_cancel_tool_interrupts_an_active_grandchild(self) -> None:
        """The model-facing path has the same native cancellation guarantee."""

        scheduler = self._scheduler()
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="owner",
            task_contract={},
        )
        grandchild = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="active nested work",
            task_contract={},
        )
        for record in (branch, grandchild):
            scheduler._bind_agent(record)
            turn = self.managed.start_turn(
                record.agent_id, prompt="active", effort="high", phase="test"
            )
            scheduler._active_turns[record.agent_id] = turn

        cancelled = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id, "cancel_agent", {"agent_id": branch.agent_id}
            )
        )

        self.assertTrue(cancelled["success"])
        self.assertEqual(AgentStatus.CANCELLED, branch.status)
        self.assertEqual(AgentStatus.CANCELLED, grandchild.status)
        self.assertCountEqual(
            [
                self.managed._threads[branch.agent_id],
                self.managed._threads[grandchild.agent_id],
            ],
            self.adapter.interrupted,
        )

    def test_worker_turn_without_completion_stays_live_for_a_peer_follow_up(self) -> None:
        """A descriptive Worker role cannot silently discard a continuing task."""

        scheduler = self._scheduler()
        worker = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="wait for review",
            task_contract={},
        )
        scheduler._bind_agent(worker)
        turn = self.managed.start_turn(
            worker.agent_id, prompt="first", effort="high", phase="test"
        )
        scheduler._active_turns[worker.agent_id] = turn

        scheduler._handle_turn_finished(
            _TurnFinished(worker.agent_id, turn, result={"status": "completed"})
        )

        self.assertEqual(AgentStatus.READY, worker.status)
        self.assertNotEqual(AgentStatus.COMPLETED, worker.status)
        self.assertIn(worker.agent_id, scheduler._awaiting_message_agents)
        scheduler.send_message(
            sender_id=self.root.agent_id,
            target_id=worker.agent_id,
            text="continue after the review",
        )
        self.assertNotIn(worker.agent_id, scheduler._awaiting_message_agents)
        self.assertEqual(["continue after the review"], [item.text for item in worker.messages])

    def test_terminal_lifecycle_keeps_control_turn_after_completion(self) -> None:
        lifecycle: list[tuple[str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=lambda kind, _agent, data: lifecycle.append((kind, dict(data)))),
        )
        scheduler._bind_agent(self.root)
        turn = self.managed.start_turn(
            self.root.agent_id, prompt="complete", effort="high", phase="test"
        )
        scheduler._active_turns[self.root.agent_id] = turn

        completed = payload(scheduler._apply_manager_tool(
            self.root.agent_id,
            "complete_session",
            {"decision": "accepted", "summary": "done", "criteria": {}},
        ))
        self.assertTrue(completed["success"])
        scheduler._handle_turn_finished(
            _TurnFinished(self.root.agent_id, turn, result={"status": "completed"})
        )

        terminal = next(data for kind, data in lifecycle if kind == "agent_terminal")
        finished = next(data for kind, data in lifecycle if kind == "turn_completed")
        self.assertEqual(turn.control_turn_id, terminal["turn_id"])
        self.assertEqual(turn.control_turn_id, finished["turn_id"])

    def test_interrupting_a_ready_agent_pauses_it_until_a_message_resumes_it(self) -> None:
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="wait for a user decision",
            task_contract={},
        )
        scheduler = self._scheduler()

        paused = scheduler.interrupt_agent(child.agent_id)
        self.assertEqual("paused", paused["status"])
        self.assertIn(child.agent_id, scheduler._interrupted_agents)
        resumed = scheduler.send_message(
            sender_id=self.root.agent_id,
            target_id=child.agent_id,
            text="continue with the revised objective",
        )

        self.assertEqual("queued", resumed["status"])
        self.assertNotIn(child.agent_id, scheduler._interrupted_agents)

    def test_adopted_native_turn_is_active_interruptible_and_not_duplicated(self) -> None:
        adapter = HeldTurnAdapter()
        self.adapter = adapter
        self.managed.adapter = adapter
        self.managed._adapter_selector = lambda _agent: adapter
        lifecycle = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=lambda kind, _agent, data: lifecycle.append((kind, dict(data)))),
        )
        self.assertTrue(scheduler._bind_agent(self.root))
        thread_id = self.managed._threads[self.root.agent_id]
        scheduler.acquire_native_control_lease(self.root.agent_id)

        adopted = scheduler.adopt_native_turn(
            agent_id=self.root.agent_id,
            provider="fixture",
            thread_id=thread_id,
            turn_id="native-ui-turn",
            cursor=0,
        )
        repeated = scheduler.adopt_native_turn(
            agent_id=self.root.agent_id,
            provider="fixture",
            thread_id=thread_id,
            turn_id="native-ui-turn",
            cursor=0,
        )

        self.assertEqual(adopted.control_turn_id, repeated.control_turn_id)
        self.assertEqual(adopted, scheduler._active_turns[self.root.agent_id])
        self.assertEqual(1, len([kind for kind, _data in lifecycle if kind == "turn_started"]))
        self.assertTrue(adapter.waiting.wait(1))
        self.assertEqual("interrupt-requested", scheduler.interrupt_agent(self.root.agent_id)["status"])
        self.assertEqual([thread_id], adapter.interrupted)
        adapter.release.set()

    def test_terminal_primary_reads_and_acknowledges_reply_before_completing(self) -> None:
        scheduler = self._scheduler()
        scheduler._watch_turn = lambda _turn: None
        scheduler.acquire_native_control_lease(self.root.agent_id)
        thread = self.managed._threads[self.root.agent_id]
        context = ToolCallContext(thread_id=thread, turn_id="terminal-root-turn")
        scheduler.adopt_native_turn(agent_id=self.root.agent_id, provider="fixture",
                                    thread_id=thread, turn_id=context.turn_id, cursor=0)
        self.control.send_user_message(self.root.agent_id, "reply received during terminal turn")
        first = scheduler._manager_handler(self.root.agent_id, "inspect", {"agent_id": "self"}, context)
        self.assertTrue(first.success)
        self.assertEqual(["reply received during terminal turn"], [m["text"] for m in first.value.get("messages", [])])
        self.assertEqual(0, self.root.delivered_message_count)
        blocked = scheduler._manager_handler(self.root.agent_id, "complete_session", {
            "decision": "accepted", "summary": "done", "criteria": {}}, context)
        self.assertFalse(blocked.success)
        self.assertEqual("unread-messages", blocked.value["error_code"])
        self.assertIn("acknowledge_messages_through", blocked.value["error"])
        acknowledged = scheduler._manager_handler(self.root.agent_id, "inspect", {
            "agent_id": "self", "acknowledge_messages_through": first.value["message_cursor"]}, context)
        self.assertTrue(acknowledged.success)
        done = scheduler._manager_handler(self.root.agent_id, "complete_session", {
            "decision": "accepted", "summary": "done", "criteria": {}}, context)
        self.assertTrue(done.success, done.value)
        self.assertEqual(AgentStatus.COMPLETED, self.root.status)
        scheduler.release_native_control_lease(self.root.agent_id)
        self.assertIsNone(scheduler._native_message_delivery(self.root.agent_id))

    def test_compact_agent_holds_the_lease_while_the_adapter_compacts(self) -> None:
        scheduler = self._scheduler()
        calls = []

        def compact(thread_id: str, *, timeout_seconds: float) -> dict:
            calls.append((thread_id, self.root.agent_id in scheduler._native_control_leases))
            self.assertEqual(scheduler.turn_timeout, timeout_seconds)
            return {"compacted": True, "pre_tokens": 10, "post_tokens": 2}

        self.managed._adapter_for(self.root.agent_id).compact = compact
        result = scheduler.compact_agent(self.root.agent_id)
        self.assertEqual({"compacted": True, "pre_tokens": 10, "post_tokens": 2, "agent_id": self.root.agent_id}, result)
        self.assertEqual([(self.managed._threads[self.root.agent_id], True)], calls)
        self.assertNotIn(self.root.agent_id, scheduler._native_control_leases)

    def test_rejected_compact_keeps_a_native_control_lease_it_did_not_acquire(self) -> None:
        # A native terminal already owns the primary.  The adapter refuses to
        # compact under terminal ownership, and that refusal must not drop the
        # terminal's lease, or its next native turn has no lease to run under.
        scheduler = self._scheduler()
        scheduler._watch_turn = lambda _turn: None
        scheduler.acquire_native_control_lease(self.root.agent_id)

        def compact(_thread_id: str, **_kwargs: object) -> dict:
            raise ClaudeRuntimeError("Claude thread is owned by a native terminal")

        self.managed._adapter_for(self.root.agent_id).compact = compact
        with self.assertRaisesRegex(ClaudeRuntimeError, "native terminal"):
            scheduler.compact_agent(self.root.agent_id)

        self.assertIn(self.root.agent_id, scheduler._native_control_leases)
        thread = self.managed._threads[self.root.agent_id]
        scheduler.adopt_native_turn(agent_id=self.root.agent_id, provider="fixture",
                                    thread_id=thread, turn_id="terminal-after-compact", cursor=0)
        self.assertIn(self.root.agent_id, scheduler._active_turns)

    def test_native_control_lease_prevents_a_competing_scheduler_turn(self) -> None:
        scheduler = self._scheduler()
        scheduler.acquire_native_control_lease(self.root.agent_id)

        scheduler._start_ready_agents()

        self.assertEqual(1, self.adapter._next_thread)
        self.assertEqual(0, self.adapter._next_turn)
        scheduler.release_native_control_lease(self.root.agent_id)

    def test_claude_reader_deferral_keeps_queued_message_until_the_same_bound_thread_is_ready(self) -> None:
        adapter = DeferredReadinessAdapter()
        self.adapter = adapter
        self.managed.adapter = adapter
        self.managed._adapter_selector = lambda _agent: adapter
        scheduler = self._scheduler()
        scheduler._watch_turn = lambda _turn: None
        self.control.send_user_message(self.root.agent_id, "resume after native lifecycle drains")

        scheduler._start_ready_agents()

        self.assertEqual(1, adapter._next_thread)
        self.assertEqual(0, adapter._next_turn)
        self.assertEqual(0, self.root.turn_count)
        self.assertEqual(0, self.root.delivered_message_count)
        bound_thread = self.managed._threads[self.root.agent_id]
        self.assertEqual([bound_thread], adapter.readiness_threads)

        scheduler._start_ready_agents()

        self.assertEqual(1, adapter._next_thread)
        self.assertEqual(1, adapter._next_turn)
        self.assertEqual(1, self.root.turn_count)
        self.assertEqual([bound_thread, bound_thread], adapter.readiness_threads)
        self.assertIn("resume after native lifecycle drains", adapter.prompts[bound_thread][-1])

    def test_a_readiness_probe_that_raises_blocks_one_child_and_spares_the_rest(self) -> None:
        """A dead probe is one worker's problem, not the session's.

        Four live sessions ended here.  The probe raised, the exception left
        the loop, and every later tool call in that chat answered only that the
        session had failed, until the user restarted the terminal.
        """

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own two workers",
            task_contract={},
        )
        stalled = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="probe never answers",
            task_contract={},
        )
        healthy = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="sibling still works",
            task_contract={},
        )
        probe = RaisingReadinessAdapter()
        default = self.adapter
        self.managed._adapter_selector = (
            lambda agent: probe if agent.agent_id == stalled.agent_id else default
        )
        recorded: list[tuple[str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, _agent, data: recorded.append((kind, dict(data))),
            ),
        )
        scheduler._watch_turn = lambda _turn: None

        scheduler._start_ready_agents()

        self.assertEqual(AgentStatus.BLOCKED, stalled.status)
        self.assertIn("readiness probe failed", stalled.blocker)
        self.assertIn(WORKER, stalled.blocker)
        self.assertIn("timed out after 2.0s", stalled.blocker)
        self.assertEqual(0, stalled.turn_count)
        # The sibling is the point: one dead probe must not stop the loop.
        self.assertEqual(1, healthy.turn_count)
        self.assertEqual(1, self.root.turn_count)
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(
            ["child-blocked"],
            [wake["reason"] for wake in session.wakes[branch.agent_id]],
        )
        errors = [data for kind, data in recorded if kind == "provider_error"]
        self.assertEqual(["can_start_turn"], [data["phase"] for data in errors])
        self.assertIn("can_start_turn", errors[0]["traceback"])

    def _approval_request_for(self, envelope_extra: dict) -> object:
        """Run one neutral approval envelope through and return its request."""

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own approval review",
            task_contract={},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Request one bounded effect",
            task_contract={},
            approvals="ask",
        )
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="native-worker-thread",
            start_result=policy(),
            tool_handler=None,
        )
        requested = threading.Event()
        seen: dict[str, str] = {}

        def record(event_type, _agent, data):
            if event_type == "approval_requested":
                seen["approval_id"] = data["approval_id"]
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )

        def review() -> None:
            scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:native-worker-item",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "native-worker-thread",
                        "turn": "native-worker-turn",
                        "request": "native-worker-item",
                    },
                    "correlation_attested": True,
                    **envelope_extra,
                },
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        request = scheduler._pending_approvals[seen["approval_id"]].request
        scheduler.resolve_approval(seen["approval_id"], "decline", "fixture")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        return request

    def test_an_approval_names_the_file_the_runtime_asked_to_write(self) -> None:
        """A manager asked to approve a write is told which file.

        Across 34 approvals resolved in one live day the command was empty
        every time, and a ZCode manager wrote in its own transcript that it
        could not see the command.  It approved anyway.
        """

        request = self._approval_request_for(
            {"effect": "modify", "event": {"tool": "Write", "path": "src/app.py"}}
        )

        self.assertEqual("Write", request.tool)
        self.assertEqual(("src/app.py",), request.command)
        self.assertEqual("modify", request.effect)
        self.assertEqual("workspace-write", request.permission)

    def test_an_approval_names_the_command_the_runtime_asked_to_run(self) -> None:
        request = self._approval_request_for(
            {
                "effect": "execute",
                "event": {"tool": "Bash", "command": ["python -m pytest tests/ -q"]},
            }
        )

        self.assertEqual("Bash", request.tool)
        self.assertEqual(("python -m pytest tests/ -q",), request.command)
        self.assertEqual("execute", request.effect)

    def test_an_approval_without_the_new_fields_is_exactly_the_one_sent_today(self) -> None:
        """An older bridge, Codex and Command Code all send no such fields.

        A missing field must never raise and never change a decision.
        """

        execute = self._approval_request_for({"effect": "execute"})

        self.assertEqual("runtime-effect", execute.tool)
        self.assertEqual((), execute.command)
        self.assertEqual("runtime requested command approval", execute.justification)
        self.assertEqual("workspace-execute", execute.permission)
        self.assertEqual("workspace", execute.target)

    def test_an_approval_whose_event_says_nothing_usable_keeps_todays_file_prompt(self) -> None:
        modify = self._approval_request_for(
            {"effect": "modify", "event": {"nothing": "usable"}}
        )

        self.assertEqual("runtime-effect", modify.tool)
        self.assertEqual((), modify.command)
        self.assertEqual("runtime requested file approval", modify.justification)
        self.assertEqual("workspace-write", modify.permission)
        self.assertEqual("workspace", modify.target)

    def _lifecycle_scheduler(self):
        """A scheduler whose lifecycle events are collected for reading."""

        lifecycle = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda event_type, _agent, data: lifecycle.append(
                    (event_type, dict(data))
                ),
            ),
        )
        return scheduler, lifecycle

    def _granted_worker(self, scheduler, objective: str, thread_id: str):
        delegated = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": objective,
                    "task_contract": {"criteria": ["observed"]},
                },
            )
        )
        worker = self.control.sessions[self.root.session_id].agents[
            delegated["agent_id"]
        ]
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id=thread_id,
            start_result=policy(),
            tool_handler=None,
        )
        return worker

    def test_a_standing_grant_records_what_was_read_or_fetched(self) -> None:
        """An accepted approval has to say what it accepted.

        A standing grant resolves without ever asking the manager, so the
        lifecycle events and the compact record are the only place an operator
        can see what a worker reached.  They carried the tool and the effect
        and stopped there: "Read / read", "WebFetch / network".  The bridge
        envelope already holds the subject under event.path -- the file for a
        read, the URL for a fetch, the query for a search -- so the answer was
        one hop away and unreachable.
        """

        scheduler, lifecycle = self._lifecycle_scheduler()
        worker = self._granted_worker(
            scheduler, "Observe bounded approvals", "subject-thread"
        )
        self.assertEqual("granted", worker.approvals)

        wanted = [
            ("Read", {"file_path": "/etc/hosts"}, "/etc/hosts"),
            ("WebSearch", {"query": "prior art"}, "prior art"),
            (
                "WebFetch",
                {"url": "https://docs.example.com/api"},
                "https://docs.example.com/api",
            ),
        ]
        for tool, arguments, _subject in wanted:
            with self.subTest(tool=tool):
                accepted = scheduler.review_approval(
                    "approval/request",
                    {
                        "approval_reference": "fixture:" + tool,
                        "provider": "fixture",
                        "provider_correlation": {
                            "session": "subject-thread",
                            "turn": "subject-turn",
                            "request": tool,
                        },
                        "correlation_attested": True,
                        **_tool_effect(tool),
                        "event": _approval_detail(tool, None, arguments),
                    },
                )
                self.assertEqual({"decision": "accept"}, accepted)

        subjects = [subject for _tool, _arguments, subject in wanted]
        requested = [data for kind, data in lifecycle if kind == "approval_requested"]
        resolved = [data for kind, data in lifecycle if kind == "approval_resolved"]
        self.assertEqual(3, len(requested))
        self.assertEqual(3, len(resolved))
        self.assertEqual(subjects, [data["subject"] for data in requested])
        # The resolved line is what an operator filters on to see what a
        # standing grant let through, so it carries the subject as well.
        self.assertEqual(subjects, [data["subject"] for data in resolved])
        self.assertEqual(
            ["Read", "WebSearch", "WebFetch"], [data["tool"] for data in requested]
        )
        records = scheduler.native_approval_records()
        self.assertEqual(subjects, [record["subject"] for record in records])
        self.assertEqual(
            ["accept", "accept", "accept"],
            [record["decision"] for record in records],
        )

    def test_a_decoy_field_never_wins_the_subject(self) -> None:
        """The subject field is the tool's to name, and it used to be a scan.

        A probe sent WebFetch a real url together with a file_path nothing
        would ever fetch, and the scan met file_path first: both lifecycle
        events and the compact record named the decoy. Every tool this project
        knows the shape of now names its own field.
        """

        scheduler, lifecycle = self._lifecycle_scheduler()
        worker = self._granted_worker(
            scheduler, "Observe a mixed-field approval", "decoy-thread"
        )
        self.assertEqual("granted", worker.approvals)

        wanted = [
            (
                "WebFetch",
                {"url": "https://real.example/api", "file_path": "/tmp/decoy"},
                "https://real.example/api",
            ),
            (
                "WebSearch",
                {"query": "real query", "path": "/tmp/decoy"},
                "real query",
            ),
            (
                "Read",
                {"file_path": "/etc/hosts", "url": "https://decoy.example"},
                "/etc/hosts",
            ),
        ]
        for tool, arguments, _subject in wanted:
            with self.subTest(tool=tool):
                accepted = scheduler.review_approval(
                    "approval/request",
                    {
                        "approval_reference": "decoy:" + tool,
                        "provider": "fixture",
                        "provider_correlation": {
                            "session": "decoy-thread",
                            "turn": "decoy-turn",
                            "request": tool,
                        },
                        "correlation_attested": True,
                        **_tool_effect(tool),
                        "event": _approval_detail(tool, None, arguments),
                    },
                )
                self.assertEqual({"decision": "accept"}, accepted)

        subjects = [subject for _tool, _arguments, subject in wanted]
        requested = [data for kind, data in lifecycle if kind == "approval_requested"]
        resolved = [data for kind, data in lifecycle if kind == "approval_resolved"]
        self.assertEqual(subjects, [data["subject"] for data in requested])
        self.assertEqual(subjects, [data["subject"] for data in resolved])
        self.assertEqual(
            subjects,
            [record["subject"] for record in scheduler.native_approval_records()],
        )

    def test_a_boundary_that_publishes_its_raw_input_is_read_by_tool_too(self) -> None:
        """A provider need not resolve the subject before the scheduler sees it.

        An envelope carrying the tool's own fields rather than a resolved
        "path" is read the same way, so a decoy cannot win there either. A
        resolved "path" alone stays the answer for a boundary that sends one.
        """

        self.assertEqual(
            ("WebFetch", ("https://real.example/api",)),
            VNextScheduler._approval_subject(
                {
                    "event": {
                        "tool": "WebFetch",
                        "file_path": "/tmp/decoy",
                        "url": "https://real.example/api",
                    }
                }
            ),
        )
        self.assertEqual(
            ("Read", ("/etc/hosts",)),
            VNextScheduler._approval_subject(
                {"event": {"tool": "Read", "path": "/etc/hosts"}}
            ),
        )

    def test_a_control_character_in_a_subject_stays_on_one_line(self) -> None:
        """A newline in a subject turned one audit line into two.

        The second line read like a record of its own, which is what an
        operator scanning a log would act on. The rendering replaces every
        character that can end a line; the structured field keeps the raw
        value, because an operator matching it against a real file name needs
        the bytes the provider actually sent.
        """

        forged = "/tmp/normal\nFORGED APPROVAL accept /etc/secrets"
        scheduler, lifecycle = self._lifecycle_scheduler()
        worker = self._granted_worker(
            scheduler, "Observe a forged subject", "forged-thread"
        )
        self.assertEqual("granted", worker.approvals)
        lifecycle.clear()  # Examine only this approval after delegation.

        accepted = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "forged:Read",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "forged-thread",
                    "turn": "forged-turn",
                    "request": "Read",
                },
                "correlation_attested": True,
                **_tool_effect("Read"),
                "event": _approval_detail("Read", None, {"file_path": forged}),
            },
        )
        self.assertEqual({"decision": "accept"}, accepted)

        rendered = "/tmp/normal\\nFORGED APPROVAL accept /etc/secrets"
        records = scheduler.native_approval_records()
        self.assertEqual(1, len(records))
        carriers = records + [data for _kind, data in lifecycle]
        for carrier in carriers:
            with self.subTest(carrier=sorted(carrier)):
                self.assertEqual(forged, carrier["subject"])
                self.assertEqual(rendered, carrier["subject_line"])
                self.assertNotIn("\n", carrier["subject_line"])
        self.assertEqual(
            rendered, VNextScheduler._approval_subject_line((forged,))
        )

    def test_a_refused_fetch_is_recorded_with_the_url_it_asked_for(self) -> None:
        """A refusal is the line an operator most wants named.

        This one is refused before any manager is reached, because nothing
        correlates the envelope to a worker.  It emits no lifecycle event, so
        the compact record is the only account of it that exists.
        """

        scheduler, _lifecycle = self._lifecycle_scheduler()
        self._granted_worker(scheduler, "Observe a refusal", "routed-thread")

        refused = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "fixture:refused-fetch",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "a-thread-nobody-owns",
                    "turn": "refused-turn",
                    "request": "refused-fetch",
                },
                "correlation_attested": True,
                **_tool_effect("WebFetch"),
                "event": _approval_detail(
                    "WebFetch", None, {"url": "https://upload.example.com/drop"}
                ),
            },
        )

        self.assertEqual(
            {
                "decision": "decline",
                "reason": _BOUNDARY_DECLINE_REASONS["unrouted"],
            },
            refused,
        )
        records = scheduler.native_approval_records()
        self.assertEqual(1, len(records))
        self.assertEqual("decline", records[0]["decision"])
        self.assertEqual("https://upload.example.com/drop", records[0]["subject"])

    def test_a_boundary_that_names_no_subject_records_what_it_did_before(self) -> None:
        """Codex and Command Code publish no event, and must stay unchanged.

        The key is absent rather than empty, so a reader cannot mistake a
        boundary that says nothing for one that read the root directory.  The
        provider's own request id stays out of the compact record too.
        """

        scheduler, lifecycle = self._lifecycle_scheduler()
        self._granted_worker(
            scheduler, "Run a command through a quiet boundary", "quiet-thread"
        )

        accepted = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "fixture:quiet-command",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "quiet-thread",
                    "turn": "quiet-turn",
                    "request": "quiet-command",
                },
                "correlation_attested": True,
                "effect": "execute",
            },
        )

        self.assertEqual({"decision": "accept"}, accepted)
        approvals = [
            data for kind, data in lifecycle
            if kind in {"approval_requested", "approval_resolved"}
        ]
        self.assertEqual(2, len(approvals))
        for data in approvals:
            self.assertNotIn("subject", data)
        records = scheduler.native_approval_records()
        self.assertEqual(1, len(records))
        self.assertNotIn("subject", records[0])
        self.assertNotIn("quiet-command", json.dumps(records))

    def test_a_websearch_from_a_granted_worker_is_accepted(self) -> None:
        """F17 end to end: the decision a Claude worker's WebSearch gets.

        The envelope is built here exactly as the Claude bridge builds it, from
        `_tool_effect` and `_approval_detail`, so this test fails the moment
        either side stops naming a web call's effect.  Before the fix the
        bridge named no effect and this reviewer answered "decline" without
        ever reading the worker's standing grant.
        """

        scheduler = self._scheduler()
        delegated = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": "Search the web for prior art",
                    "task_contract": {"criteria": ["web search answered"]},
                },
            )
        )
        worker = self.control.sessions[self.root.session_id].agents[
            delegated["agent_id"]
        ]
        self.assertEqual("granted", worker.approvals)
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="web-worker-thread",
            start_result=policy(),
            tool_handler=None,
        )

        outcome = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "fixture:web-tool-use",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "web-worker-thread",
                    "turn": "web-worker-turn",
                    "request": "web-tool-use",
                },
                "correlation_attested": True,
                **_tool_effect("WebSearch"),
                "event": _approval_detail(
                    "WebSearch", None, {"query": "prior art for vnext"}
                ),
            },
        )

        self.assertEqual({"decision": "accept"}, outcome)
        request = self._approval_request_for(
            {
                **_tool_effect("WebFetch"),
                "event": _approval_detail(
                    "WebFetch", None, {"url": "https://docs.example.com/api"}
                ),
            }
        )
        self.assertEqual("WebFetch", request.tool)
        self.assertEqual("network", request.effect)
        self.assertEqual("network-read", request.permission)
        self.assertEqual(("https://docs.example.com/api",), request.command)

    def test_a_granted_codex_workers_escalation_is_accepted(self) -> None:
        """A thread vNext started itself is routed to its worker's grant.

        The Codex adapter caches the identity `thread/start` acknowledged.  It
        used to report that cache as "started", which the router never
        accepts, so every escalation of a granted Codex worker waited out the
        grace window and was declined as unrouted ("rejected by user").
        """

        lifecycle: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: lifecycle.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )
        delegated = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": "Run one command that needs escalation",
                    "task_contract": {"criteria": ["command ran"]},
                },
            )
        )
        worker = self.control.sessions[self.root.session_id].agents[
            delegated["agent_id"]
        ]
        self.assertEqual("granted", worker.approvals)
        codex = object.__new__(VNextAppServerAdapter)
        codex.provider = "codex"
        codex._condition = threading.Condition(threading.RLock())
        codex._native_thread_attestations = {
            "codex-worker-thread": {
                "provider": "codex",
                "provider_session": "codex-worker-thread",
                "binding_phase": "started",
            }
        }
        codex.read_thread = lambda *_args, **_kwargs: self.fail(
            "a thread this adapter started is attested from its start reply"
        )
        codex.native_approval_handler = scheduler.review_approval
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="codex-worker-thread",
            start_result=policy(),
            tool_handler=None,
            adapter=codex,
        )

        with patch("vnext.vnext_scheduler.APPROVAL_ROUTING_GRACE_SECONDS", 0.2):
            answer = codex._handle_native_approval(
                "item/commandExecution/requestApproval",
                {
                    "itemId": "call-ps",
                    "threadId": "codex-worker-thread",
                    "turnId": "codex-worker-turn",
                },
            )

        self.assertEqual({"decision": "accept"}, answer)
        resolved = [data for kind, _agent, data in lifecycle if kind == "approval_resolved"]
        self.assertEqual(["standing-grant"], [data["resolver"] for data in resolved])

    def test_an_unrouted_approval_is_declined_and_recorded(self) -> None:
        lifecycle: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: lifecycle.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )

        with patch("vnext.vnext_scheduler.APPROVAL_ROUTING_GRACE_SECONDS", 0.05):
            answer = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "approval:call-ps",
                    "provider": "codex",
                    "provider_correlation": {
                        "session": "thread-nobody-owns",
                        "turn": "some-turn",
                        "request": "call-ps",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )

        self.assertEqual(
            {"decision": "decline", "reason": _BOUNDARY_DECLINE_REASONS["unrouted"]},
            answer,
        )
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [kind for kind, _agent, _data in lifecycle],
        )
        (_, requested_on, requested), (_, resolved_on, resolved) = lifecycle
        # Recorded on the session root, and it names no worker it never reached.
        self.assertEqual(self.root.agent_id, requested_on)
        self.assertEqual(self.root.agent_id, resolved_on)
        self.assertNotIn("worker_agent", requested)
        self.assertEqual("codex", requested["provider"])
        self.assertEqual("thread-nobody-owns", requested["provider_session"])
        self.assertEqual("execute", requested["effect"])
        self.assertEqual(requested["approval_id"], resolved["approval_id"])
        self.assertEqual("decline", resolved["decision"])
        self.assertEqual("vnext-approval-boundary", resolved["resolver"])
        self.assertEqual(_BOUNDARY_DECLINE_REASONS["unrouted"], resolved["reason"])
        self.assertEqual(
            _BOUNDARY_DECLINE_REASONS["unrouted"],
            scheduler.native_approval_records()[-1]["reason"],
        )

    def test_an_unrouted_local_reservation_is_not_named_a_provider_session(
        self,
    ) -> None:
        lifecycle: list[tuple[str, str, dict]] = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: lifecycle.append(
                    (kind, agent.agent_id, dict(data))
                )
            ),
        )

        with patch("vnext.vnext_scheduler.APPROVAL_ROUTING_GRACE_SECONDS", 0.05):
            answer = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "approval:call-ps",
                    "provider": "claude",
                    "routing_handle": {
                        "reservation_id": "local-unowned-reservation",
                        "turn_reference": "local-turn",
                    },
                    "correlation_attested": False,
                    "effect": "execute",
                },
            )

        self.assertEqual(
            {"decision": "decline", "reason": _BOUNDARY_DECLINE_REASONS["unrouted"]},
            answer,
        )
        (_, requested_on, requested), _resolved = lifecycle
        self.assertEqual(self.root.agent_id, requested_on)
        self.assertNotIn("provider_session", requested)
        self.assertEqual("local-unowned-reservation", requested["routing_reservation"])

    def test_a_read_outside_the_workspace_reaches_the_reviewer(self) -> None:
        request = self._approval_request_for(
            {
                **_tool_effect("Read"),
                "event": _approval_detail("Read", None, {"file_path": "/etc/hosts"}),
            }
        )

        self.assertEqual("Read", request.tool)
        self.assertEqual("read", request.effect)
        self.assertEqual("workspace-read", request.permission)
        self.assertEqual(("/etc/hosts",), request.command)

    def test_an_effect_this_reviewer_cannot_name_is_still_declined(self) -> None:
        scheduler = self._scheduler()
        self.assertEqual(
            {
                "decision": "decline",
                "reason": _BOUNDARY_DECLINE_REASONS["unrouted"],
            },
            scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "claude:unnamed",
                    "provider": "claude",
                    "provider_correlation": {
                        "session": "web-worker-thread",
                        "turn": "web-worker-turn",
                        "request": "unnamed",
                    },
                    "correlation_attested": True,
                    "effect": "exfiltrate",
                },
            ),
        )

    def test_native_control_lease_reserves_before_slow_provider_bind(self) -> None:
        adapter = BlockingBindAdapter()
        self.adapter = adapter
        self.managed.adapter = adapter
        self.managed._adapter_selector = lambda _agent: adapter
        scheduler = self._scheduler()
        errors: list[BaseException] = []

        def acquire() -> None:
            try:
                scheduler.acquire_native_control_lease(self.root.agent_id)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=acquire)
        thread.start()
        self.assertTrue(adapter.binding_started.wait(1))
        # The lease is held while start_thread is blocked. A concurrent loop
        # must not start a managed turn on the still-READY root.
        scheduler._start_ready_agents()
        self.assertEqual(0, adapter._next_turn)
        adapter.binding_release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        self.assertIn(self.root.agent_id, scheduler._native_control_leases)

    def test_completed_primary_can_lease_and_resume_from_first_native_turn(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        thread_id = self.managed._threads[self.root.agent_id]
        self.control.complete_agent(self.root.agent_id, {"decision": "accepted"})

        scheduler.acquire_native_control_lease(self.root.agent_id)
        adopted = scheduler.adopt_native_turn(
            agent_id=self.root.agent_id,
            provider="fixture",
            thread_id=thread_id,
            turn_id="native-followup-objective",
            cursor=0,
        )

        self.assertEqual(AgentStatus.RUNNING, self.root.status)
        self.assertEqual(adopted.control_turn_id, self.root.active_turn_id)
        self.assertIn(self.root.agent_id, scheduler._native_control_leases)
        self.assertEqual(1, self.adapter._next_thread)
        self.assertEqual(0, self.adapter._next_turn)

    def test_native_control_lease_refuses_an_active_managed_turn(self) -> None:
        scheduler = self._scheduler()
        self._open_root_turn(scheduler)

        with self.assertRaisesRegex(ProtocolError, "requires an idle") as refused:
            scheduler.acquire_native_control_lease(self.root.agent_id)

        self.assertEqual("native-control-active-turn", refused.exception.code)
        self.assertNotIn(self.root.agent_id, scheduler._native_control_leases)

    def test_fresh_attested_native_turn_resumes_an_awaiting_native_manager(self) -> None:
        scheduler = self._scheduler()
        scheduler._watch_turn = lambda _turn: None
        self.assertTrue(scheduler._bind_agent(self.root))
        scheduler.acquire_native_control_lease(self.root.agent_id)
        native_thread = self.managed._threads[self.root.agent_id]
        caller = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=native_thread, native_child_thread_id="native-awaiting-manager",
            attested=True, delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native manager waits",
        ))
        native_agent = self.control.sessions[self.root.session_id].agents[caller.agent_id]
        first = scheduler.adopt_native_turn(
            agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
            turn_id="native-first-turn", cursor=0,
        )
        managed_child = self.control.spawn_agent(
            requester_id=native_agent.agent_id, parent_agent_id=native_agent.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="managed child", task_contract={},
        )
        self.control.await_agents(native_agent.agent_id, [managed_child.agent_id])
        scheduler._handle_turn_finished(_TurnFinished(
            native_agent.agent_id, first, result={"status": "completed"},
        ))
        self.assertEqual(AgentStatus.AWAITING_WORKERS, native_agent.status)
        self.assertNotIn(native_agent.agent_id, scheduler._active_turns)

        followup = scheduler.adopt_native_turn(
            agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
            turn_id="native-fresh-followup", cursor=1,
        )

        self.assertEqual(AgentStatus.RUNNING, native_agent.status)
        self.assertEqual(followup, scheduler._active_turns[native_agent.agent_id])
        self.assertNotEqual(first.control_turn_id, followup.control_turn_id)
        self.assertNotIn(native_agent.agent_id, self.control.sessions[self.root.session_id].waits)
        self.assertTrue(any(
            event.event_type == "native-turn-resumed" and event.agent_id == native_agent.agent_id
            for event in self.control.sessions[self.root.session_id].events
        ))

    def test_awaiting_native_manager_replays_known_turn_without_mutating_wait(self) -> None:
        scheduler = self._scheduler()
        scheduler._watch_turn = lambda _turn: None
        self.assertTrue(scheduler._bind_agent(self.root))
        scheduler.acquire_native_control_lease(self.root.agent_id)
        native_thread = self.managed._threads[self.root.agent_id]
        caller = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=native_thread, native_child_thread_id="native-awaiting-rejects",
            attested=True, delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native manager waits",
        ))
        native_agent = self.control.sessions[self.root.session_id].agents[caller.agent_id]
        first = scheduler.adopt_native_turn(
            agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
            turn_id="native-stale-turn", cursor=0,
        )
        managed_child = self.control.spawn_agent(
            requester_id=native_agent.agent_id, parent_agent_id=native_agent.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="managed child", task_contract={},
        )
        self.control.await_agents(native_agent.agent_id, [managed_child.agent_id])
        scheduler._handle_turn_finished(_TurnFinished(
            native_agent.agent_id, first, result={"status": "completed"},
        ))

        session = self.control.sessions[self.root.session_id]
        waits_before = dict(session.waits)
        active_before = dict(scheduler._active_turns)
        waiters_before = set(scheduler._waiters)
        events_before = len(session.events)
        replayed = scheduler.adopt_native_turn(
            agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
            turn_id="native-stale-turn", cursor=1,
        )

        self.assertEqual(first, replayed)
        self.assertEqual(AgentStatus.AWAITING_WORKERS, native_agent.status)
        self.assertEqual(waits_before, session.waits)
        self.assertEqual(active_before, scheduler._active_turns)
        self.assertEqual(waiters_before, scheduler._waiters)
        self.assertEqual(events_before, len(session.events))

    def test_awaiting_native_manager_rejects_foreign_and_unleased_fresh_turns(self) -> None:
        scheduler = self._scheduler()
        scheduler._watch_turn = lambda _turn: None
        self.assertTrue(scheduler._bind_agent(self.root))
        scheduler.acquire_native_control_lease(self.root.agent_id)
        native_thread = self.managed._threads[self.root.agent_id]
        caller = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=native_thread, native_child_thread_id="native-awaiting-rejects",
            attested=True, delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native manager waits",
        ))
        native_agent = self.control.sessions[self.root.session_id].agents[caller.agent_id]
        first = scheduler.adopt_native_turn(
            agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
            turn_id="native-first-turn", cursor=0,
        )
        managed_child = self.control.spawn_agent(
            requester_id=native_agent.agent_id, parent_agent_id=native_agent.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="managed child", task_contract={},
        )
        self.control.await_agents(native_agent.agent_id, [managed_child.agent_id])
        scheduler._handle_turn_finished(_TurnFinished(
            native_agent.agent_id, first, result={"status": "completed"},
        ))

        with self.assertRaisesRegex(ProtocolError, "awaiting leased thread"):
            scheduler.adopt_native_turn(
                agent_id=native_agent.agent_id, provider="fixture", thread_id="foreign-thread",
                turn_id="native-foreign-turn", cursor=2,
            )
        self.assertEqual(AgentStatus.AWAITING_WORKERS, native_agent.status)
        scheduler.release_native_control_lease(native_agent.agent_id)
        with self.assertRaisesRegex(ProtocolError, "held control lease"):
            scheduler.adopt_native_turn(
                agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
                turn_id="native-unleased-turn", cursor=3,
            )
        self.assertEqual(AgentStatus.AWAITING_WORKERS, native_agent.status)

    def test_terminal_native_child_cannot_be_reconciled_for_a_followup_turn(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        scheduler.acquire_native_control_lease(self.root.agent_id)
        caller = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-terminal-child", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="terminal native child",
        ))
        native_agent = self.control.sessions[self.root.session_id].agents[caller.agent_id]
        self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=native_agent.agent_id)

        with self.assertRaisesRegex(ProtocolError, "cannot start a turn from cancelled"):
            scheduler.adopt_native_turn(
                agent_id=native_agent.agent_id, provider="fixture", thread_id=native_agent.thread_id,
                turn_id="terminal-native-followup", cursor=0,
            )
        self.assertEqual(AgentStatus.CANCELLED, native_agent.status)

    def test_primary_discovers_native_ids_and_completes_after_cancelled_child_settles(self) -> None:
        scheduler = self._scheduler()
        root_turn = self._open_root_turn(scheduler)
        bindings = [scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.root.thread_id, native_child_thread_id="provider-" + label,
            attested=True, delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective=label,
        )) for label in ("cancelled", "survivor")]
        children = [self.control.sessions[self.root.session_id].agents[b.agent_id] for b in bindings]
        context = ToolCallContext(thread_id=self.root.thread_id, turn_id=root_turn.runtime.turn_id,
                                  call_id="root-inspect", cursor=0)
        call = lambda tool, args: scheduler._manager_handler(self.root.agent_id, tool, args, context)
        inspected = call("inspect", {"agent_id": "self", "deep": False})
        self.assertTrue(inspected.success)
        self.assertEqual([child.agent_id for child in children],
                         [child["agent_id"] for child in inspected.value["children"]])
        self.assertEqual(["provider-cancelled", "provider-survivor"],
                         [child["runtime_thread_id"] for child in inspected.value["children"]])
        self.assertTrue(all(not child["terminal"] for child in inspected.value["children"]))
        self.assertEqual({}, scheduler._children_delivered)
        rejected = call("await_children", {"agent_ids": ["provider-cancelled"]})
        self.assertFalse(rejected.success)
        self.assertIn("Inspect self", rejected.value["error"])
        self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=children[0].agent_id)
        premature = call("complete_session", {"decision": "accepted", "summary": "done", "criteria": {}})
        self.assertFalse(premature.success)
        self.managed.complete_agent(children[1].agent_id, {"outcome": "done"})
        waited = call("await_children", {"agent_ids": [child.agent_id for child in children]})
        self.assertTrue(waited.success)
        self.assertEqual([], waited.value["awaiting"])
        self.assertTrue(waited.value["settled"])
        self.assertEqual(["cancelled", "completed"], [c["status"] for c in waited.value["children"]])
        self.assertTrue(all(child["terminal"] for child in waited.value["children"]))
        completed = call("complete_session", {"decision": "accepted", "summary": "done", "criteria": {}})
        self.assertTrue(completed.success, completed.value)
        self.assertEqual(AgentStatus.COMPLETED, self.root.status)

    def test_native_child_adoption_uses_existing_thread_and_native_turn_controls(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        scheduler.acquire_native_control_lease(self.root.agent_id)
        child_thread = "native-child-thread"
        binding = scheduler.adopt_native_child(
            NativeChildObservation(
                provider="fixture",
                parent_agent_id=self.root.agent_id,
                parent_thread_id=self.managed._threads[self.root.agent_id],
                native_child_thread_id=child_thread,
                attested=True,
                delivery_contract={
                    "context_messages": "available",
                    "interrupt": "available",
                    "history": "available",
                },
                model_id=WORKER,
                role=AgentRole.WORKER.value,
                effort="low",
                objective="Work through the native harness",
                task_contract={},
            )
        )
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]

        self.assertEqual(child_thread, child.thread_id)
        self.assertNotEqual(binding.agent_id, child_thread)
        self.assertIn(child.agent_id, scheduler._native_control_leases)
        delivered = scheduler.send_message(
            sender_id=self.root.agent_id,
            target_id=child.agent_id,
            text="retain this context for your next native tool call",
        )
        self.assertEqual("queued-for-native-tool-read", delivered["delivery"])
        self.assertEqual("retain this context for your next native tool call", child.messages[-1].text)
        before_threads, before_turns = self.adapter._next_thread, self.adapter._next_turn
        scheduler._start_ready_agents()
        self.assertEqual(before_threads, self.adapter._next_thread)
        self.assertEqual(before_turns, self.adapter._next_turn)

        adopted = scheduler.adopt_native_turn(
            agent_id=child.agent_id,
            provider="fixture",
            thread_id=child_thread,
            turn_id="native-child-turn",
            cursor=0,
        )
        tool = binding.tool_handler
        self.assertIsNotNone(tool)
        delegated = payload(tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER,
            "objective": "native child delegates a bounded read",
            "task_contract": {},
            "workspace": "shared",
            "effort": "low",
        }, None))
        self.assertTrue(delegated["success"], delegated)
        self.assertEqual(adopted, scheduler._active_turns[child.agent_id])
        self.assertEqual("interrupt-requested", scheduler.interrupt_agent(child.agent_id)["status"])
        self.assertIn(child_thread, self.adapter.interrupted)

    def test_native_tool_response_repeats_until_exact_callers_acknowledgement(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        bindings = []
        for child_thread in ("native-a", "native-b"):
            bindings.append(scheduler.adopt_native_child(NativeChildObservation(
                provider="fixture", parent_agent_id=self.root.agent_id,
                parent_thread_id=self.managed._threads[self.root.agent_id],
                native_child_thread_id=child_thread, attested=True,
                delivery_contract={"context_messages": "available", "dynamic_tools": "available"},
                model_id=WORKER, role=AgentRole.WORKER.value, objective="native work")))
        for binding, text in zip(bindings, ("message a", "message b")):
            scheduler.send_message(sender_id=self.root.agent_id, target_id=binding.agent_id, text=text)
        missing_turn = ToolCallContext(thread_id="native-a", turn_id="", call_id="unattested")
        unscoped = scheduler._manager_handler(bindings[0].agent_id, "inspect", {"agent_id": "self"}, missing_turn)
        self.assertNotIn("messages", unscoped.value)
        self.assertEqual(0, self.control.sessions[self.root.session_id].agents[bindings[0].agent_id].delivered_message_count)
        context = ToolCallContext(thread_id="native-a", turn_id="native-turn-a", call_id="call")
        first = scheduler._manager_handler(bindings[0].agent_id, "inspect", {"agent_id": "self"}, context)
        self.assertTrue(first.success)
        self.assertEqual(["message a"], [item["text"] for item in first.value["messages"]])
        second = scheduler._manager_handler(bindings[0].agent_id, "inspect", {"agent_id": "self"}, context)
        self.assertEqual(first.value["messages"], second.value["messages"])
        self.assertEqual(1, first.value["message_cursor"])
        child = self.control.sessions[self.root.session_id].agents[bindings[0].agent_id]
        self.assertEqual(0, child.delivered_message_count)
        # Completion cannot bury a message merely offered by an inspect.
        completion_args = {"outcome": "done", "verified": True, "evidence": []}
        premature = scheduler._manager_handler(bindings[0].agent_id, "complete_agent", completion_args, context)
        self.assertFalse(premature.success)
        self.assertEqual("unread-messages", premature.value["error_code"])
        for cursor in (True, -1, 2, "1"):
            rejected = scheduler._manager_handler(bindings[0].agent_id, "inspect", {
                "agent_id": "self", "acknowledge_messages_through": cursor}, context)
            self.assertFalse(rejected.success)
            self.assertEqual(0, child.delivered_message_count)
        for target, ctx, caller in (
            ("self", missing_turn, bindings[0].agent_id),
            (bindings[1].agent_id, context, bindings[0].agent_id),
            ([], context, bindings[0].agent_id),
            ("self", ToolCallContext(thread_id="native-b", turn_id="native-turn-b"), bindings[1].agent_id),
        ):
            rejected = scheduler._manager_handler(caller, "inspect", {
                "agent_id": target, "acknowledge_messages_through": 1}, ctx)
            self.assertFalse(rejected.success)
            self.assertEqual(0, child.delivered_message_count)
        # A fresh arrival cannot be acknowledged with the earlier receipt.
        scheduler.send_message(sender_id=self.root.agent_id, target_id=bindings[0].agent_id, text="later")
        acknowledged = scheduler._manager_handler(bindings[0].agent_id, "inspect", {
            "agent_id": "self", "acknowledge_messages_through": first.value["message_cursor"]}, context)
        self.assertTrue(acknowledged.success)
        self.assertEqual(["later"], [m["text"] for m in acknowledged.value["messages"]])
        self.assertEqual(1, child.delivered_message_count)
        # Replaying the same acknowledgement does not consume the next message.
        replayed = scheduler._manager_handler(bindings[0].agent_id, "inspect", {
            "agent_id": "self", "acknowledge_messages_through": 1}, context)
        self.assertEqual(["later"], [m["text"] for m in replayed.value["messages"]])
        self.assertEqual(1, child.delivered_message_count)
        final = scheduler._manager_handler(bindings[0].agent_id, "inspect", {
            "agent_id": "self", "acknowledge_messages_through": 2}, context)
        self.assertTrue(final.success)
        self.assertNotIn("messages", final.value)
        self.assertEqual(2, child.delivered_message_count)
        completed = scheduler._manager_handler(bindings[0].agent_id, "complete_agent", completion_args, context)
        self.assertTrue(completed.success)
        self.assertEqual(AgentStatus.COMPLETED, child.status)
        sibling = self.control.sessions[self.root.session_id].agents[bindings[1].agent_id]
        self.assertEqual(0, sibling.delivered_message_count)

    def test_external_root_reads_acknowledges_and_completes_without_a_native_turn(self) -> None:
        self.control.registry.cards[MANAGER] = ModelCard(
            MANAGER,
            frozenset({AgentRole.ROOT_MANAGER, AgentRole.BRANCH_MANAGER, AgentRole.WORKER}),
            provider="external",
        )
        lifecycle = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, _agent, data: lifecycle.append((kind, dict(data)))
            ),
        )
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Report one result to the external root",
            task_contract={},
        )
        body = "the bounded  check\n\tpassed"
        scheduler.send_message(
            sender_id=child.agent_id,
            target_id=self.root.agent_id,
            text=body,
        )
        self.control.complete_agent(child.agent_id, {"outcome": "reported"})
        message_event = next(
            data for kind, data in lifecycle
            if kind == "command_acknowledged" and data["command"] == "send_message"
        )
        self.assertEqual(len(body), message_event["message_characters"])
        self.assertEqual(body, message_event["message"])

        read = scheduler._manager_handler(
            self.root.agent_id,
            "inspect",
            {"agent_id": "self", "deep": False},
            None,
        )

        self.assertTrue(read.success, read.value)
        self.assertEqual(
            [body],
            [message["text"] for message in read.value["messages"]],
        )
        self.assertEqual("external-tool-response", read.value["message_delivery"])
        self.assertEqual(1, read.value["message_cursor"])
        premature = scheduler._manager_handler(
            self.root.agent_id,
            "complete_session",
            {"decision": "accepted", "summary": "done", "criteria": {}},
            None,
        )
        self.assertFalse(premature.success)
        self.assertEqual("unread-messages", premature.value["error_code"])
        ahead = scheduler._manager_handler(
            self.root.agent_id,
            "inspect",
            {"agent_id": "self", "deep": False, "acknowledge_messages_through": 2},
            None,
        )
        self.assertFalse(ahead.success)
        acknowledged = scheduler._manager_handler(
            self.root.agent_id,
            "inspect",
            {"agent_id": "self", "deep": False, "acknowledge_messages_through": 1},
            None,
        )
        self.assertTrue(acknowledged.success, acknowledged.value)
        completed = scheduler._manager_handler(
            self.root.agent_id,
            "complete_session",
            {"decision": "accepted", "summary": "done", "criteria": {}},
            None,
        )
        self.assertTrue(completed.success, completed.value)
        self.assertEqual(AgentStatus.COMPLETED, self.root.status)

    def test_cancel_native_child_before_turn_event_stops_provider_first(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-child-before-turn", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        stopped = []
        def stop(thread_id):
            self.assertNotEqual(child.status, AgentStatus.CANCELLED)
            stopped.append(thread_id)
            return {"status": "interrupt-requested"}
        self.adapter.cancel_native_child = stop
        scheduler.cancel_agent(child.agent_id)
        self.assertEqual(["native-child-before-turn"], stopped)
        self.assertEqual(AgentStatus.CANCELLED, child.status)

    def test_a_second_cancel_retries_a_native_child_stop_the_provider_refused(self) -> None:
        """A refused stop must not turn into success on the next try.

        The first cancel already moved the record to CANCELLED, so the native
        branch read "terminal" and skipped the provider entirely. The second
        call then answered success with the child's turn still running.
        """

        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-retry-after-refusal", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        stopped: list[str] = []
        refuse = True

        def stop(thread_id):
            stopped.append(thread_id)
            if refuse:
                raise TimeoutError("provider refused")
            return {"status": "interrupt-requested"}

        self.adapter.cancel_native_child = stop

        first = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}, None,
        ))
        self.assertFalse(first["success"], first)
        self.assertEqual(["native-retry-after-refusal"], stopped)

        second = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}, None,
        ))
        self.assertFalse(second["success"], second)
        self.assertEqual(
            [{"agent_id": child.agent_id, "reason": "TimeoutError: provider refused"}],
            second["uninterrupted"],
        )
        self.assertEqual(
            ["native-retry-after-refusal", "native-retry-after-refusal"], stopped
        )

        refuse = False
        third = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}, None,
        ))
        self.assertTrue(third["success"], third)
        self.assertEqual(3, len(stopped))

        # The stop landed, so a fourth cancel has nothing left to ask for.
        fourth = payload(scheduler._manager_handler(
            self.root.agent_id, "cancel_agent", {"agent_id": child.agent_id}, None,
        ))
        self.assertTrue(fourth["success"], fourth)
        self.assertEqual(3, len(stopped))

    def test_session_cancel_stops_native_child_before_its_first_turn_event(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-before-session-cancel", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        stopped = []
        def stop(thread_id):
            self.assertNotEqual(child.status, AgentStatus.CANCELLED)
            stopped.append(thread_id)
        self.adapter.cancel_native_child = stop
        scheduler._cancel_tree()
        self.assertEqual(["native-before-session-cancel"], stopped)
        self.assertEqual(AgentStatus.CANCELLED, child.status)

    def test_provider_task_completion_finishes_native_child_without_accepting_outcome(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-task", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        turn = scheduler.adopt_native_turn(agent_id=child.agent_id, provider="fixture",
                                          thread_id=child.thread_id, turn_id="task-turn", cursor=0)
        scheduler._handle_turn_finished(_TurnFinished(child.agent_id, turn,
            result={"status": "completed", "native_task_terminal": True, "summary": "provider returned"}))
        self.assertEqual(AgentStatus.COMPLETED, child.status)
        self.assertFalse(child.result["verified"])
        self.assertEqual("provider-native-task", child.result["source"])
        self.assertNotIn("decision", child.result)

    def _native_child_in_a_turn(self, scheduler, name):
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id=name, attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        turn = scheduler.adopt_native_turn(agent_id=child.agent_id, provider="fixture",
                                          thread_id=child.thread_id, turn_id=f"{name}-turn", cursor=0)
        return child, turn

    def _report(self, scheduler, child, reason):
        answer = payload(scheduler._manager_handler(
            child.agent_id, "report_blocked", {"reason": reason}, None,
        ))
        self.assertTrue(answer["success"], answer)
        _deliver_reported_blocks(scheduler)

    def test_a_native_child_that_reports_blocked_is_not_completed_by_its_task_end(self) -> None:
        """A Claude child ends its turn normally after report_blocked.

        That end arrives as a native task end, which used to complete the
        child: the parent heard child-completed, the reason was lost, and
        send_message and retry were both refused.
        """

        scheduler = self._scheduler()
        session = self.control.sessions[self.root.session_id]
        child, turn = self._native_child_in_a_turn(scheduler, "native-reports-blocked")
        self._report(scheduler, child, "needs a GitHub token")

        scheduler._handle_turn_finished(_TurnFinished(child.agent_id, turn,
            result={"status": "completed", "native_task_terminal": True, "summary": "blocked"}))

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertIn("reported by the worker: needs a GitHub token", child.blocker)
        reasons = [wake["reason"] for wake in session.wakes.get(self.root.agent_id, [])]
        self.assertEqual(1, reasons.count("child-blocked"), reasons)
        self.assertNotIn("child-completed", reasons)

    def test_two_reports_in_one_native_turn_keep_both_reasons(self) -> None:
        scheduler = self._scheduler()
        session = self.control.sessions[self.root.session_id]
        child, turn = self._native_child_in_a_turn(scheduler, "native-reports-twice")
        self._report(scheduler, child, "needs a GitHub token")
        self._report(scheduler, child, "the release branch is protected")
        self._report(scheduler, child, "needs a GitHub token")

        scheduler._handle_turn_finished(_TurnFinished(child.agent_id, turn,
            result={"status": "completed", "native_task_terminal": True}))

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertEqual(1, child.blocker.count("needs a GitHub token"), child.blocker)
        self.assertIn("the release branch is protected", child.blocker)
        reasons = [wake["reason"] for wake in session.wakes.get(self.root.agent_id, [])]
        self.assertEqual(1, reasons.count("child-blocked"), reasons)

    def test_a_native_report_keeps_the_unfinished_children_in_the_blocker(self) -> None:
        scheduler = self._scheduler()
        child, turn = self._native_child_in_a_turn(scheduler, "native-reports-with-child")
        unfinished = self.control.spawn_agent(
            requester_id=child.agent_id, parent_agent_id=child.agent_id,
            role=AgentRole.WORKER, model_id=WORKER, objective="unfinished nested work",
            task_contract={"criteria": ["nested work complete"]},
        )
        self._report(scheduler, child, "needs a GitHub token")

        scheduler._handle_turn_finished(_TurnFinished(child.agent_id, turn,
            result={"status": "completed", "native_task_terminal": True}))

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertIn("reported by the worker: needs a GitHub token", child.blocker)
        self.assertIn("1 unfinished child", child.blocker)
        self.assertIn(unfinished.agent_id, child.blocker)

    def test_native_task_with_an_unfinished_child_blocks_with_child_evidence(self) -> None:
        """An unfinished child is a real blocker, and the blocker must identify it.

        The old native-terminal branch collapsed unfinished children and unread
        mail into one sentence, so a manager could not tell which child kept
        the worker from finishing or how many children remained.
        """

        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-task-with-child", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        unfinished = self.control.spawn_agent(
            requester_id=child.agent_id,
            parent_agent_id=child.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="unfinished nested work",
            task_contract={"criteria": ["nested work complete"]},
        )
        turn = scheduler.adopt_native_turn(
            agent_id=child.agent_id,
            provider="fixture",
            thread_id=child.thread_id,
            turn_id="task-turn-with-child",
            cursor=0,
        )

        scheduler._handle_turn_finished(_TurnFinished(
            child.agent_id,
            turn,
            result={"status": "completed", "native_task_terminal": True},
        ))

        self.assertEqual(AgentStatus.BLOCKED, child.status)
        self.assertIn("1 unfinished child", child.blocker)
        self.assertIn(unfinished.agent_id, child.blocker)

    def test_a_long_list_of_unfinished_children_says_how_many_it_left_out(self) -> None:
        """The blocker names three children and counts the rest.

        A manager reading three ids out of seven would take the three for the
        whole list and go looking for the missing work in the wrong place.
        """

        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-task-with-many-children", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        for index in range(5):
            self.control.spawn_agent(
                requester_id=child.agent_id,
                parent_agent_id=child.agent_id,
                role=AgentRole.WORKER,
                model_id=WORKER,
                objective=f"nested work {index}",
                task_contract={"criteria": ["nested work complete"]},
            )
        turn = scheduler.adopt_native_turn(
            agent_id=child.agent_id,
            provider="fixture",
            thread_id=child.thread_id,
            turn_id="task-turn-with-many-children",
            cursor=0,
        )

        scheduler._handle_turn_finished(_TurnFinished(
            child.agent_id,
            turn,
            result={"status": "completed", "native_task_terminal": True},
        ))

        self.assertIn("5 unfinished children", child.blocker)
        self.assertIn("and 2 more", child.blocker)

    def test_native_task_with_unread_messages_returns_worker_ready_for_followup(self) -> None:
        """Unread mail needs a follow-up turn, not a blocker that prevents the worker reading it.

        The old native-terminal branch blocked on unread mail, which stranded
        the message because a blocked worker could never receive its next turn.
        """

        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        binding = scheduler.adopt_native_child(NativeChildObservation(
            provider="fixture", parent_agent_id=self.root.agent_id,
            parent_thread_id=self.managed._threads[self.root.agent_id],
            native_child_thread_id="native-task-with-mail", attested=True,
            delivery_contract={"context_messages": "available", "interrupt": "available"},
            model_id=WORKER, role=AgentRole.WORKER.value, objective="native work"))
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        self.control.message_agent(
            self.root.agent_id,
            child.agent_id,
            "read this before you finish",
        )
        turn = scheduler.adopt_native_turn(
            agent_id=child.agent_id,
            provider="fixture",
            thread_id=child.thread_id,
            turn_id="task-turn-with-mail",
            cursor=0,
        )

        scheduler._handle_turn_finished(_TurnFinished(
            child.agent_id,
            turn,
            result={"status": "completed", "native_task_terminal": True},
        ))

        self.assertEqual(AgentStatus.READY, child.status)
        self.assertEqual(1, scheduler._unread_message_count(child))

    def test_native_child_without_attested_inbox_refuses_undeliverable_messages(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))
        scheduler.acquire_native_control_lease(self.root.agent_id)
        binding = scheduler.adopt_native_child(
            NativeChildObservation(
                provider="fixture",
                parent_agent_id=self.root.agent_id,
                parent_thread_id=self.managed._threads[self.root.agent_id],
                native_child_thread_id="native-child-no-inbox",
                attested=True,
                delivery_contract={"context_messages": "unavailable", "interrupt": "available"},
                model_id=WORKER,
                role=AgentRole.WORKER.value,
                objective="provider does not expose an inbox",
            )
        )

        self.assertIsNone(binding.tool_handler)
        with self.assertRaisesRegex(ProtocolError, "did not attest contextual delivery"):
            scheduler.message_user(binding.agent_id, "a message the provider cannot receive")
        child = self.control.sessions[self.root.session_id].agents[binding.agent_id]
        self.assertEqual([], child.messages)

    def test_unleased_native_turn_is_refused_before_control_state_changes(self) -> None:
        scheduler = self._scheduler()
        self.assertTrue(scheduler._bind_agent(self.root))

        with self.assertRaises(ProtocolError) as refused:
            scheduler.adopt_native_turn(
                agent_id=self.root.agent_id,
                provider="fixture",
                thread_id=self.managed._threads[self.root.agent_id],
                turn_id="unleased-native-turn",
                cursor=0,
            )

        self.assertEqual("native-control-not-leased", refused.exception.code)

        self.assertEqual(AgentStatus.READY, self.root.status)
        self.assertIsNone(self.managed.control_turn_for_native(
            agent_id=self.root.agent_id,
            provider="fixture",
            native_turn_id="unleased-native-turn",
        ))

    def test_user_message_wakes_a_paused_live_child_with_user_attribution(self) -> None:
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="wait for a user decision",
            task_contract={},
        )
        scheduler = self._scheduler()
        scheduler.interrupt_agent(child.agent_id)

        result = scheduler.message_user(child.agent_id, "continue with the user decision")

        self.assertEqual("queued", result["status"])
        self.assertEqual(child.agent_id, result["agent_id"])
        self.assertNotIn(child.agent_id, scheduler._interrupted_agents)
        message = child.messages[-1]
        self.assertEqual(("user", "user", "continue with the user decision"),
                         (message.sender_id, message.kind, message.text))

    def test_user_message_rejects_terminal_child_instead_of_resurrecting_it(self) -> None:
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="finished work",
            task_contract={},
        )
        self.managed.complete_agent(child.agent_id, {"outcome": "done", "verified": True, "evidence": []})
        scheduler = self._scheduler()

        with self.assertRaisesRegex(ProtocolError, "terminal"):
            scheduler.message_user(child.agent_id, "continue anyway")
        self.assertEqual(AgentStatus.COMPLETED, child.status)

    def test_prompts_keep_context_telemetry_and_durable_session_instructions(self) -> None:
        self.root.usage = {
            "provider": "fixture",
            "tokens": {"total": 123},
            "context": {"window": 1000, "last": {"input": 800}},
            "cost": 0.4,
            "native": {"irrelevant": "not prompt telemetry"},
        }
        self.root.task_contract["instructions"] = "Use source evidence before making claims."
        scheduler = self._scheduler()

        prompt, _phase = scheduler._manager_prompt(self.root)

        context = json.loads(prompt.split("SELF_CONTEXT:\n", 1)[1].split("\nSESSION INSTRUCTIONS:", 1)[0])
        self.assertEqual("fixture", context["provider"])
        self.assertEqual(self.root.agent_id, context["agent_id"])
        self.assertEqual({"window": 1000, "last": {"input": 800}}, context["context"])
        self.assertNotIn("native", context)
        self.assertIn("Use source evidence before making claims.", prompt)

    def test_inspect_with_empty_agent_id_returns_the_callers_own_view(self) -> None:
        scheduler = self._scheduler()

        for arguments in ({"agent_id": "", "deep": False}, {"deep": False}):
            with self.subTest(arguments=arguments):
                result = scheduler._manager_handler(
                    self.root.agent_id,
                    "inspect",
                    arguments,
                    None,
                )

                self.assertTrue(result.success)
                self.assertEqual(self.root.agent_id, result.value["agent"]["agent_id"])

    def test_inspect_tool_description_names_self_as_agent_id(self) -> None:
        tools = VNextScheduler.manager_tools()
        inspect_tool = next(tool for tool in tools if tool["name"] == "inspect")

        self.assertIn('agent_id "self"', inspect_tool["description"])

    def test_binding_forwards_the_selected_agent_effort_to_its_runtime(self) -> None:
        self.root.effort = "medium"
        scheduler = self._scheduler()

        scheduler._bind_agent(self.root)

        thread_id = self.managed._threads[self.root.agent_id]
        self.assertEqual("medium", self.adapter.thread_efforts[thread_id])

    def test_terminal_root_waits_for_runtime_handler_to_drain(self) -> None:
        self.adapter.delay_root_terminal_return = True
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        outcome: dict[str, object] = {}

        def run_scheduler() -> None:
            try:
                outcome["result"] = scheduler.run()
            except BaseException as exc:  # pragma: no cover - surfaced below
                outcome["error"] = exc

        thread = threading.Thread(target=run_scheduler)
        thread.start()
        self.assertTrue(self.adapter.root_terminal_entered.wait(5))
        self.assertTrue(thread.is_alive())
        scheduler.cancel()
        self.adapter.root_terminal_release.set()
        thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual("accepted", outcome["result"]["decision"])

    def test_keep_alive_idle_observes_cancellation(self) -> None:
        """Closing an idle persistent conversation cannot leave its runner alive."""

        idle = threading.Event()
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, _agent, _data: idle.set()
                if kind == "conversation_idle"
                else None
            ),
        )
        self.managed.complete_root(
            self.root.agent_id,
            {"decision": "accepted", "summary": "first objective", "criteria": {}},
        )
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                scheduler.run(keep_alive=True)
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(idle.wait(2))
        scheduler.cancel()
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), SchedulerCancelled)

    def test_keep_alive_marks_an_ordinary_native_reply_idle_and_reuses_primary_thread(self) -> None:
        """Chat replies do not need complete_session before a user can continue."""

        self.adapter = OrdinaryReplyAdapter()
        idle = threading.Event()
        lifecycle: list[str] = []
        lifecycle_lock = threading.Lock()

        def record(kind, _agent, _data):
            with lifecycle_lock:
                lifecycle.append(kind)
            if kind == "conversation_idle":
                idle.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                scheduler.run(keep_alive=True)
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(idle.wait(2))
        self.assertEqual(AgentStatus.READY, self.root.status)
        self.assertNotIn("objective_completed", lifecycle)
        self.assertEqual(1, self.adapter._next_thread)
        self.assertEqual(1, self.adapter._next_turn)

        idle.clear()
        scheduler.message("continue the same conversation")
        self.assertTrue(idle.wait(2))
        self.assertEqual(1, self.adapter._next_thread)
        self.assertEqual(2, self.adapter._next_turn)

        scheduler.cancel()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), SchedulerCancelled)


    def test_a_granted_read_outside_the_workspace_is_accepted_and_recorded(self) -> None:
        """A standing grant covers a read the way it covers a write.

        The envelope is the one the Claude boundary builds, from `_tool_effect`
        and `_approval_detail`, so this fails again the moment that boundary
        stops forwarding a read.  The field run that found the defect saw no
        approval pair at all in runs/<session>.jsonl for the attempt.
        """

        scheduler, lifecycle = self._lifecycle_scheduler()
        worker = self._granted_worker(
            scheduler, "Compare a file in a sibling directory", "granted-read-thread"
        )
        self.assertEqual("granted", worker.approvals)
        lifecycle.clear()  # Examine only this approval after delegation.
        outside = "/private/tmp/outside-the-workspace/outside.txt"

        accepted = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "fixture:outside-read",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "granted-read-thread",
                    "turn": "granted-read-turn",
                    "request": "outside-read",
                },
                "correlation_attested": True,
                **_tool_effect("Read"),
                "event": _approval_detail("Read", None, {"file_path": outside}),
            },
        )

        self.assertEqual({"decision": "accept"}, accepted)
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [event_type for event_type, _data in lifecycle],
        )
        requested, resolved = (data for _event_type, data in lifecycle)
        self.assertEqual("read", requested["effect"])
        self.assertEqual(outside, requested["subject"])
        self.assertEqual("accept", resolved["decision"])
        self.assertEqual("standing-grant", resolved["resolver"])
        records = scheduler.native_approval_records()
        self.assertEqual(1, len(records))
        self.assertEqual("accept", records[0]["decision"])
        self.assertEqual(outside, records[0]["subject"])
        self.assertNotIn("reason", records[0])

    def test_an_ask_mode_read_outside_the_workspace_waits_for_its_manager(self) -> None:
        """Ask mode keeps its say over a read, as it does over a write."""

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own the read review",
            task_contract={"criteria": ["read reviewed"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Read one file outside the workspace",
            task_contract={"criteria": ["read reviewed"]},
            approvals="ask",
        )
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="ask-read-thread",
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle: list[tuple[str, dict]] = []
        requested = threading.Event()

        def record(event_type, _agent, data):
            lifecycle.append((event_type, dict(data)))
            if event_type == "approval_requested":
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:ask-read",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "ask-read-thread",
                        "turn": "ask-read-turn",
                        "request": "ask-read",
                    },
                    "correlation_attested": True,
                    **_tool_effect("Read"),
                    "event": _approval_detail("Read", None, {"file_path": "/etc/hosts"}),
                },
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        # The read is still waiting: no decision has been published.
        self.assertEqual({}, outcome)
        approval_id = lifecycle[-1][1]["approval_id"]
        self.assertEqual("read", lifecycle[-1][1]["effect"])
        self.assertEqual("/etc/hosts", lifecycle[-1][1]["subject"])
        scheduler.resolve_approval(
            approval_id, "accept", "bounded read", resolver="branch-manager"
        )
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual({"decision": "accept"}, outcome["value"])
        self.assertEqual("branch-manager", lifecycle[-1][1]["resolver"])

    def test_an_ask_mode_read_its_manager_refuses_says_the_manager_refused_it(self) -> None:
        """The one decline that may still be reported as the manager's."""

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Refuse one read",
            task_contract={"criteria": ["read refused"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Read one file outside the workspace",
            task_contract={"criteria": ["read refused"]},
            approvals="ask",
        )
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="refused-read-thread",
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle: list[tuple[str, dict]] = []
        requested = threading.Event()

        def record(event_type, _agent, data):
            lifecycle.append((event_type, dict(data)))
            if event_type == "approval_requested":
                requested.set()

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(lifecycle=record),
        )
        outcome: dict[str, object] = {}

        def review() -> None:
            outcome["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:refused-read",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "refused-read-thread",
                        "turn": "refused-read-turn",
                        "request": "refused-read",
                    },
                    "correlation_attested": True,
                    **_tool_effect("Read"),
                    "event": _approval_detail("Read", None, {"file_path": "/etc/hosts"}),
                },
            )

        thread = threading.Thread(target=review)
        thread.start()
        self.assertTrue(requested.wait(2))
        scheduler.resolve_approval(
            lifecycle[-1][1]["approval_id"], "decline", "out of scope",
            resolver="branch-manager",
        )
        thread.join(2)

        self.assertEqual(
            {"decision": "decline", "reason": _MANAGER_DECLINE_REASON},
            outcome["value"],
        )
        self.assertEqual(
            _MANAGER_DECLINE_REASON, scheduler.native_approval_records()[0]["reason"]
        )

    def test_the_reviewers_else_branch_records_the_request_and_names_itself(self) -> None:
        """A decline no manager made, and the record it has to leave.

        The branch declined and returned, so the attempt appeared nowhere in
        runs/<session>.jsonl and the worker was told its manager had refused
        it.  The pair below is the account of it, and the resolver names this
        boundary rather than a manager that was never asked.
        """

        scheduler, lifecycle = self._lifecycle_scheduler()
        worker = self._granted_worker(
            scheduler, "Call a tool this boundary cannot name", "unnamed-thread"
        )
        lifecycle.clear()  # Examine only this approval after delegation.

        declined = scheduler.review_approval(
            "approval/request",
            {
                "approval_reference": "fixture:unnamed-tool",
                "provider": "fixture",
                "provider_correlation": {
                    "session": "unnamed-thread",
                    "turn": "unnamed-turn",
                    "request": "unnamed-tool",
                },
                "correlation_attested": True,
                "event": {"tool": "SlashCommand", "path": "/private/tmp/anywhere"},
            },
        )

        self.assertEqual(
            {
                "decision": "decline",
                "reason": _BOUNDARY_DECLINE_REASONS["unnamed-effect"],
            },
            declined,
        )
        self.assertNotIn("manager declined", declined["reason"])
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [event_type for event_type, _data in lifecycle],
        )
        requested, resolved = (data for _event_type, data in lifecycle)
        self.assertEqual(worker.agent_id, requested["worker_agent"])
        self.assertEqual("unnamed", requested["effect"])
        self.assertEqual("/private/tmp/anywhere", requested["subject"])
        self.assertEqual("decline", resolved["decision"])
        self.assertEqual(_BOUNDARY_RESOLVER, resolved["resolver"])
        self.assertEqual(
            _BOUNDARY_DECLINE_REASONS["unnamed-effect"], resolved["reason"]
        )
        records = scheduler.native_approval_records()
        self.assertEqual(1, len(records))
        self.assertEqual(
            _BOUNDARY_DECLINE_REASONS["unnamed-effect"], records[0]["reason"]
        )

    def test_a_failed_manager_wake_still_leaves_the_approval_pair(self) -> None:
        """A decline the manager never saw, and the account it owes.

        Waking the manager raised a ProtocolError and the boundary returned
        straight away, so runs/<session>.jsonl held nothing for a read a
        worker really attempted. Every other decline leaves a requested and a
        resolved event; this one leaves the same pair, named after the
        boundary that decided it alone.
        """

        scheduler, lifecycle = self._lifecycle_scheduler()
        delegated = payload(
            scheduler._apply_manager_tool(
                self.root.agent_id,
                "delegate",
                {
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER,
                    "objective": "Read one file while its manager is unreachable",
                    "task_contract": {"criteria": ["observed"]},
                    "approvals": "ask",
                },
            )
        )
        worker = self.control.sessions[self.root.session_id].agents[
            delegated["agent_id"]
        ]
        self.managed.bind_thread(
            agent_id=worker.agent_id,
            thread_id="unreachable-manager-thread",
            start_result=policy(),
            tool_handler=None,
        )
        lifecycle.clear()

        def refuse(**_keywords):
            raise ProtocolError("unavailable", "the manager cannot be woken")

        with patch.object(
            self.control, "request_attention", side_effect=refuse
        ):
            declined = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:unreachable-manager",
                    "provider": "fixture",
                    "provider_correlation": {
                        "session": "unreachable-manager-thread",
                        "turn": "unreachable-manager-turn",
                        "request": "unreachable-manager",
                    },
                    "correlation_attested": True,
                    **_tool_effect("Read"),
                    "event": _approval_detail(
                        "Read", None, {"file_path": "/etc/hosts"}
                    ),
                },
            )

        self.assertEqual(
            {
                "decision": "decline",
                "reason": _BOUNDARY_DECLINE_REASONS["routing-failed"],
            },
            declined,
        )
        self.assertEqual(
            ["approval_requested", "approval_resolved"],
            [event_type for event_type, _data in lifecycle],
        )
        requested, resolved = (data for _event_type, data in lifecycle)
        self.assertEqual(worker.agent_id, requested["worker_agent"])
        self.assertEqual(self.root.agent_id, requested["manager_agent"])
        self.assertEqual("Read", requested["tool"])
        self.assertEqual("/etc/hosts", requested["subject"])
        self.assertEqual(requested["approval_id"], resolved["approval_id"])
        self.assertEqual("decline", resolved["decision"])
        self.assertEqual(_BOUNDARY_RESOLVER, resolved["resolver"])
        self.assertEqual(
            _BOUNDARY_DECLINE_REASONS["routing-failed"], resolved["reason"]
        )


class VNextGateEvidenceTests(unittest.TestCase):
    """The Phase 9 gates name facts. These check the facts are recorded.

    A frozen gate set that asks for something the receipt cannot show is not a
    gate, it is a wish. Each test here corresponds to one gate in the
    frozen Phase 9 cross-vendor canary gate set.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="record what the gates ask for",
            task_contract={"criteria": ["evidence recorded"]},
            session_id="gate-evidence-session",
        )
        self.adapter = ScriptedAdapter()
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return self.adapter

        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_p1_two_children_created_in_one_turn_share_a_parent_turn_id(self) -> None:
        """The whole claim of Phase 9, as a fact rather than a judgement."""

        turn_id = self.control.start_turn(self.root.agent_id)
        first = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="first lane",
            task_contract={},
        )
        second = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="second lane",
            task_contract={},
        )

        session = self.control.sessions[self.root.session_id]
        created = {
            event.agent_id: event.metadata.get("parent_turn_id")
            for event in session.events
            if event.event_type == "agent-created"
        }
        self.assertEqual(turn_id, created[first.agent_id])
        self.assertEqual(turn_id, created[second.agent_id])

    def test_p1_children_created_in_different_turns_do_not(self) -> None:
        first_turn = self.control.start_turn(self.root.agent_id)
        first = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="first lane",
            task_contract={},
        )
        self.control.finish_turn(self.root.agent_id)
        second_turn = self.control.start_turn(self.root.agent_id)
        second = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="second lane",
            task_contract={},
        )

        session = self.control.sessions[self.root.session_id]
        created = {
            event.agent_id: event.metadata.get("parent_turn_id")
            for event in session.events
            if event.event_type == "agent-created"
        }
        self.assertNotEqual(first_turn, second_turn)
        self.assertEqual(first_turn, created[first.agent_id])
        self.assertEqual(second_turn, created[second.agent_id])

    def test_p2_the_tools_a_manager_called_are_recorded_against_its_turn(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(self.root)
        turn = self.managed.start_turn(
            self.root.agent_id,
            prompt="root turn",
            effort="high",
            phase="root-initial",
        )
        scheduler._active_turns[self.root.agent_id] = turn

        scheduler._apply_manager_tool(
            self.root.agent_id,
            "delegate",
            {
                "role": AgentRole.BRANCH_MANAGER.value,
                "model_id": MANAGER,
                "objective": "a lane",
                "task_contract": {},
            },
        )
        scheduler._apply_manager_tool(
            self.root.agent_id, "inspect", {"agent_id": "self", "deep": False}
        )

        receipt = self.managed.receipt(status="passed")
        tools = [entry["tools"] for entry in receipt["turns"] if entry["tools"]]
        self.assertEqual([["delegate", "inspect"]], tools)

    def test_p3_two_agents_holding_turns_at_once_are_visible_as_an_overlap(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(self.root)
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        scheduler._bind_agent(child)
        root_turn = self.managed.start_turn(
            self.root.agent_id, prompt="root", effort="high", phase="root-initial"
        )
        child_turn = self.managed.start_turn(
            child.agent_id, prompt="branch", effort="high", phase="branch-initial"
        )
        # time.time() on Windows before Python 3.13 moves in steps of about
        # 16 ms, so two turns started back to back can share one reading and
        # overlap for zero seconds.  Let the overlap last longer than a step.
        time.sleep(0.05)

        overlaps = self.managed.concurrent_turn_intervals()

        self.assertEqual(1, len(overlaps))
        self.assertEqual(
            {self.root.agent_id, child.agent_id}, set(overlaps[0]["agents"])
        )
        self.assertGreater(overlaps[0]["seconds"], 0)
        del root_turn, child_turn

    def test_p3_turns_that_never_overlapped_report_no_overlap(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(self.root)
        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        scheduler._bind_agent(child)
        turn = self.managed.start_turn(
            self.root.agent_id, prompt="root", effort="high", phase="root-initial"
        )
        self.managed.wait_turn(turn)
        later = self.managed.start_turn(
            child.agent_id, prompt="branch", effort="high", phase="branch-initial"
        )
        self.managed.wait_turn(later)

        self.assertEqual([], self.managed.concurrent_turn_intervals())

    def test_p6_the_characters_the_delta_saved_are_counted(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(self.root)
        self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        self.managed.start_turn(
            self.root.agent_id, prompt="root", effort="high", phase="root-initial"
        )
        self.control.finish_turn(self.root.agent_id)
        record = self.control.sessions[self.root.session_id].agents[self.root.agent_id]

        self.assertEqual(0, self.managed.context_saved_characters)
        scheduler._manager_prompt(record)
        self.assertEqual(0, self.managed.context_saved_characters)
        scheduler._manager_prompt(record)

        saved = self.managed.context_saved_characters
        self.assertGreater(saved, 0)
        self.assertEqual(saved, self.managed.receipt(status="passed")["context_saved_characters"])

    def test_the_receipt_says_which_scope_each_agent_ran_under(self) -> None:
        worker_parent = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="a lane",
            task_contract={},
        )
        self.control.spawn_agent(
            requester_id=worker_parent.agent_id,
            parent_agent_id=worker_parent.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="bounded work",
            task_contract={},
            workspace_scope="private",
        )

        receipt = self.managed.receipt(status="passed")

        scopes = sorted(entry["workspace_scope"] for entry in receipt["hierarchy"])
        self.assertEqual(["private", "shared", "shared"], scopes)
        private_entry = next(
            entry for entry in receipt["hierarchy"] if entry["workspace_scope"] == "private"
        )
        self.assertTrue(private_entry["workspace_path"])
        self.assertFalse(Path(private_entry["workspace_path"]).is_absolute())

    def test_the_receipt_carries_no_wall_clock_stamp(self) -> None:
        """Durations, never timestamps: a receipt may not date the run."""

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        scheduler._bind_agent(self.root)
        turn = self.managed.start_turn(
            self.root.agent_id, prompt="root", effort="high", phase="root-initial"
        )
        self.managed.wait_turn(turn)

        receipt = self.managed.receipt(status="passed")

        for entry in receipt["turns"]:
            self.assertNotIn("started_at", entry)
            self.assertNotIn("finished_at", entry)
            self.assertIsNotNone(entry["duration_seconds"])


class VNextWorktreeWorkspaceTests(unittest.TestCase):
    """The worktree scope, exercised against real git rather than a fake.

    Git is a hard dependency of this scope and of no other, so a stub would
    prove nothing about the one thing that can actually go wrong.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        # Resolved: a CI runner hands out an 8.3 short path, and the scope
        # resolves its roots, so an unresolved workspace skews the overhead.
        self.workspace = Path(self.temp.name).resolve()
        self._git("init", "-b", "main")
        self._git("config", "user.email", "canary@example.invalid")
        self._git("config", "user.name", "canary")
        (self.workspace / "tracked.txt").write_text("committed content\n", encoding="utf-8")
        self._git("add", "tracked.txt")
        self._git("commit", "-m", "seed")

        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="exercise the worktree scope",
            task_contract={"criteria": ["worktree honoured"]},
            session_id="worktree-session",
        )
        self.adapter = ScriptedAdapter()
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return self.adapter

        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )

    def tearDown(self) -> None:
        try:
            self.managed.close_worktrees()
        except Exception:
            pass
        try:
            self._git("worktree", "prune")
        except Exception:
            pass
        self.temp.cleanup()

    def _git(self, *args: str, cwd=None) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.workspace),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(f"git {args} failed: {completed.stderr}")
        return completed.stdout

    def _worker(self, scope: str):
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own one branch",
            task_contract={},
        )
        return self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="do bounded work",
            task_contract={},
            workspace_scope=scope,
        )

    def test_a_worktree_child_gets_a_real_checkout_of_the_project(self) -> None:
        worker = self._worker("worktree")

        root = self.managed.ensure_workspace(worker.agent_id)

        self.assertTrue(root.is_dir())
        self.assertTrue((root / ".git").exists())
        # The whole point against private: the project is actually there.
        self.assertEqual(
            "committed content\n",
            (root / "tracked.txt").read_text(encoding="utf-8"),
        )
        self.assertIn(
            self.managed.worktree_branch_for(worker.agent_id),
            self._git("worktree", "list"),
        )

    def test_a_private_child_still_gets_an_empty_directory(self) -> None:
        worker = self._worker("private")

        root = self.managed.ensure_workspace(worker.agent_id)

        self.assertTrue(root.is_dir())
        self.assertEqual([], list(root.iterdir()))
        self.assertEqual("", self.managed.worktree_branch_for(worker.agent_id))

    def test_the_child_cannot_see_work_the_parent_has_not_committed(self) -> None:
        """The sharp edge, pinned so nobody rediscovers it in a live run."""

        (self.workspace / "uncommitted.txt").write_text("not committed\n", encoding="utf-8")
        (self.workspace / "tracked.txt").write_text("edited but not committed\n", encoding="utf-8")
        worker = self._worker("worktree")

        root = self.managed.ensure_workspace(worker.agent_id)

        self.assertFalse((root / "uncommitted.txt").exists())
        self.assertEqual(
            "committed content\n",
            (root / "tracked.txt").read_text(encoding="utf-8"),
        )

    def test_a_checkout_left_untouched_is_removed_with_its_branch(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        branch = self.managed.worktree_branch_for(worker.agent_id)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertFalse(outcome["changed"])
        self.assertFalse(outcome["retained"])
        self.assertFalse(root.exists())
        self.assertNotIn(branch, self._git("branch", "--list", branch))

    def test_a_checkout_that_was_changed_is_kept_and_reported(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        branch = self.managed.worktree_branch_for(worker.agent_id)
        (root / "produced.txt").write_text("worker output\n", encoding="utf-8")

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertTrue(root.exists())
        self.assertEqual(branch, outcome["branch"])
        self.assertIn("Nothing merges it for you", outcome["note"])
        self.assertIn(branch, self._git("branch", "--list", branch))

    def test_a_committed_change_counts_as_a_change(self) -> None:
        """A tidy child that commits its work must not look like an idle one."""

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "produced.txt").write_text("worker output\n", encoding="utf-8")
        self._git("add", "produced.txt", cwd=root)
        self._git("commit", "-m", "worker work", cwd=root)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertFalse(outcome["uncommitted_changes"])

    def test_the_reported_path_stays_relative_to_the_session_workspace(self) -> None:
        worker = self._worker("worktree")
        self.managed.ensure_workspace(worker.agent_id)
        (self.managed.ensure_workspace(worker.agent_id) / "x.txt").write_text("x", encoding="utf-8")

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertFalse(Path(outcome["path"]).is_absolute())
        self.assertNotIn(str(self.workspace), outcome["path"])
        self.assertEqual(
            outcome["path"], self.managed.relative_workspace_for(worker.agent_id)
        )

    def test_two_worktree_siblings_writing_one_filename_do_not_clobber(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own one branch",
            task_contract={},
        )
        first = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="first",
            task_contract={},
            workspace_scope="worktree",
        )
        second = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="second",
            task_contract={},
            workspace_scope="worktree",
        )

        first_root = self.managed.ensure_workspace(first.agent_id)
        second_root = self.managed.ensure_workspace(second.agent_id)
        (first_root / "tracked.txt").write_text("from first\n", encoding="utf-8")
        (second_root / "tracked.txt").write_text("from second\n", encoding="utf-8")

        self.assertNotEqual(first_root, second_root)
        self.assertEqual("from first\n", (first_root / "tracked.txt").read_text(encoding="utf-8"))
        self.assertEqual("from second\n", (second_root / "tracked.txt").read_text(encoding="utf-8"))
        # The parent's own copy is untouched by either.
        self.assertEqual(
            "committed content\n",
            (self.workspace / "tracked.txt").read_text(encoding="utf-8"),
        )
        self.assertNotEqual(
            self.managed.worktree_branch_for(first.agent_id),
            self.managed.worktree_branch_for(second.agent_id),
        )

    def test_resolving_a_path_creates_nothing(self) -> None:
        """Asking where a child works must not build it a workspace.

        The receipt asks every agent for its path, and it is built after the
        session has closed and settled its checkouts. While resolution created,
        writing the report recreated the very worktrees the run had just
        removed and left them on disk permanently, while reporting none
        outstanding.
        """

        worker = self._worker("worktree")

        root = self.managed.workspace_for(worker.agent_id)
        relative = self.managed.relative_workspace_for(worker.agent_id)

        self.assertFalse(root.exists())
        self.assertTrue(relative)
        self.assertEqual([], self.managed.close_worktrees())

    def test_settling_then_reporting_does_not_resurrect_a_checkout(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        branch = self.managed.worktree_branch_for(worker.agent_id)
        self.assertTrue(root.exists())

        outcome = self.managed.worktree_outcome(worker.agent_id)
        self.assertFalse(outcome["retained"])
        self.managed.receipt(status="passed")

        self.assertFalse(root.exists())
        self.assertNotIn(branch, self._git("branch", "--list", branch))

    def test_two_threads_asking_at_once_create_one_checkout(self) -> None:
        """The scheduler really does ask twice, from two threads.

        The main loop binds a new child while the manager's own handler thread
        is still resolving that child's path for its tool reply. With creation
        outside the lock, both reached `git worktree add` and the loser died on
        "reference already exists", taking the whole tree with it.
        """

        worker = self._worker("worktree")
        results: list = []
        barrier = threading.Barrier(2)

        def race() -> None:
            try:
                barrier.wait(5)
                results.append(self.managed.ensure_workspace(worker.agent_id))
            except BaseException as exc:  # pragma: no cover - reported below
                results.append(exc)

        threads = [threading.Thread(target=race) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)

        self.assertEqual(2, len(results))
        self.assertFalse(
            [item for item in results if isinstance(item, BaseException)],
            f"a racing caller failed: {results}",
        )
        self.assertEqual(1, len(set(results)))

    def test_output_the_project_ignores_is_still_work(self) -> None:
        """A child told to write into an ignored directory has not done nothing.

        Plain `git status --porcelain` is built to hide files, and everything
        it hides is something a Worker may have been told to produce. This
        project's own ignore list covers reports/, data/ and prototypes/.
        """

        (self.workspace / ".gitignore").write_text("reports/\n*.log\n", encoding="utf-8")
        self._git("add", ".gitignore")
        self._git("commit", "-m", "ignore rules")
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "reports").mkdir()
        (root / "reports" / "analysis.md").write_text("the deliverable\n", encoding="utf-8")

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertTrue((root / "reports" / "analysis.md").exists())

    def test_a_repository_that_hides_untracked_files_does_not_hide_the_work(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        self._git("config", "status.showUntrackedFiles", "no")
        (root / "produced.txt").write_text("worker output\n", encoding="utf-8")

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue((root / "produced.txt").exists())

    def test_work_a_tidy_child_stashed_is_not_treated_as_nothing(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "produced.txt").write_text("worker output\n", encoding="utf-8")
        self._git("add", "produced.txt", cwd=root)
        self._git("stash", "push", "-m", "worker stash", cwd=root)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["stashed_changes"])
        self.assertIn("stash", outcome["note"])

    def test_a_checkout_that_cannot_be_removed_keeps_its_branch(self) -> None:
        """Losing the branch orphans the directory beyond git's reach.

        The three cleanup commands used to run unconditionally with every
        failure swallowed, so a checkout still held open kept its directory,
        lost the branch that pointed at it, and was reported as removed.
        """

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        branch = self.managed.worktree_branch_for(worker.agent_id)
        original = self.managed._git

        def refuse_removal(*args, **kwargs):
            if args[:2] == ("worktree", "remove"):
                raise ManagedSessionError("git worktree failed in the session workspace")
            return original(*args, **kwargs)

        self.managed._git = refuse_removal
        try:
            outcome = self.managed.worktree_outcome(worker.agent_id)
        finally:
            self.managed._git = original

        self.assertFalse(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertIn("could not be removed", outcome["note"])
        self.assertTrue(root.exists())
        self.assertIn(branch, self._git("branch", "--list", branch))

    def test_a_checkout_that_vanished_does_not_cost_the_session_its_receipt(self) -> None:
        """A missing working directory raises OSError, not ManagedSessionError."""

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        shutil.rmtree(root)

        settled = self.managed.close_worktrees()

        self.assertEqual(1, len(settled))
        self.assertTrue(settled[0]["retained"])
        receipt = self.managed.receipt(status="passed")
        self.assertEqual(1, len(receipt["worktrees"]))
        # Through the alias table, like every other receipt section: a raw
        # agent identifier may not appear in a receipt.
        self.assertEqual("worker", receipt["worktrees"][0]["agent_id"])

    def test_the_administrative_directory_is_hidden_from_the_parent_repository(self) -> None:
        """A sibling running `git add -A` must not commit a live checkout.

        Nothing ignored it, so the parent reported it as untracked and the
        ordinary thing a Worker does in the shared workspace committed
        submodule gitlinks into the user's real history, pointing at branches
        this run would later delete.
        """

        worker = self._worker("worktree")
        self.managed.ensure_workspace(worker.agent_id)

        self._git("add", "-A")
        staged = self._git("diff", "--cached", "--name-only")

        self.assertNotIn(".vnext", staged)
        completed = subprocess.run(
            ["git", "check-ignore", ".vnext/x"],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode)


    def test_any_agent_may_choose_an_isolated_workspace(self) -> None:
        for scope in ("private", "worktree"):
            child = self.control.spawn_agent(
                requester_id=self.root.agent_id,
                parent_agent_id=self.root.agent_id,
                role=AgentRole.BRANCH_MANAGER,
                model_id=MANAGER,
                objective="a manager outside the tree",
                task_contract={},
                workspace_scope=scope,
            )
            self.assertEqual(scope, child.workspace_scope)

    def test_an_unknown_scope_is_still_refused(self) -> None:
        with self.assertRaises(ProtocolError) as refused:
            self._worker("copy")
        self.assertEqual("invalid-workspace-scope", refused.exception.code)

    def test_a_replacement_keeps_the_scope_it_replaces(self) -> None:
        worker = self._worker("worktree")
        branch_id = worker.parent_agent_id
        self.control.block_agent(worker.agent_id, "stopped")

        replacement = self.control.replace_agent(
            requester_id=branch_id,
            agent_id=worker.agent_id,
            model_id=WORKER,
            revised_task_contract={},
        )

        self.assertEqual("worktree", replacement.workspace_scope)
        self.assertNotEqual(worker.workspace_dir, replacement.workspace_dir)

    def test_closing_the_session_settles_a_checkout_nobody_completed(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        self.assertTrue(root.exists())

        settled = self.managed.close_worktrees()

        self.assertEqual(1, len(settled))
        self.assertFalse(settled[0]["retained"])
        self.assertFalse(root.exists())


    def test_a_child_whose_workspace_fails_is_blocked_without_killing_the_tree(self) -> None:
        """Blocking is only a fix if the caller stops.

        _bind_agent blocked the child and returned, and the very next line
        started its turn anyway, which raised "agent has no bound runtime
        thread" straight out of the scheduler loop. The block landed and the
        run died before any manager could read it.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        worker = self._worker("worktree")
        branch_id = worker.parent_agent_id
        original = self.managed.ensure_workspace

        def refuse(agent_id):
            if agent_id == worker.agent_id:
                raise ManagedSessionError("git worktree failed in the session workspace")
            return original(agent_id)

        self.managed.ensure_workspace = refuse
        try:
            bound = scheduler._bind_agent(worker)
        finally:
            self.managed.ensure_workspace = original

        self.assertFalse(bound)
        self.assertEqual(AgentStatus.BLOCKED, worker.status)
        self.assertIn("workspace could not be prepared", worker.blocker)
        # And its manager was told, which is the whole point of blocking.
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(
            ["child-blocked"],
            [wake["reason"] for wake in session.wakes[branch_id]],
        )

    def test_a_blocked_child_is_published_and_not_only_recorded(self) -> None:
        """A blocked worker went on being drawn as running.

        block_agent only changed the control plane, so the run record and the
        status line kept showing a worker that had died of a 400 as working.
        Across every run record on this machine the string "blocked" had never
        once been written.
        """

        published = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(record_agent=published.append),
        )
        worker = self._worker("worktree")
        scheduler._block(worker, "runtime turn ended as failed")

        self.assertEqual(AgentStatus.BLOCKED, worker.status)
        self.assertEqual([worker], published)

    def test_a_child_whose_provider_will_not_start_is_blocked_not_fatal(self) -> None:
        """A dead provider is one worker's problem, not the session's.

        A Claude worker whose bridge timed out raised out of the scheduler loop
        and failed the session. Every later manager tool call in that chat
        answered only that the session had failed, and the user had to restart
        the terminal to get vNext back.
        """

        recorded = []
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
            hooks=SchedulerHooks(
                lifecycle=lambda kind, agent, data: recorded.append((kind, agent, data)),
            ),
        )
        worker = self._worker("worktree")
        branch_id = worker.parent_agent_id
        original = self.managed.adapter_for_agent

        def refuse(agent):
            if agent.agent_id == worker.agent_id:
                try:
                    raise TimeoutError("no answer in 90s")
                except TimeoutError as exc:
                    raise RuntimeError(
                        "Claude bridge timed out waiting for start_thread"
                    ) from exc
            return original(agent)

        self.managed.adapter_for_agent = refuse
        try:
            bound = scheduler._bind_agent(worker)
        finally:
            self.managed.adapter_for_agent = original

        self.assertFalse(bound)
        self.assertEqual(AgentStatus.BLOCKED, worker.status)
        self.assertIn("runtime could not be started", worker.blocker)
        self.assertIn("timed out waiting for start_thread", worker.blocker)
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(
            ["child-blocked"],
            [wake["reason"] for wake in session.wakes[branch_id]],
        )

        # The blocker is one sentence for the manager; the run record is where
        # a person finds out what the sentence meant.
        reports = [data for kind, _, data in recorded if kind == "provider_error"]
        self.assertEqual(1, len(reports))
        self.assertEqual("start_thread", reports[0]["phase"])
        self.assertEqual("TimeoutError: no answer in 90s", reports[0]["cause"])
        self.assertIn("start_thread", reports[0]["traceback"])

    def test_binding_an_agent_that_succeeds_reports_that_it_bound(self) -> None:
        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        worker = self._worker("shared")

        self.assertTrue(scheduler._bind_agent(worker))
        self.assertTrue(scheduler._bind_agent(worker))

    def test_the_path_budget_covers_a_child_past_the_ninth(self) -> None:
        """The overhead assumed a one-digit ordinal.

        The ordinal counts private and worktree children together, so the tenth
        is agent-10 and costs one character more than the budget allowed. The
        manager was told the scope was available and the child then failed at
        bind, which used to be fatal.
        """

        from vnext.vnext_managed_session import (
            _WORKTREE_ORDINAL_DIGITS,
            _WORKTREE_PATH_OVERHEAD,
        )

        # A separator, ".vnext", a separator, a twelve-character token, a
        # separator and "agent-", then the digits.
        self.assertEqual(27 + _WORKTREE_ORDINAL_DIGITS, _WORKTREE_PATH_OVERHEAD)
        self.assertGreaterEqual(_WORKTREE_ORDINAL_DIGITS, 2)

        worker = self._worker("worktree")
        session = self.control.sessions[self.root.session_id]
        session.private_workspace_count = 41
        worker.workspace_dir = "agent-42"
        root = self.managed.workspace_for(worker.agent_id)

        overhead = len(str(root)) - len(str(self.workspace))
        self.assertLessEqual(overhead, _WORKTREE_PATH_OVERHEAD)

    def test_one_childs_stash_is_not_read_as_another_childs(self) -> None:
        """agent-1 is a prefix of agent-10, and the match was a substring."""

        first = self._worker("worktree")
        first_root = self.managed.ensure_workspace(first.agent_id)
        session = self.control.sessions[self.root.session_id]
        session.private_workspace_count = 9
        tenth = self.control.spawn_agent(
            requester_id=first.parent_agent_id,
            parent_agent_id=first.parent_agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="the tenth",
            task_contract={},
            workspace_scope="worktree",
        )
        tenth_root = self.managed.ensure_workspace(tenth.agent_id)
        self.assertTrue(
            self.managed.worktree_branch_for(first.agent_id)
            in self.managed.worktree_branch_for(tenth.agent_id)
        )
        (tenth_root / "produced.txt").write_text("work\n", encoding="utf-8")
        self._git("add", "produced.txt", cwd=tenth_root)
        self._git("stash", "push", "-m", "wip", cwd=tenth_root)

        first_outcome = self.managed.worktree_outcome(first.agent_id)
        tenth_outcome = self.managed.worktree_outcome(tenth.agent_id)

        self.assertFalse(first_outcome.get("stashed_changes", False))
        self.assertFalse(first_outcome["retained"])
        self.assertFalse(first_root.exists())
        self.assertTrue(tenth_outcome["stashed_changes"])
        self.assertTrue(tenth_outcome["retained"])

    def test_a_stash_nobody_can_be_credited_with_keeps_the_checkout(self) -> None:
        """A stash taken from a detached HEAD reads "On (no branch)".

        The work exists in the shared git directory, so deleting the checkout
        and telling the manager the child left nothing means nobody goes
        looking for it. Cannot-tell resolves to changed.
        """

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        head = self._git("rev-parse", "HEAD", cwd=root).strip()
        self._git("checkout", "--detach", head, cwd=root)
        (root / "produced.txt").write_text("work\n", encoding="utf-8")
        self._git("add", "produced.txt", cwd=root)
        self._git("stash", "push", "-m", "wip", cwd=root)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertTrue(outcome["change_undetermined"])
        self.assertTrue(root.exists())

    def test_an_adopted_checkout_still_gets_a_base_to_compare_against(self) -> None:
        """Without a base the head comparison drops out of the change rule.

        A child that committed all of its work has a clean tree, so a checkout
        adopted with no base was judged to have left nothing and its fully
        committed deliverable was destroyed.
        """

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "deliverable.txt").write_text("the work\n", encoding="utf-8")
        self._git("add", "deliverable.txt", cwd=root)
        self._git("commit", "-m", "the work", cwd=root)
        # Forget the checkout the way a fresh session would, then re-adopt it.
        self.managed._worktrees.clear()
        self.managed._worktree_bases.clear()
        self.managed._worktree_creation_locks.clear()

        readopted = self.managed.ensure_workspace(worker.agent_id)
        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertEqual(root, readopted)
        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertTrue((root / "deliverable.txt").exists())

    def test_a_private_child_also_hides_the_administrative_directory(self) -> None:
        """The shield was on the worktree path only, and the hole was private.

        It stages plain files rather than gitlinks, which is milder than the
        worktree case, but it is the same directory in the same repository.
        """

        worker = self._worker("private")

        self.managed.ensure_workspace(worker.agent_id)

        self._git("add", "-A")
        self.assertNotIn(".vnext", self._git("diff", "--cached", "--name-only"))

    def test_the_worktree_description_warns_about_submodules(self) -> None:
        """git worktree add does not initialise them, so the child sees nothing."""

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        delegate = next(
            tool for tool in scheduler.manager_tools() if tool["name"] == "delegate"
        )
        described = delegate["inputSchema"]["properties"]["workspace"]["description"]

        self.assertIn("Submodules arrive empty", described)


    def test_a_workspace_that_cannot_be_made_blocks_the_child_and_the_run_goes_on(self) -> None:
        """The unpatched trigger: a FILE named .vnext in the workspace.

        Preflight has nothing to say about it, the delegate succeeds, and the
        directory creation then fails at bind time. Blocking the child was
        supposed to let its manager route around it. Instead the manager was
        never released -- the wake landed while its own turn was still in
        flight, and nothing moves a child out of BLOCKED -- so the run died as
        a deadlock with a sibling already completed.
        """

        scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        worker = self._worker("worktree")
        branch_id = worker.parent_agent_id
        (self.workspace / ".vnext").write_text("not a directory\n", encoding="utf-8")

        bound = scheduler._bind_agent(worker)

        self.assertFalse(bound)
        self.assertEqual(AgentStatus.BLOCKED, worker.status)
        # And the manager can still park-and-be-released, or rather cannot park
        # at all, which is what keeps the run alive.
        self.control.await_agents(branch_id, [worker.agent_id])
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(AgentStatus.READY, session.agents[branch_id].status)

    def test_a_stash_made_the_ordinary_way_is_attributed_to_its_child(self) -> None:
        """`git stash` with no -m writes "WIP on <branch>", which never matched."""

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "produced.txt").write_text("work\n", encoding="utf-8")
        self._git("add", "produced.txt", cwd=root)
        self._git("stash", cwd=root)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertTrue(outcome["stashed_changes"])
        self.assertFalse(outcome["change_undetermined"])
        self.assertIn("stash", outcome["note"])

    def test_an_unrelated_user_stash_does_not_retain_every_checkout(self) -> None:
        """git stash list is repository-global.

        One ordinary stash on main, made months ago by the user, used to make
        every worktree child in every session "cannot tell", so nothing was
        ever removed and each child left a full checkout of the project behind.
        """

        (self.workspace / "tracked.txt").write_text("edited\n", encoding="utf-8")
        self._git("stash")
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertFalse(outcome["changed"])
        self.assertFalse(outcome["retained"])
        self.assertFalse(root.exists())

    def test_a_commit_subject_cannot_misattribute_a_stash(self) -> None:
        """The tip commit's subject is part of the line and can contain anything.

        Splitting on ": On " anywhere matched inside the subject and named a
        branch called "call runbook", so a real stash fell into the ignore
        bucket and its checkout was destroyed.
        """

        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "note.txt").write_text("note\n", encoding="utf-8")
        self._git("add", "note.txt", cwd=root)
        self._git("commit", "-m", "docs: On call runbook", cwd=root)
        (root / "produced.txt").write_text("work\n", encoding="utf-8")
        self._git("add", "produced.txt", cwd=root)
        self._git("stash", cwd=root)

        outcome = self.managed.worktree_outcome(worker.agent_id)

        self.assertTrue(outcome["changed"])
        self.assertTrue(outcome["retained"])
        self.assertTrue(outcome["stashed_changes"])
        self.assertTrue(root.exists())

    def test_a_stash_list_that_cannot_be_read_means_cannot_tell(self) -> None:
        worker = self._worker("worktree")
        self.managed.ensure_workspace(worker.agent_id)
        original = self.managed._git

        def refuse_stash(*args, **kwargs):
            if args[:1] == ("stash",):
                raise ManagedSessionError("git stash failed in the session workspace")
            return original(*args, **kwargs)

        self.managed._git = refuse_stash
        try:
            outcome = self.managed.worktree_outcome(worker.agent_id)
        finally:
            self.managed._git = original

        self.assertTrue(outcome["change_undetermined"])
        self.assertTrue(outcome["retained"])

    def _completing_scheduler(self) -> VNextScheduler:
        return VNextScheduler(
            managed=self.managed, root=self.root, cancellation=RunCancellation(),
        )

    def test_a_worktree_child_that_changed_something_hands_its_manager_the_branch(self) -> None:
        # The delegate tool promises the path and the branch when the child
        # finishes. Nothing called the settle step after completion moved to
        # complete_agent, so the manager heard of neither until session close.
        scheduler = self._completing_scheduler()
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        (root / "deliverable.txt").write_text("work\n", encoding="utf-8")
        branch = self.managed.worktree_branch_for(worker.agent_id)

        answer = scheduler._manager_handler(
            worker.agent_id,
            "complete_agent",
            {"outcome": "done", "verified": True, "evidence": ["wrote deliverable.txt"]},
            None,
        )
        self.assertTrue(answer.success)
        self.assertEqual(answer.value["workspace"]["branch"], branch)
        self.assertTrue(answer.value["workspace"]["retained"])

        collected = scheduler._manager_handler(
            worker.parent_agent_id, "await_children", {"agent_ids": [worker.agent_id]}, None
        )
        (child,) = collected.value["children"]
        self.assertEqual(child["workspace"]["branch"], branch)
        self.assertEqual(child["workspace"]["path"], root.relative_to(self.workspace).as_posix())
        self.assertTrue(root.exists())

    def test_a_worktree_child_that_changed_nothing_loses_its_checkout_at_completion(self) -> None:
        scheduler = self._completing_scheduler()
        worker = self._worker("worktree")
        root = self.managed.ensure_workspace(worker.agent_id)
        branch = self.managed.worktree_branch_for(worker.agent_id)

        answer = scheduler._manager_handler(
            worker.agent_id,
            "complete_agent",
            {"outcome": "nothing to change", "verified": True, "evidence": ["read it"]},
            None,
        )

        self.assertFalse(answer.value["workspace"]["retained"])
        self.assertFalse(root.exists())
        self.assertEqual(self._git("branch", "--list", branch).strip(), "")

    def test_a_branch_manager_with_a_worktree_settles_it_on_complete_branch(self) -> None:
        scheduler = self._completing_scheduler()
        manager = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="own one branch",
            task_contract={},
            workspace_scope="worktree",
        )
        root = self.managed.ensure_workspace(manager.agent_id)

        answer = scheduler._manager_handler(
            manager.agent_id,
            "complete_branch",
            {"outcome": "nothing to change", "verified": True, "evidence": ["read it"]},
            None,
        )

        self.assertFalse(answer.value["workspace"]["retained"])
        self.assertFalse(root.exists())

    def test_a_plain_directory_at_a_worktree_root_is_not_taken_for_a_checkout(self) -> None:
        # A crash or a hand-tidy can leave the directory without the checkout.
        # Adopting it handed the child an empty folder and its manager a branch
        # git had never heard of.
        worker = self._worker("worktree")
        root = self.managed.workspace_for(worker.agent_id)
        root.mkdir(parents=True)

        got = self.managed.ensure_workspace(worker.agent_id)

        self.assertEqual(
            (got / "tracked.txt").read_text(encoding="utf-8"), "committed content\n"
        )
        branch = self.managed.worktree_branch_for(worker.agent_id)
        self.assertIn(branch, self._git("branch", "--list", branch))

    def test_a_directory_with_files_at_a_worktree_root_is_refused(self) -> None:
        worker = self._worker("worktree")
        root = self.managed.workspace_for(worker.agent_id)
        root.mkdir(parents=True)
        (root / "left.txt").write_text("someone's file\n", encoding="utf-8")

        with self.assertRaisesRegex(ManagedSessionError, "not a git checkout"):
            self.managed.ensure_workspace(worker.agent_id)
        self.assertTrue((root / "left.txt").exists())

    def test_another_repository_at_a_worktree_root_is_refused(self) -> None:
        """R25 verify: a git repository of its own passed the checkout test."""

        worker = self._worker("worktree")
        root = self.managed.workspace_for(worker.agent_id)
        root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
        (root / "other.txt").write_text("different project\n", encoding="utf-8")

        with self.assertRaisesRegex(ManagedSessionError, "not a git checkout"):
            self.managed.ensure_workspace(worker.agent_id)
        self.assertTrue((root / "other.txt").exists())

    def test_a_runtime_that_will_not_start_keeps_its_secrets_from_the_manager(self) -> None:
        """R25 verify: the bind failure put the adapter's raw words in the blocker."""

        worker = self._worker("shared")
        scheduler = self._completing_scheduler()

        def rejected(agent):
            raise RuntimeError("provider at /home/someone/private/key.json; api_key=PLAINSECRET123")  # gitleaks:allow (fake key: tests redaction)

        self.managed.adapter_for_agent = rejected
        self.assertFalse(scheduler._bind_agent(worker))
        self.assertNotIn("PLAINSECRET123", worker.blocker)
        self.assertNotIn("/home/someone/private", worker.blocker)
        self.assertIn("could not be started", worker.blocker)

    def test_a_restored_child_keeps_a_root_of_its_own(self) -> None:
        private = self._worker("private")
        before = self.managed.workspace_for(private.agent_id)
        bare = self._worker("private")
        session = self.control.sessions[self.root.session_id]
        tree = []
        for agent in session.agents.values():
            row = {
                "agent_id": agent.agent_id,
                "parent_agent_id": agent.parent_agent_id or None,
                "role": agent.role.value,
                "model": agent.model_id,
                "status": "ready" if agent.agent_id in (private.agent_id, bare.agent_id) else agent.status.value,
                "workspace_scope": agent.workspace_scope,
                "approvals": agent.approvals,
            }
            if agent.agent_id != bare.agent_id:
                row["workspace_dir"] = agent.workspace_dir
            tree.append(row)

        self.control.restore_terminal_tree(self.root.session_id, tree)

        self.assertEqual(self.managed.workspace_for(private.agent_id), before)
        fresh = self.managed.workspace_for(bare.agent_id)
        self.assertNotEqual(fresh, self.managed.workspace)
        self.assertNotEqual(fresh, before)
        nxt = self._worker("private")
        self.assertNotIn(
            self.managed.workspace_for(nxt.agent_id), (before, fresh, self.managed.workspace)
        )

    def test_a_restore_refuses_a_root_that_leaves_the_session(self) -> None:
        private = self._worker("private")
        session = self.control.sessions[self.root.session_id]
        tree = [
            {
                "agent_id": agent.agent_id,
                "parent_agent_id": agent.parent_agent_id or None,
                "role": agent.role.value,
                "model": agent.model_id,
                "status": "ready" if agent.agent_id == private.agent_id else agent.status.value,
                "workspace_scope": agent.workspace_scope,
                "workspace_dir": "../outside" if agent.agent_id == private.agent_id else agent.workspace_dir,
            }
            for agent in session.agents.values()
        ]
        with self.assertRaises(ProtocolError):
            self.control.restore_terminal_tree(self.root.session_id, tree)


class VNextWorktreePreflightTests(unittest.TestCase):
    """A worktree the workspace cannot give is refused where a manager hears it.

    Every one of these used to surface as an exception out of the bind path,
    which killed the root, the branch manager and every sibling while the
    manager that asked for the scope heard nothing at all.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self) -> None:
        try:
            self.temp.cleanup()
        except OSError:
            pass

    def _session(self):
        control = OrchestrationControlPlane(registry())
        root = control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="preflight",
            task_contract={},
            session_id="preflight-session",
        )
        adapter = ScriptedAdapter()
        managed = VNextManagedSession(
            control=control,
            adapter=adapter,
            session_id=root.session_id,
            runtime_effects=RuntimeEffectJournal(self.workspace),
            adapter_for_agent=lambda agent: adapter,
        )
        return control, root, managed

    def _git(self, *args: str) -> None:
        completed = subprocess.run(
            ["git", *args], cwd=str(self.workspace), capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise AssertionError(f"git {args} failed: {completed.stderr}")

    def _delegate_worktree(self, managed, requester_id):
        return managed.spawn_from_manager(
            requester_id=requester_id,
            role=AgentRole.BRANCH_MANAGER,
            arguments={
                "role": AgentRole.BRANCH_MANAGER.value,
                "model_id": MANAGER,
                "objective": "a lane",
                "task_contract": {},
                "workspace": "worktree",
            },
        )

    def test_a_workspace_that_is_not_a_repository_is_refused_with_a_reason(self) -> None:
        _control, root, managed = self._session()

        with self.assertRaises(ManagedSessionError) as refused:
            self._delegate_worktree(managed, root.agent_id)

        self.assertIn("git repository", str(refused.exception))
        self.assertIn("shared or private", str(refused.exception))

    def test_a_repository_with_no_commit_yet_is_refused_with_a_reason(self) -> None:
        self._git("init", "-b", "main")
        _control, root, managed = self._session()

        with self.assertRaises(ManagedSessionError) as refused:
            self._delegate_worktree(managed, root.agent_id)

        self.assertIn("has none yet", str(refused.exception))

    def test_a_workspace_inside_a_larger_repository_is_refused(self) -> None:
        """Otherwise the child gets a checkout of everything above it."""

        self._git("init", "-b", "main")
        (self.workspace / "outer.txt").write_text("outer\n", encoding="utf-8")
        self._git("add", "outer.txt")
        self._git("-c", "user.email=a@b.invalid", "-c", "user.name=a", "commit", "-m", "seed")
        inner = self.workspace / "subproject"
        inner.mkdir()
        control = OrchestrationControlPlane(registry())
        root = control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=inner,
            objective="preflight",
            task_contract={},
            session_id="nested-session",
        )
        adapter = ScriptedAdapter()
        managed = VNextManagedSession(
            control=control,
            adapter=adapter,
            session_id=root.session_id,
            runtime_effects=RuntimeEffectJournal(inner),
            adapter_for_agent=lambda agent: adapter,
        )

        with self.assertRaises(ManagedSessionError) as refused:
            self._delegate_worktree(managed, root.agent_id)

        self.assertIn("larger repository", str(refused.exception))

    def test_a_workspace_path_too_long_for_this_host_is_refused(self) -> None:
        """The ceiling is a Windows MAX_PATH artifact and stays on Windows.

        macOS and Linux carry a path limit above a thousand characters. The
        rule ran on every host, so it refused worktrees git creates here
        without complaint.
        """

        from vnext import vnext_managed_session

        self._git("init", "-b", "main")
        (self.workspace / "seed.txt").write_text("seed\n", encoding="utf-8")
        self._git("add", "seed.txt")
        self._git("-c", "user.email=a@b.invalid", "-c", "user.name=a", "commit", "-m", "seed")
        _control, root, managed = self._session()
        real_workspace = managed.workspace
        # Stand the session workspace at a path past the measured budget without
        # having to create one that deep on disk.
        managed.workspace = Path("C:/" + ("x" * 400))

        with patch.object(vnext_managed_session.os, "name", "nt"):
            with self.assertRaises(ManagedSessionError) as refused:
                self._delegate_worktree(managed, root.agent_id)
        self.assertIn("shorter workspace path", str(refused.exception))

        # The same path on a POSIX host gets past the ceiling and as far as
        # git, which is the only thing that decides whether a checkout can
        # be made. Git refuses this one because it does not exist, and that
        # is a different sentence about a different problem.
        with patch.object(vnext_managed_session.os, "name", "posix"):
            with self.assertRaises(ManagedSessionError) as refused_by_git:
                managed.worktree_preflight()
        self.assertNotIn("shorter workspace path", str(refused_by_git.exception))

        managed.workspace = real_workspace

    def test_shared_and_private_do_not_need_git_at_all(self) -> None:
        _control, root, managed = self._session()

        for scope in ("shared", "private"):
            child = managed.spawn_from_manager(
                requester_id=root.agent_id,
                role=AgentRole.BRANCH_MANAGER,
                arguments={
                    "role": AgentRole.BRANCH_MANAGER.value,
                    "model_id": MANAGER,
                    "objective": "a lane",
                    "task_contract": {},
                    "workspace": scope if scope == "shared" else "shared",
                },
            )
            self.assertIsNotNone(child)


class MessageDeliveryAdapter(ScriptedAdapter):
    """A tree whose managers would each finish in a single turn.

    That is the interleaving in which a queued message is easiest to lose: the
    manager never builds a second prompt, so nothing ever reads its inbox.
    """

    def __init__(self, *, with_branch: bool = False) -> None:
        super().__init__()
        self.with_branch = with_branch
        self.before_root_completion = None
        self.before_branch_completion = None
        self.root_hook_fired = False
        self.branch_hook_fired = False
        self.root_prompts: list[str] = []
        self.branch_prompts: list[str] = []
        self.root_completions: list[dict] = []
        self.branch_completions: list[dict] = []
        self.branch_ids: list[str] = []

    def wait_turn(self, handle, *, timeout=300):
        del timeout
        handler = self.handlers[handle.thread_id]
        role = self.roles[handle.thread_id]
        prompt = self.prompts[handle.thread_id][-1]
        if role == AgentRole.ROOT_MANAGER.value:
            self.root_prompts.append(prompt)
            if self.with_branch and not self.branch_ids:
                created = payload(
                    handler(
                        "delegate",
                        {
                            "role": AgentRole.BRANCH_MANAGER.value,
                            "model_id": MANAGER,
                            "objective": "Own the only branch",
                            "task_contract": {"criteria": ["branch complete"]},
                        },
                        None,
                    )
                )
                self.assert_success(created)
                self.branch_ids.append(created["agent_id"])
                self.assert_success(
                    payload(handler("await_children", {"agent_ids": self.branch_ids}, None))
                )
                return {"status": "completed", "final_response": "root delegated"}
            if self.before_root_completion is not None and not self.root_hook_fired:
                self.root_hook_fired = True
                self.before_root_completion()
            self.root_completions.append(
                payload(
                    handler(
                        "complete_session",
                        {"decision": "accepted", "summary": "done", "criteria": {}},
                        None,
                    )
                )
            )
            return {"status": "completed", "final_response": "root turn"}
        self.branch_prompts.append(prompt)
        if self.before_branch_completion is not None and not self.branch_hook_fired:
            self.branch_hook_fired = True
            self.before_branch_completion(self.branch_ids[0])
        self.branch_completions.append(
            payload(
                handler(
                    "complete_branch",
                    {"outcome": "completed", "verified": True, "evidence": ["no workers needed"]},
                    None,
                )
            )
        )
        return {"status": "completed", "final_response": "branch turn"}


class VNextMessageDeliveryTests(unittest.TestCase):
    """A queued message must never be buried by a completion that follows it."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="Deliver every queued message before completing",
            task_contract={"criteria": ["messages delivered"]},
            session_id="message-delivery-session",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _build(self, *, with_branch: bool = False) -> VNextScheduler:
        self.adapter = MessageDeliveryAdapter(with_branch=with_branch)
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return self.adapter

        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )
        return VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

    def test_root_completion_is_refused_until_the_queued_message_is_read(self) -> None:
        scheduler = self._build()
        self.adapter.before_root_completion = lambda: scheduler.steer(
            "stop using Luna, switch to Flash"
        )

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        self.assertEqual(2, len(self.adapter.root_completions))
        refused, accepted = self.adapter.root_completions
        self.assertFalse(refused["success"])
        self.assertEqual("unread-messages", refused["error_code"])
        self.assertIn("MESSAGES", refused["error"])
        self.assertTrue(accepted["success"])
        # The refusal bought exactly one more turn, and that turn carried the
        # message the root would otherwise never have seen.
        self.assertEqual(2, len(self.adapter.root_prompts))
        messages = _messages_of(self.adapter.root_prompts[1])
        self.assertEqual(
            [("user", "stop using Luna, switch to Flash")],
            [(item["sender_id"], item["text"]) for item in messages],
        )
        self.assertEqual(AgentStatus.COMPLETED, self.root.status)

    def test_branch_completion_is_refused_until_the_queued_message_is_read(self) -> None:
        scheduler = self._build(with_branch=True)

        def queue_for_branch(branch_id: str) -> None:
            self.control.message_agent(self.root.agent_id, branch_id, "re-run the failing check")

        self.adapter.before_branch_completion = queue_for_branch

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        self.assertEqual(2, len(self.adapter.branch_completions))
        refused, accepted = self.adapter.branch_completions
        self.assertFalse(refused["success"])
        self.assertEqual("unread-messages", refused["error_code"])
        self.assertTrue(accepted["success"])
        self.assertEqual(2, len(self.adapter.branch_prompts))
        messages = _messages_of(self.adapter.branch_prompts[1])
        self.assertEqual(
            [(self.root.agent_id, "re-run the failing check")],
            [(item["sender_id"], item["text"]) for item in messages],
        )
        session = self.control.sessions[self.root.session_id]
        branch = session.agents[self.adapter.branch_ids[0]]
        self.assertEqual(AgentStatus.COMPLETED, branch.status)

    def test_completion_with_no_unread_message_is_unchanged(self) -> None:
        scheduler = self._build()

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        self.assertEqual(1, len(self.adapter.root_prompts))
        self.assertEqual([True], [item["success"] for item in self.adapter.root_completions])

    def test_steering_a_completed_conversation_reopens_the_primary(self) -> None:
        scheduler = self._build()
        self.managed.complete_root(
            self.root.agent_id,
            {"decision": "accepted", "summary": "already finished", "criteria": {}},
        )

        result = scheduler.steer("stop using Luna, switch to Flash")

        self.assertEqual("steered", result["status"])
        self.assertEqual(AgentStatus.READY, self.root.status)
        self.assertEqual(["stop using Luna, switch to Flash"], [item.text for item in self.root.messages])

    def test_manager_steer_survives_a_child_runtime_that_refuses_in_place_steering(self) -> None:
        scheduler = self._build(with_branch=True)
        scheduler._bind_agent(self.root)
        root_turn = self.managed.start_turn(
            self.root.agent_id,
            prompt="root turn",
            effort="high",
            phase="root-initial",
        )
        scheduler._active_turns[self.root.agent_id] = root_turn
        branch = self.managed.spawn_from_manager(
            requester_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            arguments={
                "role": AgentRole.BRANCH_MANAGER.value,
                "model_id": MANAGER,
                "objective": "Own the only branch",
                "task_contract": {"criteria": ["branch complete"]},
            },
        )
        scheduler._bind_agent(branch)
        branch_turn = self.managed.start_turn(
            branch.agent_id,
            prompt="branch turn",
            effort="high",
            phase="branch-initial",
        )
        scheduler._active_turns[branch.agent_id] = branch_turn

        def refuse(handle, text):
            raise ClaudeRuntimeError("this runtime does not steer a live turn")

        self.adapter.steer = refuse

        result = payload(
            scheduler._manager_handler(
                self.root.agent_id,
                "steer",
                {"agent_id": branch.agent_id, "message": "prefer the cheaper model"},
                None,
            )
        )

        scheduler._active_turns.clear()
        self.assertTrue(result["success"])
        self.assertEqual("queued-for-next-turn", result["delivery"])
        self.assertEqual(
            [(self.root.agent_id, "prefer the cheaper model")],
            [(item.sender_id, item.text) for item in branch.messages],
        )



class WorkerMessageAdapter(ScriptedAdapter):
    """A tree whose Worker finishes in a single turn.

    That is where a queued message is easiest to lose.  A Worker has no
    completion tool, so nothing refuses on its behalf: the scheduler completes
    it the moment its turn ends, and an inbox nobody read goes with it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_ids: list[str] = []
        self.worker_ids: list[str] = []
        self.worker_prompts: list[str] = []
        self.branch_prompts: list[str] = []
        self.during_worker_turn = None
        self.worker_hook_fired = False

    def wait_turn(self, handle, *, timeout=300):
        del timeout
        handler = self.handlers[handle.thread_id]
        role = self.roles[handle.thread_id]
        prompt = self.prompts[handle.thread_id][-1]
        if role == AgentRole.ROOT_MANAGER.value:
            if not self.branch_ids:
                created = payload(
                    handler(
                        "delegate",
                        {
                            "role": AgentRole.BRANCH_MANAGER.value,
                            "model_id": MANAGER,
                            "objective": "Own the only branch",
                            "task_contract": {"criteria": ["branch complete"]},
                        },
                        None,
                    )
                )
                self.assert_success(created)
                self.branch_ids.append(created["agent_id"])
                self.assert_success(
                    payload(handler("await_children", {"agent_ids": self.branch_ids}, None))
                )
                return {"status": "completed", "final_response": "root delegated"}
            self.assert_success(
                payload(
                    handler(
                        "complete_session",
                        {"decision": "accepted", "summary": "done", "criteria": {}},
                        None,
                    )
                )
            )
            return {"status": "completed", "final_response": "root turn"}
        if role == AgentRole.BRANCH_MANAGER.value:
            self.branch_prompts.append(prompt)
            if not self.worker_ids:
                created = payload(
                    handler(
                        "delegate",
                        {
                            "role": AgentRole.WORKER.value,
                            "model_id": WORKER,
                            "objective": "Write the marker",
                            "task_contract": {"criteria": ["marker written"]},
                        },
                        None,
                    )
                )
                self.assert_success(created)
                self.worker_ids.append(created["agent_id"])
                self.assert_success(
                    payload(handler("await_children", {"agent_ids": self.worker_ids}, None))
                )
                return {"status": "completed", "final_response": "branch delegated"}
            self.assert_success(
                payload(
                    handler(
                        "complete_branch",
                        {"outcome": "completed", "verified": True, "evidence": ["worker done"]},
                        None,
                    )
                )
            )
            return {"status": "completed", "final_response": "branch turn"}
        self.worker_prompts.append(prompt)
        if self.during_worker_turn is not None and not self.worker_hook_fired:
            self.worker_hook_fired = True
            self.during_worker_turn(self.worker_ids[0])
        completion = payload(
            handler(
                "complete_agent",
                {
                    "outcome": "worker task complete",
                    "verified": True,
                    "evidence": ["worker turn"],
                },
                None,
            )
        )
        # A message received during this turn deliberately refuses completion;
        # the next turn carries that message and completes explicitly.
        if not completion["success"] and completion["error_code"] != "unread-messages":
            raise AssertionError(completion)
        return {"status": "completed", "final_response": "worker turn"}


class ReportedBlockAdapter(WorkerMessageAdapter):
    """A Worker that stops for a token and its manager that hands it over.

    The worker's first turn calls report_blocked.  The branch manager, woken
    by the block, answers with send_message and then retry; the worker's next
    turn reads the answer and completes.
    """

    ANSWER = "the token is in the RELEASE_TOKEN variable"

    def __init__(self) -> None:
        super().__init__()
        self.worker_threads: list[str] = []
        self.blockers_seen: list[str] = []

    def wait_turn(self, handle, *, timeout=300):
        handler = self.handlers[handle.thread_id]
        role = self.roles[handle.thread_id]
        prompt = self.prompts[handle.thread_id][-1]
        if role == AgentRole.WORKER.value:
            self.worker_prompts.append(prompt)
            self.worker_threads.append(handle.thread_id)
            if len(self.worker_prompts) == 1:
                self.assert_success(payload(handler(
                    "report_blocked", {"reason": "needs a GitHub token"}, None,
                )))
                return {"status": "completed", "final_response": "blocked"}
            self.assert_success(payload(handler("complete_agent", {
                "outcome": "released", "verified": True, "evidence": ["token used"],
            }, None)))
            return {"status": "completed", "final_response": "worker done"}
        if role == AgentRole.BRANCH_MANAGER.value and self.worker_ids:
            worker_id = self.worker_ids[0]
            agent = payload(handler("inspect", {"agent_id": worker_id, "deep": False}, None))["agent"]
            if agent["status"] == AgentStatus.BLOCKED.value:
                self.branch_prompts.append(prompt)
                self.blockers_seen.append(agent["blocker"])
                self.assert_success(payload(handler(
                    "send_message", {"agent_id": worker_id, "message": self.ANSWER}, None,
                )))
                self.assert_success(payload(handler("retry", {
                    "agent_id": worker_id, "task_contract": {"criteria": ["marker written"]},
                }, None)))
                self.assert_success(payload(handler("await_children", {"agent_ids": [worker_id]}, None)))
                return {"status": "completed", "final_response": "answered the worker"}
        return super().wait_turn(handle, timeout=timeout)


class VNextWorkerMessageDeliveryTests(unittest.TestCase):
    """A Worker must read what it was sent before the scheduler completes it."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="Deliver every queued message before completing",
            task_contract={"criteria": ["messages delivered"]},
            session_id="worker-message-session",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _build(self, adapter_class=None) -> VNextScheduler:
        self.adapter = (adapter_class or WorkerMessageAdapter)()
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return self.adapter

        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )
        return VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

    def test_a_one_turn_worker_reads_its_message_before_it_is_completed(self) -> None:
        scheduler = self._build()

        def queue_for_worker(worker_id: str) -> None:
            self.control.message_agent(
                self.adapter.branch_ids[0], worker_id, "use the other marker path"
            )

        self.adapter.during_worker_turn = queue_for_worker

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        # The Worker would have finished in one turn.  The guarantee bought it a
        # second one, and that turn is where the message was handed over.
        self.assertEqual(2, len(self.adapter.worker_prompts))
        delivered = _messages_of(self.adapter.worker_prompts[1])
        self.assertEqual(
            [(self.adapter.branch_ids[0], "use the other marker path")],
            [(item["sender_id"], item["text"]) for item in delivered],
        )
        self.assertIn("[WORKER RESUME]", self.adapter.worker_prompts[1])
        self.assertEqual(
            [], _messages_of(self.adapter.worker_prompts[0])
        )
        session = self.control.sessions[self.root.session_id]
        worker = session.agents[self.adapter.worker_ids[0]]
        self.assertEqual(AgentStatus.COMPLETED, worker.status)

    def test_a_message_survives_a_runtime_that_refuses_to_start_the_turn(self) -> None:
        """Building the prompt is not delivering it.

        The offset advances when the prompt is built, but a runtime that
        refuses to start the turn never shows that prompt to anybody. The agent
        is blocked, which is not terminal, so nothing announces the loss either
        -- the message is simply gone, and the retry that follows never sees it.
        """

        scheduler = self._build()
        session = self.control.sessions[self.root.session_id]
        record = session.agents[self.root.agent_id]
        scheduler._bind_agent(self.root)
        # A first turn has to be behind it: an opening prompt carries no
        # MESSAGES section, so only a resume prompt can spend the queue.
        self.control.start_turn(self.root.agent_id)
        self.control.finish_turn(self.root.agent_id)
        self.control.send_user_message(self.root.agent_id, "STOP: the spec changed")

        def refuse(*args, **kwargs):
            raise RuntimeError("this runtime will not start a turn")

        self.adapter.start_turn = refuse
        with self.assertRaises(RuntimeError):
            scheduler._start_turn(record)

        self.assertEqual(
            1,
            scheduler._unread_message_count(record),
            "a message was marked read by a turn that never ran",
        )
        self.assertEqual(AgentStatus.BLOCKED, record.status)

    def test_a_reported_block_resumes_the_same_session_with_the_answer(self) -> None:
        """The manager's answer reaches the worker's next turn on its own thread."""

        scheduler = self._build(ReportedBlockAdapter)

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        self.assertEqual(
            ["reported by the worker: needs a GitHub token. Answer with send_message, "
             "then retry it, to resume it on the same session"],
            self.adapter.blockers_seen,
        )
        self.assertEqual(2, len(self.adapter.worker_prompts))
        delivered = _messages_of(self.adapter.worker_prompts[1])
        self.assertEqual(
            [(self.adapter.branch_ids[0], ReportedBlockAdapter.ANSWER)],
            [(item["sender_id"], item["text"]) for item in delivered],
        )
        self.assertIn("[WORKER RESUME]", self.adapter.worker_prompts[1])
        # One provider thread for the worker, used by both turns: the same
        # session and its context, and no fresh agent.
        worker_threads = [
            thread for thread, role in self.adapter.roles.items()
            if role == AgentRole.WORKER.value
        ]
        self.assertEqual(1, len(worker_threads))
        self.assertEqual(worker_threads * 2, self.adapter.worker_threads)
        session = self.control.sessions[self.root.session_id]
        self.assertEqual(
            AgentStatus.COMPLETED, session.agents[self.adapter.worker_ids[0]].status,
        )

    def test_a_worker_with_no_message_still_completes_in_one_turn(self) -> None:
        scheduler = self._build()

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        self.assertEqual(1, len(self.adapter.worker_prompts))

    def test_a_message_racing_the_completion_is_refused_by_the_control_plane(self) -> None:
        """The pre-check can be beaten; the guard under the session lock cannot.

        Reporting zero unread simulates a message that lands after the scheduler
        has decided to complete.  ``complete_agent`` compares the delivered
        offset under the same lock ``message_agent`` takes, so the completion is
        refused and the Worker still gets its turn.
        """

        scheduler = self._build()
        real_count = scheduler._unread_message_count
        blinded: dict[str, str | None] = {"worker": None}

        def blind(agent):
            if agent.agent_id == blinded["worker"]:
                blinded["worker"] = None
                return 0
            return real_count(agent)

        def queue_for_worker(worker_id: str) -> None:
            blinded["worker"] = worker_id
            self.control.message_agent(
                self.adapter.branch_ids[0], worker_id, "stop and re-read the contract"
            )

        scheduler._unread_message_count = blind
        self.adapter.during_worker_turn = queue_for_worker

        result = scheduler.run()

        self.assertEqual("accepted", result["decision"])
        self.assertEqual(2, len(self.adapter.worker_prompts))
        delivered = _messages_of(self.adapter.worker_prompts[1])
        self.assertEqual(["stop and re-read the contract"], [item["text"] for item in delivered])

    def test_a_queue_larger_than_one_batch_is_delivered_rather_than_marked_read(self) -> None:
        scheduler = self._build()
        session = self.control.sessions[self.root.session_id]
        record = session.agents[self.root.agent_id]

        for index in range(MESSAGE_BATCH + 3):
            self.control.send_user_message(self.root.agent_id, f"note {index}")

        first = scheduler._new_messages(record)
        second = scheduler._new_messages(record)
        third = scheduler._new_messages(record)

        self.assertEqual(MESSAGE_BATCH, len(first))
        self.assertEqual(3, len(second))
        self.assertEqual([], third)
        self.assertEqual(
            [f"note {index}" for index in range(MESSAGE_BATCH + 3)],
            [item["text"] for item in first + second],
        )
        self.assertEqual(0, scheduler._unread_message_count(record))



class LateIdentityAdapter(ScriptedAdapter):
    """A runtime that binds its provider session partway through a turn.

    The Claude bridge does exactly this: a reservation starts unbound and the
    provider's session identity arrives with the first message of the turn.
    Every approval the runtime relays after that point carries an attested
    provider correlation, while the control plane's cached attestation is still
    the unbound snapshot taken when the turn started.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bound_threads: set[str] = set()

    def thread_identity_attestation(self, thread_id):
        if thread_id not in self.bound_threads:
            return {
                "runtime_thread": thread_id,
                "provider": "fixture",
                "bound": False,
                "provider_session": None,
                "binding_phase": "reserved",
                "synthetic": False,
            }
        return {
            "runtime_thread": thread_id,
            "provider": "fixture",
            "bound": True,
            "provider_session": f"native-{thread_id}",
            "binding_phase": "attested",
            "synthetic": False,
        }


class VNextLateIdentityApprovalTests(unittest.TestCase):
    """An approval must reach its manager even when identity binds mid-turn."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="Route an approval whose identity bound mid-turn",
            task_contract={"criteria": ["approval routed"]},
            session_id="late-identity-session",
        )
        self.adapter = LateIdentityAdapter()
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return self.adapter

        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )
        self.scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _worker_mid_turn(self):
        """Bring a Worker to the state a live approval arrives in."""

        self.scheduler._bind_agent(self.root)
        branch = self.managed.spawn_from_manager(
            requester_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            arguments={
                "role": AgentRole.BRANCH_MANAGER.value,
                "model_id": MANAGER,
                "objective": "Own the only branch",
                "task_contract": {"criteria": ["branch complete"]},
            },
        )
        self.scheduler._bind_agent(branch)
        worker = self.managed.spawn_from_manager(
            requester_id=branch.agent_id,
            role=AgentRole.WORKER,
            arguments={
                "role": AgentRole.WORKER.value,
                "model_id": WORKER,
                "objective": "Write the marker",
                "task_contract": {"criteria": ["marker written"]},
            },
        )
        self.scheduler._bind_agent(worker)
        turn = self.managed.start_turn(
            worker.agent_id,
            prompt="worker turn",
            effort="high",
            phase="worker-execution",
        )
        self.scheduler._active_turns[worker.agent_id] = turn
        # The provider binds its session only now, after the snapshot the turn
        # start took.  This is the window every first native approval falls in.
        self.adapter.bound_threads.add(turn.runtime.thread_id)
        return branch, worker, turn

    def _envelope(self, turn) -> dict:
        return {
            "approval_reference": "fixture:tool-use-1",
            "provider": "fixture",
            "effect": "modify",
            "correlation_attested": True,
            "provider_correlation": {
                "session": f"native-{turn.runtime.thread_id}",
                "turn": turn.runtime.turn_id,
                "request": "tool-use-1",
            },
        }

    def test_an_approval_bound_mid_turn_still_resolves_to_its_worker(self) -> None:
        _branch, worker, turn = self._worker_mid_turn()
        cached = self.managed._identity_attestations[worker.agent_id]
        self.assertEqual(
            "reserved",
            cached.get("binding_phase"),
            "the snapshot this test exists for was not taken",
        )

        routed = self.scheduler._approval_worker(self._envelope(turn), "fixture")

        self.assertEqual(
            worker.agent_id,
            routed,
            "an approval whose identity bound mid-turn found no route to its worker",
        )
        self.scheduler._active_turns.clear()

    def test_a_session_no_agent_is_attested_to_is_still_refused(self) -> None:
        _branch, _worker, turn = self._worker_mid_turn()
        envelope = self._envelope(turn)
        envelope["provider_correlation"]["session"] = "native-someone-else"

        routed = self.scheduler._approval_worker(envelope, "fixture")

        self.assertIsNone(
            routed, "an unattested provider session was given a route anyway"
        )
        self.scheduler._active_turns.clear()

    def test_rebinding_leaves_an_idle_agent_alone(self) -> None:
        _branch, worker, turn = self._worker_mid_turn()
        root_identity = dict(self.managed._identity_attestations[self.root.agent_id])

        self.scheduler._approval_worker(self._envelope(turn), "fixture")

        self.assertEqual(
            root_identity,
            self.managed._identity_attestations[self.root.agent_id],
            "an agent with no active turn had its identity re-read",
        )
        self.assertEqual(
            "attested",
            self.managed._identity_attestations[worker.agent_id].get("binding_phase"),
            "the worker holding the turn was not re-read",
        )
        self.scheduler._active_turns.clear()



class VNextStrandedMessageTests(unittest.TestCase):
    """Mail a terminal transition cannot deliver must still be reported."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="Report mail no transition can deliver",
            task_contract={"criteria": ["nothing vanishes"]},
            session_id="stranded-message-session",
        )
        self.branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own the only branch",
            task_contract={"criteria": ["branch complete"]},
        )
        self.worker = self.control.spawn_agent(
            requester_id=self.branch.agent_id,
            parent_agent_id=self.branch.agent_id,
            role=AgentRole.WORKER,
            model_id=WORKER,
            objective="Write the marker",
            task_contract={"criteria": ["marker written"]},
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _events(self, event_type: str) -> list:
        session = self.control.sessions[self.root.session_id]
        return [event for event in session.events if event.event_type == event_type]

    def test_replacing_a_worker_reports_the_mail_it_never_read(self) -> None:
        self.control.message_agent(
            self.branch.agent_id, self.worker.agent_id, "use the other marker path"
        )

        self.control.replace_agent(
            requester_id=self.branch.agent_id,
            agent_id=self.worker.agent_id,
            model_id=WORKER,
            revised_task_contract={"criteria": ["marker written"]},
        )

        stranded = self._events("messages-undelivered")
        self.assertEqual(1, len(stranded), "a replaced Worker's inbox vanished quietly")
        self.assertEqual(self.worker.agent_id, stranded[0].agent_id)
        self.assertEqual(
            {"count": 1, "reason": "replaced"},
            stranded[0].metadata,
        )

    def test_cancelling_a_worker_reports_the_mail_it_never_read(self) -> None:
        self.control.message_agent(
            self.branch.agent_id, self.worker.agent_id, "stop, the contract changed"
        )

        self.control.cancel_agent(
            requester_id=self.branch.agent_id, agent_id=self.worker.agent_id
        )

        stranded = self._events("messages-undelivered")
        self.assertEqual(1, len(stranded), "a cancelled Worker's inbox vanished quietly")
        self.assertEqual(
            {"count": 1, "reason": "cancelled"},
            stranded[0].metadata,
        )

    def test_mail_already_delivered_is_not_reported_as_stranded(self) -> None:
        self.control.message_agent(
            self.branch.agent_id, self.worker.agent_id, "a message that was read"
        )
        self.worker.delivered_message_count = len(self.worker.messages)

        self.control.cancel_agent(
            requester_id=self.branch.agent_id, agent_id=self.worker.agent_id
        )

        self.assertEqual([], self._events("messages-undelivered"))

    def test_a_terminal_agent_is_refused_for_being_terminal(self) -> None:
        """Liveness outranks the message guard, and the order is load-bearing.

        A completion refused with the wrong reason sends the caller into a
        recovery that cannot work: the scheduler would try to give another turn
        to an agent that no longer has one.
        """

        self.control.message_agent(
            self.branch.agent_id, self.worker.agent_id, "unread when it died"
        )
        self.control.cancel_agent(
            requester_id=self.branch.agent_id, agent_id=self.worker.agent_id
        )

        with self.assertRaises(ProtocolError) as caught:
            self.control.complete_agent(
                self.worker.agent_id, {"outcome": "completed"}, require_messages_read=True
            )

        self.assertEqual("terminal-agent", caught.exception.code)

    def test_a_refused_completion_records_no_progress_and_wakes_nobody(self) -> None:
        self.control.start_turn(self.worker.agent_id)
        self.control.await_agents(self.branch.agent_id, [self.worker.agent_id])
        self.control.message_agent(
            self.branch.agent_id, self.worker.agent_id, "read me first"
        )
        session = self.control.sessions[self.root.session_id]
        before = len(session.events)

        with self.assertRaises(ProtocolError) as caught:
            self.control.complete_agent(
                self.worker.agent_id,
                {"outcome": "completed"},
                require_messages_read=True,
                progress={
                    "activity": "completed bounded execution",
                    "progress": "should never be recorded",
                },
            )

        self.assertEqual("unread-messages", caught.exception.code)
        self.assertNotEqual("should never be recorded", self.worker.latest_progress)
        self.assertEqual(
            before,
            len(session.events),
            "a refused completion still emitted events",
        )
        self.assertEqual(AgentStatus.AWAITING_WORKERS, self.branch.status)

    def test_a_worker_that_went_terminal_meanwhile_does_not_crash_the_scheduler(
        self,
    ) -> None:
        """The recovery path must survive losing its Worker mid-recovery.

        The unread pre-check and the turn it buys are not one atomic step.  A
        cancel landing between them leaves a Worker with no turn to give, and
        `finish_turn` refuses from a terminal state.  Left unhandled that
        refusal escapes `_handle_turn_finished` and takes the whole scheduler
        down -- a worse outcome than the dropped message it was recovering from.
        """

        adapter = ScriptedAdapter()
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return adapter

        managed = VNextManagedSession(
            control=self.control,
            adapter=adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )
        scheduler = VNextScheduler(
            managed=managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        self.control.message_agent(
            self.branch.agent_id, self.worker.agent_id, "read me before you finish"
        )
        self.control.cancel_agent(
            requester_id=self.branch.agent_id, agent_id=self.worker.agent_id
        )

        scheduler._worker_reads_messages(self.worker)

        self.assertEqual(AgentStatus.CANCELLED, self.worker.status)
        self.assertEqual(1, len(self._events("messages-undelivered")))



class WorkspacePartitionAdapter(ScriptedAdapter):
    """Two sibling Workers that both write the same filename, concurrently.

    Each Worker writes into whatever root it was handed, which is the only thing
    that decides whether the second one destroys the first one's work.  The
    writes are held at a barrier so they genuinely overlap rather than happening
    to be ordered.
    """

    def __init__(self, *, workspace_scope: str) -> None:
        super().__init__()
        self.workspace_scope = workspace_scope
        self.branch_ids: list[str] = []
        self.worker_ids: list[str] = []
        self.thread_workspaces: dict[str, str] = {}
        self.written: list[Path] = []
        self._barrier = threading.Barrier(2, timeout=10)
        self._write_lock = threading.Lock()

    def start_thread(self, *, workspace=None, **kwargs):
        thread_id, result = super().start_thread(**kwargs)
        with self._lock:
            self.thread_workspaces[thread_id] = str(workspace)
        return thread_id, result

    def wait_turn(self, handle, *, timeout=300):
        del timeout
        handler = self.handlers[handle.thread_id]
        role = self.roles[handle.thread_id]
        if role == AgentRole.ROOT_MANAGER.value:
            if not self.branch_ids:
                created = payload(
                    handler(
                        "delegate",
                        {
                            "role": AgentRole.BRANCH_MANAGER.value,
                            "model_id": MANAGER,
                            "objective": "Own the only branch",
                            "task_contract": {"criteria": ["branch complete"]},
                        },
                        None,
                    )
                )
                self.assert_success(created)
                self.branch_ids.append(created["agent_id"])
                self.assert_success(
                    payload(handler("await_children", {"agent_ids": self.branch_ids}, None))
                )
                return {"status": "completed", "final_response": "root delegated"}
            self.assert_success(
                payload(
                    handler(
                        "complete_session",
                        {"decision": "accepted", "summary": "done", "criteria": {}},
                        None,
                    )
                )
            )
            return {"status": "completed", "final_response": "root turn"}
        if role == AgentRole.BRANCH_MANAGER.value:
            if not self.worker_ids:
                for index in (1, 2):
                    created = payload(
                        handler(
                            "delegate",
                            {
                                "role": AgentRole.WORKER.value,
                                "model_id": WORKER,
                                "objective": f"Write the marker, attempt {index}",
                                "task_contract": {"criteria": ["marker written"]},
                                "workspace": self.workspace_scope,
                            },
                            None,
                        )
                    )
                    self.assert_success(created)
                    self.worker_ids.append(created["agent_id"])
                self.assert_success(
                    payload(handler("await_children", {"agent_ids": self.worker_ids}, None))
                )
                return {"status": "completed", "final_response": "branch delegated"}
            # A material-status wake can arrive while a sibling is still
            # running, so the branch checks before it claims the work is done.
            terminal = {"completed", "failed", "cancelled", "replaced"}
            statuses = []
            for worker_id in self.worker_ids:
                view = payload(handler("inspect", {"agent_id": worker_id, "deep": False}, None))
                self.assert_success(view)
                statuses.append(view["agent"]["status"])
            if not all(status in terminal for status in statuses):
                waited = payload(
                    handler("await_children", {"agent_ids": self.worker_ids}, None)
                )
                if waited.get("success") and not waited.get("settled"):
                    return {"status": "completed", "final_response": "branch waiting"}
                # The last sibling can finish between the inspect and the await.
                # That is a legal race, not a failure, and the branch completes.
                if waited.get("error_code") != "no-active-children":
                    self.assert_success(waited)
            self.assert_success(
                payload(
                    handler(
                        "complete_branch",
                        {"outcome": "completed", "verified": True, "evidence": ["both ran"]},
                        None,
                    )
                )
            )
            return {"status": "completed", "final_response": "branch turn"}
        root = Path(self.thread_workspaces[handle.thread_id])
        marker = root / "marker.txt"
        body = handle.thread_id
        # Both Workers arrive here before either writes, so the run really is
        # concurrent rather than accidentally serialised.
        self._barrier.wait()
        marker.write_text(body, encoding="utf-8")
        with self._write_lock:
            self.written.append(marker)
        self.assert_success(
            payload(
                handler(
                    "complete_agent",
                    {
                        "outcome": "marker written",
                        "verified": True,
                        "evidence": ["marker.txt"],
                    },
                    None,
                )
            )
        )
        return {"status": "completed", "final_response": "worker turn"}


class VNextWorkspacePartitionTests(unittest.TestCase):
    """Sibling Workers a manager isolated must not be able to clobber."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        # Resolved: a CI runner hands out an 8.3 short path, and written
        # paths come back resolved.
        self.workspace = Path(self.temp.name).resolve()
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="product-restricted",
            root_model_id=MANAGER,
            workspace=self.workspace,
            objective="Two workers, one filename",
            task_contract={"criteria": ["both markers survive"]},
            session_id="workspace-partition-session",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, *, workspace_scope: str) -> WorkspacePartitionAdapter:
        adapter = WorkspacePartitionAdapter(workspace_scope=workspace_scope)
        runtime_effects = RuntimeEffectJournal(self.workspace)

        def select_adapter(agent):
            runtime_effects.bind_reader(agent.agent_id, ClaudeRuntimeEffectReader())
            return adapter

        managed = VNextManagedSession(
            control=self.control,
            adapter=adapter,
            session_id=self.root.session_id,
            runtime_effects=runtime_effects,
            adapter_for_agent=select_adapter,
        )
        scheduler = VNextScheduler(
            managed=managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        outcome = scheduler.run()
        self.assertEqual("accepted", outcome["decision"])
        return adapter

    def test_two_private_siblings_writing_one_filename_do_not_clobber(self) -> None:
        adapter = self._run(workspace_scope="private")

        self.assertEqual(2, len(adapter.written))
        first, second = adapter.written
        self.assertNotEqual(
            first, second, "two isolated Workers were handed the same file path"
        )
        for marker, thread_id in zip(
            adapter.written,
            [
                path.read_text(encoding="utf-8")
                for path in adapter.written
            ],
            strict=True,
        ):
            self.assertTrue(marker.is_file())
            self.assertEqual(
                marker.read_text(encoding="utf-8"),
                thread_id,
                "an isolated Worker's file carries another Worker's content",
            )
        # Both roots sit inside the session workspace, so the branch that
        # delegated them can still read what each produced.
        for marker in adapter.written:
            self.assertIn(self.workspace, marker.parents)

    def test_shared_siblings_are_left_alone(self) -> None:
        """The default is unchanged, and deliberately still collides.

        A manager that wants two Workers on one tree gets exactly that.  This
        asserts the old behaviour survives rather than claiming it is safe.
        """

        adapter = self._run(workspace_scope="shared")

        self.assertEqual(2, len(adapter.written))
        self.assertEqual(
            {self.workspace / "marker.txt"},
            set(adapter.written),
            "a shared Worker was moved out of the session workspace",
        )

    def test_a_private_manager_can_own_an_isolated_branch(self) -> None:
        manager = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own a branch from inside a box",
            task_contract={"criteria": ["isolated"]},
            workspace_scope="private",
        )

        self.assertEqual("private", manager.workspace_scope)

    def test_an_unknown_scope_is_refused(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id=MANAGER,
            objective="Own the only branch",
            task_contract={"criteria": ["branch complete"]},
        )

        with self.assertRaises(ProtocolError) as caught:
            self.control.spawn_agent(
                requester_id=branch.agent_id,
                parent_agent_id=branch.agent_id,
                role=AgentRole.WORKER,
                model_id=WORKER,
                objective="Write the marker",
                task_contract={"criteria": ["marker written"]},
                workspace_scope="somewhere-else",
            )

        self.assertEqual("invalid-workspace-scope", caught.exception.code)


if __name__ == "__main__":
    unittest.main()


class CoordinationReceiptsDoNotEchoThePacketTests(unittest.TestCase):
    """await_children is polled repeatedly; the objective is not resent each time."""

    def test_a_one_line_objective_survives_whole(self):
        self.assertEqual("Build the reader", _objective_line("Build the reader"))

    def test_only_the_first_line_is_kept(self):
        line = _objective_line("Build the reader\n\nDO NOT TOUCH any existing file.\nMore.")
        self.assertEqual("Build the reader", line)

    def test_a_long_first_line_is_cut_and_marked(self):
        line = _objective_line("x" * 400)
        self.assertEqual(120, len(line))
        self.assertTrue(line.endswith("\u2026"))

    def test_an_empty_objective_is_an_empty_line(self):
        self.assertEqual("", _objective_line(""))
        self.assertEqual("", _objective_line("   \n  "))

    def test_the_receipt_carries_the_line_and_not_the_packet(self):
        packet = "Write the thing\n" + "secret packet body\n" * 200
        agent = types.SimpleNamespace(
            agent_id="a", thread_id="t", parent_agent_id="p", model_id="m",
            objective=packet, status=AgentStatus.RUNNING,
        )
        receipt = VNextScheduler._coordination_child(agent)
        self.assertEqual("Write the thing", receipt["objective_line"])
        self.assertNotIn("objective", receipt)
        self.assertNotIn("secret packet body", json.dumps(receipt))


class TheFailureReporterCannotBeWhatFailsTests(unittest.TestCase):
    """It runs on the scheduler thread during a failure; it must never raise."""

    def test_an_exception_whose_str_raises_is_named_not_propagated(self):
        class Boom(Exception):
            def __str__(self):
                raise RuntimeError("str exploded")

        self.assertEqual("<unprintable Boom>", _safe_detail(Boom()))

    def test_no_exception_is_an_empty_detail(self):
        self.assertEqual("", _safe_detail(None))

    def test_a_bare_credential_is_redacted(self):
        """R25: a bearer token and a key=value secret passed through intact."""

        detail = _safe_detail(RuntimeError(
            "HTTP 401 Authorization: Bearer sk-secret123; api_key=cc-secret456; token: ghp_abc123"
        ))
        for secret in ("sk-secret123", "cc-secret456", "ghp_abc123"):
            self.assertNotIn(secret, detail)
        self.assertIn("HTTP 401", detail)
        self.assertIn("Bearer", detail)

    def test_quoted_and_lowercase_credentials_are_redacted(self):
        """R25 verify: a quoted value and a lowercase bearer passed through."""

        for said, secret in (
            ('client_secret: "ordinaryvalue123"', "ordinaryvalue123"),
            ("authorization: bearer ordinaryvalue123", "ordinaryvalue123"),
            ("password='hunter2hunter2'", "hunter2hunter2"),
        ):
            with self.subTest(said=said):
                self.assertNotIn(secret, _safe_detail(RuntimeError(said)))

    def test_an_ordinary_sentence_after_bearer_is_left_alone(self):
        """R25 verify: "Bearer service unavailable" lost its useful word."""

        for said in ("Bearer service unavailable", "Basic authentication failed"):
            with self.subTest(said=said):
                self.assertEqual(said, _safe_detail(RuntimeError(said)))

    def test_a_turn_result_reason_and_subtype_are_scrubbed(self):
        """R25: terminal_reason and subtype reached the manager unscrubbed."""

        blocker = VNextScheduler._runtime_failure_blocker("failed", {
            "terminal_reason": "provider at /home/someone/Private/ACCESS_KEY/token.json",
            "subtype": "HTTP https://api.example.com/v1/redeem/SECRET?sig=TOPSECRET",
        })
        for secret in ("ACCESS_KEY", "Private", "SECRET", "TOPSECRET", "redeem"):
            self.assertNotIn(secret, blocker)
        self.assertIn("token.json", blocker)
        self.assertIn("api.example.com", blocker)

    def test_a_windows_path_is_reduced_to_its_last_segment(self):
        detail = _safe_detail(RuntimeError(r"failed reading C:\Users\someone\Documents\secret.txt now"))
        self.assertEqual("failed reading secret.txt now", detail)

    def test_a_posix_path_is_reduced_to_its_last_segment(self):
        self.assertEqual(
            "cannot open creds.json",
            _safe_detail(RuntimeError("cannot open /home/u/.config/creds.json")),
        )

    def test_provider_key_hint_keeps_home_relative_path_but_hides_absolute_path(self):
        hint = "Z.ai provider refused the credential; correct the key in ~/.vnext/providers.json and delegate again"
        self.assertEqual(hint, _safe_detail(RuntimeError(hint)))
        self.assertEqual(
            "cannot open providers.json",
            _safe_detail(RuntimeError("cannot open /home/someone/.vnext/providers.json")),
        )

    def test_every_home_relative_hint_in_worker_errors_survives(self):
        for hint in (
            "~/.vnext/providers.json",
            "~/.vnext/providers.json",
        ):
            with self.subTest(hint=hint):
                self.assertEqual(f"open {hint}", _safe_detail(RuntimeError(f"open {hint}")))

    def test_a_url_keeps_its_scheme_and_host(self):
        for sentence in (
            "unexpected status 401 from https://api.openai.com/v1/responses",
            "socket closed: wss://chatgpt.com/backend-api/codex/responses",
            "see http://localhost:8080/health for details",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(sentence, _safe_detail(RuntimeError(sentence)))

    def test_a_drive_letter_path_after_a_word_is_still_reduced(self):
        self.assertEqual(
            "failed: secret.txt",
            _safe_detail(RuntimeError(r"failed: D:\work\secret.txt")),
        )

    def test_a_relative_path_stays_whole(self):
        """R18: ``./src/app/config.json`` came out as ``.config.json``."""

        for said in ("cannot read ./src/app/config.json", "cannot read ../lib/vnext/x.py",
                     "cannot read src-old/app/x.py"):
            with self.subTest(said=said):
                self.assertEqual(said, _safe_detail(RuntimeError(said)))

    def test_a_windows_share_path_keeps_only_its_file_name(self):
        """R19: ``\\\\server\\share\\secret\\token.txt`` came out whole."""

        for said in (r"cannot read \\server\share\secret\token.txt",
                     r"cannot read \\?\D:\shared\secret\token.txt",
                     "cannot read //server/share/secret/token.txt"):
            with self.subTest(said=said):
                self.assertEqual("cannot read token.txt", _safe_detail(RuntimeError(said)))

    def test_a_url_loses_its_query_fragment_and_login(self):
        """R20: ``?token=SECRET123`` reached the manager's blocker."""

        for said, kept in (
            ("request failed at https://api.example.com/download?token=SECRET123",
             "request failed at https://api.example.com/download?\u2026"),
            ("request failed at https://ada:SECRET123@api.example.com/v1",
             "request failed at https://api.example.com/v1"),
            ("request failed at https://api.example.com/v1#SECRET123",
             "request failed at https://api.example.com/v1#\u2026"),
        ):
            with self.subTest(said=said):
                detail = _safe_detail(RuntimeError(said))
                self.assertEqual(kept, detail)
                self.assertNotIn("SECRET123", detail)

    def test_a_url_path_keeps_only_its_plain_words(self):
        """R22: ``/redeem/SECRET123`` and ``;token=SECRET123`` reached the manager."""

        for said, kept in (
            ("provider failed at https://api.example.com/redeem/SECRET123",
             "provider failed at https://api.example.com/\u2026"),
            # R23: a lowercase credential reads as a word, so spelling is no test.
            ("provider failed at https://api.example.com/redeem/abcdefghijklmnopqrstuvwx",
             "provider failed at https://api.example.com/\u2026"),
            ("provider failed at https://api.example.com/v1/abcdefghijklmnopqrstuvwx",
             "provider failed at https://api.example.com/v1/\u2026"),
            ("provider failed at https://api.example.com/api/coding/paas/v4/messages",
             "provider failed at https://api.example.com/api/coding/paas/v4/messages"),
            ("provider failed at https://api.example.com/download;token=SECRET123",
             "provider failed at https://api.example.com/\u2026"),
            ("provider failed at https://api.example.com/v1/responses",
             "provider failed at https://api.example.com/v1/responses"),
            ("provider failed at https://open.example.com/api/paas/v4/chat/completions",
             "provider failed at https://open.example.com/api/paas/v4/chat/completions"),
        ):
            with self.subTest(said=said):
                detail = _safe_detail(RuntimeError(said))
                self.assertEqual(kept, detail)
                self.assertNotIn("SECRET123", detail)
                self.assertNotIn("abcdefghijklmnopqrstuvwx", detail)

    def test_a_file_url_loses_its_query_and_fragment(self):
        """R21: ``file:///.../download?token=SECRET123`` reached the manager."""

        for said, kept in (
            ("request failed at file:///home/someone/download?token=SECRET123#session",
             "request failed at download?\u2026#\u2026"),
            ("request failed at file:///home/someone/download#SECRET123",
             "request failed at download#\u2026"),
            ("request failed at /home/someone/download?token=SECRET123",
             "request failed at download?\u2026"),
        ):
            with self.subTest(said=said):
                detail = _safe_detail(RuntimeError(said))
                self.assertEqual(kept, detail)
                self.assertNotIn("SECRET123", detail)

    def test_a_path_with_spaces_keeps_only_its_file_name(self):
        """R20: ``Alice Smith\\token.txt`` survived the reduction."""

        for said in (r"cannot read \\server\share\Alice Smith\token.txt",
                     r"cannot read D:\shared\Jane Doe\token.txt",
                     "cannot read /home/Jane Doe/token.txt"):
            with self.subTest(said=said):
                self.assertEqual("cannot read token.txt", _safe_detail(RuntimeError(said)))
        self.assertEqual(
            "copied a.txt to b.txt",
            _safe_detail(RuntimeError("copied /srv/x/a.txt to /srv/y/b.txt")),
        )

    def test_a_home_relative_path_keeps_only_vnext_hints_whole(self):
        """R24: ``~/Private/SECRET123/token.json`` and its Windows form came out whole."""

        for said in ("failed to open ~/Private/SECRET123/token.json",
                     r"failed to open ~\Private\SECRET123\token.json",
                     "failed to open ~/.vnext/SECRET123/token.json"):
            with self.subTest(said=said):
                detail = _safe_detail(RuntimeError(said))
                self.assertEqual("failed to open token.json", detail)
        for hint in ("~/.vnext/providers.json", "~/.vnext/providers.json"):
            with self.subTest(hint=hint):
                self.assertEqual(f"open {hint}", _safe_detail(RuntimeError(f"open {hint}")))

    def test_an_ipv6_host_keeps_its_brackets(self):
        """R24: ``http://[::1]:8080/health`` came out as ``http://::1:8080/health``."""

        for said in ("see http://[::1]:8080/health for details",
                     "see http://[::1]/health for details"):
            with self.subTest(said=said):
                self.assertEqual(said, _safe_detail(RuntimeError(said)))

    def test_the_detail_is_bounded(self):
        detail = _safe_detail(RuntimeError("x" * 5000))
        self.assertEqual(300, len(detail))
        self.assertTrue(detail.endswith("\u2026"))

    def test_the_timeout_sentence_survives_unchanged(self):
        sentence = "app-server event wait timed out after 600s of waiting"
        self.assertEqual(sentence, _safe_detail(RuntimeError(sentence)))


class RuntimeFailureBlockerTests(unittest.TestCase):
    """What a manager is told when a child's turn dies under it.

    An Opus worker died 43 seconds into a read-only packet on 2026-09-18 and
    cost $0.80.  The blocker read "runtime turn ended as failed" and nothing
    more, which is also what a worker that argued itself into a corner
    produces.  The provider had already said which of the two it was, in the
    turn result, and the scheduler was dropping it.
    """

    blocker = staticmethod(VNextScheduler._runtime_failure_blocker)

    def test_a_provider_abort_says_so_and_says_the_packet_may_be_sound(self):
        line = self.blocker("failed", {
            "status": "failed",
            "is_error": True,
            "subtype": "error_during_execution",
            "terminal_reason": "aborted_streaming",
            "api_error_status": None,
            "rate_limited": False,
        })
        self.assertIn("aborted_streaming", line)
        self.assertIn("error_during_execution", line)
        self.assertIn("worth retrying", line)
        self.assertEqual(1, line.count(":"), f"the blocker must stay one sentence: {line}")

    def test_a_rate_limit_reads_as_a_rate_limit_and_never_as_worth_retrying(self):
        line = self.blocker("failed", {
            "status": "failed",
            "is_error": True,
            "subtype": "error_during_execution",
            "terminal_reason": "aborted_streaming",
            "api_error_status": 429,
            "rate_limited": True,
            "retry_after_seconds": 30,
        })
        self.assertIn("rate-limited", line)
        self.assertIn("HTTP 429", line)
        self.assertIn("30s", line)
        self.assertNotIn("worth retrying", line)

    def test_a_rejected_key_says_the_key_is_wrong_and_never_worth_retrying(self):
        """What a mistyped provider key used to be reported as.

        Measured on 2026-09-21 with a key the endpoint rejects: the manager
        was told "the provider ended the turn (api_error, success, HTTP 401),
        so the packet itself may still be sound and worth retrying".  A 401
        reproduces on every retry, and the SDK's subtype "success" means only
        that its own result message arrived intact.
        """

        line = self.blocker("failed", {
            "status": "failed",
            "is_error": True,
            "subtype": "success",
            "terminal_reason": "api_error",
            "api_error_status": 401,
            "rate_limited": False,
        })
        self.assertIn("HTTP 401", line)
        self.assertIn("retrying will not help", line)
        self.assertNotIn("worth retrying", line)
        self.assertNotIn("success", line)
        self.assertEqual(1, line.count(":"), f"the blocker must stay one sentence: {line}")

    def test_a_forbidden_status_reads_the_same_way(self):
        line = self.blocker("failed", {
            "status": "failed", "is_error": True, "api_error_status": 403,
        })
        self.assertIn("retrying will not help", line)

    def test_an_authentication_code_with_no_status_still_says_so(self):
        line = self.blocker("failed", {"status": "failed", "code": "authentication_failed"})
        self.assertIn("retrying will not help", line)

    def test_the_sdk_success_subtype_is_left_out_of_an_ordinary_failure(self):
        line = self.blocker("failed", {
            "status": "failed",
            "is_error": True,
            "subtype": "success",
            "terminal_reason": "aborted_streaming",
            "api_error_status": None,
            "rate_limited": False,
        })
        self.assertIn("aborted_streaming", line)
        self.assertNotIn("success", line)

    def test_a_codex_turn_error_is_quoted_and_a_401_reads_as_a_wrong_key(self):
        """A Command Code worker with a mistyped key, round 16 of the release review.

        The manager read "runtime turn ended as failed" and nothing else, while
        the turn result carried the provider's own sentence naming the 401.
        """

        line = self.blocker("failed", {"status": "failed", "error": {
            "message": "unexpected status 401 Unauthorized: Invalid 'Authorization' "
                       "header or token., url: http://127.0.0.1:52089/v1/responses",
            "codexErrorInfo": "other"}})
        self.assertIn("401 Unauthorized", line)
        self.assertIn("retrying will not help", line)
        self.assertNotIn("worth retrying", line)

    def test_a_codex_turn_error_that_is_no_login_problem_is_quoted_as_it_was_said(self):
        line = self.blocker("failed", {"status": "failed", "error": {
            "message": "stream disconnected before completion", "codexErrorInfo": "other"}})
        self.assertIn("stream disconnected before completion", line)
        self.assertNotIn("retrying will not help", line)

    def test_a_provider_error_loses_its_link_secrets_and_paths(self):
        """R24: a failed turn's error.message reached the manager unscrubbed."""

        line = self.blocker("failed", {"status": "failed", "error": {
            "message": "request denied at https://api.example.com/v1/redeem/SECRET123?token=SECONDSECRET"
                       " reading /home/someone/Private/THIRDSECRET/token.json"}})
        self.assertIn("https://api.example.com/v1/\u2026?\u2026", line)
        self.assertIn("token.json", line)
        for secret in ("SECRET123", "SECONDSECRET", "THIRDSECRET"):
            self.assertNotIn(secret, line)

    def test_a_failure_the_provider_said_nothing_about_keeps_the_bare_sentence(self):
        # A worker that ran itself into the ground gets no provider detail, and
        # inventing one would be the opposite of the fix.
        self.assertEqual(
            "runtime turn ended as failed",
            self.blocker("failed", {"status": "failed"}),
        )

    def test_a_terminal_reason_that_is_not_a_string_is_left_out(self):
        self.assertEqual(
            "runtime turn ended as failed",
            self.blocker("failed", {"status": "failed", "terminal_reason": 17, "subtype": None}),
        )

    def test_the_status_word_the_runtime_used_is_the_one_reported(self):
        self.assertTrue(
            self.blocker("stalled", {"status": "stalled"}).startswith("runtime turn ended as stalled")
        )


class ProviderFailureOutcomeRecordTests(unittest.TestCase):
    def test_provider_turn_error_writes_failed_outcome(self) -> None:
        class FailingWorker(ExternalScriptedWorker):
            failures = 0

            def turn_process_ended(self, handle=None):
                """The connection broke because the provider process died.

                That is the evidence vNext now asks for before it treats a
                failed wait as a stop, and it is what lets the retry start at
                once instead of waiting out the one-writer barrier.
                """

                del handle
                return True

            def wait_turn(self, handle, *, timeout=300):
                self.failures += 1
                if self.failures == 1:
                    raise RuntimeError("provider connection closed during turn")
                return super().wait_turn(handle, timeout=timeout)

        class FailedResultWorker(ExternalScriptedWorker):
            def wait_turn(self, handle, *, timeout=300):
                return {"status": "failed", "terminal_reason": "provider_error"}

        for worker_type in (FailingWorker, FailedResultWorker):
            with self.subTest(worker_type=worker_type.__name__):
                with tempfile.TemporaryDirectory() as workspace:
                    service = VNextMcpService(
                        workspace=workspace,
                        catalog=[{"provider": "codex", "model": "worker-model"}],
                        adapter_factories={"codex": worker_type},
                        event_log=None,
                        status_file=None,
                    )
                    try:
                        created = payload(service.session.external_tool_call(
                            tool="delegate",
                            arguments={
                                "role": AgentRole.WORKER.value,
                                "model_id": "worker-model",
                                "objective": "Attempt bounded work",
                                "task_contract": {"criteria": ["work attempted"]},
                            },
                        ))
                        self.assertTrue(created["success"], created)
                        agent_id = created["agent_id"]
                        service.session.external_tool_call(
                            tool="await_children", arguments={"agent_ids": [agent_id]}, timeout=30,
                        )
                        path = service.outcome_log
                        self.assertIsNotNone(path)
                        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                        self.assertEqual(1, len(rows))
                        self.assertEqual(agent_id, rows[0]["agent_id"])
                        self.assertEqual("failed", rows[0]["status"])
                        if worker_type is FailingWorker:
                            retried = payload(service.session.external_tool_call(
                                tool="retry",
                                arguments={
                                    "agent_id": agent_id,
                                    "task_contract": {"criteria": ["work attempted"]},
                                },
                            ))
                            self.assertTrue(retried["success"], retried)
                            service.session.external_tool_call(
                                tool="await_children", arguments={"agent_ids": [agent_id]}, timeout=30,
                            )
                            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                            self.assertEqual(1, len(rows))
                            self.assertEqual("completed", rows[0]["status"])
                    finally:
                        service.close()

VNextSchedulerTests.test_delegate_does_not_read_counts_as_minutes = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_does_not_read_counts_as_minutes)

VNextSchedulerTests.test_delegate_omits_warnings_for_clean_objective = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_omits_warnings_for_clean_objective)

VNextSchedulerTests.test_delegate_returns_invalid_effort_as_a_tool_result_before_spawn = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_returns_invalid_effort_as_a_tool_result_before_spawn)

VNextSchedulerTests.test_delegate_succeeds_when_preflight_warnings_raise = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_succeeds_when_preflight_warnings_raise)

VNextSchedulerTests.test_delegate_time_warning_is_route_aware = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_time_warning_is_route_aware)

VNextSchedulerTests.test_delegate_warns_when_objective_exceeds_turn_cap = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_warns_when_objective_exceeds_turn_cap)

VNextSchedulerTests.test_delegate_warns_when_objective_needs_browser = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_delegate_warns_when_objective_needs_browser)

VNextSchedulerTests.test_replace_refuses_effort_inherited_by_incompatible_provider = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextSchedulerTests.test_replace_refuses_effort_inherited_by_incompatible_provider)
