"""A primary vNext does not own can still run a vNext child tree.

These tests stand up a real session whose root is external, then drive it the
way Claude Code will: by calling manager tools from outside, with no prompt and
no provider turn behind them.  No model runs; the worker is scripted.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from vnext.host_contract import SessionStartRequest
from vnext.process_supervisor import ProcessCleanup
from vnext.vnext_app_server import TurnHandle
from vnext.vnext_external_primary import ExternalPrimaryAdapter, ExternalPrimaryError
from vnext.vnext_orchestration import AgentRole
from vnext.vnext_runtime_types import RuntimeCleanup
from vnext.vnext_session_runtime import VNextRuntimeSession


WORKER_MODEL = "worker-model"
EXTERNAL_MODEL = "claude-code"


def _policy() -> dict:
    return {
        "posture": {
            "workspace_writes": True,
            "network": "restricted",
            "approvals_requested": True,
            "reviewer": "auto_review",
            "environment_ready": True,
        }
    }


def _read(result) -> dict:
    value = json.loads(result.as_json_text())
    value["success"] = result.success
    return value


class ScriptedWorker:
    """A child runtime that completes its one turn without a model."""

    provider = "codex"
    harness = "app-server"

    def __init__(self) -> None:
        self.native_approval_handler = None
        self._lock = threading.RLock()
        self.handlers: dict[str, object] = {}
        self.tools: dict[str, list[dict]] = {}
        self.objectives: list[str] = []
        # When set, the worker's turn stays open until the test releases it.
        self.hold: threading.Event | None = None
        # Threads named here end their turn without completing, which is what
        # the scheduler turns into a blocker.  They never wait on ``hold``.
        self.blocked_threads: set[str] = set()
        # Threads named here answer in words and end the turn with no
        # complete_agent call, the way a model sometimes does.
        self.silent_threads: set[str] = set()
        # An approval is routed to a worker only through an attested identity.
        # The default stays the phase a freshly started thread really reports;
        # a test that needs the approval path sets this to "attested".
        self.binding_phase = "thread-started"
        # What the worker writes into complete_agent. A test that checks the
        # bound on a long report sets this to a long string.
        self.outcome_text = "completed bounded execution"
        self._threads = 0
        self._turns = 0
        self.on_start_turn = None
        self.closed = False

    def initialize(self, *, timeout: float = 30) -> dict:
        return {}

    def start_thread(self, *, model, developer_instructions, tools, tool_handler,
                     requested_posture, **_kwargs):
        del model, developer_instructions
        with self._lock:
            self._threads += 1
            thread_id = f"worker-thread-{self._threads}"
            self.handlers[thread_id] = tool_handler
            self.tools[thread_id] = list(tools)
        result = _policy()
        result["posture"]["reviewer"] = str(requested_posture.reviewer)
        return thread_id, result

    def tool_registration_attestation(self, thread_id):
        tools = self.tools[thread_id]
        return {
            "acknowledged": True,
            "model_id": WORKER_MODEL,
            "tool_count": len(tools),
            "tool_names": [str(tool.get("name") or "") for tool in tools],
            "definition_sha256": "b" * 64,
            "handler_registered": self.handlers[thread_id] is not None,
        }

    def thread_identity_attestation(self, thread_id):
        return {
            "runtime_thread": thread_id,
            "provider": self.provider,
            "bound": True,
            "provider_session": f"session-{thread_id}",
            "binding_phase": self.binding_phase,
            "synthetic": False,
        }

    def start_turn(self, *, thread_id, prompt, **_kwargs):
        with self._lock:
            self._turns += 1
            turn_id = f"worker-turn-{self._turns}"
            self.objectives.append(prompt)
        if self.on_start_turn is not None:
            self.on_start_turn(thread_id, turn_id)
        return TurnHandle(thread_id=thread_id, turn_id=turn_id, cursor=0)

    def wait_turn(self, handle, *, timeout=300):
        if handle.thread_id in self.blocked_threads:
            return {"status": "stopped", "final_response": "",
                    "terminal_reason": "the scripted provider stopped this turn"}
        if handle.thread_id in self.silent_threads:
            return {"status": "completed", "final_response": "ok"}
        if self.hold is not None:
            self.hold.wait(timeout)
        result = _read(self.handlers[handle.thread_id](
            "complete_agent",
            {"outcome": self.outcome_text, "verified": True,
             "evidence": [f"scripted worker evidence {index}" for index in range(12)]},
            None,
        ))
        if not result["success"]:
            raise AssertionError(result)
        return {"status": "completed", "final_response": "worker evidence"}

    def events_since(self, cursor=None):
        return ()

    def close(self):
        self.closed = True
        return RuntimeCleanup(ProcessCleanup("scripted-worker", 0, 0, ()), True, True, ())


class ExternalPrimaryAdapterTest(unittest.TestCase):
    """The adapter refuses exactly the controls it does not own."""

    def test_execution_controls_are_refused(self) -> None:
        adapter = ExternalPrimaryAdapter(client="claude-code")
        self.assertEqual(adapter.provider, "external")
        with self.assertRaises(ExternalPrimaryError):
            adapter.start_turn(thread_id="t", prompt="hello")
        with self.assertRaises(ExternalPrimaryError):
            adapter.interrupt(None)
        with self.assertRaises(ExternalPrimaryError):
            adapter.steer(None, "stop")
        self.assertFalse(adapter.can_start_turn("t"))

    def test_dispatch_requires_a_bound_tree(self) -> None:
        adapter = ExternalPrimaryAdapter(client="claude-code")
        with self.assertRaises(ExternalPrimaryError):
            adapter.dispatch_external_call(tool="delegate", arguments={})

    def test_unknown_tools_are_rejected_before_the_handler(self) -> None:
        seen: list[str] = []
        adapter = ExternalPrimaryAdapter(client="claude-code")
        adapter.start_thread(
            model="claude-code", developer_instructions="ROLE=root_manager",
            tools=[{"name": "delegate"}],
            tool_handler=lambda tool, arguments, context: seen.append(tool),
            requested_posture=type("P", (), {"reviewer": "user"})(),
            workspace=".",
        )
        with self.assertRaises(ExternalPrimaryError):
            adapter.dispatch_external_call(tool="rm_rf", arguments={})
        self.assertEqual(seen, [])


class ExternalPrimarySessionTest(unittest.TestCase):
    """A real session whose root is a client, driven only from outside."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.worker = ScriptedWorker()
        self.events: list[object] = []

    def _session(self) -> VNextRuntimeSession:
        request = SessionStartRequest(
            session_id="external-session-1",
            workspace=str(self.workspace),
            primary_agent_id="primary",
            primary={"provider": "external", "model": EXTERNAL_MODEL, "effort": "high"},
            main_preset={"instructions": "Coordinate this workforce."},
            catalog_config={"models": [
                {"provider": "external", "model": EXTERNAL_MODEL},
                {"provider": "codex", "model": WORKER_MODEL},
            ]},
            config={"external_client": "claude-code"},
        )
        session = VNextRuntimeSession(
            request, self.events.append,
            adapter_factories={"codex": lambda: self.worker},
        )
        self.addCleanup(session.close)
        return session

    def test_session_stands_up_without_a_prompt(self) -> None:
        session = self._session()
        snapshot = session.start()
        self.assertTrue(session.external_primary)
        self.assertEqual(snapshot["primary_agent_id"], "primary")
        names = {str(tool.get("name") or "") for tool in session.external_tools()}
        self.assertIn("delegate", names)
        self.assertIn("await_children", names)
        self.assertIn("cancel_agent", names)

    def test_bad_claude_effort_is_refused_without_spawn_or_event(self) -> None:
        session = self._session()
        session.start()
        session.registry.cards[WORKER_MODEL] = replace(
            session.registry.cards[WORKER_MODEL], provider="claude"
        )
        before_agents = set(session.control.sessions[session.root.session_id].agents)
        result = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "effort": "ludicrous",
            "task_contract": {"criteria": ["section written"]},
        }))
        self.assertFalse(result["success"], result)
        self.assertEqual("invalid-request", result["error_code"])
        self.assertIn("high, low, max, medium, xhigh", result["error"])
        self.assertEqual(before_agents, set(session.control.sessions[session.root.session_id].agents))
        self.assertNotIn("ludicrous", repr(self.events))

    def test_external_delegate_creates_and_runs_a_child(self) -> None:
        session = self._session()
        session.start()
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        }))
        self.assertTrue(created["success"], created)
        child_id = created["agent_id"]

        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        ))
        self.assertTrue(waited["success"], waited)

        snapshot = session.snapshot()
        children = [agent for agent in snapshot["agents"] if agent["agent_id"] != "primary"]
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["agent_id"], child_id)
        self.assertEqual(children[0]["status"], "completed")
        self.assertEqual(len(self.worker.objectives), 1)

    def test_the_root_never_takes_a_turn(self) -> None:
        session = self._session()
        session.start()
        session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        adapter = session._adapters["external"]
        self.assertEqual(adapter.dispatched_calls, 1)
        # start_turn raises, so a scheduler that tried would have failed the
        # session outright.  Assert the session is still alive instead.
        self.assertNotEqual(session.snapshot()["status"], "failed")

    def test_a_failed_session_tells_the_client_why(self) -> None:
        """The cause, not the consequence, which the first real run got wrong.

        When the runtime dies the root stops being bound, and reporting that is
        useless to a client with no access to the event stream: the first real
        delegation from Claude Code failed on a bad Codex config and reported
        only "external primary is not bound to a control tree".
        """

        session = self._session()
        session.start()
        session._status = "failed"
        session._failure = "config.toml:203:1: invalid type: boolean `true`"
        with self.assertRaises(RuntimeError) as caught:
            session.external_tool_call(tool="inspect", arguments={"agent_id": "primary", "deep": False})
        self.assertIn("config.toml:203:1", str(caught.exception))

    def test_a_wait_that_runs_out_is_resumable(self) -> None:
        """A client holding a call open is told to call again, not told it finished."""

        release = threading.Event()
        self.worker.hold = release
        session = self._session()
        session.start()
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        }))
        child_id = created["agent_id"]
        session._scheduler.external_await_budget = 0.5
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]},
        ))
        self.assertTrue(waited["success"], waited)
        self.assertFalse(waited["settled"])
        self.assertEqual(waited["awaiting"], [child_id])
        self.assertIn("await_children again", waited["note"])

        release.set()
        session._scheduler.external_await_budget = 30
        settled = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]},
        ))
        self.assertTrue(settled["settled"], settled)

    def test_the_default_wait_answers_before_the_restart_proxy_gives_up(self) -> None:
        """On 2026-09-23 every long await_children through the plugin came back as
        "MCP error -32001: Request timed out", and so did the inspect behind it.

        The reloaderoo proxy forwards a call with the MCP SDK's 60-second
        default, and the budget was 110 seconds.
        """

        session = self._session()
        session.start()
        self.assertLess(session._scheduler.external_await_budget, 60)

    def test_a_waiting_approval_ends_the_wait_and_names_itself(self) -> None:
        """The defect this covers cost a real worker and $0.26 on 2026-09-17.

        A worker asked to run a command, the request reached neither of the two
        routes an external primary has, and 240 seconds later it was declined by
        timeout.  The client had been sitting inside await_children the whole
        time.  So the wait must end when the question arrives.
        """

        release = threading.Event()
        self.worker.hold = release
        self.worker.binding_phase = "attested"
        self.addCleanup(release.set)
        session = self._session()
        session.start()
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Run one bounded command",
            "task_contract": {"criteria": ["command run"]},
            "approvals": "ask",
        }))
        child_id = created["agent_id"]
        scheduler = session._scheduler
        # A long budget, so anything but the approval ending the wait shows up
        # as a test that hangs rather than one that passes by coincidence.
        scheduler.external_await_budget = 30

        reviewed: dict[str, object] = {}
        requested = threading.Event()

        def review() -> None:
            requested.set()
            reviewed["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:worker-item",
                    "provider": "codex",
                    "provider_correlation": {
                        "session": "session-worker-thread-1",
                        "turn": "worker-turn-1",
                        "request": "worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )

        thread = threading.Thread(target=review, daemon=True)
        thread.start()
        self.assertTrue(requested.wait(2))

        started = time.monotonic()
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]},
        ))
        elapsed = time.monotonic() - started

        self.assertTrue(waited["success"], waited)
        self.assertFalse(waited["settled"])
        self.assertLess(elapsed, 20, "the wait ran to its budget instead of ending on the approval")
        approvals = waited["pending_approvals"]
        self.assertEqual(len(approvals), 1, waited)
        self.assertEqual(approvals[0]["worker_agent_id"], child_id)
        self.assertEqual(approvals[0]["effect"], "execute")
        self.assertTrue(approvals[0]["approval_id"])
        self.assertIn("resolve_approval", waited["note"])

        answered = scheduler.resolve_approval(
            approvals[0]["approval_id"], "accept", "the packet asked for this command",
        )
        self.assertEqual(answered["decision"], "accept")
        thread.join(2)
        self.assertEqual({"decision": "accept"}, reviewed["value"])

    def test_approval_during_turn_publication_window_is_pending_for_worker(self) -> None:
        """An approval arriving before the scheduler publishes the turn is routed."""

        release = threading.Event()
        self.worker.hold = release
        self.addCleanup(release.set)
        session = self._session()
        session.start()
        scheduler = session._scheduler
        requested = threading.Event()
        routing_started = threading.Event()
        review_thread: threading.Thread | None = None

        original_route = scheduler._worker_for_attested_provider_session

        def route(provider: str, provider_session: str) -> str | None:
            routing_started.set()
            return original_route(provider, provider_session)

        scheduler._worker_for_attested_provider_session = route

        def review() -> None:
            requested.set()
            scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:publication-window",
                    "provider": "codex",
                    "provider_correlation": {
                        "session": "session-worker-thread-1",
                        "turn": "worker-turn-1",
                        "request": "worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )

        def during_start(_thread_id: str, _turn_id: str) -> None:
            nonlocal review_thread
            self.assertEqual(scheduler._active_turns, {})
            self.worker.binding_phase = "attested"
            review_thread = threading.Thread(target=review, daemon=True)
            review_thread.start()
            self.assertTrue(requested.wait(2))
            self.assertTrue(routing_started.wait(2))

        self.worker.on_start_turn = during_start
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Run one bounded command",
            "task_contract": {"criteria": ["command run"]},
            "approvals": "ask",
        }))
        self.assertTrue(created["success"], created)
        child_id = created["agent_id"]

        seen: dict = {}
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            seen = _read(session.external_tool_call(
                tool="inspect", arguments={"agent_id": "primary", "deep": True},
            ))
            if seen.get("pending_approvals"):
                break
            time.sleep(0.05)
        self.assertTrue(seen["success"], seen)
        approvals = seen["pending_approvals"]
        self.assertEqual(len(approvals), 1, seen)
        self.assertEqual(approvals[0]["worker_agent_id"], child_id)
        scheduler.resolve_approval(approvals[0]["approval_id"], "decline", "test teardown")
        if review_thread is not None:
            review_thread.join(2)

    def test_inspect_shows_the_approval_a_blocked_child_is_waiting_on(self) -> None:
        release = threading.Event()
        self.worker.hold = release
        self.worker.binding_phase = "attested"
        self.addCleanup(release.set)
        session = self._session()
        session.start()
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Run one bounded command",
            "task_contract": {"criteria": ["command run"]},
            "approvals": "ask",
        }))
        child_id = created["agent_id"]
        scheduler = session._scheduler

        requested = threading.Event()

        def review() -> None:
            requested.set()
            scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:worker-item",
                    "provider": "codex",
                    "provider_correlation": {
                        "session": "session-worker-thread-1",
                        "turn": "worker-turn-1",
                        "request": "worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )

        thread = threading.Thread(target=review, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.assertTrue(requested.wait(2))

        seen: dict = {}
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            seen = _read(session.external_tool_call(
                tool="inspect", arguments={"agent_id": "primary", "deep": True},
            ))
            if seen.get("pending_approvals"):
                break
            time.sleep(0.05)

        self.assertTrue(seen["success"], seen)
        approvals = seen["pending_approvals"]
        self.assertEqual(len(approvals), 1, seen)
        self.assertEqual(approvals[0]["worker_agent_id"], child_id)
        scheduler.resolve_approval(
            approvals[0]["approval_id"], "decline", "test teardown",
        )

    def test_a_decision_word_outside_the_enum_is_refused_not_read_as_decline(self) -> None:
        """Measured here on 2026-09-18: "approve" silently declined a live command.

        The tool schema says accept or decline.  The external MCP surface does
        not enforce the enum, so any other word used to reach the DECLINE arm
        and the worker's command was refused with nothing said about why.
        """

        release = threading.Event()
        self.worker.hold = release
        self.worker.binding_phase = "attested"
        self.addCleanup(release.set)
        session = self._session()
        session.start()
        session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Run one bounded command",
            "task_contract": {"criteria": ["command run"]},
            "approvals": "ask",
        })
        scheduler = session._scheduler

        reviewed: dict[str, object] = {}
        requested = threading.Event()

        def review() -> None:
            requested.set()
            reviewed["value"] = scheduler.review_approval(
                "approval/request",
                {
                    "approval_reference": "fixture:worker-item",
                    "provider": "codex",
                    "provider_correlation": {
                        "session": "session-worker-thread-1",
                        "turn": "worker-turn-1",
                        "request": "worker-item",
                    },
                    "correlation_attested": True,
                    "effect": "execute",
                },
            )

        thread = threading.Thread(target=review, daemon=True)
        thread.start()
        self.assertTrue(requested.wait(2))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not scheduler._approvals_for_manager("primary"):
            time.sleep(0.05)
        approval_id = scheduler._approvals_for_manager("primary")[0].approval_id

        with self.assertRaises(ValueError) as caught:
            scheduler.resolve_approval(approval_id, "approve", "meant to allow it")
        self.assertIn("accept", str(caught.exception))
        self.assertIn("still pending", str(caught.exception))

        # Still pending, so the caller's second attempt is the one that counts.
        self.assertEqual(len(scheduler._approvals_for_manager("primary")), 1)
        scheduler.resolve_approval(approval_id, "accept", "meant to allow it")
        thread.join(2)
        self.assertEqual({"decision": "accept"}, reviewed["value"])

    def _delegate(self, session, objective: str) -> str:
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": objective,
            "task_contract": {"criteria": ["section written"]},
        }))
        self.assertTrue(created["success"], created)
        return created["agent_id"]

    def _status_of(self, session, agent_id: str) -> str:
        for agent in session.snapshot()["agents"]:
            if agent["agent_id"] == agent_id:
                return agent["status"]
        raise AssertionError(f"{agent_id} is not in the snapshot")

    def _wait_until(self, predicate, seconds: float = 5.0) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError("the expected state never arrived")

    def test_a_blocked_child_ends_the_wait_and_carries_its_blocker(self) -> None:
        """A blocked child is settled, and the wait used to run to its timeout.

        Nothing moves a blocked child but a decision from the caller, and the
        caller was inside await_children.  So the whole budget was spent to
        answer "still running" about the one child that never would be.
        """

        self.worker.blocked_threads.add("worker-thread-1")
        session = self._session()
        session.start()
        child_id = self._delegate(session, "Write the report section")
        scheduler = session._scheduler
        # Long enough that anything but the blocker ending the wait reads as a
        # hang rather than as a pass.
        scheduler.external_await_budget = 30
        self._wait_until(lambda: self._status_of(session, child_id) == "blocked")

        started = time.monotonic()
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]},
        ))
        elapsed = time.monotonic() - started

        self.assertTrue(waited["success"], waited)
        self.assertTrue(waited["settled"], waited)
        self.assertEqual([], waited["awaiting"])
        self.assertLess(elapsed, 20, "the wait ran to its budget on a blocked child")
        decisions = waited["children_needing_a_decision"]
        self.assertEqual(1, len(decisions), waited)
        self.assertEqual(child_id, decisions[0]["agent_id"])
        self.assertEqual("blocked", decisions[0]["status"])
        self.assertTrue(decisions[0]["blocker"], decisions[0])
        self.assertIn("waiting on a decision", waited["note"])

    def test_a_child_that_answered_without_completing_ends_the_wait(self) -> None:
        """R24: a worker that replied in words left await_children open for good.

        Its turn ended, it went back to ready, and nothing moves it again but
        a message from the caller, who was inside the wait.
        """

        self.worker.silent_threads.add("worker-thread-1")
        session = self._session()
        session.start()
        child_id = self._delegate(session, "Reply with the single word ok")
        scheduler = session._scheduler
        scheduler.external_await_budget = 30
        self._wait_until(lambda: child_id in scheduler._awaiting_message_agents)

        started = time.monotonic()
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]},
        ))
        elapsed = time.monotonic() - started

        self.assertTrue(waited["success"], waited)
        self.assertLess(elapsed, 20, "the wait ran to its budget on a child that had stopped")
        self.assertEqual([], waited["awaiting"])
        self.assertEqual(
            [child_id],
            [entry["agent_id"] for entry in waited["children_waiting_for_a_message"]],
        )
        self.assertIn("complete_agent", waited["note"])
        self.assertIn("send_message", waited["note"])

    def test_one_blocked_child_does_not_end_a_wait_on_a_working_sibling(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        self.worker.hold = release
        self.worker.blocked_threads.add("worker-thread-1")
        session = self._session()
        session.start()
        blocked_id = self._delegate(session, "The section that stops")
        self._wait_until(lambda: self._status_of(session, blocked_id) == "blocked")
        working_id = self._delegate(session, "The section still being written")
        session._scheduler.external_await_budget = 0.5

        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [blocked_id, working_id]},
        ))
        self.assertTrue(waited["success"], waited)
        self.assertFalse(waited["settled"], waited)
        self.assertEqual([working_id], waited["awaiting"])
        self.assertIn("await_children again", waited["note"])

        release.set()
        session._scheduler.external_await_budget = 30
        settled = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [blocked_id, working_id]},
        ))
        self.assertTrue(settled["settled"], settled)
        self.assertEqual([], settled["awaiting"])
        self.assertEqual(
            [blocked_id],
            [entry["agent_id"] for entry in settled["children_needing_a_decision"]],
        )

    def test_a_prompt_to_an_external_primary_is_refused(self) -> None:
        session = self._session()
        session.start()
        with self.assertRaises(Exception):
            session.prompt("primary", "do the thing", "command-1")


class ExternalPrimaryReadsTheResultTests(unittest.TestCase):
    """A manager that waits for a worker gets back what the worker reported.

    Until this was built the outcome text went only to
    ``.vnext/outcomes/<session>.jsonl``, which a client vNext does not run
    cannot read from inside a tool call.  So the README's first job -- wait for
    the worker, then show me what came back -- could not be done.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.worker = ScriptedWorker()
        self.events: list[object] = []

    def _session(self) -> VNextRuntimeSession:
        request = SessionStartRequest(
            session_id="external-session-outcome",
            workspace=str(self.workspace),
            primary_agent_id="primary",
            primary={"provider": "external", "model": EXTERNAL_MODEL, "effort": "high"},
            main_preset={"instructions": "Coordinate this workforce."},
            catalog_config={"models": [
                {"provider": "external", "model": EXTERNAL_MODEL},
                {"provider": "codex", "model": WORKER_MODEL},
            ]},
            config={"external_client": "claude-code"},
        )
        session = VNextRuntimeSession(
            request, self.events.append,
            adapter_factories={"codex": lambda: self.worker},
        )
        self.addCleanup(session.close)
        return session

    def _delegate(self, session) -> str:
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        }))
        self.assertTrue(created["success"], created)
        return created["agent_id"]

    def test_await_children_hands_back_the_outcome(self) -> None:
        session = self._session()
        session.start()
        child_id = self._delegate(session)

        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        ))
        self.assertTrue(waited["settled"], waited)
        child = waited["children"][0]
        self.assertEqual(child["outcome"], "completed bounded execution")
        self.assertTrue(child["verified"])
        self.assertEqual(len(child["evidence"]), 10)
        self.assertEqual(child["evidence"][0], "scripted worker evidence 0")

    def test_a_long_report_is_cut_and_says_where_the_whole_text_is(self) -> None:
        self.worker.outcome_text = "a" * 9000
        session = self._session()
        session.start()
        child_id = self._delegate(session)

        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        ))
        outcome = waited["children"][0]["outcome"]
        self.assertLessEqual(len(outcome), 4000)
        self.assertTrue(outcome.startswith("aaaa"), outcome[:20])
        self.assertTrue(
            outcome.endswith(
                "… (cut; the full text is in "
                ".vnext/outcomes/external-session-outcome.jsonl)"
            ),
            outcome[-120:],
        )

    def test_inspect_of_a_finished_child_returns_its_outcome(self) -> None:
        session = self._session()
        session.start()
        child_id = self._delegate(session)
        session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        )

        seen = _read(session.external_tool_call(
            tool="inspect", arguments={"agent_id": child_id},
        ))
        self.assertEqual(seen["agent"]["status"], "completed")
        self.assertEqual(seen["agent"]["outcome"], "completed bounded execution")
        self.assertTrue(seen["agent"]["verified"])
        self.assertEqual(len(seen["agent"]["evidence"]), 10)

    def test_a_child_that_is_still_working_reports_no_outcome(self) -> None:
        release = threading.Event()
        self.worker.hold = release
        self.addCleanup(release.set)
        session = self._session()
        session.start()
        child_id = self._delegate(session)
        session._scheduler.external_await_budget = 0.5

        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]},
        ))
        self.assertFalse(waited["settled"], waited)
        child = waited["children"][0]
        self.assertNotIn("outcome", child)
        self.assertNotIn("verified", child)
        self.assertNotIn("evidence", child)

        seen = _read(session.external_tool_call(
            tool="inspect", arguments={"agent_id": child_id},
        ))
        self.assertNotIn("outcome", seen["agent"])

    def test_the_tool_descriptions_say_the_outcome_comes_back(self) -> None:
        session = self._session()
        session.start()
        descriptions = {
            str(tool.get("name") or ""): str(tool.get("description") or "")
            for tool in session.external_tools()
        }
        self.assertIn("outcome", descriptions["await_children"])
        self.assertIn("outcome", descriptions["inspect"])


class WorkforceSkillSaysWhereTheResultIsTests(unittest.TestCase):
    """The skill a manager reads says the outcome comes back through the tool."""

    SKILL = (Path(__file__).resolve().parents[1]
             / "plugins" / "vnext" / "skills" / "vnext-workforce" / "SKILL.md")

    def test_the_skill_names_the_three_fields(self) -> None:
        body = self.SKILL.read_text(encoding="utf-8")
        start = body.index("## await_children")
        section = body[start:body.index("\n## ", start + 4)]

        self.assertIn("`outcome`", section)
        self.assertIn("`verified`", section)
        self.assertIn("`evidence`", section)
        self.assertIn("4000", section)


class ACleanCloseIsRecordedAsOneTests(unittest.TestCase):
    """What the run record says when the client that held the root goes away.

    Closing the pipe cancels the root, because that is how a live conversation
    is released, and the record read `cancelled` for a session whose every
    worker had completed.  Three reviewers read a good run as a cancellation.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.worker = ScriptedWorker()
        self.events: list = []

    def _session(self) -> VNextRuntimeSession:
        request = SessionStartRequest(
            session_id="external-close-1",
            workspace=str(self.workspace),
            primary_agent_id="primary",
            primary={"provider": "external", "model": EXTERNAL_MODEL, "effort": "high"},
            main_preset={"instructions": "Coordinate this workforce."},
            catalog_config={"models": [
                {"provider": "external", "model": EXTERNAL_MODEL},
                {"provider": "codex", "model": WORKER_MODEL},
            ]},
            config={"external_client": "claude-code"},
        )
        session = VNextRuntimeSession(
            request, self.events.append,
            adapter_factories={"codex": lambda: self.worker},
        )
        self.addCleanup(session.close)
        session.start()
        return session

    def _delegate(self, session: VNextRuntimeSession) -> str:
        created = _read(session.external_tool_call(tool="delegate", arguments={
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        }))
        self.assertTrue(created["success"], created)
        return created["agent_id"]

    def _wait_for(self, session: VNextRuntimeSession, agent_id: str, status: str) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            for agent in session.snapshot()["agents"]:
                if agent["agent_id"] == agent_id and agent["status"] == status:
                    return
            time.sleep(0.05)
        raise AssertionError(f"{agent_id} never reached {status}: {session.snapshot()}")

    def _wait_for_session(self, session: VNextRuntimeSession, status: str) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if session.snapshot()["status"] == status:
                return
            time.sleep(0.05)
        raise AssertionError(f"session never reached {status}: {session.snapshot()['status']}")

    def _last(self, event_type: str, agent_id: str | None = None) -> dict:
        found = [
            dict(event.payload) for event in self.events
            if event.type == event_type and (agent_id is None or event.agent_id == agent_id)
        ]
        self.assertTrue(found, f"no {event_type} for {agent_id}: {self.events}")
        return found[-1]

    def test_a_close_with_every_worker_finished_is_completed(self) -> None:
        session = self._session()
        child_id = self._delegate(session)
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        ))
        self.assertTrue(waited["settled"], waited)

        session.close()

        self.assertEqual(
            {"status": "completed", "ended_by": "server close"},
            self._last("session.upsert"),
        )
        root = self._last("agent.terminal", "primary")
        self.assertEqual("completed", root["status"])
        self.assertEqual("server close", root["ended_by"])
        self.assertEqual("completed", self._last("agent.upsert", "primary")["status"])
        self.assertEqual("completed", self._last("agent.upsert", child_id)["status"])

    def test_a_close_with_nothing_delegated_is_completed(self) -> None:
        """A client that opened a session and asked for no work ended it well."""

        session = self._session()
        session.close()
        self.assertEqual(
            {"status": "completed", "ended_by": "server close"},
            self._last("session.upsert"),
        )
        self.assertEqual("completed", self._last("agent.terminal", "primary")["status"])

    def test_a_worker_still_live_at_close_is_still_cancelled(self) -> None:
        """Work that really was stopped keeps the word that says so."""

        self.worker.blocked_threads = {"worker-thread-1"}
        session = self._session()
        child_id = self._delegate(session)
        self._wait_for(session, child_id, "blocked")

        session.close()

        self.assertEqual({"status": "cancelled"}, self._last("session.upsert"))
        root = self._last("agent.terminal", "primary")
        self.assertEqual("cancelled", root["status"])
        self.assertNotIn("ended_by", root)
        self.assertEqual("cancelled", self._last("agent.upsert", child_id)["status"])

    def test_a_session_finished_with_complete_session_is_completed(self) -> None:
        """The route that ended `idle` also ends as a run that finished."""

        session = self._session()
        child_id = self._delegate(session)
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        ))
        self.assertTrue(waited["settled"], waited)
        done = _read(session.external_tool_call(tool="complete_session", arguments={
            "decision": "accepted",
            "summary": "the section is written",
            "criteria": {"section written": True},
        }))
        self.assertTrue(done["success"], done)
        self._wait_for_session(session, "idle")

        session.close()

        self.assertEqual(
            {"status": "completed", "ended_by": "server close"},
            self._last("session.upsert"),
        )

    def test_cancelling_a_child_that_already_finished_says_so(self) -> None:
        """The reviewer's probe: the reply must not be indistinguishable from a
        real cancel, and no cancel belongs in the record."""

        session = self._session()
        child_id = self._delegate(session)
        waited = _read(session.external_tool_call(
            tool="await_children", arguments={"agent_ids": [child_id]}, timeout=30,
        ))
        self.assertTrue(waited["settled"], waited)

        answer = _read(session.external_tool_call(
            tool="cancel_agent", arguments={"agent_id": child_id},
        ))

        self.assertTrue(answer["success"], answer)
        self.assertEqual("already-finished", answer["status"])
        self.assertEqual("completed", answer["agent_status"])
        self.assertEqual([], [
            dict(event.payload) for event in self.events
            if event.type == "command.acknowledged"
            and dict(event.payload).get("command") == "cancel_agent"
        ])
        self.assertEqual("completed", self._last("agent.upsert", child_id)["status"])


if __name__ == "__main__":
    unittest.main()

ExternalPrimarySessionTest.test_bad_claude_effort_is_refused_without_spawn_or_event = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ExternalPrimarySessionTest.test_bad_claude_effort_is_refused_without_spawn_or_event)
