from __future__ import annotations

import sys
import tempfile
import time
import threading
import unittest
import asyncio
import inspect
import io
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext.vnext_claude import (
    ClaudeCodeAdapter,
    ClaudeRuntimeError,
    _DEFAULT_BRIDGE_MODULE,
    _NEUTRAL_APPROVAL_EFFECTS,
    _fatal_code,
    _fatal_stage,
    _owned_package_import_prelude,
    _owned_package_module_command,
)
from vnext.vnext_app_server import project_tool_result
from vnext import vnext_claude_bridge
from vnext.vnext_claude_bridge import (
    BridgeError,
    _approval_detail,
    _assert_tools_not_auto_allowed,
    _requested_permission_mode,
    _tool_effect,
    _Bridge,
    _CREDENTIAL_OVERRIDES,
    _MAX_PENDING_EFFECTS,
    _MAX_PENDING_SYSTEM_MESSAGES,
    _TOOL_EFFECTS,
)
from vnext.vnext_claude_native_identity import NativeChildIdentity, NativeChildIdentityError
from vnext.vnext_diagnostics import FailureCategory
from vnext.vnext_runtime_effects import ClaudeRuntimeEffectReader, RuntimeEffectJournal
from vnext.vnext_runtime_types import NativeChildBinding, RuntimePosture, ToolCallResult, TurnHandle
from vnext.vnext_claude_terminal import ClaudeTerminalRelayEvent
from vnext.vnext_scheduler import (
    VNextScheduler,
    _BOUNDARY_DECLINE_REASONS,
    _MANAGER_DECLINE_REASON,
    _READ_ONLY_APPROVALS,
)


CLAUDE_WORKER_MODEL = "claude-opus-4-7"
REQUESTED_POSTURE = RuntimePosture(
    workspace_writes=True,
    network="restricted",
    approvals_requested=True,
    reviewer="auto_review",
    environment_ready=True,
)
RESOLVED_POSTURE = replace(REQUESTED_POSTURE, network="approval_gated")


class ClaudeOwnedBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Same Windows race ClaudeManagerHostingTests already names: an owned
        # bridge process can still be releasing its cwd when the case ends.
        # assert_clean has already checked what matters -- zero residual
        # processes, drained streams and handlers, no errors -- so a handle
        # the OS has not finished releasing is a teardown artefact, not a
        # result. It surfaced here once the suite grew long enough to slow
        # the fixture's exit. Checked rather than assumed: after a full run
        # no descendant of the spawn-child fixture survives, so the
        # ownership probe this class exists for is still doing its job.
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        # Resolved: a CI runner hands out an 8.3 short path, and the bridge
        # compares resolved paths, so an unresolved workspace reads as foreign.
        self.workspace = str(Path(self.temp.name).resolve())
        self.fixture = Path(__file__).with_name("claude_bridge_fixture.py")
        self.adapters: list[ClaudeCodeAdapter] = []

    def tearDown(self) -> None:
        for adapter in self.adapters:
            adapter.close()
        self.temp.cleanup()

    def adapter(self, mode: str = "normal", **kwargs: object) -> ClaudeCodeAdapter:
        adapter = ClaudeCodeAdapter(
            workspace=self.workspace,
            bridge_command=(sys.executable, str(self.fixture), mode),
            # The fixture is a fresh Python subprocess.  A 200 ms protocol
            # deadline flakes under concurrent integration tests before the
            # child can import and read its first JSONL line; individual
            # timeout behavior is exercised explicitly below with 50 ms.
            request_timeout_seconds=1.0,
            **kwargs,
        )
        self.adapters.append(adapter)
        return adapter

    def test_default_launcher_imports_its_own_package_before_workspace(self) -> None:
        """A workspace shadow cannot redirect the default bridge module.

        The subprocess imports only ``vnext_diagnostics`` to prove module
        provenance; it neither starts the bridge nor imports or creates a
        Claude SDK client. A shadow copy in the caller workspace exits 23 if
        it wins.
        """

        shadow = Path(self.workspace) / "vnext"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("", encoding="utf-8")
        (shadow / "vnext_diagnostics.py").write_text(
            "raise SystemExit(23)\n", encoding="utf-8"
        )
        default = ClaudeCodeAdapter(workspace=self.workspace)
        self.adapters.append(default)
        self.assertEqual(
            _owned_package_module_command(_DEFAULT_BRIDGE_MODULE), default._command
        )
        probe = (
            sys.executable,
            "-c",
            _owned_package_import_prelude()
            + "import vnext.vnext_diagnostics as module; print(module.__file__)",
        )
        result = subprocess.run(
            probe,
            cwd=self.workspace,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        own_package = Path(__file__).resolve().parents[1] / "vnext"
        self.assertTrue(Path(result.stdout.strip()).resolve().is_relative_to(own_package))

        custom = (sys.executable, str(self.fixture), "normal")
        explicit = ClaudeCodeAdapter(workspace=self.workspace, bridge_command=custom)
        self.adapters.append(explicit)
        self.assertEqual(custom, explicit._command)

    def assert_clean(self, adapter: ClaudeCodeAdapter) -> None:
        cleanup = adapter.close()
        self.assertEqual(cleanup.process.residual_count, 0, cleanup)
        self.assertTrue(cleanup.streams_drained, cleanup)
        self.assertTrue(cleanup.handler_threads_drained, cleanup)
        self.assertEqual(cleanup.errors, (), cleanup)

    def test_fatal_diagnostics_classify_the_boundary_without_the_message(self) -> None:
        self.assertEqual("bridge-protocol", _fatal_stage("Claude bridge emitted malformed JSONL"))
        self.assertEqual("native-child-validation", _fatal_stage("Claude native child task conflicts with prior observation"))
        self.assertEqual("tool-call-validation", _fatal_stage("Claude tool call has no active local turn"))
        self.assertEqual(
            "native-child-task-conflict",
            _fatal_code("Claude native child task conflicts with prior observation"),
        )
        self.assertIsNone(_fatal_code("Claude tool call has no active local turn"))
        self.assertIsNone(_fatal_stage(None))

    def start_thread(
        self,
        adapter: ClaudeCodeAdapter,
        *,
        model: str = CLAUDE_WORKER_MODEL,
        tools: tuple[dict, ...] = (),
        posture: RuntimePosture = REQUESTED_POSTURE,
    ) -> tuple[str, object]:
        return adapter.start_thread(
            model=model,
            developer_instructions="Claude test fixture",
            tools=tools,
            tool_handler=None,
            requested_posture=posture,
            workspace=self.workspace,
        )

    def test_completion_without_provider_identity_fails_closed(self) -> None:
        adapter = self.adapter()
        initialized = adapter.initialize()
        self.assertEqual(initialized["tool_support"], {"accepted": True, "requires_empty": False})
        self.assertNotIn("required_posture", initialized)
        thread_id, policy = self.start_thread(adapter)
        self.assertTrue(thread_id.startswith("claude-reservation-"))
        self.assertFalse(adapter.thread_identity_attestation(thread_id)["bound"])
        self.assertEqual(policy["posture"], RESOLVED_POSTURE.as_dict())
        self.assertEqual(
            adapter.tool_registration_attestation(thread_id)["model_id"],
            CLAUDE_WORKER_MODEL,
        )
        self.assertEqual(
            adapter.tool_registration_attestation(thread_id),
            {
                "provider_echo": True,
                "acknowledged": True,
                "tool_count": 0,
                "tool_names": [],
                "definition_sha256": None,
                "handler_registered": False,
                "model_id": CLAUDE_WORKER_MODEL,
            },
        )
        handle = adapter.start_turn(thread_id, "safe fixture request")
        self.assertEqual((handle.thread_id, handle.turn_id, handle.cursor), (thread_id, "fixture-turn", 1))
        with self.assertRaisesRegex(ClaudeRuntimeError, "completion lacks attested"):
            adapter.wait_turn(handle)
        self.assertFalse(adapter.thread_identity_attestation(thread_id)["bound"])
        self.assert_clean(adapter)

    def test_leaf_refuses_nonempty_tools_and_unsatisfied_posture(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        # A leaf supplies no tool handler, so tools handed to it are refused
        # before any definition is even read.
        with self.assertRaisesRegex(ClaudeRuntimeError, "exactly one caller handler"):
            self.start_thread(adapter, tools=({"name": "not-supported"},))
        with self.assertRaisesRegex(ClaudeRuntimeError, "cannot satisfy"):
            self.start_thread(
                adapter,
                posture=replace(REQUESTED_POSTURE, network="open"),
            )
        self.assert_clean(adapter)

    def test_turn_rejects_reviewer_drift_from_connected_thread(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        thread_id, policy = self.start_thread(
            adapter,
            posture=replace(REQUESTED_POSTURE, reviewer="user"),
        )
        self.assertEqual(policy["posture"]["reviewer"], "user")
        with self.assertRaisesRegex(ClaudeRuntimeError, "reviewer drifted"):
            adapter.start_turn(thread_id, "safe fixture request", approvals_reviewer="auto_review")
        self.assert_clean(adapter)

    def test_turn_carries_the_bound_effort_and_refuses_drift(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        thread_id, _ = adapter.start_thread(
            model=CLAUDE_WORKER_MODEL,
            developer_instructions="Claude test fixture",
            tools=(),
            tool_handler=None,
            requested_posture=REQUESTED_POSTURE,
            workspace=self.workspace,
            effort="low",
        )
        with self.assertRaisesRegex(ClaudeRuntimeError, "effort drifted"):
            adapter.start_turn(thread_id, "wrong effort", model=CLAUDE_WORKER_MODEL, effort="high")
        handle = adapter.start_turn(thread_id, "low effort", model=CLAUDE_WORKER_MODEL, effort="low")
        self.assertEqual(thread_id, handle.thread_id)
        self.assert_clean(adapter)

    def test_resume_requires_connected_evidence_and_preserves_reviewer(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        result = adapter.resume_thread(
            thread_id="native-resumed-session",
            model=CLAUDE_WORKER_MODEL,
            approvals_reviewer="user",
            workspace=self.workspace,
        )
        self.assertEqual(
            result["connection_evidence"],
            {"connected": True, "server_info_received": True},
        )
        self.assertEqual(result["policy"]["posture"], replace(RESOLVED_POSTURE, reviewer="user").as_dict())
        self.assertEqual(
            adapter.thread_identity_attestation("native-resumed-session")["binding_phase"],
            "resumed",
        )
        self.assert_clean(adapter)

    def test_attested_resume_keeps_local_reservation_out_of_native_session(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        adapter.resume_attested_thread(
            runtime_thread="claude-local-reservation",
            provider_session="native-resumed-session",
            model=CLAUDE_WORKER_MODEL,
            approvals_reviewer="user",
            workspace=self.workspace,
        )
        self.assertEqual(
            {
                "runtime_thread": "claude-local-reservation",
                "provider": "claude",
                "bound": True,
                "provider_session": "native-resumed-session",
                "binding_phase": "resumed",
                "synthetic": False,
            },
            adapter.thread_identity_attestation("claude-local-reservation"),
        )
        self.assert_clean(adapter)

    def test_fresh_attested_resume_replays_explicit_manager_registration(self) -> None:
        """A new controller has no cache, so it must re-attest root tools."""

        adapter = self.adapter()
        adapter.initialize()

        def handler(_tool, _arguments, _context):
            return ToolCallResult(True, {"ok": True})

        adapter.resume_attested_thread(
            runtime_thread="claude-local-reservation",
            provider_session="native-resumed-session",
            model=CLAUDE_WORKER_MODEL,
            tool_handler=handler,
            tools=MANAGER_TOOLS,
            developer_instructions="You are the Root Manager.",
            approvals_reviewer="user",
            workspace=self.workspace,
        )
        registration = adapter.tool_registration_attestation("claude-local-reservation")
        self.assertEqual(CLAUDE_WORKER_MODEL, registration["model_id"])
        self.assertEqual(2, registration["tool_count"])
        self.assertEqual(["delegate", "complete_session"], registration["tool_names"])
        self.assertTrue(registration["handler_registered"])
        self.assertEqual(
            [dict(value) for value in MANAGER_TOOLS],
            adapter._tool_definitions["claude-local-reservation"],
        )
        self.assertIs(handler, adapter._tool_handlers["claude-local-reservation"])
        self.assert_clean(adapter)

    def test_fresh_attested_resume_refuses_missing_registration_receipt(self) -> None:
        """A resumed provider client never enables controls without its receipt."""

        adapter = self.adapter("resume-missing-registration")
        adapter.initialize()
        with self.assertRaisesRegex(ClaudeRuntimeError, "tool registration"):
            adapter.resume_attested_thread(
                runtime_thread="claude-local-reservation",
                provider_session="native-resumed-session",
                model=CLAUDE_WORKER_MODEL,
                tool_handler=lambda *_args: ToolCallResult(True, {}),
                tools=MANAGER_TOOLS,
                developer_instructions="You are the Root Manager.",
                workspace=self.workspace,
            )
        self.assertFalse(adapter.thread_identity_attestation("claude-local-reservation")["bound"])
        with self.assertRaisesRegex(ClaudeRuntimeError, "unknown provider thread"):
            adapter.tool_registration_attestation("claude-local-reservation")
        self.assert_clean(adapter)

    def test_fresh_tool_resume_requires_explicit_developer_instructions(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        with self.assertRaisesRegex(ClaudeRuntimeError, "requires developer instructions"):
            adapter.resume_attested_thread(
                runtime_thread="claude-local-reservation",
                provider_session="native-resumed-session",
                model=CLAUDE_WORKER_MODEL,
                tool_handler=lambda *_args: ToolCallResult(True, {}),
                tools=MANAGER_TOOLS,
                workspace=self.workspace,
            )
        self.assertFalse(adapter.thread_identity_attestation("claude-local-reservation")["bound"])
        self.assert_clean(adapter)

    def test_idle_native_terminal_handoff_releases_then_resumes_same_session(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        runtime_thread = "claude-local-reservation"
        session_id = "59637658-108e-4c83-8407-b29a1c1869a8"
        adapter._threads[runtime_thread] = {
            "policy": {"posture": RESOLVED_POSTURE.as_dict()},
            "provider_session": session_id,
            "binding_phase": "bound",
            "generation": 0,
            "model": CLAUDE_WORKER_MODEL,
            "effort": "high",
            "active_turn": None,
        }
        with tempfile.TemporaryDirectory() as lease_root:
            launch = adapter.begin_native_terminal(
                runtime_thread,
                relay_handler=lambda _event: {"jsonrpc": "2.0", "id": 1, "result": {}},
                lease_dir=lease_root,
            )
            self.assertEqual(session_id, launch.native_session_id)
            self.assertNotIn("token", launch.environment)
            resumed = adapter.resume_after_native_terminal(runtime_thread, terminal_stopped=True)
        self.assertEqual(session_id, resumed["thread_id"])
        self.assertEqual("resumed", adapter.thread_identity_attestation(runtime_thread)["binding_phase"])
        self.assert_clean(adapter)

    def _idle_terminal_candidate(self, adapter: ClaudeCodeAdapter, runtime_thread: str, session_id: str) -> None:
        adapter._threads[runtime_thread] = {
            "policy": {"posture": RESOLVED_POSTURE.as_dict()},
            "provider_session": session_id,
            "binding_phase": "bound",
            "generation": 2,
            "model": CLAUDE_WORKER_MODEL,
            "effort": "high",
            "active_turn": None,
        }

    def test_terminal_handoff_is_refused_while_a_native_child_runs(self) -> None:
        """The handoff drops the SDK reader a running child's frame needs."""

        adapter = self.adapter()
        adapter.initialize()
        runtime_thread = "claude-terminal-busy-reservation"
        session_id = "59637658-108e-4c83-8407-b29a1c1869a8"
        self._idle_terminal_candidate(adapter, runtime_thread, session_id)
        adapter._native_child_tasks["claude-native-child-1"] = {
            "reservation_id": runtime_thread, "task_id": "task-1", "status": "running",
        }
        adapter._native_child_tasks["claude-native-child-2"] = {
            "reservation_id": runtime_thread, "task_id": "task-2", "status": "running",
        }
        adapter._native_child_tasks["claude-native-child-3"] = {
            "reservation_id": runtime_thread, "task_id": "task-3", "status": "completed",
        }
        adapter._native_child_tasks["claude-native-child-other"] = {
            "reservation_id": "claude-other-reservation", "task_id": "task-4", "status": "running",
        }
        with tempfile.TemporaryDirectory() as lease_root:
            with self.assertRaisesRegex(ClaudeRuntimeError, "2 native child task\\(s\\) are running"):
                adapter.begin_native_terminal(
                    runtime_thread,
                    relay_handler=lambda _event: {"jsonrpc": "2.0", "id": 1, "result": {}},
                    lease_dir=lease_root,
                )
        # The refusal leaves the reservation and its children exactly as they were.
        self.assertEqual({}, adapter._terminal_leases)
        self.assertEqual(
            "running", adapter._native_child_tasks["claude-native-child-1"]["status"]
        )
        self.assertIsNone(adapter._threads[runtime_thread]["active_turn"])
        adapter._native_child_tasks.clear()
        self.assert_clean(adapter)

    def test_terminal_handoff_still_works_with_no_running_native_child(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        runtime_thread = "claude-terminal-idle-reservation"
        session_id = "59637658-108e-4c83-8407-b29a1c1869a8"
        self._idle_terminal_candidate(adapter, runtime_thread, session_id)
        adapter._native_child_tasks["claude-native-child-done"] = {
            "reservation_id": runtime_thread, "task_id": "task-1", "status": "completed",
        }
        with tempfile.TemporaryDirectory() as lease_root:
            launch = adapter.begin_native_terminal(
                runtime_thread,
                relay_handler=lambda _event: {"jsonrpc": "2.0", "id": 1, "result": {}},
                lease_dir=lease_root,
            )
            self.assertEqual(session_id, launch.native_session_id)
            adapter.resume_after_native_terminal(runtime_thread, terminal_stopped=True)
        adapter._native_child_tasks.clear()
        self.assert_clean(adapter)

    def test_native_terminal_hook_turn_routes_tools_waits_and_interrupts(self) -> None:
        """The hook turn is local evidence, never a claimed SDK turn ID."""
        adapter = self.adapter()
        adapter.initialize()
        runtime_thread = "claude-terminal-hook-reservation"
        session_id = "59637658-108e-4c83-8407-b29a1c1869a8"
        adapter._threads[runtime_thread] = {
            "policy": {"posture": RESOLVED_POSTURE.as_dict()},
            "provider_session": session_id,
            "binding_phase": "bound",
            "generation": 3,
            "model": CLAUDE_WORKER_MODEL,
            "effort": "high",
            "active_turn": None,
        }
        adapter._tool_definitions[runtime_thread] = [{
            "name": "delegate", "description": "Start a tracked worker.", "inputSchema": {"type": "object"},
        }]
        adapter._tool_handlers[runtime_thread] = lambda _name, _arguments, _context: ToolCallResult(
            True, {"child": "worker-1"}
        )
        with tempfile.TemporaryDirectory() as lease_root:
            adapter.begin_native_terminal(
                runtime_thread,
                relay_handler=lambda _event: {"jsonrpc": "2.0", "id": 1, "result": {}},
                lease_dir=lease_root,
            )
            interrupted: list[bool] = []
            adapter.set_native_terminal_interrupt(runtime_thread, lambda: interrupted.append(True))
            relay = adapter.terminal_relay_handler(runtime_thread)
            self.assertEqual(
                {"continue": True},
                relay(ClaudeTerminalRelayEvent("lease-a", "hook", {"hook_event_name": "UserPromptSubmit"})),
            )
            started = next(event for event in adapter.events_since() if event["name"] == "terminal_turn_started")
            self.assertEqual("terminal-hook", started["turn_source"])
            self.assertEqual(runtime_thread, started["reservation_id"])
            self.assertEqual(3, started["generation"])
            self.assertEqual({"session": session_id}, started["provider_correlation"])
            turn = type("Turn", (), {"thread_id": runtime_thread, "turn_id": started["turn_reference"]})()
            adapter.interrupt(turn)
            self.assertEqual([True], interrupted)
            invoked = relay(ClaudeTerminalRelayEvent("lease-a", "mcp", {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "delegate", "arguments": {"objective": "inspect"}},
            }))
            self.assertFalse(invoked["result"]["isError"])
            self.assertEqual({"child": "worker-1"}, json.loads(invoked["result"]["content"][0]["text"]))
            mcp = next(event for event in reversed(adapter.events_since()) if event["name"] == "terminal_mcp_tool")
            self.assertEqual(started["turn_reference"], mcp["turn_reference"])
            self.assertEqual("terminal-hook", mcp["turn_source"])
            self.assertEqual(
                {"continue": True},
                relay(ClaudeTerminalRelayEvent("lease-a", "hook", {"hook_event_name": "Stop"})),
            )
            self.assertEqual("interrupted", adapter.wait_turn(turn, timeout_seconds=0.1)["status"])
            finished = next(event for event in adapter.events_since() if event["name"] == "terminal_turn_completed")
            self.assertEqual(started["turn_reference"], finished["turn_reference"])
            adapter.resume_after_native_terminal(runtime_thread, terminal_stopped=True)
        self.assert_clean(adapter)

    def test_native_terminal_exit_settles_a_hook_turn_without_stop(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        runtime_thread = "claude-terminal-exit-reservation"
        session_id = "59637658-108e-4c83-8407-b29a1c1869a8"
        adapter._threads[runtime_thread] = {
            "policy": {"posture": RESOLVED_POSTURE.as_dict()},
            "provider_session": session_id,
            "binding_phase": "bound",
            "generation": 0,
            "model": CLAUDE_WORKER_MODEL,
            "effort": "high",
            "active_turn": None,
        }
        with tempfile.TemporaryDirectory() as lease_root:
            adapter.begin_native_terminal(
                runtime_thread,
                relay_handler=lambda _event: {"jsonrpc": "2.0", "id": 1, "result": {}},
                lease_dir=lease_root,
            )
            relay = adapter.terminal_relay_handler(runtime_thread)
            relay(ClaudeTerminalRelayEvent("lease-a", "hook", {"hook_event_name": "UserPromptSubmit"}))
            started = next(event for event in adapter.events_since() if event["name"] == "terminal_turn_started")
            turn = type("Turn", (), {"thread_id": runtime_thread, "turn_id": started["turn_reference"]})()
            adapter.resume_after_native_terminal(runtime_thread, terminal_stopped=True)
            self.assertEqual("interrupted", adapter.wait_turn(turn, timeout_seconds=0.1)["status"])
        self.assert_clean(adapter)

    def test_a_dropped_delta_notice_is_recorded_rather_than_fatal(self) -> None:
        """The bridge reports what it discarded.  The adapter has to take it."""

        adapter = self.adapter()
        adapter.initialize()
        reservation, _ = self.start_thread(adapter)
        adapter._record_event({"event": {
            "name": "stream_deltas_dropped", "reservation_id": reservation, "generation": 0,
            "turn_reference": "turn-a", "dropped": 5, "dropped_total": 5,
        }})

        recorded = adapter.events_since()[-1]
        self.assertEqual("stream_deltas_dropped", recorded["name"])
        self.assertEqual(reservation, recorded["reservation_id"])
        self.assertIsNone(adapter._fatal)
        self.assert_clean(adapter)

    def test_attested_native_agent_task_adopts_routes_tools_stops_and_waits(self) -> None:
        """A ledger-attested Claude task never borrows its parent's turn."""

        adapter = self.adapter()
        adapter.initialize()
        reservation, _ = self.start_thread(adapter)
        adapter._record_event({"event": {
            "name": "system", "reservation_id": reservation, "generation": 0,
            "native_identity": {"session_id": "parent-session", "source": "AssistantMessage"},
        }})
        observations = []
        tool_contexts = []

        def observe(observation):
            observations.append(observation)
            return NativeChildBinding(
                agent_id="worker-1",
                provider="claude",
                native_thread_id=observation.native_child_thread_id,
                delivery_contract=observation.delivery_contract,
                tool_handler=lambda _name, _args, context: tool_contexts.append(context) or ToolCallResult(True, {"ok": True}),
            )

        adapter.set_native_child_observer(observe, parent_agent_resolver=lambda thread: "manager" if thread == reservation else None)
        adapter._threads[reservation]["generation"] = 1
        child_runtime = "claude-native:parent-session:agent-child"
        task_event = {
            "name": "native_child", "reservation_id": reservation, "generation": 1,
            "turn_reference": "task-1", "parent_turn_reference": "parent-turn-1",
            "provider_correlation": {"session": "parent-session", "turn": "parent-turn-1"},
            "correlation_attested": True, "tracking": "attested", "status": "running",
            "native_agent_id": "agent-child", "native_runtime_thread_id": child_runtime,
            "parent_tool_use_id": "tool-parent", "task_id": "task-1",
            "task_contract": {
                "role": "worker", "role_source": "vnext-native-task-mapping",
                "objective": "Inspect one bounded area", "requested_model": CLAUDE_WORKER_MODEL,
                "model_source": "explicit",
            },
        }
        adapter._record_event({"event": task_event})
        self.assertEqual(1, len(observations))
        self.assertEqual(child_runtime, observations[0].native_child_thread_id)
        child_identity = adapter.thread_identity_attestation(child_runtime)
        self.assertEqual("native-task-attested", child_identity["binding_phase"])
        self.assertEqual(
            {
                "origin": "native",
                "native_agent_id": "agent-child",
                "native_task_id": "task-1",
                "parent_runtime_thread_id": reservation,
                "parent_local_turn_reference": "parent-turn-1",
                "parent_turn_source": "bridge",
                "parent_tool_use_id": "tool-parent",
            },
            {key: child_identity[key] for key in (
                "origin", "native_agent_id", "native_task_id", "parent_runtime_thread_id",
                "parent_local_turn_reference", "parent_turn_source", "parent_tool_use_id",
            )},
        )
        stale_running = dict(task_event, generation=2)
        adapter._record_event({"event": stale_running})
        self.assertEqual("native_child_rejected", adapter.events_since()[-1]["name"])
        self.assertEqual(1, len(observations))
        adapter._record_event({"event": {
            "name": "tool_call", "reservation_id": reservation, "generation": 1,
            "turn_reference": "task-1", "call_id": "child-call", "tool": "inspect", "arguments": {},
            "native_runtime_thread_id": child_runtime, "native_child_task_id": "task-1",
        }})
        self.assertEqual(child_runtime, tool_contexts[0].thread_id)
        self.assertEqual("task-1", tool_contexts[0].turn_id)
        self.assertEqual(
            "interrupt-requested", adapter.cancel_native_child(child_runtime)["status"]
        )
        self.assertTrue(adapter.interrupt(TurnHandle(child_runtime, "task-1"))["interrupted"])
        adapter._record_event({"event": {
            "name": "native_child_usage", "reservation_id": reservation, "generation": 1,
            "turn_reference": "task-1", "task_id": "task-1", "native_runtime_thread_id": child_runtime,
            "parent_turn_reference": "parent-turn-1",
            "provider_correlation": {"session": "parent-session", "turn": "parent-turn-1"},
            "correlation_attested": True, "usage": {"input_tokens": 3}, "usage_source": "native_task",
        }})
        adapter._record_event({"event": {
            "name": "native_child_completed", "reservation_id": reservation, "generation": 1,
            "turn_reference": "task-1", "task_id": "task-1", "native_runtime_thread_id": child_runtime,
            "parent_turn_reference": "parent-turn-1",
            "provider_correlation": {"session": "parent-session", "turn": "parent-turn-1"},
            "correlation_attested": True, "status": "completed", "summary": "bounded task complete",
        }})
        result = adapter.wait_turn(TurnHandle(child_runtime, "task-1"), timeout_seconds=0.1)
        self.assertEqual("completed", result["status"])
        self.assertTrue(result["native_task_terminal"])
        self.assertEqual({"input_tokens": 3}, result["usage"])
        self.assertEqual("bounded task complete", result["summary"])
        # Replayed lifecycle evidence has a later adapter cursor but must not
        # reset an already terminal task or create another managed child.
        adapter._record_event({"event": dict(task_event, cursor=99)})
        self.assertEqual("completed", adapter.wait_turn(TurnHandle(child_runtime, "task-1"), timeout_seconds=0.1)["status"])
        self.assertEqual("native_child_rejected", adapter.events_since()[-1]["name"])
        adapter._record_event({"event": {
            "name": "tool_call", "reservation_id": reservation, "generation": 1,
            "turn_reference": "task-1", "call_id": "late-child-call", "tool": "inspect", "arguments": {},
            "native_runtime_thread_id": child_runtime, "native_child_task_id": "task-1",
        }})
        self.assertEqual(1, len(tool_contexts))
        rejected_usage = {
            "name": "native_child_usage", "reservation_id": reservation, "generation": 2,
            "turn_reference": "task-1", "task_id": "task-1", "native_runtime_thread_id": child_runtime,
            "parent_turn_reference": "parent-turn-1",
            "provider_correlation": {"session": "parent-session", "turn": "parent-turn-1"},
            "correlation_attested": True, "usage": {"input_tokens": 99}, "usage_source": "native_task",
        }
        adapter._record_event({"event": rejected_usage})
        self.assertEqual("native_child_usage_rejected", adapter.events_since()[-1]["name"])
        unsupported = dict(task_event, native_runtime_thread_id="claude-native:parent-session:agent-alias", task_id="task-alias", turn_reference="task-alias")
        unsupported["task_contract"] = dict(task_event["task_contract"], requested_model="opus")
        adapter._record_event({"event": unsupported})
        self.assertIsNone(adapter._fatal)
        self.assertEqual(1, len(adapter.native_child_attestations()))
        self.assert_clean(adapter)

    def test_native_terminal_relay_projects_mcp_tools_and_hook_evidence(self) -> None:
        adapter = self.adapter()
        runtime_thread = "claude-terminal-reservation"
        adapter._threads[runtime_thread] = {"policy": {"posture": RESOLVED_POSTURE.as_dict()}}
        adapter._tool_definitions[runtime_thread] = [{
            "name": "delegate",
            "description": "Start a tracked worker.",
            "inputSchema": {"type": "object"},
        }]
        calls: list[dict] = []
        events: list[dict] = []
        relay = adapter.native_terminal_relay_handler(
            runtime_thread,
            invoke_tool=lambda thread, name, args, context: calls.append({"thread": thread, "name": name, "args": dict(args), "context": dict(context)}) or {"success": True, "value": {"child": "worker-1"}},
            emit_event=lambda _thread, value: events.append(dict(value)) or None,
        )
        listing = relay(ClaudeTerminalRelayEvent("lease-a", "mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
        self.assertEqual("delegate", listing["result"]["tools"][0]["name"])
        invoked = relay(ClaudeTerminalRelayEvent("lease-a", "mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "delegate", "arguments": {"objective": "inspect"}}}))
        self.assertEqual({"child": "worker-1"}, json.loads(invoked["result"]["content"][0]["text"]))
        self.assertEqual(runtime_thread, calls[0]["thread"])
        hook = relay(ClaudeTerminalRelayEvent("lease-a", "hook", {"hook_event_name": "SubagentStart", "agent_id": "native-child"}))
        self.assertEqual({"continue": True}, hook)
        self.assertEqual("SubagentStart", events[-1]["hook_event_name"])
        self.assertEqual("lease-a", events[-1]["lease_id"])
        self.assert_clean(adapter)

    def test_unbound_permission_race_uses_only_a_local_routing_handle(self) -> None:
        approvals: list[dict] = []

        def approve(method: str, params: dict) -> dict:
            approvals.append({"method": method, "params": params})
            return {"decision": "decline"}

        adapter = self.adapter("permission", native_approval_handler=approve)
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        with self.assertRaisesRegex(ClaudeRuntimeError, "completion lacks attested"):
            adapter.wait_turn(handle)
        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0]["method"], "approval/request")
        envelope = approvals[0]["params"]
        self.assertEqual(envelope["approval_reference"], "claude:fixture-tool-use")
        self.assertEqual(envelope["provider"], "claude")
        self.assertEqual(
            envelope["routing_handle"],
            {"reservation_id": handle.thread_id, "turn_reference": handle.turn_id},
        )
        self.assertFalse(envelope["correlation_attested"])
        # The fixture's permission event names no effect, so `effect` is absent
        # by the fail-closed rule and the exact key set stays correct as it is.
        # The positive case is covered by
        # `test_permission_envelope_publishes_only_a_neutral_effect`.
        self.assertEqual(
            {
                "approval_reference",
                "provider",
                "routing_handle",
                "correlation_attested",
                "event",
            },
            set(envelope),
        )
        self.assert_clean(adapter)

    def test_permission_envelope_publishes_only_a_neutral_effect(self) -> None:
        # `VNextScheduler.review_approval` reads `params["effect"]` at the
        # envelope top level, so a neutral effect must arrive there, and an
        # effect this boundary cannot name must not arrive at all.
        for requested, expected in (
            ("modify", "modify"),
            ("execute", "execute"),
            ("exfiltrate", None),
        ):
            with self.subTest(effect=requested):
                approvals: list[dict] = []

                def approve(method: str, params: dict) -> dict:
                    approvals.append({"method": method, "params": params})
                    return {"decision": "decline"}

                adapter = self.adapter(native_approval_handler=approve)
                adapter.initialize()
                reservation_id, _ = self.start_thread(adapter)
                handle = adapter.start_turn(reservation_id, "bind before approval")
                adapter._record_event({"event": {
                    "name": "system", "reservation_id": reservation_id, "generation": 1,
                    "native_identity": {
                        "session_id": "native-attested-session",
                        "source": "AssistantMessage",
                    },
                }})
                with patch.object(adapter, "_send"):
                    adapter._record_event({"event": {
                        "name": "permission", "reservation_id": reservation_id,
                        "turn_reference": handle.turn_id, "generation": 1,
                        "tool_use_id": "effect-tool-use", "effect": requested,
                        "provider_correlation": {
                            "session": "native-attested-session",
                            "turn": handle.turn_id,
                            "request": "effect-tool-use",
                        },
                        "correlation_attested": True,
                    }})

                self.assertEqual(1, len(approvals))
                envelope = approvals[0]["params"]
                if expected is None:
                    self.assertNotIn("effect", envelope)
                else:
                    self.assertEqual(expected, envelope["effect"])
                self.assert_clean(adapter)

    def test_a_granted_workers_read_and_search_reach_the_reviewer(self) -> None:
        """The decision a Claude worker's read of an outside path gets.

        A field run handed a granted worker "manager declined this approval"
        for a Read of an absolute path, and the session record held no approval
        at all.  The cause is this boundary: it forwarded "modify" and
        "execute" only, so a read reached `review_approval` with no effect,
        fell to its else branch and was declined before the worker's standing
        grant was ever read.  `_READ_ONLY_APPROVALS` already held the entries
        that branch could not reach.
        """

        for effect in sorted(_READ_ONLY_APPROVALS):
            with self.subTest(effect=effect):
                approvals: list[dict] = []

                def approve(method: str, params: dict) -> dict:
                    approvals.append({"method": method, "params": params})
                    return {"decision": "accept"}

                adapter = self.adapter(native_approval_handler=approve)
                adapter.initialize()
                reservation_id, _ = self.start_thread(adapter)
                handle = adapter.start_turn(reservation_id, "bind before approval")
                adapter._record_event({"event": {
                    "name": "system", "reservation_id": reservation_id, "generation": 1,
                    "native_identity": {
                        "session_id": "native-attested-session",
                        "source": "AssistantMessage",
                    },
                }})
                with patch.object(adapter, "_send") as send:
                    adapter._record_event({"event": {
                        "name": "permission", "reservation_id": reservation_id,
                        "turn_reference": handle.turn_id, "generation": 1,
                        "tool_use_id": "outside-read", "effect": effect,
                        "provider_correlation": {
                            "session": "native-attested-session",
                            "turn": handle.turn_id,
                            "request": "outside-read",
                        },
                        "correlation_attested": True,
                    }})

                self.assertEqual(1, len(approvals))
                self.assertEqual(effect, approvals[0]["params"]["effect"])
                self.assertEqual(
                    "accept", send.call_args.args[0]["decision"]["decision"]
                )
                self.assert_clean(adapter)

    def test_every_effect_this_boundary_emits_is_one_the_reviewer_reads(self) -> None:
        """Three lists have to agree, and nothing else makes them.

        `_TOOL_EFFECTS` says what the bridge emits, `_NEUTRAL_APPROVAL_EFFECTS`
        says what the adapter forwards, and `review_approval` says what the
        reviewer branches on.  "read" and "network" sat in the first and the
        third and were missing from the second, which is the whole defect.
        """

        self.assertEqual(
            {"modify", "execute"} | set(_READ_ONLY_APPROVALS),
            set(_NEUTRAL_APPROVAL_EFFECTS),
        )
        self.assertEqual(
            set(), set(_TOOL_EFFECTS.values()) - set(_NEUTRAL_APPROVAL_EFFECTS)
        )

    def test_the_only_envelope_left_with_no_effect_is_an_unnamed_tool(self) -> None:
        """What else can still reach the reviewer's else branch.

        `_TOOL_EFFECTS` is a closed table over the four neutral effects, so no
        fifth effect value can be emitted from here.  One shape remains: a tool
        absent from the table carries no effect at all.  That request is the
        else branch's, and the branch has to name itself as the decliner --
        `test_the_reviewers_else_branch_records_the_request_and_names_itself`
        in the scheduler suite holds that half.
        """

        self.assertEqual(
            {"modify", "execute", "read", "network"}, set(_TOOL_EFFECTS.values())
        )
        for unnamed in ("SlashCommand", "Monitor", "Artifact"):
            self.assertEqual({}, _tool_effect(unnamed), unnamed)
        self.assertTrue(
            _BOUNDARY_DECLINE_REASONS["unnamed-effect"].startswith("vNext declined")
        )
        self.assertNotIn("manager", _BOUNDARY_DECLINE_REASONS["unnamed-effect"])
        self.assertEqual("manager declined this approval", _MANAGER_DECLINE_REASON)

    def test_bound_permission_uses_only_the_attested_native_session(self) -> None:
        approvals: list[dict] = []

        def approve(method: str, params: dict) -> dict:
            approvals.append({"method": method, "params": params})
            return {"decision": "decline"}

        adapter = self.adapter(native_approval_handler=approve)
        adapter.initialize()
        reservation_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(reservation_id, "bind before approval")
        adapter._record_event({"event": {
            "name": "system", "reservation_id": reservation_id, "generation": 1,
            "native_identity": {"session_id": "native-attested-session", "source": "AssistantMessage"},
        }})
        with patch.object(adapter, "_send") as send:
            adapter._record_event({"event": {
                "name": "permission", "reservation_id": reservation_id,
                "turn_reference": handle.turn_id, "generation": 1,
                "tool_use_id": "bound-tool-use", "effect": "execute",
                "provider_correlation": {
                    "session": "native-attested-session", "turn": handle.turn_id,
                    "request": "bound-tool-use",
                },
                "correlation_attested": True,
            }})
        self.assertEqual(1, len(approvals))
        envelope = approvals[0]["params"]
        self.assertEqual(
            envelope["provider_correlation"],
            {"session": "native-attested-session", "turn": handle.turn_id, "request": "bound-tool-use"},
        )
        self.assertTrue(envelope["correlation_attested"])
        self.assertNotIn("routing_handle", envelope)
        sent = send.call_args.args[0]
        self.assertEqual("decline", sent["decision"]["decision"])
        self.assert_clean(adapter)

    def test_missing_local_reservation_attestation_fails_closed(self) -> None:
        adapter = self.adapter("missing-echo")
        adapter.initialize()
        with self.assertRaisesRegex(ClaudeRuntimeError, "connected local reservation"):
            self.start_thread(adapter)
        self.assert_clean(adapter)

    def test_steer_reports_that_the_runtime_must_queue_it(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        with self.assertRaisesRegex(ClaudeRuntimeError, "queue it for the next turn"):
            adapter.steer(handle, "do something")
        self.assert_clean(adapter)

    def test_owned_descendant_probe_normal_close(self) -> None:
        adapter = self.adapter("spawn-child")
        adapter.initialize()
        ready = Path(self.workspace) / "fixture-child-ready"
        deadline = time.monotonic() + 2.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "fixture descendant did not report ready")
        self.assert_clean(adapter)

    def test_interrupt_cleanup(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        self.assertTrue(adapter.interrupt(handle)["interrupted"])
        self.assert_clean(adapter)

    def test_compact_round_trips_through_the_owned_bridge(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        self.assertEqual(
            {"compacted": True, "trigger": "manual", "pre_tokens": 15919, "post_tokens": 1999},
            dict(adapter.compact(thread_id)),
        )
        self.assert_clean(adapter)

    def test_init_failure_cleanup(self) -> None:
        adapter = self.adapter("init-failure")
        with self.assertRaisesRegex(ClaudeRuntimeError, "fixture init failure"):
            adapter.initialize()
        self.assert_clean(adapter)

    def test_timeout_cleanup(self) -> None:
        adapter = self.adapter("timeout")
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        with self.assertRaisesRegex(ClaudeRuntimeError, "timed out"):
            adapter.wait_turn(handle, timeout_seconds=0.05)
        self.assert_clean(adapter)

    def _auth_error_event(self, thread_id: str) -> dict:
        """What the bridge projects from the SDK's first 401, verbatim."""

        return {
            "event": {
                "name": "provider_error",
                "reservation_id": thread_id,
                "generation": 1,
                "provider_error": {
                    "source": "AssistantMessage",
                    "is_error": True,
                    "code": "authentication_failed",
                    "api_error_status": None,
                    "retry_after_seconds": None,
                },
            }
        }

    def test_a_refused_credential_ends_the_turn_on_the_first_event(self) -> None:
        """The "timeout" fixture never answers a wait, which is exactly the
        three minutes a worker with a wrong key spent looking busy while the
        SDK retried the 401 behind it. The event is enough to end it."""

        adapter = self.adapter("timeout")
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        adapter._record_event(self._auth_error_event(thread_id))
        started = time.monotonic()
        with self.assertRaises(ClaudeRuntimeError) as raised:
            adapter.wait_turn(handle, timeout_seconds=30)
        spent = time.monotonic() - started
        self.assertLess(spent, 2.0, f"waited {spent:.1f}s")
        said = str(raised.exception)
        self.assertIn("refused the credential", said)
        self.assertIn("retrying will not help", said)
        self.assertEqual(
            ["provider_error"], [event["name"] for event in adapter._events]
        )
        self.assert_clean(adapter)

    def test_a_401_with_no_code_reads_the_same_way(self) -> None:
        adapter = self.adapter("timeout")
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        record = self._auth_error_event(thread_id)
        record["event"]["provider_error"] = {
            "source": "ResultMessage",
            "is_error": True,
            "subtype": "error_during_execution",
            "api_error_status": 401,
            "rate_limited": False,
            "terminal_reason": "api_error",
        }
        adapter._record_event(record)
        with self.assertRaisesRegex(ClaudeRuntimeError, "HTTP 401"):
            adapter.wait_turn(handle, timeout_seconds=30)
        self.assert_clean(adapter)

    def test_a_provider_error_that_is_not_the_credential_leaves_the_turn_alone(
        self,
    ) -> None:
        adapter = self.adapter("timeout")
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "safe fixture request")
        record = self._auth_error_event(thread_id)
        record["event"]["provider_error"]["code"] = "rate_limit"
        adapter._record_event(record)
        with self.assertRaisesRegex(ClaudeRuntimeError, "timed out"):
            adapter.wait_turn(handle, timeout_seconds=0.05)
        self.assertEqual(
            ["provider_error"], [event["name"] for event in adapter._events]
        )
        self.assert_clean(adapter)

    def test_malformed_bridge_record_fails_closed(self) -> None:
        adapter = self.adapter("malformed")
        with self.assertRaisesRegex(ClaudeRuntimeError, "malformed JSONL"):
            adapter.initialize()
        self.assert_clean(adapter)

    def test_oversized_bridge_record_fails_closed(self) -> None:
        adapter = self.adapter("oversized")
        with self.assertRaisesRegex(ClaudeRuntimeError, "oversized JSONL"):
            adapter.initialize()
        self.assert_clean(adapter)

    def test_stale_response_id_fails_closed(self) -> None:
        adapter = self.adapter("stale-response")
        with self.assertRaisesRegex(ClaudeRuntimeError, "no pending request"):
            adapter.initialize()
        self.assert_clean(adapter)

    def test_late_read_only_diagnostic_response_is_discarded_without_fatal(self) -> None:
        """A timed-out probe cannot turn its own delayed reply into a fault."""

        adapter = self.adapter()
        adapter._closing = True
        adapter._expired_read_only_requests.append(41)
        adapter._owned = SimpleNamespace(
            process=SimpleNamespace(
                stdout=io.StringIO('{"v":1,"kind":"response","id":41,"ok":true,"result":{}}\n'),
                poll=lambda: 1,
            )
        )

        adapter._read_stdout()

        self.assertIsNone(adapter._fatal)
        self.assertEqual(1, adapter.diagnostics()["adapter"]["late_read_only_response_count"])
        adapter._owned = None

    def test_late_response_to_a_timed_out_turn_does_not_kill_the_bridge(self) -> None:
        """One worker's overrun must not take down its siblings.

        Measured 2026-09-18 on a GLM worker: wait_turn hit the scheduler's
        600s bound, the provider answered afterwards, and the late reply was
        read as a forged response.  The bridge went fatal, a sibling turn died
        on it, and the session followed.
        """

        adapter = self.adapter()
        adapter._closing = True
        adapter._expired_requests.append(7)
        adapter._owned = SimpleNamespace(
            process=SimpleNamespace(
                stdout=io.StringIO(
                    '{"v":1,"kind":"response","id":7,"ok":true,'
                    '"result":{"status":"completed"}}\n'
                ),
                poll=lambda: 1,
            )
        )

        adapter._read_stdout()

        self.assertIsNone(adapter._fatal)
        self.assertEqual(1, adapter.diagnostics()["adapter"]["late_response_count"])
        adapter._owned = None

    def test_a_response_this_adapter_never_issued_still_fails_closed(self) -> None:
        """The forgery guard the tombstone must leave standing."""

        adapter = self.adapter()
        adapter._closing = True
        adapter._owned = SimpleNamespace(
            process=SimpleNamespace(
                stdout=io.StringIO('{"v":1,"kind":"response","id":9001,"ok":true,"result":{}}\n'),
                poll=lambda: 1,
            )
        )

        adapter._read_stdout()

        self.assertEqual("Claude bridge response has no pending request", adapter._fatal)
        adapter._owned = None

    def test_oversized_request_fails_before_write(self) -> None:
        adapter = self.adapter()
        adapter.initialize()
        with self.assertRaisesRegex(ClaudeRuntimeError, "exceeds JSONL bound"):
            adapter._request("probe", {"value": "x" * 65_537})
        self.assert_clean(adapter)

    def test_parser_does_not_reuse_stale_request_id(self) -> None:
        bridge = _Bridge()
        source = io.StringIO(
            '{"v":1,"kind":"request","id":7,"op":"unsupported","payload":{}}\n'
            'not-json\n'
        )
        sink = io.StringIO()
        with patch("sys.stdin", source), patch("sys.stdout", sink):
            asyncio.run(bridge.run())
        records = [json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertEqual([record["id"] for record in records], [7, 0])

    def test_adapter_binds_identity_emitted_during_start_turn_acknowledgement(self) -> None:
        adapter = ClaudeCodeAdapter(workspace=self.workspace, bridge_command=(sys.executable, str(self.fixture)))
        self.addCleanup(adapter.close)
        adapter._threads["reservation-a"] = {
            "policy": {"posture": RESOLVED_POSTURE.as_dict()},
            "provider_session": None,
            "binding_phase": "reserved",
            "generation": 0,
        }

        def acknowledge(op: str, payload: dict, **_: object) -> dict:
            self.assertEqual(op, "start_turn")
            adapter._record_event({"event": {
                "name": "system", "reservation_id": payload["reservation_id"], "generation": payload["generation"],
                "native_identity": {"session_id": "native-race-safe", "source": "AssistantMessage"},
            }})
            return {"reservation_echo": payload["reservation_id"], "turn_echo": payload["turn_reference"], "turn_id": payload["turn_reference"], "cursor": 0}

        with patch.object(adapter, "_request", side_effect=acknowledge):
            handle = adapter.start_turn("reservation-a", "race-safe first turn")
        self.assertEqual(handle.thread_id, "reservation-a")
        self.assertEqual(adapter.thread_identity_attestation("reservation-a")["provider_session"], "native-race-safe")

    def test_global_cursor_excludes_prior_turn_effects_from_second_poll(self) -> None:
        adapter = self.adapter("attested-effects")
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        effects = RuntimeEffectJournal(self.workspace)
        effects.bind_reader("claude-agent", ClaudeRuntimeEffectReader())

        first = adapter.start_turn(thread_id, "first native fixture turn")
        self.assertEqual("completed", adapter.wait_turn(first)["status"])
        first_events = adapter.events_since(first.cursor)
        self.assertEqual([1, 2, 3], [event["cursor"] for event in first_events])
        effects.observe_turn(
            agent_id="claude-agent",
            thread_id=thread_id,
            turn_id=first.turn_id,
            events=first_events,
        )

        second = adapter.start_turn(thread_id, "second native fixture turn")
        self.assertEqual(4, second.cursor)
        self.assertEqual("completed", adapter.wait_turn(second)["status"])
        second_events = adapter.events_since(second.cursor)
        self.assertEqual([4, 5, 6], [event["cursor"] for event in second_events])
        self.assertEqual(
            {second.turn_id},
            {
                event["turn_reference"]
                for event in second_events
                if event["name"] != "system"
            },
        )
        effects.observe_turn(
            agent_id="claude-agent",
            thread_id=thread_id,
            turn_id=second.turn_id,
            events=second_events,
        )

        self.assertEqual(2, len(effects.records("claude-agent")))
        self.assertEqual(0, effects.summary("claude-agent")["uncorrelated_item_count"])
        self.assert_clean(adapter)

    def test_system_message_event_reaches_the_runtime_projection(self) -> None:
        from vnext.vnext_runtime_projection import project_native_event

        adapter = self.adapter("system-message")
        adapter.initialize()
        thread_id, _ = self.start_thread(adapter)
        handle = adapter.start_turn(thread_id, "init fixture turn")
        self.assertEqual("completed", adapter.wait_turn(handle)["status"])
        events = [event for event in adapter.events_since(handle.cursor) if event["name"] == "system_message"]
        self.assertEqual(1, len(events))
        projected = [item for item in project_native_event("claude", "root", events[0])
                     if item.type == "provider.system"]
        self.assertEqual(1, len(projected))
        self.assertEqual("init", projected[0].payload["subtype"])
        self.assertEqual(["compact", "context"], projected[0].payload["data"]["slash_commands"])
        self.assert_clean(adapter)

    def test_sdk_options_allow_base64_image_messages_up_to_64_mib(self) -> None:
        class FakeSDK:
            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

        FakeSDK.ClaudeAgentOptions.__dataclass_fields__ = {"max_buffer_size": None}
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = FakeSDK()
        options = bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a",
        )
        self.assertEqual(64 * 1024 * 1024, options.max_buffer_size)

    def test_sdk_options_omit_image_buffer_for_older_sdks(self) -> None:
        class FakeSDK:
            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                self.assertNotIn("max_buffer_size", kwargs)
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = FakeSDK()
        for fields in (None, {}):
            with self.subTest(dataclass_fields=fields):
                if fields is not None:
                    FakeSDK.ClaudeAgentOptions.__dataclass_fields__ = fields
                bridge._options(
                    model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a",
                )

    def test_fake_sdk_reservation_then_first_turn_binds_emitted_identity(self) -> None:
        class AssistantMessage:
            def __init__(self, session_id: str) -> None:
                self.session_id = session_id

        class ResultMessage(AssistantMessage):
            pass

        class FakeSDK:
            def __init__(self) -> None:
                self.clients: list[object] = []

            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def tool(name: str, description: str, schema: dict):
                def decorate(handler):
                    return SimpleNamespace(name=name, description=description, schema=schema, handler=handler)

                return decorate

            @staticmethod
            def create_sdk_mcp_server(name: str, version: str, *, tools: list) -> object:
                return SimpleNamespace(name=name, version=version, tools=tools)

            def ClaudeSDKClient(sdk_self, *, options: object):
                class Client:
                    def __init__(self) -> None:
                        self.options = options
                        self.prompts: list[str] = []

                    async def connect(self, prompt: object) -> None:
                        self.connect_prompt = prompt

                    async def get_server_info(self) -> dict:
                        return {"commands": []}

                    async def query(self, prompt: str) -> None:
                        self.prompts.append(prompt)

                    async def receive_messages(self):
                        yield AssistantMessage("native-after-first-turn")
                        yield ResultMessage("native-after-first-turn")

                    async def disconnect(self) -> None:
                        pass

                client = Client()
                sdk_self.clients.append(client)
                return client

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = FakeSDK()
        events: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=events.append):
            reserved = asyncio.run(bridge.dispatch("start_thread", {
                "workspace": self.workspace, "tools": [],
                "requested_posture": REQUESTED_POSTURE.as_dict(),
                "model": CLAUDE_WORKER_MODEL,
                "effort": "low",
                "reservation_id": "reservation-a",
            }))
            self.assertEqual(reserved["reservation_echo"], "reservation-a")
            self.assertIsNone(bridge._reservations["reservation-a"]["session_id"])
            asyncio.run(bridge.dispatch("start_turn", {
                "reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1,
                "prompt": "first provider turn",
            }))
            self.assertEqual(asyncio.run(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))["status"], "completed")
        self.assertEqual(bridge._reservations["reservation-a"]["session_id"], "native-after-first-turn")
        identity_events = [event["event"] for event in events if event["event"].get("native_identity")]
        self.assertEqual(identity_events[0]["native_identity"]["source"], "AssistantMessage")
        client = bridge._sdk.clients[0]
        self.assertIsNone(client.connect_prompt)
        self.assertEqual(client.prompts, ["first provider turn"])
        options = client.options
        self.assertIsNone(options.resume)
        self.assertEqual("low", options.effort)
        self.assertFalse(hasattr(options, "session_id"))

    def test_sdk_tool_blocks_emit_sanitized_effect_events_only_after_identity(self) -> None:
        class ToolUseBlock:
            def __init__(self, tool_id: str, name: str, input_data: dict) -> None:
                self.id, self.name, self.input = tool_id, name, input_data

        class ToolResultBlock:
            def __init__(self, tool_id: str, content: str, is_error: bool) -> None:
                self.tool_use_id, self.content, self.is_error = tool_id, content, is_error

        class UserMessage:
            def __init__(self, content: list[object]) -> None:
                self.content = content

        class AssistantMessage:
            def __init__(self, session_id: str, content: list[object]) -> None:
                self.session_id, self.content = session_id, content

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._reservations["reservation-a"] = {
            "generation": 1,
            "session_id": None,
            "turn_reference": "local-turn",
            "pending_effects": [],
            "tool_effect_types": {},
        }
        tool_use = ToolUseBlock("native-tool", "Bash", {"command": "private command"})
        tool_result = ToolResultBlock("native-tool", "private output", False)
        write_use = ToolUseBlock(
            "native-write",
            "Write",
            {
                "file_path": str(Path(self.workspace) / "nested" / "answer.txt"),
                "content": "PRIVATE FILE CONTENT",
            },
        )
        write_result = ToolResultBlock("native-write", "PRIVATE WRITE RESULT", False)
        events: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=events.append):
            bridge._emit_message(
                "reservation-a", 1, UserMessage([tool_use, tool_result, write_use, write_result])
            )
            self.assertEqual([], events)
            self.assertEqual(4, len(bridge._reservations["reservation-a"]["pending_effects"]))
            bridge._emit_message("reservation-a", 1, AssistantMessage("native-session", []))

        emitted = [record["event"] for record in events]
        self.assertEqual(
            ["system", "tool_use", "tool_result", "tool_use", "tool_result", "message", "message"],
            [event["name"] for event in emitted],
        )
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], [event["cursor"] for event in emitted])
        self.assertEqual("command", emitted[2]["effect_type"])
        self.assertEqual("completed", emitted[2]["status"])
        self.assertEqual("file", emitted[4]["effect_type"])
        self.assertEqual(
            [{"path": "nested/answer.txt", "kind": {"type": "not_reported"}}],
            emitted[4]["changes"],
        )
        self.assertFalse(emitted[4]["evidence_limited"])
        self.assertEqual(
            {"session": "native-session", "turn": "local-turn", "request": "native-tool"},
            emitted[2]["provider_correlation"],
        )
        encoded = json.dumps(events)
        self.assertNotIn("private command", encoded)
        self.assertNotIn("private output", encoded)
        self.assertNotIn("PRIVATE FILE CONTENT", encoded)
        self.assertNotIn("PRIVATE WRITE RESULT", encoded)
        self.assertNotIn(str(self.workspace), encoded)

        journal = RuntimeEffectJournal(Path(self.workspace))
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        effects = journal.observe_turn(
            agent_id="worker",
            thread_id="reservation-a",
            turn_id="local-turn",
            events=emitted,
        )
        self.assertEqual(["command", "file_change"], [effect.effect for effect in effects])
        self.assertFalse(effects[1].evidence_limited)
        self.assertEqual(
            [("nested/answer.txt", "not_reported")],
            [(change.path, change.kind) for change in effects[1].changes],
        )

    def test_a_long_worker_is_not_stopped_by_tool_calls_that_already_settled(self) -> None:
        # A live opus build Worker died at its 65th Bash/Write with "Claude
        # tool effect correlation capacity exceeded": every settled call kept
        # its slot for the life of the reservation.
        class ToolUseBlock:
            def __init__(self, tool_id: str) -> None:
                self.id, self.name, self.input = tool_id, "Bash", {"command": "true"}

        class ToolResultBlock:
            def __init__(self, tool_id: str) -> None:
                self.tool_use_id, self.content, self.is_error = tool_id, "", False

        class AssistantMessage:
            def __init__(self, content: list[object]) -> None:
                self.session_id, self.content = "native-session", content

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._reservations["reservation-a"] = {
            "generation": 1,
            "session_id": "native-session",
            "turn_reference": "local-turn",
            "pending_effects": [],
            "tool_effect_types": {},
        }
        with patch("vnext.vnext_claude_bridge._write"):
            for index in range(200):
                tool_id = f"tool-{index}"
                bridge._emit_message(
                    "reservation-a", 1, AssistantMessage([ToolUseBlock(tool_id), ToolResultBlock(tool_id)])
                )
            self.assertEqual({}, bridge._reservations["reservation-a"]["tool_effect_types"])
            # Calls still awaiting a result stay bounded.
            with self.assertRaisesRegex(Exception, "correlation capacity exceeded"):
                for index in range(65):
                    bridge._emit_message("reservation-a", 1, AssistantMessage([ToolUseBlock(f"open-{index}")]))

    def test_sdk_write_path_bounds_mark_invalid_facts_limited(self) -> None:
        class ToolUseBlock:
            def __init__(self, tool_id: str, input_data: object) -> None:
                self.id, self.name, self.input = tool_id, "Write", input_data

        class ToolResultBlock:
            def __init__(self, tool_id: str) -> None:
                self.tool_use_id, self.content, self.is_error = tool_id, "PRIVATE RESULT", False

        class UserMessage:
            def __init__(self, content: list[object]) -> None:
                self.content = content

        class AssistantMessage:
            def __init__(self, session_id: str) -> None:
                self.session_id, self.content = session_id, []

        cases = [
            ("missing", {}, []),
            ("malformed", {"file_path": 42, "content": "PRIVATE CONTENT"}, []),
            (
                "outside",
                {
                    "file_path": str(Path(self.workspace).parent / "outside-private.txt"),
                    "content": "PRIVATE CONTENT",
                },
                [{"path": "<outside-workspace>", "kind": {"type": "not_reported"}}],
            ),
        ]
        for label, input_data, expected_changes in cases:
            with self.subTest(label=label):
                bridge = _Bridge()
                bridge._workspace = Path(self.workspace)
                bridge._reservations["reservation-a"] = {
                    "generation": 1,
                    "session_id": None,
                    "turn_reference": "local-turn",
                    "pending_effects": [],
                    "tool_effect_types": {},
                }
                events: list[dict] = []
                with patch("vnext.vnext_claude_bridge._write", side_effect=events.append):
                    bridge._emit_message(
                        "reservation-a",
                        1,
                        UserMessage([ToolUseBlock("write-tool", input_data), ToolResultBlock("write-tool")]),
                    )
                    bridge._emit_message("reservation-a", 1, AssistantMessage("native-session"))
                result = next(record["event"] for record in events if record["event"]["name"] == "tool_result")
                self.assertTrue(result["evidence_limited"])
                self.assertEqual(expected_changes, result["changes"])
                encoded = json.dumps(events)
                self.assertNotIn("PRIVATE CONTENT", encoded)
                self.assertNotIn("PRIVATE RESULT", encoded)
                self.assertNotIn("outside-private.txt", encoded)

    def test_sdk_permission_callback_relays_unbound_then_bound_control_decisions(self) -> None:
        class PermissionResultAllow:
            pass

        class PermissionResultDeny:
            def __init__(self, *, message: str, interrupt: bool) -> None:
                self.message, self.interrupt = message, interrupt

        class FakeSDK:
            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

        FakeSDK.PermissionResultAllow = PermissionResultAllow
        FakeSDK.PermissionResultDeny = PermissionResultDeny

        async def scenario() -> tuple[list[dict], object, object]:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = FakeSDK()
            state = {
                "generation": 1,
                "session_id": None,
                "task": asyncio.current_task(),
                "turn_reference": "local-turn",
                "pending_permissions": {},
            }
            bridge._reservations["local-reservation"] = state
            options = bridge._options(
                model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="local-reservation"
            )
            events: list[dict] = []
            with patch("vnext.vnext_claude_bridge._write", side_effect=events.append):
                unbound = asyncio.create_task(
                    options.can_use_tool("Write", {}, SimpleNamespace(tool_use_id="first-tool"))
                )
                await asyncio.sleep(0)
                await bridge._serve(json.dumps({
                    "v": 1, "kind": "control", "op": "permission_response",
                    "reservation_id": "local-reservation", "turn_reference": "local-turn",
                    "tool_use_id": "first-tool", "decision": {"decision": "decline"},
                }))
                first = await unbound
                state["session_id"] = "native-bound-session"
                bound = asyncio.create_task(
                    options.can_use_tool("Bash", {}, SimpleNamespace(tool_use_id="second-tool"))
                )
                await asyncio.sleep(0)
                await bridge._serve(json.dumps({
                    "v": 1, "kind": "control", "op": "permission_response",
                    "reservation_id": "local-reservation", "turn_reference": "local-turn",
                    "tool_use_id": "second-tool", "decision": {"decision": "accept"},
                }))
                second = await bound
            return events, first, second

        events, first, second = asyncio.run(scenario())
        unbound, bound = (record["event"] for record in events)
        self.assertEqual([1, 2], [unbound["cursor"], bound["cursor"]])
        self.assertEqual(
            unbound["routing_handle"],
            {"reservation_id": "local-reservation", "turn_reference": "local-turn"},
        )
        self.assertFalse(unbound["correlation_attested"])
        self.assertNotIn("provider_correlation", unbound)
        self.assertEqual(
            bound["provider_correlation"],
            {"session": "native-bound-session", "turn": "local-turn", "request": "second-tool"},
        )
        self.assertTrue(bound["correlation_attested"])
        self.assertNotIn("routing_handle", bound)
        self.assertIsInstance(first, PermissionResultDeny)
        self.assertIsInstance(second, PermissionResultAllow)

    def test_declined_approval_denies_without_interrupting_the_worker(self) -> None:
        class PermissionResultDeny:
            def __init__(self, *, message: str, interrupt: bool) -> None:
                self.message, self.interrupt = message, interrupt

        class FakeSDK:
            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

        FakeSDK.PermissionResultDeny = PermissionResultDeny

        async def scenario() -> PermissionResultDeny:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = FakeSDK()
            bridge._reservations["local-reservation"] = {
                "generation": 1,
                "session_id": "native-session",
                "task": asyncio.current_task(),
                "turn_reference": "local-turn",
                "pending_permissions": {},
            }
            options = bridge._options(
                model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="local-reservation"
            )
            pending = asyncio.create_task(
                options.can_use_tool("Write", {}, SimpleNamespace(tool_use_id="declined-tool"))
            )
            await asyncio.sleep(0)
            await bridge._serve(json.dumps({
                "v": 1, "kind": "control", "op": "permission_response",
                "reservation_id": "local-reservation", "turn_reference": "local-turn",
                "tool_use_id": "declined-tool", "decision": {"decision": "decline"},
            }))
            return await pending

        result = asyncio.run(scenario())
        self.assertIsInstance(result, PermissionResultDeny)
        # A decline refuses the one effect and leaves the worker's turn
        # alive, so it can report what it needed the tool for.
        self.assertFalse(result.interrupt)
        self.assertIn("declined", result.message)
        self.assertIn("report", result.message)

    def test_a_decline_no_manager_made_does_not_say_the_manager_made_it(self) -> None:
        """What the worker reads when the boundary refused it, not a manager.

        The message was one fixed sentence naming the manager, printed on every
        decline including the ones where no manager was asked.  The reviewer's
        reason travels with the decision now; a decline that arrives without
        one names nobody.
        """

        class PermissionResultDeny:
            def __init__(self, *, message: str, interrupt: bool) -> None:
                self.message, self.interrupt = message, interrupt

        class FakeSDK:
            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

        FakeSDK.PermissionResultDeny = PermissionResultDeny

        async def scenario(decision: dict) -> PermissionResultDeny:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = FakeSDK()
            bridge._reservations["local-reservation"] = {
                "generation": 1,
                "session_id": "native-session",
                "task": asyncio.current_task(),
                "turn_reference": "local-turn",
                "pending_permissions": {},
            }
            options = bridge._options(
                model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="local-reservation"
            )
            pending = asyncio.create_task(
                options.can_use_tool("Read", {}, SimpleNamespace(tool_use_id="read-tool"))
            )
            await asyncio.sleep(0)
            await bridge._serve(json.dumps({
                "v": 1, "kind": "control", "op": "permission_response",
                "reservation_id": "local-reservation", "turn_reference": "local-turn",
                "tool_use_id": "read-tool", "decision": decision,
            }))
            return await asyncio.wait_for(pending, 2)

        boundary = _BOUNDARY_DECLINE_REASONS["unnamed-effect"]
        named = asyncio.run(
            scenario({"decision": "decline", "reason": boundary})
        )
        self.assertTrue(named.message.startswith(boundary))
        self.assertNotIn("manager", named.message)
        self.assertIn("report", named.message)
        self.assertFalse(named.interrupt)

        from_manager = asyncio.run(
            scenario({"decision": "decline", "reason": _MANAGER_DECLINE_REASON})
        )
        self.assertTrue(from_manager.message.startswith(_MANAGER_DECLINE_REASON))

        silent = asyncio.run(scenario({"decision": "decline"}))
        self.assertNotIn("manager", silent.message)
        self.assertIn("declined", silent.message)

    def test_fake_sdk_resume_connects_without_session_id_option(self) -> None:
        class FakeSDK:
            def __init__(self) -> None:
                self.clients: list[object] = []

            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            def ClaudeSDKClient(sdk_self, *, options: object):
                class Client:
                    def __init__(self) -> None:
                        self.options = options

                    async def connect(self, prompt: object) -> None:
                        self.connect_prompt = prompt

                    async def get_server_info(self) -> dict:
                        return {"commands": []}

                    async def disconnect(self) -> None:
                        pass

                client = Client()
                sdk_self.clients.append(client)
                return client

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = FakeSDK()
        result = asyncio.run(bridge.dispatch("resume", {
            "session_id": "native-resumed-session",
            "model": CLAUDE_WORKER_MODEL,
            "requested_posture": REQUESTED_POSTURE.as_dict(),
        }))
        self.assertEqual(
            result["connection_evidence"],
            {"connected": True, "server_info_received": True},
        )
        client = bridge._sdk.clients[0]
        self.assertIsNone(client.connect_prompt)
        self.assertEqual(client.options.resume, "native-resumed-session")
        self.assertFalse(hasattr(client.options, "session_id"))
        self.assertEqual(0, result["tool_registration"]["tool_count"])
        self.assertFalse(result["tool_registration"]["handler_registered"])
        asyncio.run(bridge.dispatch("close", {}))

    def test_terminal_release_disconnects_only_its_client_once_and_resumes(self) -> None:
        class FakeSDK:
            def __init__(self) -> None:
                self.clients: list[object] = []

            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            def ClaudeSDKClient(sdk_self, *, options: object):
                class Client:
                    def __init__(self) -> None:
                        self.options = options
                        self.disconnect_calls = 0

                    async def connect(self, prompt: object) -> None:
                        pass

                    async def get_server_info(self) -> dict:
                        return {"commands": []}

                    async def disconnect(self) -> None:
                        self.disconnect_calls += 1

                client = Client()
                sdk_self.clients.append(client)
                return client

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = FakeSDK()
        for reservation in ("ended", "running", "failed-before-identity"):
            asyncio.run(bridge.dispatch("start_thread", {
                "workspace": self.workspace, "tools": [],
                "requested_posture": REQUESTED_POSTURE.as_dict(),
                "model": CLAUDE_WORKER_MODEL,
                "reservation_id": reservation,
            }))
        ended, running, failed = bridge._sdk.clients
        bridge._reservations["ended"]["session_id"] = "native-ended-session"

        for _ in range(2):
            result = asyncio.run(bridge.dispatch("release_agent", {"reservation_id": "ended"}))
            self.assertTrue(result["released"])
        self.assertEqual(1, ended.disconnect_calls)
        self.assertEqual(0, running.disconnect_calls)
        self.assertIs(bridge._reservations["running"]["client"], running)

        resumed = asyncio.run(bridge.dispatch("resume", {
            "reservation_id": "ended", "session_id": "native-ended-session",
            "model": CLAUDE_WORKER_MODEL,
            "requested_posture": REQUESTED_POSTURE.as_dict(),
        }))
        self.assertEqual("ended", resumed["reservation_echo"])
        self.assertEqual("native-ended-session", bridge._sdk.clients[-1].options.resume)
        self.assertIs(bridge._reservations["ended"]["client"], bridge._sdk.clients[-1])
        self.assertEqual(0, running.disconnect_calls)

        asyncio.run(bridge.dispatch("release_agent", {"reservation_id": "failed-before-identity"}))
        restarted = asyncio.run(bridge.dispatch("restart_thread", {
            "workspace": self.workspace, "tools": [],
            "requested_posture": REQUESTED_POSTURE.as_dict(),
            "model": CLAUDE_WORKER_MODEL,
            "reservation_id": "failed-before-identity",
        }))
        self.assertEqual("failed-before-identity", restarted["reservation_echo"])
        self.assertEqual(1, failed.disconnect_calls)
        self.assertIsNone(bridge._sdk.clients[-1].options.resume)
        self.assertEqual(0, running.disconnect_calls)
        asyncio.run(bridge.dispatch("close", {}))

    def test_terminal_release_waits_for_the_sdk_reader_to_finish(self) -> None:
        class ResultMessage:
            pass

        class FakeClient:
            def __init__(self) -> None:
                self.reading = asyncio.Event()
                self.finish = asyncio.Event()
                self.disconnect_calls = 0

            async def query(self, prompt: str) -> None:
                pass

            async def receive_messages(self):
                self.reading.set()
                await self.finish.wait()
                yield ResultMessage()

            async def disconnect(self) -> None:
                self.disconnect_calls += 1

        async def scenario() -> None:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = object()
            client = FakeClient()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = client
            bridge._reservations["ended"] = state

            def bind_identity(_reservation: str, _generation: int, _message: object) -> None:
                state["session_id"] = "native-ended-session"

            with patch.object(bridge, "_emit_message", side_effect=bind_identity):
                reader = asyncio.create_task(bridge._run_query("ended", 1, "prompt"))
                state["task"] = reader
                await client.reading.wait()
                release = await bridge.dispatch("release_agent", {"reservation_id": "ended"})
                self.assertFalse(release["released"])
                self.assertEqual(0, client.disconnect_calls)
                client.finish.set()
                await reader
            self.assertEqual(1, client.disconnect_calls)
            self.assertIsNone(state["client"])

        asyncio.run(scenario())

    def test_cancelled_release_stops_only_its_active_reader(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.reading = asyncio.Event()
                self.disconnect_calls = 0

            async def query(self, prompt: str) -> None:
                pass

            async def receive_messages(self):
                self.reading.set()
                await asyncio.Event().wait()
                yield None

            async def disconnect(self) -> None:
                self.disconnect_calls += 1

        async def scenario() -> None:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = object()
            ended, sibling = FakeClient(), FakeClient()
            for reservation, client in (("ended", ended), ("sibling", sibling)):
                state = bridge._reservation_state("auto_review", None, [])
                state["client"] = client
                bridge._reservations[reservation] = state
            state = bridge._reservations["ended"]
            reader = asyncio.create_task(bridge._run_query("ended", 1, "prompt"))
            state["task"] = reader
            await ended.reading.wait()
            released = await bridge.dispatch("release_agent", {
                "reservation_id": "ended", "status": "cancelled",
            })
            self.assertTrue(released["released"])
            self.assertTrue(reader.cancelled())
            self.assertEqual(1, ended.disconnect_calls)
            self.assertEqual(0, sibling.disconnect_calls)
            self.assertIs(bridge._reservations["sibling"]["client"], sibling)

        asyncio.run(scenario())

    def test_empty_sentinel_never_binds_and_later_turn_uses_same_connected_client(self) -> None:
        class AssistantMessage:
            def __init__(self, session_id: str) -> None:
                self.session_id = session_id

        class ResultMessage(AssistantMessage):
            pass

        class FakeSDK:
            def __init__(self) -> None:
                self.clients: list[object] = []
                self.batches = [[AssistantMessage(""), ResultMessage("default")], [ResultMessage("native-id")], [ResultMessage("native-id")]]

            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            def ClaudeSDKClient(sdk_self, *, options: object):
                class Client:
                    def __init__(self) -> None:
                        self.options = options
                        self.prompts: list[str] = []
                        self.batch: list[object] = []

                    async def connect(self, prompt: object) -> None:
                        self.connect_prompt = prompt

                    async def get_server_info(self) -> dict:
                        return {"commands": []}

                    async def query(self, prompt: str) -> None:
                        self.prompts.append(prompt)
                        self.batch = sdk_self.batches.pop(0)

                    async def receive_messages(self):
                        for message in self.batch:
                            yield message

                    async def disconnect(self) -> None:
                        pass

                client = Client()
                sdk_self.clients.append(client)
                return client

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = FakeSDK()
        asyncio.run(bridge.dispatch("start_thread", {
            "workspace": self.workspace, "tools": [], "requested_posture": REQUESTED_POSTURE.as_dict(),
            "model": CLAUDE_WORKER_MODEL, "reservation_id": "reservation-a",
        }))
        with patch("vnext.vnext_claude_bridge._write"):
            asyncio.run(bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"}))
            with self.assertRaisesRegex(BridgeError, "completed before native"):
                asyncio.run(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))
        self.assertIsNone(bridge._reservations["reservation-a"]["session_id"])
        client = bridge._sdk.clients[0]
        self.assertIsNone(client.options.resume)
        with patch("vnext.vnext_claude_bridge._write"):
            asyncio.run(bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-b", "generation": 2, "prompt": "second"}))
            asyncio.run(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-b"}))
        self.assertEqual(bridge._reservations["reservation-a"]["session_id"], "native-id")
        with patch("vnext.vnext_claude_bridge._write"):
            asyncio.run(bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-c", "generation": 3, "prompt": "third"}))
            asyncio.run(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-c"}))
        self.assertEqual(client.prompts, ["first", "second", "third"])
        self.assertEqual(len(bridge._sdk.clients), 1)
        self.assertFalse(hasattr(client.options, "session_id"))

    def test_conflicting_or_stale_identity_is_rejected_without_rebinding(self) -> None:
        class AssistantMessage:
            def __init__(self, session_id: str) -> None:
                self.session_id = session_id

        bridge = _Bridge()
        bridge._reservations["reservation-a"] = {"generation": 2, "session_id": None, "task": None}
        with patch("vnext.vnext_claude_bridge._write"):
            bridge._emit_message("reservation-a", 2, AssistantMessage("native-a"))
            with self.assertRaisesRegex(BridgeError, "conflicting native"):
                bridge._emit_message("reservation-a", 2, AssistantMessage("native-b"))
            with self.assertRaisesRegex(BridgeError, "stale local generation"):
                bridge._emit_message("reservation-a", 1, AssistantMessage("native-a"))
        self.assertEqual(bridge._reservations["reservation-a"]["session_id"], "native-a")

    def test_query_reads_native_completion_after_parent_result(self) -> None:
        class ResultMessage:
            pass
        class TaskNotificationMessage:
            pass
        seen = []
        class Client:
            async def query(self, prompt):
                pass
            async def receive_messages(self):
                yield ResultMessage()
                yield TaskNotificationMessage()
                raise AssertionError("reader continued after all tasks completed")
        bridge = _Bridge()
        bridge._sdk, bridge._workspace = SimpleNamespace(), Path.cwd()
        state = {"client": Client(), "session_id": "native-session",
                 "native_children": {"child": {"status": "running"}}, "terminal": {}}
        bridge._reservations["reservation"] = state
        def emit(_reservation, _generation, message):
            seen.append(type(message).__name__)
            if isinstance(message, TaskNotificationMessage):
                state["native_children"]["child"]["status"] = "completed"
        with patch.object(bridge, "_emit_message", side_effect=emit):
            asyncio.run(bridge._run_query("reservation", 1, "bounded work"))
        self.assertEqual(["ResultMessage", "TaskNotificationMessage"], seen)

    def test_interrupt_keeps_the_sole_reader_until_native_child_terminal(self) -> None:
        """Stop releases the parent waiter, never the reader needed by a child."""

        class ResultMessage:
            session_id, result = "native-session", "interrupted"
            usage = total_cost_usd = model_usage = None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = True, None, "success"
            api_error_status, terminal_reason, errors = None, "aborted_streaming", ()

        class TaskNotificationMessage:
            task_id, session_id, status = "child-1", "native-session", "completed"
            usage = summary = None

        class Stream:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.messages: asyncio.Queue[object] = asyncio.Queue()

            async def __aiter__(self):
                self.started.set()
                while True:
                    message = await self.messages.get()
                    if message is StopAsyncIteration:
                        return
                    yield message

        class FakeClient:
            def __init__(self, stream: Stream) -> None:
                self.stream, self.receive_calls, self.interrupt_calls = stream, 0, 0

            async def query(self, prompt: str) -> None:
                pass

            def receive_messages(self) -> Stream:
                self.receive_calls += 1
                return self.stream

            async def interrupt(self) -> None:
                self.interrupt_calls += 1
                await self.stream.messages.put(ResultMessage())
                await asyncio.sleep(0)

        async def exercise() -> None:
            stream, bridge = Stream(), _Bridge()
            client = FakeClient(stream)
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = client
            state["native_children"] = {"child-1": {"status": "running"}}
            bridge._reservations["reservation-a"] = state
            original_emit = bridge._emit_message

            def emit(reservation: str, generation: int, message: object) -> None:
                original_emit(reservation, generation, message)
                if isinstance(message, TaskNotificationMessage):
                    state["native_children"]["child-1"]["status"] = "completed"

            with patch("vnext.vnext_claude_bridge._write"):
                with patch.object(bridge, "_emit_message", side_effect=emit):
                    await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
                    await stream.started.wait()
                    reader = state["task"]
                    with self.assertRaisesRegex(BridgeError, "no active local turn"):
                        await bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "other-turn"})
                    waiting = asyncio.create_task(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))
                    self.assertEqual({"reservation_echo": "reservation-a", "interrupted": True}, await bridge.dispatch("interrupt", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))
                    self.assertIs(reader, state["task"])
                    self.assertFalse(reader.done())
                    self.assertEqual("interrupted", (await waiting)["status"])
                    self.assertIs(reader, state["task"])
                    self.assertFalse(reader.done())
                    await stream.messages.put(TaskNotificationMessage())
                    await reader
            self.assertEqual(1, client.receive_calls)
            self.assertEqual(1, client.interrupt_calls)
            self.assertIsNone(state["task"])

        asyncio.run(exercise())

    def test_compact_sends_slash_compact_through_the_idle_reservation_client(self) -> None:
        """The SDK has no compact method; the CLI runs /compact sent as a prompt."""

        class SystemMessage:
            def __init__(self, subtype: str, data: dict) -> None:
                self.subtype, self.data = subtype, data

        class ResultMessage:
            session_id, result, is_error, subtype = "native-session", "", False, "success"

        boundary = SystemMessage("compact_boundary", {"compact_metadata": {
            "trigger": "manual", "pre_tokens": 15919, "post_tokens": 1999,
        }})

        class Client:
            def __init__(self, messages: list) -> None:
                self.messages, self.prompts = messages, []

            async def query(self, prompt: str) -> None:
                self.prompts.append(prompt)

            async def receive_messages(self):
                for message in self.messages:
                    yield message

        async def exercise() -> None:
            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", "native-session", [])
            client = Client([SystemMessage("status", {}), boundary, ResultMessage()])
            state["client"] = client
            bridge._reservations["reservation-a"] = state
            result = await bridge.dispatch("compact", {"reservation_id": "reservation-a"})
            self.assertEqual(["/compact"], client.prompts)
            self.assertEqual(
                {"reservation_echo": "reservation-a", "compacted": True,
                 "trigger": "manual", "pre_tokens": 15919, "post_tokens": 1999},
                dict(result),
            )
            self.assertIsNone(state["task"])
            self.assertEqual(0, state["generation"])
            # No boundary means the CLI answered as an ordinary prompt.
            state["client"] = Client([ResultMessage()])
            with self.assertRaisesRegex(BridgeError, "compact boundary"):
                await bridge.dispatch("compact", {"reservation_id": "reservation-a"})
            self.assertIsNone(state["task"])
            # A live reader owns receive_messages; compact must not race it.
            state["task"] = asyncio.get_running_loop().create_future()
            with self.assertRaisesRegex(BridgeError, "idle"):
                await bridge.dispatch("compact", {"reservation_id": "reservation-a"})
            state["task"].cancel()

        asyncio.run(exercise())

    def test_compact_emits_the_boundary_and_the_result_usage_as_events(self) -> None:
        """the host sees the compact the way it sees any turn: provider.system and usage."""

        class SystemMessage:
            subtype = "compact_boundary"
            data = {"session_id": "native-session", "cwd": "/private/path",
                    "compact_metadata": {"trigger": "manual", "pre_tokens": 15919, "post_tokens": 1999}}

        class ResultMessage:
            session_id, result = "native-session", ""
            usage = {"input_tokens": 3, "output_tokens": 120, "cache_read_input_tokens": 1999}
            total_cost_usd, model_usage = 0.01, None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = False, None, "success"
            api_error_status, terminal_reason, errors = None, None, ()

        class Client:
            async def query(self, _prompt: str) -> None:
                pass

            async def receive_messages(self):
                yield SystemMessage()
                yield ResultMessage()

        bridge = _Bridge()
        bridge._workspace, bridge._sdk = Path(self.workspace), object()
        state = bridge._reservation_state("auto_review", "native-session", [])
        state["client"] = Client()
        bridge._reservations["reservation-a"] = state
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            result = asyncio.run(bridge.dispatch("compact", {"reservation_id": "reservation-a"}))
        self.assertIs(True, result["compacted"])
        events = [json.loads(line)["event"] for line in stdout.getvalue().splitlines()]
        boundary = [event for event in events if event["name"] == "system_message"]
        self.assertEqual(1, len(boundary))
        self.assertEqual("compact_boundary", boundary[0]["subtype"])
        self.assertEqual({"trigger": "manual", "pre_tokens": 15919}, boundary[0]["data"]["compact_metadata"])
        self.assertNotIn("cwd", boundary[0]["data"])
        usage = [event for event in events if event["name"] == "usage"]
        self.assertEqual(1, len(usage))
        self.assertEqual(ResultMessage.usage, usage[0]["usage"])
        self.assertEqual(0.01, usage[0]["total_cost_usd"])
        self.assertEqual(len(events), bridge._event_cursor)

    def test_a_timed_out_compact_holds_the_next_prompt_until_the_bridge_is_free(self) -> None:
        """A compact slower than its RPC timeout must not let a prompt fail the session.

        The adapter runs against a real in-process bridge.  Compact keeps the
        bridge's sole reader after the adapter stops waiting, and a freshly
        resumed thread (generation 0) used to skip the bridge readiness check,
        so start_turn hit the bridge's busy reservation and raised.
        """

        import concurrent.futures

        class SystemMessage:
            subtype = "compact_boundary"
            data = {"session_id": "native-session", "compact_metadata": {"trigger": "manual", "pre_tokens": 9}}

        class ResultMessage:
            session_id, result = "native-session", ""
            usage = total_cost_usd = model_usage = None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = False, None, "success"
            api_error_status, terminal_reason, errors = None, None, ()

        release_compact = threading.Event()

        class Client:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def query(self, prompt: str) -> None:
                self.prompts.append(prompt)

            async def receive_messages(self):
                if self.prompts[-1] == "/compact":
                    while not release_compact.is_set():
                        await asyncio.sleep(0.01)
                    yield SystemMessage()
                yield ResultMessage()

        loop = asyncio.new_event_loop()
        runner = threading.Thread(target=loop.run_forever, daemon=True)
        runner.start()
        self.addCleanup(lambda: (loop.call_soon_threadsafe(loop.stop), runner.join(2), loop.close()))
        bridge = _Bridge()
        bridge._workspace, bridge._sdk = Path(self.workspace), object()
        state = bridge._reservation_state("auto_review", "native-session", [])
        state["client"], state["effort"] = Client(), "high"
        bridge._reservations["reservation-a"] = state
        compact_timeouts: list[float | None] = []

        def request(op: str, payload: dict, *, timeout_seconds: float | None = None, read_only: bool = False) -> dict:
            if op == "compact":
                compact_timeouts.append(timeout_seconds)
            future = asyncio.run_coroutine_threadsafe(bridge.dispatch(op, payload), loop)
            try:
                # The compact wait is cut short here the way the adapter's
                # own deadline would end it: the bridge keeps compacting.
                return dict(future.result(timeout=0.1 if op == "compact" else 5))
            except concurrent.futures.TimeoutError:
                raise ClaudeRuntimeError(f"Claude bridge timed out waiting for {op}") from None
            except BridgeError as exc:
                raise ClaudeRuntimeError(str(exc)) from None

        adapter = ClaudeCodeAdapter(workspace=self.workspace)
        adapter._threads["reservation-a"] = {
            "generation": 0, "active_turn": None, "model": CLAUDE_WORKER_MODEL, "effort": "high",
            "policy": {"posture": {"reviewer": "auto_review"}}, "provider_session": "native-session",
        }
        adapter._request = request
        with patch("vnext.vnext_claude_bridge._write"):
            with self.assertRaisesRegex(ClaudeRuntimeError, "timed out waiting for compact"):
                adapter.compact("reservation-a")
            self.assertGreaterEqual(compact_timeouts[0], 1800)
            # The bridge still compacts: the next prompt waits.
            self.assertFalse(adapter.can_start_turn("reservation-a"))
            release_compact.set()
            deadline = time.monotonic() + 2
            while not adapter.can_start_turn("reservation-a") and time.monotonic() < deadline:
                time.sleep(0.3)
            self.assertTrue(adapter.can_start_turn("reservation-a"))
            handle = adapter.start_turn("reservation-a", "after compact")
        self.assertEqual("after compact", state["client"].prompts[-1])
        self.assertEqual(1, state["generation"])
        self.assertEqual(handle.turn_id, adapter._threads["reservation-a"]["active_turn"])

    def test_error_result_keeps_reader_through_taskupdated_terminal(self) -> None:
        """A provider error must not discard a later exact child terminal."""

        class ResultMessage:
            session_id, result = "native-session", "failed"
            usage = total_cost_usd = model_usage = None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = True, None, "success"
            api_error_status, terminal_reason, errors = None, "api_error", ()

        class TaskUpdatedMessage:
            task_id, session_id, patch = "child-1", "native-session", {"status": "completed"}
            usage = summary = None

        async def exercise() -> None:
            messages = [ResultMessage(), TaskUpdatedMessage()]

            class Client:
                async def query(self, prompt: str) -> None:
                    pass

                async def receive_messages(self):
                    for message in messages:
                        yield message

            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = Client()
            state["native_children"] = {"child-1": {"status": "running"}}
            bridge._reservations["reservation-a"] = state
            original_emit = bridge._emit_message

            def emit(reservation: str, generation: int, message: object) -> None:
                original_emit(reservation, generation, message)
                if isinstance(message, TaskUpdatedMessage):
                    state["native_children"]["child-1"]["status"] = "completed"

            with patch("vnext.vnext_claude_bridge._write"), patch.object(bridge, "_emit_message", side_effect=emit):
                await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
                self.assertEqual("failed", (await bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))["status"])
                while state["task"] is not None:
                    await asyncio.sleep(0)
            self.assertEqual("completed", state["native_children"]["child-1"]["status"])

        asyncio.run(exercise())

    def test_interrupt_ack_does_not_relabel_a_nonabort_error(self) -> None:
        class ResultMessage:
            session_id, result = "native-session", "failed"
            usage = total_cost_usd = model_usage = None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = True, None, "success"
            api_error_status, terminal_reason, errors = None, "api_error", ()

        async def exercise() -> None:
            class Client:
                async def query(self, prompt: str) -> None:
                    pass

                async def receive_messages(self):
                    await asyncio.sleep(0)
                    yield ResultMessage()

                async def interrupt(self) -> None:
                    pass

            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = Client()
            bridge._reservations["reservation-a"] = state
            with patch("vnext.vnext_claude_bridge._write"):
                await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
                self.assertTrue((await bridge.dispatch("interrupt", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))["interrupted"])
                self.assertEqual("failed", (await bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))["status"])

        asyncio.run(exercise())

    def test_followup_result_after_child_wake_preserves_first_primary_outcome(self) -> None:
        """A child-triggered provider continuation is not a second user turn."""

        class ResultMessage:
            def __init__(self, result: str, is_error: bool, reason: str) -> None:
                self.session_id, self.result = "native-session", result
                self.usage = self.total_cost_usd = self.model_usage = None
                self.duration_ms = self.duration_api_ms = self.num_turns = None
                self.is_error, self.stop_reason, self.subtype = is_error, None, "success"
                self.api_error_status, self.terminal_reason, self.errors = None, reason, ()

        class TaskNotificationMessage:
            task_id, session_id, status = "child-1", "native-session", "completed"
            usage = summary = None

        async def exercise() -> None:
            first = ResultMessage("primary", False, "end_turn")
            continuation = ResultMessage("follow-up", True, "api_error")

            class Client:
                async def query(self, prompt: str) -> None:
                    pass

                async def receive_messages(self):
                    yield first
                    yield continuation
                    yield TaskNotificationMessage()

            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = Client()
            state["native_children"] = {"child-1": {"status": "running"}}
            bridge._reservations["reservation-a"] = state
            original_emit = bridge._emit_message

            def emit(reservation: str, generation: int, message: object) -> None:
                original_emit(reservation, generation, message)
                if isinstance(message, TaskNotificationMessage):
                    state["native_children"]["child-1"]["status"] = "completed"

            with patch("vnext.vnext_claude_bridge._write"), patch.object(bridge, "_emit_message", side_effect=emit):
                await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
                result = await bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"})
                while state["task"] is not None:
                    await asyncio.sleep(0)
            self.assertEqual("completed", result["status"])
            self.assertFalse(result["is_error"])
            self.assertEqual("end_turn", state["turn_outcomes"][1]["terminal"]["terminal_reason"])
            self.assertEqual("completed", state["native_children"]["child-1"]["status"])

        asyncio.run(exercise())

    def test_next_primary_is_refused_until_native_reader_drains(self) -> None:
        """Result frames have no public prompt id for safe overlap demultiplexing."""

        class ResultMessage:
            session_id, result = "native-session", "finished"
            usage = total_cost_usd = model_usage = None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = False, None, "success"
            api_error_status = terminal_reason = None
            errors = ()

        class Stream:
            def __init__(self) -> None:
                self.started, self.release = asyncio.Event(), asyncio.Event()

            async def __aiter__(self):
                self.started.set()
                yield ResultMessage()
                await self.release.wait()

        async def exercise() -> None:
            stream = Stream()

            class Client:
                async def query(self, prompt: str) -> None:
                    pass

                def receive_messages(self) -> Stream:
                    return stream

            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = Client()
            state["native_children"] = {"child-1": {"status": "running"}}
            bridge._reservations["reservation-a"] = state
            with patch("vnext.vnext_claude_bridge._write"):
                await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
                await stream.started.wait()
                await asyncio.sleep(0)
                readiness = await bridge.dispatch(
                    "can_start_turn", {"reservation_id": "reservation-a", "generation": 1}
                )
                self.assertEqual(
                    {"reservation_echo": "reservation-a", "generation": 1, "ready": False},
                    readiness,
                )
                with self.assertRaisesRegex(BridgeError, "stale reservation generation"):
                    await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-b", "generation": 2, "prompt": "second"})
                state["native_children"]["child-1"]["status"] = "completed"
                stream.release.set()
                while state["task"] is not None:
                    await asyncio.sleep(0)
                readiness = await bridge.dispatch(
                    "can_start_turn", {"reservation_id": "reservation-a", "generation": 1}
                )
                self.assertTrue(readiness["ready"])

        asyncio.run(exercise())

    def test_bridge_close_alone_cancels_reader_and_releases_waiter(self) -> None:
        """Cleanup owns reader cancellation; a primary Stop does not."""

        class Stream:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def __aiter__(self):
                self.started.set()
                await asyncio.Event().wait()
                yield None

        async def exercise() -> None:
            stream = Stream()

            class Client:
                def __init__(self) -> None:
                    self.disconnect_calls = 0

                async def query(self, prompt: str) -> None:
                    pass

                def receive_messages(self) -> Stream:
                    return stream

                async def disconnect(self) -> None:
                    self.disconnect_calls += 1

            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            client = Client()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = client
            bridge._reservations["reservation-a"] = state
            await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
            await stream.started.wait()
            waiter = asyncio.create_task(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))
            await bridge._cancel_all()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertEqual(1, client.disconnect_calls)
            self.assertIsNone(state["task"])

        asyncio.run(exercise())

    def test_cancelled_waiter_does_not_cancel_or_relabel_the_primary(self) -> None:
        """Cancelling a local wait is neither a provider Stop nor reader cleanup."""

        class Stream:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def __aiter__(self):
                self.started.set()
                await asyncio.Event().wait()
                yield None

        async def exercise() -> None:
            stream = Stream()

            class Client:
                async def query(self, prompt: str) -> None:
                    pass

                def receive_messages(self) -> Stream:
                    return stream

                async def disconnect(self) -> None:
                    pass

            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(self.workspace), object()
            state = bridge._reservation_state("auto_review", None, [])
            state["client"] = Client()
            bridge._reservations["reservation-a"] = state
            await bridge.dispatch("start_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a", "generation": 1, "prompt": "first"})
            await stream.started.wait()
            waiter = asyncio.create_task(bridge.dispatch("wait_turn", {"reservation_id": "reservation-a", "turn_reference": "turn-a"}))
            await asyncio.sleep(0)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            outcome = state["turn_outcomes"][1]
            self.assertFalse(outcome["completion"].done())
            self.assertFalse(state["task"].done())
            self.assertFalse(outcome["interrupt_requested"])
            await bridge._cancel_all()

        asyncio.run(exercise())

    def test_real_pinned_bridge_initializes_and_reserves_without_prompt(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        python = repo / ".venv-phase5" / "Scripts" / "python.exe"
        bridge = repo / "vnext" / "vnext_claude_bridge.py"
        if not python.is_file():
            self.skipTest("pinned Phase 5 SDK environment is unavailable")
        adapter = ClaudeCodeAdapter(
            workspace=self.workspace,
            bridge_command=(str(python), str(bridge)),
            request_timeout_seconds=10.0,
        )
        self.adapters.append(adapter)
        with patch.dict(os.environ, {}, clear=False):
            for name in _CREDENTIAL_OVERRIDES:
                os.environ.pop(name, None)
            adapter.initialize()
            reservation_id, _ = self.start_thread(adapter)
            self.assertFalse(adapter.thread_identity_attestation(reservation_id)["bound"])
        self.assert_clean(adapter)


MANAGER_TOOLS = (
    {
        "type": "function",
        "name": "delegate",
        "description": "Create one direct child chosen by this manager.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "objective": {"type": "string", "minLength": 1},
            },
            "required": ["role", "objective"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "complete_session",
        "description": "Record the Root Manager's final judgment.",
        "inputSchema": {
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
            "additionalProperties": False,
        },
    },
)


class ClaudeManagerHostingTests(unittest.TestCase):
    """A Claude session acting as a vNext manager, not only as a leaf."""

    def setUp(self) -> None:
        # An owned bridge process can still be releasing its cwd when the case
        # ends; that Windows race is a teardown artefact, not a result.
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.workspace = self.temp.name
        self.fixture = Path(__file__).with_name("claude_bridge_fixture.py")
        self.adapters: list[ClaudeCodeAdapter] = []

    def tearDown(self) -> None:
        for adapter in self.adapters:
            adapter.close()
        self.temp.cleanup()

    def adapter(self, mode: str = "normal", timeout: float = 5.0) -> ClaudeCodeAdapter:
        adapter = ClaudeCodeAdapter(
            workspace=self.workspace,
            bridge_command=(sys.executable, str(self.fixture), mode),
            request_timeout_seconds=timeout,
        )
        self.adapters.append(adapter)
        adapter.initialize()
        return adapter

    def bind_manager(self, adapter: ClaudeCodeAdapter, handler) -> str:
        thread_id, _ = adapter.start_thread(
            model=CLAUDE_WORKER_MODEL,
            developer_instructions="You are the Root Manager.",
            tools=MANAGER_TOOLS,
            tool_handler=handler,
            requested_posture=REQUESTED_POSTURE,
            workspace=self.workspace,
        )
        return thread_id

    def test_manager_binds_a_real_registration_and_a_worker_still_binds_none(self) -> None:
        adapter = self.adapter()
        manager_thread = self.bind_manager(adapter, lambda *args: ToolCallResult(True, {}))
        registration = adapter.tool_registration_attestation(manager_thread)
        self.assertTrue(registration["acknowledged"])
        self.assertEqual(2, registration["tool_count"])
        self.assertEqual(["delegate", "complete_session"], registration["tool_names"])
        self.assertTrue(registration["handler_registered"])
        self.assertIsInstance(registration["definition_sha256"], str)
        self.assertEqual(64, len(registration["definition_sha256"]))

        worker_thread, _ = adapter.start_thread(
            model=CLAUDE_WORKER_MODEL,
            developer_instructions="You are a Worker.",
            tools=(),
            tool_handler=None,
            requested_posture=REQUESTED_POSTURE,
            workspace=self.workspace,
        )
        # Phase 5E Gate 1: a Worker binding registers nothing at all.
        self.assertEqual(
            {
                "provider_echo": True,
                "acknowledged": True,
                "model_id": CLAUDE_WORKER_MODEL,
                "tool_count": 0,
                "tool_names": [],
                "definition_sha256": None,
                "handler_registered": False,
            },
            adapter.tool_registration_attestation(worker_thread),
        )

    def test_tools_without_a_handler_and_a_handler_without_tools_are_both_refused(self) -> None:
        adapter = self.adapter()
        with self.assertRaisesRegex(ClaudeRuntimeError, "exactly one caller handler"):
            adapter.start_thread(
                model=CLAUDE_WORKER_MODEL,
                developer_instructions="",
                tools=MANAGER_TOOLS,
                tool_handler=None,
                requested_posture=REQUESTED_POSTURE,
                workspace=self.workspace,
            )
        with self.assertRaisesRegex(ClaudeRuntimeError, "exactly one caller handler"):
            adapter.start_thread(
                model=CLAUDE_WORKER_MODEL,
                developer_instructions="",
                tools=(),
                tool_handler=lambda *args: ToolCallResult(True, {}),
                requested_posture=REQUESTED_POSTURE,
                workspace=self.workspace,
            )

    def test_malformed_manager_tool_definition_is_refused_before_a_session_exists(self) -> None:
        adapter = self.adapter()
        with self.assertRaisesRegex(ClaudeRuntimeError, "malformed definition"):
            adapter.start_thread(
                model=CLAUDE_WORKER_MODEL,
                developer_instructions="",
                tools=({"name": "delegate", "description": "no schema"},),
                tool_handler=lambda *args: ToolCallResult(True, {}),
                requested_posture=REQUESTED_POSTURE,
                workspace=self.workspace,
            )

    def test_manager_tool_call_round_trips_and_returns_the_handler_value(self) -> None:
        calls: list[tuple[str, dict, object]] = []

        def handler(tool, arguments, context):
            calls.append((tool, dict(arguments), context))
            return ToolCallResult(True, {"agent_id": "agent-7", "role": "worker"})

        adapter = self.adapter("manager-tools")
        thread_id = self.bind_manager(adapter, handler)
        turn = adapter.start_turn(thread_id, "root turn")
        self.assertEqual("completed", adapter.wait_turn(turn, timeout=10)["status"])

        self.assertEqual(1, len(calls))
        tool, arguments, context = calls[0]
        self.assertEqual("delegate", tool)
        self.assertEqual({"role": "worker", "objective": "fixture objective"}, arguments)
        self.assertEqual(thread_id, context.thread_id)
        self.assertEqual("fixture-call", context.call_id)

        echoed = [
            event["tool_answer_echo"]
            for event in adapter.events_since()
            if event.get("tool_answer_echo") is not None
        ]
        self.assertEqual(1, len(echoed))
        self.assertEqual(
            {
                "v": 1,
                "kind": "control",
                "op": "tool_call_response",
                "reservation_id": thread_id,
                "turn_reference": turn.turn_id,
                "call_id": "fixture-call",
                "result": {"success": True, "value": {"agent_id": "agent-7", "role": "worker"}},
            },
            echoed[0],
        )
        # A manager tool call is control plane, never an observed effect.
        self.assertEqual([], [event for event in adapter.events_since() if event["name"] == "tool_call"])

    def test_raising_manager_tool_handler_is_recorded_and_answers_the_model(self) -> None:
        def handler(tool, arguments, context):
            raise RuntimeError("handler defect")

        recorded: list[tuple] = []
        adapter = self.adapter("manager-tools")
        thread_id = self.bind_manager(adapter, handler)
        turn = adapter.start_turn(thread_id, "root turn")
        with patch(
            "vnext.vnext_claude.record_failure",
            side_effect=lambda category, **kwargs: recorded.append((category, kwargs)),
        ):
            self.assertEqual("completed", adapter.wait_turn(turn, timeout=10)["status"])

        tool_failures = [
            kwargs for category, kwargs in recorded
            if category is FailureCategory.MANAGER_TOOL_HANDLER_FAILED
        ]
        self.assertEqual(1, len(tool_failures))
        self.assertEqual("tool_call", tool_failures[0]["step"])
        self.assertEqual("RuntimeError", tool_failures[0]["exception_type"])
        # Content safety: only fixed labels, counts, flags and a duration.
        self.assertEqual(
            {"phase", "step", "exception_type", "duration_ms", "counts", "flags"},
            set(tool_failures[0]),
        )

        echoed = [
            event["tool_answer_echo"]
            for event in adapter.events_since()
            if event.get("tool_answer_echo") is not None
        ]
        self.assertEqual(
            {"success": False, "value": {"error_code": "tool-handler-failed"}},
            echoed[0]["result"],
        )

    def test_neutral_tool_result_projects_into_both_provider_wire_shapes(self) -> None:
        neutral = ToolCallResult(False, {"error_code": "not-direct-child"})
        self.assertEqual(
            {
                "success": False,
                "contentItems": [
                    {"type": "inputText", "text": '{"error_code":"not-direct-child"}'}
                ],
            },
            project_tool_result(neutral),
        )
        # The Claude projection carries the identical text in an SDK content
        # block, which is what keeps the scheduler free of either shape.
        self.assertEqual('{"error_code":"not-direct-child"}', neutral.as_json_text())


class ClaudeBridgeTranscriptProjectionTests(unittest.TestCase):
    """The bridge must give the host replayable content and accounting facts."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.bridge = _Bridge()
        self.bridge._workspace = Path(self.temp.name)
        state = self.bridge._reservation_state("auto_review", None, [])
        state.update({"generation": 1, "turn_reference": "turn-a", "workspace": Path(self.temp.name)})
        self.bridge._reservations["reservation-a"] = state

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_diagnostics_are_aggregate_and_exclude_reservation_identity(self) -> None:
        state = self.bridge._reservations["reservation-a"]
        state["session_id"] = "native-session"
        state["pending_permissions"]["toolu-private"] = object()
        state["pending_tool_calls"]["call-private"] = object()
        state["pending_messages"].append({"name": "message"})
        state["transcript"].append({"sequence": 1})
        state["turn_started_monotonic"] = time.monotonic() - 1.0
        state["last_message_monotonic"] = time.monotonic() - 0.5

        result = asyncio.run(self.bridge.dispatch("diagnostics", {}))

        self.assertEqual(1, result["reservation_count"])
        self.assertEqual(1, result["bound_session_count"])
        self.assertEqual(1, result["pending_permission_count"])
        self.assertEqual(1, result["pending_tool_call_count"])
        self.assertEqual(1, result["pending_message_count"])
        self.assertEqual(1, result["transcript_item_count"])
        self.assertNotIn("reservation-a", json.dumps(result))
        self.assertNotIn("native-session", json.dumps(result))

    def test_service_manager_options_expose_each_native_registration_boundary(self) -> None:
        """The real scheduler tool set must retain the bridge enrollment seam."""

        recorder: list = []
        bridge = self.bridge
        bridge._sdk = ClaudeBridgeToolHostingTests._fake_sdk(recorder)
        definitions = VNextScheduler.manager_tools()
        state = bridge._reservation_state("auto_review", None, definitions)
        state.update({"generation": 1, "turn_reference": "turn-a"})
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            definitions=definitions,
            permission_mode="acceptEdits",
        )

        options = recorder[-1]
        self.assertEqual("acceptEdits", options["permission_mode"])
        self.assertIn("PreToolUse", options["hooks"])
        self.assertIn("vnext", options["mcp_servers"])
        self.assertNotIn(
            "vnext_register_native_child",
            [tool.name for tool in options["mcp_servers"]["vnext"].tools],
        )
        relay = options["can_use_tool"]
        allowed = asyncio.run(
            relay(
                "mcp__vnext__vnext_register_native_child",
                {},
                SimpleNamespace(tool_use_id="provider-tool"),
            )
        )
        self.assertFalse(allowed.allow)
        diagnostic = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(1, diagnostic["enabled_reservation_count"])
        self.assertEqual(0, diagnostic["registration_server_count"])
        self.assertEqual(1, diagnostic["pretool_hook_count"])
        self.assertEqual(1, diagnostic["registration_permission_denied"])
        self.assertNotIn("reservation-a", json.dumps(diagnostic))

    def test_system_messages_forward_only_allowlisted_fields(self) -> None:
        class SystemMessage:
            def __init__(self, subtype: str, data: dict) -> None:
                self.subtype = subtype
                self.data = data

        class AssistantMessage:
            session_id = "native-session"
            content = []
            model = "claude-test"
            parent_tool_use_id = None
            stop_reason = None

        init = SystemMessage("init", {
            "type": "system", "subtype": "init", "session_id": "native-session",
            "cwd": "/private/workspace", "apiKeySource": "user", "model": "claude-test",
            "tools": ["Bash", "Read"], "slash_commands": ["compact", "context"],
            "mcp_servers": [{"name": "vnext", "status": "connected", "source": "https://host/?token=secret"}],
            "permissionMode": "default", "output_style": "default", "agents": ["general-purpose"],
            "skills": ["review"], "plugins": [{"name": "p", "path": "/private/plugin"}],
            "mcp_server_errors": [{"url": "https://host/?token=secret"}],
            "memory_paths": {"user": "/private/memory"}, "claude_code_version": "2.1.288", "uuid": "u-1",
        })
        compact = SystemMessage("compact_boundary", {
            "type": "system", "subtype": "compact_boundary", "session_id": "native-session", "uuid": "u-2",
            "compact_metadata": {"trigger": "manual", "pre_tokens": 1200, "user_context": "private words"},
        })
        hook = SystemMessage("hook_response", {"session_id": "native-session", "stdout": "token=secret"})
        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            # The init message arrives before the first assistant frame
            # attests the native session, so it is held, then released.
            self.bridge._emit_message("reservation-a", 1, init)
            self.assertEqual([], [r for r in written if r["event"]["name"] == "system_message"])
            self.bridge._emit_message("reservation-a", 1, AssistantMessage())
            self.bridge._emit_message("reservation-a", 1, compact)
            self.bridge._emit_message("reservation-a", 1, hook)
        events = [r["event"] for r in written if r["event"]["name"] == "system_message"]
        self.assertEqual(["init", "compact_boundary", "hook_response"], [e["subtype"] for e in events])
        self.assertEqual({
            "session_id": "native-session", "model": "claude-test", "tools": ["Bash", "Read"],
            "slash_commands": ["compact", "context"],
            "mcp_servers": [{"name": "vnext", "status": "connected"}],
            "permissionMode": "default", "output_style": "default", "agents": ["general-purpose"],
            "skills": ["review"], "claude_code_version": "2.1.288", "uuid": "u-1",
        }, events[0]["data"])
        self.assertEqual({
            "session_id": "native-session", "uuid": "u-2",
            "compact_metadata": {"trigger": "manual", "pre_tokens": 1200},
        }, events[1]["data"])
        self.assertEqual({"session_id": "native-session"}, events[2]["data"])
        self.assertTrue(all(e["correlation_attested"] for e in events))
        self.assertNotIn("secret", json.dumps(events))
        self.assertNotIn("/private", json.dumps(events))

    def test_structured_provider_rate_limit_is_projected_and_fails_the_turn(self) -> None:
        class AssistantMessage:
            session_id = "native-session"
            content = []
            model = "claude-test"
            stop_reason = None
            error = "rate_limit"

        class ResultMessage:
            session_id = "native-session"
            result = None
            usage = None
            total_cost_usd = None
            model_usage = None
            duration_ms = 1
            duration_api_ms = 1
            num_turns = 1
            is_error = True
            stop_reason = None
            subtype = "success"
            api_error_status = 429
            terminal_reason = "api_error"
            errors = ["provider prose must not be projected"]

        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            self.bridge._emit_message("reservation-a", 1, AssistantMessage())
            self.bridge._emit_message("reservation-a", 1, ResultMessage())
        errors = [record["event"] for record in written if record["event"]["name"] == "provider_error"]
        self.assertEqual("rate_limit", errors[0]["provider_error"]["code"])
        self.assertEqual(429, errors[1]["provider_error"]["api_error_status"])
        self.assertTrue(errors[1]["provider_error"]["rate_limited"])
        self.assertEqual(1, errors[1]["provider_error"]["error_count"])
        self.assertNotIn("provider prose", json.dumps(errors))
        self.assertTrue(self.bridge._reservations["reservation-a"]["terminal"]["is_error"])

        async def wait_for_terminal() -> dict:
            state = self.bridge._reservations["reservation-a"]
            state["turn_outcomes"] = {
                1: {
                    "turn_reference": "turn-a",
                    "interrupt_requested": False,
                    "completion": asyncio.get_running_loop().create_future(),
                }
            }
            self.bridge._complete_turn_outcome(state, 1, state["terminal"])
            return await self.bridge._wait_turn({"reservation_id": "reservation-a", "turn_reference": "turn-a"})

        self.assertEqual("failed", asyncio.run(wait_for_terminal())["status"])

    def test_messages_usage_and_read_thread_are_projected_without_losing_content(self) -> None:
        class TextBlock:
            def __init__(self, text: str) -> None:
                self.text = text

        class ThinkingBlock:
            def __init__(self, thinking: str) -> None:
                self.thinking, self.signature = thinking, "opaque-signature"

        class ToolUseBlock:
            def __init__(self) -> None:
                self.id, self.name, self.input = "toolu_1", "Read", {"file_path": "notes.md"}

        class ToolResultBlock:
            def __init__(self) -> None:
                self.tool_use_id, self.content, self.is_error = "toolu_1", "contents", False

        class AssistantMessage:
            def __init__(self) -> None:
                self.session_id = "native-session"
                self.content = [TextBlock("answer"), ThinkingBlock("reasoning"), ToolUseBlock(), ToolResultBlock()]
                self.model, self.stop_reason = "claude-test", "tool_use"

        class ResultMessage:
            def __init__(self) -> None:
                self.session_id = "native-session"
                self.result = "finished"
                self.usage = {"input_tokens": 12, "output_tokens": 8}
                self.total_cost_usd = 0.004
                self.model_usage = {"claude-test": {"inputTokens": 12, "outputTokens": 8, "costUSD": 0.004}}
                self.duration_ms, self.duration_api_ms, self.num_turns = 90, 80, 2
                self.is_error, self.stop_reason = False, "end_turn"

        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            self.bridge._emit_message("reservation-a", 1, AssistantMessage())
            self.bridge._emit_message("reservation-a", 1, ResultMessage())

        messages = [record["event"] for record in written if record["event"]["name"] == "message"]
        self.assertEqual("assistant", messages[0]["message"]["role"])
        self.assertEqual(
            ["text", "thinking", "tool_use", "tool_result"],
            [block["type"] for block in messages[0]["message"]["content"]],
        )
        self.assertEqual("native-session", messages[0]["provider_correlation"]["session"])
        usage = [record["event"] for record in written if record["event"]["name"] == "usage"]
        self.assertEqual({"input_tokens": 12, "output_tokens": 8}, usage[0]["usage"])
        self.assertEqual(0.004, usage[0]["total_cost_usd"])
        page = self.bridge._read_thread({"thread_id": "reservation-a", "after": 0, "limit": 10})
        self.assertEqual(2, len(page["items"]))
        self.assertEqual("answer", page["items"][0]["content"][0]["text"])
        self.assertEqual("finished", page["items"][1]["content"][0]["text"])

    def test_unenrolled_native_task_is_not_adopted_or_controllable(self) -> None:
        class TaskStartedMessage:
            task_id = "task-7"
            tool_use_id = "toolu_parent"
            session_id = "native-child-session"

        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            self.bridge._emit_message("reservation-a", 1, TaskStartedMessage())
        event = written[0]["event"]
        self.assertEqual("native_child", event["name"])
        self.assertEqual("toolu_parent", event["parent_tool_use_id"])
        self.assertEqual("native-child-session", event["reported_session_id"])
        self.assertIsNone(event["child_session_id"])
        self.assertEqual("unavailable", event["tracking"])
        self.assertEqual("awaiting-exact-metadata", event["status"])


class ClaudeBridgeToolHostingTests(unittest.TestCase):
    """In-process SDK tool hosting inside the owned bridge."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = self.temp.name

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _fake_sdk(recorder: list) -> object:
        class FakeSDK:
            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                recorder.append(kwargs)
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def tool(name: str, description: str, schema: dict):
                def decorate(handler):
                    return SimpleNamespace(
                        name=name, description=description, schema=schema, handler=handler
                    )

                return decorate

            @staticmethod
            def create_sdk_mcp_server(name: str, version: str, *, tools: list) -> object:
                return SimpleNamespace(name=name, version=version, tools=tools)

            @staticmethod
            def PermissionResultAllow(**kwargs: object) -> object:
                return SimpleNamespace(allow=True, **kwargs)

            @staticmethod
            def PermissionResultDeny(**kwargs: object) -> object:
                return SimpleNamespace(allow=False, **kwargs)

        return FakeSDK()

    def _bridge(self, recorder: list) -> _Bridge:
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = self._fake_sdk(recorder)
        return bridge

    def test_pending_native_child_eviction_becomes_a_run_note(self) -> None:
        from vnext.vnext_runtime_projection import project_native_event

        bridge = self._bridge([])
        state = bridge._reservation_state("auto_review", "native-session", [], reservation_id="reservation-a")
        state["turn_reference"] = "turn-a"
        written: list[dict] = []
        with patch.object(bridge, "_write_turn_event", side_effect=lambda _state, event: written.append(dict(event))):
            resolver = state["native_child_automatic_resolver"]
            for index in range(67):
                resolver.record_subagent_start("native-session", f"stale-{index}")
        self.assertEqual(["stale-0", "stale-1", "stale-2"], [event["agent_id"] for event in written])
        self.assertEqual(64, len(resolver.pending_agents("native-session")))
        notes = [projected for event in written
                 for projected in project_native_event("claude", "root", event)
                 if projected.type == "runtime.note"]
        self.assertEqual(3, len(notes))
        for index, note in enumerate(notes):
            self.assertIn(f"stale-{index} dropped from the identity join", note.payload["note"])

    def test_hosted_manager_tools_never_appear_in_allowed_tools(self) -> None:
        """The CRITICAL TRAP guard: a whole-tool allow entry skips can_use_tool.

        Naming an MCP tool in `allowed_tools` makes the SDK auto-approve it
        before the permission callback runs, which is exactly where vNext
        routes its approval rendezvous.  A manager tool must therefore never
        be listed there.
        """

        recorder: list = []
        bridge = self._bridge(recorder)
        bridge._reservations["reservation-a"] = bridge._reservation_state(
            "auto_review", None, list(MANAGER_TOOLS)
        )
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            definitions=list(MANAGER_TOOLS),
            developer_instructions="You are the Root Manager.",
        )
        options = recorder[-1]
        self.assertEqual([], options["allowed_tools"])
        for definition in MANAGER_TOOLS:
            self.assertNotIn(definition["name"], options["allowed_tools"])
            self.assertNotIn(f"mcp__vnext__{definition['name']}", options["allowed_tools"])
        # And the guard fails a binding that ever tried to.
        with self.assertRaisesRegex(BridgeError, "must not be auto-allowed"):
            _assert_tools_not_auto_allowed(["mcp__vnext__delegate"], list(MANAGER_TOOLS))
        with self.assertRaisesRegex(BridgeError, "must not be auto-allowed"):
            _assert_tools_not_auto_allowed(["delegate"], list(MANAGER_TOOLS))

    def test_editing_and_command_tools_name_an_effect_the_manager_understands(self) -> None:
        """The defect this replaces: only Write and Bash were ever named."""

        for tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            self.assertEqual({"effect": "modify"}, _tool_effect(tool), tool)
        for tool in ("Bash", "BashOutput", "KillShell"):
            self.assertEqual({"effect": "execute"}, _tool_effect(tool), tool)

    def test_a_tool_outside_the_table_names_nothing_and_stays_blocked(self) -> None:
        """An unnamed tool reaches the reviewer bare, and the reviewer declines it.

        This used to keep the two web tools out, on the reading that a
        Claude-family worker runs with network access withheld.  The field run
        behind F17 showed what that costs: a worker whose manager held a
        standing grant was refused two WebSearch calls and had no way to ask
        again, so both now name a "network" effect the manager decides on.
        What stays out is everything that runs a command or writes through a
        route this bridge cannot name.
        """

        for tool in ("SlashCommand", "Monitor", "Artifact", "CronCreate"):
            self.assertEqual({}, _tool_effect(tool), tool)

    def test_every_named_effect_is_one_the_manager_accepts(self) -> None:
        """The two files have to agree, and nothing else makes them.

        `review_approval` reads the effect off the envelope and declines every
        value it does not branch on.  Renaming an effect on either side breaks
        every Claude-family worker at its first write, which is how the missing
        Edit entry was found.  This test fails on that rename instead.
        """

        source = inspect.getsource(VNextScheduler.review_approval)
        for effect in sorted(set(_TOOL_EFFECTS.values())):
            branched = (
                f'effect == "{effect}"' in source or effect in _READ_ONLY_APPROVALS
            )
            self.assertTrue(branched, effect)


    def test_resume_rehosts_manager_tools_and_returns_their_registration(self) -> None:
        """Resume creates a new SDK client, so it must rebuild the MCP server."""

        recorder: list = []
        bridge = self._bridge(recorder)

        class Client:
            async def connect(self, prompt: object) -> None:
                self.prompt = prompt

            async def get_server_info(self) -> dict:
                return {"commands": []}

            async def disconnect(self) -> None:
                pass

        client = Client()
        bridge._sdk.ClaudeSDKClient = lambda *, options: (setattr(client, "options", options) or client)
        result = asyncio.run(bridge.dispatch("resume", {
            "session_id": "native-resumed-session",
            "reservation_id": "local-reservation",
            "model": CLAUDE_WORKER_MODEL,
            "requested_posture": REQUESTED_POSTURE.as_dict(),
            "tools": list(MANAGER_TOOLS),
            "developer_instructions": "You are the Root Manager.",
        }))
        self.assertEqual(
            ["delegate", "complete_session"],
            [tool.name for tool in client.options.mcp_servers["vnext"].tools],
        )
        self.assertEqual(
            "You are the Root Manager.", client.options.system_prompt["append"]
        )
        self.assertEqual(2, result["tool_registration"]["tool_count"])
        self.assertTrue(result["tool_registration"]["handler_registered"])
        asyncio.run(bridge.dispatch("close", {}))

    def test_manager_session_hosts_tools_and_preserves_normal_claude_capabilities(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        bridge._reservations["reservation-a"] = bridge._reservation_state(
            "auto_review", None, list(MANAGER_TOOLS)
        )
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            definitions=list(MANAGER_TOOLS),
            developer_instructions="You are the Root Manager.",
        )
        options = recorder[-1]
        server = options["mcp_servers"]["vnext"]
        self.assertEqual(
            ["delegate", "complete_session"],
            [tool.name for tool in server.tools],
        )
        # A raw JSON-Schema dict must reach the SDK untouched.
        self.assertEqual("object", server.tools[0].schema["type"])
        self.assertEqual(
            {"type": "string", "description": "Internal vNext native-child context. Supplied by Claude hook only."},
            server.tools[0].schema["properties"]["_vnext_native_child_context"],
        )
        self.assertEqual([], options["disallowed_tools"])
        self.assertIsNone(options["tools"])
        self.assertEqual(["user", "project", "local"], options["setting_sources"])
        self.assertFalse(options["strict_mcp_config"])
        # A worker that loaded the vNext plugin started a server of its own.
        # The public export moves the marketplace to the repository root.
        root = Path(__file__).resolve().parents[1]
        found = [
            path
            for path in (
                root / "plugins/.claude-plugin/marketplace.json",
                root / ".claude-plugin/marketplace.json",
            )
            if path.exists()
        ]
        marketplace = json.loads(found[0].read_text())
        plugin_id = f"{marketplace['plugins'][0]['name']}@{marketplace['name']}"
        self.assertEqual(
            {"enabledPlugins": {plugin_id: False}},
            json.loads(options["settings"]),
        )
        self.assertEqual("all", options["skills"])
        self.assertTrue(options["include_partial_messages"])
        self.assertTrue(options["include_hook_events"])
        self.assertTrue(options["forward_subagent_text"])
        self.assertIn("PreToolUse", options["hooks"])
        self.assertEqual(
            {"type": "preset", "preset": "claude_code", "append": "You are the Root Manager."},
            options["system_prompt"],
        )

    def test_accept_edits_is_explicit_and_does_not_open_bypass_mode(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            permission_mode="acceptEdits",
        )
        self.assertEqual("acceptEdits", recorder[-1]["permission_mode"])
        with self.assertRaisesRegex(BridgeError, "unsupported Claude permission mode"):
            _requested_permission_mode("dontAsk")

    def test_a_worker_with_no_tools_still_registers_the_post_tool_clock_hook(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        # A worker with a turn under way: the reservation holds both halves of
        # the budget, so the hook can report the full line.
        bridge._reservations["reservation-a"] = {
            "turn_started_monotonic": time.monotonic() - 123.0,
            "turn_timeout": 1800,
        }

        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a")

        hooks = recorder[-1]["hooks"]
        self.assertEqual(["PostToolUse"], list(hooks))
        matcher = hooks["PostToolUse"][0]
        self.assertIsNone(matcher.matcher)
        payload = asyncio.run(matcher.hooks[0]({}, None, None))
        output = payload["hookSpecificOutput"]
        self.assertEqual("PostToolUse", output["hookEventName"])
        self.assertRegex(output["additionalContext"], r"^\[clock\] \d\d:\d\d .+ · turn 2m 0\ds / 1800s$")

    def test_the_clock_hook_degrades_when_a_resume_cleared_the_turn_budget(self) -> None:
        # This is the state _resume leaves behind: the whole reservation dict
        # is rebuilt and turn_started_monotonic is None until the next turn.
        recorder: list = []
        bridge = self._bridge(recorder)

        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a")

        matcher = recorder[-1]["hooks"]["PostToolUse"][0]
        payload = asyncio.run(matcher.hooks[0]({}, None, None))
        self.assertRegex(
            payload["hookSpecificOutput"]["additionalContext"],
            r"^\[clock\] \d\d:\d\d [^·]+$",
        )

    def test_worker_session_hosts_no_server_and_no_system_prompt(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a")
        options = recorder[-1]
        self.assertEqual({}, options["mcp_servers"])
        self.assertNotIn("system_prompt", options)

    def test_hosted_tool_handler_blocks_until_the_control_plane_answers(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"] = "turn-a"
        state["generation"] = 1
        bridge._reservations["reservation-a"] = state
        handler = bridge._tool_handler("reservation-a", "delegate")
        written: list = []

        async def drive() -> dict:
            task = asyncio.create_task(handler({"role": "worker", "objective": "ship it"}))
            for _ in range(100):
                if written:
                    break
                await asyncio.sleep(0)
            request = written[0]["event"]
            bridge._resolve_tool_call({
                "v": 1,
                "kind": "control",
                "op": "tool_call_response",
                "reservation_id": "reservation-a",
                "turn_reference": "turn-a",
                "call_id": request["call_id"],
                "result": {"success": True, "value": {"agent_id": "agent-3"}},
            })
            return await task

        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            result = asyncio.run(drive())

        self.assertEqual("tool_call", written[0]["event"]["name"])
        self.assertEqual("delegate", written[0]["event"]["tool"])
        self.assertEqual({"role": "worker", "objective": "ship it"}, written[0]["event"]["arguments"])
        # A tool call carries no cursor: it is not an effect, so it must not
        # advance the effect stream the receipts are built from.
        self.assertNotIn("cursor", written[0]["event"])
        self.assertEqual(0, bridge._event_cursor)
        self.assertEqual(
            {"content": [{"type": "text", "text": '{"agent_id":"agent-3"}'}], "is_error": False},
            result,
        )
        self.assertEqual({}, state["pending_tool_calls"])

    def test_unsuccessful_tool_answer_reaches_the_model_as_an_error_result(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"] = "turn-a"
        bridge._reservations["reservation-a"] = state
        handler = bridge._tool_handler("reservation-a", "delegate")
        written: list = []

        async def drive() -> dict:
            task = asyncio.create_task(handler({"role": "worker"}))
            for _ in range(100):
                if written:
                    break
                await asyncio.sleep(0)
            bridge._resolve_tool_call({
                "v": 1,
                "kind": "control",
                "op": "tool_call_response",
                "reservation_id": "reservation-a",
                "turn_reference": "turn-a",
                "call_id": written[0]["event"]["call_id"],
                "result": {"success": False, "value": {"error_code": "not-direct-child"}},
            })
            return await task

        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            result = asyncio.run(drive())
        self.assertTrue(result["is_error"])
        self.assertEqual('{"error_code":"not-direct-child"}', result["content"][0]["text"])

    def test_malformed_or_uncorrelated_tool_answers_are_dropped(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"] = "turn-a"
        bridge._reservations["reservation-a"] = state
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            state["pending_tool_calls"]["call-0-0"] = future
            base = {
                "v": 1,
                "kind": "control",
                "op": "tool_call_response",
                "reservation_id": "reservation-a",
                "turn_reference": "turn-a",
                "call_id": "call-0-0",
                "result": {"success": True, "value": {}},
            }
            for mutation in (
                {"turn_reference": "other-turn"},
                {"reservation_id": "reservation-b"},
                {"call_id": "call-9-9"},
                {"result": {"success": "yes", "value": {}}},
                {"result": {"success": True}},
                {"v": 2},
            ):
                bridge._resolve_tool_call({**base, **mutation})
                self.assertFalse(future.done(), mutation)
            bridge._resolve_tool_call(base)
            self.assertTrue(future.done())
        finally:
            loop.close()

    def test_manager_tool_permission_is_allowed_without_reaching_the_reviewer(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        bridge._reservations["reservation-a"] = bridge._reservation_state(
            "auto_review", None, list(MANAGER_TOOLS)
        )
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            definitions=list(MANAGER_TOOLS),
        )
        relay = recorder[-1]["can_use_tool"]
        context = SimpleNamespace(tool_use_id="toolu_1")
        written: list = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            allowed = asyncio.run(relay("mcp__vnext__delegate", {}, context))
            denied = asyncio.run(relay("mcp__vnext__not_hosted", {}, context))
        self.assertTrue(allowed.allow)
        self.assertFalse(denied.allow)
        # No approval event was published for either: a control-plane tool call
        # is not an effect the reviewer decides on.
        self.assertEqual([], written)

    def test_a_worker_that_may_launch_a_native_subagent_may_also_read_its_result(self) -> None:
        """R19 claude-code F1: launching is allowed, so collecting must be too.

        With native children on, a backgrounded subagent answers its parent
        with a task id and the parent reads the result with TaskOutput.  A
        refusal there left the subagent running and billed with nobody able to
        read what it found, under a reason that called it untracked.  Continuing
        a subagent and building a team stay refused, and their reason says why.
        """

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state.update(turn_reference="turn-a", generation=1)
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            definitions=list(MANAGER_TOOLS),
        )
        self.assertTrue(state["native_child_registration"]["options_native_children_enabled"])
        relay = recorder[-1]["can_use_tool"]
        written: list = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            answers = {
                tool: asyncio.run(relay(tool, {"task_id": "x"}, SimpleNamespace(tool_use_id=f"{tool}-1")))
                for tool in ("Agent", "TaskOutput", "SendMessage", "TeamCreate")
            }
        self.assertTrue(answers["Agent"].allow)
        self.assertTrue(answers["TaskOutput"].allow)
        for tool in ("SendMessage", "TeamCreate"):
            with self.subTest(tool=tool):
                self.assertFalse(answers[tool].allow)
                self.assertNotIn("native subagents are not tracked", answers[tool].message)
                self.assertIn("launch a new subagent with the Agent tool", answers[tool].message)
        blocked = [
            value["event"]["tool"] for value in written
            if value.get("kind") == "event" and value["event"].get("name") == "native_child"
        ]
        self.assertEqual(["SendMessage", "TeamCreate"], blocked)

    def test_a_websearch_reaches_the_reviewer_and_a_todo_needs_no_review(self) -> None:
        """F17 at the callback: what the reviewer is actually asked about.

        WebSearch now publishes a "network" effect with its query, so the
        reviewer has an envelope it can decide.  TodoWrite changes nothing
        outside the turn, so it is allowed here rather than costing the
        manager an approval per item.
        """

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", [])
        state.update(turn_reference="turn-web", generation=1)
        bridge._reservations["reservation-web"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-web"
        )
        relay = recorder[-1]["can_use_tool"]
        written: list[dict] = []

        async def exercise() -> tuple[object, object]:
            state["task"] = asyncio.current_task()
            with patch(
                "vnext.vnext_claude_bridge._write", side_effect=written.append
            ):
                search = asyncio.create_task(
                    relay(
                        "WebSearch",
                        {"query": "vnext approval effect"},
                        SimpleNamespace(tool_use_id="web-call"),
                    )
                )
                await asyncio.sleep(0)
                pending = state["pending_permissions"]["web-call"]
                pending.set_result("accept")
                # A todo sent to the reviewer would wait here for an answer
                # nobody owes it, so the timeout is what keeps this test a
                # failure rather than a hang.
                todo = await asyncio.wait_for(
                    relay(
                        "TodoWrite",
                        {"todos": []},
                        SimpleNamespace(tool_use_id="todo-call"),
                    ),
                    timeout=5,
                )
                return await asyncio.wait_for(search, timeout=5), todo

        search_decision, todo_decision = asyncio.run(exercise())

        self.assertTrue(search_decision.allow)
        self.assertTrue(todo_decision.allow)
        # One envelope, for the search alone.
        self.assertEqual(1, len(written))
        event = written[0]["event"]
        self.assertEqual("permission", event["name"])
        self.assertEqual("network", event["effect"])
        self.assertEqual("WebSearch", event["tool"])
        self.assertEqual("vnext approval effect", event["path"])
        diagnostic = bridge._diagnostics()["permissions"]
        self.assertEqual(1, diagnostic["review_requested"])
        self.assertEqual(1, diagnostic["review_accepted"])
        self.assertEqual(1, diagnostic["allowed_no_effect"])

    def test_permission_diagnostics_identify_local_outcomes_without_payloads(self) -> None:
        recorder = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "private-session", list(MANAGER_TOOLS))
        state.update(turn_reference="private-turn", generation=1)
        bridge._reservations["private-reservation"] = state
        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None,
            reservation_id="private-reservation", definitions=list(MANAGER_TOOLS))
        relay = recorder[-1]["can_use_tool"]
        async def exercise():
            responses = []
            for name, call_id, agent in (
                ("mcp__vnext__delegate", None, None),
                ("mcp__vnext__delegate", "private-call", None),
                ("mcp__vnext__delegate", "private-child-call", "private-agent"),
                ("mcp__vnext__private-unhosted", "private-denied", None),
                ("ToolSearch", "private-search", None),
            ):
                responses.append(await relay(name, {"secret": "private-argument"},
                    SimpleNamespace(tool_use_id=call_id, agent_id=agent)))
            return responses
        self.assertEqual([False, True, True, False, True], [x.allow for x in asyncio.run(exercise())])
        diagnostic = bridge._diagnostics()["permissions"]
        self.assertEqual(5, diagnostic["callback_count"])
        self.assertEqual(2, diagnostic["hosted_callback_without_agent_id"])
        self.assertEqual(1, diagnostic["hosted_callback_with_agent_id"])
        self.assertEqual(1, diagnostic["denied_missing_tool_use_id"])
        self.assertEqual(2, diagnostic["allowed_hosted"])
        self.assertEqual(1, diagnostic["denied_unhosted"])
        self.assertEqual(1, diagnostic["allowed_discovery"])
        self.assertEqual(0, diagnostic["review_requested"])
        self.assertNotIn("private", json.dumps(diagnostic))
        state["permission_diagnostics"]["private-unexpected-key"] = 999
        self.assertNotIn("private", json.dumps(bridge._diagnostics()["permissions"]))

    def test_tool_search_discovers_hosted_controls_without_effect_approval(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state.update(turn_reference="turn-a", generation=1)
        bridge._reservations["reservation-a"] = state
        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None,
            reservation_id="reservation-a", definitions=list(MANAGER_TOOLS))
        relay = recorder[-1]["can_use_tool"]
        written = []
        def deny_unknown_effect(record):
            written.append(record)
            event = record.get("event", {})
            if event.get("name") == "permission":
                # This is the existing reviewer's decision for an unnamed effect.
                state["pending_permissions"][event["tool_use_id"]].set_result("decline")
        async def exercise():
            state["task"] = asyncio.current_task()
            return await relay("ToolSearch", {"query": "select:mcp__vnext__inspect"},
                SimpleNamespace(tool_use_id="search-1"))
        with patch("vnext.vnext_claude_bridge._write", side_effect=deny_unknown_effect):
            result = asyncio.run(exercise())
        self.assertTrue(result.allow, "discovering registered control tools must not abort the primary")
        self.assertEqual([], written, "tool discovery is not a workspace effect")

    def test_tool_search_permission_does_not_authorize_effects_or_unhosted_tools(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state.update(turn_reference="turn-a", generation=1)
        bridge._reservations["reservation-a"] = state
        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None,
            reservation_id="reservation-a", definitions=list(MANAGER_TOOLS))
        relay = recorder[-1]["can_use_tool"]
        written = []
        def deny(record):
            written.append(record)
            event = record.get("event", {})
            if event.get("name") == "permission":
                state["pending_permissions"][event["tool_use_id"]].set_result("decline")
        async def exercise():
            state["task"] = asyncio.current_task()
            results = []
            for index, name in enumerate(("ToolSearch", "mcp__vnext__not_hosted", "Bash", "Write", "ToolSearchOther")):
                results.append(await relay(name, {}, SimpleNamespace(tool_use_id=str(index))))
            results.append(await relay("ToolSearch", {}, SimpleNamespace(tool_use_id=None)))
            bridge._options(model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=[])
            results.append(await recorder[-1]["can_use_tool"]("ToolSearch", {}, SimpleNamespace(tool_use_id="empty")))
            return results
        with patch("vnext.vnext_claude_bridge._write", side_effect=deny):
            results = asyncio.run(exercise())
        self.assertEqual([True, False, False, False, False, False, False], [r.allow for r in results])
        permissions = [r["event"] for r in written if r.get("event", {}).get("name") == "permission"]
        self.assertEqual(["execute", "modify", None, None], [e.get("effect") for e in permissions])
        self.assertEqual([], recorder[0]["allowed_tools"])

    def test_manager_tool_call_is_never_projected_as_an_effect(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"] = "turn-a"
        class ToolUseBlock:
            def __init__(self) -> None:
                self.id, self.name, self.input = "toolu_1", "mcp__vnext__delegate", {}

        self.assertIsNone(bridge._effect_descriptor(state, ToolUseBlock()))

    def test_native_child_enrollment_scopes_task_lifecycle_and_coordination_tool(self) -> None:
        """A child can reach hosted tools only through its SDK hook identity."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"] = "turn-a"
        state["generation"] = 1
        state["native_child_legacy_enrollment"] = True
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL,
            resume=None,
            reservation_id="reservation-a",
            definitions=list(MANAGER_TOOLS),
        )
        options = recorder[-1]
        hook = options["hooks"]["PreToolUse"][0].hooks[0]
        server = options["mcp_servers"]["vnext"]
        register = next(item.handler for item in server.tools if item.name == "vnext_register_native_child")
        delegate = next(item.handler for item in server.tools if item.name == "delegate")
        session = "native-parent-session"

        parent = asyncio.run(hook({
            "tool_name": "Agent", "session_id": session, "tool_input": {"prompt": "child work"},
        }, "parent-tool", None))
        parent_prompt = parent["hookSpecificOutput"]["updatedInput"]["prompt"]
        self.assertIn("mcp__vnext__vnext_register_native_child", parent_prompt)
        token = re.search(r"enrollment_token=('[^']+')", parent_prompt).group(1)[1:-1]
        nested = asyncio.run(hook({
            "tool_name": "Agent", "session_id": session, "agent_id": "agent-1",
            "tool_input": {"prompt": "untracked nested child"},
        }, "nested-parent-tool", None))
        self.assertEqual("deny", nested["hookSpecificOutput"]["permissionDecision"])
        # The provider can publish TaskStarted before the child reaches MCP.
        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "task-1", "parent-tool", session

        class TaskNotificationMessage:
            task_id, session_id, status = "task-1", session, "completed"
            usage = {"total_tokens": 7, "tool_uses": 1, "duration_ms": 3}
            summary = "native child completed"

        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            register_hook = asyncio.run(hook({
                "tool_name": "mcp__vnext__vnext_register_native_child",
                "session_id": session,
                "agent_id": "agent-1",
                "tool_input": {"enrollment_token": token},
            }, "register-tool", None))
            registered = asyncio.run(register(register_hook["hookSpecificOutput"]["updatedInput"]))
            tool_hook = asyncio.run(hook({
                "tool_name": "mcp__vnext__delegate",
                "session_id": session,
                "agent_id": "agent-1",
                "tool_input": {"role": "worker", "objective": "do work"},
            }, "delegate-tool", None))

            async def call_child_tool() -> dict:
                pending = asyncio.create_task(delegate(tool_hook["hookSpecificOutput"]["updatedInput"]))
                for _ in range(100):
                    if any(item["event"]["name"] == "tool_call" for item in written):
                        break
                    await asyncio.sleep(0)
                call = next(item["event"] for item in written if item["event"]["name"] == "tool_call")
                bridge._resolve_tool_call({
                    "v": 1, "kind": "control", "op": "tool_call_response",
                    "reservation_id": "reservation-a", "turn_reference": "task-1",
                    "call_id": call["call_id"], "result": {"success": True, "value": {"ok": True}},
                })
                return await pending

            answer = asyncio.run(call_child_tool())
            # A child may report its lifecycle after a later parent turn has
            # begun. Its event must retain the originating Agent turn.
            state["turn_reference"], state["generation"] = "turn-b", 2
            bridge._emit_message("reservation-a", 1, TaskNotificationMessage())
            ConflictingTaskNotificationMessage = type(
                "TaskNotificationMessage", (),
                {"task_id": "task-1", "session_id": session, "status": "failed"},
            )

            with self.assertRaisesRegex(BridgeError, "terminal status conflicts"):
                bridge._emit_message("reservation-a", 1, ConflictingTaskNotificationMessage())

        events = [item["event"] for item in written]
        self.assertTrue(registered["is_error"] is False)
        self.assertFalse(answer["is_error"])
        running = next(item for item in events if item["name"] == "native_child" and item["tracking"] == "attested")
        self.assertEqual("task-1", running["task_id"])
        self.assertEqual("claude-native:native-parent-session:agent-1", running["native_runtime_thread_id"])
        call = next(item for item in events if item["name"] == "tool_call")
        self.assertEqual({"role": "worker", "objective": "do work"}, call["arguments"])
        self.assertEqual("claude-native:native-parent-session:agent-1", call["native_runtime_thread_id"])
        self.assertEqual("task-1", call["native_child_task_id"])
        self.assertEqual("task-1", call["turn_reference"])
        self.assertEqual("turn-a", call["parent_turn_reference"])
        usage = next(item for item in events if item["name"] == "native_child_usage")
        self.assertEqual("native_task", usage["usage_source"])
        completed = next(item for item in events if item["name"] == "native_child_completed")
        self.assertEqual("completed", completed["status"])
        self.assertEqual("native child completed", completed["summary"])
        self.assertEqual("turn-a", completed["parent_turn_reference"])
        self.assertEqual("turn-a", completed["provider_correlation"]["turn"])
        self.assertEqual(1, completed["generation"])
        self.assertEqual(1, sum(item["name"] == "native_child_completed" for item in events))
        enrollment = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(2, enrollment["agent_hook_seen"])
        self.assertEqual(1, enrollment["agent_rewrite_applied"])
        self.assertEqual(1, enrollment["registration_hook_seen"])
        self.assertEqual(1, enrollment["registration_proof_injected"])
        self.assertEqual(1, enrollment["registration_handler_called"])
        self.assertEqual(1, enrollment["registration_handler_accepted"])
        self.assertEqual(1, enrollment["task_started_seen"])
        self.assertEqual(0, enrollment["task_started_enrollment_found"])
        self.assertEqual(1, enrollment["task_started_joined"])

    def test_stopped_and_killed_keep_the_first_attested_terminal_status(self) -> None:
        """Equivalent native interruption facts must retain their raw first status."""

        for first, second in (("stopped", "killed"), ("killed", "stopped")):
            with self.subTest(first=first, second=second):
                recorder: list = []
                bridge = self._bridge(recorder)
                state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
                state["native_child_origins"][("native-session", "parent-tool")] = {
                    "turn_reference": "root-turn", "generation": 4,
                }
                identity = NativeChildIdentity(
                    "native-session", "parent-tool", "child-agent", "child-task"
                )
                written: list[dict] = []
                with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
                    bridge._emit_attested_native_child_lifecycle(
                        "reservation-a", 4, state, identity,
                        {"source": "TaskUpdatedMessage", "status": first},
                    )
                    bridge._emit_attested_native_child_lifecycle(
                        "reservation-a", 4, state, identity,
                        {"source": "TaskNotificationMessage", "status": second},
                    )
                    with self.assertRaisesRegex(BridgeError, "terminal status conflicts"):
                        bridge._emit_attested_native_child_lifecycle(
                            "reservation-a", 4, state, identity,
                            {"source": "TaskNotificationMessage", "status": "failed"},
                        )

                completed = [
                    item["event"] for item in written
                    if item["event"]["name"] == "native_child_completed"
                ]
                self.assertEqual(1, len(completed))
                self.assertEqual(first, completed[0]["status"])
                self.assertEqual(first, state["native_children"]["child-task"]["terminal_status"])

    def test_pending_native_terminal_keeps_first_status_and_agent_result_replays_it(self) -> None:
        """A foreground result cannot overwrite an already buffered raw terminal fact."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 5
        state["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        state["native_child_origins"][("native-session", "parent-tool")] = {
            "turn_reference": "root-turn", "generation": 5,
        }
        resolver = state["native_child_automatic_resolver"]
        resolver.record_agent_origin(
            "native-session", "parent-tool", None, parent_agent_id_present=True,
            tool_input={"model": CLAUDE_WORKER_MODEL},
        )
        resolver.record_subagent_start("native-session", "child-agent")
        resolver.record_metadata("native-session", "child-agent", "parent-tool", None)
        resolver.record_task_started("native-session", "parent-tool", "child-task")
        (identity,) = resolver.resolve_ready()
        key = ("native-session", "child-task")
        state["native_child_pending_lifecycle"][key] = {
            "terminal": {"source": "TaskUpdatedMessage", "status": "killed"},
        }
        written: list[dict] = []
        result = SimpleNamespace(
            parent_tool_use_id=None,
            content=[SimpleNamespace(tool_use_id="parent-tool")],
            tool_use_result={"status": "stopped", "agentId": "child-agent", "task_id": "child-task"},
        )
        with (
            patch("vnext.vnext_claude_bridge._write", side_effect=written.append),
            patch.object(bridge, "_refresh_automatic_native_metadata"),
        ):
            bridge._observe_native_agent_result(state, result, reservation_id="reservation-a")

        self.assertNotIn(key, state["native_child_pending_lifecycle"])
        completed = [
            item["event"] for item in written
            if item["event"]["name"] == "native_child_completed"
        ]
        self.assertEqual(1, len(completed))
        self.assertEqual("killed", completed[0]["status"])
        self.assertEqual("TaskUpdatedMessage", completed[0]["source"])
        self.assertEqual(identity.task_id, completed[0]["task_id"])

    def test_pending_terminal_history_preserves_first_raw_status(self) -> None:
        """The pre-identity lifecycle buffer has the same stopped/killed rule."""

        bridge = self._bridge([])
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        class TaskNotificationMessage:
            session_id, task_id = "native-session", "child-task"

            def __init__(self, status: str) -> None:
                self.status = status

        with patch("vnext.vnext_claude_bridge._write"):
            bridge._emit_native_child_event("reservation-a", 1, state, TaskNotificationMessage("stopped"))
            bridge._emit_native_child_event("reservation-a", 1, state, TaskNotificationMessage("killed"))
            with self.assertRaisesRegex(BridgeError, "terminal status conflicts"):
                bridge._emit_native_child_event("reservation-a", 1, state, TaskNotificationMessage("failed"))

        terminal = state["native_child_pending_lifecycle"][("native-session", "child-task")]["terminal"]
        self.assertEqual("stopped", terminal["status"])

    def test_saved_session_metadata_joins_a_no_tool_child_without_registration(self) -> None:
        """Automatic identity has no prompt or child-MCP enrollment dependency."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        session = "native-parent-session"
        parent_decision = asyncio.run(pretool({
            "tool_name": "Agent", "session_id": session,
            "tool_input": {"prompt": "child work", "description": "bounded child objective", "model": "sonnet"},
        }, "parent-tool", None))
        self.assertEqual({"continue": True}, parent_decision)
        self.assertNotIn(
            "vnext_register_native_child",
            [tool.name for tool in options["mcp_servers"]["vnext"].tools],
        )

        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session

        class TaskProgressMessage:
            task_id, session_id = "child-task", session
            usage = {"total_tokens": 1}

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": CLAUDE_WORKER_MODEL}

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()]
        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            asyncio.run(started({"session_id": session, "agent_id": "no-tool-child"}, None, None))
            for _ in range(65):
                bridge._emit_message("reservation-a", 1, TaskProgressMessage())

        events = [item["event"] for item in written]
        identity = next(item for item in events if item["name"] == "native_child_identity")
        running = next(item for item in events if item["name"] == "native_child" and item["tracking"] == "attested")
        self.assertEqual("saved-session-metadata", identity["identity_source"])
        self.assertEqual("before-terminal-observation", identity["identity_resolution"])
        self.assertIsNone(identity["parent_native_agent_id"])
        self.assertEqual("no-tool-child", running["native_agent_id"])
        self.assertEqual("child-task", running["task_id"])
        self.assertEqual({}, state["native_child_pending_lifecycle"])
        self.assertEqual("bounded child objective", running["task_contract"]["objective"])
        self.assertEqual("sonnet", running["task_contract"]["requested_model"])
        tool_hook = options["hooks"]["PreToolUse"][0].hooks[0]
        scoped = asyncio.run(tool_hook({
            "tool_name": "mcp__vnext__delegate", "session_id": session, "agent_id": "no-tool-child",
            "tool_input": {"role": "worker", "objective": "follow up"},
        }, "child-tool", None))
        self.assertIn("_vnext_native_child_context", scoped["hookSpecificOutput"]["updatedInput"])

    def test_late_saved_metadata_replays_terminal_no_tool_lifecycle(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        session = "native-parent-session"
        asyncio.run(pretool({
            "tool_name": "Agent", "session_id": session, "tool_input": {"prompt": "child work"},
        }, "parent-tool", None))
        metadata_available = False

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": CLAUDE_WORKER_MODEL}

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()] if metadata_available else []
        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session
        class TaskNotificationMessage:
            task_id, session_id, status, summary = "child-task", session, "completed", "done"
            usage = {"total_tokens": 3}

        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(started({"session_id": session, "agent_id": "no-tool-child"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            metadata_available = True
            bridge._emit_message("reservation-a", 1, TaskNotificationMessage())
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())

        events = [item["event"] for item in written]
        identity = next(item for item in events if item["name"] == "native_child_identity")
        completed = next(item for item in events if item["name"] == "native_child_completed")
        self.assertEqual("after-terminal-observation", identity["identity_resolution"])
        self.assertEqual("completed", completed["status"])
        self.assertEqual("done", completed["summary"])
        self.assertEqual(
            1,
            sum(item["name"] == "native_child" and item.get("tracking") == "attested" for item in events),
        )
        self.assertEqual({}, state["native_child_pending_lifecycle"])

    def test_forwarded_child_assistant_model_releases_exact_pending_task_before_terminal(self) -> None:
        """A typed SDK child message completes model evidence without a saved-file reread."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        session = "native-parent-session"

        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            # This early metadata establishes the exact agent-to-parent join,
            # but its alias must never become the effective model.
            message = {"role": "assistant", "model": "opus"}

        class AssistantMessage:
            session_id, parent_tool_use_id = session, "parent-tool"
            model, content, stop_reason = CLAUDE_WORKER_MODEL, [], None

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()]
        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(pretool({
                "tool_name": "Agent", "session_id": session,
                "tool_input": {"prompt": "child work", "description": "bounded child", "model": "opus"},
            }, "parent-tool", None))
            # The typed message can arrive before TaskStarted. It is only
            # retained until the independent hook, metadata and task joins
            # identify the child; it cannot itself emit an attestation.
            bridge._emit_message("reservation-a", 1, AssistantMessage())
            self.assertFalse(any(item["event"].get("tracking") == "attested" for item in written))
            asyncio.run(started({"session_id": session, "agent_id": "child-agent"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            running = next(item["event"] for item in written if item["event"].get("status") == "running")
            self.assertEqual("forwarded-child-assistant-message", running["task_contract"]["observed_model_source"])
            self.assertEqual(CLAUDE_WORKER_MODEL, running["task_contract"]["observed_model"])
            self.assertEqual({}, state["native_child_forwarded_models"])
            conflicting = type("AssistantMessage", (), {
                "session_id": session, "parent_tool_use_id": "parent-tool",
                "model": "claude-sonnet-4-5", "content": [], "stop_reason": None,
            })()
            with self.assertRaisesRegex(NativeChildIdentityError, "effective model conflicts"):
                bridge._emit_message("reservation-a", 1, conflicting)
            result = SimpleNamespace(
                parent_tool_use_id=None, content=[SimpleNamespace(tool_use_id="parent-tool")],
                tool_use_result={"status": "completed", "agentId": "child-agent", "task_id": "child-task"},
            )
            bridge._observe_native_agent_result(state, result, reservation_id="reservation-a")
        events = [item["event"] for item in written]
        self.assertLess(
            events.index(running),
            next(index for index, item in enumerate(events) if item["name"] == "native_child_completed"),
        )

    def test_forwarded_child_model_rejects_empty_foreign_alias_and_stale_inputs(self) -> None:
        bridge = self._bridge([])
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["native_child_origins"][("native-session", "parent-tool")] = {
            "turn_reference": "root-turn", "generation": 1,
        }
        state["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        bridge._record_forwarded_child_assistant_model(state, "native-session", None, CLAUDE_WORKER_MODEL)
        bridge._record_forwarded_child_assistant_model(state, "native-session", "other-tool", CLAUDE_WORKER_MODEL)
        bridge._record_forwarded_child_assistant_model(state, "native-session", "parent-tool", "opus")
        self.assertEqual({}, state["native_child_forwarded_models"])
        bridge._record_forwarded_child_assistant_model(state, "native-session", "parent-tool", CLAUDE_WORKER_MODEL)
        self.assertEqual(1, len(state["native_child_forwarded_models"]))
        state["native_child_forwarded_models"][("native-session", "parent-tool")]["model"] = "other-canonical"
        with self.assertRaisesRegex(NativeChildIdentityError, "effective model conflicts"):
            bridge._record_forwarded_child_assistant_model(state, "native-session", "parent-tool", CLAUDE_WORKER_MODEL)

        conflict = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        conflict["generation"] = 1
        conflict["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        conflict["native_child_origins"][("native-session", "parent-tool")] = {
            "turn_reference": "root-turn", "generation": 1,
        }
        bridge._reservations["reservation-conflict"] = conflict

        matching = type("AssistantMessage", (), {
            "session_id": "native-session", "parent_tool_use_id": "parent-tool",
            "model": CLAUDE_WORKER_MODEL, "content": [], "stop_reason": None,
        })()
        conflicting = type("AssistantMessage", (), {
            "session_id": "native-session", "parent_tool_use_id": "parent-tool",
            "model": "claude-sonnet-4-5", "content": [], "stop_reason": None,
        })()

        with patch("vnext.vnext_claude_bridge._write"):
            bridge._emit_message("reservation-conflict", 1, matching)
            with self.assertRaisesRegex(NativeChildIdentityError, "effective model conflicts"):
                bridge._emit_message("reservation-conflict", 1, conflicting)

        capacity = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        capacity["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        for tool in ("first-tool", "second-tool"):
            capacity["native_child_origins"][("native-session", tool)] = {
                "turn_reference": "root-turn", "generation": 1,
            }
        with patch("vnext.vnext_claude_bridge._MAX_PENDING_EFFECTS", 1):
            bridge._record_forwarded_child_assistant_model(
                capacity, "native-session", "first-tool", CLAUDE_WORKER_MODEL
            )
            with self.assertRaisesRegex(BridgeError, "forwarded child model capacity exceeded"):
                bridge._record_forwarded_child_assistant_model(
                    capacity, "native-session", "second-tool", CLAUDE_WORKER_MODEL
                )

        stale = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        stale["generation"] = 2
        stale["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        stale["native_child_origins"][("native-session", "parent-tool")] = {
            "turn_reference": "root-turn", "generation": 2,
        }
        bridge._reservations["reservation-stale"] = stale

        class AssistantMessage:
            session_id, parent_tool_use_id = "native-session", "parent-tool"
            model, content, stop_reason = CLAUDE_WORKER_MODEL, [], None

        with self.assertRaisesRegex(BridgeError, "stale local generation"):
            bridge._emit_message("reservation-stale", 1, AssistantMessage())
        self.assertEqual({}, stale["native_child_forwarded_models"])

    def test_canonical_saved_metadata_model_mismatch_is_sticky(self) -> None:
        bridge = self._bridge([])
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        resolver = state["native_child_automatic_resolver"]

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": "claude-sonnet-4-5"}

        with self.assertRaisesRegex(NativeChildIdentityError, "observed model conflicts with configured"):
            bridge._record_automatic_native_metadata(
                state, resolver, "native-session", "child-agent", [Metadata()]
            )
        self.assertEqual({}, state["native_child_observed_models"])

    def test_forwarded_child_models_are_drained_after_65_sequential_exact_joins(self) -> None:
        bridge = self._bridge([])
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["native_child_configured_model"] = CLAUDE_WORKER_MODEL
        for index in range(65):
            parent_tool, agent_id = f"parent-tool-{index}", f"child-agent-{index}"
            state["native_child_origins"][("native-session", parent_tool)] = {
                "turn_reference": "root-turn", "generation": 1,
            }
            bridge._record_forwarded_child_assistant_model(
                state, "native-session", parent_tool, CLAUDE_WORKER_MODEL
            )
            bridge._apply_forwarded_child_assistant_model(
                state,
                NativeChildIdentity(
                    parent_session_id="native-session", parent_tool_use_id=parent_tool,
                    agent_id=agent_id, task_id=f"child-task-{index}",
                    identity_source="saved_session_metadata",
                ),
            )
            self.assertEqual({}, state["native_child_forwarded_models"])
        self.assertEqual(65, len(state["native_child_observed_models"]))

    def test_child_tool_refreshes_late_metadata_before_scoped_proof(self) -> None:
        """An authenticated child hook can unlock its exact pending task."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        stopped = options["hooks"]["SubagentStop"][0].hooks[0]
        session = "native-parent-session"
        metadata_available = False

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": CLAUDE_WORKER_MODEL}

        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()] if metadata_available else []
        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(pretool({
                "tool_name": "Agent", "session_id": session,
                "tool_input": {"prompt": "child work", "description": "late metadata child"},
            }, "parent-tool", None))
            asyncio.run(started({"session_id": session, "agent_id": "child-agent"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            metadata_available = True
            scoped = asyncio.run(pretool({
                "tool_name": "mcp__vnext__delegate", "session_id": session,
                "agent_id": "child-agent", "tool_input": {"role": "worker", "objective": "follow up"},
            }, "child-tool", None))
            asyncio.run(stopped({
                "session_id": session, "agent_id": "child-agent", "stop_hook_active": False,
            }, None, None))

        self.assertIn("_vnext_native_child_context", scoped["hookSpecificOutput"]["updatedInput"])
        events = [item["event"] for item in written]
        identity = next(item for item in events if item["name"] == "native_child_identity")
        observed_stop = next(item for item in events if item["name"] == "native_child_stop")
        self.assertEqual("before-terminal-observation", identity["identity_resolution"])
        self.assertEqual("after-terminal-observation", observed_stop["identity_resolution"])
        self.assertEqual(1, sum(item["name"] == "native_child" and item.get("tracking") == "attested" for item in events))
        self.assertFalse(any(item["name"] == "native_child_completed" for item in events))
        diagnostic = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(1, diagnostic["subagent_start_hook_seen"])
        self.assertEqual(1, diagnostic["subagent_stop_hook_seen"])
        self.assertEqual(2, diagnostic["metadata_read_empty"])
        self.assertEqual(2, diagnostic["metadata_read_matched"])
        self.assertEqual(1, diagnostic["automatic_identity_joined"])

    def test_unresolved_child_model_denies_coordination_before_tool_dispatch(self) -> None:
        """A joined identity without model evidence cannot reach the adapter."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        session = "native-parent-session"

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None

        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()]
        written: list[dict] = []
        with (
            patch("vnext.vnext_claude_bridge._write", side_effect=written.append),
            patch("vnext.vnext_claude_bridge._NATIVE_CHILD_COORDINATION_METADATA_ATTEMPTS", 1),
        ):
            asyncio.run(pretool({
                "tool_name": "Agent", "session_id": session,
                "tool_input": {"prompt": "child work", "description": "bounded child"},
            }, "parent-tool", None))
            asyncio.run(started({"session_id": session, "agent_id": "child-agent"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            denied = asyncio.run(pretool({
                "tool_name": "mcp__vnext__delegate", "session_id": session,
                "agent_id": "child-agent", "tool_input": {"role": "worker", "objective": "follow up"},
            }, "child-tool", None))

        self.assertEqual("deny", denied["hookSpecificOutput"]["permissionDecision"])
        self.assertFalse(any(item["event"]["name"] == "tool_call" for item in written))
        self.assertFalse(any(item["event"]["name"] == "native_child_identity" for item in written))
        self.assertIn((session, "child-task"), state["native_child_pending_lifecycle"])
        diagnostic = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(1, diagnostic["coordination_metadata_unresolved"])
        self.assertEqual(0, diagnostic["automatic_identity_joined"])

    def test_child_tool_retries_canonical_model_before_scoped_proof(self) -> None:
        """A later exact model frame can release an already joined child once."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        session = "native-parent-session"
        release_model = False
        release_reads = 0

        class MetadataWithoutModel:
            parent_tool_use_id, parent_agent_id = "parent-tool", None

        class MetadataWithModel:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": CLAUDE_WORKER_MODEL}

        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session

        def read_metadata(*_args, **_kwargs):
            nonlocal release_reads
            if not release_model:
                return [MetadataWithoutModel()]
            release_reads += 1
            return [MetadataWithoutModel()] if release_reads == 1 else [MetadataWithModel()]

        bridge._sdk.get_subagent_messages = read_metadata
        written: list[dict] = []
        with (
            patch("vnext.vnext_claude_bridge._write", side_effect=written.append),
            patch("vnext.vnext_claude_bridge._NATIVE_CHILD_COORDINATION_METADATA_ATTEMPTS", 2),
            patch("vnext.vnext_claude_bridge._NATIVE_CHILD_COORDINATION_METADATA_DELAY_SECONDS", 0),
        ):
            asyncio.run(pretool({
                "tool_name": "Agent", "session_id": session,
                "tool_input": {"prompt": "child work", "description": "bounded child"},
            }, "parent-tool", None))
            asyncio.run(started({"session_id": session, "agent_id": "child-agent"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            release_model = True
            scoped = asyncio.run(pretool({
                "tool_name": "mcp__vnext__delegate", "session_id": session,
                "agent_id": "child-agent", "tool_input": {"role": "worker", "objective": "follow up"},
            }, "child-tool", None))

        self.assertIn("_vnext_native_child_context", scoped["hookSpecificOutput"]["updatedInput"])
        events = [item["event"] for item in written]
        identity_at = next(index for index, event in enumerate(events) if event["name"] == "native_child_identity")
        running_at = next(index for index, event in enumerate(events) if event["name"] == "native_child" and event["status"] == "running")
        self.assertLess(identity_at, running_at)
        self.assertEqual(CLAUDE_WORKER_MODEL, events[running_at]["task_contract"]["observed_model"])
        diagnostic = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(1, diagnostic["coordination_metadata_retry"])
        self.assertEqual(0, diagnostic["coordination_metadata_unresolved"])

    def test_stop_hook_recovers_late_metadata_without_completion(self) -> None:
        """SubagentStop proves only an observed stop, never task completion."""

        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a", definitions=list(MANAGER_TOOLS)
        )
        options = recorder[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        stopped = options["hooks"]["SubagentStop"][0].hooks[0]
        session = "native-parent-session"
        metadata_available = False

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": CLAUDE_WORKER_MODEL}

        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", session

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()] if metadata_available else []
        written: list[dict] = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(pretool({
                "tool_name": "Agent", "session_id": session, "tool_input": {"prompt": "child work"},
            }, "parent-tool", None))
            asyncio.run(started({"session_id": session, "agent_id": "child-agent"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            metadata_available = True
            asyncio.run(stopped({
                "session_id": session, "agent_id": "child-agent", "stop_hook_active": False,
            }, None, None))

        events = [item["event"] for item in written]
        identity = next(item for item in events if item["name"] == "native_child_identity")
        observed_stop = next(item for item in events if item["name"] == "native_child_stop")
        self.assertEqual("after-terminal-observation", identity["identity_resolution"])
        self.assertEqual("observed-stop", observed_stop["status"])
        self.assertTrue(observed_stop["stop_observed"])
        self.assertFalse(any(item["name"] == "native_child_completed" for item in events))
        diagnostic = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(1, diagnostic["subagent_stop_hook_seen"])
        self.assertEqual(2, diagnostic["metadata_read_empty"])
        self.assertEqual(1, diagnostic["metadata_read_matched"])
        self.assertEqual(1, diagnostic["automatic_identity_joined"])

    def test_live_agent_result_diagnostics_keep_only_allowed_shape(self) -> None:
        bridge = self._bridge([])
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["session_id"] = "native-session"
        state["native_child_origins"][("native-session", "parent-tool")] = {
            "turn_reference": "turn", "generation": 1,
        }
        bridge._reservations["reservation-a"] = state
        bridge._observe_native_agent_result(
            state,
            SimpleNamespace(
                parent_tool_use_id=None,
                content=[SimpleNamespace(tool_use_id="parent-tool")],
                tool_use_result={"status": "completed", "agentId": "agent", "task_id": "task", "opaque": "discarded"},
            ),
        )
        diagnostic = bridge._diagnostics()["native_child_registration"]
        self.assertEqual(1, diagnostic["agent_result_seen"])
        self.assertEqual(1, diagnostic["agent_result_has_status"])
        self.assertEqual(1, diagnostic["agent_result_has_agent_id"])
        self.assertEqual(1, diagnostic["agent_result_has_task_id"])
        self.assertEqual(1, diagnostic["agent_result_status_completed"])

    def test_foreground_alias_waits_for_observed_model_and_exact_result(self) -> None:
        for status in ("completed", "failed", "stopped"):
            with self.subTest(status=status):
                self._check_foreground_exact_result(status)

    def _check_foreground_exact_result(self, status: str) -> None:
        recorder = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(model=CLAUDE_WORKER_MODEL, resume=None, reservation_id="reservation-a",
                        definitions=list(MANAGER_TOOLS))
        hooks = recorder[-1]["hooks"]
        metadata = SimpleNamespace(parent_tool_use_id="parent-tool", parent_agent_id=None,
                                   message={"role": "user"})
        messages = [metadata]
        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: messages
        class TaskStartedMessage:
            task_id, tool_use_id, session_id = "child-task", "parent-tool", "native-session"
        written = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(hooks["PreToolUse"][0].hooks[0]({
                "tool_name": "Agent", "session_id": "native-session",
                "tool_input": {"model": "opus", "description": "bounded child"},
            }, "parent-tool", None))
            asyncio.run(hooks["SubagentStart"][0].hooks[0]({
                "session_id": "native-session", "agent_id": "child-agent"}, None, None))
            bridge._emit_message("reservation-a", 1, TaskStartedMessage())
            self.assertFalse(any(item["event"].get("tracking") == "attested" for item in written))
            result = SimpleNamespace(parent_tool_use_id=None,
                content=[SimpleNamespace(tool_use_id="parent-tool")],
                tool_use_result={"status": status, "agentId": "different-child"})
            bridge._observe_native_agent_result(state, result, reservation_id="reservation-a")
            self.assertFalse(any(item["event"].get("name") == "native_child_completed" for item in written))
            result.tool_use_result["agentId"] = "child-agent"
            result.content.append(SimpleNamespace(tool_use_id="other-result"))
            bridge._observe_native_agent_result(state, result, reservation_id="reservation-a")
            self.assertNotIn("terminal", state["native_child_pending_lifecycle"][("native-session", "child-task")])
            result.content.pop()
            bridge._observe_native_agent_result(state, result, reservation_id="reservation-a")
            self.assertFalse(any(item["event"].get("name") == "native_child_completed" for item in written))
            messages.append(SimpleNamespace(parent_tool_use_id="parent-tool", parent_agent_id=None,
                                             message={"role": "assistant", "model": CLAUDE_WORKER_MODEL}))
            class AssistantMessage:
                session_id, model, content = "native-session", CLAUDE_WORKER_MODEL, []
            bridge._emit_message("reservation-a", 1, AssistantMessage())
            bridge._observe_native_agent_result(state, result, reservation_id="reservation-a")
        events = [item["event"] for item in written]
        running = next(item for item in events if item.get("status") == "running")
        self.assertEqual("opus", running["task_contract"]["requested_model"])
        self.assertEqual(CLAUDE_WORKER_MODEL, running["task_contract"]["observed_model"])
        self.assertEqual("after-terminal-observation", running["identity_resolution"])
        completed = [item for item in events if item["name"] == "native_child_completed"]
        self.assertEqual(1, len(completed))
        self.assertEqual("AgentToolResult", completed[0]["source"])
        self.assertEqual(status, completed[0]["status"])

    def test_forged_child_context_never_reaches_the_control_plane(self) -> None:
        recorder: list = []
        bridge = self._bridge(recorder)
        state = bridge._reservation_state("auto_review", "native-session", list(MANAGER_TOOLS))
        state["turn_reference"] = "turn-a"
        bridge._reservations["reservation-a"] = state
        handler = bridge._tool_handler("reservation-a", "delegate")
        with self.assertRaisesRegex(BridgeError, "not exactly attested"):
            asyncio.run(handler({"role": "worker", "_vnext_native_child_context": "forged"}))


class ClaudeWorkspaceContainmentTests(unittest.TestCase):
    """The containment check is the safety boundary, so it gets its own tests.

    `_workspace_inside` decides whether a requested root belongs to this
    adapter.  Every private Worker root passes through it, and nothing else
    stops a path outside the connected workspace from being served.  It had no
    coverage at all, which meant a future edit could replace it with a string
    prefix compare and no test would notice.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = (Path(self.temp.name) / "b").resolve()
        self.workspace.mkdir()
        (self.workspace / "inside").mkdir()
        self.sibling = (Path(self.temp.name) / "bc").resolve()
        self.sibling.mkdir()
        self.adapter = ClaudeCodeAdapter(workspace=str(self.workspace))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_the_owned_workspace_and_anything_under_it_is_accepted(self) -> None:
        self.assertEqual(
            self.workspace, self.adapter._workspace_inside(str(self.workspace))
        )
        self.assertEqual(
            self.workspace / "inside",
            self.adapter._workspace_inside(str(self.workspace / "inside")),
        )
        # A root that does not exist yet is still ours; the bridge is what
        # refuses to run in a directory nobody created.
        self.assertEqual(
            self.workspace / "not-yet",
            self.adapter._workspace_inside(str(self.workspace / "not-yet")),
        )
        self.assertEqual(self.workspace, self.adapter._workspace_inside(None))

    def test_a_sibling_sharing_a_string_prefix_is_refused(self) -> None:
        """`/a/bc` is not inside `/a/b`, however much the strings agree."""

        self.assertIsNone(self.adapter._workspace_inside(str(self.sibling)))

    def test_a_path_escaping_through_a_parent_segment_is_refused(self) -> None:
        escape = str(self.workspace / ".." / "bc")

        self.assertIsNone(self.adapter._workspace_inside(escape))

    def test_a_relative_path_is_refused(self) -> None:
        """Relative resolves against the process directory, which is not ours."""

        self.assertIsNone(self.adapter._workspace_inside("inside"))
        self.assertIsNone(self.adapter._workspace_inside(""))

    def test_a_path_outside_the_workspace_is_refused(self) -> None:
        self.assertIsNone(self.adapter._workspace_inside(str(Path(self.temp.name))))

    def test_a_thread_outside_the_owned_workspace_is_refused_by_name(self) -> None:
        posture = RuntimePosture(
            workspace_writes=True,
            network="restricted",
            approvals_requested=True,
            reviewer="user",
            environment_ready=True,
        )
        self.adapter._initialized = True

        with self.assertRaises(ClaudeRuntimeError) as caught:
            self.adapter.start_thread(
                model="sonnet",
                developer_instructions="ROLE=worker fixture",
                tools=(),
                tool_handler=None,
                requested_posture=posture,
                workspace=str(self.sibling),
            )

        self.assertIn("workspace", str(caught.exception))


class ClaudeBridgePendingStreamDeltaTests(unittest.TestCase):
    """A held stream is display text.  A held message is a protocol fact."""

    def setUp(self) -> None:
        self.bridge = _Bridge()
        state = self.bridge._reservation_state("auto_review", None, [])
        state.update({"generation": 1, "turn_reference": "turn-a"})
        self.bridge._reservations["reservation-a"] = state
        self.state = state

    def _delta(self, index: int) -> dict:
        return {
            "name": "stream",
            "reservation_id": "reservation-a",
            "turn_reference": "turn-a",
            "generation": 1,
            "content": {"type": "thinking_delta", "thinking": "step-" + str(index)},
        }

    def test_query_survives_96_system_messages_before_identity_binding(self) -> None:
        """System status bursts must not consume the whole-message budget."""
        class SystemMessage:
            subtype = "status"
            data = {"session_id": "native-a", "status": "busy", "stdout": "secret"}

        class StreamEvent:
            event = {"delta": {"type": "text_delta", "text": "OK"}}

        class ResultMessage:
            session_id, result = "native-a", "OK"
            usage = total_cost_usd = model_usage = None
            duration_ms = duration_api_ms = num_turns = None
            is_error, stop_reason, subtype = False, None, "success"
            api_error_status, terminal_reason, errors = None, None, ()

        class Client:
            async def query(self, prompt):
                pass

            async def receive_messages(self):
                for _ in range(96):
                    yield SystemMessage()
                yield StreamEvent()
                yield ResultMessage()

        self.bridge._sdk, self.bridge._workspace = object(), Path.cwd()
        self.state["client"] = Client()
        written = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(self.bridge._run_query("reservation-a", 1, "fixture"))
        self.assertEqual("native-a", self.state["session_id"])
        events = [row["event"] for row in written]
        systems = [event for event in events if event["name"] == "system_message"]
        self.assertEqual(96, len(systems))
        self.assertTrue(all(event["correlation_attested"] for event in systems))
        self.assertTrue(all(event["data"] == {"session_id": "native-a"} for event in systems))
        self.assertEqual(1, sum(event["name"] == "stream" for event in events))
        self.assertEqual([], self.state["pending_messages"])

    def test_a_thinking_flood_before_binding_keeps_the_reservation_alive(self) -> None:
        """This is the GLM-5.3 failure: high effort, no session id yet."""

        for index in range(_MAX_PENDING_EFFECTS * 4):
            self.bridge._release_or_hold_message(self.state, self._delta(index))

        pending = self.state["pending_messages"]
        self.assertEqual(_MAX_PENDING_EFFECTS, len(pending))
        self.assertEqual(_MAX_PENDING_EFFECTS * 3, self.state["dropped_stream_deltas"])
        # The oldest delta went and the newest stayed.
        self.assertEqual("step-192", pending[0]["content"]["thinking"])
        self.assertEqual("step-255", pending[-1]["content"]["thinking"])

    def test_a_full_buffer_of_whole_messages_still_fails_hard(self) -> None:
        for index in range(_MAX_PENDING_EFFECTS):
            self.bridge._release_or_hold_message(
                self.state,
                {"name": "message", "generation": 1, "message": {"sequence": index}},
            )

        with self.assertRaises(BridgeError) as caught:
            self.bridge._release_or_hold_message(
                self.state, {"name": "message", "generation": 1, "message": {}},
            )

        self.assertIn("capacity exceeded", str(caught.exception))
        self.assertEqual(0, self.state["dropped_stream_deltas"])

    def test_the_capacity_error_names_what_filled_the_buffer(self) -> None:
        """A live failure has to say which kind of event flooded the hold."""

        for index in range(_MAX_PENDING_EFFECTS - 2):
            self.bridge._release_or_hold_message(
                self.state,
                {"name": "message", "generation": 1, "message": {"sequence": index}},
            )
        for index in range(2):
            self.bridge._release_or_hold_message(
                self.state,
                {"name": "child_lifecycle", "generation": 1, "sequence": index},
            )

        with self.assertRaises(BridgeError) as caught:
            self.bridge._release_or_hold_message(
                self.state, {"name": "message", "generation": 1, "message": {}},
            )

        detail = str(caught.exception)
        self.assertIn("arriving message", detail)
        self.assertIn("2 child_lifecycle", detail)
        self.assertIn(f"{_MAX_PENDING_EFFECTS - 2} message", detail)

    def test_a_delta_arriving_on_a_buffer_of_messages_fails_hard(self) -> None:
        """Nothing here is display text, so there is nothing safe to drop."""

        for index in range(_MAX_PENDING_EFFECTS):
            self.bridge._release_or_hold_message(
                self.state,
                {"name": "message", "generation": 1, "message": {"sequence": index}},
            )

        with self.assertRaises(BridgeError):
            self.bridge._release_or_hold_message(self.state, self._delta(0))

        self.assertEqual(_MAX_PENDING_EFFECTS, len(self.state["pending_messages"]))

    def test_system_messages_do_not_displace_stream_deltas(self) -> None:
        """System and transcript events retain their independent budgets."""

        for index in range(30):
            self.bridge._release_or_hold_message(self.state, self._delta(index))
        for index in range(_MAX_PENDING_EFFECTS - 30):
            self.bridge._release_or_hold_message(
                self.state,
                {"name": "system_message", "generation": 1, "subtype": "hook_started", "data": {"n": index}},
            )

        arriving = {"name": "system_message", "generation": 1, "subtype": "hook_response", "data": {}}
        self.bridge._release_or_hold_message(self.state, arriving)

        pending = self.state["pending_messages"]
        self.assertEqual(_MAX_PENDING_EFFECTS + 1, len(pending))
        self.assertEqual(0, self.state["dropped_stream_deltas"])
        self.assertEqual(arriving, pending[-1])
        self.assertEqual("step-0", pending[0]["content"]["thinking"])
        self.assertEqual(30, sum(1 for held in pending if held["name"] == "stream"))

    def test_system_budget_stays_bounded_without_weakening_whole_message_guard(self) -> None:
        system = {"name": "system_message", "generation": 1, "subtype": "status", "data": {}}
        for _ in range(_MAX_PENDING_SYSTEM_MESSAGES):
            self.bridge._release_or_hold_message(self.state, system)
        with self.assertRaisesRegex(BridgeError, "system message capacity exceeded"):
            self.bridge._release_or_hold_message(self.state, system)
        for index in range(_MAX_PENDING_EFFECTS):
            self.bridge._release_or_hold_message(
                self.state, {"name": "message", "generation": 1, "message": {"sequence": index}},
            )
        with self.assertRaisesRegex(BridgeError, "pending message capacity exceeded"):
            self.bridge._release_or_hold_message(self.state, {"name": "message", "message": {}})
        self.assertEqual(_MAX_PENDING_SYSTEM_MESSAGES + _MAX_PENDING_EFFECTS,
                         len(self.state["pending_messages"]))

    def test_the_flush_says_how_many_deltas_it_lost(self) -> None:
        for index in range(_MAX_PENDING_EFFECTS + 5):
            self.bridge._release_or_hold_message(self.state, self._delta(index))
        self.state["session_id"] = "11111111-2222-3333-4444-555555555555"

        written: list = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            self.bridge._flush_pending_messages("reservation-a", 1, self.state)

        names = [record["event"]["name"] for record in written]
        self.assertEqual(["stream_deltas_dropped"], names[-1:])
        self.assertEqual(_MAX_PENDING_EFFECTS, names.count("stream"))
        record = written[-1]["event"]
        self.assertEqual(5, record["dropped"])
        self.assertEqual(5, record["dropped_total"])
        # Without this the session runtime cannot attribute the notice
        # to an agent and drops it, so the run log never sees the loss.
        self.assertEqual("reservation-a", record["reservation_id"])
        self.assertTrue(record["correlation_attested"])
        self.assertEqual([], self.state["pending_messages"])

        # A second flush repeats nothing, because nothing else was lost.
        written.clear()
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            self.bridge._flush_pending_messages("reservation-a", 1, self.state)
        self.assertEqual([], written)

    def test_diagnostics_count_the_dropped_deltas(self) -> None:
        for index in range(_MAX_PENDING_EFFECTS + 7):
            self.bridge._release_or_hold_message(self.state, self._delta(index))

        result = asyncio.run(self.bridge.dispatch("diagnostics", {}))

        self.assertEqual(7, result["dropped_stream_delta_count"])
        self.assertEqual(_MAX_PENDING_EFFECTS, result["pending_message_count"])


class ApprovalDetailTests(unittest.TestCase):
    """What a permission request tells the manager about its own effect."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)

    def test_a_write_carries_its_path_and_never_its_contents(self) -> None:
        secret = "AKIA-not-a-real-key and the whole file body"

        detail = _approval_detail(
            "Write",
            self.workspace,
            {"file_path": "src/app.py", "content": secret},
        )

        self.assertEqual({"tool": "Write", "path": "src/app.py"}, detail)
        self.assertNotIn(secret, json.dumps(detail))

    def test_a_write_outside_the_workspace_says_so_without_the_real_path(self) -> None:
        detail = _approval_detail(
            "Write", self.workspace, {"file_path": "../elsewhere/app.py"}
        )

        self.assertEqual({"tool": "Write", "path": "<outside-workspace>"}, detail)

    def test_the_tool_says_which_field_is_the_subject(self) -> None:
        """A scan of candidate names recorded a decoy over the real subject.

        A probe sent WebFetch a real url together with a file_path nothing
        would ever fetch, and the fixed scan met file_path first: the audit
        record named the decoy. Each tool now names its own subject field.
        """

        self.assertEqual(
            "https://real.example/api",
            _approval_detail(
                "WebFetch",
                self.workspace,
                {"url": "https://real.example/api", "file_path": "/tmp/decoy"},
            )["path"],
        )
        self.assertEqual(
            "real query",
            _approval_detail(
                "WebSearch",
                self.workspace,
                {"query": "real query", "path": "/tmp/decoy"},
            )["path"],
        )
        self.assertEqual(
            "/etc/hosts",
            _approval_detail(
                "Read",
                self.workspace,
                {"file_path": "/etc/hosts", "url": "https://decoy.example"},
            )["path"],
        )
        self.assertEqual(
            "notebook.ipynb",
            _approval_detail(
                "NotebookRead",
                self.workspace,
                {"notebook_path": "notebook.ipynb", "file_path": "/tmp/decoy"},
            )["path"],
        )
        # A search carries what it looked for and where it looked.
        self.assertEqual(
            "TODO src/",
            _approval_detail(
                "Grep", self.workspace, {"pattern": "TODO", "path": "src/"}
            )["path"],
        )

    def test_a_read_tool_nobody_named_keeps_the_old_scan(self) -> None:
        """The fallback is for a tool this project does not know the shape of.

        An older bridge, a provider with tools of its own, or a rename upstream
        still gets a subject: the first candidate field that carries one.
        """

        with patch.dict(
            vnext_claude_bridge._TOOL_EFFECTS, {"ReadSomethingNew": "read"}
        ):
            detail = _approval_detail(
                "ReadSomethingNew", self.workspace, {"path": "/var/log/system.log"}
            )

        self.assertEqual("/var/log/system.log", detail["path"])

    def test_a_bash_call_carries_one_truncated_command_line(self) -> None:
        detail = _approval_detail("Bash", self.workspace, {"command": "x" * 900})

        self.assertEqual("Bash", detail["tool"])
        self.assertEqual(1, len(detail["command"]))
        self.assertEqual(400, len(detail["command"][0]))

    def test_a_tool_with_no_named_effect_carries_its_name_alone(self) -> None:
        self.assertEqual(
            {"tool": "SlashCommand"},
            _approval_detail("SlashCommand", self.workspace, {"command": "/deploy"}),
        )
        self.assertEqual(
            {"tool": "Bash"}, _approval_detail("Bash", self.workspace, {})
        )

    def test_a_read_carries_the_file_it_asked_for(self) -> None:
        self.assertEqual(
            {"tool": "Read", "path": "/etc/hosts"},
            _approval_detail("Read", self.workspace, {"file_path": "/etc/hosts"}),
        )
        self.assertEqual(
            {"tool": "Grep", "path": "AKIA"},
            _approval_detail("Grep", self.workspace, {"pattern": "AKIA"}),
        )

    def test_a_web_call_carries_its_url_or_its_query(self) -> None:
        self.assertEqual(
            {"tool": "WebFetch", "path": "https://docs.example.com/api"},
            _approval_detail(
                "WebFetch",
                self.workspace,
                {"url": "https://docs.example.com/api", "prompt": "summarise"},
            ),
        )
        self.assertEqual(
            {"tool": "WebSearch", "path": "claude agent sdk can_use_tool"},
            _approval_detail(
                "WebSearch",
                self.workspace,
                {"query": "claude agent sdk can_use_tool"},
            ),
        )
        self.assertEqual(
            400,
            len(
                _approval_detail(
                    "WebFetch", self.workspace, {"url": "https://x/" + "y" * 900}
                )["path"]
            ),
        )

    def test_the_web_tools_a_stranger_expects_reach_the_reviewer(self) -> None:
        """F17: a tool absent from the table is declined before the grant.

        A field run refused a worker two WebSearch calls while its manager held
        a standing grant, because `review_approval` only understood "modify"
        and "execute" and an unnamed effect is declined.
        """

        self.assertEqual({"effect": "network"}, _tool_effect("WebSearch"))
        self.assertEqual({"effect": "network"}, _tool_effect("WebFetch"))
        self.assertEqual({"effect": "read"}, _tool_effect("Read"))
        self.assertEqual({"effect": "read"}, _tool_effect("Grep"))
        self.assertEqual({"effect": "read"}, _tool_effect("Glob"))
        self.assertEqual({"effect": "read"}, _tool_effect("NotebookRead"))
        # An exec or write tool this bridge cannot name still fails closed.
        self.assertEqual({}, _tool_effect("SlashCommand"))
        self.assertEqual({}, _tool_effect("Monitor"))


class BlockedAgentReleasesItsProcessTests(unittest.TestCase):
    """A blocked Claude worker used to keep its CLI process alive.

    Blocked is not terminal, so the release refused it, and the process sat
    there until somebody cancelled the agent. Release it and let the saved
    native session come back when a later message gives the agent a turn.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def test_the_bridge_disconnects_a_blocked_reservation_and_keeps_it_resumable(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.disconnect_calls = 0

            async def disconnect(self) -> None:
                self.disconnect_calls += 1

        async def scenario() -> None:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = object()
            client = FakeClient()
            state = bridge._reservation_state("auto_review", "native-blocked-session", [])
            state["client"] = client
            bridge._reservations["blocked"] = state

            released = await bridge.dispatch("release_agent", {
                "reservation_id": "blocked", "status": "blocked",
            })

            self.assertTrue(released["released"])
            # Not a terminal owner: the reservation stays this bridge's to
            # reconnect, which is what a resume needs.
            self.assertFalse(released["terminal_owner"])
            self.assertEqual(1, client.disconnect_calls)
            self.assertIsNone(state["client"])

        asyncio.run(scenario())

    def test_a_blocked_release_waits_for_an_unresolved_native_child(self) -> None:
        """This reader is the only observer of the parent's native children.
        Blocked is not terminal, so cancelling it would throw away the terminal
        frame of a child that is still running."""

        class FakeClient:
            def __init__(self) -> None:
                self.disconnect_calls = 0

            async def disconnect(self) -> None:
                self.disconnect_calls += 1

        async def scenario() -> None:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._sdk = object()
            client = FakeClient()
            pending = asyncio.Event()
            reader = asyncio.create_task(pending.wait())
            await asyncio.sleep(0)

            state = bridge._reservation_state("auto_review", "native-blocked-session", [])
            state["client"] = client
            state["task"] = reader
            state["native_children"] = {"task-1": {"status": "running"}}
            bridge._reservations["blocked"] = state

            deferred = await bridge.dispatch("release_agent", {
                "reservation_id": "blocked", "status": "blocked",
            })

            self.assertFalse(deferred["released"])
            self.assertEqual("unresolved-native-children", deferred["deferred"])
            self.assertFalse(reader.done())
            self.assertIs(client, state["client"])
            self.assertEqual(0, client.disconnect_calls)
            # The release is only waiting: the reader itself completes it once
            # the children settle.
            self.assertTrue(state["release_requested"])

            # A terminal status keeps today's behaviour and stops the reader.
            cancelled = await bridge.dispatch("release_agent", {
                "reservation_id": "blocked", "status": "cancelled",
            })

            self.assertTrue(cancelled["released"])
            self.assertTrue(reader.cancelled())
            self.assertEqual(1, client.disconnect_calls)
            self.assertIsNone(state["client"])
            pending.set()

        asyncio.run(scenario())

    def test_a_status_that_is_not_settled_is_still_refused(self) -> None:
        async def scenario() -> None:
            bridge = _Bridge()
            bridge._workspace = Path(self.workspace)
            bridge._reservations["running"] = bridge._reservation_state("auto_review", None, [])
            with self.assertRaises(BridgeError) as refused:
                await bridge.dispatch("release_agent", {
                    "reservation_id": "running", "status": "running",
                })
            self.assertIn("settled status", str(refused.exception))

        asyncio.run(scenario())

    def test_the_adapter_releases_on_blocked_and_the_next_turn_reconnects(self) -> None:
        adapter = ClaudeCodeAdapter(workspace=self.workspace)
        adapter._threads["reservation"] = {
            "policy": {"posture": {"workspace_writes": True, "network": "approval_gated",
                                   "approvals_requested": True, "reviewer": "auto_review",
                                   "environment_ready": True}},
            "provider_session": "native-blocked-session",
            "binding_phase": "attested",
            "generation": 1,
            "model": CLAUDE_WORKER_MODEL,
            "effort": "high",
            "permission_mode": "default",
            "developer_instructions": "ROLE=worker",
            "workspace": self.workspace,
            "released": False,
        }
        asked: list[tuple[str, dict]] = []

        def request(op, payload, **_kwargs):
            asked.append((op, dict(payload)))
            if op == "release_agent":
                return {"reservation_echo": payload["reservation_id"],
                        "released": True, "terminal_owner": False}
            if op == "resume":
                return {
                    "provider_echo": True,
                    "thread_id": "native-blocked-session",
                    "reservation_echo": payload["reservation_id"],
                    "policy": {"posture": {"workspace_writes": True,
                                           "network": "approval_gated",
                                           "approvals_requested": True,
                                           "reviewer": "auto_review",
                                           "environment_ready": True}},
                    "connection_evidence": {"connected": True, "server_info_received": True},
                    "tool_registration": {"acknowledged": True, "model_id": CLAUDE_WORKER_MODEL,
                                          "tool_count": 0, "tool_names": [],
                                          "definition_sha256": None, "handler_registered": False},
                }
            if op == "start_turn":
                return {"reservation_echo": payload["reservation_id"],
                        "turn_echo": payload["turn_reference"], "cursor": 1}
            raise AssertionError(f"unexpected bridge op: {op}")

        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread("reservation", status="blocked")
            self.assertEqual(("release_agent", {"reservation_id": "reservation",
                                                "status": "blocked"}), asked[0])
            self.assertTrue(adapter._threads["reservation"]["released"])

            # The agent is resumable: readiness reconnects the saved session
            # and the next prompt really starts a turn on it.
            self.assertTrue(adapter.can_start_turn("reservation"))
            self.assertEqual("resume", asked[1][0])
            self.assertFalse(adapter._threads["reservation"]["released"])

            handle = adapter.start_turn(thread_id="reservation", prompt="carry on")

        self.assertEqual("reservation", handle.thread_id)
        self.assertEqual("start_turn", asked[2][0])
        self.assertEqual("carry on", asked[2][1]["prompt"])


CATALOG_ALIAS = "opus"
PROVIDER_MODEL = "claude-opus-5"


class NativeChildAliasModelTests(unittest.TestCase):
    """A catalog alias launches the worker; the provider answers canonically.

    Production configures a Claude worker with the catalog id ``opus``
    (``DEFAULT_CATALOG`` in ``vnext_mcp_server.py``) and the adapter passes that
    string into the bridge, where it becomes ``native_child_configured_model``.
    The provider's child AssistantMessage carries ``claude-opus-5``.  Comparing
    the two for equality killed a live worker's turn on 2026-09-30 23:22:09 UTC.
    The rest of this module configures ``claude-opus-4-7``, so the two strings
    always matched and the suite never saw it.
    """

    SESSION = "native-parent-session"

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def _prepare(self, model: str = CATALOG_ALIAS):
        recorder: list = []
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = ClaudeBridgeToolHostingTests._fake_sdk(recorder)
        state = bridge._reservation_state("auto_review", None, list(MANAGER_TOOLS))
        state["turn_reference"], state["generation"] = "root-turn", 1
        bridge._reservations["reservation-a"] = state
        bridge._options(
            model=model,
            resume=None,
            reservation_id="reservation-a",
            definitions=list(MANAGER_TOOLS),
        )
        return bridge, state, recorder[-1]

    def _launch(self, options, session: str):
        """Run the two hooks that join one native child to its parent tool use."""

        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        started = options["hooks"]["SubagentStart"][0].hooks[0]
        asyncio.run(pretool({
            "tool_name": "Agent", "session_id": session,
            "tool_input": {
                "prompt": "Map close path and lock order",
                "description": "Map close path and lock order",
                "model": CATALOG_ALIAS,
            },
        }, "parent-tool", None))
        asyncio.run(started({"session_id": session, "agent_id": "child-agent"}, None, None))

    def test_a_canonical_child_assistant_message_keeps_the_parent_turn(self) -> None:
        bridge, state, options = self._prepare()

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": CATALOG_ALIAS}

        class AssistantMessage:
            session_id, parent_tool_use_id = NativeChildAliasModelTests.SESSION, "parent-tool"
            model, content, stop_reason = PROVIDER_MODEL, [], None

        bridge._sdk.get_subagent_messages = lambda *_a, **_k: [Metadata()]
        with patch("vnext.vnext_claude_bridge._write"):
            self._launch(options, self.SESSION)
            # This is the frame the live reader was on when the turn died.
            bridge._emit_message("reservation-a", 1, AssistantMessage())
        forwarded = state["native_child_forwarded_models"][(self.SESSION, "parent-tool")]
        self.assertEqual(PROVIDER_MODEL, forwarded["model"])

    def test_canonical_saved_metadata_is_admitted_under_an_alias_launch(self) -> None:
        bridge, state, options = self._prepare()

        class Metadata:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": PROVIDER_MODEL}

        class ResultMessage:
            session_id, parent_tool_use_id = NativeChildAliasModelTests.SESSION, None
            model, content, stop_reason = PROVIDER_MODEL, [], None

        bridge._sdk.get_subagent_messages = lambda *_a, **_k: [Metadata()]
        with patch("vnext.vnext_claude_bridge._write"):
            self._launch(options, self.SESSION)
            bridge._emit_message("reservation-a", 1, ResultMessage())
        key = (self.SESSION, "child-agent")
        self.assertEqual(PROVIDER_MODEL, state["native_child_observed_models"][key])
        self.assertEqual(
            "saved-child-assistant-message",
            state["native_child_observed_model_sources"][key],
        )

    def test_the_live_order_upgrades_the_alias_to_the_canonical_id(self) -> None:
        """The exact live order: ``agent-*.meta.json`` says ``opus``, the
        child's own assistant frame in the store says ``claude-opus-5``."""

        bridge, state, options = self._prepare()
        resolver = state["native_child_automatic_resolver"]

        class FakeMirror:
            @staticmethod
            def is_conflicted(*_a, **_k):
                return False

            @staticmethod
            def metadata_for_agent(*_a, **_k):
                # Verbatim shape of the live agent-a6620021c3e890ec0.meta.json
                return {"toolUseId": "parent-tool", "model": CATALOG_ALIAS,
                        "description": "Map close path and lock order"}

        class StoreMessage:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": PROVIDER_MODEL}

        key = (self.SESSION, "child-agent")
        with patch("vnext.vnext_claude_bridge._write"):
            self._launch(options, self.SESSION)
            bridge._record_mirrored_agent_metadata(
                state, resolver, self.SESSION, "child-agent", FakeMirror()
            )
            self.assertEqual(CATALOG_ALIAS, state["native_child_observed_models"][key])
            bridge._record_automatic_native_metadata(
                state, resolver, self.SESSION, "child-agent", [StoreMessage()]
            )
        self.assertEqual(PROVIDER_MODEL, state["native_child_observed_models"][key])
        # Mirror metadata that still echoes the alias cannot undo that.
        bridge._record_mirrored_agent_metadata(
            state, resolver, self.SESSION, "child-agent", FakeMirror()
        )
        self.assertEqual(PROVIDER_MODEL, state["native_child_observed_models"][key])

    def test_an_alias_launch_still_refuses_two_canonical_models_for_one_child(self) -> None:
        """The guard the equality check was built for stays."""

        bridge, state, options = self._prepare()
        resolver = state["native_child_automatic_resolver"]

        def store(model: str):
            return type("StoreMessage", (), {
                "parent_tool_use_id": "parent-tool", "parent_agent_id": None,
                "message": {"role": "assistant", "model": model},
            })()

        with patch("vnext.vnext_claude_bridge._write"):
            self._launch(options, self.SESSION)
            bridge._record_automatic_native_metadata(
                state, resolver, self.SESSION, "child-agent", [store(PROVIDER_MODEL)]
            )
            with self.assertRaisesRegex(NativeChildIdentityError, "conflicts with prior metadata"):
                bridge._record_automatic_native_metadata(
                    state, resolver, self.SESSION, "child-agent", [store("claude-sonnet-4-5")]
                )
        key = (self.SESSION, "child-agent")
        self.assertEqual(PROVIDER_MODEL, state["native_child_observed_models"][key])

    def test_an_alias_launch_still_refuses_a_second_canonical_forwarded_model(self) -> None:
        bridge, state, options = self._prepare()

        def assistant(model: str):
            return type("AssistantMessage", (), {
                "session_id": NativeChildAliasModelTests.SESSION,
                "parent_tool_use_id": "parent-tool",
                "model": model, "content": [], "stop_reason": None,
            })()

        with patch("vnext.vnext_claude_bridge._write"):
            self._launch(options, self.SESSION)
            bridge._emit_message("reservation-a", 1, assistant(PROVIDER_MODEL))
            with self.assertRaisesRegex(NativeChildIdentityError, "effective model conflicts"):
                bridge._emit_message("reservation-a", 1, assistant("claude-sonnet-4-5"))

    def test_a_canonical_launch_still_refuses_a_different_canonical_child(self) -> None:
        """An exact configured provider id keeps exact comparison."""

        bridge, state, options = self._prepare(model=CLAUDE_WORKER_MODEL)
        resolver = state["native_child_automatic_resolver"]

        class StoreMessage:
            parent_tool_use_id, parent_agent_id = "parent-tool", None
            message = {"role": "assistant", "model": PROVIDER_MODEL}

        with patch("vnext.vnext_claude_bridge._write"):
            self._launch(options, self.SESSION)
            with self.assertRaisesRegex(NativeChildIdentityError, "conflicts with configured model"):
                bridge._record_automatic_native_metadata(
                    state, resolver, self.SESSION, "child-agent", [StoreMessage()]
                )
        self.assertEqual({}, state["native_child_observed_models"])


class TurnFailureCarriesItsReasonTests(unittest.TestCase):
    """A lost turn has to say what ended it.

    The reader formatted the exception type alone, so the live record read
    ``Claude query failed: NativeChildIdentityError`` and the sentence that
    named the disagreement stayed in the bridge subprocess.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def _bridge(self) -> _Bridge:
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = object()
        return bridge

    def test_the_reader_failure_names_the_exception_message(self) -> None:
        class Client:
            async def query(self, _prompt: str) -> None:
                pass

            async def receive_messages(self):
                yield SimpleNamespace(name="frame")

        bridge = self._bridge()
        state = bridge._reservation_state("auto_review", "native-session", [])
        state["client"] = Client()
        bridge._reservations["reservation"] = state

        def explode(*_args: object) -> None:
            raise NativeChildIdentityError("native child observed model conflicts with prior metadata")

        with patch.object(bridge, "_emit_message", side_effect=explode):
            with self.assertRaises(BridgeError) as failed:
                asyncio.run(bridge._run_query("reservation", 1, "bounded work"))

        self.assertEqual(
            "Claude query failed: NativeChildIdentityError: "
            "native child observed model conflicts with prior metadata",
            str(failed.exception),
        )

    def test_an_abandoned_wait_failure_names_the_exception_message(self) -> None:
        async def scenario() -> None:
            bridge = self._bridge()
            state = bridge._reservation_state("auto_review", "native-session", [])
            bridge._reservations["reservation"] = state
            completion: asyncio.Future = asyncio.get_running_loop().create_future()
            state["turn_outcomes"] = {"turn-a": {
                "turn_reference": "turn-a", "completion": completion,
            }}
            completion.set_exception(ValueError("the store said claude-sonnet-4-5"))
            with self.assertRaises(BridgeError) as failed:
                await bridge.dispatch("wait_turn", {
                    "reservation_id": "reservation", "turn_reference": "turn-a",
                })
            self.assertEqual(
                "Claude query failed: ValueError: the store said claude-sonnet-4-5",
                str(failed.exception),
            )

        asyncio.run(scenario())


class NativeChildOfAReleasedParentTests(unittest.TestCase):
    """A child cannot outlive the reader that was its only route home.

    On 2026-09-30 a failed parent turn released its SDK client while a native
    child was still running. ``cancel_agent`` on the child then answered
    ``native task stop has no active SDK controller``, and the child's vNext
    wait sat for its whole 1800 s timeout for a frame no reader could deliver.
    """

    PARENT = "reservation"
    CHILD = "claude-native:native-blocked-session:agent-child"

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def _adapter(self, *, released: bool) -> ClaudeCodeAdapter:
        adapter = ClaudeCodeAdapter(workspace=self.workspace)
        adapter._threads[self.PARENT] = {
            "policy": {"posture": RESOLVED_POSTURE.as_dict()},
            "provider_session": "native-blocked-session",
            "binding_phase": "attested",
            "generation": 1,
            "model": CATALOG_ALIAS,
            "effort": "high",
            "permission_mode": "default",
            "developer_instructions": "ROLE=worker",
            "workspace": self.workspace,
            "released": released,
            "active_turn": None,
        }
        adapter._native_child_tasks[self.CHILD] = {
            "reservation_id": self.PARENT,
            "task_id": "task-1",
            "status": "running",
            "summary": None,
        }
        return adapter

    @staticmethod
    def _request(asked: list):
        def request(op, payload, **_kwargs):
            asked.append((op, dict(payload)))
            if op == "release_agent":
                return {"reservation_echo": payload["reservation_id"],
                        "released": True, "terminal_owner": False}
            raise AssertionError(f"unexpected bridge op: {op}")

        return request

    def test_releasing_a_parent_ends_its_running_child_wait_at_once(self) -> None:
        adapter = self._adapter(released=False)
        asked: list = []
        with patch.object(adapter, "_request", side_effect=self._request(asked)):
            adapter.release_terminal_thread(self.PARENT, status="blocked")
        self.assertEqual("release_agent", asked[0][0])

        started = time.monotonic()
        result = adapter.wait_turn(TurnHandle(self.CHILD, "task-1"), timeout_seconds=1800)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual("interrupted", result["status"])
        self.assertTrue(result["native_task_terminal"])
        self.assertIn("SDK reader can no longer deliver", result["summary"])

    def test_a_deferred_release_leaves_a_running_child_alone(self) -> None:
        """A release the bridge deferred still has its reader."""

        adapter = self._adapter(released=False)

        def request(op, payload, **_kwargs):
            self.assertEqual("release_agent", op)
            return {"reservation_echo": payload["reservation_id"], "released": False,
                    "terminal_owner": False, "deferred": "unresolved-native-children"}

        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.PARENT, status="blocked")
        self.assertEqual("running", adapter._native_child_tasks[self.CHILD]["status"])
        self.assertIs(False, adapter._threads[self.PARENT]["released"])

    def test_cancelling_a_child_of_a_deferred_parent_still_asks_the_bridge(self) -> None:
        """A deferred release keeps the controller that owns ``stop_task``.

        On 2026-09-30 the adapter marked the parent released even when the
        bridge answered ``released=False, deferred=unresolved-native-children``.
        The next cancel took the parent-released branch, killed the child in the
        ledger alone, and never called ``stop_native_task`` on the live client.
        """

        adapter = self._adapter(released=False)
        asked: list[str] = []

        def request(op, payload, **_kwargs):
            asked.append(op)
            if op == "release_agent":
                return {"reservation_echo": payload["reservation_id"], "released": False,
                        "terminal_owner": False, "deferred": "unresolved-native-children"}
            if op == "stop_native_task":
                return {"reservation_echo": payload["reservation_id"],
                        "task_id": payload["task_id"], "accepted": True}
            raise AssertionError(f"unexpected bridge op: {op}")

        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.PARENT, status="blocked")
            result = adapter.cancel_native_child(self.CHILD)

        self.assertEqual("interrupt-requested", result["status"])
        self.assertEqual(["release_agent", "stop_native_task"], asked)
        self.assertEqual("running", adapter._native_child_tasks[self.CHILD]["status"])
        self.assertIsNone(adapter._fatal)

    def test_a_late_terminal_frame_after_a_local_kill_is_recorded_and_ignored(self) -> None:
        """The provider's own outcome can arrive after vNext ended the child.

        Settling the child locally is a vNext fact. A frame the provider sent
        before the disconnect contradicts no earlier provider evidence, so it is
        kept as late news: one terminal state, no second settlement, no fatal.
        """

        adapter = self._adapter(released=False)
        task = adapter._native_child_tasks[self.CHILD]
        task.update(generation=1, parent_turn_reference="root-turn",
                    parent_session_id="native-blocked-session")
        with patch.object(adapter, "_request", side_effect=self._request([])):
            adapter.release_terminal_thread(self.PARENT, status="blocked")
        self.assertEqual("killed", task["status"])

        late = {
            "native_runtime_thread_id": self.CHILD, "task_id": "task-1",
            "status": "completed", "turn_reference": "task-1",
            "reservation_id": self.PARENT, "generation": 1,
            "parent_turn_reference": "root-turn", "correlation_attested": True,
            "provider_correlation": {"session": "native-blocked-session", "turn": "root-turn"},
        }
        self.assertFalse(adapter._record_native_child_completion(late))
        self.assertIsNone(adapter._fatal)
        self.assertEqual("killed", task["status"])
        self.assertEqual("completed", task["late_provider_status"])
        # Repeating it stays harmless.
        self.assertFalse(adapter._record_native_child_completion(late))
        self.assertIsNone(adapter._fatal)
        self.assertEqual(
            "interrupted",
            adapter.wait_turn(TurnHandle(self.CHILD, "task-1"), timeout_seconds=0.1)["status"],
        )

    def test_closing_the_adapter_ends_every_running_native_child_wait(self) -> None:
        """Shutdown removes the readers, so no child can still report home."""

        adapter = self._adapter(released=False)
        adapter.close()
        task = adapter._native_child_tasks[self.CHILD]
        self.assertEqual("killed", task["status"])
        self.assertIn("no SDK reader remains", task["summary"])

        started = time.monotonic()
        result = adapter.wait_turn(TurnHandle(self.CHILD, "task-1"), timeout_seconds=1800)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual("interrupted", result["status"])

    def test_cancelling_a_child_after_its_parent_was_released_succeeds(self) -> None:
        adapter = self._adapter(released=True)

        def refuse(op, _payload, **_kwargs):
            # What the bridge answers once it disconnected the client that
            # owned ``stop_task``.
            raise ClaudeRuntimeError(
                "Claude bridge rejected " + op + ": native task stop has no active SDK controller"
            )

        with patch.object(adapter, "_request", side_effect=refuse):
            result = adapter.cancel_native_child(self.CHILD)
        self.assertEqual("parent-released", result["status"])
        self.assertIn("SDK reader can no longer deliver", result["reason"])
        self.assertEqual("killed", adapter._native_child_tasks[self.CHILD]["status"])
        self.assertEqual(
            "interrupted",
            adapter.wait_turn(TurnHandle(self.CHILD, "task-1"), timeout_seconds=0.1)["status"],
        )


class DeferredReleaseCarriesTheFinalUsageTests(unittest.TestCase):
    """A quit right after a worker settles used to throw away its bill.

    ``agent.terminal`` is written about a second before the SDK's ``result``
    message, the one that carries ``usage`` and ``total_cost_usd``.  The bridge
    defers the release and keeps the reader, so the figure does arrive; what was
    missing is anybody waiting for it.  Measured on 2026-10-01: three quits with
    no pause lost the record, two with a five second pause kept it (207,933
    tokens, $0.238).
    """

    THREAD = "reservation"

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def _adapter(self) -> ClaudeCodeAdapter:
        adapter = ClaudeCodeAdapter(workspace=self.workspace)
        adapter._threads[self.THREAD] = {
            "provider_session": "native-session",
            "binding_phase": "attested",
            "generation": 1,
            "model": "claude-opus-4-7",
            "workspace": self.workspace,
            "released": False,
        }
        return adapter

    def test_a_deferred_release_is_named_as_usage_still_owed(self) -> None:
        adapter = self._adapter()

        def request(op, payload, **_kwargs):
            self.assertEqual("release_agent", op)
            return {"reservation_echo": payload["reservation_id"],
                    "released": False, "terminal_owner": False}

        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.THREAD, status="completed")

        self.assertEqual((self.THREAD,), adapter.pending_usage_releases())

    def test_awaiting_a_deferred_release_collects_the_usage_the_reader_owed(self) -> None:
        adapter = self._adapter()
        asked: list[str] = []

        def request(op, payload, **_kwargs):
            asked.append(op)
            if len(asked) == 1:
                return {"reservation_echo": payload["reservation_id"],
                        "released": False, "terminal_owner": False}
            # The reader has drained by now, so it emitted the final usage
            # before it released itself.
            adapter._events.append({"name": "usage", "reservation_id": self.THREAD,
                                    "usage": {"totalTokens": 207933},
                                    "total_cost_usd": 0.238})
            return {"reservation_echo": payload["reservation_id"],
                    "released": True, "terminal_owner": False}

        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.THREAD, status="completed")
            received = adapter.await_pending_usage_releases(2.0)

        self.assertEqual({self.THREAD: True}, received)
        self.assertEqual((), adapter.pending_usage_releases())
        self.assertTrue(adapter._threads[self.THREAD]["released"])
        self.assertTrue(any(event.get("name") == "usage" for event in adapter._events))

    def test_a_reader_that_never_answers_gives_the_budget_back_with_a_verdict(self) -> None:
        adapter = self._adapter()

        def request(op, payload, **_kwargs):
            return {"reservation_echo": payload["reservation_id"],
                    "released": False, "terminal_owner": False}

        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.THREAD, status="completed")
            started = time.monotonic()
            received = adapter.await_pending_usage_releases(0.3)
            spent = time.monotonic() - started

        self.assertEqual({self.THREAD: False}, received)
        self.assertLess(spent, 2.0)

    def test_nothing_pending_asks_the_bridge_nothing_and_returns_at_once(self) -> None:
        adapter = self._adapter()
        adapter._threads[self.THREAD]["released"] = True
        asked: list[str] = []

        def request(op, _payload, **_kwargs):
            asked.append(op)
            raise AssertionError("a close with no deferred release must not call the bridge")

        with patch.object(adapter, "_request", side_effect=request):
            started = time.monotonic()
            self.assertEqual({}, adapter.await_pending_usage_releases(5.0))
            spent = time.monotonic() - started

        self.assertEqual([], asked)
        self.assertLess(spent, 0.05)

    def test_a_bridge_write_failure_is_recorded_as_a_named_reason(self) -> None:
        """An OSError on the pipe must still leave a verdict behind.

        ``_send`` converts an ordinary write error into a ``ClaudeRuntimeError``,
        but a pipe that breaks under the lock, an EOF, or any other fault
        escaped this wait as itself.  The caller then recorded a
        ``telemetry.error`` and nothing else, so the outcome row kept a silent
        null where the reason belonged.
        """

        adapter = self._adapter()

        def request(op, payload, **_kwargs):
            if len(asked) == 0:
                asked.append(op)
                return {"reservation_echo": payload["reservation_id"],
                        "released": False, "terminal_owner": False}
            raise OSError("bridge write failed")

        asked: list[str] = []
        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.THREAD, status="completed")
            received = adapter.await_pending_usage_releases(0.3)

        self.assertEqual({self.THREAD: False}, received)
        reason = adapter.usage_release_failures().get(self.THREAD, "")
        self.assertIn("bridge write failed", reason)
        self.assertIn("OSError", reason)

    def test_an_interrupt_during_the_wait_is_not_swallowed(self) -> None:
        adapter = self._adapter()

        def request(op, payload, **_kwargs):
            if not asked:
                asked.append(op)
                return {"reservation_echo": payload["reservation_id"],
                        "released": False, "terminal_owner": False}
            raise KeyboardInterrupt

        asked: list[str] = []
        with patch.object(adapter, "_request", side_effect=request):
            adapter.release_terminal_thread(self.THREAD, status="completed")
            with self.assertRaises(KeyboardInterrupt):
                adapter.await_pending_usage_releases(0.3)

    def test_a_stalled_reservation_leaves_the_budget_for_its_ready_sibling(self) -> None:
        """One slow reader must not spend the sibling's share of the wait.

        The stalled reservation answers only when its own request deadline has
        run out.  Asked with the whole budget it consumed all of it, the ready
        sibling was never asked, and that sibling's row was called missing while
        its bill was already queued.
        """

        adapter = self._adapter()
        adapter._threads["ready"] = dict(adapter._threads[self.THREAD])
        asked: list[str] = []

        def request(_op, payload, **kwargs):
            name = payload["reservation_id"]
            asked.append(name)
            if name == self.THREAD:
                time.sleep(kwargs["timeout_seconds"])
                return {"reservation_echo": name, "released": False, "terminal_owner": False}
            adapter._events.append({"name": "usage", "reservation_id": name,
                                    "usage": {"totalTokens": 100}, "total_cost_usd": 0.01})
            return {"reservation_echo": name, "released": True, "terminal_owner": False}

        with patch.object(adapter, "_request", side_effect=request):
            adapter._threads[self.THREAD]["release_deferred"] = True
            adapter._threads[self.THREAD]["release_status"] = "completed"
            adapter._threads["ready"]["release_deferred"] = True
            adapter._threads["ready"]["release_status"] = "completed"
            received = adapter.await_pending_usage_releases(0.3)

        self.assertIn("ready", asked)
        self.assertEqual({self.THREAD: False, "ready": True}, received)

    def test_a_stalled_reservation_that_overruns_its_share_still_leaves_the_sibling_one_ask(self) -> None:
        """A slow host oversleeps a request deadline; the sibling is still asked.

        GitHub's macOS runners returned from the stalled request after the whole
        budget was gone, and the ready sibling was skipped.  The overrun here is
        made explicit so the test does not depend on the host's timer.
        """

        adapter = self._adapter()
        adapter._threads["ready"] = dict(adapter._threads[self.THREAD])
        asked: list[str] = []

        def request(_op, payload, **kwargs):
            name = payload["reservation_id"]
            asked.append(name)
            if name == self.THREAD:
                time.sleep(kwargs["timeout_seconds"] + 0.3)
                return {"reservation_echo": name, "released": False, "terminal_owner": False}
            adapter._events.append({"name": "usage", "reservation_id": name,
                                    "usage": {"totalTokens": 100}, "total_cost_usd": 0.01})
            return {"reservation_echo": name, "released": True, "terminal_owner": False}

        with patch.object(adapter, "_request", side_effect=request):
            for thread_id in (self.THREAD, "ready"):
                adapter._threads[thread_id]["release_deferred"] = True
                adapter._threads[thread_id]["release_status"] = "completed"
            received = adapter.await_pending_usage_releases(0.3)

        self.assertEqual([self.THREAD, "ready"], asked)
        self.assertEqual({self.THREAD: False, "ready": True}, received)


class AnErrorCodeThatIsNotTextTests(unittest.TestCase):
    """R18: a list or an object as an error code raised TypeError.

    Both readers tested the code with set membership, which hashes the value.
    A provider that sent a structured code took down the reader that was there
    to describe the failure.
    """

    def test_the_bridge_ignores_a_structured_assistant_error(self):
        for code in (["authentication_failed"], {"kind": "rate_limit"}):
            with self.subTest(code=code):
                message = SimpleNamespace(error=code)
                self.assertIsNone(_Bridge._provider_error_projection(message, "AssistantMessage"))
        message = SimpleNamespace(error="rate_limit")
        self.assertEqual("rate_limit", _Bridge._provider_error_projection(message, "AssistantMessage")["code"])

    def test_the_adapter_reads_a_structured_code_as_no_named_refusal(self):
        detail = ClaudeCodeAdapter._credential_refusal_detail
        self.assertIsNone(detail({"code": ["authentication_failed"], "subtype": {"a": 1}}))
        self.assertEqual("HTTP 401", detail({"code": ["x"], "api_error_status": 401}))
        self.assertEqual("authentication_failed", detail({"code": "authentication_failed"}))


class ClaudeExactModelIdentityTests(unittest.TestCase):
    """The alias a worker asked for is never the only name a record keeps."""

    ROWS = [
        {"value": "default", "resolvedModel": "claude-opus-5-5"},
        {"value": "opus", "resolvedModel": "claude-opus-5-5"},
        {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5"},
        {"value": "haiku", "resolvedModel": "claude-haiku-4-5-20251001"},
        {"value": "claude-fable-5-1", "resolvedModel": "claude-fable-5-1"},
    ]

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = str(Path(self.temp.name).resolve())

    def _fake_sdk(self, rows_by_call: list) -> object:
        class FakeSDK:
            def __init__(self) -> None:
                self.clients: list[object] = []

            @staticmethod
            def ClaudeAgentOptions(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HookMatcher(**kwargs: object) -> object:
                return SimpleNamespace(**kwargs)

            def ClaudeSDKClient(sdk_self, *, options: object):
                rows = rows_by_call[len(sdk_self.clients)]

                class Client:
                    async def connect(self, prompt: object) -> None:
                        pass

                    async def get_server_info(self) -> dict:
                        return {"commands": [], "models": rows}

                    async def disconnect(self) -> None:
                        pass

                client = Client()
                sdk_self.clients.append(client)
                return client

        return FakeSDK()

    def _start(self, bridge: object, model: str, reservation: str) -> dict:
        return asyncio.run(bridge.dispatch("start_thread", {
            "workspace": self.workspace,
            "model": model,
            "requested_posture": REQUESTED_POSTURE.as_dict(),
            "reservation_id": reservation,
        }))

    def test_an_alias_with_its_own_row_resolves_from_server_info(self) -> None:
        from vnext.vnext_model_identity import resolve_claude_alias

        self.assertEqual(("claude-opus-5-5", "server_info"), resolve_claude_alias("opus", self.ROWS))
        self.assertEqual(("claude-sonnet-5-5", "server_info"), resolve_claude_alias("sonnet", self.ROWS))

    def test_fable_has_no_row_and_resolves_by_its_family(self) -> None:
        from vnext.vnext_model_identity import resolve_claude_alias

        self.assertEqual(("claude-fable-5-1", "family_match"), resolve_claude_alias("fable", self.ROWS))
        two = self.ROWS + [{"value": "x", "resolvedModel": "claude-fable-5-2"}]
        self.assertEqual((None, None), resolve_claude_alias("fable", two))
        self.assertEqual((None, None), resolve_claude_alias("glm-5.3", self.ROWS))

    def test_start_thread_reports_the_exact_model_for_the_alias(self) -> None:
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = self._fake_sdk([self.ROWS])
        result = self._start(bridge, "fable", "claude-reservation-fable")
        self.assertEqual("claude-fable-5-1", result["model_identity"]["model_exact"])
        self.assertEqual("family_match", result["model_identity"]["model_exact_source"])
        state = bridge._reservations["claude-reservation-fable"]
        self.assertEqual("claude-fable-5-1", state["model_exact"])
        asyncio.run(bridge.dispatch("close", {}))

    def test_resume_resolves_again_and_keeps_what_the_alias_meant_before(self) -> None:
        later = [dict(row) for row in self.ROWS]
        later[1] = {"value": "opus", "resolvedModel": "claude-opus-6"}
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        bridge._sdk = self._fake_sdk([self.ROWS, later])
        self._start(bridge, "opus", "claude-reservation-opus")
        state = bridge._reservations["claude-reservation-opus"]
        state["release_requested"] = True
        result = asyncio.run(bridge.dispatch("resume", {
            "session_id": "native-resumed-session",
            "reservation_id": "claude-reservation-opus",
            "workspace": self.workspace,
            "model": "opus",
            "requested_posture": REQUESTED_POSTURE.as_dict(),
        }))
        identity = result["model_identity"]
        self.assertEqual("claude-opus-6", identity["model_exact"])
        self.assertEqual(["claude-opus-5-5"], identity["model_exact_history"])
        asyncio.run(bridge.dispatch("close", {}))

    def test_a_manager_tool_call_carries_the_model_that_answered_so_far(self) -> None:
        """complete_agent can arrive mid-turn; the adapter must already know model_ran."""

        from vnext.vnext_claude import ClaudeCodeAdapter

        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        state = bridge._reservation_state("auto_review", None, [])
        bridge._reservations["r"] = state
        state["model_ran"] = "claude-opus-5-5"
        state["model_ran_first"] = "claude-opus-5-5"
        state["turn_reference"] = "turn-1"
        written: list = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=lambda *a, **k: written.append(a)):
            try:
                asyncio.run(asyncio.wait_for(bridge._call_control_plane("r", "complete_agent", {}), 0.2))
            except Exception:
                pass
        events = [item for args in written for item in args if isinstance(item, dict)]
        calls = [e.get("event", e) for e in events if e.get("event", e).get("name") == "tool_call"]
        self.assertTrue(calls, written)
        self.assertEqual("claude-opus-5-5", calls[0]["model_ran"])

        adapter = object.__new__(ClaudeCodeAdapter)
        adapter._threads = {"r": {}}
        with patch.object(adapter, "_handle_tool_call"):
            adapter._record_event({"event": calls[0]})
        self.assertEqual("claude-opus-5-5", adapter.model_identity("r")["model_ran"])

    def test_only_the_workers_own_reply_sets_model_ran(self) -> None:
        bridge = _Bridge()
        bridge._workspace = Path(self.workspace)
        state = bridge._reservation_state("auto_review", None, [])
        bridge._reservations["r"] = state

        class AssistantMessage:
            def __init__(self, model: str, parent: str | None) -> None:
                self.model = model
                self.parent_tool_use_id = parent
                self.session_id = None
                self.content = []

        with patch.object(bridge, "_project_message", return_value=None):
            bridge._emit_message("r", 0, AssistantMessage("claude-haiku-4-5", "toolu_child"))
            self.assertNotIn("model_ran", state)
            bridge._emit_message("r", 0, AssistantMessage("claude-opus-5-5", None))
            bridge._emit_message("r", 0, AssistantMessage("claude-haiku-4-5", "toolu_child"))
        self.assertEqual("claude-opus-5-5", state["model_ran"])
        self.assertEqual("claude-opus-5-5", state["model_ran_first"])
