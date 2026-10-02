"""Provider-free automatic Claude child path through the service seam."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vnext.vnext_claude import ClaudeCodeAdapter
from vnext.vnext_claude_bridge import _Bridge
from vnext.vnext_managed_session import VNextManagedSession
from vnext.vnext_orchestration import (
    AgentRole,
    EconomicPreset,
    ModelCard,
    ModelRegistry,
    OrchestrationControlPlane,
)
from vnext.vnext_runtime_effects import RuntimeEffectJournal, runtime_effect_reader
from vnext.vnext_scheduler import VNextScheduler
from vnext.workforce_contracts import RunCancellation


MODEL = "claude-service-fixture"
RESERVATION = "reservation-service"
SESSION = "native-parent-session"
PARENT_TOOL = "parent-agent-tool"
TASK = "native-child-task"
CHILD = "native-child-agent"

DELEGATE = {
    "name": "delegate",
    "description": "Delegate a bounded worker task.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "model_id": {"type": "string"},
            "objective": {"type": "string"},
            "task_contract": {"type": "object"},
            "workspace": {"type": "string"},
            "effort": {"type": "string"},
        },
    },
}


class _FakeSDK:
    def __init__(self) -> None:
        self.options: list[dict[str, object]] = []

    def ClaudeAgentOptions(self, **kwargs: object) -> object:
        self.options.append(kwargs)
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

    @staticmethod
    def PermissionResultAllow(**kwargs: object) -> object:
        return SimpleNamespace(allow=True, **kwargs)

    @staticmethod
    def PermissionResultDeny(**kwargs: object) -> object:
        return SimpleNamespace(allow=False, **kwargs)


class AutomaticClaudeServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        registry = ModelRegistry(
            cards=[ModelCard(MODEL, frozenset({AgentRole.ROOT_MANAGER, AgentRole.WORKER}), provider="claude")],
            presets=[EconomicPreset("fixture", frozenset({MODEL}), frozenset({"claude"}))],
        )
        self.control = OrchestrationControlPlane(registry)
        self.root = self.control.create_session(
            preset_id="fixture",
            root_model_id=MODEL,
            workspace=self.workspace,
            objective="Exercise automatic native child adoption",
            task_contract={"criteria": ["automatic child is bound"]},
            session_id="automatic-service",
        )
        self.adapter = ClaudeCodeAdapter(workspace=str(self.workspace))
        self.adapter._threads[RESERVATION] = {
            "policy": {"posture": self._posture()},
            "provider_session": SESSION,
            "binding_phase": "attested",
            "generation": 1,
            "model": MODEL,
            "effort": "high",
            "permission_mode": "default",
            "active_turn": None,
        }
        effects = RuntimeEffectJournal(self.workspace)
        # Production selects the effect decoder together with the adapter
        # before a parent can adopt a native child.  This direct service fixture
        # binds its parent manually, so it must establish that same boundary.
        effects.bind_reader(self.root.agent_id, runtime_effect_reader("claude"))
        self.managed = VNextManagedSession(
            control=self.control,
            adapter=self.adapter,
            session_id=self.root.session_id,
            runtime_effects=effects,
        )
        self.managed.bind_thread(
            agent_id=self.root.agent_id,
            thread_id=RESERVATION,
            start_result={"posture": self._posture()},
            tool_handler=lambda *_args: {"success": False, "value": {}},
        )
        self.scheduler = VNextScheduler(
            managed=self.managed,
            root=self.root,
            cancellation=RunCancellation(),
        )
        self.adapter.set_native_child_observer(
            self.scheduler.adopt_native_child,
            parent_agent_resolver=self.managed.agent_for_thread,
        )

    def tearDown(self) -> None:
        self.managed.close()
        self.temp.cleanup()

    @staticmethod
    def _posture() -> dict[str, object]:
        return {
            "workspace_writes": True,
            "network": "approval_gated",
            "approvals_requested": True,
            "reviewer": "auto_review",
            "environment_ready": True,
        }

    def test_long_agent_description_denial_explains_the_limit(self) -> None:
        bridge = _Bridge()
        sdk = _FakeSDK()
        bridge._sdk = sdk
        bridge._workspace = self.workspace
        state = bridge._reservation_state("auto_review", None, [DELEGATE])
        state["turn_reference"] = "root-turn"
        state["generation"] = 1
        bridge._reservations[RESERVATION] = state
        bridge._options(model=MODEL, resume=None, reservation_id=RESERVATION, definitions=[DELEGATE])
        pretool = sdk.options[-1]["hooks"]["PreToolUse"][0].hooks[0]
        decision = asyncio.run(pretool({
            "tool_name": "Agent",
            "session_id": SESSION,
            "tool_input": {"prompt": "SECRET_SENTINEL", "description": "x" * 257},
        }, PARENT_TOOL, None))
        output = decision["hookSpecificOutput"]
        self.assertEqual("deny", output["permissionDecision"])
        self.assertIn("description", output["permissionDecisionReason"])
        self.assertIn("256", output["permissionDecisionReason"])
        self.assertNotIn("SECRET_SENTINEL", output["permissionDecisionReason"])

    def test_blank_agent_model_denial_names_the_shape(self) -> None:
        bridge = _Bridge()
        sdk = _FakeSDK()
        bridge._sdk = sdk
        bridge._workspace = self.workspace
        state = bridge._reservation_state("auto_review", None, [DELEGATE])
        state["turn_reference"] = "root-turn"
        state["generation"] = 1
        bridge._reservations[RESERVATION] = state
        bridge._options(model=MODEL, resume=None, reservation_id=RESERVATION, definitions=[DELEGATE])
        pretool = sdk.options[-1]["hooks"]["PreToolUse"][0].hooks[0]
        decision = asyncio.run(pretool({
            "tool_name": "Agent", "session_id": SESSION,
            "tool_input": {"prompt": "SECRET_SENTINEL", "model": " "},
        }, PARENT_TOOL, None))
        reason = decision["hookSpecificOutput"]["permissionDecisionReason"]
        self.assertIn("non-empty model name", reason)
        self.assertNotIn("SECRET_SENTINEL", reason)

    def test_an_agent_launch_vnext_cannot_track_is_refused(self) -> None:
        """A PreToolUse record missing its identity used to let the child run.

        The hook answered {"continue": true} and left a counter behind, while
        `relay_permission` allowed Agent on its own, so the child ran with no
        recorded origin: `cancel_agent` and parentage had no target for it.
        Reachability: the SDK passes `request_data.get("input")` and
        `request_data.get("tool_use_id")` straight to the callback with no
        validation (claude_agent_sdk/_internal/query.py:541-543), and declares
        `tool_use_id: str | None` (types.py:2399), so an incomplete record is
        this boundary's to refuse.
        """

        bridge = _Bridge()
        sdk = _FakeSDK()
        bridge._sdk = sdk
        bridge._workspace = self.workspace
        state = bridge._reservation_state("auto_review", None, [DELEGATE])
        state["turn_reference"] = "root-turn"
        state["generation"] = 1
        bridge._reservations[RESERVATION] = state
        bridge._options(
            model=MODEL, resume=None, reservation_id=RESERVATION, definitions=[DELEGATE]
        )
        options = sdk.options[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        launch = {
            "prompt": "Work this child would have done untracked.",
            "description": "Untrackable child objective",
        }

        for record, tool_use_id, missing in (
            ({"tool_name": "Agent", "tool_input": launch}, PARENT_TOOL, "session"),
            (
                {"tool_name": "Agent", "session_id": SESSION, "tool_input": launch},
                None,
                "tool-use id",
            ),
            (
                {"tool_name": "Agent", "session_id": SESSION, "tool_input": "a string"},
                PARENT_TOOL,
                "tool input",
            ),
            # An empty or blank identity names nothing, so it is as untracked
            # as an absent one and earns the same audit event.
            (
                {"tool_name": "Agent", "session_id": "", "tool_input": launch},
                PARENT_TOOL,
                "empty session",
            ),
            (
                {"tool_name": "Agent", "session_id": SESSION, "tool_input": launch},
                "",
                "empty tool-use id",
            ),
            (
                {"tool_name": "Agent", "session_id": "   ", "tool_input": launch},
                PARENT_TOOL,
                "blank session",
            ),
            (
                {"tool_name": "Agent", "session_id": SESSION, "tool_input": launch},
                "\t",
                "blank tool-use id",
            ),
        ):
            with self.subTest(missing=missing):
                emitted: list[dict[str, object]] = []
                with patch(
                    "vnext.vnext_claude_bridge._write",
                    side_effect=emitted.append,
                ):
                    decision = asyncio.run(pretool(record, tool_use_id, None))

                hook_output = decision["hookSpecificOutput"]
                self.assertEqual("PreToolUse", hook_output["hookEventName"])
                self.assertEqual("deny", hook_output["permissionDecision"])
                self.assertIn(
                    "vNext could not track this native child",
                    hook_output["permissionDecisionReason"],
                )
                blocked = next(
                    value["event"] for value in emitted
                    if value.get("kind") == "event"
                    and value["event"].get("name") == "native_child"
                )
                self.assertEqual("blocked-untracked", blocked["status"])
                self.assertEqual("unavailable", blocked["tracking"])
                self.assertEqual("Agent", blocked["tool"])
                self.assertEqual(
                    {}, state["native_child_origins"],
                    "an untracked launch must leave no origin behind",
                )

    def test_no_instruction_child_reaches_adapter_scheduler_and_scoped_tool_handler(self) -> None:
        bridge = _Bridge()
        sdk = _FakeSDK()
        bridge._sdk = sdk
        bridge._workspace = self.workspace
        state = bridge._reservation_state("auto_review", None, [DELEGATE])
        state["turn_reference"] = "root-turn"
        state["generation"] = 1
        bridge._reservations[RESERVATION] = state
        bridge._options(model=MODEL, resume=None, reservation_id=RESERVATION, definitions=[DELEGATE])
        options = sdk.options[-1]
        pretool = options["hooks"]["PreToolUse"][0].hooks[0]
        subagent_start = options["hooks"]["SubagentStart"][0].hooks[0]

        class Metadata:
            parent_tool_use_id = PARENT_TOOL
            parent_agent_id = None

        class TaskStartedMessage:
            task_id = TASK
            tool_use_id = PARENT_TOOL
            session_id = SESSION

        class TaskNotificationMessage:
            task_id = TASK
            session_id = SESSION
            status = "completed"
            summary = "automatic child completed"
            usage = {"total_tokens": 3}

        bridge._sdk.get_subagent_messages = lambda *_args, **_kwargs: [Metadata()]
        emitted: list[dict[str, object]] = []

        def send_control(record: dict[str, object]) -> None:
            self.assertEqual("tool_call_response", record["op"])
            pending = state["pending_tool_calls"]
            answer = pending[record["call_id"]]
            answer.set_result(dict(record["result"]))

        self.adapter._send = send_control  # type: ignore[method-assign]

        def deliver(record: dict[str, object]) -> None:
            emitted.append(record)
            if record.get("kind") == "event":
                self.adapter._record_event(record)

        with patch("vnext.vnext_claude_bridge._write", side_effect=deliver):
            decision = asyncio.run(pretool({
                "tool_name": "Agent",
                "session_id": SESSION,
                "tool_input": {
                    "prompt": "This text must not be required for adoption.",
                    "description": "Bounded automatic child objective",
                    "model": MODEL,
                },
            }, PARENT_TOOL, None))
            self.assertEqual({"continue": True}, decision)
            bridge._emit_message(RESERVATION, 1, TaskStartedMessage())
            asyncio.run(subagent_start({"session_id": SESSION, "agent_id": CHILD}, None, None))

            child_event = next(
                record["event"] for record in emitted
                if record.get("kind") == "event"
                and record["event"].get("name") == "native_child"
                and record["event"].get("tracking") == "attested"
            )
            self.assertEqual(TASK, child_event["turn_reference"])
            self.assertEqual("Bounded automatic child objective", child_event["task_contract"]["objective"])

            scoped = asyncio.run(pretool({
                "tool_name": "mcp__vnext__delegate",
                "session_id": SESSION,
                "agent_id": CHILD,
                "tool_input": {
                    "role": "worker", "model_id": MODEL,
                    "objective": "Nested task from the native child",
                    "task_contract": {}, "workspace": "shared", "effort": "high",
                },
            }, "child-tool-call", None))
            updated = scoped["hookSpecificOutput"]["updatedInput"]
            self.assertIn("_vnext_native_child_context", updated)
            hosted = next(tool for tool in options["mcp_servers"]["vnext"].tools if tool.name == "delegate")
            # A bridge/control correlation regression must fail this fixture
            # promptly instead of holding the complete suite indefinitely.
            reply = asyncio.run(asyncio.wait_for(hosted.handler(updated), timeout=2.0))
            self.assertFalse(reply["is_error"])

            bridge._emit_message(RESERVATION, 1, TaskNotificationMessage())
            bridge._emit_message(RESERVATION, 1, TaskNotificationMessage())

        native_binding = self.adapter.native_child_attestations()
        self.assertEqual(1, len(native_binding))
        self.assertEqual(TASK, native_binding[0]["task_id"])
        self.assertEqual("completed", native_binding[0]["status"])
        native_agent_id = native_binding[0]["bound_agent_id"]
        self.assertIsInstance(native_agent_id, str)
        native_agent = self.control.sessions[self.root.session_id].agents[native_agent_id]
        self.assertEqual(self.root.agent_id, native_agent.parent_agent_id)
        self.assertEqual("Bounded automatic child objective", native_agent.objective)
        self.assertEqual(1, len(native_agent.child_ids), "child tool call must route through the adopted child")
        delegated = self.control.sessions[self.root.session_id].agents[native_agent.child_ids[0]]
        self.assertEqual(native_agent.agent_id, delegated.parent_agent_id)
        self.assertEqual(1, len(self.adapter._turns))
        self.assertEqual(1, sum(
            record.get("kind") == "event"
            and record["event"].get("name") == "native_child_completed"
            for record in emitted
        ))
        self.assertEqual("completed", self.adapter._native_child_tasks["claude-native:" + SESSION + ":" + CHILD]["status"])
        self.assertEqual(
            {"agent_id": delegated.agent_id, "role": "worker", "model_id": MODEL,
             "model_exact": "pending: resolved when the worker connects",
             "evidence_request_recorded": False,
             # A shared child works in the session workspace itself, and the
             # answer names that directory rather than an empty string.
             "workspace_path": str(self.workspace.resolve())},
            json.loads(reply["content"][0]["text"]),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
