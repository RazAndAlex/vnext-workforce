"""Provider-free adapter checks for exact nested identity and task control."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from vnext.vnext_claude import ClaudeCodeAdapter
from vnext.vnext_claude_bridge import _Bridge
from vnext.vnext_runtime_types import NativeChildBinding


class NativeLineageTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeCodeAdapter(workspace=str(Path.cwd()))
        self.adapter._threads["reservation"] = {
            "provider_session": "session", "generation": 1,
            "model": "claude-model", "effort": "high",
        }
        self.parents = {"reservation": "primary"}
        self.observed = []

        def observe(value):
            self.observed.append(value)
            agent = "agent-" + value.native_child_id
            self.parents[value.native_child_thread_id] = agent
            return NativeChildBinding(agent, "claude", value.native_child_thread_id,
                                      value.delivery_contract)

        self.adapter.set_native_child_observer(observe, parent_agent_resolver=self.parents.get)

    @staticmethod
    def event(child, parent=None):
        event = {
            "name": "native_child", "reservation_id": "reservation", "generation": 1,
            "turn_reference": "task-" + child, "task_id": "task-" + child,
            "parent_turn_reference": "local-turn",
            "provider_correlation": {"session": "session", "turn": "local-turn"},
            "correlation_attested": True, "tracking": "attested", "status": "running",
            "native_agent_id": child, "native_runtime_thread_id": "claude-native:session:" + child,
            "parent_tool_use_id": "tool-" + child, "parent_native_agent_id": parent,
            "task_contract": {"role": "worker", "role_source": "vnext-native-task-mapping",
                              "objective": "bounded task", "requested_model": "claude-model"},
        }
        if parent is not None:
            event.update(parent_runtime_thread_id="claude-native:session:" + parent,
                         parent_native_task_id="task-" + parent,
                         parent_native_turn_id="task-" + parent)
        return event

    def record(self, event):
        self.adapter._record_event({"event": event})

    def test_nested_child_uses_bound_parent_and_exact_stop_leaves_sibling_running(self):
        for event in (self.event("a"), self.event("b"), self.event("c", "a")):
            self.record(event)
        self.assertEqual(["primary", "primary", "agent-a"],
                         [item.parent_agent_id for item in self.observed])
        self.assertIsNone(self.observed[0].parent_native_turn_id)
        self.assertEqual("task-a", self.observed[2].parent_native_turn_id)
        identity = self.adapter.thread_identity_attestation("claude-native:session:c")
        self.assertEqual("claude-native:session:a", identity["parent_runtime_thread_id"])
        self.assertEqual("a", identity["parent_native_agent_id"])
        self.assertEqual("task-a", identity["parent_native_turn_id"])
        self.assertEqual("local-turn", identity["parent_local_turn_reference"])
        with patch.object(self.adapter, "stop_native_task") as stop:
            result = self.adapter.cancel_native_child("claude-native:session:c")
        stop.assert_called_once_with("reservation", "task-c")
        self.assertEqual("interrupt-requested", result["status"])
        self.record(dict(self.event("c", "a"), name="native_child_completed", status="stopped"))
        self.assertEqual("running", self.adapter._native_child_tasks["claude-native:session:b"]["status"])
        self.assertEqual("running", self.adapter._native_child_tasks["claude-native:session:a"]["status"])
        self.assertEqual("stopped", self.adapter._native_child_tasks["claude-native:session:c"]["status"])

    def test_bridge_stop_receipt_is_supported_without_completing_a_child(self):
        self.record(self.event("a"))
        self.record(self.event("b"))
        bridge = _Bridge()
        state = bridge._reservation_state("auto_review", "session", [])
        state.update(generation=1, turn_reference="local-turn",
                     native_children={"task-a": {"status": "running"}},
                     client=SimpleNamespace(stop_task=AsyncMock()))
        bridge._reservations["reservation"] = state
        with patch("vnext.vnext_claude_bridge._write", side_effect=self.adapter._record_event):
            result = asyncio.run(bridge._stop_native_task({"reservation_id": "reservation", "task_id": "task-a"}))
        self.assertTrue(result["accepted"])
        self.assertIsNone(self.adapter._fatal)
        self.assertEqual("native_child_control", self.adapter.events_since()[-1]["name"])
        self.record({
            "name": "native_child_control", "reservation_id": "reservation",
            "task_id": "other-task", "action": "stop-requested",
        })
        self.assertEqual("native_child_control_rejected", self.adapter.events_since()[-1]["name"])
        for child in ("a", "b"):
            self.assertEqual("running", self.adapter._native_child_tasks["claude-native:session:" + child]["status"])

    def test_parent_turn_readiness_defers_while_native_task_runs_then_probes_exactly(self):
        self.record(self.event("a"))
        with patch.object(self.adapter, "_request") as request:
            self.assertFalse(self.adapter.can_start_turn("reservation"))
            request.assert_not_called()
            self.adapter._native_child_tasks["claude-native:session:a"]["status"] = "completed"
            request.return_value = {
                "reservation_echo": "reservation", "generation": 1, "ready": False,
            }
            self.assertFalse(self.adapter.can_start_turn("reservation"))
            request.assert_called_once_with(
                "can_start_turn", {"reservation_id": "reservation", "generation": 1},
                timeout_seconds=2.0, read_only=True,
            )
            # A later retry is allowed only after the bounded negative cache;
            # simulate that expiry without sleeping in the test.
            self.adapter._turn_readiness.clear()
            request.return_value = {
                "reservation_echo": "reservation", "generation": 1, "ready": True,
            }
            self.assertTrue(self.adapter.can_start_turn("reservation"))

    def test_only_initial_claude_generation_bypasses_the_reader_probe(self):
        self.adapter._threads["reservation"]["generation"] = 0
        with patch.object(self.adapter, "_request") as request:
            self.assertTrue(self.adapter.can_start_turn("reservation"))
            request.assert_not_called()
            self.adapter._threads["reservation"]["generation"] = 1
            request.return_value = {
                "reservation_echo": "reservation", "generation": 1, "ready": False,
            }
            self.assertFalse(self.adapter.can_start_turn("reservation"))
            request.assert_called_once()

    def test_unbound_parent_is_not_flattened_and_can_be_replayed_after_parent(self):
        nested = self.event("c", "a")
        self.record(nested)
        self.assertEqual([], self.observed)
        self.assertEqual("native_child_rejected", self.adapter.events_since()[-1]["name"])
        self.record(self.event("a"))
        self.record(nested)
        self.record(nested)
        self.assertEqual(2, len(self.observed))
        self.assertEqual("agent-a", self.observed[1].parent_agent_id)

    def test_forged_runtime_key_self_parent_and_cross_session_are_refused(self):
        self.record(dict(self.event("a"), native_runtime_thread_id="arbitrary-key"))
        self.record(self.event("a", "a"))
        self.record(dict(self.event("a"), provider_correlation={"session": "other", "turn": "local-turn"}))
        self.assertEqual([], self.observed)

    def test_alias_requires_exact_observed_child_model(self):
        event = self.event("a")
        event["task_contract"]["requested_model"] = "opus"
        self.record(event)
        self.assertEqual([], self.observed)
        event["task_contract"].update(observed_model="claude-model", observed_model_source="guessed-alias")
        self.record(event)
        self.assertEqual([], self.observed)
        event["task_contract"]["observed_model_source"] = "saved-child-assistant-message"
        self.record(event)
        self.assertEqual("claude-model", self.observed[0].model_id)
        self.assertEqual("opus", self.observed[0].task_contract["requested_model"])
        forwarded = self.event("b")
        forwarded["task_contract"].update(
            requested_model="opus", observed_model="claude-model",
            observed_model_source="forwarded-child-assistant-message",
        )
        self.record(forwarded)
        self.assertEqual("claude-model", self.observed[1].model_id)

    def test_conflicting_parent_lifecycle_cannot_complete_child(self):
        self.record(self.event("a"))
        self.record(self.event("c", "a"))
        self.record(dict(self.event("c", "other"), name="native_child_completed", status="completed"))
        self.assertEqual("running", self.adapter._native_child_tasks["claude-native:session:c"]["status"])

    def test_authenticated_stop_observation_does_not_complete_or_fail_reader(self):
        self.record(self.event("a"))
        stop = dict(self.event("a"), name="native_child_stop", status="observed-stop", stop_observed=True)
        self.record(stop)
        self.assertEqual("native_child_stop", self.adapter.events_since()[-1]["name"])
        self.assertEqual("running", self.adapter._native_child_tasks["claude-native:session:a"]["status"])
        self.record(dict(stop, task_id="different-task"))
        self.assertEqual("native_child_stop_rejected", self.adapter.events_since()[-1]["name"])
        self.record(dict(self.event("a"), name="native_child_completed", status="completed"))
        self.assertEqual("completed", self.adapter._native_child_tasks["claude-native:session:a"]["status"])

    def test_nested_parent_task_and_runtime_evidence_must_match_binding(self):
        self.record(self.event("a"))
        for field in ("parent_runtime_thread_id", "parent_native_task_id", "parent_native_turn_id"):
            event = self.event("c", "a")
            self.record(dict(event, **{field: "other"}))
            del event[field]
            self.record(event)
        self.assertEqual(1, len(self.observed))


if __name__ == "__main__":
    unittest.main()


class NativeAdoptionRefusalTests(unittest.TestCase):
    def test_a_child_vnext_refuses_to_adopt_leaves_a_rejection_in_the_record(self):
        from vnext.vnext_orchestration import ProtocolError

        adapter = ClaudeCodeAdapter(workspace=str(Path.cwd()))
        adapter._threads["reservation"] = {
            "provider_session": "session", "generation": 1,
            "model": "claude-model", "effort": "high",
        }

        def refuse(_observation):
            raise ProtocolError("unknown-model", "unknown model: claude-opus-5")

        adapter.set_native_child_observer(refuse, parent_agent_resolver={"reservation": "primary"}.get)
        adapter._record_event({"event": NativeLineageTests.event("a")})
        last = adapter.events_since()[-1]
        self.assertEqual("native_child_rejected", last["name"])
        self.assertNotIn("native_runtime_thread_id", last)
        self.assertIn("ProtocolError", last["reason"])
