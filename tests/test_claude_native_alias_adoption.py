"""A live-shaped native child under an ``opus`` worker becomes a vNext agent.

The bridge half replays the shape of a recorded live run, with synthetic
identifiers in place of the recorded ones: an ``opus`` vNext worker launches one Agent-tool subagent, the CLI saves the child
transcript late, and the child's assistant frame names the canonical provider
model ``claude-opus-5``.  The adapter half feeds exactly those bridge events to
a real adapter, managed session and scheduler whose catalog names the worker
``opus``.  In the live run the attested ``native_child`` event was accepted,
the adoption then failed silently, and every event addressed to the child's
runtime thread was held forever, so no ``in_parent`` row could exist.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.vnext_claude import ClaudeCodeAdapter
from vnext.vnext_claude_bridge import _Bridge
from vnext.vnext_claude_native_identity import NativeChildAutomaticResolver
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


# Synthetic identifiers in the formats the live record used.
SESSION = "00000000-0000-4000-8000-000000000018"
PARENT_TOOL_USE = "toolu_01SyntheticAgentToolUse18"
TASK = "a00000000000000018"
RESERVATION = "claude-reservation-00000000000000000000000000000018"
TURN = "claude-turn-00000000000000000000000000000018"
CATALOG_MODEL = "opus"
PROVIDER_MODEL = "claude-opus-5"


class _SavedMessage:
    def __init__(self, role: str, model: str | None) -> None:
        self.parent_tool_use_id = PARENT_TOOL_USE
        self.parent_agent_id = None
        self.message = {"role": role, "model": model}


class TaskStartedMessage:
    tool_use_id, session_id, task_id = PARENT_TOOL_USE, SESSION, TASK
    usage = summary = status = None


class TaskUpdatedMessage:
    tool_use_id, session_id, task_id = PARENT_TOOL_USE, SESSION, TASK
    patch = {"status": "completed"}
    usage = summary = None


class ResultMessage:
    session_id, result = SESSION, "done"
    usage = total_cost_usd = model_usage = None
    duration_ms = duration_api_ms = num_turns = None
    is_error, stop_reason, subtype = False, None, "success"
    api_error_status, terminal_reason, errors = None, "end_turn", ()


def _live_bridge_events(provider_model: str = PROVIDER_MODEL) -> list[dict]:
    """Run the bridge through the live message order and return its events."""

    written: list[dict] = []
    saved = [_SavedMessage("user", None), _SavedMessage("assistant", provider_model)]
    readable = {"now": False}

    class FakeSDK:
        @staticmethod
        def list_subagents(session_id, *, directory):
            del directory
            return [TASK] if readable["now"] and session_id == SESSION else []

        @staticmethod
        def get_subagent_messages(session_id, agent_id, *, directory, limit):
            del directory, limit
            if not readable["now"] or session_id != SESSION or agent_id != TASK:
                return []
            return saved

    async def exercise() -> None:
        class Client:
            async def query(self, prompt: str) -> None:
                pass

            async def receive_messages(self):
                yield TaskStartedMessage()
                yield TaskUpdatedMessage()
                readable["now"] = True
                yield ResultMessage()

        with tempfile.TemporaryDirectory() as workspace:
            bridge = _Bridge()
            bridge._workspace, bridge._sdk = Path(workspace), FakeSDK()
            state = bridge._reservation_state("auto_review", SESSION, [])
            resolver = NativeChildAutomaticResolver()
            resolver.record_agent_origin(
                SESSION, PARENT_TOOL_USE, None, parent_agent_id_present=True,
                tool_input={"description": "read hello.txt", "subagent_type": "worker-opus-low"},
            )
            resolver.record_subagent_start(SESSION, TASK)
            state["native_child_automatic_resolver"] = resolver
            state["native_child_configured_model"] = CATALOG_MODEL
            state["native_child_metadata_directory"] = workspace
            state["native_child_hook_sessions"].add(SESSION)
            state["native_child_origins"][(SESSION, PARENT_TOOL_USE)] = {
                "turn_reference": TURN, "generation": 1,
                "parent_native_agent_id": None, "background_requested": None,
            }
            state["client"] = Client()
            state["turn_reference"] = TURN
            bridge._reservations[RESERVATION] = state

            def record(payload):
                written.append(json.loads(json.dumps(payload, default=str)))

            with patch("vnext.vnext_claude_bridge._write", side_effect=record):
                await bridge.dispatch("start_turn", {
                    "reservation_id": RESERVATION, "turn_reference": TURN,
                    "generation": 1, "prompt": "launch one subagent",
                })
                await bridge.dispatch("wait_turn", {"reservation_id": RESERVATION, "turn_reference": TURN})
                while state["task"] is not None:
                    await asyncio.sleep(0)

    asyncio.run(exercise())
    return [record for record in written if record.get("kind") == "event"]


class AliasWorkerNativeChildAdoptionTests(unittest.TestCase):
    catalog: tuple[str, ...] = (CATALOG_MODEL,)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        posture = {
            "workspace_writes": True, "network": "approval_gated", "approvals_requested": True,
            "reviewer": "auto_review", "environment_ready": True,
        }
        registry = ModelRegistry(
            cards=[
                ModelCard(model, frozenset({AgentRole.ROOT_MANAGER, AgentRole.WORKER}), provider="claude")
                for model in self.catalog
            ],
            presets=[EconomicPreset("fixture", frozenset(self.catalog), frozenset({"claude"}))],
        )
        self.control = OrchestrationControlPlane(registry)
        self.root = self.control.create_session(
            preset_id="fixture", root_model_id=CATALOG_MODEL, workspace=self.workspace,
            objective="opus worker launches one Agent subagent",
            task_contract={"criteria": ["the subagent is a vNext agent"]},
            session_id="alias-adoption",
        )
        self.adapter = ClaudeCodeAdapter(workspace=str(self.workspace))
        self.adapter._threads[RESERVATION] = {
            "policy": {"posture": posture}, "provider_session": SESSION, "binding_phase": "attested",
            "generation": 1, "model": CATALOG_MODEL, "effort": "low",
            "permission_mode": "default", "active_turn": None,
        }
        effects = RuntimeEffectJournal(self.workspace)
        effects.bind_reader(self.root.agent_id, runtime_effect_reader("claude"))
        self.managed = VNextManagedSession(
            control=self.control, adapter=self.adapter,
            session_id=self.root.session_id, runtime_effects=effects,
        )
        self.managed.bind_thread(
            agent_id=self.root.agent_id, thread_id=RESERVATION, start_result={"posture": posture},
            tool_handler=lambda *_args: {"success": False, "value": {}},
        )
        self.scheduler = VNextScheduler(managed=self.managed, root=self.root, cancellation=RunCancellation())
        self.adapter.set_native_child_observer(
            self.scheduler.adopt_native_child, parent_agent_resolver=self.managed.agent_for_thread,
        )

    def tearDown(self) -> None:
        self.managed.close()
        self.temp.cleanup()

    def test_live_shaped_child_under_an_alias_worker_is_adopted_and_routable(self) -> None:
        records = _live_bridge_events()
        running = [
            r["event"] for r in records
            if r["event"].get("name") == "native_child" and r["event"].get("tracking") == "attested"
        ]
        self.assertEqual(1, len(running), "the bridge emitted no attested running child")
        self.assertEqual("running", running[0]["status"])
        self.assertEqual(PROVIDER_MODEL, running[0]["task_contract"].get("observed_model"))
        for record in records:
            self.adapter._record_event(record)

        thread = "claude-native:" + SESSION + ":" + TASK
        bound = self.managed.agent_for_thread(thread)
        self.assertIsNotNone(bound, "the child was never adopted, so its events are held forever")
        child = self.control.sessions[self.root.session_id].agents[bound]
        self.assertEqual(self.root.agent_id, child.parent_agent_id)
        # The vNext node runs on the catalog entry that admitted it; the
        # provider's canonical answer stays in the attested task contract.
        self.assertEqual(CATALOG_MODEL, child.model_id)
        self.assertEqual(PROVIDER_MODEL, child.task_contract.get("observed_model"))
        self.assertEqual("completed", self.adapter._native_child_tasks[thread]["status"])
        names = [e.get("name") for e in self.adapter._events]
        self.assertNotIn("native_child_rejected", names)
        self.assertNotIn("native_child_completed_rejected", names)


    def test_a_turn_writes_its_counters_snapshot_once(self) -> None:
        names = [r["event"].get("name") for r in _live_bridge_events()]
        self.assertEqual(1, names.count("native_child_counters"), names)

    def test_a_child_of_a_family_the_catalog_lacks_is_refused_with_a_reason(self) -> None:
        """An ``opus`` worker whose subagent answers on Sonnet is not an Opus child."""

        for record in _live_bridge_events("claude-sonnet-4"):
            self.adapter._record_event(record)

        thread = "claude-native:" + SESSION + ":" + TASK
        self.assertIsNone(self.managed.agent_for_thread(thread))
        agents = self.control.sessions[self.root.session_id].agents.values()
        self.assertEqual([self.root.agent_id], [agent.agent_id for agent in agents])
        rejected = [e for e in self.adapter._events if e.get("name") == "native_child_rejected"]
        self.assertEqual(1, len(rejected), [e.get("name") for e in self.adapter._events])
        self.assertIn("sonnet", rejected[0].get("reason", ""))

    def _refusals_after(self, change) -> list[dict]:
        for record in _live_bridge_events():
            event = record["event"]
            if event.get("name") == "native_child" and event.get("tracking") == "attested":
                record = {**record, "event": change(json.loads(json.dumps(event)))}
            self.adapter._record_event(record)
        return [e for e in self.adapter._events if e.get("name") == "native_child_rejected"]

    def test_every_refused_child_carries_a_reason(self) -> None:
        """R18: a refusal with no reason left the operator guessing why."""

        def unknown_source(event):
            event["task_contract"]["observed_model_source"] = "a-source-nobody-attests"
            return event

        rejected = self._refusals_after(unknown_source)
        self.assertEqual(1, len(rejected))
        self.assertIn("a-source-nobody-attests", rejected[0].get("reason", ""))

    def test_a_child_with_no_objective_is_refused_for_that(self) -> None:
        """R19: an empty objective was blamed on identity or lifecycle."""

        def no_objective(event):
            event["task_contract"]["objective"] = ""
            return event

        rejected = self._refusals_after(no_objective)
        self.assertEqual(1, len(rejected))
        self.assertIn("objective", rejected[0].get("reason", ""))

    def test_a_child_whose_objective_is_blank_is_refused_for_that(self) -> None:
        """R20 gpt-code F5: whitespace passed as an objective."""

        def blank_objective(event):
            event["task_contract"]["objective"] = " \n  "
            return event

        rejected = self._refusals_after(blank_objective)
        self.assertEqual(1, len(rejected))
        self.assertIn("objective", rejected[0].get("reason", ""))

    def test_a_refusal_with_no_specific_cause_still_says_it_was_refused(self) -> None:
        def other_turn(event):
            event["turn_reference"] = "a-task-this-child-is-not"
            return event

        rejected = self._refusals_after(other_turn)
        self.assertEqual(1, len(rejected))
        self.assertTrue(rejected[0].get("reason"), rejected[0])


class ChildOnAnotherCatalogFamilyTests(AliasWorkerNativeChildAdoptionTests):
    """With Sonnet in the catalog, a Sonnet subagent is named for what it ran on."""

    catalog = (CATALOG_MODEL, "sonnet")

    def test_the_child_is_adopted_under_its_own_family(self) -> None:
        for record in _live_bridge_events("claude-sonnet-4"):
            self.adapter._record_event(record)

        thread = "claude-native:" + SESSION + ":" + TASK
        bound = self.managed.agent_for_thread(thread)
        self.assertIsNotNone(bound)
        child = self.control.sessions[self.root.session_id].agents[bound]
        self.assertEqual("sonnet", child.model_id)
        self.assertEqual("claude-sonnet-4", child.task_contract.get("observed_model"))

    def test_a_child_of_a_family_the_catalog_lacks_is_refused_with_a_reason(self) -> None:
        self.skipTest("this catalog has the Sonnet family")


if __name__ == "__main__":
    unittest.main()
