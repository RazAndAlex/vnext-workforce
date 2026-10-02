"""A native child whose saved transcript lands late still joins at the terminal.

Every field shape here follows one recorded live run of an ``opus`` vNext
worker that launched one Agent-tool subagent, and the identifiers are synthetic
values in the recorded formats: the Agent tool-use id, the task id
that equals the subagent's agent id, the session id, and the first saved child
message carrying ``parent_tool_use_id`` with ``parent_agent_id`` absent and a
``claude-opus-5`` assistant frame.  The one thing the fixture controls is when
that saved transcript becomes readable, because that is what the live record
could not settle.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.vnext_claude_bridge import _Bridge
from vnext.vnext_claude_native_identity import NativeChildAutomaticResolver


SESSION = "00000000-0000-4000-8000-000000000017"
PARENT_TOOL_USE = "toolu_01SyntheticAgentToolUse17"
TASK = "a00000000000000017"


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


class LateNativeChildMetadataTests(unittest.TestCase):
    """The join must survive a saved child transcript that is not there yet."""

    def _run(self, *, readable_from_the_start: bool, subagent_start_hook_fired: bool = True) -> list[dict]:
        written: list[dict] = []

        saved = [_SavedMessage("user", None), _SavedMessage("assistant", "claude-opus-5")]
        readable = {"now": readable_from_the_start}

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
                    # The real CLI writes the child transcript when the Agent
                    # tool answers, which is after its task completes and
                    # before the parent can report the answer.
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
                if subagent_start_hook_fired:
                    resolver.record_subagent_start(SESSION, TASK)
                state["native_child_automatic_resolver"] = resolver
                state["native_child_configured_model"] = "opus"
                state["native_child_metadata_directory"] = workspace
                state["native_child_hook_sessions"].add(SESSION)
                state["native_child_origins"][(SESSION, PARENT_TOOL_USE)] = {
                    "turn_reference": "turn-a", "generation": 1,
                    "parent_native_agent_id": None, "background_requested": None,
                }
                state["client"] = Client()
                state["turn_reference"] = "turn-a"
                bridge._reservations["reservation-a"] = state

                def record(payload):
                    written.append(json.loads(json.dumps(payload, default=str)))

                with patch("vnext.vnext_claude_bridge._write", side_effect=record):
                    await bridge.dispatch("start_turn", {
                        "reservation_id": "reservation-a", "turn_reference": "turn-a",
                        "generation": 1, "prompt": "launch one subagent",
                    })
                    await bridge.dispatch("wait_turn", {
                        "reservation_id": "reservation-a", "turn_reference": "turn-a",
                    })
                    while state["task"] is not None:
                        await asyncio.sleep(0)

        asyncio.run(exercise())
        return [record["event"] for record in written if record.get("kind") == "event"]

    def test_child_joins_when_its_saved_transcript_appears_only_at_the_parent_terminal(self) -> None:
        events = self._run(readable_from_the_start=False)
        names = [event.get("name") for event in events]
        self.assertIn("native_child_identity", names, f"no exact join was ever emitted: {names}")
        joined = next(event for event in events if event.get("name") == "native_child_identity")
        self.assertEqual(TASK, joined.get("task_id"))
        self.assertEqual(PARENT_TOOL_USE, joined.get("parent_tool_use_id"))

    def test_counters_reach_the_run_record_so_a_missed_read_is_readable(self) -> None:
        events = self._run(readable_from_the_start=False)
        counters = [event for event in events if event.get("name") == "native_child_counters"]
        self.assertTrue(counters, "the turn wrote no enrollment counter snapshot")
        fired = counters[-1]["counters"]
        self.assertGreaterEqual(fired.get("metadata_read_empty", 0), 1)
        self.assertGreaterEqual(fired.get("metadata_read_matched", 0), 1)
        self.assertEqual("reservation-a", counters[-1].get("reservation_id"))

    def test_child_joins_when_no_subagent_start_hook_ever_reached_the_bridge(self) -> None:
        events = self._run(readable_from_the_start=False, subagent_start_hook_fired=False)
        names = [event.get("name") for event in events]
        self.assertIn("native_child_identity", names, f"no exact join was ever emitted: {names}")

    def test_an_already_readable_transcript_still_joins_exactly_once(self) -> None:
        events = self._run(readable_from_the_start=True)
        joins = [event for event in events if event.get("name") == "native_child_identity"]
        self.assertEqual(1, len(joins))


if __name__ == "__main__":
    unittest.main()
