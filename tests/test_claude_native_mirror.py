import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vnext.vnext_claude_bridge import _Bridge
from vnext.vnext_claude_native_mirror import NativeMetadataMirror
from vnext.vnext_claude_native_identity import NativeChildIdentity


class MirrorTests(unittest.TestCase):
    def test_nested_identity_reports_timing_without_losing_exact_parentage(self):
        bridge = _Bridge()
        state = bridge._reservation_state("auto_review", "session", [])
        state["native_child_origins"][("session", "tool")] = {"generation": 1, "turn_reference": "primary-turn", "background_requested": True}
        identity = NativeChildIdentity("session", "tool", "grandchild", "grandchild-task",
            parent_agent_id="child", parent_task_id="child-task", identity_source="saved_session_metadata")
        def event():
            return bridge._native_identity_event(state, identity, reservation_id="reservation", source="fixture", status="running")
        before = event()
        self.assertEqual("before-terminal-observation", before["identity_resolution"])
        self.assertEqual("child", before["parent_native_agent_id"])
        self.assertEqual("claude-native:session:child", before["parent_runtime_thread_id"])
        self.assertEqual("child-task", before["parent_native_task_id"])
        self.assertIs(True, before["task_contract"]["background_requested"])
        state["native_child_terminal_observed"].add(("session", "grandchild-task"))
        self.assertEqual("after-terminal-observation", event()["identity_resolution"])

    def test_metadata_conflicts_in_same_batch_are_sticky_and_never_last_write_wins(self):
        class Store:
            async def append(self, key, entries):
                pass
        seen = []
        async def inspect(key):
            seen.append(mirror.metadata_for_agent("session", "child"))
        mirror = NativeMetadataMirror(SimpleNamespace(InMemorySessionStore=Store), inspect)
        first = {"type": "agent_metadata", "toolUseId": "first", "model": "full-model"}
        conflicting = {**first, "toolUseId": "second"}
        key = {"session_id": "session", "subpath": "subagents/agent-child"}
        asyncio.run(mirror.append(key, [first, conflicting]))
        asyncio.run(mirror.append(key, [first]))
        self.assertEqual([None, None], seen)
        self.assertTrue(mirror.is_conflicted("session", "child"))
        self.assertEqual(1, len(mirror.conflicted))

    def test_observation_failure_does_not_fail_sdk_append_or_hide_cancellation(self):
        class Store:
            async def append(self, key, entries):
                self.entries = entries
        async def failure(key):
            raise RuntimeError("private diagnostic must not be retained")
        mirror = NativeMetadataMirror(SimpleNamespace(InMemorySessionStore=Store), failure)
        entries = [{"type": "fixture"}]
        asyncio.run(mirror.append({"session_id": "session"}, entries))
        self.assertIs(mirror.store.entries, entries)
        self.assertEqual(1, mirror.callback_errors)
        async def cancelled(key):
            raise asyncio.CancelledError()
        mirror.on_append = cancelled
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(mirror.append({"session_id": "session"}, entries))
        self.assertEqual(1, mirror.callback_errors)

    def test_mirror_is_opt_in_and_never_changes_resume_storage(self):
        bridge = _Bridge()
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual({}, bridge._native_mirror_options({}, "reservation", None, True))
        with patch.dict("os.environ", {"VNEXT_CLAUDE_NATIVE_MIRROR": "1"}):
            self.assertEqual({}, bridge._native_mirror_options({}, "reservation", "session", True))
            self.assertEqual({}, bridge._native_mirror_options({}, "reservation", None, False))

    def test_append_keeps_sdk_entries_and_joins_only_exact_known_child(self):
        class Store:
            async def append(self, key, entries):
                self.key, self.entries = key, entries
            async def load(self, key):
                return self.entries
        calls = []
        async def read(store, session, agent, **kwargs):
            calls.append((session, agent))
            return []
        bridge = _Bridge()
        bridge._sdk = SimpleNamespace(InMemorySessionStore=Store, get_subagent_messages_from_store=read)
        state = bridge._reservation_state("auto_review", "session", [])
        state.update(generation=1, turn_reference="root-turn", native_child_configured_model="full-model",
                     native_child_metadata_directory=str(Path.cwd()))
        state["native_child_origins"][("session", "tool")] = {"generation": 1, "turn_reference": "root-turn"}
        resolver = state["native_child_automatic_resolver"]
        resolver.record_agent_origin("session", "tool", None, parent_agent_id_present=True,
                                     tool_input={"model": "opus", "description": "bounded child"})
        resolver.record_subagent_start("session", "child")
        resolver.record_task_started("session", "tool", "task")
        state["native_child_pending_lifecycle"][("session", "task")] = {"start": {"source": "TaskStartedMessage"}}
        with patch.dict("os.environ", {"VNEXT_CLAUDE_NATIVE_MIRROR": "1"}):
            options = bridge._native_mirror_options(state, "reservation", None, True)
        self.assertEqual("eager", options["session_store_flush"])
        mirror = options["session_store"]
        entries = [{"type": "agent_metadata", "toolUseId": "tool", "parentAgentId": None, "model": "full-model"}]
        written = []
        with patch("vnext.vnext_claude_bridge._write", side_effect=written.append):
            asyncio.run(mirror.append({"session_id": "foreign", "subpath": "subagents/agent-child"}, entries))
            self.assertEqual([], calls)
            asyncio.run(mirror.append({"session_id": "session", "subpath": "subagents/agent-child"}, entries))
        self.assertEqual([("session", "child")], calls)
        self.assertIs(entries, asyncio.run(mirror.load({"session_id": "session"})))
        running = next(item["event"] for item in written if item["event"].get("status") == "running")
        self.assertEqual("before-terminal-observation", running["identity_resolution"])
        self.assertEqual("full-model", running["task_contract"]["observed_model"])
        self.assertEqual("sdk-agent-metadata", running["task_contract"]["observed_model_source"])
        self.assertEqual({}, state["native_child_pending_lifecycle"])


if __name__ == "__main__":
    unittest.main()
