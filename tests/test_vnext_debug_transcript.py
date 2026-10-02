from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vnext.vnext_runtime_types import TurnHandle
from vnext.vnext_debug_transcript import ScopedDebugEvent, VNextDebugTranscript


class VNextDebugTranscriptTests(unittest.TestCase):
    def test_records_one_histogram_and_raw_item_events_per_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcript = VNextDebugTranscript("run-1", root=temp)
            codex_transcript = transcript.for_provider("codex")
            handle = TurnHandle("thread-1", "turn-1", 0)
            events = [
                ScopedDebugEvent(
                    {
                    "method": "item/started",
                    "params": {
                        "item": {"type": "reasoning", "text": "raw model content"},
                    },
                    },
                    is_current_turn=True,
                ),
                ScopedDebugEvent(
                    {
                    "method": "item/completed",
                    "params": {
                        "item": {"type": "reasoning", "text": "raw model content"},
                    },
                    },
                    is_current_turn=True,
                ),
                ScopedDebugEvent(
                    {
                    "method": "item/completed",
                    "params": {
                        "item": {"type": "agentMessage", "text": "worker report"},
                    },
                    },
                    is_current_turn=True,
                ),
            ]

            codex_transcript.record_turn(handle, events)
            codex_transcript.record_turn(handle, events)

            rows = [
                json.loads(line)
                for line in transcript.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("turn_summary", rows[0]["kind"])
            self.assertEqual(
                {"agentMessage": 1, "reasoning": 1},
                rows[0]["item_completed_type_histogram"],
            )
            self.assertEqual(4, len(rows))
            self.assertTrue(all(row["run_id"] == "run-1" for row in rows))
            self.assertEqual(
                ["item/started", "item/completed", "item/completed"],
                [row["event"]["method"] for row in rows[1:]],
            )

    def test_histogram_mode_can_omit_raw_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcript = VNextDebugTranscript(
                "run-summary-only",
                root=temp,
                include_raw_items=False,
            )
            transcript.for_provider("codex").record_turn(
                TurnHandle("thread", "turn", 0),
                [
                    ScopedDebugEvent(
                        {
                        "method": "item/completed",
                        "params": {
                            "item": {"type": "toolCall"},
                        },
                        },
                        is_current_turn=True,
                    )
                ],
            )

            rows = transcript.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(rows))
            self.assertEqual(
                {"toolCall": 1},
                json.loads(rows[0])["item_completed_type_histogram"],
            )

    def test_codex_reader_excludes_explicitly_unscoped_sibling_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcript = VNextDebugTranscript("run-scoped", root=temp)
            transcript.for_provider("codex").record_turn(
                TurnHandle("thread", "turn", 0),
                [
                    ScopedDebugEvent(
                        {
                            "method": "item/completed",
                            "params": {"item": {"type": "agentMessage", "text": "current"}},
                        },
                        is_current_turn=True,
                    ),
                    ScopedDebugEvent(
                        {
                            "method": "item/completed",
                            "params": {"item": {"type": "agentMessage", "text": "sibling"}},
                        },
                        is_current_turn=False,
                    ),
                ],
            )

            rows = [
                json.loads(line)
                for line in transcript.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({"agentMessage": 1}, rows[0]["item_completed_type_histogram"])
            self.assertEqual(2, len(rows))
            self.assertEqual("current", rows[1]["event"]["params"]["item"]["text"])

    def test_codex_reader_rejects_raw_unscoped_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcript = VNextDebugTranscript("run-unscoped", root=temp)
            transcript.for_provider("codex").record_turn(
                TurnHandle("thread", "turn", 0),
                [
                    {
                        "method": "item/completed",
                        "params": {"item": {"type": "agentMessage", "text": "untrusted"}},
                    }
                ],
            )

            rows = [
                json.loads(line)
                for line in transcript.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(1, len(rows))
            self.assertEqual({}, rows[0]["item_completed_type_histogram"])

    def test_records_raw_native_approval_only_in_local_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcript = VNextDebugTranscript("run-approval", root=temp)
            transcript.for_provider("codex").record_native_approval(
                "approval/request",
                {
                    "provider": "codex",
                    "provider_correlation": {
                        "session": "thread",
                        "turn": "turn",
                        "request": "item",
                    },
                    "command": "sensitive raw command",
                },
                {"decision": "decline"},
            )

            row = json.loads(transcript.path.read_text(encoding="utf-8"))
            self.assertEqual("native_approval", row["kind"])
            self.assertEqual("decline", row["decision"])
            self.assertEqual("sensitive raw command", row["params"]["command"])
            self.assertTrue(row["thread_correlated"])
            self.assertTrue(row["turn_correlated"])
            self.assertTrue(row["item_correlated"])

    def test_claude_reader_decodes_bridge_events_without_codex_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcript = VNextDebugTranscript("run-claude", root=temp)
            transcript.for_provider("claude").record_turn(
                TurnHandle("reservation-1", "turn-ref-1", 0),
                [
                    {
                        "name": "tool_result",
                        "reservation_id": "reservation-1",
                        "turn_reference": "turn-ref-1",
                    },
                    {
                        "name": "tool_result",
                        "reservation_id": "other-reservation",
                        "turn_reference": "turn-ref-1",
                    },
                ],
            )

            rows = [
                json.loads(line)
                for line in transcript.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({"tool_result": 1}, rows[0]["item_completed_type_histogram"])
            self.assertEqual(2, len(rows))


if __name__ == "__main__":
    unittest.main()
