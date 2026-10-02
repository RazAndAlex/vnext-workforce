from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vnext.vnext_runtime_effects import (
    ClaudeRuntimeEffectReader,
    CodexRuntimeEffectReader,
    CommandCodeRuntimeEffectReader,
    RuntimeEffectJournal,
    RuntimeEffectProjectionError,
    effect_reader_hint,
    runtime_effect_reader,
)


def completed(thread_id: str, turn_id: str, item: dict) -> dict:
    return {
        "method": "item/completed",
        "params": {"threadId": thread_id, "turnId": turn_id, "item": item},
    }


OMIT = object()


def claude_result(
    reservation_id: str,
    turn_reference: str,
    tool_use_id: str,
    effect_type: str,
    *,
    status: object = "completed",
    changes: list[dict] | None = None,
    evidence_limited: bool | None = None,
) -> dict:
    record = {
        "name": "tool_result",
        "reservation_id": reservation_id,
        "turn_reference": turn_reference,
        "tool_use_id": tool_use_id,
        "effect_type": effect_type,
        "provider_correlation": {
            "session": "native-session",
            "turn": turn_reference,
            "request": tool_use_id,
        },
        "correlation_attested": True,
    }
    if status is not OMIT:
        record["status"] = status
    if changes is not None:
        record["changes"] = changes
    if evidence_limited is not None:
        record["evidence_limited"] = evidence_limited
    return record


class RuntimeEffectJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        (self.workspace / "sub").mkdir()
        self.journal = RuntimeEffectJournal(self.workspace)
        self.journal.bind_reader("worker", CodexRuntimeEffectReader())
        self.journal.bind_reader("worker-secret", CodexRuntimeEffectReader())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_native_child_inherits_exact_parent_decoder_with_scoped_effects(self) -> None:
        self.journal.bind_reader("claude-parent", ClaudeRuntimeEffectReader())
        self.journal.inherit_reader(parent_agent_id="claude-parent", child_agent_id="child")
        event = claude_result("child-thread", "child-task", "tool", "command")
        effects = self.journal.observe_turn(agent_id="child", thread_id="child-thread",
            turn_id="child-task", events=[event])
        self.assertEqual(1, len(effects))
        self.assertEqual("claude", effects[0].provider)
        self.assertEqual((), self.journal.observe_turn(agent_id="child", thread_id="foreign-thread",
            turn_id="child-task", events=[event]))
        with self.assertRaises(RuntimeEffectProjectionError):
            self.journal.inherit_reader(parent_agent_id="worker", child_agent_id="child")
        with self.assertRaises(RuntimeEffectProjectionError):
            self.journal.inherit_reader(parent_agent_id="missing", child_agent_id="unbound")

    def test_projects_command_facts_without_command_or_output_content(self) -> None:
        event = completed(
            "thread-secret",
            "turn-secret",
            {
                "id": "item-secret",
                "type": "commandExecution",
                "command": "private command text",
                "aggregatedOutput": "private output text",
                "cwd": str(self.workspace / "sub"),
                "commandActions": [
                    {"type": "read", "command": "private", "path": "private"},
                    {"type": "search", "command": "private", "query": "private"},
                ],
                "status": "completed",
                "exitCode": 0,
                "durationMs": 12,
            },
        )

        projected = self.journal.observe_turn(
            agent_id="worker-secret",
            thread_id="thread-secret",
            turn_id="turn-secret",
            events=[event],
        )

        self.assertEqual(1, len(projected))
        self.assertEqual("command", projected[0].effect)
        self.assertEqual("sub", projected[0].cwd)
        self.assertEqual(("read", "search"), projected[0].action_types)
        self.assertFalse(projected[0].evidence_limited)
        encoded = json.dumps(self.journal.receipt({"worker-secret": "worker"}))
        for private in (
            "thread-secret",
            "turn-secret",
            "item-secret",
            "worker-secret",
            "private command text",
            "private output text",
            str(self.workspace),
        ):
            self.assertNotIn(private, encoded)

    def test_projects_file_changes_and_marks_outside_or_unknown_evidence(self) -> None:
        outside = self.workspace.parent / "outside-private.txt"
        event = completed(
            "thread",
            "turn",
            {
                "id": "file-item",
                "type": "fileChange",
                "status": "completed",
                "changes": [
                    {
                        "path": str(self.workspace / "sub" / "inside.txt"),
                        "kind": {"type": "add"},
                        "diff": "private inside diff",
                    },
                    {
                        "path": str(outside),
                        "kind": {"type": "future-kind"},
                        "diff": "private outside diff",
                    },
                ],
            },
        )

        effect = self.journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[event]
        )[0]

        self.assertEqual(
            [("sub/inside.txt", "add"), ("<outside-workspace>", "unknown")],
            [(value.path, value.kind) for value in effect.changes],
        )
        self.assertTrue(effect.evidence_limited)
        encoded = json.dumps(self.journal.receipt({"worker": "worker"}))
        self.assertNotIn("private inside diff", encoded)
        self.assertNotIn("private outside diff", encoded)
        self.assertNotIn(str(outside), encoded)

    def test_requires_exact_turn_correlation_and_deduplicates_item_id(self) -> None:
        item = {
            "id": "command-item",
            "type": "commandExecution",
            "command": "ignored",
            "cwd": str(self.workspace),
            "commandActions": [],
            "status": "completed",
            "exitCode": 0,
        }
        wrong = completed("other-thread", "turn", item)
        right = completed("thread", "turn", item)

        first = self.journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[wrong, right]
        )
        second = self.journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[right]
        )

        self.assertEqual(1, len(first))
        self.assertEqual(0, len(second))
        self.assertEqual(1, self.journal.summary()["uncorrelated_item_count"])

    def test_malformed_effect_is_counted_but_not_recorded(self) -> None:
        event = completed(
            "thread",
            "turn",
            {
                "type": "commandExecution",
                "cwd": ".",
                "commandActions": [],
                "status": "completed",
            },
        )

        effects = self.journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[event]
        )

        self.assertEqual((), effects)
        self.assertEqual(1, self.journal.summary()["malformed_item_count"])

    def test_a_full_journal_keeps_the_newest_effect_and_counts_the_drop(self) -> None:
        journal = RuntimeEffectJournal(self.workspace, max_effects=1)
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        base = {
            "type": "commandExecution",
            "command": "ignored",
            "cwd": ".",
            "commandActions": [],
            "status": "completed",
        }
        first = journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="turn",
            events=[completed("thread", "turn", {**base, "id": "one"})],
        )

        second = journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="turn",
            events=[completed("thread", "turn", {**base, "id": "two"})],
        )

        self.assertEqual(1, len(first))
        self.assertEqual(1, len(second))
        records = journal.records()
        self.assertEqual((second[0],), records)
        summary = journal.summary()
        self.assertEqual(1, summary["effect_count"])
        self.assertEqual(1, summary["dropped_effect_count"])
        self.assertFalse(summary["evidence_complete"])
        # The sequence numbers keep counting, so a reader sees that the run of
        # effects it holds does not start at the first one.
        self.assertEqual(2, records[0].sequence)

    def test_a_full_journal_never_fails_the_turn(self) -> None:
        journal = RuntimeEffectJournal(self.workspace, max_effects=2)
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        base = {
            "type": "commandExecution",
            "command": "ignored",
            "cwd": ".",
            "commandActions": [],
            "status": "completed",
        }
        for index in range(64):
            journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id=f"turn-{index}",
                events=[completed("thread", f"turn-{index}", {**base, "id": str(index)})],
            )

        summary = journal.summary()
        self.assertEqual(2, summary["effect_count"])
        self.assertEqual(62, summary["dropped_effect_count"])
        self.assertFalse(summary["evidence_complete"])

    def test_an_unfilled_journal_reports_complete_evidence(self) -> None:
        summary = self.journal.summary()

        self.assertEqual(0, summary["dropped_effect_count"])
        self.assertTrue(summary["evidence_complete"])

    def test_memory_stays_bounded_past_ten_times_the_capacity(self) -> None:
        capacity = 16
        journal = RuntimeEffectJournal(self.workspace, max_effects=capacity)
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        base = {
            "type": "commandExecution",
            "command": "ignored",
            "cwd": ".",
            "commandActions": [],
            "status": "completed",
        }
        inserts = capacity * 10
        for index in range(inserts):
            journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id=f"turn-{index}",
                events=[completed("thread", f"turn-{index}", {**base, "id": str(index)})],
            )

        self.assertEqual(capacity, len(journal.records()))
        self.assertEqual(inserts - capacity, journal.summary()["dropped_effect_count"])
        # The de-duplication set is the other thing that could grow for ever.
        self.assertLessEqual(len(journal._seen), capacity)
        self.assertEqual(capacity, len(journal.receipt({"worker": "worker"})))

    def _command(self, item_id: str) -> dict:
        return {
            "type": "commandExecution",
            "command": "ignored",
            "cwd": ".",
            "commandActions": [],
            "status": "completed",
            "id": item_id,
        }

    def test_an_item_replayed_after_its_effect_was_dropped_is_projected_once(
        self,
    ) -> None:
        journal = RuntimeEffectJournal(self.workspace, max_effects=2)
        journal.bind_reader("worker", CodexRuntimeEffectReader())

        projected = {}
        for item_id in ("A", "B", "C", "A"):
            projected[item_id] = journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id="turn",
                events=[completed("thread", "turn", self._command(item_id))],
            )

        # The replay of A projects nothing at all.
        self.assertEqual((), projected["A"])
        receipt = journal.receipt({"worker": "worker"})
        self.assertEqual([2, 3], [row["sequence"] for row in receipt])
        # B keeps its place; the replay does not push it out or sit above C.
        self.assertEqual(
            [projected["B"][0].item_ref, projected["C"][0].item_ref],
            [row["item_ref"] for row in receipt],
        )
        summary = journal.summary()
        self.assertEqual(2, summary["effect_count"])
        self.assertEqual(1, summary["dropped_effect_count"])
        self.assertEqual(1, summary["replayed_item_count"])
        self.assertEqual(1, journal.replayed_item_count)

    def test_a_finished_turn_replayed_after_its_key_expired_is_projected_once(
        self,
    ) -> None:
        # The replay memory is deliberately smaller than the run of drops, so
        # A's key is gone by the time A comes back.  What stops it is that the
        # agent has moved on to a later turn on the same thread.
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=2, max_replay_memory=1
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())

        for item_id in ("A", "B", "C"):
            journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id="first",
                events=[completed("thread", "first", self._command(item_id))],
            )
        journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="second",
            events=[completed("thread", "second", self._command("D"))],
        )
        self.assertNotIn(
            ("worker", "thread", "first", journal.records()[0].item_ref),
            journal._dropped_key_set,
        )

        replay = journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="first",
            events=[completed("thread", "first", self._command("A"))],
        )

        self.assertEqual((), replay)
        receipt = journal.receipt({"worker": "worker"})
        self.assertEqual([3, 4], [row["sequence"] for row in receipt])
        self.assertEqual(1, journal.replayed_item_count)

    def test_replay_memory_stays_bounded_past_ten_times_the_capacity(self) -> None:
        capacity = 8
        replay_memory = 4
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=capacity, max_replay_memory=replay_memory
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())

        inserts = capacity * 10
        for index in range(inserts):
            thread = f"thread-{index % 10}"
            turn = f"turn-{index}"
            journal.observe_turn(
                agent_id="worker",
                thread_id=thread,
                turn_id=turn,
                events=[completed(thread, turn, self._command(str(index)))],
            )
            # Replay the same item straight away, to feed the replay stores.
            journal.observe_turn(
                agent_id="worker",
                thread_id=thread,
                turn_id=turn,
                events=[completed(thread, turn, self._command(str(index)))],
            )

        self.assertEqual(capacity, len(journal.records()))
        self.assertLessEqual(len(journal._seen), capacity)
        self.assertLessEqual(len(journal._dropped_keys), replay_memory)
        self.assertLessEqual(len(journal._dropped_key_set), replay_memory)
        self.assertLessEqual(len(journal._closed_turns), replay_memory)
        self.assertLessEqual(len(journal._closed_turn_set), replay_memory)
        self.assertLessEqual(len(journal._current_turn), replay_memory)

    def test_the_replay_window_is_twice_the_effect_capacity_by_default(self) -> None:
        journal = RuntimeEffectJournal(self.workspace, max_effects=64)

        self.assertEqual(128, journal.max_replay_memory)

    def test_a_thread_re_read_inside_the_window_projects_nothing_again(self) -> None:
        # A resume hands back the whole thread.  While the window still holds
        # every key the thread produced, none of it is projected a second time.
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=8, max_replay_memory=400
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())

        def report(index: int) -> tuple:
            return journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id=f"turn-{index}",
                events=[
                    completed("thread", f"turn-{index}", self._command(f"item-{index}"))
                ],
            )

        for index in range(200):
            report(index)
        before = journal.summary()
        before_receipt = journal.receipt({"worker": "worker"})

        re_read = [report(index) for index in range(200)]

        self.assertTrue(all(value == () for value in re_read))
        self.assertEqual(before_receipt, journal.receipt({"worker": "worker"}))
        after = journal.summary()
        self.assertEqual(before["dropped_effect_count"], after["dropped_effect_count"])
        # The eight effects the journal still holds are caught by the plain
        # de-duplication set; the other 192 are caught by the replay window.
        self.assertEqual(192, after["replayed_item_count"])

    def test_a_re_read_is_caught_up_to_the_capacity_plus_the_window(self) -> None:
        # This is the reach of the fix, pinned: a thread whose history is no
        # longer than capacity + window is re-read without projecting anything
        # twice.  One item more and the re-read outruns the window, spends it on
        # its own re-projections, and the whole re-read lands in the journal.
        capacity, window = 8, 16

        def reprojected(history: int) -> int:
            journal = RuntimeEffectJournal(
                self.workspace, max_effects=capacity, max_replay_memory=window
            )
            journal.bind_reader("worker", CodexRuntimeEffectReader())
            for pass_number in range(2):
                total = 0
                for index in range(history):
                    total += len(
                        journal.observe_turn(
                            agent_id="worker",
                            thread_id="thread",
                            turn_id=f"turn-{index}",
                            events=[
                                completed(
                                    "thread",
                                    f"turn-{index}",
                                    self._command(f"item-{index}"),
                                )
                            ],
                        )
                    )
            return total

        for history in range(1, capacity + window + 1):
            with self.subTest(history=history):
                self.assertEqual(0, reprojected(history))
        self.assertEqual(
            capacity + window + 1, reprojected(capacity + window + 1)
        )

    def test_a_re_read_that_stops_short_still_projects_the_work_that_follows(
        self,
    ) -> None:
        # Erring towards projecting means a partial re-read never costs the
        # journal the live work that comes after it.
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=4, max_replay_memory=64
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())

        def report(index: int) -> tuple:
            return journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id=f"turn-{index}",
                events=[
                    completed("thread", f"turn-{index}", self._command(f"item-{index}"))
                ],
            )

        for index in range(20):
            report(index)
        for index in range(3):  # a re-read that breaks off after three turns
            report(index)

        fresh = [report(index) for index in range(20, 40)]

        self.assertTrue(all(len(value) == 1 for value in fresh))
        self.assertEqual(
            [value[0].item_ref for value in fresh[-4:]],
            [row["item_ref"] for row in journal.receipt({"worker": "worker"})],
        )

    def _turn_reporter(self, journal):
        def report(turn: str, item_id: str) -> tuple:
            return journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id=turn,
                events=[completed("thread", turn, self._command(item_id))],
            )

        return report

    def test_a_new_item_on_a_closed_turn_the_journal_remembers_whole_is_kept(
        self,
    ) -> None:
        # A on a first turn, then B and C on a second, then an item the journal
        # has never seen arrives for the first turn.  Every key the first turn
        # produced is still remembered, so an unknown key cannot be a repeat of
        # one of them: it is work never reported before, and it is recorded.
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=1, max_replay_memory=8
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        report = self._turn_reporter(journal)

        report("first", "A")
        report("second", "B")
        report("second", "C")

        late = report("first", "D")

        self.assertEqual(1, len(late))
        self.assertEqual(4, late[0].sequence)
        self.assertEqual(
            [late[0].item_ref],
            [row["item_ref"] for row in journal.receipt({"worker": "worker"})],
        )
        summary = journal.summary()
        self.assertEqual(0, summary["replayed_item_count"])
        self.assertEqual(0, summary["ambiguous_item_count"])

    def test_a_new_item_on_a_closed_turn_with_a_forgotten_key_is_disclosed(
        self,
    ) -> None:
        # The same arrival, with a replay window too small to hold the first
        # turn's key.  Now the journal cannot tell late work from an ancient
        # replay, so it suppresses the item and says so: the ambiguous count
        # rises and the evidence is no longer complete.
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=1, max_replay_memory=1
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        report = self._turn_reporter(journal)

        report("first", "A")
        report("second", "B")
        report("second", "C")

        late = report("first", "D")

        self.assertEqual((), late)
        summary = journal.summary()
        self.assertEqual(1, summary["ambiguous_item_count"])
        self.assertEqual(1, journal.ambiguous_item_count)
        self.assertFalse(summary["evidence_complete"])
        # The ambiguous count names the unproven share of the replay count, so
        # a reader never reads this suppression as a certain repeat.
        self.assertEqual(1, summary["replayed_item_count"])

    def test_a_re_read_of_a_closed_turn_the_journal_remembers_is_a_replay(
        self,
    ) -> None:
        # The other side of the same rule: an item the journal does remember is
        # a certain repeat, stays suppressed, and is not called ambiguous.
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=1, max_replay_memory=8
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        report = self._turn_reporter(journal)

        report("first", "A")
        report("second", "B")

        replay = report("first", "A")

        self.assertEqual((), replay)
        self.assertEqual(1, journal.replayed_item_count)
        self.assertEqual(0, journal.ambiguous_item_count)

    def test_the_whole_key_marks_stay_inside_the_replay_window(self) -> None:
        # The store grows with max_replay_memory and with nothing else.  Each
        # mark now lives as long as the turn it describes, so there is at most
        # one per closed turn plus one per current turn.
        capacity, window = 8, 4
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=capacity, max_replay_memory=window
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())

        for index in range(capacity * 10):
            thread = f"thread-{index % 10}"
            turn = f"turn-{index}"
            journal.observe_turn(
                agent_id="worker",
                thread_id=thread,
                turn_id=turn,
                events=[completed(thread, turn, self._command(str(index)))],
            )

        self.assertLessEqual(len(journal._turn_keys_intact), 2 * window)
        self.assertLessEqual(
            len(journal._turn_keys_intact),
            len(journal._closed_turns) + len(journal._current_turn),
        )

    def test_a_closed_turn_keeps_its_mark_through_many_empty_turns(self) -> None:
        # One effect on the first turn, then enough empty turns to push the
        # first turn's mark out of a store trimmed by age while the turn itself
        # is still closed and its key is still held.  A truly new item on it is
        # new work and not an ambiguous repeat.
        window = 4
        journal = RuntimeEffectJournal(
            self.workspace, max_effects=8, max_replay_memory=window
        )
        journal.bind_reader("worker", CodexRuntimeEffectReader())
        report = self._turn_reporter(journal)

        self.assertEqual(1, len(report("first", "A")))
        for index in range(window):
            journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id=f"empty-{index}",
                events=[],
            )

        self.assertIn(("worker", "thread", "first", "A"), journal._seen)
        self.assertIn(("worker", "thread", "first"), journal._closed_turn_set)
        self.assertEqual(0, journal.dropped_effect_count)

        late = report("first", "B")

        summary = journal.summary()
        self.assertEqual(1, len(late))
        self.assertEqual(0, summary["ambiguous_item_count"])
        self.assertTrue(summary["evidence_complete"])

    def test_provider_reader_must_be_bound_at_adapter_selection(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)

        with self.assertRaisesRegex(RuntimeEffectProjectionError, "was not selected"):
            journal.observe_turn(
                agent_id="worker",
                thread_id="thread",
                turn_id="turn",
                events=[],
            )

    def test_claude_reader_projects_sanitized_command_and_file_effects(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        command = claude_result("thread", "turn", "tool-command", "command")
        command["unsafe_input"] = "private command"
        command["unsafe_result"] = "private output"
        file = claude_result(
            "thread",
            "turn",
            "tool-file",
            "file",
            status="failed",
            changes=[{"path": "sub/from-claude.txt", "kind": {"type": "not_reported"}}],
        )

        effects = journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="turn",
            events=[{"name": "tool_use"}, command, file],
        )

        self.assertEqual(["command", "file_change"], [effect.effect for effect in effects])
        self.assertFalse(any(effect.evidence_limited for effect in effects))
        self.assertEqual(["claude", "claude"], [effect.provider for effect in effects])
        self.assertEqual(
            ("cwd", "actions", "exit_code", "duration_ms"),
            effects[0].not_reported_fields,
        )
        self.assertEqual(("change_kind",), effects[1].not_reported_fields)
        receipt = journal.receipt({"worker": "worker"})
        self.assertEqual("claude", receipt[0]["provider"])
        self.assertEqual(
            {"provider": "claude", "fields": ["cwd", "actions", "exit_code", "duration_ms"]},
            receipt[0]["not_reported"],
        )
        encoded = json.dumps(receipt)
        self.assertNotIn("private command", encoded)
        self.assertNotIn("private output", encoded)

    def test_claude_reader_marks_genuine_degradation_limited(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        degraded = claude_result(
            "thread",
            "turn",
            "tool-file",
            "file",
            changes=[{"path": "sub/degraded.txt"}],
            evidence_limited=True,
        )

        effect = journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[degraded]
        )[0]

        self.assertTrue(effect.evidence_limited)
        self.assertEqual(("change_kind",), effect.not_reported_fields)

    def test_claude_reader_rejects_a_reported_change_kind(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        event = claude_result(
            "thread",
            "turn",
            "tool-file",
            "file",
            changes=[{"path": "sub/reported-kind.txt", "kind": {"type": "add"}}],
        )

        effect = journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[event]
        )[0]

        self.assertTrue(effect.evidence_limited)
        self.assertEqual(("change_kind",), effect.not_reported_fields)
        self.assertEqual(
            [("sub/reported-kind.txt", "unknown")],
            [(change.path, change.kind) for change in effect.changes],
        )

    def test_unreported_status_is_recorded_without_limiting_the_evidence(self) -> None:
        # Observed live on Claude Code 0.2.143: a successful Write tool result
        # carries no `is_error`, so the bridge has no status to forward.
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        for label, status in (
            ("bridge placeholder", "unknown"),
            ("explicit null", None),
            ("field absent", OMIT),
        ):
            with self.subTest(status=label):
                event = claude_result(
                    "thread",
                    "turn",
                    f"tool-write-{label}",
                    "file",
                    status=status,
                    changes=[{"path": "sub/written.txt", "kind": {"type": "not_reported"}}],
                )

                effect = journal.observe_turn(
                    agent_id="worker", thread_id="thread", turn_id="turn", events=[event]
                )[0]

                self.assertEqual("unknown", effect.status)
                self.assertIn("status", effect.not_reported_fields)
                self.assertFalse(effect.evidence_limited)

    def test_failed_status_is_reported_and_not_declared_absent(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        event = claude_result(
            "thread",
            "turn",
            "tool-failed-write",
            "file",
            status="failed",
            changes=[{"path": "sub/failed.txt", "kind": {"type": "not_reported"}}],
        )

        effect = journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[event]
        )[0]

        self.assertEqual("failed", effect.status)
        self.assertEqual(("change_kind",), effect.not_reported_fields)
        self.assertNotIn("status", effect.not_reported_fields)
        self.assertFalse(effect.evidence_limited)

    def test_a_status_sent_while_declared_absent_is_contradictory(self) -> None:
        # An unrecognised status must not be laundered into an unlimited
        # `unknown` by the boundary also listing `status` as not reported.
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        event = claude_result(
            "thread",
            "turn",
            "tool-sneaky",
            "file",
            status="sneaky",
            changes=[{"path": "sub/sneaky.txt", "kind": {"type": "not_reported"}}],
        )

        effect = journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[event]
        )[0]

        self.assertEqual("unknown", effect.status)
        self.assertEqual(("change_kind", "status"), effect.not_reported_fields)
        self.assertTrue(effect.evidence_limited)

    def test_any_field_sent_while_declared_absent_is_contradictory(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        # Each case is the fully-declared-absent command or file item that the
        # Claude reader emits — proved unlimited by
        # `test_a_field_genuinely_absent_stays_unlimited` — with exactly one
        # field re-supplied.  The re-supplied field is therefore the only
        # possible source of a limited verdict.
        command = {
            "type": "command",
            "status": None,
            "cwd": None,
            "actions": None,
            "exit_code": None,
            "duration_ms": None,
            "not_reported": ["cwd", "actions", "exit_code", "duration_ms", "status"],
        }
        file = {
            "type": "file",
            "status": None,
            "changes": [{"path": "sub/x.txt", "kind": {"type": "not_reported"}}],
            "not_reported": ["change_kind", "status"],
        }
        cases = [
            {**command, "status": "completed"},
            {**command, "status": "sneaky"},
            {**command, "cwd": "sub"},
            {**command, "exit_code": 0},
            {**command, "duration_ms": 5},
            {**command, "actions": [{"type": "read"}]},
            {**file, "status": "failed"},
            {**file, "changes": [{"path": "sub/x.txt", "kind": {"type": "add"}}]},
        ]

        for index, item in enumerate(cases):
            with self.subTest(item=item):
                effect = journal._project_item("worker", "claude", f"item-{index}", item)
                assert effect is not None
                self.assertTrue(effect.evidence_limited)
                self.assertEqual(
                    tuple(item["not_reported"]), effect.not_reported_fields
                )

    def test_a_field_genuinely_absent_stays_unlimited(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        item = {
            "type": "command",
            "status": None,
            "cwd": None,
            "actions": None,
            "exit_code": None,
            "duration_ms": None,
            "not_reported": ["cwd", "actions", "exit_code", "duration_ms", "status"],
        }

        effect = journal._project_item("worker", "claude", "item", item)

        assert effect is not None
        self.assertEqual("unknown", effect.status)
        self.assertFalse(effect.evidence_limited)

        # The same file item, differing only in whether the change kind was
        # actually sent, is the discriminating pair for `change_kind`.
        file_item = {
            "type": "file",
            "status": None,
            "changes": [{"path": "sub/x.txt", "kind": {"type": "not_reported"}}],
            "not_reported": ["change_kind", "status"],
        }

        file_effect = journal._project_item("worker", "claude", "item-file", file_item)

        assert file_effect is not None
        self.assertEqual("unknown", file_effect.status)
        self.assertFalse(file_effect.evidence_limited)

    def test_not_reported_schema_is_effect_specific_and_fail_closed(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        cases = [
            {"type": "command", "not_reported": ["change_kind"]},
            {"type": "command", "not_reported": ["cwd", "cwd"]},
            {"type": "file", "not_reported": ["actions"]},
            {"type": "file", "not_reported": "change_kind"},
            {"type": "command", "not_reported": ["cwd", 7]},
        ]

        for item in cases:
            with self.subTest(item=item):
                effect = journal._project_item("worker", "claude", "item", item)
                self.assertIsNotNone(effect)
                assert effect is not None
                self.assertTrue(effect.evidence_limited)
                self.assertEqual((), effect.not_reported_fields)

    def test_claude_reader_deduplicates_and_rejects_malformed_correlation(self) -> None:
        journal = RuntimeEffectJournal(self.workspace)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        right = claude_result("thread", "turn", "tool-one", "command")
        malformed = claude_result("thread", "turn", "tool-two", "file")
        malformed["provider_correlation"] = {"session": "native-session", "turn": "other", "request": "tool-two"}

        first = journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[right, right, malformed]
        )
        second = journal.observe_turn(
            agent_id="worker", thread_id="thread", turn_id="turn", events=[right]
        )

        self.assertEqual(1, len(first))
        self.assertEqual((), second)
        self.assertEqual(1, journal.summary()["malformed_item_count"])

    def test_claude_reader_drops_the_oldest_effect_when_full(self) -> None:
        journal = RuntimeEffectJournal(self.workspace, max_effects=1)
        journal.bind_reader("worker", ClaudeRuntimeEffectReader())
        journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="turn",
            events=[claude_result("thread", "turn", "tool-one", "command")],
        )

        second = journal.observe_turn(
            agent_id="worker",
            thread_id="thread",
            turn_id="turn",
            events=[claude_result("thread", "turn", "tool-two", "file")],
        )

        self.assertEqual(1, len(second))
        self.assertEqual(("file_change",), tuple(v.effect for v in journal.records()))
        self.assertEqual(1, journal.summary()["dropped_effect_count"])

    def test_opaque_provider_correlations_do_not_deduplicate_across_agents(self) -> None:
        self.journal.bind_reader("other-worker", CodexRuntimeEffectReader())
        event = completed(
            "shared-thread-ref",
            "shared-turn-ref",
            {
                "id": "shared-item-ref",
                "type": "commandExecution",
                "cwd": ".",
                "commandActions": [],
                "status": "completed",
            },
        )

        first = self.journal.observe_turn(
            agent_id="worker",
            thread_id="shared-thread-ref",
            turn_id="shared-turn-ref",
            events=[event],
        )
        second = self.journal.observe_turn(
            agent_id="other-worker",
            thread_id="shared-thread-ref",
            turn_id="shared-turn-ref",
            events=[event],
        )

        self.assertEqual(1, len(first))
        self.assertEqual(1, len(second))


class RegisteredProviderEffectReaderTests(unittest.TestCase):
    """Round-5 finding 3: --provider NAME=module:attr blocked on its first turn.

    ``--provider`` is documented and validated, and a registered provider's
    card reached a reader lookup keyed on the catalog name alone, which raised.
    The event wire belongs to the harness the adapter speaks, so the adapter is
    what answers.
    """

    def test_a_built_in_card_still_resolves_from_its_name_alone(self) -> None:
        for provider, expected in (
            ("codex", CodexRuntimeEffectReader),
            ("claude", ClaudeRuntimeEffectReader),
            ("commandcode", CommandCodeRuntimeEffectReader),
        ):
            with self.subTest(provider):
                self.assertIsInstance(runtime_effect_reader(provider), expected)

    def test_an_adapter_may_supply_its_own_reader(self) -> None:
        class AcmeReader:
            provider = "acme"

            def completed_effect(self, event):
                return None

        supplied = AcmeReader()

        class Adapter:
            runtime_effect_reader = supplied

        class FactoryAdapter:
            @staticmethod
            def runtime_effect_reader():
                return supplied

        self.assertIs(supplied, runtime_effect_reader("acme", Adapter()))
        self.assertIs(supplied, runtime_effect_reader("acme", FactoryAdapter()))

    def test_a_supplied_reader_that_decodes_nothing_is_refused(self) -> None:
        class Adapter:
            runtime_effect_reader = object()

        with self.assertRaises(RuntimeEffectProjectionError) as caught:
            runtime_effect_reader("acme", Adapter())

        self.assertIn("completed_effect", str(caught.exception))

    def test_an_adapter_may_name_the_harness_it_speaks_for(self) -> None:
        class Adapter:
            provider = "codex"
            harness = "app-server"

        self.assertIsInstance(
            runtime_effect_reader("acme", Adapter()), CodexRuntimeEffectReader
        )

    def test_an_adapter_that_answers_neither_way_says_both_ways(self) -> None:
        with self.assertRaises(RuntimeEffectProjectionError) as caught:
            runtime_effect_reader("acme", object())

        message = str(caught.exception)
        self.assertIn("no native effect reader is registered for provider 'acme'", message)
        self.assertIn("runtime_effect_reader", message)
        self.assertIn("codex", message)

    def test_the_hint_reads_a_class_without_starting_a_process(self) -> None:
        """The CLI asks at parse time, so no adapter may be constructed."""

        class Supplying:
            def __init__(self) -> None:  # pragma: no cover - must never run
                raise AssertionError("the hint must not construct an adapter")

            runtime_effect_reader = object()

        class Naming:
            def __init__(self) -> None:  # pragma: no cover - must never run
                raise AssertionError("the hint must not construct an adapter")

            provider = "claude"

        class Silent:
            pass

        self.assertEqual("runtime_effect_reader", effect_reader_hint(Supplying))
        self.assertEqual("claude", effect_reader_hint(Naming))
        self.assertIsNone(effect_reader_hint(Silent))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
