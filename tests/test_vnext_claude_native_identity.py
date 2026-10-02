"""Unit proof for the ephemeral Claude native-child identity ledger."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import unittest

from vnext.vnext_claude_native_identity import (
    NativeChildAutomaticResolver,
    NativeChildIdentity,
    NativeChildIdentityError,
    NativeChildIdentityLedger,
)


def _updated_input(decision: dict) -> dict:
    return decision["hookSpecificOutput"]["updatedInput"]


def _token(decision: dict) -> str:
    prompt = _updated_input(decision)["prompt"]
    match = re.search(r"enrollment_token=('[^']+')", prompt)
    if match is None:
        raise AssertionError("the native Agent enrollment instruction lacks its token")
    return match.group(1)[1:-1]


class NativeChildIdentityLedgerTests(unittest.TestCase):
    SESSION = "native-parent-uuid"

    def test_concurrent_children_bind_by_exact_capabilities_not_event_timing(self) -> None:
        ledger = NativeChildIdentityLedger()
        first = ledger.rewrite_agent_input(self.SESSION, "parent-tool-a", {"prompt": "first"})
        second = ledger.rewrite_agent_input(self.SESSION, "parent-tool-b", {"prompt": "second"})

        # Task messages may arrive before a child can call its inherited MCP
        # registration tool.  They remain keyed to their own parent tool use.
        self.assertIsNone(ledger.record_task_started(self.SESSION, "parent-tool-b", "task-b"))
        self.assertIsNone(ledger.record_task_started(self.SESSION, "parent-tool-a", "task-a"))

        first_register = ledger.rewrite_register_input(
            self.SESSION, "agent-a", "child-hook-a", {"enrollment_token": _token(first)}
        )
        second_register = ledger.rewrite_register_input(
            self.SESSION, "agent-b", "child-hook-b", {"enrollment_token": _token(second)}
        )
        # Register concurrently and in reverse submission order.  There is no
        # shared latest-event slot for one child to overwrite.
        with ThreadPoolExecutor(max_workers=2) as pool:
            second_future = pool.submit(
                ledger.register_from_mcp,
                self.SESSION,
                _token(second),
                _updated_input(second_register)["registration_proof"],
            )
            first_future = pool.submit(
                ledger.register_from_mcp,
                self.SESSION,
                _token(first),
                _updated_input(first_register)["registration_proof"],
            )
            second_identity = second_future.result()
            first_identity = first_future.result()

        self.assertEqual(
            NativeChildIdentity(self.SESSION, "parent-tool-a", "agent-a", "task-a"), first_identity
        )
        self.assertEqual(
            NativeChildIdentity(self.SESSION, "parent-tool-b", "agent-b", "task-b"), second_identity
        )
        self.assertEqual(first_identity, ledger.identity_for_task(self.SESSION, "task-a"))
        self.assertEqual(second_identity, ledger.identity_for_agent(self.SESSION, "agent-b"))

    def test_forged_or_replayed_capabilities_cannot_register_a_child(self) -> None:
        ledger = NativeChildIdentityLedger()
        enrollment = ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": "work"})
        token = _token(enrollment)

        with self.assertRaisesRegex(NativeChildIdentityError, "unknown or consumed"):
            ledger.rewrite_register_input(self.SESSION, "forged-agent", "hook-forged", {"enrollment_token": "forged"})

        register = ledger.rewrite_register_input(
            self.SESSION, "agent-real", "hook-real", {"enrollment_token": token}
        )
        with self.assertRaisesRegex(NativeChildIdentityError, "not attested"):
            ledger.register_from_mcp(self.SESSION, token, "forged-proof")
        identity = ledger.register_from_mcp(
            self.SESSION, token, _updated_input(register)["registration_proof"]
        )
        self.assertEqual("agent-real", identity.agent_id)
        with self.assertRaisesRegex(NativeChildIdentityError, "not attested"):
            ledger.register_from_mcp(self.SESSION, token, _updated_input(register)["registration_proof"])
        with self.assertRaisesRegex(NativeChildIdentityError, "unknown or consumed"):
            ledger.rewrite_register_input("other-parent", "agent-other", "hook-other", {"enrollment_token": token})

    def test_replayed_parent_hook_is_idempotent_but_conflicting_reuse_fails_closed(self) -> None:
        ledger = NativeChildIdentityLedger()
        enrollment = ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": "work"})
        replay = ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": "work"})
        self.assertEqual(_token(enrollment), _token(replay))
        with self.assertRaisesRegex(NativeChildIdentityError, "conflicting input"):
            ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": "different work"})

    def test_duplicate_agent_and_unenrolled_task_fail_closed(self) -> None:
        ledger = NativeChildIdentityLedger()
        enrollment = ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": "work"})
        with self.assertRaisesRegex(NativeChildIdentityError, "no enrolled parent"):
            ledger.record_task_started(self.SESSION, "unknown-parent-tool", "task")
        ledger.rewrite_register_input(
            self.SESSION, "agent", "hook-one", {"enrollment_token": _token(enrollment)}
        )
        other = ledger.rewrite_agent_input(self.SESSION, "parent-tool-two", {"prompt": "other"})
        with self.assertRaisesRegex(NativeChildIdentityError, "already bound"):
            ledger.rewrite_register_input(
                self.SESSION, "agent", "hook-two", {"enrollment_token": _token(other)}
            )

    def test_snapshot_excludes_raw_prompt_and_capabilities(self) -> None:
        ledger = NativeChildIdentityLedger()
        prompt = "do not retain this private instruction"
        enrollment = ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": prompt})
        token = _token(enrollment)
        register = ledger.rewrite_register_input(
            self.SESSION, "agent", "hook", {"enrollment_token": token}
        )
        ledger.register_from_mcp(self.SESSION, token, _updated_input(register)["registration_proof"])
        snapshot = repr(ledger.snapshot())
        self.assertNotIn(prompt, snapshot)
        self.assertNotIn(token, snapshot)
        self.assertNotIn(_updated_input(register)["registration_proof"], snapshot)

    def test_coordination_context_is_child_scoped_one_shot_and_overwrites_forgery(self) -> None:
        ledger = NativeChildIdentityLedger()
        enrollment = ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"prompt": "work"})
        token = _token(enrollment)
        register = ledger.rewrite_register_input(
            self.SESSION, "agent", "register-hook", {"enrollment_token": token}
        )
        ledger.register_from_mcp(self.SESSION, token, _updated_input(register)["registration_proof"])
        decision = ledger.rewrite_coordination_input(
            self.SESSION,
            "agent",
            "coordination-hook",
            {"role": "worker", "_vnext_native_child_context": "forged"},
            context_field="_vnext_native_child_context",
        )
        proof = _updated_input(decision)["_vnext_native_child_context"]
        self.assertNotEqual("forged", proof)
        identity = ledger.consume_coordination_context(self.SESSION, proof)
        self.assertEqual("agent", identity.agent_id)
        with self.assertRaisesRegex(NativeChildIdentityError, "not attested"):
            ledger.consume_coordination_context(self.SESSION, proof)

    def test_agent_prompt_shape_is_a_required_live_contract(self) -> None:
        ledger = NativeChildIdentityLedger()
        with self.assertRaisesRegex(NativeChildIdentityError, "string prompt"):
            ledger.rewrite_agent_input(self.SESSION, "parent-tool", {"description": "no prompt"})


class NativeChildAutomaticResolverTests(unittest.TestCase):
    """Offline proof for the SDK metadata join; no provider is contacted."""

    SESSION = "native-parent-uuid"

    def _root_child(self, resolver: NativeChildAutomaticResolver, *, agent: str = "child") -> None:
        resolver.record_agent_origin(
            self.SESSION, "root-tool", None, parent_agent_id_present=True
        )
        resolver.record_subagent_start(self.SESSION, agent)
        resolver.record_metadata(self.SESSION, agent, "root-tool", None)
        resolver.record_task_started(self.SESSION, "root-tool", "root-task")

    def test_out_of_order_concurrent_children_need_exact_metadata_and_task(self) -> None:
        resolver = NativeChildAutomaticResolver()
        resolver.record_agent_origin(self.SESSION, "tool-a", None, parent_agent_id_present=True)
        resolver.record_agent_origin(self.SESSION, "tool-b", None, parent_agent_id_present=True)
        resolver.record_subagent_start(self.SESSION, "agent-a")
        resolver.record_subagent_start(self.SESSION, "agent-b")
        resolver.record_task_started(self.SESSION, "tool-b", "task-b")
        resolver.record_metadata(self.SESSION, "agent-a", "tool-a", None)
        self.assertEqual((), resolver.resolve_ready())
        resolver.record_metadata(self.SESSION, "agent-b", "tool-b", None)
        resolver.record_task_started(self.SESSION, "tool-a", "task-a")

        identities = resolver.resolve_ready()
        self.assertEqual({"agent-a", "agent-b"}, {item.agent_id for item in identities})
        self.assertEqual({"task-a", "task-b"}, {item.task_id for item in identities})
        self.assertEqual((), resolver.resolve_ready())

    def test_missing_parent_agent_id_is_not_root_without_authenticated_origin(self) -> None:
        resolver = NativeChildAutomaticResolver()
        resolver.record_subagent_start(self.SESSION, "child")
        resolver.record_metadata(self.SESSION, "child", "tool", None)
        resolver.record_task_started(self.SESSION, "tool", "task")
        self.assertEqual((), resolver.resolve_ready())

        # A missing hook field is not evidence that the caller was the root.
        resolver.record_agent_origin(self.SESSION, "tool", None, parent_agent_id_present=False)
        self.assertEqual((), resolver.resolve_ready())
        resolver.record_agent_origin(self.SESSION, "tool", None, parent_agent_id_present=True)
        identity = resolver.resolve_ready()[0]
        self.assertIsNone(identity.parent_agent_id)
        self.assertEqual("saved_session_metadata", identity.identity_source)

    def test_nested_children_wait_for_an_exact_joined_parent_and_emit_in_order(self) -> None:
        resolver = NativeChildAutomaticResolver()
        resolver.record_agent_origin(self.SESSION, "root-tool", None, parent_agent_id_present=True)
        resolver.record_agent_origin(self.SESSION, "nested-tool", "parent", parent_agent_id_present=True)
        resolver.record_subagent_start(self.SESSION, "parent")
        resolver.record_subagent_start(self.SESSION, "child")
        resolver.record_metadata(self.SESSION, "parent", "root-tool", None)
        resolver.record_metadata(self.SESSION, "child", "nested-tool", "parent")
        resolver.record_task_started(self.SESSION, "nested-tool", "child-task")
        self.assertEqual((), resolver.resolve_ready())
        resolver.record_task_started(self.SESSION, "root-tool", "parent-task")

        joined = resolver.resolve_ready()
        self.assertEqual(["parent", "child"], [item.agent_id for item in joined])
        self.assertEqual("parent-task", joined[1].parent_task_id)
        self.assertEqual("parent", joined[1].parent_agent_id)

    def test_duplicate_conflicting_and_stale_facts_fail_closed(self) -> None:
        resolver = NativeChildAutomaticResolver()
        self._root_child(resolver)
        self.assertEqual(1, len(resolver.resolve_ready()))
        # Exact replay is idempotent after restart/reconnect.
        resolver.record_subagent_start(self.SESSION, "child")
        resolver.record_metadata(self.SESSION, "child", "root-tool", None)
        resolver.record_task_started(self.SESSION, "root-tool", "root-task")
        self.assertEqual((), resolver.resolve_ready())
        with self.assertRaisesRegex(NativeChildIdentityError, "conflicts"):
            resolver.record_metadata(self.SESSION, "child", "other-tool", None)
        with self.assertRaisesRegex(NativeChildIdentityError, "conflicting task"):
            resolver.record_task_started(self.SESSION, "root-tool", "stale-task")

    def test_task_and_agent_tool_collisions_are_refused(self) -> None:
        resolver = NativeChildAutomaticResolver()
        resolver.record_task_started(self.SESSION, "tool-a", "shared-task")
        with self.assertRaisesRegex(NativeChildIdentityError, "already bound to another Agent tool"):
            resolver.record_task_started(self.SESSION, "tool-b", "shared-task")
        resolver.record_metadata(self.SESSION, "agent-a", "tool-a", None)
        with self.assertRaisesRegex(NativeChildIdentityError, "already bound to another child"):
            resolver.record_metadata(self.SESSION, "agent-b", "tool-a", None)

    def test_content_free_replay_restores_pending_and_joined_state(self) -> None:
        resolver = NativeChildAutomaticResolver()
        self._root_child(resolver, agent="joined")
        resolver.record_subagent_start(self.SESSION, "pending")
        resolver.record_metadata(self.SESSION, "pending", "late-tool", None)
        resolver.resolve_ready()
        replay = resolver.replay_state()
        self.assertNotIn("prompt", repr(replay))
        restored = NativeChildAutomaticResolver.from_replay_state(replay)
        self.assertEqual({"joined"}, {item.agent_id for item in restored.snapshot()})
        restored.record_agent_origin(self.SESSION, "late-tool", None, parent_agent_id_present=True)
        restored.record_task_started(self.SESSION, "late-tool", "late-task")
        self.assertEqual("pending", restored.resolve_ready()[0].agent_id)
        with self.assertRaisesRegex(NativeChildIdentityError, "invalid schema"):
            NativeChildAutomaticResolver.from_replay_state({"version": 1})

    def test_replay_preserves_metadata_that_arrived_before_subagent_start(self) -> None:
        resolver = NativeChildAutomaticResolver()
        resolver.record_agent_origin(self.SESSION, "tool", None, parent_agent_id_present=True)
        resolver.record_metadata(self.SESSION, "child", "tool", None)
        restored = NativeChildAutomaticResolver.from_replay_state(resolver.replay_state())
        restored.record_subagent_start(self.SESSION, "child")
        restored.record_task_started(self.SESSION, "tool", "task")
        self.assertEqual("child", restored.resolve_ready()[0].agent_id)

    def test_no_tool_child_is_joined_from_hook_metadata_and_task_only(self) -> None:
        resolver = NativeChildAutomaticResolver()
        self._root_child(resolver, agent="no-tool-child")
        identity = resolver.resolve_ready()[0]
        self.assertEqual("no-tool-child", identity.agent_id)
        self.assertEqual("root-task", resolver.identity_for_task(self.SESSION, "root-task").task_id)

    def test_resolved_children_do_not_consume_the_pending_child_bound(self) -> None:
        resolver = NativeChildAutomaticResolver()
        for index in range(65):
            tool, agent, task = f"tool-{index}", f"agent-{index}", f"task-{index}"
            resolver.record_agent_origin(self.SESSION, tool, None, parent_agent_id_present=True)
            resolver.record_subagent_start(self.SESSION, agent)
            resolver.record_metadata(self.SESSION, agent, tool, None)
            resolver.record_task_started(self.SESSION, tool, task)
            self.assertEqual(agent, resolver.resolve_ready()[0].agent_id)

    def test_metadata_first_eviction_clears_old_tool_binding(self) -> None:
        evictions: list[tuple[str, str, int]] = []
        resolver = NativeChildAutomaticResolver(on_eviction=lambda *row: evictions.append(row))
        for index in range(64):
            resolver.record_metadata(self.SESSION, f"stale-{index}", f"tool-{index}")
        resolver.record_metadata(self.SESSION, "healthy", "healthy-tool")
        self.assertEqual([(self.SESSION, "stale-0", 64)], evictions)
        self.assertLessEqual(len(resolver._children), 64)
        resolver.record_metadata(self.SESSION, "replacement", "tool-0")
        self.assertEqual((self.SESSION, "stale-1", 64), evictions[-1])
        self.assertLessEqual(len(resolver._children), 64)

    def test_oldest_unjoined_child_is_evicted_and_healthy_child_joins(self) -> None:
        evictions: list[tuple[str, str, int]] = []
        resolver = NativeChildAutomaticResolver(on_eviction=lambda *row: evictions.append(row))
        for index in range(64):
            resolver.record_subagent_start(self.SESSION, f"stale-{index}")
        resolver.record_agent_origin(self.SESSION, "healthy-tool", None, parent_agent_id_present=True)
        resolver.record_task_started(self.SESSION, "healthy-tool", "healthy-task")
        resolver.record_subagent_start(self.SESSION, "healthy")
        resolver.record_metadata(self.SESSION, "healthy", "healthy-tool", None)
        self.assertEqual("healthy", resolver.resolve_ready()[0].agent_id)
        self.assertEqual([(self.SESSION, "stale-0", 64)], evictions)
        self.assertNotIn("stale-0", resolver.pending_agents(self.SESSION))
        self.assertLessEqual(len(resolver._children), 64)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
