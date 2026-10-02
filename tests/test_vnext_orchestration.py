from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.vnext_orchestration import (
    AgentRole,
    AgentStatus,
    EconomicPreset,
    ModelCard,
    ModelRegistry,
    OrchestrationControlPlane,
    ProtocolError,
    TERMINAL_STATUSES,
)


def registry() -> ModelRegistry:
    return ModelRegistry(
        cards=[
            ModelCard("root-capable", frozenset({AgentRole.ROOT_MANAGER})),
            ModelCard("branch-capable", frozenset({AgentRole.BRANCH_MANAGER})),
            ModelCard("economy-worker", frozenset({AgentRole.WORKER})),
            ModelCard("strong-worker", frozenset({AgentRole.WORKER})),
            ModelCard("outside-envelope", frozenset({AgentRole.WORKER})),
        ],
        presets=[
            EconomicPreset(
                "balanced",
                frozenset({"root-capable", "branch-capable", "economy-worker", "strong-worker"}),
            )
        ],
    )


class VNextOrchestrationPrototypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=self.workspace,
            objective="Deliver the requested outcome",
            task_contract={"criteria": ["verified"]},
            session_id="session-a",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def branch_and_worker(self):
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Own one coherent branch",
            task_contract={"criteria": ["branch verified"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="Perform one bounded task",
            task_contract={"criteria": ["artifact written"]},
        )
        return branch, worker

    def test_manager_explicitly_selects_topology_and_model(self) -> None:
        self.assertEqual([self.root.agent_id], list(self.control.sessions["session-a"].agents))
        branch, worker = self.branch_and_worker()
        self.assertEqual("branch-capable", branch.model_id)
        self.assertEqual("economy-worker", worker.model_id)
        self.assertEqual(self.root.agent_id, branch.parent_agent_id)
        self.assertEqual(branch.agent_id, worker.parent_agent_id)

    def test_a_manager_may_delegate_past_the_ceiling_that_used_to_exist(self) -> None:
        """No number bounds how wide a manager may go.

        The old ``max_active_agents`` refused a spawn once four agents were
        non-terminal.  A refusal there does not save money: the manager that
        cannot delegate does the work in its own long-lived thread, which is
        the expensive one.  Twelve live agents is comfortably past the old
        ceiling and past the product preset's eight, and none of it is refused.
        """

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Own one coherent branch",
            task_contract={"criteria": ["branch verified"]},
        )
        for index in range(10):
            self.control.spawn_agent(
                requester_id=branch.agent_id,
                parent_agent_id=branch.agent_id,
                role=AgentRole.WORKER,
                model_id="economy-worker",
                objective=f"Perform bounded task {index}",
                task_contract={"criteria": ["artifact written"]},
            )

        session = self.control.sessions["session-a"]
        live = [
            agent
            for agent in session.agents.values()
            if agent.status not in TERMINAL_STATUSES
        ]
        self.assertEqual(12, len(live))

    def test_the_preset_carries_no_concurrency_number_at_all(self) -> None:
        """A smaller cap is still a cap, so there is no field to set one in."""

        preset = self.control.registry.preset("balanced")
        self.assertFalse(hasattr(preset, "max_active_agents"))
        self.assertNotIn(
            "max_active_agents",
            {field.name for field in dataclasses.fields(EconomicPreset)},
        )

    def test_agent_capabilities_are_not_fixed_by_depth_but_preset_eligibility_remains(self) -> None:
        branch, worker = self.branch_and_worker()
        grandchild = self.control.spawn_agent(
            requester_id=worker.agent_id,
            parent_agent_id=worker.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="delegate a narrow inspection",
            task_contract={},
        )
        self.assertEqual(worker.agent_id, grandchild.parent_agent_id)
        self.assertEqual(AgentRole.WORKER, grandchild.role)
        with self.assertRaises(ProtocolError) as outside:
            self.control.replace_agent(
                requester_id=branch.agent_id,
                agent_id=worker.agent_id,
                model_id="outside-envelope",
                revised_task_contract={"criteria": []},
            )
        self.assertEqual("model-not-allowed", outside.exception.code)
        self.assertEqual(AgentStatus.READY, worker.status)

    def test_peers_can_exchange_attributed_mail_without_changing_lifecycle_ownership(self) -> None:
        branch, first = self.branch_and_worker()
        second = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="review the first worker's result",
            task_contract={},
        )

        self.control.message_agent(first.agent_id, second.agent_id, "Check the failing test output.", kind="message")

        queued = self.control.inspect_agent(second.agent_id, second.agent_id, deep=True)
        self.assertEqual(
            {"sender_id": first.agent_id, "kind": "message", "direct_override": True},
            queued["recent_messages"][-1],
        )
        self.assertEqual(["manager-message"], [wake["reason"] for wake in self.control.drain_wakes(second.agent_id)])

    def test_a_child_that_blocks_while_its_manager_works_is_not_lost(self) -> None:
        """A manager that has not parked still has to hear that a child stopped.

        Managers are now told to fan out and keep working before they yield, so
        the window in which a manager is busy rather than parked is the ordinary
        case rather than a corner. A blocker raised in that window used to leave
        no record at all, because wakes were only ever recorded for a manager
        that had already registered a wait.
        """

        branch, worker = self.branch_and_worker()
        session = self.control.sessions["session-a"]
        self.assertNotIn(branch.agent_id, session.waits)

        self.control.block_agent(worker.agent_id, "needs a decision")

        wakes = self.control.drain_wakes(branch.agent_id)
        self.assertEqual(
            [("child-blocked", worker.agent_id)],
            [(wake["reason"], wake["source_agent_id"]) for wake in wakes],
        )

    def test_a_parked_manager_is_only_woken_by_a_child_it_selected(self) -> None:
        """Un-parking stays the manager's choice; being told does not.

        await_children names the children whose movement is worth resuming for.
        A child left out of that list may still report, and the report is kept
        for the manager's next prompt, but it does not drag the manager back.
        """

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Own one coherent branch",
            task_contract={"criteria": ["branch verified"]},
        )
        watched = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="the one that matters",
            task_contract={},
        )
        ignored = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="the one left out",
            task_contract={},
        )
        self.control.await_agents(branch.agent_id, [watched.agent_id])
        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)

        self.control.block_agent(ignored.agent_id, "stopped on its own")

        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)
        self.control.block_agent(watched.agent_id, "stopped too")
        self.assertEqual(AgentStatus.READY, branch.status)
        wakes = self.control.drain_wakes(branch.agent_id)
        self.assertEqual(
            [ignored.agent_id, watched.agent_id],
            [wake["source_agent_id"] for wake in wakes],
        )

    def test_a_satisfied_wait_does_not_keep_un_parking_the_manager(self) -> None:
        """A stale watch list would resume a manager for children it moved on from."""

        branch, worker = self.branch_and_worker()
        self.control.await_agents(branch.agent_id, [worker.agent_id])
        self.control.block_agent(worker.agent_id, "stopped")
        self.assertEqual(AgentStatus.READY, branch.status)

        session = self.control.sessions["session-a"]
        self.assertNotIn(branch.agent_id, session.waits)

    def test_one_child_reporting_the_same_thing_twice_wakes_once(self) -> None:
        branch, worker = self.branch_and_worker()

        self.control.block_agent(worker.agent_id, "stopped")
        self.control.retry_agent(
            requester_id=branch.agent_id,
            agent_id=worker.agent_id,
            revised_task_contract={},
        )
        self.control.block_agent(worker.agent_id, "stopped again")

        wakes = self.control.drain_wakes(branch.agent_id)
        self.assertEqual(1, len(wakes))
        self.assertEqual("child-blocked", wakes[0]["reason"])

    def test_a_manager_still_parks_after_acting_on_a_child_itself(self) -> None:
        """Its own actions must not cost a manager the turn it wanted to yield.

        Wakes now reach a manager whether or not it has parked, so its queue
        routinely holds news it already has: a child it cancelled itself, a
        blocker it has just read. Refusing to park for any of that would spend a
        whole extra turn, with every child re-rendered, telling the manager what
        it did a moment ago. Only news about a child it is now waiting on is
        worth refusing the park for.
        """

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="own one branch",
            task_contract={},
        )
        abandoned = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="the one it gives up on",
            task_contract={},
        )
        kept = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="the one it waits for",
            task_contract={},
        )
        self.control.cancel_agent(requester_id=branch.agent_id, agent_id=abandoned.agent_id)

        self.control.await_agents(branch.agent_id, [kept.agent_id])

        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)
        # The news is kept, it just does not refuse the park.
        pending = self.control.sessions["session-a"].wakes[branch.agent_id]
        self.assertEqual([abandoned.agent_id], [item["source_agent_id"] for item in pending])

    def test_a_watched_child_that_moved_first_still_refuses_the_park(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.block_agent(worker.agent_id, "stopped before the manager yielded")

        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.assertEqual(AgentStatus.READY, branch.status)
        events = [item.event_type for item in self.control.sessions["session-a"].events]
        self.assertIn("wait-skipped-pending-wake", events)

    def test_cancelling_a_subtree_leaves_no_mail_for_agents_that_are_gone(self) -> None:
        branch, worker = self.branch_and_worker()
        session = self.control.sessions["session-a"]

        self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=branch.agent_id)

        self.assertEqual(AgentStatus.CANCELLED, branch.status)
        self.assertEqual(AgentStatus.CANCELLED, worker.status)
        # The root is still live and is told its branch went. The branch is not,
        # so nothing is left addressed to it.
        self.assertNotIn(branch.agent_id, session.wakes)
        self.assertEqual(
            [branch.agent_id],
            [item["source_agent_id"] for item in session.wakes[self.root.agent_id]],
        )

    def test_restored_wakes_go_back_in_front_and_do_not_duplicate(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.block_agent(worker.agent_id, "stopped")
        drained = self.control.drain_wakes(branch.agent_id)
        self.assertEqual(1, len(drained))
        self.control.retry_agent(
            requester_id=branch.agent_id,
            agent_id=worker.agent_id,
            revised_task_contract={},
        )
        self.control.block_agent(worker.agent_id, "stopped again")

        self.control.restore_wakes(branch.agent_id, drained)

        pending = self.control.drain_wakes(branch.agent_id)
        self.assertEqual(1, len(pending))
        self.assertEqual("child-blocked", pending[0]["reason"])

    def test_one_completion_wakes_its_manager_once(self) -> None:
        """The final progress record and the completion are one event.

        A completing agent writes its last progress inside the same transaction
        as the completion. Recorded as material it woke the parent about that,
        and then the completion woke it again about the same thing: two entries
        in the next prompt for one thing that happened.
        """

        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)

        self.control.complete_agent(
            worker.agent_id,
            {"outcome": "completed"},
            progress={
                "activity": "done",
                "progress": "finished the bounded task",
                "files_touched": [],
                "commands": [],
            },
        )

        wakes = self.control.drain_wakes(branch.agent_id)
        self.assertEqual(["child-completed"], [wake["reason"] for wake in wakes])
        # The progress itself is still recorded; only the second wake is gone.
        view = self.control.inspect_agent(branch.agent_id, worker.agent_id, deep=True)
        self.assertEqual("finished the bounded task", view["latest_progress"])

    def test_final_progress_preserves_runtime_usage_totals_it_does_not_name(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        worker.usage = {
            "cost_tokens": {"basis": "call", "total_tokens": 123},
            "cost_usd": 1.25,
            "provider_calls": 4,
            "provider": "codex",
        }

        self.control.complete_agent(
            worker.agent_id,
            {"outcome": "completed"},
            progress={
                "activity": "done",
                "progress": "finished the bounded task",
                "files_touched": [],
                "commands": [],
                "usage": {"provider": "claude"},
            },
        )

        view = self.control.inspect_agent(branch.agent_id, worker.agent_id, deep=True)
        self.assertEqual(
            {
                "cost_tokens": {"basis": "call", "total_tokens": 123},
                "cost_usd": 1.25,
                "provider_calls": 4,
                "provider": "claude",
            },
            view["usage"],
        )

    def test_a_manager_parks_when_one_of_its_children_is_still_running(self) -> None:
        """One stopped child among several is not a deadlock.

        Refusing the park while a sibling can still report denies a legitimate
        yield and costs a turn every time round. The manager parks, wakes when
        the others finish, and is refused then, with everything settled and
        something to decide.
        """

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="own one branch",
            task_contract={},
        )
        stopped = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="the one that stopped",
            task_contract={},
        )
        running = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="the one still going",
            task_contract={},
        )
        self.control.block_agent(stopped.agent_id, "needs a decision")
        self.control.drain_wakes(branch.agent_id)
        self.control.start_turn(running.agent_id)

        self.control.await_agents(
            branch.agent_id, [stopped.agent_id, running.agent_id]
        )

        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)

    def test_a_manager_does_not_park_waiting_on_a_child_that_already_stopped(self) -> None:
        """Waiting for a child that is waiting for you is a deadlock.

        Reading the wake queue was not enough to catch it. A child can block
        while the manager's own turn is still in flight, so the manager drains
        that wake in its next prompt and then parks anyway, after the only
        event that could have released it has been consumed. Nothing moves a
        child out of BLOCKED except its manager, so the wait could never be
        satisfied and the run died as a deadlock.
        """

        branch, worker = self.branch_and_worker()
        self.control.block_agent(worker.agent_id, "needs a decision")
        self.control.drain_wakes(branch.agent_id)

        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.assertEqual(AgentStatus.READY, branch.status)
        events = [item.event_type for item in self.control.sessions["session-a"].events]
        self.assertIn("wait-skipped-settled-child", events)

    def test_a_manager_does_not_park_waiting_on_a_child_that_already_finished(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.complete_agent(worker.agent_id, {"outcome": "completed"})
        self.control.drain_wakes(branch.agent_id)

        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.assertEqual(AgentStatus.READY, branch.status)

    def test_a_manager_still_parks_when_every_watched_child_is_running(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)

        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)

    def test_active_worker_is_compactly_inspectable_and_nudgeable(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id, thread_id="thread-worker")
        self.control.record_progress(
            worker.agent_id,
            activity="editing fixture",
            progress="implemented the core behavior",
            files_touched=["fixture.txt"],
            commands=["python -m unittest"],
        )
        self.control.message_agent(branch.agent_id, worker.agent_id, "Keep the existing format.")
        compact = self.control.inspect_agent(branch.agent_id, worker.agent_id)
        self.assertNotIn("result", compact)
        self.assertEqual("editing fixture", compact["current_activity"])
        self.assertEqual("steer", compact["recent_messages"][-1]["kind"])
        self.control.message_agent(self.root.agent_id, worker.agent_id, "Do not redesign the parser.")
        override = self.control.inspect_agent(self.root.agent_id, worker.agent_id)
        self.assertTrue(override["recent_messages"][-1]["direct_override"])

    def test_user_wakes_waiting_root_without_stopping_descendants(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(self.root.agent_id)
        self.control.start_turn(worker.agent_id)
        self.control.await_agents(self.root.agent_id, [branch.agent_id])
        self.assertEqual(AgentStatus.AWAITING_WORKERS, self.root.status)
        self.control.send_user_message(self.root.agent_id, "How is it going?")
        self.assertEqual(AgentStatus.READY, self.root.status)
        self.assertEqual(AgentStatus.RUNNING, worker.status)
        self.assertEqual("user-message", self.control.drain_wakes(self.root.agent_id)[-1]["reason"])
        self.control.start_turn(self.root.agent_id)
        self.control.await_agents(self.root.agent_id, [branch.agent_id])
        self.assertEqual(AgentStatus.RUNNING, worker.status)

    def test_attested_native_followup_resumes_only_an_awaiting_bound_manager(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(branch.agent_id, thread_id="native-branch-thread")
        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.control.resume_awaiting_agent_for_native_turn(
            branch.agent_id, thread_id="native-branch-thread",
        )

        self.assertEqual(AgentStatus.READY, branch.status)
        self.assertNotIn(branch.agent_id, self.control.sessions["session-a"].waits)
        event = self.control.sessions["session-a"].events[-1]
        self.assertEqual("native-turn-resumed", event.event_type)
        self.assertEqual({"source": "attested-native-turn"}, event.metadata)
        with self.assertRaisesRegex(ProtocolError, "cannot resume native turn from ready"):
            self.control.resume_awaiting_agent_for_native_turn(
                branch.agent_id, thread_id="native-branch-thread",
            )

    def test_attested_native_followup_rejects_a_foreign_or_unregistered_wait(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(branch.agent_id, thread_id="native-branch-thread")
        self.control.await_agents(branch.agent_id, [worker.agent_id])
        with self.assertRaisesRegex(ProtocolError, "awaiting agent thread"):
            self.control.resume_awaiting_agent_for_native_turn(branch.agent_id, thread_id="foreign-thread")
        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)
        self.control.sessions["session-a"].waits.pop(branch.agent_id)
        with self.assertRaisesRegex(ProtocolError, "no registered child wait"):
            self.control.resume_awaiting_agent_for_native_turn(
                branch.agent_id, thread_id="native-branch-thread",
            )
        self.assertEqual(AgentStatus.AWAITING_WORKERS, branch.status)

    def test_runtime_turn_finish_returns_a_manager_to_ready(self) -> None:
        self.control.start_turn(self.root.agent_id, thread_id="root-thread")

        self.control.finish_turn(self.root.agent_id)

        self.assertEqual(AgentStatus.READY, self.root.status)
        self.assertIsNone(self.root.active_turn_id)
        self.assertEqual(
            "turn-finished",
            self.control.sessions["session-a"].events[-1].event_type,
        )

    def test_child_attention_wakes_only_its_awaiting_parent(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(branch.agent_id)
        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.control.request_attention(
            manager_id=branch.agent_id,
            source_agent_id=worker.agent_id,
            reason="approval-requested",
        )

        self.assertEqual(AgentStatus.READY, branch.status)
        self.assertEqual(
            {"reason": "approval-requested", "source_agent_id": worker.agent_id},
            self.control.drain_wakes(branch.agent_id)[-1],
        )
        with self.assertRaises(ProtocolError) as not_parent:
            self.control.request_attention(
                manager_id=self.root.agent_id,
                source_agent_id=worker.agent_id,
                reason="approval-requested",
            )
        self.assertEqual("not-direct-child", not_parent.exception.code)

    def test_attention_arriving_during_a_manager_turn_is_not_lost_on_await(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(branch.agent_id)

        self.control.request_attention(
            manager_id=branch.agent_id,
            source_agent_id=worker.agent_id,
            reason="approval-requested",
        )
        self.control.await_agents(branch.agent_id, [worker.agent_id])

        self.assertEqual(AgentStatus.READY, branch.status)
        self.assertEqual(
            "approval-requested",
            self.control.drain_wakes(branch.agent_id)[-1]["reason"],
        )

    def test_child_completion_wakes_each_manager_and_branch_returns_compact_evidence(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(self.root.agent_id)
        self.control.await_agents(self.root.agent_id, [branch.agent_id])
        self.control.start_turn(branch.agent_id)
        self.control.await_agents(branch.agent_id, [worker.agent_id])
        self.control.start_turn(worker.agent_id)
        self.control.complete_agent(worker.agent_id, {"evidence": ["fixture.txt hash abc"]})
        self.assertEqual(AgentStatus.READY, branch.status)
        self.assertEqual("child-completed", self.control.drain_wakes(branch.agent_id)[-1]["reason"])
        self.control.complete_branch(
            branch.agent_id,
            {"outcome": "completed", "criteria": {"verified": True}, "evidence": ["worker receipt"]},
        )
        self.assertEqual(AgentStatus.READY, self.root.status)
        compact = self.control.inspect_agent(self.root.agent_id, branch.agent_id)
        deep = self.control.inspect_agent(self.root.agent_id, branch.agent_id, deep=True)
        self.assertNotIn("result", compact)
        self.assertEqual("completed", deep["result"]["outcome"])

    def test_root_cannot_complete_while_a_branch_is_active(self) -> None:
        branch, _worker = self.branch_and_worker()
        self.control.start_turn(self.root.agent_id)

        with self.assertRaises(ProtocolError) as active:
            self.control.complete_agent(self.root.agent_id, {"decision": "accepted"})

        self.assertEqual("active-children", active.exception.code)
        self.assertEqual(AgentStatus.RUNNING, self.root.status)
        self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=branch.agent_id)
        self.control.complete_agent(self.root.agent_id, {"decision": "accepted"})
        self.assertEqual(AgentStatus.COMPLETED, self.root.status)

    def test_branch_cannot_complete_while_its_worker_is_active(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(branch.agent_id)
        self.control.start_turn(worker.agent_id)

        with self.assertRaises(ProtocolError) as active:
            self.control.complete_branch(branch.agent_id, {"outcome": "done"})

        self.assertEqual("active-children", active.exception.code)
        self.assertEqual(AgentStatus.RUNNING, branch.status)
        self.assertEqual(AgentStatus.RUNNING, worker.status)
        self.control.complete_agent(worker.agent_id, {"outcome": "done"})
        self.control.complete_branch(branch.agent_id, {"outcome": "done"})
        self.assertEqual(AgentStatus.COMPLETED, branch.status)

    def test_a_refused_replace_leaves_the_failed_child_failed(self) -> None:
        """A refusal must change nothing, including the record it refused on.

        Replace accepts a FAILED child, so the refusal for a manager that has
        already completed comes from the manager's own state -- and it used to
        come after the old attempt had been flipped to REPLACED, leaving a
        child marked replaced with nothing replacing it.
        """

        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.fail_agent(worker.agent_id, "the provider refused the model")
        self.control.complete_agent(branch.agent_id, {"outcome": "reported the failure"})
        children_before = list(branch.child_ids)

        with self.assertRaises(ProtocolError) as refused:
            self.control.replace_agent(
                requester_id=branch.agent_id,
                agent_id=worker.agent_id,
                model_id="strong-worker",
                revised_task_contract={"criteria": ["try another model"]},
            )

        self.assertEqual("terminal-agent", refused.exception.code)
        self.assertEqual(AgentStatus.FAILED, worker.status)
        self.assertIsNone(worker.replaced_by_agent_id)
        self.assertEqual(AgentStatus.COMPLETED, branch.status)
        self.assertEqual(children_before, list(branch.child_ids))

    def test_a_manager_cannot_complete_while_a_grandchild_runs_under_a_failed_child(self) -> None:
        """FAILED is terminal, and it does not stop the work underneath it.

        The completion guard read direct children only, and counted a FAILED
        child as finished. A worker that died holding a running child of its
        own therefore let its manager report the branch done while that
        grandchild was still spending a provider turn.
        """

        branch, worker = self.branch_and_worker()
        grandchild = self.control.spawn_agent(
            requester_id=worker.agent_id,
            parent_agent_id=worker.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="Sweep the fixtures",
            task_contract={"criteria": ["sweep recorded"]},
        )
        self.control.start_turn(grandchild.agent_id)
        self.control.fail_agent(worker.agent_id, "the provider turn errored")

        with self.assertRaises(ProtocolError) as active:
            self.control.complete_branch(branch.agent_id, {"outcome": "done"})

        self.assertEqual("active-children", active.exception.code)
        self.assertIn(grandchild.agent_id, str(active.exception))
        self.assertEqual(AgentStatus.READY, branch.status)

        self.control.cancel_agent(requester_id=branch.agent_id, agent_id=grandchild.agent_id)
        self.control.complete_branch(branch.agent_id, {"outcome": "done"})
        self.assertEqual(AgentStatus.COMPLETED, branch.status)

    def test_branch_manager_can_replace_worker_and_history_is_retained(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.block_agent(worker.agent_id, "capability mismatch")
        replacement = self.control.replace_agent(
            requester_id=branch.agent_id,
            agent_id=worker.agent_id,
            model_id="strong-worker",
            revised_task_contract={"criteria": ["resolve the evidenced blocker only"]},
        )
        self.assertEqual(AgentStatus.REPLACED, worker.status)
        self.assertEqual(worker.agent_id, replacement.retry_of_agent_id)
        self.assertEqual(replacement.agent_id, worker.replaced_by_agent_id)
        self.assertEqual("strong-worker", replacement.model_id)
        self.assertEqual(worker.objective, replacement.objective)

    def test_branch_manager_can_replace_a_failed_worker(self) -> None:
        """FAILED is where a model swap is the remedy, so replace accepts it.

        retry_agent already accepts BLOCKED and FAILED. replace refused every
        terminal status, which removed the one cure for a provider that will
        not run the model at all.
        """

        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.fail_agent(worker.agent_id, "HTTP 400: model requires a newer Codex")
        self.assertEqual(AgentStatus.FAILED, worker.status)

        replacement = self.control.replace_agent(
            requester_id=branch.agent_id,
            agent_id=worker.agent_id,
            model_id="strong-worker",
            revised_task_contract={"criteria": ["run on a model the provider serves"]},
        )

        self.assertEqual(AgentStatus.REPLACED, worker.status)
        self.assertEqual(worker.agent_id, replacement.retry_of_agent_id)
        self.assertEqual("strong-worker", replacement.model_id)

    def test_replace_still_refuses_a_cancelled_a_completed_and_a_replaced_worker(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.cancel_agent(requester_id=branch.agent_id, agent_id=worker.agent_id)
        self.assertEqual(AgentStatus.CANCELLED, worker.status)
        with self.assertRaises(ProtocolError) as cancelled:
            self.control.replace_agent(
                requester_id=branch.agent_id, agent_id=worker.agent_id,
                model_id="strong-worker", revised_task_contract={},
            )
        self.assertEqual("terminal-agent", cancelled.exception.code)

        done = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="finish and stop",
            task_contract={},
        )
        self.control.start_turn(done.agent_id)
        self.control.complete_agent(done.agent_id, {"outcome": "done"})
        with self.assertRaises(ProtocolError) as completed:
            self.control.replace_agent(
                requester_id=branch.agent_id, agent_id=done.agent_id,
                model_id="strong-worker", revised_task_contract={},
            )
        self.assertEqual("terminal-agent", completed.exception.code)

        second = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="be replaced once",
            task_contract={},
        )
        self.control.start_turn(second.agent_id)
        self.control.block_agent(second.agent_id, "capability mismatch")
        self.control.replace_agent(
            requester_id=branch.agent_id, agent_id=second.agent_id,
            model_id="strong-worker", revised_task_contract={},
        )
        self.assertEqual(AgentStatus.REPLACED, second.status)
        with self.assertRaises(ProtocolError) as replaced:
            self.control.replace_agent(
                requester_id=branch.agent_id, agent_id=second.agent_id,
                model_id="strong-worker", revised_task_contract={},
            )
        self.assertEqual("terminal-agent", replaced.exception.code)

    def test_replace_with_a_missing_contract_leaves_the_failed_worker_alone(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.fail_agent(worker.agent_id, "provider turn errored")
        children_before = list(branch.child_ids)

        with self.assertRaises(ProtocolError) as refused:
            self.control.replace_agent(
                requester_id=branch.agent_id,
                agent_id=worker.agent_id,
                model_id="strong-worker",
                revised_task_contract=None,  # type: ignore[arg-type]
            )

        self.assertEqual("invalid-task-contract", refused.exception.code)
        self.assertEqual(AgentStatus.FAILED, worker.status)
        self.assertIsNone(worker.replaced_by_agent_id)
        self.assertEqual(children_before, branch.child_ids)

    def test_branch_manager_can_replace_worker_with_a_new_objective(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id)
        self.control.block_agent(worker.agent_id, "capability mismatch")

        replacement = self.control.replace_agent(
            requester_id=branch.agent_id,
            agent_id=worker.agent_id,
            model_id="strong-worker",
            objective="Deliver the corrected bounded task",
            revised_task_contract={"criteria": ["resolve the evidenced blocker only"]},
        )

        self.assertEqual("Deliver the corrected bounded task", replacement.objective)

    def test_branch_manager_can_retry_same_worker_with_revised_contract(self) -> None:
        branch, worker = self.branch_and_worker()
        self.control.start_turn(worker.agent_id, thread_id="persistent-worker-thread")
        self.control.block_agent(worker.agent_id, "missing compatibility detail")
        retried = self.control.retry_agent(
            requester_id=branch.agent_id,
            agent_id=worker.agent_id,
            revised_task_contract={"criteria": ["preserve compatibility X"]},
        )
        self.assertIs(worker, retried)
        self.assertEqual(AgentStatus.READY, worker.status)
        self.assertEqual("persistent-worker-thread", worker.thread_id)
        self.assertEqual(["preserve compatibility X"], worker.task_contract["criteria"])

    def test_separate_top_level_sessions_cancel_independently(self) -> None:
        branch_a, worker_a = self.branch_and_worker()
        root_b = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=self.workspace,
            objective="Independent outcome",
            task_contract={},
            session_id="session-b",
        )
        branch_b = self.control.spawn_agent(
            requester_id=root_b.agent_id,
            parent_agent_id=root_b.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Independent branch",
            task_contract={},
        )
        self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=branch_a.agent_id)
        self.assertEqual(AgentStatus.CANCELLED, worker_a.status)
        self.assertEqual(AgentStatus.READY, branch_b.status)

    def test_an_id_no_session_has_is_a_bad_handle_and_another_sessions_is_cross_session(self) -> None:
        """R18: inspect called a mistyped id another session's agent.

        cancel_agent answered the same id invalid-handle, so a manager that
        branches on error_code got two diagnoses for one input.
        """

        root_b = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=self.workspace,
            objective="Independent outcome",
            task_contract={},
            session_id="session-b",
        )
        with self.assertRaises(ProtocolError) as raised:
            self.control.inspect_agent(self.root.agent_id, "does-not-exist")
        self.assertEqual("invalid-handle", raised.exception.code)
        with self.assertRaises(ProtocolError) as raised:
            self.control.inspect_agent(self.root.agent_id, root_b.agent_id)
        self.assertEqual("cross-session-access", raised.exception.code)

    def test_cancelling_a_bad_or_foreign_handle_names_the_mistake(self) -> None:
        """R19: the control plane's cancel_agent raised a bare KeyError."""

        root_b = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=self.workspace,
            objective="Independent outcome",
            task_contract={},
            session_id="session-b",
        )
        for target, code in (("does-not-exist", "invalid-handle"), (root_b.agent_id, "cross-session-access")):
            with self.subTest(target=target):
                with self.assertRaises(ProtocolError) as raised:
                    self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=target)
                self.assertEqual(code, raised.exception.code)
        self.assertNotEqual("cancelled", self.control.inspect_agent(root_b.agent_id, root_b.agent_id)["status"])

    def test_restoring_one_session_preserves_other_session_agent_handles(self) -> None:
        root_b = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=self.workspace,
            objective="Independent outcome",
            task_contract={},
            session_id="session-b",
        )

        self.control.restore_terminal_tree("session-a", [{
            "agent_id": self.root.agent_id,
            "parent_agent_id": None,
            "role": AgentRole.ROOT_MANAGER.value,
            "model": "root-capable",
            "status": AgentStatus.COMPLETED.value,
            "turn_id": None,
        }])

        self.assertEqual("session-b", self.control.agent_sessions[root_b.agent_id])
        self.assertEqual(AgentStatus.READY, root_b.status)
        self.assertIsInstance(self.control.start_turn(root_b.agent_id), str)

    def test_external_worker_actor_performs_real_file_task_and_reports_evidence(self) -> None:
        branch, worker = self.branch_and_worker()
        target = self.workspace / "worker-output.txt"
        self.control.start_turn(worker.agent_id)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('worker result\\n', encoding='utf-8')",
                str(target),
            ],
            check=True,
        )
        self.control.record_progress(
            worker.agent_id,
            activity="verifying output",
            progress="wrote and read back the requested file",
            files_touched=[target.name],
            commands=["external worker process"],
        )
        self.control.complete_agent(
            worker.agent_id,
            {"path": target.name, "verified_text": target.read_text(encoding="utf-8")},
        )
        evidence = self.control.inspect_agent(branch.agent_id, worker.agent_id, deep=True)
        self.assertEqual("worker result\n", evidence["result"]["verified_text"])


if __name__ == "__main__":
    unittest.main()


class UnknownModelNamesTheAlternativesTests(unittest.TestCase):
    """A refusal that repeats the rejected name alone teaches a manager nothing."""

    def setUp(self) -> None:
        self.registry = ModelRegistry(
            cards=[
                ModelCard("root-capable", frozenset({AgentRole.ROOT_MANAGER}), provider="external"),
                ModelCard("economy-worker", frozenset({AgentRole.WORKER}), provider="zai"),
                ModelCard(
                    "vendor/strong-worker", frozenset({AgentRole.WORKER}), provider="commandcode"
                ),
            ],
            presets=[
                EconomicPreset(
                    "balanced",
                    frozenset({"root-capable", "economy-worker", "vendor/strong-worker"}),
                )
            ],
        )

    def test_a_guessed_model_id_is_answered_with_the_ones_that_exist(self) -> None:
        with self.assertRaises(ProtocolError) as guessed:
            self.registry.validate_selection(
                preset_id="balanced",
                model_id="strong-worker-flash",
                role=AgentRole.WORKER,
            )
        message = str(guessed.exception)
        self.assertEqual("unknown-model", guessed.exception.code)
        self.assertIn("unknown model: strong-worker-flash", message)
        self.assertIn("vendor/strong-worker (commandcode)", message)
        self.assertIn("economy-worker (zai)", message)

    def test_the_answer_holds_only_models_the_asked_for_role_can_run(self) -> None:
        with self.assertRaises(ProtocolError) as guessed:
            self.registry.validate_selection(
                preset_id="balanced",
                model_id="invented",
                role=AgentRole.WORKER,
            )
        self.assertNotIn("root-capable", str(guessed.exception))


class ReopeningACompletedPrimaryTests(unittest.TestCase):
    """New work reopens the persistent primary conversation.

    Completing an objective ended a turn, and a root that then decided to hand
    out more work was refused with "agent is terminal: completed".  An external
    primary is the one client vNext cannot wake by itself, so that refusal left
    it holding a task it could neither delegate nor give back.  A user message
    already reopened the same conversation; delegation now reads the same way.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=Path(self.temp.name),
            objective="Deliver the requested outcome",
            task_contract={"criteria": ["verified"]},
            session_id="session-reopen",
        )
        self.session = self.control.sessions["session-reopen"]

    def _event_types(self) -> list[str]:
        return [event.event_type for event in self.session.events]

    def test_delegating_from_a_completed_root_reopens_the_conversation(self) -> None:
        self.control.complete_agent(self.root.agent_id, {"outcome": "first objective done"})
        self.assertIs(AgentStatus.COMPLETED, self.root.status)

        child = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="Perform the newly asked-for task",
            task_contract={"criteria": ["artifact written"]},
        )

        self.assertIs(AgentStatus.READY, self.root.status)
        self.assertIsNone(self.root.active_turn_id)
        self.assertEqual(self.root.agent_id, child.parent_agent_id)
        self.assertIn("conversation-resumed", self._event_types())

    def test_a_refused_delegate_leaves_the_completed_root_completed(self) -> None:
        """The reopen is a state change, so a request that is going to be
        refused must not cause one: an invalid model left the root READY with a
        resumed conversation and no child to show for it."""

        self.control.complete_agent(self.root.agent_id, {"outcome": "first objective done"})

        with self.assertRaises(ProtocolError):
            self.control.spawn_agent(
                requester_id=self.root.agent_id,
                parent_agent_id=self.root.agent_id,
                role=AgentRole.WORKER,
                model_id="no-such-model",
                objective="Work with a model this preset does not allow",
                task_contract={"criteria": ["artifact written"]},
            )

        self.assertIs(AgentStatus.COMPLETED, self.root.status)
        self.assertNotIn("conversation-resumed", self._event_types())

    def test_a_completed_child_is_still_refused(self) -> None:
        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Own one coherent branch",
            task_contract={"criteria": ["branch verified"]},
        )
        self.control.complete_agent(branch.agent_id, {"outcome": "branch done"})

        with self.assertRaises(ProtocolError) as refused:
            self.control.spawn_agent(
                requester_id=branch.agent_id,
                parent_agent_id=branch.agent_id,
                role=AgentRole.WORKER,
                model_id="economy-worker",
                objective="Work nobody asked this branch for",
                task_contract={"criteria": ["artifact written"]},
            )

        self.assertEqual("terminal-agent", refused.exception.code)
        self.assertIn("terminal: completed", str(refused.exception))
        self.assertNotIn("conversation-resumed", self._event_types())

    def test_a_blocked_descendant_is_terminal_after_its_parent_is_cancelled(self) -> None:
        """Seen live: ten rows still reading "blocked" under a cancelled parent."""

        branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Own one coherent branch",
            task_contract={"criteria": ["branch verified"]},
        )
        worker = self.control.spawn_agent(
            requester_id=branch.agent_id,
            parent_agent_id=branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="Perform one bounded task",
            task_contract={"criteria": ["artifact written"]},
        )
        self.control.block_agent(worker.agent_id, "the provider ended the turn")
        self.assertIs(AgentStatus.BLOCKED, worker.status)

        self.control.cancel_agent(requester_id=self.root.agent_id, agent_id=branch.agent_id)

        self.assertIs(AgentStatus.CANCELLED, branch.status)
        self.assertIs(AgentStatus.CANCELLED, worker.status)
        self.assertIn(worker.status, TERMINAL_STATUSES)


class BeginClosingRefusesEveryBirthOfATurnTests(unittest.TestCase):
    """The flag lives on the session record, so every route reads one answer.

    delegate, retry, replace and send_message each start a provider turn, and
    each is refused here once a close has begun.  steer, interrupt and cancel
    reach an agent that is already running and are how a close lands, so they
    keep working.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.control = OrchestrationControlPlane(registry())
        self.root = self.control.create_session(
            preset_id="balanced",
            root_model_id="root-capable",
            workspace=self.workspace,
            objective="Deliver the requested outcome",
            task_contract={"criteria": ["verified"]},
            session_id="closing-session",
        )
        self.branch = self.control.spawn_agent(
            requester_id=self.root.agent_id,
            parent_agent_id=self.root.agent_id,
            role=AgentRole.BRANCH_MANAGER,
            model_id="branch-capable",
            objective="Own one coherent branch",
            task_contract={"criteria": ["branch verified"]},
        )
        self.worker = self.control.spawn_agent(
            requester_id=self.branch.agent_id,
            parent_agent_id=self.branch.agent_id,
            role=AgentRole.WORKER,
            model_id="economy-worker",
            objective="Perform one bounded task",
            task_contract={"criteria": ["artifact written"]},
        )
        self.control.block_agent(self.worker.agent_id, "the provider ended the turn")
        self.session = self.control.sessions["closing-session"]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_begin_closing_marks_the_session_record(self) -> None:
        self.assertFalse(self.session.closing)
        self.control.begin_closing()
        self.assertTrue(self.session.closing)

    def test_a_delegate_is_refused_and_no_child_is_created(self) -> None:
        self.control.begin_closing()
        before = set(self.session.agents)
        with self.assertRaises(ProtocolError) as raised:
            self.control.spawn_agent(
                requester_id=self.branch.agent_id,
                parent_agent_id=self.branch.agent_id,
                role=AgentRole.WORKER,
                model_id="economy-worker",
                objective="A child born during the close",
                task_contract={"criteria": ["never runs"]},
            )
        self.assertEqual("session-closing", raised.exception.code)
        self.assertEqual(before, set(self.session.agents))

    def test_a_retry_is_refused_and_the_blocked_child_stays_blocked(self) -> None:
        self.control.begin_closing()
        with self.assertRaises(ProtocolError) as raised:
            self.control.retry_agent(
                requester_id=self.branch.agent_id,
                agent_id=self.worker.agent_id,
                revised_task_contract={"criteria": ["second attempt"]},
            )
        self.assertEqual("session-closing", raised.exception.code)
        self.assertIs(AgentStatus.BLOCKED, self.worker.status)
        self.assertEqual({"criteria": ["artifact written"]}, self.worker.task_contract)

    def test_a_replace_is_refused_and_the_old_attempt_is_untouched(self) -> None:
        self.control.begin_closing()
        before = set(self.session.agents)
        with self.assertRaises(ProtocolError) as raised:
            self.control.replace_agent(
                requester_id=self.branch.agent_id,
                agent_id=self.worker.agent_id,
                model_id="strong-worker",
                revised_task_contract={"criteria": ["another model"]},
            )
        self.assertEqual("session-closing", raised.exception.code)
        self.assertIs(AgentStatus.BLOCKED, self.worker.status)
        self.assertEqual(before, set(self.session.agents))

    def test_a_peer_message_is_refused_and_nothing_is_queued(self) -> None:
        self.control.begin_closing()
        with self.assertRaises(ProtocolError) as raised:
            self.control.message_agent(
                self.root.agent_id,
                self.branch.agent_id,
                "start something new",
                kind="message",
            )
        self.assertEqual("session-closing", raised.exception.code)
        self.assertEqual([], self.branch.messages)

    def test_a_steer_still_reaches_an_agent_that_is_already_running(self) -> None:
        self.control.begin_closing()
        self.control.message_agent(
            self.root.agent_id, self.branch.agent_id, "wrap up now", kind="steer"
        )
        self.assertEqual(
            ["wrap up now"], [message.text for message in self.branch.messages]
        )

    def test_a_close_cannot_land_between_the_check_and_the_child(self) -> None:
        """The refusal and the work share one lock, so nothing fits between them.

        The guard would be decorative if a close could set the flag after it
        read false and before the child existed.  It cannot: both run inside the
        session lock, and ``begin_closing`` has to take that same lock.  This
        drives a close at the narrowest point -- the validation that happens
        after the guard and before the record -- and shows it waits.
        """

        closing_returned = threading.Event()
        close_waited: list[bool] = []
        original = self.control.registry.validate_selection

        def close_from_another_thread(**kwargs):
            result = original(**kwargs)
            closer = threading.Thread(
                target=lambda: (self.control.begin_closing(), closing_returned.set()),
                name="closer",
            )
            self.addCleanup(closer.join, 5)
            closer.start()
            close_waited.append(not closing_returned.wait(1.0))
            return result

        with patch.object(
            self.control.registry, "validate_selection", close_from_another_thread
        ):
            child = self.control.spawn_agent(
                requester_id=self.branch.agent_id,
                parent_agent_id=self.branch.agent_id,
                role=AgentRole.WORKER,
                model_id="strong-worker",
                objective="A child the close raced",
                task_contract={"criteria": ["one attempt"]},
            )

        self.assertEqual(
            [True], close_waited, "a close set the flag in the middle of a spawn"
        )
        self.assertIn(child.agent_id, self.session.agents)
        self.assertTrue(closing_returned.wait(5), "the close never finished")
        self.assertTrue(self.session.closing)

    def test_a_turn_is_refused_and_the_agent_stays_ready(self) -> None:
        """Admission is the last gate before a provider is asked to work."""

        self.control.begin_closing()
        with self.assertRaises(ProtocolError) as raised:
            self.control.start_turn(self.branch.agent_id, thread_id="branch-thread")
        self.assertEqual("session-closing", raised.exception.code)
        self.assertIn("start_turn", str(raised.exception))
        self.assertIs(AgentStatus.READY, self.branch.status)
        self.assertIsNone(self.branch.active_turn_id)
        self.assertEqual(0, self.branch.turn_count)

    def test_a_close_cannot_land_between_admission_and_the_running_record(self) -> None:
        """The same proof as the spawn, at the other seam.

        A close driven from inside the locked block has to wait for the
        transition to finish, so there is no state where the flag is set and an
        admitted turn is invisible to the close.
        """

        closing_returned = threading.Event()
        close_waited: list[bool] = []
        original_emit = OrchestrationControlPlane._emit

        def close_from_another_thread(session, agent_id, event_type, metadata):
            original_emit(session, agent_id, event_type, metadata)
            if event_type != "turn-started":
                return
            closer = threading.Thread(
                target=lambda: (self.control.begin_closing(), closing_returned.set()),
                name="closer",
            )
            self.addCleanup(closer.join, 5)
            closer.start()
            close_waited.append(not closing_returned.wait(1.0))

        with patch.object(
            OrchestrationControlPlane, "_emit", staticmethod(close_from_another_thread)
        ):
            turn_id = self.control.start_turn(
                self.branch.agent_id, thread_id="branch-thread"
            )

        self.assertEqual(
            [True], close_waited, "a close set the flag in the middle of an admission"
        )
        self.assertEqual(turn_id, self.branch.active_turn_id)
        self.assertIs(AgentStatus.RUNNING, self.branch.status)
        self.assertTrue(closing_returned.wait(5), "the close never finished")
        self.assertTrue(self.session.closing)

    def test_admission_under_an_outer_lock_never_deadlocks_against_a_close(self) -> None:
        """The lock order, stated and then driven hard.

        The scheduler's own RLock is the outer lock everywhere it meets the
        session lock, and the session lock is always the inner one: the control
        plane reaches nothing that takes a scheduler lock, so there is no path
        in the other direction.  Admission is written that way too -- the
        scheduler holds its lock across the call that takes the session lock --
        and this drives the two orders against each other until both finish.
        Every outcome is one of the two legal ones, never half a transition.
        """

        scheduler_lock = threading.RLock()
        outcomes: list[str] = []
        failures: list[BaseException] = []

        for attempt in range(60):
            control = OrchestrationControlPlane(registry())
            root = control.create_session(
                preset_id="balanced",
                root_model_id="root-capable",
                workspace=self.workspace,
                objective="Deliver the requested outcome",
                task_contract={"criteria": ["verified"]},
                session_id=f"race-session-{attempt}",
            )
            start = threading.Barrier(2)

            def admit() -> None:
                try:
                    start.wait(5)
                    with scheduler_lock:
                        control.start_turn(root.agent_id, thread_id="root-thread")
                    outcomes.append("admitted")
                except ProtocolError as exc:
                    outcomes.append(exc.code)
                except BaseException as exc:  # pragma: no cover - a failure here
                    failures.append(exc)

            def close() -> None:
                try:
                    start.wait(5)
                    control.begin_closing()
                except BaseException as exc:  # pragma: no cover - a failure here
                    failures.append(exc)

            threads = [
                threading.Thread(target=admit, name="admit"),
                threading.Thread(target=close, name="close"),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
                self.assertFalse(
                    thread.is_alive(),
                    f"{thread.name} never returned: admission and the close deadlocked",
                )
            session = control.sessions[f"race-session-{attempt}"]
            self.assertTrue(session.closing)
            if root.status is AgentStatus.RUNNING:
                self.assertIsNotNone(
                    root.active_turn_id,
                    "an admitted turn left no reservation for the close to find",
                )
            else:
                self.assertIs(AgentStatus.READY, root.status)
                self.assertIsNone(
                    root.active_turn_id, "a refused turn left an active turn id"
                )

        self.assertEqual([], failures, failures)
        self.assertEqual(60, len(outcomes), outcomes)
        self.assertLessEqual(
            set(outcomes), {"admitted", "session-closing"}, outcomes
        )

    def test_a_cancel_still_stops_the_subtree(self) -> None:
        self.control.begin_closing()
        self.control.cancel_agent(
            requester_id=self.root.agent_id, agent_id=self.branch.agent_id
        )
        self.assertIs(AgentStatus.CANCELLED, self.branch.status)
        self.assertIs(AgentStatus.CANCELLED, self.worker.status)
