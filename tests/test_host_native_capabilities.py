"""Service snapshots preserve adopted native-child capability limits."""

from __future__ import annotations

import tempfile
import threading
import unittest
from types import SimpleNamespace

from vnext.host_contract import SessionStartRequest
from vnext.vnext_orchestration import AgentRole
from vnext.vnext_session_runtime import VNextRuntimeSession


class NativeCapabilitySnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.TemporaryDirectory()
        self.runtime = VNextRuntimeSession(
            SessionStartRequest(
                "native-capabilities",
                self.workspace.name,
                "primary",
                {"provider": "codex", "model": "codex-model"},
            ),
            lambda _event: None,
        )
        self.runtime.start()
        self.child = self.runtime.control.spawn_agent(
            requester_id="primary",
            parent_agent_id="primary",
            role=AgentRole.WORKER,
            model_id="codex-model",
            objective="provider-created child",
            task_contract={},
        )
        self.scheduler = SimpleNamespace(
            _lock=threading.RLock(),
            _native_child_delivery_contracts={},
        )
        self.managed = SimpleNamespace(_lock=threading.RLock(), _handlers={})
        self.runtime._scheduler = self.scheduler
        self.runtime._managed = self.managed

    def tearDown(self) -> None:
        # These snapshots use small read-only scheduler/managed seams rather
        # than a started provider runtime; restore the ordinary no-thread close
        # path before asking the session to shut down.
        self.runtime._scheduler = None
        self.runtime._managed = None
        self.runtime.close()
        self.workspace.cleanup()

    def test_unattested_native_child_operations_are_not_advertised(self) -> None:
        self.scheduler._native_child_delivery_contracts[self.child.agent_id] = {
            "context_messages": "unavailable",
            "interrupt": "unavailable",
        }
        self.managed._handlers[self.child.agent_id] = None

        capabilities = self.runtime._agent_view(self.child)["capabilities"]

        self.assertEqual("unavailable", capabilities["prompt"])
        self.assertEqual("unavailable", capabilities["peer_message"])
        self.assertEqual("unavailable", capabilities["interrupt"])
        self.assertEqual("unavailable", capabilities["delegate"])

    def test_attested_native_child_operations_are_advertised(self) -> None:
        self.scheduler._native_child_delivery_contracts[self.child.agent_id] = {
            "context_messages": "available",
            "interrupt": "available",
        }
        self.managed._handlers[self.child.agent_id] = lambda _tool, _arguments, _context: None

        capabilities = self.runtime._agent_view(self.child)["capabilities"]

        self.assertEqual("available", capabilities["prompt"])
        self.assertEqual("available", capabilities["peer_message"])
        self.assertEqual("available", capabilities["interrupt"])
        self.assertEqual("available", capabilities["delegate"])

    def test_native_lineage_keeps_exact_ids_and_excludes_adapter_private_state(self) -> None:
        self.child.thread_id = "runtime-child-key"
        identity = {
            "bound": True, "provider_session": "actual-session-uuid",
            "origin": "native", "native_agent_id": "provider-agent-id",
            "native_task_id": "provider-task-id",
            "parent_runtime_thread_id": "parent-runtime-key",
            "parent_native_turn_id": "parent-turn-id",
            "parent_local_turn_reference": "local-parent-turn", "parent_turn_source": "bridge",
            "parent_tool_use_id": "parent-tool-id", "private_token": "do-not-project",
        }
        self.runtime._adapters["codex"] = SimpleNamespace(
            thread_identity_attestation=lambda _thread: identity)
        view = self.runtime._agent_view(self.child)
        self.assertEqual("actual-session-uuid", view["native_session_id"])
        self.assertEqual("runtime-child-key", view["runtime_thread_id"])
        self.assertEqual({key: value for key, value in identity.items()
                          if key not in {"bound", "provider_session", "private_token"}},
                         view["native_identity"])
        identity["bound"] = False
        view = self.runtime._agent_view(self.child)
        self.assertIsNone(view["native_session_id"])
        self.assertIsNone(view["native_identity"])
        self.runtime._adapters.clear()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
