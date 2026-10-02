import json
import threading
import time
import types
import unittest
from pathlib import Path

from vnext.vnext_app_server import (
    TurnHandle,
    VNextAppServerAdapter,
    VNextAppServerError,
    effective_thread_policy,
    function_tool,
    require_workspace_policy,
)
from vnext.vnext_runtime import (
    RuntimePolicyError,
    require_workspace_policy as require_neutral_workspace_policy,
)
from vnext.vnext_runtime_types import NativeChildBinding, RuntimePosture


class _ClosedInput:
    def write(self, value):
        del value
        raise ValueError("I/O operation on closed file")

    def flush(self):
        raise AssertionError("flush must not follow a failed write")


class _LiveProcessWithClosedInput:
    stdin = _ClosedInput()

    @staticmethod
    def poll():
        return None


class VNextAppServerShutdownTests(unittest.TestCase):
    @staticmethod
    def _native_child_adapter() -> VNextAppServerAdapter:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.provider = "codex"
        adapter._condition = threading.Condition(threading.RLock())
        adapter._events = []
        adapter._tool_handlers = {}
        adapter._native_thread_attestations = {
            "parent-thread": {
                "provider": "codex",
                "provider_session": "parent-thread",
                "binding_phase": "started",
            }
        }
        adapter._native_child_observer = None
        adapter._native_parent_agent_resolver = None
        adapter._native_child_observations = {}
        adapter._native_child_bindings = {}
        adapter._native_child_hints = {}
        adapter._native_child_start_events = {}
        adapter._native_child_failures = {}
        adapter._native_child_discovery_in_flight = set()
        return adapter

    def test_a_child_call_waits_for_a_discovery_already_running_for_its_parent(self) -> None:
        """R19 claude-code F2: a child's call waits for a discovery already in flight.

        The background poll holds a parent's in-flight mark for its whole
        thread/list round trip, and a slow app-server makes that window long.
        A child's first MCP call that lands inside it was refused at once,
        before the running discovery or a fresh one could find the child.
        """

        handler = lambda _tool, _arguments, _context: {"ok": True}

        def build(*, in_flight: bool) -> VNextAppServerAdapter:
            adapter = self._native_child_adapter()
            adapter.set_native_child_observer(
                lambda observation: NativeChildBinding(
                    agent_id="managed-child",
                    provider="codex",
                    native_thread_id=observation.native_child_thread_id,
                    delivery_contract={"context_messages": "available", "interrupt": "available"},
                    tool_handler=handler,
                ),
                parent_agent_resolver=lambda thread: "managed-parent" if thread == "parent-thread" else None,
            )
            adapter._tool_handlers["child-thread"] = handler
            adapter.request = lambda method, params, timeout=30: {
                "data": [{"id": "child-thread", "parentThreadId": "parent-thread", "model": "gpt-5.6-terra", "preview": "do the thing"}]
            }
            adapter.read_thread = lambda thread_id, **_kwargs: {
                "thread": {"id": thread_id, "turns": [{"id": "child-turn", "status": "inProgress"}]}
            }
            if in_flight:
                adapter._native_child_discovery_in_flight.add("parent-thread")
            return adapter

        def call(adapter: VNextAppServerAdapter) -> object:
            return adapter.dispatch_native_mcp_call(
                tool="complete_agent", arguments={}, metadata={"threadId": "child-thread", "itemId": "call-1"}
            )

        self.assertEqual({"ok": True}, call(build(in_flight=False)))

        # The running discovery ends without having seen the child: it began
        # before the child existed.  The call then looks for itself.
        adapter = build(in_flight=True)

        def finish_discovery() -> None:
            time.sleep(0.2)
            with adapter._condition:
                adapter._native_child_discovery_in_flight.discard("parent-thread")
                adapter._condition.notify_all()

        finisher = threading.Thread(target=finish_discovery)
        finisher.start()
        try:
            self.assertEqual({"ok": True}, call(adapter))
        finally:
            finisher.join()

    def test_attested_native_child_binds_once_and_routes_its_tool_handler(self) -> None:
        adapter = self._native_child_adapter()
        received = []
        child_handler = lambda _tool, _arguments, _context: {"ok": True}

        def observer(observation):
            received.append(observation)
            return NativeChildBinding(
                agent_id="managed-child",
                provider="codex",
                native_thread_id=observation.native_child_thread_id,
                delivery_contract={"context_messages": "available", "interrupt": "available"},
                tool_handler=child_handler,
            )

        adapter.set_native_child_observer(observer, parent_agent_resolver=lambda thread: "managed-parent" if thread == "parent-thread" else None)
        adapter._observe_native_child_event(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "parent-thread",
                    "turnId": "parent-turn",
                    "item": {
                        "type": "collabAgentToolCall",
                        "tool": "spawnAgent",
                        "receiverThreadIds": ["child-thread"],
                        "senderThreadId": "parent-thread",
                        "model": "gpt-5.6-luna",
                        "reasoningEffort": "high",
                        "prompt": "write the focused implementation",
                    },
                },
            },
            4,
        )
        event = {
            "method": "thread/started",
            "params": {
                "thread": {
                    "id": "child-thread",
                    "parentThreadId": "parent-thread",
                    "source": {"thread_spawn": {"parent_thread_id": "parent-thread", "agent_path": "child-path"}},
                    "model": "gpt-5.6-luna",
                    "agentRole": "worker",
                    "reasoningEffort": "high",
                }
            },
        }
        adapter._observe_native_child_event(event, 5)
        adapter._observe_native_child_event(event, 6)

        self.assertEqual(1, len(received))
        observed = received[0]
        self.assertTrue(observed.attested)
        self.assertEqual("managed-parent", observed.parent_agent_id)
        self.assertEqual("parent-turn", observed.parent_native_turn_id)
        self.assertEqual("write the focused implementation", observed.objective)
        self.assertEqual("available", observed.delivery_contract["interrupt"])
        self.assertIs(child_handler, adapter._tool_handlers["child-thread"])
        attestation = adapter.native_child_attestations()[0]
        self.assertEqual("managed-child", attestation["agent_id"])
        self.assertEqual("native", attestation["origin"])
        self.assertEqual("parent-thread", attestation["parent_runtime_thread_id"])
        self.assertEqual("parent-turn", attestation["parent_native_turn_id"])
        self.assertEqual("available", attestation["delivery_contract"]["context_messages"])
        identity = adapter.thread_identity_attestation("child-thread")
        self.assertEqual("child-thread", identity["provider_session"])
        self.assertEqual("native", identity["origin"])
        self.assertEqual("parent-thread", identity["parent_runtime_thread_id"])
        self.assertEqual("parent-turn", identity["parent_native_turn_id"])

    def test_native_child_requires_an_attested_parent_and_matching_source_edge(self) -> None:
        adapter = self._native_child_adapter()
        received = []
        adapter.set_native_child_observer(
            lambda observation: received.append(observation) or NativeChildBinding(
                agent_id="managed-child", provider="codex", native_thread_id=observation.native_child_thread_id,
                delivery_contract={"context_messages": "available", "interrupt": "available"},
            ),
            parent_agent_resolver=lambda _thread: "managed-parent",
        )
        adapter._observe_native_child_event(
            {"method": "thread/started", "params": {"thread": {"id": "foreign-child", "parentThreadId": "foreign-parent"}}},
            0,
        )
        adapter._observe_native_child_event(
            {
                "method": "thread/started",
                "params": {"thread": {
                    "id": "conflict-child", "parentThreadId": "parent-thread",
                    "source": {"thread_spawn": {"parent_thread_id": "different-parent"}},
                }},
            },
            1,
        )

        self.assertEqual([], received)
        self.assertNotIn("foreign-child", adapter._native_thread_attestations)
        self.assertEqual("provider child source conflicts with parent thread", adapter._native_child_failures["conflict-child"])

    def test_listed_native_child_uses_provider_preview_and_normalizes_provider_role(self) -> None:
        adapter = self._native_child_adapter()
        received = []
        adapter.set_native_child_observer(
            lambda observation: received.append(observation) or NativeChildBinding(
                agent_id="managed-child", provider="codex", native_thread_id=observation.native_child_thread_id,
                delivery_contract={"context_messages": "available", "interrupt": "available"},
            ),
            parent_agent_resolver=lambda thread: "managed-parent" if thread == "parent-thread" else None,
        )
        adapter.request = lambda method, params, timeout=30: {
            "data": [{
                "id": "listed-child",
                "parentThreadId": "parent-thread",
                "model": "gpt-5.6-terra",
                "agentRole": "explorer",
                "preview": "inspect the focused module",
            }]
        }

        self.assertEqual(1, adapter.discover_native_children("parent-thread"))
        self.assertEqual(1, len(received))
        observed = received[0]
        self.assertEqual("worker", observed.role)
        self.assertEqual("explorer", observed.capabilities["native_role"])
        self.assertEqual("inspect the focused module", observed.objective)
        self.assertEqual("managed-child", adapter.native_child_attestations()[0]["agent_id"])

    def test_a_listed_child_whose_preview_is_blank_is_refused_for_that(self) -> None:
        """R20 gpt-code F5: whitespace passed as an objective."""

        adapter = self._native_child_adapter()
        received = []
        adapter.set_native_child_observer(
            lambda observation: received.append(observation) or NativeChildBinding(
                agent_id="managed-child", provider="codex", native_thread_id=observation.native_child_thread_id,
                delivery_contract={"context_messages": "available", "interrupt": "available"},
            ),
            parent_agent_resolver=lambda thread: "managed-parent" if thread == "parent-thread" else None,
        )
        adapter.request = lambda method, params, timeout=30: {
            "data": [{
                "id": "blank-child",
                "parentThreadId": "parent-thread",
                "model": "gpt-5.6-terra",
                "preview": " \n  ",
            }]
        }

        adapter.discover_native_children("parent-thread")
        self.assertEqual([], received)
        self.assertEqual(
            "native child objective is unavailable", adapter._native_child_failures["blank-child"]
        )

    def test_cancel_native_child_reads_the_one_current_turn_before_interrupting(self) -> None:
        adapter = self._native_child_adapter()
        adapter._native_child_observations = {
            ("codex", "parent-thread", "child-thread"): types.SimpleNamespace(
                native_child_thread_id="child-thread"
            )
        }
        calls = []
        adapter.read_thread = lambda thread_id, **_kwargs: {
            "thread": {
                "id": thread_id,
                "turns": [
                    {"id": "finished-turn", "status": "completed"},
                    {"id": "active-turn", "status": "inProgress"},
                ],
            }
        }
        adapter.request = lambda method, params, timeout=30: calls.append((method, params, timeout)) or {}

        result = adapter.cancel_native_child("child-thread", timeout=12)

        self.assertEqual("interrupt-requested", result["status"])
        self.assertEqual("active-turn", result["turn_id"])
        self.assertEqual(
            [("turn/interrupt", {"threadId": "child-thread", "turnId": "active-turn"}, 12)],
            calls,
        )

    def test_active_native_child_turn_uses_only_the_fresh_provider_turn(self) -> None:
        adapter = self._native_child_adapter()
        adapter._native_child_observations = {
            ("codex", "parent-thread", "child-thread"): types.SimpleNamespace(
                native_child_thread_id="child-thread"
            )
        }
        adapter.read_thread = lambda thread_id, **_kwargs: {
            "thread": {"id": thread_id, "turns": [{"id": "active-turn", "status": "inProgress"}]}
        }

        handle = adapter.active_native_child_turn("child-thread", timeout=12)

        self.assertIsNotNone(handle)
        self.assertEqual("child-thread", handle.thread_id)
        self.assertEqual("active-turn", handle.turn_id)

    def test_cancel_native_child_fails_closed_for_unknown_or_completed_children(self) -> None:
        adapter = self._native_child_adapter()
        with self.assertRaisesRegex(VNextAppServerError, "not attested"):
            adapter.cancel_native_child("unknown-child")

        adapter._native_child_observations = {
            ("codex", "parent-thread", "child-thread"): types.SimpleNamespace(
                native_child_thread_id="child-thread"
            )
        }
        adapter.read_thread = lambda thread_id, **_kwargs: {
            "thread": {"id": thread_id, "turns": [{"id": "finished", "status": "completed"}]}
        }
        self.assertEqual(
            {"status": "not-running", "thread_id": "child-thread"},
            adapter.cancel_native_child("child-thread"),
        )
    def test_fresh_thread_identity_uses_start_acknowledgement_before_rollout_exists(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.provider = "codex"
        adapter._condition = threading.Condition(threading.RLock())
        adapter._native_thread_attestations = {
            "fresh-thread": {
                "provider": "codex",
                "provider_session": "fresh-thread",
                "binding_phase": "started",
            }
        }
        adapter.read_thread = lambda *_args, **_kwargs: self.fail("fresh thread must not be read before its rollout exists")

        identity = adapter.thread_identity_attestation("fresh-thread")

        self.assertTrue(identity["bound"])
        self.assertEqual("fresh-thread", identity["provider_session"])
        self.assertEqual("started", identity["binding_phase"])

    def test_provider_read_attestation_is_retained_for_native_child_discovery(self) -> None:
        adapter = self._native_child_adapter()
        adapter._native_thread_attestations = {}
        adapter.read_thread = lambda *_args, **_kwargs: {"thread": {"id": "parent-thread"}}
        received = []
        adapter.set_native_child_observer(
            lambda observation: received.append(observation) or NativeChildBinding(
                agent_id="managed-child",
                provider="codex",
                native_thread_id=observation.native_child_thread_id,
                delivery_contract={"context_messages": "available", "interrupt": "available"},
            ),
            parent_agent_resolver=lambda thread: "managed-parent" if thread == "parent-thread" else None,
        )

        identity = adapter.thread_identity_attestation("parent-thread")
        adapter._observe_native_child_event(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "parent-thread",
                        "model": "gpt-5.6-luna",
                        "preview": "inspect the focused module",
                    }
                },
            },
            0,
        )

        self.assertEqual("attested", identity["binding_phase"])
        self.assertEqual("attested", adapter._native_thread_attestations["parent-thread"]["binding_phase"])
        self.assertEqual(["child-thread"], [item.native_child_thread_id for item in received])

    def test_dynamic_tool_registration_and_call_are_attested_separately(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.workspace = Path.cwd()
        adapter.codex_home = Path.cwd()
        adapter.client_name = "vnext_tool_probe_test"
        adapter._condition = threading.Condition(threading.RLock())
        adapter._tool_handlers = {}
        adapter._tool_registrations = {}
        adapter._native_thread_attestations = {}
        adapter._tool_calls = []
        adapter._closing = False
        adapter.process = _LiveProcessWithClosedInput()
        sent = []
        adapter._send = sent.append
        captured = []

        def request(method, params, *, timeout):
            captured.append({"method": method, "params": params, "timeout": timeout})
            if method == "environment/status":
                return {"status": "ready"}
            if method == "environment/info":
                return {"shell": {"name": "powershell", "path": "powershell.exe"}}
            return {"thread": {"id": "luna-thread"}}

        adapter.request = request
        tool = function_tool(
            "run_command",
            "Diagnostic copy of the product command tool.",
            properties={
                "argv": {"type": "array", "items": {"type": "string"}},
                "justification": {"type": "string"},
            },
            required=["argv", "justification"],
        )
        handled = []

        thread_id, _ = adapter.start_thread(
            model="gpt-5.6-luna",
            effort="ultra",
            developer_instructions="bounded diagnostic",
            tools=[tool],
            tool_handler=lambda name, arguments, context: (
                handled.append((name, dict(arguments), context)) or {"ok": True}
            ),
            requested_posture=RuntimePosture(
                workspace_writes=True,
                network="restricted",
                approvals_requested=True,
                reviewer="user",
                environment_ready=True,
            ),
        )

        registration = adapter.tool_registration_attestation(thread_id)
        self.assertEqual("thread/start", captured[0]["method"])
        self.assertEqual([tool], captured[0]["params"]["dynamicTools"])
        self.assertEqual("on-request", captured[0]["params"]["approvalPolicy"])
        self.assertEqual("user", captured[0]["params"]["approvalsReviewer"])
        self.assertNotIn("effort", captured[0]["params"], "Codex attests effort at turn start")
        self.assertNotIn("environments", captured[0]["params"])
        self.assertEqual("environment/status", captured[1]["method"])
        self.assertEqual({"environmentId": "local"}, captured[1]["params"])
        self.assertEqual("environment/info", captured[2]["method"])
        self.assertEqual({"environmentId": "local"}, captured[2]["params"])
        self.assertTrue(registration["acknowledged"])
        self.assertTrue(registration["handler_registered"])
        self.assertEqual(1, registration["tool_count"])
        self.assertEqual(["run_command"], registration["tool_names"])
        self.assertEqual(64, len(registration["definition_sha256"]))

        adapter._handle_server_request(
            {
                "id": 17,
                "method": "item/tool/call",
                "params": {
                    "threadId": thread_id,
                    "turnId": "turn-1",
                    "callId": "call-1",
                    "tool": "run_command",
                    "arguments": {"argv": ["python", "-c", "pass"], "justification": "probe"},
                },
            }
        )

        calls = adapter.tool_call_attestations(thread_id)
        self.assertEqual(1, len(calls))
        self.assertEqual("run_command", calls[0]["tool"])
        self.assertTrue(calls[0]["handler_registered"])
        self.assertTrue(calls[0]["turn_correlated"])
        self.assertTrue(calls[0]["call_correlated"])
        self.assertEqual(1, len(handled))
        self.assertEqual(0, handled[0][2].cursor)
        self.assertEqual(17, sent[-1]["id"])


        # A native child must never inherit this host handler by accidental
        # singleton fallback. Provider-side inheritance, if supported, is
        # independently attested when the child is adopted.
        adapter._handle_server_request(
            {
                "id": 18,
                "method": "item/tool/call",
                "params": {
                    "threadId": "unbound-native-child",
                    "turnId": "child-turn-1",
                    "callId": "child-call-1",
                    "tool": "run_command",
                    "arguments": {},
                },
            }
        )
        self.assertEqual(1, len(handled))
        self.assertEqual(18, sent[-1]["id"])
        self.assertEqual(-32000, sent[-1]["error"]["code"])
        self.assertFalse(adapter.tool_call_attestations("unbound-native-child")[0]["handler_registered"])

    def test_turn_omits_environment_override_after_windows_setup(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.workspace = Path.cwd()
        adapter._condition = threading.Condition(threading.RLock())
        adapter._events = []
        order = []

        def ensure_windows_sandbox_ready(*, workspace):
            order.append(("setup", workspace))

        def request(method, params, *, timeout):
            order.append((method, dict(params), timeout))
            return {"turn": {"id": "turn-1"}}

        adapter.ensure_windows_sandbox_ready = ensure_windows_sandbox_ready
        adapter.request = request

        handle = adapter.start_turn(
            thread_id="thread-1",
            prompt="bounded task",
            model="worker",
            effort="high",
        )

        self.assertEqual("turn-1", handle.turn_id)
        self.assertEqual("setup", order[0][0])
        self.assertEqual("turn/start", order[1][0])
        self.assertNotIn("environments", order[1][1])

    @staticmethod
    def _windows_sandbox_adapter() -> VNextAppServerAdapter:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.workspace = Path.cwd()
        adapter._platform_name = "nt"
        adapter._windows_sandbox_lock = threading.Lock()
        adapter._windows_sandbox_ready = False
        adapter._windows_sandbox_attestation = {
            "platform_windows": True,
            "feature": "elevated_windows_sandbox",
            "feature_enabled": None,
            "setup_attempted": False,
            "setup_completed": False,
            "skip_reason": None,
        }
        return adapter

    def test_windows_sandbox_setup_is_bounded_and_runs_once(self) -> None:
        adapter = self._windows_sandbox_adapter()
        calls = []
        adapter.event_cursor = lambda: 7

        def request(method, params, *, timeout):
            calls.append((method, dict(params), timeout))
            if method == "experimentalFeature/list":
                return {
                    "data": [
                        {"name": "shell_tool", "enabled": True},
                        {"name": "elevated_windows_sandbox", "enabled": True},
                    ]
                }
            return {"started": True}

        def wait_event(predicate, *, cursor, timeout):
            calls.append(("wait", cursor, timeout))
            event = {
                "method": "windowsSandbox/setupCompleted",
                "params": {"success": True, "mode": "elevated"},
            }
            self.assertTrue(predicate(event))
            return event

        adapter.request = request
        adapter.wait_event = wait_event

        adapter.ensure_windows_sandbox_ready(timeout=23)
        adapter.ensure_windows_sandbox_ready(timeout=23)

        self.assertEqual("experimentalFeature/list", calls[0][0])
        self.assertEqual("windowsSandbox/setupStart", calls[1][0])
        self.assertEqual("elevated", calls[1][1]["mode"])
        self.assertEqual(str(Path.cwd().resolve()), calls[1][1]["cwd"])
        self.assertEqual(15, calls[1][2])
        self.assertEqual(("wait", 7, 23), calls[2])
        self.assertEqual(3, len(calls))
        attestation = adapter.windows_sandbox_attestation()
        self.assertTrue(attestation["feature_enabled"])
        self.assertTrue(attestation["setup_attempted"])
        self.assertTrue(attestation["setup_completed"])
        self.assertIsNone(attestation["skip_reason"])

    def test_windows_sandbox_setup_is_skipped_when_the_feature_is_disabled(self) -> None:
        adapter = self._windows_sandbox_adapter()
        calls = []
        adapter.event_cursor = lambda: 7

        def request(method, params, *, timeout):
            calls.append((method, dict(params), timeout))
            if method == "experimentalFeature/list":
                return {
                    "data": [
                        {"name": "shell_tool", "enabled": True},
                        {"name": "elevated_windows_sandbox", "enabled": False},
                    ]
                }
            raise AssertionError(f"unexpected request while the feature is disabled: {method}")

        adapter.request = request
        adapter.wait_event = lambda *_args, **_kwargs: self.fail("no setup event is expected")

        adapter.ensure_windows_sandbox_ready(timeout=23)
        adapter.ensure_windows_sandbox_ready(timeout=23)

        self.assertEqual(["experimentalFeature/list"], [call[0] for call in calls])
        attestation = adapter.windows_sandbox_attestation()
        self.assertFalse(attestation["feature_enabled"])
        self.assertFalse(attestation["setup_attempted"])
        self.assertFalse(attestation["setup_completed"])
        self.assertEqual("runtime reports the feature disabled", attestation["skip_reason"])

    def test_windows_sandbox_setup_is_skipped_when_the_feature_is_absent(self) -> None:
        adapter = self._windows_sandbox_adapter()
        adapter.event_cursor = lambda: 0
        adapter.request = lambda method, params, *, timeout: {"data": []}
        adapter.wait_event = lambda *_args, **_kwargs: self.fail("no setup event is expected")

        adapter.ensure_windows_sandbox_ready(timeout=23)

        self.assertFalse(adapter.windows_sandbox_attestation()["feature_enabled"])

    def test_workspace_policy_requires_selected_ready_environment(self) -> None:
        result = {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "auto_review",
            "activePermissionProfile": {"id": ":workspace"},
            "sandbox": {"type": "workspaceWrite", "networkAccess": False},
            "environmentSelection": {
                "environmentId": "local",
                "selected": True,
                "status": "ready",
                "shellAvailable": True,
            },
        }

        require_workspace_policy(result)
        policy = effective_thread_policy(result)
        self.assertEqual("local", policy["environment_id"])
        self.assertTrue(policy["environment_selected"])
        self.assertEqual("ready", policy["environment_status"])
        self.assertTrue(policy["environment_shell_available"])

        for environment in (
            {},
            {"environmentId": "local", "selected": False, "status": "ready"},
            {"environmentId": "local", "selected": True, "status": "pending"},
            {
                "environmentId": "local",
                "selected": True,
                "status": "ready",
                "shellAvailable": False,
            },
        ):
            with self.subTest(environment=environment):
                invalid = dict(result)
                invalid["environmentSelection"] = environment
                with self.assertRaises(VNextAppServerError):
                    require_workspace_policy(invalid)

    def test_neutral_posture_unknown_values_fail_closed(self) -> None:
        baseline = {
            "posture": {
                "workspace_writes": True,
                "network": "restricted",
                "approvals_requested": True,
                "reviewer": "auto_review",
                "environment_ready": True,
            }
        }

        require_neutral_workspace_policy(baseline, approvals_reviewer="auto_review")
        for field, value in (
            ("workspace_writes", None),
            ("network", "unknown"),
            ("approvals_requested", None),
            ("reviewer", ""),
            ("environment_ready", None),
        ):
            with self.subTest(field=field):
                candidate = {"posture": dict(baseline["posture"])}
                candidate["posture"][field] = value
                with self.assertRaises(RuntimePolicyError):
                    require_neutral_workspace_policy(candidate, approvals_reviewer="auto_review")

    def test_environment_info_proves_ready_shell_on_legacy_runtime(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        calls = []

        def request(method, params, *, timeout):
            calls.append((method, dict(params), timeout))
            if method == "environment/status":
                raise VNextAppServerError(
                    "environment/status failed: {'code': -32601, 'message': 'unsupported'}"
                )
            return {"shell": {"name": "powershell", "path": "powershell.exe"}}

        adapter.request = request
        result = adapter._attest_default_environment(
            {"thread": {"id": "legacy-thread"}},
            timeout=60,
        )

        self.assertEqual("ready", result["environmentSelection"]["status"])
        self.assertTrue(result["environmentSelection"]["shellAvailable"])
        self.assertEqual(["environment/status", "environment/info"], [value[0] for value in calls])

    def test_wait_turn_carries_only_the_matching_final_agent_message(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter._condition = threading.Condition()
        adapter._fatal = None
        recorded_turns = []

        class Transcript:
            @staticmethod
            def record_turn(handle, events):
                recorded_turns.append((handle, tuple(events)))

        adapter.debug_transcript = Transcript()
        matching = {
            "method": "item/completed",
            "params": {
                "threadId": "worker-thread",
                "turnId": "worker-turn",
                "item": {"type": "agentMessage", "text": "bounded worker report"},
            },
        }
        adapter._events = [
            {
                "method": "item/completed",
                "params": {
                    "threadId": "branch-thread",
                    "turnId": "branch-turn",
                    "item": {"type": "agentMessage", "text": "unrelated branch report"},
                },
            },
            matching,
            {
                "method": "turn/completed",
                "params": {"turn": {"id": "worker-turn", "status": "completed"}},
            },
        ]

        result = adapter.wait_turn(TurnHandle("worker-thread", "worker-turn", 0), timeout=0.1)

        self.assertEqual("completed", result["status"])
        self.assertEqual("bounded worker report", result["final_response"])
        self.assertEqual(1, len(recorded_turns))
        self.assertEqual("worker-turn", recorded_turns[0][0].turn_id)
        self.assertEqual(1, len(recorded_turns[0][1]))
        scoped_event = recorded_turns[0][1][0]
        self.assertTrue(scoped_event.is_current_turn)
        self.assertEqual(matching, scoped_event.event)

    def test_native_approval_handler_declines_without_json_rpc_error(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter._condition = threading.Condition()
        adapter._handler_threads = set()
        adapter._native_thread_attestations = {
            "thread": {"provider": "codex", "provider_session": "thread"}
        }
        adapter._closing = False
        sent = []
        recorded = []
        adapter._send = sent.append

        def decline(method, params):
            recorded.append((method, dict(params)))
            return {"decision": "decline"}

        adapter.native_approval_handler = decline
        for request_id, method in enumerate(
            (
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
                "item/permissions/requestApproval",
            ),
            start=1,
        ):
            adapter._handle_server_request(
                {
                    "id": request_id,
                    "method": method,
                    "params": {
                        "threadId": "thread",
                        "turnId": "turn",
                        "itemId": f"item-{request_id}",
                    },
                }
            )

        self.assertEqual(3, len(recorded))
        self.assertTrue(all(method == "approval/request" for method, _params in recorded))
        self.assertTrue(
            all("provider_correlation" in params for _method, params in recorded)
        )
        self.assertTrue(
            all(
                not {"threadId", "turnId", "itemId"}.intersection(params)
                for _method, params in recorded
            )
        )
        self.assertTrue(all("error" not in response for response in sent))
        self.assertEqual({"decision": "decline"}, sent[0]["result"])
        self.assertEqual({"decision": "decline"}, sent[1]["result"])
        self.assertEqual(
            {"scope": "turn", "permissions": {}},
            sent[2]["result"],
        )

    def test_native_approval_without_attested_correlation_is_not_routed(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter._condition = threading.Condition()
        adapter._handler_threads = set()
        adapter._native_thread_attestations = {}
        adapter._closing = False
        sent = []
        recorded = []
        adapter._send = sent.append
        adapter.native_approval_handler = lambda method, params: recorded.append(
            (method, dict(params))
        ) or {"decision": "accept"}

        adapter._handle_server_request(
            {
                "id": 1,
                "method": "item/commandExecution/requestApproval",
                "params": {"threadId": "unbound", "turnId": "turn", "itemId": "item"},
            }
        )

        self.assertEqual([], recorded)
        self.assertEqual({"decision": "decline"}, sent[0]["result"])

    def test_native_approval_with_blank_or_malformed_correlation_is_not_routed(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter._condition = threading.Condition()
        adapter._handler_threads = set()
        adapter._native_thread_attestations = {
            "thread": {"provider": "codex", "provider_session": "thread"}
        }
        adapter._closing = False
        sent = []
        recorded = []
        adapter._send = sent.append
        adapter.native_approval_handler = lambda method, params: recorded.append(
            (method, dict(params))
        ) or {"decision": "accept"}

        for request_id, params in enumerate(
            (
                {"threadId": " thread", "turnId": "turn", "itemId": "item"},
                {"threadId": "thread", "turnId": " ", "itemId": "item"},
                {"threadId": "thread", "turnId": "turn", "itemId": ""},
            ),
            start=1,
        ):
            adapter._handle_server_request(
                {
                    "id": request_id,
                    "method": "item/commandExecution/requestApproval",
                    "params": params,
                }
            )

        self.assertEqual([], recorded)
        self.assertEqual(
            [{"decision": "decline"}] * 3,
            [response["result"] for response in sent],
        )

    def test_late_tool_reply_is_discarded_after_stdin_closes(self) -> None:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.process = _LiveProcessWithClosedInput()
        adapter._send_lock = threading.Lock()
        adapter._condition = threading.Condition()
        adapter._tool_handlers = {"thread": lambda tool, arguments, context: {"ok": True}}
        adapter._closing = True
        adapter.native_approval_handler = None

        adapter._handle_server_request(
            {
                "id": 7,
                "method": "item/tool/call",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn",
                    "callId": "call",
                    "tool": "read_file",
                    "arguments": {"path": "fixture.py"},
                },
            }
        )


class TurnWaitIsAnIdleBudgetTests(unittest.TestCase):
    """A working turn must not be abandoned for taking a long time.

    Five workers were blocked mid-answer because the turn wait was wall-clock:
    ten minutes after it started it gave up, while the model was still running
    commands on the other side of the same socket.
    """

    @staticmethod
    def _adapter() -> VNextAppServerAdapter:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter._events = []
        adapter._condition = threading.Condition()
        adapter._fatal = None
        adapter._closing = False
        return adapter

    def test_activity_on_the_same_turn_pushes_the_deadline_out(self):
        adapter = self._adapter()
        handle = TurnHandle(thread_id="t1", turn_id="u1", cursor=0)
        completed = {
            "method": "turn/completed",
            "params": {"turn": {"id": "u1"}, "threadId": "t1", "turnId": "u1"},
        }

        def feed():
            # Six ticks of work, each arriving after the 0.15s budget would
            # have expired on its own.  A wall-clock wait dies on the first.
            for index in range(6):
                time.sleep(0.05)
                with adapter._condition:
                    adapter._events.append({
                        "method": "item/completed",
                        "params": {"threadId": "t1", "turnId": "u1", "index": index},
                    })
                    adapter._condition.notify_all()
            time.sleep(0.05)
            with adapter._condition:
                adapter._events.append(completed)
                adapter._condition.notify_all()

        worker = threading.Thread(target=feed, daemon=True)
        worker.start()
        event = adapter.wait_event(
            lambda value: value.get("method") == "turn/completed",
            cursor=0,
            timeout=0.15,
            progress=lambda value: adapter._belongs_to_turn(value, handle),
        )
        worker.join(timeout=5)
        self.assertIs(completed, event)

    def test_a_busy_sibling_does_not_keep_a_stuck_turn_alive(self):
        """Sustained sibling traffic, not a preloaded batch, for the whole wait.

        A fixed list of sibling events is drained once and then the queue is
        quiet, which any wall-clock wait survives.  The case that matters is a
        sibling that never stops talking for longer than the budget: the stuck
        turn must still time out underneath it.
        """

        adapter = self._adapter()
        handle = TurnHandle(thread_id="t1", turn_id="u1", cursor=0)
        stop = threading.Event()
        appended: list[float] = []

        def flood():
            index = 0
            while not stop.is_set():
                with adapter._condition:
                    adapter._events.append({
                        "method": "item/completed",
                        "params": {"threadId": "t2", "turnId": "u2", "index": index},
                    })
                    adapter._condition.notify_all()
                appended.append(time.monotonic())
                index += 1
                time.sleep(0.005)

        feeder = threading.Thread(target=flood, daemon=True)
        feeder.start()
        self.addCleanup(feeder.join, 2)
        self.addCleanup(stop.set)
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(VNextAppServerError, "without activity"):
                adapter.wait_event(
                    lambda value: value.get("method") == "turn/completed",
                    cursor=0,
                    timeout=0.2,
                    progress=lambda value: adapter._belongs_to_turn(value, handle),
                )
        finally:
            stop.set()
        ended = time.monotonic()
        elapsed = ended - started
        self.assertGreaterEqual(elapsed, 0.2)
        self.assertLess(elapsed, 3.0)
        # The sibling really was talking the whole time; the wait ignored it.
        # A count of events depended on how often a loaded runner scheduled
        # the feeder (8 and 10 on a GitHub macOS runner), so the claim is
        # checked directly: the sibling was still talking in the second half
        # of the budget, after any preloaded batch would have drained.
        self.assertTrue(
            any(started + 0.1 <= moment <= ended for moment in list(appended)),
            f"no sibling event in the second half of the wait: {appended}",
        )

    def test_without_progress_the_wait_is_still_wall_clock(self):
        adapter = self._adapter()
        with self.assertRaisesRegex(VNextAppServerError, "of waiting"):
            adapter.wait_event(lambda value: False, cursor=0, timeout=0.05)

    def test_an_event_for_another_turn_is_not_progress(self):
        handle = TurnHandle(thread_id="t1", turn_id="u1", cursor=0)
        belongs = VNextAppServerAdapter._belongs_to_turn
        self.assertTrue(belongs({"params": {"turnId": "u1"}}, handle))
        self.assertFalse(belongs({"params": {"turnId": "u2"}}, handle))
        # No turn id yet: the thread is the next best correlation.
        self.assertTrue(belongs({"params": {"threadId": "t1"}}, handle))
        self.assertFalse(belongs({"params": {"threadId": "t2"}}, handle))
        self.assertFalse(belongs({"params": {}}, handle))
        self.assertFalse(belongs({}, handle))


if __name__ == "__main__":
    unittest.main()


class UnroutableResponseTests(unittest.TestCase):
    """R22 claude-code: a reply id that is not an integer killed the reader."""

    @staticmethod
    def _reading_adapter(lines: list[str]) -> VNextAppServerAdapter:
        adapter = object.__new__(VNextAppServerAdapter)
        adapter.provider = "codex"
        adapter._condition = threading.Condition(threading.RLock())
        adapter._events = []
        adapter._responses = {}
        adapter._fatal = None
        adapter._closing = False
        adapter._stderr_lines = []
        adapter.codex_home = Path("/home/someone/.codex")
        adapter.workspace = Path("/home/someone/project")
        adapter.process = types.SimpleNamespace(stdout=iter(lines), poll=lambda: None)
        return adapter

    def test_a_reply_with_a_null_or_string_id_is_kept_aside_and_reading_goes_on(self) -> None:
        for odd in (None, "abc", True):
            with self.subTest(id=odd):
                adapter = self._reading_adapter([
                    json.dumps({"jsonrpc": "2.0", "id": odd,
                                "error": {"code": -32700, "message": "Parse error"}}) + "\n",
                    json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}) + "\n",
                ])
                adapter._dispatch_stdout()
                self.assertEqual({"ok": True}, adapter._responses[7]["result"])
                self.assertEqual([odd], [m.get("id") for m in adapter._unroutable_responses])

    def test_a_reader_that_fails_says_why_instead_of_calling_stdout_closed(self) -> None:
        adapter = self._reading_adapter(['{"id": 1, "result": {}}\n'])

        def explode(_line: str) -> bool:
            raise RuntimeError("router broke")

        adapter._dispatch_line = explode
        adapter._dispatch_stdout()
        self.assertIn("router broke", adapter._fatal)
        self.assertNotIn("stdout closed unexpectedly", adapter._fatal)


class CodexExactModelIdentityTests(unittest.TestCase):
    """A Codex worker's record names the model id the app-server reported."""

    def _adapter(self, answers):
        from vnext.vnext_app_server import VNextAppServerAdapter

        adapter = object.__new__(VNextAppServerAdapter)
        adapter.workspace = Path.cwd()
        adapter.codex_home = Path.cwd()
        adapter.provider = "codex"
        adapter.client_name = "vnext_identity_test"
        adapter._condition = threading.Condition(threading.RLock())
        adapter._tool_handlers = {}
        adapter._tool_registrations = {}
        adapter._native_thread_attestations = {}
        adapter._tool_calls = []
        adapter._events = []
        adapter._responses = {}
        adapter._closing = False
        adapter.notify = lambda method, params: None
        sent = []

        def request(method, params, *, timeout=30):
            sent.append((method, dict(params)))
            answer = answers(method, params)
            if isinstance(answer, Exception):
                raise answer
            return answer

        adapter.request = request
        adapter._attest_default_environment = lambda result, timeout: dict(result)
        adapter._clock_hook_config = lambda cwd, timeout: {}
        return adapter, sent

    @staticmethod
    def _posture():
        return RuntimePosture(
            workspace_writes=True, network="restricted", approvals_requested=True,
            reviewer="auto_review", environment_ready=True,
        )

    def _start(self, adapter, model):
        thread_id, _ = adapter.start_thread(
            model=model, developer_instructions="x", tools=[], tool_handler=None,
            requested_posture=self._posture(),
        )
        return thread_id

    def test_initialize_lists_models_once_and_follows_the_cursor(self):
        from vnext.vnext_model_identity import identity_view

        def answers(method, params):
            if method == "model/list" and not params.get("cursor"):
                return {"data": [{"id": "gpt-6.1-sol", "model": "gpt-6.1-sol"}], "nextCursor": "page-2"}
            if method == "model/list":
                return {"data": [{"id": "gpt-6-luna", "model": "gpt-6-luna-2026-09"}], "nextCursor": None}
            if method == "thread/start":
                return {"thread": {"id": "t-1"}, "model": "gpt-6-luna-2026-09"}
            return {}

        adapter, sent = self._adapter(answers)
        adapter.initialize()
        lists = [params for method, params in sent if method == "model/list"]
        self.assertEqual(2, len(lists))
        self.assertTrue(all(params.get("includeHidden") is True for params in lists))
        thread_id = self._start(adapter, "gpt-6-luna")
        view = identity_view(adapter.model_identity(thread_id))
        self.assertEqual("gpt-6-luna-2026-09", view["model_exact"])
        self.assertEqual("model_list", view["model_exact_source"])
        self.assertEqual("gpt-6-luna-2026-09", view["model_ran"])
        self.assertFalse(view["model_mismatch"])

    def test_a_failed_model_list_says_so_and_the_run_continues(self):
        from vnext.vnext_app_server import VNextAppServerError
        from vnext.vnext_model_identity import identity_view

        def answers(method, params):
            if method == "model/list":
                return VNextAppServerError("model/list failed: method not found")
            if method == "thread/start":
                return {"thread": {"id": "t-2"}}
            return {}

        adapter, _ = self._adapter(answers)
        adapter.initialize()
        view = identity_view(adapter.model_identity(self._start(adapter, "gpt-6.1-sol")))
        self.assertEqual("exact model unknown: model/list failed: method not found", view["model_exact"])

    def test_a_reroute_notification_changes_model_ran_and_is_kept(self):
        from vnext.vnext_model_identity import identity_view

        def answers(method, params):
            if method == "model/list":
                return {"data": [{"id": "gpt-6.1-sol", "model": "gpt-6.1-sol"}]}
            if method == "thread/start":
                return {"thread": {"id": "t-3"}, "model": "gpt-6.1-sol"}
            return {}

        adapter, _ = self._adapter(answers)
        adapter.initialize()
        thread_id = self._start(adapter, "gpt-6.1-sol")
        adapter._dispatch_line(json.dumps({"method": "model/rerouted", "params": {
            "threadId": thread_id, "turnId": "turn-1", "fromModel": "gpt-6.1-sol",
            "toModel": "gpt-5.6-sol", "reason": "highRiskCyberActivity",
        }}))
        view = identity_view(adapter.model_identity(thread_id))
        self.assertEqual("gpt-5.6-sol", view["model_ran"])
        self.assertTrue(view["model_mismatch"])
        self.assertIn("gpt-6.1-sol", view["model_note"])
        self.assertIn("gpt-5.6-sol", view["model_note"])
        self.assertEqual("turn-1", view["model_reroutes"][0]["turn_id"])

    def test_thread_resume_reads_the_model_it_resumed_with(self):
        from vnext.vnext_model_identity import identity_view

        def answers(method, params):
            if method == "model/list":
                return {"data": [{"id": "gpt-6.1-sol", "model": "gpt-6.1-sol"}]}
            if method == "thread/start":
                return {"thread": {"id": "t-4"}, "model": "gpt-6.1-sol"}
            if method == "thread/resume":
                return {"thread": {"id": "t-4"}, "model": "gpt-6.1-sol-mini"}
            return {}

        adapter, _ = self._adapter(answers)
        adapter.initialize()
        thread_id = self._start(adapter, "gpt-6.1-sol")
        adapter.resume_thread(thread_id=thread_id, model="gpt-6.1-sol")
        view = identity_view(adapter.model_identity(thread_id))
        self.assertEqual("gpt-6.1-sol-mini", view["model_ran"])
        self.assertTrue(view["model_mismatch"])
