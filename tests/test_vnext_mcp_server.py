"""The MCP endpoint a coding client is pointed at, exercised over real HTTP.

These tests do not call the service's Python methods where a client would call
its socket.  A client that cannot parse the listing, or that gets a 501 where it
expected a 405, fails in ways an in-process test never sees.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext import vnext_mcp_server as server
from vnext.vnext_codex_mcp import MCP_PROTOCOL_VERSIONS
from vnext.vnext_mcp_server import VNextMcpService, VNextMcpServiceError
from vnext.vnext_orchestration import AgentMessage, AgentRole
from vnext.vnext_scheduler import VNextScheduler
from vnext.vnext_report import read_workspace

from tests.test_vnext_external_primary import ScriptedWorker, WORKER_MODEL


CLIENT = "claude-code"


def hermetic_server_arguments(directory: str | Path) -> list[str]:
    """Server flags that keep a spawned server off this machine's logins.

    A server started with no --catalog validates the default catalog against
    the local Codex and Claude logins and the local provider keys, so these
    cases read whatever the host happened to be signed in to: on a machine
    with no credentials the server exited at startup and the test reported
    "the server exited before it announced itself", which reads as a broken
    server rather than a missing login.  One provider the server has no
    built-in credential rule for, with the scripted worker registered
    against it, asks the host for nothing.
    """

    catalog = Path(directory) / "hermetic-catalog.json"
    catalog.write_text(
        json.dumps({"models": [{"provider": "scripted", "model": WORKER_MODEL}]}),
        encoding="utf-8",
    )
    return [
        "--catalog", str(catalog),
        "--provider", "scripted=tests.test_vnext_external_primary:ScriptedWorker",
    ]


class _LockedSink:
    """What a pipe gives serve_stdio: a writer several threads may share."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._written: list[str] = []

    def write(self, text: str) -> int:
        with self._lock:
            self._written.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def answers(self) -> dict:
        with self._lock:
            text = "".join(self._written)
        return {
            answer["id"]: answer
            for answer in (json.loads(line) for line in text.splitlines() if line.strip())
        }


class ModelRosterTests(unittest.TestCase):
    def test_default_catalog_filters_each_unavailable_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            (home / "auth.json").write_text('{"tokens":{"access_token":"fixture"}}')
            # The SDK and the CLI are environment facts and this test is about
            # the login filters, so both are held present for both halves: a
            # runner with no Claude CLI installed read as signed out.
            with patch.object(
                server, "_claude_sdk_installed", return_value=True
            ), patch.object(
                server, "_claude_executable", return_value=str(home / "claude")
            ), patch.dict(
                os.environ,
                {"CODEX_HOME": str(home), "HOME": str(home), "USERPROFILE": str(home)},
                clear=True,
            ), patch(
                "vnext.vnext_mcp_server.load_zai_provider", return_value=None
            ), patch(
                "vnext.vnext_mcp_server.load_commandcode_provider", return_value=None
            ), patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='{"loggedIn":false}')):
                available = server._validate_catalog(server.DEFAULT_CATALOG)
                self.assertEqual({"codex"}, {entry["provider"] for entry in available})

                (home / "auth.json").unlink()
                with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='{"loggedIn":true}')):
                    available = server._validate_catalog(server.DEFAULT_CATALOG)
                self.assertEqual({"claude"}, {entry["provider"] for entry in available})

    def test_model_roster_includes_claim_statements(self) -> None:
        roster = server._model_roster([{
            "provider": "codex",
            "model": "gpt-6-luna",
            "claims": [{"kind": "observation", "statement": "Delegate it with effort high."}],
        }])

        self.assertIn(
            "gpt-6-luna (codex via app-server; observation: Delegate it with effort high.)",
            roster,
        )

    def test_model_roster_without_claims_is_unchanged(self) -> None:
        roster = server._model_roster([{"provider": "codex", "model": "gpt-6-sol"}])

        self.assertEqual(
            server._MODEL_ROSTER_PREAMBLE + "gpt-6-sol (codex via app-server)."
            + server._MODEL_ROSTER_EPILOGUE,
            roster,
        )

    def test_the_served_retry_tool_offers_the_unconfirmed_stop_override(self) -> None:
        """The flag a manager has to name is on the tool it actually calls.

        An external client reads this listing and nothing else, and the schema
        refuses any argument it does not declare, so a flag missing here is a
        flag no MCP manager can send.
        """

        tools = server._external_tools(VNextScheduler.manager_tools())

        retry = next(tool for tool in tools if tool["name"] == "retry")
        override = retry["inputSchema"]["properties"]["start_despite_unconfirmed_stop"]
        self.assertEqual("boolean", override["type"])
        self.assertIn("never confirmed it stopped", override["description"])

    def test_default_catalog_claim_is_in_the_served_delegate_description(self) -> None:
        tools = server._external_tools(
            [{"name": "delegate", "description": "Delegate work."}],
            server.DEFAULT_CATALOG,
        )

        delegate = next(tool for tool in tools if tool["name"] == "delegate")
        self.assertIn("gpt-6-luna", delegate["description"])
        self.assertIn("effort high", delegate["description"])

    def test_delegate_and_replace_descriptions_include_worker_limits(self) -> None:
        tools = server._external_tools(VNextScheduler.manager_tools(), server.DEFAULT_CATALOG)
        for name in ("delegate", "replace"):
            description = next(tool["description"] for tool in tools if tool["name"] == name)
            with self.subTest(tool=name):
                self.assertEqual(1, description.count("Worker limits:"))
                for limit in ("1800", "30 min", "64 MiB", "64 KiB", "API", "browser",
                              "network restricted", "acknowledge_messages_through", "unread-messages",
                              "(claude, zai) it is a wall-clock cap", "(codex, commandcode) every turn event renews it",
                              "~/.codex"):
                    self.assertIn(limit, description)


class RunFolderTests(unittest.TestCase):
    def test_dispatch_preserves_warnings_when_adding_wake_command(self) -> None:
        from vnext.vnext_runtime_types import ToolCallResult

        service = object.__new__(VNextMcpService)
        service._closing = False
        service.workspace = Path(".")
        service._session_id = "fake-session"
        service.session = SimpleNamespace(external_tool_call=lambda **kwargs: ToolCallResult(
            True, {"agent_id": "fake-child", "warnings": ["turn cap", "no browser"]},
        ))
        result = service._dispatch("delegate", {"objective": "Use browser for 90 minutes"}, {})
        self.assertTrue(result.success)
        self.assertEqual(["turn cap", "no browser"], result.value["warnings"])
        self.assertIn("wake_command", result.value)

    def test_new_project_run_folder_ignores_its_own_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": ScriptedWorker},
            )
            try:
                self.assertEqual("*\n", (workspace / ".vnext" / ".gitignore").read_text())
            finally:
                service.close()


class ADelegatedChildIsInTheRunLogBeforeProviderStartTests(unittest.TestCase):
    def test_delegate_is_durable_while_provider_start_is_held(self) -> None:
        class HeldStart(ScriptedWorker):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def start_thread(self, **kwargs):
                self.entered.set()
                if not self.release.wait(5):
                    raise RuntimeError("test did not release provider start")
                return super().start_thread(**kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            worker = HeldStart()
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                port=0,
                adapter_factories={"codex": lambda: worker},
            )
            try:
                result = service.session.external_tool_call(tool="delegate", arguments={
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER_MODEL,
                    "objective": "Hold provider start while checking the journal",
                    "task_contract": {"criteria": ["journal names the child"]},
                })
                self.assertTrue(result.success, result.as_json_text())
                child_id = json.loads(result.as_json_text())["agent_id"]
                self.assertTrue(worker.entered.wait(2), "provider start never began")
                log = workspace / ".vnext" / "runs" / f"{service._session_id}.jsonl"
                rows = [json.loads(line) for line in log.read_text().splitlines()]
                child_rows = [row for row in rows if row.get("agent_id") == child_id]
                self.assertTrue(child_rows, f"delegate returned {child_id}, but log has no child record")
                self.assertIn("agent.delegated", [row["type"] for row in child_rows])
                self.assertIn("ready", [row["payload"].get("status") for row in child_rows
                                        if row["type"] == "agent.upsert"])
                self.assertNotIn("agent.spawned", [row["type"] for row in child_rows])
                report = read_workspace(workspace)["sessions"][0]
                self.assertEqual({"primary", child_id}, report["agents"])

                worker.release.set()
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    rows = [json.loads(line) for line in log.read_text().splitlines()]
                    if any(row["type"] == "agent.spawned" and row.get("agent_id") == child_id
                           for row in rows):
                        break
                    time.sleep(0.01)
                self.assertEqual(1, len([row for row in rows if row["type"] == "agent.spawned"
                                         and row.get("agent_id") == child_id]))
                self.assertEqual({"primary", child_id},
                                 read_workspace(workspace)["sessions"][0]["agents"])
            finally:
                worker.release.set()
                service.close()


    def test_failed_provider_start_replaces_ready_with_blocked_in_the_log(self) -> None:
        class Refusing(ScriptedWorker):
            def start_thread(self, **kwargs):
                raise RuntimeError("provider refused the test child")

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                port=0,
                adapter_factories={"codex": Refusing},
            )
            try:
                result = service.session.external_tool_call(tool="delegate", arguments={
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER_MODEL,
                    "objective": "Fail provider start after delegation",
                    "task_contract": {},
                })
                self.assertTrue(result.success, result.as_json_text())
                child_id = json.loads(result.as_json_text())["agent_id"]
                log = workspace / ".vnext" / "runs" / f"{service._session_id}.jsonl"
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        rows = [json.loads(line) for line in log.read_text().splitlines()]
                    except json.JSONDecodeError:
                        continue
                    states = [row["payload"]["status"] for row in rows
                              if row.get("agent_id") == child_id
                              and row["type"] == "agent.upsert"]
                    if "blocked" in states:
                        break
                    time.sleep(0.01)
                self.assertEqual("ready", states[0])
                self.assertEqual("blocked", states[-1])
                self.assertNotIn("agent.spawned", [row["type"] for row in rows
                                                    if row.get("agent_id") == child_id])
                self.assertEqual({"primary", child_id},
                                 read_workspace(workspace)["sessions"][0]["agents"])
            finally:
                service.close()


class VNextMcpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp.name).resolve()
        self.addCleanup(self._temp.cleanup)
        self.worker = ScriptedWorker()
        self.log = self.workspace / "events.jsonl"
        self.service = VNextMcpService(
            workspace=self.workspace,
            catalog=[{"provider": "codex", "model": WORKER_MODEL}],
            client=CLIENT,
            port=0,
            event_log=self.log,
            adapter_factories={"codex": lambda: self.worker},
        )
        self.addCleanup(self.service.close)
        self.address = self.service.start()

    def _call(self, payload: dict, *, token: str | None = None, method: str = "POST") -> dict:
        request = urllib.request.Request(
            self.address.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {token or self.address.token}",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _read_log(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
        ]

    def _await_log(self, ready, timeout: float = 10.0) -> list[dict]:
        """Records are appended from the scheduler's threads, not the caller's.

        `delegate` answers as soon as the child has a handle.  Binding its
        runtime, and the `agent_spawned` record that follows, happen on the
        scheduler loop afterwards, so a test that reads the file the moment
        the call returns reads it mid-write.
        """

        deadline = time.monotonic() + timeout
        lines: list[dict] = []
        while True:
            try:
                lines = self._read_log()
            except json.JSONDecodeError:
                # A record caught halfway through its own write.  The next
                # pass reads the whole line.
                lines = []
            if ready(lines):
                return lines
            if time.monotonic() >= deadline:
                self.fail(
                    f"the run log never reached the expected state in {timeout}s: "
                    f"{[line['type'] for line in lines]}"
                )
            time.sleep(0.02)

    def _tool(self, name: str, arguments: dict) -> dict:
        answer = self._call({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        return answer["result"]

    def test_initialize_answers_in_the_client_s_own_revision(self) -> None:
        for version in MCP_PROTOCOL_VERSIONS:
            answer = self._call({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": version, "capabilities": {},
                           "clientInfo": {"name": CLIENT, "version": "1"}},
            })
            self.assertEqual(version, answer["result"]["protocolVersion"])
        unknown = self._call({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        })
        self.assertEqual(MCP_PROTOCOL_VERSIONS[-1], unknown["result"]["protocolVersion"])

    def test_the_listing_holds_only_keys_mcp_defines(self) -> None:
        tools = self._call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        names = {tool["name"] for tool in tools}
        self.assertIn("delegate", names)
        self.assertIn("await_children", names)
        allowed = {"name", "title", "description", "inputSchema", "outputSchema", "annotations"}
        for tool in tools:
            self.assertEqual(set(tool) - allowed, set(), tool["name"])
            self.assertTrue(tool["description"])
            self.assertEqual("object", tool["inputSchema"]["type"])


    def test_tools_list_serves_worker_limits_for_delegate_and_replace(self) -> None:
        tools = self._call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        for name in ("delegate", "replace"):
            description = next(tool["description"] for tool in tools if tool["name"] == name)
            with self.subTest(tool=name):
                self.assertEqual(1, description.count("Worker limits:"))
                for limit in ("1800", "30 min", "64 MiB", "64 KiB", "API", "browser",
                              "network restricted", "acknowledge_messages_through", "unread-messages",
                              "(claude, zai) it is a wall-clock cap", "(codex, commandcode) every turn event renews it",
                              "~/.codex"):
                    self.assertIn(limit, description)

    def test_delegate_warnings_survive_mcp_dispatch(self) -> None:
        created = self._tool("delegate", {
            "role": AgentRole.WORKER.value, "model_id": WORKER_MODEL,
            "objective": "Use playwright for 96 minutes",
        })
        self.assertFalse(created["isError"], created)
        # This worker has no Claude SDK route, so only the browser warning applies.
        self.assertEqual(1, len(created["structuredContent"]["warnings"]))
        self.assertEqual(json.loads(created["content"][0]["text"]), created["structuredContent"])

    def test_a_client_delegates_and_waits_over_http(self) -> None:
        created = self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        self.assertFalse(created["isError"], created)
        child_id = created["structuredContent"]["agent_id"]

        waited = self._tool("await_children", {"agent_ids": [child_id]})
        self.assertFalse(waited["isError"], waited)
        self.assertTrue(waited["structuredContent"]["settled"])

        inspected = self._tool("inspect", {"agent_id": child_id})
        self.assertFalse(inspected["isError"], inspected)
        # The text block carries the same value; a client that renders only
        # content must still see the outcome.
        self.assertEqual(
            json.loads(inspected["content"][0]["text"]), inspected["structuredContent"]
        )
        self.assertEqual(1, len(self.worker.objectives))
        self.assertIn("Write the report section", self.worker.objectives[0])

    def test_a_worker_whose_provider_dies_does_not_take_the_chat_with_it(self) -> None:
        """The failure the user hit: one bad worker, and vNext was gone.

        A Claude worker whose bridge timed out raised out of the scheduler and
        failed the session. Every later tool call in that window answered only
        that the session had failed, until the terminal was restarted.
        """

        class Refusing(ScriptedWorker):
            def start_thread(self, **kwargs):
                raise RuntimeError("Claude bridge timed out waiting for start_thread")

        elsewhere = self.workspace / "broken"
        elsewhere.mkdir()
        broken = VNextMcpService(
            workspace=elsewhere,
            catalog=[{"provider": "codex", "model": WORKER_MODEL}],
            client=CLIENT,
            port=0,
            adapter_factories={"codex": lambda: Refusing()},
        )
        self.addCleanup(broken.close)
        address = broken.start()

        def call(name: str, arguments: dict) -> dict:
            request = urllib.request.Request(
                address.endpoint,
                data=json.dumps({
                    "jsonrpc": "2.0", "id": 9, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Authorization": f"Bearer {address.token}",
                },
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))["result"]

        created = call("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        self.assertFalse(created["isError"], created)
        child_id = created["structuredContent"]["agent_id"]

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            inspected = call("inspect", {"agent_id": child_id})
            self.assertFalse(inspected["isError"], inspected)
            if inspected["structuredContent"]["agent"]["status"] == "blocked":
                break
            time.sleep(0.02)
        else:
            self.fail(f"the child never blocked: {inspected}")

        blocker = inspected["structuredContent"]["agent"]["blocker"]
        self.assertIn("runtime could not be started", blocker)
        self.assertIn("timed out waiting for start_thread", blocker)
        # And the session is still the client's to use, which is the whole point.
        again = call("inspect", {"agent_id": "primary"})
        self.assertFalse(again["isError"], again)

    def test_bad_claude_effort_is_a_tool_error_without_a_spawn_record(self) -> None:
        registry = self.service.session.registry
        registry.cards[WORKER_MODEL] = replace(registry.cards[WORKER_MODEL], provider="claude")
        before = set(self.service.session.control.sessions[self.service.session.root.session_id].agents)
        answer = self._tool("delegate", {
            "role": "worker", "model_id": WORKER_MODEL, "objective": "work",
            "effort": "ludicrous", "task_contract": {"criteria": ["done"]},
        })
        self.assertTrue(answer["isError"], answer)
        error = answer["structuredContent"]["error"]
        self.assertIn("high, low, max, medium, xhigh", error)
        self.assertEqual(before, set(self.service.session.control.sessions[self.service.session.root.session_id].agents))
        self.assertNotIn('"effort": "ludicrous"', self.log.read_text(encoding="utf-8"))

    def test_a_refused_call_is_an_error_result_not_a_broken_server(self) -> None:
        answer = self._tool("delegate", {"role": "worker", "model_id": "not-in-catalog",
                                         "objective": "x", "task_contract": {"criteria": ["x"]}})
        self.assertTrue(answer["isError"], answer)
        self.assertTrue(answer["structuredContent"]["error"])

    def test_an_unknown_tool_is_rejected_before_the_tree(self) -> None:
        answer = self._call({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "rm_rf", "arguments": {}},
        })
        self.assertEqual(-32602, answer["error"]["code"])

    def test_a_wrong_credential_is_refused(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._call({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}, token="wrong-token")
        self.assertEqual(403, caught.exception.code)

    def test_a_stream_request_is_declined_the_way_the_spec_says(self) -> None:
        request = urllib.request.Request(
            self.address.endpoint, method="GET",
            headers={"Accept": "text/event-stream",
                     "Authorization": f"Bearer {self.address.token}"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(405, caught.exception.code)

    def test_the_run_is_recorded_beside_the_workspace(self) -> None:
        self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        lines = self._await_log(
            lambda lines: "agent.spawned" in {line["type"] for line in lines}
        )
        self.assertIn("agent.spawned", {line["type"] for line in lines})

    def test_a_sent_message_is_recorded_verbatim(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        self.worker.hold = release
        created = self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Wait for one message",
            "task_contract": {"criteria": ["message received"]},
        })
        child_id = created["structuredContent"]["agent_id"]
        message = "first  line\n\tsecond line"

        sent = self._tool("send_message", {"agent_id": child_id, "message": message})
        self.assertFalse(sent["isError"], sent)

        lines = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        records = [
            line for line in lines
            if line.get("payload", {}).get("command") == "send_message"
        ]
        self.assertEqual(1, len(records))
        self.assertEqual(message, records[0]["payload"]["message"])
        self.assertEqual(len(message), records[0]["payload"]["message_characters"])

        release.set()

    def test_every_run_log_record_has_a_utc_timestamp(self) -> None:
        self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })

        lines = self._await_log(
            lambda lines: "agent.spawned" in {line["type"] for line in lines}
        )
        self.assertGreaterEqual(len({line["type"] for line in lines}), 2)
        for line in lines:
            recorded_at = datetime.fromisoformat(line["timestamp"])
            self.assertIsNotNone(recorded_at.tzinfo)
            self.assertEqual(timedelta(0), recorded_at.utcoffset())

    def _await_roster(self, path: Path, ready, timeout: float = 10.0) -> dict:
        """The roster is written from the scheduler's threads, not the caller's."""

        deadline = time.monotonic() + timeout
        snapshot: dict = {}
        while time.monotonic() < deadline:
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                snapshot = {}
            if snapshot and ready(snapshot):
                return snapshot
            time.sleep(0.02)
        self.fail(f"roster never settled: {snapshot}")

    def test_the_roster_names_the_live_workers(self) -> None:
        """What the user asked for: something that says workers exist.

        Written as a whole file so the thing drawing it -- a status line that
        redraws every few hundred milliseconds -- reads one small snapshot
        instead of tailing an event log that grows across sessions.
        """

        status = self.service.status_file
        self.assertEqual(0, json.loads(status.read_text(encoding="utf-8"))["working"])

        # Held open, or the scripted worker finishes inside the delegate call
        # and the roster would never be seen with anyone in it.
        release = threading.Event()
        self.worker.hold = release
        created = self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        child_id = created["structuredContent"]["agent_id"]
        working = self._await_roster(status, lambda s: s["working"] == 1)
        self.assertTrue(working["live"])
        self.assertEqual([child_id], [agent["agent_id"] for agent in working["agents"]])
        self.assertEqual(WORKER_MODEL, working["agents"][0]["model"])

        release.set()
        self._tool("await_children", {"agent_ids": [child_id]})
        settled = self._await_roster(status, lambda s: s["working"] == 0)
        self.assertEqual(1, settled["finished"])

    def test_a_blocked_row_in_the_roster_says_what_stopped_the_worker(self) -> None:
        """The roster counts blocked workers, so it has to say why they are.

        One real session left 33 of them in the file with nothing but the
        word "blocked" against each.
        """

        self.service._remember({
            "agent_id": "worker-1",
            "role": AgentRole.WORKER.value,
            "model": WORKER_MODEL,
            "provider": "codex",
            "status": "blocked",
            "effort": "medium",
            "parent_agent_id": "primary",
            "blocker": "waiting on an approval nobody answered",
        })
        self.service._publish_status()

        snapshot = json.loads(self.service.status_file.read_text(encoding="utf-8"))
        self.assertEqual(1, snapshot["blocked"])
        self.assertEqual(
            "waiting on an approval nobody answered", snapshot["agents"][0]["blocker"]
        )

    def test_a_worker_that_was_never_blocked_carries_no_blocker_key(self) -> None:
        self.service._remember({
            "agent_id": "worker-1",
            "role": AgentRole.WORKER.value,
            "model": WORKER_MODEL,
            "provider": "codex",
            "status": "running",
            "effort": "medium",
            "parent_agent_id": "primary",
        })
        self.service._publish_status()

        snapshot = json.loads(self.service.status_file.read_text(encoding="utf-8"))
        self.assertNotIn("blocker", snapshot["agents"][0])

    def test_a_sibling_session_in_this_process_is_not_read_as_a_child(self) -> None:
        """Two sessions in one process are siblings, so neither names a parent."""

        sibling = VNextMcpService(
            workspace=self.workspace,
            catalog=[{"provider": "codex", "model": WORKER_MODEL}],
            client=CLIENT,
            port=0,
            adapter_factories={"codex": lambda: ScriptedWorker()},
        )
        self.addCleanup(sibling.close)
        for service in (self.service, sibling):
            snapshot = json.loads(service.status_file.read_text(encoding="utf-8"))
            self.assertNotIn("parent_session", snapshot)

    def test_a_server_started_inside_a_worker_names_the_session_above_it(self) -> None:
        """A Claude worker's own vnext server used to look like a second project.

        The parent leaves its session id and pid in the environment the worker
        inherits.  A server that finds another process's id there is running
        inside that session's worker, and says so where the report can read it.
        """

        with patch.dict(os.environ, {"VNEXT_PARENT_SESSION": "the-parent-session:1"}):
            nested = VNextMcpService(
                workspace=self.workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                port=0,
                adapter_factories={"codex": lambda: ScriptedWorker()},
            )
            self.addCleanup(nested.close)
            snapshot = json.loads(nested.status_file.read_text(encoding="utf-8"))
            self.assertEqual("the-parent-session", snapshot["parent_session"])

    def test_two_servers_on_one_workspace_keep_separate_files(self) -> None:
        """Several Claude Code windows open one project, each with its own server.

        They were all appending to one events.jsonl: four of the first 4328
        records in this project's log were left truncated and NUL-padded by
        overlapping appends, and no record said which run it came from.
        """

        second = VNextMcpService(
            workspace=self.workspace,
            catalog=[{"provider": "codex", "model": WORKER_MODEL}],
            client=CLIENT,
            port=0,
            adapter_factories={"codex": lambda: ScriptedWorker()},
        )
        self.addCleanup(second.close)
        default = VNextMcpService(
            workspace=self.workspace,
            catalog=[{"provider": "codex", "model": WORKER_MODEL}],
            client=CLIENT,
            port=0,
            adapter_factories={"codex": lambda: ScriptedWorker()},
        )
        self.addCleanup(default.close)

        self.assertNotEqual(second.status_file, default.status_file)
        self.assertNotEqual(second.outcome_log, default.outcome_log)
        self.assertEqual(self.workspace / ".vnext" / "status", second.status_file.parent)

    def test_a_finished_agent_leaves_one_row_to_rate_it_by(self) -> None:
        """A rating pass reads one row per agent, not thousands of events."""

        created = self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        child_id = created["structuredContent"]["agent_id"]
        self._tool("await_children", {"agent_ids": [child_id]})
        path = self.service.outcome_log
        self._await_roster(self.service.status_file, lambda s: s["finished"] == 1)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(child_id, row["agent_id"])
        self.assertEqual(WORKER_MODEL, row["model"])
        self.assertEqual("completed", row["status"])
        self.assertEqual(self.service._session_id, row["session_id"])
        self.assertIn("claimed_verified", row)

    def test_a_row_carries_the_counts_and_the_price_whoever_reported_them(self) -> None:
        """Every Claude worker's row read empty while the numbers were there.

        The writer read the Codex spellings only, so a row for the other vendor
        carried five nulls and no price, and a session could not be costed.
        """

        path = self.service.outcome_log
        claude = {
            "agent_id": "a1", "role": "worker", "provider": "claude",
            "model": "sonnet", "status": "completed",
            "usage": {
                "tokens": {"input_tokens": 12, "cache_read_input_tokens": 300339,
                           "cache_creation_input_tokens": 33202, "output_tokens": 2074},
                "cost_tokens": {"cache_read_tokens": 300339, "input_tokens": 12,
                                "output_tokens": 2074, "total_tokens": 335627,
                                "basis": "call"},
                "cost_usd": 0.3216827,
            },
        }
        codex = {
            "agent_id": "a2", "role": "worker", "provider": "codex",
            "model": "gpt-5.6-luna", "status": "completed",
            "usage": {
                "tokens": {"totalTokens": 18066, "inputTokens": 17949,
                           "cachedInputTokens": 0, "outputTokens": 117,
                           "reasoningOutputTokens": 36},
                "cost_tokens": {"cache_read_tokens": 0, "input_tokens": 17949,
                                "output_tokens": 117, "total_tokens": 18066,
                                "basis": "thread"},
            },
        }
        for view in (claude, codex):
            self.service._record_outcome(view, time.time())
        rows = {row["agent_id"]: row
                for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

        first = rows["a1"]
        self.assertEqual(335627, first["tokens"]["total"])
        self.assertEqual(300339, first["tokens"]["cached_input"])
        self.assertEqual(2074, first["tokens"]["output"])
        self.assertEqual("call", first["tokens"]["basis"])
        # The provider stated this one, so it is carried rather than modelled.
        self.assertEqual(0.3216827, first["cost_usd"])
        self.assertEqual("provider", first["cost_source"])

        second = rows["a2"]
        self.assertEqual(18066, second["tokens"]["total"])
        self.assertEqual(36, second["tokens"]["reasoning_output"])
        self.assertEqual("thread", second["tokens"]["basis"])
        # Codex states no figure, so this is the published-rate equivalent.
        self.assertAlmostEqual(0.00373, second["cost_usd"], places=5)
        self.assertEqual("api_rates", second["cost_source"])

    def test_a_closed_service_stops_claiming_workers(self) -> None:
        """A roster left behind would report a workforce that no longer exists."""

        self._tool("delegate", {
            "role": AgentRole.WORKER.value,
            "model_id": WORKER_MODEL,
            "objective": "Write the report section",
            "task_contract": {"criteria": ["section written"]},
        })
        self.service.close()
        closed = json.loads(self.service.status_file.read_text(encoding="utf-8"))
        self.assertFalse(closed["live"])
        self.assertEqual(0, closed["working"])

    def test_a_late_event_does_not_republish_a_closed_service_as_live(self) -> None:
        """A workforce step that overruns keeps emitting after the close.

        Each emission republished the roster with the default live=True, so the
        status file said a live session with reachable workers moments after the
        close had said it was stopped. The closing state is sticky.
        """

        self.service.close()
        self.assertFalse(
            json.loads(self.service.status_file.read_text(encoding="utf-8"))["live"]
        )

        self.service._record(
            SimpleNamespace(
                type="agent.upsert",
                agent_id="worker-late",
                payload={
                    "agent_id": "worker-late",
                    "role": AgentRole.WORKER.value,
                    "model": WORKER_MODEL,
                    "provider": "codex",
                    "status": "running",
                    "parent_agent_id": "primary",
                },
            )
        )

        snapshot = json.loads(self.service.status_file.read_text(encoding="utf-8"))
        self.assertFalse(snapshot["live"])
        self.assertEqual(0, snapshot["working"])

    def test_a_publish_already_in_flight_cannot_land_after_the_stopped_write(self) -> None:
        """The closing flag was read before the lock, so a gap opened after it.

        A publish that read the flag as False could then wait at the roster
        lock for the whole of close(), and write live=True over the stopped
        status close() had just written. The gate below holds one late event at
        exactly that point, so the interleaving is the test rather than a
        matter of timing.
        """

        publish_reached_the_lock = threading.Event()
        let_the_publish_go = threading.Event()
        real_lock = self.service._roster_lock

        class GateOneThreadBeforeTheLock:
            """Hold the late event's publish just before it takes the lock."""

            def __init__(self) -> None:
                self._taken = threading.local()

            def __enter__(self):
                taken = getattr(self._taken, "count", 0) + 1
                self._taken.count = taken
                # _record takes this lock twice in turn: once in _remember for
                # the roster row, then again inside _publish_status. The second
                # one is the one to hold.
                if threading.current_thread().name == "late-event" and taken == 2:
                    publish_reached_the_lock.set()
                    if not let_the_publish_go.wait(10):
                        raise AssertionError("the gated publish was never released")
                real_lock.acquire()
                return self

            def __exit__(self, *_args) -> None:
                real_lock.release()

        self.service._roster_lock = GateOneThreadBeforeTheLock()
        late = threading.Thread(
            target=self.service._record,
            args=(
                SimpleNamespace(
                    type="agent.upsert",
                    agent_id="worker-late",
                    payload={
                        "agent_id": "worker-late",
                        "role": AgentRole.WORKER.value,
                        "model": WORKER_MODEL,
                        "provider": "codex",
                        "status": "running",
                        "parent_agent_id": "primary",
                    },
                ),
            ),
            name="late-event",
        )
        late.start()
        try:
            self.assertTrue(
                publish_reached_the_lock.wait(10),
                "the late publish never reached the lock",
            )
            self.service.close()
            stopped = json.loads(self.service.status_file.read_text(encoding="utf-8"))
            self.assertFalse(stopped["live"])
        finally:
            let_the_publish_go.set()
            late.join(10)
        self.assertFalse(late.is_alive())

        after = json.loads(self.service.status_file.read_text(encoding="utf-8"))
        self.assertFalse(after["live"], "a publish in flight landed after the close")
        self.assertEqual(0, after["working"])

    def test_the_registration_line_names_the_endpoint_and_credential(self) -> None:
        line = self.address.claude_mcp_add("vnext")
        self.assertIn("--transport http", line)
        self.assertIn(self.address.endpoint, line)
        self.assertIn(self.address.token, line)

    def test_ping_is_answered_with_an_empty_result(self) -> None:
        answer = self._call({"jsonrpc": "2.0", "id": 3, "method": "ping"})
        self.assertEqual({}, answer["result"])
        self.assertNotIn("error", answer)

    def test_the_delegate_schema_names_the_fields_the_server_requires(self) -> None:
        tools = self._call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        delegate = next(tool for tool in tools if tool["name"] == "delegate")
        self.assertEqual(
            ["role", "model_id", "objective"], delegate["inputSchema"]["required"])

    def test_a_missing_role_is_answered_in_plain_words(self) -> None:
        answer = self._tool("delegate", {
            "model_id": WORKER_MODEL, "effort": "low", "objective": "x",
            "task_contract": {"criteria": ["c"]},
        })
        self.assertTrue(answer["isError"], answer)
        message = answer["structuredContent"]["error"]
        self.assertEqual(
            "role is required: one of root-manager, branch-manager, worker", message)

    def test_an_unknown_role_names_the_roles_that_work(self) -> None:
        answer = self._tool("delegate", {
            "role": "boss", "model_id": WORKER_MODEL, "objective": "x",
        })
        self.assertTrue(answer["isError"], answer)
        self.assertEqual(
            "role must be one of root-manager, branch-manager, worker, and was 'boss'",
            answer["structuredContent"]["error"])


class TestVNextMcpCostSource(unittest.TestCase):
    """Whether an outcome row's price was stated or modelled.

    Found by a cross-vendor review, 2026-09-22: cost_usd carried either the
    provider's own figure or vNext's estimate at published API rates, and the
    row did not say which, so an estimate read as a bill.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.path = Path(self._temp.name) / "outcomes.jsonl"
        self.service = object.__new__(VNextMcpService)
        self.service._session_id = "cost-test"
        self.service._failing_writes = set()
        self.service.workspace = Path(self._temp.name)
        self.service._client = CLIENT
        self.service._outcome_log = self.path
        self.service._log_lock = threading.Lock()
        self.service._outcome_rows = {}
        self.service._latest_usage = {}
        self.service._usage_unavailable = {}

    def _row(self, usage: dict, model: str = "gpt-5.6-luna") -> dict:
        self.service._record_outcome({
            "agent_id": "worker", "role": "worker", "provider": "codex",
            "model": model, "status": "completed", "usage": usage,
        }, time.time())
        return json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])

    def test_the_row_names_the_exact_model_beside_the_alias(self) -> None:
        self.service._record_outcome({
            "agent_id": "worker", "role": "worker", "provider": "claude",
            "model": "fable", "status": "completed", "usage": {},
            "model_exact": "claude-fable-5-1", "model_exact_source": "family_match",
            "model_ran": "claude-fable-5-1", "model_mismatch": False,
        }, time.time())
        row = json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual("fable", row["model"])
        self.assertEqual("claude-fable-5-1", row["model_exact"])
        self.assertEqual("family_match", row["model_exact_source"])
        self.assertEqual("claude-fable-5-1", row["model_ran"])
        self.assertFalse(row["model_mismatch"])

    def test_a_row_without_an_exact_model_says_it_is_unknown(self) -> None:
        row = self._row({})
        self.assertEqual("exact model unknown: the provider reported no model for this worker", row["model_exact"])

    def test_a_mismatch_states_both_ids(self) -> None:
        self.service._record_outcome({
            "agent_id": "worker", "role": "worker", "provider": "codex",
            "model": "gpt-6.1-sol", "status": "completed", "usage": {},
            "model_exact": "gpt-6.1-sol", "model_exact_source": "model_list",
            "model_ran": "gpt-5.6-sol", "model_mismatch": True,
            "model_note": "asked for gpt-6.1-sol, the provider ran gpt-5.6-sol",
        }, time.time())
        row = json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertTrue(row["model_mismatch"])
        self.assertIn("gpt-5.6-sol", row["model_note"])
        self.assertIn("gpt-6.1-sol", row["model_note"])

    def test_a_stated_cost_is_labeled_provider(self) -> None:
        row = self._row({"tokens": {"inputTokens": 100, "outputTokens": 20}, "cost_usd": 0.25})
        self.assertEqual(0.25, row["cost_usd"])
        self.assertEqual("provider", row["cost_source"])

    def test_an_unstated_cost_for_a_priced_model_is_labeled_api_rates(self) -> None:
        row = self._row({"tokens": {"inputTokens": 100, "outputTokens": 20}})
        self.assertAlmostEqual(0.000044, row["cost_usd"], places=6)
        self.assertEqual("api_rates", row["cost_source"])

    def test_a_zero_stated_cost_falls_back_to_rates_and_is_labeled(self) -> None:
        row = self._row({"tokens": {"inputTokens": 100, "outputTokens": 20}, "cost_usd": 0.0})
        self.assertAlmostEqual(0.000044, row["cost_usd"], places=6)
        self.assertEqual("api_rates", row["cost_source"])

    def test_native_child_usage_is_attributed_to_its_parent(self) -> None:
        self.service._record_outcome({
            "agent_id": "native-child", "parent_agent_id": "parent",
            "role": "worker", "provider": "claude", "model": "opus",
            "harness": "claude-agent-sdk",
            "status": "completed", "native_identity": {"origin": "native"},
            "usage": {"tokens": {"totalTokens": 50, "inputTokens": 40,
                                  "outputTokens": 10}, "cost_usd": 0.02},
        }, time.time())
        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(row["cost_usd"])
        self.assertEqual("in_parent", row["cost_source"])
        self.assertEqual("in_parent", row["tokens"]["basis"])
        self.assertTrue(all(row["tokens"][key] is None for key in
                            ("total", "input", "cached_input", "output", "reasoning_output")))
        self.service._refresh_outcome_usage("native-child", {
            "tokens": {"totalTokens": 100, "inputTokens": 90, "outputTokens": 10},
            "cost_usd": 0.04,
        })
        refreshed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(row, refreshed)

    def test_a_codex_native_child_keeps_its_own_price(self) -> None:
        # A Codex native child runs on a thread of its own, whose count its
        # parent's thread does not include.
        self.service._record_outcome({
            "agent_id": "native-child", "parent_agent_id": "parent",
            "role": "worker", "provider": "codex", "model": "gpt-5.6-luna",
            "harness": "app-server",
            "status": "completed", "native_identity": {"origin": "native"},
            "usage": {"tokens": {"inputTokens": 100, "outputTokens": 20}},
        }, time.time())
        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual("api_rates", row["cost_source"])
        self.assertAlmostEqual(0.000044, row["cost_usd"], places=6)


class TestVNextMcpOutcomeRefresh(unittest.TestCase):
    """A terminal status can arrive before the provider's final usage."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.path = Path(self._temp.name) / "outcomes.jsonl"
        self.service = object.__new__(VNextMcpService)
        self.service._session_id = "refresh-test"
        self.service._failing_writes = set()
        self.service.workspace = Path(self._temp.name)
        self.service._client = CLIENT
        self.service._outcome_log = self.path
        self.service._event_log = None
        self.service._status_file = None
        self.service._log_lock = threading.Lock()
        self.service._roster_lock = threading.Lock()
        self.service._roster = {}
        self.service._outcome_rows = {}
        self.service._latest_usage = {}
        self.service._usage_unavailable = {}

    @staticmethod
    def _usage(total: int) -> dict:
        return {
            "tokens": {
                "totalTokens": total,
                "inputTokens": total - 10,
                "cachedInputTokens": 20,
                "outputTokens": 10,
            },
            "cost_tokens": {
                "total_tokens": total,
                "input_tokens": total - 30,
                "cache_read_tokens": 20,
                "output_tokens": 10,
                "basis": "thread",
            },
        }

    def _event(self, event_type: str, payload: dict) -> SimpleNamespace:
        return SimpleNamespace(
            type=event_type, agent_id="worker", turn_id="turn", payload=payload
        )

    def test_later_upsert_and_usage_rewrite_the_one_outcome_row(self) -> None:
        view = {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "codex", "model": "gpt-5.6-luna",
            "effort": "low", "status": "completed", "result": {},
            "usage": self._usage(100),
        }
        self.service._record(self._event("agent.upsert", view))
        first = json.loads(self.path.read_text(encoding="utf-8"))

        view = {**view, "usage": self._usage(200)}
        self.service._record(self._event("agent.upsert", view))
        self.service._record(self._event("usage.updated", self._usage(300)))

        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(1, len(rows))
        self.assertEqual(300, rows[0]["tokens"]["total"])
        self.assertEqual(270, rows[0]["tokens"]["input"])
        self.assertGreater(rows[0]["cost_usd"], first["cost_usd"])
        self.assertEqual(first["duration_s"], rows[0]["duration_s"])

    def test_codex_outcome_fallback_uses_fresh_input_on_create_and_refresh(self) -> None:
        def usage(total: int) -> dict:
            return {"tokens": {"totalTokens": total, "inputTokens": total - 10,
                                "cachedInputTokens": 20, "outputTokens": 10}}

        view = {"agent_id": "worker", "parent_agent_id": "primary",
                "role": "worker", "provider": "codex", "model": "gpt-5.6-luna",
                "status": "completed", "result": {}, "usage": usage(100)}
        self.service._record(self._event("agent.upsert", view))
        first = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(70, first["tokens"]["input"])
        self.service._record(self._event("usage.updated", usage(200)))
        refreshed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(170, refreshed["tokens"]["input"])

    def test_refreshed_cost_uses_the_first_row_timestamp(self) -> None:
        peak = datetime(2026, 9, 22, 7, 30, tzinfo=server.timezone.utc).timestamp()
        view = {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "commandcode",
            "model": "deepseek/deepseek-v4.1-flash", "effort": "low",
            "status": "completed", "result": {}, "usage": self._usage(100),
        }
        with patch.object(server.time, "time", return_value=peak):
            self.service._record(self._event("agent.upsert", view))

        self.service._record(self._event("usage.updated", self._usage(300)))

        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(peak, row["ts"])
        self.assertAlmostEqual(0.00009312, row["cost_usd"], places=12)

    def test_a_provider_price_at_an_unchanged_token_total_reaches_the_row(self) -> None:
        """Round-4 finding 2: cost attribution can lag the counts it belongs to."""

        view = {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "claude", "model": "unpriced-model",
            "effort": "low", "status": "completed", "result": {},
            "usage": self._usage(100),
        }
        self.service._record(self._event("agent.upsert", view))
        first = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(first["cost_usd"])
        self.assertIsNone(first["cost_source"])

        # Same totals, and now the figure the provider itself states.
        self.service._record(
            self._event("usage.updated", {**self._usage(100), "cost_usd": 0.25})
        )

        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(1, len(rows))
        self.assertEqual(0.25, rows[0]["cost_usd"])
        self.assertEqual("provider", rows[0]["cost_source"])
        # The counts did not move, so neither did they.
        self.assertEqual(100, rows[0]["tokens"]["total"])

    def test_a_usage_message_the_quit_outran_is_named_in_the_row(self) -> None:
        """A null price with a reason beats a silent null."""

        view = {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "claude", "model": "unpriced-model",
            "effort": "low", "status": "completed", "result": {}, "usage": {},
        }
        self.service._record(self._event("agent.upsert", view))
        self.assertIsNone(json.loads(self.path.read_text(encoding="utf-8"))["cost_usd"])

        self.service._record(
            self._event("usage.unavailable", {"reason": "the close grace expired first"})
        )

        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(row["cost_usd"])
        self.assertEqual("the close grace expired first", row["usage_unavailable"])

    def test_a_reason_that_arrives_before_the_row_still_reaches_it(self) -> None:
        self.service._record(
            self._event("usage.unavailable", {"reason": "the close grace expired first"})
        )
        self.service._record(self._event("agent.upsert", {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "claude", "model": "unpriced-model",
            "effort": "low", "status": "completed", "result": {}, "usage": {},
        }))

        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual("the close grace expired first", row["usage_unavailable"])

    def test_usage_that_arrives_late_clears_the_missing_reason(self) -> None:
        """Round-4 finding: a row said the cost was both $0.25 and unknown."""

        self.service._record(self._event("agent.upsert", {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "claude", "model": "unpriced-model",
            "effort": "low", "status": "completed", "result": {}, "usage": {},
        }))
        self.service._record(self._event(
            "usage.unavailable",
            {"reason": "the close grace expired before final usage arrived"},
        ))
        self.assertIn(
            "usage_unavailable",
            json.loads(self.path.read_text(encoding="utf-8")),
        )

        # The provider's message was late rather than lost.
        self.service._record(
            self._event("usage.updated", {**self._usage(100), "cost_usd": 0.25})
        )

        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(0.25, row["cost_usd"])
        self.assertEqual(100, row["tokens"]["total"])
        self.assertNotIn("usage_unavailable", row)

    def test_late_usage_before_the_row_leaves_no_missing_reason(self) -> None:
        """The same race with the outcome row written last."""

        self.service._record(self._event(
            "usage.unavailable",
            {"reason": "the close grace expired before final usage arrived"},
        ))
        self.service._record(
            self._event("usage.updated", {**self._usage(100), "cost_usd": 0.25})
        )
        self.service._record(self._event("agent.upsert", {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "claude", "model": "unpriced-model",
            "effort": "low", "status": "completed", "result": {}, "usage": {},
        }))

        row = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(0.25, row["cost_usd"])
        self.assertNotIn("usage_unavailable", row)

    def test_a_repeated_price_at_an_unchanged_total_rewrites_nothing(self) -> None:
        view = {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "claude", "model": "unpriced-model",
            "effort": "low", "status": "completed", "result": {},
            "usage": {**self._usage(100), "cost_usd": 0.25},
        }
        self.service._record(self._event("agent.upsert", view))
        row = dict(self.service._outcome_rows["worker"])

        self.assertFalse(
            self.service._apply_outcome_usage(
                self.service._outcome_rows["worker"],
                {**self._usage(100), "cost_usd": 0.25},
            )
        )
        self.assertEqual(row, self.service._outcome_rows["worker"])


    def _priced_row(self, total: int) -> dict:
        self.service._record(self._event("agent.upsert", {
            "agent_id": "worker", "parent_agent_id": "primary",
            "role": "worker", "provider": "codex", "model": "gpt-6-sol",
            "effort": "low", "status": "completed", "result": {},
            "usage": self._usage(total),
        }))
        return self.service._outcome_rows["worker"]

    def test_an_older_usage_event_leaves_the_counts_and_their_price_alone(self) -> None:
        """A delayed observation of 100 tokens once re-priced a 1000-token row.

        The counts stayed at 1000 and the cost dropped to the price of 100, so
        one row disagreed with itself by ten times.
        """

        row = self._priced_row(1000)
        before = dict(row)
        self.assertIsNotNone(before["cost_usd"])

        self.service._record(self._event("usage.updated", self._usage(100)))

        self.assertEqual(before, self.service._outcome_rows["worker"])
        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(before["cost_usd"], on_disk["cost_usd"])

    def test_a_provider_price_at_the_same_total_still_replaces_the_estimate(self) -> None:
        row = self._priced_row(1000)
        self.assertEqual("api_rates", row["cost_source"])

        self.service._record(
            self._event("usage.updated", {**self._usage(1000), "cost_usd": 0.5})
        )

        row = self.service._outcome_rows["worker"]
        self.assertEqual(0.5, row["cost_usd"])
        self.assertEqual("provider", row["cost_source"])


class VNextMcpStdioTests(unittest.TestCase):
    """The transport a client uses when it launches the server itself."""

    def test_a_launching_client_lists_and_calls_over_a_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            worker = ScriptedWorker()
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: worker},
            )
            self.addCleanup(service.close)
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                    "name": "delegate",
                    "arguments": {"role": AgentRole.WORKER.value, "model_id": WORKER_MODEL,
                                  "objective": "Write the report section",
                                  "task_contract": {"criteria": ["section written"]}}}},
            ]
            source = io.StringIO(
                "".join(json.dumps(request) + chr(10) for request in requests)
                + "not json" + chr(10)
            )
            sink = io.StringIO()
            self.assertEqual(0, service.serve_stdio(stdin=source, stdout=sink))
            # Inside the workspace, not on cleanup: the service keeps writing a
            # roster while a delegated worker runs, and deleting the directory
            # out from under it left an empty-but-undeletable .vnext/status.
            service.close()

        answers = [json.loads(line) for line in sink.getvalue().splitlines()]
        # The notification is accepted silently, so four requests make four
        # answers only if the malformed line is answered too -- which it must be.
        # A tool call answers on its own thread, so it may land after a line that
        # arrived later; the client pairs an answer with its request by id, and
        # this test does the same.
        by_id = {answer["id"]: answer for answer in answers}
        self.assertEqual({1, 2, 3, None}, set(by_id))
        self.assertEqual(MCP_PROTOCOL_VERSIONS[-1], by_id[1]["result"]["protocolVersion"])
        self.assertIn("delegate", {tool["name"] for tool in by_id[2]["result"]["tools"]})
        self.assertFalse(by_id[3]["result"]["isError"], by_id[3])
        self.assertEqual(-32700, by_id[None]["error"]["code"])
        # The handshake keeps its place in the queue: a client that sent
        # tools/list before it was initialised would get an error back.
        self.assertLess(answers.index(by_id[1]), answers.index(by_id[2]))

    def test_an_inspect_answers_while_an_await_children_is_still_waiting(self) -> None:
        """A client could not look at the children it was waiting on.

        The loop read one line, answered it, and read the next.  An
        ``await_children`` over a child that has not finished waits for real --
        up to ``external_await_budget`` seconds -- so every request behind it
        waited too, `inspect` included.  The wait itself holds no scheduler
        lock (``vnext_scheduler.py`` await_children_blocking reads the roster
        and sleeps in slices), so a second call can genuinely be answered.
        """

        release = threading.Event()
        self.addCleanup(release.set)
        with tempfile.TemporaryDirectory() as workspace:
            worker = ScriptedWorker()
            worker.hold = release
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: worker},
            )
            self.addCleanup(service.close)
            created = json.loads(
                service.session.external_tool_call(tool="delegate", arguments={
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER_MODEL,
                    "objective": "Write the report section",
                    "task_contract": {"criteria": ["section written"]},
                }).as_json_text()
            )
            child_id = created["agent_id"]
            # Long enough that a serialised inspect could not slip in, short
            # enough that a broken run ends by itself.
            service.session._scheduler.external_await_budget = 4.0

            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "await_children",
                    "arguments": {"agent_ids": [child_id]}}},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                    "name": "inspect",
                    "arguments": {"agent_id": "self", "deep": False}}},
            ]
            lines = [json.dumps(request) + chr(10) for request in requests]

            written: list[str] = []
            write_lock = threading.Lock()

            class Sink:
                """What a pipe gives: a writer two threads may share."""

                def write(self, text: str) -> int:
                    with write_lock:
                        written.append(text)
                    return len(text)

                def flush(self) -> None:
                    return None

            served = threading.Thread(
                target=service.serve_stdio,
                kwargs={"stdin": iter(lines), "stdout": Sink()},
                name="serve-stdio-under-test",
                daemon=True,
            )
            served.start()

            def answers() -> dict:
                with write_lock:
                    text = "".join(written)
                return {
                    answer["id"]: answer
                    for answer in (json.loads(line) for line in text.splitlines() if line.strip())
                }

            deadline = time.monotonic() + 3.0
            while 3 not in answers() and time.monotonic() < deadline:
                time.sleep(0.02)
            seen = answers()
            self.assertIn(3, seen, "inspect never answered while the wait was open")
            self.assertNotIn(2, seen, "the wait had already ended, so nothing was proven")
            self.assertFalse(seen[3]["result"]["isError"], seen[3])

            release.set()
            served.join(timeout=20)
            self.assertFalse(served.is_alive())
            waited = answers()[2]
            self.assertFalse(waited["result"]["isError"], waited)
            service.close()

    def test_a_closed_pipe_does_not_hold_the_quit_past_the_grace(self) -> None:
        """EOF joined every call still open, however long it had left to run.

        A client stops this server by closing the pipe, and the restart proxy
        that does it (reloaderoo 1.1.5) sends SIGTERM and never follows with the
        SIGKILL its own timer arms, so this quit has to finish inside
        ``CLOSE_GRACE_SECONDS`` on its own.  Measured before the bound, on
        2026-09-30: EOF with one 50 s ``await_children`` open returned from
        serve_stdio after 50.06 s and the process exited after 50.11 s.
        """

        release = threading.Event()
        self.addCleanup(release.set)
        with tempfile.TemporaryDirectory() as workspace:
            worker = ScriptedWorker()
            worker.hold = release
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: worker},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            self.addCleanup(service.close)
            created = json.loads(
                service.session.external_tool_call(tool="delegate", arguments={
                    "role": AgentRole.WORKER.value,
                    "model_id": WORKER_MODEL,
                    "objective": "Hold this turn open",
                    "task_contract": {"criteria": ["held"]},
                }).as_json_text()
            )
            # Past the grace by a wide margin, so a quit that waited for this
            # call cannot pass for a quit that abandoned it.
            service.session._scheduler.external_await_budget = 20.0
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "await_children",
                    "arguments": {"agent_ids": [created["agent_id"]]}}},
            ]
            # The lines run out right after the wait request: that is the pipe
            # closing with one call still open.
            lines = iter([json.dumps(request) + chr(10) for request in requests])

            noise = io.StringIO()
            sink = _LockedSink()
            started = time.monotonic()
            with contextlib.redirect_stderr(noise):
                self.assertEqual(0, service.serve_stdio(stdin=lines, stdout=sink))
                drained = time.monotonic() - started
                release.set()
                service.close()
            whole_quit = time.monotonic() - started

        self.assertLess(drained, 3.0, f"the EOF drain took {drained:.1f}s")
        self.assertLess(
            whole_quit,
            server.CLOSE_GRACE_SECONDS,
            f"serve_stdio plus close took {whole_quit:.1f}s",
        )
        # The call that was abandoned is said out loud, on stderr, because
        # stdout belongs to the protocol.
        self.assertIn("unanswered", noise.getvalue())
        self.assertIn(1, sink.answers(), "the handshake was never answered")

    def test_a_call_that_raised_is_answered_under_the_id_it_came_with(self) -> None:
        """The error went back as ``id: null``, which answers nothing.

        A client pairs a response with its request by id and by nothing else.
        A handler that threw produced an error carrying no id, so a client
        holding several calls open could not tell which one had failed, and the
        call it sent stayed open with no answer it could recognise.
        """

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": ScriptedWorker},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            self.addCleanup(service.close)
            request = {"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {
                "name": "inspect", "arguments": {"agent_id": "self", "deep": False}}}
            sink = _LockedSink()

            def raises(_self: object, _payload: object) -> None:
                raise RuntimeError("probe boom")

            # A tool call with an id runs on its own thread, which is the path
            # that lost the id.
            with patch.object(server.McpProtocol, "handle", raises):
                self.assertEqual(0, service.serve_stdio(
                    stdin=iter([json.dumps(request) + chr(10)]), stdout=sink
                ))

        answers = sink.answers()
        self.assertEqual([99], list(answers), answers)
        self.assertEqual(-32603, answers[99]["error"]["code"])
        self.assertIn("probe boom", answers[99]["error"]["message"])

    def test_short_operator_token_is_a_plain_argument_error(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            token = "shortsecret"
            result = subprocess.run(
                [sys.executable, "-m", "vnext.vnext_mcp_server",
                 "--workspace", workspace, "--token", token,
                 *hermetic_server_arguments(workspace)],
                text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(2, result.returncode)
        self.assertEqual("", result.stdout)
        self.assertEqual(1, len(result.stderr.splitlines()))
        self.assertIn("--token must be at least 16 characters", result.stderr)
        self.assertNotIn(token, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_32_character_operator_token_starts_stdio_service(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            process = subprocess.run(
                [sys.executable, "-m", "vnext.vnext_mcp_server",
                 "--workspace", workspace, "--stdio", "--token", "x" * 32,
                 *hermetic_server_arguments(workspace)],
                input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]}}) + "\n",
                text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(0, process.returncode)
        self.assertIn('"id": 1', process.stdout)
        self.assertNotIn("Traceback", process.stderr)

    @unittest.skipIf(os.name == "nt", "SIGTERM is a hard TerminateProcess on Windows")
    def test_a_terminate_signal_closes_the_session_before_exiting(self) -> None:
        # The restart proxy stops the server with SIGTERM.  Without a handler the
        # process died without closing its session, and the Claude bridge kept
        # running every worker that was mid-turn with no parent to report to.
        with tempfile.TemporaryDirectory() as workspace:
            status = Path(workspace) / "status.json"
            process = subprocess.Popen(
                [sys.executable, "-m", "vnext.vnext_mcp_server", "--stdio",
                 "--workspace", workspace, "--status-file", str(status),
                 "--event-log", "none", "--outcome-log", "none",
                 *hermetic_server_arguments(workspace)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(process.kill)
            # An answer comes only once serve_stdio runs, so the handler is in place.
            process.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]}}) + chr(10))
            process.stdin.flush()
            self.assertEqual(1, json.loads(process.stdout.readline())["id"])
            self.assertTrue(json.loads(status.read_text(encoding="utf-8"))["live"])

            process.send_signal(signal.SIGTERM)
            self.assertEqual(0, process.wait(timeout=30))
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

            self.assertFalse(json.loads(status.read_text(encoding="utf-8"))["live"])


class UnencodableAnswerTests(unittest.TestCase):
    """An answer that will not serialise used to be no answer at all.

    json.dumps raised inside the writer, on a daemon thread with nobody to
    catch it, so the call it belonged to stayed open forever while the client
    waited.  The client gets -32603 under the same id, and the pipe keeps
    working for the calls after it.
    """

    def test_an_unserialisable_answer_becomes_an_error_under_its_own_id(self) -> None:
        real_handle = server.McpProtocol.handle

        def broken(self, payload):
            if payload.get("id") == 2:
                return {"jsonrpc": "2.0", "id": 2, "result": {"value": object()}}
            return real_handle(self, payload)

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": ScriptedWorker},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            self.addCleanup(service.close)
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            ]
            source = io.StringIO(
                "".join(json.dumps(request) + chr(10) for request in requests)
            )
            sink = _LockedSink()
            with patch.object(server.McpProtocol, "handle", broken):
                self.assertEqual(0, service.serve_stdio(stdin=source, stdout=sink))

        answers = sink.answers()
        self.assertIn(2, answers, answers)
        self.assertEqual(-32603, answers[2]["error"]["code"], answers[2])
        # Still serving: the request behind the broken one was answered.
        self.assertIn("delegate", {tool["name"] for tool in answers[3]["result"]["tools"]})


class VNextMcpCloseTests(unittest.TestCase):
    """What a quit is allowed to cost, and what a failed quit leaves behind."""

    def _service(self) -> VNextMcpService:
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        return VNextMcpService(
            workspace=workspace.name,
            catalog=[{"provider": "codex", "model": "gpt-5.6-sol"}],
            # An explicit factory owns its own credential check, so these
            # cases stop asking whether this machine is signed in to Codex.
            adapter_factories={"codex": ScriptedWorker},
            event_log=None,
            status_file=None,
            outcome_log=None,
        )

    def test_a_clean_close_says_so_on_stderr(self) -> None:
        """A clean close printed nothing, so a log read the same as a crash.

        Both quits land here -- a signal and the client closing the pipe --
        and the overrun and failure paths answer before this line, so one
        plain sentence on stderr is what tells a reader the workforce stopped
        on purpose.  stderr only: on stdio, stdout carries the protocol.
        """

        service = self._service()
        errors = io.StringIO()

        with contextlib.redirect_stderr(errors):
            service.close()

        self.assertTrue(service._closed)
        self.assertEqual(
            ["vnext: closed cleanly (0 agents recorded)"],
            errors.getvalue().splitlines(),
        )

    def test_a_single_recorded_agent_is_singular_on_close(self) -> None:
        service = self._service()
        service._roster["worker"] = {"agent_id": "worker"}
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            service.close()
        self.assertEqual(["vnext: closed cleanly (1 agent recorded)"],
                         errors.getvalue().splitlines())

    def test_a_close_that_overran_its_budget_does_not_claim_a_clean_close(self) -> None:
        service = self._service()
        errors = io.StringIO()

        def never_stops() -> None:
            time.sleep(0.3)

        with patch.object(service.session, "close", never_stops):
            with contextlib.redirect_stderr(errors):
                service.close(deadline=0.0)

        self.assertFalse(service._closed)
        self.assertNotIn("closed cleanly", errors.getvalue())

    def test_a_close_that_ran_out_of_time_keeps_counting_who_is_still_stopping(self) -> None:
        """R19 gpt-code F1: an overrun published a stopped roster.

        The close wrote ``live: false`` and ``working: 0`` before it checked
        whether its steps had finished, so a status line read nothing at work
        while a worker was still stopping under a live pid.  A close that ran
        out of time now says ``stopping`` and keeps the count; the close that
        does finish writes the stopped roster.
        """

        service = self._service()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        service._status_file = Path(folder.name) / "status.json"
        service._roster["worker"] = {"agent_id": "worker", "status": "running"}
        release = threading.Event()
        self.addCleanup(release.set)

        def stops_late() -> None:
            release.wait(30)

        def read() -> dict:
            return json.loads(service._status_file.read_text(encoding="utf-8"))

        with patch.object(service.session, "close", stops_late), contextlib.redirect_stderr(io.StringIO()):
            service.close(deadline=0.05)
            overran = read()
            # R20 gpt-code F2: an agent event arriving now published with the
            # defaults and wrote working 0 over the stopping roster.
            service._roster["other"] = {"agent_id": "other", "status": "running"}
            service._publish_status()
            late = read()
            release.set()
            service.close()
        stopped = read()

        self.assertFalse(overran["live"])
        self.assertTrue(overran["stopping"])
        self.assertEqual(1, overran["working"])
        self.assertFalse(late["live"])
        self.assertTrue(late["stopping"])
        self.assertEqual(2, late["working"])
        self.assertTrue(service._closed)
        self.assertFalse(stopped["live"])
        self.assertNotIn("stopping", stopped)
        self.assertEqual(0, stopped["working"])

    def test_a_close_that_failed_is_tried_again(self) -> None:
        """``_closed`` was set before the work, so nothing retried it.

        A workforce that raised on the way down left the bridge and every
        worker behind it running, and the second close saw the flag and
        returned without touching any of it.
        """

        service = self._service()
        attempts: list[int] = []

        def refuses_once() -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("the workforce will not stop")

        with patch.object(service.session, "close", refuses_once):
            with self.assertRaises(RuntimeError):
                service.close()
            self.assertFalse(service._closed, "a failed close must stay retryable")
            service.close()

        self.assertEqual(2, len(attempts))
        self.assertTrue(service._closed)

    def test_a_worker_whose_stop_hangs_cannot_hold_the_close(self) -> None:
        """The quit has to finish inside the grace the restart proxy allows.

        reloaderoo 1.1.5 arms its SIGKILL five seconds after the SIGTERM
        (dist/process-manager.js terminateChild), so the close carries that
        figure as its own budget and reports what is still stopping.
        """

        self.assertEqual(5.0, server.CLOSE_GRACE_SECONDS)
        service = self._service()
        release = threading.Event()
        self.addCleanup(release.set)

        def never_returns() -> None:
            release.wait(30)

        noise = io.StringIO()
        with patch.object(service.session, "close", never_returns):
            started = time.monotonic()
            with contextlib.redirect_stderr(noise):
                service.close(deadline=0.3)
            spent = time.monotonic() - started

        self.assertLess(spent, 3.0, f"the close took {spent:.1f}s")
        self.assertIn("workforce", noise.getvalue())
        self.assertFalse(service._closed, "an unfinished close must stay retryable")
        release.set()

    def test_two_closes_at_once_run_the_endpoint_step_once(self) -> None:
        """Nothing held one caller back, so the work was done twice.

        Measured on 2026-09-30 over three runs: two closes entered together
        stopped the endpoint twice every time, and in two of the three one
        caller raised RuntimeError ("cannot join thread before it is started")
        after reading the other caller's thread in the moment between its
        creation and its start.  A close is a stop, so a second caller waits on
        the one already running, inside its own deadline.
        """

        service = self._service()
        release = threading.Event()
        self.addCleanup(release.set)
        endpoint_calls: list[int] = []
        tally = threading.Lock()

        def endpoint_that_hangs() -> None:
            with tally:
                endpoint_calls.append(1)
            release.wait(30)

        real_thread = threading.Thread

        def slow_to_build(*args: object, **kwargs: object) -> threading.Thread:
            # Widens the window the two callers raced in: whoever is deciding
            # whether the step already has a thread holds that decision open
            # for as long as building one takes.
            if kwargs.get("name") == "vnext-close-endpoint":
                time.sleep(0.2)
            return real_thread(*args, **kwargs)

        both_in = threading.Barrier(2)
        errors: list[str] = []

        def closer() -> None:
            try:
                both_in.wait(5)
                service.close(deadline=1.0)
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                errors.append(f"{type(exc).__name__}: {exc}")

        noise = io.StringIO()
        callers = [real_thread(target=closer, name=f"closer-{index}")
                   for index in range(2)]
        with patch.object(service, "_close_relay", endpoint_that_hangs), \
                patch.object(threading, "Thread", slow_to_build), \
                contextlib.redirect_stderr(noise):
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(20)

        self.assertEqual([], [caller.name for caller in callers if caller.is_alive()])
        self.assertEqual([], errors, "a caller of close was handed an exception")
        self.assertEqual(
            1, len(endpoint_calls),
            f"the endpoint step ran {len(endpoint_calls)} times",
        )
        self.assertFalse(service._closed, "an unfinished close must stay retryable")
        release.set()

    def test_a_delegate_after_the_workforce_closed_starts_no_worker(self) -> None:
        """What makes the half-closed window in a close safe.

        The steps share one deadline and a step that overruns keeps running, so
        a hung endpoint leaves the relay accepting calls while the workforce
        stops behind it.  Under a hard deadline that is the intended trade.  It
        is safe because a call arriving then is refused before it can reach a
        tree, and that refusal lives in the external adapter's own closed flag
        rather than here -- one layer away from the close that depends on it,
        with nothing naming it.  Measured on 2026-09-30, a delegate arriving
        after the workforce closed built no provider adapter at all.
        """

        built: list[ScriptedWorker] = []

        def counted() -> ScriptedWorker:
            worker = ScriptedWorker()
            built.append(worker)
            return worker

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": counted},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            self.addCleanup(service.close)
            service.session.close()
            before = len(built)
            with self.assertRaises(RuntimeError) as refused:
                # A short budget on purpose.  The default is 30 seconds of
                # waiting for a binding that has already gone, and the refusal
                # is the whole subject here.
                service.session.external_tool_call(
                    tool="delegate",
                    arguments={
                        "role": AgentRole.WORKER.value,
                        "model_id": WORKER_MODEL,
                        "objective": "Slip past a close",
                        "task_contract": {"criteria": ["never runs"]},
                    },
                    timeout=0.5,
                )

        self.assertIn("not bound to a control tree", str(refused.exception))
        self.assertEqual(before, len(built), "a worker was built after the close")

    @unittest.skipIf(os.name == "nt", "SIGTERM is a hard TerminateProcess on Windows")
    def test_a_terminate_signal_closes_the_session_in_http_mode_too(self) -> None:
        """Only Ctrl-C was handled on the listening path.

        Python's default for SIGTERM ends the process without unwinding, so the
        close never ran: the workforce outlived the endpoint with every worker
        still in a turn.  A service manager, a closing terminal and the restart
        proxy all send SIGTERM rather than SIGINT.
        """

        with tempfile.TemporaryDirectory() as workspace:
            status = Path(workspace) / "status.json"
            process = subprocess.Popen(
                # -u because stdout is a pipe here: the announcement lines the
                # listening path prints are block-buffered otherwise and this
                # test would wait on a line the process is still holding.
                [sys.executable, "-u", "-m", "vnext.vnext_mcp_server",
                 "--workspace", workspace, "--port", "0", "--status-file", str(status),
                 "--event-log", "none", "--outcome-log", "none",
                 *hermetic_server_arguments(workspace)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(process.kill)
            # The endpoint line is printed once the socket is bound and the
            # handler is in place, so it is the signal that the wait has begun.
            # Read it on another thread: readline would block for good on a
            # process that never gets there, and take the suite with it.
            announced = threading.Event()
            threading.Thread(
                target=lambda: [
                    announced.set()
                    for line in iter(process.stdout.readline, "")
                    if "endpoint:" in line
                ],
                daemon=True,
            ).start()
            self.assertTrue(
                announced.wait(60), "the HTTP endpoint never announced itself"
            )
            self.assertTrue(json.loads(status.read_text(encoding="utf-8"))["live"])

            process.send_signal(signal.SIGTERM)
            self.assertEqual(0, process.wait(timeout=30))

            self.assertFalse(json.loads(status.read_text(encoding="utf-8"))["live"])

    def test_the_endpoint_line_reaches_a_pipe_reader_at_once(self) -> None:
        """No `-u` here, on purpose.

        A supervisor that starts this server reads its stdout through a pipe,
        and a pipe is block-buffered: the announcement lines stayed in the
        buffer until the server stopped, so whoever waited for the endpoint
        waited for the whole run.  Measured on the merged branch: the line
        arrived in 0.87 s with `-u` and never within 15 s without it.
        """

        with tempfile.TemporaryDirectory() as workspace:
            process = subprocess.Popen(
                [sys.executable, "-m", "vnext.vnext_mcp_server",
                 "--workspace", workspace, "--port", "0",
                 "--status-file", str(Path(workspace) / "status.json"),
                 "--event-log", "none", "--outcome-log", "none",
                 *hermetic_server_arguments(workspace)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(process.kill)
            # On another thread: readline blocks for good on a buffer nobody
            # flushed, and would take the suite with it.
            announced = threading.Event()
            threading.Thread(
                target=lambda: [
                    announced.set()
                    for line in iter(process.stdout.readline, "")
                    if "endpoint:" in line
                ],
                daemon=True,
            ).start()
            arrived = announced.wait(10)
            # Still running, so the line came from a flush rather than from
            # the buffer emptying on exit.
            self.assertIsNone(process.poll(), "the server exited before it announced itself")
            self.assertTrue(arrived, "the endpoint line never left the stdout buffer")

    def test_a_delegate_arriving_during_a_close_is_refused_over_the_pipe(self) -> None:
        """The window between the start of a close and the end of it.

        The older test only proved a delegate is refused once the workforce had
        already closed.  close() stops the endpoint first, and a step that
        overruns keeps its thread, so a hung endpoint step leaves the session
        accepting for as long as it hangs -- and a delegate arriving then built
        a real provider worker that nothing was left to stop.  The client now
        gets a refusal under its own request id instead.
        """

        built: list[ScriptedWorker] = []

        def counted() -> ScriptedWorker:
            worker = ScriptedWorker()
            built.append(worker)
            return worker

        entered = threading.Event()
        release = threading.Event()

        def hung_endpoint() -> None:
            entered.set()
            release.wait(15)

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": counted},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            closer = threading.Thread(
                target=lambda: service.close(deadline=8.0), name="closer"
            )

            def requests():
                yield json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": MCP_PROTOCOL_VERSIONS[-1]},
                }) + chr(10)
                yield json.dumps(
                    {"jsonrpc": "2.0", "method": "notifications/initialized"}
                ) + chr(10)
                closer.start()
                self.assertTrue(
                    entered.wait(10), "the close never reached its endpoint step"
                )
                yield json.dumps({
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                        "name": "delegate",
                        "arguments": {
                            "role": AgentRole.WORKER.value,
                            "model_id": WORKER_MODEL,
                            "objective": "Slip in while the close is under way",
                            "task_contract": {"criteria": ["never runs"]},
                        },
                    },
                }) + chr(10)

            sink = _LockedSink()
            try:
                with patch.object(service, "_close_relay", hung_endpoint):
                    self.assertEqual(
                        0, service.serve_stdio(stdin=requests(), stdout=sink)
                    )
            finally:
                release.set()
                closer.join(15)
                service.close(deadline=5.0)

        answers = sink.answers()
        self.assertIn(2, answers, answers)
        self.assertTrue(answers[2]["result"]["isError"], answers[2])
        self.assertEqual([], built, "a worker was built while the session was closing")


class AChildAlreadyRunningWhenACloseBeginsTests(unittest.TestCase):
    """The close guard on the client's seam is not on the path a child takes.

    A branch manager already in a turn when close() starts calls delegate from
    inside its own provider handler.  That goes to the scheduler and on to a
    spawn without ever passing the service seam, so during a close whose
    endpoint step was still running the child's delegate answered success and a
    second provider thread started.
    """

    def test_a_running_child_cannot_start_a_worker_while_the_close_waits(self) -> None:
        branch_running = threading.Event()
        endpoint_entered = threading.Event()
        release = threading.Event()
        child_call_done = threading.Event()
        answer: dict = {}
        built: list = []

        class RunningBranch(ScriptedWorker):
            """A branch manager that delegates from the middle of its turn."""

            def start_turn(self, *, thread_id, prompt, **kwargs):
                if self._turns >= 1:
                    return super().start_turn(thread_id=thread_id, prompt=prompt, **kwargs)
                branch_running.set()
                if not endpoint_entered.wait(15):
                    raise AssertionError("the close never reached its endpoint step")
                try:
                    result = self.handlers[thread_id](
                        "delegate",
                        {
                            "role": AgentRole.WORKER.value,
                            "model_id": WORKER_MODEL,
                            "objective": "A grandchild born during the close",
                            "task_contract": {"criteria": ["never runs"]},
                        },
                        None,
                    )
                    answer.update(json.loads(result.as_json_text()))
                    answer["success"] = result.success
                except BaseException as exc:  # a refusal must not come as one
                    answer["exception"] = f"{type(exc).__name__}: {exc}"
                finally:
                    child_call_done.set()
                return super().start_turn(thread_id=thread_id, prompt=prompt, **kwargs)

        def counted() -> ScriptedWorker:
            worker = RunningBranch() if not built else ScriptedWorker()
            built.append(worker)
            return worker

        def hung_endpoint() -> None:
            endpoint_entered.set()
            release.wait(15)

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": counted},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            closer = threading.Thread(
                target=lambda: service.close(deadline=8.0), name="closer"
            )
            try:
                with patch.object(service, "_close_relay", hung_endpoint):
                    started = service._dispatch(
                        "delegate",
                        {
                            "role": AgentRole.BRANCH_MANAGER.value,
                            "model_id": WORKER_MODEL,
                            "objective": "Run while the close begins",
                            "task_contract": {"criteria": ["one turn"]},
                        },
                        {},
                    )
                    self.assertTrue(started.success, started)
                    self.assertTrue(
                        branch_running.wait(15), "the branch manager never ran"
                    )
                    agents_before = len(service.session.snapshot()["agents"])
                    closer.start()
                    self.assertTrue(
                        endpoint_entered.wait(15),
                        "the close never reached its endpoint step",
                    )
                    self.assertTrue(
                        child_call_done.wait(15), "the child's delegate never answered"
                    )
                    threads_started = built[0]._threads
                    agents_after = len(service.session.snapshot()["agents"])
            finally:
                release.set()
                closer.join(15)
                service.close(deadline=5.0)

        self.assertNotIn("exception", answer, answer)
        self.assertFalse(answer.get("success", True), answer)
        self.assertEqual("session-closing", answer.get("error_code"), answer)
        self.assertIn("delegate", str(answer.get("error", "")), answer)
        self.assertEqual(
            1,
            threads_started,
            "a provider thread was started for a worker born during the close",
        )
        self.assertEqual(
            agents_before,
            agents_after,
            "an agent record was created before the refusal",
        )
        self.assertEqual(
            1, len(built), "a second runtime was built during the close"
        )


class NoProviderWorkStartsOnceTheCloseBeginsTests(unittest.TestCase):
    """Two roads that still reached the provider after the closing flag was set.

    The guards added earlier sit inside the control plane, where a record is
    created.  A replacement stops its target's turn before it reaches one, and a
    child that is already READY needs no control-plane call at all: the
    scheduler's own scan starts its turn.  Both are measured against a close
    whose endpoint step is held, so the session and the service both read
    closing while the provider is still being asked to work.
    """

    def test_a_replace_refused_by_the_close_never_interrupts_its_target(self) -> None:
        branch_started = threading.Event()
        worker_started = threading.Event()
        release_turns = threading.Event()
        endpoint_entered = threading.Event()
        release_endpoint = threading.Event()
        interrupts: list = []

        class HeldWorker(ScriptedWorker):
            """Two turns that stay open, and an interrupt that is only counted."""

            def start_turn(self, *, thread_id, prompt, **kwargs):
                handle = super().start_turn(thread_id=thread_id, prompt=prompt, **kwargs)
                (branch_started if thread_id == "worker-thread-1" else worker_started).set()
                return handle

            def wait_turn(self, handle, *, timeout=300):
                release_turns.wait(15)
                return {
                    "status": "stopped",
                    "final_response": "",
                    "terminal_reason": "the test stopped this turn",
                }

            def interrupt(self, handle):
                interrupts.append((handle.thread_id, handle.turn_id))

        adapter = HeldWorker()

        def hung_endpoint() -> None:
            endpoint_entered.set()
            release_endpoint.wait(15)

        answer: dict = {}
        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: adapter},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            closer = None
            try:
                branch = service._dispatch(
                    "delegate",
                    {
                        "role": AgentRole.BRANCH_MANAGER.value,
                        "model_id": WORKER_MODEL,
                        "objective": "A manager whose child is replaced",
                        "task_contract": {"criteria": ["one turn"]},
                    },
                    {},
                )
                self.assertTrue(branch.success, branch)
                self.assertTrue(branch_started.wait(15), "the manager never ran")
                child = json.loads(
                    adapter.handlers["worker-thread-1"](
                        "delegate",
                        {
                            "role": AgentRole.WORKER.value,
                            "model_id": WORKER_MODEL,
                            "objective": "The child a replacement would stop",
                            "task_contract": {"criteria": ["one turn"]},
                        },
                        None,
                    ).as_json_text()
                )
                self.assertTrue(worker_started.wait(15), "the child never ran")
                with patch.object(service, "_close_relay", hung_endpoint):
                    closer = threading.Thread(
                        target=lambda: service.close(deadline=10.0), name="closer"
                    )
                    closer.start()
                    self.assertTrue(
                        endpoint_entered.wait(15),
                        "the close never reached its endpoint step",
                    )
                    before_replace = list(interrupts)
                    replaced = adapter.handlers["worker-thread-1"](
                        "replace",
                        {
                            "agent_id": child["agent_id"],
                            "model_id": WORKER_MODEL,
                            "task_contract": {"criteria": ["a replacement"]},
                        },
                        None,
                    )
                    answer.update(json.loads(replaced.as_json_text()))
                    answer["success"] = replaced.success
                    after_replace = list(interrupts)
            finally:
                release_turns.set()
                release_endpoint.set()
                if closer is not None:
                    closer.join(15)
                service.close(deadline=5.0)

        self.assertEqual([], before_replace, "something interrupted a turn too early")
        self.assertFalse(answer.get("success", True), answer)
        self.assertEqual("session-closing", answer.get("error_code"), answer)
        self.assertIn("replace", str(answer.get("error", "")), answer)
        self.assertEqual(
            [],
            after_replace,
            "the refused replacement still stopped its target's provider turn",
        )

    def test_a_ready_child_starts_no_turn_once_the_close_has_begun(self) -> None:
        binding_entered = threading.Event()
        release_binding = threading.Event()
        endpoint_entered = threading.Event()
        release_endpoint = threading.Event()
        readiness_asked = threading.Event()
        turn_started = threading.Event()

        class SlowBindWorker(ScriptedWorker):
            """A child whose runtime binding spans the start of the close."""

            def start_thread(self, **kwargs):
                value = super().start_thread(**kwargs)
                binding_entered.set()
                release_binding.wait(15)
                return value

            def can_start_turn(self, thread_id):
                # The scheduler asks this after binding and before it starts a
                # turn, so it is the moment the READY scan has got past the
                # slow bind above.  Whatever happens next is the decision under
                # test.
                readiness_asked.set()
                return True

            def start_turn(self, *, thread_id, prompt, **kwargs):
                handle = super().start_turn(thread_id=thread_id, prompt=prompt, **kwargs)
                turn_started.set()
                return handle

            def wait_turn(self, handle, *, timeout=300):
                return {
                    "status": "stopped",
                    "final_response": "",
                    "terminal_reason": "the test stopped this turn",
                }

        adapter = SlowBindWorker()

        def hung_endpoint() -> None:
            endpoint_entered.set()
            release_endpoint.wait(15)

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: adapter},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            closer = None
            try:
                started = service._dispatch(
                    "delegate",
                    {
                        "role": AgentRole.WORKER.value,
                        "model_id": WORKER_MODEL,
                        "objective": "A child bound while the close begins",
                        "task_contract": {"criteria": ["never runs"]},
                    },
                    {},
                )
                self.assertTrue(started.success, started)
                self.assertTrue(binding_entered.wait(15), "the binding never began")
                with patch.object(service, "_close_relay", hung_endpoint):
                    closer = threading.Thread(
                        target=lambda: service.close(deadline=10.0), name="closer"
                    )
                    closer.start()
                    self.assertTrue(
                        endpoint_entered.wait(15),
                        "the close never reached its endpoint step",
                    )
                    session = service.session.control.sessions[
                        service.session.root.session_id
                    ]
                    self.assertTrue(session.closing, "the session was not marked closing")
                    release_binding.set()
                    self.assertTrue(
                        readiness_asked.wait(15),
                        "the scan never got past the runtime binding",
                    )
                    started_a_turn = turn_started.wait(3)
                    # The binding was already under way, so its provider thread
                    # exists.  What must not follow is a turn on it.
                    threads_during_close = adapter._threads
                    turns_during_close = adapter._turns
            finally:
                release_binding.set()
                release_endpoint.set()
                if closer is not None:
                    closer.join(15)
                service.close(deadline=5.0)

        self.assertEqual(
            1, threads_during_close, "an unexpected number of provider threads"
        )
        self.assertFalse(
            started_a_turn,
            "a provider turn started after the session was marked closing",
        )
        self.assertEqual(0, turns_during_close, "the provider was given a turn")


    def test_a_close_landing_after_the_guard_starts_no_provider_turn(self) -> None:
        """The gap the earlier guard left: the check passed, then the close began.

        The scheduler read ``closing``, found it false, and went on to build the
        prompt.  The close landed in that gap, and the adapter was still told to
        start a turn afterwards, because nothing re-read the flag at the moment
        the turn was created.  Admission is now one step under ``session.lock``:
        the control plane reads ``closing`` and makes the READY to RUNNING
        transition in the same locked block, so a close cannot slip between
        them.  The prompt the refused turn built is given back, mail included.
        """

        prompt_entered = threading.Event()
        release_prompt = threading.Event()
        endpoint_entered = threading.Event()
        release_endpoint = threading.Event()
        turn_started = threading.Event()
        counted: dict = {}

        class CountingWorker(ScriptedWorker):
            def start_turn(self, *, thread_id, prompt, **kwargs):
                handle = super().start_turn(thread_id=thread_id, prompt=prompt, **kwargs)
                turn_started.set()
                return handle

            def wait_turn(self, handle, *, timeout=300):
                return {
                    "status": "stopped",
                    "final_response": "",
                    "terminal_reason": "the test stopped this turn",
                }

        adapter = CountingWorker()

        def hung_endpoint() -> None:
            endpoint_entered.set()
            release_endpoint.wait(15)

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: adapter},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            closer = None
            try:
                scheduler = service.session._scheduler
                original_prompt = scheduler._agent_prompt

                def held_prompt(agent):
                    # One unread message, so the restore below has something to
                    # restore.  Building the prompt is what marks mail read.
                    agent.messages.append(
                        AgentMessage(
                            scheduler.root.agent_id,
                            agent.agent_id,
                            "mail that no prompt may eat",
                            "message",
                            time.time(),
                            False,
                        )
                    )
                    counted["unread_before"] = (
                        len(agent.messages) - agent.delivered_message_count
                    )
                    value = original_prompt(agent)
                    counted["unread_after_prompt"] = (
                        len(agent.messages) - agent.delivered_message_count
                    )
                    prompt_entered.set()
                    release_prompt.wait(15)
                    return value

                with patch.object(scheduler, "_agent_prompt", held_prompt), \
                        patch.object(service, "_close_relay", hung_endpoint):
                    started = service._dispatch(
                        "delegate",
                        {
                            "role": AgentRole.WORKER.value,
                            "model_id": WORKER_MODEL,
                            "objective": "A child whose prompt spans the close",
                            "task_contract": {"criteria": ["never runs"]},
                        },
                        {},
                    )
                    self.assertTrue(started.success, started)
                    self.assertTrue(
                        prompt_entered.wait(15),
                        "the scheduler never got past the closing guard",
                    )
                    session = service.session.control.sessions[
                        service.session.root.session_id
                    ]
                    child = next(
                        agent
                        for agent in session.agents.values()
                        if agent.role is AgentRole.WORKER
                    )
                    self.assertIsNone(child.active_turn_id)
                    closer = threading.Thread(
                        target=lambda: service.close(deadline=10.0), name="closer"
                    )
                    closer.start()
                    self.assertTrue(
                        endpoint_entered.wait(15),
                        "the close never reached its endpoint step",
                    )
                    self.assertTrue(session.closing, "the session was not marked closing")
                    release_prompt.set()
                    started_a_turn = turn_started.wait(3)
                    turns_during_close = adapter._turns
                    release_endpoint.set()
                    closer.join(15)
                    self.assertFalse(closer.is_alive(), "the close never finished")
                    counted["unread_after_refusal"] = (
                        len(child.messages) - child.delivered_message_count
                    )
                    closed = service._closed
                    final_status = child.status.value
                    final_turn_id = child.active_turn_id
            finally:
                release_prompt.set()
                release_endpoint.set()
                if closer is not None:
                    closer.join(15)
                service.close(deadline=5.0)

        self.assertFalse(
            started_a_turn,
            "a provider turn started after the session was marked closing",
        )
        self.assertEqual(0, turns_during_close, "the provider was given a turn")
        self.assertEqual("cancelled", final_status, "the close left the child behind")
        self.assertIsNone(final_turn_id, "a refused turn left an active turn id")
        self.assertTrue(closed, "the close did not finish inside its budget")
        self.assertEqual(1, counted["unread_before"])
        self.assertEqual(0, counted["unread_after_prompt"])
        self.assertEqual(
            1,
            counted["unread_after_refusal"],
            "the refused turn kept the mail its prompt had marked read",
        )

    def test_a_turn_admitted_before_the_close_is_ended_by_it(self) -> None:
        """The other order: admission wins, and then the close has to undo it.

        Admission is the control plane's READY to RUNNING transition, so a turn
        that won the lock is already visible to the close as a running agent
        with an active turn id.  The close waits for the start to return by
        joining the scheduler thread, then interrupts the turn and cancels the
        agent.  Nothing is left running behind the close.
        """

        start_entered = threading.Event()
        release_start = threading.Event()
        release_wait = threading.Event()
        interrupts: list = []

        class HeldStartWorker(ScriptedWorker):
            def start_turn(self, *, thread_id, prompt, **kwargs):
                handle = super().start_turn(thread_id=thread_id, prompt=prompt, **kwargs)
                start_entered.set()
                release_start.wait(15)
                return handle

            def wait_turn(self, handle, *, timeout=300):
                release_wait.wait(15)
                return {
                    "status": "stopped",
                    "final_response": "",
                    "terminal_reason": "the test stopped this turn",
                }

            def interrupt(self, handle):
                # A real provider's interrupt is what ends the open turn, so
                # the wait above only returns once the close has asked for it.
                interrupts.append((handle.thread_id, handle.turn_id))
                release_wait.set()

        adapter = HeldStartWorker()

        with tempfile.TemporaryDirectory() as workspace:
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": lambda: adapter},
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
            closer = None
            try:
                started = service._dispatch(
                    "delegate",
                    {
                        "role": AgentRole.WORKER.value,
                        "model_id": WORKER_MODEL,
                        "objective": "A child admitted just before the close",
                        "task_contract": {"criteria": ["one turn"]},
                    },
                    {},
                )
                self.assertTrue(started.success, started)
                self.assertTrue(
                    start_entered.wait(15), "the provider turn never began"
                )
                session = service.session.control.sessions[
                    service.session.root.session_id
                ]
                child = next(
                    agent
                    for agent in session.agents.values()
                    if agent.role is AgentRole.WORKER
                )
                self.assertEqual("running", child.status.value)
                admitted_turn_id = child.active_turn_id
                self.assertIsNotNone(
                    admitted_turn_id, "the admitted turn left no reservation"
                )
                closer = threading.Thread(
                    target=lambda: service.close(deadline=20.0), name="closer"
                )
                closer.start()
                deadline = time.monotonic() + 15
                while not session.closing and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(session.closing, "the close never began")
                # Only the start is released.  The turn itself stays open, so
                # the close has to end it rather than outwait it.
                release_start.set()
                closer.join(20)
                self.assertFalse(closer.is_alive(), "the close never finished")
                closed = service._closed
                final_status = child.status.value
                final_turn_id = child.active_turn_id
                turns = adapter._turns
                stopped = list(interrupts)
            finally:
                release_start.set()
                release_wait.set()
                if closer is not None:
                    closer.join(20)
                service.close(deadline=5.0)

        self.assertEqual(1, turns, "the close let a second turn start")
        self.assertIn(
            final_status,
            {"cancelled", "completed"},
            "the admitted turn outlived the close",
        )
        self.assertEqual("cancelled", final_status, "the close did not end the child")
        self.assertIsNone(final_turn_id, "the close left an active turn id behind")
        self.assertTrue(stopped, "the close never stopped the admitted turn")
        self.assertTrue(closed, "the close did not finish inside its budget")


class RunFolderIsAFileTests(unittest.TestCase):
    """A regular file named .vnext took the whole session down at startup.

    Same arrangement as a read-only .vnext, and the same answer: the session is
    worth more than the records, so it starts and says on stderr that it keeps
    none.
    """

    def test_a_file_where_the_run_folder_goes_still_starts_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # The service resolves its workspace, and on Windows that turns a
            # short 8.3 temp name such as RUNNER~1 into the long one.
            workspace = Path(temporary).resolve()
            (workspace / ".vnext").write_text("not a folder\n", encoding="utf-8")
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise):
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                    client=CLIENT,
                    adapter_factories={"codex": ScriptedWorker},
                )
            try:
                self.assertIsNone(service.status_file)
                said = [line for line in noise.getvalue().splitlines() if line.strip()]
                self.assertEqual(1, len(said), said)
                self.assertIn(str(workspace / ".vnext"), said[0])
                self.assertIn("record", said[0])
            finally:
                service.close()


class VNextMcpServiceArgumentTests(unittest.TestCase):
    def test_http_bind_refusal_is_one_line_and_closes_service(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            flags = hermetic_server_arguments(workspace)
            output, errors = io.StringIO(), io.StringIO()
            closed = []
            original_close = VNextMcpService.close

            def close_and_record(service, *args, **kwargs):
                original_close(service, *args, **kwargs)
                closed.append(service._closed)

            with patch("vnext.vnext_codex_mcp.ThreadingHTTPServer", side_effect=OSError(48, "Address already in use")), \
                 patch.object(VNextMcpService, "close", autospec=True, side_effect=close_and_record), \
                 contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                code = server.main(["--workspace", workspace, *flags, "--host", "127.0.0.1", "--port", "8765"])
            self.assertEqual(1, code)
            self.assertEqual("", output.getvalue())
            self.assertEqual(1, len(errors.getvalue().splitlines()))
            self.assertIn("vnext cannot start:", errors.getvalue())
            self.assertIn("127.0.0.1:8765", errors.getvalue())
            self.assertIn("--port 0", errors.getvalue())
            self.assertEqual([True], closed)

    def test_non_loopback_host_is_a_parser_refusal(self) -> None:
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = server.main(["--host", "0.0.0.0", "--port", "0"])
        self.assertEqual(2, code)
        self.assertEqual("", output.getvalue())
        self.assertEqual(1, len(errors.getvalue().splitlines()))
        self.assertIn("loopback", errors.getvalue())

    def test_loopback_host_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            flags = hermetic_server_arguments(workspace)
            with patch.object(VNextMcpService, "serve_stdio", return_value=0):
                code = server.main(["--workspace", workspace, *flags, "--host", "127.0.0.1", "--stdio"])
            self.assertEqual(0, code)

    def test_a_missing_workspace_is_refused(self) -> None:
        with self.assertRaises(VNextMcpServiceError):
            VNextMcpService(workspace=Path(tempfile.gettempdir()) / "no-such-workspace-9f2")

    def test_an_empty_catalog_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaises(VNextMcpServiceError):
                VNextMcpService(workspace=workspace, catalog=[])

    # The subject here is the pinned CLI's model allowlist, so the Codex
    # login is stubbed: without it these two read the host's ~/.codex and
    # failed with "a workforce needs at least one available worker model" on
    # a machine that has not run `codex login`.
    def test_a_codex_model_the_pin_cannot_run_is_refused_at_startup(self) -> None:
        model = "gpt-99-unknown"
        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(server, "_codex_login_available", return_value=True):
                with self.assertRaises(VNextMcpServiceError) as caught:
                    VNextMcpService(
                        workspace=workspace,
                        catalog=[{"provider": "codex", "model": model}],
                    )
        self.assertIn(model, str(caught.exception))
        self.assertIn(server.RELEASE_CODEX_VERSION, str(caught.exception))

    def test_a_catalog_supported_by_the_pin_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(server, "_codex_login_available", return_value=True):
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "codex", "model": "gpt-5.6-sol"}],
                    event_log=None,
                    status_file=None,
                    outcome_log=None,
                )
            self.addCleanup(service.close)
            self.assertIn("gpt-5.6-sol", service.session.registry.cards)

    def test_provider_cli_factory_reaches_the_runtime(self) -> None:
        seen: dict = {}

        def serve_stdio(service, stdin=None, stdout=None):
            seen.update(service.session._factories)
            return 0

        with tempfile.TemporaryDirectory() as workspace:
            catalog = Path(workspace) / "catalog.json"
            catalog.write_text(
                json.dumps({"models": [
                    {"provider": "scripted", "model": WORKER_MODEL},
                    {"provider": "scripted-two", "model": "worker-model-two"},
                ]}),
                encoding="utf-8",
            )
            with patch.object(VNextMcpService, "serve_stdio", autospec=True,
                              side_effect=serve_stdio):
                result = server.main([
                    "--workspace", workspace,
                    "--catalog", str(catalog),
                    "--provider",
                    "scripted=tests.test_vnext_external_primary:ScriptedWorker",
                    "--provider",
                    "scripted-two=tests.test_vnext_external_primary:ScriptedWorker",
                    "--event-log", "none",
                    "--status-file", "none",
                    "--outcome-log", "none",
                    "--stdio",
                ])

        self.assertEqual(0, result)
        self.assertIs(ScriptedWorker, seen["scripted"])
        self.assertIs(ScriptedWorker, seen["scripted-two"])

    def _refused_start(self, argv: list[str]) -> str:
        """Run a start that must be refused and return the one line it printed.

        The refusal used to leave VNextMcpServiceError to the interpreter, so a
        person who mistyped a flag read a Python traceback where ``--check`` --
        the same question, asked of the same resolution -- prints one line.
        """

        errors = io.StringIO()
        with patch.object(sys, "stderr", errors):
            code = server.main(argv)
        self.assertEqual(1, code)
        printed = errors.getvalue().strip().splitlines()
        self.assertEqual(1, len(printed), printed)
        self.assertTrue(printed[0].startswith("vnext cannot start: "), printed[0])
        return printed[0]

    def test_a_malformed_provider_registration_fails_at_startup(self) -> None:
        line = self._refused_start(["--provider", "scripted", "--stdio"])
        self.assertIn("malformed", line)
        self.assertIn("NAME=module:attribute", line)

    def test_an_unimportable_provider_module_fails_at_startup(self) -> None:
        module = "no_such_vnext_provider_module"
        line = self._refused_start(["--provider", f"scripted={module}:factory", "--stdio"])
        self.assertIn(module, line)
        self.assertIn("could not be imported", line)

    def test_a_missing_provider_attribute_fails_at_startup(self) -> None:
        attribute = "no_such_adapter_factory"
        line = self._refused_start(["--provider", f"scripted=json:{attribute}", "--stdio"])
        self.assertIn(attribute, line)
        self.assertIn("has no attribute", line)

    def test_a_provider_cannot_shadow_a_built_in(self) -> None:
        line = self._refused_start([
            "--provider", "codex=tests.test_vnext_external_primary:ScriptedWorker",
            "--stdio",
        ])
        self.assertIn("collides with a built-in provider", line)


if __name__ == "__main__":
    unittest.main()


class TheListingNamesTheModelsTests(unittest.TestCase):
    """A client vNext does not run reads model names from the tool listing alone."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp.name).resolve()
        self.addCleanup(self._temp.cleanup)
        # The Command Code card is dropped when the machine holds no key, so the
        # roster is tested against a credential this test owns.
        credential = patch.object(server, "load_commandcode_provider", lambda: object())
        credential.start()
        self.addCleanup(credential.stop)
        self.service = VNextMcpService(
            workspace=self.workspace,
            catalog=[
                {"provider": "codex", "model": WORKER_MODEL},
                {"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"},
            ],
            client=CLIENT,
            port=0,
            event_log=self.workspace / "events.jsonl",
            adapter_factories={"codex": lambda: ScriptedWorker()},
        )
        self.addCleanup(self.service.close)
        self.tools = {tool["name"]: tool for tool in self.service._tools}

    def test_the_tools_that_choose_a_model_name_every_one_on_offer(self) -> None:
        for name in ("delegate", "replace"):
            description = self.tools[name]["description"]
            self.assertIn("deepseek/deepseek-v4.1-flash (commandcode via app-server)", description)
            self.assertIn(f"{WORKER_MODEL} (codex via app-server)", description)

    def test_the_client_s_own_model_is_left_out_of_the_roster(self) -> None:
        self.assertNotIn(CLIENT, self.tools["delegate"]["description"])

    def test_a_tool_that_chooses_no_model_carries_no_roster(self) -> None:
        self.assertNotIn("deepseek", self.tools["await_children"]["description"])


class ClaudeExecutableResolutionTests(unittest.TestCase):
    """The login check has to find the CLI where the installers put it.

    Round-1 findings 3 and 4: the check ran the bare name ``claude``, which a
    GUI-launched Claude Code cannot resolve (launchd PATH omits
    ``~/.local/bin``) and which Windows resolves without walking PATHEXT, so a
    ``claude.cmd`` shim was invisible. Windows is simulated, per the release
    plan: no Windows host is available here.
    """

    def test_simulated_windows_finds_the_claude_cmd_shim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shim = root / "bin" / "claude.cmd"
            shim.parent.mkdir(parents=True)
            shim.write_text("@echo off\n", encoding="utf-8")
            shim.chmod(0o755)
            empty_home = root / "home"
            empty_home.mkdir()
            environment = {
                "PATH": str(shim.parent),
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                "HOME": str(empty_home),
                "USERPROFILE": str(empty_home),
            }
            # The module's own platform question is simulated: on a real
            # Windows host os.name already says "nt", and patching it on a
            # POSIX host breaks pathlib.
            with patch.dict(os.environ, environment, clear=True), patch.object(
                server, "_is_windows", return_value=True
            ):
                found = server._claude_executable()
            # Windows names the file with the PATHEXT entry's case (claude.CMD);
            # its file system ignores case, so both name the same shim.
            self.assertEqual(os.path.normcase(str(shim)), os.path.normcase(found or ""))

    def test_simulated_windows_falls_back_to_the_installer_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            executable = home / ".local" / "bin" / "claude.exe"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            environment = {
                "PATH": str(home / "an-empty-folder"),
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                "HOME": str(home),
                "USERPROFILE": str(home),
            }
            # ``os.name`` stays as it is here: pathlib dispatches ``Path`` on
            # it and refuses to build a ``WindowsPath`` on this host, so the
            # simulation goes through the module's own platform question.
            with patch.dict(os.environ, environment, clear=True), patch.object(
                server, "_is_windows", return_value=True
            ):
                self.assertEqual(str(executable), server._claude_executable())

    @unittest.skipIf(sys.platform == "win32", "the macOS installer layout; Windows has its own cases above")
    def test_a_gui_path_still_finds_the_macos_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            cli = home / ".local" / "bin" / "claude"
            cli.parent.mkdir(parents=True)
            cli.write_text("#!/bin/sh\n", encoding="utf-8")
            cli.chmod(0o755)
            # The PATH a launchd-started process carries on this machine.
            environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home)}
            with patch.dict(os.environ, environment, clear=True):
                self.assertEqual(str(cli), server._claude_executable())

    def test_the_login_check_runs_the_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            name = "claude.exe" if sys.platform == "win32" else "claude"
            cli = home / ".claude" / "local" / name
            cli.parent.mkdir(parents=True)
            cli.write_text("#!/bin/sh\n", encoding="utf-8")
            cli.chmod(0o755)
            environment = {
                "PATH": str(home / "an-empty-folder"),
                "HOME": str(home),
                "USERPROFILE": str(home),
            }
            with patch.dict(os.environ, environment, clear=True), patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout='{"loggedIn":true}'),
            ) as run:
                self.assertEqual(server.CLAUDE_LOGIN_AVAILABLE, server._claude_login_state())

            self.assertEqual(str(cli), run.call_args.args[0][0])

    def test_no_claude_anywhere_reads_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {"PATH": str(Path(temporary) / "empty"), "HOME": temporary}
            with patch.dict(os.environ, environment, clear=True), patch.object(
                server, "_claude_install_locations", return_value=()
            ):
                self.assertIsNone(server._claude_executable())
                self.assertEqual(server.CLAUDE_LOGIN_ABSENT, server._claude_login_state())


class CodexLoginCredentialTests(unittest.TestCase):
    """Round-3 finding 6: a junk auth.json advertised six models that could only fail."""

    def _codex_home(self, contents: str | None) -> str:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        if contents is not None:
            (Path(temporary.name) / "auth.json").write_text(contents, encoding="utf-8")
        return temporary.name

    def test_a_file_that_is_not_json_is_not_a_login(self) -> None:
        with patch.dict(os.environ, {"CODEX_HOME": self._codex_home("not json at all")}):
            self.assertFalse(server._codex_login_available())
            self.assertIn("does not parse as JSON", server._codex_login_problem() or "")

    def test_json_without_a_credential_field_is_not_a_login(self) -> None:
        with patch.dict(
            os.environ,
            {"CODEX_HOME": self._codex_home('{"note": "synthetic placeholder"}')},
        ):
            self.assertFalse(server._codex_login_available())
            self.assertIn("holds no credential", server._codex_login_problem() or "")

    def test_an_empty_api_key_is_not_a_login(self) -> None:
        with patch.dict(
            os.environ,
            {"CODEX_HOME": self._codex_home('{"OPENAI_API_KEY": "", "tokens": {}}')},
        ):
            self.assertFalse(server._codex_login_available())

    def test_a_missing_file_says_to_sign_in(self) -> None:
        with patch.dict(os.environ, {"CODEX_HOME": self._codex_home(None)}):
            self.assertIn("codex login", server._codex_login_problem() or "")

    def test_an_api_key_login_counts(self) -> None:
        with patch.dict(
            os.environ,
            {"CODEX_HOME": self._codex_home('{"OPENAI_API_KEY": "placeholder-not-real"}')},
        ):
            self.assertIsNone(server._codex_login_problem())
            self.assertTrue(server._codex_login_available())

    def test_a_chatgpt_token_login_counts(self) -> None:
        home = self._codex_home(
            '{"auth_mode": "chatgpt", "OPENAI_API_KEY": null, '
            '"tokens": {"access_token": "placeholder-not-real"}}'
        )
        with patch.dict(os.environ, {"CODEX_HOME": home}):
            self.assertIsNone(server._codex_login_problem())

    def test_a_junk_file_removes_the_codex_models_from_the_roster(self) -> None:
        home = self._codex_home('{"note": "synthetic placeholder"}')
        with patch.dict(os.environ, {"CODEX_HOME": home}), patch.object(
            server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_ABSENT
        ), patch.object(
            server, "_claude_sdk_installed", return_value=True
        ), patch(
            "vnext.vnext_mcp_server.load_zai_provider", return_value=object()
        ), patch(
            "vnext.vnext_mcp_server.load_commandcode_provider", return_value=None
        ):
            entries = server._validate_catalog(server.DEFAULT_CATALOG)

        self.assertEqual({"zai"}, {entry.get("provider") for entry in entries})


class StartupCheckTests(unittest.TestCase):
    """Round-3 finding 4: the startup error had to be printed, with the roster."""

    def test_configured_providers_absent_from_catalog_say_catalog_names_none(self) -> None:
        probe = MagicMock()
        probe.claude_state.return_value = server.CLAUDE_LOGIN_AVAILABLE
        with patch.object(server, "_codex_login_problem", return_value=None), patch.object(
            server, "_claude_sdk_installed", return_value=True
        ), patch.object(server, "load_zai_provider", return_value=object()), patch.object(
            server, "load_commandcode_provider", return_value=object()
        ):
            notes = server.startup_check_notes(
                [{"provider": "codex", "model": "gpt-6-sol"}], login=probe
            )

        self.assertEqual(
            [
                "claude models left out: the catalog names none",
                "zai models left out: the catalog names none",
                "commandcode models left out: the catalog names none",
            ],
            notes,
        )

    def test_the_check_prints_one_model_a_line_with_its_provider(self) -> None:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server,
            "_load_catalog",
            return_value=[
                {"model": "glm-5.3", "provider": "zai"},
                {"model": "opus", "provider": "claude"},
            ],
        ), patch.object(server, "startup_check_notes", return_value=[]):
            code = server.run_startup_check(["--workspace", workspace], out=out)

        self.assertEqual(0, code)
        printed = out.getvalue().splitlines()
        self.assertIn(
            "glm-5.3 (zai) -> exact model unknown until the first reply: "
            "this provider publishes no model table",
            printed,
        )
        self.assertIn(
            "opus (claude) -> exact model unknown: "
            "the model probe is switched off (VNEXT_CHECK_SKIP_MODEL_PROBE=1)",
            printed,
        )

    def test_the_check_prints_the_exact_model_beside_each_alias(self) -> None:
        out = io.StringIO()
        rows = [
            {"value": "default", "resolvedModel": "claude-opus-5-5"},
            {"value": "opus", "resolvedModel": "claude-opus-5-5"},
            {"value": "claude-fable-5-1", "resolvedModel": "claude-fable-5-1"},
            {"value": "claude-fable-5", "resolvedModel": "claude-fable-5"},
            {"value": "claude-haiku-4-5", "resolvedModel": "claude-haiku-4-5"},
            {"value": "claude-haiku-4-6", "resolvedModel": "claude-haiku-4-6"},
        ]
        with tempfile.TemporaryDirectory() as workspace, patch.dict(
            os.environ, {"VNEXT_CHECK_SKIP_MODEL_PROBE": "0"}
        ), patch.object(
            server,
            "_load_catalog",
            return_value=[
                {"model": "opus", "provider": "claude"},
                {"model": "fable", "provider": "claude"},
                {"model": "sonnet", "provider": "claude"},
                {"model": "haiku", "provider": "claude"},
                {"model": "gpt-6.1-sol", "provider": "codex"},
                {"model": "gpt-9", "provider": "codex"},
            ],
        ), patch.object(server, "startup_check_notes", return_value=[]), patch.object(
            server,
            "_probe_claude_models",
            side_effect=lambda _ws, model=None: (
                rows + ([{"value": "fable", "resolvedModel": "claude-fable-5-1"}] if model == "fable" else []),
                None,
            ),
        ), patch.object(
            server, "_probe_codex_models", return_value=({"gpt-6.1-sol": "gpt-6.1-sol"}, None)
        ):
            code = server.run_startup_check(["--workspace", workspace], out=out)

        self.assertEqual(0, code)
        printed = out.getvalue().splitlines()
        self.assertIn("opus (claude) -> claude-opus-5-5", printed)
        self.assertIn("fable (claude) -> claude-fable-5-1", printed)
        self.assertIn(
            "sonnet (claude) -> exact model unknown: the CLI lists no row for it", printed
        )
        self.assertIn(
            "haiku (claude) -> exact model unknown: the CLI lists claude-haiku-4-5, "
            "claude-haiku-4-6; the worker's first reply names the one that runs",
            printed,
        )
        self.assertIn("gpt-6.1-sol (codex) -> gpt-6.1-sol", printed)
        self.assertIn("gpt-9 (codex) -> exact model unknown: model/list does not list it", printed)

    def test_a_startup_error_is_printed_and_the_exit_code_is_one(self) -> None:
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server,
            "_load_catalog",
            side_effect=VNextMcpServiceError("a workforce needs at least one available worker model"),
        ):
            code = server.run_startup_check(["--workspace", workspace], err=errors)

        self.assertEqual(1, code)
        self.assertIn("at least one available worker model", errors.getvalue())

    @unittest.skipIf(os.name == "nt", "simulates SIGBREAK with a POSIX signal")
    def test_a_ctrl_break_unwinds_through_the_close_like_sigterm(self) -> None:
        # The Windows launcher asks the proxy group to stop with CTRL_BREAK.
        # Python's default for SIGBREAK exits without unwinding, so the close
        # that writes the run record would not run.
        with patch.object(server.signal, "SIGBREAK", signal.SIGUSR1, create=True):
            previous = server._install_stop_handlers()
            try:
                with self.assertRaises(SystemExit) as stopped:
                    os.kill(os.getpid(), signal.SIGUSR1)
                    time.sleep(1)
            finally:
                server._restore_stop_handlers(previous)
        self.assertEqual(0, stopped.exception.code)

    def test_the_check_refuses_the_host_and_token_the_server_refuses(self) -> None:
        # The check answers for the command a person runs; a yes to a command
        # the server then refuses sends them back to it unchanged.
        for extra, said in (
            (["--host", "0.0.0.0"], "--host must be a loopback host"),
            (["--token", "short"], "--token must be at least 16 characters"),
        ):
            with self.subTest(extra=extra):
                errors = io.StringIO()
                with tempfile.TemporaryDirectory() as workspace, patch.object(
                    server, "_load_catalog", return_value=[{"model": "opus", "provider": "claude"}]
                ), patch.object(server, "startup_check_notes", return_value=[]):
                    code = server.run_startup_check(
                        ["--workspace", workspace, *extra], out=io.StringIO(), err=errors
                    )
                # R21: the server exits 2 on these, and so does the check.
                self.assertEqual(2, code)
                self.assertIn(said, errors.getvalue())

    def test_the_check_refuses_the_arguments_the_server_refuses(self) -> None:
        """R20 gpt-code F4: the check dropped what it did not know.

        ``--port not-a-number`` and a misspelt ``--catlog`` passed the check
        with exit 0, and the server then exited 2 on the same command line.
        """

        for extra, said in (
            (["--port", "not-a-number"], "invalid int value"),
            (["--catlog", "missing.json"], "unrecognized arguments"),
        ):
            with self.subTest(extra=extra):
                errors = io.StringIO()
                with tempfile.TemporaryDirectory() as workspace, patch.object(
                    server, "_load_catalog", return_value=[{"model": "opus", "provider": "claude"}]
                ), patch.object(server, "startup_check_notes", return_value=[]):
                    code = server.run_startup_check(
                        ["--workspace", workspace, *extra], out=io.StringIO(), err=errors
                    )
                self.assertEqual(2, code)
                self.assertIn(said, errors.getvalue())
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server, "_load_catalog", return_value=[{"model": "opus", "provider": "claude"}]
        ), patch.object(server, "startup_check_notes", return_value=[]):
            code = server.run_startup_check(
                ["--workspace", workspace, "--port", "9000", "--name", "x", "--stdio"],
                out=io.StringIO(), err=io.StringIO(),
            )
        self.assertEqual(0, code)

    def test_a_missing_workspace_is_a_startup_error(self) -> None:
        errors = io.StringIO()
        code = server.run_startup_check(
            ["--workspace", "/no/such/workspace/anywhere"], err=errors
        )

        self.assertEqual(1, code)
        self.assertIn("workspace does not exist", errors.getvalue())

    def test_a_price_that_is_not_finite_is_no_price(self) -> None:
        """R23: a NaN cost was written as NaN and the report printed $nan."""

        for reported in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(reported=reported):
                self.assertIsNone(server._reported_cost({"cost_usd": reported}))
        self.assertEqual(0.25, server._reported_cost({"cost_usd": 0.25}))
        # R25: an integer too large for a float raised OverflowError while recording.
        self.assertIsNone(server._reported_cost({"cost_usd": 10**1000}))

    def test_a_rate_table_price_that_overflows_is_no_price(self) -> None:
        """R24: token counts near 10**307 priced as Infinity, and 10**400 raised."""

        for count in (10**307, 10**400):
            with self.subTest(digits=len(str(count))):
                tokens = {"inputTokens": count, "cachedInputTokens": 0, "outputTokens": count}
                self.assertIsNone(server._api_equivalent("gpt-6-astra", tokens))
        self.assertIsNotNone(server._api_equivalent(
            "gpt-6-astra", {"inputTokens": 1000, "cachedInputTokens": 0, "outputTokens": 10}))

    def test_a_workspace_that_is_a_file_is_called_a_file(self) -> None:
        # Round-22 stranger: --workspace README.md said "does not exist" for a file that exists.
        errors = io.StringIO()
        with tempfile.NamedTemporaryFile() as handle:
            code = server.run_startup_check(["--workspace", handle.name], err=errors)

        self.assertEqual(1, code)
        self.assertIn("workspace is not a folder", errors.getvalue())
        self.assertNotIn("does not exist", errors.getvalue())

    def test_a_catalog_that_names_no_provider_model_says_so_before_any_login(self) -> None:
        # Round-15 stranger: a Z.ai-only catalog sent the user to 'codex login'
        # and 'claude' sign-in for models the catalog could never offer.
        with tempfile.TemporaryDirectory() as workspace:
            catalog = Path(workspace) / "catalog.json"
            catalog.write_text(
                json.dumps({"models": [{"provider": "zai", "model": "glm-5.3"}]}), encoding="utf-8"
            )
            out = io.StringIO()
            with patch.object(
                server, "_codex_login_problem", return_value="auth.json does not exist; run 'codex login'"
            ), patch.object(server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_ABSENT), patch.object(
                server, "_claude_sdk_installed", return_value=True
            ), patch.object(server, "load_zai_provider", return_value=object()), patch.object(
                server, "load_commandcode_provider", return_value=None
            ), patch.object(server, "describe_provider_config_problem", return_value=None):
                code = server.run_startup_check(
                    ["--workspace", workspace, "--catalog", str(catalog)], out=out
                )

        self.assertEqual(0, code)
        printed = out.getvalue()
        self.assertIn("codex models left out: the catalog names none", printed)
        self.assertIn("claude models left out: the catalog names none", printed)
        self.assertIn("commandcode models left out: the catalog names none", printed)
        self.assertNotIn("codex login", printed)

    def test_the_check_says_why_the_codex_models_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "auth.json").write_text("not json", encoding="utf-8")
            with patch.dict(os.environ, {"CODEX_HOME": temporary}), patch.object(
                server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_ABSENT
            ), patch(
                "vnext.vnext_mcp_server.load_commandcode_provider", return_value=None
            ):
                notes = server.startup_check_notes([{"provider": "zai", "model": "glm-5.3"}])

        joined = "\n".join(notes)
        self.assertIn("codex models left out", joined)
        self.assertIn("does not parse as JSON", joined)


class CatalogFileValidationTests(unittest.TestCase):
    """Round-5 finding 2: a --catalog file reached none of the roster guards.

    Every guard lives in ``_validate_catalog``: the pinned-CLI Codex model
    gate, the per-provider login and key filters, and the non-empty check. The
    file path returned its contents whole, so a catalog file could offer a
    model that can only fail once a manager delegates to it -- the exact
    outcome the filters exist to prevent.
    """

    def _catalog(self, payload: object) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "catalog.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _logged_in(self):
        """Both built-in logins present, so only the gate under test fires."""

        return (
            patch.object(server, "_codex_login_available", return_value=True),
            patch.object(
                server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_AVAILABLE
            ),
        )

    def test_a_file_cannot_offer_a_model_the_pinned_codex_cli_cannot_run(self) -> None:
        path = self._catalog({"models": [{"provider": "codex", "model": "gpt-7-nonexistent"}]})
        codex, claude = self._logged_in()
        with codex, claude, self.assertRaises(VNextMcpServiceError) as caught:
            server._load_catalog(path)

        self.assertIn("gpt-7-nonexistent", str(caught.exception))
        self.assertIn("pinned Codex CLI", str(caught.exception))

    def test_a_file_cannot_re_admit_a_provider_with_no_configured_key(self) -> None:
        path = self._catalog({"models": [
            {"provider": "codex", "model": "gpt-6-sol"},
            {"provider": "zai", "model": "glm-5.3"},
            {"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"},
        ]})
        codex, claude = self._logged_in()
        with codex, claude, patch(
            "vnext.vnext_mcp_server.load_zai_provider", return_value=None
        ), patch(
            "vnext.vnext_mcp_server.load_commandcode_provider", return_value=None
        ):
            entries = server._load_catalog(path)

        self.assertEqual([{"provider": "codex", "model": "gpt-6-sol"}], list(entries))

    def test_a_file_whose_every_card_is_filtered_out_is_refused(self) -> None:
        path = self._catalog({"models": [{"provider": "zai", "model": "glm-5.3"}]})
        codex, claude = self._logged_in()
        with codex, claude, patch(
            "vnext.vnext_mcp_server.load_zai_provider", return_value=None
        ), self.assertRaises(VNextMcpServiceError) as caught:
            server._load_catalog(path)

        self.assertIn("at least one available worker model", str(caught.exception))

    def test_invalid_json_names_catalog_path(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "broken.json"
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(Exception) as caught:
            server._load_catalog(str(path))
        self.assertIn(str(path), str(caught.exception))

    def test_an_empty_models_list_is_still_refused_by_its_own_message(self) -> None:
        for payload in ({"models": []}, [], {"models": "glm-5.3"}):
            with self.subTest(payload=payload):
                path = self._catalog(payload)
                with self.assertRaises(VNextMcpServiceError) as caught:
                    server._load_catalog(path)

                self.assertIn("non-empty models list", str(caught.exception))

    def test_a_registered_provider_card_keeps_its_own_credential_check(self) -> None:
        """A --provider factory answers for its own credentials, as before."""

        path = self._catalog({"models": [{"provider": "acme", "model": "acme-one"}]})
        codex, claude = self._logged_in()
        with codex, claude:
            entries = server._load_catalog(path, adapter_factories={"acme": ScriptedWorker})

        self.assertEqual([{"provider": "acme", "model": "acme-one"}], list(entries))

    def test_the_check_exits_nonzero_for_a_file_the_server_would_refuse(self) -> None:
        """The GPT reviewer's probe: --check reported the model as offered."""

        path = self._catalog({"models": [{"provider": "codex", "model": "gpt-7-nonexistent"}]})
        errors = io.StringIO()
        out = io.StringIO()
        codex, claude = self._logged_in()
        with tempfile.TemporaryDirectory() as workspace, codex, claude:
            code = server.run_startup_check(
                ["--workspace", workspace, "--catalog", path], out=out, err=errors
            )

        self.assertEqual(1, code)
        self.assertIn("pinned Codex CLI", errors.getvalue())
        self.assertNotIn("gpt-7-nonexistent (codex)", out.getvalue())

    def test_malformed_catalog_cards_fail_check_and_start_with_the_same_one_line_error(self) -> None:
        cases = (
            ("missing model", {"provider": "codex"}, "requires a non-empty string model"),
            ("blank model", {"provider": "codex", "model": "  "}, "requires a non-empty string model"),
            ("number model", {"provider": "codex", "model": 5}, "requires a non-empty string model"),
            ("missing provider", {"model": "gpt-6-sol"}, "names provider"),
            ("number provider", {"provider": 5, "model": "gpt-6-sol"}, "names provider"),
            ("list provider", {"provider": [], "model": "gpt-6-sol"}, "names provider"),
            ("string entry", "opus", "each entry must be an object with provider and model"),
            ("number entry", 123, "each entry must be an object with provider and model"),
            ("null entry", None, "each entry must be an object with provider and model"),
            ("list entry", [], "each entry must be an object with provider and model"),
            ("bad claims", {"provider": "codex", "model": "gpt-6-sol", "claims": 5}, "claims must be a list"),
            ("bad harness", {"provider": "codex", "model": "gpt-6-sol", "harness": 5}, "unsupported harness"),
        )
        with tempfile.TemporaryDirectory() as workspace:
            for name, card, reason in cases:
                with self.subTest(name=name):
                    path = self._catalog({"models": [card]})
                    outputs = []
                    for module, flags in (
                        ("vnext.vnext_mcp_reload", ["--check"]),
                        ("vnext.vnext_mcp_server", ["--stdio"]),
                    ):
                        result = subprocess.run(
                            [sys.executable, "-m", module, *flags, "--workspace", workspace,
                             "--catalog", path],
                            input="", text=True, capture_output=True, timeout=20,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                                 "PYTHONDONTWRITEBYTECODE": "1"},
                        )
                        self.assertNotEqual(0, result.returncode, (name, module, result))
                        self.assertEqual("", result.stdout, (name, module, result))
                        self.assertNotIn("Traceback", result.stderr, (name, module, result))
                        self.assertEqual(1, len(result.stderr.splitlines()), (name, module, result))
                        self.assertIn(path, result.stderr)
                        self.assertIn("entry 1", result.stderr)
                        self.assertIn(reason, result.stderr)
                        outputs.append(result.stderr)
                    self.assertEqual(outputs[0], outputs[1], name)

    def test_a_card_that_names_its_model_with_model_id_is_accepted(self) -> None:
        # The runtime reads model_id as another name for model; the check
        # must not refuse a card the start would take.
        path = self._catalog({"models": [{"provider": "codex", "model_id": "gpt-6-sol"}]})
        codex, claude = self._logged_in()
        with codex, claude:
            entries = server._load_catalog(path)
        self.assertEqual("gpt-6-sol", entries[0]["model_id"])

    def test_the_pinned_cli_gate_reads_a_model_id_card_too(self) -> None:
        path = self._catalog({"models": [{"provider": "codex", "model_id": "gpt-7-nonexistent"}]})
        codex, claude = self._logged_in()
        with codex, claude, self.assertRaises(VNextMcpServiceError) as caught:
            server._load_catalog(path)
        self.assertIn("gpt-7-nonexistent", str(caught.exception))


class RegisteredProviderDelegationTests(unittest.TestCase):
    """Round-5 finding 3: a --provider worker's first delegate blocked.

    The reviewer drove this over real stdio and got
    ``{"status":"blocked","blocker":"worker-model runtime could not be started:
    no native effect reader is registered for provider 'scripted'"}``.  A
    documented extension point accepted a provider that could never run a turn.
    """

    def _service(self, workspace: str, factories: dict):
        service = VNextMcpService(
            workspace=workspace,
            catalog=[{"provider": "acme", "model": WORKER_MODEL}],
            event_log=None,
            status_file=None,
            outcome_log=None,
            adapter_factories=factories,
        )
        self.addCleanup(service.close)
        return service

    def test_a_registered_provider_s_worker_runs_its_turn(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            service = self._service(workspace, {"acme": ScriptedWorker})
            created = json.loads(
                service.session.external_tool_call(
                    tool="delegate",
                    arguments={
                        "role": AgentRole.WORKER.value,
                        "model_id": WORKER_MODEL,
                        "objective": "Run one turn under a registered provider",
                        "task_contract": {"criteria": ["turn ran"]},
                    },
                ).as_json_text()
            )
            waited = service.session.external_tool_call(
                tool="await_children",
                arguments={"agent_ids": [created["agent_id"]]},
                timeout=30,
            )
            self.assertTrue(waited.success, waited.as_json_text())
            answer = json.loads(waited.as_json_text())

        self.assertNotIn("no native effect reader", json.dumps(answer))
        blockers = [
            str(child.get("blocker") or "")
            for child in answer.get("children", [])
        ]
        self.assertEqual([""] * len(blockers), blockers, answer)

    def test_a_provider_with_no_reader_is_refused_when_it_is_registered(self) -> None:
        """The refusal is at --provider parse time, where the error is cheap."""

        with self.assertRaises(VNextMcpServiceError) as caught:
            server._load_provider_factories(
                ["acme=tests.test_vnext_mcp_server:_NoReaderAdapter"]
            )

        message = str(caught.exception)
        self.assertIn("'acme'", message)
        self.assertIn("runtime_effect_reader", message)
        self.assertIn("codex", message)

    def test_a_provider_that_names_a_built_in_harness_is_accepted(self) -> None:
        factories = server._load_provider_factories(
            ["acme=tests.test_vnext_external_primary:ScriptedWorker"]
        )

        self.assertIs(ScriptedWorker, factories["acme"])


class _NoReaderAdapter:
    """An adapter that answers for its effects in neither of the two ways."""


class StartupCheckProviderAgreementTests(unittest.TestCase):
    """Round-5 verification finding 2: --check read no --provider.

    ``--check --catalog f.json`` exited 0 for a card naming a provider that
    arrives through --provider, while a real start refused two of those cases:
    with no --provider at all the runtime knows no such provider, and with a
    --provider naming a class that answers for no effect shape the registration
    itself is refused.  Both paths now resolve through ``resolve_startup``, and
    the last test here walks the verifier's six cases.
    """

    SCRIPTED = "scripted=tests.test_vnext_external_primary:ScriptedWorker"
    NO_READER = "scripted=tests.test_vnext_mcp_server:_NoReaderAdapter"
    CUSTOM_CARD = [{"provider": "scripted", "model": WORKER_MODEL}]

    def _logins(self, codex, claude, zai, commandcode):
        """Deterministic answers for every credential the roster asks about."""

        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(server, "_codex_login_available", return_value=codex))
        stack.enter_context(patch.object(
            server,
            "_claude_login_state",
            return_value=server.CLAUDE_LOGIN_AVAILABLE if claude else server.CLAUDE_LOGIN_ABSENT,
        ))
        stack.enter_context(patch.object(
            server, "_codex_login_problem", return_value="no login in this test"
        ))
        stack.enter_context(patch.object(
            server, "load_zai_provider", return_value=object() if zai else None
        ))
        stack.enter_context(patch.object(
            server, "load_commandcode_provider", return_value=object() if commandcode else None
        ))
        return stack

    def _would_start(self, workspace, catalog, registrations):
        """The normal startup path, composed as it was before the check shared it."""

        try:
            factories = server._load_provider_factories(list(registrations))
            entries = server._load_catalog(catalog, adapter_factories=factories)
            service = VNextMcpService(
                workspace=workspace,
                catalog=entries,
                adapter_factories=factories,
                event_log=None,
                status_file=None,
                outcome_log=None,
            )
        except Exception as exc:  # the question is only whether it starts
            return False, f"{type(exc).__name__}: {exc}"
        service.close()
        return True, "started"

    def _case(self, models, registrations=(), **logins):
        with tempfile.TemporaryDirectory() as workspace:
            catalog = Path(workspace) / "catalog.json"
            catalog.write_text(json.dumps({"models": models}), encoding="utf-8")
            arguments = ["--workspace", workspace, "--catalog", str(catalog)]
            for registration in registrations:
                arguments += ["--provider", registration]
            out, errors = io.StringIO(), io.StringIO()
            with self._logins(
                logins.get("codex", False),
                logins.get("claude", False),
                logins.get("zai", False),
                logins.get("commandcode", False),
            ):
                code = server.run_startup_check(arguments, out=out, err=errors)
                started, reason = self._would_start(workspace, str(catalog), registrations)
        return SimpleNamespace(
            code=code, out=out.getvalue(), err=errors.getvalue(), started=started, reason=reason
        )

    def test_a_card_whose_provider_is_registered_nowhere_is_refused(self) -> None:
        """Case (a): no --provider, so nothing backs the card."""

        answer = self._case(self.CUSTOM_CARD, codex=True, claude=True)

        self.assertFalse(answer.started, answer.reason)
        self.assertEqual(1, answer.code, answer.out)
        self.assertIn("scripted", answer.err)
        self.assertIn("--provider", answer.err)
        self.assertNotIn(f"{WORKER_MODEL} (scripted)", answer.out)

    def test_a_provider_that_answers_for_no_effect_shape_is_refused(self) -> None:
        """Case (b): a --provider the registration itself rejects."""

        answer = self._case(
            self.CUSTOM_CARD, [self.NO_READER], codex=True, claude=True
        )

        self.assertFalse(answer.started, answer.reason)
        self.assertEqual(1, answer.code, answer.out)
        self.assertIn("runtime_effect_reader", answer.err)
        self.assertNotIn(f"{WORKER_MODEL} (scripted)", answer.out)

    def test_a_valid_registration_is_offered_by_the_check(self) -> None:
        answer = self._case(self.CUSTOM_CARD, [self.SCRIPTED], codex=True, claude=True)

        self.assertTrue(answer.started, answer.reason)
        self.assertEqual(0, answer.code, answer.err)
        self.assertIn(f"{WORKER_MODEL} (scripted)", answer.out)

    def test_the_check_and_a_real_start_agree_in_every_case(self) -> None:
        """The verifier's six cases, check code against what a start does."""

        cases = [
            ("bad-codex-model", [{"provider": "codex", "model": "gpt-7-nonexistent"}], (),
             {"codex": True}),
            ("no-logins", [{"provider": "codex", "model": "gpt-6-sol"},
                           {"provider": "claude", "model": "sonnet"}], (), {}),
            ("key-only-without-key",
             [{"provider": "zai", "model": "glm-5.3"},
              {"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"}], (), {}),
            ("valid-registered-card", self.CUSTOM_CARD, (self.SCRIPTED,),
             {"codex": True, "claude": True}),
            ("registered-card-missing-factory", self.CUSTOM_CARD, (),
             {"codex": True, "claude": True}),
            ("registered-card-invalid-factory", self.CUSTOM_CARD, (self.NO_READER,),
             {"codex": True, "claude": True}),
        ]
        for label, models, registrations, logins in cases:
            with self.subTest(case=label):
                answer = self._case(models, registrations, **logins)
                self.assertEqual(
                    answer.started,
                    answer.code == 0,
                    f"{label}: check {answer.code}, start {answer.reason}",
                )


class ClaudeLoginTimeoutTests(unittest.TestCase):
    """Round-1 finding 6: a 5 s timeout was reported as "not signed in"."""

    def test_the_timeout_is_long_enough_for_a_loaded_machine(self) -> None:
        """Round-3: 5 s emptied a Claude-only roster on a loaded machine."""

        self.assertGreaterEqual(server._CLAUDE_LOGIN_TIMEOUT_SECONDS, 15.0)
        with patch.object(
            server, "_claude_executable", return_value="/somewhere/claude"
        ), patch(
            "subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout='{"loggedIn":true}'),
        ) as run:
            server._claude_login_state()

        self.assertEqual(
            server._CLAUDE_LOGIN_TIMEOUT_SECONDS, run.call_args.kwargs["timeout"]
        )

    def test_a_timeout_is_unchecked_rather_than_signed_out(self) -> None:
        with patch.object(
            server, "_claude_executable", return_value="/somewhere/claude"
        ), patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
        ):
            self.assertEqual(server.CLAUDE_LOGIN_UNCHECKED, server._claude_login_state())
            self.assertFalse(server._claude_login_available())

    def test_the_empty_roster_message_says_the_check_did_not_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"CODEX_HOME": temporary, "HOME": temporary, "USERPROFILE": temporary},
                clear=True,
            ), patch(
                "vnext.vnext_mcp_server.load_zai_provider", return_value=None
            ), patch(
                "vnext.vnext_mcp_server.load_commandcode_provider",
                return_value=None,
            ), patch.object(
                server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_UNCHECKED
            ):
                with self.assertRaises(VNextMcpServiceError) as raised:
                    server._validate_catalog(server.DEFAULT_CATALOG)

        message = str(raised.exception)
        self.assertIn("could not be checked", message)
        self.assertIn("did not answer", message)

    def test_one_startup_asks_the_claude_cli_once(self) -> None:
        answers = []

        def answer(command, **keywords):
            answers.append(command)
            return SimpleNamespace(returncode=0, stdout='{"loggedIn":true}')

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            probe = server._LoginProbe()
            with patch.object(
                server, "_claude_executable", return_value="/somewhere/claude"
            ), patch.object(
                server, "_claude_sdk_installed", return_value=True
            ), patch("subprocess.run", side_effect=answer), patch.dict(
                os.environ,
                {
                    "CODEX_HOME": str(workspace / "no-codex-login"),
                    "HOME": str(workspace),
                    "USERPROFILE": str(workspace),
                },
                clear=True,
            ), patch(
                "vnext.vnext_mcp_server.load_zai_provider", return_value=None
            ), patch(
                "vnext.vnext_mcp_server.load_commandcode_provider",
                return_value=None,
            ):
                # The two call sites main() has: the catalog load, then the service.
                catalog = server._load_catalog(None, login=probe)
                self.assertEqual({"claude"}, {entry["provider"] for entry in catalog})
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=catalog,
                    client=CLIENT,
                    login_probe=probe,
                )
                service.close()

        self.assertEqual(1, len(answers))


_POSIX_FOLDER_MODE = (
    "a folder's write bit is a POSIX mode; on Windows chmod only sets the "
    "read-only attribute, which folders ignore"
)


class ReadOnlyRunFolderTests(unittest.TestCase):
    """Round-1 finding 5: a read-only .vnext killed server construction."""

    @unittest.skipIf(sys.platform == "win32", _POSIX_FOLDER_MODE)
    def test_a_read_only_run_folder_still_starts_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            home = workspace / ".vnext"
            home.mkdir()
            home.chmod(0o500)
            service = None
            try:
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                    client=CLIENT,
                    adapter_factories={"codex": ScriptedWorker},
                )
                self.assertFalse((home / ".gitignore").exists())
                # R21: the path is kept, so the roster appears once it frees.
                self.assertIsNotNone(service.status_file)
                self.assertFalse(service.status_file.exists())
            finally:
                home.chmod(0o700)
                if service is not None:
                    service.close()

    @unittest.skipIf(sys.platform == "win32", _POSIX_FOLDER_MODE)
    def test_an_unwritable_run_folder_still_starts_the_session(self) -> None:
        """Round-4 finding 1: an existing read-only runs/ aborted the service."""

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            runs = workspace / ".vnext" / "runs"
            runs.mkdir(parents=True)
            runs.chmod(0o555)
            service = None
            noise = io.StringIO()
            try:
                with contextlib.redirect_stderr(noise):
                    service = VNextMcpService(
                        workspace=workspace,
                        session_id="probe",
                        catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                        client=CLIENT,
                        adapter_factories={"codex": ScriptedWorker},
                    )
                self.assertIsNotNone(service._event_log)
                self.assertIn("could not be written", noise.getvalue())
                self.assertIn("the next change tries again", noise.getvalue())
                self.assertIn("probe.jsonl", noise.getvalue())
            finally:
                runs.chmod(0o755)
                if service is not None:
                    service.close()

    def test_a_directory_standing_where_the_log_belongs_still_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            occupied = workspace / "events.jsonl"
            occupied.mkdir()
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise):
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                    client=CLIENT,
                    adapter_factories={"codex": ScriptedWorker},
                    event_log=occupied,
                )
            try:
                self.assertEqual(occupied.resolve(), service._event_log)
                self.assertIn("events.jsonl", noise.getvalue())
                self.assertTrue(occupied.is_dir())
            finally:
                service.close()

    def test_a_record_path_refused_at_startup_is_written_once_it_frees(self) -> None:
        """R21: a refusal at startup dropped the record for the whole session."""

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            occupied = workspace / "events.jsonl"
            occupied.mkdir()
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise):
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                    client=CLIENT,
                    adapter_factories={"codex": ScriptedWorker},
                    event_log=occupied,
                )
            try:
                self.assertEqual(occupied.resolve(), service._event_log)
                self.assertIn("the next change tries again", noise.getvalue())
                occupied.rmdir()
                event = SimpleNamespace(
                    type="turn.started", agent_id="worker", turn_id="turn", payload={}
                )
                with contextlib.redirect_stderr(noise):
                    service._record(event)
                self.assertIn("written again", noise.getvalue())
                self.assertEqual(
                    "turn.started",
                    json.loads(occupied.read_text(encoding="utf-8").splitlines()[-1])["type"],
                )
            finally:
                service.close()

    def test_a_lone_surrogate_in_provider_text_is_written_and_read_back(self) -> None:
        """R21: ``"\\ud800"`` raised UnicodeEncodeError out of the event writer."""

        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            service = object.__new__(VNextMcpService)
            service._session_id = "probe"
            service.workspace = home
            service._client = CLIENT
            service._log_lock = threading.Lock()
            service._roster_lock = threading.Lock()
            service._failing_writes = set()
            service._closing = False
            service._stopping = False
            service._parent_session = None
            service._event_log = home / "events.jsonl"
            service._status_file = home / "status.json"
            service._outcome_log = home / "outcomes.jsonl"
            service._roster = {"worker": {"agent_id": "worker", "status": "running",
                                          "objective": "a\ud800b"}}
            service._outcome_rows = {"worker": {"agent_id": "worker", "outcome": "a\ud800b"}}
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise):
                service._record(SimpleNamespace(
                    type="turn.started", agent_id="worker", turn_id="turn",
                    payload={"text": "a\ud800b"},
                ))
                service._rewrite_outcomes_locked()
                service._publish_status()
            self.assertEqual("", noise.getvalue())
            line = (home / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual("a\ud800b", json.loads(line)["payload"]["text"])
            row = (home / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual("a\ud800b", json.loads(row)["outcome"])
            status = json.loads((home / "status.json").read_text(encoding="utf-8"))
            self.assertEqual("a\ud800b", status["agents"][0]["objective"])

    def test_a_log_that_breaks_mid_session_says_so_once_and_resumes(self) -> None:
        """A revoked permission or a full disk reaches _record, not _resolve.

        R20: the first failure dropped the log for the rest of the session, so
        a full disk freed a second later still cost every later record.
        """

        with tempfile.TemporaryDirectory() as temporary:
            service = object.__new__(VNextMcpService)
            service._session_id = "probe"
            service._log_lock = threading.Lock()
            service._failing_writes = set()
            service._event_log = Path(temporary) / "live.jsonl"
            service._event_log.mkdir()
            event = SimpleNamespace(
                type="turn.started", agent_id="worker", turn_id="turn", payload={}
            )
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise):
                service._record(event)
                service._record(event)
                # One warning, naming the path, for however many events follow.
                self.assertEqual(1, noise.getvalue().count("could not be written"))
                self.assertIn("live.jsonl", noise.getvalue())
                service._event_log.rmdir()
                service._record(event)
            self.assertIn("written again", noise.getvalue())
            self.assertEqual(1, len(service._event_log.read_text(encoding="utf-8").splitlines()))

    @unittest.skipIf(sys.platform == "win32", _POSIX_FOLDER_MODE)
    def test_a_roster_and_outcome_log_that_fail_once_heal_on_the_next_write(self) -> None:
        """R20: one refused rename froze the roster with a finished worker running."""

        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "locked"
            home.mkdir()
            service = object.__new__(VNextMcpService)
            service._session_id = "probe"
            service.workspace = Path(temporary)
            service._client = CLIENT
            service._log_lock = threading.Lock()
            service._roster_lock = threading.Lock()
            service._failing_writes = set()
            service._closing = False
            service._stopping = False
            service._parent_session = None
            service._roster = {"worker": {"agent_id": "worker", "status": "running"}}
            service._outcome_rows = {"worker": {"agent_id": "worker"}}
            service._status_file = home / "status.json"
            service._outcome_log = home / "outcomes.jsonl"
            home.chmod(0o555)
            noise = io.StringIO()
            try:
                with contextlib.redirect_stderr(noise):
                    service._rewrite_outcomes_locked()
                    service._publish_status()
            finally:
                home.chmod(0o755)
            self.assertIn("outcomes.jsonl", noise.getvalue())
            self.assertIn("status.json", noise.getvalue())
            self.assertFalse((home / "status.json").exists())

            service._roster["worker"]["status"] = "completed"
            service._outcome_rows["other"] = {"agent_id": "other"}
            with contextlib.redirect_stderr(noise):
                service._rewrite_outcomes_locked()
                service._publish_status()
            status = json.loads((home / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(0, status["working"])
            self.assertEqual(
                ["other", "worker"],
                [json.loads(line)["agent_id"] for line in (home / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()],
            )
            self.assertEqual(2, noise.getvalue().count("written again"))

    def test_a_refused_status_rename_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            service = object.__new__(VNextMcpService)
            service._session_id = "probe"
            service.workspace = home
            service._client = CLIENT
            service._roster_lock = threading.Lock()
            service._failing_writes = set()
            service._closing = False
            service._stopping = False
            service._parent_session = None
            service._roster = {}
            service._status_file = home / "status.json"
            with patch("vnext.vnext_mcp_server.os.replace", side_effect=PermissionError(13, "in use")), \
                    contextlib.redirect_stderr(io.StringIO()):
                service._publish_status()
            self.assertEqual([], sorted(path.name for path in home.iterdir()))

    def test_an_existing_user_gitignore_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            home = workspace / ".vnext"
            home.mkdir()
            (home / ".gitignore").write_text("# mine\nrandom\n", encoding="utf-8")
            service = VNextMcpService(
                workspace=workspace,
                catalog=[{"provider": "codex", "model": WORKER_MODEL}],
                client=CLIENT,
                adapter_factories={"codex": ScriptedWorker},
            )
            try:
                self.assertEqual(
                    "# mine\nrandom\n", (home / ".gitignore").read_text(encoding="utf-8")
                )
            finally:
                service.close()

ClaudeExecutableResolutionTests.test_the_login_check_runs_the_resolved_path = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ClaudeExecutableResolutionTests.test_the_login_check_runs_the_resolved_path)

ClaudeLoginTimeoutTests.test_a_timeout_is_unchecked_rather_than_signed_out = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ClaudeLoginTimeoutTests.test_a_timeout_is_unchecked_rather_than_signed_out)

ClaudeLoginTimeoutTests.test_one_startup_asks_the_claude_cli_once = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ClaudeLoginTimeoutTests.test_one_startup_asks_the_claude_cli_once)

ClaudeLoginTimeoutTests.test_the_empty_roster_message_says_the_check_did_not_answer = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ClaudeLoginTimeoutTests.test_the_empty_roster_message_says_the_check_did_not_answer)

ClaudeLoginTimeoutTests.test_the_timeout_is_long_enough_for_a_loaded_machine = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ClaudeLoginTimeoutTests.test_the_timeout_is_long_enough_for_a_loaded_machine)

ModelRosterTests.test_default_catalog_filters_each_unavailable_provider = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ModelRosterTests.test_default_catalog_filters_each_unavailable_provider)

ModelRosterTests.test_delegate_and_replace_descriptions_include_worker_limits = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(ModelRosterTests.test_delegate_and_replace_descriptions_include_worker_limits)

StartupCheckTests.test_a_catalog_that_names_no_provider_model_says_so_before_any_login = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(StartupCheckTests.test_a_catalog_that_names_no_provider_model_says_so_before_any_login)

StartupCheckTests.test_configured_providers_absent_from_catalog_say_catalog_names_none = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(StartupCheckTests.test_configured_providers_absent_from_catalog_say_catalog_names_none)

VNextMcpServerTests.test_bad_claude_effort_is_a_tool_error_without_a_spawn_record = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextMcpServerTests.test_bad_claude_effort_is_a_tool_error_without_a_spawn_record)

VNextMcpServerTests.test_tools_list_serves_worker_limits_for_delegate_and_replace = __import__('unittest').skip('Private Claude worker test is excluded from the public build.')(VNextMcpServerTests.test_tools_list_serves_worker_limits_for_delegate_and_replace)
