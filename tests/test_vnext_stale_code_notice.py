"""A running server says when newer vNext code is on disk than it loaded.

On 2 October 2026 a server started at 09:10 refused ``gpt-6.1-sol`` at 11:47
because the release that added it was merged after the process started.  The
refusal named the old catalog and nothing else, so the session found
``restart_server`` only by reading the source.  These cases pin the sentence
that closes that gap, and that the check stays silent and cheap otherwise.

Every case points the check at a temporary folder; no real package file has
its modification time changed.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext import vnext_mcp_server as server
from vnext.clock_format import format_time
from vnext.vnext_mcp_server import VNextMcpService
from vnext.vnext_orchestration import AgentRole

from tests.test_vnext_external_primary import ScriptedWorker, WORKER_MODEL


CLIENT = "claude-code"


def _package(folder: Path) -> Path:
    (folder / "a.py").write_text("A = 1\n", encoding="utf-8")
    (folder / "b.py").write_text("B = 1\n", encoding="utf-8")
    old = time.time() - 3600
    for name in ("a.py", "b.py"):
        os.utime(folder / name, (old, old))
    return folder


def _make_newer(folder: Path) -> None:
    now = time.time()
    os.utime(folder / "b.py", (now, now))


class FingerprintTests(unittest.TestCase):
    def test_a_missing_package_folder_is_never_newer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            freshness = server.CodeFreshness(Path(temp) / "gone", recheck_after=0.0)
            self.assertFalse(freshness.newer_on_disk())
            self.assertIsNone(freshness.stale_notice())

    def test_an_unreadable_package_folder_is_never_newer(self) -> None:
        def unreadable(folder: Path):
            raise PermissionError("denied")

        with tempfile.TemporaryDirectory() as temp:
            folder = _package(Path(temp))
            with patch.object(server, "_code_fingerprint", unreadable):
                freshness = server.CodeFreshness(folder, recheck_after=0.0)
                self.assertFalse(freshness.newer_on_disk())

    def test_a_folder_that_becomes_unreadable_later_is_never_newer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = _package(Path(temp))
            freshness = server.CodeFreshness(folder, recheck_after=0.0)
            with patch.object(server, "_code_fingerprint", lambda folder: None):
                self.assertFalse(freshness.newer_on_disk())

    def test_a_newer_file_and_a_new_file_both_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = _package(Path(temp))
            freshness = server.CodeFreshness(folder, recheck_after=0.0)
            self.assertFalse(freshness.newer_on_disk())
            _make_newer(folder)
            self.assertTrue(freshness.newer_on_disk())
        with tempfile.TemporaryDirectory() as temp:
            folder = _package(Path(temp))
            freshness = server.CodeFreshness(folder, recheck_after=0.0)
            old = time.time() - 3600
            (folder / "c.py").write_text("C = 1\n", encoding="utf-8")
            os.utime(folder / "c.py", (old, old))
            self.assertTrue(freshness.newer_on_disk())

    def test_the_fingerprint_is_cached_between_close_calls(self) -> None:
        calls = []
        now = [100.0]

        def counted(folder: Path):
            calls.append(folder)
            return (1.0, 2)

        with tempfile.TemporaryDirectory() as temp, patch.object(
            server, "_code_fingerprint", counted
        ):
            freshness = server.CodeFreshness(
                Path(temp), recheck_after=5.0, clock=lambda: now[0]
            )
            self.assertEqual(1, len(calls), "the start records one fingerprint")
            for _ in range(10):
                freshness.newer_on_disk()
            self.assertEqual(2, len(calls), "ten close calls read the disk once")
            now[0] += 6.0
            freshness.newer_on_disk()
            self.assertEqual(3, len(calls), "a call after the interval reads it again")

    def test_the_sentence_names_the_start_time_and_the_fix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = _package(Path(temp))
            freshness = server.CodeFreshness(folder, recheck_after=0.0)
            _make_newer(folder)
            self.assertEqual(
                "newer vNext code is on disk than this server loaded at "
                f"{format_time(freshness.started_at)}; restart_server loads it, "
                "and running workers stop when it does",
                freshness.stale_notice(),
            )


class ServiceNoticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.package = root / "package"
        self.package.mkdir()
        _package(self.package)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        patcher = patch.object(server, "_PACKAGE_FOLDER", self.package)
        patcher.start()
        self.addCleanup(patcher.stop)
        interval = patch.object(server, "_CODE_RECHECK_SECONDS", 0.0)
        interval.start()
        self.addCleanup(interval.stop)
        self.service = VNextMcpService(
            workspace=self.workspace,
            catalog=[{"provider": "codex", "model": WORKER_MODEL}],
            client=CLIENT,
            adapter_factories={"codex": ScriptedWorker},
            event_log=None,
            status_file=None,
            outcome_log=None,
        )
        self.addCleanup(self._temp.cleanup)
        self.addCleanup(self.service.close)

    def _guess(self):
        return self.service._dispatch(
            "delegate",
            {
                "role": AgentRole.WORKER.value,
                "model_id": "gpt-6.1-sol",
                "objective": "Run on a model this server never loaded",
                "task_contract": {"criteria": ["one turn"]},
            },
            {},
        )

    def test_the_unknown_model_refusal_is_unchanged_when_the_code_is_current(self) -> None:
        refused = self._guess()
        self.assertFalse(refused.success)
        self.assertEqual("unknown-model", refused.value["error_code"])
        self.assertEqual(
            f"unknown model: gpt-6.1-sol; worker can run on {WORKER_MODEL} (codex)",
            refused.value["error"],
        )

    def test_the_unknown_model_refusal_names_restart_server_when_newer_code_is_on_disk(self) -> None:
        _make_newer(self.package)
        refused = self._guess()
        self.assertFalse(refused.success)
        self.assertEqual("unknown-model", refused.value["error_code"])
        started = format_time(self.service._code_freshness.started_at)
        self.assertEqual(
            f"unknown model: gpt-6.1-sol; worker can run on {WORKER_MODEL} (codex); "
            f"newer vNext code is on disk than this server loaded at {started}; "
            "restart_server loads it, and running workers stop when it does",
            refused.value["error"],
        )

    def test_inspect_self_shows_the_line_only_when_newer_code_is_on_disk(self) -> None:
        current = self.service._dispatch("inspect", {"agent_id": "self"}, {})
        self.assertTrue(current.success, current)
        self.assertFalse(
            any("restart_server" in line for line in current.value.get("runtime", [])),
            current.value.get("runtime"),
        )
        _make_newer(self.package)
        stale = self.service._dispatch("inspect", {"agent_id": "self"}, {})
        self.assertTrue(stale.success, stale)
        self.assertIn(
            self.service._code_freshness.stale_notice(), stale.value["runtime"]
        )

    def test_the_delegate_description_says_when_the_model_list_was_read(self) -> None:
        tools = server._external_tools(
            [{"name": "delegate", "description": "Delegate work."}],
            [{"provider": "codex", "model": "gpt-6-sol"}],
        )
        delegate = next(tool for tool in tools if tool["name"] == "delegate")
        self.assertIn(
            "This list was read when this session started; a refusal for an "
            "unknown model lists what the server offers now.",
            delegate["description"],
        )


if __name__ == "__main__":
    unittest.main()
