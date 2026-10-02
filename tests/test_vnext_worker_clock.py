"""The clock a vNext worker reads: its shape, its hook, and its state file."""

from __future__ import annotations

import datetime
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext import clock_hook
from vnext.clock_format import format_duration, mid_turn_line, turn_start_line
from vnext.vnext_app_server import DEFAULT_CLOCK_LIMIT, VNextAppServerAdapter
from vnext.vnext_clock_hook_config import (
    TRUST_KEY,
    hook_cli_args,
    hook_command_string,
    hook_thread_config,
    merge_config,
    trusted_hash_from_hooks_list,
)
from vnext.vnext_commandcode import CommandCodeBridge
from vnext.vnext_runtime_types import RuntimePosture

MOMENT = 1790277240.0  # 2026-09-24 21:14:00 CEST


def _hhmm(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment).astimezone().strftime("%H:%M %Z")


class ClockFormatTests(unittest.TestCase):
    def test_a_duration_changes_units_with_its_magnitude(self) -> None:
        self.assertEqual("0s", format_duration(0))
        self.assertEqual("45s", format_duration(45))
        self.assertEqual("59s", format_duration(59.9))
        self.assertEqual("1m 00s", format_duration(60))
        self.assertEqual("2m 03s", format_duration(123))
        self.assertEqual("59m 59s", format_duration(3599))
        self.assertEqual("1h 00m", format_duration(3600))
        self.assertEqual("2h 14m", format_duration(2 * 3600 + 14 * 60 + 59))
        self.assertEqual("0s", format_duration(-5))

    def test_the_turn_leads_and_the_agent_age_rides_behind_it(self) -> None:
        self.assertEqual(
            f"[clock] {_hhmm(MOMENT)} · turn 0s / 1800s · agent running 2h 14m",
            turn_start_line(MOMENT, elapsed=0, limit=1800, agent_age=2 * 3600 + 14 * 60),
        )

    def test_a_mid_turn_line_without_a_turn_clock_keeps_the_wall_clock_half(self) -> None:
        self.assertEqual(
            f"[clock] {_hhmm(MOMENT)} · turn 2m 03s / 1800s",
            mid_turn_line(MOMENT, elapsed=123, limit=1800),
        )
        self.assertEqual(f"[clock] {_hhmm(MOMENT)}", mid_turn_line(MOMENT, elapsed=None, limit=1800))


class ClockHookScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="clock-hook-test-"))
        self.state = self.directory / "turns.json"

    def _run(self, payload: dict, *, limit: str = "1800") -> str:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = clock_hook.main(
                ["clock_hook.py", str(self.state), limit], io.StringIO(json.dumps(payload))
            )
        self.assertEqual(0, code)
        return out.getvalue().strip()

    def test_monotonic_start_survives_a_wall_clock_step(self) -> None:
        import time

        self.state.write_text(json.dumps({"thread-1": {
            "started": time.time() - 3600.0,
            "started_monotonic": time.monotonic() - 1500.0,
            "limit": 1800,
        }}))
        printed = json.loads(self._run({"session_id": "thread-1", "turn_id": "turn-1"}))
        self.assertIn("turn 25m 00s / 1800s",
                      printed["hookSpecificOutput"]["additionalContext"])

    def test_it_reports_the_turn_recorded_for_this_session(self) -> None:
        import time

        started = time.time() - 123.0
        self.state.write_text(json.dumps({"thread-1": {"started": started, "limit": 900}}))

        printed = json.loads(self._run({"session_id": "thread-1", "turn_id": "turn-1"}))

        output = printed["hookSpecificOutput"]
        self.assertEqual("PostToolUse", output["hookEventName"])
        self.assertRegex(output["additionalContext"], r"^\[clock\] \d\d:\d\d .+ · turn 2m 0\ds / 900s$")

    def test_a_session_the_state_file_does_not_know_falls_back_to_the_first_call(self) -> None:
        self.state.write_text(json.dumps({"other-thread": {"started": 1.0, "limit": 1800}}))

        first = json.loads(self._run({"session_id": "stray", "turn_id": "turn-9"}))
        second = json.loads(self._run({"session_id": "stray", "turn_id": "turn-9"}))

        # The first call of the turn is the zero point, and the second call
        # measures from it rather than resetting.
        self.assertRegex(first["hookSpecificOutput"]["additionalContext"], r"· turn 0s / 1800s$")
        self.assertRegex(second["hookSpecificOutput"]["additionalContext"], r"· turn 0s / 1800s$")
        seen = json.loads((self.directory / "turns.json.turns.json").read_text())
        self.assertEqual(["turn-9"], list(seen))
        self.assertIn("started_monotonic", seen["turn-9"])

    def test_no_state_file_means_no_line_and_no_failure(self) -> None:
        self.assertEqual("", self._run({"session_id": "thread-1", "turn_id": "turn-1"}))

    def test_unreadable_input_never_fails_the_turn(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            self.assertEqual(0, clock_hook.main(["clock_hook.py", str(self.state)], io.StringIO("not json")))
        self.assertEqual("", out.getvalue())

    def test_it_runs_as_a_bare_file_path_on_the_stdlib_alone(self) -> None:
        # Codex runs this as a loose script, so the package may not be on the
        # interpreter's path.  The command string quotes both paths because the
        # repository path has a space in it.
        import time

        self.state.write_text(json.dumps({"thread-1": {"started": time.time(), "limit": 1800}}))
        completed = subprocess.run(
            [sys.executable, str(Path(clock_hook.__file__)), str(self.state), "1800"],
            input=json.dumps({"session_id": "thread-1", "turn_id": "t"}),
            capture_output=True,
            text=True,
            cwd=tempfile.gettempdir(),
            env={"PATH": os.environ.get("PATH", "")},
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertRegex(payload["hookSpecificOutput"]["additionalContext"], r"^\[clock\] ")


class ClockHookConfigTests(unittest.TestCase):
    def test_every_path_in_the_command_is_quoted(self) -> None:
        command = hook_command_string(
            python="/a path/python", script="/a path/clock_hook.py", state_path="/a path/s.json", limit=1800
        )
        self.assertEqual("'/a path/python' '/a path/clock_hook.py' '/a path/s.json' 1800", command)

    def test_the_startup_flag_is_toml_and_arrives_as_two_argv_entries(self) -> None:
        args = hook_cli_args("cmd")
        self.assertEqual("-c", args[0])
        self.assertEqual(2, len(args))
        # TOML inline syntax.  Under --strict-config the pinned 0.156.0 binary
        # rejects a JSON array of objects here and exits 1.
        self.assertTrue(args[1].startswith("hooks.PostToolUse=[{hooks=[{type=\"command\""))
        self.assertNotIn('"hooks":', args[1])

    def test_the_thread_table_carries_the_hook_and_its_trust(self) -> None:
        table = hook_thread_config("cmd", "sha256:abc")["hooks"]
        self.assertEqual("cmd", table["PostToolUse"][0]["hooks"][0]["command"])
        self.assertEqual({"trusted_hash": "sha256:abc"}, table["state"][TRUST_KEY])

    def test_a_merge_never_overwrites_what_the_config_map_already_carries(self) -> None:
        bearer = CommandCodeBridge.config_overrides.__doc__ is not None
        self.assertTrue(bearer)
        existing = {
            "model_providers.commandcode.http_headers.Authorization": "Bearer secret",
            "model_provider": "commandcode",
        }
        merged = merge_config(existing, hook_thread_config("cmd", "sha256:abc"))
        self.assertEqual("Bearer secret", merged["model_providers.commandcode.http_headers.Authorization"])
        self.assertEqual("commandcode", merged["model_provider"])
        self.assertIn("PostToolUse", merged["hooks"])

    def test_the_trust_hash_is_read_out_of_a_hooks_list_answer(self) -> None:
        listed = {
            "data": [
                {
                    "hooks": [
                        {"eventName": "preToolUse", "currentHash": "sha256:wrong", "command": "cmd"},
                        {"eventName": "postToolUse", "currentHash": "sha256:right", "command": "cmd"},
                    ]
                }
            ]
        }
        self.assertEqual("sha256:right", trusted_hash_from_hooks_list(listed, "cmd"))
        self.assertIsNone(trusted_hash_from_hooks_list({"data": []}, "cmd"))

    def test_only_the_clock_hook_under_the_trust_key_gives_its_hash(self) -> None:
        # Another hook whose command contains the clock command, one with no
        # command, and one under a different key all come first; none of their
        # hashes may be trusted for the clock.
        listed = {
            "data": [
                {
                    "hooks": [
                        {"eventName": "postToolUse", "key": "user:post_tool_use:0:0",
                         "currentHash": "sha256:wrapper", "command": "sh -c 'cmd && extra'"},
                        {"eventName": "postToolUse", "key": TRUST_KEY, "currentHash": "sha256:empty"},
                        {"eventName": "postToolUse", "key": "user:post_tool_use:1:0",
                         "currentHash": "sha256:other-key", "command": "cmd"},
                        {"eventName": "postToolUse", "key": TRUST_KEY,
                         "currentHash": "sha256:right", "command": "cmd"},
                    ]
                }
            ]
        }
        self.assertEqual("sha256:right", trusted_hash_from_hooks_list(listed, "cmd"))


class _FakeAdapter(VNextAppServerAdapter):
    """A adapter with the process replaced, so thread/start can be inspected."""

    def __init__(self, *, overrides: dict | None = None) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="clock-ws-"))
        self.client_name = "test-client"
        self.mcp_config_overrides = dict(overrides or {})
        self.requests: list[tuple[str, dict]] = []
        self.hooks_answer: dict = {
            "data": [{"hooks": [{"eventName": "postToolUse", "currentHash": "sha256:pinned", "command": "cmd"}]}]
        }
        self._clock_state_dir = tempfile.mkdtemp(prefix="clock-state-")
        self._clock_state_path = Path(self._clock_state_dir) / "turns.json"
        import threading

        self._clock_state_lock = threading.Lock()
        self._clock_command = "cmd"
        self._clock_trusted_hash = None
        self._clock_hash_attempted = False
        self._clock_disabled_reason = None
        self._manager_relay_name = None
        self._condition = threading.Condition(threading.RLock())
        self._events: list = []
        self._tool_registrations: dict = {}
        self._native_thread_attestations: dict = {}
        self._managed_thread_mcp: dict = {}
        self._tool_handlers: dict = {}

    def request(self, method: str, params, *, timeout: float = 30) -> dict:
        self.requests.append((method, dict(params)))
        if method == "hooks/list":
            return self.hooks_answer
        if method == "thread/start":
            return {"thread": {"id": "thread-1"}}
        if method == "thread/resume":
            return {"thread": {"id": params.get("threadId")}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        return {}

    def _attest_default_environment(self, result, *, timeout):
        return dict(result)

    def _attach_neutral_posture(self, result):
        return dict(result)

    def _tool_registration(self, *_args, **_kwargs):
        return {}

    def ensure_windows_sandbox_ready(self, **_kwargs) -> None:
        return None


def _posture() -> RuntimePosture:
    return RuntimePosture(
        workspace_writes=True,
        network="restricted",
        approvals_requested=True,
        reviewer="reviewer",
        environment_ready=True,
    )


class ThreadStartClockTests(unittest.TestCase):
    def _start(self, adapter: _FakeAdapter) -> dict:
        adapter.start_thread(
            model="model",
            developer_instructions="instructions",
            tools=[],
            requested_posture=_posture(),
        )
        return dict(next(params for method, params in adapter.requests if method == "thread/start"))

    def test_a_thread_with_no_overrides_still_carries_the_hook_table(self) -> None:
        adapter = _FakeAdapter()

        params = self._start(adapter)

        hooks = params["config"]["hooks"]
        self.assertEqual("cmd", hooks["PostToolUse"][0]["hooks"][0]["command"])
        self.assertEqual({"trusted_hash": "sha256:pinned"}, hooks["state"][TRUST_KEY])

    def test_command_codes_bearer_survives_the_merge_untouched(self) -> None:
        overrides = {
            "model_providers.commandcode.base_url": "https://example.invalid/v1",
            "model_providers.commandcode.http_headers.Authorization": "Bearer keep-me",
            "model_provider": "commandcode",
        }
        adapter = _FakeAdapter(overrides=overrides)

        params = self._start(adapter)

        for key, value in overrides.items():
            self.assertEqual(value, params["config"][key])
        self.assertIn("hooks", params["config"])

    def test_a_failed_hooks_list_starts_the_thread_with_no_hook_and_says_why(self) -> None:
        adapter = _FakeAdapter()
        adapter.hooks_answer = {"data": []}

        params = self._start(adapter)

        self.assertNotIn("config", params)
        self.assertEqual(
            "hooks/list reported no clock hook", adapter.clock_hook_attestation()["disabled_reason"]
        )

    def test_the_hash_is_fetched_once_and_reused_across_threads(self) -> None:
        adapter = _FakeAdapter()

        self._start(adapter)
        adapter.start_thread(
            model="model", developer_instructions="instructions", tools=[], requested_posture=_posture()
        )

        self.assertEqual(1, [method for method, _ in adapter.requests].count("hooks/list"))

    def test_starting_a_turn_records_its_clock_where_the_hook_reads_it(self) -> None:
        adapter = _FakeAdapter()

        adapter.start_turn(thread_id="thread-1", prompt="hello", model="model", effort="high", turn_timeout=900)

        state = json.loads(adapter._clock_state_path.read_text())
        self.assertEqual(900, state["thread-1"]["limit"])
        self.assertGreater(state["thread-1"]["started"], 0)
        self.assertGreater(state["thread-1"]["started_monotonic"], 0)

    def test_the_default_limit_in_the_command_matches_the_schedulers(self) -> None:
        self.assertEqual(1800, DEFAULT_CLOCK_LIMIT)


class ThreadResumeClockTests(unittest.TestCase):
    """A resumed thread keeps no config from its start, so it is sent again."""

    def _resume_params(self, adapter: _FakeAdapter) -> dict:
        return dict(next(params for method, params in adapter.requests if method == "thread/resume"))

    def test_a_resumed_worker_thread_carries_the_hook_table_again(self) -> None:
        adapter = _FakeAdapter()

        adapter.resume_thread(thread_id="thread-1", model="model")

        params = self._resume_params(adapter)
        hooks = params["config"]["hooks"]
        self.assertEqual("cmd", hooks["PostToolUse"][0]["hooks"][0]["command"])
        self.assertEqual({"trusted_hash": "sha256:pinned"}, hooks["state"][TRUST_KEY])

    def test_a_resume_the_hash_fetch_failed_for_sends_no_config(self) -> None:
        adapter = _FakeAdapter()
        adapter.hooks_answer = {"data": []}

        adapter.resume_thread(thread_id="thread-1", model="model")

        self.assertNotIn("config", self._resume_params(adapter))

    def test_the_resumed_root_relay_carries_the_hook_table_and_no_overrides(self) -> None:
        adapter = _FakeAdapter(
            overrides={"model_providers.commandcode.http_headers.Authorization": "Bearer keep-me"}
        )
        adapter._manager_relay_name = "relay"
        adapter.read_thread = lambda thread_id, **_kwargs: {"thread": {"id": thread_id}}
        adapter._attest_manager_relay_inventory = lambda **_kwargs: None

        adapter.resume_attested_thread(
            runtime_thread="thread-1",
            provider_session="thread-1",
            model="model",
            tools=[{"name": "tool", "description": "d", "inputSchema": {}}],
            developer_instructions="instructions",
            tool_handler=lambda *_a, **_k: {},
        )

        params = self._resume_params(adapter)
        hooks = params["config"]["hooks"]
        self.assertEqual("cmd", hooks["PostToolUse"][0]["hooks"][0]["command"])
        # The Command Code overrides belong to thread/start alone; sending them
        # here would change what a Command Code resume does.
        self.assertEqual({"hooks"}, set(params["config"]))


if __name__ == "__main__":
    unittest.main()
