from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "vnext"


def command_line(arguments: list[str]) -> str:
    return subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)


def run_hook(payload: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).resolve()), str((PLUGIN / "wake.py").resolve())],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        text=True, capture_output=True, timeout=15,
    )


class VNextPluginWakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name).resolve()

    def test_hook_configuration(self) -> None:
        configuration = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
        self.assertEqual(set(configuration), {"hooks"})
        self.assertEqual(set(configuration["hooks"]), {"PostToolUse"})
        entries = configuration["hooks"]["PostToolUse"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["matcher"], "mcp__plugin_vnext_vnext__delegate")
        hooks = entries[0]["hooks"]
        self.assertEqual(len(hooks), 1)
        hook = hooks[0]
        self.assertEqual(hook["type"], "command")
        self.assertEqual(hook["command"], "uv")
        self.assertEqual(hook["args"], ["run", "--no-project", "--quiet", "${CLAUDE_PLUGIN_ROOT}/wake.py"])
        self.assertIs(hook["asyncRewake"], True)
        self.assertGreater(hook["timeout"], 1800)

    def test_wait_result_wakes_manager(self) -> None:
        for exit_code in (0, 3, 2):
            for representation in ("string", "dict", "content"):
                with self.subTest(exit_code=exit_code, representation=representation):
                    self.check_wait_result(representation, exit_code)

    def check_wait_result(self, representation: str, exit_code: int) -> None:
        output = "child-777 completed 2026-10-05T17:00:00Z" if exit_code == 0 else "child-777 running still running after 30m"
        command = command_line([
            str(Path(sys.executable).resolve()), "-c",
            f"import sys; print({output!r}); print('wait diagnostic', file=sys.stderr); sys.exit({exit_code})",
        ])
        response = {"agent_id": "child-777", "wake_command": command}
        if representation == "string":
            response = json.dumps(response)
        elif representation == "content":
            response = [
                {"type": "image", "data": "ignored"},
                {"type": "text", "text": json.dumps(response)},
                {"type": "text", "text": "ignored"},
            ]
        result = run_hook({"tool_response": response})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        if exit_code == 0:
            self.assertEqual(
                result.stderr,
                f"vNext: worker child-777 stopped. Its report:\n{output}\nUse inspect for the full record.\n",
            )
        elif exit_code == 3:
            self.assertEqual(result.stderr, (
                f"vNext: worker child-777 is still running after 30 minutes: {output}. "
                f"Check it with inspect; to keep waiting, run this in the background: {command}\n"
            ))
        else:
            self.assertEqual(result.stderr, "vNext: could not wait for worker child-777 (exit 2): wait diagnostic\n")

    def test_missing_command_is_silent(self) -> None:
        payloads = [
            {}, {"tool_response": {"success": False, "error": "rejected"}},
            {"tool_response": "not JSON"},
            {"tool_response": [{"type": "image", "data": "ignored"}]},
            "not JSON", [], {"tool_response": {"wake_command": ""}},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                result = run_hook(payload)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")

    def test_launch_failure_wakes_manager(self) -> None:
        missing = self.tmp_path / "missing waiter"
        command = command_line([str(missing)])
        result = run_hook({"tool_response": {"agent_id": "child-777", "wake_command": command}})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("vNext: could not wait for worker child-777 (exit 2):", result.stderr)
        self.assertTrue(result.stderr.split("(exit 2):", 1)[1].strip())

    def test_success_notice_preserves_report_line_breaks(self) -> None:
        output = "child-777 completed timestamp\nFixed it.\nTests passed.\nverified: yes, evidence: 1 item(s)\n\nSecond report"
        command = command_line([str(Path(sys.executable).resolve()), "-c", f"print({output!r})"])
        result = run_hook({"tool_response": {"agent_id": "child-777", "wake_command": command}})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, (
            f"vNext: worker child-777 stopped. Its report:\n{output}\n"
            "Use inspect for the full record.\n"
        ))

    def test_report_characters_survive_a_pipe_with_a_narrow_code_page(self) -> None:
        # A Windows pipe defaults to the locale code page, which has no arrow.
        runs = self.tmp_path / ".vnext" / "runs"
        runs.mkdir(parents=True)
        (runs / "s.jsonl").write_text(json.dumps({
            "type": "agent.upsert", "agent_id": "a1", "timestamp": "t",
            "payload": {"status": "completed", "result": {"outcome": "fixed A → B", "verified": True}},
        }) + "\n", encoding="utf-8")
        wait = [str(Path(sys.executable).resolve()), "-m", "vnext.vnext_wait",
                "--workspace", str(self.tmp_path), "--agent", "a1", "--session", "s", "--deadline", "1m"]
        root = str(PLUGIN.parents[1])
        narrow = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONPATH": root}
        direct = subprocess.run(wait, cwd=root, env=narrow, capture_output=True, timeout=30)
        self.assertEqual(direct.returncode, 0, direct.stderr)
        hook = subprocess.run(
            [str(Path(sys.executable).resolve()), str((PLUGIN / "wake.py").resolve())],
            input=json.dumps({"tool_response": {"agent_id": "a1", "wake_command": command_line(wait)}}),
            cwd=root, env=narrow, encoding="utf-8", capture_output=True, timeout=30,
        )
        self.assertEqual(hook.returncode, 2)
        self.assertIn("fixed A → B", hook.stderr)


if __name__ == "__main__":
    unittest.main()
