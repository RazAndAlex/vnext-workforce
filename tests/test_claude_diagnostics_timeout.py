"""Regression for a delayed read-only bridge diagnostics response.

The temporary JSONL child has no provider SDK or credentials.  It proves the
public adapter API drains a response that arrived after the caller gave up,
then remains usable for the next diagnostic probe and orderly shutdown.
"""

from __future__ import annotations

import tempfile
import sys
import time
import unittest
from pathlib import Path

from vnext.vnext_claude import ClaudeCodeAdapter


_BRIDGE = r'''
import json
import sys
import time
from pathlib import Path

marker = Path(sys.argv[1])
diagnostics = 0

def reply(request_id, result):
    print(json.dumps({"v": 1, "kind": "response", "id": request_id, "ok": True, "result": result}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if request.get("kind") != "request":
        continue
    operation = request["op"]
    if operation == "initialize":
        reply(request["id"], {
            "provider": "claude", "harness": "claude-agent-sdk",
            "tool_support": {"accepted": True, "requires_empty": False},
            "credential_override_rejected": True,
        })
    elif operation == "diagnostics":
        diagnostics += 1
        if diagnostics == 1:
            time.sleep(0.12)
        reply(request["id"], {"probe": diagnostics})
        if diagnostics == 1:
            print(json.dumps({"v": 1, "kind": "event", "event": {"name": "system", "marker": "late"}}), flush=True)
            marker.write_text("ready", encoding="utf-8")
    elif operation == "close":
        reply(request["id"], {"closed": True})
        break
'''


class ClaudeDiagnosticsTimeoutTests(unittest.TestCase):
    def test_late_public_diagnostics_reply_drains_and_subsequent_probe_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            bridge = workspace / "delayed_bridge.py"
            marker = workspace / "late-output-ready"
            bridge.write_text(_BRIDGE, encoding="utf-8")
            adapter = ClaudeCodeAdapter(
                workspace=workspace,
                bridge_command=(sys.executable, str(bridge), str(marker)),
                request_timeout_seconds=0.5,
            )
            try:
                adapter.initialize()
                first = adapter.diagnostics(timeout_seconds=0.01)
                self.assertEqual({"available": False, "error_type": "ClaudeRuntimeError"}, first["bridge"])

                deadline = time.monotonic() + 2
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "delayed bridge output was not emitted")

                with adapter._condition:
                    self.assertIsNone(adapter._fatal)
                    self.assertTrue(adapter._reader_threads[0].is_alive())
                    self.assertEqual((), tuple(adapter._expired_read_only_requests))
                    self.assertEqual(1, sum(event.get("marker") == "late" for event in adapter._events))

                second = adapter.diagnostics(timeout_seconds=0.5)
                self.assertEqual(2, second["bridge"]["probe"])
                self.assertEqual(1, second["adapter"]["late_read_only_response_count"])
            finally:
                cleanup = adapter.close()
            self.assertEqual(0, cleanup.process.residual_count)
            self.assertTrue(cleanup.streams_drained)
            self.assertEqual((), cleanup.errors)


if __name__ == "__main__":
    unittest.main()
