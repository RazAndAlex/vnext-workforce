"""A scripted runtime adapter that answers one turn and then closes.

Several test modules drive a session through this adapter, and some of them
ship in the public export while others stay behind. A helper module keeps the
shared class out of any single test file, so an exported test never has to
import a test module the export omits.
"""

from __future__ import annotations

from vnext.process_supervisor import ProcessCleanup
from vnext.vnext_runtime_types import RuntimeCleanup, ToolCallContext

from tests.test_vnext_scheduler import ScriptedAdapter


class ReplyAdapter(ScriptedAdapter):
    provider = "codex"
    harness = "app-server"

    def initialize(self):
        return {}

    def events_since(self, cursor=None):
        return tuple(self.events[cursor or 0:])

    def wait_turn(self, handle, *, timeout=300):
        handler = self.handlers[handle.thread_id]
        context = ToolCallContext(handle.thread_id, handle.turn_id, "test-call")
        handler("read_messages", {}, context)
        result = handler("complete_session", {"decision": "accepted", "summary": "Verified answer",
            "criteria": {"responded": True}}, context)
        if not result.success:
            raise AssertionError(result.as_json_text())
        self.events.append({"method": "item/completed", "params": {"threadId": handle.thread_id,
            "turnId": handle.turn_id, "item": {"id": "answer-" + handle.turn_id,
            "type": "agentMessage", "text": "Answer " + handle.turn_id}}})
        return {"status": "completed"}

    def close(self):
        return RuntimeCleanup(ProcessCleanup("test", 0, 0, ()), True, True, ())
