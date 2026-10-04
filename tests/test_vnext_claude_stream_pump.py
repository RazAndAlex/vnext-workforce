"""The bridge keeps reading a Claude session between vNext turns.

These tests drive the real SDK client and its ``Query`` read loop through a
fake transport.  The CLI is the only thing faked: the SDK's 100-frame message
buffer, its blocking send and its hook-callback routing are the real code
whose interaction stalled a worker for 16 hours on 3-4 October 2026.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    import claude_agent_sdk
    from claude_agent_sdk._internal.transport import Transport
except ImportError:  # pragma: no cover - the gate interpreter carries the SDK
    claude_agent_sdk = None
    Transport = object  # type: ignore[assignment,misc]

from vnext.vnext_claude_bridge import _Bridge

SESSION = "6f9c2f0e-5d1c-4a59-9b1e-2f6c1f0d8a11"
RESERVATION = "claude-reservation-pump"


def _assistant(text: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "model": "claude-opus-5-5", "content": [{"type": "text", "text": text}]},
        "parent_tool_use_id": None,
        "session_id": SESSION,
    }


def _stream_delta(text: str) -> dict[str, Any]:
    return {
        "type": "stream_event",
        "uuid": "u",
        "session_id": SESSION,
        "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        "parent_tool_use_id": None,
    }


def _result(text: str, *, num_turns: int = 1) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": num_turns,
        "session_id": SESSION,
        "total_cost_usd": 0.0,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "result": text,
    }


# The whole tool result the CLI (2.1.286) writes when a PreToolUse hook does
# not answer, as the 4 October worker transcript recorded it.
HOOK_SENTENCE = (
    "PreToolUse hook did not respond before its timeout (host client may be unreachable). "
    "The tool call was not executed; other configured hooks may not have completed."
)


def _tool_round(index: int, text: str, *, is_error: bool, as_blocks: bool = False) -> list[dict[str, Any]]:
    """One Bash call and its result, as the CLI writes them."""

    tool_use_id = f"toolu_{index:04d}"
    content: Any = [{"type": "text", "text": text}] if as_blocks else text
    return [
        {
            "type": "assistant",
            "message": {
                "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": f"step {index}"}}],
            },
            "parent_tool_use_id": None,
            "session_id": SESSION,
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}],
            },
            "parent_tool_use_id": None,
            "session_id": SESSION,
        },
    ]


def _rounds_ending_in_hook_failures(failing: int) -> list[dict[str, Any]]:
    """A long turn: early hook failures it got past, a git hook failing inside
    Bash, then a trailing run of tool calls whose hooks never answered."""

    frames: list[dict[str, Any]] = []
    for index in range(60):
        if 10 <= index < 13:
            frames += _tool_round(index, HOOK_SENTENCE, is_error=True)
        elif index == 50:
            frames += _tool_round(index, "husky - pre-commit hook failed (add --no-verify to bypass)", is_error=True)
        else:
            frames += _tool_round(index, f"output {index}\n" * 40, is_error=False)
    for index in range(60, 60 + failing):
        frames += _tool_round(index, HOOK_SENTENCE, is_error=True, as_blocks=index % 2 == 0)
    return frames


class FakeCLI(Transport):  # type: ignore[misc]
    """Answers initialize, answers each prompt with one turn, records writes."""

    def __init__(self) -> None:
        self.outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.written: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self.hold_prompts = False
        self.held: list[str] = []
        self.answered = asyncio.Event()
        self.scripts: dict[str, list[dict[str, Any]]] = {}
        # A prompt sent with a uuid is echoed back when its turn starts, as
        # the CLI does under --replay-user-messages (probed on 2.1.286).
        self.uuids: dict[str, str] = {}
        # False plays a CLI that accepts the flag and never echoes.
        self.echoes = True

    async def connect(self) -> None:
        return None

    def answer(self, prompt: str) -> None:
        uuid = self.uuids.pop(prompt, None)
        if uuid is not None and self.echoes:
            self.outbox.put_nowait({
                "type": "user", "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None, "session_id": SESSION, "uuid": uuid,
            })
        for frame in self.scripts.pop(prompt, []):
            self.outbox.put_nowait(frame)
        self.outbox.put_nowait(_assistant(f"answer to {prompt}"))
        self.outbox.put_nowait(_result(f"answer to {prompt}"))

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if not line.strip():
                continue
            frame = json.loads(line)
            self.written.append(frame)
            if frame.get("type") == "control_request":
                request = frame.get("request", {})
                response: dict[str, Any] = {}
                if request.get("subtype") == "initialize":
                    response = {"commands": [], "models": [], "account": {}}
                self.outbox.put_nowait({
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": frame["request_id"], "response": response},
                })
            elif frame.get("type") == "control_response":
                self.answered.set()
            elif frame.get("type") == "user":
                prompt = frame["message"]["content"]
                self.prompts.append(prompt)
                if frame.get("uuid"):
                    self.uuids[prompt] = frame["uuid"]
                if self.hold_prompts:
                    self.held.append(prompt)
                else:
                    self.answer(prompt)

    def release_held(self) -> None:
        self.hold_prompts = False
        for prompt in self.held:
            self.answer(prompt)
        self.held.clear()

    async def read_messages(self):  # type: ignore[override]
        while True:
            frame = await self.outbox.get()
            if frame is None:
                return
            yield frame

    async def close(self) -> None:
        self.outbox.put_nowait(None)

    def is_ready(self) -> bool:
        return True

    async def end_input(self) -> None:
        return None

    def hook_answer(self, request_id: str) -> dict[str, Any] | None:
        for frame in self.written:
            response = frame.get("response") if frame.get("type") == "control_response" else None
            if isinstance(response, dict) and response.get("request_id") == request_id:
                return response
        return None


@unittest.skipIf(claude_agent_sdk is None, "claude-agent-sdk is not installed")
class StreamPumpTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name).resolve()
        self.events: list[dict[str, Any]] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def _connected(
        self, hook_calls: list[str], *, replay: bool = False,
    ) -> tuple[_Bridge, FakeCLI, Any]:
        async def pre_tool_use(_data: Any, _tool_use_id: Any, _context: Any) -> dict[str, Any]:
            hook_calls.append("PreToolUse")
            return {"continue": True}

        transport = FakeCLI()
        options = claude_agent_sdk.ClaudeAgentOptions(
            hooks={"PreToolUse": [claude_agent_sdk.HookMatcher(matcher=None, hooks=[pre_tool_use])]},
            **({"extra_args": {"replay-user-messages": None}} if replay else {}),
        )
        client = claude_agent_sdk.ClaudeSDKClient(options=options, transport=transport)
        await client.connect(None)
        bridge = _Bridge()
        bridge._sdk = claude_agent_sdk
        bridge._workspace = self.workspace
        state = bridge._reservation_state("auto_review", None, [], reservation_id=RESERVATION)
        state["client"] = client
        state["effort"] = "medium"
        bridge._reservations[RESERVATION] = state
        return bridge, transport, client

    async def _turn(self, bridge: _Bridge, generation: int) -> dict[str, Any]:
        reference = f"turn-{generation}"
        await bridge._start_turn({
            "reservation_id": RESERVATION, "turn_reference": reference,
            "generation": generation, "prompt": f"prompt {generation}", "effort": "medium",
        })
        return dict(await asyncio.wait_for(
            bridge._wait_turn({"reservation_id": RESERVATION, "turn_reference": reference}), 5,
        ))

    @staticmethod
    def _hook_request(request_id: str, callback_id: str) -> dict[str, Any]:
        return {
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "hook_callback",
                "callback_id": callback_id,
                "input": {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}},
                "tool_use_id": "toolu_between_turns",
            },
        }

    def test_hook_between_turns_is_answered_after_150_unread_frames(self) -> None:
        """The incident: a turn the CLI started alone fills the SDK buffer.

        Before the fix nobody read the stream after the vNext turn's result,
        so the SDK read loop stopped at frame 101 and never reached the hook
        request that came after it.
        """

        async def scenario() -> None:
            hook_calls: list[str] = []
            bridge, cli, client = await self._connected(hook_calls)
            first = await self._turn(bridge, 1)
            self.assertEqual(first["status"], "completed")
            callback_id = next(iter(client._query.hook_callbacks))
            # The CLI starts a turn of its own: a background Bash finished.
            for index in range(150):
                cli.outbox.put_nowait(_stream_delta(f"delta {index} "))
            cli.outbox.put_nowait(self._hook_request("req_hook_between_turns", callback_id))
            try:
                await asyncio.wait_for(cli.answered.wait(), 3)
            except asyncio.TimeoutError:
                pass
            answer = cli.hook_answer("req_hook_between_turns")
            self.assertIsNotNone(answer, "the hook request after 150 unread frames got no answer")
            self.assertEqual(answer["subtype"], "success")
            self.assertEqual(hook_calls, ["PreToolUse"])
            # The CLI's own turn ends, and the bridge records it as one.
            cli.outbox.put_nowait(_assistant("background Bash finished"))
            cli.outbox.put_nowait(_result("background Bash finished"))
            for _ in range(100):
                if any(
                    event["event"].get("name") == "unsolicited_turn"
                    and event["event"].get("status") == "completed"
                    for event in self.events
                ):
                    break
                await asyncio.sleep(0.01)
            unsolicited = [
                event["event"] for event in self.events if event["event"].get("name") == "unsolicited_turn"
            ]
            self.assertEqual([event["status"] for event in unsolicited], ["started", "completed"])
            self.assertEqual(unsolicited[0]["after_turn_reference"], "turn-1")
            self.assertEqual(unsolicited[1]["frames"], 152)
            streamed = [
                event["event"] for event in self.events
                if event["event"].get("name") == "stream" and event["event"]["content"].get("text", "").startswith("delta ")
            ]
            self.assertEqual(len(streamed), 150)
            # The next vNext turn still gets its own answer.
            second = await self._turn(bridge, 2)
            self.assertEqual(second["status"], "completed")
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})
            self.assertIsNone(bridge._reservations[RESERVATION]["pump"])

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_a_malformed_frame_between_turns_does_not_stop_the_reader(self) -> None:
        """The SDK raises MessageParseError for a frame it cannot parse.

        Before the fix that ended the reader between turns without a word:
        no event, and 150 frames later the SDK buffer was full again and the
        next hook got no answer, the stall this reader exists to prevent.
        """

        async def scenario() -> None:
            hook_calls: list[str] = []
            bridge, cli, client = await self._connected(hook_calls)
            await self._turn(bridge, 1)
            callback_id = next(iter(client._query.hook_callbacks))
            # An assistant frame with no model: the SDK cannot parse it.
            cli.outbox.put_nowait({
                "type": "assistant", "message": {"role": "assistant", "content": []},
                "parent_tool_use_id": None, "session_id": SESSION,
            })
            for index in range(150):
                cli.outbox.put_nowait(_stream_delta(f"delta {index} "))
            cli.outbox.put_nowait(self._hook_request("req_hook_after_bad_frame", callback_id))
            try:
                await asyncio.wait_for(cli.answered.wait(), 3)
            except asyncio.TimeoutError:
                pass
            self.assertIsNotNone(
                cli.hook_answer("req_hook_after_bad_frame"),
                "the hook request after a malformed frame and 150 more got no answer",
            )
            dropped = self._unsolicited("frame-dropped")
            self.assertEqual([event["error"] for event in dropped], ["MessageParseError"])
            cli.outbox.put_nowait(_assistant("background done"))
            cli.outbox.put_nowait(_result("background done"))
            second = await self._turn(bridge, 2)
            self.assertEqual(second["status"], "completed")
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})
            # A release stops the reader on purpose: that is no lost stream.
            self.assertEqual(self._unsolicited("reader-ended"), [])

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_a_stream_that_ends_between_turns_is_reported(self) -> None:
        """The CLI went away while no vNext turn ran: nothing will answer again."""

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            await self._turn(bridge, 1)
            await cli.close()
            for _ in range(200):
                if self._unsolicited("reader-ended"):
                    break
                await asyncio.sleep(0.01)
            ended = self._unsolicited("reader-ended")
            self.assertEqual(len(ended), 1, "the reader ended between turns and said nothing")
            self.assertEqual(ended[0]["after_turn_reference"], "turn-1")
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def _unsolicited(self, status: str) -> list[dict[str, Any]]:
        return [
            event["event"] for event in self.events
            if event["event"].get("name") == "unsolicited_turn" and event["event"].get("status") == status
        ]

    def test_a_clis_own_turn_that_had_not_streamed_yet_never_completes_the_next_vnext_turn(self) -> None:
        """Review finding 1: the CLI started its own turn, but no frame of it
        had arrived when vNext sent the next prompt.

        Nothing marked a turn open, so the new turn did not wait, and before
        the fix it took the CLI's result (num_turns 7) as its own answer.  The
        CLI echoes a prompt's uuid when that prompt's turn starts, and only
        then; the frames before the echo belong to the CLI's own turn.
        """

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([], replay=True)
            first = await self._turn(bridge, 1)
            self.assertEqual(first["num_turns"], 1)
            cli.hold_prompts = True
            await bridge._start_turn({
                "reservation_id": RESERVATION, "turn_reference": "turn-2",
                "generation": 2, "prompt": "prompt 2", "effort": "medium",
            })
            waiter = asyncio.ensure_future(
                bridge._wait_turn({"reservation_id": RESERVATION, "turn_reference": "turn-2"})
            )
            for _ in range(100):
                if "prompt 2" in cli.prompts:
                    break
                await asyncio.sleep(0.01)
            # The CLI's own turn runs before the queued prompt.
            cli.outbox.put_nowait(_assistant("background Bash finished"))
            cli.outbox.put_nowait(_result("background Bash finished", num_turns=7))
            await asyncio.sleep(0.3)
            self.assertFalse(
                waiter.done() and waiter.result().get("num_turns") == 7,
                "the CLI's own result completed the vNext turn that came after it",
            )
            cli.release_held()
            second = dict(await asyncio.wait_for(waiter, 5))
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["num_turns"], 1)
            completed = self._unsolicited("completed")
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["after_turn_reference"], "turn-1")
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_a_cli_that_never_echoes_its_prompts_still_completes_every_turn(self) -> None:
        """A CLI version that takes --replay-user-messages and never echoes.

        Before the fallback the first turn waited for an echo that never
        came, recorded its own answer as the CLI's turn, and hung.  A client
        that has not echoed yet keeps the handling it had before the echo.
        """

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([], replay=True)
            cli.echoes = False
            for generation in (1, 2, 3):
                outcome = await self._turn(bridge, generation)
                self.assertEqual(outcome["status"], "completed")
                self.assertEqual(outcome["num_turns"], 1)
            self.assertEqual(self._unsolicited("completed"), [])
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_the_echo_of_the_first_prompt_is_not_part_of_the_turn(self) -> None:
        """An echoing CLI: the first turn learns the echo and drops it."""

        async def scenario() -> None:
            bridge, _cli, _client = await self._connected([], replay=True)
            for generation in (1, 2):
                outcome = await self._turn(bridge, generation)
                self.assertEqual(outcome["status"], "completed")
            self.assertEqual(self._unsolicited("completed"), [])
            self.assertNotIn('"role": "user"', json.dumps(self.events))
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_an_unsolicited_turns_result_never_completes_the_next_vnext_turn(self) -> None:
        """The second half of the incident: the resume turn closed on an old result.

        The CLI's own turn is still running when vNext starts the next turn.
        Its ResultMessage arrives first and carries no prompt correlation, so
        before the fix the new turn took it as its own answer.
        """

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            first = await self._turn(bridge, 1)
            self.assertEqual(first["num_turns"], 1)
            # The CLI's own turn: a long stream, and no result yet.
            for index in range(120):
                cli.outbox.put_nowait(_stream_delta(f"background {index} "))
            cli.outbox.put_nowait(_assistant("background Bash finished"))
            for _ in range(100):
                if self._unsolicited("started"):
                    break
                await asyncio.sleep(0.01)
            # vNext resumes the worker while that turn is open.  The CLI
            # queues the prompt behind its running turn.
            cli.hold_prompts = True
            await bridge._start_turn({
                "reservation_id": RESERVATION, "turn_reference": "turn-2",
                "generation": 2, "prompt": "prompt 2", "effort": "medium",
            })
            waiter = asyncio.ensure_future(
                bridge._wait_turn({"reservation_id": RESERVATION, "turn_reference": "turn-2"})
            )
            cli.outbox.put_nowait(_result("background Bash finished", num_turns=7))
            await asyncio.sleep(0.3)
            self.assertFalse(
                waiter.done() and waiter.result().get("num_turns") == 7,
                "the CLI's own result completed the vNext turn that came after it",
            )
            cli.release_held()
            second = dict(await asyncio.wait_for(waiter, 5))
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["num_turns"], 1)
            self.assertEqual(cli.prompts, ["prompt 1", "prompt 2"])
            completed = self._unsolicited("completed")
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["frames"], 122)
            self.assertEqual(completed[0]["after_turn_reference"], "turn-1")
            # The new turn's own answer reached the record under its reference.
            answers = [
                event["event"] for event in self.events
                if event["event"].get("name") == "message" and event["event"].get("turn_reference") == "turn-2"
            ]
            self.assertTrue(answers)
            self.assertIn("answer to prompt 2", json.dumps(answers))
            self.assertNotIn("background", json.dumps(answers))
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_an_interrupted_turn_that_waits_for_the_clis_own_turn_never_sends_its_prompt(self) -> None:
        """Interrupting turn 2 while it waits must stop it, not just delay it.

        Before the fix the interrupt reached the CLI's own turn, the bridge
        answered interrupted=true, and when that turn's abort result came the
        reader still sent prompt 2 and the turn ended "completed".
        """

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            await self._turn(bridge, 1)
            cli.outbox.put_nowait(_assistant("background Bash finished"))
            for _ in range(100):
                if self._unsolicited("started"):
                    break
                await asyncio.sleep(0.01)
            await bridge._start_turn({
                "reservation_id": RESERVATION, "turn_reference": "turn-2",
                "generation": 2, "prompt": "prompt 2", "effort": "medium",
            })
            waiter = asyncio.ensure_future(
                bridge._wait_turn({"reservation_id": RESERVATION, "turn_reference": "turn-2"})
            )
            await asyncio.sleep(0.05)
            answer = await asyncio.wait_for(
                bridge._interrupt({"reservation_id": RESERVATION, "turn_reference": "turn-2"}), 5,
            )
            self.assertEqual(answer["interrupted"], True)
            # The CLI's own turn ends on the interrupt.
            aborted = _result("background Bash finished", num_turns=3)
            aborted.update({"subtype": "error_during_execution", "is_error": True})
            cli.outbox.put_nowait(aborted)
            second = dict(await asyncio.wait_for(waiter, 5))
            await asyncio.sleep(0.1)
            self.assertEqual(cli.prompts, ["prompt 1"], "the interrupted turn still sent its prompt")
            self.assertEqual(second["status"], "interrupted")
            # The next turn runs normally.
            third = await self._turn(bridge, 3)
            self.assertEqual(third["status"], "completed")
            self.assertEqual(cli.prompts, ["prompt 1", "prompt 3"])
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    async def _interrupt_before_echo(self) -> tuple[_Bridge, FakeCLI, asyncio.Future[Any]]:
        """An echoing client, turn 2 sent and interrupted before its echo.

        The CLI discards the queued prompt on the interrupt, so prompt 2 is
        never echoed and never answered.
        """

        bridge, cli, _client = await self._connected([], replay=True)
        first = await self._turn(bridge, 1)
        self.assertEqual(first["status"], "completed")
        cli.hold_prompts = True
        await bridge._start_turn({
            "reservation_id": RESERVATION, "turn_reference": "turn-2",
            "generation": 2, "prompt": "prompt 2", "effort": "medium",
        })
        waiter = asyncio.ensure_future(
            bridge._wait_turn({"reservation_id": RESERVATION, "turn_reference": "turn-2"})
        )
        for _ in range(100):
            if "prompt 2" in cli.prompts:
                break
            await asyncio.sleep(0.01)
        self.assertIn("prompt 2", cli.prompts)
        answer = await asyncio.wait_for(
            bridge._interrupt({"reservation_id": RESERVATION, "turn_reference": "turn-2"}), 5,
        )
        self.assertEqual(answer["interrupted"], True)
        cli.held.clear()
        cli.uuids.clear()
        cli.hold_prompts = False
        return bridge, cli, waiter

    def test_a_turn_interrupted_before_its_echo_ends_on_the_abort_result(self) -> None:
        """Before the fix the turn kept waiting for an echo the CLI would
        never send, and recorded the abort result as the CLI's own turn."""

        async def scenario() -> None:
            bridge, cli, waiter = await self._interrupt_before_echo()
            aborted = _result("aborted", num_turns=3)
            aborted.update({"subtype": "error_during_execution", "is_error": True})
            cli.outbox.put_nowait(aborted)
            second = dict(await asyncio.wait_for(waiter, 5))
            self.assertEqual(second["status"], "interrupted")
            # The next turn runs normally and still waits for its own echo.
            third = await self._turn(bridge, 3)
            self.assertEqual(third["status"], "completed")
            self.assertEqual(third["num_turns"], 1)
            self.assertIn("answer to prompt 3", json.dumps([
                event["event"] for event in self.events if event["event"].get("turn_reference") == "turn-3"
            ]))
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_a_turn_interrupted_before_its_echo_ends_when_the_stream_ends(self) -> None:
        async def scenario() -> None:
            bridge, cli, waiter = await self._interrupt_before_echo()
            await cli.close()
            second = dict(await asyncio.wait_for(waiter, 5))
            self.assertEqual(second["status"], "interrupted")
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_a_turn_ending_on_hook_failures_reports_them_to_the_scheduler(self) -> None:
        """Gap 2: the turn result says its last tool calls failed in a hook.

        Before the fix the result carried nothing of the kind, so the
        scheduler could only read the turn as an ordinary end.
        """

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            cli.scripts["prompt 1"] = _rounds_ending_in_hook_failures(5)
            first = await self._turn(bridge, 1)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first.get("hook_failures"), {"count": 5, "first": HOOK_SENTENCE})
            # The next turn starts clean, and one whose tools ran says nothing.
            second = await self._turn(bridge, 2)
            self.assertNotIn("hook_failures", second)
            # Only the CLI's sentence left the bridge, never Bash output.
            self.assertNotIn("output 3", json.dumps(self.events))
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_command_output_that_quotes_the_hook_sentence_is_not_a_hook_failure(self) -> None:
        """A Bash call that ran and failed is no hook failure, whatever it printed.

        Before the fix any line of a failed tool result that named a hook
        and said "did not respond" counted, so a pytest run asserting on that
        very sentence blocked the worker.
        """

        pytest_output = "\n".join([
            "FAILED tests/test_hooks.py::test_timeout - AssertionError:",
            "AssertionError: PreToolUse hook did not respond before its timeout",
            "E   PreToolUse hook failed with an unexpected error",
            "1 failed, 12 passed in 0.31s",
        ])

        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            cli.scripts["prompt 1"] = (
                _tool_round(1, pytest_output, is_error=True)
                + _tool_round(2, pytest_output, is_error=True, as_blocks=True)
                + _tool_round(3, "PostToolUse hook timed out", is_error=True)
            )
            first = await self._turn(bridge, 1)
            self.assertNotIn("hook_failures", first)
            # The CLI's own sentence, the whole tool result, still counts.
            cli.scripts["prompt 2"] = _tool_round(4, HOOK_SENTENCE, is_error=True)
            second = await self._turn(bridge, 2)
            self.assertEqual(second.get("hook_failures"), {"count": 1, "first": HOOK_SENTENCE})
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_an_unsolicited_turn_ending_on_hook_failures_says_so_when_it_ends(self) -> None:
        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            await self._turn(bridge, 1)
            cli.outbox.put_nowait(_assistant("background Bash finished"))
            for frame in _rounds_ending_in_hook_failures(3):
                cli.outbox.put_nowait(frame)
            cli.outbox.put_nowait(_result("background Bash finished"))
            for _ in range(200):
                if self._unsolicited("completed"):
                    break
                await asyncio.sleep(0.01)
            completed = self._unsolicited("completed")
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["hook_failures"], {"count": 3, "first": HOOK_SENTENCE})
            # The unsolicited turn's failures are its own, never the next turn's.
            second = await self._turn(bridge, 2)
            self.assertNotIn("hook_failures", second)
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_the_cli_is_told_to_wait_30_seconds_for_each_bridge_hook(self) -> None:
        """Gap 3: the four hooks carry a timeout the CLI receives at initialize.

        The bridge's own options go through the real SDK client, and the
        assertion reads the initialize request the CLI would get.  Approvals
        stay on can_use_tool, which the SDK sends as a flag and not a hook.
        """

        tool = {
            "type": "function", "name": "delegate", "description": "Create one direct child.",
            "inputSchema": {"type": "object", "properties": {"objective": {"type": "string"}},
                            "required": ["objective"], "additionalProperties": False},
        }

        async def scenario() -> None:
            bridge = _Bridge()
            bridge._sdk = claude_agent_sdk
            bridge._workspace = self.workspace
            state = bridge._reservation_state("auto_review", None, [tool], reservation_id=RESERVATION)
            bridge._reservations[RESERVATION] = state
            options = bridge._options(
                model="claude-opus-5-5", resume=None, reservation_id=RESERVATION, definitions=[tool],
            )
            self.assertIsNotNone(options.can_use_tool)
            transport = FakeCLI()
            client = claude_agent_sdk.ClaudeSDKClient(options=options, transport=transport)
            await client.connect(None)
            try:
                initialize = next(
                    frame["request"] for frame in transport.written
                    if frame.get("type") == "control_request" and frame["request"].get("subtype") == "initialize"
                )
                hooks = initialize["hooks"]
                self.assertEqual({"PostToolUse", "PreToolUse", "SubagentStart", "SubagentStop"}, set(hooks))
                self.assertEqual(
                    {name: [30] for name in hooks},
                    {name: [matcher.get("timeout") for matcher in matchers] for name, matchers in hooks.items()},
                )
                self.assertNotIn("PermissionRequest", hooks)
            finally:
                await client.disconnect()

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_the_cli_is_started_with_the_flag_that_echoes_each_prompt(self) -> None:
        """The bridge's own options reach the CLI command line through the
        real SDK transport, and the turn reader sees the flag on the client."""

        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

        from vnext.vnext_claude_bridge import _replays_prompts

        bridge = _Bridge()
        bridge._sdk = claude_agent_sdk
        bridge._workspace = self.workspace
        bridge._reservations[RESERVATION] = bridge._reservation_state(
            "auto_review", None, [], reservation_id=RESERVATION,
        )
        options = bridge._options(
            model="claude-opus-5-5", resume=None, reservation_id=RESERVATION, definitions=[],
        )
        transport = SubprocessCLITransport(prompt="", options=options)
        transport._cli_path = "claude"
        self.assertIn("--replay-user-messages", transport._build_command())
        client = claude_agent_sdk.ClaudeSDKClient(options=options, transport=FakeCLI())
        self.assertTrue(_replays_prompts(client))

    def test_a_tool_call_in_the_clis_own_turn_goes_to_the_agents_approvals(self) -> None:
        """Live on 4 October: a worker's own turn had its first Bash refused.

        The request came while no vNext turn ran, so the bridge answered
        "missing local approval routing handle" with interrupt and the CLI
        stopped the turn.  The request now reaches the reviewer under the turn
        the CLI's turn followed, and its answer still lands after vNext has
        started the next turn.
        """

        async def scenario() -> None:
            bridge = _Bridge()
            bridge._sdk = claude_agent_sdk
            bridge._workspace = self.workspace
            state = bridge._reservation_state("auto_review", None, [], reservation_id=RESERVATION)
            state["effort"] = "medium"
            bridge._reservations[RESERVATION] = state
            options = bridge._options(model="claude-opus-5-5", resume=None, reservation_id=RESERVATION, definitions=[])
            cli = FakeCLI()
            client = claude_agent_sdk.ClaudeSDKClient(options=options, transport=cli)
            await client.connect(None)
            state["client"] = client
            await self._turn(bridge, 1)
            # The CLI's own turn: a long stream, then a Bash it wants to run.
            for index in range(120):
                cli.outbox.put_nowait(_stream_delta(f"background {index} "))
            cli.outbox.put_nowait(_tool_round(900, "", is_error=False)[0])
            cli.outbox.put_nowait({
                "type": "control_request",
                "request_id": "req_permission_between_turns",
                "request": {
                    "subtype": "can_use_tool", "tool_name": "Bash",
                    "input": {"command": "git status"}, "tool_use_id": "toolu_0900",
                },
            })
            for _ in range(200):
                if any(event["event"].get("name") == "permission" for event in self.events):
                    break
                if cli.hook_answer("req_permission_between_turns") is not None:
                    break
                await asyncio.sleep(0.01)
            refused = cli.hook_answer("req_permission_between_turns")
            self.assertIsNone(refused, f"the CLI's own tool call was answered without review: {refused}")
            permissions = [event["event"] for event in self.events if event["event"].get("name") == "permission"]
            self.assertEqual(len(permissions), 1)
            self.assertEqual(permissions[0]["turn_reference"], "turn-1")
            self.assertEqual(permissions[0]["tool_use_id"], "toolu_0900")
            self.assertEqual(permissions[0]["effect"], "execute")
            # vNext resumes the worker while the review is still open.
            cli.hold_prompts = True
            await bridge._start_turn({
                "reservation_id": RESERVATION, "turn_reference": "turn-2",
                "generation": 2, "prompt": "prompt 2", "effort": "medium",
            })
            bridge._resolve_permission({
                "v": 1, "kind": "control", "op": "permission_response", "reservation_id": RESERVATION,
                "turn_reference": "turn-1", "tool_use_id": "toolu_0900", "decision": {"decision": "accept"},
            })
            for _ in range(200):
                if cli.hook_answer("req_permission_between_turns") is not None:
                    break
                await asyncio.sleep(0.01)
            answer = cli.hook_answer("req_permission_between_turns")
            self.assertIsNotNone(answer, "the reviewer's answer never reached the CLI")
            self.assertEqual(answer["response"]["behavior"], "allow")
            cli.outbox.put_nowait(_result("background Bash finished", num_turns=3))
            cli.release_held()
            second = dict(await asyncio.wait_for(
                bridge._wait_turn({"reservation_id": RESERVATION, "turn_reference": "turn-2"}), 5,
            ))
            self.assertEqual(second["num_turns"], 1)
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())

    def test_compact_refuses_while_an_unsolicited_turn_is_open(self) -> None:
        async def scenario() -> None:
            bridge, cli, _client = await self._connected([])
            await self._turn(bridge, 1)
            cli.outbox.put_nowait(_assistant("background Bash finished"))
            for _ in range(100):
                if self._unsolicited("started"):
                    break
                await asyncio.sleep(0.01)
            with self.assertRaisesRegex(Exception, "idle connected Claude reservation"):
                await bridge._compact({"reservation_id": RESERVATION})
            cli.outbox.put_nowait(_result("background Bash finished"))
            await asyncio.sleep(0.1)
            await bridge._release_agent({"reservation_id": RESERVATION, "status": "cancelled"})

        with patch("vnext.vnext_claude_bridge._write", side_effect=self.events.append):
            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
