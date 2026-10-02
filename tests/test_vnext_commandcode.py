"""The bridge that lets the Codex runtime reach a Command Code model.

Two things decide whether this provider works at all, and both are checked
against a live HTTP exchange rather than against the rewrite helper alone.

Command Code accepts only ``type: "function"`` tools, and the Codex CLI always
sends a ``web_search`` entry for a model slug it does not recognise.  A request
that arrives upstream still carrying one is the whole failure.

The account key must reach Command Code and must reach nothing else.  Codex is
configured with a bearer generated for the session, so a test that only reads
``config_overrides`` would pass while the real key leaked through the socket.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext.vnext_commandcode import (
    COMMANDCODE_ENDPOINT,
    CommandCodeBridge,
    CommandCodeBridgeError,
)
from vnext.host_contract import SessionStartRequest
from vnext.vnext_app_server import VNextAppServerAdapter
from vnext.vnext_mcp_server import DEFAULT_CATALOG, _validate_catalog
from vnext.vnext_provider_config import load_commandcode_provider
from vnext.vnext_orchestration import AgentRole
from vnext.vnext_runtime_effects import runtime_effect_reader
from vnext.vnext_runtime_projection import (
    next_native_cursor,
    project_native_event,
)
from vnext.vnext_scheduler import VNextScheduler
from vnext.vnext_session_runtime import PROVIDERS, VNextRuntimeSession


ACCOUNT_KEY = "cc-account-key-never-leaves-the-bridge"


class _Upstream:
    """Stand in for Command Code and record exactly what it was sent."""

    def __init__(self, *, status: int = 200, body: bytes = b"data: ok\n\n") -> None:
        self.status = status
        self.body = body
        self.requests: list[urllib.request.Request] = []
        self.payloads: list[dict] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None):
        del timeout
        self.requests.append(request)
        self.payloads.append(json.loads(request.data.decode("utf-8")))
        if self.status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, self.status, "refused", {"Content-Type": "application/json"},
                io.BytesIO(self.body),
            )
        answer = io.BytesIO(self.body)
        answer.status = self.status  # type: ignore[attr-defined]
        answer.headers = {"Content-Type": "text/event-stream"}  # type: ignore[attr-defined]
        return answer


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body


def _post(bridge: CommandCodeBridge, payload: object, *, token: str | None = None) -> _Response:
    """Send one request to the bridge the way the Codex CLI would."""

    body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
    bearer = bridge.bearer_token if token is None else token
    request = urllib.request.Request(
        f"{bridge.endpoint}/responses",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
    )
    try:
        answer = urllib.request.urlopen(request, timeout=10)
        return _Response(answer.status, answer.read())
    except urllib.error.HTTPError as exc:
        return _Response(exc.code, exc.read())


class BearerComparisonTests(unittest.TestCase):
    def test_bearer_comparison_uses_compare_digest(self) -> None:
        import vnext.vnext_commandcode as commandcode

        bridge = object.__new__(CommandCodeBridge)
        bridge._token = "expected-token"
        request = SimpleNamespace(headers={"Authorization": "Bearer guessed"})
        with patch.object(commandcode.secrets, "compare_digest", wraps=commandcode.secrets.compare_digest) as compare, \
             patch.object(bridge, "_reject") as reject:
            bridge._receive(request)
            request.headers = {}
            bridge._receive(request)
        compare.assert_any_call(b"Bearer guessed", b"Bearer expected-token")
        compare.assert_any_call(b"", b"Bearer expected-token")
        self.assertEqual(2, compare.call_count)
        self.assertEqual(2, reject.call_count)

    def test_a_non_ascii_credential_is_refused_without_an_exception(self) -> None:
        # http.server decodes headers as latin-1, and compare_digest raises
        # TypeError on a str holding non-ASCII characters.
        bridge = object.__new__(CommandCodeBridge)
        bridge._token = "expected-token"
        request = SimpleNamespace(headers={"Authorization": "Bearer \u00e9t\u00e9"})
        with patch.object(bridge, "_reject") as reject:
            bridge._receive(request)
        reject.assert_called_once_with(request, 403, "invalid Command Code bridge credential")


class CommandCodeBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = _Upstream()
        self.bridge = CommandCodeBridge(api_key=ACCOUNT_KEY)
        self.bridge.start()
        self.addCleanup(self.bridge.close)
        patcher = patch("vnext.vnext_commandcode.urlopen", self.upstream)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_tool_command_code_refuses_never_reaches_it(self) -> None:
        """This is the whole reason the bridge exists.

        Codex sends ``web_search`` for every unrecognised model slug and no
        configuration key removes it.  Command Code answers a request carrying
        one with ``Only type:"function" tools are supported``, so the turn dies
        before the model sees it.
        """

        answer = _post(self.bridge, {
            "model": "deepseek/deepseek-v4.1-flash",
            "tools": [
                {"type": "function", "name": "shell"},
                {"type": "web_search"},
                {"type": "namespace", "name": "multi_agent_v1"},
                {"type": "function", "name": "apply_patch"},
            ],
        })

        self.assertEqual(200, answer.status)
        sent = self.upstream.payloads[0]["tools"]
        self.assertEqual(["shell", "apply_patch"], [tool["name"] for tool in sent])
        self.assertEqual({"function"}, {tool["type"] for tool in sent})
        self.assertEqual(2, self.bridge.receipt().tools_dropped)

    def test_a_request_that_names_no_tools_is_forwarded_as_it_arrived(self) -> None:
        """An absent tool list and an empty one mean different things upstream."""

        _post(self.bridge, {"model": "deepseek/deepseek-v4.1-flash", "input": "hello"})

        self.assertNotIn("tools", self.upstream.payloads[0])
        self.assertEqual(0, self.bridge.receipt().tools_dropped)

    def test_a_turn_with_no_reasoning_effort_is_accepted(self) -> None:
        """The live endpoint refused this exact body on 2026-09-18.

        The CLI sends ``"reasoning": null`` for a turn that asks for none, and
        Command Code answers ``expected object, received null``.  Omitting the
        key asks for the same turn in a spelling it accepts.
        """

        _post(self.bridge, {
            "model": "deepseek/deepseek-v4.1-flash",
            "reasoning": None,
            "input": "hello",
        })

        sent = self.upstream.payloads[0]
        self.assertNotIn("reasoning", sent)
        self.assertEqual("hello", sent["input"])

    def test_a_reasoning_effort_that_was_asked_for_is_forwarded(self) -> None:
        """Only the null spelling is dropped; a real request survives."""

        _post(self.bridge, {
            "model": "deepseek/deepseek-v4.1-flash",
            "reasoning": {"effort": "low"},
        })

        self.assertEqual({"effort": "low"}, self.upstream.payloads[0]["reasoning"])

    def test_the_account_key_goes_upstream_and_the_codex_bearer_does_not(self) -> None:
        """The two credentials must never be the same value in either direction."""

        _post(self.bridge, {"model": "deepseek/deepseek-v4.1-flash"})

        sent = self.upstream.requests[0]
        self.assertEqual(f"Bearer {ACCOUNT_KEY}", sent.get_header("Authorization"))
        self.assertNotIn(self.bridge.bearer_token, str(sent.headers))
        self.assertNotEqual(ACCOUNT_KEY, self.bridge.bearer_token)

    def test_the_user_agent_cloudflare_accepts_is_the_one_sent(self) -> None:
        """Cloudflare answers an unrecognised client with error code 1010."""

        _post(self.bridge, {"model": "deepseek/deepseek-v4.1-flash"})

        self.assertEqual("command-code/1.0.0", self.upstream.requests[0].get_header("User-agent"))

    def test_a_caller_without_the_bearer_is_refused_and_nothing_is_forwarded(self) -> None:
        """Anything else on this machine can reach a loopback port."""

        answer = _post(self.bridge, {"model": "deepseek/deepseek-v4.1-flash"}, token="guessed")

        self.assertEqual(403, answer.status)
        self.assertEqual([], self.upstream.requests)
        self.assertEqual(0, self.bridge.receipt().requests_forwarded)

    def test_an_upstream_refusal_reaches_the_caller_with_its_own_status(self) -> None:
        """Codex acts on the provider's status, so the bridge must not mask it."""

        self.upstream.status = 429
        self.upstream.body = b'{"error":{"message":"slow down"}}'

        answer = _post(self.bridge, {"model": "deepseek/deepseek-v4.1-flash"})

        self.assertEqual(429, answer.status)
        self.assertIn(b"slow down", answer.body)
        self.assertEqual(1, self.bridge.receipt().upstream_failures)

    def test_a_body_that_is_not_json_is_refused_before_any_upstream_call(self) -> None:
        answer = _post(self.bridge, b"{not json")

        self.assertEqual(400, answer.status)
        self.assertEqual([], self.upstream.requests)

    def test_the_streamed_answer_arrives_whole(self) -> None:
        """The wire is server-sent events, so the body is relayed in chunks."""

        self.upstream.body = b"data: one\n\n" + b"data: two\n\n" * 4000

        answer = _post(self.bridge, {"model": "deepseek/deepseek-v4.1-flash"})

        self.assertEqual(self.upstream.body, answer.body)


class CommandCodeStreamFailureTests(unittest.TestCase):
    """R24: what a worker reads when the upstream stream breaks or limits it."""

    def setUp(self) -> None:
        self.bridge = CommandCodeBridge(api_key=ACCOUNT_KEY)
        self.bridge.start()
        self.addCleanup(self.bridge.close)

    def _raw_post(self) -> bytes:
        import socket
        from urllib.parse import urlsplit

        parts = urlsplit(self.bridge.endpoint)
        body = json.dumps({"model": "deepseek/deepseek-v4.1-flash"}).encode("utf-8")
        head = (
            f"POST /v1/responses HTTP/1.1\r\nHost: {parts.netloc}\r\n"
            f"Authorization: Bearer {self.bridge.bearer_token}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        received = b""
        with socket.create_connection((parts.hostname, parts.port), timeout=5) as connection:
            connection.sendall(head + body)
            try:
                while True:
                    piece = connection.recv(65536)
                    if not piece:
                        break
                    received += piece
            except socket.timeout:
                self.fail(f"the bridge left the connection open after: {received!r}")
        return received

    def test_a_stream_cut_midway_ends_the_connection_with_one_response(self) -> None:
        class Cut(io.RawIOBase):
            status = 200
            headers = {"Content-Type": "text/event-stream"}

            def __init__(self) -> None:
                self.sent = False

            def read(self, size: int = -1) -> bytes:
                if not self.sent:
                    self.sent = True
                    return b'data: {"delta":"hel"}\n\n'
                raise ConnectionResetError("upstream went away")

        with patch("vnext.vnext_commandcode.urlopen", lambda request, timeout=None: Cut()):
            received = self._raw_post()

        self.assertEqual(1, received.count(b"HTTP/1.1 "), received)
        self.assertIn(b'data: {"delta":"hel"}', received)
        self.assertFalse(received.endswith(b"0\r\n\r\n"), "a cut stream must not read as finished")
        self.assertEqual(1, self.bridge.receipt().upstream_failures)

    def test_a_rate_limit_keeps_its_back_off_headers(self) -> None:
        def limited(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests",
                {"Content-Type": "application/json", "Retry-After": "37",
                 "x-ratelimit-remaining-requests": "0", "Set-Cookie": "session=SECRET"},
                io.BytesIO(b'{"error":{"message":"rate limited"}}'),
            )

        with patch("vnext.vnext_commandcode.urlopen", limited):
            received = self._raw_post().lower()

        self.assertIn(b"http/1.1 429", received)
        self.assertIn(b"retry-after: 37", received)
        self.assertIn(b"x-ratelimit-remaining-requests: 0", received)
        self.assertNotIn(b"secret", received)


    def test_a_lowercase_content_type_still_reaches_the_worker(self) -> None:
        class Stream(io.RawIOBase):
            status = 200
            headers = {"content-type": "text/event-stream"}

            def read(self, size: int = -1) -> bytes:
                return b""

        with patch("vnext.vnext_commandcode.urlopen", lambda request, timeout=None: Stream()):
            received = self._raw_post()

        self.assertIn(b"Content-Type: text/event-stream", received)
        self.assertNotIn(b"application/json", received)

    def test_an_upstream_that_fails_to_close_gets_one_response(self) -> None:
        class Sticky(io.RawIOBase):
            status = 200
            headers = {"Content-Type": "application/json"}

            def __init__(self) -> None:
                self.sent = False

            def read(self, size: int = -1) -> bytes:
                if not self.sent:
                    self.sent = True
                    return b"{}"
                return b""

            def close(self) -> None:
                # Once: the object's own finaliser calls close again.
                if not getattr(self, "closed_once", False):
                    self.closed_once = True
                    raise OSError("TLS close failed")

        with patch("vnext.vnext_commandcode.urlopen", lambda request, timeout=None: Sticky()):
            received = self._raw_post()

        self.assertEqual(1, received.count(b"HTTP/1.1 "), received)
        self.assertTrue(received.endswith(b"0\r\n\r\n"), received)


class CommandCodeRedirectTests(unittest.TestCase):
    """The account key goes to the configured endpoint and nowhere else."""

    def test_a_redirect_is_refused_and_the_key_stays_home(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from vnext import vnext_commandcode

        reached: list[str] = []

        class Elsewhere(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                reached.append(self.headers.get("Authorization") or "")
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_GET

            def log_message(self, *args) -> None:
                pass

        other = ThreadingHTTPServer(("127.0.0.1", 0), Elsewhere)
        threading.Thread(target=other.serve_forever, daemon=True).start()
        self.addCleanup(other.server_close)
        self.addCleanup(other.shutdown)
        target = f"http://127.0.0.1:{other.server_address[1]}/collect"

        class Redirecting(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        first = ThreadingHTTPServer(("127.0.0.1", 0), Redirecting)
        threading.Thread(target=first.serve_forever, daemon=True).start()
        self.addCleanup(first.server_close)
        self.addCleanup(first.shutdown)

        request = urllib.request.Request(
            f"http://127.0.0.1:{first.server_address[1]}/v1/responses",
            data=b"{}", method="POST", headers={"authorization": "Bearer SECRET"},
        )
        with self.assertRaises(urllib.error.HTTPError) as refused:
            vnext_commandcode.urlopen(request, timeout=5)
        refused.exception.close()
        self.assertEqual(302, refused.exception.code)
        self.assertEqual([], reached)


class CommandCodeToolPairingTests(unittest.TestCase):
    """The endpoint wants every tool call answered before anything else speaks."""

    @staticmethod
    def _call(identifier: str, name: str = "shell") -> dict[str, object]:
        return {
            "type": "function_call",
            "call_id": identifier,
            "name": name,
            "arguments": '{"command":["ls"]}',
        }

    @staticmethod
    def _output(identifier: str) -> dict[str, object]:
        return {"type": "function_call_output", "call_id": identifier, "output": "ok"}

    @staticmethod
    def _hook(text: str = "[clock] 23:40 CEST") -> dict[str, object]:
        return {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": text}],
        }

    def _rewrite(self, items: list[object]) -> list[object]:
        body, _ = CommandCodeBridge.rewrite_request({"model": "x", "input": items})
        return body["input"]

    def test_the_hook_context_moves_to_just_after_the_tool_output(self) -> None:
        items = [self._hook("before"), self._call("c1"), self._hook(), self._output("c1")]

        self.assertEqual(
            [items[0], items[1], items[3], items[2]],
            self._rewrite(list(items)),
        )

    def test_parallel_calls_keep_their_outputs_and_the_gap_items_follow(self) -> None:
        calls = [self._call("c1"), self._call("c2")]
        outputs = [self._output("c1"), self._output("c2")]
        hook = self._hook()
        items = [calls[0], calls[1], outputs[0], hook, outputs[1]]

        self.assertEqual(
            [calls[0], calls[1], outputs[0], outputs[1], hook],
            self._rewrite(list(items)),
        )

    def test_a_repeated_call_id_waits_for_every_one_of_its_outputs(self) -> None:
        # Found by the cross-vendor review: with one slot per call_id the first
        # output closed the batch and left the second behind a hook message.
        first, second = self._call("x"), self._call("x")
        out_one, out_two = self._output("x"), self._output("x")
        hook_one, hook_two = self._hook("one"), self._hook("two")
        items = [first, hook_one, second, out_one, hook_two, out_two]

        self.assertEqual(
            [first, second, out_one, out_two, hook_one, hook_two],
            self._rewrite(list(items)),
        )

    def test_every_tool_call_type_codex_sends_is_paired(self) -> None:
        cases = [
            ("custom_tool_call", "custom_tool_call_output"),
            ("local_shell_call", "function_call_output"),
            ("tool_search_call", "tool_search_output"),
        ]
        for call_type, output_type in cases:
            with self.subTest(call_type):
                call = {"type": call_type, "call_id": "c1"}
                output = {"type": output_type, "call_id": "c1"}
                hook = self._hook()

                self.assertEqual(
                    [call, output, hook],
                    self._rewrite([call, hook, output]),
                )

    def test_a_call_with_no_output_is_left_where_it_stands(self) -> None:
        items = [self._call("c1"), self._hook(), self._call("c2"), self._output("c2")]

        self.assertEqual(items, self._rewrite(list(items)))

    def test_a_thread_with_no_calls_and_a_request_with_no_input_pass_unchanged(self) -> None:
        plain = [self._hook("one"), self._hook("two")]

        self.assertEqual(plain, self._rewrite(list(plain)))
        body, _ = CommandCodeBridge.rewrite_request({"model": "x"})
        self.assertNotIn("input", body)

    def test_no_item_is_dropped_or_edited(self) -> None:
        items = [self._call("c1"), self._hook(), self._output("c1"), self._hook("after")]
        original = json.dumps(items, sort_keys=True)

        rewritten = self._rewrite(list(items))

        self.assertEqual(len(items), len(rewritten))
        self.assertEqual(
            sorted(json.dumps(item, sort_keys=True) for item in items),
            sorted(json.dumps(item, sort_keys=True) for item in rewritten),
        )
        self.assertEqual(original, json.dumps(items, sort_keys=True))


class CommandCodeBridgeConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = CommandCodeBridge(api_key=ACCOUNT_KEY)
        self.addCleanup(self.bridge.close)

    def test_codex_is_pointed_at_the_bridge_and_handed_the_throwaway_bearer(self) -> None:
        overrides = self.bridge.config_overrides("commandcode")

        self.assertEqual(self.bridge.endpoint, overrides["model_providers.commandcode.base_url"])
        self.assertEqual("responses", overrides["model_providers.commandcode.wire_api"])
        self.assertIs(False, overrides["model_providers.commandcode.requires_openai_auth"])
        self.assertEqual("commandcode", overrides["model_provider"])
        self.assertEqual(
            f"Bearer {self.bridge.bearer_token}",
            overrides["model_providers.commandcode.http_headers.Authorization"],
        )

    def test_the_account_key_is_absent_from_everything_codex_is_given(self) -> None:
        """Codex writes its configuration to disk and echoes it when it fails."""

        rendered = json.dumps(self.bridge.config_overrides("commandcode"))

        self.assertNotIn(ACCOUNT_KEY, rendered)
        self.assertNotIn(ACCOUNT_KEY, repr(self.bridge))
        self.assertNotIn(ACCOUNT_KEY, self.bridge.endpoint)

    def test_the_bridge_binds_loopback_only(self) -> None:
        with self.assertRaises(ValueError):
            CommandCodeBridge(api_key=ACCOUNT_KEY, host="0.0.0.0")

    def test_the_bridge_forwards_over_https_only(self) -> None:
        with self.assertRaises(ValueError):
            CommandCodeBridge(api_key=ACCOUNT_KEY, upstream="http://api.commandcode.ai/provider/v1")

    def test_an_unusable_key_is_refused_at_construction(self) -> None:
        for key in ("", "   ", " padded ", None):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    CommandCodeBridge(api_key=key)  # type: ignore[arg-type]

    def test_a_closed_bridge_will_not_start_again(self) -> None:
        self.bridge.close()

        with self.assertRaises(CommandCodeBridgeError):
            self.bridge.start()

    def test_the_default_upstream_is_the_command_code_endpoint(self) -> None:
        self.assertEqual("https://api.commandcode.ai/provider/v1", COMMANDCODE_ENDPOINT)


class CommandCodeDrainDeadlineTests(unittest.TestCase):
    """A close cannot wait forever on a request the upstream never answers.

    ``close`` drained with ``while self._active_requests: self._condition.wait()``
    and no deadline.  One forwarded request left unanswered -- Command Code
    behind a hung TLS handshake is enough -- held the quit there for good, and
    the restart proxy gives the whole process about five seconds to stop.
    """

    def test_a_request_that_never_answers_cannot_hold_the_close(self) -> None:
        upstream_entered = threading.Event()
        release_upstream = threading.Event()

        def hanging_upstream(request, timeout=None):  # noqa: ANN001 - stdlib shape
            upstream_entered.set()
            release_upstream.wait(30)
            raise urllib.error.URLError("released")

        bridge = CommandCodeBridge(api_key=ACCOUNT_KEY)
        bridge.start()
        self.addCleanup(release_upstream.set)
        with patch("vnext.vnext_commandcode.urlopen", hanging_upstream):
            caller = threading.Thread(
                target=_post, args=(bridge, {"model": "deepseek/deepseek-v4.1-flash"}), daemon=True
            )
            caller.start()
            self.assertTrue(upstream_entered.wait(10), "the bridge never forwarded the request")

            started = time.monotonic()
            bridge.close(drain=0.2)
            spent = time.monotonic() - started

        # Well inside the five-second grace the restart proxy allows, measured
        # against the pinned reloaderoo 1.1.5 on 2026-09-30.
        self.assertLess(spent, 3.0, f"the close took {spent:.1f}s")
        self.assertEqual(1, bridge.receipt().abandoned_requests)
        release_upstream.set()
        caller.join(timeout=10)

    def test_a_bridge_with_nothing_in_flight_reports_no_abandoned_request(self) -> None:
        """A clean drain must not look like a forced one."""

        bridge = CommandCodeBridge(api_key=ACCOUNT_KEY)
        bridge.start()

        bridge.close(drain=0.2)

        self.assertEqual(0, bridge.receipt().abandoned_requests)


class CommandCodeCredentialTests(unittest.TestCase):
    """The key is read from the file vNext owns, and from nowhere else."""

    def _write(self, payload: object) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        target = Path(directory.name) / "providers.json"
        target.write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
        return target

    def test_a_configured_key_is_returned(self) -> None:
        path = self._write({"providers": {"commandcode": {"api_key": ACCOUNT_KEY}}})

        config = load_commandcode_provider(path)

        self.assertIsNotNone(config)
        self.assertEqual(ACCOUNT_KEY, config.api_key)

    def test_the_key_stays_out_of_the_configuration_repr(self) -> None:
        path = self._write({"providers": {"commandcode": {"api_key": ACCOUNT_KEY}}})

        self.assertNotIn(ACCOUNT_KEY, repr(load_commandcode_provider(path)))

    def test_a_zai_only_file_offers_no_command_code_key(self) -> None:
        """Each provider answers for itself; one configured key is not two."""

        path = self._write({"providers": {"zai": {"api_key": "zai-key"}}})

        self.assertIsNone(load_commandcode_provider(path))

    def test_an_unusable_entry_reads_as_no_key_at_all(self) -> None:
        cases = {
            "missing file": Path("does-not-exist.json"),
            "not json": self._write("{"),
            "empty key": self._write({"providers": {"commandcode": {"api_key": ""}}}),
            "untrimmed key": self._write({"providers": {"commandcode": {"api_key": " k "}}}),
            "key is not a string": self._write({"providers": {"commandcode": {"api_key": 7}}}),
            "no providers block": self._write({"commandcode": {"api_key": ACCOUNT_KEY}}),
        }
        for label, path in cases.items():
            with self.subTest(label):
                self.assertIsNone(load_commandcode_provider(path))


class CommandCodeCatalogTests(unittest.TestCase):
    """A card a manager can delegate to is a card the session can actually run."""

    def test_the_flash_card_is_offered_when_the_key_is_configured(self) -> None:
        with patch("vnext.vnext_mcp_server.load_commandcode_provider",
                   return_value=object()):
            entries = _validate_catalog(DEFAULT_CATALOG)

        self.assertIn(
            {"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"}, entries
        )

    def test_the_card_is_withdrawn_when_no_key_is_configured(self) -> None:
        """A model that can only fail after delegation is worse than no model.

        The Codex login is stubbed because an empty roster is refused
        outright: on a machine with no credentials at all this case died on
        "a workforce needs at least one available worker model" before it
        could look at the Command Code card it is about.
        """

        with patch("vnext.vnext_mcp_server.load_commandcode_provider", return_value=None), \
                patch("vnext.vnext_mcp_server._codex_login_available", return_value=True):
            entries = _validate_catalog(DEFAULT_CATALOG)

        self.assertEqual([], [e for e in entries if e.get("provider") == "commandcode"])

    def test_the_pinned_codex_allowlist_leaves_a_reseller_card_alone(self) -> None:
        """The allowlist names models the CLI has built in.  This is not one."""

        with patch("vnext.vnext_mcp_server.load_commandcode_provider",
                   return_value=object()):
            entries = _validate_catalog(
                [{"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"}]
            )

        self.assertEqual(1, len(entries))

    def test_the_provider_runs_the_codex_harness_with_its_own_credential(self) -> None:
        harness, credential = PROVIDERS["commandcode"]

        self.assertEqual("app-server", harness)
        self.assertIn("providers.json", credential)


class CommandCodeReviewerTests(unittest.TestCase):
    """Who answers a Command Code child's approval requests.

    ``auto_review`` runs a model named ``codex-auto-review``.  Command Code
    does not serve it, so the runtime refuses every write and the child reports
    a blocker nobody can clear -- measured live on 2026-09-18, three turns and
    no file written.  The same turn with the manager as reviewer wrote the file
    on the first attempt.
    """

    @staticmethod
    def _reviewer(role: AgentRole, provider: str) -> str:
        card = SimpleNamespace(provider=provider)
        scheduler = SimpleNamespace(
            managed=SimpleNamespace(
                control=SimpleNamespace(registry=SimpleNamespace(cards={"m": card}))
            ),
            _REVIEWS_THROUGH_THE_MANAGER=VNextScheduler._REVIEWS_THROUGH_THE_MANAGER,
        )
        agent = SimpleNamespace(role=role, model_id="m")
        return VNextScheduler._reviewer_for(scheduler, agent)

    def test_a_command_code_manager_sends_its_approvals_to_the_manager_above(self) -> None:
        self.assertEqual(
            "user", self._reviewer(AgentRole.BRANCH_MANAGER, "commandcode")
        )

    def test_a_vnext_keeps_the_reviewer_it_has_always_had(self) -> None:
        self.assertEqual("auto_review", self._reviewer(AgentRole.BRANCH_MANAGER, "codex"))

    def test_every_worker_sends_its_approvals_to_its_manager(self) -> None:
        for provider in ("codex", "commandcode", "claude", "zai"):
            with self.subTest(provider=provider):
                self.assertEqual("user", self._reviewer(AgentRole.WORKER, provider))


class CommandCodeCleanupTests(unittest.TestCase):
    """What survives a bridge that will not close.

    Found by a verifier, 2026-09-18: the first version cleared both references
    before closing anything, so a close that raised left the temporary Codex
    home on disk and handed the retry path nothing to close.
    """

    def _session(self, bridge: object, home: Path) -> object:
        session = VNextRuntimeSession.__new__(VNextRuntimeSession)
        session._lock = threading.RLock()
        session._commandcode_bridge = bridge
        session._commandcode_home = home
        return session

    def test_a_close_that_raises_still_removes_the_temporary_home(self) -> None:
        class Stubborn:
            def close(self) -> None:
                raise RuntimeError("the bridge will not close")

        with TemporaryDirectory() as parent:
            home = Path(parent) / "vnext-commandcode-home"
            home.mkdir()
            session = self._session(Stubborn(), home)

            with self.assertRaises(RuntimeError):
                VNextRuntimeSession._close_commandcode_bridge(session)

            self.assertFalse(home.exists())
            self.assertIsNone(session._commandcode_home)

    def test_a_bridge_that_refused_to_close_is_still_there_to_retry(self) -> None:
        class Stubborn:
            def __init__(self) -> None:
                self.attempts = 0

            def close(self) -> None:
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("the bridge will not close")

        bridge = Stubborn()
        with TemporaryDirectory() as parent:
            home = Path(parent) / "vnext-commandcode-home"
            home.mkdir()
            session = self._session(bridge, home)
            with self.assertRaises(RuntimeError):
                VNextRuntimeSession._close_commandcode_bridge(session)

            self.assertIs(bridge, session._commandcode_bridge)
            VNextRuntimeSession._close_commandcode_bridge(session)

        self.assertEqual(2, bridge.attempts)
        self.assertIsNone(session._commandcode_bridge)

    def test_a_replacement_bridge_survives_the_close_of_the_one_it_replaced(self) -> None:
        """A retry that starts a second bridge keeps it.

        Cleanup drains without the lock, so a provider start can install a new
        bridge while the old one is still shutting down. Clearing the field
        unconditionally at the end would drop the live bridge and leave the
        session holding a loopback server nothing can stop.
        """

        replacement = object()

        class Replaced:
            def __init__(self, session_box: list) -> None:
                self._session_box = session_box

            def close(self) -> None:
                self._session_box[0]._commandcode_bridge = replacement

        box: list = []
        with TemporaryDirectory() as parent:
            home = Path(parent) / "vnext-commandcode-home"
            home.mkdir()
            session = self._session(Replaced(box), home)
            box.append(session)

            VNextRuntimeSession._close_commandcode_bridge(session)

        self.assertIs(replacement, session._commandcode_bridge)


class CommandCodeRuntimeObservationTests(unittest.TestCase):
    """What the session makes of a Command Code worker's events.

    The provider was added to the catalog and to the runtime without being
    added to either place that decodes what a running agent does.  Binding a
    reader failed the agent before its first turn, so the whole team it
    belonged to never started; projection would then have dropped its output
    and its token usage in silence.

    Found on 2026-09-22: the adapter the runtime builds still called itself
    ``codex`` while the agent's turns were filed under ``commandcode``, so
    every native event looked for a turn that did not exist and was parked.
    A live DeepSeek run finished with no usage at all.  The projection test
    below passed throughout, because it names the provider itself.
    """

    def test_the_runtime_built_command_code_adapter_reports_its_provider(self) -> None:
        class Bridge:
            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def config_overrides(self, provider: str) -> dict[str, str]:
                self.provider = provider
                return {"provider": provider}

        class Idle:
            """A transport that stays open and says nothing, so no process starts."""

            def __init__(self) -> None:
                self._closed = threading.Event()

            @property
            def closed(self) -> bool:
                return self._closed.is_set()

            def recv(self) -> None:
                self._closed.wait()
                return None

            def send(self, payload: str) -> None:
                pass

            def close(self) -> None:
                self._closed.set()

        def without_a_process(**kwargs: object) -> VNextAppServerAdapter:
            # The real constructor starts Codex, which a stub executable cannot
            # stand in for on Windows.  Everything the factory passed is kept.
            return VNextAppServerAdapter(**kwargs, transport=Idle())

        with TemporaryDirectory() as workspace, TemporaryDirectory() as home:
            request = SessionStartRequest(
                session_id="commandcode-session",
                workspace=workspace,
                primary_agent_id="primary",
                primary={"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"},
                catalog_config={"models": [{
                    "provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash",
                }]},
            )
            runtime = VNextRuntimeSession(request, lambda _event: None)
            runtime._commandcode_home = Path(home)
            with patch("vnext.vnext_commandcode.CommandCodeBridge", return_value=Bridge()), \
                 patch("vnext.vnext_provider_config.load_commandcode_provider",
                       return_value=SimpleNamespace(api_key=ACCOUNT_KEY)), \
                 patch("vnext.live_runtime.resolve_session_runtime",
                       return_value=(Path(sys.executable), {"source": "test"})), \
                 patch("vnext.vnext_app_server.VNextAppServerAdapter", without_a_process):
                adapter = runtime._commandcode_adapter(runtime.root)

            self.assertIsInstance(adapter, VNextAppServerAdapter)
            self.assertEqual("commandcode", adapter.provider)
            self.assertEqual("commandcode", adapter.mcp_config_overrides.get("provider"))
            adapter.close()
            runtime._close_commandcode_bridge()

    def test_a_worker_on_this_provider_can_have_its_effects_read(self) -> None:
        reader = runtime_effect_reader("commandcode")

        self.assertEqual("commandcode", reader.provider)

    def test_effects_are_decoded_from_the_codex_app_server_shape(self) -> None:
        candidate = runtime_effect_reader("commandcode").completed_effect(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-ref",
                    "turnId": "turn-ref",
                    "item": {
                        "id": "item-ref",
                        "type": "commandExecution",
                        "cwd": ".",
                        "commandActions": [],
                        "status": "completed",
                    },
                },
            }
        )

        self.assertIsNotNone(candidate)
        self.assertEqual("item-ref", candidate.item_ref)
        self.assertEqual("command", candidate.item["type"])

    def test_the_cursor_counts_events_the_way_the_app_server_does(self) -> None:
        self.assertEqual(7, next_native_cursor("commandcode", 5, [{}, {}]))

    def test_output_and_token_usage_reach_the_manager(self) -> None:
        projected = project_native_event(
            "commandcode",
            "agent",
            {
                "method": "item/completed",
                "params": {
                    "turnId": "turn",
                    "item": {"id": "item", "type": "agentMessage", "text": "projected"},
                },
            },
        )
        content = next(item for item in projected if item.type == "content.final")

        self.assertEqual("projected", content.payload["blocks"][0]["text"])

        usage = project_native_event(
            "commandcode",
            "agent",
            {
                "method": "thread/tokenUsage/updated",
                "params": {"turnId": "turn", "tokenUsage": {"total": 1234}},
            },
        )
        reported = next(item for item in usage if item.type == "usage.updated")

        self.assertEqual(1234, reported.payload["tokens"])
        self.assertEqual("commandcode", reported.payload["provider"])


class CommandCodeFailureStderrTests(unittest.TestCase):
    """A bind that fails must not write the relay bearer into the run log.

    The Codex child carries ``Authorization: Bearer <loopback token>`` on its
    own command line, so a child that echoes its configuration while failing
    puts that token on stderr.  The run log is the file a user attaches to a
    bug report, and the bearer authorizes the loopback relay that proxies to
    Command Code with the real account key.
    """

    def _service(self, workspace: Path, log: Path):
        from vnext.vnext_mcp_server import VNextMcpService

        return VNextMcpService(
            workspace=workspace,
            catalog=[{"provider": "codex", "model": "gpt-6-sol"}],
            event_log=log,
            status_file=workspace / "status.json",
            outcome_log=workspace / "outcomes.jsonl",
            adapter_factories={"codex": lambda: None},
        )

    def _report(self, provider: str, line: str) -> list[dict]:
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            log = workspace / "run.jsonl"
            service = self._service(workspace, log)
            try:
                scheduler = service.session._prepare_scheduler()
                adapter = SimpleNamespace(
                    provider=provider, captured_stderr=lambda: (line,)
                )
                scheduler._report_binding_failure(
                    service.session.root, adapter, RuntimeError("bind failed")
                )
            finally:
                service.close()
            raw = log.read_text(encoding="utf-8")
            rows = [json.loads(entry) for entry in raw.splitlines()]
            self.assertNotIn(line, raw)
            return [row for row in rows if row.get("type") == "provider.error"]

    def test_commandcode_failure_stderr_is_absent_from_run_log(self) -> None:
        token = "cc-relay-bearer-must-not-be-persisted"
        rows = self._report("commandcode", f"Authorization: Bearer {token}")

        self.assertEqual(1, len(rows))
        self.assertEqual([], rows[0]["payload"]["provider_stderr"])

    def test_zai_failure_stderr_is_absent_from_run_log(self) -> None:
        rows = self._report("zai", "authorization failed for zai-key-abcdef")

        self.assertEqual(1, len(rows))
        self.assertEqual([], rows[0]["payload"]["provider_stderr"])

    def test_a_codex_failure_still_records_its_stderr(self) -> None:
        """Suppression is for the two credential-adjacent children alone."""

        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            log = workspace / "run.jsonl"
            service = self._service(workspace, log)
            try:
                scheduler = service.session._prepare_scheduler()
                adapter = SimpleNamespace(
                    provider="codex", captured_stderr=lambda: ("ImportError: pydantic",)
                )
                scheduler._report_binding_failure(
                    service.session.root, adapter, RuntimeError("bind failed")
                )
            finally:
                service.close()
            rows = [
                json.loads(entry)
                for entry in log.read_text(encoding="utf-8").splitlines()
            ]
            errors = [row for row in rows if row.get("type") == "provider.error"]

        self.assertEqual(1, len(errors))
        self.assertEqual(
            ["ImportError: pydantic"], errors[0]["payload"]["provider_stderr"]
        )


class CommandCodeFatalMessageTests(unittest.TestCase):
    """A fatal message becomes an exception string, which the run log keeps.

    Suppressing ``provider_stderr`` is not enough on its own: the app-server
    adapter used to paste the child's stderr into the fatal message it raises
    with, and the scheduler persists ``str(exc)``.  That is a second route to
    the same file for the same bearer.
    """

    def _adapter(self, provider: str, lines: tuple[str, ...]):
        from vnext.vnext_app_server import VNextAppServerAdapter

        adapter = VNextAppServerAdapter.__new__(VNextAppServerAdapter)
        adapter.provider = provider
        adapter.codex_home = Path("/codex-home")
        adapter.workspace = Path("/workspace")
        adapter._stderr_lines = list(lines)
        return adapter

    def test_a_command_code_child_s_stderr_is_withheld_and_said_so(self) -> None:
        token = "cc-relay-bearer-must-not-reach-a-fatal-message"
        adapter = self._adapter("commandcode", (f"Authorization: Bearer {token}",))

        reported = adapter._reportable_stderr()

        self.assertNotIn(token, reported)
        self.assertIn("withheld", reported)
        self.assertIn("commandcode", reported)

    def test_a_codex_child_s_stderr_is_still_reported(self) -> None:
        adapter = self._adapter("codex", ("ImportError: pydantic", "exit 1"))

        reported = adapter._reportable_stderr()

        self.assertIn("ImportError: pydantic", reported)
        self.assertIn("exit 1", reported)


if __name__ == "__main__":
    unittest.main()
