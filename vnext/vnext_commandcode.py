"""A loopback bridge that lets the Codex runtime reach a Command Code model.

Command Code resells a large model catalog behind one OpenAI-compatible
endpoint, and its ``/responses`` route speaks the same wire the Codex CLI
speaks.  That is most of a provider for free: point Codex at the endpoint with
a ``model_providers`` entry and it runs a turn against DeepSeek.

Two measured facts, both 2026-09-18, are why a bridge stands in the middle
instead of Codex talking to Command Code directly.

The endpoint accepts only ``type: "function"`` tools.  Codex sends a
``namespace`` entry for every MCP server, plugin and built-in app it has, plus
one ``web_search``.  An isolated Codex home and the feature switches vNext
already passes remove the namespaces.  ``web_search`` survives all of them:
Command Code's slugs are unknown to Codex, so it falls back to default model
metadata, and that metadata names a web-search tool no user-facing key
overrides.  One request rewrite settles what no amount of configuration will.

Cloudflare sits in front of the endpoint and answers an unrecognised client
with ``error code: 1010``.  The Codex CLI's own user agent passes; a bare
stdlib one does not.  The bridge sends the agent Command Code expects.

A third difference showed up on 2026-09-24, once vNext started giving
Codex-harness workers a PostToolUse hook.  Codex records the hook's context as
a message item and places it between the tool call and the tool output, seven
milliseconds ahead of the output.  Command Code reads the thread as chat
completions, where an assistant message carrying ``tool_calls`` has to be
followed straight away by the tool replies, and answers HTTP 400:
``insufficient tool messages following tool_calls message``.  The bridge moves
whatever sat in the gap to just after the outputs.  The turn keeps every item
and every word of it; only the order changes.

The credential is the reason this is a service and not a config block.  The
account key stays in this process.  Codex is handed a bearer generated for the
session, which is worth nothing anywhere else and dies with the bridge.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .http_rejection import discard_rejected_body


class _NoRedirect(HTTPRedirectHandler):
    """Refuse every redirect on the credentialed request.

    urllib's default handler follows a 302 to any host and copies the
    Authorization header with it, so a redirect from the configured endpoint
    sent the Command Code account key wherever it pointed.  The worker gets
    the redirect status instead, and the key goes nowhere new.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = build_opener(_NoRedirect)


def urlopen(request: Request, timeout: float | None = None):
    return _OPENER.open(request, timeout=timeout)


COMMANDCODE_ENDPOINT = "https://api.commandcode.ai/provider/v1"

# Cloudflare rejects a client it does not recognise before the API sees it.
_CLIENT_USER_AGENT = "command-code/1.0.0"

# How long a close lets live requests finish before it takes the bridge down
# anyway.  The restart proxy allows the whole server about five seconds to stop
# (measured against the pinned reloaderoo 1.1.5 on 2026-09-30), and the bridge
# is one step of a close with several, so it gets a slice rather than all of it.
CLOSE_DRAIN_SECONDS = 2.0

# Optional fields the CLI spells as null and the endpoint will only accept as
# an object or not at all.
_OMITTED_WHEN_NULL = frozenset({"reasoning", "text", "truncation"})

# The item types Codex 0.156.0 puts on the wire for a tool call and its reply.
# Read off the ``ResponseItem`` variants the core crate serialises: FunctionCall,
# CustomToolCall, LocalShellCall and ToolSearchCall on one side, their outputs on
# the other.  A local shell call is answered with a ``function_call_output``.
_TOOL_CALL_TYPES = frozenset({
    "function_call",
    "custom_tool_call",
    "local_shell_call",
    "tool_search_call",
})
_TOOL_OUTPUT_TYPES = frozenset({
    "function_call_output",
    "custom_tool_call_output",
    "tool_search_output",
})

_MAX_REQUEST_BYTES = 8_388_608
_STREAM_CHUNK_BYTES = 8_192


class CommandCodeBridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandCodeReceipt:
    """What the bridge did, in shapes and counts that carry no content."""

    requests_forwarded: int
    tools_dropped: int
    upstream_failures: int
    rejected_requests: int
    # Requests still in flight when the close deadline ran out.  Any count
    # above zero says the bridge was cut off rather than drained, which is the
    # difference between a clean quit and one that dropped a worker's turn.
    abandoned_requests: int = 0


class CommandCodeBridge:
    """One loopback endpoint that makes Command Code answer a Codex turn."""

    def __init__(
        self,
        *,
        api_key: str,
        upstream: str = COMMANDCODE_ENDPOINT,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("the Command Code bridge must bind a loopback host")
        if not isinstance(api_key, str) or not api_key or api_key.strip() != api_key:
            raise ValueError("the Command Code bridge needs a trimmed, nonempty API key")
        if not isinstance(upstream, str) or not upstream.startswith("https://"):
            raise ValueError("the Command Code bridge forwards over HTTPS only")
        self._api_key = api_key
        self._upstream = upstream.rstrip("/")
        self._token = secrets.token_urlsafe(32)
        self._condition = threading.Condition(threading.RLock())
        self._accepting = True
        self._closed = False
        self._active_requests = 0
        self._abandoned_requests = 0
        self._requests_forwarded = 0
        self._tools_dropped = 0
        self._upstream_failures = 0
        self._rejected_requests = 0
        parent = self

        class Receiver(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib entrypoint
                parent._receive(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Receiver)
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        """Keep the key out of a traceback, a log line and a debugger."""

        return f"<CommandCodeBridge upstream={self._upstream!r}>"

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def bearer_token(self) -> str:
        """The throwaway bearer Codex presents.  The account key is never this."""

        return self._token

    def config_overrides(self, name: str = "commandcode") -> dict[str, Any]:
        """The Codex configuration that routes a thread through this bridge.

        Codex reads a provider credential from either the environment or these
        headers.  The header carries it, because the app-server child's
        environment is built from an allowlist and nothing here is worth
        widening that allowlist for.
        """

        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("a model provider name is required")
        return {
            f"model_providers.{name}.name": "Command Code",
            f"model_providers.{name}.base_url": self.endpoint,
            f"model_providers.{name}.wire_api": "responses",
            f"model_providers.{name}.requires_openai_auth": False,
            f"model_providers.{name}.http_headers.Authorization": f"Bearer {self._token}",
            "model_provider": name,
        }

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise CommandCodeBridgeError("the Command Code bridge is closed")
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="vnext-commandcode-bridge",
                daemon=True,
            )
            self._thread.start()

    def close(self, drain: float = CLOSE_DRAIN_SECONDS) -> None:
        """Stop accepting, let live requests drain, then close either way.

        The drain had no deadline: ``while self._active_requests`` waited on a
        forwarded request that an unreachable upstream may never answer, and a
        quit stopped there for good.  The wait is bounded now, and a forced
        close puts the count of abandoned requests on the receipt so a caller
        can tell a drained bridge from a cut-off one.
        """

        with self._condition:
            if self._closed:
                return
            self._accepting = False
            deadline = time.monotonic() + max(0.0, float(drain))
            while self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._abandoned_requests = self._active_requests
                    break
                self._condition.wait(remaining)
            self._closed = True
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def receipt(self) -> CommandCodeReceipt:
        with self._condition:
            return CommandCodeReceipt(
                requests_forwarded=self._requests_forwarded,
                tools_dropped=self._tools_dropped,
                upstream_failures=self._upstream_failures,
                rejected_requests=self._rejected_requests,
                abandoned_requests=self._abandoned_requests,
            )

    @classmethod
    def rewrite_request(cls, payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        """Put a Codex request into the shape Command Code accepts.

        Three differences showed up against the live endpoint, all of them the
        Codex CLI spelling something one way and the endpoint reading it
        another: a null optional, a tool type outside the one the endpoint
        takes, and an item sitting between a tool call and its output.  None of
        them changes what the request asks for.

        Returns the rewritten body and how many tools were dropped.
        """

        body = cls._drop_null_optionals(payload)
        if isinstance(body.get("input"), list):
            body["input"] = cls._pair_tool_calls(body["input"])
        tools = body.get("tools")
        if not isinstance(tools, list):
            return body, 0
        kept = [
            tool for tool in tools
            if isinstance(tool, Mapping) and tool.get("type") == "function"
        ]
        body["tools"] = kept
        return body, len(tools) - len(kept)

    @staticmethod
    def _pair_tool_calls(items: list[Any]) -> list[Any]:
        """Close the gap between a tool call and the output that answers it.

        Every call that has an output later in the thread is emitted with its
        outputs behind it, matched on ``call_id``.  Whatever sat in between —
        the hook's context message, or anything else of any role — follows the
        last of those outputs in the order it arrived.  A call with no output
        stays where it is, and no item is dropped, copied or edited.
        """

        def kind(item: object) -> str:
            return item.get("type") if isinstance(item, Mapping) else None

        def call_id(item: object) -> str | None:
            value = item.get("call_id") if isinstance(item, Mapping) else None
            return value if isinstance(value, str) else None

        answered: dict[int, str] = {}
        outstanding: dict[str, list[int]] = {}
        for index in range(len(items) - 1, -1, -1):
            identifier = call_id(items[index])
            if identifier is None:
                continue
            if kind(items[index]) in _TOOL_OUTPUT_TYPES:
                outstanding.setdefault(identifier, []).append(index)
            elif kind(items[index]) in _TOOL_CALL_TYPES and outstanding.get(identifier):
                outstanding[identifier].pop()
                answered[index] = identifier

        paired: list[Any] = []
        index = 0
        while index < len(items):
            if index not in answered:
                paired.append(items[index])
                index += 1
                continue
            calls: list[Any] = []
            outputs: list[Any] = []
            displaced: list[Any] = []
            # A count per call_id: a thread may repeat one, and each call
            # waits for its own output.
            pending: dict[str, int] = {}
            while index < len(items):
                item = items[index]
                identifier = call_id(item)
                if index in answered:
                    calls.append(item)
                    pending[answered[index]] = pending.get(answered[index], 0) + 1
                elif kind(item) in _TOOL_OUTPUT_TYPES and pending.get(identifier):
                    outputs.append(item)
                    pending[identifier] -= 1
                    if not pending[identifier]:
                        del pending[identifier]
                else:
                    displaced.append(item)
                index += 1
                if not pending:
                    break
            paired.extend(calls + outputs + displaced)
        return paired

    @staticmethod
    def _drop_null_optionals(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Say "no value" by omission, which is how the endpoint reads it.

        A turn with no reasoning effort goes out as ``"reasoning": null`` and
        the endpoint answers ``expected object, received null``.  Omitting the
        key asks for exactly the same turn and is accepted.
        """

        return {
            key: value for key, value in payload.items()
            if not (key in _OMITTED_WHEN_NULL and value is None)
        }

    def _receive(self, request: BaseHTTPRequestHandler) -> None:
        if not secrets.compare_digest(
            request.headers.get("Authorization", "").encode(), f"Bearer {self._token}".encode()
        ):
            self._reject(request, 403, "invalid Command Code bridge credential")
            return
        with self._condition:
            if not self._accepting:
                self._reject(request, 503, "the Command Code bridge is closing")
                return
            self._active_requests += 1
        try:
            body = self._read_body(request)
            forwarded, dropped = self._rewrite(body)
            with self._condition:
                self._tools_dropped += dropped
            self._forward(request, forwarded)
        except ValueError as exc:
            with self._condition:
                self._rejected_requests += 1
            self._respond_json(request, 400, {"error": {"message": str(exc)}})
        except Exception as exc:
            with self._condition:
                self._upstream_failures += 1
            self._respond_json(request, 502, {
                "error": {"message": f"the Command Code bridge failed: {type(exc).__name__}"},
            })
        finally:
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()

    @staticmethod
    def _read_body(request: BaseHTTPRequestHandler) -> bytes:
        length = int(request.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > _MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        return request.rfile.read(length)

    def _rewrite(self, body: bytes) -> tuple[bytes, int]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("request body must be an object")
        rewritten, dropped = self.rewrite_request(payload)
        return json.dumps(rewritten, ensure_ascii=False).encode("utf-8"), dropped

    def _forward(self, request: BaseHTTPRequestHandler, body: bytes) -> None:
        path = request.path.split("/v1", 1)[-1] if "/v1" in request.path else request.path
        upstream_request = Request(
            f"{self._upstream}{path}",
            data=body,
            method="POST",
            headers={
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
                "accept": request.headers.get("Accept", "text/event-stream"),
                "user-agent": _CLIENT_USER_AGENT,
            },
        )
        try:
            upstream = urlopen(upstream_request, timeout=600)
            status, headers = upstream.status, dict(upstream.headers)
        except HTTPError as exc:
            upstream, status, headers = exc, exc.code, dict(exc.headers)
            with self._condition:
                self._upstream_failures += 1
        with self._condition:
            self._requests_forwarded += 1
        request.send_response(status)
        # Header names are case-insensitive and a plain dict is not: an
        # upstream "content-type: text/event-stream" was relabelled JSON.
        content_type = next(
            (value for name, value in headers.items() if name.lower() == "content-type"),
            "application/json",
        )
        request.send_header("Content-Type", content_type)
        # A worker told it is rate-limited needs the interval the provider
        # asked for, so the back-off headers travel with the status.
        for name, value in headers.items():
            lowered = name.lower()
            if lowered == "retry-after" or lowered.startswith(("x-ratelimit-", "ratelimit-")):
                request.send_header(name, value)
        request.send_header("Transfer-Encoding", "chunked")
        request.end_headers()
        # The status line is on the wire now, so a failure past this point can
        # no longer become a 502.  The connection closes with the body
        # unterminated, which is how a chunked reader learns the answer broke.
        try:
            while True:
                chunk = upstream.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                request.wfile.write(format(len(chunk), "x").encode("ascii") + b"\r\n" + chunk + b"\r\n")
                request.wfile.flush()
            request.wfile.write(b"0\r\n\r\n")
        except Exception:
            request.close_connection = True
            with self._condition:
                self._upstream_failures += 1
        finally:
            # The response is already on the wire, so a close that fails is a
            # failure of that response; raised from here it became a second one.
            try:
                upstream.close()
            except Exception:
                request.close_connection = True

    def _reject(self, request: BaseHTTPRequestHandler, status: int, message: str) -> None:
        with self._condition:
            self._rejected_requests += 1
        self._respond_json(request, status, {"error": {"message": message}})
        discard_rejected_body(request)

    @staticmethod
    def _respond_json(request: BaseHTTPRequestHandler, status: int, body: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(body), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request.send_response(status)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)
