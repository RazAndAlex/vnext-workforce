"""Private, content-safe streamable-HTTP MCP relay for Codex diagnostics.

This is an opt-in probe building block. It does not enable native participation
in the normal session service. The relay accepts only a generated loopback
bearer, exposes a fixed tool list, and records metadata shape rather than tool
arguments or model text.
"""

from __future__ import annotations

import json
import secrets
import threading
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Sequence


from .http_rejection import discard_rejected_body
from .vnext_runtime_types import ToolCallResult


class CodexMcpRelayError(RuntimeError):
    pass


# Revisions this relay is willing to answer under.  It has no version-specific
# behaviour -- it speaks plain JSON-RPC over POST with no streaming -- so the
# list exists to echo a client's own revision back rather than to gate on it.
# A client asking for something unknown gets the newest entry, which is what
# the specification asks a server to do.
MCP_PROTOCOL_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
)

# Keys the MCP tool descriptor defines.  Internal tool records carry provider
# fields beside these -- the Codex app-server adds ``"type": "function"`` -- and
# a strict client rejects the whole listing over one unknown key.
_MCP_TOOL_KEYS = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")


McpDispatcher = Callable[[str, Mapping[str, Any], Mapping[str, Any]], ToolCallResult | Mapping[str, Any]]


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


@dataclass(frozen=True)
class McpRelayReceipt:
    method_counts: dict[str, int]
    tool_calls: int
    host_thread_metadata_present: int
    metadata_keys: tuple[str, ...]
    failure_categories: dict[str, int]


class McpProtocol:
    """The JSON-RPC half of an MCP server, with no transport attached.

    It was inside the HTTP relay until a second transport needed it.  A client
    that launches its server as a subprocess speaks the same methods over stdio
    that a configured one speaks over POST, and the difference between those two
    is framing, not protocol -- so the protocol lives here and each transport
    only moves bytes.
    """

    def __init__(
        self,
        *,
        tools: Sequence[Mapping[str, Any]],
        dispatcher: McpDispatcher,
        server_info: Mapping[str, str],
        server_name: str | None = None,
    ) -> None:
        self._tools = tuple(dict(tool) for tool in tools)
        # Public and rebindable: a host may swap the dispatcher after the server
        # is built, and a copy taken at construction would silently ignore it.
        self.dispatcher = dispatcher
        self._server_info = dict(server_info)
        self._server_name = server_name
        self._lock = threading.RLock()
        self.methods: Counter[str] = Counter()
        self.tool_calls = 0
        self.meta_thread_ids = 0
        self.metadata_keys: set[str] = set()

    @property
    def tool_names(self) -> set[str]:
        return {str(tool.get("name") or "") for tool in self._tools}

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Answer one JSON-RPC request, or ``None`` for an accepted notification."""

        request_id = payload.get("id")
        method = payload.get("method")
        if not isinstance(method, str):
            return self.error(request_id, -32600, "MCP method is required")
        with self._lock:
            self.methods[method] += 1
        if method == "initialize":
            params = payload.get("params")
            requested = params.get("protocolVersion") if isinstance(params, Mapping) else None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": (
                        requested if requested in MCP_PROTOCOL_VERSIONS
                        else MCP_PROTOCOL_VERSIONS[-1]
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": dict(self._server_info),
                },
            }
        if method.startswith("notifications/"):
            if request_id is None:
                return None
            return self.error(request_id, -32600, "MCP notification must not have an id")
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": [self.tool_descriptor(tool) for tool in self._tools]},
            }
        if method == "ping":
            # The specification requires a receiver to answer a ping with an
            # empty result.  Answering -32601 made every heartbeat from a
            # client that keeps a connection alive this way look like a
            # failure, and some clients drop the connection over it.
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method != "tools/call":
            return self.error(request_id, -32601, "MCP method is not supported")
        params = payload.get("params")
        if not isinstance(params, Mapping):
            return self.error(request_id, -32602, "MCP tool params are required")
        name = params.get("name")
        arguments = params.get("arguments", {})
        metadata = params.get("_meta", {})
        if (not isinstance(name, str) or name not in self.tool_names
                or not isinstance(arguments, Mapping)):
            return self.error(request_id, -32602, "MCP tool invocation is invalid")
        if not isinstance(metadata, Mapping):
            return self.error(request_id, -32602, "MCP tool metadata is invalid")
        with self._lock:
            self.tool_calls += 1
            self.metadata_keys.update(str(key) for key in metadata)
            self.meta_thread_ids += int(
                isinstance(metadata.get("threadId"), str) and bool(metadata["threadId"])
            )
        dispatch_metadata = dict(metadata)
        # The server owns this field.  A model-controlled ``_meta`` value can
        # never select a handler branch on behalf of another configured server.
        if self._server_name is not None:
            dispatch_metadata["serverName"] = self._server_name
        answer = self.dispatcher(name, dict(arguments), dispatch_metadata)
        if isinstance(answer, ToolCallResult):
            value, is_error = dict(answer.value), not answer.success
        elif isinstance(answer, Mapping):
            value, is_error = dict(answer), answer.get("success") is False
        else:
            raise CodexMcpRelayError("MCP dispatcher returned an invalid result")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                }],
                "structuredContent": value,
                "isError": is_error,
            },
        }

    @staticmethod
    def tool_descriptor(tool: Mapping[str, Any]) -> dict[str, Any]:
        """Project one internal tool record onto the keys MCP defines."""

        return {key: tool[key] for key in _MCP_TOOL_KEYS if key in tool}

    @staticmethod
    def error(request_id: object, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class CodexMcpRelay:
    """One loopback bearer-protected MCP endpoint with drain-on-close semantics."""

    def __init__(
        self,
        *,
        tools: Sequence[Mapping[str, Any]],
        dispatcher: McpDispatcher,
        host: str = "127.0.0.1",
        port: int = 0,
        server_name: str | None = None,
        token: str | None = None,
        server_info: Mapping[str, str] | None = None,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError("Codex MCP relay must bind a loopback host")
        self._tools = tuple(dict(tool) for tool in tools)
        if server_name is not None and (
            not isinstance(server_name, str) or not server_name.strip() or server_name != server_name.strip()
        ):
            raise ValueError("MCP server name must be a nonempty trimmed string")
        self._server_name = server_name
        # A generated token suits a relay whose client is a process this service
        # starts and configures.  A client the user configures by hand needs a
        # token that survives a restart, so a caller may supply one.
        if token is not None and (not isinstance(token, str) or len(token.strip()) < 16):
            raise ValueError("an MCP relay token must be at least 16 characters")
        self._token = token.strip() if token is not None else secrets.token_urlsafe(32)
        self._server_info = dict(server_info or {"name": "vnext-codex-relay", "version": "1"})
        self._protocol = McpProtocol(
            tools=self._tools,
            dispatcher=dispatcher,
            server_info=self._server_info,
            server_name=server_name,
        )
        self._condition = threading.Condition(threading.RLock())
        self._accepting = True
        self._closed = False
        self._active_requests = 0
        self._failure_categories: Counter[str] = Counter()
        parent = self

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib entrypoint
                parent._receive(self)

            def do_GET(self) -> None:  # noqa: N802 - stdlib entrypoint
                # Streamable HTTP lets a client open a server-to-client SSE
                # stream with GET.  This relay has nothing to push, and the
                # specification's answer for that is 405 -- which clients read
                # as "no stream here" and carry on from.  The stdlib default,
                # 501, reads as a broken server instead.
                parent._unsupported(self)

            def do_DELETE(self) -> None:  # noqa: N802 - stdlib entrypoint
                # Likewise for explicit session termination: this relay keeps no
                # MCP session id, so there is none to delete.
                parent._unsupported(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Receiver)
        self._thread: threading.Thread | None = None

    @property
    def _dispatcher(self) -> McpDispatcher:
        return self._protocol.dispatcher

    @_dispatcher.setter
    def _dispatcher(self, dispatcher: McpDispatcher) -> None:
        self._protocol.dispatcher = dispatcher

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/mcp"

    @property
    def bearer_token(self) -> str:
        return self._token

    def config_overrides(self, name: str = "vnext_relay") -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("MCP server name is required")
        return {
            f"mcp_servers.{name}.url": self.endpoint,
            f"mcp_servers.{name}.http_headers.Authorization": f"Bearer {self._token}",
        }

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise CodexMcpRelayError("MCP relay is closed")
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="vnext-codex-mcp-relay",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._accepting = False
            while self._active_requests:
                self._condition.wait()
            self._closed = True
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def receipt(self) -> McpRelayReceipt:
        with self._condition:
            return McpRelayReceipt(
                method_counts=dict(self._protocol.methods),
                tool_calls=self._protocol.tool_calls,
                host_thread_metadata_present=self._protocol.meta_thread_ids,
                metadata_keys=tuple(sorted(self._protocol.metadata_keys)),
                failure_categories=dict(self._failure_categories),
            )

    def _receive(self, request: BaseHTTPRequestHandler) -> None:
        if request.path != "/mcp":
            self._reject(request, 404, {"error": "unknown MCP endpoint"})
            return
        if not secrets.compare_digest(
            request.headers.get("Authorization", "").encode(), f"Bearer {self._token}".encode()
        ):
            self._reject(request, 403, {"error": "invalid MCP relay credential"})
            return
        with self._condition:
            if not self._accepting:
                self._reject(request, 503, {"error": "MCP relay is closing"})
                return
            self._active_requests += 1
        try:
            try:
                length = int(request.headers.get("Content-Length", "0"))
            except ValueError:
                self._reject(request, 400, {"error": "request body size is invalid"})
                return
            if length <= 0 or length > 1_048_576:
                self._reject(request, 400, {"error": "request body size is invalid"})
                return
            payload = json.loads(request.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("request body must be an object")
            result = self._dispatch(payload)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            self._respond(request, 400, {"error": str(exc)})
        except Exception as exc:
            with self._condition:
                self._failure_categories[self._failure_category(exc)] += 1
            self._respond(request, 502, {"error": f"vNext MCP relay failed: {type(exc).__name__}"})
        else:
            if result is None:
                self._accepted(request)
            else:
                self._respond(request, 200, result)
        finally:
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()

    def _dispatch(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        return self._protocol.handle(payload)

    @classmethod
    def _unsupported(cls, request: BaseHTTPRequestHandler) -> None:
        cls._reject(request, 405, {"error": "this MCP endpoint accepts POST only"})

    @staticmethod
    def _failure_category(exc: Exception) -> str:
        """Classify only relay-local, content-free dispatch outcomes."""

        if type(exc).__name__ != "VNextAppServerError":
            return "relay_exception"
        message = str(exc)
        known = {
            "native MCP call lacks host threadId metadata": "missing_host_thread_metadata",
            "native MCP thread is not an attested child": "unbound_native_child",
            "native MCP child has no registered handler": "unregistered_child_handler",
            "native MCP child has no active exact turn": "no_active_child_turn",
        }
        return known.get(message, "native_dispatch_rejected")

    @classmethod
    def _reject(cls, request: BaseHTTPRequestHandler, status: int, body: Mapping[str, Any]) -> None:
        """Reply without dispatch, then bound the drain of an unread POST body.

        Closing a Windows socket with unread request bytes can reset it before
        the client receives the rejection. Never parse or retain these bytes.
        """
        cls._respond(request, status, body)
        discard_rejected_body(request)

    @staticmethod
    def _respond(request: BaseHTTPRequestHandler, status: int, body: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(body), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request.send_response(status)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)

    @staticmethod
    def _accepted(request: BaseHTTPRequestHandler) -> None:
        request.send_response(202)
        request.send_header("Content-Length", "0")
        request.end_headers()
