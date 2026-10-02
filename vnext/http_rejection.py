"""Bounded cleanup for loopback HTTP replies that reject an unread body."""
from __future__ import annotations

import time
from http.server import BaseHTTPRequestHandler


def discard_rejected_body(request: BaseHTTPRequestHandler) -> None:
    """Drain at most 1 MiB for 250 ms after sending an early rejection.

    Windows may reset a socket closed with unread request bytes before the
    client receives the response. The body is never parsed or retained.
    The caller must use this only when it has not read the request body.
    """
    request.wfile.flush()
    try:
        remaining = int(request.headers.get("Content-Length", "0"))
    except ValueError:
        return
    if not 0 < remaining <= 1_048_576:
        return
    original_timeout = request.connection.gettimeout()
    deadline = time.monotonic() + 0.25
    try:
        while remaining > 0:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            request.connection.settimeout(timeout)
            chunk = request.rfile.read1(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)
    except OSError:
        pass
    finally:
        request.connection.settimeout(original_timeout)
