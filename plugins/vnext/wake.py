"""Wait for a delegated worker, then wake Claude Code with a system reminder."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys


def delegate_response(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    response = payload.get("tool_response")
    if isinstance(response, list):
        response = next(
            (block.get("text") for block in response
             if isinstance(block, dict) and block.get("type") == "text"),
            None,
        )
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except ValueError:
            return {}
    return response if isinstance(response, dict) else {}


def main() -> int:
    try:
        response = delegate_response(json.load(sys.stdin))
    except (ValueError, OSError):
        return 0
    command = response.get("wake_command")
    if not isinstance(command, str) or not command.strip():
        return 0
    agent = response.get("agent_id", "unknown")
    try:
        result = subprocess.run(
            command if os.name == "nt" else shlex.split(command),
            capture_output=True, encoding="utf-8", errors="replace",
            # A worker's report can hold characters a Windows pipe's code page lacks.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, ValueError) as exc:
        message = f"vNext: could not wait for worker {agent} (exit 2): {exc}"
    else:
        lines = result.stdout.strip()
        if result.returncode == 0:
            message = (
                f"vNext: worker {agent} stopped. Its report:\n{lines}\n"
                "Use inspect for the full record."
            )
        elif result.returncode == 3:
            lines = "; ".join(result.stdout.strip().splitlines())
            message = (
                f"vNext: worker {agent} is still running after 30 minutes: {lines}. "
                "Check it with inspect; to keep waiting, run this in the background: "
                f"{command}"
            )
        else:
            message = (
                f"vNext: could not wait for worker {agent} (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
