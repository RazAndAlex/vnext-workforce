"""Deterministic owned-JSONL fixture; never used by the product runtime."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any


def emit(record: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def respond(request_id: int, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    value: dict[str, Any] = {"v": 1, "kind": "response", "id": request_id, "ok": error is None}
    if error is not None:
        value["error"] = error
    else:
        value["result"] = result or {}
    emit(value)


def policy(reviewer: str = "auto_review") -> dict[str, Any]:
    return {
        "posture": resolved_posture(reviewer),
    }


def resolved_posture(reviewer: str = "auto_review") -> dict[str, Any]:
    return {
        "workspace_writes": True,
        "network": "approval_gated",
        "approvals_requested": True,
        "reviewer": reviewer,
        "environment_ready": True,
    }


def registration(model: str, tools: list) -> dict[str, Any]:
    """Attest exactly what the caller asked to bind, as the real bridge does."""

    digest = None
    if tools:
        digest = hashlib.sha256(
            json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return {
        "provider_echo": True,
        "acknowledged": True,
        "model_id": model,
        "tool_count": len(tools),
        "tool_names": [str(value["name"]) for value in tools],
        "definition_sha256": digest,
        "handler_registered": bool(tools),
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    reservation_id: str | None = None
    turn_reference: str | None = None
    generation: int | None = None
    event_cursor = 0
    if mode == "spawn-child":
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The ownership probe must not race bridge shutdown against descendant
        # creation: this marker is visible only after Popen returned.
        with open("fixture-child-ready", "w", encoding="utf-8") as marker:
            marker.write(str(os.getpid()))
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("kind") != "request":
            continue
        request_id, op = request["id"], request["op"]
        if op == "initialize":
            if mode == "init-failure":
                respond(request_id, error="fixture init failure")
            elif mode == "malformed":
                sys.stdout.write("not-json\n")
                sys.stdout.flush()
            elif mode == "oversized":
                sys.stdout.write("x" * 65_537 + "\n")
                sys.stdout.flush()
            elif mode == "stale-response":
                respond(request_id + 999, {"unexpected": True})
            else:
                respond(request_id, {
                    "provider": "claude",
                    "harness": "claude-agent-sdk",
                    "tool_support": {"accepted": True, "requires_empty": False},
                    "credential_override_rejected": True,
                })
        elif op == "start_thread":
            payload = request["payload"]
            reservation_id = payload.get("reservation_id")
            requested = payload.get("requested_posture", {})
            reviewer = requested.get("reviewer")
            tools = payload.get("tools")
            if (
                not isinstance(tools, list)
                or requested.get("workspace_writes") is not True
                or requested.get("network") not in {"restricted", "approval_gated"}
                or requested.get("approvals_requested") is not True
                or not isinstance(reviewer, str)
                or not reviewer
                or requested.get("environment_ready") is not True
            ):
                respond(request_id, error="fixture refused neutral thread request")
                continue
            if mode == "missing-echo":
                respond(request_id, {"thread_id": "fixture-session", "policy": policy(reviewer)})
            else:
                model = payload.get("model", "")
                respond(request_id, {
                    "reservation_echo": payload.get("reservation_id"),
                    "policy": policy(reviewer),
                    "connection_evidence": {"connected": True, "server_info_received": True},
                    "tool_registration": registration(model, tools),
                    "effort": {"requested": payload.get("effort", "high"), "applied": "sdk-option"},
                    "permission_mode": payload.get("permission_mode", "default"),
                })
        elif op == "start_turn":
            payload = request["payload"]
            turn_reference = payload.get("turn_reference")
            generation = payload.get("generation")
            if mode in {"attested-effects", "manager-tools"}:
                respond(request_id, {
                    "reservation_echo": reservation_id,
                    "turn_echo": turn_reference,
                    "turn_id": turn_reference,
                    "cursor": event_cursor + 1,
                })
            else:
                respond(request_id, {"provider_echo": True, "turn_id": "fixture-turn", "cursor": 1})
        elif op == "can_start_turn":
            payload = request["payload"]
            respond(request_id, {
                "reservation_echo": payload.get("reservation_id"),
                "generation": payload.get("generation"),
                "ready": True,
            })
        elif op == "wait_turn":
            if mode == "timeout":
                time.sleep(60)
            if mode == "attested-effects":
                assert isinstance(reservation_id, str)
                assert isinstance(turn_reference, str)
                assert isinstance(generation, int)
                correlation = {
                    "session": "fixture-native-session",
                    "turn": turn_reference,
                    "request": "fixture-tool-use",
                }
                emit({"v": 1, "kind": "event", "event": {
                    "name": "system",
                    "reservation_id": reservation_id,
                    "generation": generation,
                    "cursor": event_cursor + 1,
                    "native_identity": {"session_id": "fixture-native-session", "source": "AssistantMessage"},
                }})
                event_cursor += 1
                emit({"v": 1, "kind": "event", "event": {
                    "name": "tool_use",
                    "reservation_id": reservation_id,
                    "turn_reference": turn_reference,
                    "generation": generation,
                    "tool_use_id": "fixture-tool-use",
                    "effect_type": "command",
                    "provider_correlation": correlation,
                    "correlation_attested": True,
                    "cursor": event_cursor + 1,
                }})
                event_cursor += 1
                emit({"v": 1, "kind": "event", "event": {
                    "name": "tool_result",
                    "reservation_id": reservation_id,
                    "turn_reference": turn_reference,
                    "generation": generation,
                    "tool_use_id": "fixture-tool-use",
                    "effect_type": "command",
                    "provider_correlation": correlation,
                    "correlation_attested": True,
                    "status": "completed",
                    "cursor": event_cursor + 1,
                }})
                event_cursor += 1
            if mode == "manager-tools":
                # Drive the bidirectional tool-call op the way the real bridge
                # does: emit one correlated request, block on stdin for the
                # adapter's control answer, then echo that answer back as an
                # observable event so the round trip is assertable end to end.
                assert isinstance(reservation_id, str)
                assert isinstance(turn_reference, str)
                assert isinstance(generation, int)
                emit({"v": 1, "kind": "event", "event": {
                    "name": "system",
                    "reservation_id": reservation_id,
                    "generation": generation,
                    "cursor": event_cursor + 1,
                    "native_identity": {"session_id": "fixture-native-session", "source": "AssistantMessage"},
                }})
                event_cursor += 1
                emit({"v": 1, "kind": "event", "event": {
                    "name": "tool_call",
                    "reservation_id": reservation_id,
                    "turn_reference": turn_reference,
                    "generation": generation,
                    "call_id": "fixture-call",
                    "tool": "delegate",
                    "arguments": {"role": "worker", "objective": "fixture objective"},
                }})
                answer = None
                for control_line in sys.stdin:
                    control = json.loads(control_line)
                    if (
                        control.get("kind") == "control"
                        and control.get("op") == "tool_call_response"
                        and control.get("call_id") == "fixture-call"
                    ):
                        answer = control
                        break
                emit({"v": 1, "kind": "event", "event": {
                    "name": "system",
                    "reservation_id": reservation_id,
                    "generation": generation,
                    "cursor": event_cursor + 1,
                    "tool_answer_echo": answer,
                }})
                event_cursor += 1
            if mode == "permission":
                emit({"v": 1, "kind": "event", "event": {"name": "permission", "thread_id": "fixture-session", "turn_id": "fixture-turn", "tool_use_id": "fixture-tool-use", "cursor": 2}})
                for control_line in sys.stdin:
                    control = json.loads(control_line)
                    if control.get("kind") == "control" and control.get("tool_use_id") == "fixture-tool-use":
                        break
            respond(request_id, {"provider_echo": True, "status": "completed"})
        elif op == "interrupt":
            respond(request_id, {"provider_echo": True, "interrupted": True})
        elif op == "stop_native_task":
            payload = request["payload"]
            respond(request_id, {
                "reservation_echo": payload.get("reservation_id"),
                "task_id": payload.get("task_id"),
                "accepted": True,
            })
        elif op == "release_for_terminal":
            payload = request["payload"]
            respond(request_id, {
                "reservation_echo": payload.get("reservation_id"),
                "session_id": payload.get("session_id"),
                "released": True,
            })
        elif op == "steer":
            respond(request_id, {
                "reservation_echo": request["payload"].get("reservation_id"),
                "accepted": False,
                "reason": "native-steer-not-supported",
            })
        elif op == "resume":
            payload = request["payload"]
            reviewer = payload.get("requested_posture", {}).get("reviewer", "")
            tools = payload.get("tools", [])
            if not isinstance(tools, list):
                respond(request_id, error="fixture resume tools are malformed")
                continue
            if mode == "attested-effects":
                resumed_reservation = payload.get("reservation_id")
                if (
                    payload.get("session_id") != "fixture-native-session"
                    or not isinstance(resumed_reservation, str)
                    or not resumed_reservation
                ):
                    respond(request_id, error="attested resume lacks native session and local reservation")
                    continue
                reservation_id = resumed_reservation
            result = {
                "provider_echo": True,
                "thread_id": payload.get("session_id"),
                "reservation_echo": payload.get("reservation_id"),
                "policy": policy(reviewer),
                "connection_evidence": {"connected": True, "server_info_received": True},
            }
            if mode != "resume-missing-registration":
                result["tool_registration"] = registration(payload.get("model", ""), tools)
            respond(request_id, result)
        elif op == "read_thread":
            payload = request["payload"]
            respond(request_id, {"provider_echo": True, "thread_id": payload["thread_id"], "items": []})
        elif op == "close":
            respond(request_id, {"closed": True})
            return 0
        else:
            respond(request_id, error="unsupported fixture op")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
