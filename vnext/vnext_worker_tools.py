"""Bounded contracts retained after native Worker-tool retirement.

Workers execute through runtime-provided native file and terminal tools. This
module now contains only content-safe approval values used by the scheduler and
the optional manager-selected command-evidence contract; it owns no tools,
subprocesses, file operations, or approval reviewers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ApprovalDecision(str, Enum):
    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"


@dataclass(frozen=True)
class ApprovalOutcome:
    decision: ApprovalDecision
    rationale: str = ""


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str
    tool: str
    effect: str
    permission: str
    target: str
    justification: str
    command: tuple[str, ...] = ()


class RequiredToolEffectError(ValueError):
    """A manager-selected required_tool_effect violated the closed shape."""


REQUIRED_EFFECT_SUPPORTED_TOOLS = ("run_command",)
REQUIRED_EFFECT_MIN_TIMEOUT = 0.1
REQUIRED_EFFECT_MAX_TIMEOUT = 120.0


@dataclass(frozen=True)
class RequiredToolEffect:
    """Optional manager-selected command evidence transported to a Worker."""

    tool: str
    argv: tuple[str, ...]
    cwd: str
    timeout_seconds: float
    exit_code: int


def _require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RequiredToolEffectError(f"{where} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, where: str) -> None:
    actual = set(value.keys())
    if actual != keys:
        raise RequiredToolEffectError(
            f"{where} must contain exactly {sorted(keys)}; got {sorted(actual)}"
        )


def validate_required_tool_effect(value: Any) -> RequiredToolEffect:
    """Strictly validate the closed optional command-evidence shape."""

    top = _require_mapping(value, where="required_tool_effect")
    _require_exact_keys(top, {"tool", "arguments", "completion"}, where="required_tool_effect")
    tool = top["tool"]
    if tool not in REQUIRED_EFFECT_SUPPORTED_TOOLS:
        raise RequiredToolEffectError(
            f"required_tool_effect tool must be one of {list(REQUIRED_EFFECT_SUPPORTED_TOOLS)}"
        )
    arguments = _require_mapping(top["arguments"], where="arguments")
    _require_exact_keys(arguments, {"argv", "cwd", "timeout_seconds"}, where="arguments")
    argv = arguments["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > 64
        or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
    ):
        raise RequiredToolEffectError(
            "arguments.argv must be a non-empty array of up to 64 non-empty strings"
        )
    cwd = arguments["cwd"]
    if not isinstance(cwd, str) or not cwd or "\x00" in cwd:
        raise RequiredToolEffectError("arguments.cwd must be a non-empty string")
    timeout = arguments["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise RequiredToolEffectError("arguments.timeout_seconds must be a number")
    timeout = float(timeout)
    if not (REQUIRED_EFFECT_MIN_TIMEOUT <= timeout <= REQUIRED_EFFECT_MAX_TIMEOUT):
        raise RequiredToolEffectError(
            f"arguments.timeout_seconds must be within "
            f"[{REQUIRED_EFFECT_MIN_TIMEOUT}, {REQUIRED_EFFECT_MAX_TIMEOUT}]"
        )
    completion = _require_mapping(top["completion"], where="completion")
    _require_exact_keys(completion, {"exit_code"}, where="completion")
    exit_code = completion["exit_code"]
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise RequiredToolEffectError("completion.exit_code must be an integer")
    return RequiredToolEffect(
        tool=str(tool),
        argv=tuple(argv),
        cwd=cwd,
        timeout_seconds=timeout,
        exit_code=exit_code,
    )


def normalize_command_cwd(workspace: str | Path, raw_cwd: str) -> str | None:
    """Return a workspace-relative command cwd, or ``None`` on escape."""

    root = Path(workspace).resolve()
    candidate = (root / raw_cwd).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    rendered = relative.as_posix()
    return rendered if rendered != "." else "."


def required_effect_contract_facts(
    contract: RequiredToolEffect,
    *,
    workspace: str | Path,
) -> dict[str, Any]:
    """Return normalized facts without model prose or runtime output."""

    return {
        "tool": contract.tool,
        "argv": list(contract.argv),
        "cwd": normalize_command_cwd(workspace, contract.cwd) or contract.cwd,
        "timeout_seconds": contract.timeout_seconds,
        "exit_code": contract.exit_code,
    }
