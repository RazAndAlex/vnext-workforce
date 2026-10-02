from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkforceRequest:
    task: str
    task_name: str
    workspace: Path
    preset: str = "sol"

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task cannot be empty")
        if not self.task_name.strip():
            raise ValueError("task name cannot be empty")
        workspace = self.workspace.resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        if self.preset != "sol":
            raise ValueError("the first external-thread build supports the Sol preset only")
        object.__setattr__(self, "workspace", workspace)


@dataclass(frozen=True)
class WorkforceProgress:
    run_id: str
    stage_id: str
    role: str
    state: str
    message_code: str
    display_message: str
    observed_at: str


class RunCancellation:
    """Atomic cancellation/finalization boundary shared by host and controller."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._phase = "open"

    def request(self) -> bool:
        """Accept cancellation only while finalization is still reversible."""

        with self._lock:
            if self._phase != "open":
                return False
            self._phase = "cancelling"
            self._event.set()
            return True

    def begin_finalization(self) -> None:
        """Atomically claim the irreversible promotion boundary."""

        with self._lock:
            if self._phase == "cancelling":
                raise RunCancelledError("run was cancelled before finalization")
            if self._phase != "open":
                raise RuntimeError(f"cannot begin finalization from {self._phase}")
            self._phase = "finalizing"

    def begin_cleanup(self) -> str:
        """Seal an exceptional outcome before its terminal cleanup begins."""

        with self._lock:
            if self._phase == "open":
                self._phase = "finalizing"
            return self._phase

    def mark_finished(self) -> None:
        with self._lock:
            if self._phase != "cancelling":
                self._phase = "finished"

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def finalizing(self) -> bool:
        with self._lock:
            return self._phase == "finalizing"

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


class RunCancelledError(RuntimeError):
    pass


def cleanup_receipt_is_terminal(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("outcome"), str)
        and isinstance(value.get("residual_count"), int)
        and not isinstance(value.get("residual_count"), bool)
        and isinstance(value.get("streams_drained"), bool)
    )


def cleanup_receipt_is_clean(value: Any) -> bool:
    return (
        cleanup_receipt_is_terminal(value)
        and value.get("outcome") == "clean"
        and value.get("residual_count") == 0
        and value.get("streams_drained") is True
    )
