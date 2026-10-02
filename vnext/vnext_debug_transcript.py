"""Opt-in local transcripts for vNext runtime observation.

These JSONL files are deliberately separate from durable receipts. They may
contain model-authored text and raw app-server item payloads, so callers must
enable them explicitly and keep them under the gitignored local data tree.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .vnext_runtime_types import TurnHandle


SCHEMA_VERSION = "VNEXT_DEBUG_TRANSCRIPT_V1"


@dataclass(frozen=True)
class ScopedDebugEvent:
    """A raw event that a provider boundary has already scoped to one turn."""

    event: Mapping[str, Any]
    is_current_turn: bool


class VNextDebugTranscript:
    """Thread-safe JSONL writer for one vNext product run."""

    def __init__(
        self,
        run_id: str,
        *,
        root: str | Path = Path("data/transcripts"),
        include_raw_items: bool = True,
    ) -> None:
        if not run_id or any(character in run_id for character in "/\\\r\n\x00"):
            raise ValueError("run_id must be a safe filename component")
        self.run_id = run_id
        self.path = Path(root).resolve() / f"{run_id}.jsonl"
        self.include_raw_items = bool(include_raw_items)
        self._lock = threading.Lock()
        self._recorded_turns: set[tuple[str, str, str]] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def for_provider(self, provider: str) -> ProviderDebugTranscript:
        """Bind debug decoding to the same provider selected for an adapter."""

        return ProviderDebugTranscript(self, debug_event_reader(provider))

    def _record_turn(
        self,
        reader: DebugEventReader,
        handle: TurnHandle,
        events: Sequence[Mapping[str, Any] | ScopedDebugEvent],
    ) -> None:
        key = (reader.provider, handle.thread_id, handle.turn_id)
        with self._lock:
            if key in self._recorded_turns:
                return
            self._recorded_turns.add(key)

            item_events: list[dict[str, Any]] = []
            for supplied_event in events:
                if isinstance(supplied_event, ScopedDebugEvent):
                    if not supplied_event.is_current_turn:
                        continue
                    event = supplied_event.event
                else:
                    event = supplied_event
                    if not reader.accepts_unscoped_event(event, handle):
                        continue
                if reader.is_item_event(event):
                    item_events.append(dict(event))
            histogram = Counter()
            for event in item_events:
                item_type = reader.completed_item_type(event)
                if isinstance(item_type, str) and item_type:
                    histogram[item_type] += 1

            self._append_locked(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "turn_summary",
                    "run_id": self.run_id,
                    "thread_id": handle.thread_id,
                    "turn_id": handle.turn_id,
                    "item_completed_type_histogram": dict(sorted(histogram.items())),
                }
            )
            if self.include_raw_items:
                for event in item_events:
                    self._append_locked(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "kind": "raw_item_event",
                            "run_id": self.run_id,
                            "thread_id": handle.thread_id,
                            "turn_id": handle.turn_id,
                            "event": event,
                        }
                    )

    def _record_native_approval(
        self,
        reader: DebugEventReader,
        method: str,
        params: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        with self._lock:
            correlations = reader.approval_correlations(params)
            value: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "kind": "native_approval",
                "run_id": self.run_id,
                "method": method,
                "decision": "decline",
                "thread_correlated": correlations[0],
                "turn_correlated": correlations[1],
                "item_correlated": correlations[2],
                "response": dict(response),
            }
            if self.include_raw_items:
                value["params"] = dict(params)
            self._append_locked(value)

    def _append_locked(self, value: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            )
            stream.write("\n")


class DebugEventReader(Protocol):
    provider: str

    def is_item_event(self, event: Mapping[str, Any]) -> bool: ...

    def accepts_unscoped_event(self, event: Mapping[str, Any], handle: TurnHandle) -> bool: ...

    def completed_item_type(self, event: Mapping[str, Any]) -> str | None: ...

    def approval_correlations(self, params: Mapping[str, Any]) -> tuple[bool, bool, bool]: ...


class CodexDebugEventReader:
    """Decode Codex app-server events for opt-in local diagnostics."""

    provider = "codex"

    @staticmethod
    def is_item_event(event: Mapping[str, Any]) -> bool:
        return str(event.get("method") or "").startswith("item/")

    @staticmethod
    def accepts_unscoped_event(event: Mapping[str, Any], handle: TurnHandle) -> bool:
        # Native events need an explicit boundary attestation.  Keeping the
        # raw correlation protocol outside this writer prevents unscoped input
        # from leaking a sibling turn into the transcript.
        del event, handle
        return False

    @staticmethod
    def completed_item_type(event: Mapping[str, Any]) -> str | None:
        if event.get("method") != "item/completed":
            return None
        params = event.get("params")
        item = params.get("item") if isinstance(params, Mapping) else None
        item_type = item.get("type") if isinstance(item, Mapping) else None
        return item_type if isinstance(item_type, str) else None

    @staticmethod
    def approval_correlations(params: Mapping[str, Any]) -> tuple[bool, bool, bool]:
        correlation = params.get("provider_correlation")
        correlation = correlation if isinstance(correlation, Mapping) else {}
        return (
            bool(correlation.get("session")),
            bool(correlation.get("turn")),
            bool(correlation.get("request")),
        )


class ClaudeDebugEventReader:
    """Decode Claude bridge events without importing bridge schema upstream."""

    provider = "claude"

    @staticmethod
    def is_item_event(event: Mapping[str, Any]) -> bool:
        return event.get("name") in {"tool_use", "tool_result"}

    @staticmethod
    def accepts_unscoped_event(event: Mapping[str, Any], handle: TurnHandle) -> bool:
        return (
            event.get("reservation_id") == handle.thread_id
            and event.get("turn_reference") == handle.turn_id
        )

    @staticmethod
    def completed_item_type(event: Mapping[str, Any]) -> str | None:
        return str(event["name"]) if event.get("name") == "tool_result" else None

    @staticmethod
    def approval_correlations(params: Mapping[str, Any]) -> tuple[bool, bool, bool]:
        correlation = params.get("provider_correlation")
        correlation = correlation if isinstance(correlation, Mapping) else {}
        return (
            bool(correlation.get("session")),
            bool(correlation.get("turn")),
            bool(correlation.get("request")),
        )


def debug_event_reader(provider: str) -> DebugEventReader:
    readers: dict[str, DebugEventReader] = {
        "codex": CodexDebugEventReader(),
        "claude": ClaudeDebugEventReader(),
    }
    try:
        return readers[provider]
    except KeyError as exc:
        raise ValueError(f"unsupported debug transcript provider: {provider!r}") from exc


class ProviderDebugTranscript:
    """Provider-bound view over one neutral transcript writer.

    Providers whose raw correlations are decoded upstream must pass
    ``ScopedDebugEvent`` values, so this generic writer can reject foreign
    events without knowing provider protocol fields.
    """

    def __init__(self, transcript: VNextDebugTranscript, reader: DebugEventReader) -> None:
        self._transcript = transcript
        self._reader = reader

    def record_turn(
        self,
        handle: TurnHandle,
        events: Sequence[Mapping[str, Any] | ScopedDebugEvent],
    ) -> None:
        self._transcript._record_turn(self._reader, handle, events)

    def record_native_approval(
        self,
        method: str,
        params: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        self._transcript._record_native_approval(self._reader, method, params, response)
