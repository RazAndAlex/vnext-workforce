"""Content-safe projection of authoritative native-runtime effects.

An adapter may supply recognized completed-item records. The manager only needs
stable facts for scheduling, receipts, and acceptance. This module correlates
those facts to one managed turn, deduplicates them, and retains no command text,
output, diff, prompt, file content, or raw runtime identifier. Unrecognized
provider records project no effects.
"""

from __future__ import annotations

import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class RuntimeEffectProjectionError(RuntimeError):
    """The evidence journal cannot decode or correlate a runtime record."""


@dataclass(frozen=True)
class NativeEffectCandidate:
    """One provider-decoded effect candidate with opaque correlation."""

    thread_ref: str | None
    turn_ref: str | None
    item_ref: str | None
    item: Mapping[str, Any] | None
    malformed: bool = False


class RuntimeEffectReader(Protocol):
    """Decode one provider event without leaking its schema into the journal."""

    provider: str

    def completed_effect(self, event: Mapping[str, Any]) -> NativeEffectCandidate | None: ...


class CodexRuntimeEffectReader:
    """Decode Codex app-server completed command and file items."""

    provider = "codex"

    def completed_effect(self, event: Mapping[str, Any]) -> NativeEffectCandidate | None:
        if event.get("method") != "item/completed":
            return None
        params = event.get("params")
        if not isinstance(params, Mapping):
            return NativeEffectCandidate(None, None, None, None, malformed=True)
        item = params.get("item")
        if not isinstance(item, Mapping):
            return None
        native_type = item.get("type")
        if native_type not in {"commandExecution", "fileChange"}:
            return None
        raw_item_ref = item.get("id")
        if native_type == "commandExecution":
            normalized_item = {
                "type": "command",
                "status": item.get("status"),
                "cwd": item.get("cwd"),
                "actions": item.get("commandActions"),
                "exit_code": item.get("exitCode"),
                "duration_ms": item.get("durationMs"),
            }
        else:
            normalized_item = {
                "type": "file",
                "status": item.get("status"),
                "changes": item.get("changes"),
            }
        return NativeEffectCandidate(
            thread_ref=_opaque_ref(params.get("threadId")),
            turn_ref=_opaque_ref(params.get("turnId")),
            item_ref=_opaque_ref(raw_item_ref),
            item=normalized_item,
            malformed=not isinstance(raw_item_ref, str) or not raw_item_ref,
        )


class ClaudeRuntimeEffectReader:
    """Decode sanitized, attested Claude tool-result events only."""

    provider = "claude"

    def completed_effect(self, event: Mapping[str, Any]) -> NativeEffectCandidate | None:
        if event.get("name") != "tool_result":
            return None
        thread_ref = _opaque_ref(event.get("reservation_id"))
        turn_ref = _opaque_ref(event.get("turn_reference"))
        item_ref = _opaque_ref(event.get("tool_use_id"))
        effect_type = event.get("effect_type")
        correlation = event.get("provider_correlation")
        if (
            event.get("correlation_attested") is not True
            or not isinstance(correlation, Mapping)
            or not isinstance(thread_ref, str)
            or not isinstance(turn_ref, str)
            or not isinstance(item_ref, str)
            or dict(correlation)
            != {"session": correlation.get("session"), "turn": turn_ref, "request": item_ref}
            or not _opaque_ref(correlation.get("session"))
            or effect_type not in {"command", "file"}
        ):
            return NativeEffectCandidate(thread_ref, turn_ref, item_ref, None, malformed=True)
        status = event.get("status")
        # The boundary emits "unknown" when the provider's tool result carried
        # no `is_error` at all.  Observed live on 0.2.143: a Bash result sets
        # `is_error=False`, a Write result leaves it `None`.  An absent field
        # is a fact about the provider's reporting, not degraded evidence, so
        # it is declared not-reported rather than left to read as a failed
        # status decode.  Nothing here infers success from the absence.
        status_not_reported = [] if status in {"completed", "failed"} else ["status"]
        if effect_type == "command":
            item = {
                "type": effect_type,
                "status": status,
                "cwd": None,
                "actions": None,
                "exit_code": None,
                "duration_ms": None,
                "not_reported": [
                    "cwd",
                    "actions",
                    "exit_code",
                    "duration_ms",
                    *status_not_reported,
                ],
            }
        else:
            changes = event.get("changes")
            if isinstance(changes, list):
                normalized_changes: list[Any] = []
                for change in changes:
                    if not isinstance(change, Mapping):
                        normalized_changes.append(change)
                        continue
                    # Claude's Write result does not report a change kind.  Do
                    # not accept a supplied kind as an inferred add/update:
                    # retain the path but mark that attempted schema drift as
                    # degraded evidence.
                    kind = change.get("kind")
                    if kind != {"type": "not_reported"}:
                        normalized_change = dict(change)
                        normalized_change["kind"] = {"type": "unknown"}
                        normalized_changes.append(normalized_change)
                    else:
                        normalized_changes.append(dict(change))
                changes = normalized_changes
            item = {
                "type": effect_type,
                "status": status,
                "changes": changes,
                "not_reported": ["change_kind", *status_not_reported],
            }
            if event.get("evidence_limited") is True or (
                isinstance(changes, list)
                and any(
                    isinstance(change, Mapping)
                    and change.get("kind") == {"type": "unknown"}
                    for change in changes
                )
            ):
                item["evidence_limited"] = True
        return NativeEffectCandidate(thread_ref, turn_ref, item_ref, item)


class ZaiRuntimeEffectReader(ClaudeRuntimeEffectReader):
    """Decode the shared SDK event shape under z.ai provider identity."""

    provider = "zai"


class CommandCodeRuntimeEffectReader(CodexRuntimeEffectReader):
    """Decode the Codex app-server event shape under Command Code identity.

    Command Code is the same harness reading the same wire: the pinned Codex
    CLI, pointed at a reseller endpoint by a loopback bridge.  Only the
    provider identity on the journal entry differs.
    """

    provider = "commandcode"


class ExternalRuntimeEffectReader:
    """Decode nothing: an external primary's effects are not observed here.

    The client that owns an external primary runs its own tools through its own
    harness, so vNext sees no event stream for it and must not invent one.  An
    empty reader records that absence honestly; the alternative was a missing
    registration, which failed the whole session at startup.
    """

    provider = "external"

    def completed_effect(self, event: Mapping[str, Any]) -> NativeEffectCandidate | None:
        return None


NATIVE_EFFECT_READER_HOOK = "runtime_effect_reader"
BUILT_IN_EFFECT_READER_PROVIDERS = frozenset(
    {"codex", "claude", "zai", "commandcode", "external"}
)


def _built_in_effect_readers() -> dict[str, RuntimeEffectReader]:
    """One fresh reader per provider; the names are BUILT_IN_EFFECT_READER_PROVIDERS."""

    return {
        "codex": CodexRuntimeEffectReader(),
        "claude": ClaudeRuntimeEffectReader(),
        "zai": ZaiRuntimeEffectReader(),
        "commandcode": CommandCodeRuntimeEffectReader(),
        "external": ExternalRuntimeEffectReader(),
    }


def effect_reader_hint(target: Any) -> str | None:
    """Say how an adapter answers for its own effects, or why it cannot.

    A provider registered through ``--provider NAME=module:attribute`` has no
    built-in decoder, so the adapter class owns the answer in one of two ways:
    a ``runtime_effect_reader`` of its own, or a ``provider`` naming the
    built-in harness whose wire it speaks.  Both are class attributes, so the
    CLI can ask this question at parse time without starting a process.

    Returns the hook name, the built-in provider name, or None.
    """

    if getattr(target, NATIVE_EFFECT_READER_HOOK, None) is not None:
        return NATIVE_EFFECT_READER_HOOK
    native = getattr(target, "provider", None)
    if isinstance(native, str) and native in _built_in_effect_readers():
        return native
    return None


def runtime_effect_reader(
    provider: str, adapter: Any = None
) -> RuntimeEffectReader:
    """Select the decoder at the concrete provider boundary.

    The catalog card's provider answers for every built-in.  A registered
    provider's name means nothing here, so its adapter is asked: the effect
    wire is a property of the harness the adapter speaks, never of the name a
    catalog gave it.  Without the adapter this used to raise, and the worker
    blocked on its first turn.
    """

    readers = _built_in_effect_readers()
    reader = readers.get(provider)
    if reader is not None:
        return reader
    supplied = getattr(adapter, NATIVE_EFFECT_READER_HOOK, None)
    if supplied is not None:
        candidate = supplied() if callable(supplied) else supplied
        if candidate is not None and hasattr(candidate, "completed_effect"):
            return candidate
        raise RuntimeEffectProjectionError(
            f"provider {provider!r} supplied a {NATIVE_EFFECT_READER_HOOK} with no "
            "completed_effect method"
        )
    native = getattr(adapter, "provider", None)
    if isinstance(native, str) and native in readers:
        return readers[native]
    raise RuntimeEffectProjectionError(
        f"no native effect reader is registered for provider {provider!r}; a "
        f"registered provider's adapter must carry a {NATIVE_EFFECT_READER_HOOK} "
        "or a 'provider' naming the built-in provider whose event shape it "
        f"speaks ({', '.join(sorted(BUILT_IN_EFFECT_READER_PROVIDERS))})"
    )


def _opaque_ref(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class RuntimeFileChange:
    path: str
    kind: str


@dataclass(frozen=True)
class RuntimeEffect:
    sequence: int
    agent_id: str
    provider: str
    item_ref: str
    effect: str
    status: str
    evidence_limited: bool
    not_reported_fields: tuple[str, ...] = ()
    exit_code: int | None = None
    duration_ms: int | None = None
    cwd: str | None = None
    action_types: tuple[str, ...] = ()
    changes: tuple[RuntimeFileChange, ...] = ()


class RuntimeEffectJournal:
    """Own bounded, content-safe native effect evidence for one workspace."""

    _COMMAND_STATUSES = frozenset({"completed", "failed", "declined"})
    _FILE_STATUSES = frozenset({"completed", "failed", "declined"})
    _ACTION_TYPES = frozenset({"read", "listFiles", "search", "unknown"})
    _CHANGE_KINDS = frozenset({"add", "delete", "update", "not_reported"})

    def __init__(
        self,
        workspace: str | Path,
        *,
        max_effects: int = 2_048,
        max_actions_per_effect: int = 32,
        max_changes_per_effect: int = 256,
        max_replay_memory: int | None = None,
    ) -> None:
        if max_effects < 1 or max_actions_per_effect < 1 or max_changes_per_effect < 1:
            raise ValueError("runtime effect bounds must be positive")
        if max_replay_memory is None:
            # Keys are small next to the effects they name, so the journal can
            # afford to remember more of them than it keeps effects.
            max_replay_memory = 2 * max_effects
        if max_replay_memory < 1:
            raise ValueError("runtime effect bounds must be positive")
        self.workspace = Path(workspace).resolve()
        self.max_effects = max_effects
        self.max_replay_memory = max_replay_memory
        self.max_actions_per_effect = max_actions_per_effect
        self.max_changes_per_effect = max_changes_per_effect
        self._effects: deque[RuntimeEffect] = deque()
        # One key per retained effect, in the same order, so a dropped effect
        # takes its de-duplication key with it and neither side can grow past
        # ``max_effects``.
        self._keys: deque[tuple[str, str, str, str]] = deque()
        self._seen: set[tuple[str, str, str, str]] = set()
        # A dropped effect loses its place in ``_seen``, so the journal keeps
        # its key here for a while longer.  Without this, an item that the
        # runtime reports a second time is projected a second time and lands
        # in the receipt with a fresh sequence number, above effects that were
        # really newer than it.  Both stores below are capped by
        # ``max_replay_memory``, so neither grows with the number of effects.
        self._dropped_keys: deque[tuple[str, str, str, str]] = deque()
        self._dropped_key_set: set[tuple[str, str, str, str]] = set()
        # A turn is finished once the same agent reports a later turn on the
        # same thread.  Anything that arrives for a finished turn afterwards is
        # a re-read of history, so it is ignored even when its key has already
        # left ``_dropped_keys``.  This is what covers a resume or a
        # ``thread/read`` that replays a whole old thread.
        self._closed_turns: deque[tuple[str, str, str]] = deque()
        self._closed_turn_set: set[tuple[str, str, str]] = set()
        self._current_turn: dict[tuple[str, str], str] = {}
        # For each turn the journal still tracks, whether every key that turn
        # produced is still remembered.  While that holds, an unknown key on a
        # closed turn cannot be a repeat of one of them, so it is work never
        # reported before and the journal keeps it.  Once one of the turn's keys
        # has been forgotten, or the turn itself has left this store, the
        # journal cannot tell a late item from an ancient replay, and it says so
        # instead of discarding the item in silence.  An entry
        # lives as long as the turn it describes, so this store holds at most
        # one entry per closed turn plus one per current turn, which is
        # ``2 * max_replay_memory``.
        self._turn_keys_intact: dict[tuple[str, str, str], bool] = {}
        self._replayed_items = 0
        self._ambiguous_items = 0
        self._dropped_effects = 0
        self._sequence = 0
        self._malformed_items = 0
        self._uncorrelated_items = 0
        self._readers: dict[str, RuntimeEffectReader] = {}
        self._lock = threading.RLock()

    def bind_reader(self, agent_id: str, reader: RuntimeEffectReader) -> None:
        """Bind an agent to the decoder selected with its runtime adapter."""

        with self._lock:
            existing = self._readers.get(agent_id)
            if existing is not None and existing.provider != reader.provider:
                raise RuntimeEffectProjectionError(
                    "an agent cannot change native effect providers"
                )
            self._readers[agent_id] = reader

    def inherit_reader(self, *, parent_agent_id: str, child_agent_id: str) -> None:
        """Use the selected decoder for a child owned by the same adapter."""
        with self._lock:
            reader = self._readers.get(parent_agent_id)
            if reader is None:
                raise RuntimeEffectProjectionError("native parent effect reader is not bound")
            self.bind_reader(child_agent_id, reader)

    def observe_turn(
        self,
        *,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[RuntimeEffect, ...]:
        """Project new authoritative effects for exactly one runtime turn."""

        projected: list[RuntimeEffect] = []
        with self._lock:
            reader = self._readers.get(agent_id)
            if reader is None:
                raise RuntimeEffectProjectionError(
                    "native effect reader was not selected with the runtime adapter"
                )
            self._follow_turn(agent_id, thread_id, turn_id)
            for event in events:
                candidate = reader.completed_effect(event)
                if candidate is None:
                    continue
                if candidate.malformed:
                    self._malformed_items += 1
                    continue
                if candidate.thread_ref != thread_id or candidate.turn_ref != turn_id:
                    self._uncorrelated_items += 1
                    continue
                if candidate.item_ref is None or candidate.item is None:
                    self._malformed_items += 1
                    continue
                key = (agent_id, thread_id, turn_id, candidate.item_ref)
                if key in self._seen:
                    continue
                if key in self._dropped_key_set:
                    # Projected once already; its effect has since been dropped.
                    self._replayed_items += 1
                    continue
                turn = (agent_id, thread_id, turn_id)
                if turn in self._closed_turn_set and not self._turn_keys_intact.get(
                    turn, False
                ):
                    # The turn is over and the journal no longer holds all of
                    # its keys, so this item may be a repeat of one it has
                    # forgotten or may be late work.  It is suppressed, and the
                    # ambiguous count with ``evidence_complete`` report that the
                    # journal is unsure which of the two it was.
                    self._replayed_items += 1
                    self._ambiguous_items += 1
                    continue
                # A closed turn whose keys are all still remembered cannot be
                # the source of an unknown key, so what follows is new work.
                effect = self._project_item(
                    agent_id,
                    reader.provider,
                    candidate.item_ref,
                    candidate.item,
                )
                if effect is None:
                    self._malformed_items += 1
                    continue
                # A full journal drops its oldest effects rather than failing
                # the turn.  Bookkeeping that runs out of room must never cost
                # the worker the work it was recording.
                self._make_room()
                self._sequence += 1
                self._seen.add(key)
                self._keys.append(key)
                self._effects.append(effect)
                projected.append(effect)
        return tuple(projected)

    def _make_room(self) -> None:
        """Drop oldest effects until one more fits. Caller holds the lock."""

        while len(self._effects) >= self.max_effects:
            self._effects.popleft()
            key = self._keys.popleft()
            self._seen.discard(key)
            self._remember_dropped_key(key)
            self._dropped_effects += 1

    def _remember_dropped_key(self, key: tuple[str, str, str, str]) -> None:
        """Keep a dropped effect's key so a replay of it is still known.

        Caller holds the lock.
        """

        if key in self._dropped_key_set:
            return
        self._dropped_key_set.add(key)
        self._dropped_keys.append(key)
        while len(self._dropped_keys) > self.max_replay_memory:
            forgotten = self._dropped_keys.popleft()
            self._dropped_key_set.discard(forgotten)
            self._forget_turn_key(forgotten)

    def _follow_turn(self, agent_id: str, thread_id: str, turn_id: str) -> None:
        """Note which turn an agent's thread is on. Caller holds the lock."""

        if (agent_id, thread_id, turn_id) in self._closed_turn_set:
            # A re-read of a finished turn does not move the thread backwards.
            return
        pair = (agent_id, thread_id)
        previous = self._current_turn.get(pair)
        if previous == turn_id:
            return
        if previous is not None:
            self._close_turn((agent_id, thread_id, previous))
        self._current_turn[pair] = turn_id
        self._track_turn_keys((agent_id, thread_id, turn_id))
        while len(self._current_turn) > self.max_replay_memory:
            evicted_pair = next(iter(self._current_turn))
            evicted_turn = (
                evicted_pair[0],
                evicted_pair[1],
                self._current_turn.pop(evicted_pair),
            )
            if evicted_turn not in self._closed_turn_set:
                # Its open-turn mark has nothing left to speak for it.
                self._turn_keys_intact.pop(evicted_turn, None)

    def _track_turn_keys(self, turn: tuple[str, str, str]) -> None:
        """Start a turn with all of its keys remembered. Caller holds the lock.

        A turn is only tracked here when the thread first moves onto it, so it
        starts out with no keys at all and nothing to have forgotten.

        The mark is not trimmed by age.  It lives exactly as long as the turn
        it describes: it goes when the turn leaves ``_closed_turns`` or, for a
        turn still open, when its thread leaves ``_current_turn``.  So the two
        records can never disagree about a turn the journal still remembers,
        and the dict holds at most one entry per closed turn plus one per
        current turn, which is ``2 * max_replay_memory``.
        """

        self._turn_keys_intact[turn] = True

    def _forget_turn_key(self, key: tuple[str, str, str, str]) -> None:
        """Note a turn has lost one of its keys. Caller holds the lock."""

        turn = (key[0], key[1], key[2])
        if turn in self._turn_keys_intact:
            self._turn_keys_intact[turn] = False

    def _close_turn(self, turn: tuple[str, str, str]) -> None:
        """Mark one turn finished. Caller holds the lock."""

        if turn in self._closed_turn_set:
            return
        self._closed_turn_set.add(turn)
        self._closed_turns.append(turn)
        while len(self._closed_turns) > self.max_replay_memory:
            forgotten = self._closed_turns.popleft()
            self._closed_turn_set.discard(forgotten)
            if self._current_turn.get((forgotten[0], forgotten[1])) != forgotten[2]:
                self._turn_keys_intact.pop(forgotten, None)

    @property
    def dropped_effect_count(self) -> int:
        """How many effects were dropped to keep the journal in its bounds."""

        with self._lock:
            return self._dropped_effects

    @property
    def replayed_item_count(self) -> int:
        """How many reported items were already projected by an earlier turn."""

        with self._lock:
            return self._replayed_items

    @property
    def ambiguous_item_count(self) -> int:
        """How many suppressed items the journal could not prove were repeats.

        These are the share of ``replayed_item_count`` that arrived for a turn
        already finished, carrying a key the journal no longer remembers.  Each
        one is either a re-read of old history or work reported late, and the
        journal cannot tell which, so it reports the doubt here as well.
        """

        with self._lock:
            return self._ambiguous_items

    def records(self, agent_id: str | None = None) -> tuple[RuntimeEffect, ...]:
        with self._lock:
            if agent_id is None:
                return tuple(self._effects)
            return tuple(value for value in self._effects if value.agent_id == agent_id)

    def summary(self, agent_id: str | None = None) -> dict[str, Any]:
        records = self.records(agent_id)
        files = sorted(
            {
                change.path
                for effect in records
                for change in effect.changes
                if change.path != "<outside-workspace>"
            }
        )
        with self._lock:
            malformed = self._malformed_items
            uncorrelated = self._uncorrelated_items
            dropped = self._dropped_effects
            replayed = self._replayed_items
            ambiguous = self._ambiguous_items
        return {
            "effect_count": len(records),
            "command_count": sum(value.effect == "command" for value in records),
            "file_change_count": sum(value.effect == "file_change" for value in records),
            "limited_count": sum(value.evidence_limited for value in records),
            "files": files,
            "malformed_item_count": malformed,
            "uncorrelated_item_count": uncorrelated,
            "dropped_effect_count": dropped,
            "replayed_item_count": replayed,
            "ambiguous_item_count": ambiguous,
            "evidence_complete": dropped == 0 and ambiguous == 0,
        }

    def receipt(self, aliases: Mapping[str, str]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for effect in self.records():
            values.append(
                {
                    "sequence": effect.sequence,
                    "agent": aliases.get(effect.agent_id, "unknown"),
                    "provider": effect.provider,
                    "item_ref": effect.item_ref,
                    "effect": effect.effect,
                    "status": effect.status,
                    "evidence_limited": effect.evidence_limited,
                    "not_reported": {
                        "provider": effect.provider,
                        "fields": list(effect.not_reported_fields),
                    },
                    "exit_code": effect.exit_code,
                    "duration_ms": effect.duration_ms,
                    "cwd": effect.cwd,
                    "action_types": list(effect.action_types),
                    "changes": [
                        {"path": value.path, "kind": value.kind}
                        for value in effect.changes
                    ],
                }
            )
        return values

    def _project_item(
        self,
        agent_id: str,
        provider: str,
        item_id: str,
        item: Mapping[str, Any],
    ) -> RuntimeEffect | None:
        item_type = item.get("type")
        item_ref = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]
        # A monotonic counter rather than the retained length, so a sequence
        # is never reused after a drop and the first number a reader holds
        # shows how much came before it.
        sequence = self._sequence + 1
        if item_type == "command":
            not_reported, not_reported_limited = self._not_reported(
                item.get("not_reported"),
                frozenset({"cwd", "actions", "exit_code", "duration_ms", "status"}),
            )
            status, limited = self._declared_status(item, not_reported, self._COMMAND_STATUSES)
            # Same rule as the status field: a boundary cannot declare a field
            # absent and send it anyway.  The declared field is still dropped,
            # but the contradiction is recorded as degraded evidence.
            declared_absent_but_sent = any(
                self._contradicts_absence(item.get(field))
                for field in ("cwd", "actions", "exit_code", "duration_ms")
                if field in not_reported
            )
            cwd, outside = (None, False) if "cwd" in not_reported else self._workspace_path(item.get("cwd"))
            actions, actions_limited = (
                ((), False)
                if "actions" in not_reported
                else self._action_types(item.get("actions"))
            )
            exit_code = item.get("exit_code")
            duration_ms = item.get("duration_ms")
            if "exit_code" in not_reported:
                exit_code = None
            elif not isinstance(exit_code, int):
                exit_code = None
            if "duration_ms" in not_reported:
                duration_ms = None
            elif not isinstance(duration_ms, int) or duration_ms < 0:
                duration_ms = None
            return RuntimeEffect(
                sequence=sequence,
                agent_id=agent_id,
                provider=provider,
                item_ref=item_ref,
                effect="command",
                status=status,
                evidence_limited=(
                    limited
                    or outside
                    or actions_limited
                    or not_reported_limited
                    or declared_absent_but_sent
                    or item.get("evidence_limited") is True
                ),
                not_reported_fields=not_reported,
                exit_code=exit_code,
                duration_ms=duration_ms,
                cwd=cwd,
                action_types=actions,
            )
        if item_type == "file":
            not_reported, not_reported_limited = self._not_reported(
                item.get("not_reported"), frozenset({"change_kind", "status"})
            )
            status, limited = self._declared_status(item, not_reported, self._FILE_STATUSES)
            changes, changes_limited = self._changes(item.get("changes"))
            # A change kind declared not-reported that nonetheless arrives is
            # the same contradiction: the recognised kind is still projected,
            # but it no longer passes as unlimited evidence.
            raw_changes = item.get("changes")
            kind_declared_absent_but_sent = "change_kind" in not_reported and isinstance(
                raw_changes, list
            ) and any(
                isinstance(change, Mapping)
                and change.get("kind") is not None
                and change.get("kind") != {"type": "not_reported"}
                for change in raw_changes[: self.max_changes_per_effect]
            )
            return RuntimeEffect(
                sequence=sequence,
                agent_id=agent_id,
                provider=provider,
                item_ref=item_ref,
                effect="file_change",
                status=status,
                evidence_limited=(
                    limited
                    or changes_limited
                    or not_reported_limited
                    or kind_declared_absent_but_sent
                    or item.get("evidence_limited") is True
                ),
                not_reported_fields=not_reported,
                changes=changes,
            )
        return None

    @staticmethod
    def _not_reported(
        value: Any, allowed: frozenset[str]
    ) -> tuple[tuple[str, ...], bool]:
        if value is None:
            return (), False
        if not isinstance(value, list):
            return (), True
        result: list[str] = []
        for field in value:
            if not isinstance(field, str) or not field or field not in allowed or field in result:
                return (), True
            result.append(field)
        return tuple(result), False

    @classmethod
    def _declared_status(
        cls,
        item: Mapping[str, Any],
        not_reported: tuple[str, ...],
        allowed: frozenset[str],
    ) -> tuple[str, bool]:
        """Separate a status the provider never reported from a bad decode.

        A boundary that declares ``status`` not-reported is stating a fact
        about the provider's surface.  Recorded as ``unknown``, that fact
        alone does not mark the evidence limited, which is reserved for a
        value that arrived and could not be trusted.  The declaration only
        stands when no status actually arrived: the field absent, ``None``,
        or the boundary's own ``"unknown"`` placeholder.  Any other value —
        recognised or not — was sent while being declared absent, which is
        contradictory evidence and fails closed.  Declaring a field absent
        therefore never launders a value the provider did in fact send.
        """

        if "status" in not_reported:
            if cls._contradicts_absence(item.get("status")):
                return "unknown", True
            return "unknown", False
        return cls._status(item.get("status"), allowed)

    @staticmethod
    def _contradicts_absence(value: Any) -> bool:
        """Report whether a field declared not-reported nonetheless carried one.

        ``None`` and the boundary's ``"unknown"`` placeholder are how an
        absent provider field reaches this journal, so neither contradicts
        the declaration.  Everything else does.
        """

        return value is not None and value != "unknown"

    @staticmethod
    def _status(value: Any, allowed: frozenset[str]) -> tuple[str, bool]:
        if isinstance(value, str) and value in allowed:
            return value, False
        return "unknown", True

    def _action_types(self, value: Any) -> tuple[tuple[str, ...], bool]:
        if not isinstance(value, list):
            return (), True
        limited = len(value) > self.max_actions_per_effect
        result: list[str] = []
        for action in value[: self.max_actions_per_effect]:
            raw_type = action.get("type") if isinstance(action, Mapping) else None
            if isinstance(raw_type, str) and raw_type in self._ACTION_TYPES:
                result.append(raw_type)
            else:
                result.append("unknown")
                limited = True
        return tuple(result), limited

    def _changes(self, value: Any) -> tuple[tuple[RuntimeFileChange, ...], bool]:
        if not isinstance(value, list):
            return (), True
        limited = len(value) > self.max_changes_per_effect
        result: list[RuntimeFileChange] = []
        for change in value[: self.max_changes_per_effect]:
            if not isinstance(change, Mapping):
                limited = True
                continue
            path, outside = self._workspace_path(change.get("path"))
            kind_value = change.get("kind")
            raw_kind = kind_value.get("type") if isinstance(kind_value, Mapping) else None
            if not isinstance(raw_kind, str) or raw_kind not in self._CHANGE_KINDS:
                raw_kind = "unknown"
                limited = True
            if path is None:
                path = "<unknown>"
                limited = True
            result.append(RuntimeFileChange(path=path, kind=raw_kind))
            limited = limited or outside
        return tuple(result), limited

    def _workspace_path(self, value: Any) -> tuple[str | None, bool]:
        if not isinstance(value, str) or not value:
            return None, True
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            relative = candidate.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return "<outside-workspace>", True
        rendered = relative.as_posix()
        return rendered if rendered else ".", False
