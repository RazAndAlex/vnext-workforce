"""Which exact model a worker asked for and which one really answered.

A catalog name such as ``opus`` is an alias: the provider decides which model
it means on the day.  For a whole review loop ``opus`` quietly ran an older
model while every record said only ``opus``.  This module holds the small,
provider-neutral rules that turn an alias into the exact id the provider
reported, so no record is left with only the alias.

Three ids are kept apart:

* ``model_id`` stays the alias.  It is a selector that managers pass back in.
* ``model_exact`` is the id the provider said the alias resolves to.
* ``model_ran`` is the id the provider reported on the worker's own replies.

When ``model_ran`` and ``model_exact`` differ the row says so.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

UNKNOWN = "exact model unknown"
PENDING = "pending: resolved when the worker connects"

_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[str, str]] = {}


def resolve_claude_alias(alias: str, rows: object) -> tuple[str | None, str | None]:
    """Resolve one Claude alias against the CLI's ``server_info["models"]``.

    The row whose ``value`` is the alias wins.  The CLI lists rows only for
    some aliases, so an alias with no row takes the single row whose
    ``resolvedModel`` belongs to its family (``claude-<alias>-...``).  Zero or
    several family rows leave it unresolved.
    """

    if not isinstance(alias, str) or not alias or not isinstance(rows, Sequence):
        return None, None
    table = [row for row in rows if isinstance(row, Mapping)]
    for row in table:
        resolved = row.get("resolvedModel")
        if row.get("value") == alias and isinstance(resolved, str) and resolved:
            return resolved, "server_info"
    if alias.startswith("claude-"):
        family = alias
    else:
        family = f"claude-{alias}-"
    matches = sorted({
        row["resolvedModel"]
        for row in table
        if isinstance(row.get("resolvedModel"), str)
        and (row["resolvedModel"] == alias or row["resolvedModel"].startswith(family))
    })
    if len(matches) == 1:
        return matches[0], "family_match"
    return None, None


def model_table(rows: object) -> list[dict[str, str]]:
    """Keep only the alias and its exact id from each server_info row."""

    if not isinstance(rows, Sequence):
        return []
    return [
        {"value": str(row.get("value")), "resolvedModel": str(row.get("resolvedModel"))}
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("resolvedModel"), str)
    ]


def remember(provider: str, alias: str, exact: str | None, source: str | None) -> None:
    if isinstance(exact, str) and exact and isinstance(alias, str) and alias:
        with _cache_lock:
            _cache[(str(provider or ""), alias)] = (exact, str(source or ""))


def remember_claude_table(provider: str, rows: object, aliases: Sequence[str] = ()) -> None:
    """Cache every alias the table names, plus the asked-for aliases it implies."""

    for row in model_table(rows):
        remember(provider, row["value"], row["resolvedModel"], "server_info")
    for alias in aliases:
        exact, source = resolve_claude_alias(alias, rows)
        remember(provider, alias, exact, source)


def cached_exact(provider: str, alias: str) -> str | None:
    with _cache_lock:
        value = _cache.get((str(provider or ""), str(alias)))
    return value[0] if value else None


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def identity_view(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten one adapter identity into the keys every view and row carry.

    An empty identity gives an empty dict, so a view of an agent that never
    reached a provider keeps its old shape.
    """

    if not isinstance(identity, Mapping) or not identity:
        return {}
    exact = identity.get("model_exact")
    source = identity.get("model_exact_source")
    ran = identity.get("model_ran")
    ran = ran if isinstance(ran, str) and ran else None
    if not isinstance(exact, str) or not exact:
        first = identity.get("model_ran_first")
        first = first if isinstance(first, str) and first else ran
        if first:
            exact, source = first, "first_reply"
        else:
            error = identity.get("model_exact_error")
            if not isinstance(error, str) or not error:
                error = "the provider's model table does not name this alias, and no reply has named a model yet"
            exact = f"{UNKNOWN}: {error}"
            source = None
    view: dict[str, Any] = {"model_exact": exact, "model_exact_source": source}
    if ran is not None:
        view["model_ran"] = ran
        known = not exact.startswith(UNKNOWN)
        view["model_mismatch"] = bool(known and ran != exact)
        if view["model_mismatch"]:
            view["model_note"] = f"asked for {exact}, the provider ran {ran}"
    history = identity.get("model_exact_history")
    if isinstance(history, list) and history:
        view["model_exact_history"] = list(history)
    reroutes = identity.get("model_reroutes")
    if isinstance(reroutes, list) and reroutes:
        view["model_reroutes"] = [dict(item) for item in reroutes if isinstance(item, Mapping)]
    return view


def read_identity(adapter: object, thread_id: str | None) -> dict[str, Any]:
    """Ask an adapter for one thread's identity; any failure gives nothing."""

    reader = getattr(adapter, "model_identity", None)
    if not callable(reader) or not thread_id:
        return {}
    try:
        value = reader(thread_id)
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}
