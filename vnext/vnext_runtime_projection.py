"""Project recorded native events without inventing usage or shared context."""
from __future__ import annotations

from typing import Any, Mapping

from .host_contract import RuntimeEvent


def next_native_cursor(provider: str, previous: Any, events: list[Mapping[str, Any]]) -> Any:
    if not events:
        return previous
    # Claude Agent SDK cursors are inclusive; Codex cursors are zero-based
    # event offsets. z.ai uses the same SDK transport and event schema, and
    # Command Code the same app-server transport as Codex.
    if provider in {"claude", "zai"}:
        for event in reversed(events):
            cursor = event.get("cursor")
            if isinstance(cursor, int) and not isinstance(cursor, bool):
                return cursor + 1
        # The Claude bridge stamps every event the adapter retains. Live
        # polling cannot loop here; this fallback is for malformed batches.
        return previous
    return (previous or 0) + len(events)


_COST_TOKEN_FIELDS = ("cache_read_tokens", "input_tokens", "output_tokens", "total_tokens")


def _normalized_cost(provider: str, tokens: Any) -> dict[str, Any] | None:
    """Say the same four numbers in one set of names, whoever reported them.

    ``input_tokens`` is fresh input, excluding ``cache_read_tokens`` for every
    provider. SDK cache creation is included in ``total_tokens``.

    ``basis`` says what kind of number those four are.  Codex and Command Code
    report a running total for the whole provider thread, so their block is
    already the agent's life to date; the Claude Agent SDK shapes report one
    call.  Adding a thread total onto a stored one would count every earlier
    call again, so the apply site needs to be told which it just received.
    """
    if not isinstance(tokens, Mapping):
        return None
    # Command Code has emitted no usage record in any local journal; verify its
    # field shape when one appears before treating its thread basis as observed.
    codex_shaped = provider in {"codex", "commandcode"}
    # Fourth name differs: Codex reports a total, the SDK providers report
    # cache creation and no total at all, so the total is summed below.
    names = (("cachedInputTokens", "inputTokens", "outputTokens", "totalTokens") if codex_shaped
             else ("cache_read_input_tokens", "input_tokens", "output_tokens", "cache_creation_input_tokens"))
    found = {name: tokens[name] for name in names
             if isinstance(tokens.get(name), int) and not isinstance(tokens.get(name), bool)}
    if not found:
        return None
    cache_read, inputs, outputs, fourth = (found.get(name, 0) for name in names)
    if codex_shaped:
        inputs = max(0, inputs - cache_read)
    return {"cache_read_tokens": cache_read, "input_tokens": inputs, "output_tokens": outputs,
            "total_tokens": fourth if codex_shaped else cache_read + inputs + outputs + fourth,
            "basis": "thread" if codex_shaped else "call"}


def accumulate_cost(stored: Any, incoming: Any) -> dict[str, Any] | None:
    """Carry a cost block forward so the stored one is the agent's running total.

    A ``thread`` block replaces what is stored, because the provider has
    already done the adding.  A ``call`` block is added onto a stored ``call``
    block field by field.  When the two bases disagree the incoming block
    stands alone: one agent reports through one provider, so a disagreement
    means the stored block came from somewhere else and summing the two would
    produce a number that is neither of them.
    """
    if not isinstance(incoming, Mapping):
        return dict(stored) if isinstance(stored, Mapping) else None
    carried = dict(incoming)
    if carried.get("basis") != "call":
        return carried
    if not isinstance(stored, Mapping) or stored.get("basis") != "call":
        return carried
    for field in _COST_TOKEN_FIELDS:
        previous = stored.get(field)
        current = carried.get(field)
        if (isinstance(previous, int) and not isinstance(previous, bool)
                and isinstance(current, int) and not isinstance(current, bool)):
            carried[field] = previous + current
    return carried


def project_native_event(provider: str, agent_id: str, event: Mapping[str, Any]) -> list[RuntimeEvent]:
    params = event.get("params", event)
    if not isinstance(params, Mapping):
        return []
    turn = params.get("turnId") or params.get("turn_reference")
    result = [RuntimeEvent("runtime.event", {"provider": provider, "event": dict(event)}, agent_id, turn)]
    if provider in {"codex", "commandcode"}:
        method = event.get("method", "")
        item = params.get("item")
        item = item if isinstance(item, Mapping) else {}
        item_type = item.get("type")
        if method == "item/agentMessage/delta":
            result.append(RuntimeEvent("content.delta", {"item_id": params.get("itemId"),
                "delta": params.get("delta", ""), "role": "assistant"}, agent_id, turn))
        elif method == "item/completed" and item_type == "agentMessage":
            result.append(RuntimeEvent("content.final", {"item_id": item.get("id"),
                "role": "assistant", "blocks": [{"type": "text", "text": item.get("text", "")}]}, agent_id, turn))
        elif method in {"item/started", "item/completed"} and item_type not in {None, "agentMessage", "userMessage"}:
            result.append(RuntimeEvent("tool.lifecycle", {"phase": method.split("/")[-1],
                "item_id": item.get("id"), "tool": item_type, "item": dict(item)}, agent_id, turn))
        elif method == "turn/completed":
            # A Codex-route turn that failed says why on the turn itself.  No
            # other record carried it, so a Command Code worker that died on a
            # 401 read as a clean session in vnext-report.
            ended = params.get("turn")
            ended = ended if isinstance(ended, Mapping) else {}
            if ended.get("status") == "failed":
                error = ended.get("error")
                said = error.get("message") if isinstance(error, Mapping) else error
                result.append(RuntimeEvent("provider.error", {"provider": provider,
                    "error": said if isinstance(said, str) and said.strip()
                        else "the provider ended the turn as failed",
                    "native": dict(error) if isinstance(error, Mapping) else None},
                    agent_id, ended.get("id") or turn))
        elif method == "thread/tokenUsage/updated":
            usage = params.get("tokenUsage", {})
            tokens = usage.get("total") if isinstance(usage, Mapping) else None
            cost_tokens = _normalized_cost(provider, tokens)
            result.append(RuntimeEvent("usage.updated", {"provider": provider,
                "source": "codex.thread_token_usage", "attribution_scope": "self",
                "aggregation": "cumulative", "aggregation_window": "provider_thread",
                "native": usage, "tokens": tokens,
                "context": {"window": usage.get("modelContextWindow"), "last": usage.get("last")}
                    if isinstance(usage, Mapping) else None, "cost": None,
                **({"cost_tokens": cost_tokens} if cost_tokens is not None else {})}, agent_id, turn))
    elif provider in {"claude", "zai"}:
        name = params.get("name")
        if name in {"message", "content"}:
            message = params.get("message", params.get("content", {}))
            if isinstance(message, Mapping):
                role = message.get("role", "assistant")
                blocks = message.get("content", message.get("blocks", []))
                result.append(RuntimeEvent("content.final", {"role": role, "blocks": blocks,
                    "native": dict(message)}, agent_id, turn))
        elif name == "stream":
            result.append(RuntimeEvent("content.delta", {"native": dict(params)}, agent_id, turn))
        elif name in {"tool_use", "tool_result"}:
            result.append(RuntimeEvent("tool.lifecycle", {"phase": name, "native": dict(params)}, agent_id, turn))
        elif name == "native_child" and params.get("status") == "dropped-from-identity-join":
            result.append(RuntimeEvent("runtime.note", {
                "provider": provider, "note": params.get("note"),
                "native_child_agent_id": params.get("agent_id"),
            }, agent_id, turn))
        elif name == "provider_error":
            result.append(RuntimeEvent("provider.error", {"provider": provider,
                "error": params.get("provider_error"), "native": dict(params)}, agent_id, turn))
        elif name == "native_child_completed" and isinstance(params.get("summary"), str) and params["summary"]:
            result.append(RuntimeEvent("content.final", {"role": "assistant",
                "blocks": [{"type": "text", "text": params["summary"]}],
                "source": "provider-task-summary"}, agent_id, turn))
        elif name in {"usage", "native_child_usage"} or (name == "result" and ("usage" in params or "total_cost_usd" in params)):
            native_task = name == "native_child_usage"
            aggregate_metadata = {"source": "claude.result", "attribution_scope": "aggregate",
                "aggregation": "cumulative", "aggregation_window": "provider_call"}
            cost_tokens = _normalized_cost(provider, params.get("usage"))
            # Already cumulative where it is reported at all, and Codex reports
            # no dollar figure.  A zero would read as free, so the key is left
            # out entirely rather than defaulted.
            reported_usd = params.get("total_cost_usd")
            has_usd = isinstance(reported_usd, (int, float)) and not isinstance(reported_usd, bool)
            result.append(RuntimeEvent("usage.updated", {"provider": provider,
                "source": "claude.native_task" if native_task else "claude.result",
                "attribution_scope": "unknown" if native_task else "self", "aggregation": "cumulative",
                "aggregation_window": "native_task" if native_task else "turn",
                "tokens": params.get("usage"), "cost": params.get("total_cost_usd"),
                "cost_metadata": {**aggregate_metadata, "estimate": True} if not native_task else None,
                "model_usage_metadata": aggregate_metadata if not native_task else None,
                "model_usage": params.get("model_usage"), "native": dict(params),
                **({"cost_tokens": cost_tokens} if cost_tokens is not None else {}),
                **({"cost_usd": reported_usd} if has_usd else {})}, agent_id, turn))
    return result
