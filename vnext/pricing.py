from __future__ import annotations

from datetime import datetime, timezone


# Public API-equivalent USD per million tokens. These are comparison values,
# not Codex subscription invoices; unlisted internal models remain unpriced.
API_PRICES_PER_MILLION = {
    "gpt-6-astra": {"input": 10.0, "cached": 1.0, "output": 50.0},
    # Checked 2026-10-05 against https://developers.openai.com/api/docs/pricing.
    # Requests over 272K input tokens pay 2x input and 1.5x output; this is not modelled.
    "gpt-6.1-sol": {"input": 2.0, "cached": 0.1, "output": 10.0},
    "gpt-6-sol": {"input": 2.0, "cached": 0.2, "output": 10.0},
    "gpt-6-luna": {"input": 0.1, "cached": 0.01, "output": 0.5},
    "gpt-5.6-sol": {"input": 5.0, "cached": 0.5, "output": 30.0},
    "gpt-5.6-terra": {"input": 2.0, "cached": 0.2, "output": 12.0},
    "gpt-5.6-luna": {"input": 0.2, "cached": 0.02, "output": 1.2},
    # Checked 2026-09-22 against https://api-docs.deepseek.com/quick_start/pricing
    # and https://commandcode.ai/models/deepseek-v4-1-flash.  All three rates
    # double in these weekday UTC windows; Chinese public holidays are not modeled.
    "deepseek/deepseek-v4.1-flash": {
        "input": 0.15,
        "cached": 0.003,
        "output": 0.60,
        "peak": {
            "multiplier": 2.0,
            "hours_utc": ((1, 4), (6, 10)),
            "weekdays": True,
        },
    },
}


def api_equivalent(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    *,
    at: float | None = None,
) -> float | None:
    prices = API_PRICES_PER_MILLION.get(model)
    if not prices or min(input_tokens, cached_input_tokens, output_tokens) < 0:
        return None
    if cached_input_tokens > input_tokens:
        return None
    multiplier = 1.0
    peak = prices.get("peak")
    if at is not None and isinstance(peak, dict):
        moment = datetime.fromtimestamp(at, timezone.utc)
        is_weekday = not peak.get("weekdays") or moment.weekday() < 5
        hours = peak.get("hours_utc") or ()
        if is_weekday and any(start <= moment.hour < end for start, end in hours):
            multiplier = peak["multiplier"]
    uncached = input_tokens - cached_input_tokens
    return multiplier * (
        uncached * prices["input"]
        + cached_input_tokens * prices["cached"]
        + output_tokens * prices["output"]
    ) / 1_000_000
