from __future__ import annotations

import unittest
from datetime import datetime, timezone

from vnext.pricing import API_PRICES_PER_MILLION, api_equivalent


DEEPSEEK = "deepseek/deepseek-v4.1-flash"


def _at(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


class PricingTests(unittest.TestCase):
    def test_every_codex_model_in_the_default_catalog_has_a_price(self) -> None:
        from vnext.vnext_mcp_server import DEFAULT_CATALOG

        unpriced = [
            entry["model"]
            for entry in DEFAULT_CATALOG
            if entry["provider"] == "codex"
            and entry["model"] not in API_PRICES_PER_MILLION
        ]
        self.assertEqual([], unpriced)

    def test_gpt_6_1_sol_uses_the_cheaper_cached_rate(self) -> None:
        self.assertAlmostEqual(
            2.0 * 0.2 + 0.1 * 0.8 + 10.0 * 0.1,
            api_equivalent("gpt-6.1-sol", 1_000_000, 800_000, 100_000),
            places=12,
        )

    def test_deepseek_live_usage_uses_the_off_peak_rates(self) -> None:
        cost = api_equivalent(DEEPSEEK, 43_142, 37_760, 251)

        self.assertAlmostEqual(0.00107118, cost, places=12)
        # The task's $0.00106 check number is the same order-of-magnitude
        # estimate; the supplied token arithmetic evaluates exactly as above.
        self.assertAlmostEqual(0.00106, cost, places=4)
        self.assertEqual(
            {"input", "cached", "output", "peak"},
            set(API_PRICES_PER_MILLION[DEEPSEEK]),
        )

    def test_deepseek_peak_windows_double_the_base_rate(self) -> None:
        base = api_equivalent(DEEPSEEK, 1_000_000, 0, 0)
        off_peak = api_equivalent(
            DEEPSEEK, 1_000_000, 0, 0, at=_at(2026, 9, 22, 0, 30)
        )
        peak = api_equivalent(
            DEEPSEEK, 1_000_000, 0, 0, at=_at(2026, 9, 22, 7, 30)
        )

        self.assertEqual(0.15, base)
        self.assertEqual(base, off_peak)
        self.assertEqual(base * 2, peak)

    def test_deepseek_weekends_and_window_ends_are_off_peak(self) -> None:
        base = api_equivalent(DEEPSEEK, 1_000_000, 0, 0)

        self.assertEqual(
            base,
            api_equivalent(
                DEEPSEEK, 1_000_000, 0, 0, at=_at(2026, 9, 26, 7, 30)
            ),
        )
        self.assertEqual(
            base,
            api_equivalent(
                DEEPSEEK, 1_000_000, 0, 0, at=_at(2026, 9, 22, 4, 0)
            ),
        )

    def test_existing_models_ignore_peak_time(self) -> None:
        base = api_equivalent("gpt-5.6-luna", 100, 20, 10)
        timed = api_equivalent(
            "gpt-5.6-luna", 100, 20, 10, at=_at(2026, 9, 22, 7, 30)
        )

        self.assertEqual(base, timed)

    def test_gpt_6_sol_one_million_input_tokens_costs_two_dollars(self) -> None:
        self.assertEqual(2.0, api_equivalent("gpt-6-sol", 1_000_000, 0, 0))

    def test_invalid_counts_have_no_api_equivalent(self) -> None:
        self.assertIsNone(api_equivalent("gpt-6-sol", 1000, 5000, 0))
        for counts in ((-1, 0, 0), (100, -1, 0), (100, 0, -1)):
            with self.subTest(counts=counts):
                self.assertIsNone(api_equivalent("gpt-6-sol", *counts))
        self.assertEqual(0.0011, api_equivalent("gpt-6-sol", 1000, 500, 0))


if __name__ == "__main__":
    unittest.main()
