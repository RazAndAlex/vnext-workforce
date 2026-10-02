"""Provider usage scopes survive projection without false additive claims."""
import unittest

from vnext.vnext_runtime_projection import project_native_event


class UsageSemanticsTests(unittest.TestCase):
    def usage(self, provider, event):
        return next(item.payload for item in project_native_event(provider, "agent", event)
                    if item.type == "usage.updated")

    def test_codex_total_is_thread_cumulative_not_turn_cost(self):
        usage = self.usage("codex", {"method": "thread/tokenUsage/updated", "params": {
            "threadId": "native", "turnId": "turn", "tokenUsage": {
                "total": {"totalTokens": 100}, "last": {"totalTokens": 20}}}})
        self.assertEqual("provider_thread", usage["aggregation_window"])
        self.assertEqual("cumulative", usage["aggregation"])
        self.assertEqual("self", usage["attribution_scope"])
        self.assertEqual({"totalTokens": 100}, usage["tokens"])
        self.assertIsNone(usage["cost"])

    def test_claude_tokens_and_cost_have_different_scopes(self):
        usage = self.usage("claude", {"name": "usage", "turn_reference": "turn",
            "usage": {"input_tokens": 100, "output_tokens": 20}, "total_cost_usd": .2,
            "model_usage": {"model": {"inputTokens": 300}}})
        self.assertEqual("self", usage["attribution_scope"])
        self.assertEqual("turn", usage["aggregation_window"])
        self.assertEqual("aggregate", usage["cost_metadata"]["attribution_scope"])
        self.assertEqual("provider_call", usage["cost_metadata"]["aggregation_window"])
        self.assertTrue(usage["cost_metadata"]["estimate"])
        self.assertEqual("aggregate", usage["model_usage_metadata"]["attribution_scope"])

    def test_native_task_progress_is_not_a_token_delta(self):
        usage = self.usage("claude", {"name": "native_child_usage", "usage": {"total_tokens": 12}})
        self.assertEqual("native_task", usage["aggregation_window"])
        self.assertEqual("cumulative", usage["aggregation"])
        self.assertEqual("claude.native_task", usage["source"])
        self.assertEqual("unknown", usage["attribution_scope"])
        self.assertIsNone(usage["cost_metadata"])
