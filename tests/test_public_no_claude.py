"""Public-build regressions for the private Claude worker boundary.

These tests also run from the private tree. Each runtime case lowers the switch
locally, so the private default remains unchanged for the rest of the suite.
"""
from __future__ import annotations

import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import suite_environment  # noqa: F401  # unittest does not read conftest.py
from vnext import vnext_mcp_server as server
from vnext import vnext_session_runtime as runtime
from vnext.vnext_scheduler import VNextScheduler


DENIAL = (
    "Claude workers are not part of the public vNext build. "
    "Use Claude Code's own subagents for Claude work."
)


class PublicRuntimeGuardTests(unittest.TestCase):
    def test_explicit_claude_catalog_entries_are_denied_in_session_registry(self) -> None:
        request = SimpleNamespace(
            catalog_config={"models": [{"provider": "claude", "model": "custom-alias"}]},
            primary={"provider": "codex", "model": "manager"},
        )
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False):
            with self.assertRaises(ValueError) as caught:
                runtime.session_registry(request)
        self.assertEqual(DENIAL, str(caught.exception))

    def test_server_denies_claude_catalog_even_with_registered_factory(self) -> None:
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False):
            with self.assertRaises(server.VNextMcpServiceError) as caught:
                server._validate_catalog(
                    [{"provider": "claude", "model_id": "private-alias"}],
                    adapter_factories={"claude": lambda: object()},
                )
        self.assertEqual(DENIAL, str(caught.exception))

    def test_default_catalog_filters_claude_at_import_in_isolated_process(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code = (
            "from vnext import vnext_session_runtime as runtime; "
            "runtime.CLAUDE_SUBSCRIPTION_WORKERS = False; "
            "from vnext import vnext_mcp_server as server; "
            "assert not any(card.get('provider') == 'claude' for card in server.DEFAULT_CATALOG)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=False
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_claude_route_is_denied_while_zai_still_uses_sdk_adapter(self) -> None:
        fake_module = types.ModuleType("vnext.vnext_claude")
        adapter = Mock(name="sdk_adapter")
        fake_module.ClaudeCodeAdapter = Mock(return_value=adapter)
        fake_module.side_runtime_bridge_command = Mock()
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False), patch.dict(
            sys.modules, {"vnext.vnext_claude": fake_module}
        ):
            with self.assertRaises(ValueError) as caught:
                runtime.claude_route_adapter("claude", ".", {})
            self.assertIs(adapter, runtime.claude_route_adapter("zai", ".", {}))
        self.assertEqual(DENIAL, str(caught.exception))
        fake_module.ClaudeCodeAdapter.assert_called_once()
        self.assertEqual("zai", fake_module.ClaudeCodeAdapter.call_args.kwargs["provider"])

    def test_claude_login_check_does_not_resolve_or_run_cli(self) -> None:
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False), patch.object(
            server, "_claude_executable"
        ) as executable, patch.object(server.subprocess, "run") as run:
            self.assertEqual(server.CLAUDE_LOGIN_ABSENT, server._claude_login_state())
        executable.assert_not_called()
        run.assert_not_called()

    def test_served_model_roster_hides_claude_aliases_when_disabled(self) -> None:
        entries = [
            {"provider": "claude", "model": "sonnet"},
            {"provider": "claude", "model": "opus"},
            {"provider": "claude", "model": "fable"},
            {"provider": "zai", "model": "glm-5.3"},
        ]
        tools = [
            {"name": "delegate", "description": "delegate"},
            {"name": "replace", "description": "replace"},
        ]
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False):
            roster = server._model_roster(entries)
            served = server._external_tools(tools, entries)
        for alias in ("sonnet", "opus", "fable"):
            self.assertNotIn(alias, roster)
            self.assertTrue(
                all(alias not in item["description"] for item in served), alias
            )
        self.assertIn("glm-5.3", roster)

    def test_startup_notes_never_probe_or_report_claude_when_catalog_is_empty(self) -> None:
        probe = Mock()
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False), patch.object(
            server, "_claude_sdk_installed", return_value=True
        ), patch.object(server, "_codex_login_problem", return_value="no codex login"), patch.object(
            server, "load_zai_provider", return_value=object()
        ), patch.object(server, "load_commandcode_provider", return_value=object()):
            notes = server.startup_check_notes([], login=probe)
        self.assertFalse(any("claude" in note.lower() for note in notes), notes)
        probe.claude_state.assert_not_called()

    def test_zai_and_commandcode_catalogs_survive_with_sdk_and_keys(self) -> None:
        catalog = [
            {"provider": "zai", "model": "glm-custom"},
            {"provider": "commandcode", "model": "deepseek-custom"},
        ]
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False), patch.object(
            server, "_claude_sdk_installed", return_value=True
        ), patch.object(server, "load_zai_provider", return_value=object()), patch.object(
            server, "load_commandcode_provider", return_value=object()
        ):
            result = server._validate_catalog(catalog)
        self.assertEqual(["zai", "commandcode"], [entry["provider"] for entry in result])


class ManagerToolGuardTests(unittest.TestCase):
    def _harness(self, target_model: str | None = None, *, card_provider: str | None = None):
        root = SimpleNamespace(agent_id="root")
        agents = {"root": root}
        cards = {}
        if target_model is not None:
            agents["target"] = SimpleNamespace(agent_id="target", model_id=target_model)
            if card_provider is not None:
                cards[target_model] = SimpleNamespace(provider=card_provider)
        session = SimpleNamespace(agents=agents)
        harness = SimpleNamespace(
            managed=SimpleNamespace(control=SimpleNamespace(registry=SimpleNamespace(cards=cards))),
            _session=lambda: session,
            _note_manager_tool=Mock(),
        )
        return harness, session

    def test_delegate_and_replace_reject_each_claude_alias_before_mutation(self) -> None:
        with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False):
            for tool in ("delegate", "replace"):
                for model in ("sonnet", "opus", "fable"):
                    with self.subTest(tool=tool, model=model):
                        harness, session = self._harness()
                        before = dict(session.agents)
                        with self.assertRaises(ValueError) as caught:
                            VNextScheduler._apply_manager_tool(
                                harness, "root", tool, {"model_id": model}
                            )
                        self.assertEqual(DENIAL, str(caught.exception))
                        self.assertEqual(before, session.agents)
                        harness._note_manager_tool.assert_not_called()

    def test_retry_rejects_stored_claude_model_and_custom_alias_before_mutation(self) -> None:
        for model, provider in (("sonnet", None), ("private-worker-name", "claude")):
            with self.subTest(model=model):
                harness, session = self._harness(model, card_provider=provider)
                before = dict(session.agents)
                with patch.object(runtime, "CLAUDE_SUBSCRIPTION_WORKERS", False):
                    with self.assertRaises(ValueError) as caught:
                        VNextScheduler._apply_manager_tool(
                            harness, "root", "retry", {"agent_id": "target"}
                        )
                self.assertEqual(DENIAL, str(caught.exception))
                self.assertEqual(before, session.agents)
                harness._note_manager_tool.assert_not_called()
