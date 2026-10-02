from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext import vnext_mcp_server as server
from vnext import vnext_provider_config as provider_config
from vnext.host_contract import SessionStartRequest
from vnext.vnext_claude import ClaudeCodeAdapter
from vnext.vnext_claude_bridge import BridgeError, _Bridge, _CREDENTIAL_OVERRIDES
from vnext.vnext_mcp_server import VNextMcpService
from vnext.vnext_orchestration import AgentRole
from vnext.vnext_provider_config import ZAI_ENDPOINT, load_zai_provider
from vnext.vnext_runtime_projection import next_native_cursor, project_native_event
from vnext.vnext_session_runtime import VNextRuntimeSession

from tests.test_vnext_external_primary import ScriptedWorker


def _provider_document(api_key: str | None = None) -> dict[str, object]:
    providers: dict[str, object] = {}
    if api_key is not None:
        providers["zai"] = {"api_key": api_key}
    return {"providers": providers}


class _ZaiScriptedWorker(ScriptedWorker):
    provider = "zai"
    harness = "claude-agent-sdk"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key
        self.used_provider_credential = False

    def initialize(self, *, timeout: float = 30) -> dict:
        del timeout
        self.used_provider_credential = bool(self._api_key)
        return {
            "provider": self.provider,
            "harness": self.harness,
            "credential_override_rejected": True,
        }

    def tool_registration_attestation(self, thread_id):
        value = super().tool_registration_attestation(thread_id)
        value["model_id"] = "glm-5.2"
        return value

    def thread_identity_attestation(self, thread_id):
        value = super().thread_identity_attestation(thread_id)
        value["endpoint_attestation"] = {
            "endpoint_owner": "z.ai",
            "endpoint": ZAI_ENDPOINT,
            "endpoint_kind": "named-non-anthropic",
            "credential_owner": "z.ai provider",
            "credential_kind": "provider-api-key",
            "anthropic_subscription_credential": False,
        }
        return value


class ZaiProviderConfigTests(unittest.TestCase):
    def test_missing_malformed_and_incomplete_config_are_quietly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = {
                "missing": root / "missing.json",
                "malformed": root / "malformed.json",
                "no-zai": root / "no-zai.json",
                "blank-key": root / "blank-key.json",
            }
            cases["malformed"].write_text("{", encoding="utf-8")
            cases["no-zai"].write_text(
                json.dumps(_provider_document()), encoding="utf-8"
            )
            cases["blank-key"].write_text(
                json.dumps(_provider_document("")), encoding="utf-8"
            )
            for label, path in cases.items():
                with self.subTest(label=label):
                    self.assertIsNone(load_zai_provider(path))

    def test_default_catalog_fails_closed_and_server_starts_without_zai(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            for label, document in (("missing", None), ("no-zai", _provider_document())):
                path = workspace / f"{label}.json"
                if document is not None:
                    path.write_text(json.dumps(document), encoding="utf-8")
                # The Codex login is stubbed: this case is about z.ai leaving
                # the roster, and with no credentials anywhere the roster is
                # empty, which the server refuses before it gets that far.
                with self.subTest(label=label), patch.object(
                    provider_config, "PROVIDER_CONFIG_PATH", path
                ), patch.object(server, "_codex_login_available", return_value=True):
                    catalog = server._load_catalog(None)
                    self.assertNotIn("zai", {entry["provider"] for entry in catalog})
                    service = VNextMcpService(
                        workspace=workspace,
                        event_log=None,
                        status_file=None,
                        outcome_log=None,
                    )
                    try:
                        providers = {
                            card.provider for card in service.session.registry.cards.values()
                        }
                        self.assertNotIn("zai", providers)
                        self.assertNotEqual("failed", service.session.snapshot()["status"])
                    finally:
                        service.close()

    def test_configured_catalog_offers_both_glm_models_with_provider_owned_credential(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            path = workspace / "providers.json"
            path.write_text(
                json.dumps(_provider_document(secrets.token_urlsafe(32))), encoding="utf-8"
            )
            # The Claude Agent SDK is the harness these models run on, and
            # whether it is installed is a fact about the machine: this test is
            # about the credential, so the harness is held present.
            with patch.object(provider_config, "PROVIDER_CONFIG_PATH", path), patch.object(
                server, "_claude_sdk_installed", return_value=True
            ):
                catalog = server._load_catalog(None)
                zai_models = {
                    entry["model"] for entry in catalog if entry.get("provider") == "zai"
                }
                self.assertEqual({"glm-5.3-flash", "glm-5.3", "glm-5.2"}, zai_models)
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=catalog,
                    event_log=None,
                    status_file=None,
                    outcome_log=None,
                )
                try:
                    for model in zai_models:
                        card = service.session.registry.cards[model]
                        self.assertEqual("zai", card.provider)
                        self.assertEqual("claude-agent-sdk", card.harness)
                        self.assertEqual(
                            "z.ai provider API key (~/.vnext/providers.json)",
                            card.credential_location,
                        )
                finally:
                    service.close()

    def test_built_in_zai_runtime_reuses_the_claude_adapter(self) -> None:
        captured: list[dict[str, object]] = []

        class FakeClaudeAdapter:
            def __init__(self, **kwargs: object) -> None:
                captured.append(kwargs)

            def initialize(self) -> dict[str, object]:
                return {}

        with tempfile.TemporaryDirectory() as temp:
            request = SessionStartRequest(
                session_id="zai-adapter-selection",
                workspace=temp,
                primary_agent_id="primary",
                primary={"provider": "zai", "model": "glm-5.2", "effort": "high"},
                main_preset={"instructions": ""},
                catalog_config={"models": [{"provider": "zai", "model": "glm-5.2"}]},
                config={},
            )
            runtime = VNextRuntimeSession(request, lambda _event: None)
            with patch(
                "vnext.vnext_claude.ClaudeCodeAdapter", FakeClaudeAdapter
            ):
                runtime._adapter(runtime.root)
        self.assertEqual(1, len(captured))
        self.assertEqual("zai", captured[0]["provider"])
        self.assertEqual(str(Path(temp).resolve()), captured[0]["workspace"])


class ZaiBridgeTests(unittest.TestCase):
    class _FakeSDK:
        @staticmethod
        def ClaudeAgentOptions(**kwargs: object) -> object:
            return SimpleNamespace(**kwargs)

        @staticmethod
        def HookMatcher(**kwargs: object) -> object:
            return SimpleNamespace(**kwargs)

    def test_sdk_error_does_not_echo_zai_token_in_bridge_response(self) -> None:
        token = secrets.token_urlsafe(48)
        bridge = _Bridge()
        responses: list[dict[str, object]] = []

        async def failing_dispatch(_op: str, _payload: object) -> dict[str, object]:
            raise RuntimeError(f"upstream error accidentally contained {token}")

        def capture_response(
            request_id: int,
            *,
            result: object = None,
            error: str | None = None,
        ) -> None:
            responses.append({"id": request_id, "result": result, "error": error})

        bridge.dispatch = failing_dispatch
        request = json.dumps(
            {"v": 1, "kind": "request", "id": 1, "op": "start_turn", "payload": {}}
        )
        with patch(
            "vnext.vnext_claude_bridge._response",
            side_effect=capture_response,
        ):
            asyncio.run(bridge._serve(request))

        serialized = json.dumps(responses)
        self.assertNotIn(token, serialized)
        self.assertIn("Claude bridge operation failed: RuntimeError", serialized)

    def test_sdk_options_receive_zai_environment_without_mutating_bridge_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            token = secrets.token_urlsafe(48)
            path = workspace / "providers.json"
            path.write_text(json.dumps(_provider_document(token)), encoding="utf-8")
            bridge = _Bridge()
            bridge._load_sdk = lambda: setattr(bridge, "_sdk", self._FakeSDK())
            clean_environment = {
                name: value
                for name, value in os.environ.items()
                if not name.startswith("ANTHROPIC_")
            }
            with patch.object(provider_config, "PROVIDER_CONFIG_PATH", path), patch.dict(
                os.environ, clean_environment, clear=True
            ):
                attestation = asyncio.run(
                    bridge.dispatch(
                        "initialize", {"workspace": str(workspace), "provider": "zai"}
                    )
                )
                self.assertFalse(
                    any(name.startswith("ANTHROPIC_") for name in os.environ)
                )
                self.assertTrue(attestation["credential_override_rejected"])
                self.assertNotIn(token, json.dumps(attestation))
                bridge._reservations["reservation"] = bridge._reservation_state(
                    "auto_review", None, []
                )
                options = bridge._options(
                    model="glm-5.2", resume=None, reservation_id="reservation"
                )
                self.assertEqual(
                    {
                        "ANTHROPIC_BASE_URL": ZAI_ENDPOINT,
                        "ANTHROPIC_AUTH_TOKEN": token,
                    },
                    options.env,
                )
                resumed_options = bridge._options(
                    model="glm-5.2",
                    resume="zai-provider-session",
                    reservation_id="reservation",
                )
                self.assertEqual(options.env, resumed_options.env)
                self.assertFalse(
                    any(name.startswith("ANTHROPIC_") for name in os.environ)
                )
                asyncio.run(bridge.dispatch("close", {}))
                self.assertEqual({}, bridge._sdk_environment)

    def test_existing_credential_guard_still_rejects_bridge_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bridge = _Bridge()
            bridge._load_sdk = lambda: setattr(bridge, "_sdk", self._FakeSDK())
            with patch.dict(os.environ, {_CREDENTIAL_OVERRIDES[1]: "not-a-real-key"}):
                with self.assertRaisesRegex(
                    BridgeError, "credential override environment is forbidden"
                ):
                    asyncio.run(bridge.dispatch("initialize", {"workspace": temp}))

    def test_zai_thread_attestation_names_endpoint_and_provider_credential(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            adapter = ClaudeCodeAdapter(workspace=temp, provider="zai")
            adapter._start_owned_bridge = lambda: None
            adapter._request = lambda op, payload: {
                "provider": "zai",
                "harness": "claude-agent-sdk",
                "tool_support": {"accepted": True, "requires_empty": False},
                "credential_override_rejected": True,
                "endpoint_attestation": {
                    "endpoint_owner": "z.ai",
                    "endpoint": ZAI_ENDPOINT,
                    "endpoint_kind": "named-non-anthropic",
                    "credential_owner": "z.ai provider",
                    "credential_kind": "provider-api-key",
                    "anthropic_subscription_credential": False,
                },
            }
            adapter.initialize()
            identity = adapter.thread_identity_attestation("local-reservation")
            self.assertEqual("zai", identity["provider"])
            endpoint = identity["endpoint_attestation"]
            self.assertEqual("z.ai", endpoint["endpoint_owner"])
            self.assertEqual(ZAI_ENDPOINT, endpoint["endpoint"])
            self.assertEqual("named-non-anthropic", endpoint["endpoint_kind"])
            self.assertEqual("z.ai provider", endpoint["credential_owner"])
            self.assertFalse(endpoint["anthropic_subscription_credential"])

    def test_zai_uses_the_shared_sdk_cursor_and_event_projection(self) -> None:
        events = [{"cursor": 5}, {"cursor": 9}]
        self.assertEqual(10, next_native_cursor("zai", 5, events))

        projected = project_native_event(
            "zai",
            "agent",
            {
                "params": {
                    "name": "message",
                    "turn_reference": "turn",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "projected"}],
                    },
                }
            },
        )
        runtime_event = next(item for item in projected if item.type == "runtime.event")
        content = next(item for item in projected if item.type == "content.final")
        self.assertEqual("zai", runtime_event.payload["provider"])
        self.assertEqual("assistant", content.payload["role"])
        self.assertEqual("projected", content.payload["blocks"][0]["text"])

    def test_zai_token_is_absent_from_run_log_after_a_configured_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            token = secrets.token_urlsafe(48)
            path = workspace / "providers.json"
            path.write_text(json.dumps(_provider_document(token)), encoding="utf-8")
            log = workspace / "run.jsonl"
            worker = _ZaiScriptedWorker(load_zai_provider(path).api_key)
            with patch.object(provider_config, "PROVIDER_CONFIG_PATH", path):
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "zai", "model": "glm-5.2"}],
                    event_log=log,
                    status_file=workspace / "status.json",
                    outcome_log=workspace / "outcome.jsonl",
                    adapter_factories={"zai": lambda: worker},
                )
            try:
                created = json.loads(
                    service.session.external_tool_call(
                        tool="delegate",
                        arguments={
                            "role": AgentRole.WORKER.value,
                            "model_id": "glm-5.2",
                            "objective": "Complete a credential-isolation probe",
                            "task_contract": {"criteria": ["probe completed"]},
                        },
                    ).as_json_text()
                )
                waited = service.session.external_tool_call(
                    tool="await_children",
                    arguments={"agent_ids": [created["agent_id"]]},
                    timeout=30,
                )
                self.assertTrue(waited.success, waited.as_json_text())
                self.assertTrue(worker.used_provider_credential)
            finally:
                service.close()
            self.assertNotIn(token, log.read_text(encoding="utf-8"))
            self.assertNotIn(token, (workspace / "status.json").read_text(encoding="utf-8"))
            self.assertNotIn(token, (workspace / "outcome.jsonl").read_text(encoding="utf-8"))

    def test_zai_failure_stderr_is_absent_from_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            token = secrets.token_urlsafe(48)
            path = workspace / "providers.json"
            path.write_text(json.dumps(_provider_document(token)), encoding="utf-8")
            log = workspace / "run.jsonl"
            worker = _ZaiScriptedWorker(load_zai_provider(path).api_key)
            worker.used_provider_credential = True
            worker.captured_stderr = lambda: (f"authorization failed for {token}",)
            with patch.object(provider_config, "PROVIDER_CONFIG_PATH", path):
                service = VNextMcpService(
                    workspace=workspace,
                    catalog=[{"provider": "zai", "model": "glm-5.2"}],
                    event_log=log,
                    status_file=workspace / "status.json",
                    outcome_log=workspace / "outcome.jsonl",
                    adapter_factories={"zai": lambda: worker},
                )
                try:
                    service.session._emit_provider_error(
                        "zai",
                        service.session.root,
                        worker,
                        RuntimeError("z.ai worker initialization failed"),
                    )
                finally:
                    service.close()

            raw_log = log.read_text(encoding="utf-8")
            self.assertTrue(worker.used_provider_credential)
            self.assertNotIn(token, raw_log)
            provider_errors = [
                json.loads(line)
                for line in raw_log.splitlines()
                if json.loads(line).get("type") == "provider.error"
            ]
            self.assertEqual(1, len(provider_errors))
            self.assertEqual([], provider_errors[0]["payload"]["provider_stderr"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
