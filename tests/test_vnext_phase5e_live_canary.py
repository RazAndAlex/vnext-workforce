"""Opt-in Phase 5E cross-vendor live canary.

This test is deliberately skipped unless a human enables the live provider
run.  Its receipt assertions use only provider-neutral, content-safe facts.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import unittest
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

# The diagnostic channel is imported first and on its own, so that it exists
# to observe a failure in any of the imports that follow it.
#
# THE HOLE THIS GUARD DOES NOT COVER: its own import.  If `vnext_diagnostics`
# itself fails to import, this line raises, the guarded block below never runs,
# and nothing is recorded — `record_failure` lives in the module that failed.
# That is structural and accepted: a channel cannot report its own absence, and
# any fallback would have to be a second, unverified channel.  The failure is
# still loud (an import error and a non-zero exit), just unlabelled.
#
# This import is deliberately NOT made defensive, unlike the production-side
# import in `vnext/vnext_claude.py`, which falls back to inert stubs so
# a broken diagnostics module cannot take down the runtime it observes.  The
# runtime must survive that defect; the test suite must shout about it.  Making
# both sides defensive would leave nothing anywhere to notice it.
from vnext.vnext_diagnostics import (
    FailureCategory,
    classify_exception,
    record_failure,
)

# Arm the channel at MODULE IMPORT time, not only in setUp.
#
# `setUp` is far too late to cover the most likely sub-300ms exit.  If an
# import below fails, unittest replaces this module with a synthetic
# `_FailedTest`, the real class is never constructed, and `setUp` never runs
# — so the run died before the channel that exists to explain it was armed.
# Confirmed empirically: a broken import produced `returncode 1`, `Ran 1
# test`, `FAILED (errors=1)`, and no record at all.  Module import, the
# `skipUnless` decorator, and the module constants are all outside `setUp`.
#
# Arming here is unconditional and idempotent; `record_failure` is a no-op
# unless VNEXT_DIAGNOSTICS is set, so this cannot write
# anything for an ordinary (non-live) discovery run.
# Arming is scoped to the live opt-in.  An ordinary discovery run must not
# have its environment mutated by importing a test module, and has no sink to
# populate; the live run is exactly the run whose early death needs a reason.
_LIVE_OPT_IN = "VNEXT_PHASE5E_LIVE_CANARY"
_DIAGNOSTICS_ENV = "VNEXT_DIAGNOSTICS"
_IMPORT_DIAGNOSTICS_PRIOR = os.environ.get(_DIAGNOSTICS_ENV)
_IMPORT_LIVE = os.environ.get(_LIVE_OPT_IN) == "1"
if _IMPORT_LIVE:
    os.environ[_DIAGNOSTICS_ENV] = "1"
_IMPORT_STARTED = time.monotonic()

try:
    from vnext.vnext_claude import ClaudeCodeAdapter
    from vnext.vnext_managed_session import VNextManagedSession
    from vnext.vnext_orchestration import (
        AgentRole,
        EconomicPreset,
        ModelCard,
        ModelRegistry,
        OrchestrationControlPlane,
    )
    from vnext.vnext_runtime_effects import (
        ClaudeRuntimeEffectReader,
        RuntimeEffectJournal,
    )
    from vnext.vnext_runtime_types import RuntimePosture
    from vnext.vnext_scheduler import SchedulerHooks, VNextScheduler
    from vnext.workforce_contracts import RunCancellation
except BaseException as _import_exc:  # Record the reason, then fail as before.
    # The record is an ADDITION to the existing failure, never a substitute:
    # the original exception is re-raised untouched by a bare `raise`, so
    # collection still produces the same `_FailedTest` and the same non-zero
    # exit.  `record_failure` is total and cannot raise, so it cannot mask
    # the import error it is describing.
    # Deliberately NOT `classify_exception` here.  That maps any ImportError
    # to `sdk_not_installed`, which is right for the bridge's optional SDK
    # import but actively misleading for this block: none of these imports
    # touch the Claude SDK, so a broken `vnext` name would be
    # reported as a missing SDK and send the reader to the wrong place.
    # The exception class is the honest discriminator and is recorded below.
    record_failure(
        FailureCategory.IMPORT_FAILED,
        phase="canary",
        step="module_import",
        exception_type=type(_import_exc).__name__,
        duration_ms=(time.monotonic() - _IMPORT_STARTED) * 1000.0,
        flags={
            "diagnostics_preset": _IMPORT_DIAGNOSTICS_PRIOR is not None,
            "live_opt_in": _IMPORT_LIVE,
        },
    )
    raise


def _restore_import_arming() -> None:
    """Undo the process-wide arming this module did at import time.

    `_IMPORT_DIAGNOSTICS_PRIOR` was captured above but only ever READ as a
    boolean flag, so under the live opt-in the module set
    VNEXT_DIAGNOSTICS=1 for the whole process and never put it
    back.  Every other test module in the same run then inherited armed
    diagnostics from an import it had nothing to do with.  Restoring is
    idempotent, so calling it from both hooks below is safe.
    """

    if not _IMPORT_LIVE:
        return
    if _IMPORT_DIAGNOSTICS_PRIOR is None:
        os.environ.pop(_DIAGNOSTICS_ENV, None)
    else:
        os.environ[_DIAGNOSTICS_ENV] = _IMPORT_DIAGNOSTICS_PRIOR


def tearDownModule() -> None:
    _restore_import_arming()


# `tearDownModule` runs only when unittest actually runs this module's tests.
# A run that imports the module and then selects no test from it — name
# filtering, a collection error elsewhere — would leave the arming standing, so
# the same restoration is also registered for process exit.
atexit.register(_restore_import_arming)


# `_LIVE_OPT_IN` and `_DIAGNOSTICS_ENV` are defined above the guarded import
# block, because the guard needs them before this point in the module.
_HISTORY_ENV = "CLAUDE_CODE_SKIP_PROMPT_HISTORY"
_FORBIDDEN_CREDENTIAL_OVERRIDES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
# A Claude model alias accepted by the provider CLI.  See the ModelCard note
# below: this value is forwarded to the runtime, not just used as a local key.
_CLAUDE_WORKER_MODEL = "sonnet"
_REQUESTED_POSTURE = RuntimePosture(
    workspace_writes=True,
    network="restricted",
    approvals_requested=True,
    reviewer="auto_review",
    environment_ready=True,
)
_RESOLVED_POSTURE = {
    **_REQUESTED_POSTURE.as_dict(),
    "network": "approval_gated",
}


@unittest.skipUnless(
    os.environ.get(_LIVE_OPT_IN) == "1",
    "set VNEXT_PHASE5E_LIVE_CANARY=1 to run the live canary",
)
class Phase5ELiveCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        """Arm the content-safe diagnostic sink for this run only.

        The sink deliberately lives outside the temporary workspace: every
        failure before the final receipt line happens while that directory is
        still scheduled for destruction, and a pre-bind failure has no managed
        session to write through.
        """

        prior = os.environ.get(_DIAGNOSTICS_ENV)
        had = _DIAGNOSTICS_ENV in os.environ
        os.environ[_DIAGNOSTICS_ENV] = "1"

        def restore() -> None:
            if had:
                os.environ[_DIAGNOSTICS_ENV] = prior or ""
            else:
                os.environ.pop(_DIAGNOSTICS_ENV, None)

        self.addCleanup(restore)

    def _assert_zero_tool_registration(
        self, registration: Mapping[str, object], model_id: str
    ) -> None:
        self.assertEqual(
            {
                "acknowledged": True,
                "model_id": model_id,
                "tool_count": 0,
                "tool_names": [],
                "definition_sha256": None,
                "handler_registered": False,
            },
            dict(registration),
        )

    def _assert_resolved_posture(self, policy: Mapping[str, object]) -> None:
        self.assertEqual(_RESOLVED_POSTURE, dict(policy.get("posture", {})))

    def _initialize_adapter(self, workspace: Path) -> ClaudeCodeAdapter:
        """Enable prompt-history skipping only during local bridge startup."""

        prior_history = os.environ.get(_HISTORY_ENV)
        had_history = _HISTORY_ENV in os.environ
        config_before = os.environ.get("CLAUDE_CONFIG_DIR")
        adapter = ClaudeCodeAdapter(workspace=str(workspace))
        started = time.monotonic()
        try:
            os.environ[_HISTORY_ENV] = "1"
            adapter.initialize()
        except BaseException as exc:  # Classify before the detail is suppressed.
            # Cleanup is in a `finally` so that recording cannot skip it.
            # Previously `record_failure` ran before `adapter.close()`, so a
            # raise from the diagnostic would have leaked the owned bridge —
            # defeating the zero-residual property the canary exists to show.
            # `record_failure` is total now, but the ordering must not depend
            # on that: closing the bridge outranks explaining why.
            try:
                record_failure(
                    classify_exception(exc),
                    phase="canary",
                    step="initialize_adapter",
                    exception_type=type(exc).__name__,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    flags={"history_env_present": had_history},
                )
            finally:
                adapter.close()
            raise
        finally:
            if had_history:
                os.environ[_HISTORY_ENV] = prior_history or ""
            else:
                os.environ.pop(_HISTORY_ENV, None)
        if (
            config_before != os.environ.get("CLAUDE_CONFIG_DIR")
            or had_history != (_HISTORY_ENV in os.environ)
            or prior_history != os.environ.get(_HISTORY_ENV)
        ):
            record_failure(
                FailureCategory.ENVIRONMENT_RESTORATION_FAILED,
                phase="canary",
                step="restore_environment",
                flags={"history_env_present": had_history},
            )
        self.assertTrue(
            config_before == os.environ.get("CLAUDE_CONFIG_DIR"),
            "live canary must not mutate CLAUDE_CONFIG_DIR",
        )
        self.assertEqual(had_history, _HISTORY_ENV in os.environ)
        self.assertTrue(
            prior_history == os.environ.get(_HISTORY_ENV),
            "prompt-history environment was not restored",
        )
        return adapter

    @staticmethod
    def _close_control_turn(managed: VNextManagedSession, agent: object) -> None:
        """Close a control-plane turn this harness opened by hand.

        `VNextManagedSession.start_turn` moves the agent READY -> RUNNING via
        `OrchestrationControlPlane.start_turn`.  The only production caller of
        the matching `finish_turn` is the scheduler's driving loop, and this
        canary never runs that loop — it uses the scheduler solely for the
        native approval rendezvous and its approval records.  So the harness
        owns every turn it opens and must close it, or the next `start_turn`
        fails with `cannot start a turn from running`.

        `finish_turn` requires RUNNING and raises otherwise, so it is NOT
        idempotent.  The status is checked first: after a runtime start
        failure the agent is BLOCKED, and after a close it is already READY.
        Guarding here keeps a failed or timed-out turn from turning one
        failure into a second, less informative one, and leaves the control
        plane coherent rather than wedged.
        """

        status = getattr(agent, "status", None)
        if status is None or status.value != "running":
            return
        managed.control.finish_turn(agent.agent_id)  # type: ignore[attr-defined]

    def _wait_with_branch_approvals(
        self,
        *,
        managed: VNextManagedSession,
        scheduler: VNextScheduler,
        branch_id: str,
        turn: object,
        approvals: queue.Queue[dict[str, object]],
        agent: object,
        require_completed: bool = True,
    ) -> list[dict[str, object]]:
        """Wait without bypassing the scheduler's native approval rendezvous."""

        result: dict[str, object] = {}
        errors: list[BaseException] = []

        def wait_for_turn() -> None:
            try:
                result.update(managed.wait_turn(turn, timeout=180))  # type: ignore[arg-type]
            except BaseException as exc:  # Preserve cleanup while hiding provider detail.
                record_failure(
                    classify_exception(exc),
                    phase="canary",
                    step="wait_turn",
                    exception_type=type(exc).__name__,
                    counts={"approvals_observed": approvals.qsize()},
                    flags={"require_completed": require_completed},
                )
                errors.append(exc)

        waiter = threading.Thread(target=wait_for_turn, daemon=True)
        waiter.start()
        observed: list[dict[str, object]] = []
        deadline = time.monotonic() + 200
        try:
            while waiter.is_alive():
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "worker turn did not finish")
                try:
                    approval = approvals.get(timeout=min(remaining, 5))
                except queue.Empty:
                    continue
                observed.append(approval)
                approval_id = approval.get("approval_id")
                self.assertTrue(
                    isinstance(approval_id, str),
                    "approval request did not contain a usable reference",
                )
                scheduler.resolve_approval(
                    approval_id, "accept", resolver=branch_id  # type: ignore[arg-type]
                )
            waiter.join(timeout=1)
        finally:
            # The harness opened this control-plane turn, so it closes it on
            # every exit path — including a timed-out or raising one.
            self._close_control_turn(managed, agent)
        self.assertEqual(0, len(errors), "worker turn raised without a safe result")
        if require_completed:
            self.assertEqual("completed", result.get("status"))
        return observed

    def test_claude_canary_contract(self) -> None:
        """Exercise the frozen Phase 5E gate contract when explicitly enabled."""

        workspace: Path | None = None
        self.assertFalse(
            any(name in os.environ for name in _FORBIDDEN_CREDENTIAL_OVERRIDES),
            "live canary must use existing subscription authentication",
        )
        with TemporaryDirectory(prefix="vnext-phase5e-") as workspace_text:
            workspace = Path(workspace_text)
            adapter = self._initialize_adapter(workspace)
            managed: VNextManagedSession | None = None
            try:
                # A control-plane construction failure happens before any
                # managed session exists to write a receipt through.
                try:
                    registry = ModelRegistry(
                        cards=[
                            ModelCard(
                                "vnext",
                                frozenset({AgentRole.ROOT_MANAGER, AgentRole.BRANCH_MANAGER}),
                                provider="codex",
                            ),
                            # The card id reaches the provider verbatim: the
                            # adapter forwards it to the bridge, which passes
                            # it to ClaudeAgentOptions(model=...).  It must
                            # therefore be a real Claude model alias, not a
                            # descriptive local label.
                            ModelCard(
                                _CLAUDE_WORKER_MODEL,
                                frozenset({AgentRole.WORKER}),
                                provider="claude",
                            ),
                        ],
                        presets=[
                            EconomicPreset(
                                "phase5e-live",
                                frozenset({"vnext", _CLAUDE_WORKER_MODEL}),
                            )
                        ],
                    )
                    control = OrchestrationControlPlane(registry)
                    root = control.create_session(
                        preset_id="phase5e-live",
                        root_model_id="vnext",
                        workspace=workspace,
                        objective="exercise the Phase 5E live canary",
                        task_contract={},
                        session_id="phase5e-live-canary",
                    )
                    branch = control.spawn_agent(
                        requester_id=root.agent_id,
                        parent_agent_id=root.agent_id,
                        role=AgentRole.BRANCH_MANAGER,
                        model_id="vnext",
                        objective="approve the worker's native runtime effects",
                        task_contract={},
                    )
                    worker = control.spawn_agent(
                        requester_id=branch.agent_id,
                        parent_agent_id=branch.agent_id,
                        role=AgentRole.WORKER,
                        model_id=_CLAUDE_WORKER_MODEL,
                        objective="perform the bounded Phase 5E canary effects",
                        task_contract={},
                    )
                except BaseException as exc:
                    record_failure(
                        FailureCategory.CONTROL_PLANE_CONSTRUCTION_FAILED,
                        phase="canary",
                        step="control_plane",
                        exception_type=type(exc).__name__,
                    )
                    raise
                effects = RuntimeEffectJournal(workspace)
                effects.bind_reader(worker.agent_id, ClaudeRuntimeEffectReader())
                managed = VNextManagedSession(
                    control=control,
                    adapter=adapter,
                    session_id=root.session_id,
                    runtime_effects=effects,
                )
                approval_events: queue.Queue[dict[str, object]] = queue.Queue()

                def lifecycle(
                    event_type: str, _agent: object, data: Mapping[str, object]
                ) -> None:
                    if event_type == "approval_requested":
                        approval_events.put(
                            {key: data.get(key) for key in ("approval_id", "worker_agent", "manager_agent", "effect")}
                        )

                scheduler = VNextScheduler(
                    managed=managed,
                    root=root,
                    cancellation=RunCancellation(),
                    hooks=SchedulerHooks(lifecycle=lifecycle),
                )
                runtime_thread, policy = adapter.start_thread(
                    model=worker.model_id,
                    developer_instructions="Follow only the current bounded canary turn.",
                    tools=[],
                    tool_handler=None,
                    requested_posture=_REQUESTED_POSTURE,
                    workspace=str(workspace),
                )
                self._assert_zero_tool_registration(
                    adapter.tool_registration_attestation(runtime_thread), worker.model_id
                )
                self._assert_resolved_posture(policy)
                managed.bind_thread(
                    agent_id=worker.agent_id,
                    thread_id=runtime_thread,
                    start_result=policy,
                    tool_handler=None,
                    adapter=adapter,
                )
                reserved_identity = adapter.thread_identity_attestation(runtime_thread)
                self.assertEqual("claude", reserved_identity.get("provider"))
                self.assertFalse(reserved_identity.get("bound"))
                self.assertEqual("reserved", reserved_identity.get("binding_phase"))
                self.assertFalse(reserved_identity.get("synthetic"))

                preflight = managed.start_turn(
                    worker.agent_id,
                    prompt="Reply with READY and do not use a tool.",
                    effort="low",
                    phase="identity-preflight",
                )
                try:
                    preflight_result = managed.wait_turn(preflight, timeout=180)
                finally:
                    self._close_control_turn(managed, worker)
                self.assertEqual("completed", preflight_result.get("status"))
                self.assertEqual(0, len(effects.records(worker.agent_id)))
                bound_identity = adapter.thread_identity_attestation(runtime_thread)
                self.assertEqual("claude", bound_identity.get("provider"))
                self.assertTrue(bound_identity.get("bound"))
                self.assertEqual("attested", bound_identity.get("binding_phase"))
                self.assertFalse(bound_identity.get("synthetic"))
                managed.mark_check("provider_identity_bound_on_first_turn", True)
                managed.mark_check("provider_identity_not_synthetic", True)

                marker_turn = managed.start_turn(
                    worker.agent_id,
                    prompt=(
                        "Use only the Write tool to create phase5e-marker.bin "
                        "containing exactly phase5e-marker. Do not do anything else."
                    ),
                    effort="low",
                    phase="write-marker",
                )
                # Root and Branch must still be locally responsive while the
                # provider-owned worker is executing its effect turn.
                self.assertEqual("vnext", root.model_id)
                self.assertEqual("vnext", branch.model_id)
                self.assertEqual("ready", root.status.value)
                self.assertEqual("ready", branch.status.value)
                marker_approvals = self._wait_with_branch_approvals(
                    managed=managed,
                    scheduler=scheduler,
                    branch_id=branch.agent_id,
                    turn=marker_turn,
                    approvals=approval_events,
                    agent=worker,
                )
                self.assertEqual(1, len(marker_approvals))
                marker_approval = marker_approvals[0]
                self.assertTrue(
                    marker_approval.get("worker_agent") == worker.agent_id,
                    "marker approval was not correlated to the worker",
                )
                self.assertTrue(
                    marker_approval.get("manager_agent") == branch.agent_id,
                    "marker approval was not resolved by the Branch Manager",
                )
                self.assertEqual("modify", marker_approval.get("effect"))
                self.assertTrue(
                    (workspace / "phase5e-marker.bin").read_bytes()
                    == b"phase5e-marker",
                    "marker file did not contain the exact expected bytes",
                )

                command_prompt = (
                    "Use only the Bash tool to run exactly this command and do nothing else: "
                    "printf 'phase5e-command\\n' > phase5e-command.bin"
                )
                command_turn = managed.start_turn(
                    worker.agent_id,
                    prompt=command_prompt,
                    effort="low",
                    phase="write-command-marker",
                )
                command_approvals = self._wait_with_branch_approvals(
                    managed=managed,
                    scheduler=scheduler,
                    branch_id=branch.agent_id,
                    turn=command_turn,
                    approvals=approval_events,
                    agent=worker,
                    require_completed=False,
                )
                self.assertEqual(1, len(command_approvals))
                command_approval = command_approvals[0]
                self.assertTrue(
                    command_approval.get("worker_agent") == worker.agent_id,
                    "command approval was not correlated to the worker",
                )
                self.assertTrue(
                    command_approval.get("manager_agent") == branch.agent_id,
                    "command approval was not resolved by the Branch Manager",
                )
                self.assertEqual("execute", command_approval.get("effect"))
                self.assertTrue(
                    (workspace / "phase5e-command.bin").read_bytes()
                    == b"phase5e-command\n",
                    "command marker did not contain the exact expected bytes",
                )
                managed.mark_check("command_proof_route_python_exact_bytes", True)
                managed.mark_check("command_provider_status_not_relied_upon", True)

                native_approvals = scheduler.native_approval_records()
                self.assertEqual(2, len(native_approvals))
                for approval in native_approvals:
                    self.assertIsInstance(approval.get("approval_reference"), str)
                    self.assertEqual("accept", approval.get("decision"))
                    self.assertTrue(approval.get("session_correlated"))
                    self.assertTrue(approval.get("turn_correlated"))
                    self.assertTrue(approval.get("request_correlated"))

                runtime_effects = effects.records(worker.agent_id)
                self.assertEqual(2, len(runtime_effects))
                self.assertEqual(
                    {"command", "file_change"},
                    {effect.effect for effect in runtime_effects},
                )
                command_effect = next(
                    effect for effect in runtime_effects if effect.effect == "command"
                )
                file_effect = next(
                    effect for effect in runtime_effects if effect.effect == "file_change"
                )
                self.assertEqual("claude", command_effect.provider)
                self.assertEqual("claude", file_effect.provider)
                self.assertFalse(command_effect.evidence_limited)
                self.assertFalse(file_effect.evidence_limited)
                self.assertEqual(
                    ("cwd", "actions", "exit_code", "duration_ms"),
                    command_effect.not_reported_fields,
                )
                # Observed live on claude-agent-sdk 0.2.143: a Bash tool result
                # carries `is_error=False`, while a Write tool result leaves
                # `is_error` as None.  So the provider reports a status for the
                # command effect and reports none for the file effect.  Gate 6
                # requires an unreported field to be recorded as not reported
                # and not to count as limited, which is asserted just above.
                self.assertEqual(
                    ("change_kind", "status"), file_effect.not_reported_fields
                )
                self.assertEqual("unknown", file_effect.status)
                self.assertEqual("completed", command_effect.status)
                projection = managed.runtime_effect_summary(worker.agent_id)
                self.assertEqual(2, projection.get("effect_count"))
                self.assertEqual(1, projection.get("command_count"))
                self.assertEqual(1, projection.get("file_change_count"))
                self.assertEqual(0, projection.get("limited_count"))
                self.assertEqual(0, projection.get("malformed_item_count"))
                self.assertEqual(0, projection.get("uncorrelated_item_count"))

                # This run deliberately uses the existing subscription config;
                # no disposable Claude home or config directory is created.
                managed.mark_check("disposable_home_used_false", True)
                managed.close()
                receipt_path = workspace / "phase5e-content-safe-receipt.json"
                managed.persist_content_safe_receipt(
                    receipt_path,
                    status="passed",
                    canary_evidence={
                        "proof_route": "python_exact_bytes",
                        "provider_status_relied_upon": False,
                        "disposable_home_used": False,
                    },
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                canary_evidence = receipt.get("canary_evidence", {})
                self.assertTrue(
                    canary_evidence.get("proof_route") == "python_exact_bytes",
                    "receipt did not retain the selected proof route",
                )
                self.assertTrue(
                    canary_evidence.get("provider_status_relied_upon") is False,
                    "receipt incorrectly relied on provider status",
                )
                self.assertTrue(
                    canary_evidence.get("disposable_home_used") is False,
                    "receipt incorrectly reported disposable-home use",
                )
                self.assertEqual("passed", receipt.get("status"))
                self.assertTrue(
                    receipt.get("checks", {}).get(
                        "command_proof_route_python_exact_bytes"
                    )
                )
                self.assertTrue(
                    receipt.get("checks", {}).get(
                        "command_provider_status_not_relied_upon"
                    )
                )
                self.assertTrue(
                    receipt.get("checks", {}).get("disposable_home_used_false")
                )
                self.assertTrue(
                    receipt.get("checks", {}).get("provider_identity_bound_on_first_turn")
                )
                self.assertTrue(receipt.get("checks", {}).get("zero_orphan_shutdown"))
                cleanup = receipt.get("cleanup", [])
                self.assertTrue(cleanup)
                for item in cleanup:
                    self.assertEqual(0, item.get("residual_count"))
                    self.assertTrue(item.get("streams_drained"))
                    self.assertTrue(item.get("handler_threads_drained"))
                    self.assertEqual(0, item.get("error_count"))

                # Decision 3: receipt data contains summaries and opaque refs,
                # never the exact provider instruction or command text.
                receipt_text = json.dumps(receipt, sort_keys=True)
                self.assertFalse(
                    "Use only the Write tool to create phase5e-marker.bin "
                    "containing exactly phase5e-marker. Do not do anything else."
                    in receipt_text,
                    "receipt retained the provider instruction",
                )
                self.assertFalse(
                    command_prompt in receipt_text,
                    "receipt retained the command prompt",
                )
                provider_session = bound_identity.get("provider_session")
                known_raw_ids = [
                    root.session_id,
                    root.agent_id,
                    branch.agent_id,
                    worker.agent_id,
                    runtime_thread,
                    preflight.control_turn_id,
                    preflight.runtime.turn_id,
                    marker_turn.control_turn_id,
                    marker_turn.runtime.turn_id,
                    command_turn.control_turn_id,
                    command_turn.runtime.turn_id,
                ]
                if isinstance(provider_session, str):
                    known_raw_ids.append(provider_session)
                for approval in [
                    *marker_approvals,
                    *command_approvals,
                    *native_approvals,
                ]:
                    for key in ("approval_id", "approval_reference"):
                        value = approval.get(key)
                        if isinstance(value, str):
                            known_raw_ids.append(value)
                for raw_id in known_raw_ids:
                    self.assertNotIn(
                        raw_id,
                        receipt_text,
                        "persisted receipt retained a raw manager/provider identifier",
                    )
            finally:
                if managed is None:
                    adapter.close()
                else:
                    managed.close()
        self.assertTrue(
            workspace is not None and not workspace.exists(),
            "temporary canary workspace remained after context cleanup",
        )


if __name__ == "__main__":
    unittest.main()
