"""Deterministic tests for the content-safe failure classification channel."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import unittest
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple
from unittest import mock

import vnext.vnext_claude
import vnext.vnext_diagnostics as vnext_diagnostics
from vnext.vnext_claude import ClaudeCodeAdapter, ClaudeRuntimeError
from vnext.vnext_diagnostics import (
    FailureCategory,
    build_record,
    classify_exception,
    classify_message,
    classify_stderr,
    drain_stderr,
    override_is_usable,
    record_failure,
    record_values,
    sink_path,
)


_ENABLE_ENV = "VNEXT_DIAGNOSTICS"
_DIR_ENV = "VNEXT_DIAGNOSTIC_DIR"

# What the module actually guarantees: every string that survives into a
# record is either empty or a bare label token.  This is the module's own
# `_TOKEN` shape, asserted independently so a change to the regex cannot
# quietly weaken the test along with it.
_LABEL_TOKEN = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.\-]*\Z")

# The real, repository-local default sink.  No test in this file may write to
# it: a record appended once per suite run grows without bound, and `data/` is
# gitignored, so `git status` never shows it and nobody notices.  One test did
# exactly that for the life of this work.
_REAL_DEFAULT_SINK = vnext_diagnostics._default_sink_dir() / "vnext-failures.jsonl"
_REAL_DEFAULT_SINK_AT_IMPORT: tuple[bool, int] = (False, 0)


def _sink_fingerprint() -> tuple[bool, int]:
    try:
        return (True, _REAL_DEFAULT_SINK.stat().st_size)
    except OSError:
        return (False, 0)


def setUpModule() -> None:
    global _REAL_DEFAULT_SINK_AT_IMPORT
    _REAL_DEFAULT_SINK_AT_IMPORT = _sink_fingerprint()


def tearDownModule() -> None:
    """Structural guard: no test here may touch the repository's real sink.

    Checking each test individually is the kind of assurance that decays as
    tests are added.  Comparing the real sink before and after the whole module
    catches the leak wherever it comes from, including from a test written
    later by someone who never read this comment.
    """

    after = _sink_fingerprint()
    if after != _REAL_DEFAULT_SINK_AT_IMPORT:
        raise AssertionError(
            "a test in this module wrote to the repository's default "
            f"diagnostic sink ({_REAL_DEFAULT_SINK}).  Point the sink at a "
            "temporary directory: either via "
            "VNEXT_DIAGNOSTIC_DIR with an ABSOLUTE path, or by "
            "patching vnext_diagnostics._default_sink_dir when the point of "
            "the test is that the override is refused."
        )


class ClassifyMessageTests(unittest.TestCase):
    CASES = {
        "workspace is unavailable": FailureCategory.WORKSPACE_UNAVAILABLE,
        "workspace does not exist": FailureCategory.WORKSPACE_UNAVAILABLE,
        "workspace is not a folder": FailureCategory.WORKSPACE_UNAVAILABLE,
        "unable to start owned Claude bridge": FailureCategory.BRIDGE_SPAWN_REFUSED,
        "owned Claude bridge has no captured streams": FailureCategory.BRIDGE_STREAMS_UNAVAILABLE,
        "owned Claude bridge stdin is unavailable": FailureCategory.BRIDGE_STDIN_UNAVAILABLE,
        "cannot write owned Claude bridge": FailureCategory.BRIDGE_WRITE_FAILED,
        "Claude bridge stdout closed unexpectedly": FailureCategory.BRIDGE_EXITED_AT_STARTUP,
        "owned Claude bridge is not running": FailureCategory.BRIDGE_NOT_RUNNING,
        "Claude bridge timed out waiting for start_thread": FailureCategory.BRIDGE_REQUEST_TIMEOUT,
        "Claude bridge emitted malformed JSONL": FailureCategory.BRIDGE_PROTOCOL_VIOLATION,
        "the Claude SDK (claude-agent-sdk==0.2.163) could not be imported: ": (
            FailureCategory.SDK_NOT_INSTALLED
        ),
        "installed Claude SDK version is not exactly 0.2.163": FailureCategory.SDK_VERSION_MISMATCH,
        "credential override environment is forbidden": FailureCategory.CREDENTIAL_OVERRIDE_PRESENT,
        "bridge initialization did not attest Claude capabilities": (
            FailureCategory.INITIALIZE_ATTESTATION_MISMATCH
        ),
        "Claude connection lacks server capability evidence": FailureCategory.PROVIDER_CONNECT_FAILED,
        "Claude query failed: TimeoutError": FailureCategory.PROVIDER_QUERY_FAILED,
        "Claude reservation lacks local turn outcome state": FailureCategory.EFFECT_STATE_EXHAUSTED,
        "Claude turn completion is malformed": FailureCategory.BRIDGE_PROTOCOL_VIOLATION,
        "Claude message stream ended before turn terminal result": FailureCategory.PROVIDER_QUERY_FAILED,
        "Claude leaf cannot satisfy the requested neutral posture": FailureCategory.POSTURE_REJECTED,
        "bridge did not attest the empty tool registration": (
            FailureCategory.TOOL_REGISTRATION_REJECTED
        ),
        "Claude resume did not attest the requested tool registration": (
            FailureCategory.TOOL_REGISTRATION_REJECTED
        ),
        "Claude bridge emitted an invalid native session identity": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
        "Claude permission event lacks local correlation": (
            FailureCategory.APPROVAL_CORRELATION_FAILED
        ),
        "unsupported Claude bridge operation: sing": FailureCategory.UNSUPPORTED_OPERATION,
        "Claude session is owned by an active native terminal": (
            FailureCategory.TERMINAL_LEASE_CONFLICT
        ),
        "cannot hand Claude session to terminal while an SDK turn is active": (
            FailureCategory.EFFECT_STATE_EXHAUSTED
        ),
        "terminal release does not match a bound Claude session": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
        "native task stop lacks provider task identity": FailureCategory.REQUEST_PAYLOAD_INVALID,
        "Claude native terminal interrupt lacks an active lease": (
            FailureCategory.TERMINAL_LEASE_CONFLICT
        ),
        "Claude native terminal interrupt must be callable": FailureCategory.REQUEST_PAYLOAD_INVALID,
        "Claude native terminal has no interrupt controller": FailureCategory.PROVIDER_QUERY_FAILED,
        "native Claude terminal tool controller failed": FailureCategory.MANAGER_TOOL_HANDLER_FAILED,
        "native Claude terminal tool controller returned an unsupported result": (
            FailureCategory.BRIDGE_PROTOCOL_VIOLATION
        ),
        "Claude terminal turn timed out": FailureCategory.BRIDGE_REQUEST_TIMEOUT,
        "Claude terminal turn has an unattested status": FailureCategory.BRIDGE_PROTOCOL_VIOLATION,
        "native child registration lacks enrollment hooks": FailureCategory.TOOL_REGISTRATION_REJECTED,
        "native child registration was not exactly attested": FailureCategory.IDENTITY_BINDING_FAILED,
        "native task lacks parent tool use": FailureCategory.APPROVAL_CORRELATION_FAILED,
        "Claude native child interrupt lacks exact task correlation": (
            FailureCategory.APPROVAL_CORRELATION_FAILED
        ),
        "Claude native child wait lacks exact task correlation": (
            FailureCategory.APPROVAL_CORRELATION_FAILED
        ),
        "Claude native child tool call lacks exact task correlation": (
            FailureCategory.APPROVAL_CORRELATION_FAILED
        ),
        "Claude native child wait timed out": FailureCategory.BRIDGE_REQUEST_TIMEOUT,
        "Claude native child has an unsupported terminal state": (
            FailureCategory.BRIDGE_PROTOCOL_VIOLATION
        ),
        "Claude native child parent agent is not exactly resolved": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
        "Claude native child observer returned an invalid binding": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
        "Claude native child identity conflicts with prior observation": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
        "Claude native child task conflicts with prior observation": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
        "Claude native child tool call is not attested": FailureCategory.IDENTITY_BINDING_FAILED,
        "Claude native child terminal status conflicts with prior terminal": (
            FailureCategory.BRIDGE_PROTOCOL_VIOLATION
        ),
        "Claude native child terminal state conflicts with prior evidence": (
            FailureCategory.BRIDGE_PROTOCOL_VIOLATION
        ),
        "native child origin state is unavailable": FailureCategory.EFFECT_STATE_EXHAUSTED,
        "native child origin lacks an active parent turn": FailureCategory.APPROVAL_CORRELATION_FAILED,
        "native child event lacks immutable parent origin": FailureCategory.APPROVAL_CORRELATION_FAILED,
        "native child event has invalid immutable parent origin": FailureCategory.APPROVAL_CORRELATION_FAILED,
        "Claude native child cancel lacks an attested task": FailureCategory.IDENTITY_BINDING_FAILED,
        "Claude reservation lacks metadata retry state": FailureCategory.EFFECT_STATE_EXHAUSTED,
        "Claude metadata retry cursor is malformed": FailureCategory.BRIDGE_PROTOCOL_VIOLATION,
        "Claude reservation lacks automatic native-child replay state": (
            FailureCategory.EFFECT_STATE_EXHAUSTED
        ),
        "native child lifecycle lacks a source": FailureCategory.BRIDGE_PROTOCOL_VIOLATION,
        "Claude reservation lacks native child state": FailureCategory.EFFECT_STATE_EXHAUSTED,
        "Claude reservation lacks automatic native-child lifecycle state": (
            FailureCategory.EFFECT_STATE_EXHAUSTED
        ),
        "Claude native child lifecycle history is malformed": (
            FailureCategory.BRIDGE_PROTOCOL_VIOLATION
        ),
        "nested native child lacks an exact parent task": FailureCategory.IDENTITY_BINDING_FAILED,
        "Claude reservation lacks native child identity ledger": (
            FailureCategory.EFFECT_STATE_EXHAUSTED
        ),
        "automatic native child conflicts with hook identity ledger": (
            FailureCategory.IDENTITY_BINDING_FAILED
        ),
    }

    def test_every_known_literal_maps_to_its_category(self) -> None:
        for message, expected in self.CASES.items():
            with self.subTest(message=message):
                self.assertEqual(expected, classify_message(message))

    def test_unknown_and_empty_messages_are_unclassified(self) -> None:
        for message in ("", "   ", None, 17, "a wholly novel failure"):
            with self.subTest(message=message):
                self.assertEqual(FailureCategory.UNCLASSIFIED, classify_message(message))


# Raising one of these is control flow, not a failure report, so it carries no
# message for the classifier to read.  `raise SystemExit(main())` in the
# bridge's `__main__` block is the concrete case.
_CONTROL_FLOW_EXCEPTIONS = frozenset(
    {
        "SystemExit",
        "KeyboardInterrupt",
        "StopIteration",
        "StopAsyncIteration",
        "GeneratorExit",
    }
)

# Keyword names under which a raise may pass its message instead of using the
# first positional argument.
_MESSAGE_KEYWORDS = ("msg", "message", "reason", "detail", "text")


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, so a raise may use one."""

    constants: dict[str, str] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value.value
    return constants


def _literal_of(node: ast.AST, constants: Mapping[str, str]) -> str | None:
    """The statically visible literal prefix of a message, or None.

    The narrow original handled only a bare string and an f-string with a
    constant head.  Every other form a message can take was invisible, and an
    invisible raise is an unclassified failure nobody is told about.  The forms
    resolved here are the ones whose literal really is readable from the source:

    * ``"..." .format(...)``, ``"..." % (...)`` and ``"..." + x`` — the fixed
      part is the left/receiver operand.
    * ``NAME`` bound to a module-level string constant.
    * ``str(x)`` — a transparent wrapper around the message it stringifies.
    * ``a or "fallback"`` — the fallback arm is a fixed literal even though the
      first arm is computed.

    Anything else returns None and its site is REPORTED as unresolvable rather
    than skipped; see `SourceCoverageTests.test_every_raise_site_is_readable`.
    """

    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                break  # Interpolation ends the fixed prefix.
        # An f-string that STARTS with interpolation has no fixed prefix at
        # all, so it is unresolvable rather than empty.
        return "".join(parts) if parts else None
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return _literal_of(node.left, constants)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        for value in node.values:
            literal = _literal_of(value, constants)
            if literal:
                return literal
        return None
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            return _literal_of(func.value, constants)
        if isinstance(func, ast.Name) and func.id == "str" and len(node.args) == 1:
            return _literal_of(node.args[0], constants)
    return None


def _called_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _message_argument(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg in _MESSAGE_KEYWORDS:
            return keyword.value
    return None


def _fatal_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to ``_set_fatal``, e.g. ``fatal = self._set_fatal``."""

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        bound = (
            (isinstance(value, ast.Attribute) and value.attr == "_set_fatal")
            or (isinstance(value, ast.Name) and value.id in aliases)
        )
        if bound:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.add(target.id)
    return aliases


class _Site(NamedTuple):
    module: str
    lineno: int
    raiser: str
    literal: str | None
    expression: str


def _source_failure_sites(module: str, path: Path) -> list[_Site]:
    """Every raise/fatal site in a module, resolved to a literal or flagged.

    Both halves matter.  The literals feed the classification assertion; the
    unresolved sites feed the readability assertion, because a raise the scan
    cannot read is a failure the classifier cannot cover — and silence about it
    is the defect, not the raise itself.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = _module_string_constants(tree)
    aliases = _fatal_aliases(tree)
    sites: list[_Site] = []

    def add(node: ast.AST, raiser: str, argument: ast.expr) -> None:
        literal = _literal_of(argument, constants)
        if literal is not None and not literal.strip():
            literal = None
        sites.append(
            _Site(module, node.lineno, raiser, literal, ast.unparse(argument))
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            exc = node.exc
            if exc is None:
                continue  # A bare re-raise adds no new message.
            if isinstance(exc, ast.Call):
                raiser = _called_name(exc.func) or "<computed>"
                if raiser in _CONTROL_FLOW_EXCEPTIONS:
                    continue
                argument = _message_argument(exc)
                if argument is None:
                    if not exc.args and not exc.keywords:
                        continue  # `raise Err()` carries no message at all.
                    sites.append(
                        _Site(module, node.lineno, raiser, None, ast.unparse(exc))
                    )
                    continue
                add(node, raiser, argument)
            elif isinstance(exc, (ast.Name, ast.Attribute)):
                name = _called_name(exc)
                if name in _CONTROL_FLOW_EXCEPTIONS:
                    continue
                # `raise err` after building the instance elsewhere: the
                # message is not visible from the raise, so say so.
                sites.append(
                    _Site(module, node.lineno, name or "<instance>", None, ast.unparse(exc))
                )
            else:
                sites.append(
                    _Site(module, node.lineno, "<computed>", None, ast.unparse(exc))
                )
        elif isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name != "_set_fatal" and name not in aliases:
                continue
            argument = _message_argument(node)
            if argument is None:
                if not node.args and not node.keywords:
                    continue
                sites.append(
                    _Site(module, node.lineno, "_set_fatal", None, ast.unparse(node))
                )
                continue
            add(node, "_set_fatal", argument)
    return sites


class SourceCoverageTests(unittest.TestCase):
    """The classifier is checked against the real source, not a copy of it.

    The tests this replaces iterated a hand-written dict of literals, so they
    could only ever confirm that the table agreed with itself.  A literal
    added to the adapter or the bridge without a matching fragment stayed
    invisible, and `unclassified` is just UNKNOWN wearing a different label.
    These tests parse the shipped modules instead, so that gap fails loudly.
    """

    MODULES = ("vnext_claude.py", "vnext_claude_bridge.py")

    # EXACT literal counts, per module, deliberately pinned.
    #
    # The floor this replaces was `assertGreater(len(collected), 80)` against 96
    # literals — fifteen literals of slack, and slack in a coverage guard is a
    # hole.  Demonstrated: fifteen real literals were mechanically rewritten as
    # `.format()` calls, which the scanner of the day could not read; runtime
    # behaviour was identical and the whole suite stayed green.
    #
    # UPDATE THESE NUMBERS DELIBERATELY when you add or remove a raise site.
    # Having to edit the number is the point: it is the moment you confirm the
    # new literal is classified rather than merely tolerated.
    EXPECTED_LITERALS = {"vnext_claude.py": 120, "vnext_claude_bridge.py": 132}

    # Raise sites whose message genuinely cannot be read from the source, each
    # with the reason it is acceptable.  Keyed by (module, raiser, expression)
    # so a NEW unreadable site cannot hide behind an old exemption.
    ACKNOWLEDGED_UNRESOLVABLE = {
        # Re-raise of a fatal that `_set_fatal` already classified and recorded
        # at the point it was set.  Classifying it again here would only
        # duplicate a label that has already been written.
        ("vnext_claude.py", "ClaudeRuntimeError", "self._fatal"),
        # Python's attribute protocol, raised by the inert diagnostics stub's
        # `__getattr__`.  It is not a runtime failure and never reaches the
        # classifier.
        ("vnext_claude.py", "AttributeError", "name"),
    }

    def _package_dir(self) -> Path:
        return Path(vnext.vnext_claude.__file__).resolve().parent

    def _sites(self) -> list[_Site]:
        collected: list[_Site] = []
        for name in self.MODULES:
            collected.extend(_source_failure_sites(name, self._package_dir() / name))
        return collected

    def _literals(self) -> list[_Site]:
        return [site for site in self._sites() if site.literal is not None]

    def test_the_scan_actually_finds_the_failure_literals(self) -> None:
        """Guard the guard: a scan that silently matched nothing would pass."""

        counted = Counter(site.module for site in self._literals())
        self.assertEqual(
            self.EXPECTED_LITERALS,
            dict(counted),
            "the failure-literal count changed.  If you added or removed a "
            "raise site, confirm the new literal classifies and then update "
            "SourceCoverageTests.EXPECTED_LITERALS.  If you did not, the AST "
            "scan has gone blind to a form it used to read.",
        )

    def test_every_raise_site_is_readable(self) -> None:
        """A raise the scan cannot read must be loud, not silently skipped.

        The scanner resolves the forms it reasonably can — `.format()`, `%`,
        `+`, a module constant, a keyword argument, `str(...)`, an `or`
        fallback.  What it cannot resolve it reports HERE, because a raise
        whose message is invisible to the scan is a failure the classifier is
        structurally unable to cover, and reporting nothing about it is exactly
        the blindness this guard exists to remove.
        """

        unreadable = [
            f"{site.module}:{site.lineno}: {site.raiser}({site.expression})"
            for site in self._sites()
            if site.literal is None
            and (site.module, site.raiser, site.expression)
            not in self.ACKNOWLEDGED_UNRESOLVABLE
        ]
        self.assertEqual(
            [],
            unreadable,
            "these raise sites carry a message the coverage scan cannot read, "
            "so nothing can confirm they classify.  Use a plain string literal "
            "(an f-string with a constant head is fine), or add the site to "
            "SourceCoverageTests.ACKNOWLEDGED_UNRESOLVABLE with the reason:\n"
            + "\n".join(unreadable),
        )

    def test_no_source_literal_is_unclassified(self) -> None:
        """Every raise/fatal literal in the real source must carry a reason."""

        unclassified = [
            f"{site.module}:{site.lineno}: {site.literal!r}"
            for site in self._literals()
            if classify_message(site.literal) is FailureCategory.UNCLASSIFIED
        ]
        self.assertEqual(
            [],
            unclassified,
            "these failures would report `unclassified`; add a fragment to "
            "the appended block of _FRAGMENTS in vnext_diagnostics.py:\n"
            + "\n".join(unclassified),
        )

    def test_harness_only_categories_are_the_documented_exceptions(self) -> None:
        """Categories with no source literal must be deliberate, not stale."""

        reached = {classify_message(site.literal) for site in self._literals()}
        harness_only = {
            FailureCategory.ENVIRONMENT_RESTORATION_FAILED,
            FailureCategory.CONTROL_PLANE_CONSTRUCTION_FAILED,
            FailureCategory.IMPORT_FAILED,
            FailureCategory.UNCLASSIFIED,
        }
        self.assertEqual(
            set(FailureCategory),
            reached | harness_only,
            "a category is unreachable from any source literal and is not "
            "declared harness-only",
        )
        self.assertEqual(
            set(),
            reached & harness_only,
            "a harness-only category is now reachable from a source literal",
        )


class ClassifyExceptionTests(unittest.TestCase):
    def test_class_name_wins_over_message(self) -> None:
        exc = FileNotFoundError(2, "No such file or directory", r"C:\\Users\\x\\bridge.py")
        self.assertEqual(FailureCategory.BRIDGE_SPAWN_REFUSED, classify_exception(exc))

    def test_import_error_maps_to_sdk_not_installed(self) -> None:
        self.assertEqual(
            FailureCategory.SDK_NOT_INSTALLED,
            classify_exception(ModuleNotFoundError("No module named 'claude_agent_sdk'")),
        )

    def test_unknown_exception_falls_back_to_its_message(self) -> None:
        self.assertEqual(
            FailureCategory.BRIDGE_REQUEST_TIMEOUT,
            classify_exception(RuntimeError("Claude bridge timed out waiting for wait_turn")),
        )
        self.assertEqual(FailureCategory.UNCLASSIFIED, classify_exception(RuntimeError("boom")))
        self.assertEqual(FailureCategory.UNCLASSIFIED, classify_exception(None))


class ClassifyStderrTests(unittest.TestCase):
    TRACEBACK = [
        "Traceback (most recent call last):",
        r'  File "C:\\Users\\Utente\\vnext\\vnext_claude_bridge.py", line 295, in _load_sdk',
        "    import claude_agent_sdk as sdk",
        "ModuleNotFoundError: No module named 'claude_agent_sdk'",
    ]

    def test_traceback_is_reduced_to_a_category(self) -> None:
        self.assertEqual(FailureCategory.SDK_NOT_INSTALLED, classify_stderr(self.TRACEBACK))

    def test_unrecognised_output_still_reports_a_dead_bridge(self) -> None:
        self.assertEqual(
            FailureCategory.BRIDGE_EXITED_AT_STARTUP, classify_stderr(["some noise"])
        )

    def test_empty_stderr_is_unclassified(self) -> None:
        self.assertEqual(FailureCategory.UNCLASSIFIED, classify_stderr([]))

    def test_drain_consumes_the_buffer_entirely(self) -> None:
        buffer: deque[str] = deque(self.TRACEBACK, maxlen=32)
        self.assertEqual(FailureCategory.SDK_NOT_INSTALLED, drain_stderr(buffer))
        self.assertEqual(0, len(buffer))
        self.assertEqual(FailureCategory.UNCLASSIFIED, drain_stderr(buffer))

    def test_drain_tolerates_a_non_buffer(self) -> None:
        self.assertEqual(FailureCategory.UNCLASSIFIED, drain_stderr(None))


class LabelShapeAssertions(unittest.TestCase):
    """Assert the guarantee the module makes, not a weaker proxy for it."""

    def assert_only_label_tokens(self, record: Mapping[str, object]) -> None:
        for value in record_values(record):
            if isinstance(value, str) and value:
                with self.subTest(value=value):
                    self.assertRegex(
                        value,
                        _LABEL_TOKEN,
                        "record retained a value that is not a bare label token",
                    )


class RecordShapeTests(LabelShapeAssertions):
    def test_free_text_and_paths_are_scrubbed_from_every_value(self) -> None:
        record = build_record(
            FailureCategory.BRIDGE_SPAWN_REFUSED,
            phase=r"C:\\Users\\Utente\\secret",
            step="the bridge could not be started at all",
            exception_type="/usr/lib/python3/os.py",
            duration_ms=302.4567,
            counts={"pending requests": 2, "threads": 1},
            flags={"bridge_started": False},
        )
        self.assert_only_label_tokens(record)
        self.assertEqual("invalid_label", record["phase"])
        self.assertEqual("invalid_label", record["step"])
        self.assertEqual("invalid_label", record["exception_type"])
        self.assertEqual({"invalid_label": 2, "threads": 1}, record["counts"])

    def test_a_well_formed_record_keeps_its_facts(self) -> None:
        record = build_record(
            FailureCategory.SDK_NOT_INSTALLED,
            phase="claude_adapter",
            step="initialize",
            exception_type="ModuleNotFoundError",
            duration_ms=302.0,
            counts={"pending_requests": 1},
            flags={"initialized": False},
        )
        self.assertEqual("sdk_not_installed", record["category"])
        self.assertEqual("claude_adapter", record["phase"])
        self.assertEqual("initialize", record["step"])
        self.assertEqual("ModuleNotFoundError", record["exception_type"])
        self.assertEqual(302.0, record["duration_ms"])
        self.assertEqual({"pending_requests": 1}, record["counts"])
        self.assertEqual({"initialized": False}, record["flags"])

    def test_a_trailing_newline_cannot_smuggle_a_value_through(self) -> None:
        """`$` matches before a trailing newline; the scrubber must use `\\Z`."""

        record = build_record(
            FailureCategory.UNCLASSIFIED,
            phase="leaked_secret_value\n",
            step="initialize\n",
            exception_type="RuntimeError\n",
            counts={"pending\n": 1},
            flags={"armed\n": True},
        )
        self.assertEqual("invalid_label", record["phase"])
        self.assertEqual("invalid_label", record["step"])
        self.assertEqual("invalid_label", record["exception_type"])
        self.assertEqual({"invalid_label": 1}, record["counts"])
        self.assertEqual({"invalid_label": True}, record["flags"])
        self.assert_only_label_tokens(record)

    def test_the_scrubber_is_a_shape_filter_not_a_redactor(self) -> None:
        """Pin the real limitation so nobody mistakes this for redaction.

        A credential is token-shaped, so it passes `_safe_token` verbatim.
        The sink is content-safe because of CALL-SITE DISCIPLINE — only
        codebase-chosen literals are ever passed — not because this function
        can recognise a secret.  A test that implied otherwise would be
        false comfort, so the limitation is asserted rather than hidden.
        """

        credential = "sk-ant-api03-1a2b3c4d5e6f7g8h"
        record = build_record(FailureCategory.UNCLASSIFIED, phase=credential)
        self.assertEqual(
            credential,
            record["phase"],
            "if this now scrubs, update the documented call-site discipline",
        )
        # And it is genuinely token-shaped, i.e. the weaker path/whitespace
        # check used previously could never have caught it either.
        self.assertRegex(credential, _LABEL_TOKEN)

    def test_counts_reject_non_integers_and_booleans(self) -> None:
        record = build_record(
            FailureCategory.UNCLASSIFIED,
            phase="canary",
            counts={"ok": 3, "bad": True, "worse": "many"},  # type: ignore[dict-item]
        )
        self.assertEqual({"ok": 3}, record["counts"])


class _RaisingStr:
    """A value whose ``__str__`` raises, as a hostile argument would."""

    def __str__(self) -> str:
        raise RuntimeError("str() must not be trusted on the failure path")


class _RaisingBool:
    def __bool__(self) -> bool:
        raise RuntimeError("bool() must not be trusted on the failure path")


class _RaisingItems:
    def items(self):  # noqa: ANN201 - deliberately hostile
        raise RuntimeError("items() must not be trusted on the failure path")


class _BadShapeItems:
    """``items()`` that returns entries which are not key/value pairs.

    This is the escape the original totality fix missed.  `_safe_items` caught
    a raise from `items()` itself, but the two-value unpacking happened at the
    CALL SITE, so a three-element entry raised `ValueError: too many values to
    unpack` out of `build_record`; `record_failure` swallowed it and returned
    `None`, losing the diagnostic entirely.
    """

    def __init__(self, entries: object) -> None:
        self._entries = entries

    def items(self):  # noqa: ANN201 - deliberately hostile
        return self._entries


class _NotIterableItems:
    def items(self):  # noqa: ANN201 - deliberately hostile
        return 17


class _EndlessItems:
    """``items()`` that never stops — hanging is untotal too, just quieter."""

    def items(self):  # noqa: ANN201 - deliberately hostile
        def endless():  # noqa: ANN202
            index = 0
            while True:
                yield (f"k{index}", index)
                index += 1

        return endless()


class _FaultingBaseException(BaseException):
    """NOT an `Exception`, so a bare `except Exception` lets it through."""


class _RaisingStrBase:
    def __str__(self) -> str:
        raise _FaultingBaseException("a BaseException that is not an Exception")


class TotalityTests(LabelShapeAssertions):
    """`record_failure` observes a failure; it must never replace one.

    Every case below was a demonstrated escape: `build_record` ran outside
    the guard and only `OSError` was caught, so a hostile or merely odd
    argument turned a diagnosable failure into a different exception raised
    from the diagnostic itself.
    """

    def setUp(self) -> None:
        self._prior = {name: os.environ.get(name) for name in (_ENABLE_ENV, _DIR_ENV)}
        self._dir = TemporaryDirectory(prefix="vnext-diag-")
        self.addCleanup(self._dir.cleanup)
        self.addCleanup(self._restore)
        os.environ[_ENABLE_ENV] = "1"
        os.environ[_DIR_ENV] = self._dir.name

    def _restore(self) -> None:
        for name, value in self._prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    ESCAPES = {
        "phase_str_raises": {"phase": _RaisingStr()},
        "duration_is_not_a_number": {"phase": "canary", "duration_ms": "soon"},
        "counts_key_str_raises": {"phase": "canary", "counts": {_RaisingStr(): 1}},
        "flags_value_bool_raises": {"phase": "canary", "flags": {"f": _RaisingBool()}},
        "exception_type_str_raises": {"phase": "canary", "exception_type": _RaisingStr()},
        # Reusing the same five escapes for both functions meant a defect
        # present in only one of them was invisible.  Everything below is a
        # form neither function had ever been given.
        "counts_items_yield_triples": {
            "phase": "canary",
            "counts": _BadShapeItems([("a", 1, "extra")]),
        },
        "flags_items_yield_scalars": {
            "phase": "canary",
            "flags": _BadShapeItems([1, 2, 3]),
        },
        "counts_items_yield_one_element_tuples": {
            "phase": "canary",
            "counts": _BadShapeItems([("lonely",)]),
        },
        "counts_items_are_not_iterable": {
            "phase": "canary",
            "counts": _NotIterableItems(),
        },
        "counts_items_never_end": {"phase": "canary", "counts": _EndlessItems()},
        "phase_str_raises_a_bare_base_exception": {"phase": _RaisingStrBase()},
        "counts_key_str_raises_a_bare_base_exception": {
            "phase": "canary",
            "counts": {_RaisingStrBase(): 1},
        },
        "category_str_raises": {"phase": "canary", "step": _RaisingStr()},
        "duration_is_a_hostile_object": {"phase": "canary", "duration_ms": _RaisingStr()},
    }

    def test_no_argument_can_make_record_failure_raise(self) -> None:
        for name, kwargs in self.ESCAPES.items():
            with self.subTest(escape=name):
                try:
                    record_failure(FailureCategory.UNCLASSIFIED, **kwargs)  # type: ignore[arg-type]
                except (KeyboardInterrupt, SystemExit):  # pragma: no cover
                    raise  # The two escapes the docstring reserves.
                except BaseException as exc:  # pragma: no cover - the defect itself
                    self.fail(f"record_failure raised {type(exc).__name__} for {name}")

    def test_no_argument_can_make_build_record_raise(self) -> None:
        for name, kwargs in self.ESCAPES.items():
            with self.subTest(escape=name):
                try:
                    record = build_record(FailureCategory.UNCLASSIFIED, **kwargs)  # type: ignore[arg-type]
                except (KeyboardInterrupt, SystemExit):  # pragma: no cover
                    raise
                except BaseException as exc:  # pragma: no cover - the defect itself
                    self.fail(f"build_record raised {type(exc).__name__} for {name}")
                self.assert_only_label_tokens(record)

    def test_a_hostile_mapping_is_tolerated(self) -> None:
        record = build_record(
            FailureCategory.UNCLASSIFIED,
            phase="canary",
            counts=_RaisingItems(),  # type: ignore[arg-type]
            flags=_RaisingItems(),  # type: ignore[arg-type]
        )
        self.assertEqual({}, record["counts"])
        self.assertEqual({}, record["flags"])

    def test_records_survive_the_escapes_and_stay_readable_json(self) -> None:
        """A swallowed failure must not leave a half-written sink line."""

        for kwargs in self.ESCAPES.values():
            record_failure(FailureCategory.UNCLASSIFIED, **kwargs)  # type: ignore[arg-type]
        path = Path(self._dir.name) / "vnext-failures.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        self.assertEqual(len(self.ESCAPES), len(lines))
        for line in lines:
            self.assert_only_label_tokens(json.loads(line))

    def test_non_finite_durations_never_poison_the_sink(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(duration=value):
                record = build_record(
                    FailureCategory.UNCLASSIFIED, phase="canary", duration_ms=value
                )
                self.assertIsNone(record["duration_ms"])
                # `allow_nan=False` would raise on any of these.
                json.dumps(record, allow_nan=False)


class SinkTests(LabelShapeAssertions):
    def setUp(self) -> None:
        self._prior = {name: os.environ.get(name) for name in (_ENABLE_ENV, _DIR_ENV)}
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self._prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_nothing_is_written_unless_diagnostics_are_armed(self) -> None:
        with TemporaryDirectory(prefix="vnext-diag-") as text:
            os.environ.pop(_ENABLE_ENV, None)
            os.environ[_DIR_ENV] = text
            self.assertIsNone(
                record_failure(FailureCategory.UNCLASSIFIED, phase="canary")
            )
            self.assertEqual([], list(Path(text).iterdir()))

    def test_records_are_appended_outside_any_caller_workspace(self) -> None:
        with TemporaryDirectory(prefix="vnext-diag-") as text:
            sink = Path(text) / "nested"
            os.environ[_ENABLE_ENV] = "1"
            os.environ[_DIR_ENV] = str(sink)
            first = record_failure(
                FailureCategory.SDK_NOT_INSTALLED,
                phase="claude_adapter",
                step="initialize",
                exception_type="ModuleNotFoundError",
                duration_ms=302.0,
            )
            second = record_failure(FailureCategory.BRIDGE_EXITED_AT_STARTUP, phase="canary")
            self.assertEqual(first, second)
            self.assertIsNotNone(first)
            written = [
                json.loads(line)
                for line in Path(str(first)).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(2, len(written))
            self.assertEqual(
                ["sdk_not_installed", "bridge_exited_at_startup"],
                [record["category"] for record in written],
            )
            for record in written:
                self.assert_only_label_tokens(record)

    def test_default_sink_is_the_gitignored_local_runtime_directory(self) -> None:
        os.environ.pop(_DIR_ENV, None)
        path = sink_path()
        self.assertEqual("vnext-failures.jsonl", path.name)
        self.assertEqual("diagnostics", path.parent.name)
        self.assertEqual("data", path.parent.parent.name)
        repo = path.parent.parent.parent
        # Ask git rather than reading .gitignore text: a later negation rule
        # (`!data/diagnostics/`) would satisfy a substring check while
        # leaving the sink committable.  git is the authority on the answer.
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=repo,
            capture_output=True,
        )
        if result.returncode == 128:  # pragma: no cover - no git available
            self.skipTest("git is unavailable to answer the ignore question")
        self.assertEqual(
            0, result.returncode, "the default diagnostic sink is not gitignored"
        )

    def test_a_relative_override_is_refused_in_favour_of_the_default(self) -> None:
        """A relative override resolves against the CWD.

        When that is the repository it drops an untracked JSONL into the
        source tree, which is neither gitignored nor outside the workspace.
        """

        self.assertFalse(override_is_usable("data/diagnostics"))
        self.assertFalse(override_is_usable("./somewhere"))
        self.assertFalse(override_is_usable(""))
        os.environ[_DIR_ENV] = "vnext-diagnostics-relative"
        path = sink_path()
        self.assertTrue(path.is_absolute())
        self.assertEqual("diagnostics", path.parent.name)
        self.assertEqual("data", path.parent.parent.name)

    def test_a_relative_override_lands_in_the_default_sink_not_the_cwd(self) -> None:
        """Assert where the record WENT, not only where it did not go.

        The version this replaces armed diagnostics, set a relative override,
        `chdir`-ed away, and checked that the working directory stayed empty.
        It did — because the refused override fell back to the DEFAULT sink,
        inside the repository, appending one record on every suite run.  The
        assertion that would have caught that is the one about where the record
        actually landed, so it is made here, and the default is redirected into
        a temporary directory so the test cannot write outside its own area.
        """

        with TemporaryDirectory(prefix="vnext-cwd-") as cwd_text:
            with TemporaryDirectory(prefix="vnext-diag-") as default_text:
                os.environ[_ENABLE_ENV] = "1"
                os.environ[_DIR_ENV] = "relative-sink"
                prior_cwd = os.getcwd()
                os.chdir(cwd_text)
                try:
                    with mock.patch.object(
                        vnext_diagnostics,
                        "_default_sink_dir",
                        lambda: Path(default_text),
                    ):
                        written = record_failure(
                            FailureCategory.UNCLASSIFIED, phase="canary"
                        )
                finally:
                    os.chdir(prior_cwd)
                self.assertEqual(
                    [],
                    list(Path(cwd_text).iterdir()),
                    "a relative override resolved against the working directory",
                )
                self.assertEqual(Path(default_text) / "vnext-failures.jsonl", written)
                self.assertEqual(
                    1,
                    len(Path(str(written)).read_text(encoding="utf-8").splitlines()),
                    "the refused override did not fall back to exactly one "
                    "record in the default sink",
                )

    def test_an_absolute_override_is_still_honoured(self) -> None:
        with TemporaryDirectory(prefix="vnext-diag-") as text:
            self.assertTrue(override_is_usable(text))
            os.environ[_DIR_ENV] = text
            self.assertEqual(Path(text) / "vnext-failures.jsonl", sink_path())

    def test_an_unwritable_sink_never_raises(self) -> None:
        with TemporaryDirectory(prefix="vnext-diag-") as text:
            blocker = Path(text) / "blocked"
            blocker.write_text("", encoding="utf-8")
            os.environ[_ENABLE_ENV] = "1"
            os.environ[_DIR_ENV] = str(blocker / "under-a-file")
            self.assertIsNone(record_failure(FailureCategory.UNCLASSIFIED, phase="canary"))


class DiagnosticsDisarmedTests(unittest.TestCase):
    """Switching diagnostics off must change nothing except the recording."""

    def setUp(self) -> None:
        self._prior = os.environ.get(_ENABLE_ENV)
        self.addCleanup(self._restore)
        os.environ.pop(_ENABLE_ENV, None)

    def _restore(self) -> None:
        if self._prior is None:
            os.environ.pop(_ENABLE_ENV, None)
        else:
            os.environ[_ENABLE_ENV] = self._prior

    def test_stderr_is_not_drained_when_diagnostics_are_off(self) -> None:
        """`drain_stderr` empties the buffer, so it must be gated.

        Running it unconditionally meant a fatal discarded the captured
        bridge stderr even with the channel disabled — an observable
        behaviour change caused purely by the diagnostic code path.
        """

        with TemporaryDirectory(prefix="vnext-diag-") as workspace:
            adapter = ClaudeCodeAdapter(workspace=workspace)
            adapter._stderr.append("first")
            adapter._stderr.append("second")
            adapter._set_fatal("Claude bridge stdout closed unexpectedly")
            self.assertEqual(
                2,
                len(adapter._stderr),
                "captured stderr was consumed while diagnostics were off",
            )
            self.assertEqual(
                "Claude bridge stdout closed unexpectedly", adapter._fatal
            )


class AdapterFunnelTests(LabelShapeAssertions):
    """Drive the adapter funnels without starting any provider process."""

    def setUp(self) -> None:
        self._prior = {name: os.environ.get(name) for name in (_ENABLE_ENV, _DIR_ENV)}
        self._dir = TemporaryDirectory(prefix="vnext-diag-")
        self.addCleanup(self._dir.cleanup)
        self.addCleanup(self._restore)
        os.environ[_ENABLE_ENV] = "1"
        os.environ[_DIR_ENV] = self._dir.name

    def _restore(self) -> None:
        for name, value in self._prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _records(self) -> list[dict[str, object]]:
        path = Path(self._dir.name) / "vnext-failures.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_missing_workspace_records_a_label_and_hides_the_path(self) -> None:
        missing = Path(self._dir.name) / "absent-workspace"
        with self.assertRaises(ClaudeRuntimeError) as caught:
            ClaudeCodeAdapter(workspace=str(missing))
        self.assertEqual("workspace does not exist", str(caught.exception))
        self.assertNotIn("absent-workspace", str(caught.exception))
        records = self._records()
        self.assertEqual(1, len(records))
        self.assertEqual("workspace_unavailable", records[0]["category"])
        self.assertEqual("adapter.init", records[0]["phase"])
        self.assertIsInstance(records[0]["duration_ms"], float)

    def test_fatal_consumes_stderr_and_records_the_refined_category(self) -> None:
        adapter = ClaudeCodeAdapter(workspace=self._dir.name)
        for line in ClassifyStderrTests.TRACEBACK:
            adapter._stderr.append(line)
        adapter._set_fatal("Claude bridge stdout closed unexpectedly")
        self.assertEqual(0, len(adapter._stderr))
        records = self._records()
        self.assertEqual(1, len(records))
        self.assertEqual("sdk_not_installed", records[0]["category"])
        self.assertEqual("claude_adapter", records[0]["phase"])
        self.assertEqual("bridge_stream", records[0]["step"])
        self.assertEqual(
            {"bridge_started": False, "closing": False, "initialized": False},
            records[0]["flags"],
        )
        adapter._set_fatal("a second fatal must not double-record")
        self.assertEqual(1, len(self._records()))
        self.assert_only_label_tokens(records[0])

    def test_the_send_step_names_the_in_flight_operation(self) -> None:
        """A write failure must say WHICH op was in flight, not just `send`."""

        self.assertEqual(
            "send.initialize", ClaudeCodeAdapter._send_step({"op": "initialize"})
        )
        self.assertEqual(
            "send.permission_response",
            ClaudeCodeAdapter._send_step({"op": "permission_response"}),
        )
        self.assertEqual("send_request", ClaudeCodeAdapter._send_step({}))
        self.assertEqual("send_request", ClaudeCodeAdapter._send_step({"op": 7}))

    def test_request_without_a_bridge_records_the_not_running_label(self) -> None:
        adapter = ClaudeCodeAdapter(workspace=self._dir.name)
        with self.assertRaises(ClaudeRuntimeError):
            adapter._request("initialize", {})
        records = self._records()
        self.assertEqual(1, len(records))
        self.assertEqual("bridge_not_running", records[0]["category"])
        self.assertEqual("initialize", records[0]["step"])


if __name__ == "__main__":
    unittest.main()
