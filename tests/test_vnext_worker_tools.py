from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from vnext.vnext_worker_tools import (
    RequiredToolEffect,
    RequiredToolEffectError,
    normalize_command_cwd,
    required_effect_contract_facts,
    validate_required_tool_effect,
)


def valid_contract() -> dict:
    return {
        "tool": "run_command",
        "arguments": {
            "argv": [sys.executable, "-m", "unittest", "-v"],
            "cwd": ".",
            "timeout_seconds": 60,
        },
        "completion": {"exit_code": 0},
    }


class RequiredToolEffectValidationTests(unittest.TestCase):
    def test_valid_closed_contract_is_accepted_and_normalized(self) -> None:
        contract = validate_required_tool_effect(valid_contract())
        self.assertIsInstance(contract, RequiredToolEffect)
        self.assertEqual("run_command", contract.tool)
        self.assertEqual((sys.executable, "-m", "unittest", "-v"), contract.argv)
        self.assertEqual(60.0, contract.timeout_seconds)

    def test_unknown_tool_and_bad_top_level_shapes_are_rejected(self) -> None:
        unknown = valid_contract()
        unknown["tool"] = "read_file"
        extra = valid_contract()
        extra["policy"] = {}
        missing = valid_contract()
        del missing["completion"]
        for value in (unknown, extra, missing, ["run_command"]):
            with self.subTest(value=value):
                with self.assertRaises(RequiredToolEffectError):
                    validate_required_tool_effect(value)

    def test_bad_argument_shapes_are_rejected(self) -> None:
        bad_values = (
            ("argv", []),
            ("argv", "python"),
            ("argv", ["python", 3]),
            ("argv", ["bad\x00value"]),
            ("cwd", ""),
            ("cwd", None),
            ("timeout_seconds", 0),
            ("timeout_seconds", 121),
            ("timeout_seconds", True),
        )
        for key, value in bad_values:
            contract = valid_contract()
            contract["arguments"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(RequiredToolEffectError):
                    validate_required_tool_effect(contract)

    def test_argv_and_completion_bounds_are_closed(self) -> None:
        too_many = valid_contract()
        too_many["arguments"]["argv"] = ["python"] + [f"arg-{index}" for index in range(64)]
        with self.assertRaises(RequiredToolEffectError):
            validate_required_tool_effect(too_many)
        boundary = valid_contract()
        boundary["arguments"]["argv"] = ["python"] + [f"arg-{index}" for index in range(63)]
        self.assertEqual(64, len(validate_required_tool_effect(boundary).argv))
        for value in ("0", 0.0, True, None):
            contract = valid_contract()
            contract["completion"]["exit_code"] = value
            with self.subTest(exit_code=value):
                with self.assertRaises(RequiredToolEffectError):
                    validate_required_tool_effect(contract)

    def test_contract_facts_normalize_workspace_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / "sub").mkdir()
            contract = validate_required_tool_effect(valid_contract())
            self.assertEqual(
                {
                    "tool": "run_command",
                    "argv": list(contract.argv),
                    "cwd": ".",
                    "timeout_seconds": 60.0,
                    "exit_code": 0,
                },
                required_effect_contract_facts(contract, workspace=workspace),
            )
            self.assertEqual("sub", normalize_command_cwd(workspace, "sub/"))
            self.assertIsNone(normalize_command_cwd(workspace, ".."))


if __name__ == "__main__":
    unittest.main()
