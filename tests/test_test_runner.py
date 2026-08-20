"""Tests for run_tests and its pytest-summary parser.

The end-to-end cases run against a throwaway copy of
demo/sample_repos/calculator, which is committed in a state where exactly one
test fails — so a correct parser must report 1 failed, 1 passed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.tools import Workspace, run_tests, write_file
from agent.tools.test_runner import _parse_counts


# --------------------------------------------------------------------------
# end to end, against the buggy fixture repo
# --------------------------------------------------------------------------


def test_detects_the_calculator_bug(calculator_ws: Workspace):
    result = run_tests(calculator_ws)
    # 13 of the fixture's 14 tests fail — every function but subtract() ships
    # with a deliberate bug; see demo/sample_repos/calculator/calculator/calc.py.
    assert result["failed"] == 13
    assert result["passed"] == 1
    assert result["errors"] == 0
    assert result["total"] == 14
    assert result["success"] is False
    assert result["timed_out"] is False
    assert "test_add" in result["raw_output"]


def test_scoped_to_a_single_test_file(calculator_ws: Workspace):
    result = run_tests(calculator_ws, "calculator/test_calc.py")
    assert result["failed"] == 13
    assert result["passed"] == 1


def test_reports_success_once_the_bug_is_fixed(calculator_ws: Workspace):
    """Same fixture, patched in the tmp copy: the parser must flip to success."""
    from .test_baseline import _ALL_BUGS_FIXED_CALC

    write_file(calculator_ws, "calculator/calc.py", _ALL_BUGS_FIXED_CALC)
    result = run_tests(calculator_ws)
    assert result["passed"] == 14
    assert result["failed"] == 0
    assert result["total"] == 14
    assert result["success"] is True
    assert result["exit_code"] == 0


# --------------------------------------------------------------------------
# end to end, edge cases
# --------------------------------------------------------------------------


def test_zero_tests_collected_is_not_success(ws: Workspace):
    write_file(ws, "not_a_test.py", "x = 1\n")
    result = run_tests(ws)
    assert result["total"] == 0
    assert result["passed"] == 0
    assert result["failed"] == 0
    assert result["success"] is False


def test_collection_error_is_counted_as_an_error(ws: Workspace):
    write_file(ws, "test_broken.py", "def test_x(:\n")  # syntax error
    result = run_tests(ws)
    assert result["errors"] >= 1
    assert result["success"] is False


def test_all_passing_workspace(ws: Workspace):
    write_file(ws, "test_ok.py", "def test_a():\n    assert True\n\n\ndef test_b():\n    assert 1 + 1 == 2\n")
    result = run_tests(ws)
    assert (result["passed"], result["failed"], result["total"]) == (2, 0, 2)
    assert result["success"] is True


# --------------------------------------------------------------------------
# sandboxed dispatch (mocked run_command, no real Docker needed)
# --------------------------------------------------------------------------


def test_sandboxed_uses_the_containers_python_not_sys_executable(ws: Workspace, monkeypatch):
    """Inside the container sys.executable names the *host* interpreter, which
    does not exist there — the sandbox image's own `python` must be used.
    """
    import agent.tools.test_runner as test_runner_module

    captured: dict = {}

    def fake_run_command(workspace, command, timeout=30, sandboxed=False, sandbox=None):
        captured["command"] = command
        captured["sandboxed"] = sandboxed
        captured["sandbox"] = sandbox
        return {"stdout": "1 passed in 0.01s", "stderr": "", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(test_runner_module, "run_command", fake_run_command)

    run_tests(ws, sandboxed=True)

    assert captured["command"].startswith("python -m pytest")
    assert captured["sandboxed"] is True


def test_not_sandboxed_still_pins_sys_executable(ws: Workspace, monkeypatch):
    import sys

    import agent.tools.test_runner as test_runner_module

    captured: dict = {}

    def fake_run_command(workspace, command, timeout=30, sandboxed=False, sandbox=None):
        captured["command"] = command
        return {"stdout": "1 passed in 0.01s", "stderr": "", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(test_runner_module, "run_command", fake_run_command)

    run_tests(ws, sandboxed=False)

    assert captured["command"].startswith(f'"{sys.executable}" -m pytest')


def test_an_open_sandbox_is_forwarded_to_run_command(ws: Workspace, monkeypatch):
    import agent.tools.test_runner as test_runner_module

    fake_sandbox = MagicMock()
    captured: dict = {}

    def fake_run_command(workspace, command, timeout=30, sandboxed=False, sandbox=None):
        captured["sandbox"] = sandbox
        return {"stdout": "1 passed in 0.01s", "stderr": "", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(test_runner_module, "run_command", fake_run_command)

    run_tests(ws, sandboxed=True, sandbox=fake_sandbox)

    assert captured["sandbox"] is fake_sandbox


def test_timeout_is_surfaced(calculator_ws: Workspace):
    result = run_tests(calculator_ws, timeout=0)
    assert result["timed_out"] is True
    assert result["success"] is False


# --------------------------------------------------------------------------
# summary parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("2 passed in 0.05s", {"passed": 2}),
        ("1 failed, 1 passed in 0.05s", {"failed": 1, "passed": 1}),
        ("2 passed, 1 failed in 0.05s", {"passed": 2, "failed": 1}),
        (
            "==== 1 failed, 2 passed, 3 errors in 1.23s ====",
            {"failed": 1, "passed": 2, "error": 3},
        ),
        ("1 error in 0.10s", {"error": 1}),
        ("3 passed, 1 skipped in 0.4s", {"passed": 3, "skipped": 1}),
        ("1 passed, 2 warnings in 0.02s", {"passed": 1, "warning": 2}),
        ("10 passed in 12.34 s", {"passed": 10}),
        ("no tests ran in 0.01s", {}),
        ("a duration but no counts in 0.01s", {}),
        ("", {}),
        ("this output has no summary line at all", {}),
    ],
)
def test_parse_counts(line: str, expected: dict):
    assert _parse_counts(line) == expected


def test_parse_counts_uses_the_last_summary_line():
    """Test output can echo earlier runs; the final summary is the real one."""
    output = "3 passed in 9.99s\nsome later noise\n1 failed, 1 passed in 0.05s\n"
    assert _parse_counts(output) == {"failed": 1, "passed": 1}


def test_parse_counts_ignores_tracebacks_and_progress_lines():
    output = (
        "collected 2 items\n"
        "calculator/test_calc.py .F\n"
        "E       assert 5 == -1\n"
        "1 failed, 1 passed in 0.05s\n"
    )
    assert _parse_counts(output) == {"failed": 1, "passed": 1}
