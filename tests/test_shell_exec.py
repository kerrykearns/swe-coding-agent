"""Tests for run_command.

Commands are written as ``python -c ...`` rather than shell builtins so the
suite behaves identically on Windows (cmd.exe) and POSIX (/bin/sh).
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.tools import Workspace, run_command
from agent.tools.shell_exec import _as_text

PY = f'"{sys.executable}"'


def test_returns_the_documented_keys(ws: Workspace):
    result = run_command(ws, f"{PY} -c \"print('hi')\"")
    assert set(result) == {"stdout", "stderr", "exit_code", "timed_out"}


def test_captures_stdout(ws: Workspace):
    result = run_command(ws, f"{PY} -c \"print('hello from the shell')\"")
    assert "hello from the shell" in result["stdout"]
    assert result["exit_code"] == 0
    assert result["timed_out"] is False


def test_captures_stderr(ws: Workspace):
    result = run_command(ws, f"{PY} -c \"import sys; sys.stderr.write('boom')\"")
    assert "boom" in result["stderr"]
    assert result["exit_code"] == 0


def test_propagates_nonzero_exit_code(ws: Workspace):
    result = run_command(ws, f'{PY} -c "import sys; sys.exit(3)"')
    assert result["exit_code"] == 3
    assert result["timed_out"] is False


def test_failing_command_does_not_raise(ws: Workspace):
    """A bad command is data, not an exception — the agent must see the error."""
    result = run_command(ws, "definitely_not_a_real_command_xyz")
    assert result["exit_code"] != 0
    assert result["timed_out"] is False


def test_runs_with_cwd_at_workspace_root(ws: Workspace):
    result = run_command(ws, f'{PY} -c "import os; print(os.getcwd())"')
    assert Path(result["stdout"].strip()).resolve() == ws.root


def test_sees_workspace_files(ws: Workspace):
    (ws.root / "marker.txt").write_text("found me", encoding="utf-8")
    result = run_command(ws, f'{PY} -c "print(open(\'marker.txt\').read())"')
    assert "found me" in result["stdout"]


def test_timeout_is_reported_not_raised(ws: Workspace):
    result = run_command(ws, f'{PY} -c "import time; time.sleep(30)"', timeout=1)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert "timed out" in result["stderr"]


def test_partial_output_normalisation():
    """TimeoutExpired hands back str, bytes, or None depending on how it died."""
    assert _as_text(None) == ""
    assert _as_text("already text") == "already text"
    assert _as_text(b"raw bytes") == "raw bytes"
    assert _as_text(b"bad \xff byte") == "bad � byte"


def test_multiline_output_preserved(ws: Workspace):
    result = run_command(ws, f'{PY} -c "print(1); print(2); print(3)"')
    assert [line.strip() for line in result["stdout"].splitlines() if line.strip()] == [
        "1",
        "2",
        "3",
    ]
