"""Run pytest inside a workspace and parse its summary into structured counts.

This is the agent's feedback signal: it needs "2 passed, 1 failed" as data,
not as prose to re-read every turn.
"""

from __future__ import annotations

import re
import sys
from typing import Optional

from .sandbox import SandboxContainer
from .shell_exec import run_command
from .workspace import Workspace

__all__ = ["run_tests"]

#: pytest exit code 5 == "no tests were collected".
_EXIT_NO_TESTS_COLLECTED = 5

#: Individual "<n> <outcome>" pairs inside a pytest summary line.
_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b"
)

#: A pytest summary line always ends with a duration, e.g. "in 0.05s".
_DURATION_RE = re.compile(r"\bin\s+[\d.]+\s*s")

#: Outcomes that represent a test that actually existed and was accounted for.
_TEST_OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")


def run_tests(
    ws: Workspace,
    test_path: str = ".",
    timeout: int = 300,
    sandboxed: bool = False,
    sandbox: Optional[SandboxContainer] = None,
) -> dict:
    """Run ``pytest {test_path} --tb=short -q`` in the workspace and parse it.

    Outside of ``sandboxed=True``, pytest is invoked as
    ``{sys.executable} -m pytest`` rather than a bare ``pytest``: it is the
    same command, but it pins the run to the interpreter we are already using
    instead of whatever happens to be on ``PATH``, which matters when the
    workspace is a checkout with no activated virtualenv. When sandboxed,
    ``sys.executable`` names the host interpreter, not one that exists inside
    the container, so the sandbox image's own ``python`` is used instead —
    the image's Dockerfile pins the version that runs there.

    Args:
        ws: Workspace to run in; pytest's cwd is the workspace root (or, when
            sandboxed, the directory mounted into the container).
        test_path: File or directory to test, relative to the workspace root.
        timeout: Seconds before the run is killed (test suites are slow).
        sandboxed: Run pytest inside a Docker container with no network
            access (see :mod:`agent.tools.sandbox`) instead of locally.
        sandbox: An already-open
            :class:`~agent.tools.sandbox.SandboxContainer` to run in. Only
            meaningful when ``sandboxed`` is True; see
            :func:`agent.tools.shell_exec.run_command` for when to pass one.

    Returns:
        A dict with:

        * ``passed`` / ``failed`` / ``errors`` — test counts (ints).
        * ``total`` — tests accounted for, including skips and xfails.
        * ``success`` — True only if pytest exited 0, nothing failed or
          errored, and the run did not time out. A run that collected zero
          tests is *not* a success: nothing was verified.
        * ``raw_output`` — stdout and stderr, for showing the agent tracebacks.
        * ``exit_code`` / ``timed_out`` — the underlying process result.
    """
    python_bin = "python" if sandboxed else f'"{sys.executable}"'
    command = f"{python_bin} -m pytest {test_path} --tb=short -q"
    result = run_command(
        ws, command, timeout=timeout, sandboxed=sandboxed, sandbox=sandbox
    )

    raw_output = result["stdout"]
    if result["stderr"]:
        raw_output = f"{raw_output}\n{result['stderr']}" if raw_output else result["stderr"]

    counts = _parse_counts(raw_output)
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    errors = counts.get("error", 0)
    total = sum(counts.get(key, 0) for key in _TEST_OUTCOMES)

    collected_nothing = (
        result["exit_code"] == _EXIT_NO_TESTS_COLLECTED or total == 0
    )
    success = (
        not result["timed_out"]
        and result["exit_code"] == 0
        and failed == 0
        and errors == 0
        and not collected_nothing
    )

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "success": success,
        "raw_output": raw_output,
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
    }


def _parse_counts(output: str) -> dict[str, int]:
    """Extract outcome counts from pytest's summary line.

    Scans from the bottom of the output for the last line that looks like a
    summary (contains a duration such as ``in 0.05s``) and carries at least one
    ``<n> <outcome>`` pair. This tolerates both the ``-q`` form
    (``1 failed, 1 passed in 0.05s``) and the banner form
    (``==== 1 failed, 1 passed in 0.05s ====``), and returns an empty mapping
    for ``no tests ran in 0.01s``.
    """
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if not _DURATION_RE.search(stripped):
            continue
        if "no tests ran" in stripped:
            return {}
        found = _COUNT_RE.findall(stripped)
        if found:
            # Singularise so "1 error" and "2 errors" land on the same key.
            return {
                outcome.rstrip("s") if outcome.startswith(("error", "warning")) else outcome: int(number)
                for number, outcome in found
            }
    return {}
