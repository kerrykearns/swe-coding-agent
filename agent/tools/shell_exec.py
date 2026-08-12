"""Shell command execution, scoped to a workspace root.

SAFETY NOTE: this function executes arbitrary shell commands with no risk
filtering. The safety/refusal layer (M6) will wrap this before the agent is
allowed to call it unsupervised. Do not expose this directly to untrusted
input.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from .sandbox import SandboxContainer
from .workspace import Workspace

__all__ = ["run_command"]


def run_command(
    ws: Workspace,
    command: str,
    timeout: int = 30,
    sandboxed: bool = False,
    sandbox: Optional[SandboxContainer] = None,
) -> dict:
    """Run ``command`` with cwd set to ``ws.root``, locally or in a sandbox.

    SAFETY NOTE: outside of ``sandboxed=True``, this function executes
    arbitrary shell commands with no risk filtering — no allowlist, no
    denylist. Anything the current OS user can do, the command can do,
    including reaching *outside* the workspace (``ws`` only pins the working
    directory; it does not confine the child process). The safety/refusal
    layer (M6) will wrap this before the agent is allowed to call it
    unsupervised. Do not expose this directly to untrusted input.

    Args:
        ws: Workspace whose root becomes the child process's cwd (or, when
            sandboxed, the directory mounted into the container).
        command: Shell command line, run via the platform shell (``cmd.exe``
            on Windows, ``/bin/sh`` elsewhere) when not sandboxed, or always
            via ``/bin/sh`` inside the container when sandboxed.
        timeout: Seconds to wait before killing the process.
        sandboxed: Run inside a Docker container with no network access
            (see :mod:`agent.tools.sandbox`) instead of a local subprocess.
        sandbox: An already-open :class:`~agent.tools.sandbox.SandboxContainer`
            to run in. Only meaningful when ``sandboxed`` is True. Pass this
            when a caller needs several commands in the same container (e.g.
            one per agent turn) — starting a container per call is slow.
            When omitted (and ``sandboxed`` is True), a container is opened
            and torn down just for this one call.

    Returns:
        ``{"stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}``.
        On timeout the process is killed, ``timed_out`` is True, ``exit_code``
        is -1, and any output captured before the kill is still returned.
        This function does not raise on command failure or timeout; check
        ``exit_code`` and ``timed_out``.
    """
    if sandboxed:
        if sandbox is not None:
            return sandbox.run(command, timeout=timeout)
        with SandboxContainer(ws) as opened:
            return opened.run(command, timeout=timeout)

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(ws.root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "stdout": _as_text(exc.stdout),
            "stderr": _as_text(exc.stderr)
            + f"\nCommand timed out after {timeout} seconds and was killed.",
            "exit_code": -1,
            "timed_out": True,
        }

    return {
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "exit_code": completed.returncode,
        "timed_out": False,
    }


def _as_text(stream: str | bytes | None) -> str:
    """Normalise partial output from TimeoutExpired, which may be bytes or None."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream
