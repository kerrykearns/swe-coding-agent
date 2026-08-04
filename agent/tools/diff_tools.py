"""Git-backed diff, status, and patch application inside a workspace.

Assumes the workspace root is already a git repository. All git invocations go
through :func:`~agent.tools.shell_exec.run_command`, so they inherit its cwd
pinning to the workspace root.
"""

from __future__ import annotations

import os
import tempfile

from .shell_exec import run_command
from .workspace import Workspace

__all__ = ["get_diff", "get_status", "apply_patch"]


def get_diff(ws: Workspace, staged: bool = False) -> str:
    """Return the raw unified diff of the workspace's working tree.

    Args:
        ws: Workspace pointing at a git repository.
        staged: Diff the index against HEAD (``git diff --cached``) instead of
            the working tree against the index.

    Returns:
        The diff text, or ``""`` when there are no changes.

    Raises:
        RuntimeError: If git fails (e.g. the workspace is not a repository).
    """
    command = "git --no-pager diff --no-color"
    if staged:
        command += " --cached"
    result = run_command(ws, command, timeout=60)
    _raise_if_git_failed(result, command)
    return result["stdout"]


def get_status(ws: Workspace) -> list[str]:
    """Return the paths of all changed files, via ``git status --porcelain``.

    Covers modified, added, deleted, renamed, and untracked files. For a
    rename (``R  old -> new``) the *new* path is reported, since that is the
    file that now exists on disk.

    Returns:
        A list of repository-relative paths, in the order git reported them.

    Raises:
        RuntimeError: If git fails (e.g. the workspace is not a repository).
    """
    command = "git status --porcelain"
    result = run_command(ws, command, timeout=60)
    _raise_if_git_failed(result, command)

    paths: list[str] = []
    for line in result["stdout"].splitlines():
        if not line.strip():
            continue
        # Porcelain v1 format: two status chars, a space, then the path.
        path = line[3:] if len(line) > 3 else line.strip()
        if " -> " in path:  # rename or copy: "old -> new"
            path = path.split(" -> ", 1)[1]
        paths.append(_unquote(path.strip()))
    return paths


def apply_patch(ws: Workspace, patch_text: str) -> dict:
    """Apply a unified diff to the workspace with ``git apply``.

    The patch is written to a temporary file outside the workspace (so it never
    shows up as a workspace change) and removed afterwards, whatever happens.

    Args:
        ws: Workspace pointing at a git repository.
        patch_text: A unified diff. A trailing newline is added if missing,
            since ``git apply`` rejects patches that lack one.

    Returns:
        ``{"success": bool, "error": str | None}``. On failure, ``error``
        carries git's own message (typically which hunk failed to apply).
    """
    if not patch_text.strip():
        return {"success": False, "error": "Patch is empty; nothing to apply."}

    if not patch_text.endswith("\n"):
        patch_text += "\n"

    handle, patch_path = tempfile.mkstemp(suffix=".patch", prefix="agent-patch-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as patch_file:
            patch_file.write(patch_text)

        result = run_command(ws, f'git apply "{patch_path}"', timeout=60)
        if result["timed_out"]:
            return {"success": False, "error": "git apply timed out."}
        if result["exit_code"] != 0:
            error = (result["stderr"] or result["stdout"]).strip()
            return {
                "success": False,
                "error": error or f"git apply failed with exit code {result['exit_code']}.",
            }
        return {"success": True, "error": None}
    finally:
        try:
            os.unlink(patch_path)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def _raise_if_git_failed(result: dict, command: str) -> None:
    """Turn a failed git invocation into a RuntimeError with git's own message."""
    if result["timed_out"]:
        raise RuntimeError(f"`{command}` timed out.")
    if result["exit_code"] != 0:
        message = (result["stderr"] or result["stdout"]).strip()
        raise RuntimeError(
            f"`{command}` failed with exit code {result['exit_code']}: "
            f"{message or 'no output'}"
        )


def _unquote(path: str) -> str:
    """Strip the quotes git adds around paths containing unusual characters."""
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        return path[1:-1]
    return path
