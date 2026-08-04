"""File read/write/list/search primitives, all mediated by a :class:`Workspace`.

Every function takes a ``Workspace`` as its first argument and routes every
path through ``ws.resolve()``. No function here ever touches a raw path.
Errors are raised as standard exception types with messages phrased for an
agent to read and recover from, rather than surfacing raw tracebacks.
"""

from __future__ import annotations

from pathlib import Path

from .workspace import Workspace

__all__ = ["read_file", "write_file", "list_files", "search_text"]

#: Directories skipped when walking the workspace. They are large, generated,
#: and never what an agent means by "the code".
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".idea",
        ".eggs",
        "htmlcov",
    }
)


def _is_skipped(path: Path, root: Path) -> bool:
    """True if any directory component between ``root`` and ``path`` is noise."""
    return any(part in _SKIP_DIRS for part in path.relative_to(root).parts[:-1])


def read_file(ws: Workspace, path: str) -> str:
    """Return the full text of ``path`` (UTF-8).

    Raises:
        WorkspaceViolation: If ``path`` escapes the workspace.
        FileNotFoundError: If the file does not exist, with the workspace root
            included so the caller can tell *where* it looked.
        IsADirectoryError: If ``path`` names a directory.
        ValueError: If the file is not valid UTF-8 text (i.e. binary).
    """
    target = ws.resolve(path)
    if not target.exists():
        raise FileNotFoundError(
            f"No such file: {ws.relative(path)!r} (workspace root: {ws.root}). "
            "Use list_files() to see what exists."
        )
    if target.is_dir():
        raise IsADirectoryError(
            f"{ws.relative(path)!r} is a directory, not a file. "
            "Use list_files() to list its contents."
        )
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{ws.relative(path)!r} is not valid UTF-8 text "
            "(it looks like a binary file), so it cannot be read as source."
        ) from exc


def write_file(ws: Workspace, path: str, content: str) -> None:
    """Write ``content`` to ``path``, creating parent directories as needed.

    Overwrites existing files. Newlines are written verbatim (no platform
    translation) so content round-trips byte-for-byte through
    :func:`read_file` and stays stable inside diffs.

    Raises:
        WorkspaceViolation: If ``path`` escapes the workspace.
        IsADirectoryError: If ``path`` names an existing directory.
    """
    target = ws.resolve(path)
    if target.is_dir():
        raise IsADirectoryError(
            f"Cannot write to {ws.relative(path)!r}: it is an existing directory."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def list_files(ws: Workspace, subdir: str = ".", pattern: str = "*") -> list[str]:
    """List files under ``subdir`` matching glob ``pattern``, recursively.

    Args:
        ws: Workspace to search within.
        subdir: Directory to start from, relative to the workspace root.
        pattern: Glob matched against each file name, e.g. ``"*.py"``.

    Returns:
        Sorted POSIX-style paths relative to the *workspace root* (not to
        ``subdir``), so they can be fed straight back into the other tools.
        Directories, and anything inside a generated directory such as
        ``.git`` or ``__pycache__``, are omitted.

    Raises:
        WorkspaceViolation: If ``subdir`` escapes the workspace.
        FileNotFoundError: If ``subdir`` does not exist.
        NotADirectoryError: If ``subdir`` is not a directory.
    """
    base = ws.resolve(subdir)
    if not base.exists():
        raise FileNotFoundError(
            f"No such directory: {ws.relative(subdir)!r} (workspace root: {ws.root})"
        )
    if not base.is_dir():
        raise NotADirectoryError(f"{ws.relative(subdir)!r} is not a directory")

    found = [
        entry.relative_to(ws.root).as_posix()
        for entry in base.rglob(pattern)
        if entry.is_file() and not _is_skipped(entry, ws.root)
    ]
    return sorted(found)


def search_text(
    ws: Workspace,
    query: str,
    subdir: str = ".",
    pattern: str = "*",
    max_results: int = 500,
) -> list[dict]:
    """Find every line under ``subdir`` containing the substring ``query``.

    Args:
        ws: Workspace to search within.
        query: Literal substring to look for (not a regex).
        subdir: Directory to search, relative to the workspace root.
        pattern: Glob restricting which files are searched, e.g. ``"*.py"``.
        max_results: Stop after this many matches, so a broad query cannot
            return an unbounded blob.

    Returns:
        A list of ``{"file", "line_number", "line_content"}`` dicts, where
        ``file`` is relative to the workspace root, ``line_number`` is
        1-indexed, and ``line_content`` has its trailing newline stripped.
        Files that are not UTF-8 text or cannot be opened are skipped
        silently — a search should never fail on account of one bad file.

    Raises:
        WorkspaceViolation: If ``subdir`` escapes the workspace.
        ValueError: If ``query`` is empty.
    """
    if not query:
        raise ValueError("search_text() requires a non-empty query")

    matches: list[dict] = []
    for relative in list_files(ws, subdir, pattern):
        target = ws.resolve(relative)
        try:
            text = target.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(
                    {
                        "file": relative,
                        "line_number": line_number,
                        "line_content": line,
                    }
                )
                if len(matches) >= max_results:
                    return matches
    return matches
