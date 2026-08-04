"""Workspace containment: the single gatekeeper for all filesystem access.

Every tool in this package resolves paths through :class:`Workspace` so that a
misbehaving (or adversarial) agent cannot read or write outside the directory it
was handed. Nothing here knows anything about LLMs — it is pure pathlib.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["Workspace", "WorkspaceViolation"]


class WorkspaceViolation(Exception):
    """Raised when a path would resolve outside the workspace root."""


class Workspace:
    """A rooted view of the filesystem.

    Args:
        root: Directory the agent is allowed to touch. Resolved eagerly, so
            symlinks in the root itself are collapsed once, up front.
        create: Create ``root`` (and parents) if it does not exist yet.

    Raises:
        NotADirectoryError: If ``root`` exists but is not a directory.
        FileNotFoundError: If ``root`` does not exist and ``create`` is False.
    """

    def __init__(self, root: str | Path, create: bool = False) -> None:
        root_path = Path(root).expanduser()
        if create:
            root_path.mkdir(parents=True, exist_ok=True)
        resolved = root_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Workspace root is not a directory: {resolved}")
        self.root: Path = resolved

    def resolve(self, relative_path: str | Path = ".") -> Path:
        """Resolve ``relative_path`` against the root, enforcing containment.

        Containment is checked on the *fully resolved* path (symlinks collapsed,
        ``..`` segments applied), not by string-matching for ``".."`` — so
        ``"a/../../etc/passwd"`` and a symlink pointing outside the workspace
        are both rejected. Absolute paths are permitted only if they already
        point inside the root.

        Returns:
            An absolute :class:`~pathlib.Path` inside the workspace. The path
            need not exist; callers decide whether that matters.

        Raises:
            WorkspaceViolation: If the path escapes the root, or is not a
                usable path (empty-after-strip, NUL bytes, etc.).
        """
        if isinstance(relative_path, str) and "\x00" in relative_path:
            raise WorkspaceViolation("Path contains a NUL byte")

        candidate = Path(relative_path)
        # Path("/x") and Path("C:/x") replace the root under `/`, so absolute
        # inputs are handled by the containment check below rather than joined.
        target = candidate if candidate.is_absolute() else self.root / candidate

        try:
            resolved = target.resolve()
        except (OSError, ValueError) as exc:  # malformed path, loops, etc.
            raise WorkspaceViolation(
                f"Could not resolve path {relative_path!r}: {exc}"
            ) from exc

        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceViolation(
                f"Path {relative_path!r} resolves to {resolved}, "
                f"which is outside the workspace root {self.root}"
            )
        return resolved

    def relative(self, path: str | Path) -> str:
        """Return ``path`` as a POSIX-style path relative to the root.

        Useful for reporting paths back to the agent without leaking absolute
        host paths. Goes through :meth:`resolve`, so it enforces containment.
        """
        resolved = self.resolve(path)
        if resolved == self.root:
            return "."
        return resolved.relative_to(self.root).as_posix()

    def exists(self, relative_path: str | Path) -> bool:
        """True if ``relative_path`` exists inside the workspace."""
        return self.resolve(relative_path).exists()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Workspace(root={str(self.root)!r})"
