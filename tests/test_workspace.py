"""Tests for Workspace — the containment gatekeeper.

The safety path matters more here than anywhere else in the tool layer: if
resolve() can be talked into returning a path outside the root, every other
tool leaks with it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent.tools import Workspace, WorkspaceViolation


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_root_is_absolute_and_resolved(tmp_path: Path):
    workspace = Workspace(tmp_path)
    assert workspace.root.is_absolute()
    assert workspace.root == tmp_path.resolve()


def test_create_makes_missing_root(tmp_path: Path):
    root = tmp_path / "nested" / "workspace"
    workspace = Workspace(root, create=True)
    assert workspace.root.is_dir()


def test_missing_root_without_create_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Workspace(tmp_path / "does_not_exist")


def test_file_as_root_raises(tmp_path: Path):
    target = tmp_path / "a_file.txt"
    target.write_text("hi", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        Workspace(target)


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_resolve_relative_path(ws: Workspace):
    assert ws.resolve("file.txt") == ws.root / "file.txt"


def test_resolve_nested_path(ws: Workspace):
    assert ws.resolve("pkg/module.py") == ws.root / "pkg" / "module.py"


def test_resolve_dot_is_root(ws: Workspace):
    assert ws.resolve(".") == ws.root
    assert ws.resolve() == ws.root


def test_resolve_accepts_path_objects(ws: Workspace):
    assert ws.resolve(Path("pkg") / "module.py") == ws.root / "pkg" / "module.py"


def test_resolve_allows_nonexistent_paths(ws: Workspace):
    """Writes need to resolve paths that do not exist yet."""
    resolved = ws.resolve("brand/new/file.txt")
    assert not resolved.exists()
    assert ws.root in resolved.parents


def test_interior_dotdot_that_stays_inside_is_allowed(ws: Workspace):
    assert ws.resolve("pkg/sub/../module.py") == ws.root / "pkg" / "module.py"


def test_absolute_path_inside_root_is_allowed(ws: Workspace):
    inside = ws.root / "pkg" / "module.py"
    assert ws.resolve(str(inside)) == inside


def test_relative_round_trips(ws: Workspace):
    assert ws.relative("pkg/module.py") == "pkg/module.py"
    assert ws.relative(str(ws.root / "pkg" / "module.py")) == "pkg/module.py"
    assert ws.relative(".") == "."


def test_exists(ws: Workspace):
    (ws.root / "here.txt").write_text("x", encoding="utf-8")
    assert ws.exists("here.txt")
    assert not ws.exists("gone.txt")


# --------------------------------------------------------------------------
# safety path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile_path",
    [
        "../outside.txt",
        "../../etc/passwd",
        "../../../../../../etc/passwd",
        "pkg/../../outside.txt",
        "./../outside.txt",
        "..",
    ],
)
def test_traversal_is_rejected(ws: Workspace, hostile_path: str):
    with pytest.raises(WorkspaceViolation):
        ws.resolve(hostile_path)


def test_traversal_error_names_the_offending_path(ws: Workspace):
    with pytest.raises(WorkspaceViolation, match="outside the workspace root"):
        ws.resolve("../../etc/passwd")


def test_absolute_path_outside_root_is_rejected(tmp_path: Path, ws: Workspace):
    """tmp_path is the workspace's parent, so it is out of bounds."""
    secret = tmp_path / "secret.txt"
    secret.write_text("classified", encoding="utf-8")
    with pytest.raises(WorkspaceViolation):
        ws.resolve(str(secret))


def test_nul_byte_is_rejected(ws: Workspace):
    with pytest.raises(WorkspaceViolation, match="NUL"):
        ws.resolve("file\x00.txt")


def _link_to_directory(link: Path, target: Path) -> None:
    """Create a directory link at ``link`` pointing at ``target``.

    Prefers a real symlink; falls back to a Windows directory junction, which
    (unlike a symlink) needs no elevation, so this safety check still runs on an
    ordinary Windows dev box. Skips the test if neither is available.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError) as symlink_error:
        if sys.platform != "win32":
            pytest.skip(f"cannot create symlinks here: {symlink_error}")

    junction = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if junction.returncode != 0 or not link.exists():
        pytest.skip(
            "cannot create a symlink or junction here: "
            f"{symlink_error}; mklink said {junction.stderr.strip() or junction.stdout.strip()}"
        )


def test_symlink_escape_is_rejected(tmp_path: Path, ws: Workspace):
    """A link inside the workspace pointing out of it must not be followed."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified", encoding="utf-8")

    _link_to_directory(ws.root / "escape_hatch", outside)

    # Sanity check: the link really does reach the file, so the rejection below
    # is containment working and not just a broken link.
    assert (ws.root / "escape_hatch" / "secret.txt").read_text(encoding="utf-8") == (
        "classified"
    )

    with pytest.raises(WorkspaceViolation):
        ws.resolve("escape_hatch/secret.txt")


def test_root_itself_is_in_bounds(ws: Workspace):
    """Containment must accept the root, not just its children."""
    assert ws.resolve("") == ws.root
