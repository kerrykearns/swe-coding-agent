"""Shared fixtures for the tool-layer tests.

Every fixture builds its workspace under pytest's ``tmp_path``, so no test
writes to the real project tree or to the ``demo/sample_repos`` fixtures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.tools import Workspace

#: The buggy fixture repo shipped with the project. Tests copy it; they never
#: run against it in place, which would leave __pycache__/.pytest_cache behind.
CALCULATOR_FIXTURE = (
    Path(__file__).resolve().parent.parent / "demo" / "sample_repos" / "calculator"
)

_COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command in ``root``, raising with git's message on failure."""
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )


def init_repo(root: Path, message: str = "initial commit") -> None:
    """Turn ``root`` into a git repo and commit whatever is already in it.

    ``core.autocrlf=false`` keeps line endings byte-stable on Windows, so
    generated diffs apply cleanly; identity and signing are set locally so the
    tests do not depend on the developer's global git config.
    """
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "user.name", "Tool Layer Tests")
    git(root, "config", "user.email", "tests@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    """An empty workspace nested inside tmp_path.

    Nested deliberately: ``tmp_path`` itself is then *outside* the workspace,
    which gives the containment tests a real neighbouring directory to try to
    escape into.
    """
    return Workspace(tmp_path / "workspace", create=True)


@pytest.fixture
def git_ws(tmp_path: Path) -> Workspace:
    """A workspace that is a git repo with one committed file, ``tracked.txt``."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "tracked.txt").write_text("one\n", encoding="utf-8", newline="")
    init_repo(root)
    return Workspace(root)


@pytest.fixture
def calculator_ws(tmp_path: Path) -> Workspace:
    """A throwaway copy of demo/sample_repos/calculator, committed in its buggy state."""
    root = tmp_path / "calculator"
    shutil.copytree(CALCULATOR_FIXTURE, root, ignore=_COPY_IGNORE)
    init_repo(root, "buggy calculator")
    return Workspace(root)
