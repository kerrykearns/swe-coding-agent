"""Tests for git diff / status / apply, against throwaway repos in tmp_path."""

from __future__ import annotations

import pytest

from agent.tools import (
    Workspace,
    apply_patch,
    get_diff,
    get_status,
    read_file,
    write_file,
)
from agent.tools import diff_tools

from .conftest import git


def _timed_out_command(*args, **kwargs) -> dict:
    """Stand-in for run_command that reports a killed process."""
    return {"stdout": "", "stderr": "killed", "exit_code": -1, "timed_out": True}


# --------------------------------------------------------------------------
# get_diff
# --------------------------------------------------------------------------


def test_diff_is_empty_on_clean_repo(git_ws: Workspace):
    assert get_diff(git_ws) == ""


def test_diff_shows_modified_lines(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    diff = get_diff(git_ws)
    assert "tracked.txt" in diff
    assert "-one" in diff
    assert "+two" in diff


def test_diff_ignores_untracked_files(git_ws: Workspace):
    """`git diff` is about tracked changes; get_status is what sees new files."""
    write_file(git_ws, "brand_new.txt", "hello\n")
    assert get_diff(git_ws) == ""


def test_diff_staged(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    git(git_ws.root, "add", "tracked.txt")
    assert get_diff(git_ws) == ""  # nothing left in the working tree
    assert "+two" in get_diff(git_ws, staged=True)


def test_diff_on_the_calculator_fixture(calculator_ws: Workspace):
    write_file(
        calculator_ws,
        "calculator/calc.py",
        "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n",
    )
    diff = get_diff(calculator_ws)
    assert "calculator/calc.py" in diff
    assert "-    return a - b" in diff
    assert "+    return a + b" in diff


def test_diff_outside_a_repo_raises(ws: Workspace):
    with pytest.raises(RuntimeError, match="git"):
        get_diff(ws)


# --------------------------------------------------------------------------
# get_status
# --------------------------------------------------------------------------


def test_status_is_empty_on_clean_repo(git_ws: Workspace):
    assert get_status(git_ws) == []


def test_status_lists_modified_files(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "changed\n")
    assert get_status(git_ws) == ["tracked.txt"]


def test_status_lists_untracked_files(git_ws: Workspace):
    write_file(git_ws, "pkg/new_module.py", "x = 1\n")
    # Untracked directories are reported as the directory itself until staged.
    git(git_ws.root, "add", "pkg/new_module.py")
    assert get_status(git_ws) == ["pkg/new_module.py"]


def test_status_lists_deleted_files(git_ws: Workspace):
    git_ws.resolve("tracked.txt").unlink()
    assert get_status(git_ws) == ["tracked.txt"]


def test_status_reports_new_path_for_renames(git_ws: Workspace):
    git(git_ws.root, "mv", "tracked.txt", "renamed.txt")
    assert get_status(git_ws) == ["renamed.txt"]


def test_status_handles_paths_with_spaces(git_ws: Workspace):
    write_file(git_ws, "a file with spaces.txt", "hi\n")
    git(git_ws.root, "add", "-A")
    assert get_status(git_ws) == ["a file with spaces.txt"]


def test_status_lists_several_files(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "changed\n")
    write_file(git_ws, "added.txt", "new\n")
    assert set(get_status(git_ws)) == {"tracked.txt", "added.txt"}


def test_status_outside_a_repo_raises(ws: Workspace):
    with pytest.raises(RuntimeError, match="git"):
        get_status(ws)


# --------------------------------------------------------------------------
# apply_patch
# --------------------------------------------------------------------------


def test_apply_patch_round_trip(git_ws: Workspace):
    """Diff a change, revert it, re-apply the patch, and get the change back."""
    write_file(git_ws, "tracked.txt", "two\nthree\n")
    patch = get_diff(git_ws)
    git(git_ws.root, "checkout", "--", "tracked.txt")
    assert read_file(git_ws, "tracked.txt") == "one\n"

    result = apply_patch(git_ws, patch)
    assert result == {"success": True, "error": None}
    assert read_file(git_ws, "tracked.txt") == "two\nthree\n"


def test_apply_patch_that_adds_a_new_file(git_ws: Workspace):
    write_file(git_ws, "brand_new.txt", "fresh\n")
    git(git_ws.root, "add", "brand_new.txt")
    patch = get_diff(git_ws, staged=True)
    git(git_ws.root, "rm", "-q", "-f", "--cached", "brand_new.txt")
    git_ws.resolve("brand_new.txt").unlink()

    assert apply_patch(git_ws, patch)["success"] is True
    assert read_file(git_ws, "brand_new.txt") == "fresh\n"


def test_apply_patch_tolerates_missing_trailing_newline(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    patch = get_diff(git_ws).rstrip("\n")
    git(git_ws.root, "checkout", "--", "tracked.txt")

    assert apply_patch(git_ws, patch)["success"] is True
    assert read_file(git_ws, "tracked.txt") == "two\n"


def test_apply_patch_reports_conflict_instead_of_raising(git_ws: Workspace):
    """A patch whose context does not match must fail cleanly with git's message."""
    write_file(git_ws, "tracked.txt", "two\n")
    patch = get_diff(git_ws)
    git(git_ws.root, "checkout", "--", "tracked.txt")
    write_file(git_ws, "tracked.txt", "something else entirely\n")

    result = apply_patch(git_ws, patch)
    assert result["success"] is False
    assert result["error"]


def test_apply_patch_rejects_malformed_input(git_ws: Workspace):
    result = apply_patch(git_ws, "this is not a diff at all\n")
    assert result["success"] is False
    assert result["error"]


def test_apply_patch_rejects_empty_patch(git_ws: Workspace):
    result = apply_patch(git_ws, "   \n")
    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_apply_patch_reports_a_timeout_rather_than_hanging(
    git_ws: Workspace, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(diff_tools, "run_command", _timed_out_command)
    result = apply_patch(git_ws, "--- a/x\n+++ b/x\n")
    assert result["success"] is False
    assert "timed out" in result["error"]


@pytest.mark.parametrize("call", [get_diff, get_status])
def test_git_timeout_raises_runtime_error(
    git_ws: Workspace, monkeypatch: pytest.MonkeyPatch, call
):
    monkeypatch.setattr(diff_tools, "run_command", _timed_out_command)
    with pytest.raises(RuntimeError, match="timed out"):
        call(git_ws)


def test_apply_patch_leaves_no_temp_file_in_the_workspace(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    patch = get_diff(git_ws)
    git(git_ws.root, "checkout", "--", "tracked.txt")

    apply_patch(git_ws, patch)
    assert get_status(git_ws) == ["tracked.txt"]
