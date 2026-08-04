"""Tests for read/write/list/search, including that they inherit containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import (
    Workspace,
    WorkspaceViolation,
    list_files,
    read_file,
    search_text,
    write_file,
)


@pytest.fixture
def populated(ws: Workspace) -> Workspace:
    """A small tree: two modules, a test, a doc, and some noise to be skipped."""
    write_file(ws, "pkg/__init__.py", "")
    write_file(ws, "pkg/calc.py", "def add(a, b):\n    return a - b\n")
    write_file(ws, "pkg/util.py", "def helper():\n    return 42\n")
    write_file(ws, "tests/test_calc.py", "def test_add():\n    assert add(2, 3) == 5\n")
    write_file(ws, "README.md", "# demo\n\ndef add is broken\n")
    write_file(ws, "__pycache__/calc.cpython-311.pyc", "junk")
    write_file(ws, ".git/config", "[core]\n")
    return ws


# --------------------------------------------------------------------------
# read_file / write_file
# --------------------------------------------------------------------------


def test_write_then_read_round_trip(ws: Workspace):
    write_file(ws, "hello.txt", "hello world\n")
    assert read_file(ws, "hello.txt") == "hello world\n"


def test_write_creates_parent_directories(ws: Workspace):
    write_file(ws, "a/b/c/deep.txt", "deep")
    assert (ws.root / "a" / "b" / "c" / "deep.txt").is_file()
    assert read_file(ws, "a/b/c/deep.txt") == "deep"


def test_write_overwrites_existing_file(ws: Workspace):
    write_file(ws, "f.txt", "first")
    write_file(ws, "f.txt", "second")
    assert read_file(ws, "f.txt") == "second"


def test_write_preserves_newlines_exactly(ws: Workspace):
    """No platform newline translation, so content is diff-stable on Windows."""
    content = "line one\nline two\n"
    write_file(ws, "lf.txt", content)
    assert (ws.root / "lf.txt").read_bytes() == b"line one\nline two\n"
    assert read_file(ws, "lf.txt") == content


def test_write_empty_file(ws: Workspace):
    write_file(ws, "empty.txt", "")
    assert read_file(ws, "empty.txt") == ""


def test_write_unicode(ws: Workspace):
    write_file(ws, "u.txt", "héllo — ✅\n")
    assert read_file(ws, "u.txt") == "héllo — ✅\n"


def test_read_missing_file_raises_helpful_error(ws: Workspace):
    with pytest.raises(FileNotFoundError) as excinfo:
        read_file(ws, "nope.py")
    message = str(excinfo.value)
    assert "nope.py" in message
    assert str(ws.root) in message
    assert "list_files" in message


def test_read_directory_raises(ws: Workspace):
    write_file(ws, "pkg/mod.py", "x = 1\n")
    with pytest.raises(IsADirectoryError, match="pkg"):
        read_file(ws, "pkg")


def test_read_binary_file_raises_value_error(ws: Workspace):
    ws.resolve("blob.bin").write_bytes(b"\x00\x01\xff\xfe binary")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_file(ws, "blob.bin")


def test_write_over_directory_raises(ws: Workspace):
    write_file(ws, "pkg/mod.py", "x = 1\n")
    with pytest.raises(IsADirectoryError):
        write_file(ws, "pkg", "clobber")


# --------------------------------------------------------------------------
# list_files
# --------------------------------------------------------------------------


def test_list_files_is_recursive_and_sorted(populated: Workspace):
    listing = list_files(populated)
    assert listing == sorted(listing)
    assert "pkg/calc.py" in listing
    assert "tests/test_calc.py" in listing
    assert "README.md" in listing


def test_list_files_returns_posix_relative_paths(populated: Workspace):
    for entry in list_files(populated):
        assert not Path(entry).is_absolute()
        assert "\\" not in entry


def test_list_files_skips_generated_directories(populated: Workspace):
    listing = list_files(populated)
    assert not any(entry.startswith("__pycache__/") for entry in listing)
    assert not any(entry.startswith(".git/") for entry in listing)


def test_list_files_honours_pattern(populated: Workspace):
    listing = list_files(populated, pattern="*.py")
    assert "pkg/calc.py" in listing
    assert "README.md" not in listing
    assert all(entry.endswith(".py") for entry in listing)


def test_list_files_scoped_to_subdir(populated: Workspace):
    listing = list_files(populated, subdir="pkg")
    assert set(listing) == {"pkg/__init__.py", "pkg/calc.py", "pkg/util.py"}


def test_list_files_excludes_directories(populated: Workspace):
    listing = list_files(populated)
    assert "pkg" not in listing
    assert "tests" not in listing


def test_list_files_empty_workspace(ws: Workspace):
    assert list_files(ws) == []


def test_list_files_missing_subdir_raises(ws: Workspace):
    with pytest.raises(FileNotFoundError, match="ghost"):
        list_files(ws, subdir="ghost")


def test_list_files_on_a_file_raises(ws: Workspace):
    write_file(ws, "f.txt", "x")
    with pytest.raises(NotADirectoryError):
        list_files(ws, subdir="f.txt")


# --------------------------------------------------------------------------
# search_text
# --------------------------------------------------------------------------


def test_search_text_reports_file_line_and_content(populated: Workspace):
    hits = search_text(populated, "return a - b")
    assert len(hits) == 1
    assert hits[0] == {
        "file": "pkg/calc.py",
        "line_number": 2,
        "line_content": "    return a - b",
    }


def test_search_text_finds_multiple_files(populated: Workspace):
    files = {hit["file"] for hit in search_text(populated, "def add")}
    assert files == {"pkg/calc.py", "README.md"}


def test_search_text_no_matches_returns_empty_list(populated: Workspace):
    assert search_text(populated, "quantum entanglement") == []


def test_search_text_honours_pattern(populated: Workspace):
    files = {hit["file"] for hit in search_text(populated, "def add", pattern="*.md")}
    assert files == {"README.md"}


def test_search_text_scoped_to_subdir(populated: Workspace):
    files = {hit["file"] for hit in search_text(populated, "def ", subdir="tests")}
    assert files == {"tests/test_calc.py"}


def test_search_text_is_case_sensitive(populated: Workspace):
    assert search_text(populated, "DEF ADD") == []


def test_search_text_skips_binary_files(ws: Workspace):
    ws.resolve("blob.bin").write_bytes(b"\x00needle\xff")
    write_file(ws, "text.txt", "needle\n")
    hits = search_text(ws, "needle")
    assert [hit["file"] for hit in hits] == ["text.txt"]


def test_search_text_respects_max_results(ws: Workspace):
    write_file(ws, "many.txt", "needle\n" * 50)
    assert len(search_text(ws, "needle", max_results=5)) == 5


def test_search_text_rejects_empty_query(ws: Workspace):
    with pytest.raises(ValueError, match="non-empty"):
        search_text(ws, "")


# --------------------------------------------------------------------------
# containment is inherited by every file operation
# --------------------------------------------------------------------------


def test_read_file_blocks_traversal(ws: Workspace):
    with pytest.raises(WorkspaceViolation):
        read_file(ws, "../../etc/passwd")


def test_write_file_blocks_traversal(tmp_path: Path, ws: Workspace):
    with pytest.raises(WorkspaceViolation):
        write_file(ws, "../escaped.txt", "should never be written")
    assert not (tmp_path / "escaped.txt").exists()


def test_list_files_blocks_traversal(ws: Workspace):
    with pytest.raises(WorkspaceViolation):
        list_files(ws, subdir="..")


def test_search_text_blocks_traversal(ws: Workspace):
    with pytest.raises(WorkspaceViolation):
        search_text(ws, "secret", subdir="../..")
