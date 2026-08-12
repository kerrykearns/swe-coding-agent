"""Tests for the local git plumbing: clone, branch, commit.

Two different kinds of test live here, on purpose:

* **Clone** is exercised against a fake ``run_command``. A clone is the one
  operation that must talk to github.com, and what is worth asserting about it
  is the command that gets built — that the token is in the clone URL, that it is
  gone from the remote afterwards, and that it never appears in an error message.
  A real clone would verify none of that any better and would need the network.

* **Branch and commit** are exercised against real git repositories in
  ``tmp_path``, the same way M1's ``diff_tools`` tests are. These functions'
  interesting behaviour *is* what git does with them — "nothing to commit", a
  message containing a quote — and a fake git would only be a restatement of
  what this code already assumes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from agent.core import repo_manager
from agent.core.github_client import RepoNotFoundError
from agent.core.repo_manager import (
    BranchError,
    CloneError,
    RepoManagerError,
    branch_name_for_issue,
    clone_repo,
    commit_changes,
    create_branch,
    sanitize_branch_name,
)
from agent.tools import Workspace, read_file, write_file

from .conftest import git

#: A token-shaped string. Never a real token, and never sent anywhere.
FAKE_TOKEN = "github_pat_fake_000000000000000000"


# --------------------------------------------------------------------------
# Branch name sanitization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title, expected",
    [
        ("add() gives the wrong result", "add-gives-the-wrong-result"),
        ("Simple Title", "simple-title"),
        # Punctuation of every kind collapses to single hyphens.
        (
            "Fix: `divide()` raises ZeroDivisionError!!!",
            "fix-divide-raises-zerodivisionerror",
        ),
        ("  leading and trailing whitespace  ", "leading-and-trailing-whitespace"),
        ("multiple     spaces", "multiple-spaces"),
        ("already-hyphenated-title", "already-hyphenated-title"),
        # Characters git forbids in a ref, all in one title.
        ("bug~with^weird:chars?and*more[here]", "bug-with-weird-chars-and-more-here"),
        # A path-shaped title must not create nested refs.
        ("feature/sub/thing", "feature-sub-thing"),
        # ".." and a trailing ".lock" are both illegal in a ref.
        ("version..bump", "version-bump"),
        ("branch.lock", "branch-lock"),
        # Digits and case.
        ("Issue 42: HTTP 500 on POST", "issue-42-http-500-on-post"),
        # Nothing usable at all.
        ("!!!", "issue"),
        ("", "issue"),
        ("   ", "issue"),
        ("---", "issue"),
        # Non-ASCII collapses rather than reaching git as bytes some tools mangle.
        ("café crash", "caf-crash"),
        ("日本語のタイトル", "issue"),
        ("emoji 🔥 in title", "emoji-in-title"),
    ],
)
def test_sanitize_branch_name_handles_messy_titles(title, expected):
    assert sanitize_branch_name(title) == expected


def test_sanitize_branch_name_truncates_at_a_word_boundary():
    slug = sanitize_branch_name(
        "the calculator module returns incorrect values for every operation"
    )
    assert len(slug) <= repo_manager.MAX_SLUG_LENGTH
    # Cut back to a hyphen rather than left mid-word.
    assert slug == "the-calculator-module-returns-incorrect"


def test_sanitize_branch_name_keeps_a_mid_word_cut_over_losing_half_the_budget():
    """One very long word is more informative truncated than thrown away."""
    slug = sanitize_branch_name("short " + "a" * 60, max_length=20)
    assert len(slug) == 20
    assert slug.startswith("short-aaa")


def test_sanitize_branch_name_respects_a_custom_max_length():
    slug = sanitize_branch_name("add() gives the wrong result", max_length=10)
    assert len(slug) <= 10
    assert slug == "add-gives"


def test_sanitize_branch_name_never_ends_with_the_truncation_hyphen():
    """Cutting at the character before a hyphen must not leave a trailing one."""
    slug = sanitize_branch_name("abcde fghij", max_length=6)
    assert not slug.endswith("-")
    assert not slug.startswith("-")


def test_sanitize_branch_name_rejects_a_useless_max_length():
    with pytest.raises(ValueError):
        sanitize_branch_name("anything", max_length=0)


MESSY_TITLES = [
    "add() gives the wrong result",
    "Fix: `divide()` raises ZeroDivisionError!!!",
    "bug~with^weird:chars?and*more[here]",
    "feature/sub/thing",
    "version..bump",
    "branch.lock",
    "日本語のタイトル",
    "emoji 🔥 in title",
    "---",
    "",
    "the calculator module returns incorrect values for every operation",
    'quote " and backtick ` and dollar $HOME',
    "back\\slash",
    "@{unusual}",
]


@pytest.mark.parametrize("title", MESSY_TITLES)
def test_branch_names_from_messy_titles_are_legal_refs(title):
    """Verified by git itself, not by our own reading of the ref rules.

    ``git check-ref-format --branch`` is the authority on what git will accept,
    and it knows the rules this sanitizer only approximates (``..``, trailing
    ``.lock``, ``@{``, leading ``-``, control characters).
    """
    name = branch_name_for_issue(1, title)
    completed = subprocess.run(
        ["git", "check-ref-format", "--branch", name],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"git rejected {name!r} (from title {title!r}): "
        f"{completed.stderr.strip() or completed.stdout.strip()}"
    )


# --------------------------------------------------------------------------
# Branch names for issues
# --------------------------------------------------------------------------


def test_branch_name_for_issue_without_a_title():
    assert branch_name_for_issue(7) == "agent-fix/issue-7"


def test_branch_name_for_issue_with_a_title():
    assert (
        branch_name_for_issue(7, "add() gives the wrong result")
        == "agent-fix/issue-7-add-gives-the-wrong-result"
    )


def test_branch_name_for_issue_ignores_a_blank_title():
    """A blank title must not leave a dangling hyphen on the branch name."""
    assert branch_name_for_issue(7, "   ") == "agent-fix/issue-7"
    assert branch_name_for_issue(7, None) == "agent-fix/issue-7"


def test_branch_name_for_issue_falls_back_when_a_title_sanitizes_to_nothing():
    assert branch_name_for_issue(7, "!!!") == "agent-fix/issue-7-issue"


def test_branch_name_for_issue_accepts_a_custom_prefix():
    assert branch_name_for_issue(7, prefix="bot") == "bot/issue-7"


def test_branch_name_leads_with_the_issue_number():
    """The number must precede the slug, so branches sort and grep by issue."""
    name = branch_name_for_issue(12, "some title")
    assert name.startswith("agent-fix/issue-12-")


# --------------------------------------------------------------------------
# Clone — against a fake git, so nothing touches the network
# --------------------------------------------------------------------------


class _FakeGit:
    """Stands in for ``run_command``, recording every command it is handed.

    Args:
        create_at: On a successful ``git clone``, create this directory with a
            ``.git`` inside, as a real clone would.
        exit_code / stdout / stderr / timed_out: What the *clone* reports. Every
            other command always succeeds, so a test about clone failure does
            not also have to describe the follow-up calls.
    """

    def __init__(
        self,
        create_at: Path | None = None,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
    ) -> None:
        self.create_at = create_at
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.calls: list[dict] = []

    def __call__(self, ws: Workspace, command: str, timeout: int = 30) -> dict:
        self.calls.append(
            {"root": Path(ws.root), "command": command, "timeout": timeout}
        )
        if not command.startswith("git clone"):
            return {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False}

        if self.exit_code == 0 and not self.timed_out and self.create_at is not None:
            (self.create_at / ".git").mkdir(parents=True, exist_ok=True)
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }

    @property
    def commands(self) -> list[str]:
        return [call["command"] for call in self.calls]

    def command_starting(self, prefix: str) -> str:
        matches = [command for command in self.commands if command.startswith(prefix)]
        assert matches, f"no command started with {prefix!r}; got {self.commands}"
        return matches[0]


def _always(
    exit_code: int = 1, stdout: str = "", stderr: str = "", timed_out: bool = False
):
    """A ``run_command`` replacement that answers every command the same way.

    ``_FakeGit`` deliberately only fails the clone, so the branch and commit
    failure paths need something blunter.
    """

    def run(ws: Workspace, command: str, timeout: int = 30) -> dict:
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
        }

    return run


@pytest.fixture
def fake_git(monkeypatch, tmp_path: Path) -> _FakeGit:
    """Replace repo_manager's ``run_command`` with a recorder, and set a token."""
    fake = _FakeGit(create_at=tmp_path / "clones" / "repo")
    monkeypatch.setattr(repo_manager, "run_command", fake)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    return fake


def _destination(tmp_path: Path) -> Path:
    """The path ``fake_git`` is primed to create."""
    return tmp_path / "clones" / "repo"


def test_clone_repo_returns_a_workspace_rooted_at_the_clone(fake_git, tmp_path):
    workspace = clone_repo("owner/repo", _destination(tmp_path))
    assert isinstance(workspace, Workspace)
    assert workspace.root == _destination(tmp_path).resolve()


def test_clone_repo_runs_git_from_the_destinations_parent(fake_git, tmp_path):
    """The destination does not exist yet, so it cannot be its own cwd."""
    clone_repo("owner/repo", _destination(tmp_path))
    first = fake_git.calls[0]
    assert first["root"] == (tmp_path / "clones").resolve()
    assert first["command"].startswith("git clone")


def test_clone_repo_creates_the_parent_directory(fake_git, tmp_path):
    destination = tmp_path / "deeply" / "nested" / "repo"
    fake_git.create_at = destination
    clone_repo("owner/repo", destination)
    assert destination.exists()


def test_clone_repo_quotes_the_destination_for_paths_with_spaces(fake_git, tmp_path):
    destination = tmp_path / "with space" / "repo dir"
    fake_git.create_at = destination
    clone_repo("owner/repo", destination)
    assert '"repo dir"' in fake_git.command_starting("git clone")


def test_clone_repo_authenticates_with_the_token_from_the_environment(fake_git, tmp_path):
    clone_repo("owner/repo", _destination(tmp_path))
    command = fake_git.command_starting("git clone")
    assert f"https://x-access-token:{FAKE_TOKEN}@github.com/owner/repo.git" in command


def test_clone_repo_accepts_an_explicit_token_over_the_environment(fake_git, tmp_path):
    clone_repo("owner/repo", _destination(tmp_path), token="explicit-token")
    command = fake_git.command_starting("git clone")
    assert "explicit-token" in command
    assert FAKE_TOKEN not in command


def test_clone_repo_scrubs_the_token_from_the_remote_after_cloning(fake_git, tmp_path):
    """Cloning with credentials in the URL makes git persist them in .git/config."""
    clone_repo("owner/repo", _destination(tmp_path))
    reset = fake_git.command_starting("git remote set-url")
    assert reset == 'git remote set-url origin "https://github.com/owner/repo.git"'
    assert FAKE_TOKEN not in reset


def test_clone_repo_scrubs_from_inside_the_clone(fake_git, tmp_path):
    """The set-url has to run in the new checkout, not in its parent."""
    clone_repo("owner/repo", _destination(tmp_path))
    reset = next(
        call for call in fake_git.calls if call["command"].startswith("git remote")
    )
    assert reset["root"] == _destination(tmp_path).resolve()


def test_clone_repo_can_clone_anonymously(fake_git, tmp_path):
    clone_repo("owner/repo", _destination(tmp_path), use_token=False)
    assert fake_git.command_starting("git clone") == (
        'git clone "https://github.com/owner/repo.git" "repo"'
    )
    # Nothing to scrub, so nothing is scrubbed.
    assert not [c for c in fake_git.commands if c.startswith("git remote")]


def test_clone_repo_clones_anonymously_when_no_token_is_configured(
    monkeypatch, tmp_path
):
    fake = _FakeGit(create_at=_destination(tmp_path))
    monkeypatch.setattr(repo_manager, "run_command", fake)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    clone_repo("owner/repo", _destination(tmp_path))
    assert "x-access-token" not in fake.command_starting("git clone")


def test_clone_repo_gives_the_clone_a_longer_timeout_than_local_git(fake_git, tmp_path):
    """A network-bound clone killed at the 30-second default leaves a half-repo."""
    clone_repo("owner/repo", _destination(tmp_path))
    assert fake_git.calls[0]["timeout"] == repo_manager.CLONE_TIMEOUT_SECONDS
    assert repo_manager.CLONE_TIMEOUT_SECONDS > 30


def test_clone_repo_redacts_the_token_from_a_failure_message(monkeypatch, tmp_path):
    """git echoes the remote URL in its errors, token and all."""
    fake = _FakeGit(
        exit_code=128,
        stderr=(
            f"fatal: unable to access 'https://x-access-token:{FAKE_TOKEN}"
            "@github.com/owner/repo.git/': failed to connect"
        ),
    )
    monkeypatch.setattr(repo_manager, "run_command", fake)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)

    with pytest.raises(CloneError) as caught:
        clone_repo("owner/repo", _destination(tmp_path))

    message = str(caught.value)
    assert FAKE_TOKEN not in message
    assert "***" in message
    assert "failed to connect" in message
    assert "128" in message


def test_clone_repo_reports_a_timeout(monkeypatch, tmp_path):
    fake = _FakeGit(timed_out=True, exit_code=-1)
    monkeypatch.setattr(repo_manager, "run_command", fake)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)

    with pytest.raises(CloneError) as caught:
        clone_repo("owner/repo", _destination(tmp_path), timeout=7)
    assert "timed out after 7 seconds" in str(caught.value)


def test_clone_repo_reports_a_failure_with_no_git_output(monkeypatch, tmp_path):
    fake = _FakeGit(exit_code=1)
    monkeypatch.setattr(repo_manager, "run_command", fake)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    with pytest.raises(CloneError) as caught:
        clone_repo("owner/repo", _destination(tmp_path))
    assert "no output from git" in str(caught.value)


def test_clone_repo_fails_when_git_succeeds_but_leaves_no_repository(
    monkeypatch, tmp_path
):
    """Exit code 0 is not proof of a usable checkout."""
    destination = _destination(tmp_path)
    destination.mkdir(parents=True)
    fake = _FakeGit(create_at=None)  # succeeds, creates nothing
    monkeypatch.setattr(repo_manager, "run_command", fake)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)

    with pytest.raises(CloneError) as caught:
        clone_repo("owner/repo", destination)
    assert ".git" in str(caught.value)


def test_clone_repo_accepts_an_existing_empty_directory(fake_git, tmp_path):
    destination = _destination(tmp_path)
    destination.mkdir(parents=True)
    assert clone_repo("owner/repo", destination).root == destination.resolve()


def test_clone_repo_refuses_a_non_empty_destination(fake_git, tmp_path):
    destination = _destination(tmp_path)
    destination.mkdir(parents=True)
    (destination / "somebody-elses-work.txt").write_text("do not clobber me")

    with pytest.raises(CloneError) as caught:
        clone_repo("owner/repo", destination)
    assert "not empty" in str(caught.value)
    assert fake_git.calls == []  # nothing ran
    assert (destination / "somebody-elses-work.txt").exists()


def test_clone_repo_refuses_a_destination_that_is_a_file(fake_git, tmp_path):
    destination = tmp_path / "clones" / "repo"
    destination.parent.mkdir(parents=True)
    destination.write_text("not a directory")

    with pytest.raises(CloneError) as caught:
        clone_repo("owner/repo", destination)
    assert "not a directory" in str(caught.value)


def test_clone_repo_rejects_a_malformed_repository_name(fake_git, tmp_path):
    """Validated before a URL is built around it, so no git runs."""
    with pytest.raises(RepoNotFoundError):
        clone_repo("not-a-repo-name", _destination(tmp_path))
    assert fake_git.calls == []


def test_clone_repo_normalises_the_repository_name(fake_git, tmp_path):
    clone_repo("  owner/repo  ", _destination(tmp_path))
    assert "github.com/owner/repo.git" in fake_git.command_starting("git clone")


# --------------------------------------------------------------------------
# Keeping our own tooling's droppings out of the agent's commit
# --------------------------------------------------------------------------


def test_cloning_excludes_the_artifacts_of_our_own_test_runs(fake_git, tmp_path):
    workspace = clone_repo("owner/repo", _destination(tmp_path))
    written = (workspace.root / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    for pattern in repo_manager.AGENT_ARTIFACT_EXCLUDES:
        assert pattern in written.splitlines()


def test_the_excludes_are_local_to_the_checkout(git_ws: Workspace):
    """Never the repository's own .gitignore: that would be an unrequested
    change in the very diff a human is about to review."""
    before = (git_ws.root / ".gitignore").exists()
    repo_manager.exclude_agent_artifacts(git_ws)

    assert (git_ws.root / ".git" / "info" / "exclude").exists()
    assert (git_ws.root / ".gitignore").exists() is before
    assert git(git_ws.root, "status", "--porcelain").stdout.strip() == ""


def test_pytest_caches_are_not_committed(git_ws: Workspace):
    """The bug this exists for, reproduced: observed in the first live M4 run.

    ``run_tests`` makes pytest write bytecode caches into the checkout, and
    ``commit_changes`` stages with ``git add -A``, so three ``.pyc`` files landed
    in a commit whose message was about a one-line fix.
    """
    repo_manager.exclude_agent_artifacts(git_ws)
    write_file(git_ws, "tracked.txt", "two\n")
    cache = git_ws.root / "__pycache__"
    cache.mkdir()
    (cache / "calc.cpython-311.pyc").write_bytes(b"\x00compiled")
    (git_ws.root / ".pytest_cache").mkdir()
    (git_ws.root / ".pytest_cache" / "CACHEDIR.TAG").write_text("x")

    result = commit_changes(git_ws, "Fix: tracked.txt (closes #1)")

    assert result["success"] is True
    committed = git(git_ws.root, "show", "--name-only", "--pretty=", "HEAD").stdout
    assert "tracked.txt" in committed
    assert ".pyc" not in committed
    assert "__pycache__" not in committed
    assert ".pytest_cache" not in committed


def test_the_excludes_are_written_only_once(git_ws: Workspace):
    """A second run over the same checkout must not keep appending."""
    assert repo_manager.exclude_agent_artifacts(git_ws) is True
    once = (git_ws.root / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert repo_manager.exclude_agent_artifacts(git_ws) is True
    assert (git_ws.root / ".git" / "info" / "exclude").read_text(encoding="utf-8") == once


def test_the_excludes_do_not_disturb_an_existing_exclude_file(git_ws: Workspace):
    exclude_file = git_ws.root / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    exclude_file.write_text("# somebody else's rule\nsecrets.txt", encoding="utf-8")

    repo_manager.exclude_agent_artifacts(git_ws)

    written = exclude_file.read_text(encoding="utf-8")
    assert "secrets.txt" in written.splitlines()
    assert "__pycache__/" in written.splitlines()


def test_the_exclude_file_is_created_if_the_info_directory_is_missing(ws: Workspace):
    assert repo_manager.exclude_agent_artifacts(ws) is True
    assert (ws.root / ".git" / "info" / "exclude").exists()


def test_an_unwritable_exclude_file_does_not_lose_a_good_clone(ws: Workspace):
    """Best-effort housekeeping: a ``.git`` *file*, as a worktree has, is not a
    directory to write into, and that must not fail the clone."""
    (ws.root / ".git").write_text("gitdir: ../elsewhere/.git", encoding="utf-8")
    assert repo_manager.exclude_agent_artifacts(ws) is False


def test_clone_errors_are_catchable_as_repo_manager_errors():
    for exception in (CloneError, BranchError):
        assert issubclass(exception, RepoManagerError)
    assert issubclass(RepoManagerError, RuntimeError)


# --------------------------------------------------------------------------
# create_branch — against real git repositories
# --------------------------------------------------------------------------


def _current_branch(ws: Workspace) -> str:
    return git(ws.root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def test_create_branch_creates_and_checks_out_the_branch(git_ws: Workspace):
    create_branch(git_ws, "agent-fix/issue-1")
    assert _current_branch(git_ws) == "agent-fix/issue-1"


def test_create_branch_keeps_the_working_tree(git_ws: Workspace):
    """Branching must not disturb the files the agent is about to edit."""
    create_branch(git_ws, "agent-fix/issue-1")
    assert read_file(git_ws, "tracked.txt") == "one\n"


def test_create_branch_trims_surrounding_whitespace(git_ws: Workspace):
    create_branch(git_ws, "  agent-fix/issue-2  ")
    assert _current_branch(git_ws) == "agent-fix/issue-2"


def test_create_branch_rejects_an_existing_branch(git_ws: Workspace):
    create_branch(git_ws, "agent-fix/issue-1")
    git(git_ws.root, "checkout", "-q", "main")

    with pytest.raises(BranchError) as caught:
        create_branch(git_ws, "agent-fix/issue-1")
    assert "already exists" in str(caught.value)


@pytest.mark.parametrize("name", ["", "   "])
def test_create_branch_rejects_an_empty_name(git_ws: Workspace, name):
    with pytest.raises(BranchError) as caught:
        create_branch(git_ws, name)
    assert "empty" in str(caught.value)


def test_create_branch_lets_git_reject_an_illegal_ref(git_ws: Workspace):
    """git is the authority on ref names, and its message is passed through."""
    with pytest.raises(BranchError) as caught:
        create_branch(git_ws, "bad..name")
    assert "bad..name" in str(caught.value)


def test_create_branch_fails_outside_a_git_repository(ws: Workspace):
    with pytest.raises(BranchError) as caught:
        create_branch(ws, "agent-fix/issue-1")
    assert "Could not create branch" in str(caught.value)


def test_create_branch_reports_a_timeout(monkeypatch, git_ws: Workspace):
    monkeypatch.setattr(
        repo_manager, "run_command", _always(exit_code=-1, timed_out=True)
    )
    with pytest.raises(BranchError) as caught:
        create_branch(git_ws, "agent-fix/issue-1")
    assert "timed out" in str(caught.value)


# --------------------------------------------------------------------------
# commit_changes — against real git repositories
# --------------------------------------------------------------------------


def _subject(ws: Workspace) -> str:
    return git(ws.root, "log", "-1", "--pretty=%s").stdout.strip()


def _full_message(ws: Workspace) -> str:
    return git(ws.root, "log", "-1", "--pretty=%B").stdout.strip()


def test_commit_changes_commits_a_modification(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")

    result = commit_changes(git_ws, "Fix: tracked.txt (closes #1)")

    assert result["success"] is True
    assert result["reason"] is None
    assert re.fullmatch(r"[0-9a-f]{40}", result["commit_hash"] or "")
    assert _subject(git_ws) == "Fix: tracked.txt (closes #1)"


def test_the_reported_hash_is_the_new_head(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    result = commit_changes(git_ws, "a message")
    assert result["commit_hash"] == git(git_ws.root, "rev-parse", "HEAD").stdout.strip()


def test_commit_changes_leaves_a_clean_working_tree(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    commit_changes(git_ws, "a message")
    assert git(git_ws.root, "status", "--porcelain").stdout.strip() == ""


def test_commit_changes_stages_new_files(git_ws: Workspace):
    """``git add -A``, so a file the agent created is included."""
    write_file(git_ws, "brand_new.py", "print('hello')\n")

    result = commit_changes(git_ws, "Add brand_new.py")

    assert result["success"] is True
    committed = git(git_ws.root, "show", "--name-only", "--pretty=", "HEAD").stdout
    assert "brand_new.py" in committed


def test_commit_changes_stages_deletions(git_ws: Workspace):
    (git_ws.root / "tracked.txt").unlink()
    result = commit_changes(git_ws, "Remove tracked.txt")
    assert result["success"] is True
    assert "tracked.txt" in git(
        git_ws.root, "show", "--name-only", "--pretty=", "HEAD"
    ).stdout


def test_commit_changes_reports_nothing_to_commit_as_success_false(git_ws: Workspace):
    """An agent that changed nothing is an ordinary outcome, not a crash."""
    result = commit_changes(git_ws, "Fix: nothing at all")

    assert result["success"] is False
    assert result["commit_hash"] is None
    assert "Nothing to commit" in result["reason"]
    assert "not an error" in result["reason"]


def test_nothing_to_commit_does_not_add_a_commit(git_ws: Workspace):
    before = git(git_ws.root, "rev-parse", "HEAD").stdout.strip()
    commit_changes(git_ws, "Fix: nothing at all")
    assert git(git_ws.root, "rev-parse", "HEAD").stdout.strip() == before


def test_commit_changes_does_not_let_a_message_reach_the_shell(git_ws: Workspace):
    """The message comes from an issue title, i.e. from a stranger.

    ``git commit -m "<title>"`` would interpolate it into a shell command line,
    where a quote plus a redirect in somebody's issue title decides what runs.
    ``-F <file>`` cannot be talked out of treating it as text.
    """
    write_file(git_ws, "tracked.txt", "two\n")
    hostile = 'Fix: "add()" & echo pwned > pwned.txt (closes #1)'

    result = commit_changes(git_ws, hostile)

    assert result["success"] is True
    assert _subject(git_ws) == hostile
    assert not (git_ws.root / "pwned.txt").exists()


def test_commit_changes_preserves_a_multi_line_message(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    message = "Fix: add() (closes #1)\n\nThe operator was inverted."

    commit_changes(git_ws, message)

    assert _subject(git_ws) == "Fix: add() (closes #1)"
    assert _full_message(git_ws) == message


def test_commit_changes_rejects_a_blank_message(git_ws: Workspace):
    """A caller that produced an empty message has a bug, not a repo state."""
    write_file(git_ws, "tracked.txt", "two\n")
    for blank in ("", "   ", "\n"):
        with pytest.raises(ValueError):
            commit_changes(git_ws, blank)


def test_commit_changes_reports_a_non_repository_instead_of_raising(ws: Workspace):
    (ws.root / "a_file.txt").write_text("content")

    result = commit_changes(ws, "a message")

    assert result["success"] is False
    assert result["commit_hash"] is None
    assert "git add failed" in result["reason"]
    assert result["output"]


def test_commit_changes_returns_gits_transcript(git_ws: Workspace):
    write_file(git_ws, "tracked.txt", "two\n")
    result = commit_changes(git_ws, "Fix: tracked.txt")
    assert "$ git add -A" in result["output"]
    assert "$ git commit" in result["output"]


def test_commit_changes_reports_a_failing_commit(monkeypatch, git_ws: Workspace):
    """E.g. a machine with no git identity configured, or a signing key missing."""
    write_file(git_ws, "tracked.txt", "two\n")
    real = repo_manager.run_command

    def refuse_the_commit(workspace, command, timeout=30):
        if command.startswith("git commit"):
            return {
                "stdout": "",
                "stderr": "Author identity unknown",
                "exit_code": 128,
                "timed_out": False,
            }
        return real(workspace, command, timeout=timeout)

    monkeypatch.setattr(repo_manager, "run_command", refuse_the_commit)

    result = commit_changes(git_ws, "a message")
    assert result["success"] is False
    assert "git commit failed" in result["reason"]
    assert "Author identity unknown" in result["reason"]


def test_commit_changes_reports_a_failing_add(monkeypatch, git_ws: Workspace):
    monkeypatch.setattr(
        repo_manager, "run_command", _always(exit_code=1, stderr="permission denied")
    )
    result = commit_changes(git_ws, "a message")
    assert result["success"] is False
    assert "permission denied" in result["reason"]


def test_commit_changes_tolerates_an_unreadable_head(monkeypatch, git_ws: Workspace):
    """A commit that landed but whose hash could not be read is still a success."""
    write_file(git_ws, "tracked.txt", "two\n")
    real = repo_manager.run_command

    def hide_the_revision(workspace, command, timeout=30):
        if command.startswith("git rev-parse"):
            return {"stdout": "", "stderr": "nope", "exit_code": 1, "timed_out": False}
        return real(workspace, command, timeout=timeout)

    monkeypatch.setattr(repo_manager, "run_command", hide_the_revision)

    result = commit_changes(git_ws, "a message")
    assert result["success"] is True
    assert result["commit_hash"] is None


# --------------------------------------------------------------------------
# M4 scoping: this module does not push
# --------------------------------------------------------------------------


def test_no_git_push_is_ever_issued(git_ws: Workspace, monkeypatch, tmp_path):
    """M4's scoping constraint, asserted at the level of the commands run.

    A local commit is undone with ``git reset``; a pushed branch is public.
    Pushing waits for M6's confirmation gate, so every command this module
    issues is recorded here and checked. Asserted behaviourally rather than by
    grepping the source, which would also match the docstrings that explain the
    constraint.
    """
    recorded: list[str] = []
    real = repo_manager.run_command

    def record(workspace, command, timeout=30):
        recorded.append(command)
        return real(workspace, command, timeout=timeout)

    monkeypatch.setattr(repo_manager, "run_command", record)

    create_branch(git_ws, "agent-fix/issue-1")
    write_file(git_ws, "tracked.txt", "two\n")
    commit_changes(git_ws, "Fix: tracked.txt (closes #1)")

    assert recorded, "no git commands were recorded, so this proves nothing"
    assert not any("push" in command for command in recorded), recorded
