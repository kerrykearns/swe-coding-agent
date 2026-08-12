"""Local git plumbing: clone a target repository, branch it, commit the agent's work.

Everything here shells out through :func:`agent.tools.shell_exec.run_command`,
which pins the child process's cwd to a :class:`~agent.tools.workspace.Workspace`
root — the same path every other git-touching tool in this project takes (see
:mod:`agent.tools.diff_tools`). No new subprocess handling is introduced, and
``get_status`` is reused rather than reimplemented as a second porcelain parser.

PUSHING (M6)
============
:func:`push_branch` exists so :mod:`agent.core.run_on_issue` can push a
branch — but it must never be called except from behind
:class:`agent.safety.confirmation_gate.ConfirmationGate`. Nothing in this
module calls it itself, and nothing here decides that a push should happen;
that decision, and the human confirmation behind it, belongs entirely to the
gate. A local commit is trivially undone with ``git reset``; a pushed branch
is public — see the safety policy's ``confirm_push_branch`` rule.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from ..tools import Workspace, get_status, run_command
from . import github_client

__all__ = [
    "AGENT_ARTIFACT_EXCLUDES",
    "BranchError",
    "CloneError",
    "RepoManagerError",
    "branch_name_for_issue",
    "clone_repo",
    "commit_changes",
    "create_branch",
    "exclude_agent_artifacts",
    "push_branch",
    "sanitize_branch_name",
]

#: How long a clone may take before it is killed. Generous compared with the
#: 30-second default in :func:`~agent.tools.shell_exec.run_command`, because the
#: work is network-bound and the failure mode of a too-short timeout is a
#: half-written directory.
CLONE_TIMEOUT_SECONDS = 300

#: Timeout for the local git commands (branch, add, commit), which touch only disk.
GIT_TIMEOUT_SECONDS = 60

#: Default cap on the slug taken from an issue title. Refs may be far longer, but
#: a branch name is something a human reads in a PR list and types at a prompt.
MAX_SLUG_LENGTH = 40

#: What a title that sanitizes down to nothing becomes, so the branch name is
#: still a legal ref rather than a trailing hyphen.
_SLUG_FALLBACK = "issue"

#: Substituted for the token wherever command output is reported, so a PAT never
#: reaches a log, a traceback, or a terminal.
_REDACTED = "***"

#: Droppings that *our own tooling* leaves inside the checkout. Every
#: ``run_tests`` call makes pytest write bytecode caches and a ``.pytest_cache``
#: directory, and ``commit_changes`` stages with ``git add -A``, so without this
#: the agent commits three ``.pyc`` files alongside its one-line fix — observed in
#: the first live M4 run. They are written to ``.git/info/exclude`` rather than to
#: the repository's own ``.gitignore``: this is our opinion about our tools, it
#: belongs to this checkout only, and editing a target repo's ``.gitignore`` would
#: be an unrequested change in the very diff a human is about to review.
AGENT_ARTIFACT_EXCLUDES = (
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".coverage",
)


class RepoManagerError(RuntimeError):
    """Base class for the git-level failures this module raises."""


class CloneError(RepoManagerError):
    """The clone did not produce a usable repository."""


class BranchError(RepoManagerError):
    """The branch could not be created."""


# --------------------------------------------------------------------------
# Branch names
# --------------------------------------------------------------------------


def sanitize_branch_name(text: str, max_length: int = MAX_SLUG_LENGTH) -> str:
    """Turn arbitrary text — an issue title — into a safe branch-name fragment.

    Every run of characters outside ``[a-z0-9]`` collapses to a single hyphen,
    which is a deliberately blunt rule: it is easier to reason about than an
    allowlist of "safe" punctuation, and it rules out every character git
    forbids in a ref (space, ``~``, ``^``, ``:``, ``?``, ``*``, ``[``, ``\\``,
    ``..``, a trailing ``.lock``) by construction rather than by enumeration.
    Non-ASCII text collapses too, so a title in another script yields the
    fallback rather than a ref that some tools will mangle.

    Truncation prefers a word boundary: the slug is cut to ``max_length`` and
    then, if that landed mid-word, walked back to the previous hyphen — unless
    doing so would throw away more than half the budget, in which case the
    mid-word cut is kept as the more informative of two poor options.

    Args:
        text: The raw title.
        max_length: Maximum length of the returned slug.

    Returns:
        A lowercase hyphen-separated slug, never empty, never starting or
        ending with a hyphen. Text with nothing usable in it returns
        ``"issue"``.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")

    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    if not slug:
        return _SLUG_FALLBACK

    if len(slug) > max_length:
        cut = slug[:max_length]
        if slug[max_length] != "-":  # the cut landed inside a word
            boundary = cut.rfind("-")
            if boundary >= max_length // 2:
                cut = cut[:boundary]
        slug = cut.strip("-") or _SLUG_FALLBACK

    return slug


def branch_name_for_issue(
    issue_number: int,
    title: Optional[str] = None,
    prefix: str = "agent-fix",
) -> str:
    """Build the branch name for work on one issue.

    Returns ``"{prefix}/issue-{number}"``, with a sanitized title slug appended
    when a ``title`` is given. The issue number leads the descriptive part so
    that the branch sorts and greps by issue, and so the name is unambiguous
    even when the slug is the fallback.

    Args:
        issue_number: The issue's number.
        title: Optional issue title, sanitized via :func:`sanitize_branch_name`.
        prefix: Namespace segment, kept out of the default branch's way.
    """
    name = f"{prefix}/issue-{issue_number}"
    if title and title.strip():
        return f"{name}-{sanitize_branch_name(title)}"
    return name


# --------------------------------------------------------------------------
# Clone
# --------------------------------------------------------------------------


def _clone_url(repo_full_name: str, token: Optional[str]) -> str:
    """Build the HTTPS clone URL, embedding ``token`` as basic-auth if given.

    ``x-access-token`` is the username GitHub documents for token-in-URL auth,
    and it works for both classic PATs and fine-grained ones.
    """
    if token:
        return f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
    return f"https://github.com/{repo_full_name}.git"


def _redact(text: str, token: Optional[str]) -> str:
    """Remove ``token`` from ``text`` so it cannot leak into output.

    Applied to every string this module reports — git echoes the remote URL in
    both its progress output and its error messages, so an un-redacted failure
    would print the PAT to the terminal and into any captured log.
    """
    if not token or not text:
        return text
    return text.replace(token, _REDACTED)


def exclude_agent_artifacts(ws: Workspace) -> bool:
    """Tell this checkout to ignore the files the agent's own tools generate.

    Writes :data:`AGENT_ARTIFACT_EXCLUDES` to ``.git/info/exclude``, which git
    reads exactly like a ``.gitignore`` but which lives inside ``.git`` and is
    therefore never committed and never shown to a reviewer. Already-tracked
    files are unaffected, so a repository that deliberately versions a ``.pyc``
    keeps it.

    Args:
        ws: Workspace pointing at a git repository.

    Returns:
        True if the excludes are in place. False if they could not be written —
        this is best-effort housekeeping, and a checkout that is otherwise good
        must not be thrown away over it.
    """
    exclude_file = ws.root / ".git" / "info" / "exclude"
    try:
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
        )
        missing = [
            pattern
            for pattern in AGENT_ARTIFACT_EXCLUDES
            if pattern not in existing.splitlines()
        ]
        if not missing:
            return True

        prefix = "" if not existing or existing.endswith("\n") else "\n"
        with exclude_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                f"{prefix}# Added by the agent: artifacts of its own test runs.\n"
            )
            handle.write("".join(f"{pattern}\n" for pattern in missing))
        return True
    except OSError:
        return False


def clone_repo(
    repo_full_name: str,
    dest_dir: str | Path,
    token: Optional[str] = None,
    use_token: bool = True,
    timeout: int = CLONE_TIMEOUT_SECONDS,
) -> Workspace:
    """Clone ``repo_full_name`` into ``dest_dir`` and return it as a Workspace.

    The clone runs with cwd set to ``dest_dir``'s *parent* (created if needed),
    which is what lets it go through :func:`~agent.tools.shell_exec.run_command`
    like every other git call in this project: the destination does not exist
    yet, so it cannot be its own workspace root.

    On success the checkout is prepared for agent work by
    :func:`exclude_agent_artifacts`, so that pytest's caches — which only exist
    because the agent runs the tests — stay out of its commit.

    Also on success the remote is rewritten to the token-free URL. Cloning with
    credentials in the URL makes git persist them in ``.git/config``, where they
    would sit in plaintext for the rest of the checkout's life; resetting the
    remote afterwards costs one local command and keeps the token in memory
    only. A later push therefore needs its credentials supplied again — which is
    correct, because pushing is M6's decision to make, not this function's.

    Args:
        repo_full_name: ``"owner/repo"``.
        dest_dir: Where the working tree goes. Must not already exist as a
            non-empty directory.
        token: Token to authenticate with. Defaults to ``GITHUB_TOKEN`` from the
            environment when ``use_token`` is set and one is configured.
        use_token: Set False to clone anonymously — useful for a public repo
            when no token is configured at all.
        timeout: Seconds before the clone is killed.

    Returns:
        A :class:`~agent.tools.workspace.Workspace` rooted at the new clone.

    Raises:
        CloneError: If the destination is unusable, git is missing, the clone
            times out, or git exits non-zero. Git's own message is included,
            with the token redacted.
    """
    repo_full_name = github_client.normalise_repo_name(repo_full_name)

    destination = Path(dest_dir).expanduser()
    if destination.exists():
        if not destination.is_dir():
            raise CloneError(
                f"Clone destination {destination} already exists and is not a directory."
            )
        if any(destination.iterdir()):
            raise CloneError(
                f"Clone destination {destination} already exists and is not empty. "
                "Refusing to clone into it — remove it, or pick another path."
            )

    if token is None and use_token and github_client.token_present():
        token = github_client.get_token()
    if not use_token:
        token = None

    try:
        parent = Workspace(destination.parent, create=True)
    except OSError as exc:
        raise CloneError(
            f"Could not create the parent directory for {destination}: {exc}"
        ) from exc

    url = _clone_url(repo_full_name, token)
    # The URL is quoted because it carries the token; the destination is quoted
    # because a workspace path on Windows very often contains spaces.
    command = f'git clone "{url}" "{destination.name}"'
    result = run_command(parent, command, timeout=timeout)

    if result["timed_out"]:
        raise CloneError(
            f"Cloning {repo_full_name} timed out after {timeout} seconds."
        )
    if result["exit_code"] != 0:
        detail = _redact((result["stderr"] or result["stdout"]).strip(), token)
        raise CloneError(
            f"Cloning {repo_full_name} failed with exit code "
            f"{result['exit_code']}: {detail or 'no output from git'}"
        )
    if not (destination / ".git").exists():
        raise CloneError(
            f"git reported success but {destination} does not contain a .git "
            "directory, so the clone is not usable."
        )

    workspace = Workspace(destination)
    exclude_agent_artifacts(workspace)

    if token:
        # Drop the credential from .git/config. Best-effort: the clone itself
        # succeeded, and failing the whole call here would discard a good
        # checkout over a remote URL nobody in M4 pushes to.
        run_command(
            workspace,
            f'git remote set-url origin "{_clone_url(repo_full_name, None)}"',
            timeout=GIT_TIMEOUT_SECONDS,
        )

    return workspace


# --------------------------------------------------------------------------
# Branch and commit
# --------------------------------------------------------------------------


def create_branch(ws: Workspace, branch_name: str) -> None:
    """Create ``branch_name`` and check it out, via ``git checkout -b``.

    Args:
        ws: Workspace pointing at a git repository.
        branch_name: The new branch's name. Already-sanitized names come from
            :func:`branch_name_for_issue`; anything else is passed to git, which
            is the authority on what a legal ref is.

    Raises:
        BranchError: If the name is empty, or git refuses — most often because
            a branch with that name already exists.
    """
    if not branch_name or not branch_name.strip():
        raise BranchError("Branch name is empty.")
    branch_name = branch_name.strip()

    result = run_command(
        ws, f'git checkout -b "{branch_name}"', timeout=GIT_TIMEOUT_SECONDS
    )
    if result["timed_out"]:
        raise BranchError(f"Creating branch {branch_name!r} timed out.")
    if result["exit_code"] != 0:
        detail = (result["stderr"] or result["stdout"]).strip()
        raise BranchError(
            f"Could not create branch {branch_name!r} (exit code "
            f"{result['exit_code']}): {detail or 'no output from git'}"
        )


def commit_changes(ws: Workspace, message: str) -> dict:
    """Stage everything and commit it.

    This function reports rather than raises: git-level failure is an outcome
    the caller has to describe to a user, not an exception it can do anything
    about. "Nothing to commit" is the case that matters most — an agent that
    finished without editing a file is a perfectly ordinary result, and it must
    not look like a crash.

    The message is passed via ``git commit -F <file>`` rather than ``-m``,
    because the message is built from an issue title and would otherwise be
    interpolated into a shell command line where a quote or a backtick in
    somebody's issue title decides what runs.

    Args:
        ws: Workspace pointing at a git repository.
        message: The commit message. Must not be blank.

    Returns:
        ``{"success": bool, "commit_hash": str | None, "output": str,
        "reason": str | None}``. ``reason`` is ``None`` on success and
        otherwise a one-line explanation written for a human; ``output`` is
        git's own combined output, kept whole for the trajectory record.

    Raises:
        ValueError: If ``message`` is blank — git rejects an empty commit
            message, and a caller that produced one has a bug rather than a
            repository in an interesting state.
    """
    if not message or not message.strip():
        raise ValueError("Commit message is empty; git would reject the commit.")

    transcript: list[str] = []

    staged = run_command(ws, "git add -A", timeout=GIT_TIMEOUT_SECONDS)
    transcript.append(_transcribe("git add -A", staged))
    if staged["timed_out"] or staged["exit_code"] != 0:
        return {
            "success": False,
            "commit_hash": None,
            "output": "\n".join(transcript),
            "reason": (
                "git add failed, so nothing was committed: "
                f"{(staged['stderr'] or staged['stdout']).strip() or 'no output'}"
            ),
        }

    try:
        changed = get_status(ws)
    except RuntimeError as exc:  # not a git repository, or git is unavailable
        return {
            "success": False,
            "commit_hash": None,
            "output": "\n".join(transcript + [str(exc)]),
            "reason": f"Could not read the repository's status: {exc}",
        }

    if not changed:
        return {
            "success": False,
            "commit_hash": None,
            "output": "\n".join(transcript),
            "reason": (
                "Nothing to commit — the working tree is clean, so no files were "
                "changed. This is not an error."
            ),
        }

    handle, message_path = tempfile.mkstemp(suffix=".txt", prefix="agent-commit-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as message_file:
            message_file.write(message.strip() + "\n")

        committed = run_command(
            ws, f'git commit -F "{message_path}"', timeout=GIT_TIMEOUT_SECONDS
        )
    finally:
        try:
            os.unlink(message_path)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass

    transcript.append(_transcribe("git commit", committed))
    if committed["timed_out"] or committed["exit_code"] != 0:
        detail = (committed["stderr"] or committed["stdout"]).strip()
        return {
            "success": False,
            "commit_hash": None,
            "output": "\n".join(transcript),
            "reason": f"git commit failed: {detail or 'no output from git'}",
        }

    revision = run_command(ws, "git rev-parse HEAD", timeout=GIT_TIMEOUT_SECONDS)
    commit_hash = revision["stdout"].strip() if revision["exit_code"] == 0 else None

    return {
        "success": True,
        "commit_hash": commit_hash,
        "output": "\n".join(transcript),
        "reason": None,
    }


def push_branch(
    ws: Workspace,
    branch_name: str,
    repo_full_name: str,
    token: Optional[str] = None,
    use_token: bool = True,
    timeout: int = GIT_TIMEOUT_SECONDS,
) -> dict:
    """Push ``branch_name`` to ``origin``, authenticating for this one call only.

    NOT SAFE TO CALL DIRECTLY FROM AGENT-DRIVEN CODE — see the module
    docstring. Every caller in this codebase must go through
    :class:`agent.safety.confirmation_gate.ConfirmationGate` first.

    Follows :func:`clone_repo`'s credential hygiene rather than
    ``origin``'s: the token is embedded only in the URL given to this one
    ``git push`` and is never written to ``.git/config`` — the stored
    remote stays exactly as ``clone_repo`` left it, token-free.

    Args:
        ws: Workspace pointing at the git repository to push from.
        branch_name: The local branch to push, pushed to a remote branch of
            the same name (``git push <url> "<branch>:<branch>"``).
        repo_full_name: ``"owner/repo"``, used only to rebuild the
            authenticated URL — the push does not touch the stored remote.
        token: Token to authenticate with. Defaults to ``GITHUB_TOKEN`` from
            the environment when ``use_token`` is set and one is configured.
        use_token: Set False to push anonymously (will fail against a
            private repository, or any repository requiring write auth).
        timeout: Seconds before the push is killed.

    Returns:
        ``{"success": bool, "output": str, "reason": str | None}`` — reports
        rather than raises for anything git says, the same shape as
        :func:`commit_changes`, since "the push failed" is an outcome a
        caller has to describe, not an exception it can do anything about.
        ``output`` has the token redacted, the same as every other string
        this module reports.

    Raises:
        ValueError: If ``branch_name`` is blank — a caller bug, not a state
            of the repository.
    """
    if not branch_name or not branch_name.strip():
        raise ValueError("Branch name is empty; there is nothing to push.")
    branch_name = branch_name.strip()
    repo_full_name = github_client.normalise_repo_name(repo_full_name)

    if token is None and use_token and github_client.token_present():
        token = github_client.get_token()
    if not use_token:
        token = None

    url = _clone_url(repo_full_name, token)
    command = f'git push "{url}" "{branch_name}:{branch_name}"'
    result = run_command(ws, command, timeout=timeout)
    transcript = _redact(_transcribe("git push", result), token)

    if result["timed_out"]:
        return {
            "success": False,
            "output": transcript,
            "reason": f"Pushing {branch_name!r} timed out after {timeout} seconds.",
        }
    if result["exit_code"] != 0:
        detail = _redact((result["stderr"] or result["stdout"]).strip(), token)
        return {
            "success": False,
            "output": transcript,
            "reason": (
                f"git push failed (exit code {result['exit_code']}): "
                f"{detail or 'no output from git'}"
            ),
        }

    return {"success": True, "output": transcript, "reason": None}


def _transcribe(label: str, result: dict) -> str:
    """Render one command's result as a line or two for the returned transcript."""
    body = "\n".join(
        part for part in ((result["stdout"] or "").strip(), (result["stderr"] or "").strip()) if part
    )
    header = f"$ {label} (exit {result['exit_code']})"
    return f"{header}\n{body}" if body else header
