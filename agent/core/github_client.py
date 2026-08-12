"""GitHub REST access — issues in, pull requests out — via PyGithub.

This module is the agent's only door to github.com. It exists to do two things
the rest of the codebase should not have to think about:

* **Configuration.** ``GITHUB_TOKEN`` is read from the environment (``.env`` is
  loaded once, at import time), exactly as :mod:`agent.core.llm_client` does for
  the model provider. Nothing is validated until :func:`get_client` is actually
  called, so importing this module can never fail on a missing token — the test
  suite imports it to unit-test issue formatting without ever reaching the API.

* **Errors worth reading.** PyGithub reports a missing repository, a missing
  issue, and a dead token as the same class (``GithubException``) with different
  status codes buried in the payload. Every call here is translated into one of
  the distinct exceptions below, carrying a message that says what was being
  looked up and what to do about it. A raw PyGithub traceback never escapes.

SCOPING CONSTRAINT (M4)
=======================
:func:`create_pull_request` is built and tested, but it is deliberately not
called from anywhere in this milestone: nothing here is wired into a flow that
pushes to a remote. Opening a PR is an outward-facing, hard-to-undo action, so
it waits for M6's safety/confirmation gate. M4 stops at a local commit.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import (
    BadCredentialsException,
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
)

__all__ = [
    "GitHubAPIError",
    "GitHubAuthError",
    "GitHubError",
    "IssueNotFoundError",
    "RepoNotFoundError",
    "create_pull_request",
    "get_client",
    "get_issue",
    "get_token",
    "list_open_issues",
    "normalise_repo_name",
    "token_present",
]

# Load `.env` once, on import, so every entrypoint (CLI, tests, eval harness)
# sees the same configuration without each having to remember to call this.
load_dotenv()

#: Values that mean "the developer copied .env.example but never filled it in".
#: Treated as a missing token so the error message points at the real problem.
_PLACEHOLDER_TOKENS = frozenset(
    {
        "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "your_github_token_here",
        "changeme",
    }
)

_MISSING_TOKEN_MESSAGE = (
    "GITHUB_TOKEN is not set (or is still the .env.example placeholder).\n"
    "Fix: copy .env.example to .env and set GITHUB_TOKEN to a real token from "
    "https://github.com/settings/tokens — a fine-grained PAT needs read access "
    "to Issues and Contents on the target repository. The .env file lives in "
    "the project root and is gitignored."
)


class GitHubError(RuntimeError):
    """Base class for every failure this module reports.

    Callers that do not care *why* GitHub said no can catch this one class and
    print ``str(exc)``; every message below is written to be read by a human.
    """


class GitHubAuthError(GitHubError):
    """The token is missing, malformed, expired, or lacks the needed scope."""


class RepoNotFoundError(GitHubError):
    """The repository does not exist, or the token cannot see it.

    GitHub deliberately answers "no such repository" (404) rather than
    "forbidden" (403) for a private repo the token cannot read, so this one
    exception genuinely covers both cases and its message says so.
    """


class IssueNotFoundError(GitHubError):
    """The repository exists, but it has no issue with that number."""


class GitHubAPIError(GitHubError):
    """GitHub refused the call for any other reason (rate limit, 5xx, 422)."""


def _env(name: str) -> str:
    """Return the stripped value of environment variable ``name``, or ``""``."""
    return (os.getenv(name) or "").strip()


def token_present() -> bool:
    """True if a usable (non-placeholder) ``GITHUB_TOKEN`` is in the environment.

    Exists so callers — notably the integration tests — can decide to skip
    rather than provoke :class:`GitHubAuthError`.
    """
    token = _env("GITHUB_TOKEN")
    return bool(token) and token.lower() not in _PLACEHOLDER_TOKENS


def get_token() -> str:
    """Return the configured GitHub token.

    Raises:
        GitHubAuthError: If ``GITHUB_TOKEN`` is missing or still the placeholder
            value from ``.env.example``.
    """
    if not token_present():
        raise GitHubAuthError(_MISSING_TOKEN_MESSAGE)
    return _env("GITHUB_TOKEN")


def get_client() -> Github:
    """Build an authenticated PyGithub client.

    The token is validated here, at call time, not at import time. Note that
    PyGithub does not contact GitHub when the client is constructed, so a token
    that is present but *wrong* surfaces as :class:`GitHubAuthError` on the
    first real call, not here.

    Raises:
        GitHubAuthError: If ``GITHUB_TOKEN`` is missing or a placeholder.
    """
    return Github(auth=Auth.Token(get_token()))


@contextmanager
def _github_errors(
    what: str,
    not_found: type[GitHubError] = GitHubAPIError,
    not_found_message: str = "",
) -> Iterator[None]:
    """Translate PyGithub's exceptions into this module's, for one operation.

    Args:
        what: What was being attempted, phrased to follow "while ..." —
            e.g. ``"looking up repository 'octocat/hello'"``.
        not_found: Which exception a 404 means *here*. A 404 from
            ``get_repo`` means the repo is missing; the identical 404 from
            ``get_issue`` means the issue is. Only the caller knows which.
        not_found_message: The 404 message. Falls back to a generic one.

    The ``except`` clauses are ordered most-specific-first on purpose:
    ``BadCredentialsException``, ``UnknownObjectException``, and
    ``RateLimitExceededException`` are all subclasses of ``GithubException``,
    so a bare ``GithubException`` clause placed earlier would swallow them.
    """
    try:
        yield
    except BadCredentialsException as exc:
        raise GitHubAuthError(
            f"GitHub rejected the credentials while {what}: {_message(exc)}\n"
            "Fix: the token in GITHUB_TOKEN is expired, revoked, or malformed — "
            "issue a new one at https://github.com/settings/tokens."
        ) from exc
    except UnknownObjectException as exc:
        raise not_found(
            not_found_message or f"GitHub returned 404 while {what}: {_message(exc)}"
        ) from exc
    except RateLimitExceededException as exc:
        raise GitHubAPIError(
            f"GitHub rate limit exceeded while {what}: {_message(exc)}\n"
            "Fix: wait for the limit to reset, or use a token with a higher quota."
        ) from exc
    except GithubException as exc:
        # 401 without the BadCredentials subclass, 403, 422, and every 5xx.
        # A 403 is usually a permission problem, so it gets the auth exception:
        # what the caller must do about it is change the token, not retry.
        if exc.status in (401, 403):
            raise GitHubAuthError(
                f"GitHub refused the request ({exc.status}) while {what}: "
                f"{_message(exc)}\n"
                "Fix: GITHUB_TOKEN is valid but lacks permission for this "
                "operation — check the token's repository access and scopes."
            ) from exc
        raise GitHubAPIError(
            f"GitHub call failed ({exc.status}) while {what}: {_message(exc)}"
        ) from exc


def _message(exc: GithubException) -> str:
    """Pull the human-readable part out of a PyGithub exception's payload.

    GitHub's error bodies are ``{"message": ..., "errors": [...]}``, but
    PyGithub will also hand back a bare string or ``None`` depending on what
    the server sent, so every shape is flattened to one line here rather than
    letting a ``dict`` repr reach the user.
    """
    data = getattr(exc, "data", None)
    if isinstance(data, dict):
        message = str(data.get("message") or "").strip()
        details = data.get("errors")
        if isinstance(details, list) and details:
            rendered = "; ".join(
                str(item.get("message") or item) if isinstance(item, dict) else str(item)
                for item in details
            )
            message = f"{message} ({rendered})" if message else rendered
        if message:
            return message
    if data:
        return str(data).strip()
    return str(exc)


def _split_repo_name(repo_full_name: str) -> tuple[str, str]:
    """Validate ``owner/name`` locally, before spending a network round trip.

    Raises:
        RepoNotFoundError: If the name is not two non-empty slash-separated
            parts. This is the same thing the caller wants to hear about a
            repository that does not exist, and catching one exception is
            easier than catching two.
    """
    parts = (repo_full_name or "").strip().strip("/").split("/")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise RepoNotFoundError(
            f"{repo_full_name!r} is not a usable repository name — expected "
            '"owner/repo", e.g. "kerrykearns/agent-test-playground".'
        )
    return parts[0].strip(), parts[1].strip()


def normalise_repo_name(repo_full_name: str) -> str:
    """Return ``repo_full_name`` as a validated, whitespace-trimmed ``owner/repo``.

    Public because :mod:`agent.core.repo_manager` builds clone URLs from the
    same string and should reject a malformed name with the same message,
    before a URL is assembled around it.

    Raises:
        RepoNotFoundError: If the name is not ``owner/repo``.
    """
    owner, name = _split_repo_name(repo_full_name)
    return f"{owner}/{name}"


def _issue_dict(issue) -> dict:
    """Project a PyGithub ``Issue`` onto the five fields the agent needs.

    ``url`` is the ``html_url`` — the page a human can open — not the API URL,
    because every consumer of this dict is either printing it for a person or
    embedding it in a PR body. ``body`` is normalised from GitHub's ``null`` to
    ``""``, since an issue with a title and no description is perfectly normal
    and should not make downstream formatting handle ``None``.
    """
    return {
        "number": issue.number,
        "title": issue.title or "",
        "body": issue.body or "",
        "url": issue.html_url or "",
        "state": issue.state or "",
    }


def _get_repo(client: Github, repo_full_name: str):
    """Fetch a repository, mapping a 404 to :class:`RepoNotFoundError`."""
    with _github_errors(
        f"looking up repository {repo_full_name!r}",
        not_found=RepoNotFoundError,
        not_found_message=(
            f"Repository {repo_full_name!r} was not found. Either it does not "
            "exist, or GITHUB_TOKEN cannot see it — GitHub answers 404 rather "
            "than 403 for a private repository the token has no access to, so "
            "check the spelling *and* the token's repository access."
        ),
    ):
        return client.get_repo(repo_full_name)


def get_issue(
    repo_full_name: str, issue_number: int, client: Optional[Github] = None
) -> dict:
    """Fetch one issue.

    Args:
        repo_full_name: ``"owner/repo"``.
        issue_number: The issue's number as it appears in the UI (``#7`` → 7).
        client: Pre-built client, mainly for tests. Built via :func:`get_client`
            when omitted.

    Returns:
        ``{"number", "title", "body", "url", "state"}``. GitHub models pull
        requests as issues, so a number that belongs to a PR returns that PR
        here rather than raising — see :func:`list_open_issues`, which does
        filter them out.

    Raises:
        GitHubAuthError: Token missing, rejected, or insufficient.
        RepoNotFoundError: No such repository, or the token cannot see it.
        IssueNotFoundError: The repository has no issue with that number.
        GitHubAPIError: Any other refusal (rate limit, 5xx).
    """
    repo_full_name = normalise_repo_name(repo_full_name)
    client = client or get_client()
    repo = _get_repo(client, repo_full_name)

    with _github_errors(
        f"fetching issue #{issue_number} of {repo_full_name!r}",
        not_found=IssueNotFoundError,
        not_found_message=(
            f"{repo_full_name} has no issue #{issue_number}. Check the number "
            "in the issue's URL — it is per-repository, not global."
        ),
    ):
        return _issue_dict(repo.get_issue(issue_number))


def list_open_issues(repo_full_name: str, client: Optional[Github] = None) -> list[dict]:
    """List every open issue, newest first (GitHub's default ordering).

    Pull requests are excluded. GitHub's issues endpoint returns PRs alongside
    real issues — a PR *is* an issue with a ``pull_request`` field — and an
    agent asked to "fix issue #12" must never be handed a pull request, so they
    are filtered out here rather than at each call site. This is what M8's task
    selection will read.

    Args:
        repo_full_name: ``"owner/repo"``.
        client: Pre-built client, mainly for tests.

    Returns:
        A list of dicts shaped exactly like :func:`get_issue`'s return value.
        Empty if the repository has no open issues — that is not an error.

    Raises:
        GitHubAuthError: Token missing, rejected, or insufficient.
        RepoNotFoundError: No such repository, or the token cannot see it.
        GitHubAPIError: Any other refusal (rate limit, 5xx).
    """
    repo_full_name = normalise_repo_name(repo_full_name)
    client = client or get_client()
    repo = _get_repo(client, repo_full_name)

    with _github_errors(f"listing open issues of {repo_full_name!r}"):
        # Iterating a PaginatedList is what actually issues the HTTP requests,
        # so the loop belongs inside this block, not after it.
        return [
            _issue_dict(issue)
            for issue in repo.get_issues(state="open")
            if getattr(issue, "pull_request", None) is None
        ]


def create_pull_request(
    repo_full_name: str,
    branch: str,
    base: str,
    title: str,
    body: str,
    client: Optional[Github] = None,
) -> dict:
    """Open a pull request from ``branch`` into ``base``.

    NOT CALLED IN M4. This function is complete and tested, but nothing in this
    milestone invokes it: opening a PR is an outward-facing action that cannot
    be quietly undone, and the branch would have to be pushed first. Both wait
    for M6's safety/confirmation gate. It lives here now so that the GitHub
    surface is finished and covered in one place.

    ``branch`` must already exist on the remote — GitHub has no view of a local
    branch, and a 422 "no commits between" or "invalid head" is what comes back
    if it does not.

    Args:
        repo_full_name: ``"owner/repo"``.
        branch: The head branch, i.e. the one carrying the changes. A plain
            branch name refers to a branch in the same repository; a
            cross-repository PR needs ``"owner:branch"``.
        base: The branch to merge into, e.g. ``"main"``.
        title: PR title.
        body: PR description, in markdown.

    Returns:
        ``{"number", "url", "title", "state", "head", "base"}``, where ``url``
        is the ``html_url`` of the new PR.

    Raises:
        GitHubAuthError: Token missing, rejected, or lacking write access.
        RepoNotFoundError: No such repository, or the token cannot see it.
        GitHubAPIError: GitHub refused to open the PR — most often a 422
            because the head branch was never pushed, the base does not exist,
            or an identical PR is already open.
    """
    repo_full_name = normalise_repo_name(repo_full_name)
    client = client or get_client()
    repo = _get_repo(client, repo_full_name)

    with _github_errors(
        f"opening a pull request from {branch!r} into {base!r} on {repo_full_name!r}",
        not_found=RepoNotFoundError,
        not_found_message=(
            f"GitHub returned 404 opening a pull request on {repo_full_name}. "
            "The repository is visible but the token likely lacks write access "
            "to it, which GitHub reports as 404 rather than 403."
        ),
    ):
        pull = repo.create_pull(title=title, body=body, head=branch, base=base)
        return {
            "number": pull.number,
            "url": pull.html_url or "",
            "title": pull.title or "",
            "state": pull.state or "",
            "head": branch,
            "base": base,
        }
